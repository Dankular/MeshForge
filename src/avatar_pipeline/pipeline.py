"""
MeshForge Remastered — Avatar Pipeline

CODEX strategy:
  TripoSG   = hi-poly source mesh (body geometry)
  Sapiens   = semantic backbone   (normals, depth, seg, pose — all persist)
  Fornos    = bakes FROM TripoSG source ONTO runtime UV atlas
  SkinTokens = jointly generates the skeleton and skinning weights
  Canonical = TODO: replace sphere template with a proper humanoid base mesh

Current implementation uses TripoSG mesh directly with xatlas UV unwrap,
projecting the reference image as texture via front-view camera.
The canonical sphere is suspended until a proper humanoid template exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from avatar_pipeline.baking.ao_bake import AmbientOcclusionBaker
from avatar_pipeline.baking.mv_adapter_texture import MVAdapterTextureBaker
from avatar_pipeline.export.glb import GLBExporter
from avatar_pipeline.fusion.head_fusion import fuse_head_prerig
from avatar_pipeline.generators.pshuman_head import PSHumanHeadGenerator
from avatar_pipeline.generators.triposg_body import TripoSGBodyGenerator, TripoSGConfig
from avatar_pipeline.models.mesh import BakedTextures, Mesh
from avatar_pipeline.preprocess.rembg_processor import RembgProcessor
from avatar_pipeline.rigging.skintokens import (
    DEFAULT_QWEN_CONFIG,
    SKINTOKENS_REPO,
    SkinTokensRigger,
)
from avatar_pipeline.runtime.contracts import (
    TEXTURE_SCHEMA_VERSION,
    has_sparse_albedo,
    validate_runtime_contracts,
    validate_texture_contracts,
)
from avatar_pipeline.runtime.cache import (
    AO_BAKE_VERSION,
    MV_VIEWS_VERSION,
    TEXTURED_MESH_VERSION,
    NORMAL_BAKE_VERSION,
    RIG_VERSION,
    artifact_get,
    artifact_put,
    content_key,
)
from avatar_pipeline.runtime.memory import (
    create_shared_profile,
    normalize_mmgp_module,
    validate_shared_profile,
)
from avatar_pipeline.sapiens.depth import DepthEstimator
from avatar_pipeline.sapiens.human_parsing import HumanParser
from avatar_pipeline.sapiens.normals import SurfaceNormals
from avatar_pipeline.sapiens.pointmap import PointmapEstimator
from avatar_pipeline.sapiens.pose_estimation import PoseEstimator


_REPO_ROOT = Path(__file__).resolve().parents[2]  # src/avatar_pipeline/ → src/ → repo root
_TEXTURE_SCHEMA_KEY = "texture_schema_version"


@dataclass
class PipelineConfig:
    uv_size: int = 2048
    checkpoint_root: str = str(_REPO_ROOT / "checkpoints")
    # Use PSHuman's native full-figure reconstruction (SMPL-X-guided) for
    # body AND head, skipping TripoSG and the head transplant entirely
    # (CODEX migration step 3).
    full_pshuman: bool = False


# ── UV unwrap ─────────────────────────────────────────────────────────────────

def _fuse_head(body: Mesh, head_obj: str, state: dict) -> Mesh:
    """Fuse the PSHuman head onto the body (raw geometry, no UVs).

    The UV unwrap happens inside the reference TexturePipeline. PSHuman's
    per-vertex colors and the head vertex range are recorded so the head
    region of the final atlas can be composited from PSHuman's own views
    (vertices are matched by exact position after the unwrap re-indexes).

    Keeps the pre-fusion body in the snapshot so a future head upgrade can
    re-fuse from clean geometry instead of double-fusing."""
    state["body_prefusion"] = body
    fused, head_colors, head_start = fuse_head_prerig(body, head_obj)
    if head_colors is not None:
        state["head_colors"] = head_colors.astype(np.float32)
        state["head_verts"] = fused.vertices[head_start:].astype(np.float32)
    else:
        state.pop("head_colors", None)
        state.pop("head_verts", None)
    state.pop("head_colors_uv", None)
    state.pop("head_faces", None)
    # Persist the pre-unwrap mesh: it is the only valid texture-stage input
    # (and its content keys the views cache), and it makes mid-bake crash
    # snapshots resumable.
    state["body_fused"] = fused
    return fused


def _ensure_outward_winding(mesh: Mesh) -> tuple[Mesh, bool]:
    """Flip clockwise-wound (inward-normal) faces to outward orientation.

    Snapshots written before the TripoSG winding fix carry inward meshes;
    those black out the MV-Adapter projection (every texel fails the
    aoi_cos test) and render inside-out in backface-culling GLB viewers.
    """
    import trimesh

    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    if tm.volume >= 0:
        return mesh, False
    return (
        Mesh(
            vertices=mesh.vertices,
            faces=mesh.faces[:, ::-1].copy(),
            normals=None if mesh.normals is None else -mesh.normals,
            uvs=mesh.uvs,
            semantic_regions=mesh.semantic_regions,
        ),
        True,
    )


def _snapshot_texture_rebuild_reason(
    state: dict[str, object],
    expected_uv_size: int | None = None,
) -> str | None:
    version = state.get(_TEXTURE_SCHEMA_KEY)
    if version != TEXTURE_SCHEMA_VERSION:
        return (
            f"texture schema {version!r} does not match "
            f"{TEXTURE_SCHEMA_VERSION}"
        )

    textures = state.get("textures")
    if not isinstance(textures, BakedTextures):
        return "textures are missing or use an unsupported type"
    try:
        validate_texture_contracts(textures)
    except ValueError as exc:
        return f"texture contract failed: {exc}"
    if (
        expected_uv_size is not None
        and textures.albedo.shape[:2] != (expected_uv_size, expected_uv_size)
    ):
        return (
            f"texture resolution {textures.albedo.shape[:2]} does not match "
            f"{(expected_uv_size, expected_uv_size)}"
        )
    if has_sparse_albedo(textures):
        return "albedo atlas is black or sparse"
    return None


# ── Pipeline ───────────────────────────────────────────────────────────────────

class AvatarPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config   = config or PipelineConfig()
        self.preprocess = RembgProcessor()
        self.parser     = HumanParser()
        self.pose       = PoseEstimator()
        self.depth      = DepthEstimator()
        self.normals    = SurfaceNormals()
        self.pointmap   = PointmapEstimator()
        self.generator  = TripoSGBodyGenerator(TripoSGConfig())
        self.head_detail = PSHumanHeadGenerator()
        self.ao_baker   = AmbientOcclusionBaker()
        self.tex_baker  = MVAdapterTextureBaker()
        self.rigger     = SkinTokensRigger()
        self.exporter   = GLBExporter()
        self._offload   = None  # mmgp offload domain — set in _setup_offload()
        self._managed_modules: dict[str, torch.nn.Module] = {}
        self._offload_report = None
        self._load_checkpoints(Path(self.config.checkpoint_root))

    def _load_checkpoints(self, root: Path) -> None:
        from avatar_pipeline.checkpoints import resolve_checkpoint_paths
        paths = resolve_checkpoint_paths(root)

        triposg = root / "triposg"
        if triposg.exists():
            self.generator.load_pretrained(triposg)

        # Sapiens estimators normally lazy-load on first use; load them now so
        # every heavy model exists before _setup_offload() builds one shared
        # mmgp memory-management domain spanning the whole pipeline.
        self.parser.load_pretrained(None)
        self.pose.load_pretrained(None)
        self.depth.load_pretrained(None)
        self.normals.load_pretrained(None)

        tokenrig_checkpoint = paths.get(
            "skintokens_tokenrig",
            SKINTOKENS_REPO
            / "experiments"
            / "articulation_xl_quantization_256_token_4"
            / "grpo_1400.ckpt",
        )
        skin_vae_checkpoint = paths.get(
            "skintokens_vae",
            SKINTOKENS_REPO
            / "experiments"
            / "skin_vae_2_10_32768"
            / "last.ckpt",
        )
        qwen_config = paths.get("skintokens_qwen_config")
        qwen_config_dir = (
            qwen_config.parent if qwen_config is not None else DEFAULT_QWEN_CONFIG
        )
        self.rigger.load_pretrained(
            tokenrig_checkpoint,
            skin_vae_checkpoint,
            qwen_config_dir,
        )

        self.tex_baker.load_pretrained()
        self.head_detail.load_pretrained()

        self._setup_offload()

    def _setup_offload(self) -> None:
        """Wrap every heavy model in one shared mmgp memory-management domain.

        Without this, TripoSG's pipeline (loaded at construction) and the
        Sapiens models (one per analysis stage) compete for the RTX 4060's
        8.6GB VRAM simultaneously and thrash badly. mmgp keeps weights pinned
        in host RAM and streams only what's actively running onto the GPU.
        """
        modules = self._collect_heavy_modules()
        print(
            f"[mmgp] wrapping {len(modules)} models in one shared offload domain: "
            f"{sorted(modules)}"
        )
        self._offload, self._offload_report = create_shared_profile(modules)
        self._managed_modules = modules

    def _collect_heavy_modules(self) -> dict[str, torch.nn.Module]:
        modules: dict[str, torch.nn.Module] = {}
        owners: dict[int, str] = {}

        def add(name: str, module: object) -> None:
            if not isinstance(module, torch.nn.Module):
                return
            module = normalize_mmgp_module(module)
            prior = owners.setdefault(id(module), name)
            if prior != name:
                raise RuntimeError(
                    f"Heavy module is exposed by both {prior!r} and {name!r}"
                )
            modules[name] = module

        pipe = self.generator._pipe
        if pipe is not None:
            for name, comp in pipe.components.items():
                add(f"triposg_{name}", comp)
        add("rmbg", self.generator._rmbg)

        for name, est in (
            ("sapiens_seg", self.parser),
            ("sapiens_pose", self.pose),
            ("sapiens_depth", self.depth),
            ("sapiens_normal", self.normals),
        ):
            add(name, est._model)

        add("skintokens", self.rigger._model)

        mv_pipe = self.tex_baker._pipe
        if mv_pipe is not None:
            for name, comp in mv_pipe.components.items():
                add(f"mvadapter_{name}", comp)
            add("mvadapter_cond_encoder", getattr(mv_pipe, "cond_encoder", None))

        if self.head_detail._pipe is not None:
            for name, comp in self.head_detail.heavy_components().items():
                add(name, comp)

        if not modules:
            raise RuntimeError("Pipeline loaded no heavy modules for MMGP")
        return modules

    def validate_memory_profile(self) -> None:
        """Fail if any registered heavy model has escaped shared MMGP control."""
        validate_shared_profile(self._offload, self._managed_modules)

    def _bake_textures(
        self,
        body: Mesh,
        processed: np.ndarray,
        hi_nrm: np.ndarray,
        state: dict | None = None,
        persist=None,
        work_dir: str = "outputs/texture_work",
    ) -> tuple[BakedTextures, Mesh]:
        """Texture the raw fused mesh via the reference TexturePipeline.

        Returns (textures, textured_body): the TexturePipeline performs the
        UV unwrap (open3d UVAtlas) internally, so the mesh geometry that
        carries the atlas comes back from it. Each artifact is keyed by the
        content hash of its real inputs plus a stage version.
        """
        state = state if state is not None else {}

        uv_size = self.config.uv_size
        mesh_key = content_key(body.vertices, body.faces)

        views_key = content_key(
            mesh_key,
            processed,
            "seed=0",
            f"steps={self.tex_baker.num_inference_steps}",
            f"v{MV_VIEWS_VERSION}",
        )
        views = artifact_get(state, "mv_views", views_key)
        if views is None:
            views = self.tex_baker.generate_views(
                body.vertices, body.faces, processed, seed=0
            )
            artifact_put(state, "mv_views", views_key, views)
            if persist is not None:
                # Persist immediately: the diffusion is the expensive step,
                # and a failure in any later bake stage must not lose it.
                persist()
        else:
            print("  [cache] reusing MV-Adapter diffusion views")

        textured_key = content_key(
            views_key, f"uv={uv_size}", f"v{TEXTURED_MESH_VERSION}"
        )
        tm = artifact_get(state, "textured_mesh", textured_key)
        if tm is None:
            verts2, faces2, uvs2, albedo = self.tex_baker.texture_mesh(
                body.vertices, body.faces, views, uv_size, work_dir
            )
            tm = {"verts": verts2, "faces": faces2, "uvs": uvs2, "albedo": albedo}
            artifact_put(state, "textured_mesh", textured_key, tm)
            if persist is not None:
                persist()
        else:
            print("  [cache] reusing textured mesh")
        textured_body = Mesh(
            vertices=tm["verts"],
            faces=tm["faces"],
            uvs=tm["uvs"],
            semantic_regions=body.semantic_regions,
        )
        albedo = tm["albedo"]
        mesh2_key = content_key(
            textured_body.vertices, textured_body.faces, textured_body.uvs
        )

        # Head charts take PSHuman's own multiview colors instead of the
        # full-figure bake, where the face is only a few dozen reference
        # pixels. The unwrap re-indexes vertices but preserves positions
        # exactly (preprocess=False), so the head region is recovered by
        # exact-position matching.
        head_colors = state.get("head_colors")
        head_verts = state.get("head_verts")
        if head_colors is not None and head_verts is not None:
            from avatar_pipeline.runtime.cache import HEAD_COMPOSITE_VERSION
            from scipy.spatial import cKDTree

            composite_key = content_key(
                textured_key, head_colors, head_verts, f"v{HEAD_COMPOSITE_VERSION}"
            )
            composited = artifact_get(state, "albedo_head", composite_key)
            if composited is None:
                tree = cKDTree(head_verts)
                dists, idx = tree.query(textured_body.vertices, k=1)
                is_head = dists < 1e-6
                colors_uv = np.zeros((len(textured_body.vertices), 3), dtype=np.float32)
                colors_uv[is_head] = head_colors[idx[is_head]]
                head_faces = textured_body.faces[
                    np.all(is_head[textured_body.faces], axis=1)
                ]
                if len(head_faces):
                    head_atlas, head_mask = self.tex_baker.bake_vertex_attribute(
                        head_faces,
                        textured_body.uvs,
                        colors_uv,
                        uv_size,
                        background=0.0,
                    )
                    composited = np.where(
                        head_mask[:, :, None],
                        np.clip(head_atlas, 0.0, 1.0),
                        albedo,
                    ).astype(np.float32)
                    print(
                        f"  PSHuman head colors over {head_mask.mean():.1%} "
                        "of the atlas"
                    )
                else:
                    composited = albedo
                artifact_put(state, "albedo_head", composite_key, composited)
            else:
                print("  [cache] reusing head-color composite")
            albedo = composited

        normal_key = content_key(
            mesh2_key, hi_nrm, f"uv={uv_size}", f"v{NORMAL_BAKE_VERSION}"
        )
        normal = artifact_get(state, "normal", normal_key)
        if normal is None:
            normal = self.tex_baker.bake_normal_map(
                textured_body.vertices,
                textured_body.faces,
                textured_body.uvs,
                hi_nrm,
                np.clip(processed[:, :, 3], 0.0, 1.0),
                uv_size,
            )
            artifact_put(state, "normal", normal_key, normal)
        else:
            print("  [cache] reusing normal atlas")

        ao_key = content_key(
            mesh2_key,
            f"uv={uv_size}",
            f"samples={self.ao_baker.num_samples}",
            f"seed={self.ao_baker.seed}",
            f"v{AO_BAKE_VERSION}",
        )
        ambient_occlusion = artifact_get(state, "ao", ao_key)
        if ambient_occlusion is None:
            ao_per_vertex = self.ao_baker.bake(textured_body)
            ambient_occlusion, _ = self.tex_baker.bake_vertex_attribute(
                textured_body.faces,
                textured_body.uvs,
                ao_per_vertex[:, None],
                uv_size,
                background=1.0,
            )
            # Rasterizer barycentrics extrapolate ~1e-3 beyond the vertex
            # range at triangle-edge pixels; the inputs are exact [0, 1].
            ambient_occlusion = np.clip(ambient_occlusion, 0.0, 1.0)
            artifact_put(state, "ao", ao_key, ambient_occlusion)
        else:
            print("  [cache] reusing AO atlas")

        textures = BakedTextures(
            albedo=np.clip(albedo, 0.0, 1.0).astype(np.float32),
            normal=normal,
            ambient_occlusion=ambient_occlusion,
        )
        validate_texture_contracts(textures)
        if has_sparse_albedo(textures):
            raise RuntimeError(
                "Texture bake produced a black or sparse albedo atlas; "
                "snapshot was not updated"
            )
        return textures, textured_body

    def _generate_full_body(
        self,
        processed: np.ndarray,
        state: dict,
        work_dir: str,
    ) -> Mesh:
        """PSHuman full-figure mesh (body + head in one piece), cached by the
        processed image content. The mesh is decimated to body density,
        repaired to UVAtlas cleanliness, and wound outward. PSHuman's own
        vertex colors above the neck line feed the head-color composite,
        exactly like the transplant path (the face is only a few dozen
        reference pixels in the full-figure bake views)."""
        from avatar_pipeline.fusion.head_fusion import (
            _neck_line,
            load_decimated_colored,
            uvatlas_defects,
        )
        from avatar_pipeline.runtime.cache import FULL_BODY_VERSION

        full_key = content_key(
            processed,
            f"seed={self.head_detail.seed}",
            f"steps={self.head_detail.num_inference_steps}",
            f"v{FULL_BODY_VERSION}",
        )
        full_obj = artifact_get(state, "full_body_obj", full_key)
        if full_obj is not None and Path(full_obj).exists():
            print("  [cache] reusing PSHuman full-figure mesh")
        else:
            full_obj = self.head_detail.generate_full(processed, work_dir=work_dir)
            artifact_put(state, "full_body_obj", full_key, full_obj)

        tm, colors = load_decimated_colored(full_obj, target_faces=110_000)
        body = Mesh(
            vertices=np.asarray(tm.vertices, dtype=np.float32),
            faces=np.asarray(tm.faces, dtype=np.int32),
        )
        body, _ = _ensure_outward_winding(body)
        defects = uvatlas_defects(body.vertices, body.faces)
        if any(defects.values()):
            raise RuntimeError(
                f"PSHuman full-figure mesh is not UVAtlas-clean: {defects}"
            )
        print(
            f"  PSHuman full figure: {len(body.vertices):,} verts, "
            f"{len(body.faces):,} faces, colors: {colors is not None}"
        )
        from avatar_pipeline.runtime.validators import print_proportion_report

        print_proportion_report("PSHuman full figure", body.vertices)

        neck_y = _neck_line(np.asarray(body.vertices, dtype=np.float64))
        if colors is not None and neck_y is not None:
            head_sel = body.vertices[:, 1] >= neck_y
            state["head_colors"] = colors[head_sel].astype(np.float32)
            state["head_verts"] = body.vertices[head_sel].astype(np.float32)
            print(
                f"  head-color region above neck y={neck_y:.3f}: "
                f"{int(head_sel.sum()):,} verts"
            )
        else:
            state.pop("head_colors", None)
            state.pop("head_verts", None)

        state["body_fused"] = body
        state["full_pshuman"] = True
        state["head_detail"] = True  # nothing left to transplant
        return body

    def _refuse_head_from_cache(
        self,
        state: dict,
        processed: np.ndarray,
        output_dir: str,
        fallback_body: Mesh | None,
    ) -> Mesh:
        """Rebuild the pre-unwrap fused mesh from the snapshot's cached parts.

        Fuses onto body_prefusion (never onto an already-fused mesh); the
        PSHuman head artifact hits its content cache unless its key is
        genuinely stale. *fallback_body* covers pre-head-detail snapshots
        where the stored body IS the pre-fusion body.
        """
        if state.get("head_detail") and state.get("body_prefusion") is None:
            raise RuntimeError(
                "Snapshot has a fused body but no pre-fusion body to "
                "upgrade from; rerun without --snapshot"
            )
        base = state.get("body_prefusion")
        if base is None:
            base = fallback_body
        if base is None:
            raise RuntimeError(
                "Snapshot has no pre-fusion body to fuse onto; rerun "
                "without --snapshot"
            )
        base, _ = _ensure_outward_winding(base)
        # The cached segmentation may predate the Goliath label fix;
        # recompute it for the head crop.
        seg = self.parser.parse(processed)
        state["seg"] = seg
        head_obj = self._generate_head_detail(
            processed, seg, state,
            work_dir=str(Path(output_dir) / "pshuman"),
        )
        fused = _fuse_head(base, head_obj, state)
        state["head_detail"] = True
        return fused

    def _head_detail_is_stale(self, state: dict) -> bool:
        """True when the head must be (re)generated: never fused, artifact
        key mismatch (inputs or HEAD_DETAIL_VERSION changed), or the cached
        mesh file is gone. Uses the snapshot's stored segmentation for the
        key; a missing/legacy seg counts as stale."""
        if not state.get("head_detail"):
            return True
        from avatar_pipeline.generators.pshuman_head import extract_head_crop_rgba
        from avatar_pipeline.runtime.cache import HEAD_DETAIL_VERSION

        seg = state.get("seg")
        processed = state.get("processed")
        if seg is None or not hasattr(seg, "labels") or processed is None:
            return True
        try:
            crop = extract_head_crop_rgba(processed, seg.labels)
        except RuntimeError:
            return True
        head_key = content_key(
            crop,
            f"seed={self.head_detail.seed}",
            f"steps={self.head_detail.num_inference_steps}",
            f"v{HEAD_DETAIL_VERSION}",
        )
        head_obj = artifact_get(state, "head_obj", head_key)
        return head_obj is None or not Path(head_obj).exists()

    def _generate_head_detail(
        self,
        processed: np.ndarray,
        seg,
        state: dict,
        work_dir: str,
    ) -> str:
        """PSHuman head mesh, cached by the head-crop content."""
        from avatar_pipeline.generators.pshuman_head import extract_head_crop_rgba
        from avatar_pipeline.runtime.cache import HEAD_DETAIL_VERSION

        crop = extract_head_crop_rgba(processed, seg.labels)
        head_key = content_key(
            crop,
            f"seed={self.head_detail.seed}",
            f"steps={self.head_detail.num_inference_steps}",
            f"v{HEAD_DETAIL_VERSION}",
        )
        head_obj = artifact_get(state, "head_obj", head_key)
        if head_obj is not None and Path(head_obj).exists():
            print("  [cache] reusing PSHuman head mesh")
            return head_obj
        head_obj = self.head_detail.generate(processed, seg.labels, work_dir=work_dir)
        artifact_put(state, "head_obj", head_key, head_obj)
        return head_obj

    @staticmethod
    def _write_snapshot(path: Path, state: dict[str, object]) -> None:
        import pickle

        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.tmp")
        try:
            with temp_path.open("wb") as file:
                pickle.dump(state, file)
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)

    def run(self, input_image: str, output_dir: str, snapshot: str | None = None) -> Path:
        self.validate_memory_profile()
        # Snapshot = pickled (processed, seg, pose, depth, hi_nrm, pointmap, body,
        # textures) from steps 1-5. TripoSG generation + the MV-Adapter diffusion
        # bake are by far the slowest stages (~25 min combined) and are
        # input-image-deterministic given a fixed seed — caching their output lets
        # later-stage work (PSHuman, SkinTokens) iterate without rerunning them.
        snap_path = Path(snapshot) if snapshot else None
        if snap_path is not None and snap_path.exists():
            import pickle
            print(f"[snapshot] loading cached pre-rig state from {snap_path}")
            with open(snap_path, "rb") as f:
                state = pickle.load(f)
            processed, seg, pose, depth, hi_nrm, pointmap = (
                state["processed"], state["seg"], state["pose"], state["depth"],
                state["hi_nrm"], state["pointmap"],
            )
            # Two meshes live in the snapshot: "body_fused" is the pre-unwrap
            # fused mesh (the ONLY valid input for the texture stage — its
            # content also keys the views cache), and "body" is the
            # UVAtlas-unwrapped textured mesh used for rig/export. Feeding
            # "body" back through the texture stage double-unwraps a
            # seam-split mesh, which UVAtlas rejects as non-manifold.
            body = state.get("body")
            if body is not None:
                body, flipped = _ensure_outward_winding(body)
                if flipped:
                    print("[snapshot] flipped cached mesh winding to outward orientation")
                    state["body"] = body
            body_fused = state.get("body_fused")

            rebuild_reason = None
            if state.get("full_pshuman"):
                # No transplant in this mode; if the pre-unwrap mesh is gone
                # rebuild it from the cached full-figure artifact.
                if body is None and body_fused is None:
                    body_fused = self._generate_full_body(
                        processed, state,
                        work_dir=str(Path(output_dir) / "pshuman_full"),
                    )
                    rebuild_reason = "full-figure mesh rebuilt"
            elif self._head_detail_is_stale(state) or (
                body is None and body_fused is None
            ):
                # The head artifact is missing or its content key is stale
                # (new inputs or a HEAD_DETAIL_VERSION bump). Re-fuse from
                # the pre-fusion body — never onto an already-fused mesh.
                print("[snapshot] (re)generating PSHuman head detail")
                body_fused = self._refuse_head_from_cache(
                    state, processed, output_dir, fallback_body=body
                )
                rebuild_reason = "head detail added to mesh"

            if rebuild_reason is None:
                rebuild_reason = _snapshot_texture_rebuild_reason(
                    state,
                    expected_uv_size=self.config.uv_size,
                )
            if rebuild_reason is None:
                textures = state["textures"]
            else:
                print(f"[snapshot] rebaking textures: {rebuild_reason}")
                if body_fused is None:
                    if state.get("full_pshuman"):
                        body_fused = self._generate_full_body(
                            processed, state,
                            work_dir=str(Path(output_dir) / "pshuman_full"),
                        )
                    elif state.get("body_prefusion") is not None:
                        # Snapshot predates body_fused: rebuild the pre-unwrap
                        # mesh from cached parts (PSHuman head hits its cache).
                        print("[snapshot] rebuilding pre-unwrap fused mesh")
                        body_fused = self._refuse_head_from_cache(
                            state, processed, output_dir, fallback_body=None
                        )
                    else:
                        # Legacy snapshot with only the stored body: bake from
                        # it directly (pre-TexturePipeline snapshots stored
                        # the pre-unwrap mesh here).
                        body_fused = body
                textures, body = self._bake_textures(
                    body_fused, processed, hi_nrm, state,
                    persist=lambda: self._write_snapshot(snap_path, state),
                    work_dir=str(Path(output_dir) / "texture"),
                )
                state["body"] = body
                state["textures"] = textures
                state[_TEXTURE_SCHEMA_KEY] = TEXTURE_SCHEMA_VERSION
                self._write_snapshot(snap_path, state)
                print(f"[snapshot] updated texture cache in {snap_path}")
        else:
            # 1. Preprocess
            print("[1/8] Preprocessing ...")
            image     = np.array(Image.open(input_image).convert("RGB"), dtype=np.uint8)
            processed = self.preprocess.process_rmbg2(image)
            alpha     = np.clip(processed[:, :, 3:4], 0.0, 1.0)
            fg_rgb    = np.clip(processed[:, :, :3] * alpha + (1.0 - alpha) * 0.5, 0.0, 1.0)
            cond_img  = np.clip(fg_rgb * 255.0, 0, 255).astype(np.uint8)

            # 2. Sapiens semantic backbone
            print("[2/8] Sapiens analysis ...")
            seg     = self.parser.parse(processed)
            pose    = self.pose.estimate(processed)
            depth   = self.depth.estimate(processed)
            hi_nrm  = self.normals.estimate(processed)
            pointmap = self.pointmap.estimate(depth, seg)

            state = {
                "processed": processed, "seg": seg, "pose": pose, "depth": depth,
                "hi_nrm": hi_nrm, "pointmap": pointmap,
                _TEXTURE_SCHEMA_KEY: TEXTURE_SCHEMA_VERSION,
            }

            if self.config.full_pshuman:
                # 3-5. PSHuman full-figure reconstruction: body + head in one
                # SMPL-X-guided piece — no TripoSG, no transplant.
                print("[3-5/8] PSHuman full-figure reconstruction ...")
                body = self._generate_full_body(
                    processed, state,
                    work_dir=str(Path(output_dir) / "pshuman_full"),
                )
            else:
                # 3. TripoSG body mesh
                print("[3/8] TripoSG ...")
                body = self.generator.generate(
                    image=cond_img, semantic_map=seg, pose_data=pose,
                    depth=depth, normals=hi_nrm,
                )
                # body = Mesh(vertices, faces) — no UVs yet
                from avatar_pipeline.runtime.validators import (
                    print_proportion_report,
                )

                print_proportion_report("TripoSG body", body.vertices)

                # 4. PSHuman head detail + pre-rig fusion
                print("[4/8] PSHuman head detail ...")
                head_obj = self._generate_head_detail(
                    processed, seg, state, work_dir=str(Path(output_dir) / "pshuman")
                )

                # 5. Pre-rig head fusion (raw geometry; the UV unwrap happens
                # inside the reference TexturePipeline in the next stage)
                print("[5/8] Head fusion ...")
                body = _fuse_head(body, head_obj, state)  # also sets body_fused
                state["head_detail"] = True

            # 6. Texture: MV-Adapter views + reference TexturePipeline
            # (UVAtlas unwrap, view upscale, view inpaint) — Space-exact.
            print("[6/8] Texturing ...")
            textures, body = self._bake_textures(
                body, processed, hi_nrm, state,
                persist=(
                    (lambda: self._write_snapshot(snap_path, state))
                    if snap_path is not None
                    else None
                ),
                work_dir=str(Path(output_dir) / "texture"),
            )
            state["body"] = body
            state["textures"] = textures

            if snap_path is not None:
                self._write_snapshot(snap_path, state)
                print(f"[snapshot] saved pre-rig state to {snap_path}")

        # 7. Rig (cached against the mesh geometry)
        print("[7/8] Rigging ...")
        rig_key = content_key(body.vertices, body.faces, f"v{RIG_VERSION}")
        rigged = artifact_get(state, "rig", rig_key)
        if rigged is None:
            rigged = self.rigger.rig(body)
            artifact_put(state, "rig", rig_key, rigged)
            if snap_path is not None:
                self._write_snapshot(snap_path, state)
        else:
            print("  [cache] reusing SkinTokens rig")
        from avatar_pipeline.runtime.validators import print_rig_report

        print_rig_report(rigged)

        # 8. Export
        print("[8/8] Exporting GLB ...")
        validate_runtime_contracts(rigged, textures)
        out = str(Path(output_dir) / "avatar.glb")
        return self.exporter.export(rigged, textures, out)
