"""
MeshForge Remastered — Avatar Pipeline

CODEX strategy:
  TripoSG   = hi-poly source mesh (body geometry)
  Sapiens   = semantic backbone   (normals, depth, seg, pose — all persist)
  Fornos    = bakes FROM TripoSG source ONTO runtime UV atlas
  UniRig    = rigs the final mesh
  Canonical = TODO: replace sphere template with a proper humanoid base mesh

Current implementation uses TripoSG mesh directly with xatlas UV unwrap,
projecting the reference image as texture via front-view camera.
The canonical sphere is suspended until a proper humanoid template exists.
"""
from __future__ import annotations

import io
import sys
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from avatar_pipeline.baking.ao_bake import AmbientOcclusionBaker
from avatar_pipeline.baking.fornos_backend import BakeRequest, FornosBackend
from avatar_pipeline.export.glb import GLBExporter
from avatar_pipeline.generators.triposg_body import TripoSGBodyGenerator, TripoSGConfig
from avatar_pipeline.models.mesh import BakedTextures, Mesh
from avatar_pipeline.preprocess.rembg_processor import RembgProcessor
from avatar_pipeline.rigging.unirig import UniRig
from avatar_pipeline.runtime.contracts import validate_runtime_contracts
from avatar_pipeline.sapiens.depth import DepthEstimator
from avatar_pipeline.sapiens.human_parsing import HumanParser
from avatar_pipeline.sapiens.normals import SurfaceNormals
from avatar_pipeline.sapiens.pointmap import PointmapEstimator
from avatar_pipeline.sapiens.pose_estimation import PoseEstimator


_REPO_ROOT = Path(__file__).resolve().parents[2]  # src/avatar_pipeline/ → src/ → repo root


@dataclass
class PipelineConfig:
    uv_size: int = 1024
    checkpoint_root: str = str(_REPO_ROOT / "checkpoints")


# ── UV unwrap ─────────────────────────────────────────────────────────────────

def _xatlas_unwrap(vertices: np.ndarray, faces: np.ndarray):
    """xatlas UV parameterisation. Returns (new_verts, new_faces, uvs)."""
    import xatlas
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices.astype(np.float32), faces.astype(np.uint32))
    atlas.generate(
        chart_options=xatlas.ChartOptions(),
        pack_options=xatlas.PackOptions(),
    )
    vmapping, new_faces, uvs = atlas[0]
    new_verts = vertices[vmapping].astype(np.float32)
    print(f"  xatlas: {len(new_verts):,} verts after unwrap")
    return new_verts, new_faces.astype(np.int32), uvs.astype(np.float32)


# ── Texture bake from reference image ─────────────────────────────────────────

def _crop_to_person(image_rgb: np.ndarray, seg_map) -> np.ndarray:
    """Tight crop of person + padding, returned as square."""
    mask = seg_map.labels > 0
    ys, xs = np.where(mask)
    if ys.size == 0:
        return image_rgb
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    pad = max(y1 - y0, x1 - x0) // 8
    y0 = max(0, y0 - pad); y1 = min(image_rgb.shape[0] - 1, y1 + pad)
    x0 = max(0, x0 - pad); x1 = min(image_rgb.shape[1] - 1, x1 + pad)
    crop = image_rgb[y0:y1 + 1, x0:x1 + 1]
    h, w = crop.shape[:2]
    sz = max(h, w)
    sq = np.full((sz, sz, 3), 128, dtype=np.uint8)
    dy, dx = (sz - h) // 2, (sz - w) // 2
    sq[dy:dy + h, dx:dx + w] = crop
    return sq


def _bake_texture_to_uv(
    person_img: np.ndarray,  # uint8 (H, W, 3) — person centred, square
    verts: np.ndarray,       # (N, 3) xatlas-unwrapped mesh vertices
    faces: np.ndarray,       # (F, 3)
    uvs: np.ndarray,         # (N, 2) in [0,1]
    uv_size: int,
) -> np.ndarray:
    """
    Bake reference image onto mesh UV atlas.

    For each UV texel: find its 3D world position on the mesh via
    barycentric interpolation, then project orthographically to the
    reference image (front view = +Z camera).
    """
    # Normalise mesh to [-1,1] for projection
    v_min = verts.min(axis=0)
    v_max = verts.max(axis=0)
    v_range = v_max - v_min + 1e-8
    v_norm = (verts - v_min) / v_range * 2.0 - 1.0   # (N,3) in [-1,1]

    src    = cv2.resize(person_img, (uv_size, uv_size)).astype(np.float32) / 255.0
    H_s = W_s = uv_size
    out    = np.full((uv_size, uv_size, 3), 0.55, dtype=np.float32)
    cover  = np.zeros((uv_size, uv_size), dtype=np.uint8)
    uv_px  = uvs * (uv_size - 1)

    for fi in range(len(faces)):
        i0, i1, i2 = faces[fi]
        us = uv_px[[i0, i1, i2], 0]
        vs = uv_px[[i0, i1, i2], 1]
        u_lo = max(0,            int(np.floor(us.min())))
        u_hi = min(uv_size - 1,  int(np.ceil(us.max())))
        v_lo = max(0,            int(np.floor(vs.min())))
        v_hi = min(uv_size - 1,  int(np.ceil(vs.max())))
        if u_hi < u_lo or v_hi < v_lo:
            continue

        pu = np.arange(u_lo, u_hi + 1, dtype=np.float32)
        pv = np.arange(v_lo, v_hi + 1, dtype=np.float32)
        pu_g, pv_g = np.meshgrid(pu, pv)
        pu_f = pu_g.ravel(); pv_f = pv_g.ravel()

        d   = uv_px[i1] - uv_px[i0]
        e   = uv_px[i2] - uv_px[i0]
        px  = pu_f - uv_px[i0, 0]
        py  = pv_f - uv_px[i0, 1]
        den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < 1e-8:
            continue
        la = (px * e[1] - py * e[0]) / den
        lb = (d[0] * py - d[1] * px) / den
        lc = 1.0 - la - lb
        ins = (la >= 0) & (lb >= 0) & (lc >= 0)
        if not ins.any():
            continue

        # World position of inside texels
        p3 = (la[ins, None] * v_norm[i0]
              + lb[ins, None] * v_norm[i1]
              + lc[ins, None] * v_norm[i2])

        # Orthographic front projection: X right, Y up → image
        sx = np.clip(((p3[:, 0] + 1.0) * 0.5 * (W_s - 1)), 0, W_s - 1).astype(np.int32)
        sy = np.clip(((1.0 - (p3[:, 1] + 1.0) * 0.5) * (H_s - 1)), 0, H_s - 1).astype(np.int32)

        pu_i = pu_f[ins].astype(np.int32)
        pv_i = pv_f[ins].astype(np.int32)
        out[pv_i, pu_i] = src[sy, sx]
        cover[pv_i, pu_i] = 1

    # Inpaint holes
    hole = ((1 - cover) * 255).astype(np.uint8)
    out_u8 = np.clip(out * 255, 0, 255).astype(np.uint8)
    return cv2.inpaint(out_u8, hole, 3, cv2.INPAINT_TELEA).astype(np.float32) / 255.0


def _ao_to_uv(ao_pv: np.ndarray, uvs: np.ndarray, uv_size: int) -> np.ndarray:
    out = np.full((uv_size, uv_size, 1), 0.8, dtype=np.float32)
    px = np.clip((uvs * (uv_size - 1)).astype(np.int32), 0, uv_size - 1)
    out[px[:, 1], px[:, 0], 0] = ao_pv
    return out


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
        self.ao_baker   = AmbientOcclusionBaker()
        self.bake_back  = FornosBackend()
        self.rigger     = UniRig()
        self.exporter   = GLBExporter()
        self._load_checkpoints(Path(self.config.checkpoint_root))

    def _load_checkpoints(self, root: Path) -> None:
        from avatar_pipeline.checkpoints import resolve_checkpoint_paths
        paths = resolve_checkpoint_paths(root)

        triposg = root / "triposg"
        if triposg.exists():
            self.generator.load_pretrained(triposg)

        if "unirig_skeleton" in paths and "unirig_skin" in paths:
            self.rigger.load_pretrained(paths["unirig_skeleton"], paths["unirig_skin"])

        mv = Path(r"D:\Dev Proj\avatar\MeshForge\external\MV-Adapter\checkpoints")
        if mv.exists():
            self.bake_back.load_mv_adapter_checkpoints(mv)

    def run(self, input_image: str, output_dir: str) -> Path:
        # 1. Preprocess
        print("[1/7] Preprocessing ...")
        image     = np.array(Image.open(input_image).convert("RGB"), dtype=np.uint8)
        processed = self.preprocess.process_rmbg2(image)
        alpha     = np.clip(processed[:, :, 3:4], 0.0, 1.0)
        fg_rgb    = np.clip(processed[:, :, :3] * alpha + (1.0 - alpha) * 0.5, 0.0, 1.0)
        cond_img  = np.clip(fg_rgb * 255.0, 0, 255).astype(np.uint8)

        # 2. Sapiens semantic backbone
        print("[2/7] Sapiens analysis ...")
        seg     = self.parser.parse(processed)
        pose    = self.pose.estimate(processed)
        depth   = self.depth.estimate(processed)
        hi_nrm  = self.normals.estimate(processed)
        pointmap = self.pointmap.estimate(depth, seg)

        # 3. TripoSG body mesh
        print("[3/7] TripoSG ...")
        body = self.generator.generate(
            image=cond_img, semantic_map=seg, pose_data=pose,
            depth=depth, normals=hi_nrm,
        )
        # body = Mesh(vertices, faces) — no UVs yet

        # 4. UV unwrap with xatlas
        print("[4/7] UV unwrap ...")
        verts, faces, uvs = _xatlas_unwrap(body.vertices, body.faces)
        body = Mesh(vertices=verts, faces=faces, uvs=uvs,
                    semantic_regions=body.semantic_regions)

        # 5. Bake textures
        print("[5/7] Baking textures ...")
        uv_sz      = self.config.uv_size
        person_img = _crop_to_person(cond_img, seg)
        albedo     = _bake_texture_to_uv(person_img, verts, faces, uvs, uv_sz)

        # Normal map: Sapiens hi-res normals resized to UV atlas
        nrm_r = cv2.resize(
            (hi_nrm * 0.5 + 0.5).clip(0, 1).astype(np.float32),
            (uv_sz, uv_sz), interpolation=cv2.INTER_LINEAR,
        )
        nrm_s = nrm_r * 2.0 - 1.0
        nrm_tex = (nrm_s / (np.linalg.norm(nrm_s, axis=2, keepdims=True) + 1e-8)).astype(np.float32)

        ao_pv  = self.ao_baker.bake(body)
        ao_tex = _ao_to_uv(ao_pv, uvs, uv_sz)

        lo_nrm = np.zeros((uv_sz, uv_sz, 3), dtype=np.float32); lo_nrm[:, :, 2] = 1.0
        textures = self.bake_back.bake(BakeRequest(
            albedo=albedo,
            high_normals=nrm_tex,
            low_normals=lo_nrm,
            ao_per_vertex=ao_pv,
        ))

        # 6. Rig
        print("[6/7] Rigging ...")
        rigged = self.rigger.rig(body)

        # 7. Export
        print("[7/7] Exporting GLB ...")
        validate_runtime_contracts(rigged, textures)
        out = str(Path(output_dir) / "avatar.glb")
        return self.exporter.export(rigged, textures, out)
