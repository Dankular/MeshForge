"""TripoSG body generator — uses VAST-AI/TripoSG weights directly."""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch

from avatar_pipeline.models.mesh import Mesh
from avatar_pipeline.models.semantic import PoseData, SemanticMap

TRIPOSG_REPO = Path(r"D:\Dev Proj\avatar\MeshForge\external\TripoSG")
CHECKPOINT   = Path(__file__).resolve().parents[3] / "checkpoints" / "triposg"

if str(TRIPOSG_REPO) not in sys.path:
    sys.path.insert(0, str(TRIPOSG_REPO))
if str(TRIPOSG_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(TRIPOSG_REPO / "scripts"))


@dataclass
class TripoSGConfig:
    num_inference_steps: int = 50
    guidance_scale: float = 7.0
    num_tokens: int = 2048
    octree_depth: int = 9
    use_flash_decoder: bool = True
    target_faces: int = 80_000   # decimate before UV unwrap and SkinTokens


class TripoSGBodyGenerator:
    """Image → 3D mesh via VAST-AI TripoSG flow-matching pipeline."""

    def __init__(self, config: TripoSGConfig | None = None) -> None:
        self.config = config or TripoSGConfig()
        self._pipe = None
        self._rmbg = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_pretrained(self, checkpoint_path=None) -> None:
        from triposg.pipelines.pipeline_triposg import TripoSGPipeline
        from briarmbg import BriaRMBG

        ckpt = str(checkpoint_path or CHECKPOINT)
        # Loaded onto CPU deliberately — mmgp.offload manages GPU residency
        # (see AvatarPipeline._setup_offload, which wraps every heavy model
        # in one shared memory-management domain).
        self._pipe = TripoSGPipeline.from_pretrained(ckpt)
        rmbg_dir = TRIPOSG_REPO / "pretrained_weights" / "RMBG-1.4"
        self._rmbg = BriaRMBG.from_pretrained(str(rmbg_dir)) if rmbg_dir.exists() else None

    def _ensure_loaded(self) -> None:
        if self._pipe is None:
            self.load_pretrained()

    def generate(
        self,
        image: np.ndarray,
        semantic_map: SemanticMap,
        pose_data: PoseData,
        depth: np.ndarray,
        normals: np.ndarray,
    ) -> Mesh:
        self._ensure_loaded()
        from image_process import prepare_image
        from PIL import Image as PILImage

        img_pil = PILImage.fromarray(image[:, :, :3])

        # Use rembg if available, else pass raw PIL
        if self._rmbg is not None:
            img_prepared = prepare_image(img_pil, bg_color=np.array([1., 1., 1.]), rmbg_net=self._rmbg)
        else:
            # Apply sapiens mask as alpha and pass RGBA
            if semantic_map is not None:
                alpha = (semantic_map.labels > 0).astype(np.uint8) * 255
                rgba = np.dstack([image[:, :, :3], alpha])
                img_prepared = PILImage.fromarray(rgba.astype(np.uint8), "RGBA")
            else:
                img_prepared = img_pil

        with torch.no_grad():
            result = self._pipe(
                image=img_prepared,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                num_tokens=self.config.num_tokens,
                use_flash_decoder=self.config.use_flash_decoder,
                flash_octree_depth=self.config.octree_depth,
            )

        mesh_ts = result.meshes[0]

        # Decimate using pymeshlab (same method as TripoSG's own inference script)
        import pymeshlab
        target_faces = self.config.target_faces
        raw_v = np.array(mesh_ts.vertices, dtype=np.float64)
        raw_f = np.array(mesh_ts.faces,    dtype=np.int32)
        print(f"  TripoSG raw mesh: {len(raw_v):,} verts  {len(raw_f):,} faces")
        if len(raw_f) > target_faces:
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(vertex_matrix=raw_v, face_matrix=raw_f))
            ms.meshing_merge_close_vertices()          # same as TripoSG inference
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
            m = ms.current_mesh()
            verts = m.vertex_matrix().astype(np.float32)
            faces = m.face_matrix().astype(np.int32)
        else:
            verts = raw_v.astype(np.float32)
            faces = raw_f

        # TripoSG's flash decoder can emit clockwise-wound faces (inward
        # normals). Downstream consumers need outward orientation: MV-Adapter's
        # camera projection rejects texels whose normals face away from every
        # camera (aoi_cos threshold), and GLB viewers backface-cull. A negative
        # signed volume on a winding-consistent mesh means a single global flip.
        import trimesh
        if trimesh.Trimesh(vertices=verts, faces=faces, process=False).volume < 0:
            faces = faces[:, ::-1].copy()
            print("  TripoSG mesh: flipped face winding to outward orientation")

        print(f"  TripoSG mesh: {len(verts):,} verts  {len(faces):,} faces")
        return Mesh(
            vertices=verts,
            faces=faces,
            semantic_regions=["skin", "hair", "cloth", "eyes", "mouth"],
        )
