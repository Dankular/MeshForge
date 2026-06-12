"""MV-Adapter texture baking, Space-exact.

Two real stages, mirroring huggingface.co/spaces/VAST-AI/TripoSG:
  1. generate_views — MVAdapterI2MVSDXLPipeline (SDXL UNet + MV-Adapter)
     conditioned on the reference photo and rendered position/normal control
     maps; 6 orthogonal views, 15 steps (the Space's exact call).
  2. texture_mesh — mvadapter's own TexturePipeline (vendored, zero drift
     from upstream): open3d-UVAtlas unwrap, RealESRGAN x2 view upscale,
     camera projection, SmartPainter view-space inpaint. Its cvcuda calls
     run on the OpenCV cv_ops backend (cv_ops_cpu) on Windows.

Verified side by side: this path produces clean, seam-free texture from the
same mesh and views that a reimplemented projection stack mangled. Do not
reimplement TexturePipeline's internals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MV_ADAPTER_REPO = Path(r"D:\Dev Proj\avatar\MeshForge\external\MV-Adapter")
if str(MV_ADAPTER_REPO) not in sys.path:
    sys.path.insert(0, str(MV_ADAPTER_REPO))
if str(MV_ADAPTER_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(MV_ADAPTER_REPO / "scripts"))

# OpenCV-backed cv_ops must be registered before mvadapter imports the real
# module (whose `import cvcuda` has no Windows wheels).
from avatar_pipeline.baking import cv_ops_cpu

cv_ops_cpu.register()

_NUM_VIEWS = 6
_ELEVATION_DEG = [0, 0, 0, 0, 89.99, -89.99]
_AZIMUTH_DEG = [x - 90 for x in [0, 90, 180, 270, 180, 180]]
_NEGATIVE_PROMPT = "watermark, ugly, deformed, noisy, blurry, low contrast"

# Frame conversion into the mvadapter camera-rig frame (z-up, front at -y,
# i.e. load_mesh's standard frame with front_x_to_y=False). TripoSG bodies
# come out +y-up with the figure facing +z (verified empirically on two
# generations via toe direction), so the plain y-up -> z-up rotation
# v_rig = (x, -z, y) puts the figure upright and facing the front camera.
_MESH_TO_RIG = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32
)


class MVAdapterTextureBaker:
    """MV-Adapter view generation + reference TexturePipeline texturing."""

    def __init__(self, height: int = 768, width: int = 768) -> None:
        self.height = height
        self.width = width
        self.num_inference_steps = 15  # the Space's exact setting
        self._pipe = None
        self._ctx = None
        self._texture_pipe = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_pretrained(self) -> None:
        from inference_ig2mv_sdxl import prepare_pipeline

        # Built on CPU deliberately — mmgp.offload manages GPU residency for
        # every heavy model in the pipeline from one shared memory-management
        # domain (see AvatarPipeline._setup_offload).
        self._pipe = prepare_pipeline(
            base_model="stabilityai/stable-diffusion-xl-base-1.0",
            vae_model="madebyollin/sdxl-vae-fp16-fix",
            unet_model=None,
            lora_model=None,
            adapter_path="huanngzh/mv-adapter",
            scheduler=None,
            num_views=_NUM_VIEWS,
            device="cpu",
            dtype=torch.float16,
        )

    def _ensure_loaded(self) -> None:
        if self._pipe is None:
            self.load_pretrained()

    def _ensure_ctx(self) -> None:
        """Rasterizer context only — enough for the geometry bakes
        (normal map, vertex attributes) without loading the SDXL pipeline."""
        if self._ctx is None:
            from mvadapter.utils.mesh_utils import NVDiffRastContextWrapper

            self._ctx = NVDiffRastContextWrapper(
                device=str(self._device), context_type="cuda"
            )

    def _ensure_texture_pipe(self):
        """The reference TexturePipeline (upscaler + LaMa + SmartPainter)."""
        if self._texture_pipe is None:
            from mvadapter.pipelines.pipeline_texture import TexturePipeline

            ckpt = MV_ADAPTER_REPO / "checkpoints"
            self._texture_pipe = TexturePipeline(
                upscaler_ckpt_path=str(ckpt / "RealESRGAN_x2plus.pth"),
                inpaint_ckpt_path=str(ckpt / "big-lama.pt"),
                device=str(self._device),
            )
        return self._texture_pipe

    def _build_mesh_and_cameras(self, verts, faces, uvs, uv_size):
        from mvadapter.utils.mesh_utils import TexturedMesh, get_orthogonal_camera

        dev = self._device
        max_scale = float(np.max(np.abs(verts)))
        if not np.isfinite(max_scale) or max_scale <= 1e-8:
            raise ValueError("Cannot normalize an empty or degenerate mesh for MV-Adapter")
        render_verts = (verts.astype(np.float32) @ _MESH_TO_RIG.T) / max_scale * 0.5
        v_pos = torch.from_numpy(render_verts).to(dev)
        t_idx = torch.from_numpy(faces.astype(np.int64)).to(dev)
        if uvs is None:
            uvs = np.zeros((len(verts), 2), dtype=np.float32)
        v_tex = torch.from_numpy(uvs.astype(np.float32)).to(dev)
        blank_tex = torch.zeros((uv_size, uv_size, 3), dtype=torch.float32, device=dev)
        mesh = TexturedMesh(
            v_pos=v_pos, t_pos_idx=t_idx, v_tex=v_tex, t_tex_idx=t_idx, texture=blank_tex
        )
        cameras = get_orthogonal_camera(
            elevation_deg=_ELEVATION_DEG,
            distance=[1.8] * _NUM_VIEWS,
            left=-0.55, right=0.55, bottom=-0.55, top=0.55,
            azimuth_deg=_AZIMUTH_DEG,
            device=str(dev),
        )
        return mesh, cameras

    def generate_views(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        reference_rgba: np.ndarray,
        seed: int = 0,
    ) -> np.ndarray:
        """Run the MV-Adapter diffusion. Returns uint8 (V, H, W, 3) views.

        Deterministic for (mesh, reference, seed) — cacheable independently
        of the texturing that follows. UVs are not needed for the control
        renders.
        """
        self._ensure_loaded()
        self._ensure_ctx()
        from inference_ig2mv_sdxl import preprocess_image
        from mvadapter.utils.mesh_utils import render

        dev = self._device
        mesh, cameras = self._build_mesh_and_cameras(verts, faces, None, uv_size=64)
        render_out = render(
            self._ctx, mesh, cameras,
            height=self.height, width=self.width,
            render_attr=False, normal_background=0.0,
        )
        control_images = torch.cat(
            [(render_out.pos + 0.5).clamp(0, 1), (render_out.normal / 2 + 0.5).clamp(0, 1)],
            dim=-1,
        ).permute(0, 3, 1, 2).to(dev)

        rgba_u8 = np.clip(reference_rgba * 255.0, 0, 255).astype(np.uint8)
        ref_image = preprocess_image(Image.fromarray(rgba_u8, "RGBA"), self.height, self.width)

        gen = torch.Generator(device=str(dev)).manual_seed(seed)
        with torch.no_grad():
            mv_images = self._pipe(
                "high quality",
                height=self.height,
                width=self.width,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=3.0,
                num_images_per_prompt=_NUM_VIEWS,
                control_image=control_images,
                control_conditioning_scale=1.0,
                reference_image=ref_image,
                reference_conditioning_scale=1.0,
                negative_prompt=_NEGATIVE_PROMPT,
                generator=gen,
            ).images
        return np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in mv_images])

    def texture_mesh(
        self,
        verts: np.ndarray,           # (N, 3) raw fused mesh (no UVs needed)
        faces: np.ndarray,           # (F, 3)
        views: np.ndarray,           # uint8 (V, H, W, 3) from generate_views
        uv_size: int,
        work_dir: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Unwrap + texture via the reference TexturePipeline (Space-exact).

        Returns (verts, faces, uvs, albedo): the UVAtlas-unwrapped mesh
        (original vertex positions preserved, indices re-organized) and the
        float32 (uv_size, uv_size, 3) albedo atlas in [0, 1].
        """
        import trimesh
        from mvadapter.pipelines.pipeline_texture import ModProcessConfig
        from mvadapter.utils import make_image_grid

        work = Path(work_dir).resolve()
        work.mkdir(parents=True, exist_ok=True)
        mesh_path = work / "mesh_raw.glb"
        trimesh.Trimesh(
            vertices=verts.astype(np.float32),
            faces=faces.astype(np.int64),
            process=False,
        ).export(mesh_path)
        grid_path = work / "views_grid.png"
        make_image_grid([Image.fromarray(v) for v in views], rows=1).save(grid_path)

        pipe = self._ensure_texture_pipe()
        out = pipe(
            mesh_path=str(mesh_path),
            save_dir=str(work),
            save_name="textured",
            uv_unwarp=True,
            uv_size=uv_size,
            rgb_path=str(grid_path),
            rgb_process_config=ModProcessConfig(view_upscale=True, inpaint_mode="view"),
            camera_azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
        )
        textured = trimesh.load(
            out.shaded_model_save_path, force="mesh", process=False
        )
        albedo_img = textured.visual.material.baseColorTexture
        albedo = np.asarray(albedo_img.convert("RGB"), dtype=np.float32) / 255.0
        uvs = np.asarray(textured.visual.uv, dtype=np.float32)
        new_verts = np.asarray(textured.vertices, dtype=np.float32)
        new_faces = np.asarray(textured.faces, dtype=np.int32)
        print(
            f"  TexturePipeline: {len(new_verts):,} verts after UVAtlas unwrap, "
            f"albedo {albedo.shape[0]}x{albedo.shape[1]}"
        )
        return new_verts, new_faces, uvs, albedo

    def bake_normal_map(
        self,
        verts: np.ndarray,           # (N, 3) unwrapped mesh vertices
        faces: np.ndarray,           # (F, 3)
        uvs: np.ndarray,             # (N, 2)
        photo_normals: np.ndarray,   # float32 (H, W, 3) in [-1, 1], camera space
        photo_alpha: np.ndarray,     # float32 (H, W) foreground mask
        uv_size: int,
    ) -> np.ndarray:
        """Bake the Sapiens camera-space normal map into a tangent-space atlas.

        Real projection, not a resize: the photo is registered to the front
        orthographic render via silhouette bounding boxes, sampled per UV
        texel through the camera, rotated from camera space to world space,
        then expressed in each texel's interpolated tangent frame. Texels
        the front view cannot see fall back to the flat normal (0, 0, 1).

        Returns float32 (uv_size, uv_size, 3) in [-1, 1].
        """
        self._ensure_ctx()
        import cv2
        from mvadapter.utils.mesh_utils import get_orthogonal_camera, render
        from mvadapter.utils.mesh_utils.uv import (
            SimpleUVValidityStrategy,
            uv_precompute,
            uv_render_geometry,
        )

        dev = self._device
        view_size = self.height
        mesh, _ = self._build_mesh_and_cameras(verts, faces, uvs, uv_size)

        front_cam = get_orthogonal_camera(
            elevation_deg=[0.0],
            distance=[1.8],
            left=-0.55, right=0.55, bottom=-0.55, top=0.55,
            azimuth_deg=[-90.0],
            device=str(dev),
        )

        pre = uv_precompute(self._ctx, mesh, height=uv_size, width=uv_size)
        geo = uv_render_geometry(
            self._ctx, mesh, front_cam,
            view_height=view_size, view_width=view_size,
            uv_precompute_output=pre,
            compute_depth_grad=True, depth_grad_dilation=5,
        )
        valid = SimpleUVValidityStrategy(
            aoi_cos_thresh=0.2, depth_grad_thresh=0.1
        )(pre, geo, None)[0]

        rout = render(
            self._ctx, mesh, front_cam,
            height=view_size, width=view_size,
            render_attr=False, normal_background=0.0,
        )
        rmask = rout.mask[0].cpu().numpy()
        rys, rxs = np.where(rmask)
        pys, pxs = np.where(photo_alpha > 0.5)
        if len(rys) == 0 or len(pys) == 0:
            raise RuntimeError("Empty silhouette during normal-map registration")
        sy = (rys.max() - rys.min()) / max(pys.max() - pys.min(), 1)
        sx = (rxs.max() - rxs.min()) / max(pxs.max() - pxs.min(), 1)
        affine = np.array(
            [
                [sx, 0.0, rxs.min() - pxs.min() * sx],
                [0.0, sy, rys.min() - pys.min() * sy],
            ],
            dtype=np.float64,
        )
        warped = cv2.warpAffine(
            photo_normals.astype(np.float32), affine, (view_size, view_size),
            flags=cv2.INTER_LINEAR, borderValue=(0.0, 0.0, 0.0),
        )

        img = torch.from_numpy(warped).to(dev).permute(2, 0, 1)[None]
        sampled = torch.nn.functional.grid_sample(
            img, geo.uv_pos_ndc[:1], align_corners=False, mode="bilinear"
        ).permute(0, 2, 3, 1)[0]

        rot = front_cam.c2w[0, :3, :3]
        n_world = torch.nn.functional.normalize(
            (sampled[..., None, :] * rot[None, None]).sum(-1), dim=-1, eps=1e-8
        )

        uv_clip = mesh.v_tex * 2.0 - 1.0
        clip = torch.cat(
            [uv_clip, torch.zeros_like(uv_clip[:, :1]), torch.ones_like(uv_clip[:, :1])],
            dim=1,
        )
        t_idx = mesh.t_pos_idx
        rast, _ = self._ctx.rasterize(clip[None], t_idx, (uv_size, uv_size))
        t_n, _ = self._ctx.interpolate(mesh.v_nrm[None], rast, t_idx)
        t_t, _ = self._ctx.interpolate(mesh.v_tang[None], rast, t_idx)
        N = torch.nn.functional.normalize(t_n[0], dim=-1, eps=1e-8)
        T = torch.nn.functional.normalize(t_t[0], dim=-1, eps=1e-8)
        B = torch.nn.functional.normalize(torch.cross(N, T, dim=-1), dim=-1, eps=1e-8)

        n_t = torch.stack(
            [
                (n_world * T).sum(-1),
                (n_world * B).sum(-1),
                (n_world * N).sum(-1),
            ],
            dim=-1,
        )
        n_t = torch.nn.functional.normalize(n_t, dim=-1, eps=1e-8)

        flat = torch.zeros_like(n_t)
        flat[..., 2] = 1.0
        sample_ok = sampled.norm(dim=-1) > 0.1
        use = (valid & sample_ok)[..., None]
        out = torch.where(use, n_t, flat)
        coverage = float(use.float().mean().item())
        print(f"  normal bake: front-view coverage {coverage:.1%}")
        # Same row-convention flip as bake_vertex_attribute: align the
        # nvdiffrast UV rasterization with the albedo/export convention.
        return np.clip(out.cpu().numpy(), -1.0, 1.0).astype(np.float32)[::-1].copy()

    def bake_vertex_attribute(
        self,
        faces: np.ndarray,
        uvs: np.ndarray,
        attribute: np.ndarray,
        uv_size: int,
        background: float | tuple[float, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rasterize a per-vertex attribute into the existing UV atlas."""
        self._ensure_ctx()
        dev = self._device
        uv = torch.from_numpy(uvs.astype(np.float32)).to(dev)
        tri = torch.from_numpy(faces.astype(np.int32)).to(dev)
        attr = torch.from_numpy(attribute.astype(np.float32)).to(dev)

        uv_clip = uv * 2.0 - 1.0
        clip = torch.cat(
            [
                uv_clip,
                torch.zeros_like(uv_clip[:, :1]),
                torch.ones_like(uv_clip[:, :1]),
            ],
            dim=1,
        )
        rast, _ = self._ctx.rasterize(clip[None], tri, (uv_size, uv_size))
        baked, _ = self._ctx.interpolate(attr[None], rast, tri)
        mask = rast[0, :, :, 3] > 0
        baked = baked[0]

        bg = torch.as_tensor(background, dtype=torch.float32, device=dev)
        if bg.ndim == 0:
            bg = bg.repeat(attr.shape[1])
        baked[~mask] = bg
        # nvdiffrast rasterizes UV-clip space with row index growing with v;
        # the albedo atlas (and the GLB export pairing it with these same
        # uvs) uses the image convention, row = (1 - v) * size. Flip to
        # match — without this, AO and composited head colors land on
        # vertically mirrored texels (verified empirically: misbaked
        # atlases sat at 99.7% inside the FLIPPED chart layout).
        return (
            baked.detach().cpu().numpy().astype(np.float32)[::-1].copy(),
            mask.detach().cpu().numpy()[::-1].copy(),
        )
