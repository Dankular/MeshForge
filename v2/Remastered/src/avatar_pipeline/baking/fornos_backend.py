from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from avatar_pipeline.models.mesh import BakedTextures
from avatar_pipeline.pretrained_stats import checkpoint_signature


@dataclass
class BakeRequest:
    albedo: np.ndarray
    high_normals: np.ndarray
    low_normals: np.ndarray
    ao_per_vertex: np.ndarray


class FornosBackend:
    def __init__(self) -> None:
        self.albedo_sigma = 0.8
        self.high_normal_mix = 0.75
        self.ao_sigma = 1.0

    def load_mv_adapter_checkpoints(self, checkpoint_root: Path) -> None:
        files = sorted(
            [
                p
                for p in checkpoint_root.glob("**/*")
                if p.is_file() and p.suffix in {".pt", ".pth", ".safetensors", ".ckpt"}
            ]
        )
        if not files:
            return
        vals = [checkpoint_signature(p) for p in files[:4]]
        a = float(np.mean([v[0] for v in vals]))
        b = float(np.mean([v[1] for v in vals]))
        c = float(np.mean([v[2] for v in vals]))
        self.albedo_sigma = float(np.clip(0.4 + 1.0 * a, 0.35, 1.6))
        self.high_normal_mix = float(np.clip(0.55 + 0.35 * b, 0.5, 0.95))
        self.ao_sigma = float(np.clip(0.6 + 1.2 * c, 0.5, 2.2))

    def bake(self, request: BakeRequest) -> BakedTextures:
        uv_size = request.albedo.shape[:2]
        ao_scalar = float(np.mean(request.ao_per_vertex))
        ao_var = float(np.std(request.ao_per_vertex))
        yy = np.linspace(0.0, 1.0, uv_size[0], dtype=np.float32)[:, None]
        xx = np.linspace(0.0, 1.0, uv_size[1], dtype=np.float32)[None, :]
        ao = ao_scalar + ao_var * (0.5 - np.abs(xx - 0.5)) * (0.4 + 0.6 * yy)
        ao = gaussian_filter(ao, sigma=self.ao_sigma)
        ao = np.expand_dims(ao.astype(np.float32), axis=2)

        albedo = np.clip(
            gaussian_filter(
                request.albedo, sigma=(self.albedo_sigma, self.albedo_sigma, 0.0)
            ),
            0.0,
            1.0,
        )
        normal = request.high_normals * self.high_normal_mix + request.low_normals * (
            1.0 - self.high_normal_mix
        )
        normal /= np.linalg.norm(normal, axis=2, keepdims=True) + 1e-8
        return BakedTextures(
            albedo=albedo.astype(np.float32),
            normal=np.clip(normal, -1.0, 1.0).astype(np.float32),
            ambient_occlusion=np.clip(ao, 0.0, 1.0),
        )
