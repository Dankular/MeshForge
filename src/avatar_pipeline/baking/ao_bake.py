"""Per-vertex ambient occlusion via real hemispheric ray casting.

For every vertex, rays are cast over the cosine-weighted hemisphere around
the vertex normal (Malley's method: normal + uniform sphere sample) and
tested against the full mesh with Embree (trimesh's ray backend). AO is the
unoccluded fraction — a genuine sky-visibility integral, not a curvature
heuristic.
"""
from __future__ import annotations

import numpy as np
import trimesh


class AmbientOcclusionBaker:
    def __init__(self, num_samples: int = 64, seed: int = 0) -> None:
        self.num_samples = num_samples
        self.seed = seed

    def bake(self, mesh) -> np.ndarray:
        tm = trimesh.Trimesh(
            vertices=mesh.vertices.astype(np.float64),
            faces=mesh.faces.astype(np.int64),
            process=False,
        )
        if not trimesh.ray.has_embree:
            print(
                "[ao_bake] WARNING: Embree backend unavailable "
                "(pip install embreex); ray casting will be slow"
            )

        verts = np.asarray(tm.vertices)
        normals = np.asarray(tm.vertex_normals)
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
        n_verts = len(verts)

        # Offset origins along the normal so rays don't self-hit the
        # originating surface. Scale-relative epsilon.
        eps = float(np.linalg.norm(tm.bounds[1] - tm.bounds[0])) * 1e-4
        origins = verts + normals * eps

        rng = np.random.default_rng(self.seed)
        unoccluded = np.zeros(n_verts, dtype=np.float64)
        for _ in range(self.num_samples):
            s = rng.normal(size=(n_verts, 3))
            s /= np.linalg.norm(s, axis=1, keepdims=True) + 1e-12
            d = normals + s  # cosine-weighted hemisphere about the normal
            norm = np.linalg.norm(d, axis=1, keepdims=True)
            # Degenerate case (sample exactly opposite the normal): use normal
            d = np.where(norm > 1e-6, d / np.maximum(norm, 1e-12), normals)
            hit = tm.ray.intersects_any(origins, d)
            unoccluded += ~hit

        ao = (unoccluded / self.num_samples).astype(np.float32)
        return np.clip(ao, 0.0, 1.0)
