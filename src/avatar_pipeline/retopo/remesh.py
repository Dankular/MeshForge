import numpy as np
from scipy.spatial import cKDTree

from avatar_pipeline.models.mesh import Mesh


class Remesher:
    def remesh(self, mesh: Mesh) -> Mesh:
        vertices = mesh.vertices.astype(np.float32).copy()
        tree = cKDTree(vertices)
        for _ in range(3):
            dists, idx = tree.query(vertices, k=min(10, len(vertices)))
            nbr = vertices[idx]
            smooth = np.mean(nbr, axis=1)
            vertices = 0.65 * vertices + 0.35 * smooth
            tree = cKDTree(vertices)

        rounded = np.round(vertices, 5)
        unique, inv = np.unique(rounded, axis=0, return_inverse=True)
        faces = inv[mesh.faces]
        valid = (
            (faces[:, 0] != faces[:, 1])
            & (faces[:, 1] != faces[:, 2])
            & (faces[:, 0] != faces[:, 2])
        )
        faces = faces[valid]

        centered = unique - np.mean(unique, axis=0, keepdims=True)
        max_norm = float(np.max(np.linalg.norm(centered, axis=1))) + 1e-8
        normalized = (centered / max_norm).astype(np.float32)
        return Mesh(
            vertices=normalized,
            faces=faces.astype(np.int32),
            normals=mesh.normals,
            uvs=mesh.uvs,
            semantic_regions=mesh.semantic_regions,
        )
