import numpy as np
from scipy.spatial import cKDTree

from avatar_pipeline.models.mesh import Mesh


class MeshFusion:
    def fuse(self, body_mesh: Mesh, head_mesh: Mesh) -> Mesh:
        body_vertices = body_mesh.vertices.astype(np.float32)
        head_vertices = head_mesh.vertices.astype(np.float32)

        body_top = np.max(body_vertices[:, 1])
        head_bottom = np.min(head_vertices[:, 1])
        body_band = body_vertices[:, 1] > (
            body_top - 0.05 * (body_top - np.min(body_vertices[:, 1]) + 1e-8)
        )
        head_band = head_vertices[:, 1] < (
            head_bottom + 0.20 * (np.max(head_vertices[:, 1]) - head_bottom + 1e-8)
        )

        adjusted_head = head_vertices.copy()
        if np.any(body_band) and np.any(head_band):
            tree = cKDTree(body_vertices[body_band][:, [0, 2]])
            hb_idx = np.where(head_band)[0]
            _, nn = tree.query(head_vertices[hb_idx][:, [0, 2]], k=1)
            targets = body_vertices[body_band][nn]
            adjusted_head[hb_idx] = 0.65 * head_vertices[hb_idx] + 0.35 * targets

        vertices = np.concatenate([body_vertices, adjusted_head], axis=0)
        head_faces = head_mesh.faces + body_mesh.vertices.shape[0]
        faces = np.concatenate([body_mesh.faces, head_faces], axis=0)
        return Mesh(
            vertices=vertices,
            faces=faces,
            semantic_regions=body_mesh.semantic_regions + head_mesh.semantic_regions,
        )
