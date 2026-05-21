import numpy as np
from scipy.spatial import cKDTree

from avatar_pipeline.models.mesh import Mesh


class CanonicalTransfer:
    def transfer(self, source_mesh: Mesh, canonical_mesh: Mesh) -> Mesh:
        tree = cKDTree(source_mesh.vertices)
        _, idx = tree.query(canonical_mesh.vertices, k=1)
        transferred = source_mesh.vertices[idx]
        # Strong displacement: canonical acts as a cage, TripoSG detail shapes it
        blended = 0.35 * canonical_mesh.vertices + 0.65 * transferred
        faces = canonical_mesh.faces

        return Mesh(
            vertices=blended.astype(np.float32),
            faces=faces,
            normals=source_mesh.normals,
            uvs=canonical_mesh.uvs,
            semantic_regions=source_mesh.semantic_regions,
        )
