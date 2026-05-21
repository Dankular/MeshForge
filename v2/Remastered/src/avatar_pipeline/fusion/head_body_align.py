import numpy as np

from avatar_pipeline.models.mesh import Mesh


class HeadBodyAligner:
    def align(self, body: Mesh, head: Mesh) -> Mesh:
        body_top = float(np.max(body.vertices[:, 1]))
        neck_band = body.vertices[:, 1] > (
            body_top - 0.08 * (body_top - np.min(body.vertices[:, 1]) + 1e-8)
        )
        neck_center = (
            np.mean(body.vertices[neck_band], axis=0)
            if np.any(neck_band)
            else np.mean(body.vertices, axis=0)
        )

        head_bottom = float(np.min(head.vertices[:, 1]))
        chin_band = head.vertices[:, 1] < (
            head_bottom + 0.12 * (np.max(head.vertices[:, 1]) - head_bottom + 1e-8)
        )
        chin_center = (
            np.mean(head.vertices[chin_band], axis=0)
            if np.any(chin_band)
            else np.mean(head.vertices, axis=0)
        )

        offset = (neck_center - chin_center).astype(np.float32)
        offset[1] += 0.01
        return Mesh(
            vertices=head.vertices + offset,
            faces=head.faces,
            normals=head.normals,
            uvs=head.uvs,
            semantic_regions=head.semantic_regions,
        )
