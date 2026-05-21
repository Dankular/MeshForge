import numpy as np
from scipy.ndimage import gaussian_filter1d

from avatar_pipeline.models.mesh import Mesh


class HeadGenerator:
    def _head_shape_from_mask(
        self, head_mask: np.ndarray
    ) -> tuple[float, float, float]:
        h, w = head_mask.shape
        ys, xs = np.where(head_mask)
        if ys.size < 8:
            return 1.0, 1.1, 0.9
        width = float(np.max(xs) - np.min(xs) + 1)
        height = float(np.max(ys) - np.min(ys) + 1)
        ratio = np.clip(width / max(height, 1.0), 0.6, 1.2)

        row_widths = np.zeros(h, dtype=np.float32)
        for y in range(h):
            row = np.where(head_mask[y])[0]
            if row.size > 1:
                row_widths[y] = float(row[-1] - row[0] + 1)
        row_widths = gaussian_filter1d(row_widths, sigma=1.4)
        mid = int(np.argmax(row_widths)) if np.any(row_widths > 0) else h // 2
        upper = np.mean(row_widths[max(0, mid - h // 6) : mid + 1])
        lower = np.mean(row_widths[mid : min(h, mid + h // 6)])
        jaw_ratio = np.clip(lower / max(upper, 1e-6), 0.65, 1.1)

        sx = float(0.85 + 0.35 * ratio)
        sy = float(1.0 + 0.25 * (1.0 / max(ratio, 0.6)))
        sz = float(0.78 + 0.25 * jaw_ratio)
        return sx, sy, sz

    def generate(self, inputs: dict) -> Mesh:
        radius = float(inputs.get("radius", 0.22))
        subdivisions = int(inputs.get("subdivisions", 2))
        head_mask = inputs.get("head_mask")

        phi = (1.0 + np.sqrt(5.0)) / 2.0
        vertices = np.array(
            [
                [-1, phi, 0],
                [1, phi, 0],
                [-1, -phi, 0],
                [1, -phi, 0],
                [0, -1, phi],
                [0, 1, phi],
                [0, -1, -phi],
                [0, 1, -phi],
                [phi, 0, -1],
                [phi, 0, 1],
                [-phi, 0, -1],
                [-phi, 0, 1],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [
                [0, 11, 5],
                [0, 5, 1],
                [0, 1, 7],
                [0, 7, 10],
                [0, 10, 11],
                [1, 5, 9],
                [5, 11, 4],
                [11, 10, 2],
                [10, 7, 6],
                [7, 1, 8],
                [3, 9, 4],
                [3, 4, 2],
                [3, 2, 6],
                [3, 6, 8],
                [3, 8, 9],
                [4, 9, 5],
                [2, 4, 11],
                [6, 2, 10],
                [8, 6, 7],
                [9, 8, 1],
            ],
            dtype=np.int32,
        )

        for _ in range(max(0, subdivisions)):
            edge_mid = {}
            new_faces = []

            def midpoint(a: int, b: int) -> int:
                key = (a, b) if a < b else (b, a)
                if key in edge_mid:
                    return edge_mid[key]
                v = (vertices[a] + vertices[b]) * 0.5
                edge_mid[key] = len(v_list)
                v_list.append(v)
                return edge_mid[key]

            v_list = [v for v in vertices]
            for tri in faces:
                a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                ab = midpoint(a, b)
                bc = midpoint(b, c)
                ca = midpoint(c, a)
                new_faces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])

            vertices = np.asarray(v_list, dtype=np.float64)
            faces = np.asarray(new_faces, dtype=np.int32)

        norms = np.linalg.norm(vertices, axis=1, keepdims=True) + 1e-8
        vertices = (vertices / norms) * radius
        sx, sy, sz = (1.0, 1.1, 0.9)
        if isinstance(head_mask, np.ndarray):
            sx, sy, sz = self._head_shape_from_mask(head_mask.astype(bool))
        vertices[:, 0] *= sx
        vertices[:, 1] *= sy
        vertices[:, 2] *= sz

        jaw_mask = vertices[:, 1] < np.quantile(vertices[:, 1], 0.28)
        vertices[jaw_mask, 0] *= 0.92
        vertices[jaw_mask, 2] *= 0.95
        vertices = vertices.astype(np.float32)
        return Mesh(vertices=vertices, faces=faces, semantic_regions=["head"])
