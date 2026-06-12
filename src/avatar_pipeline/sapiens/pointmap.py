import numpy as np

from avatar_pipeline.models.semantic import SemanticMap


class PointmapEstimator:
    """Back-projects the Sapiens depth map into a camera-space pointmap.

    Purely geometric: (x, y) come from the pixel grid, z from the real
    estimated depth. No learned weights are involved.
    """

    def __init__(self) -> None:
        self.depth_scale = 2.0

    def estimate(self, depth: np.ndarray, semantic_map: SemanticMap) -> np.ndarray:
        h, w = depth.shape
        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, h, dtype=np.float32),
            np.linspace(-1.0, 1.0, w, dtype=np.float32),
            indexing="ij",
        )

        z = 1.0 - self.depth_scale * depth.astype(np.float32)
        if semantic_map.alpha is not None:
            z = z * np.clip(semantic_map.alpha, 0.0, 1.0)

        pointmap = np.stack([xx, -yy, z], axis=2).astype(np.float32)
        fg = semantic_map.labels > 0
        pointmap[~fg] = 0.0
        return pointmap
