import numpy as np

from avatar_pipeline.models.semantic import SemanticMap
from avatar_pipeline.pretrained_stats import checkpoint_signature


class PointmapEstimator:
    def __init__(self) -> None:
        self.depth_scale = 2.0

    def load_pretrained(self, checkpoint_path) -> None:
        _, _, c = checkpoint_signature(checkpoint_path)
        self.depth_scale = float(np.clip(1.6 + 1.0 * c, 1.4, 2.8))

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
