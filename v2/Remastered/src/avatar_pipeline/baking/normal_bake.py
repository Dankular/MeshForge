import numpy as np
from scipy.ndimage import gaussian_filter


class NormalBaker:
    def bake(self, high: np.ndarray, low: np.ndarray) -> np.ndarray:
        high = high.astype(np.float32)
        low = low.astype(np.float32)
        detail = gaussian_filter(high - low, sigma=(1.0, 1.0, 0.0))
        detail[:, :, 2] = np.maximum(detail[:, :, 2], -0.25)
        normal = low + 0.7 * detail
        norm = np.linalg.norm(normal, axis=2, keepdims=True) + 1e-8
        return (normal / norm).astype(np.float32)
