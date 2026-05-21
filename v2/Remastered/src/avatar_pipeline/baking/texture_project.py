import numpy as np
from scipy.ndimage import gaussian_filter


class TextureProjector:
    def project(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        if source.shape != target.shape:
            raise ValueError("Source and target textures must have the same shape")

        source = np.clip(source.astype(np.float32), 0.0, 1.0)
        target = np.clip(target.astype(np.float32), 0.0, 1.0)
        source_grad = np.sum(np.abs(np.gradient(source, axis=(0, 1))), axis=0)
        target_grad = np.sum(np.abs(np.gradient(target, axis=(0, 1))), axis=0)
        source_conf = source_grad / (source_grad + target_grad + 1e-8)
        source_conf = gaussian_filter(source_conf, sigma=(1.0, 1.0, 0.0))
        return np.clip(source_conf * source + (1.0 - source_conf) * target, 0.0, 1.0)
