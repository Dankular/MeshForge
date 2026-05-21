import numpy as np
from scipy.spatial import cKDTree


class AmbientOcclusionBaker:
    def bake(self, mesh) -> np.ndarray:
        v = mesh.vertices.astype(np.float32)
        tree = cKDTree(v)
        _, idx = tree.query(v, k=min(16, len(v)))
        nbr = v[idx]
        local_center = np.mean(nbr, axis=1)
        cavity = np.linalg.norm(v - local_center, axis=1)
        cavity = cavity / (np.max(cavity) + 1e-8)

        y = v[:, 1]
        h_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)
        ao = 1.0 - (0.55 * cavity + 0.30 * h_norm)
        return np.clip(ao, 0.2, 1.0).astype(np.float32)
