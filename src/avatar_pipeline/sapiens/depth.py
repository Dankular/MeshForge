from __future__ import annotations

import numpy as np
import torch

from avatar_pipeline.sapiens._loader import load_model, preprocess, unwrap


class DepthEstimator:
    """Sapiens 1b depth estimator.

    Primary output  : float32 (H, W) relative depth map in [0, 1].
    Secondary output: float32 (H, W, 3) XYZ pointmap via pinhole back-projection,
                      available via the last call to ``pointmap`` property.
    """

    def __init__(self, fov_deg: float = 60.0) -> None:
        self._model: torch.jit.ScriptModule | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fov_deg = fov_deg
        self._last_pointmap: np.ndarray | None = None

    def load_pretrained(self, checkpoint_path=None) -> None:
        self._model = load_model("depth", self._device)

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model = load_model("depth", self._device)

    @property
    def pointmap(self) -> np.ndarray | None:
        """XYZ pointmap from the most recent ``estimate()`` call."""
        return self._last_pointmap

    @torch.inference_mode()
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth from an RGBA image.

        Parameters
        ----------
        image : np.ndarray
            uint8 or float32 (H, W, 4) RGBA.

        Returns
        -------
        np.ndarray
            float32 (H, W) depth in [0, 1] (higher = farther).
        """
        self._ensure_loaded()
        h, w = image.shape[:2]
        tensor = preprocess(image, self._device)

        out = unwrap(self._model(tensor))
        # depth model may return (1,1,H,W) or (1,H,W)
        if out.ndim == 4:
            d = out[0, 0]
        else:
            d = out[0]
        d = d.float().cpu().numpy()

        import cv2
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)

        d_min, d_max = d.min(), d.max()
        if d_max > d_min:
            d_norm = (d - d_min) / (d_max - d_min)
        else:
            d_norm = np.zeros_like(d)

        self._last_pointmap = self._back_project(d_norm, h, w)
        return d_norm.astype(np.float32)

    def _back_project(self, depth: np.ndarray, h: int, w: int) -> np.ndarray:
        """Pinhole back-projection: depth → XYZ pointmap."""
        fx = (w / 2.0) / np.tan(np.deg2rad(self.fov_deg / 2.0))
        cx, cy = w / 2.0, h / 2.0
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        z = depth.astype(np.float32)
        x = (xs - cx) / fx * z
        y = (ys - cy) / fx * z
        return np.stack([x, y, z], axis=2)
