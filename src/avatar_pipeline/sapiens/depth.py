from __future__ import annotations

import numpy as np
import torch

from avatar_pipeline.sapiens._loader import load_model, preprocess, unwrap


class DepthEstimator:
    """Sapiens 1b depth estimator: float32 (H, W) relative depth in [0, 1].

    (Geometric pointmap derivation lives in sapiens/pointmap.py.)
    """

    def __init__(self) -> None:
        self._model: torch.jit.ScriptModule | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_pretrained(self, checkpoint_path=None) -> None:
        self._model = load_model("depth")

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model = load_model("depth")

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

        return d_norm.astype(np.float32)
