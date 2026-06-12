from __future__ import annotations

import numpy as np
import torch

from avatar_pipeline.sapiens._loader import INF_H, INF_W, load_model, preprocess, unwrap


class SurfaceNormals:
    """Sapiens 1b surface-normal estimator.

    Output: float32 (H, W, 3) unit normals in [-1, 1] camera space.
    """

    def __init__(self) -> None:
        self._model: torch.jit.ScriptModule | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_pretrained(self, checkpoint_path=None) -> None:
        # checkpoint_path unused — model is fetched from HuggingFace
        self._model = load_model("normal")

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model = load_model("normal")

    @torch.inference_mode()
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Estimate surface normals from an RGBA image.

        Parameters
        ----------
        image : np.ndarray
            uint8 or float32 (H, W, 4) RGBA.

        Returns
        -------
        np.ndarray
            float32 (H, W, 3) normals in [-1, 1].
        """
        self._ensure_loaded()
        h, w = image.shape[:2]
        tensor = preprocess(image, self._device)

        out = unwrap(self._model(tensor))          # (1, 3, 1024, 768)
        nrm = out[0].permute(1, 2, 0).float().cpu().numpy()   # (1024, 768, 3)

        import cv2
        nrm = cv2.resize(nrm, (w, h), interpolation=cv2.INTER_LINEAR)

        denom = np.linalg.norm(nrm, axis=2, keepdims=True) + 1e-8
        return (nrm / denom).astype(np.float32)
