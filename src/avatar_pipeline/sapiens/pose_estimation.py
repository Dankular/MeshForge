from __future__ import annotations

import numpy as np
import torch

from avatar_pipeline.models.semantic import PoseData
from avatar_pipeline.sapiens._loader import load_model, preprocess, unwrap

# COCO-17 joint names in Sapiens output order
_COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class PoseEstimator:
    """Sapiens 1b pose estimator — COCO 17-keypoint heatmap decoder.

    Returns a PoseData with keypoints (x, y, confidence) in image pixel coords.
    """

    def __init__(self, n_kpts: int = 17) -> None:
        self._model: torch.jit.ScriptModule | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_kpts = n_kpts

    def load_pretrained(self, checkpoint_path=None) -> None:
        self._model = load_model("pose")

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model = load_model("pose")

    @torch.inference_mode()
    def estimate(self, image: np.ndarray) -> PoseData:
        """Estimate 2-D keypoints from an RGBA image.

        Parameters
        ----------
        image : np.ndarray
            uint8 or float32 (H, W, 4) RGBA.

        Returns
        -------
        PoseData
            17 COCO keypoints with (x_px, y_px, confidence).
        """
        self._ensure_loaded()
        h, w = image.shape[:2]
        tensor = preprocess(image, self._device)

        out = unwrap(self._model(tensor))          # (1, K, hH, hW)
        hms = out[0].float().cpu().numpy()         # (K, hH, hW)

        k = min(self.n_kpts, hms.shape[0])
        keypoints = np.zeros((k, 3), dtype=np.float32)

        for i in range(k):
            hm = hms[i]
            conf = float(hm.max())
            iy, ix = np.unravel_index(hm.argmax(), hm.shape)
            # Scale heatmap coords back to original image size
            x = ix / hm.shape[1] * w
            y = iy / hm.shape[0] * h
            keypoints[i] = [x, y, conf]

        names = _COCO17_NAMES[:k]
        return PoseData(joint_names=names, keypoints=keypoints)
