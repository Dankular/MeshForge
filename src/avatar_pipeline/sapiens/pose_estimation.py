from __future__ import annotations

import numpy as np
import torch

from avatar_pipeline.models.semantic import PoseData
from avatar_pipeline.sapiens._loader import load_model, preprocess, unwrap

# Sapiens-1b pose is the 308-keypoint Goliath head, NOT COCO-17: the first
# 15 channels are COCO-like but WITHOUT wrists (ids 9/10 are the hips), and
# the wrists live in the hand sections (right 41, left 62). Verified against
# sapiens/pose/configs/_base_/datasets/goliath.py. Decoding the first 17
# channels under COCO names mislabeled hips as wrists.
_GOLIATH_SUBSET: dict[str, int] = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_hip": 9, "right_hip": 10,
    "left_knee": 11, "right_knee": 12,
    "left_ankle": 13, "right_ankle": 14,
    "right_wrist": 41, "left_wrist": 62,
}


class PoseEstimator:
    """Sapiens 1b pose estimator — COCO 17-keypoint heatmap decoder.

    Returns a PoseData with keypoints (x, y, confidence) in image pixel coords.
    """

    def __init__(self) -> None:
        self._model: torch.jit.ScriptModule | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        names = list(_GOLIATH_SUBSET)
        keypoints = np.zeros((len(names), 3), dtype=np.float32)
        for i, name in enumerate(names):
            channel = _GOLIATH_SUBSET[name]
            if channel >= hms.shape[0]:
                continue  # leaves (0, 0, 0) — conf 0 fails downstream gates
            hm = hms[channel]
            conf = float(hm.max())
            iy, ix = np.unravel_index(hm.argmax(), hm.shape)
            # Scale heatmap coords back to original image size
            x = ix / hm.shape[1] * w
            y = iy / hm.shape[0] * h
            keypoints[i] = [x, y, conf]

        return PoseData(joint_names=names, keypoints=keypoints)
