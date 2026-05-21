from __future__ import annotations

import numpy as np
import torch

from avatar_pipeline.models.semantic import SemanticMap
from avatar_pipeline.sapiens._loader import load_model, preprocess, unwrap

# Goliath 28-class → pipeline internal palette
# Goliath classes 0 = background; 1–27 = body parts
# We collapse into 8 semantic regions used downstream.
_GOLIATH_TO_INTERNAL: dict[int, int] = {
    0:  0,   # background
    1:  6,   # torso-skin  → skin
    2:  6,   # right-hand  → skin
    3:  6,   # left-hand   → skin
    4:  6,   # right-foot  → skin
    5:  6,   # left-foot   → skin
    6:  1,   # right-face  → head
    7:  1,   # left-face   → head
    8:  6,   # right-arm-lower → skin
    9:  6,   # left-arm-lower  → skin
    10: 6,   # right-arm-upper → skin
    11: 6,   # left-arm-upper  → skin
    12: 3,   # right-leg-lower → legs
    13: 3,   # left-leg-lower  → legs
    14: 3,   # right-leg-upper → legs
    15: 3,   # left-leg-upper  → legs
    16: 1,   # head    → head
    17: 1,   # neck    → head
    18: 7,   # left-eye  → eyes
    19: 7,   # right-eye → eyes
    20: 1,   # left-brow  → head
    21: 1,   # right-brow → head
    22: 1,   # mouth → head
    23: 1,   # nose  → head
    24: 4,   # clothing → cloth
    25: 5,   # hair
    26: 1,   # left-ear  → head
    27: 1,   # right-ear → head
}

_LUT = np.array(
    [_GOLIATH_TO_INTERNAL.get(i, 0) for i in range(28)], dtype=np.int32
)

_PALETTE: dict[int, str] = {
    0: "background",
    1: "head",
    2: "torso",
    3: "legs",
    4: "cloth",
    5: "hair",
    6: "skin",
    7: "eyes",
}


class HumanParser:
    """Sapiens 1b 28-class body-part segmentation.

    Returns a SemanticMap with labels collapsed to 8 internal regions.
    """

    def __init__(self) -> None:
        self._model: torch.jit.ScriptModule | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_pretrained(self, checkpoint_path=None) -> None:
        self._model = load_model("seg", self._device)

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model = load_model("seg", self._device)

    @torch.inference_mode()
    def parse(self, image: np.ndarray) -> SemanticMap:
        """Segment body parts from an RGBA image.

        Parameters
        ----------
        image : np.ndarray
            uint8 or float32 (H, W, 4) RGBA.

        Returns
        -------
        SemanticMap
            ``labels`` are 8-class internal palette IDs (int32).
        """
        self._ensure_loaded()
        h, w = image.shape[:2]
        tensor = preprocess(image, self._device)

        out = unwrap(self._model(tensor))              # (1, 28, 1024, 768)
        goliath = out[0].argmax(0).byte().cpu().numpy()   # (1024, 768) uint8

        import cv2
        goliath = cv2.resize(goliath, (w, h), interpolation=cv2.INTER_NEAREST)

        labels = _LUT[goliath.astype(np.int32)].astype(np.int32)

        # alpha for downstream: foreground = any non-background label
        alpha = (labels > 0).astype(np.float32)
        confidence = np.where(labels > 0, 0.9, 0.1).astype(np.float32)

        return SemanticMap(
            labels=labels,
            confidence=confidence,
            palette=_PALETTE,
            alpha=alpha,
        )
