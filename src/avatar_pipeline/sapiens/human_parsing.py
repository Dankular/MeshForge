from __future__ import annotations

import numpy as np
import torch

from avatar_pipeline.models.semantic import SemanticMap
from avatar_pipeline.sapiens._loader import load_model, preprocess, unwrap

# Goliath 28-class → pipeline internal palette.
# Class order verified against the official sapiens repo
# (seg/mmseg/datasets/goliath.py GOLIATH_CLASSES — the 34 original classes
# minus the 6 removed ones), NOT guessed: 0=Background, 1=Apparel,
# 2=Face_Neck, 3=Hair, 4=Left_Foot, 5=Left_Hand, 6=Left_Lower_Arm,
# 7=Left_Lower_Leg, 8=Left_Shoe, 9=Left_Sock, 10=Left_Upper_Arm,
# 11=Left_Upper_Leg, 12=Lower_Clothing, 13=Right_Foot, 14=Right_Hand,
# 15=Right_Lower_Arm, 16=Right_Lower_Leg, 17=Right_Shoe, 18=Right_Sock,
# 19=Right_Upper_Arm, 20=Right_Upper_Leg, 21=Torso, 22=Upper_Clothing,
# 23=Lower_Lip, 24=Upper_Lip, 25=Lower_Teeth, 26=Upper_Teeth, 27=Tongue.
_GOLIATH_TO_INTERNAL: dict[int, int] = {
    0:  0,   # Background
    1:  4,   # Apparel          → cloth
    2:  1,   # Face_Neck        → head
    3:  5,   # Hair             → hair
    4:  3,   # Left_Foot        → legs
    5:  6,   # Left_Hand        → skin
    6:  6,   # Left_Lower_Arm   → skin
    7:  3,   # Left_Lower_Leg   → legs
    8:  4,   # Left_Shoe        → cloth
    9:  4,   # Left_Sock        → cloth
    10: 6,   # Left_Upper_Arm   → skin
    11: 3,   # Left_Upper_Leg   → legs
    12: 4,   # Lower_Clothing   → cloth
    13: 3,   # Right_Foot       → legs
    14: 6,   # Right_Hand       → skin
    15: 6,   # Right_Lower_Arm  → skin
    16: 3,   # Right_Lower_Leg  → legs
    17: 4,   # Right_Shoe       → cloth
    18: 4,   # Right_Sock       → cloth
    19: 6,   # Right_Upper_Arm  → skin
    20: 3,   # Right_Upper_Leg  → legs
    21: 2,   # Torso            → torso
    22: 4,   # Upper_Clothing   → cloth
    23: 1,   # Lower_Lip        → head
    24: 1,   # Upper_Lip        → head
    25: 1,   # Lower_Teeth      → head
    26: 1,   # Upper_Teeth      → head
    27: 1,   # Tongue           → head
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
        self._model = load_model("seg")

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model = load_model("seg")

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
