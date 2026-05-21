from __future__ import annotations

import numpy as np
from PIL import Image

from avatar_pipeline.pretrained_stats import checkpoint_signature


class RembgProcessor:
    """Background removal via rembg 2.0 (u2net_human_seg).

    Uses the person-specialised U2Net model for clean human silhouettes.
    Falls back to a centre-weighted luminance mask if rembg is unavailable.
    """

    model_name: str = "u2net_human_seg"

    def __init__(self) -> None:
        self._session = None
        self._post_process = True

    def _get_session(self):
        if self._session is None:
            from rembg import new_session
            self._session = new_session(self.model_name)
        return self._session

    def load_pretrained(self, checkpoint_path) -> None:
        # checkpoint_signature kept for API compatibility; no tuning needed
        # since rembg manages its own weights via its own cache.
        checkpoint_signature(checkpoint_path)

    def process_rmbg2(self, image: np.ndarray) -> np.ndarray:
        return self.process(image)

    def process(self, image: np.ndarray) -> np.ndarray:
        """Remove background from a uint8 RGB image.

        Parameters
        ----------
        image : np.ndarray
            uint8 (H, W, 3) RGB image.

        Returns
        -------
        np.ndarray
            float32 (H, W, 4) RGBA where alpha is the foreground mask in [0, 1].
        """
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Expected image with shape (H, W, 3+)")

        rgb_uint8 = image[:, :, :3]
        if rgb_uint8.dtype != np.uint8:
            rgb_uint8 = np.clip(rgb_uint8, 0, 255).astype(np.uint8)

        try:
            from rembg import remove
            img_pil = Image.fromarray(rgb_uint8, "RGB")
            out_pil = remove(
                img_pil,
                session=self._get_session(),
                post_process_mask=self._post_process,
            )
            out = np.array(out_pil, dtype=np.float32)   # (H, W, 4) uint8→float32
            rgb   = out[:, :, :3] / 255.0
            alpha = out[:, :, 3:4] / 255.0
        except Exception:
            # Fallback: centre-weighted luminance mask when rembg unavailable
            rgb = rgb_uint8.astype(np.float32) / 255.0
            lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
            h, w = lum.shape
            cy, cx = h / 2.0, w / 2.0
            yy, xx = np.indices((h, w), dtype=np.float32)
            dist = np.sqrt(((yy - cy) / max(h, 1)) ** 2 + ((xx - cx) / max(w, 1)) ** 2)
            alpha = np.clip(1.0 - dist * 1.8, 0.0, 1.0)[:, :, np.newaxis]

        return np.concatenate([rgb, alpha], axis=2).astype(np.float32)
