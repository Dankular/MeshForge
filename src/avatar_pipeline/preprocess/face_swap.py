"""Identity enhancement for generated references — HyperSwap + GFPGAN.

Mirrors the deployed VAST app's recipe (app.py hyperswap_views): ArcFace-256
alignment via estimateAffinePartial2D, hyperswap_1a_256 onnx swap
conditioned on the source's normalized ArcFace embedding, soft-mask blend
back into the frame, then GFPGAN v1.4 restore (weight 0.5) on the padded
face crop.

Why this stage exists: IP-Adapter FaceID conditions composition but cannot
impose identity on a ~50px face in a full-body frame — measured 0.11 FaceID
cosine between the candid and the raw SD reference, and every downstream
stage faithfully carries that loss. Swapping transplants the true identity
pixels before PSHuman ever sees the reference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]

HYPERSWAP_PATH = _REPO_ROOT / "checkpoints" / "face_enhance" / "hyperswap_1a_256.onnx"
GFPGAN_PATH = _REPO_ROOT / "checkpoints" / "face_enhance" / "GFPGANv1.4.pth"

# ArcFace 112 5-point template scaled to the 256px swap input (app.py).
ARCFACE_256 = (
    np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32,
    )
    * (256 / 112)
    + (256 - 112 * (256 / 112)) / 2
)


class IdentityEnhancer:
    def __init__(self, checkpoint_root: Path | None = None) -> None:
        root = checkpoint_root or (_REPO_ROOT / "checkpoints")
        self._swap_path = root / "face_enhance" / "hyperswap_1a_256.onnx"
        self._gfpgan_path = root / "face_enhance" / "GFPGANv1.4.pth"
        self._sess = None
        self._restorer = None

    def _get_session(self):
        if self._sess is None:
            import onnxruntime as ort

            if not self._swap_path.exists():
                raise FileNotFoundError(
                    f"HyperSwap weights missing: {self._swap_path}"
                )
            self._sess = ort.InferenceSession(
                str(self._swap_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        return self._sess

    def _get_restorer(self):
        if self._restorer is None:
            from gfpgan import GFPGANer

            if not self._gfpgan_path.exists():
                raise FileNotFoundError(
                    f"GFPGAN weights missing: {self._gfpgan_path}"
                )
            self._restorer = GFPGANer(
                model_path=str(self._gfpgan_path),
                upscale=2,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
        return self._restorer

    def enhance(
        self,
        image_rgb: np.ndarray,
        source_embedding: np.ndarray,
        extractor,
        full_frame_upscale: bool = False,
    ) -> np.ndarray:
        """Swap the source identity onto the largest face and restore it.

        image_rgb: uint8 RGB frame (the generated T-pose reference).
        source_embedding: 512-d ArcFace embedding of the candid face
            (insightface normed_embedding).
        extractor: a FaceIdentityExtractor (reused so detection behaves
            identically to the scoring path).
        Returns the enhanced uint8 RGB frame. Fails loudly when the frame
        has no detectable face — no silent passthrough.
        """
        import cv2

        hit = extractor.detect(image_rgb)
        if hit is None:
            raise RuntimeError(
                "identity enhancement found no face in the generated "
                "reference; cannot swap"
            )
        face, bgr = hit

        emb = np.asarray(source_embedding, dtype=np.float32).reshape(1, -1)
        emb = emb / max(float(np.linalg.norm(emb)), 1e-9)

        M, _ = cv2.estimateAffinePartial2D(
            face.kps, ARCFACE_256, method=cv2.RANSAC, ransacReprojThreshold=100
        )
        h, w = bgr.shape[:2]
        aligned = cv2.warpAffine(bgr, M, (256, 256), flags=cv2.INTER_LINEAR)
        t = (
            ((aligned.astype(np.float32) / 255 - 0.5) / 0.5)[:, :, ::-1]
            .copy()
            .transpose(2, 0, 1)[None]
        )
        out, mask = self._get_session().run(
            None, {"source": emb, "target": t}
        )
        out_bgr = (
            ((out[0].transpose(1, 2, 0) + 1) / 2 * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )[:, :, ::-1].copy()
        m = (mask[0, 0] * 255).clip(0, 255).astype(np.uint8)
        Mi = cv2.invertAffineTransform(M)
        of = cv2.warpAffine(out_bgr, Mi, (w, h), flags=cv2.INTER_LINEAR)
        mf = (
            cv2.warpAffine(m, Mi, (w, h), flags=cv2.INTER_LINEAR)
            .astype(np.float32)[:, :, None]
            / 255
        )
        swapped = (of * mf + bgr * (1 - mf)).clip(0, 255).astype(np.uint8)

        if full_frame_upscale:
            # Portrait path: restore + 2x upscale the whole frame (face
            # dominates it), e.g. 512 -> 1024 for the PSHuman head carve.
            _, _, restored = self._get_restorer().enhance(
                swapped, has_aligned=False, only_center_face=True,
                paste_back=True, weight=0.5,
            )
            if restored is None:
                raise RuntimeError("GFPGAN returned no restored frame")
            return np.ascontiguousarray(restored[:, :, ::-1])

        # GFPGAN restore on the padded face crop (same bbox, weight 0.5).
        b = face.bbox.astype(int)
        pad = 0.35
        bw, bh = b[2] - b[0], b[3] - b[1]
        x1 = max(0, b[0] - int(bw * pad))
        y1 = max(0, b[1] - int(bh * pad))
        x2 = min(w, b[2] + int(bw * pad))
        y2 = min(h, b[3] + int(bh * pad))
        crop = swapped[y1:y2, x1:x2]
        _, _, restored = self._get_restorer().enhance(
            crop, has_aligned=False, only_center_face=True,
            paste_back=True, weight=0.5,
        )
        if restored is not None:
            ch, cw = y2 - y1, x2 - x1
            if restored.shape[:2] != (ch, cw):
                restored = cv2.resize(
                    restored, (cw, ch), interpolation=cv2.INTER_LANCZOS4
                )
            swapped[y1:y2, x1:x2] = restored

        return np.ascontiguousarray(swapped[:, :, ::-1])  # BGR -> RGB
