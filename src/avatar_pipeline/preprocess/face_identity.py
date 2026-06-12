"""Reusable insightface identity extraction and similarity scoring.

One detector pipeline serves three call sites: the T-pose reference gate
(candid vs candidate references), the --from-candid identity extraction,
and the A/B comparison harness (candid vs final avatar render).

Detection mirrors FaceAnalysis.get() but sweeps the SCRFD detect size and
the image rotation:
  - get() freezes the 640 canvas, and faces that fill the frame (tight
    face crops) exceed its anchor scales after resize — they only detect
    at smaller input sizes. detect(input_size=...) is the
    upstream-supported per-call override.
  - candid photos are frequently rotated 90/180/270 deg (EXIF stripped);
    SCRFD does not detect sideways faces, so each rotation is tried and
    the upright orientation is used for landmarks/embedding/crops.
"""
from __future__ import annotations

import numpy as np

_DET_SIZES = (640, 320, 160)


class FaceIdentityExtractor:
    def __init__(self) -> None:
        self._app = None

    def _get_app(self):
        if self._app is None:
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=0, det_size=(640, 640))
        return self._app

    def detect(self, image_rgb: np.ndarray):
        """Largest face in a uint8 RGB image, or None.

        Returns (face, upright_bgr): the insightface Face with landmarks and
        recognition embedding populated, and the BGR image in the rotation
        the face was found at (insightface operates on BGR throughout).
        """
        from insightface.app.common import Face

        app = self._get_app()
        bgr = np.ascontiguousarray(image_rgb[:, :, :3][:, :, ::-1])
        for quarter_turns in (0, 1, 2, 3):
            candidate = np.ascontiguousarray(np.rot90(bgr, k=quarter_turns))
            for det_size in _DET_SIZES:
                bboxes, kpss = app.det_model.detect(
                    candidate,
                    input_size=(det_size, det_size),
                    max_num=0,
                    metric="default",
                )
                if not bboxes.shape[0]:
                    continue
                if quarter_turns:
                    print(
                        f"  face found after rotating the input "
                        f"{quarter_turns * 90} deg counter-clockwise"
                    )
                # Candid photos may contain bystanders: take the largest face.
                areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                i = int(np.argmax(areas))
                face = Face(
                    bbox=bboxes[i, 0:4],
                    kps=kpss[i] if kpss is not None else None,
                    det_score=bboxes[i, 4],
                )
                for taskname, model in app.models.items():
                    if taskname == "detection":
                        continue
                    model.get(candidate, face)
                return face, candidate
        return None

    def embed(self, image_rgb: np.ndarray) -> np.ndarray | None:
        """Normalized FaceID embedding of the largest face, or None."""
        hit = self.detect(image_rgb)
        if hit is None:
            return None
        face, _ = hit
        return np.asarray(face.normed_embedding, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def identity_similarity(
    image_a_rgb: np.ndarray,
    image_b_rgb: np.ndarray,
    extractor: FaceIdentityExtractor | None = None,
) -> float | None:
    """FaceID cosine similarity between the largest faces of two RGB images.

    None when either image has no detectable face. Same-person scores
    typically land around 0.4-0.7 at decent resolution; small rendered
    faces score lower — treat thresholds per call site, not here.
    """
    extractor = extractor or FaceIdentityExtractor()
    ea = extractor.embed(image_a_rgb)
    eb = extractor.embed(image_b_rgb)
    if ea is None or eb is None:
        return None
    return cosine_similarity(ea, eb)
