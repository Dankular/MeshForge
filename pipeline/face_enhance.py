"""
Face enhancement for MV-Adapter multiview textures.

Pipeline per visible-face view:
  1. InsightFace buffalo_l  — detect faces, extract 5-pt landmarks & 512-d embeddings
  2. HyperSwap 1A 256       — swap reference identity (embedding) onto each view face
     (falls back to inswapper_128 if hyperswap not present)
  3. GFPGAN v1.4            — restore + upscale face details

HyperSwap I/O:
    source  [1, 512]          — face embedding from recognition model
    target  [1, 3, 256, 256]  — aligned face crop (float32, RGB, [0,1])
    output  [1, 3, 256, 256]  — swapped face crop
    mask    [1, 1, 256, 256]  — alpha mask for seamless paste-back

Usage (standalone):
    python -m scripts.face_enhance \
        --multiview  /tmp/user_tex4/result.png \
        --reference  /tmp/tex_input_768.png \
        --output     /tmp/user_tex4/result_enhanced.png \
        --checkpoints /root/MV-Adapter/checkpoints
"""

import argparse
import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


# ── helpers ────────────────────────────────────────────────────────────────────

def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def split_multiview(mv: Image.Image, n: int = 6):
    w_each = mv.width // n
    return [mv.crop((i * w_each, 0, (i + 1) * w_each, mv.height)) for i in range(n)]


def stitch_views(views):
    total_w = sum(v.width for v in views)
    out = Image.new("RGB", (total_w, views[0].height))
    x = 0
    for v in views:
        out.paste(v, (x, 0))
        x += v.width
    return out


# ── HyperSwap 1A 256 — custom ONNX wrapper ────────────────────────────────────

class HyperSwapper:
    """
    Direct ONNX inference for HyperSwap 1A 256.
    source [1,512] × target [1,3,256,256] → output [1,3,256,256], mask [1,1,256,256]
    """

    # Standard 5-point face alignment template (112×112 base, scaled to crop_size)
    _TEMPLATE_112 = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)

    def __init__(self, ckpt_path: str, providers=None):
        self.crop_size = 256
        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(ckpt_path, providers=self.providers)
        print(f"[HyperSwapper] Loaded {os.path.basename(ckpt_path)} "
              f"(providers: {self.sess.get_providers()})")

    def _get_affine(self, kps: np.ndarray) -> np.ndarray:
        """Estimate affine transform from 5 face keypoints to standard template."""
        template = self._TEMPLATE_112 / 112.0 * self.crop_size
        from cv2 import estimateAffinePartial2D
        M, _ = estimateAffinePartial2D(kps, template, method=cv2.RANSAC)
        return M  # [2, 3]

    def _crop_face(self, img_bgr: np.ndarray, kps: np.ndarray):
        """Crop and align face to crop_size × crop_size."""
        M = self._get_affine(kps)
        crop = cv2.warpAffine(img_bgr, M, (self.crop_size, self.crop_size),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return crop, M

    def _paste_back(self, img_bgr: np.ndarray, crop_bgr: np.ndarray,
                    mask: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Paste swapped face crop back into the original frame using the mask."""
        h, w = img_bgr.shape[:2]
        IM = cv2.invertAffineTransform(M)

        # Warp swapped crop and mask back to original image space
        warped = cv2.warpAffine(crop_bgr, IM, (w, h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # mask is [256,256] float32 [0,1]
        mask_img = (mask * 255).clip(0, 255).astype(np.uint8)
        mask_warped = cv2.warpAffine(mask_img, IM, (w, h), flags=cv2.INTER_LINEAR)
        mask_f = mask_warped.astype(np.float32)[:, :, np.newaxis] / 255.0

        result = img_bgr.astype(np.float32) * (1.0 - mask_f) + warped.astype(np.float32) * mask_f
        return result.clip(0, 255).astype(np.uint8)

    def get(self, img_bgr: np.ndarray, target_face, source_face,
            paste_back: bool = True):
        """
        Swap source_face identity onto target_face in img_bgr.
        face objects are InsightFace Face instances with .embedding and .kps.
        """
        # 1. Source embedding [1, 512]
        emb = source_face.embedding.astype(np.float32)
        emb /= np.linalg.norm(emb)          # L2-normalise
        source_input = emb.reshape(1, -1)   # [1, 512]

        # 2. Crop and align target face to 256×256
        kps = target_face.kps.astype(np.float32)
        crop_bgr, M = self._crop_face(img_bgr, kps)

        # Convert BGR→RGB, normalize to [-1, 1], HWC→CHW, add batch dim
        crop_rgb = crop_bgr[:, :, ::-1].astype(np.float32) / 255.0
        crop_rgb = (crop_rgb - 0.5) / 0.5                         # [−1, 1]
        target_input = crop_rgb.transpose(2, 0, 1)[np.newaxis]   # [1, 3, 256, 256]

        # 3. Inference
        outputs = self.sess.run(None, {"source": source_input, "target": target_input})
        out_tensor = outputs[0][0]   # [3, 256, 256]  values in [-1, 1]
        mask_tensor = outputs[1][0, 0]  # [256, 256]

        # 4. Convert output back to BGR uint8  ([-1,1] → [0,255])
        out_rgb = ((out_tensor.transpose(1, 2, 0) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        out_bgr = out_rgb[:, :, ::-1]

        if not paste_back:
            return out_bgr, mask_tensor

        # 5. Paste back into the original frame
        return self._paste_back(img_bgr, out_bgr, mask_tensor, M)


# ── model loading ─────────────────────────────────────────────────────────────

_ORT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def load_face_analyzer():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=_ORT_PROVIDERS)
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def load_swapper(ckpt_dir: str):
    """HyperSwap 1A 256 if present, else fall back to inswapper_128."""
    import insightface.model_zoo as model_zoo

    hyperswap = os.path.join(ckpt_dir, "hyperswap_1a_256.onnx")
    inswapper = os.path.join(ckpt_dir, "inswapper_128.onnx")

    if os.path.exists(hyperswap):
        print(f"[face_enhance] Using HyperSwap 1A 256")
        return HyperSwapper(hyperswap, providers=_ORT_PROVIDERS)

    if os.path.exists(inswapper):
        print(f"[face_enhance] Using inswapper_128 (fallback)")
        return model_zoo.get_model(inswapper, providers=_ORT_PROVIDERS)

    raise FileNotFoundError(
        f"No swapper model found in {ckpt_dir}. "
        "Add hyperswap_1a_256.onnx or inswapper_128.onnx."
    )


def load_gfpgan(ckpt_dir: str, upscale: int = 2):
    from gfpgan import GFPGANer
    model_path = os.path.join(ckpt_dir, "GFPGANv1.4.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GFPGANv1.4.pth not found in {ckpt_dir}")
    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )
    return restorer


# ── core enhancement ──────────────────────────────────────────────────────────

def get_reference_face(analyzer, ref_bgr: np.ndarray):
    faces = analyzer.get(ref_bgr)
    if not faces:
        raise RuntimeError("No face detected in reference image.")
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0]


def _gfpgan_face_only(frame_bgr, bbox, gfpgan_restorer, pad: float = 0.35) -> np.ndarray:
    """Run GFPGAN on a padded face crop and paste the result back. Body is untouched."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox[:4].astype(int)
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    cx1 = max(0, x1 - px);  cy1 = max(0, y1 - py)
    cx2 = min(w, x2 + px);  cy2 = min(h, y2 + py)

    crop = frame_bgr[cy1:cy2, cx1:cx2].copy()
    try:
        _, _, restored_crop = gfpgan_restorer.enhance(
            crop,
            has_aligned=False,
            only_center_face=True,
            paste_back=True,
            weight=0.5,
        )
    except Exception as e:
        import traceback as _tb
        print(f"[enhance_view] GFPGAN failed on face crop: {e}\n{_tb.format_exc()}")
        return frame_bgr

    # Resize back to original crop size (GFPGAN may upscale)
    crop_h, crop_w = cy2 - cy1, cx2 - cx1
    if restored_crop.shape[:2] != (crop_h, crop_w):
        restored_crop = cv2.resize(restored_crop, (crop_w, crop_h),
                                   interpolation=cv2.INTER_LANCZOS4)

    result = frame_bgr.copy()
    result[cy1:cy2, cx1:cx2] = restored_crop
    return result


def enhance_view(view_bgr, analyzer, swapper, gfpgan_restorer, source_face,
                 gfpgan_upscale: int = 2) -> np.ndarray:
    target_faces = analyzer.get(view_bgr)
    if not target_faces:
        return view_bgr

    swapped = view_bgr.copy()
    for face in target_faces:
        swapped = swapper.get(swapped, face, source_face, paste_back=True)
    print(f"[enhance_view] HyperSwap applied to {len(target_faces)} face(s)")

    # Apply GFPGAN to each face bbox only — body is never touched
    result = swapped
    for i, face in enumerate(target_faces):
        result = _gfpgan_face_only(result, face.bbox, gfpgan_restorer)
        print(f"[enhance_view] GFPGAN restored face {i}")

    return result


def enhance_multiview(
    multiview_path: str,
    reference_path: str,
    output_path: str,
    ckpt_dir: str,
    n_views: int = 6,
    gfpgan_upscale: int = 2,
    face_views: tuple = (0, 1, 3, 4),
):
    print("[face_enhance] Loading models...")
    analyzer = load_face_analyzer()
    swapper = load_swapper(ckpt_dir)
    gfpgan_restorer = load_gfpgan(ckpt_dir, upscale=gfpgan_upscale)
    print("[face_enhance] Models loaded.")

    ref_pil = Image.open(reference_path).convert("RGB")
    ref_bgr = pil_to_bgr(ref_pil)
    source_face = get_reference_face(analyzer, ref_bgr)
    print(f"[face_enhance] Reference face bbox={source_face.bbox.astype(int)}")

    mv = Image.open(multiview_path).convert("RGB")
    views = split_multiview(mv, n=n_views)
    enhanced = []

    for i, view_pil in enumerate(views):
        if i in face_views:
            view_bgr = pil_to_bgr(view_pil)
            result_bgr = enhance_view(view_bgr, analyzer, swapper, gfpgan_restorer,
                                      source_face, gfpgan_upscale=gfpgan_upscale)
            enhanced.append(bgr_to_pil(result_bgr))
            n_faces = len(analyzer.get(view_bgr))
            print(f"[face_enhance] View {i}: {n_faces} face(s) processed.")
        else:
            enhanced.append(view_pil)

    stitch_views(enhanced).save(output_path)
    print(f"[face_enhance] Saved → {output_path}")
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiview",   required=True)
    parser.add_argument("--reference",   required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--checkpoints", default="./checkpoints")
    parser.add_argument("--n_views",     type=int, default=6)
    parser.add_argument("--upscale",     type=int, default=2)
    args = parser.parse_args()

    enhance_multiview(
        multiview_path=args.multiview,
        reference_path=args.reference,
        output_path=args.output,
        ckpt_dir=args.checkpoints,
        n_views=args.n_views,
        gfpgan_upscale=args.upscale,
    )
