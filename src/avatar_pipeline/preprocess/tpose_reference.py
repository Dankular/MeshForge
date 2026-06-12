"""T-pose reference synthesis from a candid photo — IP-Adapter FaceID PlusV2.

Builds the front-facing T-pose reference image the rest of the pipeline
expects when the user only has a candid face (or partial face+body) photo.

Identity comes from IP-Adapter-FaceID-PlusV2 (h94/IP-Adapter-FaceID), run
through the official tencent-ailab/IP-Adapter code in external/IP-Adapter
using the model card's exact recipe: insightface buffalo_l for the ID embed
and aligned 224px face crop, Realistic Vision 4.0 base, sd-vae-ft-mse,
CLIP ViT-H image encoder, DDIM, shortcut=True (the v2 path).

The T-pose itself is enforced with the OpenPose ControlNet
(lllyasviel/control_v11p_sd15_openpose) conditioned on a synthetic T-pose
skeleton rendered below — FaceID carries identity only, and a text prompt
alone does not reliably hold a strict T-pose. IPAdapterFaceIDPlus.generate()
forwards unknown kwargs to the wrapped pipe, so the ControlNet pipeline slots
in without touching the upstream code.

This stage runs standalone *before* AvatarPipeline is constructed and frees
its models afterwards; it is deliberately not part of the shared mmgp domain.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[3]
IP_ADAPTER_REPO = _REPO_ROOT / "external" / "IP-Adapter"

IP_CKPT_NAME = "ip_adapter_faceid_plusv2"
IP_CKPT_RELPATH = "ip_adapter/ip-adapter-faceid-plusv2_sd15.bin"


def _bootstrap_ip_adapter_imports() -> None:
    if str(IP_ADAPTER_REPO) not in sys.path:
        sys.path.insert(0, str(IP_ADAPTER_REPO))


# ── Synthetic OpenPose T-pose skeleton ───────────────────────────────────────
#
# 18-keypoint COCO body in OpenPose order, normalized (x, y). Arms straight
# out at shoulder height (the defining T-pose property), legs straight and
# slightly apart, figure centered with headroom and foot margin.
_TPOSE_KEYPOINTS_NORM: tuple[tuple[float, float], ...] = (
    (0.500, 0.130),  # 0  nose
    (0.500, 0.230),  # 1  neck
    (0.420, 0.230),  # 2  R shoulder
    (0.300, 0.230),  # 3  R elbow
    (0.180, 0.230),  # 4  R wrist
    (0.580, 0.230),  # 5  L shoulder
    (0.700, 0.230),  # 6  L elbow
    (0.820, 0.230),  # 7  L wrist
    (0.455, 0.510),  # 8  R hip
    (0.448, 0.700),  # 9  R knee
    (0.442, 0.880),  # 10 R ankle
    (0.545, 0.510),  # 11 L hip
    (0.552, 0.700),  # 12 L knee
    (0.558, 0.880),  # 13 L ankle
    (0.485, 0.112),  # 14 R eye
    (0.515, 0.112),  # 15 L eye
    (0.468, 0.126),  # 16 R ear
    (0.532, 0.126),  # 17 L ear
)

# Limb sequence and palette exactly as ControlNet's annotator/openpose/util.py
# draw_bodypose — control_v11p_sd15_openpose was trained on this rendering.
_LIMB_SEQ = (
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
)
_POSE_COLORS = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
)


def tpose_keypoints(width: int, height: int) -> np.ndarray:
    """(18, 2) float32 pixel coordinates of the T-pose skeleton."""
    pts = np.array(_TPOSE_KEYPOINTS_NORM, dtype=np.float32)
    pts[:, 0] *= width
    pts[:, 1] *= height
    return pts


def render_tpose_skeleton(width: int, height: int) -> Image.Image:
    """OpenPose-style T-pose conditioning image (ControlNet draw_bodypose:
    stickwidth-4 ellipse limbs at 0.6 alpha, radius-4 joint circles)."""
    import cv2

    pts = tpose_keypoints(width, height)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    stickwidth = 4
    for limb, color in zip(_LIMB_SEQ, _POSE_COLORS):
        (x0, y0), (x1, y1) = pts[limb[0]], pts[limb[1]]
        m_x, m_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        length = math.hypot(x1 - x0, y1 - y0)
        angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
        polygon = cv2.ellipse2Poly(
            (int(m_x), int(m_y)), (int(length / 2), stickwidth),
            int(angle), 0, 360, 1,
        )
        overlay = canvas.copy()
        cv2.fillConvexPoly(overlay, polygon, color)
        canvas = cv2.addWeighted(canvas, 0.4, overlay, 0.6, 0)
    for (x, y), color in zip(pts, _POSE_COLORS):
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)
    return Image.fromarray(canvas)


# ── Generator ────────────────────────────────────────────────────────────────

class TPoseReferenceGenerator:
    """Candid photo → front-facing T-pose reference PNG."""

    base_model = "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    controlnet_model = "lllyasviel/control_v11p_sd15_openpose"

    width = 512
    height = 768
    prompt = (
        "full body photo of a person standing upright in a T-pose, arms held "
        "straight out horizontally to the sides, fingers extended, legs "
        "straight, feet slightly apart, facing the camera, neutral "
        "expression, casual fitted clothing, plain light gray studio "
        "background, even soft lighting, sharp focus, whole body in frame"
    )
    negative_prompt = (
        "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, "
        "cropped, out of frame, arms down, arms bent, hands on hips, sitting"
    )

    def __init__(
        self,
        num_inference_steps: int = 30,
        seed: int = 2023,
        checkpoint_root: Path | None = None,
    ) -> None:
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.checkpoint_root = checkpoint_root or (_REPO_ROOT / "checkpoints")
        self._ip_model = None
        self._face_app = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── identity extraction ──────────────────────────────────────────────
    def _get_face_app(self):
        if self._face_app is None:
            from insightface.app import FaceAnalysis

            self._face_app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._face_app.prepare(ctx_id=0, det_size=(640, 640))
        return self._face_app

    def extract_face_identity(
        self, image_rgb: np.ndarray
    ) -> tuple[torch.Tensor, np.ndarray]:
        """FaceID embedding + aligned 224px face crop from a uint8 RGB image.

        insightface operates on BGR (the model card feeds cv2.imread output
        straight through, aligned crop included); the crop is therefore kept
        in BGR exactly as the official demo passes it to generate().
        Fails loudly when no face is detected — no fallback identity.
        """
        from insightface.utils import face_align

        bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        faces = self._get_face_app().get(bgr)
        if not faces:
            raise RuntimeError(
                "insightface found no face in the candid input; cannot build "
                "a T-pose reference without a face identity"
            )
        # Candid photos may contain bystanders: take the largest face.
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        faceid_embeds = torch.from_numpy(face.normed_embedding).unsqueeze(0)
        face_image = face_align.norm_crop(bgr, landmark=face.kps, image_size=224)
        return faceid_embeds, face_image

    # ── model loading ────────────────────────────────────────────────────
    def _resolve_ip_ckpt(self) -> Path:
        ckpt = self.checkpoint_root / IP_CKPT_RELPATH
        if not ckpt.exists():
            from avatar_pipeline.checkpoints import download_all

            print(f"  downloading {IP_CKPT_NAME} to {ckpt} ...")
            download_all(self.checkpoint_root, names=[IP_CKPT_NAME])
        return ckpt

    def load_pretrained(self) -> None:
        _bootstrap_ip_adapter_imports()
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            DDIMScheduler,
            StableDiffusionControlNetPipeline,
        )
        from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus

        ip_ckpt = self._resolve_ip_ckpt()

        # Model-card-exact components; only the pipeline class differs
        # (ControlNet variant) so the T-pose skeleton can steer the pose.
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained(self.vae_model).to(
            dtype=torch.float16
        )
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_model, torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.base_model,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None,
        )
        self._ip_model = IPAdapterFaceIDPlus(
            pipe, self.image_encoder_path, str(ip_ckpt), self._device
        )

    # ── generation ───────────────────────────────────────────────────────
    def generate(self, input_image: str, out_path: Path) -> Path:
        if self._ip_model is None:
            self.load_pretrained()

        image = np.array(Image.open(input_image).convert("RGB"), dtype=np.uint8)
        faceid_embeds, face_image = self.extract_face_identity(image)
        pose_image = render_tpose_skeleton(self.width, self.height)

        images = self._ip_model.generate(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            face_image=face_image,
            faceid_embeds=faceid_embeds,
            shortcut=True,  # PlusV2 ("v2") path per the model card
            s_scale=1.0,
            num_samples=1,
            width=self.width,
            height=self.height,
            num_inference_steps=self.num_inference_steps,
            seed=self.seed,
            image=pose_image,  # ControlNet conditioning, forwarded via kwargs
        )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(out_path)
        return out_path

    def unload(self) -> None:
        """Release GPU memory before the main pipeline loads its models."""
        self._ip_model = None
        self._face_app = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_tpose_reference(
    input_image: str, out_path: Path, seed: int = 2023
) -> Path:
    """One-shot helper: generate the reference, then free everything."""
    generator = TPoseReferenceGenerator(seed=seed)
    try:
        return generator.generate(input_image, out_path)
    finally:
        generator.unload()
