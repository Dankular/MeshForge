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
# 18-keypoint COCO body in OpenPose order, normalized (x, y) on a 640x768
# canvas. Anatomical T-pose proportions: wrist-to-wrist span ~0.84 of
# standing height (fingertip span ~= height once hands extend past the
# wrists). The earlier 512-wide canvas could not fit a proportional
# wingspan, so the skeleton encoded T-rex arms and ControlNet faithfully
# reproduced them — and the model's attempts to draw anatomically longer
# arms past the too-short wrist marks were the original fingertip clipping.
_TPOSE_KEYPOINTS_NORM: tuple[tuple[float, float], ...] = (
    (0.500, 0.130),  # 0  nose
    (0.500, 0.230),  # 1  neck
    (0.400, 0.230),  # 2  R shoulder
    (0.252, 0.230),  # 3  R elbow
    (0.105, 0.230),  # 4  R wrist
    (0.600, 0.230),  # 5  L shoulder
    (0.748, 0.230),  # 6  L elbow
    (0.895, 0.230),  # 7  L wrist
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


# ── T-pose validity scoring (pure; testable without models) ─────────────────

def tpose_pose_score(
    keypoints: np.ndarray,
    joint_names: list[str],
    min_conf: float = 0.3,
    arm_tolerance: float = 0.15,
) -> dict:
    """Score COCO-17 pixel keypoints for strict-T-pose validity.

    The defining property: the whole arm chain (elbows + wrists) sits on the
    shoulder line. arm_dev is the worst arm-joint deviation from that line,
    normalized by torso length; spread requires wrists wider apart than the
    shoulders; span_ratio requires an anatomical wingspan (wrist-to-wrist
    ~0.75-1.05 of nose-to-ankle height — catches T-rex arms that are
    perfectly horizontal but half the proper length); upright requires hips
    below shoulders (image y grows down).
    """
    idx = {name: i for i, name in enumerate(joint_names)}

    def kp(name: str) -> np.ndarray:
        return keypoints[idx[name]]

    invalid = {"valid": False, "conf_ok": False, "arm_dev": float("inf"),
               "spread": False, "upright": False, "span_ratio": 0.0}
    required = (
        "nose", "left_shoulder", "right_shoulder", "left_elbow",
        "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_ankle", "right_ankle",
    )
    if not all(float(kp(n)[2]) >= min_conf for n in required):
        return invalid

    shoulder_y = (float(kp("left_shoulder")[1]) + float(kp("right_shoulder")[1])) / 2
    hip_y = (float(kp("left_hip")[1]) + float(kp("right_hip")[1])) / 2
    ankle_y = (float(kp("left_ankle")[1]) + float(kp("right_ankle")[1])) / 2
    torso = abs(hip_y - shoulder_y)
    stature = abs(ankle_y - float(kp("nose")[1]))
    if torso < 1.0 or stature < 1.0:
        return {**invalid, "conf_ok": True}

    arm_dev = max(
        abs(float(kp(n)[1]) - shoulder_y)
        for n in ("left_elbow", "right_elbow", "left_wrist", "right_wrist")
    ) / torso
    wrist_span = abs(float(kp("left_wrist")[0]) - float(kp("right_wrist")[0]))
    shoulder_span = abs(
        float(kp("left_shoulder")[0]) - float(kp("right_shoulder")[0])
    )
    spread = wrist_span > 2.0 * shoulder_span
    # nose-to-ankle underestimates full stature by the head/foot caps;
    # wrist span underestimates fingertip span by the hands — the ratio of
    # the two truncated spans still lands near 0.9 for real anatomy.
    span_ratio = wrist_span / stature
    span_ok = 0.75 <= span_ratio <= 1.05
    upright = hip_y > shoulder_y

    return {
        "valid": arm_dev <= arm_tolerance and spread and span_ok and upright,
        "conf_ok": True,
        "arm_dev": float(arm_dev),
        "spread": bool(spread),
        "span_ratio": float(span_ratio),
        "upright": bool(upright),
    }


# ── Generator ────────────────────────────────────────────────────────────────

class TPoseReferenceGenerator:
    """Candid photo → front-facing T-pose reference PNG."""

    base_model = "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    controlnet_model = "lllyasviel/control_v11p_sd15_openpose"

    width = 640   # wide enough for an anatomical wingspan (see skeleton)
    height = 768
    # Perspective-neutral close-up for the PSHuman head carve: a candid is a
    # single warped view (selfie-distance perspective bakes a distorted face
    # shape); FaceID re-synthesizes the identity at portrait framing with
    # telephoto perspective, and PSHuman's 7-view diffusion + SMPL-X carve
    # then supplies the missing views/depth for corrected face geometry.
    portrait_size = 512
    portrait_prompt_template = (
        "frontal head and shoulders studio portrait photo of {subject}, "
        "looking directly at the camera, neutral expression, hair fully "
        "visible, even soft lighting, plain light gray background, "
        "85mm lens, sharp focus, high detail skin texture"
    )
    portrait_negative_prompt = (
        "monochrome, lowres, bad anatomy, worst quality, low quality, "
        "blurry, tilted head, profile, side view, sunglasses, hat, cropped"
    )
    # {subject} is filled from the identity phase (insightface genderage —
    # "a woman" / "a man", falling back to "a person"). With no_clothes the
    # {clothing} fragment is simply omitted and every garment term moves to
    # the negative — the prompt is otherwise untouched.
    prompt_template = (
        "full body photo of {subject} standing upright in a T-pose, arms "
        "held straight out horizontally to the sides, fingers extended, "
        "legs straight, feet slightly apart, facing the camera, neutral "
        "expression, {clothing}plain light gray studio "
        "background, even soft lighting, sharp focus, whole body in frame"
    )
    clothing_default = "casual fitted clothing, "
    negative_prompt_base = (
        "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, "
        "cropped, out of frame, arms down, arms bent, hands on hips, sitting"
    )
    negative_no_clothes = (
        ", underwear, briefs, bra, panties, clothing, garment, fabric, "
        "cloth, dress, skirt, shorts, shirt, trousers, shoes"
    )

    def __init__(
        self,
        num_inference_steps: int = 30,
        seed: int = 2023,
        checkpoint_root: Path | None = None,
        no_clothes: bool = False,
    ) -> None:
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.checkpoint_root = checkpoint_root or (_REPO_ROOT / "checkpoints")
        self.no_clothes = no_clothes
        self.subject = "a person"  # refined by the identity phase
        self._ip_model = None
        self._face_app = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def prompt(self) -> str:
        return self.prompt_template.format(
            subject=self.subject,
            clothing="" if self.no_clothes else self.clothing_default,
        )

    @property
    def negative_prompt(self) -> str:
        if self.no_clothes:
            return self.negative_prompt_base + self.negative_no_clothes
        return self.negative_prompt_base

    @property
    def portrait_prompt(self) -> str:
        return self.portrait_prompt_template.format(subject=self.subject)

    # ── identity extraction ──────────────────────────────────────────────
    def _get_extractor(self):
        if self._face_app is None:
            from avatar_pipeline.preprocess.face_identity import (
                FaceIdentityExtractor,
            )

            self._face_app = FaceIdentityExtractor()
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

        hit = self._get_extractor().detect(image_rgb)
        if hit is None:
            raise RuntimeError(
                "insightface found no face in the candid input (tried all "
                "four rotations); cannot build a T-pose reference without "
                "a face identity"
            )
        face, upright_bgr = hit
        # The identity phase also fixes the prompt subject: buffalo_l's
        # genderage head already ran inside detect().
        sex = getattr(face, "sex", None)
        self.subject = {"F": "a woman", "M": "a man"}.get(sex, "a person")
        print(
            f"  identity: sex={sex} age={getattr(face, 'age', '?')} -> "
            f"prompts use {self.subject!r}"
        )
        faceid_embeds = torch.from_numpy(face.normed_embedding).unsqueeze(0)
        face_image = face_align.norm_crop(
            upright_bgr, landmark=face.kps, image_size=224
        )
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
    def generate_one(
        self,
        faceid_embeds: torch.Tensor,
        face_image: np.ndarray,
        pose_image: Image.Image,
        seed: int,
    ) -> Image.Image:
        if self._ip_model is None:
            self.load_pretrained()
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
            seed=seed,
            image=pose_image,  # ControlNet conditioning, forwarded via kwargs
        )
        return images[0]

    def generate_portrait_one(
        self,
        faceid_embeds: torch.Tensor,
        face_image: np.ndarray,
        seed: int,
    ) -> Image.Image:
        """Frontal close-up portrait (no pose control: black control image
        at conditioning scale 0 turns the ControlNet pipe into plain SD15)."""
        if self._ip_model is None:
            self.load_pretrained()
        blank_control = Image.new(
            "RGB", (self.portrait_size, self.portrait_size), (0, 0, 0)
        )
        images = self._ip_model.generate(
            prompt=self.portrait_prompt,
            negative_prompt=self.portrait_negative_prompt,
            face_image=face_image,
            faceid_embeds=faceid_embeds,
            shortcut=True,
            s_scale=1.0,
            num_samples=1,
            width=self.portrait_size,
            height=self.portrait_size,
            num_inference_steps=self.num_inference_steps,
            seed=seed,
            image=blank_control,
            controlnet_conditioning_scale=0.0,
        )
        return images[0]

    def generate(self, input_image: str, out_path: Path) -> Path:
        """Single-shot generation at the configured seed (no gate)."""
        image = np.array(Image.open(input_image).convert("RGB"), dtype=np.uint8)
        faceid_embeds, face_image = self.extract_face_identity(image)
        pose_image = render_tpose_skeleton(self.width, self.height)
        result = self.generate_one(faceid_embeds, face_image, pose_image, self.seed)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)
        return out_path

    def unload(self) -> None:
        """Release GPU memory before the main pipeline loads its models."""
        self._ip_model = None
        self._face_app = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_tpose_reference(
    input_image: str,
    out_path: Path,
    seed: int = 2023,
    num_candidates: int = 4,
    min_identity: float = 0.15,
    portrait_out: Path | None = None,
    min_portrait_identity: float = 0.4,
    no_clothes: bool = False,
) -> Path:
    """Gated best-of-N reference synthesis.

    Generates *num_candidates* references at sequential seeds, scores each
    for strict-T-pose validity (Sapiens pose keypoints — a failed ControlNet
    pose surfaces here in seconds, not 40 minutes into the pipeline) and
    FaceID similarity to the candid input, then keeps the highest-identity
    pose-valid candidate. Fails loudly with the full score table when no
    candidate passes. All candidates are kept next to *out_path* for
    inspection.
    """
    out_path = Path(out_path)
    cand_dir = out_path.parent / "tpose_candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    generator = TPoseReferenceGenerator(seed=seed, no_clothes=no_clothes)
    try:
        image = np.array(Image.open(input_image).convert("RGB"), dtype=np.uint8)
        faceid_embeds, face_image = generator.extract_face_identity(image)
        candid_embed = faceid_embeds[0].cpu().numpy()
        pose_image = render_tpose_skeleton(generator.width, generator.height)

        candidates: list[tuple[int, Image.Image, Path]] = []
        for i in range(num_candidates):
            cand_seed = seed + i
            img = generator.generate_one(
                faceid_embeds, face_image, pose_image, cand_seed
            )
            path = cand_dir / f"seed_{cand_seed}.png"
            img.save(path)
            candidates.append((cand_seed, img, path))

        # Perspective-neutral portrait (same SD load) for the PSHuman head
        # carve — the candid's single warped view never feeds geometry.
        portrait_img = None
        if portrait_out is not None:
            portrait_img = generator.generate_portrait_one(
                faceid_embeds, face_image, seed
            )

        # Identity scores while insightface is still loaded.
        from avatar_pipeline.preprocess.face_identity import cosine_similarity

        extractor = generator._get_extractor()
        id_scores: list[float | None] = []
        for _, img, _ in candidates:
            emb = extractor.embed(np.asarray(img))
            id_scores.append(
                None if emb is None else cosine_similarity(candid_embed, emb)
            )
    finally:
        generator.unload()

    # Pose gate after the SD stack is freed (Sapiens 1b needs the VRAM).
    from avatar_pipeline.sapiens.pose_estimation import PoseEstimator

    pose_estimator = PoseEstimator()
    pose_estimator.load_pretrained(None)
    if torch.cuda.is_available():
        # Standalone use: no mmgp domain manages this model, so it stays on
        # CPU after loading while estimate() sends inputs to CUDA.
        pose_estimator._model = pose_estimator._model.cuda()
    pose_scores = []
    for _, img, _ in candidates:
        pose_data = pose_estimator.estimate(np.asarray(img, dtype=np.uint8))
        pose_scores.append(
            tpose_pose_score(pose_data.keypoints, list(pose_data.joint_names))
        )
    del pose_estimator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows = []
    best = None  # (identity, index)
    for i, ((cand_seed, _, path), id_score, ps) in enumerate(
        zip(candidates, id_scores, pose_scores)
    ):
        ok = ps["valid"] and id_score is not None and id_score >= min_identity
        rows.append(
            f"  seed {cand_seed}: pose_valid={ps['valid']} "
            f"arm_dev={ps['arm_dev']:.3f} identity="
            f"{'n/a' if id_score is None else f'{id_score:.3f}'}"
            f"{'  <- candidate' if ok else ''}"
        )
        if ok and (best is None or id_score > best[0]):
            best = (id_score, i)
    print("[tpose-gate] candidate scores:")
    for row in rows:
        print(row)
    if best is None:
        raise RuntimeError(
            "No T-pose reference candidate passed the gate "
            f"(pose-valid + identity >= {min_identity}):\n" + "\n".join(rows)
        )

    _, idx = best
    win_seed, win_img, _ = candidates[idx]
    print(
        f"[tpose-gate] selected seed {win_seed} "
        f"(identity {best[0]:.3f}, arm_dev {pose_scores[idx]['arm_dev']:.3f})"
    )

    # Identity enhancement: FaceID conditioning cannot impose identity on a
    # ~50px face (the gate winners measure ~0.1 cosine); transplant the true
    # candid face (HyperSwap) and restore it (GFPGAN) before the pipeline
    # ever sees the reference.
    from avatar_pipeline.preprocess.face_identity import (
        FaceIdentityExtractor,
        cosine_similarity,
    )
    from avatar_pipeline.preprocess.face_swap import IdentityEnhancer

    extractor = FaceIdentityExtractor()
    enhancer = IdentityEnhancer()
    enhanced = enhancer.enhance(
        np.asarray(win_img, dtype=np.uint8), candid_embed, extractor
    )
    post_emb = extractor.embed(enhanced)
    post_sim = (
        None if post_emb is None
        else cosine_similarity(candid_embed, post_emb)
    )
    print(
        f"[tpose-gate] identity after swap+restore: "
        f"{'n/a' if post_sim is None else f'{post_sim:.3f}'} "
        f"(was {best[0]:.3f})"
    )
    if post_sim is None or post_sim < best[0]:
        raise RuntimeError(
            "identity enhancement degraded or lost the face "
            f"(post-swap similarity {post_sim}); refusing to continue"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(enhanced).save(out_path)

    if portrait_img is not None:
        portrait = enhancer.enhance(
            np.asarray(portrait_img, dtype=np.uint8),
            candid_embed,
            extractor,
            full_frame_upscale=True,
        )
        p_emb = extractor.embed(portrait)
        p_sim = (
            None if p_emb is None else cosine_similarity(candid_embed, p_emb)
        )
        print(
            f"[tpose-gate] portrait identity after swap+restore: "
            f"{'n/a' if p_sim is None else f'{p_sim:.3f}'}"
        )
        if p_sim is None or p_sim < min_portrait_identity:
            raise RuntimeError(
                f"face portrait identity {p_sim} below "
                f"{min_portrait_identity}; refusing to feed it to the "
                "head carve"
            )
        portrait_out = Path(portrait_out)
        portrait_out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(portrait).save(portrait_out)
        print(f"[tpose-gate] face portrait written to: {portrait_out}")

    return out_path
