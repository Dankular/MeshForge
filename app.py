import sys
import os
import subprocess
import tempfile
import shutil
import traceback
import json
import random
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = REPO_DIR / "pipeline"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

try:
    from pipeline.enhance_surface import (
        run_stable_normal,
        run_depth_anything,
        bake_normal_into_glb,
        bake_depth_as_occlusion,
        unload_models,
    )
    import pipeline.enhance_surface as _enh_mod
except Exception:
    from enhance_surface import (
        run_stable_normal,
        run_depth_anything,
        bake_normal_into_glb,
        bake_depth_as_occlusion,
        unload_models,
    )
    import enhance_surface as _enh_mod

import cv2
import gradio as gr
import torch
import numpy as np
from PIL import Image

PYTHON = os.getenv("MESHFORGE_PYTHON", sys.executable)
TRIPOSG_DIR = os.getenv("MESHFORGE_TRIPOSG_DIR", str(REPO_DIR / "external" / "TripoSG"))
MVADAPTER_DIR = os.getenv(
    "MESHFORGE_MVADAPTER_DIR", str(REPO_DIR / "external" / "MV-Adapter")
)
CKPT_DIR = os.getenv("MESHFORGE_CKPT_DIR", str(Path(MVADAPTER_DIR) / "checkpoints"))
FIRERED_DIR = os.getenv(
    "MESHFORGE_FIRERED_DIR", str(REPO_DIR / "external" / "FireRed-Image-Edit")
)
TMP_DIR = Path(os.getenv("MESHFORGE_TMP_DIR", tempfile.gettempdir())) / "meshforge"
TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_CDN_BACKEND_ENABLED"] = "False"
os.environ["GRADIO_UPLOAD_CHUNK_SIZE"] = (
    "8388608"  # 8 MB chunks — fixes 504 timeout on gradio.live tunnel
)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True"  # reduces fragmentation for 17GB transformer + 5GB activations
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy-loaded models (kept in memory between calls)
_triposg_pipe = None
_rmbg_net = None
_last_glb_path = None
_hyperswap_sess = None
_gfpgan_restorer = None
_rmbg_version = None  # "2.0"
_firered_pipe = None
_init_seed = random.randint(0, 2**31 - 1)

import threading

_model_load_lock = threading.Lock()

ARCFACE_256 = (
    np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    * (256 / 112)
    + (256 - 112 * (256 / 112)) / 2
)

VIEW_NAMES = ["front", "3q_front", "side", "back", "3q_back"]
VIEW_PATHS = [str(TMP_DIR / f"render_{n}.png") for n in VIEW_NAMES]


def load_triposg():
    global _triposg_pipe, _rmbg_net, _rmbg_version
    if _triposg_pipe is not None:
        _triposg_pipe.to(DEVICE)
        if _rmbg_net is not None:
            _rmbg_net.to(DEVICE)
        return _triposg_pipe, _rmbg_net
    print("Loading TripoSG pipeline...")
    sys.path.insert(0, TRIPOSG_DIR)
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    from huggingface_hub import snapshot_download

    weights_path = snapshot_download("VAST-AI/TripoSG")
    _triposg_pipe = TripoSGPipeline.from_pretrained(
        weights_path, torch_dtype=torch.float16
    ).to(DEVICE)

    _load_rmbg()
    return _triposg_pipe, _rmbg_net


def load_gfpgan():
    global _gfpgan_restorer
    if _gfpgan_restorer is not None:
        return _gfpgan_restorer
    try:
        from gfpgan import GFPGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model_path = os.path.join(CKPT_DIR, "GFPGANv1.4.pth")
        if not os.path.exists(model_path):
            print(f"[GFPGAN] Not found at {model_path}")
            return None

        # RealESRGAN x2plus as background upsampler — upscales face crop 2x before GFPGAN
        realesrgan_path = os.path.join(CKPT_DIR, "RealESRGAN_x2plus.pth")
        bg_upsampler = None
        if os.path.exists(realesrgan_path):
            bg_model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path=realesrgan_path,
                model=bg_model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True,
            )
            print("[GFPGAN] RealESRGAN x2plus bg_upsampler loaded")
        else:
            print("[GFPGAN] RealESRGAN_x2plus.pth not found, running without upsampler")

        _gfpgan_restorer = GFPGANer(
            model_path=model_path,
            upscale=2,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=bg_upsampler,
        )
        print("[GFPGAN] Loaded GFPGANv1.4 (upscale=2 + RealESRGAN bg_upsampler)")
        return _gfpgan_restorer
    except Exception as e:
        print(f"[GFPGAN] Load failed: {e}")
        return None


def _load_rmbg():
    """Load RMBG-2.0 or fallback to RMBG-1.4."""
    global _rmbg_net, _rmbg_version
    if _rmbg_net is not None:
        return

    # Try RMBG-2.0 with transformers 5.x compatibility patches
    try:
        from transformers import AutoModelForImageSegmentation
        from transformers import PreTrainedModel as _PTM

        # Patch mark_tied_weights_as_initialized for transformers 5.x
        _orig_mark_tied = _PTM.mark_tied_weights_as_initialized

        def _safe_mark_tied(self, loading_info):
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = None
            return _orig_mark_tied(self, loading_info)

        _PTM.mark_tied_weights_as_initialized = _safe_mark_tied

        try:
            # Load with low_cpu_mem_usage=False to avoid meta device issues
            _rmbg_net = AutoModelForImageSegmentation.from_pretrained(
                "1038lab/RMBG-2.0",
                trust_remote_code=True,
                low_cpu_mem_usage=False,
                torch_dtype=torch.float32,
            )
            _rmbg_net.to(DEVICE).eval()
            _rmbg_version = "2.0"
            print("RMBG-2.0 loaded successfully.")
        finally:
            _PTM.mark_tied_weights_as_initialized = _orig_mark_tied

    except Exception as e:
        print(f"RMBG-2.0 load failed ({type(e).__name__}: {str(e)[:80]}...) - falling back to RMBG-1.4")
        _rmbg_net = None
        _rmbg_version = None

        # Fallback to RMBG-1.4
        try:
            from huggingface_hub import snapshot_download
            from external.TripoSG.scripts.briarmbg import BriaRMBG

            rmbg_weights_dir = snapshot_download("briaai/RMBG-1.4")
            _rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(DEVICE).eval()
            _rmbg_version = "1.4"
            print("RMBG-1.4 fallback loaded successfully.")
        except Exception as e2:
            _rmbg_net = None
            _rmbg_version = None
            print(f"RMBG-1.4 fallback failed ({type(e2).__name__}: {str(e2)[:80]}...) - background removal disabled.")


def load_rmbg_only():
    """Load RMBG standalone without loading TripoSG."""
    _load_rmbg()
    return _rmbg_net


def load_firered():
    """Lazy-load FireRed image-edit pipeline using GGUF-quantized transformer.

    Transformer: loaded from GGUF via from_single_file (Q4_K_M, ~12 GB on disk).
    Tries Arunk25/Qwen-Image-Edit-Rapid-AIO-GGUF first (fine-tuned, merged model).
    Falls back to unsloth/Qwen-Image-Edit-2511-GGUF (base model) if key mapping fails.

    text_encoder: 4-bit NF4 on GPU (~5.6 GB).
    GGUF transformer: dequantized on-the-fly, dispatched with 18 GiB GPU budget.
    Lightning scheduler: 4 steps, CFG 1.0 → ~1-2 min per inference.

    GPU budget: ~18 GB transformer + ~5.6 GB text_encoder + ~0.3 GB VAE ≈ 24 GB.
    """
    global _firered_pipe
    if _firered_pipe is not None:
        return _firered_pipe

    import math
    from diffusers import (
        QwenImageEditPlusPipeline,
        FlowMatchEulerDiscreteScheduler,
        GGUFQuantizationConfig,
    )
    from diffusers.models import QwenImageTransformer2DModel
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from accelerate import dispatch_model, infer_auto_device_map
    from huggingface_hub import hf_hub_download

    # Patch SDPA to cast K/V to match Q dtype.
    import torch.nn.functional as _F

    _orig_sdpa = _F.scaled_dot_product_attention

    def _dtype_safe_sdpa(query, key, value, *a, **kw):
        if key.dtype != query.dtype:
            key = key.to(query.dtype)
        if value.dtype != query.dtype:
            value = value.to(query.dtype)
        return _orig_sdpa(query, key, value, *a, **kw)

    _F.scaled_dot_product_attention = _dtype_safe_sdpa

    torch.cuda.empty_cache()

    # Load RMBG NOW — before dispatch_model creates meta tensors that poison later loads
    _load_rmbg()

    gguf_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)

    # ── Transformer: GGUF Q4_K_M — try fine-tuned Rapid-AIO first, fall back to base ──
    transformer = None

    # Attempt 1: Arunk25 Rapid-AIO GGUF (fine-tuned, fully merged, ~12.4 GB)
    try:
        print(
            "[FireRed] Downloading Arunk25/Qwen-Image-Edit-Rapid-AIO-GGUF Q4_K_M (~12 GB)..."
        )
        gguf_path = hf_hub_download(
            repo_id="Arunk25/Qwen-Image-Edit-Rapid-AIO-GGUF",
            filename="v23/Qwen-Rapid-AIO-NSFW-v23-Q4_K_M.gguf",
        )
        print("[FireRed] Loading Rapid-AIO transformer from GGUF...")
        transformer = QwenImageTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=gguf_config,
            torch_dtype=torch.bfloat16,
            config="Qwen/Qwen-Image-Edit-2511",
            subfolder="transformer",
        )
        print("[FireRed] Rapid-AIO GGUF transformer loaded OK.")
    except Exception as e:
        print(
            f"[FireRed] Rapid-AIO GGUF failed ({e}), falling back to unsloth base GGUF..."
        )
        transformer = None

    # Attempt 2: unsloth base GGUF Q4_K_M (~12.3 GB)
    if transformer is None:
        print(
            "[FireRed] Downloading unsloth/Qwen-Image-Edit-2511-GGUF Q4_K_M (~12 GB)..."
        )
        gguf_path = hf_hub_download(
            repo_id="unsloth/Qwen-Image-Edit-2511-GGUF",
            filename="qwen-image-edit-2511-Q4_K_M.gguf",
        )
        print("[FireRed] Loading base transformer from GGUF...")
        transformer = QwenImageTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=gguf_config,
            torch_dtype=torch.bfloat16,
            config="Qwen/Qwen-Image-Edit-2511",
            subfolder="transformer",
        )
        print("[FireRed] Base GGUF transformer loaded OK.")

    print("[FireRed] Dispatching transformer (18 GiB GPU, rest CPU)...")
    device_map = infer_auto_device_map(
        transformer,
        max_memory={0: "18GiB", "cpu": "90GiB"},
        dtype=torch.bfloat16,
    )
    n_gpu = sum(1 for d in device_map.values() if str(d) in ("0", "cuda", "cuda:0"))
    n_cpu = sum(1 for d in device_map.values() if str(d) == "cpu")
    print(f"[FireRed] Dispatched: {n_gpu} modules on GPU, {n_cpu} on CPU")
    transformer = dispatch_model(transformer, device_map=device_map)
    used_mb = torch.cuda.memory_allocated() // (1024**2)
    print(f"[FireRed] Transformer dispatched — VRAM: {used_mb} MB")

    # ── text_encoder: 4-bit NF4 on GPU (~5.6 GB) ──────────────────────────────
    bnb_enc = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print("[FireRed] Loading text_encoder (4-bit NF4)...")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        subfolder="text_encoder",
        quantization_config=bnb_enc,
        device_map="auto",
    )
    used_mb = torch.cuda.memory_allocated() // (1024**2)
    print(f"[FireRed] Text encoder loaded — VRAM: {used_mb} MB")

    # ── Pipeline: VAE + scheduler + processor + tokenizer ─────────────────────
    print("[FireRed] Loading pipeline...")
    _firered_pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
    )
    _firered_pipe.vae.to(DEVICE)

    # Lightning scheduler — 4 steps, use_dynamic_shifting, matches reference space config
    _firered_pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "time_shift_type": "exponential",
            "use_dynamic_shifting": True,
        }
    )

    used_mb = torch.cuda.memory_allocated() // (1024**2)
    print(f"[FireRed] Pipeline ready — total VRAM: {used_mb} MB")
    return _firered_pipe


def _gallery_to_pil_list(gallery_value):
    """Convert a Gradio Gallery value (list of various formats) to a list of PIL Images."""
    pil_images = []
    if not gallery_value:
        return pil_images
    for item in gallery_value:
        try:
            if isinstance(item, np.ndarray):
                pil_images.append(Image.fromarray(item).convert("RGB"))
                continue
            if isinstance(item, Image.Image):
                pil_images.append(item.convert("RGB"))
                continue
            # Gradio 6 Gallery returns dicts: {"image": FileData, "caption": ...}
            if isinstance(item, dict):
                img_data = item.get("image") or item
                if isinstance(img_data, dict):
                    path = (
                        img_data.get("path")
                        or img_data.get("url")
                        or img_data.get("name")
                    )
                else:
                    path = img_data
            elif isinstance(item, (list, tuple)):
                path = item[0]
            else:
                path = item
            if path and os.path.exists(str(path)):
                pil_images.append(Image.open(str(path)).convert("RGB"))
        except Exception as e:
            print(f"[FireRed] Could not load gallery image: {e}")
    return pil_images


def _firered_resize(img):
    """Resize to max 1024px maintaining aspect ratio, align dims to multiple of 8."""
    w, h = img.size
    if max(w, h) > 1024:
        if w > h:
            nw, nh = 1024, int(1024 * h / w)
        else:
            nw, nh = int(1024 * w / h), 1024
    else:
        nw, nh = w, h
    nw, nh = max(8, (nw // 8) * 8), max(8, (nh // 8) * 8)
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.LANCZOS)
    return img


_FIRERED_NEGATIVE = (
    "worst quality, low quality, bad anatomy, bad hands, text, error, "
    "missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, "
    "signature, watermark, username, blurry"
)


def firered_generate(
    gallery_images,
    prompt,
    seed,
    randomize_seed,
    guidance_scale,
    steps,
    progress=gr.Progress(),
):
    """Run FireRed image-edit inference on one or more reference images (max 3 natively)."""
    pil_images = _gallery_to_pil_list(gallery_images)
    if not pil_images:
        return None, int(seed), "Please upload at least one image."
    if not prompt or not prompt.strip():
        return None, int(seed), "Please enter an edit prompt."
    try:
        import gc

        progress(0.05, desc="Loading FireRed pipeline...")
        pipe = load_firered()

        if randomize_seed:
            seed = random.randint(0, 2**31 - 1)

        # FireRed natively handles 1-3 images; cap silently and warn
        if len(pil_images) > 3:
            print(
                f"[FireRed] {len(pil_images)} images given, truncating to 3 (native limit)."
            )
            pil_images = pil_images[:3]

        # Resize to max 1024px and align to multiple of 8 (prevents padding bars)
        pil_images = [_firered_resize(img) for img in pil_images]
        height, width = pil_images[0].height, pil_images[0].width
        print(f"[FireRed] Input size after resize: {width}x{height}")

        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

        progress(0.4, desc=f"Running FireRed edit ({len(pil_images)} image(s))...")
        with torch.inference_mode():
            result = pipe(
                image=pil_images,
                prompt=prompt.strip(),
                negative_prompt=_FIRERED_NEGATIVE,
                num_inference_steps=int(steps),
                generator=generator,
                true_cfg_scale=float(guidance_scale),
                num_images_per_prompt=1,
                height=height,
                width=width,
            ).images[0]

        gc.collect()
        torch.cuda.empty_cache()
        progress(1.0, desc="Done!")
        n = len(pil_images)
        note = (
            " (truncated to 3)"
            if n == 3 and len(_gallery_to_pil_list(gallery_images)) > 3
            else ""
        )
        return np.array(result), int(seed), f"Preview ready — {n} image(s) used{note}."
    except Exception:
        return None, int(seed), f"FireRed error:\n{traceback.format_exc()}"


def firered_load_into_pipeline(
    firered_output, threshold, erode_px, progress=gr.Progress()
):
    """Load a FireRed output into the main pipeline with automatic background removal."""
    if firered_output is None:
        return None, None, "No FireRed output — generate an image first."
    try:
        progress(0.1, desc="Loading RMBG model...")
        load_rmbg_only()

        img = Image.fromarray(firered_output).convert("RGB")
        if _rmbg_net is not None:
            progress(0.5, desc="Removing background...")
            composited = _remove_bg_rmbg(
                img, threshold=float(threshold), erode_px=int(erode_px)
            )
            result = np.array(composited)
            msg = "Loaded into pipeline — background removed."
        else:
            result = firered_output
            msg = "Loaded into pipeline (RMBG unavailable — background not removed)."

        progress(1.0, desc="Done!")
        return result, result, msg
    except Exception:
        return None, None, f"Error:\n{traceback.format_exc()}"


def generate_shape(
    input_image,
    remove_background,
    num_steps,
    guidance_scale,
    seed,
    face_count,
    progress=gr.Progress(),
):
    if input_image is None:
        return None, "Please upload an image."
    try:
        progress(0.05, desc="Freeing VRAM from FireRed (if loaded)...")
        global _firered_pipe
        if _firered_pipe is not None:
            # dispatch_model attaches accelerate hooks — remove them before .to("cpu")
            try:
                from accelerate.hooks import remove_hook_from_submodules

                remove_hook_from_submodules(_firered_pipe.transformer)
                _firered_pipe.transformer.to("cpu")
            except Exception as _e:
                print(f"[TripoSG] Transformer CPU offload: {_e}")
            try:
                _firered_pipe.text_encoder.to("cpu")
            except Exception as _e:
                print(f"[TripoSG] TextEncoder CPU offload: {_e}")
            try:
                _firered_pipe.vae.to("cpu")
            except Exception as _e:
                print(f"[TripoSG] VAE CPU offload: {_e}")
            # Mark pipe for full reload next FireRed call (hooks are gone)
            _firered_pipe = None
            torch.cuda.empty_cache()
            print("[TripoSG] FireRed offloaded — VRAM freed for shape generation.")

        progress(0.1, desc="Loading TripoSG...")
        sys.path.insert(0, TRIPOSG_DIR)
        from scripts.inference_triposg import run_triposg
        from scripts.image_process import prepare_image

        pipe, rmbg_net = load_triposg()

        img = Image.fromarray(input_image).convert("RGB")
        img_path = str(TMP_DIR / "triposg_input.png")
        img.save(img_path)

        progress(0.5, desc="Generating shape (SDF diffusion)...")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            mesh = run_triposg(
                pipe=pipe,
                image_input=img_path,
                rmbg_net=rmbg_net,  # always pass; TripoSG always calls it internally
                seed=int(seed),
                num_inference_steps=int(num_steps),
                guidance_scale=float(guidance_scale),
                faces=int(face_count) if int(face_count) > 0 else -1,
            )

        out_path = str(TMP_DIR / "triposg_shape.glb")
        mesh.export(out_path)

        # Offload models to CPU to free VRAM for texture subprocess
        _triposg_pipe.to("cpu")
        if _rmbg_net is not None:
            _rmbg_net.to("cpu")
        torch.cuda.empty_cache()

        return out_path, "Shape generated!"
    except Exception:
        return None, f"Error:\n{traceback.format_exc()}"


def _remove_bg_rmbg(img_pil, threshold=0.5, erode_px=2):
    """
    Remove background using RMBG (2.0 or 1.4), return RGB composited on neutral gray.
    threshold : float [0,1] — mask confidence cutoff; raise to cut more background
    erode_px  : int        — shrink mask by this many pixels to remove fringe
    """
    import torch
    import numpy as np
    import torchvision.transforms.functional as TF
    from torchvision import transforms

    if _rmbg_net is None:
        return img_pil

    device = next(_rmbg_net.parameters()).device
    _rmbg_net.eval()

    # Resize and preprocess
    img_resized = img_pil.resize((1024, 1024))
    img_tensor = transforms.ToTensor()(img_resized)
    img_tensor = TF.normalize(
        img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        result = _rmbg_net(img_tensor)

    # Handle both RMBG-2.0 (returns list) and RMBG-1.4 (returns tensor)
    if isinstance(result, (list, tuple)):
        candidate = result[-1]
        if isinstance(candidate, (list, tuple)):
            candidate = candidate[0]
    else:
        candidate = result

    # Extract mask and apply sigmoid if needed
    if candidate.dim() == 4:
        mask_tensor = candidate[0, 0]
    else:
        mask_tensor = candidate

    if mask_tensor.max() > 1.0:  # Already in [0, 1] after sigmoid
        mask_tensor = torch.sigmoid(mask_tensor)

    mask_pil = transforms.ToPILImage()(mask_tensor.cpu())
    mask = np.array(mask_pil.resize(img_pil.size, Image.BILINEAR), dtype=np.float32) / 255.0

    # Apply threshold
    mask = (mask >= threshold).astype(np.float32) * mask

    # Erode mask to remove background fringe
    if erode_px > 0:
        import cv2 as _cv2
        kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (erode_px * 2 + 1,) * 2)
        mask = _cv2.erode((mask * 255).astype(np.uint8), kernel).astype(np.float32) / 255.0

    # Composite on gray background
    rgb = np.array(img_pil.convert("RGB"), dtype=np.float32) / 255.0
    alpha = mask[:, :, np.newaxis]
    composited = rgb * alpha + 0.5 * (1.0 - alpha)
    composited = (composited * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(composited)


def _load_realesrgan(scale: int = 4):
    """Load RealESRGAN upsampler (x4plus by default). Returns RealESRGANer or None."""
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        if scale == 4:
            model_path = os.path.join(CKPT_DIR, "RealESRGAN_x4plus.pth")
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
        else:
            model_path = os.path.join(CKPT_DIR, "RealESRGAN_x2plus.pth")
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
        if not os.path.exists(model_path):
            print(f"[RealESRGAN] {model_path} not found")
            return None
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=512,
            tile_pad=32,
            pre_pad=0,
            half=True,
        )
        print(f"[RealESRGAN] Loaded x{scale}plus")
        return upsampler
    except Exception as e:
        print(f"[RealESRGAN] Load failed: {e}")
        return None


def _enhance_glb_texture(glb_path: str) -> bool:
    """
    Extract the base-color UV texture atlas from a GLB, upscale with RealESRGAN x4,
    downscale back to original resolution (sharper detail), then repack in-place.
    Returns True if enhancement was applied.
    """
    import pygltflib

    upsampler = _load_realesrgan(scale=4)
    if upsampler is None:
        # Try x2 fallback
        upsampler = _load_realesrgan(scale=2)
    if upsampler is None:
        print("[enhance_glb] No RealESRGAN checkpoint available")
        return False

    glb = pygltflib.GLTF2().load(glb_path)
    blob = bytearray(glb.binary_blob() or b"")

    for mat in glb.materials:
        bct = getattr(mat.pbrMetallicRoughness, "baseColorTexture", None)
        if bct is None:
            continue
        tex = glb.textures[bct.index]
        if tex.source is None:
            continue
        img_obj = glb.images[tex.source]
        if img_obj.bufferView is None:
            continue
        bv = glb.bufferViews[img_obj.bufferView]
        offset, length = bv.byteOffset or 0, bv.byteLength

        img_arr = np.frombuffer(blob[offset : offset + length], dtype=np.uint8)
        atlas_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if atlas_bgr is None:
            continue

        orig_h, orig_w = atlas_bgr.shape[:2]
        print(f"[enhance_glb] atlas {orig_w}x{orig_h}, upscaling with RealESRGAN…")

        try:
            upscaled, _ = upsampler.enhance(atlas_bgr, outscale=4)
        except Exception as e:
            print(f"[enhance_glb] RealESRGAN enhance failed: {e}")
            continue

        # Downscale back to original resolution — net effect: sharper details
        restored = cv2.resize(
            upscaled, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4
        )

        ok, new_bytes = cv2.imencode(".png", restored)
        if not ok:
            continue
        new_bytes = new_bytes.tobytes()
        new_len = len(new_bytes)

        if new_len > length:
            before = bytes(blob[:offset])
            after = bytes(blob[offset + length :])
            blob = bytearray(before + new_bytes + after)
            delta = new_len - length
            bv.byteLength = new_len
            for other_bv in glb.bufferViews:
                if (other_bv.byteOffset or 0) > offset:
                    other_bv.byteOffset += delta
            glb.buffers[0].byteLength += delta
        else:
            blob[offset : offset + new_len] = new_bytes
            bv.byteLength = new_len

        glb.set_binary_blob(bytes(blob))
        glb.save(glb_path)
        print(f"[enhance_glb] GLB texture enhanced OK (was {length}B → {new_len}B)")
        return True

    print("[enhance_glb] No base-color texture found in GLB")
    return False


def apply_texture(
    glb_path,
    input_image,
    remove_background,
    variant,
    tex_seed,
    enhance_face,
    rembg_threshold=0.5,
    rembg_erode=2,
    progress=gr.Progress(),
):
    if glb_path is None:
        glb_path = str(TMP_DIR / "triposg_shape.glb")
    if not os.path.exists(glb_path):
        return None, None, "Generate a shape first."
    if input_image is None:
        return None, None, "Please upload an image."
    try:
        progress(0.1, desc="Preprocessing image...")
        img = Image.fromarray(input_image).convert("RGB")

        # Save original photo before any processing — used as HyperSwap face source
        face_ref_path = str(TMP_DIR / "triposg_face_ref.png")
        img.save(face_ref_path)

        if remove_background and _rmbg_net is not None:
            img = _remove_bg_rmbg(
                img, threshold=float(rembg_threshold), erode_px=int(rembg_erode)
            )

        img = img.resize((768, 768), Image.LANCZOS)
        img_path = str(TMP_DIR / "tex_input.png")
        img.save(img_path)

        # Free GPU memory before launching SDXL subprocess (~15 GB peak)
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        out_dir = str(TMP_DIR / "tex_out")
        os.makedirs(out_dir, exist_ok=True)
        out_name = "textured"

        cmd = [
            PYTHON,
            "-m",
            "scripts.texture_i2tex",
            "--mesh",
            glb_path,
            "--image",
            img_path,
            "--save_dir",
            out_dir,
            "--save_name",
            out_name,
            "--variant",
            variant,
            "--seed",
            str(int(tex_seed)),
            "--device",
            DEVICE,
            "--reference_conditioning_scale",
            "1.5",
            "--text",
            "photorealistic person, detailed skin texture, realistic clothing",
            "--preprocess_mesh",
        ]
        # face enhancement is handled in-app after texture subprocess returns

        env = os.environ.copy()
        env["CUDA_HOME"] = "/root/miniconda/envs/triposg"
        env["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9;9.0;12.0"
        env["PATH"] = "/root/miniconda/envs/triposg/bin:" + env.get("PATH", "")
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        conda_lib = "/root/miniconda/envs/triposg/lib"
        env["LD_LIBRARY_PATH"] = conda_lib + ":" + env.get("LD_LIBRARY_PATH", "")

        progress(0.3, desc="Running MV-Adapter SDXL...")
        result = subprocess.run(
            cmd,
            cwd=MVADAPTER_DIR,
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
        )

        out_glb = f"{out_dir}/{out_name}_shaded.glb"
        mv_png = f"{out_dir}/{out_name}.png"

        if os.path.exists(out_glb):
            final_path = str(TMP_DIR / "triposg_textured.glb")
            shutil.copy(out_glb, final_path)

            # Face enhancement: extract UV texture atlas from GLB, run GFPGAN, repack
            face_enhanced = False
            if enhance_face:
                try:
                    import pygltflib

                    face_enhanced = _enhance_glb_texture(final_path)
                except Exception as _fe:
                    print(f"[enhance_glb] {_fe}")

            mv_out = mv_png if os.path.exists(mv_png) else None
            label = "Texture applied" + (" + face enhanced!" if face_enhanced else "!")
            global _last_glb_path
            _last_glb_path = final_path
            return final_path, mv_out, label
        else:
            err = result.stderr[-2000:] if result.stderr else "No output"
            return None, None, f"Texture failed:\n{err}"
    except Exception:
        return None, None, f"Error:\n{traceback.format_exc()}"


def preview_rembg(input_image, do_remove_bg, threshold, erode_px):
    """Preview REMBG result on upload. Returns composited RGB numpy array."""
    if input_image is None:
        return None
    if not do_remove_bg:
        return input_image
    if _rmbg_net is None:
        return input_image  # models not loaded yet — skip blocking load
    try:
        img = Image.fromarray(input_image).convert("RGB")
        composited = _remove_bg_rmbg(
            img, threshold=float(threshold), erode_px=int(erode_px)
        )
        return np.array(composited)
    except Exception:
        return input_image


def render_views(glb_file):
    """Render a GLB from 5 standard angles using nvdiffrast."""
    if not glb_file:
        return []
    if isinstance(glb_file, str):
        glb_path = glb_file
    elif isinstance(glb_file, dict):
        glb_path = glb_file.get("path") or glb_file.get("name") or ""
    else:
        glb_path = str(glb_file)
    if not glb_path or not os.path.exists(glb_path):
        msg = f"render_views: GLB not found ({glb_path!r})"
        print(msg)
        return [{"image": None, "caption": msg}]
    print(f"render_views: loading {glb_path} ({os.path.getsize(glb_path) // 1024}KB)")
    try:
        sys.path.insert(0, MVADAPTER_DIR)
        print("render_views: importing nvdiffrast utils...")
        from mvadapter.utils.mesh_utils import (
            NVDiffRastContextWrapper,
            load_mesh,
            render,
            get_orthogonal_camera,
        )

        device = "cuda"
        ctx = NVDiffRastContextWrapper(device=device, context_type="cuda")
        print("render_views: loading mesh...")
        mesh = load_mesh(glb_path, rescale=True, device=device)
        print(f"render_views: mesh loaded, rendering...")

        azimuth_deg = [x - 90 for x in [0, 45, 90, 180, 315]]
        cameras = get_orthogonal_camera(
            elevation_deg=[0, 0, 0, 0, 0],
            distance=[1.8] * 5,
            left=-0.55,
            right=0.55,
            bottom=-0.55,
            top=0.55,
            azimuth_deg=azimuth_deg,
            device=device,
        )

        render_out = render(
            ctx,
            mesh,
            cameras,
            height=1024,
            width=768,
            render_attr=True,
            normal_background=0.0,
        )
        print(f"render_views: render complete, attr shape={render_out.attr.shape}")

        names = ["front", "3q_front", "side", "back", "3q_back"]
        save_dir = os.path.dirname(glb_path)
        results = []
        for i, name in enumerate(names):
            arr = (render_out.attr[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            path = os.path.join(save_dir, f"render_{name}.png")
            Image.fromarray(arr).save(path)
            results.append((path, name))
            print(f"render_views: saved {name} -> {path}")

        return results
    except Exception:
        err = traceback.format_exc()
        print(f"render_views FAILED:\n{err}")
        return []


def hyperswap_views(embedding_json: str):
    """
    Stage 6 — run HyperSwap on the last rendered views.
    embedding_json: JSON string of the 512-d ArcFace embedding list.
    Returns a gallery of (swapped_image_path, view_name) tuples.
    """
    global _hyperswap_sess
    try:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        embedding = np.array(json.loads(embedding_json), dtype=np.float32)
        embedding /= np.linalg.norm(embedding)

        # Load HyperSwap once
        if _hyperswap_sess is None:
            hs_path = os.path.join(CKPT_DIR, "hyperswap_1a_256.onnx")
            _hyperswap_sess = ort.InferenceSession(
                hs_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            print(f"[hyperswap_views] Loaded {hs_path}")

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.1)

        results = []
        for view_path, name in zip(VIEW_PATHS, VIEW_NAMES):
            if not os.path.exists(view_path):
                print(f"[hyperswap_views] Missing {view_path}, skipping")
                continue

            bgr = cv2.imread(view_path)
            faces = app.get(bgr)
            if not faces:
                print(f"[hyperswap_views] {name}: no face detected")
                out_path = view_path  # return original
            else:
                face = faces[0]
                M, _ = cv2.estimateAffinePartial2D(
                    face.kps, ARCFACE_256, method=cv2.RANSAC, ransacReprojThreshold=100
                )
                H, W = bgr.shape[:2]
                aligned = cv2.warpAffine(bgr, M, (256, 256), flags=cv2.INTER_LINEAR)
                t = (
                    ((aligned.astype(np.float32) / 255 - 0.5) / 0.5)[:, :, ::-1]
                    .copy()
                    .transpose(2, 0, 1)[None]
                )
                out, mask = _hyperswap_sess.run(
                    None,
                    {
                        "source": embedding.reshape(1, -1),
                        "target": t,
                    },
                )
                out_bgr = (
                    ((out[0].transpose(1, 2, 0) + 1) / 2 * 255)
                    .clip(0, 255)
                    .astype(np.uint8)
                )[:, :, ::-1].copy()
                m = (mask[0, 0] * 255).clip(0, 255).astype(np.uint8)
                Mi = cv2.invertAffineTransform(M)
                of = cv2.warpAffine(out_bgr, Mi, (W, H), flags=cv2.INTER_LINEAR)
                mf = (
                    cv2.warpAffine(m, Mi, (W, H), flags=cv2.INTER_LINEAR).astype(
                        np.float32
                    )[:, :, None]
                    / 255
                )
                swapped = (of * mf + bgr * (1 - mf)).clip(0, 255).astype(np.uint8)

                # GFPGAN face restoration — use the SAME bbox from the already-detected face
                # (avoids re-running InsightFace at det_thresh=0.1 which can latch onto skin/body)
                restorer = load_gfpgan()
                if restorer is not None:
                    b = face.bbox.astype(int)
                    h2, w2 = swapped.shape[:2]
                    pad = 0.35
                    bw2, bh2 = b[2] - b[0], b[3] - b[1]
                    cx1 = max(0, b[0] - int(bw2 * pad))
                    cy1 = max(0, b[1] - int(bh2 * pad))
                    cx2 = min(w2, b[2] + int(bw2 * pad))
                    cy2 = min(h2, b[3] + int(bh2 * pad))
                    crop = swapped[cy1:cy2, cx1:cx2]
                    try:
                        _, _, rest = restorer.enhance(
                            crop,
                            has_aligned=False,
                            only_center_face=True,
                            paste_back=True,
                            weight=0.5,
                        )
                        if rest is not None:
                            ch, cw = cy2 - cy1, cx2 - cx1
                            if rest.shape[:2] != (ch, cw):
                                rest = cv2.resize(
                                    rest, (cw, ch), interpolation=cv2.INTER_LANCZOS4
                                )
                            swapped[cy1:cy2, cx1:cx2] = rest
                    except Exception as _ge:
                        print(f"[hyperswap_views] GFPGAN failed: {_ge}")

                out_path = view_path.replace("render_", "swapped_")
                cv2.imwrite(out_path, swapped)
                print(f"[hyperswap_views] {name}: swapped+restored OK -> {out_path}")

            results.append((out_path, name))

        return results
    except Exception:
        err = traceback.format_exc()
        print(f"hyperswap_views FAILED:\n{err}")
        return []


def gradio_tpose(glb_state_path, export_skel_flag, progress=gr.Progress()):
    """Rig surface mesh with YOLO-pose + optionally export SKEL bone mesh."""
    try:
        glb = glb_state_path or _last_glb_path or str(TMP_DIR / "triposg_textured.glb")
        if not os.path.exists(glb):
            return (
                None,
                None,
                "No GLB found — run Generate Shape + Apply Texture first.",
            )

        # Surface: YOLO-rig (replaces broken inverse-LBS T-pose)
        progress(0.1, desc="YOLO pose detection + rigging surface ...")
        sys.path.insert(0, "/root")
        from rig_yolo import rig_yolo

        out_dir = str(TMP_DIR / "rig_out")
        os.makedirs(out_dir, exist_ok=True)
        rigged, _rigged_skel = rig_yolo(
            glb, os.path.join(out_dir, "anatomy_rigged.glb"), debug_dir=None
        )

        # SKEL bone mesh (zero-pose T-posed skeleton)
        bones = None
        if export_skel_flag:
            progress(0.7, desc="Generating SKEL bone mesh ...")
            import torch
            from tpose_smpl import export_skel_bones

            bones = export_skel_bones(
                torch.zeros(10), str(TMP_DIR / "tposed_bones.glb"), gender="male"
            )

        status = f"Rigged surface: {os.path.getsize(rigged) // 1024} KB"
        if bones:
            status += f"\nSKEL bone mesh: {os.path.getsize(bones) // 1024} KB"
        elif export_skel_flag:
            status += "\nSKEL bone mesh: failed (check logs)"
        progress(1.0, desc="Done!")
        return rigged, bones, status
    except Exception:
        return None, None, f"Error:\n{traceback.format_exc()}"


UNIRIG_DIR = "/root/UniRig"
UNIRIG_PY = "/root/miniconda/envs/unirig/bin/python"
UNIRIG_BASH = "/root/miniconda/envs/unirig/bin"  # prepended to PATH for launch scripts


def _run_unirig(glb_path: str, out_dir: str) -> str:
    """
    Run the 3-step UniRig pipeline on a textured GLB.
    Returns path to the final rigged GLB, or raises on failure.
    """
    if not os.path.exists(UNIRIG_PY):
        raise RuntimeError("UniRig conda env not found — run setup_unirig.sh first")

    os.makedirs(out_dir, exist_ok=True)
    skel_fbx = os.path.join(out_dir, "skeleton.fbx")
    skin_fbx = os.path.join(out_dir, "skin.fbx")
    rigged = os.path.join(out_dir, "rigged.glb")

    env = os.environ.copy()
    env["PATH"] = f"{UNIRIG_BASH}:{env.get('PATH', '')}"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    def _launch(script: str, extra_args: list[str]):
        sh = os.path.join(UNIRIG_DIR, "launch", "inference", script)
        cmd = ["bash", sh] + extra_args
        r = subprocess.run(
            cmd, cwd=UNIRIG_DIR, capture_output=True, text=True, timeout=300, env=env
        )
        if r.returncode != 0:
            raise RuntimeError(f"{script} failed:\n{r.stderr[-2000:]}")
        return r

    print("[UniRig] Step 1/3 — generate skeleton...")
    _launch("generate_skeleton.sh", ["--input", glb_path, "--output", skel_fbx])

    print("[UniRig] Step 2/3 — generate skinning...")
    _launch("generate_skin.sh", ["--input", skel_fbx, "--output", skin_fbx])

    print("[UniRig] Step 3/3 — merge rig into mesh...")
    _launch(
        "merge.sh", ["--source", skin_fbx, "--target", glb_path, "--output", rigged]
    )

    # UniRig ignores --output dir and always writes to /tmp/rig_out/rigged.glb
    # Fall back to that location if the requested path isn't populated.
    if not os.path.exists(rigged):
        fallback = str(TMP_DIR / "rig_out" / "rigged.glb")
        if os.path.exists(fallback):
            import shutil

            shutil.copy2(fallback, rigged)
        else:
            raise RuntimeError(
                f"UniRig finished but output not found at {rigged} or {fallback}"
            )

    print(f"[UniRig] Done — {os.path.getsize(rigged) // 1024} KB")
    return rigged


def gradio_rig(
    input_image,
    glb_state_path,
    export_fbx_flag,
    pshuman_weight_threshold: float,
    pshuman_retract_mm: float,
    progress=gr.Progress(),
):
    """
    Rig pipeline — three stages run automatically in one click:
      1. UniRig: skeleton + skinning weights on the TripoSG mesh
      2. PSHuman: generate HD face from portrait (RMBG → RGBA → subprocess)
      3. Face transplant: stitch PSHuman face into rigged mesh via bone-weight
         head detection + KNN weight transfer → final rigged+HD-face GLB
    If no portrait is available, stages 2-3 are skipped.
    """
    try:
        glb = glb_state_path or _last_glb_path or str(TMP_DIR / "triposg_textured.glb")
        if not os.path.exists(glb):
            return (
                None,
                None,
                None,
                "No GLB found — run Generate Shape + Apply Texture first.",
                None,
                None,
                None,
            )

        out_dir = str(TMP_DIR / "rig_out")
        os.makedirs(out_dir, exist_ok=True)

        # ── Stage 1: UniRig ───────────────────────────────────────────────────
        progress(0.05, desc="Stage 1/3: UniRig — generating skeleton + skinning...")
        rigged = _run_unirig(glb, out_dir)
        final = rigged

        # ── Stage 2+3: PSHuman face (only if portrait is loaded) ───────��─────
        if input_image is not None:
            try:
                _meshforge_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "MeshForge"
                )
                if not os.path.isdir(_meshforge_dir):
                    _meshforge_dir = os.path.dirname(os.path.abspath(__file__))
                if _meshforge_dir not in sys.path:
                    sys.path.insert(0, _meshforge_dir)

                work_dir = tempfile.mkdtemp(prefix="pshuman_rig_")
                img_path = os.path.join(work_dir, "portrait.png")

                progress(
                    0.6,
                    desc="Stage 2/3: PSHuman — RMBG + multi-view face generation...",
                )
                pil_img = (
                    Image.fromarray(input_image)
                    if isinstance(input_image, np.ndarray)
                    else input_image
                )
                rgba = _portrait_to_rgba(pil_img)
                rgba.save(img_path)

                from pipeline.pshuman_client import generate_pshuman_mesh

                face_obj = os.path.join(work_dir, "pshuman_face.obj")
                generate_pshuman_mesh(
                    image_path=img_path, output_path=face_obj, service_url="direct"
                )

                progress(
                    0.85,
                    desc="Stage 3/3: Face transplant — stitching into rigged mesh...",
                )
                from pipeline.face_transplant import transplant_face

                final = os.path.join(work_dir, "rigged_hd_face.glb")
                transplant_face(
                    body_glb_path=rigged,
                    pshuman_mesh_path=face_obj,
                    output_path=final,
                    weight_threshold=float(pshuman_weight_threshold),
                    retract_amount=float(pshuman_retract_mm) / 1000.0,
                )
                print(f"[rig] PSHuman face transplant complete: {final}")
            except Exception as _pse:
                print(
                    f"[rig] PSHuman stage failed, using plain rig: {_pse}\n{traceback.format_exc()}"
                )
                final = rigged

        fbx = None
        if export_fbx_flag:
            progress(0.92, desc="Exporting FBX...")
            try:
                sys.path.insert(0, "/root")
                from rig_stage import export_fbx as _export_fbx

                fbx_path = os.path.join(out_dir, "rigged.fbx")
                fbx = fbx_path if _export_fbx(final, fbx_path) else None
            except Exception as _fe:
                print(f"[rig] FBX export failed: {_fe}")

        had_pshuman = input_image is not None and final != rigged
        status_msg = (
            "Rigged + PSHuman HD face: " if had_pshuman else "Rigged: "
        ) + os.path.basename(final)
        if fbx:
            status_msg += "  |  FBX: " + os.path.basename(fbx)
        progress(1.0, desc="Done!")
        return final, None, fbx, status_msg, final, final, None
    except Exception:
        return None, None, None, f"Error:\n{traceback.format_exc()}", None, None, None


def run_full_pipeline(
    input_image,
    remove_background,
    num_steps,
    guidance,
    seed,
    face_count,
    variant,
    tex_seed,
    enhance_face,
    rembg_threshold,
    rembg_erode,
    export_fbx,
    progress=gr.Progress(),
):
    """Single-click full pipeline: shape → texture → rig."""
    progress(0.0, desc="Stage 1/3: Generating shape...")
    glb, status = generate_shape(
        input_image, remove_background, num_steps, guidance, seed, face_count
    )
    if not glb:
        return None, None, None, None, None, None, status

    progress(0.33, desc="Stage 2/3: Applying texture + face enhancement...")
    glb, mv_img, status = apply_texture(
        glb,
        input_image,
        remove_background,
        variant,
        tex_seed,
        enhance_face,
        rembg_threshold,
        rembg_erode,
    )
    if not glb:
        return None, None, None, None, None, None, status

    progress(0.66, desc="Stage 3/3: Rigging (UniRig + PSHuman)...")
    rigged, animated, fbx, rig_status, _, _, _skel = gradio_rig(
        input_image, glb, export_fbx, 0.5, 2.0
    )

    progress(1.0, desc="Pipeline complete!")
    combined_status = f"[Texture] {status}\n[Rig] {rig_status}"
    return glb, glb, mv_img, rigged, fbx, combined_status


# ─────────────────────────────────────────────────────────────────────────────
# Animate tab — motion search + bake
# ─────────────────────────────────────────────────────────────────────────────


def gradio_search_motions(query: str, progress=gr.Progress()):
    """Stream TeoGchx/HumanML3D and return matching motions as radio choices."""
    if not query.strip():
        return (
            gr.update(choices=[], visible=False),
            [],
            "Enter a motion description and click Search.",
        )
    try:
        progress(0.1, desc="Connecting to HumanML3D dataset…")
        sys.path.insert(0, "/root")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from Retarget.search import search_motions, format_choice_label

        progress(0.3, desc="Streaming dataset…")
        results = search_motions(query, top_k=8)
        progress(1.0)
        if not results:
            return (
                gr.update(
                    choices=["No matches — try different keywords"], visible=True
                ),
                [],
                f"No motions matched '{query}'. Try broader terms.",
            )
        choices = [format_choice_label(r) for r in results]
        status = f"Found {len(results)} motions matching '{query}'"
        return (
            gr.update(choices=choices, value=choices[0], visible=True),
            results,
            status,
        )
    except Exception:
        return (
            gr.update(choices=[], visible=False),
            [],
            f"Search error:\n{traceback.format_exc()}",
        )


def gradio_animate(
    rigged_glb_path,
    selected_label: str,
    motion_results: list,
    fps: int,
    max_frames: int,
    progress=gr.Progress(),
):
    """Bake selected HumanML3D motion onto the UniRig-rigged GLB."""
    try:
        glb = rigged_glb_path or str(TMP_DIR / "rig_out" / "rigged.glb")
        if not os.path.exists(glb):
            return None, "No rigged GLB — run the Rig step first.", None

        if not motion_results or not selected_label:
            return None, "No motion selected — run Search first.", None

        # Resolve which result was selected
        sys.path.insert(0, "/root")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from Retarget.search import format_choice_label

        idx = 0
        for i, r in enumerate(motion_results):
            if format_choice_label(r) == selected_label:
                idx = i
                break

        chosen = motion_results[idx]
        motion = chosen["motion"]  # np.ndarray [T, 263]
        caption = chosen["caption"]
        T_total = motion.shape[0]
        n_frames = min(max_frames, T_total) if max_frames > 0 else T_total

        progress(0.2, desc="Parsing skeleton…")
        from Retarget.animate import animate_glb_from_hml3d

        out_path = str(TMP_DIR / "animated_out" / "animated.glb")
        os.makedirs(str(TMP_DIR / "animated_out"), exist_ok=True)

        progress(0.4, desc="Mapping bones to SMPL joints…")
        animated = animate_glb_from_hml3d(
            motion=motion,
            rigged_glb=glb,
            output_glb=out_path,
            fps=int(fps),
            num_frames=int(n_frames),
        )
        progress(1.0, desc="Done!")
        status = f"Animated: {n_frames} frames @ {fps} fps\nMotion: {caption[:120]}"
        return animated, status, animated

    except Exception:
        return None, f"Error:\n{traceback.format_exc()}", None


# ─────────────────────────────────────────────────────────────────────────────
# PSHuman Face Transplant tab
# ─────────────────────────────────────────────────────────────────────────────


def _portrait_to_rgba(img_pil: Image.Image) -> Image.Image:
    """
    Run RMBG on a portrait and return an RGBA PIL image where alpha = foreground mask.
    PSHuman's dataset loader expects RGBA — it reads channel 3 as the alpha/mask.
    Falls back to fully-opaque RGBA if RMBG is unavailable.
    """
    import torchvision.transforms.functional as _TF
    from torchvision import transforms as _tvt

    load_rmbg_only()
    if _rmbg_net is None:
        return img_pil.convert("RGBA")

    # Run on CPU — keeps GPU free for the PSHuman subprocess that follows
    _rmbg_net.to("cpu").eval()

    src = img_pil.convert("RGB")
    img_t = _tvt.ToTensor()(src.resize((1024, 1024)))
    img_t = _TF.normalize(
        img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ).unsqueeze(0)
    with torch.no_grad():
        result = _rmbg_net(img_t)
    if isinstance(result, (list, tuple)):
        candidate = result[-1]
        if isinstance(candidate, (list, tuple)):
            candidate = candidate[0]
    else:
        candidate = result

    mask_t = candidate.sigmoid()[0, 0].cpu()
    mask_pil = _tvt.ToPILImage()(mask_t).resize(src.size, Image.BILINEAR)

    rgba = src.convert("RGBA")
    rgba.putalpha(mask_pil)
    return rgba


def gradio_pshuman_face(
    input_image,
    rigged_glb_path,
    weight_threshold: float,
    retract_mm: float,
    progress=gr.Progress(),
):
    """
    PSHuman face transplant — post-rig pipeline:
      1. Run RMBG on portrait → RGBA (PSHuman needs alpha channel as foreground mask)
      2. Run PSHuman on RGBA portrait → colored OBJ face mesh (direct subprocess)
      3. Transplant face into rigged GLB: bone weights ID head verts, KNN transfers
         skinning to PSHuman face. Output is a fully rigged mesh — no second rig pass.
    """
    try:
        if input_image is None:
            return None, "No portrait found — run Generate first.", None
        rigged = rigged_glb_path
        if not rigged or not os.path.exists(str(rigged)):
            return None, "No rigged GLB found — run Rig & Export first.", None

        work_dir = tempfile.mkdtemp(prefix="pshuman_transplant_")
        img_path = os.path.join(work_dir, "portrait.png")

        progress(0.03, desc="Preparing portrait (RMBG → RGBA)...")
        pil_img = (
            Image.fromarray(input_image)
            if isinstance(input_image, np.ndarray)
            else input_image
        )
        rgba = _portrait_to_rgba(pil_img)
        rgba.save(img_path)
        print(f"[pshuman] Portrait saved as RGBA {rgba.size} → {img_path}")

        # Pipeline modules live at /root/MeshForge/pipeline/ on the instance
        _meshforge_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "MeshForge"
        )
        if not os.path.isdir(_meshforge_dir):
            _meshforge_dir = os.path.dirname(os.path.abspath(__file__))
        if _meshforge_dir not in sys.path:
            sys.path.insert(0, _meshforge_dir)

        # ── Step 2: PSHuman inference ──────────────────────────────────────────
        progress(0.08, desc="Step 2/3: Running PSHuman (multi-view face generation)...")
        from pipeline.pshuman_client import generate_pshuman_mesh

        face_obj = os.path.join(work_dir, "pshuman_face.obj")
        generate_pshuman_mesh(
            image_path=img_path,
            output_path=face_obj,
            service_url="direct",
        )

        # ── Step 3: Transplant into rigged GLB (bone-weight head detection + KNN) ──
        progress(0.7, desc="Step 3/3: Transplanting PSHuman face into rigged GLB...")
        out_glb = os.path.join(work_dir, "rigged_pshuman_face.glb")

        from pipeline.face_transplant import transplant_face

        transplant_face(
            body_glb_path=str(rigged),
            pshuman_mesh_path=face_obj,
            output_path=out_glb,
            weight_threshold=float(weight_threshold),
            retract_amount=float(retract_mm) / 1000.0,
        )

        progress(1.0, desc="Done!")
        return out_glb, "PSHuman face transplant complete.", out_glb

    except Exception:
        return None, f"Error:\n{traceback.format_exc()}", None


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="TripoSG + MV-Adapter 3D Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# TripoSG + MV-Adapter 3D Studio")
    glb_state = gr.State(None)
    rigged_glb_state = gr.State(None)  # persists UniRig output for Animate tab

    with gr.Tabs() as tabs:
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("Edit", id=0):
            gr.Markdown(
                "### Image Edit — FireRed\n"
                "Upload one or more reference images, write an edit prompt, preview the result, "
                "then click **Load to Generate** to send it to the 3D pipeline."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    firered_gallery = gr.Gallery(
                        label="Reference Images (1–3 images, drag & drop)",
                        interactive=True,
                        columns=3,
                        height=220,
                        object_fit="contain",
                    )
                    firered_prompt = gr.Textbox(
                        label="Edit Prompt",
                        placeholder="make the person wear a red jacket",
                        lines=2,
                    )
                    with gr.Row():
                        firered_seed = gr.Number(
                            value=_init_seed, label="Seed", precision=0
                        )
                        firered_rand = gr.Checkbox(label="Random Seed", value=True)
                    with gr.Row():
                        firered_guidance = gr.Slider(
                            1.0, 10.0, value=1.0, step=0.5, label="Guidance Scale"
                        )
                        firered_steps = gr.Slider(
                            1, 40, value=4, step=1, label="Inference Steps"
                        )
                    firered_btn = gr.Button("Generate Preview", variant="secondary")
                    firered_status = gr.Textbox(
                        label="Status", lines=2, interactive=False
                    )
                with gr.Column(scale=1):
                    firered_output_img = gr.Image(
                        label="FireRed Output", type="numpy", interactive=False
                    )
                    load_to_generate_btn = gr.Button(
                        "Load to Generate", variant="primary"
                    )

        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("Generate", id=1):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(label="Input Image", type="numpy")
                    remove_bg_check = gr.Checkbox(label="Remove Background", value=True)
                    with gr.Row():
                        rembg_threshold = gr.Slider(
                            0.1,
                            0.95,
                            value=0.5,
                            step=0.05,
                            label="BG Threshold (higher = stricter)",
                        )
                        rembg_erode = gr.Slider(
                            0, 8, value=2, step=1, label="Edge Erode (px)"
                        )

                    with gr.Accordion("Shape Settings", open=True):
                        num_steps = gr.Slider(
                            20, 100, value=50, step=5, label="Inference Steps"
                        )
                        guidance = gr.Slider(
                            1.0, 20.0, value=7.0, step=0.5, label="Guidance Scale"
                        )
                        seed = gr.Number(value=_init_seed, label="Seed", precision=0)
                        face_count = gr.Number(
                            value=0, label="Max Faces (0 = unlimited)", precision=0
                        )

                    with gr.Accordion("Texture Settings", open=True):
                        variant = gr.Radio(
                            ["sdxl", "sd21"],
                            value="sdxl",
                            label="Model (sdxl = better quality, sd21 = less VRAM)",
                        )
                        tex_seed = gr.Number(
                            value=_init_seed, label="Texture Seed", precision=0
                        )
                        enhance_face_check = gr.Checkbox(
                            label="Enhance Face (HyperSwap + RealESRGAN)", value=True
                        )

                    with gr.Row():
                        shape_btn = gr.Button(
                            "Generate Shape",
                            variant="primary",
                            scale=2,
                            interactive=False,
                        )
                        texture_btn = gr.Button(
                            "Apply Texture", variant="secondary", scale=2
                        )
                        render_btn = gr.Button(
                            "Render Views", variant="secondary", scale=1
                        )
                    run_all_btn = gr.Button(
                        "▶ Run Full Pipeline (Shape + Texture + Rig)",
                        variant="primary",
                        interactive=False,
                    )

                with gr.Column(scale=1):
                    rembg_preview = gr.Image(
                        label="BG Removed Preview", type="numpy", interactive=False
                    )
                    status = gr.Textbox(label="Status", lines=3, interactive=False)
                    model_3d = gr.Model3D(
                        label="3D Preview", clear_color=[0.9, 0.9, 0.9, 1.0]
                    )
                    download_file = gr.File(label="Download GLB")
                    multiview_img = gr.Image(
                        label="Multiview", type="filepath", interactive=False
                    )

            render_gallery = gr.Gallery(label="Rendered Views", columns=5, height=300)

            # ── wiring: Generate tab ──────────────────────────────────────
            _rembg_inputs = [input_image, remove_bg_check, rembg_threshold, rembg_erode]
            _pipeline_btns = [shape_btn, run_all_btn]

            input_image.upload(
                fn=lambda: (gr.update(interactive=True), gr.update(interactive=True)),
                inputs=[],
                outputs=_pipeline_btns,
            )
            input_image.clear(
                fn=lambda: (gr.update(interactive=False), gr.update(interactive=False)),
                inputs=[],
                outputs=_pipeline_btns,
            )

            input_image.upload(
                fn=preview_rembg, inputs=_rembg_inputs, outputs=[rembg_preview]
            )
            remove_bg_check.change(
                fn=preview_rembg, inputs=_rembg_inputs, outputs=[rembg_preview]
            )
            rembg_threshold.release(
                fn=preview_rembg, inputs=_rembg_inputs, outputs=[rembg_preview]
            )
            rembg_erode.release(
                fn=preview_rembg, inputs=_rembg_inputs, outputs=[rembg_preview]
            )

            shape_btn.click(
                fn=generate_shape,
                inputs=[
                    input_image,
                    remove_bg_check,
                    num_steps,
                    guidance,
                    seed,
                    face_count,
                ],
                outputs=[glb_state, status],
            ).then(
                fn=lambda p: (p, p) if p else (None, None),
                inputs=[glb_state],
                outputs=[model_3d, download_file],
            )

            texture_btn.click(
                fn=apply_texture,
                inputs=[
                    glb_state,
                    input_image,
                    remove_bg_check,
                    variant,
                    tex_seed,
                    enhance_face_check,
                    rembg_threshold,
                    rembg_erode,
                ],
                outputs=[glb_state, multiview_img, status],
            ).then(
                fn=lambda p: (p, p) if p else (None, None),
                inputs=[glb_state],
                outputs=[model_3d, download_file],
            )

            render_btn.click(
                fn=render_views, inputs=[download_file], outputs=[render_gallery]
            )

        # ── Edit tab wiring (after Generate so all components are defined) ──
        firered_btn.click(
            fn=firered_generate,
            inputs=[
                firered_gallery,
                firered_prompt,
                firered_seed,
                firered_rand,
                firered_guidance,
                firered_steps,
            ],
            outputs=[firered_output_img, firered_seed, firered_status],
            api_name="firered_generate",
        )

        load_to_generate_btn.click(
            fn=firered_load_into_pipeline,
            inputs=[firered_output_img, rembg_threshold, rembg_erode],
            outputs=[input_image, rembg_preview, firered_status],
        ).then(
            fn=lambda img: (
                gr.update(interactive=img is not None),
                gr.update(interactive=img is not None),
                gr.update(selected=1),
            ),
            inputs=[input_image],
            outputs=[shape_btn, run_all_btn, tabs],
        )

        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("Rig & Export"):
            with gr.Row():
                # ── Left column: controls ──────────────────────────────────
                with gr.Column(scale=1):
                    gr.Markdown("### UniRig + PSHuman — Rig & HD Face")
                    gr.Markdown(
                        "One click runs the full pipeline:\n"
                        "1. **UniRig** skeletonises + skins the mesh\n"
                        "2. **PSHuman** generates an HD face from your portrait (RMBG → multi-view diffusion)\n"
                        "3. **Face transplant** stitches the HD face into the rigged mesh using bone weights + KNN\n\n"
                        "Portrait is pulled automatically from the Generate tab."
                    )
                    export_fbx_check = gr.Checkbox(label="Export FBX", value=True)
                    with gr.Accordion("PSHuman settings", open=False):
                        pshuman_weight_thresh = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.35,
                            step=0.05,
                            label="Head bone weight threshold",
                            info="Vertices with head-bone weight above this get replaced",
                        )
                        pshuman_retract_mm = gr.Slider(
                            minimum=0.0,
                            maximum=20.0,
                            value=4.0,
                            step=0.5,
                            label="Face retract (mm)",
                            info="How far to push original face verts inward to avoid z-fighting",
                        )
                    rig_btn = gr.Button("Rig with UniRig", variant="primary")

                # ── Right column: preview + downloads ─────────────────────
                with gr.Column(scale=2):
                    rig_status = gr.Textbox(label="Status", lines=4, interactive=False)
                    rig_model_3d = gr.Model3D(
                        label="Preview", clear_color=[0.9, 0.9, 0.9, 1.0]
                    )
                    with gr.Row():
                        rig_glb_dl = gr.File(label="Download Rigged GLB")
                        rig_fbx_dl = gr.File(label="Download FBX")

            rig_btn.click(
                fn=gradio_rig,
                inputs=[
                    input_image,
                    glb_state,
                    export_fbx_check,
                    pshuman_weight_thresh,
                    pshuman_retract_mm,
                ],
                outputs=[
                    rig_glb_dl,
                    gr.State(None),
                    rig_fbx_dl,
                    rig_status,
                    rig_model_3d,
                    rigged_glb_state,
                    gr.State(None),
                ],
            )

        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("Enhancement"):
            gr.Markdown("""
            **Surface Enhancement** — runs on the reference portrait to produce
            calibrated normal + depth maps that are baked into the GLB as PBR textures.
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### StableNormal")
                    run_normal_check = gr.Checkbox(label="Run StableNormal", value=True)
                    normal_res = gr.Slider(
                        512, 1024, value=768, step=128, label="Resolution"
                    )
                    normal_strength = gr.Slider(
                        0.1, 3.0, value=1.0, step=0.1, label="Normal Strength"
                    )

                    gr.Markdown("### Depth-Anything V2")
                    run_depth_check = gr.Checkbox(
                        label="Run Depth-Anything V2", value=True
                    )
                    depth_res = gr.Slider(
                        512, 1024, value=768, step=128, label="Resolution"
                    )
                    displacement_scale = gr.Slider(
                        0.1, 3.0, value=1.0, step=0.1, label="Displacement Scale"
                    )

                    enhance_btn = gr.Button("Run Enhancement", variant="primary")
                    unload_btn = gr.Button(
                        "Unload Models (free VRAM)", variant="secondary"
                    )

                with gr.Column(scale=2):
                    enhance_status = gr.Textbox(
                        label="Status", lines=5, interactive=False
                    )
                    with gr.Row():
                        normal_map_img = gr.Image(label="Normal Map", type="pil")
                        depth_map_img = gr.Image(label="Depth Map", type="pil")
                    enhanced_glb_dl = gr.File(label="Download Enhanced GLB")
                    enhanced_model_3d = gr.Model3D(
                        label="Enhanced Preview", clear_color=[0.9, 0.9, 0.9, 1.0]
                    )

            def gradio_enhance(
                glb_path,
                ref_img_np,
                do_normal,
                norm_res,
                norm_strength,
                do_depth,
                dep_res,
                disp_scale,
            ):
                if not glb_path:
                    return None, None, None, None, "No GLB loaded — run Generate first."
                if ref_img_np is None:
                    return (
                        None,
                        None,
                        None,
                        None,
                        "No reference image — run Generate first.",
                    )
                try:
                    ref_pil = Image.fromarray(ref_img_np.astype(np.uint8))
                    out_path = glb_path.replace(".glb", "_enhanced.glb")
                    import shutil as _sh

                    _sh.copy2(glb_path, out_path)

                    normal_out = None
                    depth_out = None
                    log = []

                    if do_normal:
                        log.append("[StableNormal] Running...")
                        yield None, None, None, None, "\n".join(log)
                        normal_out = run_stable_normal(ref_pil, resolution=norm_res)
                        out_path = bake_normal_into_glb(
                            out_path,
                            normal_out,
                            out_path,
                            normal_strength=norm_strength,
                        )
                        log.append(
                            f"[StableNormal] Done → baked normalTexture (strength {norm_strength})"
                        )
                        yield normal_out, depth_out, None, None, "\n".join(log)

                    if do_depth:
                        log.append("[Depth-Anything] Running...")
                        yield normal_out, depth_out, None, None, "\n".join(log)
                        depth_out = run_depth_anything(ref_pil, resolution=dep_res)
                        out_path = bake_depth_as_occlusion(
                            out_path, depth_out, out_path, displacement_scale=disp_scale
                        )
                        depth_preview = depth_out.convert("L").convert("RGB")
                        log.append(
                            f"[Depth-Anything] Done → baked occlusionTexture (scale {disp_scale})"
                        )
                        yield normal_out, depth_preview, None, None, "\n".join(log)

                    log.append("Enhancement complete.")
                    yield (
                        normal_out,
                        (depth_out.convert("L").convert("RGB") if depth_out else None),
                        out_path,
                        out_path,
                        "\n".join(log),
                    )

                except Exception as e:
                    yield None, None, None, None, f"Error:\n{traceback.format_exc()}"

            enhance_btn.click(
                fn=gradio_enhance,
                inputs=[
                    glb_state,
                    input_image,
                    run_normal_check,
                    normal_res,
                    normal_strength,
                    run_depth_check,
                    depth_res,
                    displacement_scale,
                ],
                outputs=[
                    normal_map_img,
                    depth_map_img,
                    enhanced_glb_dl,
                    enhanced_model_3d,
                    enhance_status,
                ],
            )

            unload_btn.click(
                fn=lambda: (unload_models(), "Models unloaded — VRAM freed.")[1],
                inputs=[],
                outputs=[enhance_status],
            )

        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("Settings"):

            def get_vram_status():
                lines = []
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated() / 1024**3
                    reserv = torch.cuda.memory_reserved() / 1024**3
                    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    free = total - reserv
                    lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
                    lines.append(f"VRAM total:     {total:.1f} GB")
                    lines.append(f"VRAM allocated: {alloc:.1f} GB")
                    lines.append(f"VRAM reserved:  {reserv:.1f} GB")
                    lines.append(f"VRAM free:      {free:.1f} GB")
                else:
                    lines.append("No CUDA device available.")
                lines.append("")
                lines.append("Loaded models:")
                lines.append(
                    f"  TripoSG pipeline: {'✓ loaded' if _triposg_pipe is not None else '○ not loaded'}"
                )
                lines.append(
                    f"  RMBG-{_rmbg_version or '?'}:        {'✓ loaded' if _rmbg_net is not None else '○ not loaded'}"
                )
                lines.append(
                    f"  StableNormal:     {'✓ loaded' if _enh_mod._normal_pipe is not None else '○ not loaded'}"
                )
                lines.append(
                    f"  Depth-Anything:   {'✓ loaded' if _enh_mod._depth_pipe is not None else '○ not loaded'}"
                )
                return "\n".join(lines)

            def preload_triposg():
                try:
                    load_triposg()
                    return get_vram_status()
                except Exception as e:
                    return f"Preload failed:\n{traceback.format_exc()}"

            def unload_triposg():
                global _triposg_pipe, _rmbg_net
                with _model_load_lock:
                    if _triposg_pipe is not None:
                        _triposg_pipe.to("cpu")
                        del _triposg_pipe
                        _triposg_pipe = None
                    if _rmbg_net is not None:
                        _rmbg_net.to("cpu")
                        del _rmbg_net
                        _rmbg_net = None
                torch.cuda.empty_cache()
                return get_vram_status()

            def unload_enhancement():
                unload_models()
                return get_vram_status()

            def unload_all():
                unload_triposg()
                unload_models()
                return get_vram_status()

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### VRAM Management")
                    preload_btn = gr.Button(
                        "Preload TripoSG + RMBG to VRAM", variant="primary"
                    )
                    unload_triposg_btn = gr.Button("Unload TripoSG / RMBG")
                    unload_enh_btn = gr.Button(
                        "Unload Enhancement Models (StableNormal / Depth)"
                    )
                    unload_all_btn = gr.Button("Unload All Models", variant="stop")
                    refresh_btn = gr.Button("Refresh Status")

                with gr.Column(scale=1):
                    gr.Markdown("### GPU Status")
                    vram_status = gr.Textbox(
                        label="",
                        lines=12,
                        interactive=False,
                        value="Click Refresh to check VRAM status.",
                    )

            preload_btn.click(fn=preload_triposg, inputs=[], outputs=[vram_status])
            unload_triposg_btn.click(
                fn=unload_triposg, inputs=[], outputs=[vram_status]
            )
            unload_enh_btn.click(
                fn=unload_enhancement, inputs=[], outputs=[vram_status]
            )
            unload_all_btn.click(fn=unload_all, inputs=[], outputs=[vram_status])
            refresh_btn.click(fn=get_vram_status, inputs=[], outputs=[vram_status])

        # ── run_all wiring (after Rig tab so all components are defined) ──
        run_all_btn.click(
            fn=run_full_pipeline,
            inputs=[
                input_image,
                remove_bg_check,
                num_steps,
                guidance,
                seed,
                face_count,
                variant,
                tex_seed,
                enhance_face_check,
                rembg_threshold,
                rembg_erode,
                export_fbx_check,
            ],
            outputs=[
                glb_state,
                download_file,
                multiview_img,
                rig_glb_dl,
                rig_fbx_dl,
                status,
            ],
        ).then(
            fn=lambda p: (p, p) if p else (None, None),
            inputs=[glb_state],
            outputs=[model_3d, download_file],
        )

    # ── Hidden API endpoints — use invisible Gallery (State is stripped from API in Gradio 6) ──
    _api_render_gallery = gr.Gallery(visible=False)
    _api_swap_gallery = gr.Gallery(visible=False)

    def _render_last():
        path = _last_glb_path or str(TMP_DIR / "triposg_textured.glb")
        return render_views(path)

    _hs_emb_input = gr.Textbox(visible=False)

    gr.Button(visible=False).click(
        fn=_render_last,
        inputs=[],
        outputs=[_api_render_gallery],
        api_name="render_last",
    )
    gr.Button(visible=False).click(
        fn=hyperswap_views,
        inputs=[_hs_emb_input],
        outputs=[_api_swap_gallery],
        api_name="hyperswap_views",
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        allowed_paths=["/tmp"],
        max_threads=4,
        max_file_size="50mb",
    )
