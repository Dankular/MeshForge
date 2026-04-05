import sys
import os
import tempfile
import shutil
import traceback
import json
import random
from pathlib import Path

import cv2
import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent
PIPELINE_DIR = HERE / "pipeline"
CKPT_DIR     = Path(os.environ.get("CKPT_DIR", "/tmp/checkpoints"))
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Add pipeline dir so local overrides (patched files) take priority
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PIPELINE_DIR))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy-loaded models (persist between ZeroGPU calls when Space is warm)
_triposg_pipe  = None
_rmbg_net      = None
_rmbg_version  = None
_last_glb_path = None
_init_seed     = random.randint(0, 2**31 - 1)

ARCFACE_256 = (np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                          [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
               * (256 / 112) + (256 - 112 * (256 / 112)) / 2)

VIEW_NAMES = ["front", "3q_front", "side", "back", "3q_back"]
VIEW_PATHS = [f"/tmp/render_{n}.png" for n in VIEW_NAMES]


# ── Weight download helpers ────────────────────────────────────────────────────

def _ensure_weight(url: str, dest: Path) -> Path:
    """Download a file if not already cached."""
    if not dest.exists():
        import urllib.request
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[weights] Downloading {dest.name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"[weights] Saved → {dest}")
    return dest


def _ensure_ckpts():
    """Download all face-enhancement checkpoints to CKPT_DIR."""
    weights = {
        "hyperswap_1a_256.onnx": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/hyperswap_1a_256.onnx",
        "inswapper_128.onnx":    "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x4plus.pth",
        "GFPGANv1.4.pth":        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    }
    for name, url in weights.items():
        _ensure_weight(url, CKPT_DIR / name)


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_triposg():
    global _triposg_pipe, _rmbg_net, _rmbg_version
    if _triposg_pipe is not None:
        _triposg_pipe.to(DEVICE)
        if _rmbg_net is not None:
            _rmbg_net.to(DEVICE)
        return _triposg_pipe, _rmbg_net

    print("[load_triposg] Loading TripoSG pipeline...")
    from huggingface_hub import snapshot_download
    weights_path = snapshot_download("VAST-AI/TripoSG")

    # TripoSG ships its own pipeline — add to path
    triposg_pkg = Path(weights_path)
    if (triposg_pkg / "triposg").exists():
        sys.path.insert(0, str(triposg_pkg))
    else:
        # Try installed package from the cloned repo (if installed with pip -e)
        import importlib.util
        if importlib.util.find_spec("triposg") is None:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(triposg_pkg), "-q"], check=False)

    from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    _triposg_pipe = TripoSGPipeline.from_pretrained(
        weights_path, torch_dtype=torch.float16
    ).to(DEVICE)

    try:
        from transformers import AutoModelForImageSegmentation
        _rmbg_net = AutoModelForImageSegmentation.from_pretrained(
            "1038lab/RMBG-2.0", trust_remote_code=True, low_cpu_mem_usage=False
        ).to(DEVICE)
        _rmbg_net.eval()
        _rmbg_version = "2.0"
        print("[load_triposg] TripoSG + RMBG-2.0 loaded.")
    except Exception as e:
        print(f"[load_triposg] RMBG-2.0 failed ({e}). BG removal disabled.")
        _rmbg_net = None

    return _triposg_pipe, _rmbg_net


# ── Background removal helper ─────────────────────────────────────────────────

def _remove_bg_rmbg(img_pil, threshold=0.5, erode_px=2):
    if _rmbg_net is None:
        return img_pil
    import torchvision.transforms.functional as TF
    from torchvision import transforms

    img_tensor = transforms.ToTensor()(img_pil.resize((1024, 1024)))
    if _rmbg_version == "2.0":
        img_tensor = TF.normalize(img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)
    else:
        img_tensor = TF.normalize(img_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]).unsqueeze(0)

    with torch.no_grad():
        result = _rmbg_net(img_tensor)

    if isinstance(result, (list, tuple)):
        candidate = result[-1] if _rmbg_version == "2.0" else result[0]
        if isinstance(candidate, (list, tuple)):
            candidate = candidate[0]
    else:
        candidate = result

    mask_tensor = candidate.sigmoid()[0, 0].cpu()
    mask = np.array(transforms.ToPILImage()(mask_tensor).resize(img_pil.size, Image.BILINEAR),
                    dtype=np.float32) / 255.0
    mask = (mask >= threshold).astype(np.float32) * mask
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1,) * 2)
        mask = cv2.erode((mask * 255).astype(np.uint8), kernel).astype(np.float32) / 255.0

    rgb   = np.array(img_pil.convert("RGB"), dtype=np.float32) / 255.0
    alpha = mask[:, :, np.newaxis]
    comp  = (rgb * alpha + 0.5 * (1.0 - alpha) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(comp)


def preview_rembg(input_image, do_remove_bg, threshold, erode_px):
    if input_image is None or not do_remove_bg or _rmbg_net is None:
        return input_image
    try:
        return np.array(_remove_bg_rmbg(Image.fromarray(input_image).convert("RGB"),
                                         threshold=float(threshold), erode_px=int(erode_px)))
    except Exception:
        return input_image


# ── Stage 1: Shape generation ─────────────────────────────────────────────────

@spaces.GPU(duration=180)
def generate_shape(input_image, remove_background, num_steps, guidance_scale,
                   seed, face_count, progress=gr.Progress()):
    if input_image is None:
        return None, "Please upload an image."
    try:
        progress(0.1, desc="Loading TripoSG...")

        # Add TripoSG scripts to path after model download
        from huggingface_hub import snapshot_download
        weights_path = snapshot_download("VAST-AI/TripoSG")
        sys.path.insert(0, weights_path)

        pipe, rmbg_net = load_triposg()

        img = Image.fromarray(input_image).convert("RGB")
        img_path = "/tmp/triposg_input.png"
        img.save(img_path)

        progress(0.5, desc="Generating shape (SDF diffusion)...")
        from scripts.inference_triposg import run_triposg
        mesh = run_triposg(
            pipe=pipe,
            image_input=img_path,
            rmbg_net=rmbg_net if remove_background else None,
            seed=int(seed),
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            faces=int(face_count) if int(face_count) > 0 else -1,
        )

        out_path = "/tmp/triposg_shape.glb"
        mesh.export(out_path)

        # Offload to CPU before next stage
        _triposg_pipe.to("cpu")
        if _rmbg_net is not None:
            _rmbg_net.to("cpu")
        torch.cuda.empty_cache()

        return out_path, "Shape generated!"
    except Exception:
        return None, f"Error:\n{traceback.format_exc()}"


# ── Stage 2: Texture ──────────────────────────────────────────────────────────

@spaces.GPU(duration=300)
def apply_texture(glb_path, input_image, remove_background, variant, tex_seed,
                  enhance_face, rembg_threshold=0.5, rembg_erode=2,
                  progress=gr.Progress()):
    if glb_path is None:
        glb_path = "/tmp/triposg_shape.glb"
    if not os.path.exists(glb_path):
        return None, None, "Generate a shape first."
    if input_image is None:
        return None, None, "Please upload an image."
    try:
        progress(0.1, desc="Preprocessing image...")
        img = Image.fromarray(input_image).convert("RGB")
        face_ref_path = "/tmp/triposg_face_ref.png"
        img.save(face_ref_path)

        if remove_background and _rmbg_net is not None:
            img = _remove_bg_rmbg(img, threshold=float(rembg_threshold), erode_px=int(rembg_erode))

        img = img.resize((768, 768), Image.LANCZOS)
        img_path = "/tmp/tex_input_768.png"
        img.save(img_path)

        out_dir = "/tmp/tex_out"
        os.makedirs(out_dir, exist_ok=True)

        # ── Run MV-Adapter in-process ─────────────────────────────────────
        progress(0.3, desc="Loading MV-Adapter pipeline...")
        import importlib
        from huggingface_hub import snapshot_download

        mvadapter_weights = snapshot_download("huanngzh/mv-adapter")

        # Resolve SD pipeline
        if variant == "sdxl":
            from diffusers import StableDiffusionXLPipeline
            sd_id = "stabilityai/stable-diffusion-xl-base-1.0"
        else:
            from diffusers import StableDiffusionPipeline
            sd_id = "stabilityai/stable-diffusion-2-1-base"

        from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
        from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
        from mvadapter.utils import get_orthogonal_camera, get_ipadapter_image
        import torchvision.transforms.functional as TF

        progress(0.4, desc=f"Running MV-Adapter ({variant})...")

        pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(
            sd_id,
            torch_dtype=torch.float16,
        ).to(DEVICE)

        pipe.init_adapter(
            image_encoder_path="openai/clip-vit-large-patch14",
            ipa_weight_path=os.path.join(mvadapter_weights, "mvadapter_i2mv_sdxl.safetensors"),
            adapter_tokens=256,
        )

        ref_pil = Image.open(img_path).convert("RGB")
        cameras = get_orthogonal_camera(
            elevation_deg=[0, 0, 0, 0, 0, 0],
            distance=[1.8] * 6,
            left=-0.55, right=0.55, bottom=-0.55, top=0.55,
            azimuth_deg=[x - 90 for x in [0, 45, 90, 135, 180, 270]],
            device=DEVICE,
        )

        with torch.autocast(DEVICE):
            out = pipe(
                image=ref_pil,
                height=768, width=768,
                num_images_per_prompt=6,
                guidance_scale=3.0,
                num_inference_steps=30,
                generator=torch.Generator(device=DEVICE).manual_seed(int(tex_seed)),
                cameras=cameras,
            )

        mv_grid = out.images  # list of 6 PIL images
        grid_w  = mv_grid[0].width * len(mv_grid)
        mv_pil  = Image.new("RGB", (grid_w, mv_grid[0].height))
        for i, v in enumerate(mv_grid):
            mv_pil.paste(v, (i * mv_grid[0].width, 0))
        mv_path = os.path.join(out_dir, "multiview.png")
        mv_pil.save(mv_path)

        # Offload before face-enhance (saves VRAM)
        del pipe
        torch.cuda.empty_cache()

        # ── Face enhancement ─────────────────────────────────────────────
        if enhance_face:
            progress(0.75, desc="Running face enhancement...")
            _ensure_ckpts()
            try:
                from pipeline.face_enhance import enhance_multiview
                enh_path = os.path.join(out_dir, "multiview_enhanced.png")
                enhance_multiview(
                    multiview_path=mv_path,
                    reference_path=face_ref_path,
                    output_path=enh_path,
                    ckpt_dir=str(CKPT_DIR),
                )
                mv_path = enh_path
            except Exception as _fe:
                print(f"[apply_texture] face enhance failed: {_fe}")

        # ── Bake textures onto mesh ─────────────────────────────────────
        progress(0.85, desc="Baking UV texture onto mesh...")
        from mvadapter.utils.mesh_utils import (
            NVDiffRastContextWrapper, load_mesh, bake_texture,
        )

        ctx  = NVDiffRastContextWrapper(device=DEVICE, context_type="cuda")
        mesh = load_mesh(glb_path, rescale=True, device=DEVICE)
        tex_pil = Image.open(mv_path)

        baked = bake_texture(ctx, mesh, tex_pil, cameras=cameras, height=1024, width=1024)
        out_glb = os.path.join(out_dir, "textured_shaded.glb")
        baked.export(out_glb)

        final_path = "/tmp/triposg_textured.glb"
        shutil.copy(out_glb, final_path)

        global _last_glb_path
        _last_glb_path = final_path

        torch.cuda.empty_cache()
        return final_path, mv_path, "Texture applied!"
    except Exception:
        return None, None, f"Error:\n{traceback.format_exc()}"


# ── Stage 3a: SKEL Anatomy ────────────────────────────────────────────────────

@spaces.GPU(duration=90)
def gradio_tpose(glb_state_path, export_skel_flag, progress=gr.Progress()):
    try:
        glb = glb_state_path or _last_glb_path or "/tmp/triposg_textured.glb"
        if not os.path.exists(glb):
            return None, None, "No GLB found — run Generate + Texture first."

        progress(0.1, desc="YOLO pose detection + rigging...")
        from pipeline.rig_yolo import rig_yolo
        out_dir = "/tmp/rig_out"
        os.makedirs(out_dir, exist_ok=True)
        rigged, _rigged_skel = rig_yolo(glb, os.path.join(out_dir, "anatomy_rigged.glb"), debug_dir=None)

        bones = None
        if export_skel_flag:
            progress(0.7, desc="Generating SKEL bone mesh...")
            from pipeline.tpose_smpl import export_skel_bones
            bones = export_skel_bones(torch.zeros(10), "/tmp/tposed_bones.glb", gender="male")

        status = f"Rigged surface: {os.path.getsize(rigged)//1024} KB"
        if bones:
            status += f"\nSKEL bone mesh: {os.path.getsize(bones)//1024} KB"
        elif export_skel_flag:
            status += "\nSKEL bone mesh: failed (check logs)"

        torch.cuda.empty_cache()
        return rigged, bones, status
    except Exception:
        return None, None, f"Error:\n{traceback.format_exc()}"


# ── Stage 3b: Rig & Export ────────────────────────────────────────────────────

@spaces.GPU(duration=180)
def gradio_rig(glb_state_path, export_fbx_flag, mdm_prompt, mdm_n_frames,
               progress=gr.Progress()):
    try:
        from pipeline.rig_yolo import rig_yolo
        from pipeline.rig_stage import export_fbx

        glb = glb_state_path or _last_glb_path or "/tmp/triposg_textured.glb"
        if not os.path.exists(glb):
            return None, None, None, "No GLB found — run Generate + Texture first.", None, None, None

        out_dir = "/tmp/rig_out"
        os.makedirs(out_dir, exist_ok=True)

        progress(0.1, desc="YOLO pose detection + rigging...")
        rigged, rigged_skel = rig_yolo(glb, os.path.join(out_dir, "rigged.glb"),
                                        debug_dir=os.path.join(out_dir, "debug"))

        fbx = None
        if export_fbx_flag:
            progress(0.7, desc="Exporting FBX...")
            fbx_path = os.path.join(out_dir, "rigged.fbx")
            fbx = fbx_path if export_fbx(rigged, fbx_path) else None

        animated = None
        if mdm_prompt.strip():
            progress(0.75, desc="Generating MDM animation...")
            from pipeline.rig_stage import run_rig_pipeline
            mdm_result = run_rig_pipeline(
                glb_path=glb,
                reference_image_path="/tmp/triposg_face_ref.png",
                out_dir=out_dir,
                device=DEVICE,
                export_fbx_flag=False,
                mdm_prompt=mdm_prompt.strip(),
                mdm_n_frames=int(mdm_n_frames),
            )
            animated = mdm_result.get("animated_glb")

        parts = ["Rigged: " + os.path.basename(rigged)]
        if fbx:     parts.append("FBX: " + os.path.basename(fbx))
        if animated: parts.append("Animation: " + os.path.basename(animated))

        torch.cuda.empty_cache()
        return rigged, animated, fbx, "  |  ".join(parts), rigged, rigged, rigged_skel
    except Exception:
        return None, None, None, f"Error:\n{traceback.format_exc()}", None, None, None


# ── Stage 4: Surface enhancement ─────────────────────────────────────────────

@spaces.GPU(duration=120)
def gradio_enhance(glb_path, ref_img_np, do_normal, norm_res, norm_strength,
                   do_depth, dep_res, disp_scale):
    if not glb_path:
        yield None, None, None, None, "No GLB loaded — run Generate first."
        return
    if ref_img_np is None:
        yield None, None, None, None, "No reference image — run Generate first."
        return
    try:
        from pipeline.enhance_surface import (
            run_stable_normal, run_depth_anything,
            bake_normal_into_glb, bake_depth_as_occlusion,
        )
        import pipeline.enhance_surface as _enh_mod

        ref_pil  = Image.fromarray(ref_img_np.astype(np.uint8))
        out_path = glb_path.replace(".glb", "_enhanced.glb")
        shutil.copy2(glb_path, out_path)
        normal_out = depth_out = None
        log = []

        if do_normal:
            log.append("[StableNormal] Running...")
            yield None, None, None, None, "\n".join(log)
            normal_out = run_stable_normal(ref_pil, resolution=norm_res)
            out_path = bake_normal_into_glb(out_path, normal_out, out_path,
                                             normal_strength=norm_strength)
            log.append(f"[StableNormal] Done → normalTexture (strength {norm_strength})")
            yield normal_out, depth_out, None, None, "\n".join(log)

        if do_depth:
            log.append("[Depth-Anything] Running...")
            yield normal_out, depth_out, None, None, "\n".join(log)
            depth_out = run_depth_anything(ref_pil, resolution=dep_res)
            out_path  = bake_depth_as_occlusion(out_path, depth_out, out_path,
                                                 displacement_scale=disp_scale)
            log.append(f"[Depth-Anything] Done → occlusionTexture (scale {disp_scale})")
            yield normal_out, depth_out.convert("L").convert("RGB"), None, None, "\n".join(log)

        torch.cuda.empty_cache()
        log.append("Enhancement complete.")
        yield normal_out, (depth_out.convert("L").convert("RGB") if depth_out else None), out_path, out_path, "\n".join(log)

    except Exception:
        yield None, None, None, None, f"Error:\n{traceback.format_exc()}"


# ── Render views ──────────────────────────────────────────────────────────────

@spaces.GPU(duration=60)
def render_views(glb_file):
    if not glb_file:
        return []
    glb_path = glb_file if isinstance(glb_file, str) else (glb_file.get("path") if isinstance(glb_file, dict) else str(glb_file))
    if not glb_path or not os.path.exists(glb_path):
        return []
    try:
        from mvadapter.utils.mesh_utils import (
            NVDiffRastContextWrapper, load_mesh, render, get_orthogonal_camera,
        )
        ctx  = NVDiffRastContextWrapper(device="cuda", context_type="cuda")
        mesh = load_mesh(glb_path, rescale=True, device="cuda")
        cams = get_orthogonal_camera(
            elevation_deg=[0]*5, distance=[1.8]*5,
            left=-0.55, right=0.55, bottom=-0.55, top=0.55,
            azimuth_deg=[x - 90 for x in [0, 45, 90, 180, 315]],
            device="cuda",
        )
        out = render(ctx, mesh, cams, height=1024, width=768, render_attr=True, normal_background=0.0)
        save_dir = os.path.dirname(glb_path)
        results  = []
        for i, name in enumerate(VIEW_NAMES):
            arr  = (out.attr[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            path = os.path.join(save_dir, f"render_{name}.png")
            Image.fromarray(arr).save(path)
            results.append((path, name))
        torch.cuda.empty_cache()
        return results
    except Exception:
        print(f"render_views FAILED:\n{traceback.format_exc()}")
        return []


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_full_pipeline(input_image, remove_background, num_steps, guidance, seed, face_count,
                      variant, tex_seed, enhance_face, rembg_threshold, rembg_erode,
                      export_fbx, mdm_prompt, mdm_n_frames, progress=gr.Progress()):
    progress(0.0, desc="Stage 1/3: Generating shape...")
    glb, status = generate_shape(input_image, remove_background, num_steps, guidance, seed, face_count)
    if not glb:
        return None, None, None, None, None, None, status

    progress(0.33, desc="Stage 2/3: Applying texture...")
    glb, mv_img, status = apply_texture(glb, input_image, remove_background, variant, tex_seed,
                                         enhance_face, rembg_threshold, rembg_erode)
    if not glb:
        return None, None, None, None, None, None, status

    progress(0.66, desc="Stage 3/3: Rigging + animation...")
    rigged, animated, fbx, rig_status, _, _, _ = gradio_rig(glb, export_fbx, mdm_prompt, mdm_n_frames)

    progress(1.0, desc="Pipeline complete!")
    return glb, glb, mv_img, rigged, animated, fbx, f"[Texture] {status}\n[Rig] {rig_status}"


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Image2Model", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Image2Model — Portrait to Rigged 3D Mesh")
    glb_state = gr.State(None)

    with gr.Tabs():

        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image    = gr.Image(label="Input Image", type="numpy")
                    remove_bg_check = gr.Checkbox(label="Remove Background", value=True)
                    with gr.Row():
                        rembg_threshold = gr.Slider(0.1, 0.95, value=0.5, step=0.05,
                                                    label="BG Threshold")
                        rembg_erode     = gr.Slider(0, 8, value=2, step=1,
                                                    label="Edge Erode (px)")

                    with gr.Accordion("Shape Settings", open=True):
                        num_steps  = gr.Slider(20, 100, value=50, step=5,  label="Inference Steps")
                        guidance   = gr.Slider(1.0, 20.0, value=7.0, step=0.5, label="Guidance Scale")
                        seed       = gr.Number(value=_init_seed, label="Seed", precision=0)
                        face_count = gr.Number(value=0, label="Max Faces (0 = unlimited)", precision=0)

                    with gr.Accordion("Texture Settings", open=True):
                        variant    = gr.Radio(["sdxl", "sd21"], value="sdxl",
                                              label="Model (sdxl = quality, sd21 = less VRAM)")
                        tex_seed   = gr.Number(value=_init_seed, label="Texture Seed", precision=0)
                        enhance_face_check = gr.Checkbox(
                            label="Enhance Face (HyperSwap + RealESRGAN)", value=True)

                    with gr.Row():
                        shape_btn   = gr.Button("Generate Shape",  variant="primary",   scale=2, interactive=False)
                        texture_btn = gr.Button("Apply Texture",   variant="secondary", scale=2)
                        render_btn  = gr.Button("Render Views",    variant="secondary", scale=1)
                    run_all_btn = gr.Button("▶ Run Full Pipeline", variant="primary", interactive=False)

                with gr.Column(scale=1):
                    rembg_preview  = gr.Image(label="BG Removed Preview", type="numpy", interactive=False)
                    status         = gr.Textbox(label="Status", lines=3, interactive=False)
                    model_3d       = gr.Model3D(label="3D Preview", clear_color=[0.9, 0.9, 0.9, 1.0])
                    download_file  = gr.File(label="Download GLB")
                    multiview_img  = gr.Image(label="Multiview", type="filepath", interactive=False)

            render_gallery = gr.Gallery(label="Rendered Views", columns=5, height=300)

            _rembg_inputs  = [input_image, remove_bg_check, rembg_threshold, rembg_erode]
            _pipeline_btns = [shape_btn, run_all_btn]

            input_image.upload(
                fn=lambda: (gr.update(interactive=True), gr.update(interactive=True)),
                inputs=[], outputs=_pipeline_btns,
            )
            input_image.clear(
                fn=lambda: (gr.update(interactive=False), gr.update(interactive=False)),
                inputs=[], outputs=_pipeline_btns,
            )
            input_image.upload(fn=preview_rembg,      inputs=_rembg_inputs, outputs=[rembg_preview])
            remove_bg_check.change(fn=preview_rembg,  inputs=_rembg_inputs, outputs=[rembg_preview])
            rembg_threshold.release(fn=preview_rembg, inputs=_rembg_inputs, outputs=[rembg_preview])
            rembg_erode.release(fn=preview_rembg,     inputs=_rembg_inputs, outputs=[rembg_preview])

            shape_btn.click(
                fn=generate_shape,
                inputs=[input_image, remove_bg_check, num_steps, guidance, seed, face_count],
                outputs=[glb_state, status],
            ).then(
                fn=lambda p: (p, p) if p else (None, None),
                inputs=[glb_state], outputs=[model_3d, download_file],
            )

            texture_btn.click(
                fn=apply_texture,
                inputs=[glb_state, input_image, remove_bg_check, variant, tex_seed,
                        enhance_face_check, rembg_threshold, rembg_erode],
                outputs=[glb_state, multiview_img, status],
            ).then(
                fn=lambda p: (p, p) if p else (None, None),
                inputs=[glb_state], outputs=[model_3d, download_file],
            )

            render_btn.click(fn=render_views, inputs=[download_file], outputs=[render_gallery])

        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("Rig & Export"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Step 1 — SKEL Anatomy Layer")
                    tpose_skel_check  = gr.Checkbox(label="Export SKEL bone mesh", value=False)
                    tpose_btn         = gr.Button("Rig + SKEL Anatomy", variant="secondary")
                    tpose_status      = gr.Textbox(label="Anatomy Status", lines=3, interactive=False)
                    with gr.Row():
                        tpose_surface_dl = gr.File(label="Rigged Surface GLB")
                        tpose_bones_dl   = gr.File(label="SKEL Bone Mesh GLB")

                    gr.Markdown("---")
                    gr.Markdown("### Step 2 — Rig & Export")
                    export_fbx_check  = gr.Checkbox(label="Export FBX (requires Blender)", value=True)
                    mdm_prompt_box    = gr.Textbox(label="Motion Prompt (MDM)",
                                                   placeholder="a person walks forward", value="")
                    mdm_frames_slider = gr.Slider(60, 300, value=120, step=30,
                                                  label="Animation Frames (at 20 fps)")
                    rig_btn           = gr.Button("Rig Mesh", variant="primary")

                with gr.Column(scale=2):
                    rig_status      = gr.Textbox(label="Rig Status", lines=4, interactive=False)
                    show_skel_check = gr.Checkbox(label="Show Skeleton", value=False)
                    rig_model_3d    = gr.Model3D(label="Preview", clear_color=[0.9, 0.9, 0.9, 1.0])
                    with gr.Row():
                        rig_glb_dl      = gr.File(label="Download Rigged GLB")
                        rig_animated_dl = gr.File(label="Download Animated GLB")
                        rig_fbx_dl      = gr.File(label="Download FBX")

            rigged_base_state = gr.State(None)
            skel_glb_state    = gr.State(None)

            tpose_btn.click(
                fn=gradio_tpose,
                inputs=[glb_state, tpose_skel_check],
                outputs=[tpose_surface_dl, tpose_bones_dl, tpose_status],
            ).then(
                fn=lambda p: (p["path"] if isinstance(p, dict) else p) if p else None,
                inputs=[tpose_surface_dl], outputs=[rig_model_3d],
            )

            rig_btn.click(
                fn=gradio_rig,
                inputs=[glb_state, export_fbx_check, mdm_prompt_box, mdm_frames_slider],
                outputs=[rig_glb_dl, rig_animated_dl, rig_fbx_dl, rig_status,
                         rig_model_3d, rigged_base_state, skel_glb_state],
            )

            show_skel_check.change(
                fn=lambda show, base, skel: skel if (show and skel) else base,
                inputs=[show_skel_check, rigged_base_state, skel_glb_state],
                outputs=[rig_model_3d],
            )

        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("Enhancement"):
            gr.Markdown("**Surface Enhancement** — bakes normal + depth maps into the GLB as PBR textures.")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### StableNormal")
                    run_normal_check   = gr.Checkbox(label="Run StableNormal", value=True)
                    normal_res         = gr.Slider(512, 1024, value=768, step=128, label="Resolution")
                    normal_strength    = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Normal Strength")

                    gr.Markdown("### Depth-Anything V2")
                    run_depth_check    = gr.Checkbox(label="Run Depth-Anything V2", value=True)
                    depth_res          = gr.Slider(512, 1024, value=768, step=128, label="Resolution")
                    displacement_scale = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Displacement Scale")

                    enhance_btn = gr.Button("Run Enhancement", variant="primary")

                with gr.Column(scale=2):
                    enhance_status    = gr.Textbox(label="Status", lines=5, interactive=False)
                    with gr.Row():
                        normal_map_img = gr.Image(label="Normal Map", type="pil")
                        depth_map_img  = gr.Image(label="Depth Map", type="pil")
                    enhanced_glb_dl   = gr.File(label="Download Enhanced GLB")
                    enhanced_model_3d = gr.Model3D(label="Preview", clear_color=[0.9, 0.9, 0.9, 1.0])

            enhance_btn.click(
                fn=gradio_enhance,
                inputs=[glb_state, input_image,
                        run_normal_check, normal_res, normal_strength,
                        run_depth_check, depth_res, displacement_scale],
                outputs=[normal_map_img, depth_map_img,
                         enhanced_glb_dl, enhanced_model_3d, enhance_status],
            )

        # ── Run All wiring ────────────────────────────────────────────────
        run_all_btn.click(
            fn=run_full_pipeline,
            inputs=[
                input_image, remove_bg_check, num_steps, guidance, seed, face_count,
                variant, tex_seed, enhance_face_check, rembg_threshold, rembg_erode,
                export_fbx_check, mdm_prompt_box, mdm_frames_slider,
            ],
            outputs=[glb_state, download_file, multiview_img,
                     rig_glb_dl, rig_animated_dl, rig_fbx_dl, status],
        ).then(
            fn=lambda p: (p, p) if p else (None, None),
            inputs=[glb_state], outputs=[model_3d, download_file],
        )


if __name__ == "__main__":
    demo.launch()
