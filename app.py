import sys
import os
import subprocess
import tempfile
import shutil
import traceback
import json
import random
from pathlib import Path

sys.path.insert(0, "/root")
from enhance_surface import (
    run_stable_normal, run_depth_anything,
    bake_normal_into_glb, bake_depth_as_occlusion,
    unload_models,
)
import enhance_surface as _enh_mod

import cv2
import gradio as gr
import torch
import numpy as np
from PIL import Image

PYTHON = "/root/miniconda/envs/triposg/bin/python"
TRIPOSG_DIR = "/root/TripoSG"
MVADAPTER_DIR = "/root/MV-Adapter"
CKPT_DIR = "/root/MV-Adapter/checkpoints"
os.environ["GRADIO_CDN_BACKEND_ENABLED"] = "False"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy-loaded models (kept in memory between calls)
_triposg_pipe = None
_rmbg_net = None
_last_glb_path = None
_hyperswap_sess = None
_gfpgan_restorer = None
_rmbg_version   = None   # "2.0" or "1.4"
_init_seed = random.randint(0, 2**31 - 1)

import threading
_model_load_lock = threading.Lock()

ARCFACE_256 = (np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                          [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
               * (256 / 112) + (256 - 112 * (256 / 112)) / 2)

VIEW_NAMES = ["front", "3q_front", "side", "back", "3q_back"]
VIEW_PATHS = [f"/tmp/render_{n}.png" for n in VIEW_NAMES]


def load_triposg():
    global _triposg_pipe, _rmbg_net, _rmbg_version
    if _triposg_pipe is not None:
        _triposg_pipe.to(DEVICE)
        if _rmbg_net is not None: _rmbg_net.to(DEVICE)
        return _triposg_pipe, _rmbg_net
    print("Loading TripoSG pipeline...")
    sys.path.insert(0, TRIPOSG_DIR)
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    from huggingface_hub import snapshot_download

    weights_path = snapshot_download("VAST-AI/TripoSG")
    _triposg_pipe = TripoSGPipeline.from_pretrained(
        weights_path, torch_dtype=torch.float16
    ).to(DEVICE)

    # Load RMBG-2.0 from public mirror (briaai/RMBG-2.0 is gated)
    try:
        from transformers import AutoModelForImageSegmentation
        _rmbg_net = AutoModelForImageSegmentation.from_pretrained(
            "1038lab/RMBG-2.0", trust_remote_code=True, low_cpu_mem_usage=False
        ).to(DEVICE)
        _rmbg_net.eval()
        _rmbg_version = "2.0"
        print("TripoSG + RMBG-2.0 loaded (1038lab mirror).")
    except Exception as e:
        print(f"RMBG-2.0 failed ({e}). Background removal disabled.")
        _rmbg_net = None
        _rmbg_version = None

    return _triposg_pipe, _rmbg_net


def load_gfpgan():
    global _gfpgan_restorer
    if _gfpgan_restorer is not None:
        return _gfpgan_restorer
    try:
        from gfpgan import GFPGANer
        model_path = os.path.join(CKPT_DIR, "GFPGANv1.4.pth")
        if not os.path.exists(model_path):
            print(f"[GFPGAN] Not found at {model_path}")
            return None
        _gfpgan_restorer = GFPGANer(
            model_path=model_path, upscale=1, arch="clean",
            channel_multiplier=2, bg_upsampler=None,
        )
        print("[GFPGAN] Loaded GFPGANv1.4 (upscale=1 for rendered views)")
        return _gfpgan_restorer
    except Exception as e:
        print(f"[GFPGAN] Load failed: {e}")
        return None


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
        progress(0.1, desc="Loading TripoSG...")
        sys.path.insert(0, TRIPOSG_DIR)
        from scripts.inference_triposg import run_triposg
        from scripts.image_process import prepare_image

        pipe, rmbg_net = load_triposg()

        img = Image.fromarray(input_image).convert("RGB")
        img_path = "/tmp/triposg_input.png"
        img.save(img_path)

        progress(0.5, desc="Generating shape (SDF diffusion)...")
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

        # Offload models to CPU to free VRAM for texture subprocess
        _triposg_pipe.to("cpu")
        _rmbg_net.to("cpu")
        torch.cuda.empty_cache()

        return out_path, "Shape generated!"
    except Exception:
        return None, f"Error:\n{traceback.format_exc()}"


def _remove_bg_rmbg(img_pil, threshold=0.5, erode_px=2):
    """
    Remove background using BriaRMBG, return RGB composited on neutral gray.
    threshold : float [0,1] — mask confidence cutoff; raise to cut more background
    erode_px  : int        — shrink mask by this many pixels to remove fringe
    """
    import torch
    import numpy as np
    import torchvision.transforms.functional as TF
    from torchvision import transforms

    if _rmbg_net is None:
        return img_pil  # BG removal unavailable
    _rmbg_net.to("cpu").eval()

    img_tensor = transforms.ToTensor()(img_pil.resize((1024, 1024)))
    if _rmbg_version == "2.0":
        # RMBG-2.0: ImageNet normalisation
        img_tensor = TF.normalize(img_tensor,
                                   [0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225]).unsqueeze(0)
    else:
        # RMBG-1.4: 0.5/1.0 normalisation
        img_tensor = TF.normalize(img_tensor, [0.5, 0.5, 0.5],
                                   [1.0, 1.0, 1.0]).unsqueeze(0)
    with torch.no_grad():
        result = _rmbg_net(img_tensor)
    # RMBG-2.0: result is a list of tensors, last = final prediction
    # RMBG-1.4 (BriaRMBG/U2Net): result is a tuple of tensors, first = main output
    if isinstance(result, (list, tuple)):
        candidate = result[-1] if _rmbg_version == "2.0" else result[0]
        # If candidate is still a list/tuple (nested), unwrap one level
        if isinstance(candidate, (list, tuple)):
            candidate = candidate[0]
    else:
        candidate = result
    mask_tensor = candidate.sigmoid()[0, 0].cpu()
    mask = np.array(transforms.ToPILImage()(mask_tensor).resize(
        img_pil.size, Image.BILINEAR
    ), dtype=np.float32) / 255.0

    # Hard threshold — pixels below cutoff become fully transparent
    mask = (mask >= threshold).astype(np.float32) * mask

    # Erode mask edges to remove background fringe
    if erode_px > 0:
        import cv2 as _cv2
        kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (erode_px * 2 + 1,) * 2)
        mask = _cv2.erode((mask * 255).astype(np.uint8), kernel).astype(np.float32) / 255.0

    rgb = np.array(img_pil.convert("RGB"), dtype=np.float32) / 255.0
    alpha = mask[:, :, np.newaxis]
    composited = rgb * alpha + 0.5 * (1.0 - alpha)
    composited = (composited * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(composited)


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
        glb_path = "/tmp/triposg_shape.glb"
    if not os.path.exists(glb_path):
        return None, None, "Generate a shape first."
    if input_image is None:
        return None, None, "Please upload an image."
    try:
        progress(0.1, desc="Preprocessing image...")
        img = Image.fromarray(input_image).convert("RGB")

        # Save original photo before any processing — used as HyperSwap face source
        face_ref_path = "/tmp/triposg_face_ref.png"
        img.save(face_ref_path)

        if remove_background and _rmbg_net is not None:
            img = _remove_bg_rmbg(img, threshold=float(rembg_threshold), erode_px=int(rembg_erode))

        img = img.resize((768, 768), Image.LANCZOS)
        img_path = "/tmp/tex_input.png"
        img.save(img_path)

        # Free GPU memory before launching SDXL subprocess (~15 GB peak)
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        out_dir = "/tmp/tex_out"
        os.makedirs(out_dir, exist_ok=True)
        out_name = "textured"

        cmd = [
            PYTHON, "-m", "scripts.texture_i2tex",
            "--mesh", glb_path,
            "--image", img_path,
            "--save_dir", out_dir,
            "--save_name", out_name,
            "--variant", variant,
            "--seed", str(int(tex_seed)),
            "--device", DEVICE,
            "--reference_conditioning_scale", "1.5",
            "--text", "photorealistic person, detailed skin texture, realistic clothing",
            "--preprocess_mesh",
        ]
        if enhance_face:
            cmd += [
                "--enhance_face",
                "--face_checkpoints", CKPT_DIR,
                "--gfpgan_upscale", "2",
                "--face_reference", face_ref_path,
            ]

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
        mv_png = f"{out_dir}/{out_name}_face_enhanced.png" if enhance_face else f"{out_dir}/{out_name}.png"
        if not os.path.exists(mv_png):
            mv_png = f"{out_dir}/{out_name}.png"

        if os.path.exists(out_glb):
            final_path = "/tmp/triposg_textured.glb"
            shutil.copy(out_glb, final_path)
            mv_out = mv_png if os.path.exists(mv_png) else None
            label = "Texture applied" + (" + face enhanced!" if enhance_face else "!")
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
        composited = _remove_bg_rmbg(img, threshold=float(threshold), erode_px=int(erode_px))
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
    print(f"render_views: loading {glb_path} ({os.path.getsize(glb_path)//1024}KB)")
    try:
        sys.path.insert(0, MVADAPTER_DIR)
        print("render_views: importing nvdiffrast utils...")
        from mvadapter.utils.mesh_utils import (
            NVDiffRastContextWrapper, load_mesh, render, get_orthogonal_camera,
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
            left=-0.55, right=0.55, bottom=-0.55, top=0.55,
            azimuth_deg=azimuth_deg,
            device=device,
        )

        render_out = render(
            ctx, mesh, cameras,
            height=1024, width=768,
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
            _hyperswap_sess = ort.InferenceSession(hs_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
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
                M, _ = cv2.estimateAffinePartial2D(face.kps, ARCFACE_256,
                                                    method=cv2.RANSAC, ransacReprojThreshold=100)
                H, W = bgr.shape[:2]
                aligned = cv2.warpAffine(bgr, M, (256, 256), flags=cv2.INTER_LINEAR)
                t = ((aligned.astype(np.float32) / 255 - 0.5) / 0.5)[:, :, ::-1].copy().transpose(2, 0, 1)[None]
                out, mask = _hyperswap_sess.run(None, {
                    "source": embedding.reshape(1, -1),
                    "target": t,
                })
                out_bgr = (((out[0].transpose(1, 2, 0) + 1) / 2 * 255)
                           .clip(0, 255).astype(np.uint8))[:, :, ::-1].copy()
                m = (mask[0, 0] * 255).clip(0, 255).astype(np.uint8)
                Mi = cv2.invertAffineTransform(M)
                of = cv2.warpAffine(out_bgr, Mi, (W, H), flags=cv2.INTER_LINEAR)
                mf = cv2.warpAffine(m, Mi, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32)[:, :, None] / 255
                swapped = (of * mf + bgr * (1 - mf)).clip(0, 255).astype(np.uint8)

                # GFPGAN face restoration — use the SAME bbox from the already-detected face
                # (avoids re-running InsightFace at det_thresh=0.1 which can latch onto skin/body)
                restorer = load_gfpgan()
                if restorer is not None:
                    b = face.bbox.astype(int)
                    h2, w2 = swapped.shape[:2]
                    pad = 0.35
                    bw2, bh2 = b[2]-b[0], b[3]-b[1]
                    cx1 = max(0, b[0]-int(bw2*pad)); cy1 = max(0, b[1]-int(bh2*pad))
                    cx2 = min(w2, b[2]+int(bw2*pad)); cy2 = min(h2, b[3]+int(bh2*pad))
                    crop = swapped[cy1:cy2, cx1:cx2]
                    try:
                        _, _, rest = restorer.enhance(
                            crop, has_aligned=False, only_center_face=True,
                            paste_back=True, weight=0.5)
                        if rest is not None and rest.shape[:2] == (cy2-cy1, cx2-cx1):
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
        glb = glb_state_path or _last_glb_path or "/tmp/triposg_textured.glb"
        if not os.path.exists(glb):
            return None, None, "No GLB found — run Generate Shape + Apply Texture first."

        # Surface: YOLO-rig (replaces broken inverse-LBS T-pose)
        progress(0.1, desc="YOLO pose detection + rigging surface ...")
        sys.path.insert(0, "/root")
        from rig_yolo import rig_yolo
        out_dir = "/tmp/rig_out"
        os.makedirs(out_dir, exist_ok=True)
        rigged, _rigged_skel = rig_yolo(glb, os.path.join(out_dir, "anatomy_rigged.glb"), debug_dir=None)

        # SKEL bone mesh (zero-pose T-posed skeleton)
        bones = None
        if export_skel_flag:
            progress(0.7, desc="Generating SKEL bone mesh ...")
            import torch
            from tpose_smpl import export_skel_bones
            bones = export_skel_bones(torch.zeros(10), "/tmp/tposed_bones.glb", gender='male')

        status = f"Rigged surface: {os.path.getsize(rigged)//1024} KB"
        if bones:
            status += f"\nSKEL bone mesh: {os.path.getsize(bones)//1024} KB"
        elif export_skel_flag:
            status += "\nSKEL bone mesh: failed (check logs)"
        progress(1.0, desc="Done!")
        return rigged, bones, status
    except Exception:
        return None, None, f"Error:\n{traceback.format_exc()}"


def gradio_rig(glb_state_path, export_fbx_flag, mdm_prompt, mdm_n_frames, progress=gr.Progress()):
    """Gradio wrapper -- YOLO-pose joint detection replaces HMR2."""
    try:
        sys.path.insert(0, "/root")
        from rig_yolo import rig_yolo
        from rig_stage import export_fbx

        glb = glb_state_path or _last_glb_path or "/tmp/triposg_textured.glb"
        if not os.path.exists(glb):
            return None, None, None, "No GLB found -- run Generate Shape + Apply Texture first.", None, None, None

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
            from rig_stage import run_rig_pipeline
            mdm_result = run_rig_pipeline(
                glb_path=glb,
                reference_image_path="/tmp/triposg_face_ref.png",
                out_dir=out_dir, device=DEVICE,
                export_fbx_flag=False,
                mdm_prompt=mdm_prompt.strip(),
                mdm_n_frames=int(mdm_n_frames),
            )
            animated = mdm_result.get("animated_glb")

        status_parts = ["Rigged: " + os.path.basename(rigged)]
        if fbx: status_parts.append("FBX: " + os.path.basename(fbx))
        if animated: status_parts.append("Animation: " + os.path.basename(animated))
        progress(1.0, desc="Done!")
        return rigged, animated, fbx, "  |  ".join(status_parts), rigged, rigged, rigged_skel
    except Exception:
        return None, None, None, f"Error:\n{traceback.format_exc()}", None, None, None

def run_full_pipeline(
    input_image, remove_background, num_steps, guidance, seed, face_count,
    variant, tex_seed, enhance_face, rembg_threshold, rembg_erode,
    export_fbx, mdm_prompt, mdm_n_frames,
    progress=gr.Progress(),
):
    """Single-click full pipeline: shape → texture → rig → animate."""
    progress(0.0, desc="Stage 1/3: Generating shape...")
    glb, status = generate_shape(
        input_image, remove_background, num_steps, guidance, seed, face_count)
    if not glb:
        return None, None, None, None, None, None, status

    progress(0.33, desc="Stage 2/3: Applying texture + face enhancement...")
    glb, mv_img, status = apply_texture(
        glb, input_image, remove_background, variant, tex_seed,
        enhance_face, rembg_threshold, rembg_erode)
    if not glb:
        return None, None, None, None, None, None, status

    progress(0.66, desc="Stage 3/3: Rigging + MDM animation...")
    rigged, animated, fbx, rig_status, _, _, _skel = gradio_rig(glb, export_fbx, mdm_prompt, mdm_n_frames)

    progress(1.0, desc="Pipeline complete!")
    combined_status = f"[Texture] {status}\n[Rig] {rig_status}"
    return glb, glb, mv_img, rigged, animated, fbx, combined_status


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="TripoSG + MV-Adapter 3D Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# TripoSG + MV-Adapter 3D Studio")
    glb_state = gr.State(None)

    with gr.Tabs():

        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(label="Input Image", type="numpy")
                    remove_bg_check = gr.Checkbox(label="Remove Background", value=True)
                    with gr.Row():
                        rembg_threshold = gr.Slider(0.1, 0.95, value=0.5, step=0.05,
                                                    label="BG Threshold (higher = stricter)")
                        rembg_erode = gr.Slider(0, 8, value=2, step=1,
                                                label="Edge Erode (px)")

                    with gr.Accordion("Shape Settings", open=True):
                        num_steps  = gr.Slider(20, 100, value=50, step=5, label="Inference Steps")
                        guidance   = gr.Slider(1.0, 20.0, value=7.0, step=0.5, label="Guidance Scale")
                        seed       = gr.Number(value=_init_seed, label="Seed", precision=0)
                        face_count = gr.Number(value=0, label="Max Faces (0 = unlimited)", precision=0)

                    with gr.Accordion("Texture Settings", open=True):
                        variant = gr.Radio(["sdxl", "sd21"], value="sdxl",
                                           label="Model (sdxl = better quality, sd21 = less VRAM)")
                        tex_seed          = gr.Number(value=_init_seed, label="Texture Seed", precision=0)
                        enhance_face_check = gr.Checkbox(
                            label="Enhance Face (HyperSwap + RealESRGAN)", value=True)

                    with gr.Row():
                        shape_btn   = gr.Button("Generate Shape",  variant="primary",    scale=2, interactive=False)
                        texture_btn = gr.Button("Apply Texture",   variant="secondary",  scale=2)
                        render_btn  = gr.Button("Render Views",    variant="secondary",  scale=1)
                    run_all_btn = gr.Button("▶ Run Full Pipeline (Shape + Texture + Rig)", variant="primary", interactive=False)

                with gr.Column(scale=1):
                    rembg_preview = gr.Image(label="BG Removed Preview", type="numpy",
                                             interactive=False)
                    status      = gr.Textbox(label="Status", lines=3, interactive=False)
                    model_3d    = gr.Model3D(label="3D Preview", clear_color=[0.9, 0.9, 0.9, 1.0])
                    download_file = gr.File(label="Download GLB")
                    multiview_img = gr.Image(label="Multiview", type="filepath",
                                             interactive=False)

            render_gallery = gr.Gallery(label="Rendered Views", columns=5, height=300)

            # ── wiring: Generate tab ──────────────────────────────────────
            _rembg_inputs = [input_image, remove_bg_check, rembg_threshold, rembg_erode]
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
                # ── Left column: controls ──────────────────────────────────
                with gr.Column(scale=1):
                    gr.Markdown("### Step 1 — SKEL Anatomy Layer")
                    tpose_skel_check = gr.Checkbox(label="Export SKEL bone mesh", value=False)
                    tpose_btn = gr.Button("Rig + SKEL Anatomy", variant="secondary")
                    tpose_status = gr.Textbox(label="Anatomy Status", lines=3, interactive=False)
                    with gr.Row():
                        tpose_surface_dl = gr.File(label="Rigged Surface GLB")
                        tpose_bones_dl   = gr.File(label="SKEL Bone Mesh GLB")

                    gr.Markdown("---")
                    gr.Markdown("### Step 2 — Rig & Export")
                    export_fbx_check  = gr.Checkbox(label="Export FBX (requires Blender)", value=True)
                    mdm_prompt_box    = gr.Textbox(
                        label="Motion Prompt (MDM — leave blank to skip)",
                        placeholder="a person walks forward",
                        value="",
                    )
                    mdm_frames_slider = gr.Slider(
                        60, 300, value=120, step=30, label="Animation Frames (at 20 fps)")
                    rig_btn = gr.Button("Rig Mesh", variant="primary")

                # ── Right column: shared 3D preview ───────────────────────
                with gr.Column(scale=2):
                    rig_status      = gr.Textbox(label="Rig Status", lines=4, interactive=False)
                    show_skel_check = gr.Checkbox(label="Show Skeleton", value=False)
                    rig_model_3d    = gr.Model3D(label="Preview",
                                                  clear_color=[0.9, 0.9, 0.9, 1.0])
                    with gr.Row():
                        rig_glb_dl      = gr.File(label="Download Rigged GLB")
                        rig_animated_dl = gr.File(label="Download Animated GLB (MDM)")
                        rig_fbx_dl      = gr.File(label="Download FBX")

            rigged_base_state = gr.State(None)
            skel_glb_state    = gr.State(None)

            tpose_btn.click(
                fn=gradio_tpose,
                inputs=[glb_state, tpose_skel_check],
                outputs=[tpose_surface_dl, tpose_bones_dl, tpose_status],
            ).then(
                fn=lambda p: (p["path"] if isinstance(p, dict) else p) if p else None,
                inputs=[tpose_surface_dl],
                outputs=[rig_model_3d],
            )

            rig_btn.click(
                fn=gradio_rig,
                inputs=[glb_state, export_fbx_check, mdm_prompt_box, mdm_frames_slider],
                outputs=[rig_glb_dl, rig_animated_dl, rig_fbx_dl, rig_status,
                         rig_model_3d, rigged_base_state, skel_glb_state],
            )

            show_skel_check.change(
                fn=lambda show, base, skel: (skel if (show and skel) else base),
                inputs=[show_skel_check, rigged_base_state, skel_glb_state],
                outputs=[rig_model_3d],
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
                    run_normal_check   = gr.Checkbox(label="Run StableNormal", value=True)
                    normal_res         = gr.Slider(512, 1024, value=768, step=128,
                                                   label="Resolution")
                    normal_strength    = gr.Slider(0.1, 3.0, value=1.0, step=0.1,
                                                   label="Normal Strength")

                    gr.Markdown("### Depth-Anything V2")
                    run_depth_check    = gr.Checkbox(label="Run Depth-Anything V2", value=True)
                    depth_res          = gr.Slider(512, 1024, value=768, step=128,
                                                   label="Resolution")
                    displacement_scale = gr.Slider(0.1, 3.0, value=1.0, step=0.1,
                                                   label="Displacement Scale")

                    enhance_btn        = gr.Button("Run Enhancement", variant="primary")
                    unload_btn         = gr.Button("Unload Models (free VRAM)", variant="secondary")

                with gr.Column(scale=2):
                    enhance_status     = gr.Textbox(label="Status", lines=5, interactive=False)
                    with gr.Row():
                        normal_map_img = gr.Image(label="Normal Map", type="pil")
                        depth_map_img  = gr.Image(label="Depth Map", type="pil")
                    enhanced_glb_dl    = gr.File(label="Download Enhanced GLB")
                    enhanced_model_3d  = gr.Model3D(label="Enhanced Preview",
                                                     clear_color=[0.9, 0.9, 0.9, 1.0])

            def gradio_enhance(glb_path, ref_img_np,
                               do_normal, norm_res, norm_strength,
                               do_depth, dep_res, disp_scale):
                if not glb_path:
                    return None, None, None, None, "No GLB loaded — run Generate first."
                if ref_img_np is None:
                    return None, None, None, None, "No reference image — run Generate first."
                try:
                    ref_pil = Image.fromarray(ref_img_np.astype(np.uint8))
                    out_path = glb_path.replace(".glb", "_enhanced.glb")
                    import shutil as _sh
                    _sh.copy2(glb_path, out_path)

                    normal_out = None
                    depth_out  = None
                    log = []

                    if do_normal:
                        log.append("[StableNormal] Running...")
                        yield None, None, None, None, "\n".join(log)
                        normal_out = run_stable_normal(ref_pil, resolution=norm_res)
                        out_path = bake_normal_into_glb(out_path, normal_out, out_path,
                                                         normal_strength=norm_strength)
                        log.append(f"[StableNormal] Done → baked normalTexture (strength {norm_strength})")
                        yield normal_out, depth_out, None, None, "\n".join(log)

                    if do_depth:
                        log.append("[Depth-Anything] Running...")
                        yield normal_out, depth_out, None, None, "\n".join(log)
                        depth_out = run_depth_anything(ref_pil, resolution=dep_res)
                        out_path = bake_depth_as_occlusion(out_path, depth_out, out_path,
                                                            displacement_scale=disp_scale)
                        depth_preview = depth_out.convert("L").convert("RGB")
                        log.append(f"[Depth-Anything] Done → baked occlusionTexture (scale {disp_scale})")
                        yield normal_out, depth_preview, None, None, "\n".join(log)

                    log.append("Enhancement complete.")
                    yield normal_out, (depth_out.convert("L").convert("RGB") if depth_out else None), out_path, out_path, "\n".join(log)

                except Exception as e:
                    yield None, None, None, None, f"Error:\n{traceback.format_exc()}"

            enhance_btn.click(
                fn=gradio_enhance,
                inputs=[glb_state, input_image,
                        run_normal_check, normal_res, normal_strength,
                        run_depth_check, depth_res, displacement_scale],
                outputs=[normal_map_img, depth_map_img,
                         enhanced_glb_dl, enhanced_model_3d, enhance_status],
            )

            unload_btn.click(
                fn=lambda: (unload_models(), "Models unloaded — VRAM freed.")[1],
                inputs=[], outputs=[enhance_status],
            )

        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("Settings"):

            def get_vram_status():
                lines = []
                if torch.cuda.is_available():
                    alloc  = torch.cuda.memory_allocated()  / 1024**3
                    reserv = torch.cuda.memory_reserved()   / 1024**3
                    total  = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    free   = total - reserv
                    lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
                    lines.append(f"VRAM total:     {total:.1f} GB")
                    lines.append(f"VRAM allocated: {alloc:.1f} GB")
                    lines.append(f"VRAM reserved:  {reserv:.1f} GB")
                    lines.append(f"VRAM free:      {free:.1f} GB")
                else:
                    lines.append("No CUDA device available.")
                lines.append("")
                lines.append("Loaded models:")
                lines.append(f"  TripoSG pipeline: {'✓ loaded' if _triposg_pipe is not None else '○ not loaded'}")
                lines.append(f"  RMBG-{_rmbg_version or '?'}:        {'✓ loaded' if _rmbg_net is not None else '○ not loaded'}")
                lines.append(f"  StableNormal:     {'✓ loaded' if _enh_mod._normal_pipe is not None else '○ not loaded'}")
                lines.append(f"  Depth-Anything:   {'✓ loaded' if _enh_mod._depth_pipe is not None else '○ not loaded'}")
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
                    preload_btn    = gr.Button("Preload TripoSG + RMBG to VRAM", variant="primary")
                    unload_triposg_btn  = gr.Button("Unload TripoSG / RMBG")
                    unload_enh_btn      = gr.Button("Unload Enhancement Models (StableNormal / Depth)")
                    unload_all_btn      = gr.Button("Unload All Models", variant="stop")
                    refresh_btn    = gr.Button("Refresh Status")

                with gr.Column(scale=1):
                    gr.Markdown("### GPU Status")
                    vram_status = gr.Textbox(
                        label="", lines=12, interactive=False,
                        value="Click Refresh to check VRAM status.",
                    )

            preload_btn.click(fn=preload_triposg, inputs=[], outputs=[vram_status])
            unload_triposg_btn.click(fn=unload_triposg, inputs=[], outputs=[vram_status])
            unload_enh_btn.click(fn=unload_enhancement, inputs=[], outputs=[vram_status])
            unload_all_btn.click(fn=unload_all, inputs=[], outputs=[vram_status])
            refresh_btn.click(fn=get_vram_status, inputs=[], outputs=[vram_status])

        # ── run_all wiring (after Rig tab so all components are defined) ──
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

    # ── Hidden API endpoints — use invisible Gallery (State is stripped from API in Gradio 6) ──
    _api_render_gallery  = gr.Gallery(visible=False)
    _api_swap_gallery    = gr.Gallery(visible=False)

    def _render_last():
        path = _last_glb_path or "/tmp/triposg_textured.glb"
        return render_views(path)

    _hs_emb_input = gr.Textbox(visible=False)

    gr.Button(visible=False).click(
        fn=_render_last, inputs=[], outputs=[_api_render_gallery], api_name="render_last")
    gr.Button(visible=False).click(
        fn=hyperswap_views, inputs=[_hs_emb_input], outputs=[_api_swap_gallery],
        api_name="hyperswap_views")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        allowed_paths=["/tmp"],
        max_threads=4,
    )
