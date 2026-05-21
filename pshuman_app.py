"""
pshuman_app.py
==============
Gradio wrapper for PSHuman inference running on the Vast.ai instance.

Exposes one endpoint:
  /generate_face  — takes a portrait image, returns a colored OBJ mesh

Runs on port 7862 (triposg=7860, facefusion=7861).
Called by pipeline/pshuman_client.py on the main app instance.
"""
import os
import sys
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import gradio as gr

PSHUMAN_DIR  = "/root/PSHuman"
CONDA_PYTHON = "/root/miniconda/envs/pshuman/bin/python"
CONFIG       = f"{PSHUMAN_DIR}/configs/inference-768-6view.yaml"
HF_MODEL     = f"{PSHUMAN_DIR}/checkpoints/PSHuman_Unclip_768_6views"

# Override to use HuggingFace Hub name if local checkpoint not downloaded
if not Path(HF_MODEL).exists():
    HF_MODEL = "pengHTYX/PSHuman_Unclip_768_6views"


def run_pshuman(image_path: str, work_dir: str) -> str:
    """
    Run inference.py on a single image, return path to colored OBJ.
    """
    # PSHuman expects a directory of images
    img_dir = os.path.join(work_dir, "input")
    out_dir = os.path.join(work_dir, "out")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Copy + rename to predictable name
    scene = "face"
    dst = os.path.join(img_dir, f"{scene}.png")
    shutil.copy(image_path, dst)

    cmd = [
        CONDA_PYTHON, f"{PSHUMAN_DIR}/inference.py",
        "--config", CONFIG,
        f"pretrained_model_name_or_path={HF_MODEL}",
        f"validation_dataset.root_dir={img_dir}",
        f"save_dir={out_dir}",
        "validation_dataset.crop_size=740",
        "with_smpl=false",
        "num_views=7",
        "save_mode=rgb",
        "seed=42",
    ]

    print(f"[pshuman_app] Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=PSHUMAN_DIR,
        capture_output=False,
        text=True,
        timeout=600,
    )
    elapsed = time.time() - t0
    print(f"[pshuman_app] Inference finished in {elapsed:.1f}s (exit={result.returncode})")

    if result.returncode != 0:
        raise RuntimeError(f"PSHuman inference failed (exit {result.returncode})")

    # Find output OBJ — PSHuman writes to {out_dir}/{scene}/result_clr_scale4_{scene}.obj
    candidate_patterns = [
        f"{out_dir}/{scene}/result_clr_scale4_{scene}.obj",
        f"{out_dir}/{scene}/result_clr_scale*_{scene}.obj",
        f"{out_dir}/**/*.obj",
    ]

    import glob
    obj_path = None
    for pat in candidate_patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            # Prefer colored OBJ (has 'clr' in name)
            colored = [h for h in hits if "clr" in h]
            obj_path = (colored or hits)[-1]
            break

    if not obj_path:
        # Fallback: list everything produced
        all_files = list(Path(out_dir).rglob("*"))
        objs = [str(f) for f in all_files if f.suffix in (".obj", ".ply", ".glb")]
        if objs:
            obj_path = objs[-1]
        else:
            raise FileNotFoundError(
                f"No mesh output found in {out_dir}. Files: {[str(f) for f in all_files[:20]]}"
            )

    print(f"[pshuman_app] Output mesh: {obj_path}")
    return obj_path


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def gradio_generate_face(image):
    """
    Gradio handler: receives a PIL/numpy image, runs PSHuman, returns OBJ path.
    """
    if image is None:
        return None, "No image provided"

    work_dir = tempfile.mkdtemp(prefix="pshuman_")
    try:
        from PIL import Image
        import numpy as np

        # Save input image
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        img_path = os.path.join(work_dir, "portrait.png")
        img.save(img_path)

        obj_path = run_pshuman(img_path, work_dir)

        # Copy OBJ + companions to a stable temp location
        out_stem  = Path(work_dir) / "pshuman_face"
        shutil.copy(obj_path, str(out_stem) + ".obj")
        mtl = Path(obj_path).with_suffix(".mtl")
        if mtl.exists():
            shutil.copy(str(mtl), str(out_stem) + ".mtl")
        for tex in Path(obj_path).parent.glob("*.png"):
            shutil.copy(str(tex), work_dir)

        return str(out_stem) + ".obj", "Done"

    except Exception as e:
        import traceback
        return None, f"Error: {e}\n{traceback.format_exc()}"


with gr.Blocks(title="PSHuman Face Generator") as demo:
    gr.Markdown("## PSHuman — Single-image Face Mesh Generation")
    gr.Markdown(
        "Upload a portrait image. PSHuman generates a high-detail colored 3D face/body mesh "
        "which is used as the face donor in the MeshForge face transplant pipeline."
    )
    with gr.Row():
        with gr.Column(scale=1):
            img_input  = gr.Image(label="Portrait image", type="pil")
            run_btn    = gr.Button("Generate face mesh", variant="primary")
        with gr.Column(scale=1):
            status_box = gr.Textbox(label="Status", lines=3, interactive=False)
            mesh_dl    = gr.File(label="Download OBJ mesh")

    run_btn.click(
        fn=gradio_generate_face,
        inputs=[img_input],
        outputs=[mesh_dl, status_box],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=True,
    )
