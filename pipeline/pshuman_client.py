"""
pshuman_client.py
=================
Call PSHuman to generate a high-detail 3D face mesh from a portrait image.

Two modes:
- Direct (default when service_url is localhost): runs PSHuman inference.py
  as a subprocess without going through Gradio HTTP.  Avoids the gradio_client
  API-info bug that affects the pshuman Gradio env.
- Remote: uses gradio_client to call a running pshuman_app.py service.

Usage (standalone)
------------------
python -m pipeline.pshuman_client \\
    --image /path/to/portrait.png \\
    --output /tmp/pshuman_face.obj \\
    [--url http://remote-host:7862]   # omit for direct/local mode

Requires: gradio-client (remote mode only)
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import time
from pathlib import Path

# Default: assume running on the same instance (local)
_DEFAULT_URL = os.environ.get("PSHUMAN_URL", "http://localhost:7862")

# ── Paths (on the Vast instance) ──────────────────────────────────────────────
PSHUMAN_DIR  = "/root/PSHuman"
CONDA_PYTHON = "/root/miniconda/envs/pshuman/bin/python"
CONFIG       = f"{PSHUMAN_DIR}/configs/inference-768-6view.yaml"
HF_MODEL_DIR = f"{PSHUMAN_DIR}/checkpoints/PSHuman_Unclip_768_6views"
HF_MODEL_HUB = "pengHTYX/PSHuman_Unclip_768_6views"


def _run_pshuman_direct(image_path: str, work_dir: str) -> str:
    """
    Run PSHuman inference.py directly as a subprocess.
    Returns path to the colored OBJ mesh.
    """
    img_dir = os.path.join(work_dir, "input")
    out_dir = os.path.join(work_dir, "out")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    scene = "face"
    dst = os.path.join(img_dir, f"{scene}.png")
    shutil.copy(image_path, dst)

    hf_model = HF_MODEL_DIR if Path(HF_MODEL_DIR).exists() else HF_MODEL_HUB

    cmd = [
        CONDA_PYTHON, f"{PSHUMAN_DIR}/inference.py",
        "--config", CONFIG,
        f"pretrained_model_name_or_path={hf_model}",
        f"validation_dataset.root_dir={img_dir}",
        f"save_dir={out_dir}",
        "validation_dataset.crop_size=740",
        "with_smpl=false",
        "num_views=7",
        "save_mode=rgb",
        "seed=42",
    ]

    print(f"[pshuman] Running direct inference: {' '.join(cmd[:4])} ...")
    t0 = time.time()

    # Set CUDA_HOME + extra include dirs so nvdiffrast/torch JIT can compile.
    # On Vast.ai, triposg conda env ships nvcc at bin/nvcc and CUDA headers
    # scattered across site-packages/nvidia/{pkg}/include/ directories.
    env = os.environ.copy()
    # Strip env vars that the triposg parent sets but pshuman's older PyTorch can't handle.
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is a PyTorch 2.0+ feature — older
    # pshuman env crashes at CUDA allocator init with an INTERNAL ASSERT FAILED if it's set.
    for _bad in ("PYTORCH_CUDA_ALLOC_CONF", "TORCH_CUDA_ARCH_LIST"):
        env.pop(_bad, None)

    if "CUDA_HOME" not in env:
        _triposg   = "/root/miniconda/envs/triposg"
        _targets   = os.path.join(_triposg, "targets", "x86_64-linux")
        _nvcc_bin  = os.path.join(_triposg, "bin")
        _cuda_home = _targets  # has include/cuda_runtime_api.h

        _nvvm_bin  = os.path.join(_triposg, "nvvm", "bin")   # contains cicc
        _nvcc_real = os.path.join(_targets, "bin")            # contains nvcc (real one)

        if (os.path.exists(os.path.join(_cuda_home, "include", "cuda_runtime_api.h"))
                and (os.path.exists(os.path.join(_nvcc_bin, "nvcc"))
                     or os.path.exists(os.path.join(_nvcc_real, "nvcc")))):
            env["CUDA_HOME"] = _cuda_home
            # Build PATH: nvvm/bin (cicc) + targets/.../bin (nvcc real) + conda bin (nvcc wrapper)
            path_parts = []
            if os.path.isdir(_nvvm_bin):
                path_parts.append(_nvvm_bin)
            if os.path.isdir(_nvcc_real):
                path_parts.append(_nvcc_real)
            path_parts.append(_nvcc_bin)
            env["PATH"] = ":".join(path_parts) + ":" + env.get("PATH", "")

            # Collect all nvidia sub-package include dirs (cusparse, cublas, etc.)
            _nvidia_site = os.path.join(_triposg, "lib", "python3.10",
                                        "site-packages", "nvidia")
            _extra_incs = []
            if os.path.isdir(_nvidia_site):
                import glob as _glob
                for _inc in _glob.glob(os.path.join(_nvidia_site, "*/include")):
                    if os.path.isdir(_inc):
                        _extra_incs.append(_inc)
            if _extra_incs:
                _sep = ":"
                _existing = env.get("CPATH", "")
                env["CPATH"] = _sep.join(_extra_incs) + (_sep + _existing if _existing else "")
            print(f"[pshuman] CUDA_HOME={_cuda_home}, {len(_extra_incs)} nvidia include dirs added")

    proc = subprocess.run(
        cmd, cwd=PSHUMAN_DIR,
        capture_output=False,
        text=True,
        timeout=600,
        env=env,
    )
    elapsed = time.time() - t0
    print(f"[pshuman] Inference done in {elapsed:.1f}s (exit={proc.returncode})")

    if proc.returncode != 0:
        raise RuntimeError(f"PSHuman inference failed (exit {proc.returncode})")

    # Locate output OBJ — PSHuman may save relative to its CWD (/root/PSHuman/out/)
    # rather than to the specified save_dir, so check both locations.
    # IMPORTANT: always check for colored OBJs (result_clr_scale*) before falling
    # back to any OBJ — the colored file has vertex colors (v x y z r g b) which
    # are required for the face to render with skin tones.
    cwd_out_dir = os.path.join(PSHUMAN_DIR, "out", scene)
    colored_patterns = [
        f"{out_dir}/{scene}/result_clr_scale4_{scene}.obj",
        f"{out_dir}/{scene}/result_clr_scale*_{scene}.obj",
        f"{cwd_out_dir}/result_clr_scale4_{scene}.obj",
        f"{cwd_out_dir}/result_clr_scale*_{scene}.obj",
        f"{PSHUMAN_DIR}/out/**/result_clr_scale*_{scene}.obj",
    ]
    fallback_patterns = [
        f"{out_dir}/**/*.obj",
        f"{cwd_out_dir}/*.obj",
        f"{PSHUMAN_DIR}/out/**/*.obj",
    ]
    obj_path = None
    # First pass: colored OBJs only
    for pat in colored_patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            obj_path = hits[-1]
            print(f"[pshuman] Found colored OBJ: {obj_path}")
            break
    # Second pass: any OBJ (fallback)
    if not obj_path:
        for pat in fallback_patterns:
            hits = sorted(glob.glob(pat, recursive=True))
            if hits:
                colored = [h for h in hits if "clr" in h]
                obj_path = (colored or hits)[-1]
                if colored:
                    print(f"[pshuman] Found colored OBJ (fallback): {obj_path}")
                else:
                    print(f"[pshuman] WARNING: no colored OBJ found, using: {obj_path}")
                break

    if not obj_path:
        all_files = list(Path(out_dir).rglob("*"))
        objs = [str(f) for f in all_files if f.suffix in (".obj", ".ply", ".glb")]
        if objs:
            obj_path = objs[-1]
        if not obj_path and Path(cwd_out_dir).exists():
            for f in Path(cwd_out_dir).rglob("*.obj"):
                obj_path = str(f)
                break
        if not obj_path:
            raise FileNotFoundError(
                f"No mesh output found in {out_dir}. "
                f"Files: {[str(f) for f in all_files[:20]]}"
            )

    print(f"[pshuman] Output mesh: {obj_path}")
    return obj_path


def generate_pshuman_mesh(
    image_path: str,
    output_path: str,
    service_url: str = _DEFAULT_URL,
    timeout: float = 600.0,
) -> str:
    """
    Generate a PSHuman face mesh and save it to *output_path*.

    When service_url points to localhost, PSHuman inference.py is run directly
    (no Gradio HTTP, avoids gradio_client API-info bug).
    For remote URLs, gradio_client is used.

    Parameters
    ----------
    image_path   : local PNG/JPG path of the portrait
    output_path  : where to save the downloaded OBJ
    service_url  : base URL of pshuman_app.py, or "direct" to skip HTTP
    timeout      : seconds to wait for inference (used in remote mode)

    Returns
    -------
    output_path  (convenience)
    """
    import tempfile

    output_path = str(output_path)
    os.makedirs(Path(output_path).parent, exist_ok=True)

    is_local = (
        "localhost" in service_url
        or "127.0.0.1" in service_url
        or service_url.strip().lower() == "direct"
        or not service_url.strip()
    )

    if is_local:
        # ── Direct mode: run subprocess ───────────────────────────────────────
        print(f"[pshuman] Direct mode (no HTTP) — running inference on {image_path}")
        work_dir = tempfile.mkdtemp(prefix="pshuman_direct_")
        obj_tmp  = _run_pshuman_direct(image_path, work_dir)
    else:
        # ── Remote mode: call Gradio service ──────────────────────────────────
        try:
            from gradio_client import Client
        except ImportError:
            raise ImportError("pip install gradio-client")

        print(f"[pshuman] Connecting to {service_url}")
        client = Client(service_url)

        print(f"[pshuman] Submitting: {image_path}")
        result = client.predict(
            image=image_path,
            api_name="/gradio_generate_face",
        )

        if isinstance(result, (list, tuple)):
            obj_tmp = result[0]
            status  = result[1] if len(result) > 1 else "ok"
        elif isinstance(result, dict):
            obj_tmp = result.get("obj_path") or result.get("value")
            status  = result.get("status", "ok")
        else:
            obj_tmp = result
            status  = "ok"

        if not obj_tmp or "Error" in str(status):
            raise RuntimeError(f"PSHuman service error: {status}")

        if isinstance(obj_tmp, dict):
            obj_tmp = obj_tmp.get("path") or obj_tmp.get("name") or str(obj_tmp)

        work_dir = str(Path(str(obj_tmp)).parent)

    # ── Copy OBJ + companions to output location ───────────────────────────
    shutil.copy(str(obj_tmp), output_path)
    print(f"[pshuman] Saved OBJ -> {output_path}")

    src_dir = Path(str(obj_tmp)).parent
    out_dir = Path(output_path).parent
    for ext in ("*.mtl", "*.png", "*.jpg"):
        for f in src_dir.glob(ext):
            dest = out_dir / f.name
            if not dest.exists():
                shutil.copy(str(f), str(dest))

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate PSHuman face mesh from portrait image"
    )
    parser.add_argument("--image",   required=True, help="Portrait image path")
    parser.add_argument("--output",  required=True, help="Output OBJ path")
    parser.add_argument(
        "--url", default=_DEFAULT_URL,
        help="PSHuman service URL, or 'direct' to run inference locally "
             "(default: http://localhost:7862 → auto-selects direct mode)",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    generate_pshuman_mesh(
        image_path  = args.image,
        output_path = args.output,
        service_url = args.url,
        timeout     = args.timeout,
    )


if __name__ == "__main__":
    main()
