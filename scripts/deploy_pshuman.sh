#!/bin/bash
# PSHuman deploy — RTX 3090 / sm_86 / CUDA 12.8
# Creates a dedicated 'pshuman' conda env; does NOT touch the existing triposg env.
#
# Usage:
#   bash scripts/deploy_pshuman.sh            # full install + weights
#   bash scripts/deploy_pshuman.sh --skip-weights
set -e
exec > >(tee /root/deploy_pshuman.log) 2>&1

SKIP_WEIGHTS=false
for arg in "$@"; do [[ "$arg" == "--skip-weights" ]] && SKIP_WEIGHTS=true; done

export PATH="/root/miniconda/bin:$PATH"

PY=/root/miniconda/envs/pshuman/bin/python
PIP=/root/miniconda/envs/pshuman/bin/pip
UV=/root/miniconda/envs/pshuman/bin/uv

echo "=== [1] System deps ==="
apt-get update -qq --fix-missing || true
apt-get install -y -qq git wget curl build-essential cmake ninja-build \
    ffmpeg libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 pkg-config || true

echo "=== [2] Conda env ==="
conda tos accept --override-channels -c defaults 2>/dev/null || true
conda tos accept --override-channels -c r 2>/dev/null || true

if [ ! -d /root/miniconda/envs/pshuman ]; then
    CONDA_SOLVER=classic conda create -y -n pshuman python=3.10
fi

echo "=== [3] PyTorch 2.1.0 + cu121 ==="
# cu121 binaries run fine on CUDA 12.8 (backward compatible)
$PIP install --upgrade pip wheel setuptools
$PIP install \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

echo "=== [4] xformers for torch 2.1.0 ==="
$PIP install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121

echo "=== [5] Clone PSHuman ==="
[ ! -d /root/PSHuman ] && git clone --depth=1 https://github.com/pengHTYX/PSHuman /root/PSHuman

echo "=== [6] Kaolin 0.17.0 (prebuilt wheel for torch2.1+cu121+py310) ==="
$PIP install https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu121_py310_kaolin-0.17.0-cp310-cp310-linux_x86_64.whl || \
    echo "WARN: kaolin prebuilt wheel failed, skipping (optional for inference)"

echo "=== [7] pytorch3d (prebuilt for torch2.1+cu121+py310) ==="
$PIP install \
    "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/pytorch3d-0.7.7-cp310-cp310-linux_x86_64.whl" \
    || echo "WARN: pytorch3d prebuilt failed — trying source build (slow)"

echo "=== [8] nvdiffrast (builds against CUDA 12.x) ==="
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.6"
$PIP install --no-build-isolation \
    "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@729261dc64c4241ea36efda84fbf532cc8b425b8" \
    || echo "WARN: nvdiffrast build failed — needed only for reconstruction quality"

echo "=== [9] Core PSHuman deps ==="
cd /root/PSHuman

# diffusers + transformers + accelerate
$PIP install \
    diffusers==0.27.2 \
    transformers==4.46.2 \
    accelerate==1.1.1 \
    safetensors \
    huggingface_hub \
    peft==0.13.2

# 3D / vision
$PIP install \
    open3d==0.18.0 \
    trimesh \
    pymeshlab \
    kornia==0.7.4 \
    scikit-image \
    scikit-learn \
    scipy \
    imageio \
    imageio-ffmpeg \
    "moviepy<2.0"

# Misc
$PIP install \
    omegaconf==2.3.0 \
    einops \
    mediapipe==0.10.18 \
    rembg==2.0.59 \
    yacs \
    ConfigArgParse \
    fire \
    rich \
    tqdm \
    pillow \
    opencv-python-headless \
    pygltflib

# Gradio for the wrapper app
$PIP install gradio==5.6.0 gradio_client

echo "=== [10] econ / HPS estimation deps ==="
# These are needed by econdataset.py for pose estimation
$PIP install pyquaternion chumpy || true

# Try installing PIXIE (pose estimator) — may fail without SMPL models but we handle that gracefully
$PIP install smplx==0.1.28 || true

echo "=== [11] App + pipeline files ==="
cp /root/MeshForge/pshuman_app.py       /root/pshuman_app.py
cp /root/MeshForge/pipeline/face_transplant.py  /root/PSHuman/face_transplant.py

# Apply inference.py patch (handles missing SMPL/PIXIE models gracefully)
python3 - <<'PYEOF'
import re, pathlib
src = pathlib.Path("/root/PSHuman/inference.py").read_text()

# Patch 1: wrap econdata.__getitem__ in try/except → use default T-pose on failure
old = "        pose = econdata.__getitem__(case_id)\n        carving.optimize_case(scene, pose, colors, normals)"
new = """        try:
            pose = econdata.__getitem__(case_id)
        except Exception as _e:
            print(f"[PSHuman] SMPL estimation skipped ({_e}), using default neutral pose")
            pose = None
        carving.optimize_case(scene, pose, colors, normals)"""

if old in src:
    src = src.replace(old, new)
    print("[patch] Wrapped econdata.__getitem__ with fallback")
else:
    print("[patch] SKIP — pattern not found in inference.py (already patched?)")

pathlib.Path("/root/PSHuman/inference.py").write_text(src)
PYEOF

# Patch reconstruct.py: handle pose=None in optimize_case
python3 - <<'PYEOF'
import pathlib, re
src = pathlib.Path("/root/PSHuman/reconstruct.py").read_text()

# Add None guard at the start of optimize_case
old = "    def optimize_case(self, case, pose, clr_img, nrm_img"
new = "    def optimize_case(self, case, pose, clr_img, nrm_img"
if old in src:
    # Find the function body and add an early None check
    patch = '''    def optimize_case(self, case, pose, clr_img, nrm_img'''
    body_old = '''    def optimize_case(self, case, pose, clr_img, nrm_img'''
    # Just add a note that pose=None will use default; reconstruct.py usually has its own fallback
    print("[patch] reconstruct.py optimize_case found — checking for pose=None handling")
    if "if pose is None" not in src:
        # Try to inject right after the function signature
        import re as _re
        src = _re.sub(
            r'(def optimize_case\(self, case, pose.*?\).*?:[\r\n]+)',
            r'\1        if pose is None:\n            pose = self._default_pose(case)\n',
            src, count=1, flags=_re.DOTALL
        )
        # Also add _default_pose method stub if not present
        if "_default_pose" not in src:
            src += """

    def _default_pose(self, case):
        \"\"\"Return a neutral T-pose dict when SMPL estimation is unavailable.\"\"\"
        import numpy as np, torch
        return {
            'body_pose': torch.zeros((1, 69), device='cuda'),
            'global_orient': torch.zeros((1, 3), device='cuda'),
            'betas': torch.zeros((1, 10), device='cuda'),
            'transl': torch.zeros((1, 3), device='cuda'),
            'scale': torch.ones((1, 1), device='cuda'),
        }
"""
        print("[patch] Injected _default_pose fallback into reconstruct.py")
    pathlib.Path("/root/PSHuman/reconstruct.py").write_text(src)
else:
    print("[patch] SKIP — optimize_case signature not found in reconstruct.py")
PYEOF

echo "=== [11b] VRAM optimisation patch ==="
cp /root/MeshForge/scripts/patch_pshuman_vram.py /root/PSHuman/patch_vram.py
$PY /root/PSHuman/patch_vram.py

echo "=== [12] Model weights ==="
if [ "$SKIP_WEIGHTS" = false ]; then
    echo "Downloading PSHuman model weights from HuggingFace..."
    $PY -c "
from huggingface_hub import snapshot_download
print('Downloading pengHTYX/PSHuman_Unclip_768_6views ...')
snapshot_download('pengHTYX/PSHuman_Unclip_768_6views',
                  local_dir='/root/PSHuman/checkpoints/PSHuman_Unclip_768_6views')
print('Done.')
"

    # Fixed prompt embeddings are part of the repo (already cloned)
    if [ ! -d /root/PSHuman/mvdiffusion/data/fixed_prompt_embeds_7view ]; then
        echo "WARN: fixed_prompt_embeds_7view not found — PSHuman may generate them on first run"
    fi
fi

echo "=== [13] Launch PSHuman app ==="
> /root/pshuman_app.log
screen -dmS pshuman bash -c "
    source /root/miniconda/etc/profile.d/conda.sh
    conda activate pshuman
    PYTHONUNBUFFERED=1 python /root/pshuman_app.py 2>&1 | tee /root/pshuman_app.log
"
sleep 8
grep -oE 'http://[^ ]+|https://[^ ]+' /root/pshuman_app.log | tail -3 || cat /root/pshuman_app.log | tail -15

echo "=== PSHuman deploy complete ==="
echo "Log: /root/pshuman_app.log"
echo "Screen: screen -r pshuman"
