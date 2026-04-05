#!/bin/bash
# MeshForge deploy — RTX 3090 / sm_86 / CUDA 12.8
# Usage: bash scripts/deploy.sh [--skip-weights]
set -e
exec > >(tee /root/deploy_meshforge.log) 2>&1

SKIP_WEIGHTS=false
for arg in "$@"; do [[ "$arg" == "--skip-weights" ]] && SKIP_WEIGHTS=true; done

export PATH="/root/miniconda/bin:$PATH"
PIP=/root/miniconda/envs/triposg/bin/pip
PY=/root/miniconda/envs/triposg/bin/python

# ── [1] System deps ────────────────────────────────────────────────────────────
apt-get update -qq --fix-missing || true
apt-get install -y -qq git wget curl build-essential cmake ninja-build \
    ffmpeg libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    pkg-config || true

# ── [2] Miniconda ──────────────────────────────────────────────────────────────
if [ ! -d /root/miniconda ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /root/miniconda
fi
conda tos accept --override-channels -c defaults 2>/dev/null || true
conda tos accept --override-channels -c r 2>/dev/null || true

# ── [3] Conda env ──────────────────────────────────────────────────────────────
if [ ! -d /root/miniconda/envs/triposg ]; then
    CONDA_SOLVER=classic conda env create -f /root/MeshForge/environment.yml
fi

# Build env for CUDA extensions
export CUDA_HOME=/root/miniconda/envs/triposg
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"

# ── [4] Clone repos ────────────────────────────────────────────────────────────
[ ! -d /root/TripoSG ]    && git clone --depth=1 https://github.com/VAST-AI-Research/TripoSG /root/TripoSG
[ ! -d /root/MV-Adapter ] && git clone --depth=1 https://github.com/huanngzh/MV-Adapter /root/MV-Adapter
[ ! -d /root/MDM ]        && git clone --depth=1 https://github.com/GuyTevet/motion-diffusion-model /root/MDM

# ── [5] Install all Python deps ────────────────────────────────────────────────
$PIP install -r /root/MeshForge/requirements.txt

# Fix basicsr torchvision compat (rgb_to_grayscale moved in torchvision 0.17+)
BASICSR_DEG=$(find /root/miniconda/envs/triposg -name degradations.py -path "*/basicsr/*" 2>/dev/null | head -1)
if [ -n "$BASICSR_DEG" ]; then
    sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$BASICSR_DEG" || true
fi

# ── [6] Apply patches ─────────────────────────────────────────────────────────
cp /root/MeshForge/patches/TripoSG_image_process.py   /root/TripoSG/scripts/image_process.py
cp /root/MeshForge/patches/MDM_rotation2xyz.py         /root/MDM/model/rotation2xyz.py
cp /root/MeshForge/patches/MDM_mdm.py                  /root/MDM/model/mdm.py
mkdir -p /root/MV-Adapter/scripts
cp /root/MeshForge/pipeline/face_enhance.py            /root/MV-Adapter/scripts/face_enhance.py

# ── [7] App files ─────────────────────────────────────────────────────────────
cp /root/MeshForge/app.py                              /root/triposg_app.py
cp /root/MeshForge/pipeline/rig_yolo.py               /root/MV-Adapter/scripts/rig_yolo.py
cp /root/MeshForge/pipeline/rig_stage.py              /root/MV-Adapter/scripts/rig_stage.py
cp /root/MeshForge/pipeline/tpose_smpl.py             /root/MV-Adapter/scripts/tpose_smpl.py
cp /root/MeshForge/pipeline/enhance_surface.py        /root/enhance_surface.py

# ── [8] Model weights ─────────────────────────────────────────────────────────
if [ "$SKIP_WEIGHTS" = false ]; then
    CKPT=/root/MV-Adapter/checkpoints
    mkdir -p "$CKPT" /root/models/stable-normal /root/models/depth-anything-v2

    # Face enhancement weights
    [ ! -f "$CKPT/hyperswap_1a_256.onnx" ] && \
        wget -q "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/hyperswap_1a_256.onnx" -O "$CKPT/hyperswap_1a_256.onnx" || true
    [ ! -f "$CKPT/inswapper_128.onnx" ] && \
        wget -q "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx" -O "$CKPT/inswapper_128.onnx" || true
    [ ! -f "$CKPT/RealESRGAN_x4plus.pth" ] && \
        wget -q "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x4plus.pth" -O "$CKPT/RealESRGAN_x4plus.pth" || true
    [ ! -f "$CKPT/GFPGANv1.4.pth" ] && \
        wget -q "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" -O "$CKPT/GFPGANv1.4.pth" || true

    # StableNormal
    $PY -c "from huggingface_hub import snapshot_download; snapshot_download('Stable-X/yoso-normal-v1-5', local_dir='/root/models/stable-normal')"

    # Depth-Anything V2 Large
    $PY -c "from huggingface_hub import snapshot_download; snapshot_download('depth-anything/Depth-Anything-V2-Large-hf', local_dir='/root/models/depth-anything-v2')"

    # MDM checkpoint
    mkdir -p /root/MDM/save/humanml_trans_enc_512
    # Download from: https://github.com/GuyTevet/motion-diffusion-model#pretrained-models
    echo "NOTE: Download MDM checkpoint manually to /root/MDM/save/humanml_trans_enc_512/model000200000.pt"
fi

# ── [9] RAM-disk symlinks for fast model loading ───────────────────────────────
# Copies 11GB of TripoSG+MV-Adapter weights to /dev/shm for ~50x faster load
# Run only if /dev/shm has >=15GB free
SHM_FREE=$(df /dev/shm | awk 'NR==2{print $4}')
if [ "$SHM_FREE" -gt 15000000 ]; then
    echo "Setting up /dev/shm symlinks for fast model load..."
    HF_CACHE=/root/.cache/huggingface/hub
    for MODEL in models--VAST-AI-Research--TripoSG models--huanngzh--mv-adapter; do
        SRC="$HF_CACHE/$MODEL"
        DST="/dev/shm/$MODEL"
        if [ -d "$SRC" ] && [ ! -L "$SRC" ]; then
            cp -r "$SRC" "$DST" && rm -rf "$SRC" && ln -s "$DST" "$SRC"
            echo "  Symlinked $MODEL to /dev/shm"
        fi
    done
fi

# ── [10] Launch ────────────────────────────────────────────────────────────────
> /root/triposg_app.log
PYTHONUNBUFFERED=1 nohup $PY /root/triposg_app.py > /root/triposg_app.log 2>&1 &
sleep 15
grep -oE 'https://[a-z0-9]+\.gradio\.live' /root/triposg_app.log | tail -1 || cat /root/triposg_app.log | tail -20

echo "=== Deploy complete ==="
