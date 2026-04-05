#!/bin/bash
# TripoSG + MV-Adapter full deploy script
# RTX 3090 / sm_86 / CUDA 12.8 / cu128 PyTorch
set -e
exec > >(tee /root/deploy_triposg.log) 2>&1

export PATH="/root/miniconda/bin:$PATH"
PIP=/root/miniconda/envs/triposg/bin/pip
PY=/root/miniconda/envs/triposg/bin/python

echo "=== [1] System deps ==="
apt-get update -qq --fix-missing || true
apt-get install -y -qq --fix-missing git wget curl tmux build-essential cmake \
    ffmpeg libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    pkg-config ninja-build || true

echo "=== [2] Miniconda ==="
if [ ! -d /root/miniconda ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /root/miniconda
fi
conda tos accept --override-channels -c defaults 2>/dev/null || true
conda tos accept --override-channels -c r 2>/dev/null || true

echo "=== [3] Conda env triposg (Python 3.10) ==="
if [ ! -d /root/miniconda/envs/triposg ]; then
    CONDA_SOLVER=classic conda create -n triposg python=3.10 -y
fi

echo "=== [4] PyTorch cu128 (RTX 3090 sm_86) ==="
$PIP install --quiet torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Verify CUDA is available
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE'; print('CUDA OK:', torch.version.cuda, 'GPU:', torch.cuda.get_device_name(0))"

echo "=== [5] NVCC for CUDA extensions ==="
conda install -n triposg -y cuda-nvcc cuda-cudart-dev \
    -c "nvidia/label/cuda-12.8.0" || \
conda install -n triposg -y -c conda-forge cudatoolkit-dev || true
export CUDA_HOME=/root/miniconda/envs/triposg
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"  # sm_86 for RTX 3090

echo "=== [6] Clone TripoSG ==="
if [ ! -d /root/TripoSG ]; then
    git clone --depth=1 https://github.com/VAST-AI-Research/TripoSG /root/TripoSG
fi
cd /root/TripoSG
$PIP install --quiet -r requirements.txt

echo "=== [7] Clone MV-Adapter ==="
if [ ! -d /root/MV-Adapter ]; then
    git clone --depth=1 https://github.com/huanngzh/MV-Adapter /root/MV-Adapter
fi
cd /root/MV-Adapter
$PIP install --quiet -r requirements.txt || true
$PIP install --quiet -e . || true

echo "=== [8] Extra deps ==="
$PIP install --quiet \
    gradio huggingface_hub diffusers accelerate transformers \
    xformers --index-url https://download.pytorch.org/whl/cu128 || true
$PIP install --quiet \
    trimesh pymeshlab open3d einops omegaconf \
    opencv-python-headless scikit-image scipy onnxruntime-gpu \
    insightface gfpgan basicsr facexlib || true

# Fix basicsr torchvision compat
BASICSR_DEG=$(find /root/miniconda/envs/triposg -name degradations.py -path "*/basicsr/*" 2>/dev/null | head -1)
if [ -n "$BASICSR_DEG" ]; then
    sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$BASICSR_DEG" || true
    echo "basicsr patch applied"
fi

echo "=== [9] nvdiffrast ==="
$PIP install --quiet git+https://github.com/NVlabs/nvdiffrast.git || true

echo "=== [10] MV-Adapter checkpoints ==="
mkdir -p /root/MV-Adapter/checkpoints
cd /root/MV-Adapter/checkpoints

# RealESRGAN upscaler
[ ! -f RealESRGAN_x2plus.pth ] && \
    wget -q "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" || true

# big-lama inpainter
[ ! -f big-lama.pt ] && \
    wget -q "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.pt" || true

# GFPGAN v1.4
[ ! -f GFPGANv1.4.pth ] && \
    wget -q "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" || true

# HyperSwap 1A 256 (face swap model)
[ ! -f hyperswap_1a_256.onnx ] && \
    wget -q "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/hyperswap_1a_256.onnx" || true

# inswapper fallback
[ ! -f inswapper_128.onnx ] && \
    wget -q "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx" || true

echo "=== [11] Upload app files ==="
# These are uploaded by pscp after this script runs — placeholder step
echo "Remember to upload: triposg_app.py, texture_i2tex.py, face_enhance.py"

echo "=== [12] Launch ==="
mkdir -p /root/MV-Adapter/scripts
# Copy face_enhance to scripts dir (will be uploaded separately)

> /root/triposg_app.log
PYTHONUNBUFFERED=1 nohup $PY /root/triposg_app.py > /root/triposg_app.log 2>&1 &
sleep 15
cat /root/triposg_app.log

echo "=== Deploy complete ==="
