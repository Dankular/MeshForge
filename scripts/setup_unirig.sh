#!/bin/bash
# UniRig conda env setup
LOG=/root/unirig_setup.log
exec > >(tee -a "$LOG") 2>&1
set -e

echo "[$(date)] === UniRig Setup Start ==="

source /root/miniconda/etc/profile.d/conda.sh

# Clone UniRig
if [ ! -d /root/UniRig ]; then
    echo "[$(date)] Cloning UniRig..."
    git clone https://github.com/VAST-AI-Research/UniRig /root/UniRig
else
    echo "[$(date)] UniRig already cloned, pulling latest..."
    cd /root/UniRig && git pull
fi

# Create env
if conda env list | grep -q "^unirig "; then
    echo "[$(date)] unirig env already exists, skipping create"
else
    echo "[$(date)] Creating unirig conda env (Python 3.11)..."
    CONDA_SOLVER=classic conda create -n unirig python=3.11 -y
fi

PYTHON=/root/miniconda/envs/unirig/bin/python
PIP=/root/miniconda/envs/unirig/bin/pip

echo "[$(date)] Python: $($PYTHON --version)"

# PyTorch — use latest available for cu128 (2.7.0+ only available on cu128 index)
echo "[$(date)] Installing PyTorch..."
$PIP install torch torchvision --index-url https://download.pytorch.org/whl/cu128

TORCH_VERSION=$($PYTHON -c "import torch; print(torch.__version__)")
echo "[$(date)] torch=$TORCH_VERSION"

# Core deps from UniRig requirements (minus spconv/torch_scatter which need special wheels)
echo "[$(date)] Installing core requirements..."
cd /root/UniRig
$PIP install -r requirements.txt --no-deps 2>/dev/null || true
$PIP install trimesh pygltflib einops scipy scikit-learn plyfile tqdm PyYAML omegaconf hydra-core tensorboard

# numpy pinned (UniRig requires 1.x)
$PIP install "numpy==1.26.4" --force-reinstall

# spconv — cu121 wheel runs on CUDA 12.8 (binary compat)
echo "[$(date)] Installing spconv-cu121..."
$PIP install spconv-cu121

# torch_scatter + torch_cluster — PyG wheels typically lag behind torch releases;
# build from source if no prebuilt wheel matches
echo "[$(date)] Installing torch_scatter and torch_cluster..."
TORCH_VER=$($PYTHON -c "import torch; v=torch.__version__.split('+')[0]; print(v)")
$PIP install torch_scatter torch_cluster \
    -f "https://data.pyg.org/whl/torch-${TORCH_VER}+cu128.html" 2>/dev/null || \
$PIP install torch_scatter torch_cluster \
    -f "https://data.pyg.org/whl/torch-${TORCH_VER}+cu121.html" 2>/dev/null || \
$PIP install torch_scatter torch_cluster --no-build-isolation 2>/dev/null || \
echo "[$(date)] WARNING: torch_scatter/torch_cluster install failed — UniRig may still work without them"

# flash_attn not compilable on cu128 — inject shim instead (patches run.py + unirig_skin.py)
echo "[$(date)] Applying flash_attn + torch.load patches to UniRig source..."
apt-get install -y libegl1 2>/dev/null || true   # required by pyrender during skin inference
$PYTHON /tmp/patch_unirig_flash.py   2>/dev/null || true
$PYTHON /tmp/patch_unirig_restore_mha.py 2>/dev/null || true
$PYTHON /tmp/patch_unirig_mha_v2.py  2>/dev/null || true
$PYTHON /tmp/patch_cloud_io.py       2>/dev/null || true
$PYTHON /tmp/patch_torch_load.py     2>/dev/null || true
# Patch all flash_attention_2 in configs → sdpa
find /root/UniRig/configs -name '*.yaml' -exec sed -i 's/_attn_implementation: flash_attention_2/_attn_implementation: sdpa/g' {} +

# Install remaining deps not in requirements.txt
$PIP install python-box bpy fast-simplification lightning pytorch_lightning addict timm open3d pyrender huggingface_hub wandb transformers==4.51.3 2>/dev/null

# Verify
echo "[$(date)] Verifying imports..."
$PYTHON -c "
import torch
print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
import spconv
print('spconv OK')
import torch_scatter
print('torch_scatter OK')
import torch_cluster
print('torch_cluster OK')
"

echo "[$(date)] === UniRig Setup Complete ==="
