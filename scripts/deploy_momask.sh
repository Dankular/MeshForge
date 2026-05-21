#!/bin/bash
# deploy_momask.sh
# Install MoMask on the Vast.ai instance and start the inference server.
#
# Run on instance:
#   bash /root/deploy_momask.sh
#
# Requirements: conda env 'triposg' already exists (Python 3.10, torch+CUDA).
# MoMask adds ~3GB VRAM; safe to run alongside triposg_app (which loads on demand).

set -e

MOMASK_DIR=/root/momask-codes
PORT=8765
CONDA_PYTHON=/root/miniconda/envs/triposg/bin/python
PIP=/root/miniconda/envs/triposg/bin/pip

# ── 1. Clone MoMask ──────────────────────────────────────────────────────────
if [ ! -d "$MOMASK_DIR" ]; then
  echo "[deploy] Cloning MoMask..."
  git clone https://github.com/EricGuo5513/momask-codes "$MOMASK_DIR"
fi

# ── 2. Install Python deps ───────────────────────────────────────────────────
echo "[deploy] Installing MoMask deps..."
$PIP install flask clip scipy einops ftfy regex tqdm --quiet

# Install CLIP (OpenAI)
$PIP install git+https://github.com/openai/CLIP.git --quiet || true

# ── 3. Download MoMask checkpoints ──────────────────────────────────────────
echo "[deploy] Downloading MoMask checkpoints (~1.5GB)..."
CKPT_DIR=$MOMASK_DIR/checkpoints/t2m

# VQ-VAE (residual codebook)
mkdir -p $CKPT_DIR/Comp_v6_KLD005
if [ ! -f "$CKPT_DIR/Comp_v6_KLD005/net_last.pth" ]; then
  wget -q --show-progress \
    "https://huggingface.co/EricGuo5513/momask-codes/resolve/main/checkpoints/t2m/Comp_v6_KLD005/net_last.pth" \
    -O "$CKPT_DIR/Comp_v6_KLD005/net_last.pth"
fi

# Mask Transformer
TRANS_DIR=$CKPT_DIR/t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns
mkdir -p $TRANS_DIR/meta
if [ ! -f "$TRANS_DIR/net_last.pth" ]; then
  wget -q --show-progress \
    "https://huggingface.co/EricGuo5513/momask-codes/resolve/main/checkpoints/t2m/t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns/net_last.pth" \
    -O "$TRANS_DIR/net_last.pth"
fi

# opt.txt (model config)
if [ ! -f "$TRANS_DIR/opt.txt" ]; then
  wget -q \
    "https://huggingface.co/EricGuo5513/momask-codes/resolve/main/checkpoints/t2m/t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns/opt.txt" \
    -O "$TRANS_DIR/opt.txt"
fi

# Normalisation stats (mean/std)
if [ ! -f "$TRANS_DIR/meta/mean.npy" ]; then
  wget -q \
    "https://huggingface.co/EricGuo5513/momask-codes/resolve/main/checkpoints/t2m/t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns/meta/mean.npy" \
    -O "$TRANS_DIR/meta/mean.npy"
  wget -q \
    "https://huggingface.co/EricGuo5513/momask-codes/resolve/main/checkpoints/t2m/t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns/meta/std.npy" \
    -O "$TRANS_DIR/meta/std.npy"
fi

echo "[deploy] Checkpoints ready."

# ── 4. Upload and start server ───────────────────────────────────────────────
echo "[deploy] Starting MoMask server on port $PORT..."
screen -S momask -X quit 2>/dev/null || true
screen -dmS momask bash -c "
  cd $MOMASK_DIR
  $CONDA_PYTHON /root/momask_server.py \
    --momask-root $MOMASK_DIR \
    --port $PORT \
    --device cuda \
    2>&1 | tee /root/momask_server.log
"

sleep 5
if screen -list | grep -q momask; then
  echo "[deploy] MoMask server running in screen 'momask' on port $PORT"
  echo "[deploy] Logs: /root/momask_server.log"
  echo "[deploy] Health: curl http://localhost:$PORT/health"
else
  echo "[deploy] ERROR: server failed to start — check /root/momask_server.log"
  exit 1
fi
