#!/usr/bin/env python3
"""
Local Windows Setup for MeshForge Pipeline
Creates directories, installs packages, downloads model weights.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"
TRIPOSG_DIR = MODELS_DIR / "triposg"
MVADAPTER_DIR = MODELS_DIR / "mvadapter"
RMBG_DIR = MODELS_DIR / "rmbg"
PSHUMAN_DIR = MODELS_DIR / "pshuman"

# Create directories
print("Creating local model directories...")
for d in [MODELS_DIR, TRIPOSG_DIR, MVADAPTER_DIR, RMBG_DIR, PSHUMAN_DIR]:
    d.mkdir(parents=True, exist_ok=True)
    print(f"  {d}")

# Check Python version
print(f"\nPython version: {sys.version}")
if sys.version_info < (3, 10):
    print("ERROR: Python 3.10+ required")
    sys.exit(1)

# Check CUDA
print("\nChecking PyTorch CUDA...")
try:
    import torch

    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("  PyTorch not installed!")
    sys.exit(1)

# Install core dependencies
print("\nInstalling/upgrading core dependencies...")
core_packages = [
    "huggingface_hub==0.25.2",
    "transformers==4.45.2",
    "diffusers==0.31.0",
    "accelerate==0.34.2",
    "safetensors==0.4.5",
    "peft==0.13.2",
    "tokenizers==0.20.1",
    "timm==1.0.11",
    "einops==0.8.0",
    "trimesh==4.5.0",
    "pygltflib==1.16.5",
    "scipy==1.14.1",
    "scikit-learn==1.5.2",
    "opencv-python==4.10.0.84",
    "Pillow==10.4.0",
    "gradio==5.3.0",
    "tqdm",
    "requests",
]

for pkg in core_packages:
    print(f"  Installing {pkg}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("\nCore dependencies installed!")

# Download TripoSG weights
print("\nDownloading TripoSG weights (this may take a while)...")
try:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="VAST-AI/TripoSG",
        local_dir=str(TRIPOSG_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"  TripoSG weights saved to: {TRIPOSG_DIR}")
except Exception as e:
    print(f"  ERROR downloading TripoSG: {e}")
    print("  You may need to run: huggingface-cli login")

# Download RMBG-2-Studio weights
print("\nDownloading RMBG-2-Studio weights...")
try:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="briaai/RMBG-2-Studio",
        local_dir=str(RMBG_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"  RMBG weights saved to: {RMBG_DIR}")
except Exception as e:
    print(f"  ERROR downloading RMBG: {e}")

print("\n" + "=" * 60)
print("Setup complete!")
print(f"Models directory: {MODELS_DIR}")
print("\nNext steps:")
print("1. Install remaining dependencies: pip install -r requirements_local.txt")
print("2. Run: python run_pipeline_local.py input_image.webp")
print("=" * 60)
