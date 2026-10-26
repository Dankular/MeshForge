"""Download GeoWizard SD2.1 depth+normal weights to checkpoints/geowizard.

Model: lemonaddie/geowizard (joint monocular depth + surface-normal
diffusion, ECCV 2024). ~3.5 GB fp16. Network-only; safe alongside a run.
"""
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", r"D:\HuggingFaceCache\huggingface")
from _hf_env import require_hf_token

TOKEN = require_hf_token()
from huggingface_hub import snapshot_download

REPO = "lemonaddie/geowizard"
DST = Path(r"D:\Dev Proj\avatar\MeshForge\checkpoints\geowizard")
DST.mkdir(parents=True, exist_ok=True)

print(f"downloading {REPO} -> {DST} ...", flush=True)
path = snapshot_download(
    REPO,
    token=TOKEN,
    local_dir=str(DST),
    allow_patterns=[
        "vae/*", "unet/*", "scheduler/*",
        "image_encoder/*", "feature_extractor/*",
        "model_index.json",
    ],
)
print(f"GeoWizard weights ready in {path}", flush=True)
