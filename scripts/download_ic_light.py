"""Download the IC-Light SD1.5 FC checkpoint to checkpoints/ic_light.

Only iclight_sd15_fc.safetensors (Foreground Conditioned) is used —
ic_light_relight.ICLightRelighter loads it to flat-light the candid before
embed extraction.

Network-only; safe to run alongside a pipeline run.
"""
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", r"D:\HuggingFaceCache\huggingface")
from _hf_env import require_hf_token

TOKEN = require_hf_token()
from huggingface_hub import hf_hub_download

REPO = "lllyasviel/ic-light"
DST = Path(r"D:\Dev Proj\avatar\MeshForge\checkpoints\ic_light")
DST.mkdir(parents=True, exist_ok=True)

FILES = [
    "iclight_sd15_fc.safetensors",
]

for i, name in enumerate(FILES, start=1):
    print(f"{i}/{len(FILES)} {name} ...", flush=True)
    path = hf_hub_download(REPO, name, token=TOKEN, local_dir=str(DST))
    print(f"  -> {path}  ({Path(path).stat().st_size / 1e6:.0f} MB)", flush=True)

print("\nIC-Light checkpoints ready in", DST)
