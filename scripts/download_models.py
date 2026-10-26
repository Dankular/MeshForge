"""Download remaining PuLID-FLUX dependencies to correct locations."""
import os, sys

os.environ["HF_HOME"] = r"D:\HuggingFaceCache\huggingface"
from _hf_env import require_hf_token

TOKEN = require_hf_token()
from huggingface_hub import hf_hub_download, snapshot_download

MODELS = r"D:\Dev Proj\avatar\MeshForge\models"
CHECKPOINTS = r"D:\Dev Proj\avatar\MeshForge\checkpoints"
PULID_DIR = os.path.join(CHECKPOINTS, "flux")

os.makedirs(PULID_DIR, exist_ok=True)

# 1. T5 text encoder (xlabs-ai/xflux_text_encoders)
# PuLID's load_t5 uses HFEmbedder which calls snapshot_download internally —
# we just need to ensure it's in the HF cache.
print("1/3 T5 text encoders (xlabs-ai/xflux_text_encoders) ...")
snapshot_download("xlabs-ai/xflux_text_encoders", token=TOKEN)
print("  done")

# 2. CLIP (openai/clip-vit-large-patch14)
print("2/3 CLIP (openai/clip-vit-large-patch14) ...")
snapshot_download("openai/clip-vit-large-patch14", token=TOKEN)
print("  done")

# 3. PuLID flux weights
print("3/3 PuLID weights (guozinan/PuLID) ...")
dst = hf_hub_download(
    "guozinan/PuLID",
    "pulid_flux_v0.9.1.safetensors",
    token=TOKEN,
    local_dir=PULID_DIR,
)
print(f"  -> {dst}")

print("\nAll supporting models downloaded.")
