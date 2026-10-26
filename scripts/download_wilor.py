"""Download WiLoR hand-reconstruction weights to external/WiLoR/pretrained_models."""
import os
from pathlib import Path
os.environ.setdefault("HF_HOME", r"D:\HuggingFaceCache\huggingface")
from _hf_env import require_hf_token

TOKEN = require_hf_token()
from huggingface_hub import hf_hub_download

DST = Path(r"D:\Dev Proj\avatar\MeshForge\external\WiLoR\pretrained_models")
DST.mkdir(parents=True, exist_ok=True)
for f in ["detector.pt", "wilor_final.ckpt", "model_config.yaml"]:
    print(f"downloading {f} ...", flush=True)
    p = hf_hub_download("rolpotamias/WiLoR", f"pretrained_models/{f}",
                        repo_type="space", token=TOKEN, local_dir=str(DST.parent))
    print(f"  -> {p}", flush=True)
print("WiLoR weights ready in", DST, flush=True)
