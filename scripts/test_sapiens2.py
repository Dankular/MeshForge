"""Probe loading facebook/sapiens2-normal-1b via trust_remote_code (no pkg change)."""
import os, numpy as np
os.environ.setdefault("HF_HOME", r"D:\HuggingFaceCache\huggingface")
from _hf_env import require_hf_token

TOKEN = require_hf_token()
import torch
from PIL import Image

REPO = "facebook/sapiens2-normal-1b"
print("loading", REPO, "...", flush=True)
ok = False
try:
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(REPO, token=TOKEN, trust_remote_code=True)
    model = AutoModel.from_pretrained(REPO, token=TOKEN, trust_remote_code=True, torch_dtype=torch.float16).to("cuda").eval()
    print("LOAD OK via AutoModel(trust_remote_code)", flush=True)
    img = Image.open("outputs/GG_TEST_02/tpose_reference.png").convert("RGB")
    inp = proc(images=img, return_tensors="pt")
    inp = {k: (v.to("cuda").half() if v.dtype.is_floating_point else v.to("cuda")) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp)
    print("OUTPUT keys/attrs:", [a for a in dir(out) if not a.startswith("_")][:12], flush=True)
    # try common output names
    for name in ("predicted_normal", "normals", "logits", "last_hidden_state", "prediction"):
        v = getattr(out, name, None)
        if v is not None and hasattr(v, "shape"):
            print(f"  {name}: {tuple(v.shape)} range [{float(v.min()):.3f},{float(v.max()):.3f}]", flush=True)
    ok = True
except Exception as e:
    import traceback; traceback.print_exc()
    print("LOAD FAILED:", type(e).__name__, str(e)[:160], flush=True)
print("RESULT:", "OK" if ok else "FAIL", flush=True)
