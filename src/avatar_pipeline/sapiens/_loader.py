"""Shared TorchScript model loader for all Sapiens inference modules.

All Sapiens 1b torchscript models:
- compiled for (H=1024, W=768) input — confirmed empirically
- expect ImageNet-normalised CHW float tensors
- downloaded via huggingface_hub to ~/.cache/sapiens
"""
from __future__ import annotations

import numpy as np
import torch

# All confirmed at 1024×768 from live testing
INF_H, INF_W = 1024, 768

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# repo_id → (list-files endpoint picks the last .pt2 alphabetically)
_REPOS: dict[str, str] = {
    "normal":  "facebook/sapiens-normal-1b-torchscript",
    "depth":   "facebook/sapiens-depth-1b-torchscript",
    "seg":     "facebook/sapiens-seg-1b-torchscript",
    "pose":    "facebook/sapiens-pose-1b-torchscript",
}

_CACHE_DIR = "~/.cache/sapiens"


def preprocess(image_rgba: np.ndarray, device: torch.device) -> torch.Tensor:
    """RGBA uint8/float → normalised (1, 3, 1024, 768) CUDA tensor."""
    import cv2

    rgb = image_rgba[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    resized = cv2.resize(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),   # cv2 works in BGR
        (INF_W, INF_H),
        interpolation=cv2.INTER_LINEAR,
    )
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    t = resized.astype(np.float32) / 255.0
    t = (t - _MEAN) / _STD
    return torch.from_numpy(t).permute(2, 0, 1).unsqueeze(0).to(device)


def load_model(task: str) -> torch.jit.ScriptModule:
    """Download (once) and load a Sapiens 1b TorchScript model."""
    from huggingface_hub import hf_hub_download, list_repo_files
    import pathlib

    repo = _REPOS[task]
    cache = str(pathlib.Path(_CACHE_DIR).expanduser())

    files = sorted(f for f in list_repo_files(repo) if f.endswith(".pt2"))
    if not files:
        raise FileNotFoundError(f"No .pt2 in {repo}")

    local = hf_hub_download(repo_id=repo, filename=files[-1], cache_dir=cache)
    # Loaded onto CPU deliberately — mmgp.offload manages GPU residency for
    # every heavy model in the pipeline from one shared memory-management domain.
    model = torch.jit.load(local, map_location="cpu")
    model.eval()
    return model


def unwrap(output) -> torch.Tensor:
    """Unpack list/tuple outputs from TorchScript models."""
    return output[0] if isinstance(output, (list, tuple)) else output
