"""
generate.py
───────────────────────────────────────────────────────────────────────────────
Text-to-motion generation.

Primary backend:  MoMask inference server running on the Vast.ai instance.
                  Returns [T, 263] HumanML3D features directly — no SMPL
                  body mesh required.

Fallback backend: HumanML3D dataset keyword search (offline / no GPU needed).

Usage
─────
    from Retarget.generate import generate_motion

    # Use MoMask on instance
    motion = generate_motion("a person walks forward",
                             backend_url="http://ssh4.vast.ai:8765")

    # Local fallback (streams HuggingFace dataset)
    motion = generate_motion("a person walks forward")

    # Returned motion: np.ndarray [T, 263]
    # Feed directly to animate_glb()
"""
from __future__ import annotations
import json
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_motion(
    prompt:      str,
    backend_url: str | None = None,
    num_frames:  int   = 196,
    fps:         float = 20.0,
    seed:        int   = -1,
) -> np.ndarray:
    """
    Generate a HumanML3D [T, 263] motion array from a text prompt.

    Parameters
    ----------
    prompt
        Natural language description of the desired motion.
        Examples: "a person walks forward", "someone does a jumping jack",
                  "a man waves hello with his right hand"
    backend_url
        URL of the MoMask inference server.  E.g. "http://ssh4.vast.ai:8765".
        If None or if the server is unreachable, falls back to dataset search.
    num_frames
        Desired clip length in frames (at 20 fps; max ~196 ≈ 9.8 s).
    fps
        Target fps (MoMask natively produces 20 fps).
    seed
        Random seed for reproducibility (-1 = random).

    Returns
    -------
    np.ndarray  shape [T, 263]  HumanML3D feature vector.
    """
    if backend_url:
        try:
            return _call_momask(prompt, backend_url, num_frames, seed)
        except Exception as exc:
            print(f"[generate] MoMask unreachable ({exc}) — falling back to dataset search")

    return _dataset_search_fallback(prompt)


# ──────────────────────────────────────────────────────────────────────────────
# MoMask backend
# ──────────────────────────────────────────────────────────────────────────────

def _call_momask(
    prompt:     str,
    url:        str,
    num_frames: int,
    seed:       int,
) -> np.ndarray:
    """POST to the MoMask inference server; return [T, 263] array."""
    import urllib.request

    payload = json.dumps({
        "prompt":     prompt,
        "num_frames": num_frames,
        "seed":       seed,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{url.rstrip('/')}/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read())

    motion = np.array(result["motion"], dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] < 193:
        raise ValueError(f"Server returned unexpected shape {motion.shape}")

    print(f"[generate] MoMask: {motion.shape[0]} frames for '{prompt}'")
    return motion


# ──────────────────────────────────────────────────────────────────────────────
# Dataset search fallback
# ──────────────────────────────────────────────────────────────────────────────

def _dataset_search_fallback(prompt: str) -> np.ndarray:
    """
    Keyword search in TeoGchx/HumanML3D dataset (streaming, HuggingFace).
    Used when no MoMask server is available.
    """
    from .search import search_motions, format_choice_label

    print(f"[generate] Searching HumanML3D dataset for: '{prompt}'")
    results = search_motions(prompt, top_k=5, split="test", max_scan=500)
    if not results:
        raise RuntimeError(
            f"No motion found in dataset for prompt: {prompt!r}\n"
            "Check your internet connection or deploy MoMask on the instance."
        )

    best = results[0]
    print(f"[generate] Best match: {format_choice_label(best)}")
    return np.array(best["motion"], dtype=np.float32)
