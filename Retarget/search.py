"""
search.py
Stream TeoGchx/HumanML3D from HuggingFace and match motions by keyword.

Dataset: https://huggingface.co/datasets/TeoGchx/HumanML3D
  Format: motion column is [T, 263] inline in parquet (standard HumanML3D)
  Splits: train (23 384), val (1 460), test (4 384)

Usage
-----
    from Retarget.search import search_motions

    results = search_motions("a person walks forward", top_k=5)
    for r in results:
        print(r["caption"], r["frames"], "frames")
        # r["motion"]  →  np.ndarray [T, 263]
"""
from __future__ import annotations
import re
from typing import List, Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Caption cleaning
# ─────────────────────────────────────────────────────────────────────────────

_SEP = re.compile(r'#|\|')
_POS_TAG = re.compile(r'^(?:[A-Z]{1,4}\s*)+$')   # lines that look like POS tags


def _clean_caption(raw: str) -> str:
    """
    HumanML3D captions are stored as multiple sentences joined by '#',
    sometimes followed by POS tag strings.  Return the first human-readable
    sentence.
    """
    parts = _SEP.split(raw)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        words = part.split()
        # Skip if >50 % of tokens look like POS tags (all-caps, ≤4 chars)
        pos_count = sum(1 for w in words if w.isupper() and len(w) <= 4)
        if len(words) > 0 and pos_count / len(words) < 0.5:
            return part
    return parts[0].strip() if parts else raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────────────────

def search_motions(
    query: str,
    top_k: int = 8,
    split: str = "test",
    max_scan: int = 4384,
    cached: bool = False,
) -> List[dict]:
    """
    Stream TeoGchx/HumanML3D and return up to top_k motions matching query.

    Parameters
    ----------
    query       Natural-language description, e.g. "a person walks forward"
    top_k       Maximum number of results to return
    split       Dataset split — "test" (4 384 rows) is fastest to stream
    max_scan    Hard cap on rows examined before returning

    Returns
    -------
    List of dicts, sorted by relevance score (descending):
        caption  str           clean human-readable description
        motion   np.ndarray    shape [T, 263], standard HumanML3D features
        frames   int           number of frames (T)
        duration float         duration in seconds (at 20 fps)
        name     str           original clip ID from dataset
        score    int           keyword match score
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "pip install datasets   (HuggingFace datasets library required)"
        )

    if cached:
        # Downloads the split once (~400MB) and caches to ~/.cache/huggingface.
        # Subsequent calls are instant.  Use for local dev / testing.
        ds = load_dataset("TeoGchx/HumanML3D", split=split)
    else:
        # Streaming: no disk cache, re-downloads each run. Good for server use.
        ds = load_dataset("TeoGchx/HumanML3D", split=split, streaming=True)

    # Tokenise query; remove punctuation
    query_words = re.sub(r"[^\w\s]", "", query.lower()).split()
    if not query_words:
        return []

    results: List[dict] = []
    scanned = 0

    for row in ds:
        if scanned >= max_scan:
            break
        scanned += 1

        caption_raw = row.get("caption", "") or ""
        caption_clean = _clean_caption(caption_raw)
        caption_lower = caption_clean.lower()

        # Score: word-boundary matches count 2, substring matches count 1
        score = 0
        for kw in query_words:
            if kw in caption_lower:
                if re.search(r"\b" + re.escape(kw) + r"\b", caption_lower):
                    score += 2
                else:
                    score += 1

        if score == 0:
            continue

        motion_raw = row.get("motion")
        if motion_raw is None:
            continue

        motion = np.array(motion_raw, dtype=np.float32)  # [T, 263]
        meta   = row.get("meta_data") or {}

        T        = motion.shape[0]
        frames   = int(meta.get("num_frames", T))
        duration = float(meta.get("duration", T / 20.0))

        results.append({
            "caption":  caption_clean,
            "motion":   motion,
            "frames":   frames,
            "duration": duration,
            "name":     str(meta.get("name", "")),
            "score":    score,
        })

        # Stop as soon as we have top_k results
        if len(results) >= top_k:
            break

    results.sort(key=lambda x: -x["score"])
    return results[:top_k]


def format_choice_label(result: dict) -> str:
    """Short label for Gradio Radio component."""
    caption = result["caption"]
    if len(caption) > 72:
        caption = caption[:72] + "…"
    return f"{caption}  ({result['frames']} frames, {result['duration']:.1f}s)"
