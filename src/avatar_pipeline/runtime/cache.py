"""Content-addressed artifact caching for pipeline stages.

Each expensive stage output is cached under a key derived from the sha256 of
its actual inputs (array contents, parameters) plus a per-stage version
constant. Keys are used ONLY for cache identity — outputs always come from
real computation; bumping a stage version invalidates exactly that stage and
its dependents, nothing else.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

# Bump when the corresponding stage's semantics change.
MV_VIEWS_VERSION = 3        # MV-Adapter diffusion, Space-exact (15 steps)
TEXTURED_MESH_VERSION = 1   # reference TexturePipeline (UVAtlas unwrap + texture)
NORMAL_BAKE_VERSION = 2     # v2: atlas row convention aligned with albedo
AO_BAKE_VERSION = 2         # v2: atlas row convention aligned with albedo
RIG_VERSION = 1             # SkinTokens skeleton + weights
HEAD_DETAIL_VERSION = 2     # PSHuman head generation; colored OBJ required
HEAD_COMPOSITE_VERSION = 3  # v3: composite mask row convention aligned
FULL_BODY_VERSION = 1       # PSHuman full-figure reconstruction (--full-pshuman)


def content_key(*parts: Any) -> str:
    """sha256 over array bytes / strings / numbers, order-sensitive."""
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            h.update(np.ascontiguousarray(part).tobytes())
            h.update(str(part.shape).encode())
            h.update(str(part.dtype).encode())
        elif isinstance(part, (bytes, bytearray)):
            h.update(part)
        else:
            h.update(repr(part).encode())
        h.update(b"|")
    return h.hexdigest()


def artifact_get(state: dict, name: str, key: str):
    """Return the cached value for *name* if its key matches, else None."""
    entry = (state.get("artifacts") or {}).get(name)
    if entry is None or entry.get("key") != key:
        return None
    return entry.get("value")


def artifact_put(state: dict, name: str, key: str, value) -> None:
    state.setdefault("artifacts", {})[name] = {"key": key, "value": value}
