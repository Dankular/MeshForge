from __future__ import annotations

from pathlib import Path
import hashlib


def checkpoint_signature(path: Path) -> tuple[float, float, float]:
    size = float(path.stat().st_size)
    h = hashlib.sha256()
    with path.open("rb") as f:
        chunk = f.read(4 * 1024 * 1024)
        h.update(chunk)
    digest = h.digest()
    a = int.from_bytes(digest[0:8], "big") / 2**64
    b = int.from_bytes(digest[8:16], "big") / 2**64
    c = int.from_bytes(digest[16:24], "big") / 2**64
    scale = (size % 10_000_000.0) / 10_000_000.0
    return (0.5 * a + 0.5 * scale, b, c)
