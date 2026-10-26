"""Resumable chunked downloader for FLUX fp8 via XetHub.

XetHub CDN signs URLs with ~1-hour expiry and stalls downloads mid-stream.
This script fetches a fresh signed URL before each 256 MB chunk, appending
to the partial file so any restart picks up where it left off.
"""
import os
import sys
import time
import requests
from pathlib import Path

from _hf_env import require_hf_token

TOKEN = require_hf_token()
DEST  = Path("D:/Dev Proj/avatar/MeshForge/models/flux-dev-fp8.safetensors")
CHUNK = 256 * 1024 * 1024   # 256 MB per signed URL (well within 1-hour expiry)

DEST.parent.mkdir(parents=True, exist_ok=True)

def fresh_url() -> tuple[str, int]:
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    url  = hf_hub_url("XLabs-AI/flux-dev-fp8", "flux-dev-fp8.safetensors")
    meta = get_hf_file_metadata(url, token=TOKEN)
    return meta.location, meta.size or 0

offset = DEST.stat().st_size if DEST.exists() else 0

cdn, total = fresh_url()
print(f"Total: {total/1e9:.2f} GB  Resuming from: {offset/1e9:.3f} GB")

t_start = time.time()
bytes_this_session = 0

while offset < total:
    chunk_end = min(offset + CHUNK - 1, total - 1)
    # Refresh URL for every chunk (each chunk takes <10 min at even 1 MB/s)
    cdn, _ = fresh_url()

    headers = {"Range": f"bytes={offset}-{chunk_end}"}
    try:
        with requests.get(cdn, headers=headers, stream=True, timeout=90) as r:
            r.raise_for_status()
            with open(DEST, "ab") as f:
                chunk_bytes = 0
                t_chunk = time.time()
                for data in r.iter_content(chunk_size=4 * 1024 * 1024):
                    f.write(data)
                    chunk_bytes += len(data)
                    offset += len(data)
                    bytes_this_session += len(data)
                    elapsed = time.time() - t_start + 1e-6
                    mbps = bytes_this_session / 1e6 / elapsed
                    pct  = offset / total * 100
                    sys.stdout.write(
                        f"\r  {offset/1e9:.3f}/{total/1e9:.2f} GB  "
                        f"{mbps:.1f} MB/s  {pct:.1f}%      "
                    )
                    sys.stdout.flush()
    except (requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError) as e:
        print(f"\n  Connection dropped at {offset/1e9:.3f} GB: {e!r} — retrying ...")
        time.sleep(5)
        offset = DEST.stat().st_size  # re-read actual written bytes

print(f"\nDone! {offset/1e9:.2f} GB written to {DEST}")
