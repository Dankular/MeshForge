"""Parallel segmented downloader for FLUX fp8 via XetHub CDN.

XetHub throttles each connection to ~60 KB/s after an initial burst.
This script runs N concurrent requests to different byte ranges, each getting
its own burst window.  Segments are merged in-place via seek+write.

Usage:
    python scripts/download_flux_parallel.py [--workers N] [--seg-mb M]
"""
import argparse
import concurrent.futures
import os
import sys
import time
import threading
import requests

from _hf_env import require_hf_token

TOKEN = require_hf_token()
DEST  = "D:/Dev Proj/avatar/MeshForge/models/flux-dev-fp8.safetensors"

parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--seg-mb", type=int, default=64, help="Segment size in MB")
args = parser.parse_args()

SEG = args.seg_mb * 1024 * 1024
WORKERS = args.workers

def fresh_url():
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    url  = hf_hub_url("XLabs-AI/flux-dev-fp8", "flux-dev-fp8.safetensors")
    meta = get_hf_file_metadata(url, token=TOKEN)
    return meta.location, meta.size or 0

# ── discover total size and pre-allocate if needed ───────────────────────────
cdn, total = fresh_url()
print(f"Total: {total/1e9:.3f} GB  Workers: {WORKERS}  Seg: {args.seg_mb} MB")

if not os.path.exists(DEST) or os.path.getsize(DEST) < total:
    print(f"Pre-allocating {total/1e9:.2f} GB file ...")
    with open(DEST, "r+b" if os.path.exists(DEST) else "wb") as f:
        f.seek(total - 1)
        f.write(b"\x00")

# ── build work list: segments not yet downloaded ──────────────────────────────
# We detect "done" segments by checking if all bytes in the range are non-zero.
# For a fresh file this is wrong — pre-allocated zeros look empty.
# Instead track a simple progress file.
PROG = DEST + ".progress"
done_segments: set[int] = set()
if os.path.exists(PROG):
    with open(PROG) as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                done_segments.add(int(line))

all_segments = list(range(0, total, SEG))
todo = [s for s in all_segments if s not in done_segments]
print(f"Segments: {len(all_segments)} total, {len(todo)} remaining, "
      f"{len(done_segments)} done")

prog_lock = threading.Lock()
bytes_done = len(done_segments) * SEG
t0 = time.time()

def download_segment(seg_start: int) -> int:
    seg_end = min(seg_start + SEG - 1, total - 1)
    for attempt in range(5):
        try:
            cdn_url, _ = fresh_url()
            headers = {"Range": f"bytes={seg_start}-{seg_end}"}
            with requests.get(cdn_url, headers=headers, timeout=120) as r:
                r.raise_for_status()
                data = r.content
                if len(data) == 0:
                    raise ValueError("Empty response")
                with open(DEST, "r+b") as f:
                    f.seek(seg_start)
                    f.write(data)
                with prog_lock:
                    with open(PROG, "a") as pf:
                        pf.write(f"{seg_start}\n")
                return len(data)
        except Exception as e:
            if attempt < 4:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                print(f"\n  FAILED seg {seg_start/1e9:.3f} GB after 5 attempts: {e!r}")
                return 0
    return 0

with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futures = {ex.submit(download_segment, s): s for s in todo}
    for i, fut in enumerate(concurrent.futures.as_completed(futures)):
        n = fut.result()
        with prog_lock:
            bytes_done += n
        elapsed = time.time() - t0 + 1e-6
        mbps = bytes_done / 1e6 / elapsed
        pct  = bytes_done / total * 100
        sys.stdout.write(
            f"\r  [{i+1}/{len(todo)}]  "
            f"{bytes_done/1e9:.3f}/{total/1e9:.2f} GB  "
            f"{mbps:.1f} MB/s  {pct:.1f}%        "
        )
        sys.stdout.flush()

# Clean up progress file on success
if os.path.exists(PROG):
    os.remove(PROG)
print(f"\nDone! {os.path.getsize(DEST)/1e9:.3f} GB")
