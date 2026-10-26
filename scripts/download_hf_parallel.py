"""General-purpose parallel segmented downloader for XetHub-backed HF files.

XetHub throttles single connections to ~60-80 KB/s after a brief burst.
This script opens N concurrent HTTP range-request connections so each gets
its own burst window, achieving much higher aggregate throughput.

Usage:
    python scripts/download_hf_parallel.py REPO FILE DEST [--workers N] [--seg-mb M]

Examples:
    python scripts/download_hf_parallel.py \\
        receptektas/black-forest-labs-ae_safetensors ae.safetensors models/ae.safetensors

    python scripts/download_hf_parallel.py \\
        xlabs-ai/xflux_text_encoders t5xxl_fp16.safetensors \\
        "D:/HuggingFaceCache/huggingface/hub/models--xlabs-ai--xflux_text_encoders/snapshots/main/t5xxl_fp16.safetensors"
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
parser = argparse.ArgumentParser()
parser.add_argument("repo",    help="HuggingFace repo id  (e.g. receptektas/black-forest-labs-ae_safetensors)")
parser.add_argument("file",    help="Filename inside the repo  (e.g. ae.safetensors)")
parser.add_argument("dest",    help="Local destination path")
parser.add_argument("--workers", type=int, default=32)
parser.add_argument("--seg-mb",  type=int, default=32, help="Segment size in MB")
parser.add_argument("--revision", default="main")
args = parser.parse_args()

SEG     = args.seg_mb * 1024 * 1024
WORKERS = args.workers
DEST    = args.dest

def fresh_url() -> tuple[str, int]:
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    url  = hf_hub_url(args.repo, args.file, revision=args.revision)
    meta = get_hf_file_metadata(url, token=TOKEN)
    return meta.location, meta.size or 0

os.makedirs(os.path.dirname(os.path.abspath(DEST)), exist_ok=True)

cdn, total = fresh_url()
print(f"Repo: {args.repo}/{args.file}  ({total/1e6:.1f} MB)")
print(f"Dest: {DEST}")
print(f"Workers: {WORKERS}  Seg: {args.seg_mb} MB")

if not os.path.exists(DEST) or os.path.getsize(DEST) < total:
    print(f"Pre-allocating {total/1e6:.1f} MB ...")
    with open(DEST, "r+b" if os.path.exists(DEST) else "wb") as f:
        f.seek(total - 1)
        f.write(b"\x00")

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
print(f"Segments: {len(all_segments)} total, {len(todo)} remaining, {len(done_segments)} done")

if not todo:
    print("Already complete.")
    sys.exit(0)

prog_lock = threading.Lock()
bytes_done = len(done_segments) * SEG
t0 = time.time()

def download_segment(seg_start: int) -> int:
    seg_end = min(seg_start + SEG - 1, total - 1)
    for attempt in range(5):
        try:
            cdn_url, _ = fresh_url()
            headers = {"Range": f"bytes={seg_start}-{seg_end}"}
            with requests.get(cdn_url, headers=headers, timeout=300) as r:
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
                time.sleep(2 ** attempt)
            else:
                print(f"\n  FAILED seg {seg_start/1e6:.0f} MB after 5 attempts: {e!r}")
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
            f"{bytes_done/1e6:.0f}/{total/1e6:.0f} MB  "
            f"{mbps:.1f} MB/s  {pct:.1f}%        "
        )
        sys.stdout.flush()

if os.path.exists(PROG):
    os.remove(PROG)
size = os.path.getsize(DEST)
print(f"\nDone! {size/1e6:.1f} MB at {DEST}")
