"""Run MV-Adapter's own TexturePipeline on a mesh + generated views,
exactly as the VAST-AI/TripoSG Space does (uv_unwarp + view_upscale +
view inpaint). Used to validate the reference texture path before it is
wired into the pipeline.

Usage:
  python scripts/texture_reference.py --snapshot outputs/tpose_pre_rig.pkl \
      --out debug/texref [--front-x | --no-front-x] [--uv-size 2048]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# cv_ops shim must be registered before mvadapter imports (done inside):
from avatar_pipeline.baking import cv_ops_cpu  # noqa: E402

cv_ops_cpu.register()
sys.path.insert(0, str(REPO_ROOT / "external" / "MV-Adapter"))

from mvadapter.pipelines.pipeline_texture import (  # noqa: E402
    ModProcessConfig,
    TexturePipeline,
)
from mvadapter.utils import make_image_grid  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--uv-size", type=int, default=2048)
    ap.add_argument("--front-x", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.snapshot, "rb") as f:
        state = pickle.load(f)
    body = state["body"]
    views = state["artifacts"]["mv_views"]["value"]

    mesh_path = out_dir / "mesh_raw.glb"
    trimesh.Trimesh(
        vertices=body.vertices, faces=body.faces, process=False
    ).export(mesh_path)

    grid_path = out_dir / "views_grid.png"
    make_image_grid([Image.fromarray(v) for v in views], rows=1).save(grid_path)

    ckpt = REPO_ROOT / "external" / "MV-Adapter" / "checkpoints"
    t0 = time.time()
    pipe = TexturePipeline(
        upscaler_ckpt_path=str(ckpt / "RealESRGAN_x2plus.pth"),
        inpaint_ckpt_path=str(ckpt / "big-lama.pt"),
        device="cuda",
    )
    out = pipe(
        mesh_path=str(mesh_path),
        save_dir=str(out_dir),
        save_name="textured",
        uv_unwarp=True,
        uv_size=args.uv_size,
        front_x=args.front_x,
        rgb_path=str(grid_path),
        rgb_process_config=ModProcessConfig(view_upscale=True, inpaint_mode="view"),
        camera_azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
    )
    print(f"TexturePipeline done in {time.time()-t0:.0f}s")
    print("OUTPUT:", out.shaded_model_save_path)


if __name__ == "__main__":
    main()
