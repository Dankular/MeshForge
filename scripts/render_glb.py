"""Render a textured GLB to PNG turntable views for visual inspection.

Single-file debug tool: uses the same nvdiffrast + mvadapter rendering
stack the pipeline bakes with, so what you see is what the bake produced.

Usage:
    python scripts/render_glb.py <model.glb> [--out debug] [--size 768]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external" / "MV-Adapter"))

from mvadapter.utils.mesh_utils import (  # noqa: E402
    NVDiffRastContextWrapper,
    get_orthogonal_camera,
    load_mesh,
    render,
)

VIEWS = [
    ("front", 0.0, 0.0),
    ("three_quarter", 45.0, 0.0),
    ("side", 90.0, 0.0),
    ("back", 180.0, 0.0),
    ("top", 0.0, 60.0),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("glb")
    ap.add_argument("--out", default=str(REPO_ROOT / "debug"))
    ap.add_argument("--size", type=int, default=768)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = NVDiffRastContextWrapper(device=device, context_type="cuda")
    mesh = load_mesh(args.glb, rescale=True, device=device)
    print(f"loaded {args.glb}: {len(mesh.v_pos):,} verts")

    azimuths = [a - 90.0 for (_, a, _) in VIEWS]
    elevations = [e for (_, _, e) in VIEWS]
    cameras = get_orthogonal_camera(
        elevation_deg=elevations,
        distance=[1.8] * len(VIEWS),
        left=-0.55, right=0.55, bottom=-0.55, top=0.55,
        azimuth_deg=azimuths,
        device=device,
    )
    textured = mesh.v_tex is not None and mesh.texture is not None
    out = render(
        ctx, mesh, cameras,
        height=args.size, width=args.size,
        render_attr=textured, attr_background=0.25,
        normal_background=0.0,
    )
    shaded = out.attr if textured else (out.normal * 0.5 + 0.5)

    stem = Path(args.glb).stem
    paths = []
    for i, (name, _, _) in enumerate(VIEWS):
        rgb = (shaded[i].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        p = out_dir / f"{stem}_{name}.png"
        Image.fromarray(rgb).save(p)
        paths.append(p)
        print(f"  {p}")

    # Contact sheet for one-look inspection
    tiles = [np.asarray(Image.open(p)) for p in paths]
    sheet = np.concatenate(tiles, axis=1)
    sheet_path = out_dir / f"{stem}_sheet.png"
    Image.fromarray(sheet).save(sheet_path)
    print(f"  {sheet_path}")


if __name__ == "__main__":
    main()
