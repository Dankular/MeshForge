"""A/B comparison harness for pipeline runs.

Takes two (or more) run output dirs and emits a side-by-side render sheet
plus a metrics table: mesh stats, anatomical proportions, silhouette IoU
against the run's own rembg alpha, FaceID identity similarity against the
original candid photo, and rig hygiene. Makes every reconstruction-arm
iteration comparable on numbers instead of impressions.

Usage:
    python scripts/compare_runs.py outputs\\runA outputs\\runB
        [--candid path\\to\\candid.png] [--out outputs\\compare]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external" / "MV-Adapter"))
sys.path.insert(0, str(REPO_ROOT / "src"))


def render_views(glb_path: Path, size: int = 640) -> dict[str, np.ndarray]:
    """Front + three-quarter textured renders, plus the front coverage mask."""
    from mvadapter.utils.mesh_utils import (
        NVDiffRastContextWrapper,
        get_orthogonal_camera,
        load_mesh,
        render,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = NVDiffRastContextWrapper(device=device, context_type="cuda")
    mesh = load_mesh(str(glb_path), rescale=True, device=device)
    cameras = get_orthogonal_camera(
        elevation_deg=[0.0, 0.0],
        distance=[1.8, 1.8],
        left=-0.55, right=0.55, bottom=-0.55, top=0.55,
        azimuth_deg=[-90.0, -45.0],
        device=device,
    )
    out = render(
        ctx, mesh, cameras, height=size, width=size,
        render_attr=True, attr_background=0.25, normal_background=0.0,
    )
    front = (out.attr[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    three_q = (out.attr[1].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    mask = out.mask[0].cpu().numpy().astype(bool)
    return {"front": front, "three_quarter": three_q, "front_mask": mask}


def silhouette_iou(render_mask: np.ndarray, processed_alpha: np.ndarray) -> float:
    """IoU of bbox-normalized silhouettes (render vs rembg alpha)."""
    import cv2

    def normalized(mask: np.ndarray, size: int = 256) -> np.ndarray:
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return np.zeros((size, size), dtype=bool)
        crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return cv2.resize(
            crop.astype(np.uint8), (size, size),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    a = normalized(render_mask)
    b = normalized(processed_alpha > 0.5)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def run_metrics(run_dir: Path, candid_rgb: np.ndarray | None) -> dict:
    import trimesh

    from avatar_pipeline.runtime.validators import (
        proportion_report,
        proportion_warnings,
        rig_report,
        rig_warnings,
    )

    glb = run_dir / "avatar.glb"
    meta = json.loads((glb.with_suffix(".meta.json")).read_text(encoding="utf-8"))
    tm = trimesh.load(glb, force="mesh", process=False)
    verts = np.asarray(tm.vertices, dtype=np.float64)

    metrics: dict = {
        "run": run_dir.name,
        "vertices": meta["mesh"]["vertex_count"],
        "faces": meta["mesh"]["face_count"],
        "joints": meta["rig"]["joint_count"],
    }
    prop = proportion_report(verts)
    metrics["proportions"] = prop
    metrics["proportion_warnings"] = proportion_warnings(prop)

    views = render_views(glb)
    metrics["_views"] = views

    state_path = run_dir / "state.pkl"
    if state_path.exists():
        with open(state_path, "rb") as f:
            state = pickle.load(f)
        processed = state.get("processed")
        if processed is not None:
            metrics["silhouette_iou"] = silhouette_iou(
                views["front_mask"], processed[:, :, 3]
            )
        rig_entry = (state.get("artifacts") or {}).get("rig")
        if rig_entry is not None:
            rep = rig_report(rig_entry["value"])
            metrics["rig"] = rep
            metrics["rig_warnings"] = rig_warnings(rep)

    if candid_rgb is not None:
        from avatar_pipeline.preprocess.face_identity import identity_similarity

        metrics["identity_vs_candid"] = identity_similarity(
            candid_rgb, views["front"]
        )
    return metrics


def fmt(value, spec: str = "") -> str:
    if value is None:
        return "n/a"
    return format(value, spec) if spec else str(value)


def write_report(all_metrics: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Side-by-side sheet: front + three-quarter per run, labeled.
    import cv2

    columns = []
    for m in all_metrics:
        tile = np.concatenate(
            [m["_views"]["front"], m["_views"]["three_quarter"]], axis=0
        )
        cv2.putText(
            tile, m["run"], (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (255, 255, 255), 2,
        )
        columns.append(tile)
    sheet = np.concatenate(columns, axis=1)
    Image.fromarray(sheet).save(out_dir / "sheet.png")

    lines = ["# A/B run comparison", ""]
    header = ["metric"] + [m["run"] for m in all_metrics]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    def row(label: str, getter, spec: str = "") -> None:
        cells = [fmt(getter(m), spec) for m in all_metrics]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    row("vertices", lambda m: m["vertices"], ",")
    row("faces", lambda m: m["faces"], ",")
    row("joints", lambda m: m["joints"])
    row("height", lambda m: m["proportions"]["height"], ".3f")
    row("wingspan/height", lambda m: m["proportions"]["wingspan_ratio"], ".2f")
    row("head fraction", lambda m: m["proportions"]["head_fraction"], ".1%")
    row("center offset", lambda m: m["proportions"]["center_offset"], ".1%")
    row("silhouette IoU", lambda m: m.get("silhouette_iou"), ".3f")
    row("identity vs candid", lambda m: m.get("identity_vs_candid"), ".3f")
    row(
        "rig weight dev",
        lambda m: (m.get("rig") or {}).get("weight_sum_max_dev"),
        ".5f",
    )
    row(
        "rig asymmetric joints",
        lambda m: (m.get("rig") or {}).get("asymmetric_joints"),
    )
    lines.append("")
    for m in all_metrics:
        warnings = m.get("proportion_warnings", []) + m.get("rig_warnings", [])
        if warnings:
            lines.append(f"## Warnings — {m['run']}")
            lines.extend(f"- {w}" for w in warnings)
            lines.append("")
    lines.append("![side-by-side](sheet.png)")

    report = out_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"report: {report}")
    print(f"sheet:  {out_dir / 'sheet.png'}")
    print()
    print("\n".join(lines[2:len(header) + 14]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="pipeline output dirs")
    ap.add_argument("--candid", help="original candid photo for identity scoring")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "compare"))
    args = ap.parse_args()

    candid_rgb = None
    if args.candid:
        candid_rgb = np.array(
            Image.open(args.candid).convert("RGB"), dtype=np.uint8
        )

    all_metrics = [run_metrics(Path(r), candid_rgb) for r in args.runs]
    write_report(all_metrics, Path(args.out))


if __name__ == "__main__":
    main()
