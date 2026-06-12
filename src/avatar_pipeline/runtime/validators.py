"""Cheap quantitative validators for reconstruction and rig outputs.

Every expensive failure this pipeline has produced was visible in
milliseconds of mesh math: the vestigial-head TripoSG body (head occupying
3% of standing height) cost a full bake to discover by eye, yet its width
profile told the story instantly. These validators print [validate] reports
right after the mesh and rig stages so anatomical breakage surfaces before
the next 30 minutes of GPU work — and they double as the metric set for A/B
comparison between reconstruction arms.

Warnings, not hard failures: a TripoSG body with off proportions still
textures and rigs; the point is observability and comparability.
"""
from __future__ import annotations

import numpy as np

# Anatomical ranges for an adult figure in a T-pose (Vitruvian-ish):
# head 1/10 .. 1/5.5 of standing height; wingspan roughly equal to height.
HEAD_FRACTION_RANGE = (0.10, 0.18)
WINGSPAN_RATIO_RANGE = (0.85, 1.20)
CENTER_OFFSET_LIMIT = 0.05  # lateral asymmetry, fraction of wingspan


def proportion_report(verts: np.ndarray) -> dict:
    """Anatomical proportion metrics of a y-up T-pose body mesh."""
    from avatar_pipeline.fusion.head_fusion import _neck_line

    v = np.asarray(verts, dtype=np.float64)
    y = v[:, 1]
    height = float(y.max() - y.min())
    wingspan = float(v[:, 0].max() - v[:, 0].min())

    neck_y = _neck_line(v)
    head_fraction = (
        None if neck_y is None or height <= 0
        else float((y.max() - neck_y) / height)
    )

    # Lateral symmetry: the upper-body midline should match the full-body
    # midline (a leaning or lopsided reconstruction drifts).
    upper = v[y >= y.min() + 0.6 * height]
    center_offset = (
        abs(float(upper[:, 0].min() + upper[:, 0].max()) / 2.0
            - float(v[:, 0].min() + v[:, 0].max()) / 2.0) / max(wingspan, 1e-9)
    )

    return {
        "height": height,
        "wingspan": wingspan,
        "wingspan_ratio": wingspan / max(height, 1e-9),
        "head_fraction": head_fraction,
        "center_offset": center_offset,
        "vertex_count": int(len(v)),
    }


def proportion_warnings(report: dict) -> list[str]:
    warnings = []
    hf = report["head_fraction"]
    if hf is None:
        warnings.append(
            "no neck line found in the width profile (headless or "
            "non-T-pose reconstruction?)"
        )
    elif not HEAD_FRACTION_RANGE[0] <= hf <= HEAD_FRACTION_RANGE[1]:
        warnings.append(
            f"head fraction {hf:.1%} outside anatomical range "
            f"{HEAD_FRACTION_RANGE[0]:.0%}-{HEAD_FRACTION_RANGE[1]:.0%} "
            "(vestigial or oversized head)"
        )
    wr = report["wingspan_ratio"]
    if not WINGSPAN_RATIO_RANGE[0] <= wr <= WINGSPAN_RATIO_RANGE[1]:
        warnings.append(
            f"wingspan/height ratio {wr:.2f} outside "
            f"{WINGSPAN_RATIO_RANGE[0]}-{WINGSPAN_RATIO_RANGE[1]} "
            "(arms not in T-pose, or distorted proportions)"
        )
    if report["center_offset"] > CENTER_OFFSET_LIMIT:
        warnings.append(
            f"upper-body midline off center by "
            f"{report['center_offset']:.1%} of wingspan (leaning figure)"
        )
    return warnings


def print_proportion_report(label: str, verts: np.ndarray) -> dict:
    report = proportion_report(verts)
    hf = report["head_fraction"]
    print(
        f"[validate] {label}: height={report['height']:.3f} "
        f"wingspan_ratio={report['wingspan_ratio']:.2f} "
        f"head_fraction={'n/a' if hf is None else f'{hf:.1%}'} "
        f"center_offset={report['center_offset']:.1%}"
    )
    for w in proportion_warnings(report):
        print(f"[validate]   WARNING: {w}")
    return report


# ── Rig validation ───────────────────────────────────────────────────────────

def rig_report(rigged) -> dict:
    """Sanity metrics for a RiggedMesh: weight hygiene, joint placement,
    left/right joint symmetry."""
    weights = np.asarray(rigged.skin_weights, dtype=np.float64)
    row_sums = weights.sum(axis=1)
    influences = (weights > 1e-6).sum(axis=1)

    joints = np.asarray(rigged.joint_positions, dtype=np.float64)
    verts = np.asarray(rigged.mesh.vertices, dtype=np.float64)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    margin = 0.05 * (hi - lo)
    outside = np.any((joints < lo - margin) | (joints > hi + margin), axis=1)

    # Symmetry: every laterally-offset joint should have a mirrored partner.
    span = max(float(hi[0] - lo[0]), 1e-9)
    lateral = joints[np.abs(joints[:, 0]) > 0.02 * span]
    unmatched = 0
    for j in lateral:
        mirrored = j * np.array([-1.0, 1.0, 1.0])
        d = np.linalg.norm(joints - mirrored, axis=1).min()
        if d > 0.05 * span:
            unmatched += 1

    return {
        "joint_count": int(len(joints)),
        "weight_sum_max_dev": float(np.abs(row_sums - 1.0).max()),
        "max_influences": int(influences.max()),
        "unweighted_vertices": int((row_sums < 1e-6).sum()),
        "joints_outside_bbox": int(outside.sum()),
        "lateral_joints": int(len(lateral)),
        "asymmetric_joints": int(unmatched),
    }


def rig_warnings(report: dict) -> list[str]:
    warnings = []
    if report["weight_sum_max_dev"] > 1e-3:
        warnings.append(
            f"skin weight rows deviate from 1.0 by up to "
            f"{report['weight_sum_max_dev']:.4f}"
        )
    if report["unweighted_vertices"]:
        warnings.append(
            f"{report['unweighted_vertices']} vertices carry no skin weight"
        )
    if report["joints_outside_bbox"]:
        warnings.append(
            f"{report['joints_outside_bbox']} joints sit outside the mesh "
            "bounding box (+5% margin)"
        )
    if report["lateral_joints"] and (
        report["asymmetric_joints"] > 0.2 * report["lateral_joints"]
    ):
        warnings.append(
            f"{report['asymmetric_joints']}/{report['lateral_joints']} "
            "lateral joints have no mirrored partner (asymmetric skeleton)"
        )
    return warnings


def print_rig_report(rigged) -> dict:
    report = rig_report(rigged)
    print(
        f"[validate] rig: joints={report['joint_count']} "
        f"weight_dev={report['weight_sum_max_dev']:.5f} "
        f"max_influences={report['max_influences']} "
        f"asymmetric={report['asymmetric_joints']}/{report['lateral_joints']}"
    )
    for w in rig_warnings(report):
        print(f"[validate]   WARNING: {w}")
    return report
