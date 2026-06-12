"""Pose smoke test: bend the rig's elbows via LBS and render the result.

A rig can pass every numeric check and still candy-wrapper at the joints —
deformation quality only shows up posed. This bends both forearms down by
a canonical angle using the exported skin weights (rigid-forearm FK: the
whole subtree below each elbow shares the elbow rotation) and renders
rest vs posed front views side by side.

Usage:
    python scripts/pose_smoke.py <run_dir>   # expects <run_dir>/state.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external" / "MV-Adapter"))
sys.path.insert(0, str(REPO_ROOT / "src"))

# y-up mesh frame -> z-up camera rig frame (same as the bake pipeline).
_MESH_TO_RIG = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32
)


def _descendants(parents: np.ndarray, root: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for j, p in enumerate(parents):
        children.setdefault(int(p), []).append(j)
    out, stack = [], [root]
    while stack:
        j = stack.pop()
        out.append(j)
        stack.extend(children.get(j, []))
    return out


def find_elbows(joints: np.ndarray) -> list[int]:
    """One 'elbow' per side: the arm joint nearest 55% of the arm reach.

    Arm joints sit near the shoulder line (the y of the laterally most
    distant joint) in a T-pose.
    """
    reach = np.abs(joints[:, 0]).max()
    shoulder_y = float(joints[np.abs(joints[:, 0]).argmax(), 1])
    span_y = joints[:, 1].max() - joints[:, 1].min()
    elbows = []
    for sign in (1.0, -1.0):
        arm = [
            j for j in range(len(joints))
            if np.sign(joints[j, 0]) == sign
            and abs(joints[j, 0]) > 0.25 * reach
            and abs(joints[j, 1] - shoulder_y) < 0.15 * span_y
        ]
        if not arm:
            continue
        target = 0.55 * reach
        elbows.append(min(arm, key=lambda j: abs(abs(joints[j, 0]) - target)))
    return elbows


def pose_bend_elbows(
    verts: np.ndarray,
    weights: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
    angle_deg: float = 50.0,
) -> np.ndarray:
    """LBS with both forearms rotated downward about their elbow pivots."""
    n_joints = len(joints)
    rotations = [np.eye(3, dtype=np.float64)] * n_joints
    pivots = [np.zeros(3)] * n_joints
    bent = np.zeros(n_joints, dtype=bool)

    for elbow in find_elbows(joints):
        sign = np.sign(joints[elbow, 0])
        a = np.deg2rad(-sign * angle_deg)  # rotate the forearm downward
        c, s = np.cos(a), np.sin(a)
        rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        for j in _descendants(parents, elbow):
            rotations[j] = rot_z
            pivots[j] = joints[elbow]
            bent[j] = True
    if not bent.any():
        raise RuntimeError("No elbow joints identified; cannot pose the rig")

    v = verts.astype(np.float64)
    posed = np.zeros_like(v)
    for j in range(n_joints):
        w = weights[:, j:j + 1].astype(np.float64)
        if w.max() <= 0:
            continue
        if bent[j]:
            tv = (v - pivots[j]) @ rotations[j].T + pivots[j]
        else:
            tv = v
        posed += w * tv
    return posed.astype(np.float32)


def render_front(verts: np.ndarray, faces: np.ndarray, size: int) -> np.ndarray:
    from mvadapter.utils.mesh_utils import (
        NVDiffRastContextWrapper,
        TexturedMesh,
        get_orthogonal_camera,
        render,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scale = float(np.abs(verts).max())
    rig_verts = (verts @ _MESH_TO_RIG.T) / scale * 0.5
    mesh = TexturedMesh(
        v_pos=torch.from_numpy(rig_verts.astype(np.float32)).to(device),
        t_pos_idx=torch.from_numpy(faces.astype(np.int64)).to(device),
        v_tex=torch.zeros((len(verts), 2), device=device),
        t_tex_idx=torch.from_numpy(faces.astype(np.int64)).to(device),
        texture=torch.zeros((8, 8, 3), device=device),
    )
    cameras = get_orthogonal_camera(
        elevation_deg=[0.0], distance=[1.8],
        left=-0.55, right=0.55, bottom=-0.55, top=0.55,
        azimuth_deg=[-90.0], device=device,
    )
    ctx = NVDiffRastContextWrapper(device=device, context_type="cuda")
    out = render(
        ctx, mesh, cameras, height=size, width=size,
        render_attr=False, normal_background=0.0,
    )
    shaded = (out.normal[0] * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
    return (shaded * 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="pipeline output dir containing state.pkl")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--angle", type=float, default=50.0)
    args = ap.parse_args()

    run = Path(args.run_dir)
    with open(run / "state.pkl", "rb") as f:
        state = pickle.load(f)
    rig_entry = (state.get("artifacts") or {}).get("rig")
    if rig_entry is None:
        raise SystemExit("snapshot has no rig artifact; run the pipeline first")
    rigged = rig_entry["value"]

    verts = np.asarray(rigged.mesh.vertices, dtype=np.float32)
    faces = np.asarray(rigged.mesh.faces, dtype=np.int64)
    joints = np.asarray(rigged.joint_positions, dtype=np.float64)
    parents = np.asarray(rigged.joint_parents, dtype=np.int64)
    weights = np.asarray(rigged.skin_weights, dtype=np.float32)
    print(
        f"rig: {len(joints)} joints, weights {weights.shape}, "
        f"elbows -> {find_elbows(joints)}"
    )

    posed = pose_bend_elbows(verts, weights, joints, parents, args.angle)
    rest_img = render_front(verts, faces, args.size)
    posed_img = render_front(posed, faces, args.size)

    sheet = np.concatenate([rest_img, posed_img], axis=1)
    out_path = run / "pose_smoke.png"
    Image.fromarray(sheet).save(out_path)
    print(f"rest | posed ({args.angle:.0f} deg elbow bend): {out_path}")


if __name__ == "__main__":
    main()
