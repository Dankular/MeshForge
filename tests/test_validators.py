import numpy as np

from avatar_pipeline.models.mesh import Mesh, RiggedMesh
from avatar_pipeline.preprocess.tpose_reference import tpose_pose_score
from avatar_pipeline.runtime.validators import (
    proportion_report,
    proportion_warnings,
    rig_report,
    rig_warnings,
)

COCO17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def _block(x0, x1, y0, y1, n=400, z=0.05):
    rng = np.random.default_rng(0)
    pts = rng.uniform(
        [min(x0, x1), y0, -z], [max(x0, x1), y1, z], size=(n, 3)
    )
    return pts


def _tpose_body(head_fraction=0.12, height=1.8):
    """Synthetic T-pose vertex cloud: head block on torso, arms at the
    shoulder line spanning the full height, legs below."""
    neck_y = height * (1.0 - head_fraction)
    arm_top = neck_y
    arm_bottom = neck_y - 0.08 * height
    return np.vstack([
        _block(-0.09, 0.09, neck_y, height),            # head
        _block(-0.18, 0.18, 0.9 * neck_y / 1.0, neck_y),  # upper torso
        _block(-0.9, 0.9, arm_bottom, arm_top),          # arms (wingspan)
        _block(-0.18, 0.18, 0.0, 0.9 * neck_y),          # lower body
    ]).astype(np.float32)


def test_proportions_pass_for_anatomical_tpose():
    report = proportion_report(_tpose_body(head_fraction=0.12))
    assert report["head_fraction"] is not None
    assert 0.10 <= report["head_fraction"] <= 0.16
    assert 0.9 <= report["wingspan_ratio"] <= 1.1
    assert proportion_warnings(report) == []


def test_proportions_flag_vestigial_head():
    report = proportion_report(_tpose_body(head_fraction=0.03))
    warnings = proportion_warnings(report)
    assert any("head fraction" in w or "neck line" in w for w in warnings)


def _tpose_keypoints(arm_drop=0.0, conf=0.9):
    """COCO-17 (x, y, conf) for a 1000px-tall figure facing the camera.
    Image y grows downward; person's left appears at larger x."""
    shoulder_y = 250.0
    kp = {
        "nose": (500, 150), "left_eye": (515, 140), "right_eye": (485, 140),
        "left_ear": (530, 145), "right_ear": (470, 145),
        "left_shoulder": (560, shoulder_y), "right_shoulder": (440, shoulder_y),
        "left_elbow": (680, shoulder_y + arm_drop),
        "right_elbow": (320, shoulder_y + arm_drop),
        "left_wrist": (800, shoulder_y + arm_drop),
        "right_wrist": (200, shoulder_y + arm_drop),
        "left_hip": (540, 520), "right_hip": (460, 520),
        "left_knee": (535, 720), "right_knee": (465, 720),
        "left_ankle": (530, 920), "right_ankle": (470, 920),
    }
    return np.array(
        [[*kp[name], conf] for name in COCO17], dtype=np.float32
    )


def test_pose_score_accepts_strict_tpose():
    score = tpose_pose_score(_tpose_keypoints(arm_drop=10.0), COCO17)
    assert score["valid"]
    assert score["arm_dev"] < 0.05


def test_pose_score_rejects_dropped_arms():
    # Arms hanging 60% of torso length below the shoulder line.
    score = tpose_pose_score(_tpose_keypoints(arm_drop=160.0), COCO17)
    assert not score["valid"]
    assert score["arm_dev"] > 0.5


def test_pose_score_rejects_low_confidence():
    score = tpose_pose_score(_tpose_keypoints(conf=0.1), COCO17)
    assert not score["valid"]
    assert not score["conf_ok"]


def _rigged(weight_break=False):
    verts = _tpose_body().astype(np.float32)
    mesh = Mesh(vertices=verts, faces=np.zeros((1, 3), dtype=np.int32))
    joints = np.array(
        [[0.0, 0.9, 0.0], [0.0, 1.4, 0.0],
         [0.45, 1.55, 0.0], [-0.45, 1.55, 0.0],
         [0.85, 1.55, 0.0], [-0.85, 1.55, 0.0]],
        dtype=np.float32,
    )
    weights = np.zeros((len(verts), len(joints)), dtype=np.float32)
    weights[:, 0] = 1.0
    if weight_break:
        weights[0, 0] = 0.5  # row no longer sums to 1
    return RiggedMesh(
        mesh=mesh,
        joint_names=[f"bone_{i}" for i in range(len(joints))],
        joint_positions=joints,
        skin_weights=weights,
        joint_parents=np.array([-1, 0, 1, 1, 2, 3], dtype=np.int32),
    )


def test_rig_report_clean_for_symmetric_rig():
    report = rig_report(_rigged())
    assert report["weight_sum_max_dev"] < 1e-6
    assert report["asymmetric_joints"] == 0
    assert rig_warnings(report) == []


def test_rig_report_flags_unnormalized_weights():
    warnings = rig_warnings(rig_report(_rigged(weight_break=True)))
    assert any("weight" in w for w in warnings)
