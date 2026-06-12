import numpy as np
from typer.testing import CliRunner

from avatar_pipeline.preprocess.tpose_reference import (
    render_tpose_skeleton,
    tpose_keypoints,
)

# OpenPose 18-keypoint indices used below
NECK, R_SHO, R_WRI, L_SHO, L_WRI = 1, 2, 4, 5, 7
R_HIP, R_KNE, R_ANK = 8, 9, 10
L_HIP, L_KNE, L_ANK = 11, 12, 13


def test_tpose_keypoints_geometry():
    width, height = 512, 768
    pts = tpose_keypoints(width, height)

    assert pts.shape == (18, 2)
    assert (pts[:, 0] > 0).all() and (pts[:, 0] < width).all()
    assert (pts[:, 1] > 0).all() and (pts[:, 1] < height).all()

    # The defining T-pose property: the whole arm chain sits on the
    # shoulder line (horizontal arms), for both sides.
    shoulder_y = pts[NECK, 1]
    for j in (R_SHO, R_WRI, L_SHO, L_WRI):
        assert abs(pts[j, 1] - shoulder_y) < 1.0

    # Arms span outward: wrists are the extreme x of each side.
    assert pts[R_WRI, 0] < pts[R_SHO, 0] < pts[NECK, 0]
    assert pts[L_WRI, 0] > pts[L_SHO, 0] > pts[NECK, 0]

    # Legs descend hip -> knee -> ankle on both sides.
    for hip, knee, ankle in ((R_HIP, R_KNE, R_ANK), (L_HIP, L_KNE, L_ANK)):
        assert pts[hip, 1] < pts[knee, 1] < pts[ankle, 1]

    # Bilateral symmetry about the vertical center line.
    for right, left in ((R_SHO, L_SHO), (R_WRI, L_WRI), (R_HIP, L_HIP), (R_ANK, L_ANK)):
        assert abs((pts[right, 0] + pts[left, 0]) / 2.0 - width / 2.0) < 1.0
        assert abs(pts[right, 1] - pts[left, 1]) < 1.0


def test_render_tpose_skeleton():
    width, height = 512, 768
    img = render_tpose_skeleton(width, height)
    arr = np.asarray(img)

    assert arr.shape == (height, width, 3)
    assert arr.dtype == np.uint8
    # Black background with a drawn skeleton: some, but few, lit pixels.
    lit = (arr.sum(axis=2) > 0).mean()
    assert 0.005 < lit < 0.25
    # Limbs are color-coded; a single-color render means the palette logic broke.
    colors = np.unique(arr.reshape(-1, 3), axis=0)
    assert len(colors) > 10


def test_cli_exposes_from_candid_flag():
    from avatar_pipeline.main import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--from-candid" in result.stdout
