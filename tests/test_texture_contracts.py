import re

import numpy as np
import pytest

from avatar_pipeline.models.mesh import BakedTextures
from avatar_pipeline.runtime.contracts import (
    TEXTURE_SCHEMA_VERSION,
    albedo_active_ratio,
    has_sparse_albedo,
    validate_texture_contracts,
)


def _textures(albedo: np.ndarray | None = None) -> BakedTextures:
    size = 16
    return BakedTextures(
        albedo=(
            albedo
            if albedo is not None
            else np.full((size, size, 3), 0.5, dtype=np.float32)
        ),
        normal=np.dstack(
            (
                np.zeros((size, size), dtype=np.float32),
                np.zeros((size, size), dtype=np.float32),
                np.ones((size, size), dtype=np.float32),
            )
        ),
        ambient_occlusion=np.ones((size, size, 1), dtype=np.float32),
    )


def test_texture_schema_version_is_explicit():
    assert TEXTURE_SCHEMA_VERSION == 5


def test_texture_contract_accepts_float_maps_with_matching_resolution():
    validate_texture_contracts(_textures())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("albedo", 1.1, "Albedo values must be in [0, 1]"),
        ("normal", -1.1, "Normal values must be in [-1, 1]"),
        ("ambient_occlusion", np.nan, "AO contains non-finite values"),
    ],
)
def test_texture_contract_rejects_invalid_values(field, value, message):
    textures = _textures()
    array = getattr(textures, field).copy()
    array.flat[0] = value
    setattr(textures, field, array)

    with pytest.raises(ValueError, match=re.escape(message)):
        validate_texture_contracts(textures)


def test_sparse_albedo_diagnostic_matches_known_snapshot_failure():
    albedo = np.zeros((64, 64, 3), dtype=np.float32)
    albedo[0, 0] = 0.8
    textures = _textures(albedo)

    assert albedo_active_ratio(albedo) == pytest.approx(1.0 / (64 * 64))
    assert has_sparse_albedo(textures)


def test_populated_dark_albedo_is_not_sparse():
    albedo = np.full((16, 16, 3), 2.0 / 255.0, dtype=np.float32)

    assert not has_sparse_albedo(_textures(albedo))
