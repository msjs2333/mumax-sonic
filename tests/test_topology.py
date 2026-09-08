import math

import numpy as np
import pytest

from mumax_sonic.observers.topology import topology


def _bp_texture(size=129, extent=8.0, center=(0.0, 0.0), winding=1):
    """Core -z, background +z Belavin-Polyakov texture."""
    points = np.linspace(-extent, extent, size)
    x, y = np.meshgrid(points, points, indexing="xy")
    x = x - center[0]
    y = y - center[1]
    radius = np.hypot(x, y)
    polar = 2.0 * np.arctan2(1.0, radius)
    azimuth = winding * np.arctan2(y, x)
    return np.stack((np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)), axis=-1), points[1] - points[0]


def _rotation():
    axis = np.array((0.3, -0.4, 0.7))
    axis /= np.linalg.norm(axis)
    angle = 0.73
    cross = np.array(((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0)))
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * cross @ cross


def test_bp_solid_angle_converges_to_negative_unit_charge():
    field, step = _bp_texture()
    result = topology(field, step, step)
    assert result.q_net == pytest.approx(-0.98, abs=0.015)
    assert result.q_neg > 0.97
    assert result.q_pos < 1e-12
    assert result.coverage == 1.0


def test_uniform_field_is_zero_for_both_estimators():
    field = np.zeros((11, 13, 3))
    field[..., 2] = 7.0  # normalization is internal and does not depend on magnitude.
    for method in ("solid_angle", "finite_difference"):
        result = topology(field, 2e-9, 3e-9, method=method)
        assert result.q_net == pytest.approx(0.0, abs=1e-15)
        assert result.q_abs == pytest.approx(0.0, abs=1e-15)


def test_spin_inversion_and_proper_rotation_have_required_effects():
    field, step = _bp_texture(97)
    original = topology(field, step, step)
    inverted = topology(-field, step, step)
    rotated = topology(field @ _rotation().T, step, step)
    assert inverted.q_net == pytest.approx(-original.q_net, rel=1e-12, abs=1e-12)
    assert rotated.q_net == pytest.approx(original.q_net, rel=1e-12, abs=1e-12)


def test_transposing_physical_xy_reverses_oriented_charge():
    field, step = _bp_texture(97)
    original = topology(field, step, step)
    transposed = topology(field.transpose(1, 0, 2), step, step)
    assert transposed.q_net == pytest.approx(-original.q_net, rel=1e-12, abs=1e-12)


def test_opposite_textures_have_zero_net_and_nonzero_absolute_charge():
    field, step = _bp_texture(161, extent=10.0, center=(-3.0, 0.0), winding=1)
    opposite, _ = _bp_texture(161, extent=10.0, center=(3.0, 0.0), winding=-1)
    # The textures are separated well enough that their deviations can be added.
    combined = field + opposite - np.array((0.0, 0.0, 1.0))
    combined /= np.linalg.norm(combined, axis=-1, keepdims=True)
    result = topology(combined, step, step)
    assert abs(result.q_net) < 0.04
    assert result.q_abs > 1.5


def test_mask_hole_and_nonfinite_sites_are_invalid_and_never_zero_filled():
    field, step = _bp_texture(33)
    mask = np.ones(field.shape[:2], dtype=bool)
    mask[14:19, 14:19] = False
    field[6, 6] = np.nan
    result = topology(field, step, step, mask=mask)
    assert not result.valid[14:19, 14:19].any()
    assert math.isnan(result.positive[14, 14])
    assert result.coverage < 1.0
    assert any("excluded" in warning for warning in result.warnings)


def test_periodic_uniform_field_covers_every_anchor_and_open_does_not():
    field = np.zeros((7, 9, 3))
    field[..., 2] = 1.0
    periodic = topology(field, 1.0, 1.0, boundary="periodic")
    open_result = topology(field, 1.0, 1.0)
    assert periodic.valid.all()
    assert periodic.coverage == 1.0
    assert not open_result.valid[-1, :].any()
    assert not open_result.valid[:, -1].any()
    assert np.array_equal(periodic.x_m[0], np.arange(9.0) + 0.5)
    assert np.array_equal(periodic.y_m[:, 0], np.arange(7.0) + 0.5)
    finite_difference = topology(field, 1.0, 1.0, method="finite_difference")
    assert np.array_equal(finite_difference.x_m[0], np.arange(9.0))


def test_ambiguous_triangle_is_excluded_and_all_invalid_totals_are_nan():
    field = np.zeros((3, 3, 3))
    field[..., 2] = np.where(np.indices((3, 3)).sum(axis=0) % 2, -1.0, 1.0)
    result = topology(field, 1.0, 1.0)
    assert not result.valid.any()
    assert math.isnan(result.q_net)
    assert any("ambiguous" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("field", "dx", "dy", "kwargs"),
    [
        (np.zeros((2, 3, 3)), 1.0, 1.0, {}),
        (np.zeros((3, 3, 2)), 1.0, 1.0, {}),
        (np.zeros((3, 3, 3)), 0.0, 1.0, {}),
        (np.zeros((3, 3, 3)), np.nan, 1.0, {}),
        (np.zeros((3, 3, 3)), 1.0, 1.0, {"orientation": 0}),
        (np.zeros((3, 3, 3)), 1.0, 1.0, {"mask": np.ones((3, 3), dtype=int)}),
    ],
)
def test_invalid_shapes_spacing_and_options_are_rejected(field, dx, dy, kwargs):
    with pytest.raises(ValueError):
        topology(field, dx, dy, **kwargs)
