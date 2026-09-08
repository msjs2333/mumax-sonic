import math

import numpy as np
import pytest

from mumax_sonic.observers.texture import direction_angle, wall_angle


def _direction(phi, axis=(1.0, 0.0, 0.0)):
    axis = np.asarray(axis, dtype=float)
    e1 = np.asarray((0.0, 1.0, 0.0))
    e1 -= np.dot(e1, axis) * axis / np.dot(axis, axis)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis / np.linalg.norm(axis), e1)
    return math.cos(phi) * e1 + math.sin(phi) * e2


def test_inplane_angle_and_out_of_plane_projection():
    values = np.array([[_direction(-math.pi + 0.02), _direction(math.pi - 0.02)]])
    result = direction_angle(values)
    assert result.valid.tolist() == [[True, True]]
    assert np.allclose(result.angle_rad, [[-math.pi + 0.02, math.pi - 0.02]])
    assert np.allclose(result.cos_angle[0, 0], result.cos_angle[0, 1], atol=1e-3)
    assert np.allclose(result.sin_angle[0, 0], -result.sin_angle[0, 1], atol=1e-3)

    out_of_plane = np.array([[[0.0, 1.0, 2.0]]])
    out = direction_angle(out_of_plane)
    assert out.valid[0, 0]
    assert np.allclose(np.asarray(out.projection)[0, 0], (0.0, 1.0, 2.0) / np.sqrt(5.0))


def test_rotating_axes_preserves_local_angle_and_axes_are_right_handed():
    # A proper rotation applied to both the field and the explicit axes must
    # preserve the measured local angle.
    angle = 0.73
    values = np.array([[ _direction(angle) ]])
    rotation = np.array(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    rotated = values @ rotation.T
    result = direction_angle(rotated, rotation @ np.array((1.0, 0.0, 0.0)),
                              rotation @ np.array((0.0, 1.0, 0.0)))
    assert result.valid[0, 0]
    assert result.angle_rad[0, 0] == pytest.approx(angle)
    assert np.allclose(np.cross(result.e1, result.e2), result.d)


def test_projection_floor_mask_and_invalid_vectors_are_explicit():
    values = np.array([[[1.0, 1e-8, 0.0], [0.0, 1.0, 0.0], [np.nan, 0.0, 0.0]]])
    mask = np.array([[True, False, True]])
    result = direction_angle(values, mask=mask, projection_floor=1e-6)
    assert result.valid.tolist() == [[False, False, False]]
    assert np.all(np.isnan(result.angle_rad))
    assert np.all(np.isnan(result.projection[~result.valid]))


def test_inputs_are_not_mutated():
    values = np.array([[[2.0, 1.0, 0.0]]])
    domain = np.array((3.0, 0.0, 0.0))
    reference = np.array((0.0, 4.0, 0.0))
    values_before, domain_before, reference_before = values.copy(), domain.copy(), reference.copy()
    direction_angle(values, domain, reference)
    assert np.array_equal(values, values_before)
    assert np.array_equal(domain, domain_before)
    assert np.array_equal(reference, reference_before)


def test_degenerate_reference_and_unordered_domains_are_rejected():
    with pytest.raises(ValueError):
        direction_angle(np.array([[[0.0, 1.0, 0.0]]]), domain_axis=(1, 0, 0), reference_axis=(2, 0, 0))
    with pytest.raises(ValueError):
        wall_angle(np.array([[[0.0, 1.0, 0.0]]]), (1, 0, 0), (0, 1, 0), (0, 0, 1))


def test_wall_angle_uses_ordered_antiparallel_difference():
    core = np.array([[[0.0, -1.0, 0.0]]])
    result = wall_angle(core, (0, 0, 1), (0, 0, -1), (1, 0, 0))
    assert result.valid[0, 0]
    assert result.angle_rad[0, 0] == pytest.approx(math.pi / 2.0)
