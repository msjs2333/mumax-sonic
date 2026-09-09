"""Scientific invariants for the pairwise angular activity observer.

These cases keep their reference calculation separate from the observer's
implementation so numerical refactors remain constrained by the definition.
"""

import math

import numpy as np
import pytest

from mumax_sonic.fields import FieldFrame
from mumax_sonic.observers.activity import angular_activity


def _frame(vectors, *, time=0.0, sequence=0, mask=None, **kwargs):
    values = np.asarray(vectors, dtype=float)
    if mask is None:
        mask = np.ones(values.shape[:2], dtype=bool)
    return FieldFrame(
        values,
        dx_m=2e-9,
        dy_m=3e-9,
        sim_time_s=time,
        sequence=sequence,
        mask=np.asarray(mask, dtype=bool),
        **kwargs,
    )


def _reference_rate(previous, current, dt):
    """Stable scalar reference: normalize with math.hypot, then atan2."""
    out = np.full(previous.shape[:2], np.nan, dtype=float)
    for index in np.ndindex(previous.shape[:2]):
        a = tuple(float(x) for x in previous[index])
        b = tuple(float(x) for x in current[index])
        if not all(math.isfinite(x) for x in (*a, *b)):
            continue
        na = math.hypot(*a)
        nb = math.hypot(*b)
        if na == 0.0 or nb == 0.0:
            continue
        au = tuple(x / na for x in a)
        bu = tuple(x / nb for x in b)
        dot = sum(x * y for x, y in zip(au, bu))
        cross = (
            au[1] * bu[2] - au[2] * bu[1],
            au[2] * bu[0] - au[0] * bu[2],
            au[0] * bu[1] - au[1] * bu[0],
        )
        angle = math.atan2(math.hypot(*cross), min(1.0, max(-1.0, dot)))
        out[index] = angle / dt
    return out


def test_random_non_coplanar_xyz_matches_independent_reference_over_300_decades():
    rng = np.random.default_rng(20260910)
    shape = (7, 11)
    prior = rng.normal(size=(*shape, 3))
    later = rng.normal(size=(*shape, 3))
    scales = np.power(10.0, rng.uniform(-300.0, 300.0, size=(*shape, 1)))
    prior *= scales
    later *= np.power(10.0, rng.uniform(-300.0, 300.0, size=(*shape, 1)))
    dt = 0.037
    result = angular_activity(_frame(prior), _frame(later, time=dt, sequence=1))
    expected = _reference_rate(prior, later, dt)
    assert result.validity == "valid"
    assert np.all(result.valid)
    np.testing.assert_allclose(result.rate_rad_s, expected, rtol=3e-13, atol=2e-13)


@pytest.mark.parametrize("angle", [1e-14, 1e-10, math.pi - 1e-10, math.pi - 1e-14])
def test_known_rotations_remain_accurate_near_zero_and_pi(angle):
    prior = np.empty((3, 3, 3), dtype=float)
    prior[...] = (0.6, -0.8, 0.0)
    # Rodrigues rotation, with a non-axis-aligned starting direction.
    x, y, z = prior[0, 0]
    rotated = (x * math.cos(angle) - y * math.sin(angle),
               x * math.sin(angle) + y * math.cos(angle), z)
    later = np.empty_like(prior)
    later[...] = rotated
    dt = 0.25
    result = angular_activity(_frame(prior), _frame(later, time=dt, sequence=1))
    assert result.rate_rad_s[0, 0] == pytest.approx(angle / dt, rel=3e-12, abs=3e-13)


def test_invalid_mask_nan_inf_and_zero_are_excluded_without_zero_filling():
    prior = np.ones((3, 4, 3), dtype=float)
    later = np.zeros_like(prior)
    later[..., 1] = 1.0
    prior[0, 1] = 0.0
    later[0, 2] = np.nan
    prior[1, 0] = np.inf
    mask = np.ones((3, 4), dtype=bool)
    mask[1, 3] = False
    result = angular_activity(_frame(prior, mask=mask), _frame(later, time=0.5, sequence=1, mask=mask))
    assert result.validity == "invalid"
    assert result.coverage == pytest.approx(8 / 11)
    assert np.isnan(result.rate_rad_s[0, 1])
    assert np.isnan(result.rate_rad_s[0, 2])
    assert np.isnan(result.rate_rad_s[1, 0])
    assert np.isnan(result.rate_rad_s[1, 3])
    assert np.all(result.rate_rad_s[result.valid] > 0.0)


def test_subnormal_dt_overflow_invalidates_rates_but_keeps_pair_diagnostic():
    prior = np.empty((3, 3, 3), dtype=float)
    prior[...] = (1.0, 0.0, 0.0)
    later = np.empty_like(prior)
    later[...] = (-1.0, 0.0, 0.0)
    dt = np.nextafter(0.0, 1.0)
    result = angular_activity(_frame(prior), _frame(later, time=dt, sequence=1))
    assert result.dt_s == dt
    assert result.validity == "invalid"
    assert not result.valid.any()
    assert np.isnan(result.rate_rad_s).all()
    assert any("non-finite angular rates" in warning for warning in result.warnings)


def test_observer_does_not_modify_vectors_or_masks():
    rng = np.random.default_rng(4)
    prior_values = rng.normal(size=(3, 5, 3))
    later_values = rng.normal(size=(3, 5, 3))
    prior_mask = np.ones((3, 5), dtype=bool)
    later_mask = np.ones((3, 5), dtype=bool)
    prior = _frame(prior_values, mask=prior_mask)
    later = _frame(later_values, time=0.1, sequence=1, mask=later_mask)
    before = (prior.vectors.copy(), prior.mask.copy(), later.vectors.copy(), later.mask.copy())
    angular_activity(prior, later)
    assert np.array_equal(prior.vectors, before[0])
    assert np.array_equal(prior.mask, before[1])
    assert np.array_equal(later.vectors, before[2])
    assert np.array_equal(later.mask, before[3])


def test_tiling_at_arbitrary_boundaries_matches_whole_field():
    rng = np.random.default_rng(99)
    shape = (13, 17)
    prior_values = rng.normal(size=(*shape, 3)) * np.power(10.0, rng.uniform(-20, 20, size=(*shape, 1)))
    later_values = rng.normal(size=(*shape, 3)) * np.power(10.0, rng.uniform(-20, 20, size=(*shape, 1)))
    mask = rng.random(shape) > 0.15
    whole = angular_activity(_frame(prior_values, mask=mask), _frame(later_values, time=0.13, sequence=1, mask=mask))
    stitched = np.full(shape, np.nan)
    stitched_valid = np.zeros(shape, dtype=bool)
    for row0, row1 in ((0, 3), (3, 8), (8, 13)):
        for col0, col1 in ((0, 4), (4, 10), (10, 17)):
            pair = angular_activity(
                _frame(prior_values[row0:row1, col0:col1], mask=mask[row0:row1, col0:col1]),
                _frame(later_values[row0:row1, col0:col1], time=0.13, sequence=1, mask=mask[row0:row1, col0:col1]),
            )
            stitched[row0:row1, col0:col1] = pair.rate_rad_s
            stitched_valid[row0:row1, col0:col1] = pair.valid
    np.testing.assert_array_equal(stitched_valid, whole.valid)
    np.testing.assert_allclose(stitched, whole.rate_rad_s, rtol=0.0, atol=0.0, equal_nan=True)
