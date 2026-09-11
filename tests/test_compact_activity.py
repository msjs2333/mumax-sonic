"""Activity compact contribution contracts, checked against per-site physics."""

from dataclasses import replace

import numpy as np
import pytest

from mumax_sonic.attention import Attention
from mumax_sonic.field_pipeline import apply_aggregation, observe_field
from mumax_sonic.fields import FieldFrame
from mumax_sonic.observers.focused_activity import FocusedActivity


def _pair(shape=(5, 7), *, dt=2.5e-9, origin=(2.3e-6, -1.7e-6, 4e-9),
          dx=3.0e-9, dy=5.0e-9, mask=None, sequence=1):
    ny, nx = shape
    yy, xx = np.indices(shape)
    # Distinct, deliberately non-uniform angular rates at every site.
    angles = 0.03 + 0.011 * yy + 0.017 * xx + 0.002 * yy * xx
    prior = np.zeros((ny, nx, 3), dtype=float)
    prior[..., 0] = 1.0
    current = np.empty_like(prior)
    current[..., 0] = np.cos(angles)
    current[..., 1] = np.sin(angles)
    current[..., 2] = 0.0
    material = np.ones(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    before = FieldFrame(prior, dx, dy, 1.0, origin, material, sequence=sequence - 1)
    after = FieldFrame(current, dx, dy, 1.0 + dt, origin, material, sequence=sequence)
    return before, after, angles, material


def _assert_same_physics(default, compact):
    assert compact.contributions is None
    assert compact.sample == default.sample
    assert compact.summary == default.summary
    assert compact.diagnostic.keys() == default.diagnostic.keys()
    for key in compact.diagnostic:
        left, right = compact.diagnostic[key], default.diagnostic[key]
        if isinstance(left, float) and isinstance(right, float) and np.isnan(left) and np.isnan(right):
            continue
        assert left == right


def test_compact_matches_default_and_independent_array_split_reference():
    previous, current, angles, material = _pair()
    default = observe_field(current, "activity", previous=previous)
    compact = observe_field(current, "activity", previous=previous,
                            activity_contributions=False)
    _assert_same_physics(default, compact)
    assert default.contributions is not None
    assert default.contributions.positive.shape == current.vectors.shape[:2]

    ny, nx = angles.shape
    x = current.origin_m[0] + np.arange(nx) * current.dx_m
    y = current.origin_m[1] + np.arange(ny) * current.dy_m
    rows = np.array_split(np.arange(ny), 4)
    cols = np.array_split(np.arange(nx), 4)
    denominator = int(material.sum())
    expected = []
    for j, row in enumerate(rows):
        for i, col in enumerate(cols):
            weights = np.where(material[np.ix_(row, col)], angles[np.ix_(row, col)] / (current.sim_time_s - previous.sim_time_s), 0.0)
            total = float(weights.sum())
            if total <= 0:
                continue
            xx, yy = np.meshgrid(x[col], y[row])
            expected.append((f"activity:{j}-{i}", total / denominator,
                             float(np.sum(xx * weights) / total),
                             float(np.sum(yy * weights) / total)))
    actual = [(o.source_id, o.strength, o.position_m[0], o.position_m[1])
              for o in default.sample.observations]
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got[0] == want[0]
        assert got[1:] == pytest.approx(want[1:])


def test_mask_hole_is_excluded_from_strength_centroid_and_grid():
    mask = np.ones((5, 7), dtype=bool)
    mask[1, 2] = False
    mask[3, 5] = False
    previous, current, angles, material = _pair(mask=mask)
    view = observe_field(current, "activity", previous=previous)
    assert view.sample.validity == "valid"
    denominator = int(material.sum())
    assert sum(o.strength for o in view.sample.observations) == pytest.approx(
        float(np.where(material, angles, 0).sum()) / (current.sim_time_s - previous.sim_time_s) / denominator)
    assert view.contributions is not None
    assert view.contributions.positive[~material].tolist() == [0.0, 0.0]
    assert np.all(view.contributions.positive[material] > 0)


@pytest.mark.parametrize("kwargs", [{}, {"sequence": 3}, {"dt": 1e-9}])
def test_warmup_or_invalid_compact_preserves_default_semantics(kwargs):
    previous, current, _, _ = _pair(**kwargs)
    if kwargs.get("sequence") == 3:
        # The pair helper makes the sequence adjacent; create a real gap.
        current = replace(current, sequence=previous.sequence + 2)
    warm = observe_field(current, "activity", previous=None)
    full = observe_field(current, "activity", previous=previous)
    compact_warm = observe_field(current, "activity", previous=None,
                                 activity_contributions=False)
    compact_full = observe_field(current, "activity", previous=previous,
                                 activity_contributions=False)
    _assert_same_physics(warm, compact_warm)
    _assert_same_physics(full, compact_full)
    assert warm.sample.validity == "warming_up"
    if kwargs.get("sequence") == 3:
        assert full.sample.validity == "warming_up"


def test_invalid_nan_material_site_has_same_compact_semantics():
    previous, current, _, _ = _pair(shape=(5, 7))
    vectors = np.array(current.vectors, copy=True)
    vectors[2, 3, 0] = np.nan
    current = replace(current, vectors=vectors)
    full = observe_field(current, "activity", previous=previous)
    compact = observe_field(current, "activity", previous=previous,
                            activity_contributions=False)
    _assert_same_physics(full, compact)
    assert full.sample.validity == "invalid"


def test_small_3_by_n_focused_fallback_keeps_full_grid():
    previous, current, _, _ = _pair(shape=(3, 9))
    view = FocusedActivity(0.01)
    try:
        result = view.observe(previous, current, Attention(extent_m=30e-9,
                                                           origin_m=current.origin_m[:2]))
        assert result.contributions is not None
        assert result.contributions.positive.shape == (3, 9)
    finally:
        view.close()


def test_focused_crop_returns_compact_grid_usable_by_adaptive_aggregation():
    previous, current, _, _ = _pair(shape=(32, 32), origin=(4e-6, -2e-6, 0.0),
                                    dx=1e-9, dy=2e-9)
    focused = FocusedActivity(0.01)
    try:
        attention = Attention(extent_m=40e-9, origin_m=current.origin_m[:2], radius=.35)
        result = focused.observe(previous, current, attention)
        assert result.contributions is not None
        assert result.contributions.positive.shape[0] == 1
        assert result.contributions.positive.shape[1] == len(result.sample.observations)
        adaptive = apply_aggregation(result, attention, budget=4, mode="adaptive")
        assert adaptive.sample.validity == "valid"
        assert adaptive.diagnostic["adaptive_aggregation"]["status"] == "valid"
    finally:
        focused.close()


def test_direct_compact_view_adaptive_falls_back_without_changing_observations():
    previous, current, _, _ = _pair(shape=(5, 7))
    compact = observe_field(current, "activity", previous=previous,
                            activity_contributions=False)
    result = apply_aggregation(compact, Attention(extent_m=40e-9), budget=4,
                                mode="adaptive")
    assert result.sample == compact.sample
    assert result.diagnostic["adaptive_aggregation"]["status"] == "unsupported"
    assert result.diagnostic["adaptive_aggregation"]["fallback"] == "fixed_4x4"
