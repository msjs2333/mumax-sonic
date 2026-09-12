"""Contract tests for ordered AFM sublattice pairing.

The expected values below are calculated from the public definitions rather
than from the implementation's intermediate arrays.
"""

import numpy as np
import pytest

from mumax_sonic.fields import FieldFrame
from mumax_sonic.sublattices import combine_sublattices
from mumax_sonic.observers.activity import angular_activity
from mumax_sonic.observers.topology import topology


def _frame(values, *, entity_id, mask=None, **changes):
    values = np.asarray(values, dtype=float)
    if mask is None:
        mask = np.ones(values.shape[:2], dtype=bool)
    options = dict(dx_m=2e-9, dy_m=3e-9, mask=mask, entity_id=entity_id,
                   segment_id="pair-segment", sequence=7, sim_time_s=2e-9,
                   source_kind="replay")
    options.update(changes)
    return FieldFrame(values, **options)


def test_equal_weight_afm_keeps_neel_activity_when_net_moment_cancels():
    a = np.zeros((3, 4, 3)); a[..., 0] = 7.0
    b = np.zeros_like(a); b[..., 0] = -2.0
    result = combine_sublattices(a=_frame(a, entity_id="a"), b=_frame(b, entity_id="b"),
                                 pair_id="a|b", msat_a_A_m=3.0, msat_b_A_m=3.0)

    assert result.pair_id == "a|b"
    np.testing.assert_allclose(result.neel.vectors[0, 0], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(result.net_A_m, 0.0)
    assert np.isnan(result.net_direction.vectors).all()
    assert np.all(result.valid_mask)


def test_unequal_weights_produce_known_net_direction_and_magnitude():
    a = np.zeros((3, 3, 3)); a[..., 2] = 1.0
    b = np.zeros_like(a); b[..., 2] = -1.0
    result = combine_sublattices(_frame(a, entity_id="left"), _frame(b, entity_id="right"),
                                 pair_id="left|right", msat_a_A_m=3.0,
                                 msat_b_A_m=1.0)

    np.testing.assert_allclose(result.neel.vectors[0, 0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(result.net_A_m[0, 0], [0.0, 0.0, 2.0])
    # Direction retains the weighted magnitude: M/(Ms_a+Ms_b) = 2/4.
    np.testing.assert_allclose(result.net_direction.vectors[0, 0], [0.0, 0.0, 0.5])


def test_exchange_flips_neel_sign_but_preserves_activity_and_net_magnitude():
    a = np.zeros((3, 3, 3)); a[..., 1] = 1.0
    b = np.zeros_like(a); b[..., 1] = -1.0
    first = combine_sublattices(_frame(a, entity_id="a"), _frame(b, entity_id="b"),
                                pair_id="a-b", msat_a_A_m=2.0, msat_b_A_m=1.0)
    exchanged = combine_sublattices(_frame(b, entity_id="b"), _frame(a, entity_id="a"),
                                    pair_id="b-a", msat_a_A_m=1.0, msat_b_A_m=2.0)

    np.testing.assert_allclose(exchanged.neel.vectors, -first.neel.vectors)
    np.testing.assert_allclose(exchanged.neel_raw, -first.neel_raw)
    np.testing.assert_allclose(exchanged.net_A_m, first.net_A_m)
    np.testing.assert_array_equal(exchanged.valid_mask, first.valid_mask)


def test_two_frame_afm_rotation_has_zero_net_magnetization_but_nonzero_neel_activity():
    a0 = np.zeros((3, 3, 3)); a0[..., 0] = 1.0
    b0 = -a0
    a1 = np.zeros_like(a0); a1[..., 1] = 1.0
    b1 = -a1
    n0 = combine_sublattices(_frame(a0, entity_id="a", sequence=0, sim_time_s=0.0),
                             _frame(b0, entity_id="b", sequence=0, sim_time_s=0.0),
                             pair_id="pair", msat_a_A_m=2.0, msat_b_A_m=2.0).neel
    n1 = combine_sublattices(_frame(a1, entity_id="a", sequence=1, sim_time_s=1e-9),
                             _frame(b1, entity_id="b", sequence=1, sim_time_s=1e-9),
                             pair_id="pair", msat_a_A_m=2.0, msat_b_A_m=2.0)
    rate = angular_activity(n0, n1.neel)
    assert rate.validity == "valid"
    assert rate.mean_rad_s == pytest.approx(np.pi / (2e-9))
    np.testing.assert_allclose(n1.net_A_m, 0.0)


def test_exchange_of_both_frames_preserves_neel_activity_and_flips_topology_sign():
    field = np.zeros((33, 33, 3)); yy, xx = np.indices((33, 33));
    radius = np.hypot(xx - 16, yy - 16) / 7.0
    polar = 2.0 * np.arctan2(1.0, radius)
    azimuth = np.arctan2(yy - 16, xx - 16)
    field[...] = np.stack((np.sin(polar) * np.cos(azimuth),
                           np.sin(polar) * np.sin(azimuth), np.cos(polar)), axis=-1)
    n0 = combine_sublattices(_frame(field, entity_id="a", sequence=0, sim_time_s=0.0),
                             _frame(-field, entity_id="b", sequence=0, sim_time_s=0.0),
                             pair_id="pair", msat_a_A_m=1.0, msat_b_A_m=1.0)
    n1 = combine_sublattices(_frame(np.roll(field, 1, axis=1), entity_id="a", sequence=1, sim_time_s=1e-9),
                             _frame(-np.roll(field, 1, axis=1), entity_id="b", sequence=1, sim_time_s=1e-9),
                             pair_id="pair", msat_a_A_m=1.0, msat_b_A_m=1.0)
    swapped0 = combine_sublattices(_frame(-field, entity_id="b", sequence=0, sim_time_s=0.0),
                                   _frame(field, entity_id="a", sequence=0, sim_time_s=0.0),
                                   pair_id="pair", msat_a_A_m=1.0, msat_b_A_m=1.0)
    swapped1 = combine_sublattices(_frame(-np.roll(field, 1, axis=1), entity_id="b", sequence=1, sim_time_s=1e-9),
                                   _frame(np.roll(field, 1, axis=1), entity_id="a", sequence=1, sim_time_s=1e-9),
                                   pair_id="pair", msat_a_A_m=1.0, msat_b_A_m=1.0)
    assert angular_activity(n0.neel, n1.neel).mean_rad_s == pytest.approx(
        angular_activity(swapped0.neel, swapped1.neel).mean_rad_s)
    q = topology(n0.neel.vectors, 1.0, 1.0)
    q_swapped = topology(swapped0.neel.vectors, 1.0, 1.0)
    assert abs(q.q_net) > 0.5
    assert q_swapped.q_net == pytest.approx(-q.q_net, rel=1e-12, abs=1e-12)
    assert q_swapped.q_abs == pytest.approx(q.q_abs, rel=1e-12, abs=1e-12)


def test_derived_quantity_identity_prevents_neel_net_cross_pairing():
    values = np.zeros((3, 3, 3)); values[..., 0] = 1.0
    pair = combine_sublattices(_frame(values, entity_id="a"), _frame(-values, entity_id="b"),
                               pair_id="pair", msat_a_A_m=2.0, msat_b_A_m=1.0)
    later = combine_sublattices(_frame(values, entity_id="a", sequence=8, sim_time_s=3e-9),
                                _frame(-values, entity_id="b", sequence=8, sim_time_s=3e-9),
                                pair_id="pair", msat_a_A_m=2.0, msat_b_A_m=1.0)
    assert np.isfinite(later.net_direction.vectors).all()
    result = angular_activity(pair.neel, later.net_direction)
    assert result.validity == "warming_up"


def test_missing_partner_is_rejected_and_parallel_neel_is_undefined():
    values = np.ones((3, 3, 3))
    a = _frame(values, entity_id='a')
    with pytest.raises(ValueError):
        combine_sublattices(a, None, pair_id='pair', msat_a_A_m=1, msat_b_A_m=1)
    pair = combine_sublattices(a, _frame(values, entity_id='b'), pair_id='pair',
                               msat_a_A_m=1, msat_b_A_m=1)
    assert pair.valid_mask.all()
    assert np.isnan(pair.neel.vectors).all()
    np.testing.assert_array_equal(pair.neel_raw, 0)
    assert not pair.neel_raw.flags.writeable


def test_same_pair_id_with_ordered_entity_exchange_cannot_cross_time_pair():
    values = np.zeros((3, 3, 3)); values[..., 0] = 1.0
    first = combine_sublattices(_frame(values, entity_id="a", sequence=0, sim_time_s=0.0),
                                _frame(-values, entity_id="b", sequence=0, sim_time_s=0.0),
                                pair_id="pair", msat_a_A_m=1.0, msat_b_A_m=1.0)
    second = combine_sublattices(_frame(-values, entity_id="b", sequence=1, sim_time_s=1e-9),
                                 _frame(values, entity_id="a", sequence=1, sim_time_s=1e-9),
                                 pair_id="pair", msat_a_A_m=1.0, msat_b_A_m=1.0)
    assert angular_activity(first.neel, second.neel).validity == "warming_up"


def test_raw_magnitudes_are_preserved_and_invalid_cells_are_nan_not_zero():
    a = np.zeros((3, 3, 3)); a[..., 0] = 4.0; a[0, 0] = 0.0
    b = np.zeros_like(a); b[..., 0] = -2.0; b[0, 1] = np.nan
    mask = np.ones((3, 3), dtype=bool); mask[0, 2] = False
    result = combine_sublattices(_frame(a, entity_id="a", mask=mask),
                                 _frame(b, entity_id="b", mask=mask),
                                 pair_id="a-b", msat_a_A_m=1.0, msat_b_A_m=1.0)

    np.testing.assert_allclose(result.a.vectors[1, 1], [4.0, 0.0, 0.0])
    np.testing.assert_allclose(result.b.vectors[1, 1], [-2.0, 0.0, 0.0])
    assert not result.valid_mask[0, 0]
    assert not result.valid_mask[0, 1]
    assert not result.valid_mask[0, 2]
    assert np.isnan(result.neel.vectors[0, 0]).all()
    assert np.isnan(result.net_A_m[0, 1]).all()


def test_mask_mismatch_is_rejected_and_material_nan_reduces_valid_coverage():
    values = np.zeros((3, 3, 3)); values[..., 0] = 1.0
    mask_a = np.ones((3, 3), dtype=bool)
    mask_b = mask_a.copy(); mask_b[0, 0] = False
    with pytest.raises(ValueError):
        combine_sublattices(_frame(values, entity_id="a", mask=mask_a),
                            _frame(-values, entity_id="b", mask=mask_b),
                            pair_id="pair", msat_a_A_m=1.0, msat_b_A_m=1.0)
    invalid = values.copy(); invalid[0, 0] = np.nan
    result = combine_sublattices(_frame(invalid, entity_id="a"),
                                 _frame(-values, entity_id="b"),
                                 pair_id="pair", msat_a_A_m=1.0, msat_b_A_m=1.0)
    assert result.valid_mask.sum() == 8
    assert np.isnan(result.neel.vectors[0, 0]).all()


@pytest.mark.parametrize("change", [
    {"entity_id": "a"},
    {"dx_m": 9e-9},
    {"dy_m": 9e-9},
    {"sim_time_s": 3e-9},
    {"sequence": 8},
    {"segment_id": "other"},
    {"source_kind": "synthetic"},
])
def test_pair_requires_distinct_entities_and_matching_frame_identity(change):
    a = _frame(np.dstack([np.ones((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))]), entity_id="a")
    b_kwargs = dict(entity_id="b")
    b_kwargs.update(change)
    b = _frame(np.dstack([-np.ones((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))]), **b_kwargs)
    with pytest.raises(ValueError):
        combine_sublattices(a, b, pair_id="a-b", msat_a_A_m=1.0, msat_b_A_m=1.0)


@pytest.mark.parametrize("bad_pair", [None, "", ("a", "a")])
def test_missing_or_non_distinct_pair_id_is_rejected(bad_pair):
    values = np.ones((3, 3, 3))
    with pytest.raises((TypeError, ValueError)):
        combine_sublattices(_frame(values, entity_id="a"), _frame(-values, entity_id="b"),
                            pair_id=bad_pair, msat_a_A_m=1.0, msat_b_A_m=1.0)


@pytest.mark.parametrize("kwargs", [
    {"msat_a_A_m": 0.0}, {"msat_b_A_m": np.inf},
    {"msat_a_A_m": np.array([1.0, 2.0])},
])
def test_material_weights_must_be_positive_finite_and_shape_compatible(kwargs):
    values = np.ones((3, 3, 3))
    options = {"msat_a_A_m": 1.0, "msat_b_A_m": 1.0}
    options.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        combine_sublattices(_frame(values, entity_id="a"), _frame(-values, entity_id="b"),
                            pair_id="a-b", **options)


def test_per_material_array_weights_are_applied_cellwise():
    values = np.zeros((3, 3, 3)); values[..., 2] = 1.0
    weights_a = np.full((3, 3), 3.0); weights_a[0, 0] = 2.0
    weights_b = np.ones((3, 3))
    result = combine_sublattices(_frame(values, entity_id="a"), _frame(-values, entity_id="b"),
                                 pair_id="a-b", msat_a_A_m=weights_a, msat_b_A_m=weights_b)
    np.testing.assert_allclose(result.net_A_m[0, 0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(result.net_A_m[1, 1], [0.0, 0.0, 2.0])


def test_input_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        combine_sublattices(_frame(np.ones((3, 3, 3)), entity_id="a"),
                            _frame(-np.ones((4, 3, 3)), entity_id="b"),
                            pair_id="a-b", msat_a_A_m=1.0, msat_b_A_m=1.0)
