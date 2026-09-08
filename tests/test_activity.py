import math

import numpy as np
import pytest

from mumax_sonic.fields import FieldFrame
from mumax_sonic.observers.activity import angular_activity


def _frame(vectors, *, time=0.0, sequence=0, **changes):
    dx_m = changes.pop("dx_m", 2e-9)
    dy_m = changes.pop("dy_m", 3e-9)
    return FieldFrame(
        vectors=np.asarray(vectors, dtype=float),
        dx_m=dx_m,
        dy_m=dy_m,
        sim_time_s=time,
        sequence=sequence,
        **changes,
    )


def _uniform(vector, shape=(4, 5)):
    values = np.empty((*shape, 3), dtype=float)
    values[...] = vector
    return values


def test_uniform_rotation_measures_known_physical_angular_speed_including_out_of_plane():
    angle = 0.3
    earlier = _frame(_uniform((2.0, 0.0, 0.0)), time=1.0, sequence=8)
    later = _frame(_uniform((3.0 * math.cos(angle), 0.0, 3.0 * math.sin(angle))), time=1.2, sequence=9)
    result = angular_activity(earlier, later)
    assert result.validity == "valid"
    assert result.coverage == 1.0
    assert result.mean_rad_s == pytest.approx(angle / 0.2)
    assert result.max_rad_s == pytest.approx(angle / 0.2)


def test_proper_spin_rotation_and_tiny_angle_are_measured_without_magnitude_dependence():
    tiny = 1e-9
    prior = _frame(_uniform((0.0, 0.0, 1e300)), time=0.0, sequence=0)
    current = _frame(_uniform((1e-300 * math.sin(tiny), 0.0, 1e-300 * math.cos(tiny))), time=0.5, sequence=1)
    result = angular_activity(prior, current)
    assert result.mean_rad_s == pytest.approx(tiny / 0.5, rel=1e-6)


def test_zero_activity_is_valid_and_distinct_from_warming_up():
    prior = _frame(_uniform((0.0, 0.0, 1.0)), time=2.0, sequence=3)
    current = _frame(_uniform((0.0, 0.0, 8.0)), time=2.25, sequence=4)
    measured = angular_activity(prior, current)
    waiting = angular_activity(None, current)
    assert measured.validity == "valid"
    assert np.all(measured.rate_rad_s == 0.0)
    assert waiting.validity == "warming_up"
    assert np.isnan(waiting.rate_rad_s).all()


def test_pi_step_warns_about_shortest_angle_aliasing_limit():
    prior = _frame(_uniform((1.0, 0.0, 0.0)), time=0.0, sequence=0)
    current = _frame(_uniform((-1.0, 0.0, 0.0)), time=0.1, sequence=1)
    result = angular_activity(prior, current)
    assert result.mean_rad_s == pytest.approx(math.pi / 0.1)
    assert any("alias" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"sequence": 2}, "warming_up"),
        ({"sequence": 0}, "invalid"),
        ({"sequence": 1, "time": 0.0}, "invalid"),
        ({"segment_id": "next"}, "warming_up"),
        ({"source_kind": "replay"}, "warming_up"),
        ({"entity_id": "n"}, "warming_up"),
        ({"dx_m": 4e-9}, "warming_up"),
    ],
)
def test_temporal_discontinuities_are_not_measured(changes, expected):
    prior = _frame(_uniform((1.0, 0.0, 0.0)), time=0.0, sequence=0)
    kwargs = {"time": 0.2, "sequence": 1}
    kwargs.update(changes)
    current = _frame(_uniform((0.0, 1.0, 0.0)), **kwargs)
    result = angular_activity(prior, current)
    assert result.validity == expected
    assert not result.valid.any()
    assert np.isnan(result.rate_rad_s).all()


def test_changed_mask_and_relaxation_are_not_dynamic_pairs():
    prior = _frame(_uniform((1.0, 0.0, 0.0)), time=0.0, sequence=0)
    changed_mask = np.ones((4, 5), dtype=bool)
    changed_mask[0, 0] = False
    changed = _frame(_uniform((0.0, 1.0, 0.0)), time=0.1, sequence=1, mask=changed_mask)
    relaxing = _frame(_uniform((0.0, 1.0, 0.0)), time=0.1, sequence=1, time_kind="relaxation")
    assert angular_activity(prior, changed).validity == "warming_up"
    assert angular_activity(prior, relaxing).validity == "unsupported"
    assert angular_activity(None, relaxing).validity == "unsupported"


def test_invalid_vector_pairs_preserve_usable_rates_and_report_incomplete_coverage():
    prior_values = _uniform((1.0, 0.0, 0.0))
    current_values = _uniform((0.0, 1.0, 0.0))
    prior_values[0, 0] = 0.0
    current_values[0, 1] = np.nan
    mask = np.ones((4, 5), dtype=bool)
    mask[3, 4] = False
    result = angular_activity(
        _frame(prior_values, time=0.0, sequence=0, mask=mask),
        _frame(current_values, time=0.5, sequence=1, mask=mask),
    )
    assert result.validity == "invalid"
    assert result.coverage == pytest.approx(17 / 19)
    assert result.valid[1, 1]
    assert result.rate_rad_s[1, 1] == pytest.approx(math.pi)
    assert np.isnan(result.rate_rad_s[0, 0])
    assert np.isnan(result.rate_rad_s[0, 1])


def test_mask_holes_are_excluded_from_coverage_denominator_and_empty_material_is_invalid():
    mask = np.ones((4, 5), dtype=bool)
    mask[:, :2] = False
    prior = _frame(_uniform((1.0, 0.0, 0.0)), time=0.0, sequence=0, mask=mask)
    current = _frame(_uniform((0.0, 1.0, 0.0)), time=1.0, sequence=1, mask=mask)
    result = angular_activity(prior, current)
    assert result.coverage == 1.0
    assert np.isnan(result.rate_rad_s[:, :2]).all()
    empty = _frame(_uniform((1.0, 0.0, 0.0)), mask=np.zeros((4, 5), dtype=bool))
    assert angular_activity(None, empty).validity == "invalid"


def test_max_dt_is_explicit_and_nonuniform_forward_times_use_actual_time():
    prior = _frame(_uniform((1.0, 0.0, 0.0)), time=4.0, sequence=10)
    current = _frame(_uniform((0.0, 1.0, 0.0)), time=4.4, sequence=11)
    accepted = angular_activity(prior, current)
    limited = angular_activity(prior, current, max_dt_s=0.3)
    assert accepted.dt_s == pytest.approx(0.4)
    assert accepted.mean_rad_s == pytest.approx(math.pi / 2 / 0.4)
    assert limited.validity == "warming_up"
    assert limited.dt_s == pytest.approx(0.4)
    for bad in (0.0, -1.0, float("nan"), float("inf"), "x"):
        with pytest.raises(ValueError):
            angular_activity(prior, current, max_dt_s=bad)


def test_max_dt_allows_only_relative_floating_point_boundary_noise():
    prior = _frame(_uniform((1.0, 0.0, 0.0)), time=1e-9, sequence=0)
    current = _frame(_uniform((0.0, 1.0, 0.0)), time=1.05e-9, sequence=1)
    at_boundary = angular_activity(prior, current, max_dt_s=5e-11)
    larger_interval = angular_activity(prior, current, max_dt_s=4e-11)
    assert at_boundary.validity == "valid"
    assert larger_interval.validity == "warming_up"


def test_subnormal_dt_excludes_overflowed_rates_without_reporting_validity():
    prior = _frame(_uniform((1.0, 0.0, 0.0)), time=0.0, sequence=0)
    current = _frame(_uniform((-1.0, 0.0, 0.0)), time=np.nextafter(0.0, 1.0), sequence=1)
    result = angular_activity(prior, current)
    assert result.validity == "invalid"
    assert not result.valid.any()
    assert np.isnan(result.rate_rad_s).all()
    assert any("non-finite angular rates" in warning for warning in result.warnings)


def test_activity_does_not_mutate_frame_inputs():
    values = _uniform((1.0, 0.0, 0.0))
    previous = _frame(values, time=0.0, sequence=0)
    current = _frame(_uniform((0.0, 1.0, 0.0)), time=1.0, sequence=1)
    before_vectors = previous.vectors.copy()
    before_mask = previous.mask.copy()
    angular_activity(previous, current)
    assert np.array_equal(previous.vectors, before_vectors)
    assert np.array_equal(previous.mask, before_mask)
