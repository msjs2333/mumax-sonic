import math

import numpy as np
import pytest

from mumax_sonic.fields import FieldFrame
from mumax_sonic.observers.band import BandConfig, band_power


N = 32
DT = 1e-10
CONFIG = BandConfig(2e9, 4e9, N, (0, 0, 1))


def _history(*, frequency=3.125e9, amplitude=0.2, changes=None):
    changes = {} if changes is None else changes
    result = []
    for index in range(N):
        phase = 2 * math.pi * frequency * index * DT
        values = np.zeros((4, 5, 3), dtype=float)
        values[..., 0] = amplitude * math.sin(phase)
        values[..., 2] = math.sqrt(1 - values[0, 0, 0] ** 2)
        kwargs = dict(changes.get(index, {}))
        result.append(FieldFrame(values, 1e-9, 1e-9, sim_time_s=index * DT, sequence=index, **kwargs))
    return result


def test_band_is_local_and_has_prescribed_windowed_sine_normalization():
    history = _history()
    result = band_power(history, CONFIG)
    # Exact-bin sinusoid, periodic Hann and the stated one-sided normalization.
    expected = 0.5 * 0.2 ** 2
    assert result.validity == "valid"
    assert result.coverage == 1
    assert result.mean_power == pytest.approx(expected, rel=2e-12)
    assert result.max_power == pytest.approx(expected, rel=2e-12)
    assert result.frequency_resolution_hz == pytest.approx(1 / (N * DT))
    assert result.window_span_s == pytest.approx((N - 1) * DT)


def test_out_of_band_and_spatial_opposite_phase_do_not_cancel_each_other():
    outside = band_power(_history(frequency=4.6875e9), CONFIG)
    assert outside.validity == "valid"
    assert outside.max_power == pytest.approx(0.0, abs=1e-25)

    history = _history()
    for index, frame in enumerate(history):
        values = frame.vectors.copy()
        values[:, :2, 0] *= -1
        history[index] = FieldFrame(values, frame.dx_m, frame.dy_m, sim_time_s=frame.sim_time_s, sequence=frame.sequence)
    phased = band_power(history, CONFIG)
    assert phased.mean_power == pytest.approx(band_power(_history(), CONFIG).mean_power)


def test_coordinate_rotation_with_reference_axis_preserves_power():
    original = band_power(_history(), CONFIG)
    rotated = []
    for frame in _history():
        values = frame.vectors[..., (2, 1, 0)].copy()
        rotated.append(FieldFrame(values, frame.dx_m, frame.dy_m, sim_time_s=frame.sim_time_s, sequence=frame.sequence))
    result = band_power(rotated, BandConfig(2e9, 4e9, N, (1, 0, 0)))
    assert result.power == pytest.approx(original.power)


def test_short_history_gap_and_mask_change_restart_the_contiguous_suffix():
    assert band_power(_history()[:-1], CONFIG).validity == "warming_up"
    gapped = _history()
    for index in range(10, N):
        frame = gapped[index]
        gapped[index] = FieldFrame(frame.vectors, 1e-9, 1e-9, sim_time_s=index * DT, sequence=index + 1)
    assert band_power(gapped, CONFIG).validity == "warming_up"
    mask = np.ones((4, 5), dtype=bool)
    mask[0, 0] = False
    changed = _history(changes={10: {"mask": mask}})
    assert band_power(changed, CONFIG).validity == "warming_up"


def test_nonuniform_times_nyquist_and_sequence_order_are_not_spectral_data():
    nonuniform = _history()
    frame = nonuniform[15]
    nonuniform[15] = FieldFrame(frame.vectors, 1e-9, 1e-9, sim_time_s=15 * DT + 2e-16, sequence=15)
    assert band_power(nonuniform, CONFIG).validity == "unsupported"
    too_high = BandConfig(2e9, 5e9, N)
    assert band_power(_history(), too_high).validity == "unsupported"
    duplicate = _history()
    frame = duplicate[5]
    duplicate[5] = FieldFrame(frame.vectors, 1e-9, 1e-9, sim_time_s=5 * DT, sequence=4)
    assert band_power(duplicate, CONFIG).validity == "invalid"


def test_invalid_vector_anywhere_invalidates_that_site_for_the_full_window():
    history = _history()
    values = history[8].vectors.copy()
    values[1, 2] = 0
    history[8] = FieldFrame(values, 1e-9, 1e-9, sim_time_s=8 * DT, sequence=8)
    result = band_power(history, CONFIG)
    assert result.validity == "invalid"
    assert result.coverage == pytest.approx(19 / 20)
    assert not result.valid[1, 2]
    assert np.isnan(result.power[1, 2])


@pytest.mark.parametrize("kwargs", [
    {"low_hz": 0}, {"low_hz": 4, "high_hz": 4}, {"window_samples": 15},
    {"window_samples": 16.0}, {"reference_axis": (0, 0, 0)},
])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        BandConfig(**kwargs)


def test_extreme_finite_reference_axis_normalizes_without_overflow():
    assert BandConfig(reference_axis=(1e300, 1e-300, 0)).reference_axis[0] == pytest.approx(1)
