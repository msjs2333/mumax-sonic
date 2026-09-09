import math

import numpy as np
import pytest

from mumax_sonic.sources.band_demo import STEP_S, SCENARIOS, make_band_frame


def test_frames_have_expected_geometry_timeline_and_unit_vectors():
    frame = make_band_frame("band_in", 3)
    assert frame.vectors.shape == (17, 17, 3)
    assert frame.sim_time_s == pytest.approx(3 * STEP_S)
    assert frame.origin_m == (-1e-6, -1e-6, 0)
    assert frame.dx_m == frame.dy_m == pytest.approx(2e-6 / 16)
    assert frame.source_kind == "synthetic"
    assert frame.time_kind == "dynamics"
    assert np.allclose(np.linalg.norm(frame.vectors, axis=-1), 1.0)
    assert np.all(frame.vectors[..., 1] == 0)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenarios_are_described_and_provenance_is_auditable(scenario):
    frame = make_band_frame(scenario, 1)
    assert SCENARIOS[scenario]
    assert frame.segment_id == scenario
    assert f"analytic:{scenario}" in frame.provenance
    assert "frequencies_hz:" in frame.provenance
    assert "phase_rad:" in frame.provenance
    assert "amplitude:0.1" in frame.provenance


def test_opposite_is_antisymmetric_and_mixed_has_two_halves():
    frame = make_band_frame("band_opposite", 5)
    assert np.allclose(frame.vectors[:, :8, 0], -frame.vectors[:, 8:16, 0])
    mixed = make_band_frame("band_mixed", 7)
    assert np.allclose(mixed.vectors[:, :8, 0], mixed.vectors[:, 0:1, 0])
    assert np.allclose(mixed.vectors[:, 8:, 0], mixed.vectors[:, 8:9, 0])


def test_sampling_inputs_are_validated():
    for args in (("missing", 0), ("band_in", -1), ("band_in", True)):
        with pytest.raises(ValueError):
            make_band_frame(*args)
    for dt in (0, -1, math.nan, math.inf, "x", True):
        with pytest.raises(ValueError):
            make_band_frame("band_in", 0, dt_s=dt)
    for size in (2, 2.5, True):
        with pytest.raises(ValueError):
            make_band_frame("band_in", 0, size=size)


def test_cache_returns_same_immutable_frame_for_same_inputs():
    first = make_band_frame("band_out", 2)
    second = make_band_frame("band_out", 2)
    assert first is second
    assert not first.vectors.flags.writeable
