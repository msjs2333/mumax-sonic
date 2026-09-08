import numpy as np
import pytest

from mumax_sonic.sources.analytic import SCENARIOS, make_field


def test_scenarios_are_deterministic_finite_and_unit_normalized():
    for scenario in SCENARIOS:
        first = make_field(scenario, sim_time_s=0.25e-9, size=65)
        second = make_field(scenario, sim_time_s=0.25e-9, size=65)
        assert first.shape == (65, 65, 3)
        assert np.array_equal(first, second)
        assert np.all(np.isfinite(first))
        assert np.allclose(np.linalg.norm(first, axis=-1), 1.0, rtol=0, atol=2e-15)


def test_uniform_and_texture_core_directions():
    uniform = make_field("uniform", size=9)
    assert np.allclose(uniform, (0.0, 0.0, 1.0))
    skyrmion = make_field("skyrmion", size=65)
    assert skyrmion[32, 32, 2] < -0.999999
    pair = make_field("opposite_pair", size=65)
    assert pair[32, 16, 2] < -0.99
    assert pair[32, 48, 2] < -0.99


@pytest.mark.parametrize("scenario, d_index, transverse_index", [
    ("wall_inplane", 0, 1),
    ("wall_pma", 2, 0),
])
def test_wall_domains_and_continuous_phase(scenario, d_index, transverse_index):
    at_zero = make_field(scenario, sim_time_s=0.0, size=65)
    at_period = make_field(scenario, sim_time_s=8e-9, size=65)
    just_before = make_field(scenario, sim_time_s=8e-9 - 1e-13, size=65)
    just_after = make_field(scenario, sim_time_s=8e-9 + 1e-13, size=65)
    assert np.allclose(at_zero, at_period)
    assert np.max(np.abs(just_after - just_before)) < 2e-3
    left = at_zero[32, 0]
    right = at_zero[32, -1]
    assert left[d_index] < -0.99
    assert right[d_index] > 0.99
    assert abs(at_zero[32, 32, transverse_index]) > 0.99


def test_invalid_fixture_arguments_are_rejected():
    with pytest.raises(ValueError):
        make_field("missing")
    with pytest.raises(ValueError):
        make_field("uniform", sim_time_s=np.inf)
    with pytest.raises(ValueError):
        make_field("uniform", size=1)
