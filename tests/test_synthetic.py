import math

import pytest

from mumax_sonic.sources.synthetic import SCENARIOS, make_sample


def test_scenes_are_deterministic_and_well_formed():
    for scenario in SCENARIOS:
        first = make_sample(scenario, 0.25e-9, sequence=3, seed=19)
        second = make_sample(scenario, 0.25e-9, sequence=3, seed=19)
        assert first == second
        assert 1 <= len(first.observations) <= 6
        assert len({o.source_id for o in first.observations}) == len(first.observations)
        assert all(-1e-6 <= p <= 1e-6 for o in first.observations for p in o.position_m)
        assert all(0 <= o.strength <= 1 and o.sign in (-1, 1) for o in first.observations)


def test_moving_scene_uses_simulation_time_for_both_axes():
    a = make_sample("moving", 0.0)
    b = make_sample("moving", 0.25e-9)
    primary_a = a.observations[0]
    primary_b = b.observations[0]
    assert primary_a.position_m != primary_b.position_m
    assert a.time_kind == b.time_kind == "dynamics"
    assert make_sample("moving", 0.0).observations == make_sample("moving", 8e-9).observations


def test_signed_channels_are_preserved_even_when_colocated():
    for scenario in ("signed_pair", "colocated"):
        sample = make_sample(scenario, 0.0)
        assert {o.sign for o in sample.observations} == {-1, 1}
        positive = [o for o in sample.observations if o.sign == 1]
        negative = [o for o in sample.observations if o.sign == -1]
        assert positive and negative
        if scenario == "colocated":
            assert positive[0].position_m == negative[0].position_m
            assert positive[0].strength == negative[0].strength


def test_static_and_quality_labels_are_explicit():
    for scenario in ("signed_pair", "colocated", "roi_challenge"):
        assert make_sample(scenario, 17e-9).time_kind == "static"
    assert make_sample("orientation", 17e-9).time_kind == "dynamics"
    assert make_sample("stale", 0).validity == "stale"
    assert make_sample("invalid", 0).validity == "invalid"


def test_orientation_is_continuous_and_finite():
    sample = make_sample("orientation", 0.3e-9)
    angles = [o.orientation_rad for o in sample.observations]
    assert all(math.isfinite(a) for a in angles)
    assert len(set(angles)) == len(angles)
    before = make_sample("orientation", 8e-9 - 1e-15).observations[0].orientation_rad
    after = make_sample("orientation", 8e-9 + 1e-15).observations[0].orientation_rad
    assert abs(after - before) < 1e-2
    assert make_sample("orientation", 0).time_kind == "dynamics"


def test_bad_scenario_and_time_are_rejected():
    with pytest.raises(ValueError):
        make_sample("missing", 0)
    with pytest.raises(ValueError):
        make_sample("moving", -1)
    with pytest.raises(ValueError):
        make_sample("moving", math.inf)
