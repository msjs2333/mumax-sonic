import pytest

from mumax_sonic.attention import Attention, select_sources
from mumax_sonic.mapping import map_sample_with_report
from mumax_sonic.model import MAX_SOURCE_BUDGET, Observation, Sample


def observation(source_id, strength, sign=1, *, position=(0, 0, 0),
                entity="demo", quantity="q", unit="1"):
    return Observation(source_id, position, strength, sign, entity_id=entity,
                       quantity=quantity, unit=unit)


def sample(*observations, **kwargs):
    return Sample(0.0, observations, **kwargs)


def test_budget_requires_strict_bounded_python_int():
    attention = Attention()
    observations = (observation("one", 1),)
    for budget in (True, 1.0, 0, MAX_SOURCE_BUDGET + 1):
        with pytest.raises(ValueError):
            select_sources(observations, attention, budget)
        with pytest.raises(ValueError):
            map_sample_with_report(sample(*observations), attention, budget=budget)


def test_one_slot_prefers_strong_negative_roi_over_positive():
    attention = Attention()
    observations = (observation("positive", 2, 1), observation("negative", 3, -1))
    assert [o.source_id for o in select_sources(observations, attention, 1)] == ["negative"]


def test_report_conserves_raw_strength_for_each_channel():
    attention = Attention(background=0)
    report = map_sample_with_report(sample(
        observation("p", 2, 1), observation("n", 3, -1),
        observation("outside", 5, 1, position=(2e-6, 0, 0)),
    ), attention, mode="positive", budget=1).report
    group = report["groups"][0]
    for channel in ("absolute", "positive", "negative"):
        item = group[channel]
        assert item["total"] == pytest.approx(item["selected"] + item["omitted_budget"]
                                                + item["excluded_sign"] + item["excluded_attention"])
    assert group["negative"]["excluded_sign"] == 3
    assert group["absolute"]["excluded_attention"] == 5


def test_report_separates_incompatible_units_and_mutes_output_only():
    attention = Attention()
    result = map_sample_with_report(sample(
        observation("a", 2, unit="1"), observation("b", 4, unit="rad/s"),
    ), attention, master_gain=0, audible=False)
    assert len(result.report["groups"]) == 2
    assert result.report["selected_ids"] == ["b", "a"]
    assert result.report["output_ids"] == []
    assert all(group["absolute"]["output"] == 0 for group in result.report["groups"])


def test_zero_strength_and_zero_attention_do_not_allocate_slots():
    attention = Attention(background=0)
    result = map_sample_with_report(sample(
        observation("zero", 0), observation("outside", 4, position=(2e-6, 0, 0)),
        observation("inside", 1),
    ), attention, budget=2)
    assert result.report["selected_ids"] == ["inside"]
    group = result.report["groups"][0]["absolute"]
    assert group["excluded_attention"] == 4


def test_increasing_budget_represents_uniform_sources():
    attention = Attention()
    observations = tuple(observation(f"s{i}", 1, 1) for i in range(4))
    low = map_sample_with_report(sample(*observations), attention, budget=2).report
    high = map_sample_with_report(sample(*observations), attention, budget=4).report
    assert low["groups"][0]["absolute"]["selected"] == 2
    assert high["groups"][0]["absolute"]["selected"] == 4


def test_warming_reason_survives_incomplete_coverage_and_settings_are_reported():
    attention = Attention(center=(0.2, -0.1), radius=0.7, background=0.3,
                          extent_m=2e-6, origin_m=(1e-6, -1e-6))
    result = map_sample_with_report(
        sample(observation("one", 1), validity="warming_up", coverage=0), attention,
        mode="negative", master_gain=0.2, budget=3, strength_reference=2,
    )
    assert result.scene.validity == "warming_up"
    assert result.report["validity"] == "warming_up"
    assert result.report["groups"][0]["absolute"]["total"] is None
    assert result.report["settings"] == {
        "mode": "negative", "master_gain": 0.2, "strength_reference": 2,
        "gain_divisor": 3,
        "attention": {
            "center": (0.2, -0.1), "radius": 0.7, "background": 0.3,
            "extent_m": 2e-6, "origin_m": (1e-6, -1e-6),
        },
    }
