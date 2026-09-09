"""P2f adaptive spatial aggregation preserves observer physics and quality."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest

from mumax_sonic.attention import Attention
from mumax_sonic.field_pipeline import apply_aggregation, observe_field
from mumax_sonic.fields import FieldFrame
from mumax_sonic.sources.analytic import make_field
from mumax_sonic.sources.activity_demo import make_activity_frame
from mumax_sonic.sources.band_demo import make_band_frame
from mumax_sonic.reporting import report_json


def _opposite_view():
    values = make_field("opposite_pair")
    return observe_field(
        FieldFrame(values, 2e-6 / 64, 2e-6 / 64, origin_m=(-1e-6, -1e-6, 0.0)),
        "topology",
    )


def test_adaptive_topology_preserves_physical_channels_and_raw_mass():
    view = _opposite_view()
    before = deepcopy(view.diagnostic)
    original_ids = tuple(item.source_id for item in view.sample.observations)
    attention = Attention(extent_m=1e-6, background=1.0)

    for budget in (2, 4, 8):
        adapted = apply_aggregation(view, attention, budget=budget, mode="adaptive")
        assert adapted.diagnostic["spatial_aggregation"]["method"] == "adaptive"
        assert adapted.diagnostic["adaptive_aggregation"]["status"] == "valid"
        assert len(adapted.sample.observations) <= budget
        assert adapted.diagnostic["q_pos"] == pytest.approx(view.diagnostic["q_pos"])
        assert adapted.diagnostic["q_neg"] == pytest.approx(view.diagnostic["q_neg"])
        assert adapted.diagnostic["q_abs"] == pytest.approx(view.diagnostic["q_abs"])
        channels = adapted.diagnostic["spatial_aggregation"]["channels"]
        for label in ("positive", "negative", "absolute"):
            assert channels[label]["represented"] == pytest.approx(channels[label]["input"])
            assert channels[label]["omitted"] == pytest.approx(0.0)
        assert sum(item.strength for item in adapted.sample.observations) == pytest.approx(
            view.diagnostic["q_abs"]
        )

    # Aggregation returns a new view and must not rewrite the fixed view.
    assert view.diagnostic == before
    assert tuple(item.source_id for item in view.sample.observations) == original_ids


def test_adaptive_activity_preserves_mean_signed_sum_and_warming_quality():
    previous = make_activity_frame("activity_rotation", 2)
    current = make_activity_frame("activity_rotation", 3)
    view = observe_field(current, "activity", previous=previous)
    before = deepcopy(view.diagnostic)
    adapted = apply_aggregation(view, Attention(background=1.0), budget=4, mode="adaptive")

    assert adapted.diagnostic["mean_rad_s"] == pytest.approx(view.diagnostic["mean_rad_s"])
    assert sum(item.strength for item in adapted.sample.observations) == pytest.approx(
        sum(item.strength for item in view.sample.observations)
    )
    assert len(adapted.sample.observations) <= 4
    assert view.diagnostic == before

    warming = observe_field(current, "activity", previous=None)
    warmed = apply_aggregation(warming, Attention(), budget=4, mode="adaptive")
    assert warmed.sample.validity == "warming_up"
    assert warmed.sample.observations == ()
    assert math.isnan(warmed.diagnostic["mean_rad_s"])
    assert json.loads(report_json(warmed.diagnostic))["mean_rad_s"] is None
    assert warmed.diagnostic["adaptive_aggregation"]["status"] == "warming_up"


def test_adaptive_requires_explicit_fallback_for_direction_and_one_signed_slot():
    vectors = np.zeros((8, 8, 3), dtype=float)
    vectors[..., 1] = 1.0
    direction = observe_field(FieldFrame(vectors, 1e-9, 1e-9), "direction")
    fallback = apply_aggregation(direction, Attention(), budget=4, mode="adaptive")
    assert fallback.sample == direction.sample
    assert fallback.diagnostic["spatial_aggregation"]["method"] == "fixed_4x4"
    assert fallback.diagnostic["adaptive_aggregation"]["status"] == "unsupported"
    assert fallback.diagnostic["adaptive_aggregation"]["fallback"] == "fixed_4x4"

    signed = apply_aggregation(_opposite_view(), Attention(), budget=1, mode="adaptive")
    assert signed.sample.observations == _opposite_view().sample.observations
    assert signed.diagnostic["spatial_aggregation"]["method"] == "fixed_4x4"
    assert signed.diagnostic["adaptive_aggregation"]["status"] == "unsupported"
    assert "both signs" in signed.diagnostic["adaptive_aggregation"]["reason"]


def test_adaptive_band_preserves_window_power_and_raw_mass():
    history = tuple(make_band_frame("band_in", index) for index in range(256))
    view = observe_field(history[-1], "band", history=history)
    assert view.sample.validity == "valid"
    for budget in (2, 4, 8):
        adapted = apply_aggregation(view, Attention(background=1.0), budget=budget,
                                    mode="adaptive")
        assert adapted.sample.validity == "valid"
        assert len(adapted.sample.observations) <= budget
        assert adapted.diagnostic["mean_power"] == pytest.approx(view.diagnostic["mean_power"])
        assert sum(item.strength for item in adapted.sample.observations) == pytest.approx(
            view.diagnostic["mean_power"]
        )
        channels = adapted.diagnostic["spatial_aggregation"]["channels"]
        assert channels["positive"]["input"] == pytest.approx(view.diagnostic["mean_power"])
        assert channels["positive"]["represented"] == pytest.approx(view.diagnostic["mean_power"])
        assert channels["positive"]["omitted"] == pytest.approx(0.0)


def test_fixed_default_is_identity_and_invalid_mode_is_rejected():
    view = _opposite_view()
    assert apply_aggregation(view, Attention(), budget=8) is view
    with pytest.raises(ValueError):
        apply_aggregation(view, Attention(), mode="unknown")


def test_contribution_grid_is_read_only():
    view = _opposite_view()
    assert view.contributions is not None
    for array in (view.contributions.x_m, view.contributions.y_m,
                  view.contributions.positive, view.contributions.negative):
        assert not array.flags.writeable
