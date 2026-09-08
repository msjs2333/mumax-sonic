"""Physical rates must be invariant to listening controls and navigation."""
import json
from dataclasses import replace
import numpy as np
import pytest
from mumax_sonic.attention import Attention
from mumax_sonic.field_pipeline import observe_field
from mumax_sonic.mapping import map_sample
from mumax_sonic.reporting import report_json
from mumax_sonic.session import Transport
from mumax_sonic.sources.activity_demo import make_activity_frame, OMEGA_RAD_S, STEP_S
from mumax_sonic.sources.replay import FieldReplay


def test_activity_tile_sum_and_fixed_reference_are_independent_of_roi():
    previous = make_activity_frame('activity_rotation', 2)
    current = make_activity_frame('activity_rotation', 3)
    view = observe_field(current, 'activity', previous=previous)
    assert view.sample.validity == 'valid'
    assert sum(o.strength for o in view.sample.observations) == pytest.approx(OMEGA_RAD_S)
    assert all(o.unit == 'rad/s' for o in view.sample.observations)
    first = map_sample(view.sample, Attention(), strength_reference=1e9)
    second = map_sample(view.sample, Attention(), strength_reference=4e9)
    assert [s.source_id for s in first.sources] == [s.source_id for s in second.sources]
    assert [s.gain for s in second.sources] == pytest.approx([s.gain/2 for s in first.sources])
    map_sample(view.sample, Attention(center=(1, 1), radius=.2), strength_reference=1e9)
    assert view.diagnostic['mean_rad_s'] == pytest.approx(OMEGA_RAD_S)


def test_localized_activity_retains_spatial_structure():
    previous = make_activity_frame('activity_localized', 1)
    current = make_activity_frame('activity_localized', 2)
    view = observe_field(current, 'activity', previous=previous)
    assert 0 < view.diagnostic['mean_rad_s'] < view.diagnostic['max_rad_s'] <= OMEGA_RAD_S
    strongest = max(view.sample.observations, key=lambda o: o.strength)
    assert strongest.position_m[0] > 0 and strongest.position_m[1] < 0


def test_speed_pause_and_backward_seek_use_source_pairs():
    frames = tuple(make_activity_frame('activity_rotation', i) for i in range(12))
    replay = FieldReplay(frames, 'test')
    for speed in (.25, 1, 4):
        transport = Transport(playing=True)
        transport.set_speed(speed)
        transport.advance(.4/speed)
        previous, current = replay.pair_at(transport.sim_time_s)
        view = observe_field(current, 'activity', previous=previous)
        assert view.diagnostic['dt_s'] == pytest.approx(STEP_S)
        assert view.diagnostic['mean_rad_s'] == pytest.approx(OMEGA_RAD_S)
        transport.playing = False
        before = transport.sim_time_s
        transport.advance(100)
        assert transport.sim_time_s == before
        transport.seek(STEP_S)
        previous, current = replay.pair_at(transport.sim_time_s)
        assert observe_field(current, 'activity', previous=previous).diagnostic['mean_rad_s'] == pytest.approx(OMEGA_RAD_S)


def test_gap_and_first_frame_silence_but_valid_zero_is_measured():
    first = make_activity_frame('activity_rotation', 0)
    later = make_activity_frame('activity_rotation', 3)
    for previous in (None, first):
        view = observe_field(later, 'activity', previous=previous)
        assert view.sample.validity == 'warming_up'
        assert not map_sample(view.sample, Attention(), strength_reference=1e9).sources
        assert json.loads(report_json(view.diagnostic))['mean_rad_s'] is None
    unchanged = replace(first, sequence=1, sim_time_s=STEP_S)
    view = observe_field(unchanged, 'activity', previous=first)
    assert view.sample.validity == 'valid' and view.sample.coverage == 1
    assert view.diagnostic['mean_rad_s'] == 0
    assert json.loads(report_json(view.diagnostic))['mean_rad_s'] == 0


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1])
def test_transport_rejects_invalid_clock_and_seek(bad):
    t = Transport(playing=True)
    with pytest.raises(ValueError): t.advance(bad)
    with pytest.raises(ValueError): t.seek(bad)
    with pytest.raises(ValueError): t.set_speed(bad)
    assert t.sim_time_s == 0
