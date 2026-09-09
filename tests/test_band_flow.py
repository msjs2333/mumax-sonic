"""Frequency observations preserve physical history through replay and attention."""
from dataclasses import replace
import json
import numpy as np
import pytest
from mumax_sonic.sources.band_demo import make_band_frame
from mumax_sonic.sources.replay import save_replay, load_replay
from mumax_sonic.field_pipeline import observe_field
from mumax_sonic.attention import Attention
from mumax_sonic.mapping import map_sample
from mumax_sonic.reporting import report_json
from mumax_sonic.observers.band import BandConfig, band_power


def history(name='band_in', count=256):
    return tuple(make_band_frame(name, i, size=5) for i in range(count))


def test_spatial_power_is_conserved_before_attention():
    frames = history('band_opposite')
    view = observe_field(frames[-1], 'band', history=frames)
    assert view.sample.validity == 'valid'
    assert sum(o.strength for o in view.sample.observations) == pytest.approx(view.diagnostic['mean_power'])
    assert view.diagnostic['mean_power'] == pytest.approx(.005, rel=.001)
    for attention in (Attention(), Attention(center=(.8, .8), radius=.1)):
        scene = map_sample(view.sample, attention, strength_reference=.005)
        assert len(scene.sources) <= 4
    assert sum(o.strength for o in view.sample.observations) == pytest.approx(.005, rel=.001)


def test_replay_roundtrip_and_gap_recovery(tmp_path):
    frames = history(count=520)
    path = tmp_path/'band.npz'
    save_replay(path, frames[:260]+frames[261:])
    replay = load_replay(path)
    def observe(index):
        selected = replay.frames[max(0,index-255):index+1]
        return observe_field(selected[-1], 'band', history=selected)
    assert observe(255).diagnostic['mean_power'] == pytest.approx(.005, rel=.001)
    assert observe(260).sample.validity == 'warming_up'
    assert not map_sample(observe(260).sample, Attention()).sources
    assert observe(515).sample.validity == 'valid'


def test_missing_data_export_is_null_not_zero():
    frames = list(history())
    broken = frames[120].vectors.copy()
    broken[0,0] = np.nan
    frames[120] = replace(frames[120], vectors=broken)
    view = observe_field(frames[-1], 'band', history=frames)
    assert view.sample.validity == 'invalid'
    assert view.sample.coverage == pytest.approx(24/25)
    assert not map_sample(view.sample, Attention()).sources
    first = observe_field(frames[0], 'band')
    assert json.loads(report_json(first.diagnostic))['mean_power'] is None


def test_history_cannot_end_after_observed_frame():
    frames = history(count=2)
    with pytest.raises(ValueError):
        observe_field(frames[0], 'band', history=frames)


def test_static_empty_and_unresolved_band_are_unavailable():
    frames = history()
    assert band_power([replace(f, time_kind='relaxation') for f in frames]).validity == 'unsupported'
    assert band_power([replace(f, mask=np.zeros((5,5), dtype=bool)) for f in frames]).validity == 'invalid'
    assert band_power(frames, BandConfig(9.9e9,10.1e9)).validity == 'unsupported'


def test_normalization_extremes_and_constant_background():
    frames = history()
    expected = band_power(frames).mean_power
    scaled = tuple(replace(f, vectors=f.vectors*(1e300 if i%2 else 1e-300)) for i,f in enumerate(frames))
    assert band_power(scaled).mean_power == pytest.approx(expected)
    constant = tuple(replace(f, vectors=frames[0].vectors) for f in frames)
    assert band_power(constant).mean_power == 0
    assert band_power(constant).validity == 'valid'


def test_old_corruption_is_outside_current_window():
    frames = history(count=257)
    corrupt_old = replace(frames[0], sequence=99)
    result = band_power((corrupt_old,)+frames[1:])
    assert result.validity == 'valid'
    assert all(8e9 <= f <= 12e9 for f in result.bin_frequencies_hz)
