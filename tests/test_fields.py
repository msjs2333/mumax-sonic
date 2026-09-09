import numpy as np
import pytest
from mumax_sonic.fields import FieldFrame
from mumax_sonic.sources.replay import load_replay, save_replay
from mumax_sonic.sources.analytic import make_field
from mumax_sonic.field_pipeline import observe_field
from mumax_sonic.attention import Attention
from mumax_sonic.mapping import map_sample
from mumax_sonic.audio.openal import orientation_controls


def frame(name='opposite_pair', t=0, n=33):
    return FieldFrame(make_field(name, t, n), 2e-6/(n-1), 2e-6/(n-1), t, (-1e-6, -1e-6, 0))


def test_field_owns_input_and_preserves_raw_magnitude():
    raw = np.ones((3, 4, 3))*2
    field = FieldFrame(raw, 1e-9, 2e-9)
    raw[:] = 0
    assert np.all(field.vectors == 2)
    assert not field.vectors.flags.writeable
    with pytest.raises(ValueError): FieldFrame(raw, float('nan'), 1)


def test_replay_roundtrip_has_identity_and_holds_physical_frames(tmp_path):
    from dataclasses import replace
    frames = [replace(frame('wall_inplane', t), sequence=i) for i, t in enumerate((1e-9, 2e-9))]
    path = tmp_path/'field.npz'
    save_replay(path, frames)
    replay = load_replay(path)
    assert len(replay.sha256) == 64
    assert replay.at(1.8e-9).sim_time_s == 1e-9
    assert replay.at(99).sim_time_s == 2e-9
    assert np.array_equal(replay.frames[0].vectors, frames[0].vectors)
    assert replay.frames[0].source_kind == 'replay'
    assert 'original_source:synthetic' in replay.frames[0].provenance
    with pytest.raises(FileExistsError): save_replay(path, frames)
    with pytest.raises(ValueError): save_replay(tmp_path/'bad.npz', frames[::-1])


def test_source_label_shares_readonly_payload_without_mutating_original():
    original = frame()
    relabeled = original.with_source_kind('live')
    assert relabeled.source_kind == 'live'
    assert original.source_kind == 'synthetic'
    assert relabeled.vectors is original.vectors
    assert relabeled.mask is original.mask
    with pytest.raises(ValueError):
        relabeled.vectors[0, 0, 0] = 0


def test_positive_negative_aggregation_conserves_charge_before_attention():
    view = observe_field(frame())
    assert view.sample.validity == 'valid'
    assert abs(view.diagnostic['q_net']) < 1e-10
    assert view.diagnostic['q_abs'] > 1.9
    for sign, key in ((1, 'q_pos'), (-1, 'q_neg')):
        assert sum(o.strength for o in view.sample.observations if o.sign == sign) == pytest.approx(view.diagnostic[key])
    before = view.diagnostic.copy()
    map_sample(view.sample, Attention(center=(.8, .8)), mode='negative')
    assert view.diagnostic == before
    assert {s.sign for s in map_sample(view.sample, Attention(background=1)).sources} == {-1, 1}


def test_direction_pipeline_uses_all_components_and_emits_periodic_code():
    view = observe_field(frame('wall_inplane', 1e-9), 'direction')
    assert view.sample.observations
    assert all(o.orientation_rad == pytest.approx(np.pi/4) for o in view.sample.observations)
    scene = map_sample(view.sample, Attention())
    assert all(o.orientation_enabled for o in scene.sources)
    assert orientation_controls(np.pi-1e-9) == pytest.approx(orientation_controls(-np.pi+1e-9), abs=1e-8)
    assert orientation_controls(np.pi/2) != orientation_controls(-np.pi/2)


def test_mask_hole_is_partial_support_not_valid_zero():
    f = frame('uniform')
    mask = f.mask.copy()
    mask[16, 16] = False
    view = observe_field(FieldFrame(f.vectors, f.dx_m, f.dy_m, mask=mask))
    assert view.sample.validity == 'invalid'
    assert view.sample.coverage < 1
    assert not map_sample(view.sample, Attention()).sources


def test_coordinate_translation_preserves_audio_direction():
    from mumax_sonic.model import Observation, Sample
    a = map_sample(Sample(0, (Observation('a', (.5, 0, 0), 1),)), Attention(extent_m=1))
    b = map_sample(Sample(0, (Observation('a', (10.5, 20, 0), 1),)), Attention(extent_m=1, origin_m=(10, 20)))
    assert a.sources[0].position == b.sources[0].position
    assert a.sources[0].gain == b.sources[0].gain
