import json
import numpy as np
import pytest

from mumax_sonic.fields import FieldFrame
from mumax_sonic.sources.replay import FieldReplay, load_replay, save_replay


def _frame(time_s, sequence, provenance='analytical:probe'):
    vectors = np.zeros((3, 3, 3), dtype=float)
    vectors[..., 0] = sequence + 1
    return FieldFrame(vectors, 1e-9, 1e-9, time_s, sequence=sequence,
                      provenance=provenance)


def test_v2_roundtrip_preserves_gaps_and_per_frame_provenance(tmp_path):
    frames = (_frame(0.1, 4, 'analytical:first'),
              _frame(0.4, 9, 'analytical:gap-after-first'),
              _frame(1.2, 20, 'analytical:last'))
    path = tmp_path / 'gapped.npz'
    save_replay(path, frames)

    replay = load_replay(path)
    assert [frame.sequence for frame in replay.frames] == [4, 9, 20]
    assert [frame.sim_time_s for frame in replay.frames] == [0.1, 0.4, 1.2]
    assert all('analytical:' in frame.provenance for frame in replay.frames)
    assert 'original_source:synthetic' in replay.frames[0].provenance
    assert replay.sha256 in replay.frames[0].provenance


def test_v2_rejects_duplicate_or_decreasing_sequences(tmp_path):
    with pytest.raises(ValueError, match='sequences'):
        save_replay(tmp_path / 'duplicate.npz', (_frame(0.0, 2), _frame(1.0, 2)))
    with pytest.raises(ValueError, match='sequences'):
        save_replay(tmp_path / 'decreasing.npz', (_frame(0.0, 3), _frame(1.0, 1)))


def test_v1_load_assigns_contiguous_sequences_and_marks_inference(tmp_path):
    vectors = np.zeros((2, 3, 3, 3), dtype=float)
    mask = np.ones((2, 3, 3), dtype=bool)
    metadata = dict(schema_version=1, axes='yx', components='xyz', spacing_unit='m',
                    dx_m=1e-9, dy_m=1e-9, origin_m=(0.0, 0.0, 0.0),
                    entity_id='m', segment_id='field', time_kind='dynamics',
                    original_source_kind='synthetic')
    path = tmp_path / 'v1.npz'
    np.savez_compressed(path, vectors=vectors, mask=mask,
                        times_s=np.array([0.2, 0.9]), metadata=np.array(json.dumps(metadata)))

    replay = load_replay(path)
    assert [frame.sequence for frame in replay.frames] == [0, 1]
    assert all('sequence_inferred:v1' in frame.provenance for frame in replay.frames)


def test_index_and_pair_use_source_predecessor_for_fast_or_slow_display():
    replay = FieldReplay((_frame(0.0, 10), _frame(1.0, 20), _frame(2.0, 30)), 'a' * 64)
    assert replay.index_at(-1.0) == 0
    assert replay.index_at(0.9) == 0
    assert replay.index_at(1.0) == 1
    assert replay.at(1.8).sequence == 20
    previous, current = replay.pair_at(1.8)
    assert previous.sequence == 10
    assert current.sequence == 20
    previous, current = replay.pair_at(99.0)
    assert previous.sequence == 20
    assert current.sequence == 30


def test_pair_before_first_has_no_predecessor_and_times_must_increase():
    replay = FieldReplay((_frame(2.0, 5), _frame(4.0, 8)), 'b' * 64)
    previous, current = replay.pair_at(0.0)
    assert previous is None
    assert current.sequence == 5
    with pytest.raises(ValueError, match='strictly increasing'):
        FieldReplay((_frame(1.0, 1), _frame(1.0, 2)), 'c' * 64)
