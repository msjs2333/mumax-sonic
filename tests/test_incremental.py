"""Bounded append-only OVF manifest cache behaviour."""
import json

import numpy as np
import pytest

from mumax_sonic.sources.incremental import IncrementalOVFReader
import mumax_sonic.sources.ovf_replay as ovf_replay
from test_ovf_replay import manifest, write_ovf


def _publish(path, meta):
    path.write_text(json.dumps(meta), encoding='utf-8')


def test_appends_decode_only_new_ovf_bodies(tmp_path, monkeypatch):
    path, appended = manifest(tmp_path, count=3)
    meta = json.loads(json.dumps(appended))
    meta['frames'] = meta['frames'][:2]
    _publish(path, meta)
    reader = IncrementalOVFReader(retain_frames=2)
    calls = []
    original = ovf_replay.read_ovf
    monkeypatch.setattr(ovf_replay, 'read_ovf', lambda value: calls.append(value) or original(value))
    assert [f.sequence for f in reader.load(path).frames] == [0, 1]
    assert len(calls) == 2
    assert [f.sequence for f in reader.load(path).frames] == [0, 1]
    assert len(calls) == 2
    _publish(path, appended)
    assert [f.sequence for f in reader.load(path).frames] == [1, 2]
    assert len(calls) == 3
    assert reader.diagnostics['read_frames_total'] == 3
    assert reader.diagnostics['cache_hits_total'] >= 2


def test_eviction_keeps_metadata_for_full_append_prefix(tmp_path):
    path, appended = manifest(tmp_path, count=5)
    meta = json.loads(json.dumps(appended))
    meta['frames'] = meta['frames'][:4]
    _publish(path, meta)
    reader = IncrementalOVFReader(retain_frames=2)
    replay = reader.load(path)
    assert [frame.sequence for frame in replay.frames] == [2, 3]
    assert [key[1] for key in reader.history_keys] == [0, 1, 2, 3]
    assert [record[1] for record in reader.records] == [0, 1, 2, 3]
    _publish(path, appended)
    assert [frame.sequence for frame in reader.load(path).frames] == [3, 4]


def test_historical_change_fails_closed_and_recovery_preserves_state(tmp_path):
    path, meta = manifest(tmp_path, count=3)
    reader = IncrementalOVFReader(retain_frames=2)
    good = reader.load(path)
    source = tmp_path / meta['frames'][0]['file']
    raw = source.read_bytes()
    source.write_bytes(raw + b'changed')
    with pytest.raises(ValueError, match='source changed'):
        reader.load(path)
    assert [frame.sequence for frame in good.frames] == [1, 2]
    source.write_bytes(raw)
    assert [frame.sequence for frame in reader.load(path).frames] == [1, 2]


def test_changed_historical_manifest_record_fails_closed(tmp_path):
    path, meta = manifest(tmp_path, count=3)
    reader = IncrementalOVFReader(retain_frames=2)
    reader.load(path)
    changed = json.loads(json.dumps(meta))
    changed['frames'][0]['sha256'] = '0' * 64
    _publish(path, changed)
    with pytest.raises(ValueError, match='provenance changed'):
        reader.load(path)


def test_cross_frame_geometry_and_header_time_regressions_fail_closed(tmp_path):
    path, meta = manifest(tmp_path, count=2)
    reader = IncrementalOVFReader(retain_frames=2)
    reader.load(path)
    wrong_shape = tmp_path / 'm000002.ovf'
    digest = write_ovf(wrong_shape, np.zeros((1, 4, 4, 3)), 2e-11)
    changed = json.loads(json.dumps(meta))
    changed['frames'].append(dict(file=wrong_shape.name, sha256=digest, sequence=2))
    _publish(path, changed)
    with pytest.raises(ValueError, match='mesh'):
        reader.load(path)
    # A fresh reader reaches the time-order check even when records rely on
    # OVF header times instead of explicit manifest times.
    path, meta = manifest(tmp_path, count=2)
    late = tmp_path / 'm000002.ovf'
    digest = write_ovf(late, np.zeros((1, 3, 4, 3)), .5e-11)
    meta['frames'].append(dict(file=late.name, sha256=digest, sequence=2))
    _publish(path, meta)
    with pytest.raises(ValueError, match='times'):
        IncrementalOVFReader(retain_frames=2).load(path)


def test_requested_tail_over_memory_limit_never_drops_below_retain_count(tmp_path):
    path, _ = manifest(tmp_path, count=3)
    with pytest.raises(ValueError, match='requested retained'):
        IncrementalOVFReader(retain_frames=3, max_bytes=700).load(path)
