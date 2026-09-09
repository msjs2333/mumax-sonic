"""Batch publication and physical window contracts for live OVF reading."""
import hashlib
import json

import numpy as np
import pytest

from mumax_sonic.sources.incremental import IncrementalOVFReader
from mumax_sonic.sources.ovf_replay import load_ovf_replay
from test_ovf_replay import manifest, write_ovf


def _publish(path, meta):
    path.write_text(json.dumps(meta), encoding="utf-8")


def _snapshot(reader):
    return reader.history_keys, reader.records, reader.diagnostics


@pytest.mark.parametrize('retain', [2, 4])
def test_explicit_timeline_reads_only_required_window(tmp_path, retain):
    path, meta = manifest(tmp_path, count=10)
    for record in meta['frames']:
        record['time_s'] = record['sequence']*1e-11
    # Obsolete bodies need not be accessible for the live observation window.
    for record in meta['frames'][:-retain]:
        (tmp_path/record['file']).unlink()
    _publish(path, meta)
    reader = IncrementalOVFReader(retain_frames=retain)
    replay = reader.load(path)
    assert [frame.sequence for frame in replay.frames] == list(range(10-retain, 10))
    assert len(reader.records) == 10
    assert reader.diagnostics['read_frames_total'] == retain
    assert reader.new_source_frames_count == 10


def test_skipped_body_still_has_ordered_physical_time(tmp_path):
    path, meta = manifest(tmp_path, count=6)
    for record in meta['frames']:
        record['time_s'] = record['sequence']*1e-11
    meta['frames'][2]['time_s'] = 0
    _publish(path, meta)
    with pytest.raises(ValueError, match='times'):
        IncrementalOVFReader(retain_frames=2).load(path)


def test_batch_with_truncated_last_ovf_is_transactional_then_recovers(tmp_path):
    path, published = manifest(tmp_path, count=5)
    initial = json.loads(json.dumps(published))
    initial["frames"] = initial["frames"][:2]
    _publish(path, initial)

    reader = IncrementalOVFReader(retain_frames=2)
    assert [frame.sequence for frame in reader.load(path).frames] == [0, 1]
    before = _snapshot(reader)

    broken = json.loads(json.dumps(published))
    last_source = tmp_path / broken["frames"][-1]["file"]
    last_raw = last_source.read_bytes()
    last_source.write_bytes(last_raw[:-20])
    _publish(path, broken)
    with pytest.raises(ValueError, match="OVF"):
        reader.load(path)
    assert _snapshot(reader) == before

    last_source.write_bytes(last_raw)
    _publish(path, published)
    assert [frame.sequence for frame in reader.load(path).frames] == [3, 4]
    assert reader.new_source_frames_count == 3


def test_wrong_mask_shape_rejects_new_batch_and_preserves_cache(tmp_path):
    path, published = manifest(tmp_path, count=3)
    mask_path = tmp_path / "mask.npy"
    mask = np.ones((3, 4), dtype=bool)
    np.save(mask_path, mask)
    published["mask"] = {
        "file": mask_path.name,
        "sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
    }
    initial = json.loads(json.dumps(published))
    initial["frames"] = initial["frames"][:2]
    _publish(path, initial)
    reader = IncrementalOVFReader(retain_frames=2)
    reader.load(path)
    before = _snapshot(reader)

    np.save(mask_path, np.ones((2, 2), dtype=bool))
    _publish(path, published)
    with pytest.raises(ValueError, match="mask shape"):
        reader.load(path)
    assert _snapshot(reader) == before


def test_time_evidence_is_not_read_by_realtime_batch_reader(tmp_path):
    path, published = manifest(tmp_path, count=3)
    evidence = tmp_path / "source-times.json"
    published["time_evidence"] = {
        "file": "missing-source-times.json",
        "sha256": "0" * 64,
        "description": "Recorded solver times in seconds",
    }
    for i, record in enumerate(published["frames"]):
        record["time_s"] = i * 1e-11
    initial = json.loads(json.dumps(published))
    initial["frames"] = initial["frames"][:2]
    _publish(path, initial)
    reader = IncrementalOVFReader(retain_frames=2)
    reader.load(path)
    _publish(path, published)
    assert [frame.sequence for frame in reader.load(path).frames] == [1, 2]
    assert reader.new_source_frames_count == 1


def test_batch_matches_offline_xyz_time_and_mask(tmp_path):
    path, published = manifest(tmp_path, count=4)
    mask_path = tmp_path / "mask.npy"
    mask = np.ones((3, 4), dtype=bool)
    mask[0, 1] = False
    np.save(mask_path, mask)
    published["mask"] = {
        "file": mask_path.name,
        "sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
    }
    _publish(path, published)

    offline = load_ovf_replay(path)
    incremental = IncrementalOVFReader(retain_frames=4).load(path)
    assert [frame.sequence for frame in incremental.frames] == [0, 1, 2, 3]
    for actual, expected in zip(incremental.frames, offline.frames):
        np.testing.assert_array_equal(actual.vectors, expected.vectors)
        np.testing.assert_array_equal(actual.mask, expected.mask)
        assert actual.sim_time_s == pytest.approx(expected.sim_time_s)
        assert actual.origin_m == pytest.approx(expected.origin_m)
