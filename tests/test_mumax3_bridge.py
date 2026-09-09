import struct
import time

import numpy as np
import pytest

from mumax_sonic.sources.mumax3_bridge import MuMax3Bridge


def _ovf(time_s=1e-12, values=None):
    values = np.zeros(12, dtype=float) if values is None else np.asarray(values, dtype=float)
    header = "\n".join(("# OOMMF OVF 2.0", "# Segment count: 1", "# Begin: Segment", "# Begin: Header",
        "# meshtype: rectangular", "# meshunit: nm", "# xbase: 0", "# ybase: 0", "# zbase: 0",
        "# xstepsize: 1", "# ystepsize: 1", "# zstepsize: 1", "# xnodes: 2", "# ynodes: 2", "# znodes: 1",
        "# valuedim: 3", "# valuelabels: m_x m_y m_z", "# valueunits: 1",
        *([] if time_s is None else [f"# Desc: Total simulation time: {time_s} s"]),
        "# End: Header", "# Begin: Data Text")) + "\n"
    return (header + " ".join(map(str, values)) + "\n# End: Data Text\n# End: Segment\n").encode()


def _bridge(tmp_path):
    return MuMax3Bridge(tmp_path / "out", tmp_path / "manifest.json", entity_id="m", segment_id="s", all_material=True)


def test_requires_mask_and_publishes_after_two_stable_polls(tmp_path):
    path = tmp_path / "out"; path.mkdir()
    with pytest.raises(ValueError):
        MuMax3Bridge(path, tmp_path / "m.json", entity_id="m", segment_id="s")
    (path / "m000.ovf").write_bytes(_ovf())
    bridge = _bridge(tmp_path)
    try:
        assert bridge.poll()["state"] == "waiting"
        result = bridge.poll()
        assert result["state"] == "current"
        assert result["published_frames"] == 1
        assert (tmp_path / "manifest.json").exists()
    finally:
        bridge.close()


def test_half_written_stable_file_waits_and_missing_time_is_visible(tmp_path):
    path = tmp_path / "out"; path.mkdir(); source = path / "m000.ovf"
    source.write_bytes(_ovf()[:80])
    bridge = _bridge(tmp_path)
    try:
        bridge.poll()
        result = bridge.poll()
        assert result["state"] == "waiting"
        source.write_bytes(_ovf(None))
        bridge._stable.clear()
        bridge.poll()
        assert "missing physical time" in bridge.poll()["reason"]
    finally:
        bridge.close()


def test_published_history_is_trusted_and_existing_output_is_not_overwritten(tmp_path):
    path = tmp_path / "out"; path.mkdir(); (path / "m000.ovf").write_bytes(_ovf())
    bridge = _bridge(tmp_path)
    bridge.poll(); bridge.poll()
    (path / "m000.ovf").write_bytes(_ovf(values=np.ones(12)))
    try:
        assert bridge.poll()["state"] == "current"
    finally:
        bridge.close()
    with pytest.raises(FileExistsError):
        _bridge(tmp_path)


def test_append_replaces_owned_manifest_and_does_not_retain_arrays(tmp_path):
    import json
    path = tmp_path / 'out'; path.mkdir()
    first = path / 'm000.ovf'; first.write_bytes(_ovf(0))
    with _bridge(tmp_path) as bridge:
        bridge.poll(); assert bridge.poll()['published_frames'] == 1
        (path / 'm001.ovf').write_bytes(_ovf(1e-12))
        bridge.poll(); result = bridge.poll()
        assert result['state'] == 'current'
        assert result['published_frames'] == 2
        records = json.loads((tmp_path/'manifest.json').read_text())['frames']
        assert [r['sequence'] for r in records] == [0, 1]
        assert all(not hasattr(entry[2], 'vectors') for entry in bridge._seen.values())


def test_partial_file_finishes_without_restarting_bridge(tmp_path):
    path = tmp_path/'out'; path.mkdir(); source = path/'m000.ovf'
    source.write_bytes(_ovf()[:80])
    with _bridge(tmp_path) as bridge:
        bridge.poll(); assert bridge.poll()['state']=='waiting'
        source.write_bytes(_ovf())
        bridge.poll(); assert bridge.poll()['published_frames']==1


def test_live_bridge_and_import_work_without_content_hashing(tmp_path, monkeypatch):
    import hashlib
    from test_ovf_replay import manifest
    from mumax_sonic.sources.ovf_replay import load_ovf_replay
    source = tmp_path/'out'
    source.mkdir()
    manifest(source, count=2)
    def forbidden(*args, **kwargs):
        raise AssertionError('live bridge must not hash content')
    monkeypatch.setattr(hashlib, 'sha256', forbidden)
    with _bridge(tmp_path) as bridge:
        bridge.poll()
        assert bridge.poll()['published_frames'] == 2
        replay = load_ovf_replay(tmp_path/'manifest.json')
        assert replay.frames[-1].sequence == 1
        assert replay.sha256 == ''
