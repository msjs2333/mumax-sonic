import json
import tempfile
import time

import numpy as np
import pytest

from mumax_sonic.observers.band import BandConfig
from mumax_sonic.sources.attachment import AttachedSimulation
from mumax_sonic.sources.incremental import IncrementalOVFReader
from mumax_sonic.sources.live import LiveConfig
from mumax_sonic.sources.mumax3_bridge import MuMax3Bridge


@pytest.fixture(autouse=True)
def _scratch_inside_workspace(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))


def _write_ovf(path, sim_time_s, phase):
    basis = np.array([[np.cos(phase), np.sin(phase), 0.0],
                      [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    values = np.tile(basis, (3, 1)).reshape(-1)
    header = "\n".join(("# OOMMF OVF 2.0", "# Segment count: 1", "# Begin: Segment",
        "# Begin: Header", "# meshtype: rectangular", "# meshunit: nm", "# xbase: 0",
        "# ybase: 0", "# zbase: 0", "# xstepsize: 1", "# ystepsize: 1",
        "# zstepsize: 1", "# xnodes: 3", "# ynodes: 3", "# znodes: 1",
        "# valuedim: 3", "# valuelabels: m_x m_y m_z", "# valueunits: 1",
        f"# Desc: Total simulation time: {sim_time_s} s", "# End: Header",
        "# Begin: Data Text")) + "\n"
    path.write_bytes((header + " ".join(map(str, values)) +
                      "\n# End: Data Text\n# End: Segment\n").encode())


def _snapshot_file(directory, sequence, *, prefix="relax", sim_time_s=0.0, phase=0.0):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}{sequence:04d}.ovf"
    _write_ovf(path, sim_time_s, phase)
    return path


def _wait_for(attachment, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        latest = attachment.snapshot()
        if predicate(latest):
            return latest
        time.sleep(0.01)
    raise AssertionError(f"condition not reached; last snapshot: {latest!r}")


def _poll_until_current(bridge):
    result = None
    for _ in range(5):
        result = bridge.poll()
        if result["state"] == "current":
            return result
        time.sleep(0.01)
    return result


def _close(attachment):
    attachment.close()
    if attachment._thread is not None:
        attachment._thread.join(timeout=2.0)
        assert not attachment._thread.is_alive(), "attachment worker did not stop after close"


@pytest.mark.parametrize("time_kind", ["static", "relaxation"])
def test_snapshot_bridge_publishes_only_latest_with_new_segment_and_incremental_roundtrip(tmp_path, time_kind):
    output = tmp_path / "out"
    older = _snapshot_file(output, 0, sim_time_s=0.0, phase=0.1)
    latest = _snapshot_file(output, 1, sim_time_s=2e-12, phase=0.4)
    old_bytes, latest_bytes = older.read_bytes(), latest.read_bytes()
    manifest_path = tmp_path / "snapshot.json"
    reader = IncrementalOVFReader(retain_frames=16)
    bridge = MuMax3Bridge(output, manifest_path, entity_id="m", segment_id="requested",
        all_material=True, time_kind=time_kind, pattern="relax[0-9]*.ovf")
    try:
        first_result = _poll_until_current(bridge)
        assert first_result["state"] == "current"
        first_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert [(f["sequence"], f["time_s"]) for f in first_manifest["frames"]] == [(1, 2e-12)]
        assert first_manifest["time_kind"] == time_kind
        first_segment = first_manifest["segment_id"]
        first_replay = reader.load(manifest_path)
        assert [(f.sequence, f.sim_time_s) for f in first_replay.frames] == [(1, 2e-12)]

        newest = _snapshot_file(output, 2, sim_time_s=0.0, phase=0.8)
        newest_bytes = newest.read_bytes()
        result = _poll_until_current(bridge)
        assert result["state"] == "current"
        next_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert [(f["sequence"], f["time_s"]) for f in next_manifest["frames"]] == [(2, 0.0)]
        assert next_manifest["segment_id"] != first_segment
        next_replay = reader.load(manifest_path)
        assert [(f.sequence, f.sim_time_s) for f in next_replay.frames] == [(2, 0.0)]
        assert reader.new_source_frames_count == 1
        assert (older.read_bytes(), latest.read_bytes(), newest.read_bytes()) == (
            old_bytes, latest_bytes, newest_bytes)
    finally:
        bridge.close()


@pytest.mark.parametrize("time_kind", ["static", "relaxation"])
def test_attached_topology_follows_same_time_snapshots_without_writing_source_files(tmp_path, time_kind):
    output = tmp_path / "out"
    first = _snapshot_file(output, 0, sim_time_s=0.0, phase=0.1)
    latest = _snapshot_file(output, 1, sim_time_s=0.0, phase=0.4)
    original_bytes = {path: path.read_bytes() for path in (first, latest)}
    attachment = AttachedSimulation(output, LiveConfig(recipe="topology", poll_interval_s=0.01,
        stale_after_s=1.0), all_material=True, time_kind=time_kind,
        pattern="relax[0-9]*.ovf").start()
    try:
        initial = _wait_for(attachment, lambda s: s["state"] == "current")
        assert initial["view"].sample.validity == "valid"
        assert initial["view"].field.sequence == 1
        assert initial["view"].field.sim_time_s == 0.0
        assert initial["view"].sample.time_kind == time_kind
        first_segment = initial["view"].field.segment_id
        old_updates = initial["updates"]

        newest = _snapshot_file(output, 2, sim_time_s=0.0, phase=0.8)
        original_bytes[newest] = newest.read_bytes()
        advanced = _wait_for(attachment, lambda s: s["state"] == "current" and
                             s["updates"] > old_updates)
        assert advanced["view"].field.sequence == 2
        assert advanced["view"].field.sim_time_s == 0.0
        assert advanced["view"].field.segment_id != first_segment
        assert advanced["view"].sample.validity == "valid"
        assert advanced["age_s"] < 1.0
        assert {path: path.read_bytes() for path in original_bytes} == original_bytes
    finally:
        _close(attachment)


@pytest.mark.parametrize("time_kind", ["static", "relaxation"])
@pytest.mark.parametrize("recipe", ["activity", "band"])
def test_non_dynamic_activity_and_band_are_unsupported_not_rates(tmp_path, time_kind, recipe):
    output = tmp_path / f"{time_kind}-{recipe}"
    _snapshot_file(output, 0, sim_time_s=0.0, phase=0.1)
    _snapshot_file(output, 1, sim_time_s=0.0, phase=0.4)
    config = LiveConfig(recipe=recipe, poll_interval_s=0.01, stale_after_s=1.0,
                        band_config=BandConfig(window_samples=16) if recipe == "band" else None)
    attachment = AttachedSimulation(output, config, all_material=True, time_kind=time_kind,
                                    pattern="relax[0-9]*.ovf").start()
    try:
        result = _wait_for(attachment, lambda s: s["state"] == "current")
        view = result["view"]
        assert view.sample.validity == "unsupported"
        assert view.sample.time_kind == time_kind
        assert not view.sample.observations
        assert view.diagnostic["validity"] == "unsupported"
        assert "requires dynamic field frames" in view.diagnostic["reason"]
        assert result["attachment"]["published_frames"] == 1
    finally:
        _close(attachment)


def test_dynamics_still_rejects_equal_physical_times(tmp_path):
    output = tmp_path / "out"
    _snapshot_file(output, 0, prefix="m", sim_time_s=1e-12, phase=0.1)
    bridge = MuMax3Bridge(output, tmp_path / "dynamics.json", entity_id="m", segment_id="dynamics",
                          all_material=True, time_kind="dynamics", pattern="m[0-9]*.ovf")
    try:
        assert bridge.poll()["state"] == "waiting"
        assert bridge.poll()["state"] == "current"
        _snapshot_file(output, 1, prefix="m", sim_time_s=1e-12, phase=0.4)
        assert bridge.poll()["state"] == "waiting"
        result = bridge.poll()
        assert result["state"] == "invalid"
        assert "physical times must increase" in result["reason"]
    finally:
        bridge.close()


@pytest.mark.parametrize("time_kind", ["static", "relaxation"])
def test_latest_half_written_snapshot_waits_then_recovers_from_stale(tmp_path, time_kind):
    output = tmp_path / 'out'
    _snapshot_file(output, 0)
    attachment = AttachedSimulation(output, LiveConfig(recipe='topology', poll_interval_s=.01,
        stale_after_s=.15), all_material=True, time_kind=time_kind,
        pattern='relax[0-9]*.ovf').start()
    try:
        initial = _wait_for(attachment, lambda s: s['state'] == 'current')
        next_path = output / 'relax0001.ovf'
        next_path.write_bytes(b'# OOMMF OVF 2.0\n# Begin: Segment\n')
        stale = _wait_for(attachment, lambda s: s['state'] == 'stale')
        assert stale['view'].field.sequence == 0
        assert stale['updates'] == initial['updates']
        _write_ovf(next_path, 0.0, .8)
        recovered = _wait_for(attachment, lambda s: s['state'] == 'current'
            and s['view'].field.sequence == 1)
        assert recovered['view'].field.sim_time_s == 0.0
        assert recovered['view'].sample.validity == 'valid'
        assert recovered['updates'] > initial['updates']
    finally:
        _close(attachment)
