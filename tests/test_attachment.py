import tempfile
import time

import numpy as np
import pytest

from mumax_sonic.sources.attachment import AttachedSimulation
from mumax_sonic.sources.live import LiveConfig


@pytest.fixture(autouse=True)
def _attachment_tempdir(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))


def _frame(path, sequence, *, phase=0.0):
    path.mkdir(parents=True, exist_ok=True)
    # Nonzero unit vectors make accidental empty/zero-field observations visible.
    basis = np.array([[np.cos(phase), np.sin(phase), 0.0],
                      [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    vectors = np.tile(basis, (3, 1)).reshape(-1)
    target = path / f"m{sequence:05d}.ovf"
    header = "\n".join(("# OOMMF OVF 2.0", "# Segment count: 1", "# Begin: Segment",
        "# Begin: Header", "# meshtype: rectangular", "# meshunit: nm", "# xbase: 0",
        "# ybase: 0", "# zbase: 0", "# xstepsize: 1", "# ystepsize: 1",
        "# zstepsize: 1", "# xnodes: 3", "# ynodes: 3", "# znodes: 1",
        "# valuedim: 3", "# valuelabels: m_x m_y m_z", "# valueunits: 1",
        f"# Desc: Total simulation time: {(sequence + 1) * 1e-12} s",
        "# End: Header", "# Begin: Data Text")) + "\n"
    target.write_bytes((header + " ".join(map(str, vectors)) +
                        "\n# End: Data Text\n# End: Segment\n").encode())
    return target


def _attach(path, *, poll=0.01, stale=1.0):
    return AttachedSimulation(path, LiveConfig(recipe="activity", poll_interval_s=poll,
                                                stale_after_s=stale), all_material=True)


def _wait_for(attachment, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        latest = attachment.snapshot()
        if predicate(latest):
            return latest
        time.sleep(0.01)
    raise AssertionError(f"condition not reached; last attachment snapshot: {latest!r}")


def _close(attachment):
    attachment.close()
    _wait_for_thread = attachment._thread
    if _wait_for_thread is not None:
        _wait_for_thread.join(timeout=2.0)
        assert not _wait_for_thread.is_alive(), "attachment worker did not stop after close"


def test_waits_without_data_and_publishes_only_after_half_written_frame_finishes(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    attachment = _attach(output).start()
    try:
        assert _wait_for(attachment, lambda s: s["state"] == "waiting")["view"] is None
        target = _frame(output, 0)
        content = target.read_bytes()
        target.write_bytes(content[:80])
        waiting = _wait_for(attachment, lambda s: s["attachment"].get("published_frames") == 0)
        assert waiting["state"] == "waiting"
        target.write_bytes(content)
        result = _wait_for(attachment, lambda s: s["state"] == "current")
        assert result["view"].field.sequence == 0
        assert np.all(np.linalg.norm(result["view"].field.vectors, axis=-1) > 0)
        assert result["attachment"]["published_frames"] == 1
    finally:
        _close(attachment)


def test_attaches_to_recent_frames_then_observes_appended_frame(tmp_path):
    output = tmp_path / "out"
    for sequence in range(6):
        _frame(output, sequence, phase=sequence * 0.1)
    attachment = _attach(output).start()
    try:
        initial = _wait_for(attachment, lambda s: s["state"] == "current")
        assert initial["attachment"]["published_frames"] == 2
        assert initial["view"].field.sequence == 5
        updates = initial["updates"]
        _frame(output, 6, phase=0.8)
        appended = _wait_for(attachment, lambda s: s["updates"] > updates)
        assert appended["view"].field.sequence == 6
    finally:
        _close(attachment)


def test_stale_attachment_recovers_when_new_file_arrives(tmp_path):
    output = tmp_path / "out"
    _frame(output, 0)
    attachment = _attach(output, stale=0.12).start()
    try:
        _wait_for(attachment, lambda s: s["state"] == "current")
        stale = _wait_for(attachment, lambda s: s["state"] == "stale")
        assert stale["view"].field.sequence == 0
        _frame(output, 1, phase=0.4)
        recovered = _wait_for(attachment, lambda s: s["state"] == "current" and
                              s["view"].field.sequence == 1)
        assert recovered["updates"] > stale["updates"]
    finally:
        _close(attachment)


def test_missing_directory_can_appear_after_attachment_starts(tmp_path):
    output = tmp_path / "late" / "out"
    attachment = _attach(output).start()
    try:
        _wait_for(attachment, lambda s: s["state"] == "waiting")
        _frame(output, 0)
        current = _wait_for(attachment, lambda s: s["state"] == "current")
        assert current["attachment"]["directory"] == str(output.resolve())
        assert current["view"].field.sequence == 0
    finally:
        _close(attachment)


def test_close_and_switch_keeps_second_attachment_isolated(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    _frame(first_dir, 0, phase=0.1)
    first = _attach(first_dir).start()
    try:
        first_current = _wait_for(first, lambda s: s["state"] == "current")
        assert first_current["view"].field.segment_id
    finally:
        _close(first)
    assert first.snapshot()["state"] == "closed"

    _frame(second_dir, 0, phase=1.1)
    second = _attach(second_dir).start()
    try:
        switched = _wait_for(second, lambda s: s["state"] == "current")
        assert switched["attachment"]["directory"] == str(second_dir.resolve())
        assert switched["view"].field.segment_id != first_current["view"].field.segment_id
        assert switched["view"].field.vectors[0, 0, 1] == pytest.approx(np.sin(1.1))
        _frame(first_dir, 1, phase=0.9)
        time.sleep(0.05)
        assert first.snapshot()["state"] == "closed"
        assert switched["view"].field.sequence == 0
    finally:
        _close(second)


def test_large_preexisting_series_starts_from_recent_tail_without_decoding_old_files(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    for sequence in range(4103):
        (output / f"m{sequence:05d}.ovf").touch()
    _frame(output, 4103, phase=0.3)
    _frame(output, 4104, phase=0.4)
    attachment = _attach(output).start()
    try:
        current = _wait_for(attachment, lambda s: s["state"] == "current")
        assert current["attachment"]["published_frames"] == 2
        assert current["view"].field.sequence == 4104
        assert current["view"].field.sim_time_s == pytest.approx(4.105e-9)
    finally:
        _close(attachment)


def test_publication_limit_rotates_observation_segment_without_touching_source(tmp_path, monkeypatch):
    from mumax_sonic.sources import mumax3_bridge
    monkeypatch.setattr(mumax3_bridge, 'MAX_FRAMES', 4)
    output = tmp_path / 'out'
    _frame(output, 0)
    _frame(output, 1, phase=.1)
    attachment = _attach(output).start()
    try:
        initial = _wait_for(attachment, lambda s: s['state'] == 'current')
        segment = initial['view'].field.segment_id
        original = (output / 'm00000.ovf').read_bytes()
        for sequence in range(2, 5):
            _frame(output, sequence, phase=.1 * sequence)
        rotated = _wait_for(attachment, lambda s: s['state'] == 'current'
            and s['view'].field.sequence == 4 and s['view'].field.segment_id != segment)
        assert rotated['attachment']['published_frames'] == 2
        assert (output / 'm00000.ovf').read_bytes() == original
        assert len(list(output.glob('*.ovf'))) == 5
    finally:
        _close(attachment)
