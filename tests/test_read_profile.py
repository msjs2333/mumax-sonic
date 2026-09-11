"""Contract tests for the local read-cost profiling helper."""
import importlib.util
import sys
import threading
from pathlib import Path

import numpy as np

from mumax_sonic.sources.ovf import read_ovf
from test_ovf_replay import manifest


def _profile_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "profile_read_costs_test_module", root / "scripts" / "profile_read_costs.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_decode_last_manifest_frame_only_decodes_last_and_preserves_values(tmp_path):
    profile = _profile_module()
    path, meta = manifest(tmp_path, count=3)
    collector = profile.ProfileCollector()

    with collector.instrument():
        frame, record = profile.decode_last_manifest_frame(path, collector)

    expected = read_ovf(tmp_path / record["file"], hash_content=False).vectors[0]
    np.testing.assert_array_equal(frame.vectors, expected)
    assert record["sequence"] == meta["frames"][-1]["sequence"]
    decoded = [sample for sample in collector.snapshot() if sample.name == "read_ovf"]
    assert len(decoded) == 1
    assert decoded[0].path.endswith(record["file"])
    assert all(record["file"] in (sample.path or "") for sample in decoded)


def test_instrument_restores_patched_symbols_after_context(tmp_path):
    profile = _profile_module()
    path, _ = manifest(tmp_path, count=1)
    collector = profile.ProfileCollector()
    original_read = profile.ovf_replay.read_ovf
    original_open = Path.open

    with collector.instrument():
        assert profile.ovf_replay.read_ovf is not original_read
        assert Path.open is not original_open
        profile.decode_last_manifest_frame(path, collector)

    assert profile.ovf_replay.read_ovf is original_read
    assert Path.open is original_open


def test_profile_samples_are_labelled_with_worker_thread(tmp_path):
    profile = _profile_module()
    path, _ = manifest(tmp_path, count=1)
    collector = profile.ProfileCollector()
    errors = []

    def worker():
        try:
            with collector.instrument():
                profile.decode_last_manifest_frame(path, collector)
        except Exception as exc:  # surface worker failures in the main thread
            errors.append(exc)

    thread = threading.Thread(target=worker, name="profile-worker")
    thread.start()
    thread.join()
    assert not errors
    samples = collector.snapshot()
    assert samples
    assert {sample.thread for sample in samples} == {"profile-worker"}
