"""Threaded reliability coverage for direct host-frame ingestion."""
import threading
import time

import pytest

from mumax_sonic.observers.band import BandConfig
from mumax_sonic.sources.activity_demo import make_activity_frame
from mumax_sonic.sources.band_demo import make_band_frame
from mumax_sonic.sources.live import LiveConfig
from mumax_sonic.sources.stream import FrameStream


def _wait(stream, predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = stream.snapshot()
        if predicate(snapshot):
            return snapshot
        time.sleep(0.005)
    pytest.fail(f"stream did not reach requested state: {stream.snapshot()}")


def test_bounded_queue_drops_oldest_pending_frames_while_observer_is_busy(monkeypatch):
    """A slow observer cannot turn host ingestion into an unbounded backlog."""
    import mumax_sonic.sources.stream as stream_module

    entered, release = threading.Event(), threading.Event()
    observed = []
    original = stream_module.observe_field

    def blocked_observe(frame, *args, **kwargs):
        observed.append(frame.sequence)
        entered.set()
        assert release.wait(2)
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(stream_module, "observe_field", blocked_observe)
    stream = FrameStream(LiveConfig(recipe="activity"), max_pending=2)
    stream.start()
    try:
        stream.submit(make_activity_frame("activity_rotation", 0))
        assert entered.wait(1)
        for index in range(1, 7):
            stream.submit(make_activity_frame("activity_rotation", index))
            assert stream.snapshot()["pending_frames"] <= 2
        overloaded = stream.snapshot()
        assert overloaded["pending_frames"] == 2
        assert overloaded["dropped_frames"] == 4
        release.set()
        result = _wait(stream, lambda state: state["updates"] == 2)
        assert observed == [0, 6]
        assert result["skipped_observation_frames"] == 1
        assert result["view"].sample.sequence == 6
    finally:
        release.set()
        stream.close()


def test_blocked_observer_becomes_stale_and_a_later_frame_recovers(monkeypatch):
    """Freshness uses arrival time, including observer latency, then recovers."""
    import mumax_sonic.sources.stream as stream_module

    entered, release = threading.Event(), threading.Event()
    original = stream_module.observe_field
    calls = 0

    def block_second_observation(frame, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            assert release.wait(2)
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(stream_module, "observe_field", block_second_observation)
    config = LiveConfig(recipe="activity", stale_after_s=0.5)
    stream = FrameStream(config).start()
    try:
        stream.submit(make_activity_frame("activity_rotation", 0))
        _wait(stream, lambda state: state["updates"] == 1)
        stream.submit(make_activity_frame("activity_rotation", 1))
        assert entered.wait(1)
        stale = _wait(stream, lambda state: state["state"] == "stale")
        assert stale["view"].sample.sequence == 0
        assert stale["age_s"] >= config.stale_after_s
        release.set()
        delayed = _wait(stream, lambda state: state["updates"] == 2)
        assert delayed["state"] == "stale"
        assert delayed["view"].sample.sequence == 1
        stream.submit(make_activity_frame("activity_rotation", 2))
        recovered = _wait(stream, lambda state: state["state"] == "current" and state["updates"] == 3)
        assert recovered["view"].sample.sequence == 2
        assert recovered["age_s"] < config.stale_after_s
    finally:
        release.set()
        stream.close()


def test_band_gap_restarts_physical_window_without_synthesizing_missing_frames():
    config = LiveConfig(recipe="band", band_config=BandConfig(8e9, 30e9, window_samples=16))
    stream = FrameStream(config, max_pending=32).start()
    try:
        for index in range(16):
            stream.submit(make_band_frame("band_in", index))
        ready = _wait(stream, lambda state: state["view"] is not None and state["view"].sample.validity == "valid")
        assert ready["view"].sample.sequence == 15

        stream.submit(make_band_frame("band_in", 17))
        stream.submit(make_band_frame("band_in", 18))
        gap = _wait(stream, lambda state: state["view"] is not None and state["view"].sample.sequence == 18)
        assert gap["view"].sample.validity == "warming_up"
        assert gap["view"].diagnostic["samples_available"] == 2
        assert gap["view"].diagnostic["samples_required"] == 16

        for index in range(19, 33):
            stream.submit(make_band_frame("band_in", index))
        rewarmed = _wait(stream, lambda state: state["view"] is not None and state["view"].sample.sequence == 32)
        assert rewarmed["view"].sample.validity == "valid"
        assert rewarmed["view"].diagnostic["samples_available"] == 16
    finally:
        stream.close()


def test_concurrent_finish_and_fail_leave_invalid_state_and_discard_worker_output(monkeypatch):
    """A producer failure wins even when normal completion is declared concurrently."""
    import mumax_sonic.sources.stream as stream_module

    entered, release, begin = threading.Event(), threading.Event(), threading.Event()
    original = stream_module.observe_field

    def blocked_observe(frame, *args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(stream_module, "observe_field", blocked_observe)
    stream = FrameStream(LiveConfig(recipe="activity")).start()
    try:
        stream.submit(make_activity_frame("activity_rotation", 0))
        assert entered.wait(1)
        finish_thread = threading.Thread(target=lambda: (begin.wait(), stream.finish()))
        fail_thread = threading.Thread(target=lambda: (begin.wait(), stream.fail("backend disconnected")))
        finish_thread.start()
        fail_thread.start()
        begin.set()
        finish_thread.join(1)
        fail_thread.join(1)
        assert not finish_thread.is_alive()
        assert not fail_thread.is_alive()
        release.set()
        invalid = _wait(stream, lambda state: state["state"] == "invalid")
        assert invalid["last_error"] == "backend disconnected"
        assert invalid["view"] is None
        assert invalid["pending_frames"] == 0
    finally:
        release.set()
        stream.close()
