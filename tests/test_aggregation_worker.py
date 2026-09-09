import threading
import time

from mumax_sonic.ui.aggregation_worker import LatestAggregation
import mumax_sonic.ui.aggregation_worker as worker_module


def _poll_until(worker, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = worker.poll()
        if result is not None:
            return result
        time.sleep(.005)
    raise AssertionError("worker did not publish a result")


def test_request_is_nonblocking_and_replaces_the_single_pending_job(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    last_entered = threading.Event()
    release_last = threading.Event()
    calls = []

    def slow(view, attention, budget, mode):
        calls.append(view)
        if view == "first":
            entered.set()
            assert release.wait(1)
        if view == "last":
            last_entered.set()
            assert release_last.wait(1)
        return (view, attention, budget, mode)

    monkeypatch.setattr(worker_module, "apply_aggregation", slow)
    worker = LatestAggregation()
    started = time.monotonic()
    assert worker.request("one", "first", "a", 2)
    assert time.monotonic() - started < .1
    assert entered.wait(1)
    assert worker.request("two", "middle", "b", 3)
    assert not worker.request("two", "middle", "b", 3)
    assert worker.request("three", "last", "c", 4)
    release.set()
    assert last_entered.wait(1)
    assert _poll_until(worker) == ("one", ("first", "a", 2, "adaptive"), None)
    release_last.set()
    assert _poll_until(worker) == ("three", ("last", "c", 4, "adaptive"), None)
    assert calls == ["first", "last"]
    worker.close()


def test_close_discards_pending_and_late_running_result(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def slow(view, attention, budget, mode):
        entered.set()
        assert release.wait(1)
        return view

    monkeypatch.setattr(worker_module, "apply_aggregation", slow)
    worker = LatestAggregation()
    assert worker.request("running", "running", None, 1)
    assert entered.wait(1)
    assert worker.request("pending", "pending", None, 1)
    worker.close()
    release.set()
    time.sleep(.03)
    assert worker.poll() is None
    assert not worker.request("after-close", "ignored", None, 1)


def test_returning_to_the_running_key_discards_an_obsolete_pending_roi(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def slow(view, attention, budget, mode):
        calls.append(view)
        entered.set()
        assert release.wait(1)
        return view

    monkeypatch.setattr(worker_module, "apply_aggregation", slow)
    worker = LatestAggregation()
    assert worker.request("roi-a", "a", None, 1)
    assert entered.wait(1)
    assert worker.request("roi-b", "b", None, 1)
    assert worker.request("roi-a", "a", None, 1)
    release.set()
    assert _poll_until(worker) == ("roi-a", "a", None)
    time.sleep(.03)
    assert calls == ["a"]
    worker.close()


def test_error_is_reported_and_a_later_job_recovers(monkeypatch):
    def flaky(view, attention, budget, mode):
        if view == "bad":
            raise ValueError("bad aggregation")
        return view

    monkeypatch.setattr(worker_module, "apply_aggregation", flaky)
    worker = LatestAggregation()
    assert worker.request("bad", "bad", None, 1)
    key, result, error = _poll_until(worker)
    assert (key, result) == ("bad", None)
    assert isinstance(error, ValueError)
    assert worker.request("good", "good", None, 1)
    assert _poll_until(worker) == ("good", "good", None)
    assert not worker.request("good", "good", None, 1)
    worker.close()
