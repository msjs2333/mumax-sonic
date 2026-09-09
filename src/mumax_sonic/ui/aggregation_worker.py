"""Latest-only background aggregation for the Tk event loop.

The worker owns no UI or audio objects.  It applies only the already-computed
field contribution aggregation, so the UI can continue drawing while a drag
updates the attention region.
"""
from __future__ import annotations

from threading import Condition, Thread
from typing import Any

from ..field_pipeline import apply_aggregation


_MISSING = object()


class LatestAggregation:
    """Run one aggregation at a time and retain at most one newer request.

    ``poll`` consumes the newest completed result.  A completed result remains
    available even when a newer request is waiting, letting callers decide
    whether an earlier result is still useful for their physical base view.
    """

    def __init__(self) -> None:
        self._condition = Condition()
        self._pending: tuple[Any, Any, Any, int] | None = None
        self._running_key: Any = _MISSING
        self._completed_key: Any = _MISSING
        self._result: tuple[Any, Any | None, Exception | None] | None = None
        self._closed = False
        self._thread: Thread | None = None

    def request(self, key: Any, view: Any, attention: Any, budget: int) -> bool:
        """Queue the latest request without waiting for aggregation.

        Returns ``True`` when a request was queued and ``False`` for a duplicate
        or after closure.  Only the currently running job and one replaceable
        pending job can exist.
        """
        with self._condition:
            if self._closed:
                return False
            # Returning to the view being computed (or already completed) makes
            # a different pending ROI obsolete.  Drop that pending work rather
            # than letting it overwrite the caller's newest choice.
            if key == self._running_key or key == self._completed_key:
                if self._pending is not None:
                    self._pending = None
                    return True
                return False
            if self._pending is not None and key == self._pending[0]:
                return False
            self._pending = (key, view, attention, budget)
            if self._thread is None:
                self._thread = Thread(target=self._run, name="mumax-aggregation", daemon=True)
                self._thread.start()
            self._condition.notify()
            return True

    def poll(self) -> tuple[Any, Any | None, Exception | None] | None:
        """Return and consume the newest completed job, if any."""
        with self._condition:
            if self._closed:
                return None
            result = self._result
            self._result = None
            return result

    def close(self) -> None:
        """Discard outstanding work and prevent the worker from publishing again."""
        with self._condition:
            self._closed = True
            self._pending = None
            self._result = None
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._closed and self._pending is None:
                    self._condition.wait()
                if self._closed:
                    return
                key, view, attention, budget = self._pending
                self._pending = None
                self._running_key = key
            try:
                result = apply_aggregation(view, attention, budget, "adaptive")
                error = None
            except Exception as exc:
                result = None
                error = exc
            with self._condition:
                self._running_key = _MISSING
                if not self._closed:
                    self._result = (key, result, error)
                    self._completed_key = key
