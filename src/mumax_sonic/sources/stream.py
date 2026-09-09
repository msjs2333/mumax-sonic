"""Bounded host-frame ingestion; solver calls stay on the producer's thread."""
from collections import deque
from dataclasses import replace
from threading import Condition, Thread
import time

from ..fields import FieldFrame
from ..field_pipeline import observe_field
from .live import LiveConfig


class FrameStream:
    def __init__(self, config=LiveConfig(), *, max_pending=8, max_bytes=256*1024*1024):
        if type(max_pending) is not int or max_pending < 1 or type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('positive queue and memory limits required')
        self.config = config
        self.max_pending, self.max_bytes = max_pending, max_bytes
        self._condition = Condition()
        self._pending = deque()
        self._thread = None
        self._closed = False
        self._finished = False
        self._processing = False
        self._generation = 0
        self._identity = None
        self._last_sequence = self._last_time = None
        self._view = None
        self._received_at = None
        self._state, self._reason = 'waiting', 'waiting for captured host frame'
        self._updates = self._dropped = self._skipped = 0
        self._processing_ms = None
        self._last_error = None
        self._history_limit = (config.band_config.window_samples if config.band_config is not None else 256) if config.recipe == 'band' else 2

    def start(self):
        with self._condition:
            if self._closed:
                raise RuntimeError('stream is closed')
            if self._thread is None:
                self._thread = Thread(target=self._run, name='mumax-host-observer', daemon=True)
                self._thread.start()
        return self

    def submit(self, frame):
        """Enqueue an immutable host frame, without waiting for its observation."""
        if not isinstance(frame, FieldFrame):
            raise TypeError('submit requires FieldFrame')
        cost = frame.vectors.nbytes + frame.mask.nbytes
        # Allow one worker batch, one pending batch, retained history and view.
        if cost * (2*self.max_pending + self._history_limit + 2) > self.max_bytes:
            raise ValueError('configured history and queue exceed host memory budget')
        identity = (frame.entity_id, frame.segment_id)
        with self._condition:
            if self._closed:
                raise RuntimeError('stream is closed')
            if self._finished:
                raise RuntimeError('stream is finished')
            if identity != self._identity:
                self._generation += 1
                self._pending.clear()
                self._identity = identity
                self._last_sequence = self._last_time = None
                self._view, self._received_at = None, None
                self._state, self._reason = 'waiting', 'new entity or segment'
            if self._last_sequence is not None and (frame.sequence <= self._last_sequence or frame.sim_time_s <= self._last_time):
                raise ValueError('sequence and physical time must increase; reset with a new segment')
            self._last_sequence, self._last_time = frame.sequence, frame.sim_time_s
            if len(self._pending) == self.max_pending:
                self._pending.popleft()
                self._dropped += 1
            self._pending.append((frame, time.monotonic(), self._generation))
            self._condition.notify()

    def finish(self):
        """Declare normal producer completion; retain and drain accepted frames."""
        with self._condition:
            self._finished = True

    def fail(self, error):
        """Expose a producer/solver failure and invalidate queued output."""
        with self._condition:
            if self._closed:
                return
            self._generation += 1
            self._pending.clear()
            self._state, self._reason, self._last_error = 'invalid', 'capture failed', str(error)

    def snapshot(self):
        with self._condition:
            age = None if self._received_at is None else time.monotonic()-self._received_at
            state, reason = self._state, self._reason
            if state == 'current' and age is not None and age >= self.config.stale_after_s:
                state, reason = 'stale', 'captured result exceeded freshness deadline'
            if self._finished and not self._pending and not self._processing and state in ('waiting', 'current', 'stale'):
                state, reason = 'finished', '采样已结束，物理声源静音；保留最后一帧供查看'
            return dict(view=self._view, state=state, reason=reason, age_s=age, updates=self._updates,
                dropped_frames=self._dropped, skipped_observation_frames=self._skipped,
                pending_frames=len(self._pending), processing_ms=self._processing_ms, last_error=self._last_error)

    def close(self):
        with self._condition:
            self._closed = True
            self._pending.clear()
            self._state, self._reason = 'closed', 'closed'
            self._condition.notify_all()

    def _run(self):
        history = deque(maxlen=self._history_limit)
        generation = None
        while True:
            with self._condition:
                while not self._closed and not self._pending:
                    self._condition.wait()
                if self._closed:
                    return
                batch = tuple(self._pending)
                self._pending.clear()
                self._processing = True
            started = time.monotonic()
            frame, received, job_generation = batch[-1]
            if job_generation != generation:
                history.clear()
                generation = job_generation
            history.extend(item if item.source_kind == 'live' else replace(item, source_kind='live') for item, _, _ in batch)
            try:
                frames = tuple(history)
                view = observe_field(frames[-1], self.config.recipe, method=self.config.method,
                    previous=frames[-2] if len(frames)>1 else None, max_dt_s=self.config.max_dt_s,
                    history=frames, band_config=self.config.band_config)
                error = None
            except Exception as exc:
                view, error = None, str(exc)
            with self._condition:
                self._processing = False
                if self._closed:
                    return
                if job_generation != self._generation:
                    continue
                self._processing_ms = (time.monotonic()-started)*1000
                self._skipped += len(batch)-1
                self._last_error = error
                if error:
                    self._state, self._reason = 'invalid', 'observer failed'
                else:
                    self._view, self._received_at = view, received
                    self._updates += 1
                    self._state, self._reason = 'current', 'direct host frame observed'
