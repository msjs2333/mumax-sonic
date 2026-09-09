"""Background follower for atomically published, OVF manifests.

The follower deliberately has no audio-thread responsibilities.  A worker
validates the manifest and incrementally decodes new OVF frames; snapshots are
small references protected by a short lock.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from threading import Event, Lock, Thread
import time

from ..field_pipeline import FieldView, observe_field
from .incremental import IncrementalOVFReader


@dataclass(frozen=True)
class LiveConfig:
    recipe: str = 'topology'
    method: str = 'solid_angle'
    max_dt_s: float | None = None
    band_config: object | None = None
    poll_interval_s: float = .05
    stale_after_s: float = 2.0

    def __post_init__(self):
        if not all(isfinite(v) and v > 0 for v in (self.poll_interval_s, self.stale_after_s)):
            raise ValueError('poll_interval_s and stale_after_s must be finite and positive')
        if self.recipe not in {'topology', 'activity', 'band'}:
            raise ValueError('unsupported live recipe')


class LiveFollower:
    """Follow one published replay manifest without ever guessing source files.

    Accepted history is trusted and never reread. New source segments must
    use a new segment ID; only new frames advance physical freshness.
    """

    def __init__(self, path, config: LiveConfig = LiveConfig()):
        self.path = Path(path)
        if not isinstance(config, LiveConfig):
            raise TypeError('config must be a LiveConfig')
        self.config = config
        retain = (config.band_config.window_samples if config.band_config is not None else 256) if config.recipe == 'band' else 2
        self._reader = IncrementalOVFReader(retain_frames=retain)
        self._reader_diagnostics = {}
        self._lock = Lock()
        self._stop = Event()
        self._wake = Event()
        self._thread: Thread | None = None
        self._view: FieldView | None = None
        self._state = 'waiting'
        self._reason = 'waiting for published manifest'
        self._last_error: str | None = None
        self._processing_ms: float | None = None
        self._processing_breakdown_ms = {}
        self._attention = None
        self._focus_enabled = False
        self._attention_revision = 0
        self._applied_attention_revision = 0
        self._focus = None
        self._updates = 0
        self._skipped_observation_frames = 0
        self._last_new_at: float | None = None
        self._waiting_since = time.monotonic()
        self._last_stamp = None
        self._frames = ()
        self._records = ()
        self._job_id = 0
        self._last_job = None

    def start(self):
        with self._lock:
            if self._state == 'closed':
                raise RuntimeError('LiveFollower is closed')
            if self._thread is None:
                self._thread = Thread(target=self._run, name='mumax-sonic-live', daemon=True)
                self._thread.start()
        return self

    def close(self):
        self._stop.set()
        self._wake.set()
        with self._lock:
            self._state = 'closed'
            self._reason = 'closed'
        if self._focus is not None:
            self._focus.close()

    def set_attention(self, attention, enabled=False):
        enabled = bool(enabled) and self.config.recipe == 'activity'
        with self._lock:
            if self._state == 'closed':
                return
            if (enabled and attention != self._attention) or enabled != self._focus_enabled:
                self._attention, self._focus_enabled = attention, enabled
                self._attention_revision += 1
                self._wake.set()

    def snapshot(self) -> dict:
        """Return the current worker result; this does no I/O or array copies."""
        now = time.monotonic()
        with self._lock:
            age = None if self._last_new_at is None else max(0.0, now - self._last_new_at)
            state, reason = self._state, self._reason
            if state == 'current' and age is not None and age >= self.config.stale_after_s:
                state, reason = 'stale', 'no new physical frame before stale deadline'
            return dict(view=self._view, state=state, reason=reason,
                        age_s=age, updates=self._updates, skipped_observation_frames=self._skipped_observation_frames, last_error=self._last_error,
                        processing_ms=self._processing_ms, reader=dict(self._reader_diagnostics),
                        processing_breakdown_ms=dict(self._processing_breakdown_ms),
                        last_job=None if self._last_job is None else dict(
                            self._last_job, timings_ms=dict(self._last_job['timings_ms'])) )

    def _run(self):
        while not self._stop.is_set():
            self._wake.clear()
            started = time.monotonic()
            try:
                stamp = self._stamp()
                if stamp is None:
                    self._publish_waiting(started, 'manifest is missing')
                else:
                    with self._lock:
                        changed = stamp != self._last_stamp
                        invalid = self._state == 'invalid'
                        attention_changed = self._attention_revision != self._applied_attention_revision
                        eligible_roi = self._state in ('current', 'stale') and attention_changed and bool(self._frames)
                    if changed or invalid or eligible_roi:
                        if changed:
                            self._last_stamp = stamp
                        self._reload(started, roi_only=eligible_roi and not changed and not invalid)
                    else:
                        self._refresh_staleness(started)
            except Exception as exc:  # unexpected worker errors remain fail-closed
                self._publish_invalid(started, f'{type(exc).__name__}: {exc}')
            self._wake.wait(self.config.poll_interval_s)

    def _stamp(self):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return (stat.st_mtime_ns, stat.st_size, getattr(stat, 'st_ino', None))

    def _reload(self, started, roi_only=False):
        timings = {}
        stage_started = time.monotonic()
        with self._lock:
            attention, focused, revision = self._attention, self._focus_enabled, self._attention_revision
            cached_frames, cached_records = self._frames, self._records
        kind = 'roi' if roi_only else 'manifest'
        job_sequence = None
        try:
            if roi_only:
                frames, records = cached_frames, cached_records
            else:
                replay = self._reader.load(self.path)
                frames, records = replay.frames, self._reader.records
                with self._lock:
                    self._frames, self._records = frames, records
            timings['read_validate'] = (time.monotonic()-stage_started)*1000
            stage_started = time.monotonic()
            latest = frames[-1]
            job_sequence = latest.sequence
            key = self._key(latest)
            # A re-published manifest with no new physical record only updates
            # metadata; it must not reset the stale timer or recompute physics.
            is_new = self._view is None or key != self._key(self._view.field)
            timings['frame_identity'] = (time.monotonic()-stage_started)*1000
            stage_started = time.monotonic()
            if is_new or revision != self._applied_attention_revision:
                previous = frames[-2] if len(frames) > 1 else None
                if focused:
                    if self._focus is None:
                        from ..observers.focused_activity import FocusedActivity
                        with self._lock:
                            if self._state == 'closed':
                                return
                            self._focus = FocusedActivity()
                    view = self._focus.observe(previous, latest, attention, max_dt_s=self.config.max_dt_s)
                else:
                    view = observe_field(latest, self.config.recipe, method=self.config.method,
                        previous=previous, max_dt_s=self.config.max_dt_s, history=frames,
                        band_config=self.config.band_config)
                view = replace(view, field=latest.with_source_kind('live'),
                    sample=replace(view.sample, source_kind='live'), diagnostic=dict(view.diagnostic, source_kind='live'))
            else:
                view = self._view
            timings['observe'] = (time.monotonic()-stage_started)*1000
            stage_started = time.monotonic()
            self._commit_valid(started, records, view, is_new)
            timings['commit'] = (time.monotonic()-stage_started)*1000
            with self._lock:
                self._reader_diagnostics = self._reader.diagnostics
                self._applied_attention_revision = revision
        except Exception as exc:
            self._publish_invalid(started, str(exc))
        finally:
            with self._lock:
                self._processing_breakdown_ms = timings
                self._job_id += 1
                self._last_job = dict(job_id=self._job_id, sequence=job_sequence, kind=kind,
                                      started_at_s=started, completed_at_s=time.monotonic(),
                                      timings_ms=dict(timings), state=self._state)

    @staticmethod
    def _key(frame):
        return (frame.segment_id, frame.sequence, frame.sim_time_s)

    def _commit_valid(self, started, records, view, is_new):
        now = time.monotonic()
        with self._lock:
            if self._state == 'closed':
                return
            if is_new:
                if self._view is not None and view.field.segment_id == self._view.field.segment_id:
                    self._skipped_observation_frames += sum(
                        self._view.field.sequence < sequence < view.field.sequence
                        for _, sequence, _, _ in records)
                self._view = view
                self._updates += 1
                self._last_new_at = started  # include read/observer latency in freshness
            self._view = view
            if is_new:
                self._state = 'current' if self._last_new_at is not None else 'waiting'
                self._reason = 'new physical frame observed'
            elif (self._last_new_at is not None and
                  now - self._last_new_at >= self.config.stale_after_s):
                self._state = 'stale'
                self._reason = 'attention region recomputed; no new physical frame'
            elif self._state not in ('stale', 'closed'):
                self._state = 'current' if self._last_new_at is not None else 'waiting'
                self._reason = 'published manifest has no new physical frame'
            else:
                self._reason = 'attention region recomputed; no new physical frame'
            self._last_error = None
            self._processing_ms = (now - started) * 1000.0

    def _publish_waiting(self, started, reason):
        with self._lock:
            if self._state == 'closed':
                return
            elapsed = time.monotonic() - self._waiting_since
            self._state = 'stale' if elapsed >= self.config.stale_after_s else 'waiting'
            self._reason = reason if self._state == 'waiting' else f'{reason}; no physical frame before stale deadline'
            self._last_error = None
            self._processing_ms = (time.monotonic() - started) * 1000.0

    def _refresh_staleness(self, now):
        with self._lock:
            if self._state == 'closed':
                return
            if self._state != 'invalid' and self._last_new_at is not None and now - self._last_new_at >= self.config.stale_after_s:
                self._state = 'stale'
                self._reason = 'no new physical frame before stale deadline'

    def _publish_invalid(self, started, error):
        with self._lock:
            if self._state == 'closed':
                return
            self._state = 'invalid'
            self._reason = 'published manifest is incomplete or invalid'
            self._last_error = error
            self._processing_ms = (time.monotonic() - started) * 1000.0
