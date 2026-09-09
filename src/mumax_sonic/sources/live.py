"""Background follower for atomically published, hash-bound OVF manifests.

The follower deliberately has no audio-thread responsibilities.  A worker
reloads the complete manifest through :func:`load_field_replay`; snapshots are
small references protected by a short lock.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from math import isfinite
from pathlib import Path
from threading import Event, Lock, Thread
import time

from ..field_pipeline import FieldView, observe_field
from .ovf_replay import load_field_replay
from .incremental import IncrementalOVFReader


@dataclass(frozen=True)
class LiveConfig:
    recipe: str = 'topology'
    method: str = 'solid_angle'
    max_dt_s: float | None = None
    band_config: object | None = None
    poll_interval_s: float = .1
    stale_after_s: float = 2.0

    def __post_init__(self):
        if not all(isfinite(v) and v > 0 for v in (self.poll_interval_s, self.stale_after_s)):
            raise ValueError('poll_interval_s and stale_after_s must be finite and positive')
        if self.recipe not in {'topology', 'activity', 'band'}:
            raise ValueError('unsupported live recipe')


class LiveFollower:
    """Follow one published replay manifest without ever guessing source files.

    A published record is immutable within a segment.  Reusing its sequence
    and physical time with different raw OVF lineage fails closed, preserving
    the last good view for callers that want to show it while muting output.
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
        self._thread: Thread | None = None
        self._view: FieldView | None = None
        self._state = 'waiting'
        self._reason = 'waiting for published manifest'
        self._last_error: str | None = None
        self._processing_ms: float | None = None
        self._updates = 0
        self._skipped_observation_frames = 0
        self._last_new_at: float | None = None
        self._waiting_since = time.monotonic()
        self._last_stamp = None
        self._last_manifest_hash: str | None = None
        self._known: dict[tuple[str, int, float], str] = {}
        self._sequence_times: dict[tuple[str, int], float] = {}
        self._max_sequence: dict[str, int] = {}

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
        with self._lock:
            self._state = 'closed'
            self._reason = 'closed'

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
                        processing_ms=self._processing_ms, reader=dict(self._reader_diagnostics))

    def _run(self):
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                stamp = self._stamp()
                if stamp is None:
                    self._publish_waiting(started, 'manifest is missing')
                elif stamp != self._last_stamp or self._state == 'invalid':
                    self._last_stamp = stamp
                    self._reload(started)
                else:
                    self._refresh_staleness(started)
            except Exception as exc:  # unexpected worker errors remain fail-closed
                self._publish_invalid(started, f'{type(exc).__name__}: {exc}')
            self._stop.wait(self.config.poll_interval_s)

    def _stamp(self):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return (stat.st_mtime_ns, stat.st_size, getattr(stat, 'st_ino', None))

    def _reload(self, started):
        try:
            replay = self._reader.load(self.path)
            manifest_hash = replay.sha256
            if manifest_hash == self._last_manifest_hash:
                # A recovered valid publication may have exactly the same
                # bytes as the last accepted manifest after a half-write.
                # It clears invalid status but never resets physical freshness.
                self._republish_known_valid(started)
                return
            frames = replay.frames
            fingerprints = [((segment, sequence, sim_time), fingerprint)
                            for segment, sequence, sim_time, fingerprint in self._reader.records]
            self._validate_immutable(fingerprints)
            latest = frames[-1]
            key = self._key(replay.frames[-1])
            prior = self._known.get(key)
            # A re-published manifest with no new physical record only updates
            # metadata; it must not reset the stale timer or recompute physics.
            is_new = prior is None
            if is_new:
                previous = frames[-2] if len(frames) > 1 else None
                view = observe_field(latest, self.config.recipe, method=self.config.method,
                    previous=previous, max_dt_s=self.config.max_dt_s, history=frames,
                    band_config=self.config.band_config)
                view = replace(view, field=replace(latest, source_kind='live'),
                    sample=replace(view.sample, source_kind='live'), diagnostic=dict(view.diagnostic, source_kind='live'))
            else:
                view = self._view
            self._commit_valid(started, manifest_hash, fingerprints, view, is_new)
            with self._lock:
                self._reader_diagnostics = self._reader.diagnostics
        except Exception as exc:
            self._publish_invalid(started, str(exc))

    @staticmethod
    def _key(frame):
        return (frame.segment_id, frame.sequence, frame.sim_time_s)

    @staticmethod
    def _raw_fingerprint(frame):
        """Stable raw-frame lineage, deliberately excluding manifest publication hash."""
        try:
            data = json.loads(frame.provenance)
        except (TypeError, ValueError, json.JSONDecodeError):
            return hashlib.sha256(frame.provenance.encode('utf-8')).hexdigest()
        if isinstance(data, dict) and data.get('format') == 'OVF2':
            # Publication metadata (including time-evidence prose and the
            # manifest digest) may legitimately change on append.  Bind only
            # the raw bytes and physical identity/geometry of this record.
            raw = dict(
                ovf_sha256=data.get('ovf_sha256'),
                quantity=data.get('quantity'), input_unit=data.get('input_unit'), origin=data.get('origin'),
                mask_source=data.get('mask_source'),
                components=data.get('components'), labels=data.get('labels'),
                z_index=data.get('z_index'), mesh_shape_zyx=data.get('mesh_shape_zyx'),
                step_m=data.get('step_m'), entity_id=frame.entity_id,
                segment_id=frame.segment_id, sequence=frame.sequence,
                sim_time_s=frame.sim_time_s, time_kind=frame.time_kind,
                dx_m=frame.dx_m, dy_m=frame.dy_m, origin_m=frame.origin_m,
            )
            return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return hashlib.sha256(frame.provenance.encode('utf-8')).hexdigest()

    def _validate_immutable(self, fingerprints):
        # A segment publication is append-only, not a sliding window.  Keep
        # explicit full history so sampling gaps cannot be silently concealed.
        segment = fingerprints[-1][0][0]
        published = {key for key, _ in fingerprints}
        if any(key[0] == segment and key not in published for key in self._known):
            raise ValueError('published history was truncated within a segment')
        by_segment: dict[str, list[tuple[tuple[str, int, float], str]]] = {}
        for key, fingerprint in fingerprints:
            by_segment.setdefault(key[0], []).append((key, fingerprint))
            prior_time = self._sequence_times.get((key[0], key[1]))
            if prior_time is not None and prior_time != key[2]:
                raise ValueError('published sequence was reused with a different physical time')
            prior = self._known.get(key)
            if prior is not None and prior != fingerprint:
                raise ValueError('published physical frame provenance changed')
        for segment, entries in by_segment.items():
            maximum = self._max_sequence.get(segment)
            if maximum is not None and max(key[1] for key, _ in entries) < maximum:
                raise ValueError('published source regressed within an existing segment')

    def _commit_valid(self, started, manifest_hash, fingerprints, view, is_new):
        now = time.monotonic()
        with self._lock:
            if self._state == 'closed':
                return
            for key, fingerprint in fingerprints:
                self._known[key] = fingerprint
                self._sequence_times[(key[0], key[1])] = key[2]
                self._max_sequence[key[0]] = max(self._max_sequence.get(key[0], -1), key[1])
            self._last_manifest_hash = manifest_hash
            if is_new:
                if self._view is not None and view.field.segment_id == self._view.field.segment_id:
                    self._skipped_observation_frames += sum(
                        self._view.field.sequence < key[1] < view.field.sequence
                        for key, _ in fingerprints)
                self._view = view
                self._updates += 1
                self._last_new_at = started  # include read/observer latency in freshness
                # Bound lineage retention to the current segment (4096 frames
                # at most, enforced by the manifest reader).
                segment = fingerprints[-1][0][0]
                self._known = {k: v for k, v in self._known.items() if k[0] == segment}
                self._sequence_times = {k: v for k, v in self._sequence_times.items() if k[0] == segment}
                self._max_sequence = {k: v for k, v in self._max_sequence.items() if k == segment}
            self._state = 'current' if self._last_new_at is not None else 'waiting'
            self._reason = 'new physical frame observed' if is_new else 'published manifest has no new physical frame'
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

    def _republish_known_valid(self, started):
        now = time.monotonic()
        with self._lock:
            if self._state == 'closed':
                return
            if self._last_new_at is not None and now - self._last_new_at >= self.config.stale_after_s:
                self._state = 'stale'
                self._reason = 'no new physical frame before stale deadline'
            else:
                self._state = 'current' if self._last_new_at is not None else 'waiting'
                self._reason = 'published manifest has no new physical frame'
            self._last_error = None
            self._processing_ms = (now - started) * 1000.0

    def _publish_invalid(self, started, error):
        with self._lock:
            if self._state == 'closed':
                return
            self._state = 'invalid'
            self._reason = 'published manifest is incomplete or invalid'
            self._last_error = error
            self._processing_ms = (time.monotonic() - started) * 1000.0
