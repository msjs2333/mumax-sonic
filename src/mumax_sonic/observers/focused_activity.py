"""Low-latency focus activity with a deliberately small cached overview.

The foreground is measured from the pair the caller supplied.  A single
worker occasionally measures the complete pair and retains only 16 by 16
tile summaries; it never retains a full activity image.  This makes the
time difference between the two parts of the displayed field explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import threading
import time

import numpy as np

from ..aggregation import ContributionGrid
from ..attention import Attention
from ..field_pipeline import FieldView, observe_field
from ..fields import FieldFrame
from ..model import Observation, Sample
from .activity import angular_activity


_DIVISIONS = 16


@dataclass(frozen=True)
class _Background:
    key: tuple
    sim_time_s: float
    enqueued_s: float
    rows: tuple[np.ndarray, ...]
    cols: tuple[np.ndarray, ...]
    strengths: tuple[float, ...]
    x_m: tuple[float, ...]
    y_m: tuple[float, ...]
    material_counts: tuple[int, ...]
    coverage: float
    validity: str
    elapsed_ms: float


def _tile_indices(shape: tuple[int, int]) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    return (tuple(np.array_split(np.arange(shape[0]), _DIVISIONS)),
            tuple(np.array_split(np.arange(shape[1]), _DIVISIONS)))


def _key(frame: FieldFrame) -> tuple:
    return (frame.entity_id, frame.source_kind, frame.segment_id, frame.sequence,
            frame.sim_time_s, frame.vectors.shape, frame.dx_m, frame.dy_m, frame.origin_m)


def _crop(frame: FieldFrame, r0: int, r1: int, c0: int, c1: int) -> FieldFrame:
    return FieldFrame(frame.vectors[r0:r1, c0:c1], frame.dx_m, frame.dy_m,
                      frame.sim_time_s,
                      (frame.origin_m[0] + c0 * frame.dx_m,
                       frame.origin_m[1] + r0 * frame.dy_m, frame.origin_m[2]),
                      frame.mask[r0:r1, c0:c1], frame.entity_id, frame.source_kind,
                      frame.segment_id, frame.sequence, frame.time_kind, frame.provenance)


class FocusedActivity:
    """Observe activity immediately near ``attention`` and cache an overview.

    The cache is intentionally best effort.  ``observe`` never waits for it,
    so a returned view may describe a current foreground plus an older
    background.  Its diagnostic records that scope and age.
    """

    def __init__(self, background_interval_s: float = 2.0):
        try:
            interval = float(background_interval_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("background_interval_s must be a finite positive number") from exc
        if not isfinite(interval) or interval <= 0:
            raise ValueError("background_interval_s must be a finite positive number")
        self.background_interval_s = interval
        self._condition = threading.Condition()
        self._task: tuple[FieldFrame | None, FieldFrame, float | None, float] | None = None
        self._background: _Background | None = None
        self._background_error: str | None = None
        self._closed = False
        self._last_started_s: float | None = None
        self._thread = threading.Thread(target=self._run, name="focused-activity-background", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop accepting work and prevent an in-flight worker from publishing."""
        with self._condition:
            self._closed = True
            self._task = None
            self._condition.notify_all()
        # A caller closing a UI must not wait for an arbitrarily configured
        # cache interval.  The daemon worker observes _closed before publish.
        self._thread.join(timeout=1.0)

    def observe(self, previous: FieldFrame | None, current: FieldFrame, attention, *,
                max_dt_s: float | None = None) -> FieldView:
        """Return a current tile-aligned focus calculation without waiting."""
        if attention is None:
            attention = Attention(extent_m=current.extent_m, origin_m=current.center_m)
        if min(current.vectors.shape[:2]) < _DIVISIONS:
            # A 16-by-16 overview cannot be represented faithfully on a small
            # field.  Keep the established full-field observer contract.
            return observe_field(current, "activity", previous=previous, max_dt_s=max_dt_s)

        bounds = self._focus_bounds(current, attention)
        if bounds is None:
            return self._empty_focus(current, previous, attention, max_dt_s)
        r0, r1, c0, c1 = bounds
        if r1 - r0 < 3 or c1 - c0 < 3:
            # FieldFrame deliberately rejects sub-3-cell sections.  On a
            # minimally resolved 16-tile grid a one-tile circle therefore
            # falls back to the established full observer rather than
            # inventing padded physical samples.
            return observe_field(current, "activity", previous=previous, max_dt_s=max_dt_s)
        started = time.monotonic()
        local = observe_field(_crop(current, r0, r1, c0, c1), "activity",
                              previous=_crop(previous, r0, r1, c0, c1) if previous is not None else None,
                              max_dt_s=max_dt_s)
        focus_ms = (time.monotonic() - started) * 1000.0
        self._enqueue(previous, current, max_dt_s)
        return self._merged(current, local, bounds, focus_ms)

    def _focus_bounds(self, frame: FieldFrame, attention) -> tuple[int, int, int, int] | None:
        ny, nx = frame.vectors.shape[:2]
        rows, cols = _tile_indices((ny, nx))
        # Attention coordinates share the view convention used by aggregation.
        cx = attention.origin_m[0] + attention.center[0] * attention.extent_m
        cy = attention.origin_m[1] + attention.center[1] * attention.extent_m
        radius = attention.radius * attention.extent_m
        selected_rows = [i for i, item in enumerate(rows)
                         if item.size and frame.origin_m[1] + item[0] * frame.dy_m <= cy + radius
                         and frame.origin_m[1] + item[-1] * frame.dy_m >= cy - radius]
        selected_cols = [i for i, item in enumerate(cols)
                         if item.size and frame.origin_m[0] + item[0] * frame.dx_m <= cx + radius
                         and frame.origin_m[0] + item[-1] * frame.dx_m >= cx - radius]
        if not selected_rows or not selected_cols:
            return None
        r0, r1 = int(rows[min(selected_rows)][0]), int(rows[max(selected_rows)][-1] + 1)
        c0, c1 = int(cols[min(selected_cols)][0]), int(cols[max(selected_cols)][-1] + 1)
        return r0, r1, c0, c1

    def _enqueue(self, previous: FieldFrame | None, current: FieldFrame, max_dt_s: float | None) -> None:
        with self._condition:
            if not self._closed:
                # One slot: retaining only the newest pair makes completed old
                # segments incapable of replacing a current cache.
                self._task = (previous, current, max_dt_s, time.monotonic())
                self._condition.notify()

    @staticmethod
    def _compatible(frame: FieldFrame, newer: FieldFrame) -> bool:
        """Permit an older overview only within one physical field geometry."""
        return _key(frame)[:3] == _key(newer)[:3] and _key(frame)[5:] == _key(newer)[5:]

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._closed and self._task is None:
                    self._condition.wait()
                if self._closed:
                    return
                if self._last_started_s is not None:
                    remaining = self.background_interval_s - (time.monotonic() - self._last_started_s)
                    if remaining > 0:
                        self._condition.wait(timeout=remaining)
                        continue
                task = self._task
                self._task = None
                self._last_started_s = time.monotonic()
            assert task is not None
            previous, current, max_dt_s, enqueued_s = task
            began = time.monotonic()
            try:
                result = angular_activity(previous, current, max_dt_s=max_dt_s)
                background = self._summarize(current, result, enqueued_s, began)
            except Exception as exc:  # worker must remain usable after one bad pair
                with self._condition:
                    if not self._closed:
                        self._background_error = f"{type(exc).__name__}: {exc}"
                continue
            with self._condition:
                if self._closed:
                    return
                # A queued same-field pair must not starve this worker: cached
                # background is intentionally allowed to be temporally older.
                # Segment or geometry changes remain a hard publication fence.
                if self._task is None or self._compatible(current, self._task[1]):
                    self._background = background
                    self._background_error = None

    def _summarize(self, frame: FieldFrame, result, enqueued_s: float, started_s: float) -> _Background:
        rows, cols = _tile_indices(frame.vectors.shape[:2])
        denominator = max(1, int(np.count_nonzero(frame.mask)))
        strengths: list[float] = []; xs: list[float] = []; ys: list[float] = []; counts: list[int] = []
        out_rows: list[np.ndarray] = []; out_cols: list[np.ndarray] = []
        for row in rows:
            for col in cols:
                index = np.ix_(row, col)
                valid = result.valid[index]
                rates = np.where(valid, result.rate_rad_s[index], 0.0)
                total = float(np.sum(rates)) / denominator
                count = int(np.count_nonzero(frame.mask[index]))
                if total > 0:
                    weights = rates
                    raw_total = float(np.sum(weights))
                    x = frame.origin_m[0] + np.broadcast_to(col[None, :], weights.shape) * frame.dx_m
                    y = frame.origin_m[1] + np.broadcast_to(row[:, None], weights.shape) * frame.dy_m
                    xs.append(float(np.sum(x * weights) / raw_total)); ys.append(float(np.sum(y * weights) / raw_total))
                else:
                    xs.append(float(frame.origin_m[0] + np.mean(col) * frame.dx_m))
                    ys.append(float(frame.origin_m[1] + np.mean(row) * frame.dy_m))
                strengths.append(total); counts.append(count); out_rows.append(row); out_cols.append(col)
        return _Background(_key(frame), frame.sim_time_s, enqueued_s, tuple(out_rows), tuple(out_cols),
                           tuple(strengths), tuple(xs), tuple(ys), tuple(counts), result.coverage,
                           result.validity, (time.monotonic() - started_s) * 1000.0)

    def _merged(self, current: FieldFrame, local: FieldView, bounds, focus_ms: float) -> FieldView:
        r0, r1, c0, c1 = bounds
        total_material = max(1, int(np.count_nonzero(current.mask)))
        local_material = int(np.count_nonzero(local.field.mask))
        scale = local_material / total_material
        entries: list[tuple[str, float, float, float]] = []
        for item in local.sample.observations:
            entries.append((f"focus:{item.source_id}", item.position_m[0], item.position_m[1], item.strength * scale))
        with self._condition:
            background = self._background
            background_error = self._background_error
        matches = (background is not None and background.key[0:3] == _key(current)[0:3]
                   and background.key[5:] == _key(current)[5:]
                   and background.validity == "valid" and background.coverage == 1.0)
        if matches:
            for i, (rows, cols, strength, x, y) in enumerate(zip(background.rows, background.cols, background.strengths,
                                                                    background.x_m, background.y_m)):
                overlap = not (rows[-1] < r0 or rows[0] >= r1 or cols[-1] < c0 or cols[0] >= c1)
                if not overlap and strength > 0:
                    entries.append((f"background:{i}", x, y, strength))
        observations = tuple(Observation(name, (x, y, current.origin_m[2]), value, 1,
                                         entity_id=current.entity_id,
                                         quantity="angular_activity_mean_contribution", unit="rad/s")
                             for name, x, y, value in entries)
        x = np.asarray([[item.position_m[0] for item in observations]], dtype=float)
        y = np.asarray([[item.position_m[1] for item in observations]], dtype=float)
        positive = np.asarray([[item.strength for item in observations]], dtype=float)
        contributions = ContributionGrid(x, y, positive, np.zeros_like(positive), current.origin_m[2],
                                         current.entity_id, "angular_activity_mean_contribution", "rad/s")
        age = time.monotonic() - background.enqueued_s if matches else None
        input_total = float(positive.sum())
        diagnostic = dict(recipe="activity", rate_unit="rad/s", mean_rad_s=None,
            max_rad_s=local.diagnostic.get("max_rad_s"), dt_s=local.diagnostic.get("dt_s"),
            max_dt_s=local.diagnostic.get("max_dt_s"), coverage=local.sample.coverage,
            validity=local.sample.validity, reason=local.diagnostic.get("reason"),
            warnings=list(local.diagnostic.get("warnings", [])), sequence=current.sequence,
            previous_sequence=None, previous_sim_time_s=None,
            background_error=background_error,
            focus_compute=dict(focus_cells=int((r1-r0)*(c1-c0)), fraction=float((r1-r0)*(c1-c0)/(current.vectors.shape[0]*current.vectors.shape[1])),
                bounds_indices=[r0, r1, c0, c1], background_age_s=age,
                background_sim_time_s=background.sim_time_s if matches else None,
                background_ready=bool(matches), focus_ms=focus_ms,
                background_ms=background.elapsed_ms if matches else None,
                background_error=background_error,
                scope="focus_current_background_cached", global_current_mean_rad_s=None),
            spatial_aggregation=dict(method="focused_16x16", basis="current focus 4x4 tiles plus non-overlapping cached 16x16 background tile sums; each strength is divided by full-frame material site count", channels={
                "positive": dict(input=input_total, represented=input_total, omitted=0.0),
                "negative": dict(input=0.0, represented=0.0, omitted=0.0),
                "absolute": dict(input=input_total, represented=input_total, omitted=0.0)}),
            source_kind=current.source_kind, entity_id=current.entity_id, provenance=current.provenance,
            shape=list(current.vectors.shape), dx_m=current.dx_m, dy_m=current.dy_m,
            origin_m=list(current.origin_m), sim_time_s=current.sim_time_s)
        summary = (f"活动：关注区即时计算 {focus_ms:.1f} ms；背景缓存"
                   f" {'%.2f s' % age if age is not None else '未就绪'}，非全域当前均值")
        sample = Sample(current.sim_time_s, observations, current.sequence, current.segment_id,
                        local.sample.validity, local.sample.coverage, current.source_kind,
                        time_kind=current.time_kind)
        return FieldView(current, sample, summary, diagnostic, contributions)

    def _empty_focus(self, current, previous, attention, max_dt_s):
        # An ROI outside the field is a valid empty selection, never a request
        # to calculate the whole frame on the foreground path.
        empty = FieldFrame(np.zeros((3, 3, 3)), current.dx_m, current.dy_m, current.sim_time_s,
                           current.origin_m, np.zeros((3, 3), dtype=bool), current.entity_id,
                           current.source_kind, current.segment_id, current.sequence, current.time_kind,
                           current.provenance)
        local = observe_field(empty, "activity", previous=None, max_dt_s=max_dt_s)
        self._enqueue(previous, current, max_dt_s)
        return self._merged(current, local, (0, 0, 0, 0), 0.0)
