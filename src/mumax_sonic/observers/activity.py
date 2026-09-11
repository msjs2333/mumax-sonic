"""Stateless, pairwise angular activity for physical field frames.

The observer intentionally measures only the shortest angular displacement
between two adjacent physical samples.  It neither retains history nor tries
to infer rotations that may have occurred between samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from mumax_sonic.fields import FieldFrame


_LARGE_STEP_RAD = pi / 2.0
_DT_REL_TOLERANCE = 1.0e-9
_ACTIVITY_BLOCK_VOXELS = 4096


@dataclass(frozen=True)
class ActivityResult:
    """Per-site sampled angular speed, in radians per physical second."""

    rate_rad_s: np.ndarray
    valid: np.ndarray
    coverage: float
    validity: str
    dt_s: float | None
    warnings: tuple[str, ...]
    reason: str

    @property
    def mean_rad_s(self) -> float:
        if not np.any(self.valid):
            return float("nan")
        return float(np.mean(self.rate_rad_s[self.valid]))

    @property
    def max_rad_s(self) -> float:
        if not np.any(self.valid):
            return float("nan")
        return float(np.max(self.rate_rad_s[self.valid]))


def angular_activity(
    previous: FieldFrame | None,
    current: FieldFrame,
    *,
    max_dt_s: float | None = None,
) -> ActivityResult:
    """Measure one shortest sampled angular displacement per material site.

    A result is usable only for adjacent, forward, dynamic frames describing
    the same field.  Incomplete vector samples retain rates at their usable
    sites, while the result as a whole is marked ``invalid`` so callers do not
    mistake partial coverage for a complete measurement.
    """

    threshold = _validate_max_dt(max_dt_s)
    shape = current.vectors.shape[:2]
    if not np.any(current.mask):
        return _unavailable(shape, "invalid", "current frame has no material sites")

    if previous is None:
        if current.time_kind != "dynamics":
            return _unavailable(shape, "unsupported", "activity requires dynamic field frames")
        return _unavailable(shape, "warming_up", "no previous physical frame")

    if current.time_kind != "dynamics" or previous.time_kind != "dynamics":
        return _unavailable(shape, "unsupported", "activity requires dynamic field frames")

    if current.entity_id != previous.entity_id:
        return _unavailable(shape, "warming_up", "field entity changed")
    if current.source_kind != previous.source_kind:
        return _unavailable(shape, "warming_up", "field source kind changed")
    if current.segment_id != previous.segment_id:
        return _unavailable(shape, "warming_up", "field segment changed")
    if not _same_geometry(previous, current):
        return _unavailable(shape, "warming_up", "field geometry changed")
    if not np.array_equal(previous.mask, current.mask):
        return _unavailable(shape, "warming_up", "material mask changed")

    if current.sequence <= previous.sequence:
        return _unavailable(shape, "invalid", "sequence is duplicate or moves backward")
    if current.sim_time_s <= previous.sim_time_s:
        return _unavailable(shape, "invalid", "physical time moves backward or does not advance")
    dt_s = float(current.sim_time_s - previous.sim_time_s)
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        return _unavailable(shape, "invalid", "nonpositive physical time interval")
    if current.sequence != previous.sequence + 1:
        return _unavailable(shape, "warming_up", "physical input sequence has a gap", dt_s=dt_s)
    # Decimal sample times can acquire a few ulps of subtraction error.  This
    # is deliberately relative-only: it does not turn a physically meaningful
    # tiny interval into an arbitrary absolute grace period.
    if threshold is not None and dt_s > threshold * (1.0 + _DT_REL_TOLERANCE):
        return _unavailable(shape, "warming_up", "physical time interval exceeds max_dt_s", dt_s=dt_s)

    rate = np.full(shape, np.nan, dtype=float)
    valid = np.zeros(shape, dtype=bool)
    warnings: list[str] = []
    excluded_samples, overflowed_rates, large_step = _angular_rates_by_block(
        previous.vectors,
        current.vectors,
        current.mask,
        dt_s,
        rate,
        valid,
    )
    if excluded_samples:
        warnings.append("non-finite or zero-magnitude material sites excluded")
    if overflowed_rates:
        warnings.append("non-finite angular rates excluded")
    if large_step:
        warnings.append("large angular step approaches the pi aliasing limit")

    material_count = int(np.count_nonzero(current.mask))
    coverage = float(np.count_nonzero(valid) / material_count)
    if coverage < 1.0:
        warnings.append("activity coverage is incomplete")
        validity = "invalid"
        reason = "not every material site has a valid vector pair"
    else:
        validity = "valid"
        reason = "adjacent dynamic frames compared"
    if not np.any(valid):
        warnings.append("no valid activity samples")
    return ActivityResult(rate, valid, coverage, validity, dt_s, tuple(warnings), reason)


def _validate_max_dt(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_dt_s must be a finite positive number or None") from exc
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("max_dt_s must be a finite positive number or None")
    return threshold


def _same_geometry(previous: FieldFrame, current: FieldFrame) -> bool:
    return (
        previous.vectors.shape == current.vectors.shape
        and previous.dx_m == current.dx_m
        and previous.dy_m == current.dy_m
        and previous.origin_m == current.origin_m
    )


def _angular_rates_by_block(
    previous: np.ndarray,
    current: np.ndarray,
    material: np.ndarray,
    dt_s: float,
    rate: np.ndarray,
    valid: np.ndarray,
) -> tuple[bool, bool, bool]:
    """Fill activity outputs in bounded voxel blocks without full-field normalization.

    Scaling each finite vector by its largest absolute component makes all
    intermediate products bounded.  The omitted vector-norm product is the
    same positive factor in both the cross magnitude and dot product, so it
    cancels exactly in ``atan2``.

    Rectangular blocks keep temporary arrays bounded for large fields and do
    not require flattening an input, which could copy a Fortran-layout frame.
    """

    excluded_samples = False
    overflowed_rates = False
    large_step = False
    rows, columns = material.shape
    block_columns = min(columns, _ACTIVITY_BLOCK_VOXELS)
    block_rows = max(1, _ACTIVITY_BLOCK_VOXELS // block_columns)
    for row0 in range(0, rows, block_rows):
        row1 = min(rows, row0 + block_rows)
        for col0 in range(0, columns, block_columns):
            col1 = min(columns, col0 + block_columns)
            prior_block = previous[row0:row1, col0:col1]
            current_block = current[row0:row1, col0:col1]
            material_block = material[row0:row1, col0:col1]
            prior_x, prior_y, prior_z = prior_block[..., 0], prior_block[..., 1], prior_block[..., 2]
            current_x, current_y, current_z = current_block[..., 0], current_block[..., 1], current_block[..., 2]
            prior_finite = np.isfinite(prior_x) & np.isfinite(prior_y) & np.isfinite(prior_z)
            current_finite = np.isfinite(current_x) & np.isfinite(current_y) & np.isfinite(current_z)
            prior_scale = np.maximum(np.abs(prior_x), np.maximum(np.abs(prior_y), np.abs(prior_z)))
            current_scale = np.maximum(np.abs(current_x), np.maximum(np.abs(current_y), np.abs(current_z)))
            usable = material_block & prior_finite & current_finite & (prior_scale > 0.0) & (current_scale > 0.0)
            if np.any(material_block & ~usable):
                excluded_samples = True
            if not np.any(usable):
                continue

            prior_x = np.divide(prior_x, prior_scale, out=np.zeros_like(prior_x), where=usable)
            prior_y = np.divide(prior_y, prior_scale, out=np.zeros_like(prior_y), where=usable)
            prior_z = np.divide(prior_z, prior_scale, out=np.zeros_like(prior_z), where=usable)
            current_x = np.divide(current_x, current_scale, out=np.zeros_like(current_x), where=usable)
            current_y = np.divide(current_y, current_scale, out=np.zeros_like(current_y), where=usable)
            current_z = np.divide(current_z, current_scale, out=np.zeros_like(current_z), where=usable)
            dot = prior_x * current_x + prior_y * current_y + prior_z * current_z
            cross_x = prior_y * current_z - prior_z * current_y
            cross_y = prior_z * current_x - prior_x * current_z
            cross_z = prior_x * current_y - prior_y * current_x
            angle = np.arctan2(np.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z), dot)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                local_rate = angle / dt_s
            finite_rate = np.isfinite(local_rate)
            block_valid = usable & finite_rate
            np.copyto(rate[row0:row1, col0:col1], local_rate, where=block_valid)
            valid[row0:row1, col0:col1] = block_valid
            if np.any(usable & ~finite_rate):
                overflowed_rates = True
            if np.any(block_valid & (angle >= _LARGE_STEP_RAD)):
                large_step = True
    return excluded_samples, overflowed_rates, large_step


def _unavailable(
    shape: tuple[int, int], validity: str, reason: str, *, dt_s: float | None = None
) -> ActivityResult:
    return ActivityResult(
        np.full(shape, np.nan, dtype=float),
        np.zeros(shape, dtype=bool),
        0.0,
        validity,
        dt_s,
        (),
        reason,
    )
