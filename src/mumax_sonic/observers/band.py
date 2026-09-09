"""Windowed, local frequency-band power for dynamic vector fields.

The observer is deliberately stateless: callers provide the physical history
ending at the frame they want to inspect.  It never fills gaps or resamples
timestamps, because either operation could manufacture spectral content.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from mumax_sonic.fields import FieldFrame


@dataclass(frozen=True)
class BandConfig:
    """Physical FFT band and window settings."""

    low_hz: float = 8e9
    high_hz: float = 12e9
    window_samples: int = 256
    reference_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        try:
            low, high = float(self.low_hz), float(self.high_hz)
        except (TypeError, ValueError) as exc:
            raise ValueError("band limits must be finite positive frequencies") from exc
        if not np.isfinite(low) or not np.isfinite(high) or not (0.0 < low < high):
            raise ValueError("band limits must satisfy finite 0 < low_hz < high_hz")
        if isinstance(self.window_samples, bool) or not isinstance(self.window_samples, Integral):
            raise ValueError("window_samples must be an integer from 16 through 4096")
        samples = int(self.window_samples)
        if not 16 <= samples <= 4096:
            raise ValueError("window_samples must be an integer from 16 through 4096")
        try:
            axis = np.asarray(self.reference_axis, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("reference_axis must contain three finite values") from exc
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("reference_axis must contain three finite values")
        scale = float(np.max(np.abs(axis)))
        scaled = axis / scale if scale > 0.0 else axis
        norm = float(np.linalg.norm(scaled))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("reference_axis must be nonzero")
        object.__setattr__(self, "low_hz", low)
        object.__setattr__(self, "high_hz", high)
        object.__setattr__(self, "window_samples", samples)
        object.__setattr__(self, "reference_axis", tuple(scaled / norm))


@dataclass(frozen=True)
class BandResult:
    """Per-site band power (input-unit squared) and its validity evidence."""

    power: np.ndarray
    valid: np.ndarray
    coverage: float
    validity: str
    reason: str
    warnings: tuple[str, ...]
    mean_power: float
    max_power: float
    samples_available: int
    samples_required: int
    dt_s: float | None
    frequency_resolution_hz: float | None
    window_span_s: float | None
    bin_frequencies_hz: tuple[float, ...]


def band_power(history, config: BandConfig = BandConfig()) -> BandResult:
    """Measure local transverse FFT power in ``config``'s inclusive band.

    A full contiguous window is required.  Geometry/provenance changes and
    sequence gaps discard the older prefix; duplicate or backward sequence and
    time values are invalid rather than silently treated as a new run.
    """
    if not isinstance(config, BandConfig):
        raise TypeError("config must be a BandConfig")
    frames = tuple(history)
    if not frames:
        raise ValueError("history must contain at least one FieldFrame")
    if not all(isinstance(frame, FieldFrame) for frame in frames):
        raise TypeError("history must contain only FieldFrame instances")
    # Older observations cannot affect the requested FFT window, including an
    # old run boundary before the last N samples.
    frames = frames[-config.window_samples:]
    current = frames[-1]
    shape = current.vectors.shape[:2]
    required = config.window_samples
    if not np.any(current.mask):
        return _unavailable(shape, "invalid", "current frame has no material sites", len(frames), required)
    if current.time_kind != "dynamics":
        return _unavailable(shape, "unsupported", "band power requires dynamic field frames", len(frames), required)

    # A non-increasing sequence is corrupt ordering even if a later boundary
    # would otherwise begin a new suffix.
    for previous, following in zip(frames, frames[1:]):
        if following.sequence <= previous.sequence:
            return _unavailable(shape, "invalid", "sequence is duplicate or moves backward", len(frames), required)
        if following.sim_time_s <= previous.sim_time_s:
            return _unavailable(shape, "invalid", "physical time moves backward or does not advance", len(frames), required)
        if following.time_kind != "dynamics" or previous.time_kind != "dynamics":
            return _unavailable(shape, "unsupported", "band power requires dynamic field frames", len(frames), required)

    suffix_start = 0
    for index, (previous, following) in enumerate(zip(frames, frames[1:]), start=1):
        if not _continuous(previous, following):
            suffix_start = index
    suffix = frames[suffix_start:]
    available = len(suffix)
    if available < required:
        return _unavailable(shape, "warming_up", "need a full contiguous physical window", available, required)
    window = suffix[-required:]
    times = np.asarray([frame.sim_time_s for frame in window], dtype=float)
    intervals = np.diff(times)
    dt = float(intervals[0])
    if not np.isfinite(dt) or dt <= 0.0:
        return _unavailable(shape, "invalid", "nonpositive or non-finite physical time interval", available, required)
    # Relative-only tolerance retains meaning at sub-nanosecond simulation time.
    if not np.all(np.isfinite(intervals)) or not np.allclose(intervals, dt, rtol=1e-6, atol=0.0):
        return _unavailable(shape, "unsupported", "nonuniform physical timestamps require resampling", available, required, dt_s=dt)
    nyquist = 0.5 / dt
    frequencies = np.fft.rfftfreq(required, d=dt)
    included = (frequencies >= config.low_hz) & (frequencies <= config.high_hz) & (frequencies > 0.0)
    if not config.high_hz < nyquist:
        return _unavailable(shape, "unsupported", "band upper edge must be strictly below Nyquist", available, required, dt_s=dt,
                            frequencies=())
    if int(np.count_nonzero(included)) < 2:
        return _unavailable(shape, "unsupported", "band contains fewer than two positive FFT bins", available, required, dt_s=dt,
                            frequencies=())

    vectors = np.stack([frame.vectors for frame in window], axis=0)
    material = current.mask
    # All frames have identical masks within a continuous suffix.  A site is
    # usable only if it is valid for the entire window, never partially filled.
    normalized, usable = _normalized(vectors, material)
    full_valid = np.all(usable, axis=0)
    power = np.full(shape, np.nan, dtype=float)
    warnings: list[str] = []
    if np.any(material & ~full_valid):
        warnings.append("non-finite or zero-magnitude material sites excluded over the full window")
    if np.any(full_valid):
        ref = np.asarray(config.reference_axis)
        transverse = normalized - np.sum(normalized * ref, axis=-1, keepdims=True) * ref
        signal = transverse[:, full_valid, :]
        signal = signal - np.mean(signal, axis=0, keepdims=True)
        taper = np.hanning(required + 1)[:-1]
        spectrum = np.fft.rfft(signal * taper[:, None, None], axis=0)
        spectral_power = np.abs(spectrum) ** 2 / (required * float(np.sum(taper ** 2)))
        # One-sided real FFT: all non-DC and non-Nyquist bins represent paired
        # negative frequencies.
        doubles = np.ones(frequencies.shape, dtype=float)
        doubles[1:] = 2.0
        if required % 2 == 0:
            doubles[-1] = 1.0
        site_power = np.sum(spectral_power[included] * doubles[included, None, None], axis=(0, 2))
        power[full_valid] = site_power
    material_count = int(np.count_nonzero(material))
    coverage = float(np.count_nonzero(full_valid) / material_count)
    if coverage < 1.0:
        warnings.append("band power coverage is incomplete")
        validity = "invalid"
        reason = "not every material site has valid vectors over the full window"
    else:
        validity = "valid"
        reason = "full contiguous dynamic window analyzed"
    if not np.any(full_valid):
        warnings.append("no valid band-power samples")
    return BandResult(power, full_valid, coverage, validity, reason, tuple(warnings),
                      float(np.mean(power[full_valid])) if np.any(full_valid) else float("nan"),
                      float(np.max(power[full_valid])) if np.any(full_valid) else float("nan"),
                      available, required, dt, 1.0 / (required * dt), (required - 1) * dt,
                      tuple(float(value) for value in frequencies[included]))


def _continuous(previous: FieldFrame, current: FieldFrame) -> bool:
    return (
        current.sequence == previous.sequence + 1
        and current.entity_id == previous.entity_id
        and current.source_kind == previous.source_kind
        and current.segment_id == previous.segment_id
        and _same_geometry(previous, current)
        and np.array_equal(previous.mask, current.mask)
    )


def _same_geometry(previous: FieldFrame, current: FieldFrame) -> bool:
    return (previous.vectors.shape == current.vectors.shape and previous.dx_m == current.dx_m
            and previous.dy_m == current.dy_m and previous.origin_m == current.origin_m)


def _normalized(vectors: np.ndarray, material: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.all(np.isfinite(vectors), axis=-1)
    safe = np.where(finite[..., None], vectors, 0.0)
    scale = np.max(np.abs(safe), axis=-1)
    scaled = np.divide(safe, scale[..., None], out=np.zeros_like(safe), where=scale[..., None] > 0.0)
    norm = np.linalg.norm(scaled, axis=-1)
    usable = material[None, ...] & finite & (scale > 0.0) & np.isfinite(norm) & (norm > 0.0)
    normalized = np.zeros_like(safe)
    normalized[usable] = scaled[usable] / norm[usable, None]
    return normalized, usable


def _unavailable(shape, validity, reason, available, required, *, dt_s=None, frequencies=()) -> BandResult:
    return BandResult(np.full(shape, np.nan, dtype=float), np.zeros(shape, dtype=bool), 0.0,
                      validity, reason, (), float("nan"), float("nan"), available, required,
                      dt_s, None, None, tuple(float(value) for value in frequencies))
