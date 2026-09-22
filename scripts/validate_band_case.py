"""Validate the completed P4c MuMax3 frequency-band replay cases.

The solver run is deliberately outside this validator.  It only reads the
three explicit OVF replay manifests beneath a completed case root and checks
their physical time, local spectral power, phase, and downstream spatial
aggregation evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mumax_sonic.attention import Attention
from mumax_sonic.field_pipeline import apply_aggregation, observe_field
from mumax_sonic.mapping import map_sample_with_report
from mumax_sonic.observers.band import BandConfig, band_power
from mumax_sonic.sources.ovf_replay import load_ovf_replay


_CASE_NAMES = ("in_phase", "opposite_phase", "out_band")
_WINDOW = 256
_FRAME_COUNT = 320
_DT_S = 5e-12
_CONE_AMPLITUDE = 0.1
_EXPECTED_POWER = _CONE_AMPLITUDE ** 2
_FREQUENCY_HZ = {"in_phase": 9.375e9, "opposite_phase": 9.375e9, "out_band": 18.75e9}
_BAND = BandConfig(8e9, 12e9, _WINDOW, (0, 0, 1))

# These account for a real LLG trajectory written as binary OVF rather than a
# synthetic array.  They are deliberately much tighter than the requested
# scientific distinction (in-band power versus the 1e-6 out-of-band limit).
_POWER_RTOL = 2e-3
_FREQUENCY_RTOL = 1e-3
_CONE_DRIFT_ATOL = 1e-4
_Z_COMPONENT_ATOL = 1e-4
_OPPOSITION_RTOL = 1e-3


def _manifest_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read manifest: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported replay manifest")
    if value.get("value_unit") != "1" or value.get("mask") != "all":
        raise ValueError(f"{path}: P4c requires unitless all-material directions")
    if value.get("time_kind") != "dynamics":
        raise ValueError(f"{path}: P4c requires dynamic frames")
    return value


def _fit_frequency(frames, expected_hz: float) -> tuple[float, float, float]:
    """Return median local absolute frequency and the largest local residual."""
    times = np.asarray([frame.sim_time_s for frame in frames], dtype=float)
    transverse = np.stack([frame.vectors[..., :2] for frame in frames])
    phase = np.unwrap(np.angle(transverse[..., 0] + 1j * transverse[..., 1]), axis=0)
    centered = times - float(np.mean(times))
    slopes = np.sum(centered[:, None, None] * phase, axis=0) / float(np.sum(centered ** 2))
    fitted = np.abs(slopes) / (2 * math.pi)
    # A linear phase fit also guards against a transient that happens to leave
    # the right endpoint frequency intact.
    intercept = np.mean(phase, axis=0) - slopes * float(np.mean(times))
    residual = phase - (times[:, None, None] * slopes + intercept)
    relative_error = float(np.max(np.abs(fitted - expected_hz) / expected_hz))
    if relative_error > _FREQUENCY_RTOL:
        raise ValueError("fitted local precession frequency is outside tolerance")
    return float(np.median(fitted)), relative_error, float(np.max(np.abs(residual)))


def _check_frames(name: str, frames) -> dict[str, Any]:
    if len(frames) != _FRAME_COUNT:
        raise ValueError(f"{name}: expected exactly {_FRAME_COUNT} frames")
    first = frames[0]
    if first.vectors.shape != (4, 8, 3):
        raise ValueError(f"{name}: expected frozen 8x4x1 grid")
    if any(frame.vectors.shape != first.vectors.shape for frame in frames):
        raise ValueError(f"{name}: frame geometry changes")
    if any(not np.all(frame.mask) for frame in frames):
        raise ValueError(f"{name}: every grid site must be material")
    sequences = np.asarray([frame.sequence for frame in frames], dtype=int)
    if not np.array_equal(sequences, np.arange(_FRAME_COUNT)):
        raise ValueError(f"{name}: frame sequences must be contiguous 0..319")
    times = np.asarray([frame.sim_time_s for frame in frames], dtype=float)
    intervals = np.diff(times)
    if not np.allclose(intervals, _DT_S, rtol=1e-6, atol=0.0):
        raise ValueError(f"{name}: physical timestamps are not regular 5 ps samples")
    if not math.isclose(float(times[0]), 0.0, abs_tol=1e-18):
        raise ValueError(f"{name}: first physical time must be zero")
    vectors = np.stack([frame.vectors for frame in frames])
    norms = np.linalg.norm(vectors, axis=-1)
    if not np.all(np.isfinite(vectors)) or not np.allclose(norms, 1.0, rtol=2e-5, atol=2e-5):
        raise ValueError(f"{name}: all-material directions must be finite and unit length")
    amplitude = np.linalg.norm(vectors[..., :2], axis=-1)
    drift = float(np.max(np.abs(amplitude - _CONE_AMPLITUDE)))
    if drift > _CONE_DRIFT_ATOL:
        raise ValueError(f"{name}: transverse cone amplitude drift exceeds {_CONE_DRIFT_ATOL:g}")
    z_error = float(np.max(np.abs(vectors[..., 2] - math.sqrt(0.99))))
    if z_error > _Z_COMPONENT_ATOL:
        raise ValueError(f"{name}: z component drift exceeds {_Z_COMPONENT_ATOL:g}")
    expected_frequency = _FREQUENCY_HZ[name]
    fitted_frequency, frequency_relative_error, phase_residual = _fit_frequency(frames, expected_frequency)
    return {
        "frames": len(frames), "shape_yx": [4, 8], "material_sites": 32,
        "dt_s": float(intervals[0]), "time_range_s": [float(times[0]), float(times[-1])],
        "cone_amplitude_target": _CONE_AMPLITUDE, "cone_amplitude_drift_max": drift,
        "z_target": math.sqrt(0.99), "z_error_max": z_error,
        "fitted_frequency_hz": fitted_frequency, "frequency_target_hz": expected_frequency,
        "frequency_relative_error_max": frequency_relative_error,
        "phase_fit_residual_rad_max": phase_residual,
    }


def _windows_and_sound(name: str, frames) -> tuple[dict[str, Any], list[float]]:
    warmup = valid = 0
    powers: list[float] = []
    local_power_errors: list[float] = []
    adaptive_sources: list[int] = []
    fixed_sources: list[int] = []
    for index, frame in enumerate(frames):
        history = frames[:index + 1]
        result = band_power(history, _BAND)
        view = observe_field(frame, "band", history=history, band_config=_BAND)
        if index < _WINDOW - 1:
            if result.validity != "warming_up" or view.sample.validity != "warming_up":
                raise ValueError(f"{name}: frame {index} must remain band warmup")
            warmup += 1
            continue
        if result.validity != "valid" or result.coverage != 1.0:
            raise ValueError(f"{name}: frame {index} does not have a valid full band window")
        valid += 1
        powers.append(float(result.mean_power))
        if name in ("in_phase", "opposite_phase"):
            local_error = float(np.max(np.abs(result.power - _EXPECTED_POWER)))
            local_power_errors.append(local_error)
            if not np.allclose(result.power, _EXPECTED_POWER, rtol=_POWER_RTOL, atol=1e-12):
                raise ValueError(f"{name}: a valid-window local power differs from analytic amplitude squared")
        elif result.mean_power >= 1e-6 * _EXPECTED_POWER:
            raise ValueError("out_band: a valid-window leakage exceeds 1e-6 of reference")
        if view.sample.validity != "valid":
            raise ValueError(f"{name}: field pipeline did not preserve valid band window")
        attention = Attention(extent_m=frame.extent_m, origin_m=frame.center_m[:2])
        adapted = apply_aggregation(view, attention, budget=8, mode="adaptive")
        if adapted.diagnostic.get("adaptive_aggregation", {}).get("status") != "valid":
            raise ValueError(f"{name}: adaptive band aggregation is invalid")
        channels = adapted.diagnostic["spatial_aggregation"]["channels"]["positive"]
        if not math.isclose(float(channels["represented"]), float(channels["input"]), rel_tol=2e-12, abs_tol=1e-14):
            raise ValueError(f"{name}: adaptive aggregation does not preserve per-site band power")
        mapped = map_sample_with_report(adapted.sample, attention, budget=8, strength_reference=_EXPECTED_POWER)
        if len(mapped.scene.sources) > 8 or any(
                not math.isfinite(source.gain) or source.gain < 0 for source in mapped.scene.sources):
            raise ValueError(f"{name}: mapped sound sources are not finite within budget")
        adaptive_sources.append(len(mapped.scene.sources))
        fixed_sources.append(len(view.sample.observations))
    if (warmup, valid) != (_WINDOW - 1, _FRAME_COUNT - _WINDOW + 1):
        raise ValueError(f"{name}: unexpected rolling-window state counts")
    final = band_power(frames[-_WINDOW:], _BAND)
    return ({
        "states": {"warming_up": warmup, "valid": valid},
        "window_samples": _WINDOW, "window_dt_s": final.dt_s,
        "window_span_s": final.window_span_s, "frequency_resolution_hz": final.frequency_resolution_hz,
        "included_bin_frequencies_hz": list(final.bin_frequencies_hz),
        "mean_power": float(np.mean(powers)), "power_range": {"min": float(min(powers)), "max": float(max(powers))},
        "local_power_error_max": float(max(local_power_errors)) if local_power_errors else None,
        "sound_sources": {"adaptive_budget": 8, "fixed_count_range": {"min": min(fixed_sources), "max": max(fixed_sources)},
                          "adaptive_count_range": {"min": min(adaptive_sources), "max": max(adaptive_sources)},
                          "strength_reference": _EXPECTED_POWER},
    }, powers)


def _opposite_checks(frames, final_power: float) -> dict[str, Any]:
    transverse = np.stack([frame.vectors[..., :2] for frame in frames])
    spatial_mean = np.mean(transverse, axis=(1, 2))
    maximum_mean = float(np.max(np.linalg.norm(spatial_mean, axis=-1)))
    if maximum_mean > _OPPOSITION_RTOL * _CONE_AMPLITUDE:
        raise ValueError("opposite_phase: spatial transverse mean does not cancel")
    paired_sum = transverse[:, :, :4, :] + transverse[:, :, 4:, :]
    paired_sum_max = float(np.max(np.linalg.norm(paired_sum, axis=-1)))
    if paired_sum_max > _OPPOSITION_RTOL * _CONE_AMPLITUDE:
        raise ValueError("opposite_phase: corresponding left/right transverse vectors are not opposed")
    left = band_power(tuple(frames[-_WINDOW:]), _BAND).power[:, :4]
    right = band_power(tuple(frames[-_WINDOW:]), _BAND).power[:, 4:]
    left_mean, right_mean = float(np.mean(left)), float(np.mean(right))
    if not math.isclose(left_mean, right_mean, rel_tol=_POWER_RTOL, abs_tol=1e-12):
        raise ValueError("opposite_phase: left and right local powers are unequal")
    if not math.isclose(final_power, _EXPECTED_POWER, rel_tol=_POWER_RTOL, abs_tol=1e-12):
        raise ValueError("opposite_phase: local power is not preserved despite phase opposition")
    return {"spatial_mean_transverse_max": maximum_mean, "paired_transverse_sum_max": paired_sum_max, "left_mean_power": left_mean,
            "right_mean_power": right_mean}


def validate_case(root: Path) -> dict[str, Any]:
    """Validate ``root/{in_phase,opposite_phase,out_band}/replay.json``."""
    root = Path(root).resolve()
    reports: dict[str, Any] = {}
    window_powers: dict[str, list[float]] = {}
    stored_frames = {}
    for name in _CASE_NAMES:
        manifest = root / name / "replay.json"
        _manifest_metadata(manifest)
        try:
            frames = tuple(load_ovf_replay(manifest).frames)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{name}: replay manifest cannot be loaded") from exc
        physical = _check_frames(name, frames)
        spectral, powers = _windows_and_sound(name, frames)
        reports[name] = {"manifest": str(manifest), "physical": physical, "band": spectral}
        # Compact aliases keep the report useful to callers that only need the
        # acceptance quantities, while the nested fields retain the evidence.
        reports[name].update(status="valid", mean_power=spectral["mean_power"])
        window_powers[name] = powers
        stored_frames[name] = frames
    for name in ("in_phase", "opposite_phase"):
        if not all(math.isclose(power, _EXPECTED_POWER, rel_tol=_POWER_RTOL, abs_tol=1e-12)
                   for power in window_powers[name]):
            raise ValueError(f"{name}: in-band power differs from analytic cone-amplitude squared reference")
    if not np.allclose(window_powers["in_phase"], window_powers["opposite_phase"], rtol=_POWER_RTOL, atol=1e-12):
        raise ValueError("in_phase and opposite_phase band powers differ")
    if any(power >= 1e-6 * _EXPECTED_POWER for power in window_powers["out_band"]):
        raise ValueError("out_band: leakage exceeds 1e-6 of the analytic in-band reference")
    reports["opposite_phase"]["opposition"] = _opposite_checks(stored_frames["opposite_phase"], window_powers["opposite_phase"][-1])
    reports["opposite_phase"]["spatial_mean_transverse"] = reports["opposite_phase"]["opposition"]["spatial_mean_transverse_max"]
    return {
        "status": "valid", "case_root": str(root),
        "frames": {"count": _FRAME_COUNT, "dt_s": _DT_S, "window": _WINDOW},
        "band": {"low_hz": _BAND.low_hz, "high_hz": _BAND.high_hz},
        "contract": {"grid_xyz": [8, 4, 1], "frames": _FRAME_COUNT, "dt_s": _DT_S,
                     "band_hz": [_BAND.low_hz, _BAND.high_hz], "reference_axis": list(_BAND.reference_axis),
                     "analytic_per_site_power": _EXPECTED_POWER,
                     "out_band_upper_bound": 1e-6 * _EXPECTED_POWER,
                     "frequency_relative_tolerance": _FREQUENCY_RTOL,
                     "cone_amplitude_drift_absolute_tolerance": _CONE_DRIFT_ATOL},
        "cases": reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="completed root containing the three P4c case directories")
    parser.add_argument("--output", type=Path, help="write JSON report to this path")
    args = parser.parse_args(argv)
    try:
        report = validate_case(args.root)
    except ValueError as exc:
        parser.error(str(exc))
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
