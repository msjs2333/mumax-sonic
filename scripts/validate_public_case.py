"""Offline consistency checks for an explicitly declared MuMax3 field case.

The validator reads the completed manifest and MuMax3 ``table.txt`` supplied
by a case.  It does not follow a live directory, modify solver output, or make
claims about numerical convergence or listening tests.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mumax_sonic.attention import Attention
from mumax_sonic.field_pipeline import apply_aggregation, observe_field
from mumax_sonic.fields import FieldFrame
from mumax_sonic.mapping import map_sample_with_report
from mumax_sonic.observers.activity import angular_activity
from mumax_sonic.observers.topology import topology
from mumax_sonic.sources.ovf_replay import load_ovf_replay


# Binary4 OVF samples and MuMax's text table have different rounding paths.
_MEAN_RTOL = 2.0e-6
_MEAN_ATOL = 2.0e-7
_ACTIVITY_RTOL = 3.0e-10
_ACTIVITY_ATOL = 2.0e-3


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read JSON manifest") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema")
    if raw.get("value_unit") != "1":
        raise ValueError("public case validation requires unitless magnetization directions")
    if raw.get("mask") != "all":
        raise ValueError("public case validation requires an explicit all-material mask")
    if raw.get("z_index") is not None:
        raise ValueError("public case validation requires a single-layer manifest without z_index")
    if raw.get("time_kind") != "dynamics":
        raise ValueError("activity validation requires dynamic frames")
    return raw


def _header_key(value: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", value.strip().lower().split("(", 1)[0])


def _read_table(path: Path) -> list[dict[str, float]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError("cannot read MuMax3 table") from exc
    header = next((line.lstrip()[1:].strip() for line in lines if line.lstrip().startswith("#")), None)
    if not header:
        raise ValueError("MuMax3 table has no header")
    columns = re.split(r"\t+", header)
    if len(columns) < 4:
        columns = header.split()
    keys = [_header_key(column) for column in columns]
    aliases = {"t": ("t",), "mx": ("mx",), "my": ("my",), "mz": ("mz",)}
    try:
        positions = {name: next(i for i, key in enumerate(keys) if key in candidates)
                     for name, candidates in aliases.items()}
    except StopIteration as exc:
        raise ValueError("MuMax3 table must contain t, mx, my, and mz columns") from exc
    units = {}
    for name, position in positions.items():
        match = re.fullmatch(r".*?\((.*?)\)\s*", columns[position].strip())
        if match is None:
            raise ValueError("MuMax3 table headers must state units")
        units[name] = match.group(1).strip()
    if units != {"t": "s", "mx": "", "my": "", "mz": ""}:
        raise ValueError("public case table must use seconds and unitless mx/my/mz")
    result: list[dict[str, float]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = re.split(r"\t+", stripped)
        if len(values) < len(columns):
            values = stripped.split()
        if len(values) <= max(positions.values()):
            raise ValueError("MuMax3 table data row is shorter than its header")
        row = {name: _number(values[position], f"table {name}") for name, position in positions.items()}
        if result and row["t"] <= result[-1]["t"]:
            raise ValueError("MuMax3 table times must be strictly increasing")
        result.append(row)
    if not result:
        raise ValueError("MuMax3 table contains no data rows")
    return result


def _independent_activity(previous: FieldFrame, current: FieldFrame) -> np.ndarray:
    prior = previous.vectors
    later = current.vectors
    prior_norm = np.linalg.norm(prior, axis=-1)
    later_norm = np.linalg.norm(later, axis=-1)
    usable = np.isfinite(prior).all(axis=-1) & np.isfinite(later).all(axis=-1) & (prior_norm > 0) & (later_norm > 0)
    if not np.all(usable):
        raise ValueError("all-material case has non-finite or zero-magnitude vector samples")
    unit_prior = prior / prior_norm[..., None]
    unit_later = later / later_norm[..., None]
    dots = np.sum(unit_prior * unit_later, axis=-1)
    # atan2(|u×v|, u·v) is the independently normalized geometric angle.
    # It remains useful for the small 10 ps rotations where arccos loses
    # most of its significant digits.
    sine = np.linalg.norm(np.cross(unit_prior, unit_later), axis=-1)
    return np.arctan2(sine, np.clip(dots, -1.0, 1.0)) / (current.sim_time_s - previous.sim_time_s)


def _range(values: list[float]) -> dict[str, float]:
    return {"min": float(min(values)), "max": float(max(values))}


def validate_case(manifest_path, table_path) -> dict[str, Any]:
    """Validate one completed public field case and return a JSON-ready report.

    Invalid input or a failed numerical consistency condition raises ``ValueError``.
    """
    manifest = Path(manifest_path).resolve()
    table_file = Path(table_path).resolve()
    meta = _load_manifest(manifest)
    try:
        frames = list(load_ovf_replay(manifest).frames)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("manifest does not satisfy the existing OVF replay contract") from exc
    if len(frames) < 2:
        raise ValueError("public case validation requires at least two replay frames")
    table = _read_table(table_file)
    if len(table) != len(frames):
        raise ValueError("manifest frame count and MuMax3 table row count differ")

    component_errors = {name: 0.0 for name in ("mx", "my", "mz")}
    time_error = 0.0
    raw_norms: list[np.ndarray] = []
    for frame, row in zip(frames, table):
        time_error = max(time_error, abs(frame.sim_time_s - row["t"]))
        means = np.mean(frame.vectors, axis=(0, 1))
        for component, value in zip(("mx", "my", "mz"), means):
            error = abs(float(value) - row[component])
            component_errors[component] = max(component_errors[component], error)
            if not math.isclose(float(value), row[component], rel_tol=_MEAN_RTOL, abs_tol=_MEAN_ATOL):
                raise ValueError(f"OVF mean {component} disagrees with MuMax3 table")
        raw_norms.append(np.linalg.norm(frame.vectors, axis=-1))
    if time_error > 1e-18 + 1e-9 * max(abs(frame.sim_time_s) for frame in frames):
        raise ValueError("OVF frame times disagree with MuMax3 table")
    magnitudes = np.concatenate([item.ravel() for item in raw_norms])
    if not np.isfinite(magnitudes).all() or np.any(magnitudes <= 0.0):
        raise ValueError("all-material vectors must be finite and nonzero")
    if not np.allclose(magnitudes, 1.0, rtol=_MEAN_RTOL, atol=_MEAN_ATOL):
        raise ValueError("unitless public-case magnetization directions must have unit magnitude")

    warmup = angular_activity(None, frames[0])
    if warmup.validity != "warming_up" or np.any(warmup.valid):
        raise ValueError("first activity frame must remain a no-predecessor warmup")
    warmup_view = observe_field(frames[0], "activity", previous=None)
    if warmup_view.sample.validity != "warming_up":
        raise ValueError("field pipeline did not preserve first-frame activity warmup")

    activity_means: list[float] = []
    activity_maxima: list[float] = []
    activity_reference_error = 0.0
    aggregation_coverage: list[float] = []
    aggregation_error = 0.0
    for previous, current in zip(frames, frames[1:]):
        observed = angular_activity(previous, current)
        if observed.validity != "valid" or observed.coverage != 1.0 or observed.dt_s is None:
            raise ValueError("existing activity observer did not produce a complete adjacent-frame observation")
        reference = _independent_activity(previous, current)
        error = float(np.max(np.abs(observed.rate_rad_s - reference)))
        activity_reference_error = max(activity_reference_error, error)
        if not np.allclose(observed.rate_rad_s, reference, rtol=_ACTIVITY_RTOL, atol=_ACTIVITY_ATOL):
            raise ValueError("activity observer disagrees with independent angle/dt reference")
        activity_means.append(observed.mean_rad_s)
        activity_maxima.append(observed.max_rad_s)
        view = observe_field(current, "activity", previous=previous)
        attention = Attention(extent_m=current.extent_m, origin_m=current.center_m[:2])
        sound_view = apply_aggregation(view, attention, budget=8, mode="adaptive")
        info = sound_view.diagnostic.get("adaptive_aggregation", {})
        if info.get("status") != "valid":
            raise ValueError("activity contribution aggregation is not valid")
        observations = sound_view.sample.observations
        if not observations or any(not math.isfinite(item.strength) or item.strength < 0 for item in observations):
            raise ValueError("activity sound-field aggregation produced invalid observations")
        mapped = map_sample_with_report(sound_view.sample, attention, budget=8,
                                        strength_reference=1e10)
        if len(mapped.scene.sources) > 8:
            raise ValueError("activity observations exceeded the requested SonicScene budget")
        channels = sound_view.diagnostic["spatial_aggregation"]["channels"]
        absolute = channels["absolute"]
        if absolute["input"] is None or absolute["represented"] is None:
            raise ValueError("activity contribution coverage is unavailable")
        aggregation_coverage.append(float(absolute["represented"] / absolute["input"]) if absolute["input"] else 1.0)
        aggregation_error = max(aggregation_error, abs(float(absolute["represented"] - absolute["input"])))
        if not math.isclose(float(absolute["represented"]), float(absolute["input"]), rel_tol=2e-12, abs_tol=1e-6):
            raise ValueError("activity sound-field aggregation does not preserve contribution coverage")

    topology_rows: list[tuple[float, float, float, float, float]] = []
    ledger_error = 0.0
    for frame in frames:
        result = topology(frame.vectors, frame.dx_m, frame.dy_m, mask=frame.mask, boundary="open", method="solid_angle")
        if result.coverage != 1.0 or not np.any(result.valid):
            raise ValueError("open-boundary topology support is incomplete")
        if not (math.isfinite(result.q_pos) and math.isfinite(result.q_neg) and result.q_pos >= 0 and result.q_neg >= 0):
            raise ValueError("topology sign ledger is invalid")
        ledger_error = max(ledger_error, abs(result.q_net - (result.q_pos - result.q_neg)),
                           abs(result.q_abs - (result.q_pos + result.q_neg)))
        topology_rows.append((result.q_pos, result.q_neg, result.q_net, result.q_abs, result.coverage))
    if ledger_error > 1e-12:
        raise ValueError("topology signed ledger does not close within tolerance")

    times = [frame.sim_time_s for frame in frames]
    sequences = [frame.sequence for frame in frames]
    report = {
        "status": "valid",
        "case": {"manifest": str(manifest), "table": str(table_file), "entity_id": meta.get("entity_id"),
                 "segment_id": meta.get("segment_id"), "value_unit": meta.get("value_unit"),
                 "shape_yx": list(frames[0].vectors.shape[:2]), "step_m": [frames[0].dx_m, frames[0].dy_m],
                 "origin_m": list(frames[0].origin_m)},
        "frames": {"count": len(frames), "sequence_range": [min(sequences), max(sequences)],
                   "time_s": {"start": times[0], "end": times[-1], "dt_range": _range([b - a for a, b in zip(times, times[1:])])}},
        "ovf_table": {"table_rows": len(table), "max_time_error_s": time_error,
                      "max_mean_component_error": component_errors, "mean_tolerance": {"rtol": _MEAN_RTOL, "atol": _MEAN_ATOL}},
        "vectors": {"finite": True, "raw_magnitude_range": {"min": float(magnitudes.min()), "max": float(magnitudes.max())},
                    "unit_magnitude_tolerance": {"rtol": _MEAN_RTOL, "atol": _MEAN_ATOL},
                    "normalization": "each finite unit vector was independently normalized before angular checks"},
        "activity": {"warmup_observations": 1, "actual_observations": len(activity_means), "unit": "rad/s", "dt_s": _range([b - a for a, b in zip(times, times[1:])]),
                     "mean_rad_s": _range(activity_means), "max_rad_s": _range(activity_maxima),
                     "max_reference_error_rad_s": activity_reference_error},
        "topology_open_solid_angle": {"frames": len(topology_rows), "q_pos": _range([row[0] for row in topology_rows]),
                                      "q_neg": _range([row[1] for row in topology_rows]), "q_net": _range([row[2] for row in topology_rows]),
                                      "q_abs": _range([row[3] for row in topology_rows]), "coverage": _range([row[4] for row in topology_rows]),
                                      "max_ledger_error": ledger_error,
                                      "note": "Open boundaries are checked as a signed contribution ledger; no integer charge is required."},
        "activity_sound_field": {"frames": len(aggregation_coverage), "budget": 8,
                                 "strength_reference_rad_s": 1e10,
                                 "attention": "default gain settings with field extent and field centre as physical viewport",
                                 "contribution_coverage": _range(aggregation_coverage),
                                 "max_unrepresented_contribution_rad_s": aggregation_error,
                                 "note": "This checks existing adaptive aggregation and mapping to finite SonicScene sources, not device playback."},
        "limits": ["Offline consistency check only; it does not establish solver convergence or scientific classification.",
                   "No audio device, HRTF, or human listening validation was run."],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("table", type=Path)
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    args = parser.parse_args(argv)
    try:
        report = validate_case(args.manifest, args.table)
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
