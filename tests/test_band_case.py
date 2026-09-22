"""Independent P4c frequency-band case checks.

These tests deliberately build the three replay directories locally.  They do
not invoke MuMax3 and do not depend on the case producer's fixture helpers.
The validator is the only implementation under test.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.validate_band_case import validate_case


FRAME_COUNT = 320
WINDOW = 256
DT_S = 5e-12
GRID_Y, GRID_X = 4, 8
AMPLITUDE = 0.1
Z_COMPONENT = math.sqrt(0.99)
IN_BAND_HZ = 9.375e9
OUT_BAND_HZ = 18.75e9


def _write_ovf(path: Path, vectors: np.ndarray, time_s: float) -> None:
    """Write one tiny Binary 8, full-XYZ, single-layer OVF frame."""
    header = "\n".join(
        (
            "# OOMMF OVF 2.0",
            "# Segment count: 1",
            "# Begin: Segment",
            "# Begin: Header",
            "# meshtype: rectangular",
            "# meshunit: m",
            "# xbase: 0",
            "# ybase: 0",
            "# zbase: 0",
            "# xstepsize: 1e-9",
            "# ystepsize: 1e-9",
            "# zstepsize: 1e-9",
            f"# xnodes: {GRID_X}",
            f"# ynodes: {GRID_Y}",
            "# znodes: 1",
            "# valuedim: 3",
            "# valuelabels: m_x m_y m_z",
            "# valueunits: 1 1 1",
            f"# Desc: Total simulation time: {time_s:.17g} s",
            "# End: Header",
            "# Begin: Data Binary 8",
            "",
        )
    ).encode()
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(struct.pack("<d", 123456789012345.0))
        stream.write(np.asarray(vectors, dtype="<f8").tobytes())
        stream.write(b"\n# End: Data Binary 8\n# End: Segment\n")


def _vectors(frequency_hz: float, frame: int, *, opposite: bool) -> np.ndarray:
    phase = 2.0 * math.pi * frequency_hz * frame * DT_S
    # Circular transverse motion makes total transverse power exactly .01.
    x = AMPLITUDE * math.cos(phase)
    y = AMPLITUDE * math.sin(phase)
    result = np.empty((GRID_Y, GRID_X, 3), dtype=float)
    result[..., 0] = x
    result[..., 1] = y
    result[..., 2] = Z_COMPONENT
    if opposite:
        result[:, GRID_X // 2 :, :2] *= -1.0
    return result


def _make_case(root: Path, name: str, frequency_hz: float, *, opposite: bool = False) -> None:
    directory = root / name
    directory.mkdir()
    frames = []
    for frame in range(FRAME_COUNT):
        filename = f"m{frame:04d}.ovf"
        _write_ovf(directory / filename, _vectors(frequency_hz, frame, opposite=opposite), frame * DT_S)
        frames.append({"file": filename, "sequence": frame, "time_s": frame * DT_S})
    manifest = {
        "schema_version": 1,
        "integrity": "unchecked",
        "entity_id": name,
        "segment_id": "p4c-synthetic",
        "origin": "synthetic",
        "time_kind": "dynamics",
        "quantity": "magnetization_direction",
        "value_unit": "1",
        "components": ["x", "y", "z"],
        "mask": "all",
        "frames": frames,
    }
    (directory / "replay.json").write_text(json.dumps(manifest), encoding="utf-8")


def _case_root(tmp_path: Path) -> Path:
    root = tmp_path / "band-case"
    root.mkdir()
    _make_case(root, "in_phase", IN_BAND_HZ)
    _make_case(root, "opposite_phase", IN_BAND_HZ, opposite=True)
    _make_case(root, "out_band", OUT_BAND_HZ)
    return root


def _case_report(report: dict, name: str) -> dict:
    """Keep report-key expectations explicit at the validator boundary."""
    return report["cases"][name]


def test_band_case_accepts_three_complete_replays_and_preserves_local_power(tmp_path):
    report = validate_case(_case_root(tmp_path))

    assert report["status"] == "valid"
    assert report["frames"] == {"count": FRAME_COUNT, "dt_s": DT_S, "window": WINDOW}
    assert report["band"] == {"low_hz": 8e9, "high_hz": 12e9}
    in_phase = _case_report(report, "in_phase")
    opposite = _case_report(report, "opposite_phase")
    outside = _case_report(report, "out_band")
    assert in_phase["status"] == opposite["status"] == outside["status"] == "valid"
    assert in_phase["band"]["mean_power"] == pytest.approx(0.01, rel=2e-4, abs=2e-7)
    assert opposite["band"]["mean_power"] == pytest.approx(in_phase["band"]["mean_power"], rel=2e-4, abs=2e-7)
    assert opposite["opposition"]["spatial_mean_transverse_max"] == pytest.approx(0.0, abs=1e-14)
    assert outside["band"]["mean_power"] < 1e-8


def test_band_case_rejects_nonuniform_timestamps(tmp_path):
    root = _case_root(tmp_path)
    manifest_path = root / "in_phase" / "replay.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    altered_time = manifest["frames"][100]["time_s"] + 1e-15
    manifest["frames"][100]["time_s"] = altered_time
    # Keep the manifest and OVF header mutually consistent: this must fail on
    # the nonuniform replay window, rather than on a metadata disagreement.
    record = manifest["frames"][100]
    _write_ovf(root / "in_phase" / record["file"], _vectors(IN_BAND_HZ, 100, opposite=False), altered_time)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="uniform|timestamp|time|sampling"):
        validate_case(root)


def test_band_case_rejects_wrong_band_frequency(tmp_path):
    root = _case_root(tmp_path)
    directory = root / "in_phase"
    for frame in range(FRAME_COUNT):
        _write_ovf(directory / f"m{frame:04d}.ovf", _vectors(OUT_BAND_HZ, frame, opposite=False), frame * DT_S)
    with pytest.raises(ValueError, match="band|frequency|in_phase"):
        validate_case(root)


def test_band_case_rejects_missing_phase_opposition_despite_preserved_local_power(tmp_path):
    root = _case_root(tmp_path)
    directory = root / "opposite_phase"
    # Deliberately replace the opposite case with an in-phase wave.  A
    # validator that only checks frequency and power would accept this.
    for frame in range(FRAME_COUNT):
        _write_ovf(directory / f"m{frame:04d}.ovf", _vectors(IN_BAND_HZ, frame, opposite=False), frame * DT_S)
    with pytest.raises(ValueError, match="opposite|local|power|phase"):
        validate_case(root)


@pytest.mark.parametrize("mutation", ["incomplete", "missing"])
def test_band_case_rejects_incomplete_or_missing_window(tmp_path, mutation):
    root = _case_root(tmp_path)
    manifest_path = root / "in_phase" / "replay.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "incomplete":
        manifest["frames"] = manifest["frames"][: WINDOW - 1]
    else:
        manifest["frames"].pop(160)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="frame|window|sequence|gap|missing"):
        validate_case(root)
