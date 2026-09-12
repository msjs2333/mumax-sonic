"""Contract tests for the small, synthetic public MuMax3 case.

The fields here are deliberately generated in the test: they exercise the
manifest/table validator without requiring MuMax3 or a GPU.  The four frame
times are irregular so a validator that derives ``dt`` from frame numbers
cannot accidentally pass the activity check.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.validate_public_case import validate_case


TIMES = (0.0, 0.25, 0.75, 1.0)
OMEGA_RAD_S = 4.0
PHASE_RAD = 0.3
TILT_RAD = 0.7


def _mean_at(time_s: float) -> tuple[float, float, float]:
    """A great-circle rotation whose three XYZ means are nonzero."""
    angle = PHASE_RAD + OMEGA_RAD_S * time_s
    return (math.cos(angle), math.sin(angle) * math.cos(TILT_RAD),
            math.sin(angle) * math.sin(TILT_RAD))


def _write_ovf(path: Path, time_s: float) -> None:
    """Write a tiny full-XYZ, one-layer OVF."""
    values = np.zeros((1, 3, 3, 3), dtype="<f8")
    values[..., 0], values[..., 1], values[..., 2] = _mean_at(time_s)
    header = "\n".join(
        (
            "# OOMMF OVF 2.0",
            "# Segment count: 1",
            "# Begin: Segment",
            "# Begin: Header",
            "# meshtype: rectangular",
            "# meshunit: nm",
            "# xbase: 0",
            "# ybase: 0",
            "# zbase: 0",
            "# xstepsize: 1",
            "# ystepsize: 1",
            "# zstepsize: 1",
            "# xnodes: 3",
            "# ynodes: 3",
            "# znodes: 1",
            "# valuedim: 3",
            "# valuelabels: m_x m_y m_z",
            "# valueunits: 1",
            f"# Desc: Total simulation time: {time_s:.17g} s",
            "# End: Header",
            "# Begin: Data Binary 8",
            "",
        )
    ).encode()
    raw = (
        header
        + np.asarray(123456789012345.0, dtype="<f8").tobytes()
        + values.tobytes()
        + b"# End: Data Binary 8\n# End: Segment\n"
    )
    path.write_bytes(raw)


def _case(tmp_path, *, sequences=None):
    sequences = list(range(len(TIMES))) if sequences is None else list(sequences)
    records = []
    for sequence in sequences:
        path = tmp_path / f"m{sequence:06d}.ovf"
        _write_ovf(path, TIMES[sequence])
        records.append(
            {
                "file": path.name,
                "sequence": sequence,
            }
        )
    manifest = {
        "schema_version": 1,
        "entity_id": "m",
        "segment_id": "public-synthetic",
        "origin": "synthetic",
        "time_kind": "dynamics",
        "quantity": "magnetization_direction",
        "value_unit": "1",
        "components": ["x", "y", "z"],
        "mask": "all",
        "integrity": "unchecked",
        "frames": records,
    }
    manifest_path = tmp_path / "public-case.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _table(tmp_path, *, rows=None):
    """Write the MuMax3 ``table.txt`` format (time and mean XYZ)."""
    rows = [
        (TIMES[i], *_mean_at(TIMES[i]))
        for i in range(len(TIMES))
    ] if rows is None else rows
    path = tmp_path / "table.txt"
    with path.open("w", newline="", encoding="utf-8") as stream:
        stream.write("# t (s)\tmx ()\tmy ()\tmz ()\n")
        for row in rows:
            stream.write("\t".join(f"{float(value):.17g}" for value in row) + "\n")
    return path


def test_public_case_accepts_full_xyz_manifest_and_activity_table(tmp_path):
    manifest = _case(tmp_path)
    table = _table(tmp_path)

    result = validate_case(manifest, table)

    assert isinstance(result, dict)
    assert result["status"] == "valid"
    assert result["frames"]["count"] == 4
    assert result["activity"]["actual_observations"] == 3
    assert result["activity"]["mean_rad_s"]["min"] == pytest.approx(OMEGA_RAD_S)
    assert result["activity"]["mean_rad_s"]["max"] == pytest.approx(OMEGA_RAD_S)
    assert result["ovf_table"]["max_time_error_s"] == 0.0
    assert result["ovf_table"]["max_mean_component_error"]["mx"] == pytest.approx(0.0)
    assert result["ovf_table"]["max_mean_component_error"]["my"] == pytest.approx(0.0)
    assert result["ovf_table"]["max_mean_component_error"]["mz"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda rows: rows.__setitem__(1, (rows[1][0], rows[1][1] + 1.0, rows[1][2], rows[1][3])),
        lambda rows: rows.__setitem__(1, (0.70, rows[1][1], rows[1][2], rows[1][3])),
    ),
    ids=("mean-mismatch", "time-mismatch"),
)
def test_public_case_rejects_inconsistent_activity_table(tmp_path, mutation):
    manifest = _case(tmp_path)
    rows = [
        tuple(float(value) for value in line.split())
        for line in _table(tmp_path).read_text(encoding="utf-8").splitlines()[1:]
    ]
    mutation(rows)
    table = _table(tmp_path, rows=rows)

    with pytest.raises(ValueError):
        validate_case(manifest, table)


def test_public_case_rejects_missing_frame(tmp_path):
    manifest = _case(tmp_path, sequences=(0, 1, 3))
    valid_rows = [
        (TIMES[i], *_mean_at(TIMES[i]))
        for i in (0, 1, 3)
    ]
    table = _table(tmp_path, rows=valid_rows)

    with pytest.raises(ValueError):
        validate_case(manifest, table)


def test_public_case_incremental_reader_matches_four_frame_replay(tmp_path):
    from scripts.check_replay_incremental import check_incremental

    report = check_incremental(_case(tmp_path))

    assert report["frames_compared"] == 4
    assert report["exact_match"] is True
    assert report["diagnostics"]["read_frames_total"] == 4
    assert report["diagnostics"]["skipped_decode_frames_total"] == 0
    assert report["diagnostics"]["retained_frames"] == 2


def test_aggregation_clamps_one_ulp_rms_rounding_but_rejects_real_weight_violation(monkeypatch):
    """The closed attention-weight bound survives a floating reduction edge."""
    import mumax_sonic.aggregation as aggregation

    grid = aggregation.ContributionGrid(
        np.array([[0.0, 1.0, 2.0]]), np.zeros((1, 3)), np.ones((1, 3)), np.zeros((1, 3)),
        1.0, "test", "activity", "rad/s",
    )
    node = aggregation._Node("root", np.arange(3), np.ones(3), np.ones(3))
    point_weight = np.ones(3)
    original_dot = aggregation.np.dot

    def rounded_dot(left, right):
        value = original_dot(left, right)
        if left is node.masses and np.array_equal(right, point_weight ** 2):
            return float(node.masses.sum()) * (1.0 + 2.0 * np.finfo(float).eps)
        return value

    monkeypatch.setattr(aggregation.np, "dot", rounded_dot)
    observation = aggregation._observation(node, 1, "test", grid,
                                           grid.x_m.ravel(), grid.y_m.ravel(), point_weight)
    assert observation.attention_weight == 1.0

    with pytest.raises(ValueError, match="attention weight"):
        aggregation._observation(node, 1, "test", grid, grid.x_m.ravel(), grid.y_m.ravel(),
                                 np.full(3, 1.01))
