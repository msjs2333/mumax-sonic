"""Independent checks for the replay listening-check preparation contract.

The fixture is a tiny synthetic OVF replay.  It exercises physical replay
preparation only; no solver, GPU, or real audio device is involved.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest


def _write_ovf(path: Path, angle: float, time_s: float) -> None:
    values = np.zeros((1, 3, 3, 3), dtype="<f8")
    values[..., 0] = math.cos(angle)
    values[..., 1] = math.sin(angle)
    values[..., 2] = 0.0
    header = "\n".join(
        (
            "# OOMMF OVF 2.0", "# Segment count: 1", "# Begin: Segment",
            "# Begin: Header", "# meshtype: rectangular", "# meshunit: nm",
            "# xbase: 0", "# ybase: 0", "# zbase: 0", "# xstepsize: 1",
            "# ystepsize: 1", "# zstepsize: 1", "# xnodes: 3", "# ynodes: 3",
            "# znodes: 1", "# valuedim: 3", "# valuelabels: m_x m_y m_z",
            "# valueunits: 1", f"# Desc: Total simulation time: {time_s:.17g} s",
            "# End: Header", "# Begin: Data Binary 8", "",
        )
    ).encode()
    path.write_bytes(
        header + np.asarray(123456789012345.0, dtype="<f8").tobytes()
        + values.tobytes() + b"# End: Data Binary 8\n# End: Segment\n"
    )


def _make_replay(tmp_path: Path, count: int = 16, *, irregular: bool = True) -> Path:
    frames = []
    for index in range(count):
        # Deliberately use physical timestamps rather than frame-number timing.
        time_s = ((0.0, 0.1, 0.3)[index] if index < 3 else 0.3 + (index - 2) * 0.1) if irregular else index * 0.1
        filename = f"m{index:04d}.ovf"
        _write_ovf(tmp_path / filename, 0.5 * time_s, time_s)
        frames.append({"file": filename, "sequence": index})
    manifest = {
        "schema_version": 1, "integrity": "unchecked", "entity_id": "m", "segment_id": "listen-synthetic",
        "origin": "synthetic", "time_kind": "dynamics", "quantity": "magnetization_direction",
        "value_unit": "1", "components": ["x", "y", "z"], "mask": "all", "frames": frames,
    }
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _clip(manifest: Path, *, recipe="activity", frame_index=1, reference=1.0, band=None):
    value = {
        "name": "synthetic", "manifest": manifest.name, "recipe": recipe,
        "frame_index": frame_index, "reference": reference,
    }
    if band is not None:
        value["band"] = band
    return value


def _module():
    from scripts import replay_listening_check
    return replay_listening_check


def test_activity_clip_warms_first_frame_and_reports_physical_predecessor(tmp_path):
    module = _module()
    manifest = _make_replay(tmp_path)
    with pytest.raises(ValueError, match="warm|previous|predecessor"):
        module.prepare_clip(_clip(manifest, frame_index=0), tmp_path)
    scene, report = module.prepare_clip(_clip(manifest, frame_index=2), tmp_path)
    assert scene is not None
    assert report["frame_index"] == 2
    assert report["diagnostic"]["previous_sequence"] == 1
    assert report["diagnostic"]["dt_s"] == pytest.approx(0.2)
    assert report["diagnostic"]["mean_rad_s"] == pytest.approx(0.5)


def test_band_clip_warms_until_full_window(tmp_path):
    module = _module()
    manifest = _make_replay(tmp_path, irregular=False)
    band = {"low_hz": 1.0, "high_hz": 4.0, "window_samples": 16, "reference_axis": [0, 0, 1]}
    with pytest.raises(ValueError, match="warm|window"):
        module.prepare_clip(_clip(manifest, recipe="band", frame_index=1, band=band), tmp_path)
    scene, report = module.prepare_clip(_clip(manifest, recipe="band", frame_index=15, band=band), tmp_path)
    assert scene is not None
    assert report["frame_index"] == 15
    assert report["diagnostic"]["samples_required"] == 16


@pytest.mark.parametrize("field,value", [("frame_index", -1), ("reference", 0.0), ("reference", -1.0)])
def test_clip_rejects_invalid_index_and_reference(tmp_path, field, value):
    module = _module()
    manifest = _make_replay(tmp_path)
    kwargs = {field: value}
    with pytest.raises((TypeError, ValueError)):
        module.prepare_clip(_clip(manifest, **kwargs), tmp_path)


def test_default_cli_never_opens_audio(monkeypatch, tmp_path, capsys):
    module = _module()

    class FailingEngine:
        def open(self, *_args, **_kwargs):
            raise AssertionError("default replay listening check must stay silent")

    try:
        import mumax_sonic.audio as audio
    except ImportError:
        pytest.skip("audio backend unavailable")
    monkeypatch.setattr(audio, "AudioEngine", FailingEngine)
    manifest = _make_replay(tmp_path)
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps({"clips": [_clip(manifest)]}), encoding="utf-8")
    args = module.parser().parse_args([str(playlist)])
    assert module.run(args) == 0
    assert "validated 1 frozen clip" in capsys.readouterr().out.lower()


def test_playlist_manifest_paths_are_resolved_relative_to_playlist(tmp_path):
    module = _module()
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    manifest = _make_replay(replay_dir)
    playlist = tmp_path / "playlist.json"
    clip = _clip(manifest)
    clip["manifest"] = "replays/replay.json"
    playlist.write_text(json.dumps({"clips": [clip]}), encoding="utf-8")
    args = module.parser().parse_args([str(playlist)])
    result = module.run(args)
    assert result == 0


def test_playlist_accepts_topology_recipe_and_relative_manifest(tmp_path):
    module = _module()
    manifest = _make_replay(tmp_path)
    clip = _clip(manifest, recipe="topology", frame_index=0)
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps({"clips": [clip]}), encoding="utf-8")
    loaded = module.load_playlist(playlist)
    assert loaded[0]["recipe"] == "topology"
