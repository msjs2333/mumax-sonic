"""Bounded, frozen-frame listening preview for declared OVF replay clips.

This utility deliberately prepares every physical observation before it opens
OpenAL.  Playback repeats a precomputed scene only to keep the renderer's
stale-scene watchdog fed; it never advances a solver, reads a replay, or turns
the frozen scene into a timeline audition.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


_RECIPES = {"activity", "band", "topology"}
_MASTER_GAIN = 0.2
_BUDGET = 8


def _require(value, name, kind):
    if not isinstance(value, kind):
        raise ValueError(f"{name} must be a {kind.__name__}")
    return value


def load_playlist(path: Path) -> tuple[dict, ...]:
    """Validate the small portable playlist format and return its clips."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or set(payload) != {"clips"}:
        raise ValueError("playlist must be an object containing only clips")
    clips = payload["clips"]
    if not isinstance(clips, list) or not clips:
        raise ValueError("playlist clips must be a nonempty list")
    result = []
    names = set()
    for number, clip in enumerate(clips):
        if not isinstance(clip, dict):
            raise ValueError(f"clips[{number}] must be an object")
        allowed = {"name", "manifest", "recipe", "frame_index", "reference", "band"}
        if set(clip) - allowed or not {"name", "manifest", "recipe", "frame_index", "reference"} <= set(clip):
            raise ValueError(f"clips[{number}] has missing or unknown keys")
        name = _require(clip["name"], f"clips[{number}].name", str)
        if not name or name in names:
            raise ValueError("clip names must be nonempty and unique")
        names.add(name)
        manifest = _require(clip["manifest"], f"clips[{number}].manifest", str)
        if not manifest:
            raise ValueError("clip manifest must be a nonempty path")
        recipe = clip["recipe"]
        if recipe not in _RECIPES:
            raise ValueError("clip recipe must be activity, band, or topology")
        index = clip["frame_index"]
        if type(index) is not int or index < 0:
            raise ValueError("clip frame_index must be a nonnegative integer")
        reference = clip["reference"]
        if (isinstance(reference, bool) or not isinstance(reference, (int, float))
                or not math.isfinite(reference) or reference <= 0):
            raise ValueError("clip reference must be finite and positive")
        band = clip.get("band")
        if recipe != "band" and band is not None:
            raise ValueError("band settings are only valid for the band recipe")
        if band is not None:
            if not isinstance(band, dict) or set(band) != {"low_hz", "high_hz", "window_samples", "reference_axis"}:
                raise ValueError("band must declare low_hz, high_hz, window_samples, and reference_axis")
        result.append(dict(clip, reference=float(reference)))
    return tuple(result)


def _band_config(clip: dict):
    if clip.get("band") is None:
        return None
    from mumax_sonic.observers.band import BandConfig
    band = clip["band"]
    return BandConfig(band["low_hz"], band["high_hz"], band["window_samples"], band["reference_axis"])


def _attention_for(frame):
    from mumax_sonic.attention import Attention
    # The physical viewport is the complete stored plane.  Attention only
    # changes representative voices; it never crops the observer's integral.
    return Attention(center=(0.0, 0.0), radius=0.55, background=0.15,
                     extent_m=frame.extent_m, origin_m=frame.center_m[:2])


def prepare_clip(clip, base: Path) -> tuple[object, dict]:
    """Build one frozen valid scene and its evidence without opening audio."""
    from mumax_sonic.field_pipeline import apply_aggregation, field_selection_report, observe_field
    from mumax_sonic.mapping import map_sample_with_report
    from mumax_sonic.sources.ovf_replay import load_ovf_replay

    if not isinstance(clip, dict):
        raise TypeError("clip must be a validated playlist object")
    for key in ("name", "manifest", "recipe", "frame_index", "reference"):
        if key not in clip:
            raise ValueError(f"clip is missing {key}")
    if clip["recipe"] not in _RECIPES:
        raise ValueError("clip recipe must be activity, band, or topology")
    if type(clip["frame_index"]) is not int or clip["frame_index"] < 0:
        raise ValueError("clip frame_index must be a nonnegative integer")
    if (isinstance(clip["reference"], bool) or not isinstance(clip["reference"], (int, float))
            or not math.isfinite(clip["reference"]) or clip["reference"] <= 0):
        raise ValueError("clip reference must be finite and positive")
    manifest_path = Path(clip["manifest"])
    manifest = manifest_path if manifest_path.is_absolute() else Path(base) / manifest_path
    replay = load_ovf_replay(manifest)
    index = clip["frame_index"]
    if index >= len(replay.frames):
        raise ValueError(f"clip {clip['name']}: frame_index {index} is outside {len(replay.frames)} frames")
    frame = replay.frames[index]
    recipe = clip["recipe"]
    kwargs = {}
    if recipe == "activity":
        kwargs["previous"] = replay.frames[index - 1] if index else None
    elif recipe == "band":
        config = _band_config(clip)
        kwargs["band_config"] = config
        required = config.window_samples if config is not None else 256
        kwargs["history"] = replay.frames[max(0, index - required + 1):index + 1]
    view = observe_field(frame, recipe, **kwargs)
    if view.sample.validity != "valid" or view.sample.coverage != 1:
        reason = view.diagnostic.get("reason", "observation is not ready or not fully valid")
        raise ValueError(f"clip {clip['name']}: {view.sample.validity}: {reason}")
    attention = _attention_for(frame)
    sound_view = apply_aggregation(view, attention, budget=_BUDGET, mode="adaptive")
    adaptive = sound_view.diagnostic.get("adaptive_aggregation", {})
    if adaptive.get("status") != "valid":
        raise ValueError(f"clip {clip['name']}: adaptive aggregation unavailable: {adaptive.get('reason')}")
    mapped = map_sample_with_report(sound_view.sample, attention, master_gain=1.0,
                                    budget=_BUDGET, strength_reference=clip["reference"])
    scene = replace(mapped.scene, sources=tuple(
        replace(source, gain=source.gain * _MASTER_GAIN) for source in mapped.scene.sources))
    mapping = field_selection_report(sound_view, mapped.report)
    ovf_input = view.diagnostic.get("input")
    if not isinstance(ovf_input, dict) or "origin" not in ovf_input:
        raise ValueError(f"clip {clip['name']}: physical origin is missing from OVF manifest provenance")
    report = {
        "name": clip["name"], "manifest": str(manifest), "recipe": recipe,
        "frame_index": index, "frame_count": len(replay.frames),
        "physical_time_s": frame.sim_time_s, "sequence": frame.sequence,
        "source_kind": frame.source_kind, "physical_origin": ovf_input["origin"],
        "quantity": {"entity_id": frame.entity_id, "input_quantity": ovf_input.get("quantity"),
                     "input_unit": ovf_input.get("input_unit"),
                     "observed_quantity": next((o.quantity for o in sound_view.sample.observations), None),
                     "observed_unit": next((o.unit for o in sound_view.sample.observations), None)},
        "strength_reference": clip["reference"], "band": clip.get("band"),
        "diagnostic": sound_view.diagnostic, "mapping": mapping,
        "scene": {"source_count": len(scene.sources), "budget": _BUDGET,
                  "master_gain": _MASTER_GAIN,
                  "gains": [{"source_id": source.source_id, "gain": source.gain,
                             "sign": source.sign} for source in scene.sources]},
        "preview_limit": "frozen final-frame scene; repeated submission holds audio only and does not audition dynamic motion or a full timeline",
    }
    if recipe == "activity":
        report.update(predecessor_frame_index=index - 1, dt_s=view.diagnostic.get("dt_s"),
                      mean_rad_s=view.diagnostic.get("mean_rad_s"))
    if recipe == "band":
        report["window_samples"] = view.diagnostic.get("samples_required")
    return scene, report


def _play_scene(engine, scene, seconds: float) -> dict:
    """Keep one immutable scene fresh; only renderer time advances here."""
    started = time.monotonic()
    deadline = started + seconds
    while time.monotonic() < deadline:
        engine.update(scene)
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    diagnostic = engine.diagnostics()
    submitted = (
        diagnostic.get("last_control_applied_at_s") is not None
        and diagnostic["last_control_applied_at_s"] >= started
        and diagnostic.get("last_applied_sim_time_s") == scene.sim_time_s)
    healthy = (submitted and diagnostic.get("state") == "open"
               and diagnostic.get("connected") is True and not diagnostic.get("last_error"))
    return {"started_at_monotonic_s": started, "renderer_diagnostic": diagnostic,
            "control_submission_observed": (
                submitted), "renderer_healthy": healthy}


def _write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run(args: argparse.Namespace) -> int:
    playlist_path = Path(args.playlist)
    clips = load_playlist(playlist_path)
    if args.report is not None and Path(args.report).exists():
        raise FileExistsError(f"refusing to overwrite report: {args.report}")
    # All file I/O and physics finish before any audio worker is opened.
    prepared = [prepare_clip(clip, playlist_path.parent) for clip in clips]
    report = {"playlist": str(playlist_path), "play_requested": args.play,
              "seconds_per_clip": args.seconds, "human_listening_verified": False,
              "clips": [item[1] for item in prepared], "engine": None,
              "note": "This is a device playback record, not a human listening verification."}
    engine = None
    exit_code = 0
    try:
        if args.play:
            from mumax_sonic.audio import AudioConfig, AudioEngine
            engine = AudioEngine()
            engine.open(AudioConfig(hrtf=True))
            report["engine"] = {"opened": engine.diagnostics(), "clips": []}
            for scene, clip_report in prepared:
                print(f"playing {clip_report['name']}", flush=True)
                playback = _play_scene(engine, scene, args.seconds)
                clip_report["playback"] = playback
                report["engine"]["clips"].append({"name": clip_report["name"], **playback})
                engine.stop()
                if not playback["renderer_healthy"]:
                    raise RuntimeError(f"audio renderer did not confirm clip {clip_report['name']}")
                time.sleep(0.3)
        else:
            print(f"validated {len(prepared)} frozen clip(s); pass --play to open audio")
    except KeyboardInterrupt:
        report["interrupted"] = True
        exit_code = 130
        print("Interrupted; closing audio.")
    except Exception as exc:
        report["playback_error"] = str(exc)
        exit_code = 1
    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception as exc:
                report["close_error"] = str(exc)
                exit_code = 1
            if report["engine"] is None:
                report["engine"] = {}
            report["engine"]["closed"] = engine.diagnostics()
    if args.report is not None:
        _write_report(Path(args.report), report)
        print(f"saved {args.report}")
    if exit_code and "playback_error" in report:
        print(f"playback failed: {report['playback_error']}", file=sys.stderr)
    return exit_code


def _seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seconds must be a number") from exc
    if not 1 <= seconds <= 10:
        raise argparse.ArgumentTypeError("seconds must be in [1, 10]")
    return seconds


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Prepare or audibly preview frozen OVF replay frames")
    result.add_argument("playlist", help="playlist JSON")
    result.add_argument("--play", action="store_true", help="open OpenAL and play each prepared frozen scene")
    result.add_argument("--report", help="new JSON report path; existing files are never overwritten")
    result.add_argument("--seconds", type=_seconds, default=3.0, help="seconds per clip, 1 through 10 (default: 3)")
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
