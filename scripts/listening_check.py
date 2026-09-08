"""Blind P1 listening checks.

The default invocation is silent and only prints a trial plan.  Pass ``--play``
to opt into opening the configured OpenAL device and entering responses.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@dataclass(frozen=True)
class Trial:
    trial_id: str
    task: str
    answer: str
    x_m: float
    sign: int


def make_trials(seed: int = 7, count: int = 20) -> tuple[Trial, ...]:
    """Return a deterministic balanced set: 20 left/right and 20 sign trials."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    result: list[Trial] = []
    for task, options in (("left_right", ("l", "r")), ("sign", ("+", "-"))):
        answers = list(options) * (count // 2)
        if count % 2:
            answers.append(options[0])
        rng.shuffle(answers)
        for index, answer in enumerate(answers, 1):
            x_m = 0.0 if task == "sign" else (-0.8e-6 if answer == "l" else 0.8e-6)
            result.append(Trial(f"{task}-{index:02d}", task, answer, x_m,
                                1 if answer == "+" else -1))
    return tuple(result)


def score_response(trial: Trial, response: str) -> bool:
    response = response.strip().lower()
    return response in {"l", "r", "+", "-"} and response == trial.answer


def unique_path(directory: Path, prefix: str, suffix: str = ".csv") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for _ in range(100):
        candidate = directory / f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("could not allocate a unique result filename")


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["trial_id", "task", "seed", "correct_answer", "gain",
              "headphones", "device", "hrtf", "system_volume_note",
              "response", "correct", "task_correct", "task_total",
              "response_time_s", "notes"]
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    trials = make_trials(args.seed, args.count)
    if not args.play:
        print(f"silent plan: {len(trials)} trials; pass --play to start audio")
        return 0
    print("Synthetic/demo only. Set a comfortable volume before continuing.")
    headphones = args.headphones or input("Headphones: ").strip()
    volume = args.system_volume_note or input("System volume (optional note): ").strip()
    if args.calibrate:
        print("Calibration: use the device volume controls; the default gain is 0.05.")
        input("Press Enter when comfortable...")
    engine = None
    rows: list[dict[str, object]] = []
    try:
        from mumax_sonic.attention import Attention
        from mumax_sonic.mapping import map_sample
        from mumax_sonic.model import Observation, Sample
        if args.play:
            from mumax_sonic.audio.openal import AudioConfig, AudioEngine
            engine = AudioEngine()
            engine.open(AudioConfig(device_name=args.device, hrtf=True))
        diagnostics = engine.diagnostics() if engine else {}
        hrtf = diagnostics.get("hrtf_status", "not played")
        device = diagnostics.get("device", args.device or "not played")
        gain = 0.05
        if args.calibrate:
            print("Calibration: labelled short tones (+, -, l, r); use device controls for a comfortable level.")
            for label in ("l", "r", "+", "-"):
                calibration = Trial("calibration", "calibration", label,
                                    -0.8e-6 if label == "l" else (0.8e-6 if label == "r" else 0.0),
                                    1 if label == "+" else -1)
                print(f"Label {label}", flush=True)
                if engine:
                    _play_trial(engine, calibration, duration_s=0.25, gain=gain)
                input(f"Label {label}; press Enter when heard...")
        for trial in trials:
            observation = Observation("blind", (trial.x_m, 0.0, 0.0), 0.7, trial.sign)
            sample = Sample(0.0, (observation,), segment_id=trial.task, source_kind="synthetic")
            if engine:
                _play_trial(engine, trial, duration_s=1.0, gain=gain,
                            sample=sample, attention=Attention(radius=2.0, background=1.0),
                            map_sample_fn=map_sample)
            started = time.monotonic()
            response = input(f"{trial.trial_id}: response (l/r or +/-): ")
            elapsed = time.monotonic() - started
            if engine:
                engine.stop()
            rows.append({"trial_id": trial.trial_id, "task": trial.task, "seed": args.seed,
                         "correct_answer": trial.answer, "gain": gain,
                         "headphones": headphones, "device": device, "hrtf": hrtf,
                         "system_volume_note": volume,
                         "response": response.strip().lower(),
                         "correct": score_response(trial, response),
                         "task_correct": "", "task_total": "",
                         "response_time_s": round(elapsed, 3), "notes": ""})
    except KeyboardInterrupt:
        print("Interrupted; saving completed trials.")
    finally:
        if engine is not None:
            engine.close()
    for task in ("left_right", "sign"):
        task_rows = [row for row in rows if row["task"] == task]
        task_correct = sum(bool(row["correct"]) for row in task_rows)
        for row in task_rows:
            row["task_correct"], row["task_total"] = task_correct, len(task_rows)
        if task_rows:
            print(f"{task}: {task_correct}/{len(task_rows)} = {task_correct / len(task_rows):.1%}")
    path = unique_path(Path(args.output), "listening_check")
    _write_rows(path, rows)
    print(f"saved {path} ({len(rows)} completed trials)")
    return 0


def _play_trial(engine, trial, duration_s=1.0, gain=0.05, *, sample=None,
                attention=None, map_sample_fn=None):
    """Keep a scene fresh while a one second trial is audible."""
    if sample is None:
        from mumax_sonic.model import Observation, Sample
        sample = Sample(0.0, (Observation("blind", (trial.x_m, 0.0, 0.0), 0.7, trial.sign),),
                        segment_id=trial.task, source_kind="synthetic")
    from mumax_sonic.attention import Attention
    from mumax_sonic.mapping import map_sample
    attention = attention or Attention(radius=2.0, background=1.0)
    mapper = map_sample_fn or map_sample
    scene = mapper(sample, attention, master_gain=gain, budget=1)
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        engine.update(scene)
        time.sleep(0.025)
    engine.stop()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Blind synthetic spatial/sign listening check")
    p.add_argument("--play", action="store_true", help="opt in to real OpenAL playback")
    p.add_argument("--calibrate", action="store_true", help="pause for manual volume calibration")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--count", type=int, default=20, help="trials per task")
    p.add_argument("--output", default="local/listening_trials")
    p.add_argument("--device")
    p.add_argument("--headphones")
    p.add_argument("--system-volume-note", help="Uncalibrated system volume setting or note")
    return p


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
