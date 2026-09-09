"""Reproducible paired MuMax+ solver/capture benchmark (GPU run required)."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=4, help="number of paired cases (default: 4)")
    parser.add_argument("--steps", type=int, default=80, help="timed solver steps per case (default: 80)")
    parser.add_argument("--size", type=int, default=16, help="XY grid size (default: 16)")
    parser.add_argument("--sample-every", type=int, default=4, dest="sample_every",
                        help="capture every N timed steps (default: 4)")
    parser.add_argument("--seed", type=int, default=0, help="case-order seed (default: 0)")
    parser.add_argument("--report", type=Path, help="write the JSON report to this path")
    return parser


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not 1 <= args.pairs <= 1000:
        parser.error("pairs must be in 1..1000")
    if not 1 <= args.steps <= 100000:
        parser.error("steps must be in 1..100000")
    if not 3 <= args.size <= 4096:
        parser.error("size must be in 3..4096")
    if not 1 <= args.sample_every <= args.steps:
        parser.error("sample-every must be in 1..steps")


def _rss_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _make_case(size: int):
    """Construct the complete physical case; called only by its owner thread."""
    from mumaxplus import Ferromagnet, Grid, World

    world = World(cellsize=(5e-9,) * 3)
    magnet = Ferromagnet(world, Grid((size, size, 1)))
    magnet.msat = 8e5
    magnet.aex = 13e-12
    magnet.alpha = 0.02
    magnet.enable_demag = False
    magnet.magnetization = (1, 0, 0)
    world.bias_magnetic_field = (0, 0, 0.1)
    return world, magnet


def _run_case(size: int, steps: int, sample_every: int, monitored: bool) -> dict:
    from mumax_sonic.sources.live import LiveConfig
    from mumax_sonic.sources.mumaxplus import MuMaxPlusSampler
    from mumax_sonic.sources.stream import FrameStream

    world, magnet = _make_case(size)
    stream = None
    sampler = None
    if monitored:
        stream = FrameStream(LiveConfig(recipe="activity", stale_after_s=30.0)).start()
        sampler = MuMaxPlusSampler(world, magnet, entity_id="m", segment_id="benchmark")

    # One solver call is deliberately outside all measurements.
    world.timesolver.run(5e-12)
    if sampler:
        sampler.sample(0)
    else:
        # Match the monitored warmup's final quantity eval without creating a frame.
        magnet.magnetization.eval()

    captures = []
    capture_count = 0
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    try:
        for step in range(1, steps + 1):
            world.timesolver.run(5e-12)
            if sampler and (step % sample_every == 0 or step == steps):
                capture_count += 1
                frame = sampler.sample(capture_count)
                captures.append(sampler.last_capture_ms)
                stream.submit(frame)
        # Final eval is part of solver_loop timing and synchronizes both cases.
        final_xyz = np.array(magnet.magnetization.eval(), copy=True)
        end_wall = time.perf_counter()
        end_cpu = time.process_time()
        if sampler:
            stream.finish()
            drain_start = time.monotonic()
            deadline = drain_start + max(10.0, steps * 0.25)
            while time.monotonic() < deadline:
                snapshot = stream.snapshot()
                if snapshot["state"] == "finished":
                    break
                if snapshot["state"] == "invalid":
                    raise RuntimeError(snapshot.get("last_error") or "FrameStream observer failed")
                time.sleep(0.001)
            else:
                raise TimeoutError("FrameStream observer drain timed out")
            drain_s = time.monotonic() - drain_start
            stream_snapshot = stream.snapshot()
            stream_snapshot.pop("view", None)
        else:
            drain_s = 0.0
            stream_snapshot = None
        return {
            "monitored": monitored,
            "solver_loop_wall_s": end_wall - start_wall,
            "solver_loop_cpu_s": end_cpu - start_cpu,
            "observer_drain_wall_s": drain_s if monitored else None,
            "captures": len(captures),
            "capture_median_ms": (float(np.median(captures)) if captures else None),
            "capture_rate_hz": (len(captures) / (end_wall - start_wall)) if captures and end_wall > start_wall else None,
            "actual_capture_fps": (len(captures) / (end_wall - start_wall)) if captures and end_wall > start_wall else None,
            "sim_time_s": float(world.timesolver.time),
            "final_xyz_sha256": hashlib.sha256(final_xyz.tobytes()).hexdigest(),
            "_final_xyz": final_xyz,
            "rss_bytes": _rss_bytes(),
            "stream": stream_snapshot,
        }
    finally:
        if stream is not None:
            stream.close()


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    _validate(args, parser)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mumax_sonic.reporting import report_json
    from mumax_sonic.sources.mumaxplus import probe_mumaxplus

    order_rng = random.Random(args.seed)
    cases = []
    started = time.perf_counter()
    try:
        for pair in range(args.pairs):
            order = ["baseline", "monitored"]
            order_rng.shuffle(order)
            results = {name: _run_case(args.size, args.steps, args.sample_every, name == "monitored") for name in order}
            baseline, monitored = results["baseline"], results["monitored"]
            base_wall = baseline["solver_loop_wall_s"]
            final_difference = float(np.max(np.abs(baseline.pop("_final_xyz") - monitored.pop("_final_xyz"))))
            cases.append({
                "pair": pair,
                "order": order,
                "baseline": baseline,
                "monitored": monitored,
                "solver_wall_overhead_fraction": (monitored["solver_loop_wall_s"] / base_wall - 1.0) if base_wall else None,
                "solver_wall_overhead_s": monitored["solver_loop_wall_s"] - base_wall,
                "final_xyz_max_abs_difference": final_difference,
            })
    except Exception as exc:
        report = {"error": str(exc), "backend": probe_mumaxplus(), "pairs_completed": len(cases),
                  "parameters": {key: str(value) if isinstance(value, Path) else value
                                 for key, value in vars(args).items()},
                  "device_measurement": False,
                  "note": "Real MuMax+ GPU model case failed before completion; no device/listening measurement."}
        encoded = report_json(report)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(encoded, encoding="utf-8")
        print(encoded)
        return 1
    report = {
        "benchmark": "mumaxplus_paired_solver_capture",
        "backend": probe_mumaxplus(),
        "parameters": {"pairs": args.pairs, "steps": args.steps, "size": args.size,
                       "cellsize_m": [5e-9] * 3, "dt_s": 5e-12, "sample_every": args.sample_every,
                       "sample_frequency_hz": 1.0 / (args.sample_every * 5e-12), "seed": args.seed,
                       "demag": False, "msat_A_m": 8e5, "aex_J_m": 13e-12, "alpha": 0.02,
                       "field_T": [0, 0, 0.1], "initial_m": [1, 0, 0]},
        "warmup_steps": 1,
        "cases": cases,
        "elapsed_wall_s": time.perf_counter() - started,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "host": {"platform": sys.platform, "python": sys.version, "machine": __import__("platform").platform()},
        "device_measurement": False,
        "note": "Real MuMax+ GPU model case timing; no device/listening measurement. Solver loop wall/CPU timing includes the final magnetization.eval() synchronization. Observer drain is reported separately. Overhead is descriptive; no 5% pass claim is made.",
    }
    encoded = report_json(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
