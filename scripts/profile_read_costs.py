"""Profile one real live-OVF decode without changing production code.

This is deliberately a local diagnostic.  It parses only the final manifest
record, uses the shared live decoder with hash checking disabled, and records
the nested timings that explain its cost.  The collector is also importable by
continuous-read harnesses; every sample carries its calling thread name.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Any
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mumax_sonic.sources import ovf_replay


@dataclass(frozen=True)
class Timing:
    name: str
    elapsed_ns: int
    thread: str
    path: str | None = None
    requested_bytes: int | None = None
    returned_bytes: int | None = None


class _TimedStream:
    """A transparent binary stream proxy which times whole ``read`` calls."""

    def __init__(self, stream, collector: "ProfileCollector", path: Path):
        self._stream = stream
        self._collector = collector
        self._path = str(path)

    def read(self, size: int = -1):
        started = time.perf_counter_ns()
        result = self._stream.read(size)
        self._collector.record("file_read", time.perf_counter_ns() - started,
                               path=self._path, requested_bytes=size,
                               returned_bytes=len(result))
        return result

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, *args):
        return self._stream.__exit__(*args)


class ProfileCollector:
    """Thread-labelled timing collector for production-equivalent OVF reads.

    Use ``with collector.instrument():`` around the operation of interest.
    It patches only this process and restores all production symbols on exit.
    """

    def __init__(self):
        self.samples: list[Timing] = []
        self._lock = threading.Lock()
        self._mask_paths: set[str] = set()

    def record(self, name: str, elapsed_ns: int, *, path: str | None = None,
               requested_bytes: int | None = None, returned_bytes: int | None = None):
        item = Timing(name, elapsed_ns, threading.current_thread().name, path,
                      requested_bytes, returned_bytes)
        with self._lock:
            self.samples.append(item)

    def instrument(self):
        collector = self
        original_open = Path.open
        original_stat = Path.stat
        original_read_ovf = ovf_replay.read_ovf
        original_bounded_bytes = ovf_replay._bounded_bytes
        from mumax_sonic.sources import ovf
        original_binary = ovf._read_binary
        original_frame = ovf_replay.FieldFrame
        original_fstat = ovf_replay.os.fstat

        def timed_open(path, *args, **kwargs):
            started = time.perf_counter_ns()
            stream = original_open(path, *args, **kwargs)
            collector.record("file_open", time.perf_counter_ns() - started, path=str(path))
            mode = args[0] if args else kwargs.get("mode", "r")
            return _TimedStream(stream, collector, Path(path)) if "b" in mode else stream

        def timed_stat(path, *args, **kwargs):
            started = time.perf_counter_ns()
            result = original_stat(path, *args, **kwargs)
            collector.record("path_stat", time.perf_counter_ns() - started, path=str(path))
            return result

        def timed_fstat(fd):
            started = time.perf_counter_ns()
            result = original_fstat(fd)
            collector.record("file_fstat", time.perf_counter_ns() - started)
            return result

        def timed_bounded_bytes(path, limit):
            started = time.perf_counter_ns()
            result = original_bounded_bytes(path, limit)
            normalized = os.path.normcase(os.path.abspath(os.fspath(path)))
            if normalized in collector._mask_paths:
                collector.record("mask_read_total", time.perf_counter_ns() - started,
                                 path=normalized, returned_bytes=len(result))
            return result

        def timed_read_ovf(*args, **kwargs):
            started = time.perf_counter_ns()
            result = original_read_ovf(*args, **kwargs)
            collector.record("read_ovf", time.perf_counter_ns() - started,
                             path=str(args[0]) if args else None)
            return result

        def timed_binary(*args, **kwargs):
            started = time.perf_counter_ns()
            result = original_binary(*args, **kwargs)
            collector.record("payload_decode_binary", time.perf_counter_ns() - started)
            return result

        def timed_frame(*args, **kwargs):
            started = time.perf_counter_ns()
            result = original_frame(*args, **kwargs)
            collector.record("fieldframe_construct", time.perf_counter_ns() - started)
            return result

        stack = ExitStack()
        stack.enter_context(patch.object(Path, "open", timed_open))
        stack.enter_context(patch.object(Path, "stat", timed_stat))
        stack.enter_context(patch.object(ovf_replay.os, "fstat", timed_fstat))
        stack.enter_context(patch.object(ovf_replay, "_bounded_bytes", timed_bounded_bytes))
        stack.enter_context(patch.object(ovf_replay, "read_ovf", timed_read_ovf))
        stack.enter_context(patch.object(ovf, "_read_binary", timed_binary))
        stack.enter_context(patch.object(ovf_replay, "FieldFrame", timed_frame))
        return stack

    def snapshot(self) -> list[Timing]:
        with self._lock:
            return list(self.samples)

    def summary_by_thread(self) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[Timing]] = defaultdict(list)
        for sample in self.snapshot():
            grouped[sample.thread].append(sample)
        return {thread: _summary(samples) for thread, samples in sorted(grouped.items())}

    def write_report(self, path: str | Path, **extra: Any) -> dict[str, Any]:
        """Write a local JSON snapshot, convenient for continuous harnesses."""
        samples = self.snapshot()
        report = dict(timings=_summary(samples),
                      timings_by_thread=self.summary_by_thread(),
                      samples=[asdict(sample) for sample in samples], **extra)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report


def _summary(samples: list[Timing]) -> dict[str, Any]:
    groups: dict[str, list[Timing]] = defaultdict(list)
    for sample in samples:
        groups[sample.name].append(sample)
    result = {}
    for name, values in sorted(groups.items()):
        elapsed = [item.elapsed_ns / 1e6 for item in values]
        entry: dict[str, Any] = dict(calls=len(values), median_ms=statistics.median(elapsed),
                                     p95_ms=_percentile(elapsed, 95), total_ms=sum(elapsed),
                                     threads=sorted({item.thread for item in values}))
        reads = [item for item in values if item.returned_bytes is not None]
        if reads:
            entry["requested_bytes"] = sum(item.requested_bytes or 0 for item in reads)
            entry["returned_bytes"] = sum(item.returned_bytes or 0 for item in reads)
            entry["paths"] = sorted({item.path for item in reads})
        result[name] = entry
    return result


def _percentile(values: list[float], percentile: int) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100
    low, high = int(rank), min(int(rank) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def decode_last_manifest_frame(manifest: str | Path, collector: ProfileCollector):
    """Decode one final record through the live shared parser, without hashes."""
    manifest = Path(manifest)
    raw = ovf_replay._bounded_bytes(manifest, 8 * 1024 * 1024)
    meta = json.loads(raw, object_pairs_hook=ovf_replay._unique_keys)
    records = meta.get("frames")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest must contain at least one frame")
    selected = dict(meta)
    selected["frames"] = [records[-1]]
    mask_spec = meta.get("mask")
    if isinstance(mask_spec, dict) and isinstance(mask_spec.get("file"), str):
        collector._mask_paths.add(os.path.normcase(os.path.abspath(
            os.fspath(manifest.parent / mask_spec["file"]))))
    started = time.perf_counter_ns()
    frame = next(ovf_replay._iter_ovf_manifest(manifest, selected, raw,
                                                cumulative_limit=False,
                                                verify_hashes=False))
    collector.record("last_frame_parse_total", time.perf_counter_ns() - started,
                     path=str(manifest))
    return frame, records[-1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--report", type=Path, required=True, help="local JSON report path")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    collector = ProfileCollector()
    chosen = None
    with collector.instrument():
        for _ in range(args.repeats):
            frame, chosen = decode_last_manifest_frame(args.manifest, collector)
    report = collector.write_report(args.report,
        method="shared _iter_ovf_manifest single final frame; verify_hashes=False",
        manifest=str(args.manifest.resolve()),
        repeats=args.repeats,
        selected_sequence=chosen.get("sequence") if chosen else None,
        selected_file=chosen.get("file") if chosen else None,
        field_shape=list(frame.vectors.shape),
        limitations=[
            "All reported sections are nested and must not be added: file I/O and payload decode occur inside read_ovf, and read_ovf plus FieldFrame construction occur inside last_frame_parse_total.",
            "file_fstat records the os.fstat calls made by _bounded_bytes; path_stat separately records Path.stat calls.",
            "Only the final manifest record is decoded; the manifest JSON itself is read and parsed each repeat, while mask metadata is retained.",
            "verify_hashes=False intentionally matches the trusted live path; this report contains no content hashes.",
        ],
    )
    print(json.dumps(dict(manifest=report["manifest"], repeats=args.repeats,
                          selected_sequence=report["selected_sequence"], timings=report["timings"]),
          ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
