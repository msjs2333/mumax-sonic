"""Read-only OVF catch-up profile; no solver, publication changes, or audio."""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mumax_sonic.sources.incremental import IncrementalOVFReader
from mumax_sonic.sources.live import LiveFollower, LiveConfig
from mumax_sonic.reporting import report_json


def measure(reader, path):
    profile = cProfile.Profile()
    started = time.perf_counter()
    profile.enable()
    replay = reader.load(path)
    profile.disable()
    elapsed = (time.perf_counter()-started)*1000
    calls = [dict(function=f'{Path(filename).name}:{line}({name})', calls=total,
                  self_ms=own*1000, cumulative_ms=cumulative*1000)
             for (filename, line, name), (_, total, own, cumulative, _) in pstats.Stats(profile).stats.items()]
    return dict(elapsed_ms=elapsed, reader=reader.diagnostics,
                final_sequence=replay.frames[-1].sequence,
                top_calls=sorted(calls, key=lambda item:item['cumulative_ms'], reverse=True)[:12])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error('repeats must be 1..20')
    cold = [measure(IncrementalOVFReader(retain_frames=2), args.manifest) for _ in range(args.repeats)]
    reader = IncrementalOVFReader(retain_frames=2)
    reader.load(args.manifest)
    hot = [measure(reader, args.manifest) for _ in range(args.repeats)]
    # Synchronous worker entry for stage timing, with no competing polling.
    follower = LiveFollower(args.manifest, LiveConfig(recipe='activity'))
    follower._reload(time.monotonic())
    snapshot = follower.snapshot()
    view = snapshot.pop('view')
    follower.close()
    report = dict(manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cold=cold, hot=hot, cold_median_ms=statistics.median(r['elapsed_ms'] for r in cold),
        hot_median_ms=statistics.median(r['elapsed_ms'] for r in hot),
        follower=snapshot, final_sequence=view.sample.sequence if view else None,
        note='cProfile adds overhead; cold means fresh reader, not flushed OS cache. Stage timing includes no UI/audio. Use an immutable completed manifest.')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_json(report), encoding='utf-8')
    print(report_json(report))
    return 1 if snapshot['state'] == 'invalid' else 0


if __name__ == '__main__':
    raise SystemExit(main())
