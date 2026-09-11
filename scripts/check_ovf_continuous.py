"""Paced archived OVFs through the real follower, Tk and OpenAL control path.

Input manifest records require time_s and wall_offset_s. Offsets describe
observed/declared publication cadence, never simulation time scaling.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from mumax_sonic.sources.live import LiveConfig, LiveFollower
from mumax_sonic.ui.app import SonicApp, configure_dpi
from mumax_sonic.reporting import report_json
from check_live_soak import rss_bytes


def distribution(values):
    return dict(count=len(values), first=float(values[0]), last=float(values[-1]), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), maximum=float(max(values))) if values else dict(count=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--phase-s', type=float, default=45)
    parser.add_argument('--dll')
    parser.add_argument('--stale-s', type=float, default=2.0,
                        help='Explicit freshness budget; reported separately from latency')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--read-profile', type=Path,
                        help='Optional nested read timings; diagnostic run only')
    args = parser.parse_args()
    if not math.isfinite(args.phase_s) or not 10 <= args.phase_s <= 600:
        parser.error('phase-s must be 10..600')
    if not math.isfinite(args.stale_s) or args.stale_s <= 0:
        parser.error('stale-s must be finite and positive')
    if args.read_profile:
        from dataclasses import asdict
        from profile_read_costs import ProfileCollector, _summary
        collector = ProfileCollector()
        with collector.instrument():
            run(args, parser)
        samples = collector.snapshot()
        by_thread = {name: _summary([s for s in samples if s.thread == name])
                     for name in sorted({s.thread for s in samples})}
        args.read_profile.parent.mkdir(parents=True, exist_ok=True)
        args.read_profile.write_text(json.dumps(dict(
            by_thread=by_thread, samples=[asdict(s) for s in samples],
            note='Nested wall-clock timings, not additive. Includes instrumentation overhead. '
                 'mumax-sonic-live isolates follower reads; fieldframe_construct only instruments '
                 'the OVF loader constructor, not ROI crops. No content hashing.'), indent=2), encoding='utf-8')
    else:
        run(args, parser)


def run(args, parser):
    meta = json.loads(args.manifest.read_text(encoding='utf-8'))
    records = meta.pop('frames')
    base = args.manifest.resolve().parent
    for record in records:
        record['file'] = str((base / record['file']).resolve())
    if isinstance(meta.get('mask'), dict):
        meta['mask']['file'] = str((base / meta['mask']['file']).resolve())
    offsets = [float(r['wall_offset_s']) for r in records]
    physical_times = [float(r['time_s']) for r in records]
    if any(not math.isfinite(v) for v in physical_times) or any(b <= a for a, b in zip(physical_times, physical_times[1:])):
        parser.error('one segment with unique increasing physical times required for timing correlation')
    if len(records) < 2 or not all(math.isfinite(v) and v >= 0 for v in offsets) or any(b < a for a, b in zip(offsets, offsets[1:])):
        parser.error('at least two nondecreasing wall offsets required')
    duration = args.phase_s * 4
    if offsets[-1] < duration:
        parser.error('source cadence must span all four phases')
    phases = ['small_static', 'large_static', 'small_drag', 'large_drag']
    buckets = {name: dict(tick_ms=[], source_to_control_ms=[], source_to_observed_ms=[],
        processing_ms=[], observed_to_control_ms=[], job_stages={}, frame_lag=[], rss_bytes=[], states={}, audio_errors=[], applied_sequences=[]) for name in phases}
    stop = threading.Event()
    published = {}
    publication_lateness = []
    failures = []
    source_errors = set()
    replace_retries = []
    observations = set()
    applied = set()
    observed_events = {}
    applied_events = {}
    jobs = []
    result = {}
    with tempfile.TemporaryDirectory(prefix='mumax-sonic-soak-') as temporary:
        live_path = Path(temporary) / 'live.json'
        follower = LiveFollower(live_path, LiveConfig(recipe='activity', stale_after_s=args.stale_s)).start()
        configure_dpi()
        import tkinter as tk
        root = tk.Tk()
        app = SonicApp(root, dll_path=args.dll)
        app.focus_compute.set(True)
        app.source_budget.set(8)
        app.aggregation_mode.set('adaptive')
        app.master.set(.015)
        app.radius.set(.1)
        app.attach_live_source(follower)
        app.open_audio()
        start = time.monotonic()
        cpu_start = time.process_time()
        peak_voices = 0
        state_last = None
        state_started = start
        state_durations = {}
        original_commit = follower._commit_valid
        def commit(*positional, **keywords):
            original_commit(*positional, **keywords)
            snap = follower.snapshot()
            view = snap['view']
            if view is not None:
                stamp = published.get(view.field.sim_time_s)
                if stamp:
                    observed_events.setdefault(stamp[1], (time.monotonic(), stamp[0], snap['processing_ms']))
        follower._commit_valid = commit
        original_reload = follower._reload
        def reload(*positional, **keywords):
            original_reload(*positional, **keywords)
            job = follower.snapshot().get('last_job')
            if job:
                jobs.append(job)
        follower._reload = reload
        original_diagnostic = app.engine._set_diagnostic
        def diagnostic(**values):
            original_diagnostic(**values)
            if 'last_applied_sim_time_s' in values:
                stamp = published.get(values['last_applied_sim_time_s'])
                if stamp:
                    applied_events.setdefault(stamp[1], (values['last_control_applied_at_s'], stamp[0]))
        app.engine._set_diagnostic = diagnostic

        def publish():
            accepted = []
            try:
                for record in records:
                    target = start + record['wall_offset_s']
                    if target - start >= duration or stop.wait(max(0, target - time.monotonic())):
                        break
                    accepted.append({k: v for k, v in record.items() if k != 'wall_offset_s'})
                    payload = dict(meta, frames=accepted)
                    scratch = live_path.with_suffix('.tmp')
                    scratch.write_text(json.dumps(payload), encoding='utf-8')
                    # Timestamp just before the atomic publication; no source payload copy.
                    deadline = time.monotonic() + .5
                    retries = 0
                    while True:
                        now = time.monotonic()
                        published[record['time_s']] = (now, record['sequence'])
                        try:
                            scratch.replace(live_path)
                            break
                        except PermissionError:
                            if time.monotonic() >= deadline:
                                raise
                            retries += 1
                            stop.wait(.005)
                    replace_retries.append(retries)
                    publication_lateness.append((now - target) * 1000)
            except Exception as exc:
                failures.append(f'publisher: {exc}')

        producer = threading.Thread(target=publish, daemon=True)
        producer.start()
        original_tick = app._tick

        def tick():
            began = time.monotonic()
            index = min(3, int((began - start) / args.phase_s))
            app.radius.set(.1 if index in (0, 2) else .4)
            app.center = (.35 * math.sin((began - start) * .8), 0) if index >= 2 else (0, 0)
            original_tick()
            buckets[phases[index]]['tick_ms'].append((time.monotonic() - began) * 1000)
        app._tick = tick

        def check():
            nonlocal peak_voices, state_last, state_started
            now = time.monotonic()
            elapsed = now - start
            bucket = buckets[phases[min(3, int(elapsed / args.phase_s))]]
            snap = follower.snapshot()
            state = snap['state']
            if snap['last_error']:
                source_errors.add(snap['last_error'])
            if state != state_last:
                if state_last is not None:
                    state_durations[state_last] = state_durations.get(state_last, 0) + now - state_started
                state_last, state_started = state, now
            bucket['states'][state] = bucket['states'].get(state, 0) + 1
            rss = rss_bytes()
            if rss is not None:
                bucket['rss_bytes'].append(rss)
            view = snap['view']
            if view is not None:
                seq = view.field.sequence
                latest = max((v[1] for v in published.values()), default=seq)
                bucket['frame_lag'].append(latest - seq)
            diag = app.engine.diagnostics()
            peak_voices = max(peak_voices, diag.get('active_source_count', 0))
            for seq, (at, ready, processing) in tuple(observed_events.items()):
                if seq not in observations:
                    observations.add(seq)
                    target = buckets[phases[min(3, int((at-start)/args.phase_s))]]
                    target['source_to_observed_ms'].append((at-ready)*1000)
                    target['processing_ms'].append(processing)
            for seq, (at, ready) in tuple(applied_events.items()):
                if seq not in applied:
                    applied.add(seq)
                    target = buckets[phases[min(3, int((at-start)/args.phase_s))]]
                    target['applied_sequences'].append(seq)
                    target['source_to_control_ms'].append((at-ready)*1000)
                    if seq in observed_events:
                        target['observed_to_control_ms'].append((at-observed_events[seq][0])*1000)
            if diag.get('last_error') and diag['last_error'] not in bucket['audio_errors']:
                bucket['audio_errors'].append(diag['last_error'])
            if elapsed >= duration or failures:
                state_durations[state_last] = state_durations.get(state_last, 0) + now - state_started
                stop.set()
                producer.join(timeout=2)
                snap.pop('view', None)
                completed_jobs = tuple(jobs)
                for job in completed_jobs:
                    target = buckets[phases[min(3, max(0, int((job['completed_at_s']-start)/args.phase_s)))]]
                    for stage, cost in job['timings_ms'].items():
                        target['job_stages'].setdefault(f"{job['kind']}:{stage}", []).append(cost)
                for values in buckets.values():
                    values['job_stages'] = {key: distribution(costs) for key, costs in values['job_stages'].items()}
                    for key in ('tick_ms', 'source_to_control_ms', 'source_to_observed_ms', 'observed_to_control_ms', 'processing_ms', 'frame_lag', 'rss_bytes'):
                        values[key] = distribution(values[key])
                result.update(duration_s=elapsed, phases=buckets, state_duration_s=state_durations,
                    published_frames=len(published), observed_unique=len(observations), applied_unique=len(applied),
                    not_observed_by_end=sorted({v[1] for v in published.values()}-observations),
                    not_applied_by_end=sorted({v[1] for v in published.values()}-applied),
                    peak_voices=peak_voices, audio=diag, final_source=snap,
                    cpu_core_equivalents=(time.process_time()-cpu_start)/elapsed,
                    publication_lateness_ms=distribution(publication_lateness), failures=failures,
                    source_errors=sorted(source_errors), publication_replace_retries=sum(replace_retries),
                    stale_after_s=args.stale_s, jobs_completed=len(completed_jobs),
                    slowest_jobs=sorted(completed_jobs, key=lambda job: job['completed_at_s']-job['started_at_s'], reverse=True)[:12],
                    note='Archived real fields and declared wall cadence; atomic manifest publication to first native control application, event timestamps correlated by unique physical time. Excludes solver, OVF writing, bridge stability/decode and acoustic/smoothing completion. Phases use event completion time, boundaries may contain prior-phase work. RSS/state/lag sampled from Tk about every 50ms, no RSS peak or leak guarantee. Missing sets are not completed by cutoff, including warmup/in-flight frames. No human listening.')
                app.close()
                return
            root.after(50, check)
        root.after(50, check)
        try:
            root.mainloop()
        finally:
            stop.set()
            producer.join(timeout=2)
            follower.close()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_json(result), encoding='utf-8')
    print(report_json(result))


if __name__ == '__main__':
    main()
