"""Small real MuMax+ example: solver-owned capture and background sonification."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from mumax_sonic.sources.live import LiveConfig
from mumax_sonic.sources.stream import FrameStream
from mumax_sonic.sources.mumaxplus import MuMaxPlusSampler, probe_mumaxplus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--interval-s', type=float, default=.05)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--no-audio', action='store_true')
    args = parser.parse_args()
    import math
    if not 2 <= args.frames <= 4096 or not math.isfinite(args.interval_s) or args.interval_s < 0:
        parser.error('frames must be 2..4096 and interval-s finite and nonnegative')
    stream = FrameStream(LiveConfig(recipe='activity', stale_after_s=2)).start()
    stop = threading.Event()
    report = dict(backend=probe_mumaxplus(), input='real MuMax+ uniform precession',
        grid=[16,16,1], cellsize_m=[5e-9]*3, dt_s=5e-12, frames=args.frames,
        msat_A_m=8e5, aex_J_m=13e-12, alpha=.02, field_T=[0,0,.1], demag=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def produce():
        try:
            from mumaxplus import World, Grid, Ferromagnet
            world = World(cellsize=(5e-9,)*3)
            magnet = Ferromagnet(world, Grid((16,16,1)))
            magnet.msat=8e5; magnet.aex=13e-12; magnet.alpha=.02
            magnet.enable_demag=False
            magnet.magnetization=(1,0,0)
            world.bias_magnetic_field=(0,0,.1)
            sampler = MuMaxPlusSampler(world, magnet, entity_id='m', segment_id='uniform-precession')
            captures=[]; solves=[]; frame=None
            for sequence in range(args.frames):
                if stop.is_set():
                    break
                if sequence:
                    started=time.perf_counter(); world.timesolver.run(5e-12)
                    solves.append((time.perf_counter()-started)*1000)
                frame=sampler.sample(sequence)
                captures.append(sampler.last_capture_ms)
                stream.submit(frame)
                stop.wait(args.interval_s)
            import numpy as np
            if frame is None:
                return
            report.update(last_physical_time_s=frame.sim_time_s,
                capture_median_ms=float(np.median(captures)),
                solver_step_median_ms=float(np.median(solves)) if solves else None,
                final_source_sha256=None)
            if not stop.is_set():
                stream.finish()
        except Exception as exc:
            report['error']=str(exc)
            stream.fail(exc)
    producer = threading.Thread(target=produce, name='mumaxplus-solver-owner', daemon=True)
    producer.start()
    try:
        if args.headless:
            producer.join()
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                snapshot=stream.snapshot()
                if snapshot['view'] is not None and snapshot['view'].sample.sim_time_s == report.get('last_physical_time_s'):
                    break
                if 'error' in report: break
                time.sleep(.01)
        else:
            from mumax_sonic.ui.app import run
            run(no_audio=args.no_audio, live_source=stream, source_budget=8, aggregation='adaptive')
        snapshot=stream.snapshot()
        view=snapshot.pop('view')
        report['stream']=snapshot
        if view is not None:
            report['observation']=view.diagnostic
        report['native_audio_verified']=False
    finally:
        stop.set()
        producer.join(timeout=5)
        if producer.is_alive():
            report['error']='solver owner did not stop within 5 seconds'
        stream.close()
    from mumax_sonic.reporting import report_json
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(report_json(report),encoding='utf-8')
    print(report_json(report))
    return 1 if 'error' in report else 0


if __name__ == '__main__':
    raise SystemExit(main())
