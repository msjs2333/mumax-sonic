"""Paced synthetic live soak; optional real OpenAL output, no solver or Tk."""
import argparse
import hashlib
import math
from pathlib import Path
import sys
import time
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from mumax_sonic.sources.activity_demo import make_activity_frame
from mumax_sonic.sources.live import LiveConfig
from mumax_sonic.sources.stream import FrameStream
from mumax_sonic.field_pipeline import apply_aggregation
from mumax_sonic.attention import Attention
from mumax_sonic.mapping import map_sample_with_report
from mumax_sonic.audio import AudioEngine, AudioConfig
from mumax_sonic.reporting import report_json


def rss_bytes():
    """Working set on Windows; optional psutil on other platforms."""
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize
        return None
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration-s', type=float, default=120)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--hz', type=float, default=20)
    parser.add_argument('--audio', action='store_true')
    parser.add_argument('--dll', type=str)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not (math.isfinite(args.duration_s) and 1 <= args.duration_s <= 3600
            and math.isfinite(args.hz) and 1 <= args.hz <= 100 and 3 <= args.size <= 512):
        parser.error('duration 1..3600 s, hz 1..100, size 3..512 required')
    stream = FrameStream(LiveConfig(recipe='activity')).start()
    engine = AudioEngine() if args.audio else None
    latencies, processing, rss, audio_latencies = [], [], [], []
    states = set()
    sequence = 0
    observed = -1
    submitted = {}
    start = time.monotonic()
    cpu = time.process_time()
    next_frame = start
    peak_voices = 0
    try:
        if engine:
            engine.open(AudioConfig(dll_path=args.dll))
        start = time.monotonic()
        next_frame = start
        while time.monotonic()-start < args.duration_s:
            now = time.monotonic()
            if now >= next_frame:
                frame = make_activity_frame('activity_localized', sequence, size=args.size)
                submitted[sequence] = time.monotonic()
                stream.submit(frame)
                sequence += 1
                next_frame = now + 1/args.hz
            snapshot = stream.snapshot()
            states.add(snapshot['state'])
            view = snapshot['view']
            if view is not None and view.sample.sequence != observed:
                observed = view.sample.sequence
                attention = Attention(center=(.65*math.sin(now-start), 0))
                view = apply_aggregation(view, attention, 8, 'adaptive')
                if snapshot['state'] != 'current':
                    view = replace(view, sample=replace(view.sample, validity='stale'))
                mapped = map_sample_with_report(view.sample, attention, budget=8, master_gain=.02)
                if engine:
                    engine.update(mapped.scene)
                    diag = engine.diagnostics()
                    value = diag.get('control_apply_latency_ms')
                    if value is not None:
                        audio_latencies.append(value)
                peak_voices = max(peak_voices, len(mapped.scene.sources))
                latencies.append((time.monotonic()-submitted[observed])*1000)
                processing.append(snapshot['processing_ms'])
                submitted = {k:v for k,v in submitted.items() if k > observed}
                memory = rss_bytes()
                if memory is not None:
                    rss.append(memory)
            time.sleep(.002)
        stream.finish()
        deadline = time.monotonic()+5
        while stream.snapshot()['state'] not in ('finished', 'invalid') and time.monotonic()<deadline:
            time.sleep(.01)
        snapshot = stream.snapshot()
        snapshot.pop('view')
        audio = engine.diagnostics() if engine else None
    finally:
        if engine:
            engine.close()
        stream.close()
    report = dict(input='synthetic activity_localized; full XYZ; 50 ps physical interval',
        size=args.size, requested_hz=args.hz, duration_s=time.monotonic()-start,
        cpu_s=time.process_time()-cpu, submitted_frames=sequence, observed_updates=len(latencies),
        peak_voices=peak_voices, states=sorted(states), final_stream=snapshot,
        submit_to_mapping_p95_ms=float(np.percentile(latencies,95)) if latencies else None,
        submit_to_control_enqueue_p95_ms=float(np.percentile(latencies,95)) if latencies and engine else None,
        observer_p95_ms=float(np.percentile(processing,95)) if processing else None,
        audio_apply_sample_p95_ms=float(np.percentile(audio_latencies,95)) if audio_latencies else None,
        rss_first_bytes=rss[0] if rss else None, rss_last_bytes=rss[-1] if rss else None,
        rss_peak_bytes=max(rss) if rss else None, audio=audio,
        rss_first_quarter_median_bytes=float(np.median(rss[:max(1,len(rss)//4)])) if rss else None,
        rss_last_quarter_median_bytes=float(np.median(rss[-max(1,len(rss)//4):])) if rss else None,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        acoustic_latency_ms=None, human_listening_verified=False,
        limitation='No Tk or GPU; enqueue latency excludes audio application, device and acoustics. Audio samples may repeat diagnostics; no end-to-end p95 inferred by adding percentiles.')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_json(report), encoding='utf-8')
    print(report_json(report))
    return 0 if snapshot['state'] == 'finished' and 'invalid' not in states else 1


if __name__ == '__main__':
    raise SystemExit(main())
