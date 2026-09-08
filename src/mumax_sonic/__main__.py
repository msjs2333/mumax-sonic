"""GUI by default; explicit diagnostics and bounded device smoke checks."""
import argparse
import json
import math
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description="MuMax-Sonic P1 · synthetic spatial audio demo")
    parser.add_argument("--no-audio", action="store_true", help="GUI preview without opening an audio device")
    parser.add_argument("--dll", help="Explicit OpenAL Soft DLL path")
    parser.add_argument("--doctor", action="store_true", help="Enumerate devices and open a silent HRTF context")
    parser.add_argument("--audio-smoke", type=float, metavar="SECONDS", help="Bounded low-gain playback device check (1–1800 s)")
    parser.add_argument("--device", help="Exact OpenAL device name for doctor/smoke")
    parser.add_argument("--no-hrtf", action="store_true", help="Use non-HRTF output for doctor/smoke comparison")
    parser.add_argument("--report", type=Path, help="Write diagnostics JSON for doctor/smoke")
    parser.add_argument('--field-demo', choices=['uniform', 'skyrmion', 'opposite_pair', 'wall_inplane', 'wall_pma'], help='Open an analytical vector-field scene')
    parser.add_argument('--replay', type=Path, help='Open a vector NPZ replay in the GUI')
    parser.add_argument('--inspect-field', type=Path, help='Compute and print topology of all NPZ frames without audio')
    parser.add_argument('--method', choices=['solid_angle', 'finite_difference'], default='solid_angle', help='Method for --inspect-field')
    args = parser.parse_args()
    if args.audio_smoke is not None and (not math.isfinite(args.audio_smoke) or not 1 <= args.audio_smoke <= 1800):
        parser.error("--audio-smoke must be between 1 and 1800 seconds")
    if args.inspect_field:
        from .sources.replay import load_replay
        from .field_pipeline import observe_field
        replay = load_replay(args.inspect_field)
        report = {'sha256': replay.sha256, 'frames': [observe_field(f, method=args.method).diagnostic for f in replay.frames]}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    elif args.doctor or args.audio_smoke is not None:
        from .audio import AudioConfig, AudioEngine, enumerate_devices
        from .attention import Attention
        from .mapping import map_sample
        from .sources.synthetic import make_sample
        engine = AudioEngine()
        report = {"source_kind": "synthetic", "physical_topology_computed": False,
                  "human_listening_verified": False, "acoustic_latency_measured": False}
        try:
            report["devices"] = enumerate_devices(args.dll)
            engine.open(AudioConfig(dll_path=args.dll, device_name=args.device, hrtf=not args.no_hrtf))
            start = time.monotonic()
            sequence = 0
            if args.audio_smoke is not None:
                while (elapsed := time.monotonic()-start) < args.audio_smoke:
                    sample = make_sample("moving", elapsed*1e-9, sequence)
                    engine.update(map_sample(sample, Attention(background=1), master_gain=0.06))
                    sequence += 1
                    time.sleep(0.025)
                engine.stop()
                time.sleep(0.1)
            report.update(duration_s=time.monotonic()-start, updates=sequence, audio=engine.diagnostics())
            if report["audio"].get("state") != "open" or report["audio"].get("last_error"):
                report["error"] = "Audio engine reported a device/control failure."
        except Exception as exc:
            report.update(error=str(exc), audio=engine.diagnostics())
        finally:
            engine.close()
        report["state_after_close"] = engine.diagnostics().get("state")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if "error" in report:
            raise SystemExit(1)
    else:
        from .ui.app import run
        run(no_audio=args.no_audio, dll_path=args.dll, field_demo=args.field_demo, replay_path=args.replay)


if __name__ == "__main__":
    main()
