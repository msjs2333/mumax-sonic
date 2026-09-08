"""GUI by default; explicit diagnostics and bounded device smoke checks."""
import argparse
import math
from pathlib import Path
import time
from .reporting import report_json


def main():
    parser = argparse.ArgumentParser(description="MuMax-Sonic · vector-field observations and spatial audio")
    parser.add_argument("--no-audio", action="store_true", help="GUI preview without opening an audio device")
    parser.add_argument("--dll", help="Explicit OpenAL Soft DLL path")
    parser.add_argument("--doctor", action="store_true", help="Enumerate devices and open a silent HRTF context")
    parser.add_argument("--audio-smoke", type=float, metavar="SECONDS", help="Bounded low-gain playback device check (1–1800 s)")
    parser.add_argument("--device", help="Exact OpenAL device name for doctor/smoke")
    parser.add_argument("--no-hrtf", action="store_true", help="Use non-HRTF output for doctor/smoke comparison")
    parser.add_argument("--report", type=Path, help="Write diagnostics JSON for doctor/smoke")
    parser.add_argument('--field-demo', choices=['uniform', 'skyrmion', 'opposite_pair', 'wall_inplane', 'wall_pma', 'activity_rotation', 'activity_localized'], help='Open a field scene; activity scenes also support --audio-smoke')
    parser.add_argument('--replay', type=Path, help='Open a vector NPZ replay in the GUI')
    parser.add_argument('--inspect-field', type=Path, help='Compute and print observations of all NPZ frames without audio')
    parser.add_argument('--method', choices=['solid_angle', 'finite_difference'], default='solid_angle', help='Method for --inspect-field')
    parser.add_argument('--recipe', choices=['topology', 'activity'], default='topology', help='NPZ GUI/inspection recipe')
    parser.add_argument('--max-dt-ps', type=float, help='Explicit largest allowed activity interval; no cadence inferred when omitted')
    parser.add_argument('--activity-reference-rad-s', type=float, default=1e9, help='Fixed auditory reference; does not change physical rates')
    args = parser.parse_args()
    if args.audio_smoke is not None and (not math.isfinite(args.audio_smoke) or not 1 <= args.audio_smoke <= 1800):
        parser.error("--audio-smoke must be between 1 and 1800 seconds")
    if args.max_dt_ps is not None and (not math.isfinite(args.max_dt_ps) or args.max_dt_ps <= 0):
        parser.error('--max-dt-ps must be finite and positive')
    if not math.isfinite(args.activity_reference_rad_s) or args.activity_reference_rad_s <= 0:
        parser.error('--activity-reference-rad-s must be finite and positive')
    if args.audio_smoke and args.field_demo and not args.field_demo.startswith('activity_'):
        parser.error('--audio-smoke supports activity field demos or the default P1 moving source')
    max_dt_s = args.max_dt_ps*1e-12 if args.max_dt_ps is not None else None
    if args.inspect_field:
        from .sources.replay import load_replay
        from .field_pipeline import observe_field
        replay = load_replay(args.inspect_field)
        report = {'sha256': replay.sha256, 'frames': [observe_field(f, args.recipe, method=args.method,
            previous=replay.frames[i-1] if i else None, max_dt_s=max_dt_s).diagnostic for i, f in enumerate(replay.frames)]}
        print(report_json(report))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report_json(report), encoding='utf-8')
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
                    reference = 1.0
                    if args.field_demo:
                        from .sources.activity_demo import make_activity_frame, STEP_S
                        from .field_pipeline import observe_field
                        index = int(elapsed*1e-9/STEP_S)
                        frame = make_activity_frame(args.field_demo, index)
                        previous = make_activity_frame(args.field_demo, index-1) if index else None
                        view = observe_field(frame, 'activity', previous=previous, max_dt_s=max_dt_s)
                        sample = view.sample
                        report['activity'] = view.diagnostic
                        reference = args.activity_reference_rad_s
                        report['activity_reference_rad_s'] = reference
                    else:
                        sample = make_sample("moving", elapsed*1e-9, sequence)
                    engine.update(map_sample(sample, Attention(background=1), master_gain=0.06, strength_reference=reference))
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
            args.report.write_text(report_json(report), encoding="utf-8")
        print(report_json(report))
        if "error" in report:
            raise SystemExit(1)
    else:
        from .ui.app import run
        run(no_audio=args.no_audio, dll_path=args.dll, field_demo=args.field_demo, replay_path=args.replay,
            recipe=args.recipe, max_dt_s=max_dt_s, activity_reference=args.activity_reference_rad_s)


if __name__ == "__main__":
    main()
