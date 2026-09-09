"""Follow a MuMax3 OVF directory and publish a growing manifest."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mumax_sonic.sources.mumax3_bridge import MuMax3Bridge


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--time-kind", choices=("dynamics", "static", "relaxation"), default="dynamics")
    parser.add_argument("--origin", choices=("simulation", "synthetic", "unknown"), default="simulation")
    material = parser.add_mutually_exclusive_group(required=True)
    material.add_argument("--all-material", action="store_true")
    material.add_argument("--mask", type=Path)
    parser.add_argument("--z-index", type=int)
    parser.add_argument("--pattern", default="m*.ovf")
    parser.add_argument("--interval-s", type=float, default=0.1)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if not 0 < args.interval_s <= 3600 or not 0 < args.duration_s <= 86400:
        parser.error("interval-s and duration-s must be positive and bounded")
    try:
        with MuMax3Bridge(args.directory, args.manifest, entity_id=args.entity, segment_id=args.segment,
                          time_kind=args.time_kind, origin=args.origin, all_material=args.all_material,
                          mask_path=args.mask, z_index=args.z_index, pattern=args.pattern) as bridge:
            started = time.monotonic()
            result = bridge.poll()
            if args.once and result["state"] == "waiting":
                time.sleep(args.interval_s)
                result = bridge.poll()
            while not args.once and time.monotonic() - started < args.duration_s:
                if args.report:
                    args.report.parent.mkdir(parents=True, exist_ok=True)
                    temporary = args.report.with_name("." + args.report.name + ".tmp")
                    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                    temporary.replace(args.report)
                if result["state"] == "invalid":
                    break
                time.sleep(args.interval_s)
                result = bridge.poll()
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.report.with_name("." + args.report.name + ".tmp")
                temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(args.report)
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result["state"] != "invalid" else 2
    except (OSError, ValueError) as exc:
        print(json.dumps({"state": "invalid", "reason": str(exc), "published_frames": 0, "pending_files": []}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
