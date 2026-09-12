"""Run the bounded public SP4 Field 1 case and publish its replay manifest."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mumax_sonic.sources.mumax3_bridge import MuMax3Bridge


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="new run directory (must not exist)")
    parser.add_argument("--mumax3", default=shutil.which("mumax3"))
    args = parser.parse_args(argv)
    if not args.mumax3:
        parser.error("mumax3 is not on PATH; supply --mumax3")
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    script = destination / "field1.mx3"
    shutil.copyfile(ROOT / "examples/standard_problem4/field1.mx3", script)
    command = [args.mumax3, "-http=", "-o", str(destination / "simulation.out"), str(script)]
    started = time.perf_counter()
    with (destination / "solver.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, check=False)
    metadata = {"command": command, "returncode": result.returncode,
                "elapsed_s": time.perf_counter() - started,
                "case": "SP4 Field 1, official MuMax3 relaxed initialization",
                "scope": "ingestion and observer validation; no mesh convergence certification"}
    (destination / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"MuMax3 exited {result.returncode}; see {destination / 'solver.log'}")
    manifest = destination / "replay.json"
    with MuMax3Bridge(destination / "simulation.out", manifest, entity_id="m",
                      segment_id="sp4-field1", all_material=True) as bridge:
        bridge.poll()
        publication = bridge.poll()
    if publication["state"] == "invalid" or publication["published_frames"] != 101:
        raise RuntimeError(f"Expected 101 complete dynamic frames: {publication}")
    print(json.dumps({"manifest": str(manifest), "publication": publication}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
