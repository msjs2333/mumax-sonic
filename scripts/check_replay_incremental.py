"""Compare incremental publication with offline decoding of one small replay."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mumax_sonic.sources.incremental import IncrementalOVFReader
from mumax_sonic.sources.ovf_replay import load_ovf_replay


def check_incremental(manifest):
    manifest = Path(manifest).resolve()
    offline = load_ovf_replay(manifest)
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    records = meta["frames"]
    for record in records:
        record["file"] = str((manifest.parent / record["file"]).resolve())
    if isinstance(meta.get("mask"), dict):
        meta["mask"]["file"] = str((manifest.parent / meta["mask"]["file"]).resolve())
    reader = IncrementalOVFReader(retain_frames=2)
    # Publication is simulated; vectors and physical times are real source data.
    with tempfile.TemporaryDirectory(prefix="sonic-incremental-") as scratch:
        publication = Path(scratch) / "live.json"
        for count, expected in enumerate(offline.frames, start=1):
            meta["frames"] = records[:count]
            publication.write_text(json.dumps(meta), encoding="utf-8")
            tail = reader.load(publication)
            actual = tail.frames[-1]
            for name in ("vectors", "mask"):
                if not np.array_equal(getattr(actual, name), getattr(expected, name)):
                    raise ValueError(f"Incremental {name} differs at frame {count}")
            for name in ("sequence", "sim_time_s", "dx_m", "dy_m", "origin_m", "time_kind", "entity_id", "segment_id"):
                if getattr(actual, name) != getattr(expected, name):
                    raise ValueError(f"Incremental {name} differs at frame {count}")
            if count > 1 and not np.array_equal(tail.frames[-2].vectors, offline.frames[count-2].vectors):
                raise ValueError("Incremental predecessor differs")
    return {"frames_compared": len(offline.frames), "exact_match": True,
            "publication": "simulated one-frame append of existing solver output",
            "diagnostics": reader.diagnostics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(check_incremental(args.manifest), indent=2)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
