"""Publish a deterministic synthetic OVF stream and atomically updated manifest."""
import argparse
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import struct
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mumax_sonic.sources.activity_demo import make_activity_frame, STEP_S
from mumax_sonic.sources.ovf import read_ovf
from mumax_sonic.sources.ovf_replay import load_ovf_replay


def _ovf_bytes(frame):
    """Return a complete OVF2 binary payload for one FieldFrame."""
    size_y, size_x, _ = frame.vectors.shape
    header = "\n".join([
        "# OOMMF OVF 2.0", "# Segment count: 1", "# Begin: Segment",
        "# Begin: Header", "# Title: analytic activity_rotation; synthetic live demo",
        "# meshtype: rectangular", "# meshunit: m", "# valuedim: 3",
        "# valuelabels: m_x m_y m_z", "# valueunits: 1 1 1",
        f"# Desc: Total simulation time: {frame.sim_time_s:.17g} s",
        f"# xnodes: {size_x}", f"# ynodes: {size_y}", "# znodes: 1",
        f"# xbase: {frame.origin_m[0]}", f"# ybase: {frame.origin_m[1]}", "# zbase: 0",
        f"# xstepsize: {frame.dx_m}", f"# ystepsize: {frame.dy_m}", "# zstepsize: 1e-9",
        "# End: Header", "# Begin: Data Binary 4", "",
    ]).encode()
    return header + struct.pack("<f", 1234567) + frame.vectors.astype("<f4").tobytes() + b"# End: Data Binary 4\n# End: Segment\n"


def publish_manifest_atomic(path, metadata, validator=load_ovf_replay):
    """Validate and replace *path* atomically; return the validated metadata."""
    path = Path(path)
    raw = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                    prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
    try:
        validator(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return metadata


def _metadata(records):
    return dict(schema_version=1, entity_id="m", segment_id="activity_rotation",
        time_kind="dynamics", origin="synthetic", source_kind="synthetic",
        quantity="magnetization_direction",
        value_unit="1", components=["x", "y", "z"], z_index=None, mask="all",
        frames=records, sequence_source="numeric filename suffix",
        selection=dict(matched=len(records), imported=len(records), all_matches_selected=True, gaps_within_selected=[]))


def publish_live(output, frames=60, interval_s=0.1, manifest_name="live.json", sleep_fn=time.sleep,
                 on_publish=None):
    """Generate and publish ``frames`` frames, returning the manifest path."""
    output = Path(output)
    if type(frames) is not int or not 2 <= frames <= 4096:
        raise ValueError("frames must be in [2, 4096]")
    if not isinstance(interval_s, (int, float)) or not isfinite(interval_s) or interval_s <= 0:
        raise ValueError("interval_s must be > 0")
    candidate_manifest = Path(manifest_name)
    if candidate_manifest.parent != Path(".") or not candidate_manifest.name:
        raise ValueError("manifest must be a file directly within output directory")
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError("output directory must be empty")
    else:
        output.mkdir(parents=True)
    manifest = output / candidate_manifest.name
    records = []
    for index in range(frames):
        frame = make_activity_frame("activity_rotation", index, dt_s=STEP_S, size=17)
        payload = _ovf_bytes(frame)
        final = output / f"m{index:06d}.ovf"
        with tempfile.NamedTemporaryFile("wb", dir=output, prefix=f".{final.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        try:
            os.replace(temporary, final)
        finally:
            temporary.unlink(missing_ok=True)
        parsed = read_ovf(final)
        digest = hashlib.sha256(final.read_bytes()).hexdigest()
        if parsed.sha256 != digest or parsed.time_s != frame.sim_time_s:
            raise RuntimeError("published OVF failed integrity check")
        records.append(dict(file=str(final.resolve()), sha256=digest, sequence=index,
                            time_s=float(frame.sim_time_s)))
        publish_manifest_atomic(manifest, _metadata(records))
        if on_publish is not None:
            on_publish(manifest)
        if index + 1 < frames:
            sleep_fn(float(interval_s))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--interval-s", type=float, default=0.1)
    parser.add_argument("--manifest", default="live.json")
    args = parser.parse_args(argv)
    try:
        path = publish_live(args.output, args.frames, args.interval_s, args.manifest)
    except (ValueError, FileExistsError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Published {args.frames} synthetic activity OVF frames; manifest: {path}")


if __name__ == "__main__":
    main()
