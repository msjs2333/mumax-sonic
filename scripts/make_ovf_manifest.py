"""Create a hash-bound OVF manifest without running or modifying a simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from mumax_sonic.sources.ovf import read_ovf
from mumax_sonic.sources.ovf_replay import load_ovf_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--pattern', default='m*.ovf', help='One numeric-suffix output series; no recursive scan')
    parser.add_argument('--limit', type=int, default=32, help='Small preview: first N files, preserving source numbers')
    parser.add_argument('--time-kind', choices=['dynamics', 'static', 'relaxation'], required=True)
    parser.add_argument('--origin', choices=['simulation', 'synthetic', 'unknown'], required=True)
    parser.add_argument('--entity', default='m')
    parser.add_argument('--segment', default='ovf-segment')
    parser.add_argument('--quantity', choices=['magnetization_direction', 'order_parameter_direction'], default='magnetization_direction')
    parser.add_argument('--value-unit', choices=['1', 'A/m'], default='1')
    parser.add_argument('--components', nargs=3, default=['x', 'y', 'z'])
    parser.add_argument('--z-index', type=int)
    mask = parser.add_mutually_exclusive_group(required=True)
    mask.add_argument('--all-material', action='store_true', help='Explicitly treat every grid site as material; zero vectors remain invalid')
    mask.add_argument('--mask-file', type=Path, help='Boolean NPY material mask for selected plane or whole mesh')
    args = parser.parse_args()
    if not 1 <= args.limit <= 4096:
        parser.error('limit must be 1..4096')
    if '/' in args.pattern or '\\' in args.pattern or '**' in args.pattern:
        parser.error('pattern must select files directly in the specified directory')
    if args.output.exists():
        parser.error('output exists; choose a new manifest path')
    paths = list(args.directory.glob(args.pattern))
    if not paths:
        parser.error('no matching OVF files')
    indexed = []
    prefixes = set()
    for path in paths:
        match = re.fullmatch(r'(.*?)(\d+)', path.stem)
        if match:
            prefixes.add(match[1])
            indexed.append((int(match[2]), path))
        elif len(paths) == 1:
            indexed.append((0, path))
        else:
            parser.error('multiple files require numeric suffixes; hand-write a manifest for named checkpoints')
    if len(prefixes) > 1 or len({i for i, _ in indexed}) != len(indexed):
        parser.error('ambiguous series or duplicate frame numbers; narrow --pattern')
    indexed.sort(key=lambda pair: pair[0])
    records = []
    for sequence, path in indexed[:args.limit]:
        field = read_ovf(path)
        if field.time_s is None:
            parser.error('OVF has no total physical time; hand-write explicit time_s records, never infer from filenames')
        records.append(dict(file=str(path.resolve()), sha256=field.sha256, sequence=sequence))
    material = 'all'
    if args.mask_file:
        if args.mask_file.stat().st_size > 256*1024*1024:
            parser.error('mask exceeds prototype size limit')
        material = dict(file=str(args.mask_file.resolve()), sha256=hashlib.sha256(args.mask_file.read_bytes()).hexdigest())
    metadata = dict(schema_version=1, entity_id=args.entity, segment_id=args.segment,
        time_kind=args.time_kind, origin=args.origin, quantity=args.quantity, value_unit=args.value_unit,
        components=args.components, z_index=args.z_index, mask=material, frames=records,
        sequence_source='numeric filename suffix' if prefixes else 'single named checkpoint',
        selection=dict(matched=len(indexed), imported=len(records), all_matches_selected=len(records)==len(indexed),
            gaps_within_selected=[[a['sequence']+1,b['sequence']-1] for a,b in zip(records,records[1:]) if b['sequence']>a['sequence']+1]))
    raw = json.dumps(metadata, ensure_ascii=False, indent=2)
    # Validate the exact bytes and data contract before publishing the manifest.
    with tempfile.TemporaryDirectory(prefix='sonic-ovf-') as temporary:
        candidate = Path(temporary)/'manifest.json'
        candidate.write_text(raw, encoding='utf-8')
        replay = load_ovf_replay(candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(raw)
    print(f'Validated {len(replay.frames)} of {len(indexed)} OVF files; manifest: {args.output}')


if __name__ == '__main__':
    main()
