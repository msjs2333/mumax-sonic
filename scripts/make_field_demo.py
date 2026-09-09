"""Create a reproducible three-component NPZ example without running a solver."""
import argparse
from pathlib import Path
import sys
import math
from dataclasses import replace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from mumax_sonic.fields import FieldFrame
from mumax_sonic.sources.analytic import make_field, SCENARIOS
from mumax_sonic.sources.replay import save_replay
from mumax_sonic.sources.activity_demo import SCENARIOS as ACTIVITY_SCENARIOS, make_activity_frame

from mumax_sonic.sources.band_demo import SCENARIOS as BAND_SCENARIOS, make_band_frame

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--scenario', choices=[*SCENARIOS, *ACTIVITY_SCENARIOS, *BAND_SCENARIOS], default='opposite_pair')
    parser.add_argument('--frames', type=int, default=1)
    parser.add_argument('--step-ps', type=float, default=None, help='Physical interval between recorded samples')
    parser.add_argument('--drop-frame', type=int, action='append', default=[], help='Omit this source index, preserving the sequence gap')
    parser.add_argument('--time-kind', choices=['dynamics', 'static', 'relaxation'], default='dynamics')
    args = parser.parse_args()
    if args.step_ps is None:
        args.step_ps = 5 if args.scenario in BAND_SCENARIOS else 50
    limit = 4096 if args.scenario in BAND_SCENARIOS else 100
    if not 1 <= args.frames <= limit:
        parser.error(f'frames must be between 1 and {limit}')
    if not math.isfinite(args.step_ps) or args.step_ps <= 0:
        parser.error('step-ps must be finite and positive')
    if any(i < 0 or i >= args.frames for i in args.drop_frame) or len(set(args.drop_frame)) == args.frames:
        parser.error('drop-frame must leave at least one frame and name an existing index')
    frames = []
    for i in range(args.frames):
        if i in args.drop_frame:
            continue
        if args.scenario in BAND_SCENARIOS:
            frames.append(replace(make_band_frame(args.scenario, i, dt_s=args.step_ps*1e-12), time_kind=args.time_kind))
            continue
        if args.scenario in ACTIVITY_SCENARIOS:
            frames.append(replace(make_activity_frame(args.scenario, i, dt_s=args.step_ps*1e-12), time_kind=args.time_kind))
            continue
        t = i*args.step_ps*1e-12
        vectors = make_field(args.scenario, t)
        spacing = 2e-6/(vectors.shape[1]-1)
        frames.append(FieldFrame(vectors, spacing, spacing, t, (-1e-6, -1e-6, 0),
                                 sequence=i, time_kind=args.time_kind, provenance=f'analytic:{args.scenario}'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_replay(args.output, frames)
    print(f'Saved {len(frames)} synthetic vector frame(s): {args.output}')


if __name__ == '__main__':
    main()
