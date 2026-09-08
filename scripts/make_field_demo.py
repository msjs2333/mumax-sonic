"""Create a reproducible three-component NPZ example without running a solver."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from mumax_sonic.fields import FieldFrame
from mumax_sonic.sources.analytic import make_field, SCENARIOS
from mumax_sonic.sources.replay import save_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--scenario', choices=SCENARIOS, default='opposite_pair')
    parser.add_argument('--frames', type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.frames <= 100:
        parser.error('frames must be between 1 and 100')
    frames = []
    for i in range(args.frames):
        t = i*1e-10
        vectors = make_field(args.scenario, t)
        spacing = 2e-6/(vectors.shape[1]-1)
        frames.append(FieldFrame(vectors, spacing, spacing, t, (-1e-6, -1e-6, 0),
                                 sequence=i, provenance=f'analytic:{args.scenario}'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_replay(args.output, frames)
    print(f'Saved {len(frames)} synthetic vector frame(s): {args.output}')


if __name__ == '__main__':
    main()
