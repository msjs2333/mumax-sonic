"""Generate a small analytic OVF2 sequence; these are not solver results."""
import argparse
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from mumax_sonic.sources.band_demo import make_band_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--frames', type=int, default=320)
    args = parser.parse_args()
    if not 1 <= args.frames <= 4096:
        parser.error('frames must be 1..4096')
    args.directory.mkdir(parents=True, exist_ok=False)
    for i in range(args.frames):
        frame = make_band_frame('band_mixed', i)
        header = '\n'.join(['# OOMMF OVF 2.0', '# Segment count: 1', '# Begin: Segment', '# Begin: Header',
            '# Title: analytic band_mixed; not a simulation', '# meshtype: rectangular', '# meshunit: m',
            '# valuedim: 3', '# valuelabels: m_x m_y m_z', '# valueunits: 1 1 1',
            f'# Desc: Total simulation time: {frame.sim_time_s:.17g} s',
            '# xnodes: 17', '# ynodes: 17', '# znodes: 1',
            f'# xbase: {frame.origin_m[0]}', f'# ybase: {frame.origin_m[1]}', '# zbase: 0',
            f'# xstepsize: {frame.dx_m}', f'# ystepsize: {frame.dy_m}', '# zstepsize: 1e-9',
            '# End: Header', '# Begin: Data Binary 4', '']).encode()
        with (args.directory/f'm{i:06d}.ovf').open('xb') as stream:
            stream.write(header)
            stream.write(struct.pack('<f', 1234567))
            stream.write(frame.vectors.astype('<f4').tobytes())
            stream.write(b'# End: Data Binary 4\n# End: Segment\n')
    print(f'Generated {args.frames} analytic OVF2 frames in {args.directory}')


if __name__ == '__main__':
    main()
