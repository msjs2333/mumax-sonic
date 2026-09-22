"""Build a labelled P4 comparison playlist from completed validation runs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mumax_sonic.fields import FieldFrame
from mumax_sonic.observers.topology import topology
from mumax_sonic.sources.analytic import make_field
from run_mumaxplus_afm import write_frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, help='new playlist directory')
    parser.add_argument('--fm', type=Path, required=True, help='P4a replay manifest')
    parser.add_argument('--afm', type=Path, required=True, help='P4b Neel manifest')
    parser.add_argument('--band-root', type=Path, required=True, help='P4c three-case directory')
    args = parser.parse_args(argv)
    manifests = [args.fm, args.afm] + [args.band_root / name / 'replay.json'
                 for name in ('in_phase', 'opposite_phase', 'out_band')]
    for path in manifests:
        if not path.is_file():
            parser.error(f'missing source manifest: {path}')
    folder = args.destination.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    clips = []
    for name, path, recipe, index, reference in [
        ('FM activity (real)', manifests[0], 'activity', 50, 1e10),
        ('AFM Neel activity (real)', manifests[1], 'activity', 40, 1e11),
        ('Band in-phase (real)', manifests[2], 'band', 319, .01),
        ('Band opposite-phase (real)', manifests[3], 'band', 319, .01),
        ('Band outside (real)', manifests[4], 'band', 319, .01),
    ]:
        clip = dict(name=name, manifest=str(path.resolve()), recipe=recipe,
                    frame_index=index, reference=reference)
        if recipe == 'band':
            clip['band'] = dict(low_hz=8e9, high_hz=12e9, window_samples=256, reference_axis=[0, 0, 1])
        clips.append(clip)
    positive = make_field('skyrmion', size=65)
    if topology(positive, 2e-6/64, 2e-6/64).q_net < 0:
        positive = -positive
    for name, vectors in [('positive', positive), ('negative', -positive),
                          ('opposite_pair', make_field('opposite_pair', size=65))]:
        frame = FieldFrame(vectors, 2e-6/64, 2e-6/64, origin_m=(-1e-6, -1e-6, 0),
                           entity_id='synthetic-topology', time_kind='static')
        filename = f'{name}.ovf'
        write_frame(folder / filename, frame, 1e-9)
        manifest = dict(schema_version=1, integrity='unchecked', entity_id=frame.entity_id,
                        segment_id=name, origin='synthetic', time_kind='static',
                        quantity='magnetization_direction', value_unit='1', components=['x','y','z'],
                        mask='all', frames=[dict(file=filename, sequence=0, time_s=0.0)])
        (folder / f'{name}.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        clips.append(dict(name=f'Topology {name} (synthetic)', manifest=f'{name}.json',
                          recipe='topology', frame_index=0, reference=1.0))
    (folder / 'playlist.json').write_text(json.dumps(dict(clips=clips), indent=2), encoding='utf-8')
    print(folder / 'playlist.json')


if __name__ == '__main__':
    main()
