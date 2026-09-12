"""Bounded two-sublattice MuMax+ capture, semantic checks and replay export."""
import argparse
import json
from pathlib import Path
import struct
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mumax_sonic.sources.mumaxplus import MuMaxPlusSampler
from mumax_sonic.sublattices import combine_sublattices
from mumax_sonic.observers.activity import angular_activity
from mumax_sonic.field_pipeline import observe_field, apply_aggregation
from mumax_sonic.attention import Attention
from mumax_sonic.mapping import map_sample_with_report
from mumax_sonic.sources.ovf_replay import load_ovf_replay


def write_frame(path, frame, dz):
    ny, nx = frame.vectors.shape[:2]
    header = ['# OOMMF OVF 2.0', '# Segment count: 1', '# Begin: Segment', '# Begin: Header',
              '# meshtype: rectangular', '# meshunit: m', '# valuedim: 3',
              '# valuelabels: m_x m_y m_z', '# valueunits: 1 1 1',
              f'# Desc: Total simulation time: {frame.sim_time_s:.17g} s',
              f'# xnodes: {nx}', f'# ynodes: {ny}', '# znodes: 1']
    for axis, origin, step in zip('xyz', frame.origin_m, (frame.dx_m, frame.dy_m, dz)):
        header.extend([f'# {axis}base: {origin:.17g}', f'# {axis}stepsize: {step:.17g}'])
    header.extend(['# End: Header', '# Begin: Data Binary 8', ''])
    with path.open('xb') as stream:
        stream.write('\n'.join(header).encode())
        stream.write(struct.pack('<d', 123456789012345.0))
        stream.write(frame.vectors.astype('<f8').tobytes())
        stream.write(b'\n# End: Data Binary 8\n# End: Segment\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, help='new output directory')
    args = parser.parse_args(argv)
    output = args.destination.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / 'run_mumaxplus_afm.py')
    import mumaxplus
    from mumaxplus import World, Grid, Antiferromagnet
    parameters = dict(grid=[8, 8, 1], cellsize_m=[1e-9]*3, msat_A_m=200e3,
                      afmex_cell_J_m=-100e-12, afmex_nn_J_m=-15e-12,
                      aex_J_m=10e-12, ku1_J_m3=1e3, latcon_m=.35e-9,
                      alpha=.01, anisU=[0, 0, 1], demag=False,
                      initial_A=[0, .1, .9], initial_B=[0, -.1, -.9],
                      bias_T=[0, 0, 0], frames=41, interval_s=5e-14)
    (output / 'parameters.json').write_text(json.dumps(parameters, indent=2), encoding='utf-8')
    started = time.perf_counter()
    world = World(cellsize=tuple(parameters['cellsize_m']))
    magnet = Antiferromagnet(world, Grid(tuple(parameters['grid'])))
    magnet.afmex_cell = parameters['afmex_cell_J_m']
    magnet.afmex_nn = parameters['afmex_nn_J_m']
    magnet.latcon = parameters['latcon_m']
    magnet.enable_demag = False
    world.bias_magnetic_field = tuple(parameters['bias_T'])
    for sub, initial in zip(magnet.sublattices, (parameters['initial_A'], parameters['initial_B'])):
        sub.msat = parameters['msat_A_m']
        sub.aex = parameters['aex_J_m']
        sub.ku1 = parameters['ku1_J_m3']
        sub.anisU = tuple(parameters['anisU'])
        sub.alpha = parameters['alpha']
        sub.magnetization = tuple(initial)
    samplers = [MuMaxPlusSampler(world, sub, entity_id=name, segment_id='afm-small')
                for sub, name in zip(magnet.sublattices, ('A', 'B'))]
    series = {name: [] for name in ('A', 'B', 'neel')}
    raw_net, raw_neel, rates = [], [], []
    previous = None
    max_net_error = max_neel_error = 0.0
    for index in range(parameters['frames']):
        if index:
            world.timesolver.run(parameters['interval_s'])
        # Both captures and backend references run on the solver owner thread.
        a, b = (sampler.sample(index) for sampler in samplers)
        pair = combine_sublattices(a, b, pair_id='afm-small',
                                  msat_a_A_m=parameters['msat_A_m'], msat_b_A_m=parameters['msat_A_m'])
        backend_net = np.moveaxis(magnet.full_magnetization.eval()[:, 0], 0, -1)
        backend_neel = np.moveaxis(magnet.neel_vector.eval()[:, 0], 0, -1)
        max_net_error = max(max_net_error, float(np.max(np.abs(pair.net_A_m - backend_net))))
        max_neel_error = max(max_neel_error, float(np.max(np.abs(pair.neel_raw - backend_neel))))
        # Backend fields are float32; pair construction normalizes directions in float64.
        if not np.allclose(pair.net_A_m, backend_net, atol=.1, rtol=2e-6):
            raise ValueError('net magnetization differs from backend full_magnetization')
        if not np.allclose(pair.neel_raw, backend_neel, atol=3e-7, rtol=2e-6):
            raise ValueError('equal-weight Neel differs from equal-Ms backend reference')
        observation = angular_activity(previous, pair.neel)
        if index == 0:
            if observation.validity != 'warming_up':
                raise ValueError('first Neel frame must warm up')
        else:
            if observation.validity != 'valid' or observation.coverage != 1:
                raise ValueError('Neel activity is incomplete')
            rates.append(observation.mean_rad_s)
            view = observe_field(pair.neel, 'activity', previous=previous)
            attention = Attention(extent_m=pair.neel.extent_m, origin_m=pair.neel.center_m[:2])
            aggregated = apply_aggregation(view, attention, 8, 'adaptive')
            mapped = map_sample_with_report(aggregated.sample, attention, budget=8, strength_reference=1e11)
            if not mapped.scene.sources or len(mapped.scene.sources) > 8:
                raise ValueError('Neel activity failed bounded sound mapping')
        previous = pair.neel
        raw_net.append(pair.net_A_m)
        raw_neel.append(pair.neel_raw)
        for name, frame in zip(series, (a, b, pair.neel)):
            filename = f'{name}-{index:04d}.ovf'
            write_frame(output / filename, frame, parameters['cellsize_m'][2])
            series[name].append(dict(file=filename, sequence=index, time_s=frame.sim_time_s))
    if max(rates) <= 0:
        raise ValueError('expected nonzero Neel dynamics')
    for name, frame in zip(series, (a, b, pair.neel)):
        manifest = dict(schema_version=1, integrity='unchecked', entity_id=frame.entity_id,
                        segment_id=frame.segment_id, origin='simulation', time_kind='dynamics',
                        quantity='order_parameter_direction' if name == 'neel' else 'magnetization_direction',
                        value_unit='1', components=['x', 'y', 'z'], mask='all', frames=series[name],
                        derivation=json.loads(pair.neel.provenance) if name == 'neel' else None)
        (output / f'{name}.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    np.savez(output / 'physical-arrays.npz', net_A_m=raw_net, neel_raw=raw_neel,
             times_s=[record['time_s'] for record in series['neel']])
    replay = load_ovf_replay(output / 'neel.json')
    replay_rates = [angular_activity(a, b).mean_rad_s for a, b in zip(replay.frames, replay.frames[1:])]
    if not np.allclose(replay_rates, rates, rtol=1e-12, atol=1e-5):
        raise ValueError('exported Neel activity differs from captured activity')
    report = dict(status='valid', backend='mumaxplus', version=mumaxplus.__version__,
                  frames=parameters['frames'], valid_activity_pairs=len(rates), warmup_frames=1,
                  last_time_s=pair.neel.sim_time_s, elapsed_s=time.perf_counter()-started,
                  max_backend_net_error_A_m=max_net_error, max_backend_neel_error=max_neel_error,
                  neel_mean_activity_rad_s=dict(min=min(rates), max=max(rates)),
                  initial_max_net_A_m=float(np.max(np.abs(raw_net[0]))),
                  sound_mapping_frames=len(rates), source_budget=8,
                  replay_activity_max_error_rad_s=float(np.max(np.abs(np.array(replay_rates)-rates))),
                  device_listening_verified=False, convergence_verified=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
