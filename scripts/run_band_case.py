"""Run three small MuMax3 precession cases for real frequency-band validation."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mumax_sonic.sources.mumax3_bridge import MuMax3Bridge


def simulation(frequency_hz, opposite):
    return f'''// P4c independent spins: analytic constant-cone precession.
SetGridSize(8, 4, 1)
SetCellSize(5e-9, 5e-9, 5e-9)
Msat = 800e3
Aex = 0
alpha = 0
EnableDemag = false
GammaLL = 2*pi*28e9
B_ext = vector(0, 0, {frequency_hz:.17g}/28e9)
m = uniform(0.1, 0, sqrt(0.99))
{ 'DefRegion(1, XRange(-inf, 0))' if opposite else '' }
{ 'm.SetRegion(1, uniform(-0.1, 0, sqrt(0.99)))' if opposite else '' }
SetSolver(4)
FixDt = 0.5e-12
Save(m)
TableSave()
for sampleIndex := 1; sampleIndex < 320; sampleIndex++ {{
    Steps(10)
    Save(m)
    TableSave()
}}
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, help='new output directory')
    parser.add_argument('--mumax3', default=shutil.which('mumax3'))
    args = parser.parse_args(argv)
    if not args.mumax3:
        parser.error('mumax3 is not on PATH; supply --mumax3')
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, destination / 'run_band_case.py')
    runs = []
    for name, frequency, opposite in [('in_phase', 9.375e9, False),
                                       ('opposite_phase', 9.375e9, True),
                                       ('out_band', 18.75e9, False)]:
        folder = destination / name
        folder.mkdir()
        script = folder / 'simulation.mx3'
        script.write_text(simulation(frequency, opposite), encoding='utf-8')
        command = [args.mumax3, '-http=', '-o', str(folder / 'simulation.out'), str(script)]
        started = time.perf_counter()
        with (folder / 'solver.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, check=False)
        record = dict(case=name, frequency_hz=frequency, opposite_phase=opposite,
                      command=command, returncode=result.returncode,
                      elapsed_s=time.perf_counter()-started)
        runs.append(record)
        (destination / 'runs.json').write_text(json.dumps(runs, indent=2), encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {folder / "solver.log"}')
        with MuMax3Bridge(folder / 'simulation.out', folder / 'replay.json',
                         entity_id='m', segment_id=name, all_material=True) as bridge:
            bridge.poll()
            publication = bridge.poll()
        if publication['state'] != 'current' or publication['published_frames'] != 320:
            raise RuntimeError(f'incomplete {name}: {publication}')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
