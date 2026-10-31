"""Bounded, read-only discovery of MuMax3 output directories."""
from dataclasses import dataclass
import csv
import io
import os
from pathlib import Path
import re
import subprocess
import sys


@dataclass(frozen=True)
class SimulationCandidate:
    directory: Path
    pids: tuple[int, ...] = ()
    evidence: str = ''
    has_frames: bool = False


@dataclass(frozen=True)
class DiscoveryResult:
    candidates: tuple[SimulationCandidate, ...]
    warnings: tuple[str, ...]


_FRAME = re.compile(r'^m[0-9]+\.ovf$', re.IGNORECASE)
_MAX_ENTRIES = 4096


def _processes(psutil):
    if sys.platform != 'win32':
        return psutil.process_iter()
    # psutil 5.x obtains Windows names by resolving every executable path.
    # Unrelated protected/network executables can make that scan take seconds.
    # Filter names with the OS first, then inspect only matching processes.
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq mumax3.exe', '/FO', 'CSV', '/NH'],
                            capture_output=True, text=True, errors='replace', timeout=5,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), check=True)
    def matching():
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row) >= 2 and row[0].lower() == 'mumax3.exe':
                try:
                    yield psutil.Process(int(row[1]))
                except (ValueError, psutil.NoSuchProcess):
                    continue
    return matching()


def discover_simulations(roots=(), *, processes=None, max_directories=256) -> DiscoveryResult:
    """Inspect explicit roots, their immediate .out children, and process outputs.

    A PID indicates a matching process, not that it is producing frames. Directory
    and per-directory entry limits prevent an unbounded filesystem walk.
    """
    warnings = []
    found = {}
    scanned = set()

    def key(path):
        return os.path.normcase(os.path.abspath(os.fspath(path)))

    def add(path, pid=None, evidence=''):
        path = Path(os.path.abspath(os.fspath(path)))
        k = key(path)
        item = found.setdefault(k, {'path': path, 'pids': set(), 'evidence': set(), 'frames': False})
        if pid is not None:
            item['pids'].add(pid)
        if evidence:
            item['evidence'].add(evidence)
        return item

    def scan(path, children=False):
        k = key(path)
        if k in scanned:
            return
        if len(scanned) >= max(0, max_directories):
            warnings.append('Directory scan limit reached; discovery is incomplete')
            return
        scanned.add(k)
        try:
            if path.is_symlink():
                warnings.append(f'Skipped symlink directory: {path}')
                return
            with os.scandir(path) as entries:
                for index, entry in enumerate(entries):
                    if index >= _MAX_ENTRIES:
                        warnings.append(f'Entry scan limit reached: {path}')
                        break
                    try:
                        if _FRAME.fullmatch(entry.name) and entry.is_file(follow_symlinks=False):
                            add(path, evidence='frames')['frames'] = True
                        elif children and entry.name.lower().endswith('.out') and entry.is_dir(follow_symlinks=False):
                            add(Path(entry.path), evidence='output directory')
                            scan(Path(entry.path))
                    except OSError as exc:
                        warnings.append(f'Cannot inspect {entry.path}: {exc}')
        except OSError as exc:
            warnings.append(f'Cannot scan {path}: {exc}')

    if processes is None:
        try:
            import psutil
            processes = _processes(psutil)
        except ImportError:
            warnings.append('psutil unavailable; running processes were not inspected')
            processes = ()
        except Exception as exc:
            warnings.append(f'Cannot enumerate processes: {exc}')
            processes = ()

    try:
        for process in processes:
            try:
                name = process.name()
                if name.lower() not in ('mumax3', 'mumax3.exe'):
                    continue
                pid = int(process.pid)
                args = process.cmdline()
                if not args:
                    warnings.append(f'Cannot inspect command line for MuMax3 PID {pid}')
                    continue
                output = None
                script = None
                i = 1
                while i < len(args):
                    arg = args[i]
                    if arg == '-o':
                        i += 1
                        if i < len(args):
                            output = args[i]
                    elif arg.startswith('-o='):
                        output = arg[3:]
                    elif not arg.startswith('-') and arg.lower().endswith('.mx3'):
                        script = arg
                    i += 1
                target = Path(output) if output else (Path(script).with_suffix('.out') if script else None)
                if target is None:
                    warnings.append(f'No output path found for MuMax3 PID {pid}')
                    continue
                if not target.is_absolute():
                    try:
                        cwd = process.cwd()
                    except Exception as exc:
                        warnings.append(f'Cannot determine cwd for MuMax3 PID {pid}: {exc}')
                        continue
                    if not cwd:
                        warnings.append(f'Cannot determine cwd for MuMax3 PID {pid}')
                        continue
                    target = Path(cwd) / target
                add(target, pid, 'running process')
                scan(target)
            except Exception as exc:
                warnings.append(f'Cannot inspect process: {exc}')
    except Exception as exc:
        warnings.append(f'Cannot enumerate processes: {exc}')

    for root in roots or (Path.cwd(),):
        scan(Path(root), children=True)
    candidates = tuple(SimulationCandidate(v['path'], tuple(sorted(v['pids'])),
                                            ', '.join(sorted(v['evidence'])), v['frames'])
                       for _, v in sorted(found.items()))
    return DiscoveryResult(candidates, tuple(dict.fromkeys(warnings)))
