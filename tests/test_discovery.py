from pathlib import Path

from mumax_sonic.sources.discovery import discover_simulations


class Process:
    def __init__(self, pid, name, args, cwd=None):
        self.pid, self._name, self._args, self._cwd = pid, name, args, cwd

    def name(self):
        return self._name

    def cmdline(self):
        return self._args

    def cwd(self):
        if isinstance(self._cwd, Exception):
            raise self._cwd
        return self._cwd


def test_process_paths_deduplicate_and_do_not_guess_cwd(tmp_path):
    out = tmp_path / 'run.out'
    out.mkdir()
    (out / 'm0001.ovf').write_text('')
    ps = [Process(3, 'mumax3.exe', ['mumax3', 'run.mx3'], str(tmp_path)),
          Process(2, 'mumax3', ['mumax3', '-o', str(out)], PermissionError()),
          Process(4, 'mumax3', ['mumax3', '-o=relative'], PermissionError('denied')),
          Process(5, 'other.exe', ['other', 'mumax3', '-o', str(tmp_path / 'bad')])]
    result = discover_simulations([tmp_path], processes=ps)
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.directory == out
    assert candidate.pids == (2, 3)
    assert candidate.has_frames
    assert any('cwd' in warning for warning in result.warnings)


def test_roots_are_shallow_and_frames_require_numeric_suffix(tmp_path):
    direct = tmp_path / 'first.out'
    direct.mkdir()
    (direct / 'mfoo.ovf').write_text('')
    nested = direct / 'nested.out'
    nested.mkdir()
    (nested / 'm1.ovf').write_text('')
    result = discover_simulations([tmp_path], processes=[])
    assert [c.directory for c in result.candidates] == [direct]
    assert not result.candidates[0].has_frames
    assert not result.candidates[0].pids


def test_limit_and_missing_process_output(tmp_path):
    for name in ('a.out', 'b.out'):
        (tmp_path / name).mkdir()
    missing = tmp_path / 'missing.out'
    result = discover_simulations([tmp_path], processes=[Process(1, 'mumax3', ['mumax3', '-o=' + str(missing)])], max_directories=1)
    assert any(c.directory == missing and c.pids == (1,) for c in result.candidates)
    assert any('limit' in warning.lower() for warning in result.warnings)
    assert any('Cannot scan' in warning for warning in result.warnings)


def test_symlink_child_is_not_traversed(tmp_path):
    target = tmp_path / 'target'
    target.mkdir()
    (target / 'm1.ovf').write_text('')
    link = tmp_path / 'linked.out'
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        return
    result = discover_simulations([tmp_path], processes=[])
    assert not result.candidates


def test_windows_tasklist_prefilter_parses_csv_and_limits_scope(monkeypatch):
    import types
    from mumax_sonic.sources import discovery

    calls = []
    selected = []
    class Gone(Exception):
        pass
    fake_psutil = types.SimpleNamespace(
        NoSuchProcess=Gone,
        Process=lambda pid: selected.append(pid) or pid,
    )
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return types.SimpleNamespace(stdout=(
            '"MuMax3.EXE","123","Console","1","1,234 K"\n'
            '"other.exe","999","name with ""quotes"" and spaces","1","0 K"\n'
            'INFO: No tasks are running which match the specified criteria.\n'
            '"mumax3.exe","invalid","Console","1","0 K"\n'))
    monkeypatch.setattr(discovery.sys, 'platform', 'win32')
    monkeypatch.setattr(discovery.subprocess, 'run', run)
    assert list(discovery._processes(fake_psutil)) == [123]
    assert selected == [123]
    args, options = calls[0]
    assert args == ['tasklist', '/FI', 'IMAGENAME eq mumax3.exe', '/FO', 'CSV', '/NH']
    assert options['timeout'] == 5
    assert not options.get('shell', False)
    assert options['check'] is True


def test_windows_tasklist_failures_warn_but_roots_are_scanned(tmp_path, monkeypatch):
    import subprocess
    import sys
    import types
    from mumax_sonic.sources import discovery

    out = tmp_path / 'a folder.out'
    out.mkdir()
    fake_psutil = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'psutil', fake_psutil)
    monkeypatch.setattr(discovery.sys, 'platform', 'win32')
    for error in (FileNotFoundError('tasklist unavailable'),
                  subprocess.TimeoutExpired('tasklist', 5)):
        def fail(*args, **kwargs):
            raise error
        monkeypatch.setattr(discovery.subprocess, 'run', fail)
        result = discover_simulations([tmp_path])
        assert [candidate.directory for candidate in result.candidates] == [out]
        assert any('Cannot enumerate processes' in warning for warning in result.warnings)
