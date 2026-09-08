import importlib.util
import sys
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location("listening_check", Path("scripts/listening_check.py"))
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_trial_generation_is_deterministic_and_balanced():
    trials = _MODULE.make_trials(seed=12)
    assert trials == _MODULE.make_trials(seed=12)
    assert len(trials) == 40
    for task in ("left_right", "sign"):
        answers = [t.answer for t in trials if t.task == task]
        assert answers.count(answers[0]) == 10
        assert set(answers) == ({"l", "r"} if task == "left_right" else {"+", "-"})
        if task == "sign":
            assert {t.x_m for t in trials if t.task == task} == {0.0}


def test_scoring_accepts_only_task_symbols():
    trial = _MODULE.make_trials(1)[0]
    assert _MODULE.score_response(trial, trial.answer)
    assert not _MODULE.score_response(trial, "garbage")
    assert not _MODULE.score_response(trial, "r" if trial.answer == "l" else "l")


def test_silent_cli_does_not_open_audio(capsys):
    args = _MODULE.parser().parse_args(["--count", "2"])
    assert _MODULE.run(args) == 0
    assert "pass --play" in capsys.readouterr().out


def test_unique_path_never_overwrites():
    # Keep pytest independent of locked-down system temp directories.
    directory = Path("local/listening_check_test")
    directory.mkdir(parents=True, exist_ok=True)
    first = _MODULE.unique_path(directory, "result")
    first.touch()
    second = _MODULE.unique_path(directory, "result")
    assert first != second


def test_play_flow_records_separate_scores_and_actual_device(monkeypatch, tmp_path):
    import csv
    from mumax_sonic.audio import openal
    class FakeEngine:
        closed = False
        def open(self, config): pass
        def diagnostics(self): return {'device': 'fake endpoint', 'hrtf_status': 'enabled'}
        def stop(self): pass
        def close(self): self.closed = True
    engine = FakeEngine()
    monkeypatch.setattr(openal, 'AudioEngine', lambda: engine)
    monkeypatch.setattr(_MODULE, '_play_trial', lambda *a, **kw: None)
    answers = iter(t.answer for t in _MODULE.make_trials(count=2))
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    args = _MODULE.parser().parse_args(['--play', '--count', '2', '--headphones', 'test',
        '--system-volume-note', '20%', '--output', str(tmp_path)])
    assert _MODULE.run(args) == 0 and engine.closed
    with next(tmp_path.glob('*.csv')).open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert all(r['task_correct'] == r['task_total'] == '2' for r in rows)
    assert all(r['device'] == 'fake endpoint' for r in rows)


def test_interrupt_before_first_trial_saves_empty_result(monkeypatch, tmp_path):
    from mumax_sonic.audio import openal
    class FakeEngine:
        closed = False
        def open(self, config): raise KeyboardInterrupt
        def close(self): self.closed = True
    engine = FakeEngine()
    monkeypatch.setattr(openal, 'AudioEngine', lambda: engine)
    args = _MODULE.parser().parse_args(['--play', '--headphones', 'test',
        '--system-volume-note', '20%', '--output', str(tmp_path)])
    assert _MODULE.run(args) == 0 and engine.closed
    assert len(list(tmp_path.glob('*.csv'))) == 1
