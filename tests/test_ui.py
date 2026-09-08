"""Real Tk widget smoke tests, without claiming a native audio/ear test."""
import tkinter as tk
from types import SimpleNamespace
import pytest
from mumax_sonic.ui.app import SonicApp, audio_status
from mumax_sonic.sources.synthetic import SCENARIOS


@pytest.fixture(scope='module')
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def app(tk_root):
    window = tk.Toplevel(tk_root)
    window.withdraw()
    instance = SonicApp(window, no_audio=True)
    yield instance
    instance.close()


def test_widget_flow_pause_solo_roi_and_quality(app):
    app.scenario.set(SCENARIOS["colocated"])
    app._tick()
    original = app.sample.observations
    assert {s.sign for s in app.scene.sources} == {-1, 1}
    app.mode.set("negative")
    app._tick()
    assert {s.sign for s in app.scene.sources} == {-1}
    assert app.sample.observations == original
    app.static.set(False)
    app._tick()
    assert not app.scene.sources
    app.toggle_play()
    app._tick()
    assert app.scene.sources
    app.move_roi(SimpleNamespace(x=-10000, y=10000))
    assert app.center == (-1, -1)
    app.scenario.set(SCENARIOS["invalid"])
    app._tick()
    assert app.scene.validity == "invalid" and not app.scene.sources


def test_no_audio_start_is_explicit(app):
    app.open_audio()
    assert "--no-audio" in app.status.get()
    assert app.engine is None


def test_mute_during_device_open_remains_muted(app):
    app.opening = True
    app.mute()
    app._open_result = (True, "")
    app._tick()
    assert not app.audio_ready
    assert "静音" in app.status.get()


def test_device_status_does_not_disguise_hrtf_failure():
    text = audio_status({"state": "open", "device": "test", "hrtf_status": "denied"})
    assert "未启用" in text


def test_physical_field_scenes_and_roi_preserve_topology(app):
    from mumax_sonic.ui.app import FIELD_SCENARIOS
    app.scenario.set(FIELD_SCENARIOS['field:opposite_pair'])
    app._tick()
    assert app.field_view.diagnostic['q_abs'] == pytest.approx(2)
    original = app.field_view.diagnostic.copy()
    app.move_roi(SimpleNamespace(x=50, y=50))
    app.mode.set('negative')
    app._tick()
    assert app.field_view.diagnostic == original
    assert all(s.sign == -1 for s in app.scene.sources)
    app.mode.set('both')
    app.scenario.set(FIELD_SCENARIOS['field:wall_pma'])
    app.transport.sim_time_s = 1e-9
    app._tick()
    assert app.field_view.diagnostic['recipe'] == 'direction'
    assert app.scene.sources and all(s.orientation_enabled for s in app.scene.sources)
