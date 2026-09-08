"""Real Tk widget smoke tests, without claiming a native audio/ear test."""
import tkinter as tk
from types import SimpleNamespace
import pytest
from mumax_sonic.ui.app import SonicApp, audio_status
from mumax_sonic.sources.synthetic import SCENARIOS


@pytest.fixture
def app():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    instance = SonicApp(root, no_audio=True)
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
