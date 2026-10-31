import gc
import threading
import tkinter as tk
from types import SimpleNamespace

import pytest
from tkinter import ttk

from mumax_sonic.sources.live import LiveConfig
from mumax_sonic.ui.app import SonicApp


@pytest.fixture(scope="module")
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    yield root
    root.destroy()
    gc.collect()


@pytest.fixture
def app(tk_root):
    window = tk.Toplevel(tk_root)
    window.withdraw()
    instance = SonicApp(window, no_audio=False)
    yield instance
    instance.close()
    # Tk variables can participate in cycles through widget callbacks. Drop
    # the app's references and collect them now, on the Tk-owning thread.
    instance.__dict__.clear()
    del instance
    gc.collect()


def test_dialog_discovery_selection_and_manual_waiting_attach_switches_and_mutes(app, tmp_path, monkeypatch):
    import mumax_sonic.ui.attach as attach_ui
    from mumax_sonic.sources.discovery import DiscoveryResult, SimulationCandidate

    candidate_dir = tmp_path / "running.out"
    candidate_dir.mkdir()
    roots = (tmp_path,)
    done = threading.Event()

    def discover(actual_roots):
        assert actual_roots == roots
        done.set()
        return DiscoveryResult((SimulationCandidate(candidate_dir, pids=(123,),
                                                       evidence="running process"),), ())

    monkeypatch.setattr(attach_ui, "discover_simulations", discover)
    previous = SimpleNamespace(closed=False)
    previous.close = lambda: setattr(previous, "closed", True)
    app.live_follower = previous
    made = {}

    class WaitingSource:
        directory = candidate_dir
        config = LiveConfig(recipe="topology")

        def start(self):
            made["started"] = True
            return self

        def snapshot(self):
            return {"state": "waiting", "view": None}

        def close(self):
            made["closed"] = True

    def factory(directory, config, **options):
        made.update(directory=directory, config=config, options=options)
        return WaitingSource()

    monkeypatch.setattr(attach_ui, "AttachedSimulation", factory)
    dialog = attach_ui.AttachmentDialog(app, roots)
    try:
        assert done.wait(2), "discovery worker did not complete"
        dialog.window.after_cancel(dialog.timer)
        dialog.poll()
        assert dialog.list.get_children() == ("0",)
        dialog.list.selection_set("0")
        dialog.select()
        assert dialog.directory.get() == str(candidate_dir)

        # Manual selection is allowed to start before any OVF frame exists.
        manual_dir = tmp_path / "manually-entered"
        manual_dir.mkdir()
        assert not tuple(manual_dir.glob("*.ovf"))
        dialog.directory.set(str(manual_dir))
        dialog.all_material.set(True)
        dialog.attach()

        assert made["directory"] == str(manual_dir)
        assert made["options"]["all_material"] is True
        assert made["started"] is True
        assert app._external_live_source is app.live_follower
        assert app.live_follower.snapshot()["state"] == "waiting"
        assert previous.closed is True
        assert app._mute_requested is True
        assert app.engine is None and not app.audio_ready
        assert dialog.closed is True
    finally:
        if not dialog.closed:
            dialog.close()
        dialog.__dict__.clear()
        del dialog
        gc.collect()


def test_polling_attachment_recipe_change_restarts_session_at_same_directory(app, tmp_path, monkeypatch):
    from mumax_sonic.sources.attachment import AttachedSimulation

    started, closed = [], []

    def start(source):
        started.append(source)
        return source

    def close(source):
        closed.append(source)

    monkeypatch.setattr(AttachedSimulation, "start", start)
    monkeypatch.setattr(AttachedSimulation, "close", close)
    directory = tmp_path / "simulation.out"
    directory.mkdir()
    original = AttachedSimulation(directory, LiveConfig(recipe="topology"), all_material=True)
    app.attach_live_source(original)

    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)

    recipe_box = next(child for child in descendants(app.window)
                      if isinstance(child, ttk.Combobox) and
                      child.cget("textvariable") == str(app.replay_recipe))
    app.replay_recipe.set(tuple(recipe_box.cget("values"))[-1])

    snapshot = app._poll_live()
    replacement = app._external_live_source
    assert replacement is not original
    assert replacement.directory == directory.resolve()
    assert replacement.config.recipe == "band"
    assert started == [replacement]
    assert closed == [original]
    assert snapshot["state"] == "waiting"
    assert app._mute_requested is True
    assert app.engine is None
