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


def test_roi_outline_is_independent_of_output_selection(app):
    from mumax_sonic.attention import Attention
    from mumax_sonic.mapping import map_sample
    from mumax_sonic.model import Observation, Sample
    inside = tuple(Observation(f'in-{i}', (i*1e-8, 0, 0), .1+i*.1) for i in range(5))
    outside = Observation('out', (.9e-6, 0, 0), 1)
    app.sample = Sample(0, inside+(outside,))
    attention = Attention()
    app.scene = map_sample(app.sample, attention)
    app._draw(attention)
    assert not app.audio_ready  # still a preview, no device claim
    assert all(float(app.canvas.itemcget(f'source:{o.source_id}', 'width')) == 3 for o in inside)
    assert float(app.canvas.itemcget('source:out', 'width')) == 1
    assert app.canvas.find_withtag('selected:out')
    assert sum(bool(app.canvas.find_withtag(f'selected:{o.source_id}')) for o in inside) == 3


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


def test_activity_navigation_and_pause_do_not_create_rates(app):
    from mumax_sonic.ui.app import FIELD_SCENARIOS
    from mumax_sonic.sources.activity_demo import OMEGA_RAD_S, STEP_S
    app.scenario.set(FIELD_SCENARIOS['field:activity_rotation'])
    app.reset()
    app._tick()
    assert app.sample.validity == 'warming_up' and not app.scene.sources
    app.step_frame(1)
    app.mode.set('negative')  # unsigned activity cannot be hidden by an old sign solo
    app._tick()
    assert app.field_view.diagnostic['mean_rad_s'] == pytest.approx(OMEGA_RAD_S)
    assert app.scene.sources
    assert all('disabled' in button.state() for button in app.sign_buttons)
    before = app.sample
    app.speed.set('4')
    app.change_speed()
    app._tick()
    assert app.sample is before  # same physical pair, even during a paused fast-play setting
    app.static.set(False)
    app._tick()
    assert not app.scene.sources
    assert app.field_view.diagnostic['mean_rad_s'] == pytest.approx(OMEGA_RAD_S)
    app.seek_to(8*STEP_S)
    app._tick()
    app.seek_to(2*STEP_S)
    app._tick()
    assert app.field_view.diagnostic['mean_rad_s'] == pytest.approx(OMEGA_RAD_S)
    app.max_dt_ns.set('0.025')
    app.apply_max_dt()
    app._tick()
    assert app.sample.validity == 'warming_up'


def test_replay_activity_gap_and_recovery_in_window(app):
    from mumax_sonic.sources.activity_demo import make_activity_frame, STEP_S, OMEGA_RAD_S
    from mumax_sonic.sources.replay import FieldReplay
    app.replay = FieldReplay(tuple(make_activity_frame('activity_rotation', i) for i in (0, 1, 4, 5)), 'test-gap')
    app._labels['测试回放'] = 'replay'
    app.scenario.set('测试回放')
    app.replay_recipe.set('活动')
    app.seek_to(4*STEP_S)
    app._tick()
    assert app.sample.validity == 'warming_up' and not app.scene.sources
    app.step_frame(1)
    app._tick()
    assert app.field_view.diagnostic['mean_rad_s'] == pytest.approx(OMEGA_RAD_S)
    assert app.sample.validity == 'valid' and app.scene.sources
    assert not app.transport.playing  # end of replay holds measured result


def test_band_navigation_switch_and_unsigned_controls(app):
    from mumax_sonic.ui.app import FIELD_SCENARIOS
    from mumax_sonic.sources.band_demo import STEP_S
    app.scenario.set(FIELD_SCENARIOS['field:band_in'])
    app.reset()
    app._tick()
    assert app.sample.validity == 'warming_up' and not app.scene.sources
    app.step_frame(1)
    assert app.transport.sim_time_s == STEP_S
    app.seek_to(300*STEP_S)
    app.mode.set('negative')
    app._tick()
    assert app.sample.validity == 'valid' and app.scene.sources
    assert all('disabled' in button.state() for button in app.sign_buttons)
    power = app.field_view.diagnostic['mean_power']
    app.speed.set('4')
    app.change_speed()
    app._tick()
    assert app.field_view.diagnostic['mean_power'] == power
    app.band_low.set('28')
    app.band_high.set('32')
    app.apply_band()
    app._tick()
    assert app.field_view.diagnostic['mean_power'] < power*1e-6
    app.seek_to(0)
    app._tick()
    assert app.sample.validity == 'warming_up' and not app.scene.sources
    previous = app.band_config
    app.band_axis.set('0,0,0')
    app.apply_band()
    assert app.band_config == previous


def test_ovf_manifest_window_load_and_navigation(app, tmp_path, monkeypatch):
    # Independent bytes fixture; the UI must use the same source records as CLI.
    from test_ovf_replay import manifest
    from mumax_sonic.ui import app as app_module
    path, _ = manifest(tmp_path, indices=(0, 1, 4, 5))
    monkeypatch.setattr(app_module.filedialog, 'askopenfilename', lambda **kw: str(path))
    app.load_field()
    app.replay_recipe.set('活动')
    app._tick()
    assert app.sample.validity == 'warming_up'
    app.step_frame(1)
    app._tick()
    assert app.field_view.diagnostic['mean_rad_s'] == pytest.approx(1e10)
    assert 'OVF' in app.subtitle.get()
    assert 'XYZ' in app.subtitle.get()
    app.step_frame(1)
    app._tick()
    assert app.sample.validity == 'warming_up' and not app.scene.sources


def test_budget_changes_selection_not_physics_and_pause_keeps_report(app):
    from mumax_sonic.ui.app import FIELD_SCENARIOS
    app.scenario.set(FIELD_SCENARIOS['field:activity_rotation'])
    app.seek_to(1e-9)
    app.source_budget.set(4)
    app._tick()
    before=app.field_view.diagnostic.copy()
    assert len(app.selection_report['selected_ids'])==4
    app.source_budget.set(16)
    app._tick()
    assert app.field_view.diagnostic==before
    assert len(app.scene.sources)==16
    assert app.selection_report['selected_fraction_observer_input']['absolute']==pytest.approx(1)
    selected=app.selection_report['groups']
    app.static.set(False)
    app._tick()
    assert not app.scene.sources
    assert app.selection_report['groups'][0]['absolute']['selected']==selected[0]['absolute']['selected']
    assert app.selection_report['groups'][0]['absolute']['output']==0
    assert '非零输出候选 0 路' in app.selection_note.get()


def _await_aggregation(app):
    import time
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        app._tick()
        if app._aggregation_key is not None and app._aggregation_key[1].center == app.center:
            return
        time.sleep(.01)
    pytest.fail('aggregation did not complete')


def test_adaptive_roi_reuses_physics_and_regroups(app):
    from mumax_sonic.ui.app import FIELD_SCENARIOS
    app.scenario.set(FIELD_SCENARIOS['field:opposite_pair'])
    app.aggregation_mode.set('adaptive')
    app._tick()
    _await_aggregation(app)
    before = app.field_view
    app.center = (.3, -.2)
    app._tick()
    _await_aggregation(app)
    after = app.field_view
    assert after.contributions is before.contributions
    assert after.diagnostic['q_abs'] == before.diagnostic['q_abs']
    assert after is not before
    assert len(after.sample.observations) <= 4
    assert after.diagnostic['adaptive_aggregation']['status'] == 'valid'
    app._tick()
    assert app.field_view is after
    assert '空间 RMS' in app.selection_note.get()


def test_slow_aggregation_does_not_block_roi_or_apply_old_scene(app, monkeypatch):
    from threading import Event
    from mumax_sonic.ui.app import FIELD_SCENARIOS
    from mumax_sonic.ui import aggregation_worker
    entered, release = Event(), Event()
    original = aggregation_worker.apply_aggregation
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)
    monkeypatch.setattr(aggregation_worker, 'apply_aggregation', slow)
    app.scenario.set(FIELD_SCENARIOS['field:opposite_pair'])
    app.aggregation_mode.set('adaptive')
    app.source_budget.set(8)
    try:
        app._tick()
        assert entered.wait(1)
        app.center = (.4, -.2)
        app._tick()  # must return while the background job is blocked
        assert not release.is_set()
        assert app.sample.validity == 'warming_up'
        assert not app.scene.sources
        assert '空间聚合更新中' in app.data_note.get()
        app.scenario.set(SCENARIOS['invalid'])
        app._tick()
        assert app.sample.validity == 'invalid'
    finally:
        release.set()
    app._tick()
    assert app.sample.validity == 'invalid'


from dataclasses import replace
from mumax_sonic.field_pipeline import observe_field
from mumax_sonic.sources.activity_demo import make_activity_frame


def test_live_quality_stales_and_recovers_without_advancing_physical_time(app):
    old = make_activity_frame('activity_rotation', 1)
    current = make_activity_frame('activity_rotation', 2)
    view = observe_field(replace(current, source_kind='live'), 'activity', previous=replace(old, source_kind='live'))
    state = dict(view=view, state='current', reason='new frame', age_s=.1, updates=1,
                 last_error=None, processing_ms=1)
    def poll():
        app.live_snapshot = dict(state)
        return app.live_snapshot
    app.follow_live('unused.json')
    app._poll_live = poll
    app._tick()
    assert app.sample.sim_time_s == current.sim_time_s
    assert app.scene.sources
    assert app.sample.source_kind == 'live'
    state.update(state='stale', age_s=3)
    app._tick()
    assert app.sample.validity == 'stale'
    assert not app.scene.sources
    assert app.selection_report['selected_fraction_observer_input']['absolute'] is None
    assert app.sample.sim_time_s == current.sim_time_s
    state.update(state='invalid', reason='SHA256 mismatch')
    app._tick()
    assert not app.scene.sources
    assert 'SHA256' in app.data_note.get()
    state.update(state='current', age_s=.1)
    app._tick()
    assert app.scene.sources
    app.toggle_play()
    assert not app.transport.playing
    app.seek_to(1)
    assert app.sample.sim_time_s == current.sim_time_s
    state.update(state='finished', reason='采样已结束，物理声源静音')
    app._tick()
    assert not app.scene.sources
    assert app.sample.sim_time_s == current.sim_time_s
    assert '采样已结束' in app.data_note.get()
    texts = [app.canvas.itemcget(item, 'text') for item in app.canvas.find_all()
             if app.canvas.type(item) == 'text']
    assert '采样已结束\n物理声源静音' in texts


def test_live_waiting_has_no_fake_zero_measurement(app):
    app.follow_live('missing.json')
    state = dict(view=None, state='waiting', reason='waiting for publication', age_s=None,
                 updates=0, last_error=None, processing_ms=None)
    def poll():
        app.live_snapshot = state
        return state
    app._poll_live = poll
    app._tick()
    assert app.sample.validity == 'warming_up'
    assert app.sample.source_kind == 'live'
    assert not app.scene.sources
    assert app.selection_report['validity'] == 'warming_up'
