import math
import time

import numpy as np
import pytest

from mumax_sonic.attention import Attention
from mumax_sonic.fields import FieldFrame
from mumax_sonic.observers.focused_activity import FocusedActivity


def pair(shape=(32, 32), *, angle=.2, segment="a", sequence=1):
    a = np.zeros((*shape, 3)); a[..., 0] = 1
    b = np.zeros_like(a); b[..., 0] = math.cos(angle); b[..., 1] = math.sin(angle)
    return (FieldFrame(a, 1e-9, 1e-9, 0, segment_id=segment, sequence=sequence-1),
            FieldFrame(b, 1e-9, 1e-9, .1, segment_id=segment, sequence=sequence))


def ready(observer, previous, current, attention):
    observer.observe(previous, current, attention)
    for _ in range(100):
        time.sleep(.01)
        view = observer.observe(previous, current, attention)
        if view.diagnostic["focus_compute"]["background_ready"]:
            return view
    pytest.fail("background did not become ready")


def test_uniform_rate_conserves_total_after_background_arrives():
    previous, current = pair(); observer = FocusedActivity(.01)
    try:
        view = ready(observer, previous, current, Attention(extent_m=16e-9, radius=.25))
        assert sum(x.strength for x in view.sample.observations) == pytest.approx(.2 / .1)
        assert view.diagnostic["focus_compute"]["global_current_mean_rad_s"] is None
    finally:
        observer.close()


def test_same_pair_and_tiles_reuse_physics_but_new_pair_recomputes(monkeypatch):
    import mumax_sonic.observers.focused_activity as module
    calls = []
    original = module.observe_field
    def measured(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'observe_field', measured)
    previous, current = pair((64, 64))
    observer = FocusedActivity(10)
    try:
        attention = Attention(extent_m=current.extent_m, origin_m=current.center_m[:2], radius=.2)
        observer.observe(previous, current, attention)
        repeated = observer.observe(previous, current, attention)
        assert len(calls) == 1
        assert repeated.diagnostic['focus_compute']['foreground_cache_hit']
        p2, c2 = pair((64, 64), sequence=2)
        observer.observe(p2, c2, attention)
        assert len(calls) == 2
        observer.observe(p2, c2, Attention(center=(.5, 0), extent_m=current.extent_m,
                                         origin_m=current.center_m[:2], radius=.2))
        assert len(calls) == 3
    finally:
        observer.close()


def test_background_tiles_do_not_overlap_focus_bounds():
    previous, current = pair(); observer = FocusedActivity(.01)
    try:
        attention = Attention(extent_m=16e-9, radius=.2)
        view = ready(observer, previous, current, attention)
        r0, r1, c0, c1 = observer._focus_bounds(current, attention)
        for item in view.sample.observations:
            if not item.source_id.startswith("background:"):
                continue
            col = item.position_m[0] / current.dx_m
            row = item.position_m[1] / current.dy_m
            assert not (c0 <= col < c1 and r0 <= row < r1)
        assert all(x.strength > 0 for x in view.sample.observations)
    finally:
        observer.close()


def test_focus_calls_observer_on_crop(monkeypatch):
    import mumax_sonic.observers.focused_activity as module
    seen = []; original = module.observe_field
    def wrapped(frame, *args, **kwargs):
        seen.append(frame.vectors.shape[:2]); return original(frame, *args, **kwargs)
    monkeypatch.setattr(module, "observe_field", wrapped)
    previous, current = pair((64, 64)); observer = FocusedActivity(10)
    try:
        observer.observe(previous, current, Attention(extent_m=32e-9, radius=.1))
        assert seen and seen[0][0] < 64 and seen[0][1] < 64
    finally:
        observer.close()


def test_segment_change_does_not_use_old_background():
    previous, current = pair(segment="one"); observer = FocusedActivity(.01)
    try:
        ready(observer, previous, current, Attention(extent_m=16e-9))
        p2, c2 = pair(segment="two")
        view = observer.observe(p2, c2, Attention(extent_m=16e-9))
        assert not view.diagnostic["focus_compute"]["background_ready"]
    finally:
        observer.close()


def test_geometry_change_does_not_reuse_background():
    previous, current = pair((32, 32)); observer = FocusedActivity(.01)
    try:
        ready(observer, previous, current, Attention(extent_m=16e-9))
        p2, c2 = pair((48, 48))
        view = observer.observe(p2, c2, Attention(extent_m=24e-9))
        assert not view.diagnostic["focus_compute"]["background_ready"]
    finally:
        observer.close()


def test_new_same_geometry_tasks_do_not_starve_background(monkeypatch):
    import mumax_sonic.observers.focused_activity as module
    original = module.angular_activity
    def slow(*args, **kwargs):
        time.sleep(.03)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "angular_activity", slow)
    observer = FocusedActivity(.001); attention = Attention(extent_m=16e-9)
    try:
        previous, current = pair()
        observer.observe(previous, current, attention)
        # Keep replacing the one queued slot while the first full calculation runs.
        for sequence in range(2, 8):
            prior, latest = pair(sequence=sequence)
            observer.observe(prior, latest, attention)
            time.sleep(.004)
        for _ in range(100):
            view = observer.observe(prior, latest, attention)
            if view.diagnostic["focus_compute"]["background_ready"]:
                break
            time.sleep(.01)
        assert view.diagnostic["focus_compute"]["background_ready"]
    finally:
        observer.close()


def test_warmup_empty_roi_and_close():
    previous, current = pair(); observer = FocusedActivity(.01)
    view = observer.observe(None, current, Attention(center=(1, 1), origin_m=(1e-3, 1e-3), extent_m=1e-9))
    assert view.sample.observations == ()
    assert view.sample.validity in {"warming_up", "invalid"}
    observer.close()
    assert not observer._thread.is_alive()
