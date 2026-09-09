import time

import numpy as np

from mumax_sonic.attention import Attention
from mumax_sonic.fields import FieldFrame
from mumax_sonic.sources.live import LiveConfig
from mumax_sonic.sources.stream import FrameStream


def _frame(sequence, time_s, size=64):
    vectors = np.zeros((size, size, 3), dtype=np.float64)
    angle = 0.1 * sequence
    vectors[..., 0] = np.cos(angle)
    vectors[..., 1] = np.sin(angle)
    return FieldFrame(vectors, 1e-9, 1e-9, time_s, entity_id="m",
                      segment_id="focus-test", sequence=sequence,
                      source_kind="live")


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def test_default_activity_stream_accepts_750k_point_frame_without_worker():
    vectors = np.zeros((500, 1500, 3), dtype=np.float64)
    vectors[..., 2] = 1.0
    frame = FieldFrame(vectors, 1e-9, 1e-9, 0.0, entity_id="m",
                       segment_id="budget-test", sequence=0, source_kind="live")
    stream = FrameStream(LiveConfig(recipe="activity"))
    stream.submit(frame)
    assert stream.snapshot()["pending_frames"] == 1
    stream.close()


def test_focus_recompute_does_not_count_as_update_or_refresh_frame_age():
    stream = FrameStream(LiveConfig(recipe="activity", stale_after_s=0.5))
    stream.start()
    try:
        stream.submit(_frame(0, 0.0))
        stream.submit(_frame(1, 1e-12))
        _wait_until(lambda: stream.snapshot()["view"] is not None and stream.snapshot()["view"].sample.sequence == 1)
        before = stream.snapshot()
        time.sleep(0.03)
        stream.set_attention(Attention(center=(0.2, -0.1), extent_m=31.5e-9), enabled=True)
        _wait_until(lambda: "focus_compute" in stream.snapshot()["view"].diagnostic)
        after = stream.snapshot()
        assert after["updates"] == before["updates"]
        assert after["age_s"] > before["age_s"]
        assert after["state"] == "current"
    finally:
        stream.close()
        assert stream.snapshot()["state"] == "closed"


def test_manifest_focus_drag_preserves_freshness_and_full_field_extent(tmp_path):
    import json
    from mumax_sonic.sources.live import LiveFollower
    from test_ovf_replay import manifest, write_ovf

    path, meta = manifest(tmp_path, count=2)
    for i, record in enumerate(meta['frames']):
        write_ovf(tmp_path / record['file'], _frame(i, i * 1e-11).vectors[None], i * 1e-11)
        record.pop('sha256')
    path.write_text(json.dumps(meta), encoding='utf-8')
    follower = LiveFollower(path, LiveConfig(recipe='activity', poll_interval_s=.01, stale_after_s=.5)).start()
    try:
        _wait_until(lambda: follower.snapshot()['view'] is not None)
        before = follower.snapshot()
        field = before['view'].field
        follower.set_attention(Attention(radius=.2, extent_m=field.extent_m, origin_m=field.center_m[:2]), enabled=True)
        _wait_until(lambda: 'focus_compute' in follower.snapshot()['view'].diagnostic)
        after = follower.snapshot()
        assert after['updates'] == before['updates']
        assert after['age_s'] >= before['age_s']
        assert after['view'].field.extent_m == field.extent_m
        assert after['view'].diagnostic['focus_compute']['fraction'] < 1
        _wait_until(lambda: follower.snapshot()['state'] == 'stale')
        old_view = follower.snapshot()['view']
        follower.set_attention(Attention(center=(.2, 0), radius=.2, extent_m=field.extent_m, origin_m=field.center_m[:2]), enabled=True)
        _wait_until(lambda: follower.snapshot()['view'] is not old_view)
        assert follower.snapshot()['state'] == 'stale'
        assert follower.snapshot()['updates'] == before['updates']
    finally:
        follower.close()
