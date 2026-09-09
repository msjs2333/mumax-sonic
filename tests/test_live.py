"""Worker-only live OVF manifest following behaviour."""
import json
import time

from mumax_sonic.attention import Attention
from mumax_sonic.sources.live import LiveConfig, LiveFollower
from test_ovf_replay import manifest


def _publish(path, meta):
    path.write_text(json.dumps(meta), encoding='utf-8')


def _wait(follower, state, timeout=1.5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = follower.snapshot()
        if result['state'] == state:
            return result
        time.sleep(.01)
    raise AssertionError(f'expected {state}, got {follower.snapshot()}')


def test_missing_then_current_stale_and_append(tmp_path):
    path = tmp_path / 'live.json'
    follower = LiveFollower(path, LiveConfig(poll_interval_s=.01, stale_after_s=.5)).start()
    try:
        assert _wait(follower, 'waiting')['view'] is None
        _, meta = manifest(tmp_path, count=1)
        _publish(path, meta)
        current = _wait(follower, 'current')
        assert current['updates'] == 1
        assert current['view'].field.source_kind == 'live'
        assert current['view'].field.sequence == 0
        stale = _wait(follower, 'stale')
        assert stale['age_s'] >= .5
        # append a separately hash-bound physical frame and republish atomically
        _, appended = manifest(tmp_path, count=2)
        _publish(path, appended)
        current = _wait(follower, 'current')
        assert current['updates'] == 2
        assert current['view'].field.sequence == 1
        assert current['view'].field.sim_time_s == 1e-11
    finally:
        follower.close()
    assert follower.snapshot()['state'] == 'closed'


def test_partial_and_corrupt_publication_preserve_good_view_then_recover(tmp_path):
    path, meta = manifest(tmp_path, count=1)
    follower = LiveFollower(path, LiveConfig(poll_interval_s=.01, stale_after_s=.5)).start()
    try:
        good = _wait(follower, 'current')
        path.write_text('{', encoding='utf-8')
        bad = _wait(follower, 'invalid')
        assert bad['view'] is good['view']
        assert bad['last_error']
        _publish(path, meta)
        recovered = _wait(follower, 'current')
        assert recovered['view'] is good['view']
        changed = json.loads(json.dumps(meta))
        for record in changed['frames']:
            record.pop('sha256', None)
        _publish(path, changed)
        current = _wait(follower, 'current')
        assert current['updates'] == 1
        assert current['view'] is good['view']
    finally:
        follower.close()


def test_rewritten_published_frame_does_not_reset_staleness_or_updates(tmp_path):
    path, meta = manifest(tmp_path, count=2)
    follower = LiveFollower(path, LiveConfig(poll_interval_s=.01, stale_after_s=.5, recipe='activity')).start()
    try:
        good = _wait(follower, 'current')
        frame = good['view'].field
        assert (frame.segment_id, frame.sequence, frame.sim_time_s) == ('test', 1, 1e-11)
        rewritten = json.loads(json.dumps(meta))
        source = tmp_path / rewritten['frames'][1]['file']
        raw = bytearray(source.read_bytes())
        end = raw.index(b'# End: Data Binary 8')
        raw[end - 1] ^= 1  # alter a final vector byte while retaining OVF syntax
        raw = bytes(raw)
        source.write_bytes(raw)
        _publish(path, rewritten)
        current = _wait(follower, 'current')
        assert current['view'] is good['view']
        assert current['updates'] == 1
        stale = _wait(follower, 'stale')
        assert stale['updates'] == 1
    finally:
        follower.close()


def test_snapshot_becomes_stale_even_while_reload_is_blocked(tmp_path, monkeypatch):
    from threading import Event
    import mumax_sonic.sources.live as live
    path, meta = manifest(tmp_path, count=1)
    follower = LiveFollower(path, LiveConfig(poll_interval_s=.01, stale_after_s=.1)).start()
    release, entered = Event(), Event()
    try:
        _wait(follower, 'current')
        original = follower._reader.load
        def blocked(path):
            entered.set()
            assert release.wait(2)
            return original(path)
        monkeypatch.setattr(follower._reader, 'load', blocked)
        meta['note'] = 'republish'
        _publish(path, meta)
        assert entered.wait(1)
        assert _wait(follower, 'stale')['view'] is not None
    finally:
        release.set()
        follower.close()


def test_missing_frame_recovers_without_manifest_republication(tmp_path):
    path, meta = manifest(tmp_path, count=1)
    source = tmp_path / meta['frames'][0]['file']
    raw = source.read_bytes()
    source.unlink()
    follower = LiveFollower(path, LiveConfig(poll_interval_s=.01)).start()
    try:
        _wait(follower, 'invalid')
        source.write_bytes(raw)
        assert _wait(follower, 'current')['updates'] == 1
    finally:
        follower.close()


def test_published_history_cannot_be_removed(tmp_path):
    path, meta = manifest(tmp_path, count=2)
    follower = LiveFollower(path, LiveConfig(poll_interval_s=.01)).start()
    try:
        _wait(follower, 'current')
        meta['frames'] = meta['frames'][1:]
        _publish(path, meta)
        assert 'truncated' in _wait(follower, 'invalid')['last_error']
    finally:
        follower.close()


def test_roi_only_reuses_last_loaded_frames(tmp_path, monkeypatch):
    path, meta = manifest(tmp_path, count=2)
    follower = LiveFollower(path, LiveConfig(recipe='activity', poll_interval_s=.01)).start()
    calls = []
    original_load = follower._reader.load

    def counted_load(source):
        calls.append(source)
        return original_load(source)

    monkeypatch.setattr(follower._reader, 'load', counted_load)
    try:
        before = _wait(follower, 'current')
        load_count = len(calls)
        follower.set_attention(Attention(center=(.1, 0), radius=.2,
                                         extent_m=before['view'].field.extent_m,
                                         origin_m=before['view'].field.center_m[:2]), enabled=True)
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            focused = follower.snapshot()
            if focused['last_job'] and focused['last_job']['kind'] == 'roi':
                break
            time.sleep(.01)
        end = time.monotonic() + .25
        while time.monotonic() < end:
            follower.set_attention(Attention(center=(.1, 0), radius=.2,
                                             extent_m=before['view'].field.extent_m,
                                             origin_m=before['view'].field.center_m[:2]), enabled=True)
            time.sleep(.005)
        assert focused['last_job']['kind'] == 'roi'
        assert len(calls) == load_count
        assert follower.snapshot()['updates'] == before['updates']
    finally:
        follower.close()


def test_roi_does_not_revive_stale_or_invalid_state(tmp_path):
    path, meta = manifest(tmp_path, count=2)
    follower = LiveFollower(path, LiveConfig(recipe='activity', poll_interval_s=.01,
                                              stale_after_s=.5)).start()
    try:
        current = _wait(follower, 'current')
        stale = _wait(follower, 'stale')
        follower.set_attention(Attention(center=(.2, 0), radius=.2,
                                         extent_m=current['view'].field.extent_m,
                                         origin_m=current['view'].field.center_m[:2]), enabled=True)
        time.sleep(.05)
        stale_after_roi = follower.snapshot()
        assert stale_after_roi['state'] == 'stale'

        path.write_text('{', encoding='utf-8')
        _wait(follower, 'invalid')
        follower.set_attention(Attention(center=(-.2, 0), radius=.2,
                                         extent_m=current['view'].field.extent_m,
                                         origin_m=current['view'].field.center_m[:2]), enabled=True)
        time.sleep(.05)
        bad = follower.snapshot()
        assert bad['state'] == 'invalid'
        assert bad['view'] is stale_after_roi['view']
    finally:
        follower.close()


def test_new_manifest_frame_progresses_during_continuous_roi_requests(tmp_path):
    path, meta = manifest(tmp_path, count=1)
    follower = LiveFollower(path, LiveConfig(recipe='activity', poll_interval_s=.01,
                                              stale_after_s=.5)).start()
    try:
        first = _wait(follower, 'current')
        field = first['view'].field
        for index in range(10):
            follower.set_attention(Attention(center=(index * .01, 0), radius=.2,
                                             extent_m=field.extent_m,
                                             origin_m=field.center_m[:2]), enabled=True)
            time.sleep(.005)
        _, appended = manifest(tmp_path, count=2)
        _publish(path, appended)
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            latest = follower.snapshot()
            if latest['state'] == 'current' and latest['view'].field.sequence == 1:
                break
            follower.set_attention(Attention(center=((time.monotonic() % .2), 0), radius=.2,
                                             extent_m=field.extent_m, origin_m=field.center_m[:2]), enabled=True)
            time.sleep(.01)
        assert latest['view'].field.sequence == 1
        assert latest['updates'] == 2
    finally:
        follower.close()
