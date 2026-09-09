"""Worker-only live OVF manifest following behaviour."""
import hashlib
import json
import time

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
        changed['frames'][0]['sha256'] = '0' * 64
        _publish(path, changed)
        bad = _wait(follower, 'invalid')
        assert bad['view'] is good['view']
    finally:
        follower.close()


def test_rewritten_published_frame_fails_closed_and_preserves_physical_metadata(tmp_path):
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
        rewritten['frames'][1]['sha256'] = hashlib.sha256(raw).hexdigest()
        _publish(path, rewritten)
        invalid = _wait(follower, 'invalid')
        assert invalid['view'] is good['view']
        assert 'provenance changed' in invalid['last_error']
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
