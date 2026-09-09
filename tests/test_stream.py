import time
import pytest
from mumax_sonic.sources.stream import FrameStream
from mumax_sonic.sources.live import LiveConfig
from mumax_sonic.sources.activity_demo import make_activity_frame


def wait(stream, predicate):
    deadline = time.monotonic()+2
    while time.monotonic()<deadline:
        snapshot = stream.snapshot()
        if predicate(snapshot):
            return snapshot
        time.sleep(.005)
    pytest.fail('stream did not reach requested state')


def test_direct_queue_preserves_physical_pair_and_stales():
    stream = FrameStream(LiveConfig(recipe='activity', stale_after_s=.5))
    stream.submit(make_activity_frame('activity_rotation', 0))
    stream.submit(make_activity_frame('activity_rotation', 1))
    stream.start()
    try:
        result = wait(stream, lambda s:s['view'] is not None)
        assert result['view'].sample.validity == 'valid'
        assert result['view'].sample.sim_time_s == 50e-12
        assert result['view'].diagnostic['mean_rad_s'] == pytest.approx(2*3.141592653589793/8e-9)
        assert result['skipped_observation_frames'] == 1
        assert wait(stream, lambda s:s['state']=='stale')['age_s'] >= .5
    finally:
        stream.close()
    assert stream.snapshot()['state'] == 'closed'


def test_overflow_counts_drops_without_inventing_intermediate_samples():
    stream = FrameStream(LiveConfig(recipe='activity'), max_pending=1)
    for i in range(5):
        stream.submit(make_activity_frame('activity_rotation', i))
    assert stream.snapshot()['pending_frames'] == 1
    stream.start()
    try:
        result = wait(stream, lambda s:s['view'] is not None)
        assert result['dropped_frames'] == 4
        assert result['view'].sample.validity == 'warming_up'
        stream.fail('backend disconnected')
        assert stream.snapshot()['state'] == 'invalid'
    finally:
        stream.close()


def test_memory_budget_and_source_regression_are_explicit():
    frame = make_activity_frame('activity_rotation', 0)
    with pytest.raises(ValueError, match='memory'):
        FrameStream(max_bytes=1).submit(frame)
    stream = FrameStream()
    stream.submit(frame)
    with pytest.raises(ValueError, match='increase'):
        stream.submit(frame)
    stream.close()
