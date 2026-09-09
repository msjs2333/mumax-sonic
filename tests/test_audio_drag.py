from mumax_sonic.audio import AudioConfig, AudioEngine, openal
from mumax_sonic.model import SonicScene, SonicSource
from test_audio import _FakeOpenAL
from test_audio_budget import _until


def test_adaptive_partition_rename_keeps_native_loops_running(monkeypatch):
    fake = _FakeOpenAL()
    monkeypatch.setattr(openal, '_resolve_dll', lambda path: 'fake')
    monkeypatch.setattr(openal, '_load_openal', lambda path: fake)
    engine = AudioEngine()
    engine.open(AudioConfig(dll_path='fake'))
    def scene(prefix):
        return SonicScene(tuple(SonicSource(f'adaptive:positive:focus:{prefix}{i}',
                          (i / 10, 0, -1), .05) for i in range(8)))
    try:
        engine.update(scene('old'))
        _until(lambda: engine.diagnostics()['active_source_count'] == 8)
        allocations = fake.sources
        stops = len(fake.stop_calls)
        engine.update(scene('new'))
        _until(lambda: engine.diagnostics()['target_source_count'] == 8)
        import time
        time.sleep(.08)
        assert fake.sources == allocations
        assert len(fake.stop_calls) == stops
        assert engine.diagnostics()['active_source_count'] == 8
        assert engine.diagnostics()['last_error'] is None
    finally:
        engine.close()
