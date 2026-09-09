from __future__ import annotations

import time

from mumax_sonic.audio import AudioConfig, AudioEngine
from mumax_sonic.model import MAX_SOURCE_BUDGET, SonicScene, SonicSource

from test_audio import _FakeOpenAL
from mumax_sonic.audio import openal


class _BoundedFakeOpenAL(_FakeOpenAL):
    def __init__(self):
        super().__init__()
        self.live_sources: set[int] = set()
        self.max_live_sources = 0
        self.alDeleteSources.callback = self._delete_source

    def _gen_source(self, count, destination):
        super()._gen_source(count, destination)
        self.live_sources.add(destination._obj.value)
        self.max_live_sources = max(self.max_live_sources, len(self.live_sources))

    def _delete_source(self, count, source):
        source_id = source._obj.value if hasattr(source, "_obj") else self._value(source)
        self.live_sources.discard(source_id)
        self._record(count, source)


def _scene(count: int) -> SonicScene:
    return SonicScene(tuple(
        SonicSource(f"source-{index}", (index / 16.0, 0.0, -1.0), 0.4)
        for index in range(count)
    ))


def _until(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


def test_fake_openal_source_pool_is_bounded_across_retargeting(monkeypatch):
    fake = _BoundedFakeOpenAL()
    dll = "fake-soft-oal.dll"
    monkeypatch.setattr(openal, "_resolve_dll", lambda path: dll)
    monkeypatch.setattr(openal, "_load_openal", lambda path: fake)
    engine = AudioEngine()
    engine.open(AudioConfig(dll_path=str(dll)))
    try:
        engine.update(_scene(MAX_SOURCE_BUDGET))
        _until(lambda: engine.diagnostics()["active_source_count"] == MAX_SOURCE_BUDGET)
        assert engine.diagnostics()["target_source_count"] == MAX_SOURCE_BUDGET
        assert fake.max_live_sources == MAX_SOURCE_BUDGET

        engine.update(_scene(4))
        _until(lambda: engine.diagnostics()["target_source_count"] == 4)
        transition = engine.diagnostics()
        assert transition["active_source_count"] <= MAX_SOURCE_BUDGET
        assert transition["active_source_count"] >= transition["target_source_count"]

        engine.update(SonicScene(_scene(4).sources, validity="stale"))
        _until(lambda: engine.diagnostics()["data_state"] == "stale")
        assert engine.diagnostics()["target_source_count"] == 0

        engine.update(_scene(MAX_SOURCE_BUDGET))
        _until(lambda: engine.diagnostics()["target_source_count"] == MAX_SOURCE_BUDGET)
        _until(lambda: engine.diagnostics()["active_source_count"] == MAX_SOURCE_BUDGET)

        engine.stop()
        _until(lambda: engine.diagnostics()["target_source_count"] == 0)
        assert engine.diagnostics()["active_source_count"] <= MAX_SOURCE_BUDGET
        assert all(gain == 0 for gain in fake.source_gains.values())

        engine.update(_scene(MAX_SOURCE_BUDGET))
        _until(lambda: engine.diagnostics()["target_source_count"] == MAX_SOURCE_BUDGET)
        _until(lambda: engine.diagnostics()["active_source_count"] == MAX_SOURCE_BUDGET)
        diagnostic = engine.diagnostics()
        assert diagnostic["source_capacity"] == MAX_SOURCE_BUDGET
        assert diagnostic["last_error"] is None
        assert fake.al_error == 0
    finally:
        engine.close()
    assert engine.diagnostics()["active_source_count"] == 0
