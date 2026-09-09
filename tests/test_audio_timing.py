"""Timing-lineage diagnostics for the audio worker (no device claim)."""

from __future__ import annotations

from mumax_sonic.audio import AudioConfig, AudioEngine
from mumax_sonic.audio import openal
from mumax_sonic.model import SonicScene, SonicSource
from test_audio import _FakeOpenAL, _until


def _scene(sim_time_s: float, validity: str = "valid") -> SonicScene:
    return SonicScene((SonicSource("one", (0.25, 0.0, -1.0), 0.5),),
                      sim_time_s=sim_time_s, validity=validity)


def test_last_applied_lineage_is_written_after_valid_controls_and_not_reused_for_invalid(monkeypatch, tmp_path):
    fake = _FakeOpenAL()
    dll = tmp_path / "soft_oal.dll"
    dll.touch()
    monkeypatch.setattr(openal, "_load_openal", lambda path: fake)
    engine = AudioEngine()
    engine.open(AudioConfig(dll_path=str(dll)))
    try:
        engine.update(_scene(12.5))
        _until(lambda: engine.diagnostics()["last_applied_sim_time_s"] == 12.5)
        valid = engine.diagnostics()
        assert valid["data_state"] == "current"
        assert isinstance(valid["last_control_applied_at_s"], float)
        assert valid["control_apply_latency_ms"] is not None

        engine.update(_scene(13.0, validity="invalid"))
        _until(lambda: engine.diagnostics()["data_state"] == "invalid")
        invalid = engine.diagnostics()
        assert invalid["scene_validity"] == "invalid"
        assert invalid["last_applied_sim_time_s"] == 12.5
        assert invalid["last_control_applied_at_s"] == valid["last_control_applied_at_s"]
    finally:
        engine.close()
