from __future__ import annotations

import ctypes
import json
import threading
import time

import pytest

from mumax_sonic.audio import AudioConfig, AudioEngine
from mumax_sonic.audio import openal
from mumax_sonic.model import SonicScene, SonicSource


class _Function:
    def __init__(self, callback=lambda *args: None):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class _FakeOpenAL:
    """Enough of the ABI to check ownership and control behavior, not audio."""

    def __init__(self):
        self.calls = []
        self.buffers = 10
        self.sources = 20
        self.source_state = {}
        self.source_gains = {}
        self.source_positions = {}
        self.stop_calls = []
        self.played = threading.Event()
        self.stopped = threading.Event()
        self.gain_applied = threading.Event()
        self.control_entered = threading.Event()
        self.control_gate = threading.Event()
        self.block_controls = False
        self.extension_reads = 0
        self.source_control_entered = threading.Event()
        self.source_control_gate = threading.Event()
        self.block_source_control = False
        self.al_error = 0
        self.runtime_error_on_gain = False
        self._strings = [ctypes.create_string_buffer(value) for value in (
            b"Fake Device\0", b"ALC_EXT_disconnect ALC_ENUMERATE_ALL_EXT\0",
            b"Fake Renderer\0", b"Fake Vendor\0", b"1.1\0", b"Fake Device\0\0",
        )]
        self.alcOpenDevice = _Function(lambda name: 1)
        self.alcCloseDevice = _Function(lambda device: True)
        self.alcCreateContext = _Function(lambda device, attrs: 2)
        self.alcMakeContextCurrent = _Function(lambda context: True)
        self.alcDestroyContext = _Function()
        self.alcGetError = _Function(lambda device: 0)
        self.alGetError = _Function(self._get_al_error)
        self.alcGetString = _Function(self._get_string)
        self.alcGetIntegerv = _Function(self._get_int)
        self.alGetString = _Function(self._get_al_string)
        self.alDistanceModel = _Function(self._record)
        self.alDopplerFactor = _Function(self._record)
        self.alGenBuffers = _Function(self._gen_buffer)
        self.alDeleteBuffers = _Function(self._record)
        self.alBufferData = _Function(self._record)
        self.alGenSources = _Function(self._gen_source)
        self.alDeleteSources = _Function(self._record)
        self.alSourcei = _Function(self._record)
        self.alSourcef = _Function(self._sourcef)
        self.alSource3f = _Function(self._source3f)
        self.alSourcePlay = _Function(self._play)
        self.alSourceStop = _Function(self._stop)
        self.alGetSourcei = _Function(self._source_int)

    @staticmethod
    def _value(value):
        return getattr(value, "value", value)

    def _record(self, *args):
        self.calls.append((threading.get_ident(), args))

    def _get_string(self, device, token):
        token = self._value(token)
        if token == openal.ALC_EXTENSIONS:
            self.extension_reads += 1
            if self.block_controls and self.extension_reads > 1:
                self.control_entered.set()
                self.control_gate.wait(1.0)
        index = {openal.ALC_DEVICE_SPECIFIER: 0, openal.ALC_EXTENSIONS: 1,
                 openal.ALC_ALL_DEVICES_SPECIFIER: 5}.get(token)
        if index is None:
            return 0
        return ctypes.addressof(self._strings[index])

    def _get_int(self, device, token, count, destination):
        token = self._value(token)
        destination._obj.value = 1 if token in (openal.ALC_CONNECTED, openal.ALC_HRTF_STATUS_SOFT) else 0

    def _get_al_string(self, token):
        token = self._value(token)
        return {openal.AL_RENDERER: b"Fake Renderer", openal.AL_VENDOR: b"Fake Vendor", openal.AL_VERSION: b"1.1"}.get(token)

    def _get_al_error(self):
        error, self.al_error = self.al_error, 0
        return error

    def _gen_buffer(self, count, destination):
        destination._obj.value = self.buffers
        self.buffers += 1

    def _gen_source(self, count, destination):
        destination._obj.value = self.sources
        self.source_state[self.sources] = 0
        self.sources += 1

    def _play(self, source):
        self.source_state[self._value(source)] = openal.AL_PLAYING
        self.played.set()
        self._record(source)

    def _stop(self, source):
        source = self._value(source)
        self.source_state[source] = 0
        self.stop_calls.append(source)
        self.stopped.set()
        self._record(source)

    def _sourcef(self, source, parameter, value):
        source = self._value(source)
        parameter = self._value(parameter)
        if parameter == openal.AL_GAIN:
            self.source_gains[source] = float(self._value(value))
            if self.source_gains[source] > 0:
                self.gain_applied.set()
                if self.runtime_error_on_gain:
                    self.al_error = 0xA004
        self._record(source, parameter, value)

    def _source3f(self, source, parameter, x, y, z):
        source = self._value(source)
        if self._value(parameter) == openal.AL_POSITION:
            self.source_positions[source] = tuple(float(self._value(v)) for v in (x, y, z))
            if self.block_source_control:
                self.source_control_entered.set()
                self.source_control_gate.wait(1.0)
        self._record(source, parameter, x, y, z)

    def _source_int(self, source, token, destination):
        destination._obj.value = self.source_state.get(self._value(source), 0)


def _scene(sign=1, gain=0.5, position=(0.5, 0.0, -1.0)):
    return SonicScene((SonicSource("one", position, gain, sign),), sim_time_s=1.0)


def _until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(0.005)
    assert predicate()


def test_sign_loops_have_controlled_equal_rms_and_peak():
    signals = []
    for sign in (-1, 1):
        pcm, rate = openal._tone_pcm(sign)
        values = [value / 32767 for value in pcm]
        rms = (sum(value * value for value in values) / len(values)) ** 0.5
        signals.append((rate, rms, max(map(abs, values)), bytes(pcm)))
    assert signals[0][0] == signals[1][0] == 48_000
    assert signals[0][1] == pytest.approx(signals[1][1], abs=1e-4)
    assert signals[0][2] == pytest.approx(signals[1][2], abs=1e-4)
    assert signals[0][3] != signals[1][3]


def test_engine_latest_wins_stops_and_reports_control_latency(monkeypatch, tmp_path):
    fake = _FakeOpenAL()
    fake.block_source_control = True
    dll = tmp_path / "soft_oal.dll"
    dll.touch()
    monkeypatch.setattr(openal, "_load_openal", lambda path: fake)
    engine = AudioEngine()
    engine.open(AudioConfig(dll_path=str(dll)))
    engine.update(_scene(1, 0.2, (-0.25, 0.0, -1.0)))
    assert fake.source_control_entered.wait(1.0)
    engine.update(_scene(1, 0.4, (0.0, 0.0, -1.0)))
    engine.update(_scene(-1, 0.8, (0.75, 0.0, -1.0)))
    fake.source_control_gate.set()
    assert fake.played.wait(1.0)
    assert fake.gain_applied.wait(1.0)
    _until(lambda: engine._latest is None and engine.diagnostics()["control_apply_latency_ms"] is not None)
    diagnostic = engine.diagnostics()
    assert diagnostic["state"] == "open"
    assert diagnostic["renderer"] == "Fake Renderer"
    assert diagnostic["hrtf_status"] == "enabled"
    assert diagnostic["dropped_updates"] == 1
    assert diagnostic["control_apply_latency_ms"] is not None
    assert diagnostic["orientation_mapping"] == "not implemented in P1"
    json.dumps(diagnostic)
    assert fake.source_gains[20] > 0
    drops_after_burst = diagnostic["dropped_updates"]
    engine.update(_scene(-1, 0.3, (0.1, 0.0, -1.0)))
    _until(lambda: engine._latest is None)
    assert engine.diagnostics()["dropped_updates"] == drops_after_burst
    fake.stopped.clear()
    engine.stop()
    assert fake.stopped.wait(1.0)
    assert fake.source_state[20] == 0
    assert fake.source_gains[20] == 0
    fake.played.clear()
    engine.update(_scene(1, 0.4))
    assert fake.played.wait(1.0)
    engine.close()
    assert engine.diagnostics()["state"] == "closed"
    worker_threads = {thread_id for thread_id, _ in fake.calls}
    assert len(worker_threads) == 1


def test_explicit_stale_scene_stops_output_and_valid_scene_recovers(monkeypatch, tmp_path):
    fake = _FakeOpenAL()
    dll = tmp_path / "soft_oal.dll"
    dll.touch()
    monkeypatch.setattr(openal, "_load_openal", lambda path: fake)
    engine = AudioEngine()
    engine.open(AudioConfig(dll_path=str(dll)))
    engine.update(_scene())
    assert fake.played.wait(1.0)
    assert fake.gain_applied.wait(1.0)
    fake.stopped.clear()
    engine.update(SonicScene((_scene().sources[0],), validity="stale"))
    assert fake.stopped.wait(1.0)
    _until(lambda: engine.diagnostics()["data_state"] == "stale")
    assert fake.source_state[20] == 0
    assert fake.source_gains[20] == 0
    assert engine.diagnostics()["last_error"] is None
    fake.played.clear()
    engine.update(_scene())
    assert fake.played.wait(1.0)
    _until(lambda: engine.diagnostics()["data_state"] == "current")
    assert engine.diagnostics()["data_message"] is None
    engine.close()


def test_no_update_after_a_valid_scene_expires_and_mutes(monkeypatch, tmp_path):
    monkeypatch.setattr(openal, "_SCENE_STALE_S", 0.03)
    fake = _FakeOpenAL()
    dll = tmp_path / "soft_oal.dll"
    dll.touch()
    monkeypatch.setattr(openal, "_load_openal", lambda path: fake)
    engine = AudioEngine()
    engine.open(AudioConfig(dll_path=str(dll)))
    engine.update(_scene())
    assert fake.played.wait(1.0)
    fake.stopped.clear()
    assert fake.stopped.wait(1.0)
    _until(lambda: engine.diagnostics()["data_state"] == "stale")
    assert engine.diagnostics()["scene_age_ms"] >= 30
    assert fake.source_state[20] == 0
    assert fake.source_gains[20] == 0
    engine.close()


def test_runtime_openal_error_mutes_and_rejects_new_updates(monkeypatch, tmp_path):
    fake = _FakeOpenAL()
    fake.runtime_error_on_gain = True
    dll = tmp_path / "soft_oal.dll"
    dll.touch()
    monkeypatch.setattr(openal, "_load_openal", lambda path: fake)
    engine = AudioEngine()
    engine.open(AudioConfig(dll_path=str(dll)))
    engine.update(_scene())
    assert fake.gain_applied.wait(1.0)
    _until(lambda: engine.diagnostics()["state"] == "error")
    assert fake.source_state[20] == 0
    assert fake.source_gains[20] == 0
    with pytest.raises(RuntimeError, match="worker failed"):
        engine.update(_scene())
    engine.close()


def test_missing_dll_is_explicit():
    engine = AudioEngine()
    with pytest.raises(RuntimeError, match="Configured OpenAL Soft DLL does not exist"):
        engine.open(AudioConfig(dll_path="does-not-exist.dll"))


def test_enumerate_devices_uses_openal_soft_library(monkeypatch, tmp_path):
    fake = _FakeOpenAL()
    dll = tmp_path / "soft_oal.dll"
    dll.touch()
    monkeypatch.setattr(openal, "_load_openal", lambda path: fake)
    assert openal.enumerate_devices(str(dll)) == ["Fake Device"]
