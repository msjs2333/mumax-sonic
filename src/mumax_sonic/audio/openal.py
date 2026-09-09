"""A deliberately small ctypes wrapper around the OpenAL Soft playback path.

The public thread never calls OpenAL.  It only replaces one immutable
``SonicScene``; the worker consumes the newest scene and owns the context for
its complete lifetime.  This is important on Windows, where contexts are
thread-local.
"""
from __future__ import annotations

import ctypes
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from math import cos, sin, sqrt, tau, exp
from pathlib import Path
from typing import Any

from mumax_sonic.model import MAX_SOURCE_BUDGET, SonicScene


# Core OpenAL / ALC values.  They are kept here to avoid a Python binding.
AL_NO_ERROR = 0
AL_NONE = 0
AL_PITCH = 0x1003
AL_POSITION = 0x1004
AL_LOOPING = 0x1007
AL_BUFFER = 0x1009
AL_GAIN = 0x100A
AL_SOURCE_STATE = 0x1010
AL_PLAYING = 0x1012
AL_VENDOR = 0xB001
AL_RENDERER = 0xB003
AL_VERSION = 0xB002
AL_FORMAT_MONO16 = 0x1101
AL_REFERENCE_DISTANCE = 0x1020
AL_ROLLOFF_FACTOR = 0x1021
AL_MAX_DISTANCE = 0x1023
AL_DOPPLER_FACTOR = 0xC000
AL_DISTANCE_MODEL = 0xD000

ALC_DEFAULT_DEVICE_SPECIFIER = 0x1004
ALC_DEVICE_SPECIFIER = 0x1005
ALC_EXTENSIONS = 0x1006
ALC_ALL_DEVICES_SPECIFIER = 0x1013
ALC_CONNECTED = 0x0313  # ALC_EXT_disconnect
ALC_HRTF_SOFT = 0x1992
ALC_HRTF_STATUS_SOFT = 0x1993

_SAMPLE_RATE = 48_000
_MAX_SOURCES = MAX_SOURCE_BUDGET
_SCENE_STALE_S = 0.75
_CONTROL_PERIOD_S = 0.01
_SMOOTH_TAU_S = 0.045


@dataclass(frozen=True)
class AudioConfig:
    """OpenAL Soft selection.  ``dll_path`` takes precedence over discovery."""

    dll_path: str | None = None
    device_name: str | None = None
    hrtf: bool = True


@dataclass
class _LiveSource:
    source: int
    sign: int
    gain: float = 0.0
    target_gain: float = 0.0
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    started: bool = False
    orientation_enabled: bool = False
    orientation_rad: float = 0.0
    pitch: float = 1.0
    modulation_hz: float = 4.0
    modulation_phase: float = 0.0


def orientation_controls(angle_rad):
    """Two periodic controls retain both cos(phi) and sin(phi), no wrap seam.

    This is a listening code, not a magnetic oscillation frequency.
    """
    return 2 ** (0.25*cos(angle_rad)), 4.0 + 2.0*sin(angle_rad)


def _dll_candidates(explicit: str | None = None) -> list[Path]:
    """Return only explicit OpenAL Soft locations; never fall back to system AL."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = os.environ.get("MUMAX_SONIC_OPENAL")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    root = Path(__file__).resolve().parents[3]
    candidates.extend(
        (
            root / "local" / "openal-soft" / "bin" / "Win64" / "soft_oal.dll",
            root / "local" / "openal-soft" / "bin" / "soft_oal.dll",
            root / "local" / "openal-soft" / "soft_oal.dll",
        )
    )
    # Preserve ordering while accepting equivalent spellings only once.
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _resolve_dll(explicit: str | None = None) -> Path:
    if explicit:
        configured = Path(explicit).expanduser()
        if configured.is_file():
            return configured.resolve()
        raise RuntimeError(f"Configured OpenAL Soft DLL does not exist: {configured}")
    for candidate in _dll_candidates(explicit):
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in _dll_candidates(explicit))
    raise RuntimeError(
        "OpenAL Soft DLL was not found. Set AudioConfig(dll_path=...) or "
        f"MUMAX_SONIC_OPENAL. Searched: {searched}"
    )


def _load_openal(path: Path) -> Any:
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        raise RuntimeError("OpenAL Soft playback is currently supported on Windows only")
    try:
        return loader(str(path))
    except OSError as exc:
        raise RuntimeError(f"Could not load OpenAL Soft DLL {path}: {exc}") from exc


def _bind(lib: Any, name: str, restype: Any, argtypes: list[Any]) -> Any:
    try:
        function = getattr(lib, name)
    except AttributeError as exc:
        raise RuntimeError(f"OpenAL Soft DLL is missing required symbol {name}") from exc
    function.restype = restype
    function.argtypes = argtypes
    return function


class _AL:
    """Bound OpenAL functions.  Construct only on the audio worker thread."""

    def __init__(self, lib: Any):
        c_void_p = ctypes.c_void_p
        c_char_p = ctypes.c_char_p
        c_int = ctypes.c_int
        c_uint = ctypes.c_uint
        c_float = ctypes.c_float
        c_short = ctypes.c_short
        self.alcOpenDevice = _bind(lib, "alcOpenDevice", c_void_p, [c_char_p])
        self.alcCloseDevice = _bind(lib, "alcCloseDevice", ctypes.c_bool, [c_void_p])
        self.alcCreateContext = _bind(lib, "alcCreateContext", c_void_p, [c_void_p, ctypes.POINTER(c_int)])
        self.alcMakeContextCurrent = _bind(lib, "alcMakeContextCurrent", ctypes.c_bool, [c_void_p])
        self.alcDestroyContext = _bind(lib, "alcDestroyContext", None, [c_void_p])
        self.alcGetString = _bind(lib, "alcGetString", c_void_p, [c_void_p, c_int])
        self.alcGetIntegerv = _bind(lib, "alcGetIntegerv", None, [c_void_p, c_int, c_int, ctypes.POINTER(c_int)])
        self.alcGetError = _bind(lib, "alcGetError", c_int, [c_void_p])
        self.alGetError = _bind(lib, "alGetError", c_int, [])
        self.alGetString = _bind(lib, "alGetString", c_char_p, [c_int])
        self.alDistanceModel = _bind(lib, "alDistanceModel", None, [c_int])
        self.alDopplerFactor = _bind(lib, "alDopplerFactor", None, [c_float])
        self.alGenBuffers = _bind(lib, "alGenBuffers", None, [c_int, ctypes.POINTER(c_uint)])
        self.alDeleteBuffers = _bind(lib, "alDeleteBuffers", None, [c_int, ctypes.POINTER(c_uint)])
        self.alBufferData = _bind(lib, "alBufferData", None, [c_uint, c_int, c_void_p, c_int, c_int])
        self.alGenSources = _bind(lib, "alGenSources", None, [c_int, ctypes.POINTER(c_uint)])
        self.alDeleteSources = _bind(lib, "alDeleteSources", None, [c_int, ctypes.POINTER(c_uint)])
        self.alSourcei = _bind(lib, "alSourcei", None, [c_uint, c_int, c_int])
        self.alSourcef = _bind(lib, "alSourcef", None, [c_uint, c_int, c_float])
        self.alSource3f = _bind(lib, "alSource3f", None, [c_uint, c_int, c_float, c_float, c_float])
        self.alSourcePlay = _bind(lib, "alSourcePlay", None, [c_uint])
        self.alSourceStop = _bind(lib, "alSourceStop", None, [c_uint])
        self.alGetSourcei = _bind(lib, "alGetSourcei", None, [c_uint, c_int, ctypes.POINTER(c_int)])
        self.c_short = c_short


def _tone_pcm(sign: int) -> tuple[Any, int]:
    """Create an equal-RMS, equal-peak periodic mono clip for a sign channel."""
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    # Integer cycle counts per 100 ms make the loop continuous.  Harmonic
    # spectra distinguish the channels without treating negative audio phase
    # as a physical sign encoding.
    count = _SAMPLE_RATE // 10
    base = 220 if sign > 0 else 330
    values = [sin(tau * base * index / _SAMPLE_RATE) + 0.38 * sin(tau * 2 * base * index / _SAMPLE_RATE)
              for index in range(count)]
    rms = sqrt(sum(value * value for value in values) / count)
    scale = 0.25 / rms
    peak = max(abs(value * scale) for value in values)
    if peak > 0.80:
        scale *= 0.80 / peak
    pcm = (ctypes.c_short * count)(*(round(max(-1.0, min(1.0, value * scale)) * 32767) for value in values))
    return pcm, _SAMPLE_RATE


def _pointer_value(pointer: Any) -> int:
    return int(pointer.value) if isinstance(pointer, ctypes.c_void_p) else int(pointer or 0)


def _decode_alc_text(value: bytes) -> str:
    """Decode Windows endpoint labels using the process ANSI code page.

    OpenAL's ALCchar device names are 8-bit strings; Windows backends commonly
    return the local endpoint label rather than UTF-8.  ASCII renderer strings
    remain unchanged, while Chinese/Japanese device labels stay displayable.
    """
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        # Some older Windows backends use the process ANSI code page even
        # though OpenAL Soft's enumerated endpoint strings are UTF-8.
        return value.decode("mbcs" if os.name == "nt" else "utf-8", "replace")


def _read_c_string(pointer: Any) -> str:
    address = _pointer_value(pointer)
    return _decode_alc_text(ctypes.string_at(address)) if address else ""


def _read_multistring(pointer: Any, limit: int = 16_384) -> list[str]:
    address = _pointer_value(pointer)
    if not address:
        return []
    values: list[str] = []
    start = 0
    for offset in range(limit):
        if ctypes.string_at(address + offset, 1) != b"\0":
            continue
        if offset == start:
            return values
        values.append(_decode_alc_text(ctypes.string_at(address + start, offset - start)))
        start = offset + 1
    return values


def enumerate_devices(dll_path: str | None = None) -> list[str]:
    """List OpenAL Soft playback device names without opening a context."""
    lib = _load_openal(_resolve_dll(dll_path))
    al = _AL(lib)
    extensions = _read_c_string(al.alcGetString(None, ALC_EXTENSIONS))
    token = ALC_ALL_DEVICES_SPECIFIER if "ALC_ENUMERATE_ALL_EXT" in extensions else ALC_DEVICE_SPECIFIER
    devices = _read_multistring(al.alcGetString(None, token))
    if devices:
        return devices
    default = _read_c_string(al.alcGetString(None, ALC_DEFAULT_DEVICE_SPECIFIER))
    return [default] if default else []


class AudioEngine:
    """Latest-wins spatial source renderer backed by a single OpenAL worker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._stop_requested = False
        self._close_requested = False
        self._latest: tuple[SonicScene, float, int] | None = None
        self._last_received_at: float | None = None
        self._sequence = 0
        self._thread: threading.Thread | None = None
        self._config: AudioConfig | None = None
        self._diagnostic: dict[str, Any] = self._blank_diagnostics()

    @staticmethod
    def _blank_diagnostics() -> dict[str, Any]:
        return {
            "state": "closed", "dll_path": None, "device": None, "renderer": None,
            "requested_device": None, "vendor": None, "version": None, "hrtf_requested": False, "hrtf_status": "unknown",
            "connected": None, "orientation_mapping": "optional periodic pitch/tremolo; P1 labels unchanged",
            "last_error": None, "unexpected_stopped_sources": 0,
            "source_capacity": MAX_SOURCE_BUDGET, "active_source_count": 0,
            "adaptive_voice_matching": "nearest within sign and foreground/background role",
            "adaptive_smoothing_tau_ms": 120,
            "target_source_count": 0,
            "control_apply_latency_ms": None, "control_apply_latency_p95_ms": None,
            "dropped_updates": 0, "scene_age_ms": None, "scene_validity": None,
            "data_state": "no_scene", "data_message": None,
        }

    def open(self, config: AudioConfig | None = None) -> None:
        """Open the selected OpenAL Soft device; errors name the failed stage."""
        config = config or AudioConfig()
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("AudioEngine is already open; call close() before reopening")
            self._config = config
            self._ready.clear()
            self._closed.clear()
            self._close_requested = False
            self._stop_requested = False
            self._latest = None
            self._last_received_at = None
            self._sequence = 0
            self._diagnostic = self._blank_diagnostics()
            self._diagnostic["state"] = "opening"
            self._diagnostic["hrtf_requested"] = config.hrtf
            self._thread = threading.Thread(target=self._run, name="mumax-sonic-openal", daemon=True)
            self._thread.start()
        if not self._ready.wait(5.0):
            self.close()
            raise RuntimeError("Timed out while opening OpenAL Soft audio worker")
        diagnostic = self.diagnostics()
        if diagnostic["state"] != "open":
            self.close()
            raise RuntimeError(f"Could not open OpenAL Soft: {diagnostic['last_error']}")

    def update(self, scene: SonicScene) -> None:
        """Queue a scene without waiting for audio I/O; only the newest survives."""
        if not isinstance(scene, SonicScene):
            raise TypeError("update() requires a SonicScene")
        now = time.monotonic()
        with self._lock:
            state = self._diagnostic["state"]
            if self._thread is None or self._close_requested or state != "open":
                if state == "disconnected":
                    raise RuntimeError("OpenAL device is disconnected; call close() then open() to recover")
                if state == "error":
                    raise RuntimeError(f"AudioEngine worker failed: {self._diagnostic['last_error']}")
                raise RuntimeError("AudioEngine is not open")
            if self._latest is not None:
                self._diagnostic["dropped_updates"] += 1
            self._sequence += 1
            self._latest = (scene, now, self._sequence)
            self._last_received_at = now
            self._stop_requested = False  # a new immutable scene resumes output
        self._wake.set()

    def stop(self) -> None:
        """Request immediate silence.  A later update() resumes playback."""
        with self._lock:
            if self._thread is None:
                return
            self._stop_requested = True
            self._latest = None
            self._last_received_at = None
        self._wake.set()

    def close(self) -> None:
        """Release sources, context, device, and worker.  It is safe to repeat."""
        with self._lock:
            thread = self._thread
            if thread is None:
                return
            self._close_requested = True
            self._latest = None
        self._wake.set()
        thread.join(5.0)
        if thread.is_alive():
            raise RuntimeError("OpenAL audio worker did not close within 5 seconds")
        with self._lock:
            self._thread = None

    def diagnostics(self) -> dict[str, Any]:
        """Return a detached, JSON-serializable status snapshot."""
        with self._lock:
            return dict(self._diagnostic)

    def _set_diagnostic(self, **values: Any) -> None:
        with self._lock:
            self._diagnostic.update(values)

    def _check_al_error(self, al: _AL, prefix: str) -> bool:
        error = int(al.alGetError())
        if error != AL_NO_ERROR:
            self._set_diagnostic(last_error=f"{prefix}: AL error 0x{error:04x}")
            return False
        return True

    def _run(self) -> None:
        al: _AL | None = None
        device: Any = None
        context: Any = None
        buffers: dict[int, int] = {}
        sources: dict[str, _LiveSource] = {}
        from .continuity import AdaptiveVoiceTracker
        adaptive_voices = AdaptiveVoiceTracker()
        # Active counts allocated OpenAL voices, including voices fading after
        # they leave the latest scene. Target counts the latest desired scene.
        target_source_count = 0
        latency_samples: deque[float] = deque(maxlen=128)
        last_tick = time.monotonic()
        try:
            assert self._config is not None
            dll = _resolve_dll(self._config.dll_path)
            self._set_diagnostic(dll_path=str(dll))
            al = _AL(_load_openal(dll))
            requested_device = self._config.device_name.encode("utf-8") if self._config.device_name else None
            device = al.alcOpenDevice(requested_device)
            if not device:
                raise RuntimeError("alcOpenDevice returned no device")
            attributes = (ctypes.c_int * 3)(ALC_HRTF_SOFT, 1 if self._config.hrtf else 0, 0)
            context = al.alcCreateContext(device, attributes)
            if not context:
                error = int(al.alcGetError(device))
                raise RuntimeError(f"alcCreateContext failed (ALC error 0x{error:04x})")
            if not al.alcMakeContextCurrent(context):
                raise RuntimeError("alcMakeContextCurrent failed")
            al.alDistanceModel(AL_NONE)
            al.alDopplerFactor(0.0)
            if not self._check_al_error(al, "renderer setup"):
                raise RuntimeError(self.diagnostics()["last_error"])
            for sign in (-1, 1):
                buffer_id = ctypes.c_uint()
                al.alGenBuffers(1, ctypes.byref(buffer_id))
                pcm, rate = _tone_pcm(sign)
                al.alBufferData(buffer_id.value, AL_FORMAT_MONO16, ctypes.cast(pcm, ctypes.c_void_p), ctypes.sizeof(pcm), rate)
                buffers[sign] = buffer_id.value
            if not self._check_al_error(al, "tone buffer upload"):
                raise RuntimeError(self.diagnostics()["last_error"])
            hrtf_value = ctypes.c_int(-1)
            al.alcGetIntegerv(device, ALC_HRTF_STATUS_SOFT, 1, ctypes.byref(hrtf_value))
            hrtf_status = {0: "disabled", 1: "enabled", 2: "denied", 3: "required", 4: "headphones_detected", 5: "unsupported_format"}.get(hrtf_value.value, "unavailable")
            device_extensions = _read_c_string(al.alcGetString(device, ALC_EXTENSIONS))
            # ALC_DEVICE_SPECIFIER may only say "OpenAL Soft".  The ALL
            # extension reports the actual endpoint selected by the backend.
            actual_device_token = ALC_ALL_DEVICES_SPECIFIER if "ALC_ENUMERATE_ALL_EXT" in device_extensions else ALC_DEVICE_SPECIFIER
            self._set_diagnostic(
                state="open",
                requested_device=self._config.device_name,
                device=_read_c_string(al.alcGetString(device, actual_device_token)),
                renderer=(al.alGetString(AL_RENDERER) or b"").decode("utf-8", "replace"),
                vendor=(al.alGetString(AL_VENDOR) or b"").decode("utf-8", "replace"),
                version=(al.alGetString(AL_VERSION) or b"").decode("utf-8", "replace"),
                hrtf_status=hrtf_status,
                connected=True,
            )
            self._ready.set()
            while True:
                self._wake.wait(_CONTROL_PERIOD_S)
                self._wake.clear()
                now = time.monotonic()
                applied_submitted_at: float | None = None
                with self._lock:
                    closing = self._close_requested
                    stopping = self._stop_requested
                    pending = self._latest
                    self._latest = None  # one-slot consume: only unconsumed frames are drops
                    last_received_at = self._last_received_at
                if closing:
                    break
                if last_received_at is not None:
                    self._set_diagnostic(scene_age_ms=round(max(0.0, now - last_received_at) * 1000.0, 3))
                if stopping:
                    for live in sources.values():
                        al.alSourceStop(live.source)
                        al.alSourcef(live.source, AL_GAIN, 0.0)
                        live.gain = live.target_gain = 0.0
                        live.started = False
                    with self._lock:
                        self._stop_requested = False
                    target_source_count = 0
                    self._set_diagnostic(
                        active_source_count=len(sources), target_source_count=0,
                        data_state="stopped", data_message=None,
                    )
                    continue
                if "ALC_EXT_disconnect" in device_extensions:
                    connected = ctypes.c_int(1)
                    al.alcGetIntegerv(device, ALC_CONNECTED, 1, ctypes.byref(connected))
                    if not connected.value:
                        for live in sources.values():
                            al.alSourceStop(live.source)
                            al.alSourcef(live.source, AL_GAIN, 0.0)
                        self._set_diagnostic(connected=False, state="disconnected", last_error="OpenAL device disconnected; close() then open() to recover")
                        continue
                if pending is not None:
                    scene, submitted, _sequence = pending
                    scene = adaptive_voices.map(scene)
                    age = now - submitted
                    if scene.validity == "stale" or age > _SCENE_STALE_S:
                        for live in sources.values():
                            # Staleness is a data-quality state, rather than a
                            # quiet physical value: do not leave a fading tail
                            # that could be mistaken for current data.
                            al.alSourceStop(live.source)
                            al.alSourcef(live.source, AL_GAIN, 0.0)
                            live.gain = 0.0
                            live.target_gain = 0.0
                            live.started = False
                        target_source_count = 0
                        self._set_diagnostic(scene_validity="stale", data_state="stale", data_message="scene is stale; sources muted")
                    elif scene.validity != "valid":
                        for live in sources.values():
                            live.target_gain = 0.0
                        target_source_count = 0
                        self._set_diagnostic(scene_validity=scene.validity, data_state=scene.validity,
                                             data_message=f"scene validity is {scene.validity}; sources muted")
                    else:
                        desired = scene.sources[:_MAX_SOURCES]
                        target_source_count = len(desired)
                        desired_ids = {item.source_id for item in desired}
                        seen = set()
                        for item in desired:
                            seen.add(item.source_id)
                            live = sources.get(item.source_id)
                            if live is None:
                                # Keep active plus fading voices within one fixed source budget.
                                while len(sources) >= _MAX_SOURCES:
                                    candidates = [key for key in sources if key not in desired_ids]
                                    if not candidates:
                                        candidates = list(sources)
                                    evicted_id = min(candidates, key=lambda key: (sources[key].target_gain, sources[key].gain))
                                    evicted = sources.pop(evicted_id)
                                    al.alSourceStop(evicted.source)
                                    al.alDeleteSources(1, ctypes.byref(ctypes.c_uint(evicted.source)))
                                source_id = ctypes.c_uint()
                                al.alGenSources(1, ctypes.byref(source_id))
                                al.alSourcei(source_id.value, AL_BUFFER, buffers[item.sign])
                                al.alSourcei(source_id.value, AL_LOOPING, 1)
                                al.alSourcef(source_id.value, AL_PITCH, 1.0)
                                al.alSourcef(source_id.value, AL_ROLLOFF_FACTOR, 0.0)
                                al.alSourcef(source_id.value, AL_REFERENCE_DISTANCE, 1.0)
                                al.alSourcef(source_id.value, AL_MAX_DISTANCE, 1.0)
                                live = _LiveSource(source=source_id.value, sign=item.sign,
                                                   position=item.position, target_position=item.position)
                                sources[item.source_id] = live
                            elif live.sign != item.sign:
                                # Rebinding is needed because sign selects a different loop buffer.
                                al.alSourceStop(live.source)
                                al.alSourcei(live.source, AL_BUFFER, buffers[item.sign])
                                live.sign = item.sign
                                live.started = False
                            live.target_gain = item.gain
                            live.target_position = item.position
                            live.orientation_enabled = item.orientation_enabled
                            live.orientation_rad = item.orientation_rad
                        for source_id, live in sources.items():
                            if source_id not in seen:
                                live.target_gain = 0.0
                        applied_submitted_at = submitted
                        self._set_diagnostic(scene_validity="valid", data_state="current", data_message=None)
                elif last_received_at is not None and now - last_received_at > _SCENE_STALE_S:
                    # A producer that stops calling update is stale too.  It is
                    # not represented by a valid zero-gain scene.
                    if self.diagnostics()["data_state"] != "stale":
                        for live in sources.values():
                            al.alSourceStop(live.source)
                            al.alSourcef(live.source, AL_GAIN, 0.0)
                            live.gain = live.target_gain = 0.0
                            live.started = False
                        target_source_count = 0
                        self._set_diagnostic(scene_validity="stale", data_state="stale",
                                             data_message="no scene update within freshness timeout; sources muted")
                dt = min(0.10, max(0.0, now - last_tick))
                last_tick = now
                alpha = 1.0 if dt == 0 else min(1.0, dt / _SMOOTH_TAU_S)
                for source_id, live in list(sources.items()):
                    # Spatial partitions may change abruptly as attention moves.
                    # Keep their renderer voices alive and soften the retargeting.
                    spatial_alpha = 1 - exp(-dt / 0.12) if source_id.startswith('sonic-adaptive:') else alpha
                    live.gain += (live.target_gain - live.gain) * spatial_alpha
                    live.position = tuple(current + (target - current) * spatial_alpha for current, target in zip(live.position, live.target_position))
                    pitch, rate = orientation_controls(live.orientation_rad) if live.orientation_enabled else (1.0, 4.0)
                    live.pitch += (pitch-live.pitch)*alpha
                    live.modulation_hz += (rate-live.modulation_hz)*alpha
                    live.modulation_phase = (live.modulation_phase + tau*live.modulation_hz*dt) % tau
                    modulation = (1 + 0.25*sin(live.modulation_phase))/1.25 if live.orientation_enabled else 1.0
                    al.alSourcef(live.source, AL_PITCH, live.pitch)
                    al.alSourcef(live.source, AL_GAIN, max(0.0, live.gain)*modulation)
                    al.alSource3f(live.source, AL_POSITION, *live.position)
                    if live.gain > 0.0001:
                        state = ctypes.c_int()
                        al.alGetSourcei(live.source, AL_SOURCE_STATE, ctypes.byref(state))
                        if state.value != AL_PLAYING:
                            if live.started:
                                with self._lock:
                                    self._diagnostic["unexpected_stopped_sources"] += 1
                            al.alSourcePlay(live.source)
                            live.started = True
                    if live.target_gain == 0.0 and live.gain < 0.0001:
                        al.alSourceStop(live.source)
                        al.alDeleteSources(1, ctypes.byref(ctypes.c_uint(live.source)))
                        del sources[source_id]
                self._set_diagnostic(active_source_count=len(sources), target_source_count=target_source_count)
                if applied_submitted_at is not None:
                    # This ends after alSourcef/alSource3f have submitted the
                    # new controls.  It is not a device-buffer or acoustic latency.
                    latency_ms = max(0.0, (time.monotonic() - applied_submitted_at) * 1000.0)
                    latency_samples.append(latency_ms)
                    ordered = sorted(latency_samples)
                    p95 = ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]
                    self._set_diagnostic(control_apply_latency_ms=round(latency_ms, 3), control_apply_latency_p95_ms=round(p95, 3))
                if not self._check_al_error(al, "source control"):
                    for live in sources.values():
                        al.alSourceStop(live.source)
                        al.alSourcef(live.source, AL_GAIN, 0.0)
                        live.gain = live.target_gain = 0.0
                        live.started = False
                    raise RuntimeError(self.diagnostics()["last_error"])
        except Exception as exc:
            self._set_diagnostic(state="error", last_error=str(exc), connected=False)
            self._ready.set()
        finally:
            if al is not None:
                for live in sources.values():
                    al.alSourceStop(live.source)
                    al.alDeleteSources(1, ctypes.byref(ctypes.c_uint(live.source)))
                for buffer_id in buffers.values():
                    al.alDeleteBuffers(1, ctypes.byref(ctypes.c_uint(buffer_id)))
            if al is not None and context:
                al.alcMakeContextCurrent(None)
                al.alcDestroyContext(context)
            if al is not None and device:
                al.alcCloseDevice(device)
            self._set_diagnostic(active_source_count=0, target_source_count=0)
            self._set_diagnostic(state="closed" if self.diagnostics()["state"] != "error" else "error")
            self._closed.set()
            self._ready.set()
