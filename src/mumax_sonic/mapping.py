"""Explicit shared-strength scale and physical-to-auditory coordinates."""
from math import cos, sin, pi, sqrt, isfinite
from .attention import Attention, select_sources
from .model import Sample, SonicScene, SonicSource


def map_sample(sample: Sample, attention: Attention, *, mode: str = "both",
               master_gain: float = 0.15, budget: int = 4, audible: bool = True,
               strength_reference: float = 1.0) -> SonicScene:
    if mode not in {"both", "positive", "negative"}:
        raise ValueError("unknown sign mode")
    if not isfinite(master_gain) or not 0 <= master_gain <= 1:
        raise ValueError("master gain must be in [0,1]")
    if budget < 1:
        raise ValueError("budget must be positive")
    if not isfinite(strength_reference) or strength_reference <= 0:
        raise ValueError('strength reference must be finite and positive')
    if not audible or sample.validity != "valid" or sample.coverage < 1:
        state = "invalid" if sample.validity == "valid" and sample.coverage < 1 else sample.validity
        return SonicScene((), sample.sim_time_s, state)
    candidates = tuple(o for o in sample.observations
                       if mode == "both" or o.sign == (1 if mode == "positive" else -1))
    result = []
    for o in select_sources(candidates, attention, budget):
        x, y, _ = o.position_m
        azimuth = max(-1, min(1, (x-attention.origin_m[0]) / attention.extent_m)) * pi / 3
        elevation = max(-1, min(1, (y-attention.origin_m[1]) / attention.extent_m)) * pi / 6
        position = (sin(azimuth)*cos(elevation), sin(elevation), -cos(azimuth)*cos(elevation))
        # Same fixed reference (strength=1) and bus headroom for both signs.
        # No per-frame/per-sign normalization, including during solo.
        gain = master_gain / budget * sqrt(min(o.strength / strength_reference, 1.0)) * attention.weight(o)
        result.append(SonicSource(o.source_id, position, gain, o.sign, o.orientation_rad, o.orientation_enabled))
    return SonicScene(tuple(result), sample.sim_time_s)
