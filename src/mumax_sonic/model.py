"""Immutable P1 contracts. Coordinates in metres, time in simulation seconds.

P1 consumes prelabelled observations; full vector-field acquisition belongs to P2.
"""
from dataclasses import dataclass
from math import isfinite

VALIDITIES = {"valid", "invalid", "stale", "warming_up", "unsupported"}


@dataclass(frozen=True)
class Observation:
    source_id: str
    position_m: tuple[float, float, float]
    strength: float
    sign: int = 1
    orientation_rad: float = 0.0
    entity_id: str = "demo"
    quantity: str = "synthetic_strength"
    unit: str = "1"

    def __post_init__(self):
        object.__setattr__(self, "position_m", tuple(self.position_m))
        if not self.source_id or len(self.position_m) != 3:
            raise ValueError("source_id and three physical coordinates required")
        if not all(isfinite(v) for v in (*self.position_m, self.strength, self.orientation_rad)):
            raise ValueError("observation must contain finite values")
        if self.strength < 0 or self.sign not in (-1, 1):
            raise ValueError("strength must be nonnegative; sign must be -1 or +1")


@dataclass(frozen=True)
class Sample:
    sim_time_s: float
    observations: tuple[Observation, ...]
    sequence: int = 0
    segment_id: str = "demo"
    validity: str = "valid"
    coverage: float = 1.0
    source_kind: str = "synthetic"
    schema_version: int = 1
    time_kind: str = "dynamics"

    def __post_init__(self):
        object.__setattr__(self, "observations", tuple(self.observations))
        if not isfinite(self.sim_time_s) or self.sim_time_s < 0 or type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("invalid simulation time or sequence")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported sample schema version")
        if self.validity not in VALIDITIES or not 0 <= self.coverage <= 1:
            raise ValueError("invalid quality state")
        if self.source_kind not in {"synthetic", "replay", "live"}:
            raise ValueError("unknown source kind")
        if self.time_kind not in {"static", "dynamics", "relaxation"}:
            raise ValueError("unknown time kind")
        ids = [o.source_id for o in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate observation ID")


@dataclass(frozen=True)
class SonicSource:
    source_id: str
    position: tuple[float, float, float]  # OpenAL coordinates: +x right, +y up, -z front
    gain: float                         # nonnegative linear amplitude
    sign: int = 1
    orientation_rad: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "position", tuple(self.position))
        if not self.source_id or len(self.position) != 3:
            raise ValueError("source ID and three audio coordinates required")
        if not all(isfinite(v) for v in (*self.position, self.gain, self.orientation_rad)):
            raise ValueError("non-finite sound parameters")
        if not 0 <= self.gain <= 1 or self.sign not in (-1, 1):
            raise ValueError("invalid gain/sign")


@dataclass(frozen=True)
class SonicScene:
    sources: tuple[SonicSource, ...] = ()
    sim_time_s: float = 0.0
    validity: str = "valid"

    def __post_init__(self):
        object.__setattr__(self, "sources", tuple(self.sources))
        if self.validity not in VALIDITIES or not isfinite(self.sim_time_s) or self.sim_time_s < 0:
            raise ValueError("invalid scene")
        if len({s.source_id for s in self.sources}) != len(self.sources):
            raise ValueError("duplicate sound source ID")
