"""Attention acts on listening gain/selection, never on physical observations."""
from dataclasses import dataclass
from math import hypot, isfinite
from .model import Observation


@dataclass(frozen=True)
class Attention:
    center: tuple[float, float] = (0.0, 0.0)  # normalized viewport coordinates
    radius: float = 0.55
    background: float = 0.15
    extent_m: float = 1e-6
    origin_m: tuple[float, float] = (0.0, 0.0)  # viewport centre in physical XY

    def __post_init__(self):
        object.__setattr__(self, "center", tuple(self.center))
        object.__setattr__(self, "origin_m", tuple(self.origin_m))
        if len(self.origin_m) != 2 or not all(isfinite(v) for v in self.origin_m):
            raise ValueError("invalid viewport origin")
        if len(self.center) != 2 or not all(isfinite(v) for v in self.center):
            raise ValueError("invalid ROI center")
        if not all(isfinite(v) for v in (self.radius, self.background, self.extent_m)):
            raise ValueError("non-finite attention setting")
        if not 0.05 <= self.radius <= 2 or not 0 <= self.background <= 1 or self.extent_m <= 0:
            raise ValueError("invalid attention setting")
        object.__setattr__(self, "center", tuple(max(-1.0, min(1.0, v)) for v in self.center))

    def distance(self, observation: Observation) -> float:
        x, y, _ = observation.position_m
        return hypot((x-self.origin_m[0]) / self.extent_m - self.center[0], (y-self.origin_m[1]) / self.extent_m - self.center[1])

    def contains(self, observation: Observation) -> bool:
        return self.distance(observation) <= self.radius

    def weight(self, observation: Observation) -> float:
        # Flat core and C1-continuous feathering across the drawn ROI boundary.
        t = max(0.0, min(1.0, (self.distance(observation) / self.radius - 0.65) / 0.7))
        return self.background + (1 - self.background) * (1 - t * t * (3 - 2 * t))


def select_sources(observations, attention: Attention, budget: int = 4):
    """Reserve ROI sources before strongest-first selection, plus outside overview.

    With at least two slots, both signs in the ROI receive a slot when present.
    IDs break ties so repeated identical input produces the same selection.
    """
    if budget < 1:
        raise ValueError("source budget must be positive")
    ranked = sorted(observations, key=lambda o: (-o.strength * attention.weight(o), o.source_id))
    inside = [o for o in ranked if attention.contains(o)]
    outside = [o for o in ranked if not attention.contains(o)]
    selected = []
    for sign in (1, -1):
        matching = [o for o in inside if o.sign == sign]
        if matching and len(selected) < budget:
            selected.append(matching[0])
    # Reserve an overview only when audible; it must not crowd out both ROI signs.
    outside_slots = int(bool(outside) and attention.background > 0 and budget >= 3)
    for o in inside:
        if len(selected) >= budget - outside_slots:
            break
        if o not in selected:
            selected.append(o)
    for o in outside + inside:
        if len(selected) >= budget:
            break
        if o not in selected:
            selected.append(o)
    return tuple(selected)
