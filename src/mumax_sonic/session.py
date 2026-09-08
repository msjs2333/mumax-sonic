"""Testable transport: pause freezes simulation time, not wall-clock freshness."""
from dataclasses import dataclass
from math import isfinite
from .model import Sample


@dataclass
class Transport:
    sim_time_s: float = 0.0
    playing: bool = False
    static_listen: bool = True
    seconds_per_second: float = 1e-9  # illustrative synthetic time scaling
    playback_rate: float = 1.0

    def seek(self, sim_time_s: float):
        if not isfinite(sim_time_s) or sim_time_s < 0:
            raise ValueError('seek requires a finite nonnegative physical time')
        self.sim_time_s = sim_time_s

    def set_speed(self, multiplier: float):
        if not isfinite(multiplier) or multiplier <= 0:
            raise ValueError('playback multiplier must be finite and positive')
        self.playback_rate = multiplier

    def advance(self, wall_dt_s: float):
        if not isfinite(wall_dt_s) or wall_dt_s < 0:
            raise ValueError("clock moved backwards")
        if not all(isfinite(v) and v > 0 for v in (self.seconds_per_second, self.playback_rate)):
            raise ValueError('invalid playback clock scale')
        if self.playing:
            next_time = self.sim_time_s + wall_dt_s * self.seconds_per_second * self.playback_rate
            self.seek(next_time)

    @property
    def audible(self):
        return self.playing or self.static_listen


class Timeline:
    """Small temporal guard for a future source: no silent reordering/gap fill."""
    def __init__(self):
        self.previous: Sample | None = None

    def accept(self, sample: Sample) -> str:
        previous = self.previous
        self.previous = sample
        if sample.validity != "valid":
            return sample.validity
        if sample.coverage < 1:
            return "invalid"
        if previous is None:
            return "valid"
        if previous.segment_id != sample.segment_id:
            return "warming_up"
        if sample.sim_time_s < previous.sim_time_s or sample.sequence <= previous.sequence:
            return "invalid"
        if sample.sequence != previous.sequence + 1:
            return "warming_up"
        return "valid"
