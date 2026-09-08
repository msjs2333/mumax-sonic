"""Testable transport: pause freezes simulation time, not wall-clock freshness."""
from dataclasses import dataclass
from .model import Sample


@dataclass
class Transport:
    sim_time_s: float = 0.0
    playing: bool = False
    static_listen: bool = True
    seconds_per_second: float = 1e-9  # illustrative synthetic time scaling

    def advance(self, wall_dt_s: float):
        if wall_dt_s < 0:
            raise ValueError("clock moved backwards")
        if self.playing:
            self.sim_time_s += wall_dt_s * self.seconds_per_second

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
