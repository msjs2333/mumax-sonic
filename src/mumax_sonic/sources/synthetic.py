"""Small, deterministic labelled scenes used by the P1 demo.

These are synthetic labels for exercising the audio and attention pipeline.  They
do not calculate magnetic charge, topological charge, or any other physical
observable from a field.
"""

from __future__ import annotations

from math import isfinite, pi, sin

from mumax_sonic.model import Observation, Sample

SCENARIOS: dict[str, str] = {
    "moving": "单源左右与上下移动",
    "signed_pair": "静态异位正负等强",
    "colocated": "静态同位正负对消标签",
    "roi_challenge": "弱内源与强外源",
    "orientation": "连续取向角标签",
    "stale": "过期数据",
    "invalid": "无效数据",
}

_PERIOD_S = 8.0e-9  # 8-second demo loop at 1 real second = 1 ns simulation
_EXTENT_M = 1.0e-6


def _phase(sim_time_s: float, seed: int) -> float:
    """Return a deterministic phase; no wall-clock or global random state."""
    return 2.0 * pi * ((sim_time_s % _PERIOD_S) / _PERIOD_S) + (seed % 17) * 0.07


def _continuous_phase(sim_time_s: float, seed: int) -> float:
    return 2.0 * pi * (sim_time_s / _PERIOD_S) + (seed % 17) * 0.07


def _obs(source_id: str, x: float, y: float, strength: float, sign: int,
         orientation_rad: float = 0.0) -> Observation:
    # Clamp only generated values at the contract boundary; all values here are
    # intentionally within the physical demo extent.
    x = max(-_EXTENT_M, min(_EXTENT_M, x))
    y = max(-_EXTENT_M, min(_EXTENT_M, y))
    return Observation(source_id, (x, y, 0.0), max(0.0, min(1.0, strength)), sign,
                       orientation_rad=orientation_rad)


def make_sample(scenario: str, sim_time_s: float, sequence: int = 0,
                seed: int = 7) -> Sample:
    """Create a reproducible labelled :class:`Sample` for ``scenario``.

    ``sim_time_s`` is simulation time.  In the moving scene it is interpreted
    modulo one nanosecond solely to make the intended one-second demo loop
    explicit.  Every scene has 3--6 stable source IDs and strengths in [0, 1].
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown synthetic scenario: {scenario!r}")
    if not isinstance(sim_time_s, (int, float)) or not isfinite(sim_time_s) or sim_time_s < 0:
        raise ValueError("sim_time_s must be a nonnegative finite number")
    if not isinstance(sequence, int) or sequence < 0:
        raise ValueError("sequence must be a nonnegative integer")
    phase = _phase(float(sim_time_s), int(seed))
    validity = "valid"
    time_kind = "static"

    if scenario == "moving":
        # The first chain is deliberately a single moving source.
        x = 0.82e-6 * sin(phase)
        y = 0.62e-6 * sin(phase + pi / 2.0)
        observations = (
            _obs("moving-primary", x, y, 0.85, 1),
        )
        time_kind = "dynamics"
    elif scenario == "signed_pair":
        observations = (
            _obs("pair-positive", -0.45e-6, 0.0, 0.70, 1),
            _obs("pair-negative", 0.45e-6, 0.0, 0.70, -1),
        )
    elif scenario == "colocated":
        observations = (
            _obs("co-positive", 0.0, 0.0, 0.65, 1),
            _obs("co-negative", 0.0, 0.0, 0.65, -1),
        )
    elif scenario == "roi_challenge":
        observations = (
            _obs("roi-inner-weak", 0.0, 0.0, 0.12, 1),
            _obs("roi-outer-west", -0.90e-6, 0.0, 0.95, -1),
            _obs("roi-outer-east", 0.90e-6, 0.0, 0.90, 1),
            _obs("roi-outer-north", 0.0, 0.90e-6, 0.80, -1),
            _obs("roi-outer-south", 0.0, -0.90e-6, 0.75, 1),
            _obs("roi-outer-diagonal", 0.85e-6, 0.85e-6, 0.70, -1),
        )
    elif scenario == "orientation":
        phase = _continuous_phase(float(sim_time_s), int(seed))
        observations = tuple(
            _obs(f"orientation-{i}", x, y, 0.55, 1 if i % 2 == 0 else -1,
                 orientation_rad=phase / 3.0 + i * (pi / 4.0))
            for i, (x, y) in enumerate(((-0.75e-6, 0.0), (-0.25e-6, 0.0),
                                         (0.25e-6, 0.0), (0.75e-6, 0.0)))
        )
        time_kind = "dynamics"
    elif scenario == "stale":
        observations = (
            _obs("stale-positive", -0.30e-6, 0.0, 0.60, 1),
            _obs("stale-negative", 0.30e-6, 0.0, 0.60, -1),
            _obs("stale-reference", 0.0, 0.60e-6, 0.20, 1),
        )
        validity = "stale"
    else:  # invalid
        observations = (
            _obs("invalid-positive", -0.30e-6, 0.0, 0.60, 1),
            _obs("invalid-negative", 0.30e-6, 0.0, 0.60, -1),
            _obs("invalid-reference", 0.0, 0.60e-6, 0.20, 1),
        )
        validity = "invalid"

    return Sample(float(sim_time_s), observations, sequence=sequence,
                  segment_id=scenario, validity=validity, coverage=1.0,
                  source_kind="synthetic", time_kind=time_kind)
