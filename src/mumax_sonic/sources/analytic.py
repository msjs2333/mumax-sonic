"""Small deterministic analytical magnetic fields for observer validation.

These fields are CPU-generated fixtures.  They are useful for numerical
contracts and examples, and do not represent a GPU simulation or a material
specific model.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np


SCENARIOS: dict[str, str] = {
    "uniform": "均匀磁化",
    "skyrmion": "单个紧支撑斯格明子",
    "opposite_pair": "相反拓扑双斑",
    "wall_inplane": "面内畴壁连续内角",
    "wall_pma": "PMA 畴壁连续内角",
}

_EXTENT_M = 1.0e-6
_PERIOD_S = 8.0e-9


def _smooth_theta(radius: np.ndarray, core_radius: float) -> np.ndarray:
    """Compact radial profile with zero first derivative at its edge."""

    s = np.clip(radius / core_radius, 0.0, 1.0)
    return math.pi * (1.0 - 3.0 * s * s + 2.0 * s * s * s)


def _compact_texture(
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    radius: float,
    winding: int,
) -> np.ndarray:
    """Return one compact core texture on the supplied coordinate mesh."""

    local_x = x - center_x
    local_y = y
    radial = np.hypot(local_x, local_y)
    inside = radial < radius
    theta = _smooth_theta(radial, radius)
    phase = winding * np.arctan2(local_y, local_x)
    transverse = np.sin(theta)
    field = np.empty(x.shape + (3,), dtype=float)
    field[..., 0] = transverse * np.cos(phase)
    field[..., 1] = transverse * np.sin(phase)
    field[..., 2] = np.cos(theta)
    field[~inside] = (0.0, 0.0, 1.0)
    return field


def _wall_field(
    x: np.ndarray,
    phase: float,
    d: tuple[float, float, float],
    e1: tuple[float, float, float],
    e2: tuple[float, float, float],
) -> np.ndarray:
    coordinate = x / 0.15e-6
    longitudinal = np.tanh(coordinate)
    transverse = 1.0 / np.cosh(coordinate)
    direction = (
        longitudinal[..., np.newaxis] * np.asarray(d)
        + transverse[..., np.newaxis]
        * (
            math.cos(phase) * np.asarray(e1)
            + math.sin(phase) * np.asarray(e2)
        )
    )
    # tanh^2 + sech^2 = 1 analytically, but normalize to remove the tiny
    # floating point residual before exposing a fixture to tests/callers.
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    return direction


def make_field(scenario: str, sim_time_s: float = 0.0, size: int = 65) -> np.ndarray:
    """Create a reproducible ``(ny, nx, 3)`` unit-vector field.

    Coordinates are square, ordered as ``(y, x)`` in the returned array, and
    span -1e-6 to +1e-6 metres on both axes.  Wall phase is continuous in
    simulation time with an 8 ns period.
    """

    if scenario not in SCENARIOS:
        raise ValueError(f"unknown analytic scenario: {scenario!r}")
    if not isinstance(sim_time_s, Real) or isinstance(sim_time_s, bool) or not math.isfinite(float(sim_time_s)):
        raise ValueError("sim_time_s must be a finite real number")
    if not isinstance(size, Integral) or isinstance(size, bool) or int(size) < 2:
        raise ValueError("size must be an integer of at least 2")
    size = int(size)
    axis = np.linspace(-_EXTENT_M, _EXTENT_M, size, dtype=float)
    x, y = np.meshgrid(axis, axis, indexing="xy")

    if scenario == "uniform":
        field = np.zeros((size, size, 3), dtype=float)
        field[..., 2] = 1.0
    elif scenario == "skyrmion":
        field = _compact_texture(x, y, 0.0, 0.45e-6, +1)
    elif scenario == "opposite_pair":
        left = _compact_texture(x, y, -0.48e-6, 0.38e-6, +1)
        right = _compact_texture(x, y, +0.48e-6, 0.38e-6, -1)
        left_inside = np.hypot(x + 0.48e-6, y) < 0.38e-6
        right_inside = np.hypot(x - 0.48e-6, y) < 0.38e-6
        field = np.where(left_inside[..., np.newaxis], left, (0.0, 0.0, 1.0))
        field = np.where(right_inside[..., np.newaxis], right, field)
    elif scenario == "wall_inplane":
        phase = 2.0 * math.pi * float(sim_time_s) / _PERIOD_S
        field = _wall_field(x, phase, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    else:  # wall_pma
        phase = 2.0 * math.pi * float(sim_time_s) / _PERIOD_S
        field = _wall_field(x, phase, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

    if not np.all(np.isfinite(field)):
        raise RuntimeError("analytic fixture generated non-finite values")
    norms = np.linalg.norm(field, axis=-1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-15):
        raise RuntimeError("analytic fixture generated a non-unit field")
    return field
