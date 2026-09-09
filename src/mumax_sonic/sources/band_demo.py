"""Analytical transverse oscillations for the frequency-band demo."""

from functools import lru_cache
import math

import numpy as np

from ..fields import FieldFrame


STEP_S = 5e-12
_AMPLITUDE = 0.1
_F_IN_HZ = 10e9
_F_OUT_HZ = 30e9

SCENARIOS = {
    "band_in": "频带内 · 10 GHz",
    "band_out": "频带外 · 30 GHz",
    "band_opposite": "反相空间波 · 10 GHz",
    "band_mixed": "混合频率 · 10/30 GHz",
}


def _valid_inputs(scenario, index, dt_s, size):
    if scenario not in SCENARIOS or type(index) is not int or index < 0:
        raise ValueError("unknown band scenario or invalid frame index")
    if isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)):
        raise ValueError("invalid sampling interval or mesh")
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("invalid sampling interval or mesh")
    if type(size) is not int or size < 3:
        raise ValueError("invalid sampling interval or mesh")


@lru_cache(maxsize=512)
def _make_band_frame(scenario, index, dt_s, size):
    t = index * dt_s
    if not math.isfinite(t):
        raise ValueError("non-finite physical time")

    axis = np.linspace(-1e-6, 1e-6, size)
    x_grid, _ = np.meshgrid(axis, axis)
    phase = 2 * math.pi * (_F_IN_HZ if scenario != "band_out" else _F_OUT_HZ) * t
    x = _AMPLITUDE * np.sin(phase) * np.ones_like(x_grid)

    if scenario == "band_opposite":
        x = x * np.where(x_grid < 0, 1.0, -1.0)
    elif scenario == "band_mixed":
        left = x_grid < 0
        x = np.where(
            left,
            _AMPLITUDE * np.sin(2 * math.pi * _F_IN_HZ * t),
            _AMPLITUDE * np.sin(2 * math.pi * _F_OUT_HZ * t),
        )

    z = np.sqrt(1.0 - x * x)
    vectors = np.stack((x, np.zeros_like(x), z), axis=-1)
    spacing = 2e-6 / (size - 1)
    frequencies = "10GHz" if scenario in {"band_in", "band_opposite"} else (
        "30GHz" if scenario == "band_out" else "10GHz|30GHz"
    )
    phases = "0" if scenario != "band_opposite" else "0|pi"
    return FieldFrame(
        vectors,
        spacing,
        spacing,
        t,
        (-1e-6, -1e-6, 0),
        sequence=index,
        segment_id=scenario,
        time_kind="dynamics",
        source_kind="synthetic",
        provenance=(f"analytic:{scenario}; frequencies_hz:{frequencies}; "
                    f"phase_rad:{phases}; amplitude:{_AMPLITUDE:g}"),
    )


def make_band_frame(scenario, index, *, dt_s=STEP_S, size=17):
    """Return one immutable synthetic frame for a band-observer scenario."""
    _valid_inputs(scenario, index, dt_s, size)
    return _make_band_frame(scenario, index, dt_s, size)
