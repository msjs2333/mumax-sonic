"""Analytical XZ rotations sampled on a fixed physical timeline."""
import math
import numpy as np
from ..fields import FieldFrame

STEP_S = 5e-11
OMEGA_RAD_S = 2*math.pi/8e-9
SCENARIOS = {'activity_rotation': '活动 · 均匀旋转', 'activity_localized': '活动 · 局域旋转'}


def make_activity_frame(scenario, index, *, dt_s=STEP_S, size=49):
    if scenario not in SCENARIOS or type(index) is not int or index < 0:
        raise ValueError('unknown activity scenario or invalid frame index')
    if not math.isfinite(dt_s) or dt_s <= 0 or type(size) is not int or size < 3:
        raise ValueError('invalid sampling interval or mesh')
    t = index*dt_s
    if not math.isfinite(t):
        raise ValueError('non-finite physical time')
    x, y = np.meshgrid(np.linspace(-1e-6, 1e-6, size), np.linspace(-1e-6, 1e-6, size))
    profile = np.ones_like(x) if scenario == 'activity_rotation' else np.exp(-((x-.25e-6)**2+(y+.1e-6)**2)/(2*(.3e-6)**2))
    angle = OMEGA_RAD_S*t*profile
    vectors = np.stack((np.cos(angle), np.zeros_like(angle), np.sin(angle)), axis=-1)
    spacing = 2e-6/(size-1)
    return FieldFrame(vectors, spacing, spacing, t, (-1e-6, -1e-6, 0), sequence=index,
                      segment_id=scenario, provenance=f'analytic:{scenario}; omega_rad_s:{OMEGA_RAD_S}')
