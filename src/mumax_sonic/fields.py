"""Regular XY sections with XYZ vector components and explicit provenance."""
from dataclasses import dataclass
from math import isfinite
import numpy as np


@dataclass(frozen=True)
class FieldFrame:
    vectors: np.ndarray
    dx_m: float
    dy_m: float
    sim_time_s: float = 0.0
    origin_m: tuple = (0.0, 0.0, 0.0)  # first sample centre, not cell edge
    mask: np.ndarray | None = None
    entity_id: str = "m"
    source_kind: str = "synthetic"
    segment_id: str = "field"
    sequence: int = 0
    time_kind: str = "dynamics"
    provenance: str = ""

    def __post_init__(self):
        a = np.array(self.vectors, dtype=np.float64, copy=True)
        if a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 3:
            raise ValueError("vectors must have shape (ny>=3,nx>=3,3), components XYZ")
        if not all(isfinite(v) and v > 0 for v in (self.dx_m, self.dy_m)):
            raise ValueError("positive finite SI cell spacing required")
        if not isfinite(self.sim_time_s) or self.sim_time_s < 0:
            raise ValueError("invalid physical time")
        origin = tuple(self.origin_m)
        if len(origin) != 3 or not all(isfinite(v) for v in origin):
            raise ValueError("three finite origin coordinates required")
        if self.source_kind not in {'synthetic', 'replay', 'live'} or self.time_kind not in {'static', 'dynamics', 'relaxation'}:
            raise ValueError("unknown field source/time kind")
        if type(self.sequence) is not int or self.sequence < 0 or not self.entity_id or not self.segment_id:
            raise ValueError("invalid field identity")
        if self.mask is not None and np.asarray(self.mask).dtype != np.bool_:
            raise ValueError('mask must be boolean; missing material cannot be inferred from numeric labels')
        mask = np.ones(a.shape[:2], dtype=bool) if self.mask is None else np.array(self.mask, dtype=bool, copy=True)
        if mask.shape != a.shape[:2]:
            raise ValueError("mask must match spatial shape")
        # Preserve raw magnitudes and invalid samples for observer quality checks.
        a.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, 'vectors', a)
        object.__setattr__(self, 'mask', mask)
        object.__setattr__(self, 'origin_m', origin)

    @property
    def extent_m(self):
        ny, nx = self.vectors.shape[:2]
        return max((nx-1)*self.dx_m, (ny-1)*self.dy_m)/2

    @property
    def center_m(self):
        ny, nx = self.vectors.shape[:2]
        return (self.origin_m[0]+(nx-1)*self.dx_m/2,
                self.origin_m[1]+(ny-1)*self.dy_m/2, self.origin_m[2])
