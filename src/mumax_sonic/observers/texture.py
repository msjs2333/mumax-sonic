"""Continuous direction observables for explicitly chosen axes.

The routines in this module do not detect walls or classify textures.  They
take the physical direction and reference axes supplied by the caller and
return a continuous in-plane angle around that direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_VECTOR_TOL = 0.0
_AXIS_TOL = 1e-12


@dataclass(frozen=True)
class DirectionResult:
    """Result of projecting a vector field into a directed transverse plane.

    ``angle_rad`` is in ``[-pi, pi]`` and is NaN where the direction is not
    defined.  ``projection`` contains the normalized input direction after
    removing its component along ``d``; invalid entries are NaN.  The axes
    are tuples so callers can safely retain the exact convention used for a
    measurement.
    """

    angle_rad: np.ndarray
    projection: np.ndarray
    valid: np.ndarray
    e1: tuple[float, float, float]
    e2: tuple[float, float, float]
    d: tuple[float, float, float]

    @property
    def cos_angle(self) -> np.ndarray:
        """Circular cosine cache for interpolation and audio mapping."""

        return np.cos(self.angle_rad)

    @property
    def sin_angle(self) -> np.ndarray:
        """Circular sine cache for interpolation and audio mapping."""

        return np.sin(self.angle_rad)


def _axis(value: Any, name: str) -> np.ndarray:
    """Validate and normalize one three-component physical axis."""

    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a finite three-component vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= _VECTOR_TOL:
        raise ValueError(f"{name} must be nonzero")
    return array / norm


def direction_angle(
    u: np.ndarray,
    domain_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    reference_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    *,
    mask: np.ndarray | None = None,
    projection_floor: float = 1e-6,
) -> DirectionResult:
    """Measure the directed transverse angle of a vector field.

    ``u`` has shape ``(..., 3)`` (normally ``(ny, nx, 3)``).  Every finite,
    nonzero input direction is normalized before projection.  The domain axis
    is normalized to ``d``; the reference axis is projected into the plane
    perpendicular to ``d`` and rejected if that projection is degenerate.
    ``e2 = cross(d, e1)`` gives the right-handed basis.  A vector is valid
    only when it is finite, nonzero, unmasked, and its transverse projection
    has norm strictly above ``projection_floor``.
    """

    values = np.asarray(u, dtype=float)
    if values.ndim < 2 or values.shape[-1] != 3:
        raise ValueError("u must have shape (..., 3)")
    if not np.isfinite(projection_floor) or projection_floor < 0:
        raise ValueError("projection_floor must be a finite nonnegative number")

    d_array = _axis(domain_axis, "domain_axis")
    reference = _axis(reference_axis, "reference_axis")
    reference_perp = reference - float(np.dot(reference, d_array)) * d_array
    reference_norm = float(np.linalg.norm(reference_perp))
    if not np.isfinite(reference_norm) or reference_norm <= _AXIS_TOL:
        raise ValueError("reference_axis must not be parallel to domain_axis")
    e1_array = reference_perp / reference_norm
    e2_array = np.cross(d_array, e1_array)
    e2_norm = float(np.linalg.norm(e2_array))
    if not np.isfinite(e2_norm) or e2_norm <= _VECTOR_TOL:
        raise ValueError("could not construct a transverse basis")
    e2_array /= e2_norm

    leading_shape = values.shape[:-1]
    if mask is None:
        mask_array = np.ones(leading_shape, dtype=bool)
    else:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != leading_shape:
            raise ValueError("mask must match the leading dimensions of u")

    norms = np.linalg.norm(values, axis=-1)
    finite = np.all(np.isfinite(values), axis=-1) & np.isfinite(norms)
    nonzero = norms > _VECTOR_TOL
    safe_norms = np.where(finite & nonzero, norms, 1.0)
    normalized = values / safe_norms[..., np.newaxis]
    parallel = np.sum(normalized * d_array, axis=-1)
    projection = normalized - parallel[..., np.newaxis] * d_array
    projection_norm = np.linalg.norm(projection, axis=-1)
    valid = (
        mask_array
        & finite
        & nonzero
        & np.isfinite(projection_norm)
        & (projection_norm > float(projection_floor))
    )

    angle = np.full(leading_shape, np.nan, dtype=float)
    angle[valid] = np.arctan2(
        np.sum(projection[valid] * e2_array, axis=-1),
        np.sum(projection[valid] * e1_array, axis=-1),
    )
    projection_out = np.array(projection, dtype=float, copy=True)
    projection_out[~valid] = np.nan

    return DirectionResult(
        angle_rad=angle,
        projection=projection_out,
        valid=valid.copy(),
        e1=tuple(float(x) for x in e1_array),
        e2=tuple(float(x) for x in e2_array),
        d=tuple(float(x) for x in d_array),
    )


def wall_angle(
    core: np.ndarray,
    domain_a: tuple[float, float, float],
    domain_b: tuple[float, float, float],
    reference_axis: tuple[float, float, float],
) -> DirectionResult:
    """Measure a wall-core angle for an explicitly ordered domain pair.

    The domains must be near antiparallel.  Their order is retained through
    ``d = normalize(domain_b - domain_a)``; swapping them therefore changes
    the reported convention as expected.
    """

    a = _axis(domain_a, "domain_a")
    b = _axis(domain_b, "domain_b")
    if float(np.dot(a, b)) >= -0.95:
        raise ValueError("domain_a and domain_b must be ordered antiparallel domains")
    d = b - a
    d_norm = float(np.linalg.norm(d))
    if not np.isfinite(d_norm) or d_norm <= _VECTOR_TOL:
        raise ValueError("domain pair has a degenerate difference")
    return direction_angle(core, tuple(float(x) for x in d), reference_axis)
