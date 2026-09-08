"""Bounded two-dimensional skyrmion-charge observers.

The arrays returned here contain *charge contributions*, rather than a density:
their sums are dimensionless topological charges.  Invalid output locations are
represented by ``NaN`` and must be interpreted together with ``valid``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np


_BRANCH_TOLERANCE = 1.0e-10
_STRONG_GRADIENT_RAD = pi / 2.0


@dataclass(frozen=True)
class TopologyResult:
    """Signed charge split into non-negative spatial contribution channels."""

    positive: np.ndarray
    negative: np.ndarray
    valid: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    coverage: float
    warnings: tuple[str, ...]
    method: str

    def _total(self, array: np.ndarray) -> float:
        if not np.any(self.valid):
            return float("nan")
        return float(np.sum(array[self.valid]))

    @property
    def q_pos(self) -> float:
        return self._total(self.positive)

    @property
    def q_neg(self) -> float:
        return self._total(self.negative)

    @property
    def q_net(self) -> float:
        if not np.any(self.valid):
            return float("nan")
        return self.q_pos - self.q_neg

    @property
    def q_abs(self) -> float:
        if not np.any(self.valid):
            return float("nan")
        return self.q_pos + self.q_neg


def topology(
    u: np.ndarray,
    dx_m: float,
    dy_m: float,
    *,
    mask: np.ndarray | None = None,
    boundary: str = "open",
    method: str = "solid_angle",
    orientation: int = 1,
) -> TopologyResult:
    """Estimate oriented skyrmion charge on a sampled two-dimensional field.

    ``u`` is indexed ``(y, x, component)``.  The result has the same ``(y, x)``
    shape.  Solid-angle contributions occupy the array slot of each plaquette's
    lower-left anchor but carry the plaquette-centre coordinates; finite-
    difference contributions occupy their central sample.  Thus open boundaries
    deliberately have unsupported output locations.
    """
    field = _validate_field(u)
    dx = _validate_spacing(dx_m, "dx_m")
    dy = _validate_spacing(dy_m, "dy_m")
    if boundary not in {"open", "periodic"}:
        raise ValueError("boundary must be 'open' or 'periodic'")
    if method not in {"solid_angle", "finite_difference"}:
        raise ValueError("method must be 'solid_angle' or 'finite_difference'")
    if orientation not in {-1, 1}:
        raise ValueError("orientation must be -1 or +1")

    ny, nx, _ = field.shape
    material = _validate_mask(mask, (ny, nx))
    finite = np.all(np.isfinite(field), axis=-1)
    # Scaling first keeps a finite, very large input vector normalizable.
    safe_field = np.where(finite[..., None], field, 0.0)
    scale = np.max(np.abs(safe_field), axis=-1)
    scaled = np.divide(safe_field, scale[..., None], out=np.zeros_like(safe_field), where=scale[..., None] > 0.0)
    scaled_norms = np.linalg.norm(scaled, axis=-1)
    usable = material & finite & (scale > 0.0)
    normalized = np.full_like(field, np.nan, dtype=float)
    normalized[usable] = scaled[usable] / scaled_norms[usable, None]

    if method == "solid_angle":
        positive, negative, valid, candidate_count, ambiguous, strong = _solid_angle(
            normalized, usable, boundary, orientation
        )
    else:
        positive, negative, valid, candidate_count, strong = _finite_difference(
            normalized, usable, dx, dy, boundary, orientation
        )
        ambiguous = False

    warnings: list[str] = []
    if np.any(material & ~usable):
        warnings.append("non-finite or zero-magnitude material sites excluded")
    if np.any(~material):
        warnings.append("masked sites excluded; stencils do not bridge holes")
    if ambiguous:
        warnings.append("antipodal or branch-ambiguous triangles excluded")
    if strong:
        warnings.append("strong spin gradients may be under-resolved")
    warnings.append("finite-difference and solid-angle methods use distinct spatial supports")
    coverage = 0.0 if candidate_count == 0 else float(np.count_nonzero(valid) / candidate_count)
    if coverage < 1.0:
        warnings.append("topology support is incomplete")
    if not np.any(valid):
        warnings.append("no valid topology support")

    coordinate_offset_x = dx / 2.0 if method == "solid_angle" else 0.0
    coordinate_offset_y = dy / 2.0 if method == "solid_angle" else 0.0
    # Plaquette anchors record their physical centres.  For a periodic final
    # plaquette this is deliberately unwrapped beyond the last sample, so its
    # position remains the centre of the physical seam rather than jumping to 0.
    x_values = np.arange(nx, dtype=float) * dx + coordinate_offset_x
    y_values = np.arange(ny, dtype=float) * dy + coordinate_offset_y
    x_m, y_m = np.meshgrid(x_values, y_values, indexing="xy")
    return TopologyResult(
        positive=positive,
        negative=negative,
        valid=valid,
        x_m=x_m,
        y_m=y_m,
        coverage=coverage,
        warnings=tuple(warnings),
        method=method,
    )


def _validate_field(u: np.ndarray) -> np.ndarray:
    try:
        field = np.asarray(u, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("u must be a numeric array with shape (ny, nx, 3)") from exc
    if field.ndim != 3 or field.shape[-1] != 3 or field.shape[0] < 3 or field.shape[1] < 3:
        raise ValueError("u must have shape (ny, nx, 3) with ny and nx at least 3")
    return field


def _validate_spacing(value: float, name: str) -> float:
    try:
        spacing = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return spacing


def _validate_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    supplied = np.asarray(mask)
    if supplied.shape != shape or supplied.dtype != bool:
        raise ValueError("mask must be a boolean array with shape u.shape[:2]")
    return supplied


def _empty(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.full(shape, np.nan), np.full(shape, np.nan), np.zeros(shape, dtype=bool)


def _solid_angle(
    u: np.ndarray, usable: np.ndarray, boundary: str, orientation: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool, bool]:
    ny, nx, _ = u.shape
    positive, negative, valid = _empty((ny, nx))
    if boundary == "periodic":
        a = u
        b = np.roll(u, -1, axis=1)
        c = np.roll(b, -1, axis=0)
        d = np.roll(u, -1, axis=0)
        support = usable & np.roll(usable, -1, axis=1) & np.roll(usable, (-1, -1), axis=(0, 1)) & np.roll(usable, -1, axis=0)
        output_slice = np.s_[:, :]
    else:
        a, b, c, d = u[:-1, :-1], u[:-1, 1:], u[1:, 1:], u[1:, :-1]
        support = usable[:-1, :-1] & usable[:-1, 1:] & usable[1:, 1:] & usable[1:, :-1]
        output_slice = np.s_[:-1, :-1]
    first, first_ambiguous = _triangle_charge_array(a, b, c)
    second, second_ambiguous = _triangle_charge_array(a, c, d)
    local_valid = support & ~first_ambiguous & ~second_ambiguous
    first *= orientation
    second *= orientation
    # Split each triangle before the plaquette sum so opposite sub-triangle
    # contributions are retained as independent non-negative channels.
    local_positive = np.maximum(first, 0.0) + np.maximum(second, 0.0)
    local_negative = np.maximum(-first, 0.0) + np.maximum(-second, 0.0)
    positive[output_slice] = np.where(local_valid, local_positive, np.nan)
    negative[output_slice] = np.where(local_valid, local_negative, np.nan)
    valid[output_slice] = local_valid
    edge_dots = np.stack((
        np.sum(a * b, axis=-1), np.sum(b * c, axis=-1),
        np.sum(c * d, axis=-1), np.sum(d * a, axis=-1),
    ))
    strong = bool(np.any(local_valid & (np.min(edge_dots, axis=0) <= np.cos(_STRONG_GRADIENT_RAD))))
    return positive, negative, valid, int(support.size), bool(np.any(support & (first_ambiguous | second_ambiguous))), strong


def _triangle_charge_array(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ab = np.sum(a * b, axis=-1)
    bc = np.sum(b * c, axis=-1)
    ca = np.sum(c * a, axis=-1)
    numerator = np.sum(a * np.cross(b, c), axis=-1)
    denominator = 1.0 + ab + bc + ca
    ambiguous = (
        (np.minimum(np.minimum(ab, bc), ca) <= -1.0 + _BRANCH_TOLERANCE)
        | (np.hypot(numerator, denominator) <= _BRANCH_TOLERANCE)
        # atan2(±0, negative) changes from +pi to -pi with numerical noise.
        | ((np.abs(numerator) <= _BRANCH_TOLERANCE) & (denominator < 0.0))
    )
    return 2.0 * np.arctan2(numerator, denominator) / (4.0 * pi), ambiguous


def _finite_difference(
    u: np.ndarray, usable: np.ndarray, dx: float, dy: float, boundary: str, orientation: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    ny, nx, _ = u.shape
    positive, negative, valid = _empty((ny, nx))
    if boundary == "periodic":
        center = u
        left, right = np.roll(u, 1, axis=1), np.roll(u, -1, axis=1)
        above, below = np.roll(u, 1, axis=0), np.roll(u, -1, axis=0)
        support = usable & np.roll(usable, 1, axis=1) & np.roll(usable, -1, axis=1) & np.roll(usable, 1, axis=0) & np.roll(usable, -1, axis=0)
        output_slice = np.s_[:, :]
    else:
        center = u[1:-1, 1:-1]
        left, right = u[1:-1, :-2], u[1:-1, 2:]
        above, below = u[:-2, 1:-1], u[2:, 1:-1]
        support = usable[1:-1, 1:-1] & usable[1:-1, :-2] & usable[1:-1, 2:] & usable[:-2, 1:-1] & usable[2:, 1:-1]
        output_slice = np.s_[1:-1, 1:-1]
    du_dx = (right - left) / (2.0 * dx)
    du_dy = (below - above) / (2.0 * dy)
    charge = orientation * np.sum(center * np.cross(du_dx, du_dy), axis=-1) * dx * dy / (4.0 * pi)
    positive[output_slice] = np.where(support, np.maximum(charge, 0.0), np.nan)
    negative[output_slice] = np.where(support, np.maximum(-charge, 0.0), np.nan)
    valid[output_slice] = support
    neighbor_dots = np.stack((
        np.sum(center * left, axis=-1), np.sum(center * right, axis=-1),
        np.sum(center * above, axis=-1), np.sum(center * below, axis=-1),
    ))
    strong = bool(np.any(support & (np.min(neighbor_dots, axis=0) <= np.cos(_STRONG_GRADIENT_RAD))))
    return positive, negative, valid, int(support.size), strong
