"""Explicit, bounded two-sublattice derived fields.

This module intentionally accepts two already-sampled :class:`FieldFrame`
objects.  It does not align time series, infer material support, or assign a
physical convention beyond the formulas recorded in the returned provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

from .fields import FieldFrame


@dataclass(frozen=True)
class SublatticePair:
    """Two ordered sublattices and their explicitly derived quantities."""

    a: FieldFrame
    b: FieldFrame
    neel: FieldFrame
    net_A_m: np.ndarray
    net_direction: FieldFrame
    neel_raw: np.ndarray
    valid_mask: np.ndarray
    msat_a_A_m: np.ndarray
    msat_b_A_m: np.ndarray
    pair_id: str


def combine_sublattices(
    a: FieldFrame,
    b: FieldFrame,
    *,
    pair_id: str,
    msat_a_A_m,
    msat_b_A_m,
    direction_epsilon: float = 1e-12,
) -> SublatticePair:
    """Combine an ordered pair of sublattice directions at one exact sample.

    ``neel`` is the unnormalised ``(u_a-u_b)/2`` convention.  ``net_A_m`` is
    ``Ms_a*u_a + Ms_b*u_b``; its dimensionless companion retains the weighted
    magnitude by division by ``Ms_a + Ms_b``.  Values without usable input or
    with a derived magnitude at most ``direction_epsilon`` are represented as
    NaN in directional frames.
    """
    if not isinstance(a, FieldFrame) or not isinstance(b, FieldFrame):
        raise ValueError("a and b must be FieldFrame instances")
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("pair_id must be a non-empty string")
    if a.entity_id == b.entity_id:
        raise ValueError("ordered sublattices must have distinct entity_id values")
    _validate_compatible(a, b)
    epsilon = _positive_finite_scalar(direction_epsilon, "direction_epsilon", allow_zero=True)

    material = a.mask
    ms_a = _msat_array(msat_a_A_m, material, "msat_a_A_m")
    ms_b = _msat_array(msat_b_A_m, material, "msat_b_A_m")
    ua, usable_a = _unit_directions(a.vectors, material)
    ub, usable_b = _unit_directions(b.vectors, material)
    valid = usable_a & usable_b

    # Initialise NaN so invalid data never becomes a numerical zero signal.
    neel_raw = np.full_like(a.vectors, np.nan, dtype=np.float64)
    net = np.full_like(a.vectors, np.nan, dtype=np.float64)
    neel_raw[valid] = (ua[valid] - ub[valid]) / 2.0
    net[valid] = ms_a[valid, None] * ua[valid] + ms_b[valid, None] * ub[valid]

    neel_vectors = neel_raw.copy()
    neel_norm = _vector_norm(neel_raw)
    neel_vectors[valid & (neel_norm <= epsilon)] = np.nan
    net_direction_vectors = np.full_like(a.vectors, np.nan, dtype=np.float64)
    net_direction_vectors[valid] = net[valid] / (ms_a[valid, None] + ms_b[valid, None])
    net_direction_norm = _vector_norm(net_direction_vectors)
    net_direction_vectors[valid & (net_direction_norm <= epsilon)] = np.nan

    entity_base = "sublattice_pair:" + json.dumps(
        [pair_id, a.entity_id, b.entity_id], ensure_ascii=True, separators=(",", ":")
    )
    common = dict(
        dx_m=a.dx_m,
        dy_m=a.dy_m,
        sim_time_s=a.sim_time_s,
        origin_m=a.origin_m,
        mask=material,
        source_kind=a.source_kind,
        segment_id=a.segment_id,
        sequence=a.sequence,
        time_kind=a.time_kind,
    )
    return SublatticePair(
        a=a,
        b=b,
        neel=FieldFrame(
            neel_vectors,
            entity_id=entity_base + ":neel",
            provenance=_provenance(pair_id, a.entity_id, b.entity_id, epsilon, "neel"),
            **common,
        ),
        net_A_m=_readonly(net),
        net_direction=FieldFrame(
            net_direction_vectors,
            entity_id=entity_base + ":net-direction",
            provenance=_provenance(pair_id, a.entity_id, b.entity_id, epsilon, "net_direction"),
            **common,
        ),
        neel_raw=_readonly(neel_raw),
        valid_mask=_readonly(valid),
        msat_a_A_m=_readonly(ms_a),
        msat_b_A_m=_readonly(ms_b),
        pair_id=pair_id,
    )


def _provenance(pair_id: str, entity_a: str, entity_b: str, epsilon: float, quantity: str) -> str:
    return json.dumps(
        {
            "direction_epsilon": epsilon,
            "formulas": {
                "neel": "equal_weight_neel_direction_difference: (u_a-u_b)/2",
                "net_A_m": "Ms_a*u_a+Ms_b*u_b",
                "net_direction": "net_A_m/(Ms_a+Ms_b)",
            },
            "ordered_entity_ids": [entity_a, entity_b],
            "pair_id": pair_id,
            "quantity": quantity,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_compatible(a: FieldFrame, b: FieldFrame) -> None:
    """Reject any identity or geometry mismatch before deriving a pair."""
    if a.segment_id != b.segment_id or a.sequence != b.sequence:
        raise ValueError("sublattice frames must have the same segment_id and sequence")
    if a.sim_time_s != b.sim_time_s or a.time_kind != b.time_kind or a.source_kind != b.source_kind:
        raise ValueError("sublattice frames must have exact matching time, time_kind, and source_kind")
    if (a.vectors.shape != b.vectors.shape or a.dx_m != b.dx_m or a.dy_m != b.dy_m
            or a.origin_m != b.origin_m):
        raise ValueError("sublattice frames must have matching shape, spacing, and origin")
    if not np.array_equal(a.mask, b.mask):
        raise ValueError("sublattice frames must have identical material masks")


def _positive_finite_scalar(value, name: str, *, allow_zero: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite {'non-negative' if allow_zero else 'positive'} scalar") from exc
    if not np.isfinite(result) or result < 0.0 or (not allow_zero and result == 0.0):
        raise ValueError(f"{name} must be a finite {'non-negative' if allow_zero else 'positive'} scalar")
    return result


def _msat_array(value, material: np.ndarray, name: str) -> np.ndarray:
    """Copy a declared scalar or per-sample saturation magnetisation."""
    supplied = np.asarray(value)
    if supplied.ndim == 0:
        scalar = _positive_finite_scalar(value, name)
        result = np.full(material.shape, scalar, dtype=np.float64)
    elif supplied.shape == material.shape:
        try:
            result = np.array(supplied, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a scalar or numeric array with shape (ny, nx)") from exc
        if np.any(~np.isfinite(result[material])) or np.any(result[material] <= 0.0):
            raise ValueError(f"{name} must be finite and strictly positive at every material sample")
    else:
        raise ValueError(f"{name} must be a scalar or array with shape (ny, nx)")
    return result


def _unit_directions(vectors: np.ndarray, material: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.all(np.isfinite(vectors), axis=-1)
    safe = np.where(finite[..., None], vectors, 0.0)
    scale = np.max(np.abs(safe), axis=-1)
    scaled = np.divide(safe, scale[..., None], out=np.zeros_like(safe), where=scale[..., None] > 0.0)
    scaled_norm = np.linalg.norm(scaled, axis=-1)
    usable = material & finite & (scale > 0.0) & np.isfinite(scaled_norm) & (scaled_norm > 0.0)
    unit = np.full_like(vectors, np.nan, dtype=np.float64)
    unit[usable] = scaled[usable] / scaled_norm[usable, None]
    return unit, usable


def _vector_norm(vectors: np.ndarray) -> np.ndarray:
    finite = np.all(np.isfinite(vectors), axis=-1)
    safe = np.where(finite[..., None], vectors, 0.0)
    scale = np.max(np.abs(safe), axis=-1)
    scaled = np.divide(safe, scale[..., None], out=np.zeros_like(safe), where=scale[..., None] > 0.0)
    return scale * np.linalg.norm(scaled, axis=-1)


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result
