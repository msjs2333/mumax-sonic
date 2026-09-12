"""Adaptive, sign-preserving aggregation of spatial contribution grids.

The grid remains a physical contribution field.  Attention only determines how
many spatial representatives are useful and the listening gain metadata carried
by each representative.
"""
from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np

from .attention import Attention
from .model import MAX_SOURCE_BUDGET, Observation


@dataclass(frozen=True)
class ContributionGrid:
    """Non-negative positive and negative contributions at physical XY sites."""

    x_m: np.ndarray
    y_m: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    z_m: float
    entity_id: str
    quantity: str
    unit: str

    def __post_init__(self):
        arrays = (self.x_m, self.y_m, self.positive, self.negative)
        if not all(isinstance(value, np.ndarray) for value in arrays):
            raise ValueError("grid coordinates and contributions must be ndarrays")
        if any(value.ndim != 2 for value in arrays):
            raise ValueError("grid coordinates and contributions must be two-dimensional")
        if len({value.shape for value in arrays}) != 1:
            raise ValueError("grid arrays must have the same shape")
        if not all(np.issubdtype(value.dtype, np.number) and not np.issubdtype(value.dtype, np.complexfloating)
                   for value in arrays):
            raise ValueError("grid arrays must be numeric")
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("grid arrays must be finite")
        if (self.positive < 0).any() or (self.negative < 0).any():
            raise ValueError("grid contributions must be nonnegative")
        if not isfinite(self.z_m):
            raise ValueError("grid z coordinate must be finite")
        if not all(isinstance(value, str) and value for value in (self.entity_id, self.quantity, self.unit)):
            raise ValueError("grid entity, quantity, and unit are required")
        # Frozen dataclasses do not make ndarray payloads immutable.  Retain an
        # isolated physical cache so callers cannot change a grid during use.
        for name, value in zip(("x_m", "y_m", "positive", "negative"), arrays):
            retained = np.array(value, copy=True)
            retained.setflags(write=False)
            object.__setattr__(self, name, retained)


@dataclass(frozen=True)
class AggregationResult:
    observations: tuple[Observation, ...]
    diagnostic: dict


@dataclass(frozen=True)
class _Node:
    path: str
    sites: np.ndarray
    masses: np.ndarray
    importance: np.ndarray


def _roi_components(x_m: np.ndarray, y_m: np.ndarray, attention: Attention) -> tuple[np.ndarray, np.ndarray]:
    """Return feather-only foreground fraction and effective point gain."""
    distance = np.hypot(
        (x_m - attention.origin_m[0]) / attention.extent_m - attention.center[0],
        (y_m - attention.origin_m[1]) / attention.extent_m - attention.center[1],
    )
    t = np.clip((distance / attention.radius - 0.65) / 0.7, 0.0, 1.0)
    foreground = 1.0 - t * t * (3.0 - 2.0 * t)
    point_weight = attention.background + (1.0 - attention.background) * foreground
    return foreground, point_weight


def _centroid(node: _Node, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    total = float(node.masses.sum())
    return (
        float(np.dot(node.masses, x[node.sites]) / total),
        float(np.dot(node.masses, y[node.sites]) / total),
    )


def _weighted_sse(node: _Node, x: np.ndarray, y: np.ndarray) -> float:
    total = float(node.importance.sum())
    if total <= 0:
        return 0.0
    xs, ys = x[node.sites], y[node.sites]
    cx = float(np.dot(node.importance, xs) / total)
    cy = float(np.dot(node.importance, ys) / total)
    return float(np.dot(node.importance, (xs - cx) ** 2 + (ys - cy) ** 2))


def _split(node: _Node, x: np.ndarray, y: np.ndarray) -> tuple[_Node, _Node] | None:
    """Split at the geometric midpoint of the longest occupied coordinate span."""
    xs, ys = x[node.sites], y[node.sites]
    x_span = float(xs.max() - xs.min())
    y_span = float(ys.max() - ys.min())
    if x_span <= 0 and y_span <= 0:
        return None
    use_x = x_span >= y_span  # fixed tie break makes IDs reproducible
    values = xs if use_x else ys
    midpoint = float((values.min() + values.max()) / 2.0)
    left = values <= midpoint
    if not left.any() or left.all():
        return None
    return (
        _Node(node.path + ".L", node.sites[left], node.masses[left], node.importance[left]),
        _Node(node.path + ".R", node.sites[~left], node.masses[~left], node.importance[~left]),
    )


def _split_candidate(node: _Node, x: np.ndarray, y: np.ndarray, parent_sse: float | None = None
                     ) -> tuple[float, tuple[_Node, _Node] | None, tuple[float, float] | None]:
    """Return a split and its score, retaining child SSEs for their next turn."""
    children = _split(node, x, y)
    if children is None:
        return 0.0, None, None
    parent_sse = _weighted_sse(node, x, y) if parent_sse is None else parent_sse
    child_sses = (_weighted_sse(children[0], x, y), _weighted_sse(children[1], x, y))
    score = parent_sse - child_sses[0] - child_sses[1]
    return max(0.0, score), children, child_sses


def _spatial_sse(nodes: list[_Node], x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Raw-portion-mass spatial error and represented mass for diagnostics.

    This deliberately differs from split priority: diagnostics measure physical
    centroid representation error, including background, without ROI weighting.
    """
    error = 0.0
    mass = 0.0
    for node in nodes:
        cx, cy = _centroid(node, x, y)
        dx = x[node.sites] - cx
        dy = y[node.sites] - cy
        error += float(np.dot(node.masses, dx * dx + dy * dy))
        mass += float(node.masses.sum())
    return error, mass


def _sign_name(sign: int) -> str:
    return "positive" if sign == 1 else "negative"


def _observation(node: _Node, sign: int, source_id: str, grid: ContributionGrid,
                 x: np.ndarray, y: np.ndarray, point_weight: np.ndarray) -> Observation:
    total = float(node.masses.sum())
    cx, cy = _centroid(node, x, y)
    effective = sqrt(float(np.dot(node.masses, point_weight[node.sites] ** 2) / total))
    # ``point_weight`` is constructed by _roi_components in the closed [0, 1]
    # interval, so its weighted RMS is also in that interval mathematically.
    # A near-one dot-product reduction can nevertheless round one ulp above
    # the closed Observation contract.  Do not hide a material violation.
    if 1.0 < effective <= 1.0 + 8.0 * np.finfo(float).eps:
        effective = 1.0
    return Observation(
        source_id, (cx, cy, grid.z_m), total, sign,
        entity_id=grid.entity_id, quantity=grid.quantity, unit=grid.unit,
        attention_weight=effective,
    )


def _diagnostic(*, status: str, reason: str, budget: int, positive: float, negative: float,
                observations: tuple[Observation, ...], foreground_nodes: list[_Node],
                background_nodes: list[_Node], overview_nodes: list[_Node], x: np.ndarray,
                y: np.ndarray, reduced_detail: bool = False) -> dict:
    represented_positive = sum(item.strength for item in observations if item.sign == 1)
    represented_negative = sum(item.strength for item in observations if item.sign == -1)
    all_nodes = foreground_nodes + background_nodes + overview_nodes
    total_error, total_mass = _spatial_sse(all_nodes, x, y)
    foreground_error, foreground_mass = _spatial_sse(foreground_nodes, x, y)
    return {
        "status": status,
        "reason": reason,
        "method": "adaptive",
        "budget": budget,
        "foreground_sources": len(foreground_nodes),
        "background_sources": len(background_nodes),
        "overview_sources": len(overview_nodes),
        "reduced_detail": reduced_detail,
        "input_positive": positive,
        "input_negative": negative,
        "represented_positive": represented_positive,
        "represented_negative": represented_negative,
        "spatial_rms_m": sqrt(total_error / total_mass) if total_mass else 0.0,
        "foreground_spatial_rms_m": sqrt(foreground_error / foreground_mass) if foreground_mass else 0.0,
    }


def aggregate(grid: ContributionGrid, attention: Attention, budget: int = 4) -> AggregationResult:
    """Aggregate every nonzero raw contribution without mixing physical signs."""
    if type(budget) is not int or not 1 <= budget <= MAX_SOURCE_BUDGET:
        raise ValueError(f"source budget must be an integer in [1, {MAX_SOURCE_BUDGET}]")
    if not isinstance(attention, Attention):
        raise ValueError("attention must be an Attention")

    x = grid.x_m.ravel()
    y = grid.y_m.ravel()
    positive_values = grid.positive.ravel()
    negative_values = grid.negative.ravel()
    foreground_fraction, point_weight = _roi_components(x, y, attention)
    totals = {1: float(positive_values.sum()), -1: float(negative_values.sum())}
    signs = [sign for sign in (1, -1) if totals[sign] > 0]

    if budget < len(signs):
        diagnostic = _diagnostic(
            status="unsupported", reason="budget cannot represent both signs", budget=budget,
            positive=totals[1], negative=totals[-1], observations=(), foreground_nodes=[],
            background_nodes=[], overview_nodes=[], x=x, y=y,
        )
        return AggregationResult((), diagnostic)

    if not signs:
        diagnostic = _diagnostic(
            status="valid", reason="", budget=budget, positive=0.0, negative=0.0,
            observations=(), foreground_nodes=[], background_nodes=[], overview_nodes=[], x=x, y=y,
        )
        return AggregationResult((), diagnostic)

    foreground_nodes: list[tuple[int, _Node]] = []
    background_nodes: list[tuple[int, _Node]] = []
    overview_nodes: list[tuple[int, _Node]] = []
    detailed = budget >= 2 * len(signs)
    for sign in signs:
        values = positive_values if sign == 1 else negative_values
        active = np.flatnonzero(values > 0)
        foreground_masses = values[active] * foreground_fraction[active]
        background_masses = values[active] * (1.0 - foreground_fraction[active])
        if detailed:
            if float(foreground_masses.sum()) > 0:
                foreground_sites = active[foreground_masses > 0]
                foreground_masses = foreground_masses[foreground_masses > 0]
                foreground_nodes.append((sign, _Node("root", foreground_sites, foreground_masses,
                                                     foreground_masses * point_weight[foreground_sites])))
            if float(background_masses.sum()) > 0:
                background_sites = active[background_masses > 0]
                background_masses = background_masses[background_masses > 0]
                background_nodes.append((sign, _Node("background", background_sites, background_masses,
                                                     background_masses * point_weight[background_sites])))
        else:
            overview_nodes.append((sign, _Node("root", active, values[active],
                                                values[active] * point_weight[active])))

    # Only foreground/detail nodes and reduced-detail overviews are eligible for splitting.
    splittable = foreground_nodes if detailed else overview_nodes
    # Scores depend only on immutable node payloads.  Keep each candidate until
    # that leaf is selected, then score only the two newly-created children.
    candidates: dict[tuple[int, str], tuple[float, tuple[_Node, _Node], tuple[float, float]]] = {}
    for sign, node in splittable:
        score, children, child_sses = _split_candidate(node, x, y)
        if children is not None:
            candidates[(sign, node.path)] = (score, children, child_sses)
    while len(foreground_nodes) + len(background_nodes) + len(overview_nodes) < budget:
        if not candidates:
            break
        # Higher spatial reduction first; sign/path resolve equal numeric scores.
        (sign, path), (_, children, child_sses) = min(
            candidates.items(), key=lambda item: (-item[1][0], _sign_name(item[0][0]), item[0][1])
        )
        del candidates[(sign, path)]
        index = next(index for index, item in enumerate(splittable)
                     if item[0] == sign and item[1].path == path)
        splittable[index:index + 1] = [(sign, children[0]), (sign, children[1])]
        for child, child_sse in zip(children, child_sses):
            score, grandchildren, grandchild_sses = _split_candidate(child, x, y, child_sse)
            if grandchildren is not None:
                candidates[(sign, child.path)] = (score, grandchildren, grandchild_sses)

    observations = []
    for sign, node in sorted(background_nodes, key=lambda item: (item[0] != 1, item[1].path)):
        observations.append(_observation(node, sign, f"adaptive:{_sign_name(sign)}:background", grid, x, y, point_weight))
    for sign, node in sorted(foreground_nodes, key=lambda item: (item[0] != 1, item[1].path)):
        observations.append(_observation(node, sign, f"adaptive:{_sign_name(sign)}:focus:{node.path}", grid, x, y, point_weight))
    for sign, node in sorted(overview_nodes, key=lambda item: (item[0] != 1, item[1].path)):
        observations.append(_observation(node, sign, f"adaptive:{_sign_name(sign)}:overview:{node.path}", grid, x, y, point_weight))
    observation_tuple = tuple(observations)
    diagnostic = _diagnostic(
        status="valid", reason="reduced_detail" if not detailed else "", budget=budget,
        positive=totals[1], negative=totals[-1], observations=observation_tuple,
        foreground_nodes=[node for _, node in foreground_nodes],
        background_nodes=[node for _, node in background_nodes],
        overview_nodes=[node for _, node in overview_nodes], x=x, y=y,
        reduced_detail=not detailed,
    )
    return AggregationResult(observation_tuple, diagnostic)
