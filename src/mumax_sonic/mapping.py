"""Explicit shared-strength scale and physical-to-auditory coordinates."""
from dataclasses import dataclass
from math import cos, sin, pi, sqrt, isfinite
from .attention import Attention, select_sources
from .model import MAX_SOURCE_BUDGET, Sample, SonicScene, SonicSource


@dataclass(frozen=True)
class MappingResult:
    """A mapped scene together with raw-strength selection accounting."""

    scene: SonicScene
    report: dict


_ACCOUNTING_KEYS = (
    "total", "eligible", "selected", "omitted_budget", "excluded_sign",
    "excluded_attention", "output",
)


def _empty_accounting():
    return {key: 0.0 for key in _ACCOUNTING_KEYS}


def _with_fractions(accounting, *, valid: bool):
    if not valid:
        return {key: None for key in (*_ACCOUNTING_KEYS,
                                      "selected_fraction_total",
                                      "selected_fraction_eligible")}
    result = dict(accounting)
    result["selected_fraction_total"] = (
        result["selected"] / result["total"] if result["total"] else None)
    result["selected_fraction_eligible"] = (
        result["selected"] / result["eligible"] if result["eligible"] else None)
    return result


def _effective_validity(sample: Sample) -> str:
    # Coverage only downgrades an otherwise-valid sample.  Preserve a more
    # informative acquisition reason such as warming_up or stale.
    return "invalid" if sample.validity == "valid" and sample.coverage < 1 else sample.validity


def _validate_arguments(mode, master_gain, budget, strength_reference):
    if mode not in {"both", "positive", "negative"}:
        raise ValueError("unknown sign mode")
    if not isfinite(master_gain) or not 0 <= master_gain <= 1:
        raise ValueError("master gain must be in [0,1]")
    if type(budget) is not int or not 1 <= budget <= MAX_SOURCE_BUDGET:
        raise ValueError(f"budget must be an integer in [1, {MAX_SOURCE_BUDGET}]")
    if not isfinite(strength_reference) or strength_reference <= 0:
        raise ValueError('strength reference must be finite and positive')


def _report_groups(sample, attention, mode, selected_ids, output_ids, valid):
    grouped = {}
    for observation in sample.observations:
        key = (observation.entity_id, observation.quantity, observation.unit)
        channels = grouped.setdefault(key, {
            "absolute": _empty_accounting(),
            "positive": _empty_accounting(),
            "negative": _empty_accounting(),
        })
        applicable = ("absolute", "positive" if observation.sign > 0 else "negative")
        for channel in applicable:
            entry = channels[channel]
            strength = observation.strength
            entry["total"] += strength
            if mode != "both" and observation.sign != (1 if mode == "positive" else -1):
                entry["excluded_sign"] += strength
            elif strength <= 0 or attention.weight(observation) <= 0:
                entry["excluded_attention"] += strength
            else:
                entry["eligible"] += strength
                if observation.source_id in selected_ids:
                    entry["selected"] += strength
                    if observation.source_id in output_ids:
                        entry["output"] += strength
                else:
                    entry["omitted_budget"] += strength
    return [
        {
            "entity_id": key[0], "quantity": key[1], "unit": key[2],
            **{channel: _with_fractions(value, valid=valid)
               for channel, value in channels.items()},
        }
        for key, channels in sorted(grouped.items())
    ]


def map_sample_with_report(sample: Sample, attention: Attention, *, mode: str = "both",
                           master_gain: float = 0.15, budget: int = 4,
                           audible: bool = True,
                           strength_reference: float = 1.0) -> MappingResult:
    _validate_arguments(mode, master_gain, budget, strength_reference)
    validity = _effective_validity(sample)
    science_valid = validity == "valid"
    candidates = tuple(o for o in sample.observations
                       if mode == "both" or o.sign == (1 if mode == "positive" else -1))
    selected = select_sources(candidates, attention, budget) if science_valid else ()
    selected_ids = [o.source_id for o in selected]
    output_enabled = bool(audible and master_gain > 0)
    result = []
    # Preserve the legacy master-gain behaviour (allocated zero-gain sources
    # remain visible in a valid scene), while a paused/non-audible scene has no
    # active source objects.  Both cases retain selection accounting below.
    if science_valid and audible:
        selected_for_scene = selected
    else:
        selected_for_scene = ()
    for o in selected_for_scene:
        x, y, _ = o.position_m
        azimuth = max(-1, min(1, (x-attention.origin_m[0]) / attention.extent_m)) * pi / 3
        elevation = max(-1, min(1, (y-attention.origin_m[1]) / attention.extent_m)) * pi / 6
        position = (sin(azimuth)*cos(elevation), sin(elevation), -cos(azimuth)*cos(elevation))
        # Same fixed reference (strength=1) and bus headroom for both signs.
        # No per-frame/per-sign normalization, including during solo.
        gain = master_gain / budget * sqrt(min(o.strength / strength_reference, 1.0)) * attention.weight(o)
        result.append(SonicSource(o.source_id, position, gain, o.sign, o.orientation_rad, o.orientation_enabled))
    scene = SonicScene(tuple(result), sample.sim_time_s, validity)
    output_ids = [source.source_id for source in scene.sources if source.gain > 0]
    report = {
        "budget": budget,
        "validity": validity,
        "effective_sample_quality": validity,
        "data_coverage": sample.coverage,
        "settings": {
            "mode": mode,
            "master_gain": master_gain,
            "strength_reference": strength_reference,
            "gain_divisor": budget,
            "attention": {
                "center": attention.center,
                "radius": attention.radius,
                "background": attention.background,
                "extent_m": attention.extent_m,
                "origin_m": attention.origin_m,
            },
        },
        "output_enabled": output_enabled,
        "selected_ids": selected_ids,
        "output_ids": output_ids,
        "groups": _report_groups(sample, attention, mode, selected_ids, output_ids,
                                   science_valid),
    }
    return MappingResult(scene, report)


def map_sample(sample: Sample, attention: Attention, *, mode: str = "both",
               master_gain: float = 0.15, budget: int = 4, audible: bool = True,
               strength_reference: float = 1.0) -> SonicScene:
    """Map a sample to a scene; use :func:`map_sample_with_report` for audit data.

    ``master_gain / budget`` is intentional bus headroom; it reduces each
    allocated source gain without changing the raw-strength selection report.
    """
    return map_sample_with_report(
        sample, attention, mode=mode, master_gain=master_gain, budget=budget,
        audible=audible, strength_reference=strength_reference).scene
