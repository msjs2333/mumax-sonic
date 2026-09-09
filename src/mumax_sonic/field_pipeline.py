"""Field observations are computed before attention and audio source selection."""
from dataclasses import dataclass, replace
import json
import numpy as np
from .fields import FieldFrame
from .model import Observation, Sample
from .observers.topology import topology
from .observers.texture import direction_angle
from .aggregation import ContributionGrid, aggregate


@dataclass(frozen=True)
class FieldView:
    field: FieldFrame
    sample: Sample
    summary: str
    diagnostic: dict
    contributions: ContributionGrid | None = None


def _tiles(shape, divisions=4):
    for j, rows in enumerate(np.array_split(np.arange(shape[0]), min(divisions, shape[0]))):
        for i, cols in enumerate(np.array_split(np.arange(shape[1]), min(divisions, shape[1]))):
            yield f'{j}-{i}', np.ix_(rows, cols)


def observe_field(frame, recipe='topology', *, method='solid_angle', boundary='open',
                  domain_axis=(1, 0, 0), reference_axis=(0, 1, 0), previous=None, max_dt_s=None,
                  history=None, band_config=None):
    observations = []
    contributions = None
    validity_override = None
    input_positive = input_negative = 0.0
    aggregation_basis = 'observer contributions before fixed 4x4 spatial tiles'
    if recipe == 'topology':
        result = topology(frame.vectors, frame.dx_m, frame.dy_m, mask=frame.mask,
                          method=method, boundary=boundary)
        x, y = result.x_m + frame.origin_m[0], result.y_m + frame.origin_m[1]
        contributions = ContributionGrid(x, y, np.where(result.valid, result.positive, 0),
            np.where(result.valid, result.negative, 0), frame.origin_m[2], frame.entity_id, f'Q_tile_{method}', '1')
        for tile, index in _tiles(result.positive.shape):
            for sign, channel in ((1, result.positive), (-1, result.negative)):
                weights = np.where(result.valid[index], channel[index], 0.0)
                total = float(np.sum(weights))
                if total <= 0 or not np.isfinite(total):
                    continue
                position = (float(np.sum(x[index]*weights)/total),
                            float(np.sum(y[index]*weights)/total), frame.origin_m[2])
                observations.append(Observation(f'q:{tile}:{sign:+d}', position, total, sign,
                                                entity_id=frame.entity_id, quantity=f'Q_tile_{method}', unit='1'))
        coverage = result.coverage
        input_positive, input_negative = result.q_pos, result.q_neg
        summary = f'Q+ {result.q_pos:.4f}  Q− {result.q_neg:.4f}  Qnet {result.q_net:.4f}  Qabs {result.q_abs:.4f}'
        diagnostic = dict(recipe=recipe, method=method, boundary=boundary, surface='+z',
                          q_pos=result.q_pos, q_neg=result.q_neg, q_net=result.q_net,
                          q_abs=result.q_abs, coverage=coverage, warnings=list(result.warnings))
    elif recipe == 'direction':
        result = direction_angle(frame.vectors, domain_axis, reference_axis, mask=frame.mask)
        ny, nx = frame.vectors.shape[:2]
        input_positive = float(np.sum(np.where(result.valid, np.sum(result.projection**2, axis=-1), 0)))/(ny*nx)
        aggregation_basis = 'valid transverse projection weight before circular tile reduction; not all texture content'
        x, y = np.meshgrid(np.arange(nx)*frame.dx_m+frame.origin_m[0],
                           np.arange(ny)*frame.dy_m+frame.origin_m[1])
        # Transverse projection highlights a declared antiparallel-domain core.
        # It is not automatic wall detection or object tracking.
        for tile, index in _tiles(result.valid.shape):
            weights = np.where(result.valid[index], np.sum(result.projection[index]**2, axis=-1), 0.0)
            total = float(np.sum(weights))
            if total <= 1e-12:
                continue
            angle = np.where(result.valid[index], result.angle_rad[index], 0)
            circular = np.sum(weights*np.exp(1j*angle))
            if abs(circular) <= 1e-6*total:
                continue  # opposing directions have no single aggregate angle
            position = (float(np.sum(x[index]*weights)/total), float(np.sum(y[index]*weights)/total), frame.origin_m[2])
            observations.append(Observation(f'phi:{tile}', position, total/(ny*nx), 1,
                float(np.angle(circular)), frame.entity_id, 'transverse_area_fraction', '1', True))
        finite = np.all(np.isfinite(frame.vectors), axis=-1) & (np.linalg.norm(frame.vectors, axis=-1)>1e-12)
        coverage = float(np.sum(finite & frame.mask)/np.sum(frame.mask)) if np.any(frame.mask) else 0.0
        summary = f'连续取向 · 有效投影 {np.count_nonzero(result.valid)}/{np.count_nonzero(frame.mask)} · 固定声明基底'
        diagnostic = dict(recipe=recipe, domain_axis=list(result.d), e1=list(result.e1), e2=list(result.e2),
                          coverage=coverage, angle_unit='rad', automatic_wall_detection=False)
    elif recipe == 'activity':
        from .observers.activity import angular_activity
        result = angular_activity(previous, frame, max_dt_s=max_dt_s)
        coverage = result.coverage
        input_positive = result.mean_rad_s
        validity_override = result.validity
        ny, nx = frame.vectors.shape[:2]
        x, y = np.meshgrid(np.arange(nx)*frame.dx_m+frame.origin_m[0],
                           np.arange(ny)*frame.dy_m+frame.origin_m[1])
        material_count = int(np.count_nonzero(frame.mask))
        contributions = ContributionGrid(x, y, np.where(result.valid, result.rate_rad_s, 0)/max(1, material_count),
            np.zeros_like(x), frame.origin_m[2], frame.entity_id, 'angular_activity_mean_contribution', 'rad/s')
        for tile, index in _tiles(result.valid.shape):
            weights = np.where(result.valid[index], result.rate_rad_s[index], 0.0)
            total = float(np.sum(weights))
            if total <= 0 or not np.isfinite(total):
                continue
            position = (float(np.sum(x[index]*weights)/total), float(np.sum(y[index]*weights)/total), frame.origin_m[2])
            observations.append(Observation(f'activity:{tile}', position, total/material_count, 1,
                entity_id=frame.entity_id, quantity='angular_activity_mean_contribution', unit='rad/s'))
        summary = f'活动均值 {result.mean_rad_s:.3g} rad/s · 峰值 {result.max_rad_s:.3g} rad/s'
        diagnostic = dict(recipe=recipe, rate_unit='rad/s', mean_rad_s=result.mean_rad_s,
            max_rad_s=result.max_rad_s, dt_s=result.dt_s, max_dt_s=max_dt_s,
            coverage=coverage, validity=result.validity, reason=result.reason, warnings=list(result.warnings),
            sequence=frame.sequence, previous_sequence=previous.sequence if previous is not None else None,
            previous_sim_time_s=previous.sim_time_s if previous is not None else None,
            aggregation='sum of tile rates / current material site count; equals material mean on full coverage')
        if 'sequence_inferred:v1' in frame.provenance:
            diagnostic['warnings'].append('source sequences inferred from v1 records; sequence gaps cannot be detected')
    elif recipe == 'band':
        from .observers.band import BandConfig, band_power
        config = band_config or BandConfig()
        history = tuple(history) if history is not None else (frame,)
        if not history or history[-1] is not frame:
            raise ValueError('band history must end at the observed frame')
        result = band_power(history, config)
        coverage, validity_override = result.coverage, result.validity
        input_positive = result.mean_power
        ny, nx = frame.vectors.shape[:2]
        x, y = np.meshgrid(np.arange(nx)*frame.dx_m+frame.origin_m[0],
                           np.arange(ny)*frame.dy_m+frame.origin_m[1])
        count = int(np.count_nonzero(frame.mask))
        contributions = ContributionGrid(x, y, np.where(result.valid, result.power, 0)/max(1, count),
            np.zeros_like(x), frame.origin_m[2], frame.entity_id, 'transverse_band_mean_square_contribution', '1')
        for tile, index in _tiles(result.valid.shape):
            weights = np.where(result.valid[index], result.power[index], 0.0)
            total = float(np.sum(weights))
            if total <= 0 or not np.isfinite(total):
                continue
            position = (float(np.sum(x[index]*weights)/total), float(np.sum(y[index]*weights)/total), frame.origin_m[2])
            observations.append(Observation(f'band:{tile}', position, total/count,
                entity_id=frame.entity_id, quantity='transverse_band_mean_square_contribution', unit='1'))
        summary = f'频带 {config.low_hz/1e9:g}–{config.high_hz/1e9:g} GHz · 均方强度 {result.mean_power:.3g}'
        diagnostic = dict(recipe=recipe, method='trailing periodic-Hann periodogram; temporal mean removed',
            low_hz=config.low_hz, high_hz=config.high_hz, reference_axis=list(config.reference_axis),
            mean_power=result.mean_power, max_power=result.max_power, unit='1',
            samples_available=result.samples_available, samples_required=result.samples_required,
            dt_s=result.dt_s, frequency_resolution_hz=result.frequency_resolution_hz,
            window_span_s=result.window_span_s, bin_frequencies_hz=list(result.bin_frequencies_hz),
            coverage=coverage, validity=result.validity, reason=result.reason, warnings=list(result.warnings),
            aggregation='per-site spectral power before tile sum / material site count',
            physical_energy_computed=False)
        if 'sequence_inferred:v1' in frame.provenance:
            diagnostic['warnings'].append('v1 inferred sequences cannot detect missing source frames')
    else:
        raise ValueError('unknown field recipe')
    validity = validity_override or ('valid' if coverage == 1 else 'invalid')
    sample = Sample(frame.sim_time_s, tuple(observations), frame.sequence, frame.segment_id,
                    validity, coverage, frame.source_kind, time_kind=frame.time_kind)
    aggregation = {}
    for label, sign, before in (('positive', 1, input_positive), ('negative', -1, input_negative),
                                ('absolute', None, input_positive+input_negative)):
        represented = sum(o.strength for o in observations if sign is None or o.sign == sign)
        usable = validity == 'valid' and coverage == 1
        aggregation[label] = dict(input=before if usable else None,
            represented=represented if usable else None,
            omitted=max(0.0, before-represented) if usable else None)
    diagnostic['spatial_aggregation'] = dict(method='fixed_4x4', basis=aggregation_basis, channels=aggregation)
    diagnostic.update(source_kind=frame.source_kind, entity_id=frame.entity_id, provenance=frame.provenance,
                      shape=list(frame.vectors.shape), dx_m=frame.dx_m, dy_m=frame.dy_m,
                      origin_m=list(frame.origin_m), sim_time_s=frame.sim_time_s)
    # Preserve structured OVF lineage through the existing NPZ provenance prefix.
    if '{"format": "OVF2"' in frame.provenance:
        try:
            info = json.loads(frame.provenance[frame.provenance.index('{"format": "OVF2"'):])
            if isinstance(info, dict) and info.get('format') == 'OVF2':
                diagnostic['input'] = info
        except (ValueError, TypeError):
            pass  # arbitrary legacy provenance remains available as text
    return FieldView(frame, sample, summary, diagnostic, contributions)


def apply_aggregation(view, attention, budget=4, mode='fixed'):
    """Regroup already computed scalar contributions; never recompute physics."""
    if mode == 'fixed':
        return view
    if mode != 'adaptive':
        raise ValueError('unknown spatial aggregation mode')
    diagnostic = dict(view.diagnostic)
    if view.sample.validity != 'valid' or view.sample.coverage != 1:
        info = dict(status=view.sample.validity, reason='physical observation is not fully valid')
    elif view.contributions is None:
        info = dict(status='unsupported', reason='continuous direction requires circular aggregation; using fixed tiles')
    else:
        result = aggregate(view.contributions, attention, budget)
        info = result.diagnostic
        if info['status'] == 'valid':
            channels = {}
            for label, signs in (('positive', (1,)), ('negative', (-1,)), ('absolute', (1, -1))):
                before = view.diagnostic['spatial_aggregation']['channels'][label]['input']
                represented = sum(o.strength for o in result.observations if o.sign in signs)
                channels[label] = dict(input=before, represented=represented, omitted=max(0.0, before-represented))
            diagnostic['spatial_aggregation'] = dict(method='adaptive',
                basis='per-site observer contributions; signs conserved separately; soft focus and background', channels=channels)
            diagnostic['adaptive_aggregation'] = info
            return replace(view, sample=replace(view.sample, observations=result.observations), diagnostic=diagnostic)
    diagnostic['adaptive_aggregation'] = dict(info, fallback='fixed_4x4')
    return replace(view, diagnostic=diagnostic)


def field_selection_report(view, report):
    """Relate candidate coverage to the separately measured pre-tile quantity."""
    result = dict(report)
    aggregation = view.diagnostic['spatial_aggregation']
    result['spatial_aggregation'] = aggregation
    if 'adaptive_aggregation' in view.diagnostic:
        result['adaptive_aggregation'] = view.diagnostic['adaptive_aggregation']
    fractions = {}
    for channel, values in aggregation['channels'].items():
        before = values['input']
        selected = sum(g[channel]['selected'] or 0.0 for g in report['groups'])
        fractions[channel] = min(1.0, max(0.0, selected/before)) if before is not None and before > 0 else None
    result['selected_fraction_observer_input'] = fractions
    return result
