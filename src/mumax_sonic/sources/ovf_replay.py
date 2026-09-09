"""Explicit OVF sequence manifests; filenames and wall time are not physical time."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ..fields import FieldFrame
from .replay import FieldReplay, MAX_BYTES
from .ovf import read_ovf


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def _path(base, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('manifest file paths must be nonempty strings')
    path = Path(value)
    return path if path.is_absolute() else base/path


def _hash(raw, expected, name):
    digest = hashlib.sha256(raw).hexdigest()
    if not isinstance(expected, str) or digest != expected.lower():
        raise ValueError(f'{name} SHA256 mismatch or missing hash')
    return digest


def _bounded_bytes(path, limit):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit+1)
    if len(raw) > limit:
        raise ValueError('input exceeds prototype size limit')
    return raw


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate manifest key: {key}')
        result[key] = value
    return result


def load_ovf_replay(path):
    """Read one declared physical segment, using immutable, hash-bound inputs."""
    path = Path(path)
    if path.stat().st_size > 8*1024*1024:
        raise ValueError('OVF manifest exceeds 8 MiB')
    raw = _bounded_bytes(path, 8*1024*1024)
    meta = json.loads(raw, object_pairs_hook=_unique_keys)
    return _load_ovf_manifest(path, meta, raw)


def _load_ovf_manifest(path, meta, raw=None):
    """Validate already-read manifest metadata and decode its declared frames.

    This internal entry point lets a follower apply exactly the offline OVF
    validation to a newly appended record without reading historical bodies.
    """
    path = Path(path)
    if raw is None:
        raw = json.dumps(meta, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    if not isinstance(meta, dict) or type(meta.get('schema_version')) is not int or meta['schema_version'] != 1:
        raise ValueError('unsupported OVF manifest schema')
    for name in ('entity_id', 'segment_id'):
        if not isinstance(meta.get(name), str) or not meta[name].strip():
            raise ValueError(f'{name} must be explicitly declared')
    if meta.get('time_kind') not in ('dynamics', 'static', 'relaxation'):
        raise ValueError('declare dynamics, static or relaxation time_kind')
    if meta.get('origin') not in ('simulation', 'synthetic', 'unknown'):
        raise ValueError('declare origin as simulation, synthetic or unknown')
    if meta.get('quantity') not in ('magnetization_direction', 'order_parameter_direction'):
        raise ValueError('only declared magnetic/order-parameter directions are supported')
    unit = meta.get('value_unit')
    if unit not in ('1', 'A/m') or (meta['quantity'] == 'order_parameter_direction' and unit != '1'):
        raise ValueError('direction input units must be 1, or A/m for magnetization')
    components = meta.get('components')
    if not isinstance(components, list) or len(components) != 3 or set(components) != {'x', 'y', 'z'}:
        raise ValueError('declare the three stored components in their actual order')
    order = [components.index(axis) for axis in ('x', 'y', 'z')]
    records = meta.get('frames')
    if not isinstance(records, list) or not 1 <= len(records) <= 4096:
        raise ValueError('manifest must contain 1..4096 explicit frame records')
    z_index = meta.get('z_index')
    if z_index is not None:
        _integer(z_index, 'z_index')
    # No zero-vector heuristic: absent geometry is an explicit full-grid choice.
    mask_spec = meta.get('mask')
    mask = None
    mask_identity = 'all grid sites explicitly declared material'
    if mask_spec != 'all':
        if not isinstance(mask_spec, dict) or set(mask_spec) != {'file', 'sha256'}:
            raise ValueError('mask must be "all" or a hash-bound boolean NPY file')
        mask_path = _path(path.parent, mask_spec['file'])
        if mask_path.stat().st_size > MAX_BYTES:
            raise ValueError('mask exceeds prototype memory limit')
        mask_raw = _bounded_bytes(mask_path, MAX_BYTES)
        mask_identity = _hash(mask_raw, mask_spec['sha256'], 'mask')
        import io
        stream = io.BytesIO(mask_raw)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError('material mask requires NPY version 1 or 2')
        if dtype != np.bool_ or len(shape) not in (2,3) or any(n <= 0 for n in shape):
            raise ValueError('material mask must be a boolean NPY array')
        count = math.prod(shape)
        if count > MAX_BYTES or len(mask_raw)-stream.tell() != count:
            raise ValueError('material mask payload size disagrees with header')
        mask = np.frombuffer(mask_raw, dtype=np.bool_, count=count, offset=stream.tell()).reshape(shape, order='F' if fortran else 'C')
    manifest_hash = hashlib.sha256(raw).hexdigest()
    time_evidence = meta.get('time_evidence')
    evidence_identity = None
    if time_evidence is not None:
        if not isinstance(time_evidence, dict) or not isinstance(time_evidence.get('description'), str) or not time_evidence['description'].strip():
            raise ValueError('time evidence requires a description and hash-bound file')
        evidence_path = _path(path.parent, time_evidence.get('file'))
        if evidence_path.stat().st_size > 8*1024*1024:
            raise ValueError('time evidence exceeds 8 MiB')
        evidence_identity = _hash(_bounded_bytes(evidence_path,8*1024*1024), time_evidence.get('sha256'), 'time evidence')
    frames, seen = [], set()
    total_bytes = decoded_bytes = 0
    geometry = None
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('frame record must be an object')
        source = _path(path.parent, record.get('file')).resolve()
        if source in seen:
            raise ValueError('duplicate OVF file in manifest')
        seen.add(source)
        sequence = _integer(record.get('sequence'), 'sequence')
        if total_bytes + source.stat().st_size > MAX_BYTES:
            raise ValueError('OVF sequence exceeds 256 MiB input limit')
        field = read_ovf(source)
        total_bytes += field.byte_count
        if total_bytes > MAX_BYTES:
            raise ValueError('OVF sequence exceeds 256 MiB input limit')
        if not isinstance(record.get('sha256'), str) or field.sha256 != record['sha256'].lower():
            raise ValueError('OVF SHA256 mismatch or missing hash')
        if tuple(field.units) != (unit,)*3:
            raise ValueError('OVF value units disagree with manifest')
        # Recognisable labels may not contradict the declared component order.
        for label, component in zip(field.labels, components):
            suffix = label.lower().rsplit('_', 1)[-1]
            if suffix in ('x', 'y', 'z') and suffix != component:
                raise ValueError('OVF component labels contradict manifest order')
        declared_time = record.get('time_s')
        if declared_time is None:
            if field.time_s is None:
                raise ValueError('missing physical time: provide time_s or MuMax total-time header')
            time_s, time_source = field.time_s, 'OVF Total simulation time'
        else:
            time_s = _number(declared_time, 'time_s')
            time_source = 'manifest time_s'
            if field.time_s is not None and not math.isclose(time_s, field.time_s, rel_tol=1e-9, abs_tol=0.0):
                raise ValueError('manifest physical time conflicts with OVF header')
        if time_s < 0:
            raise ValueError('physical time must be nonnegative')
        layer = z_index
        if layer is None:
            if field.vectors.shape[0] != 1:
                raise ValueError('multilayer OVF requires explicit z_index; layers are never averaged')
            layer = 0
        if layer >= field.vectors.shape[0]:
            raise ValueError('z_index outside OVF mesh')
        current_geometry = (field.vectors.shape, tuple(field.step_m), tuple(field.origin_m), tuple(field.labels))
        if geometry is None:
            geometry = current_geometry
        elif current_geometry != geometry:
            raise ValueError('OVF mesh or component labels change within one declared segment')
        vectors = field.vectors[layer][..., order]
        decoded_bytes += vectors.size*8 + vectors.shape[0]*vectors.shape[1]
        if decoded_bytes > MAX_BYTES:
            raise ValueError('decoded OVF sequence exceeds 256 MiB limit')
        selected_mask = mask
        if mask is not None:
            if mask.shape == field.vectors.shape[:3]:
                selected_mask = mask[layer]
            elif mask.shape != vectors.shape[:2]:
                raise ValueError('mask shape must match selected plane or complete OVF mesh')
        provenance = json.dumps(dict(format='OVF2', ovf_sha256=field.sha256,
            manifest_sha256=manifest_hash, origin=meta['origin'], quantity=meta['quantity'],
            input_unit=unit, components=components, labels=list(field.labels), z_index=layer,
            mesh_shape_zyx=list(field.vectors.shape[:3]), step_m=list(field.step_m),
            mask_source=mask_identity, time_source=time_source, encoding=field.encoding,
            header_time_s=field.time_s, header_time_hint=list(getattr(field, 'time_hint', ())),
            time_evidence_sha256=evidence_identity,
            time_evidence_description=time_evidence['description'] if time_evidence else None,
            sequence_source=meta.get('sequence_source', 'explicit manifest records')), ensure_ascii=False)
        origin = (field.origin_m[0], field.origin_m[1], field.origin_m[2]+layer*field.step_m[2])
        frames.append(FieldFrame(vectors, field.step_m[0], field.step_m[1], time_s, origin,
            selected_mask, meta['entity_id'], 'replay', meta['segment_id'], sequence, meta['time_kind'], provenance))
    return FieldReplay(tuple(frames), manifest_hash)


def load_field_replay(path):
    """Shared GUI/CLI dispatch, keeping all file I/O outside the audio worker."""
    if Path(path).suffix.lower() == '.json':
        return load_ovf_replay(path)
    from .replay import load_replay
    return load_replay(path)
