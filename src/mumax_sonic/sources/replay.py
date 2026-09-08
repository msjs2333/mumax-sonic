"""Portable NPZ vector replay, with no pickle and content hash provenance."""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from ..fields import FieldFrame

MAX_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class FieldReplay:
    frames: tuple[FieldFrame, ...]
    sha256: str

    def __post_init__(self):
        frames = tuple(self.frames)
        _validate_sequence(frames)
        object.__setattr__(self, 'frames', frames)

    def at(self, time_s):
        # Hold previous measured frame; never interpolate physical vectors.
        return self.frames[self.index_at(time_s)]

    def index_at(self, time_s):
        """Return the held frame index for a physical replay time.

        The index is clamped to the first/last source record outside the
        recorded time range.  It is an index into source records, so a seek
        or a slow renderer cannot change which record is the predecessor.
        """
        if not np.isfinite(time_s):
            raise ValueError('non-finite replay time')
        times = [f.sim_time_s for f in self.frames]
        return max(0, min(len(times) - 1,
                          int(np.searchsorted(times, time_s, side='right')) - 1))

    def pair_at(self, time_s):
        """Return ``(source predecessor, held current frame)`` at *time_s*."""
        index = self.index_at(time_s)
        return (self.frames[index - 1] if index else None, self.frames[index])


def save_replay(path, frames):
    frames = tuple(frames)
    _validate_sequence(frames)
    if any(frame.sequence > np.iinfo(np.int64).max for frame in frames):
        raise ValueError('replay sequence exceeds int64 storage range')
    first = frames[0]
    metadata = dict(schema_version=2, axes='yx', components='xyz', spacing_unit='m',
                    dx_m=first.dx_m, dy_m=first.dy_m, origin_m=first.origin_m,
                    entity_id=first.entity_id, segment_id=first.segment_id,
                    time_kind=first.time_kind, original_source_kind=first.source_kind)
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, vectors=np.stack([f.vectors for f in frames]),
                            mask=np.stack([f.mask for f in frames]),
                            times_s=np.array([f.sim_time_s for f in frames]),
                            sequences=np.array([f.sequence for f in frames], dtype=np.int64),
                            provenance=np.array([f.provenance for f in frames], dtype=np.str_),
                            metadata=np.array(json.dumps(metadata)))


def _validate_sequence(frames):
    if not frames:
        raise ValueError('empty replay')
    first = frames[0]
    for i, frame in enumerate(frames):
        if i and frame.sim_time_s <= frames[i-1].sim_time_s:
            raise ValueError('replay times must be strictly increasing')
        if i and frame.sequence <= frames[i-1].sequence:
            raise ValueError('replay sequences must be strictly increasing')
        if any(getattr(frame, name) != getattr(first, name) for name in
               ('dx_m', 'dy_m', 'origin_m', 'entity_id', 'segment_id', 'time_kind', 'source_kind')) or frame.vectors.shape != first.vectors.shape:
            raise ValueError('replay geometry/entity/segment must remain fixed')


def load_replay(path):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('replay exceeds 256 MiB prototype limit')
    raw = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if sum(i.file_size for i in archive.infolist()) > MAX_BYTES:
            raise ValueError('unpacked replay exceeds 256 MiB prototype limit')
    digest = hashlib.sha256(raw).hexdigest()
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        try:
            meta = json.loads(str(data['metadata'].item()))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError('invalid replay metadata') from exc
        if not isinstance(meta, dict):
            raise ValueError('invalid replay metadata')
        version = meta.get('schema_version')
        if (type(version) is not int or version not in (1, 2) or
                (meta.get('axes'), meta.get('components'), meta.get('spacing_unit')) != ('yx', 'xyz', 'm')):
            raise ValueError('unsupported replay schema or coordinate convention')
        try:
            vectors, mask, times = data['vectors'], data['mask'], data['times_s']
        except KeyError as exc:
            raise ValueError('missing replay array') from exc
        if vectors.ndim != 4 or vectors.shape[-1] != 3 or mask.shape != vectors.shape[:-1] or times.shape != (len(vectors),) or not len(vectors):
            raise ValueError('inconsistent replay array shapes')
        if mask.dtype != np.bool_:
            raise ValueError('mask must be boolean')
        if not (np.issubdtype(times.dtype, np.integer) or
                np.issubdtype(times.dtype, np.floating)):
            raise ValueError('times must be finite non-negative numbers')
        try:
            if not np.all(np.isfinite(times)) or np.any(times < 0):
                raise ValueError('times must be finite non-negative numbers')
        except TypeError as exc:
            raise ValueError('times must be finite non-negative numbers') from exc
        if np.any(np.diff(times) <= 0):
            raise ValueError('replay times must be strictly increasing')

        if version == 2:
            try:
                sequences, provenance = data['sequences'], data['provenance']
            except KeyError as exc:
                raise ValueError('missing v2 replay array') from exc
            if sequences.shape != (len(vectors),) or not np.issubdtype(sequences.dtype, np.integer):
                raise ValueError('sequences must be a one-dimensional integer array')
            sequence_values = [int(value) for value in sequences]
            if any(value < 0 for value in sequence_values) or any(
                    current <= previous
                    for previous, current in zip(sequence_values, sequence_values[1:])):
                raise ValueError('replay sequences must be strictly increasing')
            if provenance.shape != (len(vectors),) or provenance.dtype.kind != 'U':
                raise ValueError('provenance must be a one-dimensional Unicode array')
            provenance_values = [str(value) for value in provenance]
        else:
            sequence_values = list(range(len(vectors)))
            provenance_values = ['sequence_inferred:v1'] * len(vectors)

        try:
            source_kind = meta.get('original_source_kind', 'unknown')
            frames = tuple(FieldFrame(v, meta['dx_m'], meta['dy_m'], float(t), tuple(meta['origin_m']),
                                      m, meta['entity_id'], 'replay', meta['segment_id'], sequence,
                                      meta['time_kind'], _loaded_provenance(digest, source_kind, provenance_text))
                           for v, m, t, sequence, provenance_text in
                           zip(vectors, mask, times, sequence_values, provenance_values))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('invalid replay frame metadata') from exc
    return FieldReplay(frames, digest)


def _loaded_provenance(digest, source_kind, frame_provenance):
    """Prefix loaded provenance with the immutable archive identity."""
    parts = [f'sha256:{digest}', f'original_source:{source_kind}']
    if frame_provenance:
        parts.append(frame_provenance)
    return '; '.join(parts)
