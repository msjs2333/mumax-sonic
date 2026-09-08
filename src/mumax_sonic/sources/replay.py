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

    def at(self, time_s):
        if not np.isfinite(time_s):
            raise ValueError('non-finite replay time')
        # Hold previous measured frame; never interpolate physical vectors.
        times = [f.sim_time_s for f in self.frames]
        index = max(0, min(len(times)-1, int(np.searchsorted(times, time_s, side='right'))-1))
        return self.frames[index]


def save_replay(path, frames):
    frames = tuple(frames)
    _validate_sequence(frames)
    first = frames[0]
    metadata = dict(schema_version=1, axes='yx', components='xyz', spacing_unit='m',
                    dx_m=first.dx_m, dy_m=first.dy_m, origin_m=first.origin_m,
                    entity_id=first.entity_id, segment_id=first.segment_id,
                    time_kind=first.time_kind, original_source_kind=first.source_kind)
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, vectors=np.stack([f.vectors for f in frames]),
                            mask=np.stack([f.mask for f in frames]),
                            times_s=np.array([f.sim_time_s for f in frames]),
                            metadata=np.array(json.dumps(metadata)))


def _validate_sequence(frames):
    if not frames:
        raise ValueError('empty replay')
    first = frames[0]
    for i, frame in enumerate(frames):
        if i and frame.sim_time_s <= frames[i-1].sim_time_s:
            raise ValueError('replay times must be strictly increasing')
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
        meta = json.loads(str(data['metadata'].item()))
        if (meta.get('schema_version'), meta.get('axes'), meta.get('components'), meta.get('spacing_unit')) != (1, 'yx', 'xyz', 'm'):
            raise ValueError('unsupported replay schema or coordinate convention')
        vectors, mask, times = data['vectors'], data['mask'], data['times_s']
        if vectors.ndim != 4 or vectors.shape[-1] != 3 or mask.shape != vectors.shape[:-1] or times.shape != (len(vectors),):
            raise ValueError('inconsistent replay array shapes')
        if mask.dtype != np.bool_:
            raise ValueError('mask must be boolean')
        frames = tuple(FieldFrame(v, meta['dx_m'], meta['dy_m'], float(t), tuple(meta['origin_m']),
                                  m, meta['entity_id'], 'replay', meta['segment_id'], i,
                                  meta['time_kind'], f'sha256:{digest}; original_source:{meta.get("original_source_kind", "unknown")}')
                       for i, (v, m, t) in enumerate(zip(vectors, mask, times)))
    _validate_sequence(frames)
    return FieldReplay(frames, digest)
