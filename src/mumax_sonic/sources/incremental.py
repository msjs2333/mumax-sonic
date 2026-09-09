"""Bounded, append-only OVF manifest reader for live publication workers."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

from .ovf_replay import _bounded_bytes, _load_ovf_manifest, _number, _integer, _unique_keys
from .ovf import read_ovf
from .replay import FieldReplay, MAX_BYTES


_MANIFEST_LIMIT = 8 * 1024 * 1024


def _stamp(path):
    stat = Path(path).stat()
    return (stat.st_size, stat.st_mtime_ns, getattr(stat, 'st_ino', None))


def _fingerprint(frame):
    """The live raw-lineage fingerprint, kept here to avoid a live import cycle."""
    data = json.loads(frame.provenance)
    raw = dict(
        ovf_sha256=data.get('ovf_sha256'), quantity=data.get('quantity'),
        input_unit=data.get('input_unit'), origin=data.get('origin'),
        mask_source=data.get('mask_source'), components=data.get('components'),
        labels=data.get('labels'), z_index=data.get('z_index'),
        mesh_shape_zyx=data.get('mesh_shape_zyx'), step_m=data.get('step_m'),
        entity_id=frame.entity_id, segment_id=frame.segment_id,
        sequence=frame.sequence, sim_time_s=frame.sim_time_s,
        time_kind=frame.time_kind, dx_m=frame.dx_m, dy_m=frame.dy_m,
        origin_m=frame.origin_m,
    )
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _frame_bytes(frame):
    return frame.vectors.nbytes + frame.mask.nbytes


class IncrementalOVFReader:
    """Read immutable manifest prefixes once and retain only a recent frame tail.

    ``records`` and ``history_keys`` retain bounded (at most 4096) metadata for
    the active segment.  Vector data are retained only for the requested tail.
    A failed publication leaves the previous accepted state intact.
    """

    def __init__(self, retain_frames=256, max_bytes=MAX_BYTES):
        if type(retain_frames) is not int or retain_frames < 2:
            raise ValueError('retain_frames must be an integer of at least 2')
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('max_bytes must be a positive integer')
        self.retain_frames = retain_frames
        self.max_bytes = max_bytes
        self._tail = OrderedDict()
        self._metadata = OrderedDict()
        self._segment = None
        self._static = None
        self._geometry = None
        self._manifest_hash = None
        self._read_frames_total = 0
        self._cache_hits_total = 0
        self._new_source_frames_count = 0
        self._records = ()

    @property
    def diagnostics(self):
        return dict(read_frames_total=self._read_frames_total,
                    cache_hits_total=self._cache_hits_total,
                    retained_frames=len(self._tail),
                    retained_bytes=sum(_frame_bytes(frame) for frame, _ in self._tail.values()),
                    source_frame_count=len(self._metadata))

    @property
    def records(self):
        return self._records

    @property
    def history_keys(self):
        return tuple(key for key, _ in self._metadata.items())

    @property
    def new_source_frames_count(self):
        return self._new_source_frames_count

    def load(self, path):
        """Return a replay tail after transactionally accepting one publication."""
        path = Path(path)
        if path.stat().st_size > _MANIFEST_LIMIT:
            raise ValueError('OVF manifest exceeds 8 MiB')
        raw = _bounded_bytes(path, _MANIFEST_LIMIT)
        meta = json.loads(raw, object_pairs_hook=_unique_keys)
        self._validate_manifest_shape(meta)
        records = meta['frames']
        records_by_sequence = {record['sequence']: record for record in records}
        segment = meta['segment_id']
        static = self._static_identity(meta)
        reset = self._segment is not None and segment != self._segment
        old_metadata = OrderedDict() if reset else self._metadata
        old_tail = OrderedDict() if reset else self._tail
        self._validate_prefix(meta, records_by_sequence, old_metadata, static, reset)

        # Work in copies.  No bad or half-written publication changes state.
        candidate_metadata = OrderedDict((key, dict(value)) for key, value in old_metadata.items())
        candidate_tail = OrderedDict(old_tail)
        metadata_by_sequence = {key[1]: (key, value) for key, value in candidate_metadata.items()}
        candidate_geometry = None if reset else self._geometry
        decoded = 0
        hits = 0
        for record in records:
            source = (path.parent / record['file']).resolve()
            key, prior = metadata_by_sequence.get(record['sequence'], (None, None))
            if prior is not None:
                if prior['record_identity'] != self._record_identity(record):
                    raise ValueError('published physical frame provenance changed')
                if _stamp(source) != prior['stamp']:
                    # A changed stamp is unusual for an immutable source.  Hash
                    # it before accepting a byte-identical atomic restore.
                    try:
                        changed = read_ovf(source)
                    except Exception as exc:
                        raise ValueError('published OVF source changed after acceptance') from exc
                    if changed.sha256 != record.get('sha256', '').lower():
                        raise ValueError('published OVF source changed after acceptance')
                    prior['stamp'] = _stamp(source)
                hits += 1
                continue
            one = dict(meta)
            one['frames'] = [record]
            replay = _load_ovf_manifest(path, one, raw)
            frame = replay.frames[0]
            key = (segment, frame.sequence, frame.sim_time_s)
            geometry = self._frame_geometry(frame)
            if candidate_geometry is None:
                candidate_geometry = geometry
            elif geometry != candidate_geometry:
                raise ValueError('OVF mesh, selected layer, or material mask changes within a segment')
            # Keep a strict N-frame working set while reading a large append.
            # A too-large requested tail fails; it is never silently shortened.
            if len(candidate_tail) >= self.retain_frames:
                candidate_tail.popitem(last=False)
            candidate_metadata[key] = dict(record_identity=self._record_identity(record), stamp=_stamp(source),
                                           fingerprint=_fingerprint(frame))
            metadata_by_sequence[frame.sequence] = (key, candidate_metadata[key])
            candidate_tail[key] = (frame, candidate_metadata[key])
            decoded += 1

        # A current manifest must name the entire accepted prefix.  Retained
        # data may be evicted, but metadata may not be silently slid away.
        published_sequences = {r['sequence'] for r in records}
        if any(key[1] not in published_sequences for key in candidate_metadata):
            raise ValueError('published history was truncated within a segment')
        if sum(_frame_bytes(frame) for frame, _ in candidate_tail.values()) > self.max_bytes:
            raise ValueError('requested retained OVF tail exceeds memory limit')
        if len(records) >= 2 and len(candidate_tail) < 2:
            raise ValueError('retained OVF tail must contain at least two frames')
        ordered = [metadata_by_sequence[record['sequence']][0] for record in records]
        if any(current[2] <= previous[2] for previous, current in zip(ordered, ordered[1:])):
            raise ValueError('replay times must be strictly increasing')

        self._segment, self._static, self._geometry = segment, static, candidate_geometry
        self._metadata, self._tail = candidate_metadata, candidate_tail
        self._manifest_hash = hashlib.sha256(raw).hexdigest()
        self._read_frames_total += decoded
        self._cache_hits_total += hits
        self._new_source_frames_count = decoded
        self._records = tuple((key[0], key[1], key[2],
                               candidate_metadata[key]['fingerprint'])
                              for key in candidate_metadata)
        frames = tuple(frame for frame, _ in candidate_tail.values())
        return FieldReplay(frames, self._manifest_hash)

    @staticmethod
    def _validate_manifest_shape(meta):
        if not isinstance(meta, dict) or meta.get('schema_version') != 1:
            raise ValueError('unsupported OVF manifest schema')
        for name in ('entity_id', 'segment_id'):
            if not isinstance(meta.get(name), str) or not meta[name].strip():
                raise ValueError(f'{name} must be explicitly declared')
        records = meta.get('frames')
        if not isinstance(records, list) or not 1 <= len(records) <= 4096:
            raise ValueError('manifest must contain 1..4096 explicit frame records')
        previous = None
        for record in records:
            if not isinstance(record, dict):
                raise ValueError('frame record must be an object')
            sequence = _integer(record.get('sequence'), 'sequence')
            time_s = IncrementalOVFReader._record_time(record)
            if time_s is not None and time_s < 0:
                raise ValueError('physical time must be nonnegative')
            if previous is not None and sequence <= previous[0]:
                raise ValueError('replay sequences must be strictly increasing')
            if previous is not None and time_s is not None and previous[1] is not None and time_s <= previous[1]:
                raise ValueError('replay times must be strictly increasing')
            previous = sequence, time_s
            if not isinstance(record.get('file'), str) or not record['file'].strip():
                raise ValueError('manifest file paths must be nonempty strings')

    @staticmethod
    def _record_time(record):
        # Header-derived times are verified by the shared decoder.  Metadata
        # history requires an explicit key, so use the already-decoded value
        # only on records that declare it; implicit-time records are decoded
        # once then canonicalised in metadata below through their frame time.
        value = record.get('time_s')
        if value is None:
            return None
        return _number(value, 'time_s')

    @staticmethod
    def _record_identity(record):
        return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    @staticmethod
    def _frame_geometry(frame):
        return (frame.vectors.shape, frame.dx_m, frame.dy_m, frame.origin_m,
                frame.mask.shape, hashlib.sha256(frame.mask.tobytes()).hexdigest())

    @staticmethod
    def _static_identity(meta):
        # Only fields that define the raw physical interpretation are frozen
        # across an append.  Publication annotations may change legitimately.
        static = {name: meta.get(name) for name in (
            'schema_version', 'entity_id', 'segment_id', 'time_kind', 'origin',
            'quantity', 'value_unit', 'components', 'mask', 'z_index',
        )}
        return hashlib.sha256(json.dumps(static, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def _validate_prefix(self, meta, records_by_sequence, old, static, reset):
        if not reset and self._segment is not None and static != self._static:
            raise ValueError('manifest physical specification changed within a segment')
        if old:
            published = set(records_by_sequence)
            if any(key[1] not in published for key in old):
                raise ValueError('published history was truncated within a segment')
            for key, entry in old.items():
                record = records_by_sequence.get(key[1])
                if record is None or entry['record_identity'] != self._record_identity(record):
                    raise ValueError('published physical frame provenance changed')
