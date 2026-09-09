"""Bounded, append-only OVF manifest reader for live publication workers."""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

from .ovf_replay import _bounded_bytes, _iter_ovf_manifest, _number, _integer, _unique_keys
from .replay import FieldReplay, MAX_BYTES


_MANIFEST_LIMIT = 8 * 1024 * 1024

def _frame_bytes(frame):
    return frame.vectors.nbytes + frame.mask.nbytes


class IncrementalOVFReader:
    """Read an append-only live manifest while retaining only a recent tail.

    Accepted sequence numbers are trusted during a segment.  Lightweight keys
    retain sequence/time history for append and truncation checks; vector data
    are retained only for the requested tail.  A failed append changes no
    reader state.
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
        self._geometry = None
        self._read_frames_total = 0
        self._skipped_decode_frames_total = 0
        self._cache_hits_total = 0
        self._new_source_frames_count = 0
        self._records = ()

    @property
    def diagnostics(self):
        return dict(read_frames_total=self._read_frames_total,
                    skipped_decode_frames_total=self._skipped_decode_frames_total,
                    cache_hits_total=self._cache_hits_total,
                    retained_frames=len(self._tail),
                    retained_bytes=sum(_frame_bytes(frame) for frame in self._tail.values()),
                    source_frame_count=len(self._metadata))

    @property
    def records(self):
        return self._records

    @property
    def history_keys(self):
        return tuple(self._metadata)

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
        segment = meta['segment_id']
        reset = self._segment is not None and segment != self._segment
        old_metadata = OrderedDict() if reset else self._metadata
        old_tail = OrderedDict() if reset else self._tail
        self._validate_append(records, old_metadata)

        # Work in copies.  No bad or half-written publication changes state.
        candidate_metadata = OrderedDict(old_metadata)
        candidate_tail = OrderedDict(old_tail)
        metadata_by_sequence = {key[1]: key for key in candidate_metadata}
        candidate_geometry = None if reset else self._geometry
        decoded = 0
        new_records = [record for record in records if record['sequence'] not in metadata_by_sequence]
        skip_count = max(0, len(new_records) - self.retain_frames)
        skipped_records = new_records[:skip_count]
        if skipped_records and any(record.get('time_s') is None for record in skipped_records):
            skipped_records = []
            skip_count = 0
        for record in skipped_records:
            key = (segment, record['sequence'], self._record_time(record))
            if candidate_metadata and key[2] <= next(reversed(candidate_metadata))[2]:
                raise ValueError('replay times must be strictly increasing')
            candidate_metadata[key] = None
            metadata_by_sequence[record['sequence']] = key
        append_meta = dict(meta)
        append_meta['frames'] = new_records[skip_count:]
        new_frames = (() if not append_meta['frames'] else
                      _iter_ovf_manifest(path, append_meta, raw, cumulative_limit=False,
                                         verify_hashes=False))
        for frame in new_frames:
            key = (segment, frame.sequence, frame.sim_time_s)
            if candidate_metadata and key[2] <= next(reversed(candidate_metadata))[2]:
                raise ValueError('replay times must be strictly increasing')
            geometry = self._frame_geometry(frame)
            if candidate_geometry is None:
                candidate_geometry = geometry
            elif geometry != candidate_geometry:
                raise ValueError('OVF mesh, selected layer, or material mask changes within a segment')
            # Keep a strict N-frame working set while reading a large append.
            # A too-large requested tail fails; it is never silently shortened.
            if len(candidate_tail) >= self.retain_frames:
                candidate_tail.popitem(last=False)
            candidate_metadata[key] = None
            metadata_by_sequence[frame.sequence] = key
            candidate_tail[key] = frame
            decoded += 1

        if sum(_frame_bytes(frame) for frame in candidate_tail.values()) > self.max_bytes:
            raise ValueError('requested retained OVF tail exceeds memory limit')
        if len(records) >= 2 and len(candidate_tail) < 2:
            raise ValueError('retained OVF tail must contain at least two frames')
        self._segment, self._geometry = segment, candidate_geometry
        self._metadata, self._tail = candidate_metadata, candidate_tail
        self._read_frames_total += decoded
        self._skipped_decode_frames_total += skip_count
        self._cache_hits_total += len(records) - len(new_records)
        self._new_source_frames_count = len(new_records)
        self._records = tuple((key[0], key[1], key[2], '') for key in candidate_metadata)
        return FieldReplay(tuple(candidate_tail.values()), '')

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
    def _frame_geometry(frame):
        return (frame.vectors.shape, frame.dx_m, frame.dy_m, frame.origin_m,
                frame.mask.shape)

    @staticmethod
    def _validate_append(records, old):
        """Reject a same-segment history retreat without auditing old records."""
        if not old:
            return
        published = {record['sequence'] for record in records}
        accepted_sequences = {key[1] for key in old}
        if not accepted_sequences.issubset(published):
            raise ValueError('published history was truncated within a segment')
        last_sequence = next(reversed(old))[1]
        if any(record['sequence'] not in accepted_sequences and record['sequence'] <= last_sequence
               for record in records):
            raise ValueError('new sequence must append after accepted history')
