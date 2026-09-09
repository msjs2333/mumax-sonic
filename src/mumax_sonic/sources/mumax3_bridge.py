"""Poll a MuMax3 OVF output series and publish a live manifest."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any
from types import SimpleNamespace

import numpy as np

from .ovf import probe_ovf, read_ovf


MAX_FRAMES = 4096
_SEQUENCE = re.compile(r"^(.*?)(\d+)$")


class _IncompleteOVF(ValueError):
    pass


def _mask_info(path: Path):
    raw = path.read_bytes()
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError("material mask exceeds limit")
    import io
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("material mask requires NPY version 1 or 2")
    if dtype != np.bool_ or len(shape) not in (2, 3) or any(n <= 0 for n in shape):
        raise ValueError("material mask must be a boolean NPY array")
    count = math.prod(shape)
    if count > 64 * 1024 * 1024 or len(raw) - stream.tell() != count:
        raise ValueError("material mask payload size disagrees with header")
    # Decode once so malformed NPY payloads and the declared array are checked.
    np.frombuffer(raw, dtype=np.bool_, count=count, offset=stream.tell()).reshape(
        shape, order="F" if fortran else "C"
    )
    return {"file": str(path.resolve())}, shape


class MuMax3Bridge:
    """Follow one numeric OVF series and publish a growing manifest.

    A bridge has one writer lock for its lifetime.  It never modifies source
    OVF files or an already-existing manifest.
    """

    def __init__(self, directory, manifest_path, *, entity_id, segment_id,
                 time_kind="dynamics", origin="simulation", all_material=False,
                 mask_path=None, z_index=None, pattern="m[0-9]*.ovf"):
        self.directory = Path(directory).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        if self.manifest_path.is_relative_to(self.directory):
            raise ValueError("manifest must be outside the MuMax source directory")
        if not isinstance(entity_id, str) or not entity_id.strip() or not isinstance(segment_id, str) or not segment_id.strip():
            raise ValueError("entity_id and segment_id are required")
        if time_kind not in ("dynamics", "static", "relaxation"):
            raise ValueError("invalid time_kind")
        if origin not in ("simulation", "synthetic", "unknown"):
            raise ValueError("invalid origin")
        if bool(all_material) == (mask_path is not None):
            raise ValueError("exactly one of all_material or mask_path is required")
        if z_index is not None and (type(z_index) is not int or z_index < 0):
            raise ValueError("z_index must be a nonnegative integer")
        if "/" in pattern or "\\" in pattern or "**" in pattern:
            raise ValueError("pattern must select files directly in the source directory")
        self.entity_id, self.segment_id = entity_id, segment_id
        self.time_kind, self.origin = time_kind, origin
        self.z_index, self.pattern = z_index, pattern
        self._mask, self._mask_shape = ("all", None) if all_material else _mask_info(Path(mask_path).resolve())
        self._manifest_created = False
        self._seen: dict[Path, tuple[int, int, Any]] = {}
        self._stable: dict[Path, tuple[int, int, int]] = {}
        self._published: dict[Path, bool] = {}
        self._published_count = 0
        self._geometry = None
        self._unit = None
        self._closed = False
        self._lock_path = Path(str(self.manifest_path) + ".lock")
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._lock = self._lock_path.open("x", encoding="ascii")
            self._lock.write(str(os.getpid()))
            self._lock.flush()
        except FileExistsError as exc:
            raise ValueError("manifest writer lock is already held") from exc
        if self.manifest_path.exists():
            self.close()
            raise FileExistsError(f"manifest already exists: {self.manifest_path}")

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self._lock.close()
            finally:
                try:
                    self._lock_path.unlink()
                except FileNotFoundError:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def _labels(labels):
        normalized = [re.sub(r"[^a-z0-9]", "", x.lower()) for x in labels]
        if normalized != ["mx", "my", "mz"]:
            raise ValueError("OVF labels must clearly declare m_x, m_y, m_z")

    def _files(self):
        paths = sorted((p for p in self.directory.glob(self.pattern)), key=lambda p: p.name)
        indexed = []
        for path in paths:
            match = _SEQUENCE.fullmatch(path.stem)
            if not match:
                raise ValueError("every matched OVF filename must end in a numeric sequence")
            indexed.append((int(match.group(2)), path))
        if len({n for n, _ in indexed}) != len(indexed):
            raise ValueError("duplicate numeric OVF sequence")
        return sorted(indexed)

    def _validate_frame(self, path, stat):
        signature = (stat.st_size, stat.st_mtime_ns)
        cached = self._seen.get(path)
        if cached and cached[:2] == signature:
            return cached[2]
        try:
            # Binary MuMax output is validated from its header, check value,
            # declared payload extent, and trailers.  It avoids decoding a
            # transient full magnetization array on every poll.
            field = probe_ovf(path)
        except ValueError as exc:
            with path.open('rb') as stream:
                stream.seek(max(0, stat.st_size-256))
                tail = stream.read(256).lower()
            if b'end: segment' not in tail:
                raise _IncompleteOVF('waiting for complete OVF end marker') from exc
            raise
        self._labels(field.labels)
        if tuple(field.units) not in (("1",) * 3, ("A/m",) * 3):
            raise ValueError("OVF value units must be all 1 or all A/m")
        if field.time_s is None:
            raise ValueError("missing physical time in OVF header")
        if not math.isfinite(field.time_s) or field.time_s < 0:
            raise ValueError("physical time must be finite and nonnegative")
        if self.z_index is None and field.shape[0] != 1:
            raise ValueError("multilayer OVF requires explicit z_index")
        if self.z_index is not None and self.z_index >= field.shape[0]:
            raise ValueError("z_index outside OVF mesh")
        if self._mask_shape is not None and tuple(self._mask_shape) not in (field.shape[:3], field.shape[1:3]):
            raise ValueError('material mask shape does not match OVF mesh')
        geometry = (field.shape, tuple(field.step_m), tuple(field.origin_m), tuple(field.labels))
        unit = field.units[0]
        if self._geometry is None:
            self._geometry, self._unit = geometry, unit
        elif geometry != self._geometry or unit != self._unit:
            raise ValueError("OVF mesh, labels, or units change within one segment")
        # Never retain raw field arrays in the publication bridge.
        info = SimpleNamespace(time_s=field.time_s, units=field.units)
        self._seen[path] = (*signature, info)
        return info

    def _manifest(self, records):
        frames = []
        for sequence, path, field in records:
            frames.append({"file": str(path), "sequence": sequence,
                           "time_s": field.time_s})
        gaps = [[a[0] + 1, b[0] - 1] for a, b in zip(records, records[1:]) if b[0] > a[0] + 1]
        return {"schema_version": 1, "integrity": "unchecked", "entity_id": self.entity_id, "segment_id": self.segment_id,
                "time_kind": self.time_kind, "origin": self.origin,
                "quantity": "magnetization_direction", "value_unit": "A/m" if records and records[0][2].units[0] == "A/m" else "1",
                "components": ["x", "y", "z"], "z_index": self.z_index, "mask": self._mask,
                "frames": frames, "sequence_source": "numeric filename suffix",
                "selection": {"matched": len(records), "imported": len(records), "all_matches_selected": True,
                              "gaps_within_selected": gaps}}

    def _publish(self, meta):
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")
        fd, name = tempfile.mkstemp(prefix=".mumax3-manifest-", suffix=".tmp", dir=self.manifest_path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
            if not self._manifest_created:
                try:
                    os.link(name, self.manifest_path)
                except FileExistsError:
                    raise FileExistsError("manifest appeared during publication")
            else:
                # Windows virus scanners and readers can briefly hold the old
                # manifest open.  Retry only a bounded number of times; a
                # later poll can publish the same accumulated records.
                for attempt in range(3):
                    try:
                        os.replace(name, self.manifest_path)
                        break
                    except PermissionError:
                        if attempt == 2:
                            return False
                        time.sleep(0.01 * (attempt + 1))
            self._manifest_created = True
            return True
        finally:
            try: os.unlink(name)
            except FileNotFoundError: pass

    def poll(self):
        if self._closed:
            return {"state": "invalid", "reason": "bridge is closed", "published_frames": self._published_count, "pending_files": []}
        try:
            indexed = self._files()
            if len(indexed) > MAX_FRAMES:
                raise ValueError("OVF series exceeds 4096 frames")
            # Observe all stat signatures in parallel logical passes, so an
            # already complete N-frame directory needs two polls, not N+1.
            stats = {path: path.stat() for _, path in indexed if path not in self._published}
            unstable = set()
            for _, path in indexed:
                if path in self._published:
                    continue
                stat = stats[path]
                sig = (stat.st_size, stat.st_mtime_ns)
                if path not in self._stable or self._stable[path][:2] != sig:
                    unstable.add(path)
                    self._stable[path] = (*sig, 1)
            pending = []
            records = []
            for sequence, path in indexed:
                if path in self._published:
                    records.append((sequence, path, self._seen[path][2]))
                    continue
                stat = stats[path]
                sig = (stat.st_size, stat.st_mtime_ns)
                if path in unstable:
                    pending.append(str(path)); break
                try:
                    field = self._validate_frame(path, stat)
                except _IncompleteOVF:
                    pending.append(str(path)); break
                if records and field.time_s <= records[-1][2].time_s:
                    raise ValueError('physical times must increase with source sequence')
                records.append((sequence, path, field))
            if not records:
                return {"state": "waiting", "reason": "awaiting complete OVF files", "published_frames": self._published_count, "pending_files": pending}
            if len(records) > self._published_count:
                if self._publish(self._manifest(records)) is False:
                    return {"state": "waiting", "reason": "manifest replace is temporarily busy", "published_frames": self._published_count, "pending_files": []}
                self._published_count = len(records)
                self._published = {p: True for _, p, _ in records}
            return {"state": "waiting" if pending else "current", "reason": "awaiting next complete frame" if pending else None, "published_frames": self._published_count, "pending_files": pending}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return {"state": "invalid", "reason": str(exc), "published_frames": self._published_count, "pending_files": []}
