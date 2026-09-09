"""Capture-only adapter for a caller-owned MuMax+ magnet.

This module deliberately does not construct a world, advance a solver, or
derive a net antiferromagnetic field.  It snapshots the explicitly supplied
magnetization (or another declared three-component quantity) on its owner
thread.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import threading
import time
from typing import Any

import numpy as np

from ..fields import FieldFrame


_SEMANTICS = {"magnetization_direction", "order_parameter_direction"}


def probe_mumaxplus() -> dict:
    """Report import-level MuMax+ capture capabilities, without creating a world."""
    try:
        package = importlib.import_module("mumaxplus")
    except Exception as exc:
        return {"package_available": False, "version": None, "import_error": str(exc),
                "capabilities": {}}
    classes = ("World", "Ferromagnet", "Antiferromagnet", "NcAfm", "Magnet", "FieldQuantity")
    magnet = getattr(package, "Magnet", None)
    world = getattr(package, "World", None)
    quantity = getattr(package, "FieldQuantity", None)
    return {
        "package_available": True,
        "version": getattr(package, "__version__", None),
        "import_error": None,
        "capabilities": {
            "classes": {name: hasattr(package, name) for name in classes},
            "capture_api": {
                "field_quantity_eval": callable(getattr(quantity, "eval", None)),
                "magnet_meshgrid": isinstance(getattr(magnet, "meshgrid", None), property),
                "magnet_geometry": isinstance(getattr(magnet, "geometry", None), property),
                "world_cellsize": isinstance(getattr(world, "cellsize", None), property),
                "timesolver_time": isinstance(getattr(getattr(package, "TimeSolver", None), "time", None), property),
            },
            "gpu_usable": None,
        },
    }


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _layer_mask(mask: Any, shape: tuple[int, int, int], z_index: int) -> np.ndarray:
    result = np.asarray(mask)
    if result.dtype != np.bool_:
        raise ValueError("material mask must be boolean")
    if result.shape == shape:
        return np.array(result[z_index], dtype=bool, copy=True)
    if shape[0] == 1 and result.shape == shape[1:]:
        return np.array(result, dtype=bool, copy=True)
    raise ValueError("material mask must match (nz, ny, nx), or (ny, nx) for one layer")


class MuMaxPlusSampler:
    """Snapshot one explicit MuMax+ vector quantity into immutable ``FieldFrame``s."""

    def __init__(self, world: Any, magnet: Any, *, entity_id: str, segment_id: str,
                 quantity: Any = None, quantity_semantics: str = "magnetization_direction",
                 mask: np.ndarray | None = None, time_kind: str = "dynamics", z_index: int | None = None):
        if not isinstance(entity_id, str) or not entity_id or not isinstance(segment_id, str) or not segment_id:
            raise ValueError("entity_id and segment_id are required")
        if quantity_semantics not in _SEMANTICS:
            raise ValueError("quantity_semantics must declare magnetization or order-parameter direction")
        if time_kind not in {"static", "dynamics", "relaxation"}:
            raise ValueError("unknown time kind")
        self.world = world
        self.magnet = magnet
        self.quantity = magnet.magnetization if quantity is None else quantity
        unit = getattr(self.quantity, 'unit', '1')
        if unit not in ({'1', '', 'A/m'} if quantity_semantics == 'magnetization_direction' else {'1', ''}):
            raise ValueError('quantity units do not describe the declared magnetic direction')
        if not callable(getattr(self.quantity, "eval", None)):
            raise ValueError("quantity must provide eval()")
        self.entity_id = entity_id
        self.segment_id = segment_id
        self.quantity_semantics = quantity_semantics
        self.time_kind = time_kind
        self.z_index = z_index
        self._explicit_mask = None if mask is None else np.array(mask, copy=True)
        self._owner_thread_id = threading.get_ident()
        self._last_sequence: int | None = None
        self._last_time_s: float | None = None
        self.last_capture_ms: float | None = None

    def _geometry_mask(self, shape: tuple[int, int, int], z_index: int) -> np.ndarray:
        if self._explicit_mask is not None:
            return _layer_mask(self._explicit_mask, shape, z_index)
        try:
            geometry = self.magnet.geometry
        except (AttributeError, NotImplementedError):
            geometry = None
        if geometry is None:
            raise ValueError("explicit mask is required when magnet geometry is unavailable")
        return _layer_mask(geometry, shape, z_index)

    def sample(self, sequence: int) -> FieldFrame:
        """Capture without moving the solver; reject concurrent solver advancement."""
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("MuMax+ capture must run on its constructor owner thread")
        if type(sequence) is not int or sequence < 0:
            raise ValueError("sequence must be a nonnegative integer")
        if self._last_sequence is not None and sequence <= self._last_sequence:
            raise ValueError("sequence must increase within a segment")
        started = time.perf_counter()
        before_time = float(self.world.timesolver.time)
        raw = np.array(self.quantity.eval(), copy=True)
        after_time = float(self.world.timesolver.time)
        if before_time != after_time:
            raise RuntimeError("MuMax+ solver advanced during capture")
        if raw.ndim != 4 or raw.shape[0] != 3:
            raise ValueError("quantity eval() must return shape (3, nz, ny, nx)")
        _, nz, ny, nx = raw.shape
        if ny < 3 or nx < 3:
            raise ValueError("MuMax+ XY layer must be at least 3 by 3")
        if nz > 1 and self.z_index is None:
            raise ValueError("z_index is required for a multi-layer MuMax+ magnet")
        z_index = 0 if self.z_index is None else self.z_index
        if type(z_index) is not int or not 0 <= z_index < nz:
            raise ValueError("z_index is outside the MuMax+ vector field")
        if self._last_time_s is not None and before_time < self._last_time_s:
            raise ValueError("physical time must not decrease within a segment")
        meshgrid = np.asarray(self.magnet.meshgrid)
        if meshgrid.shape != (3, nz, ny, nx):
            raise ValueError("magnet.meshgrid must have shape (3, nz, ny, nx)")
        cellsize = tuple(float(value) for value in self.world.cellsize)
        if len(cellsize) != 3 or cellsize[0] <= 0 or cellsize[1] <= 0:
            raise ValueError("world.cellsize must provide positive XY spacing")
        material_mask = self._geometry_mask((nz, ny, nx), z_index)
        vectors = np.moveaxis(raw[:, z_index, :, :], 0, -1)
        coordinates = meshgrid[:, z_index, :, :]
        origin = tuple(float(coordinates[axis, 0, 0]) for axis in range(3))
        xx, yy = np.meshgrid(origin[0]+np.arange(nx)*cellsize[0], origin[1]+np.arange(ny)*cellsize[1])
        for actual, expected in zip(coordinates, (xx, yy, np.full((ny,nx), origin[2]))):
            if not np.allclose(actual, expected, rtol=2e-6, atol=min(cellsize[:2])*1e-5):
                raise ValueError('only regular XY mesh coordinates are supported')
        quantity_name = getattr(self.quantity, "name", None) or (
            "magnetization" if self.quantity is getattr(self.magnet, "magnetization", None) else "custom_vector_quantity"
        )
        self.last_capture_ms = (time.perf_counter() - started) * 1000.0
        provenance = json.dumps({
            "backend": "mumaxplus", "version": probe_mumaxplus()["version"],
            "entity_id": self.entity_id, "segment_id": self.segment_id,
            "quantity": quantity_name, "quantity_semantics": self.quantity_semantics,
            "raw_shape": list(raw.shape), "z_index": z_index,
            "host_bytes_sha256": _sha256(raw), "coordinates_sha256": _sha256(coordinates),
            "mask_sha256": _sha256(material_mask), "sim_time_s": before_time,
            "transfer_and_geometry_ms": self.last_capture_ms,
        }, sort_keys=True)
        frame = FieldFrame(vectors, cellsize[0], cellsize[1], before_time, origin, material_mask,
                           self.entity_id, "live", self.segment_id, sequence, self.time_kind, provenance)
        self._last_sequence = sequence
        self._last_time_s = before_time
        self.last_capture_ms = (time.perf_counter() - started) * 1000.0
        return frame
