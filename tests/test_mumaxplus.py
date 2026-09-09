import json
import threading

import numpy as np
import pytest

from mumax_sonic.sources.mumaxplus import MuMaxPlusSampler, probe_mumaxplus


class _Solver:
    def __init__(self, time=2e-9): self.time = time


class _World:
    def __init__(self):
        self.timesolver = _Solver()
        self.cellsize = (2e-9, 3e-9, 4e-9)


class _Quantity:
    def __init__(self, values, world, advance=False, name="m"):
        self.values, self.world, self.advance, self.name = values, world, advance, name
    def eval(self):
        if self.advance: self.world.timesolver.time += 1e-12
        return self.values


class _Magnet:
    def __init__(self, values, world, geometry=None):
        self.magnetization = _Quantity(values, world)
        nz, ny, nx = values.shape[1:]
        z, y, x = np.mgrid[:nz, :ny, :nx]
        self.meshgrid = np.stack((1e-8 + x * world.cellsize[0], 2e-8 + y * world.cellsize[1],
                                  3e-8 + z * world.cellsize[2]))
        self.geometry = geometry


def _values(nz=1):
    return np.arange(3 * nz * 3 * 4, dtype=np.float32).reshape(3, nz, 3, 4)


def test_capture_preserves_raw_xyz_coordinates_mask_layer_and_provenance():
    world, values = _World(), _values(2)
    geometry = np.array([[[True, False, True, True]] * 3, [[False, True, True, False]] * 3])
    magnet = _Magnet(values, world, geometry)
    sampler = MuMaxPlusSampler(world, magnet, entity_id="fm", segment_id="run", z_index=1)
    frame = sampler.sample(4)
    assert frame.source_kind == "live" and frame.sim_time_s == 2e-9
    assert frame.origin_m == pytest.approx((1e-8, 2e-8, 3e-8 + 4e-9))
    assert frame.dx_m == 2e-9 and frame.dy_m == 3e-9
    assert np.array_equal(frame.vectors, np.moveaxis(values[:, 1], 0, -1))
    assert np.array_equal(frame.mask, geometry[1])
    assert not frame.vectors.flags.writeable and not frame.mask.flags.writeable
    info = json.loads(frame.provenance)
    assert info["quantity_semantics"] == "magnetization_direction"
    assert info["raw_shape"] == [3, 2, 3, 4] and info["z_index"] == 1


def test_layer_mask_and_explicit_mask_requirements_and_custom_vector_shape():
    world, values = _World(), _values()
    magnet = _Magnet(values, world, None)
    with pytest.raises(ValueError, match="explicit mask"):
        MuMaxPlusSampler(world, magnet, entity_id="m", segment_id="s").sample(0)
    mask = np.ones((3, 4), dtype=bool)
    sampler = MuMaxPlusSampler(world, magnet, entity_id="m", segment_id="s", mask=mask,
                               quantity=_Quantity(values, world, name="n"),
                               quantity_semantics="order_parameter_direction")
    assert sampler.sample(0).mask.shape == (3, 4)
    sampler.quantity = _Quantity(np.ones((2, 1, 3, 4)), world)
    with pytest.raises(ValueError, match="shape"):
        sampler.sample(1)


def test_snapshot_rejects_solver_advance_sequence_reuse_time_reversal_and_other_thread():
    world, values = _World(), _values()
    magnet = _Magnet(values, world, np.ones((1, 3, 4), dtype=bool))
    sampler = MuMaxPlusSampler(world, magnet, entity_id="m", segment_id="s")
    sampler.quantity = _Quantity(values, world, advance=True)
    with pytest.raises(RuntimeError, match="advanced"):
        sampler.sample(0)
    sampler.quantity = magnet.magnetization
    sampler.sample(0)
    with pytest.raises(ValueError, match="increase"):
        sampler.sample(0)
    world.timesolver.time = 1e-9
    with pytest.raises(ValueError, match="must not decrease"):
        sampler.sample(1)
    world.timesolver.time = 3e-9
    failures = []
    thread = threading.Thread(target=lambda: failures.append(pytest.raises(RuntimeError, sampler.sample, 1)))
    thread.start(); thread.join()
    assert failures


def test_probe_only_reports_import_capabilities_without_gpu_claim():
    report = probe_mumaxplus()
    assert set(report) == {"package_available", "version", "import_error", "capabilities"}
    if report["package_available"]:
        assert report["capabilities"]["gpu_usable"] is None


def test_rejects_wrong_units_and_nonregular_coordinates():
    world = _World(); values = _values()
    magnet = _Magnet(values, world, np.ones((1,3,4), dtype=bool))
    magnet.magnetization.unit = 'T'
    with pytest.raises(ValueError, match='units'):
        MuMaxPlusSampler(world, magnet, entity_id='m', segment_id='s')
    magnet.magnetization.unit = '1'
    magnet.meshgrid[0,0,1,1] += world.cellsize[0]
    with pytest.raises(ValueError, match='regular XY'):
        MuMaxPlusSampler(world, magnet, entity_id='m', segment_id='s').sample(0)
