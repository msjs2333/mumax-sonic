import pytest

from mumax_sonic.model import Observation, Sample


def test_model_rejects_duplicate_ids_and_bad_units():
    one = Observation("same", (0.0, 0.0, 0.0), 0.5, 1)
    with pytest.raises(ValueError):
        Sample(0.0, (one, one))
    with pytest.raises(ValueError):
        Observation("bad", (0.0, 0.0, 0.0), -0.1, 1)
    with pytest.raises(ValueError):
        Observation("bad-sign", (0.0, 0.0, 0.0), 0.1, 0)


def test_sample_schema_time_units_and_quality_are_retained():
    observation = Observation("s", (1e-6, -1e-6, 0.0), 0.2, -1)
    sample = Sample(2e-9, (observation,), sequence=4, validity="stale",
                    source_kind="synthetic", time_kind="static")
    assert sample.sim_time_s == 2e-9
    assert sample.sequence == 4
    assert sample.observations[0].position_m == (1e-6, -1e-6, 0.0)
    assert sample.observations[0].unit == "1"
    assert sample.validity == "stale"
    assert sample.time_kind == "static"
