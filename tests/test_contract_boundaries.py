import pytest
from mumax_sonic.model import Sample, SonicScene
from mumax_sonic.attention import Attention
from mumax_sonic.mapping import map_sample
from mumax_sonic.sources.synthetic import make_sample


@pytest.mark.parametrize('kwargs', [{'sequence': 0.5}, {'sequence': True}, {'schema_version': 2}])
def test_sample_rejects_ambiguous_sequence_and_unknown_schema(kwargs):
    with pytest.raises(ValueError):
        Sample(0, (), **kwargs)


def test_negative_scene_time_is_not_accepted():
    with pytest.raises(ValueError):
        SonicScene(sim_time_s=-1)


def test_solo_preserves_same_source_gain_and_signs_share_reference():
    sample = make_sample('colocated', 0)
    both = map_sample(sample, Attention())
    positive = map_sample(sample, Attention(), mode='positive')
    gains = {source.sign: source.gain for source in both.sources}
    assert gains[1] == gains[-1] == positive.sources[0].gain
