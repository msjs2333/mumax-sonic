import math

from mumax_sonic.attention import Attention, select_sources
from mumax_sonic.mapping import map_sample
from mumax_sonic.model import Observation, Sample
from mumax_sonic.session import Timeline, Transport
from mumax_sonic.sources.synthetic import make_sample


def test_roi_reserves_weak_inner_source_and_both_signs():
    sample = make_sample("roi_challenge", 0.0)
    selected = select_sources(sample.observations, Attention(), budget=4)
    ids = {o.source_id for o in selected}
    assert "roi-inner-weak" in ids
    assert {o.sign for o in selected} == {-1, 1}


def test_solo_changes_output_selection_only_and_positions_are_unit_vectors():
    sample = make_sample("signed_pair", 0.0)
    raw = tuple((o.source_id, o.strength, o.sign) for o in sample.observations)
    both = map_sample(sample, Attention(), mode="both", budget=2, master_gain=0.2)
    positive = map_sample(sample, Attention(), mode="positive", budget=2, master_gain=0.2)
    assert tuple((o.source_id, o.strength, o.sign) for o in sample.observations) == raw
    assert {s.sign for s in both.sources} == {-1, 1}
    assert {s.sign for s in positive.sources} == {1}
    for source in both.sources:
        assert math.isclose(math.sqrt(sum(x * x for x in source.position)), 1.0)
        assert source.gain >= 0


def test_quality_failure_and_valid_zero_are_distinct():
    stale = map_sample(make_sample("stale", 0), Attention())
    invalid = map_sample(make_sample("invalid", 0), Attention())
    zero = Sample(0.0, (Observation("zero", (0, 0, 0), 0.0, 1),))
    zero_scene = map_sample(zero, Attention())
    assert stale.validity == "stale" and invalid.validity == "invalid"
    assert not stale.sources and not invalid.sources
    assert zero_scene.validity == "valid" and not zero_scene.sources


def test_pause_freezes_time_but_static_listen_remains_audible():
    transport = Transport(sim_time_s=3e-9, playing=False, static_listen=True)
    transport.advance(2.0)
    assert transport.sim_time_s == 3e-9
    assert transport.audible
    transport.playing = True
    transport.advance(2.0)
    assert transport.sim_time_s == 5e-9


def test_timeline_rejects_reordering_and_marks_gaps():
    timeline = Timeline()
    first = make_sample("moving", 0.0, sequence=0)
    next_sample = make_sample("moving", 1e-10, sequence=1)
    gap = make_sample("moving", 3e-10, sequence=3)
    old = make_sample("moving", 2e-10, sequence=2)
    assert timeline.accept(first) == "valid"
    assert timeline.accept(next_sample) == "valid"
    assert timeline.accept(gap) == "warming_up"
    assert timeline.accept(old) == "invalid"
