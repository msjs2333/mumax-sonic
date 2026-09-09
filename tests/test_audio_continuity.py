from dataclasses import replace
from mumax_sonic.audio.continuity import AdaptiveVoiceTracker
from mumax_sonic.model import SonicScene, SonicSource


def test_renaming_or_swapping_paths_preserves_nearest_voice_and_parameters():
    tracker = AdaptiveVoiceTracker()
    initial = SonicScene((SonicSource('adaptive:positive:focus:L', (-.5, 0, -1), .2),
                          SonicSource('adaptive:positive:focus:R', (.5, 0, -1), .3)))
    first = tracker.map(initial)
    renamed = replace(initial, sources=(replace(initial.sources[1], source_id='adaptive:positive:focus:L'),
                                       replace(initial.sources[0], source_id='adaptive:positive:focus:R')))
    second = tracker.map(renamed)
    assert [s.source_id for s in second.sources] == [s.source_id for s in reversed(first.sources)]
    assert [(s.position, s.gain, s.sign) for s in second.sources] == [(s.position, s.gain, s.sign) for s in renamed.sources]


def test_roles_signs_and_quality_are_separate_and_ids_never_reused_after_reset():
    tracker = AdaptiveVoiceTracker()
    source = SonicSource('adaptive:positive:focus:root', (0, 0, -1), .2)
    first = tracker.map(SonicScene((source,))).sources[0].source_id
    for changed in (replace(source, sign=-1), replace(source, source_id='adaptive:positive:background')):
        assert tracker.map(SonicScene((changed,))).sources[0].source_id != first
    stale = SonicScene(validity='stale')
    assert tracker.map(stale) is stale
    assert tracker.map(SonicScene((source,))).sources[0].source_id != first
    fixed = SonicScene((replace(source, source_id='fixed'),))
    assert tracker.map(fixed) is fixed
