import numpy as np
import pytest

from mumax_sonic.aggregation import ContributionGrid, aggregate
from mumax_sonic.attention import Attention


def _grid(positive, negative=None, x=None, y=None):
    positive = np.asarray(positive, dtype=float)
    negative = np.zeros_like(positive) if negative is None else np.asarray(negative, dtype=float)
    yy, xx = np.indices(positive.shape, dtype=float)
    return ContributionGrid(
        xx if x is None else np.asarray(x, dtype=float), yy if y is None else np.asarray(y, dtype=float),
        positive, negative, 2e-9, "sample", "q", "1",
    )


def test_signed_raw_contributions_are_conserved_without_cancellation():
    result = aggregate(_grid([[1, 2], [3, 4]], [[4, 3], [2, 1]]), Attention(extent_m=1), budget=4)
    assert result.diagnostic["input_positive"] == 10
    assert result.diagnostic["input_negative"] == 10
    assert result.diagnostic["represented_positive"] == 10
    assert result.diagnostic["represented_negative"] == 10
    assert {item.sign for item in result.observations} == {1, -1}


def test_foreground_background_conserve_mass_and_gain_is_portion_weighted():
    grid = _grid([[1, 9]], x=[[0, 4]], y=[[0, 0]])
    attention = Attention(center=(0, 0), radius=.1, background=.2, extent_m=1)
    result = aggregate(grid, attention, budget=2)
    assert result.diagnostic["foreground_sources"] == 1
    assert result.diagnostic["background_sources"] == 1
    assert sum(item.strength for item in result.observations) == pytest.approx(10)
    background = next(item for item in result.observations if item.source_id.endswith(":background"))
    assert background.attention_weight == pytest.approx(.2)
    assert all(item.strength >= 0 for item in result.observations)


def test_one_voice_cannot_represent_both_signs():
    result = aggregate(_grid([[1]], [[1]]), Attention(extent_m=1), budget=1)
    assert result.observations == ()
    assert result.diagnostic["status"] == "unsupported"
    assert result.diagnostic["reason"] == "budget cannot represent both signs"


def test_zero_grid_is_valid_and_has_no_false_fraction():
    result = aggregate(_grid([[0, 0]]), Attention(extent_m=1), budget=4)
    assert result.observations == ()
    assert result.diagnostic["status"] == "valid"
    assert result.diagnostic["spatial_rms_m"] == 0
    assert result.diagnostic["foreground_spatial_rms_m"] == 0


def test_ids_are_deterministic_and_roi_changes_do_not_change_raw_total():
    grid = _grid([[1, 1, 1, 1]], x=[[0, 1, 2, 3]], y=[[0, 0, 0, 0]])
    first = aggregate(grid, Attention(center=(0, 0), radius=.2, extent_m=1), budget=4)
    again = aggregate(grid, Attention(center=(0, 0), radius=.2, extent_m=1), budget=4)
    moved = aggregate(grid, Attention(center=(1, 0), radius=.2, extent_m=1), budget=4)
    assert [item.source_id for item in first.observations] == [item.source_id for item in again.observations]
    assert first.diagnostic["represented_positive"] == moved.diagnostic["represented_positive"] == 4


def test_background_gain_uses_site_weights_not_its_centroid_location():
    # Most background mass is far outside the ROI, but a small feathered portion
    # shifts its centroid inward.  Its gain must still be dominated by outside mass.
    grid = _grid([[8, 2]], x=[[.8, .18]], y=[[0, 0]])
    attention = Attention(center=(0, 0), radius=.2, background=.1, extent_m=1)
    result = aggregate(grid, attention, budget=2)
    background = next(item for item in result.observations if item.source_id.endswith(":background"))
    assert background.position_m[0] < .8
    assert background.attention_weight < .3


def test_grid_rejects_nonfinite_or_negative_values():
    with pytest.raises(ValueError):
        _grid([[np.nan]])
    with pytest.raises(ValueError):
        _grid([[-1]])
