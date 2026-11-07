"""The orchid skeleton must fail loudly, never return a made-up value.

When the real library is set up, the sub-issue deletes this file along with the
placeholders, and scripts/check_orchid.py becomes the check.
"""

import pytest

import orchid_api

# The entry points from v2 section 2, as listed in the kickoff's Task 0.
ENTRY_POINTS = [
    "RoadGraph",
    "clean_network",
    "EdgeIndex",
    "map_match",
    "snap_point",
    "GroupKFold",
    "TimeSeriesSplit",
    "GradientBoostingRegressor",
    "user_equilibrium",
]


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_placeholder_raises_when_called(name):
    with pytest.raises(orchid_api.OrchidNotAvailable, match=name):
        getattr(orchid_api, name)("any", "arguments", option=1)


def test_skeleton_exposes_only_the_named_entry_points():
    assert sorted(orchid_api.__all__) == sorted(
        ENTRY_POINTS + ["OrchidNotAvailable"]
    )
