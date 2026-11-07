"""The one place Signal touches the orchid library. Currently a skeleton.

orchid is not set up yet, so every name below is a placeholder that raises
OrchidNotAvailable when called. A placeholder never returns a value, because
every number must come from real code.

Signal code imports these names from here, never from orchid directly:

    from orchid_api import map_match

When orchid is set up, replace the placeholders with real imports in this file
only, delete tests/test_orchid_api.py, and confirm the names and signatures with
scripts/check_orchid.py. Placeholders take any arguments because the real
signatures are unknown until then.

This file is deliberately not called orchid.py: that would shadow the real
library and make check_orchid.py report a false pass.
"""

__all__ = [
    "OrchidNotAvailable",
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


class OrchidNotAvailable(NotImplementedError):
    """Raised when code calls an orchid entry point before orchid is set up."""


def _placeholder(name):
    """Return a callable standing in for an orchid function or class."""

    def placeholder(*args, **kwargs):
        raise OrchidNotAvailable(f"{name} needs the real orchid library")

    placeholder.__name__ = name
    return placeholder


# street_network
RoadGraph = _placeholder("RoadGraph")
clean_network = _placeholder("clean_network")
EdgeIndex = _placeholder("EdgeIndex")
map_match = _placeholder("map_match")
snap_point = _placeholder("snap_point")

# core/ml/data_splitting
GroupKFold = _placeholder("GroupKFold")
TimeSeriesSplit = _placeholder("TimeSeriesSplit")

# core/ml/ensembles
GradientBoostingRegressor = _placeholder("GradientBoostingRegressor")

# traffic_assignment
user_equilibrium = _placeholder("user_equilibrium")
