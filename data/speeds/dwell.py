"""Decide which ping pairs are dwell at a stop. The rule is Q3 in DECISIONS.md.

Dwell pairs are kept, but their time is stored apart from moving time. A bus
standing still anywhere else is queuing, which is the delay we want to measure,
so that time counts as moving time and stays in the speed.
"""

from math import asin, cos, radians, sin, sqrt

STOP_RADIUS_M = 20.0
STATIONARY_M = 10.0
EARTH_RADIUS_M = 6_371_000.0


def distance_m(a, b):
    """Return the great-circle distance in metres between two (lat, lon) points."""
    lat_a, lon_a, lat_b, lon_b = map(radians, (*a, *b))
    half_chord = (
        sin((lat_b - lat_a) / 2) ** 2
        + cos(lat_a) * cos(lat_b) * sin((lon_b - lon_a) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * asin(sqrt(half_chord))


def is_dwell(pair, stops):
    """Return True if the bus moved under 10 m and was within 20 m of a stop."""
    if distance_m(pair.start_position, pair.end_position) >= STATIONARY_M:
        return False
    return any(
        distance_m(position, stop) <= STOP_RADIUS_M
        for stop in stops
        for position in (pair.start_position, pair.end_position)
    )
