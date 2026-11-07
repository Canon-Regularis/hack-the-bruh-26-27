"""A fake bus with a known speed and one dwell, for testing. Output is synthetic.

The bus drives north along five 300 m edges at 18 km/h, and stops for 60 s halfway
along the third. Matching does not exist yet, so edge distances are exact, and
GPS noise is added to the positions only.
"""

import random
from datetime import datetime, timedelta, timezone

from matching.table import PingPair

TRUE_SPEED_KMH = 18.0
STOP_EDGE = "edge_3"

SPEED_M_PER_S = TRUE_SPEED_KMH / 3.6
EDGE_LENGTH_M = 300.0
EDGE_COUNT = 5
STOP_AT_M = 750.0
DWELL_S = 60.0
PING_INTERVAL_S = 30
START_POSITION = (55.95, -3.19)
METRES_PER_DEGREE_LAT = 111_194.9
START_TIME = datetime(2026, 11, 7, 18, 0, 0, tzinfo=timezone.utc)


def distance_at(seconds):
    """Return the metres driven after this many seconds, allowing for the dwell."""
    arrive_at_stop = STOP_AT_M / SPEED_M_PER_S
    if seconds <= arrive_at_stop:
        return seconds * SPEED_M_PER_S
    if seconds <= arrive_at_stop + DWELL_S:
        return STOP_AT_M
    return (seconds - DWELL_S) * SPEED_M_PER_S


def position_at(distance, noise_m, rng):
    """Return (lat, lon) for a distance along the path, with optional GPS noise."""
    lat = START_POSITION[0] + (distance + rng.gauss(0, noise_m)) / METRES_PER_DEGREE_LAT
    lon = START_POSITION[1] + rng.gauss(0, noise_m) / METRES_PER_DEGREE_LAT
    return lat, lon


def edges_between(start_m, end_m):
    """Return (edge_id, metres) for the stretch of path between two distances."""
    edges = []
    for index in range(EDGE_COUNT):
        edge_start = index * EDGE_LENGTH_M
        overlap = min(end_m, edge_start + EDGE_LENGTH_M) - max(start_m, edge_start)
        if overlap > 0:
            edges.append((f"edge_{index + 1}", overlap))
    if edges:
        return edges
    # A bus that did not move lists the edge it is standing on.
    return [(f"edge_{int(start_m // EDGE_LENGTH_M) + 1}", 0.0)]


def synthetic_bus(first_ping_s=0, noise_m=0.0, seed=1):
    """Return (ping pairs, stops, street names) for one run of the fake bus."""
    rng = random.Random(seed)
    trip_seconds = EDGE_COUNT * EDGE_LENGTH_M / SPEED_M_PER_S + DWELL_S
    times = range(first_ping_s, int(trip_seconds) + 1, PING_INTERVAL_S)
    pings = [(t, distance_at(t)) for t in times]
    positions = [position_at(distance, noise_m, rng) for _, distance in pings]

    pairs = [
        PingPair(
            vehicle_id="synthetic_bus_1",
            start_time_utc=START_TIME + timedelta(seconds=start[0]),
            end_time_utc=START_TIME + timedelta(seconds=end[0]),
            start_position=start_position,
            end_position=end_position,
            edges=edges_between(start[1], end[1]),
            source="synthetic",
        )
        for start, end, start_position, end_position in zip(
            pings, pings[1:], positions, positions[1:]
        )
    ]
    stops = [position_at(STOP_AT_M, 0.0, rng)]
    street_names = {
        f"edge_{index + 1}": f"Synthetic Street {index + 1}"
        for index in range(EDGE_COUNT)
    }
    return pairs, stops, street_names
