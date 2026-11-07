"""Map matching (ALG-MVP-3): turn the collector's raw ping rows into PingPair rows.

PingPair is also the hand-off format to the speeds stage. match_pings parses
and deduplicates the collector's rows, splits them into per-vehicle trips,
then runs street_network.map_match (Newson-Krumm HMM, solved exactly by
Viterbi) once per trip. Each consecutive pair of matched points becomes the
edges actually driven between them: the remaining stretch of the first edge,
any edges fully crossed in between (via the library's own public
shortest_path, the same formula its internal _route_distance uses), and the
entered stretch of the second edge. A pair with no route between its matched
points -- map_match's own "break" case -- is dropped rather than given a
made-up distance.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from engine.orchid import street_network as sn

from matching.trips import pings_from_rows, split_trips

SOURCES = {"observed", "synthetic"}
MATCH_MODE = "bus"


class MatchingNotAvailable(NotImplementedError):
    """Kept only so scripts/build_edge_speeds.py's import and its monkeypatched
    "not built yet" test keep working. Real matching never raises this."""


@dataclass(frozen=True)
class PingPair:
    """Two consecutive pings from one bus, after map matching.

    Times are the GPS fix times in UTC, not the time we polled for them.
    Positions are the raw GPS (latitude, longitude), not matched positions. Edges
    are (edge_id, metres) in the order driven, from matching. A bus standing still
    lists the edge it is on, with 0 metres.
    """

    vehicle_id: str
    start_time_utc: datetime
    end_time_utc: datetime
    start_position: tuple[float, float]
    end_position: tuple[float, float]
    edges: list[tuple[str, float]]
    source: str

    def __post_init__(self):
        # A time with no timezone has no offset, so this rejects those too.
        for moment in (self.start_time_utc, self.end_time_utc):
            if moment.utcoffset() != timedelta(0):
                raise ValueError("ping times must be in UTC")
        if self.end_time_utc <= self.start_time_utc:
            raise ValueError("a ping pair must end after it starts")
        if not self.edges:
            raise ValueError("a ping pair needs at least one edge")
        if any(metres < 0 for _, metres in self.edges):
            raise ValueError("edge distances cannot be negative")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {sorted(SOURCES)}")

    @property
    def seconds(self):
        return (self.end_time_utc - self.start_time_utc).total_seconds()


def _edge_metres(graph, c0, c1):
    """Return (str(edge id), metres) for every edge driven between two candidates.

    None if there is no route between them (map_match's "break" case).
    """
    ea = graph.edge(c0.eid)
    eb = graph.edge(c1.eid)
    if c0.eid == c1.eid:
        metres = (c1.offset_frac - c0.offset_frac) * ea.length_m
        if metres < 0:
            return None  # backward along the same edge: not a real transition
        return [(str(c0.eid), metres)]

    path = sn.shortest_path(graph, ea.v, eb.u, weight="length_m", mode=MATCH_MODE)
    if path is None:
        return None

    remaining = (1.0 - c0.offset_frac) * ea.length_m
    entered = c1.offset_frac * eb.length_m
    middle = [(str(eid), graph.edge(eid).length_m) for eid in path.edges]
    return [(str(c0.eid), remaining), *middle, (str(c1.eid), entered)]


def match_trip(graph, idx, trip):
    """Match one vehicle's trip (>=2 Ping) and return its PingPair rows."""
    points = [(p.lat, p.lon) for p in trip]
    result = sn.map_match(graph, points, geographic=True, index=idx, mode=MATCH_MODE)

    pairs = []
    for i in range(len(trip) - 1):
        edges = _edge_metres(graph, result.chosen[i], result.chosen[i + 1])
        if edges is None:
            continue
        pairs.append(
            PingPair(
                vehicle_id=trip[i].vehicle_id,
                start_time_utc=trip[i].time_utc,
                end_time_utc=trip[i + 1].time_utc,
                start_position=(trip[i].lat, trip[i].lon),
                end_position=(trip[i + 1].lat, trip[i + 1].lon),
                edges=edges,
                source="observed",
            )
        )
    return pairs


def match_pings(pings, graph=None, idx=None, *, max_gap_s=None):
    """Turn the collector's raw ping rows into PingPair rows via real map matching.

    pings: the collector's rows (e.g. from csv.DictReader) -- dicts with at
    least vehicle_id, vehicle_timestamp and lat/lon as collector.py writes
    them. Parsed and deduplicated by (vehicle_id, time_utc) here, so the
    caller does not need to clean them first.

    graph/idx: pass a RoadGraph/EdgeIndex (e.g. for a small test fixture) to
    skip loading the real frozen graph. Left unloaded when there is nothing to
    match, so an empty or trip-less `pings` never touches disk.
    """
    trip_kwargs = {} if max_gap_s is None else {"max_gap_s": max_gap_s}
    trips = split_trips(pings_from_rows(pings), **trip_kwargs)
    if not trips:
        return []

    if graph is None:
        from src.graph.build_graph import load_graph

        graph = load_graph()
    if idx is None:
        idx = sn.EdgeIndex(graph, mode=MATCH_MODE)

    pairs = []
    for trip in trips:
        pairs.extend(match_trip(graph, idx, trip))
    return pairs
