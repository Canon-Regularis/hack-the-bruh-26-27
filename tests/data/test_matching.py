"""match_pings against a tiny, hand-built real RoadGraph (not synthetic edges)."""

from datetime import datetime, timedelta, timezone

import pytest

from engine.orchid import street_network as sn
from matching.table import match_pings, match_trip
from matching.trips import Ping, pings_from_rows, split_trips

T0 = datetime(2026, 11, 7, 18, 0, 0, tzinfo=timezone.utc)

# A straight line of four nodes, three one-way edges: A (n0->n1), B (n1->n2),
# C (n2->n3). Same latitude throughout, so edge lengths only depend on the
# longitude step -- no need to hand-derive the graph's own projection maths.
NODES = {
    "n0": (55.9500, -3.2000),
    "n1": (55.9500, -3.1990),
    "n2": (55.9500, -3.1980),
    "n3": (55.9500, -3.1970),
}


@pytest.fixture
def graph():
    g = sn.RoadGraph()
    for nid, (lat, lon) in NODES.items():
        g.add_node(nid, lat=lat, lon=lon)
    g.add_edge("n0", "n1", oneway=True, street_name="A")
    g.add_edge("n1", "n2", oneway=True, street_name="B")
    g.add_edge("n2", "n3", oneway=True, street_name="C")
    return g


@pytest.fixture
def idx(graph):
    return sn.EdgeIndex(graph, mode="bus")


def edge_id(graph, u, v):
    for e in graph.iter_edges():
        if e.u == u and e.v == v:
            return e.eid
    raise AssertionError(f"no edge {u}->{v}")


def midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def row(vehicle_id, time_utc, position):
    """A raw collector-style row, as csv.DictReader would hand it to us."""
    return {
        "vehicle_id": vehicle_id,
        "vehicle_timestamp": time_utc.isoformat(),
        "lat": str(position[0]),
        "lon": str(position[1]),
    }


def test_match_trip_splits_metres_within_one_edge(graph, idx):
    # Two pings both on edge A: start, then its midpoint.
    trip = [
        Ping("bus_1", T0, *NODES["n0"]),
        Ping("bus_1", T0 + timedelta(seconds=15), *midpoint(NODES["n0"], NODES["n1"])),
    ]
    pairs = match_trip(graph, idx, trip)

    assert len(pairs) == 1
    [pair] = pairs
    a_eid = str(edge_id(graph, "n0", "n1"))
    assert [eid for eid, _ in pair.edges] == [a_eid]

    a_length = graph.edge(edge_id(graph, "n0", "n1")).length_m
    [(_, metres)] = pair.edges
    assert metres == pytest.approx(a_length / 2, rel=0.05)


def test_match_trip_crosses_a_fully_traversed_intermediate_edge(graph, idx):
    # Pings at the midpoint of A and the midpoint of C: B is never pinged
    # directly, but the bus must have driven all of it to get there.
    trip = [
        Ping("bus_1", T0, *midpoint(NODES["n0"], NODES["n1"])),
        Ping("bus_1", T0 + timedelta(seconds=30), *midpoint(NODES["n2"], NODES["n3"])),
    ]
    pairs = match_trip(graph, idx, trip)

    assert len(pairs) == 1
    [pair] = pairs
    a_eid = str(edge_id(graph, "n0", "n1"))
    b_eid = str(edge_id(graph, "n1", "n2"))
    c_eid = str(edge_id(graph, "n2", "n3"))
    assert [eid for eid, _ in pair.edges] == [a_eid, b_eid, c_eid]

    b_length = graph.edge(edge_id(graph, "n1", "n2")).length_m
    middle_metres = dict(pair.edges)[b_eid]
    assert middle_metres == pytest.approx(b_length, rel=0.01)


def test_match_pings_groups_by_vehicle_and_splits_trips(graph, idx):
    # Raw collector-style rows, not Ping objects -- this is match_pings' real
    # contract: scripts/build_edge_speeds.py calls it this way.
    rows = [
        row("bus_1", T0, NODES["n0"]),
        row("bus_1", T0 + timedelta(seconds=15), midpoint(NODES["n0"], NODES["n1"])),
        # A different vehicle, far later in time: its own trip, not stitched
        # onto bus_1's.
        row("bus_2", T0 + timedelta(hours=2), NODES["n1"]),
        row("bus_2", T0 + timedelta(hours=2, seconds=15), midpoint(NODES["n1"], NODES["n2"])),
    ]
    pairs = match_pings(rows, graph=graph, idx=idx)

    assert {p.vehicle_id for p in pairs} == {"bus_1", "bus_2"}
    assert len(pairs) == 2


def test_match_pings_is_empty_without_touching_the_graph():
    # No graph/idx passed, and nothing to match -- must not try to load the
    # real frozen graph from disk.
    assert match_pings([]) == []


def test_pings_from_rows_deduplicates_and_skips_bad_rows():
    rows = [
        row("bus_1", T0, NODES["n0"]),
        row("bus_1", T0, NODES["n0"]),  # the same fix, seen again on a later poll
        {"vehicle_id": "", "vehicle_timestamp": T0.isoformat(), "lat": "0", "lon": "0"},
        {"vehicle_id": "bus_2", "vehicle_timestamp": "not a timestamp", "lat": "0", "lon": "0"},
        {"vehicle_id": "bus_3", "vehicle_timestamp": T0.isoformat(), "lat": "", "lon": "0"},
    ]
    pings = pings_from_rows(rows)
    assert len(pings) == 1
    assert pings[0].vehicle_id == "bus_1"


def test_split_trips_breaks_on_a_large_gap():
    pings = [
        Ping("bus_1", T0, 55.95, -3.19),
        Ping("bus_1", T0 + timedelta(seconds=30), 55.951, -3.19),
        Ping("bus_1", T0 + timedelta(hours=1), 55.952, -3.19),
        Ping("bus_1", T0 + timedelta(hours=1, seconds=30), 55.953, -3.19),
    ]
    trips = split_trips(pings, max_gap_s=300)
    assert len(trips) == 2
    assert [p.time_utc for p in trips[0]] == [pings[0].time_utc, pings[1].time_utc]
    assert [p.time_utc for p in trips[1]] == [pings[2].time_utc, pings[3].time_utc]


def test_split_trips_drops_single_ping_trips():
    pings = [Ping("bus_1", T0, 55.95, -3.19)]
    assert split_trips(pings) == []
