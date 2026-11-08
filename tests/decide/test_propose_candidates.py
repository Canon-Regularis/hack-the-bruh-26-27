import json

import pytest

from engine.orchid import street_network as sn
from src.decide import interventions as iv
from src.decide import propose_candidates as pc
from src.decide.pedestrian_crossing_delay import find_crossings

CROSSING = {"highway": "crossing", "crossing": "uncontrolled"}
SOURCE = "CMP You Said We Did v2, para 3.32"


@pytest.fixture
def graph():
    """Edges in add order (eids 2i, 2i+1): Princes Street (primary), George Street, Cowgate, The Mound,
    Outside Road (leaves the study area), Quiet Lane, Lothian Road (primary). Crossings at nodes 0 and 6."""
    g = sn.RoadGraph()
    points = [(55.950, -3.200), (55.950, -3.195), (55.951, -3.190), (55.952, -3.185), (55.953, -3.180),
              (55.950, -3.300), (55.954, -3.175)]
    for nid, (lat, lon) in enumerate(points):
        g.add_node(nid, lat=lat, lon=lon, osm=CROSSING if nid in (0, 6) else {})
    for u, v, name, cls in [(0, 1, "Princes Street", "primary"), (1, 2, "George Street", "tertiary"),
                            (2, 3, "Cowgate", "tertiary"), (3, 4, "The Mound", "tertiary"),
                            (0, 5, "Outside Road", "tertiary"), (4, 6, "Quiet Lane", "tertiary"),
                            (2, 6, "Lothian Road", "primary")]:
        g.add_edge(u, v, oneway=False, street_name=name, highway_class=cls)
    return g


def ue(rows):
    return [{"edge_id": str(eid), "flow_veh_h": str(flow), "vc": str(vc)} for eid, flow, vc in rows]


UE = ue([(0, 2400, 1.4), (1, 100, 0.1), (2, 900, 0.7), (3, 50, 0.05), (4, 70, 0.05), (5, 0, 0),
         (6, 1600, 1.6), (7, 0, 0), (8, 5000, 2.0), (9, 0, 0), (10, 0, 0), (11, 0, 0), (12, 4000, 0.65), (13, 0, 0)])


def crossing(node, edges, delay, control="unsignalised-one-stage", flow=500, los="F"):
    return {"node": str(node), "edge_ids": edges, "control": control, "vehicle_flow_veh_h": str(flow),
            "delay_s": str(delay), "los": los, "width_m": "6.5"}


CROSSINGS = [
    crossing(1, "0", 60, flow=2400), crossing(0, "1", 60, flow=2400),
    crossing(6, "10", 30, los="E"),
    crossing(2, "2", 90, control="signalised"),
    crossing(5, "8", 99),
    crossing(4, "6", 0, los="A"),
]

TWO_EACH = {iv.SIGNALISE: 2, iv.REALLOCATE: 2, iv.CLOSE: 2}

CONCERNS = [{"theme": "rat_running", "street": s, "count": "1", "source": SOURCE}
            for s in ["George Street", "Lothian Road", "Cowgate", "Quiet Lane", "Princes Street"]]


def test_propose_picks_each_kind_from_the_data(graph):
    picks = pc.propose(graph, UE, CROSSINGS, CONCERNS, "08:00-09:00 weekday", counts=TWO_EACH, closure_street="The Mound")
    assert [(c["id"], c["kind"]) for c in picks] == [
        ("signal-princes-street", "signalise_crossing"), ("signal-quiet-lane", "signalise_crossing"),
        ("bus-lane-george-street", "reallocate"), ("bus-lane-lothian-road", "reallocate"),
        ("close-cowgate", "close"),
    ]
    assert [c["node"] for c in picks[:2]] == ["0", "6"]
    assert [c.get("capacity_factor") for c in picks] == [pc.SIGNAL_FACTOR] * 2 + [pc.BUS_LANE_FACTOR] * 2 + [None]


def test_one_candidate_per_street_and_none_on_the_closure_street(graph):
    picks = pc.propose(graph, UE, CROSSINGS, CONCERNS, "08:00-09:00 weekday", counts=TWO_EACH, closure_street="The Mound")
    streets = [c["street"] for c in picks]
    assert len(streets) == len(set(streets))
    assert "The Mound" not in streets and "Outside Road" not in streets


def test_complaint_closures_skip_main_roads_and_streets_without_cars(graph):
    loads = pc.street_loads(graph, UE)
    picks = pc.complaint_streets(loads, CONCERNS, taken=set(), n=5)
    assert [c["street"] for c in picks] == ["George Street", "Cowgate"]


def test_reasons_cite_the_numbers_they_were_picked_on(graph):
    picks = {c["id"]: c for c in pc.propose(graph, UE, CROSSINGS, CONCERNS, "08:00-09:00 weekday",
                                            counts=TWO_EACH, closure_street="The Mound")}
    assert "60 s (level of service F)" in picks["signal-princes-street"]["reason"]
    assert "900 veh/h at 70% of capacity (08:00-09:00 weekday)" in picks["bus-lane-george-street"]["reason"]
    assert SOURCE in picks["close-cowgate"]["reason"]
    signal = picks["signal-princes-street"]["evidence"]
    assert signal["delay_s"] == 60.0 and signal["signalised_delay_s"] == 37.4 and signal["signalised_los"] == "E"
    assert picks["close-cowgate"]["evidence"] == {"flow_veh_h": 70.0, "themes": ["rat running"], "sources": [SOURCE]}


def test_proposed_candidates_resolve_in_the_interventions_check(graph, tmp_path):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(pc.propose(graph, UE, CROSSINGS, CONCERNS, "08:00-09:00 weekday",
                                          counts=TWO_EACH, closure_street="The Mound")))
    candidates, notes = iv.load_candidates(path, graph, find_crossings(graph))
    assert notes == []
    assert len(candidates) == 5


def test_default_counts_take_one_crossing_one_bus_lane_and_two_closures(graph):
    concerns = CONCERNS + [{"theme": "noise", "street": "Lothian Road", "count": "1", "source": SOURCE}]
    picks = pc.propose(graph, UE, CROSSINGS, concerns, "08:00-09:00 weekday", closure_street="The Mound")
    assert [c["id"] for c in picks] == ["signal-princes-street", "bus-lane-george-street", "close-cowgate"]
    assert pc.COUNTS == {iv.SIGNALISE: 1, iv.REALLOCATE: 1, iv.CLOSE: 2}
