import csv
import json

import pytest

from engine.orchid import pedestrian_flow as pf
from engine.orchid import street_network as sn
from src.decide import pedestrian_crossing_delay as pcd

ZEBRA = {"highway": "crossing", "crossing": "uncontrolled"}
PUFFIN = {"highway": "crossing", "crossing": "traffic_signals"}


@pytest.fixture(scope="module")
def graph():
    """A two-way street with a zebra at node 1, a one-way street with a puffin at node 4,
    a crossing on a footway only (node 7), and a signalised junction that isn't a crossing (node 9)."""
    g = sn.RoadGraph()
    tags = {1: ZEBRA, 4: PUFFIN, 7: ZEBRA, 9: {"highway": "traffic_signals"}}
    for nid in range(10):
        g.add_node(nid, lat=55.9490 + 0.001 * (nid // 3), lon=-3.2000 + 0.001 * (nid % 3), osm=tags.get(nid, {}))
    g.add_edge(0, 1, oneway=False, highway_class="tertiary")      # 0, 1
    g.add_edge(1, 2, oneway=False, highway_class="tertiary")      # 2, 3
    g.add_edge(3, 4, highway_class="residential")                 # 4
    g.add_edge(4, 5, highway_class="residential")                 # 5
    g.add_edge(6, 7, oneway=False, highway_class="footway")       # 6, 7
    g.add_edge(8, 9, oneway=False, highway_class="tertiary")      # 8, 9
    return g


@pytest.fixture(scope="module")
def crossings(graph):
    return pcd.find_crossings(graph)


FLOWS = {"0": 300.0, "3": 200.0, "4": 450.0}


def test_find_crossings_on_car_roads_only(crossings):
    assert crossings == [
        {"node": 1, "edge_ids": ["0", "3"], "signalised": False, "width_m": 2 * pcd.LANE_WIDTH_M},
        {"node": 4, "edge_ids": ["4"], "signalised": True, "width_m": pcd.LANE_WIDTH_M},
    ]


def test_unsignalised_delay_uses_both_directions_flow(crossings):
    zebra = pcd.crossing_delays(crossings, FLOWS)[0]
    gap = pf.critical_gap(2 * pcd.LANE_WIDTH_M)
    assert zebra.control == "unsignalised-one-stage" and zebra.detail["combinedFlow"] == 500.0
    assert zebra.delay == pytest.approx(pf.adams_delay(500.0, gap))


def test_signalised_delay_comes_from_the_timings_only(crossings):
    puffin = pcd.crossing_delays(crossings, FLOWS)[1]
    assert puffin.control == "signalised"
    assert puffin.delay == pytest.approx((pcd.CYCLE_S - pcd.WALK_GREEN_S) ** 2 / (2 * pcd.CYCLE_S))
    assert pcd.crossing_delays(crossings, {})[1].delay == puffin.delay


def test_missing_edges_carry_no_traffic(crossings):
    assert pcd.crossing_delays(crossings, {})[0].delay == 0.0


def test_j_ped_sums_delays_and_grows_with_traffic(crossings):
    delays = pcd.crossing_delays(crossings, FLOWS)
    assert pcd.j_ped(crossings, FLOWS) == pytest.approx(delays[0].delay + delays[1].delay)
    busier = {eid: 2 * flow for eid, flow in FLOWS.items()}
    assert pcd.j_ped(crossings, busier) > pcd.j_ped(crossings, FLOWS)


def test_load_flows_reads_the_baseline_csv(tmp_path):
    path = tmp_path / "baseline_user_equilibrium.csv"
    path.write_text("edge_id,length_m,free_flow_time_s,ue_time_s,flow_veh_h,vc\n0,80,6,7,300.5,0.2\n3,80,6,7,200,0.1\n")
    assert pcd.load_flows(path) == {"0": 300.5, "3": 200.0}


def test_fruin_levels_worsen_as_crowds_grow():
    grades = pcd.fruin_levels()
    assert grades == {"quiet": "A", "busy": "C", "festival_peak": "D"}


def test_main_writes_crossings_and_baseline_j_ped(graph, crossings, tmp_path, monkeypatch, capsys):
    ue = tmp_path / "baseline_user_equilibrium.csv"
    ue.write_text("edge_id,length_m,free_flow_time_s,ue_time_s,flow_veh_h,vc\n"
                  + "".join(f"{eid},80,6,7,{flow},0.1\n" for eid, flow in FLOWS.items()))
    monkeypatch.setattr(pcd, "UE_PATH", ue)
    monkeypatch.setattr(pcd, "CROSSINGS_PATH", tmp_path / "demand" / "pedestrian_crossing_delay.csv")
    monkeypatch.setattr(pcd, "J_PED_PATH", tmp_path / "demand" / "pedestrian_crossing_delay.json")
    monkeypatch.setattr(pcd, "load_graph", lambda: graph)

    pcd.main()

    with (tmp_path / "demand" / "pedestrian_crossing_delay.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [(r["node"], r["edge_ids"], r["control"]) for r in rows] == [
        ("1", "0;3", "unsignalised-one-stage"), ("4", "4", "signalised")]
    assert rows[0]["vehicle_flow_veh_h"] == "500.0" and rows[1]["vehicle_flow_veh_h"] == ""
    summary = json.loads((tmp_path / "demand" / "pedestrian_crossing_delay.json").read_text())
    assert summary["j_ped_s"] == pytest.approx(pcd.j_ped(crossings, FLOWS))
    assert summary["crossings"] == 2 and summary["signalised"] == 1
    assert summary["synthetic_fruin_los"]["facility"] == "queueing"
    assert capsys.readouterr().out.startswith("Baseline J_ped = ")
