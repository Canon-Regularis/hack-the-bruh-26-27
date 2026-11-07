import csv
import json
from types import SimpleNamespace

import pytest

from engine.orchid import street_network as sn
from engine.orchid import traffic_assignment as ta
from src.decide import baseline_user_equilibrium as bue
from src.decide.network import to_network

CENTROIDS = [0, 2, 6, 8]
MATRIX = [[0, 900, 400, 300], [700, 0, 300, 500], [350, 250, 0, 800], [300, 600, 700, 0]]


@pytest.fixture(scope="module")
def graph():
    """A 3x3 grid of two-way main roads."""
    g = sn.RoadGraph()
    for r in range(3):
        for c in range(3):
            g.add_node(r * 3 + c, lat=55.9450 + 0.0018 * r, lon=-3.2000 + 0.0032 * c)
    for r in range(3):
        for c in range(3):
            n = r * 3 + c
            if c < 2:
                g.add_edge(n, n + 1, oneway=False, highway_class="secondary")
            if r < 2:
                g.add_edge(n, n + 3, oneway=False, highway_class="secondary")
    return g


@pytest.fixture(scope="module")
def built(graph):
    return to_network(graph)


@pytest.fixture(scope="module")
def od_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("demand") / "od_matrix.json"
    path.write_text(json.dumps({"label": "estimated", "hour": "08:00-09:00 weekday", "units": "veh/h",
                                "zone_centroid_nodes": CENTROIDS, "matrix": MATRIX}))
    return path


@pytest.fixture(scope="module")
def loaded(built, od_path):
    return bue.load_od(od_path, built[1])


@pytest.fixture(scope="module")
def result(built, loaded):
    return bue.run_ue(built[0], loaded[0])


def test_load_od_places_trips_on_centroid_nodes(built, loaded):
    _, node_ids = built
    od, hour = loaded
    assert hour == "08:00-09:00 weekday"
    demand = od.as_dict()
    assert len(demand) == 12 and demand[(node_ids.index(0), node_ids.index(2))] == 900
    assert all(o != d for o, d in demand)


def test_run_ue_converges_and_agrees_with_another_solver(built, loaded, result):
    """Equal-time twin routes on the grid let solvers split flow differently, so compare times."""
    net, _ = built
    assert result.converged and result.relative_gap <= 1e-6
    other = ta.user_equilibrium(net, loaded[0], method="gp")
    assert result.total_travel_time == pytest.approx(other.total_travel_time, rel=1e-5)
    assert result.travel_times == pytest.approx(other.travel_times, abs=0.01)


def test_run_ue_raises_when_not_converged(built, loaded, monkeypatch):
    monkeypatch.setattr(bue.ta, "user_equilibrium",
                        lambda *a, **kw: SimpleNamespace(converged=False, relative_gap=0.02, iterations=200))
    with pytest.raises(RuntimeError, match="did not converge: relative gap 2.00e-02 after 200 iterations"):
        bue.run_ue(built[0], loaded[0])


def test_ue_rows_one_per_directed_edge_in_seconds_and_metres(built, graph, result):
    net, _ = built
    rows = bue.ue_rows(net, result)
    assert [r["edge_id"] for r in rows] == [str(e) for e in graph.edges]
    for row, flow, vc in zip(rows, result.flows, result.vc_ratios(net)):
        edge = graph.edge(int(row["edge_id"]))
        assert row["length_m"] == edge.length_m and row["free_flow_time_s"] == pytest.approx(edge.free_flow_time_s)
        assert row["ue_time_s"] >= row["free_flow_time_s"]
        assert row["flow_veh_h"] == flow and row["vc"] == pytest.approx(vc)


def test_main_writes_edge_csv_and_hour(graph, od_path, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bue, "OD_PATH", od_path)
    monkeypatch.setattr(bue, "UE_PATH", tmp_path / "demand" / "baseline_user_equilibrium.csv")
    monkeypatch.setattr(bue, "UE_META_PATH", tmp_path / "demand" / "baseline_user_equilibrium.json")
    monkeypatch.setattr(bue, "load_graph", lambda: graph)

    bue.main()

    with (tmp_path / "demand" / "baseline_user_equilibrium.csv").open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert tuple(reader.fieldnames) == bue.UE_FIELDS and len(rows) == graph.num_edges
    meta = json.loads((tmp_path / "demand" / "baseline_user_equilibrium.json").read_text())
    assert meta["hour"] == "08:00-09:00 weekday" and meta["converged"] and meta["method"] == "Algorithm B"
    assert capsys.readouterr().out.startswith("Baseline equilibrium for 08:00-09:00 weekday: 24 edges")
