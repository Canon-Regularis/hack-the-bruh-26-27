import csv
import json
import math

import pytest

from engine.orchid import street_network as sn
from engine.orchid import traffic_assignment as ta
from engine.orchid import travel_demand as td
from src.decide import od_matrix_estimation as od
from src.decide.network import link_maps, to_network

SPUR = 100


@pytest.fixture(scope="module")
def graph():
    """A 3x3 grid of two-way main roads, plus a one-way dead-end spur from the corner."""
    g = sn.RoadGraph()
    for r in range(3):
        for c in range(3):
            g.add_node(r * 3 + c, lat=55.9450 + 0.0018 * r, lon=-3.2000 + 0.0032 * c)
    g.add_node(SPUR, lat=55.9440, lon=-3.2010)
    for r in range(3):
        for c in range(3):
            n = r * 3 + c
            if c < 2:
                g.add_edge(n, n + 1, oneway=False, highway_class="primary")
            if r < 2:
                g.add_edge(n, n + 3, oneway=False, highway_class="primary")
    g.add_edge(0, SPUR, highway_class="residential")
    return g


@pytest.fixture(scope="module")
def built(graph):
    net, node_ids = to_network(graph)
    return net, node_ids, link_maps(net)[0]


def links_of(graph, edge_to_link, u, v):
    """Link indices for both directions of the road between OSM nodes u and v."""
    return [edge_to_link[str(e)] for a, b in ((u, v), (v, u)) for e in graph.edges_between(a, b)]


def test_largest_strong_component_drops_the_dead_end(built):
    net, node_ids, _ = built
    assert [node_ids[n] for n in od._largest_strong_component(net)] == list(range(9))


def test_make_zones_gives_distinct_reachable_centroids_every_time(built, graph):
    net, node_ids, _ = built
    xy = [graph.node(nid).xy for nid in node_ids]
    centroids = od.make_zones(net, xy, 4)
    assert len(set(centroids)) == 4 and SPUR not in {node_ids[c] for c in centroids}
    assert od.make_zones(net, xy, 4) == centroids


def test_free_flow_costs_are_shortest_path_seconds(built):
    net, node_ids, _ = built
    free_flow = net.free_flow_times()
    centroids = [node_ids.index(0), node_ids.index(8)]
    costs = od.free_flow_costs(net, centroids, free_flow)
    assert costs[0][0] == costs[1][1] == 0.0
    assert costs[0][1] == pytest.approx(ta.shortest_path(net, centroids[0], centroids[1], free_flow)[0])
    assert costs[0][1] == pytest.approx(costs[1][0])


def test_gravity_prior_has_equal_trip_ends_and_no_intrazonal_trips():
    costs = [[0, 60, 120], [60, 0, 90], [120, 90, 0]]
    prior = od.gravity_prior(costs, 300.0)
    assert all(prior[i][i] == 0.0 for i in range(3))
    assert [math.fsum(row) for row in prior] == pytest.approx([100.0] * 3)
    assert [math.fsum(col) for col in zip(*prior)] == pytest.approx([100.0] * 3)
    assert prior[0][1] > prior[0][2]


def test_hourly_counts_convert_aadf_and_keep_both_directions(tmp_path):
    path = tmp_path / "dft_aadf.csv"
    path.write_text("edge_id,count_point_id,aadf\n4,77,10000\n5,77,10000\n9,80,2000\n")
    counts, edges = od.hourly_counts(path)
    assert counts == {"77": pytest.approx(900.0), "80": pytest.approx(180.0)}
    assert edges == {"77": ["4", "5"], "80": ["9"]}


def test_proportions_sum_both_directions_of_a_count_point():
    g = sn.RoadGraph()
    for nid, lon in enumerate([-3.2000, -3.1990, -3.1980]):
        g.add_node(nid, lat=55.9490, lon=lon)
    g.add_edge(0, 1, oneway=False, highway_class="primary")
    g.add_edge(1, 2, oneway=False, highway_class="primary")
    net, node_ids = to_network(g)
    edge_to_link, _ = link_maps(net)
    count_links = {"cp": links_of(g, edge_to_link, 1, 2)}
    centroids = [node_ids.index(0), node_ids.index(2)]
    proportions = od.assignment_proportions(net, centroids, [[0.0, 100.0], [50.0, 0.0]], count_links)
    assert proportions.proportions == {"cp": {(0, 1): pytest.approx(1.0), (1, 0): pytest.approx(1.0)}}


@pytest.fixture(scope="module")
def count_setup(built, graph):
    net, node_ids, edge_to_link = built
    count_links = {"10": links_of(graph, edge_to_link, 0, 1), "20": links_of(graph, edge_to_link, 4, 5),
                   "30": links_of(graph, edge_to_link, 7, 8)}
    centroids = od.make_zones(net, [graph.node(nid).xy for nid in node_ids], 4)
    return net, centroids, count_links


def test_estimate_od_fits_the_counts_better_than_the_prior(count_setup, monkeypatch):
    net, centroids, count_links = count_setup
    counts = {"10": 400.0, "20": 900.0, "30": 250.0}
    seen = {}
    real_gls = td.gls_matrix_estimation

    def spy(prior, counted, proportions, **kw):
        seen.update(prior=prior, proportions=proportions)
        return real_gls(prior, counted, proportions, **kw)

    monkeypatch.setattr(od.td, "gls_matrix_estimation", spy)
    matrix, fit = od.estimate_od(net, centroids, counts, count_links)

    assert sorted(fit) == ["10", "20", "30"] and all(c == counts[cid] for cid, (_, c) in fit.items())
    assert all(v >= 0.0 for row in matrix for v in row) and all(matrix[i][i] == pytest.approx(0.0) for i in range(4))
    prior_modelled = dict(zip(seen["proportions"].links, seen["proportions"].flows(seen["prior"])))
    assert sum(abs(m - c) for m, c in fit.values()) < sum(abs(prior_modelled[cid] - counts[cid]) for cid in counts)


def test_estimate_od_rejects_counts_no_route_uses(count_setup, built, graph):
    net, centroids, _ = count_setup
    edge_to_link = built[2]
    spur_links = [edge_to_link[str(e)] for e in graph.edges_between(0, SPUR)]
    with pytest.raises(ValueError, match="no count point lies on any zone-to-zone route"):
        od.estimate_od(net, centroids, {"99": 100.0}, {"99": spur_links})


def test_main_writes_labelled_matrix_and_count_fit(graph, tmp_path, monkeypatch, capsys):
    rows = [(e, cp, aadf) for (u, v, cp, aadf) in [(0, 1, "10", 5000), (4, 5, "20", 10000)]
            for a, b in ((u, v), (v, u)) for e in graph.edges_between(a, b)]
    aadf = tmp_path / "dft_aadf.csv"
    aadf.write_text("edge_id,count_point_id,aadf\n" + "".join(f"{e},{cp},{a}\n" for e, cp, a in rows))
    monkeypatch.setattr(od, "AADF_PATH", aadf)
    monkeypatch.setattr(od, "OD_PATH", tmp_path / "demand" / "od_matrix.json")
    monkeypatch.setattr(od, "FIT_PATH", tmp_path / "demand" / "count_fit.csv")
    monkeypatch.setattr(od, "N_ZONES", 4)
    monkeypatch.setattr(od, "load_graph", lambda: graph)

    od.main()

    out = json.loads((tmp_path / "demand" / "od_matrix.json").read_text())
    assert out["label"] == "estimated" and out["hour"] == od.MODELLED_HOUR and out["units"] == "veh/h"
    assert len(out["zone_centroid_nodes"]) == 4 and len(out["matrix"]) == 4
    with (tmp_path / "demand" / "count_fit.csv").open() as f:
        fit = list(csv.DictReader(f))
    assert [r["count_point_id"] for r in fit] == ["10", "20"] and float(fit[1]["counted_veh_h"]) == 900.0
    assert capsys.readouterr().out.startswith("Estimated OD matrix: 4 zones")
