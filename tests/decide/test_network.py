import pytest

from engine.orchid import street_network as sn
from engine.orchid import traffic_assignment as ta
from src.decide.network import link_maps, to_network


def test_braess_textbook_numbers():
    net, od = ta.braess_network()
    result = ta.detect_braess(net, od)
    assert result.paradox and result.best_link == net.link_index(1, 2)
    assert result.base_total_travel_time == pytest.approx(552)
    assert result.best_total_travel_time == pytest.approx(498)
    assert result.best_total_travel_time / od.total() == pytest.approx(83)
    cost, link = result.base.costs, net.link_index
    for route in [(0, 1, 3), (0, 2, 3), (0, 1, 2, 3)]:
        assert sum(cost[link(a, b)] for a, b in zip(route, route[1:])) == pytest.approx(92)


@pytest.fixture(scope="module")
def graph():
    g = sn.RoadGraph()
    for nid, lon in [(9001, -3.2000), (42, -3.1990), (500, -3.1980), (77, -3.1970)]:
        g.add_node(nid, lat=55.9490, lon=lon)
    g.add_edge(9001, 42, oneway=False, street_name="Lothian Road", highway_class="primary")  # 0, 1
    g.add_edge(42, 500, highway_class="residential")                                         # 2
    g.add_edge(500, 77, oneway=False, highway_class="footway")                               # 3, 4
    g.add_edge(500, 9001, highway_class="service", capacity_veh_h=0)                         # 5
    return g


@pytest.fixture(scope="module")
def built(graph):
    return to_network(graph)


def test_only_car_edges_with_capacity_become_links(built):
    net, _ = built
    assert [link.name for link in net.links] == ["0", "1", "2"]


def test_nodes_renumbered_densely_in_osm_id_order(built):
    net, node_ids = built
    assert node_ids == [42, 500, 9001]
    assert [(link.tail, link.head) for link in net.links] == [(2, 0), (0, 2), (0, 1)]


def test_links_carry_free_flow_time_capacity_and_length(built, graph):
    net, _ = built
    for link in net.links:
        edge = graph.edge(int(link.name))
        assert isinstance(link.vdf, ta.BPR)
        assert link.vdf.t0 == edge.free_flow_time_s and link.vdf.capacity == edge.capacity_veh_h
        assert link.length == edge.length_m


def test_link_maps_both_directions(built):
    edge_to_link, link_to_edge = link_maps(built[0])
    assert edge_to_link == {"0": 0, "1": 1, "2": 2}
    assert link_to_edge == ["0", "1", "2"]


def test_link_maps_rebuilt_after_without_link(built):
    edge_to_link, link_to_edge = link_maps(built[0].without_link(0))
    assert edge_to_link == {"1": 0, "2": 1}
    assert link_to_edge == ["1", "2"]


def test_assignment_runs_on_the_adapted_network(built):
    net, node_ids = built
    origin, dest = node_ids.index(9001), node_ids.index(500)
    ue = ta.user_equilibrium(net, ta.ODMatrix({(origin, dest): 600.0}))
    edge_to_link, _ = link_maps(net)
    assert ue.flows[edge_to_link["0"]] == pytest.approx(600) and ue.flows[edge_to_link["2"]] == pytest.approx(600)
    assert ue.flows[edge_to_link["1"]] == pytest.approx(0)
