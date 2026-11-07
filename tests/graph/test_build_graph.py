import json
import pickle

import pytest

from engine.orchid import street_network as sn
from src.graph.build_graph import (
    build_graph, bus_stops, highway_class, load_graph, oneway_direction, read_osm, street_names,
)


def node(nid, lat, lon, **tags):
    el = {"type": "node", "id": nid, "lat": lat, "lon": lon}
    if tags:
        el["tags"] = tags
    return el


def way(wid, nodes, **tags):
    return {"type": "way", "id": wid, "nodes": nodes, "tags": tags}


ELEMENTS = [
    way(105, [4, 11], highway="steps"),
    way(100, [1, 2, 3, 4, 5], highway="tertiary", name="Lawnmarket", motor_vehicle="no", access="destination"),
    way(101, [5, 6, 7], highway="residential", name="Upper Bow", oneway="-1"),
    way(102, [7, 8], highway="primary_link", oneway="yes"),
    way(103, [9, 10], building="yes"),
    way(104, [8, 12], highway="construction"),
    way(106, [30, 31, 32, 30], highway="pedestrian", area="yes", name="Square"),
    node(1, 55.94900, -3.19900),
    node(2, 55.94905, -3.19850, highway="crossing", crossing="zebra"),
    node(3, 55.94910, -3.19800, highway="traffic_signals"),
    node(4, 55.94915, -3.19750),
    node(5, 55.94920, -3.19700),
    node(6, 55.94950, -3.19690),
    node(7, 55.94980, -3.19700),
    node(8, 55.95000, -3.19650),
    node(9, 55.94000, -3.19000),
    node(10, 55.94001, -3.19001),
    node(11, 55.94930, -3.19760),
    node(12, 55.95010, -3.19640),
    node(20, 55.94925, -3.19705, highway="bus_stop"),
    node(21, 55.94985, -3.19710, public_transport="platform", bus="yes"),
    node(22, 55.94990, -3.19720, public_transport="platform"),
    node(23, 55.94995, -3.19660, public_transport="stop_position", bus="yes"),
    node(24, 55.94800, -3.19500, highway="traffic_signals"),
    node(30, 55.94700, -3.19600),
    node(31, 55.94710, -3.19600),
    node(32, 55.94710, -3.19580),
]
LATLON = {el["id"]: (el["lat"], el["lon"]) for el in ELEMENTS if el["type"] == "node"}


def write_osm(path, elements):
    path.write_text(json.dumps({"elements": elements}))
    return path


def same_graph(a, b):
    return a.nodes == b.nodes and a.edges == b.edges and all(vars(a.edge(e)) == vars(b.edge(e)) for e in a.edges)


def edge_between(graph, u, v):
    (eid,) = graph.edges_between(u, v)
    return graph.edge(eid)


@pytest.fixture(scope="module")
def osm_path(tmp_path_factory):
    return write_osm(tmp_path_factory.mktemp("osm") / "osm.json", ELEMENTS)


@pytest.fixture(scope="module")
def osm(osm_path):
    return read_osm(osm_path)


@pytest.fixture(scope="module")
def graph(osm):
    return build_graph(osm)


@pytest.mark.parametrize("raw, expected", [
    ("primary", "primary"),
    ("primary_link", "primary"),
    ("motorway_link", "motorway"),
    ("living_street", "living_street"),
    ("steps", "footway"),
    ("path", "footway"),
    ("busway", None),
    ("construction", None),
    ("steps_link", None),
    ("", None),
])
def test_highway_class(raw, expected):
    assert highway_class(raw) == expected


@pytest.mark.parametrize("tags, expected", [
    ({}, 0),
    ({"oneway": "yes"}, 1),
    ({"oneway": "Yes "}, 1),
    ({"oneway": "1"}, 1),
    ({"oneway": "-1"}, -1),
    ({"oneway": "reverse"}, -1),
    ({"oneway": "no"}, 0),
    ({"oneway": "alternating"}, 0),
    ({"junction": "roundabout"}, 1),
    ({"highway": "motorway_link"}, 1),
    ({"junction": "roundabout", "oneway": "no"}, 0),
])
def test_oneway_direction(tags, expected):
    assert oneway_direction(tags) == expected


def test_nodes_are_way_ends_junctions_signals_and_crossings_in_id_order(graph):
    assert graph.nodes == [1, 2, 3, 4, 5, 7, 8, 11]


def test_node_flags_and_osm_tags(graph):
    assert graph.node(3).signalised
    assert not graph.node(2).signalised
    assert graph.node(2).tags["osm"] == {"highway": "crossing", "crossing": "zebra"}
    assert graph.node(1).tags["osm"] == {}


def test_two_way_street_has_linked_twins_with_default_lanes_and_speed(graph):
    fwd, back = edge_between(graph, 1, 2), edge_between(graph, 2, 1)
    defaults = sn.HIGHWAY_DEFAULTS["tertiary"]
    assert not fwd.oneway
    assert fwd.reverse_eid == back.eid and back.reverse_eid == fwd.eid
    assert fwd.lanes == defaults["lanes"] and fwd.speed_limit_kph == defaults["speed_kph"]


def test_oneway_reverse_swaps_direction_and_follows_the_bend(graph):
    bow = edge_between(graph, 7, 5)
    assert bow.oneway and not graph.edges_between(5, 7)
    assert bow.geometry[0] == graph.node(7).xy and bow.geometry[-1] == graph.node(5).xy
    assert len(bow.geometry) == 3
    via_bend = sn.haversine_m(*LATLON[7], *LATLON[6]) + sn.haversine_m(*LATLON[6], *LATLON[5])
    assert bow.length_m == pytest.approx(via_bend, rel=1e-3)
    assert bow.length_m > sn.haversine_m(*LATLON[7], *LATLON[5])


def test_oneway_yes_only_goes_forward(graph):
    assert edge_between(graph, 7, 8).oneway and not graph.edges_between(8, 7)


def test_road_types_and_modes(graph):
    assert edge_between(graph, 7, 8).highway_class == "primary"
    steps = edge_between(graph, 4, 11)
    assert steps.highway_class == "footway" and steps.modes_permitted == ("foot",)


def test_access_tags_kept_raw_without_changing_modes(graph):
    lawnmarket = edge_between(graph, 1, 2)
    assert lawnmarket.street_name == "Lawnmarket"
    assert lawnmarket.tags["osm"] == {"way_id": 100, "access": "destination", "motor_vehicle": "no"}
    assert "car" in lawnmarket.modes_permitted
    assert edge_between(graph, 7, 5).tags["osm"]["oneway"] == "-1"


def test_skips_non_roads_unknown_types_areas_and_off_road_nodes(graph):
    assert graph.num_edges == 12
    assert {9, 10, 12, 20, 21, 22, 23, 24, 30, 31, 32}.isdisjoint(graph.nodes)


def test_same_graph_whatever_the_element_order(graph, tmp_path):
    assert same_graph(build_graph(read_osm(write_osm(tmp_path / "reversed.json", ELEMENTS[::-1]))), graph)


def test_missing_way_node_raises_clear_error(tmp_path):
    path = write_osm(tmp_path / "osm.json", [el for el in ELEMENTS if el["id"] != 6])
    with pytest.raises(ValueError, match="missing 1 way nodes"):
        build_graph(read_osm(path))


def test_bus_stops_are_tagged_stops_in_id_order(osm):
    assert bus_stops(osm) == [list(LATLON[20]), list(LATLON[21]), list(LATLON[23])]


def test_street_names_cover_every_directed_edge_with_string_ids(graph):
    names = street_names(graph)
    assert set(names) == {str(e) for e in graph.edges}
    assert names[str(edge_between(graph, 1, 2).eid)] == names[str(edge_between(graph, 2, 1).eid)] == "Lawnmarket"
    assert names[str(edge_between(graph, 7, 8).eid)] == ""


def test_load_graph_keeps_signal_and_crossing_nodes_through_cleaning(osm_path, tmp_path):
    graph = load_graph(osm_path, tmp_path)
    assert {2, 3} <= set(graph.nodes) and graph.node(3).signalised


def test_load_graph_writes_hand_off_files_then_reloads_without_rebuilding(osm_path, osm, tmp_path, monkeypatch):
    out = tmp_path / "graph"
    graph = load_graph(osm_path, out)
    assert sorted(p.name for p in out.iterdir()) == ["bus_stops.json", "graph.pkl", "street_names.json"]
    assert json.loads((out / "street_names.json").read_text()) == street_names(graph)
    assert json.loads((out / "bus_stops.json").read_text()) == bus_stops(osm)
    assert same_graph(pickle.loads((out / "graph.pkl").read_bytes()), graph)

    monkeypatch.setattr("src.graph.build_graph.read_osm", lambda _: pytest.fail("rebuilt"))
    reloaded = load_graph(osm_path, out)
    assert same_graph(reloaded, graph)
