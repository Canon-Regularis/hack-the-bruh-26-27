import json

import pytest

from engine.orchid import street_network as sn
from src.decide import interventions as iv
from src.decide.network import link_maps, to_network
from src.decide.pedestrian_crossing_delay import find_crossings

ZEBRA = {"highway": "crossing", "crossing": "uncontrolled"}
PUFFIN = {"highway": "crossing", "crossing": "traffic_signals"}

CANDIDATES = [
    {"id": "close-jt", "kind": "close", "street": "Johnston Terrace"},
    {"id": "bus-lane-lothian", "kind": "reallocate", "street": "Lothian Road", "capacity_factor": 0.5},
    {"id": "signal-lothian", "kind": "signalise_crossing", "node": 1, "capacity_factor": 0.9},
    {"id": "close-ghost", "kind": "close", "street": "Ghost Street"},
    {"id": "signal-puffin", "kind": "signalise_crossing", "node": 4},
    {"id": "widen-bank", "kind": "reallocate", "street": "Bank Street", "capacity_factor": 1.2},
    {"id": "signal-victoria", "kind": "signalise_crossing", "node": 5},
]


@pytest.fixture(scope="module")
def graph():
    """Lothian Road 0-1-2 with a zebra at 1, Johnston Terrace 2-3, Bank Street 3-4 with a puffin at 4,
    Victoria Street 4-5 with a zebra at 5. All two-way."""
    g = sn.RoadGraph()
    for nid in range(6):
        g.add_node(nid, lat=55.9490, lon=-3.2000 + 0.001 * nid, osm={1: ZEBRA, 4: PUFFIN, 5: ZEBRA}.get(nid, {}))
    g.add_edge(0, 1, oneway=False, street_name="Lothian Road", highway_class="primary")      # 0, 1
    g.add_edge(1, 2, oneway=False, street_name="Lothian Road", highway_class="primary")      # 2, 3
    g.add_edge(2, 3, oneway=False, street_name="Johnston Terrace", highway_class="tertiary")  # 4, 5
    g.add_edge(3, 4, oneway=False, street_name="Bank Street", highway_class="tertiary")       # 6, 7
    g.add_edge(4, 5, oneway=False, street_name="Victoria Street", highway_class="tertiary")   # 8, 9
    return g


@pytest.fixture(scope="module")
def crossings(graph):
    return find_crossings(graph)


@pytest.fixture(scope="module")
def loaded(graph, crossings, tmp_path_factory):
    path = tmp_path_factory.mktemp("decide") / "candidates.json"
    path.write_text(json.dumps(CANDIDATES))
    return iv.load_candidates(path, graph, crossings)


@pytest.fixture(scope="module")
def found(loaded):
    return {c["id"]: c for c in loaded[0]}


@pytest.fixture(scope="module")
def base(graph):
    net, _ = to_network(graph)
    return net, link_maps(net)[1]


def capacities(net):
    return {link.name: link.capacity for link in net.links}


def test_load_candidates_resolves_edges_and_notes_what_failed(loaded, found):
    assert list(found) == ["close-jt", "bus-lane-lothian", "signal-lothian", "widen-bank", "signal-victoria"]
    assert found["close-jt"]["edges"] == {"4", "5"}
    assert found["bus-lane-lothian"]["edges"] == {"0", "1", "2", "3"}
    assert found["signal-lothian"]["edges"] == {"0", "3"} and found["signal-lothian"]["capacity_factor"] == 0.9
    assert found["signal-victoria"]["edges"] == {"8"} and found["signal-victoria"]["capacity_factor"] is None
    assert loaded[1] == ["street not found: 'Ghost Street'", "node 4 is not an unsignalised crossing (signal-puffin)"]


@pytest.mark.parametrize("raw, message", [
    ([CANDIDATES[0], CANDIDATES[0]], "candidate ids must be unique"),
    ([{"id": "x", "kind": "widen", "street": "Bank Street"}], r"unknown kinds \['widen'\]"),
    ([{"id": "x", "kind": "reallocate", "street": "Bank Street"}], r"reallocate candidates need a capacity_factor: \['x'\]"),
])
def test_load_candidates_rejects_bad_files(graph, crossings, tmp_path, raw, message):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=message):
        iv.load_candidates(path, graph, crossings)


def test_filter_excludes_only_cuts_on_keep_open_edges(loaded):
    keep_open = {"jt-open": {"4", "5"}, "lothian-open": {"1"}, "lothian-open-2": {"2"},
                 "lothian-open-3": {"3"}, "bank-open": {"6"}, "victoria-open": {"8"}}
    allowed, excluded = iv.filter_candidates(loaded[0], keep_open)
    assert [c["id"] for c in allowed] == ["widen-bank", "signal-victoria"]
    assert excluded == [("close-jt", "jt-open"),
                        ("bus-lane-lothian", "lothian-open"), ("bus-lane-lothian", "lothian-open-2"),
                        ("bus-lane-lothian", "lothian-open-3"),
                        ("signal-lothian", "lothian-open-3")]


def test_apply_design_drops_closures_once_and_keeps_ids_in_step(base, crossings, found):
    net, link_to_edge = base
    new_net, links, new_crossings = iv.apply_design(net, link_to_edge, crossings, [found["close-jt"]],
                                                    frozenset({"8", "9"}))
    assert links == ["0", "1", "2", "3", "6", "7"]
    assert [link.name for link in new_net.links] == links
    assert new_net.n_nodes == net.n_nodes and new_crossings == crossings


def test_apply_design_scales_with_post_drop_indices(base, crossings, found):
    net, link_to_edge = base
    before = capacities(net)
    after = capacities(iv.apply_design(net, link_to_edge, crossings, [found["close-jt"], found["widen-bank"]])[0])
    assert after["6"] == pytest.approx(before["6"] * 1.2) and after["7"] == pytest.approx(before["7"] * 1.2)
    assert {eid: after[eid] for eid in ("0", "1", "2", "3", "8", "9")} == \
           {eid: before[eid] for eid in ("0", "1", "2", "3", "8", "9")}


def test_apply_design_skips_scaling_edges_that_are_closed(base, crossings, found):
    net, link_to_edge = base
    before = capacities(net)
    new_net, links, _ = iv.apply_design(net, link_to_edge, crossings, [found["widen-bank"]], ["6"])
    assert "6" not in links and capacities(new_net)["7"] == pytest.approx(before["7"] * 1.2)


def test_apply_design_multiplies_capacity_factors_on_shared_edges(base, crossings, found):
    net, link_to_edge = base
    before = capacities(net)
    after = capacities(iv.apply_design(net, link_to_edge, crossings,
                                       [found["bus-lane-lothian"], found["signal-lothian"]])[0])
    assert after["0"] == pytest.approx(before["0"] * 0.5 * 0.9)
    assert after["1"] == pytest.approx(before["1"] * 0.5)
    assert after["3"] == pytest.approx(before["3"] * 0.5 * 0.9)
    assert after["6"] == before["6"]


def test_apply_design_signalises_only_the_chosen_crossing(base, crossings, found):
    net, link_to_edge = base
    new_net, _, new_crossings = iv.apply_design(net, link_to_edge, crossings, [found["signal-victoria"]])
    assert {c["node"]: c["signalised"] for c in new_crossings} == {1: False, 4: True, 5: True}
    assert capacities(new_net) == capacities(net)


def test_apply_design_leaves_its_inputs_unchanged(base, graph, found):
    net, link_to_edge = base
    crossings = find_crossings(graph)
    expected_links, expected_crossings = list(link_to_edge), find_crossings(graph)
    iv.apply_design(net, link_to_edge, crossings, list(found.values()), frozenset({"8", "9"}))
    assert net.n_links == len(expected_links) and link_to_edge == expected_links
    assert crossings == expected_crossings


def clause(cid, street, document_id="etro"):
    return {"clause_id": cid, "document_id": document_id, "page": 1, "quote": "QUOTE", "street": street,
            "restriction": "keep_open", "times": "07:00-10:00", "exemptions": "unspecified"}


def test_main_writes_allowed_excluded_unresolved_and_abstained(graph, tmp_path, monkeypatch, capsys):
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps(CANDIDATES))
    clauses = tmp_path / "clauses.json"
    clauses.write_text(json.dumps([clause("jt-open", "Johnston Terrace"),
                                   clause("lothian-open", "Lothian Road", document_id="tro-99")]))
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "etro.txt").write_text("... QUOTE ...")
    monkeypatch.setattr(iv, "CANDIDATES_PATH", candidates)
    monkeypatch.setattr(iv, "CHECK_PATH", tmp_path / "decide" / "candidate_check.json")
    monkeypatch.setattr(iv, "CLAUSES_PATH", clauses)
    monkeypatch.setattr(iv, "SOURCES_DIR", sources)
    monkeypatch.setattr(iv, "load_graph", lambda: graph)

    iv.main()

    check = json.loads((tmp_path / "decide" / "candidate_check.json").read_text())
    assert check["allowed"] == ["bus-lane-lothian", "signal-lothian", "widen-bank", "signal-victoria"]
    assert check["excluded"] == [{"candidate": "close-jt", "clause": "jt-open"}]
    assert len(check["unresolved"]) == 2
    assert check["abstained_keep_open"] == [{"clause": "lothian-open", "reason": "source 'tro-99' is not cached"}]
    out = capsys.readouterr().out
    assert out.startswith("4 candidates allowed, 1 excluded by keep-open clauses")
    assert "keep-open clause lothian-open abstained (source 'tro-99' is not cached), so it forbids nothing" in out


def test_load_candidates_drops_streets_without_car_edges(tmp_path):
    g = sn.RoadGraph()
    for nid in range(3):
        g.add_node(nid, lat=55.9490, lon=-3.2000 + 0.001 * nid)
    g.add_edge(0, 1, oneway=False, street_name="Royal Mile", highway_class="pedestrian")
    g.add_edge(1, 2, oneway=False, street_name="Bank Street", highway_class="tertiary")
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps([{"id": "close-rm", "kind": "close", "street": "Royal Mile"},
                                {"id": "close-bank", "kind": "close", "street": "Bank Street"}]))
    candidates, notes = iv.load_candidates(path, g, find_crossings(g))
    assert [c["id"] for c in candidates] == ["close-bank"]
    assert notes == ["'Royal Mile' has no car edges, so close-rm changes nothing"]
