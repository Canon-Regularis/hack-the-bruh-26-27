import json
import math

import pytest

from engine.orchid import street_network as sn
from src.decide import scenarios as sc


@pytest.fixture(scope="module")
def graph():
    """A short two-way Johnston Terrace beside an unrelated one-way street."""
    g = sn.RoadGraph()
    for n, lon in enumerate((-3.200, -3.198, -3.196)):
        g.add_node(n, lat=55.948, lon=lon)
    g.add_node(3, lat=55.950, lon=-3.198)
    g.add_edge(0, 1, oneway=False, highway_class="tertiary", street_name="Johnston Terrace")
    g.add_edge(1, 2, oneway=False, highway_class="tertiary", street_name="Johnston Terrace")
    g.add_edge(1, 3, highway_class="residential", street_name="Castle Wynd")
    return g


def test_full_set_crosses_every_level_with_the_closure():
    full = sc.full_scenarios()
    assert len(full) == 2 * len(sc.DEMAND_LEVELS)
    assert math.isclose(math.fsum(s["probability"] for s in full), 1.0)
    peak_closed = next(s for s in full if s["id"] == "peak-closed")
    assert peak_closed["multiplier"] == 1.25 and peak_closed["closed"]
    assert math.isclose(peak_closed["probability"], 0.20 * sc.CLOSURE_PROBABILITY)


def test_full_set_drops_impossible_scenarios_and_rejects_bad_weights():
    assert all(not s["closed"] for s in sc.full_scenarios(closure_probability=0.0))
    with pytest.raises(ValueError):
        sc.full_scenarios(levels={"a": (1.0, 0.5)})
    with pytest.raises(ValueError):
        sc.full_scenarios(closure_probability=1.5)


def test_scaling_puts_both_components_on_zero_to_one():
    vectors = sc.scaled_vectors(sc.full_scenarios())
    assert min(v[0] for v in vectors) == 0.0 and max(v[0] for v in vectors) == 1.0
    assert {v[1] for v in vectors} == {0.0, 1.0}


def test_reduction_keeps_k_members_with_probabilities_summing_to_one():
    full = sc.full_scenarios()
    kept, distance = sc.reduce_scenarios(full, k=4)
    assert len(kept) == 4
    assert math.isclose(math.fsum(s["probability"] for s in kept), 1.0)
    assert {s["id"] for s in kept} <= {s["id"] for s in full}
    assert distance > 0


def test_reduction_without_dropping_is_exact():
    full = sc.full_scenarios()
    kept, distance = sc.reduce_scenarios(full, k=len(full))
    assert distance == 0
    assert {s["id"]: s["probability"] for s in kept} == pytest.approx({s["id"]: s["probability"] for s in full})


def test_reduction_to_one_keeps_all_the_mass():
    kept, _ = sc.reduce_scenarios(sc.full_scenarios(), k=1)
    assert len(kept) == 1 and math.isclose(kept[0]["probability"], 1.0)


def test_reduction_rejects_k_out_of_range():
    with pytest.raises(ValueError):
        sc.reduce_scenarios(sc.full_scenarios(), k=0)


def test_closure_edges_are_both_directions_as_strings(graph):
    ids = sc.closure_edges(graph)
    johnston = sorted((str(e.eid) for e in graph.iter_edges() if e.street_name == "Johnston Terrace"), key=int)
    assert ids == johnston and len(ids) == 4


def test_closure_edges_raise_when_the_street_is_missing(graph):
    with pytest.raises(ValueError):
        sc.closure_edges(graph, street="Nowhere Street")


def test_main_writes_kept_scenarios_with_closed_edges(graph, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sc, "SCENARIOS_PATH", tmp_path / "demand" / "scenarios.json")
    monkeypatch.setattr(sc, "ROOT", tmp_path)
    monkeypatch.setattr(sc, "load_graph", lambda: graph)
    sc.main()
    out = json.loads((tmp_path / "demand" / "scenarios.json").read_text(encoding="utf-8"))
    assert out["label"].startswith("synthetic")
    assert len(out["kept"]) == sc.N_KEPT
    assert math.isclose(math.fsum(s["probability"] for s in out["kept"]), 1.0)
    for s in out["kept"] + out["full"]:
        assert s["closed_edge_ids"] == (out["closure"]["edge_ids"] if s["closed"] else [])
    assert "4 of 8 scenarios kept" in capsys.readouterr().out
