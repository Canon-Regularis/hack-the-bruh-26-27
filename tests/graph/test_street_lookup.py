import pytest

from engine.orchid import street_network as sn
from src.graph.build_graph import street_names
from src.graph.street_lookup import build_street_lookup, find_streets, normalise_name


@pytest.mark.parametrize("raw, expected", [
    ("Johnston Ter", "johnston terrace"),
    ("Johnston Terrace", "johnston terrace"),
    ("St Mary's St", "saint marys street"),
    ("St. Mary’s Street", "saint marys street"),
    ("St Giles' Street", "saint giles street"),
    ("High St.", "high street"),
    ("  LAWNMARKET ", "lawnmarket"),
    ("Upper Bow", "upper bow"),
    ("", ""),
])
def test_normalise_name(raw, expected):
    assert normalise_name(raw) == expected


@pytest.fixture(scope="module")
def graph():
    g = sn.RoadGraph()
    for i, (lat, lon) in enumerate([(55.9490, -3.1990), (55.9492, -3.1980), (55.9494, -3.1970), (55.9500, -3.1975)]):
        g.add_node(i, lat=lat, lon=lon)
    g.add_edge(0, 1, oneway=False, street_name="Lawnmarket")
    g.add_edge(1, 2, oneway=False, street_name="Lawnmarket")
    g.add_edge(1, 3, street_name="Upper Bow")
    g.add_edge(2, 3)
    return g


@pytest.fixture(scope="module")
def lookup(graph):
    return build_street_lookup(graph)


def test_lookup_collects_both_directions_and_skips_unnamed(lookup):
    assert lookup == {"lawnmarket": ["0", "1", "2", "3"], "upper bow": ["4"]}


def test_lookup_agrees_with_street_names_json(graph, lookup):
    names = street_names(graph)
    assert all(normalise_name(names[eid]) == key for key, eids in lookup.items() for eid in eids)
    assert {eid for eids in lookup.values() for eid in eids} == {eid for eid, name in names.items() if name}


def test_find_streets_matches_variants_and_reports_unresolved(lookup):
    found, unresolved = find_streets(lookup, ["LAWNMARKET", "Upper Bow", "Bow", "Castlehill", ""])
    assert found == {"LAWNMARKET": ["0", "1", "2", "3"], "Upper Bow": ["4"]}
    assert unresolved == ["Bow", "Castlehill", ""]
