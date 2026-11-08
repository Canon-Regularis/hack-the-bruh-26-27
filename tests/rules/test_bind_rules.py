import csv
import json

import pytest

from src.graph.street_lookup import build_street_lookup
from src.rules import bind_rules as br
from tests.rules.rules_graph import build_graph

SOURCES = {"etro": "The Council ... QUOTE ... with exemptions."}


def clause(cid, street, restriction, **overrides):
    return {"clause_id": cid, "document_id": "etro", "page": 1, "quote": "QUOTE", "street": street,
            "restriction": restriction, "times": "at any time", "exemptions": [], **overrides}


@pytest.fixture(scope="module")
def graph():
    return build_graph()


@pytest.fixture(scope="module")
def lookup(graph):
    return build_street_lookup(graph)


APPLIED = [
    clause("lawn-1way", "Lawnmarket", "one_way", from_street="Castlehill", to_street="Bank Street"),
    clause("bow-1way", "Upper Bow", "one_way", from_street="Bank Street", to_street="Victoria Street"),
    clause("lawn-cars", "Lawnmarket", "no_motor_vehicles"),
    clause("bank-cars", "Bank Street", "no_motor_vehicles"),
    clause("castle-cars", "Castlehill", "no_motor_vehicles"),
    clause("vic-ped", "Victoria St", "pedestrian_only"),
]
ABSTAINED = [
    (clause("no-source", "Lawnmarket", "no_motor_vehicles", document_id="tro-99"), "source 'tro-99' is not cached"),
    (clause("bad-quote", "Lawnmarket", "no_motor_vehicles", quote="QUOTED"), "quote is not word for word in its source"),
    (clause("bad-type", "Lawnmarket", "no_entry"), "unknown restriction 'no_entry'"),
    (clause("exempt", "Lawnmarket", "no_motor_vehicles", exemptions="unspecified"), "exemptions are not listed"),
    (clause("timed", "Lawnmarket", "no_motor_vehicles", times="8am-6pm"),
     "time-limited (8am-6pm), but flags and OSM tags are all-day"),
    (clause("no-dir", "Lawnmarket", "one_way"), "one-way clause needs from_street and to_street"),
    (clause("no-street", "Johnston Terrace", "no_motor_vehicles"), "street 'Johnston Terrace' not found in the graph"),
    (clause("no-meet", "Lawnmarket", "one_way", from_street="Victoria Street", to_street="Bank Street"),
     "street doesn't meet 'Victoria Street'"),
]


@pytest.mark.parametrize("c, reason", ABSTAINED[:6])
def test_abstain_reason(c, reason):
    assert br.abstain_reason(c, SOURCES) == reason


def test_applicable_clauses_have_no_abstain_reason():
    assert all(br.abstain_reason(c, SOURCES) is None for c in APPLIED)


def eids(edges):
    return [e.eid for e in edges]


def test_bind_returns_the_street_in_both_directions(graph, lookup):
    assert eids(br.bind(APPLIED[2], graph, lookup)) == [0, 1, 2, 3]


def test_bind_one_way_returns_edges_against_the_allowed_direction(graph, lookup):
    assert eids(br.bind(APPLIED[0], graph, lookup)) == [1, 3]


def test_bind_one_way_already_right_on_osm_returns_nothing(graph, lookup):
    assert br.bind(APPLIED[1], graph, lookup) == []


@pytest.mark.parametrize("c, reason", ABSTAINED[6:])
def test_bind_raises_with_reason(graph, lookup, c, reason):
    with pytest.raises(br.BindError, match=reason):
        br.bind(c, graph, lookup)


@pytest.mark.parametrize("restriction, eid, expected", [
    ("no_motor_vehicles", 0, True),    # motor_vehicle=no
    ("no_motor_vehicles", 6, False),   # motor_vehicle=yes beats access=no
    ("no_motor_vehicles", 4, False),   # no tags on a car road
    ("no_motor_vehicles", 9, True),    # pedestrian street bans cars by default
    ("pedestrian_only", 9, True),
    ("pedestrian_only", 0, False),
    ("one_way", 1, False),             # a bound one-way edge is traffic OSM allows against the rule
    ("one_way", 9, True),              # cars can't use a pedestrian street either way
])
def test_osm_agrees(graph, restriction, eid, expected):
    assert br.osm_agrees(restriction, graph.edge(eid)) is expected


def test_check_clauses(graph):
    report = br.check_clauses(APPLIED + [c for c, _ in ABSTAINED], SOURCES, graph)
    assert report.applied == [c["clause_id"] for c in APPLIED]
    assert report.wrong_on_osm == ["lawn-1way", "bank-cars", "castle-cars"]
    assert report.abstained == [(c["clause_id"], reason) for c, reason in ABSTAINED]
    assert report.bound == {"one_way": {"1", "3"}, "no_motor_vehicles": {"0", "1", "2", "3", "4", "5", "6", "7"},
                            "pedestrian_only": {"9", "10"}}


def test_restriction_flags_one_row_per_directed_edge(graph):
    rows = br.restriction_flags(br.check_clauses(APPLIED, SOURCES, graph).bound, graph)
    assert [r["edge_id"] for r in rows] == [str(e) for e in graph.edges]
    assert rows[1] == {"edge_id": "1", "no_motor_vehicles": 1, "one_way": 1, "pedestrian_only": 0}
    assert rows[0] == {"edge_id": "0", "no_motor_vehicles": 1, "one_way": 0, "pedestrian_only": 0}
    assert rows[8] == {"edge_id": "8", "no_motor_vehicles": 0, "one_way": 0, "pedestrian_only": 0}


def test_main_writes_flags_and_prints_summary(graph, tmp_path, monkeypatch, capsys):
    clauses_path, sources_dir = tmp_path / "clauses.json", tmp_path / "sources"
    sources_dir.mkdir()
    clauses_path.write_text(json.dumps(APPLIED + [ABSTAINED[3][0], clause("bank-open", "Bank Street", "keep_open")]))
    (sources_dir / "etro.txt").write_text(SOURCES["etro"])
    monkeypatch.setattr(br, "CLAUSES_PATH", clauses_path)
    monkeypatch.setattr(br, "SOURCES_DIR", sources_dir)
    monkeypatch.setattr(br, "FLAGS_PATH", tmp_path / "rules" / "restriction_flags.csv")
    monkeypatch.setattr(br, "load_graph", lambda: graph)

    br.main()

    out = capsys.readouterr().out
    assert out.startswith("3 of 6 legal restrictions missing or wrong on OSM\n")
    assert "1 keep-open clauses bound, for the design filter\n" in out
    assert "1 abstained:\n  exempt: exemptions are not listed" in out
    with (tmp_path / "rules" / "restriction_flags.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == graph.num_edges and rows[1] == {"edge_id": "1", "no_motor_vehicles": "1", "one_way": "1",
                                                        "pedestrian_only": "0"}


def test_keep_open_clauses_are_bound_but_not_compared_with_osm(graph):
    keep_bank = clause("bank-open", "Bank Street", "keep_open")
    report = br.check_clauses(APPLIED + [keep_bank], SOURCES, graph)
    assert report.keep_open == {"bank-open": {"6", "7"}}
    assert "bank-open" not in report.applied and "bank-open" not in report.wrong_on_osm
    assert report.bound == br.check_clauses(APPLIED, SOURCES, graph).bound


def test_load_sources_skips_documents_not_cached(tmp_path):
    (tmp_path / "etro.txt").write_text("text")
    docs = [{"document_id": "etro"}, {"document_id": "tro-99"}, {"document_id": "etro"}]
    assert br.load_sources(docs, tmp_path) == {"etro": "text"}


def test_keep_open_skips_the_exemptions_and_all_day_checks():
    servicing = clause("servicing", "Bank Street", "keep_open", times="07:00-10:00", exemptions="unspecified")
    assert br.abstain_reason(servicing, SOURCES) is None
    assert br.abstain_reason({**servicing, "quote": "QUOTED"}, SOURCES) == "quote is not word for word in its source"


def test_load_clauses_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "clauses.json"
    path.write_text(json.dumps([clause("open", "Bank Street", "keep_open"), clause("open", "Castlehill", "keep_open")]))
    with pytest.raises(ValueError, match="clause ids must be unique"):
        br.load_clauses(path)


def test_report_keeps_each_bound_clauses_own_edges(graph):
    keep_bank = clause("bank-open", "Bank Street", "keep_open")
    report = br.check_clauses(APPLIED + [keep_bank] + [c for c, _ in ABSTAINED], SOURCES, graph)
    assert report.clause_edges == {
        "lawn-1way": {"1", "3"}, "bow-1way": set(), "lawn-cars": {"0", "1", "2", "3"},
        "bank-cars": {"6", "7"}, "castle-cars": {"4", "5"}, "vic-ped": {"9", "10"}, "bank-open": {"6", "7"},
    }


@pytest.fixture(scope="module")
def long_street():
    """Long Street runs 0-1-2-3; First Street meets it at 1, Second Street at 2; Far Street meets neither."""
    g = br.sn.RoadGraph()
    for nid, (lat, lon) in enumerate([(55.949, -3.200), (55.949, -3.199), (55.949, -3.198), (55.949, -3.197),
                                      (55.950, -3.199), (55.950, -3.198), (55.952, -3.190), (55.952, -3.189)]):
        g.add_node(nid, lat=lat, lon=lon)
    for u, v, name in ((0, 1, "Long Street"), (1, 2, "Long Street"), (2, 3, "Long Street"),
                       (1, 4, "First Street"), (2, 5, "Second Street"), (6, 7, "Far Street")):
        g.add_edge(u, v, oneway=False, street_name=name, osm={"way_id": u})   # 0-1, 2-3, 4-5, 6-7, 8-9, 10-11
    return g


def test_between_binds_only_the_stretch_between_the_two_junctions(long_street):
    edges = br.bind(clause("c", "Long Street", "no_motor_vehicles", between=["First Street", "Second Street"]),
                    long_street, build_street_lookup(long_street))
    assert sorted(e.eid for e in edges) == [2, 3]


def test_between_order_does_not_matter(long_street):
    edges = br.bind(clause("c", "Long Street", "no_motor_vehicles", between=["Second Street", "First Street"]),
                    long_street, build_street_lookup(long_street))
    assert sorted(e.eid for e in edges) == [2, 3]


def test_one_way_between_restricts_only_the_wrong_way_edge_of_the_stretch(long_street):
    edges = br.bind(clause("c", "Long Street", "one_way", from_street="First Street", to_street="Second Street",
                           between=["First Street", "Second Street"]), long_street, build_street_lookup(long_street))
    assert [(e.u, e.v) for e in edges] == [(2, 1)]


def test_between_a_street_that_does_not_meet_it_abstains(long_street):
    report = br.check_clauses([clause("c", "Long Street", "no_motor_vehicles", between=["First Street", "Far Street"])],
                              SOURCES, long_street)
    assert report.abstained == [("c", "street doesn't meet 'Far Street'")]
    assert report.bound["no_motor_vehicles"] == set()


@pytest.mark.parametrize("between", [["First Street"], "First Street", ["First Street", ""], [1, 2]])
def test_malformed_between_abstains(between):
    assert br.abstain_reason(clause("c", "Long Street", "no_motor_vehicles", between=between), SOURCES) == \
        "between needs two street names"
