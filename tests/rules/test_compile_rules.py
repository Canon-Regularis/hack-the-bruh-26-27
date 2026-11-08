import csv
import json

import pytest

from engine.orchid import policy_compiler as pc
from engine.orchid import street_network as sn
from src.rules import bind_rules as br
from src.rules import compile_rules as cr

SOURCES = {"etro": "The Council ... QUOTE ... with exemptions."}


def clause(cid, street, restriction, **overrides):
    return {"clause_id": cid, "document_id": "etro", "page": 1, "quote": "QUOTE", "street": street,
            "restriction": restriction, "times": "at any time", "exemptions": [], **overrides}


@pytest.fixture(scope="module")
def graph():
    g = sn.RoadGraph()
    for nid, lon in enumerate([-3.2010, -3.2000, -3.1990, -3.1980, -3.1970]):
        g.add_node(nid, lat=55.9490, lon=lon)
    g.add_node(5, lat=55.9500, lon=-3.1980)
    g.add_node(6, lat=55.9510, lon=-3.1980)
    no_cars = {"way_id": 1, "motor_vehicle": "no"}
    g.add_edge(1, 2, oneway=False, street_name="Lawnmarket", osm=no_cars)                 # 0, 1
    g.add_edge(2, 3, oneway=False, street_name="Lawnmarket", osm=no_cars)                 # 2, 3
    g.add_edge(0, 1, oneway=False, street_name="Castlehill", osm={"way_id": 2})           # 4, 5
    g.add_edge(3, 4, oneway=False, street_name="Bank Street",
               osm={"way_id": 3, "motor_vehicle": "yes", "access": "no"})                 # 6, 7
    g.add_edge(3, 5, street_name="Upper Bow", osm={"way_id": 4, "oneway": "yes"})         # 8
    g.add_edge(5, 6, oneway=False, street_name="Victoria Street", highway_class="pedestrian",
               osm={"way_id": 5})                                                         # 9, 10
    g.add_edge(4, 6, street_name="", osm={"way_id": 6})                                   # 11, unnamed
    return g


APPLIED = [
    clause("lawn-1way", "Lawnmarket", "one_way", from_street="Castlehill", to_street="Bank Street"),
    clause("bow-1way", "Upper Bow", "one_way", from_street="Bank Street", to_street="Victoria Street"),
    clause("lawn-cars", "Lawnmarket", "no_motor_vehicles"),
    clause("bank-cars", "Bank Street", "no_motor_vehicles"),
    clause("castle-cars", "Castlehill", "no_motor_vehicles"),
    clause("vic-ped", "Victoria St", "pedestrian_only"),
]
ABSTAINED = [
    (clause("no-source", "Lawnmarket", "no_motor_vehicles", document_id="tro-99"),
     "span_unverified: source document 'tro-99' missing"),
    (clause("bad-quote", "Lawnmarket", "no_motor_vehicles", quote="QUOTED"), "span_unverified: quote not found in source"),
    (clause("bad-type", "Lawnmarket", "no_entry"), "unrepresentable: unknown restriction 'no_entry'"),
    (clause("exempt", "Lawnmarket", "no_motor_vehicles", exemptions="unspecified"),
     "unrepresentable: exemptions are not listed"),
    (clause("timed-odd", "Lawnmarket", "no_motor_vehicles", times="8.00-6.30"),
     "unrepresentable: time window '8.00-6.30': a range with no am / pm and no 24-hour clock is ambiguous."),
    (clause("no-dir", "Lawnmarket", "one_way"), "unrepresentable: one-way clause needs from_street and to_street"),
    (clause("no-street", "Johnston Terrace", "no_motor_vehicles"),
     "binding_unresolved: no admissible street (type_mismatch)"),
    (clause("no-meet", "Lawnmarket", "one_way", from_street="Victoria Street", to_street="Bank Street"),
     "binding_unresolved: street doesn't meet 'Victoria Street'"),
]
TIMED = clause("timed", "Lawnmarket", "no_motor_vehicles", times="8am-6pm")
ALL = APPLIED + [c for c, _ in ABSTAINED] + [TIMED]


@pytest.fixture(scope="module")
def compiled(graph):
    return cr.compile_rules(ALL, SOURCES, graph)


def test_gazetteer_has_one_entry_per_street_with_string_edge_ids(graph):
    gz = cr.build_gazetteer(graph)
    assert gz.ids == ["bank street", "castlehill", "lawnmarket", "upper bow", "victoria street"]
    assert gz.get("lawnmarket").edge_ids == ("0", "1", "2", "3")
    assert all(isinstance(e, str) for e in gz.all_edge_ids) and "11" not in gz.all_edge_ids
    assert gz.edge_endpoints["1"] == ("2", "1")
    assert gz.node_coords["3"] == graph.node(3).xy


def test_gazetteer_merges_names_policy_compiler_reads_as_one_street():
    g = sn.RoadGraph()
    for nid in range(3):
        g.add_node(nid, x=100.0 * nid, y=0.0)
    g.add_edge(0, 1, street_name="Bank St", osm={})
    g.add_edge(1, 2, street_name="Bank Street", osm={})
    gz = cr.build_gazetteer(g)
    assert gz.ids == ["bank street"] and gz.get("bank street").edge_ids == ("0", "1")
    assert gz.resolve("Bank Street").status == "bound"


def test_exact_name_binds_next_to_a_close_name_and_a_typo_still_resolves():
    g = sn.RoadGraph()
    for nid in range(3):
        g.add_node(nid, x=100.0 * nid, y=0.0)
    g.add_edge(0, 1, street_name="Queen Street", osm={})
    g.add_edge(1, 2, street_name="Queens Street", osm={})
    assert pc.Gazetteer(cr.build_gazetteer(g).entries()).resolve("Queen Street").status == "ambiguous"
    gz = cr.build_gazetteer(g, ["Queen St"])
    assert [gz.resolve(n).target_id for n in ("Queen Street", "Queen St", "Queens St.")] == [
        "queen street", "queen street", None]   # "Queens St." was not pinned: the resolver finds it ambiguous


def test_hard_ban_closes_the_street_in_both_directions(compiled):
    fixings = compiled.result.fixings.to_dict()
    assert fixings["fixedZero"] == ["0", "1", "10", "2", "3", "4", "5", "6", "7", "9"]
    assert fixings["fixedOne"] == []
    assert fixings["provenance"]["fixedZero:0"] == ["lawn-cars"]
    assert fixings["provenance"]["fixedZero:1"] == ["lawn-1way", "lawn-cars"]


def test_abstentions_in_input_order_with_reasons(compiled):
    assert cr.abstained(compiled) == [(c["clause_id"], reason) for c, reason in ABSTAINED]


def test_refused_clauses_still_get_a_policy_check_row(graph, compiled):
    rows, _ = cr.check_design([], compiled, graph)
    status = {r["clause_id"]: (r["status"], r["detail"]) for r in rows}
    assert status["timed-odd"] == ("abstained", "unrepresentable")
    assert status["no-meet"] == ("abstained", "binding_unresolved")
    assert status["bow-1way"] == ("not_applicable", "no_operation")   # OSM already one-way the right way


def test_parsable_time_window_compiles_as_conditional_not_into_the_flags(compiled):
    fixings = compiled.result.fixings.to_dict()
    assert cr.conditional(compiled) == ["timed"]
    assert fixings["conditional"][0]["predicate"]["timeWindows"] == [[480, 1080]]
    assert fixings["provenance"]["conditional:0"] == ["timed"]
    assert "timed" not in cr.applied(compiled)


def test_timed_one_way_with_nothing_to_close_stays_out_of_the_osm_count(graph):
    c = cr.compile_rules([clause("bow-timed", "Upper Bow", "one_way", from_street="Bank Street",
                                 to_street="Victoria Street", times="Mon-Fri 8am-6pm")], SOURCES, graph)
    assert (cr.applied(c), cr.conditional(c), cr.abstained(c)) == ([], ["bow-timed"], [])
    assert cr.mismatch_line(c, graph) == "No legal restrictions missing or wrong on OSM (0 checked)"
    assert cr.outcome(c, "bow-timed") == ("compiled for its time window only (not in the all-day flags or the OSM "
                                          "check) (nothing to close)")


@pytest.mark.parametrize("times", ["", "  ", "no waiting", "no entry", "at all times", "at any time."])
def test_time_field_read_as_all_day_but_not_the_all_day_phrase_is_refused(graph, times):
    c = cr.compile_rules([clause("t", "Bank Street", "no_motor_vehicles", times=times)], SOURCES, graph)
    assert cr.abstained(c) == [("t", f"unrepresentable: time window {times!r} is not 'at any time' or a time window")]
    assert c.result.fixings.to_dict()["fixedZero"] == [] and cr.applied(c) == []
    assert br.check_clauses([clause("t", "Bank Street", "no_motor_vehicles", times=times)], SOURCES, graph).applied == []


def test_all_day_phrase_ignores_case_and_padding_like_the_glue(graph):
    c = cr.compile_rules([clause("t", "Bank Street", "no_motor_vehicles", times=" At Any Time ")], SOURCES, graph)
    assert cr.applied(c) == ["t"] and c.result.fixings.to_dict()["fixedZero"] == ["6", "7"]


def test_one_way_ids_that_normalise_alike_both_bind(graph):
    clauses = [clause(cid, "Lawnmarket", "one_way", from_street="Castlehill", to_street="Bank Street")
               for cid in ("art.3", "art-3")]
    c = cr.compile_rules(clauses, SOURCES, graph)
    assert cr.abstained(c) == [] and cr.applied(c) == ["art.3", "art-3"]
    assert cr.compare_with_glue(clauses, SOURCES, graph, c) == {
        "clauses": {}, "wrong_on_osm": {"glue_only": [], "compiler_only": []}, "flags": []}


@pytest.mark.parametrize("ids", [("a", "a"), (" a",), ("",)])
def test_duplicate_or_padded_clause_ids_raise_a_clear_error(graph, ids):
    with pytest.raises(ValueError, match="clause ids must be unique, non-empty and unpadded: " + repr(ids[0])):
        cr.compile_rules([clause(cid, "Lawnmarket", "no_motor_vehicles") for cid in ids], SOURCES, graph)


@pytest.mark.parametrize(("overrides", "reason"), [
    ({"quote": "QUOTED"}, "span_unverified: quote not found in source"),
    ({"document_id": "tro-99", "exemptions": "unspecified"}, "span_unverified: source document 'tro-99' missing"),
])
def test_refused_clause_shows_only_its_first_reason(graph, overrides, reason):
    c = cr.compile_rules([clause("both", "Lawnmarket", "no_entry", **overrides)], SOURCES, graph)
    assert cr.abstained(c) == [("both", reason)]


@pytest.fixture(scope="module")
def queen_graph():
    g = sn.RoadGraph()
    for nid in range(4):
        g.add_node(nid, x=100.0 * nid, y=0.0)
    g.add_node(4, x=0.0, y=100.0)
    g.add_edge(0, 1, oneway=False, street_name="Queen Street", osm={})    # 0, 1
    g.add_edge(1, 2, oneway=False, street_name="Queens Street", osm={})   # 2, 3
    g.add_edge(2, 3, oneway=False, street_name="Low Street", osm={})      # 4, 5
    g.add_edge(0, 4, oneway=False, street_name="Hill Street", osm={})     # 6, 7
    return g


def test_ambiguous_one_way_street_is_queued_with_its_candidates(queen_graph):
    c = cr.compile_rules([clause("amb", "Quen Street", "one_way", from_street="Low Street",
                                 to_street="Hill Street")], {"etro": "QUOTE"}, queen_graph)
    assert cr.abstained(c) == [("amb", "binding_ambiguous: street 'Quen Street': 2 admissible entries")]
    assert [i.candidates for i in c.result.queue.items()] == [("queen street", "queens street")]


def test_compile_rules_pins_the_clauses_street_names(queen_graph):
    c = cr.compile_rules([clause("qs", "Queen St", "no_motor_vehicles")], {"etro": "QUOTE"}, queen_graph)
    assert c.result.fixings.to_dict()["fixedZero"] == ["0", "1"] and cr.abstained(c) == []


@pytest.mark.parametrize(("quote", "strength", "expected"), [
    ("Vehicles should avoid", "hard", [("w", "recommendation", "should")]),
    ("Vehicles should avoid", "soft", []),
    ("Drivers are encouraged not to enter", "hard", [("w", "neutral", "encourage")]),
    ("Vehicles may not enter", "hard", []),
])
def test_wording_gate_flags(graph, quote, strength, expected):
    c = cr.compile_rules([clause("w", "Bank Street", "no_motor_vehicles", quote=quote, strength=strength)],
                         {"etro": quote}, graph)
    assert cr.wording_gate_flags(c) == expected


def test_wording_gate_flags_skip_clauses_the_gate_never_reaches(graph):
    sources = {"etro": "Vehicles should not use Bank Street."}
    c = cr.compile_rules([clause("refused", "Bank Street", "no_entry", quote="Vehicles should not use Bank Street"),
                          clause("unverified", "Bank Street", "no_motor_vehicles", quote="Vehicles should not drive")],
                         sources, graph)
    assert [cid for cid, _ in cr.abstained(c)] == ["refused", "unverified"]
    assert cr.wording_gate_flags(c) == []


def test_compare_with_glue_reports_osm_and_flag_differences(graph):
    diff = cr.compare_with_glue([clause("typo", "Castlehil", "no_motor_vehicles")], SOURCES, graph)
    assert diff == {"clauses": {"typo": {"glue": "abstained (street 'Castlehil' not found in the graph)",
                                         "compiler": "applied"}},
                    "wrong_on_osm": {"glue_only": [], "compiler_only": ["typo"]}, "flags": ["4", "5"]}


def test_soft_clause_never_compiles(graph):
    soft = cr.compile_rules([clause("soft", "Castlehill", "no_motor_vehicles", strength="soft")], SOURCES, graph)
    assert soft.result.skipped == {"soft": "soft_clause"} and soft.result.fixings.to_dict()["fixedZero"] == []


def test_mismatch_and_flags_match_the_glue(graph, compiled):
    report = br.check_clauses(ALL, SOURCES, graph)
    assert cr.applied(compiled) == report.applied
    assert cr.wrong_on_osm(compiled, graph) == report.wrong_on_osm == ["lawn-1way", "bank-cars", "castle-cars"]
    assert cr.mismatch_line(compiled, graph) == "3 of 6 legal restrictions missing or wrong on OSM"
    assert cr.bound(compiled) == report.bound
    assert cr.restriction_flags(compiled, graph) == br.restriction_flags(report.bound, graph)


def test_compare_with_glue_lists_only_the_time_window(graph, compiled):
    diff = cr.compare_with_glue(ALL, SOURCES, graph, compiled)
    assert diff == {"clauses": {"timed": {
        "glue": "abstained (time-limited (8am-6pm), but flags and OSM tags are all-day)",
        "compiler": "compiled for its time window only (not in the all-day flags or the OSM check)"}},
        "wrong_on_osm": {"glue_only": [], "compiler_only": []}, "flags": []}


def test_flags_one_row_per_directed_edge(graph, compiled):
    rows = cr.restriction_flags(compiled, graph)
    assert [r["edge_id"] for r in rows] == [str(e) for e in graph.edges]
    assert rows[1] == {"edge_id": "1", "no_motor_vehicles": 1, "one_way": 1, "pedestrian_only": 0}
    assert rows[0] == {"edge_id": "0", "no_motor_vehicles": 1, "one_way": 0, "pedestrian_only": 0}
    assert rows[9] == {"edge_id": "9", "no_motor_vehicles": 0, "one_way": 0, "pedestrian_only": 1}
    assert rows[8] == {"edge_id": "8", "no_motor_vehicles": 0, "one_way": 0, "pedestrian_only": 0}


def test_design_that_opens_a_banned_edge_is_in_tension(graph):
    c = cr.compile_rules([clause("bank-cars", "Bank Street", "no_motor_vehicles")], SOURCES, graph)
    rows, ratio = cr.check_design({"7"}, c, graph)
    assert rows == [{"clause_id": "bank-cars", "status": "in_tension", "quote": "QUOTE", "page": 1,
                     "violated": ["fixedZero:6"], "detail": "unconditional"}]
    assert ratio == 0.0
    rows, ratio = cr.check_design({"6", "7"}, c, graph)
    assert rows[0]["status"] == "satisfied" and ratio == 1.0


def test_design_lists_every_edge(graph):
    design = cr.design_from_closed(["6"], graph)
    assert design.z == {str(e): int(e != 6) for e in graph.edges}
    with pytest.raises(ValueError, match="unknown edge ids"):
        cr.design_from_closed(["99"], graph)


def test_conditional_clause_checked_at_a_moment(graph):
    c = cr.compile_rules([clause("timed", "Castlehill", "no_motor_vehicles", times="8am-6pm")], SOURCES, graph)
    rows, _ = cr.check_design(set(), c, graph, pc.Scenario(20 * 60, "mon"))
    assert rows[0]["status"] == "not_applicable"
    rows, _ = cr.check_design(set(), c, graph, pc.Scenario(10 * 60, "mon"))
    assert (rows[0]["status"], rows[0]["violated"]) == ("in_tension", ["fixedZero:4", "fixedZero:5"])


def test_status_quo_satisfies_the_bans_osm_already_shows(graph):
    c = cr.compile_rules([clause("lawn-cars", "Lawnmarket", "no_motor_vehicles"),
                          clause("vic-ped", "Victoria St", "pedestrian_only")], SOURCES, graph)
    assert cr.mismatch_line(c, graph) == "No legal restrictions missing or wrong on OSM (2 checked)"
    assert cr.osm_closed_edges(graph) == {"0", "1", "2", "3", "9", "10"}
    rows, ratio = cr.check_design(set(), c, graph)
    assert [(r["clause_id"], r["status"]) for r in rows] == [("lawn-cars", "satisfied"), ("vic-ped", "satisfied")]
    assert ratio == 1.0


def test_result_does_not_depend_on_the_quote_wording(graph):
    sources = {"etro": "No motor vehicle shall enter Bank Street. Vehicles should not use Bank Street."}
    fixings = []
    for quote in ("No motor vehicle shall enter Bank Street", "Vehicles should not use Bank Street"):
        c = cr.compile_rules([clause("bank-cars", "Bank Street", "no_motor_vehicles", quote=quote)], sources, graph)
        fixings.append(c.result.fixings.to_dict())
        assert c.result.wording["bank-cars"].modality == "obligation"
    assert fixings[0] == fixings[1] and fixings[0]["fixedZero"] == ["6", "7"]
    assert cr.wording_gate_flags(c) == [("bank-cars", "discouragement", "should not")]


def test_schema_refusal_is_queued_not_raised(graph):
    c = cr.compile_rules([clause("no-page", "Bank Street", "no_motor_vehicles", page=0)], SOURCES, graph)
    assert c.result.clauses == {}
    assert cr.abstained(c) == [("no-page", "unrepresentable: invalid clause 'no-page': /page: 0 < minimum 1")]


def _setup_main(clauses, graph, tmp_path, monkeypatch):
    clauses_path, sources_dir = tmp_path / "clauses.json", tmp_path / "sources"
    sources_dir.mkdir()
    clauses_path.write_text(json.dumps(clauses))
    (sources_dir / "etro.txt").write_text(SOURCES["etro"])
    for module in (br, cr):
        monkeypatch.setattr(module, "CLAUSES_PATH", clauses_path)
        monkeypatch.setattr(module, "SOURCES_DIR", sources_dir)
        monkeypatch.setattr(module, "load_graph", lambda: graph)
    monkeypatch.setattr(br, "FLAGS_PATH", tmp_path / "glue" / "restriction_flags.csv")
    monkeypatch.setattr(cr, "FLAGS_PATH", tmp_path / "rules" / "restriction_flags.csv")
    monkeypatch.setattr(cr, "TRADE_OFFS_PATH", tmp_path / "decide" / "best_trade_offs.json")   # no picks


def test_main_prints_the_differences_from_the_glue(graph, tmp_path, monkeypatch, capsys):
    _setup_main(APPLIED + [clause("typo", "Castlehil", "no_motor_vehicles")], graph, tmp_path, monkeypatch)
    cr.main()
    out = capsys.readouterr().out
    assert out.startswith("4 of 7 legal restrictions missing or wrong on OSM\n")
    assert out.endswith("Differences from the #20 glue:\n"
                        "  typo: glue abstained (street 'Castlehil' not found in the graph); compiler applied\n"
                        "  wrong on OSM (compiler only): typo\n")


def test_main_prints_flag_differences(graph, tmp_path, monkeypatch, capsys):
    _setup_main([clause("typo", "Castlehil", "no_motor_vehicles")], graph, tmp_path, monkeypatch)
    cr.main()
    assert capsys.readouterr().out.endswith("  wrong on OSM (compiler only): typo\n  flags differ on 2 edges: 4, 5\n")


def test_main_writes_flags_like_the_glue_and_prints_summary(graph, tmp_path, monkeypatch, capsys):
    _setup_main(APPLIED + [ABSTAINED[3][0]], graph, tmp_path, monkeypatch)

    cr.main()
    out = capsys.readouterr().out
    assert out.startswith("3 of 6 legal restrictions missing or wrong on OSM\n")
    assert "1 abstained:\n  exempt: unrepresentable: exemptions are not listed\n" in out
    assert out.endswith("Same applied clauses, OSM mismatches and flags as the #20 glue\n")

    br.main()
    capsys.readouterr()
    ours = (tmp_path / "rules" / "restriction_flags.csv").read_bytes()
    assert ours == (tmp_path / "glue" / "restriction_flags.csv").read_bytes()
    with (tmp_path / "rules" / "restriction_flags.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == graph.num_edges and rows[1] == {"edge_id": "1", "no_motor_vehicles": "1", "one_way": "1",
                                                        "pedestrian_only": "0"}


def test_between_compiles_to_only_its_stretch_and_agrees_with_the_glue():
    g = sn.RoadGraph()
    for nid, (lat, lon) in enumerate([(55.949, -3.200), (55.949, -3.199), (55.949, -3.198), (55.949, -3.197),
                                      (55.950, -3.199), (55.950, -3.198)]):
        g.add_node(nid, lat=lat, lon=lon)
    for u, v, name in ((0, 1, "Long Street"), (1, 2, "Long Street"), (2, 3, "Long Street"),
                       (1, 4, "First Street"), (2, 5, "Second Street")):
        g.add_edge(u, v, oneway=False, street_name=name, osm={"way_id": u})
    clauses = [clause("span", "Long Street", "no_motor_vehicles", between=["First Street", "Second Street"]),
               clause("bad", "Long Street", "no_motor_vehicles", between=["First Street"])]
    compiled = cr.compile_rules(clauses, SOURCES, g)
    assert cr.clause_edges(compiled) == {"span": {"2", "3"}}
    assert dict(cr.abstained(compiled))["bad"] == "unrepresentable: between needs two street names"
    assert cr.compare_with_glue(clauses, SOURCES, g, compiled) == {
        "clauses": {}, "wrong_on_osm": {"glue_only": [], "compiler_only": []}, "flags": []}
