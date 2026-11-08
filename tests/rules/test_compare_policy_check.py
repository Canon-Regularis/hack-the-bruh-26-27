"""compile_rules' keep_open encoding, check_diff and the comparison with the #31 policy check (ALG-S3-2, #33)."""

import json
import sys

import pytest

from engine.orchid import policy_compiler as pc
from src.rules import bind_rules as br
from src.rules import compile_rules as cr
from src.rules import policy_check as glue_check
from tests.rules.rules_graph import build_graph
from tests.rules.test_bind_rules import ABSTAINED, APPLIED, SOURCES, clause
from tests.rules.test_compile_rules import _setup_main
from tests.rules.test_policy_check import CLAUSES as GLUE_CLAUSES
from tests.rules.test_policy_check import DO_NOTHING, KEEP_BANK

EXEMPT = ABSTAINED[3][0]
TIMED = clause("timed", "Lawnmarket", "no_motor_vehicles", times="8am-6pm")
TYPO = clause("typo", "Castlehil", "no_motor_vehicles")
KEEP_BOW = clause("bow-open", "Upper Bow", "keep_open")
TIMED_BANK = clause("bank-t", "Bank Street", "no_motor_vehicles", times="8am-6pm")
TYPO_KEEP = clause("castle-open", "Castlehil", "keep_open")
LAWN = ["0", "1", "2", "3"]


def design(closed=(), factors=None):
    return {**DO_NOTHING, "closed_edge_ids": list(closed), "capacity_factors": factors or {}}


@pytest.fixture(scope="module")
def graph():
    return build_graph()


def glue_rows(clauses, graph, diff):
    report = br.check_clauses(clauses, SOURCES, graph)
    return glue_check.check_design(diff, cr.car_edges(graph), clauses, report.clause_edges, dict(report.abstained))


def statuses(rows):
    return {r["clause_id"]: (r["status"], r["edges"]) for r in rows}


def test_car_edges_are_the_modelled_car_links(graph):
    assert cr.car_edges(graph) == {str(e) for e in range(9)}   # Lawnmarket stays a car link; Victoria St doesn't


def test_keep_open_compiles_to_fixed_one_and_a_capacity_floor(graph):
    c = cr.compile_rules([KEEP_BANK], SOURCES, graph)
    assert c.result.clauses == {} and list(c.keep.clauses) == ["bank-open"]   # a pass of its own
    fixings = c.keep.fixings.to_dict()
    assert (fixings["fixedZero"], fixings["fixedOne"]) == ([], ["6", "7"])
    assert fixings["bounds"] == {"capacity:6": [1.0, sys.float_info.max], "capacity:7": [1.0, sys.float_info.max]}
    assert cr.abstained(c) == [] and cr.applied(c) == [] and cr.conditional(c) == []
    assert cr.kept_open(c) == {"bank-open": {"6", "7"}} == br.check_clauses([KEEP_BANK], SOURCES, graph).keep_open
    assert cr.outcome(c, "bank-open") == "kept open"
    assert cr.compare_with_glue([KEEP_BANK], SOURCES, graph, c) == {
        "clauses": {}, "wrong_on_osm": {"glue_only": [], "compiler_only": []}, "flags": []}


@pytest.mark.parametrize(("diff", "status", "edges", "violated"), [
    (design(), "satisfied", [], []),
    (design(["6", "7"]), "in_tension", ["6", "7"], ["fixedOne:6", "fixedOne:7"]),
    (design(factors={"7": 0.5}), "in_tension", ["7"], ["bounds:capacity:7"]),
    (design(factors={"6": 0.999}), "in_tension", ["6"], ["bounds:capacity:6"]),
    (design(["6"], {"7": 0.5}), "in_tension", ["6", "7"], ["bounds:capacity:7", "fixedOne:6"]),
    (design(factors={"6": 1.0, "7": 1.0}), "satisfied", [], []),
    (design(factors={"6": 1.2, "7": 1.2}), "satisfied", [], []),
    (design(LAWN, {"4": 0.5}), "satisfied", [], []),
])
def test_keep_open_is_in_tension_for_a_closure_or_a_cut_like_31(graph, diff, status, edges, violated):
    c = cr.compile_rules([KEEP_BANK], SOURCES, graph)
    rows, ratio = cr.check_diff(diff, c, cr.car_edges(graph))
    assert (rows[0]["status"], rows[0]["edges"], rows[0]["violated"]) == (status, edges, violated)
    assert ratio == (0.0 if violated else 1.0)
    assert statuses(glue_rows([KEEP_BANK], graph, diff)) == {"bank-open": (status, edges)}


def test_time_limited_keep_open_with_unlisted_exemptions_applies_all_day_like_the_glue(graph):
    am = clause("bank-am", "Bank Street", "keep_open", times="07:00-10:00", exemptions="unspecified")
    c = cr.compile_rules([am], SOURCES, graph)
    assert (cr.abstained(c), cr.conditional(c), cr.kept_open(c)) == ([], [], {"bank-am": {"6", "7"}})
    rows, _ = cr.check_diff(design(["6"]), c, cr.car_edges(graph))
    assert (rows[0]["status"], rows[0]["edges"], rows[0]["detail"]) == ("in_tension", ["6"], "unconditional")
    assert statuses(glue_rows([am], graph, design(["6"]))) == {"bank-am": ("in_tension", ["6"])}


def test_keep_open_on_a_footway_is_never_in_tension(graph):
    vic = clause("vic-open", "Victoria St", "keep_open")
    c = cr.compile_rules([vic], SOURCES, graph)
    assert c.keep.fixings.to_dict()["fixedOne"] == ["10", "9"]
    for diff in (design(), design(map(str, range(9)))):
        rows, _ = cr.check_diff(diff, c, cr.car_edges(graph))
        assert (rows[0]["status"], rows[0]["edges"]) == ("satisfied", []) == statuses(glue_rows([vic], graph, diff))[
            "vic-open"]


BANS = [APPLIED[2], APPLIED[3], APPLIED[5]]   # lawn-cars, bank-cars, vic-ped


@pytest.mark.parametrize(("diff", "expected"), [
    (design(), {"lawn-cars": ("in_tension", LAWN), "bank-cars": ("in_tension", ["6", "7"]),
                "vic-ped": ("satisfied", [])}),
    (design([*LAWN, "6"]), {"lawn-cars": ("satisfied", []), "bank-cars": ("in_tension", ["7"]),
                            "vic-ped": ("satisfied", [])}),
    (design([*LAWN, "6", "7"], {"4": 0.5}), {"lawn-cars": ("satisfied", []), "bank-cars": ("satisfied", []),
                                             "vic-ped": ("satisfied", [])}),
])
def test_ban_is_satisfied_once_its_car_links_are_closed(graph, diff, expected):
    c = cr.compile_rules(BANS, SOURCES, graph)
    rows, _ = cr.check_diff(diff, c, cr.car_edges(graph))
    assert statuses(rows) == expected == statuses(glue_rows(BANS, graph, diff))


def test_osm_closed_car_link_counts_as_open_in_check_diff_but_not_in_the_status_quo_view(graph):
    c = cr.compile_rules(BANS[:1], SOURCES, graph)   # Lawnmarket: motor_vehicle=no, but a modelled car link
    assert cr.check_diff(design(), c, cr.car_edges(graph))[0][0]["status"] == "in_tension"
    assert cr.check_design(set(), c, graph)[0][0]["status"] == "satisfied"


def flags(c, graph, restriction, eids):
    rows = cr.restriction_flags(c, graph)
    return [rows[e][restriction] for e in eids]


def test_a_keep_open_clause_never_drops_a_ban_from_the_legal_outputs(graph):
    c = cr.compile_rules([APPLIED[3], KEEP_BANK], SOURCES, graph)   # bank-cars and bank-open
    assert (cr.applied(c), cr.abstained(c), cr.kept_open(c)) == (["bank-cars"], [], {"bank-open": {"6", "7"}})
    assert flags(c, graph, "no_motor_vehicles", [6, 7]) == [1, 1]
    assert cr.mismatch_line(c, graph) == "1 of 1 legal restrictions missing or wrong on OSM"   # motor_vehicle=yes
    assert cr.keep_open_overlaps(c) == [("bank-cars", "bank-open", ["6", "7"])]
    assert cr.compare_with_glue([APPLIED[3], KEEP_BANK], SOURCES, graph, c) == {
        "clauses": {}, "wrong_on_osm": {"glue_only": [], "compiler_only": []}, "flags": []}


def test_a_keep_open_clause_never_drops_a_timed_ban_or_a_one_way(graph):
    c = cr.compile_rules([TIMED_BANK, KEEP_BANK], SOURCES, graph)
    assert (cr.conditional(c), cr.abstained(c)) == (["bank-t"], [])
    assert cr.keep_open_overlaps(c) == [("bank-t", "bank-open", ["6", "7"])]
    lawn_open = clause("lawn-open", "Lawnmarket", "keep_open")
    c = cr.compile_rules([APPLIED[0], lawn_open], SOURCES, graph)   # lawn-1way shares only edges 1 and 3
    assert (cr.applied(c), cr.abstained(c)) == (["lawn-1way"], [])
    assert cr.kept_open(c) == {"lawn-open": set(LAWN)} == br.check_clauses([APPLIED[0], lawn_open], SOURCES,
                                                                           graph).keep_open
    assert flags(c, graph, "one_way", [0, 1, 2, 3]) == [0, 1, 0, 1]
    assert cr.keep_open_overlaps(c) == [("lawn-1way", "lawn-open", ["1", "3"])]


def test_keep_open_pass_reports_its_abstentions_skips_and_wording(graph):
    sources = {"etro": SOURCES["etro"] + " Vehicles should not be stopped on Bank Street."}
    clauses = [clause("vic-open", "Victoria Street", "keep_open"), APPLIED[2],
               clause("nowhere-open", "Johnston Terrace", "keep_open"),
               clause("no-page-open", "Bank Street", "keep_open", page=0),
               clause("soft-open", "Bank Street", "keep_open", strength="soft"),
               clause("should-open", "Bank Street", "keep_open", quote="Vehicles should not be stopped on Bank Street")]
    c = cr.compile_rules(clauses, sources, graph)
    assert [(cid, why.split(":")[0]) for cid, why in cr.abstained(c)] == [
        ("nowhere-open", "binding_unresolved"), ("no-page-open", "unrepresentable")]
    assert cr.outcome(c, "soft-open") == "skipped (soft_clause)"
    assert cr.wording_gate_flags(c) == [("should-open", "discouragement", "should not")]
    rows, _ = cr.check_diff(design(["6"]), c, cr.car_edges(graph))
    assert [(r["clause_id"], r["status"]) for r in rows] == [
        ("vic-open", "satisfied"), ("lawn-cars", "in_tension"), ("nowhere-open", "abstained"),
        ("no-page-open", "abstained"), ("soft-open", "not_applicable"), ("should-open", "in_tension")]
    rows, _ = cr.check_design(set(), c, graph)   # input order, not clause-id order
    assert [r["clause_id"] for r in rows] == ["vic-open", "lawn-cars", "nowhere-open", "soft-open", "should-open"]


def test_a_refused_ban_has_no_overlap(graph):
    c = cr.compile_rules([EXEMPT, clause("lawn-open", "Lawnmarket", "keep_open")], SOURCES, graph)
    assert cr.keep_open_overlaps(c) == []


def test_status_quo_view_never_puts_a_keep_open_street_osm_closes_in_tension(graph):
    keep = [clause("lawn-open", "Lawnmarket", "keep_open"), clause("vic-open", "Victoria Street", "keep_open")]
    c = cr.compile_rules([APPLIED[2], *keep], SOURCES, graph)   # lawn-cars: OSM already closes Lawnmarket
    rows, ratio = cr.check_design(set(), c, graph)
    assert [(r["clause_id"], r["status"]) for r in rows] == [
        ("lawn-cars", "satisfied"), ("lawn-open", "satisfied"), ("vic-open", "satisfied")]
    assert ratio == 1.0
    rows, ratio = cr.check_design({"0", "9"}, c, graph)   # the design's own closures still count
    assert [(r["clause_id"], r["status"], r["violated"]) for r in rows] == [
        ("lawn-cars", "satisfied", []), ("lawn-open", "in_tension", ["fixedOne:0"]),
        ("vic-open", "in_tension", ["fixedOne:9"])]
    assert ratio == 1 / 3


@pytest.mark.parametrize(("minute", "status", "edges"), [(None, "in_tension", LAWN), (20 * 60, "not_applicable", []),
                                                         (10 * 60, "in_tension", LAWN)])
def test_check_diff_checks_a_time_window_only_when_its_scenario_holds(graph, minute, status, edges):
    c = cr.compile_rules([TIMED, KEEP_BANK], SOURCES, graph)
    scenario = None if minute is None else pc.Scenario(minute, "mon")
    rows, ratio = cr.check_diff(design(["6"]), c, cr.car_edges(graph), scenario)
    assert [(r["clause_id"], r["status"], r["edges"]) for r in rows] == [
        ("timed", status, edges), ("bank-open", "in_tension", ["6"])]
    assert ratio == 0.0


def test_check_diff_ratio_counts_both_passes(graph):
    rows, ratio = cr.check_diff(design(), cr.compile_rules([APPLIED[3], KEEP_BANK], SOURCES, graph),
                                cr.car_edges(graph))
    assert [r["status"] for r in rows] == ["in_tension", "satisfied"] and ratio == 0.5


def test_design_from_diff_lists_car_edges_and_capacity_factors(graph):
    d = cr.design_from_diff(design(["0", "9"], {"6": 0.5, "7": 1.2}), cr.car_edges(graph))
    assert d.z == {"0": 0, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1, "9": 0}
    assert d.y == {"capacity:6": 0.5, "capacity:7": 1.2} and (d.w, d.k) == ({}, {})


def test_violated_edges_reads_every_key_kind():
    assert cr.violated_edges(["bounds:capacity:10", "fixedOne:9", "fixedZero:2", "fixedZero:10"]) == ["2", "9", "10"]


def test_schema_refused_clause_is_reported_abstained(graph):
    c = cr.compile_rules([clause("no-page", "Bank Street", "no_motor_vehicles", page=0)], SOURCES, graph)
    rows, ratio = cr.check_diff(design(), c, cr.car_edges(graph))
    assert rows == [{"clause_id": "no-page", "status": "abstained", "edges": [], "violated": [],
                     "detail": "unrepresentable: invalid clause 'no-page': /page: 0 < minimum 1", "quote": "QUOTE",
                     "page": 0, "document_id": "etro"}]
    assert ratio == 1.0


AGREE = [APPLIED[0], *APPLIED[2:], KEEP_BOW, EXEMPT]   # no one-way with nothing to close, no ban on a kept street
AGREE_DESIGNS = {
    "do_nothing": design(),
    "close_lawn_cut_bow": design(LAWN, {"8": 0.5}),
    "close_bank_castle": design(["4", "5", "6", "7"]),
    "close_bow": design(["8"]),
    "widen_bow": design(factors={"8": 1.2, "6": 0.8}),
    "close_all_bans": design([*LAWN, "4", "5", "6", "7"], {"8": 1.0}),
}


def test_the_two_checks_agree_on_every_design(graph):
    rows = cr.compare_policy_checks(AGREE_DESIGNS, AGREE, SOURCES, graph)
    assert len(rows) == len(AGREE_DESIGNS) * len(AGREE)
    assert [r for r in rows if r["explanation"]] == []
    found = {(r["design"], r["clause_id"]): (r["compiler"], r["compiler_edges"]) for r in rows}
    assert {cid: found["do_nothing", cid] for cid in ("lawn-1way", "lawn-cars", "bow-open", "exempt")} == {
        "lawn-1way": ("in_tension", ["1", "3"]), "lawn-cars": ("in_tension", LAWN), "bow-open": ("satisfied", []),
        "exempt": ("abstained", [])}
    assert found["close_lawn_cut_bow", "bow-open"] == ("in_tension", ["8"]) == found["close_bow", "bow-open"]
    assert found["close_lawn_cut_bow", "lawn-1way"] == ("satisfied", []) == found["widen_bow", "bow-open"]
    assert found["widen_bow", "bank-cars"] == ("in_tension", ["6", "7"])
    assert [cid for (name, cid), (s, _) in found.items() if name == "close_all_bans" and s == "in_tension"] == []


def test_every_difference_on_the_31_fixture_is_explained(graph):
    designs = {"do_nothing": design(), "close_lawn_cut_bank": design(LAWN, {"6": 0.5, "7": 0.5}),
               "close_bank": design(["6", "7"])}
    rows = cr.compare_policy_checks(designs, GLUE_CLAUSES + [TIMED, TYPO], SOURCES, graph)
    differ = [(r["design"], r["clause_id"], r["glue"], r["glue_edges"], r["compiler"], r["compiler_edges"],
               r["explanation"]) for r in rows if r["explanation"]]
    assert differ == [
        ("do_nothing", "bow-1way", "satisfied", [], "not_applicable", [], "nothing_to_close"),
        ("do_nothing", "timed", "abstained", [], "in_tension", LAWN, "time_window"),
        ("do_nothing", "typo", "abstained", [], "in_tension", ["4", "5"], "compile_outcome"),
        ("close_lawn_cut_bank", "bow-1way", "satisfied", [], "not_applicable", [], "nothing_to_close"),
        ("close_lawn_cut_bank", "timed", "abstained", [], "satisfied", [], "time_window"),
        ("close_lawn_cut_bank", "typo", "abstained", [], "in_tension", ["4", "5"], "compile_outcome"),
        ("close_bank", "bow-1way", "satisfied", [], "not_applicable", [], "nothing_to_close"),
        ("close_bank", "timed", "abstained", [], "in_tension", LAWN, "time_window"),
        ("close_bank", "typo", "abstained", [], "in_tension", ["4", "5"], "compile_outcome"),
    ]
    found = {(r["design"], r["clause_id"]): (r["compiler"], r["compiler_edges"]) for r in rows}
    assert found["do_nothing", "bank-cars"] == ("in_tension", ["6", "7"])   # the ban and bank-open both judged
    assert found["do_nothing", "bank-open"] == ("satisfied", []) == found["close_bank", "bank-cars"]
    assert found["close_lawn_cut_bank", "bank-open"] == ("in_tension", ["6", "7"]) == found["close_bank", "bank-open"]


@pytest.mark.parametrize(("clauses", "designs", "expected"), [
    ([TIMED_BANK, KEEP_BANK], {"none": design(), "close": design(["6", "7"])}, [
        ("none", "bank-t", "abstained", [], "in_tension", ["6", "7"], "time_window"),
        ("none", "bank-open", "satisfied", [], "satisfied", [], None),
        ("close", "bank-t", "abstained", [], "satisfied", [], "time_window"),
        ("close", "bank-open", "in_tension", ["6", "7"], "in_tension", ["6", "7"], None)]),
    ([APPLIED[4], TYPO_KEEP], {"cut4": design(factors={"4": 0.5})}, [
        ("cut4", "castle-cars", "in_tension", ["4", "5"], "in_tension", ["4", "5"], None),
        ("cut4", "castle-open", "abstained", [], "in_tension", ["4"], "compile_outcome")]),
    ([clause("bow-1way-t", "Upper Bow", "one_way", times="8am-6pm", from_street="Bank Street",
             to_street="Victoria Street")], {"none": design()}, [
        ("none", "bow-1way-t", "abstained", [], "not_applicable", [], "time_window")]),
])
def test_a_clause_beside_one_31_abstains_gets_its_own_explanation(graph, clauses, designs, expected):
    rows = cr.compare_policy_checks(designs, clauses, SOURCES, graph)
    assert [tuple(r.values()) for r in rows] == expected
    assert "nothing to close" in cr.EXPLANATIONS["time_window"] and "conflict" not in cr.EXPLANATIONS


def test_a_difference_no_category_covers_is_unexplained(graph, monkeypatch):
    real = glue_check.check_design

    def first_edge_only(*args):   # the same status as the compiler, on fewer edges
        return [{**r, "edges": r["edges"][:1]} if r["clause_id"] == "lawn-cars" else r for r in real(*args)]

    monkeypatch.setattr(cr.glue_check, "check_design", first_edge_only)
    rows = cr.compare_policy_checks({"do_nothing": design()}, BANS, SOURCES, graph)
    assert [(r["clause_id"], r["glue"], r["glue_edges"], r["compiler"], r["compiler_edges"], r["explanation"])
            for r in rows if r["explanation"]] == [("lawn-cars", "in_tension", ["0"], "in_tension", LAWN, "unexplained")]


def test_pick_diffs_keeps_one_design_per_mask():
    lawn = design(LAWN)
    picks = {"car_best": {"mask": 0, "diff": DO_NOTHING}, "knee": {"mask": 1, "diff": lawn},
             "pedestrian_best": {"mask": 1, "diff": lawn}}
    assert cr.pick_diffs(picks) == {"car_best": DO_NOTHING, "knee, pedestrian_best": lawn}


def test_main_compares_the_picks_with_31(graph, tmp_path, monkeypatch, capsys):
    _setup_main(GLUE_CLAUSES, graph, tmp_path, monkeypatch)
    lawn = {"mask": 1, "candidates": ["close-lawnmarket"], "diff": design(LAWN)}
    trade_offs = tmp_path / "best_trade_offs.json"
    trade_offs.write_text(json.dumps({"picks": {"car_best": {"mask": 0, "candidates": [], "diff": DO_NOTHING},
                                                "knee": lawn, "pedestrian_best": lawn}}))
    monkeypatch.setattr(cr, "TRADE_OFFS_PATH", trade_offs)
    cr.main()
    out = capsys.readouterr().out
    assert ("1 keep-open clauses compiled (may not be closed or have capacity cut)\n"
            "1 bans on edges a keep-open clause keeps open (both kept, for review):\n"
            "  bank-cars and bank-open on edges 6, 7\n") in out
    assert ("Policy check against #31 on 2 picked designs:\n"
            "  car_best: 7 of 8 clauses agree\n"
            "    bow-1way: #31 satisfied; compiler not_applicable (nothing_to_close)\n"
            "  knee, pedestrian_best: 7 of 8 clauses agree\n"
            "    bow-1way: #31 satisfied; compiler not_applicable (nothing_to_close)\n"
            "Same applied clauses, OSM mismatches and flags as the #20 glue\n") in out


def test_main_says_when_there_are_no_picks(graph, tmp_path, monkeypatch, capsys):
    _setup_main([KEEP_BANK], graph, tmp_path, monkeypatch)
    cr.main()
    out = capsys.readouterr().out
    assert "1 keep-open clauses compiled (may not be closed or have capacity cut)\n" in out
    assert "skipped the comparison with the #31 policy check\n" in out
    assert out.endswith("Same applied clauses, OSM mismatches and flags as the #20 glue\n")
