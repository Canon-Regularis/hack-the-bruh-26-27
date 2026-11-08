import json

import pytest

from src.decide.network import link_maps, to_network
from src.rules import bind_rules as br
from src.rules import policy_check as pc
from tests.rules.rules_graph import build_graph
from tests.rules.test_bind_rules import ABSTAINED, APPLIED, SOURCES, clause

KEEP_BANK = clause("bank-open", "Bank Street", "keep_open")
EXEMPT = ABSTAINED[3][0]
CLAUSES = APPLIED + [KEEP_BANK, EXEMPT]
DO_NOTHING = {"closed_edge_ids": [], "capacity_factors": {}, "signalised_crossing_nodes": []}


@pytest.fixture(scope="module")
def graph():
    return build_graph()


@pytest.fixture(scope="module")
def setup(graph):
    report = br.check_clauses(CLAUSES, SOURCES, graph)
    return set(link_maps(to_network(graph)[0])[1]), report.clause_edges, dict(report.abstained)


def check(setup, diff):
    car_edges, clause_edges, abstained = setup
    return pc.check_design(diff, car_edges, CLAUSES, clause_edges, abstained)


def statuses(rows):
    return {row["clause_id"]: (row["status"], row["edges"]) for row in rows}


def test_doing_nothing_leaves_every_car_ban_in_tension(setup):
    rows = check(setup, DO_NOTHING)
    assert statuses(rows) == {
        "lawn-1way": ("in_tension", ["1", "3"]),
        "bow-1way": ("satisfied", []),
        "lawn-cars": ("in_tension", ["0", "1", "2", "3"]),
        "bank-cars": ("in_tension", ["6", "7"]),
        "castle-cars": ("in_tension", ["4", "5"]),
        "vic-ped": ("satisfied", []),
        "bank-open": ("satisfied", []),
        "exempt": ("abstained", []),
    }


def test_closing_a_street_satisfies_its_bans_and_cutting_a_keep_open_street_is_in_tension(setup):
    diff = {"closed_edge_ids": ["0", "1", "2", "3"], "capacity_factors": {"6": 0.5, "7": 0.5},
            "signalised_crossing_nodes": []}
    found = statuses(check(setup, diff))
    assert found["lawn-1way"] == ("satisfied", []) and found["lawn-cars"] == ("satisfied", [])
    assert found["bank-open"] == ("in_tension", ["6", "7"]) and found["bank-cars"] == ("in_tension", ["6", "7"])


def test_adding_capacity_to_a_keep_open_street_is_no_tension(setup):
    diff = {**DO_NOTHING, "capacity_factors": {"6": 1.2, "7": 1.2}}
    assert statuses(check(setup, diff))["bank-open"] == ("satisfied", [])


def test_closing_a_keep_open_street_is_in_tension(setup):
    diff = {**DO_NOTHING, "closed_edge_ids": ["6", "7"]}
    rows = {row["clause_id"]: row for row in check(setup, diff)}
    assert rows["bank-open"]["status"] == "in_tension" and rows["bank-open"]["detail"] == pc.KEEP_OPEN_TENSION
    assert rows["bank-cars"]["status"] == "satisfied"


def test_rows_carry_the_clause_quote_page_and_reason(setup):
    rows = {row["clause_id"]: row for row in check(setup, DO_NOTHING)}
    assert rows["lawn-cars"] == {"clause_id": "lawn-cars", "restriction": "no_motor_vehicles", "street": "Lawnmarket",
                                 "quote": "QUOTE", "page": 1, "document_id": "etro", "status": "in_tension",
                                 "edges": ["0", "1", "2", "3"], "detail": pc.BAN_TENSION}
    assert rows["exempt"]["detail"] == "exemptions are not listed" and rows["vic-ped"]["detail"] == ""


def test_summary_counts_verified_clauses_and_names_any_in_tension():
    rows = [{"status": s} for s in ("satisfied", "satisfied", "abstained", "in_tension")]
    assert pc.summary(rows) == "Consistent with 2 verified clauses; 1 in tension; 1 abstained for review."
    assert pc.summary(rows[:3]) == "Consistent with 2 verified clauses; 1 abstained for review."
    assert pc.summary(rows[2:]) == "Consistent with 0 verified clauses; 1 in tension; 1 abstained for review."
    assert pc.summary(rows[1:3]) == "Consistent with 1 verified clause; 1 abstained for review."
    assert "legally" not in pc.summary(rows)


def test_main_checks_each_pick(graph, tmp_path, monkeypatch, capsys):
    close_lawnmarket = {"mask": 1, "candidates": ["close-lawnmarket"],
                        "diff": {**DO_NOTHING, "closed_edge_ids": ["0", "1", "2", "3"]}}
    picks = {"car_best": {"mask": 0, "candidates": [], "diff": DO_NOTHING}, "knee": close_lawnmarket,
             "pedestrian_best": close_lawnmarket}
    calls = []
    real_check = pc.check_design
    monkeypatch.setattr(pc, "check_design", lambda *a: calls.append(a[0]) or real_check(*a))
    trade_offs = tmp_path / "best_trade_offs.json"
    trade_offs.write_text(json.dumps({"picks": picks}))
    clauses = tmp_path / "clauses.json"
    clauses.write_text(json.dumps(CLAUSES))
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "etro.txt").write_text(SOURCES["etro"])
    monkeypatch.setattr(pc, "TRADE_OFFS_PATH", trade_offs)
    monkeypatch.setattr(pc, "CLAUSES_PATH", clauses)
    monkeypatch.setattr(pc, "SOURCES_DIR", sources)
    monkeypatch.setattr(pc, "POLICY_CHECK_PATH", tmp_path / "rules" / "policy_check.json")
    monkeypatch.setattr(pc, "load_graph", lambda: graph)

    pc.main()

    out = json.loads((tmp_path / "rules" / "policy_check.json").read_text())
    assert out["designs"]["car_best"]["summary"] == ("Consistent with 3 verified clauses; 4 in tension; "
                                                     "1 abstained for review.")
    assert out["designs"]["knee"]["summary"] == ("Consistent with 5 verified clauses; 2 in tension; "
                                                 "1 abstained for review.")
    assert out["designs"]["pedestrian_best"] == out["designs"]["knee"] and len(calls) == 2
    assert len(out["designs"]["knee"]["clauses"]) == len(CLAUSES) and "not a legal determination" in out["note"]
    assert capsys.readouterr().out.splitlines() == [f"{role}: {d['summary']}" for role, d in out["designs"].items()]
