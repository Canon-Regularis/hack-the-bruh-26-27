import csv
import json
import math
from itertools import combinations

import pytest

from src.concerns import concern_field as cf
from src.decide import best_trade_offs as bt
from src.decide import design_search as ds
from src.decide.baseline_user_equilibrium import ue_rows
from tests.decide.two_route import NORMAL, build_context, build_graph

VOICED = bt.OBJECTIVES + (bt.VOICE,)
NORTH = ["0", "1", "2", "3"]


def design(mask, j_car, j_ped, j_risk, j_voice=None):
    d = {"mask": mask, "candidates": [f"c{mask}"], "feasible": True,
         "mean_j_car": j_car, "mean_j_ped": j_ped, "j_risk": j_risk}
    return d if j_voice is None else {**d, "j_voice": j_voice}


# Best trade-offs A-E. In the J_car vs J_risk projection E is dominated by D, so the knee needs
# that projection cut to its own best trade-offs: cut, the chord runs A-D and the knee is B; uncut,
# it would run A-E and pick D. On the wrong projection, J_car vs J_ped, the knee would be C.
A, B, C, D, E = (design(1, 1, 9, 10), design(2, 2, 8.9, 4), design(3, 5, 4, 3), design(4, 9, 3.5, 2.5),
                 design(5, 10, 1, 9))
DOMINATED = design(6, 11, 9.5, 11)
INFEASIBLE = {"mask": 7, "candidates": ["c7"], "feasible": False, "reason": "unreachable"}


def box_union_volume(points, ref):
    """Exact hypervolume by inclusion-exclusion, independent of orchid."""
    total = 0.0
    for k in range(1, len(points) + 1):
        for subset in combinations(points, k):
            corner = [max(p[d] for p in subset) for d in range(len(ref))]
            total += (-1) ** (k + 1) * math.prod(r - c for r, c in zip(ref, corner))
    return total


def test_normalise_scales_each_objective_to_0_1_and_flat_ones_to_0():
    assert bt.normalise([[1, 10, 5], [3, 10, 1], [2, 10, 3]]) == [[0, 0, 1], [1, 0, 0], [0.5, 0, 0.5]]


def test_trade_offs_skip_infeasible_and_dominated_designs():
    trade_offs, volume = bt.trade_offs_and_hypervolume([E, INFEASIBLE, C, A, DOMINATED, D, B])
    assert [d["mask"] for d in trade_offs] == [5, 3, 1, 4, 2]
    normalised = bt.normalise([[d[name] for name in bt.OBJECTIVES] for d in trade_offs])
    assert volume == pytest.approx(box_union_volume(normalised, [bt.REFERENCE] * 3))


@pytest.mark.parametrize("dominated", [design(6, 11, 9.5, 11), design(7, 100, 900, 100), design(8, 1e3, 1e6, 1e3)])
def test_dominated_designs_do_not_change_the_hypervolume(dominated):
    without = bt.trade_offs_and_hypervolume([A, B, C, D, E])[1]
    assert bt.trade_offs_and_hypervolume([A, B, C, D, E, dominated])[1] == without
    assert without < bt.REFERENCE ** 3


def test_pick_designs_uses_the_cut_projection_for_the_knee():
    picks = bt.pick_designs([A, B, C, D, E])
    assert {role: d["mask"] for role, d in picks.items()} == {"car_best": 1, "pedestrian_best": 5, "knee": 2}


def test_pick_designs_breaks_ties_by_the_lower_mask():
    twin = {**A, "mask": 0}
    assert bt.pick_designs([A, twin, E])["car_best"]["mask"] == 0


def test_design_diff_lists_edge_ids_and_combined_factors():
    car_edges = {str(e) for e in range(12)}
    candidates = [
        {"id": "close-north", "kind": "close", "edges": frozenset({"0", "1", "2", "3"}), "capacity_factor": None,
         "node": None},
        {"id": "bus-lane-south", "kind": "reallocate", "edges": frozenset({"4", "5", "10"}), "capacity_factor": 0.5,
         "node": None},
        {"id": "signal-1", "kind": "signalise_crossing", "edges": frozenset({"0", "3"}), "capacity_factor": 0.9,
         "node": 1},
        {"id": "signal-9", "kind": "signalise_crossing", "edges": frozenset({"5", "10"}), "capacity_factor": 0.8,
         "node": 9},
    ]
    assert bt.design_diff({"mask": 0b1111}, candidates, car_edges) == {
        "closed_edge_ids": ["0", "1", "2", "3"],
        "capacity_factors": {"4": 0.5, "5": pytest.approx(0.4), "10": pytest.approx(0.4)},
        "signalised_crossing_nodes": [1, 9],
    }
    assert bt.design_diff({"mask": 0}, candidates, car_edges) == {"closed_edge_ids": [], "capacity_factors": {},
                                                                  "signalised_crossing_nodes": []}


def test_design_diff_never_shows_pedestrian_sections_as_changed():
    candidates = [
        {"id": "close-rose", "kind": "close", "edges": frozenset({"0", "1", "2", "3", "4", "5"}),
         "capacity_factor": None, "node": None},
        {"id": "bus-lane-rose", "kind": "reallocate", "edges": frozenset({"0", "1", "2", "3", "4", "5"}),
         "capacity_factor": 0.5, "node": None},
    ]
    car_edges = {"0", "1", "4", "5"}
    assert bt.design_diff({"mask": 0b01}, candidates, car_edges)["closed_edge_ids"] == ["0", "1", "4", "5"]
    assert bt.design_diff({"mask": 0b10}, candidates, car_edges)["capacity_factors"] == {
        "0": 0.5, "1": 0.5, "4": 0.5, "5": 0.5}


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    return build_context(build_graph(), tmp_path_factory.mktemp("decide"))


@pytest.fixture(scope="module")
def search(context):
    return search_of(context)


def search_of(context):
    results = {(mask, s["id"]): ds.evaluate(context, mask, s)
               for mask in range(1 << len(context["candidates"])) for s in context["scenarios"]}
    return {"hour": context["hour"], "alpha": ds.ALPHA, "candidates": [c["id"] for c in context["candidates"]],
            "fingerprint": json.loads(json.dumps(ds.fingerprint(context))), "designs": ds.summarise(context, results)}


def write_field(path, values, basis=cf.LOCATED):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cf.FIELD_COLUMNS)
        writer.writeheader()
        writer.writerows(cf.field_rows(build_graph(), values, basis))
    return path


def run_main(context, search, tmp_path, monkeypatch, field=None):
    search_path = tmp_path / "design_search.json"
    search_path.write_text(json.dumps(search))
    monkeypatch.setattr(bt, "SEARCH_PATH", search_path)
    monkeypatch.setattr(bt, "TRADE_OFFS_PATH", tmp_path / "best_trade_offs.json")
    monkeypatch.setattr(bt, "DESIGNS_DIR", tmp_path / "designs")
    monkeypatch.setattr(bt, "FIELD_PATH", field or tmp_path / "no_field.csv")
    monkeypatch.setattr(bt, "WORKERS", 2)
    monkeypatch.setattr(bt, "load_context", lambda: context)
    bt.main()
    return json.loads((tmp_path / "best_trade_offs.json").read_text())


def assert_pick_files(out, context, tmp_path):
    for role, pick in out["picks"].items():
        net, _, _, result = ds.solve(context, pick["mask"], bt.BASELINE)
        expected = [{key: str(value) for key, value in row.items()} for row in ue_rows(net, result)]
        with (tmp_path / "designs" / f"{role}.csv").open() as f:
            assert list(csv.DictReader(f)) == expected
        assert pick["diff"] == bt.design_diff(pick, context["candidates"], set(context["link_to_edge"]))


def test_main_without_a_field_keeps_three_objectives_and_warns(context, search, tmp_path, monkeypatch, capsys):
    out = run_main(context, search, tmp_path, monkeypatch)
    feasible = [d for d in search["designs"] if d["feasible"]]
    assert {d["mask"] for d in out["trade_offs"]} <= {d["mask"] for d in feasible}
    assert set(out["picks"]) == {"car_best", "pedestrian_best", "knee"}
    assert out["hour"] == "08:00-09:00 weekday" and out["reference"] == [1.1, 1.1, 1.1]
    assert out["objectives"] == list(bt.OBJECTIVES) and out["j_voice"]["status"] == bt.NOT_COMPUTED
    assert all("j_voice" not in d for d in out["trade_offs"])
    assert_pick_files(out, context, tmp_path)
    printed = capsys.readouterr().out
    assert "no concern field" in printed
    assert f"{len(out['trade_offs'])} best trade-offs over 3 objectives, hypervolume " in printed


def test_main_refuses_a_search_run_on_other_inputs(context, search, tmp_path, monkeypatch):
    edited = [{**c, "capacity_factor": 0.7} if c["kind"] == "reallocate" else c for c in context["candidates"]]
    with pytest.raises(ValueError, match="have changed since"):
        run_main({**context, "candidates": edited}, search, tmp_path, monkeypatch)


def test_main_refuses_a_search_with_no_feasible_design(context, search, tmp_path, monkeypatch):
    infeasible = [{**INFEASIBLE, "mask": d["mask"]} for d in search["designs"]]
    with pytest.raises(ValueError, match="no feasible design"):
        run_main(context, {**search, "designs": infeasible}, tmp_path, monkeypatch)


# ------------------------------------------------------------------ J_voice (#36)

F = design(9, 11, 9.5, 11, j_voice=0.0)          # worst on the other three, best on J_voice


def voiced(*designs, voices=(5.0, 4.0, 3.0, 2.0, 1.0)):
    return [{**d, "j_voice": v} for d, v in zip(designs, voices)]


def test_four_objective_trade_offs_and_hypervolume():
    trade_offs, volume = bt.trade_offs_and_hypervolume(voiced(A, B, C, D, E) + [F], VOICED)
    assert sorted(d["mask"] for d in trade_offs) == [1, 2, 3, 4, 5, 9]
    normalised = bt.normalise([[d[name] for name in VOICED] for d in trade_offs])
    assert volume == pytest.approx(box_union_volume(normalised, [bt.REFERENCE] * 4))


def test_voice_best_joins_the_picks_and_leaves_the_others_alone():
    trade_offs = bt.trade_offs_and_hypervolume(voiced(A, B, C, D, E) + [F], VOICED)[0]
    picks = bt.pick_designs(trade_offs, VOICED)
    assert {role: d["mask"] for role, d in picks.items()} == {"car_best": 1, "pedestrian_best": 5, "knee": 2,
                                                              "voice_best": 9}
    tie = [{**F, "mask": 12}, F]
    assert bt.pick_designs(voiced(A, E) + tie, VOICED)["voice_best"]["mask"] == 9


def test_attach_voice_scores_feasible_designs_and_drops_failed_baselines():
    designs = [design(0, 1, 1, 1), design(1, 2, 2, 2), INFEASIBLE]
    runs = {0: {"converged": True, "total": 8.0, "by_theme": {"noise": 8.0}},
            1: {"converged": False, "reason": "baseline: did not converge"}}
    out = bt.attach_voice(designs, runs, 8.0)
    assert out[0]["j_voice"] == 8.0 and out[0]["j_voice_improvement"] == 0.0 and out[0]["feasible"]
    assert not out[1]["feasible"] and out[1]["reason"].startswith("baseline:")
    assert out[2] is INFEASIBLE


@pytest.fixture(scope="module")
def open_context(tmp_path_factory):
    """Only the normal scenario, so all four designs are feasible."""
    ctx = build_context(build_graph(), tmp_path_factory.mktemp("open"))
    return {**ctx, "scenarios": [{**NORMAL, "probability": 1.0}]}


@pytest.fixture(scope="module")
def open_search(open_context):
    return search_of(open_context)


def test_main_with_a_located_field_optimises_j_voice(open_context, open_search, tmp_path, monkeypatch, capsys):
    values = {e: {"rat_running": 1.0} for e in NORTH} | {"8": {"parking": 2.0}}
    out = run_main(open_context, open_search, tmp_path, monkeypatch, write_field(tmp_path / "field.csv", values))
    assert out["objectives"] == list(VOICED) and out["reference"] == [1.1] * 4
    assert set(out["picks"]) == {"car_best", "pedestrian_best", "knee", "voice_best"}
    best = out["picks"]["voice_best"]
    assert best["mask"] == 1 and best["diff"]["closed_edge_ids"] == NORTH        # mask 3 ties; lower mask wins
    assert best["j_voice"] == 0.0 and best["j_voice_improvement"] == 1.0
    assert best["j_voice_by_theme"]["rat_running"] == 0.0
    voice = out["j_voice"]
    assert voice["status"] == bt.OPTIMISED and voice["basis"] == cf.LOCATED
    assert voice["j_voice_baseline"] == 4.0 and voice["excluded_mass"]["parking"] == 2.0 and voice["dropped"] == []
    assert all("j_voice" in d for d in out["trade_offs"])
    assert_pick_files(out, open_context, tmp_path)
    assert "4 objectives" in capsys.readouterr().out
    json.dumps(out, allow_nan=False)


def test_main_with_a_per_street_field_shows_but_does_not_optimise(open_context, open_search, tmp_path, monkeypatch,
                                                                  capsys):
    field = write_field(tmp_path / "field.csv", {e: {"rat_running": 1.0} for e in NORTH}, cf.PER_STREET)
    out = run_main(open_context, open_search, tmp_path, monkeypatch, field)
    assert out["objectives"] == list(bt.OBJECTIVES) and "voice_best" not in out["picks"]
    assert out["j_voice"]["status"] == bt.SHOWN and all("j_voice" in d for d in out["trade_offs"])
    assert "shown but not optimised" in capsys.readouterr().out


def test_main_refuses_a_field_that_misses_network_edges(open_context, open_search, tmp_path, monkeypatch):
    field = write_field(tmp_path / "field.csv", {})
    lines = field.read_text().splitlines()
    field.write_text("".join(line + "\n" for line in lines if not line.startswith("6,")))
    with pytest.raises(ValueError, match="no row for 1 edges"):
        run_main(open_context, open_search, tmp_path, monkeypatch, field)


def test_main_refuses_when_the_do_nothing_design_has_no_baseline(open_context, open_search, tmp_path, monkeypatch):
    real = bt.solve

    def stalled(context, mask, scenario):
        net, links, crossings, result = real(context, mask, scenario)
        result.converged = False
        return net, links, crossings, result

    monkeypatch.setattr(bt, "solve", stalled)
    with pytest.raises(RuntimeError, match="no baseline"):
        run_main(open_context, open_search, tmp_path, monkeypatch, write_field(tmp_path / "field.csv", {}))


def test_main_marks_each_record_and_removes_a_stale_voice_best_file(open_context, open_search, tmp_path, monkeypatch):
    field = write_field(tmp_path / "field.csv", {e: {"rat_running": 1.0} for e in NORTH})
    out = run_main(open_context, open_search, tmp_path, monkeypatch, field)
    assert {d["j_voice_status"] for d in out["trade_offs"]} == {bt.OPTIMISED}
    assert (tmp_path / "designs" / "voice_best.csv").exists()
    assert out["j_voice"]["field_edges_off_network"] == 0 and out["j_voice"]["floored_edges"]["rat_running"] == 0
    field.unlink()
    out = run_main(open_context, open_search, tmp_path, monkeypatch, field)
    assert "voice_best" not in out["picks"] and not (tmp_path / "designs" / "voice_best.csv").exists()


def test_a_design_whose_j_voice_cannot_be_scored_is_dropped_not_fatal(open_context, open_search, tmp_path,
                                                                       monkeypatch, capsys):
    real = bt.vo.j_voice

    def picky(field, metrics, baseline_metrics, *args):
        if metrics is not baseline_metrics and metrics[bt.vo.CAR_FLOW_VEH_H].get("6", 0.0) > 0:
            raise ValueError("cannot normalise")
        return real(field, metrics, baseline_metrics, *args)

    monkeypatch.setattr(bt.vo, "j_voice", picky)
    monkeypatch.setattr(bt, "WORKERS", 1)        # the patch only lives in this process
    field = write_field(tmp_path / "field.csv", {e: {"rat_running": 1.0} for e in NORTH})
    search_path = tmp_path / "design_search.json"
    search_path.write_text(json.dumps(open_search))
    for name, value in {"SEARCH_PATH": search_path, "TRADE_OFFS_PATH": tmp_path / "out.json",
                        "DESIGNS_DIR": tmp_path / "designs", "FIELD_PATH": field}.items():
        monkeypatch.setattr(bt, name, value)
    monkeypatch.setattr(bt, "load_context", lambda: open_context)
    bt.main()
    out = json.loads((tmp_path / "out.json").read_text())
    dropped = {d["mask"] for d in out["j_voice"]["dropped"]}
    assert dropped and all(r["reason"].startswith("baseline: J_voice can't be scored") for r in out["j_voice"]["dropped"])
    assert not dropped & {d["mask"] for d in out["trade_offs"]}
    assert "dropped" in capsys.readouterr().out


def test_main_explains_a_do_nothing_design_that_cannot_be_solved(open_context, open_search, tmp_path, monkeypatch):
    from engine.orchid import traffic_assignment as ta

    def cut_off(context, mask, scenario):
        raise ta.TrafficAssignmentError("zone cut off")

    monkeypatch.setattr(bt, "solve", cut_off)
    with pytest.raises(RuntimeError, match="can't be solved under baseline"):
        run_main(open_context, open_search, tmp_path, monkeypatch, write_field(tmp_path / "field.csv", {}))
