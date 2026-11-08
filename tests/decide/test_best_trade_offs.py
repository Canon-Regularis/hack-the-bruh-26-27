import csv
import json
import math
from itertools import combinations

import pytest

from src.decide import best_trade_offs as bt
from src.decide import design_search as ds
from src.decide.baseline_user_equilibrium import ue_rows
from tests.decide.two_route import build_context, build_graph


def design(mask, j_car, j_ped, j_risk):
    return {"mask": mask, "candidates": [f"c{mask}"], "feasible": True,
            "mean_j_car": j_car, "mean_j_ped": j_ped, "j_risk": j_risk}


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
    feasible = [E, C, A, DOMINATED, D, B]
    normalised = bt.normalise([[d[name] for name in bt.OBJECTIVES] for d in feasible])
    on_best = [normalised[i] for i, d in enumerate(feasible) if d is not DOMINATED]
    assert volume == pytest.approx(box_union_volume(on_best, [bt.REFERENCE] * 3))


def test_pick_designs_uses_the_cut_projection_for_the_knee():
    picks = bt.pick_designs([A, B, C, D, E])
    assert {role: d["mask"] for role, d in picks.items()} == {"car_best": 1, "pedestrian_best": 5, "knee": 2}


def test_pick_designs_breaks_ties_by_the_lower_mask():
    twin = {**A, "mask": 0}
    assert bt.pick_designs([A, twin, E])["car_best"]["mask"] == 0


def test_design_diff_lists_edge_ids_and_combined_factors():
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
    assert bt.design_diff({"mask": 0b1111}, candidates) == {
        "closed_edge_ids": ["0", "1", "2", "3"],
        "capacity_factors": {"4": 0.5, "5": pytest.approx(0.4), "10": pytest.approx(0.4)},
        "signalised_crossing_nodes": [1, 9],
    }
    assert bt.design_diff({"mask": 0}, candidates) == {"closed_edge_ids": [], "capacity_factors": {},
                                                       "signalised_crossing_nodes": []}


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    return build_context(build_graph(), tmp_path_factory.mktemp("decide"))


@pytest.fixture(scope="module")
def search(context):
    results = {(mask, s["id"]): ds.evaluate(context, mask, s)
               for mask in range(1 << len(context["candidates"])) for s in context["scenarios"]}
    return {"hour": context["hour"], "alpha": ds.ALPHA, "candidates": [c["id"] for c in context["candidates"]],
            "designs": ds.summarise(context, results)}


def run_main(context, search, tmp_path, monkeypatch):
    search_path = tmp_path / "design_search.json"
    search_path.write_text(json.dumps(search))
    monkeypatch.setattr(bt, "SEARCH_PATH", search_path)
    monkeypatch.setattr(bt, "TRADE_OFFS_PATH", tmp_path / "best_trade_offs.json")
    monkeypatch.setattr(bt, "DESIGNS_DIR", tmp_path / "designs")
    monkeypatch.setattr(bt, "load_context", lambda: context)
    bt.main()
    return json.loads((tmp_path / "best_trade_offs.json").read_text())


def test_main_writes_picks_and_their_baseline_flows(context, search, tmp_path, monkeypatch, capsys):
    out = run_main(context, search, tmp_path, monkeypatch)
    feasible = [d for d in search["designs"] if d["feasible"]]
    assert {d["mask"] for d in out["trade_offs"]} <= {d["mask"] for d in feasible}
    assert set(out["picks"]) == {"car_best", "pedestrian_best", "knee"}
    assert out["hour"] == "08:00-09:00 weekday" and out["reference"] == [1.1, 1.1, 1.1]
    for role, pick in out["picks"].items():
        net, _, _, result = ds.solve(context, pick["mask"], bt.BASELINE)
        expected = [{key: str(value) for key, value in row.items()} for row in ue_rows(net, result)]
        with (tmp_path / "designs" / f"{role}.csv").open() as f:
            assert list(csv.DictReader(f)) == expected
        assert pick["diff"] == bt.design_diff(pick, context["candidates"])
    assert capsys.readouterr().out.startswith(f"{len(out['trade_offs'])} best trade-offs, hypervolume ")


def test_main_refuses_a_search_run_on_other_candidates(context, search, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="candidates have changed"):
        run_main(context, {**search, "candidates": ["something-else", "bus-lane-south"]}, tmp_path, monkeypatch)


def test_main_refuses_a_search_with_no_feasible_design(context, search, tmp_path, monkeypatch):
    infeasible = [{**INFEASIBLE, "mask": d["mask"]} for d in search["designs"]]
    with pytest.raises(ValueError, match="no feasible design"):
        run_main(context, {**search, "designs": infeasible}, tmp_path, monkeypatch)
