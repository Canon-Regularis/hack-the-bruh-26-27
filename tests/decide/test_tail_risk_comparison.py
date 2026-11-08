import json

import pytest

from src.decide import design_search as ds
from src.decide import tail_risk_comparison as trc
from tests.decide.two_route import build_context, build_graph

SCENARIOS = [{"id": "normal", "probability": 0.7}, {"id": "peak", "probability": 0.25},
             {"id": "event", "probability": 0.05}]


def design(mask, j_cars, scenarios=SCENARIOS):
    probabilities = [s["probability"] for s in scenarios]
    return {"mask": mask, "candidates": [f"c{mask}"], "feasible": True,
            "mean_j_car": sum(p * j for p, j in zip(probabilities, j_cars)),
            "j_risk": ds.weighted_cvar(j_cars, probabilities),
            "per_scenario": {s["id"]: {"j_car": j, "j_ped": 0.0} for s, j in zip(scenarios, j_cars)}}


STEADY = design(1, [110.0, 120.0, 130.0])
FAST_BUT_FRAGILE = design(2, [90.0, 100.0, 300.0])
INFEASIBLE = {"mask": 3, "candidates": ["c3"], "feasible": False, "reason": "unreachable"}


def test_optimal_designs_pick_lowest_mean_and_lowest_cvar():
    mean_optimal, cvar_optimal = trc.optimal_designs([INFEASIBLE, STEADY, FAST_BUT_FRAGILE])
    assert mean_optimal["mask"] == 2 and cvar_optimal["mask"] == 1


def test_optimal_designs_break_ties_by_the_lower_mask():
    twin = {**FAST_BUT_FRAGILE, "mask": 0}
    assert [d["mask"] for d in trc.optimal_designs([FAST_BUT_FRAGILE, twin])] == [0, 0]


def test_optimal_designs_need_a_feasible_design():
    with pytest.raises(ValueError, match="no feasible design"):
        trc.optimal_designs([INFEASIBLE])


def test_the_cvar_design_never_loses_more_than_the_mean_optimal_design():
    pool = [design(m, j) for m, j in enumerate([[90.0, 100.0, 300.0], [110.0, 120.0, 130.0], [100.0, 100.0, 100.0],
                                                 [80.0, 200.0, 220.0], [95.0, 95.0, 400.0], [120.0, 125.0, 126.0]])]
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            mean_optimal, cvar_optimal = trc.optimal_designs([pool[i], pool[j]])
            assert trc.tail_loss(cvar_optimal, SCENARIOS, 0.9)[0] <= trc.tail_loss(mean_optimal, SCENARIOS, 0.9)[0]


def test_tail_loss_compares_cvar_with_the_mean():
    loss, worst_only = trc.tail_loss(STEADY, SCENARIOS, 0.9)
    assert loss == pytest.approx((STEADY["j_risk"] / STEADY["mean_j_car"] - 1) * 100)
    assert not worst_only


@pytest.mark.parametrize("probabilities, j_cars, worst_only", [
    ([0.8, 0.2], [100.0, 200.0], True),
    ([0.9, 0.1], [100.0, 200.0], True),
    ([0.95, 0.05], [100.0, 200.0], False),
    ([0.9, 0.05, 0.05], [100.0, 200.0, 200.0], True),
    ([0.9, 1 - 0.9], [100.0, 200.0], True),
])
def test_tail_is_the_worst_scenario_when_it_carries_the_tail(probabilities, j_cars, worst_only):
    scenarios = [{"id": f"s{i}", "probability": p} for i, p in enumerate(probabilities)]
    d = design(0, j_cars, scenarios)
    assert trc.tail_loss(d, scenarios, 0.9)[1] is worst_only
    if worst_only:
        assert d["j_risk"] == pytest.approx(max(j_cars))


def test_summary_sentence_reads_right_for_two_designs_and_for_one():
    assert trc.summary_sentence(41.7, 9.12, "in the worst scenario", False) == (
        "In the worst scenario, the mean-optimal design loses 41.7% against its average day; "
        "the CVaR design loses 9.1%.")
    assert trc.summary_sentence(5.0, 5.0, "in the worst 10% of scenarios", True) == (
        "In the worst 10% of scenarios, the mean-optimal design, which is also the CVaR-optimal design, "
        "loses 5.0% against its average day.")


def run_main(search, tmp_path, monkeypatch, capsys):
    path = tmp_path / "design_search.json"
    path.write_text(json.dumps(search))
    monkeypatch.setattr(trc, "SEARCH_PATH", path)
    monkeypatch.setattr(trc, "TAIL_RISK_PATH", tmp_path / "decide" / "tail_risk_comparison.json")
    trc.main()
    return json.loads((tmp_path / "decide" / "tail_risk_comparison.json").read_text()), capsys.readouterr().out


def test_main_writes_x_y_and_the_scenario_count(tmp_path, monkeypatch, capsys):
    search = {"alpha": 0.9, "hour": "08:00-09:00 weekday", "scenarios": SCENARIOS,
              "designs": [STEADY, FAST_BUT_FRAGILE, INFEASIBLE]}
    out, printed = run_main(search, tmp_path, monkeypatch, capsys)
    assert out["x_pct"] == pytest.approx(trc.tail_loss(FAST_BUT_FRAGILE, SCENARIOS, 0.9)[0])
    assert out["y_pct"] == pytest.approx(trc.tail_loss(STEADY, SCENARIOS, 0.9)[0])
    assert out["x_pct"] > out["y_pct"] and out["scenario_count"] == 3 and not out["same_design"]
    assert out["tail"] == "in the worst 10% of scenarios" and printed.strip() == out["sentence"]
    assert out["mean_optimal"]["mask"] == 2 and out["cvar_optimal"]["mask"] == 1


def test_main_on_a_real_search_run(tmp_path, monkeypatch, capsys):
    context = build_context(build_graph(), tmp_path)
    results = {(mask, s["id"]): ds.evaluate(context, mask, s)
               for mask in range(1 << len(context["candidates"])) for s in context["scenarios"]}
    search = {"alpha": ds.ALPHA, "hour": context["hour"],
              "scenarios": [{"id": s["id"], "probability": s["probability"]} for s in context["scenarios"]],
              "designs": ds.summarise(context, results)}
    out, _ = run_main(search, tmp_path, monkeypatch, capsys)
    mean_optimal, cvar_optimal = trc.optimal_designs(search["designs"])
    assert (out["mean_optimal"]["mask"], out["cvar_optimal"]["mask"]) == (mean_optimal["mask"], cvar_optimal["mask"])
    assert out["x_pct"] == pytest.approx(trc.tail_loss(mean_optimal, search["scenarios"], ds.ALPHA)[0])
    assert out["y_pct"] == pytest.approx(trc.tail_loss(cvar_optimal, search["scenarios"], ds.ALPHA)[0])
    assert out["scenario_count"] == 2 and out["tail"] == "in the worst scenario"


def test_main_says_worst_scenario_only_when_both_tails_are_one_scenario(tmp_path, monkeypatch, capsys):
    scenarios = [{"id": "normal", "probability": 0.88}, {"id": "peak", "probability": 0.07},
                 {"id": "event", "probability": 0.05}]
    fragile = design(1, [90.0, 100.0, 300.0], scenarios)
    steady = design(2, [150.0, 120.0, 110.0], scenarios)
    assert not trc.tail_loss(fragile, scenarios, 0.9)[1] and trc.tail_loss(steady, scenarios, 0.9)[1]
    search = {"alpha": 0.9, "hour": "08:00-09:00 weekday", "scenarios": scenarios, "designs": [fragile, steady]}
    out, _ = run_main(search, tmp_path, monkeypatch, capsys)
    assert out["mean_optimal"]["mask"] == 1 and out["cvar_optimal"]["mask"] == 2
    assert out["tail"] == "in the worst 10% of scenarios"
