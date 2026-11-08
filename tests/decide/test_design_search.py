import json
import signal
from types import SimpleNamespace

import pytest

from engine.orchid import stochastic_optimisation as so
from engine.orchid import traffic_assignment as ta
from src.decide import design_search as ds
from src.decide.pedestrian_crossing_delay import j_ped
from tests.decide.two_route import NORMAL, SOUTH_SHUT, build_context, build_graph


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    return build_context(build_graph(), tmp_path_factory.mktemp("decide"))


@pytest.mark.parametrize("losses, alpha", [
    ([5, 1, 3, 9, 7, 2, 8, 4, 6, 10], 0.9),
    ([5, 1, 3, 9, 7, 2, 8, 4, 6, 10], 0.95),
    ([4.0, 4.0, 4.0, 4.0], 0.9),
    ([3.5, 7.25], 0.5),
    ([2, 9, 9, 1, 7, 7, 7], 0.9),
    ([0.1 * i * i for i in range(20)], 0.9),
])
def test_weighted_cvar_matches_orchid_with_equal_weights(losses, alpha):
    equal = [1 / len(losses)] * len(losses)
    assert ds.weighted_cvar(losses, equal, alpha) == pytest.approx(so.conditional_value_at_risk(losses, alpha))


def test_weighted_cvar_is_the_mean_of_the_worst_tail():
    assert ds.weighted_cvar([100, 10, 20], [0.05, 0.5, 0.45]) == pytest.approx(60.0)


def test_evaluate_baseline_matches_a_direct_solve(context):
    run = ds.evaluate(context, 0, NORMAL)
    direct = ta.user_equilibrium(context["net"], context["od"], method="b", max_iter=ds.MAX_ITERATIONS)
    assert run["feasible"] and run["j_car"] == pytest.approx(direct.total_travel_time)
    assert run["j_ped"] == pytest.approx(j_ped(context["crossings"], dict(zip(context["link_to_edge"], direct.flows))))
    assert run["relative_gap"] <= 1e-6


def test_evaluate_closing_the_crossed_road_leaves_no_traffic_at_its_crossing(context):
    run = ds.evaluate(context, 0b11, NORMAL)
    assert run["feasible"] and run["j_ped"] == 0.0
    assert run["j_car"] > ds.evaluate(context, 0, NORMAL)["j_car"]


def test_evaluate_scales_demand_by_the_scenario(context):
    busier = ds.evaluate(context, 0, {**NORMAL, "multiplier": 1.5})
    assert busier["j_car"] > 1.5 * ds.evaluate(context, 0, NORMAL)["j_car"]


def test_evaluate_marks_a_cut_off_zone_infeasible(context):
    run = ds.evaluate(context, 0b01, SOUTH_SHUT)
    assert not run["feasible"] and "unreachable" in run["reason"]


def test_evaluate_retries_with_more_iterations(context, monkeypatch):
    real, calls = ta.user_equilibrium, []

    def solve(net, od, **kw):
        calls.append(kw["max_iter"])
        assert kw["tol"] == ds.TOLERANCE
        result = real(net, od, **kw)
        return result if len(calls) > 1 else SimpleNamespace(converged=False)

    monkeypatch.setattr(ds.ta, "user_equilibrium", solve)
    run = ds.evaluate(context, 0, NORMAL)
    assert calls == [ds.MAX_ITERATIONS, ds.MAX_ITERATIONS * ds.RETRY_ITERATION_FACTOR] and run["feasible"]


def test_evaluate_marks_a_run_that_still_does_not_converge(context, monkeypatch):
    stuck = SimpleNamespace(converged=False, relative_gap=0.02, iterations=1000)
    monkeypatch.setattr(ds.ta, "user_equilibrium", lambda *a, **kw: stuck)
    assert ds.evaluate(context, 0, NORMAL) == {"feasible": False,
                                               "reason": "did not converge: relative gap 2.00e-02 after 1000 iterations"}


def test_fingerprint_changes_when_any_input_changes(context, monkeypatch):
    base = ds.fingerprint(context)
    assert ds.fingerprint({**context}) == base
    edited = {**context, "candidates": [{**context["candidates"][1], "capacity_factor": 0.6}, context["candidates"][0]]}
    assert ds.fingerprint(edited) != base
    assert ds.fingerprint({**context, "scenarios": [NORMAL, {**SOUTH_SHUT, "multiplier": 2.0}]}) != base
    same_total = ta.ODMatrix({(0, 3): 300.0, (3, 0): 600.0, (0, 4): 100.0})
    assert same_total.total() == context["od"].total() and ds.fingerprint({**context, "od": same_total}) != base
    assert ds.fingerprint({**context, "net": context["net"].with_scaled_capacity(0, 0.5)}) != base
    signalised = [{**c, "signalised": True} for c in context["crossings"]]
    assert ds.fingerprint({**context, "crossings": signalised}) != base
    for module, name, value in [(ds.pcd, "CYCLE_S", 120.0), (ds.pcd, "MAX_WAIT_S", 90.0), (ds, "MAX_ITERATIONS", 50),
                                 (ds, "TOLERANCE", 1e-6)]:
        with monkeypatch.context() as patch:
            patch.setattr(module, name, value)
            assert ds.fingerprint(context) != base
    assert ds.fingerprint(context) == base


def write_checkpoint(path, header, records, tail=""):
    path.write_text("".join(json.dumps(x) + "\n" for x in [{"fingerprint": header}, *records]) + tail)


def test_load_checkpoint_reads_finished_runs(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    assert ds.load_checkpoint(path, {"a": 1}) == {}
    record = {"mask": 1, "scenario": "normal", "feasible": True, "j_car": 5.0, "j_ped": 1.0}
    write_checkpoint(path, {"a": 1}, [record])
    assert ds.load_checkpoint(path, {"a": 1}) == {(1, "normal"): record}


def test_load_checkpoint_refuses_another_setup(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    write_checkpoint(path, {"a": 1}, [])
    with pytest.raises(ValueError, match="belongs to a different set"):
        ds.load_checkpoint(path, {"a": 2})


def test_load_checkpoint_drops_a_half_written_last_line(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    record = {"mask": 0, "scenario": "normal", "feasible": True, "j_car": 5.0, "j_ped": 1.0}
    write_checkpoint(path, {"a": 1}, [record], tail='{"mask": 1, "scen')
    assert ds.load_checkpoint(path, {"a": 1}) == {(0, "normal"): record}
    assert path.read_text().endswith("}\n") and len(path.read_text().splitlines()) == 2


def test_summarise_weights_scenarios_and_explains_infeasible_designs(context):
    def ok(j_car, j_ped):
        return {"feasible": True, "j_car": j_car, "j_ped": j_ped}

    results = {(0, "normal"): ok(100.0, 10.0), (0, "event-south-closed"): ok(300.0, 20.0),
               (1, "normal"): ok(120.0, 0.0), (1, "event-south-closed"): {"feasible": False, "reason": "unreachable"},
               (2, "normal"): ok(110.0, 10.0), (2, "event-south-closed"): ok(300.0, 20.0),
               (3, "normal"): ok(130.0, 0.0), (3, "event-south-closed"): {"feasible": False, "reason": "unreachable"}}
    designs = ds.summarise(context, results)
    assert [d["candidates"] for d in designs] == [[], ["close-north"], ["bus-lane-south"],
                                                  ["close-north", "bus-lane-south"]]
    assert designs[0]["mean_j_car"] == pytest.approx(0.8 * 100 + 0.2 * 300)
    assert designs[0]["mean_j_ped"] == pytest.approx(0.8 * 10 + 0.2 * 20)
    assert designs[0]["j_risk"] == pytest.approx(ds.weighted_cvar([100.0, 300.0], [0.8, 0.2]))
    assert designs[1] == {"mask": 1, "candidates": ["close-north"], "feasible": False,
                          "reason": "event-south-closed: unreachable"}


def run_main(context, tmp_path, monkeypatch, capsys, *argv):
    monkeypatch.setattr(ds, "load_context", lambda: context)
    monkeypatch.setattr(ds, "CHECKPOINT_PATH", tmp_path / "decide" / "checkpoint.jsonl")
    monkeypatch.setattr(ds, "RESULTS_PATH", tmp_path / "decide" / "design_search.json")
    monkeypatch.setattr(ds, "WORKERS", 2)
    ds.main(argv)
    results = tmp_path / "decide" / "design_search.json"
    return capsys.readouterr().out, json.loads(results.read_text()) if results.exists() else None


def test_shards_split_the_runs_and_merge_into_the_same_summary(context, tmp_path, monkeypatch, capsys):
    _, plain = run_main(context, tmp_path / "plain", monkeypatch, capsys)
    split = tmp_path / "split"
    out0, none0 = run_main(context, split, monkeypatch, capsys, "--shard", "0/2")
    out1, none1 = run_main(context, split, monkeypatch, capsys, "--shard", "1/2")
    assert none0 is None and none1 is None and "4 to go" in out0 and "4 to go" in out1
    shards = [split / "decide" / f"checkpoint.shard{k}of2.jsonl" for k in range(2)]
    keys = [{(r["mask"], r["scenario"]) for r in map(json.loads, p.read_text().splitlines()[1:])} for p in shards]
    assert len(keys[0]) == len(keys[1]) == 4 and not keys[0] & keys[1]

    out, merged = run_main(context, split, monkeypatch, capsys, "--merge", "2")
    assert "8 runs already checkpointed, 0 to go" in out and merged == plain
    out, again = run_main(context, split, monkeypatch, capsys, "--merge", "2")
    assert len((split / "decide" / "checkpoint.jsonl").read_text().splitlines()) == 9 and again == plain


def test_shard_rejects_an_index_outside_the_count(context, tmp_path, monkeypatch, capsys):
    with pytest.raises(ValueError, match="0 <= K < N"):
        run_main(context, tmp_path, monkeypatch, capsys, "--shard", "2/2")


def test_main_runs_every_design_and_resumes_from_the_checkpoint(context, tmp_path, monkeypatch, capsys):
    out, first = run_main(context, tmp_path, monkeypatch, capsys)
    checkpoint = tmp_path / "decide" / "checkpoint.jsonl"
    assert "0 runs already checkpointed, 8 to go" in out and len(checkpoint.read_text().splitlines()) == 9
    assert [d["feasible"] for d in first["designs"]] == [True, False, True, False]
    assert first["hour"] == "08:00-09:00 weekday" and first["alpha"] == 0.9
    assert first["fingerprint"] == json.loads(json.dumps(ds.fingerprint(context)))
    assert "2 of 4 designs feasible" in out

    out, again = run_main(context, tmp_path, monkeypatch, capsys)
    assert "8 runs already checkpointed, 0 to go" in out and again == first

    lines = checkpoint.read_text().splitlines()
    checkpoint.write_text("\n".join(lines[:4]) + "\n" + lines[4][:10])
    out, resumed = run_main(context, tmp_path, monkeypatch, capsys)
    assert "3 runs already checkpointed, 5 to go" in out
    assert len(checkpoint.read_text().splitlines()) == 9 and resumed == first


def test_main_warns_about_abstained_keep_open_clauses(context, tmp_path, monkeypatch, capsys):
    warned = {**context, "abstained_keep_open": [("servicing", "source 'tro-99' is not cached")]}
    out, _ = run_main(warned, tmp_path, monkeypatch, capsys)
    assert out.startswith("warning: keep-open clause servicing abstained (source 'tro-99' is not cached)")


def test_sigterm_becomes_system_exit_so_the_pool_shuts_down():
    with pytest.raises(SystemExit, match="stopped by signal 15"):
        ds._stop(15, None)


def test_main_stops_on_sigterm_while_running_and_restores_the_handler_after(context, tmp_path, monkeypatch, capsys):
    during = []
    real_as_completed = ds.as_completed
    monkeypatch.setattr(ds, "as_completed", lambda fs: during.append(signal.getsignal(signal.SIGTERM)) or real_as_completed(fs))
    def before(*_):
        pass

    original = signal.signal(signal.SIGTERM, before)
    try:
        run_main(context, tmp_path, monkeypatch, capsys)
        assert during == [ds._stop] and signal.getsignal(signal.SIGTERM) is before
    finally:
        signal.signal(signal.SIGTERM, original)


def test_repairing_a_half_written_line_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    write_checkpoint(path, {"a": 1}, [], tail='{"mask": 1')
    ds.load_checkpoint(path, {"a": 1})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint.jsonl"]
