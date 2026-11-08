"""Exhaustive robust search over designs x scenarios (ALG-S2-3, #28).

Every mask of #26's allowed candidates runs under each of #27's kept scenarios. Runs go
in parallel worker processes and every result is checkpointed, so the search can resume.
Stand-in for orchid's held network_design. Run from the repo root:
``python -m src.decide.design_search``.
"""

import hashlib
import json
import math
import multiprocessing
import os
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import accumulate
from pathlib import Path

from engine.orchid import traffic_assignment as ta
from src.decide.baseline_user_equilibrium import OD_PATH, load_od
from src.decide.interventions import apply_design, checked_candidates
from src.decide.network import link_maps, to_network
from src.decide import pedestrian_crossing_delay as pcd
from src.decide.pedestrian_crossing_delay import find_crossings, j_ped
from src.decide.scenarios import SCENARIOS_PATH
from src.graph.build_graph import load_graph

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = ROOT / "data" / "decide" / "design_search_checkpoint.jsonl"
RESULTS_PATH = ROOT / "data" / "decide" / "design_search.json"

ALPHA = 0.9
MAX_ITERATIONS = 200
RETRY_ITERATION_FACTOR = 5
MAX_CANDIDATES = 8
WORKERS = max(1, (os.cpu_count() or 2) - 1)
TAIL_TOLERANCE = 1e-12

_worker_context: dict = {}


def weighted_cvar(losses: list[float], probabilities: list[float], alpha: float = ALPHA) -> float:
    """Return the probability-weighted CVaR: the expected loss in the worst 1 - alpha tail.

    Same Rockafellar-Uryasev formula as orchid's conditional_value_at_risk, which it equals
    when every probability is the same. VaR is the smallest loss whose cumulative
    probability reaches alpha.
    """
    ordered = sorted(zip(losses, probabilities))
    cumulative = accumulate(p for _, p in ordered)
    var = next(loss for (loss, _), c in zip(ordered, cumulative) if c >= alpha - TAIL_TOLERANCE)
    return var + math.fsum(p * max(0.0, loss - var) for loss, p in ordered) / (1.0 - alpha)


def load_context() -> dict:
    """Load everything a run needs once: network, link -> edge ids, OD matrix, crossings,
    allowed candidates and the kept scenarios.
    """
    graph = load_graph()
    net, node_ids = to_network(graph)
    od, hour = load_od(OD_PATH, node_ids)
    crossings = find_crossings(graph)
    candidates, _, _, abstained = checked_candidates(graph, crossings)
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError(f"{len(candidates)} candidates allowed, but the search is capped at {MAX_CANDIDATES}; "
                         "trim data/decide/candidates.json")
    return {"net": net, "link_to_edge": link_maps(net)[1], "od": od, "hour": hour, "crossings": crossings,
            "candidates": candidates, "abstained_keep_open": abstained,
            "scenarios": json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))["kept"]}


def chosen(candidates: list[dict], mask: int) -> list[dict]:
    """Return the candidates a mask switches on: bit i is candidate i."""
    return [c for bit, c in enumerate(candidates) if mask >> bit & 1]


def solve(context: dict, mask: int, scenario: dict
          ) -> tuple[ta.Network, list[str], list[dict], ta.AssignmentResult]:
    """Apply a design plus a scenario's closure, scale the OD, and solve with Algorithm B,
    retrying once with more iterations if it doesn't converge.

    Returns the design's network, link -> edge ids, crossings and the result. Raises
    TrafficAssignmentError if a closure cuts a zone off.
    """
    net, links, crossings = apply_design(context["net"], context["link_to_edge"], context["crossings"],
                                         chosen(context["candidates"], mask), scenario["closed_edge_ids"])
    od = context["od"].scaled(scenario["multiplier"])
    result = ta.user_equilibrium(net, od, method="b", max_iter=MAX_ITERATIONS)
    if not result.converged:
        result = ta.user_equilibrium(net, od, method="b", max_iter=MAX_ITERATIONS * RETRY_ITERATION_FACTOR)
    return net, links, crossings, result


def evaluate(context: dict, mask: int, scenario: dict) -> dict:
    """Run one design under one scenario.

    Returns J_car (total travel time) and J_ped, or why the run is infeasible: a zone cut
    off, or no convergence even after the retry.
    """
    try:
        _, links, crossings, result = solve(context, mask, scenario)
    except ta.TrafficAssignmentError as e:
        return {"feasible": False, "reason": str(e)}
    if not result.converged:
        return {"feasible": False, "reason": f"did not converge: relative gap {result.relative_gap:.2e} "
                                             f"after {result.iterations} iterations"}
    return {"feasible": True, "j_car": result.total_travel_time, "j_ped": j_ped(crossings, dict(zip(links, result.flows))),
            "relative_gap": result.relative_gap, "iterations": result.iterations}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()[:16]


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def fingerprint(context: dict) -> dict:
    """Return everything a run's result depends on, so a stale checkpoint or result is refused.

    Covers each candidate and scenario, hashes of the OD trips, every link (ends, free-flow
    time, capacity, BPR shape) and every crossing, and the solver and J_ped settings. Editing
    a candidate, re-estimating the OD matrix, rebuilding the graph or changing an assumed
    value all invalidate old results.
    """
    return {
        "candidates": [[c["id"], c["kind"], sorted(c["edges"], key=int), c["capacity_factor"], c["node"]]
                       for c in context["candidates"]],
        "scenarios": [[s["id"], s["multiplier"], s["probability"], s["closed_edge_ids"]] for s in context["scenarios"]],
        "od": _digest(context["od"].items()),
        "links": _digest([[link.name, link.tail, link.head, link.vdf.free_flow_time(), link.capacity,
                           link.vdf.alpha, link.vdf.beta] for link in context["net"].links]),
        "crossings": _digest(context["crossings"]),
        "settings": {"max_iterations": MAX_ITERATIONS, "retry_iteration_factor": RETRY_ITERATION_FACTOR,
                     "walking_speed_mps": pcd.WALKING_SPEED_MPS, "cycle_s": pcd.CYCLE_S,
                     "walk_green_s": pcd.WALK_GREEN_S, "zebra_delay_s": pcd.ZEBRA_DELAY_S,
                     "max_wait_s": pcd.MAX_WAIT_S},
    }


def load_checkpoint(path: Path, expected: dict) -> dict[tuple[int, str], dict]:
    """Read finished (mask, scenario id) results, refusing a checkpoint from another setup.

    The header is written atomically, so only a record line can be half-written. A crash
    mid-write leaves one at the end; it is dropped and the file atomically rewritten
    without it, so later appends start on a clean line.
    """
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not text.endswith("\n"):
        lines.pop()
        _write_atomic(path, "".join(line + "\n" for line in lines))
    if json.loads(lines[0])["fingerprint"] != expected:
        raise ValueError(f"{path} belongs to a different set of candidates, scenarios or OD; "
                         "delete it to start the search over")
    records = map(json.loads, lines[1:])
    return {(r["mask"], r["scenario"]): r for r in records}


def summarise(context: dict, results: dict[tuple[int, str], dict]) -> list[dict]:
    """Return one entry per mask: its candidate ids, whether it's feasible (and why not),
    and the weighted mean J_car, weighted mean J_ped and J_risk across scenarios.
    """
    scenarios, candidates = context["scenarios"], context["candidates"]
    probabilities = [s["probability"] for s in scenarios]
    designs = []
    for mask in range(1 << len(candidates)):
        runs = [results[(mask, s["id"])] for s in scenarios]
        entry = {"mask": mask, "candidates": [c["id"] for c in chosen(candidates, mask)]}
        failed = [f"{s['id']}: {r['reason']}" for s, r in zip(scenarios, runs) if not r["feasible"]]
        if failed:
            entry.update(feasible=False, reason="; ".join(failed))
        else:
            j_car = [r["j_car"] for r in runs]
            entry.update(feasible=True,
                         mean_j_car=math.fsum(p * j for p, j in zip(probabilities, j_car)),
                         mean_j_ped=math.fsum(p * r["j_ped"] for p, r in zip(probabilities, runs)),
                         j_risk=weighted_cvar(j_car, probabilities),
                         per_scenario={s["id"]: {"j_car": r["j_car"], "j_ped": r["j_ped"]}
                                       for s, r in zip(scenarios, runs)})
        designs.append(entry)
    return designs


def _stop(signum: int, _frame) -> None:
    """Turn SIGTERM into SystemExit, so the pool is shut down and no worker is orphaned."""
    raise SystemExit(f"stopped by signal {signum}")


def _start_worker(context: dict) -> None:
    _worker_context.update(context)


def _run(mask: int, scenario_index: int) -> dict:
    return evaluate(_worker_context, mask, _worker_context["scenarios"][scenario_index])


def main() -> None:
    """Run every unfinished (mask, scenario) in parallel, checkpointing each, then write the summary."""
    context = load_context()
    for cid, reason in context["abstained_keep_open"]:
        print(f"warning: keep-open clause {cid} abstained ({reason}), so it forbids nothing")
    expected = fingerprint(context)
    results = load_checkpoint(CHECKPOINT_PATH, expected)
    scenarios = context["scenarios"]
    todo = [(mask, i) for mask in range(1 << len(context["candidates"])) for i, s in enumerate(scenarios)
            if (mask, s["id"]) not in results]
    print(f"{len(results)} runs already checkpointed, {len(todo)} to go")

    if todo:
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not CHECKPOINT_PATH.exists():
            _write_atomic(CHECKPOINT_PATH, json.dumps({"fingerprint": expected}) + "\n")
        step = max(1, len(todo) // 20)
        pool = ProcessPoolExecutor(min(WORKERS, len(todo)), mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_start_worker, initargs=(context,))
        previous_handler = signal.signal(signal.SIGTERM, _stop)
        try:
            with CHECKPOINT_PATH.open("a", encoding="utf-8") as checkpoint:
                runs = {pool.submit(_run, mask, i): (mask, scenarios[i]["id"]) for mask, i in todo}
                for done, future in enumerate(as_completed(runs), 1):
                    mask, scenario_id = runs[future]
                    record = {"mask": mask, "scenario": scenario_id, **future.result()}
                    checkpoint.write(json.dumps(record) + "\n")
                    checkpoint.flush()
                    results[(mask, scenario_id)] = record
                    if done % step == 0 or done == len(todo):
                        print(f"  {done}/{len(todo)} runs done")
        finally:
            pool.shutdown(cancel_futures=True)
            signal.signal(signal.SIGTERM, previous_handler)

    designs = summarise(context, results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "hour": context["hour"],
        "alpha": ALPHA,
        "fingerprint": expected,
        "candidates": [c["id"] for c in context["candidates"]],
        "scenarios": [{"id": s["id"], "probability": s["probability"]} for s in scenarios],
        "designs": designs,
    }, indent=1), encoding="utf-8")
    feasible = sum(d["feasible"] for d in designs)
    print(f"{feasible} of {len(designs)} designs feasible -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
