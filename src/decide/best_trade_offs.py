"""The best trade-offs, their hypervolume and the designs to show (ALG-S2-4, #29; J_voice, #36).

The best trade-offs are the exact Pareto front: designs no other design beats on every
objective. Reads #28's design search and finds them over (mean J_car, mean J_ped, J_risk),
plus J_voice (#36) when #25's concern field allows, then picks the car-best,
pedestrian-best and knee designs (and voice-best). Each pick is re-solved under #24's
baseline conditions so its per-edge file compares line for line with the baseline.

J_voice is scored under baseline demand: every feasible design is re-solved once under
BASELINE, in parallel. What happens depends on the field (the plan's kill switches):

- no field file: three objectives, no J_voice ("drop J_voice first if behind");
- a per-street field: J_voice is computed and shown on every design and pick, but not
  optimised, so still three objectives and no voice-best ("no locatable complaints");
- a located field: J_voice is the fourth objective, with a voice-best pick.

A design that the search found feasible but that does not converge under BASELINE is
dropped from the trade-offs, with its reason. Stand-in for the held policy_compiler's
pareto_set. Run from the repo root: ``python -m src.decide.best_trade_offs``.
"""

import csv
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from engine.orchid import multi_objective as mo
from engine.orchid import traffic_assignment as ta
from src.concerns.concern_field import LOCATED
from src.decide import design_search as ds
from src.decide import voice_objective as vo
from src.decide.baseline_user_equilibrium import UE_FIELDS, ue_rows
from src.decide.design_search import RESULTS_PATH as SEARCH_PATH
from src.decide.design_search import chosen, fingerprint, load_context, solve
from src.decide.interventions import CLOSE, SIGNALISE

ROOT = Path(__file__).resolve().parents[2]
TRADE_OFFS_PATH = ROOT / "data" / "decide" / "best_trade_offs.json"
DESIGNS_DIR = ROOT / "data" / "decide" / "designs"
FIELD_PATH = vo.FIELD_PATH

OBJECTIVES = ("mean_j_car", "mean_j_ped", "j_risk")
VOICE = "j_voice"
REFERENCE = 1.1
BASELINE = {"multiplier": 1.0, "closed_edge_ids": []}
WORKERS = ds.WORKERS

OPTIMISED, SHOWN, NOT_COMPUTED = "optimised", "shown, not optimised", "not computed"

_worker_context: dict = {}


def normalise(F: list[list[float]]) -> list[list[float]]:
    """Scale each objective so the best row is 0 and the worst is 1.

    An objective that is the same for every design becomes 0.
    """
    columns = list(zip(*F))
    lows = [min(column) for column in columns]
    spans = [max(column) - low for column, low in zip(columns, lows)]
    return [[(value - low) / span if span else 0.0 for value, low, span in zip(row, lows, spans)] for row in F]


def trade_offs_and_hypervolume(designs: list[dict], objectives: tuple = OBJECTIVES) -> tuple[list[dict], float]:
    """Return the feasible designs that are best trade-offs over the objectives (the Pareto
    front), and their hypervolume against REFERENCE on every axis.

    The objectives are normalised over the best trade-offs themselves, from their ideal (0)
    to their nadir (1), so REFERENCE sits just past the nadir and dominated designs can't
    stretch the scale.
    """
    feasible = [d for d in designs if d["feasible"]]
    if not feasible:
        return [], 0.0
    F = [[d[name] for name in objectives] for d in feasible]
    best = sorted(mo.find_pareto_front(F))
    volume = mo.hypervolume(normalise([F[i] for i in best]), [REFERENCE] * len(objectives))
    return [feasible[i] for i in best], volume


def pick_designs(trade_offs: list[dict], objectives: tuple = OBJECTIVES) -> dict[str, dict]:
    """Return the car-best (lowest mean J_car), pedestrian-best (lowest mean J_ped) and knee
    designs, plus voice-best (lowest J_voice) when J_voice is an objective. The knee comes
    from the J_car vs J_risk projection, cut to its own best trade-offs first.
    Ties go to the lower mask.
    """
    projection = [[d["mean_j_car"], d["j_risk"]] for d in trade_offs]
    cut = sorted(mo.find_pareto_front(projection))
    picks = {
        "car_best": min(trade_offs, key=lambda d: (d["mean_j_car"], d["mask"])),
        "pedestrian_best": min(trade_offs, key=lambda d: (d["mean_j_ped"], d["mask"])),
        "knee": trade_offs[cut[mo.knee_point([projection[i] for i in cut])]],
    }
    if VOICE in objectives:
        picks["voice_best"] = min(trade_offs, key=lambda d: (d[VOICE], d["mask"]))
    return picks


def design_diff(design: dict, candidates: list[dict], car_edges: set[str]) -> dict:
    """Return what a design changes, as string edge ids: closed edges, edges whose capacity
    changes (with the combined factor), and signalised crossing nodes.

    Only car edges count, and capacity changes on closed edges are left out, as apply_design
    does; a street's pedestrian sections are never shown as closed.
    """
    picked = chosen(candidates, design["mask"])
    closed = car_edges & set().union(*(c["edges"] for c in picked if c["kind"] == CLOSE))
    open_car_edges = car_edges - closed
    factors = {}
    for c in picked:
        if c["kind"] != CLOSE and c["capacity_factor"] is not None:
            for eid in c["edges"] & open_car_edges:
                factors[eid] = factors.get(eid, 1.0) * c["capacity_factor"]
    return {
        "closed_edge_ids": sorted(closed, key=int),
        "capacity_factors": dict(sorted(factors.items(), key=lambda item: int(item[0]))),
        "signalised_crossing_nodes": sorted(c["node"] for c in picked if c["kind"] == SIGNALISE),
    }


def baseline_voice(context: dict, field: dict, at_crossing: dict, baseline_metrics: dict, mask: int) -> dict:
    """Solve one design under BASELINE and return {"converged": True, **J_voice}, or
    {"converged": False, "reason": ...} when it can't be scored.
    """
    try:
        net, links, crossings, result = solve(context, mask, BASELINE)
    except ta.TrafficAssignmentError as e:
        return {"converged": False, "reason": f"baseline: {e}"}
    if not result.converged:
        return {"converged": False, "reason": f"baseline: did not converge: relative gap {result.relative_gap:.2e} "
                                              f"after {result.iterations} iterations"}
    metrics = vo.edge_metrics(net, links, crossings, result, at_crossing)
    try:
        return {"converged": True, **vo.j_voice(field, metrics, baseline_metrics)}
    except ValueError as e:
        return {"converged": False, "reason": f"baseline: J_voice can't be scored: {e}"}


def _start_worker(context: dict, field: dict, at_crossing: dict, baseline_metrics: dict) -> None:
    _worker_context.update(context=context, field=field, at_crossing=at_crossing, baseline_metrics=baseline_metrics)


def _baseline_voice(mask: int) -> dict:
    c = _worker_context
    return baseline_voice(c["context"], c["field"], c["at_crossing"], c["baseline_metrics"], mask)


def voice_runs(context: dict, field: dict, at_crossing: dict, baseline_metrics: dict, masks: list[int]) -> dict:
    """Score each mask under BASELINE, in parallel worker processes; return {mask: run}."""
    if not masks:
        return {}
    if WORKERS <= 1 or len(masks) == 1:
        return {m: baseline_voice(context, field, at_crossing, baseline_metrics, m) for m in masks}
    runs = {}
    step = max(1, len(masks) // 20)
    with ProcessPoolExecutor(min(WORKERS, len(masks)), mp_context=multiprocessing.get_context("spawn"),
                             initializer=_start_worker, initargs=(context, field, at_crossing, baseline_metrics)) as pool:
        for done, (mask, run) in enumerate(zip(masks, pool.map(_baseline_voice, masks)), 1):
            runs[mask] = run
            if done % step == 0 or done == len(masks):
                print(f"  J_voice: {done}/{len(masks)} designs scored under baseline")
    return runs


def attach_voice(designs: list[dict], runs: dict, j_baseline: float) -> list[dict]:
    """Return the designs with J_voice, its per-theme split and its improvement on the feasible
    ones. A feasible design whose baseline run failed becomes infeasible, with the reason.
    """
    out = []
    for d in designs:
        if not d["feasible"]:
            out.append(d)
            continue
        run = runs[d["mask"]]
        if not run["converged"]:
            out.append({**d, "feasible": False, "reason": run["reason"]})
            continue
        out.append({**d, VOICE: run["total"], "j_voice_by_theme": run["by_theme"],
                    "j_voice_improvement": vo.improvement(run["total"], j_baseline)})
    return out


def score_voice(context: dict, designs: list[dict]) -> tuple[list[dict], tuple, dict]:
    """Add J_voice to the designs when the field allows; return (designs, objectives, J_voice summary)."""
    if not FIELD_PATH.exists():
        print(f"warning: no concern field at {FIELD_PATH}, so J_voice is not computed; "
              f"run python -m src.concerns.concern_field to add it")
        return designs, OBJECTIVES, {"status": NOT_COMPUTED, "reason": f"no concern field at {FIELD_PATH}"}
    field = vo.load_field(FIELD_PATH)
    missing = set(context["link_to_edge"]) - set(field["edge_ids"])
    if missing:
        raise ValueError(f"{FIELD_PATH} has no row for {len(missing)} edges of the network (e.g. {sorted(missing)[0]}); "
                         f"rerun python -m src.concerns.concern_field")
    at_crossing = vo.crossing_edges(context["net"], context["link_to_edge"], context["crossings"])
    try:
        net, links, crossings, result = solve(context, 0, BASELINE)
    except ta.TrafficAssignmentError as e:
        raise RuntimeError(f"the do-nothing design can't be solved under baseline conditions ({e}), "
                           f"so J_voice has no baseline") from e
    if not result.converged:
        raise RuntimeError(f"the do-nothing design did not converge under baseline conditions "
                           f"(relative gap {result.relative_gap:.2e}), so J_voice has no baseline")
    baseline_metrics = vo.edge_metrics(net, links, crossings, result, at_crossing)
    baseline = vo.j_voice(field, baseline_metrics, baseline_metrics)
    masks = [d["mask"] for d in designs if d["feasible"] and d["mask"] != 0]
    runs = {0: {"converged": True, **baseline}, **voice_runs(context, field, at_crossing, baseline_metrics, masks)}
    scored = attach_voice(designs, runs, baseline["total"])
    dropped = [{"mask": d["mask"], "reason": s["reason"]} for d, s in zip(designs, scored)
               if d["feasible"] and not s["feasible"]]
    for item in dropped:
        print(f"warning: design {item['mask']} dropped: {item['reason']}")
    status = OPTIMISED if field["basis"] == LOCATED else SHOWN
    if status == SHOWN:
        print(f"warning: the concern field is on {field['basis']!r}, not located comments, "
              f"so J_voice is shown but not optimised")
    summary = {
        "status": status, "field": str(FIELD_PATH), "basis": field["basis"],
        "pairing": vo.PAIRING, "excluded_why": vo.EXCLUDED_WHY, "normalisation": vo.NORMALISATION,
        "baseline": "the do-nothing design under baseline conditions (demand x1.0, no closure)",
        "j_voice_baseline": baseline["total"], "field_mass": vo.field_mass(field),
        "excluded_mass": baseline["excluded_mass"], "unmetered_mass": vo.unmetered_mass(field, baseline_metrics),
        "floored_edges": vo.floored_edges(field, baseline_metrics),
        "field_edges_off_network": len(set(field["edge_ids"]) - set(context["link_to_edge"])),
        "dropped": dropped,
        "notes": [
            "a closure removes its own street's complaints from J_voice; the traffic it diverts only counts "
            "where there are complaints too",
            "edges with complaints but no metric stay at m = 1 and dilute the improvement (unmetered_mass)",
            "the hypervolume is not comparable between runs with 3 and 4 objectives",
        ],
    }
    return scored, (OBJECTIVES + (VOICE,) if status == OPTIMISED else OBJECTIVES), summary


def _scores(d: dict, voice_status: str) -> dict:
    voice = ({VOICE: d[VOICE], "j_voice_improvement": d["j_voice_improvement"], "j_voice_status": voice_status}
             if voice_status != NOT_COMPUTED else {})
    return {**{name: d[name] for name in OBJECTIVES}, **voice}


def main() -> None:
    """Write the best trade-offs, hypervolume and picks with their diffs, and re-solve each pick under
    baseline conditions to write its per-edge flows and times in #24's format.
    """
    search = json.loads(SEARCH_PATH.read_text(encoding="utf-8"))
    if not any(d["feasible"] for d in search["designs"]):
        raise ValueError(f"no feasible design in {SEARCH_PATH}, so there are no trade-offs to pick from")
    context = load_context()
    if json.loads(json.dumps(fingerprint(context))) != search["fingerprint"]:
        raise ValueError(f"the candidates, scenarios, OD matrix, network or settings have changed since "
                         f"{SEARCH_PATH} was written; rerun the design search")

    designs, objectives, voice = score_voice(context, search["designs"])
    voiced = voice["status"] != NOT_COMPUTED
    status = voice["status"]
    trade_offs, volume = trade_offs_and_hypervolume(designs, objectives)
    if not trade_offs:
        raise ValueError("no design is feasible under baseline conditions, so there are no trade-offs to pick from")
    picks = pick_designs(trade_offs, objectives)

    car_edges = set(context["link_to_edge"])
    DESIGNS_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_mask = {}
    for mask in {d["mask"] for d in picks.values()}:
        net, _, _, result = solve(context, mask, BASELINE)
        if not result.converged:
            raise RuntimeError(f"design {mask} did not converge under baseline conditions: "
                               f"relative gap {result.relative_gap:.2e}")
        rows_by_mask[mask] = ue_rows(net, result)
    for role, design in picks.items():
        with (DESIGNS_DIR / f"{role}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=UE_FIELDS)
            writer.writeheader()
            writer.writerows(rows_by_mask[design["mask"]])
    if "voice_best" not in picks:
        (DESIGNS_DIR / "voice_best.csv").unlink(missing_ok=True)   # left by an earlier run with J_voice optimised

    TRADE_OFFS_PATH.write_text(json.dumps({
        "objectives": objectives,
        "reference": [REFERENCE] * len(objectives),
        "hypervolume": volume,
        "hour": search["hour"],
        "flows_conditions": "baseline: demand x1.0, no scenario closure, as in #24",
        "j_voice": voice,
        "trade_offs": [{"mask": d["mask"], "candidates": d["candidates"], **_scores(d, status)}
                       for d in trade_offs],
        "picks": {role: {"mask": d["mask"], "candidates": d["candidates"], **_scores(d, status),
                         **({"j_voice_by_theme": d["j_voice_by_theme"]} if voiced else {}),
                         "diff": design_diff(d, context["candidates"], car_edges), "flows_file": f"designs/{role}.csv"}
                  for role, d in picks.items()},
    }, indent=1, allow_nan=False), encoding="utf-8")
    print(f"{len(trade_offs)} best trade-offs over {len(objectives)} objectives, hypervolume {volume:.3f} "
          f"(normalised, ref {REFERENCE}); J_voice {voice['status']}")
    for role, d in picks.items():
        print(f"  {role}: mask {d['mask']} {d['candidates'] or ['do nothing']}")


if __name__ == "__main__":
    main()
