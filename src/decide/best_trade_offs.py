"""The best trade-offs, their hypervolume and the three designs to show (ALG-S2-4, #29).

The best trade-offs are the exact Pareto front: designs no other design beats on every
objective. Reads #28's design search, finds them over (mean J_car, mean J_ped, J_risk), and
picks the car-best, pedestrian-best and knee designs. Each pick is re-solved under #24's
baseline conditions so its per-edge file compares line for line with the baseline.
Stand-in for the held policy_compiler's pareto_set. Run from the repo root:
``python -m src.decide.best_trade_offs``.
"""

import csv
import json
from pathlib import Path

from engine.orchid import multi_objective as mo
from src.decide.baseline_user_equilibrium import UE_FIELDS, ue_rows
from src.decide.design_search import RESULTS_PATH as SEARCH_PATH
from src.decide.design_search import chosen, load_context, solve
from src.decide.interventions import CLOSE, SIGNALISE

ROOT = Path(__file__).resolve().parents[2]
TRADE_OFFS_PATH = ROOT / "data" / "decide" / "best_trade_offs.json"
DESIGNS_DIR = ROOT / "data" / "decide" / "designs"

OBJECTIVES = ("mean_j_car", "mean_j_ped", "j_risk")
REFERENCE = 1.1
BASELINE = {"multiplier": 1.0, "closed_edge_ids": []}


def normalise(F: list[list[float]]) -> list[list[float]]:
    """Scale each objective so the best feasible design is 0 and the worst is 1.

    An objective that is the same for every design becomes 0.
    """
    columns = list(zip(*F))
    lows = [min(column) for column in columns]
    spans = [max(column) - low for column, low in zip(columns, lows)]
    return [[(value - low) / span if span else 0.0 for value, low, span in zip(row, lows, spans)] for row in F]


def trade_offs_and_hypervolume(designs: list[dict]) -> tuple[list[dict], float]:
    """Return the feasible designs that are best trade-offs over OBJECTIVES (the Pareto
    front), and the hypervolume of their normalised objectives against REFERENCE on every axis.
    """
    feasible = [d for d in designs if d["feasible"]]
    F = [[d[name] for name in OBJECTIVES] for d in feasible]
    best = sorted(mo.find_pareto_front(F))
    normalised = normalise(F)
    volume = mo.hypervolume([normalised[i] for i in best], [REFERENCE] * len(OBJECTIVES))
    return [feasible[i] for i in best], volume


def pick_designs(trade_offs: list[dict]) -> dict[str, dict]:
    """Return the car-best (lowest mean J_car), pedestrian-best (lowest mean J_ped) and knee
    designs. The knee comes from the J_car vs J_risk projection, cut to its own best trade-offs first.
    Ties go to the lower mask.
    """
    projection = [[d["mean_j_car"], d["j_risk"]] for d in trade_offs]
    cut = sorted(mo.find_pareto_front(projection))
    return {
        "car_best": min(trade_offs, key=lambda d: (d["mean_j_car"], d["mask"])),
        "pedestrian_best": min(trade_offs, key=lambda d: (d["mean_j_ped"], d["mask"])),
        "knee": trade_offs[cut[mo.knee_point([projection[i] for i in cut])]],
    }


def design_diff(design: dict, candidates: list[dict]) -> dict:
    """Return what a design changes, as string edge ids: closed edges, edges whose capacity
    changes (with the combined factor), and signalised crossing nodes.

    Capacity changes on closed edges are left out, as apply_design leaves them out.
    """
    picked = chosen(candidates, design["mask"])
    closed = set().union(*(c["edges"] for c in picked if c["kind"] == CLOSE))
    factors = {}
    for c in picked:
        if c["kind"] != CLOSE and c["capacity_factor"] is not None:
            for eid in c["edges"] - closed:
                factors[eid] = factors.get(eid, 1.0) * c["capacity_factor"]
    return {
        "closed_edge_ids": sorted(closed, key=int),
        "capacity_factors": dict(sorted(factors.items(), key=lambda item: int(item[0]))),
        "signalised_crossing_nodes": sorted(c["node"] for c in picked if c["kind"] == SIGNALISE),
    }


def main() -> None:
    """Write the best trade-offs, hypervolume and picks with their diffs, and re-solve each pick under
    baseline conditions to write its per-edge flows and times in #24's format.
    """
    search = json.loads(SEARCH_PATH.read_text(encoding="utf-8"))
    trade_offs, volume = trade_offs_and_hypervolume(search["designs"])
    if not trade_offs:
        raise ValueError(f"no feasible design in {SEARCH_PATH}, so there are no trade-offs to pick from")
    picks = pick_designs(trade_offs)

    context = load_context()
    if [c["id"] for c in context["candidates"]] != search["candidates"]:
        raise ValueError(f"the candidates have changed since {SEARCH_PATH} was written; rerun the design search")
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

    TRADE_OFFS_PATH.write_text(json.dumps({
        "objectives": OBJECTIVES,
        "reference": [REFERENCE] * len(OBJECTIVES),
        "hypervolume": volume,
        "hour": search["hour"],
        "flows_conditions": "baseline: demand x1.0, no scenario closure, as in #24",
        "trade_offs": [{"mask": d["mask"], "candidates": d["candidates"], **{name: d[name] for name in OBJECTIVES}}
                       for d in trade_offs],
        "picks": {role: {"mask": d["mask"], "candidates": d["candidates"], **{name: d[name] for name in OBJECTIVES},
                         "diff": design_diff(d, context["candidates"]), "flows_file": f"designs/{role}.csv"}
                  for role, d in picks.items()},
    }, indent=1), encoding="utf-8")
    print(f"{len(trade_offs)} best trade-offs, hypervolume {volume:.3f} (normalised, ref {REFERENCE})")
    for role, d in picks.items():
        print(f"  {role}: mask {d['mask']} {d['candidates'] or ['do nothing']}")


if __name__ == "__main__":
    main()
