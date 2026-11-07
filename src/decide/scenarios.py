"""Build the demand scenarios for the design search (ALG-S2-2, #27).

Each scenario pairs a demand multiplier (low, normal, peak, event) with one closure
(Johnston Terrace) that is either open or shut. The full set is reduced with orchid's
scenario_reduction, and #28 weights the mean and the CVaR with the returned
probabilities. The multipliers and their weights are labelled assumptions.

Run from the repo root: ``python -m src.decide.scenarios``.
"""

import json
import math
from pathlib import Path

from engine.orchid import stochastic_optimisation as so
from src.graph.build_graph import load_graph
from src.graph.street_lookup import build_street_lookup, find_streets

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_PATH = ROOT / "data" / "demand" / "scenarios.json"

# Demand multiplier and prior weight per level. Assumptions, not measurements.
DEMAND_LEVELS = {
    "low": (0.75, 0.25),
    "normal": (1.00, 0.45),
    "peak": (1.25, 0.20),
    "event": (1.50, 0.10),
}
CLOSURE_STREET = "Johnston Terrace"
CLOSURE_PROBABILITY = 0.10
N_KEPT = 4


def full_scenarios(levels: dict = DEMAND_LEVELS, closure_probability: float = CLOSURE_PROBABILITY) -> list[dict]:
    """Return every demand level with the closure open and shut, with probabilities summing to 1."""
    total = math.fsum(weight for _, weight in levels.values())
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError(f"demand level weights must sum to 1, got {total}")
    if not 0.0 <= closure_probability <= 1.0:
        raise ValueError(f"closure probability must be in [0, 1], got {closure_probability}")
    scenarios = []
    for level, (multiplier, weight) in levels.items():
        for closed, p_closed in ((False, 1.0 - closure_probability), (True, closure_probability)):
            if weight * p_closed > 0:
                scenarios.append({"id": f"{level}{'-closed' if closed else ''}", "level": level,
                                  "multiplier": multiplier, "closed": closed,
                                  "probability": weight * p_closed})
    return scenarios


def scaled_vectors(scenarios: list[dict]) -> list[list[float]]:
    """Scale (multiplier, closed) into [0, 1] each, so the Euclidean metric weighs them equally."""
    multipliers = [s["multiplier"] for s in scenarios]
    low, span = min(multipliers), max(multipliers) - min(multipliers)
    return [[(s["multiplier"] - low) / span if span else 0.0, float(s["closed"])] for s in scenarios]


def reduce_scenarios(scenarios: list[dict], k: int = N_KEPT) -> tuple[list[dict], float]:
    """Keep k representative scenarios with orchid's scenario_reduction.

    Returns the kept scenarios carrying their redistributed probabilities, and the
    Kantorovich distance from the full set (0 when nothing is dropped).
    """
    result = so.scenario_reduction(scaled_vectors(scenarios), [s["probability"] for s in scenarios], k)
    kept = [dict(scenarios[i], probability=p) for i, p in zip(result["kept_indices"], result["probabilities"])]
    return kept, result["kantorovich_distance"]


def closure_edges(graph, street: str = CLOSURE_STREET) -> list[str]:
    """Return the closed street's edge ids as str(eid), both directions, via the #19 lookup."""
    found, unresolved = find_streets(build_street_lookup(graph), [street])
    if unresolved:
        raise ValueError(f"closure street not found in the road graph: {street!r}")
    return sorted(found[street], key=int)


def main() -> None:
    graph = load_graph()
    closed_ids = closure_edges(graph)
    full = full_scenarios()
    kept, distance = reduce_scenarios(full)
    for scenario in full + kept:
        scenario["closed_edge_ids"] = closed_ids if scenario["closed"] else []
    out = {
        "label": "synthetic: demand multipliers and probabilities are assumptions",
        "closure": {"street": CLOSURE_STREET, "edge_ids": closed_ids, "probability": CLOSURE_PROBABILITY},
        "demand_levels": {name: {"multiplier": m, "weight": w} for name, (m, w) in DEMAND_LEVELS.items()},
        "full": full,
        "kept": kept,
        "kantorovich_distance": distance,
    }
    SCENARIOS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCENARIOS_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{len(kept)} of {len(full)} scenarios kept (Kantorovich distance {distance:.4f}); "
          f"closure {CLOSURE_STREET}: {len(closed_ids)} edges -> {SCENARIOS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
