"""How much the mean-optimal and CVaR-optimal designs lose on bad days (ALG-S2-5, #30).

Reads one #28 design search and compares the two designs' tail J_car (CVaR at alpha 0.9)
with their own mean J_car, using the same scenario weights. Run from the repo root:
``python -m src.decide.tail_risk_comparison``.
"""

import json
import math
from pathlib import Path

from src.decide.design_search import RESULTS_PATH as SEARCH_PATH
from src.decide.design_search import TAIL_TOLERANCE

ROOT = Path(__file__).resolve().parents[2]
TAIL_RISK_PATH = ROOT / "data" / "decide" / "tail_risk_comparison.json"


def optimal_designs(designs: list[dict]) -> tuple[dict, dict]:
    """Return the feasible design with the lowest mean J_car and the one with the lowest
    J_risk; ties go to the lower mask.
    """
    feasible = [d for d in designs if d["feasible"]]
    if not feasible:
        raise ValueError("no feasible design to compare")
    return (min(feasible, key=lambda d: (d["mean_j_car"], d["mask"])),
            min(feasible, key=lambda d: (d["j_risk"], d["mask"])))


def tail_loss(design: dict, scenarios: list[dict], alpha: float) -> tuple[float, bool]:
    """Return how much worse the design's tail J_car is than its mean J_car, in percent,
    and whether the tail is just its worst scenario.

    It is when the scenarios sharing the worst J_car carry at least 1 - alpha of the
    probability between them, since CVaR then equals that J_car.
    """
    j_car = [design["per_scenario"][s["id"]]["j_car"] for s in scenarios]
    worst = max(j_car)
    worst_probability = math.fsum(s["probability"] for s, j in zip(scenarios, j_car) if j == worst)
    return (design["j_risk"] / design["mean_j_car"] - 1.0) * 100.0, worst_probability >= 1.0 - alpha - TAIL_TOLERANCE


def summary_sentence(mean_loss: float, cvar_loss: float, tail: str, same_design: bool) -> str:
    """Return the slide's sentence: "mean-optimal loses X%, the CVaR design loses Y% ...".

    tail says where the loss happens, e.g. "in the worst scenario".
    """
    if same_design:
        return (f"{tail.capitalize()}, the mean-optimal design, which is also the CVaR-optimal design, "
                f"loses {mean_loss:.1f}% against its average day.")
    return (f"{tail.capitalize()}, the mean-optimal design loses {mean_loss:.1f}% against its average day; "
            f"the CVaR design loses {cvar_loss:.1f}%.")


def main() -> None:
    """Compare the two designs from #28's results and write X, Y and the scenario count for Signal."""
    search = json.loads(SEARCH_PATH.read_text(encoding="utf-8"))
    alpha, scenarios = search["alpha"], search["scenarios"]
    mean_optimal, cvar_optimal = optimal_designs(search["designs"])
    same_design = mean_optimal["mask"] == cvar_optimal["mask"]
    x, x_worst_only = tail_loss(mean_optimal, scenarios, alpha)
    y, y_worst_only = (x, x_worst_only) if same_design else tail_loss(cvar_optimal, scenarios, alpha)
    tail = "in the worst scenario" if x_worst_only and y_worst_only else f"in the worst {1.0 - alpha:.0%} of scenarios"
    sentence = summary_sentence(x, y, tail, same_design)

    TAIL_RISK_PATH.parent.mkdir(parents=True, exist_ok=True)
    TAIL_RISK_PATH.write_text(json.dumps({
        "x_pct": x,
        "y_pct": y,
        "scenario_count": len(scenarios),
        "sentence": sentence,
        "tail": tail,
        "alpha": alpha,
        "hour": search["hour"],
        "same_design": same_design,
        "mean_optimal": {key: mean_optimal[key] for key in ("mask", "candidates", "mean_j_car", "j_risk")},
        "cvar_optimal": {key: cvar_optimal[key] for key in ("mask", "candidates", "mean_j_car", "j_risk")},
    }, indent=1), encoding="utf-8")
    print(sentence)


if __name__ == "__main__":
    main()
