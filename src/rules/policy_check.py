"""Check each picked design against each clause (ALG-S2-6, #31).

Every #29 pick gets satisfied / in_tension / abstained per #20 clause, judged on the design
itself as orchid's policy_check does: a ban is in tension while the design still has a car
link on a banned edge, and a keep_open clause is in tension if the design closes or cuts
one of its edges. Reported as "consistent with N verified clauses; M abstained for review",
never "legally compliant". Stand-in for the held policy_compiler.policy_check until #33.
Run from the repo root: ``python -m src.rules.policy_check``.
"""

import json
from collections import Counter
from pathlib import Path

from src.decide.best_trade_offs import TRADE_OFFS_PATH
from src.decide.network import link_maps, to_network
from src.graph.build_graph import load_graph
from src.rules.bind_rules import CLAUSES_PATH, KEEP_OPEN, SOURCES_DIR, check_clauses, load_clauses, load_sources

ROOT = Path(__file__).resolve().parents[2]
POLICY_CHECK_PATH = ROOT / "data" / "rules" / "policy_check.json"

BAN_TENSION = "the modelled car network still lets cars use these edges"
KEEP_OPEN_TENSION = "the design closes or cuts capacity on edges this clause keeps open"


def check_design(diff: dict, car_edges: set[str], clauses: list[dict], clause_edges: dict[str, set[str]],
                 abstained: dict[str, str]) -> list[dict]:
    """Return one row per clause for a design: clause_id, status, the edges in tension,
    quote, page, document_id, and a detail line saying why.

    diff is the design's #29 diff, car_edges the base network's car links, clause_edges
    each bound clause's edges, and abstained {clause_id: reason}.
    """
    closed = set(diff["closed_edge_ids"])
    open_car_edges = car_edges - closed
    closed_or_cut = closed | {eid for eid, factor in diff["capacity_factors"].items() if factor < 1.0}
    rows = []
    for c in clauses:
        row = {key: c[key] for key in ("clause_id", "restriction", "street", "quote", "page", "document_id")}
        if c["clause_id"] in abstained:
            rows.append({**row, "status": "abstained", "edges": [], "detail": abstained[c["clause_id"]]})
            continue
        keep_open = c["restriction"] == KEEP_OPEN
        hit = clause_edges[c["clause_id"]] & (closed_or_cut if keep_open else open_car_edges)
        rows.append({**row, "status": "in_tension" if hit else "satisfied", "edges": sorted(hit, key=int),
                     "detail": (KEEP_OPEN_TENSION if keep_open else BAN_TENSION) if hit else ""})
    return rows


def summary(rows: list[dict]) -> str:
    """Return "Consistent with N verified clauses; M abstained for review", adding how many
    are in tension when any are.
    """
    counts = Counter(row["status"] for row in rows)
    satisfied = counts["satisfied"]
    tension = f"; {counts['in_tension']} in tension" if counts["in_tension"] else ""
    return (f"Consistent with {satisfied} verified clause{'' if satisfied == 1 else 's'}{tension}; "
            f"{counts['abstained']} abstained for review.")


def main() -> None:
    """Check #29's three picks against #20's clauses and write the rows and summaries."""
    picks = json.loads(TRADE_OFFS_PATH.read_text(encoding="utf-8"))["picks"]
    graph = load_graph()
    car_edges = set(link_maps(to_network(graph)[0])[1])
    clauses = load_clauses(CLAUSES_PATH)
    report = check_clauses(clauses, load_sources(clauses, SOURCES_DIR), graph)
    abstained = dict(report.abstained)

    diffs = {pick["mask"]: pick["diff"] for pick in picks.values()}
    rows_by_mask = {mask: check_design(diff, car_edges, clauses, report.clause_edges, abstained)
                    for mask, diff in diffs.items()}
    designs = {role: {"mask": pick["mask"], "candidates": pick["candidates"],
                      "summary": summary(rows_by_mask[pick["mask"]]), "clauses": rows_by_mask[pick["mask"]]}
               for role, pick in picks.items()}
    POLICY_CHECK_PATH.parent.mkdir(parents=True, exist_ok=True)
    POLICY_CHECK_PATH.write_text(json.dumps({
        "note": "Judged on each design's modelled network against the verified clauses; not a legal determination.",
        "designs": designs,
    }, indent=1), encoding="utf-8")
    for role, design in designs.items():
        print(f"{role}: {design['summary']}")


if __name__ == "__main__":
    main()
