"""Candidate interventions as Network transforms, with the rule filter (ALG-S2-1, #26).

Candidates are written to data/decide/candidates.json by street name by propose_candidates.py. Each design
(a set of candidates) becomes a modified Network for #28's search. Run from the repo root:
``python -m src.decide.interventions``.
"""

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from engine.orchid import street_network as sn
from engine.orchid import traffic_assignment as ta
from src.decide.pedestrian_crossing_delay import find_crossings
from src.graph.build_graph import load_graph
from src.graph.street_lookup import build_street_lookup, find_streets
from src.rules.bind_rules import CLAUSES_PATH, KEEP_OPEN, SOURCES_DIR, check_clauses, load_clauses, load_sources

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES_PATH = ROOT / "data" / "decide" / "candidates.json"
CHECK_PATH = ROOT / "data" / "decide" / "candidate_check.json"

CLOSE, REALLOCATE, SIGNALISE = "close", "reallocate", "signalise_crossing"
KINDS = (CLOSE, REALLOCATE, SIGNALISE)


def load_candidates(path: Path, graph: sn.RoadGraph, crossings: list[dict]) -> tuple[list[dict], list[str]]:
    """Read the candidates and resolve each to the edge ids it changes, both directions.

    close and reallocate name a street; reallocate has a capacity_factor. signalise_crossing
    names an unsignalised crossing node, changes the edges it crosses, and may have a
    capacity_factor for the green time the signals take. A street with no car edges is
    dropped, since changing it can't change the search. Returns the resolved candidates
    and a note for each street or node that didn't resolve or was dropped.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    ids = [c["id"] for c in raw]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: candidate ids must be unique")
    if unknown := {c["kind"] for c in raw} - set(KINDS):
        raise ValueError(f"{path}: unknown kinds {sorted(unknown)}; use {', '.join(KINDS)}")
    if missing := [c["id"] for c in raw if c["kind"] == REALLOCATE and "capacity_factor" not in c]:
        raise ValueError(f"{path}: reallocate candidates need a capacity_factor: {missing}")
    if bad := [c["id"] for c in raw if c.get("capacity_factor", 1.0) <= 0]:
        raise ValueError(f"{path}: capacity_factor must be above 0: {bad}")

    found, unresolved = find_streets(build_street_lookup(graph),
                                     sorted({c["street"] for c in raw if c["kind"] != SIGNALISE}))
    unsignalised = {c["node"]: c["edge_ids"] for c in crossings if not c["signalised"]}
    notes = [f"street not found: {name!r}" for name in unresolved]
    candidates = []
    for c in raw:
        if c["kind"] == SIGNALISE:
            node = int(c["node"])
            edges = unsignalised.get(node)
            if edges is None:
                notes.append(f"node {node} is not an unsignalised crossing ({c['id']})")
                continue
        else:
            node = None
            edges = found.get(c["street"])
            if edges is None:
                continue
            if not any(graph.edge(int(eid)).permits("car") for eid in edges):
                notes.append(f"{c['street']!r} has no car edges, so {c['id']} changes nothing")
                continue
        candidates.append({"id": c["id"], "kind": c["kind"], "edges": frozenset(edges),
                           "capacity_factor": c.get("capacity_factor"), "node": node})
    return candidates, notes


def filter_candidates(candidates: list[dict],
                      keep_open: dict[str, set[str]]) -> tuple[list[dict], list[tuple[str, str]]]:
    """Drop every candidate that closes or cuts capacity on an edge a keep_open clause binds.

    keep_open is {clause_id: its edge ids}. Returns the allowed candidates and
    (candidate id, clause id) for each exclusion.
    """
    clauses_on_edge = defaultdict(set)
    for clause_id, edges in keep_open.items():
        for eid in edges:
            clauses_on_edge[eid].add(clause_id)
    allowed, excluded = [], []
    for c in candidates:
        cuts = c["kind"] == CLOSE or (c["capacity_factor"] is not None and c["capacity_factor"] < 1.0)
        forbidding = sorted(set().union(*(clauses_on_edge.get(eid, ()) for eid in c["edges"]))) if cuts else []
        if forbidding:
            excluded.extend((c["id"], clause_id) for clause_id in forbidding)
        else:
            allowed.append(c)
    return allowed, excluded


def apply_design(net: ta.Network, link_to_edge: list[str], crossings: list[dict], design: list[dict],
                 closed: Iterable[str] = ()) -> tuple[ta.Network, list[str], list[dict]]:
    """Return the Network, its link -> edge id list and the crossings with a design applied.

    The design's closures plus any extra closed edge ids (#27's closed_edge_ids, both
    directions) are dropped together, highest link index first, keeping link_to_edge in step.
    Reallocations and crossing signals then scale capacity, and signalised crossings change
    the crossings list J_ped reads. The inputs are left unchanged.
    """
    drop = set(closed).union(*(c["edges"] for c in design if c["kind"] == CLOSE))
    links = list(link_to_edge)
    for index in sorted((i for i, eid in enumerate(links) if eid in drop), reverse=True):
        net = net.without_link(index)
        del links[index]

    position = {eid: i for i, eid in enumerate(links)}
    for c in design:
        if c["kind"] != CLOSE and c["capacity_factor"] is not None:
            for eid in c["edges"] & position.keys():
                net = net.with_scaled_capacity(position[eid], c["capacity_factor"])

    signalised = {c["node"] for c in design if c["kind"] == SIGNALISE}
    return net, links, [{**x, "signalised": True} if x["node"] in signalised else x for x in crossings]


def checked_candidates(graph: sn.RoadGraph, crossings: list[dict]
                       ) -> tuple[list[dict], list[tuple[str, str]], list[str], list[tuple[str, str]]]:
    """Load the candidates and clauses and apply the rule filter.

    Returns the allowed candidates, (candidate id, clause id) exclusions, notes on what
    didn't resolve, and (clause id, reason) for each keep_open clause that abstained,
    since an abstained clause can't forbid anything.
    """
    candidates, notes = load_candidates(CANDIDATES_PATH, graph, crossings)
    clauses = load_clauses(CLAUSES_PATH)
    report = check_clauses(clauses, load_sources(clauses, SOURCES_DIR), graph)
    allowed, excluded = filter_candidates(candidates, report.keep_open)
    keep_open_ids = {c["clause_id"] for c in clauses if c["restriction"] == KEEP_OPEN}
    abstained = [(cid, reason) for cid, reason in report.abstained if cid in keep_open_ids]
    return allowed, excluded, notes, abstained


def main() -> None:
    """Load and filter the candidates, and write which were allowed, excluded and unresolved,
    plus any keep_open clause that abstained.
    """
    graph = load_graph()
    allowed, excluded, notes, abstained = checked_candidates(graph, find_crossings(graph))

    CHECK_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECK_PATH.write_text(json.dumps({
        "allowed": [c["id"] for c in allowed],
        "excluded": [{"candidate": cid, "clause": clause_id} for cid, clause_id in excluded],
        "unresolved": notes,
        "abstained_keep_open": [{"clause": cid, "reason": reason} for cid, reason in abstained],
    }, indent=1), encoding="utf-8")
    print(f"{len(allowed)} candidates allowed, {len({cid for cid, _ in excluded})} excluded by keep-open clauses")
    for cid, clause_id in excluded:
        print(f"  {cid} excluded by {clause_id}")
    for note in notes:
        print(f"  {note}")
    for cid, reason in abstained:
        print(f"  keep-open clause {cid} abstained ({reason}), so it forbids nothing")


if __name__ == "__main__":
    main()
