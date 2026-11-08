"""Bind hand-written legal clauses to edges and count where OSM disagrees (ALG-MVP-4, #20).

Stand-in for orchid's held policy_compiler until #33 swaps it in.
Run from the repo root: ``python -m src.rules.bind_rules``.
"""

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from heapq import heappop, heappush
from pathlib import Path

from engine.orchid import street_network as sn
from src.graph.build_graph import load_graph
from src.graph.street_lookup import build_street_lookup, normalise_name

ROOT = Path(__file__).resolve().parents[2]
CLAUSES_PATH = ROOT / "data" / "rules" / "clauses.json"
SOURCES_DIR = ROOT / "data" / "raw" / "sources"
FLAGS_PATH = ROOT / "data" / "rules" / "restriction_flags.csv"

RESTRICTIONS = ("no_motor_vehicles", "one_way", "pedestrian_only")
KEEP_OPEN = "keep_open"
CLAUSE_TYPES = (*RESTRICTIONS, KEEP_OPEN)
ALL_DAY = "at any time"


class BindError(ValueError):
    """A clause that can't be tied to edges; the message is its abstain reason."""


@dataclass
class Report:
    bound: dict[str, set[str]] = field(default_factory=lambda: {r: set() for r in RESTRICTIONS})
    applied: list[str] = field(default_factory=list)
    wrong_on_osm: list[str] = field(default_factory=list)
    abstained: list[tuple[str, str]] = field(default_factory=list)
    keep_open: dict[str, set[str]] = field(default_factory=dict)
    clause_edges: dict[str, set[str]] = field(default_factory=dict)


def load_clauses(path: Path) -> list[dict]:
    """Read the hand-written clauses.

    Each has clause_id, document_id, page, quote, street, restriction (one of CLAUSE_TYPES), times and
    exemptions (a list, or "unspecified"); one-way clauses also have from_street and to_street.
    Clause ids must be unique, so no clause's edges can replace another's.
    """
    clauses = json.loads(path.read_text(encoding="utf-8"))
    ids = [c["clause_id"] for c in clauses]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: clause ids must be unique")
    return clauses


def load_sources(clauses: list[dict], sources_dir: Path) -> dict[str, str]:
    """Read the cached text of every document the clauses quote, skipping any not cached."""
    return {doc: path.read_text(encoding="utf-8") for doc in {c["document_id"] for c in clauses}
            if (path := sources_dir / f"{doc}.txt").exists()}


def abstain_reason(clause: dict, sources: dict[str, str]) -> str | None:
    """Return why a clause can't be applied honestly, or None if it can.

    keep_open clauses skip the exemptions and all-day checks: a street that must stay open
    for part of the day, or for unlisted users, still can't be closed outright.
    """
    source = sources.get(clause["document_id"])
    if source is None:
        return f"source {clause['document_id']!r} is not cached"
    if clause["quote"] not in source:
        return "quote is not word for word in its source"
    if clause["restriction"] not in CLAUSE_TYPES:
        return f"unknown restriction {clause['restriction']!r}"
    if clause["restriction"] != KEEP_OPEN:
        if clause["exemptions"] == "unspecified":
            return "exemptions are not listed"
        if clause["times"].strip().lower() != ALL_DAY:
            return f"time-limited ({clause['times']}), but flags and OSM tags are all-day"
    if clause["restriction"] == "one_way" and not (clause.get("from_street") and clause.get("to_street")):
        return "one-way clause needs from_street and to_street"
    if "between" in clause and not between_ok(clause["between"]):
        return "between needs two street names"
    return None


def between_ok(between) -> bool:
    """True if a clause's between field names two streets."""
    return isinstance(between, list) and len(between) == 2 and all(isinstance(n, str) and n.strip() for n in between)


def _street_edges(graph: sn.RoadGraph, lookup: dict[str, list[str]], name: str) -> list[sn.RoadEdge]:
    eids = lookup.get(normalise_name(name))
    if not eids:
        raise BindError(f"street {name!r} not found in the graph")
    return [graph.edge(int(eid)) for eid in eids]


def _junctions(nodes: set, other: list[sn.RoadEdge], name: str) -> set:
    shared = nodes & {n for e in other for n in (e.u, e.v)}
    if not shared:
        raise BindError(f"street doesn't meet {name!r}")
    return shared


def _distance_along(edges: list[sn.RoadEdge], starts: set) -> dict:
    adjacent = defaultdict(list)
    for e in edges:
        adjacent[e.u].append((e.v, e.length_m))
        adjacent[e.v].append((e.u, e.length_m))
    dist = dict.fromkeys(starts, 0.0)
    heap = [(0.0, n) for n in starts]
    while heap:
        d, n = heappop(heap)
        if d > dist[n]:
            continue
        for m, length in adjacent[n]:
            if d + length < dist.get(m, math.inf):
                dist[m] = d + length
                heappush(heap, (d + length, m))
    return dist


def _between(edges: list[sn.RoadEdge], graph: sn.RoadGraph, lookup: dict[str, list[str]],
             between: list[str]) -> list[sn.RoadEdge]:
    """Keep the street's edges on its shortest stretch between its junctions with the two named streets."""
    first, second = between
    nodes = {n for e in edges for n in (e.u, e.v)}
    from_first = _distance_along(edges, _junctions(nodes, _street_edges(graph, lookup, first), first))
    ends = _junctions(nodes, _street_edges(graph, lookup, second), second)
    from_second = _distance_along(edges, ends)
    span = min((from_first[n] for n in ends if n in from_first), default=math.inf)
    if span == math.inf:
        raise BindError(f"street has no stretch between {first!r} and {second!r}")
    on_span = {n for n in nodes if n in from_first and n in from_second
               and from_first[n] + from_second[n] <= span + 1e-6}
    return [e for e in edges if e.u in on_span and e.v in on_span]


def bind(clause: dict, graph: sn.RoadGraph, lookup: dict[str, list[str]]) -> list[sn.RoadEdge]:
    """Return the edges the clause restricts; raise BindError if it can't be bound.

    With between, only the stretch of the street between its junctions with those two
    streets. For one_way, only the edges pointing back towards from_street, i.e. against
    the allowed from_street -> to_street direction. A street OSM already has one-way the
    right way has none.
    """
    edges = _street_edges(graph, lookup, clause["street"])
    if clause.get("between"):
        edges = _between(edges, graph, lookup, clause["between"])
    if clause["restriction"] != "one_way":
        return edges
    nodes = {n for e in edges for n in (e.u, e.v)}
    starts = _junctions(nodes, _street_edges(graph, lookup, clause["from_street"]), clause["from_street"])
    _junctions(nodes, _street_edges(graph, lookup, clause["to_street"]), clause["to_street"])
    dist = _distance_along(edges, starts)
    return [e for e in edges if e.u in dist and e.v in dist and dist[e.u] > dist[e.v]]


def osm_agrees(restriction: str, edge: sn.RoadEdge) -> bool:
    """Return True if the edge's OSM tags already show the restriction.

    For one_way the bound edges run against the rule, so OSM letting cars use them is a disagreement.
    """
    if restriction == "pedestrian_only":
        return edge.highway_class == "pedestrian"
    if restriction == "no_motor_vehicles":
        osm = edge.tags["osm"]
        value = osm.get("motor_vehicle") or osm.get("vehicle") or osm.get("access")
        return value == "no" if value else "car" not in edge.modes_permitted
    return "car" not in edge.modes_permitted


def check_clauses(clauses: list[dict], sources: dict[str, str], graph: sn.RoadGraph) -> Report:
    """Apply every clause that can be applied honestly, and note where OSM disagrees."""
    lookup = build_street_lookup(graph)
    report = Report()
    for clause in clauses:
        reason = abstain_reason(clause, sources)
        if reason is None:
            try:
                edges = bind(clause, graph, lookup)
            except BindError as e:
                reason = str(e)
        if reason:
            report.abstained.append((clause["clause_id"], reason))
            continue
        restriction, edge_ids = clause["restriction"], {str(e.eid) for e in edges}
        report.clause_edges[clause["clause_id"]] = edge_ids
        if restriction == KEEP_OPEN:
            report.keep_open[clause["clause_id"]] = edge_ids
            continue
        report.bound[restriction].update(edge_ids)
        report.applied.append(clause["clause_id"])
        if not all(osm_agrees(restriction, e) for e in edges):
            report.wrong_on_osm.append(clause["clause_id"])
    return report


def restriction_flags(bound: dict[str, set[str]], graph: sn.RoadGraph) -> list[dict]:
    """Return one row per directed edge: edge_id plus a 0/1 column per restriction type."""
    return [{"edge_id": str(e.eid), **{r: int(str(e.eid) in bound[r]) for r in RESTRICTIONS}}
            for e in graph.iter_edges()]


def main() -> None:
    """Bind every clause, write the flags CSV, and print the OSM mismatch count and abstained clauses."""
    graph = load_graph()
    clauses = load_clauses(CLAUSES_PATH)
    report = check_clauses(clauses, load_sources(clauses, SOURCES_DIR), graph)

    FLAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FLAGS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("edge_id", *RESTRICTIONS))
        writer.writeheader()
        writer.writerows(restriction_flags(report.bound, graph))

    wrong, checked = len(report.wrong_on_osm), len(report.applied)
    print(f"{wrong} of {checked} legal restrictions missing or wrong on OSM" if wrong
          else f"No legal restrictions missing or wrong on OSM ({checked} checked)")
    for cid in report.wrong_on_osm:
        print(f"  wrong on OSM: {cid}")
    print(f"{len(report.keep_open)} keep-open clauses bound, for the design filter")
    print(f"{len(report.abstained)} abstained:")
    for cid, reason in report.abstained:
        print(f"  {cid}: {reason}")


if __name__ == "__main__":
    main()
