"""Compile the hand-written legal clauses with ORCHID's policy_compiler (ALG-S3-2, #33).

Replaces the #19 street lookup with a policy_compiler Gazetteer, the #20 binding and the
#26 filter with compile_policy's CompiledFixings, and #31 with policy_check.
The #20 glue (src/rules/bind_rules.py) stays as the reference: compare_with_glue lists
every difference. Run from the repo root: ``python -m src.rules.compile_rules``.

Clause strength is set by hand, never read from the quote. compile_policy always runs its
English-wording reader (classify_wording) and abstains a hard clause whose wording reads
soft. That reader is where policy_compiler's bug hunts kept finding bugs, so every clause
carries the fixed wording HARD_WORDING through PolicyClause.wording, the documented text the
reader reads instead of the quote (only the quote is checked against the source). The gate
then always passes, and what the gate would have said about the real quote is listed
separately by wording_gate_flags for a human to see.

Differences from the glue, by design: a time window the grammar can read compiles as a
conditional rule (out of the all-day flags and the OSM count) where the glue abstains, and
duplicate or padded clause ids raise ValueError, because compile_policy keys everything by id.
As in the glue, only the exact all-day phrase "at any time" compiles unconditionally; a blank
or other time field that the grammar reads as all day ("", "no waiting", "at all times") is
queued as unrepresentable. A design for check_design lists only the edges it closes on top of
the status quo; the edges OSM already closes to cars are added (osm_closed_edges).
"""

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from engine.orchid import policy_compiler as pc
from engine.orchid import street_network as sn
from src.graph.build_graph import load_graph
from src.graph.street_lookup import normalise_name
from src.rules import bind_rules as br

CLAUSES_PATH = br.CLAUSES_PATH
SOURCES_DIR = br.SOURCES_DIR
FLAGS_PATH = br.FLAGS_PATH

HARD_WORDING = "shall"   # classify_wording reads this as an obligation, which the gate never abstains
SOFT_MODALITIES = ("recommendation", "discouragement")   # the gate's test in compile_policy


class Refused(ValueError):
    """A clause the compiler can't act on; reason is one of pc.ABSTENTION_REASONS."""

    def __init__(self, reason: str, detail: str, candidates: tuple = ()):
        super().__init__(detail)
        self.reason = reason
        self.candidates = tuple(candidates)


@dataclass
class Compiled:
    result: pc.CompilationResult
    order: list[str]                                         # clause ids in input order
    restriction: dict[str, str] = field(default_factory=dict)
    nothing_to_close: list[str] = field(default_factory=list)  # one-way streets OSM already has right (any times)


def street_key(name: str) -> str:
    """Return policy_compiler's canonical form of a street name ("Bank St." -> "bank street"), or "" if it has none."""
    try:
        return pc.split_street_name(name).canonical() if name.strip() else ""
    except pc.PolicyCompilerError:   # a name with no words
        return ""


def build_gazetteer(graph: sn.RoadGraph, names=()) -> pc.Gazetteer:
    """Return a Gazetteer with one street entry per OSM street name, keyed by street_key.

    Names that policy_compiler reads as the same street ("Bank St", "Bank Street") share one
    entry, or the resolver would call them ambiguous. Edge and node ids are str, edges in both
    directions; coordinates are the graph's planar metres. Every OSM name, and every name in
    names (e.g. the clauses' streets) whose street_key is an entry, is pinned to that entry as
    an override: an exact name binds even when another street's name is close ("Queen Street"
    and "Queens Street" are otherwise ambiguous). Other names go through the fuzzy resolver.
    """
    groups = defaultdict(lambda: {"names": [], "edges": set(), "nodes": set()})
    for e in graph.iter_edges():
        if not (key := street_key(e.street_name)):
            continue
        g = groups[key]
        if e.street_name not in g["names"]:
            g["names"].append(e.street_name)
        g["edges"].add(str(e.eid))
        g["nodes"].update((str(e.u), str(e.v)))
    xy = {str(n.node_id): n.xy for n in graph.iter_nodes()}
    entries = [pc.GazetteerEntry(key, g["names"][0], edge_ids=tuple(g["edges"]), node_ids=tuple(g["nodes"]),
                                 coords=tuple(xy[n] for n in sorted(g["nodes"])), aliases=tuple(g["names"][1:]))
               for key, g in sorted(groups.items())]
    pinned = [n for g in groups.values() for n in g["names"]] + list(names)
    return pc.Gazetteer(entries, edge_endpoints={str(e.eid): (str(e.u), str(e.v)) for e in graph.iter_edges()},
                        node_coords=xy, overrides={n: street_key(n) for n in pinned if street_key(n) in groups})


def _with_entries(gazetteer: pc.Gazetteer, extra: list[pc.GazetteerEntry]) -> pc.Gazetteer:
    if not extra:
        return gazetteer
    return pc.Gazetteer(gazetteer.entries() + extra, edge_endpoints=gazetteer.edge_endpoints,
                        node_coords=gazetteer.node_coords, overrides=gazetteer.overrides)


def wrong_way_edges(clause: dict, gazetteer: pc.Gazetteer, graph: sn.RoadGraph) -> list[str]:
    """Return the str ids of a one-way street's edges that run against the allowed direction.

    The three streets are bound by the gazetteer; which edges point back towards from_street
    is still the #20 glue's rule (bind_rules.bind). Raises Refused if a street doesn't bind
    or the streets don't meet.
    """
    found = {}
    for name in (clause["street"], clause["from_street"], clause["to_street"]):
        b = gazetteer.resolve(name)
        if b.status != "bound":
            raise Refused(f"binding_{b.status}", f"street {name!r}: {b.reason}", b.candidates)
        found[normalise_name(name)] = list(gazetteer.get(b.target_id).edge_ids)
    try:
        return [str(e.eid) for e in br.bind(clause, graph, found)]
    except br.BindError as e:
        raise Refused("binding_unresolved", str(e)) from e


def _target(clause: dict, gazetteer: pc.Gazetteer, graph: sn.RoadGraph):
    """Return (target_kind, target_name, predicate, one-way entry or None); raise Refused if unrepresentable.

    Checks run in the glue's order after the span check, which compile_policy does itself.
    """
    restriction = clause["restriction"]
    if restriction not in br.RESTRICTIONS:
        raise Refused("unrepresentable", f"unknown restriction {restriction!r}")
    if clause["exemptions"] == "unspecified":
        raise Refused("unrepresentable", "exemptions are not listed")
    times = clause["times"]
    try:
        predicate = pc.parse_predicate(times)
    except pc.PolicyCompilerError as e:
        raise Refused("unrepresentable", f"time window {times!r}: {e}") from e
    if predicate.is_trivial:   # also "" and "no waiting": the grammar reads them as all day
        if times.strip().lower() != br.ALL_DAY:
            raise Refused("unrepresentable", f"time window {times!r} is not {br.ALL_DAY!r} or a time window")
        predicate = None
    if restriction != "one_way":
        return "street", clause["street"], predicate, None
    if not (clause.get("from_street") and clause.get("to_street")):
        raise Refused("unrepresentable", "one-way clause needs from_street and to_street")
    eids = wrong_way_edges(clause, gazetteer, graph)
    if not eids:
        return "street", clause["street"], predicate, None
    entry_id = f"oneway:{clause['clause_id']}"
    name = f"oneway {clause['clause_id'].encode().hex()}"   # unique after name normalisation, unlike the id
    return "area", name, predicate, pc.GazetteerEntry(entry_id, name, kind="area", edge_ids=tuple(eids))


def to_policy_clause(clause: dict, gazetteer: pc.Gazetteer, graph: sn.RoadGraph, strength: str = "hard"):
    """Convert a glue clause into (PolicyClause or None, Refused or None, one-way GazetteerEntry or None).

    Every restriction closes edges to cars: REMOVE_EDGE REQUIRE, everything else ABSTAIN.
    A one-way clause targets an area entry holding only its wrong-way edges. A refused clause
    keeps all-ABSTAIN rulings, so compile_policy still verifies its quote and policy_check
    still gives it a row; a one-way street with nothing to close does the same, unrefused.
    strength is set by hand and wording is the constant HARD_WORDING, so the quote's English
    never changes the result. Returns no PolicyClause only if the schema refuses the clause.
    """
    rulings = pc.make_rulings(REMOVE_EDGE="REQUIRE")
    try:
        kind, name, predicate, entry = _target(clause, gazetteer, graph)
        refusal = None
        if entry is None and clause["restriction"] == "one_way":
            rulings = pc.make_rulings()
    except Refused as e:
        kind, name, predicate, entry, refusal, rulings = "street", clause["street"], None, None, e, pc.make_rulings()
    try:
        policy_clause = pc.PolicyClause(clause["clause_id"], clause["document_id"], clause["page"], clause["quote"],
                                        kind, name, rulings, predicate=predicate, strength=strength,
                                        wording=HARD_WORDING)
    except pc.PolicyCompilerError as e:
        return None, Refused("unrepresentable", str(e)), None
    return policy_clause, refusal, entry


def compile_rules(clauses: list[dict], sources: dict[str, str], graph: sn.RoadGraph) -> Compiled:
    """Compile every clause with compile_policy (no instruments), with refused clauses added to its queue.

    A refusal is queued only if compile_policy verified the clause's quote, so like the glue
    each clause shows its first reason. Strength comes from the clause's "strength" key
    (hand-written), else "hard". Raises ValueError for duplicate, empty or padded clause ids.
    """
    ids = Counter(c["clause_id"] for c in clauses)
    bad = sorted(repr(i) for i, n in ids.items() if n > 1 or not isinstance(i, str) or not i or i != i.strip())
    if bad:
        raise ValueError(f"clause ids must be unique, non-empty and unpadded: {', '.join(bad)}")
    names = [c.get(k) for c in clauses for k in ("street", "from_street", "to_street")]
    gazetteer = build_gazetteer(graph, [n for n in names if isinstance(n, str)])
    policy_clauses, refusals, extra, restriction, nothing_to_close = [], {}, [], {}, []
    for clause in clauses:
        policy_clause, refusal, entry = to_policy_clause(clause, gazetteer, graph, clause.get("strength", "hard"))
        cid = clause["clause_id"]
        if refusal is not None:
            refusals[cid] = (clause, refusal)
        if entry is not None:
            extra.append(entry)
        if policy_clause is not None:
            policy_clauses.append(policy_clause)
            restriction[cid] = clause["restriction"]
            if refusal is None and clause["restriction"] == "one_way" and entry is None:
                nothing_to_close.append(cid)
    res = pc.compile_policy(policy_clauses, _with_entries(gazetteer, extra), sources=sources)
    for cid, (clause, refusal) in refusals.items():
        if cid not in res.clauses or res.skipped.get(cid) == "no_operation":
            res.queue.add(pc.AbstentionItem(cid, refusal.reason, 1.0, refusal.candidates, clause.get("quote", ""),
                                            clause.get("page", 0), clause.get("document_id", ""), str(refusal)))
    return Compiled(res, [c["clause_id"] for c in clauses], restriction, nothing_to_close)


def clause_edges(compiled: Compiled) -> dict[str, set[str]]:
    """Map each clause id to the edges it closes all day, from the fixings' provenance."""
    edges = defaultdict(set)
    for key, cids in compiled.result.fixings.to_dict()["provenance"].items():
        if key.startswith("fixedZero:"):
            for cid in cids:
                edges[cid].add(key.removeprefix("fixedZero:"))
    return dict(edges)


def abstained(compiled: Compiled) -> list[tuple[str, str]]:
    """Return (clause_id, "reason: detail; ...") for every queued clause, in input order."""
    why = defaultdict(list)
    for item in compiled.result.queue.items():
        why[item.clause_id].append(f"{item.reason}: {item.detail}" if item.detail else item.reason)
    return [(cid, "; ".join(why[cid])) for cid in compiled.order if cid in why]


def _closes_nothing(compiled: Compiled, cid: str, timed: bool) -> bool:
    return cid in compiled.nothing_to_close and (compiled.result.clauses[cid].predicate is not None) == timed


def conditional(compiled: Compiled) -> list[str]:
    """Return the clauses compiled only for a time window (or timed one-way streets with nothing to close)."""
    res, queued = compiled.result, set(compiled.result.queue.clause_ids())
    return [cid for cid in compiled.order if cid not in queued and (
        (cid in res.per_clause and res.per_clause[cid].conditional) or _closes_nothing(compiled, cid, timed=True))]


def applied(compiled: Compiled) -> list[str]:
    """Return the clauses closing edges all day (or all-day one-way streets with nothing to close), in input order."""
    edges, queued = clause_edges(compiled), set(compiled.result.queue.clause_ids())
    return [cid for cid in compiled.order
            if cid not in queued and (cid in edges or _closes_nothing(compiled, cid, timed=False))]


def wrong_on_osm(compiled: Compiled, graph: sn.RoadGraph) -> list[str]:
    """Return the applied clauses whose closed edges OSM doesn't show as restricted (the glue's osm_agrees)."""
    edges = clause_edges(compiled)
    return [cid for cid in applied(compiled)
            if not all(br.osm_agrees(compiled.restriction[cid], graph.edge(int(e))) for e in edges.get(cid, ()))]


def mismatch_line(compiled: Compiled, graph: sn.RoadGraph) -> str:
    """Return the "N of M legal restrictions missing or wrong on OSM" line, as the glue prints it."""
    wrong, checked = len(wrong_on_osm(compiled, graph)), len(applied(compiled))
    return (f"{wrong} of {checked} legal restrictions missing or wrong on OSM" if wrong
            else f"No legal restrictions missing or wrong on OSM ({checked} checked)")


def bound(compiled: Compiled) -> dict[str, set[str]]:
    """Return {restriction: {str(eid)}} from the compiled all-day fixings, as the glue's Report.bound."""
    out = {r: set() for r in br.RESTRICTIONS}
    queued = set(compiled.result.queue.clause_ids())
    for cid, eids in clause_edges(compiled).items():
        if cid not in queued:
            out[compiled.restriction[cid]].update(eids)
    return out


def restriction_flags(compiled: Compiled, graph: sn.RoadGraph) -> list[dict]:
    """Return one row per directed edge in the glue's CSV format, from the compiled fixings.

    one_way=1 means the edge runs against the legal direction, as in #20. Time-limited
    (conditional) fixings are left out: the flags are all-day.
    """
    return br.restriction_flags(bound(compiled), graph)


def write_flags(rows: list[dict], path=None) -> None:
    """Write the flag rows to data/rules/restriction_flags.csv in the glue's format."""
    path = path or FLAGS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("edge_id", *br.RESTRICTIONS))
        writer.writeheader()
        writer.writerows(rows)


def design_from_closed(closed_eids, graph: sn.RoadGraph) -> pc.Design:
    """Return a policy_compiler Design: z=0 for each edge closed to cars, z=1 for every other edge.

    Every edge is listed, because policy_check reads a missing edge as closed.
    """
    closed = {str(e) for e in closed_eids}
    eids = {str(e.eid) for e in graph.iter_edges()}
    if closed - eids:
        raise ValueError(f"unknown edge ids {sorted(closed - eids)}")
    return pc.Design(z={e: 0 if e in closed else 1 for e in eids}, y={}, w={}, k={})


def osm_closed_edges(graph: sn.RoadGraph) -> set[str]:
    """Return the str ids of the edges OSM already closes to cars (the glue's osm_agrees)."""
    return {str(e.eid) for e in graph.iter_edges()
            if br.osm_agrees("no_motor_vehicles", e) or br.osm_agrees("pedestrian_only", e)}


def check_design(closed_eids, compiled: Compiled, graph: sn.RoadGraph, scenario=None) -> tuple[list[dict], float]:
    """Check a design (the edges it closes to cars) clause by clause with pc.policy_check.

    closed_eids are the design's closures on top of the status quo: the edges OSM already
    closes to cars are added, so the status quo (closed_eids empty) satisfies every ban OSM
    already shows. A design can't reopen them. Returns rows of clause_id, status, quote,
    page, violated and detail, and the compliance ratio. Without a scenario, time-limited
    clauses are checked as if active.
    """
    closed = {str(e) for e in closed_eids} | osm_closed_edges(graph)
    report = pc.policy_check(design_from_closed(closed, graph), compiled.result, scenario)
    rows = [{"clause_id": r.clause_id, "status": r.status, "quote": r.quote, "page": r.page,
             "violated": r.violated, "detail": r.detail} for r in report.rows]
    return rows, report.compliance_ratio


def wording_gate_flags(compiled: Compiled) -> list[tuple[str, str, str | None]]:
    """Return (clause_id, modality, cue) for hard clauses whose real quote the wording gate would abstain.

    Only clauses that reached the gate count: compile_policy runs it after the span check,
    and not on soft clauses or ones that close nothing. Those are exactly the clauses with a
    binding, the stage after the gate. For human review only; they are not queued, because a
    queued clause that still closes edges would show as abstained instead of satisfied in
    policy_check.
    """
    cue_modality = {q: m for q, m, _p in pc.WORDING_CUES}
    res, out = compiled.result, []
    for cid in sorted(res.bindings):
        w = pc.classify_wording(res.clauses[cid].quote)
        if w.modality in SOFT_MODALITIES or (w.modality == "neutral" and cue_modality.get(w.cue) in SOFT_MODALITIES):
            out.append((cid, w.modality, w.cue))
    return out


def outcome(compiled: Compiled, cid: str) -> str:
    """Return what the compiler did with one clause, in words."""
    res = compiled.result
    why = dict(abstained(compiled))
    if cid in why:
        return f"abstained ({why[cid]})"
    if cid in applied(compiled):
        return "applied" + (" (nothing to close)" if cid in compiled.nothing_to_close else "")
    if cid in conditional(compiled):
        return ("compiled for its time window only (not in the all-day flags or the OSM check)"
                + (" (nothing to close)" if cid in compiled.nothing_to_close else ""))
    return f"skipped ({res.skipped.get(cid, 'nothing compiled')})"


def compare_with_glue(clauses: list[dict], sources: dict[str, str], graph: sn.RoadGraph,
                      compiled: Compiled | None = None) -> dict:
    """Return every difference from the #20 glue; all lists empty means the two agree.

    clauses: {clause_id: {"glue": ..., "compiler": ...}} for clauses applied by one and not
    the other, or abstained by the glue but neither applied nor abstained by the compiler
    (a time window it compiled, a soft clause); wrong_on_osm: clause ids in only one list;
    flags: edge ids whose rows differ. Expected differences: a time window the grammar reads
    (glue abstains, compiler compiles it conditionally) and a street name the glue's exact
    lookup misses but the gazetteer's fuzzy resolver binds (a typo).
    """
    compiled = compiled or compile_rules(clauses, sources, graph)
    report = br.check_clauses(clauses, sources, graph)
    glue_why, ours, our_why = dict(report.abstained), applied(compiled), dict(abstained(compiled))
    differ = {}
    for cid in compiled.order:
        if (cid in report.applied, cid in glue_why) != (cid in ours, cid in our_why):
            glue = "applied" if cid in report.applied else f"abstained ({glue_why.get(cid, '?')})"
            differ[cid] = {"glue": glue, "compiler": outcome(compiled, cid)}
    glue_wrong, our_wrong = report.wrong_on_osm, wrong_on_osm(compiled, graph)
    glue_rows, our_rows = br.restriction_flags(report.bound, graph), restriction_flags(compiled, graph)
    return {"clauses": differ,
            "wrong_on_osm": {"glue_only": [c for c in glue_wrong if c not in our_wrong],
                             "compiler_only": [c for c in our_wrong if c not in glue_wrong]},
            "flags": [g["edge_id"] for g, o in zip(glue_rows, our_rows) if g != o]}


def main() -> None:
    """Compile every clause, write the flags CSV, and print the OSM mismatch count, abstentions and glue differences."""
    graph = load_graph()
    clauses = br.load_clauses(CLAUSES_PATH)
    sources = {doc: path.read_text(encoding="utf-8") for doc in {c["document_id"] for c in clauses}
               if (path := SOURCES_DIR / f"{doc}.txt").exists()}
    compiled = compile_rules(clauses, sources, graph)
    write_flags(restriction_flags(compiled, graph))

    print(mismatch_line(compiled, graph))
    for cid in wrong_on_osm(compiled, graph):
        print(f"  wrong on OSM: {cid}")
    queued = abstained(compiled)
    print(f"{len(queued)} abstained:")
    for cid, reason in queued:
        print(f"  {cid}: {reason}")
    timed = conditional(compiled)
    print(f"{len(timed)} compiled for a time window only (not in the all-day flags):")
    for cid in timed:
        print(f"  {cid}")
    gate = wording_gate_flags(compiled)
    print(f"{len(gate)} hard clauses whose quote the wording gate would have abstained (strength is set by hand;"
          " not applied, for review):")
    for cid, modality, cue in gate:
        print(f"  {cid}: {modality} wording {cue!r}")
    diff = compare_with_glue(clauses, sources, graph, compiled)
    if not (diff["clauses"] or diff["wrong_on_osm"]["glue_only"] or diff["wrong_on_osm"]["compiler_only"]
            or diff["flags"]):
        print("Same applied clauses, OSM mismatches and flags as the #20 glue")
        return
    print("Differences from the #20 glue:")
    for cid, d in diff["clauses"].items():
        print(f"  {cid}: glue {d['glue']}; compiler {d['compiler']}")
    for side, cids in diff["wrong_on_osm"].items():
        for cid in cids:
            print(f"  wrong on OSM ({side.replace('_', ' ')}): {cid}")
    if diff["flags"]:
        print(f"  flags differ on {len(diff['flags'])} edges: {', '.join(diff['flags'][:10])}")


if __name__ == "__main__":
    main()
