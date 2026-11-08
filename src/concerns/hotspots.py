"""Concern hotspots that survive a false-discovery correction (ALG-S3-4, #35).

Each street segment -- a pair of twin edges, or a one-way edge -- is one unit, placed
halfway along it. Every located concern record is snapped to its nearest segment. For
each segment, its neighbourhood is itself plus every segment whose halfway point is within
BAND_M, and the test asks: are there more comments here than if the same comments had
landed uniformly along the street network?

Under that null the number of comments in a neighbourhood is Binomial(N, share), with N
the comments placed and share the neighbourhood's fraction of the total network length, so
the p-value is the exact binomial upper tail (scipy.stats.binom.sf). The null treats every
comment as independent: a record with count c is c comments at one spot, which is what
lets a junction with many comments stand out. Benjamini-Hochberg
controls the false-discovery rate over the testable segments. Hot segments within the band
of each other form one hotspot.

Testing counts against network length, rather than permuting values between segments,
catches many comments at one spot (a junction, a geocoded address) and does not flag long
segments just because they collect more comments. The test runs on the comments
themselves, not on #25's kernel-smoothed field (smoothing makes every comment a bump); #25's
field and its CSV are unchanged. Street-only records can't be placed on a segment: they are
reported per hotspot (records naming a street the hotspot touches) but not tested.

Built on numpy, scipy and networkx instead of orchid's public_voice (#34, closed).
Run from the repo root: ``python -m src.concerns.hotspots``.
"""

import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import binom, false_discovery_control

from engine.orchid import street_network as sn
from src.concerns.concern_field import CONCERNS_PATH, CUTOFF_M, THEMES, load_concerns
from src.graph.build_graph import load_graph
from src.graph.street_lookup import build_street_lookup, find_streets

ROOT = Path(__file__).resolve().parents[2]
HOTSPOTS_PATH = ROOT / "data" / "concerns" / "hotspots.json"

BAND_M = 130.0          # a segment's neighbourhood: segments whose halfway points are within this
FDR = 0.05              # Benjamini-Hochberg false-discovery rate

HOT, NOT_SIGNIFICANT = "hot", "ns"


def _halfway(graph: sn.RoadGraph, edge) -> tuple[float, float]:
    """Return the point halfway along an edge's polyline (its end nodes if it has none)."""
    if len(edge.geometry) >= 2:
        coords = [tuple(c) for c in edge.geometry]
    else:
        coords = [graph.node(edge.u).xy, graph.node(edge.v).xy]
    steps = [math.dist(a, b) for a, b in zip(coords, coords[1:])]
    target = sum(steps) / 2.0
    for (a, b), step in zip(zip(coords, coords[1:]), steps):
        if target <= step and step > 0:
            t = target / step
            return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        target -= step
    return coords[-1]


def segments(graph: sn.RoadGraph) -> tuple[list[list[str]], np.ndarray, np.ndarray]:
    """Return the units -- [edge_id] for a one-way edge, [edge_id, twin_id] for a two-way
    street, in edge order -- their halfway points as an (n, 2) array in metres, and their
    lengths in metres (one direction's length for a twin pair).
    """
    units, points, lengths, seen = [], [], [], set()
    for edge in graph.iter_edges():
        if edge.eid in seen:
            continue
        ids = [edge.eid] if edge.reverse_eid is None else [edge.eid, edge.reverse_eid]
        seen.update(ids)
        units.append([str(i) for i in ids])
        points.append(_halfway(graph, edge))
        lengths.append(edge.length_m)
    return units, np.array(points, dtype=float).reshape(-1, 2), np.array(lengths, dtype=float)


def place_records(graph: sn.RoadGraph, units: list[list[str]], records: list[dict]
                  ) -> tuple[np.ndarray, np.ndarray, list[int | None]]:
    """Snap each located record to its nearest edge (within CUTOFF_M).

    Returns the comment count per unit, the per-unit, per-theme weight (severity * count),
    shape (n, len(THEMES)), and each record's unit index (None for a street-only record or
    one too far from every edge). Counts must be whole numbers.
    """
    unit_of = {eid: i for i, ids in enumerate(units) for eid in ids}
    index = sn.EdgeIndex(graph)
    counts = np.zeros(len(units))
    weights = np.zeros((len(units), len(THEMES)))
    placed = []
    for record in records:
        if record["count"] < 0 or record["count"] != int(record["count"]):
            raise ValueError(f"a record's count must be a whole number of comments, got {record['count']!r}")
        hit = index.nearest(*graph._project(*record["loc"]), max_dist_m=CUTOFF_M) if record["loc"] else None
        unit = unit_of[str(hit.eid)] if hit else None
        if unit is not None:
            counts[unit] += record["count"]
            weights[unit, THEMES.index(record["theme"])] += record["weight"]
        placed.append(unit)
    return counts, weights, placed


def band_neighbours(points: np.ndarray, band_m: float = BAND_M) -> list[np.ndarray]:
    """Return, for each unit, the sorted indices of the OTHER units within band_m of it."""
    neighbours = [[] for _ in range(len(points))]
    for i, j in cKDTree(points).query_pairs(band_m):
        neighbours[i].append(j)
        neighbours[j].append(i)
    return [np.array(sorted(n), dtype=int) for n in neighbours]


def neighbourhood_test(counts: np.ndarray, lengths: np.ndarray, neighbours: list[np.ndarray]
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return, per unit, the comments in its neighbourhood (itself plus its neighbours), the
    number expected if all comments had landed uniformly by network length, the exact
    binomial upper-tail p-value P(X >= observed) with X ~ Binomial(N, share), and whether
    the unit is testable (some comments placed, and its neighbourhood has length but is not
    the whole network). Untestable units get p = 1.
    """
    counts, lengths = np.asarray(counts, dtype=float), np.asarray(lengths, dtype=float)
    if not (np.all(np.isfinite(lengths)) and np.all(lengths >= 0) and lengths.sum() > 0):
        raise ValueError("segment lengths must be finite, non-negative and not all zero")
    total, n_comments = lengths.sum(), int(counts.sum())
    observed = np.array([counts[i] + counts[nb].sum() for i, nb in enumerate(neighbours)])
    share = np.array([(lengths[i] + lengths[nb].sum()) / total for i, nb in enumerate(neighbours)])
    testable = (n_comments > 0) & (share > 0) & (share < 1)
    p = np.ones(len(counts))
    p[testable] = binom.sf(observed[testable] - 1, n_comments, share[testable])
    return observed, n_comments * share, p, testable


def label_units(observed: np.ndarray, expected: np.ndarray, p: np.ndarray, testable: np.ndarray,
                fdr: float = FDR) -> tuple[np.ndarray, list[str]]:
    """Return Benjamini-Hochberg adjusted p-values (q, over the testable units only; 1 for the
    rest) and each unit's label: "hot" when testable, q <= fdr and more comments than expected.
    """
    q = np.ones(len(p))
    if testable.any():
        q[testable] = false_discovery_control(p[testable], method="bh")
    labels = [HOT if t and qi <= fdr and o > e else NOT_SIGNIFICANT
              for t, qi, o, e in zip(testable, q, observed, expected)]
    return q, labels


def hotspot_clusters(labels: list[str], neighbours: list[np.ndarray]) -> list[list[int]]:
    """Return groups of hot unit indices linked through hot neighbours within the band (straight-
    line distance, so a hotspot can span nearby parallel streets), each sorted, largest first.
    """
    hot = nx.Graph()
    hot.add_nodes_from(i for i, label in enumerate(labels) if label == HOT)
    hot.add_edges_from((i, int(j)) for i in hot for j in neighbours[i] if labels[j] == HOT)
    return sorted((sorted(c) for c in nx.connected_components(hot)), key=lambda c: (-len(c), c[0]))


def find_hotspots(graph: sn.RoadGraph, records: list[dict], band_m: float = BAND_M, fdr: float = FDR) -> dict:
    """Run the whole method on a graph and concern records; return the per-edge results,
    the hotspots and what couldn't be tested.
    """
    units, points, lengths = segments(graph)
    counts, weights, placed = place_records(graph, units, records)
    neighbours = band_neighbours(points, band_m)
    observed, expected, p, testable = neighbourhood_test(counts, lengths, neighbours)
    q, labels = label_units(observed, expected, p, testable, fdr)
    clusters = hotspot_clusters(labels, neighbours)

    street_only = sorted({r["street"] for r in records if not r["loc"] and r["street"]})
    found, unresolved = find_streets(build_street_lookup(graph), street_only)
    street_of = {str(edge.eid): edge.street_name for edge in graph.iter_edges()}
    hotspots = []
    for c in clusters:
        edges = {eid for i in c for eid in units[i]}
        members = set(c)
        hotspots.append({
            "edge_ids": sorted(edges, key=lambda e: (len(e), e)),
            "streets": sorted({street_of[e] for e in edges} - {""}),
            "comments": int(counts[c].sum()),
            "located_records": sum(unit in members for unit in placed),
            "weight": {theme: float(weights[c, t].sum()) for t, theme in enumerate(THEMES)},
            "max_rate_ratio": float(max(observed[i] / expected[i] for i in c)),
            "min_q": float(q[c].min()),
            "records_naming_its_streets": sum(1 for r in records if not r["loc"] and set(found.get(r["street"], ())) & edges),
        })
    per_edge = [{"edge_id": eid, "observed": float(observed[i]), "expected": float(expected[i]),
                 "p": float(p[i]), "q": float(q[i]), "label": labels[i]}
                for i, ids in enumerate(units) for eid in ids]
    return {
        "units": len(units), "testable_units": int(testable.sum()), "hot_units": labels.count(HOT),
        "comments_tested": int(counts.sum()),
        "located_records_tested": sum(u is not None for u in placed),
        "records_off_graph": sum(1 for r, u in zip(records, placed) if r["loc"] and u is None),
        "street_only_records": sum(1 for r in records if not r["loc"]),
        "unresolved_streets": unresolved,
        "hotspots": hotspots, "per_edge": per_edge,
    }


def main() -> None:
    """Find the hotspots among the consultation records and write them with their settings."""
    if not CONCERNS_PATH.exists():
        raise SystemExit(f"no concern records at {CONCERNS_PATH}")
    graph = load_graph()
    result = find_hotspots(graph, load_concerns(CONCERNS_PATH))
    HOTSPOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    HOTSPOTS_PATH.write_text(json.dumps({
        "method": "comments within band_m of each segment vs Binomial(N, share of network length); "
                  "exact upper tail; Benjamini-Hochberg over testable segments",
        "band_m": BAND_M, "fdr": FDR, **result,
    }, indent=1, allow_nan=False), encoding="utf-8")
    print(f"{result['hot_units']} of {result['testable_units']} testable street segments are hot after "
          f"Benjamini-Hochberg at FDR {FDR} -> {len(result['hotspots'])} hotspots -> {HOTSPOTS_PATH}")
    print(f"  tested {result['comments_tested']} comments from {result['located_records_tested']} located records; "
          f"{result['street_only_records']} street-only and {result['records_off_graph']} off-graph records not tested")
    for h in result["hotspots"]:
        print(f"  {', '.join(h['streets']) or 'unnamed'}: {len(h['edge_ids'])} edges, {h['comments']} comments, "
              f"up to {h['max_rate_ratio']:.1f}x expected, min q {h['min_q']:.3g}")
    for name in result["unresolved_streets"]:
        print(f"  street not found in the graph: {name!r}")


if __name__ == "__main__":
    main()
