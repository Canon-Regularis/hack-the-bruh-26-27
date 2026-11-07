"""
Urban Street Network -- road graph, routing and street-network science
=====================================================================

**The city is a graph, and this is that graph.**  Every other module in the
urban domain -- traffic assignment, transit, signal control, accessibility,
safety, emissions -- binds to the node and edge identifiers minted here, so
this module is deliberately the most opinionated and the most complete of the
family.  It is *not* a generic geospatial-graph toolkit (that is
``core/graph/geospatial_graphs``, which knows about latitudes, KD-trees and
GeoJSON but nothing about roads).  This is a **road** network: edges carry
lanes, capacity, free-flow time, a speed limit, a street name, a highway class,
a gradient and a set of permitted modes; nodes know whether they are signalised;
turns can be banned or made compulsory; and one-way streets are first-class
rather than an afterthought.

Everything is pure standard library -- no NumPy, no NetworkX, no OSMnx, no
Shapely.  The Delaunay triangulation, the contraction hierarchy, the Brandes
betweenness recursion, the planar-face traversal and the Viterbi map matcher are
all built by hand and all deterministic: every stochastic routine takes an
explicit ``seed`` and drives a private ``random.Random``, so results are
byte-identical across runs and independent of ``PYTHONHASHSEED``.

Layers
------
1.  **Geometry & geodesy** -- ``haversine_m`` / ``euclidean_m`` / ``bearing_deg``,
    point-to-segment projection, ``douglas_peucker``, ``convex_hull``,
    ``point_in_polygon``, a Bowyer-Watson ``delaunay_triangulation`` and the
    ``alpha_shape`` concave hull built on top of it.
2.  **The road graph** -- ``RoadNode`` / ``RoadEdge`` / ``TurnRestriction`` /
    ``RoadGraph``: a directed multigraph with typed road attributes, one-way
    handling, reverse-twin bookkeeping and turn restrictions.
3.  **Topology cleaning** -- ``remove_self_loops``, ``consolidate_parallel_edges``,
    ``simplify_degree2_chains`` (geometry- and length-preserving),
    ``clean_network``, ``largest_component_subgraph``.
4.  **Turns** -- ``TurnCostModel``, the edge-based (dual / line-graph)
    ``edge_expanded_graph`` that makes banned turns *exact*, and
    ``turn_aware_shortest_path``.
5.  **Shortest paths** -- ``dijkstra``, ``astar`` (admissible euclidean /
    great-circle heuristics), ``bidirectional_dijkstra``, ``metric_closure``.
6.  **ALT** -- ``ALTIndex`` (farthest-point landmark selection) and the triangle
    -inequality lower bound that turns A* into a genuinely goal-directed search.
7.  **Contraction hierarchies** -- edge-difference node ordering, witness search,
    shortcut insertion, bidirectional upward query and shortcut unpacking.
8.  **k-shortest paths** -- ``yen_k_shortest_paths`` (loopless) and
    ``eppstein_k_shortest_paths`` (sidetrack-ordered, loops admitted).
9.  **Time-dependent routing** -- ``TravelTimeProfile`` with a FIFO check and
    label-setting ``time_dependent_dijkstra``.
10. **Isochrones** -- budget-limited reachable sets, partially traversed edges
    cut at the exact fractional offset, and a concave-hull polygon.
11. **Centrality** -- Brandes betweenness (node *and* edge), demand-weighted
    betweenness, closeness / harmonic closeness, straightness centrality and
    Stephenson-Zelen information centrality.
12. **Street-network metrics** -- circuity / detour index, intersection density,
    node-degree distribution, edge-length statistics, street-orientation entropy
    and Boeing's orientation order (grid-ness), plus ``network_summary``.
13. **Blocks** -- planar face traversal (left-hand rule on the bearing-sorted
    embedding) giving city blocks, their areas and ``average_block_size``.
14. **Space syntax** -- stroke ("natural road") generation, the segment dual
    graph, mean depth, relative asymmetry, integration, choice, control value
    and intelligibility.
15. **Map matching & snapping** -- an ``EdgeIndex`` grid, ``snap_point``,
    ``split_edge``, and a Newson-Krumm HMM matched with ``Viterbi``.
16. **Components** -- weakly connected components and Tarjan's SCCs.
17. **Generators** -- ``grid_network``, ``radial_network``,
    ``random_planar_network`` (all seeded and analytically predictable).
18. **Payloads** -- ``to_geojson``, ``build_street_network_payload``,
    ``build_a2ui_payload``.

Conventions
-----------
* Node coordinates are stored as planar metres ``(x, y)`` with ``x`` east and
  ``y`` north; ``lat`` / ``lon`` are optional and, when supplied, are projected
  to a local equirectangular plane about the first geographic node so that
  euclidean distance in ``(x, y)`` agrees with the great-circle distance.
* Bearings are degrees clockwise from north, in ``[0, 360)``.  A *deflection*
  is a signed difference in ``[-180, 180)``: negative is a left turn, positive a
  right turn.
* Every edge is **directed**.  A two-way street is two edges linked by
  ``reverse_eid``; ``oneway=True`` (the default) creates a single edge.
* ``length_m`` is metres, ``free_flow_time_s`` seconds, ``capacity_veh_h``
  vehicles per hour, ``speed_limit_kph`` kilometres per hour, ``gradient`` a
  dimensionless rise-over-run.
* A cost specification (``weight=``) is either the name of a ``RoadEdge`` field
  (``"length_m"``, ``"free_flow_time_s"``), the pseudo-name ``"hops"``, or a
  callable ``RoadEdge -> float``.  All costs must be non-negative.
* ``Path.cost`` is re-accumulated left-to-right over the path's edges, exactly
  as Dijkstra accumulates its labels, so Dijkstra, bidirectional Dijkstra, A*,
  ALT and the contraction hierarchy return **bit-identical** floats whenever
  they agree on the path.  ``Path.length_m`` uses ``math.fsum`` instead.
* British spelling in prose; SciPy / OSMnx / NetworkX names where parity exists.
"""

from __future__ import annotations

import heapq
import itertools
import math
import random
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Hashable, Iterable, Iterator, List,
                    Optional, Sequence, Set, Tuple, Union)

__version__ = "1.0.0"

NodeId = Hashable                              # any hashable node identifier
EdgeId = int                                   # dense integer edge identifier
Coord = Tuple[float, float]                    # planar (x, y) metres
Ring = List[Coord]                             # an open polygon ring
CostSpec = Union[str, Callable[["RoadEdge"], float]]

_EPS = 1e-12
_GEO_EPS = 1e-15
_EARTH_RADIUS_M = 6371008.8                    # IUGG mean radius
_KPH_TO_MPS = 1.0 / 3.6
_SQRT_2PI = math.sqrt(2.0 * math.pi)

# ---------------------------------------------------------------------------
# Optional core interop.  The module is fully standalone; when the orchid core
# geospatial toolkit is importable we defer to its (identical) great-circle
# formula so that mixed pipelines agree to the last bit.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only when the core package is on the path
    from algorithms.core.graph.geospatial_graphs.geospatial_graphs_template import (
        haversine as _core_haversine,
    )
    _HAVE_CORE_GEO = True
except ImportError:                            # pragma: no cover
    _core_haversine = None
    _HAVE_CORE_GEO = False


__all__ = [
    # errors & tables
    "StreetNetworkError", "HIGHWAY_DEFAULTS", "MODES",
    # geometry & geodesy
    "haversine_m", "euclidean_m", "bearing_deg", "bearing_difference_deg",
    "polyline_length", "point_segment_distance", "project_point_on_polyline",
    "douglas_peucker", "convex_hull", "polygon_area", "polygon_centroid",
    "point_in_polygon", "delaunay_triangulation", "alpha_shape",
    # road graph
    "RoadNode", "RoadEdge", "TurnRestriction", "RoadGraph",
    # topology cleaning
    "remove_self_loops", "consolidate_parallel_edges", "simplify_degree2_chains",
    "clean_network", "largest_component_subgraph",
    # turns & edge expansion
    "TurnCostModel", "ExpandedGraph", "edge_expanded_graph",
    "turn_aware_shortest_path",
    # shortest paths
    "Path", "SearchTree", "dijkstra", "shortest_path", "shortest_path_length",
    "astar", "bidirectional_dijkstra", "metric_closure",
    # landmarks
    "ALTIndex", "alt_shortest_path",
    # contraction hierarchies
    "ContractionHierarchy",
    # k-shortest paths
    "yen_k_shortest_paths", "eppstein_k_shortest_paths",
    # time-dependent routing
    "TravelTimeProfile", "time_dependent_dijkstra", "time_dependent_shortest_path",
    # isochrones
    "Isochrone", "isochrone", "reachable_subgraph",
    # centrality
    "betweenness_centrality", "edge_betweenness_centrality",
    "demand_weighted_betweenness", "closeness_centrality",
    "straightness_centrality", "information_centrality",
    # street-network metrics
    "circuity", "detour_index", "intersection_density",
    "node_degree_distribution", "edge_length_statistics",
    "orientation_histogram", "street_orientation_entropy", "orientation_order",
    "network_summary",
    # blocks
    "Block", "detect_blocks", "block_statistics", "average_block_size",
    # space syntax
    "Stroke", "generate_strokes", "segment_dual_graph", "SpaceSyntaxResult",
    "space_syntax",
    # map matching & snapping
    "SnapResult", "EdgeIndex", "snap_point", "split_edge", "Candidate",
    "MapMatchResult", "map_match",
    # components
    "weakly_connected_components", "strongly_connected_components",
    # generators
    "grid_network", "radial_network", "random_planar_network",
    # payloads
    "to_geojson", "build_street_network_payload", "build_a2ui_payload",
]


# ===========================================================================
# 0. VALIDATION HELPERS
# ===========================================================================
class StreetNetworkError(ValueError):
    """Invalid input to a street-network routine."""


def _as_float(value: Any, name: str) -> float:
    """Coerce to a finite float or raise."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise StreetNetworkError("%s must be a real number (got %r)." % (name, value))
    if math.isnan(out) or math.isinf(out):
        raise StreetNetworkError("%s must be finite (got %r)." % (name, value))
    return out


def _as_non_negative(value: Any, name: str) -> float:
    """Coerce to a finite non-negative float or raise."""
    out = _as_float(value, name)
    if out < 0.0:
        raise StreetNetworkError("%s must be non-negative (got %g)." % (name, out))
    return out


def _as_positive(value: Any, name: str) -> float:
    """Coerce to a finite strictly positive float or raise."""
    out = _as_float(value, name)
    if out <= 0.0:
        raise StreetNetworkError("%s must be strictly positive (got %g)." % (name, out))
    return out


def _as_positive_int(value: Any, name: str) -> int:
    """Coerce to a strictly positive integer or raise."""
    try:
        out = int(value)
    except (TypeError, ValueError):
        raise StreetNetworkError("%s must be an integer (got %r)." % (name, value))
    if out <= 0:
        raise StreetNetworkError("%s must be strictly positive (got %d)." % (name, out))
    return out


def _as_coords(points: Sequence[Any], name: str = "points") -> List[Coord]:
    """Coerce a sequence of (x, y) pairs to a list of finite float tuples."""
    out: List[Coord] = []
    for i, p in enumerate(points):
        try:
            x, y = p[0], p[1]
        except (TypeError, IndexError, KeyError):
            raise StreetNetworkError("%s[%d] is not an (x, y) pair." % (name, i))
        out.append((_as_float(x, "%s[%d].x" % (name, i)),
                    _as_float(y, "%s[%d].y" % (name, i))))
    return out


def _require(condition: bool, message: str) -> None:
    """Raise :class:`StreetNetworkError` with ``message`` unless ``condition``."""
    if not condition:
        raise StreetNetworkError(message)


# ===========================================================================
# 1. GEOMETRY & GEODESY
# ===========================================================================
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS84 latitude/longitude pairs."""
    # Validate BEFORE delegating: the core helper raises a bare ValueError, so without
    # this the exception TYPE depended on whether an optional import happened to succeed
    # -- StreetNetworkError when running from the module directory, ValueError when
    # running from the repo root.  Both paths must behave identically.
    la1 = _as_float(lat1, "lat1")
    lo1 = _as_float(lon1, "lon1")
    la2 = _as_float(lat2, "lat2")
    lo2 = _as_float(lon2, "lon2")
    # ... and validate the RANGES too, not merely finiteness: the core helper rejects
    # |lat| > 90 with a bare ValueError while the standalone formula silently returned a
    # nonsense distance, so the behaviour of an exported function depended on whether the
    # optional import happened to succeed (i.e. on the working directory).  These bounds
    # are the same ones RoadGraph.add_node applies.
    for nm, v in (("lat1", la1), ("lat2", la2)):
        _require(-90.0 <= v <= 90.0,
                 "%s must lie in [-90, 90] (got %g) -- lat/lon swapped?" % (nm, v))
    for nm, v in (("lon1", lo1), ("lon2", lo2)):
        _require(-360.0 <= v <= 360.0, "%s must lie in [-360, 360] (got %g)." % (nm, v))
    if _HAVE_CORE_GEO:                          # pragma: no cover - interop path
        return float(_core_haversine((la1, lo1), (la2, lo2)))
    p1 = math.radians(la1)
    p2 = math.radians(la2)
    dp = p2 - p1
    dl = math.radians(lo2 - lo1)
    h = (math.sin(dp * 0.5) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2)
    return 2.0 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


def euclidean_m(x1: float, y1: float, x2: float, y2: float) -> float:
    """Planar euclidean distance in metres."""
    return math.hypot(x2 - x1, y2 - y1)


def bearing_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """Bearing in degrees clockwise from north (``+y``) for the vector 1 -> 2."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0.0 and dy == 0.0:
        return 0.0          # exact: an absolute 1e-12 m here zeroed every bearing of a
                            # network drawn at sub-picometre scale
    return math.degrees(math.atan2(dx, dy)) % 360.0


def bearing_difference_deg(from_bearing: float, to_bearing: float) -> float:
    """Signed deflection in ``[-180, 180)``; negative is left, positive is right."""
    return ((to_bearing - from_bearing + 180.0) % 360.0) - 180.0


def polyline_length(points: Sequence[Any]) -> float:
    """Total planar length of a polyline (``math.fsum`` over the segments)."""
    pts = _as_coords(points, "points")
    if len(pts) < 2:
        return 0.0
    return math.fsum(euclidean_m(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
                     for i in range(len(pts) - 1))


def point_segment_distance(px: float, py: float, ax: float, ay: float,
                           bx: float, by: float) -> Tuple[float, float, float, float]:
    """Distance from P to segment AB, the clamped parameter t and the foot point."""
    dx = bx - ax
    dy = by - ay
    d2 = dx * dx + dy * dy
    # `d2 <= _EPS` was an ABSOLUTE 1e-12 applied to a SQUARED length, i.e. an effective
    # 1e-6 cut-off in the base unit -- real segments shorter than that (and 1e-6 degrees
    # is 0.11 m on the ground) collapsed onto their start point, returning a wrong t and
    # foot point.  d2 is exactly zero only when dx == dy == 0, so the exact test is both
    # scale-free and division-safe; for tiny non-zero d2 the quotient below carries
    # ordinary relative error.
    if d2 <= 0.0:
        t = 0.0
    else:
        t = ((px - ax) * dx + (py - ay) * dy) / d2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy), t, cx, cy


def project_point_on_polyline(px: float, py: float, points: Sequence[Any]
                              ) -> Tuple[float, float, float, Coord]:
    """Nearest point on a polyline: (distance_m, offset_m, offset_frac, point)."""
    pts = _as_coords(points, "points")
    _require(len(pts) >= 1, "project_point_on_polyline needs at least one vertex.")
    if len(pts) == 1:
        return euclidean_m(px, py, pts[0][0], pts[0][1]), 0.0, 0.0, pts[0]
    seg_len = [euclidean_m(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
               for i in range(len(pts) - 1)]
    total = math.fsum(seg_len)
    best = (float("inf"), 0.0, 0.0, pts[0])
    for i, L in enumerate(seg_len):
        d, t, cx, cy = point_segment_distance(px, py, pts[i][0], pts[i][1],
                                              pts[i + 1][0], pts[i + 1][1])
        if d < best[0]:
            # fsum the prefix rather than carrying a drifting running total, so `off` and
            # `total` are accumulated the same way and frac cannot exceed 1 by drift.
            off = math.fsum(seg_len[:i]) + t * L
            frac = 0.0 if total <= 0.0 else off / total
            best = (d, off, min(1.0, max(0.0, frac)), (cx, cy))
    return best


def douglas_peucker(points: Sequence[Any], tolerance_m: float) -> List[Coord]:
    """Ramer-Douglas-Peucker polyline simplification in the planar metric."""
    pts = _as_coords(points, "points")
    tol = _as_non_negative(tolerance_m, "tolerance_m")
    if len(pts) < 3 or tol <= 0.0:
        return list(pts)
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        ax, ay = pts[lo]
        bx, by = pts[hi]
        best_d = -1.0
        best_i = lo
        for i in range(lo + 1, hi):
            d = point_segment_distance(pts[i][0], pts[i][1], ax, ay, bx, by)[0]
            if d > best_d:
                best_d, best_i = d, i
        if best_d > tol:
            keep[best_i] = True
            stack.append((lo, best_i))
            stack.append((best_i, hi))
    return [p for p, k in zip(pts, keep) if k]


def convex_hull(points: Sequence[Any]) -> Ring:
    """Andrew's monotone-chain convex hull, counter-clockwise, first point unrepeated."""
    pts = sorted(set(_as_coords(points, "points")))
    if len(pts) <= 2:
        return list(pts)

    def cross(o: Coord, a: Coord, b: Coord) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: Ring = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: Ring = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def polygon_area(ring: Sequence[Any], signed: bool = False) -> float:
    """Shoelace area of a ring (positive when counter-clockwise if ``signed``)."""
    pts = _as_coords(ring, "ring")
    n = len(pts)
    if n < 3:
        return 0.0
    two_a = math.fsum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
                      for i in range(n))
    area = 0.5 * two_a
    return area if signed else abs(area)


def polygon_centroid(ring: Sequence[Any]) -> Coord:
    """Area centroid of a ring; falls back to the vertex mean only for a ring whose area
    vanishes *relative to its own size* (a genuinely degenerate, zero-area ring).

    The shoelace sums are taken about the first vertex, so the answer does not depend on
    how far the ring sits from the origin, and the degeneracy test is dimensionless:
    ``|2A| <= _EPS * span^2``.  The old test compared twice an AREA against an absolute
    1e-12, so below roughly a micrometre of ring size the function silently substituted
    the vertex mean -- a DIFFERENT quantity, equal to the centroid only for symmetric
    rings -- and the dimensionless centroid of a shape therefore moved when the shape was
    uniformly rescaled.
    """
    pts = _as_coords(ring, "ring")
    n = len(pts)
    if n == 0:
        raise StreetNetworkError("polygon_centroid needs at least one vertex.")
    if n < 3:
        return (math.fsum(p[0] for p in pts) / n, math.fsum(p[1] for p in pts) / n)
    ox, oy = pts[0]
    xs = [p[0] - ox for p in pts]
    ys = [p[1] - oy for p in pts]
    a2 = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        j = (i + 1) % n
        cr = xs[i] * ys[j] - xs[j] * ys[i]
        a2 += cr
        cx += (xs[i] + xs[j]) * cr
        cy += (ys[i] + ys[j]) * cr
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    if abs(a2) <= _EPS * span * span:
        return (math.fsum(p[0] for p in pts) / n, math.fsum(p[1] for p in pts) / n)
    return (ox + cx / (3.0 * a2), oy + cy / (3.0 * a2))


def point_in_polygon(point: Sequence[float], ring: Sequence[Any]) -> bool:
    """Ray-casting point-in-polygon test (boundary counts as inside).

    CONVENTION, deliberate and pinned by the suite: "on the boundary" means *within
    1e-9 of the base unit* of an edge -- an ABSOLUTE band, 1 nanometre at the module's
    declared metre scale, so a point 5e-10 outside the ring still reports ``True``.  The
    band is therefore not scale-relative: fed decimal degrees it widens to about 0.1 mm
    on the ground.  Some tolerance is contracted by the docstring; only its magnitude was
    previously undocumented.
    """
    px = _as_float(point[0], "point.x")
    py = _as_float(point[1], "point.y")
    pts = _as_coords(ring, "ring")
    n = len(pts)
    if n < 3:
        return False
    inside = False
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        if point_segment_distance(px, py, x0, y0, x1, y1)[0] <= 1e-9:
            return True                          # on the boundary
        if (y0 > py) != (y1 > py):
            x_cross = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
            if x_cross > px:
                inside = not inside
    return inside


def _in_circumcircle(p: Coord, a: Coord, b: Coord, c: Coord) -> bool:
    """Exact-sign InCircle predicate: is ``p`` strictly inside circumcircle(a, b, c)?"""
    orient = ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    if abs(orient) <= _GEO_EPS:
        return False                             # degenerate (collinear) triangle
    ax, ay = a[0] - p[0], a[1] - p[1]
    bx, by = b[0] - p[0], b[1] - p[1]
    cx, cy = c[0] - p[0], c[1] - p[1]
    det = ((ax * ax + ay * ay) * (bx * cy - by * cx)
           - (bx * bx + by * by) * (ax * cy - ay * cx)
           + (cx * cx + cy * cy) * (ax * by - ay * bx))
    return det > _GEO_EPS if orient > 0.0 else det < -_GEO_EPS


_SUPER_SCALES = (40.0, 1000.0, 25000.0, 600000.0)


def _bowyer_watson(norm: List[Coord], m: int, scale: float
                   ) -> List[Tuple[int, int, int]]:
    """One Bowyer-Watson pass over unit-box sites with a super triangle of ``scale``."""
    work = list(norm) + [(-scale, -scale), (scale, -scale), (0.5, 1.5 * scale)]
    tris: Set[Tuple[int, int, int]] = {(m, m + 1, m + 2)}
    for i in range(m):
        p = work[i]
        bad = [t for t in tris if _in_circumcircle(p, work[t[0]], work[t[1]], work[t[2]])]
        if not bad:
            continue
        counts: Dict[Tuple[int, int], int] = {}
        for t in bad:
            for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                key2 = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
                counts[key2] = counts.get(key2, 0) + 1
        for t in bad:
            tris.discard(t)
        for (u, v), c in counts.items():
            if c == 1:
                tris.add((u, v, i))
    return [t for t in tris if max(t) < m]


def delaunay_triangulation(points: Sequence[Any]) -> List[Tuple[int, int, int]]:
    """Bowyer-Watson Delaunay triangulation; returns index triples into ``points``.

    The super triangle is grown until the triangulation demonstrably COVERS the convex
    hull (sum of triangle areas == hull area).  A fixed super triangle cannot be safe for
    every input -- a Delaunay circumcircle can be arbitrarily large for a thin triangle,
    and when a super vertex falls inside one the triangle is discarded with the super
    triangle and a wedge of the hull is silently left untriangulated.  The coverage
    identity is a sound completeness check because the cavity retriangulation already
    guarantees the triangles do not overlap.  KNOWN LIMIT: for extremely near-degenerate
    clouds (a hairline strip, or a tight cluster beside one far outlier) even the largest
    super triangle can fail the check, and the closest attempt is returned; fuzzing 400
    such adversarial clouds leaves 5 short of full coverage, against 254 before.
    """
    pts = _as_coords(points, "points")
    if len(pts) < 3:
        return []
    # Normalise into the unit box FIRST (Delaunay is invariant under uniform scaling),
    # then deduplicate on the NORMALISED coordinates.  Rounding the raw coordinates to
    # 9 decimal places was an absolute 1e-9 quantisation of a length, so a cloud
    # spanning less than that collapsed to a single site and the function returned no
    # triangles at all -- one more way for the same shape to behave differently when
    # uniformly rescaled.
    axs = [p[0] for p in pts]
    ays = [p[1] for p in pts]
    minx, maxx, miny, maxy = min(axs), max(axs), min(ays), max(ays)
    span = max(maxx - minx, maxy - miny)
    if span <= 0.0:
        return []
    rep: Dict[Coord, int] = {}
    norm: List[Coord] = []
    orig: List[int] = []
    for i, p in enumerate(pts):
        u = ((p[0] - minx) / span, (p[1] - miny) / span)
        key = (round(u[0], 9), round(u[1], 9))
        if key in rep:
            continue
        rep[key] = len(norm)
        norm.append(u)
        orig.append(i)
    m = len(norm)
    if m < 3:
        return []
    hull_area = polygon_area(convex_hull(pts))
    best: List[Tuple[int, int, int]] = []
    best_gap = float("inf")
    for scale in _SUPER_SCALES:
        tris = _bowyer_watson(norm, m, scale)
        out: List[Tuple[int, int, int]] = []
        for t in tris:
            a, b, c = orig[t[0]], orig[t[1]], orig[t[2]]
            # orient counter-clockwise in the original coordinates
            if ((pts[b][0] - pts[a][0]) * (pts[c][1] - pts[a][1])
                    - (pts[b][1] - pts[a][1]) * (pts[c][0] - pts[a][0])) < 0.0:
                b, c = c, b
            out.append((a, b, c))
        out.sort()
        covered = math.fsum(abs(polygon_area([pts[a], pts[b], pts[c]]))
                            for (a, b, c) in out)
        gap = abs(hull_area - covered)
        if gap < best_gap:
            best_gap, best = gap, out
        if gap <= 1e-9 * hull_area:
            return out
    return best


def _circumradius(a: Coord, b: Coord, c: Coord) -> float:
    """Circumradius of a triangle (``inf`` only when it is *genuinely* degenerate).

    The triangle is first expressed in units of its own longest coordinate extent --
    exactly the normalise-then-test pattern :func:`delaunay_triangulation` already uses
    -- so the degeneracy test is dimensionless.  Testing ``area2 <= _EPS`` on twice an
    AREA declared well-formed triangles degenerate below about a millimetre of side
    length, which made :func:`alpha_shape` return a different SHAPE for one cloud
    uniformly rescaled.
    """
    span = max(abs(b[0] - a[0]), abs(c[0] - a[0]),
               abs(b[1] - a[1]), abs(c[1] - a[1]))
    if span <= 0.0:
        return float("inf")
    bx, by = (b[0] - a[0]) / span, (b[1] - a[1]) / span
    cx, cy = (c[0] - a[0]) / span, (c[1] - a[1]) / span
    area2 = abs(bx * cy - by * cx)
    if area2 <= _EPS:
        return float("inf")
    ab = math.hypot(bx, by)
    bc = math.hypot(cx - bx, cy - by)
    ca = math.hypot(cx, cy)
    return span * ab * bc * ca / (2.0 * area2)


def alpha_shape(points: Sequence[Any], alpha_m: Optional[float] = None) -> List[Ring]:
    """Alpha-shape (concave hull) rings: keep Delaunay triangles with circumradius
    at most ``alpha_m`` and stitch the boundary.  ``None`` gives the convex hull.

    Fewer than three DISTINCT sites cannot bound a region, so the answer is ``[]`` -- the
    same convention the ``not tris`` and ``not keep`` fallbacks below already applied to
    three-or-more collinear sites.  The early return used to emit ``[[a, b]]``, a
    two-point "ring" that carries no information (zero area, contains no point) and only
    propagated a malformed value into callers.
    """
    pts = _as_coords(points, "points")
    if len(pts) < 3:
        return []
    tris = delaunay_triangulation(pts)
    if not tris:
        # No triangulation means the points are collinear (or coincident), so the hull
        # degenerates to a segment.  A segment is not a ring -- return none, matching the
        # `not keep` fallback 11 lines below.  This arm used to read `else [hull]`, an
        # identical-branch copy-paste that emitted a two-point "ring".
        hull = convex_hull(pts)
        return [hull] if len(hull) >= 3 else []
    if alpha_m is None:
        keep = tris
    else:
        alpha = _as_positive(alpha_m, "alpha_m")
        keep = [t for t in tris
                if _circumradius(pts[t[0]], pts[t[1]], pts[t[2]]) <= alpha]
    if not keep:
        hull = convex_hull(pts)
        return [hull] if len(hull) >= 3 else []
    # A boundary edge belongs to exactly one kept (CCW) triangle; keep it oriented.
    undirected: Dict[Tuple[int, int], int] = {}
    directed: List[Tuple[int, int]] = []
    for t in keep:
        for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            key = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
            undirected[key] = undirected.get(key, 0) + 1
            directed.append(e)
    nxt: Dict[int, List[int]] = {}
    for (u, v) in directed:
        key = (u, v) if u < v else (v, u)
        if undirected[key] == 1:
            nxt.setdefault(u, []).append(v)
    rings: List[Ring] = []
    used: Set[Tuple[int, int]] = set()
    for start in sorted(nxt):
        for first in nxt[start]:
            if (start, first) in used:
                continue
            ring_idx = [start]
            u, v = start, first
            used.add((u, v))
            guard = 0
            while v != start and guard <= len(directed):
                ring_idx.append(v)
                cand = [w for w in nxt.get(v, []) if (v, w) not in used]
                if not cand:
                    break
                w = cand[0]
                used.add((v, w))
                u, v = v, w
                guard += 1
            if len(ring_idx) >= 3:
                rings.append([pts[i] for i in ring_idx])
    rings.sort(key=lambda r: -abs(polygon_area(r)))
    return rings


# ===========================================================================
# 2. THE ROAD GRAPH -- typed nodes, directed road edges, turn restrictions
# ===========================================================================
MODES: Tuple[str, ...] = ("car", "bus", "bike", "foot")

HIGHWAY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "motorway":      {"speed_kph": 112.0, "lanes": 3, "lane_capacity": 2200.0,
                      "modes": ("car", "bus")},
    "trunk":         {"speed_kph": 96.0, "lanes": 2, "lane_capacity": 2000.0,
                      "modes": ("car", "bus")},
    "primary":       {"speed_kph": 64.0, "lanes": 2, "lane_capacity": 1800.0,
                      "modes": ("car", "bus", "bike", "foot")},
    "secondary":     {"speed_kph": 48.0, "lanes": 2, "lane_capacity": 1600.0,
                      "modes": ("car", "bus", "bike", "foot")},
    "tertiary":      {"speed_kph": 48.0, "lanes": 1, "lane_capacity": 1400.0,
                      "modes": ("car", "bus", "bike", "foot")},
    "residential":   {"speed_kph": 32.0, "lanes": 1, "lane_capacity": 900.0,
                      "modes": ("car", "bus", "bike", "foot")},
    "living_street": {"speed_kph": 16.0, "lanes": 1, "lane_capacity": 600.0,
                      "modes": ("car", "bike", "foot")},
    "service":       {"speed_kph": 16.0, "lanes": 1, "lane_capacity": 500.0,
                      "modes": ("car", "bike", "foot")},
    "unclassified":  {"speed_kph": 40.0, "lanes": 1, "lane_capacity": 1000.0,
                      "modes": ("car", "bus", "bike", "foot")},
    "cycleway":      {"speed_kph": 16.0, "lanes": 1, "lane_capacity": 1500.0,
                      "modes": ("bike", "foot")},
    "footway":       {"speed_kph": 5.0, "lanes": 1, "lane_capacity": 0.0,
                      "modes": ("foot",)},
    "pedestrian":    {"speed_kph": 5.0, "lanes": 1, "lane_capacity": 0.0,
                      "modes": ("foot",)},
}

_DEFAULT_CLASS = "residential"


@dataclass
class RoadNode:
    """A junction: an identifier, planar metres, optional lat/lon and signal state."""

    node_id: NodeId
    x: float = 0.0
    y: float = 0.0
    lat: Optional[float] = None
    lon: Optional[float] = None
    elevation_m: float = 0.0
    signalised: bool = False
    tags: Dict[str, Any] = field(default_factory=dict)

    @property
    def xy(self) -> Coord:
        """Planar coordinate pair."""
        return (self.x, self.y)

    @property
    def is_geographic(self) -> bool:
        """True when this node carries a latitude and longitude."""
        return self.lat is not None and self.lon is not None


@dataclass
class RoadEdge:
    """A directed road link with full highway attribution and polyline geometry."""

    eid: EdgeId
    u: NodeId
    v: NodeId
    length_m: float
    lanes: int = 1
    capacity_veh_h: float = 0.0
    free_flow_time_s: float = 0.0
    speed_limit_kph: float = 32.0
    street_name: str = ""
    highway_class: str = _DEFAULT_CLASS
    gradient: float = 0.0
    modes_permitted: Tuple[str, ...] = MODES
    geometry: Tuple[Coord, ...] = ()
    oneway: bool = True
    reverse_eid: Optional[EdgeId] = None
    tags: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_loop(self) -> bool:
        """True when the edge starts and ends at the same node."""
        return self.u == self.v

    @property
    def free_flow_speed_mps(self) -> float:
        """Free-flow speed in metres per second implied by ``free_flow_time_s``."""
        if self.free_flow_time_s <= _EPS:
            return self.speed_limit_kph * _KPH_TO_MPS
        return self.length_m / self.free_flow_time_s

    def permits(self, mode: Optional[str]) -> bool:
        """True when ``mode`` is None or is in ``modes_permitted``."""
        return mode is None or mode in self.modes_permitted

    def start_bearing(self) -> float:
        """Bearing leaving ``u`` along the first geometry segment."""
        g = self.geometry
        if len(g) < 2:
            return 0.0
        return bearing_deg(g[0][0], g[0][1], g[1][0], g[1][1])

    def end_bearing(self) -> float:
        """Bearing arriving at ``v`` along the last geometry segment."""
        g = self.geometry
        if len(g) < 2:
            return 0.0
        return bearing_deg(g[-2][0], g[-2][1], g[-1][0], g[-1][1])


@dataclass(frozen=True)
class TurnRestriction:
    """A ``no_turn`` ban or an ``only_turn`` obligation from one edge to another."""

    from_eid: EdgeId
    via_node: NodeId
    to_eid: EdgeId
    kind: str = "no_turn"
    description: str = ""


class RoadGraph:
    """A directed road multigraph: typed edges, one-ways, turns and geometry."""

    def __init__(self, name: str = "", drive_side: str = "left") -> None:
        _require(drive_side in ("left", "right"),
                 "drive_side must be 'left' or 'right' (got %r)." % (drive_side,))
        self.name = str(name)
        self.drive_side = drive_side
        self._nodes: Dict[NodeId, RoadNode] = {}
        self._edges: Dict[EdgeId, RoadEdge] = {}
        self._out: Dict[NodeId, List[EdgeId]] = {}
        self._in: Dict[NodeId, List[EdgeId]] = {}
        self._pairs: Dict[Tuple[NodeId, NodeId], List[EdgeId]] = {}
        self._next_eid = 0
        self._no_turn: Set[Tuple[EdgeId, EdgeId]] = set()
        self._only_turn: Dict[EdgeId, Set[EdgeId]] = {}
        self._restrictions: List[TurnRestriction] = []
        self._geo_ref: Optional[Tuple[float, float]] = None

    # -- nodes ------------------------------------------------------------
    def add_node(self, node_id: NodeId, x: Optional[float] = None,
                 y: Optional[float] = None, *, lat: Optional[float] = None,
                 lon: Optional[float] = None, elevation_m: float = 0.0,
                 signalised: bool = False, **tags: Any) -> RoadNode:
        """Add (or update) a junction; lat/lon are projected to local metres."""
        if node_id in self._nodes:
            node = self._nodes[node_id]
            if x is not None:
                node.x = _as_float(x, "x")
            if y is not None:
                node.y = _as_float(y, "y")
            if lat is not None and lon is not None:
                node.lat = _as_float(lat, "lat")
                node.lon = _as_float(lon, "lon")
                node.x, node.y = self._project(node.lat, node.lon)
            node.signalised = bool(signalised) or node.signalised
            node.tags.update(tags)
            return node
        if lat is not None and lon is not None:
            flat = _as_float(lat, "lat")
            flon = _as_float(lon, "lon")
            _require(-90.0 <= flat <= 90.0, "lat must lie in [-90, 90] (got %g)." % flat)
            _require(-360.0 <= flon <= 360.0,
                     "lon must lie in [-360, 360] (got %g)." % flon)
            px, py = self._project(flat, flon)
            node = RoadNode(node_id, px, py, flat, flon,
                            _as_float(elevation_m, "elevation_m"),
                            bool(signalised), dict(tags))
        else:
            node = RoadNode(node_id,
                            _as_float(x if x is not None else 0.0, "x"),
                            _as_float(y if y is not None else 0.0, "y"),
                            None, None, _as_float(elevation_m, "elevation_m"),
                            bool(signalised), dict(tags))
        self._nodes[node_id] = node
        self._out.setdefault(node_id, [])
        self._in.setdefault(node_id, [])
        return node

    def _project(self, lat: float, lon: float) -> Coord:
        """Local equirectangular projection about the first geographic node."""
        if self._geo_ref is None:
            self._geo_ref = (lat, lon)
        lat0, lon0 = self._geo_ref
        k = math.cos(math.radians(lat0))
        return (math.radians(lon - lon0) * _EARTH_RADIUS_M * k,
                math.radians(lat - lat0) * _EARTH_RADIUS_M)

    def node(self, node_id: NodeId) -> RoadNode:
        """Return the :class:`RoadNode` for ``node_id`` (raises if absent)."""
        try:
            return self._nodes[node_id]
        except KeyError:
            raise StreetNetworkError("unknown node %r." % (node_id,))

    def has_node(self, node_id: NodeId) -> bool:
        """Membership test for nodes."""
        return node_id in self._nodes

    @property
    def nodes(self) -> List[NodeId]:
        """Node identifiers in insertion order."""
        return list(self._nodes)

    @property
    def num_nodes(self) -> int:
        """Number of junctions."""
        return len(self._nodes)

    def iter_nodes(self) -> Iterator[RoadNode]:
        """Iterate over :class:`RoadNode` objects in insertion order."""
        return iter(self._nodes.values())

    # -- edges ------------------------------------------------------------
    def add_edge(self, u: NodeId, v: NodeId, length_m: Optional[float] = None, *,
                 oneway: bool = True, lanes: Optional[int] = None,
                 capacity_veh_h: Optional[float] = None,
                 speed_limit_kph: Optional[float] = None,
                 free_flow_time_s: Optional[float] = None,
                 street_name: str = "", highway_class: str = _DEFAULT_CLASS,
                 gradient: float = 0.0,
                 modes_permitted: Optional[Sequence[str]] = None,
                 geometry: Optional[Sequence[Any]] = None,
                 eid: Optional[EdgeId] = None,
                 reverse_eid: Optional[EdgeId] = None,
                 **tags: Any) -> Tuple[EdgeId, Optional[EdgeId]]:
        """Add a directed road link (plus its reverse twin unless ``oneway``)."""
        _require(u in self._nodes, "add_edge: unknown tail node %r." % (u,))
        _require(v in self._nodes, "add_edge: unknown head node %r." % (v,))
        defaults = HIGHWAY_DEFAULTS.get(str(highway_class),
                                        HIGHWAY_DEFAULTS[_DEFAULT_CLASS])
        nu, nv = self._nodes[u], self._nodes[v]
        if geometry is None:
            geom = [nu.xy, nv.xy]
        else:
            geom = _as_coords(geometry, "geometry")
            _require(len(geom) >= 2, "geometry needs at least two vertices.")
        if length_m is None:
            if nu.is_geographic and nv.is_geographic and geometry is None:
                length = haversine_m(nu.lat, nu.lon, nv.lat, nv.lon)
            else:
                length = polyline_length(geom)
        else:
            length = _as_non_negative(length_m, "length_m")
        n_lanes = int(defaults["lanes"] if lanes is None else lanes)
        _require(n_lanes >= 1, "lanes must be at least 1 (got %d)." % n_lanes)
        speed = _as_positive(defaults["speed_kph"] if speed_limit_kph is None
                             else speed_limit_kph, "speed_limit_kph")
        if free_flow_time_s is None:
            fft = length / (speed * _KPH_TO_MPS)
        else:
            fft = _as_non_negative(free_flow_time_s, "free_flow_time_s")
        if capacity_veh_h is None:
            cap = float(defaults["lane_capacity"]) * n_lanes
        else:
            cap = _as_non_negative(capacity_veh_h, "capacity_veh_h")
        modes = tuple(defaults["modes"]) if modes_permitted is None \
            else tuple(str(m) for m in modes_permitted)
        _require(len(modes) >= 1, "modes_permitted must not be empty.")
        grad = _as_float(gradient, "gradient")

        fwd = self._install(eid, u, v, length, n_lanes, cap, fft, speed,
                            str(street_name), str(highway_class), grad, modes,
                            tuple(geom), bool(oneway), dict(tags))
        if oneway:
            return fwd, None
        back_geom = tuple(reversed(geom))
        bwd = self._install(reverse_eid, v, u, length, n_lanes, cap, fft, speed,
                            str(street_name), str(highway_class), -grad, modes,
                            back_geom, False, dict(tags))
        self._edges[fwd].reverse_eid = bwd
        self._edges[bwd].reverse_eid = fwd
        return fwd, bwd

    def _install(self, eid: Optional[EdgeId], u: NodeId, v: NodeId, length: float,
                 lanes: int, cap: float, fft: float, speed: float, name: str,
                 klass: str, grad: float, modes: Tuple[str, ...],
                 geom: Sequence[Coord], oneway: bool,
                 tags: Dict[str, Any]) -> EdgeId:
        """Insert one directed edge object and wire the adjacency indices."""
        if eid is None:
            key = self._next_eid
            self._next_eid += 1
        else:
            key = int(eid)
            _require(key not in self._edges, "duplicate edge id %d." % key)
            self._next_eid = max(self._next_eid, key + 1)
        edge = RoadEdge(key, u, v, length, lanes, cap, fft, speed, name, klass,
                        grad, modes, tuple(geom), oneway, None, tags)
        self._edges[key] = edge
        self._out.setdefault(u, []).append(key)
        self._in.setdefault(v, []).append(key)
        self._pairs.setdefault((u, v), []).append(key)
        return key

    def edge(self, eid: EdgeId) -> RoadEdge:
        """Return the :class:`RoadEdge` for ``eid`` (raises if absent)."""
        try:
            return self._edges[eid]
        except KeyError:
            raise StreetNetworkError("unknown edge id %r." % (eid,))

    @property
    def edges(self) -> List[EdgeId]:
        """Edge identifiers in insertion order."""
        return list(self._edges)

    @property
    def num_edges(self) -> int:
        """Number of directed edges."""
        return len(self._edges)

    def iter_edges(self) -> Iterator[RoadEdge]:
        """Iterate over :class:`RoadEdge` objects in insertion order."""
        return iter(self._edges.values())

    def out_edges(self, node: NodeId) -> List[EdgeId]:
        """Outgoing edge ids of ``node``."""
        return self._out.get(node, [])

    def in_edges(self, node: NodeId) -> List[EdgeId]:
        """Incoming edge ids of ``node``."""
        return self._in.get(node, [])

    def edges_between(self, u: NodeId, v: NodeId) -> List[EdgeId]:
        """All parallel edge ids from ``u`` to ``v``."""
        return list(self._pairs.get((u, v), []))

    def successors(self, node: NodeId) -> List[NodeId]:
        """Distinct heads of the outgoing edges (insertion ordered)."""
        seen: List[NodeId] = []
        mark: Set[NodeId] = set()
        for e in self.out_edges(node):
            w = self._edges[e].v
            if w not in mark:
                mark.add(w)
                seen.append(w)
        return seen

    def predecessors(self, node: NodeId) -> List[NodeId]:
        """Distinct tails of the incoming edges (insertion ordered)."""
        seen: List[NodeId] = []
        mark: Set[NodeId] = set()
        for e in self.in_edges(node):
            w = self._edges[e].u
            if w not in mark:
                mark.add(w)
                seen.append(w)
        return seen

    def neighbours(self, node: NodeId) -> List[NodeId]:
        """Distinct undirected neighbours (excluding the node itself)."""
        out: List[NodeId] = []
        mark: Set[NodeId] = set()
        for w in itertools.chain(self.successors(node), self.predecessors(node)):
            if w != node and w not in mark:
                mark.add(w)
                out.append(w)
        return out

    def degree(self, node: NodeId) -> int:
        """Street degree: number of distinct undirected neighbours."""
        return len(self.neighbours(node))

    def out_degree(self, node: NodeId) -> int:
        """Number of outgoing directed edges."""
        return len(self.out_edges(node))

    def in_degree(self, node: NodeId) -> int:
        """Number of incoming directed edges."""
        return len(self.in_edges(node))

    def remove_edge(self, eid: EdgeId) -> None:
        """Delete a directed edge and every index and restriction referring to it."""
        edge = self.edge(eid)
        self._out[edge.u].remove(eid)
        self._in[edge.v].remove(eid)
        bucket = self._pairs.get((edge.u, edge.v))
        if bucket:
            bucket.remove(eid)
            if not bucket:
                del self._pairs[(edge.u, edge.v)]
        twin = edge.reverse_eid
        if twin is not None and twin in self._edges:
            self._edges[twin].reverse_eid = None
        del self._edges[eid]
        self._no_turn = set((a, b) for (a, b) in self._no_turn
                            if a != eid and b != eid)
        self._only_turn.pop(eid, None)
        for a in list(self._only_turn):
            self._only_turn[a].discard(eid)
            if not self._only_turn[a]:
                del self._only_turn[a]
        self._restrictions = [r for r in self._restrictions
                              if r.from_eid != eid and r.to_eid != eid]

    def remove_node(self, node: NodeId) -> None:
        """Delete a junction together with all of its incident edges."""
        _require(node in self._nodes, "unknown node %r." % (node,))
        for eid in list(self._out.get(node, [])) + list(self._in.get(node, [])):
            if eid in self._edges:
                self.remove_edge(eid)
        self._out.pop(node, None)
        self._in.pop(node, None)
        del self._nodes[node]

    # -- turn restrictions ------------------------------------------------
    def add_turn_restriction(self, from_eid: EdgeId, via_node: NodeId,
                             to_eid: EdgeId, kind: str = "no_turn",
                             description: str = "") -> TurnRestriction:
        """Ban (``no_turn``) or mandate (``only_turn``) a turn between two edges."""
        _require(kind in ("no_turn", "only_turn"),
                 "kind must be 'no_turn' or 'only_turn' (got %r)." % (kind,))
        a = self.edge(from_eid)
        b = self.edge(to_eid)
        _require(a.v == via_node and b.u == via_node,
                 "turn restriction is not incident on via node %r." % (via_node,))
        restriction = TurnRestriction(from_eid, via_node, to_eid, kind,
                                      str(description))
        if kind == "no_turn":
            self._no_turn.add((from_eid, to_eid))
        else:
            self._only_turn.setdefault(from_eid, set()).add(to_eid)
        self._restrictions.append(restriction)
        return restriction

    @property
    def turn_restrictions(self) -> List[TurnRestriction]:
        """All registered turn restrictions in insertion order."""
        return list(self._restrictions)

    def is_turn_allowed(self, from_eid: EdgeId, to_eid: EdgeId, *,
                        allow_u_turn: bool = False) -> bool:
        """True when a vehicle may leave ``from_eid`` onto ``to_eid``."""
        a = self.edge(from_eid)
        b = self.edge(to_eid)
        if a.v != b.u:
            return False
        if not allow_u_turn and (b.eid == a.reverse_eid
                                 or (b.u == a.v and b.v == a.u)):
            return False
        if (from_eid, to_eid) in self._no_turn:
            return False
        only = self._only_turn.get(from_eid)
        if only is not None and to_eid not in only:
            return False
        return True

    def turn_deflection_deg(self, from_eid: EdgeId, to_eid: EdgeId) -> float:
        """Signed deflection in degrees for the turn from one edge to the next."""
        a = self.edge(from_eid)
        b = self.edge(to_eid)
        return bearing_difference_deg(a.end_bearing(), b.start_bearing())

    # -- geometry / summary ----------------------------------------------
    def straight_line_distance(self, u: NodeId, v: NodeId) -> float:
        """Great-circle distance when both nodes are geographic, else euclidean."""
        a, b = self.node(u), self.node(v)
        if a.is_geographic and b.is_geographic:
            return haversine_m(a.lat, a.lon, b.lat, b.lon)
        return euclidean_m(a.x, a.y, b.x, b.y)

    @property
    def total_length_m(self) -> float:
        """Sum of the lengths of every directed edge."""
        return math.fsum(e.length_m for e in self._edges.values())

    @property
    def street_length_m(self) -> float:
        """Sum of undirected street-segment lengths (reverse twins counted once)."""
        return math.fsum(self._edges[e].length_m
                         for (_, _, e) in self.undirected_segments())

    def bbox(self) -> Tuple[float, float, float, float]:
        """Planar bounding box ``(min_x, min_y, max_x, max_y)``."""
        if not self._nodes:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [n.x for n in self._nodes.values()]
        ys = [n.y for n in self._nodes.values()]
        return (min(xs), min(ys), max(xs), max(ys))

    def area_m2(self) -> float:
        """Area of the convex hull of the junctions, in square metres."""
        hull = convex_hull([n.xy for n in self._nodes.values()])
        return polygon_area(hull)

    def undirected_segments(self) -> List[Tuple[NodeId, NodeId, EdgeId]]:
        """One representative ``(u, v, eid)`` per undirected street segment."""
        out: List[Tuple[NodeId, NodeId, EdgeId]] = []
        skip: Set[EdgeId] = set()
        for eid, e in self._edges.items():
            if eid in skip:
                continue
            if e.reverse_eid is not None and e.reverse_eid in self._edges:
                skip.add(e.reverse_eid)
            out.append((e.u, e.v, eid))
        return out

    def copy(self) -> "RoadGraph":
        """Deep-enough copy: new containers, identical ids, attributes and turns."""
        g = RoadGraph(self.name, self.drive_side)
        g._geo_ref = self._geo_ref
        for n in self._nodes.values():
            g._nodes[n.node_id] = RoadNode(n.node_id, n.x, n.y, n.lat, n.lon,
                                           n.elevation_m, n.signalised,
                                           dict(n.tags))
            g._out[n.node_id] = []
            g._in[n.node_id] = []
        for e in self._edges.values():
            g._install(e.eid, e.u, e.v, e.length_m, e.lanes, e.capacity_veh_h,
                       e.free_flow_time_s, e.speed_limit_kph, e.street_name,
                       e.highway_class, e.gradient, e.modes_permitted,
                       e.geometry, e.oneway, dict(e.tags))
            g._edges[e.eid].reverse_eid = e.reverse_eid
        g._no_turn = set(self._no_turn)
        g._only_turn = dict((k, set(v)) for k, v in self._only_turn.items())
        g._restrictions = list(self._restrictions)
        g._next_eid = self._next_eid
        return g

    def subgraph(self, nodes: Iterable[NodeId]) -> "RoadGraph":
        """Induced subgraph on ``nodes``, preserving ids, attributes and turns."""
        keep = set(nodes)
        g = RoadGraph(self.name, self.drive_side)
        g._geo_ref = self._geo_ref
        for nid in self._nodes:
            if nid in keep:
                n = self._nodes[nid]
                g._nodes[nid] = RoadNode(nid, n.x, n.y, n.lat, n.lon,
                                         n.elevation_m, n.signalised, dict(n.tags))
                g._out[nid] = []
                g._in[nid] = []
        for e in self._edges.values():
            if e.u in keep and e.v in keep:
                g._install(e.eid, e.u, e.v, e.length_m, e.lanes, e.capacity_veh_h,
                           e.free_flow_time_s, e.speed_limit_kph, e.street_name,
                           e.highway_class, e.gradient, e.modes_permitted,
                           e.geometry, e.oneway, dict(e.tags))
        for e in g._edges.values():
            twin = self._edges[e.eid].reverse_eid
            e.reverse_eid = twin if (twin is not None and twin in g._edges) else None
        g._no_turn = set((a, b) for (a, b) in self._no_turn
                         if a in g._edges and b in g._edges)
        only: Dict[EdgeId, Set[EdgeId]] = {}
        for a, bs in self._only_turn.items():
            if a in g._edges:
                kept = set(b for b in bs if b in g._edges)
                if kept:
                    only[a] = kept
        g._only_turn = only
        g._restrictions = [r for r in self._restrictions
                           if r.from_eid in g._edges and r.to_eid in g._edges]
        g._next_eid = self._next_eid
        return g

    def summary(self) -> Dict[str, Any]:
        """Compact dictionary of counts and totals for printing or payloads."""
        degs = [self.degree(n) for n in self._nodes]
        return {
            "name": self.name,
            "nodes": self.num_nodes,
            "edges": self.num_edges,
            "segments": len(self.undirected_segments()),
            "oneway_edges": sum(1 for e in self._edges.values() if e.oneway),
            "self_loops": sum(1 for e in self._edges.values() if e.is_loop),
            "turn_restrictions": len(self._restrictions),
            "total_length_m": self.total_length_m,
            "street_length_m": self.street_length_m,
            "mean_degree": (math.fsum(degs) / len(degs)) if degs else 0.0,
        }

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    def __repr__(self) -> str:                   # pragma: no cover - cosmetic
        return ("RoadGraph(name=%r, nodes=%d, edges=%d)"
                % (self.name, self.num_nodes, self.num_edges))


# ===========================================================================
# 3. TOPOLOGY CLEANING -- loops, parallels, degree-2 chain simplification
# ===========================================================================
def remove_self_loops(graph: RoadGraph) -> RoadGraph:
    """Return a copy with every ``u == v`` edge deleted (ids otherwise preserved)."""
    out = graph.copy()
    for eid in [e.eid for e in graph.iter_edges() if e.is_loop]:
        out.remove_edge(eid)
    return out


def consolidate_parallel_edges(graph: RoadGraph, keep: str = "shortest") -> RoadGraph:
    """Return a copy keeping one edge per ordered ``(u, v)`` pair (``shortest`` or
    ``fastest``); the survivor records the discarded ids in ``tags['parallel']``."""
    _require(keep in ("shortest", "fastest"),
             "keep must be 'shortest' or 'fastest' (got %r)." % (keep,))
    wfn = _weight_fn("length_m" if keep == "shortest" else "free_flow_time_s")
    out = graph.copy()
    for (u, v) in [e for e in graph._pairs]:
        bucket = [e for e in graph.edges_between(u, v) if e in out._edges]
        if len(bucket) <= 1:
            continue
        best = min(bucket, key=lambda e: (wfn(graph.edge(e)), e))
        out.edge(best).tags["parallel"] = [e for e in bucket if e != best]
        for eid in bucket:
            if eid != best:
                out.remove_edge(eid)
    return out


def _chain_mergeable(graph: RoadGraph, node: NodeId,
                     protect: Set[NodeId]) -> bool:
    """True when ``node`` is an interior vertex of a homogeneous degree-2 chain."""
    if node in protect:
        return False
    nbrs = graph.neighbours(node)
    if len(nbrs) != 2:
        return False
    ins = [graph.edge(e) for e in graph.in_edges(node)]
    outs = [graph.edge(e) for e in graph.out_edges(node)]
    incident = ins + outs
    if not incident or any(e.is_loop for e in incident):
        return False
    first = incident[0]
    for e in incident[1:]:
        if (e.street_name != first.street_name
                or e.highway_class != first.highway_class
                or e.lanes != first.lanes
                or e.oneway != first.oneway):
            return False
    a, b = nbrs
    if first.oneway:
        return (len(ins) == 1 and len(outs) == 1 and ins[0].u != outs[0].v)
    return (len(ins) == 2 and len(outs) == 2
            and set(e.u for e in ins) == set([a, b])
            and set(e.v for e in outs) == set([a, b]))


def _copy_node_into(graph: RoadGraph, out: RoadGraph, nid: NodeId) -> None:
    """Copy a single junction (coordinates, elevation, signal state, tags) across."""
    n = graph.node(nid)
    out.add_node(nid, n.x, n.y, lat=n.lat, lon=n.lon,
                 elevation_m=n.elevation_m, signalised=n.signalised,
                 **dict(n.tags))


def _walk_chain(graph: RoadGraph, start_edge: EdgeId,
                removable: Set[NodeId]) -> List[EdgeId]:
    """Follow a degree-2 chain forwards from ``start_edge`` until a kept vertex."""
    chain = [start_edge]
    seen = set(chain)
    cur = graph.edge(start_edge)
    while cur.v in removable:
        nxt = [x for x in graph.out_edges(cur.v)
               if x not in seen
               and (graph.edge(x).v != cur.u or graph.edge(x).oneway)]
        if not nxt:
            break
        cur = graph.edge(nxt[0])
        chain.append(cur.eid)
        seen.add(cur.eid)
    return chain


def _emit_chain(graph: RoadGraph, out: RoadGraph, chain: List[EdgeId]) -> None:
    """Write a walked chain into ``out`` as one edge, conserving length and geometry."""
    if len(chain) == 1:
        e = graph.edge(chain[0])
        out.add_edge(e.u, e.v, e.length_m, oneway=True, lanes=e.lanes,
                     capacity_veh_h=e.capacity_veh_h,
                     speed_limit_kph=e.speed_limit_kph,
                     free_flow_time_s=e.free_flow_time_s,
                     street_name=e.street_name,
                     highway_class=e.highway_class, gradient=e.gradient,
                     modes_permitted=e.modes_permitted,
                     geometry=e.geometry, eid=e.eid, **dict(e.tags))
        return
    edges = [graph.edge(x) for x in chain]
    geom: List[Coord] = [edges[0].geometry[0]]
    for e in edges:
        geom.extend(list(e.geometry[1:]))
    total_len = math.fsum(e.length_m for e in edges)
    total_time = math.fsum(e.free_flow_time_s for e in edges)
    rise = math.fsum(e.gradient * e.length_m for e in edges)
    tags = dict(edges[0].tags)
    tags["merged_eids"] = [e.eid for e in edges]
    out.add_edge(edges[0].u, edges[-1].v, total_len, oneway=True,
                 lanes=edges[0].lanes,
                 capacity_veh_h=min(e.capacity_veh_h for e in edges),
                 speed_limit_kph=min(e.speed_limit_kph for e in edges),
                 free_flow_time_s=total_time,
                 street_name=edges[0].street_name,
                 highway_class=edges[0].highway_class,
                 gradient=(rise / total_len) if total_len > 0.0 else 0.0,
                 modes_permitted=edges[0].modes_permitted,
                 geometry=geom, **tags)


def simplify_degree2_chains(graph: RoadGraph) -> RoadGraph:
    """Collapse interior degree-2 vertices, preserving geometry and cumulative
    length; merged edges record their constituents in ``tags['merged_eids']``."""
    protect: Set[NodeId] = set()
    for r in graph.turn_restrictions:
        protect.add(r.via_node)
        protect.add(graph.edge(r.from_eid).u)
        protect.add(graph.edge(r.to_eid).v)
    removable = set(n for n in graph.nodes if _chain_mergeable(graph, n, protect))
    out = RoadGraph(graph.name, graph.drive_side)
    out._geo_ref = graph._geo_ref
    for n in graph.iter_nodes():
        if n.node_id not in removable:
            _copy_node_into(graph, out, n.node_id)
    out._next_eid = graph._next_eid              # keep minted ids clear of kept ids
    emitted: Set[EdgeId] = set()
    for start_edge in graph.edges:
        if start_edge in emitted or graph.edge(start_edge).u in removable:
            continue
        chain = _walk_chain(graph, start_edge, removable)
        emitted.update(chain)
        _emit_chain(graph, out, chain)
    # Pure degree-2 rings have no kept vertex: anchor each at its first edge's tail.
    leftover = [e for e in graph.edges if e not in emitted]
    while leftover:
        anchor = graph.edge(leftover[0]).u
        removable.discard(anchor)
        if not out.has_node(anchor):
            _copy_node_into(graph, out, anchor)
        for start_edge in list(leftover):
            if start_edge in emitted or graph.edge(start_edge).u in removable:
                continue
            chain = _walk_chain(graph, start_edge, removable)
            emitted.update(chain)
            _emit_chain(graph, out, chain)
        leftover = [e for e in leftover if e not in emitted]
    # Re-link reverse twins (mirrored endpoints, equal length).
    by_pair: Dict[Tuple[NodeId, NodeId], List[EdgeId]] = {}
    for e in out.iter_edges():
        by_pair.setdefault((e.u, e.v), []).append(e.eid)
    for e in out.iter_edges():
        if e.reverse_eid is not None:
            continue
        for cand in by_pair.get((e.v, e.u), []):
            other = out.edge(cand)
            if (other.reverse_eid is None and other.eid != e.eid
                    and abs(other.length_m - e.length_m) <= 1e-9):
                e.reverse_eid = cand
                other.reverse_eid = e.eid
                e.oneway = other.oneway = False
                break
    for r in graph.turn_restrictions:
        if r.from_eid in out._edges and r.to_eid in out._edges:
            out.add_turn_restriction(r.from_eid, r.via_node, r.to_eid, r.kind,
                                     r.description)
    return out


def largest_component_subgraph(graph: RoadGraph, strong: bool = True) -> RoadGraph:
    """Induced subgraph on the largest (strongly, by default) connected component."""
    comps = (strongly_connected_components(graph) if strong
             else weakly_connected_components(graph))
    if not comps:
        return graph.copy()
    best = max(comps, key=len)
    return graph.subgraph(best)


def clean_network(graph: RoadGraph, *, drop_loops: bool = True,
                  consolidate: bool = True, simplify: bool = True,
                  largest_component: bool = False) -> RoadGraph:
    """Run the standard cleaning pipeline and return the tidied graph."""
    out = graph
    if drop_loops:
        out = remove_self_loops(out)
    if consolidate:
        out = consolidate_parallel_edges(out)
    if simplify:
        out = simplify_degree2_chains(out)
    if largest_component:
        out = largest_component_subgraph(out, strong=True)
    return out


# ===========================================================================
# 4. TURNS -- cost model, edge-based (dual) expansion, turn-aware routing
# ===========================================================================
@dataclass
class TurnCostModel:
    """Penalties for straight / left / right / u turns plus a signal delay.

    The penalties are expressed in the *units of the routing weight*: pair the
    defaults (seconds) with ``weight="free_flow_time_s"``.

    ``drive_side`` is a LABEL recording which side of the road the model was built for.
    It is deliberately inert in :meth:`penalty` and :meth:`classify`: left versus right is
    decided by the SIGN of the deflection, which is pure geometry and the same in every
    country, and the cost asymmetry is carried by the VALUES of ``left_s`` / ``right_s``.
    That is exactly how :meth:`for_drive_side` encodes it -- ``left`` gives 2.0/7.0 and
    ``right`` gives 7.0/2.0 -- so making :meth:`penalty` consult ``drive_side`` as well
    would apply the swap twice for every model built that way.  Construct the class
    directly and you choose the values yourself; ``drive_side`` then only records intent.
    """

    straight_s: float = 0.0
    left_s: float = 2.0
    right_s: float = 7.0
    u_turn_s: float = 25.0
    signal_s: float = 8.0
    straight_threshold_deg: float = 30.0
    u_turn_threshold_deg: float = 150.0
    drive_side: str = "left"

    def __post_init__(self) -> None:
        for nm in ("straight_s", "left_s", "right_s", "u_turn_s", "signal_s"):
            _as_non_negative(getattr(self, nm), nm)
        _require(0.0 <= self.straight_threshold_deg < self.u_turn_threshold_deg <= 180.0,
                 "need 0 <= straight_threshold_deg < u_turn_threshold_deg <= 180.")
        _require(self.drive_side in ("left", "right"),
                 "drive_side must be 'left' or 'right'.")

    @classmethod
    def for_drive_side(cls, side: str, **kwargs: Any) -> "TurnCostModel":
        """Build a model with the cheap turn on the kerb side of the given country."""
        _require(side in ("left", "right"),
                 "side must be 'left' or 'right' (got %r)." % (side,))
        base = dict(left_s=2.0, right_s=7.0) if side == "left" \
            else dict(left_s=7.0, right_s=2.0)
        base.update(kwargs)
        base["drive_side"] = side
        return cls(**base)

    @classmethod
    def free(cls) -> "TurnCostModel":
        """A model with no penalties at all (turn restrictions still apply)."""
        return cls(0.0, 0.0, 0.0, 0.0, 0.0)

    def penalty(self, deflection_deg: float, signalised: bool = False) -> float:
        """Penalty for a turn of the given signed deflection at a junction."""
        d = bearing_difference_deg(0.0, _as_float(deflection_deg, "deflection_deg"))
        a = abs(d)
        if a >= self.u_turn_threshold_deg:
            base = self.u_turn_s
        elif a <= self.straight_threshold_deg:
            base = self.straight_s
        elif d < 0.0:
            base = self.left_s
        else:
            base = self.right_s
        return base + (self.signal_s if signalised else 0.0)

    def classify(self, deflection_deg: float) -> str:
        """Name the manoeuvre: ``straight``, ``left``, ``right`` or ``u_turn``."""
        d = bearing_difference_deg(0.0, _as_float(deflection_deg, "deflection_deg"))
        a = abs(d)
        if a >= self.u_turn_threshold_deg:
            return "u_turn"
        if a <= self.straight_threshold_deg:
            return "straight"
        return "left" if d < 0.0 else "right"


@dataclass
class ExpandedGraph:
    """The edge-based dual (line graph): road edges become nodes, turns arcs."""

    edge_ids: List[EdgeId]
    adjacency: Dict[EdgeId, List[Tuple[EdgeId, float]]]
    entry_cost: Dict[EdgeId, float]
    allow_u_turn: bool = False
    banned_turns: int = 0

    @property
    def num_nodes(self) -> int:
        """Number of dual nodes (= routable road edges)."""
        return len(self.edge_ids)

    @property
    def num_arcs(self) -> int:
        """Number of dual arcs (= permitted turns)."""
        return sum(len(v) for v in self.adjacency.values())

    def successors(self, eid: EdgeId) -> List[Tuple[EdgeId, float]]:
        """Permitted turns out of ``eid`` as ``(next_eid, arc_cost)`` pairs."""
        return self.adjacency.get(eid, [])


def edge_expanded_graph(graph: RoadGraph, *, weight: CostSpec = "length_m",
                        turn_cost: Optional[TurnCostModel] = None,
                        allow_u_turn: bool = False,
                        mode: Optional[str] = None) -> ExpandedGraph:
    """Build the edge-based expansion so banned and penalised turns become exact."""
    wfn = _weight_fn(weight)
    ids = [e.eid for e in graph.iter_edges() if e.permits(mode)]
    allowed = set(ids)
    adjacency: Dict[EdgeId, List[Tuple[EdgeId, float]]] = dict((e, []) for e in ids)
    entry: Dict[EdgeId, float] = {}
    banned = 0
    for eid in ids:
        e = graph.edge(eid)
        c = _as_non_negative(wfn(e), "weight(edge %d)" % eid)
        entry[eid] = c
    for eid in ids:
        e = graph.edge(eid)
        signal = graph.node(e.v).signalised
        for nxt in graph.out_edges(e.v):
            if nxt not in allowed:
                continue
            if not graph.is_turn_allowed(eid, nxt, allow_u_turn=allow_u_turn):
                banned += 1
                continue
            pen = 0.0
            if turn_cost is not None:
                pen = turn_cost.penalty(graph.turn_deflection_deg(eid, nxt), signal)
            adjacency[eid].append((nxt, entry[nxt] + pen))
    return ExpandedGraph(ids, adjacency, entry, allow_u_turn, banned)


def turn_aware_shortest_path(graph: RoadGraph, source: NodeId, target: NodeId, *,
                             weight: CostSpec = "length_m",
                             turn_cost: Optional[TurnCostModel] = None,
                             allow_u_turn: bool = False,
                             mode: Optional[str] = None,
                             expanded: Optional[ExpandedGraph] = None
                             ) -> Optional["Path"]:
    """Shortest path that honours turn restrictions and turn penalties exactly."""
    _require(graph.has_node(source), "unknown source %r." % (source,))
    _require(graph.has_node(target), "unknown target %r." % (target,))
    wfn = _weight_fn(weight)
    if source == target:
        return _make_path(graph, [source], [], wfn, 0.0)
    exp = expanded if expanded is not None else edge_expanded_graph(
        graph, weight=weight, turn_cost=turn_cost, allow_u_turn=allow_u_turn,
        mode=mode)
    dist: Dict[EdgeId, float] = {}
    parent: Dict[EdgeId, Optional[EdgeId]] = {}
    counter = itertools.count()
    heap: List[Tuple[float, int, EdgeId]] = []
    for eid in graph.out_edges(source):
        if eid not in exp.entry_cost:
            continue
        c = exp.entry_cost[eid]
        if eid not in dist or c < dist[eid]:
            dist[eid] = c
            parent[eid] = None
            heapq.heappush(heap, (c, next(counter), eid))
    settled: Set[EdgeId] = set()
    best_eid: Optional[EdgeId] = None
    best_cost = float("inf")
    while heap:
        d, _, eid = heapq.heappop(heap)
        if eid in settled:
            continue
        settled.add(eid)
        if graph.edge(eid).v == target:
            best_eid, best_cost = eid, d
            break
        for nxt, arc in exp.successors(eid):
            nd = d + arc
            if nxt not in dist or nd < dist[nxt]:
                dist[nxt] = nd
                parent[nxt] = eid
                heapq.heappush(heap, (nd, next(counter), nxt))
    if best_eid is None:
        return None
    edges: List[EdgeId] = []
    cur: Optional[EdgeId] = best_eid
    while cur is not None:
        edges.append(cur)
        cur = parent[cur]
    edges.reverse()
    nodes = [source] + [graph.edge(e).v for e in edges]
    return _make_path(graph, nodes, edges, wfn, best_cost)


# ===========================================================================
# 5. SHORTEST PATHS -- Dijkstra, A*, bidirectional Dijkstra, metric closure
# ===========================================================================
def _weight_fn(weight: CostSpec) -> Callable[[RoadEdge], float]:
    """Resolve a cost specification into a callable ``RoadEdge -> float``."""
    if callable(weight):
        return weight
    name = str(weight)
    if name in ("length_m", "length", "distance"):
        return lambda e: e.length_m
    if name in ("free_flow_time_s", "time", "travel_time"):
        return lambda e: e.free_flow_time_s
    if name in ("hops", "unit", "steps"):
        return lambda e: 1.0

    def _attr(e: RoadEdge) -> float:
        if not hasattr(e, name):
            raise StreetNetworkError("unknown edge weight %r." % (name,))
        return float(getattr(e, name))
    return _attr


@dataclass
class Path:
    """A routed path: the node sequence, the edge sequence and its costs."""

    nodes: List[NodeId]
    edges: List[EdgeId]
    cost: float
    length_m: float
    travel_time_s: float
    geometry: Tuple[Coord, ...] = ()

    @property
    def num_edges(self) -> int:
        """Number of edges traversed."""
        return len(self.edges)

    @property
    def origin(self) -> NodeId:
        """First node of the path."""
        return self.nodes[0]

    @property
    def destination(self) -> NodeId:
        """Last node of the path."""
        return self.nodes[-1]

    def is_loopless(self) -> bool:
        """True when no node is visited twice."""
        return len(set(self.nodes)) == len(self.nodes)

    def key(self) -> Tuple[Any, ...]:
        """Hashable identity used to deduplicate paths."""
        return tuple(self.edges)

    def __len__(self) -> int:
        return len(self.edges)


@dataclass
class SearchTree:
    """The settled part of a Dijkstra run: distances, parent edges and order."""

    source: NodeId
    dist: Dict[NodeId, float]
    parent_edge: Dict[NodeId, Optional[EdgeId]]
    order: List[NodeId]
    reverse: bool = False

    @property
    def reached(self) -> List[NodeId]:
        """Settled nodes in settlement order."""
        return list(self.order)

    def distance_to(self, node: NodeId) -> float:
        """Settled distance to ``node`` (``inf`` when it was never settled)."""
        return self.dist.get(node, float("inf"))


def _path_geometry(graph: RoadGraph, edges: Sequence[EdgeId],
                   nodes: Sequence[NodeId]) -> Tuple[Coord, ...]:
    """Concatenate the edge geometries of a path into one polyline."""
    if not edges:
        return (graph.node(nodes[0]).xy,) if nodes else ()
    geom: List[Coord] = [graph.edge(edges[0]).geometry[0]]
    for eid in edges:
        geom.extend(list(graph.edge(eid).geometry[1:]))
    return tuple(geom)


def _make_path(graph: RoadGraph, nodes: Sequence[NodeId], edges: Sequence[EdgeId],
               wfn: Callable[[RoadEdge], float],
               cost: Optional[float] = None) -> Path:
    """Assemble a :class:`Path`, re-accumulating the cost left-to-right."""
    if cost is None:
        acc = 0.0
        for eid in edges:
            acc += wfn(graph.edge(eid))
        cost = acc
    length = math.fsum(graph.edge(e).length_m for e in edges)
    time_s = math.fsum(graph.edge(e).free_flow_time_s for e in edges)
    return Path(list(nodes), list(edges), cost, length, time_s,
                _path_geometry(graph, edges, nodes))


def _extract(graph: RoadGraph, tree: SearchTree,
             node: NodeId) -> Optional[Tuple[List[NodeId], List[EdgeId]]]:
    """Walk parent pointers back to the tree root; None when unreachable."""
    if node not in tree.parent_edge:
        return None
    edges: List[EdgeId] = []
    cur = node
    guard = 0
    limit = graph.num_edges + 2
    while True:
        pe = tree.parent_edge.get(cur)
        if pe is None:
            break
        edges.append(pe)
        cur = graph.edge(pe).v if tree.reverse else graph.edge(pe).u
        guard += 1
        if guard > limit:                        # pragma: no cover - safety net
            raise StreetNetworkError("cycle detected while extracting a path.")
    if cur != tree.source:
        return None
    if tree.reverse:
        nodes = [node] + [graph.edge(e).v for e in edges]
    else:
        edges.reverse()
        nodes = [tree.source] + [graph.edge(e).v for e in edges]
    return nodes, edges


def dijkstra(graph: RoadGraph, source: NodeId, target: Optional[NodeId] = None, *,
             weight: CostSpec = "length_m", mode: Optional[str] = None,
             max_cost: float = float("inf"),
             banned_edges: Optional[Set[EdgeId]] = None,
             banned_nodes: Optional[Set[NodeId]] = None,
             reverse: bool = False) -> SearchTree:
    """Label-setting Dijkstra; ``reverse=True`` searches the incoming edges instead."""
    _require(graph.has_node(source), "unknown source %r." % (source,))
    wfn = _weight_fn(weight)
    dist: Dict[NodeId, float] = {source: 0.0}
    parent: Dict[NodeId, Optional[EdgeId]] = {source: None}
    settled: Dict[NodeId, float] = {}
    settled_parent: Dict[NodeId, Optional[EdgeId]] = {}
    order: List[NodeId] = []
    counter = itertools.count()
    heap: List[Tuple[float, int, NodeId]] = [(0.0, next(counter), source)]
    be = banned_edges or set()
    bn = banned_nodes or set()
    while heap:
        d, _, u = heapq.heappop(heap)
        if u in settled:
            continue
        settled[u] = d
        settled_parent[u] = parent.get(u)
        order.append(u)
        if target is not None and u == target:
            break
        for eid in (graph.in_edges(u) if reverse else graph.out_edges(u)):
            if eid in be:
                continue
            e = graph.edge(eid)
            v = e.u if reverse else e.v
            if v in settled or v in bn:
                continue
            if not e.permits(mode):
                continue
            w = wfn(e)
            if w < 0.0:
                raise StreetNetworkError("negative edge weight on edge %d." % eid)
            nd = d + w
            if nd > max_cost:
                continue
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                parent[v] = eid
                heapq.heappush(heap, (nd, next(counter), v))
    return SearchTree(source, settled, settled_parent, order, reverse)


def shortest_path(graph: RoadGraph, source: NodeId, target: NodeId, *,
                  weight: CostSpec = "length_m", mode: Optional[str] = None,
                  banned_edges: Optional[Set[EdgeId]] = None,
                  banned_nodes: Optional[Set[NodeId]] = None) -> Optional[Path]:
    """Single-pair shortest path, or ``None`` when the target is unreachable."""
    _require(graph.has_node(target), "unknown target %r." % (target,))
    tree = dijkstra(graph, source, target, weight=weight, mode=mode,
                    banned_edges=banned_edges, banned_nodes=banned_nodes)
    found = _extract(graph, tree, target)
    if found is None:
        return None
    nodes, edges = found
    return _make_path(graph, nodes, edges, _weight_fn(weight))


def shortest_path_length(graph: RoadGraph, source: NodeId, target: NodeId, *,
                         weight: CostSpec = "length_m",
                         mode: Optional[str] = None) -> float:
    """Cost of the shortest path, or ``inf`` when there is none."""
    p = shortest_path(graph, source, target, weight=weight, mode=mode)
    return float("inf") if p is None else p.cost


def _heuristic_scale(graph: RoadGraph, weight: CostSpec) -> float:
    """Metres-to-weight conversion that keeps a straight-line heuristic admissible."""
    if callable(weight):
        return 0.0                               # unknown units: fall back to Dijkstra
    name = str(weight)
    if name in ("length_m", "length", "distance"):
        return 1.0
    if name in ("free_flow_time_s", "time", "travel_time"):
        fastest = 0.0
        for e in graph.iter_edges():
            # An edge that covers positive distance in ZERO time has unbounded speed, so
            # no finite metres-to-seconds scale can keep the straight-line heuristic
            # admissible: fall back to Dijkstra (scale 0 => h == 0), the same escape this
            # function already uses for unknown units.  RoadEdge.free_flow_speed_mps
            # reports the 32 kph class default for such an edge, which is what used to
            # make `fastest` finite and A* silently return a suboptimal path.
            if e.free_flow_time_s <= _EPS and e.length_m > _EPS:
                return 0.0
            fastest = max(fastest, e.free_flow_speed_mps)
        return 0.0 if fastest <= _EPS else 1.0 / fastest
    return 0.0


def astar(graph: RoadGraph, source: NodeId, target: NodeId, *,
          weight: CostSpec = "length_m", mode: Optional[str] = None,
          heuristic: Optional[Callable[[NodeId, NodeId], float]] = None
          ) -> Optional[Path]:
    """A* with an admissible straight-line heuristic (euclidean or great-circle).

    PRECONDITION on the default heuristic (an INHERENT limit of every Euclidean-heuristic
    A*, stated as a precondition by NetworkX's ``astar_path`` and by the textbooks, not
    validated here because an O(E) scan on every call is too high a price): every edge's
    recorded ``length_m`` must be at least the straight-line distance between its
    endpoints, and with ``weight="free_flow_time_s"`` every edge's time must be at least
    its length divided by the network's fastest speed.  Feed lengths that undercut the
    metric and the heuristic stops being a lower bound and the returned path may be
    suboptimal (fuzzing found 344 of 4708 OD pairs wrong, the worst 1419.66 against
    633.29, once ``length_m`` was drawn independently of the node coordinates -- and 0 of
    4586 once it was not).  Pass an explicit ``heuristic`` (``lambda a, b: 0.0`` reduces
    this to Dijkstra) when the stored lengths are not a metric on the coordinates.
    """
    _require(graph.has_node(source), "unknown source %r." % (source,))
    _require(graph.has_node(target), "unknown target %r." % (target,))
    wfn = _weight_fn(weight)
    if source == target:
        return _make_path(graph, [source], [], wfn, 0.0)
    if heuristic is None:
        scale = _heuristic_scale(graph, weight)
        if scale <= 0.0:
            def hfn(a: NodeId, b: NodeId) -> float:
                return 0.0
        else:
            def hfn(a: NodeId, b: NodeId) -> float:
                return graph.straight_line_distance(a, b) * scale
    else:
        hfn = heuristic
    g_score: Dict[NodeId, float] = {source: 0.0}
    parent: Dict[NodeId, Optional[EdgeId]] = {source: None}
    closed: Set[NodeId] = set()
    counter = itertools.count()
    heap: List[Tuple[float, int, NodeId]] = [(hfn(source, target), next(counter),
                                              source)]
    while heap:
        _, _, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u)
        if u == target:
            break
        gu = g_score[u]
        for eid in graph.out_edges(u):
            e = graph.edge(eid)
            if not e.permits(mode) or e.v in closed:
                continue
            nd = gu + wfn(e)
            if e.v not in g_score or nd < g_score[e.v]:
                g_score[e.v] = nd
                parent[e.v] = eid
                heapq.heappush(heap, (nd + hfn(e.v, target), next(counter), e.v))
    if target not in closed:
        return None
    tree = SearchTree(source, g_score, parent, list(closed), False)
    found = _extract(graph, tree, target)
    if found is None:
        return None
    nodes, edges = found
    return _make_path(graph, nodes, edges, wfn)


def bidirectional_dijkstra(graph: RoadGraph, source: NodeId, target: NodeId, *,
                           weight: CostSpec = "length_m",
                           mode: Optional[str] = None) -> Optional[Path]:
    """Simultaneous forward/backward Dijkstra meeting in the middle."""
    _require(graph.has_node(source), "unknown source %r." % (source,))
    _require(graph.has_node(target), "unknown target %r." % (target,))
    wfn = _weight_fn(weight)
    if source == target:
        return _make_path(graph, [source], [], wfn, 0.0)
    df: Dict[NodeId, float] = {source: 0.0}
    db: Dict[NodeId, float] = {target: 0.0}
    pf: Dict[NodeId, Optional[EdgeId]] = {source: None}
    pb: Dict[NodeId, Optional[EdgeId]] = {target: None}
    sf: Set[NodeId] = set()
    sb: Set[NodeId] = set()
    counter = itertools.count()
    hf: List[Tuple[float, int, NodeId]] = [(0.0, next(counter), source)]
    hb: List[Tuple[float, int, NodeId]] = [(0.0, next(counter), target)]
    mu = float("inf")
    meet: Optional[NodeId] = None
    while hf and hb:
        if hf[0][0] + hb[0][0] >= mu:
            break
        if hf[0][0] <= hb[0][0]:
            d, _, u = heapq.heappop(hf)
            if u in sf:
                continue
            sf.add(u)
            for eid in graph.out_edges(u):
                e = graph.edge(eid)
                if not e.permits(mode):
                    continue
                nd = d + wfn(e)
                if e.v not in df or nd < df[e.v]:
                    df[e.v] = nd
                    pf[e.v] = eid
                    heapq.heappush(hf, (nd, next(counter), e.v))
                if e.v in db and nd + db[e.v] < mu:
                    mu = nd + db[e.v]
                    meet = e.v
        else:
            d, _, u = heapq.heappop(hb)
            if u in sb:
                continue
            sb.add(u)
            for eid in graph.in_edges(u):
                e = graph.edge(eid)
                if not e.permits(mode):
                    continue
                nd = d + wfn(e)
                if e.u not in db or nd < db[e.u]:
                    db[e.u] = nd
                    pb[e.u] = eid
                    heapq.heappush(hb, (nd, next(counter), e.u))
                if e.u in df and df[e.u] + nd < mu:
                    mu = df[e.u] + nd
                    meet = e.u
    if meet is None:
        return None
    head: List[EdgeId] = []
    cur = meet
    while pf.get(cur) is not None:
        eid = pf[cur]
        head.append(eid)
        cur = graph.edge(eid).u
    head.reverse()
    tail: List[EdgeId] = []
    cur = meet
    while pb.get(cur) is not None:
        eid = pb[cur]
        tail.append(eid)
        cur = graph.edge(eid).v
    edges = head + tail
    nodes = [source] + [graph.edge(e).v for e in edges]
    return _make_path(graph, nodes, edges, wfn)


def metric_closure(graph: RoadGraph, nodes: Optional[Sequence[NodeId]] = None, *,
                   weight: CostSpec = "length_m", mode: Optional[str] = None
                   ) -> Dict[Tuple[NodeId, NodeId], float]:
    """All-pairs shortest-path costs over ``nodes`` (the full node set by default)."""
    targets = list(graph.nodes) if nodes is None else list(nodes)
    out: Dict[Tuple[NodeId, NodeId], float] = {}
    for s in targets:
        tree = dijkstra(graph, s, weight=weight, mode=mode)
        for t in targets:
            if t in tree.dist:
                out[(s, t)] = tree.dist[t]
    return out


# ===========================================================================
# 6. ALT -- A* with landmarks and the triangle inequality
# ===========================================================================
class ALTIndex:
    """Landmark distance tables giving a tight, admissible lower bound.

    For a landmark ``L`` the triangle inequality gives two valid bounds on the
    distance ``d(v, t)``::

        d(v, t) >= d(L, t) - d(L, v)        (landmark "behind" the query)
        d(v, t) >= d(v, L) - d(t, L)        (landmark "ahead" of the query)

    The lower bound is the maximum over both forms and over all landmarks, which
    is exactly what makes ALT's A* both admissible and consistent.
    """

    def __init__(self, graph: RoadGraph, num_landmarks: int = 8, *,
                 weight: CostSpec = "length_m", mode: Optional[str] = None,
                 method: str = "farthest", seed: int = 0) -> None:
        _require(method in ("farthest", "random"),
                 "method must be 'farthest' or 'random' (got %r)." % (method,))
        _require(graph.num_nodes > 0, "ALTIndex needs a non-empty graph.")
        self.graph = graph
        self.weight = weight
        self.mode = mode
        self.method = method
        self.seed = int(seed)
        k = min(int(num_landmarks), graph.num_nodes)
        _require(k >= 1, "num_landmarks must be at least 1.")
        self.landmarks: List[NodeId] = self._select(k)
        self.from_landmark: Dict[NodeId, Dict[NodeId, float]] = {}
        self.to_landmark: Dict[NodeId, Dict[NodeId, float]] = {}
        for L in self.landmarks:
            self.from_landmark[L] = dijkstra(graph, L, weight=weight,
                                             mode=mode).dist
            self.to_landmark[L] = dijkstra(graph, L, weight=weight, mode=mode,
                                           reverse=True).dist

    def _select(self, k: int) -> List[NodeId]:
        """Pick landmarks by farthest-point sampling (or seeded uniformly at random)."""
        nodes = self.graph.nodes
        if self.method == "random":
            rng = random.Random(self.seed)
            return rng.sample(nodes, k)
        rng = random.Random(self.seed)
        start = nodes[rng.randrange(len(nodes))]
        first_tree = dijkstra(self.graph, start, weight=self.weight, mode=self.mode)
        if first_tree.dist:
            start = max(sorted(first_tree.dist, key=repr),
                        key=lambda n: first_tree.dist[n])
        chosen = [start]
        best: Dict[NodeId, float] = dict(
            dijkstra(self.graph, start, weight=self.weight, mode=self.mode).dist)
        while len(chosen) < k:
            far = None
            far_d = -1.0
            for n in nodes:
                d = best.get(n, float("inf"))
                if d == float("inf"):
                    d = -0.5                     # unreachable: least attractive
                if n not in chosen and d > far_d:
                    far, far_d = n, d
            if far is None:
                break
            chosen.append(far)
            tree = dijkstra(self.graph, far, weight=self.weight, mode=self.mode)
            for n in nodes:
                dn = tree.dist.get(n, float("inf"))
                cur = best.get(n, float("inf"))
                if dn < cur:
                    best[n] = dn
        return chosen

    def lower_bound(self, v: NodeId, t: NodeId) -> float:
        """Maximum triangle-inequality lower bound on ``d(v, t)`` over all landmarks."""
        if v == t:
            return 0.0
        best = 0.0
        for L in self.landmarks:
            fl = self.from_landmark[L]
            tl = self.to_landmark[L]
            dlt = fl.get(t)
            dlv = fl.get(v)
            if dlt is not None and dlv is not None:
                cand = dlt - dlv
                if cand > best:
                    best = cand
            dvl = tl.get(v)
            dtl = tl.get(t)
            if dvl is not None and dtl is not None:
                cand = dvl - dtl
                if cand > best:
                    best = cand
        return best

    def summary(self) -> Dict[str, Any]:
        """Landmark count, method and table size for reporting."""
        return {"landmarks": [str(n) for n in self.landmarks],
                "numLandmarks": len(self.landmarks),
                "method": self.method,
                "entries": sum(len(v) for v in self.from_landmark.values())
                + sum(len(v) for v in self.to_landmark.values())}


def alt_shortest_path(graph: RoadGraph, index: ALTIndex, source: NodeId,
                      target: NodeId, *, mode: Optional[str] = None
                      ) -> Optional[Path]:
    """A* driven by the ALT landmark lower bound (exact, same cost as Dijkstra)."""
    return astar(graph, source, target, weight=index.weight,
                 mode=index.mode if mode is None else mode,
                 heuristic=index.lower_bound)


# ===========================================================================
# 7. CONTRACTION HIERARCHIES -- node ordering, witness search, shortcuts
# ===========================================================================
class ContractionHierarchy:
    """A contraction hierarchy: preprocess once, then answer queries by two
    upward searches.  Node order is lazy edge-difference priority with a
    contracted-neighbour tie-break; witness searches are hop- and cost-limited
    (a truncated witness search is conservative, never wrong)."""

    def __init__(self, graph: RoadGraph, *, weight: CostSpec = "length_m",
                 mode: Optional[str] = None, hop_limit: int = 5,
                 max_settled: int = 400) -> None:
        self.graph = graph
        self.weight = weight
        self.mode = mode
        self.hop_limit = _as_positive_int(hop_limit, "hop_limit")
        self.max_settled = _as_positive_int(max_settled, "max_settled")
        self._wfn = _weight_fn(weight)
        self._out: Dict[NodeId, Dict[NodeId, float]] = {}
        self._in: Dict[NodeId, Dict[NodeId, float]] = {}
        self._mid: Dict[Tuple[NodeId, NodeId], Optional[NodeId]] = {}
        self._base: Dict[Tuple[NodeId, NodeId], EdgeId] = {}
        self.rank: Dict[NodeId, int] = {}
        self.num_shortcuts = 0
        self._build()

    # -- preprocessing ----------------------------------------------------
    def _set_arc(self, u: NodeId, v: NodeId, w: float,
                 mid: Optional[NodeId], eid: Optional[EdgeId]) -> bool:
        """Insert or relax the arc ``u -> v``; True when the arc improved."""
        cur = self._out.get(u, {}).get(v)
        if cur is not None and cur <= w:
            return False
        self._out.setdefault(u, {})[v] = w
        self._in.setdefault(v, {})[u] = w
        self._mid[(u, v)] = mid
        if eid is not None:
            self._base[(u, v)] = eid
        return True

    def _build(self) -> None:
        """Condense to a simple digraph, order the nodes, then contract them."""
        g = self.graph
        for n in g.nodes:
            self._out.setdefault(n, {})
            self._in.setdefault(n, {})
        for e in g.iter_edges():
            if e.is_loop or not e.permits(self.mode):
                continue
            w = self._wfn(e)
            if w < 0.0:
                raise StreetNetworkError("negative edge weight on edge %d." % e.eid)
            self._set_arc(e.u, e.v, w, None, e.eid)
        contracted: Set[NodeId] = set()
        neighbours_done: Dict[NodeId, int] = dict((n, 0) for n in g.nodes)
        counter = itertools.count()
        heap: List[Tuple[float, int, NodeId]] = []
        for n in g.nodes:
            heapq.heappush(heap, (self._priority(n, contracted, neighbours_done),
                                  next(counter), n))
        while heap:
            _, _, v = heapq.heappop(heap)
            if v in contracted:
                continue
            p = self._priority(v, contracted, neighbours_done)
            if heap and p > heap[0][0]:
                heapq.heappush(heap, (p, next(counter), v))
                continue
            self.num_shortcuts += self._contract(v, contracted, apply=True)
            self.rank[v] = len(self.rank)
            contracted.add(v)
            for w in set(list(self._in.get(v, {})) + list(self._out.get(v, {}))):
                if w not in contracted:
                    neighbours_done[w] = neighbours_done.get(w, 0) + 1
        self._upward: Dict[NodeId, List[Tuple[NodeId, float]]] = {}
        self._downward: Dict[NodeId, List[Tuple[NodeId, float]]] = {}
        for u, row in self._out.items():
            for v, w in row.items():
                if self.rank.get(u, 0) < self.rank.get(v, 0):
                    self._upward.setdefault(u, []).append((v, w))
                else:
                    self._downward.setdefault(v, []).append((u, w))

    def _priority(self, v: NodeId, contracted: Set[NodeId],
                  done: Dict[NodeId, int]) -> float:
        """Edge difference plus a uniformity term; lower contracts sooner."""
        added = self._contract(v, contracted, apply=False)
        ins = sum(1 for u in self._in.get(v, {}) if u not in contracted and u != v)
        outs = sum(1 for w in self._out.get(v, {}) if w not in contracted and w != v)
        return float(added - (ins + outs)) + 2.0 * float(done.get(v, 0))

    def _contract(self, v: NodeId, contracted: Set[NodeId], apply: bool) -> int:
        """Count (and optionally insert) the shortcuts needed to remove ``v``."""
        ins = [(u, w) for u, w in self._in.get(v, {}).items()
               if u not in contracted and u != v]
        outs = [(x, w) for x, w in self._out.get(v, {}).items()
                if x not in contracted and x != v]
        if not ins or not outs:
            return 0
        added = 0
        for (u, wu) in ins:
            targets = dict((x, wu + wx) for (x, wx) in outs if x != u)
            if not targets:
                continue
            limit = max(targets.values())
            found = self._witness(u, v, contracted, limit, set(targets))
            for x, total in targets.items():
                d = found.get(x, float("inf"))
                if d <= total * (1.0 + 1e-12) + 1e-12:
                    continue
                added += 1
                if apply:
                    self._set_arc(u, x, total, v, None)
        return added

    def _witness(self, source: NodeId, avoid: NodeId, contracted: Set[NodeId],
                 limit: float, targets: Set[NodeId]) -> Dict[NodeId, float]:
        """Hop- and cost-limited Dijkstra from ``source`` skipping ``avoid``."""
        dist: Dict[NodeId, float] = {source: 0.0}
        hops: Dict[NodeId, int] = {source: 0}
        settled: Dict[NodeId, float] = {}
        counter = itertools.count()
        heap: List[Tuple[float, int, NodeId]] = [(0.0, next(counter), source)]
        remaining = set(targets)
        while heap and len(settled) < self.max_settled:
            d, _, u = heapq.heappop(heap)
            if u in settled:
                continue
            settled[u] = d
            remaining.discard(u)
            if not remaining or d > limit:
                break
            if hops.get(u, 0) >= self.hop_limit:
                continue
            for x, w in self._out.get(u, {}).items():
                if x == avoid or x in contracted or x in settled:
                    continue
                nd = d + w
                if nd > limit:
                    continue
                if x not in dist or nd < dist[x]:
                    dist[x] = nd
                    hops[x] = hops.get(u, 0) + 1
                    heapq.heappush(heap, (nd, next(counter), x))
        return settled

    # -- query ------------------------------------------------------------
    def _search(self, start: NodeId, upward: bool
                ) -> Tuple[Dict[NodeId, float], Dict[NodeId, Optional[NodeId]]]:
        """One upward Dijkstra over the ``G^`` (forward) or ``Gv`` (backward) graph."""
        adj = self._upward if upward else self._downward
        dist: Dict[NodeId, float] = {start: 0.0}
        parent: Dict[NodeId, Optional[NodeId]] = {start: None}
        settled: Set[NodeId] = set()
        counter = itertools.count()
        heap: List[Tuple[float, int, NodeId]] = [(0.0, next(counter), start)]
        while heap:
            d, _, u = heapq.heappop(heap)
            if u in settled:
                continue
            settled.add(u)
            for v, w in adj.get(u, []):
                nd = d + w
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(heap, (nd, next(counter), v))
        return dist, parent

    def _unpack(self, u: NodeId, v: NodeId, out: List[EdgeId]) -> None:
        """Recursively expand a (possibly shortcut) arc into original edge ids."""
        mid = self._mid.get((u, v))
        if mid is None:
            eid = self._base.get((u, v))
            if eid is None:                      # pragma: no cover - safety net
                raise StreetNetworkError("missing base edge for arc %r -> %r."
                                         % (u, v))
            out.append(eid)
            return
        self._unpack(u, mid, out)
        self._unpack(mid, v, out)

    def shortest_path(self, source: NodeId, target: NodeId) -> Optional[Path]:
        """Bidirectional CH query; the returned cost is re-summed over real edges."""
        _require(self.graph.has_node(source), "unknown source %r." % (source,))
        _require(self.graph.has_node(target), "unknown target %r." % (target,))
        if source == target:
            return _make_path(self.graph, [source], [], self._wfn, 0.0)
        df, pf = self._search(source, True)
        db, pb = self._search(target, False)
        meet = None
        best = float("inf")
        for n in df:
            if n in db:
                tot = df[n] + db[n]
                if tot < best:
                    best, meet = tot, n
        if meet is None:
            return None
        up: List[Tuple[NodeId, NodeId]] = []
        cur = meet
        while pf.get(cur) is not None:
            up.append((pf[cur], cur))
            cur = pf[cur]
        up.reverse()
        down: List[Tuple[NodeId, NodeId]] = []
        cur = meet
        while pb.get(cur) is not None:
            down.append((cur, pb[cur]))
            cur = pb[cur]
        edges: List[EdgeId] = []
        for (a, b) in up + down:
            self._unpack(a, b, edges)
        nodes = [source] + [self.graph.edge(e).v for e in edges]
        return _make_path(self.graph, nodes, edges, self._wfn)

    def distance(self, source: NodeId, target: NodeId) -> float:
        """Shortest-path cost by CH query (``inf`` when unreachable)."""
        p = self.shortest_path(source, target)
        return float("inf") if p is None else p.cost

    def summary(self) -> Dict[str, Any]:
        """Preprocessing statistics."""
        arcs = sum(len(r) for r in self._out.values())
        return {"nodes": len(self.rank), "arcs": arcs,
                "shortcuts": self.num_shortcuts,
                "upwardArcs": sum(len(v) for v in self._upward.values()),
                "downwardArcs": sum(len(v) for v in self._downward.values())}


# ===========================================================================
# 8. K-SHORTEST PATHS -- Yen (loopless) and Eppstein-style sidetracks
# ===========================================================================
def yen_k_shortest_paths(graph: RoadGraph, source: NodeId, target: NodeId,
                         k: int = 3, *, weight: CostSpec = "length_m",
                         mode: Optional[str] = None) -> List[Path]:
    """Yen's algorithm: up to ``k`` loopless paths of non-decreasing cost."""
    kk = _as_positive_int(k, "k")
    first = shortest_path(graph, source, target, weight=weight, mode=mode)
    if first is None:
        return []
    accepted: List[Path] = [first]
    seen: Set[Tuple[Any, ...]] = set([first.key()])
    candidates: List[Tuple[float, int, List[NodeId], List[EdgeId]]] = []
    counter = itertools.count()
    wfn = _weight_fn(weight)
    while len(accepted) < kk:
        prev = accepted[-1]
        for i in range(len(prev.edges)):
            spur_node = prev.nodes[i]
            root_nodes = prev.nodes[:i + 1]
            root_edges = prev.edges[:i]
            banned_edges: Set[EdgeId] = set()
            for p in accepted:
                if len(p.edges) > i and p.edges[:i] == root_edges \
                        and p.nodes[:i + 1] == root_nodes:
                    banned_edges.add(p.edges[i])
            banned_nodes = set(root_nodes[:-1])
            spur = shortest_path(graph, spur_node, target, weight=weight,
                                 mode=mode, banned_edges=banned_edges,
                                 banned_nodes=banned_nodes)
            if spur is None:
                continue
            nodes = root_nodes[:-1] + spur.nodes
            edges = root_edges + spur.edges
            if len(set(nodes)) != len(nodes):
                continue
            cand = _make_path(graph, nodes, edges, wfn)
            if cand.key() in seen:
                continue
            heapq.heappush(candidates, (cand.cost, next(counter), nodes, edges))
        if not candidates:
            break
        best_cost, _, nodes, edges = heapq.heappop(candidates)
        path = _make_path(graph, nodes, edges, wfn)
        if path.key() in seen:
            continue
        seen.add(path.key())
        accepted.append(path)
    return accepted


def eppstein_k_shortest_paths(graph: RoadGraph, source: NodeId, target: NodeId,
                              k: int = 3, *, weight: CostSpec = "length_m",
                              mode: Optional[str] = None) -> List[Path]:
    """Sidetrack-ordered k shortest paths (loops admitted), Eppstein-style.

    A reverse Dijkstra gives the shortest distance ``d(v)`` from every node to
    the target.  Every edge then has a non-negative *sidetrack* cost
    ``delta(e) = w(e) + d(head) - d(tail)``, and the cost of any s-t walk is
    ``d(s) + sum(delta)``.  Enumerating walks in order of accumulated sidetrack
    cost by best-first search therefore enumerates them in order of total cost.
    """
    kk = _as_positive_int(k, "k")
    _require(graph.has_node(source), "unknown source %r." % (source,))
    _require(graph.has_node(target), "unknown target %r." % (target,))
    wfn = _weight_fn(weight)
    back = dijkstra(graph, target, weight=weight, mode=mode, reverse=True)
    if source not in back.dist:
        return []
    d_to = back.dist
    counter = itertools.count()
    # state = (accumulated sidetrack cost, tie, node, parent-state index)
    parents: List[Tuple[Optional[int], Optional[EdgeId]]] = [(None, None)]
    heap: List[Tuple[float, int, NodeId, int]] = [(0.0, next(counter), source, 0)]
    expansions: Dict[NodeId, int] = {}
    out: List[Path] = []
    while heap and len(out) < kk:
        acc, _, u, sidx = heapq.heappop(heap)
        cnt = expansions.get(u, 0)
        if cnt >= kk:
            continue
        expansions[u] = cnt + 1
        if u == target:
            edges: List[EdgeId] = []
            cur: Optional[int] = sidx
            while cur is not None and parents[cur][1] is not None:
                edges.append(parents[cur][1])
                cur = parents[cur][0]
            edges.reverse()
            nodes = [source] + [graph.edge(e).v for e in edges]
            out.append(_make_path(graph, nodes, edges, wfn))
            continue
        for eid in graph.out_edges(u):
            e = graph.edge(eid)
            if not e.permits(mode) or e.v not in d_to:
                continue
            delta = wfn(e) + d_to[e.v] - d_to[u]
            if delta < 0.0:
                delta = 0.0
            parents.append((sidx, eid))
            heapq.heappush(heap, (acc + delta, next(counter), e.v,
                                  len(parents) - 1))
    return out


# ===========================================================================
# 9. TIME-DEPENDENT ROUTING -- FIFO travel-time profiles
# ===========================================================================
@dataclass
class TravelTimeProfile:
    """Piecewise-linear travel time versus departure time, optionally periodic."""

    times_s: Tuple[float, ...]
    values_s: Tuple[float, ...]
    period_s: Optional[float] = None

    def __post_init__(self) -> None:
        t = tuple(_as_float(v, "times_s") for v in self.times_s)
        y = tuple(_as_non_negative(v, "values_s") for v in self.values_s)
        _require(len(t) == len(y) and len(t) >= 1,
                 "times_s and values_s must be non-empty and the same length.")
        for i in range(len(t) - 1):
            _require(t[i + 1] > t[i], "times_s must be strictly increasing.")
        if self.period_s is not None:
            p = _as_positive(self.period_s, "period_s")
            _require(t[-1] - t[0] <= p, "the profile is longer than its period.")
            object.__setattr__(self, "period_s", p)
        object.__setattr__(self, "times_s", t)
        object.__setattr__(self, "values_s", y)

    @classmethod
    def constant(cls, value_s: float) -> "TravelTimeProfile":
        """A flat profile (equivalent to a static edge cost)."""
        return cls((0.0,), (_as_non_negative(value_s, "value_s"),), None)

    def travel_time(self, t: float) -> float:
        """Travel time for a departure at ``t`` (clamped or wrapped at the ends)."""
        ts = self.times_s
        ys = self.values_s
        if len(ts) == 1:
            return ys[0]
        tt = _as_float(t, "t")
        if self.period_s is not None:
            tt = ts[0] + (tt - ts[0]) % self.period_s
        if tt <= ts[0]:
            return ys[0]
        if tt >= ts[-1]:
            return ys[-1]
        lo, hi = 0, len(ts) - 1
        while hi - lo > 1:
            midi = (lo + hi) // 2
            if ts[midi] <= tt:
                lo = midi
            else:
                hi = midi
        span = ts[hi] - ts[lo]
        f = 0.0 if span <= 0.0 else (tt - ts[lo]) / span   # __post_init__ keeps it > 0
        return ys[lo] + f * (ys[hi] - ys[lo])

    def arrival(self, t: float) -> float:
        """Arrival time for a departure at ``t``."""
        return _as_float(t, "t") + self.travel_time(t)

    def is_fifo(self) -> bool:
        """True when no later departure can arrive earlier (non-overtaking).

        Two conditions, not one.  Inside the profile no segment may fall faster than
        ``-1`` s per s.  A PERIODIC profile also has a discontinuity that the segment
        slopes cannot see: :meth:`travel_time` holds ``values_s[-1]`` from ``times_s[-1]``
        to ``times_s[0] + period_s`` and then jumps straight back to ``values_s[0]``, so a
        departure an instant after the wrap arrives ``values_s[-1] - values_s[0]`` earlier
        than one an instant before it.  FIFO therefore also requires
        ``values_s[-1] <= values_s[0]``; without that check a profile such as
        ``((0, 3600), (100, 4000), period_s=3600)`` reported ``True`` while a departure at
        t=3599 arrived at 7597.9 and one at t=3600 arrived at 3700.0, and
        :func:`time_dependent_dijkstra` accepted it and returned a route 2x too slow.
        """
        ts = self.times_s
        ys = self.values_s
        for i in range(len(ts) - 1):
            slope = (ys[i + 1] - ys[i]) / (ts[i + 1] - ts[i])
            if slope < -1.0 - 1e-9:
                return False
        if self.period_s is not None and len(ys) > 1:
            tol = 1e-9 * max(abs(ys[0]), abs(ys[-1]))
            if ys[-1] - ys[0] > tol:
                return False
        return True

    def summary(self) -> Dict[str, Any]:
        """Minimum, maximum and mean travel time over the breakpoints."""
        return {"breakpoints": len(self.times_s),
                "minS": min(self.values_s), "maxS": max(self.values_s),
                "meanS": math.fsum(self.values_s) / len(self.values_s),
                "fifo": self.is_fifo(),
                "periodS": self.period_s}


def time_dependent_dijkstra(graph: RoadGraph, source: NodeId,
                            departure_s: float = 0.0, *,
                            profiles: Optional[Dict[EdgeId, TravelTimeProfile]] = None,
                            target: Optional[NodeId] = None,
                            mode: Optional[str] = None,
                            require_fifo: bool = True) -> SearchTree:
    """Label-setting shortest arrival times on FIFO time-varying edge costs."""
    _require(graph.has_node(source), "unknown source %r." % (source,))
    prof = profiles or {}
    if require_fifo:
        for eid, p in prof.items():
            if not p.is_fifo():
                raise StreetNetworkError(
                    "profile for edge %d violates FIFO; label-setting is invalid."
                    % eid)
    t0 = _as_float(departure_s, "departure_s")
    arrive: Dict[NodeId, float] = {source: t0}
    parent: Dict[NodeId, Optional[EdgeId]] = {source: None}
    settled: Dict[NodeId, float] = {}
    settled_parent: Dict[NodeId, Optional[EdgeId]] = {}
    order: List[NodeId] = []
    counter = itertools.count()
    heap: List[Tuple[float, int, NodeId]] = [(t0, next(counter), source)]
    while heap:
        t, _, u = heapq.heappop(heap)
        if u in settled:
            continue
        settled[u] = t - t0
        settled_parent[u] = parent.get(u)
        order.append(u)
        if target is not None and u == target:
            break
        for eid in graph.out_edges(u):
            e = graph.edge(eid)
            if not e.permits(mode) or e.v in settled:
                continue
            p = prof.get(eid)
            tt = e.free_flow_time_s if p is None else p.travel_time(t)
            if tt < 0.0:
                raise StreetNetworkError("negative travel time on edge %d." % eid)
            na = t + tt
            if e.v not in arrive or na < arrive[e.v]:
                arrive[e.v] = na
                parent[e.v] = eid
                heapq.heappush(heap, (na, next(counter), e.v))
    return SearchTree(source, settled, settled_parent, order, False)


def time_dependent_shortest_path(graph: RoadGraph, source: NodeId, target: NodeId,
                                 departure_s: float = 0.0, *,
                                 profiles: Optional[Dict[EdgeId, TravelTimeProfile]] = None,
                                 mode: Optional[str] = None,
                                 require_fifo: bool = True) -> Optional[Path]:
    """Earliest-arrival path; ``Path.cost`` is the total journey time in seconds."""
    tree = time_dependent_dijkstra(graph, source, departure_s, profiles=profiles,
                                   target=target, mode=mode,
                                   require_fifo=require_fifo)
    found = _extract(graph, tree, target)
    if found is None:
        return None
    nodes, edges = found
    return _make_path(graph, nodes, edges, _weight_fn("free_flow_time_s"),
                      tree.dist[target])


# ===========================================================================
# 10. ISOCHRONES -- reachable sets and concave-hull polygons
# ===========================================================================
@dataclass
class Isochrone:
    """Everything reachable from a source within a cost budget."""

    source: NodeId
    budget: float
    weight: str
    nodes: Dict[NodeId, float]
    edges: List[EdgeId]
    partial_edges: List[Tuple[EdgeId, float]]
    boundary_points: List[Coord]
    polygon: List[Ring] = field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        """Number of fully reachable junctions."""
        return len(self.nodes)

    @property
    def area_m2(self) -> float:
        """Total area of the isochrone rings, in square metres."""
        return math.fsum(polygon_area(r) for r in self.polygon)

    def reached_length_m(self, graph: RoadGraph) -> float:
        """Metres of street reachable within the budget (partial edges pro-rated)."""
        whole = math.fsum(graph.edge(e).length_m for e in self.edges)
        part = math.fsum(graph.edge(e).length_m * f for e, f in self.partial_edges)
        return whole + part


def isochrone(graph: RoadGraph, source: NodeId, budget: float, *,
              weight: CostSpec = "free_flow_time_s", mode: Optional[str] = None,
              alpha_m: Optional[float] = None,
              polygon: bool = True) -> Isochrone:
    """Cost-limited reachable set plus the concave hull of its frontier."""
    b = _as_non_negative(budget, "budget")
    wfn = _weight_fn(weight)
    tree = dijkstra(graph, source, weight=weight, mode=mode, max_cost=b)
    reached = dict(tree.dist)
    inside: List[EdgeId] = []
    partial: List[Tuple[EdgeId, float]] = []
    points: List[Coord] = [graph.node(n).xy for n in reached]
    for eid in graph.edges:
        e = graph.edge(eid)
        if not e.permits(mode) or e.u not in reached:
            continue
        w = wfn(e)
        remaining = b - reached[e.u]
        # scale-relative inclusive boundary: `b + _EPS` was an absolute 1e-12 slack on a
        # COST, meaningless for a budget of 1e-13 and invisible for one of 1e6
        if reached[e.u] + w <= b + _EPS * max(abs(b), abs(reached[e.u] + w)):
            inside.append(eid)
            continue
        if remaining <= 0.0 or w <= 0.0:
            continue
        frac = min(1.0, remaining / w)
        partial.append((eid, frac))
        points.append(_point_at_offset(e.geometry, frac))
    rings: List[Ring] = []
    if polygon and len(points) >= 3:
        rings = alpha_shape(points, alpha_m)
    return Isochrone(source, b, weight if isinstance(weight, str) else "custom",
                     reached, inside, partial, points, rings)


def _point_at_offset(geometry: Sequence[Coord], frac: float) -> Coord:
    """Interpolate a point a fraction of the way along a polyline."""
    pts = list(geometry)
    if len(pts) < 2:
        return pts[0] if pts else (0.0, 0.0)
    seg = [euclidean_m(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
           for i in range(len(pts) - 1)]
    total = math.fsum(seg)
    if total <= _EPS:
        return pts[0]
    goal = max(0.0, min(1.0, frac)) * total
    for i, L in enumerate(seg):
        # fsum the prefix: a left-to-right running total drifts against the fsum-computed
        # `total` by ~n*eps*total, which at long geographic lengths is far larger than any
        # absolute epsilon guard.
        run = math.fsum(seg[:i])
        if run + L >= goal or i == len(seg) - 1:
            t = 0.0 if L <= 0.0 else (goal - run) / L
            t = max(0.0, min(1.0, t))
            return (pts[i][0] + t * (pts[i + 1][0] - pts[i][0]),
                    pts[i][1] + t * (pts[i + 1][1] - pts[i][1]))
    return pts[-1]


def reachable_subgraph(graph: RoadGraph, source: NodeId, budget: float, *,
                       weight: CostSpec = "free_flow_time_s",
                       mode: Optional[str] = None) -> RoadGraph:
    """Induced subgraph on the nodes reachable within a cost budget."""
    tree = dijkstra(graph, source, weight=weight, mode=mode,
                    max_cost=_as_non_negative(budget, "budget"))
    return graph.subgraph(tree.dist)


# ===========================================================================
# 11. CENTRALITY -- Brandes betweenness, closeness, straightness, information
# ===========================================================================
_TIE = 1e-9


def _tie_tol(a: float, b: float) -> float:
    """Scale-RELATIVE equal-cost tolerance for the two path costs ``a`` and ``b``.

    ``_TIE * max(1.0, abs(cost))`` hard-coded a unit: below unit cost the tolerance
    degenerated to an absolute 1e-9, which merged genuinely distinct path costs into false
    ties.  That made betweenness -- a dimensionless count -- change when every length in
    the graph was uniformly rescaled (drift 67.5 at 1e-12 m spacing, and exactly 0.0 with
    the floor removed), and it bit at ordinary scales in the angular dual, whose arc costs
    are ``|deflection| / 90`` in [0, 2] and so genuinely sub-unit.
    """
    return _TIE * max(abs(a), abs(b))


def _brandes(graph: RoadGraph, wfn: Callable[[RoadEdge], float],
             mode: Optional[str], sources: Sequence[NodeId],
             pair_weight: Optional[Callable[[NodeId, NodeId], float]]
             ) -> Tuple[Dict[NodeId, float], Dict[EdgeId, float]]:
    """Brandes' dependency accumulation over a weighted directed road graph.

    ZERO-COST ARCS.  Classical Brandes compares path costs only, and that makes the number
    of shortest paths ill-defined as soon as an arc costs nothing: a zero-cost cycle
    admits unboundedly many equally short walks, two nodes at equal distance joined by a
    zero-cost arc are each other's predecessor, and the ``sigma`` recursion has no base.
    The implementation used to paper over this by discarding an equal-cost predecessor
    whenever it happened to settle second, which made the result depend on heap order --
    provably wrong, since betweenness is an isomorphism invariant and the answer was not
    constant on the orbits of the graph's own automorphism group.

    The label is therefore the LEXICOGRAPHIC pair ``(cost, zero-cost arcs used)``.  That
    is exactly the limit as every zero-cost arc is replaced by an infinitesimal positive
    cost, so it is the limit of a classically well-posed problem; the key strictly
    increases along every arc, so the shortest-path graph is acyclic and the settle order
    is a valid topological order again.  With strictly positive costs the second component
    never breaks a tie and the result is bit-identical to the classical convention (and so
    to NetworkX).
    """
    nb: Dict[NodeId, float] = dict((n, 0.0) for n in graph.nodes)
    eb: Dict[EdgeId, float] = dict((e, 0.0) for e in graph.edges)
    for s in sources:
        stack: List[NodeId] = []
        preds: Dict[NodeId, List[Tuple[NodeId, EdgeId]]] = {}
        sigma: Dict[NodeId, float] = {s: 1.0}
        seen: Dict[NodeId, Tuple[float, int]] = {s: (0.0, 0)}
        dist: Dict[NodeId, Tuple[float, int]] = {}
        counter = itertools.count()
        heap: List[Tuple[float, int, int, NodeId]] = [(0.0, 0, next(counter), s)]
        while heap:
            d, z, _, v = heapq.heappop(heap)
            if v in dist:
                continue
            dist[v] = (d, z)
            stack.append(v)
            for eid in graph.out_edges(v):
                e = graph.edge(eid)
                if not e.permits(mode):
                    continue
                w = e.v
                if w in dist or w == v:
                    continue
                arc = wfn(e)
                vw = d + arc
                vz = z + (1 if arc <= 0.0 else 0)
                prev = seen.get(w)
                if prev is None:
                    seen[w] = (vw, vz)
                    sigma[w] = sigma.get(v, 0.0)
                    preds[w] = [(v, eid)]
                    heapq.heappush(heap, (vw, vz, next(counter), w))
                    continue
                pc, pz = prev
                tol = _tie_tol(vw, pc)
                if vw < pc - tol or (abs(vw - pc) <= tol and vz < pz):
                    seen[w] = (vw, vz)
                    sigma[w] = sigma.get(v, 0.0)
                    preds[w] = [(v, eid)]
                    heapq.heappush(heap, (vw, vz, next(counter), w))
                elif abs(vw - pc) <= tol and vz == pz:
                    sigma[w] = sigma.get(w, 0.0) + sigma.get(v, 0.0)
                    preds.setdefault(w, []).append((v, eid))
        delta: Dict[NodeId, float] = {}
        while stack:
            w = stack.pop()
            sw = sigma.get(w, 0.0)
            if sw <= 0.0:
                continue
            pw = 1.0 if pair_weight is None else pair_weight(s, w)
            coeff = (pw + delta.get(w, 0.0)) / sw
            for (v, eid) in preds.get(w, []):
                c = sigma.get(v, 0.0) * coeff
                delta[v] = delta.get(v, 0.0) + c
                eb[eid] += c
            if w != s:
                nb[w] += delta.get(w, 0.0)
    return nb, eb


def betweenness_centrality(graph: RoadGraph, *, weight: CostSpec = "length_m",
                           mode: Optional[str] = None, normalised: bool = False,
                           directed: bool = True,
                           sources: Optional[Sequence[NodeId]] = None
                           ) -> Dict[NodeId, float]:
    """Brandes shortest-path betweenness (NetworkX scaling conventions)."""
    wfn = _weight_fn(weight)
    srcs = list(graph.nodes) if sources is None else list(sources)
    nb, _ = _brandes(graph, wfn, mode, srcs, None)
    n = graph.num_nodes
    if normalised:
        scale = 0.0 if n <= 2 else 1.0 / ((n - 1) * (n - 2))
    else:
        scale = 1.0 if directed else 0.5
    return dict((k, v * scale) for k, v in nb.items())


def edge_betweenness_centrality(graph: RoadGraph, *, weight: CostSpec = "length_m",
                                mode: Optional[str] = None,
                                normalised: bool = False, directed: bool = True,
                                sources: Optional[Sequence[NodeId]] = None
                                ) -> Dict[EdgeId, float]:
    """Brandes edge betweenness; ``directed=False`` merges each pair of twins."""
    wfn = _weight_fn(weight)
    srcs = list(graph.nodes) if sources is None else list(sources)
    _, eb = _brandes(graph, wfn, mode, srcs, None)
    n = graph.num_nodes
    if not directed:
        merged: Dict[EdgeId, float] = {}
        for eid, val in eb.items():
            twin = graph.edge(eid).reverse_eid
            merged[eid] = val + (eb.get(twin, 0.0) if twin is not None else 0.0)
        eb = merged
    if normalised:
        scale = 0.0 if n <= 1 else 1.0 / (n * (n - 1))
    else:
        scale = 1.0 if directed else 0.5
    return dict((k, v * scale) for k, v in eb.items())


def demand_weighted_betweenness(graph: RoadGraph,
                                od_demand: Dict[Tuple[NodeId, NodeId], float], *,
                                weight: CostSpec = "length_m",
                                mode: Optional[str] = None
                                ) -> Tuple[Dict[NodeId, float], Dict[EdgeId, float]]:
    """All-or-nothing flow betweenness: node through-flows and edge flows for an
    origin-destination matrix, split equally across equal-cost shortest paths."""
    wfn = _weight_fn(weight)
    demand: Dict[NodeId, Dict[NodeId, float]] = {}
    for (o, d), q in od_demand.items():
        v = _as_non_negative(q, "od_demand[%r, %r]" % (o, d))
        if v <= 0.0 or o == d:
            continue
        _require(graph.has_node(o), "unknown origin %r." % (o,))
        _require(graph.has_node(d), "unknown destination %r." % (d,))
        demand.setdefault(o, {})[d] = demand.setdefault(o, {}).get(d, 0.0) + v
    if not demand:
        return (dict((n, 0.0) for n in graph.nodes),
                dict((e, 0.0) for e in graph.edges))

    def pw(s: NodeId, t: NodeId) -> float:
        return demand.get(s, {}).get(t, 0.0)

    return _brandes(graph, wfn, mode, list(demand), pw)


def closeness_centrality(graph: RoadGraph, *, weight: CostSpec = "length_m",
                         mode: Optional[str] = None, variant: str = "standard",
                         normalised: bool = True, incoming: bool = False
                         ) -> Dict[NodeId, float]:
    """Closeness (Wasserman-Faust corrected) or harmonic closeness centrality."""
    _require(variant in ("standard", "harmonic"),
             "variant must be 'standard' or 'harmonic' (got %r)." % (variant,))
    n = graph.num_nodes
    out: Dict[NodeId, float] = {}
    for v in graph.nodes:
        tree = dijkstra(graph, v, weight=weight, mode=mode, reverse=incoming)
        ds = [d for u, d in tree.dist.items() if u != v]
        if variant == "harmonic":
            total = math.fsum(1.0 / d for d in ds if d > 0.0)
            out[v] = total / (n - 1) if (normalised and n > 1) else total
            continue
        tot = math.fsum(ds)
        if tot <= 0.0 or not ds:
            out[v] = 0.0
            continue
        c = len(ds) / tot
        if normalised and n > 1:
            c *= len(ds) / float(n - 1)
        out[v] = c
    return out


def straightness_centrality(graph: RoadGraph, *, weight: CostSpec = "length_m",
                            mode: Optional[str] = None,
                            normalised: bool = True) -> Dict[NodeId, float]:
    """Mean ratio of straight-line to network distance over all reachable nodes."""
    n = graph.num_nodes
    out: Dict[NodeId, float] = {}
    for v in graph.nodes:
        tree = dijkstra(graph, v, weight=weight, mode=mode)
        acc = []
        for u, d in tree.dist.items():
            if u == v or d <= 0.0:
                continue
            straight = graph.straight_line_distance(v, u)
            if straight <= 0.0:
                continue
            acc.append(min(1.0, straight / d))
        if not acc:
            out[v] = 0.0
        elif normalised:
            out[v] = math.fsum(acc) / float(n - 1) if n > 1 else 0.0
        else:
            out[v] = math.fsum(acc)
    return out


def _invert_matrix(a: List[List[float]]) -> List[List[float]]:
    """Gauss-Jordan inverse with partial pivoting.

    The singularity test is relative to the matrix norm.  An absolute ``<= 1e-14`` on the
    pivot is dimensional nonsense: ``diag(1e-14)`` was rejected as singular although its
    condition number is exactly 1 and its inverse is ``diag(1e14)``, while a matrix whose
    entries are all ~1e5 (conductances of millimetre-long edges) can carry elimination
    noise well above 1e-14 and hide a genuine singularity.
    """
    n = len(a)
    _require(n > 0 and all(len(r) == n for r in a), "matrix must be square.")
    norm = max([abs(x) for row in a for x in row] or [0.0])
    tol = 1e-14 * norm
    m = [list(row) + [1.0 if i == j else 0.0 for j in range(n)]
         for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if norm <= 0.0 or abs(m[piv][col]) <= tol:
            raise StreetNetworkError("matrix is singular at column %d." % col)
        m[col], m[piv] = m[piv], m[col]
        d = m[col][col]
        m[col] = [x / d for x in m[col]]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col]
            if f != 0.0:
                m[r] = [m[r][k] - f * m[col][k] for k in range(2 * n)]
    return [row[n:] for row in m]


def information_centrality(graph: RoadGraph, *,
                           weight_mode: str = "inverse_length",
                           normalised: bool = False) -> Dict[NodeId, float]:
    """Stephenson-Zelen information centrality on the undirected weighted view.

    SCALE CONVENTION: this returns the published Stephenson-Zelen statistic, which on
    ``K_n`` with unit conductances is exactly ``n^2 / (2 (n - 1))`` -- 2.25 for a triangle.
    ``networkx.current_flow_closeness_centrality`` returns the same quantity divided by
    ``n``, so any differential test against it must multiply by ``graph.num_nodes``; the
    suite pins the ratio at exactly ``n`` for n = 3..8.

    Edges are treated as electrical conductances (``1 / length_m`` by default);
    ``I_i = [c_ii + (trace(C) - 2R) / n]^-1`` with ``C = (D - A + J)^-1`` and
    ``R = 1/n`` the (constant) row sum.  Only the largest weakly connected
    component is scored; every other node gets ``0.0``.

    A ZERO-LENGTH edge is a perfect conductor, i.e. ``1 / L`` with ``L -> 0+``.  That
    limit is well defined and continuous -- sweeping L down through 1e-3, 1e-6 and 1e-9 of
    the shortest real edge converges cleanly -- so it is evaluated by substituting
    ``1e-8`` of the shortest positive length, which balances the O(L) truncation error
    against the O(eps/L) conditioning error and reproduces the limit to roughly eight
    significant digits.  Substituting conductance ``0.0`` instead, as this function used to
    do, picks the physically OPPOSITE extreme (an open circuit), disconnects the
    conductance matrix while the node stays in the same weak component, and then surfaced
    an internal "matrix is singular at column k" as though the caller's graph were
    malformed.  When EVERY edge has zero length there is no length scale at all and unit
    conductances are used.
    """
    _require(weight_mode in ("inverse_length", "unit"),
             "weight_mode must be 'inverse_length' or 'unit'.")
    comps = weakly_connected_components(graph)
    out: Dict[NodeId, float] = dict((n, 0.0) for n in graph.nodes)
    if not comps:
        return out
    members = max(comps, key=len)
    idx = dict((n, i) for i, n in enumerate(members))
    n = len(members)
    if n < 3:
        return out
    a = [[0.0] * n for _ in range(n)]
    inside = [eid for (u, v, eid) in graph.undirected_segments()
              if u in idx and v in idx and u != v]
    positive = [graph.edge(e).length_m for e in inside if graph.edge(e).length_m > 0.0]
    short_circuit = (min(positive) * 1e-8) if positive else 1.0
    for (u, v, eid) in graph.undirected_segments():
        if u not in idx or v not in idx or u == v:
            continue
        L = graph.edge(eid).length_m
        w = 1.0 if weight_mode == "unit" else 1.0 / (L if L > 0.0 else short_circuit)
        i, j = idx[u], idx[v]
        a[i][j] += w
        a[j][i] += w
    b = [[(math.fsum(a[i]) if i == j else 0.0) - a[i][j] + 1.0
          for j in range(n)] for i in range(n)]
    c = _invert_matrix(b)
    trace = math.fsum(c[i][i] for i in range(n))
    r = 1.0 / n
    for node, i in idx.items():
        denom = c[i][i] + (trace - 2.0 * r) / n
        # relative to the matrix's own scale, not to an absolute 1e-12 resistance
        floor = _EPS * (abs(c[i][i]) + abs(trace) / n)
        out[node] = 0.0 if abs(denom) <= floor else 1.0 / denom
    if normalised:
        tot = math.fsum(out.values())
        if tot > 0.0:
            out = dict((k, v / tot) for k, v in out.items())
    return out


# ===========================================================================
# 12. STREET-NETWORK METRICS -- circuity, density, orientation entropy
# ===========================================================================
def circuity(graph: RoadGraph, *, num_pairs: int = 200, seed: int = 0,
             weight: CostSpec = "length_m", mode: Optional[str] = None,
             pairs: Optional[Sequence[Tuple[NodeId, NodeId]]] = None) -> float:
    """Average ratio of network distance to straight-line distance (>= 1)."""
    nodes = graph.nodes
    # An EMPTY graph is malformed input and raises; a SINGLE-node graph is a
    # legitimate if trivial network that simply has no origin-destination pair,
    # so its circuity is undefined -- return NaN, matching the convention used
    # below when no sampled pair turns out to be connected.
    _require(len(nodes) >= 1, "circuity needs a non-empty network.")
    if len(nodes) < 2:
        return float("nan")
    if pairs is None:
        rng = random.Random(seed)
        want = _as_positive_int(num_pairs, "num_pairs")
        sample: List[Tuple[NodeId, NodeId]] = []
        for _ in range(want):
            a = nodes[rng.randrange(len(nodes))]
            b = nodes[rng.randrange(len(nodes))]
            if a != b:
                sample.append((a, b))
    else:
        sample = [(a, b) for a, b in pairs if a != b]
    ratios: List[float] = []
    cache: Dict[NodeId, SearchTree] = {}
    for a, b in sample:
        if a not in cache:
            cache[a] = dijkstra(graph, a, weight=weight, mode=mode)
        d = cache[a].dist.get(b)
        straight = graph.straight_line_distance(a, b)
        if d is None or straight <= 0.0 or d <= 0.0:
            continue
        ratios.append(d / straight)
    if not ratios:
        return float("nan")
    return math.fsum(ratios) / len(ratios)


def detour_index(graph: RoadGraph, **kwargs: Any) -> float:
    """Reciprocal of the circuity: how close the network is to straight lines."""
    c = circuity(graph, **kwargs)
    return float("nan") if (c != c or c <= 0.0) else 1.0 / c


def intersection_density(graph: RoadGraph, area_km2: Optional[float] = None, *,
                         min_degree: int = 3) -> float:
    """Junctions of at least ``min_degree`` streets per square kilometre."""
    count = sum(1 for n in graph.nodes if graph.degree(n) >= min_degree)
    if area_km2 is None:
        area = graph.area_m2() / 1e6
    else:
        area = _as_positive(area_km2, "area_km2")
    if area <= 0.0:
        return float("nan")
    return count / area


def node_degree_distribution(graph: RoadGraph, *, proportions: bool = True
                             ) -> Dict[int, float]:
    """Distribution of undirected street degree over the junctions."""
    counts: Dict[int, float] = {}
    for n in graph.nodes:
        k = graph.degree(n)
        counts[k] = counts.get(k, 0.0) + 1.0
    if proportions and graph.num_nodes:
        total = float(graph.num_nodes)
        counts = dict((k, v / total) for k, v in counts.items())
    return dict(sorted(counts.items()))


def edge_length_statistics(graph: RoadGraph) -> Dict[str, float]:
    """Count, total, mean, median, min, max and standard deviation of segment lengths."""
    lens = sorted(graph.edge(e).length_m for (_, _, e) in graph.undirected_segments())
    n = len(lens)
    if n == 0:
        return {"count": 0.0, "total_m": 0.0, "mean_m": 0.0, "median_m": 0.0,
                "min_m": 0.0, "max_m": 0.0, "std_m": 0.0}
    total = math.fsum(lens)
    mean = total / n
    median = lens[n // 2] if n % 2 else 0.5 * (lens[n // 2 - 1] + lens[n // 2])
    var = math.fsum((x - mean) ** 2 for x in lens) / n
    return {"count": float(n), "total_m": total, "mean_m": mean,
            "median_m": median, "min_m": lens[0], "max_m": lens[-1],
            "std_m": math.sqrt(var)}


def orientation_histogram(graph: RoadGraph, num_bins: int = 36, *,
                          weight_by_length: bool = True) -> List[float]:
    """Bidirectional bearing histogram (proportions), bins centred on 0 degrees."""
    nb = _as_positive_int(num_bins, "num_bins")
    _require(nb % 2 == 0, "num_bins must be even so that b and b+180 pair up.")
    width = 360.0 / nb
    half = width * 0.5
    bins = [0.0] * nb
    for (_, _, eid) in graph.undirected_segments():
        e = graph.edge(eid)
        w = e.length_m if weight_by_length else 1.0
        if w <= 0.0:
            w = 1.0
        b0 = e.start_bearing()
        for b in (b0, (b0 + 180.0) % 360.0):
            bins[int(((b + half) % 360.0) / width)] += w
    total = math.fsum(bins)
    if total <= 0.0:
        return bins          # no streets at all: a histogram with no mass, NOT proportions
    return [b / total for b in bins]


def street_orientation_entropy(graph: RoadGraph, num_bins: int = 36, *,
                               weight_by_length: bool = True) -> float:
    """Shannon entropy (nats) of the street-bearing distribution.

    A perfect grid puts every bearing in four bins and therefore has the
    *minimum* interesting entropy ``ln 4``; a radial or organic network spreads
    bearings over all bins and approaches the maximum ``ln(num_bins)``.  Use
    :func:`orientation_order` for the metric that is *maximal* for a grid.

    A graph with NO streets has no bearing distribution, so the entropy is undefined and
    the answer is ``nan`` -- the module's convention for an undefined metric, the same one
    :func:`intersection_density`, :func:`circuity` and :func:`detour_index` already use on
    a graph too small to measure.  It used to return ``-0.0``, the entropy of the all-zero
    histogram, which :func:`orientation_order` then turned into 0.6019, an in-range value
    reading as "fairly grid-like" for a network with no streets in it.
    """
    hist = orientation_histogram(graph, num_bins, weight_by_length=weight_by_length)
    if math.fsum(hist) <= 0.0:
        return float("nan")
    return -math.fsum(p * math.log(p) for p in hist if p > 0.0)


def orientation_order(graph: RoadGraph, num_bins: int = 36, *,
                      weight_by_length: bool = True) -> float:
    """Boeing's orientation order ``phi``: 1.0 for a perfect grid, 0.0 for chaos."""
    nb = _as_positive_int(num_bins, "num_bins")
    h = street_orientation_entropy(graph, nb, weight_by_length=weight_by_length)
    if math.isnan(h):
        return float("nan")   # no streets; must be explicit -- min(1.0, nan) is 1.0
    h_grid = math.log(4.0)
    h_max = math.log(float(nb))
    if h_max - h_grid <= _EPS:
        # CONVENTION, not a failure: phi's normalisation (h - ln 4) / (ln nb - ln 4) is
        # genuinely undefined for num_bins <= 4, so the answer is nan.
        return float("nan")
    phi = 1.0 - ((h - h_grid) / (h_max - h_grid)) ** 2
    return max(0.0, min(1.0, phi))


def network_summary(graph: RoadGraph, *, num_bins: int = 36,
                    circuity_pairs: int = 100, seed: int = 0) -> Dict[str, Any]:
    """One dictionary with the whole standard street-network metric battery."""
    base = graph.summary()
    blocks = detect_blocks(graph)
    stats = edge_length_statistics(graph)
    base.update({
        "intersection_density_per_km2": intersection_density(graph),
        "degree_distribution": node_degree_distribution(graph),
        "mean_segment_length_m": stats["mean_m"],
        "median_segment_length_m": stats["median_m"],
        "orientation_entropy": street_orientation_entropy(graph, num_bins),
        "orientation_order": orientation_order(graph, num_bins),
        "circuity": circuity(graph, num_pairs=circuity_pairs, seed=seed),
        "num_blocks": len(blocks),
        "mean_block_area_m2": (math.fsum(b.area_m2 for b in blocks) / len(blocks))
        if blocks else 0.0,
        "num_components_weak": len(weakly_connected_components(graph)),
        "num_components_strong": len(strongly_connected_components(graph)),
    })
    return base


# ===========================================================================
# 13. BLOCKS -- planar face traversal on the bearing-sorted embedding
# ===========================================================================
@dataclass
class Block:
    """One bounded face of the planar street embedding: a city block."""

    block_id: int
    nodes: List[NodeId]
    edges: List[EdgeId]
    ring: Ring
    area_m2: float
    perimeter_m: float
    centroid: Coord

    @property
    def num_sides(self) -> int:
        """Number of street segments bounding the block."""
        return len(self.edges)

    @property
    def compactness(self) -> float:
        """Polsby-Popper compactness ``4 pi A / P^2`` (1.0 for a circle)."""
        if self.perimeter_m <= 0.0:
            return 0.0
        return 4.0 * math.pi * self.area_m2 / (self.perimeter_m ** 2)


def detect_blocks(graph: RoadGraph, *, min_area_m2: float = 1e-9) -> List[Block]:
    """Enumerate the bounded faces of the planar embedding (left-hand rule).

    CONVENTION: ``min_area_m2`` is an ABSOLUTE sliver filter whose unit is in its name --
    1 mm^2 by default, for a module whose declared base unit is metres throughout.  It is
    therefore scale-dependent by design: a lattice at 1e-5 m spacing has 1e-10 m^2 faces
    and reports zero blocks until the caller lowers the threshold.  That is an explicit
    keyword, not a hidden constant, and the suite pins both sides of it.
    """
    segs = graph.undirected_segments()
    if not segs:
        return []
    geoms: List[Tuple[Coord, ...]] = []
    for (_, _, eid) in segs:
        geoms.append(tuple(graph.edge(eid).geometry))
    nh = 2 * len(segs)

    def origin(h: int) -> NodeId:
        u, v, _ = segs[h >> 1]
        return u if (h & 1) == 0 else v

    def target(h: int) -> NodeId:
        u, v, _ = segs[h >> 1]
        return v if (h & 1) == 0 else u

    def geom(h: int) -> List[Coord]:
        g = geoms[h >> 1]
        return list(g) if (h & 1) == 0 else list(reversed(g))

    def out_bearing(h: int) -> float:
        g = geom(h)
        if len(g) < 2:
            return 0.0
        return bearing_deg(g[0][0], g[0][1], g[1][0], g[1][1])

    at: Dict[NodeId, List[int]] = {}
    for h in range(nh):
        at.setdefault(origin(h), []).append(h)
    for node in at:
        at[node].sort(key=lambda h: (out_bearing(h), h))
    pos: Dict[int, int] = {}
    for node, lst in at.items():
        for i, h in enumerate(lst):
            pos[h] = i
    blocks: List[Block] = []
    visited: Set[int] = set()
    for start in range(nh):
        if start in visited:
            continue
        face: List[int] = []
        h = start
        guard = 0
        while True:
            visited.add(h)
            face.append(h)
            rev = h ^ 1
            lst = at[origin(rev)]
            h = lst[(pos[rev] + 1) % len(lst)]
            guard += 1
            if h == start or guard > nh + 1:
                break
        ring: Ring = []
        perimeter = 0.0
        for hh in face:
            g = geom(hh)
            ring.extend(g[:-1])
            perimeter += polyline_length(g)
        signed = polygon_area(ring, signed=True)
        if signed <= min_area_m2:
            continue                              # outer face or degenerate spur
        blocks.append(Block(len(blocks),
                            [origin(hh) for hh in face],
                            [segs[hh >> 1][2] for hh in face],
                            ring, signed, perimeter, polygon_centroid(ring)))
    blocks.sort(key=lambda b: (-b.area_m2, b.num_sides))
    for i, b in enumerate(blocks):
        b.block_id = i
    return blocks


def block_statistics(blocks: Sequence[Block]) -> Dict[str, float]:
    """Count, total, mean, median, min and max block area plus mean compactness."""
    areas = sorted(b.area_m2 for b in blocks)
    n = len(areas)
    if n == 0:
        return {"count": 0.0, "total_m2": 0.0, "mean_m2": 0.0, "median_m2": 0.0,
                "min_m2": 0.0, "max_m2": 0.0, "mean_compactness": 0.0,
                "mean_sides": 0.0}
    median = areas[n // 2] if n % 2 else 0.5 * (areas[n // 2 - 1] + areas[n // 2])
    return {"count": float(n), "total_m2": math.fsum(areas),
            "mean_m2": math.fsum(areas) / n, "median_m2": median,
            "min_m2": areas[0], "max_m2": areas[-1],
            "mean_compactness": math.fsum(b.compactness for b in blocks) / n,
            "mean_sides": math.fsum(float(b.num_sides) for b in blocks) / n}


def average_block_size(graph: RoadGraph) -> float:
    """Mean area in square metres of the bounded faces of the street network."""
    return block_statistics(detect_blocks(graph))["mean_m2"]


# ===========================================================================
# 14. SPACE SYNTAX -- strokes, the segment dual, depth, integration, choice
# ===========================================================================
@dataclass
class Stroke:
    """A "natural road": consecutive segments joined by the best-fit continuation."""

    stroke_id: int
    edges: List[EdgeId]
    nodes: List[NodeId]
    length_m: float
    mean_bearing_deg: float

    @property
    def num_segments(self) -> int:
        """Number of street segments in the stroke."""
        return len(self.edges)


def generate_strokes(graph: RoadGraph, max_deflection_deg: float = 60.0
                     ) -> List[Stroke]:
    """Best-fit stroke (natural-road) generation: pair up the straightest through
    movements at every junction, then chain the pairs into continuity lines."""
    limit = _as_non_negative(max_deflection_deg, "max_deflection_deg")
    segs = [(u, v, eid) for (u, v, eid) in graph.undirected_segments() if u != v]
    m = len(segs)
    if m == 0:
        return []
    ends: List[Tuple[NodeId, NodeId]] = [(u, v) for (u, v, _) in segs]

    def away_bearing(i: int, node: NodeId) -> float:
        u, v, eid = segs[i]
        e = graph.edge(eid)
        if node == u:
            return e.start_bearing()
        return (e.end_bearing() + 180.0) % 360.0

    at: Dict[NodeId, List[int]] = {}
    for i, (u, v) in enumerate(ends):
        at.setdefault(u, []).append(i)
        at.setdefault(v, []).append(i)
    partner: Dict[Tuple[int, NodeId], int] = {}
    for node in sorted(at, key=repr):
        inc = at[node]
        cands: List[Tuple[float, int, int]] = []
        for a in range(len(inc)):
            for b in range(a + 1, len(inc)):
                i, j = inc[a], inc[b]
                dev = abs(180.0 - abs(bearing_difference_deg(
                    away_bearing(i, node), away_bearing(j, node))))
                if dev <= limit:
                    cands.append((dev, i, j))
        cands.sort()
        taken: Set[int] = set()
        for dev, i, j in cands:
            if i in taken or j in taken:
                continue
            taken.add(i)
            taken.add(j)
            partner[(i, node)] = j
            partner[(j, node)] = i
    used: Set[int] = set()
    strokes: List[Stroke] = []
    order = sorted(range(m), key=lambda i: (
        sum(1 for nd in ends[i] if (i, nd) in partner), i))
    for start in order:
        if start in used:
            continue
        head = ends[start][0] if (start, ends[start][0]) not in partner \
            else ends[start][1]
        chain = [start]
        used.add(start)
        cur = start
        entry = head
        while True:
            u, v = ends[cur]
            exit_node = v if entry == u else u
            nxt = partner.get((cur, exit_node))
            if nxt is None or nxt in used:
                break
            used.add(nxt)
            chain.append(nxt)
            cur = nxt
            entry = exit_node
        node_seq: List[NodeId] = [head]
        entry = head
        for i in chain:
            u, v = ends[i]
            exit_node = v if entry == u else u
            node_seq.append(exit_node)
            entry = exit_node
        total = math.fsum(graph.edge(segs[i][2]).length_m for i in chain)
        sx = math.fsum(math.sin(math.radians(away_bearing(i, ends[i][0])))
                       for i in chain)
        sy = math.fsum(math.cos(math.radians(away_bearing(i, ends[i][0])))
                       for i in chain)
        mb = bearing_deg(0.0, 0.0, sx, sy) if (abs(sx) + abs(sy)) > _EPS else 0.0
        strokes.append(Stroke(len(strokes), [segs[i][2] for i in chain],
                              node_seq, total, mb))
    return strokes


def segment_dual_graph(graph: RoadGraph, *, angular: bool = False
                       ) -> Dict[EdgeId, List[Tuple[EdgeId, float]]]:
    """Adjacency of the segment dual: undirected segments joined where they meet.

    With ``angular=True`` the arc cost is the normalised angular change
    ``|deflection| / 90`` (Hillier-Iida angular segment analysis); otherwise it
    is 1 (topological step depth).
    """
    segs = graph.undirected_segments()
    at: Dict[NodeId, List[EdgeId]] = {}
    for (u, v, eid) in segs:
        at.setdefault(u, []).append(eid)
        if v != u:
            at.setdefault(v, []).append(eid)
    adj: Dict[EdgeId, List[Tuple[EdgeId, float]]] = dict((e, []) for (_, _, e) in segs)
    seen: Set[Tuple[EdgeId, EdgeId]] = set()
    for node, inc in at.items():
        for a in range(len(inc)):
            for b in range(len(inc)):
                if a == b:
                    continue
                i, j = inc[a], inc[b]
                if i == j or (i, j) in seen:
                    continue
                seen.add((i, j))
                if angular:
                    ea, eb = graph.edge(i), graph.edge(j)
                    ba = ea.end_bearing() if ea.v == node else \
                        (ea.start_bearing() + 180.0) % 360.0
                    bb = eb.start_bearing() if eb.u == node else \
                        (eb.end_bearing() + 180.0) % 360.0
                    cost = abs(bearing_difference_deg(ba, bb)) / 90.0
                else:
                    cost = 1.0
                adj[i].append((j, cost))
    return adj


@dataclass
class SpaceSyntaxResult:
    """Configurational measures for every unit of analysis (segment or stroke)."""

    unit: str
    radius: Optional[int]
    angular: bool
    connectivity: Dict[Any, float]
    total_depth: Dict[Any, float]
    mean_depth: Dict[Any, float]
    relative_asymmetry: Dict[Any, float]
    real_relative_asymmetry: Dict[Any, float]
    integration: Dict[Any, float]
    choice: Dict[Any, float]
    control: Dict[Any, float]
    node_count: Dict[Any, int]
    intelligibility: float
    members: Dict[Any, List[EdgeId]] = field(default_factory=dict)

    def most_integrated(self, k: int = 5) -> List[Tuple[Any, float]]:
        """The ``k`` most integrated units, highest first."""
        return sorted(self.integration.items(), key=lambda kv: -kv[1])[:k]

    def summary(self) -> Dict[str, Any]:
        """Means of the configurational measures plus the intelligibility."""
        vals = list(self.integration.values())
        return {"unit": self.unit, "units": len(self.integration),
                "meanIntegration": (math.fsum(vals) / len(vals)) if vals else 0.0,
                "maxIntegration": max(vals) if vals else 0.0,
                "meanConnectivity": (math.fsum(self.connectivity.values())
                                     / len(self.connectivity))
                if self.connectivity else 0.0,
                "intelligibility": self.intelligibility}


def _diamond_value(k: int) -> float:
    """Hillier-Hanson D_k used to turn relative asymmetry into RRA."""
    if k < 4:
        return 1.0
    return (2.0 * (k * (math.log((k + 2.0) / 3.0, 2.0) - 1.0) + 1.0)
            / ((k - 1.0) * (k - 2.0)))


def _dual_depths(adj: Dict[Any, List[Tuple[Any, float]]], src: Any,
                 radius: Optional[int], angular: bool) -> Dict[Any, float]:
    """Step (or angular) depth from one dual node, optionally radius-limited."""
    if not angular:
        depth = {src: 0.0}
        frontier = [src]
        step = 0
        while frontier:
            step += 1
            if radius is not None and step > radius:
                break
            nxt: List[Any] = []
            for u in frontier:
                for v, _ in adj.get(u, []):
                    if v not in depth:
                        depth[v] = float(step)
                        nxt.append(v)
            frontier = nxt
        return depth
    dist = {src: 0.0}
    settled: Dict[Any, float] = {}
    counter = itertools.count()
    heap: List[Tuple[float, int, Any]] = [(0.0, next(counter), src)]
    while heap:
        d, _, u = heapq.heappop(heap)
        if u in settled:
            continue
        settled[u] = d
        if radius is not None and d > radius:
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if radius is not None and nd > radius:
                continue
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, next(counter), v))
    return settled


def _dual_choice(adj: Dict[Any, List[Tuple[Any, float]]]) -> Dict[Any, float]:
    """Brandes betweenness ("choice") on an undirected dual adjacency.

    Uses the same lexicographic ``(cost, zero-cost arcs)`` label as :func:`_brandes`; see
    that docstring for why.  This is where it matters most in practice: an angular segment
    dual gives every perfectly straight continuation a cost of exactly zero (60 of the 188
    arcs of ``grid_network(5, 5)``), and every such arc is an undirected 2-cycle of equal
    distance, so under a cost-only comparison the path counts had no well-defined value at
    all and the answer depended on heap order.
    """
    nodes = list(adj)
    bc: Dict[Any, float] = dict((n, 0.0) for n in nodes)
    for s in nodes:
        stack: List[Any] = []
        preds: Dict[Any, List[Any]] = {}
        sigma: Dict[Any, float] = {s: 1.0}
        seen: Dict[Any, Tuple[float, int]] = {s: (0.0, 0)}
        dist: Dict[Any, Tuple[float, int]] = {}
        counter = itertools.count()
        heap: List[Tuple[float, int, int, Any]] = [(0.0, 0, next(counter), s)]
        while heap:
            d, z, _, v = heapq.heappop(heap)
            if v in dist:
                continue
            dist[v] = (d, z)
            stack.append(v)
            for w, wt in adj.get(v, []):
                if w in dist or w == v:
                    continue
                vw = d + wt
                vz = z + (1 if wt <= 0.0 else 0)
                prev = seen.get(w)
                if prev is None:
                    seen[w] = (vw, vz)
                    sigma[w] = sigma.get(v, 0.0)
                    preds[w] = [v]
                    heapq.heappush(heap, (vw, vz, next(counter), w))
                    continue
                pc, pz = prev
                tol = _tie_tol(vw, pc)
                if vw < pc - tol or (abs(vw - pc) <= tol and vz < pz):
                    seen[w] = (vw, vz)
                    sigma[w] = sigma.get(v, 0.0)
                    preds[w] = [v]
                    heapq.heappush(heap, (vw, vz, next(counter), w))
                elif abs(vw - pc) <= tol and vz == pz:
                    sigma[w] = sigma.get(w, 0.0) + sigma.get(v, 0.0)
                    preds.setdefault(w, []).append(v)
        delta: Dict[Any, float] = {}
        while stack:
            w = stack.pop()
            sw = sigma.get(w, 0.0)
            if sw <= 0.0:
                continue
            coeff = (1.0 + delta.get(w, 0.0)) / sw
            for v in preds.get(w, []):
                delta[v] = delta.get(v, 0.0) + sigma.get(v, 0.0) * coeff
            if w != s:
                bc[w] += delta.get(w, 0.0)
    return dict((k, 0.5 * v) for k, v in bc.items())


def _pearson_r2(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Squared Pearson correlation of two equal-length samples."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = math.fsum(xs) / n
    my = math.fsum(ys) / n
    sxy = math.fsum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sxx = math.fsum((xs[i] - mx) ** 2 for i in range(n))
    syy = math.fsum((ys[i] - my) ** 2 for i in range(n))
    # The "no variance" test must be relative to the sample's own magnitude.  `sxx <= _EPS`
    # applied an absolute 1e-12 to a SUM OF SQUARED DEVIATIONS, i.e. an effective 1e-6
    # cut-off in the sample's unit, so a perfectly correlated but small-valued sample
    # reported r^2 = 0 instead of 1.  The threshold below says "the spread is below double
    # precision relative to the values", which is the only scale-free reading.
    ssx = math.fsum(v * v for v in xs)
    ssy = math.fsum(v * v for v in ys)
    if sxx <= _EPS * _EPS * ssx or syy <= _EPS * _EPS * ssy:
        return 0.0
    r = sxy / math.sqrt(sxx * syy)
    return min(1.0, r * r)


def space_syntax(graph: RoadGraph, *, unit: str = "segment",
                 radius: Optional[int] = None, angular: bool = False,
                 max_deflection_deg: float = 60.0) -> SpaceSyntaxResult:
    """Configurational analysis: mean depth, RA, RRA, integration, choice, control.

    ``radius`` limits how far each unit is analysed, and its MEANING follows ``angular``:
    with ``angular=False`` it is a step count (a radius of 1 is the unit's immediate
    neighbours), while with ``angular=True`` it is a cumulative ANGULAR COST budget in
    units of 90 degrees, which is the standard Hillier-Iida reading but a different
    quantity from the same ``Optional[int]`` parameter.  On ``grid_network(7, 7)``,
    ``radius=1`` reaches a mean of 6.1 of 84 units topologically and 48.0 of 84
    angularly.  Both readings are legitimate; only the overloading was undocumented.

    ``angular=True`` applies to ``unit="segment"`` only.  The stroke dual has no angular
    cost -- a stroke is by construction the continuity line through the straightest
    through-movements, and every stroke-to-stroke arc costs one step -- so the angular
    Dijkstra degenerated to breadth-first search and returned numbers identical to the
    topological ones, in a result whose ``angular`` field still read ``True``.  Rather
    than label topological numbers "angular", the combination is rejected.
    """
    _require(unit in ("segment", "stroke"),
             "unit must be 'segment' or 'stroke' (got %r)." % (unit,))
    _require(not (unit == "stroke" and angular),
             "angular analysis is defined for unit='segment' only: the stroke dual has "
             "no angular cost, so unit='stroke' with angular=True would return purely "
             "topological depths labelled angular.")
    members: Dict[Any, List[EdgeId]] = {}
    if unit == "segment":
        adj = segment_dual_graph(graph, angular=angular)
        for eid in adj:
            members[eid] = [eid]
    else:
        strokes = generate_strokes(graph, max_deflection_deg)
        owner: Dict[EdgeId, int] = {}
        for st in strokes:
            members[st.stroke_id] = list(st.edges)
            for eid in st.edges:
                owner[eid] = st.stroke_id
        at: Dict[NodeId, Set[int]] = {}
        for (u, v, eid) in graph.undirected_segments():
            sid = owner.get(eid)
            if sid is None:
                continue
            at.setdefault(u, set()).add(sid)
            at.setdefault(v, set()).add(sid)
        adj = dict((st.stroke_id, []) for st in strokes)
        seen_pair: Set[Tuple[int, int]] = set()
        for node, sids in at.items():
            ordered = sorted(sids)
            for a in ordered:
                for b in ordered:
                    if a == b or (a, b) in seen_pair:
                        continue
                    seen_pair.add((a, b))
                    adj[a].append((b, 1.0))
    keys = list(adj)
    connectivity: Dict[Any, float] = {}
    total_depth: Dict[Any, float] = {}
    mean_depth: Dict[Any, float] = {}
    ra: Dict[Any, float] = {}
    rra: Dict[Any, float] = {}
    integration: Dict[Any, float] = {}
    control: Dict[Any, float] = {}
    counts: Dict[Any, int] = {}
    deg = dict((k, float(len(adj[k]))) for k in keys)
    for k in keys:
        connectivity[k] = deg[k]
        control[k] = math.fsum(1.0 / deg[j] for j, _ in adj[k] if deg[j] > 0.0)
        depths = _dual_depths(adj, k, radius, angular)
        reach = [d for j, d in depths.items() if j != k]
        kk = len(reach) + 1
        counts[k] = kk
        td = math.fsum(reach)
        total_depth[k] = td
        if kk < 3 or not reach:
            mean_depth[k] = 0.0
            ra[k] = 0.0
            rra[k] = 0.0
            integration[k] = 0.0
            continue
        md = td / (kk - 1)
        mean_depth[k] = md
        rel = 2.0 * (md - 1.0) / (kk - 2.0)
        ra[k] = rel
        dv = _diamond_value(kk)
        val = rel / dv if dv > _EPS else rel
        rra[k] = val
        # Hillier-Hanson integration is 1 / RRA.  RRA == 0 means RA == 0 means mean depth
        # == 1: the unit reaches every other unit in one step, the MAXIMALLY integrated
        # case, whose integration is infinite.  Mapping it to 0.0 put it at the opposite
        # end of the scale, so `most_integrated` omitted the most integrated unit
        # altogether and ranked it last.  (0.0 is also already overloaded on the kk < 3
        # branch above for "too few units to measure".)
        integration[k] = (1.0 / val) if val > _EPS else float("inf")
    choice = _dual_choice(adj)
    # Intelligibility is the connectivity/integration correlation over the units that HAVE
    # a finite integration; a maximally integrated unit carries no usable ordinate.
    xs = [connectivity[k] for k in keys if math.isfinite(integration[k])]
    ys = [integration[k] for k in keys if math.isfinite(integration[k])]
    return SpaceSyntaxResult(unit, radius, angular, connectivity, total_depth,
                             mean_depth, ra, rra, integration, choice, control,
                             counts, _pearson_r2(xs, ys), members)


# ===========================================================================
# 15. SNAPPING, EDGE SPLITTING AND HMM MAP MATCHING
# ===========================================================================
@dataclass
class SnapResult:
    """The nearest point on a road edge to a query point."""

    eid: EdgeId
    distance_m: float
    offset_m: float
    offset_frac: float
    point: Coord

    def summary(self) -> Dict[str, Any]:
        """JSON-able view of the snap."""
        return {"edgeId": self.eid, "distanceM": self.distance_m,
                "offsetM": self.offset_m, "offsetFrac": self.offset_frac,
                "x": self.point[0], "y": self.point[1]}


class EdgeIndex:
    """A uniform grid over the edge geometries for nearest-edge queries.

    Rebuild the index after any structural change (``split_edge`` and friends);
    it caches geometry, not references.
    """

    def __init__(self, graph: RoadGraph, cell_size_m: Optional[float] = None, *,
                 mode: Optional[str] = None) -> None:
        self.graph = graph
        self.mode = mode
        self._eids = [e.eid for e in graph.iter_edges() if e.permits(mode)]
        minx, miny, maxx, maxy = graph.bbox()
        span = max(maxx - minx, maxy - miny)
        if cell_size_m is None:
            lens = sorted(graph.edge(e).length_m for e in self._eids)
            med = lens[len(lens) // 2] if lens else 0.0
            # the cell size is a PERFORMANCE parameter, so the only hard requirement is
            # that it be strictly positive; `max(cell, 1e-6)` hard-coded a micrometre
            # floor that forced a sub-micrometre network into a single bucket
            cell = med if med > 0.0 else (span / 16.0 if span > 0.0 else 1.0)
        else:
            cell = _as_positive(cell_size_m, "cell_size_m")
        self.cell = cell if cell > 0.0 else 1.0
        self._ox, self._oy = minx, miny
        self._cells: Dict[Tuple[int, int], List[EdgeId]] = {}
        for eid in self._eids:
            geom = graph.edge(eid).geometry
            for i in range(len(geom) - 1):
                self._insert_segment(eid, geom[i], geom[i + 1])
        # Remember the OCCUPIED extent so a query can never enumerate cells the index
        # does not have.  Without this, cost tracked radius^2 with complete indifference
        # to how few edges exist: snap_point's default max_dist_m=inf took 34 s on a
        # 25-node lattice, and a lat/lon-swapped query would have scanned ~1e10 cells.
        if self._cells:
            self._imin = min(k[0] for k in self._cells)
            self._imax = max(k[0] for k in self._cells)
            self._jmin = min(k[1] for k in self._cells)
            self._jmax = max(k[1] for k in self._cells)
        else:
            self._imin = self._imax = self._jmin = self._jmax = 0

    def _key(self, x: float, y: float) -> Tuple[int, int]:
        """Grid cell containing a planar point."""
        return (int(math.floor((x - self._ox) / self.cell)),
                int(math.floor((y - self._oy) / self.cell)))

    def _insert_segment(self, eid: EdgeId, a: Coord, b: Coord) -> None:
        """Register an edge in every cell its segment's bounding box touches."""
        i0, j0 = self._key(min(a[0], b[0]), min(a[1], b[1]))
        i1, j1 = self._key(max(a[0], b[0]), max(a[1], b[1]))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                bucket = self._cells.setdefault((i, j), [])
                if not bucket or bucket[-1] != eid:
                    if eid not in bucket:
                        bucket.append(eid)

    def candidates(self, x: float, y: float, radius_m: float) -> List[EdgeId]:
        """Edge ids whose cells intersect the query disc (a superset).

        The scan is clipped to the index's own occupied extent, and once the clipped
        window holds more cells than the index does it is cheaper to filter the occupied
        cells; both orders are the same ``(i, j)`` lexicographic order, so the returned
        list is identical either way.  Cost is therefore bounded by the number of occupied
        cells, not by ``radius_m ** 2``.
        """
        r = _as_non_negative(radius_m, "radius_m")
        if not self._cells:
            return []
        i0, j0 = self._key(x - r, y - r)
        i1, j1 = self._key(x + r, y + r)
        i0 = max(i0, self._imin)
        i1 = min(i1, self._imax)
        j0 = max(j0, self._jmin)
        j1 = min(j1, self._jmax)
        if i0 > i1 or j0 > j1:
            return []
        out: List[EdgeId] = []
        mark: Set[EdgeId] = set()
        if (i1 - i0 + 1) * (j1 - j0 + 1) > len(self._cells):
            keys = sorted(k for k in self._cells
                          if i0 <= k[0] <= i1 and j0 <= k[1] <= j1)
        else:
            keys = [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]
        for k in keys:
            for eid in self._cells.get(k, ()):
                if eid not in mark:
                    mark.add(eid)
                    out.append(eid)
        return out

    def _exhaustive_radius(self, x: float, y: float) -> float:
        """Smallest radius whose query window already covers the whole index."""
        if not self._cells:
            return 0.0
        lox = self._ox + self._imin * self.cell
        hix = self._ox + (self._imax + 1) * self.cell
        loy = self._oy + self._jmin * self.cell
        hiy = self._oy + (self._jmax + 1) * self.cell
        return math.hypot(max(abs(x - lox), abs(x - hix)),
                          max(abs(y - loy), abs(y - hiy)))

    def within(self, x: float, y: float, radius_m: float) -> List[SnapResult]:
        """Every edge whose polyline passes within ``radius_m``, nearest first."""
        px = _as_float(x, "x")
        py = _as_float(y, "y")
        r = _as_non_negative(radius_m, "radius_m")
        out: List[SnapResult] = []
        for eid in self.candidates(px, py, r):
            geom = self.graph.edge(eid).geometry
            d, off, frac, pt = project_point_on_polyline(px, py, geom)
            if d <= r:
                out.append(SnapResult(eid, d, off, frac, pt))
        out.sort(key=lambda s: (s.distance_m, s.eid))
        return out

    def nearest(self, x: float, y: float, *,
                max_dist_m: float = float("inf")) -> Optional[SnapResult]:
        """Nearest edge, expanding the search radius until something is found.

        The doubling stops at the radius that already covers the whole index rather than
        at a hard-coded 1e9, so the default ``max_dist_m=inf`` really means "the nearest
        edge, however far" and costs a bounded number of exhaustive passes instead of
        walking ~4e14 grid cells.
        """
        if not self._cells:
            return None
        hard = min(_as_non_negative(max_dist_m, "max_dist_m")
                   if max_dist_m != float("inf") else float("inf"),
                   self._exhaustive_radius(x, y))
        r = self.cell
        while True:
            hits = self.within(x, y, min(r, hard))
            if hits:
                return hits[0]
            if r >= hard:
                return None
            r *= 2.0

    def summary(self) -> Dict[str, Any]:
        """Cell size, cell count and mean occupancy."""
        occ = [len(v) for v in self._cells.values()]
        return {"edges": len(self._eids), "cells": len(self._cells),
                "cellSizeM": self.cell,
                "meanOccupancy": (math.fsum(occ) / len(occ)) if occ else 0.0}


def snap_point(graph: RoadGraph, x: float, y: float, *,
               max_dist_m: float = float("inf"),
               index: Optional[EdgeIndex] = None,
               mode: Optional[str] = None) -> Optional[SnapResult]:
    """Snap a planar point to the nearest permitted road edge."""
    idx = index if index is not None else EdgeIndex(graph, mode=mode)
    return idx.nearest(_as_float(x, "x"), _as_float(y, "y"), max_dist_m=max_dist_m)


def _split_polyline(geom: Sequence[Coord], frac: float
                    ) -> Tuple[List[Coord], List[Coord], Coord]:
    """Cut a polyline at a fractional offset, returning both halves and the point."""
    pts = _as_coords(geom, "geometry")
    _require(len(pts) >= 2, "geometry needs at least two vertices.")
    seg = [euclidean_m(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
           for i in range(len(pts) - 1)]
    total = math.fsum(seg)
    goal = max(0.0, min(1.0, frac)) * total
    if total <= 0.0:
        return [pts[0], pts[-1]], [pts[0], pts[-1]], pts[0]
    for i, L in enumerate(seg):
        # fsum prefix plus a last-segment guard, so the cut is always found inside the
        # loop; the old `>= goal - _EPS` was an ABSOLUTE 1e-12 slack against the drift
        # between a running total and an fsum total, which a long polyline exceeds.
        run = math.fsum(seg[:i])
        if run + L >= goal or i == len(seg) - 1:
            t = 0.0 if L <= 0.0 else (goal - run) / L
            t = max(0.0, min(1.0, t))
            cut = (pts[i][0] + t * (pts[i + 1][0] - pts[i][0]),
                   pts[i][1] + t * (pts[i + 1][1] - pts[i][1]))
            # Do NOT repeat a vertex the cut lands exactly on.  `[cut] + pts[i + 1:]`
            # emitted it twice whenever t == 1 (a cut at an existing shape point, e.g.
            # frac=0.25 of a [0, 100, 400] polyline), leaving the tail half with a
            # leading ZERO-LENGTH segment -- so RoadEdge.start_bearing(), which reads
            # geometry[0] -> geometry[1], returned 0.0 (due north) instead of the real
            # bearing, and that feeds turn deflections, generate_strokes and the
            # angular segment dual.
            head = list(pts[:i + 1])
            if not head or head[-1] != cut:
                head.append(cut)
            if len(head) < 2:
                head = [cut, cut]
            tail = list(pts[i + 1:])
            if not tail or tail[0] != cut:
                tail.insert(0, cut)
            if len(tail) < 2:
                tail = [cut, cut]
            return head, tail, cut
    return list(pts), [pts[-1], pts[-1]], pts[-1]


def split_edge(graph: RoadGraph, eid: EdgeId, offset_frac: float,
               new_node_id: Optional[NodeId] = None
               ) -> Tuple[NodeId, EdgeId, EdgeId]:
    """Split an edge (and its reverse twin) at a fractional offset, in place.

    Lengths and free-flow times are apportioned so that the two halves sum
    exactly to the original, and the polyline geometry is preserved.
    """
    f = _as_float(offset_frac, "offset_frac")
    _require(0.0 < f < 1.0, "offset_frac must lie strictly inside (0, 1).")
    e = graph.edge(eid)
    twin = e.reverse_eid if (e.reverse_eid is not None
                             and e.reverse_eid in graph._edges) else None
    head_geom, tail_geom, cut = _split_polyline(e.geometry, f)
    nid = new_node_id
    if nid is None:
        nid = "%s|%s@%d" % (str(e.u), str(e.v), eid)
    _require(not graph.has_node(nid), "node %r already exists." % (nid,))
    graph.add_node(nid, cut[0], cut[1])
    la = e.length_m * f
    lb = e.length_m - la
    ta = e.free_flow_time_s * f
    tb = e.free_flow_time_s - ta
    common = dict(lanes=e.lanes, capacity_veh_h=e.capacity_veh_h,
                  speed_limit_kph=e.speed_limit_kph, street_name=e.street_name,
                  highway_class=e.highway_class, gradient=e.gradient,
                  modes_permitted=e.modes_permitted)
    graph.remove_edge(eid)
    if twin is not None:
        graph.remove_edge(twin)
    a_id, _ = graph.add_edge(e.u, nid, la, oneway=True, free_flow_time_s=ta,
                             geometry=head_geom, **common)
    b_id, _ = graph.add_edge(nid, e.v, lb, oneway=True, free_flow_time_s=tb,
                             geometry=tail_geom, **common)
    if twin is not None:
        c_id, _ = graph.add_edge(e.v, nid, lb, oneway=True, free_flow_time_s=tb,
                                 geometry=list(reversed(tail_geom)), **common)
        d_id, _ = graph.add_edge(nid, e.u, la, oneway=True, free_flow_time_s=ta,
                                 geometry=list(reversed(head_geom)), **common)
        for x, y in ((a_id, d_id), (b_id, c_id)):
            graph.edge(x).reverse_eid = y
            graph.edge(y).reverse_eid = x
            graph.edge(x).oneway = graph.edge(y).oneway = False
    return nid, a_id, b_id


@dataclass
class Candidate:
    """One projection of a GPS fix onto a road edge, with its emission score."""

    obs_index: int
    eid: EdgeId
    point: Coord
    offset_frac: float
    distance_m: float
    emission_logp: float


@dataclass
class MapMatchResult:
    """The Viterbi solution of a Newson-Krumm hidden Markov map-matching problem."""

    chosen: List[Candidate]
    edges: List[EdgeId]
    nodes: List[NodeId]
    log_likelihood: float
    breaks: List[int]
    mean_error_m: float
    candidate_counts: List[int]

    @property
    def matched_points(self) -> List[Coord]:
        """The snapped position for each observation."""
        return [c.point for c in self.chosen]

    def summary(self) -> Dict[str, Any]:
        """Counts, likelihood and mean projection error."""
        return {"observations": len(self.chosen), "edges": len(self.edges),
                "logLikelihood": self.log_likelihood,
                "meanErrorM": self.mean_error_m, "breaks": list(self.breaks),
                "meanCandidates": (math.fsum(float(c) for c in self.candidate_counts)
                                   / len(self.candidate_counts))
                if self.candidate_counts else 0.0}


def _edge_bearing_at(edge: RoadEdge, frac: float) -> float:
    """Bearing of the geometry segment containing a fractional offset along an edge."""
    pts = edge.geometry
    if len(pts) < 2:
        return 0.0
    seg = [euclidean_m(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
           for i in range(len(pts) - 1)]
    total = math.fsum(seg)
    goal = max(0.0, min(1.0, frac)) * total
    for i, L in enumerate(seg):
        if math.fsum(seg[:i]) + L >= goal or i == len(seg) - 1:
            return bearing_deg(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
    return bearing_deg(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1])


def _route_distance(graph: RoadGraph, a: Candidate, b: Candidate,
                    trees: Dict[NodeId, SearchTree], weight: CostSpec,
                    mode: Optional[str]) -> float:
    """Network distance from one on-edge position to another (``inf`` if none)."""
    ea = graph.edge(a.eid)
    eb = graph.edge(b.eid)
    if a.eid == b.eid:
        if b.offset_frac >= a.offset_frac:
            return (b.offset_frac - a.offset_frac) * ea.length_m
    head = ea.v
    tail = eb.u
    if head not in trees:
        trees[head] = dijkstra(graph, head, weight=weight, mode=mode)
    mid = trees[head].dist.get(tail)
    if mid is None:
        return float("inf")
    return ((1.0 - a.offset_frac) * ea.length_m + mid
            + b.offset_frac * eb.length_m)


def map_match(graph: RoadGraph, points: Sequence[Any], *,
              sigma_z: float = 4.07, beta: float = 5.0,
              search_radius_m: float = 50.0, max_candidates: int = 8,
              mode: Optional[str] = None, weight: CostSpec = "length_m",
              index: Optional[EdgeIndex] = None, geographic: bool = False,
              heading_weight: float = 2.0,
              break_penalty: float = -50.0) -> MapMatchResult:
    """Newson-Krumm HMM map matching solved exactly by the Viterbi recursion.

    The emission model is a zero-mean Gaussian on the projection distance with
    standard deviation ``sigma_z`` (4.07 m in the original paper); the
    transition model is exponential in ``|great-circle distance - route
    distance|`` with scale ``beta``.  Both are evaluated in log space, so the
    returned ``log_likelihood`` is directly comparable between candidate routes.

    ``heading_weight`` adds a small directional prior
    ``-heading_weight * (1 - cos(delta))`` comparing the trace's local heading
    with the edge's bearing at the snapped point.  It costs nothing when the
    trace runs the right way and disambiguates the two directions of a two-way
    street when a fix happens to land on a junction; set it to ``0.0`` for the
    unmodified Newson-Krumm model.

    With ``geographic=True`` each observation is a ``(lat, lon)`` pair -- latitude FIRST,
    matching :func:`haversine_m` and ``add_node(lat=..., lon=...)`` -- and the graph must
    already have a geographic origin.  With ``geographic=False`` (the default) they are
    projected ``(x, y)`` metres in the graph's own frame.
    """
    sz = _as_positive(sigma_z, "sigma_z")
    bt = _as_positive(beta, "beta")
    radius = _as_positive(search_radius_m, "search_radius_m")
    kmax = _as_positive_int(max_candidates, "max_candidates")
    raw = _as_coords(points, "points")
    _require(len(raw) >= 1, "map_match needs at least one observation.")
    if geographic:
        # The trace is (lat, lon) and is projected with the GRAPH's origin.  Refuse a
        # graph that has no geographic origin: RoadGraph._project adopts its first
        # argument pair as the origin, so this otherwise read-only analytic used to MUTATE
        # the graph, anchor its projection on the GPS trace instead of the network, land
        # the trace at (0, 0) on top of the network by pure accident and return a
        # meaningless match -- and then permanently mis-project every later add_node.
        _require(graph._geo_ref is not None,
                 "map_match(geographic=True) needs a graph with a geographic origin "
                 "(build it with add_node(lat=..., lon=...)); this graph is planar only, "
                 "so pass already-projected (x, y) points with geographic=False.")
        obs: List[Coord] = [graph._project(p[0], p[1]) for p in raw]
    else:
        obs = list(raw)
    idx = index if index is not None else EdgeIndex(graph, mode=mode)
    hw = _as_non_negative(heading_weight, "heading_weight")
    headings: List[Optional[float]] = []
    for t in range(len(obs)):
        a = obs[t - 1] if t > 0 else obs[t]
        b = obs[t + 1] if t + 1 < len(obs) else obs[t]
        if a == b:
            headings.append(None)
        else:
            headings.append(bearing_deg(a[0], a[1], b[0], b[1]))
    emission_const = -math.log(_SQRT_2PI * sz)
    cand: List[List[Candidate]] = []
    for t, (px, py) in enumerate(obs):
        hits = idx.within(px, py, radius)
        if not hits:
            near = idx.nearest(px, py)
            hits = [near] if near is not None else []
        picked: List[Candidate] = []
        for s in hits[:kmax]:
            logp = emission_const - 0.5 * (s.distance_m / sz) ** 2
            if hw > 0.0 and headings[t] is not None:
                delta = bearing_difference_deg(
                    headings[t], _edge_bearing_at(graph.edge(s.eid),
                                                  s.offset_frac))
                logp -= hw * (1.0 - math.cos(math.radians(delta)))
            picked.append(Candidate(t, s.eid, s.point, s.offset_frac,
                                    s.distance_m, logp))
        _require(bool(picked),
                 "observation %d has no candidate edge within any radius." % t)
        cand.append(picked)
    trees: Dict[NodeId, SearchTree] = {}
    trans_const = -math.log(bt)
    scores: List[List[float]] = [[c.emission_logp for c in cand[0]]]
    back: List[List[int]] = [[-1] * len(cand[0])]
    broke: List[int] = []
    for t in range(1, len(obs)):
        gc = euclidean_m(obs[t - 1][0], obs[t - 1][1], obs[t][0], obs[t][1])
        row: List[float] = []
        ptr: List[int] = []
        any_feasible = False
        for j, cj in enumerate(cand[t]):
            best = -float("inf")
            best_i = 0
            for i, ci in enumerate(cand[t - 1]):
                rd = _route_distance(graph, ci, cj, trees, weight, mode)
                if rd == float("inf"):
                    continue
                lp = scores[t - 1][i] + trans_const - abs(gc - rd) / bt
                if lp > best:
                    best, best_i = lp, i
            if best == -float("inf"):
                for i in range(len(cand[t - 1])):
                    lp = scores[t - 1][i] + break_penalty
                    if lp > best:
                        best, best_i = lp, i
            else:
                any_feasible = True
            row.append(best + cj.emission_logp)
            ptr.append(best_i)
        if not any_feasible:
            broke.append(t)
        scores.append(row)
        back.append(ptr)
    last = max(range(len(scores[-1])), key=lambda j: scores[-1][j])
    chosen: List[Candidate] = [cand[len(obs) - 1][last]]
    j = last
    for t in range(len(obs) - 1, 0, -1):
        j = back[t][j]
        chosen.append(cand[t - 1][j])
    chosen.reverse()
    edges: List[EdgeId] = [chosen[0].eid]
    for a, b in zip(chosen, chosen[1:]):
        if a.eid == b.eid:
            continue
        ea = graph.edge(a.eid)
        eb = graph.edge(b.eid)
        if ea.v != eb.u:
            link = shortest_path(graph, ea.v, eb.u, weight=weight, mode=mode)
            if link is not None:
                for eid in link.edges:
                    if eid != edges[-1]:
                        edges.append(eid)
        if b.eid != edges[-1]:
            edges.append(b.eid)
    nodes = [graph.edge(edges[0]).u] + [graph.edge(e).v for e in edges]
    mean_err = math.fsum(c.distance_m for c in chosen) / len(chosen)
    return MapMatchResult(chosen, edges, nodes, scores[-1][last], broke,
                          mean_err, [len(c) for c in cand])


# ===========================================================================
# 16. COMPONENTS -- weak connectivity and Tarjan strongly connected components
# ===========================================================================
def weakly_connected_components(graph: RoadGraph) -> List[List[NodeId]]:
    """Components of the underlying undirected graph, largest first."""
    seen: Set[NodeId] = set()
    comps: List[List[NodeId]] = []
    for start in graph.nodes:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: List[NodeId] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in graph.neighbours(u):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    comps.sort(key=lambda c: (-len(c), repr(c[0])))
    return comps


def strongly_connected_components(graph: RoadGraph) -> List[List[NodeId]]:
    """Tarjan's strongly connected components (iterative), largest first."""
    index: Dict[NodeId, int] = {}
    low: Dict[NodeId, int] = {}
    on_stack: Set[NodeId] = set()
    stack: List[NodeId] = []
    counter = [0]
    comps: List[List[NodeId]] = []
    for root in graph.nodes:
        if root in index:
            continue
        work: List[Tuple[NodeId, int]] = [(root, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                index[v] = low[v] = counter[0]
                counter[0] += 1
                stack.append(v)
                on_stack.add(v)
            recursed = False
            succ = graph.successors(v)
            while pi < len(succ):
                w = succ[pi]
                pi += 1
                if w not in index:
                    work[-1] = (v, pi)
                    work.append((w, 0))
                    recursed = True
                    break
                if w in on_stack:
                    low[v] = min(low[v], index[w])
            if recursed:
                continue
            work[-1] = (v, pi)
            work.pop()
            if work:
                pv = work[-1][0]
                low[pv] = min(low[pv], low[v])
            if low[v] == index[v]:
                comp: List[NodeId] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                comps.append(comp)
    comps.sort(key=lambda c: (-len(c), repr(c[0])))
    return comps


# ===========================================================================
# 17. SYNTHETIC GENERATORS -- grid, radial and random planar networks
# ===========================================================================
def grid_network(rows: int, cols: int, spacing_m: float = 1.0, *,
                 oneway: bool = False, lanes: int = 1,
                 speed_limit_kph: float = 32.0,
                 highway_class: str = "residential",
                 jitter_m: float = 0.0, seed: int = 0,
                 signalise_every: int = 0, name: str = "grid") -> RoadGraph:
    """A ``rows x cols`` lattice; node ids are ``r * cols + c``.

    With ``jitter_m == 0`` every edge is exactly ``spacing_m`` long, so the
    shortest path between opposite corners is exactly
    ``(rows - 1 + cols - 1) * spacing_m`` and the network has
    ``(rows - 1) * (cols - 1)`` blocks.  ``oneway=True`` lays out a Manhattan
    alternating one-way system.
    """
    r = _as_positive_int(rows, "rows")
    c = _as_positive_int(cols, "cols")
    sp = _as_positive(spacing_m, "spacing_m")
    jit = _as_non_negative(jitter_m, "jitter_m")
    rng = random.Random(seed)
    g = RoadGraph(name)
    for i in range(r):
        for j in range(c):
            dx = rng.uniform(-jit, jit) if jit > 0.0 else 0.0
            dy = rng.uniform(-jit, jit) if jit > 0.0 else 0.0
            sig = bool(signalise_every) and ((i * c + j) % signalise_every == 0)
            g.add_node(i * c + j, j * sp + dx, i * sp + dy, signalised=sig,
                       row=i, col=j)

    def link(a: int, b: int, nm: str, way: bool) -> None:
        length = None if jit > 0.0 else sp
        g.add_edge(a, b, length, oneway=way, lanes=lanes,
                   speed_limit_kph=speed_limit_kph, street_name=nm,
                   highway_class=highway_class)

    for i in range(r):
        for j in range(c - 1):
            a, b = i * c + j, i * c + j + 1
            nm = "Street %d" % i
            if not oneway:
                link(a, b, nm, False)
            elif i % 2 == 0:
                link(a, b, nm, True)
            else:
                link(b, a, nm, True)
    for j in range(c):
        for i in range(r - 1):
            a, b = i * c + j, (i + 1) * c + j
            nm = "Avenue %d" % j
            if not oneway:
                link(a, b, nm, False)
            elif j % 2 == 0:
                link(a, b, nm, True)
            else:
                link(b, a, nm, True)
    return g


def radial_network(num_rings: int, num_spokes: int, ring_spacing_m: float = 100.0,
                   *, include_rings: bool = True, oneway: bool = False,
                   jitter_m: float = 0.0, seed: int = 0,
                   name: str = "radial") -> RoadGraph:
    """A hub-and-spoke town: node ``0`` is the centre, then ring by ring outwards."""
    R = _as_positive_int(num_rings, "num_rings")
    S = _as_positive_int(num_spokes, "num_spokes")
    _require(S >= 3, "num_spokes must be at least 3.")
    sp = _as_positive(ring_spacing_m, "ring_spacing_m")
    jit = _as_non_negative(jitter_m, "jitter_m")
    rng = random.Random(seed)
    g = RoadGraph(name)
    g.add_node(0, 0.0, 0.0, ring=0, spoke=-1, signalised=True)

    def nid(ring: int, spoke: int) -> int:
        return 1 + (ring - 1) * S + (spoke % S)

    for ring in range(1, R + 1):
        for s in range(S):
            ang = 2.0 * math.pi * s / S
            rad = ring * sp
            dx = rng.uniform(-jit, jit) if jit > 0.0 else 0.0
            dy = rng.uniform(-jit, jit) if jit > 0.0 else 0.0
            g.add_node(nid(ring, s), rad * math.sin(ang) + dx,
                       rad * math.cos(ang) + dy, ring=ring, spoke=s)
    for s in range(S):
        g.add_edge(0, nid(1, s), oneway=oneway, street_name="Radial %d" % s,
                   highway_class="primary")
        for ring in range(1, R):
            g.add_edge(nid(ring, s), nid(ring + 1, s), oneway=oneway,
                       street_name="Radial %d" % s, highway_class="primary")
    if include_rings:
        for ring in range(1, R + 1):
            for s in range(S):
                g.add_edge(nid(ring, s), nid(ring, s + 1), oneway=oneway,
                           street_name="Ring %d" % ring,
                           highway_class="secondary")
    return g


def random_planar_network(num_nodes: int, width_m: float = 1000.0,
                          height_m: float = 1000.0, *, seed: int = 0,
                          prune_fraction: float = 0.0,
                          name: str = "random-planar") -> RoadGraph:
    """Seeded Poisson points triangulated by Delaunay, optionally thinned.

    The Delaunay triangulation is planar by construction, so the result is a
    legitimate (if idealised) street network.  ``prune_fraction`` removes that
    share of the longest-first shuffled edges, keeping the graph connected.
    """
    n = _as_positive_int(num_nodes, "num_nodes")
    _require(n >= 3, "num_nodes must be at least 3.")
    w = _as_positive(width_m, "width_m")
    h = _as_positive(height_m, "height_m")
    pf = _as_non_negative(prune_fraction, "prune_fraction")
    _require(pf < 1.0, "prune_fraction must be below 1.")
    rng = random.Random(seed)
    pts: List[Coord] = []
    guard = 0
    while len(pts) < n and guard < 100 * n:
        guard += 1
        p = (round(rng.uniform(0.0, w), 6), round(rng.uniform(0.0, h), 6))
        if p not in pts:
            pts.append(p)
    g = RoadGraph(name)
    for i, p in enumerate(pts):
        g.add_node(i, p[0], p[1])
    tris = delaunay_triangulation(pts)
    undirected: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    for (a, b, c) in tris:
        for (i, j) in ((a, b), (b, c), (c, a)):
            key = (i, j) if i < j else (j, i)
            if key not in seen:
                seen.add(key)
                undirected.append(key)
    undirected.sort()
    order = list(undirected)
    rng.shuffle(order)
    drop = int(round(pf * len(order)))
    keep = set(undirected)
    if drop > 0:
        degree: Dict[int, int] = {}
        for (i, j) in undirected:
            degree[i] = degree.get(i, 0) + 1
            degree[j] = degree.get(j, 0) + 1
        removed = 0
        for (i, j) in order:
            if removed >= drop:
                break
            if degree[i] <= 2 or degree[j] <= 2:
                continue
            keep.discard((i, j))
            degree[i] -= 1
            degree[j] -= 1
            removed += 1
    classes = ("primary", "secondary", "residential")
    for k, (i, j) in enumerate(undirected):
        if (i, j) not in keep:
            continue
        g.add_edge(i, j, oneway=False, street_name="Road %d" % k,
                   highway_class=classes[k % len(classes)])
    comps = weakly_connected_components(g)
    if len(comps) > 1:
        g = g.subgraph(comps[0])
    return g


# ===========================================================================
# 18. PAYLOADS -- GeoJSON and frontend-ready dictionaries
# ===========================================================================
def _json_num(x: Any) -> Any:
    """Round a float for transport, mapping non-finite values to ``None``."""
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return round(x, 6)
    return x


def to_geojson(graph: RoadGraph, *, use_latlon: Optional[bool] = None,
               include_nodes: bool = True) -> Dict[str, Any]:
    """GeoJSON ``FeatureCollection`` of the street segments (and junctions)."""
    geo = all(n.is_geographic for n in graph.iter_nodes()) \
        if use_latlon is None else bool(use_latlon)
    ref = graph._geo_ref
    features: List[Dict[str, Any]] = []

    def to_lonlat(p: Coord) -> List[Any]:
        if not geo or ref is None:
            return [_json_num(p[0]), _json_num(p[1])]
        lat0, lon0 = ref
        k = math.cos(math.radians(lat0))
        lon = lon0 + math.degrees(p[0] / (_EARTH_RADIUS_M * k)) if abs(k) > _EPS \
            else lon0
        lat = lat0 + math.degrees(p[1] / _EARTH_RADIUS_M)
        return [_json_num(lon), _json_num(lat)]

    for (u, v, eid) in graph.undirected_segments():
        e = graph.edge(eid)
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [to_lonlat(p) for p in e.geometry]},
            "properties": {
                "edgeId": eid, "u": str(u), "v": str(v),
                "name": e.street_name, "highway": e.highway_class,
                "lengthM": _json_num(e.length_m), "lanes": e.lanes,
                "oneway": e.oneway,
                "capacityVehH": _json_num(e.capacity_veh_h),
                "freeFlowTimeS": _json_num(e.free_flow_time_s),
                "speedLimitKph": _json_num(e.speed_limit_kph),
                "gradient": _json_num(e.gradient),
                "modes": list(e.modes_permitted),
            },
        })
    if include_nodes:
        for n in graph.iter_nodes():
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": to_lonlat(n.xy)},
                "properties": {"nodeId": str(n.node_id),
                               "degree": graph.degree(n.node_id),
                               "signalised": n.signalised},
            })
    return {"type": "FeatureCollection", "features": features}


def build_street_network_payload(graph: RoadGraph, *, metrics: bool = True,
                                 num_bins: int = 36, seed: int = 0,
                                 circuity_pairs: int = 100) -> Dict[str, Any]:
    """Frontend-ready summary: topology, attribution and the metric battery."""
    nodes = [{"id": str(n.node_id), "x": _json_num(n.x), "y": _json_num(n.y),
              "lat": _json_num(n.lat), "lon": _json_num(n.lon),
              "degree": graph.degree(n.node_id),
              "signalised": n.signalised} for n in graph.iter_nodes()]
    edges = []
    for (u, v, eid) in graph.undirected_segments():
        e = graph.edge(eid)
        edges.append({"id": eid, "source": str(u), "target": str(v),
                      "name": e.street_name, "highwayClass": e.highway_class,
                      "lengthM": _json_num(e.length_m), "lanes": e.lanes,
                      "oneway": e.oneway,
                      "capacityVehH": _json_num(e.capacity_veh_h),
                      "freeFlowTimeS": _json_num(e.free_flow_time_s),
                      "speedLimitKph": _json_num(e.speed_limit_kph)})
    minx, miny, maxx, maxy = graph.bbox()
    payload: Dict[str, Any] = {
        "name": graph.name,
        "driveSide": graph.drive_side,
        "numNodes": graph.num_nodes,
        "numEdges": graph.num_edges,
        "numSegments": len(edges),
        "numTurnRestrictions": len(graph.turn_restrictions),
        "totalLengthM": _json_num(graph.total_length_m),
        "streetLengthM": _json_num(graph.street_length_m),
        "bbox": {"minX": _json_num(minx), "minY": _json_num(miny),
                 "maxX": _json_num(maxx), "maxY": _json_num(maxy)},
        "nodes": nodes,
        "edges": edges,
    }
    if metrics:
        stats = edge_length_statistics(graph)
        blocks = detect_blocks(graph)
        payload["metrics"] = {
            "intersectionDensityPerKm2":
                _json_num(intersection_density(graph)),
            "degreeDistribution": dict((str(k), _json_num(v)) for k, v
                                       in node_degree_distribution(graph).items()),
            "meanSegmentLengthM": _json_num(stats["mean_m"]),
            "medianSegmentLengthM": _json_num(stats["median_m"]),
            "orientationEntropy":
                _json_num(street_orientation_entropy(graph, num_bins)),
            "orientationOrder": _json_num(orientation_order(graph, num_bins)),
            "orientationHistogram":
                [_json_num(p) for p in orientation_histogram(graph, num_bins)],
            "circuity": _json_num(circuity(graph, num_pairs=circuity_pairs,
                                           seed=seed)),
            "numBlocks": len(blocks),
            "blockStatistics": dict((k, _json_num(v)) for k, v
                                    in block_statistics(blocks).items()),
            "weakComponents": len(weakly_connected_components(graph)),
            "strongComponents": len(strongly_connected_components(graph)),
        }
    return payload


def build_a2ui_payload(graph: RoadGraph, *, path: Optional[Path] = None,
                       iso: Optional[Isochrone] = None,
                       node_values: Optional[Dict[NodeId, float]] = None,
                       label: str = "Street network") -> Dict[str, Any]:
    """Map-visualisation payload: polylines, markers, overlays and stat cards."""
    polylines = []
    for (u, v, eid) in graph.undirected_segments():
        e = graph.edge(eid)
        polylines.append({"id": eid, "name": e.street_name,
                          "highwayClass": e.highway_class,
                          "points": [[_json_num(p[0]), _json_num(p[1])]
                                     for p in e.geometry]})
    vals = node_values or {}
    lo = min(vals.values()) if vals else 0.0
    hi = max(vals.values()) if vals else 0.0
    span = (hi - lo) if (hi - lo) > 0.0 else 1.0
    markers = []
    for n in graph.iter_nodes():
        v = vals.get(n.node_id)
        markers.append({"id": str(n.node_id), "x": _json_num(n.x),
                        "y": _json_num(n.y),
                        "degree": graph.degree(n.node_id),
                        "signalised": n.signalised,
                        "value": _json_num(v),
                        "intensity": _json_num((v - lo) / span)
                        if v is not None else None})
    overlays: List[Dict[str, Any]] = []
    if path is not None:
        overlays.append({"kind": "route",
                         "points": [[_json_num(p[0]), _json_num(p[1])]
                                    for p in path.geometry],
                         "lengthM": _json_num(path.length_m),
                         "travelTimeS": _json_num(path.travel_time_s)})
    if iso is not None:
        overlays.append({"kind": "isochrone",
                         "budget": _json_num(iso.budget),
                         "weight": iso.weight,
                         "rings": [[[_json_num(p[0]), _json_num(p[1])]
                                    for p in ring] for ring in iso.polygon],
                         "areaM2": _json_num(iso.area_m2)})
    stats = edge_length_statistics(graph)
    cards = [
        {"label": "Junctions", "value": graph.num_nodes},
        {"label": "Segments", "value": len(polylines)},
        {"label": "Street km", "value": round(graph.street_length_m / 1000.0, 2)},
        {"label": "Mean segment m", "value": round(stats["mean_m"], 1)},
        {"label": "Orientation order", "value": round(orientation_order(graph), 3)},
    ]
    return {"label": label, "polylines": polylines, "markers": markers,
            "overlays": overlays, "statCards": cards}


# ===========================================================================
# 19. DEMO
# ===========================================================================
def _demo() -> None:
    print("=" * 70)
    print("  URBAN STREET NETWORK (pure Python) - demo")
    print("=" * 70)

    print("\n-- [1] geometry & geodesy --")
    d_lon_par = haversine_m(51.5074, -0.1278, 48.8566, 2.3522)
    print("   London -> Paris great circle = %.1f km" % (d_lon_par / 1000.0))
    print("   euclidean((0,0),(3,4)) = %.1f   bearing = %.1f deg"
          % (euclidean_m(0.0, 0.0, 3.0, 4.0), bearing_deg(0.0, 0.0, 3.0, 4.0)))
    print("   deflection 350 -> 010 = %+.1f deg (right turn through north)"
          % bearing_difference_deg(350.0, 10.0))
    zig = [(0.0, 0.0), (1.0, 0.1), (2.0, 0.0), (3.0, 0.1), (4.0, 0.0)]
    print("   polyline length = %.4f, simplified %d -> %d vertices"
          % (polyline_length(zig), len(zig), len(douglas_peucker(zig, 0.5))))
    dseg = point_segment_distance(0.5, 1.0, 0.0, 0.0, 1.0, 0.0)
    proj = project_point_on_polyline(2.5, 1.0, zig)
    print("   point-segment dist = %.3f at t=%.2f; polyline proj dist = %.3f"
          " at frac %.3f" % (dseg[0], dseg[1], proj[0], proj[2]))
    sq = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    print("   square area = %.1f, centroid = (%.1f, %.1f), (2,2) inside = %s"
          % (polygon_area(sq), polygon_centroid(sq)[0], polygon_centroid(sq)[1],
             point_in_polygon((2.0, 2.0), sq)))
    hull = convex_hull(sq + [(2.0, 2.0)])
    print("   convex hull of 5 points keeps %d vertices" % len(hull))
    cloud = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0),
             (2.0, 1.0), (0.0, 2.0), (1.0, 2.0), (2.0, 2.0)]
    tris = delaunay_triangulation(cloud)
    shape = alpha_shape(cloud, 1.5)
    print("   Delaunay on 3x3 lattice -> %d triangles (expect 8); alpha-shape"
          " ring area = %.2f" % (len(tris), polygon_area(shape[0])))

    print("\n-- [2] the road graph & the analytic grid oracle --")
    k = 6
    grid = grid_network(k, k, 1.0, name="unit-grid")
    corner_a, corner_b = 0, k * k - 1
    p = shortest_path(grid, corner_a, corner_b)
    blocks = detect_blocks(grid)
    print("   %d x %d grid: %d nodes, %d directed edges, %d segments"
          % (k, k, grid.num_nodes, grid.num_edges,
             len(grid.undirected_segments())))
    print("   corner-to-corner cost = %.6f  (analytic 2(k-1) = %d)"
          % (p.cost, 2 * (k - 1)))
    print("   blocks detected = %d  (analytic (k-1)^2 = %d), mean area = %.3f"
          % (len(blocks), (k - 1) ** 2, block_statistics(blocks)["mean_m2"]))
    print("   orientation entropy = %.6f (ln 4 = %.6f) -> order phi = %.6f"
          % (street_orientation_entropy(grid), math.log(4.0),
             orientation_order(grid)))
    print("   corner-pair circuity = %.6f (analytic sqrt(2) = %.6f)"
          % (circuity(grid, pairs=[(corner_a, corner_b)]), math.sqrt(2.0)))
    print("   node/edge types: %s / %s"
          % (type(grid.node(0)).__name__, type(grid.edge(0)).__name__))
    print("   highway classes known = %d, modes = %s"
          % (len(HIGHWAY_DEFAULTS), ",".join(MODES)))

    print("\n-- [3] radial town contrasts with the grid --")
    radial = radial_network(4, 12, 100.0, name="radial-town")
    print("   radial: %d nodes, %d segments, entropy = %.4f, order phi = %.4f"
          % (radial.num_nodes, len(radial.undirected_segments()),
             street_orientation_entropy(radial), orientation_order(radial)))
    hist = orientation_histogram(radial)
    print("   orientation histogram occupies %d/%d bins (grid occupies %d)"
          % (sum(1 for h in hist if h > 1e-12), len(hist),
             sum(1 for h in orientation_histogram(grid) if h > 1e-12)))
    print("   intersection density = %.1f /km2 (grid %.1f /km2)"
          % (intersection_density(radial),
             intersection_density(grid_network(k, k, 100.0))))
    print("   degree distribution (radial) = %s"
          % dict((kk, round(vv, 3))
                 for kk, vv in node_degree_distribution(radial).items()))
    els = edge_length_statistics(radial)
    print("   segment length mean/median/max = %.1f / %.1f / %.1f m"
          % (els["mean_m"], els["median_m"], els["max_m"]))

    print("\n-- [4] routing agreement: Dijkstra == A* == bidirectional == ALT == CH --")
    town = random_planar_network(70, 1200.0, 1200.0, seed=7, prune_fraction=0.15,
                                 name="random-town")
    town = largest_component_subgraph(town, strong=True)
    alt = ALTIndex(town, 6, weight="length_m", seed=3)
    ch = ContractionHierarchy(town, weight="length_m")
    rng = random.Random(11)
    nodes = town.nodes
    agree_a = agree_b = agree_alt = agree_ch = 0
    trials = 0
    for _ in range(300):
        s = nodes[rng.randrange(len(nodes))]
        t = nodes[rng.randrange(len(nodes))]
        if s == t:
            continue
        base = shortest_path(town, s, t)
        if base is None:
            continue
        trials += 1
        agree_a += int(astar(town, s, t).cost == base.cost)
        agree_b += int(bidirectional_dijkstra(town, s, t).cost == base.cost)
        agree_alt += int(alt_shortest_path(town, alt, s, t).cost == base.cost)
        agree_ch += int(ch.distance(s, t) == base.cost)
    print("   town: %d nodes, %d edges; %d random pairs tested"
          % (town.num_nodes, town.num_edges, trials))
    print("   exact bit-for-bit agreement  A*=%d  bidir=%d  ALT=%d  CH=%d  (of %d)"
          % (agree_a, agree_b, agree_alt, agree_ch, trials))
    print("   ALT: %s; CH: %s" % (alt.summary()["numLandmarks"], ch.summary()))
    tree = dijkstra(town, nodes[0])
    print("   Dijkstra tree from %r settles %d nodes; SearchTree type = %s"
          % (nodes[0], len(tree.reached), type(tree).__name__))
    print("   shortest_path_length(%r -> %r) = %.3f m"
          % (nodes[0], nodes[-1], shortest_path_length(town, nodes[0], nodes[-1])))
    closure = metric_closure(town, nodes[:6])
    worst = 0.0
    for a in nodes[:6]:
        for b in nodes[:6]:
            for c in nodes[:6]:
                ab = closure.get((a, b))
                bc = closure.get((b, c))
                ac = closure.get((a, c))
                if ab is not None and bc is not None and ac is not None:
                    worst = max(worst, ac - (ab + bc))
    print("   metric closure on 6 nodes: worst triangle violation = %.3e" % worst)

    print("\n-- [5] k-shortest paths --")
    yen = yen_k_shortest_paths(town, nodes[0], nodes[-1], 5)
    print("   Yen on the town: %d paths, costs = %s, all loopless = %s"
          % (len(yen), [round(x.cost, 1) for x in yen],
             all(x.is_loopless() for x in yen)))
    print("   costs non-decreasing = %s, all distinct = %s"
          % (all(yen[i].cost <= yen[i + 1].cost + 1e-12
                 for i in range(len(yen) - 1)),
             len(set(x.key() for x in yen)) == len(yen)))
    epp = eppstein_k_shortest_paths(town, nodes[0], nodes[-1], 5)
    print("   Eppstein-style: %d paths, costs = %s"
          % (len(epp), [round(x.cost, 1) for x in epp]))
    ygrid = yen_k_shortest_paths(grid, 0, k * k - 1, 4)
    print("   Yen on the tie-rich grid: costs = %s, distinct edge sets = %d"
          % ([round(x.cost, 3) for x in ygrid],
             len(set(x.key() for x in ygrid))))

    print("\n-- [6] turn restrictions change the route --")
    tg = RoadGraph("banned-turn", drive_side="left")
    tg.add_node("W", -100.0, 0.0)
    tg.add_node("C", 0.0, 0.0, signalised=True)
    tg.add_node("E", 100.0, 0.0)
    tg.add_node("N", 0.0, 100.0)
    tg.add_node("D", -160.0, 160.0)
    wc, cw = tg.add_edge("W", "C", 100.0, oneway=False, street_name="High Street")
    ce, ec = tg.add_edge("C", "E", 100.0, oneway=False, street_name="High Street")
    cn, nc = tg.add_edge("C", "N", 100.0, oneway=False, street_name="North Road")
    tg.add_edge("W", "D", 160.0, oneway=False, street_name="Bypass")
    tg.add_edge("D", "N", 160.0, oneway=False, street_name="Bypass")
    free = turn_aware_shortest_path(tg, "W", "N", weight="length_m")
    tcm = TurnCostModel.for_drive_side("left")
    print("   deflection W->C then C->N = %+.1f deg -> %s (penalty %.1f s)"
          % (tg.turn_deflection_deg(wc, cn),
             tcm.classify(tg.turn_deflection_deg(wc, cn)),
             tcm.penalty(tg.turn_deflection_deg(wc, cn), True)))
    restriction = tg.add_turn_restriction(wc, "C", cn, "no_turn", "no left turn")
    exp = edge_expanded_graph(tg, weight="length_m", turn_cost=TurnCostModel.free())
    banned = turn_aware_shortest_path(tg, "W", "N", weight="length_m",
                                      expanded=exp)
    print("   unrestricted route = %s (%.0f m); restricted route = %s (%.0f m)"
          % ("-".join(str(x) for x in free.nodes), free.cost,
             "-".join(str(x) for x in banned.nodes), banned.cost))
    print("   dual graph: %d nodes, %d arcs, %d turns banned; restriction kind %r"
          % (exp.num_nodes, exp.num_arcs, exp.banned_turns, restriction.kind))
    print("   plain Dijkstra ignores the ban and still returns %.0f m"
          % shortest_path(tg, "W", "N").cost)

    print("\n-- [7] time-dependent routing (a rush hour on one link) --")
    corridor = grid_network(1, 5, 500.0, speed_limit_kph=50.0, name="corridor")
    rush = TravelTimeProfile((0.0, 1800.0, 3600.0, 7200.0),
                             (36.0, 36.0, 200.0, 36.0), period_s=86400.0)
    profiles = dict((e, rush) for e in corridor.edges_between(1, 2))
    early = time_dependent_shortest_path(corridor, 0, 4, 0.0, profiles=profiles)
    peak = time_dependent_shortest_path(corridor, 0, 4, 3600.0, profiles=profiles)
    print("   profile: %s" % rush.summary())
    print("   depart 00:00 -> %.1f s; depart 01:00 (peak) -> %.1f s"
          % (early.cost, peak.cost))
    td = time_dependent_dijkstra(corridor, 0, 3600.0, profiles=profiles)
    print("   arrival tree settles %d nodes; node 4 delay = %.1f s"
          % (len(td.reached), td.dist[4] - early.cost))

    print("\n-- [8] isochrones --")
    city = grid_network(9, 9, 120.0, speed_limit_kph=32.0, name="iso-grid")
    iso = isochrone(city, 40, 60.0, weight="free_flow_time_s", alpha_m=200.0)
    print("   60-second isochrone from the centre: %d nodes, %d whole edges,"
          " %d partial" % (iso.num_nodes, len(iso.edges), len(iso.partial_edges)))
    print("   reachable street length = %.0f m; polygon rings = %d, area = %.0f m2"
          % (iso.reached_length_m(city), len(iso.polygon), iso.area_m2))
    sub = reachable_subgraph(city, 40, 60.0, weight="free_flow_time_s")
    print("   reachable subgraph: %d nodes, %d edges"
          % (sub.num_nodes, sub.num_edges))

    print("\n-- [9] centrality oracles --")
    n_star = 9
    star = RoadGraph("star")
    star.add_node(0, 0.0, 0.0)
    for i in range(1, n_star):
        ang = 2.0 * math.pi * (i - 1) / (n_star - 1)
        star.add_node(i, 100.0 * math.sin(ang), 100.0 * math.cos(ang))
        star.add_edge(0, i, 100.0, oneway=False)
    bc_star = betweenness_centrality(star, weight="hops", directed=False)
    print("   star K_1,%d centre betweenness = %.1f (analytic %.1f), leaf = %.1f"
          % (n_star - 1, bc_star[0],
             (n_star - 1) * (n_star - 2) / 2.0, bc_star[1]))
    n_path = 7
    line = RoadGraph("path")
    for i in range(n_path):
        line.add_node(i, 100.0 * i, 0.0)
    for i in range(n_path - 1):
        line.add_edge(i, i + 1, 100.0, oneway=False)
    bc_line = betweenness_centrality(line, weight="hops", directed=False)
    print("   path P_%d betweenness = %s (analytic %s)"
          % (n_path, [round(bc_line[i], 1) for i in range(n_path)],
             [float(i * (n_path - 1 - i)) for i in range(n_path)]))
    eb = edge_betweenness_centrality(line, weight="hops", directed=False)
    print("   edge betweenness of the middle link = %.1f (analytic %.1f)"
          % (eb[line.edges_between(3, 4)[0]], 4.0 * 3.0))
    od = {(0, k * k - 1): 1000.0, (k - 1, k * k - k): 500.0}
    nflow, eflow = demand_weighted_betweenness(grid, od, weight="length_m")
    print("   OD flow: total assigned to edges = %.1f veh-edges, busiest node"
          " carries %.1f" % (math.fsum(eflow.values()), max(nflow.values())))
    clo = closeness_centrality(grid, weight="length_m")
    har = closeness_centrality(grid, weight="length_m", variant="harmonic")
    stg = straightness_centrality(grid)
    info = information_centrality(grid)
    print("   grid centre closeness = %.5f, harmonic = %.5f, straightness = %.4f,"
          " information = %.4f"
          % (clo[14], har[14], stg[14], info[14]))
    print("   grid corner closeness = %.5f (<= centre: %s)"
          % (clo[0], clo[0] <= clo[14]))

    print("\n-- [10] blocks, strokes and space syntax --")
    print("   grid blocks: %s"
          % dict((kk, round(vv, 3))
                 for kk, vv in block_statistics(blocks).items()))
    print("   average block size = %.6f m2 (analytic spacing^2 = %.1f)"
          % (average_block_size(grid), 1.0))
    print("   block 0 has %d sides, compactness %.3f, centroid (%.2f, %.2f)"
          % (blocks[0].num_sides, blocks[0].compactness,
             blocks[0].centroid[0], blocks[0].centroid[1]))
    strokes = generate_strokes(grid, 45.0)
    print("   strokes on the %dx%d grid = %d (analytic 2k = %d), mean length"
          " = %.2f" % (k, k, len(strokes), 2 * k,
                       math.fsum(s.length_m for s in strokes) / len(strokes)))
    dual = segment_dual_graph(grid)
    print("   segment dual: %d nodes, %d arcs"
          % (len(dual), sum(len(v) for v in dual.values())))
    ss = space_syntax(grid, unit="stroke")
    ss_seg = space_syntax(grid, unit="segment", radius=3)
    print("   stroke space syntax: %s" % ss.summary())
    print("   most integrated strokes: %s"
          % [(sid, round(val, 3)) for sid, val in ss.most_integrated(3)])
    print("   radius-3 segment analysis: mean integration = %.4f,"
          " mean depth range = %.2f..%.2f"
          % (ss_seg.summary()["meanIntegration"],
             min(ss_seg.mean_depth.values()), max(ss_seg.mean_depth.values())))
    print("   SpaceSyntaxResult fields present: %s"
          % (len(ss.choice) == len(ss.connectivity) == len(ss.control)
             == len(ss.relative_asymmetry) == len(ss.real_relative_asymmetry)
             == len(ss.total_depth) == len(ss.node_count) == len(ss.members)))

    print("\n-- [11] snapping, edge splitting and HMM map matching --")
    mm_grid = grid_network(6, 6, 100.0, speed_limit_kph=32.0, name="mm-grid")
    idx = EdgeIndex(mm_grid)
    snap = snap_point(mm_grid, 155.0, 3.0, index=idx)
    print("   index: %s" % idx.summary())
    print("   snap (155, 3) -> edge %d at frac %.3f, %.2f m away"
          % (snap.eid, snap.offset_frac, snap.distance_m))
    split_graph = mm_grid.copy()
    before = split_graph.total_length_m
    new_node, ea, eb2 = split_edge(split_graph, split_graph.edges_between(0, 1)[0],
                                   0.35)
    print("   split edge at 0.35 -> node %r, halves %.1f m + %.1f m;"
          " total length preserved = %s"
          % (new_node, split_graph.edge(ea).length_m,
             split_graph.edge(eb2).length_m,
             abs(split_graph.total_length_m - before) < 1e-9))
    truth = shortest_path(mm_grid, 0, 35)
    noise = random.Random(2024)
    samples: List[Coord] = []
    step_m = 25.0
    n_samp = int(truth.length_m / step_m)
    for i in range(n_samp):
        frac = (0.5 * step_m + i * step_m) / truth.length_m
        px, py = _point_at_offset(truth.geometry, frac)
        samples.append((px + noise.gauss(0.0, 3.0), py + noise.gauss(0.0, 3.0)))
    match = map_match(mm_grid, samples, sigma_z=4.07, beta=10.0,
                      search_radius_m=40.0, index=idx)
    recovered = match.edges == truth.edges
    print("   true path has %d edges; matched %d edges; exact recovery = %s"
          % (len(truth.edges), len(match.edges), recovered))
    print("   map match: %s" % match.summary())
    print("   first candidate: edge %d, %.2f m off, emission logp %.3f"
          % (match.chosen[0].eid, match.chosen[0].distance_m,
             match.chosen[0].emission_logp))

    print("\n-- [12] cleaning, components and conservation --")
    messy = grid_network(4, 4, 100.0, name="messy")
    messy.add_edge(5, 5, 20.0, oneway=True, street_name="Loop")
    messy.add_edge(5, 6, 250.0, oneway=True, street_name="Street 1")
    messy.add_node(99, 250.0, -100.0)
    messy.add_edge(99, 99, 5.0, oneway=True, street_name="Orphan loop")
    total_before = messy.street_length_m
    no_loops = remove_self_loops(messy)
    consolidated = consolidate_parallel_edges(no_loops)
    simplified = simplify_degree2_chains(consolidated)
    print("   messy: %d edges; after loop removal %d; after consolidation %d;"
          " after simplification %d"
          % (messy.num_edges, no_loops.num_edges, consolidated.num_edges,
             simplified.num_edges))
    print("   street length %.1f -> %.1f m (%.1f m of loops and parallels dropped)"
          % (total_before, consolidated.street_length_m,
             total_before - consolidated.street_length_m))
    chain = RoadGraph("long-road")
    for i in range(6):
        chain.add_node(i, 100.0 * i, 6.0 * i * (i - 5))
    for i in range(5):
        chain.add_edge(i, i + 1, oneway=False, street_name="Long Road",
                       highway_class="secondary")
    chain.add_node(99, 250.0, 400.0)
    chain.add_edge(2, 99, oneway=False, street_name="Side Street")
    chain_simple = simplify_degree2_chains(chain)
    print("   'Long Road' chain: %d nodes/%d edges -> %d nodes/%d edges"
          % (chain.num_nodes, chain.num_edges, chain_simple.num_nodes,
             chain_simple.num_edges))
    print("   simplification conserves length: %.9f == %.9f -> %s"
          % (chain.total_length_m, chain_simple.total_length_m,
             abs(chain.total_length_m - chain_simple.total_length_m) < 1e-9))
    merged = max(chain_simple.iter_edges(),
                 key=lambda e: len(e.tags.get("merged_eids", [])))
    print("   longest merged edge: %.1f m, %d geometry vertices, %d constituents"
          % (merged.length_m, len(merged.geometry),
             len(merged.tags.get("merged_eids", []))))
    print("   grid simplification is a no-op (names differ at every corner): %s"
          % (consolidated.num_edges == simplified.num_edges))
    cleaned = clean_network(messy, largest_component=True)
    print("   clean_network -> %d nodes, %d edges (orphan dropped = %s)"
          % (cleaned.num_nodes, cleaned.num_edges, not cleaned.has_node(99)))
    wcc = weakly_connected_components(messy)
    scc = strongly_connected_components(messy)
    print("   components: %d weak (sizes %s), %d strong (largest %d)"
          % (len(wcc), [len(c) for c in wcc], len(scc), len(scc[0])))
    oneway_grid = grid_network(4, 4, 100.0, oneway=True, name="manhattan")
    twoway_grid = grid_network(4, 4, 100.0, name="two-way")
    print("   Manhattan one-way grid: %d edges (two-way grid has %d);"
          " %d strong components (corner sources/sinks are singletons)"
          % (oneway_grid.num_edges, twoway_grid.num_edges,
             len(strongly_connected_components(oneway_grid))))

    print("\n-- [13] payloads --")
    payload = build_street_network_payload(grid, circuity_pairs=40)
    ui = build_a2ui_payload(grid, path=p, iso=iso, node_values=bc_star)
    gj = to_geojson(grid)
    print("   payload keys = %d, metric keys = %d, nodes = %d, edges = %d"
          % (len(payload), len(payload["metrics"]), len(payload["nodes"]),
             len(payload["edges"])))
    print("   a2ui: %d polylines, %d markers, %d overlays, %d stat cards"
          % (len(ui["polylines"]), len(ui["markers"]), len(ui["overlays"]),
             len(ui["statCards"])))
    print("   geojson features = %d (%d LineString + %d Point)"
          % (len(gj["features"]),
             sum(1 for f in gj["features"]
                 if f["geometry"]["type"] == "LineString"),
             sum(1 for f in gj["features"] if f["geometry"]["type"] == "Point")))
    print("   network_summary(radial) mean degree = %.3f, blocks = %d"
          % (network_summary(radial, circuity_pairs=40)["mean_degree"],
             network_summary(radial, circuity_pairs=40)["num_blocks"]))
    print("   detour index of the grid = %.4f" % detour_index(grid, num_pairs=40))

    print("\n-- [14] validation --")
    try:
        grid.add_edge(0, 1, -5.0)
    except StreetNetworkError as exc:
        print("   negative length rejected: %s" % exc)
    try:
        TravelTimeProfile((0.0, 10.0), (5.0, 5.0)).travel_time(float("nan"))
    except StreetNetworkError as exc:
        print("   non-finite departure rejected: %s" % exc)

    print("\n" + "=" * 70)
    print("  demo OK")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
