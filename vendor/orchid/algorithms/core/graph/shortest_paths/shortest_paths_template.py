"""
algorithms/core/graph/shortest_paths/shortest_paths_template.py

Pure-Python, zero-dependency shortest-path library.

A self-contained, competition-ready toolkit covering single-source, all-pairs,
bidirectional, heuristic, bottleneck and k-shortest-path problems, plus a 2-D
grid pathfinder and an automatic algorithm dispatcher.  No NumPy, SciPy,
NetworkX, or any third-party package is required at runtime.

Graph container
---------------
  PathGraph                 - Weighted (un)directed graph optimised for path
                              queries; parallel-edge reduction policy, optional
                              node coordinates, O(1) edge-weight lookup,
                              transpose / reverse, matrix & dict constructors.

Result containers (frozen dataclasses)
--------------------------------------
  PathStats                 - Per-run telemetry (pops, pushes, relaxations, ms)
  Path                      - An immutable node sequence with its cost
  ShortestPathResult        - Single-source dist / pred + path reconstruction
  AllPairsResult            - All-pairs dist + next-hop reconstruction, with
                              eccentricity / diameter / radius / center
  KShortestPathsResult      - Ordered list of loopless paths

Single-source solvers
---------------------
  Dijkstra                  - Binary-heap; O((V+E) log V); non-negative weights
  DialDijkstra              - Bucket queue; O(V + E + maxW·V); integer weights
  ZeroOneBFS                - Deque; O(V+E); weights in {0, 1}
  BFSShortestPath           - Unweighted BFS; O(V+E)
  BellmanFord               - O(V·E); negative weights + negative-cycle report
  SPFA                      - Queue Bellman-Ford (+ SLF) with cycle detection
  DAGShortestPath           - Topological relaxation; O(V+E); shortest OR
                              longest path; negative weights allowed

Bidirectional & heuristic
-------------------------
  BidirectionalDijkstra     - Two-front meet-in-the-middle Dijkstra
  AStar                     - Heuristic-guided; admissible + weighted (epsilon)

All-pairs
---------
  FloydWarshall             - O(V³); path reconstruction; negative-cycle report;
                              transitive closure
  Johnson                   - O(V·E + V² log V); reweighted Dijkstra for sparse
                              graphs with negative edges

Specialised objectives
----------------------
  BottleneckPath            - Widest path (max-min edge) or minimax (min-max
                              edge); modified Dijkstra

k-shortest paths
----------------
  YenKShortestPaths         - Yen's loopless k-shortest paths

Heuristics (for A*)
-------------------
  Heuristics.zero / manhattan / euclidean / chebyshev / octile / haversine

Grid pathfinding
----------------
  GridPathfinder            - 2-D occupancy grid -> graph -> A* / Dijkstra,
                              4- or 8-connectivity, per-cell entry cost.

Convenience dispatchers
-----------------------
  shortest_path             - Auto-selects the best single-source algorithm
  all_pairs_shortest_path   - Auto-selects Floyd-Warshall vs Johnson
  k_shortest_paths          - Thin wrapper over YenKShortestPaths

Payload builders
----------------
  build_shortest_path_payload - Frontend-ready JSON dict
  build_a2ui_payload          - Visualisation-ready dict (nodes, edges, stats,
                                highlighted path, layout hints)

Run as a script for an end-to-end demo:
  python shortest_paths_template.py
"""

from __future__ import annotations

import heapq
import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import count
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List,
    Optional, Sequence, Set, Tuple, Union,
)

__version__ = "1.0.0"

__all__ = [
    # Types
    "NodeID", "Weight",
    # Graph
    "PathGraph",
    # Result containers
    "PathStats", "Path", "ShortestPathResult", "AllPairsResult",
    "KShortestPathsResult",
    # Base classes
    "BaseShortestPathSolver", "BaseAllPairsSolver",
    # Single-source
    "Dijkstra", "DialDijkstra", "ZeroOneBFS", "BFSShortestPath",
    "BellmanFord", "SPFA", "DAGShortestPath",
    # Bidirectional / heuristic
    "BidirectionalDijkstra", "AStar",
    # All-pairs
    "FloydWarshall", "Johnson",
    # Specialised
    "BottleneckPath",
    # k-shortest
    "YenKShortestPaths",
    # Heuristics / utilities
    "Heuristics", "PathAnalyzer", "GridPathfinder",
    # Dispatchers
    "shortest_path", "all_pairs_shortest_path", "k_shortest_paths",
    # Payload builders
    "build_shortest_path_payload", "build_a2ui_payload",
]

# ─────────────────────────────────────────────────────────────────────────────
# Module-level types and constants
# ─────────────────────────────────────────────────────────────────────────────

NodeID = Any            # any hashable: int, str, or tuple (e.g. grid cells)
Weight = float

_INF = math.inf
_EPS = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Graph container
# ─────────────────────────────────────────────────────────────────────────────

class PathGraph:
    """Weighted graph optimised for shortest-path queries.

    Parameters
    ----------
    directed : bool
        If True, edges are one-way; otherwise each edge is added in both
        directions with identical weight.
    allow_negative : bool
        If False, ``add_edge`` rejects negative weights up front.  Defaults to
        True so that Bellman-Ford / Johnson / DAG solvers can be used; the
        non-negative solvers validate weights themselves at solve time.
    on_duplicate : {"min", "max", "last", "first", "sum"}
        Reduction applied when the same (u, v) edge is added more than once.
        ``"min"`` (default) is the correct choice for shortest paths;
        ``"max"`` is appropriate for widest/bottleneck capacity graphs.
    """

    _REDUCERS: Dict[str, Callable[[float, float], float]] = {
        "min":   min,
        "max":   max,
        "last":  lambda old, new: new,
        "first": lambda old, new: old,
        "sum":   lambda old, new: old + new,
    }

    def __init__(self, *, directed: bool = False, allow_negative: bool = True,
                 on_duplicate: str = "min") -> None:
        if on_duplicate not in self._REDUCERS:
            raise ValueError(
                f"on_duplicate must be one of {sorted(self._REDUCERS)}; "
                f"got {on_duplicate!r}."
            )
        self._directed = directed
        self._allow_negative = allow_negative
        self._reduce = self._REDUCERS[on_duplicate]
        self._on_duplicate = on_duplicate
        self._adj: Dict[NodeID, Dict[NodeID, float]] = {}
        self._coords: Dict[NodeID, Tuple[float, ...]] = {}
        self._meta: Dict[Tuple[NodeID, NodeID], Dict[str, Any]] = {}
        self._num_edges = 0

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def directed(self) -> bool:
        return self._directed

    @property
    def nodes(self) -> List[NodeID]:
        return list(self._adj.keys())

    @property
    def num_nodes(self) -> int:
        return len(self._adj)

    @property
    def num_edges(self) -> int:
        return self._num_edges

    @property
    def edges(self) -> List[Tuple[NodeID, NodeID, float]]:
        """List of (u, v, weight).  Undirected edges are emitted once."""
        out: List[Tuple[NodeID, NodeID, float]] = []
        if self._directed:
            for u, nbrs in self._adj.items():
                for v, w in nbrs.items():
                    out.append((u, v, w))
        else:
            seen: Set[Tuple[NodeID, NodeID]] = set()
            for u, nbrs in self._adj.items():
                for v, w in nbrs.items():
                    key = (u, v) if (v, u) not in seen else (v, u)
                    if key in seen:
                        continue
                    seen.add((u, v))
                    out.append((u, v, w))
        return out

    # ── Node / edge construction ──────────────────────────────────────────────

    def add_node(self, node: NodeID, *, coord: Optional[Sequence[float]] = None) -> None:
        if node not in self._adj:
            self._adj[node] = {}
        if coord is not None:
            self._coords[node] = tuple(float(c) for c in coord)

    def add_nodes_from(self, nodes: Iterable[NodeID]) -> None:
        for n in nodes:
            self.add_node(n)

    def add_edge(self, u: NodeID, v: NodeID, weight: float = 1.0,
                 **meta: Any) -> None:
        w = float(weight)
        if not self._allow_negative and w < 0:
            raise ValueError(
                f"Negative weight {w} for edge ({u}, {v}) but allow_negative=False."
            )
        self.add_node(u)
        self.add_node(v)
        is_new = v not in self._adj[u]
        self._adj[u][v] = w if is_new else self._reduce(self._adj[u][v], w)
        if not self._directed:
            self._adj[v][u] = self._adj[u][v]
        if meta:
            self._meta[(u, v)] = {**self._meta.get((u, v), {}), **meta}
        if is_new:
            self._num_edges += 1

    def add_edges_from(self, edges: Iterable[Sequence[Any]]) -> None:
        """Accept (u, v), (u, v, w), or (u, v, w, meta_dict) tuples."""
        for e in edges:
            if len(e) == 2:
                self.add_edge(e[0], e[1])
            elif len(e) == 3:
                self.add_edge(e[0], e[1], float(e[2]))
            else:
                meta = e[3] if isinstance(e[3], dict) else {}
                self.add_edge(e[0], e[1], float(e[2]), **meta)

    def remove_edge(self, u: NodeID, v: NodeID) -> None:
        if u in self._adj and v in self._adj[u]:
            del self._adj[u][v]
            if not self._directed and u in self._adj.get(v, {}):
                del self._adj[v][u]
            self._meta.pop((u, v), None)
            self._num_edges -= 1

    def remove_node(self, node: NodeID) -> None:
        if node not in self._adj:
            return
        for v in list(self._adj[node]):
            self.remove_edge(node, v)
        # Inbound edges (directed case)
        for u in list(self._adj):
            if node in self._adj[u]:
                self.remove_edge(u, node)
        self._adj.pop(node, None)
        self._coords.pop(node, None)

    # ── Queries ───────────────────────────────────────────────────────────────

    def has_node(self, node: NodeID) -> bool:
        return node in self._adj

    def has_edge(self, u: NodeID, v: NodeID) -> bool:
        return u in self._adj and v in self._adj[u]

    def weight(self, u: NodeID, v: NodeID) -> float:
        return self._adj.get(u, {}).get(v, _INF)

    def neighbors(self, node: NodeID) -> Iterator[Tuple[NodeID, float]]:
        """Yield (neighbour, weight) pairs leaving ``node``."""
        return iter(self._adj.get(node, {}).items())

    def degree(self, node: NodeID) -> int:
        return len(self._adj.get(node, {}))

    def coord(self, node: NodeID) -> Optional[Tuple[float, ...]]:
        return self._coords.get(node)

    def set_coord(self, node: NodeID, coord: Sequence[float]) -> None:
        self.add_node(node, coord=coord)

    def has_negative_edges(self) -> bool:
        return any(w < 0 for _, _, w in self.edges)

    def is_unweighted(self, *, tol: float = _EPS) -> bool:
        return all(abs(w - 1.0) <= tol for _, _, w in self.edges)

    def is_zero_one(self, *, tol: float = _EPS) -> bool:
        return all(abs(w) <= tol or abs(w - 1.0) <= tol for _, _, w in self.edges)

    def has_integer_weights(self, *, tol: float = _EPS) -> bool:
        return all(abs(w - round(w)) <= tol for _, _, w in self.edges)

    # ── Transformations ──────────────────────────────────────────────────────

    def reverse(self) -> "PathGraph":
        """Return the transpose (reversed edges).  Undirected -> a copy."""
        g = PathGraph(directed=self._directed,
                      allow_negative=self._allow_negative,
                      on_duplicate=self._on_duplicate)
        for n in self._adj:
            g.add_node(n, coord=self._coords.get(n))
        for u, v, w in self.edges:
            if self._directed:
                g.add_edge(v, u, w)
            else:
                g.add_edge(u, v, w)
        return g

    def copy(self) -> "PathGraph":
        g = PathGraph(directed=self._directed,
                      allow_negative=self._allow_negative,
                      on_duplicate=self._on_duplicate)
        for n in self._adj:
            g.add_node(n, coord=self._coords.get(n))
        for u, nbrs in self._adj.items():
            for v, w in nbrs.items():
                g._adj[u][v] = w
        g._meta = {k: dict(val) for k, val in self._meta.items()}
        g._num_edges = self._num_edges
        return g

    def to_matrix(self, *, nodes: Optional[Sequence[NodeID]] = None
                  ) -> Tuple[List[List[float]], List[NodeID]]:
        order = list(nodes) if nodes is not None else self.nodes
        idx = {n: i for i, n in enumerate(order)}
        n = len(order)
        mat = [[_INF] * n for _ in range(n)]
        for i in range(n):
            mat[i][i] = 0.0
        for u, v, w in self.edges:
            if u in idx and v in idx:
                mat[idx[u]][idx[v]] = w
                if not self._directed:
                    mat[idx[v]][idx[u]] = w
        return mat, order

    # ── Constructors ───────────────────────────────────────────────────────────

    @classmethod
    def from_edge_list(cls, edges: Iterable[Sequence[Any]], *,
                       directed: bool = False, allow_negative: bool = True,
                       on_duplicate: str = "min") -> "PathGraph":
        g = cls(directed=directed, allow_negative=allow_negative,
                on_duplicate=on_duplicate)
        g.add_edges_from(edges)
        return g

    @classmethod
    def from_adjacency_dict(cls, adj: Dict[NodeID, Any], *,
                            directed: bool = False, allow_negative: bool = True,
                            on_duplicate: str = "min") -> "PathGraph":
        """``adj`` maps u -> {v: weight} or u -> iterable of v (weight 1)."""
        g = cls(directed=directed, allow_negative=allow_negative,
                on_duplicate=on_duplicate)
        for u, nbrs in adj.items():
            g.add_node(u)
            if isinstance(nbrs, dict):
                for v, w in nbrs.items():
                    g.add_edge(u, v, float(w))
            else:
                for v in nbrs:
                    g.add_edge(u, v, 1.0)
        return g

    @classmethod
    def from_adjacency_matrix(cls, matrix: Sequence[Sequence[float]], *,
                              nodes: Optional[Sequence[NodeID]] = None,
                              directed: bool = False, allow_negative: bool = True,
                              no_edge: Optional[float] = None,
                              on_duplicate: str = "min") -> "PathGraph":
        """Build from a dense matrix.  ``None`` / ``inf`` / ``no_edge`` =>
        absent edge (a zero is treated as a *real* zero-weight edge)."""
        n = len(matrix)
        order = list(nodes) if nodes is not None else list(range(n))
        if len(order) != n:
            raise ValueError("len(nodes) must equal the matrix dimension.")
        g = cls(directed=directed, allow_negative=allow_negative,
                on_duplicate=on_duplicate)
        g.add_nodes_from(order)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                w = matrix[i][j]
                if w is None or w == no_edge or (isinstance(w, float) and math.isinf(w)):
                    continue
                if directed or i < j:
                    g.add_edge(order[i], order[j], float(w))
        return g

    def __repr__(self) -> str:
        kind = "directed" if self._directed else "undirected"
        return f"PathGraph({kind}, |V|={self.num_nodes}, |E|={self.num_edges})"


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PathStats:
    """Per-run telemetry for performance comparison and tuning."""
    algorithm: str
    heap_pops: int = 0
    heap_pushes: int = 0
    edges_relaxed: int = 0
    iterations: int = 0
    nodes_settled: int = 0
    runtime_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "heapPops": self.heap_pops,
            "heapPushes": self.heap_pushes,
            "edgesRelaxed": self.edges_relaxed,
            "iterations": self.iterations,
            "nodesSettled": self.nodes_settled,
            "runtimeMs": round(self.runtime_ms, 4),
        }


@dataclass(frozen=True)
class Path:
    """An immutable node sequence and its total cost."""
    nodes: Tuple[NodeID, ...]
    cost: float

    @property
    def length(self) -> int:
        """Number of edges (hops)."""
        return max(0, len(self.nodes) - 1)

    def edges(self) -> List[Tuple[NodeID, NodeID]]:
        return list(zip(self.nodes[:-1], self.nodes[1:]))

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[NodeID]:
        return iter(self.nodes)

    def __repr__(self) -> str:
        body = " -> ".join(str(n) for n in self.nodes)
        return f"Path([{body}], cost={self.cost:.4g})"


@dataclass(frozen=True)
class ShortestPathResult:
    """Single-source shortest-path result with path reconstruction."""
    algorithm: str
    source: NodeID
    target: Optional[NodeID] = None
    dist: Dict[NodeID, float] = field(default_factory=dict)
    pred: Dict[NodeID, Optional[NodeID]] = field(default_factory=dict)
    has_negative_cycle: bool = False
    negative_cycle: Tuple[NodeID, ...] = ()
    stats: Optional[PathStats] = None

    # ── Distance queries ──────────────────────────────────────────────────────

    def distance_to(self, target: NodeID) -> float:
        return self.dist.get(target, _INF)

    def is_reachable(self, target: NodeID) -> bool:
        return self.dist.get(target, _INF) < _INF

    @property
    def reachable_nodes(self) -> List[NodeID]:
        return [n for n, d in self.dist.items() if d < _INF]

    # ── Path reconstruction ───────────────────────────────────────────────────

    def path_to(self, target: NodeID) -> List[NodeID]:
        """Reconstruct the node path source -> target ([] if none)."""
        if self.dist.get(target, _INF) == _INF and target != self.source:
            return []
        path: List[NodeID] = []
        seen: Set[NodeID] = set()
        cur: Optional[NodeID] = target
        while cur is not None:
            if cur in seen:          # defensive: predecessor cycle
                return []
            seen.add(cur)
            path.append(cur)
            if cur == self.source:
                break
            cur = self.pred.get(cur)
        path.reverse()
        if not path or path[0] != self.source:
            return []
        return path

    def edge_path_to(self, target: NodeID) -> List[Tuple[NodeID, NodeID]]:
        nodes = self.path_to(target)
        return list(zip(nodes[:-1], nodes[1:]))

    def get_path(self, target: NodeID) -> Path:
        return Path(tuple(self.path_to(target)), self.distance_to(target))

    def all_paths(self) -> Dict[NodeID, Path]:
        return {n: self.get_path(n) for n in self.reachable_nodes}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "source": self.source,
            "target": self.target,
            "hasNegativeCycle": self.has_negative_cycle,
            "negativeCycle": list(self.negative_cycle),
            "reachable": len(self.reachable_nodes),
            "stats": self.stats.to_dict() if self.stats else None,
        }


@dataclass(frozen=True)
class AllPairsResult:
    """All-pairs shortest-path result with next-hop reconstruction."""
    algorithm: str
    nodes: Tuple[NodeID, ...]
    dist: Dict[NodeID, Dict[NodeID, float]]
    nxt: Dict[NodeID, Dict[NodeID, Optional[NodeID]]] = field(default_factory=dict)
    has_negative_cycle: bool = False
    stats: Optional[PathStats] = None

    def distance(self, u: NodeID, v: NodeID) -> float:
        return self.dist.get(u, {}).get(v, _INF)

    def path(self, u: NodeID, v: NodeID) -> List[NodeID]:
        if not self.nxt or self.distance(u, v) == _INF:
            return [] if u != v else [u]
        if u == v:
            return [u]
        path = [u]
        cur = u
        limit = len(self.nodes) + 1
        while cur != v:
            cur = self.nxt.get(cur, {}).get(v)
            if cur is None:
                return []
            path.append(cur)
            if len(path) > limit:        # defensive against corrupt next-hop
                return []
        return path

    def eccentricity(self, u: NodeID) -> float:
        row = self.dist.get(u, {})
        finite = [d for v, d in row.items() if d < _INF]
        return max(finite) if finite else _INF

    def diameter(self) -> float:
        eccs = [self.eccentricity(u) for u in self.nodes]
        finite = [e for e in eccs if e < _INF]
        return max(finite) if finite else _INF

    def radius(self) -> float:
        eccs = [self.eccentricity(u) for u in self.nodes]
        finite = [e for e in eccs if e < _INF]
        return min(finite) if finite else _INF

    def center(self) -> List[NodeID]:
        r = self.radius()
        if r == _INF:
            return []
        return [u for u in self.nodes if abs(self.eccentricity(u) - r) <= _EPS]

    def is_all_connected(self) -> bool:
        return all(
            self.distance(u, v) < _INF
            for u in self.nodes for v in self.nodes
        )


@dataclass(frozen=True)
class KShortestPathsResult:
    """Ordered collection of loopless shortest paths."""
    algorithm: str
    source: NodeID
    target: NodeID
    paths: Tuple[Path, ...] = ()

    @property
    def costs(self) -> List[float]:
        return [p.cost for p in self.paths]

    def best(self) -> Optional[Path]:
        return self.paths[0] if self.paths else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Path:
        return self.paths[i]

    def __iter__(self) -> Iterator[Path]:
        return iter(self.paths)


# ─────────────────────────────────────────────────────────────────────────────
# Internal shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_tables(g: PathGraph, source: NodeID
                 ) -> Tuple[Dict[NodeID, float], Dict[NodeID, Optional[NodeID]]]:
    dist: Dict[NodeID, float] = {n: _INF for n in g.nodes}
    pred: Dict[NodeID, Optional[NodeID]] = {n: None for n in g.nodes}
    dist[source] = 0.0
    return dist, pred


def _extract_negative_cycle(pred: Dict[NodeID, Optional[NodeID]],
                            entry: NodeID, n: int) -> Tuple[NodeID, ...]:
    """Walk back ``n`` predecessors (guaranteeing arrival inside the cycle),
    then collect the cycle by following predecessors until it repeats."""
    x: Optional[NodeID] = entry
    for _ in range(n):
        if x is None:
            return ()
        x = pred.get(x)
    if x is None:
        return ()
    cycle: List[NodeID] = []
    cur: Optional[NodeID] = x
    seen: Set[NodeID] = set()
    while cur is not None and cur not in seen:
        seen.add(cur)
        cycle.append(cur)
        cur = pred.get(cur)
    cycle.append(x)
    cycle.reverse()
    return tuple(cycle)


def _dijkstra_core(g: PathGraph, source: NodeID, *,
                   target: Optional[NodeID] = None,
                   weight: Optional[Callable[[NodeID, NodeID, float], float]] = None,
                   blocked_nodes: Optional[Set[NodeID]] = None,
                   blocked_edges: Optional[Set[Tuple[NodeID, NodeID]]] = None,
                   check_negative: bool = True
                   ) -> Tuple[Dict[NodeID, float], Dict[NodeID, Optional[NodeID]],
                              Dict[str, int]]:
    """Heart of Dijkstra, reused by the public solver, Johnson and Yen."""
    wfn = weight
    bn = blocked_nodes or frozenset()
    be = blocked_edges or frozenset()
    dist, pred = _init_tables(g, source)
    heap: List[Tuple[float, int, NodeID]] = [(0.0, 0, source)]
    tie = count(1)
    settled: Set[NodeID] = set()
    pops = pushes = relax = 0
    while heap:
        d, _, u = heapq.heappop(heap)
        pops += 1
        if u in settled:
            continue
        settled.add(u)
        if target is not None and u == target:
            break
        for v, base in g.neighbors(u):
            if v in bn or (u, v) in be:
                continue
            w = wfn(u, v, base) if wfn else base
            if check_negative and w < 0:
                raise ValueError(
                    f"Dijkstra requires non-negative weights; edge ({u}, {v}) "
                    f"has weight {w}. Use BellmanFord, SPFA or Johnson instead."
                )
            relax += 1
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = u
                heapq.heappush(heap, (nd, next(tie), v))
                pushes += 1
    return dist, pred, {
        "pops": pops, "pushes": pushes, "relax": relax, "settled": len(settled),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Base classes
# ─────────────────────────────────────────────────────────────────────────────

class BaseShortestPathSolver(ABC):
    """Common interface for single-source shortest-path solvers.

    ``weight_fn(u, v, base_weight) -> float`` optionally transforms edge weights
    at solve time — useful for time-dependent costs, penalties, or turning a
    capacity into a cost without rebuilding the graph.
    """

    name: str = "base"

    def __init__(self, *,
                 weight_fn: Optional[Callable[[NodeID, NodeID, float], float]] = None
                 ) -> None:
        self.weight_fn = weight_fn

    def _w(self, u: NodeID, v: NodeID, base: float) -> float:
        return self.weight_fn(u, v, base) if self.weight_fn else base

    @staticmethod
    def _require_node(g: PathGraph, node: NodeID) -> None:
        if not g.has_node(node):
            raise KeyError(f"Node {node!r} is not in the graph.")

    @abstractmethod
    def solve(self, g: PathGraph, source: NodeID,
              target: Optional[NodeID] = None) -> ShortestPathResult: ...

    def shortest_path(self, g: PathGraph, source: NodeID, target: NodeID) -> Path:
        """Convenience: return just the source -> target :class:`Path`."""
        return self.solve(g, source, target).get_path(target)


class BaseAllPairsSolver(ABC):
    """Common interface for all-pairs shortest-path solvers."""

    name: str = "base"

    @abstractmethod
    def solve(self, g: PathGraph) -> AllPairsResult: ...


# ─────────────────────────────────────────────────────────────────────────────
# Single-source solvers
# ─────────────────────────────────────────────────────────────────────────────

class Dijkstra(BaseShortestPathSolver):
    """Binary-heap Dijkstra.  O((V + E) log V).  Non-negative weights only."""

    name = "Dijkstra"

    def solve(self, g: PathGraph, source: NodeID,
              target: Optional[NodeID] = None) -> ShortestPathResult:
        self._require_node(g, source)
        t0 = time.perf_counter()
        dist, pred, tel = _dijkstra_core(
            g, source, target=target, weight=self.weight_fn, check_negative=True,
        )
        dt = (time.perf_counter() - t0) * 1000.0
        stats = PathStats(
            algorithm=self.name, heap_pops=tel["pops"], heap_pushes=tel["pushes"],
            edges_relaxed=tel["relax"], nodes_settled=tel["settled"], runtime_ms=dt,
        )
        return ShortestPathResult(self.name, source, target, dist, pred, stats=stats)


class DialDijkstra(BaseShortestPathSolver):
    """Dial's algorithm — bucket-queue Dijkstra for non-negative *integer*
    weights.  O(V + E + maxW·V); beats the heap when the weight range is small.
    """

    name = "DialDijkstra"

    def solve(self, g: PathGraph, source: NodeID,
              target: Optional[NodeID] = None) -> ShortestPathResult:
        self._require_node(g, source)
        t0 = time.perf_counter()
        # Validate / collect integer non-negative weights.
        max_w = 0
        for u, v, base in g.edges:
            w = self._w(u, v, base)
            if w < 0:
                raise ValueError("DialDijkstra requires non-negative weights.")
            if abs(w - round(w)) > _EPS:
                raise ValueError("DialDijkstra requires integer weights.")
            max_w = max(max_w, int(round(w)))

        dist, pred = _init_tables(g, source)
        n = g.num_nodes
        buckets: Dict[int, List[NodeID]] = defaultdict(list)
        buckets[0].append(source)
        settled: Set[NodeID] = set()
        idx = 0
        max_idx = max_w * max(n - 1, 1)
        pops = pushes = relax = 0
        while idx <= max_idx:
            bucket = buckets.get(idx)
            if bucket:
                while bucket:
                    u = bucket.pop()
                    pops += 1
                    if u in settled or dist[u] != idx:
                        continue
                    settled.add(u)
                    if target is not None and u == target:
                        idx = max_idx + 1
                        break
                    for v, base in g.neighbors(u):
                        w = int(round(self._w(u, v, base)))
                        relax += 1
                        nd = idx + w
                        if nd < dist[v]:
                            dist[v] = nd
                            pred[v] = u
                            buckets[nd].append(v)
                            pushes += 1
            else:
                idx += 1
        dt = (time.perf_counter() - t0) * 1000.0
        stats = PathStats(self.name, pops, pushes, relax,
                          nodes_settled=len(settled), runtime_ms=dt)
        return ShortestPathResult(self.name, source, target, dist, pred, stats=stats)


class ZeroOneBFS(BaseShortestPathSolver):
    """0-1 BFS — deque-based, O(V + E).  All weights must be 0 or 1."""

    name = "ZeroOneBFS"

    def solve(self, g: PathGraph, source: NodeID,
              target: Optional[NodeID] = None) -> ShortestPathResult:
        self._require_node(g, source)
        t0 = time.perf_counter()
        dist, pred = _init_tables(g, source)
        dq: deque = deque([source])
        pops = pushes = relax = 0
        settled: Set[NodeID] = set()
        while dq:
            u = dq.popleft()
            pops += 1
            if u in settled:
                continue
            settled.add(u)
            if target is not None and u == target:
                break
            for v, base in g.neighbors(u):
                w = self._w(u, v, base)
                if abs(w) > _EPS and abs(w - 1.0) > _EPS:
                    raise ValueError("ZeroOneBFS requires weights in {0, 1}.")
                relax += 1
                nd = dist[u] + w
                if nd < dist[v]:
                    dist[v] = nd
                    pred[v] = u
                    pushes += 1
                    if w <= _EPS:
                        dq.appendleft(v)
                    else:
                        dq.append(v)
        dt = (time.perf_counter() - t0) * 1000.0
        stats = PathStats(self.name, pops, pushes, relax,
                          nodes_settled=len(settled), runtime_ms=dt)
        return ShortestPathResult(self.name, source, target, dist, pred, stats=stats)


class BFSShortestPath(BaseShortestPathSolver):
    """Breadth-first search on an unweighted graph (every edge counts as 1).

    Edge weights are ignored — this answers *fewest-hops* queries."""

    name = "BFS"

    def solve(self, g: PathGraph, source: NodeID,
              target: Optional[NodeID] = None) -> ShortestPathResult:
        self._require_node(g, source)
        t0 = time.perf_counter()
        dist, pred = _init_tables(g, source)
        dq: deque = deque([source])
        pops = pushes = relax = 0
        while dq:
            u = dq.popleft()
            pops += 1
            if target is not None and u == target:
                break
            for v, _base in g.neighbors(u):
                relax += 1
                if dist[v] == _INF:
                    dist[v] = dist[u] + 1
                    pred[v] = u
                    dq.append(v)
                    pushes += 1
        dt = (time.perf_counter() - t0) * 1000.0
        stats = PathStats(self.name, pops, pushes, relax,
                          nodes_settled=pops, runtime_ms=dt)
        return ShortestPathResult(self.name, source, target, dist, pred, stats=stats)


class BellmanFord(BaseShortestPathSolver):
    """Bellman-Ford.  O(V·E).  Handles negative weights; reports and extracts a
    reachable negative cycle when one exists."""

    name = "BellmanFord"

    def solve(self, g: PathGraph, source: NodeID,
              target: Optional[NodeID] = None) -> ShortestPathResult:
        self._require_node(g, source)
        t0 = time.perf_counter()
        dist, pred = _init_tables(g, source)
        n = g.num_nodes
        # Directed arc list (both directions already present for undirected).
        arcs = [(u, v, w) for u in g.nodes for v, w in g.neighbors(u)]
        relax = 0
        iters = 0
        for i in range(n - 1):
            iters += 1
            changed = False
            for u, v, base in arcs:
                if dist[u] == _INF:
                    continue
                w = self._w(u, v, base)
                relax += 1
                if dist[u] + w < dist[v] - _EPS:
                    dist[v] = dist[u] + w
                    pred[v] = u
                    changed = True
            if not changed:
                break
        # One more pass to detect a negative cycle.
        has_neg = False
        neg_cycle: Tuple[NodeID, ...] = ()
        for u, v, base in arcs:
            if dist[u] == _INF:
                continue
            w = self._w(u, v, base)
            if dist[u] + w < dist[v] - _EPS:
                has_neg = True
                pred[v] = u
                neg_cycle = _extract_negative_cycle(pred, v, n)
                break
        dt = (time.perf_counter() - t0) * 1000.0
        stats = PathStats(self.name, edges_relaxed=relax, iterations=iters,
                          nodes_settled=n, runtime_ms=dt)
        return ShortestPathResult(self.name, source, target, dist, pred,
                                  has_negative_cycle=has_neg,
                                  negative_cycle=neg_cycle, stats=stats)


class SPFA(BaseShortestPathSolver):
    """Shortest Path Faster Algorithm — a queue-based Bellman-Ford that is much
    faster in practice.  Optional SLF (Small-Label-First) ordering.  Detects
    negative cycles via the relaxation-count rule."""

    name = "SPFA"

    def __init__(self, *, slf: bool = True,
                 weight_fn: Optional[Callable[[NodeID, NodeID, float], float]] = None
                 ) -> None:
        super().__init__(weight_fn=weight_fn)
        self.slf = slf

    def solve(self, g: PathGraph, source: NodeID,
              target: Optional[NodeID] = None) -> ShortestPathResult:
        self._require_node(g, source)
        t0 = time.perf_counter()
        dist, pred = _init_tables(g, source)
        n = g.num_nodes
        in_queue: Set[NodeID] = {source}
        relax_count: Dict[NodeID, int] = defaultdict(int)
        dq: deque = deque([source])
        relax = pops = 0
        has_neg = False
        neg_cycle: Tuple[NodeID, ...] = ()
        while dq:
            u = dq.popleft()
            pops += 1
            in_queue.discard(u)
            for v, base in g.neighbors(u):
                w = self._w(u, v, base)
                relax += 1
                if dist[u] != _INF and dist[u] + w < dist[v] - _EPS:
                    dist[v] = dist[u] + w
                    pred[v] = u
                    if v not in in_queue:
                        relax_count[v] += 1
                        if relax_count[v] >= n:
                            has_neg = True
                            neg_cycle = _extract_negative_cycle(pred, v, n)
                            dq.clear()
                            break
                        if self.slf and dq and dist[v] < dist[dq[0]]:
                            dq.appendleft(v)
                        else:
                            dq.append(v)
                        in_queue.add(v)
        dt = (time.perf_counter() - t0) * 1000.0
        stats = PathStats(self.name, heap_pops=pops, edges_relaxed=relax,
                          nodes_settled=pops, runtime_ms=dt)
        return ShortestPathResult(self.name, source, target, dist, pred,
                                  has_negative_cycle=has_neg,
                                  negative_cycle=neg_cycle, stats=stats)


class DAGShortestPath(BaseShortestPathSolver):
    """Single-source shortest *or longest* path on a directed acyclic graph via
    topological-order relaxation.  O(V + E).  Negative weights are fine.

    mode : {"shortest", "longest"}
    """

    name = "DAGShortestPath"

    def __init__(self, *, mode: str = "shortest",
                 weight_fn: Optional[Callable[[NodeID, NodeID, float], float]] = None
                 ) -> None:
        if mode not in ("shortest", "longest"):
            raise ValueError("mode must be 'shortest' or 'longest'.")
        super().__init__(weight_fn=weight_fn)
        self.mode = mode

    def _topo_order(self, g: PathGraph) -> List[NodeID]:
        if not g.directed:
            raise ValueError("DAGShortestPath requires a directed graph.")
        indeg: Dict[NodeID, int] = {n: 0 for n in g.nodes}
        for u in g.nodes:
            for v, _w in g.neighbors(u):
                indeg[v] += 1
        dq = deque([n for n in g.nodes if indeg[n] == 0])
        order: List[NodeID] = []
        while dq:
            u = dq.popleft()
            order.append(u)
            for v, _w in g.neighbors(u):
                indeg[v] -= 1
                if indeg[v] == 0:
                    dq.append(v)
        if len(order) != g.num_nodes:
            raise ValueError("Graph is not acyclic; DAGShortestPath cannot run.")
        return order

    def solve(self, g: PathGraph, source: NodeID,
              target: Optional[NodeID] = None) -> ShortestPathResult:
        self._require_node(g, source)
        t0 = time.perf_counter()
        order = self._topo_order(g)
        longest = self.mode == "longest"
        start_val = -_INF if longest else _INF
        dist: Dict[NodeID, float] = {n: start_val for n in g.nodes}
        pred: Dict[NodeID, Optional[NodeID]] = {n: None for n in g.nodes}
        dist[source] = 0.0
        relax = 0
        better = (lambda a, b: a > b) if longest else (lambda a, b: a < b)
        for u in order:
            if dist[u] == start_val:
                continue
            for v, base in g.neighbors(u):
                w = self._w(u, v, base)
                relax += 1
                cand = dist[u] + w
                if better(cand, dist[v]):
                    dist[v] = cand
                    pred[v] = u
        # Normalise unreachable longest-path sentinels back to +inf semantics.
        if longest:
            dist = {n: (d if d != -_INF else _INF) for n, d in dist.items()}
        dt = (time.perf_counter() - t0) * 1000.0
        stats = PathStats(f"{self.name}[{self.mode}]", edges_relaxed=relax,
                          iterations=len(order), nodes_settled=len(order),
                          runtime_ms=dt)
        return ShortestPathResult(f"{self.name}[{self.mode}]", source, target,
                                  dist, pred, stats=stats)


# ─────────────────────────────────────────────────────────────────────────────
# Bidirectional & heuristic solvers
# ─────────────────────────────────────────────────────────────────────────────

class BidirectionalDijkstra(BaseShortestPathSolver):
    """Bidirectional Dijkstra: simultaneous forward search from the source and
    backward search (on the transpose) from the target.  Explores far fewer
    nodes than unidirectional Dijkstra on large graphs.  Requires a target and
    non-negative weights."""

    name = "BidirectionalDijkstra"

    def solve(self, g: PathGraph, source: NodeID,
              target: Optional[NodeID] = None) -> ShortestPathResult:
        self._require_node(g, source)
        if target is None:
            raise ValueError("BidirectionalDijkstra requires a target node.")
        self._require_node(g, target)
        t0 = time.perf_counter()

        if source == target:
            res = ShortestPathResult(self.name, source, target,
                                     {source: 0.0}, {source: None},
                                     stats=PathStats(self.name, runtime_ms=0.0))
            return res

        gr = g.reverse()
        df: Dict[NodeID, float] = {source: 0.0}
        db: Dict[NodeID, float] = {target: 0.0}
        pf: Dict[NodeID, Optional[NodeID]] = {source: None}
        pb: Dict[NodeID, Optional[NodeID]] = {target: None}
        done_f: Set[NodeID] = set()
        done_b: Set[NodeID] = set()
        tie = count(1)
        heap_f: List[Tuple[float, int, NodeID]] = [(0.0, next(tie), source)]
        heap_b: List[Tuple[float, int, NodeID]] = [(0.0, next(tie), target)]
        mu = _INF
        meet: Optional[NodeID] = None
        pops = pushes = relax = 0

        def expand(heap, dist, pred, done, other_dist, other_done, graph, forward):
            nonlocal mu, meet, pops, pushes, relax
            d, _, u = heapq.heappop(heap)
            pops += 1
            if u in done:
                return
            done.add(u)
            for v, base in graph.neighbors(u):
                # Weight transform stays oriented along original edge direction.
                w = self._w(u, v, base) if forward else self._w(v, u, base)
                if w < 0:
                    raise ValueError(
                        "BidirectionalDijkstra requires non-negative weights."
                    )
                relax += 1
                nd = d + w
                if nd < dist.get(v, _INF):
                    dist[v] = nd
                    pred[v] = u
                    heapq.heappush(heap, (nd, next(tie), v))
                    pushes += 1
                    if v in other_dist:
                        cand = nd + other_dist[v]
                        if cand < mu:
                            mu = cand
                            meet = v

        while heap_f and heap_b:
            if heap_f[0][0] + heap_b[0][0] >= mu:
                break
            # Expand the currently smaller frontier.
            if heap_f[0][0] <= heap_b[0][0]:
                expand(heap_f, df, pf, done_f, db, done_b, g, True)
            else:
                expand(heap_b, db, pb, done_b, df, done_f, gr, False)

        dist_out: Dict[NodeID, float] = {n: _INF for n in g.nodes}
        pred_out: Dict[NodeID, Optional[NodeID]] = {n: None for n in g.nodes}
        for n, d in df.items():
            dist_out[n] = d
        if meet is not None and mu < _INF:
            # Stitch the two half-paths through the meeting node.
            forward_part: List[NodeID] = []
            cur: Optional[NodeID] = meet
            while cur is not None:
                forward_part.append(cur)
                cur = pf.get(cur)
            forward_part.reverse()                 # source .. meet
            backward_part: List[NodeID] = []
            cur = pb.get(meet)
            while cur is not None:
                backward_part.append(cur)          # (after meet) .. target
                cur = pb.get(cur)
            full = forward_part + backward_part
            for i in range(1, len(full)):
                pred_out[full[i]] = full[i - 1]
            dist_out[target] = mu

        dt = (time.perf_counter() - t0) * 1000.0
        stats = PathStats(self.name, heap_pops=pops, heap_pushes=pushes,
                          edges_relaxed=relax,
                          nodes_settled=len(done_f) + len(done_b), runtime_ms=dt)
        return ShortestPathResult(self.name, source, target, dist_out, pred_out,
                                  stats=stats)


class AStar(BaseShortestPathSolver):
    """A* search.  With an admissible heuristic the path is optimal; ``epsilon``
    > 1 yields *weighted A\\** (faster, bounded-suboptimal by a factor epsilon).

    Closed nodes are re-expanded whenever a shorter path to them is discovered
    (lazy deletion), so optimality holds for any **admissible** heuristic — the
    heuristic need not be consistent/monotone.  For a consistent heuristic no
    re-expansion ever occurs, so there is no overhead in the common case.

    Parameters
    ----------
    heuristic : Callable[[NodeID], float]
        Estimated remaining cost from a node to the target.  ``None`` -> 0
        (degenerates to Dijkstra).  Build one with the :class:`Heuristics`
        factory, or pass your own.
    epsilon : float
        Heuristic inflation factor (1.0 = classic A*).
    """

    name = "AStar"

    def __init__(self, *, heuristic: Optional[Callable[[NodeID], float]] = None,
                 epsilon: float = 1.0,
                 weight_fn: Optional[Callable[[NodeID, NodeID, float], float]] = None
                 ) -> None:
        super().__init__(weight_fn=weight_fn)
        self.heuristic = heuristic
        self.epsilon = float(epsilon)

    def solve(self, g: PathGraph, source: NodeID,
              target: Optional[NodeID] = None) -> ShortestPathResult:
        self._require_node(g, source)
        if target is None:
            raise ValueError("AStar requires a target node.")
        self._require_node(g, target)
        h = self.heuristic or (lambda _n: 0.0)
        t0 = time.perf_counter()
        dist, pred = _init_tables(g, source)
        tie = count(1)
        # Heap entries carry g so stale entries can be skipped (lazy deletion);
        # this admits node re-expansion, keeping A* optimal for any admissible
        # (not necessarily consistent) heuristic.
        heap: List[Tuple[float, float, int, NodeID]] = [
            (self.epsilon * h(source), 0.0, next(tie), source)
        ]
        pops = pushes = relax = expansions = 0
        while heap:
            _f, gu, _, u = heapq.heappop(heap)
            pops += 1
            if gu > dist[u]:          # stale: a shorter path to u was found later
                continue
            expansions += 1
            if u == target:
                break
            for v, base in g.neighbors(u):
                w = self._w(u, v, base)
                if w < 0:
                    raise ValueError("AStar requires non-negative weights.")
                relax += 1
                nd = gu + w
                if nd < dist[v]:
                    dist[v] = nd
                    pred[v] = u
                    heapq.heappush(heap,
                                   (nd + self.epsilon * h(v), nd, next(tie), v))
                    pushes += 1
        dt = (time.perf_counter() - t0) * 1000.0
        stats = PathStats(self.name, heap_pops=pops, heap_pushes=pushes,
                          edges_relaxed=relax, nodes_settled=expansions,
                          runtime_ms=dt)
        return ShortestPathResult(self.name, source, target, dist, pred, stats=stats)


# ─────────────────────────────────────────────────────────────────────────────
# All-pairs solvers
# ─────────────────────────────────────────────────────────────────────────────

class FloydWarshall(BaseAllPairsSolver):
    """Floyd-Warshall all-pairs shortest paths.  O(V³).  Reconstructs paths via
    a next-hop matrix and reports negative cycles (negative diagonal)."""

    name = "FloydWarshall"

    def solve(self, g: PathGraph) -> AllPairsResult:
        t0 = time.perf_counter()
        nodes = g.nodes
        n = len(nodes)
        idx = {v: i for i, v in enumerate(nodes)}
        dist = [[_INF] * n for _ in range(n)]
        nxt: List[List[Optional[int]]] = [[None] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0.0
            nxt[i][i] = i
        for u, v, w in g.edges:
            i, j = idx[u], idx[v]
            if w < dist[i][j]:
                dist[i][j] = w
                nxt[i][j] = j
            if not g.directed and w < dist[j][i]:
                dist[j][i] = w
                nxt[j][i] = i
        for k in range(n):
            dk = dist[k]
            for i in range(n):
                dik = dist[i][k]
                if dik == _INF:
                    continue
                di = dist[i]
                ni = nxt[i]
                for j in range(n):
                    nd = dik + dk[j]
                    if nd < di[j]:
                        di[j] = nd
                        ni[j] = nxt[i][k]
        has_neg = any(dist[i][i] < -_EPS for i in range(n))
        dist_d: Dict[NodeID, Dict[NodeID, float]] = {}
        nxt_d: Dict[NodeID, Dict[NodeID, Optional[NodeID]]] = {}
        for i, u in enumerate(nodes):
            dist_d[u] = {nodes[j]: dist[i][j] for j in range(n)}
            nxt_d[u] = {
                nodes[j]: (nodes[nxt[i][j]] if nxt[i][j] is not None else None)
                for j in range(n)
            }
        dt = (time.perf_counter() - t0) * 1000.0
        stats = PathStats(self.name, iterations=n, nodes_settled=n, runtime_ms=dt)
        return AllPairsResult(self.name, tuple(nodes), dist_d, nxt_d,
                              has_negative_cycle=has_neg, stats=stats)

    def transitive_closure(self, g: PathGraph) -> Dict[NodeID, Set[NodeID]]:
        """Reachability: {u: {v : v reachable from u}} (excluding u itself)."""
        res = self.solve(g)
        return {
            u: {v for v in res.nodes
                if v != u and res.distance(u, v) < _INF}
            for u in res.nodes
        }


class Johnson(BaseAllPairsSolver):
    """Johnson's all-pairs algorithm.  Reweights with a Bellman-Ford potential,
    then runs Dijkstra from every node.  O(V·E + V²·log V) — far better than
    Floyd-Warshall on sparse graphs, and tolerates negative edges (but not
    negative cycles)."""

    name = "Johnson"

    def solve(self, g: PathGraph) -> AllPairsResult:
        t0 = time.perf_counter()
        nodes = g.nodes
        n = len(nodes)

        # 1) Potentials h(v) via Bellman-Ford from a virtual super-source.
        h: Dict[NodeID, float] = {v: 0.0 for v in nodes}
        arcs = [(u, v, w) for u in nodes for v, w in g.neighbors(u)]
        for _ in range(n):                       # n iterations (virtual node => V+1)
            changed = False
            for u, v, w in arcs:
                if h[u] + w < h[v] - _EPS:
                    h[v] = h[u] + w
                    changed = True
            if not changed:
                break
        else:
            # Did not converge in n passes — check for a real negative cycle.
            for u, v, w in arcs:
                if h[u] + w < h[v] - _EPS:
                    dt = (time.perf_counter() - t0) * 1000.0
                    return AllPairsResult(
                        self.name, tuple(nodes), {}, {}, has_negative_cycle=True,
                        stats=PathStats(self.name, runtime_ms=dt),
                    )

        # 2) Reweighted Dijkstra from every node.
        def reweight(u: NodeID, v: NodeID, base: float) -> float:
            return base + h[u] - h[v]

        dist_d: Dict[NodeID, Dict[NodeID, float]] = {}
        nxt_d: Dict[NodeID, Dict[NodeID, Optional[NodeID]]] = {}
        total_pops = total_pushes = total_relax = 0
        for s in nodes:
            d_prime, pred, tel = _dijkstra_core(
                g, s, weight=reweight, check_negative=True,
            )
            total_pops += tel["pops"]
            total_pushes += tel["pushes"]
            total_relax += tel["relax"]
            row: Dict[NodeID, float] = {}
            for v in nodes:
                dp = d_prime.get(v, _INF)
                row[v] = dp - h[s] + h[v] if dp < _INF else _INF
            dist_d[s] = row
            nxt_d[s] = _pred_tree_to_nxt(s, pred, nodes)
        dt = (time.perf_counter() - t0) * 1000.0
        stats = PathStats(self.name, heap_pops=total_pops, heap_pushes=total_pushes,
                          edges_relaxed=total_relax, iterations=n, runtime_ms=dt)
        return AllPairsResult(self.name, tuple(nodes), dist_d, nxt_d, stats=stats)


def _pred_tree_to_nxt(source: NodeID, pred: Dict[NodeID, Optional[NodeID]],
                      nodes: Sequence[NodeID]) -> Dict[NodeID, Optional[NodeID]]:
    """Convert a predecessor tree rooted at ``source`` into a next-hop map
    {v: first node after source on the path source -> v}."""
    nxt: Dict[NodeID, Optional[NodeID]] = {}
    for v in nodes:
        if v == source:
            nxt[v] = source
            continue
        seq: List[NodeID] = []
        cur: Optional[NodeID] = v
        seen: Set[NodeID] = set()
        ok = True
        while cur is not None and cur != source:
            if cur in seen:
                ok = False
                break
            seen.add(cur)
            seq.append(cur)
            cur = pred.get(cur)
        if not ok or cur != source:
            nxt[v] = None
        else:
            seq.append(source)
            seq.reverse()                 # source .. v
            nxt[v] = seq[1] if len(seq) >= 2 else None
    return nxt


# ─────────────────────────────────────────────────────────────────────────────
# Specialised objectives
# ─────────────────────────────────────────────────────────────────────────────

class BottleneckPath(BaseShortestPathSolver):
    """Optimise an edge-extremum objective instead of a sum.

    mode="widest"  : maximise the minimum edge weight on the path (the
                     maximum-capacity / bottleneck-shortest-path problem).
    mode="minimax" : minimise the maximum edge weight on the path (the
                     minimax / min-max-edge problem, e.g. minimum spanning
                     path bottleneck).

    ``dist[v]`` in the result holds the optimal bottleneck value to ``v``.
    """

    name = "BottleneckPath"

    def __init__(self, *, mode: str = "widest",
                 weight_fn: Optional[Callable[[NodeID, NodeID, float], float]] = None
                 ) -> None:
        if mode not in ("widest", "minimax"):
            raise ValueError("mode must be 'widest' or 'minimax'.")
        super().__init__(weight_fn=weight_fn)
        self.mode = mode

    def solve(self, g: PathGraph, source: NodeID,
              target: Optional[NodeID] = None) -> ShortestPathResult:
        self._require_node(g, source)
        t0 = time.perf_counter()
        widest = self.mode == "widest"
        pred: Dict[NodeID, Optional[NodeID]] = {n: None for n in g.nodes}
        tie = count(1)
        pops = pushes = relax = 0
        if widest:
            cap: Dict[NodeID, float] = {n: -_INF for n in g.nodes}
            cap[source] = _INF
            # Max-heap via negation.
            heap: List[Tuple[float, int, NodeID]] = [(-_INF, next(tie), source)]
            settled: Set[NodeID] = set()
            while heap:
                neg, _, u = heapq.heappop(heap)
                pops += 1
                if u in settled:
                    continue
                settled.add(u)
                if target is not None and u == target:
                    break
                cu = -neg
                for v, base in g.neighbors(u):
                    w = self._w(u, v, base)
                    relax += 1
                    cand = min(cu, w)
                    if cand > cap[v]:
                        cap[v] = cand
                        pred[v] = u
                        heapq.heappush(heap, (-cand, next(tie), v))
                        pushes += 1
            dist_out = cap
        else:
            cap = {n: _INF for n in g.nodes}
            cap[source] = -_INF
            heap = [(-_INF, next(tie), source)]
            settled = set()
            while heap:
                val, _, u = heapq.heappop(heap)
                pops += 1
                if u in settled:
                    continue
                settled.add(u)
                if target is not None and u == target:
                    break
                cu = val
                for v, base in g.neighbors(u):
                    w = self._w(u, v, base)
                    relax += 1
                    cand = max(cu, w)
                    if cand < cap[v]:
                        cap[v] = cand
                        pred[v] = u
                        heapq.heappush(heap, (cand, next(tie), v))
                        pushes += 1
            dist_out = cap
        dt = (time.perf_counter() - t0) * 1000.0
        stats = PathStats(f"{self.name}[{self.mode}]", heap_pops=pops,
                          heap_pushes=pushes, edges_relaxed=relax,
                          nodes_settled=len(settled), runtime_ms=dt)
        return ShortestPathResult(f"{self.name}[{self.mode}]", source, target,
                                  dist_out, pred, stats=stats)


# ─────────────────────────────────────────────────────────────────────────────
# k-shortest paths
# ─────────────────────────────────────────────────────────────────────────────

class YenKShortestPaths:
    """Yen's algorithm for the k loopless shortest paths between two nodes.

    Worst case O(k·V·(E + V log V)); uses Dijkstra as the inner solver and
    therefore requires non-negative weights.
    """

    name = "Yen"

    def __init__(self, *, k: int = 3,
                 weight_fn: Optional[Callable[[NodeID, NodeID, float], float]] = None
                 ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1.")
        self.k = k
        self.weight_fn = weight_fn

    def _path_cost(self, g: PathGraph, nodes: Sequence[NodeID]) -> float:
        total = 0.0
        for a, b in zip(nodes[:-1], nodes[1:]):
            base = g.weight(a, b)
            if base == _INF:
                return _INF
            total += self.weight_fn(a, b, base) if self.weight_fn else base
        return total

    def solve(self, g: PathGraph, source: NodeID, target: NodeID
              ) -> KShortestPathsResult:
        if not g.has_node(source) or not g.has_node(target):
            raise KeyError("Both source and target must be in the graph.")

        def dijkstra_path(blocked_nodes, blocked_edges):
            dist, pred, _ = _dijkstra_core(
                g, source, target=target, weight=self.weight_fn,
                blocked_nodes=blocked_nodes, blocked_edges=blocked_edges,
                check_negative=True,
            )
            if dist.get(target, _INF) == _INF:
                return None, _INF
            # reconstruct
            seq: List[NodeID] = []
            cur: Optional[NodeID] = target
            seen: Set[NodeID] = set()
            while cur is not None:
                if cur in seen:
                    return None, _INF
                seen.add(cur)
                seq.append(cur)
                if cur == source:
                    break
                cur = pred.get(cur)
            seq.reverse()
            if not seq or seq[0] != source:
                return None, _INF
            return seq, dist[target]

        first, first_cost = dijkstra_path(set(), set())
        if first is None:
            return KShortestPathsResult(self.name, source, target, ())

        accepted: List[Path] = [Path(tuple(first), first_cost)]
        candidates: List[Tuple[float, int, List[NodeID]]] = []
        cand_tie = count(1)
        seen_paths: Set[Tuple[NodeID, ...]] = {tuple(first)}

        while len(accepted) < self.k:
            prev = accepted[-1].nodes
            for i in range(len(prev) - 1):
                spur_node = prev[i]
                root_path = list(prev[: i + 1])
                blocked_edges: Set[Tuple[NodeID, NodeID]] = set()
                for p in accepted:
                    if list(p.nodes[: i + 1]) == root_path and len(p.nodes) > i + 1:
                        a, b = p.nodes[i], p.nodes[i + 1]
                        blocked_edges.add((a, b))
                        if not g.directed:
                            blocked_edges.add((b, a))
                # Block the root-path nodes (except the spur node) to keep loopless.
                blocked_nodes = set(root_path[:-1])

                spur_dist, spur_pred, _ = _dijkstra_core(
                    g, spur_node, target=target, weight=self.weight_fn,
                    blocked_nodes=blocked_nodes, blocked_edges=blocked_edges,
                    check_negative=True,
                )
                if spur_dist.get(target, _INF) == _INF:
                    continue
                spur_seq: List[NodeID] = []
                cur: Optional[NodeID] = target
                ok = True
                seen: Set[NodeID] = set()
                while cur is not None:
                    if cur in seen:
                        ok = False
                        break
                    seen.add(cur)
                    spur_seq.append(cur)
                    if cur == spur_node:
                        break
                    cur = spur_pred.get(cur)
                if not ok or not spur_seq or spur_seq[-1] != spur_node:
                    continue
                spur_seq.reverse()                      # spur_node .. target
                total_path = root_path[:-1] + spur_seq
                key = tuple(total_path)
                if key in seen_paths:
                    continue
                cost = self._path_cost(g, total_path)
                if cost == _INF:
                    continue
                heapq.heappush(candidates, (cost, next(cand_tie), total_path))
                seen_paths.add(key)

            if not candidates:
                break
            cost, _, best = heapq.heappop(candidates)
            accepted.append(Path(tuple(best), cost))

        return KShortestPathsResult(self.name, source, target, tuple(accepted))


# ─────────────────────────────────────────────────────────────────────────────
# Heuristics for A*
# ─────────────────────────────────────────────────────────────────────────────

class Heuristics:
    """Factory for admissible A* heuristics.

    Each method returns a ``Callable[[NodeID], float]`` estimating the remaining
    cost from a node to ``target``.  The metric variants read node coordinates
    via ``graph.coord(node)`` (set them with ``add_node(n, coord=...)`` or
    ``set_coord``).  ``scale`` lets you match the heuristic to the edge-cost
    units (and keep it admissible).
    """

    @staticmethod
    def zero() -> Callable[[NodeID], float]:
        return lambda _n: 0.0

    @staticmethod
    def _coords(graph: PathGraph, node: NodeID, target: NodeID
                ) -> Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        a, b = graph.coord(node), graph.coord(target)
        if a is None or b is None:
            return None
        return a, b

    @classmethod
    def euclidean(cls, graph: PathGraph, target: NodeID, *, scale: float = 1.0
                  ) -> Callable[[NodeID], float]:
        def h(node: NodeID) -> float:
            cc = cls._coords(graph, node, target)
            if cc is None:
                return 0.0
            a, b = cc
            return scale * math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
        return h

    @classmethod
    def manhattan(cls, graph: PathGraph, target: NodeID, *, scale: float = 1.0
                  ) -> Callable[[NodeID], float]:
        def h(node: NodeID) -> float:
            cc = cls._coords(graph, node, target)
            if cc is None:
                return 0.0
            a, b = cc
            return scale * sum(abs(x - y) for x, y in zip(a, b))
        return h

    @classmethod
    def chebyshev(cls, graph: PathGraph, target: NodeID, *, scale: float = 1.0
                  ) -> Callable[[NodeID], float]:
        def h(node: NodeID) -> float:
            cc = cls._coords(graph, node, target)
            if cc is None:
                return 0.0
            a, b = cc
            return scale * max((abs(x - y) for x, y in zip(a, b)), default=0.0)
        return h

    @classmethod
    def octile(cls, graph: PathGraph, target: NodeID, *, scale: float = 1.0
               ) -> Callable[[NodeID], float]:
        """Exact heuristic for an 8-connected grid with diagonal cost √2.

        Defined on the first two coordinate axes; if a node has fewer than two
        coordinates the heuristic degrades to 0 (admissible) rather than raising,
        matching the missing-coordinate behaviour of the other heuristics."""
        d2 = math.sqrt(2.0)
        def h(node: NodeID) -> float:
            cc = cls._coords(graph, node, target)
            if cc is None:
                return 0.0
            a, b = cc
            if len(a) < 2 or len(b) < 2:
                return 0.0
            dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
            return scale * ((dx + dy) + (d2 - 2.0) * min(dx, dy))
        return h

    @classmethod
    def haversine(cls, graph: PathGraph, target: NodeID, *,
                  radius: float = 6_371_000.0) -> Callable[[NodeID], float]:
        """Great-circle distance for (lat, lon) coordinates in degrees."""
        def h(node: NodeID) -> float:
            cc = cls._coords(graph, node, target)
            if cc is None:
                return 0.0
            a, b = cc
            lat1, lon1 = math.radians(a[0]), math.radians(a[1])
            lat2, lon2 = math.radians(b[0]), math.radians(b[1])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            hav = (math.sin(dlat / 2) ** 2 +
                   math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
            return 2.0 * radius * math.asin(min(1.0, math.sqrt(hav)))
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Path utilities
# ─────────────────────────────────────────────────────────────────────────────

class PathAnalyzer:
    """Stateless helpers for inspecting and validating paths on a graph."""

    @staticmethod
    def path_cost(g: PathGraph, nodes: Sequence[NodeID], *,
                  weight_fn: Optional[Callable[[NodeID, NodeID, float], float]] = None
                  ) -> float:
        total = 0.0
        for a, b in zip(nodes[:-1], nodes[1:]):
            base = g.weight(a, b)
            if base == _INF:
                return _INF
            total += weight_fn(a, b, base) if weight_fn else base
        return total

    @staticmethod
    def is_valid_path(g: PathGraph, nodes: Sequence[NodeID]) -> bool:
        if not nodes:
            return False
        if len(nodes) == 1:
            return g.has_node(nodes[0])
        return all(g.has_edge(a, b) for a, b in zip(nodes[:-1], nodes[1:]))

    @staticmethod
    def is_simple(nodes: Sequence[NodeID]) -> bool:
        return len(set(nodes)) == len(nodes)

    @staticmethod
    def bottleneck_edge(g: PathGraph, nodes: Sequence[NodeID]
                        ) -> Optional[Tuple[NodeID, NodeID, float]]:
        """The maximum-weight edge on the path (the 'bottleneck')."""
        worst: Optional[Tuple[NodeID, NodeID, float]] = None
        for a, b in zip(nodes[:-1], nodes[1:]):
            w = g.weight(a, b)
            if worst is None or w > worst[2]:
                worst = (a, b, w)
        return worst

    @staticmethod
    def edge_list(nodes: Sequence[NodeID]) -> List[Tuple[NodeID, NodeID]]:
        return list(zip(nodes[:-1], nodes[1:]))


# ─────────────────────────────────────────────────────────────────────────────
# Grid pathfinding
# ─────────────────────────────────────────────────────────────────────────────

class GridPathfinder:
    """Shortest paths over a 2-D occupancy grid.

    Cells are addressed as ``(row, col)``.  A cell is passable unless its value
    equals ``obstacle`` (or ``passable(value)`` returns False).  The cost of
    *entering* a cell defaults to its numeric value when ``weighted=True`` else
    1.0.  Diagonal moves (8-connectivity) cost ``diagonal_cost`` times the
    entry cost.

    Example
    -------
    >>> grid = [[1, 1, 9], [9, 1, 1], [1, 1, 1]]   # 9 = wall
    >>> gp = GridPathfinder(grid, obstacle=9, diagonal=True)
    >>> res = gp.solve((0, 0), (2, 2))
    >>> res.path_to((2, 2))
    """

    def __init__(self, grid: Sequence[Sequence[float]], *,
                 obstacle: Any = None,
                 passable: Optional[Callable[[Any], bool]] = None,
                 diagonal: bool = False,
                 diagonal_cost: float = math.sqrt(2.0),
                 weighted: bool = False) -> None:
        if not grid or not grid[0]:
            raise ValueError("grid must be a non-empty 2-D sequence.")
        self.grid = [list(row) for row in grid]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        if any(len(r) != self.cols for r in self.grid):
            raise ValueError("grid must be rectangular.")
        self.obstacle = obstacle
        self.passable = passable
        self.diagonal = diagonal
        self.diagonal_cost = float(diagonal_cost)
        self.weighted = weighted
        self._graph: Optional[PathGraph] = None

    def _is_passable(self, r: int, c: int) -> bool:
        val = self.grid[r][c]
        if self.passable is not None:
            return bool(self.passable(val))
        return val != self.obstacle

    def _entry_cost(self, r: int, c: int) -> float:
        if not self.weighted:
            return 1.0
        try:
            return float(self.grid[r][c])
        except (TypeError, ValueError):
            return 1.0

    def _min_entry_cost(self) -> float:
        """Cheapest cost of entering any passable cell (1.0 for unweighted)."""
        if not self.weighted:
            return 1.0
        costs = [self._entry_cost(r, c)
                 for r in range(self.rows) for c in range(self.cols)
                 if self._is_passable(r, c)]
        return min(costs) if costs else 1.0

    def _admissible_heuristic(self, g: PathGraph, goal: Tuple[int, int]
                              ) -> Callable[[NodeID], float]:
        """A* heuristic guaranteed admissible for *this* grid.

        A generic octile/manhattan heuristic assumes unit step costs and a √2
        diagonal; that overestimates (and breaks A* optimality) when cells cost
        less than 1 or when ``diagonal_cost`` differs from √2.  We instead use
        (minimum number of moves) × (cheapest possible move cost), which is a
        true lower bound on the remaining path cost in every configuration.
        """
        gc = g.coord(goal)
        m = self._min_entry_cost()
        if self.diagonal:
            diag_mult = min(self.diagonal_cost, 2.0)   # ≥2 ⇒ diagonals never help

            def h(node: NodeID) -> float:
                c = g.coord(node)
                if c is None or gc is None:
                    return 0.0
                dx, dy = abs(c[0] - gc[0]), abs(c[1] - gc[1])
                lo, hi = min(dx, dy), max(dx, dy)
                return ((hi - lo) + lo * diag_mult) * m
        else:
            def h(node: NodeID) -> float:
                c = g.coord(node)
                if c is None or gc is None:
                    return 0.0
                return (abs(c[0] - gc[0]) + abs(c[1] - gc[1])) * m
        return h

    def to_graph(self) -> PathGraph:
        """Materialise (and cache) the grid as a directed PathGraph with node
        coordinates set to ``(col, row)`` for geometric heuristics."""
        if self._graph is not None:
            return self._graph
        g = PathGraph(directed=True, allow_negative=False)
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        diag = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for r in range(self.rows):
            for c in range(self.cols):
                if not self._is_passable(r, c):
                    continue
                g.add_node((r, c), coord=(float(c), float(r)))
        for r in range(self.rows):
            for c in range(self.cols):
                if not self._is_passable(r, c):
                    continue
                for dr, dc in steps:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and self._is_passable(nr, nc):
                        g.add_edge((r, c), (nr, nc), self._entry_cost(nr, nc))
                if self.diagonal:
                    for dr, dc in diag:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols and self._is_passable(nr, nc):
                            g.add_edge((r, c), (nr, nc),
                                       self.diagonal_cost * self._entry_cost(nr, nc))
        self._graph = g
        return g

    def solve(self, start: Tuple[int, int], goal: Tuple[int, int], *,
              method: str = "astar") -> ShortestPathResult:
        g = self.to_graph()
        if not g.has_node(start):
            raise ValueError(f"start {start} is out of bounds or blocked.")
        if not g.has_node(goal):
            raise ValueError(f"goal {goal} is out of bounds or blocked.")
        if method == "astar":
            return AStar(heuristic=self._admissible_heuristic(g, goal)).solve(
                g, start, goal)
        if method == "dijkstra":
            return Dijkstra().solve(g, start, goal)
        if method == "bfs":
            return BFSShortestPath().solve(g, start, goal)
        raise ValueError("method must be 'astar', 'dijkstra' or 'bfs'.")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience dispatchers
# ─────────────────────────────────────────────────────────────────────────────

def shortest_path(g: PathGraph, source: NodeID,
                  target: Optional[NodeID] = None, *,
                  method: str = "auto",
                  heuristic: Optional[Callable[[NodeID], float]] = None,
                  weight_fn: Optional[Callable[[NodeID, NodeID, float], float]] = None
                  ) -> ShortestPathResult:
    """Single-source shortest path with automatic algorithm selection.

    method : {"auto", "dijkstra", "bfs", "zero_one", "dial", "bellman_ford",
              "spfa", "astar", "bidirectional"}

    ``"auto"`` inspects the graph and picks (in priority order):
      * any negative edge             -> SPFA (Bellman-Ford family)
      * a heuristic + target          -> A*
      * unweighted edges              -> BFS
      * weights in {0, 1}             -> 0-1 BFS
      * a target and no heuristic     -> bidirectional Dijkstra
      * otherwise                     -> Dijkstra

    The negative-edge check takes precedence so that ``auto`` never selects a
    solver that would raise on the given graph.
    """
    if method == "auto":
        # Correctness first: A* / Dijkstra cannot handle negative edges, so a
        # negative-weight graph forces a Bellman-Ford-family solver even when a
        # heuristic was supplied.
        if g.has_negative_edges():
            method = "spfa"
        elif heuristic is not None and target is not None:
            method = "astar"
        elif g.is_unweighted():
            method = "bfs"
        elif g.is_zero_one():
            method = "zero_one"
        elif target is not None:
            method = "bidirectional"
        else:
            method = "dijkstra"

    if method == "dijkstra":
        return Dijkstra(weight_fn=weight_fn).solve(g, source, target)
    if method == "bfs":
        return BFSShortestPath(weight_fn=weight_fn).solve(g, source, target)
    if method == "zero_one":
        return ZeroOneBFS(weight_fn=weight_fn).solve(g, source, target)
    if method == "dial":
        return DialDijkstra(weight_fn=weight_fn).solve(g, source, target)
    if method == "bellman_ford":
        return BellmanFord(weight_fn=weight_fn).solve(g, source, target)
    if method == "spfa":
        return SPFA(weight_fn=weight_fn).solve(g, source, target)
    if method == "astar":
        return AStar(heuristic=heuristic, weight_fn=weight_fn).solve(g, source, target)
    if method == "bidirectional":
        if target is None:
            return Dijkstra(weight_fn=weight_fn).solve(g, source, target)
        return BidirectionalDijkstra(weight_fn=weight_fn).solve(g, source, target)
    raise ValueError(f"Unknown method {method!r}.")


def all_pairs_shortest_path(g: PathGraph, *, method: str = "auto"
                            ) -> AllPairsResult:
    """All-pairs shortest paths.  ``"auto"`` chooses Johnson for sparse graphs
    (and any graph with negative edges) and Floyd-Warshall for dense graphs."""
    if method == "auto":
        n, m = g.num_nodes, g.num_edges
        dense = m > n * n / 4 if n else False
        method = "floyd_warshall" if (dense and not g.has_negative_edges()) else "johnson"
    if method == "floyd_warshall":
        return FloydWarshall().solve(g)
    if method == "johnson":
        return Johnson().solve(g)
    raise ValueError(f"Unknown method {method!r}.")


def k_shortest_paths(g: PathGraph, source: NodeID, target: NodeID, *,
                     k: int = 3,
                     weight_fn: Optional[Callable[[NodeID, NodeID, float], float]] = None
                     ) -> KShortestPathsResult:
    """Thin wrapper over :class:`YenKShortestPaths`."""
    return YenKShortestPaths(k=k, weight_fn=weight_fn).solve(g, source, target)


# ─────────────────────────────────────────────────────────────────────────────
# Payload builders
# ─────────────────────────────────────────────────────────────────────────────

def build_shortest_path_payload(g: PathGraph,
                                result: Optional[ShortestPathResult] = None, *,
                                target: Optional[NodeID] = None
                                ) -> Dict[str, Any]:
    """Frontend-ready JSON-serialisable payload describing the graph and an
    optional shortest-path result (with the path highlighted)."""
    path_nodes: List[NodeID] = []
    if result is not None:
        tgt = target if target is not None else result.target
        if tgt is not None:
            path_nodes = result.path_to(tgt)
    path_set = set(path_nodes)
    path_edges = set(zip(path_nodes[:-1], path_nodes[1:]))

    nodes = [
        {
            "id": n,
            "label": str(n),
            "degree": g.degree(n),
            "coord": list(g.coord(n)) if g.coord(n) is not None else None,
            "onPath": n in path_set,
            "distance": (result.distance_to(n) if result and result.distance_to(n) < _INF
                         else None),
        }
        for n in g.nodes
    ]
    edges = [
        {
            "source": u,
            "target": v,
            "weight": w,
            "onPath": (u, v) in path_edges or (not g.directed and (v, u) in path_edges),
        }
        for u, v, w in g.edges
    ]
    # JSON has no infinity: an unreachable target reports a null cost (matching
    # the per-node "distance" field and build_a2ui_payload).
    path_cost: Optional[float] = None
    if result is not None:
        tgt = target if target is not None else result.target
        if tgt is not None:
            d = result.distance_to(tgt)
            path_cost = d if d < _INF else None

    payload: Dict[str, Any] = {
        "screen": "shortestPaths",
        "directed": g.directed,
        "nNodes": g.num_nodes,
        "nEdges": g.num_edges,
        "nodes": nodes,
        "edges": edges,
        "result": result.to_dict() if result else None,
        "path": [str(n) for n in path_nodes],
        "pathCost": path_cost,
    }
    return payload


def build_a2ui_payload(g: PathGraph,
                       result: Optional[ShortestPathResult] = None, *,
                       target: Optional[NodeID] = None,
                       layout: str = "force") -> Dict[str, Any]:
    """Visualisation-ready payload: stat cards, highlighted path, layout hints.

    layout : {"force", "circular", "grid", "coord"}.  ``"coord"`` uses node
    coordinates directly (ideal for grid / geographic graphs).
    """
    base = build_shortest_path_payload(g, result, target=target)

    tgt = target if target is not None else (result.target if result else None)
    cost = None
    hops = None
    if result is not None and tgt is not None:
        d = result.distance_to(tgt)
        cost = round(d, 6) if d < _INF else None
        p = result.path_to(tgt)
        hops = max(0, len(p) - 1) if p else None

    reachable = len(result.reachable_nodes) if result else g.num_nodes
    stat_cards = [
        {"id": "nNodes",    "label": "Nodes",     "value": g.num_nodes},
        {"id": "nEdges",    "label": "Edges",     "value": g.num_edges},
        {"id": "directed",  "label": "Directed",  "value": g.directed},
        {"id": "reachable", "label": "Reachable", "value": reachable},
        {"id": "pathCost",  "label": "Path Cost", "value": cost},
        {"id": "pathHops",  "label": "Path Hops", "value": hops},
        {"id": "algorithm", "label": "Algorithm",
         "value": result.algorithm if result else None},
    ]

    # Layout positions.
    positions: Dict[NodeID, Tuple[float, float]] = {}
    ns = g.nodes
    n = len(ns)
    if layout == "coord":
        for node in ns:
            c = g.coord(node)
            if c is not None and len(c) >= 2:
                positions[node] = (c[0], c[1])
    elif layout == "circular" and n:
        for i, node in enumerate(ns):
            theta = 2 * math.pi * i / n
            positions[node] = (math.cos(theta), math.sin(theta))
    elif layout == "grid" and n:
        cols = max(1, math.isqrt(n))
        for i, node in enumerate(ns):
            positions[node] = (float(i % cols), float(i // cols))

    for nd in base["nodes"]:
        if nd["id"] in positions:
            nd["x"], nd["y"] = positions[nd["id"]]

    return {
        "statCards": stat_cards,
        "graphView": base,
        "result": result.to_dict() if result else None,
        "layoutHint": layout,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    print("=" * 64)
    print("shortest_paths_template.py — end-to-end demo")
    print("=" * 64)

    # ── 1. Build a weighted directed graph ────────────────────────────────────
    print("\n[1] Weighted directed graph + Dijkstra")
    g = PathGraph.from_edge_list([
        ("A", "B", 4), ("A", "C", 2), ("C", "B", 1), ("B", "D", 5),
        ("C", "D", 8), ("C", "E", 10), ("D", "E", 2), ("D", "F", 6),
        ("E", "F", 3),
    ], directed=True)
    res = Dijkstra().solve(g, "A")
    print(f"    graph: {g}")
    for n in g.nodes:
        print(f"    A -> {n}: dist={res.distance_to(n):>4}  path={res.path_to(n)}")

    # ── 2. A* with coordinates ────────────────────────────────────────────────
    print("\n[2] A* with a Euclidean heuristic")
    gc = PathGraph(directed=False)
    coords = {"S": (0, 0), "A": (1, 2), "B": (2, 1), "C": (3, 3), "T": (4, 0)}
    for nid, xy in coords.items():
        gc.add_node(nid, coord=xy)
    gc.add_edges_from([("S", "A", 3), ("S", "B", 2.5), ("A", "C", 3),
                       ("B", "C", 3.2), ("B", "T", 4), ("C", "T", 2)])
    h = Heuristics.euclidean(gc, "T")
    ares = AStar(heuristic=h).solve(gc, "S", "T")
    print(f"    S -> T: cost={ares.distance_to('T'):.2f}  path={ares.path_to('T')}")
    print(f"    nodes settled: {ares.stats.nodes_settled}")

    # ── 3. Negative weights: Bellman-Ford & SPFA ──────────────────────────────
    print("\n[3] Negative weights (Bellman-Ford / SPFA)")
    gn = PathGraph.from_edge_list([
        ("s", "a", 4), ("s", "b", 5), ("a", "b", -3), ("b", "c", 4),
        ("a", "c", 6),
    ], directed=True)
    bf = BellmanFord().solve(gn, "s")
    print(f"    Bellman-Ford  s->c: {bf.distance_to('c')}  path={bf.path_to('c')}")
    sp = SPFA().solve(gn, "s")
    print(f"    SPFA          s->c: {sp.distance_to('c')}  neg_cycle={sp.has_negative_cycle}")

    # ── 4. Negative cycle detection ───────────────────────────────────────────
    print("\n[4] Negative-cycle detection")
    gc2 = PathGraph.from_edge_list([
        ("x", "y", 1), ("y", "z", -3), ("z", "x", 1),
    ], directed=True)
    bf2 = BellmanFord().solve(gc2, "x")
    print(f"    has_negative_cycle={bf2.has_negative_cycle}  cycle={bf2.negative_cycle}")

    # ── 5. DAG longest path ───────────────────────────────────────────────────
    print("\n[5] DAG shortest & longest path")
    dag = PathGraph.from_edge_list([
        (0, 1, 3), (0, 2, 2), (1, 3, 4), (2, 3, 1), (3, 4, 2), (2, 4, 9),
    ], directed=True)
    short = DAGShortestPath(mode="shortest").solve(dag, 0)
    longest = DAGShortestPath(mode="longest").solve(dag, 0)
    print(f"    shortest 0->4: {short.distance_to(4)}  path={short.path_to(4)}")
    print(f"    longest  0->4: {longest.distance_to(4)}  path={longest.path_to(4)}")

    # ── 6. Bidirectional Dijkstra ─────────────────────────────────────────────
    print("\n[6] Bidirectional Dijkstra")
    bd = BidirectionalDijkstra().solve(g, "A", "F")
    print(f"    A -> F: cost={bd.distance_to('F')}  path={bd.path_to('F')}")
    one = Dijkstra().solve(g, "A", "F")
    print(f"    (unidirectional cost for check: {one.distance_to('F')})")

    # ── 7. All-pairs: Floyd-Warshall & Johnson ────────────────────────────────
    print("\n[7] All-pairs shortest paths")
    fw = FloydWarshall().solve(g)
    jo = Johnson().solve(g)
    print(f"    Floyd-Warshall A->F = {fw.distance('A', 'F')}, "
          f"path={fw.path('A', 'F')}")
    print(f"    Johnson        A->F = {jo.distance('A', 'F')}, "
          f"path={jo.path('A', 'F')}")
    print(f"    diameter={fw.diameter()}  radius={fw.radius()}  center={fw.center()}")

    # ── 8. k-shortest paths (Yen) ─────────────────────────────────────────────
    print("\n[8] Yen's k-shortest paths")
    ksp = YenKShortestPaths(k=3).solve(g, "A", "F")
    for i, p in enumerate(ksp, 1):
        print(f"    #{i}: cost={p.cost}  {list(p.nodes)}")

    # ── 9. Widest / bottleneck path ───────────────────────────────────────────
    print("\n[9] Widest (max-min capacity) path")
    cap = PathGraph.from_edge_list([
        ("A", "B", 3), ("A", "C", 5), ("B", "D", 4), ("C", "D", 2),
        ("C", "E", 6), ("D", "E", 7),
    ], directed=True, on_duplicate="max")
    wp = BottleneckPath(mode="widest").solve(cap, "A", "E")
    print(f"    widest A->E bottleneck={wp.distance_to('E')}  path={wp.path_to('E')}")

    # ── 10. Grid pathfinding ──────────────────────────────────────────────────
    print("\n[10] Grid pathfinding (A* on an occupancy grid)")
    grid = [
        [1, 1, 1, 1, 1],
        [1, 9, 9, 9, 1],
        [1, 1, 1, 9, 1],
        [9, 9, 1, 9, 1],
        [1, 1, 1, 1, 1],
    ]
    gp = GridPathfinder(grid, obstacle=9, diagonal=True)
    gres = gp.solve((0, 0), (4, 0))
    print(f"    (0,0) -> (4,0): cost={gres.distance_to((4, 0)):.3f}")
    print(f"    path={gres.path_to((4, 0))}")

    # ── 11. Auto dispatcher + payload ─────────────────────────────────────────
    print("\n[11] Auto dispatcher & payloads")
    auto = shortest_path(g, "A", "F")
    print(f"    shortest_path(auto) chose: {auto.algorithm}")
    payload = build_a2ui_payload(g, auto, target="F", layout="circular")
    print(f"    a2ui statCards: {[c['id'] for c in payload['statCards']]}")
    print(f"    highlighted path: {payload['graphView']['path']}")

    print("\n[Done]")


if __name__ == "__main__":
    _demo()
