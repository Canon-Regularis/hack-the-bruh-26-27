# Shortest Paths

Pure-Python, zero-dependency shortest-path library — 14 solvers spanning
single-source, all-pairs, bidirectional, heuristic, bottleneck and k-shortest
problems, plus a 2-D grid pathfinder, a heuristic factory, and an automatic
algorithm dispatcher. No NumPy / SciPy / NetworkX required.

## Pick

| Goal | Use |
|---|---|
| Weighted, non-negative single-source | `Dijkstra` |
| Non-negative **integer** weights, small range | `DialDijkstra` |
| Weights are all 0 or 1 | `ZeroOneBFS` |
| Unweighted (fewest hops) | `BFSShortestPath` |
| Negative edges (no negative cycle) | `BellmanFord` / `SPFA` |
| Detect / extract a negative cycle | `BellmanFord`, `SPFA` |
| Single source on a DAG (shortest **or longest**) | `DAGShortestPath` |
| Point-to-point on a big graph | `BidirectionalDijkstra` |
| Heuristic-guided point-to-point | `AStar` |
| Dense all-pairs | `FloydWarshall` |
| Sparse all-pairs (incl. negative edges) | `Johnson` |
| Reachability / transitive closure | `FloydWarshall.transitive_closure` |
| Max-min capacity (widest) / min-max edge (minimax) path | `BottleneckPath` |
| 2nd, 3rd, … shortest loopless paths | `YenKShortestPaths` |
| A* heuristics (euclidean / manhattan / octile / haversine …) | `Heuristics` |
| Validate a path, compute its cost / bottleneck | `PathAnalyzer` |
| Maze / game-map / grid routing | `GridPathfinder` |
| "Just pick the right algorithm" | `shortest_path`, `all_pairs_shortest_path` |
| Frontend JSON (competition / hackathon screen) | `build_shortest_path_payload`, `build_a2ui_payload` |

## Install

```bash
# No runtime deps — copy the file.
pip install pytest  # tests only
```

## Quick start

```python
from shortest_paths_template import (
    PathGraph, Dijkstra, AStar, Heuristics, YenKShortestPaths,
    shortest_path, build_a2ui_payload,
)

# Build a weighted directed graph
g = PathGraph.from_edge_list([
    ("A", "B", 4), ("A", "C", 2), ("C", "B", 1), ("B", "D", 5),
    ("D", "E", 2), ("E", "F", 3), ("D", "F", 6),
], directed=True)

# Single-source shortest paths
r = Dijkstra().solve(g, "A")
r.distance_to("F")          # 13.0
r.path_to("F")              # ['A', 'C', 'B', 'D', 'E', 'F']

# Point-to-point, auto-selected algorithm
r = shortest_path(g, "A", "F")
r.algorithm                 # 'BidirectionalDijkstra'

# 3 shortest loopless paths
ksp = YenKShortestPaths(k=3).solve(g, "A", "F")
[p.cost for p in ksp]       # [13.0, 14.0, 14.0]

# Frontend payload (path highlighted)
payload = build_a2ui_payload(g, r, target="F", layout="circular")
```

## API

### `PathGraph` — the container

```python
g = PathGraph(directed=False, allow_negative=True, on_duplicate="min")

g.add_node("a", coord=(0.0, 1.0))      # optional coordinate (for A* heuristics)
g.add_edge("a", "b", 2.5, name="road") # weight + arbitrary metadata
g.add_edges_from([("a", "b"), ("b", "c", 3.0), ("c", "d", 1.0, {"k": 1})])

g.has_node("a"); g.has_edge("a", "b")
g.weight("a", "b")          # 2.5   (inf if no edge)
g.neighbors("a")            # iterator of (neighbour, weight)
g.degree("a"); g.coord("a")
g.nodes; g.edges            # edges deduplicated for undirected graphs
g.num_nodes; g.num_edges

g.reverse()                 # transpose (a copy for undirected graphs)
g.copy()
g.to_matrix(nodes=[...])    # (dense matrix, node order)
```

**Parameters**

| Param | Meaning |
|---|---|
| `directed` | one-way edges if `True` |
| `allow_negative` | reject negative weights at `add_edge` time if `False` |
| `on_duplicate` | `"min"` (default), `"max"`, `"sum"`, `"first"`, `"last"` — reduction for repeated edges. Use `"max"` for capacity / widest-path graphs |

**Constructors**

```python
PathGraph.from_edge_list([(u, v), (u, v, w), (u, v, w, meta)], directed=True)
PathGraph.from_adjacency_dict({"a": {"b": 2.0}}, directed=True)   # or {"a": ["b", ...]}
PathGraph.from_adjacency_matrix(matrix, nodes=[...], no_edge=None)
```

> In `from_adjacency_matrix`, a `0` is a **real zero-weight edge**; use `None`
> / `inf` / `no_edge` to mark an absent edge.

### Single-source solvers

Every single-source solver shares one interface:

```python
solver.solve(graph, source, target=None) -> ShortestPathResult
solver.shortest_path(graph, source, target) -> Path        # convenience
```

| Solver | Constraint | Complexity |
|---|---|---|
| `Dijkstra()` | weights ≥ 0 | O((V+E) log V) |
| `DialDijkstra()` | integer weights ≥ 0 | O(V + E + maxW·V) |
| `ZeroOneBFS()` | weights ∈ {0, 1} | O(V + E) |
| `BFSShortestPath()` | unweighted (ignores weights) | O(V + E) |
| `BellmanFord()` | any weights | O(V·E) |
| `SPFA(slf=True)` | any weights | O(V·E) worst, fast in practice |
| `DAGShortestPath(mode="shortest")` | directed acyclic | O(V + E) |

```python
from shortest_paths_template import BellmanFord, DAGShortestPath

# Negative edges + cycle reporting
r = BellmanFord().solve(g, "s")
r.has_negative_cycle        # bool
r.negative_cycle            # ('y', 'z', 'x', 'y')  when present

# Longest path on a DAG (critical-path / scheduling)
r = DAGShortestPath(mode="longest").solve(dag, source=0)
r.distance_to(4)            # longest distance
```

`target` enables early exit on `Dijkstra` / `DialDijkstra` / `ZeroOneBFS` /
`BFSShortestPath`.

### Bidirectional & heuristic

```python
from shortest_paths_template import BidirectionalDijkstra, AStar, Heuristics

BidirectionalDijkstra().solve(g, "A", "F")     # requires target, weights ≥ 0

# A* — heuristic estimates remaining cost to the target.
# Optimal for any *admissible* heuristic (closed nodes are re-expanded, so the
# heuristic need not be consistent/monotone); zero overhead when it is.
h = Heuristics.euclidean(g, target="T")        # needs node coords
AStar(heuristic=h).solve(g, "S", "T")

# Weighted A* — epsilon>1 is faster, bounded-suboptimal by factor epsilon
AStar(heuristic=h, epsilon=1.5).solve(g, "S", "T")
```

### All-pairs

```python
from shortest_paths_template import FloydWarshall, Johnson

res = FloydWarshall().solve(g)         # O(V³), dense
res = Johnson().solve(g)               # O(V·E + V² log V), sparse, neg-edge OK

res.distance("A", "F")                 # 13.0
res.path("A", "F")                     # ['A', 'C', 'B', 'D', 'E', 'F']
res.eccentricity("A"); res.diameter(); res.radius(); res.center()
res.has_negative_cycle

FloydWarshall().transitive_closure(g)  # {u: {reachable nodes}, ...}
```

### Bottleneck objectives

```python
from shortest_paths_template import BottleneckPath

# Widest path: maximise the minimum edge (max-flow-style capacity routing)
BottleneckPath(mode="widest").solve(capacity_graph, "A", "E")

# Minimax path: minimise the maximum edge (e.g. minimum-bottleneck routing)
BottleneckPath(mode="minimax").solve(g, "A", "C")
# result.dist[v] holds the optimal bottleneck value to v
```

### k-shortest paths

```python
from shortest_paths_template import YenKShortestPaths, k_shortest_paths

res = YenKShortestPaths(k=3).solve(g, "A", "F")    # weights ≥ 0
res.costs                  # [13.0, 14.0, 14.0]  (ascending)
res.best()                 # the optimal Path
for p in res:
    p.nodes, p.cost, p.length

k_shortest_paths(g, "A", "F", k=3)                 # functional wrapper
```

### Heuristics (for A*)

Each returns a `Callable[[node], float]` estimating remaining cost to `target`.
Metric variants read `graph.coord(node)`.

```python
Heuristics.zero()
Heuristics.euclidean(g, target, scale=1.0)
Heuristics.manhattan(g, target)
Heuristics.chebyshev(g, target)
Heuristics.octile(g, target)        # exact for 8-connected grids (diag = √2)
Heuristics.haversine(g, target)     # great-circle for (lat, lon) coords
```

### `GridPathfinder` — routing over a 2-D grid

```python
from shortest_paths_template import GridPathfinder

grid = [
    [1, 1, 1, 1, 1],
    [1, 9, 9, 9, 1],     # 9 = wall
    [1, 1, 1, 9, 1],
    [9, 9, 1, 9, 1],
    [1, 1, 1, 1, 1],
]
gp = GridPathfinder(grid, obstacle=9, diagonal=True, weighted=False)
r = gp.solve((0, 0), (4, 0), method="astar")    # cells addressed as (row, col)
r.path_to((4, 0))           # [(0,0), (1,0), (2,1), ...]
```

| Param | Meaning |
|---|---|
| `obstacle` | cell value that marks a wall (or pass `passable=fn`) |
| `diagonal` | enable 8-connectivity (cost `diagonal_cost`, default √2) |
| `weighted` | edge cost = value of the cell being entered (else 1.0) |
| `method` | `"astar"` (auto octile/manhattan heuristic), `"dijkstra"`, `"bfs"` |

### Dispatchers

```python
shortest_path(g, source, target=None, method="auto", heuristic=None, weight_fn=None)
all_pairs_shortest_path(g, method="auto")     # Johnson if sparse/negative, else Floyd
k_shortest_paths(g, source, target, k=3)
```

`method="auto"` for `shortest_path` picks, in priority order: negative edge →
SPFA; heuristic+target → A*; unweighted → BFS; {0,1} → 0-1 BFS; target →
bidirectional Dijkstra; otherwise Dijkstra. (The negative-edge check comes first
so `auto` never selects a solver that would raise on the given graph.)

### The `weight_fn` hook

Every solver accepts `weight_fn(u, v, base_weight) -> float` to transform costs
at solve time — penalties, tolls, time-dependent costs — without rebuilding the
graph:

```python
# Add a 100-unit penalty to one edge, leave the rest unchanged
penalised = Dijkstra(weight_fn=lambda u, v, w: w + (100 if (u, v) == ("A", "C") else 0))
penalised.solve(g, "A")
```

## Data types

All result containers are **frozen dataclasses**.

| Type | Key members |
|---|---|
| `ShortestPathResult` | `dist`, `pred`, `distance_to`, `path_to`, `edge_path_to`, `get_path`, `reachable_nodes`, `has_negative_cycle`, `negative_cycle`, `stats` |
| `AllPairsResult` | `distance`, `path`, `eccentricity`, `diameter`, `radius`, `center`, `has_negative_cycle` |
| `KShortestPathsResult` | indexable / iterable, `costs`, `best()` |
| `Path` | `nodes`, `cost`, `length`, `edges()` |
| `PathStats` | `heap_pops`, `heap_pushes`, `edges_relaxed`, `nodes_settled`, `runtime_ms` |

```python
r = Dijkstra().solve(g, "A")
r.stats.runtime_ms          # per-run telemetry for benchmarking
r.stats.nodes_settled       # how much of the graph was explored
```

## Hackathon tips

**Point-to-point routing with coordinates (fastest):**
```python
for nid, (x, y) in positions.items():
    g.add_node(nid, coord=(x, y))
r = AStar(heuristic=Heuristics.euclidean(g, goal)).solve(g, start, goal)
```

**Road network with one-way penalties / tolls:**
```python
def cost(u, v, w):
    return w * toll_multiplier.get((u, v), 1.0)
shortest_path(g, src, dst, method="dijkstra", weight_fn=cost)
```

**Currency arbitrage / negative-cycle detection:**
```python
r = BellmanFord().solve(g, start)
if r.has_negative_cycle:
    print("arbitrage cycle:", r.negative_cycle)
```

**Network reliability (widest bottleneck path):**
```python
g = PathGraph(directed=True, on_duplicate="max")   # keep the fattest parallel link
BottleneckPath(mode="widest").solve(g, src, dst)
```

**Project scheduling (critical path = longest path on a DAG):**
```python
DAGShortestPath(mode="longest").solve(task_dag, start_task)
```

**Maze / game map:**
```python
GridPathfinder(grid, obstacle="#", diagonal=True).solve(start, goal)
```

## Tests / Demo

```bash
pytest _stress_test.py -v          # 204 tests
python shortest_paths_template.py  # self-contained 11-section demo
```
