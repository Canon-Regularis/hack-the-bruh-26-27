# Urban Street Network

The city is a graph, and this is that graph. Every other module in the urban
domain — traffic assignment, signal control, travel demand, accessibility,
emissions — binds to the node and edge identifiers minted here. It is not a
generic geospatial-graph toolkit (that is `core/graph/geospatial_graphs`, which
knows about latitudes and KD-trees but nothing about roads). This is a **road**
network: edges carry lanes, capacity, free-flow time, a speed limit, a street
name, a highway class, a gradient and permitted modes; nodes know whether they
are signalised; turns can be banned or made compulsory; and one-way streets are
first-class rather than an afterthought.

On top of that graph sits what a street-network study actually needs: routing
(Dijkstra, A\*, bidirectional, ALT landmarks, contraction hierarchies, Yen and
Eppstein k-shortest paths, FIFO time-dependent search), isochrones cut at the
exact fractional offset along a partly traversed edge, centrality (Brandes node
and edge betweenness, demand-weighted flow betweenness, closeness, straightness,
Stephenson–Zelen information), street-network science (circuity, orientation
entropy and Boeing's orientation order φ, planar-face block detection, space
syntax) and a Newson–Krumm HMM map matcher solved by Viterbi. Delaunay, the
contraction hierarchy, the Brandes recursion, the planar-face traversal and
Viterbi are all built by hand, and every stochastic routine takes a `seed`
driving a private `random.Random`, so output is byte-identical and
`PYTHONHASHSEED`-independent.

```python
from street_network_template import (grid_network, shortest_path, isochrone,
                                     betweenness_centrality, network_summary)

city  = grid_network(9, 9, 120.0, speed_limit_kph=32.0, name="downtown")
route = shortest_path(city, 0, 80)                      # corner to corner
print(route.length_m, route.num_edges, route.travel_time_s)   # 1920.0 16 216.0

iso = isochrone(city, 40, 60.0, weight="free_flow_time_s", alpha_m=200.0)
print(iso.num_nodes, len(iso.partial_edges), round(iso.area_m2))  # 41 60 557511

bc = betweenness_centrality(city, weight="length_m", normalised=True)
print(max(bc, key=lambda n: bc[n]))                     # 40 — the centre

s = network_summary(city, circuity_pairs=50)
print(s["num_blocks"], round(s["orientation_order"], 3))      # 64 1.0
```

---

## Pick

| Goal | Use |
| --- | --- |
| Build a road graph by hand or from a feed | `RoadGraph.add_node` / `add_edge`, `HIGHWAY_DEFAULTS` |
| Get a test network in one line | `grid_network` / `radial_network` / `random_planar_network` |
| Tidy an imported network | `clean_network` (loops, parallels, degree-2 chains, components) |
| One route, once | `shortest_path` / `astar` / `bidirectional_dijkstra` |
| Thousands of routes on a fixed network | `ContractionHierarchy`, or `ALTIndex` + `alt_shortest_path` |
| Everything reachable from a source | `dijkstra` (a whole `SearchTree`), `metric_closure` |
| Routes that obey banned turns and turn delay | `turn_aware_shortest_path` + `TurnCostModel` |
| Alternative routes | `yen_k_shortest_paths` (loopless) / `eppstein_k_shortest_paths` |
| Routing through a rush hour | `TravelTimeProfile` + `time_dependent_shortest_path` |
| A 15-minute-city catchment, with a polygon | `isochrone` (or `reachable_subgraph` for the graph) |
| Which streets carry the through traffic | `edge_betweenness_centrality`, `demand_weighted_betweenness` |
| Which junctions are central | `betweenness_centrality` / `closeness_centrality` / `information_centrality` |
| Gridiron or medieval tangle? | `orientation_order` / `street_orientation_entropy` / `circuity` |
| Block sizes and walkability | `detect_blocks` / `block_statistics` / `intersection_density` |
| Configurational analysis (space syntax) | `generate_strokes` + `space_syntax` |
| Snap a GPS trace onto the roads | `EdgeIndex` + `map_match` (`snap_point` for one fix) |
| The whole metric battery in one dict | `network_summary` |
| Something a browser can draw | `to_geojson` / `build_street_network_payload` / `build_a2ui_payload` |

---

## Install

Nothing to install. `street_network_template.py` is a single self-contained file
with **zero runtime dependencies** — copy it next to your code and import it.
The only entry in `requirements.txt` is `pytest>=7.0`, for the suite.
`core/graph/geospatial_graphs` supplies the great-circle formula when it happens
to be importable, so mixed pipelines agree to the last bit; the import is
guarded and the local fallback is the same haversine.

```bash
python street_network_template.py     # runs the demo, prints "demo OK"
pytest _stress_test.py -q             # 212 oracle-based tests
```

---

## What's inside

### 0-1 — Validation, geometry and geodesy

Every entry point coerces its arguments and raises `StreetNetworkError` (a
`ValueError`). Then `haversine_m`, `euclidean_m`, `bearing_deg`,
`bearing_difference_deg`, `polyline_length`, `point_segment_distance`,
`project_point_on_polyline`, `douglas_peucker`, `convex_hull` (monotone chain),
`polygon_area`, `polygon_centroid`, `point_in_polygon`, a Bowyer–Watson
`delaunay_triangulation` with an exact-sign InCircle predicate, and the
`alpha_shape` concave hull stitched from the triangles it keeps.

### 2-3 — The road graph and its cleaning

`RoadNode`, `RoadEdge`, `TurnRestriction`, `RoadGraph`; `MODES` and
`HIGHWAY_DEFAULTS` give speed, lanes, per-lane capacity and modes for twelve OSM
highway classes, so `add_edge(u, v, highway_class="primary")` is fully
attributed. Cleaning: `remove_self_loops`, `consolidate_parallel_edges` (keep
shortest or fastest), `simplify_degree2_chains` — which conserves cumulative
length and the full polyline and refuses to merge across a change of name,
class, lanes or direction — `largest_component_subgraph`, and the
`clean_network` pipeline over all four.

### 4 — Turns

`TurnCostModel` classifies a signed deflection as straight / left / right /
u-turn and prices it (`for_drive_side`, `free()`). `edge_expanded_graph` builds
the edge-based dual `ExpandedGraph`, where road edges are nodes and turns arcs —
the only representation that makes a banned turn *exact* — and
`turn_aware_shortest_path` routes over it, reusing a prebuilt expansion.

### 5-7 — Shortest paths, ALT and contraction hierarchies

`dijkstra` returns a whole `SearchTree` and takes `max_cost`, `banned_edges`,
`banned_nodes`, a `mode` filter and `reverse=True`; on top sit `shortest_path`,
`shortest_path_length`, `astar`, `bidirectional_dijkstra` and `metric_closure`,
all returning a `Path` (nodes, edges, cost, `length_m`, `travel_time_s`,
geometry). `ALTIndex` picks landmarks by farthest-point sampling and maximises
over both triangle-inequality forms, making `alt_shortest_path` goal-directed
yet exact; `ContractionHierarchy` does lazy edge-difference ordering, hop- and
cost-limited witness search, shortcut insertion and recursive unpacking.

### 8-9 — Alternatives and time-dependent routing

`yen_k_shortest_paths` is loopless and non-decreasing;
`eppstein_k_shortest_paths` turns every edge into a non-negative *sidetrack*
cost so best-first search enumerates walks in order of total cost.
`TravelTimeProfile` is piecewise-linear travel time versus departure time,
optionally periodic, with an explicit `is_fifo` check;
`time_dependent_dijkstra` and `time_dependent_shortest_path` are
earliest-arrival label-setting searches that **refuse** a non-FIFO profile.

### 10 — Isochrones

`isochrone` returns an `Isochrone`: reachable nodes with costs, whole edges,
partial edges cut at the exact fractional offset the remaining budget buys, the
frontier points and an `alpha_shape` polygon. `reached_length_m` pro-rates the
partials; `reachable_subgraph` gives the induced graph instead of the polygon.

### 11 — Centrality

`betweenness_centrality` and `edge_betweenness_centrality` are Brandes with the
NetworkX scaling conventions (`normalised`, `directed`, a `sources` subset);
`demand_weighted_betweenness` reuses the accumulation with an OD matrix as pair
weights — an all-or-nothing assignment split evenly across equal-cost paths.
Then `closeness_centrality` (Wasserman–Faust corrected, or harmonic),
`straightness_centrality` and Stephenson–Zelen `information_centrality`.

### 12-13 — Street-network metrics and blocks

`circuity`, `detour_index`, `intersection_density`, `node_degree_distribution`,
`edge_length_statistics`, `orientation_histogram`, `street_orientation_entropy`,
Boeing's `orientation_order` φ, and `network_summary` for the whole battery.
`detect_blocks` enumerates the bounded faces of the planar embedding by the
left-hand rule on the bearing-sorted rotation system, returning `Block` objects
(ring, signed area, perimeter, centroid, sides, Polsby–Popper `compactness`);
`block_statistics` and `average_block_size` summarise them, and a `k × k` grid
gives exactly `(k-1)²`.

### 14 — Space syntax

`generate_strokes` builds `Stroke` "natural roads" by pairing the straightest
through movements at each junction; `segment_dual_graph` builds the dual with
step depth or Hillier–Iida angular cost. `space_syntax` returns a
`SpaceSyntaxResult` — connectivity, total and mean depth, relative asymmetry,
RRA, integration, choice, control value, intelligibility r² — per segment or
per stroke, optionally radius-limited.

### 15 — Snapping, splitting and map matching

`EdgeIndex` is a uniform grid over the edge geometries (`candidates`, `within`,
a radius-doubling `nearest`); `snap_point` returns a `SnapResult`. `split_edge`
cuts an edge *and its reverse twin* in place, apportioning length and free-flow
time so the halves sum exactly to the original. `map_match` is Newson–Krumm —
Gaussian emissions on projection distance, exponential transitions, an optional
heading prior — solved exactly by Viterbi into `Candidate`s and a
`MapMatchResult`.

### 16-17 — Components and generators

`weakly_connected_components` and iterative Tarjan
`strongly_connected_components`, largest-first with deterministic tie-breaks.
`grid_network` (optionally a Manhattan alternating one-way system, with jitter
and signals), `radial_network` (hub, spokes, rings) and `random_planar_network`
(seeded Poisson points, Delaunay, optional thinning) — all analytically
predictable, which is what the suite routes over.

### 18 — Payloads

`to_geojson` emits a `FeatureCollection` of segments and junctions,
un-projecting to lon/lat when the graph is geographic.
`build_street_network_payload` is the frontend summary (topology, attribution,
bbox, metrics); `build_a2ui_payload` the map view: polylines, shaded markers,
route and isochrone overlays, stat cards. Non-finite floats become `None`, so
both are `json.dumps`-safe.

---

## Recipes

**Ban a turn and watch the route change.** A restriction is only *exact* in the
edge-expanded dual — plain Dijkstra will happily drive through it:

```python
from street_network_template import (RoadGraph, TurnCostModel, edge_expanded_graph,
                                     turn_aware_shortest_path, shortest_path)

g = RoadGraph("town centre", drive_side="left")
for nid, x, y in [("W", -100.0, 0.0), ("C", 0.0, 0.0), ("E", 100.0, 0.0),
                  ("N", 0.0, 100.0), ("D", -160.0, 160.0)]:
    g.add_node(nid, x, y, signalised=(nid == "C"))
wc, _ = g.add_edge("W", "C", 100.0, oneway=False, street_name="High Street")
cn, _ = g.add_edge("C", "N", 100.0, oneway=False, street_name="North Road")
g.add_edge("C", "E", 100.0, oneway=False, street_name="High Street")
g.add_edge("W", "D", 160.0, oneway=False, street_name="Bypass")
g.add_edge("D", "N", 160.0, oneway=False, street_name="Bypass")

d = g.turn_deflection_deg(wc, cn)
tcm = TurnCostModel.for_drive_side("left")
print(tcm.classify(d), tcm.penalty(d, signalised=True))   # left 10.0

g.add_turn_restriction(wc, "C", cn, "no_turn", "no left into North Road")
exp = edge_expanded_graph(g, weight="length_m", turn_cost=TurnCostModel.free())
r = turn_aware_shortest_path(g, "W", "N", weight="length_m", expanded=exp)
print(r.nodes, r.cost, exp.banned_turns)   # ['W','D','N'] 320.0 11
print(shortest_path(g, "W", "N").cost)     # 200.0 — the ban is invisible here
```

**Preprocess once, answer thousands of queries.** Both the hierarchy and the
landmark index return costs *bit-identical* to plain Dijkstra:

```python
import random
from street_network_template import (random_planar_network, largest_component_subgraph,
                                     ContractionHierarchy, ALTIndex, alt_shortest_path,
                                     shortest_path)

town = largest_component_subgraph(
    random_planar_network(200, 3000.0, 3000.0, seed=7, prune_fraction=0.15), True)
ch  = ContractionHierarchy(town, weight="length_m")
alt = ALTIndex(town, 8, weight="length_m", seed=3)
print(ch.summary())      # nodes / arcs / shortcuts / upward / downward arcs

rng, agree, tried = random.Random(0), 0, 0
for _ in range(200):
    a, b = (town.nodes[rng.randrange(town.num_nodes)] for _ in range(2))
    base = None if a == b else shortest_path(town, a, b)
    if base is None:
        continue
    tried += 1
    agree += int(ch.distance(a, b) == base.cost
                 == alt_shortest_path(town, alt, a, b).cost)
print(agree, tried)      # 199 199 — exact, not "within tolerance"
```

**Map-match a noisy GPS trace.** Build the spatial index once and hand it to
every call; the HMM recovers the true edge sequence through 3 m of noise:

```python
import random
from street_network_template import grid_network, EdgeIndex, shortest_path, map_match

city  = grid_network(6, 6, 100.0, speed_limit_kph=32.0, name="mm-grid")
idx   = EdgeIndex(city)
truth = shortest_path(city, 0, 35)

noise, trace = random.Random(2024), []
for (ax, ay), (bx, by) in zip(truth.geometry, truth.geometry[1:]):
    for t in (0.25, 0.5, 0.75):                     # three fixes per segment
        trace.append((ax + t * (bx - ax) + noise.gauss(0.0, 3.0),
                      ay + t * (by - ay) + noise.gauss(0.0, 3.0)))

m = map_match(city, trace, sigma_z=4.07, beta=10.0, search_radius_m=40.0, index=idx)
print(m.edges == truth.edges, m.summary()["meanErrorM"])   # True 2.64…
```

**Measure the shape of a city.** Orientation entropy, order φ, circuity, block
size and space syntax separate a gridiron from a radial town:

```python
from street_network_template import (grid_network, radial_network, circuity,
                                     street_orientation_entropy, orientation_order,
                                     average_block_size, space_syntax)

for g in (grid_network(9, 9, 120.0, name="gridiron"),
          radial_network(5, 12, 120.0, name="radial town")):
    print("%-12s H=%.3f  phi=%.3f  circuity=%.3f  block=%.0f m2"
          % (g.name, street_orientation_entropy(g), orientation_order(g),
             circuity(g, num_pairs=300, seed=1), average_block_size(g)))
# gridiron     H=1.386  phi=1.000  circuity=1.255  block=14400 m2   (H is ln 4 exactly)
# radial town  H=3.348  phi=0.203  circuity=1.094  block=18000 m2

ss = space_syntax(grid_network(9, 9, 120.0), unit="segment")
print(round(ss.intelligibility, 3), round(ss.most_integrated(3)[0][1], 3))  # 0.609 1.392
```

---

## Design notes

- **Planar metres are the working coordinate system.** Nodes store `(x, y)` in
  metres, `x` east and `y` north; `lat` / `lon` are optional and projected once
  to a local equirectangular plane anchored on the *first* geographic node, so
  euclidean distance agrees with the great circle. Bearings are clockwise from
  north in `[0, 360)`, and `bearing_difference_deg` returns a signed deflection
  in `[-180, 180)` — **negative is a left turn**, which is why `for_drive_side`
  only swaps which of `left_s` / `right_s` is the cheap one.

- **Every edge is directed, and `oneway=True` is the default.** A two-way
  street is *two* edges linked by `reverse_eid`, created with `oneway=False`.
  Forgetting this is the commonest way to build a graph in which half the
  journeys are unreachable; `strongly_connected_components` is how you notice.

- **A cost specification is a field name, `"hops"` or a callable**, and must be
  non-negative — `dijkstra` raises rather than return a quietly wrong answer.
  `Path.cost` is then re-accumulated left-to-right over the path's edges,
  exactly as Dijkstra accumulates labels, so Dijkstra, bidirectional Dijkstra,
  A\*, ALT and the CH return the *same float* whenever they agree on the path —
  `==`, not `abs(a - b) < tol`. `length_m` and `travel_time_s` use `math.fsum`,
  being reported quantities rather than search labels.

- **The default A\* heuristic knows its own units.** Scale 1 for a distance
  weight, `1 / fastest free-flow speed` for a time weight, `0` for a *callable*
  weight — units unknown, so A\* degrades gracefully to Dijkstra. Pass your own
  admissible `heuristic=` for goal direction on a custom cost.

- **Turn restrictions are honoured only by the turn-aware routines.**
  `dijkstra`, `shortest_path`, `astar`, `bidirectional_dijkstra`, `ALTIndex` and
  `ContractionHierarchy` work on the node graph and cannot see a banned turn —
  that is the reason `edge_expanded_graph` exists. U-turns are banned by default
  (`allow_u_turn=True` permits them), and `TurnCostModel` penalties are in the
  **units of the routing weight**: the defaults are seconds, so pair them with
  `weight="free_flow_time_s"`, or a right turn silently costs seven metres.

- **Entropy and order point in opposite directions.**
  `street_orientation_entropy` is *minimal* for a grid (exactly `ln 4`) and
  maximal for a chaotic network, while Boeing's `orientation_order` φ is
  *maximal* (exactly `1.0`) for a grid. Reach for the wrong one and the
  conclusion inverts; `network_summary` reports both.

- **`circuity` separates malformed input from an undefined answer.** An empty
  graph raises; a single-node graph is a legitimate if trivial network with no
  origin–destination pair, so its circuity is `NaN` — the convention already
  used when no sampled pair is connected. The payload builders map every
  non-finite float to `None`.

- **Indices cache geometry, and `split_edge` replaces ids.** Rebuild an
  `EdgeIndex` after any structural change: `split_edge` removes and re-adds both
  halves *and* both twins, so the original edge id is gone, and it mints a
  string node id `"u|v@eid"` unless you supply one. `detect_blocks` likewise
  assumes a planar embedding — a flyover or an un-noded import yields faces that
  are not city blocks, so run `clean_network` first and check Euler's
  `F = E - V + C`.

- **Two deliberate departures, both harmless.** The CH witness search is hop-
  and cost-limited, and a truncated witness search can only *fail to find* a
  witness, inserting a redundant shortcut — a larger hierarchy, never a wrong
  one. And `map_match` adds a `-w(1 - cos Δ)` heading prior to Newson–Krumm:
  free when the trace runs the right way, decisive when a fix lands on a
  junction; set `heading_weight=0.0` for the unmodified model.

- **Determinism is structural.** Landmark selection, circuity sampling and the
  generators' jitter and thinning all take a `seed` and drive a private
  `random.Random`; every heap key ends in a monotone counter so equal-cost ties
  break identically; components and blocks sort on explicit keys.

---

## Testing

`_stress_test.py` is an oracle-based suite — **212 tests** across all nineteen
sections, green under `PYTHONHASHSEED` 0, 1 and 1010. The rule is *no
hand-typed magic number where a closed form, an exact identity or a metamorphic
invariant exists*, in four tiers:

1. **Closed forms.** The `k × k` unit grid is an analytic oracle: corner-to-corner
   distance is exactly `2(k-1)`, the Manhattan distance holds for *every* pair,
   the bounded faces number exactly `(k-1)²` with area `spacing²`, the stroke
   count is exactly `2k`, the orientation entropy exactly `ln 4`, φ exactly
   `1.0`, the corner-pair circuity exactly `√2`. Brandes on a star `K₁,ₙ₋₁`
   gives `(n-1)(n-2)/2` at the centre and `0` at every leaf; on a path `Pₙ` node
   `i` scores `i(n-1-i)` and the middle edge `4·3`; information centrality on a
   unit `K₃` is exactly `2.25`.
2. **Exact identities.** `split_edge` apportions length and free-flow time so
   the halves sum *exactly* to the original; `simplify_degree2_chains` conserves
   total length exactly; `detect_blocks` satisfies Euler's `F = E - V + C` on
   every generator; an all-or-nothing OD assignment puts exactly the demand on
   each edge of a unique shortest path and splits ties evenly; reverse Dijkstra
   equals forward Dijkstra on the transpose; `to_geojson` and the payload
   builders round-trip through `json`.
3. **Metamorphic and invariant.** The flagship: Dijkstra, A\*, bidirectional
   Dijkstra, ALT and the contraction hierarchy return **bit-identical** costs
   over 1000+ seeded pairs, on 17-significant-digit lengths and across a
   nine-decade length range. Yen's k-th cost is non-decreasing and every path
   loopless and distinct; the ALT bound never exceeds the true distance; the
   metric closure obeys the triangle inequality to a scale-relative tolerance;
   sub-tolerance weight noise leaves Brandes betweenness unchanged; a banned
   turn can only lengthen a route; isochrones are monotone in the budget.
4. **Differential**, behind `pytest.importorskip` so the module stays
   zero-dependency: `networkx` pins shortest-path distances, single-pair
   routing, node and edge betweenness (raw and normalised), closeness and
   harmonic closeness, components and the k-shortest-path set to machine
   precision.

```bash
pytest _stress_test.py -q
PYTHONHASHSEED=0    pytest _stress_test.py -q
PYTHONHASHSEED=1010 pytest _stress_test.py -q
```

**What the suite caught.** One contract inconsistency in the template, plus
three defects in the tests themselves:

- **`circuity` raised on any graph with fewer than two nodes**, even though it
  already had a `NaN` convention for pairs that are not connected. An *empty*
  graph is malformed input and still raises; a *single-node* graph is a
  legitimate if trivial network whose circuity is simply undefined, and now
  returns `NaN`, so `network_summary` works on degenerate graphs.
- The **Delaunay Euler check used the strict-hull vertex count** where the
  formula `2n - 2 - h` needs the count of points *on* the hull boundary — a
  3 × 3 lattice has 8, not 4.
- A **banned-node test banned both of node 0's only neighbours** and then
  asserted that a path existed.
- A **time-dependent routing test assumed the traveller entered the rush-hour
  edge at exactly `t = 3600`**, when it actually arrives there at `t = 3636`.

---

## References

- Dijkstra (1959) — *A note on two problems in connexion with graphs*; Hart,
  Nilsson & Raphael (1968) — A\*; Tarjan (1972) — strongly connected components.
- Goldberg & Harrelson (2005) — *A\* search meets graph theory* (ALT);
  Geisberger, Sanders, Schultes & Delling (2008) — *Contraction hierarchies*;
  Bast et al. (2016) — *Route planning in transportation networks*.
- Yen (1971) — k shortest loopless paths; Eppstein (1998) — *Finding the k
  shortest paths*; Caldwell (1961), Winter (2002) — turn penalties and the
  edge-expanded representation; Dreyfus (1969), Delling & Wagner (2009) — FIFO
  time-dependent shortest paths.
- Freeman (1977) — betweenness; Brandes (2001, 2008) — the fast algorithm and
  its variants; Stephenson & Zelen (1989) — information centrality; Porta,
  Crucitti & Latora (2006) — the primal approach and straightness.
- Boeing (2017) — *OSMnx*; Boeing (2019) — *Urban spatial order: street network
  orientation, configuration, and entropy*; Barthélemy (2011) — *Spatial
  networks* (circuity, detour index).
- Hillier & Hanson (1984) — *The Social Logic of Space*; Hillier & Iida (2005) —
  angular segment analysis; Thomson (2003), Jiang & Claramunt (2004) — strokes
  and the dual street graph.
- Newson & Krumm (2009) — *Hidden Markov map matching through noise and
  sparseness*; Viterbi (1967). Bowyer (1981), Watson (1981) — incremental
  Delaunay; Edelsbrunner, Kirkpatrick & Seidel (1983) — alpha shapes; Douglas &
  Peucker (1973); Andrew (1979) — the monotone-chain hull.

---

*Pure standard library: `heapq`, `itertools`, `math`, `random`, `dataclasses`,
`typing`. `street_network_template.py` — `__version__ = "1.0.0"`, 88 public
names in `__all__`, 212 oracle-based tests.*
