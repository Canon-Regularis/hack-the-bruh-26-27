# Geospatial Graphs

Pure-Python, zero-dependency toolkit for graphs embedded on the Earth:
great-circle / ellipsoidal distance, a KD-tree spatial index (so Euclidean
nearest *is* great-circle nearest), geohashing, proximity-graph construction
(k-NN / radius / Gabriel / relative-neighbourhood), A* / Dijkstra routing with an
admissible great-circle heuristic, isochrones, computational geometry (convex
hull, point-in-polygon, spherical area, Douglas–Peucker), DBSCAN with a metric
`eps`, Web-Mercator projection, and GeoJSON I/O. No geopy / shapely / pyproj /
scipy / numpy required.

## Pick

| Goal | Use |
|---|---|
| Distance between two coordinates | `haversine` / `vincenty` / `geodesic(method=…)` |
| Bearing / move along a heading | `bearing`, `destination`, `midpoint` |
| Nearest / k-nearest / radius search | `GeoIndex.nearest` / `.k_nearest` / `.within_radius` |
| Build a graph from points | `GeoGraph.knn_graph` / `.radius_graph` / `.gabriel_graph` |
| Shortest route between nodes | `GeoGraph.astar` / `.shortest_path` |
| "What's reachable within X km" | `GeoGraph.reachable_within` (isochrone) |
| Order a visit (TSP) | `GeoGraph.nearest_neighbor_tour` + `.two_opt` |
| Cluster points by density | `dbscan(points, eps_m, min_samples)` |
| Hull / area / inside-polygon | `convex_hull` / `spherical_polygon_area` / `point_in_polygon` |
| Simplify a track | `douglas_peucker` |
| Bucket / tile coordinates | `geohash_encode` / `geohash_neighbors` |
| Export for a map frontend | `to_geojson` / `build_geo_payload` |

## Install

```bash
# No runtime deps — copy the file.
pip install pytest  # tests only
```

## Quick start

```python
from geospatial_graphs_template import GeoPoint, GeoGraph, GeoIndex, haversine

cities = {
    "London": GeoPoint(51.5074, -0.1278),
    "Paris":  GeoPoint(48.8566,  2.3522),
    "Berlin": GeoPoint(52.5200, 13.4050),
    "Madrid": GeoPoint(40.4168, -3.7038),
}

haversine(cities["London"], cities["Paris"]) / 1000     # 343.6 km

# nearest-neighbour search
idx = GeoIndex(cities)
idx.nearest((50.0, 5.0))                # ('Paris', 229_000.0) -> (id, metres)
idx.within_radius((50.0, 5.0), 500_000) # [(id, dist), ...] within 500 km

# build a graph and route on it
g = GeoGraph.knn_graph(cities, k=2)
g.astar("Madrid", "Berlin")             # ['Madrid', 'Paris', 'Berlin']
g.shortest_path_length("Madrid", "Berlin") / 1000       # km
```

## API

### Coordinates & distance

```python
GeoPoint(lat, lon)        # validated: lat in [-90,90]; lon wrapped to [-180,180]
p.ecef_unit               # (x, y, z) on the unit sphere

haversine(a, b)           # great-circle metres (default; antimeridian-safe)
law_of_cosines(a, b)
equirectangular(a, b)     # fast approximation
vincenty(a, b)            # WGS84 ellipsoid, ~0.5 mm (antipodal -> haversine)
geodesic(a, b, method)    # "haversine" | "vincenty" | "cosines" | "equirectangular"

bearing(a, b)             # initial bearing, degrees (0=N, clockwise)
final_bearing(a, b)
destination(a, bearing_deg, distance_m)        # -> GeoPoint
midpoint(a, b)
cross_track_distance(p, path_start, path_end)  # signed perpendicular distance
along_track_distance(p, path_start, path_end)
```

Any function accepting a point also accepts a raw `(lat, lon)` tuple.

### Spatial index — `GeoIndex`

A KD-tree over unit-sphere ECEF coordinates: Euclidean nearest is *exactly*
great-circle nearest, so queries are correct everywhere including across the
antimeridian and near the poles.

```python
idx = GeoIndex(points)                  # dict {id: GeoPoint} or list[GeoPoint]
idx.nearest(query)                      # (id, distance_m) or None
idx.k_nearest(query, k)                 # [(id, distance_m), …] nearest first
idx.within_radius(query, radius_m)      # [(id, distance_m), …] sorted
idx.within_bbox(min_lat, min_lon, max_lat, max_lon)
```

### Graph — `GeoGraph`

```python
g = GeoGraph(directed=False, distance="haversine")
g.add_node(id, point)
g.add_edge(u, v, weight=None)           # default weight = geodesic distance

# builders (edge weight defaults to geodesic distance)
GeoGraph.knn_graph(points, k, mutual=False)
GeoGraph.radius_graph(points, radius_m)
GeoGraph.complete_graph(points)
GeoGraph.gabriel_graph(points)                  # proximity graph, O(n^3)
GeoGraph.relative_neighborhood_graph(points)    # proximity graph, O(n^3)

# routing
g.dijkstra(source)                      # (dist, prev)
g.shortest_path(source, target)         # node list or None
g.astar(source, target, heuristic=None) # A*; default heuristic is admissible
g.reachable_within(source, budget_m)    # isochrone {node: distance}

# travelling-salesman heuristics
g.nearest_neighbor_tour(start=None)
g.two_opt(tour)
g.tour_length(tour, closed=True)
```

> **A\* optimality.** The default heuristic is the great-circle distance to the
> target. Because every edge weight is a geodesic distance and great-circle
> distance obeys the triangle inequality, the heuristic is admissible **and**
> consistent — A\* returns the same optimum as Dijkstra while expanding far
> fewer nodes. For non-geographic edge weights, pass your own `heuristic=` (or
> use `shortest_path`, which is always Dijkstra).

### Geometry

```python
bounding_box(points)                    # (min_lat, min_lon, max_lat, max_lon)
centroid(points)                        # spherical centroid
convex_hull(points)                     # CCW hull on the (lon, lat) plane
point_in_polygon(point, polygon)        # ray casting (even-odd)
spherical_polygon_area(polygon)         # square metres (spherical excess)
polygon_perimeter(polygon, closed=True)
douglas_peucker(polyline, tolerance_m)  # great-circle simplification
```

### Clustering

```python
dbscan(points, eps_m, min_samples=4)    # {id: label};  label -1 == noise
```

### Projection / geohash / I/O

```python
to_web_mercator(point)  -> (x, y)       # EPSG:3857 metres
from_web_mercator(x, y) -> GeoPoint

geohash_encode(lat, lon, precision=12)
geohash_decode(geohash)                 # cell centre
geohash_bbox(geohash)                   # (min_lat, min_lon, max_lat, max_lon)
geohash_neighbors(geohash)              # {"n","ne","e",…} -> geohash

to_geojson(graph_or_points)             # GeoJSON FeatureCollection ([lon, lat])
build_geo_payload(graph)                # frontend summary
build_a2ui_payload(graph)               # map-visualisation dict
```

## Hackathon tips

**Load a CSV of coordinates and find clusters of activity:**
```python
pts = {row["id"]: GeoPoint(float(row["lat"]), float(row["lon"])) for row in rows}
labels = dbscan(pts, eps_m=250, min_samples=5)      # hotspots; -1 = noise
```

**Nearest depot / store / sensor:**
```python
GeoIndex(depots).k_nearest(user_location, 3)
```

**Plan a delivery route over a road-like proximity graph:**
```python
g = GeoGraph.knn_graph(stops, k=4)
tour = g.two_opt(g.nearest_neighbor_tour(start="depot"))
```

**Service-area / isochrone analysis:**
```python
g.reachable_within("warehouse", budget_m=30_000)    # everything within 30 km
```

**Choose the distance model:** `haversine` for general use (fast, ~0.3% error
from sphericity), `vincenty` when you need sub-metre accuracy, `equirectangular`
for very short hops in hot loops.

## Tests / Demo

```bash
pytest _stress_test.py -v          # differential vs brute force + closed-form
python geospatial_graphs_template.py  # 10-section demo over European cities
```

The suite validates the KD-tree index (nearest / k-nearest / radius) against
brute-force haversine over hundreds of random point sets, checks that A\* returns
the same optimum as Dijkstra on random k-NN graphs, that the convex hull
contains every input point, that geohashes round-trip to centimetre precision,
that `destination` inverts distance+bearing, and that DBSCAN recovers planted
clusters — plus Web-Mercator round-trips, spherical area against closed form,
antimeridian handling, and degenerate-input behaviour (empty / single-node /
duplicate points / k larger than the point set).
