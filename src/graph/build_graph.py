"""Turn cached OSM data into the road graph everything else uses (ALG-MVP-1, #18).

Run once from the repo root to freeze the graph: ``python -m src.graph.build_graph``.
Edge ids are ints in the graph and ``str(eid)`` in every file we write.
"""

import json
import pickle
from collections import Counter
from operator import itemgetter
from pathlib import Path
from typing import NamedTuple

from engine.orchid import street_network as sn

ROOT = Path(__file__).resolve().parents[2]
OSM_PATH = ROOT / "data" / "raw" / "osm.json"
GRAPH_DIR = ROOT / "data" / "graph"

KEPT_EDGE_TAGS = ("access", "motor_vehicle", "vehicle", "bus", "psv", "foot", "bicycle", "oneway", "maxspeed")
MPH_TO_KPH = 1.609344
KEPT_NODE_TAGS = ("highway", "crossing", "crossing_ref", "crossing:markings")
SPLIT_NODE_HIGHWAYS = frozenset({"traffic_signals", "crossing"})
BUS_STOP_PUBLIC_TRANSPORT = frozenset({"platform", "stop_position"})

HIGHWAY_ALIASES = {"steps": "footway", "path": "footway"}
ONEWAY_VALUES = {"yes": 1, "true": 1, "1": 1, "-1": -1, "reverse": -1}
IMPLIED_ONEWAY_JUNCTIONS = frozenset({"roundabout", "circular"})
IMPLIED_ONEWAY_HIGHWAYS = frozenset({"motorway", "motorway_link"})


class Osm(NamedTuple):
    coords: dict
    node_tags: dict
    ways: list


def highway_class(raw: str) -> str | None:
    """Return the HIGHWAY_DEFAULTS road type for an OSM highway value, or None to skip it.

    e.g. "primary_link" -> "primary", "steps" -> "footway".
    """
    base = raw.removesuffix("_link")
    return base if base in sn.HIGHWAY_DEFAULTS else HIGHWAY_ALIASES.get(raw)


def oneway_direction(tags: dict) -> int:
    """Return 1 if the way is one-way, -1 if one-way backwards, 0 if two-way."""
    value = tags.get("oneway")
    if value is None:
        return int(tags.get("junction") in IMPLIED_ONEWAY_JUNCTIONS
                   or tags.get("highway") in IMPLIED_ONEWAY_HIGHWAYS)
    return ONEWAY_VALUES.get(value.strip().lower(), 0)


def maxspeed_kph(tags: dict) -> float | None:
    """Return the way's OSM speed limit in km/h, or None to use the road-class default.

    "20 mph" and "20mph" are miles per hour; a bare number is km/h, as OSM defines it.
    Values like "none", "signals" or "walk" give None.
    """
    value = tags.get("maxspeed", "").strip().lower()
    mph = "mph" in value
    try:
        speed = float(value.split("mph")[0] if mph else value)
    except ValueError:
        return None
    if speed <= 0:
        return None
    return speed * MPH_TO_KPH if mph else speed


def read_osm(osm_path: Path) -> Osm:
    """Read a cached Overpass JSON file once: node coordinates, tagged nodes, and road ways by id.

    Pedestrian areas (area=yes) are skipped, so a square isn't routed round its edge.
    """
    coords, node_tags, ways = {}, {}, []
    for el in json.loads(osm_path.read_text(encoding="utf-8"))["elements"]:
        tags = el.get("tags", {})
        if el["type"] == "node":
            coords[el["id"]] = (el["lat"], el["lon"])
            if tags:
                node_tags[el["id"]] = tags
        elif (el["type"] == "way" and tags.get("area") != "yes"
              and (klass := highway_class(tags.get("highway", "")))):
            ways.append((el["id"], el["nodes"], tags, klass))
    ways.sort(key=itemgetter(0))
    return Osm(coords, node_tags, ways)


def build_graph(osm: Osm) -> sn.RoadGraph:
    """Build the road graph from the OSM data.

    Ways are split into edges at junctions, traffic signals and crossings. Speed limits come
    from OSM maxspeed where it can be read, else from the road-class default.
    Nodes are added in OSM id order so the graph comes out the same every time.
    Raw OSM tags are kept on each node and edge under ``tags["osm"]``.
    """
    coords, node_tags, ways = osm
    counts = Counter(ref for _, refs, _, _ in ways for ref in refs)
    if missing := counts.keys() - coords.keys():
        raise ValueError(f"OSM cache is missing {len(missing)} way nodes, e.g. {min(missing)}; "
                         "fetch with Overpass recursion (>;) so every way's nodes are included")
    cuts = {ref for ref, n in counts.items() if n > 1}
    cuts.update(nid for nid, tags in node_tags.items()
                if nid in counts and tags.get("highway") in SPLIT_NODE_HIGHWAYS)
    cuts.update(end for _, refs, _, _ in ways for end in (refs[0], refs[-1]))

    graph = sn.RoadGraph()
    for nid in sorted(cuts):
        tags = node_tags.get(nid, {})
        lat, lon = coords[nid]
        graph.add_node(nid, lat=lat, lon=lon, signalised=tags.get("highway") == "traffic_signals",
                       osm={k: tags[k] for k in KEPT_NODE_TAGS if k in tags})

    for way_id, refs, tags, klass in ways:
        direction = oneway_direction(tags)
        limit = maxspeed_kph(tags)
        osm_tags = {"way_id": way_id, **{k: tags[k] for k in KEPT_EDGE_TAGS if k in tags}}
        start = 0
        for i in range(1, len(refs)):
            if refs[i] not in cuts:
                continue
            seg = refs[start:i + 1]
            if direction < 0:
                seg.reverse()
            u, v = seg[0], seg[-1]
            graph.add_edge(
                u, v, oneway=direction != 0, street_name=tags.get("name", ""), highway_class=klass,
                speed_limit_kph=limit,
                geometry=[graph.node(u).xy, *(graph._project(*coords[n]) for n in seg[1:-1]),
                          graph.node(v).xy],
                osm=osm_tags,
            )
            start = i
    return graph


def bus_stops(osm: Osm) -> list[list[float]]:
    """Return [lat, lon] for every bus stop, in OSM id order, for Signal's dwell check."""
    return [list(osm.coords[nid]) for nid, tags in sorted(osm.node_tags.items())
            if tags.get("highway") == "bus_stop"
            or (tags.get("public_transport") in BUS_STOP_PUBLIC_TRANSPORT and tags.get("bus") == "yes")]


def street_names(graph: sn.RoadGraph) -> dict[str, str]:
    """Return {str(edge id): street name} for every directed edge, "" when unnamed."""
    return {str(edge.eid): edge.street_name for edge in graph.iter_edges()}


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def load_graph(osm_path: Path = OSM_PATH, graph_dir: Path = GRAPH_DIR) -> sn.RoadGraph:
    """Return the frozen graph; build an sn.EdgeIndex from it where nearest-edge lookups are needed.

    Loads graph_dir/graph.pkl if it exists. Otherwise builds and cleans the graph, then
    writes street_names.json, bus_stops.json and finally graph.pkl, so edge ids never
    change and a saved graph always has its hand-off files.
    """
    frozen = graph_dir / "graph.pkl"
    if frozen.exists():
        with frozen.open("rb") as f:
            graph = pickle.load(f)
    else:
        osm = read_osm(osm_path)
        graph = sn.clean_network(build_graph(osm), simplify=False)
        graph_dir.mkdir(parents=True, exist_ok=True)
        _write_atomic(graph_dir / "street_names.json", json.dumps(street_names(graph)).encode())
        _write_atomic(graph_dir / "bus_stops.json", json.dumps(bus_stops(osm)).encode())
        _write_atomic(frozen, pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL))
    return graph


if __name__ == "__main__":
    g = load_graph()
    print(f"{g.num_nodes} nodes, {g.num_edges} edges frozen in {GRAPH_DIR}")
