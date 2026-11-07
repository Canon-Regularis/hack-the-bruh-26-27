"""Turn cached OSM data into the road graph everything else uses (ALG-MVP-1).

Run from the repo root: ``python -m src.graph.build_graph``.
"""

import json
import pickle
from collections import Counter
from operator import itemgetter
from pathlib import Path

from engine.orchid import street_network as sn

KEPT_EDGE_TAGS = ("access", "motor_vehicle", "vehicle", "bus", "psv", "foot", "bicycle", "oneway")
KEPT_NODE_TAGS = ("highway", "crossing")
SPLIT_NODE_HIGHWAYS = frozenset({"traffic_signals", "crossing"})

HIGHWAY_ALIASES = {"steps": "footway", "path": "footway"}
ONEWAY_VALUES = {"yes": 1, "true": 1, "1": 1, "-1": -1, "reverse": -1}
IMPLIED_ONEWAY_JUNCTIONS = frozenset({"roundabout", "circular"})
IMPLIED_ONEWAY_HIGHWAYS = frozenset({"motorway", "motorway_link"})


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


def build_graph(osm_path: Path) -> sn.RoadGraph:
    """Build the road graph from a cached Overpass JSON file that includes every way's nodes.

    Ways are split into edges at junctions, traffic signals and crossings.
    Nodes are added in OSM id order so the graph comes out the same every time.
    Raw OSM tags are kept on each node and edge under ``tags["osm"]``.
    """
    coords, node_tags, split_tagged, ways = {}, {}, set(), []
    for el in json.loads(osm_path.read_text(encoding="utf-8"))["elements"]:
        tags = el.get("tags", {})
        if el["type"] == "node":
            coords[el["id"]] = (el["lat"], el["lon"])
            if tags:
                node_tags[el["id"]] = tags
                if tags.get("highway") in SPLIT_NODE_HIGHWAYS:
                    split_tagged.add(el["id"])
        elif el["type"] == "way" and (klass := highway_class(tags.get("highway", ""))):
            ways.append((el["id"], el["nodes"], tags, klass))
    ways.sort(key=itemgetter(0))

    counts = Counter(ref for _, refs, _, _ in ways for ref in refs)
    if missing := counts.keys() - coords.keys():
        raise ValueError(f"{osm_path} is missing {len(missing)} way nodes, e.g. {min(missing)}; "
                         "fetch with Overpass recursion (>;) so every way's nodes are included")
    cuts = {ref for ref, n in counts.items() if n > 1 or ref in split_tagged}
    cuts.update(end for _, refs, _, _ in ways for end in (refs[0], refs[-1]))

    graph = sn.RoadGraph(osm_path.stem)
    for nid in sorted(cuts):
        tags = node_tags.get(nid, {})
        lat, lon = coords[nid]
        graph.add_node(nid, lat=lat, lon=lon, signalised=tags.get("highway") == "traffic_signals",
                       osm={k: tags[k] for k in KEPT_NODE_TAGS if k in tags})

    for way_id, refs, tags, klass in ways:
        direction = oneway_direction(tags)
        osm = {"way_id": way_id, **{k: tags[k] for k in KEPT_EDGE_TAGS if k in tags}}
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
                geometry=[graph.node(u).xy, *(graph._project(*coords[n]) for n in seg[1:-1]),
                          graph.node(v).xy],
                osm=osm,
            )
            start = i
    return graph


def load_graph(osm_path: Path, frozen_path: Path) -> tuple[sn.RoadGraph, sn.EdgeIndex]:
    """Return the cleaned graph and its edge lookup index.

    Loads the saved graph if there is one; otherwise builds, cleans and saves it,
    so edge ids never change between runs.
    """
    if frozen_path.exists():
        with frozen_path.open("rb") as f:
            graph = pickle.load(f)
    else:
        graph = sn.clean_network(build_graph(osm_path), simplify=False)
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = frozen_path.with_name(frozen_path.name + ".tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(frozen_path)
    return graph, sn.EdgeIndex(graph)
