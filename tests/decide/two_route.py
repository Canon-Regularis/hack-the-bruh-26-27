"""A small design-search setup shared by the #28 and #29 tests.

Two routes from 0 to 3: North Road via 1 (zebra at 1) and South Road via 2, plus a spur 3-4.
Candidates: close North Road (bit 0) and a bus lane halving South Road's capacity (bit 1).
"""

import json
from pathlib import Path

from engine.orchid import street_network as sn
from engine.orchid import traffic_assignment as ta
from src.decide import interventions as iv
from src.decide.network import link_maps, to_network
from src.decide.pedestrian_crossing_delay import find_crossings

ZEBRA = {"highway": "crossing", "crossing": "uncontrolled"}
NORMAL = {"id": "normal", "multiplier": 1.0, "closed_edge_ids": [], "probability": 0.8}
SOUTH_SHUT = {"id": "event-south-closed", "multiplier": 1.5, "closed_edge_ids": ["4", "5", "6", "7"], "probability": 0.2}
CANDIDATES = [{"id": "close-north", "kind": "close", "street": "North Road"},
              {"id": "bus-lane-south", "kind": "reallocate", "street": "South Road", "capacity_factor": 0.5}]


def build_graph() -> sn.RoadGraph:
    g = sn.RoadGraph()
    for nid, (lat, lon) in enumerate([(55.949, -3.200), (55.950, -3.199), (55.948, -3.199),
                                      (55.949, -3.198), (55.949, -3.197)]):
        g.add_node(nid, lat=lat, lon=lon, osm=ZEBRA if nid == 1 else {})
    for u, v, name in [(0, 1, "North Road"), (1, 3, "North Road"), (0, 2, "South Road"), (2, 3, "South Road"),
                       (3, 4, "Spur")]:
        g.add_edge(u, v, oneway=False, street_name=name, highway_class="tertiary")
    return g


def build_context(graph: sn.RoadGraph, folder: Path) -> dict:
    path = folder / "candidates.json"
    path.write_text(json.dumps(CANDIDATES))
    crossings = find_crossings(graph)
    net, _ = to_network(graph)
    return {"net": net, "link_to_edge": link_maps(net)[1], "hour": "08:00-09:00 weekday", "crossings": crossings,
            "od": ta.ODMatrix({(0, 3): 600.0, (3, 0): 300.0, (0, 4): 100.0}),
            "candidates": iv.load_candidates(path, graph, crossings)[0], "abstained_keep_open": [],
            "scenarios": [NORMAL, SOUTH_SHUT]}
