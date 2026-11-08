"""Per-edge features from the road graph, plus Engine's optional per-edge files.

Restriction flags come from src/rules/bind_rules.py and DfT AADF from
src/graph/count_points.py. Either file can be missing, and then it is reported
as missing rather than guessed.
"""

import pandas as pd

# Busiest road type first. Covers every class the graph can hold.
ROAD_CLASS_RANK = {
    "motorway": 1,
    "trunk": 2,
    "primary": 3,
    "secondary": 4,
    "tertiary": 5,
    "unclassified": 6,
    "residential": 7,
    "living_street": 8,
    "service": 9,
    "cycleway": 10,
    "footway": 11,
    "pedestrian": 12,
}
FLAG_COLUMNS = ["no_motor_vehicles", "one_way", "pedestrian_only"]


def road_class_rank(highway_class):
    """Return the road class as a number, 1 for motorway. Unknown classes raise."""
    return ROAD_CLASS_RANK[highway_class]


def edge_features(graph, edge_id):
    """Return the graph's features for one edge."""
    edge = graph.edge(int(edge_id))
    return {
        "road_class_rank": road_class_rank(edge.highway_class),
        "lanes": edge.lanes,
        "length_m": edge.length_m,
        "speed_limit_kph": edge.speed_limit_kph,
    }


def read_flags(path):
    """Return the restriction flags table, or None if it has not been made."""
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"edge_id": str})[["edge_id", *FLAG_COLUMNS]]


def read_aadf(path):
    """Return AADF per edge, averaging count points on the same edge, or None."""
    if not path.exists():
        return None
    rows = pd.read_csv(path, dtype={"edge_id": str})
    return rows.groupby("edge_id", as_index=False)["aadf"].mean()


def read_concerns(path):
    """Return concern intensity per edge from Engine's concern field, or None if not made.

    The field has one column per consultation theme. Intensity is their sum.
    """
    if not path.exists():
        return None
    field = pd.read_csv(path, dtype={"edge_id": str})
    themes = [c for c in field.columns if c not in ("edge_id", "basis")]
    field["concern_intensity"] = field[themes].sum(axis=1)
    return field[["edge_id", "concern_intensity"]]
