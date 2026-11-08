"""The prediction table: a bus-observed or predicted speed for every study-area bus road.

One row per edge and 15-minute bin. Where a bus moved on the edge in that bin
the row keeps the observed speed. Everywhere else the model, trained on every
observed row, predicts it. Dwell-only rows have no moving speed, so they are
predicted too.
"""

import pandas as pd

from features.table import (
    TARGET,
    build_feature_table,
    build_prediction_features,
    feature_columns,
    targets,
)
from model.speed_model import make_model
from speeds.edge_speeds import BIN_MINUTES

# The same box as matching.trips.DEFAULT_BBOX on feat/map-matching (not merged
# yet). Change both together.
STUDY_BBOX = {"xmin": -3.215, "ymin": 55.942, "xmax": -3.155, "ymax": 55.958}
PREDICTION_COLUMNS = ["edge_id", "street_name", "bin_start_utc", "speed_kmh", "kind",
                      "n_traversals"]


def inside(graph, edge_id, bbox):
    """True when the edge's midpoint lies in the bbox."""
    edge = graph.edge(int(edge_id))
    u, v = graph.node(edge.u), graph.node(edge.v)
    lat, lon = (u.lat + v.lat) / 2, (u.lon + v.lon) / 2
    return bbox["xmin"] <= lon <= bbox["xmax"] and bbox["ymin"] <= lat <= bbox["ymax"]


def study_edges(graph, bbox, mode="bus"):
    """Return the ids of edges in the bbox that permit the mode, sorted as strings."""
    return sorted(
        str(eid) for eid in graph.edges
        if graph.edge(eid).permits(mode) and inside(graph, eid, bbox)
    )


def night_bins(speeds):
    """Return every bin from the first observed bin to the last."""
    times = pd.to_datetime(speeds["bin_start_utc"], utc=True)
    return list(pd.date_range(times.min(), times.max(), freq=f"{BIN_MINUTES}min"))


def single_source(speeds):
    """Return the one source in the speeds. More than one would mix up the lag."""
    sources = speeds["source"].unique()
    if len(sources) != 1:
        raise ValueError(f"expected one source in the speeds, got {sorted(sources)}")
    return sources[0]


def prediction_table(speeds, graph, betweenness, flags=None, aadf=None, concerns=None,
                     bbox=STUDY_BBOX):
    """Return the prediction table for every study-area bus road and bin of the night."""
    source = single_source(speeds)
    speeds = speeds.assign(
        edge_id=speeds["edge_id"].astype(str),
        bin_start_utc=pd.to_datetime(speeds["bin_start_utc"], utc=True),
    )
    observed = targets(speeds)
    edges = sorted(
        set(study_edges(graph, bbox))
        | {e for e in observed["edge_id"].unique() if inside(graph, e, bbox)}
    )
    keys = [(edge, when) for edge in edges for when in night_bins(speeds)]

    train = build_feature_table(speeds, graph, betweenness, flags, aadf, concerns)
    columns = feature_columns(train)
    model = make_model().fit(train[columns], train[TARGET])
    features = build_prediction_features(keys, speeds, graph, betweenness, source, flags, aadf,
                                         concerns)
    predicted = model.predict(features.reindex(columns=columns)).clip(min=0.0)

    seen = observed.set_index(["edge_id", "bin_start_utc"])[[TARGET, "n_traversals"]]
    table = pd.DataFrame({
        "edge_id": features["edge_id"],
        "street_name": features["street_name"],
        "bin_start_utc": features["bin_start_utc"],
        "speed_kmh": predicted,
        "kind": "predicted",
        "n_traversals": 0,
    })
    index = pd.MultiIndex.from_frame(table[["edge_id", "bin_start_utc"]])
    hit = index.isin(seen.index)
    table.loc[hit, "speed_kmh"] = seen.loc[index[hit], TARGET].to_numpy()
    table.loc[hit, "n_traversals"] = seen.loc[index[hit], "n_traversals"].to_numpy()
    table.loc[hit, "kind"] = "observed"
    return table.sort_values(["edge_id", "bin_start_utc"], ignore_index=True)[PREDICTION_COLUMNS]
