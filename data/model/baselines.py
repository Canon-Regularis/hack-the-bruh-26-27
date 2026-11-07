"""The two simple guesses the model has to beat. Both learn from training rows only.

Baseline 1: mean speed by road class and hour.
Baseline 2: copy the nearest observed edge on another street.
"""

import numpy as np

from features.table import TARGET


class RoadClassHourMean:
    """Baseline 1: the training mean for the row's road class and hour."""

    def fit(self, train):
        self.means = train.groupby(["road_class_rank", "hour"])[TARGET].mean().to_dict()
        self.overall = float(train[TARGET].mean())
        return self

    def predict(self, test):
        keys = zip(test["road_class_rank"], test["hour"])
        return np.array([self.means.get(key, self.overall) for key in keys])


class NearestObservedEdge:
    """Baseline 2: the speed on the nearest training edge from another street.

    "same_bin" copies from the row's own bin, for the street split. "latest_bin"
    copies from the latest training bin, for the time split, where test bins are
    never in training. Rows with nothing to copy use the fallback baseline.
    """

    def __init__(self, midpoints, fallback):
        self.midpoints = midpoints
        self.fallback = fallback

    def fit(self, train):
        """Index training rows by (source, bin), so each lookup scans one bin only."""
        self.pools = {
            key: (
                rows["edge_id"].to_numpy(),
                rows["street_name"].to_numpy(),
                np.array([self.midpoints[edge] for edge in rows["edge_id"]], float),
                rows[TARGET].to_numpy(float),
            )
            for key, rows in train.groupby(["source", "bin_start_utc"])
        }
        self.latest = train.groupby("source")["bin_start_utc"].max().to_dict()
        return self

    def predict(self, test, mode="same_bin"):
        """Return (predictions, number of rows that used the fallback)."""
        backup = self.fallback.predict(test)
        predictions, fallbacks = [], 0
        for index, row in enumerate(test.itertuples(index=False)):
            value = self._copy(row, mode)
            if value is None:
                value, fallbacks = backup[index], fallbacks + 1
            predictions.append(value)
        return np.array(predictions, float), fallbacks

    def _copy(self, row, mode):
        when = row.bin_start_utc if mode == "same_bin" else self.latest.get(row.source)
        pool = self.pools.get((row.source, when))
        if pool is None:
            return None
        edges, streets, points, speeds = pool
        other = streets != row.street_name if row.street_name else edges != row.edge_id
        if not other.any():
            return None
        offsets = points[other] - np.asarray(self.midpoints[row.edge_id], float)
        return float(speeds[other][int(np.argmin((offsets**2).sum(axis=1)))])


def edge_midpoints(graph, edge_ids):
    """Return {edge_id: (x, y)}, the point halfway between each edge's ends, in metres."""
    points = {}
    for edge_id in set(edge_ids):
        edge = graph.edge(int(edge_id))
        (ux, uy), (vx, vy) = graph.node(edge.u).xy, graph.node(edge.v).xy
        points[edge_id] = ((ux + vx) / 2, (uy + vy) / 2)
    return points
