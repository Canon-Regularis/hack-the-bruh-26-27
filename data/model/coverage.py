"""How much of the study area is observed, and how slow edges near bus stops read.

coverage_summary gives the headline for the frontend: the share of street-time
with no bus observation, and the model's error on it. slow_rows checks whether
very slow bus-observed rows cluster near bus stops, where dwell may leak in.
"""

import pandas as pd

from features.table import TARGET, targets
from speeds.dwell import STOP_RADIUS_M, distance_m

LABEL = "bus-observed speed"
SLOW_KMH = 2.0  # for reporting only: nothing is filtered on it
N_SLOWEST = 10


def coverage_summary(predictions, report):
    """Return the share of street-time predicted and the model's error, as a dict."""
    times = pd.to_datetime(predictions["bin_start_utc"], utc=True)
    n_rows = len(predictions)
    n_predicted = int((predictions["kind"] == "predicted").sum())
    street, later = report["results"]["street_split"], report["results"]["time_split"]
    return {
        "label": LABEL,
        "time_range_utc": [times.min().isoformat(), times.max().isoformat()],
        "n_edges": int(predictions["edge_id"].nunique()),
        "n_bins": int(times.nunique()),
        "n_rows": n_rows,
        "n_observed": n_rows - n_predicted,
        "n_predicted": n_predicted,
        "predicted_share": n_predicted / n_rows if n_rows else 0.0,
        "error_unseen_streets_kmh": street["mae_model"],
        "error_later_bins_kmh": later["mae_model"],
        "baselines_unseen_streets_kmh": baselines(street),
        "baselines_later_bins_kmh": baselines(later),
        "interval": interval(report["results"].get("street_split_intervals")),
    }


def interval(intervals):
    """Return the conformal interval for predictions (speed plus or minus q), or None."""
    if not intervals or intervals.get("final_q_kmh") is None:
        return None
    return {
        "target_coverage": round(1 - intervals["alpha"], 6),
        "measured_coverage_unseen_streets": intervals["coverage"],
        "half_width_kmh": intervals["final_q_kmh"],
        "mean_width_kmh": intervals["mean_width_kmh"],
    }


def baselines(split):
    return {"baseline_1": split["mae_baseline_1"], "baseline_2": split["mae_baseline_2"]}


def near_stop_edges(graph, edge_ids, stops):
    """Return the edges with a bus stop within STOP_RADIUS_M of an end or the midpoint."""
    near = set()
    for edge_id in set(edge_ids):
        edge = graph.edge(int(edge_id))
        u, v = graph.node(edge.u), graph.node(edge.v)
        points = [(u.lat, u.lon), (v.lat, v.lon), ((u.lat + v.lat) / 2, (u.lon + v.lon) / 2)]
        if any(distance_m(point, stop) <= STOP_RADIUS_M
               for point in points for stop in close_stops(points, stops)):
            near.add(edge_id)
    return near


def close_stops(points, stops, margin_deg=0.001):
    """Return the stops inside the points' lat/lon box plus about 100 m, to skip the rest."""
    lats, lons = [p[0] for p in points], [p[1] for p in points]
    return [s for s in stops
            if min(lats) - margin_deg <= s[0] <= max(lats) + margin_deg
            and min(lons) - margin_deg <= s[1] <= max(lons) + margin_deg]


def slow_rows(speeds, graph, stops):
    """Return speed statistics for moving rows on edges near a stop and elsewhere."""
    moving = targets(speeds).assign(edge_id=lambda t: t["edge_id"].astype(str))
    near = near_stop_edges(graph, moving["edge_id"], stops)
    is_near = moving["edge_id"].isin(near)
    return {
        "slow_kmh": SLOW_KMH,
        "stop_radius_m": STOP_RADIUS_M,
        "near_stop": group_stats(moving[is_near]),
        "elsewhere": group_stats(moving[~is_near]),
    }


def group_stats(rows):
    speeds = rows[TARGET]
    n_slow = int((speeds < SLOW_KMH).sum())
    slowest = rows.nsmallest(N_SLOWEST, TARGET)
    return {
        "n_rows": len(rows),
        "median_kmh": float(speeds.median()) if len(rows) else None,
        "n_slow": n_slow,
        "slow_share": n_slow / len(rows) if len(rows) else 0.0,
        "slowest": [
            {"edge_id": r.edge_id, "street_name": r.street_name,
             "bin_start_utc": pd.Timestamp(r.bin_start_utc).isoformat(),
             "total_metres": float(r.total_metres), "total_seconds": float(r.total_seconds),
             "speed_kmh": float(getattr(r, TARGET))}
            for r in slowest.itertuples(index=False)
        ],
    }
