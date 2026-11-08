"""Engine's modelled link speeds against bus-observed speeds on the same edges.

Modelled speed is length_m / ue_time_s from the baseline user equilibrium, and
also length_m / free_flow_time_s. Observed speed is total metres over total
seconds per edge. The equilibrium is for one hour, so only bins in that hour are
compared once they exist; until then every bin is used and the result is
labelled as an hour mismatch. Buses are not cars: they stop and use bus lanes.

Bins are in UTC. Edinburgh is on GMT in November, so UTC hours are local hours.
"""

import json
import re

import numpy as np
import pandas as pd

from features.table import targets
from model.metrics import mae

CAVEAT = ("Bus-observed speeds, not car speeds: buses stop and use bus lanes, "
          "so they read slower than general traffic.")
HOUR_MISMATCH = ("Hour mismatch: no bus data falls in the equilibrium's hour yet, "
                 "so every observed bin is compared.")


def parse_hours(label):
    """Return (start_hour, end_hour) from a label such as '08:00-09:00 weekday'."""
    match = re.match(r"\s*(\d{1,2}):\d{2}\s*-\s*(\d{1,2}):\d{2}", label)
    if not match:
        raise ValueError(f"cannot read an hour range from {label!r}")
    return int(match.group(1)), int(match.group(2))


def read_ue(csv_path, meta_path):
    """Return (UE table with ue_kmh and free_flow_kmh, (start_hour, end_hour))."""
    ue = pd.read_csv(csv_path, dtype={"edge_id": str})
    ue["ue_kmh"] = ue["length_m"] / ue["ue_time_s"] * 3.6
    ue["free_flow_kmh"] = ue["length_m"] / ue["free_flow_time_s"] * 3.6
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return ue, parse_hours(meta["hour"])


def edge_speeds_in_hours(speeds, hours):
    """Return (per-edge observed km/h and traversals, whether the hours matched)."""
    moving = targets(speeds).assign(edge_id=lambda t: t["edge_id"].astype(str))
    hour = pd.to_datetime(moving["bin_start_utc"], utc=True).dt.hour
    in_hours = moving[(hour >= hours[0]) & (hour < hours[1])]
    matched = not in_hours.empty
    rows = in_hours if matched else moving
    per_edge = rows.groupby("edge_id", as_index=False)[
        ["total_metres", "total_seconds", "n_traversals"]].sum()
    per_edge["observed_kmh"] = per_edge["total_metres"] / per_edge["total_seconds"] * 3.6
    return per_edge[["edge_id", "observed_kmh", "n_traversals"]], matched


def compare(edge_speeds, ue, column):
    """Return n_edges, MAE and bias (modelled minus observed) on edges in both tables."""
    both = edge_speeds.merge(ue[["edge_id", column]], on="edge_id", how="inner")
    if both.empty:
        return {"n_edges": 0, "mae_kmh": None, "bias_kmh": None}
    return {
        "n_edges": len(both),
        "mae_kmh": mae(both["observed_kmh"], both[column]),
        "bias_kmh": float(np.mean(both[column] - both["observed_kmh"])),
    }


def both_speeds(edge_speeds, ue):
    return {"vs_ue": compare(edge_speeds, ue, "ue_kmh"),
            "vs_free_flow": compare(edge_speeds, ue, "free_flow_kmh")}


def assignment_check(speeds, predictions, ue, hours):
    """Return the comparison for observed edges and, apart, for predicted-only edges."""
    observed, matched = edge_speeds_in_hours(speeds, hours)
    rows = predictions[predictions["kind"] == "predicted"]
    if matched:
        hour = pd.to_datetime(rows["bin_start_utc"], utc=True).dt.hour
        rows = rows[(hour >= hours[0]) & (hour < hours[1])]
    rows = rows[~rows["edge_id"].astype(str).isin(observed["edge_id"])]
    predicted = (rows.assign(edge_id=rows["edge_id"].astype(str))
                 .groupby("edge_id", as_index=False)["speed_kmh"].mean()
                 .rename(columns={"speed_kmh": "observed_kmh"}))
    return {
        "hour_matched": matched,
        "equilibrium_hours_utc": list(hours),
        "observed": both_speeds(observed, ue),
        "predicted": both_speeds(predicted, ue),
        "caveats": [CAVEAT] + ([] if matched else [HOUR_MISMATCH]),
    }
