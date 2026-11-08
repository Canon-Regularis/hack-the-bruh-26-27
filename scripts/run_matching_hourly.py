"""Match fixed, non-overlapping UTC hours as they complete.

Backfills every [HH:00, HH+1:00) window already fully covered by collected
data, skipping any whose output already exists, then keeps running forever:
sleeps until each new hour boundary passes (plus a little slack for the
last pings to land), matches the hour that just ended, and saves it.

Each window is saved to data/derived/speeds_HHMM.csv, named by its start
hour, e.g. speeds_2100.csv for 21:00-22:00 UTC.

Usage (start once, inside tmux so it survives):
    python3 scripts/run_matching_hourly.py
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data"))

from engine.orchid import street_network as sn  # noqa: E402
from src.graph.build_graph import load_graph  # noqa: E402
from matching.trips import DEFAULT_BBOX  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_matching import DEFAULT_RAW, GRAPH_DIR, load_valid, run_once  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "derived"
SLACK_S = 5  # wait a little past the boundary for the last pings to land


def floor_hour(ts):
    return ts.replace(minute=0, second=0, microsecond=0, nanosecond=0)


def out_path_for(hour_start):
    return OUT_DIR / f"speeds_{hour_start.strftime('%H%M')}.csv"


def complete_hours(raw_path):
    """Every [HH:00, HH+1:00) window that has fully elapsed already."""
    df = load_valid(raw_path)
    if df.empty:
        return []
    first_hour = floor_hour(df["vehicle_timestamp"].min())
    current_hour = floor_hour(pd.Timestamp.now(tz="UTC"))
    hours, h = [], first_hour
    while h < current_hour:
        hours.append(h)
        h += pd.Timedelta(hours=1)
    return hours


def run_hour(hour_start, graph, idx, stops, street_names):
    out_path = out_path_for(hour_start)
    if out_path.exists():
        print(f"skip {hour_start}: {out_path} already exists")
        return
    hour_end = hour_start + pd.Timedelta(hours=1)
    print(f"=== {hour_start} -> {hour_end} ===")
    run_once(DEFAULT_RAW, hour_start, hour_end, out_path, DEFAULT_BBOX,
              graph=graph, idx=idx, stops=stops, street_names=street_names)


def main():
    graph = load_graph()
    idx = sn.EdgeIndex(graph, mode="bus")
    stops = json.load(open(GRAPH_DIR / "bus_stops.json"))
    street_names = json.load(open(GRAPH_DIR / "street_names.json"))

    backfill = complete_hours(DEFAULT_RAW)
    print(f"backfilling {len(backfill)} complete hour(s)")
    for hour_start in backfill:
        run_hour(hour_start, graph, idx, stops, street_names)

    while True:
        now = pd.Timestamp.now(tz="UTC")
        next_hour = floor_hour(now) + pd.Timedelta(hours=1)
        sleep_s = max((next_hour - now).total_seconds(), 0) + SLACK_S
        print(f"sleeping {sleep_s / 60:.1f} min until just after {next_hour}")
        time.sleep(sleep_s)
        run_hour(next_hour - pd.Timedelta(hours=1), graph, idx, stops, street_names)


if __name__ == "__main__":
    main()
