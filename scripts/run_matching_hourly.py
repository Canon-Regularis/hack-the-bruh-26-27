"""Match fixed, non-overlapping windows (UTC) as they complete.

Backfills every [start, start+interval) window already fully covered by
collected data, skipping any whose output already exists, then keeps running
forever: sleeps until each new window boundary passes (plus a little slack
for the last pings to land), matches the window that just ended, and saves
it.

Each window is saved to data/derived/speeds_HHMM.csv, named by its start
time, e.g. speeds_2100.csv for 21:00-23:00 UTC with the default 2-hour
interval.

Usage (start once, inside tmux so it survives):
    python3 scripts/run_matching_hourly.py
    python3 scripts/run_matching_hourly.py --start 2026-11-07T21:00:00+00:00
    python3 scripts/run_matching_hourly.py --start 2026-11-07T21:00:00+00:00 --interval-hours 2
"""

import argparse
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
DEFAULT_INTERVAL_HOURS = 2.0
SLACK_S = 5  # wait a little past the boundary for the last pings to land


def floor_hour(ts):
    return ts.replace(minute=0, second=0, microsecond=0, nanosecond=0)


def out_path_for(window_start):
    return OUT_DIR / f"speeds_{window_start.strftime('%H%M')}.csv"


def default_start(raw_path):
    """The earliest hour the collected data covers, if --start isn't given."""
    df = load_valid(raw_path)
    if df.empty:
        return None
    return floor_hour(df["vehicle_timestamp"].min())


def complete_windows(start, interval):
    """Every [start, start+interval), [start+interval, start+2*interval), ...
    window that has fully elapsed already, as (window_start, window_end) pairs.
    """
    now = pd.Timestamp.now(tz="UTC")
    windows = []
    window_start = start
    while window_start + interval <= now:
        windows.append((window_start, window_start + interval))
        window_start += interval
    return windows


def run_window(window_start, window_end, graph, idx, stops, street_names):
    out_path = out_path_for(window_start)
    if out_path.exists():
        print(f"skip {window_start}: {out_path} already exists")
        return
    print(f"=== {window_start} -> {window_end} ===")
    run_once(DEFAULT_RAW, window_start, window_end, out_path, DEFAULT_BBOX,
              graph=graph, idx=idx, stops=stops, street_names=street_names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=str, default=None,
                         help="pin the backfill's first window start, ISO UTC, "
                              "e.g. 2026-11-07T21:00:00+00:00")
    parser.add_argument("--interval-hours", type=float, default=DEFAULT_INTERVAL_HOURS,
                         help=f"window length in hours (default {DEFAULT_INTERVAL_HOURS})")
    args = parser.parse_args()
    interval = pd.Timedelta(hours=args.interval_hours)
    start = pd.Timestamp(args.start) if args.start else default_start(DEFAULT_RAW)
    if start is None:
        print("no data yet -- nothing to backfill or wait on")
        return

    graph = load_graph()
    idx = sn.EdgeIndex(graph, mode="bus")
    stops = json.load(open(GRAPH_DIR / "bus_stops.json"))
    street_names = json.load(open(GRAPH_DIR / "street_names.json"))

    backfill = complete_windows(start, interval)
    print(f"backfilling {len(backfill)} complete {args.interval_hours}h window(s) from {start}")
    for window_start, window_end in backfill:
        run_window(window_start, window_end, graph, idx, stops, street_names)

    next_start = backfill[-1][1] if backfill else start
    while True:
        next_end = next_start + interval
        now = pd.Timestamp.now(tz="UTC")
        sleep_s = max((next_end - now).total_seconds(), 0) + SLACK_S
        print(f"sleeping {sleep_s / 60:.1f} min until just after {next_end}")
        time.sleep(sleep_s)
        run_window(next_start, next_end, graph, idx, stops, street_names)
        next_start = next_end


if __name__ == "__main__":
    main()
