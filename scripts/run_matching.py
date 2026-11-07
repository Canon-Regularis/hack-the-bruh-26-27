"""Run the real pipeline end to end: raw pings -> matched edge speeds.

Time-window the raw rows -> match_pings (which does its own dedup and bbox
filtering) -> edge_speed_table -> saved to data/derived/.

Usage (from the repo root):
    python3 scripts/run_matching.py
    python3 scripts/run_matching.py --minutes 60 --out data/derived/speeds.parquet
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
from matching.table import match_pings  # noqa: E402
from matching.trips import DEFAULT_BBOX  # noqa: E402
from speeds.edge_speeds import edge_speed_table  # noqa: E402

DEFAULT_RAW = REPO_ROOT / "data" / "raw" / "pings.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "derived" / "speeds.parquet"


def recent_rows(raw_path, minutes):
    """Collector-row-shaped dicts from the most recent N minutes of raw_path.

    Only the time window is applied here. Dedup and the study-area bbox are
    match_pings' own job now, so every row in this window -- even ones
    outside the study area -- is handed to it unfiltered.
    """
    df = pd.read_csv(raw_path)
    df["vehicle_timestamp"] = pd.to_datetime(df["vehicle_timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["vehicle_timestamp", "lat", "lon"])

    end = df["vehicle_timestamp"].max()
    window = df[df["vehicle_timestamp"] >= end - pd.Timedelta(minutes=minutes)]
    print(f"{len(df):,} valid rows -> {len(window):,} in the most recent {minutes} min")

    return [
        {
            "vehicle_id": row.vehicle_id,
            "vehicle_timestamp": row.vehicle_timestamp.isoformat(),
            "lat": row.lat,
            "lon": row.lon,
        }
        for row in window.itertuples()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--minutes", type=int, default=20, help="most recent N minutes (default 20)")
    parser.add_argument("--xmin", type=float, default=DEFAULT_BBOX["xmin"])
    parser.add_argument("--ymin", type=float, default=DEFAULT_BBOX["ymin"])
    parser.add_argument("--xmax", type=float, default=DEFAULT_BBOX["xmax"])
    parser.add_argument("--ymax", type=float, default=DEFAULT_BBOX["ymax"])
    args = parser.parse_args()
    bbox = {"xmin": args.xmin, "ymin": args.ymin, "xmax": args.xmax, "ymax": args.ymax}

    t_start = time.monotonic()

    rows = recent_rows(args.raw, args.minutes)

    graph = load_graph()
    idx = sn.EdgeIndex(graph, mode="bus")
    stops = json.load(open(REPO_ROOT / "data" / "graph" / "bus_stops.json"))
    street_names = json.load(open(REPO_ROOT / "data" / "graph" / "street_names.json"))

    t0 = time.monotonic()
    pairs = match_pings(rows, graph=graph, idx=idx, bbox=bbox)
    match_elapsed = time.monotonic() - t0
    rate_ms = (match_elapsed / len(rows) * 1000) if rows else 0.0
    print(f"{len(pairs):,} PingPairs matched in {match_elapsed:.1f}s ({rate_ms:.1f} ms/ping)")

    table = edge_speed_table(pairs, stops, street_names)
    print(f"{len(table):,} rows in the edge speed table")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        table.to_parquet(args.out, index=False)
        print(f"saved -> {args.out}")
    except ImportError:
        csv_path = args.out.with_suffix(".csv")
        table.to_csv(csv_path, index=False)
        print(f"no pyarrow/fastparquet installed; saved as CSV instead -> {csv_path}")

    print(f"total pipeline time: {time.monotonic() - t_start:.1f}s")


if __name__ == "__main__":
    main()
