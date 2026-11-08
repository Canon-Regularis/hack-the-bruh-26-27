"""Run the real pipeline end to end: raw pings -> matched edge speeds.

Time-window the raw rows -> match_pings (which does its own dedup and bbox
filtering) -> edge_speed_table -> saved to data/derived/.

Usage (from the repo root):
    python3 scripts/run_matching.py
    python3 scripts/run_matching.py --minutes 60 --out data/derived/speeds.parquet
    python3 scripts/run_matching.py --start 2026-11-07T21:00:00+00:00 \\
        --end 2026-11-07T22:00:00+00:00 --out data/derived/speeds_2100.csv
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
GRAPH_DIR = REPO_ROOT / "data" / "graph"


def load_valid(raw_path):
    """Every row with a parseable timestamp and lat/lon, nothing else filtered."""
    df = pd.read_csv(raw_path)
    df["vehicle_timestamp"] = pd.to_datetime(df["vehicle_timestamp"], errors="coerce", utc=True)
    return df.dropna(subset=["vehicle_timestamp", "lat", "lon"])


def to_rows(window):
    return [
        {
            "vehicle_id": row.vehicle_id,
            "vehicle_timestamp": row.vehicle_timestamp.isoformat(),
            "lat": row.lat,
            "lon": row.lon,
        }
        for row in window.itertuples()
    ]


def rows_in_range(raw_path, start, end):
    """Collector-row-shaped dicts with start <= vehicle_timestamp < end.

    Only the time window is applied here. Dedup and the study-area bbox are
    match_pings' own job, so every row in this window -- even ones outside
    the study area -- is handed to it unfiltered.
    """
    df = load_valid(raw_path)
    window = df[(df["vehicle_timestamp"] >= start) & (df["vehicle_timestamp"] < end)]
    print(f"{len(df):,} valid rows -> {len(window):,} in {start} -> {end}")
    return to_rows(window)


def load_hand_off():
    stops = json.load(open(GRAPH_DIR / "bus_stops.json"))
    street_names = json.load(open(GRAPH_DIR / "street_names.json"))
    return stops, street_names


def save(table, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".csv":
        table.to_csv(out_path, index=False)
        print(f"saved -> {out_path}")
        return
    try:
        table.to_parquet(out_path, index=False)
        print(f"saved -> {out_path}")
    except ImportError:
        csv_path = out_path.with_suffix(".csv")
        table.to_csv(csv_path, index=False)
        print(f"no pyarrow/fastparquet installed; saved as CSV instead -> {csv_path}")


def run_once(raw_path, start, end, out_path, bbox, *, graph=None, idx=None,
             stops=None, street_names=None):
    """Run the whole pipeline for one fixed [start, end) window and save it."""
    t_start = time.monotonic()

    rows = rows_in_range(raw_path, start, end)

    if graph is None:
        graph = load_graph()
    if idx is None:
        idx = sn.EdgeIndex(graph, mode="bus")
    if stops is None or street_names is None:
        stops, street_names = load_hand_off()

    t0 = time.monotonic()
    pairs = match_pings(rows, graph=graph, idx=idx, bbox=bbox)
    match_elapsed = time.monotonic() - t0
    rate_ms = (match_elapsed / len(rows) * 1000) if rows else 0.0
    print(f"{len(pairs):,} PingPairs matched in {match_elapsed:.1f}s ({rate_ms:.1f} ms/ping)")

    table = edge_speed_table(pairs, stops, street_names)
    print(f"{len(table):,} rows in the edge speed table")

    save(table, out_path)
    print(f"total pipeline time: {time.monotonic() - t_start:.1f}s")
    return {"rows": len(rows), "pairs": len(pairs), "table_rows": len(table)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--minutes", type=int, default=None,
                         help="most recent N minutes of data (default 20 if --start/--end not given)")
    parser.add_argument("--start", type=str, default=None,
                         help="fixed window start, ISO UTC, e.g. 2026-11-07T21:00:00+00:00")
    parser.add_argument("--end", type=str, default=None, help="fixed window end, ISO UTC")
    parser.add_argument("--xmin", type=float, default=DEFAULT_BBOX["xmin"])
    parser.add_argument("--ymin", type=float, default=DEFAULT_BBOX["ymin"])
    parser.add_argument("--xmax", type=float, default=DEFAULT_BBOX["xmax"])
    parser.add_argument("--ymax", type=float, default=DEFAULT_BBOX["ymax"])
    args = parser.parse_args()
    bbox = {"xmin": args.xmin, "ymin": args.ymin, "xmax": args.xmax, "ymax": args.ymax}

    if args.start and args.end:
        start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    else:
        minutes = args.minutes if args.minutes is not None else 20
        end = load_valid(args.raw)["vehicle_timestamp"].max()
        start = end - pd.Timedelta(minutes=minutes)

    run_once(args.raw, start, end, args.out, bbox)


if __name__ == "__main__":
    main()
