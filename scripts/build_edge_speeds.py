"""Turn collected bus pings into the edge-speed file the speed model reads.

Run from the repo root, with the same import paths as pytest.ini:
    PYTHONPATH=.:data python3 scripts/build_edge_speeds.py

Reads data/raw/pings.csv from the collector, matches it with
matching.table.match_pings, and writes data/derived/edge_speeds.csv, which git
ignores. Then run scripts/evaluate_speed_model.py --speeds on that file. Until
map matching is built, this stops with exit code 2 and writes nothing.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from matching.table import MatchingNotAvailable, match_pings
from speeds.edge_speeds import edge_speed_table

DATA = Path(__file__).resolve().parent.parent / "data"
DEFAULT_PINGS = DATA / "raw" / "pings.csv"
DEFAULT_OUT = DATA / "derived" / "edge_speeds.csv"


def read_pings(path):
    """Return the collector's rows as dicts, unchanged. Matching cleans them."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_hand_off():
    """Return (bus stops as (lat, lon), {edge_id: street name}) from the frozen graph."""
    graph_dir = DATA / "graph"
    stops = json.loads((graph_dir / "bus_stops.json").read_text(encoding="utf-8"))
    names = json.loads((graph_dir / "street_names.json").read_text(encoding="utf-8"))
    return [tuple(stop) for stop in stops], names


def build(pings, stops, street_names, match=match_pings):
    """Match the pings, then return the edge-speed table."""
    return edge_speed_table(match(pings), stops, street_names)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pings", type=Path, default=DEFAULT_PINGS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    stops, names = load_hand_off()
    try:
        table = build(read_pings(args.pings), stops, names, match=match_pings)
    except MatchingNotAvailable as error:
        print(f"Stopped: {error}. Nothing written.")
        sys.exit(2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    moving = int((table["n_traversals"] > 0).sum())
    print(f"{len(table)} edge-speed rows ({moving} with a speed) -> {args.out}")


if __name__ == "__main__":
    main()
