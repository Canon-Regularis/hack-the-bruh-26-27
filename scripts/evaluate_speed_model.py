"""Score the speed model against both baselines and write a timestamped report.

Run from the repo root, with the same import paths as pytest.ini:
    PYTHONPATH=.:data python3 scripts/evaluate_speed_model.py --speeds FILE

FILE is an edge-speed table from speeds.edge_speeds.edge_speed_table, as
parquet or CSV. The report goes to data/derived/reports/, which git ignores.
Exits 1 if the leakage check fails.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from features.betweenness import read_betweenness
from features.street import read_aadf, read_flags
from features.table import build_feature_table, missing_inputs
from model.baselines import edge_midpoints
from model.evaluate import build_report, evaluate
from src.graph.build_graph import load_graph

DATA = Path(__file__).resolve().parent.parent / "data"
REPORT_DIR = DATA / "derived" / "reports"


def read_speeds(path):
    """Read an edge-speed table, keeping edge ids as strings and times in UTC."""
    if path.suffix == ".parquet":
        speeds = pd.read_parquet(path)
    else:
        speeds = pd.read_csv(path, dtype={"edge_id": str, "street_name": str})
    speeds["edge_id"] = speeds["edge_id"].astype(str)
    speeds["street_name"] = speeds["street_name"].fillna("")
    speeds["bin_start_utc"] = pd.to_datetime(speeds["bin_start_utc"], utc=True)
    return speeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speeds", type=Path, required=True)
    args = parser.parse_args()

    started = time.monotonic()
    graph = load_graph()
    flags = read_flags(DATA / "rules" / "restriction_flags.csv")
    aadf = read_aadf(DATA / "graph" / "dft_aadf.csv")
    betweenness = read_betweenness(DATA / "derived" / "betweenness.json")
    table = build_feature_table(read_speeds(args.speeds), graph, betweenness, flags, aadf)
    results = evaluate(table, edge_midpoints(graph, table["edge_id"]))
    report = build_report(table, results, {
        "speeds_file": str(args.speeds),
        "missing_inputs": missing_inputs(flags, aadf),
        "betweenness_computed": bool(betweenness),
        "runtime_s": round(time.monotonic() - started, 1),
    })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPORT_DIR / f"speed_model_{stamp}.json"
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    print(f"Source: {', '.join(report['data']['source'])}. Rows: {len(table)}. Report: {out}")
    for split in ("street_split", "time_split"):
        r = results[split]
        print(f"{split}: MAE model {r['mae_model']:.2f}, baseline 1 {r['mae_baseline_1']:.2f}, "
              f"baseline 2 {r['mae_baseline_2']:.2f} km/h; skill vs baseline 2 "
              f"{r['skill_vs_baseline_2']:.2f}")
    if results["leakage_check"]["leakage_suspected"]:
        print("STOP: a model on shuffled targets beat the median guess, so something is leaking.")
        sys.exit(1)


if __name__ == "__main__":
    main()
