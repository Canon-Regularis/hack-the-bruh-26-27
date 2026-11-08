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
from features.street import read_aadf, read_concerns, read_flags
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


def interval_line(intervals):
    """Return one line on interval coverage, width and q, with "n/a" for any None."""
    def show(value, spec):
        return "n/a" if value is None else format(value, spec)

    return (f"intervals (target {1 - intervals['alpha']:.0%}): measured coverage "
            f"{show(intervals['coverage'], '.3f')} "
            f"on held-out {intervals['source']} streets (n={intervals['n']}), mean width "
            f"{show(intervals['mean_width_kmh'], '.2f')} km/h, "
            f"q {show(intervals['final_q_kmh'], '.2f')} km/h")


def read_inputs():
    """Return the graph, betweenness, restriction flags, AADF and concern intensity.

    The last three are None when their file does not exist yet.
    """
    return (
        load_graph(),
        read_betweenness(DATA / "derived" / "betweenness.json"),
        read_flags(DATA / "rules" / "restriction_flags.csv"),
        read_aadf(DATA / "graph" / "dft_aadf.csv"),
        read_concerns(DATA / "concerns" / "concern_field.csv"),
    )


def stamp():
    """Return the current UTC time for output file names."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(data, path):
    """Write data as JSON, refusing NaN, and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    return path


def run_evaluation(speeds, graph, betweenness, flags, aadf, concerns, speeds_file):
    """Return the evaluation report for an edge-speed table."""
    started = time.monotonic()
    table = build_feature_table(speeds, graph, betweenness, flags, aadf, concerns)
    results = evaluate(table, edge_midpoints(graph, table["edge_id"]))
    return build_report(table, results, {
        "speeds_file": str(speeds_file),
        "missing_inputs": missing_inputs(flags, aadf, concerns),
        "betweenness_computed": bool(betweenness),
        "runtime_s": round(time.monotonic() - started, 1),
    })


def print_report(report, out):
    """Print the scores. Returns True if the leakage check failed."""
    results = report["results"]
    data = report["data"]
    print(f"Source: {', '.join(data['source'])}. Rows: {data['n_rows']}. Report: {out}")
    for split in ("street_split", "time_split"):
        r = results[split]
        print(f"{split}: MAE model {r['mae_model']:.2f}, baseline 1 {r['mae_baseline_1']:.2f}, "
              f"baseline 2 {r['mae_baseline_2']:.2f} km/h; skill vs baseline 2 "
              f"{r['skill_vs_baseline_2']:.2f}")
    print(interval_line(results["street_split_intervals"]))
    if results["leakage_check"]["leakage_suspected"]:
        print("STOP: a model on shuffled targets beat the median guess, so something is leaking.")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speeds", type=Path, required=True)
    args = parser.parse_args()

    report = run_evaluation(read_speeds(args.speeds), *read_inputs(), args.speeds)
    out = write_json(report, REPORT_DIR / f"speed_model_{stamp()}.json")
    if print_report(report, out):
        sys.exit(1)


if __name__ == "__main__":
    main()
