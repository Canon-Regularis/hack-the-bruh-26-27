"""Run Signal's whole night in one go: join, evaluate, predict, summarise, compare.

Run from the repo root, with the same import paths as pytest.ini:
    PYTHONPATH=.:data python3 scripts/run_signal.py --speeds FILE [FILE ...]

Give the edge-speed windows oldest first. A window matched again goes last, so
it replaces the old one. Writes, all under data/derived/ (git ignores it):
  speeds_night.csv                     the joined edge-speed table
  reports/speed_model_<stamp>.json     the evaluation
  predictions/speed_predictions_<stamp>.csv and summary_<stamp>.json
  reports/assignment_check_<stamp>.json   if Engine's equilibrium file exists
Exits 1 if the leakage check fails, before anything is predicted.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_speed_model import (  # noqa: E402
    DATA,
    REPORT_DIR,
    print_report,
    read_inputs,
    read_speeds,
    run_evaluation,
    stamp,
    write_json,
)
from model.assignment_check import assignment_check, read_ue  # noqa: E402
from model.coverage import coverage_summary, slow_rows  # noqa: E402
from model.predict import prediction_table  # noqa: E402
from predict_speeds import (  # noqa: E402
    OUT_DIR,
    add_bbox_args,
    bbox_from,
    print_predictions,
    write_predictions,
)
from speeds.join import join_windows  # noqa: E402

UE_CSV = DATA / "demand" / "baseline_user_equilibrium.csv"
UE_META = DATA / "demand" / "baseline_user_equilibrium.json"
STOPS = DATA / "graph" / "bus_stops.json"


def run(speed_paths, inputs, stops, ue, bbox, derived=DATA / "derived"):
    """Run every step and return {name: output path}, or None if leakage is suspected.

    inputs is (graph, betweenness, flags, aadf). ue is (table, hours), or None to skip.
    """
    graph = inputs[0]
    now = stamp()
    outputs = {"speeds": derived / "speeds_night.csv"}

    speeds = join_windows([read_speeds(path) for path in speed_paths])
    outputs["speeds"].parent.mkdir(parents=True, exist_ok=True)
    speeds.to_csv(outputs["speeds"], index=False)
    print(f"Joined {len(speed_paths)} files: {len(speeds)} rows. Output: {outputs['speeds']}")

    report = run_evaluation(speeds, *inputs, outputs["speeds"])
    outputs["report"] = write_json(report, derived / REPORT_DIR.name / f"speed_model_{now}.json")
    if print_report(report, outputs["report"]):
        return None

    predictions = prediction_table(speeds, *inputs, bbox)
    outputs["predictions"] = write_predictions(predictions, derived / OUT_DIR.name)
    print_predictions(predictions, outputs["predictions"])

    summary = coverage_summary(predictions, report)
    summary["checks"] = {"slow_rows": slow_rows(speeds, graph, stops)}
    outputs["summary"] = write_json(summary, derived / OUT_DIR.name / f"summary_{now}.json")
    print_summary(summary, outputs["summary"])

    if ue is not None:
        check = assignment_check(speeds, predictions, *ue)
        outputs["assignment"] = write_json(
            check, derived / REPORT_DIR.name / f"assignment_check_{now}.json")
        print_assignment(check, outputs["assignment"])
    else:
        print("Assignment check skipped: no equilibrium file.")
    return outputs


def print_summary(summary, out):
    slow = summary["checks"]["slow_rows"]
    print(f"Headline: {summary['predicted_share']:.0%} of street-time has no bus observation; "
          f"predicted with error {summary['error_unseen_streets_kmh']:.2f} km/h on unseen "
          f"streets. Output: {out}")
    for group in ("near_stop", "elsewhere"):
        g = slow[group]
        print(f"Slow rows {group}: {g['n_slow']} of {g['n_rows']} under {slow['slow_kmh']} km/h "
              f"({g['slow_share']:.1%}), median {g['median_kmh']} km/h")


def print_assignment(check, out):
    hours = check["equilibrium_hours_utc"]
    label = "hour matched" if check["hour_matched"] else "HOUR MISMATCH, all bins used"
    print(f"Assignment check vs {hours[0]:02d}:00-{hours[1]:02d}:00 equilibrium ({label}):")
    for kind in ("observed", "predicted"):
        for name, r in check[kind].items():
            if r["n_edges"]:
                print(f"  {kind} {name}: {r['n_edges']} edges, MAE {r['mae_kmh']:.2f}, "
                      f"bias {r['bias_kmh']:+.2f} km/h (modelled minus bus)")
    print(f"  Output: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speeds", type=Path, nargs="+", required=True)
    add_bbox_args(parser)
    args = parser.parse_args()

    stops = json.loads(STOPS.read_text(encoding="utf-8"))
    ue = read_ue(UE_CSV, UE_META) if UE_CSV.exists() else None
    outputs = run(args.speeds, read_inputs(), stops, ue, bbox_from(args))
    if outputs is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
