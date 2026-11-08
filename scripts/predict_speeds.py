"""Write the prediction table: every study-area bus road and 15-minute bin of the night.

Run from the repo root, with the same import paths as pytest.ini:
    PYTHONPATH=.:data python3 scripts/predict_speeds.py --speeds FILE

FILE is an edge-speed table, as parquet or CSV. Each row of the output is either
a bus-observed speed or the model's prediction, marked in the "kind" column.
The table goes to data/derived/predictions/, which git ignores.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_speed_model import DATA, read_inputs, read_speeds, stamp  # noqa: E402
from model.predict import STUDY_BBOX, prediction_table  # noqa: E402

OUT_DIR = DATA / "derived" / "predictions"


def write_predictions(table, out_dir=OUT_DIR):
    """Write the prediction table as CSV and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"speed_predictions_{stamp()}.csv"
    table.to_csv(out, index=False)
    return out


def print_predictions(table, out):
    counts = table["kind"].value_counts()
    predicted = counts.get("predicted", 0)
    print(f"Rows: {len(table)} ({table['edge_id'].nunique()} edges x "
          f"{table['bin_start_utc'].nunique()} bins). Observed: {counts.get('observed', 0)}, "
          f"predicted: {predicted} ({predicted / len(table):.1%}).")
    print(f"Bins: {table['bin_start_utc'].min()} to {table['bin_start_utc'].max()}. "
          f"Output: {out}")


def add_bbox_args(parser):
    for key, value in STUDY_BBOX.items():
        parser.add_argument(f"--{key}", type=float, default=value)


def bbox_from(args):
    return {key: getattr(args, key) for key in STUDY_BBOX}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speeds", type=Path, required=True)
    add_bbox_args(parser)
    args = parser.parse_args()

    table = prediction_table(read_speeds(args.speeds), *read_inputs(), bbox_from(args))
    print_predictions(table, write_predictions(table))


if __name__ == "__main__":
    main()
