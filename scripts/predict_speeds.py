"""Write the prediction table: every study-area bus road and 15-minute bin of the night.

Run from the repo root, with the same import paths as pytest.ini:
    PYTHONPATH=.:data python3 scripts/predict_speeds.py --speeds FILE

FILE is an edge-speed table, as parquet or CSV. Each row of the output is either
a bus-observed speed or the model's prediction, marked in the "kind" column.
The table goes to data/derived/predictions/, which git ignores.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_speed_model import DATA, read_speeds  # noqa: E402
from features.betweenness import read_betweenness  # noqa: E402
from features.street import read_aadf, read_flags  # noqa: E402
from model.predict import STUDY_BBOX, prediction_table  # noqa: E402
from src.graph.build_graph import load_graph  # noqa: E402

OUT_DIR = DATA / "derived" / "predictions"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speeds", type=Path, required=True)
    for key, value in STUDY_BBOX.items():
        parser.add_argument(f"--{key}", type=float, default=value)
    args = parser.parse_args()
    bbox = {key: getattr(args, key) for key in STUDY_BBOX}

    table = prediction_table(
        read_speeds(args.speeds),
        load_graph(),
        read_betweenness(DATA / "derived" / "betweenness.json"),
        read_flags(DATA / "rules" / "restriction_flags.csv"),
        read_aadf(DATA / "graph" / "dft_aadf.csv"),
        bbox,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT_DIR / f"speed_predictions_{stamp}.csv"
    table.to_csv(out, index=False)

    counts = table["kind"].value_counts()
    print(f"Rows: {len(table)} ({table['edge_id'].nunique()} edges x "
          f"{table['bin_start_utc'].nunique()} bins). Observed: {counts.get('observed', 0)}, "
          f"predicted: {counts.get('predicted', 0)} "
          f"({counts.get('predicted', 0) / len(table):.1%}).")
    print(f"Bins: {table['bin_start_utc'].min()} to {table['bin_start_utc'].max()}. "
          f"Output: {out}")


if __name__ == "__main__":
    main()
