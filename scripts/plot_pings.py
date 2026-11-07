"""Plot bus ping locations to help fix the real Old Town study-area bbox.

Reads data/raw/pings.csv (written by collector.py collect) and plots every
ping's lat/lon, so the coverage pattern is visible. Per the plan, restricted
Old Town streets will have little or no bus data; the signal lives on the
surrounding roads, which is exactly where diverted traffic goes.

This only suggests a tight bbox around every point seen so far -- it is not
the Old Town study-area bbox itself. Pick that by eye from the plot.

Usage (from the repo root):
    python3 scripts/plot_pings.py
    python3 scripts/plot_pings.py --raw data/raw/pings.csv --out data/out/ping_plot.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = REPO_ROOT / "data" / "raw" / "pings.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "out" / "ping_plot.png"


def load_points(raw_path):
    """Return (lats, lons) for every row with a valid coordinate pair."""
    lats, lons = [], []
    with open(raw_path, newline="") as f:
        for row in csv.DictReader(f):
            lat, lon = row.get("lat"), row.get("lon")
            if lat and lon:
                lats.append(float(lat))
                lons.append(float(lon))
    return lats, lons


def suggest_bbox(lats, lons, margin=0.001):
    """A tight bbox around every point seen, with a small margin."""
    return {
        "xmin": min(lons) - margin,
        "ymin": min(lats) - margin,
        "xmax": max(lons) + margin,
        "ymax": max(lats) + margin,
    }


def plot(lats, lons, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(lons, lats, s=2, alpha=0.3)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"{len(lats)} bus pings")
    ax.set_aspect("equal")
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    lats, lons = load_points(args.raw)
    if not lats:
        print(f"No points with lat/lon found in {args.raw}")
        return 1

    print(f"{len(lats)} points loaded from {args.raw}")
    bbox = suggest_bbox(lats, lons)
    print("Tight bbox around every point seen (not the Old Town study area):")
    print(
        f"  --xmin {bbox['xmin']:.4f} --ymin {bbox['ymin']:.4f} "
        f"--xmax {bbox['xmax']:.4f} --ymax {bbox['ymax']:.4f}"
    )

    plot(lats, lons, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
