"""Loading and bbox logic for the ping plot, kept apart from matplotlib."""

import csv

import pytest

from scripts.plot_pings import load_points, suggest_bbox


@pytest.fixture
def pings_csv(tmp_path):
    path = tmp_path / "pings.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lat", "lon", "other"])
        writer.writeheader()
        writer.writerow({"lat": "55.95", "lon": "-3.19", "other": "x"})
        writer.writerow({"lat": "55.96", "lon": "-3.18", "other": "y"})
        writer.writerow({"lat": "", "lon": "", "other": "no coordinates"})
    return path


def test_load_points_skips_rows_without_coordinates(pings_csv):
    lats, lons = load_points(pings_csv)
    assert lats == [55.95, 55.96]
    assert lons == [-3.19, -3.18]


def test_suggest_bbox_wraps_every_point_with_a_margin():
    bbox = suggest_bbox([55.95, 55.96], [-3.19, -3.18], margin=0.01)
    assert bbox["ymin"] < 55.95
    assert bbox["ymax"] > 55.96
    assert bbox["xmin"] < -3.19
    assert bbox["xmax"] > -3.18
