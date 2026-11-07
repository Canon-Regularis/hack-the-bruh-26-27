import csv

import pytest

from engine.orchid import street_network as sn
from src.graph import count_points as cp

DFT_HEADER = "count_point_id,year,road_name,latitude,longitude,all_motor_vehicles,cars_and_taxis\n"
DFT_ROWS = [
    ("1", 2019, "A700", 55.94903, -3.1990, 1000),
    ("1", 2023, "A700", 55.94903, -3.1990, 1200),
    ("2", 2023, "A700", 55.94913, -3.1990, 900),
    ("3", 2022, "U", 55.95003, -3.1990, 300),
    ("4", 2023, "U", 55.94960, -3.1990, 50),
    ("5", 2023, "A1", 55.96000, -3.1000, 20000),
    ("10", 2023, "U", 55.94960, -3.1985, 40),
]


@pytest.fixture(scope="module")
def dft_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("dft") / "dft_aadf.csv"
    path.write_text(DFT_HEADER + "".join(f"{c},{y},{r},{lat},{lon},{a},0\n" for c, y, r, lat, lon, a in DFT_ROWS))
    return path


@pytest.fixture(scope="module")
def graph():
    g = sn.RoadGraph()
    for nid, (lat, lon) in enumerate([(55.94900, -3.2000), (55.94900, -3.1980), (55.94915, -3.2000),
                                      (55.94915, -3.1980), (55.95000, -3.2000), (55.95000, -3.1980)]):
        g.add_node(nid, lat=lat, lon=lon)
    g.add_edge(0, 1, oneway=False, street_name="Lothian Road", highway_class="primary")   # 0, 1
    g.add_edge(2, 3, oneway=False, street_name="", highway_class="footway")              # 2, 3
    g.add_edge(4, 5, street_name="Bread Street")                                          # 4
    return g


@pytest.fixture(scope="module")
def snapped(dft_path, graph):
    return cp.snap_count_points(cp.load_count_points(dft_path), graph, sn.EdgeIndex(graph, mode="car"))


def test_load_keeps_latest_year_per_point_in_numeric_order(dft_path):
    points = {p["count_point_id"]: p for p in cp.load_count_points(dft_path)}
    assert list(points) == ["1", "2", "3", "4", "5", "10"]
    assert points["1"]["year"] == 2023 and points["1"]["aadf"] == 1200


def test_snaps_to_car_roads_and_nearest_point_wins_the_road(snapped):
    matches, dropped = snapped
    matched = {m["count_point_id"]: m["edge"].eid for m in matches}
    assert sorted(matched) == ["1", "3"]
    assert matched["1"] in (0, 1) and matched["3"] == 4
    assert {d["count_point_id"]: d["status"] for d in dropped} == {
        "2": "road taken by nearer point 1",
        "4": "no car road within 25 m",
        "10": "no car road within 25 m",
    }


def test_points_outside_the_graph_are_ignored(snapped):
    matches, dropped = snapped
    assert "5" not in {p["count_point_id"] for p in matches + dropped}


def test_aadf_rows_cover_both_directions_of_a_two_way_road(snapped):
    assert cp.aadf_rows(snapped[0]) == [
        {"edge_id": "0", "count_point_id": "1", "aadf": 1200},
        {"edge_id": "1", "count_point_id": "1", "aadf": 1200},
        {"edge_id": "4", "count_point_id": "3", "aadf": 300},
    ]


def test_review_rows_show_matches_and_drops(snapped):
    rows = {r["count_point_id"]: r for r in cp.review_rows(*snapped)}
    assert list(rows) == ["1", "2", "3", "4", "10"]
    assert rows["1"]["street_name"] == "Lothian Road" and float(rows["1"]["distance_m"]) == pytest.approx(3.3, abs=0.1)
    assert "edge_id" not in rows["4"]


def test_main_writes_both_csvs(dft_path, graph, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cp, "DFT_PATH", dft_path)
    monkeypatch.setattr(cp, "AADF_PATH", tmp_path / "graph" / "dft_aadf.csv")
    monkeypatch.setattr(cp, "REVIEW_PATH", tmp_path / "graph" / "dft_review.csv")
    monkeypatch.setattr(cp, "load_graph", lambda mode=None: (graph, sn.EdgeIndex(graph, mode=mode)))

    cp.main()

    assert capsys.readouterr().out.startswith("2 count points matched, 3 dropped")
    with (tmp_path / "graph" / "dft_aadf.csv").open() as f:
        assert [r["edge_id"] for r in csv.DictReader(f)] == ["0", "1", "4"]
    with (tmp_path / "graph" / "dft_review.csv").open() as f:
        review = list(csv.DictReader(f))
    assert [r["count_point_id"] for r in review] == ["1", "2", "3", "4", "10"] and review[3]["edge_id"] == ""
