import csv
import math

import pytest

from engine.orchid import street_network as sn
from src.concerns import concern_field as cf

HEADER = "theme,lat,lon,street,count,severity\n"
LAT = 55.9490


@pytest.fixture(scope="module")
def graph():
    """Lawnmarket (two-way) running east, then Upper Bow (one-way) 400 m further east."""
    g = sn.RoadGraph()
    for nid, lon in enumerate([-3.2000, -3.1990, -3.1930, -3.1920]):
        g.add_node(nid, lat=LAT, lon=lon)
    g.add_edge(0, 1, oneway=False, street_name="Lawnmarket")   # 0, 1
    g.add_edge(2, 3, street_name="Upper Bow")                   # 2
    return g


def write(tmp_path, rows):
    path = tmp_path / "concerns.csv"
    path.write_text(HEADER + "".join(rows))
    return path


def test_load_concerns_reads_location_street_and_weight(tmp_path):
    records = cf.load_concerns(write(tmp_path, ["road_safety,55.949,-3.2,,4,0.5\n", "parking,,,Lawnmarket,3,\n"]))
    assert records == [
        {"theme": "road_safety", "loc": (55.949, -3.2), "street": "", "weight": 2.0},
        {"theme": "parking", "loc": None, "street": "Lawnmarket", "weight": 3.0},
    ]


@pytest.mark.parametrize("row, message", [
    ("speeding,55.949,-3.2,,1,\n", "unknown theme 'speeding'"),
    ("noise,55.949,,,1,\n", "give both lat and lon, or neither"),
    ("noise,,,,1,\n", "needs a location or a street"),
])
def test_load_concerns_rejects_bad_rows_with_line_number(tmp_path, row, message):
    with pytest.raises(ValueError, match=f"concerns.csv:2: .*{message}"):
        cf.load_concerns(write(tmp_path, [row]))


def test_located_field_is_gaussian_in_distance_and_reaches_both_directions(graph):
    lat_50m_north = LAT + 50 / 111_195
    field = cf.located_field(graph, [{"theme": "noise", "loc": (lat_50m_north, -3.1995), "street": "", "weight": 2.0}])
    assert set(field) == {"0", "1"}
    assert field["0"]["noise"] == field["1"]["noise"] == pytest.approx(2.0 * math.exp(-0.5), rel=1e-3)


def test_located_field_ignores_edges_beyond_the_cutoff(graph):
    lat_200m_north = LAT + 200 / 111_195
    assert cf.located_field(graph, [{"theme": "noise", "loc": (lat_200m_north, -3.1995), "street": "", "weight": 1.0}]) == {}


def test_located_field_adds_records_per_theme(graph):
    on_road = (LAT, -3.1995)
    field = cf.located_field(graph, [
        {"theme": "noise", "loc": on_road, "street": "", "weight": 1.0},
        {"theme": "noise", "loc": on_road, "street": "", "weight": 2.0},
        {"theme": "parking", "loc": on_road, "street": "", "weight": 5.0},
    ])
    assert field["0"] == {"noise": pytest.approx(3.0), "parking": pytest.approx(5.0)}


def test_street_field_gives_full_count_to_every_edge_and_reports_unknown_streets(graph):
    field, unresolved = cf.street_field(graph, [
        {"theme": "parking", "loc": None, "street": "Lawnmarket", "weight": 3.0},
        {"theme": "parking", "loc": None, "street": "LAWNMARKET", "weight": 1.0},
        {"theme": "noise", "loc": None, "street": "Upper Bow", "weight": 2.0},
        {"theme": "noise", "loc": None, "street": "Castlehill", "weight": 9.0},
        {"theme": "noise", "loc": (LAT, -3.2), "street": "", "weight": 9.0},
    ])
    assert field == {"0": {"parking": 4.0}, "1": {"parking": 4.0}, "2": {"noise": 2.0}}
    assert unresolved == ["Castlehill"]


def test_field_rows_one_per_directed_edge_with_every_theme(graph):
    rows = cf.field_rows(graph, {"1": {"noise": 2.5}}, cf.LOCATED)
    assert [r["edge_id"] for r in rows] == ["0", "1", "2"]
    assert list(rows[1]) == list(cf.FIELD_COLUMNS)
    assert rows[1]["noise"] == 2.5 and rows[1]["parking"] == 0.0 and rows[0]["noise"] == 0.0
    assert {r["basis"] for r in rows} == {cf.LOCATED}


def run_main(graph, tmp_path, monkeypatch, rows):
    monkeypatch.setattr(cf, "CONCERNS_PATH", write(tmp_path, rows))
    monkeypatch.setattr(cf, "FIELD_PATH", tmp_path / "concerns" / "concern_field.csv")
    monkeypatch.setattr(cf, "load_graph", lambda: graph)
    cf.main()
    with (tmp_path / "concerns" / "concern_field.csv").open() as f:
        return list(csv.DictReader(f))


def test_main_uses_located_basis_when_every_record_is_located(graph, tmp_path, monkeypatch, capsys):
    rows = run_main(graph, tmp_path, monkeypatch, ["noise,55.949,-3.1995,,1,\n"])
    assert {r["basis"] for r in rows} == {cf.LOCATED} and float(rows[0]["noise"]) == pytest.approx(1.0)
    assert capsys.readouterr().out.startswith("Concern field from 1 records on located comments: 2 of 3 edges")


def test_main_falls_back_to_per_street_for_the_whole_dataset(graph, tmp_path, monkeypatch, capsys):
    rows = run_main(graph, tmp_path, monkeypatch, ["noise,55.949,-3.1995,,1,\n", "parking,,,Upper Bow,2,\n",
                                                   "parking,,,Castlehill,1,\n"])
    assert {r["basis"] for r in rows} == {cf.PER_STREET} and float(rows[2]["parking"]) == 2.0
    out = capsys.readouterr().out
    assert "1 located records have no street name" in out and "street not found in the graph: 'Castlehill'" in out
