"""Collected pings to the edge-speed file the model reads. Matching is faked here."""

import csv

import pytest

from collector import CSV_FIELDS
from matching.table import MatchingNotAvailable
from scripts.build_edge_speeds import build, main, read_pings
from scripts.evaluate_speed_model import read_speeds
from speeds.edge_speeds import OUTPUT_COLUMNS
from speeds.synthetic import TRUE_SPEED_KMH, synthetic_bus


def fake_match(pings):
    pairs, _, _ = synthetic_bus()
    return pairs


@pytest.fixture
def pings_csv(tmp_path):
    path = tmp_path / "pings.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow({"polled_at": "2026-11-07T18:00:30+00:00", "vehicle_id": "-96",
                         "vehicle_timestamp": "2026-11-07T18:00:00+00:00",
                         "lat": "55.95", "lon": "-3.19"})
    return path


def test_read_pings_keeps_every_collector_column(pings_csv):
    rows = read_pings(pings_csv)
    assert len(rows) == 1
    assert set(rows[0]) == set(CSV_FIELDS)


def test_build_turns_matched_pairs_into_the_speed_table():
    _, stops, names = synthetic_bus()
    table = build([], stops, names, match=fake_match)
    assert list(table.columns) == OUTPUT_COLUMNS
    assert table["mean_bus_observed_speed_kmh"].dropna().median() == pytest.approx(TRUE_SPEED_KMH)


def test_the_saved_file_reads_back_for_the_model(tmp_path):
    _, stops, names = synthetic_bus()
    table = build([], stops, names, match=fake_match)
    path = tmp_path / "edge_speeds.csv"
    table.to_csv(path, index=False)
    back = read_speeds(path)
    assert list(back["edge_id"]) == list(table["edge_id"])
    assert back["bin_start_utc"].dt.tz is not None
    assert list(back["bin_start_utc"]) == list(table["bin_start_utc"])


def test_main_stops_with_a_clear_message_until_matching_is_built(pings_csv, tmp_path, capsys,
                                                                   monkeypatch):
    import scripts.build_edge_speeds as module

    def not_built(pings):
        raise MatchingNotAvailable("match_pings needs the map matching code")

    monkeypatch.setattr(module, "match_pings", not_built)
    monkeypatch.setattr(module, "load_hand_off", lambda: ([], {}))
    with pytest.raises(SystemExit) as stopped:
        main(["--pings", str(pings_csv), "--out", str(tmp_path / "out.csv")])
    assert stopped.value.code == 2
    assert "map matching" in capsys.readouterr().out
    assert not (tmp_path / "out.csv").exists()
