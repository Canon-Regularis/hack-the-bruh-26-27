"""Engine's modelled link speeds against bus-observed speeds. Every row is synthetic."""

import json

import pandas as pd
import pytest

from model.assignment_check import (
    CAVEAT,
    HOUR_MISMATCH,
    assignment_check,
    compare,
    edge_speeds_in_hours,
    parse_hours,
    read_ue,
)
from speeds.edge_speeds import OUTPUT_COLUMNS
from tests.data.test_predict import speed_row

HOURS = (8, 9)


def at(edge, hour, minute, kmh):
    """A speed row at a clock time on 2026-11-08, whatever T0 is."""
    row = speed_row(edge, f"Street {edge}", 0, kmh)
    row["bin_start_utc"] = pd.Timestamp(f"2026-11-08 {hour:02d}:{minute:02d}", tz="UTC")
    return row


def speeds(*rows):
    return pd.DataFrame(list(rows), columns=OUTPUT_COLUMNS)


def ue_table(rows):
    """rows: (edge_id, ue_kmh, free_flow_kmh)."""
    return pd.DataFrame([{"edge_id": e, "ue_kmh": u, "free_flow_kmh": f} for e, u, f in rows])


def test_parse_hours_reads_the_equilibrium_label():
    assert parse_hours("08:00-09:00 weekday") == (8, 9)


def test_read_ue_turns_length_and_time_into_kmh(tmp_path):
    csv_path, meta_path = tmp_path / "ue.csv", tmp_path / "ue.json"
    pd.DataFrame([{"edge_id": 7, "length_m": 100.0, "free_flow_time_s": 10.0,
                   "ue_time_s": 20.0, "flow_veh_h": 5.0, "vc": 0.1}]).to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps({"hour": "08:00-09:00 weekday"}))
    ue, hours = read_ue(csv_path, meta_path)
    assert hours == (8, 9)
    assert ue.loc[0, "edge_id"] == "7"
    assert ue.loc[0, "ue_kmh"] == pytest.approx(18.0)
    assert ue.loc[0, "free_flow_kmh"] == pytest.approx(36.0)


def test_only_bins_in_the_hour_are_used_when_they_exist():
    table, matched = edge_speeds_in_hours(
        speeds(at("1", 8, 15, 20.0), at("1", 22, 0, 40.0)), HOURS)
    assert matched is True
    assert table.set_index("edge_id").loc["1", "observed_kmh"] == pytest.approx(20.0)


def test_every_bin_is_used_when_none_is_in_the_hour():
    table, matched = edge_speeds_in_hours(
        speeds(at("1", 22, 0, 20.0), at("1", 23, 0, 40.0)), HOURS)
    assert matched is False
    # total metres over total seconds: equal seconds, so the plain mean
    assert table.set_index("edge_id").loc["1", "observed_kmh"] == pytest.approx(30.0)
    assert table.set_index("edge_id").loc["1", "n_traversals"] == 2


def test_dwell_only_rows_are_ignored():
    table, _ = edge_speeds_in_hours(speeds(at("1", 22, 0, 20.0), {
        **at("2", 22, 0, 0.0), "n_traversals": 0, "total_metres": 0.0, "total_seconds": 0.0,
        "mean_bus_observed_speed_kmh": float("nan")}), HOURS)
    assert list(table["edge_id"]) == ["1"]


def test_compare_gives_known_mae_and_bias_as_modelled_minus_observed():
    observed = pd.DataFrame({"edge_id": ["1", "2"], "observed_kmh": [10.0, 20.0]})
    result = compare(observed, ue_table([("1", 14.0, 30.0), ("2", 18.0, 30.0)]), "ue_kmh")
    assert result["n_edges"] == 2
    assert result["mae_kmh"] == pytest.approx(3.0)
    assert result["bias_kmh"] == pytest.approx(1.0)


def test_edges_missing_from_the_equilibrium_are_dropped():
    observed = pd.DataFrame({"edge_id": ["1", "9"], "observed_kmh": [10.0, 99.0]})
    result = compare(observed, ue_table([("1", 12.0, 30.0)]), "ue_kmh")
    assert result["n_edges"] == 1
    assert result["mae_kmh"] == pytest.approx(2.0)


def test_observed_and_predicted_edges_are_reported_apart():
    observed = speeds(at("1", 22, 0, 10.0))
    predictions = pd.DataFrame({
        "edge_id": ["1", "2", "2"],
        "street_name": ["Street 1", "Street 2", "Street 2"],
        "bin_start_utc": pd.to_datetime(["2026-11-08 22:00"] * 3, utc=True),
        "speed_kmh": [10.0, 16.0, 20.0],
        "kind": ["observed", "predicted", "predicted"],
        "n_traversals": [1, 0, 0],
    })
    ue = ue_table([("1", 12.0, 30.0), ("2", 15.0, 30.0)])
    result = assignment_check(observed, predictions, ue, HOURS)
    assert result["hour_matched"] is False
    assert result["equilibrium_hours_utc"] == [8, 9]
    assert result["observed"]["vs_ue"] == {"n_edges": 1, "mae_kmh": 2.0, "bias_kmh": 2.0}
    assert result["observed"]["vs_free_flow"]["bias_kmh"] == pytest.approx(20.0)
    assert result["predicted"]["vs_ue"]["n_edges"] == 1
    assert result["predicted"]["vs_ue"]["bias_kmh"] == pytest.approx(-3.0)
    assert CAVEAT in result["caveats"] and HOUR_MISMATCH in result["caveats"]
