"""The coverage headline and the slow-rows-near-stops check. Every row here is synthetic."""

import pandas as pd
import pytest

from model.coverage import LABEL, SLOW_KMH, coverage_summary, near_stop_edges, slow_rows
from speeds.edge_speeds import OUTPUT_COLUMNS
from tests.data.test_predict import T0, speed_row, study_graph

REPORT = {"results": {
    "street_split": {"mae_model": 6.7, "mae_baseline_1": 7.1, "mae_baseline_2": 7.24},
    "time_split": {"mae_model": 5.84, "mae_baseline_1": 6.9, "mae_baseline_2": 8.6},
}}


def predictions(kinds):
    """One edge, one row per kind, 15 minutes apart."""
    return pd.DataFrame({
        "edge_id": ["1"] * len(kinds),
        "street_name": ["A"] * len(kinds),
        "bin_start_utc": [T0 + pd.Timedelta(minutes=15 * i) for i in range(len(kinds))],
        "speed_kmh": [10.0] * len(kinds),
        "kind": kinds,
        "n_traversals": [1 if k == "observed" else 0 for k in kinds],
    })


# --- Coverage headline ---


def test_predicted_share_is_predicted_rows_over_all_rows():
    summary = coverage_summary(predictions(["observed", "predicted", "predicted", "predicted"]),
                               REPORT)
    assert summary["predicted_share"] == pytest.approx(0.75)
    assert summary["n_observed"] + summary["n_predicted"] == summary["n_rows"] == 4


def test_errors_come_from_the_evaluation_report():
    summary = coverage_summary(predictions(["observed", "predicted"]), REPORT)
    assert summary["error_unseen_streets_kmh"] == 6.7
    assert summary["error_later_bins_kmh"] == 5.84
    assert summary["baselines_unseen_streets_kmh"] == {"baseline_1": 7.1, "baseline_2": 7.24}
    assert summary["baselines_later_bins_kmh"] == {"baseline_1": 6.9, "baseline_2": 8.6}


def test_summary_states_its_label_counts_and_time_range():
    summary = coverage_summary(predictions(["observed", "predicted", "observed"]), REPORT)
    assert summary["label"] == LABEL == "bus-observed speed"
    assert summary["n_edges"] == 1 and summary["n_bins"] == 3
    assert summary["time_range_utc"] == [T0.isoformat(),
                                         (T0 + pd.Timedelta(minutes=30)).isoformat()]


def test_summary_carries_the_conformal_interval_from_the_report():
    report = {"results": {**REPORT["results"], "street_split_intervals": {
        "alpha": 0.1, "coverage": 0.88, "mean_width_kmh": 14.0, "final_q_kmh": 7.0, "n": 50}}}
    interval = coverage_summary(predictions(["observed", "predicted"]), report)["interval"]
    assert interval == {"target_coverage": 0.9, "measured_coverage_unseen_streets": 0.88,
                        "half_width_kmh": 7.0, "mean_width_kmh": 14.0}


def test_summary_interval_is_none_without_one_in_the_report():
    assert coverage_summary(predictions(["observed"]), REPORT)["interval"] is None


# --- Slow rows near stops ---


def test_an_edge_with_a_stop_by_its_end_is_near_a_stop():
    graph, ids = study_graph()
    stop_at_node_0 = [[55.95001, -3.19]]  # about 1 m from node 0
    near = near_stop_edges(graph, [ids["a1"], ids["b1"]], stop_at_node_0)
    assert near == {ids["a1"]}


def test_slow_rows_split_by_stops_and_count_rows_under_the_threshold():
    graph, ids = study_graph()
    speeds = pd.DataFrame([
        speed_row(ids["a1"], "Street A", 0, 1.0),
        speed_row(ids["a1"], "Street A", 15, 20.0),
        speed_row(ids["b1"], "Street B", 0, 30.0),
    ], columns=OUTPUT_COLUMNS)
    result = slow_rows(speeds, graph, [[55.95001, -3.19]])
    near, other = result["near_stop"], result["elsewhere"]
    assert result["slow_kmh"] == SLOW_KMH
    assert near["n_rows"] == 2 and near["n_slow"] == 1 and near["slow_share"] == 0.5
    assert near["median_kmh"] == pytest.approx(10.5)
    assert other["n_rows"] == 1 and other["n_slow"] == 0
    assert near["slowest"][0]["speed_kmh"] == pytest.approx(1.0)
    assert near["slowest"][0]["edge_id"] == ids["a1"]


def test_dwell_only_rows_are_not_counted():
    graph, ids = study_graph()
    speeds = pd.DataFrame([
        speed_row(ids["b1"], "Street B", 0, 30.0),
        speed_row(ids["b1"], "Street B", 15, 0.0, n=0),
    ], columns=OUTPUT_COLUMNS)
    result = slow_rows(speeds, graph, [])
    assert result["elsewhere"]["n_rows"] == 1
    assert result["near_stop"]["n_rows"] == 0
