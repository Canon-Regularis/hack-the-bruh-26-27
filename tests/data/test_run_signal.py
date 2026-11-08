"""The one-command night run, wired end to end on a tiny synthetic graph.

The evaluation is faked: its folds need more streets than the tiny graph has,
and it has its own tests in test_speed_model.py.
"""

import json

import pandas as pd

import scripts.run_signal as run_signal
from scripts.evaluate_speed_model import read_speeds
from tests.data.test_coverage import REPORT
from tests.data.test_predict import BBOX, speeds_for, study_graph


def fake_evaluation(leaky=False):
    def run_evaluation(speeds, graph, betweenness, flags, aadf, concerns, speeds_file):
        results = {split: {**scores, "skill_vs_baseline_2": 0.1}
                   for split, scores in REPORT["results"].items()}
        return {
            "data": {"source": ["observed"], "n_rows": len(speeds)},
            "results": {**results, "leakage_check": {"leakage_suspected": leaky},
                        "street_split_intervals": {
                            "alpha": 0.1, "source": "observed", "coverage": 0.88, "n": 40,
                            "mean_width_kmh": 14.0, "final_q_kmh": 7.0}},
        }
    return run_evaluation


def windows(tmp_path, ids):
    """The night split in two files, the second matched again with one changed row."""
    night = speeds_for(ids)
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    night.to_csv(first, index=False)
    rerun = night.iloc[:1].copy()
    rerun["mean_bus_observed_speed_kmh"] = 33.0
    rerun.to_csv(second, index=False)
    return [first, second], len(night)


def run(tmp_path, monkeypatch, ue, leaky=False):
    monkeypatch.setattr(run_signal, "run_evaluation", fake_evaluation(leaky))
    graph, ids = study_graph()
    paths, n_rows = windows(tmp_path, ids)
    outputs = run_signal.run(paths, (graph, {}, None, None, None), [], ue, BBOX,
                             derived=tmp_path / "derived")
    return outputs, n_rows


def ue_for():
    graph, ids = study_graph()
    table = pd.DataFrame([{"edge_id": e, "ue_kmh": 15.0, "free_flow_kmh": 30.0}
                          for e in ids.values()])
    return table, (8, 9)


def test_every_output_is_written(tmp_path, monkeypatch):
    outputs, n_rows = run(tmp_path, monkeypatch, ue_for())
    assert set(outputs) == {"speeds", "report", "predictions", "summary", "assignment"}
    assert all(path.exists() for path in outputs.values())
    joined = read_speeds(outputs["speeds"])
    assert len(joined) == n_rows
    assert 33.0 in set(joined["mean_bus_observed_speed_kmh"])


def test_the_summary_carries_coverage_on_top_and_checks_below(tmp_path, monkeypatch):
    outputs, _ = run(tmp_path, monkeypatch, ue_for())
    summary = json.loads(outputs["summary"].read_text())
    assert 0 < summary["predicted_share"] < 1
    assert summary["error_unseen_streets_kmh"] == 6.7
    assert summary["interval"]["half_width_kmh"] == 7.0
    assert set(summary["checks"]["slow_rows"]) >= {"near_stop", "elsewhere"}


def test_no_equilibrium_file_skips_the_assignment_check(tmp_path, monkeypatch):
    outputs, _ = run(tmp_path, monkeypatch, None)
    assert "assignment" not in outputs


def test_a_leaky_evaluation_stops_before_predicting(tmp_path, monkeypatch):
    outputs, _ = run(tmp_path, monkeypatch, ue_for(), leaky=True)
    assert outputs is None
    assert not (tmp_path / "derived" / "predictions").exists()
