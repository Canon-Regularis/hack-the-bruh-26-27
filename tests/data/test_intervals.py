"""What "correct" means for the conformal prediction intervals. All data is synthetic."""

import json
import random
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from features.table import TARGET
from model.evaluate import build_report, evaluate, leakage_check, run_split, score_split
from model.intervals import conformal_quantile, interval_bounds, street_split_coverage
from model.splits import street_folds, time_folds

T0 = pd.Timestamp("2026-11-07 18:00", tz="UTC")
MIDPOINTS = {}
ALPHA = 0.1


def row(edge, minutes, target, rank=3, street=None, source="synthetic", **extra):
    when = T0 + pd.Timedelta(minutes=minutes)
    return {
        "edge_id": edge,
        "bin_start_utc": when,
        "street_name": f"Street {edge}" if street is None else street,
        "source": source,
        "road_class_rank": rank,
        "hour": when.hour,
        **extra,
        TARGET: target,
    }


def learnable_table(seed=0, source="synthetic"):
    """Speed depends on length and road class, so a model can learn it."""
    rng = random.Random(seed)
    rows = []
    for index in range(40):
        rank = rng.choice([3, 4, 5, 7])
        length = rng.uniform(20, 400)
        midpoint = (rng.uniform(0, 5000), rng.uniform(0, 5000))
        MIDPOINTS[f"m{index}"] = midpoint
        for minutes in range(0, 240, 15):
            speed = 5 + length / 20 - rank + rng.gauss(0, 1)
            rows.append(row(f"m{index}", minutes, speed, rank=rank, source=source,
                            length_m=length, lanes=1.0, weekday=5))
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def observed_scored():
    """score_fold output for each street fold of an all-observed learnable table."""
    data = learnable_table(source="observed")
    return score_split(data, street_folds(data, 5), MIDPOINTS, "same_bin")


@lru_cache(maxsize=None)
def observed_results():
    return evaluate(learnable_table(source="observed"), MIDPOINTS)


def copy_scored(scored):
    return [{key: np.array(value) if isinstance(value, np.ndarray) else value
             for key, value in fold.items()} for fold in scored]


# --- The conformal quantile ---


def test_conformal_quantile_known_answers():
    residuals = [7.0, 1.0, 10.0, 4.0, 2.0, 9.0, 3.0, 6.0, 8.0, 5.0]  # 1..10, shuffled
    # alpha 0.1, n 10: ceil(11 * 0.9) / 10 = 1.0, the largest residual.
    assert conformal_quantile(residuals, 0.1) == 10.0
    # alpha 0.5: k = ceil(11 * 0.5) = 6, the sixth smallest.
    assert conformal_quantile(residuals, 0.5) == 6.0
    # alpha 0.1, n 20: k = ceil(21 * 0.9) = 19, the nineteenth smallest.
    assert conformal_quantile(list(range(20, 0, -1)), 0.1) == 19.0
    # k = 1: the smallest.
    assert conformal_quantile(residuals, 0.01) == 10.0
    assert conformal_quantile(list(range(1, 100)), 0.99) == 1.0
    # Too few residuals for the level: capped at 1, the largest.
    assert conformal_quantile([3.0], 0.1) == 3.0
    assert conformal_quantile([], 0.1) is None


# --- Coverage on held-out streets ---


def test_coverage_holds_on_a_learnable_observed_table():
    # CV+ guarantees 1 - 2 * alpha, not 1 - alpha. Here the held-out coverage is
    # 0.873: rows on one street share an error, so the 40 streets, not the 640
    # rows, are the exchangeable units, and a fold of hard streets is left out
    # of its own q.
    intervals = observed_results()["street_split_intervals"]
    assert intervals["alpha"] == ALPHA
    assert intervals["source"] == "observed"
    assert intervals["n"] == 640
    assert intervals["coverage"] >= 1 - 2 * ALPHA
    assert intervals["mean_width_kmh"] > 0
    assert intervals["final_q_kmh"] > 0
    folds = intervals["folds"]
    assert sum(fold["n"] for fold in folds) == 640
    n_weighted = lambda key: sum(fold[key] * fold["n"] for fold in folds) / 640  # noqa: E731
    assert intervals["coverage"] == pytest.approx(n_weighted("coverage"))
    assert intervals["mean_width_kmh"] == pytest.approx(n_weighted("mean_width_kmh"))
    everything = np.concatenate([np.abs(f["truth"] - f["model"]) for f in observed_scored()])
    assert intervals["final_q_kmh"] == conformal_quantile(everything, ALPHA)
    for fold in intervals["folds"]:
        assert fold["q_kmh"] > 0
        assert fold["n_calibration"] == 640 - fold["n"]


def test_only_observed_rows_set_q_or_coverage():
    scored = copy_scored(observed_scored())
    before = street_split_coverage(scored, ALPHA)
    for fold in scored:
        extra = 50
        fold["truth"] = np.concatenate([fold["truth"], np.full(extra, 1000.0)])
        fold["model"] = np.concatenate([fold["model"], np.zeros(extra)])
        fold["source"] = np.concatenate([fold["source"], np.full(extra, "synthetic")])
    assert street_split_coverage(scored, ALPHA) == before


def test_poisoning_fold_k_held_out_targets_does_not_change_q_k():
    k = 2
    scored = copy_scored(observed_scored())
    before = street_split_coverage(scored, ALPHA)
    scored[k]["truth"] = np.full(len(scored[k]["truth"]), 1000.0)
    after = street_split_coverage(scored, ALPHA)
    assert after["folds"][k]["q_kmh"] == before["folds"][k]["q_kmh"]
    assert after["folds"][k]["coverage"] == 0.0
    # The other folds do calibrate on fold k, so their q moves.
    for j in range(len(scored)):
        if j != k:
            assert after["folds"][j]["q_kmh"] > before["folds"][j]["q_kmh"]
    assert after["final_q_kmh"] > before["final_q_kmh"]


def test_a_report_with_no_observed_rows_is_valid_json_with_none():
    data = learnable_table()  # all synthetic
    report = build_report(data, evaluate(data, MIDPOINTS), {"speeds_file": "synthetic"})
    intervals = report["results"]["street_split_intervals"]
    assert intervals["n"] == 0
    assert intervals["coverage"] is None
    assert intervals["mean_width_kmh"] is None
    assert intervals["final_q_kmh"] is None
    assert all(fold["q_kmh"] is None and fold["coverage"] is None for fold in intervals["folds"])
    assert report["config"]["interval_alpha"] == ALPHA
    assert "conformal" in report["config"]["interval_method"]
    json.dumps(report, allow_nan=False)


# --- Known answers on a hand-built scored list ---


def hand_fold(residuals):
    """A fold whose model says 10 everywhere and whose truth is off by the residuals."""
    truth = np.array([10.0 + r for r in residuals])
    return {"truth": truth, "model": np.full(len(truth), 10.0),
            "source": np.full(len(truth), "observed")}


def test_coverage_width_and_q_on_hand_built_folds():
    # alpha 0.5; each fold's q is the ceil(9 * 0.5) = 5th smallest of the other 8 residuals.
    folds = [hand_fold([1, -2, 3, -4]), hand_fold([5, 6, -7, 8]), hand_fold([2, -4, 6, 10])]
    result = street_split_coverage(folds, 0.5)
    assert [f["q_kmh"] for f in result["folds"]] == [6.0, 4.0, 5.0]
    assert [f["coverage"] for f in result["folds"]] == [1.0, 0.0, 0.5]
    assert [f["mean_width_kmh"] for f in result["folds"]] == [12.0, 8.0, 10.0]
    assert [f["n"] for f in result["folds"]] == [4, 4, 4]
    assert [f["n_calibration"] for f in result["folds"]] == [8, 8, 8]
    assert result["coverage"] == 0.5
    assert result["mean_width_kmh"] == 10.0
    assert result["n"] == 12
    # All 12 residuals, k = ceil(13 * 0.5) = 7, the seventh smallest.
    assert result["final_q_kmh"] == 5.0
    assert result["alpha"] == 0.5
    assert result["source"] == "observed"


def test_a_residual_equal_to_q_is_covered():
    result = street_split_coverage([hand_fold([3.0]), hand_fold([-3.0])], 0.1)
    assert [f["q_kmh"] for f in result["folds"]] == [3.0, 3.0]
    assert result["coverage"] == 1.0


def test_the_cli_line_does_not_claim_the_target_as_the_result():
    from scripts.evaluate_speed_model import interval_line

    intervals = street_split_coverage([hand_fold([1, -2, 3, -4]), hand_fold([5, 6, -7, 8])], 0.1)
    line = interval_line(intervals)
    assert line.startswith("intervals (target 90%): measured coverage ")
    assert "n=8" in line
    none = interval_line(street_split_coverage([], 0.1))
    assert "n/a" in none and "None" not in none and "nan" not in none


# --- Bounds ---


def test_bounds_are_never_negative():
    lower, upper = interval_bounds([0.5, 3.0, 10.0, -5.0], 2.0)
    assert list(lower) == [0.0, 1.0, 8.0, 0.0]
    # Only the lower bound is clipped; upper is pred + q, even for a negative prediction.
    assert list(upper) == [2.5, 5.0, 12.0, -3.0]
    q = observed_results()["street_split_intervals"]["final_q_kmh"]
    for fold in observed_scored():
        lower, upper = interval_bounds(fold["model"], q)
        assert (lower >= 0).all() and (upper >= lower).all()
        assert np.allclose(upper, fold["model"] + q)


def test_bounds_need_a_quantile():
    with pytest.raises(ValueError):
        interval_bounds([10.0], None)


# --- evaluate() is otherwise unchanged ---


def test_evaluate_keeps_every_existing_key_and_value():
    data = learnable_table(source="observed")
    results = observed_results()
    assert list(results) == ["street_split", "street_split_intervals", "time_split",
                             "leakage_check"]
    assert results["street_split"] == run_split(data, street_folds(data, 5), MIDPOINTS,
                                                 "same_bin")
    assert results["time_split"] == run_split(data, time_folds(data, 3), MIDPOINTS,
                                              "latest_bin")
    assert results["leakage_check"] == leakage_check(data, n_splits=5)
