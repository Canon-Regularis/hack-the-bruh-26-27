"""What "correct" means for the baselines, model and validation. All data is synthetic."""

import json
import random

import pandas as pd
import pytest

from features.table import TARGET
from model.baselines import NearestObservedEdge, RoadClassHourMean
from model.evaluate import (
    LEAKAGE_TOLERANCE,
    build_report,
    evaluate,
    leakage_check,
    median_guess,
    score_fold,
)
from model.metrics import mae, skill
from model.splits import street_folds, street_groups, time_folds

T0 = pd.Timestamp("2026-11-07 18:00", tz="UTC")
MIDPOINTS = {"e1": (0.0, 0.0), "e2": (100.0, 0.0), "e3": (1000.0, 0.0), "e4": (50.0, 0.0)}


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


def table(rows):
    return pd.DataFrame(rows)


# --- Metrics ---


def test_mae_known_answer():
    assert mae([10.0, 20.0], [12.0, 17.0]) == pytest.approx(2.5)


def test_skill_known_answers():
    assert skill(2.0, 4.0) == pytest.approx(0.5)
    assert skill(3.0, 3.0) == 0.0
    assert skill(6.0, 3.0) == pytest.approx(-1.0)


# --- Baseline 1: mean by road class and hour ---


def test_baseline_1_is_the_training_mean_by_class_and_hour():
    train = table([row("e1", 0, 10.0), row("e2", 0, 20.0), row("e3", 0, 30.0, rank=4)])
    test = table([row("e4", 0, 99.0), row("e4", 0, 99.0, rank=4)])
    assert list(RoadClassHourMean().fit(train).predict(test)) == [15.0, 30.0]


def test_baseline_1_uses_the_training_mean_for_an_unseen_class_and_hour():
    train = table([row("e1", 0, 10.0), row("e2", 0, 30.0)])
    test = table([row("e3", 0, 99.0, rank=9), row("e3", 120, 99.0)])
    assert list(RoadClassHourMean().fit(train).predict(test)) == [20.0, 20.0]


def test_baseline_1_never_sees_test_targets():
    train = table([row("e1", 0, 10.0), row("e2", 0, 20.0)])
    test = table([row("e3", 0, 15.0)])
    first = RoadClassHourMean().fit(train).predict(test)
    test[TARGET] = 1000.0
    assert list(RoadClassHourMean().fit(train).predict(test)) == list(first)


# --- Baseline 2: nearest observed edge ---


def nearest(train, test, mode="same_bin"):
    fallback = RoadClassHourMean().fit(train)
    return NearestObservedEdge(MIDPOINTS, fallback).fit(train).predict(test, mode)


def test_baseline_2_copies_the_nearest_training_edge_in_the_same_bin():
    train = table([row("e2", 0, 25.0), row("e3", 0, 5.0)])
    predictions, fallbacks = nearest(train, table([row("e1", 0, 99.0)]))
    assert (list(predictions), fallbacks) == ([25.0], 0)


def test_baseline_2_skips_edges_on_the_same_street():
    train = table([row("e4", 0, 40.0, street="Shared"), row("e2", 0, 25.0)])
    predictions, _ = nearest(train, table([row("e1", 0, 99.0, street="Shared")]))
    assert list(predictions) == [25.0]


def test_baseline_2_ignores_other_bins_and_sources_and_falls_back():
    train = table([row("e2", 15, 25.0), row("e3", 0, 5.0, source="observed")])
    test = table([row("e1", 0, 99.0)])
    predictions, fallbacks = nearest(train, test)
    expected = RoadClassHourMean().fit(train).predict(test)
    assert (list(predictions), fallbacks) == (list(expected), 1)


def test_baseline_2_uses_the_latest_training_bin_in_time_mode():
    train = table([row("e2", 0, 10.0), row("e2", 15, 20.0), row("e3", 15, 5.0)])
    predictions, fallbacks = nearest(train, table([row("e1", 60, 99.0)]), mode="latest_bin")
    assert (list(predictions), fallbacks) == ([20.0], 0)


def test_baseline_2_never_sees_test_targets():
    train = table([row("e2", 0, 25.0)])
    test = table([row("e1", 0, 99.0), row("e3", 0, 1.0)])
    predictions, _ = nearest(train, test)
    assert list(predictions) == [25.0, 25.0]


# --- Splits ---


def split_table():
    rows = []
    for index, street in enumerate(["A", "A", "B", "C", "D", "E", "F", "", ""]):
        for minutes in range(0, 120, 15):
            rows.append(row(f"e{index}", minutes, 20.0, street=street))
    return table(rows)


def test_unnamed_edges_get_their_own_groups():
    groups = street_groups(split_table())
    assert "edge:e7" in groups and "edge:e8" in groups
    assert "" not in groups


def test_no_street_is_in_both_train_and_test():
    data = split_table()
    groups = street_groups(data)
    folds = list(street_folds(data, n_splits=5))
    assert len(folds) == 5
    for train_rows, test_rows in folds:
        assert not {groups[i] for i in train_rows} & {groups[i] for i in test_rows}


def test_time_split_always_tests_after_it_trains():
    data = split_table()
    folds = list(time_folds(data, n_splits=3))
    assert len(folds) == 3
    for train_rows, test_rows in folds:
        assert data["bin_start_utc"].iloc[train_rows].max() < data["bin_start_utc"].iloc[test_rows].min()


# --- Model and leakage check ---


def learnable_table(seed=0):
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
            rows.append(row(f"m{index}", minutes, speed, rank=rank, length_m=length,
                            lanes=1.0, weekday=5))
    return table(rows)


def test_no_method_in_a_fold_sees_that_folds_test_targets():
    data = learnable_table()
    train, test = data.iloc[:400], data.iloc[400:]
    first = score_fold(train, test, MIDPOINTS, "same_bin")
    poisoned = test.copy()
    poisoned[TARGET] = 1000.0
    second = score_fold(train, poisoned, MIDPOINTS, "same_bin")
    for method in ("model", "baseline_1", "baseline_2"):
        assert list(first[method]) == list(second[method])


def test_the_model_beats_baseline_1_when_there_is_signal():
    results = evaluate(learnable_table(), MIDPOINTS)
    assert results["street_split"]["skill_vs_baseline_1"] > 0.3


def test_the_median_guess_is_the_training_median_for_every_test_row():
    train = table([row("e1", 0, 10.0), row("e2", 0, 12.0), row("e3", 0, 50.0)])
    test = table([row("e4", 0, 1000.0), row("e4", 15, -5.0)])
    assert list(median_guess(train, test)) == [12.0, 12.0]


def test_a_model_on_shuffled_targets_does_not_beat_the_median_guess():
    check = leakage_check(learnable_table(), seeds=(1, 2))
    assert check["skill_vs_median_guess"] <= LEAKAGE_TOLERANCE
    assert check["leakage_suspected"] is False


def skewed_noise_table(seed=0):
    """No signal and a long tail of fast rows, like real bus-observed speeds."""
    rng = random.Random(seed)
    rows = []
    for index in range(40):
        rank = rng.choice([3, 4, 5, 7])
        length = rng.uniform(20, 400)
        for minutes in range(0, 240, 15):
            speed = rng.expovariate(1 / 10) + 2
            rows.append(row(f"m{index}", minutes, speed, rank=rank, length_m=length,
                            lanes=1.0, weekday=5))
    return table(rows)


def test_skewed_targets_alone_do_not_flag_a_leak():
    check = leakage_check(skewed_noise_table(), seeds=(0, 1))
    assert check["leakage_suspected"] is False


def fake_shuffled_skill(skills):
    """Stand in for shuffled_skill, returning the given skill for each shuffle in turn."""
    remaining = iter(skills)
    return lambda *args, **kwargs: next(remaining)


def test_leakage_check_averages_the_skill_over_every_seed(monkeypatch):
    monkeypatch.setattr("model.evaluate.shuffled_skill", fake_shuffled_skill([0.03, 0.0, 0.0]))
    check = leakage_check(learnable_table(), seeds=(0, 1, 2))
    assert check["seeds"] == [0, 1, 2]
    assert check["skill_per_seed"] == [0.03, 0.0, 0.0]
    assert check["skill_vs_median_guess"] == pytest.approx(0.01)


def test_one_lucky_shuffle_does_not_flag_a_leak(monkeypatch):
    monkeypatch.setattr("model.evaluate.shuffled_skill",
                        fake_shuffled_skill([0.05, 0.0, 0.0, 0.0, 0.0]))
    check = leakage_check(learnable_table(), seeds=(0, 1, 2, 3, 4))
    assert check["leakage_suspected"] is False


def test_shuffles_that_beat_the_median_guess_on_average_flag_a_leak(monkeypatch):
    monkeypatch.setattr("model.evaluate.shuffled_skill", fake_shuffled_skill([0.03, 0.03, 0.03]))
    check = leakage_check(learnable_table(), seeds=(0, 1, 2))
    assert check["leakage_suspected"] is True


# --- Report ---


def test_the_report_has_config_time_range_row_counts_and_every_metric():
    data = learnable_table()
    report = build_report(data, evaluate(data, MIDPOINTS), {"speeds_file": "synthetic"})
    assert report["data"]["source"] == ["synthetic"]
    assert report["data"]["time_range_utc"][0] < report["data"]["time_range_utc"][1]
    for split in ("street_split", "time_split"):
        result = report["results"][split]
        assert sum(fold["n_test"] for fold in result["folds"]) > 0
        for key in ("mae_model", "mae_baseline_1", "mae_baseline_2",
                    "skill_vs_baseline_1", "skill_vs_baseline_2", "beats_baseline_2"):
            assert key in result
    assert "leakage_check" in report["results"]
    assert "lightgbm" in report["config"]["library"]
    json.dumps(report, allow_nan=False)
