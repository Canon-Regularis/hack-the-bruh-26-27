"""Score the model against both baselines on held-out streets and held-out hours.

Every method is fitted on each fold's training rows only, and scored on the
same test rows. The leakage check retrains on shuffled targets: if that still
beats baseline 1, something is leaking.
"""

import platform

import lightgbm
import numpy as np

from features.lag import LAG_BINS
from features.table import TARGET
from model.baselines import NearestObservedEdge, RoadClassHourMean
from model.metrics import mae, skill
from model.speed_model import MODEL_PARAMS, fit_predict
from model.splits import street_folds, time_folds

LEAKAGE_TOLERANCE = 0.02
N_STREET_SPLITS = 5
N_TIME_SPLITS = 3


def score_fold(train, test, midpoints, mode):
    """Fit all three methods on train, predict test, and return the predictions."""
    baseline_1 = RoadClassHourMean().fit(train)
    baseline_2 = NearestObservedEdge(midpoints, baseline_1).fit(train)
    predicted_2, fallbacks = baseline_2.predict(test, mode)
    return {
        "truth": test[TARGET].to_numpy(),
        "model": fit_predict(train, test),
        "baseline_1": baseline_1.predict(test),
        "baseline_2": predicted_2,
        "n_train": len(train),
        "n_test": len(test),
        "baseline_2_fallbacks": fallbacks,
    }


def summarise(scored):
    """Return MAE and skill for each method over the given folds' test rows."""
    truth = np.concatenate([s["truth"] for s in scored])
    errors = {
        f"mae_{name}": mae(truth, np.concatenate([s[name] for s in scored]))
        for name in ("model", "baseline_1", "baseline_2")
    }
    return {
        **errors,
        "skill_vs_baseline_1": skill(errors["mae_model"], errors["mae_baseline_1"]),
        "skill_vs_baseline_2": skill(errors["mae_model"], errors["mae_baseline_2"]),
    }


def run_split(table, folds, midpoints, mode):
    """Score every fold, then pool the test rows for the split's headline numbers."""
    scored = [
        score_fold(table.iloc[train_rows], table.iloc[test_rows], midpoints, mode)
        for train_rows, test_rows in folds
    ]
    folds_out = [
        {
            "n_train": s["n_train"],
            "n_test": s["n_test"],
            "baseline_2_fallbacks": s["baseline_2_fallbacks"],
            **summarise([s]),
        }
        for s in scored
    ]
    pooled = summarise(scored)
    return {**pooled, "beats_baseline_2": pooled["skill_vs_baseline_2"] > 0, "folds": folds_out}


def evaluate(table, midpoints, n_street_splits=N_STREET_SPLITS, n_time_splits=N_TIME_SPLITS):
    """Return results for the street split, the time split and the leakage check."""
    table = table.reset_index(drop=True)
    return {
        "street_split": run_split(
            table, street_folds(table, n_street_splits), midpoints, "same_bin"
        ),
        "time_split": run_split(
            table, time_folds(table, n_time_splits), midpoints, "latest_bin"
        ),
        "leakage_check": leakage_check(table, midpoints, n_splits=n_street_splits),
    }


def leakage_check(table, midpoints, seed=0, n_splits=N_STREET_SPLITS):
    """Retrain on shuffled targets. Beating baseline 1 then would mean a leak."""
    shuffled = table.reset_index(drop=True).copy()
    shuffled[TARGET] = np.random.default_rng(seed).permutation(shuffled[TARGET].to_numpy())
    result = run_split(shuffled, street_folds(shuffled, n_splits), midpoints, "same_bin")
    gain = result["skill_vs_baseline_1"]
    return {
        "seed": seed,
        "skill_vs_baseline_1": gain,
        "tolerance": LEAKAGE_TOLERANCE,
        "leakage_suspected": bool(gain > LEAKAGE_TOLERANCE),
    }


def build_report(table, results, extra_config):
    """Return the JSON-ready report: config, data summary and every metric."""
    times = table["bin_start_utc"]
    return {
        "config": {
            "library": f"lightgbm {lightgbm.__version__}",
            "python": platform.python_version(),
            "model_params": MODEL_PARAMS,
            "n_street_splits": N_STREET_SPLITS,
            "n_time_splits": N_TIME_SPLITS,
            "lag_bins": LAG_BINS,
            "leakage_tolerance": LEAKAGE_TOLERANCE,
            **extra_config,
        },
        "data": {
            "source": sorted(table["source"].unique()),
            "time_range_utc": [times.min().isoformat(), times.max().isoformat()],
            "n_rows": len(table),
            "n_edges": int(table["edge_id"].nunique()),
            "n_streets": int(table["street_name"].replace("", np.nan).nunique()),
        },
        "results": results,
    }
