"""Score the model against both baselines on held-out streets and held-out hours.

Every method is fitted on each fold's training rows only, and scored on the
same test rows. The leakage check retrains on shuffled targets several times:
if that beats a constant guess of the training median on average, something
is leaking. The median, not baseline 1, is the reference because an L1 model
on shuffled targets learns the median, which beats a mean on skewed speeds
without any leak.

The street split also checks conformal prediction intervals (model.intervals)
on its held-out observed rows. The leakage check does not.
"""

import platform

import lightgbm
import numpy as np

from features.lag import LAG_BINS
from features.table import TARGET
from model.baselines import NearestObservedEdge, RoadClassHourMean
from model.intervals import INTERVAL_ALPHA, INTERVAL_METHOD, street_split_coverage
from model.metrics import mae, skill
from model.speed_model import MODEL_PARAMS, fit_predict
from model.splits import street_folds, time_folds

LEAKAGE_TOLERANCE = 0.02
LEAKAGE_SEEDS = (0, 1, 2, 3, 4)
N_STREET_SPLITS = 5
N_TIME_SPLITS = 3


def score_fold(train, test, midpoints, mode):
    """Fit all three methods on train, predict test, and return the predictions."""
    baseline_1 = RoadClassHourMean().fit(train)
    baseline_2 = NearestObservedEdge(midpoints, baseline_1).fit(train)
    predicted_2, fallbacks = baseline_2.predict(test, mode)
    return {
        "truth": test[TARGET].to_numpy(),
        "source": test["source"].to_numpy(),
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


def score_split(table, folds, midpoints, mode):
    """Return score_fold's output for every fold."""
    return [
        score_fold(table.iloc[train_rows], table.iloc[test_rows], midpoints, mode)
        for train_rows, test_rows in folds
    ]


def summarise_split(scored):
    """Pool the scored folds' test rows for the split's headline numbers."""
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


def run_split(table, folds, midpoints, mode):
    """Score every fold, then pool the test rows for the split's headline numbers."""
    return summarise_split(score_split(table, folds, midpoints, mode))


def evaluate(table, midpoints, n_street_splits=N_STREET_SPLITS, n_time_splits=N_TIME_SPLITS):
    """Return results for the street split, its intervals, the time split and the leakage check."""
    table = table.reset_index(drop=True)
    street_scored = score_split(table, street_folds(table, n_street_splits), midpoints, "same_bin")
    return {
        "street_split": summarise_split(street_scored),
        "street_split_intervals": street_split_coverage(street_scored, INTERVAL_ALPHA),
        "time_split": run_split(
            table, time_folds(table, n_time_splits), midpoints, "latest_bin"
        ),
        "leakage_check": leakage_check(table, n_splits=n_street_splits),
    }


def median_guess(train, test):
    """Predict the training rows' median speed for every test row."""
    return np.full(len(test), float(train[TARGET].median()))


def shuffled_skill(table, seed, n_splits):
    """Skill vs the median guess of a model retrained on targets shuffled with this seed."""
    shuffled = table.reset_index(drop=True).copy()
    shuffled[TARGET] = np.random.default_rng(seed).permutation(shuffled[TARGET].to_numpy())
    truth, model, guess = [], [], []
    for train_rows, test_rows in street_folds(shuffled, n_splits):
        train, test = shuffled.iloc[train_rows], shuffled.iloc[test_rows]
        truth.append(test[TARGET].to_numpy())
        model.append(fit_predict(train, test))
        guess.append(median_guess(train, test))
    truth = np.concatenate(truth)
    return skill(mae(truth, np.concatenate(model)), mae(truth, np.concatenate(guess)))


def leakage_check(table, seeds=LEAKAGE_SEEDS, n_splits=N_STREET_SPLITS):
    """Retrain on shuffled targets, once per seed. Beating the median guess on average means a leak.

    Averaging over several shuffles stops one lucky shuffle from flagging a leak.
    """
    skills = [shuffled_skill(table, seed, n_splits) for seed in seeds]
    gain = float(np.mean(skills))
    return {
        "seeds": list(seeds),
        "skill_per_seed": skills,
        "skill_vs_median_guess": gain,
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
            "leakage_seeds": list(LEAKAGE_SEEDS),
            "interval_alpha": INTERVAL_ALPHA,
            "interval_method": INTERVAL_METHOD,
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
