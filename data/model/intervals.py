"""Prediction intervals for the speed model, by split conformal prediction (CV+ style).

The model predicts a median speed. An interval is that prediction plus or minus
q km/h, where q is a conformal quantile of absolute out-of-fold residuals.

Coverage is checked on the street split: fold k's q comes only from the
residuals of the other folds' held-out rows, then is scored on fold k's
held-out rows. Only rows from one source (observed, by default) are counted,
so synthetic rows never set or score an interval. The final q, from every
out-of-fold residual, is the one predictions should use.

CV+ promises coverage of at least 1 - 2 * alpha, not 1 - alpha. Rows on one
street share an error, so on held-out streets the measured coverage can sit a
little under the 1 - alpha target (about 0.89 for a 0.9 target on the test
table). Report it as measured, next to the target.
"""

import math

import numpy as np

INTERVAL_ALPHA = 0.1
INTERVAL_METHOD = (
    "split conformal, CV+ style: q is the conformal quantile of |truth - model| "
    "over out-of-fold street-split residuals; bounds are max(pred - q, 0) and pred + q"
)


def conformal_quantile(abs_residuals, alpha):
    """Return the conformal quantile of absolute residuals, or None if there are none.

    Returns the k-th smallest residual, k = ceil((n + 1)(1 - alpha)), or the
    largest when k > n.
    """
    residuals = np.asarray(abs_residuals, float)
    n = residuals.size
    if n == 0:
        return None
    k = min(n, math.ceil(round((n + 1) * (1 - alpha), 9)))
    # inverted_cdf at k / n is exactly the k-th smallest; "higher" would be one rank above.
    return float(np.quantile(residuals, k / n, method="inverted_cdf"))


def interval_bounds(pred, q):
    """Return (lower, upper) = (max(pred - q, 0), pred + q). Speeds are never negative,
    so only the lower bound is clipped."""
    if q is None:
        raise ValueError("no conformal quantile: there were no residuals to calibrate on")
    pred = np.asarray(pred, float)
    return np.maximum(pred - q, 0.0), pred + q


def _abs_residuals(fold, source):
    keep = np.asarray(fold["source"]) == source
    truth = np.asarray(fold["truth"], float)[keep]
    pred = np.asarray(fold["model"], float)[keep]
    return truth, pred, np.abs(truth - pred)


def _score(truth, pred, q):
    """Return (n covered, total width) of the q-intervals around pred."""
    lower, upper = interval_bounds(pred, q)
    covered = int(np.sum((lower <= truth) & (truth <= upper)))
    return covered, float(np.sum(upper - lower))


def street_split_coverage(scored, alpha=INTERVAL_ALPHA, source="observed"):
    """Return pooled and per-fold coverage and width of conformal intervals.

    scored is score_fold's output for each street fold. Every value that has no
    rows to stand on is None, never NaN, so the report stays valid JSON.
    """
    per_fold = [_abs_residuals(fold, source) for fold in scored]
    folds_out = []
    n_total, covered_total, width_total = 0, 0, 0.0
    for k, (truth, pred, _) in enumerate(per_fold):
        calibration = [r for j, (_, _, r) in enumerate(per_fold) if j != k]
        n_calibration = sum(len(r) for r in calibration)
        q = conformal_quantile(np.concatenate(calibration), alpha) if n_calibration else None
        n = len(truth)
        coverage = width = None
        if n and q is not None:
            covered, width_sum = _score(truth, pred, q)
            coverage, width = covered / n, width_sum / n
            n_total += n
            covered_total += covered
            width_total += width_sum
        folds_out.append({
            "coverage": coverage,
            "mean_width_kmh": width,
            "n": n,
            "q_kmh": q,
            "n_calibration": n_calibration,
        })
    all_residuals = [r for _, _, r in per_fold]
    return {
        "alpha": alpha,
        "source": source,
        "coverage": covered_total / n_total if n_total else None,
        "mean_width_kmh": width_total / n_total if n_total else None,
        "n": n_total,
        "final_q_kmh": conformal_quantile(
            np.concatenate(all_residuals) if all_residuals else [], alpha
        ),
        "folds": folds_out,
    }
