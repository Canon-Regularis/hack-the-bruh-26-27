"""Scores for the speed model, in km/h of bus-observed speed."""

import numpy as np


def mae(y_true, y_pred):
    """Return the mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def skill(mae_model, mae_baseline):
    """Return 1 - MAE / MAE_baseline: above 0 beats the baseline, 0 ties it."""
    return 1.0 - mae_model / mae_baseline
