"""The speed model: LightGBM gradient boosting with fixed settings.

The settings are chosen in advance and never tuned on test folds. The objective
is absolute error, the same as the reported MAE.
"""

from lightgbm import LGBMRegressor

from features.table import TARGET, feature_columns

MODEL_PARAMS = {
    "objective": "l1",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 0,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
}


def make_model():
    return LGBMRegressor(**MODEL_PARAMS)


def fit_predict(train, test):
    """Fit on the training rows' features and return predictions for the test rows."""
    columns = feature_columns(train)
    model = make_model().fit(train[columns], train[TARGET])
    return model.predict(test[columns])
