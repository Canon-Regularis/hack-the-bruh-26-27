# Ensembles

Pure-Python, zero-dependency ensemble-learning toolkit — a *generic*
model-combination framework that wraps **any** base estimator (classifier or
regressor) with a uniform interface, in a single file with no NumPy / pandas /
scikit-learn. Out-of-fold stacking, holdout blending, bagging / random-patches,
voting, generic AdaBoost (SAMME + AdaBoost.R2), pluggable-loss gradient
boosting, Caruana greedy ensemble selection, and loss-optimised weighted
averaging — plus the base learners and metrics to run it all standalone.

## Estimator contract

A base estimator is anything with:

```python
est.fit(X, y) -> self      # X: List[List[float]],  y: labels or targets
est.predict(X) -> list
est.predict_proba(X)       # optional (classifiers); enables soft voting / stacking
est.classes_               # optional (classifiers)
```

Pass an **unfitted prototype** — ensembles clone it (`copy.deepcopy`) for each
member, so the prototype is never mutated and members are independent. You can
plug in the learners bundled here, or your own.

Every ensemble itself exposes `fit` / `predict` / `score` (mean accuracy for
classifiers, R² for regressors), so it drops into the same workflows as its
bases. Heterogeneous estimators are passed as `[(name, estimator), ...]` pairs.

## Pick

| Goal | Use |
|---|---|
| Reduce variance of a high-variance model | `BaggingRegressor` / `BaggingClassifier` |
| Feature/sample subsampling (random patches) | Bagging with `max_features` / `bootstrap_features` |
| Blend several different models | `VotingRegressor` / `VotingClassifier` |
| Learn *how* to combine models | `StackingRegressor` / `StackingClassifier` (out-of-fold) |
| Faster stacking on big data | `BlendingRegressor` / `BlendingClassifier` (holdout) |
| Sequentially focus on hard cases | `AdaBoostRegressor` (R2) / `AdaBoostClassifier` (SAMME) |
| Strong tabular regressor | `GradientBoostingRegressor` (ls / lad / huber) |
| Auto-pick the best subset from a model zoo | `EnsembleSelection` (Caruana) |
| Optimal convex blend of regressors | `WeightedAverageRegressor` |

## Install

```bash
# No runtime deps — copy the file.
pip install pytest  # tests only
```

## Quick start

```python
from ensembles_template import (
    make_blobs, DecisionTreeClassifier, LogisticRegression,
    StackingClassifier, accuracy_score)

X, y = make_blobs(300, centers=3, n_features=5, cluster_std=3.0, random_state=1)
Xtr, ytr, Xte, yte = X[:210], y[:210], X[210:], y[210:]

stack = StackingClassifier(
    estimators=[("tree", DecisionTreeClassifier(max_depth=4)),
                ("lr",   LogisticRegression(epochs=150))],
    final_estimator=LogisticRegression(epochs=150),
    cv=5, random_state=0).fit(Xtr, ytr)

accuracy_score(yte, stack.predict(Xte))
```

## Meta-learners

### Bagging (bootstrap aggregating + random patches / subspaces)
```python
BaggingRegressor(base_estimator, n_estimators=10, max_samples=1.0,
                 max_features=1.0, bootstrap=True, bootstrap_features=False,
                 oob_score=False, random_state=None)
BaggingClassifier(...)          # soft-vote aggregation
```
`max_features` accepts a fraction (`0.5`), an int count (`3`), or `'sqrt'` /
`'log2'` → random subspaces; with `bootstrap=True` you also get an **out-of-bag**
score (`oob_score_`). Setting `oob_score=True, bootstrap=False` raises (no OOB
without resampling).

### Voting / averaging
```python
VotingRegressor([("a", est_a), ("b", est_b)], weights=[1.0, 2.0])    # weighted mean
VotingClassifier([...], voting="soft"|"hard", weights=None)
```

### Stacking (out-of-fold) & Blending (holdout)
```python
StackingRegressor(estimators, final_estimator, cv=5, passthrough=False)
StackingClassifier(estimators, final_estimator, cv=5, passthrough=False)
BlendingRegressor(estimators, final_estimator, holdout=0.3)
BlendingClassifier(estimators, final_estimator, holdout=0.3)
```
Stacking generates the meta-learner's features **out-of-fold** (each base
prediction comes from a model that did *not* see that row), eliminating
leakage; bases are refit on the full data for test-time prediction.
`passthrough=True` also feeds the original features to the meta-learner.

### Boosting (generic over any base estimator)
```python
AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0,
                  loss="linear"|"square"|"exponential")     # AdaBoost.R2
AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0)  # SAMME
GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                          subsample=1.0, loss="ls"|"lad"|"huber")
```
AdaBoost uses weighted **resampling**, so the base estimator need not accept
sample weights. `AdaBoostRegressor` combines weak learners by a **weighted
median**. Gradient boosting fits regression trees to the negative gradient of
the chosen loss (`lad`/`huber` are robust to outliers).

### Ensemble selection & weighted averaging
```python
EnsembleSelection(estimators, task="classification"|"regression",
                  n_iterations=50, holdout=0.3, scorer=None)
WeightedAverageRegressor(estimators, holdout=0.3, n_iter=200)
```
`EnsembleSelection` is Caruana et al.'s greedy forward selection **with
replacement** from a library of models (`weights_` = selection frequencies).
`WeightedAverageRegressor` searches **non-negative convex weights** that
minimise validation MSE.

## Base learners & metrics

Bundled so the file runs standalone (or plug in your own / scikit-learn's):

```python
DecisionTreeRegressor / DecisionTreeClassifier   # shared CART
LinearRegression / Ridge                          # exact normal equations
LogisticRegression                                # softmax GD
DummyRegressor / DummyClassifier                  # baselines

accuracy_score, log_loss, mean_squared_error, mean_absolute_error, r2_score
make_regression, make_blobs
build_ensemble_payload, build_a2ui_payload
```

## Hackathon tips

**Quick strong tabular model:** gradient boosting, then a 2-model stack.
```python
GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3)
```

**Squeeze a leaderboard with models you already have:** dump their predictions
into `EnsembleSelection` (it auto-weights the good ones and ignores the rest) or
`StackingClassifier` with a logistic meta-learner.

**Robust regression with outliers:** `GradientBoostingRegressor(loss="huber")`
or `AdaBoostRegressor` (its weighted median resists outliers).

**Combine heterogeneous learners** (tree + linear + kNN + your own): wrap them
in a soft `VotingClassifier`, or `WeightedAverageRegressor` to learn the blend.

## Tests / Demo

```bash
pytest _stress_test.py -v          # no-leakage proof + known-value checks
python ensembles_template.py       # 13-section demo (regression + classification)
```

The suite proves the parts that are easy to get subtly wrong: stacking's
out-of-fold construction via a **no-leakage test** (on pure noise a leaky
stacker memorises the training set — train accuracy ~1.0 — while a correct
out-of-fold stacker cannot, staying near the baseline); the weighted median and
weighted-average optimiser against known-best targets; that gradient boosting
drives training error to ~0 and that `lad`/`huber` resist outliers; that every
ensemble learns and beats (or matches) its base learners; clone isolation (the
passed-in prototype is never fitted, members are independent objects); and
reproducibility, input validation, and exact metric values.
