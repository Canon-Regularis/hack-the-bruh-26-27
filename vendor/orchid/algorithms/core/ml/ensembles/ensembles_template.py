"""
algorithms/core/ml/ensembles/ensembles_template.py

Pure-Python, zero-dependency ensemble-learning toolkit.

A self-contained, competition-ready framework of *generic* ensemble
meta-learners that wrap any base estimator (classifier OR regressor) exposing a
``fit(X, y)`` / ``predict(X)`` interface - the advanced model-combination
techniques you would otherwise reach to scikit-learn / mlxtend / vecstack for,
in a single file with no NumPy / pandas / scikit-learn.

The estimator contract is duck-typed: a base estimator needs ``fit(X, y)`` and
``predict(X)``; classifiers may additionally expose ``predict_proba(X)`` and
``classes_``.  Pass an *unfitted* prototype - ensembles clone it (deep copy) per
member.

Base learners (so the file runs standalone)
-------------------------------------------
  DecisionTreeRegressor / DecisionTreeClassifier   (shared CART)
  LinearRegression / Ridge                          (normal equations)
  LogisticRegression                                (softmax GD)
  DummyRegressor / DummyClassifier                  (baselines)

Ensemble meta-learners
----------------------
  BaggingRegressor / BaggingClassifier      bootstrap + random-patches + OOB
  VotingRegressor / VotingClassifier        weighted average / hard-soft vote
  StackingRegressor / StackingClassifier    out-of-fold stacked generalisation
  BlendingRegressor / BlendingClassifier    holdout-based stacking
  AdaBoostRegressor (AdaBoost.R2) / AdaBoostClassifier (SAMME)
  GradientBoostingRegressor                 pluggable loss (ls / lad / huber)
  EnsembleSelection                         Caruana greedy selection w/ replace
  WeightedAverageRegressor                  loss-optimised convex weights

Metrics & utilities
-------------------
  accuracy_score, log_loss, mean_squared_error, mean_absolute_error, r2_score
  make_regression, make_blobs, build_ensemble_payload, build_a2ui_payload

Run as a script for an end-to-end demo:
  python ensembles_template.py
"""

from __future__ import annotations

import copy
import math
import random
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import (
    Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple, Union,
)

__version__ = "1.0.0"

__all__ = [
    "DecisionTreeRegressor", "DecisionTreeClassifier",
    "LinearRegression", "Ridge", "LogisticRegression",
    "DummyRegressor", "DummyClassifier",
    "BaggingRegressor", "BaggingClassifier",
    "VotingRegressor", "VotingClassifier",
    "StackingRegressor", "StackingClassifier",
    "BlendingRegressor", "BlendingClassifier",
    "AdaBoostRegressor", "AdaBoostClassifier",
    "GradientBoostingRegressor",
    "EnsembleSelection", "WeightedAverageRegressor",
    "accuracy_score", "log_loss", "mean_squared_error", "mean_absolute_error",
    "r2_score",
    "make_regression", "make_blobs",
    "build_ensemble_payload", "build_a2ui_payload",
]

Matrix = List[List[float]]
Vector = List[float]
RandomStateLike = Union[None, int, random.Random]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_random_state(seed: RandomStateLike) -> random.Random:
    if seed is None:
        return random.Random()
    if isinstance(seed, int):
        return random.Random(seed)
    if isinstance(seed, random.Random):
        return seed
    raise ValueError("random_state must be None, int or random.Random")


def _as_2d(X: Matrix) -> Matrix:
    """Materialise X and reject a 1D sequence with a clear message (the usual
    cause: a single feature passed flat instead of as column vectors)."""
    X = list(X)
    if X and not isinstance(X[0], (list, tuple)):
        raise ValueError(
            "X must be a 2D sequence (list of feature rows); got a 1D sequence "
            "- reshape e.g. [[v] for v in X] for a single feature")
    return [list(map(float, row)) for row in X]


def _check_X_y(X: Matrix, y: Sequence) -> Tuple[Matrix, List]:
    X = _as_2d(X)
    y = list(y)
    # reject a 2D column-vector y (the usual cause: df[['target']] / an
    # un-ravelled column) before it crashes cryptically downstream
    if any(isinstance(v, (list, tuple)) for v in y):
        raise ValueError(
            "y must be a 1D sequence of labels/targets; got nested values "
            "- ravel a column vector, e.g. [row[0] for row in y]")
    if len(X) != len(y):
        raise ValueError("X and y have different lengths (%d vs %d)"
                         % (len(X), len(y)))
    if not X:
        raise ValueError("empty training set")
    if len({len(r) for r in X}) != 1:
        raise ValueError("all rows of X must have the same length")
    if any(not math.isfinite(v) for row in X for v in row):
        raise ValueError("X contains NaN or infinity")
    # reject non-finite targets (a single NaN target silently poisons the whole
    # model); float labels are the only ones that can be NaN/inf, so this leaves
    # string / int class labels untouched
    if any(isinstance(v, float) and not math.isfinite(v) for v in y):
        raise ValueError("y contains NaN or infinity")
    return X, y


def _check_X(X: Matrix, n_features: Optional[int] = None) -> Matrix:
    X = _as_2d(X)
    if X and len({len(r) for r in X}) != 1:
        raise ValueError("all rows of X must have the same length")
    if n_features is not None and X and len(X[0]) != n_features:
        raise ValueError("X has %d features, expected %d"
                         % (len(X[0]), n_features))
    if any(not math.isfinite(v) for row in X for v in row):
        raise ValueError("X contains NaN or infinity")
    return X


def _clone(estimator: Any) -> Any:
    """Return an unfitted copy of a (prototype) estimator."""
    return copy.deepcopy(estimator)


def _unique_labels(y: Sequence) -> List:
    s = set(y)
    try:
        return sorted(s)
    except TypeError:
        return sorted(s, key=repr)


def _argmax(v: Sequence[float]) -> int:
    bi, best = 0, v[0]
    for i in range(1, len(v)):
        if v[i] > best:
            best, bi = v[i], i
    return bi


def _softmax(s: Sequence[float]) -> Vector:
    m = max(s)
    e = [math.exp(v - m) for v in s]
    t = sum(e)
    return [v / t for v in e]


def _check_weights(weights: Optional[Sequence[float]], n: int) -> None:
    """Validate voting/averaging weights: right length, non-negative, and a
    strictly positive sum (a zero sum would divide by zero on aggregation)."""
    if weights is None:
        return
    if len(weights) != n:
        raise ValueError("weights length must match estimators (%d vs %d)"
                         % (len(weights), n))
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    if sum(weights) <= 0:
        raise ValueError("weights must sum to a positive value")


def _check_int(value: Any, name: str, minimum: int) -> int:
    """Coerce a whole-number count to int (accepting e.g. a computed ``10.0``)
    and enforce a lower bound, with a clear error instead of a cryptic
    ``'float' object cannot be interpreted as an integer`` deep inside fit."""
    v = value
    if isinstance(v, float):
        if not v.is_integer():
            raise ValueError("%s must be a whole number, got %r" % (name, value))
        v = int(v)
    if not isinstance(v, int):
        raise ValueError("%s must be an integer, got %r" % (name, value))
    if v < minimum:
        raise ValueError("%s must be >= %d, got %d" % (name, minimum, v))
    return v


def _label_index(p: Any, cidx: Dict) -> int:
    """Index of class label ``p`` for the one-hot fallback, with a clear error
    when an estimator yields a value that is not a training class label (the
    usual cause: a regressor was placed in a classification ensemble)."""
    try:
        return cidx[p]
    except (KeyError, TypeError):
        raise ValueError(
            "estimator produced %r, which is not one of the training class "
            "labels %s; a regressor was likely used where a classifier is "
            "required (classification ensembles need classifier bases / "
            "final_estimator)" % (p, list(cidx)))


def _check_named_estimators(estimators: Sequence) -> None:
    """Validate the ``[(name, estimator), ...]`` form, with a clear error when
    bare estimators are passed without names."""
    for item in estimators:
        if not (isinstance(item, (tuple, list)) and len(item) == 2):
            raise ValueError(
                "estimators must be a list of (name, estimator) pairs; got %r"
                % (item,))


class _ScoreMixin:
    """sklearn-style ``score``: mean accuracy for classifiers (those exposing
    ``classes_``), R^2 for regressors."""

    def _n_in(self) -> int:
        """Fitted feature count, or a clear not-fitted error — predict before
        fit would otherwise raise a cryptic AttributeError on n_features_in_."""
        n = getattr(self, "n_features_in_", None)
        if n is None:
            raise ValueError(
                "%s is not fitted yet; call fit before predict/score."
                % type(self).__name__)
        return n

    def score(self, X: Matrix, y: Sequence) -> float:
        pred = self.predict(X)
        if hasattr(self, "classes_"):
            return accuracy_score(list(y), pred)
        return r2_score([float(v) for v in y], pred)


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    pairs = sorted(zip(values, weights))
    total = sum(weights)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= total / 2.0:
            return v
    return pairs[-1][0]


def _solve(A: Matrix, b: Vector) -> Vector:
    """Solve A x = b by Gauss-Jordan with partial pivoting (A is square)."""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            M[piv][col] += 1e-12
        M[col], M[piv] = M[piv], M[col]
        d = M[col][col]
        M[col] = [v / d for v in M[col]]
        for r in range(n):
            if r != col and M[r][col]:
                f = M[r][col]
                M[r] = [a - f * b2 for a, b2 in zip(M[r], M[col])]
    return [M[i][n] for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def accuracy_score(y_true: Sequence, y_pred: Sequence) -> float:
    if not y_true:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]
                       ) -> float:
    n = len(y_true)
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n if n else 0.0


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]
                        ) -> float:
    n = len(y_true)
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n if n else 0.0


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    mean = sum(y_true) / n
    ss_tot = sum((a - mean) ** 2 for a in y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0


def log_loss(y_true: Sequence, y_proba: Sequence,
             labels: Optional[Sequence] = None) -> float:
    labels = list(labels) if labels is not None else _unique_labels(y_true)
    idx = {c: i for i, c in enumerate(labels)}
    eps = 1e-15
    # accept a 1-D sequence of P(positive class) for binary problems, in
    # addition to the 2-D per-class probability matrix (sklearn parity)
    one_d = bool(y_proba) and not hasattr(y_proba[0], "__len__")
    if one_d and len(labels) != 2:
        raise ValueError(
            "1-D y_proba (probability of the positive class) is only valid for "
            "binary targets; pass a 2-D per-class probability matrix instead.")
    total = 0.0
    for t, p in zip(y_true, y_proba):
        if one_d:                                      # p = P(labels[1])
            pos = min(1 - eps, max(eps, float(p)))
            prob = pos if idx[t] == 1 else 1.0 - pos
        else:
            prob = min(1 - eps, max(eps, p[idx[t]]))
        total += -math.log(prob)
    return total / len(y_true) if y_true else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────────────────────────────────────

def make_regression(n_samples: int = 200, *, n_features: int = 5,
                    n_informative: Optional[int] = None, noise: float = 1.0,
                    bias: float = 0.0, random_state: RandomStateLike = None
                    ) -> Tuple[Matrix, Vector]:
    rng = _check_random_state(random_state)
    n_informative = n_informative or n_features
    w = [rng.uniform(-3, 3) if j < n_informative else 0.0
         for j in range(n_features)]
    X, y = [], []
    for _ in range(n_samples):
        row = [rng.gauss(0, 1) for _ in range(n_features)]
        X.append(row)
        y.append(bias + sum(row[j] * w[j] for j in range(n_features))
                 + rng.gauss(0, noise))
    return X, y


def make_blobs(n_samples: int = 200, *, centers: int = 3, n_features: int = 2,
               cluster_std: float = 1.0, random_state: RandomStateLike = None
               ) -> Tuple[Matrix, List[int]]:
    if centers < 1:
        raise ValueError("centers must be >= 1")
    if n_features < 1:
        raise ValueError("n_features must be >= 1")
    rng = _check_random_state(random_state)
    cent = [[rng.uniform(-10, 10) for _ in range(n_features)]
            for _ in range(centers)]
    X, y = [], []
    for i in range(n_samples):
        c = i % centers
        X.append([cent[c][j] + rng.gauss(0, cluster_std)
                  for j in range(n_features)])
        y.append(c)
    order = list(range(n_samples))
    rng.shuffle(order)
    return [X[i] for i in order], [y[i] for i in order]


def _kfold_indices(n: int, k: int, rng: random.Random
                   ) -> List[Tuple[List[int], List[int]]]:
    idx = list(range(n))
    rng.shuffle(idx)
    base, rem = divmod(n, k)
    folds, start = [], 0
    for f in range(k):
        size = base + (1 if f < rem else 0)
        test = idx[start:start + size]
        train = idx[:start] + idx[start + size:]
        folds.append((train, test))
        start += size
    return folds


# ─────────────────────────────────────────────────────────────────────────────
# Shared CART (base regression / classification tree)
# ─────────────────────────────────────────────────────────────────────────────

class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self):
        self.feature = None
        self.threshold = 0.0
        self.left = None
        self.right = None
        self.value = None


class _CART:
    def __init__(self, *, criterion, max_depth, min_samples_split,
                 min_samples_leaf, max_features, n_classes, rng):
        self.criterion = criterion
        self.max_depth = max_depth if max_depth is not None else 1 << 30
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_classes = n_classes
        self.rng = rng

    def fit(self, X, y, sample_weight=None):
        self._d = len(X[0])
        n = len(X)
        sw = sample_weight if sample_weight is not None else [1.0] * n
        self.root = self._build(X, y, sw, list(range(n)), 0)
        return self

    def _mf(self):
        mf = self.max_features
        if mf is None:
            return self._d
        if mf == "sqrt":
            return max(1, int(math.sqrt(self._d)))
        if mf == "log2":
            return max(1, int(math.log2(self._d))) if self._d > 1 else 1
        if isinstance(mf, int):
            return max(1, min(mf, self._d))
        raise ValueError("bad max_features")

    def _leaf(self, y, sw, idxs):
        if self.n_classes:
            c = [0.0] * self.n_classes
            for i in idxs:
                c[y[i]] += sw[i]
            t = sum(c) or 1.0
            return [v / t for v in c]
        ws = sum(sw[i] for i in idxs) or 1.0
        return sum(y[i] * sw[i] for i in idxs) / ws

    def _impurity(self, idxs, y, sw):
        if self.n_classes:
            c = [0.0] * self.n_classes
            for i in idxs:
                c[y[i]] += sw[i]
            t = sum(c)
            if t <= 0:
                return 0.0
            if self.criterion == "gini":
                return 1.0 - sum((v / t) ** 2 for v in c)
            return -sum((v / t) * math.log(v / t) for v in c if v > 0)
        ws = sum(sw[i] for i in idxs)
        if ws <= 0:
            return 0.0
        m = sum(y[i] * sw[i] for i in idxs) / ws
        return sum(sw[i] * (y[i] - m) ** 2 for i in idxs) / ws

    def _build(self, X, y, sw, idxs, depth):
        node = _Node()
        node.value = self._leaf(y, sw, idxs)
        if (depth >= self.max_depth or len(idxs) < self.min_samples_split
                or len({y[i] for i in idxs}) == 1):
            return node
        split = self._best_split(X, y, sw, idxs)
        if split is None:
            return node
        f, thr, left, right = split
        node.feature, node.threshold = f, thr
        node.left = self._build(X, y, sw, left, depth + 1)
        node.right = self._build(X, y, sw, right, depth + 1)
        node.value = None
        return node

    def _best_split(self, X, y, sw, idxs):
        parent = self._impurity(idxs, y, sw)
        wtot = sum(sw[i] for i in idxs)
        best_gain, best = 1e-12, None
        feats = list(range(self._d))
        k = self._mf()
        if k < self._d:
            self.rng.shuffle(feats)
            feats = feats[:k]
        for f in feats:
            order = sorted(idxs, key=lambda i: X[i][f])
            for pos in range(self.min_samples_leaf,
                             len(order) - self.min_samples_leaf + 1):
                if X[order[pos - 1]][f] == X[order[pos]][f]:
                    continue
                left, right = order[:pos], order[pos:]
                wl = sum(sw[i] for i in left)
                wr = wtot - wl
                if wl <= 0 or wr <= 0:
                    continue
                imp = (wl * self._impurity(left, y, sw)
                       + wr * self._impurity(right, y, sw)) / wtot
                if parent - imp > best_gain:
                    best_gain = parent - imp
                    thr = (X[order[pos - 1]][f] + X[order[pos]][f]) / 2.0
                    best = (f, thr, left, right)
        return best

    def _row(self, row):
        return self._leaf_node(row).value

    def _leaf_node(self, row):
        node = self.root
        while node.value is None:
            node = node.left if row[node.feature] <= node.threshold \
                else node.right
        return node


# ─────────────────────────────────────────────────────────────────────────────
# Base learners
# ─────────────────────────────────────────────────────────────────────────────

class DecisionTreeRegressor(_ScoreMixin):
    def __init__(self, *, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = _check_int(min_samples_split,
                                            "min_samples_split", 2)
        self.min_samples_leaf = _check_int(min_samples_leaf,
                                           "min_samples_leaf", 1)
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X, y = _check_X_y(X, y)
        self.n_features_in_ = len(X[0])
        self._t = _CART(criterion="mse", max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        max_features=self.max_features, n_classes=0,
                        rng=_check_random_state(self.random_state))
        self._t.fit(X, [float(v) for v in y], sample_weight)
        return self

    def predict(self, X):
        return [self._t._row(r)
                for r in _check_X(X, self._n_in())]


class DecisionTreeClassifier(_ScoreMixin):
    def __init__(self, *, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features=None,
                 random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = _check_int(min_samples_split,
                                            "min_samples_split", 2)
        self.min_samples_leaf = _check_int(min_samples_leaf,
                                           "min_samples_leaf", 1)
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X, y = _check_X_y(X, y)
        self.n_features_in_ = len(X[0])
        self.classes_ = _unique_labels(y)
        idx = {c: i for i, c in enumerate(self.classes_)}
        self._t = _CART(criterion=self.criterion, max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        max_features=self.max_features,
                        n_classes=len(self.classes_),
                        rng=_check_random_state(self.random_state))
        self._t.fit(X, [idx[v] for v in y], sample_weight)
        return self

    def predict_proba(self, X):
        return [self._t._row(r)
                for r in _check_X(X, self._n_in())]

    def predict(self, X):
        return [self.classes_[_argmax(p)] for p in self.predict_proba(X)]


class LinearRegression(_ScoreMixin):
    """Ordinary least squares via the normal equations."""

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.alpha = 0.0

    def fit(self, X, y, sample_weight=None):
        X, y = _check_X_y(X, y)
        self.n_features_in_ = len(X[0])
        rows = [([1.0] + r if self.fit_intercept else list(r)) for r in X]
        d = len(rows[0])
        sw = sample_weight if sample_weight is not None else [1.0] * len(rows)
        # weighted normal equations: (X^T W X + alpha I') w = X^T W y
        A = [[0.0] * d for _ in range(d)]
        b = [0.0] * d
        for i, row in enumerate(rows):
            wi = sw[i]
            yi = float(y[i])
            for a in range(d):
                ra = row[a] * wi
                b[a] += ra * yi
                Aa = A[a]
                for c in range(d):
                    Aa[c] += ra * row[c]
        for a in range(d):
            if not (self.fit_intercept and a == 0):
                A[a][a] += self.alpha
        self.coef_ = _solve(A, b)
        return self

    def predict(self, X):
        X = _check_X(X, self._n_in())
        rows = [([1.0] + r if self.fit_intercept else list(r)) for r in X]
        return [sum(c * v for c, v in zip(self.coef_, row)) for row in rows]


class Ridge(LinearRegression):
    """L2-regularised least squares (the bias term is not penalised)."""

    def __init__(self, *, alpha=1.0, fit_intercept=True):
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.fit_intercept = fit_intercept
        self.alpha = alpha


class LogisticRegression(_ScoreMixin):
    """Multinomial logistic regression (softmax) by gradient descent."""

    def __init__(self, *, lr=0.1, epochs=200, l2=1e-4, fit_intercept=True,
                 random_state=None):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X, y = _check_X_y(X, y)
        self.n_features_in_ = len(X[0])
        self.classes_ = _unique_labels(y)
        idx = {c: i for i, c in enumerate(self.classes_)}
        K = len(self.classes_)
        rows = [([1.0] + r if self.fit_intercept else list(r)) for r in X]
        d = len(rows[0])
        n = len(rows)
        sw = sample_weight if sample_weight is not None else [1.0] * n
        W = [[0.0] * d for _ in range(K)]
        for _ in range(self.epochs):
            grad = [[0.0] * d for _ in range(K)]
            for i in range(n):
                row = rows[i]
                p = _softmax([sum(W[k][j] * row[j] for j in range(d))
                              for k in range(K)])
                yi = idx[y[i]]
                for k in range(K):
                    e = sw[i] * (p[k] - (1.0 if k == yi else 0.0))
                    gk = grad[k]
                    for j in range(d):
                        gk[j] += e * row[j]
            for k in range(K):
                for j in range(d):
                    g = grad[k][j] / n
                    if not (self.fit_intercept and j == 0):
                        g += self.l2 * W[k][j]
                    W[k][j] -= self.lr * g
        self.coef_ = W
        return self

    def _scores(self, X):
        X = _check_X(X, self._n_in())
        rows = [([1.0] + r if self.fit_intercept else list(r)) for r in X]
        return [[sum(self.coef_[k][j] * row[j] for j in range(len(row)))
                 for k in range(len(self.classes_))] for row in rows]

    def predict_proba(self, X):
        return [_softmax(s) for s in self._scores(X)]

    def predict(self, X):
        return [self.classes_[_argmax(s)] for s in self._scores(X)]


class DummyRegressor(_ScoreMixin):
    def __init__(self, *, strategy="mean"):
        if strategy not in ("mean", "median"):
            raise ValueError("strategy must be 'mean' or 'median'")
        self.strategy = strategy

    def fit(self, X, y, sample_weight=None):
        # a dummy ignores X by design; still record n_features_in_ so predict
        # can give a clear not-fitted error and a feature-count check.
        self.n_features_in_ = len(X[0])
        y = [float(v) for v in y]
        if self.strategy == "mean":
            self._c = sum(y) / len(y)
        else:
            s = sorted(y)
            self._c = s[len(s) // 2]
        return self

    def predict(self, X):
        X = _check_X(X, self._n_in())
        return [self._c] * len(X)


class DummyClassifier(_ScoreMixin):
    def __init__(self, *, strategy="prior"):
        if strategy not in ("prior", "most_frequent", "uniform"):
            raise ValueError("bad strategy")
        self.strategy = strategy

    def fit(self, X, y, sample_weight=None):
        self.n_features_in_ = len(X[0])     # dummy ignores X by design
        self.classes_ = _unique_labels(y)
        counts = Counter(y)
        n = len(y)
        self._prior = [counts[c] / n for c in self.classes_]
        self._top = max(self.classes_, key=lambda c: counts[c])
        return self

    def predict_proba(self, X):
        X = _check_X(X, self._n_in())
        if self.strategy == "uniform":
            u = 1.0 / len(self.classes_)
            return [[u] * len(self.classes_) for _ in X]
        if self.strategy == "most_frequent":
            row = [1.0 if c == self._top else 0.0 for c in self.classes_]
            return [list(row) for _ in X]
        return [list(self._prior) for _ in X]

    def predict(self, X):
        return [self.classes_[_argmax(p)] for p in self.predict_proba(X)]


# ─────────────────────────────────────────────────────────────────────────────
# Bagging  (bootstrap aggregating + random patches / subspaces + OOB)
# ─────────────────────────────────────────────────────────────────────────────

class _BaseBagging(_ScoreMixin, ABC):
    def __init__(self, base_estimator, *, n_estimators=10, max_samples=1.0,
                 max_features=1.0, bootstrap=True, bootstrap_features=False,
                 oob_score=False, random_state=None):
        if oob_score and not bootstrap:
            raise ValueError("oob_score requires bootstrap=True")
        self.base_estimator = base_estimator
        self.n_estimators = _check_int(n_estimators, "n_estimators", 1)
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.random_state = random_state

    def _n(self, frac, total):
        if isinstance(frac, str):
            if frac == "sqrt":
                return max(1, int(math.sqrt(total)))
            if frac == "log2":
                return max(1, int(math.log2(total))) if total > 1 else 1
            raise ValueError("max_features/max_samples string must be 'sqrt' "
                             "or 'log2', got %r" % frac)
        if isinstance(frac, float):
            return max(1, int(round(frac * total)))
        return max(1, min(int(frac), total))

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        self.n_features_in_ = len(X[0])
        n, d = len(X), len(X[0])
        rng = _check_random_state(self.random_state)
        self._setup(y)
        ns = self._n(self.max_samples, n)
        nf = self._n(self.max_features, d)
        self.estimators_, self.features_ = [], []
        oob_acc: Dict[int, List] = defaultdict(list)
        for _ in range(self.n_estimators):
            if self.bootstrap_features:
                feats = [rng.randrange(d) for _ in range(nf)]
            else:
                feats = list(range(d))
                rng.shuffle(feats)
                feats = sorted(feats[:nf])
            if self.bootstrap:
                rows = [rng.randrange(n) for _ in range(ns)]
            else:
                idx = list(range(n))
                rng.shuffle(idx)
                rows = idx[:ns]
            Xs = [[X[i][f] for f in feats] for i in rows]
            ys = [y[i] for i in rows]
            est = _clone(self.base_estimator).fit(Xs, ys)
            self.estimators_.append(est)
            self.features_.append(feats)
            if self.oob_score and self.bootstrap:
                inbag = set(rows)
                oob = [i for i in range(n) if i not in inbag]
                if oob:
                    Xo = [[X[i][f] for f in feats] for i in oob]
                    for i, p in zip(oob, est.predict(Xo)):
                        oob_acc[i].append(p)
        if self.oob_score and self.bootstrap:
            self.oob_score_ = self._oob(oob_acc, y)
        return self

    def _project(self, X, feats):
        return [[row[f] for f in feats] for row in X]

    @abstractmethod
    def _setup(self, y):
        ...

    @abstractmethod
    def _oob(self, oob_acc, y):
        ...


class BaggingRegressor(_BaseBagging):
    """Average the predictions of base regressors fit on bootstrap samples
    (optionally with feature subsampling = random patches / subspaces)."""

    def _setup(self, y):
        pass

    def _oob(self, oob_acc, y):
        preds, trues = [], []
        for i, ps in oob_acc.items():
            preds.append(sum(ps) / len(ps))
            trues.append(float(y[i]))
        return r2_score(trues, preds) if trues else 0.0

    def predict(self, X):
        X = _check_X(X, self._n_in())
        n = len(X)
        agg = [0.0] * n
        for est, feats in zip(self.estimators_, self.features_):
            for i, p in enumerate(est.predict(self._project(X, feats))):
                agg[i] += p
        return [a / len(self.estimators_) for a in agg]


class BaggingClassifier(_BaseBagging):
    """Soft-vote the predictions of base classifiers fit on bootstrap samples
    (with optional feature subsampling)."""

    def _setup(self, y):
        self.classes_ = _unique_labels(y)

    def _proba_of(self, est, X):
        K = len(self.classes_)
        cidx = {c: i for i, c in enumerate(self.classes_)}
        if hasattr(est, "predict_proba") and getattr(est, "classes_", None):
            sub = est.classes_
            out = []
            for row in est.predict_proba(X):
                full = [0.0] * K
                for c, v in zip(sub, row):
                    full[cidx[c]] = v
                out.append(full)
            return out
        out = []
        for p in est.predict(X):
            row = [0.0] * K
            row[_label_index(p, cidx)] = 1.0
            out.append(row)
        return out

    def _oob(self, oob_acc, y):
        correct = total = 0
        for i, ps in oob_acc.items():
            total += 1
            if Counter(ps).most_common(1)[0][0] == y[i]:
                correct += 1
        return correct / total if total else 0.0

    def predict_proba(self, X):
        X = _check_X(X, self._n_in())
        K = len(self.classes_)
        agg = [[0.0] * K for _ in X]
        for est, feats in zip(self.estimators_, self.features_):
            proba = self._proba_of(est, self._project(X, feats))
            for i in range(len(X)):
                for k in range(K):
                    agg[i][k] += proba[i][k]
        m = len(self.estimators_)
        return [[v / m for v in row] for row in agg]

    def predict(self, X):
        return [self.classes_[_argmax(p)] for p in self.predict_proba(X)]


# ─────────────────────────────────────────────────────────────────────────────
# Voting / averaging
# ─────────────────────────────────────────────────────────────────────────────

class VotingRegressor(_ScoreMixin):
    """Weighted average of several fitted base regressors."""

    def __init__(self, estimators, *, weights=None):
        if not estimators:
            raise ValueError("need at least one estimator")
        _check_named_estimators(estimators)
        _check_weights(weights, len(estimators))
        self.estimators = list(estimators)
        self.weights = weights

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        self.n_features_in_ = len(X[0])
        self._fitted = [(n, _clone(e).fit(X, y)) for n, e in self.estimators]
        return self

    def predict(self, X):
        X = _check_X(X, self._n_in())
        w = self.weights or [1.0] * len(self._fitted)
        wsum = sum(w)
        preds = [e.predict(X) for _, e in self._fitted]
        return [sum(w[k] * preds[k][i] for k in range(len(preds))) / wsum
                for i in range(len(X))]


class VotingClassifier(_ScoreMixin):
    """Hard or soft voting over several fitted base classifiers."""

    def __init__(self, estimators, *, voting="hard", weights=None):
        if voting not in ("hard", "soft"):
            raise ValueError("voting must be 'hard' or 'soft'")
        if not estimators:
            raise ValueError("need at least one estimator")
        _check_named_estimators(estimators)
        _check_weights(weights, len(estimators))
        self.estimators = list(estimators)
        self.voting = voting
        self.weights = weights

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        self.n_features_in_ = len(X[0])
        self.classes_ = _unique_labels(y)
        self._fitted = [(n, _clone(e).fit(X, y)) for n, e in self.estimators]
        if self.voting == "soft":
            missing = [n for n, e in self._fitted
                       if not hasattr(e, "predict_proba")]
            if missing:
                raise ValueError(
                    "voting='soft' requires every estimator to implement "
                    "predict_proba, but %s do not; use voting='hard' or swap "
                    "in proba-capable estimators" % missing)
        return self

    def _w(self, i):
        return 1.0 if self.weights is None else self.weights[i]

    def predict_proba(self, X):
        if self.voting != "soft":
            raise NotImplementedError("predict_proba requires voting='soft'")
        X = _check_X(X, self._n_in())
        K = len(self.classes_)
        cidx = {c: i for i, c in enumerate(self.classes_)}
        agg = [[0.0] * K for _ in X]
        for i, (_, est) in enumerate(self._fitted):
            w = self._w(i)
            sub = getattr(est, "classes_", self.classes_)
            for r, row in enumerate(est.predict_proba(X)):
                for c, v in zip(sub, row):
                    agg[r][cidx[c]] += w * v
        return [[v / (sum(row) or 1.0) for v in row] for row in agg]

    def predict(self, X):
        X = _check_X(X, self._n_in())
        if self.voting == "soft":
            return [self.classes_[_argmax(p)] for p in self.predict_proba(X)]
        cidx = {c: i for i, c in enumerate(self.classes_)}
        tally = [[0.0] * len(self.classes_) for _ in X]
        for i, (_, est) in enumerate(self._fitted):
            w = self._w(i)
            for r, p in enumerate(est.predict(X)):
                tally[r][cidx[p]] += w
        return [self.classes_[_argmax(t)] for t in tally]


# ─────────────────────────────────────────────────────────────────────────────
# Stacking (out-of-fold) and Blending (holdout)
# ─────────────────────────────────────────────────────────────────────────────

class _StackBase(_ScoreMixin, ABC):
    def __init__(self, estimators, final_estimator, *, cv=5,
                 passthrough=False, random_state=None):
        if not estimators:
            raise ValueError("need at least one base estimator")
        _check_named_estimators(estimators)
        self.estimators = list(estimators)
        self.final_estimator = final_estimator
        self.cv = _check_int(cv, "cv", 2)
        self.passthrough = passthrough
        self.random_state = random_state

    @abstractmethod
    def _meta_features_for_base(self, est, X):
        """Columns this fitted base contributes to the meta feature matrix."""

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        self.n_features_in_ = len(X[0])
        self._setup(y)
        n = len(X)
        rng = _check_random_state(self.random_state)
        folds = _kfold_indices(n, self.cv, rng)
        # out-of-fold meta features: each row built from base models that did
        # NOT see it in training -> no leakage into the meta-learner
        meta = [[] for _ in range(n)]
        for _, est in self.estimators:
            cols = [None] * n
            for train, test in folds:
                fold_est = _clone(est).fit([X[i] for i in train],
                                           [y[i] for i in train])
                feats = self._meta_features_for_base(
                    fold_est, [X[i] for i in test])
                for i, row in zip(test, feats):
                    cols[i] = row
            for i in range(n):
                meta[i].extend(cols[i])
        if self.passthrough:
            for i in range(n):
                meta[i] = meta[i] + X[i]
        # refit each base on the FULL data for test-time prediction
        self.estimators_ = [_clone(e).fit(X, y) for _, e in self.estimators]
        self.final_estimator_ = _clone(self.final_estimator).fit(meta, y)
        return self

    def _transform(self, X):
        meta = [[] for _ in X]
        for est in self.estimators_:
            feats = self._meta_features_for_base(est, X)
            for i, row in enumerate(feats):
                meta[i].extend(row)
        if self.passthrough:
            meta = [meta[i] + X[i] for i in range(len(X))]
        return meta

    @abstractmethod
    def _setup(self, y):
        ...


class StackingRegressor(_StackBase):
    """Stacked generalisation for regression: base predictions become the meta
    learner's features, generated out-of-fold to avoid leakage."""

    def _setup(self, y):
        pass

    def _meta_features_for_base(self, est, X):
        return [[p] for p in est.predict(X)]

    def predict(self, X):
        X = _check_X(X, self._n_in())
        return self.final_estimator_.predict(self._transform(X))


class StackingClassifier(_StackBase):
    """Stacked generalisation for classification.  Each base contributes its
    class-probability columns (or a one-hot of its prediction) to the meta
    feature matrix."""

    def _setup(self, y):
        self.classes_ = _unique_labels(y)

    def _meta_features_for_base(self, est, X):
        K = len(self.classes_)
        cidx = {c: i for i, c in enumerate(self.classes_)}
        if hasattr(est, "predict_proba"):
            sub = getattr(est, "classes_", self.classes_)
            out = []
            for row in est.predict_proba(X):
                full = [0.0] * K
                for c, v in zip(sub, row):
                    full[cidx[c]] = v
                out.append(full)
            return out
        out = []
        for p in est.predict(X):
            row = [0.0] * K
            row[_label_index(p, cidx)] = 1.0
            out.append(row)
        return out

    def predict_proba(self, X):
        X = _check_X(X, self._n_in())
        meta = self._transform(X)
        if hasattr(self.final_estimator_, "predict_proba"):
            sub = getattr(self.final_estimator_, "classes_", self.classes_)
            cidx = {c: i for i, c in enumerate(self.classes_)}
            out = []
            for row in self.final_estimator_.predict_proba(meta):
                full = [0.0] * len(self.classes_)
                for c, v in zip(sub, row):
                    full[cidx[c]] = v
                out.append(full)
            return out
        cidx = {c: i for i, c in enumerate(self.classes_)}
        out = []
        for p in self.final_estimator_.predict(meta):
            row = [0.0] * len(self.classes_)
            row[_label_index(p, cidx)] = 1.0
            out.append(row)
        return out

    def predict(self, X):
        X = _check_X(X, self._n_in())
        return self.final_estimator_.predict(self._transform(X))


class _BlendBase(_ScoreMixin, ABC):
    def __init__(self, estimators, final_estimator, *, holdout=0.3,
                 random_state=None):
        if not estimators:
            raise ValueError("need at least one base estimator")
        _check_named_estimators(estimators)
        if not 0.0 < holdout < 1.0:
            raise ValueError("holdout must be in (0, 1)")
        self.estimators = list(estimators)
        self.final_estimator = final_estimator
        self.holdout = holdout
        self.random_state = random_state

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        self.n_features_in_ = len(X[0])
        self._setup(y)
        n = len(X)
        rng = _check_random_state(self.random_state)
        idx = list(range(n))
        rng.shuffle(idx)
        cut = int(round((1 - self.holdout) * n))
        tr, ho = idx[:cut], idx[cut:]
        if not tr or not ho:
            raise ValueError("too few samples to blend")
        # bases learn on the train part; meta learns on their holdout predictions
        self.estimators_ = [_clone(e).fit([X[i] for i in tr],
                                          [y[i] for i in tr])
                            for _, e in self.estimators]
        meta = self._transform([X[i] for i in ho])
        self.final_estimator_ = _clone(self.final_estimator).fit(
            meta, [y[i] for i in ho])
        return self

    def _transform(self, X):
        meta = [[] for _ in X]
        for est in self.estimators_:
            for i, row in enumerate(self._cols(est, X)):
                meta[i].extend(row)
        return meta

    @abstractmethod
    def _cols(self, est, X):
        ...

    @abstractmethod
    def _setup(self, y):
        ...


class BlendingRegressor(_BlendBase):
    """Holdout-based stacking for regression (faster, slightly higher variance
    than out-of-fold stacking)."""

    def _setup(self, y):
        pass

    def _cols(self, est, X):
        return [[p] for p in est.predict(X)]

    def predict(self, X):
        X = _check_X(X, self._n_in())
        return self.final_estimator_.predict(self._transform(X))


class BlendingClassifier(_BlendBase):
    """Holdout-based stacking for classification."""

    def _setup(self, y):
        self.classes_ = _unique_labels(y)

    def _cols(self, est, X):
        K = len(self.classes_)
        cidx = {c: i for i, c in enumerate(self.classes_)}
        if hasattr(est, "predict_proba"):
            sub = getattr(est, "classes_", self.classes_)
            out = []
            for row in est.predict_proba(X):
                full = [0.0] * K
                for c, v in zip(sub, row):
                    full[cidx[c]] = v
                out.append(full)
            return out
        out = []
        for p in est.predict(X):
            row = [0.0] * K
            row[_label_index(p, cidx)] = 1.0
            out.append(row)
        return out

    def predict(self, X):
        X = _check_X(X, self._n_in())
        return self.final_estimator_.predict(self._transform(X))


# ─────────────────────────────────────────────────────────────────────────────
# Boosting (generic over any base estimator)
# ─────────────────────────────────────────────────────────────────────────────

class AdaBoostRegressor(_ScoreMixin):
    """AdaBoost.R2 (Drucker 1997): re-weighted resampling, weak learners
    combined by a weighted median."""

    def __init__(self, base_estimator=None, *, n_estimators=50,
                 learning_rate=1.0, loss="linear", random_state=None):
        if loss not in ("linear", "square", "exponential"):
            raise ValueError("loss must be linear/square/exponential")
        self.base_estimator = base_estimator
        self.n_estimators = _check_int(n_estimators, "n_estimators", 1)
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        y = [float(v) for v in y]
        self.n_features_in_ = len(X[0])
        n = len(X)
        rng = _check_random_state(self.random_state)
        base = self.base_estimator or DecisionTreeRegressor(max_depth=3)
        w = [1.0 / n] * n
        self.estimators_, self.estimator_weights_ = [], []
        for _ in range(self.n_estimators):
            sample = rng.choices(range(n), weights=w, k=n)
            est = _clone(base).fit([X[i] for i in sample],
                                   [y[i] for i in sample])
            pred = est.predict(X)
            err = [abs(pred[i] - y[i]) for i in range(n)]
            D = max(err) or 1e-12
            if self.loss == "linear":
                loss = [e / D for e in err]
            elif self.loss == "square":
                loss = [(e / D) ** 2 for e in err]
            else:
                loss = [1.0 - math.exp(-e / D) for e in err]
            avg = sum(w[i] * loss[i] for i in range(n))
            if avg <= 0:
                # base fits the targets exactly: it is a perfect predictor on
                # its own.  Boosting further would drive every sample weight to
                # zero (and crash the next weighted resample), so stop here.
                self.estimators_.append(est)
                self.estimator_weights_.append(1.0)
                break
            if avg >= 0.5:
                if not self.estimators_:
                    self.estimators_.append(est)
                    self.estimator_weights_.append(1.0)
                break
            beta = avg / (1.0 - avg)
            self.estimators_.append(est)
            self.estimator_weights_.append(
                self.learning_rate * math.log(1.0 / (beta + 1e-300)))
            for i in range(n):
                w[i] *= beta ** (self.learning_rate * (1.0 - loss[i]))
            s = sum(w) or 1.0
            w = [wi / s for wi in w]
        return self

    def predict(self, X):
        X = _check_X(X, self._n_in())
        preds = [est.predict(X) for est in self.estimators_]
        out = []
        for i in range(len(X)):
            vals = [preds[m][i] for m in range(len(self.estimators_))]
            out.append(_weighted_median(vals, self.estimator_weights_))
        return out


class AdaBoostClassifier(_ScoreMixin):
    """AdaBoost SAMME (Zhu et al. 2009) generic over any base classifier, using
    weighted resampling so the base need not accept sample weights."""

    def __init__(self, base_estimator=None, *, n_estimators=50,
                 learning_rate=1.0, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = _check_int(n_estimators, "n_estimators", 1)
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        self.n_features_in_ = len(X[0])
        self.classes_ = _unique_labels(y)
        K = len(self.classes_)
        n = len(X)
        rng = _check_random_state(self.random_state)
        base = self.base_estimator or DecisionTreeClassifier(max_depth=1)
        w = [1.0 / n] * n
        self.estimators_, self.estimator_weights_ = [], []
        for _ in range(self.n_estimators):
            sample = rng.choices(range(n), weights=w, k=n)
            est = _clone(base).fit([X[i] for i in sample],
                                   [y[i] for i in sample])
            pred = est.predict(X)
            err = sum(w[i] for i in range(n) if pred[i] != y[i])
            err = min(max(err, 1e-10), 1 - 1e-10)
            alpha = self.learning_rate * (math.log((1 - err) / err)
                                          + math.log(K - 1)) if K > 1 else 0.0
            if alpha <= 0:
                continue
            self.estimators_.append(est)
            self.estimator_weights_.append(alpha)
            for i in range(n):
                if pred[i] != y[i]:
                    w[i] *= math.exp(alpha)
            s = sum(w) or 1.0
            w = [wi / s for wi in w]
        if not self.estimators_:
            self.estimators_.append(est)
            self.estimator_weights_.append(1.0)
        return self

    def decision_function(self, X):
        X = _check_X(X, self._n_in())
        K = len(self.classes_)
        cidx = {c: i for i, c in enumerate(self.classes_)}
        out = []
        for r in range(len(X)):
            out.append([0.0] * K)
        for est, alpha in zip(self.estimators_, self.estimator_weights_):
            for r, p in enumerate(est.predict(X)):
                out[r][cidx[p]] += alpha
        return out

    def predict_proba(self, X):
        return [_softmax(s) for s in self.decision_function(X)]

    def predict(self, X):
        return [self.classes_[_argmax(s)] for s in self.decision_function(X)]


class GradientBoostingRegressor(_ScoreMixin):
    """Gradient boosting for regression with a pluggable loss (``ls`` =
    least-squares, ``lad`` = least-absolute-deviation, ``huber``)."""

    def __init__(self, *, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=1, subsample=1.0, loss="ls",
                 alpha_huber=0.9, random_state=None):
        if loss not in ("ls", "lad", "huber"):
            raise ValueError("loss must be ls/lad/huber")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 < subsample <= 1:
            raise ValueError("subsample must be in (0, 1]")
        if not 0 < alpha_huber < 1:
            raise ValueError("alpha_huber must be in (0, 1)")
        self.n_estimators = _check_int(n_estimators, "n_estimators", 1)
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.loss = loss
        self.alpha_huber = alpha_huber
        self.random_state = random_state

    def _gradient(self, res):
        """Negative gradient of the loss w.r.t. F, plus the huber delta."""
        if self.loss == "ls":
            return res, None
        if self.loss == "lad":
            return [1.0 if r > 0 else -1.0 if r < 0 else 0.0 for r in res], None
        n = len(res)
        delta = sorted(abs(r) for r in res)[int(self.alpha_huber * (n - 1))]
        delta = delta or 1e-12
        return [r if abs(r) <= delta else delta * (1.0 if r > 0 else -1.0)
                for r in res], delta

    def _line_search(self, tree, X, res, sub, delta):
        """Friedman TreeBoost: replace each leaf's value with the constant that
        optimises the loss over the residuals reaching it.  Without this, the
        bounded lad sub-gradient (in [-1, 1]) cannot move a large-magnitude
        target - each stage shifts F by only ~learning_rate."""
        groups: Dict[int, Tuple[_Node, List[float]]] = {}
        for i in sub:
            node = tree._t._leaf_node(X[i])
            groups.setdefault(id(node), (node, []))[1].append(res[i])
        for node, rs in groups.values():
            rs.sort()
            med = rs[len(rs) // 2]
            if self.loss == "lad":
                node.value = med
            else:  # huber: median + mean of the clipped deviations
                node.value = med + sum(
                    max(-delta, min(delta, r - med)) for r in rs) / len(rs)

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        y = [float(v) for v in y]
        self.n_features_in_ = len(X[0])
        n = len(X)
        rng = _check_random_state(self.random_state)
        if self.loss == "ls":
            self._init = sum(y) / n
        else:
            self._init = sorted(y)[n // 2]
        F = [self._init] * n
        self.estimators_ = []
        for _ in range(self.n_estimators):
            res = [y[i] - F[i] for i in range(n)]
            g, delta = self._gradient(res)
            if self.subsample < 1.0:
                m = max(1, int(self.subsample * n))
                sub = rng.sample(range(n), m)
            else:
                sub = list(range(n))
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         min_samples_leaf=self.min_samples_leaf,
                                         random_state=rng.randrange(2 ** 31))
            tree.fit([X[i] for i in sub], [g[i] for i in sub])
            if self.loss in ("lad", "huber"):
                self._line_search(tree, X, res, sub, delta)
            upd = tree.predict(X)
            for i in range(n):
                F[i] += self.learning_rate * upd[i]
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        X = _check_X(X, self._n_in())
        F = [self._init] * len(X)
        for tree in self.estimators_:
            for i, u in enumerate(tree.predict(X)):
                F[i] += self.learning_rate * u
        return F


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble selection & weighted averaging
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleSelection(_ScoreMixin):
    """Caruana et al. (2004) greedy ensemble selection from a library of
    estimators.  Models are added (with replacement) to greedily optimise a
    validation score; the final ensemble averages members by selection count.

    ``task`` is ``"classification"`` or ``"regression"``; ``greater_is_better``
    controls the score direction (defaults: accuracy up for classification,
    negative-MSE up for regression)."""

    def __init__(self, estimators, *, task="classification", n_iterations=50,
                 holdout=0.3, scorer=None, greater_is_better=None,
                 random_state=None):
        if not estimators:
            raise ValueError("need at least one estimator")
        _check_named_estimators(estimators)
        if task not in ("classification", "regression"):
            raise ValueError("task must be classification or regression")
        self.estimators = list(estimators)
        self.task = task
        self.n_iterations = _check_int(n_iterations, "n_iterations", 1)
        self.holdout = holdout
        self.scorer = scorer
        self.greater_is_better = greater_is_better
        self.random_state = random_state

    def _score(self, y, ensemble_pred):
        if self.scorer is not None:
            s = self.scorer(y, ensemble_pred)
            # the greedy maximises; orient the custom score to higher-is-better.
            # default assumes the scorer is already higher-better; negate only
            # when greater_is_better is explicitly False (e.g. a loss/error)
            return s if self.greater_is_better is not False else -s
        if self.task == "classification":
            preds = [self.classes_[_argmax(p)] for p in ensemble_pred]
            return accuracy_score(y, preds)
        return -mean_squared_error(y, ensemble_pred)

    def _member_pred(self, est, X):
        if self.task == "classification":
            K = len(self.classes_)
            cidx = {c: i for i, c in enumerate(self.classes_)}
            if hasattr(est, "predict_proba"):
                sub = getattr(est, "classes_", self.classes_)
                out = []
                for row in est.predict_proba(X):
                    full = [0.0] * K
                    for c, v in zip(sub, row):
                        full[cidx[c]] = v
                    out.append(full)
                return out
            out = []
            for p in est.predict(X):
                row = [0.0] * K
                row[_label_index(p, cidx)] = 1.0
                out.append(row)
            return out
        return est.predict(X)

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        self.n_features_in_ = len(X[0])
        if self.task == "classification":
            self.classes_ = _unique_labels(y)
        rng = _check_random_state(self.random_state)
        n = len(X)
        idx = list(range(n))
        rng.shuffle(idx)
        cut = int(round((1 - self.holdout) * n))
        tr, val = idx[:cut], idx[cut:]
        if not tr or not val:
            raise ValueError("too few samples for ensemble selection")
        Xtr, ytr = [X[i] for i in tr], [y[i] for i in tr]
        Xval, yval = [X[i] for i in val], [y[i] for i in val]
        self.estimators_ = [_clone(e).fit(Xtr, ytr) for _, e in self.estimators]
        val_preds = [self._member_pred(e, Xval) for e in self.estimators_]
        L = len(self.estimators_)
        counts = [0] * L
        # greedy forward selection with replacement; remember the best-scoring
        # ensemble visited (Caruana et al.) so a late drift cannot leave us with
        # an ensemble worse than the best single member picked on iteration 1
        best_counts, best_overall = None, -math.inf
        for _ in range(self.n_iterations):
            best_j, best_score = None, -math.inf
            for j in range(L):
                cnt = counts[:]
                cnt[j] += 1
                sc = self._score(yval, self._combine(val_preds, cnt))
                if sc > best_score:
                    best_score, best_j = sc, j
            counts[best_j] += 1
            if best_score > best_overall:
                best_overall, best_counts = best_score, counts[:]
        counts = best_counts if best_counts is not None else counts
        total = sum(counts) or 1
        self.weights_ = [c / total for c in counts]
        return self

    def _combine(self, preds, counts):
        total = sum(counts) or 1
        if self.task == "classification":
            K = len(self.classes_)
            out = [[0.0] * K for _ in preds[0]]
            for j, c in enumerate(counts):
                if not c:
                    continue
                for i, row in enumerate(preds[j]):
                    for k in range(K):
                        out[i][k] += c * row[k] / total
            return out
        out = [0.0] * len(preds[0])
        for j, c in enumerate(counts):
            if not c:
                continue
            for i, v in enumerate(preds[j]):
                out[i] += c * v / total
        return out

    def predict(self, X):
        X = _check_X(X, self._n_in())
        preds = [self._member_pred(e, X) for e in self.estimators_]
        counts = [round(w * 1000) for w in self.weights_]
        ens = self._combine(preds, counts)
        if self.task == "classification":
            return [self.classes_[_argmax(p)] for p in ens]
        return ens


class WeightedAverageRegressor(_ScoreMixin):
    """Combine base regressors with non-negative convex weights chosen to
    minimise validation MSE (coordinate-descent hill climbing on the simplex)."""

    def __init__(self, estimators, *, holdout=0.3, n_iter=200,
                 random_state=None):
        if not estimators:
            raise ValueError("need at least one estimator")
        _check_named_estimators(estimators)
        self.estimators = list(estimators)
        self.holdout = holdout
        self.n_iter = _check_int(n_iter, "n_iter", 1)
        self.random_state = random_state

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        y = [float(v) for v in y]
        self.n_features_in_ = len(X[0])
        rng = _check_random_state(self.random_state)
        n = len(X)
        idx = list(range(n))
        rng.shuffle(idx)
        cut = int(round((1 - self.holdout) * n))
        tr, val = idx[:cut], idx[cut:]
        Xtr, ytr = [X[i] for i in tr], [y[i] for i in tr]
        Xval, yval = [X[i] for i in val], [y[i] for i in val]
        self.estimators_ = [_clone(e).fit(Xtr, ytr) for _, e in self.estimators]
        P = [e.predict(Xval) for e in self.estimators_]
        L = len(P)
        w = [1.0 / L] * L

        def loss(weights):
            pred = [sum(weights[j] * P[j][i] for j in range(L))
                    for i in range(len(yval))]
            return mean_squared_error(yval, pred)

        best = loss(w)
        step = 0.5
        for _ in range(self.n_iter):
            improved = False
            for j in range(L):
                for s in (step, -step):
                    cand = list(w)
                    cand[j] = max(0.0, cand[j] + s)
                    tot = sum(cand) or 1.0
                    cand = [c / tot for c in cand]
                    lc = loss(cand)
                    if lc < best - 1e-12:
                        best, w, improved = lc, cand, True
            if not improved:
                step *= 0.5
                if step < 1e-4:
                    break
        self.weights_ = w
        # refit on full data for prediction
        self.estimators_ = [_clone(e).fit(X, y) for _, e in self.estimators]
        return self

    def predict(self, X):
        X = _check_X(X, self._n_in())
        P = [e.predict(X) for e in self.estimators_]
        L = len(P)
        return [sum(self.weights_[j] * P[j][i] for j in range(L))
                for i in range(len(X))]


# ─────────────────────────────────────────────────────────────────────────────
# Payloads
# ─────────────────────────────────────────────────────────────────────────────

def _json_num(x: Any) -> Any:
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return None
    return x


def build_ensemble_payload(ensemble: Any, X: Matrix, y: Sequence, *,
                           task: str = "classification") -> Dict[str, Any]:
    """Summarise an ensemble: per-member scores, ensemble score, and member
    weights where available."""
    members = getattr(ensemble, "estimators_", []) \
        or [e for _, e in getattr(ensemble, "_fitted", [])]
    member_scores = []
    for est in members:
        try:
            pred = est.predict(_check_X(X))
            sc = (accuracy_score(list(y), pred) if task == "classification"
                  else r2_score([float(v) for v in y], pred))
            member_scores.append(_json_num(round(sc, 4)))
        except Exception:
            member_scores.append(None)
    pred = ensemble.predict(X)
    ens_score = (accuracy_score(list(y), pred) if task == "classification"
                 else r2_score([float(v) for v in y], pred))
    weights = getattr(ensemble, "weights_", None) \
        or getattr(ensemble, "estimator_weights_", None)
    return {
        "ensemble": type(ensemble).__name__,
        "task": task,
        "numMembers": len(members),
        "memberScores": member_scores,
        "ensembleScore": _json_num(round(ens_score, 4)),
        "weights": [_json_num(round(w, 4)) for w in weights] if weights else None,
        "oobScore": _json_num(round(ensemble.oob_score_, 4))
        if hasattr(ensemble, "oob_score_") else None,
    }


def build_a2ui_payload(ensemble: Any, X: Matrix, y: Sequence, *,
                       task: str = "classification") -> Dict[str, Any]:
    p = build_ensemble_payload(ensemble, X, y, task=task)
    bars = [{"member": "m%d" % i, "score": s}
            for i, s in enumerate(p["memberScores"])]
    return {
        "bars": bars,
        "statCards": [
            {"label": "Members", "value": p["numMembers"]},
            {"label": "Ensemble", "value": p["ensembleScore"]},
            {"label": "Best member",
             "value": max((s for s in p["memberScores"] if s is not None),
                          default=None)},
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def _split(X, y, frac=0.7):
    cut = int(frac * len(X))
    return X[:cut], y[:cut], X[cut:], y[cut:]


def _demo() -> None:
    print("=" * 70)
    print("ENSEMBLES TEMPLATE - demo")
    print("=" * 70)

    # ----- regression -----
    Xr, yr = make_regression(300, n_features=6, noise=2.0, random_state=0)
    Xtr, ytr, Xte, yte = _split(Xr, yr)
    base_r2 = r2_score(yte, DecisionTreeRegressor(max_depth=4,
                       random_state=0).fit(Xtr, ytr).predict(Xte))
    print("\n[1] Regression - single tree R2 = %.3f" % base_r2)

    print("\n[2] BaggingRegressor (+ OOB) vs single tree")
    bag = BaggingRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=50,
                           max_features=0.8, oob_score=True,
                           random_state=0).fit(Xtr, ytr)
    print("    bagging test R2 = %.3f | OOB R2 = %.3f"
          % (r2_score(yte, bag.predict(Xte)), bag.oob_score_))

    print("\n[3] GradientBoostingRegressor (ls / lad / huber)")
    for loss in ("ls", "lad", "huber"):
        gb = GradientBoostingRegressor(n_estimators=80, learning_rate=0.1,
                                       max_depth=3, loss=loss,
                                       random_state=0).fit(Xtr, ytr)
        print("    loss=%-6s test R2 = %.3f" % (loss, r2_score(yte,
                                                gb.predict(Xte))))

    print("\n[4] AdaBoostRegressor (AdaBoost.R2)")
    ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=50,
                            random_state=0).fit(Xtr, ytr)
    print("    test R2 = %.3f" % r2_score(yte, ada.predict(Xte)))

    print("\n[5] StackingRegressor (trees + ridge -> ridge meta)")
    stack = StackingRegressor(
        [("tree", DecisionTreeRegressor(max_depth=4)),
         ("knn", DecisionTreeRegressor(max_depth=8)),
         ("ridge", Ridge(alpha=1.0))],
        final_estimator=Ridge(alpha=1.0), cv=5, random_state=0).fit(Xtr, ytr)
    print("    stacking test R2 = %.3f" % r2_score(yte, stack.predict(Xte)))

    print("\n[6] WeightedAverageRegressor (optimised convex weights)")
    wavg = WeightedAverageRegressor(
        [("t4", DecisionTreeRegressor(max_depth=4)),
         ("t8", DecisionTreeRegressor(max_depth=8)),
         ("ridge", Ridge(alpha=1.0))], random_state=0).fit(Xtr, ytr)
    print("    weights = %s | test R2 = %.3f"
          % ([round(w, 2) for w in wavg.weights_], r2_score(yte,
             wavg.predict(Xte))))

    # ----- classification -----
    Xc, yc = make_blobs(300, centers=3, n_features=5, cluster_std=3.0,
                        random_state=1)
    Xtr, ytr, Xte, yte = _split(Xc, yc)
    print("\n[7] Classification - single tree acc = %.3f"
          % accuracy_score(yte, DecisionTreeClassifier(max_depth=4,
                           random_state=0).fit(Xtr, ytr).predict(Xte)))

    print("\n[8] BaggingClassifier (+ OOB)")
    bc = BaggingClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=50,
                           oob_score=True, random_state=0).fit(Xtr, ytr)
    print("    bagging acc = %.3f | OOB acc = %.3f"
          % (accuracy_score(yte, bc.predict(Xte)), bc.oob_score_))

    print("\n[9] VotingClassifier (soft) over heterogeneous learners")
    vc = VotingClassifier(
        [("tree", DecisionTreeClassifier(max_depth=4)),
         ("lr", LogisticRegression(epochs=150)),
         ("dummy", DummyClassifier())], voting="soft").fit(Xtr, ytr)
    print("    soft-vote acc = %.3f" % accuracy_score(yte, vc.predict(Xte)))

    print("\n[10] StackingClassifier + BlendingClassifier")
    sc = StackingClassifier(
        [("tree", DecisionTreeClassifier(max_depth=4)),
         ("lr", LogisticRegression(epochs=150))],
        final_estimator=LogisticRegression(epochs=150), cv=5,
        random_state=0).fit(Xtr, ytr)
    bl = BlendingClassifier(
        [("tree", DecisionTreeClassifier(max_depth=4)),
         ("lr", LogisticRegression(epochs=150))],
        final_estimator=LogisticRegression(epochs=150),
        random_state=0).fit(Xtr, ytr)
    print("    stacking acc = %.3f | blending acc = %.3f"
          % (accuracy_score(yte, sc.predict(Xte)),
             accuracy_score(yte, bl.predict(Xte))))

    print("\n[11] AdaBoostClassifier (SAMME, stumps)")
    abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                             n_estimators=60, random_state=0).fit(Xtr, ytr)
    print("    AdaBoost acc = %.3f" % accuracy_score(yte, abc.predict(Xte)))

    print("\n[12] EnsembleSelection (Caruana greedy)")
    es = EnsembleSelection(
        [("t2", DecisionTreeClassifier(max_depth=2)),
         ("t4", DecisionTreeClassifier(max_depth=4)),
         ("t8", DecisionTreeClassifier(max_depth=8)),
         ("lr", LogisticRegression(epochs=150)),
         ("dummy", DummyClassifier())],
        task="classification", n_iterations=40, random_state=0).fit(Xtr, ytr)
    print("    selection weights =", [round(w, 2) for w in es.weights_])
    print("    ensemble-selection acc = %.3f"
          % accuracy_score(yte, es.predict(Xte)))

    print("\n[13] JSON payload")
    import json
    payload = build_ensemble_payload(bc, Xte, yte, task="classification")
    print("    keys:", sorted(payload), "| json bytes:", len(json.dumps(payload)))

    print("\n" + "=" * 70)
    print("demo OK")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
