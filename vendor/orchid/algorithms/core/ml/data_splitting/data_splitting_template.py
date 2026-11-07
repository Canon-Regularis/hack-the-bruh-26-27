"""
algorithms/core/ml/data_splitting/data_splitting_template.py

Pure-Python, zero-dependency dataset-splitting & cross-validation toolkit.

A self-contained, competition-ready implementation of the splitter family you
would otherwise reach to scikit-learn for: hold-out splits, k-fold and its
stratified / grouped / stratified-grouped variants, Monte-Carlo (shuffle)
splits, exhaustive leave-out schemes, time-series (forward-chaining) splits with
gaps, purged + embargoed k-fold for financial / leakage-sensitive data, and
bootstrap / out-of-bag resampling.  Plus the orchestration around them:
``train_test_split`` / ``train_val_test_split``, ``cross_validate``,
``nested_cv``, and split-quality diagnostics.  No NumPy / pandas / scikit-learn.

Everything is INDEX-BASED (splitters yield ``(train_idx, test_idx)`` lists of
ints), so it works with any indexable data - lists, tuples, your own arrays -
and is trivially reproducible via ``random_state``.

Hold-out
--------
  train_test_split / train_val_test_split

Cross-validators (``.split(X, y, groups)`` -> iterator of (train, test))
------------------------------------------------------------------------
  KFold, RepeatedKFold
  StratifiedKFold
  GroupKFold, StratifiedGroupKFold
  ShuffleSplit, StratifiedShuffleSplit, GroupShuffleSplit
  LeaveOneOut, LeavePOut, LeaveOneGroupOut, LeavePGroupsOut
  TimeSeriesSplit, PurgedKFold
  Bootstrap

Orchestration & diagnostics
---------------------------
  cross_validate, cross_val_scores, nested_cv
  class_distribution, split_report, build_split_payload

Run as a script for an end-to-end demo:
  python data_splitting_template.py
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import (
    Any, Callable, Dict, Hashable, Iterable, Iterator, List, Optional,
    Sequence, Tuple, Union,
)

__version__ = "1.0.0"

__all__ = [
    "Split",
    "train_test_split", "train_val_test_split",
    "KFold", "RepeatedKFold", "StratifiedKFold",
    "GroupKFold", "StratifiedGroupKFold",
    "ShuffleSplit", "StratifiedShuffleSplit", "GroupShuffleSplit",
    "LeaveOneOut", "LeavePOut", "LeaveOneGroupOut", "LeavePGroupsOut",
    "TimeSeriesSplit", "PurgedKFold", "Bootstrap",
    "cross_validate", "cross_val_scores", "nested_cv",
    "class_distribution", "split_report", "build_split_payload",
]

RandomStateLike = Union[None, int, random.Random]


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_random_state(seed: RandomStateLike) -> random.Random:
    """Coerce ``None`` / int / ``random.Random`` into a ``random.Random``."""
    if seed is None:
        return random.Random()
    if isinstance(seed, int):
        return random.Random(seed)
    if isinstance(seed, random.Random):
        return seed
    raise ValueError("random_state must be None, an int, or random.Random, "
                     "got %r" % (seed,))


def _num_samples(X: Any) -> int:
    """Number of samples: ``X`` may be an int (the count itself) or anything
    with ``len``."""
    if isinstance(X, (int,)) and not isinstance(X, bool):
        if X < 0:
            raise ValueError("n_samples must be >= 0, got %r" % (X,))
        return X
    try:
        return len(X)
    except TypeError:
        raise TypeError("cannot infer number of samples from %r" % (type(X),))


def _check_consistent_length(*arrays: Any) -> int:
    lengths = [len(a) for a in arrays if a is not None]
    if not lengths:
        return 0
    if len(set(lengths)) > 1:
        raise ValueError("inconsistent number of samples: %r" % (lengths,))
    return lengths[0]


def class_distribution(y: Sequence[Hashable]) -> Dict[Hashable, int]:
    """Map each class label to its count."""
    return dict(Counter(y))


def _largest_remainder(sizes: Sequence[int], target: int) -> List[int]:
    """Allocate ``target`` units across buckets proportionally to ``sizes``,
    rounding with the largest-remainder method and never exceeding a bucket's
    size.  The total equals ``min(target, sum(sizes))``.

    This is what keeps a stratified TEST set the right size: independent
    per-class rounding can round every small class down to 0 (an empty test),
    whereas largest-remainder distributes the leftover to hit the target."""
    total = sum(sizes)
    target = max(0, min(target, total))
    if total == 0 or target == 0:
        return [0] * len(sizes)
    exact = [s * target / total for s in sizes]
    alloc = [min(int(e), s) for e, s in zip(exact, sizes)]
    assigned = sum(alloc)
    # hand out the remaining units to the largest fractional remainders, only to
    # buckets that still have capacity
    order = sorted(range(len(sizes)), key=lambda i: -(exact[i] - int(exact[i])))
    while assigned < target:
        progressed = False
        for i in order:
            if assigned >= target:
                break
            if alloc[i] < sizes[i]:
                alloc[i] += 1
                assigned += 1
                progressed = True
        if not progressed:
            break
    return alloc


def _check_min_splits(n_splits: int, name: str = "n_splits") -> None:
    """Structural check applied at construction (independent of sample count)."""
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) \
            or n_splits < 2:
        raise ValueError("%s must be an int >= 2, got %r" % (name, n_splits))


def _validate_n_splits(n_splits: int, n_samples: int, name: str = "n_splits"
                       ) -> None:
    _check_min_splits(n_splits, name)
    if n_splits > n_samples:
        raise ValueError(
            "%s=%d cannot be greater than the number of samples=%d"
            % (name, n_splits, n_samples))


def _as_fraction_count(size: Union[int, float, None], n: int,
                       complement: Union[int, float, None],
                       default_test: float = 0.25) -> Tuple[int, int]:
    """Resolve (n_train, n_test) from possibly-fractional sizes.  Mirrors the
    train_test_split convention: ints are absolute counts, floats are
    fractions of ``n``."""
    def to_count(s: Union[int, float, None]) -> Optional[int]:
        if s is None:
            return None
        if isinstance(s, float):
            if not 0.0 < s < 1.0:
                raise ValueError("float size must be in (0, 1), got %r" % (s,))
            return int(round(s * n))
        if isinstance(s, int):
            if not 0 <= s <= n:
                raise ValueError("int size must be in [0, %d], got %r"
                                 % (n, s))
            return s
        raise ValueError("size must be int, float or None, got %r" % (s,))

    n_test = to_count(size)
    n_train = to_count(complement)
    if n_test is None and n_train is None:
        n_test = int(round(default_test * n))
        n_train = n - n_test
    elif n_test is None:
        n_test = n - n_train
    elif n_train is None:
        n_train = n - n_test
    elif isinstance(size, float) and isinstance(complement, float):
        total = size + complement
        if total >= 1.0 - 1e-9 and total <= 1.0 + 1e-9:
            # fractions sum to 1.0 -> no leftover intended; make train the exact
            # complement so independent rounding can't drop or double-count a
            # sample (e.g. 0.5 + 0.5 on odd n rounds each to ceil(n/2)).
            n_train = n - n_test
        elif total < 1.0 and n_train + n_test > n:
            # genuine subsample whose independent rounding overshot: absorb drift
            n_train = n - n_test
        # total > 1.0 falls through to the exceeds-n_samples error below
    if n_train + n_test > n:
        raise ValueError("train (%d) + test (%d) exceeds n_samples (%d)"
                         % (n_train, n_test, n))
    if n_train <= 0 or n_test <= 0:
        raise ValueError("resulting split is empty: train=%d test=%d"
                         % (n_train, n_test))
    return n_train, n_test


# ─────────────────────────────────────────────────────────────────────────────
# Split result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Split:
    """A single train/test (optionally with validation) index partition."""
    train: Tuple[int, ...]
    test: Tuple[int, ...]
    validation: Tuple[int, ...] = ()

    def __iter__(self) -> Iterator[Tuple[int, ...]]:
        yield self.train
        yield self.test

    @property
    def sizes(self) -> Dict[str, int]:
        return {"train": len(self.train), "test": len(self.test),
                "validation": len(self.validation)}


# ─────────────────────────────────────────────────────────────────────────────
# Base cross-validator
# ─────────────────────────────────────────────────────────────────────────────

class BaseCrossValidator:
    """Base class: subclasses implement ``_iter_test_indices`` (or override
    ``split``) and ``get_n_splits``."""

    def split(self, X: Any, y: Optional[Sequence] = None,
              groups: Optional[Sequence] = None
              ) -> Iterator[Tuple[List[int], List[int]]]:
        n = _num_samples(X)
        indices = list(range(n))
        for test in self._iter_test_indices(n, y, groups):
            test_set = set(test)
            train = [i for i in indices if i not in test_set]
            yield train, list(test)

    def _iter_test_indices(self, n: int, y, groups) -> Iterator[List[int]]:
        raise NotImplementedError

    def get_n_splits(self, X: Any = None, y: Optional[Sequence] = None,
                     groups: Optional[Sequence] = None) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        params = ", ".join("%s=%r" % (k, v) for k, v in sorted(vars(self).items())
                           if not k.startswith("_"))
        return "%s(%s)" % (type(self).__name__, params)


# ─────────────────────────────────────────────────────────────────────────────
# K-fold
# ─────────────────────────────────────────────────────────────────────────────

class KFold(BaseCrossValidator):
    """K-fold cross-validation: partition the samples into ``n_splits``
    consecutive folds; each fold serves once as the test set."""

    def __init__(self, n_splits: int = 5, *, shuffle: bool = False,
                 random_state: RandomStateLike = None) -> None:
        _check_min_splits(n_splits)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        if not shuffle and random_state is not None:
            raise ValueError("random_state has no effect when shuffle=False")

    def _iter_test_indices(self, n, y, groups):
        _validate_n_splits(self.n_splits, n)
        indices = list(range(n))
        if self.shuffle:
            _check_random_state(self.random_state).shuffle(indices)
        base, rem = divmod(n, self.n_splits)
        start = 0
        for f in range(self.n_splits):
            size = base + (1 if f < rem else 0)
            yield indices[start:start + size]
            start += size

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class RepeatedKFold(BaseCrossValidator):
    """Repeat :class:`KFold` ``n_repeats`` times with different shuffles."""

    def __init__(self, n_splits: int = 5, *, n_repeats: int = 10,
                 random_state: RandomStateLike = None) -> None:
        if n_repeats < 1:
            raise ValueError("n_repeats must be >= 1, got %r" % (n_repeats,))
        _check_min_splits(n_splits)
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        rng = _check_random_state(self.random_state)
        for _ in range(self.n_repeats):
            kf = KFold(self.n_splits, shuffle=True,
                       random_state=rng.randrange(2 ** 31))
            for tr, te in kf.split(X, y, groups):
                yield tr, te

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits * self.n_repeats


class StratifiedKFold(BaseCrossValidator):
    """K-fold that preserves each class's proportion in every fold.  Requires
    every class to have at least ``n_splits`` members."""

    def __init__(self, n_splits: int = 5, *, shuffle: bool = False,
                 random_state: RandomStateLike = None) -> None:
        _check_min_splits(n_splits)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        if not shuffle and random_state is not None:
            raise ValueError("random_state has no effect when shuffle=False")

    def _iter_test_indices(self, n, y, groups):
        if y is None:
            raise ValueError("StratifiedKFold requires y")
        y = list(y)
        if len(y) != n:
            raise ValueError("y has wrong length")
        _validate_n_splits(self.n_splits, n)
        by_class: Dict[Hashable, List[int]] = defaultdict(list)
        for i, label in enumerate(y):
            by_class[label].append(i)
        min_count = min(len(v) for v in by_class.values())
        if self.n_splits > min_count:
            raise ValueError(
                "n_splits=%d exceeds the smallest class count=%d; some fold "
                "would contain no samples of that class"
                % (self.n_splits, min_count))
        rng = _check_random_state(self.random_state)
        fold_of = [-1] * n
        start = 0
        # deterministic class order (by repr) so shuffling only affects members
        for label in sorted(by_class, key=repr):
            members = list(by_class[label])
            if self.shuffle:
                rng.shuffle(members)
            nc = len(members)
            base, rem = divmod(nc, self.n_splits)
            # spread the 'rem' larger folds across folds, rotating per class so
            # no single fold accumulates all the remainders (keeps folds even)
            sizes = [base + (1 if ((f - start) % self.n_splits) < rem else 0)
                     for f in range(self.n_splits)]
            pos = 0
            for f in range(self.n_splits):
                for _ in range(sizes[f]):
                    fold_of[members[pos]] = f
                    pos += 1
            start = (start + rem) % self.n_splits
        for f in range(self.n_splits):
            yield [i for i in range(n) if fold_of[i] == f]

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ─────────────────────────────────────────────────────────────────────────────
# Group-aware
# ─────────────────────────────────────────────────────────────────────────────

class GroupKFold(BaseCrossValidator):
    """K-fold where the same group never appears in both train and test.
    Groups are distributed to balance the number of samples per fold."""

    def __init__(self, n_splits: int = 5) -> None:
        _check_min_splits(n_splits)
        self.n_splits = n_splits

    def _iter_test_indices(self, n, y, groups):
        if groups is None:
            raise ValueError("GroupKFold requires groups")
        groups = list(groups)
        if len(groups) != n:
            raise ValueError("groups has wrong length")
        members: Dict[Hashable, List[int]] = defaultdict(list)
        for i, g in enumerate(groups):
            members[g].append(i)
        n_groups = len(members)
        if self.n_splits > n_groups:
            raise ValueError("n_splits=%d cannot exceed the number of groups=%d"
                             % (self.n_splits, n_groups))
        fold_sizes = [0] * self.n_splits
        group_to_fold: Dict[Hashable, int] = {}
        # largest groups first, each to the currently-lightest fold
        for g in sorted(members, key=lambda g: (-len(members[g]), repr(g))):
            f = min(range(self.n_splits), key=lambda f: fold_sizes[f])
            group_to_fold[g] = f
            fold_sizes[f] += len(members[g])
        for f in range(self.n_splits):
            yield [i for g, idxs in members.items() if group_to_fold[g] == f
                   for i in idxs]

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class StratifiedGroupKFold(BaseCrossValidator):
    """K-fold that keeps groups intact AND preserves class proportions, via a
    greedy assignment that minimises per-fold class-count imbalance."""

    def __init__(self, n_splits: int = 5, *, shuffle: bool = False,
                 random_state: RandomStateLike = None) -> None:
        _check_min_splits(n_splits)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _iter_test_indices(self, n, y, groups):
        if y is None or groups is None:
            raise ValueError("StratifiedGroupKFold requires y and groups")
        y = list(y)
        groups = list(groups)
        classes = sorted(set(y), key=repr)
        cls_idx = {c: i for i, c in enumerate(classes)}
        members: Dict[Hashable, List[int]] = defaultdict(list)
        for i, g in enumerate(groups):
            members[g].append(i)
        if self.n_splits > len(members):
            raise ValueError("n_splits=%d cannot exceed the number of groups=%d"
                             % (self.n_splits, len(members)))
        # per-group class counts
        g_counts: Dict[Hashable, List[int]] = {}
        for g, idxs in members.items():
            c = [0] * len(classes)
            for i in idxs:
                c[cls_idx[y[i]]] += 1
            g_counts[g] = c
        total = [0] * len(classes)
        for c in g_counts.values():
            for k in range(len(classes)):
                total[k] += c[k]
        ideal = [t / self.n_splits for t in total]
        order = sorted(members, key=lambda g: (-sum(g_counts[g]), repr(g)))
        if self.shuffle:
            rng = _check_random_state(self.random_state)
            rng.shuffle(order)
            order.sort(key=lambda g: -sum(g_counts[g]))   # stable-ish on size
        fold_counts = [[0] * len(classes) for _ in range(self.n_splits)]
        fold_n_groups = [0] * self.n_splits
        group_to_fold: Dict[Hashable, int] = {}
        n_total = len(order)
        for idx, g in enumerate(order):
            gc = g_counts[g]
            remaining = n_total - idx                  # groups still to place
            empty = [f for f in range(self.n_splits) if fold_n_groups[f] == 0]
            # guarantee no empty test fold: once only as many groups remain as
            # there are still-empty folds, every remaining group must seed a
            # distinct empty fold (n_groups >= n_splits is validated)
            candidates = empty if remaining <= len(empty) and empty \
                else range(self.n_splits)
            best_f, best_cost = None, math.inf
            for f in candidates:
                # marginal change in total squared class-count deviation from
                # adding this group to fold f: ((c+g-ideal)^2 - (c-ideal)^2).
                # The *marginal* form (not the absolute post-state) makes
                # under-filled folds attractive - an empty fold has a large
                # (0-ideal)^2 to shed; the absolute-cost variant starved folds
                # and left test folds EMPTY.
                cost = 0.0
                for k in range(len(classes)):
                    c = fold_counts[f][k]
                    cost += (c + gc[k] - ideal[k]) ** 2 - (c - ideal[k]) ** 2
                if cost < best_cost:
                    best_cost, best_f = cost, f
            group_to_fold[g] = best_f
            fold_n_groups[best_f] += 1
            for k in range(len(classes)):
                fold_counts[best_f][k] += gc[k]
        for f in range(self.n_splits):
            yield [i for g, idxs in members.items() if group_to_fold[g] == f
                   for i in idxs]

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ─────────────────────────────────────────────────────────────────────────────
# Monte-Carlo / shuffle splits
# ─────────────────────────────────────────────────────────────────────────────

class ShuffleSplit(BaseCrossValidator):
    """Random permutation cross-validation: ``n_splits`` independent random
    train/test partitions (samples may recur across different splits)."""

    def __init__(self, n_splits: int = 10, *,
                 test_size: Union[int, float, None] = None,
                 train_size: Union[int, float, None] = None,
                 random_state: RandomStateLike = None) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1, got %r" % (n_splits,))
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        n = _num_samples(X)
        n_train, n_test = _as_fraction_count(self.test_size, n, self.train_size)
        rng = _check_random_state(self.random_state)
        for _ in range(self.n_splits):
            perm = list(range(n))
            rng.shuffle(perm)
            yield perm[n_test:n_test + n_train], perm[:n_test]

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class StratifiedShuffleSplit(BaseCrossValidator):
    """Shuffle-split preserving class proportions in each train/test draw."""

    def __init__(self, n_splits: int = 10, *,
                 test_size: Union[int, float, None] = None,
                 train_size: Union[int, float, None] = None,
                 random_state: RandomStateLike = None) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1, got %r" % (n_splits,))
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        if y is None:
            raise ValueError("StratifiedShuffleSplit requires y")
        n = _num_samples(X)
        y = list(y)
        n_train, n_test = _as_fraction_count(self.test_size, n, self.train_size)
        by_class: Dict[Hashable, List[int]] = defaultdict(list)
        for i, label in enumerate(y):
            by_class[label].append(i)
        labels = sorted(by_class, key=repr)
        rng = _check_random_state(self.random_state)
        for _ in range(self.n_splits):
            shuffled = []
            for label in labels:
                m = list(by_class[label])
                rng.shuffle(m)
                shuffled.append(m)
            sizes = [len(m) for m in shuffled]
            # largest-remainder so the TEST side hits n_test (never collapses to
            # empty even with many tiny classes); train drawn from the leftover
            test_alloc = _largest_remainder(sizes, n_test)
            remaining = [s - t for s, t in zip(sizes, test_alloc)]
            train_alloc = _largest_remainder(remaining, n_train)
            test: List[int] = []
            train: List[int] = []
            for m, kt, ktr in zip(shuffled, test_alloc, train_alloc):
                test.extend(m[:kt])
                train.extend(m[kt:kt + ktr])
            yield sorted(train), sorted(test)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class GroupShuffleSplit(BaseCrossValidator):
    """Shuffle-split at the group level: whole groups go to train or test."""

    def __init__(self, n_splits: int = 5, *,
                 test_size: Union[int, float, None] = None,
                 train_size: Union[int, float, None] = None,
                 random_state: RandomStateLike = None) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1, got %r" % (n_splits,))
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("GroupShuffleSplit requires groups")
        n = _num_samples(X)
        groups = list(groups)
        members: Dict[Hashable, List[int]] = defaultdict(list)
        for i, g in enumerate(groups):
            members[g].append(i)
        unique = sorted(members, key=repr)
        if len(unique) < 2:
            raise ValueError(
                "GroupShuffleSplit needs at least 2 groups to split, got %d"
                % (len(unique),))
        n_train, n_test = _as_fraction_count(self.test_size, n, self.train_size)
        rng = _check_random_state(self.random_state)
        for _ in range(self.n_splits):
            order = list(unique)
            rng.shuffle(order)
            test: List[int] = []
            train: List[int] = []
            assigned = set()
            for g in order:
                if len(test) >= n_test:
                    break
                # whole groups are indivisible, so reserve at least one group
                # for train: never let the test side consume every group
                if len(assigned) >= len(order) - 1:
                    break
                assigned.add(g)
                test.extend(members[g])
            # fill train from the remaining groups up to n_train; any leftover
            # group is left out (so train_size + test_size < 1 is honoured)
            for g in order:
                if g in assigned:
                    continue
                if len(train) >= n_train:
                    break
                assigned.add(g)
                train.extend(members[g])
            yield sorted(train), sorted(test)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ─────────────────────────────────────────────────────────────────────────────
# Exhaustive leave-out
# ─────────────────────────────────────────────────────────────────────────────

class LeaveOneOut(BaseCrossValidator):
    """Each sample is the test set exactly once (n folds)."""

    def _iter_test_indices(self, n, y, groups):
        if n < 2:
            raise ValueError("LeaveOneOut needs at least 2 samples")
        for i in range(n):
            yield [i]

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return _num_samples(X)


class LeavePOut(BaseCrossValidator):
    """Every size-``p`` subset of samples is a test set (C(n, p) folds)."""

    def __init__(self, p: int = 2) -> None:
        if p < 1:
            raise ValueError("p must be >= 1, got %r" % (p,))
        self.p = p

    def _iter_test_indices(self, n, y, groups):
        if self.p >= n:
            raise ValueError("p=%d must be < n_samples=%d" % (self.p, n))
        for combo in combinations(range(n), self.p):
            yield list(combo)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        n = _num_samples(X)
        return math.comb(n, self.p)


class LeaveOneGroupOut(BaseCrossValidator):
    """Each group is the test set exactly once."""

    def _iter_test_indices(self, n, y, groups):
        if groups is None:
            raise ValueError("LeaveOneGroupOut requires groups")
        groups = list(groups)
        members: Dict[Hashable, List[int]] = defaultdict(list)
        for i, g in enumerate(groups):
            members[g].append(i)
        if len(members) < 2:
            raise ValueError("LeaveOneGroupOut needs at least 2 groups")
        for g in sorted(members, key=repr):
            yield members[g]

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if groups is None:
            raise ValueError("LeaveOneGroupOut requires groups")
        return len(set(groups))


class LeavePGroupsOut(BaseCrossValidator):
    """Every size-``p`` subset of groups is a test set."""

    def __init__(self, n_groups: int = 2) -> None:
        if n_groups < 1:
            raise ValueError("n_groups must be >= 1, got %r" % (n_groups,))
        self.n_groups = n_groups

    def _iter_test_indices(self, n, y, groups):
        if groups is None:
            raise ValueError("LeavePGroupsOut requires groups")
        groups = list(groups)
        members: Dict[Hashable, List[int]] = defaultdict(list)
        for i, g in enumerate(groups):
            members[g].append(i)
        unique = sorted(members, key=repr)
        if self.n_groups >= len(unique):
            raise ValueError("n_groups=%d must be < number of groups=%d"
                             % (self.n_groups, len(unique)))
        for combo in combinations(unique, self.n_groups):
            yield [i for g in combo for i in members[g]]

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if groups is None:
            raise ValueError("LeavePGroupsOut requires groups")
        return math.comb(len(set(groups)), self.n_groups)


# ─────────────────────────────────────────────────────────────────────────────
# Time-series
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesSplit(BaseCrossValidator):
    """Forward-chaining cross-validation for ordered data: the test fold always
    comes strictly after its training data.  Supports an optional ``gap``
    between train and test and a rolling ``max_train_size``."""

    def __init__(self, n_splits: int = 5, *,
                 test_size: Optional[int] = None, gap: int = 0,
                 max_train_size: Optional[int] = None) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2, got %r" % (n_splits,))
        if gap < 0:
            raise ValueError("gap must be >= 0, got %r" % (gap,))
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        n = _num_samples(X)
        n_folds = self.n_splits + 1
        test_size = (self.test_size if self.test_size is not None
                     else n // n_folds)
        if test_size <= 0:
            raise ValueError("too few samples (%d) for n_splits=%d"
                             % (n, self.n_splits))
        first_test = n - self.n_splits * test_size
        if first_test - self.gap <= 0:
            raise ValueError(
                "not enough samples for %d splits of size %d with gap %d"
                % (self.n_splits, test_size, self.gap))
        for i in range(self.n_splits):
            test_start = first_test + i * test_size
            train_end = test_start - self.gap
            train_start = (max(0, train_end - self.max_train_size)
                           if self.max_train_size else 0)
            yield (list(range(train_start, train_end)),
                   list(range(test_start, test_start + test_size)))

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class PurgedKFold(BaseCrossValidator):
    """K-fold for data whose labels span an interval of time (e.g. financial
    bars labelled over a horizon).  Training samples whose label interval
    overlaps a test fold are *purged*, and an *embargo* drops training samples
    immediately after each test fold - eliminating look-ahead leakage
    (Lopez de Prado, *Advances in Financial Machine Learning*).

    ``event_times[i]`` is the index at which sample ``i``'s label is resolved
    (default: ``i`` itself, i.e. labels resolve immediately)."""

    def __init__(self, n_splits: int = 5, *, embargo: float = 0.0) -> None:
        if embargo < 0:
            raise ValueError("embargo must be >= 0, got %r" % (embargo,))
        _check_min_splits(n_splits)
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, X, y=None, groups=None,
              event_times: Optional[Sequence[int]] = None):
        n = _num_samples(X)
        _validate_n_splits(self.n_splits, n)
        if event_times is None:
            event_times = list(range(n))
        else:
            event_times = list(event_times)
            if len(event_times) != n:
                raise ValueError("event_times has wrong length")
            if any(event_times[i] < i for i in range(n)):
                raise ValueError("event_times[i] must be >= i (labels cannot "
                                 "resolve before the sample occurs)")
        embargo_len = int(round(n * self.embargo))
        base, rem = divmod(n, self.n_splits)
        start = 0
        bounds = []
        for f in range(self.n_splits):
            size = base + (1 if f < rem else 0)
            bounds.append((start, start + size))
            start += size
        for (t0, t1) in bounds:
            test = list(range(t0, t1))
            test_max_event = max(event_times[i] for i in test)
            embargo_end = min(n, test_max_event + 1 + embargo_len)
            train = []
            for i in range(n):
                if t0 <= i < t1:
                    continue
                # purge: sample whose label interval [i, event_times[i]]
                # overlaps the test interval [t0, t1)
                if event_times[i] >= t0 and i < t1:
                    continue
                # embargo: drop samples just after the test fold
                if t1 <= i < embargo_end:
                    continue
                train.append(i)
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

class Bootstrap(BaseCrossValidator):
    """Bootstrap resampling: each split draws ``n`` training indices with
    replacement; the unsampled (out-of-bag) indices form the test set."""

    def __init__(self, n_splits: int = 10, *,
                 n_samples_bootstrap: Optional[int] = None,
                 random_state: RandomStateLike = None) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1, got %r" % (n_splits,))
        if n_samples_bootstrap is not None and n_samples_bootstrap < 1:
            raise ValueError("n_samples_bootstrap must be >= 1, got %r"
                             % (n_samples_bootstrap,))
        self.n_splits = n_splits
        self.n_samples_bootstrap = n_samples_bootstrap
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        n = _num_samples(X)
        # don't use `or` here: a (now-rejected) 0 would silently fall back to n
        m = n if self.n_samples_bootstrap is None else self.n_samples_bootstrap
        rng = _check_random_state(self.random_state)
        for _ in range(self.n_splits):
            train = [rng.randrange(n) for _ in range(m)]
            oob = sorted(set(range(n)) - set(train))
            yield train, oob

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ─────────────────────────────────────────────────────────────────────────────
# Hold-out splitters
# ─────────────────────────────────────────────────────────────────────────────

def train_test_split(*arrays: Sequence,
                     test_size: Union[int, float, None] = None,
                     train_size: Union[int, float, None] = None,
                     shuffle: bool = True,
                     stratify: Optional[Sequence] = None,
                     groups: Optional[Sequence] = None,
                     random_state: RandomStateLike = None) -> List[List]:
    """Split one or more equally-sized arrays into train/test subsets.

    Returns ``[a1_train, a1_test, a2_train, a2_test, ...]`` (the scikit-learn
    convention).  ``stratify`` preserves class proportions; ``groups`` keeps
    whole groups on one side; the two are mutually exclusive.  With
    ``shuffle=False`` the split is the leading/trailing contiguous block."""
    if not arrays:
        raise ValueError("at least one array is required")
    n = _check_consistent_length(*arrays)
    if stratify is not None and groups is not None:
        raise ValueError("stratify and groups are mutually exclusive")
    if not shuffle and (stratify is not None or groups is not None):
        raise ValueError("shuffle=False is incompatible with stratify/groups")

    if stratify is not None:
        if len(stratify) != n:
            raise ValueError("stratify has wrong length")
        tr, te = next(StratifiedShuffleSplit(
            1, test_size=test_size, train_size=train_size,
            random_state=random_state).split(range(n), stratify))
    elif groups is not None:
        if len(groups) != n:
            raise ValueError("groups has wrong length")
        tr, te = next(GroupShuffleSplit(
            1, test_size=test_size, train_size=train_size,
            random_state=random_state).split(range(n), groups=groups))
    else:
        n_train, n_test = _as_fraction_count(test_size, n, train_size)
        idx = list(range(n))
        if shuffle:
            _check_random_state(random_state).shuffle(idx)
            tr, te = idx[n_test:n_test + n_train], idx[:n_test]
        else:
            tr, te = idx[:n_train], idx[n_train:n_train + n_test]

    out: List[List] = []
    for arr in arrays:
        out.append([arr[i] for i in tr])
        out.append([arr[i] for i in te])
    return out


def train_val_test_split(*arrays: Sequence,
                         val_size: Union[int, float] = 0.15,
                         test_size: Union[int, float] = 0.15,
                         shuffle: bool = True,
                         stratify: Optional[Sequence] = None,
                         random_state: RandomStateLike = None) -> List[List]:
    """Three-way split into train / validation / test.  Returns
    ``[a1_train, a1_val, a1_test, a2_train, ...]``.  Fractions are taken with
    respect to the whole dataset."""
    if not arrays:
        raise ValueError("at least one array is required")
    n = _check_consistent_length(*arrays)
    n_val = val_size if isinstance(val_size, int) else int(round(val_size * n))
    n_test = test_size if isinstance(test_size, int) else int(round(test_size * n))
    if n_val + n_test >= n:
        raise ValueError("val + test sizes leave no training data")
    rng = _check_random_state(random_state)
    # first peel off test, then val from the remainder, stratifying if asked
    first = train_test_split(*arrays, range(n), test_size=n_test,
                             shuffle=shuffle, stratify=stratify,
                             random_state=rng)
    # train_test_split returned interleaved; the last pair is the index array
    rest_arrays = first[0::2]            # *_train for each array (+ idx_train)
    test_arrays = first[1::2]            # *_test for each array  (+ idx_test)
    rest_idx = rest_arrays[-1]
    rest_data = rest_arrays[:-1]
    test_data = test_arrays[:-1]
    strat_rest = ([stratify[i] for i in rest_idx]
                  if stratify is not None else None)
    # val fraction relative to the remaining pool
    n_remaining = len(rest_idx)
    val_frac_of_rest = n_val / n_remaining
    second = train_test_split(*rest_data, test_size=val_frac_of_rest,
                              shuffle=shuffle, stratify=strat_rest,
                              random_state=rng)
    out: List[List] = []
    for a in range(len(arrays)):
        out.append(second[2 * a])        # train
        out.append(second[2 * a + 1])    # val
        out.append(test_data[a])         # test
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate(estimator_fn: Callable[[List, List, List, List], float],
                   X: Sequence, y: Optional[Sequence] = None, *,
                   cv: Optional[BaseCrossValidator] = None,
                   groups: Optional[Sequence] = None
                   ) -> Dict[str, Any]:
    """Run ``estimator_fn(X_train, y_train, X_test, y_test) -> score`` over each
    fold of ``cv`` (default 5-fold KFold) and aggregate the scores."""
    cv = cv or KFold(5)
    scores: List[float] = []
    fold_sizes: List[Dict[str, int]] = []
    for train, test in cv.split(X, y, groups):
        Xtr = [X[i] for i in train]
        Xte = [X[i] for i in test]
        ytr = [y[i] for i in train] if y is not None else None
        yte = [y[i] for i in test] if y is not None else None
        scores.append(float(estimator_fn(Xtr, ytr, Xte, yte)))
        fold_sizes.append({"train": len(train), "test": len(test)})
    mean = sum(scores) / len(scores) if scores else 0.0
    var = (sum((s - mean) ** 2 for s in scores) / len(scores)
           if scores else 0.0)
    return {
        "scores": scores,
        "mean": mean,
        "std": math.sqrt(var),
        "min": min(scores) if scores else 0.0,
        "max": max(scores) if scores else 0.0,
        "n_splits": len(scores),
        "fold_sizes": fold_sizes,
    }


def cross_val_scores(estimator_fn, X, y=None, *, cv=None, groups=None
                     ) -> List[float]:
    """Just the per-fold scores from :func:`cross_validate`."""
    return cross_validate(estimator_fn, X, y, cv=cv, groups=groups)["scores"]


def nested_cv(fit_select_eval: Callable[[List, List, List, List,
                                         BaseCrossValidator], float],
              X: Sequence, y: Optional[Sequence] = None, *,
              outer_cv: Optional[BaseCrossValidator] = None,
              inner_cv: Optional[BaseCrossValidator] = None,
              groups: Optional[Sequence] = None) -> Dict[str, Any]:
    """Nested cross-validation: for each outer fold, hand the training portion
    plus an ``inner_cv`` to ``fit_select_eval`` (which should tune on the inner
    folds and return the score on the outer test fold).  Returns the
    distribution of outer scores - an (approximately) unbiased estimate of
    generalisation under model selection."""
    outer_cv = outer_cv or KFold(5)
    inner_cv = inner_cv or KFold(3)
    outer_scores: List[float] = []
    for train, test in outer_cv.split(X, y, groups):
        Xtr = [X[i] for i in train]
        Xte = [X[i] for i in test]
        ytr = [y[i] for i in train] if y is not None else None
        yte = [y[i] for i in test] if y is not None else None
        outer_scores.append(float(fit_select_eval(Xtr, ytr, Xte, yte,
                                                   inner_cv)))
    mean = sum(outer_scores) / len(outer_scores) if outer_scores else 0.0
    var = (sum((s - mean) ** 2 for s in outer_scores) / len(outer_scores)
           if outer_scores else 0.0)
    return {"outer_scores": outer_scores, "mean": mean, "std": math.sqrt(var),
            "n_splits": len(outer_scores)}


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics / reporting
# ─────────────────────────────────────────────────────────────────────────────

def _json_num(x: Any) -> Any:
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return None
    return x


def split_report(cv: BaseCrossValidator, X: Any,
                 y: Optional[Sequence] = None,
                 groups: Optional[Sequence] = None) -> Dict[str, Any]:
    """Summarise a cross-validator's folds: per-fold train/test sizes, leakage
    checks, and (if ``y`` is given) the class balance of each test fold."""
    n = _num_samples(X)
    y = list(y) if y is not None else None
    groups = list(groups) if groups is not None else None
    folds = list(cv.split(X, y, groups))
    overall = class_distribution(y) if y is not None else None
    fold_info: List[Dict[str, Any]] = []
    test_union: set = set()
    any_overlap = False
    group_leak = False
    for fi, (train, test) in enumerate(folds):
        ts = set(train)
        es = set(test)
        if ts & es:
            any_overlap = True
        test_union |= es
        info: Dict[str, Any] = {
            "fold": fi, "n_train": len(train), "n_test": len(test),
        }
        if y is not None:
            info["test_class_distribution"] = {
                str(k): v for k, v in
                class_distribution([y[i] for i in test]).items()}
        if groups is not None:
            tr_g = {groups[i] for i in train}
            te_g = {groups[i] for i in test}
            if tr_g & te_g:
                group_leak = True
            info["n_test_groups"] = len(te_g)
        fold_info.append(info)
    return {
        "splitter": type(cv).__name__,
        "n_samples": n,
        "n_splits": len(folds),
        "overall_class_distribution": (
            {str(k): v for k, v in overall.items()} if overall else None),
        "folds": fold_info,
        "train_test_overlap": any_overlap,
        "test_coverage": len(test_union),       # distinct samples ever tested
        "group_leakage": group_leak if groups is not None else None,
    }


def build_split_payload(cv: BaseCrossValidator, X: Any,
                        y: Optional[Sequence] = None,
                        groups: Optional[Sequence] = None) -> Dict[str, Any]:
    """Frontend-ready JSON describing the fold structure (sizes + balance)."""
    rep = split_report(cv, X, y, groups)
    bars = [{"fold": f["fold"], "train": _json_num(f["n_train"]),
             "test": _json_num(f["n_test"])} for f in rep["folds"]]
    return {
        "splitter": rep["splitter"],
        "numSamples": rep["n_samples"],
        "numSplits": rep["n_splits"],
        "folds": bars,
        "valid": (not rep["train_test_overlap"]
                  and not (rep["group_leakage"] or False)),
        "classBalance": rep["overall_class_distribution"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    print("=" * 70)
    print("DATA SPLITTING TEMPLATE - demo")
    print("=" * 70)

    n = 20
    X = list(range(n))
    y = [0] * 14 + [1] * 6                       # imbalanced binary target
    groups = [i // 3 for i in range(n)]          # 7 groups of ~3

    print("\n[1] train_test_split (80/20, shuffled, seeded)")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    print("    train n=%d test n=%d" % (len(Xtr), len(Xte)))

    print("\n[2] stratified train_test_split preserves the 70/30 class ratio")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y,
                                          random_state=0)
    print("    train classes:", class_distribution(ytr),
          "test classes:", class_distribution(yte))

    print("\n[3] train/val/test 70/15/15")
    parts = train_val_test_split(X, val_size=0.15, test_size=0.15,
                                 random_state=0)
    print("    sizes:", [len(p) for p in parts])

    print("\n[4] KFold(5) vs StratifiedKFold(5) - minority-class per test fold")
    for cv in (KFold(5, shuffle=True, random_state=0), StratifiedKFold(5)):
        mins = []
        for _, test in cv.split(X, y):
            mins.append(sum(1 for i in test if y[i] == 1))
        print("    %-18s minority counts per fold: %s"
              % (type(cv).__name__, mins))

    print("\n[5] GroupKFold(3) - no group spans train/test")
    rep = split_report(GroupKFold(3), X, y, groups)
    print("    group leakage:", rep["group_leakage"],
          "| fold test sizes:", [f["n_test"] for f in rep["folds"]])

    print("\n[6] TimeSeriesSplit(4, gap=1) - train always precedes test")
    for tr, te in TimeSeriesSplit(4, gap=1).split(X):
        print("    train=[..%d] test=%s" % (tr[-1] if tr else -1, te))

    print("\n[7] PurgedKFold(4, embargo=0.1) for leakage-sensitive series")
    et = [min(n - 1, i + 2) for i in range(n)]   # labels span 2 bars
    for tr, te in PurgedKFold(4, embargo=0.1).split(X, event_times=et):
        print("    test=%s -> train n=%d (purged+embargoed)" % (te, len(tr)))

    print("\n[8] Bootstrap(3) with out-of-bag test sets")
    for tr, oob in Bootstrap(3, random_state=0).split(X):
        print("    bootstrap n=%d unique=%d  oob n=%d"
              % (len(tr), len(set(tr)), len(oob)))

    print("\n[9] cross_validate with a toy 'mean-predictor' scorer")
    def scorer(Xtr, ytr, Xte, yte):
        pred = sum(ytr) / len(ytr)
        return -sum((pred - t) ** 2 for t in yte) / len(yte)   # neg MSE
    res = cross_validate(scorer, X, y, cv=StratifiedKFold(5))
    print("    neg-MSE mean=%.4f std=%.4f over %d folds"
          % (res["mean"], res["std"], res["n_splits"]))

    print("\n[10] LeaveOneOut / LeavePOut fold counts")
    print("    LeaveOneOut:", LeaveOneOut().get_n_splits(X),
          "| LeavePOut(2):", LeavePOut(2).get_n_splits(X),
          "| LeaveOneGroupOut:", LeaveOneGroupOut().get_n_splits(groups=groups))

    print("\n[11] JSON split payload")
    import json
    payload = build_split_payload(StratifiedKFold(5), X, y)
    print("    keys:", sorted(payload),
          "| valid:", payload["valid"],
          "| json bytes:", len(json.dumps(payload)))

    print("\n" + "=" * 70)
    print("demo OK")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
