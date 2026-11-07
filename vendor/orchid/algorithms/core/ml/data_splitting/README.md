# Data Splitting

Pure-Python, zero-dependency dataset-splitting & cross-validation toolkit — the
splitter family you'd otherwise reach to scikit-learn for, in a single file with
no NumPy / pandas / scikit-learn. Every splitter is **index-based**
(yields `(train_idx, test_idx)`), so it works with any indexable data and is
trivially reproducible via `random_state`.

## Pick

| Goal | Use |
|---|---|
| One-shot train/test split | `train_test_split` |
| train / val / test | `train_val_test_split` |
| Standard k-fold CV | `KFold` / `RepeatedKFold` |
| Keep class balance per fold | `StratifiedKFold` |
| Don't let a group leak across folds | `GroupKFold` |
| Class balance **and** group integrity | `StratifiedGroupKFold` |
| Many random train/test draws | `ShuffleSplit` / `StratifiedShuffleSplit` / `GroupShuffleSplit` |
| Exhaustive small-data CV | `LeaveOneOut` / `LeavePOut` / `LeaveOneGroupOut` |
| Ordered / time-series data | `TimeSeriesSplit` |
| Leakage-sensitive series (finance) | `PurgedKFold` (purge + embargo) |
| Resampling / out-of-bag | `Bootstrap` |
| Run a model over folds | `cross_validate` / `cross_val_scores` |
| Unbiased model-selection estimate | `nested_cv` |
| Inspect / visualise a split | `split_report` / `build_split_payload` |

## Install

```bash
# No runtime deps — copy the file.
pip install pytest  # tests only
```

## Quick start

```python
from data_splitting_template import train_test_split, StratifiedKFold, cross_validate

X = list(range(100))
y = [0] * 80 + [1] * 20                      # imbalanced labels

# hold-out, stratified so both sides keep the 80/20 ratio
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0)

# cross-validate any model via a scoring callback
def fit_and_score(X_train, y_train, X_test, y_test):
    pred = sum(y_train) / len(y_train)       # toy: predict the mean
    return -sum((pred - t) ** 2 for t in y_test) / len(y_test)   # neg-MSE

result = cross_validate(fit_and_score, X, y, cv=StratifiedKFold(5))
result["mean"], result["std"]                # aggregate score across folds
```

## API

### Hold-out

```python
train_test_split(*arrays, test_size=None, train_size=None, shuffle=True,
                 stratify=None, groups=None, random_state=None)
# -> [a1_train, a1_test, a2_train, a2_test, ...]   (scikit-learn convention)

train_val_test_split(*arrays, val_size=0.15, test_size=0.15, shuffle=True,
                      stratify=None, random_state=None)
# -> [a1_train, a1_val, a1_test, a2_train, ...]
```

* Sizes are **floats** (fractions of n) or **ints** (absolute counts).
* `stratify=labels` preserves class proportions; `groups=ids` keeps whole groups
  on one side. They are mutually exclusive, and both require `shuffle=True`.
* `shuffle=False` gives the leading block as train, trailing as test.

### Cross-validators

All expose the same interface:

```python
cv.split(X, y=None, groups=None)   # -> iterator of (train_idx, test_idx)
cv.get_n_splits(X, y, groups)      # number of folds
```

`X` may be the dataset (anything with `len`) **or** just an integer sample
count.

| Splitter | Notes |
|---|---|
| `KFold(n_splits=5, shuffle=False, random_state=None)` | consecutive folds; optional shuffle |
| `RepeatedKFold(n_splits, n_repeats, random_state)` | repeat shuffled k-fold |
| `StratifiedKFold(n_splits, shuffle, random_state)` | preserves class ratios; needs ≥ `n_splits` of each class |
| `GroupKFold(n_splits)` | a group is never split across train/test |
| `StratifiedGroupKFold(n_splits, shuffle, random_state)` | both constraints, greedy balance |
| `ShuffleSplit(n_splits=10, test_size, train_size, random_state)` | random draws |
| `StratifiedShuffleSplit(...)` / `GroupShuffleSplit(...)` | stratified / grouped draws |
| `LeaveOneOut()` / `LeavePOut(p)` | n / C(n, p) folds |
| `LeaveOneGroupOut()` / `LeavePGroupsOut(n_groups)` | leave whole group(s) out |
| `TimeSeriesSplit(n_splits, test_size, gap, max_train_size)` | forward-chaining; train precedes test |
| `PurgedKFold(n_splits, embargo)` | purge overlapping labels + embargo |
| `Bootstrap(n_splits, n_samples_bootstrap, random_state)` | sample w/ replacement; OOB test |

```python
from data_splitting_template import GroupKFold, TimeSeriesSplit, PurgedKFold

for train_idx, test_idx in GroupKFold(5).split(X, groups=group_ids):
    ...   # no group_id appears in both train_idx and test_idx

for train_idx, test_idx in TimeSeriesSplit(5, gap=2).split(X):
    ...   # max(train_idx) + gap < min(test_idx)

# finance: labels span a horizon -> purge overlapping bars + embargo after test
for train_idx, test_idx in PurgedKFold(5, embargo=0.01).split(
        X, event_times=label_end_index):
    ...
```

### Orchestration

```python
cross_validate(estimator_fn, X, y=None, *, cv=None, groups=None)
# estimator_fn(X_train, y_train, X_test, y_test) -> score
# -> {"scores", "mean", "std", "min", "max", "n_splits", "fold_sizes"}

cross_val_scores(estimator_fn, X, y, *, cv=None, groups=None)  # just the list

nested_cv(fit_select_eval, X, y, *, outer_cv=None, inner_cv=None, groups=None)
# fit_select_eval(X_tr, y_tr, X_te, y_te, inner_cv) -> outer-fold score
```

### Diagnostics

```python
class_distribution(y)                       # {label: count}
split_report(cv, X, y, groups)              # per-fold sizes, balance, leakage flags
build_split_payload(cv, X, y, groups)       # frontend-ready JSON
```

`split_report` returns `train_test_overlap`, `test_coverage`, and
`group_leakage` flags — handy for asserting a custom split is sound.

## Hackathon tips

**Stratified hold-out for an imbalanced target:**
```python
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=0)
```

**Group-aware CV (e.g. multiple rows per user/patient/image):**
```python
scores = cross_val_scores(fit_and_score, X, y, cv=GroupKFold(5),
                          groups=user_ids)
```

**Honest time-series back-test (no peeking ahead, with a gap):**
```python
cv = TimeSeriesSplit(n_splits=6, gap=24)     # e.g. 24-step embargo gap
```

**Financial ML (avoid label-overlap leakage):**
```python
cv = PurgedKFold(6, embargo=0.01)
cv.split(X, event_times=label_end_times)     # per-sample label resolution time
```

**Sanity-check any split is leak-free:**
```python
rep = split_report(GroupKFold(5), X, groups=g)
assert not rep["train_test_overlap"] and not rep["group_leakage"]
```

## Tests / Demo

```bash
pytest _stress_test.py -v          # invariant sweeps over random data
python data_splitting_template.py  # 11-section demo
```

The suite proves the invariants that make a splitter correct, over hundreds of
random instances: every k-fold-family splitter yields a disjoint train/test
partition whose test folds exactly partition the dataset; `StratifiedKFold`
keeps each class's proportion within one sample of the overall rate; `GroupKFold`
/ `StratifiedGroupKFold` never leak a group; `TimeSeriesSplit` always trains
strictly before testing (honouring the gap); `PurgedKFold` never lets a training
label interval overlap a test fold; and `Bootstrap` places every sample in
exactly one of train / out-of-bag with ~63% unique draws.
