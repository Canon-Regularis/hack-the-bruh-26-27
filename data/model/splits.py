"""Validation splits: hold out whole streets, or hold out the latest hours.

Both use orchid's splitters and yield (train rows, test rows) as row positions.
"""

from engine.orchid import data_splitting


def street_groups(table):
    """Return one group per row: its street, or its own edge if it has no name.

    Unnamed edges are not one street, so they must not share a group.
    """
    return [
        name if name else f"edge:{edge_id}"
        for name, edge_id in zip(table["street_name"], table["edge_id"])
    ]


def street_folds(table, n_splits=5):
    """Yield folds where no street is in both train and test."""
    splitter = data_splitting.GroupKFold(n_splits=n_splits)
    yield from splitter.split(list(range(len(table))), groups=street_groups(table))


def time_folds(table, n_splits=3):
    """Yield folds where every test bin comes after every training bin.

    Splits on whole bins, so rows from one bin are never on both sides.
    """
    bins = sorted(table["bin_start_utc"].unique())
    positions = {}
    for index, when in enumerate(table["bin_start_utc"]):
        positions.setdefault(when, []).append(index)
    splitter = data_splitting.TimeSeriesSplit(n_splits=n_splits)
    for train_bins, test_bins in splitter.split(bins):
        yield (
            [i for b in train_bins for i in positions[bins[b]]],
            [i for b in test_bins for i in positions[bins[b]]],
        )
