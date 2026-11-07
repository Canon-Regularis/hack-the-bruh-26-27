"""The feature table: one row per edge and 15-minute bin where a bus moved.

The target is the bus-observed speed. Missing values are filled with a constant
and flagged in an indicator column, because the ORCHID GBR needs finite values.
A constant, unlike a mean, leaks nothing between validation folds.
"""

from functools import lru_cache

import pandas as pd

from features.lag import lag_speed_kmh, neighbours, observed_totals
from features.street import edge_features

TARGET = "mean_bus_observed_speed_kmh"
FILL_VALUE = -1.0
ROW_KEYS = ["edge_id", "bin_start_utc", "street_name", "source"]
NOT_FEATURES = {*ROW_KEYS, TARGET}
ALWAYS_MISSING = ["concern_intensity"]  # no source exists yet


def targets(speeds):
    """Return rows with a speed. Dwell-only rows have no moving time, so they are left out."""
    return speeds[(speeds["n_traversals"] > 0) & speeds[TARGET].notna()]


def build_feature_table(speeds, graph, betweenness, flags=None, aadf=None):
    """Return the feature table for every target row in the edge-speed table."""
    totals = observed_totals(speeds)
    edge_neighbours = lru_cache(maxsize=None)(lambda edge_id: neighbours(graph, edge_id))
    rows = []
    for row in targets(speeds).itertuples(index=False):
        when = pd.Timestamp(row.bin_start_utc)
        rows.append({
            "edge_id": row.edge_id,
            "bin_start_utc": when,
            "street_name": row.street_name,
            "source": row.source,
            **edge_features(graph, row.edge_id),
            "betweenness": betweenness.get(row.edge_id),
            "hour": when.hour,
            "weekday": when.weekday(),
            "lag_speed_kmh": lag_speed_kmh(
                totals, edge_neighbours(row.edge_id), row.source, when
            ),
            TARGET: getattr(row, TARGET),
        })
    table = pd.DataFrame(rows)
    optional = ["betweenness", "lag_speed_kmh"]
    for extra in (flags, aadf):
        if extra is not None and not table.empty:
            table = table.merge(extra, on="edge_id", how="left")
            optional += [c for c in extra.columns if c != "edge_id"]
    return fill_missing(table, optional)


def fill_missing(table, columns):
    """Add a 0/1 <column>_missing indicator, then fill gaps with FILL_VALUE."""
    for column in columns:
        if column in table:
            table[f"{column}_missing"] = table[column].isna().astype(int)
            table[column] = table[column].astype(float).fillna(FILL_VALUE)
    return table


def feature_columns(table):
    """Return the model's input columns: everything except row keys and the target."""
    return [column for column in table.columns if column not in NOT_FEATURES]


def missing_inputs(flags, aadf):
    """Return the names of feature inputs that do not exist yet, sorted."""
    missing = list(ALWAYS_MISSING)
    if flags is None:
        missing.append("restriction_flags")
    if aadf is None:
        missing.append("dft_aadf")
    return sorted(missing)
