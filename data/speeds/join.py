"""Join edge-speed windows into one table for the night.

Matching runs in time windows, and a window can be matched again after a fix.
Tables are joined in the order given, and on a repeated (source, edge, bin) the
later table wins, so a rerun window replaces the old one.
"""

import pandas as pd

from speeds.edge_speeds import OUTPUT_COLUMNS

KEY = ["source", "edge_id", "bin_start_utc"]


def join_windows(tables):
    """Return one edge-speed table; later tables win on a repeated key."""
    joined = pd.concat([table[OUTPUT_COLUMNS] for table in tables], ignore_index=True)
    joined["edge_id"] = joined["edge_id"].astype(str)
    joined["bin_start_utc"] = pd.to_datetime(joined["bin_start_utc"], utc=True)
    joined = joined.drop_duplicates(KEY, keep="last")
    return joined.sort_values(["edge_id", "bin_start_utc", "source"], ignore_index=True)
