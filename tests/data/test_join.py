"""Joining edge-speed windows. Every row here is synthetic."""

import pandas as pd

from speeds.edge_speeds import OUTPUT_COLUMNS
from speeds.join import join_windows
from tests.data.test_predict import speed_row


def table(*rows):
    return pd.DataFrame(list(rows), columns=OUTPUT_COLUMNS)


def test_the_later_window_wins_on_an_overlap():
    old = table(speed_row("1", "A", 0, 50.0), speed_row("1", "A", 15, 10.0))
    rerun = table(speed_row("1", "A", 15, 12.0))
    joined = join_windows([old, rerun])
    assert len(joined) == 2
    assert joined.loc[joined["bin_start_utc"] == joined["bin_start_utc"].max(),
                      "mean_bus_observed_speed_kmh"].item() == 12.0


def test_rows_that_do_not_overlap_are_all_kept():
    first = table(speed_row("1", "A", 0, 10.0), speed_row("2", "B", 0, 20.0))
    second = table(speed_row("1", "A", 15, 11.0), speed_row("3", "C", 0, 30.0))
    assert len(join_windows([first, second])) == 4


def test_different_sources_never_replace_each_other():
    observed = table(speed_row("1", "A", 0, 10.0))
    synthetic = table(speed_row("1", "A", 0, 99.0, source="synthetic"))
    assert len(join_windows([observed, synthetic])) == 2


def test_the_result_has_the_edge_speed_columns_and_is_sorted():
    joined = join_windows([table(speed_row("2", "B", 15, 10.0)),
                           table(speed_row("1", "A", 0, 10.0))])
    assert list(joined.columns) == OUTPUT_COLUMNS
    assert list(joined["edge_id"]) == ["1", "2"]
    assert list(joined.index) == [0, 1]
