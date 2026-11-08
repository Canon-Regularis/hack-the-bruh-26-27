"""Concern intensity from Engine's concern field, as a feature. Every row is synthetic."""

import pandas as pd
import pytest

from features.street import read_concerns
from features.table import FILL_VALUE, build_feature_table, missing_inputs
from tests.data.test_features import busy_speeds, street_graph

THEMES = ["crossing_delay", "road_safety", "rat_running"]


def write_field(path, rows):
    pd.DataFrame(rows, columns=["edge_id", *THEMES, "basis"]).to_csv(path, index=False)


def test_no_field_file_means_no_concerns(tmp_path):
    assert read_concerns(tmp_path / "concern_field.csv") is None


def test_intensity_is_the_sum_of_the_theme_columns(tmp_path):
    path = tmp_path / "concern_field.csv"
    write_field(path, [[7, 0.5, 1.0, 0.0, "located comments"],
                       [8, 0.0, 0.0, 0.0, "per-street counts"]])
    concerns = read_concerns(path)
    assert list(concerns.columns) == ["edge_id", "concern_intensity"]
    assert list(concerns["edge_id"]) == ["7", "8"]
    assert concerns["concern_intensity"].tolist() == pytest.approx([1.5, 0.0])


def test_feature_table_carries_concern_intensity_and_flags_gaps():
    graph, ids = street_graph()
    concerns = pd.DataFrame({"edge_id": [ids["a1"]], "concern_intensity": [2.0]})
    table = build_feature_table(busy_speeds(ids), graph, {}, concerns=concerns)
    a1 = table[table["edge_id"] == ids["a1"]].iloc[0]
    b1 = table[table["edge_id"] == ids["b1"]].iloc[0]
    assert a1["concern_intensity"] == 2.0 and a1["concern_intensity_missing"] == 0
    assert b1["concern_intensity"] == FILL_VALUE and b1["concern_intensity_missing"] == 1


def test_concern_intensity_is_only_missing_without_the_file():
    concerns = pd.DataFrame({"edge_id": ["1"], "concern_intensity": [1.0]})
    assert "concern_intensity" in missing_inputs(None, None)
    assert "concern_intensity" not in missing_inputs(None, None, concerns)
