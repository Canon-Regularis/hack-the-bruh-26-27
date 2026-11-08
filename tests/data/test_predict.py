"""What "correct" means for the prediction table. Every row here is synthetic."""

import math

import pandas as pd
import pytest

from engine.orchid import street_network as sn
from features.table import (
    TARGET,
    build_feature_table,
    build_prediction_features,
    feature_columns,
)
from model.predict import (
    PREDICTION_COLUMNS,
    night_bins,
    prediction_table,
    study_edges,
)
from speeds.edge_speeds import OUTPUT_COLUMNS

T0 = pd.Timestamp("2026-11-07 22:00", tz="UTC")
BBOX = {"xmin": -3.20, "ymin": 55.949, "xmax": -3.18, "ymax": 55.956}


def study_graph():
    """Street A runs 0-1-2 and Street B 2-3. A footway leaves node 2 for node 4.

    An unnamed road runs from node 3 out of the box to node 5, so its midpoint
    is outside. Every road is two-way.
    """
    graph = sn.RoadGraph()
    points = [(55.950, -3.19), (55.951, -3.19), (55.952, -3.19),
              (55.953, -3.19), (55.952, -3.188), (55.970, -3.19)]
    for node, (lat, lon) in enumerate(points):
        graph.add_node(node, lat=lat, lon=lon)
    ids = {}
    for name, u, v, street, klass, modes in [
        ("a1", 0, 1, "Street A", "primary", None),
        ("a2", 1, 2, "Street A", "primary", None),
        ("b1", 2, 3, "Street B", "secondary", None),
        ("foot", 2, 4, "Path", "footway", ("foot",)),
        ("out", 3, 5, "", "residential", None),
    ]:
        forward, backward = graph.add_edge(u, v, oneway=False, street_name=street,
                                           highway_class=klass, modes_permitted=modes)
        ids[name], ids[name + "_back"] = str(forward), str(backward)
    return graph, ids


def speed_row(edge_id, street, minutes, kmh, n=1, source="observed"):
    """One edge-speed row, as edge_speed_table would write it."""
    seconds = 60.0
    return {
        "edge_id": edge_id,
        "street_name": street,
        "bin_start_utc": T0 + pd.Timedelta(minutes=minutes),
        "n_traversals": n,
        "n_vehicles": n,
        "total_metres": kmh / 3.6 * seconds if n else 0.0,
        "total_seconds": seconds if n else 0.0,
        "dwell_seconds": 0.0 if n else 30.0,
        TARGET: kmh if n else float("nan"),
        "source": source,
    }


def speeds_for(ids):
    """Street A and B observed every bin from 22:00 to 23:00, except b1 at 22:30.

    The b1_back edge has only a dwell row at 22:45. The footway is never observed.
    """
    rows = []
    for minutes in range(0, 75, 15):
        for name in ("a1", "a1_back", "a2", "a2_back", "b1", "b1_back"):
            if name == "b1" and minutes == 30:
                continue
            if name == "b1_back" and minutes == 45:
                rows.append(speed_row(ids[name], "Street B", minutes, 0.0, n=0))
                continue
            street = "Street A" if name.startswith("a") else "Street B"
            rows.append(speed_row(ids[name], street, minutes, 12.0 + minutes / 15 + len(name),
                                  n=2))
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def predict(speeds, graph):
    return prediction_table(speeds, graph, betweenness={}, bbox=BBOX)


def row_at(table, edge_id, minutes):
    when = T0 + pd.Timedelta(minutes=minutes)
    match = table[(table["edge_id"] == edge_id) & (table["bin_start_utc"] == when)]
    assert len(match) == 1
    return match.iloc[0]


# --- Which edges and bins ---


def test_study_edges_keep_bus_roads_inside_the_box():
    graph, ids = study_graph()
    kept = set(study_edges(graph, BBOX))
    expected = {ids[n] for n in ("a1", "a1_back", "a2", "a2_back", "b1", "b1_back")}
    assert kept == expected


def test_night_bins_run_every_15_minutes_from_first_to_last():
    graph, ids = study_graph()
    bins = night_bins(speeds_for(ids))
    assert bins == [T0 + pd.Timedelta(minutes=m) for m in range(0, 75, 15)]


def test_every_study_edge_and_bin_appears_once():
    graph, ids = study_graph()
    speeds = speeds_for(ids)
    table = predict(speeds, graph)
    assert len(table) == len(study_edges(graph, BBOX)) * len(night_bins(speeds))
    assert not table.duplicated(["edge_id", "bin_start_utc"]).any()
    assert ids["foot"] not in set(table["edge_id"])
    assert ids["out"] not in set(table["edge_id"])


def test_an_observed_edge_outside_the_bus_network_is_kept_as_observed():
    graph, ids = study_graph()
    speeds = pd.concat([speeds_for(ids), pd.DataFrame(
        [speed_row(ids["foot"], "Path", 15, 9.0)], columns=OUTPUT_COLUMNS)])
    table = predict(speeds, graph)
    row = row_at(table, ids["foot"], 15)
    assert row["kind"] == "observed"
    assert row["speed_kmh"] == pytest.approx(9.0)


# --- Observed and predicted rows ---


def test_observed_rows_keep_the_observed_speed_and_count():
    graph, ids = study_graph()
    table = predict(speeds_for(ids), graph)
    row = row_at(table, ids["a2"], 30)
    assert row["kind"] == "observed"
    assert row["speed_kmh"] == pytest.approx(12.0 + 2 + len("a2"))
    assert row["n_traversals"] == 2
    assert row["street_name"] == "Street A"


def test_unobserved_rows_are_predicted_with_zero_traversals():
    graph, ids = study_graph()
    table = predict(speeds_for(ids), graph)
    row = row_at(table, ids["b1"], 30)
    assert row["kind"] == "predicted"
    assert row["n_traversals"] == 0
    assert math.isfinite(row["speed_kmh"]) and row["speed_kmh"] >= 0


def test_a_dwell_only_row_is_predicted():
    graph, ids = study_graph()
    table = predict(speeds_for(ids), graph)
    row = row_at(table, ids["b1_back"], 45)
    assert row["kind"] == "predicted"
    assert row["n_traversals"] == 0


def test_no_speed_is_missing_or_negative():
    graph, ids = study_graph()
    table = predict(speeds_for(ids), graph)
    assert table["speed_kmh"].notna().all()
    assert (table["speed_kmh"] >= 0).all()


def test_columns_are_the_agreed_list_in_order():
    graph, ids = study_graph()
    table = predict(speeds_for(ids), graph)
    assert list(table.columns) == PREDICTION_COLUMNS == [
        "edge_id", "street_name", "bin_start_utc", "speed_kmh", "kind", "n_traversals"]


def test_mixed_sources_are_refused():
    graph, ids = study_graph()
    speeds = speeds_for(ids)
    speeds.loc[speeds.index[0], "source"] = "synthetic"
    with pytest.raises(ValueError, match="source"):
        predict(speeds, graph)


# --- Training and prediction share one feature builder ---


def test_prediction_features_match_training_features_for_an_observed_key():
    graph, ids = study_graph()
    speeds = speeds_for(ids)
    train = build_feature_table(speeds, graph, {})
    when = T0 + pd.Timedelta(minutes=45)
    built = build_prediction_features([(ids["a2"], when)], speeds, graph, {}, "observed")
    expected = train[(train["edge_id"] == ids["a2"]) & (train["bin_start_utc"] == when)]
    columns = feature_columns(train)
    pd.testing.assert_series_equal(
        built[columns].iloc[0], expected[columns].iloc[0], check_names=False,
        check_dtype=False)
