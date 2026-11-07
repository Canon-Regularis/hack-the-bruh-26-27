"""What "correct" means for the feature table. Every row here is synthetic."""

import pandas as pd
import pytest

from engine.orchid import street_network as sn
from features.betweenness import read_betweenness, sampled_betweenness
from features.lag import LAG_BINS, neighbours
from features.street import ROAD_CLASS_RANK, read_aadf, read_flags, road_class_rank
from features.table import (
    FILL_VALUE,
    TARGET,
    build_feature_table,
    feature_columns,
    missing_inputs,
)
from speeds.edge_speeds import OUTPUT_COLUMNS

T0 = pd.Timestamp("2026-11-07 18:00", tz="UTC")


def street_graph():
    """A north-south road with two named streets and one unnamed side road.

    Street A runs 0-1-2, Street B runs 2-3-4, and an unnamed road leaves node 2
    for node 5. Every road is two-way.
    """
    graph = sn.RoadGraph()
    points = [(55.950, -3.19), (55.951, -3.19), (55.952, -3.19),
              (55.953, -3.19), (55.954, -3.19), (55.952, -3.188)]
    for node, (lat, lon) in enumerate(points):
        graph.add_node(node, lat=lat, lon=lon)
    ids = {}
    for name, u, v, street, klass in [
        ("a1", 0, 1, "Street A", "primary"),
        ("a2", 1, 2, "Street A", "primary"),
        ("b1", 2, 3, "Street B", "secondary"),
        ("b2", 3, 4, "Street B", "secondary"),
        ("side", 2, 5, "", "residential"),
    ]:
        forward, backward = graph.add_edge(u, v, oneway=False, street_name=street,
                                           highway_class=klass)
        ids[name], ids[name + "_back"] = str(forward), str(backward)
    return graph, ids


def speed_row(edge_id, street, minutes, kmh, n=1, source="synthetic"):
    """One edge-speed row, as edge_speed_table would write it."""
    seconds = 60.0
    metres = kmh / 3.6 * seconds if n else 0.0
    return {
        "edge_id": edge_id,
        "street_name": street,
        "bin_start_utc": T0 + pd.Timedelta(minutes=minutes),
        "n_traversals": n,
        "n_vehicles": n,
        "total_metres": metres,
        "total_seconds": seconds if n else 0.0,
        "dwell_seconds": 0.0 if n else 30.0,
        TARGET: kmh if n else float("nan"),
        "source": source,
    }


def speed_table(rows):
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def busy_speeds(ids):
    """Every edge observed in every bin from 18:00 to 19:30."""
    streets = {"a": "Street A", "b": "Street B", "s": ""}
    rows = []
    for minutes in range(0, 105, 15):
        for name, edge_id in ids.items():
            kmh = 10.0 + minutes / 15 + len(name)
            rows.append(speed_row(edge_id, streets[name[0]], minutes, kmh))
    return speed_table(rows)


def features_for(table, edge_id, when):
    match = table[(table["edge_id"] == edge_id) & (table["bin_start_utc"] == when)]
    assert len(match) == 1
    return match[feature_columns(table)].iloc[0]


def build(speeds, graph, betweenness=None):
    return build_feature_table(speeds, graph, betweenness or {})


# --- Leakage firewall ---


@pytest.mark.parametrize("target_name", ["a2", "b1", "side"])
def test_features_ignore_the_edge_its_street_and_the_present_and_future(target_name):
    graph, ids = street_graph()
    speeds = busy_speeds(ids)
    edge_id, when = ids[target_name], T0 + pd.Timedelta(minutes=60)
    before = features_for(build(speeds, graph), edge_id, when)

    street = speeds.loc[speeds["edge_id"] == edge_id, "street_name"].iloc[0]
    hidden = (
        (speeds["edge_id"] == edge_id)
        | ((speeds["street_name"] == street) & (street != ""))
        | (speeds["bin_start_utc"] >= when)
    )
    changed = speeds.copy()
    changed.loc[hidden, "total_seconds"] *= 3
    changed.loc[hidden, TARGET] /= 3
    after = features_for(build(changed, graph), edge_id, when)

    pd.testing.assert_series_equal(before, after)


def test_the_firewall_test_can_fail_when_a_visible_neighbour_changes():
    graph, ids = street_graph()
    speeds = busy_speeds(ids)
    edge_id, when = ids["a2"], T0 + pd.Timedelta(minutes=60)
    before = features_for(build(speeds, graph), edge_id, when)

    changed = speeds.copy()
    visible = (changed["edge_id"] == ids["b1"]) & (
        changed["bin_start_utc"] == when - pd.Timedelta(minutes=15)
    )
    changed.loc[visible, "total_seconds"] *= 3
    after = features_for(build(changed, graph), edge_id, when)

    assert after["lag_speed_kmh"] != before["lag_speed_kmh"]


# --- Spatial lag ---


def test_neighbours_leave_out_the_edge_its_reverse_and_its_street():
    graph, ids = street_graph()
    expected = {ids["b1"], ids["b1_back"], ids["side"], ids["side_back"]}
    assert set(neighbours(graph, ids["a2"])) == expected


def test_an_unnamed_edge_keeps_its_named_neighbours():
    graph, ids = street_graph()
    expected = {ids["a2"], ids["a2_back"], ids["b1"], ids["b1_back"]}
    assert set(neighbours(graph, ids["side"])) == expected


def lag_from(rows, graph, ids, minutes=60):
    speeds = speed_table([speed_row(ids["a2"], "Street A", minutes, 20.0), *rows])
    when = T0 + pd.Timedelta(minutes=minutes)
    return features_for(build(speeds, graph), ids["a2"], when)


def test_lag_looks_back_exactly_one_hour():
    graph, ids = street_graph()
    oldest_kept = 60 - 15 * LAG_BINS
    kept = lag_from([speed_row(ids["b1"], "Street B", oldest_kept, 30.0)], graph, ids)
    assert kept["lag_speed_kmh"] == pytest.approx(30.0)
    assert kept["lag_speed_kmh_missing"] == 0

    too_old = lag_from([speed_row(ids["b1"], "Street B", oldest_kept - 15, 30.0)], graph, ids)
    assert too_old["lag_speed_kmh_missing"] == 1


def test_lag_never_uses_the_rows_own_bin():
    graph, ids = street_graph()
    same_bin = lag_from([speed_row(ids["b1"], "Street B", 60, 30.0)], graph, ids)
    assert same_bin["lag_speed_kmh_missing"] == 1


def test_lag_is_total_metres_over_total_seconds():
    graph, ids = street_graph()
    rows = [speed_row(ids["b1"], "Street B", 45, 30.0), speed_row(ids["side"], "", 30, 10.0)]
    assert lag_from(rows, graph, ids)["lag_speed_kmh"] == pytest.approx(20.0)


def test_lag_ignores_dwell_only_rows_and_other_sources():
    graph, ids = street_graph()
    rows = [
        speed_row(ids["b1"], "Street B", 45, 30.0, n=0),
        speed_row(ids["side"], "", 45, 30.0, source="observed"),
    ]
    assert lag_from(rows, graph, ids)["lag_speed_kmh_missing"] == 1


def test_missing_lag_is_filled_with_a_constant_and_flagged():
    graph, ids = street_graph()
    row = lag_from([], graph, ids)
    assert row["lag_speed_kmh"] == FILL_VALUE
    assert row["lag_speed_kmh_missing"] == 1


# --- Targets and street features ---


def test_dwell_only_rows_are_never_targets():
    graph, ids = street_graph()
    speeds = speed_table([
        speed_row(ids["a2"], "Street A", 0, 20.0),
        speed_row(ids["b1"], "Street B", 0, 0.0, n=0),
    ])
    assert list(build(speeds, graph)["edge_id"]) == [ids["a2"]]


def test_a_row_carries_its_target_street_and_time():
    graph, ids = street_graph()
    speeds = speed_table([speed_row(ids["b1"], "Street B", 75, 25.0)])
    row = build(speeds, graph).iloc[0]
    assert row[TARGET] == 25.0
    assert row["street_name"] == "Street B"
    assert (row["hour"], row["weekday"]) == (19, 5)
    assert row["road_class_rank"] == ROAD_CLASS_RANK["secondary"]
    assert row["lanes"] == graph.edge(int(ids["b1"])).lanes
    assert row["speed_limit_kph"] == graph.edge(int(ids["b1"])).speed_limit_kph
    assert row["length_m"] == pytest.approx(graph.edge(int(ids["b1"])).length_m)


def test_the_target_is_not_a_feature():
    graph, ids = street_graph()
    table = build(busy_speeds(ids), graph)
    assert TARGET not in feature_columns(table)
    assert "street_name" not in feature_columns(table)


def test_road_class_rank_runs_from_motorway_down():
    assert road_class_rank("motorway") < road_class_rank("primary") < road_class_rank("residential")
    assert road_class_rank("residential") < road_class_rank("footway")


def test_road_class_rank_covers_every_class_the_graph_can_hold():
    assert set(ROAD_CLASS_RANK) == set(sn.HIGHWAY_DEFAULTS)


# --- Betweenness ---


def test_sampled_betweenness_repeats_with_the_same_seed():
    graph, _ = street_graph()
    first = sampled_betweenness(graph, n_sources=3, seed=7)
    assert first == sampled_betweenness(graph, n_sources=3, seed=7)
    assert all(isinstance(edge_id, str) for edge_id in first)


def test_sampling_every_node_gives_the_exact_answer():
    graph, _ = street_graph()
    exact = sn.edge_betweenness_centrality(graph, mode="car", weight="free_flow_time_s")
    sampled = sampled_betweenness(graph, n_sources=100)
    assert sampled == pytest.approx({str(e): value for e, value in exact.items()})


def test_betweenness_joins_on_edge_id_and_missing_is_flagged():
    graph, ids = street_graph()
    speeds = speed_table([speed_row(ids["a2"], "Street A", 0, 20.0),
                          speed_row(ids["b1"], "Street B", 0, 20.0)])
    table = build(speeds, graph, {ids["a2"]: 4.0}).set_index("edge_id")
    assert table.loc[ids["a2"], "betweenness"] == 4.0
    assert table.loc[ids["b1"], "betweenness"] == FILL_VALUE
    assert table.loc[ids["b1"], "betweenness_missing"] == 1


def test_read_betweenness_returns_empty_when_not_computed(tmp_path):
    assert read_betweenness(tmp_path / "none.json") == {}


# --- Optional Engine files ---


def test_missing_engine_files_are_reported_not_guessed(tmp_path):
    flags = read_flags(tmp_path / "restriction_flags.csv")
    aadf = read_aadf(tmp_path / "dft_aadf.csv")
    assert flags is None and aadf is None
    assert missing_inputs(flags, aadf) == ["concern_intensity", "dft_aadf", "restriction_flags"]


def test_present_engine_files_join_on_edge_id(tmp_path):
    graph, ids = street_graph()
    flags_path = tmp_path / "restriction_flags.csv"
    flags_path.write_text(
        "edge_id,no_motor_vehicles,one_way,pedestrian_only\n"
        f"{ids['a2']},1,0,0\n{ids['b1']},0,0,0\n"
    )
    aadf_path = tmp_path / "dft_aadf.csv"
    aadf_path.write_text(f"edge_id,count_point_id,aadf\n{ids['a2']},10,1000\n{ids['a2']},11,3000\n")
    speeds = speed_table([speed_row(ids["a2"], "Street A", 0, 20.0),
                          speed_row(ids["b1"], "Street B", 0, 20.0)])

    flags, aadf = read_flags(flags_path), read_aadf(aadf_path)
    table = build_feature_table(speeds, graph, {}, flags=flags, aadf=aadf).set_index("edge_id")

    assert table.loc[ids["a2"], "no_motor_vehicles"] == 1
    assert table.loc[ids["a2"], "aadf"] == 2000
    assert table.loc[ids["b1"], "aadf_missing"] == 1
    assert missing_inputs(flags, aadf) == ["concern_intensity"]
