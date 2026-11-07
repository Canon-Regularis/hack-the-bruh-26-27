"""What "correct" means for the speeds stage. Every row here is synthetic."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from matching.table import PingPair, match_pings
from speeds.dwell import distance_m, is_dwell
from speeds.edge_speeds import OUTPUT_COLUMNS, bin_start, build_traversals, edge_speed_table
from speeds.synthetic import STOP_EDGE, TRUE_SPEED_KMH, synthetic_bus

T0 = datetime(2026, 11, 7, 18, 0, 0, tzinfo=timezone.utc)
# One degree of latitude is about 111,195 m, so 0.001 degrees is about 111 m.
AWAY = (55.95, -3.19)
STOP = (55.96, -3.19)
NAMES = {"edge_a": "Street A", "edge_b": "Street B"}


def pair(start_s, end_s, edges, vehicle="bus_1", start=AWAY, end=None, source="synthetic"):
    return PingPair(
        vehicle_id=vehicle,
        start_time_utc=T0 + timedelta(seconds=start_s),
        end_time_utc=T0 + timedelta(seconds=end_s),
        start_position=start,
        end_position=end or (start[0] + 0.001, start[1]),
        edges=edges,
        source=source,
    )


def stationary_pair(start_s, end_s, edges, position):
    return pair(start_s, end_s, edges, start=position, end=position)


def speeds_by_edge(table):
    return dict(zip(table["edge_id"], table["mean_bus_observed_speed_kmh"]))


# --- Table format and match_pings on no input ---


def test_match_pings_is_empty_for_no_input():
    assert match_pings([]) == []


def test_ping_pair_rejects_times_without_a_timezone():
    with pytest.raises(ValueError):
        PingPair("b", datetime(2026, 11, 7, 18), T0, AWAY, AWAY, [("e", 1.0)], "synthetic")


def test_ping_pair_rejects_times_not_in_utc():
    bst = timezone(timedelta(hours=1))
    end = T0 + timedelta(seconds=30)
    with pytest.raises(ValueError):
        PingPair("b", T0.astimezone(bst), end, AWAY, AWAY, [("e", 1.0)], "synthetic")


def test_ping_pair_rejects_end_before_start():
    with pytest.raises(ValueError):
        pair(30, 30, [("edge_a", 10.0)])


def test_ping_pair_rejects_negative_metres():
    with pytest.raises(ValueError):
        pair(0, 30, [("edge_a", -1.0)])


def test_ping_pair_rejects_no_edges():
    with pytest.raises(ValueError):
        pair(0, 30, [])


def test_ping_pair_rejects_unknown_source_label():
    with pytest.raises(ValueError):
        pair(0, 30, [("edge_a", 10.0)], source="made_up")


# --- Dwell rule: within 20 m of a stop, and moved less than 10 m ---


def test_distance_known_answer():
    # 0.001 degrees of latitude on a 6,371 km sphere.
    assert distance_m((55.0, -3.0), (55.001, -3.0)) == pytest.approx(111.19, abs=0.01)


def test_stationary_at_a_stop_is_dwell():
    assert is_dwell(stationary_pair(0, 30, [("edge_a", 0.0)], STOP), [STOP])


def test_stationary_away_from_a_stop_is_not_dwell():
    assert not is_dwell(stationary_pair(0, 30, [("edge_a", 0.0)], AWAY), [STOP])


def test_moving_past_a_stop_is_not_dwell():
    assert not is_dwell(pair(0, 30, [("edge_a", 111.0)], start=STOP), [STOP])


def test_just_inside_and_outside_the_thresholds():
    metre = 1 / 111_195
    at_19m = (STOP[0] + 19 * metre, STOP[1])
    at_21m = (STOP[0] + 21 * metre, STOP[1])
    assert is_dwell(stationary_pair(0, 30, [("edge_a", 0.0)], at_19m), [STOP])
    assert not is_dwell(stationary_pair(0, 30, [("edge_a", 0.0)], at_21m), [STOP])
    moved_9m = (STOP[0] + 9 * metre, STOP[1])
    moved_11m = (STOP[0] + 11 * metre, STOP[1])
    assert is_dwell(pair(0, 30, [("edge_a", 9.0)], start=STOP, end=moved_9m), [STOP])
    assert not is_dwell(pair(0, 30, [("edge_a", 11.0)], start=STOP, end=moved_11m), [STOP])


# --- Bins ---


@pytest.mark.parametrize(
    "minute, second, expected_minute",
    [(0, 0, 0), (14, 59, 0), (15, 0, 15), (29, 59, 15), (44, 1, 30), (59, 59, 45)],
)
def test_bin_start_is_aligned_to_quarter_hours(minute, second, expected_minute):
    moment = T0.replace(minute=minute, second=second, microsecond=7)
    assert bin_start(moment) == T0.replace(minute=expected_minute)


def test_traversal_goes_in_the_bin_of_its_entry_time():
    pairs = [pair(14 * 60 + 50, 15 * 60 + 20, [("edge_a", 150.0)])]
    table = edge_speed_table(pairs, [], NAMES)
    assert table["bin_start_utc"].tolist() == [T0]


# --- Traversals ---


def test_time_is_split_across_edges_by_distance():
    traversals = build_traversals([pair(0, 30, [("edge_a", 60.0), ("edge_b", 90.0)])], [])
    assert [t.edge_id for t in traversals] == ["edge_a", "edge_b"]
    assert [t.seconds for t in traversals] == pytest.approx([12.0, 18.0])
    assert traversals[1].enter_time == T0 + timedelta(seconds=12)


def test_consecutive_pairs_on_one_edge_are_one_traversal():
    pairs = [pair(0, 30, [("edge_a", 150.0)]), pair(30, 60, [("edge_a", 150.0)])]
    (traversal,) = build_traversals(pairs, [])
    assert traversal.metres == 300.0
    assert traversal.seconds == 60.0


def metres_north_of(position, metres):
    return (position[0] + metres / 111_195, position[1])


def test_gps_drift_does_not_hide_a_bus_that_matching_says_stood_still():
    drifted = metres_north_of(STOP, 15)
    assert is_dwell(pair(0, 30, [("edge_a", 0.0)], start=STOP, end=drifted), [STOP])


def test_matched_movement_counts_even_if_the_gps_positions_barely_changed():
    assert not is_dwell(stationary_pair(0, 30, [("edge_a", 50.0)], STOP), [STOP])


def test_moved_distance_is_summed_across_edges():
    assert is_dwell(stationary_pair(0, 30, [("edge_a", 4.0), ("edge_b", 4.0)], STOP), [STOP])
    assert not is_dwell(stationary_pair(0, 30, [("edge_a", 6.0), ("edge_b", 6.0)], STOP), [STOP])


def test_moving_exactly_10_m_is_not_dwell():
    assert not is_dwell(stationary_pair(0, 30, [("edge_a", 10.0)], STOP), [STOP])


def test_the_stop_check_uses_the_gps_positions():
    near_stop = metres_north_of(STOP, 15)
    far_from_stop = metres_north_of(STOP, 25)
    assert is_dwell(stationary_pair(0, 30, [("edge_a", 0.0)], near_stop), [STOP])
    assert not is_dwell(stationary_pair(0, 30, [("edge_a", 0.0)], far_from_stop), [STOP])


def test_a_dwell_is_excluded_without_splitting_the_traversal():
    pairs = [
        pair(0, 30, [("edge_a", 150.0)]),
        stationary_pair(30, 60, [("edge_a", 0.0)], STOP),
        pair(60, 90, [("edge_a", 150.0)]),
    ]
    (traversal,) = build_traversals(pairs, [STOP])
    assert traversal.seconds == 60.0
    assert traversal.dwell_seconds == 30.0


def test_no_ping_pair_time_is_lost():
    pairs = [
        pair(0, 30, [("edge_a", 150.0)]),
        stationary_pair(30, 60, [("edge_a", 0.0)], STOP),
        pair(60, 90, [("edge_a", 150.0)]),
    ]
    row = edge_speed_table(pairs, [STOP], NAMES).iloc[0]
    assert row["total_seconds"] + row["dwell_seconds"] == 90.0
    assert row["mean_bus_observed_speed_kmh"] == pytest.approx(18.0)


def test_a_bus_seen_only_dwelling_keeps_a_row_with_no_speed():
    pairs = [stationary_pair(0, 30, [("edge_a", 0.0)], STOP)]
    row = edge_speed_table(pairs, [STOP], NAMES).iloc[0]
    assert row["dwell_seconds"] == 30.0
    assert math.isnan(row["mean_bus_observed_speed_kmh"])
    assert row["n_traversals"] == 0
    assert row["n_vehicles"] == 0


def test_a_long_wait_bins_the_movement_by_when_the_bus_moved():
    pairs = [
        stationary_pair(0, 40 * 60, [("edge_a", 0.0)], STOP),
        pair(40 * 60, 40 * 60 + 30, [("edge_a", 150.0)], start=STOP),
    ]
    row = edge_speed_table(pairs, [STOP], NAMES).iloc[0]
    assert row["bin_start_utc"] == T0.replace(minute=30)
    assert row["dwell_seconds"] == 40 * 60
    assert row["mean_bus_observed_speed_kmh"] == pytest.approx(18.0)


def test_a_gap_in_pings_starts_a_new_traversal():
    pairs = [pair(0, 30, [("edge_a", 150.0)]), pair(300, 330, [("edge_a", 150.0)])]
    assert len(build_traversals(pairs, [])) == 2


def test_input_order_does_not_matter():
    pairs = [pair(30, 60, [("edge_a", 150.0)]), pair(0, 30, [("edge_a", 150.0)])]
    assert len(build_traversals(pairs, [])) == 1


def test_queueing_away_from_a_stop_stays_in_the_speed():
    pairs = [
        pair(0, 30, [("edge_a", 150.0)]),
        stationary_pair(30, 60, [("edge_a", 0.0)], AWAY),
    ]
    table = edge_speed_table(pairs, [STOP], NAMES)
    assert table["total_seconds"].tolist() == [60.0]
    assert speeds_by_edge(table)["edge_a"] == pytest.approx(9.0)


# --- Output table ---


def test_speed_is_total_distance_over_total_time():
    # Two buses on a 300 m edge at 30 km/h (36 s) and 10 km/h (108 s).
    pairs = [
        pair(0, 36, [("edge_a", 300.0)], vehicle="bus_1"),
        pair(0, 108, [("edge_a", 300.0)], vehicle="bus_2"),
    ]
    row = edge_speed_table(pairs, [], NAMES).iloc[0]
    assert row["mean_bus_observed_speed_kmh"] == pytest.approx(15.0)
    assert row["total_metres"] == 600.0
    assert row["total_seconds"] == 144.0


def test_a_tiny_edge_time_does_not_wreck_the_speed():
    pairs = [
        pair(0, 1, [("edge_a", 20.0)], vehicle="bus_1"),
        pair(0, 36, [("edge_a", 100.0)], vehicle="bus_2"),
    ]
    table = edge_speed_table(pairs, [], NAMES)
    assert speeds_by_edge(table)["edge_a"] == pytest.approx(120 / 37 * 3.6)


def test_counts_traversals_and_distinct_vehicles():
    pairs = [
        pair(0, 30, [("edge_a", 150.0)], vehicle="bus_1"),
        pair(300, 330, [("edge_a", 150.0)], vehicle="bus_1"),
        pair(0, 30, [("edge_a", 150.0)], vehicle="bus_2"),
    ]
    row = edge_speed_table(pairs, [], NAMES).iloc[0]
    assert row["n_traversals"] == 3
    assert row["n_vehicles"] == 2


def test_output_has_the_agreed_columns_and_labels():
    table = edge_speed_table([pair(0, 30, [("edge_a", 150.0)])], [], NAMES)
    assert list(table.columns) == OUTPUT_COLUMNS
    row = table.iloc[0]
    assert row["street_name"] == "Street A"
    assert row["source"] == "synthetic"


def test_sources_are_never_mixed_in_one_row():
    pairs = [
        pair(0, 30, [("edge_a", 150.0)], vehicle="bus_1", source="synthetic"),
        pair(0, 30, [("edge_a", 150.0)], vehicle="bus_2", source="observed"),
    ]
    assert sorted(edge_speed_table(pairs, [], NAMES)["source"]) == ["observed", "synthetic"]


def test_sources_stay_apart_even_for_one_vehicle():
    pairs = [
        pair(0, 30, [("edge_a", 150.0)], source="synthetic"),
        pair(30, 60, [("edge_a", 150.0)], source="observed"),
    ]
    assert sorted(edge_speed_table(pairs, [], NAMES)["source"]) == ["observed", "synthetic"]


def test_an_edge_missing_from_the_street_lookup_fails_loudly():
    with pytest.raises(KeyError):
        edge_speed_table([pair(0, 30, [("edge_unknown", 150.0)])], [], NAMES)


def test_no_rows_gives_an_empty_table_with_the_right_columns():
    assert list(edge_speed_table([], [], NAMES).columns) == OUTPUT_COLUMNS


# --- Synthetic bus: known path, known speed, one dwell ---


def test_dwell_aligned_with_pings_recovers_the_exact_speed():
    pairs, stops, names = synthetic_bus(first_ping_s=0, noise_m=0.0)
    for speed in speeds_by_edge(edge_speed_table(pairs, stops, names)).values():
        assert speed == pytest.approx(TRUE_SPEED_KMH)


def test_dwell_between_pings_with_gps_noise_stays_within_tolerance():
    pairs, stops, names = synthetic_bus(first_ping_s=15, noise_m=2.0)
    speeds = speeds_by_edge(edge_speed_table(pairs, stops, names))
    stop_edge_speed = speeds.pop(STOP_EDGE)
    for speed in speeds.values():
        assert speed == pytest.approx(TRUE_SPEED_KMH, rel=0.01)
    # Up to one ping interval of dwell can stay inside partly moving pairs.
    assert 12.0 - 1e-9 <= stop_edge_speed < TRUE_SPEED_KMH
