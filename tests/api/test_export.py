import csv
import json

import pytest

from engine.orchid import street_network as sn
from src.api import export
from src.concerns.concern_field import THEMES
from tests.rules.rules_graph import build_graph as rules_graph

UE = {"0": {"flow": 100.0, "vc": 0.2, "t": 10.0, "t0": 8.0}, "1": {"flow": 50.0, "vc": 0.5, "t": 12.0, "t0": 8.0}}


@pytest.fixture
def graph():
    """High Street (two-way, 0 -> 1) and Cockburn Street (one-way, 1 -> 2) inside the study area,
    an unnamed lane (2 -> 3) inside it, and Far Road (3 -> 4) leaving it."""
    g = sn.RoadGraph()
    for nid, (lat, lon) in enumerate([(55.9490, -3.1950), (55.9495, -3.1900), (55.9500, -3.1880),
                                      (55.9505, -3.1870), (55.9505, -3.2500)]):
        g.add_node(nid, lat=lat, lon=lon)
    g.add_edge(0, 1, oneway=False, street_name="High Street", highway_class="tertiary", osm={"way_id": 1})  # 0, 1
    g.add_edge(1, 2, street_name="Cockburn Street", highway_class="residential", osm={"way_id": 2})       # 2
    g.add_edge(2, 3, oneway=False, highway_class="service", osm={"way_id": 3})                             # 3, 4
    g.add_edge(3, 4, oneway=False, street_name="Far Road", highway_class="primary", osm={"way_id": 4})     # 5, 6
    return g


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _props(features):
    return {f["properties"]["name"]: f["properties"] for f in features}


@pytest.mark.parametrize("lon, lat, inside", [
    (-3.19, 55.95, True), (-3.215, 55.942, True), (-3.155, 55.958, True), (-3.216, 55.95, False), (-3.19, 55.959, False),
])
def test_in_bbox_includes_the_edges(lon, lat, inside):
    assert export.in_bbox(lon, lat) is inside


def test_read_ue_is_empty_until_the_equilibrium_has_run(tmp_path):
    assert export.read_ue(tmp_path / "missing.csv") == {}


def test_network_keeps_segments_inside_the_study_area_only(graph):
    names = [f["properties"]["name"] for f in export.network_features(graph, {})]
    assert sorted(names) == ["", "Cockburn Street", "High Street"]


def test_network_joins_both_directions_of_the_equilibrium(graph):
    props = _props(export.network_features(graph, UE))
    assert props["High Street"] == {
        "id": "0", "rev": "1", "name": "High Street", "cls": "tertiary", "car": True, "oneway": False,
        "len": props["High Street"]["len"], "lanes": props["High Street"]["lanes"], "limit": props["High Street"]["limit"],
        "f_flow": 100.0, "f_vc": 0.2, "f_t": 10.0, "f_t0": 8.0, "r_flow": 50.0, "r_vc": 0.5, "r_t": 12.0, "r_t0": 8.0,
        "flow": 150.0, "vc": 0.5,
    }
    assert props["High Street"]["len"] == pytest.approx(graph.edge(0).length_m, abs=0.1)


def test_network_leaves_flow_off_segments_the_equilibrium_did_not_cover(graph):
    cockburn = _props(export.network_features(graph, UE))["Cockburn Street"]
    assert cockburn["rev"] is None and cockburn["oneway"] is True
    assert not {"flow", "vc", "f_flow", "r_flow"} & cockburn.keys()


def test_network_coordinates_are_lon_lat(graph):
    high = next(f for f in export.network_features(graph, {}) if f["properties"]["name"] == "High Street")
    (lon0, lat0), (lon1, lat1) = high["geometry"]["coordinates"]
    assert (lon0, lat0) == pytest.approx((-3.1950, 55.9490), abs=1e-6)
    assert (lon1, lat1) == pytest.approx((-3.1900, 55.9495), abs=1e-6)


def test_streets_skips_unnamed_segments_and_bounds_each_name(graph):
    streets = export.streets(export.network_features(graph, {}))
    assert list(streets) == ["Cockburn Street", "High Street"]
    assert streets["High Street"]["segments"] == 1
    assert streets["High Street"]["bbox"] == pytest.approx([-3.1950, 55.9490, -3.1900, 55.9495], abs=1e-6)


def test_crossings_inside_the_study_area_only(graph, tmp_path):
    path = _write_csv(tmp_path / "crossings.csv", [
        {"node": "1", "edge_ids": "0;2", "control": "unsignalised", "width_m": "6.5",
         "vehicle_flow_veh_h": "150.0", "delay_s": "12.34", "los": "C"},
        {"node": "4", "edge_ids": "5", "control": "signalised", "width_m": "3.25",
         "vehicle_flow_veh_h": "", "delay_s": "40.0", "los": "E"},
    ])
    [crossing] = export.crossing_features(graph, path)
    assert crossing["properties"] == {"node": "1", "edges": ["0", "2"], "control": "unsignalised", "width": 6.5,
                                      "delay": 12.3, "los": "C", "flow": 150.0}
    assert crossing["geometry"]["coordinates"] == pytest.approx([-3.1900, 55.9495])


def test_crossings_are_empty_until_the_delay_has_run(graph, tmp_path):
    assert export.crossing_features(graph, tmp_path / "missing.csv") == []


DFT_ROW = {"count_point_id": "", "year": "", "road_name": "A1", "latitude": "55.95", "longitude": "-3.19",
           "all_motor_vehicles": "", "estimation_method": ""}


@pytest.fixture
def dft(tmp_path):
    raw = _write_csv(tmp_path / "dft_aadf.csv", [
        {**DFT_ROW, "count_point_id": "1", "year": "2024", "all_motor_vehicles": "900", "estimation_method": "Counted"},
        {**DFT_ROW, "count_point_id": "1", "year": "2025", "all_motor_vehicles": "1000", "estimation_method": "Estimated"},
        {**DFT_ROW, "count_point_id": "2", "year": "2025", "all_motor_vehicles": "2000", "estimation_method": "Counted"},
        {**DFT_ROW, "count_point_id": "3", "year": "2025", "all_motor_vehicles": "3000", "estimation_method": "Counted"},
        {**DFT_ROW, "count_point_id": "4", "year": "2025", "all_motor_vehicles": "4000", "estimation_method": "Counted",
         "longitude": "-3.30"},
    ])
    review = _write_csv(tmp_path / "dft_review.csv", [
        {"count_point_id": "1", "edge_id": "0", "status": "matched"},
        {"count_point_id": "2", "edge_id": "2", "status": "matched"},
        {"count_point_id": "3", "edge_id": "", "status": "no car road within 25 m"},
        {"count_point_id": "4", "edge_id": "5", "status": "matched"},
    ])
    return raw, review


def test_counts_are_matched_points_in_the_study_area_with_their_latest_year(dft):
    props = {f["properties"]["id"]: f["properties"] for f in export.count_features(*dft)}
    assert props == {
        "1": {"id": "1", "road": "A1", "aadf": 1000, "year": 2025, "counted": False, "edge": "0"},
        "2": {"id": "2", "road": "A1", "aadf": 2000, "year": 2025, "counted": True, "edge": "2"},
    }


def test_counts_are_empty_without_the_dft_files(tmp_path):
    assert export.count_features(tmp_path / "dft.csv", tmp_path / "review.csv") == []


def clause(cid, street, restriction, **overrides):
    return {"clause_id": cid, "document_id": "order", "page": 8, "quote": "QUOTE", "street": street,
            "restriction": restriction, "times": "at any time", "exemptions": ["cycles"], **overrides}


def _clause_files(folder, clauses):
    sources = folder / "sources"
    sources.mkdir(exist_ok=True)
    (sources / "order.txt").write_text("The order: QUOTE.", encoding="utf-8")
    path = folder / "clauses.json"
    path.write_text(json.dumps(clauses), encoding="utf-8")
    return path, sources


def test_rules_layer_reports_status_and_where_osm_disagrees(tmp_path):
    g = rules_graph()
    rows, edge_rule, held = export.rules_layer(g, *_clause_files(tmp_path, [
        clause("lawn", "Lawnmarket", "no_motor_vehicles"),
        clause("castle", "castlehill", "no_motor_vehicles"),
        clause("advert", "Lawnmarket", "no_motor_vehicles", exemptions="unspecified"),
    ]))
    by_id = {r["id"]: r for r in rows}
    assert {k: by_id["lawn"][k] for k in ("streets", "status", "edges", "osm_wrong_edges", "osm_says")} == {
        "streets": ["Lawnmarket"], "status": "applied", "edges": 4, "osm_wrong_edges": 0, "osm_says": []}
    assert {k: by_id["castle"][k] for k in ("streets", "status", "edges", "osm_wrong_edges", "osm_says")} == {
        "streets": ["Castlehill"], "status": "applied", "edges": 2, "osm_wrong_edges": 2,
        "osm_says": [f"highway={g.edge(4).highway_class}"]}
    assert {k: by_id["advert"][k] for k in ("status", "reason", "edges")} == {
        "status": "abstained", "reason": "exemptions are not listed", "edges": 0}
    assert by_id["lawn"]["quote"] == "QUOTE" and by_id["lawn"]["page"] == 8
    lawnmarket_m = round(g.edge(0).length_m + g.edge(2).length_m)
    assert (by_id["lawn"]["length_m"], by_id["lawn"]["osm_wrong_m"]) == (lawnmarket_m, 0)
    assert by_id["castle"]["length_m"] == by_id["castle"]["osm_wrong_m"] == round(g.edge(4).length_m)
    assert edge_rule == {**{e: ("no_motor_vehicles", True) for e in "0123"},
                         "4": ("no_motor_vehicles", False), "5": ("no_motor_vehicles", False)}
    assert held == {"0", "1", "2", "3"}


def test_rules_layer_holds_streets_named_only_by_abstained_clauses_and_keeps_their_dates(tmp_path):
    rows, edge_rule, held = export.rules_layer(rules_graph(), *_clause_files(tmp_path, [
        clause("bus", "Bank Street", "no_buses", in_force="until 29 Nov 2026 (temporary)"),
        clause("bow", "Upper Bow", "one_way_part_street"),
    ]))
    assert edge_rule == {}
    assert held == {"6", "7", "8"}
    assert [(r["status"], r["in_force"]) for r in rows] == [("abstained", "until 29 Nov 2026 (temporary)"),
                                                            ("abstained", None)]


def test_rules_layer_shows_the_osm_tags_it_disagrees_with(tmp_path):
    g = rules_graph()
    [row], _, _ = export.rules_layer(g, *_clause_files(tmp_path, [clause("bank", "Bank Street", "no_motor_vehicles")]))
    assert row["osm_says"] == [f"highway={g.edge(6).highway_class}, access=no, motor_vehicle=yes"]


def test_rules_layer_is_none_without_clauses(tmp_path):
    assert export.rules_layer(rules_graph(), tmp_path / "clauses.json", tmp_path) is None


def test_add_rules_marks_a_segment_bound_in_either_direction():
    features = [{"properties": {"id": "0", "rev": "1"}}, {"properties": {"id": "8", "rev": None}},
                {"properties": {"id": "4", "rev": "5"}}]
    export.add_rules(features, {"1": ("one_way", False), "4": ("no_motor_vehicles", True),
                                "5": ("no_motor_vehicles", False)})
    assert [f["properties"] for f in features] == [
        {"id": "0", "rev": "1", "rule": "one_way", "osm_ok": False},
        {"id": "8", "rev": None},
        {"id": "4", "rev": "5", "rule": "no_motor_vehicles", "osm_ok": False},
    ]


def test_add_rules_holds_a_segment_only_where_no_applied_clause_binds_it():
    features = [{"properties": {"id": "0", "rev": "1"}}, {"properties": {"id": "8", "rev": None}},
                {"properties": {"id": "9", "rev": None}}]
    export.add_rules(features, {"0": ("no_motor_vehicles", True)}, held={"1", "8"})
    assert [f["properties"] for f in features] == [
        {"id": "0", "rev": "1", "rule": "no_motor_vehicles", "osm_ok": True},
        {"id": "8", "rev": None, "held": True},
        {"id": "9", "rev": None},
    ]


def _concern_files(folder):
    zero = dict.fromkeys(THEMES, "0")
    field = _write_csv(folder / "concern_field.csv", [
        {"edge_id": "0", **zero, "rat_running": "1", "basis": "per-street counts"},
        {"edge_id": "1", **zero, "rat_running": "2", "noise": "0.5", "basis": "per-street counts"},
        {"edge_id": "4", **zero, "basis": "per-street counts"},
    ])
    records = _write_csv(folder / "concerns.csv", [
        {"theme": "rat_running", "lat": "", "lon": "", "street": "lawnmarket", "count": "1", "severity": "",
         "source": "Report p.3"},
        {"theme": "noise", "lat": "", "lon": "", "street": "Nowhere Street", "count": "2", "severity": "",
         "source": "x"},
    ])
    return field, records


def test_concern_layer_keeps_non_zero_themes_and_each_streets_records(tmp_path):
    field, basis, records = export.concern_layer(rules_graph(), *_concern_files(tmp_path))
    assert field == {"0": {"rat_running": 1.0}, "1": {"rat_running": 2.0, "noise": 0.5}}
    assert basis == "per-street counts"
    assert records == {"Lawnmarket": [{"theme": "rat_running", "count": 1.0, "source": "Report p.3"}]}


def test_concern_layer_is_none_until_the_field_is_built(tmp_path):
    assert export.concern_layer(rules_graph(), tmp_path / "field.csv", tmp_path / "concerns.csv") is None


def test_add_concerns_uses_the_busier_direction():
    features = [{"properties": {"id": "0", "rev": "1"}}, {"properties": {"id": "4", "rev": None}}]
    export.add_concerns(features, {"0": {"rat_running": 1.0}, "1": {"rat_running": 2.0, "noise": 0.5}})
    assert features[0]["properties"] == {"id": "0", "rev": "1", "concern": 2.5, "concern_theme": "rat_running"}
    assert features[1]["properties"] == {"id": "4", "rev": None}


SPEED_FIELDS = ("edge_id", "street_name", "bin_start_utc", "speed_kmh", "kind", "n_traversals")


def _speed_rows(*rows):
    return [dict(zip(SPEED_FIELDS, row)) for row in rows]


@pytest.fixture
def predictions(tmp_path):
    folder = tmp_path / "predictions"
    folder.mkdir()
    _write_csv(folder / "speed_predictions_20261108T010000Z.csv",
               _speed_rows(("9", "Old Road", "2026-11-07 21:00:00+00:00", "1", "observed", "1")))
    _write_csv(folder / "speed_predictions_20261108T030000Z.csv", _speed_rows(
        ("12", "High Street", "2026-11-07 21:15:00+00:00", "8.04", "predicted", "0"),
        ("12", "High Street", "2026-11-07 21:00:00+00:00", "12.36", "observed", "2"),
        ("7", "Cowgate", "2026-11-07 21:00:00+00:00", "20", "predicted", "0"),
    ))
    return folder


def test_speeds_layer_reads_the_newest_table_into_bins(predictions):
    table, meta = export.speeds_layer(predictions)
    assert table == {"bins": ["2026-11-07T21:00:00+00:00", "2026-11-07T21:15:00+00:00"],
                     "edges": {"12": {"s": [12.4, 8.0], "o": "10"}, "7": {"s": [20.0, None], "o": "00"}}}
    assert meta == {"share_predicted": 0.667, "window": ["2026-11-07T21:00:00+00:00", "2026-11-07T21:15:00+00:00"],
                    "edges": 2, "file": "speed_predictions_20261108T030000Z.csv"}


SUMMARY = {
    "label": "bus-observed speed", "predicted_share": 0.79,
    "error_unseen_streets_kmh": 6.7, "baselines_unseen_streets_kmh": {"baseline_1": 7.1, "baseline_2": 7.24},
    "error_later_bins_kmh": 5.84, "baselines_later_bins_kmh": {"baseline_1": 6.9, "baseline_2": 8.6},
    "interval": {"target_coverage": 0.9, "measured_coverage_unseen_streets": 0.8974, "half_width_kmh": 14.244},
}


def test_speeds_layer_adds_the_errors_and_interval_from_the_newest_summary(predictions):
    (predictions / "summary_20261108T010000Z.json").write_text(json.dumps({"error_unseen_streets_kmh": 99}))
    (predictions / "summary_20261108T030000Z.json").write_text(json.dumps(SUMMARY), encoding="utf-8")
    _, meta = export.speeds_layer(predictions)
    assert {k: v for k, v in meta.items() if k not in ("share_predicted", "window", "edges", "file")} == {
        "mae_kmh": 6.7, "baseline_mae_kmh": 7.1, "skill": round(1 - 6.7 / 7.1, 3),
        "time_mae_kmh": 5.84, "time_baseline_mae_kmh": 6.9, "time_skill": round(1 - 5.84 / 6.9, 3),
        "interval_kmh": 14.2, "interval_target": 0.9, "interval_coverage": 0.897}


def test_summary_meta_leaves_out_what_the_summary_lacks():
    assert export.summary_meta({"error_unseen_streets_kmh": 6.7}) == {}
    assert export.summary_meta({"error_later_bins_kmh": 5.0, "baselines_later_bins_kmh": {"b": 10.0}}) == {
        "time_mae_kmh": 5.0, "time_baseline_mae_kmh": 10.0, "time_skill": 0.5}


def test_speeds_layer_is_none_before_signal_writes_a_table(tmp_path):
    assert export.speeds_layer(tmp_path / "predictions") is None


@pytest.mark.parametrize("rows, message", [
    (_speed_rows(("1", "A", "2026-11-07 21:00:00+00:00", "5", "guessed", "0")), "unknown kinds"),
    ([{"edge_id": "1", "speed_kmh": "5"}], "missing columns"),
])
def test_speeds_layer_rejects_a_malformed_table(tmp_path, rows, message):
    _write_csv(tmp_path / "speed_predictions_1.csv", rows)
    with pytest.raises(ValueError, match=message):
        export.speeds_layer(tmp_path)


NIGHT_FIELDS = ("edge_id", "street_name", "bin_start_utc", "n_traversals", "n_vehicles", "total_metres",
                "total_seconds", "dwell_seconds", "mean_bus_observed_speed_kmh", "source")
BINS = ["2026-11-07T21:00:00+00:00", "2026-11-07T21:15:00+00:00"]


@pytest.fixture
def night(tmp_path):
    """Joined bus speeds: Far Road (edges 5, 6) leaves the study area, High Street (edge 0) is inside it."""
    def row(eid, when, n, speed):
        return dict(zip(NIGHT_FIELDS, (eid, "", when, n, n, 1, 1, 0, speed, "observed")))

    return _write_csv(tmp_path / "speeds_night.csv", [
        row("5", "2026-11-07 21:00:00+00:00", 2, "18.04"),
        row("6", "2026-11-07 21:15:00+00:00", 1, "9"),
        row("0", "2026-11-07 21:00:00+00:00", 3, "12"),
        row("5", "2026-11-07 23:00:00+00:00", 1, "30"),
        row("6", "2026-11-07 21:00:00+00:00", 0, ""),
    ])


def test_city_speed_layer_keeps_observed_segments_outside_the_study_area_in_the_shared_bins(graph, night):
    features, table = export.city_speed_layer(graph, BINS, night)
    assert table == {"bins": BINS, "edges": {"5": {"s": [18.0, None], "o": "10"}, "6": {"s": [None, 9.0], "o": "01"}}}
    [far] = features
    assert far["properties"]["name"] == "Far Road" and {far["properties"]["id"], far["properties"]["rev"]} == {"5", "6"}
    assert far["geometry"]["coordinates"][-1] == pytest.approx([-3.25, 55.9505], abs=1e-5)


def test_city_speed_layer_is_none_before_the_night_is_joined(graph, tmp_path):
    assert export.city_speed_layer(graph, BINS, tmp_path / "speeds_night.csv") is None


@pytest.fixture
def run_main(graph, tmp_path, monkeypatch):
    monkeypatch.setattr(export, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(export, "load_graph", lambda: graph)
    for name in ("UE_PATH", "UE_META_PATH", "CROSSINGS_PATH", "DFT_PATH", "REVIEW_PATH", "CLAUSES_PATH",
                 "SOURCES_DIR", "FIELD_PATH", "CONCERNS_PATH", "PREDICTIONS_DIR", "SPEEDS_NIGHT_PATH"):
        monkeypatch.setattr(export, name, tmp_path / "missing" / name)

    def run():
        export.main()
        return json.loads((tmp_path / "out" / "meta.json").read_text(encoding="utf-8"))
    return run


def test_main_lists_no_layers_before_the_pipeline_has_run(run_main, tmp_path):
    meta = run_main()
    assert meta == {"bbox": list(export.STUDY_BBOX), "layers": {}}
    assert {p.name for p in (tmp_path / "out").iterdir()} == {"meta.json", "network.geojson", "streets.json"}


def test_main_labels_counts_estimated_unless_every_point_was_counted(run_main, dft, monkeypatch):
    monkeypatch.setattr(export, "DFT_PATH", dft[0])
    monkeypatch.setattr(export, "REVIEW_PATH", dft[1])
    counts = run_main()["layers"]["counts"]
    assert counts["basis"] == "estimated"
    assert "1 of 2 counted" in counts["note"]


def test_main_lists_traffic_with_its_hour(run_main, tmp_path, monkeypatch):
    ue = _write_csv(tmp_path / "ue.csv", [
        {"edge_id": eid, "length_m": "1", "free_flow_time_s": r["t0"], "ue_time_s": r["t"],
         "flow_veh_h": r["flow"], "vc": r["vc"]} for eid, r in UE.items()
    ])
    meta_path = tmp_path / "ue.json"
    meta_path.write_text(json.dumps({"hour": "08:00-09:00 weekday"}), encoding="utf-8")
    monkeypatch.setattr(export, "UE_PATH", ue)
    monkeypatch.setattr(export, "UE_META_PATH", meta_path)
    traffic = run_main()["layers"]["traffic"]
    assert traffic["basis"] == "estimated" and traffic["hour"] == "08:00-09:00 weekday"


def test_main_writes_rules_and_concerns_onto_the_network(run_main, tmp_path, monkeypatch):
    clauses, sources = _clause_files(tmp_path, [clause("high", "High Street", "no_motor_vehicles"),
                                                clause("advert", "High Street", "no_buses")])
    zero = dict.fromkeys(THEMES, "0")
    field = _write_csv(tmp_path / "field.csv", [{"edge_id": "1", **zero, "noise": "3", "basis": "per-street counts"}])
    for name, path in (("CLAUSES_PATH", clauses), ("SOURCES_DIR", sources), ("FIELD_PATH", field)):
        monkeypatch.setattr(export, name, path)
    layers = run_main()["layers"]
    assert layers["rules"]["basis"] == "verified"
    assert {k: layers["rules"][k] for k in ("applied", "osm_wrong", "abstained")} == {"applied": 1, "osm_wrong": 1,
                                                                                    "abstained": 1}
    assert layers["concerns"]["note"].endswith("as per-street counts")
    network = json.loads((tmp_path / "out" / "network.geojson").read_text(encoding="utf-8"))
    high = _props(network["features"])["High Street"]
    assert (high["rule"], high["osm_ok"], high["concern"], high["concern_theme"]) == ("no_motor_vehicles", False, 3.0,
                                                                                   "noise")
    assert len(json.loads((tmp_path / "out" / "rules.json").read_text(encoding="utf-8"))["clauses"]) == 2


def test_main_writes_the_speeds_layer_labelled_bus_observed(run_main, predictions, tmp_path, monkeypatch):
    monkeypatch.setattr(export, "PREDICTIONS_DIR", predictions)
    speeds = run_main()["layers"]["speeds"]
    assert (speeds["label"], speeds["basis"], speeds["share_predicted"]) == ("Bus-observed speed", "predicted", 0.667)
    assert speeds["note"].startswith("Speed of buses, not of all traffic")
    assert json.loads((tmp_path / "out" / "speeds.json").read_text(encoding="utf-8"))["edges"]["12"]["o"] == "10"


def test_main_writes_city_speeds_only_alongside_the_prediction_table(run_main, predictions, night, tmp_path, monkeypatch):
    monkeypatch.setattr(export, "SPEEDS_NIGHT_PATH", night)
    assert "speeds" not in run_main()["layers"]
    assert not (tmp_path / "out" / "city_speeds.json").exists()
    monkeypatch.setattr(export, "PREDICTIONS_DIR", predictions)
    speeds = run_main()["layers"]["speeds"]
    assert speeds["city_edges"] == 2
    city = json.loads((tmp_path / "out" / "city_speeds.json").read_text(encoding="utf-8"))
    study = json.loads((tmp_path / "out" / "speeds.json").read_text(encoding="utf-8"))
    assert city["bins"] == study["bins"] and city["edges"]["5"]["o"] == "10"
    assert len(json.loads((tmp_path / "out" / "city_network.geojson").read_text(encoding="utf-8"))["features"]) == 1


def test_main_leaves_out_a_layer_whose_input_is_broken(run_main, tmp_path, monkeypatch, capsys):
    broken = tmp_path / "clauses.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(export, "CLAUSES_PATH", broken)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "rules.json").write_text("{}", encoding="utf-8")
    meta = run_main()
    assert "rules" not in meta["layers"]
    assert "warning: left out the rules layer: JSONDecodeError" in capsys.readouterr().out
    assert not (tmp_path / "out" / "rules.json").exists()
    assert (tmp_path / "out" / "network.geojson").exists()
