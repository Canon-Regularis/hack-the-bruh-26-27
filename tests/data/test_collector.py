"""Collector logic that doesn't need the network: payload shapes and row parsing."""

import json

import pytest

from collector import (
    DEFAULT_BBOX,
    build_url,
    extract_vehicles,
    normalise,
    parse_row,
    pick_lat_lon,
    pick_service,
)


def test_build_url_includes_all_bbox_params():
    url = build_url(DEFAULT_BBOX)
    assert url.startswith("https://bustimes.org/vehicles.json?")
    for key in ("xmin", "ymin", "xmax", "ymax"):
        assert f"{key}={DEFAULT_BBOX[key]}" in url


def test_extract_vehicles_plain_list():
    payload = [{"id": 1}, {"id": 2}]
    assert extract_vehicles(payload) == payload


def test_extract_vehicles_geojson_feature_collection():
    payload = {"type": "FeatureCollection", "features": [{"id": 1}]}
    assert extract_vehicles(payload) == [{"id": 1}]


def test_extract_vehicles_wrapped_in_known_key():
    payload = {"vehicles": [{"id": 1}]}
    assert extract_vehicles(payload) == [{"id": 1}]


def test_extract_vehicles_unrecognised_shape_raises():
    with pytest.raises(ValueError):
        extract_vehicles({"nope": "not a list"})


def test_normalise_flattens_geojson_feature():
    feature = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [-3.19, 55.95]},
        "properties": {"id": "42", "service": "23"},
    }
    flat = normalise(feature)
    assert flat["id"] == "42"
    assert flat["service"] == "23"
    assert flat["_lon"] == -3.19
    assert flat["_lat"] == 55.95


def test_normalise_passes_flat_record_through():
    record = {"id": "42", "lat": 55.95, "lon": -3.19}
    assert normalise(record) == record


def test_pick_lat_lon_from_coordinates_key():
    record = {"coordinates": [-3.19, 55.95]}
    assert pick_lat_lon(record) == (55.95, -3.19)


def test_pick_lat_lon_from_flat_fields():
    record = {"latitude": 55.95, "longitude": -3.19}
    assert pick_lat_lon(record) == (55.95, -3.19)


def test_pick_service_from_nested_object():
    # bustimes.org's real shape: {"service": {"line_name": "092", "url": "..."}}
    record = {"service": {"line_name": "092", "url": "/services/092-edinburgh-aberdeen"}}
    assert pick_service(record) == "092"


def test_pick_service_from_flat_string():
    assert pick_service({"service": "23"}) == "23"


def test_parse_row_from_geojson_feature():
    feature = {
        "type": "Feature",
        "geometry": {"coordinates": [-3.19, 55.95]},
        "properties": {
            "id": "42",
            "service": "23",
            "heading": 180,
            "datetime": "2026-11-07T18:00:00Z",
        },
    }
    row = parse_row(feature, polled_at="2026-11-07T18:00:30Z")
    assert row["vehicle_id"] == "42"
    assert row["service"] == "23"
    assert row["bearing"] == 180
    assert row["vehicle_timestamp"] == "2026-11-07T18:00:00Z"
    assert row["lat"] == 55.95
    assert row["lon"] == -3.19
    assert json.loads(row["raw"]) == feature


def test_parse_row_from_live_style_record_with_nested_service():
    record = {
        "id": -960168268,
        "coordinates": [-3.1910979747772217, 55.95545959472656],
        "heading": None,
        "datetime": "2026-11-07T19:31:20+00:00",
        "service": {"line_name": "092", "url": "/services/092-edinburgh-aberdeen"},
    }
    row = parse_row(record, polled_at="2026-11-07T19:31:30+00:00")
    assert row["vehicle_id"] == -960168268
    assert row["service"] == "092"
    assert row["bearing"] is None
    assert row["lat"] == 55.95545959472656
    assert row["lon"] == -3.1910979747772217
