"""Collector logic that doesn't need the network: payload shapes and row parsing."""

import csv
import json

import pytest
import requests

import collector
from collector import (
    CSV_FIELDS,
    DEFAULT_BBOX,
    MAX_WAIT_SECONDS,
    build_url,
    extract_vehicles,
    next_wait,
    normalise,
    parse_row,
    pick_lat_lon,
    pick_service,
    retry_after_seconds,
    write_rows,
)

GOOD_RECORD = {
    "id": 1,
    "coordinates": [-3.19, 55.95],
    "heading": None,
    "datetime": "2026-11-07T19:31:20+00:00",
    "service": {"line_name": "23"},
}


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


class StopPolling(Exception):
    """Ends the otherwise endless poll loop from inside a test."""


def run_polls(monkeypatch, tmp_path, replies):
    """Run run_collect over a list of replies and return the waits it asked for.

    A reply that is an exception is raised by fetch, anything else is returned.
    """
    remaining = list(replies)
    waits = []

    def fake_fetch(bbox):
        reply = remaining.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def fake_sleep(seconds):
        waits.append(seconds)
        if not remaining:
            raise StopPolling

    monkeypatch.setattr(collector, "fetch", fake_fetch)
    monkeypatch.setattr(collector.time, "sleep", fake_sleep)
    with pytest.raises(StopPolling):
        collector.run_collect(DEFAULT_BBOX, 30, tmp_path)
    return waits


def http_error(status, headers=None):
    response = requests.Response()
    response.status_code = status
    response.headers.update(headers or {})
    return requests.exceptions.HTTPError(f"{status} error", response=response)


def csv_rows(tmp_path):
    with open(tmp_path / "pings.csv", newline="") as handle:
        return list(csv.DictReader(handle))


def test_write_rows_skips_bad_records_and_keeps_good_ones(tmp_path):
    with open(tmp_path / "rows.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        written, skipped = write_rows(writer, [GOOD_RECORD, None, GOOD_RECORD], "now")
    assert (written, skipped) == (2, 1)


@pytest.mark.parametrize(
    "payload",
    [
        [GOOD_RECORD, None],
        [GOOD_RECORD, {"properties": None, "geometry": None}],
        [GOOD_RECORD, "not a record"],
    ],
)
def test_bad_record_does_not_stop_polling(monkeypatch, tmp_path, payload):
    run_polls(monkeypatch, tmp_path, [payload, [GOOD_RECORD]])
    assert [row["vehicle_id"] for row in csv_rows(tmp_path)] == ["1", "1"]


def test_reply_holding_only_error_strings_does_not_stop_polling(monkeypatch, tmp_path):
    run_polls(monkeypatch, tmp_path, [{"errors": ["rate limited"]}, [GOOD_RECORD]])
    assert len(csv_rows(tmp_path)) == 1


def test_raw_reply_is_kept_even_when_a_record_is_skipped(monkeypatch, tmp_path):
    run_polls(monkeypatch, tmp_path, [[GOOD_RECORD, None], [GOOD_RECORD]])
    lines = (tmp_path / "pings.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["payload"] == [GOOD_RECORD, None]


def test_next_wait_is_the_interval_with_no_failures():
    assert next_wait(30, 0) == 30


def test_next_wait_doubles_with_each_failure():
    assert [next_wait(30, n) for n in (1, 2, 3)] == [60, 120, 240]


def test_next_wait_stops_at_the_cap():
    assert next_wait(30, 50) == MAX_WAIT_SECONDS


def test_next_wait_never_drops_below_the_interval():
    assert next_wait(1200, 1) == 1200


def test_next_wait_uses_retry_after_when_it_is_longer():
    assert next_wait(30, 1, retry_after=300) == 300


def test_next_wait_ignores_retry_after_when_it_is_shorter():
    assert next_wait(30, 3, retry_after=5) == 240


def test_retry_after_seconds_reads_the_header():
    assert retry_after_seconds(http_error(429, {"Retry-After": "120"})) == 120


@pytest.mark.parametrize(
    "error",
    [
        http_error(429),
        http_error(429, {"Retry-After": "Sat, 07 Nov 2026 21:00:00 GMT"}),
        requests.exceptions.ConnectionError("no response at all"),
        ValueError("unrecognised payload shape"),
    ],
)
def test_retry_after_seconds_is_none_when_there_is_no_usable_value(error):
    assert retry_after_seconds(error) is None


def test_polling_backs_off_on_failures_and_resets_after_a_success(monkeypatch, tmp_path):
    failure = http_error(429)
    waits = run_polls(monkeypatch, tmp_path, [failure, failure, [GOOD_RECORD], [GOOD_RECORD]])
    assert waits == pytest.approx([60, 120, 30, 30], abs=1)


def test_polling_respects_retry_after(monkeypatch, tmp_path):
    failure = http_error(429, {"Retry-After": "300"})
    waits = run_polls(monkeypatch, tmp_path, [failure, [GOOD_RECORD]])
    assert waits == pytest.approx([300, 30], abs=1)
