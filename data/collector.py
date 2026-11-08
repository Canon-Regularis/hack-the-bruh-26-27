"""Bus-as-sensor collector for bustimes.org.

Third-party, unauthenticated API: https://bustimes.org/vehicles.json
No published terms of use were found, so confirm with the site owner before
leaving this polling overnight (see docs/HackTheBruh_Plan_v2.md, section 1).
TfE's own API is closed and BODS has no Lothian coverage, so this is the
live source; the DataShare 2014-15 traces are the fallback if this is
refused, blocked or goes down.

Run as a script from data/. For how tests import this module, see the
pytest.ini at the repo root.

Usage:
    python3 collector.py test                    # one-shot sanity check
    python3 collector.py collect                  # poll every 30s -> ./raw
    python3 collector.py collect --interval 15 --outdir raw
    python3 collector.py collect --xmin -3.215 --ymin 55.94 --xmax -3.17 --ymax 55.958
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BUSTIMES_URL = "https://bustimes.org/vehicles.json"
# Deliberately generous and city-wide, not the Old Town study area: the plan
# (docs/HackTheBruh_Plan_v2.md section 1) says to fix the real bbox only
# after plotting the first hour of live bustimes.org pings. Narrow it then.
DEFAULT_BBOX = {"xmin": -3.35, "ymin": 55.86, "xmax": -3.05, "ymax": 56.00}
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "raw"
DEFAULT_INTERVAL = 30
MAX_WAIT_SECONDS = 600
USER_AGENT = "hacktheburgh-collector/0.1 (HackTheBurgh XIII, Team Civilian$)"

# bustimes.org has no published schema for /vehicles.json, so pull by alias
# and keep the raw record too, so a field-name mismatch doesn't crash the
# whole poll loop.
FIELD_ALIASES = {
    "vehicle_id": ["id", "vehicle_id", "vehicle"],
    "bearing": ["heading", "bearing", "direction"],
    "vehicle_timestamp": ["datetime", "timestamp", "recorded_at", "when"],
}
SERVICE_ALIASES = ["service", "service_name", "line_name", "route"]
CSV_FIELDS = ["polled_at", "vehicle_id", "service", "bearing", "vehicle_timestamp", "lat", "lon", "raw"]


def build_url(bbox, base_url=BUSTIMES_URL):
    params = "&".join(f"{key}={value}" for key, value in bbox.items())
    return f"{base_url}?{params}"


def fetch(bbox, timeout=10.0):
    resp = requests.get(
        build_url(bbox),
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def extract_vehicles(payload):
    """Pull the list of vehicle records out of whatever shape the API returns."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("features"), list):  # GeoJSON FeatureCollection
            return payload["features"]
        for key in ("vehicles", "data", "results", "locations"):
            if isinstance(payload.get(key), list):
                return payload[key]
        for value in payload.values():
            if isinstance(value, list):
                return value
    raise ValueError(f"unrecognised payload shape: {type(payload)}")


def normalise(record):
    """Flatten a GeoJSON Feature to a flat dict; pass a flat record through."""
    if "properties" in record and "geometry" in record:
        flat = dict(record["properties"])
        coords = (record["geometry"] or {}).get("coordinates")
        if coords:
            flat["_lon"], flat["_lat"] = coords[0], coords[1]
        return flat
    return record


def pick(record, aliases):
    for key in aliases:
        if key in record:
            return record[key]
    return None


def pick_service(record):
    """Service may arrive as a nested {line_name, url} object or a flat string."""
    value = pick(record, SERVICE_ALIASES)
    if isinstance(value, dict):
        return value.get("line_name") or value.get("name") or value.get("url")
    return value


def pick_lat_lon(record):
    """Coordinates may arrive as a GeoJSON [lon, lat] pair or flat lat/lon keys."""
    if "_lat" in record and "_lon" in record:
        return record["_lat"], record["_lon"]
    coords = record.get("coordinates")
    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        return coords[1], coords[0]
    lat = pick(record, ["lat", "latitude"])
    lon = pick(record, ["lon", "lng", "longitude"])
    return lat, lon


def parse_row(record, polled_at):
    flat = normalise(record)
    row = {"polled_at": polled_at}
    for field, aliases in FIELD_ALIASES.items():
        row[field] = pick(flat, aliases)
    row["service"] = pick_service(flat)
    row["lat"], row["lon"] = pick_lat_lon(flat)
    row["raw"] = json.dumps(record, separators=(",", ":"))
    return row


def write_rows(writer, vehicles, polled_at):
    """Write a row per record, skipping any record that cannot be parsed.

    The API has no published schema, so one odd record must not end an overnight
    run. Skipped records are still in the raw reply in pings.jsonl.
    """
    written = skipped = 0
    for record in vehicles:
        try:
            row = parse_row(record, polled_at)
        except Exception as exc:
            skipped += 1
            print(f"[{polled_at}] skipped a record ({exc!r})")
            continue
        writer.writerow(row)
        written += 1
    return written, skipped


def retry_after_seconds(exc):
    """Return the Retry-After seconds from a failed request, or None.

    The header can also hold a date. That falls back to the normal back-off.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        return max(0, int(response.headers.get("Retry-After")))
    except (TypeError, ValueError):
        return None


def next_wait(interval, failures, retry_after=None):
    """Return the seconds to wait before the next poll.

    Doubles with each failure in a row up to MAX_WAIT_SECONDS, and never waits
    less than the interval or less than the site asked for.
    """
    if failures == 0:
        return interval
    backoff = max(interval, min(interval * 2**failures, MAX_WAIT_SECONDS))
    return max(backoff, retry_after or 0)


def run_test(bbox) -> int:
    url = build_url(bbox)
    print(f"GET {url}")
    t0 = time.monotonic()
    try:
        payload = fetch(bbox)
    except requests.exceptions.RequestException as exc:
        print(f"FAILED: {exc}")
        return 1
    print(f"OK in {time.monotonic() - t0:.2f}s")

    try:
        vehicles = extract_vehicles(payload)
    except ValueError as exc:
        print(f"Got JSON but couldn't find the vehicle list: {exc}")
        print(json.dumps(payload, indent=2)[:2000])
        return 1

    print(f"{len(vehicles)} vehicles in this snapshot")
    if vehicles:
        print("Sample record:")
        print(json.dumps(vehicles[0], indent=2))
        print("Parsed:", parse_row(vehicles[0], polled_at="test"))
    return 0


def run_collect(bbox, interval, outdir) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "pings.jsonl"
    csv_path = outdir / "pings.csv"
    write_header = not csv_path.exists()

    url = build_url(bbox)
    print(f"Collecting from {url} every {interval}s")
    print("  bustimes.org is third-party and unauthenticated; confirm the")
    print("  site owner is OK with an overnight poll before leaving this running.")
    print(f"  raw  -> {raw_path}")
    print(f"  rows -> {csv_path}")

    with open(csv_path, "a", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
            csv_f.flush()

        consecutive_failures = 0
        while True:
            polled_at = datetime.now(timezone.utc).isoformat()
            loop_start = time.monotonic()
            retry_after = None
            try:
                payload = fetch(bbox)
                vehicles = extract_vehicles(payload)
            except Exception as exc:
                consecutive_failures += 1
                retry_after = retry_after_seconds(exc)
                print(f"[{polled_at}] poll failed ({exc}); {consecutive_failures} in a row")
            else:
                consecutive_failures = 0
                with open(raw_path, "a") as raw_f:
                    raw_f.write(json.dumps({"polled_at": polled_at, "payload": payload}) + "\n")
                written, skipped = write_rows(writer, vehicles, polled_at)
                csv_f.flush()
                print(f"[{polled_at}] {written} vehicles, {skipped} skipped")

            wait = next_wait(interval, consecutive_failures, retry_after)
            sleep_for = wait - (time.monotonic() - loop_start)
            if sleep_for > 0:
                time.sleep(sleep_for)


def decode_track(polyline):
    """Decode bustimes.org's time_aware_polyline into (lat, lon, unix seconds) points.

    Google polyline varints in threes: lon and lat in 1e-5 degrees, then seconds, each a delta.
    """
    values, i = [], 0
    while i < len(polyline):
        shift = result = 0
        while True:
            b = ord(polyline[i]) - 63
            i += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        values.append(~(result >> 1) if result & 1 else result >> 1)
    points, lon, lat, t = [], 0, 0, 0
    for k in range(0, len(values) - 2, 3):
        lon, lat, t = lon + values[k], lat + values[k + 1], t + values[k + 2]
        points.append((lat / 1e5, lon / 1e5, t))
    return points


def track_rows(journey, start, end, polled_at):
    """Return collector-shaped rows for a past journey's points with start <= time < end."""
    vehicle = journey.get("vehicle") or {}
    raw = json.dumps({"journey_id": journey["id"], "vehicle": vehicle.get("slug"),
                      "route_name": journey.get("route_name"), "source": "bustimes.org history"})
    rows = []
    for lat, lon, t in decode_track(journey.get("time_aware_polyline") or ""):
        when = datetime.fromtimestamp(t, timezone.utc)
        if start <= when < end:
            rows.append({"polled_at": polled_at, "vehicle_id": vehicle["id"], "service": journey.get("route_name"),
                         "bearing": None, "vehicle_timestamp": when.isoformat(), "lat": lat, "lon": lon, "raw": raw})
    return rows


def get_json(url, min_gap=0.5, attempts=5):
    """GET bustimes.org JSON at most once per min_gap seconds, backing off on failures."""
    for failures in range(attempts):
        t0 = time.monotonic()
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            if failures == attempts - 1:
                raise
            time.sleep(next_wait(min_gap, failures + 1, retry_after_seconds(exc)))
        finally:
            time.sleep(max(0.0, min_gap - (time.monotonic() - t0)))


def run_history(vehicle_ids, operators, start, end, out_path, lead=timedelta(hours=2)):
    """Write the recorded tracks of every journey running between start and end, from bustimes.org."""
    api = "https://bustimes.org/api"
    ids = set(vehicle_ids)
    for op in operators:
        url = f"{api}/vehicles/?operator={op}&limit=1000"
        while url:
            page = get_json(url)
            ids |= {v["id"] for v in page["results"] if not v.get("withdrawn")}
            url = page.get("next")
    print(f"{len(ids)} vehicles; journeys from {start - lead} to {end}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    polled_at = datetime.now(timezone.utc).isoformat()
    written = journeys = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for n, vid in enumerate(sorted(ids), 1):
            url = f"{api}/vehiclejourneys/?vehicle={vid}&date={start.date()}"
            while url:
                page = get_json(url)
                times = [datetime.fromisoformat(j["datetime"].replace("Z", "+00:00")) for j in page["results"]]
                for j, when in zip(page["results"], times):
                    if start - lead <= when < end:
                        rows = track_rows(get_json(f"{api}/vehiclejourneys/{j['id']}/details.json"), start, end, polled_at)
                        writer.writerows(rows)
                        written, journeys = written + len(rows), journeys + bool(rows)
                url = page.get("next") if times and min(times) >= start - lead else None
            if n % 25 == 0:
                f.flush()
                print(f"{n}/{len(ids)} vehicles, {journeys} journeys, {written} points")
    print(f"done: {journeys} journeys, {written} points -> {out_path}")


def bbox_from_args(args):
    return {"xmin": args.xmin, "ymin": args.ymin, "xmax": args.xmax, "ymax": args.ymax}


def add_bbox_args(parser):
    parser.add_argument("--xmin", type=float, default=DEFAULT_BBOX["xmin"])
    parser.add_argument("--ymin", type=float, default=DEFAULT_BBOX["ymin"])
    parser.add_argument("--xmax", type=float, default=DEFAULT_BBOX["xmax"])
    parser.add_argument("--ymax", type=float, default=DEFAULT_BBOX["ymax"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    test_p = sub.add_parser("test", help="one-shot check that bustimes.org is alive")
    add_bbox_args(test_p)

    collect_p = sub.add_parser(
        "collect", help="poll bustimes.org forever, appending data to disk"
    )
    add_bbox_args(collect_p)
    collect_p.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL, help="seconds between polls (default 30)"
    )
    collect_p.add_argument(
        "--outdir", type=Path, default=DEFAULT_OUTDIR, help="where to write data (default ./raw)"
    )

    history_p = sub.add_parser("history", help="fetch past journeys' recorded tracks for a time window")
    history_p.add_argument("--start", type=datetime.fromisoformat, required=True)
    history_p.add_argument("--end", type=datetime.fromisoformat, required=True)
    history_p.add_argument("--vehicles-from", type=Path, help="a pings CSV whose vehicle ids to include")
    history_p.add_argument("--operators", nargs="*", default=["LOTH"])
    history_p.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "history":
        ids = set()
        if args.vehicles_from:
            with open(args.vehicles_from, newline="") as f:
                ids = {int(r["vehicle_id"]) for r in csv.DictReader(f) if r["vehicle_id"].lstrip("-").isdigit()
                       and int(r["vehicle_id"]) > 0}
        run_history(ids, args.operators, args.start, args.end, args.out)
    elif args.command == "test":
        sys.exit(run_test(bbox_from_args(args)))
    elif args.command == "collect":
        try:
            run_collect(bbox_from_args(args), args.interval, args.outdir)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
