"""Bus-as-sensor collector for bustimes.org.

Third-party, unauthenticated API: https://bustimes.org/vehicles.json
No published terms of use were found, so confirm with the site owner before
leaving this polling overnight (see docs/HackTheBruh_Plan_v2.md, section 1).
TfE's own API is closed and BODS has no Lothian coverage, so this is the
live source; the DataShare 2014-15 traces are the fallback if this is
refused, blocked or goes down.

Run from data/, since data/ is the pytest/import root (see pytest.ini).

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
from datetime import datetime, timezone
from pathlib import Path

import requests

BUSTIMES_URL = "https://bustimes.org/vehicles.json"
# Deliberately generous and city-wide, not the Old Town study area: the plan
# (docs/HackTheBruh_Plan_v2.md section 1) says to fix the real bbox only
# after plotting the first hour of live bustimes.org pings. Narrow it then.
DEFAULT_BBOX = {"xmin": -3.35, "ymin": 55.86, "xmax": -3.05, "ymax": 56.00}
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "raw"
DEFAULT_INTERVAL = 30
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
            try:
                payload = fetch(bbox)
                vehicles = extract_vehicles(payload)
            except Exception as exc:
                consecutive_failures += 1
                print(f"[{polled_at}] poll failed ({exc}); {consecutive_failures} in a row")
            else:
                consecutive_failures = 0
                with open(raw_path, "a") as raw_f:
                    raw_f.write(json.dumps({"polled_at": polled_at, "payload": payload}) + "\n")
                for record in vehicles:
                    writer.writerow(parse_row(record, polled_at))
                csv_f.flush()
                print(f"[{polled_at}] {len(vehicles)} vehicles")

            sleep_for = interval - (time.monotonic() - loop_start)
            if sleep_for > 0:
                time.sleep(sleep_for)


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

    args = parser.parse_args()

    if args.command == "test":
        sys.exit(run_test(bbox_from_args(args)))
    elif args.command == "collect":
        try:
            run_collect(bbox_from_args(args), args.interval, args.outdir)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
