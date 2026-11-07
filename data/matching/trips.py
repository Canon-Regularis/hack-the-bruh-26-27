"""Group raw pings into per-vehicle trips, breaking on time gaps.

A trip is one bus's pings in time order. A gap too large to represent as one
continuous path -- the bus went offline or parked -- starts a new trip rather
than being stitched across. A trip of fewer than two pings cannot become a
PingPair, so it is dropped here rather than passed on.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

DEFAULT_MAX_GAP_S = 300.0  # 5 minutes


@dataclass(frozen=True)
class Ping:
    """One raw GPS fix, already deduplicated by (vehicle_id, time_utc)."""

    vehicle_id: str
    time_utc: datetime
    lat: float
    lon: float


def pings_from_rows(rows):
    """Parse and deduplicate the collector's raw CSV rows into Ping objects.

    A row with a missing or unparseable vehicle_id, vehicle_timestamp, lat or
    lon is skipped, as is one whose timestamp has no UTC offset -- Ping
    requires a usable fix, not that every raw row becomes one. A repeated
    (vehicle_id, time_utc) fix -- the same underlying GPS fix seen across
    several polls -- keeps its first occurrence only.
    """
    seen = {}
    for row in rows:
        vehicle_id = row.get("vehicle_id")
        if not vehicle_id:
            continue
        try:
            time_utc = datetime.fromisoformat(row["vehicle_timestamp"])
            lat = float(row["lat"])
            lon = float(row["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        if time_utc.utcoffset() != timedelta(0):
            continue
        seen.setdefault((vehicle_id, time_utc), Ping(vehicle_id, time_utc, lat, lon))
    return list(seen.values())


def split_trips(pings, max_gap_s=DEFAULT_MAX_GAP_S):
    """Return trips: lists of >=2 Ping, one vehicle each, sorted and gap-split."""
    by_vehicle = {}
    for ping in pings:
        by_vehicle.setdefault(ping.vehicle_id, []).append(ping)

    gap = timedelta(seconds=max_gap_s)
    trips = []
    for vehicle_pings in by_vehicle.values():
        vehicle_pings.sort(key=lambda p: p.time_utc)
        trip = [vehicle_pings[0]]
        for ping in vehicle_pings[1:]:
            if ping.time_utc - trip[-1].time_utc > gap:
                if len(trip) > 1:
                    trips.append(trip)
                trip = [ping]
            else:
                trip.append(ping)
        if len(trip) > 1:
            trips.append(trip)
    return trips
