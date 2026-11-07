"""Turn matched ping pairs into bus-observed speeds per edge per 15-minute bin.

Everything stays in metres and seconds until the last step, which gives km/h.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from speeds.dwell import is_dwell

BIN_MINUTES = 15
OUTPUT_COLUMNS = [
    "edge_id",
    "street_name",
    "bin_start_utc",
    "n_traversals",
    "n_vehicles",
    "total_metres",
    "total_seconds",
    "dwell_seconds",
    "mean_bus_observed_speed_kmh",
    "source",
]


@dataclass
class Traversal:
    """One bus passing along one edge, with moving and dwell time kept apart."""

    vehicle_id: str
    edge_id: str
    enter_time: datetime
    metres: float
    seconds: float
    dwell_seconds: float
    source: str


def bin_start(moment):
    """Return the start of the 15-minute bin holding this moment."""
    minute = moment.minute - moment.minute % BIN_MINUTES
    return moment.replace(minute=minute, second=0, microsecond=0)


def split_pair_across_edges(pair):
    """Yield (edge_id, enter_time, metres, seconds) for each edge in a pair.

    Two pings cannot show speed changes between them, so time is shared out by
    distance, as if the bus held one speed. A bus that did not move has no
    distance to share by, so its time is shared equally.
    """
    total_metres = sum(metres for _, metres in pair.edges)
    enter_time = pair.start_time_utc
    for edge_id, metres in pair.edges:
        share = metres / total_metres if total_metres else 1 / len(pair.edges)
        seconds = pair.seconds * share
        if seconds > 0:
            yield edge_id, enter_time, metres, seconds
            enter_time += timedelta(seconds=seconds)


def build_traversals(pairs, stops):
    """Join every ping pair into traversals, storing dwell time separately."""
    traversals = []
    current = None
    previous = None
    for pair in sorted(pairs, key=lambda p: (p.source, p.vehicle_id, p.start_time_utc)):
        follows_on = (
            previous is not None
            and pair.vehicle_id == previous.vehicle_id
            and pair.source == previous.source
            and pair.start_time_utc == previous.end_time_utc
        )
        if not follows_on:
            current = None  # A new bus, a new source, or a gap in its pings.
        previous = pair

        # A bus can dwell partway along an edge, so a dwell does not end the
        # traversal. The bus carries on along the same edge afterwards.
        dwell = is_dwell(pair, stops)
        for edge_id, enter_time, metres, seconds in split_pair_across_edges(pair):
            if current is None or current.edge_id != edge_id:
                current = Traversal(
                    pair.vehicle_id, edge_id, enter_time, 0.0, 0.0, 0.0, pair.source
                )
                traversals.append(current)
            if dwell:
                current.dwell_seconds += seconds
            else:
                # After a wait, bin by when the bus moved, not when it arrived.
                if current.seconds == 0:
                    current.enter_time = enter_time
                current.metres += metres
                current.seconds += seconds
    return traversals


def edge_speed_table(pairs, stops, street_names):
    """Return one row per edge, 15-minute bin and source label.

    total_seconds is moving time, and dwell_seconds holds the rest. The speed is
    total metres over total seconds, not an average of each bus's
    speed. That matches how long a street takes to drive,
    and stops a very short traversal from swinging the result. Dwell is left out
    of the speed but stored, so a speed with dwell can be recomputed. A row with
    only dwell has no moving time, so its speed is empty.
    """
    traversals = build_traversals(pairs, stops)
    if not traversals:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rows = pd.DataFrame(traversals)
    rows["bin_start_utc"] = rows["enter_time"].map(bin_start)
    # Only a bus that moved counts as a traversal. Both counts skip empty values.
    rows["moving_vehicle"] = rows["vehicle_id"].where(rows["seconds"] > 0)
    table = rows.groupby(["edge_id", "bin_start_utc", "source"], as_index=False).agg(
        n_traversals=("moving_vehicle", "count"),
        n_vehicles=("moving_vehicle", "nunique"),
        total_metres=("metres", "sum"),
        total_seconds=("seconds", "sum"),
        dwell_seconds=("dwell_seconds", "sum"),
    )
    metres_per_second = table["total_metres"] / table["total_seconds"]
    table["mean_bus_observed_speed_kmh"] = metres_per_second * 3.6
    # A missing edge means two stages disagree on edge IDs, so let it raise.
    table["street_name"] = [street_names[edge_id] for edge_id in table["edge_id"]]
    return table[OUTPUT_COLUMNS]
