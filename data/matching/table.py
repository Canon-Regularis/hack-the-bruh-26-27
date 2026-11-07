"""The hand-off table from map matching to the speeds stage.

Matthew Li owns matching. PingPair is the format the speeds stage assumes until
that code exists. match_pings is a skeleton that raises, so it can never return
made-up matches. See Q5 in DECISIONS.md.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

SOURCES = {"observed", "synthetic"}


class MatchingNotAvailable(NotImplementedError):
    """Raised until Matthew Li's matching replaces the skeleton."""


@dataclass(frozen=True)
class PingPair:
    """Two consecutive pings from one bus, after map matching.

    Times are the GPS fix times in UTC, not the time we polled for them.
    Positions are (latitude, longitude). Edges are (edge_id, metres) in the order
    driven. A bus standing still lists the edge it is on, with 0 metres.
    """

    vehicle_id: str
    start_time_utc: datetime
    end_time_utc: datetime
    start_position: tuple[float, float]
    end_position: tuple[float, float]
    edges: list[tuple[str, float]]
    source: str

    def __post_init__(self):
        # A time with no timezone has no offset, so this rejects those too.
        for moment in (self.start_time_utc, self.end_time_utc):
            if moment.utcoffset() != timedelta(0):
                raise ValueError("ping times must be in UTC")
        if self.end_time_utc <= self.start_time_utc:
            raise ValueError("a ping pair must end after it starts")
        if not self.edges:
            raise ValueError("a ping pair needs at least one edge")
        if any(metres < 0 for _, metres in self.edges):
            raise ValueError("edge distances cannot be negative")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {sorted(SOURCES)}")

    @property
    def seconds(self):
        return (self.end_time_utc - self.start_time_utc).total_seconds()


def match_pings(pings):
    """Turn raw pings into PingPair rows. Skeleton for Matthew Li's matching."""
    raise MatchingNotAvailable("match_pings needs Matthew Li's matching code")
