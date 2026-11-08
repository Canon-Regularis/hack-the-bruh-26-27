"""Spread published consultation concerns onto edges, per theme (ALG-S1-8, #25).

s_e^j = sum_i severity_i * count_i * K(dist(e, loc_i)) * 1[theme_i = j], with a Gaussian K
scaled so K(0) = 1. Stand-in for orchid's held public_voice until #35 swaps it in.
Run from the repo root: ``python -m src.concerns.concern_field``.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

from engine.orchid import street_network as sn
from src.graph.build_graph import load_graph
from src.graph.street_lookup import build_street_lookup, find_streets

ROOT = Path(__file__).resolve().parents[2]
CONCERNS_PATH = ROOT / "data" / "raw" / "concerns.csv"
FIELD_PATH = ROOT / "data" / "concerns" / "concern_field.csv"

THEMES = ("crossing_delay", "road_safety", "rat_running", "air_quality", "missing_cycle_link",
          "accessibility", "parking", "severance", "noise", "other")
SIGMA_M = 50.0
CUTOFF_M = 3 * SIGMA_M
LOCATED, PER_STREET = "located comments", "per-street counts"
FIELD_COLUMNS = ("edge_id", *THEMES, "basis")


def load_concerns(path: Path = CONCERNS_PATH) -> list[dict]:
    """Read the consultation records: theme, lat, lon, street, count and severity.

    Each record becomes theme, loc ((lat, lon), or None when only a street is known),
    street, count (how many comments the row stands for) and weight = severity * count,
    with severity defaulting to 1.
    """
    records = []
    with path.open(newline="", encoding="utf-8") as f:
        for line, row in enumerate(csv.DictReader(f), start=2):
            if row["theme"] not in THEMES:
                raise ValueError(f"{path}:{line}: unknown theme {row['theme']!r}; use one of {', '.join(THEMES)}")
            if bool(row["lat"]) != bool(row["lon"]):
                raise ValueError(f"{path}:{line}: give both lat and lon, or neither")
            if not (row["lat"] or row["street"]):
                raise ValueError(f"{path}:{line}: a record needs a location or a street")
            records.append({
                "theme": row["theme"],
                "loc": (float(row["lat"]), float(row["lon"])) if row["lat"] else None,
                "street": row["street"],
                "count": float(row["count"]),
                "weight": float(row["severity"] or 1.0) * float(row["count"]),
            })
    return records


def located_field(graph: sn.RoadGraph, records: list[dict]) -> dict[str, dict[str, float]]:
    """Return {edge_id: {theme: intensity}} from the Gaussian kernel around each located record.

    Only edges within CUTOFF_M of a record are touched.
    """
    index = sn.EdgeIndex(graph)
    exponent_scale = -0.5 / SIGMA_M ** 2
    field = defaultdict(lambda: defaultdict(float))
    for record in records:
        x, y = graph._project(*record["loc"])
        for hit in index.within(x, y, CUTOFF_M):
            field[str(hit.eid)][record["theme"]] += record["weight"] * math.exp(exponent_scale * hit.distance_m ** 2)
    return field


def street_field(graph: sn.RoadGraph, records: list[dict]) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Return {edge_id: {theme: weight}} giving each named street's full weight (severity * count)
    to all its edges, and the street names that didn't resolve.
    """
    found, unresolved = find_streets(build_street_lookup(graph), sorted({r["street"] for r in records if r["street"]}))
    field = defaultdict(lambda: defaultdict(float))
    for record in records:
        for eid in found.get(record["street"], ()):
            field[eid][record["theme"]] += record["weight"]
    return field, unresolved


def field_rows(graph: sn.RoadGraph, field: dict[str, dict[str, float]], basis: str) -> list[dict]:
    """Return one row per directed edge: edge_id, one column per theme, and basis."""
    no_concerns = {}
    rows = []
    for edge in graph.iter_edges():
        eid = str(edge.eid)
        values = field.get(eid, no_concerns)
        rows.append({"edge_id": eid, **{theme: values.get(theme, 0.0) for theme in THEMES}, "basis": basis})
    return rows


def main() -> None:
    """Build the field on one basis (located if every record has a location, else per-street) and write the CSV."""
    graph = load_graph()
    records = load_concerns(CONCERNS_PATH)
    if all(r["loc"] for r in records):
        field, basis, unresolved, unplaced = located_field(graph, records), LOCATED, [], 0
    else:
        field, unresolved = street_field(graph, records)
        basis, unplaced = PER_STREET, sum(not r["street"] for r in records)

    FIELD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIELD_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_COLUMNS)
        writer.writeheader()
        writer.writerows(field_rows(graph, field, basis))

    print(f"Concern field from {len(records)} records on {basis}: {len(field)} of {graph.num_edges} edges touched")
    if unplaced:
        print(f"  {unplaced} located records have no street name, so they can't be used per street")
    for name in unresolved:
        print(f"  street not found in the graph: {name!r}")


if __name__ == "__main__":
    main()
