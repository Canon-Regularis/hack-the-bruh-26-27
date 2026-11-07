"""Attach DfT traffic count points to their nearest road edge (ALG-S1-2, #21).

Run from the repo root: ``python -m src.graph.count_points``.
"""

import csv
from pathlib import Path

from engine.orchid import street_network as sn
from src.graph.build_graph import load_graph

ROOT = Path(__file__).resolve().parents[2]
DFT_PATH = ROOT / "data" / "raw" / "dft_aadf.csv"
AADF_PATH = ROOT / "data" / "graph" / "dft_aadf.csv"
REVIEW_PATH = ROOT / "data" / "graph" / "dft_review.csv"

MAX_SNAP_M = 25.0
AADF_FIELDS = ("edge_id", "count_point_id", "aadf")
REVIEW_FIELDS = ("count_point_id", "year", "road_name", "aadf", "edge_id", "street_name", "distance_m", "status")


def load_count_points(path: Path = DFT_PATH) -> list[dict]:
    """Read the DfT AADF file and keep each count point's latest year.

    Returns count_point_id, year, road_name, lat, lon and aadf (all motor vehicles, two-way total).
    """
    latest = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year = int(row["year"])
            kept = latest.get(row["count_point_id"])
            if kept is None or year > kept[0]:
                latest[row["count_point_id"]] = (year, row)
    return [{"count_point_id": cid, "year": year, "road_name": row["road_name"],
             "lat": float(row["latitude"]), "lon": float(row["longitude"]),
             "aadf": int(row["all_motor_vehicles"])}
            for cid, (year, row) in sorted(latest.items(), key=lambda item: int(item[0]))]


def snap_count_points(points: list[dict], graph: sn.RoadGraph,
                      index: sn.EdgeIndex) -> tuple[list[dict], list[dict]]:
    """Snap each point to the nearest edge in index (built for cars) within MAX_SNAP_M.

    Points outside the graph's area are ignored. Returns the matches (one per road, nearest
    point wins) and the points in the area that were dropped, each with a status saying why.
    """
    min_x, min_y, max_x, max_y = graph.bbox()
    best, dropped = {}, []
    for point in points:
        x, y = graph._project(point["lat"], point["lon"])
        if not (min_x - MAX_SNAP_M <= x <= max_x + MAX_SNAP_M and min_y - MAX_SNAP_M <= y <= max_y + MAX_SNAP_M):
            continue
        hit = index.nearest(x, y, max_dist_m=MAX_SNAP_M)
        if hit is None:
            dropped.append({**point, "status": f"no car road within {MAX_SNAP_M:g} m"})
            continue
        edge = graph.edge(hit.eid)
        match = {**point, "edge": edge, "distance_m": hit.distance_m, "status": "matched"}
        road = min(edge.eid, edge.reverse_eid) if edge.reverse_eid is not None else edge.eid
        kept = best.get(road)
        if kept is None or hit.distance_m < kept["distance_m"]:
            best[road], loser = match, kept
        else:
            loser = match
        if loser is not None:
            dropped.append({**loser, "status": f"road taken by nearer point {best[road]['count_point_id']}"})
    return list(best.values()), dropped


def aadf_rows(matches: list[dict]) -> list[dict]:
    """Return edge_id, count_point_id and aadf rows for the matched edge and its reverse twin."""
    rows = []
    for match in matches:
        edge = match["edge"]
        eids = (edge.eid,) if edge.reverse_eid is None else (edge.eid, edge.reverse_eid)
        rows.extend({"edge_id": str(eid), "count_point_id": match["count_point_id"], "aadf": match["aadf"]}
                    for eid in eids)
    return sorted(rows, key=lambda r: int(r["edge_id"]))


def review_rows(matches: list[dict], dropped: list[dict]) -> list[dict]:
    """Return one row per count point in the area for checking the matches by eye."""
    rows = []
    for point in sorted(matches + dropped, key=lambda p: int(p["count_point_id"])):
        row = {k: point[k] for k in ("count_point_id", "year", "road_name", "aadf", "status")}
        if "edge" in point:
            row.update(edge_id=str(point["edge"].eid), street_name=point["edge"].street_name,
                       distance_m=f"{point['distance_m']:.1f}")
        rows.append(row)
    return rows


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Snap the count points, write the AADF hand-off CSV and the by-eye review CSV."""
    graph = load_graph()
    matches, dropped = snap_count_points(load_count_points(DFT_PATH), graph, sn.EdgeIndex(graph, mode="car"))
    _write_csv(AADF_PATH, AADF_FIELDS, aadf_rows(matches))
    _write_csv(REVIEW_PATH, REVIEW_FIELDS, review_rows(matches, dropped))
    print(f"{len(matches)} count points matched, {len(dropped)} dropped; check {REVIEW_PATH} by eye")


if __name__ == "__main__":
    main()
