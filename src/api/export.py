"""Write the files the demo site reads into frontend/public/data/, for the study area only.

Run from the repo root after the pipeline: ``python -m src.api.export``. A layer whose
inputs don't exist yet is left out of meta.json, and the site hides it. A layer whose inputs
can't be read is left out too, with a warning, so one bad file never stops the export.
"""

import csv
import json
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from engine.orchid import street_network as sn
from src.concerns.concern_field import CONCERNS_PATH, FIELD_PATH, THEMES
from src.graph.build_graph import load_graph
from src.graph.count_points import DFT_PATH, REVIEW_PATH, load_count_points
from src.graph.street_lookup import build_street_lookup, normalise_name
from src.rules.bind_rules import (CLAUSES_PATH, KEEP_OPEN, SOURCES_DIR, check_clauses, load_clauses,
                                  load_sources, osm_agrees)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "frontend" / "public" / "data"
UE_PATH = ROOT / "data" / "demand" / "baseline_user_equilibrium.csv"
UE_META_PATH = ROOT / "data" / "demand" / "baseline_user_equilibrium.json"
CROSSINGS_PATH = ROOT / "data" / "demand" / "pedestrian_crossing_delay.csv"
PREDICTIONS_DIR = ROOT / "data" / "derived" / "predictions"
SPEEDS_NIGHT_PATH = ROOT / "data" / "derived" / "speeds_night.csv"

STUDY_BBOX = (-3.215, 55.942, -3.155, 55.958)  # west, south, east, north: Signal's matching box
DECIMALS = 6
CITY_DECIMALS = 5
OSM_TAGS_SHOWN = ("access", "motor_vehicle", "vehicle", "oneway")
SPEED_COLUMNS = ("edge_id", "bin_start_utc", "speed_kmh", "kind")
SPEED_KINDS = {"observed": "1", "predicted": "0"}
LAYER_FILES = {"crossings": "crossings.geojson", "counts": "counts.geojson", "rules": "rules.json",
               "concerns": "concerns.json", "speeds": "speeds.json", "city": "city_speeds.json",
               "city_network": "city_network.geojson"}


def in_bbox(lon: float, lat: float, bbox=STUDY_BBOX) -> bool:
    west, south, east, north = bbox
    return west <= lon <= east and south <= lat <= north


def read_ue(path: Path = UE_PATH) -> dict[str, dict]:
    """Return {edge_id: {flow, vc, t, t0}} from #24's baseline CSV, or {} if it hasn't been run."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["edge_id"]: {"flow": float(row["flow_veh_h"]), "vc": float(row["vc"]),
                                 "t": float(row["ue_time_s"]), "t0": float(row["free_flow_time_s"])}
                for row in csv.DictReader(f)}


def _direction(prefix: str, run: dict | None) -> dict:
    if run is None:
        return {}
    return {f"{prefix}_flow": round(run["flow"], 1), f"{prefix}_vc": round(run["vc"], 3),
            f"{prefix}_t": round(run["t"], 2), f"{prefix}_t0": round(run["t0"], 2)}


def network_features(graph: sn.RoadGraph, ue: dict[str, dict]) -> list[dict]:
    """Return one LineString per street segment with both ends in the study area.

    Properties carry both directions: f_* for edge id, r_* for its reverse twin, plus
    flow (both directions summed) and vc (the busier direction) for styling.
    """
    sub = graph.subgraph(n.node_id for n in graph.iter_nodes() if in_bbox(n.lon, n.lat))
    features = []
    for feature in sn.to_geojson(sub, include_nodes=False)["features"]:
        edge = graph.edge(feature["properties"]["edgeId"])
        rev = str(edge.reverse_eid) if edge.reverse_eid is not None else None
        fwd, back = ue.get(str(edge.eid)), ue.get(rev) if rev else None
        runs = [r for r in (fwd, back) if r]
        props = {"id": str(edge.eid), "rev": rev, "name": edge.street_name, "cls": edge.highway_class,
                 "car": edge.permits("car"), "oneway": edge.oneway, "len": round(edge.length_m, 1),
                 "lanes": edge.lanes, "limit": edge.speed_limit_kph,
                 **_direction("f", fwd), **_direction("r", back)}
        if runs:
            props.update(flow=round(sum(r["flow"] for r in runs), 1), vc=round(max(r["vc"] for r in runs), 3))
        coords = [[round(x, DECIMALS), round(y, DECIMALS)] for x, y in feature["geometry"]["coordinates"]]
        features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                         "properties": props})
    return features


def streets(features: list[dict]) -> dict[str, dict]:
    """Return {street name: {bbox, segments}} for the search box."""
    found = defaultdict(lambda: {"bbox": [180.0, 90.0, -180.0, -90.0], "segments": 0})
    for feature in features:
        name = feature["properties"]["name"]
        if not name:
            continue
        street = found[name]
        street["segments"] += 1
        for lon, lat in feature["geometry"]["coordinates"]:
            box = street["bbox"]
            box[:] = [min(box[0], lon), min(box[1], lat), max(box[2], lon), max(box[3], lat)]
    return dict(sorted(found.items()))


def crossing_features(graph: sn.RoadGraph, path: Path = CROSSINGS_PATH) -> list[dict]:
    """Return one Point per #45 crossing in the study area, or [] if it hasn't been run."""
    if not path.exists():
        return []
    features = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            node = graph.node(int(row["node"]))
            if not in_bbox(node.lon, node.lat):
                continue
            flow = row["vehicle_flow_veh_h"]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(node.lon, DECIMALS), round(node.lat, DECIMALS)]},
                "properties": {"node": row["node"], "edges": row["edge_ids"].split(";"), "control": row["control"],
                               "width": round(float(row["width_m"]), 1), "delay": round(float(row["delay_s"]), 1),
                               "los": row["los"], "flow": round(float(flow), 1) if flow else None},
            })
    return features


def count_features(dft_path: Path = DFT_PATH, review_path: Path = REVIEW_PATH) -> list[dict]:
    """Return one Point per DfT count point in the study area that #21 matched to an edge.

    counted is False when DfT estimated that year's AADF instead of counting it.
    """
    if not (dft_path.exists() and review_path.exists()):
        return []
    with review_path.open(newline="", encoding="utf-8") as f:
        matched = {row["count_point_id"]: row for row in csv.DictReader(f) if row["status"] == "matched"}
    with dft_path.open(newline="", encoding="utf-8") as f:
        method = {(row["count_point_id"], int(row["year"])): row["estimation_method"] for row in csv.DictReader(f)}
    return [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
             "properties": {"id": p["count_point_id"], "road": p["road_name"], "aadf": p["aadf"], "year": p["year"],
                            "counted": method[(p["count_point_id"], p["year"])] == "Counted",
                            "edge": matched[p["count_point_id"]]["edge_id"]}}
            for p in load_count_points(dft_path) if p["count_point_id"] in matched and in_bbox(p["lon"], p["lat"])]


def traffic_layer(ue_path: Path = UE_PATH, meta_path: Path = UE_META_PATH) -> tuple[dict[str, dict], str] | None:
    """Return #24's per-edge results and the hour they model, or None if it hasn't been run."""
    ue = read_ue(ue_path)
    if not ue:
        return None
    return ue, json.loads(meta_path.read_text(encoding="utf-8"))["hour"]


def osm_says(edge: sn.RoadEdge, restriction: str = "") -> str:
    """Return what OSM says about an edge's access, e.g. "highway=pedestrian, motor_vehicle=private".

    A one-way clause binds only edges against its direction, so for one_way say which way OSM runs.
    """
    tags = edge.tags["osm"]
    said = ", ".join([f"highway={edge.highway_class}", *(f"{k}={tags[k]}" for k in OSM_TAGS_SHOWN if k in tags)])
    if restriction == "one_way":
        return f"{'two-way' if edge.reverse_eid is not None else 'one-way the opposite way'} ({said})"
    return said


def _metres(edges: list[sn.RoadEdge]) -> int:
    """Return the street length the edges cover, counting both directions of a two-way street once."""
    seen = {min(e.eid, e.reverse_eid) if e.reverse_eid is not None else e.eid: e.length_m for e in edges}
    return round(sum(seen.values()))


def rules_layer(graph: sn.RoadGraph, clauses_path: Path = CLAUSES_PATH, sources_dir: Path = SOURCES_DIR
                ) -> tuple[list[dict], dict[str, tuple[str, bool]], set[str]] | None:
    """Return one row per #20 clause for the site, {edge id: (restriction, OSM agrees)} for the edges
    applied clauses bind, and the ids of every edge on a street only abstained clauses name.

    streets are the graph's names for the street a clause names. status is applied, keep_open or
    abstained (with its reason). An applied clause lists how many of its edges OSM disagrees with
    and what OSM says on them, so a tagging difference reads differently from a missing restriction.
    None without clauses.
    """
    if not clauses_path.exists():
        return None
    clauses = load_clauses(clauses_path)
    report = check_clauses(clauses, load_sources(clauses, sources_dir), graph)
    abstained = dict(report.abstained)
    lookup = build_street_lookup(graph)
    rows, edge_rule, held = [], {}, set()
    for c in clauses:
        cid = c["clause_id"]
        edges = [graph.edge(int(e)) for e in sorted(report.clause_edges.get(cid, ()), key=int)]
        wrong = {e.eid for e in edges if not osm_agrees(c["restriction"], e)} if cid in report.applied else set()
        status = "abstained" if cid in abstained else "keep_open" if c["restriction"] == KEEP_OPEN else "applied"
        named = lookup.get(normalise_name(c["street"]), [])
        if status == "abstained":
            held.update(named)
        names = sorted({graph.edge(int(e)).street_name for e in named})
        rows.append({"id": cid, "streets": names or [c["street"]], "restriction": c["restriction"],
                     "quote": c["quote"], "document": c["document_id"], "page": c["page"], "times": c["times"],
                     "in_force": c.get("in_force"),
                     "exemptions": c["exemptions"], "status": status, "reason": abstained.get(cid),
                     "edges": len(edges), "osm_wrong_edges": len(wrong), "length_m": _metres(edges),
                     "osm_wrong_m": _metres([e for e in edges if e.eid in wrong]),
                     "osm_says": sorted({osm_says(e, c["restriction"]) for e in edges if e.eid in wrong})})
        for e in edges:
            restriction, agrees = edge_rule.get(str(e.eid), (c["restriction"], True))
            edge_rule[str(e.eid)] = (restriction, agrees and e.eid not in wrong)
    return rows, edge_rule, held


def add_rules(features: list[dict], edge_rule: dict[str, tuple[str, bool]], held: set[str] = frozenset()) -> None:
    """Mark each segment a clause binds in either direction with rule and osm_ok (OSM agrees both ways),
    and each other segment on a street only abstained clauses name with held."""
    for feature in features:
        props = feature["properties"]
        hits = [edge_rule[e] for e in (props["id"], props["rev"]) if e in edge_rule]
        if hits:
            props.update(rule=hits[0][0], osm_ok=all(agrees for _, agrees in hits))
        elif props["id"] in held or props["rev"] in held:
            props["held"] = True


def concern_layer(graph: sn.RoadGraph, field_path: Path = FIELD_PATH, records_path: Path = CONCERNS_PATH
                  ) -> tuple[dict[str, dict[str, float]], str, dict[str, list[dict]]] | None:
    """Return #25's non-zero intensities per edge, the field's basis, and the source records per street.

    Records are keyed by the graph's street name, for the street card. None if the field hasn't been built.
    """
    if not field_path.exists():
        return None
    field, basis = {}, ""
    with field_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            basis = row["basis"]
            values = {theme: float(row[theme]) for theme in THEMES if float(row[theme]) > 0}
            if values:
                field[row["edge_id"]] = values
    records = defaultdict(list)
    if records_path.exists():
        lookup = build_street_lookup(graph)
        with records_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                eids = lookup.get(normalise_name(row["street"]), ()) if row.get("street") else ()
                for name in sorted({graph.edge(int(e)).street_name for e in eids}):
                    records[name].append({"theme": row["theme"], "count": float(row["count"]),
                                          "source": row.get("source") or ""})
    return field, basis, dict(records)


def add_concerns(features: list[dict], field: dict[str, dict[str, float]]) -> None:
    """Give each touched segment concern (its busier direction's total) and concern_theme (that direction's top theme)."""
    for feature in features:
        props = feature["properties"]
        values = [field[e] for e in (props["id"], props["rev"]) if e in field]
        if values:
            busiest = max(values, key=lambda v: sum(v.values()))
            props.update(concern=round(sum(busiest.values()), 3), concern_theme=max(busiest, key=busiest.get))


def newest(folder: Path, pattern: str) -> Path | None:
    """Return the file in folder matching pattern that sorts last (Signal stamps names with UTC time)."""
    files = sorted(folder.glob(pattern)) if folder.exists() else []
    return files[-1] if files else None


def speeds_layer(folder: Path = PREDICTIONS_DIR) -> tuple[dict, dict] | None:
    """Return Signal's newest prediction table for the site and its meta, or None before it exists.

    The table is {bins, edges: {edge_id: {s, o}}}: s holds km/h per 15-minute bin (None where the
    table has no row) and o one character per bin, "1" where a bus was seen and "0" where the model
    predicts. The meta counts the predicted share from the table itself, and adds the errors and
    the prediction interval from Signal's newest summary file when there is one.
    """
    path = newest(folder, "speed_predictions_*.csv")
    if path is None:
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path.name} has no rows")
    if missing := set(SPEED_COLUMNS) - rows[0].keys():
        raise ValueError(f"{path.name} is missing columns {sorted(missing)}")
    if kinds := {r["kind"] for r in rows} - SPEED_KINDS.keys():
        raise ValueError(f"{path.name} has unknown kinds {sorted(kinds)}")
    stamp = {raw: _utc(raw) for raw in {r["bin_start_utc"] for r in rows}}
    bins = sorted(set(stamp.values()))
    position = {b: i for i, b in enumerate(bins)}
    edges = {}
    for r in rows:
        edge = edges.setdefault(r["edge_id"], {"s": [None] * len(bins), "o": ["0"] * len(bins)})
        i = position[stamp[r["bin_start_utc"]]]
        edge["s"][i] = round(float(r["speed_kmh"]), 1)
        edge["o"][i] = SPEED_KINDS[r["kind"]]
    for edge in edges.values():
        edge["o"] = "".join(edge["o"])
    observed = sum(r["kind"] == "observed" for r in rows)
    meta = {"share_predicted": round(1 - observed / len(rows), 3), "window": [bins[0], bins[-1]],
            "edges": len(edges), "file": path.name}
    if summary_path := newest(folder, "summary_*.json"):
        meta.update(summary_meta(json.loads(summary_path.read_text(encoding="utf-8"))))
    return {"bins": bins, "edges": edges}, meta


def _utc(raw: str) -> str:
    return datetime.fromisoformat(raw).astimezone(timezone.utc).isoformat()


def city_speed_layer(graph: sn.RoadGraph, bins: list[str], speeds_path: Path = SPEEDS_NIGHT_PATH
                     ) -> tuple[list[dict], dict] | None:
    """Return bus-observed speeds outside the study area: one LineString per segment a bus moved on,
    and {bins, edges} in speeds.json's format, every value observed (o is "1" wherever s has a speed).

    Reads Signal's joined night (run_signal.py's speeds_night.csv). bins are the prediction table's, so
    one slider drives both; rows in other bins are left out. Segments inside the study area are left out
    because the prediction table covers them. None before the night has been joined.
    """
    if not speeds_path.exists():
        return None
    position = {b: i for i, b in enumerate(bins)}
    edges = {}
    with speeds_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            speed, i = row["mean_bus_observed_speed_kmh"], position.get(_utc(row["bin_start_utc"]))
            if not speed or i is None or float(row["n_traversals"] or 0) <= 0:
                continue
            edge = graph.edge(int(row["edge_id"]))
            if all(in_bbox(graph.node(n).lon, graph.node(n).lat) for n in (edge.u, edge.v)):
                continue
            entry = edges.setdefault(row["edge_id"], {"s": [None] * len(bins), "o": ["0"] * len(bins)})
            entry["s"][i], entry["o"][i] = round(float(speed), 1), "1"
    seen = {int(e) for e in edges}
    sub = graph.subgraph({n for eid in seen for n in (graph.edge(eid).u, graph.edge(eid).v)})
    features = []
    for feature in sn.to_geojson(sub, include_nodes=False)["features"]:
        edge = graph.edge(feature["properties"]["edgeId"])
        if edge.eid not in seen and edge.reverse_eid not in seen:
            continue
        coords = [[round(x, CITY_DECIMALS), round(y, CITY_DECIMALS)] for x, y in feature["geometry"]["coordinates"]]
        features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                         "properties": {"id": str(edge.eid), "name": edge.street_name,
                                        "rev": str(edge.reverse_eid) if edge.reverse_eid is not None else None}})
    return features, {"bins": bins, "edges": {k: {"s": v["s"], "o": "".join(v["o"])} for k, v in edges.items()}}


def summary_meta(summary: dict) -> dict:
    """Pick the site's headline numbers out of Signal's summary file.

    Errors are compared with the better of Signal's two baselines, so "better than the best
    simple guess" holds. Either split, or the interval, is left out if the summary lacks it.
    """
    meta = {}
    for prefix, error, baselines in (("", "error_unseen_streets_kmh", "baselines_unseen_streets_kmh"),
                                     ("time_", "error_later_bins_kmh", "baselines_later_bins_kmh")):
        if error in summary and summary.get(baselines):
            best = min(summary[baselines].values())
            meta.update({f"{prefix}mae_kmh": round(summary[error], 2), f"{prefix}baseline_mae_kmh": round(best, 2),
                         f"{prefix}skill": round(1 - summary[error] / best, 3)})
    interval = summary.get("interval", {})
    if {"half_width_kmh", "target_coverage", "measured_coverage_unseen_streets"} <= interval.keys():
        meta.update(interval_kmh=round(interval["half_width_kmh"], 1), interval_target=interval["target_coverage"],
                    interval_coverage=round(interval["measured_coverage_unseen_streets"], 3))
    return meta


def _optional(name: str, build: Callable):
    """Build one optional layer; if its input can't be read, leave the layer out with a warning."""
    try:
        return build()
    except Exception as e:  # one bad input file costs only its own layer
        print(f"warning: left out the {name} layer: {type(e).__name__}: {e}")
        return None


def _write(name: str, data) -> None:
    (OUT_DIR / name).write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


def _collection(features: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    """Write network, streets, every layer that has data, and meta.json listing them.

    The graph is required; every other input is optional. Files of layers left out are deleted,
    so the site never reads a stale one.
    """
    graph = load_graph()
    traffic = _optional("traffic", lambda: traffic_layer(UE_PATH, UE_META_PATH))
    network = network_features(graph, traffic[0] if traffic else {})
    crossings = _optional("crossings", lambda: crossing_features(graph, CROSSINGS_PATH)) or []
    counts = _optional("counts", lambda: count_features(DFT_PATH, REVIEW_PATH)) or []
    rules = _optional("rules", lambda: rules_layer(graph, CLAUSES_PATH, SOURCES_DIR))
    concerns = _optional("concerns", lambda: concern_layer(graph, FIELD_PATH, CONCERNS_PATH))
    speeds = _optional("speeds", lambda: speeds_layer(PREDICTIONS_DIR))
    city = (_optional("city speeds", lambda: city_speed_layer(graph, speeds[0]["bins"], SPEEDS_NIGHT_PATH))
            if speeds else None)
    if rules:
        add_rules(network, rules[1], rules[2])
    if concerns:
        add_concerns(network, concerns[0])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write("network.geojson", _collection(network))
    _write("streets.json", streets(network))
    layers = {}
    if traffic:
        layers["traffic"] = {"label": "Modelled car traffic", "basis": "estimated", "hour": traffic[1],
                             "note": "Modelled from DfT counts"}
    if crossings:
        _write("crossings.geojson", _collection(crossings))
        layers["crossings"] = {"label": "Pedestrian crossing delay", "basis": "modelled",
                               "note": "Widths, signal timings and the zebra wait are assumed"}
    if counts:
        _write("counts.geojson", _collection(counts))
        counted = sum(c["properties"]["counted"] for c in counts)
        layers["counts"] = {"label": "DfT traffic counts", "basis": "observed" if counted == len(counts) else "estimated",
                            "note": f"Annual average daily flow, latest year. {counted} of {len(counts)} counted "
                                    "that year; DfT estimated the rest"}
    if rules:
        rows = rules[0]
        applied = [r for r in rows if r["status"] == "applied"]
        _write("rules.json", {"clauses": rows})
        layers["rules"] = {"label": "Legal restrictions", "basis": "verified",
                           "note": "Every quote checked word for word against the cached order. Abstained clauses "
                                   "can't be encoded honestly, so they aren't applied",
                           "applied": len(applied), "osm_wrong": sum(r["osm_wrong_edges"] > 0 for r in applied),
                           "abstained": sum(r["status"] == "abstained" for r in rows)}
    if concerns:
        _write("concerns.json", concerns[2])
        layers["concerns"] = {"label": "Consultation concerns", "basis": "observed",
                              "note": f"From published consultation reports, as {concerns[1] or 'per-street counts'}"}
    if speeds:
        _write("speeds.json", speeds[0])
        layers["speeds"] = {"label": "Bus-observed speed", "basis": "predicted",
                            "note": "Speed of buses, not of all traffic. Solid where a bus was seen in that "
                                    "quarter-hour; dashed where the model predicts it", **speeds[1]}
    kept = {LAYER_FILES[layer] for layer in layers if layer in LAYER_FILES}
    if speeds and city and city[1]["edges"]:
        _write("city_network.geojson", _collection(city[0]))
        _write("city_speeds.json", city[1])
        layers["speeds"]["city_edges"] = len(city[1]["edges"])
        kept |= {LAYER_FILES["city"], LAYER_FILES["city_network"]}
    for name in set(LAYER_FILES.values()) - kept:
        (OUT_DIR / name).unlink(missing_ok=True)
    _write("meta.json", {"bbox": STUDY_BBOX, "layers": layers})
    print(f"{len(network)} segments, {len(crossings)} crossings, {len(counts)} count points -> {OUT_DIR}")
    print(f"layers: {', '.join(layers) or 'none'}")


if __name__ == "__main__":
    main()
