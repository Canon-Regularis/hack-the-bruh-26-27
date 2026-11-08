"""Build #26's candidate interventions from the data, each with the reason it was picked.

Three kinds, COUNTS of each, all inside the study area:

- signalise_crossing: the unsignalised crossings with the longest modelled wait (#45);
- reallocate: the streets with the highest peak load in #24's baseline, as a bus lane;
- close: the streets named in consultation material (#25) that carry the most modelled cars,
  closed to through traffic. Main roads are left out: filtering one is not on the menu.

Four candidates (16 designs) keep #28's search to one round on three lab machines.

A street gets at most one candidate, and #27's closure street gets none, so it stays a
disruption rather than a design choice. Run from the repo root:
``python -m src.decide.propose_candidates``, then ``python -m src.decide.interventions``.
"""

import csv
import json
import re
from pathlib import Path

from engine.orchid import street_network as sn
from src.api.export import in_bbox
from src.decide.interventions import CANDIDATES_PATH, CLOSE, REALLOCATE, SIGNALISE
from src.decide.pedestrian_crossing_delay import crossing_delay
from src.decide.scenarios import CLOSURE_STREET
from src.graph.build_graph import load_graph

ROOT = Path(__file__).resolve().parents[2]
UE_PATH = ROOT / "data" / "demand" / "baseline_user_equilibrium.csv"
UE_META_PATH = ROOT / "data" / "demand" / "baseline_user_equilibrium.json"
CROSSINGS_PATH = ROOT / "data" / "demand" / "pedestrian_crossing_delay.csv"
CONCERNS_PATH = ROOT / "data" / "raw" / "concerns.csv"

BUS_LANE_FACTOR = 0.5
SIGNAL_FACTOR = 0.85
MAIN_ROADS = {"motorway", "trunk", "primary"}
COUNTS = {SIGNALISE: 1, REALLOCATE: 1, CLOSE: 2}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _inside(graph: sn.RoadGraph, nid: int) -> bool:
    node = graph.node(nid)
    return in_bbox(node.lon, node.lat)


def street_loads(graph: sn.RoadGraph, ue_rows: list[dict]) -> dict[str, dict]:
    """Per named study-area street: the busiest direction's flow and v/c, and its road classes."""
    loads: dict[str, dict] = {}
    for row in ue_rows:
        edge = graph.edge(int(row["edge_id"]))
        if not edge.street_name or not (_inside(graph, edge.u) and _inside(graph, edge.v)):
            continue
        load = loads.setdefault(edge.street_name, {"flow": 0.0, "vc": 0.0, "classes": set()})
        load["flow"] = max(load["flow"], float(row["flow_veh_h"]))
        load["vc"] = max(load["vc"], float(row["vc"]))
        load["classes"].add(edge.highway_class)
    return loads


def worst_crossings(graph: sn.RoadGraph, rows: list[dict], taken: set[str], n: int) -> list[dict]:
    pool = []
    for r in rows:
        if not r["control"].startswith("unsignalised") or not float(r["delay_s"] or 0) > 0:
            continue
        if not _inside(graph, int(r["node"])):
            continue
        names = sorted({graph.edge(int(e)).street_name for e in r["edge_ids"].split(";") if e} - {""})
        pool.append((-float(r["delay_s"]), -float(r["vehicle_flow_veh_h"] or 0), int(r["node"]), names, r))
    picks = []
    for _, _, node, names, r in sorted(pool, key=lambda p: p[:3]):
        street = names[0] if names else f"crossing {node}"
        if street in taken:
            continue
        taken.add(street)
        delay, flow = float(r["delay_s"]), float(r["vehicle_flow_veh_h"] or 0)
        signalised = crossing_delay({"signalised": True, "zebra": False, "width_m": float(r["width_m"]), "edge_ids": []}, {})
        picks.append({
            "id": f"signal-{_slug(street)}", "kind": SIGNALISE, "node": str(node), "street": street,
            "capacity_factor": SIGNAL_FACTOR,
            "reason": (f"One of the longest waits to cross in the study area: {delay:.0f} s (level of service {r['los']}) "
                       f"across {flow:,.0f} veh/h on {street}. Signals give people on foot their own phase."),
            "evidence": {"delay_s": round(delay, 1), "los": r["los"], "vehicle_flow_veh_h": round(flow, 1),
                         "signalised_delay_s": round(signalised.delay, 1), "signalised_los": signalised.level_of_service},
        })
        if len(picks) == n:
            break
    return picks


def overloaded_streets(loads: dict[str, dict], taken: set[str], n: int, hour: str) -> list[dict]:
    ranked = sorted((s for s, load in loads.items() if load["flow"] > 0 and s not in taken),
                    key=lambda s: (-loads[s]["vc"], -loads[s]["flow"], s))
    picks = []
    for street in ranked[:n]:
        taken.add(street)
        load = loads[street]
        picks.append({
            "id": f"bus-lane-{_slug(street)}", "kind": REALLOCATE, "street": street, "capacity_factor": BUS_LANE_FACTOR,
            "reason": (f"One of the most overloaded streets in the study area: its busiest direction carries "
                       f"{load['flow']:,.0f} veh/h at {load['vc']:.0%} of capacity ({hour}). "
                       f"A bus lane gives half its car capacity to buses."),
            "evidence": {"flow_veh_h": round(load["flow"], 1), "vc": round(load["vc"], 3)},
        })
    return picks


def complaint_streets(loads: dict[str, dict], concern_rows: list[dict], taken: set[str], n: int) -> list[dict]:
    named: dict[str, list[dict]] = {}
    for r in concern_rows:
        if r["street"] in loads:
            named.setdefault(r["street"], []).append(r)
    eligible = [s for s in named if s not in taken and loads[s]["flow"] > 0 and not loads[s]["classes"] & MAIN_ROADS]
    mentions = {s: sum(float(r["count"] or 1) for r in named[s]) for s in eligible}
    ranked = sorted(eligible, key=lambda s: (-loads[s]["flow"], -mentions[s], s))
    picks = []
    for street in ranked[:n]:
        taken.add(street)
        themes = sorted({r["theme"].replace("_", " ") for r in named[street]})
        sources = sorted({r["source"] for r in named[street] if r.get("source")})
        picks.append({
            "id": f"close-{_slug(street)}", "kind": CLOSE, "street": street,
            "reason": (f"Named in consultation material for {', '.join(themes)}"
                       f"{' (' + '; '.join(sources) + ')' if sources else ''}, and carries "
                       f"{loads[street]['flow']:,.0f} veh/h in the model. Closing it to through traffic answers the complaint."),
            "evidence": {"flow_veh_h": round(loads[street]["flow"], 1), "themes": themes, "sources": sources},
        })
    return picks


def propose(graph: sn.RoadGraph, ue_rows: list[dict], crossing_rows: list[dict], concern_rows: list[dict],
            hour: str, counts: dict[str, int] = COUNTS, closure_street: str = CLOSURE_STREET) -> list[dict]:
    """Return the candidates in the order crossings, overloaded streets, complaint streets."""
    taken = {closure_street}
    loads = street_loads(graph, ue_rows)
    return (worst_crossings(graph, crossing_rows, taken, counts.get(SIGNALISE, 0))
            + overloaded_streets(loads, taken, counts.get(REALLOCATE, 0), hour)
            + complaint_streets(loads, concern_rows, taken, counts.get(CLOSE, 0)))


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    hour = json.loads(UE_META_PATH.read_text(encoding="utf-8"))["hour"]
    candidates = propose(load_graph(), _rows(UE_PATH), _rows(CROSSINGS_PATH), _rows(CONCERNS_PATH), hour)
    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATES_PATH.write_text(json.dumps(candidates, indent=2) + "\n", encoding="utf-8")
    for c in candidates:
        print(f"{c['id']}: {c['reason']}")
    print(f"{len(candidates)} candidates -> {CANDIDATES_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
