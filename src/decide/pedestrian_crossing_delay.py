"""Pedestrian crossing delay, level of service and the baseline J_ped (ALG-S1-7, #45).

Signalised crossings wait for the green. Zebras give pedestrians priority, so they wait only
for traffic to yield. Other unsignalised crossings use Adams' gap-acceptance delay, capped at
MAX_WAIT_S, past which pedestrians force a gap (HCM rates waits over 45 s as LOS F); uncapped,
Adams' delay grows exponentially with traffic. Crossing widths, signal timings, the zebra
delay, the cap and pedestrian densities are assumed, not measured.
Run from the repo root: ``python -m src.decide.pedestrian_crossing_delay``.
"""

import csv
import json
import math
from pathlib import Path

from engine.orchid import pedestrian_flow as pf
from engine.orchid import street_network as sn
from src.graph.build_graph import load_graph

ROOT = Path(__file__).resolve().parents[2]
UE_PATH = ROOT / "data" / "demand" / "baseline_user_equilibrium.csv"
CROSSINGS_PATH = ROOT / "data" / "demand" / "pedestrian_crossing_delay.csv"
J_PED_PATH = ROOT / "data" / "demand" / "pedestrian_crossing_delay.json"

LANE_WIDTH_M = 3.25
WALKING_SPEED_MPS = 1.2
CYCLE_S = 90.0
WALK_GREEN_S = 8.0
ZEBRA_DELAY_S = 3.0
MAX_WAIT_S = 60.0
ZEBRA_REFS = frozenset({"zebra", "tiger"})
SYNTHETIC_DENSITIES = {"quiet": 0.5, "busy": 1.5, "festival_peak": 3.0}
CROSSING_FIELDS = ("node", "edge_ids", "control", "width_m", "vehicle_flow_veh_h", "delay_s", "los")


def find_crossings(graph: sn.RoadGraph) -> list[dict]:
    """Return every highway=crossing node on a car road: node id, the str ids of the car edges entering it,
    whether it's signalised (crossing=traffic_signals) or a zebra (crossing_ref zebra or tiger,
    zebra markings, or the old crossing=zebra), and its width (lanes crossed x LANE_WIDTH_M).

    On a two-way road the entering edges are one per direction, so their lanes add up to the full width.
    """
    crossings = []
    for node in graph.iter_nodes():
        osm = node.tags.get("osm", {})
        if osm.get("highway") != "crossing":
            continue
        car_edges = [edge for edge in map(graph.edge, graph.in_edges(node.node_id)) if edge.permits("car")]
        if car_edges:
            signalised = osm.get("crossing") == "traffic_signals"
            crossings.append({
                "node": node.node_id,
                "edge_ids": [str(edge.eid) for edge in car_edges],
                "signalised": signalised,
                "zebra": not signalised and (osm.get("crossing_ref") in ZEBRA_REFS or osm.get("crossing") == "zebra"
                                             or osm.get("crossing:markings", "").startswith("zebra")),
                "width_m": LANE_WIDTH_M * sum(edge.lanes for edge in car_edges),
            })
    return crossings


def load_flows(ue_path: Path = UE_PATH) -> dict[str, float]:
    """Read #24's baseline CSV as {edge_id: flow in veh/h}."""
    with ue_path.open(newline="", encoding="utf-8") as f:
        return {row["edge_id"]: float(row["flow_veh_h"]) for row in csv.DictReader(f)}


def _fixed_delay(control: str, width_m: float, delay_s: float, detail: dict) -> pf.CrossingDelayResult:
    crossing_time = width_m / WALKING_SPEED_MPS
    return pf.CrossingDelayResult(control, delay_s, crossing_time, delay_s + crossing_time, None, 1,
                                  pf.pedestrian_delay_los(delay_s).grade, detail)


def _unsignalised_delay(width_m: float, vehicle_flow: float) -> pf.CrossingDelayResult:
    """Adams' gap-acceptance delay, or MAX_WAIT_S when that is longer."""
    try:
        result = pf.crossing_delay(width_m, control="unsignalised", vehicle_flow=vehicle_flow,
                                   walking_speed=WALKING_SPEED_MPS)
        if result.delay <= MAX_WAIT_S:
            return result
    except pf.PedestrianFlowError:
        pass  # orchid: the crossing never clears, so far past the cap
    return _fixed_delay("unsignalised-capped", width_m, MAX_WAIT_S, {"combinedFlow": vehicle_flow})


def crossing_delay(crossing: dict, flows: dict[str, float]) -> pf.CrossingDelayResult:
    """Return one crossing's delay: the signal wait, the zebra wait, or Adams' delay capped at
    MAX_WAIT_S on the summed flow of the edges it crosses.

    An edge missing from flows carries no traffic, e.g. one a design has closed.
    """
    if crossing["signalised"]:
        return pf.crossing_delay(crossing["width_m"], control="signalised", cycle=CYCLE_S, walk_green=WALK_GREEN_S,
                                 walking_speed=WALKING_SPEED_MPS)
    if crossing["zebra"]:
        return _fixed_delay("zebra", crossing["width_m"], ZEBRA_DELAY_S, {})
    return _unsignalised_delay(crossing["width_m"], math.fsum(flows.get(eid, 0.0) for eid in crossing["edge_ids"]))


def crossing_delays(crossings: list[dict], flows: dict[str, float]) -> list[pf.CrossingDelayResult]:
    """Return each crossing's delay, as crossing_delay does."""
    return [crossing_delay(c, flows) for c in crossings]


def j_ped(crossings: list[dict], flows: dict[str, float]) -> float:
    """Return J_ped, the summed crossing delay in seconds, for a set of flows keyed by edge id.

    #28 calls this per design and scenario.
    """
    return math.fsum(result.delay for result in crossing_delays(crossings, flows))


def fruin_levels() -> dict[str, str]:
    """Return the Fruin level of service for people queueing at a kerb, per synthetic density case."""
    return {case: pf.fruin_los_from_density(rho, "queueing").grade for case, rho in SYNTHETIC_DENSITIES.items()}


def main() -> None:
    """Write each crossing's delay and LOS, and the baseline J_ped with its assumptions."""
    crossings = find_crossings(load_graph())
    results = crossing_delays(crossings, load_flows(UE_PATH))
    baseline = math.fsum(result.delay for result in results)

    CROSSINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CROSSINGS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CROSSING_FIELDS)
        writer.writeheader()
        writer.writerows({"node": c["node"], "edge_ids": ";".join(c["edge_ids"]), "control": r.control,
                          "width_m": c["width_m"], "vehicle_flow_veh_h": r.detail.get("combinedFlow", ""),
                          "delay_s": r.delay, "los": r.level_of_service}
                         for c, r in zip(crossings, results))
    J_PED_PATH.write_text(json.dumps({
        "j_ped_s": baseline,
        "crossings": len(crossings),
        "signalised": sum(c["signalised"] for c in crossings),
        "flows_from": UE_PATH.name,
        "assumed": {"lane_width_m": LANE_WIDTH_M, "walking_speed_mps": WALKING_SPEED_MPS, "cycle_s": CYCLE_S,
                    "walk_green_s": WALK_GREEN_S, "zebra_delay_s": ZEBRA_DELAY_S, "max_wait_s": MAX_WAIT_S},
        "synthetic_fruin_los": {"facility": "queueing", "densities_p_per_m2": SYNTHETIC_DENSITIES,
                                "grades": fruin_levels()},
    }, indent=1), encoding="utf-8")
    print(f"Baseline J_ped = {baseline:.0f} s over {len(crossings)} crossings")


if __name__ == "__main__":
    main()
