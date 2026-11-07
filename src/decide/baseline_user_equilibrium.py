"""Run the baseline user equilibrium on the estimated OD matrix (ALG-S1-5, #24).

Writes one row per directed car edge for Signal's #5, in seconds and metres.
Run from the repo root: ``python -m src.decide.baseline_user_equilibrium``.
"""

import csv
import json
from pathlib import Path

from engine.orchid import traffic_assignment as ta
from src.decide.network import od_matrix, to_network
from src.graph.build_graph import load_graph

ROOT = Path(__file__).resolve().parents[2]
OD_PATH = ROOT / "data" / "demand" / "od_matrix.json"
UE_PATH = ROOT / "data" / "demand" / "baseline_user_equilibrium.csv"
UE_META_PATH = ROOT / "data" / "demand" / "baseline_user_equilibrium.json"

UE_FIELDS = ("edge_id", "length_m", "free_flow_time_s", "ue_time_s", "flow_veh_h", "vc")


def load_od(od_path: Path, node_ids: list[int]) -> tuple[ta.ODMatrix, str]:
    """Read #23's OD matrix and place each zone's trips on its centroid's network node.

    Returns the OD matrix in veh/h and the hour it represents.
    """
    estimate = json.loads(od_path.read_text(encoding="utf-8"))
    index = {nid: i for i, nid in enumerate(node_ids)}
    centroids = [index[nid] for nid in estimate["zone_centroid_nodes"]]
    return od_matrix(estimate["matrix"], centroids), estimate["hour"]


def run_ue(net: ta.Network, od: ta.ODMatrix) -> ta.AssignmentResult:
    """Run the Algorithm B equilibrium and raise if it didn't converge."""
    result = ta.user_equilibrium(net, od, method="b")
    if not result.converged:
        raise RuntimeError(f"Algorithm B did not converge: relative gap {result.relative_gap:.2e} "
                           f"after {result.iterations} iterations")
    return result


def ue_rows(net: ta.Network, result: ta.AssignmentResult) -> list[dict]:
    """Return one row per link: edge_id, length_m, free_flow_time_s, ue_time_s, flow_veh_h and vc."""
    return [{"edge_id": link.name, "length_m": link.length, "free_flow_time_s": link.vdf.free_flow_time(),
             "ue_time_s": time, "flow_veh_h": flow, "vc": flow / link.capacity}
            for link, time, flow in zip(net.links, result.travel_times, result.flows)]


def main() -> None:
    """Run the baseline equilibrium, write the edge CSV for Signal and the hour and gap beside it."""
    graph = load_graph()
    net, node_ids = to_network(graph)
    od, hour = load_od(OD_PATH, node_ids)
    result = run_ue(net, od)

    UE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with UE_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=UE_FIELDS)
        writer.writeheader()
        writer.writerows(ue_rows(net, result))
    UE_META_PATH.write_text(json.dumps({
        "hour": hour,
        "units": {"time": "s", "length": "m", "flow": "veh/h"},
        "method": "Algorithm B",
        "converged": result.converged,
        "relative_gap": result.relative_gap,
        "iterations": result.iterations,
    }, indent=1), encoding="utf-8")
    print(f"Baseline equilibrium for {hour}: {net.n_links} edges, relative gap {result.relative_gap:.1e} "
          f"after {result.iterations} iterations")


if __name__ == "__main__":
    main()
