"""Estimate the AM-peak OD matrix from a gravity prior and the DfT counts (ALG-S1-4, #23).

Trips are veh/h to match the BPR capacities from #22. Run from the repo root:
``python -m src.decide.od_matrix_estimation``.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from engine.orchid import traffic_assignment as ta
from engine.orchid import travel_demand as td
from src.decide.network import link_maps, od_matrix, to_network
from src.graph.build_graph import load_graph

ROOT = Path(__file__).resolve().parents[2]
AADF_PATH = ROOT / "data" / "graph" / "dft_aadf.csv"
OD_PATH = ROOT / "data" / "demand" / "od_matrix.json"
FIT_PATH = ROOT / "data" / "demand" / "count_fit.csv"

N_ZONES = 20
MODELLED_HOUR = "08:00-09:00 weekday"
PEAK_SHARE = 0.09
NO_INTRAZONAL_COST_S = 1e12


def _largest_strong_component(net: ta.Network) -> list[int]:
    """Nodes of the largest strongly connected component (Kosaraju), so every pair can reach each other."""
    forward = [[] for _ in range(net.n_nodes)]
    backward = [[] for _ in range(net.n_nodes)]
    for link in net.links:
        forward[link.tail].append(link.head)
        backward[link.head].append(link.tail)

    finished, seen = [], [False] * net.n_nodes
    for start in range(net.n_nodes):
        if seen[start]:
            continue
        seen[start] = True
        stack = [(start, iter(forward[start]))]
        while stack:
            node, successors = stack[-1]
            nxt = next((m for m in successors if not seen[m]), None)
            if nxt is None:
                stack.pop()
                finished.append(node)
            else:
                seen[nxt] = True
                stack.append((nxt, iter(forward[nxt])))

    assigned, largest = [False] * net.n_nodes, []
    for start in reversed(finished):
        if assigned[start]:
            continue
        assigned[start] = True
        component, stack = [start], [start]
        while stack:
            for m in backward[stack.pop()]:
                if not assigned[m]:
                    assigned[m] = True
                    component.append(m)
                    stack.append(m)
        if len(component) > len(largest):
            largest = component
    return sorted(largest)


def make_zones(net: ta.Network, node_xy: list[tuple[float, float]], n_zones: int = N_ZONES) -> list[int]:
    """Cluster the network's mutually reachable nodes into n_zones areas by location.

    Returns each zone's centroid: the member node nearest its cluster's centre, so all are distinct.
    """
    nodes = _largest_strong_component(net)
    xy = np.array([node_xy[n] for n in nodes])
    kmeans = KMeans(n_clusters=n_zones, n_init=10, random_state=0).fit(xy)
    sq_dist = ((xy - kmeans.cluster_centers_[kmeans.labels_]) ** 2).sum(axis=1)
    centroids = []
    for zone in range(n_zones):
        members = np.flatnonzero(kmeans.labels_ == zone)
        centroids.append(nodes[members[sq_dist[members].argmin()]])
    return centroids


def free_flow_costs(net: ta.Network, centroids: list[int], free_flow: list[float]) -> list[list[float]]:
    """Return zone-to-zone free-flow travel times in seconds, from shortest paths on the network."""
    costs = []
    for origin in centroids:
        dist = ta.shortest_path_tree(net, origin, free_flow).dist
        costs.append([dist[d] for d in centroids])
    return costs


def gravity_prior(costs: list[list[float]], total_trips: float) -> list[list[float]]:
    """Return a doubly-constrained gravity matrix with equal trips per zone summing to total_trips.

    Uses ExponentialDeterrence with beta = 1 / mean zone-to-zone cost, in inverse seconds.
    Trips within a zone are zero, because a centroid can't route to itself.
    """
    n = len(costs)
    beta = n * (n - 1) / math.fsum(c for i, row in enumerate(costs) for j, c in enumerate(row) if i != j)
    masked = [[NO_INTRAZONAL_COST_S if i == j else c for j, c in enumerate(row)] for i, row in enumerate(costs)]
    ends = [total_trips / n] * n
    return td.gravity_model(ends, ends, masked, td.ExponentialDeterrence(beta)).matrix


def hourly_counts(aadf_path: Path = AADF_PATH) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Read #21's AADF CSV and convert each count point to veh/h for MODELLED_HOUR.

    Returns {count_point_id: veh/h} and {count_point_id: its edge ids, both directions}.
    """
    counts, edges = {}, defaultdict(list)
    with aadf_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["count_point_id"]
            if cid not in edges:
                counts[cid] = float(row["aadf"]) * PEAK_SHARE
            edges[cid].append(row["edge_id"])
    return counts, dict(edges)


def assignment_proportions(net: ta.Network, centroids: list[int], prior: list[list[float]],
                           count_links: dict[str, list[int]]) -> td.AssignmentProportions:
    """Run the equilibrium on the prior and return each count point's share of every OD pair's trips.

    A count point's share sums over the links of both directions of its road.
    """
    points_on_link = defaultdict(list)
    for cid, links in count_links.items():
        for link in links:
            points_on_link[link].append(cid)
    zone = {centroid: i for i, centroid in enumerate(centroids)}
    shares = {cid: defaultdict(float) for cid in count_links}
    for path in ta.user_equilibrium(net, od_matrix(prior, centroids), method="gp").paths.all_paths():
        cell = (zone[path.origin], zone[path.destination])
        share = path.flow / prior[cell[0]][cell[1]]
        for link in path.links:
            for cid in points_on_link.get(link, ()):
                shares[cid][cell] += share
    return td.AssignmentProportions(list(count_links), (len(centroids), len(centroids)), shares)


def estimate_od(net: ta.Network, centroids: list[int], counts: dict[str, float],
                count_links: dict[str, list[int]]) -> tuple[list[list[float]], dict[str, tuple[float, float]]]:
    """Build the prior, scale it to the counts' level, and fit it to the counts with GLS.

    The scale is the least-squares factor between the counts and a free-flow assignment of
    the unit prior, so the equilibrium runs at roughly the right congestion.
    Returns the estimated OD matrix and {count_point_id: (modelled, counted)} in veh/h.
    """
    free_flow = net.free_flow_times()
    unit = gravity_prior(free_flow_costs(net, centroids, free_flow), 1.0)
    flows = ta.all_or_nothing(net, od_matrix(unit, centroids), free_flow).flows
    modelled = {cid: math.fsum(flows[link] for link in links) for cid, links in count_links.items()}
    fit_denominator = math.fsum(m * m for m in modelled.values())
    if fit_denominator == 0.0:
        raise ValueError("no count point lies on any zone-to-zone route; check the zones and the count matches")
    scale = math.fsum(counts[cid] * m for cid, m in modelled.items()) / fit_denominator
    prior = [[trips * scale for trips in row] for row in unit]

    proportions = assignment_proportions(net, centroids, prior, count_links)
    counted = [counts[cid] for cid in proportions.links]
    result = td.gls_matrix_estimation(prior, counted, proportions)
    return result.matrix, {cid: (m, c) for cid, m, c in zip(proportions.links, result.modelled_counts, counted)}


def main() -> None:
    """Estimate the OD matrix, write it labelled as estimated with its hour, and write the count fit."""
    graph = load_graph()
    net, node_ids = to_network(graph)
    edge_to_link, _ = link_maps(net)
    counts, count_edges = hourly_counts(AADF_PATH)
    count_links = {cid: [edge_to_link[eid] for eid in eids] for cid, eids in count_edges.items()}
    centroids = make_zones(net, [graph.node(nid).xy for nid in node_ids], N_ZONES)
    matrix, fit = estimate_od(net, centroids, counts, count_links)

    OD_PATH.parent.mkdir(parents=True, exist_ok=True)
    OD_PATH.write_text(json.dumps({
        "label": "estimated",
        "hour": MODELLED_HOUR,
        "units": "veh/h",
        "aadf_share": PEAK_SHARE,
        "zone_centroid_nodes": [node_ids[c] for c in centroids],
        "matrix": matrix,
    }, indent=1), encoding="utf-8")
    with FIT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("count_point_id", "modelled_veh_h", "counted_veh_h"))
        writer.writerows((cid, f"{m:.1f}", f"{c:.1f}") for cid, (m, c) in fit.items())

    total = math.fsum(map(math.fsum, matrix))
    print(f"Estimated OD matrix: {len(centroids)} zones, {total:.0f} veh/h, {MODELLED_HOUR}, fitted to {len(fit)} counts")
    for cid, (m, c) in fit.items():
        print(f"  {cid}: modelled {m:.0f} vs counted {c:.0f} veh/h")


if __name__ == "__main__":
    main()
