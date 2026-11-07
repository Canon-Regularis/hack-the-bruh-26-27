"""Turn the road graph into an orchid traffic-assignment Network (ALG-S1-3, #22).

Links are named str(eid), so results can be reported under the edge ids Signal uses.
"""

from engine.orchid import street_network as sn
from engine.orchid import traffic_assignment as ta


def to_network(graph: sn.RoadGraph) -> tuple[ta.Network, list[int]]:
    """Build a Network from the car edges with capacity, one BPR link per edge.

    Nodes are renumbered 0..n-1 in OSM id order. Returns the network and the
    OSM node id for each network node, so OD demand can be placed on it.
    """
    edges = [e for e in graph.iter_edges() if e.permits("car") and e.capacity_veh_h > 0]
    node_ids = sorted({n for e in edges for n in (e.u, e.v)})
    index = {nid: i for i, nid in enumerate(node_ids)}
    net = ta.Network("road graph")
    for e in edges:
        net.add_link(index[e.u], index[e.v], ta.BPR(e.free_flow_time_s, e.capacity_veh_h),
                     length=e.length_m, name=str(e.eid))
    return net, node_ids


def link_maps(net: ta.Network) -> tuple[dict[str, int], list[str]]:
    """Return {str(eid): link index} and the str(eid) of each link, read from link names.

    Rebuild them after without_link, which shifts later link indices down by one.
    """
    link_to_edge = [link.name for link in net.links]
    return {eid: i for i, eid in enumerate(link_to_edge)}, link_to_edge


def od_matrix(matrix: list[list[float]], centroids: list[int]) -> ta.ODMatrix:
    """Turn a zone-by-zone trip matrix into an orchid ODMatrix on the centroids' network nodes."""
    return ta.ODMatrix({(centroids[i], centroids[j]): trips
                        for i, row in enumerate(matrix) for j, trips in enumerate(row) if trips > 0.0})
