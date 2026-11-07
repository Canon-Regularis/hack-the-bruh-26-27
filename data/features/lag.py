"""Spatial lag: recent bus-observed speed on neighbouring edges.

Leakage firewall: a row's lag uses only other edges, never its own street, and
only the bins before its own. Validation holds out whole streets, so a
neighbour on the same street would leak the answer.
"""

import pandas as pd

from speeds.edge_speeds import BIN_MINUTES

LAG_BINS = 4  # the previous hour


def neighbours(graph, edge_id):
    """Return the ids of edges touching either end, minus this edge and its street."""
    edge = graph.edge(int(edge_id))
    touching = {
        other
        for node in (edge.u, edge.v)
        for other in (*graph.out_edges(node), *graph.in_edges(node))
    }

    def same_street(other):
        return bool(edge.street_name) and graph.edge(other).street_name == edge.street_name

    return sorted(
        str(other)
        for other in touching
        if other not in (edge.eid, edge.reverse_eid) and not same_street(other)
    )


def observed_totals(speeds):
    """Return {(source, edge_id, bin): (metres, seconds)} for rows where a bus moved."""
    moved = speeds[speeds["n_traversals"] > 0]
    return {
        (row.source, row.edge_id, pd.Timestamp(row.bin_start_utc)): (
            row.total_metres,
            row.total_seconds,
        )
        for row in moved.itertuples(index=False)
    }


def lag_speed_kmh(totals, neighbour_ids, source, bin_start_utc):
    """Return total metres over total seconds on the neighbours, in km/h, or None.

    Uses the LAG_BINS bins before bin_start_utc, never that bin itself.
    """
    metres = seconds = 0.0
    for back in range(1, LAG_BINS + 1):
        earlier = pd.Timestamp(bin_start_utc) - pd.Timedelta(minutes=BIN_MINUTES * back)
        for neighbour in neighbour_ids:
            edge_metres, edge_seconds = totals.get((source, neighbour, earlier), (0.0, 0.0))
            metres += edge_metres
            seconds += edge_seconds
    return metres / seconds * 3.6 if seconds else None
