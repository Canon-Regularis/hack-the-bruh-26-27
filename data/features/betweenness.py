"""Edge betweenness from a fixed sample of start nodes.

The exact answer takes about 3 hours on the full Edinburgh graph. Sampling
start nodes and scaling up is the standard approximation, and a fixed seed
gives the same values every run.
"""

import json
import random

from engine.orchid import street_network as sn

SAMPLE_SOURCES = 500
SEED = 0


def sample_sources(graph, n_sources=SAMPLE_SOURCES, seed=SEED):
    """Return the start nodes: all of them if n_sources covers the graph."""
    nodes = sorted(graph.nodes)
    if n_sources >= len(nodes):
        return nodes
    return random.Random(seed).sample(nodes, n_sources)


def sampled_betweenness(graph, n_sources=SAMPLE_SOURCES, seed=SEED):
    """Return {str(edge id): betweenness} for cars, weighted by free-flow time."""
    sources = sample_sources(graph, n_sources, seed)
    scale = graph.num_nodes / len(sources)
    values = sn.edge_betweenness_centrality(
        graph, mode="car", weight="free_flow_time_s", sources=sources
    )
    return {str(edge_id): value * scale for edge_id, value in values.items()}


def read_betweenness(path):
    """Return saved betweenness, or an empty dict if it has not been computed."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
