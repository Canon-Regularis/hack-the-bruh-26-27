import json
import math

import numpy as np
import pytest
from scipy.stats import binom

from engine.orchid import street_network as sn
from src.concerns import hotspots as hs
from src.concerns.concern_field import THEMES

LAT, LON = 55.9490, -3.2000
M_PER_DEG_LAT = 111_195
M_PER_DEG_LON = M_PER_DEG_LAT * math.cos(math.radians(LAT))


def straight_road(spacings_m: list[float], one_way_last: bool = False) -> sn.RoadGraph:
    """A two-way street running east, one segment per spacing ("Low St", then "High St" from segment 20)."""
    g = sn.RoadGraph()
    x = 0.0
    g.add_node(0, lat=LAT, lon=LON)
    for nid, step in enumerate(spacings_m, start=1):
        x += step
        g.add_node(nid, lat=LAT, lon=LON + x / M_PER_DEG_LON)
        g.add_edge(nid - 1, nid, oneway=one_way_last and nid == len(spacings_m),
                   street_name="Low St" if nid <= 20 else "High St")
    return g


def grid(size: int, spacing_m: float = 80.0) -> sn.RoadGraph:
    """A size x size grid of two-way streets: 2 * size * (size - 1) segments."""
    g = sn.RoadGraph()
    for i in range(size):
        for j in range(size):
            g.add_node(i * size + j, lat=LAT + j * spacing_m / M_PER_DEG_LAT, lon=LON + i * spacing_m / M_PER_DEG_LON)
    for i in range(size):
        for j in range(size):
            if i + 1 < size:
                g.add_edge(i * size + j, (i + 1) * size + j, oneway=False, street_name=f"Row {j}")
            if j + 1 < size:
                g.add_edge(i * size + j, i * size + j + 1, oneway=False, street_name=f"Column {i}")
    return g


def record_at(x: float, y: float, count: int = 1, theme: str = "noise", severity: float = 1.0) -> dict:
    """A located record at planar (x, y) metres from the first node."""
    return {"theme": theme, "loc": (LAT + y / M_PER_DEG_LAT, LON + x / M_PER_DEG_LON), "street": "",
            "count": float(count), "weight": severity * count}


def unit_records(graph: sn.RoadGraph, counts_by_unit: dict[int, int]) -> list[dict]:
    _, points, _ = hs.segments(graph)
    return [record_at(*points[i], count=c) for i, c in counts_by_unit.items()]


# ------------------------------------------------------------------ units and placement

def test_segments_pair_twins_and_keep_one_way_edges():
    units, points, lengths = hs.segments(straight_road([50.0, 50.0, 50.0], one_way_last=True))
    assert [len(ids) for ids in units] == [2, 2, 1]
    assert np.allclose(points[:, 0], [25.0, 75.0, 125.0], rtol=1e-3) and np.allclose(points[:, 1], 0.0, atol=1e-6)
    assert np.allclose(lengths, 50.0, rtol=1e-3)


def test_a_curved_segment_sits_halfway_along_its_polyline():
    g = sn.RoadGraph()
    g.add_node("a", x=0.0, y=0.0)
    g.add_node("b", x=100.0, y=100.0)
    g.add_edge("a", "b", geometry=[(0.0, 0.0), (0.0, 100.0), (100.0, 100.0)])
    _, points, lengths = hs.segments(g)
    assert points[0] == pytest.approx([0.0, 100.0])   # not the chord midpoint (50, 50)
    assert lengths[0] == pytest.approx(200.0)


def test_band_neighbours_are_symmetric_and_exclude_self():
    nb = hs.band_neighbours(np.array([[0.0, 0], [50, 0], [100, 0], [300, 0]]), 130.0)
    assert [list(n) for n in nb] == [[1, 2], [0, 2], [0, 1], []]


def test_records_snap_to_their_segment_and_off_graph_ones_are_dropped():
    g = straight_road([50.0] * 10)
    units, points, _ = hs.segments(g)
    records = [record_at(*points[3], 2, "noise"), record_at(*points[3], 3, "parking", severity=0.5),
               record_at(points[3][0], 1000.0, 9), {"theme": "noise", "loc": None, "street": "Low St", "count": 1.0, "weight": 1.0}]
    counts, weights, placed = hs.place_records(g, units, records)
    assert placed == [3, 3, None, None]
    assert counts[3] == 5 and counts.sum() == 5
    assert weights[3, THEMES.index("noise")] == 2.0 and weights[3, THEMES.index("parking")] == 1.5


@pytest.mark.parametrize("count", [1.5, -1.0])
def test_counts_must_be_whole_comments(count):
    g = straight_road([50.0] * 3)
    units, points, _ = hs.segments(g)
    bad = record_at(*points[0])
    bad["count"] = count
    with pytest.raises(ValueError, match="whole number"):
        hs.place_records(g, units, [bad])


# ------------------------------------------------------------------ the test itself

def test_p_value_is_the_exact_binomial_upper_tail_by_length_share():
    counts = np.array([4.0, 1.0, 0.0, 0.0, 5.0])
    lengths = np.array([100.0, 100.0, 200.0, 100.0, 500.0])
    neighbours = [np.array(n, dtype=int) for n in ([1], [0, 2], [1], [], [])]
    observed, expected, p, testable = hs.neighbourhood_test(counts, lengths, neighbours)
    assert list(observed) == [5.0, 5.0, 1.0, 0.0, 5.0]
    shares = np.array([200, 400, 300, 100, 500]) / 1000.0
    assert expected == pytest.approx(10 * shares)
    assert p == pytest.approx([binom.sf(o - 1, 10, s) for o, s in zip(observed, shares)])
    assert p[3] == 1.0 and testable.all()


def test_untestable_units_get_p_one():
    neighbours = [np.array([1]), np.array([0])]
    _, _, p, testable = hs.neighbourhood_test(np.array([3.0, 0.0]), np.array([10.0, 10.0]), neighbours)
    assert not testable.any() and list(p) == [1.0, 1.0]      # each neighbourhood is the whole network
    _, _, p, testable = hs.neighbourhood_test(np.zeros(3), np.ones(3), [np.array([], dtype=int)] * 3)
    assert not testable.any() and list(p) == [1.0] * 3        # no comments at all
    with pytest.raises(ValueError, match="lengths"):
        hs.neighbourhood_test(np.ones(2), np.array([1.0, float("nan")]), [np.array([1]), np.array([0])])


def test_benjamini_hochberg_runs_over_testable_units_only_and_needs_an_excess():
    observed, expected = np.array([9.0, 9.0, 1.0, 0.0, 0.0]), np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    p = np.array([0.01, 0.04, 0.5, 0.001, 1.0])
    q, labels = hs.label_units(observed, expected, p, np.array([True, True, True, True, False]), fdr=0.05)
    assert q == pytest.approx([0.01 * 4 / 2, 0.04 * 4 / 3, 0.5, 0.001 * 4 / 1, 1.0])
    assert labels == [hs.HOT, hs.NOT_SIGNIFICANT, hs.NOT_SIGNIFICANT, hs.NOT_SIGNIFICANT, hs.NOT_SIGNIFICANT]


def test_clusters_link_hot_units_through_hot_neighbours_only():
    labels = [hs.HOT, hs.HOT, hs.NOT_SIGNIFICANT, hs.HOT, hs.NOT_SIGNIFICANT, hs.HOT]
    nb = [np.array(n) for n in ([1], [0, 2], [1, 3], [2], [5], [4])]
    assert hs.hotspot_clusters(labels, nb) == [[0, 1], [3], [5]]


# ------------------------------------------------------------------ whole method on synthetic towns

@pytest.fixture(scope="module")
def big_grid():
    g = grid(32)                                   # 1984 segments
    _, points, _ = hs.segments(g)
    return g, points


def test_randomly_placed_comments_rarely_make_a_hotspot(big_grid):
    # Independent comments placed at random: under this null BH at FDR 0.05 keeps the chance
    # of ANY hot segment in a run at about 5% or less, so allow at most 2 of 20 runs.
    g, points = big_grid
    lo, hi = points.min(axis=0), points.max(axis=0)
    runs_with_a_hotspot = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        records = [record_at(*rng.uniform(lo, hi)) for _ in range(120)]
        runs_with_a_hotspot += hs.find_hotspots(g, records)["hot_units"] > 0
    assert runs_with_a_hotspot <= 2


def test_many_comments_at_one_junction_make_a_hotspot(big_grid):
    # The most common real shape: a geocoded address or junction with many comments.
    g, points = big_grid
    lo, hi = points.min(axis=0), points.max(axis=0)
    rng = np.random.default_rng(5)
    junction = g.node(16 * 32 + 16).xy
    records = [record_at(*junction) for _ in range(40)] + [record_at(*rng.uniform(lo, hi)) for _ in range(30)]
    result = hs.find_hotspots(g, records)
    assert len(result["hotspots"]) == 1
    top = result["hotspots"][0]
    assert top["comments"] >= 40 and top["min_q"] <= hs.FDR
    units, _, _ = hs.segments(g)
    hot_points = points[[i for i, ids in enumerate(units) if ids[0] in top["edge_ids"]]]
    # every hot segment is within the band of a segment touching the junction (40 m away)
    assert np.linalg.norm(hot_points - np.array(junction), axis=1).max() < hs.BAND_M + 40 + 1


def test_one_spread_cluster_is_found_on_a_large_graph(big_grid):
    g, points = big_grid
    lo, hi = points.min(axis=0), points.max(axis=0)
    rng = np.random.default_rng(11)
    centre = (lo + hi) / 2
    cluster = [record_at(*(centre + rng.normal(0, 40, 2))) for _ in range(20)]
    noise = [record_at(*rng.uniform(lo, hi)) for _ in range(40)]
    result = hs.find_hotspots(g, cluster + noise)
    assert len(result["hotspots"]) >= 1
    top = result["hotspots"][0]
    assert top["located_records"] >= 10 and top["min_q"] <= hs.FDR
    units, _, _ = hs.segments(g)
    hot_points = points[[i for i, ids in enumerate(units) if ids[0] in top["edge_ids"]]]
    assert np.linalg.norm(hot_points.mean(axis=0) - centre) < 150


def test_long_segments_are_not_flagged_for_collecting_more_comments():
    # 20 short (20 m) then 20 long (120 m) segments; comments uniform per metre of street.
    # A per-segment null would flag the long half; the per-metre null keeps runs with any hot
    # segment at about the 5% FDR, and no more often on the long half than the short one.
    spacings = [20.0] * 20 + [120.0] * 20
    g = straight_road(spacings)
    total = sum(spacings)
    runs_with_a_hotspot, hot_on_long = 0, 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        result = hs.find_hotspots(g, [record_at(float(rng.uniform(0, total)), 0.0) for _ in range(80)])
        runs_with_a_hotspot += result["hot_units"] > 0
        hot_on_long += sum(1 for row in result["per_edge"][::2][20:] if row["label"] == hs.HOT)
    assert runs_with_a_hotspot <= 4 and hot_on_long <= 4


def test_hotspot_reports_twins_together_and_street_only_records_by_normalised_name():
    g = straight_road([50.0] * 40)
    records = unit_records(g, {i: 4 for i in (19, 20, 21)}) + [
        {"theme": "noise", "loc": None, "street": "high street", "count": 1.0, "weight": 1.0},   # "High St"
        {"theme": "noise", "loc": None, "street": "Nowhere Rd", "count": 1.0, "weight": 9.0},
    ]
    result = hs.find_hotspots(g, records)
    assert len(result["hotspots"]) == 1
    hot = result["hotspots"][0]
    assert hot["comments"] == 12 and hot["located_records"] == 3 and hot["records_naming_its_streets"] == 1
    assert hot["weight"]["noise"] == 12.0 and hot["max_rate_ratio"] > 1
    by_edge = {row["edge_id"]: row for row in result["per_edge"]}
    units, _, _ = hs.segments(g)
    for ids in units:
        assert len({(by_edge[e]["observed"], by_edge[e]["p"], by_edge[e]["label"]) for e in ids}) == 1
    assert result["unresolved_streets"] == ["Nowhere Rd"] and result["street_only_records"] == 2


def test_no_records_and_tiny_graphs_give_no_hotspots_and_valid_json():
    for g, records in ((straight_road([50.0] * 10), []), (straight_road([50.0, 50.0]), None)):
        records = unit_records(g, {0: 3}) if records is None else records
        result = hs.find_hotspots(g, records)
        assert result["hotspots"] == [] and result["hot_units"] == 0
        json.dumps(result, allow_nan=False)


def test_same_input_same_answer():
    g = straight_road([50.0] * 30)
    records = unit_records(g, {i: i % 4 for i in range(30) if i % 4})
    assert hs.find_hotspots(g, records) == hs.find_hotspots(g, records)
