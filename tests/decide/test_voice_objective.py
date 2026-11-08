import csv

import numpy as np
import pytest

from src.concerns import concern_field as cf
from src.decide import design_search as ds
from src.decide import pedestrian_crossing_delay as pcd
from src.decide import voice_objective as vo
from src.decide.baseline_user_equilibrium import ue_rows
from tests.decide.two_route import build_context, build_graph

BASELINE = {"multiplier": 1.0, "closed_edge_ids": []}
NORTH = ["0", "1", "2", "3"]          # North Road, both directions; crossing at node 1


def write_field(path, graph, values: dict[str, dict[str, float]], basis=cf.LOCATED):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cf.FIELD_COLUMNS)
        writer.writeheader()
        writer.writerows(cf.field_rows(graph, values, basis))
    return path


def field_of(graph, values: dict[str, dict[str, float]]) -> dict:
    """An in-memory field like load_field's, one row per directed edge."""
    edge_ids = [str(e.eid) for e in graph.iter_edges()]
    return {"edge_ids": edge_ids, "basis": cf.LOCATED,
            "values": {t: np.array([values.get(e, {}).get(t, 0.0) for e in edge_ids]) for t in cf.THEMES}}


@pytest.fixture(scope="module")
def graph():
    return build_graph()


@pytest.fixture(scope="module")
def context(graph, tmp_path_factory):
    return build_context(graph, tmp_path_factory.mktemp("voice"))


@pytest.fixture(scope="module")
def at_crossing(context):
    return vo.crossing_edges(context["net"], context["link_to_edge"], context["crossings"])


def metrics_for(context, at_crossing, mask):
    net, links, crossings, result = ds.solve(context, mask, BASELINE)
    assert result.converged
    return vo.edge_metrics(net, links, crossings, result, at_crossing), (net, links, crossings, result)


@pytest.fixture(scope="module")
def baseline_metrics(context, at_crossing):
    return metrics_for(context, at_crossing, 0)[0]


# ------------------------------------------------------------------ the field

def test_load_field_reads_25s_csv(graph, tmp_path):
    path = write_field(tmp_path / "field.csv", graph, {"2": {"noise": 1.5, "parking": 2.0}})
    field = vo.load_field(path)
    assert field["edge_ids"] == [str(e.eid) for e in graph.iter_edges()] and field["basis"] == cf.LOCATED
    assert field["values"]["noise"][2] == 1.5 and field["values"]["parking"].sum() == 2.0
    assert vo.field_mass(field)["noise"] == 1.5


@pytest.mark.parametrize("edit, message", [
    (lambda rows: [row[:-1] for row in rows], "expected columns"),
    (lambda rows: rows[:1] + [rows[1][:-1] + [cf.PER_STREET]] + rows[2:], "one basis"),
    (lambda rows: rows[:1] + [[rows[1][0], "-1"] + rows[1][2:]] + rows[2:], "non-negative"),
    (lambda rows: rows + [rows[1]], "more than once"),
])
def test_load_field_refuses_malformed_files(graph, tmp_path, edit, message):
    path = write_field(tmp_path / "field.csv", graph, {})
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(edit(rows))
    with pytest.raises(ValueError, match=message):
        vo.load_field(path)


# ------------------------------------------------------------------ the burden

def test_burden_is_the_ratio_to_baseline_where_the_baseline_is_positive():
    m = vo.burden(np.array([5.0, 10.0]), np.array([10.0, 10.0]), np.ones(2))
    assert list(m) == [0.5, 1.0]


def test_burden_on_a_zero_baseline_uses_the_mean_positive_baseline_over_all_edges():
    # Edges a (b 10), b (b 0, x 20) and z (b 30, no concern mass): mean positive b = 20, m_b = 1 + 20/20.
    x, b, mass = np.array([10.0, 20.0, 30.0]), np.array([10.0, 0.0, 30.0]), np.array([1.0, 2.0, 0.0])
    assert list(vo.burden(x, b, mass)) == [1.0, 2.0, 1.0]
    assert vo.burden(np.zeros(2), np.zeros(2), np.ones(2)).tolist() == [1.0, 1.0]   # never metered: unchanged


def test_burden_with_no_positive_baseline_cannot_scale_a_new_burden():
    with pytest.raises(ValueError, match="can't be normalised"):
        vo.burden(np.array([0.0, 3.0]), np.zeros(2), np.array([0.0, 1.0]))
    assert vo.burden(np.array([3.0, 0.0]), np.zeros(2), np.array([0.0, 1.0])).tolist() == [1.0, 1.0]


# ------------------------------------------------------------------ J_voice

def test_j_voice_at_baseline_is_the_measured_field_mass(graph, baseline_metrics):
    values = {"0": {"rat_running": 2.0, "noise": 1.0, "parking": 4.0}, "5": {"crossing_delay": 3.0, "other": 1.0},
              "8": {"air_quality": 0.5, "accessibility": 2.0}}
    out = vo.j_voice(field_of(graph, values), baseline_metrics, baseline_metrics)
    assert out["total"] == 6.5
    assert out["by_theme"] == {"crossing_delay": 3.0, "road_safety": 0.0, "rat_running": 2.0, "air_quality": 0.5,
                               "severance": 0.0, "noise": 1.0}
    assert out["excluded_mass"] == {"missing_cycle_link": 0.0, "accessibility": 2.0, "parking": 4.0, "other": 1.0}
    assert vo.improvement(out["total"], out["total"]) == 0.0 and vo.improvement(0.0, 0.0) == 0.0


def test_excluded_themes_never_change_j_voice(graph, context, at_crossing, baseline_metrics):
    only_excluded = field_of(graph, {e: {"parking": 3.0, "other": 1.0, "missing_cycle_link": 2.0} for e in NORTH})
    for mask in (0, 1, 2, 3):
        metrics, _ = metrics_for(context, at_crossing, mask)
        assert vo.j_voice(only_excluded, metrics, baseline_metrics)["total"] == 0.0


def test_closing_a_complained_about_street_removes_its_rat_running_burden(graph, context, at_crossing, baseline_metrics):
    field = field_of(graph, {e: {"rat_running": 1.0} for e in NORTH})
    closed, _ = metrics_for(context, at_crossing, 1)                # close North Road
    out = vo.j_voice(field, closed, baseline_metrics)
    assert out["by_theme"]["rat_running"] == 0.0 and vo.improvement(out["total"], 4.0) == 1.0


def test_a_bus_lane_on_one_route_adds_rat_running_on_the_other(graph, context, at_crossing, baseline_metrics):
    field = field_of(graph, {e: {"rat_running": 1.0} for e in NORTH})
    bus_lane, _ = metrics_for(context, at_crossing, 2)              # halve South Road's capacity
    assert vo.j_voice(field, bus_lane, baseline_metrics)["total"] > 4.0


def test_crossing_delay_reaches_every_edge_at_the_crossing(context, at_crossing):
    assert at_crossing == {1: NORTH}                                # entering 0, 3 and leaving 1, 2
    for mask in (0, 1):
        metrics, (_, links, crossings, result) = metrics_for(context, at_crossing, mask)
        delay = pcd.crossing_delay(crossings[0], dict(zip(links, result.flows))).delay
        assert {e: metrics[vo.CROSSING_DELAY_S][e] for e in NORTH} == {e: delay for e in NORTH}


def test_vehicle_hours_are_flow_times_time_and_closed_edges_carry_no_traffic(context, at_crossing):
    metrics, (net, links, _, result) = metrics_for(context, at_crossing, 1)
    for row in ue_rows(net, result):
        assert metrics[vo.VEHICLE_HOURS][row["edge_id"]] == pytest.approx(row["flow_veh_h"] * row["ue_time_s"] / 3600)
    assert not set(NORTH) & set(metrics[vo.CAR_FLOW_VEH_H]) and not set(NORTH) & set(metrics[vo.VEHICLE_HOURS])


def test_j_voice_does_not_depend_on_the_units_of_the_metrics(graph, context, at_crossing, baseline_metrics):
    field = field_of(graph, {"0": {"rat_running": 2.0, "air_quality": 1.0}, "6": {"noise": 3.0, "crossing_delay": 1.0}})
    design, _ = metrics_for(context, at_crossing, 2)
    scale = lambda ms, k: {m: {e: v * k for e, v in values.items()} for m, values in ms.items()}  # noqa: E731
    base = vo.j_voice(field, design, baseline_metrics)["total"]
    assert vo.j_voice(field, scale(design, 1e6), scale(baseline_metrics, 1e6))["total"] == pytest.approx(base, rel=1e-12)


def test_unmetered_mass_counts_concern_on_edges_never_metered(graph, baseline_metrics):
    field = field_of(graph, {"8": {"crossing_delay": 2.0}, "0": {"crossing_delay": 1.0}})   # spur has no crossing
    assert vo.unmetered_mass(field, baseline_metrics)["crossing_delay"] == 2.0


# ------------------------------------------------------------------ guards found in review

def test_a_tiny_positive_baseline_stays_bounded_and_exactly_one_at_baseline():
    # 0.005 veh/h at baseline: a bare ratio would make 3000 veh/h a burden of 600,000.
    b = np.array([3000.0, 0.005])
    m = vo.burden(np.array([3000.0, 3000.0]), b, np.ones(2))
    assert m[0] == 1.0 and m[1] == pytest.approx(1 + (3000 - 0.005) / b.mean())
    assert vo.burden(b, b, np.ones(2)).tolist() == [1.0, 1.0]
    assert vo.burden(np.array([3000.0, 0.0]), b, np.ones(2))[1] == pytest.approx(1 - 0.005 / b.mean())


def test_floored_edges_counts_edges_with_mass_on_the_bounded_rule(graph):
    field = field_of(graph, {"0": {"noise": 1.0}, "1": {"noise": 1.0}, "2": {"parking": 1.0}})
    metrics = {vo.CAR_FLOW_VEH_H: {"0": 0.001, "1": 500.0, "2": 0.001}, vo.VEHICLE_HOURS: {}, vo.CROSSING_DELAY_S: {}}
    assert vo.floored_edges(field, metrics) == {"noise": 1}


def test_a_junction_crossing_reaches_the_crossed_road_but_not_side_streets():
    from engine.orchid import traffic_assignment as ta
    net = ta.Network("junction")
    links = []
    for name, (u, v) in {"100": (0, 1), "101": (1, 2), "102": (1, 3), "103": (3, 1), "104": (2, 1),
                         "106": (1, 0)}.items():
        net.add_link(u, v, ta.BPR(10.0, 1000.0), length=50.0, name=name)
        links.append(name)
    crossing = {"node": 7, "edge_ids": ["100", "104"], "signalised": False, "zebra": True, "width_m": 6.5}
    assert vo.crossing_edges(net, links, [crossing]) == {7: ["100", "101", "104", "106"]}
