"""J_voice: how much a design changes conditions on the streets people complained about (ALG-S3-5, #36).

    J_voice(x) = sum_j sum_e s_e^j * m_e^j(x)

s_e^j is #25's concern intensity for theme j on edge e. m_e^j(x) is design x's burden on
that edge for theme j, relative to the baseline (the do-nothing design under baseline
demand): exactly 1 at baseline, below 1 where the design relieves the edge, above 1 where
it adds to it. So J_voice(baseline) equals the field mass of the themes it measures, and
improvement = 1 - J_voice(x) / J_voice(baseline) is exactly 0 at baseline.

Theme -> metric, all lower-is-better:

    crossing_delay              crossing_delay_s  #45's delay at the crossings an edge enters or leaves
    road_safety, rat_running,   car_flow_veh_h    UE flow: exposure. Noise energy and severance scale
    severance, noise                              with flow; a fixed per-edge speed factor cancels in
                                                  the ratio, and congested speed would reward congestion
    air_quality                 vehicle_hours     flow x time: emissions_environment's edge proxy
                                                  E = a*f*t, with idle and stops not modelled, so a cancels
    missing_cycle_link,         --                excluded, field mass reported: not modelled here
    accessibility
    parking, other              --                excluded, field mass reported: unmeasurable

The burden follows public_voice's edge normalisation, with one guard: m_e = x_e / b_e where the
baseline b_e is at least FLOOR (1%) of the mean positive baseline b+ over all field edges;
otherwise m_e = 1 + (x_e - b_e) / b+. For b_e = 0 that is public_voice's 1 + x_e / b+; for a
tiny positive baseline it stays bounded (a bare ratio would blow up: 0.005 veh/h at baseline
turned ordinary traffic into a burden of thousands) and is still exactly 1 at baseline. An edge
missing from a metric (closed, not a car edge, no capacity) reads as 0. Stand-in for orchid's
public_voice voice_objective / voice_improvement, which have not converged (#34); it raises
in fewer cases than orchid (only when a new burden lands on an edge with concern mass and the
metric is zero everywhere at baseline).

Read the numbers with these in mind:
- A closure sets its own edges' traffic to 0, so the closed street's complaints vanish from
  J_voice, while the traffic it diverts only counts where there are complaints too. A design
  can look far better on J_voice than it is for the town as a whole.
- Edges with complaints but no metric (footways, edges no design touches) stay at m = 1. They
  dilute improvement; unmetered_mass reports how much.
- Crossing delay is the crossing's: a crossing on a closed street keeps its signal or zebra
  wait, so closing the street doesn't relieve its crossing_delay complaints.
- #25's kernel gives both directions of a two-way street a comment's full intensity, and the
  crossing_delay and flow themes move with J_ped and J_car, so the objectives are correlated.
"""

import csv
import math
from pathlib import Path

import numpy as np

from src.concerns.concern_field import FIELD_COLUMNS, FIELD_PATH, LOCATED, PER_STREET, THEMES
from src.decide.pedestrian_crossing_delay import crossing_delays

CROSSING_DELAY_S, CAR_FLOW_VEH_H, VEHICLE_HOURS = "crossing_delay_s", "car_flow_veh_h", "vehicle_hours"
PAIRING: dict[str, str | None] = {
    "crossing_delay": CROSSING_DELAY_S, "road_safety": CAR_FLOW_VEH_H, "rat_running": CAR_FLOW_VEH_H,
    "air_quality": VEHICLE_HOURS, "missing_cycle_link": None, "accessibility": None, "parking": None,
    "severance": CAR_FLOW_VEH_H, "noise": CAR_FLOW_VEH_H, "other": None,
}
EXCLUDED_WHY = {"missing_cycle_link": "no cycle network model", "accessibility": "no accessibility model",
                "parking": "unmeasurable", "other": "unmeasurable"}
FLOOR = 0.01            # baselines below FLOOR x the mean positive baseline use the bounded rule
NORMALISATION = ("edge: m = x / b where the baseline b >= 1% of the mean positive baseline b+, "
                 "else 1 + (x - b) / b+ (bounded, exactly 1 at baseline)")


def load_field(path: Path = FIELD_PATH) -> dict:
    """Read #25's field CSV as {"edge_ids": [...], "values": {theme: array}, "basis": str}.

    Refuses a wrong header, a repeated edge id, more than one basis or an unknown one, and
    negative or non-finite intensities.
    """
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != FIELD_COLUMNS:
            raise ValueError(f"{path}: expected columns {', '.join(FIELD_COLUMNS)}, got {reader.fieldnames}")
        rows = list(reader)
    edge_ids = [row["edge_id"] for row in rows]
    if len(set(edge_ids)) != len(edge_ids):
        raise ValueError(f"{path}: an edge id appears more than once")
    bases = {row["basis"] for row in rows}
    if len(bases) != 1 or not bases <= {LOCATED, PER_STREET}:
        raise ValueError(f"{path}: expected one basis, {LOCATED!r} or {PER_STREET!r}, got {sorted(bases)}")
    values = {theme: np.array([float(row[theme]) for row in rows]) for theme in THEMES}
    if any(not np.all(np.isfinite(v)) or np.any(v < 0) for v in values.values()):
        raise ValueError(f"{path}: concern intensities must be finite and non-negative")
    return {"edge_ids": edge_ids, "values": values, "basis": bases.pop()}


def field_mass(field: dict) -> dict[str, float]:
    """Return each theme's total concern intensity."""
    return {theme: math.fsum(field["values"][theme].tolist()) for theme in THEMES}


def crossing_edges(net, links: list[str], crossings: list[dict]) -> dict:
    """Return {crossing node: the edge ids of the road it crosses, both directions}, from the
    do-nothing network.

    #45 lists a crossing's entering edges; the leaving ones are their reverse twins (the link
    back to where an entering edge came from), so side streets at a junction crossing are not
    included. Build this once from the full network: a design that closes a street drops its
    links, but the crossing and its complaints stay where they are.
    """
    position = {eid: i for i, eid in enumerate(links)}
    out = {}
    for crossing in crossings:
        edges = set(crossing["edge_ids"])
        entering = [net.links[position[eid]] for eid in crossing["edge_ids"] if eid in position]
        back = {(link.head, link.tail) for link in entering}
        edges |= {links[i] for i, link in enumerate(net.links) if (link.tail, link.head) in back}
        out[crossing["node"]] = sorted(edges, key=lambda e: (len(e), e))
    return out


def edge_metrics(net, links: list[str], crossings: list[dict], result, at_crossing: dict) -> dict[str, dict[str, float]]:
    """Return {metric: {edge_id: value}} for one solved design: car flow, vehicle-hours, and the
    summed delay of the crossings each edge touches (see crossing_edges).
    """
    flows = dict(zip(links, result.flows))
    hours = {eid: f * t / 3600.0 for eid, f, t in zip(links, result.flows, result.travel_times)}
    delay: dict[str, float] = {}
    for crossing, outcome in zip(crossings, crossing_delays(crossings, flows)):
        for eid in at_crossing.get(crossing["node"], crossing["edge_ids"]):
            delay[eid] = delay.get(eid, 0.0) + outcome.delay
    return {CAR_FLOW_VEH_H: flows, VEHICLE_HOURS: hours, CROSSING_DELAY_S: delay}


def _vector(metric: dict[str, float], edge_ids: list[str], name: str) -> np.ndarray:
    values = np.array([metric.get(eid, 0.0) for eid in edge_ids], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"metric {name} must be finite and non-negative on every edge")
    return values


def burden(x: np.ndarray, b: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Return m_e: x/b where b >= FLOOR x mean(positive b), else 1 + (x - b) / mean(positive b).

    With no positive baseline at all, edges stay at 1, and a burden on an edge with concern
    mass can't be scaled, so it raises.
    """
    positive = b > 0
    if not positive.any():
        if np.any((x > 0) & (mass > 0)):
            raise ValueError("a metric is zero everywhere at baseline but not in the design, so it can't be normalised")
        return np.ones(len(x))
    scale = b[positive].mean()
    ratio = b >= FLOOR * scale
    return np.where(ratio, x / np.where(ratio, b, 1.0), 1.0 + (x - b) / scale)


def floored_edges(field: dict, baseline_metrics: dict, pairing: dict = PAIRING) -> dict[str, int]:
    """Return, per measured theme with concern mass, how many edges with mass have a positive
    baseline below FLOOR x the mean and so use the bounded rule instead of the ratio.
    """
    out = {}
    for theme, metric in pairing.items():
        s = field["values"][theme]
        if metric is None or not s.any():
            continue
        b = _vector(baseline_metrics[metric], field["edge_ids"], metric)
        if (b > 0).any():
            out[theme] = int(np.count_nonzero((s > 0) & (b > 0) & (b < FLOOR * b[b > 0].mean())))
    return out


def j_voice(field: dict, metrics: dict, baseline_metrics: dict, pairing: dict = PAIRING) -> dict:
    """Return {"total", "by_theme" (every measured theme), "excluded_mass" (the rest)}.

    A measured theme with no field mass contributes 0 and needs no metric.
    """
    by_theme, excluded = {}, {}
    for theme in THEMES:
        s = field["values"][theme]
        metric = pairing[theme]
        if metric is None:
            excluded[theme] = math.fsum(s.tolist())
            continue
        if not s.any():
            by_theme[theme] = 0.0
            continue
        x = _vector(metrics[metric], field["edge_ids"], metric)
        b = _vector(baseline_metrics[metric], field["edge_ids"], metric)
        by_theme[theme] = math.fsum((s * burden(x, b, s)).tolist())
    return {"total": math.fsum(by_theme.values()), "by_theme": by_theme, "excluded_mass": excluded}


def unmetered_mass(field: dict, baseline_metrics: dict, pairing: dict = PAIRING) -> dict[str, float]:
    """Return, per measured theme, the concern mass on edges whose baseline metric is 0. Those
    edges use the 1 + x / mean(positive b) rule, so they stay at 1 unless a design adds to them.
    """
    out = {}
    for theme, metric in pairing.items():
        if metric is not None:
            b = _vector(baseline_metrics[metric], field["edge_ids"], metric)
            out[theme] = math.fsum(field["values"][theme][b == 0].tolist())
    return out


def improvement(j: float, j_baseline: float) -> float:
    """Return 1 - j / j_baseline: the share of the baseline burden a design removes (0 at baseline)."""
    return 0.0 if j_baseline == 0 else 1.0 - j / j_baseline
