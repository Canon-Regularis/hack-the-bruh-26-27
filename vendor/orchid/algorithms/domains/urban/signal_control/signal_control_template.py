"""
Signal Control -- junction capacity, delay, staging and signal-timing optimisation
=================================================================================

**Green time is the scarcest currency a city has.**  Every junction is a tiny
auction in which cars, buses, cyclists and people on foot bid for the same
seconds, and the clearing price is delay.  This module is the static timing
theory of that auction: how much traffic a stopline can discharge, how much
delay a given split causes, which movements may run together, how long the
clearance periods must be, and -- given all of that -- which plan is best.

It is deliberately the *theory* half of the signals stack.  Its sibling
``realtime_operations`` does receding-horizon control: it re-solves a rolling
optimisation against live detector data.  ``signal_control`` owns everything
that plan depends on -- saturation flow, capacity, the Webster / HCM delay
families, conflict matrices and stage generation, ring-barrier structure,
intergreen and pedestrian minima, split / offset / bandwidth optimisation, and
the classical adaptive *policies* (actuated gap-out, max-pressure, Laemmer) in
their pure, self-contained form.  ``traffic_flow`` owns link dynamics;
``pedestrian_flow`` owns crowd movement; this module owns the stopline.

Everything is pure standard-library Python: the simplex used by Allsop's LP and
by MAXBAND, the Bron-Kerbosch clique enumerator used for stage generation, the
Robertson platoon recursion used for offsets, and the queue models used by the
max-pressure and Laemmer simulations are all built here from scratch.

Layers
------
0.  **Validation** -- ``SignalControlError`` plus the numeric guards.
1.  **Saturation flow & capacity** -- the HCM base rate with the full
    adjustment-factor chain (width, grade, parking, bus blockage, area type,
    heavy vehicles, lane utilisation, turning radius, pedestrian/bicycle
    conflict), capacity ``c = s * g / C``, flow ratios, degree of saturation.
2.  **Delay, queues & level of service** -- Webster's three-term delay and his
    optimal cycle, the HCM 2010/2016 control-delay chain (``d1`` uniform, ``d2``
    incremental, ``d3`` initial queue, progression factor ``PF``), Akcelik's
    overflow delay, average and 95th-percentile back of queue, LOS A-F.
3.  **Conflicts, compatibility & staging** -- movement conflict matrices,
    the compatibility graph, Bron-Kerbosch maximal cliques, stage generation
    and exhaustive conflict-freedom validation, minimum stage cover.
4.  **Ring-barrier structure** -- NEMA phase numbering, dual-ring plans and
    ring/barrier consistency validation.
5.  **Timing constraints** -- the ITE change-and-clearance formula, all-red,
    minimum green, pedestrian minima from crossing distance and walk speed,
    stage and total lost time.
6.  **The signal plan** -- ``SignalPlan``, a validated container in which the
    effective greens sum to exactly ``C - L``, plus the end-to-end designer.
7.  **Linear-programming kernel** -- a two-phase simplex with Bland's rule.
8.  **Fixed-time split optimisation** -- equisaturation / Webster splits,
    Allsop's LP (maximise the common multiplier), Smith's P0 policy, and a
    direct delay-minimising split search under minimum-green constraints.
9.  **Platoon dispersion & offsets** -- Robertson's recursion, cyclic flow
    profiles, TRANSYT-style link delay, exact DP and hill-climbing offset
    optimisation along an arterial.
10. **Bandwidth** -- exact arterial bandwidth by circular interval
    intersection, green waves, MAXBAND (integer loop variables enumerated, LP
    solved for the continuous part) and MULTIBAND's per-link bands.
11. **Actuated control** -- unit extension, gap-out / max-out, volume density.
12. **Max-pressure control** -- Varaiya's backpressure policy, a store-and-
    forward queue simulation and an LP stability margin.
13. **Self-organising control** -- the Laemmer-Helbing index policy with its
    stabilising regime.
14. **Transit signal priority** -- green extension, early start (red
    truncation), phase insertion, with compensation.
15. **Pedestrian phasing** -- exclusive stages, concurrent checks, leading
    pedestrian intervals, the Barnes dance and pedestrian delay.
16. **Unsignalised junctions** -- HCM two-way-stop gap acceptance, rank
    impedance, and the Siegloch / Raff / Tanner / maximum-likelihood critical
    gap estimators.
17. **Roundabouts** -- the HCM exponential form, Brilon-Wu, and the UK
    TRL/Kimber linear model with its geometric parameters.
18. **Payloads** -- JSON-able dictionaries for a frontend.
19. **Demo**.

Conventions
-----------
* Flows and saturation flows are in **veh/h** (pcu/h) at the interface; the
  formulae that need veh/s convert internally.  Times are in **seconds**,
  distances in **metres**, unless a name says otherwise.
* A *movement* is one turning manoeuvre from one approach; a *stage* (phase, in
  US usage) is a set of movements that runs together and must be conflict-free.
* ``C`` is the cycle time, ``L`` the total lost time, ``g`` an **effective**
  green and ``G`` a **displayed** green; ``g = G + I - l`` for a stage with
  intergreen ``I`` and lost time ``l``, so ``sum(g) == C - L`` by construction
  (``SignalPlan.validate`` checks that residual on every plan; it is zero for
  integer inputs and within 1e-9 otherwise).
* Saturation ``X = v / c`` is the degree of saturation (``v/c`` ratio);
  ``y = q / s`` is the flow ratio.
* Everything stochastic takes an explicit ``seed`` and uses ``random.Random``,
  so results are byte-identical across runs and independent of PYTHONHASHSEED.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

__version__ = "1.0.0"

Flow = List[float]
Matrix = List[List[float]]
BoolMatrix = List[List[bool]]
Stage = List[int]
Profile = List[float]

_EPS = 1e-12
_TOL = 1e-9
_INF = float("inf")

#: HCM base saturation flow rate, passenger cars per hour of green per lane.
_S0_DEFAULT = 1900.0
#: Gravitational acceleration (m/s^2) used by the ITE clearance formula.
_G_ACCEL = 9.81
#: Signalised level-of-service thresholds (control delay, s/veh), HCM.
_LOS_SIGNALISED = (10.0, 20.0, 35.0, 55.0, 80.0)
#: Unsignalised / roundabout level-of-service thresholds (s/veh), HCM.
_LOS_UNSIGNALISED = (10.0, 15.0, 25.0, 35.0, 50.0)

__all__ = [
    # errors
    "SignalControlError",
    # saturation flow & capacity
    "SaturationFlowFactors", "lane_width_factor", "grade_factor",
    "parking_factor", "bus_blockage_factor", "area_type_factor",
    "heavy_vehicle_factor", "lane_utilisation_factor", "turning_radius_factor",
    "pedestrian_bicycle_factor", "saturation_flow_rate", "movement_capacity",
    "flow_ratio", "degree_of_saturation", "critical_flow_ratios",
    "practical_cycle_length",
    # delay, queues & level of service
    "webster_optimal_cycle", "webster_delay", "webster_total_delay",
    "webster_optimal_cycle_numeric", "hcm_uniform_delay",
    "hcm_incremental_delay", "hcm_initial_queue_delay",
    "progression_adjustment_factor", "hcm_control_delay", "DelayResult",
    "level_of_service", "unsignalised_level_of_service", "akcelik_x0",
    "akcelik_delay", "average_back_of_queue", "queue_length_95",
    "queue_length_metres", "intersection_delay",
    # conflicts, compatibility & staging
    "TurnDirection", "Movement", "standard_four_arm_movements",
    "conflict_matrix", "compatibility_graph", "bron_kerbosch",
    "maximal_compatible_stages", "is_conflict_free", "validate_stages",
    "minimum_stage_cover", "stage_flow_ratio", "movements_in_stage",
    # ring-barrier & NEMA
    "NemaPhase", "RingBarrierPlan", "nema_eight_phase", "nema_phase_movements",
    "ring_barrier_sequence", "validate_ring_barrier",
    # timing constraints
    "yellow_change_interval", "all_red_clearance", "intergreen_time",
    "pedestrian_crossing_time", "minimum_green_time", "stage_lost_time",
    "total_lost_time",
    # the signal plan
    "SignalPlan", "make_signal_plan", "design_fixed_time_plan", "plan_delay",
    # linear-programming kernel
    "LPResult", "linear_program",
    # fixed-time split optimisation
    "apportion_green", "equisaturation_split", "webster_split",
    "allsop_lp_split", "smith_p0_pressure", "smith_p0_split",
    "delay_minimising_split",
    # platoon dispersion & offsets
    "PlatoonProfile", "cyclic_flow_profile", "robertson_dispersion",
    "departure_profile", "profile_delay", "link_delay_table",
    "optimise_offsets_dp", "optimise_offsets", "OffsetResult",
    # bandwidth
    "arterial_bandwidth", "green_wave_offsets", "bandwidth_efficiency",
    "BandwidthResult", "maxband", "multiband",
    # actuated control
    "ActuatedResult", "poisson_arrivals", "simulate_actuated_phase",
    "volume_density_gap", "expected_actuated_green",
    # max-pressure control
    "movement_pressure", "stage_pressure", "max_pressure_stage",
    "MaxPressureResult", "simulate_max_pressure",
    "max_pressure_stability_margin",
    # self-organising control
    "laemmer_index", "laemmer_select_stage", "simulate_laemmer",
    # transit signal priority
    "TSPRequest", "TSPResult", "tsp_green_extension", "tsp_early_start",
    "tsp_phase_insertion", "apply_transit_priority",
    # pedestrian phasing
    "exclusive_pedestrian_stage", "concurrent_pedestrian_check",
    "leading_pedestrian_interval", "barnes_dance_time", "pedestrian_delay",
    # unsignalised junctions
    "hcm_potential_capacity", "rank_impedance_factor", "twsc_movement_capacity",
    "twsc_control_delay", "siegloch_capacity", "siegloch_critical_gap",
    "raff_critical_gap", "tanner_capacity", "tanner_critical_gap",
    "maximum_likelihood_critical_gap",
    # roundabouts
    "hcm_roundabout_capacity", "brilon_wu_capacity", "KimberGeometry",
    "kimber_geometry", "kimber_capacity", "roundabout_delay",
    # payloads
    "build_signal_control_payload", "build_a2ui_payload",
]


# ===========================================================================
# 0. VALIDATION HELPERS
# ===========================================================================
class SignalControlError(ValueError):
    """Invalid input to a signal-control routine."""


def _as_floats(x: Sequence[Any], name: str = "x") -> List[float]:
    """Coerce a sequence to a list of finite floats or raise."""
    try:
        out = [float(v) for v in x]
    except (TypeError, ValueError):
        raise SignalControlError("%s must be a sequence of real numbers." % name)
    for v in out:
        if math.isnan(v) or math.isinf(v):
            raise SignalControlError("%s contains non-finite values." % name)
    return out


def _as_ints(x: Sequence[Any], name: str = "x") -> List[int]:
    """Coerce a sequence to a list of ints or raise."""
    try:
        return [int(v) for v in x]
    except (TypeError, ValueError):
        raise SignalControlError("%s must be a sequence of integers." % name)


def _finite(v: Any, name: str) -> float:
    """Coerce a scalar to a finite float or raise."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise SignalControlError("%s must be a real number." % name)
    if math.isnan(f) or math.isinf(f):
        raise SignalControlError("%s must be finite (got %r)." % (name, v))
    return f


def _positive(v: Any, name: str) -> float:
    """Coerce to a strictly positive finite float or raise."""
    f = _finite(v, name)
    if f <= 0.0:
        raise SignalControlError("%s must be > 0 (got %g)." % (name, f))
    return f


def _nonneg(v: Any, name: str) -> float:
    """Coerce to a non-negative finite float or raise."""
    f = _finite(v, name)
    if f < 0.0:
        raise SignalControlError("%s must be >= 0 (got %g)." % (name, f))
    return f


def _fraction(v: Any, name: str) -> float:
    """Coerce to a float in [0, 1] or raise."""
    f = _finite(v, name)
    if f < 0.0 or f > 1.0:
        raise SignalControlError("%s must lie in [0, 1] (got %g)." % (name, f))
    return f


def _check_len(x: Sequence[Any], n: int, name: str = "x") -> None:
    """Require a sequence of at least ``n`` items."""
    if len(x) < n:
        raise SignalControlError("%s needs at least %d entries (got %d)."
                                 % (name, n, len(x)))


def _same_len(a: Sequence[Any], b: Sequence[Any],
              na: str = "a", nb: str = "b") -> None:
    """Require two sequences of equal length."""
    if len(a) != len(b):
        raise SignalControlError("%s and %s must have equal length (%d vs %d)."
                                 % (na, nb, len(a), len(b)))


def _clamp(v: float, lo: float, hi: float) -> float:
    """Clamp ``v`` into ``[lo, hi]``."""
    return lo if v < lo else (hi if v > hi else v)


# ===========================================================================
# 1. SATURATION FLOW & CAPACITY
# ===========================================================================
def lane_width_factor(width_m: float) -> float:
    """HCM lane-width adjustment ``f_w = 1 + (W - 3.6)/9`` for W in metres."""
    w = _positive(width_m, "width_m")
    if w < 2.4:
        raise SignalControlError("lane width %g m is below the 2.4 m HCM floor." % w)
    return 1.0 + (w - 3.6) / 9.0


def grade_factor(grade_pct: float) -> float:
    """HCM grade adjustment ``f_g = 1 - G/200`` (G in percent, uphill positive)."""
    g = _finite(grade_pct, "grade_pct")
    if g < -6.0 or g > 10.0:
        raise SignalControlError("grade %g%% is outside the HCM -6..10%% range." % g)
    return 1.0 - g / 200.0


def parking_factor(n_lanes: int, manoeuvres_per_hour: float,
                   has_parking: bool = True) -> float:
    """HCM parking adjustment ``f_p = (N - 0.1 - 18 N_m/3600)/N``."""
    n = int(n_lanes)
    if n < 1:
        raise SignalControlError("n_lanes must be >= 1 (got %d)." % n)
    if not has_parking:
        return 1.0
    nm = _nonneg(manoeuvres_per_hour, "manoeuvres_per_hour")
    nm = min(nm, 180.0)                      # HCM caps N_m at 180 /h
    f = (n - 0.1 - 18.0 * nm / 3600.0) / n
    return max(f, 0.050)


def bus_blockage_factor(n_lanes: int, buses_stopping_per_hour: float) -> float:
    """HCM local-bus blockage adjustment ``f_bb = (N - 14.4 N_B/3600)/N``."""
    n = int(n_lanes)
    if n < 1:
        raise SignalControlError("n_lanes must be >= 1 (got %d)." % n)
    nb = _nonneg(buses_stopping_per_hour, "buses_stopping_per_hour")
    nb = min(nb, 250.0)                      # HCM caps N_B at 250 /h
    f = (n - 14.4 * nb / 3600.0) / n
    return max(f, 0.050)


def area_type_factor(cbd: bool = False) -> float:
    """HCM area-type adjustment: 0.90 in a central business district, else 1.0."""
    return 0.90 if cbd else 1.0


def heavy_vehicle_factor(pct_heavy: float, pce: float = 2.0) -> float:
    """HCM heavy-vehicle adjustment ``f_HV = 100/(100 + %HV (E_T - 1))``."""
    p = _nonneg(pct_heavy, "pct_heavy")
    if p > 100.0:
        raise SignalControlError("pct_heavy must be <= 100 (got %g)." % p)
    e = _finite(pce, "pce")
    if e < 1.0:
        raise SignalControlError("pce must be >= 1 (got %g)." % e)
    return 100.0 / (100.0 + p * (e - 1.0))


def lane_utilisation_factor(lane_volumes: Sequence[float]) -> float:
    """HCM lane-utilisation factor ``f_LU = v_g / (v_g1 * N)`` from lane volumes."""
    v = _as_floats(lane_volumes, "lane_volumes")
    _check_len(v, 1, "lane_volumes")
    for x in v:
        if x < 0.0:
            raise SignalControlError("lane volumes must be >= 0.")
    total = math.fsum(v)
    if total <= _EPS:
        return 1.0
    return total / (max(v) * len(v))


def turning_radius_factor(radius_m: float) -> float:
    """TRL turning-radius adjustment ``f_R = 1/(1 + 1.5/R)`` (R in metres)."""
    r = _positive(radius_m, "radius_m")
    return 1.0 / (1.0 + 1.5 / r)


def pedestrian_bicycle_factor(ped_flow: float, cycle: float, ped_green: float,
                              turn_proportion: float = 1.0,
                              bicycle_flow: float = 0.0,
                              protected_proportion: float = 0.0) -> float:
    """HCM Appendix-D pedestrian/bicycle blockage factor for a turning movement."""
    vp = _nonneg(ped_flow, "ped_flow")
    c = _positive(cycle, "cycle")
    gp = _positive(ped_green, "ped_green")
    if gp > c:
        raise SignalControlError("ped_green (%g) exceeds the cycle (%g)." % (gp, c))
    p_turn = _fraction(turn_proportion, "turn_proportion")
    vb = _nonneg(bicycle_flow, "bicycle_flow")
    p_prot = _fraction(protected_proportion, "protected_proportion")
    # pedestrian flow rate during the pedestrian service interval
    v_pedg = vp * c / gp
    if v_pedg <= 1000.0:
        occ_ped = v_pedg / 2000.0
    else:
        occ_ped = _clamp(0.4 + v_pedg / 10000.0, 0.0, 0.9)
    occ_ped = _clamp(occ_ped, 0.0, 0.9)
    # bicycle occupancy over the same interval
    v_bicg = vb * c / gp
    occ_bic = _clamp(0.02 + v_bicg / 2700.0, 0.0, 0.9) if vb > 0.0 else 0.0
    occ = occ_ped + occ_bic - occ_ped * occ_bic
    a_pbt = 1.0 - occ
    f = 1.0 - p_turn * (1.0 - a_pbt) * (1.0 - p_prot)
    return _clamp(f, 0.050, 1.0)


@dataclass
class SaturationFlowFactors:
    """The HCM adjustment-factor chain for one lane group."""

    n_lanes: int = 1
    base_rate: float = _S0_DEFAULT
    lane_width_m: float = 3.6
    grade_pct: float = 0.0
    pct_heavy: float = 0.0
    pce_heavy: float = 2.0
    parking: bool = False
    parking_manoeuvres: float = 0.0
    buses_stopping: float = 0.0
    cbd: bool = False
    lane_utilisation: float = 1.0
    turn_radius_m: Optional[float] = None
    left_turn_factor: float = 1.0
    right_turn_factor: float = 1.0
    ped_bike_factor: float = 1.0

    def factors(self) -> Dict[str, float]:
        """All multiplicative adjustment factors, keyed by their HCM symbol."""
        out = {
            "f_w": lane_width_factor(self.lane_width_m),
            "f_HV": heavy_vehicle_factor(self.pct_heavy, self.pce_heavy),
            "f_g": grade_factor(self.grade_pct),
            "f_p": parking_factor(self.n_lanes, self.parking_manoeuvres,
                                  self.parking),
            "f_bb": bus_blockage_factor(self.n_lanes, self.buses_stopping),
            "f_a": area_type_factor(self.cbd),
            "f_LU": _fraction(self.lane_utilisation, "lane_utilisation"),
            "f_LT": _fraction(self.left_turn_factor, "left_turn_factor"),
            "f_RT": _fraction(self.right_turn_factor, "right_turn_factor"),
            "f_pb": _fraction(self.ped_bike_factor, "ped_bike_factor"),
        }
        if self.turn_radius_m is not None:
            out["f_R"] = turning_radius_factor(self.turn_radius_m)
        return out

    def product(self) -> float:
        """The product of every adjustment factor."""
        p = 1.0
        for v in self.factors().values():
            p *= v
        return p


def saturation_flow_rate(factors: Optional[SaturationFlowFactors] = None,
                         **kwargs: Any) -> float:
    """Adjusted saturation flow ``s = s0 * N * prod(f)`` in veh/h of green."""
    f = factors if factors is not None else SaturationFlowFactors(**kwargs)
    if factors is not None and kwargs:
        raise SignalControlError("pass either a factors object or keywords, not both.")
    n = int(f.n_lanes)
    if n < 1:
        raise SignalControlError("n_lanes must be >= 1 (got %d)." % n)
    s0 = _positive(f.base_rate, "base_rate")
    return s0 * n * f.product()


def movement_capacity(sat_flow: float, green: float, cycle: float) -> float:
    """Movement capacity ``c = s * g / C`` (veh/h) for effective green ``g``."""
    s = _nonneg(sat_flow, "sat_flow")
    g = _nonneg(green, "green")
    c = _positive(cycle, "cycle")
    if g > c:
        raise SignalControlError("effective green (%g) exceeds the cycle (%g)."
                                 % (g, c))
    return s * g / c


def flow_ratio(demand: float, sat_flow: float) -> float:
    """Flow ratio ``y = q / s`` (dimensionless)."""
    q = _nonneg(demand, "demand")
    s = _positive(sat_flow, "sat_flow")
    return q / s


def degree_of_saturation(demand: float, capacity: float) -> float:
    """Degree of saturation ``X = v / c``; infinite when capacity is zero."""
    v = _nonneg(demand, "demand")
    c = _nonneg(capacity, "capacity")
    if c <= _EPS:
        return _INF if v > 0.0 else 0.0
    return v / c


def critical_flow_ratios(stages: Sequence[Sequence[int]],
                         demands: Sequence[float],
                         sat_flows: Sequence[float]) -> List[float]:
    """The largest flow ratio ``y = q/s`` within each stage (the critical movement)."""
    q = _as_floats(demands, "demands")
    s = _as_floats(sat_flows, "sat_flows")
    _same_len(q, s, "demands", "sat_flows")
    _check_len(stages, 1, "stages")
    out: List[float] = []
    for k, st in enumerate(stages):
        ys = [flow_ratio(q[m], s[m]) for m in st]
        if not ys:
            raise SignalControlError("stage %d is empty." % k)
        out.append(max(ys))
    return out


def practical_cycle_length(critical_y: Sequence[float], lost_time: float,
                           target_x: float = 0.90) -> float:
    """Cycle needed to hold every critical movement at ``target_x``: ``L/(1 - Y/X)``."""
    y = _as_floats(critical_y, "critical_y")
    _check_len(y, 1, "critical_y")
    L = _nonneg(lost_time, "lost_time")
    x = _positive(target_x, "target_x")
    if x > 1.0:
        raise SignalControlError("target_x must be <= 1 (got %g)." % x)
    Y = math.fsum(y)
    if Y >= x - _EPS:
        raise SignalControlError(
            "sum of critical flow ratios Y = %.4f is not below the target "
            "degree of saturation %.2f -- the junction is over capacity." % (Y, x))
    return L / (1.0 - Y / x)


# ===========================================================================
# 2. DELAY, QUEUES & LEVEL OF SERVICE
# ===========================================================================
def webster_optimal_cycle(lost_time: float, critical_y: Sequence[float]) -> float:
    """Webster's optimum cycle ``C0 = (1.5 L + 5) / (1 - Y)`` in seconds."""
    L = _nonneg(lost_time, "lost_time")
    y = _as_floats(critical_y, "critical_y")
    _check_len(y, 1, "critical_y")
    for v in y:
        if v < 0.0:
            raise SignalControlError("flow ratios must be >= 0.")
    Y = math.fsum(y)
    if Y >= 1.0 - _EPS:
        raise SignalControlError(
            "Y = %.4f >= 1: the junction cannot be signalised at any cycle." % Y)
    return (1.5 * L + 5.0) / (1.0 - Y)


def webster_delay(demand: float, sat_flow: float, green: float,
                  cycle: float) -> float:
    """Webster's three-term average delay (s/veh) for one movement."""
    q_h = _nonneg(demand, "demand")
    s = _positive(sat_flow, "sat_flow")
    g = _nonneg(green, "green")
    C = _positive(cycle, "cycle")
    if g > C:
        raise SignalControlError("green (%g) exceeds the cycle (%g)." % (g, C))
    lam = g / C
    cap = s * lam
    if cap <= _EPS:
        return _INF if q_h > 0.0 else 0.0
    x = q_h / cap
    d1 = C * (1.0 - lam) ** 2 / (2.0 * (1.0 - min(lam * x, 1.0 - 1e-15)))
    if q_h <= _EPS:
        return d1
    if x >= 1.0:
        return _INF
    q = q_h / 3600.0                      # veh/s, as Webster's formula requires
    d2 = x * x / (2.0 * q * (1.0 - x))
    d3 = 0.65 * (C / (q * q)) ** (1.0 / 3.0) * x ** (2.0 + 5.0 * lam)
    return d1 + d2 - d3


def webster_total_delay(cycle: float, demands: Sequence[float],
                        sat_flows: Sequence[float],
                        stages: Sequence[Sequence[int]],
                        lost_time: float) -> float:
    """Total Webster delay (veh-s/h) at a junction timed by equisaturation splits."""
    C = _positive(cycle, "cycle")
    L = _nonneg(lost_time, "lost_time")
    if C <= L:
        return _INF
    q = _as_floats(demands, "demands")
    s = _as_floats(sat_flows, "sat_flows")
    _same_len(q, s, "demands", "sat_flows")
    ycrit = critical_flow_ratios(stages, q, s)
    greens = equisaturation_split(C, L, ycrit)
    total = 0.0
    for k, st in enumerate(stages):
        for m in st:
            d = webster_delay(q[m], s[m], greens[k], C)
            if math.isinf(d):
                return _INF
            total += q[m] * d
    return total


def webster_optimal_cycle_numeric(demands: Sequence[float],
                                  sat_flows: Sequence[float],
                                  stages: Sequence[Sequence[int]],
                                  lost_time: float,
                                  c_min: float = 20.0, c_max: float = 200.0,
                                  step: float = 0.5) -> Tuple[float, float]:
    """Cycle minimising the *total* Webster delay, found by a fine sweep."""
    lo = _positive(c_min, "c_min")
    hi = _positive(c_max, "c_max")
    h = _positive(step, "step")
    if hi <= lo:
        raise SignalControlError("c_max must exceed c_min.")
    best_c, best_d = lo, _INF
    n = int(math.floor((hi - lo) / h)) + 1
    for i in range(n):
        c = lo + i * h
        d = webster_total_delay(c, demands, sat_flows, stages, lost_time)
        if d < best_d:
            best_c, best_d = c, d
    if math.isinf(best_d):
        raise SignalControlError("no feasible cycle in [%g, %g]." % (lo, hi))
    return best_c, best_d


def hcm_uniform_delay(cycle: float, green: float, x: float) -> float:
    """HCM uniform delay ``d1 = 0.5 C (1-g/C)^2 / (1 - min(1,X) g/C)`` (s/veh)."""
    C = _positive(cycle, "cycle")
    g = _nonneg(green, "green")
    X = _nonneg(x, "x") if not math.isinf(float(x)) else _INF
    if g > C:
        raise SignalControlError("green (%g) exceeds the cycle (%g)." % (g, C))
    lam = g / C
    xx = 1.0 if (math.isinf(X) or X > 1.0) else X
    denom = 1.0 - xx * lam
    if denom <= _EPS:
        return 0.5 * C * (1.0 - lam) ** 2 / _EPS
    return 0.5 * C * (1.0 - lam) ** 2 / denom


def hcm_incremental_delay(x: float, capacity: float, period_h: float = 0.25,
                          k: float = 0.50, upstream_i: float = 1.0) -> float:
    """HCM incremental delay ``d2 = 900T[(X-1) + sqrt((X-1)^2 + 8kIX/(cT))]``."""
    c = _nonneg(capacity, "capacity")
    T = _positive(period_h, "period_h")
    kk = _nonneg(k, "k")
    I = _positive(upstream_i, "upstream_i")
    if math.isinf(float(x)):
        return _INF
    X = _nonneg(x, "x")
    if c <= _EPS:
        return _INF if X > 0.0 else 0.0
    inner = (X - 1.0) ** 2 + 8.0 * kk * I * X / (c * T)
    return 900.0 * T * ((X - 1.0) + math.sqrt(max(inner, 0.0)))


def hcm_initial_queue_delay(initial_queue: float, capacity: float, x: float,
                            period_h: float = 0.25) -> float:
    """HCM initial-queue delay ``d3`` for an unmet queue left from the last period."""
    Qb = _nonneg(initial_queue, "initial_queue")
    c = _nonneg(capacity, "capacity")
    T = _positive(period_h, "period_h")
    if Qb <= _EPS:
        return 0.0
    if c <= _EPS:
        return _INF
    X = _INF if math.isinf(float(x)) else _nonneg(x, "x")
    xx = 1.0 if (math.isinf(X) or X > 1.0) else X
    spare = c * (1.0 - xx)
    if spare <= _EPS:
        t = T
    else:
        t = min(T, Qb / spare)
    if t < T - _TOL:
        u = 0.0
    else:
        u = max(0.0, 1.0 - c * T * (1.0 - xx) / Qb)
    return 1800.0 * Qb * (1.0 + u) * t / (c * T)


def progression_adjustment_factor(green: float, cycle: float,
                                  p_on_green: Optional[float] = None,
                                  arrival_type: int = 3) -> float:
    """HCM progression factor ``PF = (1 - P) f_PA / (1 - g/C)``."""
    C = _positive(cycle, "cycle")
    g = _nonneg(green, "green")
    if g > C:
        raise SignalControlError("green (%g) exceeds the cycle (%g)." % (g, C))
    lam = g / C
    at = int(arrival_type)
    if at < 1 or at > 6:
        raise SignalControlError("arrival_type must be 1..6 (got %d)." % at)
    # HCM Exhibit 16-12 default platoon ratios and supplemental factors
    rp_default = {1: 0.333, 2: 0.667, 3: 1.000, 4: 1.333, 5: 1.667, 6: 2.000}
    f_pa = {1: 1.00, 2: 0.93, 3: 1.00, 4: 1.15, 5: 1.00, 6: 1.00}[at]
    if p_on_green is None:
        P = _clamp(rp_default[at] * lam, 0.0, 1.0)
    else:
        P = _fraction(p_on_green, "p_on_green")
    if lam >= 1.0 - _EPS:
        return 1.0
    pf = (1.0 - P) * f_pa / (1.0 - lam)
    return _clamp(pf, 0.0, 2.0)


@dataclass
class DelayResult:
    """The HCM control-delay decomposition for one movement."""

    d1: float
    d2: float
    d3: float
    progression_factor: float
    control_delay: float
    x: float
    capacity: float
    los: str

    def as_dict(self) -> Dict[str, Any]:
        """A plain JSON-able view."""
        return {"d1": self.d1, "d2": self.d2, "d3": self.d3,
                "progressionFactor": self.progression_factor,
                "controlDelay": self.control_delay, "vOverC": self.x,
                "capacity": self.capacity, "los": self.los}


def hcm_control_delay(demand: float, sat_flow: float, green: float, cycle: float,
                      period_h: float = 0.25, k: float = 0.50,
                      upstream_i: float = 1.0, initial_queue: float = 0.0,
                      arrival_type: int = 3,
                      p_on_green: Optional[float] = None) -> DelayResult:
    """HCM 2010/2016 control delay ``d = d1*PF + d2 + d3`` with its parts."""
    v = _nonneg(demand, "demand")
    cap = movement_capacity(sat_flow, green, cycle)
    X = degree_of_saturation(v, cap)
    d1 = hcm_uniform_delay(cycle, green, X)
    pf = progression_adjustment_factor(green, cycle, p_on_green, arrival_type)
    d2 = hcm_incremental_delay(X, cap, period_h, k, upstream_i)
    d3 = hcm_initial_queue_delay(initial_queue, cap, X, period_h)
    total = d1 * pf + d2 + d3
    return DelayResult(d1=d1, d2=d2, d3=d3, progression_factor=pf,
                       control_delay=total, x=X, capacity=cap,
                       los=level_of_service(total))


def level_of_service(control_delay: float) -> str:
    """Signalised level of service A-F from control delay (s/veh), HCM bands."""
    d = float(control_delay)
    if math.isnan(d):
        raise SignalControlError("control_delay is NaN.")
    if d < 0.0:
        raise SignalControlError("control_delay must be >= 0 (got %g)." % d)
    for letter, bound in zip("ABCDE", _LOS_SIGNALISED):
        if d <= bound:
            return letter
    return "F"


def unsignalised_level_of_service(control_delay: float) -> str:
    """Two-way-stop / roundabout level of service A-F (HCM bands)."""
    d = float(control_delay)
    if math.isnan(d):
        raise SignalControlError("control_delay is NaN.")
    if d < 0.0:
        raise SignalControlError("control_delay must be >= 0 (got %g)." % d)
    for letter, bound in zip("ABCDE", _LOS_UNSIGNALISED):
        if d <= bound:
            return letter
    return "F"


def akcelik_x0(sat_flow: float, green: float) -> float:
    """Akcelik's overflow threshold ``x0 = 0.67 + s g / 600`` (s in veh/s)."""
    s = _nonneg(sat_flow, "sat_flow") / 3600.0
    g = _nonneg(green, "green")
    return 0.67 + s * g / 600.0


def akcelik_delay(demand: float, sat_flow: float, green: float, cycle: float,
                  period_h: float = 0.25) -> float:
    """Akcelik's (1981) average delay: uniform term plus a time-dependent overflow."""
    v = _nonneg(demand, "demand")
    s = _positive(sat_flow, "sat_flow")
    g = _nonneg(green, "green")
    C = _positive(cycle, "cycle")
    T = _positive(period_h, "period_h")
    if g > C:
        raise SignalControlError("green (%g) exceeds the cycle (%g)." % (g, C))
    u = g / C
    cap = s * u
    if cap <= _EPS:
        return _INF if v > 0.0 else 0.0
    x = v / cap
    y = v / s
    du = C * (1.0 - u) ** 2 / (2.0 * (1.0 - min(y, 1.0 - 1e-12)))
    x0 = akcelik_x0(s, g)
    if x <= x0:
        return du
    inner = (x - 1.0) ** 2 + 12.0 * (x - x0) / (cap * T)
    do = 900.0 * T * ((x - 1.0) + math.sqrt(max(inner, 0.0)))
    return du + do


def average_back_of_queue(demand: float, sat_flow: float, green: float,
                          cycle: float, period_h: float = 0.25
                          ) -> Tuple[float, float, float]:
    """Average back of queue (veh) as ``(uniform N1, overflow N2, total)``."""
    v = _nonneg(demand, "demand")
    s = _positive(sat_flow, "sat_flow")
    g = _nonneg(green, "green")
    C = _positive(cycle, "cycle")
    T = _positive(period_h, "period_h")
    if g > C:
        raise SignalControlError("green (%g) exceeds the cycle (%g)." % (g, C))
    u = g / C
    cap = s * u
    if cap <= _EPS:
        return (v * C / 3600.0, _INF if v > 0 else 0.0, _INF if v > 0 else 0.0)
    x = v / cap
    q = v / 3600.0                                  # veh/s
    n1 = q * C * (1.0 - u) / (1.0 - min(1.0, x) * u)
    x0 = akcelik_x0(s, g)
    if x <= x0:
        n2 = 0.0
    else:
        inner = (x - 1.0) ** 2 + 12.0 * (x - x0) / (cap * T)
        n2 = 0.25 * cap * T * ((x - 1.0) + math.sqrt(max(inner, 0.0)))
    return (n1, n2, n1 + n2)


def queue_length_95(demand: float, sat_flow: float, green: float, cycle: float,
                    period_h: float = 0.25) -> float:
    """95th-percentile back of queue (veh): ``N + 1.645 sqrt(N)`` (Poisson form)."""
    n1, n2, n = average_back_of_queue(demand, sat_flow, green, cycle, period_h)
    if math.isinf(n):
        return _INF
    return n + 1.645 * math.sqrt(max(n, 0.0))


def queue_length_metres(queue_veh: float, spacing_m: float = 7.0,
                        n_lanes: int = 1) -> float:
    """Convert a queue in vehicles to a stopline-back distance in metres."""
    q = _nonneg(queue_veh, "queue_veh") if not math.isinf(float(queue_veh)) else _INF
    sp = _positive(spacing_m, "spacing_m")
    n = int(n_lanes)
    if n < 1:
        raise SignalControlError("n_lanes must be >= 1 (got %d)." % n)
    if math.isinf(q):
        return _INF
    return q * sp / n


def intersection_delay(demands: Sequence[float],
                       delays: Sequence[float]) -> float:
    """Flow-weighted average control delay over every movement (s/veh)."""
    q = _as_floats(demands, "demands")
    d = _as_floats(delays, "delays") if not any(
        math.isinf(float(x)) for x in delays) else [float(x) for x in delays]
    _same_len(q, d, "demands", "delays")
    total_q = math.fsum(q)
    if total_q <= _EPS:
        return 0.0
    if any(math.isinf(x) for x in d):
        return _INF
    return math.fsum(q[i] * d[i] for i in range(len(q))) / total_q


# ===========================================================================
# 3. CONFLICTS, COMPATIBILITY & STAGE GENERATION
# ===========================================================================
class TurnDirection:
    """The turning manoeuvres a movement can make (plus a pedestrian crossing)."""

    LEFT = "left"
    THROUGH = "through"
    RIGHT = "right"
    UTURN = "uturn"
    PEDESTRIAN = "ped"
    ALL = (LEFT, THROUGH, RIGHT, UTURN, PEDESTRIAN)


#: Arms of a four-way junction in clockwise order, and their outward bearings.
_ARMS = ("N", "E", "S", "W")
_ARM_ANGLE = {"N": 90.0, "E": 0.0, "S": -90.0, "W": 180.0}


@dataclass(frozen=True)
class Movement:
    """One turning manoeuvre from one arm (or a pedestrian crossing of one arm)."""

    approach: str
    turn: str
    lanes: int = 1
    protected: bool = True
    label: str = ""

    def __post_init__(self) -> None:
        if self.approach not in _ARMS:
            raise SignalControlError("approach must be one of %s (got %r)."
                                     % (list(_ARMS), self.approach))
        if self.turn not in TurnDirection.ALL:
            raise SignalControlError("turn must be one of %s (got %r)."
                                     % (list(TurnDirection.ALL), self.turn))
        if int(self.lanes) < 1:
            raise SignalControlError("lanes must be >= 1 (got %r)." % (self.lanes,))
        if not self.label:
            object.__setattr__(self, "label", "%s-%s" % (self.approach, self.turn))

    @property
    def is_pedestrian(self) -> bool:
        """True for a pedestrian crossing pseudo-movement."""
        return self.turn == TurnDirection.PEDESTRIAN

    def exit_arm(self, drive_on_left: bool = True) -> Optional[str]:
        """The arm this movement discharges into (``None`` for pedestrians).

        ``approach`` names the *leg the vehicle comes from*, so a vehicle on the
        ``"N"`` approach starts north of the junction and heads south.  ``left``
        and ``right`` name the driver's turn, which is a fact of geometry and is
        therefore the same under either hand of drive: from the north arm a left
        turn always leaves by the east arm and a right turn always by the west
        arm.  What the hand of drive changes is which of the two is the
        *far-side* turn that has to cross the opposing stream -- see
        :func:`_is_far_side_turn` -- not the compass direction either turn
        points in.  ``drive_on_left`` is accepted for signature symmetry with
        the other geometry helpers and deliberately does not alter the result.
        """
        if self.is_pedestrian:
            return None
        i = _ARMS.index(self.approach)
        if self.turn == TurnDirection.THROUGH:
            return _ARMS[(i + 2) % 4]
        if self.turn == TurnDirection.UTURN:
            return _ARMS[i]
        if self.turn == TurnDirection.LEFT:
            return _ARMS[(i + 1) % 4]       # clockwise arms: N -> E, E -> S, ...
        return _ARMS[(i + 3) % 4]           # right turn: N -> W, E -> N, ...


def standard_four_arm_movements(turns: Sequence[str] = (TurnDirection.LEFT,
                                                        TurnDirection.THROUGH,
                                                        TurnDirection.RIGHT),
                                arms: Sequence[str] = _ARMS,
                                with_pedestrians: bool = False,
                                protected: bool = True) -> List[Movement]:
    """The movement list for a four-arm junction, optionally with ped crossings."""
    out: List[Movement] = []
    for a in arms:
        if a not in _ARMS:
            raise SignalControlError("unknown arm %r." % (a,))
        for t in turns:
            if t not in TurnDirection.ALL or t == TurnDirection.PEDESTRIAN:
                raise SignalControlError("unknown vehicle turn %r." % (t,))
            out.append(Movement(a, t, protected=protected))
    if with_pedestrians:
        for a in arms:
            out.append(Movement(a, TurnDirection.PEDESTRIAN))
    return out


def _arm_points(arm: str, drive_on_left: bool, offset_deg: float = 25.0
                ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """The (entry, exit) stopline points of an arm on the unit junction circle."""
    a = _ARM_ANGLE[arm]
    d = -offset_deg if drive_on_left else offset_deg
    ent = math.radians(a + d)
    ext = math.radians(a - d)
    return ((math.cos(ent), math.sin(ent)), (math.cos(ext), math.sin(ext)))


def _orient(a: Tuple[float, float], b: Tuple[float, float],
            c: Tuple[float, float]) -> float:
    """Twice the signed area of triangle abc (positive for a left turn)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(p1: Tuple[float, float], p2: Tuple[float, float],
                    q1: Tuple[float, float], q2: Tuple[float, float]) -> bool:
    """True when two segments cross properly (shared endpoints do not count)."""
    d1 = _orient(q1, q2, p1)
    d2 = _orient(q1, q2, p2)
    d3 = _orient(p1, p2, q1)
    d4 = _orient(p1, p2, q2)
    if abs(d1) < 1e-12 or abs(d2) < 1e-12 or abs(d3) < 1e-12 or abs(d4) < 1e-12:
        return False
    return ((d1 > 0.0) != (d2 > 0.0)) and ((d3 > 0.0) != (d4 > 0.0))


def _is_far_side_turn(m: Movement, drive_on_left: bool) -> bool:
    """True for the turn that crosses the opposing traffic stream."""
    if m.is_pedestrian:
        return False
    return m.turn == (TurnDirection.RIGHT if drive_on_left else TurnDirection.LEFT)


def _is_opposing_far_side_pair(a: Movement, b: Movement,
                               drive_on_left: bool) -> bool:
    """True for the two far-side turns off opposite arms (the dual-turn pair)."""
    return (_is_far_side_turn(a, drive_on_left)
            and _is_far_side_turn(b, drive_on_left)
            and (_ARMS.index(a.approach) - _ARMS.index(b.approach)) % 4 == 2)


def conflict_matrix(movements: Sequence[Movement], drive_on_left: bool = True,
                    merge_conflicts: bool = False,
                    opposing_turns_conflict: bool = False) -> BoolMatrix:
    """Pairwise conflict matrix: ``M[i][j]`` is True when i and j cannot run together.

    The two far-side turns off opposite arms (two opposing protected rights
    under drive-on-left, two opposing protected lefts under drive-on-right) are
    the one pair the stopline geometry cannot settle on its own: their paths are
    anti-parallel, so they never *cross* in the segment sense, yet whether they
    may share a stage is a matter of local practice.  ``opposing_turns_conflict``
    decides it outright and is the only thing that decides it: ``False`` (the
    default) lets them pass offside-to-offside, which is what the textbook
    32-conflict-point count for a four-arm crossroads assumes -- 16 crossing
    points, of which only four are far-side-turn-versus-far-side-turn and all
    four are between *adjacent* arms; ``True`` forbids the dual-turn stage.
    """
    mv = list(movements)
    _check_len(mv, 1, "movements")
    for m in mv:
        if not isinstance(m, Movement):
            raise SignalControlError("movements must be Movement instances.")
    n = len(mv)
    paths: List[Optional[Tuple[Tuple[float, float], Tuple[float, float]]]] = []
    for m in mv:
        if m.is_pedestrian:
            paths.append(None)
            continue
        ent = _arm_points(m.approach, drive_on_left)[0]
        ex_arm = m.exit_arm(drive_on_left)
        ext = _arm_points(str(ex_arm), drive_on_left)[1]
        paths.append((ent, ext))
    out: BoolMatrix = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = mv[i], mv[j]
            conflict = False
            if a.is_pedestrian and b.is_pedestrian:
                conflict = False
            elif a.is_pedestrian or b.is_pedestrian:
                ped, veh = (a, b) if a.is_pedestrian else (b, a)
                arm = ped.approach
                conflict = (veh.approach == arm or
                            veh.exit_arm(drive_on_left) == arm)
            elif a.approach == b.approach:
                conflict = False                     # diverging, separate lanes
            else:
                pa, pb = paths[i], paths[j]
                if _is_opposing_far_side_pair(a, b, drive_on_left):
                    # anti-parallel offside-to-offside paths: the segment test
                    # can never call this pair, so the caller's rule decides
                    conflict = bool(opposing_turns_conflict)
                else:
                    conflict = _segments_cross(pa[0], pa[1], pb[0], pb[1])
                if not conflict and merge_conflicts:
                    conflict = (a.exit_arm(drive_on_left) ==
                                b.exit_arm(drive_on_left))
                if conflict and not (a.protected and b.protected):
                    conflict = False                 # a permitted turn gives way
            out[i][j] = conflict
            out[j][i] = conflict
    return out


def compatibility_graph(conflict: BoolMatrix) -> BoolMatrix:
    """The complement of the conflict matrix (self-loops excluded)."""
    n = len(conflict)
    for row in conflict:
        if len(row) != n:
            raise SignalControlError("conflict matrix must be square.")
    for i in range(n):
        for j in range(n):
            if bool(conflict[i][j]) != bool(conflict[j][i]):
                raise SignalControlError(
                    "conflict matrix must be symmetric (entry %d,%d)." % (i, j))
    return [[(i != j) and (not bool(conflict[i][j])) for j in range(n)]
            for i in range(n)]


def bron_kerbosch(adjacency: BoolMatrix) -> List[List[int]]:
    """Every maximal clique of an undirected graph (Bron-Kerbosch with pivoting)."""
    n = len(adjacency)
    for row in adjacency:
        if len(row) != n:
            raise SignalControlError("adjacency matrix must be square.")
    nbr = [set(j for j in range(n) if j != i and adjacency[i][j]) for i in range(n)]
    for i in range(n):
        for j in nbr[i]:
            if i not in nbr[j]:
                raise SignalControlError(
                    "adjacency matrix must be symmetric (entry %d,%d)." % (i, j))
    cliques: List[List[int]] = []

    def expand(r: set, p: set, x: set) -> None:
        if not p and not x:
            cliques.append(sorted(r))
            return
        pivot = max(p | x, key=lambda v: (len(nbr[v] & p), -v))
        for v in sorted(p - nbr[pivot]):
            expand(r | {v}, p & nbr[v], x & nbr[v])
            p = p - {v}
            x = x | {v}

    if n:
        expand(set(), set(range(n)), set())
    cliques.sort(key=lambda c: (-len(c), c))
    return cliques


def maximal_compatible_stages(conflict: BoolMatrix) -> List[List[int]]:
    """Every maximal conflict-free set of movements -- the candidate stages."""
    comp = compatibility_graph(conflict)
    stages = bron_kerbosch(comp)
    n = len(conflict)
    seen = set()
    for st in stages:
        seen.update(st)
    for i in range(n):
        if i not in seen:                      # isolated vertex, singleton stage
            stages.append([i])
    stages.sort(key=lambda c: (-len(c), c))
    return stages


def is_conflict_free(stage: Sequence[int], conflict: BoolMatrix) -> bool:
    """True when no pair of movements in the stage conflicts."""
    st = _as_ints(stage, "stage")
    n = len(conflict)
    for m in st:
        if m < 0 or m >= n:
            raise SignalControlError("movement index %d out of range 0..%d."
                                     % (m, n - 1))
    for a, b in itertools.combinations(st, 2):
        if conflict[a][b]:
            return False
    return True


def validate_stages(stages: Sequence[Sequence[int]], conflict: BoolMatrix,
                    require_cover: bool = True) -> List[str]:
    """Every rule broken by a stage list; an empty list means the staging is legal."""
    problems: List[str] = []
    n = len(conflict)
    covered = set()
    for k, st in enumerate(stages):
        ids = _as_ints(st, "stage %d" % k)
        if not ids:
            problems.append("stage %d is empty" % k)
            continue
        if len(set(ids)) != len(ids):
            problems.append("stage %d repeats a movement" % k)
        for m in ids:
            if m < 0 or m >= n:
                problems.append("stage %d references movement %d out of range"
                                % (k, m))
        for a, b in itertools.combinations(sorted(set(ids)), 2):
            if 0 <= a < n and 0 <= b < n and conflict[a][b]:
                problems.append("stage %d runs conflicting movements %d and %d"
                                % (k, a, b))
        covered.update(i for i in ids if 0 <= i < n)
    if require_cover:
        for m in range(n):
            if m not in covered:
                problems.append("movement %d is never served" % m)
    return problems


def minimum_stage_cover(candidate_stages: Sequence[Sequence[int]],
                        n_movements: int,
                        max_stages: Optional[int] = None,
                        stage_costs: Optional[Sequence[float]] = None
                        ) -> List[List[int]]:
    """Smallest sub-family of stages serving every movement (ties on total cost)."""
    cands = [sorted(set(_as_ints(s, "stage"))) for s in candidate_stages]
    n = int(n_movements)
    if n < 1:
        raise SignalControlError("n_movements must be >= 1 (got %d)." % n)
    if not cands:
        raise SignalControlError("candidate_stages is empty.")
    for st in cands:
        for m in st:
            if m < 0 or m >= n:
                raise SignalControlError("movement %d out of range." % m)
    universe = (1 << n) - 1
    masks = []
    for st in cands:
        bits = 0
        for m in st:
            bits |= 1 << m
        masks.append(bits)
    union = 0
    for b in masks:
        union |= b
    if union != universe:
        raise SignalControlError(
            "the candidate stages do not cover every movement.")
    if stage_costs is None:
        costs = [0.0] * len(cands)
    else:
        costs = _as_floats(stage_costs, "stage_costs")
        if len(costs) != len(cands):
            raise SignalControlError("stage_costs needs one entry per candidate.")
        for v in costs:
            if v < 0.0:
                raise SignalControlError("stage costs must be >= 0.")
    order = sorted(range(len(cands)), key=lambda i: (-bin(masks[i]).count("1"),
                                                     costs[i], cands[i]))
    limit = len(cands) if max_stages is None else int(max_stages)
    # greedy warm start gives the initial incumbent
    covered, greedy = 0, []
    while covered != universe:
        pick = min(order, key=lambda i: (-bin(masks[i] & ~covered).count("1"),
                                         costs[i], i))
        if not masks[pick] & ~covered:
            break
        greedy.append(pick)
        covered |= masks[pick]
    best: List[int] = list(greedy) if covered == universe else []
    key0 = ((len(best), math.fsum(costs[i] for i in best)) if best
            else (limit + 1, _INF))
    state = {"best": best, "key": key0}

    def search(chosen: List[int], covered_bits: int, cost_so_far: float) -> None:
        if covered_bits == universe:
            key = (len(chosen), cost_so_far)
            if key < state["key"]:
                state["best"] = list(chosen)
                state["key"] = key
            return
        if len(chosen) >= limit:
            return
        if (len(chosen) + 1, cost_so_far) >= state["key"]:
            return
        missing = universe & ~covered_bits
        lowest = (missing & -missing).bit_length() - 1
        for idx in order:
            if not (masks[idx] >> lowest) & 1:
                continue
            if idx in chosen:
                continue
            search(chosen + [idx], covered_bits | masks[idx],
                   cost_so_far + costs[idx])

    search([], 0, 0.0)
    best = state["best"]
    if not best:
        raise SignalControlError("no stage cover found within %d stages." % limit)
    return [cands[i] for i in sorted(best, key=lambda i: (-len(cands[i]),
                                                          cands[i]))]


def stage_flow_ratio(stage: Sequence[int], demands: Sequence[float],
                     sat_flows: Sequence[float]) -> float:
    """The critical (largest) flow ratio ``y = q/s`` inside one stage."""
    q = _as_floats(demands, "demands")
    s = _as_floats(sat_flows, "sat_flows")
    _same_len(q, s, "demands", "sat_flows")
    st = _as_ints(stage, "stage")
    if not st:
        raise SignalControlError("stage is empty.")
    return max(flow_ratio(q[m], s[m]) for m in st)


def movements_in_stage(stage: Sequence[int],
                       movements: Sequence[Movement]) -> List[str]:
    """The labels of the movements served by a stage."""
    st = _as_ints(stage, "stage")
    return [movements[m].label for m in st]


# ===========================================================================
# 4. RING-BARRIER STRUCTURE & NEMA PHASE NUMBERING
# ===========================================================================
@dataclass(frozen=True)
class NemaPhase:
    """One NEMA phase: its number, ring, barrier group and movement label."""

    number: int
    ring: int
    barrier: int
    movement: str
    approach: str
    turn: str


def nema_phase_movements() -> Dict[int, Tuple[str, str, str]]:
    """The standard NEMA phase -> (label, origin arm, turn) map for a crossroads."""
    return {
        1: ("EBL", "W", TurnDirection.LEFT),
        2: ("WBT", "E", TurnDirection.THROUGH),
        3: ("NBL", "S", TurnDirection.LEFT),
        4: ("SBT", "N", TurnDirection.THROUGH),
        5: ("WBL", "E", TurnDirection.LEFT),
        6: ("EBT", "W", TurnDirection.THROUGH),
        7: ("SBL", "N", TurnDirection.LEFT),
        8: ("NBT", "S", TurnDirection.THROUGH),
    }


def nema_eight_phase() -> List[NemaPhase]:
    """The canonical dual-ring, two-barrier, eight-phase NEMA structure."""
    ring_of = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2}
    barrier_of = {1: 0, 2: 0, 5: 0, 6: 0, 3: 1, 4: 1, 7: 1, 8: 1}
    mv = nema_phase_movements()
    return [NemaPhase(number=p, ring=ring_of[p], barrier=barrier_of[p],
                      movement=mv[p][0], approach=mv[p][1], turn=mv[p][2])
            for p in range(1, 9)]


@dataclass
class RingBarrierPlan:
    """A dual-ring plan: phases, their green-plus-intergreen durations and a cycle."""

    phases: List[NemaPhase]
    durations: Dict[int, float]
    cycle: float
    offset: float = 0.0

    def rings(self) -> Dict[int, List[int]]:
        """Phase numbers in each ring, ordered by barrier then phase number."""
        out: Dict[int, List[int]] = {}
        order = {ph.number: (ph.barrier, ph.number) for ph in self.phases}
        for ph in self.phases:
            out.setdefault(ph.ring, []).append(ph.number)
        for r in out:
            out[r].sort(key=lambda p: order[p])
        return out

    def barrier_groups(self) -> Dict[int, List[int]]:
        """Phase numbers in each barrier group."""
        out: Dict[int, List[int]] = {}
        for ph in self.phases:
            out.setdefault(ph.barrier, []).append(ph.number)
        for b in out:
            out[b].sort()
        return out

    def barrier_duration(self, barrier: int, ring: int) -> float:
        """Total duration of one ring's phases inside one barrier group."""
        return math.fsum(self.durations[ph.number] for ph in self.phases
                         if ph.barrier == barrier and ph.ring == ring)

    def as_dict(self) -> Dict[str, Any]:
        """A plain JSON-able view."""
        return {"cycle": self.cycle, "offset": self.offset,
                "rings": dict((str(r), v) for r, v in self.rings().items()),
                "durations": dict((str(k), v) for k, v in self.durations.items()),
                "barriers": dict((str(b), v)
                                 for b, v in self.barrier_groups().items())}


def validate_ring_barrier(plan: RingBarrierPlan,
                          tol: float = 1e-6) -> List[str]:
    """Every ring/barrier rule broken by a dual-ring plan (empty list == legal)."""
    problems: List[str] = []
    numbers = [ph.number for ph in plan.phases]
    if len(set(numbers)) != len(numbers):
        problems.append("a phase number appears twice")
    for ph in plan.phases:
        if ph.number not in plan.durations:
            problems.append("phase %d has no duration" % ph.number)
        elif plan.durations[ph.number] < 0.0:
            problems.append("phase %d has a negative duration" % ph.number)
    if problems:
        return problems
    rings = plan.rings()
    for r in sorted(rings):
        total = math.fsum(plan.durations[p] for p in rings[r])
        if abs(total - plan.cycle) > tol:
            problems.append("ring %d sums to %.3f s, not the %.3f s cycle"
                            % (r, total, plan.cycle))
    for b in sorted(plan.barrier_groups()):
        totals = dict((r, plan.barrier_duration(b, r)) for r in sorted(rings))
        if totals and max(totals.values()) - min(totals.values()) > tol:
            problems.append(
                "barrier %d is not aligned across rings: %s"
                % (b, ", ".join("ring %d=%.3f" % (r, totals[r])
                                for r in sorted(totals))))
    # The barrier-order check has to read the DECLARED order -- the order the
    # phases appear in `plan.phases` -- not `rings()`.  `rings()` sorts each
    # ring by (barrier, number), so its output is non-decreasing in barrier by
    # construction and could never fail this test; worse, that sort silently
    # rewrites a plan whose author listed the phases in some other sequence,
    # because the data model has nowhere else to record an intended order.
    declared: Dict[int, List[int]] = {}
    for ph in plan.phases:
        declared.setdefault(ph.ring, []).append(ph.barrier)
    for r in sorted(declared):
        barriers = declared[r]
        if barriers != sorted(barriers):
            problems.append(
                "ring %d visits barriers out of order (declared %s, but a "
                "dual-ring controller serves barrier groups in order and the "
                "plan will run as %s)"
                % (r, barriers, sorted(barriers)))
    return problems


def ring_barrier_sequence(plan: RingBarrierPlan) -> List[Dict[str, Any]]:
    """Lay a dual-ring plan out on the clock: the concurrent phases in each interval."""
    problems = validate_ring_barrier(plan)
    if problems:
        raise SignalControlError("invalid ring-barrier plan: %s" % problems[0])
    rings = plan.rings()
    starts: Dict[int, float] = {}
    for r in sorted(rings):
        t = 0.0
        for p in rings[r]:
            starts[p] = t
            t += plan.durations[p]
    edges = sorted(set([0.0, plan.cycle] +
                       [starts[p] for p in starts] +
                       [starts[p] + plan.durations[p] for p in starts]))
    out: List[Dict[str, Any]] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a <= 1e-9:
            continue
        mid = 0.5 * (a + b)
        active = sorted(p for p in starts
                        if starts[p] <= mid < starts[p] + plan.durations[p])
        out.append({"start": a, "end": b, "duration": b - a, "phases": active})
    return out


# ===========================================================================
# 5. TIMING CONSTRAINTS: INTERGREEN, MINIMUM GREEN, PEDESTRIAN MINIMA
# ===========================================================================
def yellow_change_interval(speed_ms: float, reaction_time: float = 1.0,
                           deceleration: float = 3.05,
                           grade_pct: float = 0.0) -> float:
    """ITE change interval ``y = t + v / (2(a + g G))`` in seconds."""
    v = _positive(speed_ms, "speed_ms")
    t = _nonneg(reaction_time, "reaction_time")
    a = _positive(deceleration, "deceleration")
    G = _finite(grade_pct, "grade_pct") / 100.0
    denom = 2.0 * (a + _G_ACCEL * G)
    if denom <= _EPS:
        raise SignalControlError(
            "deceleration %g m/s^2 cannot overcome a %g%% downgrade."
            % (a, grade_pct))
    return t + v / denom


def all_red_clearance(crossing_width_m: float, speed_ms: float,
                      vehicle_length_m: float = 6.0) -> float:
    """ITE clearance interval ``r = (W + L) / v`` in seconds."""
    w = _nonneg(crossing_width_m, "crossing_width_m")
    v = _positive(speed_ms, "speed_ms")
    ell = _nonneg(vehicle_length_m, "vehicle_length_m")
    return (w + ell) / v


def intergreen_time(speed_ms: float, crossing_width_m: float,
                    reaction_time: float = 1.0, deceleration: float = 3.05,
                    grade_pct: float = 0.0, vehicle_length_m: float = 6.0,
                    min_yellow: float = 3.0, max_yellow: float = 6.0,
                    round_to: float = 0.1) -> float:
    """Total intergreen: the ITE change interval plus the all-red clearance."""
    y = yellow_change_interval(speed_ms, reaction_time, deceleration, grade_pct)
    lo = _nonneg(min_yellow, "min_yellow")
    hi = _positive(max_yellow, "max_yellow")
    if hi < lo:
        raise SignalControlError("max_yellow must be >= min_yellow.")
    y = _clamp(y, lo, hi)
    r = all_red_clearance(crossing_width_m, speed_ms, vehicle_length_m)
    total = y + r
    q = _positive(round_to, "round_to")
    return math.ceil(total / q - 1e-9) * q


def pedestrian_crossing_time(crossing_length_m: float, walk_speed: float = 1.2,
                             start_up: float = 3.2, ped_per_cycle: float = 0.0,
                             crossing_width_m: float = 3.0) -> float:
    """HCM pedestrian minimum green ``Gp = 3.2 + L/Sp + platoon term`` (seconds)."""
    L = _nonneg(crossing_length_m, "crossing_length_m")
    sp = _positive(walk_speed, "walk_speed")
    su = _nonneg(start_up, "start_up")
    n = _nonneg(ped_per_cycle, "ped_per_cycle")
    w = _positive(crossing_width_m, "crossing_width_m")
    if w > 3.0:
        platoon = 2.7 * n / w
    else:
        platoon = 0.27 * n
    return su + L / sp + platoon


def minimum_green_time(queue_veh: float = 0.0, sat_flow: float = _S0_DEFAULT,
                       driver_min: float = 7.0, start_up_lost: float = 2.0,
                       ped_requirement: Optional[float] = None,
                       intergreen: float = 0.0) -> float:
    """Minimum green: the largest of the driver, queue-clearance and ped minima."""
    q = _nonneg(queue_veh, "queue_veh")
    s = _positive(sat_flow, "sat_flow")
    dmin = _nonneg(driver_min, "driver_min")
    su = _nonneg(start_up_lost, "start_up_lost")
    ig = _nonneg(intergreen, "intergreen")
    clearance = su + q * 3600.0 / s
    need = max(dmin, clearance)
    if ped_requirement is not None:
        ped = _nonneg(ped_requirement, "ped_requirement")
        need = max(need, ped - ig)          # ped clearance may run into intergreen
    return need


def stage_lost_time(intergreen: float, start_lost: float = 2.0,
                    end_gain: float = 3.0) -> float:
    """Lost time of one stage ``l = l1 + (I - e)`` for intergreen I and end gain e."""
    ig = _nonneg(intergreen, "intergreen")
    l1 = _nonneg(start_lost, "start_lost")
    e = _nonneg(end_gain, "end_gain")
    l = l1 + ig - e
    if l < 0.0:
        raise SignalControlError(
            "negative stage lost time (l1=%g, I=%g, e=%g): the end gain cannot "
            "exceed the intergreen plus the start-up loss." % (l1, ig, e))
    return l


def total_lost_time(intergreens: Sequence[float], start_lost: float = 2.0,
                    end_gain: float = 3.0) -> float:
    """Total lost time per cycle, summed over the stages."""
    ig = _as_floats(intergreens, "intergreens")
    _check_len(ig, 1, "intergreens")
    return math.fsum(stage_lost_time(v, start_lost, end_gain) for v in ig)


# ===========================================================================
# 6. THE SIGNAL PLAN CONTAINER
# ===========================================================================
@dataclass
class SignalPlan:
    """A fixed-time plan: stages, displayed greens, intergreens and lost times."""

    stages: List[List[int]]
    greens: List[float]
    intergreens: List[float]
    lost_times: List[float]
    offset: float = 0.0
    name: str = "plan"

    def __post_init__(self) -> None:
        self.stages = [sorted(set(_as_ints(s, "stage"))) for s in self.stages]
        self.greens = _as_floats(self.greens, "greens")
        self.intergreens = _as_floats(self.intergreens, "intergreens")
        self.lost_times = _as_floats(self.lost_times, "lost_times")
        n = len(self.stages)
        if n == 0:
            raise SignalControlError("a plan needs at least one stage.")
        for name, seq in (("greens", self.greens),
                          ("intergreens", self.intergreens),
                          ("lost_times", self.lost_times)):
            if len(seq) != n:
                raise SignalControlError(
                    "%s has %d entries but there are %d stages."
                    % (name, len(seq), n))
            for v in seq:
                if v < 0.0:
                    raise SignalControlError("%s must be >= 0." % name)
        self.offset = _finite(self.offset, "offset")

    @property
    def n_stages(self) -> int:
        """Number of stages in the plan."""
        return len(self.stages)

    @property
    def cycle(self) -> float:
        """Cycle time ``C = sum(G_i + I_i)`` in seconds."""
        return math.fsum(self.greens) + math.fsum(self.intergreens)

    @property
    def total_lost_time(self) -> float:
        """Total lost time ``L`` per cycle."""
        return math.fsum(self.lost_times)

    def effective_greens(self) -> List[float]:
        """Effective green per stage ``g_i = G_i + I_i - l_i``."""
        return [self.greens[i] + self.intergreens[i] - self.lost_times[i]
                for i in range(self.n_stages)]

    def green_shares(self) -> List[float]:
        """Effective green ratios ``lambda_i = g_i / C``."""
        C = self.cycle
        if C <= _EPS:
            raise SignalControlError("plan has a zero cycle time.")
        return [g / C for g in self.effective_greens()]

    def movement_green(self, movement: int) -> float:
        """Total effective green served to one movement across all its stages."""
        m = int(movement)
        g = self.effective_greens()
        return math.fsum(g[k] for k in range(self.n_stages)
                         if m in self.stages[k])

    def movements(self) -> List[int]:
        """Every movement index served by the plan, sorted."""
        out = set()
        for st in self.stages:
            out.update(st)
        return sorted(out)

    def capacities(self, sat_flows: Sequence[float]) -> Dict[int, float]:
        """Capacity (veh/h) of every served movement under this plan."""
        s = _as_floats(sat_flows, "sat_flows")
        C = self.cycle
        return dict((m, movement_capacity(s[m], self.movement_green(m), C))
                    for m in self.movements())

    def validate(self, tol: float = 1e-9,
                 conflict: Optional[BoolMatrix] = None,
                 min_green: float = 0.0) -> List[str]:
        """Every rule this plan breaks; an empty list means the plan is legal."""
        problems: List[str] = []
        C = self.cycle
        L = self.total_lost_time
        gsum = math.fsum(self.effective_greens())
        if abs(gsum - (C - L)) > tol:
            problems.append(
                "effective greens sum to %.12f s, not C - L = %.12f s"
                % (gsum, C - L))
        for i, g in enumerate(self.effective_greens()):
            if g < -tol:
                problems.append("stage %d has a negative effective green (%.3f)"
                                % (i, g))
            elif g + tol < min_green:
                problems.append("stage %d green %.3f s is below the %.3f s minimum"
                                % (i, g, min_green))
        if L >= C:
            problems.append("lost time %.3f s is not below the %.3f s cycle"
                            % (L, C))
        if conflict is not None:
            problems.extend(validate_stages(self.stages, conflict,
                                            require_cover=False))
        return problems

    def as_dict(self) -> Dict[str, Any]:
        """A plain JSON-able view of the plan."""
        return {"name": self.name, "cycle": self.cycle, "offset": self.offset,
                "stages": [list(s) for s in self.stages],
                "displayedGreens": list(self.greens),
                "intergreens": list(self.intergreens),
                "lostTimes": list(self.lost_times),
                "effectiveGreens": self.effective_greens(),
                "greenShares": self.green_shares(),
                "totalLostTime": self.total_lost_time}


def make_signal_plan(stages: Sequence[Sequence[int]],
                     effective_greens: Sequence[float],
                     intergreens: Sequence[float],
                     start_lost: float = 2.0, end_gain: float = 3.0,
                     lost_times: Optional[Sequence[float]] = None,
                     offset: float = 0.0, name: str = "plan") -> SignalPlan:
    """Build a validated plan from *effective* greens and intergreens."""
    g = _as_floats(effective_greens, "effective_greens")
    ig = _as_floats(intergreens, "intergreens")
    _same_len(g, ig, "effective_greens", "intergreens")
    if len(stages) != len(g):
        raise SignalControlError("stages and greens must have equal length.")
    if lost_times is None:
        lt = [stage_lost_time(v, start_lost, end_gain) for v in ig]
    else:
        lt = _as_floats(lost_times, "lost_times")
        _same_len(lt, g, "lost_times", "effective_greens")
    displayed = [g[i] - ig[i] + lt[i] for i in range(len(g))]
    for i, d in enumerate(displayed):
        if d < -1e-9:
            raise SignalControlError(
                "stage %d needs a negative displayed green (%.3f s): the "
                "effective green %.3f s is smaller than its lost time."
                % (i, d, g[i]))
    displayed = [max(0.0, d) for d in displayed]
    plan = SignalPlan(stages=[list(s) for s in stages], greens=displayed,
                      intergreens=ig, lost_times=lt, offset=offset, name=name)
    problems = plan.validate()
    if problems:
        raise SignalControlError("invalid plan: %s" % problems[0])
    return plan


def plan_delay(plan: SignalPlan, demands: Sequence[float],
               sat_flows: Sequence[float], period_h: float = 0.25,
               k: float = 0.50, upstream_i: float = 1.0,
               arrival_type: int = 3) -> Tuple[Dict[int, DelayResult], float, str]:
    """HCM control delay of every movement in a plan, plus the junction average."""
    q = _as_floats(demands, "demands")
    s = _as_floats(sat_flows, "sat_flows")
    _same_len(q, s, "demands", "sat_flows")
    C = plan.cycle
    out: Dict[int, DelayResult] = {}
    for m in plan.movements():
        if m >= len(q):
            raise SignalControlError("movement %d has no demand entry." % m)
        out[m] = hcm_control_delay(q[m], s[m], plan.movement_green(m), C,
                                   period_h=period_h, k=k,
                                   upstream_i=upstream_i,
                                   arrival_type=arrival_type)
    ms = sorted(out)
    avg = intersection_delay([q[m] for m in ms],
                             [out[m].control_delay for m in ms])
    return out, avg, (level_of_service(avg) if not math.isinf(avg) else "F")


# ===========================================================================
# 7. LINEAR-PROGRAMMING KERNEL (TWO-PHASE SIMPLEX, BLAND'S RULE)
# ===========================================================================
@dataclass
class LPResult:
    """The outcome of a linear program: status, solution vector and objective."""

    status: str
    x: List[float]
    fun: float
    n_iterations: int

    @property
    def success(self) -> bool:
        """True when an optimal vertex was found."""
        return self.status == "optimal"

    def as_dict(self) -> Dict[str, Any]:
        """A plain JSON-able view."""
        return {"status": self.status, "x": list(self.x), "fun": self.fun,
                "iterations": self.n_iterations}


def _pivot(T: List[List[float]], row: int, col: int) -> None:
    """Gauss-Jordan pivot of a simplex tableau about ``T[row][col]``."""
    piv = T[row][col]
    inv = 1.0 / piv
    T[row] = [v * inv for v in T[row]]
    T[row][col] = 1.0
    for r in range(len(T)):
        if r == row:
            continue
        f = T[r][col]
        if f != 0.0:
            Tr = T[r]
            Trow = T[row]
            for j in range(len(Tr)):
                Tr[j] -= f * Trow[j]
            Tr[col] = 0.0


def _simplex_solve(T: List[List[float]], basis: List[int], n_cols: int,
                   tol: float, max_iter: int) -> Tuple[str, int]:
    """Run primal simplex on a canonical tableau; Bland's rule prevents cycling."""
    m = len(T) - 1
    it = 0
    while it < max_iter:
        it += 1
        col = -1
        for j in range(n_cols):                      # Bland: lowest index
            if T[m][j] < -tol:
                col = j
                break
        if col < 0:
            return "optimal", it
        row, best = -1, _INF
        for i in range(m):
            if T[i][col] > tol:
                ratio = T[i][n_cols] / T[i][col]
                if ratio < best - 1e-12 or (abs(ratio - best) <= 1e-12 and
                                            row >= 0 and basis[i] < basis[row]):
                    best, row = ratio, i
        if row < 0:
            return "unbounded", it
        _pivot(T, row, col)
        basis[row] = col
    return "max_iter", it


def linear_program(c: Sequence[float],
                   A_ub: Optional[Sequence[Sequence[float]]] = None,
                   b_ub: Optional[Sequence[float]] = None,
                   A_eq: Optional[Sequence[Sequence[float]]] = None,
                   b_eq: Optional[Sequence[float]] = None,
                   maximise: bool = False, tol: float = 1e-9,
                   max_iter: int = 20000) -> LPResult:
    """Solve ``min c.x`` (or max) s.t. ``A_ub x <= b_ub``, ``A_eq x = b_eq``, ``x >= 0``."""
    cost = _as_floats(c, "c")
    n = len(cost)
    if n == 0:
        raise SignalControlError("c must have at least one entry.")
    rows: List[List[float]] = []
    rhs: List[float] = []
    kinds: List[str] = []
    if A_ub is not None:
        if b_ub is None:
            raise SignalControlError("A_ub given without b_ub.")
        bu = _as_floats(b_ub, "b_ub")
        if len(A_ub) != len(bu):
            raise SignalControlError("A_ub and b_ub must have equal length.")
        for r, bv in zip(A_ub, bu):
            row = _as_floats(r, "A_ub row")
            if len(row) != n:
                raise SignalControlError("A_ub rows must have %d entries." % n)
            rows.append(row)
            rhs.append(bv)
            kinds.append("ub")
    if A_eq is not None:
        if b_eq is None:
            raise SignalControlError("A_eq given without b_eq.")
        be = _as_floats(b_eq, "b_eq")
        if len(A_eq) != len(be):
            raise SignalControlError("A_eq and b_eq must have equal length.")
        for r, bv in zip(A_eq, be):
            row = _as_floats(r, "A_eq row")
            if len(row) != n:
                raise SignalControlError("A_eq rows must have %d entries." % n)
            rows.append(row)
            rhs.append(bv)
            kinds.append("eq")
    m = len(rows)
    if m == 0:
        raise SignalControlError("a linear program needs at least one constraint.")
    n_slack = sum(1 for k in kinds if k == "ub")
    total = n + n_slack + m                        # vars + slacks + artificials
    T: List[List[float]] = []
    slack_at = n
    for i in range(m):
        row = [0.0] * (total + 1)
        sign = -1.0 if rhs[i] < 0.0 else 1.0
        for j in range(n):
            row[j] = sign * rows[i][j]
        if kinds[i] == "ub":
            row[slack_at] = sign * 1.0
            slack_at += 1
        row[n + n_slack + i] = 1.0                 # artificial
        row[total] = sign * rhs[i]
        T.append(row)
    basis = [n + n_slack + i for i in range(m)]
    # ---- phase 1: minimise the sum of the artificials
    obj = [0.0] * (total + 1)
    for j in range(n + n_slack, total):
        obj[j] = 1.0
    T.append(obj)
    for i in range(m):
        f = T[m][basis[i]]
        if f != 0.0:
            for j in range(total + 1):
                T[m][j] -= f * T[i][j]
    status, it1 = _simplex_solve(T, basis, total, tol, max_iter)
    if status == "max_iter":
        return LPResult("max_iter", [0.0] * n, _INF, it1)
    if -T[m][total] > 1e-7:
        return LPResult("infeasible", [0.0] * n, _INF, it1)
    # ---- drive every artificial out of the basis; drop redundant rows
    keep: List[int] = []
    for i in range(m):
        if basis[i] >= n + n_slack:
            piv = -1
            for j in range(n + n_slack):
                if abs(T[i][j]) > tol:
                    piv = j
                    break
            if piv < 0:
                continue                           # redundant constraint
            _pivot(T, i, piv)
            basis[i] = piv
        keep.append(i)
    # ---- phase 2: the real objective, with the artificial columns deleted
    n_cols = n + n_slack
    basis = [basis[i] for i in keep]
    T = [T[i][:n_cols] + [T[i][total]] for i in keep]
    m2 = len(T)
    sgn = -1.0 if maximise else 1.0
    obj2 = [0.0] * (n_cols + 1)
    for j in range(n):
        obj2[j] = sgn * cost[j]
    T.append(obj2)
    for i in range(m2):
        f = T[m2][basis[i]]
        if f != 0.0:
            for j in range(n_cols + 1):
                T[m2][j] -= f * T[i][j]
    status, it2 = _simplex_solve(T, basis, n_cols, tol, max_iter)
    x = [0.0] * n
    for i in range(m2):
        if basis[i] < n:
            x[basis[i]] = T[i][n_cols]
    x = [0.0 if abs(v) < 1e-12 else v for v in x]
    fun = math.fsum(cost[j] * x[j] for j in range(n))
    if status == "unbounded":
        return LPResult("unbounded", x, -_INF if maximise else _INF, it1 + it2)
    if status == "max_iter":
        return LPResult("max_iter", x, fun, it1 + it2)
    return LPResult("optimal", x, fun, it1 + it2)


# ===========================================================================
# 8. FIXED-TIME SPLIT OPTIMISATION
# ===========================================================================
def apportion_green(total_green: float, weights: Sequence[float],
                    min_green: float = 0.0,
                    max_green: Optional[float] = None) -> List[float]:
    """Split ``total_green`` in proportion to ``weights`` under green bounds."""
    T = _nonneg(total_green, "total_green")
    w = _as_floats(weights, "weights")
    _check_len(w, 1, "weights")
    for v in w:
        if v < 0.0:
            raise SignalControlError("weights must be >= 0.")
    n = len(w)
    lo = _nonneg(min_green, "min_green")
    hi = _INF if max_green is None else _positive(max_green, "max_green")
    if hi < lo:
        raise SignalControlError("max_green must be >= min_green.")
    if n * lo > T + 1e-9:
        raise SignalControlError(
            "%d stages at a %.2f s minimum need %.2f s but only %.2f s is "
            "available." % (n, lo, n * lo, T))
    if not math.isinf(hi) and n * hi < T - 1e-9:
        raise SignalControlError(
            "%d stages capped at %.2f s cannot absorb %.2f s." % (n, hi, T))
    g = [0.0] * n
    fixed = [False] * n
    while True:
        free = [i for i in range(n) if not fixed[i]]
        if not free:
            break
        avail = T - math.fsum(g[i] for i in range(n) if fixed[i])
        W = math.fsum(w[i] for i in free)
        if W <= _EPS:
            share = [avail / len(free)] * len(free)
        else:
            share = [avail * w[i] / W for i in free]
        # the last free stage absorbs the rounding so the sum stays exact
        share[-1] = avail - math.fsum(share[:-1])
        clash = -1
        for pos, i in enumerate(free):
            if share[pos] < lo - 1e-12:
                clash = i
                g[i] = lo
                break
            if share[pos] > hi + 1e-12:
                clash = i
                g[i] = hi
                break
        if clash < 0:
            for pos, i in enumerate(free):
                g[i] = share[pos]
            break
        fixed[clash] = True
    resid = T - math.fsum(g)
    if abs(resid) > 1e-9:
        raise SignalControlError("green apportionment failed to close (%.3e s)."
                                 % resid)
    return g


def equisaturation_split(cycle: float, lost_time: float,
                         critical_y: Sequence[float], min_green: float = 0.0,
                         max_green: Optional[float] = None) -> List[float]:
    """Effective greens proportional to the critical flow ratios (equal saturation)."""
    C = _positive(cycle, "cycle")
    L = _nonneg(lost_time, "lost_time")
    y = _as_floats(critical_y, "critical_y")
    _check_len(y, 1, "critical_y")
    for v in y:
        if v < 0.0:
            raise SignalControlError("flow ratios must be >= 0.")
    if L >= C:
        raise SignalControlError("lost time %.2f s is not below the %.2f s cycle."
                                 % (L, C))
    return apportion_green(C - L, y, min_green, max_green)


def webster_split(cycle: float, lost_time: float, critical_y: Sequence[float],
                  min_green: float = 0.0,
                  max_green: Optional[float] = None) -> List[float]:
    """Webster's split: ``g_i = (C - L) y_i / Y`` -- identical to equisaturation."""
    return equisaturation_split(cycle, lost_time, critical_y, min_green,
                                max_green)


def allsop_lp_split(cycle: float, lost_time: float, critical_y: Sequence[float],
                    min_green: float = 0.0) -> Tuple[List[float], float]:
    """Allsop's LP: maximise the common flow multiplier; returns (greens, mu).

    A junction with no demand at all has an unbounded multiplier -- every
    movement is satisfied by every split -- so that degenerate case is answered
    directly with the equal (equisaturation) split and ``mu = inf``, the same
    way :func:`max_pressure_stability_margin` answers its own zero-flow case.
    Raising the simplex's ``'unbounded'`` status there would read like an
    internal solver crash rather than the mathematical fact it is.
    """
    C = _positive(cycle, "cycle")
    L = _nonneg(lost_time, "lost_time")
    y = _as_floats(critical_y, "critical_y")
    _check_len(y, 1, "critical_y")
    for v in y:
        if v < 0.0:
            raise SignalControlError("flow ratios must be >= 0.")
    if L >= C:
        raise SignalControlError("lost time %.2f s is not below the %.2f s cycle."
                                 % (L, C))
    lo = _nonneg(min_green, "min_green")
    n = len(y)
    avail = C - L - n * lo
    if avail < -1e-9:
        raise SignalControlError(
            "minimum greens (%d x %.2f s) exceed the available green %.2f s."
            % (n, lo, C - L))
    if max(y) <= _EPS:
        return equisaturation_split(C, L, y, lo), _INF
    # variables z_0..z_{n-1} (green above the minimum) and mu
    cost = [0.0] * n + [-1.0]                     # maximise mu
    A_ub, b_ub = [], []
    for i in range(n):
        row = [0.0] * (n + 1)
        row[i] = -1.0
        row[n] = y[i] * C
        A_ub.append(row)
        b_ub.append(lo)
    A_eq = [[1.0] * n + [0.0]]
    b_eq = [max(avail, 0.0)]
    res = linear_program(cost, A_ub, b_ub, A_eq, b_eq)
    if not res.success:
        raise SignalControlError("Allsop LP failed with status %r." % res.status)
    greens = [lo + res.x[i] for i in range(n)]
    resid = (C - L) - math.fsum(greens)
    if abs(resid) > 1e-6:
        raise SignalControlError("Allsop LP greens do not close (%.3e s)." % resid)
    greens[-1] += resid
    return greens, res.x[n]


def smith_p0_pressure(stage: Sequence[int], demands: Sequence[float],
                      sat_flows: Sequence[float], green: float,
                      cycle: float) -> float:
    """Smith's P0 stage pressure ``P = sum_j s_j X_j^2`` (veh/h).

    Every saturation flow must be strictly positive, the same contract
    :func:`critical_flow_ratios` enforces: ``X = q / (s g / C)`` is undefined
    for a movement with no capacity at all.
    """
    q = _as_floats(demands, "demands")
    s = _as_floats(sat_flows, "sat_flows")
    _same_len(q, s, "demands", "sat_flows")
    for v in s:
        _positive(v, "sat_flow")
    C = _positive(cycle, "cycle")
    g = _nonneg(green, "green")
    st = _as_ints(stage, "stage")
    if g <= _EPS:
        return _INF if any(q[m] > 0.0 for m in st) else 0.0
    lam = g / C
    total = 0.0
    for m in st:
        cap = s[m] * lam
        if cap <= _EPS:
            return _INF if q[m] > 0.0 else total
        x = q[m] / cap
        total += s[m] * x * x
    return total


def smith_p0_split(cycle: float, lost_time: float,
                   stages: Sequence[Sequence[int]], demands: Sequence[float],
                   sat_flows: Sequence[float], min_green: float = 0.0
                   ) -> List[float]:
    """Smith's P0 fixed point: greens that equalise the P0 pressures across stages.

    Saturation flows must be strictly positive.  Smith's stage weight
    ``sqrt(sum_j q_j^2 / s_j)`` diverges as ``s_j -> 0``, so a single
    zero-capacity movement would take the entire cycle and starve every real
    movement at the junction -- and it does so for genuinely mathematical
    reasons, not because of a floor on the divisor, which is why the weight is
    rejected outright rather than clamped.
    """
    C = _positive(cycle, "cycle")
    L = _nonneg(lost_time, "lost_time")
    if L >= C:
        raise SignalControlError("lost time %.2f s is not below the %.2f s cycle."
                                 % (L, C))
    q = _as_floats(demands, "demands")
    s = _as_floats(sat_flows, "sat_flows")
    _same_len(q, s, "demands", "sat_flows")
    for v in s:
        _positive(v, "sat_flow")
    weights = []
    for st in stages:
        ids = _as_ints(st, "stage")
        if not ids:
            raise SignalControlError("a stage is empty.")
        a = math.fsum(q[m] * q[m] / s[m] for m in ids)
        weights.append(math.sqrt(max(a, 0.0)))
    return apportion_green(C - L, weights, min_green)


def delay_minimising_split(cycle: float, lost_time: float,
                           stages: Sequence[Sequence[int]],
                           demands: Sequence[float], sat_flows: Sequence[float],
                           min_green: float = 5.0, model: str = "hcm",
                           period_h: float = 0.25, tol: float = 1e-4
                           ) -> Tuple[List[float], float]:
    """Direct search for the split minimising total delay; returns (greens, delay)."""
    C = _positive(cycle, "cycle")
    L = _nonneg(lost_time, "lost_time")
    q = _as_floats(demands, "demands")
    s = _as_floats(sat_flows, "sat_flows")
    _same_len(q, s, "demands", "sat_flows")
    n = len(stages)
    if n < 1:
        raise SignalControlError("at least one stage is required.")
    lo = _nonneg(min_green, "min_green")
    avail = C - L
    if avail <= 0.0:
        raise SignalControlError("no green available (C = %.2f, L = %.2f)." % (C, L))
    if n * lo > avail:
        lo = avail / (n + 1.0)

    def total_delay(greens: Sequence[float]) -> float:
        acc = 0.0
        for k, st in enumerate(stages):
            for m in _as_ints(st, "stage"):
                d = _delay_model(model, q[m], s[m], greens[k], C, period_h)
                if math.isinf(d):
                    return _INF
                acc += q[m] * d
        return acc

    ycrit = critical_flow_ratios(stages, q, s)
    if math.fsum(ycrit) <= _EPS:
        g = apportion_green(avail, [1.0] * n, lo)
    else:
        g = apportion_green(avail, ycrit, lo)
    best = total_delay(g)
    step = max(avail / 8.0, tol)
    while step > tol:
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if g[i] - step < lo - 1e-12:
                    continue
                trial = list(g)
                trial[i] -= step
                trial[j] += step
                val = total_delay(trial)
                if val < best - 1e-12:
                    g, best = trial, val
                    improved = True
        if not improved:
            step *= 0.5
    resid = avail - math.fsum(g)
    if abs(resid) > 1e-9:
        g[max(range(n), key=lambda i: g[i])] += resid
    return g, best


def _delay_model(model: str, demand: float, sat_flow: float, green: float,
                 cycle: float, period_h: float = 0.25) -> float:
    """Dispatch to one of the delay families by name."""
    key = str(model).lower()
    if key == "webster":
        return webster_delay(demand, sat_flow, green, cycle)
    if key == "akcelik":
        return akcelik_delay(demand, sat_flow, green, cycle, period_h)
    if key == "hcm":
        return hcm_control_delay(demand, sat_flow, green, cycle,
                                 period_h=period_h).control_delay
    if key == "uniform":
        cap = movement_capacity(sat_flow, green, cycle)
        return hcm_uniform_delay(cycle, green, degree_of_saturation(demand, cap))
    raise SignalControlError(
        "unknown delay model %r (use webster / hcm / akcelik / uniform)." % model)


def design_fixed_time_plan(demands: Sequence[float], sat_flows: Sequence[float],
                           stages: Optional[Sequence[Sequence[int]]] = None,
                           conflict: Optional[BoolMatrix] = None,
                           intergreens: Any = 5.0, start_lost: float = 2.0,
                           end_gain: float = 3.0, method: str = "webster",
                           cycle: Optional[float] = None, min_green: float = 7.0,
                           min_cycle: float = 30.0, max_cycle: float = 180.0,
                           period_h: float = 0.25,
                           name: str = "fixed-time") -> SignalPlan:
    """End-to-end fixed-time design: stages -> cycle -> split -> validated plan."""
    q = _as_floats(demands, "demands")
    s = _as_floats(sat_flows, "sat_flows")
    _same_len(q, s, "demands", "sat_flows")
    if stages is None:
        if conflict is None:
            raise SignalControlError("give either stages or a conflict matrix.")
        cands = maximal_compatible_stages(conflict)
        # among the minimum-cardinality covers, take the one with the smallest
        # sum of critical flow ratios -- that is the one with the most capacity
        stages = minimum_stage_cover(cands, len(conflict),
                                     stage_costs=critical_flow_ratios(cands, q, s))
    stages = [sorted(set(_as_ints(st, "stage"))) for st in stages]
    if not stages:
        raise SignalControlError("no stages to time.")
    n = len(stages)
    if isinstance(intergreens, (int, float)):
        ig = [float(intergreens)] * n
    else:
        ig = _as_floats(intergreens, "intergreens")
        if len(ig) != n:
            raise SignalControlError("intergreens must have one entry per stage.")
    L = total_lost_time(ig, start_lost, end_gain)
    ycrit = critical_flow_ratios(stages, q, s)
    if cycle is None:
        C = webster_optimal_cycle(L, ycrit)
        C = _clamp(C, _positive(min_cycle, "min_cycle"),
                   _positive(max_cycle, "max_cycle"))
    else:
        C = _positive(cycle, "cycle")
    if C <= L:
        raise SignalControlError("cycle %.2f s does not exceed the %.2f s lost "
                                 "time." % (C, L))
    key = str(method).lower()
    if key in ("webster", "equisaturation"):
        greens = equisaturation_split(C, L, ycrit, min_green)
    elif key == "allsop":
        greens = allsop_lp_split(C, L, ycrit, min_green)[0]
    elif key in ("smith", "smith_p0", "p0"):
        greens = smith_p0_split(C, L, stages, q, s, min_green)
    elif key in ("delay", "min_delay"):
        greens = delay_minimising_split(C, L, stages, q, s, min_green,
                                        period_h=period_h)[0]
    else:
        raise SignalControlError(
            "unknown method %r (webster / allsop / smith_p0 / delay)." % method)
    return make_signal_plan(stages, greens, ig, start_lost=start_lost,
                            end_gain=end_gain, name=name)


# ===========================================================================
# 9. PLATOON DISPERSION & OFFSET OPTIMISATION (TRANSYT)
# ===========================================================================
@dataclass
class PlatoonProfile:
    """A cyclic flow profile: the flow (veh/h) in each step of one cycle."""

    flows: List[float]
    step: float = 1.0

    def __post_init__(self) -> None:
        self.flows = _as_floats(self.flows, "flows")
        _check_len(self.flows, 1, "flows")
        for v in self.flows:
            if v < 0.0:
                raise SignalControlError("profile flows must be >= 0.")
        self.step = _positive(self.step, "step")

    @property
    def n_steps(self) -> int:
        """Number of steps in one cycle."""
        return len(self.flows)

    @property
    def cycle(self) -> float:
        """Cycle length covered by the profile (seconds)."""
        return self.n_steps * self.step

    def vehicles(self) -> float:
        """Vehicles arriving in one cycle."""
        return math.fsum(v * self.step / 3600.0 for v in self.flows)

    def mean_flow(self) -> float:
        """Cycle-average flow (veh/h)."""
        return math.fsum(self.flows) / self.n_steps

    def shifted(self, steps: int) -> "PlatoonProfile":
        """The same profile rotated later on the clock by ``steps`` steps."""
        n = self.n_steps
        k = int(steps) % n
        return PlatoonProfile([self.flows[(i - k) % n] for i in range(n)],
                              self.step)

    def as_dict(self) -> Dict[str, Any]:
        """A plain JSON-able view."""
        return {"flows": list(self.flows), "step": self.step,
                "cycle": self.cycle, "vehiclesPerCycle": self.vehicles()}


def cyclic_flow_profile(demand: float, cycle: float, step: float = 1.0,
                        platoon_start: Optional[float] = None,
                        platoon_length: Optional[float] = None,
                        platoon_share: float = 1.0) -> PlatoonProfile:
    """Build an arrival profile: a platoon window plus a uniform background."""
    q = _nonneg(demand, "demand")
    C = _positive(cycle, "cycle")
    h = _positive(step, "step")
    n = int(round(C / h))
    if n < 1:
        raise SignalControlError("cycle %g s is shorter than the %g s step."
                                 % (C, h))
    share = _fraction(platoon_share, "platoon_share")
    if platoon_start is None or platoon_length is None:
        return PlatoonProfile([q] * n, h)
    ps = _nonneg(platoon_start, "platoon_start")
    pl = _positive(platoon_length, "platoon_length")
    if pl > C:
        raise SignalControlError("platoon_length exceeds the cycle.")
    i0 = int(round(ps / h)) % n
    k = max(1, int(round(pl / h)))
    flows = [q * (1.0 - share)] * n
    if k > 0:
        rate = q * share * n / k
        for j in range(k):
            flows[(i0 + j) % n] += rate
    return PlatoonProfile(flows, h)


def robertson_dispersion(profile: PlatoonProfile, travel_time: float,
                         alpha: float = 0.35, beta: float = 0.80,
                         n_cycles: int = 12) -> PlatoonProfile:
    """Robertson's platoon-dispersion recursion, run to a cyclic steady state."""
    if not isinstance(profile, PlatoonProfile):
        raise SignalControlError("profile must be a PlatoonProfile.")
    t = _nonneg(travel_time, "travel_time")
    a = _positive(alpha, "alpha")
    b = _positive(beta, "beta")
    if b > 1.0:
        raise SignalControlError("beta must lie in (0, 1] (got %g)." % b)
    h = profile.step
    n = profile.n_steps
    t_steps = t / h
    T = int(round(b * t_steps))
    F = 1.0 / (1.0 + a * b * t_steps)
    if F <= 0.0 or F > 1.0:
        raise SignalControlError("smoothing factor F = %g is out of range." % F)
    out = list(profile.flows)
    reps = max(2, int(n_cycles))
    for _ in range(reps):
        prev = out[-1]
        for i in range(n):
            out[i] = F * profile.flows[(i - T) % n] + (1.0 - F) * prev
            prev = out[i]
    return PlatoonProfile(out, h)


def departure_profile(arrivals: PlatoonProfile, green_start: float,
                      green_length: float, sat_flow: float,
                      initial_queue: float = 0.0, n_cycles: int = 20
                      ) -> Tuple[PlatoonProfile, List[float]]:
    """Discharge an arrival profile at a stopline; returns (departures, queues)."""
    if not isinstance(arrivals, PlatoonProfile):
        raise SignalControlError("arrivals must be a PlatoonProfile.")
    C = arrivals.cycle
    h = arrivals.step
    n = arrivals.n_steps
    gs = _nonneg(green_start, "green_start") % C
    gl = _nonneg(green_length, "green_length")
    if gl > C:
        raise SignalControlError("green_length exceeds the cycle.")
    s = _nonneg(sat_flow, "sat_flow")
    queue = _nonneg(initial_queue, "initial_queue")
    is_green = []
    for i in range(n):
        mid = (i + 0.5) * h
        is_green.append(((mid - gs) % C) < gl)
    cap_step = s * h / 3600.0
    dep = [0.0] * n
    queues = [0.0] * n
    for c in range(max(1, int(n_cycles))):
        for i in range(n):
            queue += arrivals.flows[i] * h / 3600.0
            served = min(queue, cap_step) if is_green[i] else 0.0
            queue -= served
            if c == max(1, int(n_cycles)) - 1:
                dep[i] = served * 3600.0 / h
                queues[i] = queue
    return PlatoonProfile(dep, h), queues


def profile_delay(arrivals: PlatoonProfile, green_start: float,
                  green_length: float, sat_flow: float,
                  initial_queue: float = 0.0, n_cycles: int = 20) -> float:
    """Uniform delay from a cyclic flow profile (veh-seconds per cycle)."""
    _, queues = departure_profile(arrivals, green_start, green_length, sat_flow,
                                  initial_queue, n_cycles)
    return math.fsum(q * arrivals.step for q in queues)


def link_delay_table(upstream: PlatoonProfile, travel_time: float,
                     green_length: float, sat_flow: float,
                     alpha: float = 0.35, beta: float = 0.80) -> List[float]:
    """Delay at the downstream stopline for every whole-step relative offset."""
    dispersed = robertson_dispersion(upstream, travel_time, alpha, beta)
    n = dispersed.n_steps
    h = dispersed.step
    return [profile_delay(dispersed, k * h, green_length, sat_flow)
            for k in range(n)]


@dataclass
class OffsetResult:
    """The outcome of an offset optimisation along an arterial."""

    offsets: List[float]
    total_delay: float
    per_signal_delay: List[float]
    n_evaluations: int
    method: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """A plain JSON-able view."""
        return {"offsets": list(self.offsets), "totalDelay": self.total_delay,
                "perSignalDelay": list(self.per_signal_delay),
                "evaluations": self.n_evaluations, "method": self.method}


def optimise_offsets_dp(tables: Sequence[Sequence[float]], step: float = 1.0
                        ) -> OffsetResult:
    """Exact DP over offsets on a chain: minimise the sum of link delay tables."""
    tabs = [_as_floats(t, "delay table") for t in tables]
    _check_len(tabs, 1, "tables")
    n = len(tabs[0])
    for t in tabs:
        if len(t) != n:
            raise SignalControlError("all delay tables must have %d entries." % n)
    h = _positive(step, "step")
    k = len(tabs) + 1                                  # number of signals
    best = [0.0] + [_INF] * (n - 1)                    # signal 0 offset fixed at 0
    parent: List[List[int]] = []
    evals = 0
    for i in range(k - 1):
        nxt = [_INF] * n
        par = [0] * n
        for b in range(n):
            for a in range(n):
                if math.isinf(best[a]):
                    continue
                val = best[a] + tabs[i][(b - a) % n]
                evals += 1
                if val < nxt[b] - 1e-12:
                    nxt[b], par[b] = val, a
        best, _ = nxt, None
        parent.append(par)
    end = min(range(n), key=lambda b: (best[b], b))
    offsets_idx = [0] * k
    offsets_idx[k - 1] = end
    for i in range(k - 2, -1, -1):
        offsets_idx[i] = parent[i][offsets_idx[i + 1]]
    per = [tabs[i][(offsets_idx[i + 1] - offsets_idx[i]) % n]
           for i in range(k - 1)]
    return OffsetResult(offsets=[idx * h for idx in offsets_idx],
                        total_delay=math.fsum(per), per_signal_delay=per,
                        n_evaluations=evals, method="dp")


def optimise_offsets(demands: Sequence[float], sat_flows: Sequence[float],
                     greens: Sequence[float], travel_times: Sequence[float],
                     cycle: float, step: float = 1.0, alpha: float = 0.35,
                     beta: float = 0.80,
                     initial_offsets: Optional[Sequence[float]] = None,
                     entry_profile: Optional[PlatoonProfile] = None,
                     max_rounds: int = 40) -> OffsetResult:
    """TRANSYT-style hill climb on the offsets of an arterial of signals."""
    q = _as_floats(demands, "demands")
    s = _as_floats(sat_flows, "sat_flows")
    g = _as_floats(greens, "greens")
    tt = _as_floats(travel_times, "travel_times")
    C = _positive(cycle, "cycle")
    h = _positive(step, "step")
    k = len(q)
    if k < 1:
        raise SignalControlError("at least one signal is required.")
    _same_len(q, s, "demands", "sat_flows")
    _same_len(q, g, "demands", "greens")
    if len(tt) != k - 1:
        raise SignalControlError("travel_times needs %d entries (got %d)."
                                 % (k - 1, len(tt)))
    n = int(round(C / h))
    if n < 2:
        raise SignalControlError("cycle %g s is too short for a %g s step."
                                 % (C, h))
    entry = entry_profile if entry_profile is not None else \
        cyclic_flow_profile(q[0], C, h)
    if entry.n_steps != n:
        raise SignalControlError("entry_profile has the wrong number of steps.")
    state = {"evals": 0}

    def evaluate(offsets_idx: Sequence[int]) -> Tuple[float, List[float]]:
        state["evals"] += 1
        arrivals = entry
        per: List[float] = []
        for i in range(k):
            d = profile_delay(arrivals, offsets_idx[i] * h, g[i], s[i])
            per.append(d)
            if i < k - 1:
                dep, _ = departure_profile(arrivals, offsets_idx[i] * h, g[i],
                                           s[i])
                arrivals = robertson_dispersion(dep, tt[i], alpha, beta)
        return math.fsum(per), per

    if initial_offsets is None:
        idx = [0] * k
        acc = 0.0
        for i in range(1, k):
            acc += tt[i - 1]
            idx[i] = int(round((acc % C) / h)) % n
    else:
        off = _as_floats(initial_offsets, "initial_offsets")
        if len(off) != k:
            raise SignalControlError("initial_offsets needs %d entries." % k)
        idx = [int(round((o % C) / h)) % n for o in off]
    best_val, best_per = evaluate(idx)
    stride = max(1, n // 8)
    rounds = 0
    while stride >= 1 and rounds < int(max_rounds):
        rounds += 1
        improved = False
        for i in range(1, k):                    # signal 0 anchors the clock
            for d in (stride, -stride):
                trial = list(idx)
                trial[i] = (trial[i] + d) % n
                val, per = evaluate(trial)
                if val < best_val - 1e-9:
                    idx, best_val, best_per = trial, val, per
                    improved = True
        if not improved:
            if stride == 1:
                break
            stride = max(1, stride // 2)
    return OffsetResult(offsets=[i * h for i in idx], total_delay=best_val,
                        per_signal_delay=best_per,
                        n_evaluations=state["evals"], method="hill-climb")


# ===========================================================================
# 10. BANDWIDTH: GREEN WAVES, MAXBAND & MULTIBAND
# ===========================================================================
def _band_from_arcs(starts: Sequence[float], lengths: Sequence[float],
                    cycle: float) -> Tuple[float, float]:
    """Longest common arc of a set of circular intervals: (band, band start)."""
    n = len(starts)
    if n == 0:
        return 0.0, 0.0
    if all(L >= cycle - 1e-12 for L in lengths):
        return cycle, 0.0
    best_b, best_s = 0.0, 0.0
    for a in starts:
        cov = cycle
        ok = True
        for i in range(n):
            off = (a - starts[i]) % cycle
            if off >= lengths[i] - 1e-12:
                ok = False
                break
            cov = min(cov, lengths[i] - off)
        if ok and cov > best_b + 1e-12:
            best_b, best_s = cov, a % cycle
    return best_b, best_s


def arterial_bandwidth(green_starts: Sequence[float],
                       green_lengths: Sequence[float],
                       travel_times: Sequence[float], cycle: float,
                       direction: str = "outbound") -> Tuple[float, float]:
    """Exact through band (seconds) and its start time, by arc intersection."""
    gs = _as_floats(green_starts, "green_starts")
    gl = _as_floats(green_lengths, "green_lengths")
    tt = _as_floats(travel_times, "travel_times")
    C = _positive(cycle, "cycle")
    _same_len(gs, gl, "green_starts", "green_lengths")
    k = len(gs)
    if len(tt) != k - 1:
        raise SignalControlError("travel_times needs %d entries (got %d)."
                                 % (k - 1, len(tt)))
    for L in gl:
        if L < 0.0 or L > C:
            raise SignalControlError("green lengths must lie in [0, C].")
    d = str(direction).lower()
    if d not in ("outbound", "inbound"):
        raise SignalControlError("direction must be 'outbound' or 'inbound'.")
    if d == "outbound":
        cum = [0.0]
        for t in tt:
            cum.append(cum[-1] + _nonneg(t, "travel time"))
        arcs = [(gs[i] - cum[i]) % C for i in range(k)]
    else:
        # the band runs from the far end back to signal 0
        cum = [0.0] * k
        for i in range(k - 2, -1, -1):
            cum[i] = cum[i + 1] + _nonneg(tt[i], "travel time")
        arcs = [(gs[i] - cum[i]) % C for i in range(k)]
    return _band_from_arcs(arcs, gl, C)


def green_wave_offsets(distances: Sequence[float], speed_ms: float,
                       cycle: float) -> List[float]:
    """Simple progression offsets ``theta_i = (sum d / v) mod C`` (seconds)."""
    d = _as_floats(distances, "distances")
    v = _positive(speed_ms, "speed_ms")
    C = _positive(cycle, "cycle")
    for x in d:
        if x < 0.0:
            raise SignalControlError("distances must be >= 0.")
    out = [0.0]
    acc = 0.0
    for x in d:
        acc += x / v
        out.append(acc % C)
    return out


def bandwidth_efficiency(bandwidth: float, cycle: float) -> float:
    """Bandwidth efficiency ``b / C`` -- the share of the cycle that progresses."""
    b = _nonneg(bandwidth, "bandwidth")
    C = _positive(cycle, "cycle")
    return b / C


@dataclass
class BandwidthResult:
    """Offsets and bands from a MAXBAND / MULTIBAND solve.

    CONVENTION -- ``objective`` carries whatever its own solver's literature
    maximises, and the two are NOT in the same units, so never compare one
    solver's ``objective`` with the other's:

    * :func:`maxband` reports Little's ``b + k b_in`` in SECONDS, i.e. the same
      units as ``outbound_band`` and ``inbound_band``.
    * :func:`multiband` reports Gartner's flow-weighted band as a DIMENSIONLESS
      fraction, normalised by ``sum(flows_out) + k sum(flows_in)`` and by the
      cycle.  Because that normaliser itself depends on ``inbound_weight``, a
      MULTIBAND ``objective`` is not monotone in ``inbound_weight`` and is only
      meaningful against another MULTIBAND run at the same weights and flows.

    ``outbound_band`` and ``inbound_band`` are in seconds in both cases and are
    the fields to compare across solvers.
    """

    offsets: List[float]
    outbound_band: float
    inbound_band: float
    objective: float
    loop_integers: List[int]
    cycle: float
    outbound_bands: Optional[List[float]] = None
    inbound_bands: Optional[List[float]] = None

    def efficiency(self) -> float:
        """Outbound efficiency ``b / C``."""
        return bandwidth_efficiency(self.outbound_band, self.cycle)

    def as_dict(self) -> Dict[str, Any]:
        """A plain JSON-able view."""
        out = {"offsets": list(self.offsets),
               "outboundBand": self.outbound_band,
               "inboundBand": self.inbound_band,
               "objective": self.objective, "cycle": self.cycle,
               "loopIntegers": list(self.loop_integers),
               "efficiency": self.efficiency()}
        if self.outbound_bands is not None:
            out["outboundBands"] = list(self.outbound_bands)
        if self.inbound_bands is not None:
            out["inboundBands"] = list(self.inbound_bands)
        return out


def _band_offsets(w: Sequence[float], t: Sequence[float], g: Sequence[float],
                  cycle: float) -> List[float]:
    """Green-start offsets implied by the MAXBAND interference variables."""
    k = len(g)
    theta = [0.0] * k
    for i in range(k - 1):
        theta[i + 1] = (theta[i] + w[i] + t[i] - w[i + 1]) % 1.0
    return [x * cycle for x in theta]


def maxband(green_lengths: Sequence[float], travel_out: Sequence[float],
            travel_in: Sequence[float], cycle: float,
            inbound_weight: float = 1.0, m_range: int = 1) -> BandwidthResult:
    """MAXBAND: maximise ``b + k b_in`` over offsets (integers enumerated, LP inside).

    INHERENT LIMIT of the published formulation, not of this implementation.
    Little's model constrains the interference variables by ``0 <= w_i <= g_i``
    and ``0 <= wb_i <= g_i`` at every signal, i.e. it insists that BOTH bands be
    positionable inside every green -- including a band of width zero.  When the
    true optimum gives one direction no band at all, that empty band still has
    to be placed, and the loop-integer congruence can make the placement
    infeasible.  The solver is therefore exact whenever both bands are strictly
    positive, and may fall short of the unrestricted optimum only in the corner
    where one band is zero.  It never overstates the bandwidth it returns: score
    the returned offsets with :func:`arterial_bandwidth` and you get back the
    bands reported here.  If a one-way band is what you want, optimise that
    direction on its own with ``inbound_weight=0`` and a direct offset search.
    """
    C = _positive(cycle, "cycle")
    gl = _as_floats(green_lengths, "green_lengths")
    to = _as_floats(travel_out, "travel_out")
    ti = _as_floats(travel_in, "travel_in")
    k = len(gl)
    if k < 2:
        raise SignalControlError("MAXBAND needs at least two signals.")
    if len(to) != k - 1 or len(ti) != k - 1:
        raise SignalControlError("travel times need %d entries each." % (k - 1))
    for L in gl:
        if L <= 0.0 or L > C:
            raise SignalControlError("green lengths must lie in (0, C].")
    kw = _nonneg(inbound_weight, "inbound_weight")
    g = [L / C for L in gl]
    t = [x / C for x in to]
    tb = [x / C for x in ti]
    mr = int(m_range)
    if mr < 0:
        raise SignalControlError("m_range must be >= 0.")
    # variables: b, bbar, w_0..w_{k-1}, wb_0..wb_{k-1}
    nv = 2 + 2 * k
    cost = [0.0] * nv
    cost[0] = -1.0
    cost[1] = -kw
    base_ub, base_bu = [], []
    for i in range(k):
        row = [0.0] * nv
        row[0] = 1.0
        row[2 + i] = 1.0
        base_ub.append(row)
        base_bu.append(g[i])
        row2 = [0.0] * nv
        row2[1] = 1.0
        row2[2 + k + i] = 1.0
        base_ub.append(row2)
        base_bu.append(g[i])
    best: Optional[BandwidthResult] = None
    best_obj = -_INF
    centres = [int(round(t[i] + tb[i])) for i in range(k - 1)]
    ranges = [range(centres[i] - mr, centres[i] + mr + 1) for i in range(k - 1)]
    for combo in itertools.product(*ranges):
        A_eq, b_eq = [], []
        for i in range(k - 1):
            row = [0.0] * nv
            row[2 + i] += 1.0
            row[2 + k + i] += 1.0
            row[2 + i + 1] -= 1.0
            row[2 + k + i + 1] -= 1.0
            A_eq.append(row)
            b_eq.append(combo[i] - t[i] - tb[i] + g[i] - g[i + 1])
        res = linear_program(cost, base_ub, base_bu, A_eq, b_eq)
        if not res.success:
            continue
        b, bb = res.x[0], res.x[1]
        obj = b + kw * bb
        if obj <= best_obj + 1e-12:
            continue
        best_obj = obj
        w = res.x[2:2 + k]
        offsets = _band_offsets(w, t, g, C)
        best = BandwidthResult(offsets=offsets, outbound_band=b * C,
                               inbound_band=bb * C, objective=obj * C,
                               loop_integers=list(combo), cycle=C)
    if best is None:
        raise SignalControlError("MAXBAND found no feasible offsets; widen m_range.")
    return best


def multiband(green_lengths: Sequence[float], travel_out: Sequence[float],
              travel_in: Sequence[float], cycle: float,
              flows_out: Optional[Sequence[float]] = None,
              flows_in: Optional[Sequence[float]] = None,
              inbound_weight: float = 1.0, m_range: int = 1) -> BandwidthResult:
    """MULTIBAND: flow-weighted, per-link variable bands (Gartner's generalisation)."""
    C = _positive(cycle, "cycle")
    gl = _as_floats(green_lengths, "green_lengths")
    to = _as_floats(travel_out, "travel_out")
    ti = _as_floats(travel_in, "travel_in")
    k = len(gl)
    if k < 2:
        raise SignalControlError("MULTIBAND needs at least two signals.")
    if len(to) != k - 1 or len(ti) != k - 1:
        raise SignalControlError("travel times need %d entries each." % (k - 1))
    nl = k - 1
    fo = [1.0] * nl if flows_out is None else _as_floats(flows_out, "flows_out")
    fi = [1.0] * nl if flows_in is None else _as_floats(flows_in, "flows_in")
    if len(fo) != nl or len(fi) != nl:
        raise SignalControlError("flow weights need %d entries each." % nl)
    kw = _nonneg(inbound_weight, "inbound_weight")
    g = [L / C for L in gl]
    t = [x / C for x in to]
    tb = [x / C for x in ti]
    # variables: b_0..b_{nl-1}, bb_0..bb_{nl-1}, w_0..w_{k-1}, wb_0..wb_{k-1}
    nv = 2 * nl + 2 * k
    wo = 2 * nl
    wbo = 2 * nl + k
    scale = math.fsum(fo) + kw * math.fsum(fi)
    scale = scale if scale > _EPS else 1.0
    cost = [0.0] * nv
    for i in range(nl):
        cost[i] = -fo[i] / scale
        cost[nl + i] = -kw * fi[i] / scale
    base_ub, base_bu = [], []
    for i in range(nl):
        for sig in (i, i + 1):
            row = [0.0] * nv
            row[i] = 1.0
            row[wo + sig] = 1.0
            base_ub.append(row)
            base_bu.append(g[sig])
            row2 = [0.0] * nv
            row2[nl + i] = 1.0
            row2[wbo + sig] = 1.0
            base_ub.append(row2)
            base_bu.append(g[sig])
    best: Optional[BandwidthResult] = None
    best_obj = -_INF
    centres = [int(round(t[i] + tb[i])) for i in range(nl)]
    mr = int(m_range)
    ranges = [range(centres[i] - mr, centres[i] + mr + 1) for i in range(nl)]
    for combo in itertools.product(*ranges):
        A_eq, b_eq = [], []
        for i in range(nl):
            row = [0.0] * nv
            row[wo + i] += 1.0
            row[wbo + i] += 1.0
            row[wo + i + 1] -= 1.0
            row[wbo + i + 1] -= 1.0
            A_eq.append(row)
            b_eq.append(combo[i] - t[i] - tb[i] + g[i] - g[i + 1])
        res = linear_program(cost, base_ub, base_bu, A_eq, b_eq)
        if not res.success:
            continue
        obj = -res.fun
        if obj <= best_obj + 1e-12:
            continue
        best_obj = obj
        bands = res.x[:nl]
        bands_in = res.x[nl:2 * nl]
        w = res.x[wo:wo + k]
        offsets = _band_offsets(w, t, g, C)
        best = BandwidthResult(offsets=offsets,
                               outbound_band=min(bands) * C,
                               inbound_band=min(bands_in) * C, objective=obj,
                               loop_integers=list(combo), cycle=C,
                               outbound_bands=[b * C for b in bands],
                               inbound_bands=[b * C for b in bands_in])
    if best is None:
        raise SignalControlError("MULTIBAND found no feasible offsets; widen m_range.")
    return best


# ===========================================================================
# 11. ACTUATED CONTROL: GAP-OUT, MAX-OUT, VOLUME DENSITY
# ===========================================================================
@dataclass
class ActuatedResult:
    """How an actuated phase ended, and what it served."""

    green: float
    terminated_by: str
    vehicles_served: int
    extensions: int

    def as_dict(self) -> Dict[str, Any]:
        """A plain JSON-able view."""
        return {"green": self.green, "terminatedBy": self.terminated_by,
                "vehiclesServed": self.vehicles_served,
                "extensions": self.extensions}


def poisson_arrivals(rate_veh_h: float, horizon_s: float, seed: int = 0,
                     min_headway: float = 0.0) -> List[float]:
    """Deterministically-seeded Poisson arrival times (seconds) over a horizon.

    CONVENTION -- with ``min_headway > 0`` the headways are SHIFTED exponential
    (``h = min_headway - ln(u) / lambda``, the Cowan M3 bunched-headway family
    that :func:`brilon_wu_capacity` also assumes), so ``rate_veh_h`` is the rate
    of the underlying exponential and NOT the realised flow.  The realised flow
    is ``3600 / (min_headway + 3600 / rate_veh_h)``: asking for 900 veh/h with a
    2 s minimum headway delivers 600 veh/h.  At ``min_headway = 0`` the two
    coincide.  The stream is a pure function of ``seed`` and is byte-identical
    across runs and platforms.
    """
    rate = _nonneg(rate_veh_h, "rate_veh_h")
    H = _positive(horizon_s, "horizon_s")
    mh = _nonneg(min_headway, "min_headway")
    rng = random.Random(int(seed))
    out: List[float] = []
    if rate <= _EPS:
        return out
    lam = rate / 3600.0
    t = 0.0
    while True:
        u = rng.random()
        while u <= 0.0:
            u = rng.random()
        t += mh - math.log(u) / lam
        if t > H:
            break
        out.append(t)
    return out


def volume_density_gap(elapsed_green: float, max_gap: float = 4.0,
                       min_gap: float = 2.0, time_before_reduction: float = 10.0,
                       time_to_reduce: float = 20.0) -> float:
    """Volume-density unit extension: the allowable gap shrinks as green runs on."""
    e = _nonneg(elapsed_green, "elapsed_green")
    hi = _positive(max_gap, "max_gap")
    lo = _nonneg(min_gap, "min_gap")
    if lo > hi:
        raise SignalControlError("min_gap must not exceed max_gap.")
    tbr = _nonneg(time_before_reduction, "time_before_reduction")
    ttr = _positive(time_to_reduce, "time_to_reduce")
    if e <= tbr:
        return hi
    if e >= tbr + ttr:
        return lo
    return hi - (hi - lo) * (e - tbr) / ttr


def simulate_actuated_phase(arrival_times: Sequence[float],
                            min_green: float = 7.0, max_green: float = 50.0,
                            unit_extension: float = 3.0,
                            volume_density: bool = False,
                            vd_min_gap: float = 2.0,
                            vd_time_before: float = 10.0,
                            vd_time_to_reduce: float = 20.0) -> ActuatedResult:
    """Run one actuated phase: it gaps out on the first long headway, or maxes out."""
    ts = sorted(_as_floats(arrival_times, "arrival_times"))
    gmin = _nonneg(min_green, "min_green")
    gmax = _positive(max_green, "max_green")
    if gmax < gmin:
        raise SignalControlError("max_green must be >= min_green.")
    ue = _positive(unit_extension, "unit_extension")
    terminate_at = gmin
    extensions = 0
    for t in ts:
        if t < 0.0:
            continue
        if t > terminate_at + 1e-12:
            break                                   # the gap opened: gap out
        gap = volume_density_gap(t, ue, vd_min_gap, vd_time_before,
                                 vd_time_to_reduce) if volume_density else ue
        new_term = t + gap
        if new_term > terminate_at + 1e-12:
            terminate_at = new_term
            extensions += 1
        if terminate_at >= gmax:
            break
    if terminate_at >= gmax - 1e-12:
        green, reason = gmax, "max-out"
    elif extensions == 0:
        green, reason = gmin, "min-green"
    else:
        green, reason = terminate_at, "gap-out"
    served = sum(1 for t in ts if 0.0 <= t <= green + 1e-12)
    return ActuatedResult(green=green, terminated_by=reason,
                          vehicles_served=served, extensions=extensions)


def expected_actuated_green(rate_veh_h: float, unit_extension: float = 3.0,
                            min_green: float = 7.0,
                            max_green: float = 50.0) -> float:
    """Expected green ``(e^{lambda u} - 1)/lambda`` for Poisson arrivals, clamped."""
    rate = _nonneg(rate_veh_h, "rate_veh_h")
    ue = _positive(unit_extension, "unit_extension")
    gmin = _nonneg(min_green, "min_green")
    gmax = _positive(max_green, "max_green")
    if gmax < gmin:
        raise SignalControlError("max_green must be >= min_green.")
    if rate <= _EPS:
        return gmin
    lam = rate / 3600.0
    g = (math.exp(lam * ue) - 1.0) / lam
    return _clamp(g, gmin, gmax)


# ===========================================================================
# 12. MAX-PRESSURE (BACKPRESSURE) CONTROL
# ===========================================================================
def _movement_links(movements: Sequence[Sequence[Any]]
                    ) -> List[Tuple[int, Optional[int]]]:
    """Normalise a movement list to ``(in_link, out_link or None)`` pairs."""
    out: List[Tuple[int, Optional[int]]] = []
    for m in movements:
        if len(m) != 2:
            raise SignalControlError("each movement must be (in_link, out_link).")
        a = int(m[0])
        b = m[1]
        if a < 0:
            raise SignalControlError("in_link must be >= 0.")
        if b is None or int(b) < 0:
            out.append((a, None))
        else:
            out.append((a, int(b)))
    return out


def _normalised_turn_ratios(links: Sequence[Tuple[int, Optional[int]]],
                            turn_ratios: Optional[Sequence[float]]
                            ) -> List[float]:
    """Turn proportions per movement, defaulting to an equal split per link."""
    n = len(links)
    by_link: Dict[int, List[int]] = {}
    for idx, (a, _) in enumerate(links):
        by_link.setdefault(a, []).append(idx)
    if turn_ratios is None:
        out = [0.0] * n
        for a, ids in by_link.items():
            for i in ids:
                out[i] = 1.0 / len(ids)
        return out
    tr = _as_floats(turn_ratios, "turn_ratios")
    if len(tr) != n:
        raise SignalControlError("turn_ratios needs one entry per movement.")
    for v in tr:
        if v < 0.0:
            raise SignalControlError("turn ratios must be >= 0.")
    for a, ids in by_link.items():
        tot = math.fsum(tr[i] for i in ids)
        if abs(tot - 1.0) > 1e-6:
            raise SignalControlError(
                "turn ratios out of link %d sum to %.6f, not 1." % (a, tot))
    return tr


def movement_pressure(movement: int, queues: Sequence[float],
                      movements: Sequence[Sequence[Any]],
                      turn_ratios: Optional[Sequence[float]] = None) -> float:
    """Varaiya's movement pressure ``w = x_m - sum_n p_n x_n`` over downstream turns."""
    links = _movement_links(movements)
    x = _as_floats(queues, "queues")
    if len(x) != len(links):
        raise SignalControlError("queues needs one entry per movement.")
    tr = _normalised_turn_ratios(links, turn_ratios)
    m = int(movement)
    if m < 0 or m >= len(links):
        raise SignalControlError("movement index %d out of range." % m)
    _, out_link = links[m]
    if out_link is None:
        return x[m]
    down = math.fsum(tr[n] * x[n] for n in range(len(links))
                     if links[n][0] == out_link)
    return x[m] - down


def stage_pressure(stage: Sequence[int], queues: Sequence[float],
                   movements: Sequence[Sequence[Any]],
                   sat_flows: Sequence[float],
                   turn_ratios: Optional[Sequence[float]] = None,
                   clip_negative: bool = False) -> float:
    """Stage pressure ``sum_m s_m w_m`` -- the quantity max-pressure maximises."""
    s = _as_floats(sat_flows, "sat_flows")
    st = _as_ints(stage, "stage")
    total = 0.0
    for m in st:
        w = movement_pressure(m, queues, movements, turn_ratios)
        if clip_negative and w < 0.0:
            w = 0.0
        total += s[m] * w
    return total


def max_pressure_stage(stages: Sequence[Sequence[int]],
                       queues: Sequence[float],
                       movements: Sequence[Sequence[Any]],
                       sat_flows: Sequence[float],
                       turn_ratios: Optional[Sequence[float]] = None,
                       clip_negative: bool = False) -> Tuple[int, float]:
    """The stage with the greatest pressure (lowest index wins ties)."""
    if not stages:
        raise SignalControlError("at least one stage is required.")
    best_k, best_p = 0, -_INF
    for k, st in enumerate(stages):
        p = stage_pressure(st, queues, movements, sat_flows, turn_ratios,
                           clip_negative)
        if p > best_p + 1e-12:
            best_k, best_p = k, p
    return best_k, best_p


@dataclass
class MaxPressureResult:
    """Queue trajectory and stage sequence from a max-pressure simulation."""

    queues: List[float]
    max_queue: float
    mean_queue: float
    final_total: float
    stage_sequence: List[int]
    total_history: List[float]
    total_delay: float

    def is_bounded(self, tail: float = 0.5, tol: float = 1e-6) -> bool:
        """True when the total queue stops growing over the final fraction of the run."""
        h = self.total_history
        n = len(h)
        if n < 4:
            return True
        cut = max(1, int(n * _fraction(tail, "tail")))
        first = max(h[n - cut: n - cut // 2] or [0.0])
        last = max(h[n - cut // 2:] or [0.0])
        return last <= first + tol + 1e-3 * max(1.0, first)

    def as_dict(self) -> Dict[str, Any]:
        """A plain JSON-able view."""
        return {"queues": list(self.queues), "maxQueue": self.max_queue,
                "meanQueue": self.mean_queue, "finalTotal": self.final_total,
                "stageSequence": list(self.stage_sequence),
                "totalDelay": self.total_delay}


def simulate_max_pressure(movements: Sequence[Sequence[Any]],
                          stages: Sequence[Sequence[int]],
                          sat_flows: Sequence[float],
                          demands: Sequence[float], horizon_s: float,
                          control_interval: float = 10.0,
                          turn_ratios: Optional[Sequence[float]] = None,
                          initial_queues: Optional[Sequence[float]] = None,
                          clip_negative: bool = False,
                          lost_time: float = 0.0,
                          seed: Optional[int] = None,
                          noise: float = 0.0) -> MaxPressureResult:
    """Store-and-forward simulation of the max-pressure policy at one junction."""
    links = _movement_links(movements)
    n = len(links)
    s = _as_floats(sat_flows, "sat_flows")
    d = _as_floats(demands, "demands")
    if len(s) != n or len(d) != n:
        raise SignalControlError("sat_flows and demands need one entry per "
                                 "movement.")
    tr = _normalised_turn_ratios(links, turn_ratios)
    dt = _positive(control_interval, "control_interval")
    H = _positive(horizon_s, "horizon_s")
    lt = _nonneg(lost_time, "lost_time")
    if lt >= dt:
        raise SignalControlError("lost_time must be below the control interval.")
    x = [0.0] * n if initial_queues is None else _as_floats(initial_queues,
                                                            "initial_queues")
    if len(x) != n:
        raise SignalControlError("initial_queues needs one entry per movement.")
    x = [_nonneg(v, "queue") for v in x]
    rng = random.Random(int(seed)) if seed is not None else None
    nz = _nonneg(noise, "noise")
    out_movements: Dict[int, List[int]] = {}
    for idx, (a, _) in enumerate(links):
        out_movements.setdefault(a, []).append(idx)
    seq: List[int] = []
    hist: List[float] = []
    max_q = 0.0
    acc = 0.0
    steps = max(1, int(round(H / dt)))
    for _ in range(steps):
        k, _p = max_pressure_stage(stages, x, movements, s, tr, clip_negative)
        seq.append(k)
        green = dt - lt
        served_total: List[float] = [0.0] * n
        for m in stages[k]:
            cap = s[m] * green / 3600.0
            served = min(x[m], cap)
            served_total[m] = served
        for m in range(n):
            if served_total[m] <= 0.0:
                continue
            x[m] -= served_total[m]
            out_link = links[m][1]
            if out_link is not None:
                for nxt in out_movements.get(out_link, []):
                    x[nxt] += tr[nxt] * served_total[m]
        for m in range(n):
            arr = d[m] * dt / 3600.0
            if rng is not None and nz > 0.0:
                arr = max(0.0, arr * (1.0 + rng.gauss(0.0, nz)))
            x[m] += arr
        tot = math.fsum(x)
        hist.append(tot)
        acc += tot * dt
        max_q = max(max_q, tot)
    return MaxPressureResult(queues=list(x), max_queue=max_q,
                             mean_queue=math.fsum(hist) / len(hist),
                             final_total=math.fsum(x), stage_sequence=seq,
                             total_history=hist, total_delay=acc)


def _solve_linear(A: List[List[float]], b: Sequence[float]
                  ) -> Optional[List[float]]:
    """Solve ``A x = b`` by Gaussian elimination with scaled partial pivoting.

    Returns ``None`` when ``A`` is numerically singular.  The pivot test is
    relative to each row's own magnitude, so the answer does not depend on the
    units ``A`` and ``b`` are quoted in.
    """
    n = len(b)
    if any(len(row) != n for row in A) or len(A) != n:
        raise SignalControlError("_solve_linear needs a square system.")
    aug = [list(A[i]) + [float(b[i])] for i in range(n)]
    scale = [max([abs(v) for v in A[i]] or [0.0]) for i in range(n)]
    for col in range(n):
        piv, best = -1, 0.0
        for r in range(col, n):
            sc = scale[r] if scale[r] > 0.0 else 1.0
            v = abs(aug[r][col]) / sc
            if v > best:
                best, piv = v, r
        if piv < 0 or best <= 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        scale[col], scale[piv] = scale[piv], scale[col]
        pivot = aug[col][col]
        for r in range(col + 1, n):
            factor = aug[r][col] / pivot
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        acc = aug[r][n] - math.fsum(aug[r][c] * x[c] for c in range(r + 1, n))
        x[r] = acc / aug[r][r]
    return x


def max_pressure_stability_margin(movements: Sequence[Sequence[Any]],
                                  stages: Sequence[Sequence[int]],
                                  sat_flows: Sequence[float],
                                  demands: Sequence[float],
                                  turn_ratios: Optional[Sequence[float]] = None,
                                  lost_fraction: float = 0.0
                                  ) -> Tuple[float, List[float]]:
    """Largest demand multiplier inside the stability region: ``(rho, stage shares)``."""
    links = _movement_links(movements)
    n = len(links)
    s = _as_floats(sat_flows, "sat_flows")
    d = _as_floats(demands, "demands")
    if len(s) != n or len(d) != n:
        raise SignalControlError("sat_flows and demands need one entry per "
                                 "movement.")
    tr = _normalised_turn_ratios(links, turn_ratios)
    lf = _fraction(lost_fraction, "lost_fraction")
    # Equilibrium movement flows solve (I - R) f = d exactly, where R routes a
    # movement's discharge onto the movements leaving its downstream link.
    # Solved directly rather than by Jacobi iteration: the old fixed point
    # stopped on an ABSOLUTE residual of 1e-10 veh/h, so on a near-critical
    # routing loop the same network converged or "failed" purely according to
    # the units the demand happened to be quoted in.
    mat = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for m in range(n):
        out_link = links[m][1]
        if out_link is None:
            continue
        for nn in range(n):
            if links[nn][0] == out_link:
                mat[nn][m] -= tr[nn]
    dmax = max([abs(v) for v in d] + [0.0])
    solved = _solve_linear(mat, d)
    if solved is None or any(v < -1e-9 * max(1.0, dmax) for v in solved):
        has_exit = any(links[m][1] is None for m in range(n))
        raise SignalControlError(
            "movement flows have no finite equilibrium: the routed network "
            "recirculates at least as much flow as it discharges%s."
            % ("" if has_exit else " and no movement leaves the network"))
    f = [max(0.0, v) for v in solved]
    K = len(stages)
    if K < 1:
        raise SignalControlError("at least one stage is required.")
    if max(f) <= _EPS:
        return _INF, [1.0 / K] * K
    # The LP is homogeneous of degree -1 in f (rho(c f) == rho(f) / c), so
    # normalise the flows to O(1) before the simplex and undo it afterwards.
    # Without this the simplex's absolute pivot tolerances make rho depend on
    # whether the demand was quoted in veh/h or veh/decade.
    fmax = max(f)
    fn = [v / fmax for v in f]
    # variables alpha_0..alpha_{K-1}, rho
    cost = [0.0] * K + [-1.0]
    A_ub, b_ub = [], []
    for m in range(n):
        row = [0.0] * (K + 1)
        for k, st in enumerate(stages):
            if m in [int(v) for v in st]:
                row[k] = -s[m] * (1.0 - lf)
        row[K] = fn[m]
        A_ub.append(row)
        b_ub.append(0.0)
    A_eq = [[1.0] * K + [0.0]]
    b_eq = [1.0]
    res = linear_program(cost, A_ub, b_ub, A_eq, b_eq)
    if not res.success:
        raise SignalControlError("stability LP failed with status %r." % res.status)
    return res.x[K] / fmax, res.x[:K]


# ===========================================================================
# 13. SELF-ORGANISING CONTROL (LAEMMER-HELBING)
# ===========================================================================
def laemmer_index(queue: float, demand: float, sat_flow: float,
                  setup_time: float) -> Tuple[float, float]:
    """Laemmer's optimising index ``pi`` (veh/h) and the expected clearing time."""
    x = _nonneg(queue, "queue")
    q = _nonneg(demand, "demand") / 3600.0
    s = _positive(sat_flow, "sat_flow") / 3600.0
    tau = _nonneg(setup_time, "setup_time")
    if q >= s - _EPS:
        return _INF, _INF                      # the movement cannot be cleared
    n_hat = x + q * tau                        # queue when the green starts
    ghat = n_hat / (s - q)
    denom = tau + ghat
    if denom <= _EPS:
        return 0.0, 0.0
    served = x + q * denom
    return 3600.0 * served / denom, ghat


def laemmer_select_stage(stages: Sequence[Sequence[int]],
                         queues: Sequence[float], demands: Sequence[float],
                         sat_flows: Sequence[float], setup_time: float = 5.0,
                         waiting_times: Optional[Sequence[float]] = None,
                         max_waiting: float = 120.0
                         ) -> Tuple[int, str, List[float]]:
    """Pick a stage by the Laemmer index, with the stabilising regime taking priority."""
    x = _as_floats(queues, "queues")
    q = _as_floats(demands, "demands")
    s = _as_floats(sat_flows, "sat_flows")
    _same_len(x, q, "queues", "demands")
    _same_len(x, s, "queues", "sat_flows")
    if not stages:
        raise SignalControlError("at least one stage is required.")
    tau = _nonneg(setup_time, "setup_time")
    wmax = _positive(max_waiting, "max_waiting")
    w = [0.0] * len(x) if waiting_times is None else _as_floats(waiting_times,
                                                                "waiting_times")
    _same_len(w, x, "waiting_times", "queues")
    indices: List[float] = []
    for st in stages:
        ids = _as_ints(st, "stage")
        if not ids:
            raise SignalControlError("a stage is empty.")
        ghat = 0.0
        for m in ids:
            _, g = laemmer_index(x[m], q[m], s[m], tau)
            ghat = max(ghat, 0.0 if math.isinf(g) else g)
        denom = tau + ghat
        if denom <= _EPS:
            indices.append(0.0)
            continue
        served = math.fsum(x[m] + q[m] / 3600.0 * denom for m in ids)
        indices.append(3600.0 * served / denom)
    # stabilising regime: any movement waiting too long pre-empts the index
    critical = []
    for k, st in enumerate(stages):
        ids = _as_ints(st, "stage")
        worst = max((w[m] for m in ids if x[m] > _EPS), default=0.0)
        if worst > wmax:
            critical.append((worst, -k, k))
    if critical:
        critical.sort(reverse=True)
        return critical[0][2], "stabilising", indices
    best = max(range(len(stages)), key=lambda k: (indices[k], -k))
    return best, "optimising", indices


def simulate_laemmer(stages: Sequence[Sequence[int]], demands: Sequence[float],
                     sat_flows: Sequence[float], horizon_s: float,
                     setup_time: float = 5.0, control_interval: float = 1.0,
                     min_green: float = 5.0, max_waiting: float = 120.0,
                     initial_queues: Optional[Sequence[float]] = None,
                     seed: Optional[int] = None,
                     noise: float = 0.0) -> Dict[str, Any]:
    """Time-stepped simulation of Laemmer-Helbing self-organising control."""
    q = _as_floats(demands, "demands")
    s = _as_floats(sat_flows, "sat_flows")
    _same_len(q, s, "demands", "sat_flows")
    n = len(q)
    if not stages:
        raise SignalControlError("at least one stage is required.")
    dt = _positive(control_interval, "control_interval")
    H = _positive(horizon_s, "horizon_s")
    tau = _nonneg(setup_time, "setup_time")
    gmin = _nonneg(min_green, "min_green")
    x = [0.0] * n if initial_queues is None else _as_floats(initial_queues,
                                                            "initial_queues")
    if len(x) != n:
        raise SignalControlError("initial_queues needs one entry per movement.")
    w = [0.0] * n
    rng = random.Random(int(seed)) if seed is not None else None
    nz = _nonneg(noise, "noise")
    cur, elapsed, switching = 0, 0.0, 0.0
    seq: List[int] = []
    switches = 0
    acc_delay = 0.0
    max_q = 0.0
    steps = max(1, int(round(H / dt)))
    for _ in range(steps):
        if switching > 0.0:
            switching = max(0.0, switching - dt)
        elif elapsed >= gmin:
            want, regime, _idx = laemmer_select_stage(stages, x, q, s, tau, w,
                                                      max_waiting)
            if want != cur:
                cur = want
                switches += 1
                switching = tau
                elapsed = 0.0
        seq.append(-1 if switching > 0.0 else cur)
        if switching <= 0.0:
            for m in _as_ints(stages[cur], "stage"):
                served = min(x[m], s[m] * dt / 3600.0)
                x[m] -= served
                if x[m] <= _EPS:
                    w[m] = 0.0
            elapsed += dt
        for m in range(n):
            arr = q[m] * dt / 3600.0
            if rng is not None and nz > 0.0:
                arr = max(0.0, arr * (1.0 + rng.gauss(0.0, nz)))
            x[m] += arr
            if x[m] > _EPS:
                w[m] += dt
        tot = math.fsum(x)
        acc_delay += tot * dt
        max_q = max(max_q, tot)
    return {"queues": list(x), "maxQueue": max_q, "switches": switches,
            "totalDelay": acc_delay, "meanQueue": acc_delay / (steps * dt),
            "stageSequence": seq, "maxWait": max(w) if w else 0.0}


# ===========================================================================
# 14. TRANSIT SIGNAL PRIORITY
# ===========================================================================
def _stage_windows(plan: SignalPlan) -> List[Tuple[float, float]]:
    """Displayed-green window (start, end) of every stage on the cycle clock."""
    out: List[Tuple[float, float]] = []
    t = plan.offset % plan.cycle if plan.cycle > 0.0 else 0.0
    for i in range(plan.n_stages):
        out.append((t, t + plan.greens[i]))
        t += plan.greens[i] + plan.intergreens[i]
    return out


def _reallocate_green(plan: SignalPlan, stage: int, delta: float,
                      min_green: float = 7.0,
                      name: Optional[str] = None) -> Tuple[SignalPlan, float]:
    """Move ``delta`` seconds of green to one stage from the rest; cycle preserved."""
    g = plan.effective_greens()
    k = int(stage)
    if k < 0 or k >= plan.n_stages:
        raise SignalControlError("stage %d out of range." % k)
    others = [i for i in range(plan.n_stages) if i != k]
    if not others:
        return plan, 0.0
    d = _finite(delta, "delta")
    if d > 0.0:
        slack = [max(0.0, g[i] - min_green) for i in others]
        tot = math.fsum(slack)
        if tot <= _EPS:
            return plan, 0.0
        d = min(d, tot)
        for pos, i in enumerate(others):
            g[i] -= d * slack[pos] / tot
    elif d < 0.0:
        take = -d
        take = min(take, max(0.0, g[k] - min_green))
        share = [g[i] for i in others]
        tot = math.fsum(share)
        if tot <= _EPS:
            share = [1.0] * len(others)
            tot = float(len(others))
        for pos, i in enumerate(others):
            g[i] += take * share[pos] / tot
        d = -take
    g[k] += d
    resid = math.fsum(plan.effective_greens()) - math.fsum(g)
    if abs(resid) > 1e-12:
        g[k] += resid
    return (make_signal_plan(plan.stages, g, plan.intergreens,
                             lost_times=plan.lost_times, offset=plan.offset,
                             name=name or (plan.name + "+tsp")), d)


@dataclass
class TSPRequest:
    """A priority request from a transit vehicle approaching the junction."""

    stage: int
    arrival: float
    passengers: int = 40
    lateness_s: float = 0.0
    absolute: bool = False

    def as_dict(self) -> Dict[str, Any]:
        """A plain JSON-able view."""
        return {"stage": self.stage, "arrival": self.arrival,
                "passengers": self.passengers, "latenessSeconds": self.lateness_s,
                "absolute": self.absolute}


@dataclass
class TSPResult:
    """The plan after a priority intervention, and what it bought the bus."""

    plan: SignalPlan
    strategy: str
    adjustment: float
    bus_delay_without: float
    bus_delay_with: float

    @property
    def delay_saved(self) -> float:
        """Seconds of bus delay removed by the intervention."""
        return self.bus_delay_without - self.bus_delay_with

    def as_dict(self) -> Dict[str, Any]:
        """A plain JSON-able view."""
        return {"strategy": self.strategy, "adjustment": self.adjustment,
                "busDelayWithout": self.bus_delay_without,
                "busDelayWith": self.bus_delay_with,
                "delaySaved": self.delay_saved, "plan": self.plan.as_dict()}


def _bus_delay(plan: SignalPlan, stage: int, arrival: float) -> float:
    """Seconds a bus arriving at ``arrival`` waits for the next green of its stage."""
    C = plan.cycle
    start, end = _stage_windows(plan)[int(stage)]
    t = arrival % C
    if (t - start) % C < (end - start) - 1e-12:
        return 0.0
    return (start - t) % C


def tsp_green_extension(plan: SignalPlan, stage: int, bus_arrival: float,
                        max_extension: float = 10.0,
                        min_green: float = 7.0) -> TSPResult:
    """Hold the green a little longer so a late bus clears without stopping."""
    C = plan.cycle
    k = int(stage)
    if k < 0 or k >= plan.n_stages:
        raise SignalControlError("stage %d out of range." % k)
    start, end = _stage_windows(plan)[k]
    t = float(bus_arrival) % C
    before = _bus_delay(plan, k, t)
    need = (t - end) % C
    ext = 0.0
    if before > 0.0 and need <= _nonneg(max_extension, "max_extension"):
        ext = need + 0.5
    if ext <= 0.0:
        return TSPResult(plan, "none", 0.0, before, before)
    new_plan, applied = _reallocate_green(plan, k, ext, min_green,
                                          plan.name + "+extend")
    # Always re-measure on the plan actually returned.  Assuming the bus clears
    # whenever the full extension was granted is wrong: _reallocate_green pays
    # for the extension out of EVERY other stage, including the ones that run
    # BEFORE stage k, so the whole stage-k window slides earlier as it grows and
    # the tail can advance by far less than `applied` -- or move away entirely.
    after = _bus_delay(new_plan, k, t)
    return TSPResult(new_plan, "green-extension", applied, before, after)


def tsp_early_start(plan: SignalPlan, stage: int, bus_arrival: float,
                    max_truncation: float = 10.0,
                    min_green: float = 7.0) -> TSPResult:
    """Truncate the preceding red so the bus stage starts early (red truncation)."""
    C = plan.cycle
    k = int(stage)
    if k < 0 or k >= plan.n_stages:
        raise SignalControlError("stage %d out of range." % k)
    start, _end = _stage_windows(plan)[k]
    t = float(bus_arrival) % C
    before = _bus_delay(plan, k, t)
    trunc = 0.0
    if 0.0 < before <= _nonneg(max_truncation, "max_truncation"):
        trunc = before
    if trunc <= 0.0:
        return TSPResult(plan, "none", 0.0, before, before)
    prev = (k - 1) % plan.n_stages
    g = plan.effective_greens()
    slack = max(0.0, g[prev] - min_green)
    applied = min(trunc, slack)
    if applied <= 0.0:
        return TSPResult(plan, "none", 0.0, before, before)
    g[prev] -= applied
    g[k] += applied
    new_plan = make_signal_plan(plan.stages, g, plan.intergreens,
                                lost_times=plan.lost_times,
                                offset=(plan.offset - applied) % C,
                                name=plan.name + "+early")
    # Re-measure rather than assuming the window moved exactly `applied` seconds
    # earlier: shortening the preceding stage moves it once and re-basing the
    # offset moves it again, so `before - applied` understates the benefit.
    after = _bus_delay(new_plan, k, t)
    return TSPResult(new_plan, "early-start", applied, before, after)


def tsp_phase_insertion(plan: SignalPlan, movements: Sequence[int],
                        duration: float, intergreen: float = 5.0,
                        start_lost: float = 2.0, end_gain: float = 3.0,
                        min_green: float = 7.0, after_stage: int = -1
                        ) -> TSPResult:
    """Insert a short bus-only stage, paying for it from the other stages' green."""
    dur = _positive(duration, "duration")
    ig = _nonneg(intergreen, "intergreen")
    mv = sorted(set(_as_ints(movements, "movements")))
    if not mv:
        raise SignalControlError("a bus stage needs at least one movement.")
    lost = stage_lost_time(ig, start_lost, end_gain)
    g = plan.effective_greens()
    need = dur + lost                       # the new stage's total clock time
    slack = [max(0.0, v - min_green) for v in g]
    tot = math.fsum(slack)
    if tot < need - 1e-9:
        raise SignalControlError(
            "inserting a %.1f s bus stage needs %.1f s but only %.1f s of green "
            "is spare above the %.1f s minimum." % (dur, need, tot, min_green))
    new_g = [g[i] - need * slack[i] / tot for i in range(len(g))]
    pos = plan.n_stages if after_stage < 0 else int(after_stage) + 1
    pos = max(0, min(pos, plan.n_stages))
    stages = [list(s) for s in plan.stages]
    stages.insert(pos, mv)
    new_g.insert(pos, dur)
    igs = list(plan.intergreens)
    igs.insert(pos, ig)
    lts = list(plan.lost_times)
    lts.insert(pos, lost)
    resid = plan.cycle - (math.fsum(new_g) + math.fsum(lts))
    if abs(resid) > 1e-12:
        new_g[pos] += resid
    new_plan = make_signal_plan(stages, new_g, igs, lost_times=lts,
                                offset=plan.offset, name=plan.name + "+busphase")
    return TSPResult(new_plan, "phase-insertion", dur, plan.cycle, 0.0)


def apply_transit_priority(plan: SignalPlan, request: TSPRequest,
                           max_extension: float = 10.0,
                           max_truncation: float = 10.0,
                           min_green: float = 7.0) -> TSPResult:
    """Choose and apply the cheapest priority tactic for one bus request."""
    if not isinstance(request, TSPRequest):
        raise SignalControlError("request must be a TSPRequest.")
    k = int(request.stage)
    if k < 0 or k >= plan.n_stages:
        raise SignalControlError("stage %d out of range." % k)
    before = _bus_delay(plan, k, request.arrival)
    if before <= 0.0:
        return TSPResult(plan, "none", 0.0, 0.0, 0.0)
    ext = tsp_green_extension(plan, k, request.arrival, max_extension, min_green)
    early = tsp_early_start(plan, k, request.arrival, max_truncation, min_green)
    best = max((ext, early), key=lambda r: (r.delay_saved, -r.adjustment))
    if best.delay_saved <= 0.0 and request.absolute:
        return tsp_phase_insertion(plan, plan.stages[k],
                                   duration=max(min_green, 8.0),
                                   min_green=min_green)
    return best


# ===========================================================================
# 15. PEDESTRIAN PHASING
# ===========================================================================
def concurrent_pedestrian_check(green: float, intergreen: float,
                                required_ped_time: float) -> Tuple[bool, float]:
    """Can a crossing run concurrently with the vehicle stage? Returns (ok, deficit)."""
    g = _nonneg(green, "green")
    ig = _nonneg(intergreen, "intergreen")
    need = _nonneg(required_ped_time, "required_ped_time")
    available = g + ig
    return (available + 1e-9 >= need, max(0.0, need - available))


def exclusive_pedestrian_stage(plan: SignalPlan, ped_movements: Sequence[int],
                               ped_time: float, intergreen: float = 5.0,
                               start_lost: float = 2.0, end_gain: float = 3.0,
                               min_green: float = 7.0,
                               after_stage: int = -1) -> SignalPlan:
    """Insert an exclusive pedestrian stage, holding the cycle length constant."""
    res = tsp_phase_insertion(plan, ped_movements, ped_time, intergreen,
                              start_lost, end_gain, min_green, after_stage)
    out = res.plan
    out.name = plan.name + "+pedstage"
    return out


def leading_pedestrian_interval(plan: SignalPlan, stage: int, lpi: float = 4.0
                                ) -> SignalPlan:
    """Give people on foot a head start: the LPI becomes lost time for vehicles."""
    k = int(stage)
    if k < 0 or k >= plan.n_stages:
        raise SignalControlError("stage %d out of range." % k)
    head = _positive(lpi, "lpi")
    g = plan.effective_greens()
    if g[k] - head < 0.0:
        raise SignalControlError(
            "a %.1f s leading interval exceeds the %.1f s green of stage %d."
            % (head, g[k], k))
    lts = list(plan.lost_times)
    lts[k] += head
    return SignalPlan(stages=[list(s) for s in plan.stages],
                      greens=list(plan.greens),
                      intergreens=list(plan.intergreens), lost_times=lts,
                      offset=plan.offset, name=plan.name + "+lpi")


def barnes_dance_time(width_a: float, width_b: float, walk_speed: float = 1.2,
                      start_up: float = 3.2) -> float:
    """All-way ('Barnes dance') crossing time, set by the diagonal route."""
    a = _positive(width_a, "width_a")
    b = _positive(width_b, "width_b")
    return pedestrian_crossing_time(math.hypot(a, b), walk_speed, start_up)


def pedestrian_delay(cycle: float, ped_green: float) -> float:
    """Uniform pedestrian delay ``(C - g)^2 / (2C)`` seconds per pedestrian."""
    C = _positive(cycle, "cycle")
    g = _nonneg(ped_green, "ped_green")
    if g > C:
        raise SignalControlError("ped_green (%g) exceeds the cycle (%g)." % (g, C))
    return (C - g) ** 2 / (2.0 * C)


# ===========================================================================
# 16. UNSIGNALISED JUNCTIONS: GAP ACCEPTANCE
# ===========================================================================
def hcm_potential_capacity(conflicting_flow: float, critical_gap: float,
                           follow_up: float) -> float:
    """HCM two-way-stop potential capacity from the exponential gap-acceptance form."""
    vc = _nonneg(conflicting_flow, "conflicting_flow")
    tc = _positive(critical_gap, "critical_gap")
    tf = _positive(follow_up, "follow_up")
    if vc <= _EPS:
        return 3600.0 / tf
    num = vc * math.exp(-vc * tc / 3600.0)
    den = 1.0 - math.exp(-vc * tf / 3600.0)
    if den <= _EPS:
        return 3600.0 / tf
    return num / den


def rank_impedance_factor(demand: float, capacity: float) -> float:
    """Probability a higher-rank movement's queue is empty ``p0 = 1 - v/c``."""
    v = _nonneg(demand, "demand")
    c = _nonneg(capacity, "capacity")
    if c <= _EPS:
        return 0.0
    return _clamp(1.0 - v / c, 0.0, 1.0)


def twsc_movement_capacity(potential_capacity: float,
                           impedance_factors: Sequence[float] = ()) -> float:
    """Movement capacity ``c_m = c_p * prod(p0)`` after higher-rank impedance."""
    cp = _nonneg(potential_capacity, "potential_capacity")
    p = 1.0
    for f in _as_floats(impedance_factors, "impedance_factors"):
        p *= _fraction(f, "impedance factor")
    return cp * p


def twsc_control_delay(demand: float, capacity: float, period_h: float = 0.25
                       ) -> float:
    """HCM two-way-stop control delay (s/veh), including the 5 s deceleration term."""
    v = _nonneg(demand, "demand")
    c = _nonneg(capacity, "capacity")
    T = _positive(period_h, "period_h")
    if c <= _EPS:
        return _INF if v > 0.0 else 0.0
    x = v / c
    inner = (x - 1.0) ** 2 + (3600.0 / c) * x / (450.0 * T)
    return 3600.0 / c + 900.0 * T * ((x - 1.0) + math.sqrt(max(inner, 0.0))) + 5.0


def siegloch_capacity(conflicting_flow: float, critical_gap: float,
                      follow_up: float) -> float:
    """Siegloch's capacity ``(3600/tf) exp(-q(tc - tf/2))`` for a regular gap stream."""
    vc = _nonneg(conflicting_flow, "conflicting_flow")
    tc = _positive(critical_gap, "critical_gap")
    tf = _positive(follow_up, "follow_up")
    q = vc / 3600.0
    return (3600.0 / tf) * math.exp(-q * (tc - tf / 2.0))


def siegloch_critical_gap(gaps: Sequence[float], entries: Sequence[float]
                          ) -> Tuple[float, float]:
    """Siegloch regression ``t = t0 + n tf`` -> ``(critical gap, follow-up time)``."""
    t = _as_floats(gaps, "gaps")
    n = _as_floats(entries, "entries")
    _same_len(t, n, "gaps", "entries")
    _check_len(t, 2, "gaps")
    nn = float(len(t))
    mx = math.fsum(n) / nn
    my = math.fsum(t) / nn
    sxx = math.fsum((v - mx) ** 2 for v in n)
    if sxx <= _EPS:
        raise SignalControlError("the entry counts have no spread to regress on.")
    sxy = math.fsum((n[i] - mx) * (t[i] - my) for i in range(len(t)))
    tf = sxy / sxx
    t0 = my - tf * mx
    if tf <= 0.0:
        raise SignalControlError("the regression gives a non-positive follow-up "
                                 "time (%.3f s)." % tf)
    return t0 + tf / 2.0, tf


def raff_critical_gap(accepted: Sequence[float], rejected: Sequence[float]
                      ) -> float:
    """Raff's critical gap: where the accepted and rejected gap curves cross."""
    a = sorted(_as_floats(accepted, "accepted"))
    r = sorted(_as_floats(rejected, "rejected"))
    _check_len(a, 1, "accepted")
    _check_len(r, 1, "rejected")
    na, nr = float(len(a)), float(len(r))

    def f(t: float) -> float:
        """F_accepted(<= t) minus F_rejected(> t); increasing in t."""
        fa = sum(1 for v in a if v <= t) / na
        fr = sum(1 for v in r if v > t) / nr
        return fa - fr

    lo = min(a[0], r[0])
    hi = max(a[-1], r[-1])
    if f(lo) >= 0.0:
        return lo
    if f(hi) <= 0.0:
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tanner_capacity(conflicting_flow: float, critical_gap: float,
                    follow_up: float, min_headway: float = 2.0) -> float:
    """Tanner's capacity with a minimum headway in the priority stream (veh/h)."""
    vc = _nonneg(conflicting_flow, "conflicting_flow")
    tc = _positive(critical_gap, "critical_gap")
    tf = _positive(follow_up, "follow_up")
    dl = _nonneg(min_headway, "min_headway")
    if tc < dl:
        raise SignalControlError("critical gap must be at least the minimum "
                                 "headway (%g < %g)." % (tc, dl))
    q = vc / 3600.0
    if q <= _EPS:
        return 3600.0 / tf
    if 1.0 - dl * q <= _EPS:
        return 0.0
    num = q * (1.0 - dl * q) * math.exp(-q * (tc - dl))
    den = 1.0 - math.exp(-q * tf)
    if den <= _EPS:
        return 3600.0 / tf
    return 3600.0 * num / den


def tanner_critical_gap(entry_capacity: float, conflicting_flow: float,
                        follow_up: float, min_headway: float = 2.0) -> float:
    """Invert Tanner's capacity formula for the critical gap implied by a capacity."""
    Q = _positive(entry_capacity, "entry_capacity")
    vc = _positive(conflicting_flow, "conflicting_flow")
    tf = _positive(follow_up, "follow_up")
    dl = _nonneg(min_headway, "min_headway")
    q = vc / 3600.0
    if 1.0 - dl * q <= _EPS:
        raise SignalControlError("the priority stream is saturated at its minimum "
                                 "headway; the critical gap is undefined.")
    ratio = Q * (1.0 - math.exp(-q * tf)) / (3600.0 * q * (1.0 - dl * q))
    if ratio <= 0.0:
        raise SignalControlError("no critical gap reproduces that capacity.")
    return dl - math.log(ratio) / q


def _norm_cdf(z: float) -> float:
    """Standard normal cumulative distribution via ``math.erf``."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def maximum_likelihood_critical_gap(accepted: Sequence[float],
                                    rejected: Sequence[float],
                                    tol: float = 1e-8,
                                    max_iter: int = 200
                                    ) -> Tuple[float, float, float]:
    """Troutbeck's lognormal maximum-likelihood critical gap: (mean, mu, sigma)."""
    a = _as_floats(accepted, "accepted")
    r = _as_floats(rejected, "rejected")
    _same_len(a, r, "accepted", "rejected")
    _check_len(a, 2, "accepted")
    for i in range(len(a)):
        if a[i] <= 0.0:
            raise SignalControlError("accepted gaps must be > 0.")
        if r[i] < 0.0:
            raise SignalControlError("largest rejected gaps must be >= 0.")
        if r[i] >= a[i]:
            raise SignalControlError(
                "driver %d rejected a gap (%.2f s) at least as large as the one "
                "accepted (%.2f s)." % (i, r[i], a[i]))
    la = [math.log(v) for v in a]
    lr = [math.log(v) if v > 0.0 else -30.0 for v in r]

    def nll(mu: float, sigma: float) -> float:
        """Negative log-likelihood of the interval-censored sample."""
        if sigma <= 1e-6:
            return _INF
        total = 0.0
        for i in range(len(la)):
            p = (_norm_cdf((la[i] - mu) / sigma) -
                 _norm_cdf((lr[i] - mu) / sigma))
            if p <= 1e-300:
                return _INF
            total -= math.log(p)
        return total

    mu = math.fsum(la) / len(la)
    sigma = 0.3
    step_mu, step_sg = 0.5, 0.2
    best = nll(mu, sigma)
    for _ in range(int(max_iter)):
        improved = False
        for d in (step_mu, -step_mu):
            v = nll(mu + d, sigma)
            if v < best - tol:
                mu, best, improved = mu + d, v, True
        for d in (step_sg, -step_sg):
            if sigma + d <= 1e-4:
                continue
            v = nll(mu, sigma + d)
            if v < best - tol:
                sigma, best, improved = sigma + d, v, True
        if not improved:
            step_mu *= 0.5
            step_sg *= 0.5
            if step_mu < 1e-9 and step_sg < 1e-9:
                break
    return math.exp(mu + 0.5 * sigma * sigma), mu, sigma


# ===========================================================================
# 17. ROUNDABOUT CAPACITY
# ===========================================================================
def hcm_roundabout_capacity(circulating_flow: float, A: float = 1130.0,
                            B: float = 0.001,
                            critical_gap: Optional[float] = None,
                            follow_up: Optional[float] = None) -> float:
    """HCM roundabout entry capacity ``c = A exp(-B v_c)`` (veh/h).

    Either give the fitted coefficients ``A`` and ``B`` directly (the HCM
    defaults 1130 and 0.001 are the single-lane entry / single-lane circulating
    line), or give BOTH gap parameters and let them derive the coefficients by
    the gap-acceptance identity ``A = 3600 / t_f`` and
    ``B = (t_c - t_f / 2) / 3600``.  The two routes are mutually exclusive:
    supplying only one of ``critical_gap`` / ``follow_up`` is an error rather
    than a silent no-op, and so is supplying ``A`` or ``B`` alongside a
    complete gap pair, because the derived coefficients would overwrite them.
    """
    vc = _nonneg(circulating_flow, "circulating_flow")
    if (critical_gap is None) != (follow_up is None):
        raise SignalControlError(
            "critical_gap and follow_up must be given together: %r was supplied "
            "on its own and would otherwise be ignored."
            % ("critical_gap" if follow_up is None else "follow_up",))
    if critical_gap is not None and follow_up is not None:
        if A != 1130.0 or B != 0.001:
            raise SignalControlError(
                "give either the fitted coefficients A and B or the gap pair "
                "(critical_gap, follow_up), not both -- the gap pair derives "
                "A and B and would overwrite the values supplied.")
        tc = _positive(critical_gap, "critical_gap")
        tf = _positive(follow_up, "follow_up")
        if tc <= tf / 2.0:
            raise SignalControlError(
                "critical_gap (%g s) must exceed half the follow-up headway "
                "(%g s) or the capacity would rise with circulating flow."
                % (tc, tf))
        A = 3600.0 / tf
        B = (tc - tf / 2.0) / 3600.0
    a = _positive(A, "A")
    b = _nonneg(B, "B")
    return a * math.exp(-b * vc)


def brilon_wu_capacity(circulating_flow: float, n_entry_lanes: int = 1,
                       n_circulating_lanes: int = 1, critical_gap: float = 4.1,
                       follow_up: float = 2.9, min_headway: float = 2.1
                       ) -> float:
    """Brilon-Wu roundabout entry capacity (the German bunched-headway form)."""
    vc = _nonneg(circulating_flow, "circulating_flow")
    ne = int(n_entry_lanes)
    nc = int(n_circulating_lanes)
    if ne < 1 or nc < 1:
        raise SignalControlError("lane counts must be >= 1.")
    tc = _positive(critical_gap, "critical_gap")
    tf = _positive(follow_up, "follow_up")
    tm = _nonneg(min_headway, "min_headway")
    q = vc / 3600.0
    bunch = 1.0 - tm * q / nc
    if bunch <= _EPS:
        return 0.0
    return (3600.0 * ne / tf) * (bunch ** nc) * math.exp(
        -q * (tc - tf / 2.0 - tm))


@dataclass
class KimberGeometry:
    """The TRL/Kimber geometric parameters of one roundabout entry (metres, degrees)."""

    entry_width: float = 6.0
    approach_half_width: float = 3.5
    flare_length: float = 15.0
    inscribed_diameter: float = 40.0
    entry_angle: float = 30.0
    entry_radius: float = 20.0

    def as_dict(self) -> Dict[str, Any]:
        """A plain JSON-able view."""
        return {"entryWidth": self.entry_width,
                "approachHalfWidth": self.approach_half_width,
                "flareLength": self.flare_length,
                "inscribedDiameter": self.inscribed_diameter,
                "entryAngle": self.entry_angle,
                "entryRadius": self.entry_radius}


def kimber_geometry(geom: KimberGeometry) -> Tuple[float, float, float]:
    """The Kimber coefficients ``(k, F, f_c)`` implied by the entry geometry."""
    if not isinstance(geom, KimberGeometry):
        raise SignalControlError("geom must be a KimberGeometry.")
    e = _positive(geom.entry_width, "entry_width")
    v = _positive(geom.approach_half_width, "approach_half_width")
    lp = _positive(geom.flare_length, "flare_length")
    D = _positive(geom.inscribed_diameter, "inscribed_diameter")
    phi = _finite(geom.entry_angle, "entry_angle")
    r = _positive(geom.entry_radius, "entry_radius")
    if e < v:
        raise SignalControlError("entry width must be at least the approach "
                                 "half-width (%g < %g)." % (e, v))
    S = 1.6 * (e - v) / lp
    x2 = v + (e - v) / (1.0 + 2.0 * S)
    M = math.exp((D - 60.0) / 10.0)
    tD = 1.0 + 0.5 / (1.0 + M)
    F = 303.0 * x2
    fc = 0.210 * tD * (1.0 + 0.2 * x2)
    k = 1.0 - 0.00347 * (phi - 30.0) - 0.978 * (1.0 / r - 0.05)
    return k, F, fc


def kimber_capacity(circulating_flow: float,
                    geom: Optional[KimberGeometry] = None,
                    k: Optional[float] = None, F: Optional[float] = None,
                    f_c: Optional[float] = None) -> float:
    """UK TRL/Kimber linear entry capacity ``Q_e = k (F - f_c Q_c)`` (pcu/h)."""
    qc = _nonneg(circulating_flow, "circulating_flow")
    if geom is not None:
        kk, FF, ff = kimber_geometry(geom)
    else:
        if k is None or F is None or f_c is None:
            raise SignalControlError("give a KimberGeometry or all of k, F, f_c.")
        kk, FF, ff = _finite(k, "k"), _positive(F, "F"), _nonneg(f_c, "f_c")
    return max(0.0, kk * (FF - ff * qc))


def roundabout_delay(demand: float, capacity: float, period_h: float = 0.25
                     ) -> float:
    """HCM roundabout control delay (s/veh) with the queue-deceleration term."""
    v = _nonneg(demand, "demand")
    c = _nonneg(capacity, "capacity")
    T = _positive(period_h, "period_h")
    if c <= _EPS:
        return _INF if v > 0.0 else 0.0
    x = v / c
    inner = (x - 1.0) ** 2 + (3600.0 / c) * x / (450.0 * T)
    return (3600.0 / c + 900.0 * T * ((x - 1.0) + math.sqrt(max(inner, 0.0)))
            + 5.0 * min(x, 1.0))


# ===========================================================================
# 18. PAYLOADS FOR A FRONTEND
# ===========================================================================
def _json_safe(obj: Any) -> Any:
    """Replace non-finite floats with ``None`` so the payload is strict JSON."""
    if isinstance(obj, float):
        return None if (math.isinf(obj) or math.isnan(obj)) else obj
    if isinstance(obj, dict):
        return dict((str(k), _json_safe(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def build_signal_control_payload(plan: SignalPlan, demands: Sequence[float],
                                 sat_flows: Sequence[float],
                                 movements: Optional[Sequence[Movement]] = None,
                                 period_h: float = 0.25, arrival_type: int = 3,
                                 spacing_m: float = 7.0) -> Dict[str, Any]:
    """Everything a client needs to render and audit one signalised junction."""
    q = _as_floats(demands, "demands")
    s = _as_floats(sat_flows, "sat_flows")
    _same_len(q, s, "demands", "sat_flows")
    delays, avg, los = plan_delay(plan, q, s, period_h=period_h,
                                  arrival_type=arrival_type)
    C = plan.cycle
    rows: List[Dict[str, Any]] = []
    for m in plan.movements():
        g = plan.movement_green(m)
        d = delays[m]
        n1, n2, nq = average_back_of_queue(q[m], s[m], g, C, period_h)
        q95 = queue_length_95(q[m], s[m], g, C, period_h)
        rows.append({
            "id": m,
            "label": movements[m].label if movements is not None else "mv%d" % m,
            "demand": q[m], "satFlow": s[m], "effectiveGreen": g,
            "greenRatio": g / C, "capacity": d.capacity, "vOverC": d.x,
            "flowRatio": flow_ratio(q[m], s[m]),
            "uniformDelay": d.d1, "incrementalDelay": d.d2,
            "initialQueueDelay": d.d3, "progressionFactor": d.progression_factor,
            "controlDelay": d.control_delay, "los": d.los,
            "websterDelay": webster_delay(q[m], s[m], g, C),
            "akcelikDelay": akcelik_delay(q[m], s[m], g, C, period_h),
            "averageQueueVeh": nq, "queue95Veh": q95,
            "queue95Metres": queue_length_metres(q95, spacing_m),
        })
    ycrit = critical_flow_ratios(plan.stages, q, s)
    windows = _stage_windows(plan)
    stages = [{"index": k, "movements": list(plan.stages[k]),
               "displayedGreen": plan.greens[k],
               "effectiveGreen": plan.effective_greens()[k],
               "intergreen": plan.intergreens[k],
               "lostTime": plan.lost_times[k],
               "criticalFlowRatio": ycrit[k],
               "start": windows[k][0], "end": windows[k][1]}
              for k in range(plan.n_stages)]
    summary = {
        "cycle": C, "lostTime": plan.total_lost_time,
        "sumCriticalFlowRatios": math.fsum(ycrit),
        "averageControlDelay": avg, "los": los,
        "worstVOverC": max((r["vOverC"] for r in rows), default=0.0),
        "totalDemand": math.fsum(q[m] for m in plan.movements()),
    }
    return _json_safe({"plan": plan.as_dict(), "stages": stages,
                       "movements": rows, "summary": summary})


def build_a2ui_payload(plan: SignalPlan, demands: Sequence[float],
                       sat_flows: Sequence[float],
                       movements: Optional[Sequence[Movement]] = None,
                       title: str = "Signalised junction",
                       period_h: float = 0.25) -> Dict[str, Any]:
    """A compact, camelCase dashboard payload: KPI tiles, a timeline and a table."""
    core = build_signal_control_payload(plan, demands, sat_flows, movements,
                                        period_h=period_h)
    summary = core["summary"]
    kpis = [
        {"label": "Cycle", "value": summary["cycle"], "unit": "s",
         "format": "0"},
        {"label": "Average delay", "value": summary["averageControlDelay"],
         "unit": "s/veh", "format": "0.0"},
        {"label": "Level of service", "value": summary["los"], "unit": "",
         "format": "text"},
        {"label": "Worst v/c", "value": summary["worstVOverC"], "unit": "",
         "format": "0.00"},
        {"label": "Sum y (critical)", "value": summary["sumCriticalFlowRatios"],
         "unit": "", "format": "0.000"},
        {"label": "Lost time", "value": summary["lostTime"], "unit": "s",
         "format": "0.0"},
    ]
    timeline = [{"stage": st["index"], "start": st["start"], "end": st["end"],
                 "green": st["displayedGreen"], "intergreen": st["intergreen"],
                 "movements": st["movements"]} for st in core["stages"]]
    table = {
        "columns": ["label", "demand", "capacity", "vOverC", "controlDelay",
                    "los", "queue95Veh"],
        "rows": [dict((c, r[c]) for c in ["label", "demand", "capacity",
                                          "vOverC", "controlDelay", "los",
                                          "queue95Veh"])
                 for r in core["movements"]],
    }
    worst = max(core["movements"],
                key=lambda r: (r["vOverC"] if r["vOverC"] is not None else 0.0),
                default=None)
    return _json_safe({
        "title": title, "kind": "signalControl", "version": __version__,
        "kpis": kpis, "timeline": timeline, "table": table,
        "criticalMovement": worst, "plan": core["plan"],
        "summary": summary,
    })


# ===========================================================================
# 19. DEMO
# ===========================================================================
def _demo() -> None:
    print("=" * 70)
    print("  SIGNAL CONTROL (pure Python) - demo")
    print("=" * 70)

    # ------------------------------------------------------------------ 1
    print("\n-- 1. saturation flow: the HCM adjustment chain --")
    factors = SaturationFlowFactors(
        n_lanes=2, lane_width_m=3.3, grade_pct=2.0, pct_heavy=6.0,
        parking=True, parking_manoeuvres=12.0, buses_stopping=8.0, cbd=True,
        lane_utilisation=lane_utilisation_factor([520.0, 430.0]),
        turn_radius_m=15.0,
        ped_bike_factor=pedestrian_bicycle_factor(220.0, 90.0, 30.0,
                                                  turn_proportion=0.20))
    s_adj = saturation_flow_rate(factors)
    fs = factors.factors()
    print("   f_w=%.3f f_HV=%.3f f_g=%.3f f_p=%.3f f_bb=%.3f f_a=%.2f "
          "f_LU=%.3f f_R=%.3f f_pb=%.3f" %
          (fs["f_w"], fs["f_HV"], fs["f_g"], fs["f_p"], fs["f_bb"], fs["f_a"],
           fs["f_LU"], fs["f_R"], fs["f_pb"]))
    print("   individual checks: width(3.3m)=%.4f grade(+2%%)=%.4f "
          "parking(12/h,2ln)=%.4f bus(8/h)=%.4f area(CBD)=%.2f HV(6%%)=%.4f"
          % (lane_width_factor(3.3), grade_factor(2.0),
             parking_factor(2, 12.0, True), bus_blockage_factor(2, 8.0),
             area_type_factor(True), heavy_vehicle_factor(6.0)))
    print("   turning radius 15 m -> f_R = %.4f" % turning_radius_factor(15.0))
    print("   adjusted saturation flow = %.0f veh/h (2 lanes, base 1900)" % s_adj)
    cap = movement_capacity(s_adj, 30.0, 90.0)
    print("   capacity at g=30 s of a 90 s cycle: c = s g / C = %.0f veh/h" % cap)
    print("   at 600 veh/h: y = q/s = %.4f   X = v/c = %.3f"
          % (flow_ratio(600.0, s_adj), degree_of_saturation(600.0, cap)))

    # ------------------------------------------------------------------ 2
    print("\n-- 2. delay: Webster, HCM and Akcelik --")
    y = [0.30, 0.25]
    L = 10.0
    C0 = webster_optimal_cycle(L, y)
    print("   Webster C0 = (1.5L + 5)/(1 - Y) with L=%.0f, Y=%.2f -> %.2f s"
          % (L, math.fsum(y), C0))
    demands2 = [900.0, 750.0]
    sats2 = [3000.0, 3000.0]
    st2 = [[0], [1]]
    cnum, dnum = webster_optimal_cycle_numeric(demands2, sats2, st2, L,
                                               20.0, 200.0, 0.5)
    print("   fine sweep of total Webster delay -> C* = %.1f s (%.0f veh-s/h); "
          "C0 is %.1f%% away" % (cnum, dnum, 100.0 * abs(cnum - C0) / cnum))
    print("   total Webster delay at C0 = %.0f veh-s/h, at 0.5*C0 = %.0f, "
          "at 2*C0 = %.0f"
          % (webster_total_delay(C0, demands2, sats2, st2, L),
             webster_total_delay(0.5 * C0, demands2, sats2, st2, L),
             webster_total_delay(2.0 * C0, demands2, sats2, st2, L)))
    g_eq = equisaturation_split(C0, L, y)
    d_w = webster_delay(demands2[0], sats2[0], g_eq[0], C0)
    print("   split at C0: %s s -> Webster delay on stage 1 = %.2f s/veh"
          % (["%.2f" % v for v in g_eq], d_w))
    hcm = hcm_control_delay(900.0, 3000.0, 40.0, 90.0, period_h=0.25, k=0.5)
    print("   HCM d = d1*PF + d2 + d3 = %.2f*%.3f + %.2f + %.2f = %.2f s/veh "
          "(X=%.3f, LOS %s)" % (hcm.d1, hcm.progression_factor, hcm.d2, hcm.d3,
                                hcm.control_delay, hcm.x, hcm.los))
    print("   closed-form d1 check: 0.5*C*(1-g/C)^2/(1-min(1,X)g/C) = %.4f"
          % hcm_uniform_delay(90.0, 40.0, hcm.x))
    print("   d2 alone = %.3f, d3 with a 5 veh initial queue = %.3f"
          % (hcm_incremental_delay(hcm.x, hcm.capacity),
             hcm_initial_queue_delay(5.0, hcm.capacity, hcm.x)))
    print("   PF(arrival type 4, g/C=0.44) = %.3f"
          % progression_adjustment_factor(40.0, 90.0, arrival_type=4))
    print("   delay is monotone in v/c: " + "  ".join(
        "X=%.2f:%.1f" % (xx, hcm_control_delay(xx * 1333.3, 3000.0, 40.0, 90.0)
                         .control_delay)
        for xx in (0.30, 0.60, 0.90, 0.98)))
    print("   Akcelik x0 = %.3f, delay = %.2f s/veh"
          % (akcelik_x0(3000.0, 40.0), akcelik_delay(900.0, 3000.0, 40.0, 90.0)))
    n1, n2, nq = average_back_of_queue(900.0, 3000.0, 40.0, 90.0)
    print("   back of queue: N1=%.2f N2=%.2f total=%.2f veh; 95th pct=%.2f veh "
          "= %.1f m" % (n1, n2, nq, queue_length_95(900.0, 3000.0, 40.0, 90.0),
                        queue_length_metres(
                            queue_length_95(900.0, 3000.0, 40.0, 90.0), 7.0)))
    print("   junction average of (12.0, 48.0) s/veh at (500, 900) veh/h = "
          "%.2f s/veh -> LOS %s"
          % (intersection_delay([500.0, 900.0], [12.0, 48.0]),
             level_of_service(intersection_delay([500.0, 900.0],
                                                 [12.0, 48.0]))))
    print("   unsignalised LOS at 26 s/veh = %s"
          % unsignalised_level_of_service(26.0))
    print("   DelayResult keys: %s" % ", ".join(sorted(hcm.as_dict())[:4]))

    # ------------------------------------------------------------------ 3
    print("\n-- 3. conflicts and stage generation (Bron-Kerbosch) --")
    # each arm has a through-plus-nearside-left lane group and a protected
    # offside (right) turn; opposing right turns pass offside to offside
    mvs = standard_four_arm_movements(
        turns=(TurnDirection.THROUGH, TurnDirection.RIGHT))
    conf = conflict_matrix(mvs, drive_on_left=True,
                           opposing_turns_conflict=False)
    n_conf = sum(1 for i in range(len(mvs)) for j in range(i + 1, len(mvs))
                 if conf[i][j])
    print("   %d movements (%s), %d conflicting pairs of %d"
          % (len(mvs), ", ".join(m.label for m in mvs), n_conf,
             len(mvs) * (len(mvs) - 1) // 2))
    comp = compatibility_graph(conf)
    cliques = bron_kerbosch(comp)
    exhaustive = all(
        all(comp[a][b] for a, b in itertools.combinations(cl, 2)) and
        not any(all(comp[v][u] for u in cl)
                for v in range(len(mvs)) if v not in cl)
        for cl in cliques)
    print("   compatibility graph -> %d maximal cliques, largest has %d "
          "movements; each verified maximal by brute force: %s"
          % (len(cliques), len(cliques[0]), exhaustive))
    demand = [520.0, 170.0,      # N: through+left group, offside right turn
              430.0, 140.0,      # E
              500.0, 160.0,      # S
              460.0, 130.0]      # W
    sat = [1900.0, 1650.0] * 4
    cands = maximal_compatible_stages(conf)
    cover = minimum_stage_cover(cands, len(mvs),
                                stage_costs=critical_flow_ratios(cands, demand,
                                                                 sat))
    naive = minimum_stage_cover(cands, len(mvs))
    print("   minimum stage cover uses %d stages (sum of critical y = %.3f; "
          "the cardinality-only cover gives %.3f):"
          % (len(cover), math.fsum(critical_flow_ratios(cover, demand, sat)),
             math.fsum(critical_flow_ratios(naive, demand, sat))))
    for k, st in enumerate(cover):
        print("      stage %d: %-26s y = %.3f"
              % (k + 1, " ".join(movements_in_stage(st, mvs)),
                 stage_flow_ratio(st, demand, sat)))
    print("   every stage is an independent set of the conflict matrix: %s"
          % all(is_conflict_free(st, conf) for st in cover))
    print("   validate_stages problems: %s"
          % (validate_stages(cover, conf) or "none"))

    # ------------------------------------------------------------------ 4
    print("\n-- 4. NEMA ring-barrier structure --")
    phases = nema_eight_phase()
    durations = {1: 15.0, 2: 35.0, 3: 12.0, 4: 28.0,
                 5: 18.0, 6: 32.0, 7: 14.0, 8: 26.0}
    rb = RingBarrierPlan(phases=phases, durations=durations, cycle=90.0)
    print("   rings: %s" % rb.rings())
    print("   barrier groups: %s" % rb.barrier_groups())
    print("   validation problems: %s" % (validate_ring_barrier(rb) or "none"))
    seq = ring_barrier_sequence(rb)
    print("   timeline has %d intervals; first three: %s"
          % (len(seq), ", ".join("%.0f-%.0f s phases %s"
                                 % (s["start"], s["end"], s["phases"])
                                 for s in seq[:3])))
    print("   phase 1 is %s (%s %s) in ring %d, barrier %d"
          % (phases[0].movement, phases[0].approach, phases[0].turn,
             phases[0].ring, phases[0].barrier))
    print("   NEMA map has %d phases" % len(nema_phase_movements()))

    # ------------------------------------------------------------------ 5
    print("\n-- 5. timing constraints --")
    v85 = 13.9                                      # 50 km/h
    yy = yellow_change_interval(v85, grade_pct=-2.0)
    ar = all_red_clearance(14.0, v85)
    ig = intergreen_time(v85, 14.0, grade_pct=-2.0)
    print("   50 km/h, -2%% grade, 14 m crossing: yellow=%.2f s all-red=%.2f s "
          "intergreen=%.1f s" % (yy, ar, ig))
    gped = pedestrian_crossing_time(12.0, 1.2, 3.2, ped_per_cycle=20.0,
                                    crossing_width_m=4.0)
    print("   pedestrian minimum for a 12 m crossing at 1.2 m/s with 20 peds/"
          "cycle = %.2f s" % gped)
    print("   minimum green (queue 6 veh, ped need %.1f s) = %.2f s"
          % (gped, minimum_green_time(6.0, 1900.0, 7.0, 2.0, gped, ig)))
    print("   stage lost time (I=%.1f) = %.2f s; four stages -> L = %.2f s"
          % (ig, stage_lost_time(ig), total_lost_time([ig] * 4)))

    # ------------------------------------------------------------------ 6
    print("\n-- 6. end-to-end fixed-time design --")
    plan = design_fixed_time_plan(demand, sat, stages=cover, intergreens=5.0,
                                  min_green=7.0, name="demo junction")
    auto = design_fixed_time_plan(demand, sat, conflict=conf, intergreens=5.0,
                                  min_green=7.0)
    print("   staging inferred straight from the conflict matrix reproduces "
          "the same %d stages: %s" % (auto.n_stages, auto.stages == plan.stages))
    print("   %d stages, cycle = %.2f s, L = %.2f s, sum(g) = %.9f s "
          "(C - L = %.9f s)"
          % (plan.n_stages, plan.cycle, plan.total_lost_time,
             math.fsum(plan.effective_greens()),
             plan.cycle - plan.total_lost_time))
    for k in range(plan.n_stages):
        print("      stage %d: G=%.2f s I=%.1f s g=%.2f s (%.1f%% of cycle) %s"
              % (k + 1, plan.greens[k], plan.intergreens[k],
                 plan.effective_greens()[k], 100.0 * plan.green_shares()[k],
                 " ".join(movements_in_stage(plan.stages[k], mvs))))
    print("   plan validation: %s" % (plan.validate(conflict=conf) or "none"))
    dres, avg, los = plan_delay(plan, demand, sat)
    print("   junction average control delay = %.2f s/veh -> LOS %s"
          % (avg, los))
    worst = max(dres, key=lambda m: dres[m].x)
    print("   worst movement %s: v/c = %.3f, delay %.1f s/veh, LOS %s"
          % (mvs[worst].label, dres[worst].x, dres[worst].control_delay,
             dres[worst].los))
    ycrit = critical_flow_ratios(plan.stages, demand, sat)
    print("   critical flow ratios %s (Y = %.3f); practical cycle at X=0.9 = "
          "%.1f s" % (["%.3f" % v for v in ycrit], math.fsum(ycrit),
                      practical_cycle_length(ycrit, plan.total_lost_time, 0.90)))
    hand = make_signal_plan([[0, 1], [2, 3]], [30.0, 20.0], [5.0, 5.0])
    print("   hand-built two-stage plan: C = %.1f s, g = %s, valid = %s"
          % (hand.cycle, hand.effective_greens(), not hand.validate()))

    # ------------------------------------------------------------------ 7
    print("\n-- 7. the LP kernel --")
    lp = linear_program([-3.0, -5.0], A_ub=[[1.0, 0.0], [0.0, 2.0],
                                            [3.0, 2.0]],
                        b_ub=[4.0, 12.0, 18.0])
    print("   classic textbook LP max 3x+5y: x = %s, objective = %.4f, "
          "status %s, %d iterations"
          % (["%.3f" % v for v in lp.x], -lp.fun, lp.status, lp.n_iterations))
    print("   LPResult.success = %s" % lp.success)

    # ------------------------------------------------------------------ 8
    print("\n-- 8. split optimisation --")
    Cc, Ll = plan.cycle, plan.total_lost_time
    eq = equisaturation_split(Cc, Ll, ycrit, min_green=7.0)
    wb = webster_split(Cc, Ll, ycrit, min_green=7.0)
    al, mu = allsop_lp_split(Cc, Ll, ycrit, min_green=7.0)
    p0 = smith_p0_split(Cc, Ll, plan.stages, demand, sat, min_green=7.0)
    dm, dval = delay_minimising_split(Cc, Ll, plan.stages, demand, sat,
                                      min_green=7.0)
    print("   equisaturation : %s" % ["%.2f" % v for v in eq])
    print("   Webster        : %s (identical: %s)"
          % (["%.2f" % v for v in wb],
             all(abs(wb[i] - eq[i]) < 1e-9 for i in range(len(eq)))))
    print("   Allsop LP      : %s   reserve multiplier mu = %.4f"
          % (["%.2f" % v for v in al], mu))
    print("   Smith P0       : %s   pressures %s"
          % (["%.2f" % v for v in p0],
             ["%.0f" % smith_p0_pressure(plan.stages[k], demand, sat, p0[k], Cc)
              for k in range(len(p0))]))
    print("   delay-minimal  : %s   total delay = %.0f veh-s/h"
          % (["%.2f" % v for v in dm], dval))
    print("   every split sums to C - L = %.6f s: %s"
          % (Cc - Ll, ["%.6f" % math.fsum(g) for g in (eq, al, p0, dm)]))
    sym = apportion_green(60.0, [0.25, 0.25], min_green=5.0)
    print("   symmetric two-stage apportionment of 60 s: %s (exactly equal: %s)"
          % (sym, sym[0] == sym[1]))

    # ------------------------------------------------------------------ 9
    print("\n-- 9. platoon dispersion and offsets --")
    prof = cyclic_flow_profile(600.0, 90.0, 1.0, platoon_start=10.0,
                               platoon_length=20.0, platoon_share=0.8)
    print("   arrival profile: %d steps, %.2f veh/cycle, peak %.0f veh/h"
          % (prof.n_steps, prof.vehicles(), max(prof.flows)))
    disp = robertson_dispersion(prof, travel_time=25.0)
    print("   after 25 s of Robertson dispersion: peak %.0f veh/h, "
          "vehicles conserved %.4f -> %.4f"
          % (max(disp.flows), prof.vehicles(), disp.vehicles()))
    dep, qs = departure_profile(disp, green_start=20.0, green_length=40.0,
                                sat_flow=1800.0)
    print("   stopline discharge: %.4f veh/cycle out, max queue %.2f veh, "
          "delay %.0f veh-s/cycle"
          % (dep.vehicles(), max(qs),
             profile_delay(disp, 20.0, 40.0, 1800.0)))
    print("   profile shifted 5 steps keeps its total: %.4f veh"
          % prof.shifted(5).vehicles())
    tables = [link_delay_table(dep, 22.0, 40.0, 1800.0) for _ in range(2)]
    dp = optimise_offsets_dp(tables, step=1.0)
    print("   exact DP over 2 links: offsets %s, delay %.0f veh-s "
          "(%d transitions evaluated)"
          % (["%.0f" % o for o in dp.offsets], dp.total_delay, dp.n_evaluations))
    zero = optimise_offsets([700.0] * 3, [1800.0] * 3, [40.0] * 3,
                            [25.0, 30.0], 90.0, step=2.0,
                            initial_offsets=[0.0, 0.0, 0.0], max_rounds=0)
    arte = optimise_offsets([700.0] * 3, [1800.0] * 3, [40.0] * 3,
                            [25.0, 30.0], 90.0, step=2.0,
                            initial_offsets=[0.0, 0.0, 0.0])
    print("   TRANSYT hill climb on 3 signals: all-zero offsets cost %.0f "
          "veh-s; optimised to %s giving %.0f veh-s (-%.1f%%, %d simulations)"
          % (zero.total_delay, ["%.0f" % o for o in arte.offsets],
             arte.total_delay,
             100.0 * (zero.total_delay - arte.total_delay) / zero.total_delay,
             arte.n_evaluations))

    # ----------------------------------------------------------------- 10
    print("\n-- 10. bandwidth: green waves, MAXBAND and MULTIBAND --")
    Cb = 80.0
    dists = [402.0, 536.0]
    speed = 13.4
    gw = green_wave_offsets(dists, speed, Cb)
    tt_out = [d / speed for d in dists]
    greens_b = [40.0, 38.0, 42.0]
    b_out, start_out = arterial_bandwidth(gw, greens_b, tt_out, Cb, "outbound")
    b_in_gw = arterial_bandwidth(gw, greens_b, tt_out, Cb, "inbound")[0]
    print("   green wave at %.1f m/s (%.0f s and %.0f s between signals): "
          "offsets %s" % (speed, tt_out[0], tt_out[1],
                          ["%.1f" % o for o in gw]))
    print("   outbound band = %.2f s from %.1f s (efficiency %.3f) -- it equals "
          "the smallest green %.1f s; inbound gets only %.2f s"
          % (b_out, start_out, bandwidth_efficiency(b_out, Cb), min(greens_b),
             b_in_gw))
    mb = maxband(greens_b, tt_out, tt_out, Cb, inbound_weight=1.0, m_range=2)
    chk_out = arterial_bandwidth(mb.offsets, greens_b, tt_out, Cb, "outbound")[0]
    chk_in = arterial_bandwidth(mb.offsets, greens_b, tt_out, Cb, "inbound")[0]
    print("   MAXBAND: offsets %s, b_out=%.2f s b_in=%.2f s (total %.2f s "
          "beats the green wave's %.2f s), loop integers %s"
          % (["%.1f" % o for o in mb.offsets], mb.outbound_band,
             mb.inbound_band, mb.outbound_band + mb.inbound_band,
             b_out + b_in_gw, mb.loop_integers))
    print("   independent re-measurement of those offsets: out %.2f s, in "
          "%.2f s (LP and geometry agree: %s)"
          % (chk_out, chk_in,
             abs(chk_out - mb.outbound_band) < 1e-6 and
             abs(chk_in - mb.inbound_band) < 1e-6))
    mub = multiband(greens_b, tt_out, tt_out, Cb,
                    flows_out=[900.0, 1100.0], flows_in=[700.0, 750.0],
                    m_range=2)
    print("   MULTIBAND per-link bands: outbound %s, inbound %s "
          "(flow-weighted objective %.4f of the cycle)"
          % (["%.2f" % b for b in (mub.outbound_bands or [])],
             ["%.2f" % b for b in (mub.inbound_bands or [])], mub.objective))
    print("   BandwidthResult efficiency = %.3f" % mb.efficiency())

    # ----------------------------------------------------------------- 11
    print("\n-- 11. actuated control --")
    arr = poisson_arrivals(600.0, 120.0, seed=7)
    act = simulate_actuated_phase(arr, min_green=7.0, max_green=50.0,
                                  unit_extension=3.0)
    print("   %d arrivals at 600 veh/h; phase ran %.2f s and ended by %s "
          "(%d extensions, %d vehicles served)"
          % (len(arr), act.green, act.terminated_by, act.extensions,
             act.vehicles_served))
    heavy = simulate_actuated_phase(poisson_arrivals(1400.0, 120.0, seed=3),
                                    7.0, 40.0, 3.0)
    print("   at 1400 veh/h the same phase ends by %s at %.1f s"
          % (heavy.terminated_by, heavy.green))
    vd = simulate_actuated_phase(poisson_arrivals(1000.0, 120.0, seed=5),
                                 7.0, 60.0, 4.0, volume_density=True)
    print("   volume density (gap 4.0 -> 2.0 s): green %.2f s by %s; "
          "gap at t=20 s is %.2f s"
          % (vd.green, vd.terminated_by, volume_density_gap(20.0, 4.0, 2.0)))
    n_rep = 200
    mean_sim = math.fsum(simulate_actuated_phase(
        poisson_arrivals(600.0, 300.0, seed=1000 + i), 3.0, 120.0, 3.0).green
        for i in range(n_rep)) / n_rep
    print("   with the minimum green set to the unit extension (the classical "
          "assumption), the mean of %d simulated phases = %.3f s against the "
          "analytic (e^{lu}-1)/l = %.3f s"
          % (n_rep, mean_sim, expected_actuated_green(600.0, 3.0, 3.0, 120.0)))

    # ----------------------------------------------------------------- 12
    print("\n-- 12. max-pressure (backpressure) control --")
    mp_moves = [(0, None), (1, None), (2, None), (3, None)]
    mp_stages = [[0, 2], [1, 3]]
    mp_sat = [1800.0, 1800.0, 1800.0, 1800.0]
    mp_dem = [600.0, 500.0, 620.0, 480.0]
    rho, shares = max_pressure_stability_margin(mp_moves, mp_stages, mp_sat,
                                                mp_dem)
    print("   stability margin rho = %.3f (demand can grow %.0f%% before the "
          "region is left); stage shares %s"
          % (rho, 100.0 * (rho - 1.0), ["%.3f" % v for v in shares]))
    sim = simulate_max_pressure(mp_moves, mp_stages, mp_sat, mp_dem,
                                horizon_s=3600.0, control_interval=15.0,
                                initial_queues=[8.0, 3.0, 6.0, 2.0],
                                lost_time=4.0)
    print("   1 h simulation: max total queue %.2f veh, final %.2f veh, "
          "mean %.2f veh -- bounded: %s"
          % (sim.max_queue, sim.final_total, sim.mean_queue, sim.is_bounded()))
    k_star, p_star = max_pressure_stage(mp_stages, [10.0, 2.0, 9.0, 1.0],
                                        mp_moves, mp_sat)
    brute = max(range(len(mp_stages)),
                key=lambda k: stage_pressure(mp_stages[k], [10.0, 2.0, 9.0, 1.0],
                                             mp_moves, mp_sat))
    print("   with queues (10, 2, 9, 1) the policy picks stage %d "
          "(pressure %.0f); brute force agrees: %s"
          % (k_star, p_star, k_star == brute))
    print("   movement 0 pressure = %.1f veh (exit movement -> its own queue)"
          % movement_pressure(0, [10.0, 2.0, 9.0, 1.0], mp_moves))
    over = simulate_max_pressure(mp_moves, mp_stages, mp_sat,
                                 [1500.0, 1400.0, 1500.0, 1400.0],
                                 horizon_s=3600.0, control_interval=15.0,
                                 lost_time=4.0)
    print("   oversaturated demand (rho=%.2f): queue grows to %.0f veh, "
          "bounded: %s"
          % (max_pressure_stability_margin(
              mp_moves, mp_stages, mp_sat,
              [1500.0, 1400.0, 1500.0, 1400.0])[0],
             over.max_queue, over.is_bounded()))

    # ----------------------------------------------------------------- 13
    print("\n-- 13. self-organising (Laemmer-Helbing) control --")
    pi, ghat = laemmer_index(8.0, 600.0, 1800.0, 5.0)
    print("   a stage with 8 veh queued at 600 veh/h and s=1800 veh/h: "
          "clearing time %.2f s, index %.0f veh/h" % (ghat, pi))
    k_l, regime, idx = laemmer_select_stage(mp_stages, [12.0, 2.0, 3.0, 1.0],
                                            mp_dem, mp_sat, 5.0)
    print("   selection picks stage %d in the %s regime (indices %s)"
          % (k_l, regime, ["%.0f" % v for v in idx]))
    k_s, regime_s, _ = laemmer_select_stage(mp_stages, [1.0, 2.0, 3.0, 1.0],
                                            mp_dem, mp_sat, 5.0,
                                            waiting_times=[0.0, 200.0, 0.0, 0.0],
                                            max_waiting=120.0)
    print("   after a 200 s wait on movement 1 the regime becomes %s "
          "(stage %d)" % (regime_s, k_s))
    lae = simulate_laemmer(mp_stages, mp_dem, mp_sat, horizon_s=1800.0,
                           setup_time=5.0, control_interval=1.0, min_green=5.0)
    print("   30 min: %d switches, mean queue %.2f veh, max %.2f veh, "
          "longest wait %.0f s"
          % (lae["switches"], lae["meanQueue"], lae["maxQueue"],
             lae["maxWait"]))

    # ----------------------------------------------------------------- 14
    print("\n-- 14. transit signal priority --")
    bus_stage = 0
    win = _stage_windows(plan)[bus_stage]
    late_bus = win[1] + 4.0
    req = TSPRequest(stage=bus_stage, arrival=late_bus, passengers=55,
                     lateness_s=90.0)
    ext = tsp_green_extension(plan, bus_stage, late_bus, max_extension=10.0)
    print("   stage 1 green runs %.1f-%.1f s; a bus arrives at %.1f s"
          % (win[0], win[1], late_bus))
    print("   green extension: +%.2f s -> bus delay %.1f s becomes %.1f s "
          "(cycle still %.2f s)"
          % (ext.adjustment, ext.bus_delay_without, ext.bus_delay_with,
             ext.plan.cycle))
    early_bus = (win[0] - 6.0) % plan.cycle
    early = tsp_early_start(plan, bus_stage, early_bus, max_truncation=10.0)
    print("   early start for a bus at %.1f s: -%.2f s of the previous stage, "
          "saving %.1f s" % (early_bus, early.adjustment, early.delay_saved))
    auto = apply_transit_priority(plan, req)
    print("   apply_transit_priority chose '%s' (saved %.1f s)"
          % (auto.strategy, auto.delay_saved))
    ins = tsp_phase_insertion(plan, [0], duration=8.0, intergreen=5.0)
    print("   bus-only phase inserted: %d stages, cycle %.2f s (unchanged: %s)"
          % (ins.plan.n_stages, ins.plan.cycle,
             abs(ins.plan.cycle - plan.cycle) < 1e-9))
    print("   TSPRequest: %d passengers, %.0f s late" % (req.passengers,
                                                         req.lateness_s))

    # ----------------------------------------------------------------- 15
    print("\n-- 15. pedestrian phasing --")
    need = pedestrian_crossing_time(10.0, 1.2, 3.2)
    ok, deficit = concurrent_pedestrian_check(plan.effective_greens()[0],
                                              plan.intergreens[0], need)
    print("   a 10 m crossing needs %.2f s; stage 1 offers %.2f s -> "
          "concurrent: %s (deficit %.2f s)"
          % (need, plan.effective_greens()[0] + plan.intergreens[0], ok,
             deficit))
    lpi_plan = leading_pedestrian_interval(plan, 0, lpi=4.0)
    print("   4 s leading pedestrian interval: stage 1 effective green "
          "%.2f -> %.2f s, cycle unchanged (%.2f s), valid: %s"
          % (plan.effective_greens()[0], lpi_plan.effective_greens()[0],
             lpi_plan.cycle, not lpi_plan.validate()))
    ped_plan = exclusive_pedestrian_stage(plan, [8], ped_time=12.0,
                                          intergreen=5.0, min_green=7.0)
    print("   exclusive pedestrian stage: %d stages, cycle %.2f s, greens %s"
          % (ped_plan.n_stages, ped_plan.cycle,
             ["%.1f" % g for g in ped_plan.effective_greens()]))
    print("   Barnes dance across 12 m x 9 m = %.2f s (diagonal %.1f m)"
          % (barnes_dance_time(12.0, 9.0), math.hypot(12.0, 9.0)))
    print("   pedestrian delay with a %.0f s walk in a %.0f s cycle = %.2f s"
          % (12.0, plan.cycle, pedestrian_delay(plan.cycle, 12.0)))

    # ----------------------------------------------------------------- 16
    print("\n-- 16. unsignalised junctions: gap acceptance --")
    vc = 600.0
    cp = hcm_potential_capacity(vc, 6.5, 3.5)
    p0_major = rank_impedance_factor(150.0, 1200.0)
    cm = twsc_movement_capacity(cp, [p0_major])
    print("   minor left, v_c=%.0f veh/h, t_c=6.5 s, t_f=3.5 s: c_p=%.0f veh/h"
          % (vc, cp))
    print("   impedance from a rank-2 movement p0=%.3f -> c_m=%.0f veh/h; "
          "delay at 120 veh/h = %.1f s/veh (LOS %s)"
          % (p0_major, cm, twsc_control_delay(120.0, cm),
             unsignalised_level_of_service(twsc_control_delay(120.0, cm))))
    print("   Siegloch capacity for the same gap parameters = %.0f veh/h"
          % siegloch_capacity(vc, 6.5, 3.5))
    tcs, tfs = siegloch_critical_gap([7.0, 10.0, 13.0, 16.0, 19.0],
                                     [1.0, 2.0, 3.0, 4.0, 5.0])
    print("   Siegloch regression on synthetic data -> t_c=%.2f s, t_f=%.2f s"
          % (tcs, tfs))
    acc = [4.2, 5.1, 5.6, 6.3, 6.9, 7.4, 8.1, 9.0, 5.9, 6.6]
    rej = [2.1, 3.4, 4.0, 4.6, 5.2, 5.8, 6.2, 6.7, 3.9, 4.4]
    print("   Raff critical gap = %.2f s; maximum-likelihood (lognormal) "
          "= %.2f s" % (raff_critical_gap(acc, rej),
                        maximum_likelihood_critical_gap(acc, rej)[0]))
    tq = tanner_capacity(vc, 6.5, 3.5, min_headway=2.0)
    back = tanner_critical_gap(tq, vc, 3.5, min_headway=2.0)
    print("   Tanner capacity = %.1f veh/h; inverting it recovers t_c = %.4f s "
          "(input 6.5)" % (tq, back))

    # ----------------------------------------------------------------- 17
    print("\n-- 17. roundabout capacity --")
    for qc in (0.0, 400.0, 800.0, 1200.0):
        print("   Q_c=%4.0f pcu/h: HCM %.0f   Brilon-Wu %.0f   Kimber %.0f veh/h"
              % (qc, hcm_roundabout_capacity(qc),
                 brilon_wu_capacity(qc),
                 kimber_capacity(qc, KimberGeometry())))
    kg = KimberGeometry(entry_width=7.0, approach_half_width=3.5,
                        flare_length=20.0, inscribed_diameter=45.0,
                        entry_angle=25.0, entry_radius=25.0)
    kk, FF, ffc = kimber_geometry(kg)
    print("   wider entry: k=%.4f F=%.1f f_c=%.4f -> Q_e(600) = %.0f pcu/h "
          "(linear check %.0f)"
          % (kk, FF, ffc, kimber_capacity(600.0, kg),
             max(0.0, kk * (FF - ffc * 600.0))))
    ce = kimber_capacity(600.0, kg)
    print("   entry delay at 500 veh/h against that capacity = %.1f s/veh "
          "(LOS %s)" % (roundabout_delay(500.0, ce),
                        unsignalised_level_of_service(roundabout_delay(500.0,
                                                                       ce))))
    print("   HCM with explicit gap parameters (t_c=5.0, t_f=3.2): %.0f veh/h"
          % hcm_roundabout_capacity(600.0, critical_gap=5.0, follow_up=3.2))

    # ----------------------------------------------------------------- 18
    print("\n-- 18. payloads --")
    payload = build_signal_control_payload(plan, demand, sat, mvs)
    ui = build_a2ui_payload(plan, demand, sat, mvs, title="Demo junction")
    print("   core payload: %d top-level keys, %d movement rows, %d stages"
          % (len(payload), len(payload["movements"]), len(payload["stages"])))
    print("   a2ui payload: %d KPI tiles, timeline of %d stages, table with "
          "%d columns" % (len(ui["kpis"]), len(ui["timeline"]),
                          len(ui["table"]["columns"])))
    print("   critical movement in the UI payload: %s at v/c %.3f"
          % (ui["criticalMovement"]["label"], ui["criticalMovement"]["vOverC"]))

    # ----------------------------------------------------------------- 19
    print("\n-- 19. result containers --")
    containers = (SaturationFlowFactors, DelayResult, Movement, NemaPhase,
                  RingBarrierPlan, SignalPlan, LPResult, PlatoonProfile,
                  OffsetResult, BandwidthResult, ActuatedResult,
                  MaxPressureResult, TSPRequest, TSPResult, KimberGeometry)
    live = ((factors, SaturationFlowFactors), (hcm, DelayResult),
            (mvs[0], Movement), (phases[0], NemaPhase), (rb, RingBarrierPlan),
            (plan, SignalPlan), (lp, LPResult), (prof, PlatoonProfile),
            (arte, OffsetResult), (mb, BandwidthResult), (act, ActuatedResult),
            (sim, MaxPressureResult), (req, TSPRequest), (ext, TSPResult),
            (kg, KimberGeometry))
    print("   %d dataclass containers: %s"
          % (len(containers), ", ".join(t.__name__ for t in containers)))
    print("   every one was returned live by the run above: %s"
          % all(isinstance(o, t) for o, t in live))

    print("\n-- 20. validation --")
    try:
        webster_optimal_cycle(10.0, [0.6, 0.5])
    except SignalControlError as exc:
        print("   over-capacity junction rejected: %s" % exc)
    try:
        make_signal_plan([[0], [1]], [30.0, 20.0], [5.0])
    except SignalControlError as exc:
        print("   mismatched plan rejected: %s" % exc)

    print("\n" + "=" * 70)
    print("  demo OK")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
