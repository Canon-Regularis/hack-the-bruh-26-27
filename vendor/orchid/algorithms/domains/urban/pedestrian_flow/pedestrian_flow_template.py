"""
Pedestrian Flow -- density fields, crowding, crossings and walking level of service
===================================================================================

**People are a continuum long before they are a crowd.**  This module treats
walkers the way hydraulics treats water: a density field ``rho(x, t)`` on a
pavement, a speed that falls as the density rises, and a conservation law that
never loses a single person.  It is the conservation-law layer of the urban
stack -- the place where "how many people per square metre" becomes "how long
does the crossing take", "what level of service is this footway", and "how long
does the stand take to empty".

Where it sits among its siblings:

* ``traffic_flow`` is the *vehicle* conservation layer.  Cars have a different
  fundamental diagram (triangular, lane-based, ~1800 veh/h/lane), obey lane
  discipline, and use node rules built around turning movements and signal
  capacity.  Pedestrians have a smooth concave diagram in persons per metre of
  width per second, walk in two dimensions, form counterflow lanes, and route
  themselves by a potential field rather than by a link graph.
* ``active_travel`` is the *network design quality* layer -- level of traffic
  stress, severance, route directness, provision audits.  It asks whether a
  walking network is good.  This module asks how many people fit on it, how
  fast they move, and when it fails.

Everything here is pure standard-library Python: the Godunov solver, the
fast-marching eikonal solver, the social-force integrator, the floor-field
cellular automaton and every special value are built by hand.

Layers
------
1.  **Fundamental diagrams** -- Weidmann (Kladek), Older, Fruin, SFPE,
    Predtechenskii-Milinskii (tabulated) and Virkler (Pipes-Munjal), each with
    an *analytic* capacity and critical density, plus speed / flow curves.
2.  **Speed-density modulation** -- ``v = v0 * phi(rho)`` with monotone
    decreasing ``phi`` (linear / exponential / power), Tobler's slope law.
3.  **Conservation PDE** -- ``d(rho)/dt + d(rho v)/dx = q`` solved by GODUNOV in
    the sending (demand) / receiving (supply) formulation, with an exact people
    ledger, Riemann solutions and Rankine-Hugoniot shock speeds.
4.  **Node coupling** -- merges by priority water-filling, FIFO diverges,
    general junction transfer with supply rationing, and whole-network
    simulation that conserves people across junctions.
5.  **Bidirectional flow** -- counterflow friction, the lane-formation order
    parameter and a seeded lateral-sorting model.
6.  **Eikonal & Hughes** -- fast-marching solver for the walking potential,
    steepest-descent paths, and Hughes' continuum crowd model (eikonal potential
    coupled to the continuity equation) on a 2-D grid.
7.  **Social force** -- Helbing-Molnar driving / social / wall / body / friction
    terms with symplectic (semi-implicit Euler and velocity-Verlet) integrators
    and the analytic two-agent equilibrium separation.
8.  **Floor-field CA** -- static + dynamic floor fields, parallel update,
    friction-parameter conflict resolution, fully seeded.
9.  **Level of service & crowd risk** -- Fruin A-F for walkways, queueing areas
    and stairways; crowd density bands, crowd pressure and crush risk.
10. **Crossings & delay** -- Adams' gap-acceptance delay, signalised delay
    ``(C-g)^2 / (2C)``, two-stage crossings with a central refuge, pedestrian
    minimum green, phase timings and platoon dispersion.
11. **Route choice** -- multinomial and path-size logit over walk routes with
    stress, comfort and gradient attributes, plus the logsum accessibility.
12. **Evacuation & accumulation** -- effective widths, exit capacity,
    flow-limited egress, the Predtechenskii-Milinskii staged method, cumulative
    arrival/departure accumulation and stadium egress profiles.
13. **Cross-model comparison** -- the same corridor solved by the PDE, the CA
    and the social-force model.
14. **Payloads** -- JSON-able dicts for a frontend.

Conventions
-----------
* Density ``rho`` is persons per square metre (P/m^2); speed ``v`` is m/s; the
  *specific flow* ``q = rho * v`` is persons per metre of width per second
  (P/m/s).  Total person flow through a facility is ``q * width`` in P/s.
* Time is seconds, length metres, delay seconds per person.
* Every ANALYTIC FD (linear / power / Kladek / SFPE) is concave-unimodal in
  ``q(rho)``, so the demand/supply (sending / receiving) decomposition of the
  Godunov flux is exact for them.  The tabulated Predtechenskii-Milinskii
  diagram is *empirical* and is NOT globally concave: the segment D = 0.40-0.50
  carries one interior kink (a local maximum of ``q`` at rho = 4.857 P/m^2,
  0.087 % above the flow at the next table node), so its sending function dips
  by 2.4e-3 P/m/s over a 0.14 P/m^2 window before rising again to the true
  capacity at rho = 5.8.  That is the published data, not a transcription error;
  the ledger, the uniform state and ``godunov_flux(rho, rho) == q(rho)`` stay
  exact regardless, and the flux never produces a negative or super-jam density.
* Nothing is clamped inside the PDE update: positivity and the jam bound come
  out of the demand/supply flux itself under the CFL condition, which is what
  makes the people ledger exact.  Boundary conditions and junction transfers are
  quoted as TOTAL person flows (P/s) and the solver carries them in those units
  all the way into the cell update, so a uniform state is held bit-exactly.
  The CFL condition is ENFORCED, not merely documented: ``cfl_timestep``
  refuses ``cfl > 1`` and ``simulate_walkway`` / ``simulate_network`` refuse an
  explicit ``dt`` above the same limit.  Above it the update leaves [0,
  rho_jam] while the ledger still closes, which is the worst of both worlds.
* A link END may carry more than one role -- a footway can feed a junction and
  also discharge to the street, or feed two junctions.  Each end therefore has
  ONE budget per step, its downstream demand or its upstream supply, and the
  consumers draw from what is left in the order junctions, entries, exits.
  With one role per end (the ordinary case) the remaining budget is the full
  budget and the update is bit-identical to a single-role network.
* Absolute epsilons are kept off DIMENSIONED quantities.  Densities, flows,
  speeds and capacities are compared against the problem's own magnitude, so
  the whole module is covariant under a change of units: rescaling every
  density, speed and length rescales every answer exactly.
* Everything stochastic takes an explicit ``seed`` and uses ``random.Random``,
  so results are byte-identical across runs and independent of PYTHONHASHSEED.
* British spelling in prose; SciPy/standard names where parity exists.
"""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple)

__version__ = "1.0.0"

Density = List[float]
Flow = List[float]
Grid = List[List[float]]
Mask = List[List[bool]]
Cell = Tuple[int, int]

_EPS = 1e-12
_INF = float("inf")

__all__ = [
    # errors
    "PedestrianFlowError",
    # fundamental diagrams
    "PedestrianFD", "fd_weidmann", "fd_older", "fd_fruin", "fd_sfpe",
    "fd_predtechenskii_milinskii", "fd_virkler", "get_fd", "FD_NAMES",
    "fd_speed_curve", "fd_flow_curve", "max_speed_slope",
    # speed-density modulation
    "phi_linear", "phi_exponential", "phi_power", "speed_from_phi",
    "tobler_walking_speed",
    # conservation pde (godunov)
    "WalkwayLink", "WalkwaySimulation", "demand_flow", "supply_flow",
    "godunov_flux", "godunov_step", "simulate_walkway", "cfl_timestep",
    "max_wave_speed", "RiemannSolution", "riemann_solution",
    "riemann_shock_speed", "front_position", "walkway_travel_time",
    # node coupling
    "NodeFlowResult", "merge_flows", "diverge_flows", "node_transfer",
    "Junction", "PedestrianNetwork", "NetworkSimulation", "simulate_network",
    # bidirectional flow
    "counterflow_penalty", "bidirectional_speeds", "bidirectional_capacity",
    "lane_formation_index", "simulate_lane_formation", "LaneFormationResult",
    # eikonal & hughes
    "fast_marching", "eikonal_descent_direction", "potential_descent_path",
    "HughesGrid", "HughesResult", "simulate_hughes",
    # social force
    "SocialForceParams", "Pedestrian", "Wall", "social_forces",
    "social_force_step", "simulate_social_force", "SocialForceResult",
    "social_force_equilibrium_separation",
    # floor-field cellular automaton
    "FloorFieldCA", "CAResult", "static_floor_field",
    # level of service & crowd risk
    "LOSResult", "fruin_los", "fruin_los_from_density", "los_bands",
    "CrowdBand", "crowd_density_band", "crowd_pressure", "crush_risk",
    # crossings & delay
    "critical_gap", "gap_acceptance_probability", "adams_delay",
    "signalised_pedestrian_delay", "two_stage_crossing_delay",
    "CrossingDelayResult", "crossing_delay", "pedestrian_delay_los",
    "pedestrian_minimum_green", "PedestrianPhase", "pedestrian_phase_times",
    "PlatoonResult", "platoon_dispersion",
    # route choice
    "WalkRoute", "RouteCoefficients", "route_utility", "multinomial_logit",
    "path_size_logit", "logsum_accessibility", "RouteChoiceResult",
    # evacuation & accumulation
    "effective_width", "exit_capacity", "EgressResult", "egress_time",
    "PMEgressResult", "predtechenskii_milinskii_egress", "evacuation_curve",
    "AccumulationResult", "accumulation_profile", "stadium_egress_profile",
    "safety_margin",
    # cross-model comparison
    "CrossModelResult", "cross_model_egress",
    # payloads
    "build_pedestrian_flow_payload", "build_a2ui_payload",
]


# ===========================================================================
# 0. VALIDATION HELPERS
# ===========================================================================
class PedestrianFlowError(ValueError):
    """Invalid input to a pedestrian-flow routine."""


def _as_float(x: Any, name: str = "value") -> float:
    """Coerce to a finite float or raise."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise PedestrianFlowError("%s must be a real number (got %r)." % (name, x))
    if math.isnan(v) or math.isinf(v):
        raise PedestrianFlowError("%s must be finite (got %r)." % (name, x))
    return v


def _as_floats(xs: Sequence[Any], name: str = "values") -> List[float]:
    """Coerce a sequence to a list of finite floats or raise."""
    try:
        out = [float(v) for v in xs]
    except (TypeError, ValueError):
        raise PedestrianFlowError("%s must be a sequence of real numbers." % name)
    for v in out:
        if math.isnan(v) or math.isinf(v):
            raise PedestrianFlowError("%s contains non-finite values." % name)
    return out


def _positive(x: Any, name: str = "value") -> float:
    """Coerce to a strictly positive finite float or raise."""
    v = _as_float(x, name)
    if v <= 0.0:
        raise PedestrianFlowError("%s must be > 0 (got %.6g)." % (name, v))
    return v


def _non_negative(x: Any, name: str = "value") -> float:
    """Coerce to a non-negative finite float or raise."""
    v = _as_float(x, name)
    if v < 0.0:
        raise PedestrianFlowError("%s must be >= 0 (got %.6g)." % (name, v))
    return v


def _in_unit(x: Any, name: str = "value") -> float:
    """Coerce to a float in [0, 1] or raise."""
    v = _as_float(x, name)
    if v < 0.0 or v > 1.0:
        raise PedestrianFlowError("%s must lie in [0, 1] (got %.6g)." % (name, v))
    return v


def _positive_int(x: Any, name: str = "count") -> int:
    """Coerce to a strictly positive int or raise."""
    try:
        v = int(x)
    except (TypeError, ValueError):
        raise PedestrianFlowError("%s must be an integer (got %r)." % (name, x))
    if v <= 0:
        raise PedestrianFlowError("%s must be > 0 (got %d)." % (name, v))
    return v


def _interp(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    """Piecewise-linear interpolation with flat extrapolation at both ends."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    k = bisect.bisect_right(list(xs), x) - 1
    if k >= len(xs) - 1:
        return ys[-1]
    span = xs[k + 1] - xs[k]
    if span <= _EPS:
        return ys[k]
    t = (x - xs[k]) / span
    return ys[k] + t * (ys[k + 1] - ys[k])


def _bisect_root(fn: Callable[[float], float], lo: float, hi: float,
                 iters: int = 200) -> float:
    """Bisection root of a sign-changing continuous function on [lo, hi]."""
    flo = fn(lo)
    fhi = fn(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise PedestrianFlowError("root is not bracketed on [%.6g, %.6g]." % (lo, hi))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = fn(mid)
        if fm == 0.0:
            return mid
        if flo * fm < 0.0:
            hi = mid
            fhi = fm
        else:
            lo = mid
            flo = fm
        if hi - lo < 1e-15 * max(1.0, abs(hi)):
            break
    return 0.5 * (lo + hi)


def _mean(xs: Sequence[float]) -> float:
    """Arithmetic mean using fsum."""
    if not xs:
        return 0.0
    return math.fsum(xs) / len(xs)


def _quantile(xs: Sequence[float], p: float) -> float:
    """Linear-interpolation quantile of a sample (p in [0, 1])."""
    if not xs:
        raise PedestrianFlowError("quantile of an empty sample.")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = _in_unit(p, "p") * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    t = pos - lo
    return s[lo] + t * (s[hi] - s[lo])


# ===========================================================================
# 1. PEDESTRIAN FUNDAMENTAL DIAGRAMS
# ===========================================================================
# Every diagram is a speed-density law v(rho) with rho in P/m^2 and v in m/s.
# The specific flow is q(rho) = rho * v(rho) in P/m/s.  Each family below has a
# closed-form capacity except Weidmann's Kladek law, whose stationarity
# condition dq/drho = 0 is transcendental and is solved to machine precision by
# bisection on the ANALYTIC derivative (never on a sampled grid).

# Predtechenskii-Milinskii horizontal-path table: area fraction D (dimensionless)
# against speed in metres per minute.  D = rho * f where f is the projected
# horizontal area of one person (0.100 m^2 street clothes, 0.113 mid-season,
# 0.125 winter).  PM is the "fast" model of the literature: its specific flows
# at high density sit well above Weidmann/SFPE, which is a documented feature of
# the method and not a transcription error.
_PM_D = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.92)
_PM_V_M_PER_MIN = (100.0, 80.0, 60.0, 47.0, 40.0, 33.0, 28.0, 23.0, 19.0,
                   15.0, 0.0)


@dataclass
class PedestrianFD:
    """A pedestrian speed-density law with an analytic capacity and critical density."""

    name: str
    kind: str                       # linear | power | kladek | sfpe | table
    v_free: float                   # speed as rho -> 0 (m/s)
    rho_jam: float                  # density at which v = 0 (P/m^2)
    gamma: float = 0.0              # kladek shape parameter (P/m^2)
    exponent: float = 1.0           # power-law (Pipes-Munjal) exponent
    rho_plateau: float = 0.0        # density below which v is constant
    k_ref: float = 0.0              # sfpe reference speed k (m/s)
    area_per_person: float = 0.1    # table models: f (m^2 per person)
    table_rho: Tuple[float, ...] = ()
    table_v: Tuple[float, ...] = ()
    source: str = ""
    _cap: Optional[Tuple[float, float]] = field(default=None, repr=False,
                                                compare=False)

    # -- speed / flow ------------------------------------------------------
    def speed(self, rho: float) -> float:
        """Walking speed (m/s) at density ``rho`` (P/m^2)."""
        r = _as_float(rho, "rho")
        if r <= 0.0:
            return self.v_free
        if r >= self.rho_jam:
            return 0.0
        kind = self.kind
        if kind == "linear":
            return self.v_free * (1.0 - r / self.rho_jam)
        if kind == "power":
            return self.v_free * (1.0 - (r / self.rho_jam) ** self.exponent)
        if kind == "kladek":
            arg = -self.gamma * (1.0 / r - 1.0 / self.rho_jam)
            return max(0.0, self.v_free * (1.0 - math.exp(arg)))
        if kind == "sfpe":
            if r <= self.rho_plateau:
                return self.v_free
            return max(0.0, self.k_ref * (1.0 - r / self.rho_jam))
        if kind == "table":
            if r <= self.rho_plateau:
                return self.v_free
            return max(0.0, _interp(self.table_rho, self.table_v, r))
        raise PedestrianFlowError("unknown fundamental-diagram kind %r." % kind)

    def flow(self, rho: float) -> float:
        """Specific flow q = rho * v(rho) in P/m/s."""
        r = _as_float(rho, "rho")
        if r <= 0.0:
            return 0.0
        return r * self.speed(r)

    def dflow(self, rho: float) -> float:
        """Analytic dq/drho -- the kinematic wave speed (m/s)."""
        r = _as_float(rho, "rho")
        if r <= 0.0:
            return self.v_free
        if r >= self.rho_jam:
            return self._jam_wave_speed()
        kind = self.kind
        if kind == "linear":
            return self.v_free * (1.0 - 2.0 * r / self.rho_jam)
        if kind == "power":
            n = self.exponent
            return self.v_free * (1.0 - (n + 1.0) * (r / self.rho_jam) ** n)
        if kind == "kladek":
            e = math.exp(-self.gamma * (1.0 / r - 1.0 / self.rho_jam))
            return self.v_free * ((1.0 - e) - e * self.gamma / r)
        if kind == "sfpe":
            if r <= self.rho_plateau:
                return self.v_free
            return self.k_ref * (1.0 - 2.0 * r / self.rho_jam)
        if kind == "table":
            if r <= self.rho_plateau:
                return self.v_free
            k = bisect.bisect_right(list(self.table_rho), r) - 1
            k = max(0, min(k, len(self.table_rho) - 2))
            r0, r1 = self.table_rho[k], self.table_rho[k + 1]
            v0, v1 = self.table_v[k], self.table_v[k + 1]
            slope = (v1 - v0) / (r1 - r0)
            v_here = v0 + slope * (r - r0)
            return v_here + r * slope
        raise PedestrianFlowError("unknown fundamental-diagram kind %r." % self.kind)

    def _jam_wave_speed(self) -> float:
        """Kinematic wave speed at (or beyond) the jam density."""
        kind = self.kind
        if kind == "linear":
            return -self.v_free
        if kind == "power":
            return -self.v_free * self.exponent
        if kind == "kladek":
            return -self.v_free * self.gamma / self.rho_jam
        if kind == "sfpe":
            return -self.k_ref
        if kind == "table":
            r0, r1 = self.table_rho[-2], self.table_rho[-1]
            v0, v1 = self.table_v[-2], self.table_v[-1]
            slope = (v1 - v0) / (r1 - r0)
            return v1 + r1 * slope
        return -self.v_free

    # -- capacity ----------------------------------------------------------
    def _capacity_pair(self) -> Tuple[float, float]:
        """(critical density, capacity) computed once and cached."""
        if self._cap is None:
            self._cap = self._solve_capacity()
        return self._cap

    def _solve_capacity(self) -> Tuple[float, float]:
        kind = self.kind
        if kind == "linear":
            rc = 0.5 * self.rho_jam
            return rc, self.v_free * self.rho_jam * 0.25
        if kind == "power":
            n = self.exponent
            rc = self.rho_jam * (1.0 / (n + 1.0)) ** (1.0 / n)
            return rc, self.v_free * rc * n / (n + 1.0)
        if kind == "kladek":
            lo = 1e-9 * self.rho_jam
            hi = self.rho_jam * (1.0 - 1e-12)
            rc = _bisect_root(self.dflow, lo, hi)
            return rc, self.flow(rc)
        if kind == "sfpe":
            rc = 0.5 * self.rho_jam
            if rc <= self.rho_plateau:
                rc = self.rho_plateau
                return rc, self.flow(rc)
            return rc, self.k_ref * self.rho_jam * 0.25
        if kind == "table":
            best_rho = self.rho_plateau
            best_q = self.flow(self.rho_plateau)
            for k in range(len(self.table_rho) - 1):
                r0, r1 = self.table_rho[k], self.table_rho[k + 1]
                v0, v1 = self.table_v[k], self.table_v[k + 1]
                slope = (v1 - v0) / (r1 - r0)
                cands = [r0, r1]
                if abs(slope) > _EPS:
                    # q(r) = r * (v0 + slope*(r - r0)); dq/dr = 0 at the vertex
                    vertex = (slope * r0 - v0) / (2.0 * slope)
                    if r0 < vertex < r1:
                        cands.append(vertex)
                for c in cands:
                    q = self.flow(c)
                    if q > best_q:
                        best_q, best_rho = q, c
            return best_rho, best_q
        raise PedestrianFlowError("unknown fundamental-diagram kind %r." % kind)

    def capacity(self) -> float:
        """Maximum specific flow q_max (P/m/s)."""
        return self._capacity_pair()[1]

    def critical_density(self) -> float:
        """Density at which q(rho) is maximal (P/m^2)."""
        return self._capacity_pair()[0]

    def capacity_speed(self) -> float:
        """Walking speed at capacity (m/s)."""
        return self.speed(self.critical_density())

    @property
    def rho_decline_start(self) -> float:
        """Density above which the speed law is strictly (and numerically) decreasing."""
        if self.kind in ("sfpe", "table"):
            return self.rho_plateau
        if self.kind == "kladek":
            # v differs from v_free by less than ~1e-9 below this density, so a
            # finite difference there would read as exactly zero in binary64.
            lim = math.log(1.0 / 1e-9)
            return self.gamma / (lim + self.gamma / self.rho_jam)
        return 0.0

    def density_from_flow(self, q: float, branch: str = "free") -> float:
        """Invert q(rho) on the free-flow ('free') or congested ('congested') branch."""
        qq = _non_negative(q, "q")
        rc, qmax = self._capacity_pair()
        if qq > qmax + 1e-9:
            raise PedestrianFlowError(
                "flow %.6g P/m/s exceeds the capacity %.6g P/m/s." % (qq, qmax))
        qq = min(qq, qmax)
        if branch == "free":
            lo, hi = 0.0, rc
        elif branch == "congested":
            lo, hi = rc, self.rho_jam
        else:
            raise PedestrianFlowError("branch must be 'free' or 'congested'.")
        if abs(self.flow(lo) - qq) < 1e-15:
            return lo
        if abs(self.flow(hi) - qq) < 1e-15:
            return hi
        return _bisect_root(lambda r: self.flow(r) - qq, lo, hi)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able description (no callables)."""
        rc, qmax = self._capacity_pair()
        return {
            "name": self.name,
            "kind": self.kind,
            "freeSpeed": round(self.v_free, 6),
            "jamDensity": round(self.rho_jam, 6),
            "criticalDensity": round(rc, 6),
            "capacity": round(qmax, 6),
            "capacitySpeed": round(self.speed(rc), 6),
            "source": self.source,
        }


def fd_weidmann(v_free: float = 1.34, rho_jam: float = 5.4,
                gamma: float = 1.913) -> PedestrianFD:
    """Weidmann (1993) Kladek law: v = v0 (1 - exp(-gamma (1/rho - 1/rho_jam)))."""
    return PedestrianFD(
        name="weidmann", kind="kladek",
        v_free=_positive(v_free, "v_free"), rho_jam=_positive(rho_jam, "rho_jam"),
        gamma=_positive(gamma, "gamma"),
        source="Weidmann 1993 (Kladek 1966 form), v0=1.34 m/s, rho_jam=5.4 P/m^2")


def fd_older(v_free: float = 1.31, slope: float = 0.34) -> PedestrianFD:
    """Older (1968) linear law for shopping streets: v = 1.31 - 0.34 rho."""
    v0 = _positive(v_free, "v_free")
    s = _positive(slope, "slope")
    return PedestrianFD(
        name="older", kind="linear", v_free=v0, rho_jam=v0 / s,
        source="Older 1968, v = 1.31 - 0.34 rho (m/s, P/m^2)")


def fd_fruin(v_free: float = 1.40, rho_jam: float = 3.9) -> PedestrianFD:
    """Fruin (1971) walkway law -- Greenshields form, capacity ~1.37 P/m/s."""
    return PedestrianFD(
        name="fruin", kind="linear",
        v_free=_positive(v_free, "v_free"), rho_jam=_positive(rho_jam, "rho_jam"),
        source="Fruin 1971 walkway design data (~25 P/min/ft at ~5.5 ft^2/P)")


def fd_sfpe(k: float = 1.40, a: float = 0.266,
            rho_plateau: float = 0.54) -> PedestrianFD:
    """SFPE Handbook law: v = k (1 - a rho), held flat below rho_plateau."""
    kk = _positive(k, "k")
    aa = _positive(a, "a")
    rp = _non_negative(rho_plateau, "rho_plateau")
    rho_jam = 1.0 / aa
    if rp >= rho_jam:
        raise PedestrianFlowError("rho_plateau must be below the jam density.")
    return PedestrianFD(
        name="sfpe", kind="sfpe", v_free=kk * (1.0 - aa * rp), rho_jam=rho_jam,
        rho_plateau=rp, k_ref=kk,
        source="SFPE Handbook (Nelson & Mowrer), k=1.40 m/s, a=0.266 m^2/P")


def fd_predtechenskii_milinskii(area_per_person: float = 0.1) -> PedestrianFD:
    """Predtechenskii-Milinskii tabulated horizontal-path law (D = rho * f)."""
    f = _positive(area_per_person, "area_per_person")
    rho = tuple(d / f for d in _PM_D)
    vel = tuple(v / 60.0 for v in _PM_V_M_PER_MIN)
    return PedestrianFD(
        name="predtechenskii_milinskii", kind="table",
        v_free=vel[0], rho_jam=rho[-1], rho_plateau=rho[0],
        area_per_person=f, table_rho=rho, table_v=vel,
        source="Predtechenskii & Milinskii 1978, horizontal path, comfortable")


def fd_virkler(v_free: float = 1.35, rho_jam: float = 4.0,
               exponent: float = 0.8) -> PedestrianFD:
    """Virkler & Elayadath (1994) power (Pipes-Munjal) law: v = v0 (1 - (rho/rho_j)^n)."""
    return PedestrianFD(
        name="virkler", kind="power",
        v_free=_positive(v_free, "v_free"), rho_jam=_positive(rho_jam, "rho_jam"),
        exponent=_positive(exponent, "exponent"),
        source="Virkler & Elayadath 1994, Pipes-Munjal form, n = 0.8")


FD_NAMES: Tuple[str, ...] = ("weidmann", "older", "fruin", "sfpe",
                             "predtechenskii_milinskii", "virkler")

_FD_FACTORIES: Dict[str, Callable[..., PedestrianFD]] = {
    "weidmann": fd_weidmann,
    "older": fd_older,
    "fruin": fd_fruin,
    "sfpe": fd_sfpe,
    "predtechenskii_milinskii": fd_predtechenskii_milinskii,
    "pm": fd_predtechenskii_milinskii,
    "virkler": fd_virkler,
}


def get_fd(name: str, **kwargs: Any) -> PedestrianFD:
    """Build a named fundamental diagram ('weidmann', 'older', 'fruin', ...)."""
    key = str(name).strip().lower()
    if key not in _FD_FACTORIES:
        raise PedestrianFlowError(
            "unknown fundamental diagram %r (known: %s)."
            % (name, ", ".join(sorted(_FD_FACTORIES))))
    return _FD_FACTORIES[key](**kwargs)


def fd_speed_curve(fd: PedestrianFD, n: int = 200,
                   rho_max: Optional[float] = None) -> List[Tuple[float, float]]:
    """Sample (rho, v) over [0, rho_max] (default the jam density)."""
    m = _positive_int(n, "n")
    top = fd.rho_jam if rho_max is None else _positive(rho_max, "rho_max")
    return [(top * i / m, fd.speed(top * i / m)) for i in range(m + 1)]


def fd_flow_curve(fd: PedestrianFD, n: int = 200,
                  rho_max: Optional[float] = None) -> List[Tuple[float, float]]:
    """Sample (rho, q) over [0, rho_max] (default the jam density)."""
    m = _positive_int(n, "n")
    top = fd.rho_jam if rho_max is None else _positive(rho_max, "rho_max")
    return [(top * i / m, fd.flow(top * i / m)) for i in range(m + 1)]


def max_speed_slope(fd: PedestrianFD, n: int = 2001) -> float:
    """Largest finite-difference dv/drho over the strictly declining branch (must be < 0)."""
    m = _positive_int(n, "n")
    lo = fd.rho_decline_start
    hi = fd.rho_jam
    span = hi - lo
    if span <= 0.0:
        raise PedestrianFlowError("the declining branch is empty for %s." % fd.name)
    worst = -_INF
    prev_r = lo
    prev_v = fd.speed(lo)
    for i in range(1, m + 1):
        r = lo + span * i / m
        v = fd.speed(r)
        worst = max(worst, (v - prev_v) / (r - prev_r))
        prev_r, prev_v = r, v
    return worst


# ===========================================================================
# 2. SPEED-DENSITY MODULATION  v = v0 * phi(rho)
# ===========================================================================
def phi_linear(rho: float, rho_jam: float) -> float:
    """Greenshields modulation phi = 1 - rho/rho_jam, clipped to [0, 1]."""
    r = _non_negative(rho, "rho")
    rj = _positive(rho_jam, "rho_jam")
    return max(0.0, min(1.0, 1.0 - r / rj))


def phi_exponential(rho: float, rho_jam: float, gamma: float = 1.913) -> float:
    """Kladek/Weidmann modulation phi = 1 - exp(-gamma (1/rho - 1/rho_jam))."""
    r = _non_negative(rho, "rho")
    rj = _positive(rho_jam, "rho_jam")
    g = _positive(gamma, "gamma")
    if r <= 0.0:
        return 1.0
    if r >= rj:
        return 0.0
    return max(0.0, min(1.0, 1.0 - math.exp(-g * (1.0 / r - 1.0 / rj))))


def phi_power(rho: float, rho_jam: float, exponent: float = 0.8) -> float:
    """Pipes-Munjal modulation phi = 1 - (rho/rho_jam)^n, clipped to [0, 1]."""
    r = _non_negative(rho, "rho")
    rj = _positive(rho_jam, "rho_jam")
    n = _positive(exponent, "exponent")
    if r >= rj:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (r / rj) ** n))


def speed_from_phi(v_free: float, phi: float) -> float:
    """Combine a free speed with a modulation factor: v = v0 * phi."""
    return _positive(v_free, "v_free") * _in_unit(phi, "phi")


def tobler_walking_speed(slope: float, v_flat: float = 1.3888888888888888) -> float:
    """Tobler's hiking function v = v_flat * exp(-3.5 |slope + 0.05|) / exp(-0.175)."""
    s = _as_float(slope, "slope")
    vf = _positive(v_flat, "v_flat")
    # Tobler: 6 km/h * exp(-3.5|S+0.05|); normalised so slope 0 gives v_flat.
    raw = math.exp(-3.5 * abs(s + 0.05))
    flat = math.exp(-3.5 * 0.05)
    return vf * raw / flat


# ===========================================================================
# 3. CONSERVATION PDE -- GODUNOV WITH SENDING / RECEIVING FLOWS
# ===========================================================================
# The kinematic-wave (LWR) model for pedestrians on a pavement of width W,
#
#       d(rho)/dt + d(rho v(rho))/dx = s(x, t),
#
# is solved cell by cell in flux form.  The Godunov flux across the face
# between an upstream cell U and a downstream cell D is
#
#       F = min( demand(rho_U), supply(rho_D) )
#
# where the demand (sending) flow saturates at capacity above rho_c and the
# supply (receiving) flow saturates at capacity below rho_c.  Because the
# update is a telescoping difference of face fluxes, the number of people on
# the link changes by exactly (inflow - outflow + source) * dt -- no clamping,
# no clipping, no leakage.


def demand_flow(fd: PedestrianFD, rho: float) -> float:
    """Sending flow: q(rho) below the critical density, capacity above it."""
    rc, qmax = fd._capacity_pair()
    return fd.flow(rho) if rho <= rc else qmax


def supply_flow(fd: PedestrianFD, rho: float) -> float:
    """Receiving flow: capacity below the critical density, q(rho) above it."""
    rc, qmax = fd._capacity_pair()
    return qmax if rho <= rc else fd.flow(rho)


def godunov_flux(fd: PedestrianFD, rho_up: float, rho_down: float) -> float:
    """Godunov face flux min(demand upstream, supply downstream) in P/m/s."""
    return min(demand_flow(fd, rho_up), supply_flow(fd, rho_down))


def max_wave_speed(fd: PedestrianFD, n: int = 1024) -> float:
    """Largest |dq/drho| over [0, rho_jam] -- the CFL wave speed (m/s)."""
    m = _positive_int(n, "n")
    if fd.kind == "table":
        # dq/drho is linear inside every segment, so the extremes sit at nodes.
        cands = [abs(fd.v_free)]
        for k in range(len(fd.table_rho) - 1):
            r0, r1 = fd.table_rho[k], fd.table_rho[k + 1]
            eps = 1e-9 * max(1.0, r1 - r0)
            cands.append(abs(fd.dflow(r0 + eps)))
            cands.append(abs(fd.dflow(r1 - eps)))
        return max(cands)
    worst = abs(fd.v_free)
    top = fd.rho_jam * (1.0 - 1e-12)
    for i in range(m + 1):
        worst = max(worst, abs(fd.dflow(min(fd.rho_jam * i / m, top))))
    return max(worst, abs(fd._jam_wave_speed()))


def cfl_timestep(fd: PedestrianFD, dx: float, cfl: float = 0.9) -> float:
    """Stable explicit time step dt = cfl * dx / max|dq/drho|."""
    d = _positive(dx, "dx")
    c = _positive(cfl, "cfl")
    if c > 1.0:
        raise PedestrianFlowError("cfl must be <= 1 (got %.6g)." % c)
    # SCALE-RELATIVE: ``max(w, _EPS)`` is an absolute floor on a DIMENSIONED
    # wave speed -- it silently clamped the step for any diagram slower than
    # 1e-12 m/s instead of scaling with it.  A diagram with no positive wave
    # speed has no stable step at all, and says so.
    w = max_wave_speed(fd)
    if w <= 0.0:
        raise PedestrianFlowError(
            "the %s diagram has no positive wave speed, so no explicit time "
            "step is stable." % fd.name)
    return c * d / w


@dataclass
class WalkwayLink:
    """A one-dimensional pavement / crossing segment discretised into cells."""

    length: float
    width: float
    fd: PedestrianFD
    n_cells: int = 50
    name: str = "walkway"
    density: Density = field(default_factory=list)

    def __post_init__(self) -> None:
        self.length = _positive(self.length, "length")
        self.width = _positive(self.width, "width")
        self.n_cells = _positive_int(self.n_cells, "n_cells")
        if not self.density:
            self.density = [0.0] * self.n_cells
        else:
            self.density = _as_floats(self.density, "density")
            if len(self.density) != self.n_cells:
                raise PedestrianFlowError(
                    "density has %d cells but n_cells is %d."
                    % (len(self.density), self.n_cells))

    @property
    def dx(self) -> float:
        """Cell length (m)."""
        return self.length / self.n_cells

    @property
    def area(self) -> float:
        """Plan area of the link (m^2)."""
        return self.length * self.width

    @property
    def capacity_pps(self) -> float:
        """Total capacity of the link in persons per second."""
        return self.width * self.fd.capacity()

    def people(self) -> float:
        """Number of people currently stored on the link."""
        return self.width * self.dx * math.fsum(self.density)

    def set_uniform(self, rho: float) -> "WalkwayLink":
        """Set every cell to the same density (returns self)."""
        r = _non_negative(rho, "rho")
        self.density = [r] * self.n_cells
        return self

    def set_profile(self, fn: Callable[[float], float]) -> "WalkwayLink":
        """Set the density from a function of the cell-centre position (returns self)."""
        dx = self.dx
        self.density = [_non_negative(fn(dx * (i + 0.5)), "rho")
                        for i in range(self.n_cells)]
        return self

    def copy(self) -> "WalkwayLink":
        """Copy the link with an independent density state."""
        return WalkwayLink(self.length, self.width, self.fd, self.n_cells,
                           self.name, list(self.density))

    def cell_centres(self) -> List[float]:
        """Cell-centre positions along the link (m)."""
        dx = self.dx
        return [dx * (i + 0.5) for i in range(self.n_cells)]


def godunov_step(rho: Sequence[float], fd: PedestrianFD, dx: float, dt: float,
                 flux_in: float = 0.0, flux_out: Optional[float] = None,
                 source: Optional[Sequence[float]] = None,
                 width: float = 1.0) -> Tuple[Density, Flow]:
    """One explicit Godunov update; returns the new densities and the n+1 face fluxes.

    With the default ``width`` of 1.0 every flux is a specific flow in P/m/s.
    Passing the real width makes the fluxes TOTAL person flows in P/s, which is
    what a boundary condition or a junction actually supplies -- and, because no
    division by the width is then needed anywhere, a uniform state is held
    bit-exactly rather than to one ulp per step.
    """
    r = list(rho)
    n = len(r)
    if n == 0:
        raise PedestrianFlowError("godunov_step needs at least one cell.")
    d = _positive(dx, "dx")
    t = _positive(dt, "dt")
    w = _positive(width, "width")
    f: Flow = [0.0] * (n + 1)
    f[0] = _as_float(flux_in, "flux_in")
    for i in range(n - 1):
        f[i + 1] = godunov_flux(fd, r[i], r[i + 1]) * w
    f[n] = demand_flow(fd, r[n - 1]) * w if flux_out is None else _as_float(
        flux_out, "flux_out")
    lam = t / (d * w)
    out = [r[i] + lam * (f[i] - f[i + 1]) for i in range(n)]
    if source is not None:
        src = _as_floats(source, "source")
        if len(src) != n:
            raise PedestrianFlowError("source must have one entry per cell.")
        out = [out[i] + t * src[i] for i in range(n)]
    return out, f


@dataclass
class WalkwaySimulation:
    """Result of a Godunov run on a single link, with an exact people ledger."""

    name: str
    dt: float
    dx: float
    steps: int
    time: float
    density: Density
    history: List[Tuple[float, Density]]
    outflow_series: List[Tuple[float, float]]
    inflow_total: float
    outflow_total: float
    source_total: float
    stored_initial: float
    stored_final: float
    conservation_error: float
    max_density: float
    mean_density: float
    tracer_time: Optional[float] = None
    tracer_position: Optional[float] = None

    @property
    def balance(self) -> float:
        """stored_initial + inflow + source - outflow (must equal stored_final)."""
        return math.fsum([self.stored_initial, self.inflow_total,
                          self.source_total, -self.outflow_total])


def simulate_walkway(link: WalkwayLink, duration: float, *,
                     inflow: Any = 0.0, outflow_capacity: Any = None,
                     source: Any = None, cfl: float = 0.9,
                     dt: Optional[float] = None, boundary: str = "open",
                     tracer_start: Optional[float] = None,
                     record_every: int = 0) -> WalkwaySimulation:
    """Run the Godunov solver on one link; ``inflow`` and ``outflow_capacity`` are P/s."""
    if boundary not in ("open", "periodic", "wall"):
        raise PedestrianFlowError("boundary must be 'open', 'periodic' or 'wall'.")
    work = link.copy()
    fd = work.fd
    dxc = work.dx
    if dt is None:
        step_dt = cfl_timestep(fd, dxc, cfl)
    else:
        # ``cfl_timestep`` refuses cfl > 1, so the explicit-``dt`` door must
        # refuse an over-CFL step too: above it the demand/supply update no
        # longer keeps the density inside [0, rho_jam] and the ledger closes
        # around densities that are not physical.
        step_dt = _positive(dt, "dt")
        dt_max = cfl_timestep(fd, dxc, 1.0)
        if step_dt > dt_max * (1.0 + 1e-12):
            raise PedestrianFlowError(
                "dt = %.6g s violates the CFL condition: on a %.6g m cell the "
                "largest stable step is %.6g s." % (step_dt, dxc, dt_max))
    total = _positive(duration, "duration")
    n_steps = max(1, int(math.ceil(total / step_dt)))
    width = work.width
    cap_default = width * fd.capacity()

    def _rate(spec: Any, t_now: float, default: float) -> float:
        if spec is None:
            return default
        if callable(spec):
            return _non_negative(spec(t_now), "rate")
        return _non_negative(spec, "rate")

    def _source_vec(t_now: float) -> Optional[List[float]]:
        if source is None:
            return None
        if callable(source):
            return [_as_float(source(t_now, dxc * (i + 0.5)), "source")
                    for i in range(work.n_cells)]
        vec = _as_floats(source, "source")
        if len(vec) != work.n_cells:
            raise PedestrianFlowError("source must have one entry per cell.")
        return vec

    stored_initial = work.people()
    in_total = 0.0
    out_total = 0.0
    src_total = 0.0
    history: List[Tuple[float, Density]] = []
    outflow_series: List[Tuple[float, float]] = []
    if record_every:
        history.append((0.0, list(work.density)))
    tracer_x = None if tracer_start is None else _non_negative(
        tracer_start, "tracer_start")
    tracer_t: Optional[float] = None
    t = 0.0

    for step in range(n_steps):
        rho = work.density
        if boundary == "periodic":
            fin = godunov_flux(fd, rho[-1], rho[0]) * width
            fout = fin
        elif boundary == "wall":
            fin = 0.0
            fout = 0.0
        else:
            demand_in = _rate(inflow, t, 0.0)
            fin = min(demand_in, supply_flow(fd, rho[0]) * width)
            cap_out = _rate(outflow_capacity, t, cap_default)
            fout = min(demand_flow(fd, rho[-1]) * width, cap_out)
        src = _source_vec(t)
        # advance the tracer on the pre-update field (people move at v(rho))
        if tracer_x is not None and tracer_t is None:
            idx = min(work.n_cells - 1, max(0, int(tracer_x / dxc)))
            v_here = fd.speed(rho[idx])
            if v_here > _EPS:
                nxt = tracer_x + v_here * step_dt
                if nxt >= work.length:
                    tracer_t = t + (work.length - tracer_x) / v_here
                    tracer_x = work.length
                else:
                    tracer_x = nxt
        new, _faces = godunov_step(rho, fd, dxc, step_dt, fin, fout, src, width)
        work.density = new
        in_total = math.fsum([in_total, fin * step_dt])
        out_total = math.fsum([out_total, fout * step_dt])
        if src is not None:
            src_total = math.fsum([src_total,
                                   math.fsum(src) * width * dxc * step_dt])
        t += step_dt
        outflow_series.append((t, fout))
        if record_every and (step + 1) % record_every == 0:
            history.append((t, list(work.density)))

    stored_final = work.people()
    balance = math.fsum([stored_initial, in_total, src_total, -out_total])
    return WalkwaySimulation(
        name=work.name, dt=step_dt, dx=dxc, steps=n_steps, time=t,
        density=list(work.density), history=history,
        outflow_series=outflow_series, inflow_total=in_total,
        outflow_total=out_total, source_total=src_total,
        stored_initial=stored_initial, stored_final=stored_final,
        conservation_error=abs(stored_final - balance),
        max_density=max(work.density), mean_density=_mean(work.density),
        tracer_time=tracer_t, tracer_position=tracer_x)


@dataclass
class RiemannSolution:
    """Analytic solution of the Riemann problem for a concave pedestrian flux."""

    kind: str                 # "shock" | "rarefaction" | "constant"
    rho_left: float
    rho_right: float
    flow_left: float
    flow_right: float
    speed: float              # Rankine-Hugoniot shock speed (0 if constant)
    head_speed: float         # fastest characteristic of a rarefaction fan
    tail_speed: float         # slowest characteristic of a rarefaction fan
    interface_flow: float     # Godunov flux at x/t = 0


def _riemann_tol(fd: PedestrianFD, rl: float, rr: float) -> float:
    """Scale-relative 'these two densities are the same state' tolerance (P/m^2)."""
    # Densities are dimensioned, so the "constant state" test must be measured
    # against the diagram's own jam density, never against an absolute 1e-12.
    return _EPS * max(1.0e-300, abs(rl), abs(rr), abs(fd.rho_jam))


def riemann_shock_speed(fd: PedestrianFD, rho_left: float,
                        rho_right: float) -> float:
    """Rankine-Hugoniot shock speed s = (q_R - q_L) / (rho_R - rho_L) in m/s."""
    rl = _non_negative(rho_left, "rho_left")
    rr = _non_negative(rho_right, "rho_right")
    if abs(rr - rl) <= _riemann_tol(fd, rl, rr):
        raise PedestrianFlowError("Rankine-Hugoniot needs rho_left != rho_right.")
    return (fd.flow(rr) - fd.flow(rl)) / (rr - rl)


def riemann_solution(fd: PedestrianFD, rho_left: float,
                     rho_right: float) -> RiemannSolution:
    """Classify the Riemann problem and give its wave speeds and interface flux."""
    rl = _non_negative(rho_left, "rho_left")
    rr = _non_negative(rho_right, "rho_right")
    ql, qr = fd.flow(rl), fd.flow(rr)
    face = godunov_flux(fd, rl, rr)
    if abs(rr - rl) <= _riemann_tol(fd, rl, rr):
        return RiemannSolution("constant", rl, rr, ql, qr, 0.0,
                               fd.dflow(rl), fd.dflow(rr), face)
    s = (qr - ql) / (rr - rl)
    if rl < rr:
        return RiemannSolution("shock", rl, rr, ql, qr, s,
                               fd.dflow(rl), fd.dflow(rr), face)
    return RiemannSolution("rarefaction", rl, rr, ql, qr, s,
                           fd.dflow(rr), fd.dflow(rl), face)


def front_position(rho: Sequence[float], dx: float,
                   level: float) -> Optional[float]:
    """Interpolated position (m) where the density profile first crosses ``level``."""
    r = _as_floats(rho, "rho")
    d = _positive(dx, "dx")
    lv = _as_float(level, "level")
    # SCALE-RELATIVE: densities are dimensioned, so "is this pair a genuine
    # crossing or a flat segment?" must be asked against the profile's own
    # magnitude, not against an absolute 1e-12 P/m^2.
    tol = _EPS * max([1.0e-300] + [abs(v) for v in r] + [abs(lv)])
    for i in range(len(r) - 1):
        a, b = r[i], r[i + 1]
        if (a - lv) * (b - lv) <= 0.0 and abs(b - a) > tol:
            frac = (lv - a) / (b - a)
            return d * (i + 0.5) + frac * d
    return None


def walkway_travel_time(length: float, fd: PedestrianFD,
                        density: float = 0.0) -> float:
    """Traversal time L / v(rho); at rho = 0 this is exactly L / v_free."""
    L = _positive(length, "length")
    v = fd.speed(_non_negative(density, "density"))
    if v <= _EPS:
        return _INF
    return L / v


# ===========================================================================
# 4. NODE COUPLING -- MERGES, DIVERGES AND JUNCTION TRANSFER
# ===========================================================================
@dataclass
class NodeFlowResult:
    """Flows accepted at a junction: per incoming link, per outgoing link and paired."""

    inflows: List[float]                 # accepted from each incoming link (P/s)
    outflows: List[float]                # delivered to each outgoing link (P/s)
    matrix: List[List[float]]            # matrix[i][j] = flow i -> j (P/s)
    reduction: List[float]               # alpha_i in [0, 1] applied to each sending flow
    conservation_error: float
    iterations: int
    converged: bool = True               # False if max_iter ran out mid-rationing


def merge_flows(sending: Sequence[float], capacity: float,
                priorities: Optional[Sequence[float]] = None) -> List[float]:
    """Share a merge capacity between competing streams by priority water-filling."""
    s = _as_floats(sending, "sending")
    for v in s:
        if v < 0.0:
            raise PedestrianFlowError("sending flows must be >= 0.")
    cap = _non_negative(capacity, "capacity")
    n = len(s)
    if n == 0:
        return []
    if priorities is None:
        p = [1.0] * n
    else:
        p = _as_floats(priorities, "priorities")
        if len(p) != n:
            raise PedestrianFlowError("priorities must match sending.")
        for v in p:
            if v < 0.0:
                raise PedestrianFlowError("priorities must be >= 0.")
    out = [0.0] * n
    remaining = cap
    # SCALE-RELATIVE thresholds: sending flows and capacities are DIMENSIONED
    # (P/s), so an absolute 1e-12 made a merge of two 1e-12 P/s streams return
    # all zeros while the same merge scaled up by 1e3 was exact.
    s_tol = _EPS * max([1.0e-300] + [abs(v) for v in s] + [abs(cap)])
    p_tol = _EPS * max([1.0e-300] + [abs(v) for v in p])
    active = [i for i in range(n) if s[i] > s_tol and p[i] > p_tol]
    while active and remaining > s_tol:
        ptot = math.fsum(p[i] for i in active)
        if ptot <= p_tol:
            break
        lam = min((s[i] - out[i]) / p[i] for i in active)
        if lam * ptot >= remaining:
            lam = remaining / ptot
            for i in active:
                out[i] = min(s[i], out[i] + lam * p[i])
            remaining = 0.0
            break
        done = set()
        for i in active:
            out[i] += lam * p[i]
            if s[i] - out[i] <= 1e-12 * max(1.0, s[i]):
                out[i] = s[i]
                done.add(i)
        remaining -= lam * ptot
        active = [i for i in active if i not in done]
    return [min(out[i], s[i]) for i in range(n)]


def diverge_flows(sending: float, splits: Sequence[float],
                  receiving: Sequence[float]) -> Tuple[float, List[float]]:
    """FIFO diverge: one stream, fixed splits, blocked by the tightest downstream supply."""
    s = _non_negative(sending, "sending")
    b = _as_floats(splits, "splits")
    r = _as_floats(receiving, "receiving")
    if len(b) != len(r):
        raise PedestrianFlowError("splits and receiving must be the same length.")
    if any(v < 0.0 for v in b):
        raise PedestrianFlowError("splits must be >= 0.")
    tot = math.fsum(b)
    # The splits are normalised on the very next line, so they are a pure
    # DIMENSIONLESS ratio: [1e-13, 1e-13] and [0.5, 0.5] are the same input
    # and must give the same answer.  Test the sum against the largest weight,
    # never against an absolute 1e-12.
    if tot <= _EPS * max([1.0e-300] + [abs(v) for v in b]):
        return 0.0, [0.0] * len(b)
    frac = [v / tot for v in b]
    allowed = s
    for j, fj in enumerate(frac):
        if fj > _EPS:
            allowed = min(allowed, r[j] / fj)
    allowed = max(0.0, allowed)
    return allowed, [allowed * fj for fj in frac]


def node_transfer(sending: Sequence[float], receiving: Sequence[float],
                  splits: Sequence[Sequence[float]],
                  priorities: Optional[Sequence[float]] = None,
                  max_iter: int = 50) -> NodeFlowResult:
    """General junction: FIFO-consistent supply rationing with priority water-filling."""
    s = _as_floats(sending, "sending")
    r = _as_floats(receiving, "receiving")
    n_in, n_out = len(s), len(r)
    if n_in == 0 or n_out == 0:
        return NodeFlowResult([0.0] * n_in, [0.0] * n_out,
                              [[0.0] * n_out for _ in range(n_in)],
                              [0.0] * n_in, 0.0, 0)
    beta: List[List[float]] = []
    for i in range(n_in):
        row = _as_floats(splits[i], "splits[%d]" % i)
        if len(row) != n_out:
            raise PedestrianFlowError(
                "splits rows must have one entry per outgoing link.")
        if any(v < 0.0 for v in row):
            raise PedestrianFlowError("splits must be >= 0.")
        tot = math.fsum(row)
        if tot <= _EPS:
            raise PedestrianFlowError("splits row %d sums to zero." % i)
        beta.append([v / tot for v in row])
    prio = [1.0] * n_in if priorities is None else _as_floats(priorities,
                                                              "priorities")
    if len(prio) != n_in:
        raise PedestrianFlowError("priorities must match the incoming links.")

    alpha = [1.0] * n_in
    iters = 0
    changed = False
    # SCALE-RELATIVE tolerance: sending and receiving flows are P/s, so an
    # absolute 1e-12 let a junction whose whole traffic was ~1e-12 P/s deliver
    # TWICE its downstream supply -- breaking the module's stated invariant
    # that a junction transfer never violates a downstream supply.
    scale = max([1.0e-300] + [abs(v) for v in s] + [abs(v) for v in r])
    tol = _EPS * scale
    for iters in range(1, max_iter + 1):
        changed = False
        for j in range(n_out):
            contrib = [alpha[i] * s[i] * beta[i][j] for i in range(n_in)]
            if math.fsum(contrib) > r[j] + tol:
                share = merge_flows(contrib, r[j], prio)
                for i in range(n_in):
                    if contrib[i] > tol:
                        ratio = share[i] / contrib[i]
                        if ratio < 1.0 - 1e-15:
                            new_alpha = alpha[i] * ratio
                            if new_alpha < alpha[i] - 1e-15:
                                alpha[i] = new_alpha
                                changed = True
        if not changed:
            break
    inflows = [alpha[i] * s[i] for i in range(n_in)]
    matrix = [[inflows[i] * beta[i][j] for j in range(n_out)]
              for i in range(n_in)]
    outflows = [math.fsum(matrix[i][j] for i in range(n_in))
                for j in range(n_out)]
    err = abs(math.fsum(inflows) - math.fsum(outflows))
    return NodeFlowResult(inflows, outflows, matrix, alpha, err, iters,
                          not changed)


@dataclass
class Junction:
    """A pedestrian junction: incoming link indices, outgoing link indices and splits."""

    name: str
    incoming: List[int]
    outgoing: List[int]
    splits: List[List[float]]
    priorities: Optional[List[float]] = None


@dataclass
class PedestrianNetwork:
    """Links plus junctions, with external entry demands and exit capacities (P/s)."""

    links: List[WalkwayLink]
    junctions: List[Junction] = field(default_factory=list)
    entries: Dict[int, float] = field(default_factory=dict)
    exits: Dict[int, float] = field(default_factory=dict)

    def people(self) -> float:
        """Total people stored on every link."""
        return math.fsum(lk.people() for lk in self.links)

    def copy(self) -> "PedestrianNetwork":
        """Copy with independent link states."""
        return PedestrianNetwork([lk.copy() for lk in self.links],
                                 list(self.junctions), dict(self.entries),
                                 dict(self.exits))


@dataclass
class NetworkSimulation:
    """Result of a network run, with a people ledger over the whole network."""

    dt: float
    steps: int
    time: float
    densities: Dict[str, Density]
    stored_initial: float
    stored_final: float
    entry_total: float
    exit_total: float
    conservation_error: float
    junction_flows: List[List[float]]
    exit_series: List[Tuple[float, float]]


def simulate_network(net: PedestrianNetwork, duration: float, *,
                     cfl: float = 0.9, dt: Optional[float] = None,
                     record_every: int = 0) -> NetworkSimulation:
    """Run coupled Godunov links with FIFO junction transfer; people are conserved exactly."""
    work = net.copy()
    if not work.links:
        raise PedestrianFlowError("the network has no links.")
    if dt is None:
        step_dt = min(cfl_timestep(lk.fd, lk.dx, cfl) for lk in work.links)
    else:
        step_dt = _positive(dt, "dt")
        dt_max = min(cfl_timestep(lk.fd, lk.dx, 1.0) for lk in work.links)
        if step_dt > dt_max * (1.0 + 1e-12):
            raise PedestrianFlowError(
                "dt = %.6g s violates the CFL condition: the tightest link "
                "allows %.6g s." % (step_dt, dt_max))
    total = _positive(duration, "duration")
    n_steps = max(1, int(math.ceil(total / step_dt)))

    stored_initial = work.people()
    entry_total = 0.0
    exit_total = 0.0
    junction_flows: List[List[float]] = []
    exit_series: List[Tuple[float, float]] = []
    t = 0.0

    for step in range(n_steps):
        n_links = len(work.links)
        in_flux = [0.0] * n_links     # P/s entering each link's upstream end
        out_flux = [0.0] * n_links    # P/s leaving each link's downstream end
        # A link END may carry more than one role -- a footway can feed a
        # junction AND discharge to the street, or feed two junctions.  Each
        # end therefore gets ONE budget: the downstream demand it can send and
        # the upstream supply it can receive.  Consumers draw from what is
        # left, so nobody is sent twice.  When every end carries exactly one
        # role (the ordinary case) the remaining budget is the full budget and
        # the update is bit-identical to a single-role network.
        demand_left = [demand_flow(lk.fd, lk.density[-1]) * lk.width
                       for lk in work.links]
        supply_left = [supply_flow(lk.fd, lk.density[0]) * lk.width
                       for lk in work.links]
        step_junction: List[float] = []
        for jn in work.junctions:
            sending = [demand_left[i] for i in jn.incoming]
            receiving = [supply_left[j] for j in jn.outgoing]
            res = node_transfer(sending, receiving, jn.splits, jn.priorities)
            for k, i in enumerate(jn.incoming):
                out_flux[i] += res.inflows[k]
                demand_left[i] = max(0.0, demand_left[i] - res.inflows[k])
            for k, j in enumerate(jn.outgoing):
                in_flux[j] += res.outflows[k]
                supply_left[j] = max(0.0, supply_left[j] - res.outflows[k])
            step_junction.append(math.fsum(res.inflows))
        step_entry = 0.0
        for idx in sorted(work.entries):
            f = min(_non_negative(work.entries[idx], "entry demand"),
                    supply_left[idx])
            in_flux[idx] += f
            supply_left[idx] = max(0.0, supply_left[idx] - f)
            step_entry += f
        step_exit = 0.0
        for idx in sorted(work.exits):
            f = min(demand_left[idx],
                    _non_negative(work.exits[idx], "exit capacity"))
            out_flux[idx] += f
            demand_left[idx] = max(0.0, demand_left[idx] - f)
            step_exit += f
        for idx, lk in enumerate(work.links):
            new, _faces = godunov_step(lk.density, lk.fd, lk.dx, step_dt,
                                       in_flux[idx], out_flux[idx], None,
                                       lk.width)
            lk.density = new
        entry_total = math.fsum([entry_total, step_entry * step_dt])
        exit_total = math.fsum([exit_total, step_exit * step_dt])
        t += step_dt
        exit_series.append((t, step_exit))
        if record_every and (step + 1) % record_every == 0:
            junction_flows.append(step_junction)

    stored_final = work.people()
    balance = math.fsum([stored_initial, entry_total, -exit_total])
    return NetworkSimulation(
        dt=step_dt, steps=n_steps, time=t,
        densities={lk.name: list(lk.density) for lk in work.links},
        stored_initial=stored_initial, stored_final=stored_final,
        entry_total=entry_total, exit_total=exit_total,
        conservation_error=abs(stored_final - balance),
        junction_flows=junction_flows, exit_series=exit_series)


# ===========================================================================
# 5. BIDIRECTIONAL FLOW, COUNTERFLOW FRICTION AND LANE FORMATION
# ===========================================================================
# Two opposing streams share a pavement.  The total density sets the base speed
# through the fundamental diagram; the opposing *fraction* then removes a
# further slice of speed through a friction coefficient.  Self-organised lane
# formation recovers that slice: at a lane order parameter of 1 the streams are
# fully segregated and the counterflow penalty vanishes.


def counterflow_penalty(opposing_fraction: float, friction: float = 0.25,
                        lane_index: float = 0.0) -> float:
    """Speed multiplier 1 - mu * f_opp * (1 - lane index), clipped to [0, 1]."""
    f = _in_unit(opposing_fraction, "opposing_fraction")
    mu = _non_negative(friction, "friction")
    li = _in_unit(lane_index, "lane_index")
    return max(0.0, min(1.0, 1.0 - mu * f * (1.0 - li)))


def bidirectional_speeds(rho_plus: float, rho_minus: float, fd: PedestrianFD,
                         friction: float = 0.25,
                         lane_index: float = 0.0) -> Tuple[float, float]:
    """Speeds of the two opposing streams sharing one pavement (m/s)."""
    rp = _non_negative(rho_plus, "rho_plus")
    rm = _non_negative(rho_minus, "rho_minus")
    total = rp + rm
    if total <= _EPS:
        return fd.v_free, fd.v_free
    base = fd.speed(total)
    v_plus = base * counterflow_penalty(rm / total, friction, lane_index)
    v_minus = base * counterflow_penalty(rp / total, friction, lane_index)
    return v_plus, v_minus


def bidirectional_capacity(fd: PedestrianFD, split: float = 0.5,
                           friction: float = 0.25, lane_index: float = 0.0,
                           n: int = 2000) -> Tuple[float, float]:
    """Maximum total specific flow of a two-way pavement: (rho_total, q_total)."""
    sp = _in_unit(split, "split")
    m = _positive_int(n, "n")
    best = (0.0, 0.0)
    for i in range(1, m + 1):
        total = fd.rho_jam * i / m
        rp = total * sp
        rm = total - rp
        vp, vm = bidirectional_speeds(rp, rm, fd, friction, lane_index)
        q = rp * vp + rm * vm
        if q > best[1]:
            best = (total, q)
    return best


def lane_formation_index(plus_counts: Sequence[float],
                         minus_counts: Sequence[float]) -> float:
    """Order parameter mean over strips of ((n+ - n-)/(n+ + n-))^2: 1 = fully segregated."""
    a = _as_floats(plus_counts, "plus_counts")
    b = _as_floats(minus_counts, "minus_counts")
    if len(a) != len(b):
        raise PedestrianFlowError("the two strip profiles must be the same length.")
    if not a:
        raise PedestrianFlowError("need at least one lateral strip.")
    vals = []
    for i in range(len(a)):
        tot = a[i] + b[i]
        if tot <= _EPS:
            continue
        vals.append(((a[i] - b[i]) / tot) ** 2)
    if not vals:
        return 0.0
    return math.fsum(vals) / len(vals)


@dataclass
class LaneFormationResult:
    """Trajectory of the lane order parameter during lateral self-organisation."""

    index_series: List[float]
    final_index: float
    initial_index: float
    plus_profile: List[float]
    minus_profile: List[float]
    steps: int


def simulate_lane_formation(n_strips: int = 8, n_plus: int = 40,
                            n_minus: int = 40, steps: int = 200,
                            beta: float = 2.0, seed: int = 0
                            ) -> LaneFormationResult:
    """Seeded lateral-sorting model: walkers drift towards strips of their own direction."""
    ns = _positive_int(n_strips, "n_strips")
    npl = _positive_int(n_plus, "n_plus")
    nmi = _positive_int(n_minus, "n_minus")
    nsteps = _positive_int(steps, "steps")
    b = _non_negative(beta, "beta")
    rng = random.Random(seed)
    strip_plus = [0] * ns
    strip_minus = [0] * ns
    agents: List[Tuple[int, int]] = []          # (direction, strip)
    for _ in range(npl):
        s = rng.randrange(ns)
        agents.append((1, s))
        strip_plus[s] += 1
    for _ in range(nmi):
        s = rng.randrange(ns)
        agents.append((-1, s))
        strip_minus[s] += 1
    initial = lane_formation_index([float(v) for v in strip_plus],
                                   [float(v) for v in strip_minus])
    series = [initial]
    for _ in range(nsteps):
        order = list(range(len(agents)))
        rng.shuffle(order)
        for k in order:
            direction, s = agents[k]
            same = strip_plus if direction > 0 else strip_minus
            other = strip_minus if direction > 0 else strip_plus
            cands = [s]
            if s > 0:
                cands.append(s - 1)
            if s < ns - 1:
                cands.append(s + 1)
            weights = []
            for c in cands:
                sm = same[c] - (1 if c == s else 0)
                ot = other[c]
                weights.append(math.exp(b * (sm - ot)
                                        / max(1.0, float(npl + nmi) / ns)))
            tot = math.fsum(weights)
            pick = rng.random() * tot
            acc = 0.0
            choice = cands[-1]
            for c, w in zip(cands, weights):
                acc += w
                if pick <= acc:
                    choice = c
                    break
            if choice != s:
                same[s] -= 1
                same[choice] += 1
                agents[k] = (direction, choice)
        series.append(lane_formation_index([float(v) for v in strip_plus],
                                           [float(v) for v in strip_minus]))
    return LaneFormationResult(series, series[-1], initial,
                               [float(v) for v in strip_plus],
                               [float(v) for v in strip_minus], nsteps)


# ===========================================================================
# 6. EIKONAL POTENTIAL AND HUGHES' CONTINUUM CROWD MODEL
# ===========================================================================
# Hughes (2002) closes the two-dimensional continuity equation with a walking
# potential phi that satisfies the eikonal equation
#
#       |grad phi| = 1 / (f(rho) g(rho)),        v = -f(rho) grad phi / |grad phi|
#
# so pedestrians walk down the *least remaining travel time* field rather than
# along a fixed link graph, and congestion ahead of them bends the field.  The
# potential is solved by the fast-marching method: a Dijkstra-like sweep in
# which every accepted cell solves the upwind quadratic
#
#       ((T - a)+)^2 + ((T - b)+)^2 = (dx / F)^2 .


def fast_marching(speed: Grid, sources: Sequence[Cell],
                  walkable: Optional[Mask] = None, dx: float = 1.0) -> Grid:
    """Fast-marching solution of |grad T| = 1/F from the given source cells."""
    ny = len(speed)
    if ny == 0 or len(speed[0]) == 0:
        raise PedestrianFlowError("the speed grid is empty.")
    nx = len(speed[0])
    d = _positive(dx, "dx")
    ok = walkable if walkable is not None else [[True] * nx for _ in range(ny)]
    if len(ok) != ny or any(len(row) != nx for row in ok):
        raise PedestrianFlowError("walkable must match the speed grid shape.")
    # SCALE-RELATIVE "this cell is impassable" threshold: the speed field is
    # dimensioned (m/s), so a grid whose fastest cell is 1e-12 m/s is a
    # perfectly ordinary problem in different units, not a wall everywhere.
    f_tol = _EPS * max([1.0e-300] + [abs(v) for row in speed for v in row])
    T: Grid = [[_INF] * nx for _ in range(ny)]
    frozen = [[False] * nx for _ in range(ny)]
    heap: List[Tuple[float, int, int]] = []
    for (i, j) in sources:
        if not (0 <= i < ny and 0 <= j < nx):
            raise PedestrianFlowError("source cell (%d, %d) is outside the grid." % (i, j))
        T[i][j] = 0.0
        heappush(heap, (0.0, i, j))

    def _solve(i: int, j: int) -> float:
        f = speed[i][j]
        if f <= f_tol:
            return _INF
        h = d / f
        a = _INF
        if i > 0 and frozen[i - 1][j]:
            a = min(a, T[i - 1][j])
        if i < ny - 1 and frozen[i + 1][j]:
            a = min(a, T[i + 1][j])
        b = _INF
        if j > 0 and frozen[i][j - 1]:
            b = min(b, T[i][j - 1])
        if j < nx - 1 and frozen[i][j + 1]:
            b = min(b, T[i][j + 1])
        if a == _INF and b == _INF:
            return _INF
        if a == _INF:
            return b + h
        if b == _INF:
            return a + h
        if abs(a - b) >= h:
            return min(a, b) + h
        return 0.5 * (a + b + math.sqrt(2.0 * h * h - (a - b) * (a - b)))

    while heap:
        t, i, j = heappop(heap)
        if frozen[i][j]:
            continue
        frozen[i][j] = True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            if not (0 <= ii < ny and 0 <= jj < nx):
                continue
            if frozen[ii][jj] or not ok[ii][jj]:
                continue
            cand = _solve(ii, jj)
            if cand < T[ii][jj]:
                T[ii][jj] = cand
                heappush(heap, (cand, ii, jj))
    return T


def eikonal_descent_direction(T: Grid, i: int, j: int,
                              dx: float = 1.0) -> Tuple[float, float]:
    """Unit vector (ux, uy) pointing down the potential from cell (i, j)."""
    ny = len(T)
    if ny == 0 or len(T[0]) == 0:
        raise PedestrianFlowError("the potential grid is empty.")
    nx = len(T[0])
    if not (0 <= i < ny and 0 <= j < nx):
        raise PedestrianFlowError("cell (%d, %d) is outside the grid." % (i, j))
    t0 = T[i][j]
    if t0 == _INF:
        return (0.0, 0.0)
    d = _positive(dx, "dx")
    left = T[i][j - 1] if j > 0 else _INF
    right = T[i][j + 1] if j < nx - 1 else _INF
    sx = 0.0
    if min(left, right) < t0:
        if left <= right:
            sx = -(t0 - left) / d
        else:
            sx = (t0 - right) / d
    up = T[i - 1][j] if i > 0 else _INF
    down = T[i + 1][j] if i < ny - 1 else _INF
    sy = 0.0
    if min(up, down) < t0:
        if up <= down:
            sy = -(t0 - up) / d
        else:
            sy = (t0 - down) / d
    norm = math.hypot(sx, sy)
    if norm <= _EPS:
        return (0.0, 0.0)
    return (sx / norm, sy / norm)


def potential_descent_path(T: Grid, start: Cell,
                           max_steps: int = 10000) -> List[Cell]:
    """Steepest-descent path down a potential field from ``start`` to a zero cell."""
    ny = len(T)
    if ny == 0 or len(T[0]) == 0:
        raise PedestrianFlowError("the potential grid is empty.")
    nx = len(T[0])
    i, j = start
    if not (0 <= i < ny and 0 <= j < nx):
        raise PedestrianFlowError("start cell is outside the grid.")
    if T[i][j] == _INF:
        raise PedestrianFlowError("start cell is unreachable.")
    path = [(i, j)]
    for _ in range(_positive_int(max_steps, "max_steps")):
        if T[i][j] <= 0.0:
            break
        best = None
        best_t = T[i][j]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ii, jj = i + di, j + dj
                if 0 <= ii < ny and 0 <= jj < nx and T[ii][jj] < best_t:
                    best_t = T[ii][jj]
                    best = (ii, jj)
        if best is None:
            break
        i, j = best
        path.append((i, j))
    return path


@dataclass
class HughesGrid:
    """A two-dimensional walkable domain carrying a pedestrian density field."""

    nx: int
    ny: int
    dx: float
    fd: PedestrianFD
    walkable: Mask = field(default_factory=list)
    exits: List[Cell] = field(default_factory=list)
    density: Grid = field(default_factory=list)
    exit_capacity: float = 2.0     # persons per second per exit cell
    discomfort: float = 0.0        # route-cost penalty g(rho) = 1/(1 + c rho)

    def __post_init__(self) -> None:
        self.nx = _positive_int(self.nx, "nx")
        self.ny = _positive_int(self.ny, "ny")
        self.dx = _positive(self.dx, "dx")
        if not self.walkable:
            self.walkable = [[True] * self.nx for _ in range(self.ny)]
        if not self.density:
            self.density = [[0.0] * self.nx for _ in range(self.ny)]
        if len(self.walkable) != self.ny or len(self.density) != self.ny:
            raise PedestrianFlowError("walkable/density must have ny rows.")
        # a RAGGED mask passes the row count but raises a bare IndexError from
        # fill()/people()/simulate_hughes the moment a short row is indexed
        if any(len(row) != self.nx for row in self.walkable):
            raise PedestrianFlowError(
                "every walkable row must have nx = %d entries (got %s)."
                % (self.nx, sorted({len(r) for r in self.walkable})))
        if any(len(row) != self.nx for row in self.density):
            raise PedestrianFlowError(
                "every density row must have nx = %d entries (got %s)."
                % (self.nx, sorted({len(r) for r in self.density})))
        if not self.exits:
            raise PedestrianFlowError("a Hughes grid needs at least one exit cell.")

    @property
    def cell_area(self) -> float:
        """Area of one cell (m^2)."""
        return self.dx * self.dx

    def people(self) -> float:
        """Total people stored on the grid."""
        return self.cell_area * math.fsum(
            self.density[i][j] for i in range(self.ny) for j in range(self.nx))

    def fill(self, rho: float, region: Optional[Callable[[int, int], bool]] = None
             ) -> "HughesGrid":
        """Set a uniform density over the walkable cells (optionally a sub-region)."""
        r = _non_negative(rho, "rho")
        for i in range(self.ny):
            for j in range(self.nx):
                if self.walkable[i][j] and (region is None or region(i, j)):
                    self.density[i][j] = r
        return self

    def copy(self) -> "HughesGrid":
        """Copy with an independent density field."""
        return HughesGrid(self.nx, self.ny, self.dx, self.fd,
                          [list(r) for r in self.walkable], list(self.exits),
                          [list(r) for r in self.density], self.exit_capacity,
                          self.discomfort)


@dataclass
class HughesResult:
    """Result of a Hughes continuum run with an exact people ledger."""

    dt: float
    steps: int
    time: float
    density: Grid
    potential: Grid
    history: List[Tuple[float, Grid]]
    evacuated: float
    stored_initial: float
    stored_final: float
    conservation_error: float
    egress_series: List[Tuple[float, float]]
    clearance_time: Optional[float]
    max_density: float          # peak density reached at any time, any cell


def simulate_hughes(grid: HughesGrid, duration: float, *, cfl: float = 0.4,
                    dt: Optional[float] = None, potential_every: int = 1,
                    record_every: int = 0,
                    clearance_fraction: float = 0.99) -> HughesResult:
    """Run Hughes' model: eikonal potential + upwind continuity on a 2-D grid."""
    g = grid.copy()
    fd = g.fd
    d = g.dx
    v_max = max(fd.v_free, _EPS)
    step_dt = (_positive(cfl, "cfl") * d / (2.0 * v_max)) if dt is None else \
        _positive(dt, "dt")
    total = _positive(duration, "duration")
    n_steps = max(1, int(math.ceil(total / step_dt)))
    every = max(1, int(potential_every))
    stored_initial = g.people()
    peak_density = max(max(r) for r in g.density)
    target = _in_unit(clearance_fraction, "clearance_fraction") * stored_initial
    evacuated = 0.0
    egress_series: List[Tuple[float, float]] = []
    history: List[Tuple[float, Grid]] = []
    clearance: Optional[float] = None
    potential: Grid = [[_INF] * g.nx for _ in range(g.ny)]
    t = 0.0
    if record_every:
        history.append((0.0, [list(r) for r in g.density]))

    for step in range(n_steps):
        if step % every == 0:
            speed = [[0.0] * g.nx for _ in range(g.ny)]
            for i in range(g.ny):
                for j in range(g.nx):
                    if g.walkable[i][j]:
                        base = fd.speed(g.density[i][j])
                        cost = 1.0 / (1.0 + g.discomfort * g.density[i][j])
                        # A jammed cell has speed 0, which would disconnect the
                        # potential and freeze everybody behind it for ever.
                        # The floor keeps the field connected while making the
                        # jam maximally unattractive.  It is deliberately far
                        # ABOVE fast_marching's impassability threshold (1e-12
                        # relative), and now SCALE-RELATIVE to v_free so the
                        # whole model is covariant under a change of units.
                        speed[i][j] = max(base * cost, 1e-6 * v_max)
            potential = fast_marching(speed, g.exits, g.walkable, d)
        # velocity field from the potential
        vx = [[0.0] * g.nx for _ in range(g.ny)]
        vy = [[0.0] * g.nx for _ in range(g.ny)]
        for i in range(g.ny):
            for j in range(g.nx):
                if not g.walkable[i][j]:
                    continue
                ux, uy = eikonal_descent_direction(potential, i, j, d)
                mag = fd.speed(g.density[i][j])
                vx[i][j] = mag * ux
                vy[i][j] = mag * uy
        # donor-cell fluxes with demand/supply limiting
        out_demand = [[0.0] * g.nx for _ in range(g.ny)]
        faces: List[Tuple[int, int, int, int, float]] = []
        for i in range(g.ny):
            for j in range(g.nx):
                if not g.walkable[i][j]:
                    continue
                for di, dj in ((0, 1), (1, 0)):
                    ii, jj = i + di, j + dj
                    if not (0 <= ii < g.ny and 0 <= jj < g.nx):
                        continue
                    if not g.walkable[ii][jj]:
                        continue
                    u = 0.5 * ((vx[i][j] + vx[ii][jj]) if dj else
                               (vy[i][j] + vy[ii][jj]))
                    if u >= 0.0:
                        send = g.density[i][j] * u
                        recv = supply_flow(fd, g.density[ii][jj])
                        q = min(send, recv)
                        if q > 0.0:
                            faces.append((i, j, ii, jj, q))
                            out_demand[i][j] += q * step_dt / d
                    else:
                        send = g.density[ii][jj] * (-u)
                        recv = supply_flow(fd, g.density[i][j])
                        q = min(send, recv)
                        if q > 0.0:
                            faces.append((ii, jj, i, j, q))
                            out_demand[ii][jj] += q * step_dt / d
        for (i, j) in g.exits:
            if g.walkable[i][j]:
                out_demand[i][j] += min(fd.flow(g.density[i][j]) * d,
                                        g.exit_capacity) * step_dt / (d * d)
        scale = [[1.0] * g.nx for _ in range(g.ny)]
        for i in range(g.ny):
            for j in range(g.nx):
                if out_demand[i][j] > g.density[i][j] > 0.0:
                    scale[i][j] = g.density[i][j] / out_demand[i][j]
                elif out_demand[i][j] > 0.0 and g.density[i][j] <= 0.0:
                    scale[i][j] = 0.0
        new = [list(r) for r in g.density]
        for (di_, dj_, ri, rj, q) in faces:
            move = q * step_dt / d * scale[di_][dj_]
            new[di_][dj_] -= move
            new[ri][rj] += move
        step_out = 0.0
        for (i, j) in g.exits:
            if not g.walkable[i][j]:
                continue
            flux = min(fd.flow(g.density[i][j]) * d, g.exit_capacity)
            move = flux * step_dt / (d * d) * scale[i][j]
            move = min(move, max(0.0, new[i][j]))
            new[i][j] -= move
            step_out += move * d * d
        g.density = new
        peak_density = max(peak_density, max(max(r) for r in new))
        evacuated = math.fsum([evacuated, step_out])
        t += step_dt
        egress_series.append((t, step_out / step_dt))
        if clearance is None and evacuated >= target:
            clearance = t
        if record_every and (step + 1) % record_every == 0:
            history.append((t, [list(r) for r in g.density]))

    stored_final = g.people()
    balance = stored_initial - evacuated
    return HughesResult(
        dt=step_dt, steps=n_steps, time=t,
        density=[list(r) for r in g.density], potential=potential,
        history=history, evacuated=evacuated, stored_initial=stored_initial,
        stored_final=stored_final,
        conservation_error=abs(stored_final - balance),
        egress_series=egress_series, clearance_time=clearance,
        max_density=peak_density)


# ===========================================================================
# 7. SOCIAL-FORCE MODEL (HELBING-MOLNAR)
# ===========================================================================
# m dv/dt = m (v0 e - v) / tau
#           + sum_j [ A exp((r_ij - d_ij)/B) + k g(r_ij - d_ij) ] n_ij
#           + sum_j kappa g(r_ij - d_ij) (dv_ji . t_ij) t_ij
#           + the same pair of terms against every wall,
# with g(x) = max(x, 0).  Integrated symplectically so that the driving term's
# damping, not the integrator, sets the relaxation.
#
# Stiffness note: the body-force spring has period 2 pi sqrt(m / k) ~ 0.16 s at
# the published constants, so contact-dominated runs (bottlenecks, panic) need
# dt <= 0.005 s; 0.01-0.02 s is ample while nobody is touching.  Semi-implicit
# Euler and velocity-Verlet both conserve the free-walking solution exactly: an
# agent already at its desired velocity feels zero force and drifts at v0.


@dataclass
class SocialForceParams:
    """Helbing-Molnar-Farkas-Vicsek parameters (SI units)."""

    mass: float = 80.0
    tau: float = 0.5
    a_social: float = 2000.0
    b_social: float = 0.08
    k_body: float = 120000.0
    kappa: float = 240000.0
    a_wall: float = 2000.0
    b_wall: float = 0.08
    radius: float = 0.3
    desired_speed: float = 1.34
    max_speed: float = 2.0


@dataclass
class Pedestrian:
    """One social-force agent."""

    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    desired_speed: float = 1.34
    radius: float = 0.3
    mass: float = 80.0
    ident: int = 0
    active: bool = True
    exit_time: Optional[float] = None

    def speed(self) -> float:
        """Current speed (m/s)."""
        return math.hypot(self.vx, self.vy)

    def distance_to_target(self) -> float:
        """Straight-line distance to the agent's target (m)."""
        return math.hypot(self.target_x - self.x, self.target_y - self.y)

    def copy(self) -> "Pedestrian":
        """Independent copy of this agent."""
        return Pedestrian(self.x, self.y, self.vx, self.vy, self.target_x,
                          self.target_y, self.desired_speed, self.radius,
                          self.mass, self.ident, self.active, self.exit_time)


@dataclass
class Wall:
    """A straight wall segment from (x1, y1) to (x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def nearest_point(self, x: float, y: float) -> Tuple[float, float]:
        """Closest point on the segment to (x, y)."""
        ax, ay = self.x2 - self.x1, self.y2 - self.y1
        denom = ax * ax + ay * ay
        # Only a genuinely ZERO-length wall is a point.  An absolute 1e-12 on
        # a squared length collapsed every wall shorter than a micrometre to
        # its first endpoint; the clamp below already handles short segments.
        if denom <= 0.0:
            return (self.x1, self.y1)
        t = ((x - self.x1) * ax + (y - self.y1) * ay) / denom
        t = max(0.0, min(1.0, t))
        return (self.x1 + t * ax, self.y1 + t * ay)


def social_forces(peds: Sequence[Pedestrian], walls: Sequence[Wall],
                  params: SocialForceParams) -> List[Tuple[float, float]]:
    """Total force on every agent: driving + social + wall + body + friction."""
    n = len(peds)
    out: List[Tuple[float, float]] = []
    for a in range(n):
        p = peds[a]
        if not p.active:
            out.append((0.0, 0.0))
            continue
        # driving term
        dxv = p.target_x - p.x
        dyv = p.target_y - p.y
        dist = math.hypot(dxv, dyv)
        if dist > _EPS:
            ex, ey = dxv / dist, dyv / dist
        else:
            ex, ey = 0.0, 0.0
        fx = p.mass * (p.desired_speed * ex - p.vx) / params.tau
        fy = p.mass * (p.desired_speed * ey - p.vy) / params.tau
        # pairwise terms
        for b in range(n):
            if b == a:
                continue
            o = peds[b]
            if not o.active:
                continue
            rx, ry = p.x - o.x, p.y - o.y
            dd = math.hypot(rx, ry)
            if dd <= _EPS:
                # Exactly coincident agents have no defined separation
                # direction.  SUBSTITUTE a unit normal -- do not divide the
                # un-normalised fallback by _EPS, which would multiply the
                # repulsion by 1e12 and blow the integrator up.  Orienting the
                # substitute by the pair's index keeps Newton's third law:
                # agent a is pushed +x and agent b exactly -x.
                nx_, ny_ = (1.0, 0.0) if a < b else (-1.0, 0.0)
                dd = 0.0
            else:
                nx_, ny_ = rx / dd, ry / dd
            overlap = (p.radius + o.radius) - dd
            rep = params.a_social * math.exp(overlap / params.b_social)
            fx += rep * nx_
            fy += rep * ny_
            if overlap > 0.0:
                fx += params.k_body * overlap * nx_
                fy += params.k_body * overlap * ny_
                tx, ty = -ny_, nx_
                dv_t = (o.vx - p.vx) * tx + (o.vy - p.vy) * ty
                fx += params.kappa * overlap * dv_t * tx
                fy += params.kappa * overlap * dv_t * ty
        # walls
        for w in walls:
            wx, wy = w.nearest_point(p.x, p.y)
            rx, ry = p.x - wx, p.y - wy
            dd = math.hypot(rx, ry)
            if dd <= _EPS:
                # Standing exactly on the wall: the offset direction is
                # undefined, so take the wall's own left-hand unit normal.
                # Again a SUBSTITUTION, never a division by _EPS.
                wax, way = w.x2 - w.x1, w.y2 - w.y1
                wlen = math.hypot(wax, way)
                nx_, ny_ = ((-way / wlen, wax / wlen) if wlen > 0.0
                            else (0.0, 1.0))
                dd = 0.0
            else:
                nx_, ny_ = rx / dd, ry / dd
            overlap = p.radius - dd
            rep = params.a_wall * math.exp(overlap / params.b_wall)
            fx += rep * nx_
            fy += rep * ny_
            if overlap > 0.0:
                fx += params.k_body * overlap * nx_
                fy += params.k_body * overlap * ny_
                tx, ty = -ny_, nx_
                dv_t = -(p.vx * tx + p.vy * ty)
                fx += params.kappa * overlap * dv_t * tx
                fy += params.kappa * overlap * dv_t * ty
        out.append((fx, fy))
    return out


def _cap_speed(p: Pedestrian, v_max: float) -> None:
    s = math.hypot(p.vx, p.vy)
    if s > v_max > 0.0:
        p.vx *= v_max / s
        p.vy *= v_max / s


def social_force_step(peds: List[Pedestrian], walls: Sequence[Wall],
                      params: SocialForceParams, dt: float,
                      method: str = "euler",
                      accel: Optional[List[Tuple[float, float]]] = None
                      ) -> List[Tuple[float, float]]:
    """Advance one step (semi-implicit Euler or velocity-Verlet); returns the accelerations."""
    h = _positive(dt, "dt")
    if method == "euler":
        forces = social_forces(peds, walls, params)
        acc = []
        for k, p in enumerate(peds):
            if not p.active:
                acc.append((0.0, 0.0))
                continue
            ax, ay = forces[k][0] / p.mass, forces[k][1] / p.mass
            p.vx += ax * h
            p.vy += ay * h
            _cap_speed(p, params.max_speed)
            p.x += p.vx * h
            p.y += p.vy * h
            acc.append((ax, ay))
        return acc
    if method == "verlet":
        if accel is None:
            forces = social_forces(peds, walls, params)
            accel = [(forces[k][0] / peds[k].mass, forces[k][1] / peds[k].mass)
                     for k in range(len(peds))]
        for k, p in enumerate(peds):
            if not p.active:
                continue
            p.vx += 0.5 * accel[k][0] * h
            p.vy += 0.5 * accel[k][1] * h
            p.x += p.vx * h
            p.y += p.vy * h
        forces = social_forces(peds, walls, params)
        acc = []
        for k, p in enumerate(peds):
            if not p.active:
                acc.append((0.0, 0.0))
                continue
            ax, ay = forces[k][0] / p.mass, forces[k][1] / p.mass
            p.vx += 0.5 * ax * h
            p.vy += 0.5 * ay * h
            _cap_speed(p, params.max_speed)
            acc.append((ax, ay))
        return acc
    raise PedestrianFlowError("method must be 'euler' or 'verlet'.")


@dataclass
class SocialForceResult:
    """Trajectories, exit times and diagnostics from a social-force run."""

    pedestrians: List[Pedestrian]
    trajectories: Dict[int, List[Tuple[float, float, float]]]
    exit_times: Dict[int, float]
    mean_exit_time: Optional[float]
    last_exit_time: Optional[float]
    evacuated: int
    remaining: int
    steps: int
    time: float
    dt: float
    min_separation: float
    final_separation: Optional[float]


def simulate_social_force(peds: Sequence[Pedestrian], walls: Sequence[Wall],
                          params: SocialForceParams, *, dt: float = 0.01,
                          steps: int = 1000, method: str = "euler",
                          exit_test: Optional[Callable[[Pedestrian], bool]] = None,
                          record_every: int = 0) -> SocialForceResult:
    """Integrate a social-force crowd; agents passing ``exit_test`` are deactivated."""
    agents = [p.copy() for p in peds]
    n_steps = _positive_int(steps, "steps")
    h = _positive(dt, "dt")
    trajectories: Dict[int, List[Tuple[float, float, float]]] = {
        p.ident: [(0.0, p.x, p.y)] for p in agents}
    exit_times: Dict[int, float] = {}
    accel: Optional[List[Tuple[float, float]]] = None
    min_sep = _INF
    t = 0.0
    for step in range(n_steps):
        accel = social_force_step(agents, walls, params, h, method, accel)
        t += h
        active = [p for p in agents if p.active]
        for a in range(len(active)):
            for b in range(a + 1, len(active)):
                sep = math.hypot(active[a].x - active[b].x,
                                 active[a].y - active[b].y)
                if sep < min_sep:
                    min_sep = sep
        if exit_test is not None:
            for p in agents:
                if p.active and exit_test(p):
                    p.active = False
                    p.exit_time = t
                    exit_times[p.ident] = t
        if record_every and (step + 1) % record_every == 0:
            for p in agents:
                if p.active:
                    trajectories[p.ident].append((t, p.x, p.y))
        if exit_test is not None and not any(p.active for p in agents):
            break
    times = sorted(exit_times.values())
    final_sep = None
    if len(agents) == 2:
        final_sep = math.hypot(agents[0].x - agents[1].x,
                               agents[0].y - agents[1].y)
    return SocialForceResult(
        pedestrians=agents, trajectories=trajectories, exit_times=exit_times,
        mean_exit_time=(_mean(times) if times else None),
        last_exit_time=(times[-1] if times else None),
        evacuated=len(times), remaining=sum(1 for p in agents if p.active),
        steps=step + 1, time=t, dt=h,
        min_separation=(min_sep if min_sep < _INF else 0.0),
        final_separation=final_sep)


def social_force_equilibrium_separation(params: SocialForceParams,
                                        desired_speed: Optional[float] = None,
                                        radius: Optional[float] = None) -> float:
    """Separation at which two head-on agents' repulsion balances their driving force."""
    v0 = params.desired_speed if desired_speed is None else _positive(
        desired_speed, "desired_speed")
    r = params.radius if radius is None else _positive(radius, "radius")
    drive = params.mass * v0 / params.tau
    contact = 2.0 * r
    if drive <= params.a_social:
        # no body contact: A exp((2r - d)/B) = m v0 / tau has a closed form
        return contact - params.b_social * math.log(drive / params.a_social)

    def residual(d: float) -> float:
        """Repulsion minus the driving force at separation d (zero at equilibrium)."""
        ov = contact - d
        rep = params.a_social * math.exp(ov / params.b_social)
        if ov > 0.0:
            rep += params.k_body * ov
        return rep - drive

    return _bisect_root(residual, 1e-9, contact)


# ===========================================================================
# 8. FLOOR-FIELD CELLULAR AUTOMATON
# ===========================================================================
# Burstedde-Klauck-Schadschneider-Zittartz: a parallel-update lattice gas whose
# transition probabilities are driven by a STATIC field (distance to the exit,
# solved here by fast marching so diagonals cost what they should) and a
# DYNAMIC field of virtual traces left by the people who have already walked.
# Conflicts on a shared target cell are resolved with the friction parameter mu.
#
# Two calibration facts worth knowing before comparing a CA against the PDE.
# (1) Free speed is exactly cell_size / time_step, but only if k_s is large
#     enough that the walk is not a downhill-biased random walk: at the classic
#     k_s = 2.5 with 0.4 m cells an agent advances ~0.67 cells per step.
# (2) The parallel update lets at most one person per exit cell per time step,
#     so an exit's ceiling is 1 / (cell_size * time_step) P/m/s -- well above
#     the observed ~1.3 P/m/s unless the time step is chosen to match.
# ``cross_model_egress`` therefore compares FREE-FLOW traversal, where all three
# models are calibrated to the same v0 and must agree.


def static_floor_field(walkable: Mask, exits: Sequence[Cell],
                       cell_size: float = 0.4) -> Grid:
    """Distance-to-nearest-exit field (metres) by fast marching at unit speed."""
    ny = len(walkable)
    if ny == 0:
        raise PedestrianFlowError("the walkable mask is empty.")
    nx = len(walkable[0])
    speed = [[1.0 if walkable[i][j] else 0.0 for j in range(nx)]
             for i in range(ny)]
    return fast_marching(speed, exits, walkable, _positive(cell_size, "cell_size"))


@dataclass
class CAResult:
    """Egress statistics from a floor-field cellular-automaton run."""

    egress_times: Dict[int, float]
    mean_egress: Optional[float]
    last_egress: Optional[float]
    median_egress: Optional[float]
    evacuated: int
    remaining: int
    steps: int
    time: float
    flow_series: List[Tuple[float, float]]
    occupancy_series: List[Tuple[float, int]]


class FloorFieldCA:
    """Floor-field CA with a static distance field and a decaying dynamic field."""

    def __init__(self, walkable: Mask, exits: Sequence[Cell],
                 agents: Sequence[Cell], *, k_s: float = 2.5, k_d: float = 1.0,
                 alpha: float = 0.3, delta: float = 0.3, friction: float = 0.0,
                 cell_size: float = 0.4, time_step: float = 0.3,
                 diagonal: bool = True, seed: int = 0) -> None:
        self.walkable = [list(r) for r in walkable]
        self.ny = len(self.walkable)
        if self.ny == 0:
            raise PedestrianFlowError("the walkable mask is empty.")
        self.nx = len(self.walkable[0])
        self.exits = [tuple(c) for c in exits]
        if not self.exits:
            raise PedestrianFlowError("a floor-field CA needs at least one exit.")
        self.k_s = _as_float(k_s, "k_s")
        self.k_d = _as_float(k_d, "k_d")
        self.alpha = _in_unit(alpha, "alpha")
        self.delta = _in_unit(delta, "delta")
        self.friction = _in_unit(friction, "friction")
        self.cell_size = _positive(cell_size, "cell_size")
        self.time_step = _positive(time_step, "time_step")
        self.diagonal = bool(diagonal)
        self.rng = random.Random(seed)
        self.static_field = static_floor_field(self.walkable, self.exits,
                                               self.cell_size)
        self.dynamic_field: Grid = [[0.0] * self.nx for _ in range(self.ny)]
        self.agents: Dict[int, Cell] = {}
        self.egress_times: Dict[int, float] = {}
        self.time = 0.0
        self.steps = 0
        for k, cell in enumerate(agents):
            i, j = int(cell[0]), int(cell[1])
            if not (0 <= i < self.ny and 0 <= j < self.nx):
                raise PedestrianFlowError("agent cell (%d, %d) is outside the grid."
                                          % (i, j))
            if not self.walkable[i][j]:
                raise PedestrianFlowError("agent cell (%d, %d) is not walkable."
                                          % (i, j))
            self.agents[k] = (i, j)

    @property
    def free_speed(self) -> float:
        """Free walking speed implied by the cell size and time step (m/s)."""
        return self.cell_size / self.time_step

    def _neighbourhood(self, i: int, j: int) -> List[Cell]:
        cells = [(i, j)]
        offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
        if self.diagonal:
            offsets = offsets + ((-1, -1), (-1, 1), (1, -1), (1, 1))
        for di, dj in offsets:
            ii, jj = i + di, j + dj
            if 0 <= ii < self.ny and 0 <= jj < self.nx and self.walkable[ii][jj]:
                if self.static_field[ii][jj] < _INF:
                    cells.append((ii, jj))
        return cells

    def _diffuse(self) -> None:
        nxt = [[0.0] * self.nx for _ in range(self.ny)]
        for i in range(self.ny):
            for j in range(self.nx):
                if not self.walkable[i][j]:
                    continue
                nb = []
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < self.ny and 0 <= jj < self.nx \
                            and self.walkable[ii][jj]:
                        nb.append(self.dynamic_field[ii][jj])
                keep = (1.0 - self.alpha) * self.dynamic_field[i][j]
                spread = (self.alpha * math.fsum(nb) / len(nb)) if nb else 0.0
                nxt[i][j] = (1.0 - self.delta) * (keep + spread)
        self.dynamic_field = nxt

    def step(self) -> int:
        """Advance one parallel update; returns the number of people who left."""
        occupied = set(self.agents.values())
        wishes: Dict[Cell, List[Tuple[int, float]]] = {}
        for ident, (i, j) in self.agents.items():
            s_here = self.static_field[i][j]
            d_here = self.dynamic_field[i][j]
            cands = []
            weights = []
            for (ii, jj) in self._neighbourhood(i, j):
                if (ii, jj) != (i, j) and (ii, jj) in occupied:
                    continue
                ds = -(self.static_field[ii][jj] - s_here)
                dd = self.dynamic_field[ii][jj] - d_here
                cands.append((ii, jj))
                weights.append(math.exp(self.k_s * ds + self.k_d * dd))
            tot = math.fsum(weights)
            if tot <= _EPS:
                wishes.setdefault((i, j), []).append((ident, 1.0))
                continue
            pick = self.rng.random() * tot
            acc = 0.0
            chosen = cands[-1]
            chosen_p = weights[-1] / tot
            for c, w in zip(cands, weights):
                acc += w
                if pick <= acc:
                    chosen = c
                    chosen_p = w / tot
                    break
            wishes.setdefault(chosen, []).append((ident, chosen_p))
        moves: Dict[int, Cell] = {}
        for cell, claimants in wishes.items():
            if len(claimants) == 1:
                moves[claimants[0][0]] = cell
                continue
            if self.rng.random() < self.friction:
                continue                      # friction: nobody moves this step
            tot = math.fsum(p for _, p in claimants)
            pick = self.rng.random() * (tot if tot > _EPS else 1.0)
            acc = 0.0
            winner = claimants[-1][0]
            for ident, p in claimants:
                acc += p
                if pick <= acc:
                    winner = ident
                    break
            moves[winner] = cell
        self.steps += 1
        self.time += self.time_step
        left = 0
        for ident, cell in list(moves.items()):
            old = self.agents[ident]
            if cell != old:
                self.dynamic_field[old[0]][old[1]] += 1.0
                self.agents[ident] = cell
        for ident, cell in list(self.agents.items()):
            if cell in self.exits:
                self.egress_times[ident] = self.time
                del self.agents[ident]
                left += 1
        self._diffuse()
        return left

    def run(self, max_steps: int = 2000) -> CAResult:
        """Run until everyone has left or ``max_steps`` is reached."""
        flow: List[Tuple[float, float]] = []
        occ: List[Tuple[float, int]] = []
        for _ in range(_positive_int(max_steps, "max_steps")):
            if not self.agents:
                break
            left = self.step()
            flow.append((self.time, left / self.time_step))
            occ.append((self.time, len(self.agents)))
        times = sorted(self.egress_times.values())
        return CAResult(
            egress_times=dict(self.egress_times),
            mean_egress=(_mean(times) if times else None),
            last_egress=(times[-1] if times else None),
            median_egress=(_quantile(times, 0.5) if times else None),
            evacuated=len(times), remaining=len(self.agents),
            steps=self.steps, time=self.time, flow_series=flow,
            occupancy_series=occ)


# ===========================================================================
# 9. LEVEL OF SERVICE AND CROWD RISK
# ===========================================================================
# Fruin's A-F bands, stated in the units the designer actually has to hand:
# the pedestrian area module (m^2 per person) and the unit flow rate
# (persons per minute per metre of effective width).  The imperial originals
# (35/25/15/10/5 ft^2 for walkways, 7/10/15/20/25 P/min/ft) are converted once,
# here, so no downstream code has to know about feet.

_LOS_AREA_BANDS: Dict[str, Tuple[Tuple[str, float, float, str], ...]] = {
    "walkway": (
        # 35 ft^2 * 0.09290304 = 3.2516 m^2 -> 3.25, not 3.24.  Every other
        # boundary in these three tables is the exact two-decimal conversion
        # of its published imperial integer; this one was a transcription slip.
        ("A", 3.25, _INF, "free circulation, no conflicts"),
        ("B", 2.32, 3.25, "minor conflicts, speed freely chosen"),
        ("C", 1.39, 2.32, "speeds slightly restricted, passing constrained"),
        ("D", 0.93, 1.39, "speeds restricted, reverse flows difficult"),
        ("E", 0.46, 0.93, "shuffling, all speeds reduced, at capacity"),
        ("F", 0.0, 0.46, "complete breakdown, involuntary contact"),
    ),
    "queueing": (
        ("A", 1.21, _INF, "free standing and circulation"),
        ("B", 0.93, 1.21, "standing comfortably, circulation restricted"),
        ("C", 0.65, 0.93, "standing without touching, limited circulation"),
        ("D", 0.28, 0.65, "standing close, circulation severely restricted"),
        ("E", 0.19, 0.28, "touching unavoidable, no internal circulation"),
        ("F", 0.0, 0.19, "crush density, potential for panic"),
    ),
    "stairway": (
        ("A", 1.86, _INF, "free choice of speed, passing possible"),
        ("B", 1.58, 1.86, "minor conflicts, passing occasionally blocked"),
        ("C", 1.11, 1.58, "speeds restricted, reverse flows minor conflict"),
        ("D", 0.74, 1.11, "speeds restricted, passing not possible"),
        ("E", 0.46, 0.74, "shuffling, all speeds reduced, at capacity"),
        ("F", 0.0, 0.46, "complete breakdown, momentary stoppages"),
    ),
}

_LOS_FLOW_BANDS: Dict[str, Tuple[Tuple[str, float, float, str], ...]] = {
    "walkway": (
        ("A", 0.0, 23.0, "<= 7 P/min/ft"),
        ("B", 23.0, 33.0, "7-10 P/min/ft"),
        ("C", 33.0, 49.0, "10-15 P/min/ft"),
        ("D", 49.0, 66.0, "15-20 P/min/ft"),
        ("E", 66.0, 82.0, "20-25 P/min/ft, capacity"),
        ("F", 82.0, _INF, "> 25 P/min/ft, unstable"),
    ),
    "stairway": (
        ("A", 0.0, 16.0, "<= 5 P/min/ft"),
        ("B", 16.0, 23.0, "5-7 P/min/ft"),
        ("C", 23.0, 33.0, "7-10 P/min/ft"),
        ("D", 33.0, 43.0, "10-13 P/min/ft"),
        ("E", 43.0, 56.0, "13-17 P/min/ft, capacity"),
        ("F", 56.0, _INF, "> 17 P/min/ft, unstable"),
    ),
}


@dataclass
class LOSResult:
    """A Fruin level-of-service grade with the band it fell into."""

    grade: str
    facility: str
    basis: str
    value: float
    lower: float
    upper: float
    description: str
    density: Optional[float] = None


def los_bands(facility: str = "walkway",
              basis: str = "area") -> List[Dict[str, Any]]:
    """The Fruin band table for a facility and basis ('area' or 'flow')."""
    fac = str(facility).strip().lower()
    bas = str(basis).strip().lower()
    table = _LOS_AREA_BANDS if bas == "area" else _LOS_FLOW_BANDS
    if bas not in ("area", "flow"):
        raise PedestrianFlowError("basis must be 'area' or 'flow'.")
    if fac not in table:
        raise PedestrianFlowError(
            "unknown facility %r for basis %r (known: %s)."
            % (facility, basis, ", ".join(sorted(table))))
    return [{"grade": g, "lower": lo, "upper": (None if hi == _INF else hi),
             "description": desc} for (g, lo, hi, desc) in table[fac]]


def fruin_los(value: float, facility: str = "walkway",
              basis: str = "area") -> LOSResult:
    """Fruin level of service from an area module (m^2/P) or a unit flow (P/min/m)."""
    v = _non_negative(value, "value")
    fac = str(facility).strip().lower()
    bas = str(basis).strip().lower()
    if bas == "area":
        table = _LOS_AREA_BANDS
    elif bas == "flow":
        table = _LOS_FLOW_BANDS
    else:
        raise PedestrianFlowError("basis must be 'area' or 'flow'.")
    if fac not in table:
        raise PedestrianFlowError(
            "unknown facility %r for basis %r (known: %s)."
            % (facility, basis, ", ".join(sorted(table))))
    for (grade, lo, hi, desc) in table[fac]:
        if lo <= v < hi or (hi == _INF and v >= lo):
            dens = (1.0 / v if (bas == "area" and v > _EPS) else None)
            return LOSResult(grade, fac, bas, v, lo, hi, desc, dens)
    last = table[fac][-1]
    return LOSResult(last[0], fac, bas, v, last[1], last[2], last[3], None)


def fruin_los_from_density(rho: float, facility: str = "walkway") -> LOSResult:
    """Fruin level of service from a density in P/m^2 (area module = 1/rho)."""
    r = _non_negative(rho, "rho")
    fac = str(facility).strip().lower()
    if fac not in _LOS_AREA_BANDS:
        # validate BEFORE the early branch: otherwise an unknown facility
        # raises KeyError below the density epsilon and PedestrianFlowError
        # above it, so the exception type depends on the density.
        raise PedestrianFlowError(
            "unknown facility %r for basis 'area' (known: %s)."
            % (facility, ", ".join(sorted(_LOS_AREA_BANDS))))
    if r <= _EPS:
        # an empty facility is an unbounded area module: always grade A
        res = fruin_los(_LOS_AREA_BANDS[fac][0][1], facility, "area")
        res.density = 0.0
        return res
    res = fruin_los(1.0 / r, facility, "area")
    res.density = r
    return res


@dataclass
class CrowdBand:
    """A named crowd-density band with a risk score from 0 (free) to 5 (crush)."""

    name: str
    lower: float
    upper: float
    risk: int
    description: str
    density: float = 0.0


_CROWD_BANDS: Tuple[Tuple[str, float, float, int, str], ...] = (
    ("free", 0.0, 1.0, 0, "unimpeded walking, personal space preserved"),
    ("comfortable", 1.0, 2.0, 1, "speed lightly reduced, overtaking possible"),
    ("constrained", 2.0, 3.0, 2, "speed clearly reduced, overtaking difficult"),
    ("congested", 3.0, 4.0, 3, "shuffling, involuntary contact, LOS E/F"),
    ("critical", 4.0, 5.0, 4, "no free movement, pressure waves possible"),
    ("crush", 5.0, _INF, 5, "crush risk: turbulence, compressive asphyxia"),
)


def crowd_density_band(rho: float) -> CrowdBand:
    """Classify a crowd density (P/m^2) into a named band with a risk score."""
    r = _non_negative(rho, "rho")
    for (name, lo, hi, risk, desc) in _CROWD_BANDS:
        if lo <= r < hi or (hi == _INF and r >= lo):
            return CrowdBand(name, lo, hi, risk, desc, r)
    last = _CROWD_BANDS[-1]
    return CrowdBand(last[0], last[1], last[2], last[3], last[4], r)


def crowd_pressure(density: float,
                   velocities: Sequence[Sequence[float]]) -> float:
    """Helbing's crowd pressure P = rho * Var(v) in 1/s^2 (turbulence above ~0.02)."""
    r = _non_negative(density, "density")
    # Any fixed number of components is accepted and EVERY component counts:
    # truncating to the first two would report a crowd whose entire velocity
    # variance sits in a discarded component at pressure 0.0 -- maximally
    # safe -- which is the one direction a safety metric must never err in.
    vecs: List[Tuple[float, ...]] = []
    for v in velocities:
        if isinstance(v, (int, float)):
            vals = [float(v)]
        else:
            vals = _as_floats(list(v), "velocity")
        if not vals:
            raise PedestrianFlowError("velocity entries need at least one component.")
        vecs.append(tuple(vals))
    if not vecs:
        raise PedestrianFlowError("crowd_pressure needs at least one velocity.")
    dim = max(len(v) for v in vecs)
    if any(len(v) != dim for v in vecs):
        raise PedestrianFlowError(
            "every velocity needs the same number of components (got %s)."
            % sorted({len(v) for v in vecs}))
    mean = [math.fsum(v[k] for v in vecs) / len(vecs) for k in range(dim)]
    var = math.fsum(math.fsum((v[k] - mean[k]) ** 2 for k in range(dim))
                    for v in vecs) / len(vecs)
    return r * var


def crush_risk(density: float, pressure: Optional[float] = None,
               pressure_threshold: float = 0.02) -> Dict[str, Any]:
    """Combine a density band and (optionally) the crowd pressure into a risk verdict."""
    band = crowd_density_band(density)
    thr = _positive(pressure_threshold, "pressure_threshold")
    score = band.risk
    notes = [band.description]
    if pressure is not None:
        p = _non_negative(pressure, "pressure")
        if p >= thr:
            score = max(score, 4)
            notes.append("crowd pressure %.4f 1/s^2 exceeds the turbulence "
                         "threshold %.3f" % (p, thr))
        elif p >= 0.5 * thr:
            score = max(score, 3)
            notes.append("crowd pressure %.4f 1/s^2 approaching turbulence" % p)
    level = ("safe", "watch", "caution", "elevated", "danger", "critical")[
        min(score, 5)]
    return {
        "density": _non_negative(density, "density"),
        "band": band.name,
        "pressure": pressure,
        "score": score,
        "level": level,
        "notes": notes,
    }


# ===========================================================================
# 10. CROSSINGS, DELAY AND SIGNAL TIMING
# ===========================================================================
def critical_gap(crossing_width: float, walking_speed: float = 1.2,
                 safety_margin: float = 1.5, start_up: float = 0.0) -> float:
    """Gap a pedestrian needs to cross: width / speed + safety margin + start-up (s)."""
    w = _positive(crossing_width, "crossing_width")
    v = _positive(walking_speed, "walking_speed")
    return w / v + _non_negative(safety_margin, "safety_margin") + \
        _non_negative(start_up, "start_up")


def gap_acceptance_probability(vehicle_flow: float,
                               critical_gap_s: float) -> float:
    """P(a random gap in a Poisson stream is acceptable) = exp(-q t_c), q in veh/h."""
    q = _non_negative(vehicle_flow, "vehicle_flow") / 3600.0
    tc = _non_negative(critical_gap_s, "critical_gap_s")
    return math.exp(-q * tc)


def adams_delay(vehicle_flow: float, critical_gap_s: float) -> float:
    """Adams' (1936) mean pedestrian delay (e^(q t_c) - q t_c - 1)/q, q in veh/h."""
    q = _non_negative(vehicle_flow, "vehicle_flow") / 3600.0
    tc = _non_negative(critical_gap_s, "critical_gap_s")
    x = q * tc
    if x < 3e-3:
        # Five-term Maclaurin series of (e^x - x - 1)/q
        #   = t_c x (1/2 + x/6 + x^2/24 + x^3/120 + x^4/720) + O(t_c x^6),
        # which needs no division and so also covers q = 0 exactly.
        # The cut sits where the series truncation (~x^5/2520) and the
        # cancellation in the closed form cross, BOTH below 1e-14 relative.
        # The old cut at x = 1e-6 sat on the closed form's WORST point: the
        # relative error there peaked at 1.2e-4, ten orders worse than either
        # branch is capable of.
        return tc * x * (0.5 + x * (1.0 / 6.0 + x * (1.0 / 24.0
                                    + x * (1.0 / 120.0 + x / 720.0))))
    if x > 700.0:
        raise PedestrianFlowError(
            "q * t_c = %.3g is too large for Adams' formula (the crossing never "
            "clears)." % x)
    # expm1 keeps the subtraction of 1 exact; only x is then cancelled.
    return (math.expm1(x) - x) / q


def signalised_pedestrian_delay(cycle: float, walk_green: float) -> float:
    """Uniform pedestrian delay at a signal: (C - g)^2 / (2C) seconds."""
    c = _positive(cycle, "cycle")
    g = _non_negative(walk_green, "walk_green")
    if g > c:
        raise PedestrianFlowError(
            "the pedestrian green (%.6g s) cannot exceed the cycle (%.6g s)." % (g, c))
    return (c - g) ** 2 / (2.0 * c)


def two_stage_crossing_delay(vehicle_flow_1: float, vehicle_flow_2: float,
                             critical_gap_1: float, critical_gap_2: float,
                             refuge_dwell: float = 0.0) -> float:
    """Total gap-acceptance delay crossing two half-carriageways via a refuge (s)."""
    d1 = adams_delay(vehicle_flow_1, critical_gap_1)
    d2 = adams_delay(vehicle_flow_2, critical_gap_2)
    return d1 + d2 + _non_negative(refuge_dwell, "refuge_dwell")


_PED_DELAY_LOS: Tuple[Tuple[str, float, float, str], ...] = (
    ("A", 0.0, 5.0, "virtually no conflict, immediate crossing"),
    ("B", 5.0, 10.0, "occasional wait, compliance likely"),
    ("C", 10.0, 20.0, "noticeable wait, compliance still likely"),
    ("D", 20.0, 30.0, "frequent wait, risk-taking begins"),
    ("E", 30.0, 45.0, "long wait, non-compliance common"),
    ("F", 45.0, _INF, "very long wait, non-compliance near certain"),
)


def pedestrian_delay_los(delay: float) -> LOSResult:
    """HCM-style pedestrian level of service from the average crossing delay (s)."""
    d = _non_negative(delay, "delay")
    for (grade, lo, hi, desc) in _PED_DELAY_LOS:
        if lo <= d < hi or (hi == _INF and d >= lo):
            return LOSResult(grade, "crossing", "delay", d, lo, hi, desc, None)
    last = _PED_DELAY_LOS[-1]
    return LOSResult(last[0], "crossing", "delay", d, last[1], last[2], last[3])


@dataclass
class CrossingDelayResult:
    """Delay, crossing time and level of service for one pedestrian crossing."""

    control: str
    delay: float
    crossing_time: float
    total_time: float
    critical_gap: Optional[float]
    stages: int
    level_of_service: str
    detail: Dict[str, Any] = field(default_factory=dict)


def crossing_delay(crossing_width: float, *, control: str = "unsignalised",
                   vehicle_flow: float = 0.0,
                   vehicle_flow_2: Optional[float] = None,
                   walking_speed: float = 1.2, safety_margin: float = 1.5,
                   cycle: Optional[float] = None,
                   walk_green: Optional[float] = None,
                   refuge: bool = False,
                   refuge_dwell: float = 0.0) -> CrossingDelayResult:
    """Pedestrian crossing delay for a signalised, unsignalised or two-stage crossing."""
    w = _positive(crossing_width, "crossing_width")
    v = _positive(walking_speed, "walking_speed")
    ctl = str(control).strip().lower()
    if ctl == "signalised":
        if cycle is None or walk_green is None:
            raise PedestrianFlowError(
                "a signalised crossing needs both 'cycle' and 'walk_green'.")
        d = signalised_pedestrian_delay(cycle, walk_green)
        ct = w / v
        res = CrossingDelayResult(
            "signalised", d, ct, d + ct, None, 1,
            pedestrian_delay_los(d).grade,
            {"cycle": float(cycle), "walkGreen": float(walk_green)})
        return res
    if ctl != "unsignalised":
        raise PedestrianFlowError("control must be 'signalised' or 'unsignalised'.")
    if refuge:
        half = 0.5 * w
        tc = critical_gap(half, v, safety_margin)
        q1 = vehicle_flow
        q2 = vehicle_flow if vehicle_flow_2 is None else vehicle_flow_2
        d = two_stage_crossing_delay(q1, q2, tc, tc, refuge_dwell)
        ct = w / v + refuge_dwell
        return CrossingDelayResult(
            "unsignalised-two-stage", d, ct, d + ct, tc, 2,
            pedestrian_delay_los(d).grade,
            {"stageGap": tc, "flow1": float(q1), "flow2": float(q2),
             "stage1Delay": adams_delay(q1, tc),
             "stage2Delay": adams_delay(q2, tc)})
    tc = critical_gap(w, v, safety_margin)
    q = vehicle_flow if vehicle_flow_2 is None else \
        _non_negative(vehicle_flow, "vehicle_flow") + \
        _non_negative(vehicle_flow_2, "vehicle_flow_2")
    d = adams_delay(q, tc)
    ct = w / v
    return CrossingDelayResult(
        "unsignalised-one-stage", d, ct, d + ct, tc, 1,
        pedestrian_delay_los(d).grade,
        {"combinedFlow": float(q),
         "acceptanceProbability": gap_acceptance_probability(q, tc)})


def pedestrian_minimum_green(crossing_distance: float,
                             walking_speed: float = 1.2,
                             start_up: float = 0.0,
                             mobility_impaired: bool = False) -> float:
    """Minimum pedestrian green: crossing distance / walking speed (+ start-up)."""
    d = _positive(crossing_distance, "crossing_distance")
    v = 1.0 if mobility_impaired else _positive(walking_speed, "walking_speed")
    return d / v + _non_negative(start_up, "start_up")


@dataclass
class PedestrianPhase:
    """A pedestrian phase broken into walk, clearance and total green."""

    walk_interval: float
    clearance_interval: float
    minimum_green: float
    total_pedestrian_time: float
    walking_speed: float
    crossing_distance: float
    cycle: Optional[float] = None
    delay: Optional[float] = None


def pedestrian_phase_times(crossing_distance: float,
                           walking_speed: float = 1.2,
                           walk_interval: float = 7.0,
                           start_up: float = 0.0,
                           mobility_impaired: bool = False,
                           cycle: Optional[float] = None) -> PedestrianPhase:
    """Walk + clearance intervals for a crossing, with the delay if a cycle is given."""
    v = 1.0 if mobility_impaired else _positive(walking_speed, "walking_speed")
    clearance = pedestrian_minimum_green(crossing_distance, v, start_up,
                                         mobility_impaired)
    walk = _non_negative(walk_interval, "walk_interval")
    total = walk + clearance
    delay = None
    if cycle is not None:
        delay = signalised_pedestrian_delay(cycle, min(total, float(cycle)))
    return PedestrianPhase(walk, clearance, clearance, total, v,
                           _positive(crossing_distance, "crossing_distance"),
                           (None if cycle is None else float(cycle)), delay)


@dataclass
class PlatoonResult:
    """Robertson platoon dispersion applied to a pedestrian arrival profile."""

    profile: List[float]
    smoothing: float
    lag_steps: int
    lag_seconds: float
    mass_in: float
    mass_out: float
    peak_in: float
    peak_out: float
    peak_reduction: float


def platoon_dispersion(profile: Sequence[float], travel_time: float,
                       dt: float = 1.0, alpha: float = 0.5,
                       beta: float = 0.8,
                       extra_steps: Optional[int] = None) -> PlatoonResult:
    """Robertson's recursive platoon-dispersion filter on a pedestrian arrival profile."""
    prof = _as_floats(profile, "profile")
    if not prof:
        raise PedestrianFlowError("the arrival profile is empty.")
    if any(v < 0.0 for v in prof):
        raise PedestrianFlowError("arrival profile entries must be >= 0.")
    tt = _positive(travel_time, "travel_time")
    step = _positive(dt, "dt")
    a = _positive(alpha, "alpha")
    b = _positive(beta, "beta")
    ta = tt / step                     # travel time in profile steps
    lag = int(round(b * ta))
    f = 1.0 / (1.0 + a * b * ta)
    if extra_steps is None:
        # run long enough that the geometric tail (1 - f)^n is negligible
        if f >= 1.0:
            tail = 1
        else:
            tail = int(math.ceil(math.log(1e-12) / math.log(1.0 - f)))
        extra_steps = min(max(tail, 1), 20000)
    else:
        extra_steps = int(extra_steps)
        if extra_steps < 0:
            raise PedestrianFlowError(
                "extra_steps must be >= 0 (got %d)." % extra_steps)
    n_out = len(prof) + lag + int(extra_steps)
    out = [0.0] * n_out
    prev = 0.0
    for i in range(n_out):
        src = prof[i - lag] if 0 <= i - lag < len(prof) else 0.0
        prev = f * src + (1.0 - f) * prev
        out[i] = prev
    mass_in = math.fsum(prof)
    mass_out = math.fsum(out)
    return PlatoonResult(out, f, lag, lag * step, mass_in, mass_out,
                         max(prof), max(out),
                         (1.0 - max(out) / max(prof)) if max(prof) > 0 else 0.0)


# ===========================================================================
# 11. ROUTE CHOICE -- MULTINOMIAL AND PATH-SIZE LOGIT
# ===========================================================================
@dataclass
class WalkRoute:
    """A candidate walking route with the attributes a pedestrian trades off."""

    name: str
    length: float
    walk_time: Optional[float] = None
    crossings: int = 0
    crossing_delay: float = 0.0
    stress: float = 0.0            # 0 = calm, 1 = hostile
    comfort: float = 1.0           # 0 = poor, 1 = excellent
    gradient: float = 0.0          # mean signed slope (rise / run)
    links: Tuple[str, ...] = ()

    def time(self, walking_speed: float = 1.3) -> float:
        """Walking time (s): given, or length / Tobler speed on the mean gradient."""
        if self.walk_time is not None:
            return _positive(self.walk_time, "walk_time")
        v = tobler_walking_speed(self.gradient, walking_speed)
        return _positive(self.length, "length") / max(v, _EPS)


@dataclass
class RouteCoefficients:
    """Linear utility weights (per minute, per crossing, per kilometre)."""

    time: float = -0.08
    delay: float = -0.12
    crossing: float = -0.15
    stress: float = -0.60
    comfort: float = 0.40
    gradient: float = -1.50
    constant: float = 0.0


def route_utility(route: WalkRoute, coef: Optional[RouteCoefficients] = None,
                  walking_speed: float = 1.3) -> float:
    """Linear-in-parameters utility of a walking route."""
    c = coef if coef is not None else RouteCoefficients()
    minutes = route.time(walking_speed) / 60.0
    km = _positive(route.length, "length") / 1000.0
    climb = max(0.0, route.gradient) * route.length / 100.0   # metres of climb / 100
    return math.fsum([
        c.constant,
        c.time * minutes,
        c.delay * (_non_negative(route.crossing_delay, "crossing_delay") / 60.0),
        c.crossing * float(route.crossings),
        c.stress * _in_unit(route.stress, "stress") * km,
        c.comfort * _in_unit(route.comfort, "comfort") * km,
        c.gradient * climb,
    ])


@dataclass
class RouteChoiceResult:
    """Utilities, choice probabilities and the logsum for a walking choice set."""

    names: List[str]
    utilities: List[float]
    probabilities: List[float]
    logsum: float
    scale: float
    path_size: Optional[List[float]] = None
    best: str = ""

    def as_dict(self) -> Dict[str, float]:
        """Probabilities keyed by route name."""
        return {n: p for n, p in zip(self.names, self.probabilities)}


def _softmax(utilities: Sequence[float], scale: float) -> List[float]:
    mu = _positive(scale, "scale")
    top = max(utilities)
    ex = [math.exp(mu * (u - top)) for u in utilities]
    tot = math.fsum(ex)
    return [e / tot for e in ex]


def logsum_accessibility(utilities: Sequence[float],
                         scale: float = 1.0) -> float:
    """Expected maximum utility (1/mu) ln sum exp(mu V) -- the logsum accessibility."""
    us = _as_floats(utilities, "utilities")
    if not us:
        raise PedestrianFlowError("the choice set is empty.")
    mu = _positive(scale, "scale")
    top = max(us)
    return top + math.log(math.fsum(math.exp(mu * (u - top)) for u in us)) / mu


def multinomial_logit(routes: Sequence[WalkRoute],
                      coef: Optional[RouteCoefficients] = None,
                      scale: float = 1.0,
                      walking_speed: float = 1.3) -> RouteChoiceResult:
    """Multinomial logit choice probabilities over a walking choice set."""
    if not routes:
        raise PedestrianFlowError("the choice set is empty.")
    us = [route_utility(r, coef, walking_speed) for r in routes]
    ps = _softmax(us, scale)
    names = [r.name for r in routes]
    best = names[max(range(len(ps)), key=lambda i: ps[i])]
    return RouteChoiceResult(names, us, ps,
                             logsum_accessibility(us, scale), float(scale),
                             None, best)


def path_size_logit(routes: Sequence[WalkRoute],
                    coef: Optional[RouteCoefficients] = None,
                    scale: float = 1.0, beta_ps: float = 1.0,
                    walking_speed: float = 1.3,
                    link_lengths: Optional[Dict[str, float]] = None
                    ) -> RouteChoiceResult:
    """Path-size logit: correct the MNL for overlapping walking routes."""
    if not routes:
        raise PedestrianFlowError("the choice set is empty.")
    lengths = [_positive(r.length, "length") for r in routes]
    shortest = min(lengths)
    ps: List[float] = []
    for k, r in enumerate(routes):
        if not r.links:
            ps.append(1.0)
            continue
        if link_lengths is None:
            la = {a: lengths[k] / len(r.links) for a in r.links}
        else:
            missing = [a for a in r.links if a not in link_lengths]
            if missing:
                raise PedestrianFlowError(
                    "link_lengths is missing an entry for %s (route %r)."
                    % (", ".join(repr(m) for m in sorted(missing)), r.name))
            la = {a: _positive(link_lengths[a], "link length") for a in r.links}
        terms = []
        for a in r.links:
            share = math.fsum(shortest / lengths[j]
                              for j, other in enumerate(routes)
                              if a in other.links)
            if share <= _EPS:
                continue
            terms.append((la[a] / lengths[k]) / share)
        ps.append(max(_EPS, math.fsum(terms)))
    us = [route_utility(r, coef, walking_speed) + beta_ps * math.log(ps[k])
          for k, r in enumerate(routes)]
    probs = _softmax(us, scale)
    names = [r.name for r in routes]
    best = names[max(range(len(probs)), key=lambda i: probs[i])]
    return RouteChoiceResult(names, us, probs,
                             logsum_accessibility(us, scale), float(scale),
                             ps, best)


# ===========================================================================
# 12. EVACUATION, EGRESS AND ACCUMULATION
# ===========================================================================
# SFPE boundary layers: a pedestrian stream does not use the full geometric
# width of a facility, so effective width = width - 2 * boundary layer.
_BOUNDARY_LAYERS: Dict[str, float] = {
    "stairway": 0.15,
    "handrail": 0.09,
    "theatre_seating": 0.0,
    "obstacle": 0.10,
    "corridor": 0.20,
    "wide_concourse": 0.46,
    "doorway": 0.15,
    "none": 0.0,
}


def effective_width(width: float, kind: str = "corridor") -> float:
    """SFPE effective width = clear width minus two boundary layers (m)."""
    w = _positive(width, "width")
    key = str(kind).strip().lower()
    if key not in _BOUNDARY_LAYERS:
        raise PedestrianFlowError(
            "unknown boundary-layer kind %r (known: %s)."
            % (kind, ", ".join(sorted(_BOUNDARY_LAYERS))))
    return max(0.0, w - 2.0 * _BOUNDARY_LAYERS[key])


def exit_capacity(width: float, specific_flow: float = 1.3,
                  kind: str = "doorway") -> float:
    """Exit capacity in persons per second = specific flow * effective width."""
    return _positive(specific_flow, "specific_flow") * effective_width(width, kind)


@dataclass
class EgressResult:
    """Flow-limited egress calculation for a single exit."""

    total_time: float
    travel_time: float
    flow_time: float
    pre_movement: float
    capacity: float
    effective_width: float
    controlling: str
    people: float


def egress_time(n_people: float, exit_width: float, travel_distance: float,
                walking_speed: float = 1.2, specific_flow: float = 1.3,
                pre_movement: float = 0.0, kind: str = "doorway") -> EgressResult:
    """Hand-calculation egress time: pre-movement + max(travel time, queue time)."""
    n = _non_negative(n_people, "n_people")
    cap = exit_capacity(exit_width, specific_flow, kind)
    weff = effective_width(exit_width, kind)
    travel = _non_negative(travel_distance, "travel_distance") / \
        _positive(walking_speed, "walking_speed")
    flow = (n / cap) if cap > _EPS else _INF
    pre = _non_negative(pre_movement, "pre_movement")
    controlling = "flow" if flow >= travel else "travel"
    return EgressResult(pre + max(travel, flow), travel, flow, pre, cap, weff,
                        controlling, n)


def _pm_speed_m_per_min(d_area: float) -> float:
    """Predtechenskii-Milinskii horizontal-path speed (m/min) at area fraction D."""
    return _interp(_PM_D, _PM_V_M_PER_MIN, _non_negative(d_area, "D"))


def _pm_specific_flow(d_area: float) -> float:
    """Predtechenskii-Milinskii specific flow q = D v (m/min) at area fraction D."""
    return d_area * _pm_speed_m_per_min(d_area)


def _pm_capacity() -> Tuple[float, float]:
    best_d, best_q = _PM_D[0], _pm_specific_flow(_PM_D[0])
    for k in range(len(_PM_D) - 1):
        d0, d1 = _PM_D[k], _PM_D[k + 1]
        v0, v1 = _PM_V_M_PER_MIN[k], _PM_V_M_PER_MIN[k + 1]
        slope = (v1 - v0) / (d1 - d0)
        cands = [d0, d1]
        if abs(slope) > _EPS:
            vertex = (slope * d0 - v0) / (2.0 * slope)
            if d0 < vertex < d1:
                cands.append(vertex)
        for c in cands:
            q = _pm_specific_flow(c)
            if q > best_q:
                best_d, best_q = c, q
    return best_d, best_q


_PM_D_CAP, _PM_Q_MAX = _pm_capacity()


def _pm_density_from_flow(q: float) -> float:
    """Invert q = D v(D) on the uncongested branch of the PM table."""
    qq = min(_non_negative(q, "q"), _PM_Q_MAX)
    lo, hi = 0.0, _PM_D_CAP
    if qq <= _EPS:
        return 0.0
    return _bisect_root(lambda d: _pm_specific_flow(d) - qq, lo, hi)


@dataclass
class PMEgressResult:
    """Predtechenskii-Milinskii staged egress calculation along a path of sections."""

    total_time: float
    section_times: List[float]
    densities: List[float]
    speeds: List[float]
    specific_flows: List[float]
    delays: List[float]
    merge_limited: List[bool]
    people: float
    area_per_person: float


def predtechenskii_milinskii_egress(
        sections: Sequence[Tuple[float, float]], n_people: float,
        area_per_person: float = 0.1) -> PMEgressResult:
    """Predtechenskii-Milinskii method: stage-by-stage speeds, flows and queueing delays."""
    if not sections:
        raise PedestrianFlowError("need at least one section (length, width).")
    n = _positive(n_people, "n_people")
    f = _positive(area_per_person, "area_per_person")
    area_people = n * f
    times: List[float] = []
    dens: List[float] = []
    speeds: List[float] = []
    flows: List[float] = []
    delays: List[float] = []
    limited: List[bool] = []
    prev_q = 0.0
    prev_b = 0.0
    for idx, sec in enumerate(sections):
        length = _positive(sec[0], "section length")
        wid = _positive(sec[1], "section width")
        delay = 0.0
        flag = False
        if idx == 0:
            d_area = area_people / (length * wid)
            if d_area >= _PM_D[-1]:
                # Clamping D to just below the jam point used to manufacture a
                # speed of ~7.5e-7 m/min and a "finite" clearance of years,
                # returned with no flag -- and it made the v <= _EPS guard
                # below unreachable, because the clamp always left v > 0.
                raise PedestrianFlowError(
                    "section 0 holds %.6g people at %.6g m^2 each in %.6g m^2 "
                    "(D = %.4f), at or above the Predtechenskii-Milinskii jam "
                    "fraction D = %.2f: no egress is possible."
                    % (n, f, length * wid, d_area, _PM_D[-1]))
        else:
            q_unc = prev_q * prev_b / wid
            if q_unc > _PM_Q_MAX:
                flag = True
                q_now = _PM_Q_MAX
                delay = area_people * (1.0 / (q_now * wid)
                                       - 1.0 / (prev_q * prev_b))
            else:
                q_now = q_unc
            d_area = _pm_density_from_flow(q_now)
        v = _pm_speed_m_per_min(d_area)
        # SCALE-RELATIVE: 1e-12 m/min is meaningless next to a table whose free
        # speed is 100 m/min.  A section crawling below a millionth of the free
        # speed is jammed for every practical purpose.
        if v <= 1e-6 * _PM_V_M_PER_MIN[0]:
            raise PedestrianFlowError(
                "section %d is jammed solid (D = %.4f, v = %.3g m/min): no "
                "egress is possible." % (idx, d_area, v))
        q = _pm_specific_flow(d_area)
        t_min = length / v + max(0.0, delay)
        times.append(t_min * 60.0)
        dens.append(d_area / f)
        speeds.append(v / 60.0)
        flows.append(q / f / 60.0)
        delays.append(max(0.0, delay) * 60.0)
        limited.append(flag)
        prev_q, prev_b = q, wid
    return PMEgressResult(math.fsum(times), times, dens, speeds, flows, delays,
                          limited, n, f)


def evacuation_curve(n_people: float, capacity: float,
                     pre_movement: float = 0.0, travel_time: float = 0.0,
                     dt: float = 1.0) -> List[Tuple[float, float]]:
    """Cumulative egress curve (t, people out) for a constant-capacity exit."""
    n = _non_negative(n_people, "n_people")
    cap = _positive(capacity, "capacity")
    start = _non_negative(pre_movement, "pre_movement") + \
        _non_negative(travel_time, "travel_time")
    step = _positive(dt, "dt")
    end = start + n / cap
    out: List[Tuple[float, float]] = []
    t = 0.0
    while t < end + step:
        done = 0.0 if t <= start else min(n, (t - start) * cap)
        out.append((t, done))
        t += step
    out.append((end, n))
    # the exact completion point can fall between (or before) two samples, so
    # sort: a cumulative curve must be non-decreasing on BOTH axes.
    out.sort(key=lambda p: p[0])
    return out


@dataclass
class AccumulationResult:
    """Deterministic queueing at a destination: cumulative curves and the delay."""

    times: List[float]
    queue: List[float]
    departures: List[float]
    cumulative_arrivals: List[float]
    cumulative_departures: List[float]
    max_queue: float
    total_delay: float
    mean_delay: float
    clearance_time: float
    total_people: float


def accumulation_profile(arrivals: Sequence[float], capacity: float,
                         dt: float = 1.0,
                         initial_queue: float = 0.0,
                         max_steps: int = 1000000) -> AccumulationResult:
    """Cumulative arrival/departure (input-output) diagram with a capacity-limited server.

    A vanishingly small ``capacity`` makes the queue take astronomically many
    steps to clear.  ``max_steps`` bounds the work: a run that would need more
    raises rather than materialising hundreds of millions of samples (which
    exhausts memory long before it returns an answer).
    """
    arr = _as_floats(arrivals, "arrivals")
    if any(v < 0.0 for v in arr):
        raise PedestrianFlowError("arrivals must be >= 0.")
    cap = _positive(capacity, "capacity")
    step = _positive(dt, "dt")
    queue = _non_negative(initial_queue, "initial_queue")
    per_step = cap * step
    times = [0.0]
    q_series = [queue]
    dep: List[float] = []
    cum_arr = [0.0]
    cum_dep = [0.0]
    total_people = math.fsum(arr) + queue
    i = 0
    limit = _positive_int(max_steps, "max_steps")
    if per_step > 0.0:
        needed = len(arr) + int(math.ceil(total_people / per_step)) + 2
    else:
        needed = limit + 1
    if needed > limit:
        raise PedestrianFlowError(
            "clearing %.6g people at %.6g people per step would need %d steps, "
            "above max_steps = %d: raise the capacity, the time step or "
            "max_steps." % (total_people, per_step, needed, limit))
    step_cap = needed
    while i < step_cap:
        a = arr[i] if i < len(arr) else 0.0
        available = queue + a
        served = min(available, per_step)
        queue = available - served
        dep.append(served)
        cum_arr.append(cum_arr[-1] + a)
        cum_dep.append(cum_dep[-1] + served)
        q_series.append(queue)
        times.append((i + 1) * step)
        i += 1
        if queue <= _EPS and i >= len(arr):
            break
    total_delay = step * math.fsum(
        0.5 * (q_series[k] + q_series[k + 1]) for k in range(len(q_series) - 1))
    clearance = times[-1]
    for k in range(len(q_series) - 1, -1, -1):
        if q_series[k] > _EPS:
            clearance = times[min(k + 1, len(times) - 1)]
            break
    else:
        clearance = 0.0
    return AccumulationResult(
        times=times, queue=q_series, departures=dep,
        cumulative_arrivals=cum_arr, cumulative_departures=cum_dep,
        max_queue=max(q_series), total_delay=total_delay,
        mean_delay=(total_delay / total_people if total_people > _EPS else 0.0),
        clearance_time=clearance, total_people=total_people)


def stadium_egress_profile(n_people: float, exit_widths: Sequence[float],
                           release_duration: float = 300.0, dt: float = 10.0,
                           shape: str = "triangular",
                           specific_flow: float = 1.3,
                           kind: str = "doorway") -> AccumulationResult:
    """Accumulation at a stadium's exits given a shaped release of the stand."""
    n = _positive(n_people, "n_people")
    widths = _as_floats(exit_widths, "exit_widths")
    if not widths:
        raise PedestrianFlowError("need at least one exit width.")
    cap = math.fsum(exit_capacity(w, specific_flow, kind) for w in widths)
    dur = _positive(release_duration, "release_duration")
    step = _positive(dt, "dt")
    n_int = max(1, int(math.ceil(dur / step)))
    shp = str(shape).strip().lower()
    weights: List[float] = []
    for i in range(n_int):
        u = (i + 0.5) / n_int
        if shp == "uniform":
            w = 1.0
        elif shp == "triangular":
            w = 1.0 - abs(2.0 * u - 1.0)
        elif shp == "exponential":
            w = math.exp(-3.0 * u)
        elif shp == "front_loaded":
            w = (1.0 - u) ** 2
        else:
            raise PedestrianFlowError(
                "shape must be 'uniform', 'triangular', 'exponential' or "
                "'front_loaded'.")
        weights.append(max(w, 0.0))
    tot = math.fsum(weights)
    if tot <= _EPS:
        raise PedestrianFlowError("the release profile has zero mass.")
    arrivals = [n * w / tot for w in weights]
    return accumulation_profile(arrivals, cap, step)


def safety_margin(aset: float, rset: float) -> Dict[str, Any]:
    """Life-safety margin ASET - RSET with a pass/fail verdict and a ratio."""
    a = _positive(aset, "aset")
    r = _positive(rset, "rset")
    return {
        "aset": a,
        "rset": r,
        "margin": a - r,
        "ratio": a / r,
        "safe": bool(a > r),
        "verdict": "adequate" if a >= 1.5 * r else
                   ("marginal" if a > r else "inadequate"),
    }


# ===========================================================================
# 13. CROSS-MODEL COMPARISON (PDE vs CA vs SOCIAL FORCE)
# ===========================================================================
# The same corridor, the same free speed and the same background density solved
# three ways.  At low density all three reduce to free flow and their traversal
# times must agree; ABOVE free flow they are genuinely different models and the
# helper's job is to say so honestly rather than to look agreeable:
#
#   * the Godunov PDE holds the uniform state exactly and traverses it in
#     L / v(rho) to machine precision at ANY density, free or congested;
#   * the floor-field CA is the independent solver -- its traversal time rises
#     with the background density because agents block one another, but the
#     parallel update and the finite lattice mean it does NOT reproduce v(rho);
#   * the social-force branch reports a time only when it can actually lay the
#     requested crowd out without overlap AND stay inside its agent budget.
#     Its known limit, measured rather than assumed: a HOMOGENEOUS crowd all
#     walking the same way at the same desired speed down an OPEN corridor
#     barely slows with the background density, because the Helbing repulsion
#     of a co-moving stream is fore/aft symmetric and largely cancels.
#
# A model that cannot answer is named in ``models_missing`` and forces
# ``max_relative_spread`` to infinity: a differential tier whose spread
# IMPROVES because the disagreeing model silently dropped out is worse than no
# tier at all.  The CA is run with a strong static-field coupling (k_s) so that
# its free-flow speed really is cell_size / time_step rather than a
# downhill-biased random walk.


@dataclass
class CrossModelResult:
    """Traversal time of one corridor computed by three independent models."""

    length: float
    width: float
    density: float
    free_flow_time: float
    pde_time: float
    ca_time: float
    social_force_time: float
    max_relative_spread: float
    detail: Dict[str, Any] = field(default_factory=dict)
    models_reported: List[str] = field(default_factory=list)
    models_missing: List[str] = field(default_factory=list)

    def times(self) -> Dict[str, float]:
        """The three traversal times keyed by model name."""
        return {"pde": self.pde_time, "ca": self.ca_time,
                "socialForce": self.social_force_time}

    @property
    def complete(self) -> bool:
        """True when every model produced a finite traversal time."""
        return not self.models_missing


def cross_model_egress(length: float = 20.0, width: float = 3.0,
                       density: float = 0.15,
                       fd: Optional[PedestrianFD] = None, *,
                       cell_size: float = 0.4, seed: int = 7,
                       k_s: float = 20.0, sf_dt: float = 0.02,
                       sf_max_agents: int = 40) -> CrossModelResult:
    """Traverse one corridor with the Godunov PDE, the floor-field CA and social force.

    At the background density ``rho`` the honest reference is ``L / v(rho)``,
    not ``L / v0``: every run is therefore given twice the CONGESTED traversal
    time, and each model reports whether it answered at all.  A model that
    cannot answer is named in ``models_missing`` and forces
    ``max_relative_spread`` to infinity -- a differential tier whose spread
    IMPROVES because the disagreeing model silently dropped out is worse than
    no tier.

    Known limit of the social-force branch, measured rather than assumed: a
    HOMOGENEOUS crowd all walking the same way at the same desired speed down
    an OPEN corridor does not slow down with the background density, because
    the Helbing repulsion of a co-moving stream is fore/aft symmetric and
    cancels.  It only reports a time when the requested crowd can actually be
    placed without overlap; above the close packing limit of the agent radius
    (~3.2 P/m^2 at r = 0.3 m) it declares itself missing instead of answering
    with a crowd it silently truncated.
    """
    L = _positive(length, "length")
    W = _positive(width, "width")
    rho = _non_negative(density, "density")
    diagram = fd if fd is not None else fd_weidmann()
    v0 = diagram.v_free
    cs = _positive(cell_size, "cell_size")
    free_time = L / v0
    v_rho = diagram.speed(rho)
    if v_rho <= _EPS:
        raise PedestrianFlowError(
            "density %.6g P/m^2 is at or above the jam density %.6g P/m^2 of "
            "the %s diagram: nobody traverses the corridor."
            % (rho, diagram.rho_jam, diagram.name))
    congested_time = L / v_rho
    rng = random.Random(seed)

    # --- 1. continuum: a tracer advected by the Godunov density field --------
    n_cells = max(8, int(round(L / cs)))
    link = WalkwayLink(L, W, diagram, n_cells, "corridor")
    link.set_uniform(rho)
    # Feed AND drain the link at the uniform state's own flow.  Draining at
    # the full capacity instead (the old setting) is larger than q(rho) on the
    # congested branch, so above rho_c the corridor emptied from the far end
    # and the tracer measured a rarefaction rather than the requested density.
    sim = simulate_walkway(link, 2.0 * congested_time,
                           inflow=diagram.flow(rho) * W,
                           outflow_capacity=diagram.flow(rho) * W,
                           tracer_start=0.0)
    pde_time = sim.tracer_time if sim.tracer_time is not None else _INF

    # --- 2. floor-field cellular automaton -----------------------------------
    ncols = int(round(L / cs)) + 1
    nrows = max(2, int(round(W / cs)))
    walkable = [[True] * ncols for _ in range(nrows)]
    exits = [(i, ncols - 1) for i in range(nrows)]
    tagged = (nrows // 2, 0)
    cells = [tagged]
    n_agents = int(round(rho * L * W))
    taken = {tagged}
    guard = 0
    while len(cells) < max(1, n_agents) and guard < 10000:
        guard += 1
        c = (rng.randrange(nrows), rng.randrange(2, max(3, ncols - 1)))
        if c not in taken:
            taken.add(c)
            cells.append(c)
    ca_dt = cs / v0
    ca = FloorFieldCA(walkable, exits, cells, k_s=k_s, k_d=0.0, alpha=0.0,
                      delta=1.0, friction=0.0, cell_size=cs,
                      time_step=ca_dt, seed=seed)
    # the CA also needs enough steps to answer at the CONGESTED time, not the
    # free-flow one, or it drops out exactly where the models start to differ
    ca_res = ca.run(max_steps=max(6 * ncols,
                                  int(math.ceil(3.0 * congested_time / ca_dt))))
    ca_time = ca_res.egress_times.get(0, _INF)

    # --- 3. social force ------------------------------------------------------
    params = SocialForceParams(desired_speed=v0)
    peds = [Pedestrian(x=0.0, y=0.5 * W, vx=v0, vy=0.0, target_x=L + 5.0,
                       target_y=0.5 * W, desired_speed=v0, ident=0)]
    # Spread the background crowd over the WHOLE corridor on a jittered
    # lattice, instead of rejection-sampling into a fixed 1.2 m window that
    # silently capped every run above rho ~ 0.4 P/m^2 at 19 agents while
    # ``detail['agents']`` went on reporting the density's full count.
    r_body = params.radius
    budget = _positive_int(sf_max_agents, "sf_max_agents")
    want = max(0, n_agents - 1)
    x0, x1 = 2.0 * r_body, L - r_body
    y0, y1 = r_body, W - r_body
    pitch = _INF
    skipped = ""
    if want > 0:
        nrows_sf = max(1, int((y1 - y0) / (2.0 * r_body)))
        ncols_sf = max(1, int(math.ceil(want / nrows_sf)))
        sx = max(x1 - x0, _EPS) / ncols_sf
        sy = max(y1 - y0, _EPS) / nrows_sf
        pitch = min(sx, sy)
        jit = 0.25 * max(0.0, pitch - 2.0 * r_body)
        for k in range(want):
            rr_, cc_ = divmod(k, ncols_sf)
            px = x0 + (cc_ + 0.5) * sx + (rng.random() - 0.5) * jit
            py = y0 + (min(rr_, nrows_sf - 1) + 0.5) * sy \
                + (rng.random() - 0.5) * jit
            peds.append(Pedestrian(x=px, y=py, vx=v0, vy=0.0,
                                   target_x=L + 5.0, target_y=py,
                                   desired_speed=v0, ident=k + 1))
    if pitch < 2.0 * r_body:
        # above the close packing limit of the body radius the crowd simply
        # cannot be laid out, so there is no honest time to quote
        skipped = "packing"
    elif n_agents > budget:
        # an O(n^2) integrator is not free: decline LOUDLY rather than quietly
        skipped = "budget"
    walls = [Wall(-1.0, 0.0, L + 6.0, 0.0), Wall(-1.0, W, L + 6.0, W)]
    step = _positive(sf_dt, "sf_dt")
    if skipped:
        sf_res = simulate_social_force(peds[:1], walls, params, dt=step,
                                       steps=1, exit_test=lambda p: False)
        sf_time = _INF
    else:
        sf_res = simulate_social_force(
            peds, walls, params, dt=step,
            steps=int(math.ceil(2.5 * congested_time / step)),
            exit_test=lambda p: p.x >= L)
        sf_time = sf_res.exit_times.get(0, _INF)

    names = ("pde", "ca", "socialForce")
    vals = [pde_time, ca_time, sf_time]
    reported = [names[k] for k in range(3) if vals[k] < _INF]
    missing = [names[k] for k in range(3) if vals[k] >= _INF]
    finite = [v for v in vals if v < _INF]
    if missing or not finite:
        # never let the spread LOOK better because a model dropped out
        spread = _INF
    else:
        mu = _mean(finite)
        spread = (max(finite) - min(finite)) / mu if mu > _EPS else 0.0
    return CrossModelResult(
        length=L, width=W, density=rho, free_flow_time=free_time,
        pde_time=pde_time, ca_time=ca_time, social_force_time=sf_time,
        max_relative_spread=spread,
        detail={"agents": n_agents, "caCells": ncols * nrows,
                "caTimeStep": ca_dt, "pdeCells": n_cells,
                "pdeConservationError": sim.conservation_error,
                "congestedReference": congested_time,
                "socialForceRequested": n_agents,
                "socialForceAgents": len(peds),
                "socialForcePitch": (None if pitch == _INF else pitch),
                "socialForceSkipped": skipped or None,
                "socialForceBudget": budget,
                "caEvacuated": ca_res.evacuated,
                "caMeanEgress": ca_res.mean_egress},
        models_reported=reported, models_missing=missing)


# ===========================================================================
# 14. FRONTEND PAYLOADS
# ===========================================================================
def _json_safe(x: Any) -> Any:
    """Replace non-finite floats with None so the dict survives json.dumps."""
    if isinstance(x, float):
        if math.isinf(x) or math.isnan(x):
            return None
        return round(x, 8)
    return x


def build_pedestrian_flow_payload(
        fd: PedestrianFD, *,
        simulation: Optional[WalkwaySimulation] = None,
        los: Optional[LOSResult] = None,
        crossing: Optional[CrossingDelayResult] = None,
        egress: Optional[EgressResult] = None,
        routes: Optional[RouteChoiceResult] = None,
        accumulation: Optional[AccumulationResult] = None,
        curve_points: int = 60) -> Dict[str, Any]:
    """Frontend-ready JSON dict: diagram curve, density profile, LOS, delay, egress."""
    pts = _positive_int(curve_points, "curve_points")
    curve = []
    for (r, q) in fd_flow_curve(fd, pts):
        curve.append({"density": _json_safe(r), "flow": _json_safe(q),
                      "speed": _json_safe(fd.speed(r))})
    payload: Dict[str, Any] = {
        "fundamentalDiagram": dict(fd.to_dict(), curve=curve),
        "crowdBands": [{"name": n, "lower": lo,
                        "upper": (None if hi == _INF else hi), "risk": rk,
                        "description": desc}
                       for (n, lo, hi, rk, desc) in _CROWD_BANDS],
    }
    if simulation is not None:
        dx = simulation.dx
        payload["densityProfile"] = [
            {"x": _json_safe(dx * (i + 0.5)),
             "density": _json_safe(simulation.density[i]),
             "speed": _json_safe(fd.speed(simulation.density[i]))}
            for i in range(len(simulation.density))]
        payload["ledger"] = {
            "storedInitial": _json_safe(simulation.stored_initial),
            "inflow": _json_safe(simulation.inflow_total),
            "outflow": _json_safe(simulation.outflow_total),
            "source": _json_safe(simulation.source_total),
            "storedFinal": _json_safe(simulation.stored_final),
            "conservationError": _json_safe(simulation.conservation_error),
            "steps": simulation.steps,
            "dt": _json_safe(simulation.dt),
        }
        payload["outflowSeries"] = [
            {"t": _json_safe(t), "flow": _json_safe(q)}
            for (t, q) in simulation.outflow_series[::max(
                1, len(simulation.outflow_series) // 200)]]
    if los is not None:
        payload["levelOfService"] = {
            "grade": los.grade, "facility": los.facility, "basis": los.basis,
            "value": _json_safe(los.value), "lower": _json_safe(los.lower),
            "upper": (None if los.upper == _INF else _json_safe(los.upper)),
            "description": los.description,
            "density": _json_safe(los.density) if los.density is not None else None,
        }
    if crossing is not None:
        payload["crossing"] = {
            "control": crossing.control,
            "delay": _json_safe(crossing.delay),
            "crossingTime": _json_safe(crossing.crossing_time),
            "totalTime": _json_safe(crossing.total_time),
            "criticalGap": _json_safe(crossing.critical_gap)
            if crossing.critical_gap is not None else None,
            "stages": crossing.stages,
            "levelOfService": crossing.level_of_service,
            "detail": {k: _json_safe(v) for k, v in crossing.detail.items()},
        }
    if egress is not None:
        payload["egress"] = {
            "totalTime": _json_safe(egress.total_time),
            "travelTime": _json_safe(egress.travel_time),
            "flowTime": _json_safe(egress.flow_time),
            "preMovement": _json_safe(egress.pre_movement),
            "capacity": _json_safe(egress.capacity),
            "effectiveWidth": _json_safe(egress.effective_width),
            "controlling": egress.controlling,
            "people": _json_safe(egress.people),
        }
    if routes is not None:
        payload["routeChoice"] = {
            "logsum": _json_safe(routes.logsum),
            "scale": _json_safe(routes.scale),
            "best": routes.best,
            "routes": [
                {"name": routes.names[i],
                 "utility": _json_safe(routes.utilities[i]),
                 "probability": _json_safe(routes.probabilities[i]),
                 "pathSize": (_json_safe(routes.path_size[i])
                              if routes.path_size else None)}
                for i in range(len(routes.names))],
        }
    if accumulation is not None:
        stride = max(1, len(accumulation.times) // 200)
        payload["accumulation"] = {
            "maxQueue": _json_safe(accumulation.max_queue),
            "totalDelay": _json_safe(accumulation.total_delay),
            "meanDelay": _json_safe(accumulation.mean_delay),
            "clearanceTime": _json_safe(accumulation.clearance_time),
            "totalPeople": _json_safe(accumulation.total_people),
            "series": [{"t": _json_safe(accumulation.times[i]),
                        "queue": _json_safe(accumulation.queue[i]),
                        "arrivals": _json_safe(accumulation.cumulative_arrivals[i]),
                        "departures": _json_safe(accumulation.cumulative_departures[i])}
                       for i in range(0, len(accumulation.times), stride)],
        }
    return payload


def build_a2ui_payload(fd: PedestrianFD, *,
                       simulation: Optional[WalkwaySimulation] = None,
                       los: Optional[LOSResult] = None,
                       crossing: Optional[CrossingDelayResult] = None,
                       egress: Optional[EgressResult] = None,
                       accumulation: Optional[AccumulationResult] = None,
                       cross_model: Optional[CrossModelResult] = None
                       ) -> Dict[str, Any]:
    """Visualisation-ready dict: stat cards, chart series and colour-coded bands."""
    rc, qmax = fd._capacity_pair()
    cards: List[Dict[str, Any]] = [
        {"label": "Free speed", "value": round(fd.v_free, 2), "unit": "m/s"},
        {"label": "Capacity", "value": round(qmax, 3), "unit": "P/m/s"},
        {"label": "Critical density", "value": round(rc, 3), "unit": "P/m^2"},
        {"label": "Jam density", "value": round(fd.rho_jam, 2), "unit": "P/m^2"},
    ]
    if los is not None:
        cards.append({"label": "Level of service", "value": los.grade,
                      "unit": los.facility})
    if crossing is not None:
        cards.append({"label": "Crossing delay",
                      "value": round(crossing.delay, 1), "unit": "s"})
    if egress is not None:
        cards.append({"label": "Egress time",
                      "value": round(egress.total_time, 1), "unit": "s"})
    if cross_model is not None:
        cards.append({"label": "Cross-model spread",
                      "value": _json_safe(100.0 * cross_model.max_relative_spread),
                      "unit": "%"})
    charts: Dict[str, Any] = {
        "fundamentalDiagram": [
            {"density": _json_safe(r), "flow": _json_safe(q)}
            for (r, q) in fd_flow_curve(fd, 50)],
        "speedDensity": [
            {"density": _json_safe(r), "speed": _json_safe(v)}
            for (r, v) in fd_speed_curve(fd, 50)],
    }
    if simulation is not None:
        dx = simulation.dx
        charts["densityProfile"] = [
            {"x": _json_safe(dx * (i + 0.5)),
             "density": _json_safe(simulation.density[i]),
             "band": crowd_density_band(simulation.density[i]).name}
            for i in range(len(simulation.density))]
    if accumulation is not None:
        stride = max(1, len(accumulation.times) // 120)
        charts["accumulation"] = [
            {"t": _json_safe(accumulation.times[i]),
             "queue": _json_safe(accumulation.queue[i])}
            for i in range(0, len(accumulation.times), stride)]
    if cross_model is not None:
        charts["crossModel"] = [{"model": k, "time": _json_safe(v)}
                                for k, v in cross_model.times().items()]
    return {
        "title": "Pedestrian flow -- %s" % fd.name,
        "statCards": cards,
        "charts": charts,
        "losBands": los_bands("walkway", "area"),
        "crowdBands": [{"name": n, "risk": rk, "lower": lo,
                        "upper": (None if hi == _INF else hi)}
                       for (n, lo, hi, rk, _d) in _CROWD_BANDS],
    }


# ===========================================================================
# 15. DEMO
# ===========================================================================
def _demo() -> None:
    print("=" * 70)
    print("  PEDESTRIAN FLOW (pure Python) - demo")
    print("=" * 70)

    # -- 1. fundamental diagrams ------------------------------------------
    print("\n-- 1. fundamental diagrams: analytic capacity vs numerical argmax --")
    print("   %-26s %6s %7s %8s %8s %9s" %
          ("model", "v0", "rho_c", "q_max", "v(q_max)", "argmax err"))
    for name in FD_NAMES:
        fd = get_fd(name)
        rc, qmax = fd.critical_density(), fd.capacity()
        grid = fd_flow_curve(fd, 200000)
        best = max(grid, key=lambda p: p[1])
        err = abs(best[1] - qmax)
        print("   %-26s %6.2f %7.3f %8.4f %8.3f %9.2e"
              % (fd.name, fd.v_free, rc, qmax, fd.capacity_speed(), err))
    wd = fd_weidmann()
    print("   Weidmann capacity %.4f P/m/s at %.3f P/m^2 (documented 1.2-1.3)"
          % (wd.capacity(), wd.critical_density()))
    print("   monotone check, max dv/drho on the declining branch:")
    for name in FD_NAMES:
        fd = get_fd(name)
        print("      %-26s %+.4e  (must be < 0, from rho=%.3f)"
              % (name, max_speed_slope(fd, 400), fd.rho_decline_start))
    print("   speed curve sample (Weidmann): %s"
          % ["%.2f" % v for (_r, v) in fd_speed_curve(wd, 5)])
    print("   invert q: density at 0.8 P/m/s free=%.4f congested=%.4f P/m^2"
          % (wd.density_from_flow(0.8, "free"),
             wd.density_from_flow(0.8, "congested")))

    # -- 2. speed-density modulation --------------------------------------
    print("\n-- 2. v = v0 * phi(rho) --")
    print("   phi_linear(2.0, 5.4)=%.4f  phi_exponential(2.0, 5.4)=%.4f  "
          "phi_power(2.0, 5.4)=%.4f"
          % (phi_linear(2.0, 5.4), phi_exponential(2.0, 5.4),
             phi_power(2.0, 5.4)))
    print("   speed_from_phi(1.34, phi_exponential(2.0, 5.4)) = %.4f m/s"
          % speed_from_phi(1.34, phi_exponential(2.0, 5.4)))
    print("   Tobler: flat=%.3f  -5%%=%.3f (max)  +10%%=%.3f  -20%%=%.3f m/s"
          % (tobler_walking_speed(0.0), tobler_walking_speed(-0.05),
             tobler_walking_speed(0.10), tobler_walking_speed(-0.20)))

    # -- 3. Godunov conservation ------------------------------------------
    print("\n-- 3. Godunov PDE: conservation, uniform state, free flow --")
    link = WalkwayLink(40.0, 3.0, wd, 200, "high-street")
    print("   dx=%.2f m  CFL dt=%.4f s  max wave speed=%.3f m/s  capacity=%.2f P/s"
          % (link.dx, cfl_timestep(wd, link.dx), max_wave_speed(wd),
             link.capacity_pps))
    print("   demand(0.5)=%.4f supply(0.5)=%.4f | demand(4.0)=%.4f supply(4.0)=%.4f"
          % (demand_flow(wd, 0.5), supply_flow(wd, 0.5),
             demand_flow(wd, 4.0), supply_flow(wd, 4.0)))
    print("   godunov_flux(0.5 -> 4.0) = %.4f P/m/s (blocked by supply)"
          % godunov_flux(wd, 0.5, 4.0))
    surge = [0.004] * 200
    sim = simulate_walkway(link, 120.0, inflow=2.4, outflow_capacity=1.5,
                           source=surge, record_every=200)
    print("   surge run: in=%.4f out=%.4f source=%.4f stored=%.4f -> %.4f people"
          % (sim.inflow_total, sim.outflow_total, sim.source_total,
             sim.stored_initial, sim.stored_final))
    print("   CONSERVATION error = %.3e people (balance %.6f)"
          % (sim.conservation_error, sim.balance))
    print("   max density %.3f P/m^2 (band: %s)"
          % (sim.max_density, crowd_density_band(sim.max_density).name))

    uni = WalkwayLink(40.0, 3.0, wd, 100, "uniform")
    uni.set_uniform(0.8)
    usim = simulate_walkway(uni, 60.0, inflow=wd.flow(0.8) * 3.0,
                            outflow_capacity=3.0 * wd.capacity())
    print("   uniform state drift = %.3e P/m^2 (must be exactly 0)"
          % max(abs(r - 0.8) for r in usim.density))
    one, _faces = godunov_step([0.8] * 5, wd, 0.4, 0.2, wd.flow(0.8), wd.flow(0.8))
    print("   single godunov_step on a uniform field: max |drho| = %.3e"
          % max(abs(v - 0.8) for v in one))

    empty = WalkwayLink(30.0, 3.0, wd, 150, "empty")
    fsim = simulate_walkway(empty, 40.0, tracer_start=0.0)
    print("   FREE FLOW tracer = %.9f s, L/v0 = %.9f s, err = %.2e s"
          % (fsim.tracer_time, 30.0 / wd.v_free,
             abs(fsim.tracer_time - 30.0 / wd.v_free)))
    print("   walkway_travel_time(30 m, rho=0) = %.9f s | at rho=2.0: %.3f s"
          % (walkway_travel_time(30.0, wd, 0.0),
             walkway_travel_time(30.0, wd, 2.0)))

    print("\n-- 3b. Riemann problem: Rankine-Hugoniot shock tracking --")
    rl, rr = 1.0, 4.5
    rs = riemann_solution(wd, rl, rr)
    s_exact = riemann_shock_speed(wd, rl, rr)
    shock = WalkwayLink(40.0, 3.0, wd, 400, "shock")
    shock.set_profile(lambda x: rl if x < 20.0 else rr)
    ssim = simulate_walkway(shock, 20.0, inflow=wd.flow(rl) * 3.0,
                            outflow_capacity=wd.flow(rr) * 3.0)
    x_front = front_position(ssim.density, ssim.dx, 0.5 * (rl + rr))
    print("   %s: analytic speed %.5f m/s, predicted front %.3f m"
          % (rs.kind, s_exact, 20.0 + s_exact * ssim.time))
    print("   solver front at %.3f m after %.2f s -> speed %.5f m/s (err %.2e)"
          % (x_front, ssim.time, (x_front - 20.0) / ssim.time,
             abs((x_front - 20.0) / ssim.time - s_exact)))
    raref = riemann_solution(wd, 4.0, 0.5)
    print("   rarefaction check: kind=%s head=%.3f tail=%.3f m/s interface q=%.4f"
          % (raref.kind, raref.head_speed, raref.tail_speed,
             raref.interface_flow))

    # -- 4. node coupling --------------------------------------------------
    print("\n-- 4. junction coupling: merges, diverges, whole-network ledger --")
    got = merge_flows([2.0, 3.0, 1.0], 4.0, [1.0, 1.0, 2.0])
    print("   merge 2.0/3.0/1.0 into 4.0 P/s -> %s (sum %.6f)"
          % (["%.3f" % v for v in got], math.fsum(got)))
    total, per = diverge_flows(5.0, [0.6, 0.4], [10.0, 1.2])
    print("   diverge 5.0 P/s 60/40 with supplies 10.0/1.2 -> total %.3f, %s (FIFO)"
          % (total, ["%.3f" % v for v in per]))
    nres = node_transfer([4.0, 2.0], [3.0, 2.0],
                         [[0.7, 0.3], [0.5, 0.5]], [1.0, 1.0])
    print("   node_transfer in=%s out=%s alpha=%s err=%.2e (%d iters)"
          % (["%.3f" % v for v in nres.inflows],
             ["%.3f" % v for v in nres.outflows],
             ["%.3f" % v for v in nres.reduction],
             nres.conservation_error, nres.iterations))

    a = WalkwayLink(20.0, 4.0, wd, 50, "approach-a")
    b = WalkwayLink(20.0, 3.0, wd, 50, "approach-b")
    c = WalkwayLink(30.0, 2.5, wd, 75, "shared-bridge")
    net = PedestrianNetwork(
        links=[a, b, c],
        junctions=[Junction("merge", [0, 1], [2], [[1.0], [1.0]], [1.0, 1.0])],
        entries={0: 3.0, 1: 2.0}, exits={2: 2.0})
    nsim = simulate_network(net, 180.0, record_every=200)
    print("   network: entered %.4f, left %.4f, stored %.4f -> %.4f people"
          % (nsim.entry_total, nsim.exit_total, nsim.stored_initial,
             nsim.stored_final))
    print("   NETWORK CONSERVATION error = %.3e people (dt=%.4f, %d steps)"
          % (nsim.conservation_error, nsim.dt, nsim.steps))
    print("   bridge density profile head/tail = %.3f / %.3f P/m^2"
          % (nsim.densities["shared-bridge"][0],
             nsim.densities["shared-bridge"][-1]))

    # -- 5. bidirectional flow --------------------------------------------
    print("\n-- 5. bidirectional flow and lane formation --")
    vp, vm = bidirectional_speeds(0.8, 0.8, wd, friction=0.25, lane_index=0.0)
    vp2, vm2 = bidirectional_speeds(0.8, 0.8, wd, friction=0.25, lane_index=1.0)
    print("   50/50 at rho=1.6: mixed v=%.3f m/s, fully laned v=%.3f m/s (+%.1f%%)"
          % (vp, vp2, 100.0 * (vp2 / vp - 1.0)))
    print("   counterflow_penalty(0.5, mu=0.25, lanes=0) = %.4f"
          % counterflow_penalty(0.5, 0.25, 0.0))
    r_one, q_one = bidirectional_capacity(wd, 1.0, 0.25, 0.0)
    r_two, q_two = bidirectional_capacity(wd, 0.5, 0.25, 0.0)
    print("   one-way capacity %.4f P/m/s at %.3f; two-way %.4f at %.3f (-%.1f%%)"
          % (q_one, r_one, q_two, r_two, 100.0 * (1.0 - q_two / q_one)))
    lane = simulate_lane_formation(8, 40, 40, 150, beta=2.0, seed=3)
    print("   lane order parameter %.4f -> %.4f over %d steps"
          % (lane.initial_index, lane.final_index, lane.steps))
    print("   perfect segregation index = %.3f, perfect mixing = %.3f"
          % (lane_formation_index([10, 0, 10, 0], [0, 10, 0, 10]),
             lane_formation_index([5, 5, 5, 5], [5, 5, 5, 5])))

    # -- 6. eikonal + Hughes ----------------------------------------------
    print("\n-- 6. fast marching and Hughes' continuum model --")
    ny, nx, dxg = 16, 24, 0.5
    walk = [[True] * nx for _ in range(ny)]
    for i in range(3, ny - 3):
        walk[i][12] = False                      # a wall with a gap at each end
    exits = [(i, nx - 1) for i in range(ny)]
    tt = fast_marching([[1.34 if walk[i][j] else 0.0 for j in range(nx)]
                        for i in range(ny)], exits, walk, dxg)
    straight = tt[0][0]
    print("   FMM travel time from the far corner = %.3f s (free line-of-sight "
          "%.3f s)" % (straight, (nx - 1) * dxg / 1.34))
    ux, uy = eikonal_descent_direction(tt, 8, 2, dxg)
    print("   descent direction at (8, 2) = (%.3f, %.3f)" % (ux, uy))
    path = potential_descent_path(tt, (8, 0))
    print("   steepest-descent path length = %d cells, ends at %s"
          % (len(path), str(path[-1])))

    hg = HughesGrid(nx=nx, ny=ny, dx=dxg, fd=wd, walkable=walk, exits=exits,
                    exit_capacity=1.5, discomfort=0.5)
    hg.fill(1.2, region=lambda i, j: j < 10)
    hres = simulate_hughes(hg, 60.0, cfl=0.8, potential_every=5, record_every=100)
    print("   Hughes: start %.3f people, evacuated %.3f, left %.3f"
          % (hres.stored_initial, hres.evacuated, hres.stored_final))
    print("   HUGHES CONSERVATION error = %.3e people (dt=%.4f, %d steps)"
          % (hres.conservation_error, hres.dt, hres.steps))
    print("   99%% clearance at %s s, peak density %.3f P/m^2"
          % ("%.1f" % hres.clearance_time if hres.clearance_time else "n/a",
             hres.max_density))

    # -- 7. social force ---------------------------------------------------
    print("\n-- 7. social force (Helbing-Molnar) --")
    sp = SocialForceParams()
    d_eq = social_force_equilibrium_separation(sp)
    two = [Pedestrian(x=0.0, y=0.0, target_x=50.0, target_y=0.0,
                      desired_speed=sp.desired_speed, ident=0),
           Pedestrian(x=3.0, y=0.0, target_x=-50.0, target_y=0.0,
                      desired_speed=sp.desired_speed, ident=1)]
    res2 = simulate_social_force(two, [], sp, dt=0.005, steps=4000,
                                 method="verlet")
    print("   two head-on agents: analytic equilibrium %.5f m, simulated %.5f m "
          "(err %.2e)" % (d_eq, res2.final_separation,
                          abs(res2.final_separation - d_eq)))
    f0 = social_forces(res2.pedestrians, [], sp)
    print("   residual force at equilibrium = %.4f N (driving m v0 / tau = %.1f N)"
          % (math.hypot(*f0[0]), sp.mass * sp.desired_speed / sp.tau))

    room_walls = [Wall(0.0, 0.0, 8.0, 0.0), Wall(0.0, 6.0, 8.0, 6.0),
                  Wall(0.0, 0.0, 0.0, 6.0),
                  Wall(8.0, 0.0, 8.0, 2.5), Wall(8.0, 3.5, 8.0, 6.0)]
    rng = random.Random(11)
    crowd = [Pedestrian(x=1.0 + 4.0 * rng.random(), y=1.0 + 4.0 * rng.random(),
                        target_x=9.5, target_y=3.0, desired_speed=1.34,
                        ident=k) for k in range(10)]
    rres = simulate_social_force(crowd, room_walls, sp, dt=0.005, steps=5000,
                                 method="euler",
                                 exit_test=lambda p: p.x >= 8.3,
                                 record_every=50)
    print("   room egress: %d/%d out, mean %.2f s, last %.2f s, min separation "
          "%.3f m" % (rres.evacuated, len(crowd),
                      rres.mean_exit_time or float("nan"),
                      rres.last_exit_time or float("nan"),
                      rres.min_separation))
    solo = [Pedestrian(x=0.0, y=0.0, vx=1.34, target_x=20.0, target_y=0.0,
                       desired_speed=1.34, ident=0)]
    acc = social_force_step(solo, [], sp, 0.01, "euler")
    print("   one step at the desired velocity: |a| = %.2e m/s^2, x = %.4f m"
          % (math.hypot(*acc[0]), solo[0].x))

    # -- 8. floor-field CA -------------------------------------------------
    print("\n-- 8. floor-field cellular automaton --")
    ca_rows, ca_cols = 10, 25
    ca_walk = [[True] * ca_cols for _ in range(ca_rows)]
    ca_exits = [(4, ca_cols - 1), (5, ca_cols - 1)]
    sfield = static_floor_field(ca_walk, ca_exits, 0.4)
    print("   static field: corner distance %.2f m, exit distance %.2f m"
          % (sfield[0][0], sfield[4][ca_cols - 1]))
    rngc = random.Random(5)
    spots = set()
    while len(spots) < 60:
        spots.add((rngc.randrange(ca_rows), rngc.randrange(0, 12)))
    ca = FloorFieldCA(ca_walk, ca_exits, sorted(spots), k_s=3.0, k_d=1.0,
                      alpha=0.3, delta=0.3, friction=0.1, cell_size=0.4,
                      time_step=0.3, seed=17)
    cres = ca.run(max_steps=600)
    print("   %d agents through a 2-cell exit: mean %.2f s, median %.2f s, "
          "last %.2f s" % (cres.evacuated, cres.mean_egress,
                           cres.median_egress, cres.last_egress))
    peak = max(f for (_t, f) in cres.flow_series)
    print("   CA free speed %.3f m/s, peak exit flow %.2f P/s, %d steps"
          % (ca.free_speed, peak, cres.steps))

    # -- 9. level of service and crowd risk --------------------------------
    print("\n-- 9. Fruin level of service and crowd risk --")
    for rho in (0.2, 0.7, 1.5, 2.5, 4.0):
        w_los = fruin_los_from_density(rho, "walkway")
        band = crowd_density_band(rho)
        print("   rho=%.1f P/m^2 -> %.2f m^2/P, walkway LOS %s (%s), band %s "
              "(risk %d)" % (rho, 1.0 / rho, w_los.grade, w_los.description,
                             band.name, band.risk))
    print("   queueing at 0.5 m^2/P -> LOS %s | stairway at 1.2 m^2/P -> LOS %s"
          % (fruin_los(0.5, "queueing").grade,
             fruin_los(1.2, "stairway").grade))
    print("   walkway unit flow 55 P/min/m -> LOS %s | stairway 50 -> LOS %s"
          % (fruin_los(55.0, "walkway", "flow").grade,
             fruin_los(50.0, "stairway", "flow").grade))
    print("   walkway area bands: %s"
          % [(b["grade"], b["lower"]) for b in los_bands("walkway", "area")])
    vels = [(0.2, 0.1), (-0.3, 0.4), (0.1, -0.5), (0.4, 0.2), (-0.1, -0.2)]
    press = crowd_pressure(4.5, vels)
    risk = crush_risk(4.5, press)
    print("   crowd pressure at rho=4.5 with turbulent velocities = %.4f 1/s^2"
          % press)
    print("   crush risk: level=%s score=%d (%s)"
          % (risk["level"], risk["score"], risk["notes"][-1]))

    # -- 10. crossings and signal timing -----------------------------------
    print("\n-- 10. crossing delay and signal timing --")
    tc = critical_gap(7.3, 1.2, 1.5)
    print("   7.3 m crossing at 1.2 m/s -> critical gap %.3f s, P(accept "
          "@600 vph) = %.4f" % (tc, gap_acceptance_probability(600.0, tc)))
    print("   Adams' delay:  %s"
          % "  ".join("%d vph=%.2fs" % (q, adams_delay(q, tc))
                      for q in (0, 200, 600, 1000, 1400)))
    print("   signalised delay (C=90): g=0 -> %.2f s (=C/2), g=30 -> %.2f s, "
          "g=90 -> %.2f s" % (signalised_pedestrian_delay(90.0, 0.0),
                              signalised_pedestrian_delay(90.0, 30.0),
                              signalised_pedestrian_delay(90.0, 90.0)))
    one_stage = crossing_delay(7.3, control="unsignalised", vehicle_flow=900.0)
    two_stage = crossing_delay(7.3, control="unsignalised", vehicle_flow=450.0,
                               vehicle_flow_2=450.0, refuge=True,
                               refuge_dwell=2.0)
    sig = crossing_delay(7.3, control="signalised", cycle=90.0, walk_green=25.0)
    print("   one-stage @900 vph: %.2f s (LOS %s) | two-stage via refuge: "
          "%.2f s (LOS %s)" % (one_stage.delay, one_stage.level_of_service,
                               two_stage.delay, two_stage.level_of_service))
    print("   signalised C=90 g=25: delay %.2f s (LOS %s), total crossing "
          "%.2f s" % (sig.delay, sig.level_of_service, sig.total_time))
    print("   two_stage_crossing_delay(450, 450, %.2f, %.2f) = %.3f s"
          % (tc / 2.0, tc / 2.0,
             two_stage_crossing_delay(450.0, 450.0, tc / 2.0, tc / 2.0)))
    print("   pedestrian delay LOS at 12 s = %s (%s)"
          % (pedestrian_delay_los(12.0).grade,
             pedestrian_delay_los(12.0).description))
    print("   minimum green for 7.2 m: standard 1.2 m/s -> %.4f s, "
          "mobility-impaired 1.0 m/s -> %.4f s"
          % (pedestrian_minimum_green(7.2, 1.2),
             pedestrian_minimum_green(7.2, mobility_impaired=True)))
    phase = pedestrian_phase_times(7.2, 1.2, walk_interval=7.0, cycle=90.0)
    print("   phase: walk %.1f s + clearance %.1f s = %.1f s, signal delay "
          "%.2f s" % (phase.walk_interval, phase.clearance_interval,
                      phase.total_pedestrian_time, phase.delay))
    plat = platoon_dispersion([0, 0, 40, 60, 50, 20, 0, 0], 30.0, dt=5.0)
    print("   platoon dispersion: F=%.4f lag=%d steps (%.1f s), peak %.1f -> "
          "%.1f (-%.1f%%)" % (plat.smoothing, plat.lag_steps, plat.lag_seconds,
                              plat.peak_in, plat.peak_out,
                              100.0 * plat.peak_reduction))
    print("   platoon mass in %.4f, out %.4f (retained %.6f)"
          % (plat.mass_in, plat.mass_out, plat.mass_out / plat.mass_in))

    # -- 11. route choice --------------------------------------------------
    print("\n-- 11. walking route choice (MNL and path-size logit) --")
    routes = [
        WalkRoute("main-road", 620.0, crossings=3, crossing_delay=45.0,
                  stress=0.8, comfort=0.4, gradient=0.0,
                  links=("a1", "a2", "a3")),
        WalkRoute("back-street", 720.0, crossings=1, crossing_delay=12.0,
                  stress=0.2, comfort=0.8, gradient=0.0,
                  links=("a1", "b2", "b3")),
        WalkRoute("park-path", 780.0, crossings=0, crossing_delay=0.0,
                  stress=0.05, comfort=0.95, gradient=0.03,
                  links=("c1", "c2", "c3")),
    ]
    coef = RouteCoefficients()
    mnl = multinomial_logit(routes, coef, scale=1.0)
    psl = path_size_logit(routes, coef, scale=1.0, beta_ps=1.0)
    for i, nm in enumerate(mnl.names):
        print("   %-12s V=%+.4f  MNL p=%.4f  PSL p=%.4f (PS=%.3f)  walk %.1f min"
              % (nm, mnl.utilities[i], mnl.probabilities[i],
                 psl.probabilities[i], psl.path_size[i],
                 routes[i].time() / 60.0))
    print("   best route: MNL=%s PSL=%s | logsum accessibility %.4f (direct "
          "%.4f)" % (mnl.best, psl.best, mnl.logsum,
                     logsum_accessibility(mnl.utilities, 1.0)))
    print("   MNL probabilities as a dict: %s"
          % {k: round(v, 4) for k, v in mnl.as_dict().items()})

    # -- 12. evacuation and accumulation -----------------------------------
    print("\n-- 12. evacuation, egress and accumulation --")
    print("   effective width: 2.0 m corridor -> %.2f m, 1.2 m doorway -> "
          "%.2f m, 1.6 m stair -> %.2f m"
          % (effective_width(2.0, "corridor"), effective_width(1.2, "doorway"),
             effective_width(1.6, "stairway")))
    print("   exit capacity of a 1.2 m doorway = %.3f P/s (%.1f P/min)"
          % (exit_capacity(1.2), 60.0 * exit_capacity(1.2)))
    eg = egress_time(450.0, 1.8, 35.0, 1.2, 1.3, pre_movement=60.0)
    print("   450 people, 1.8 m exit, 35 m travel: travel %.1f s, flow %.1f s, "
          "total %.1f s (%s-controlled)"
          % (eg.travel_time, eg.flow_time, eg.total_time, eg.controlling))
    pm = predtechenskii_milinskii_egress(
        [(20.0, 3.0), (15.0, 2.0), (10.0, 1.4)], 300.0, 0.1)
    print("   PM staged egress (300 people, 3 sections): total %.1f s"
          % pm.total_time)
    for i, t in enumerate(pm.section_times):
        print("      section %d: rho=%.2f P/m^2 v=%.3f m/s q=%.3f P/m/s "
              "t=%.1f s%s" % (i + 1, pm.densities[i], pm.speeds[i],
                              pm.specific_flows[i], t,
                              "  [merge-limited]" if pm.merge_limited[i] else ""))
    curve = evacuation_curve(450.0, exit_capacity(1.8), 60.0, 29.2, dt=20.0)
    print("   evacuation curve: %d points, half out at ~%.0f s, complete at "
          "%.1f s" % (len(curve),
                      next(t for (t, n) in curve if n >= 225.0), curve[-1][0]))
    stadium = stadium_egress_profile(9000.0, [3.0, 3.0, 2.4, 2.4],
                                     release_duration=420.0, dt=15.0,
                                     shape="triangular")
    print("   stadium: %.0f people, %d exits, max queue %.0f, clearance "
          "%.0f s, mean delay %.1f s"
          % (stadium.total_people, 4, stadium.max_queue,
             stadium.clearance_time, stadium.mean_delay))
    free_flowing = accumulation_profile([5.0] * 20, 1.0, dt=10.0)
    print("   under-capacity accumulation: max queue %.3f, total delay %.3f "
          "(both must be 0)" % (free_flowing.max_queue,
                                free_flowing.total_delay))
    marg = safety_margin(900.0, stadium.clearance_time)
    print("   ASET 900 s vs RSET %.0f s -> margin %.0f s, ratio %.2f, %s"
          % (marg["rset"], marg["margin"], marg["ratio"], marg["verdict"]))

    # -- 13. cross-model agreement -----------------------------------------
    print("\n-- 13. cross-model oracle: PDE vs CA vs social force --")
    cm = cross_model_egress(20.0, 3.0, 0.15, wd, seed=7)
    print("   corridor %.0f m at %.2f P/m^2 (free-flow reference %.4f s)"
          % (cm.length, cm.density, cm.free_flow_time))
    print("   PDE tracer %.4f s | CA %.4f s | social force %.4f s"
          % (cm.pde_time, cm.ca_time, cm.social_force_time))
    print("   maximum relative spread = %.3f %% (%d agents, %d CA cells)"
          % (100.0 * cm.max_relative_spread, cm.detail["agents"],
             cm.detail["caCells"]))

    # -- 14. payloads -------------------------------------------------------
    print("\n-- 14. frontend payloads --")
    los_now = fruin_los_from_density(sim.mean_density, "walkway")
    payload = build_pedestrian_flow_payload(
        wd, simulation=sim, los=los_now, crossing=sig, egress=eg,
        routes=psl, accumulation=stadium, curve_points=40)
    ui = build_a2ui_payload(wd, simulation=sim, los=los_now, crossing=sig,
                            egress=eg, accumulation=stadium, cross_model=cm)
    print("   payload keys: %s" % sorted(payload))
    print("   diagram curve points %d | density profile %d | a2ui cards %d"
          % (len(payload["fundamentalDiagram"]["curve"]),
             len(payload["densityProfile"]), len(ui["statCards"])))
    print("   a2ui cards: %s"
          % ["%s=%s" % (c["label"], c["value"]) for c in ui["statCards"][:5]])

    # -- 15. validation ------------------------------------------------------
    print("\n-- 15. input validation --")
    for bad in (lambda: fd_weidmann(v_free=-1.0),
                lambda: signalised_pedestrian_delay(60.0, 90.0),
                lambda: fruin_los(1.0, "escalator"),
                lambda: get_fd("teleporter")):
        try:
            bad()
            print("   ERROR: no exception raised")
        except PedestrianFlowError as exc:
            print("   PedestrianFlowError: %s" % exc)

    print("\n" + "=" * 70)
    print("  demo OK")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
