"""
Traffic Flow -- macroscopic kinematic-wave and microscopic car-following models
==============================================================================

**How flow evolves in time on a link** -- the dynamics that live underneath the
static equilibrium.  ``traffic_assignment`` answers *which route*; this module
answers *what happens on the road once the vehicles are there*: where the shock
forms, how far back the queue reaches, when the bottleneck discharges, how much
delay the signal imposes, and whether a platoon of drivers amplifies a brake tap
into a phantom jam.  It is deliberately DISTINCT from its two siblings:
``traffic_assignment`` is static, network-wide and about route choice;
``microsimulation`` is stochastic and agent-level.  This module is the
deterministic continuum + car-following layer between them.

Everything is pure standard library: the fundamental-diagram zoo, the Riemann
solver, the Godunov / Cell-Transmission / Link-Transmission solvers, Newell's
cumulative-curve calculus, the signal-delay formulae, the Macroscopic
Fundamental Diagram, the second-order (METANET / Payne-Whitham / Aw-Rascle-
Zhang) models, six car-following laws, MOBIL and the Nagel-Schreckenberg
cellular automaton.  No NumPy, no SciPy, no sibling-module imports -- drop
``traffic_flow_template.py`` into a hackathon sandbox and go.

Contents
--------
0.  **Validation helpers** -- ``TrafficFlowError`` and the value guards.
1.  **Fundamental diagrams** -- ``Greenshields``, ``Greenberg``, ``Underwood``,
    ``Drake`` (Northwestern), ``NewellDiagram``, ``TriangularDiagram``
    (Daganzo), ``TrapezoidalDiagram`` and ``DelCastillo``, all behind one
    ``FundamentalDiagram`` surface with analytic capacity, critical density,
    wave speed and the demand / supply (sending / receiving) decomposition.
2.  **Kinematic wave theory (LWR)** -- characteristic speeds, the
    Rankine-Hugoniot shock speed ``(q2-q1)/(k2-k1)``, rarefaction fans and the
    complete ``riemann_solve`` similarity solution.
3.  **Godunov scheme** -- the demand/supply flux, the CFL condition and a
    conservative finite-volume solver for the LWR equation.
4.  **Cell Transmission Model** -- Daganzo's CTM as a network: ordinary links,
    priority merges, FIFO diverges, spillback and the capacity-drop extension.
5.  **Link Transmission Model** -- Yperman's LTM on Newell-Luke cumulative
    curves (sending / receiving flows from lagged cumulatives).
6.  **Cumulative curves** -- ``N(x, t)``, Newell's three-detector problem,
    delay as the area between curves, queue as the vertical gap and travel time
    as the horizontal gap.
7.  **Queues** -- the vertical (point) queue and the horizontal (physical)
    queue with storage-limited spillback.
8.  **Signal delay** -- the deterministic uniform term, Webster's stochastic
    delay and Akcelik's time-dependent overflow delay, plus Webster's optimal
    cycle length.
9.  **Shockwave analysis** -- queue-length estimation at a signal from the
    stopping / starting / recovery waves.
10. **Store-and-forward** -- the linear link-queue model used downstream by
    model-predictive signal control.
11. **Macroscopic Fundamental Diagram** -- Edie's generalised definitions from
    trajectories, lane-km weighted network aggregation, polynomial fitting and
    the accumulation-based network dynamics.
12. **Second-order models** -- METANET, Payne-Whitham and Aw-Rascle-Zhang.
13. **Car-following** -- Newell, Gipps, IDM, Optimal Velocity, Full Velocity
    Difference and Krauss, each with an explicit equilibrium-spacing function,
    plus a platoon simulator.
14. **Lane changing** -- the MOBIL incentive / safety rule with politeness.
15. **Cellular automata** -- the seeded Nagel-Schreckenberg model and its
    emergent fundamental diagram.
16. **Linear stability** -- local (platoon) and string stability of a
    car-following law, and the measured amplification of a real platoon.
17. **Payload builders** -- JSON-able dicts for a frontend.
18. **Demo**.

Conventions
-----------
* **Macroscopic units are (km, h)**: density ``k`` in veh/km *per lane*, speed
  ``v`` in km/h, flow ``q`` in veh/h *per lane*, length in km, time in hours.
  ``seconds(30.0)`` converts a signal timing into hours.
* **Microscopic units are SI (m, s)**: car-following gaps in metres, speeds in
  m/s, accelerations in m/s^2.  ``kmh_to_ms`` / ``ms_to_kmh`` bridge the two.
  Every micro function says so again in its own banner.
* ``gap`` in the car-following layer is the *clear* (bumper-to-bumper) distance;
  ``spacing = gap + vehicle_length``.
* A fundamental diagram is concave, ``q(0) = 0`` and ``q`` is maximised once, so
  the demand (sending) function is ``q`` capped above ``k_critical`` and the
  supply (receiving) function is ``q`` capped below it.
* All conservative solvers (Godunov, CTM, LTM, ARZ) conserve vehicles to
  machine precision; every result object carries ``conservation_error``.
* Everything stochastic takes an explicit ``seed`` and uses ``random.Random``,
  so results are byte-identical across runs and independent of PYTHONHASHSEED.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__version__ = "1.0.0"

Density = List[float]
Flow = List[float]
Speed = List[float]
Trajectory = List[Tuple[float, float]]
FlowFn = Callable[[float], float]

_EPS = 1e-12
# Smallest clear gap any car-following law is ever evaluated at, metres.  It is
# a module constant because string_stability has to differentiate across it.
_CF_MIN_GAP = 1e-6
_TINY = 1e-15
_GOLDEN_INV = (math.sqrt(5.0) - 1.0) / 2.0

#: km/h per m/s.
KMH_PER_MS = 3.6

__all__ = [
    # units
    "KMH_PER_MS", "seconds", "hours_to_seconds", "kmh_to_ms", "ms_to_kmh",
    # validation
    "TrafficFlowError",
    # fundamental diagrams
    "FundamentalDiagram", "Greenshields", "Greenberg", "Underwood", "Drake",
    "NewellDiagram", "TriangularDiagram", "TrapezoidalDiagram", "DelCastillo",
    "fd_from_name", "fd_capacity_numeric", "fd_table",
    # kinematic wave theory
    "characteristic_speed", "shock_speed", "rarefaction_density",
    "RiemannSolution", "riemann_solve",
    # godunov
    "godunov_flux", "max_wave_speed", "cfl_number", "cfl_timestep",
    "GodunovResult", "godunov_simulate",
    # cell transmission model
    "merge_flows", "diverge_flows", "CTMCell", "CTMNode", "CTMNetwork",
    "CTMResult", "ctm_corridor",
    # link transmission model
    "LTMLink", "LTMResult", "ltm_simulate",
    # cumulative curves
    "CumulativeCurve", "cumulative_from_flows", "delay_between_curves",
    "queue_from_curves", "travel_time_from_curves", "newell_three_detector",
    # queues
    "QueueResult", "vertical_queue", "horizontal_queue",
    # signal delay
    "deterministic_delay", "webster_delay", "akcelik_delay",
    "webster_optimal_cycle", "DelayBreakdown", "signal_delay",
    # shockwave analysis
    "ShockwaveResult", "shockwave_queue",
    # store and forward
    "SFLink", "StoreAndForwardResult", "StoreAndForwardModel",
    # macroscopic fundamental diagram
    "edie_flow_density", "aggregate_network_state", "MFDModel", "fit_mfd",
    "MFDAccumulationResult", "mfd_accumulation",
    # second-order models
    "SecondOrderResult", "METANETParams", "metanet_simulate",
    "payne_whitham_simulate", "arz_simulate",
    # car following
    "CarFollowingModel", "NewellCF", "GippsCF", "IDM", "OptimalVelocity",
    "FullVelocityDifference", "KraussCF", "PlatoonResult", "simulate_platoon",
    # lane changing
    "CFState", "MobilParams", "MobilDecision", "mobil_decision",
    # cellular automata
    "NaSchResult", "nagel_schreckenberg", "nasch_fundamental_diagram",
    # stability
    "StabilityResult", "string_stability", "platoon_amplification",
    # payloads
    "build_traffic_flow_payload", "build_a2ui_payload",
]


# ===========================================================================
# 0. VALIDATION HELPERS
# ===========================================================================
class TrafficFlowError(ValueError):
    """Invalid input to a traffic-flow routine."""


def _as_floats(x: Sequence[Any], name: str = "x") -> List[float]:
    """Coerce a sequence to a list of finite floats or raise ``TrafficFlowError``."""
    try:
        out = [float(v) for v in x]
    except (TypeError, ValueError):
        raise TrafficFlowError("%s must be a sequence of real numbers." % name)
    for v in out:
        if math.isnan(v) or math.isinf(v):
            raise TrafficFlowError("%s contains non-finite values." % name)
    return out


def _as_float(v: Any, name: str) -> float:
    """Coerce a scalar to a finite float or raise ``TrafficFlowError``."""
    try:
        out = float(v)
    except (TypeError, ValueError):
        raise TrafficFlowError("%s must be a real number." % name)
    if math.isnan(out) or math.isinf(out):
        raise TrafficFlowError("%s must be finite (got %r)." % (name, v))
    return out


def _positive(v: Any, name: str) -> float:
    """Coerce to a strictly positive float."""
    out = _as_float(v, name)
    if out <= 0.0:
        raise TrafficFlowError("%s must be > 0 (got %g)." % (name, out))
    return out


def _nonnegative(v: Any, name: str) -> float:
    """Coerce to a non-negative float."""
    out = _as_float(v, name)
    if out < 0.0:
        raise TrafficFlowError("%s must be >= 0 (got %g)." % (name, out))
    return out


def _fraction(v: Any, name: str) -> float:
    """Coerce to a float in [0, 1]."""
    out = _as_float(v, name)
    if out < 0.0 or out > 1.0:
        raise TrafficFlowError("%s must lie in [0, 1] (got %g)." % (name, out))
    return out


def _check_len(x: Sequence[Any], min_len: int, name: str = "x") -> None:
    """Raise unless ``x`` has at least ``min_len`` entries."""
    if len(x) < min_len:
        raise TrafficFlowError("%s needs at least %d entries (got %d)."
                               % (name, min_len, len(x)))


def _clip(v: float, lo: float, hi: float) -> float:
    """Clamp ``v`` into ``[lo, hi]``."""
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _as_flow_fn(v: Any, name: str) -> FlowFn:
    """Turn a constant or a callable ``f(t)`` into a callable flow profile."""
    if callable(v):
        return v
    const = _nonnegative(v, name)
    return lambda t: const


def _argmax_unimodal(f: Callable[[float], float], lo: float, hi: float,
                     n_scan: int = 512, n_refine: int = 200
                     ) -> Tuple[float, float]:
    """Maximiser of a unimodal ``f`` on ``[lo, hi]`` (coarse scan + golden section)."""
    if hi <= lo:
        return lo, f(lo)
    step = (hi - lo) / float(n_scan)
    best_x, best_f = lo, f(lo)
    for i in range(1, n_scan + 1):
        x = lo + i * step
        fx = f(x)
        if fx > best_f:
            best_f, best_x = fx, x
    a = max(lo, best_x - step)
    b = min(hi, best_x + step)
    c = b - _GOLDEN_INV * (b - a)
    d = a + _GOLDEN_INV * (b - a)
    fc, fdd = f(c), f(d)
    for _ in range(n_refine):
        if fc > fdd:
            b, d, fdd = d, c, fc
            c = b - _GOLDEN_INV * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fdd
            d = a + _GOLDEN_INV * (b - a)
            fdd = f(d)
        if (b - a) <= 1e-14 * max(1.0, abs(b)):
            break
    x = 0.5 * (a + b)
    fx = f(x)
    if best_f > fx:
        return best_x, best_f
    return x, fx


def _bisect(f: Callable[[float], float], a: float, b: float,
            tol: float = 1e-14, max_iter: int = 400) -> float:
    """Bisection root of ``f`` on a sign-changing bracket ``[a, b]``."""
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise TrafficFlowError("root is not bracketed on [%g, %g] "
                               "(f(a)=%g, f(b)=%g)." % (a, b, fa, fb))
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if fm == 0.0 or (b - a) <= tol * max(1.0, abs(m)):
            return m
        if fa * fm < 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def _mean(xs: Sequence[float]) -> float:
    """Arithmetic mean using ``math.fsum``."""
    if not xs:
        raise TrafficFlowError("cannot take the mean of an empty sequence.")
    return math.fsum(xs) / float(len(xs))


# --- unit helpers ----------------------------------------------------------
def seconds(s: float) -> float:
    """Convert a duration in seconds to hours (the macroscopic time unit)."""
    return _as_float(s, "s") / 3600.0


def hours_to_seconds(h: float) -> float:
    """Convert a duration in hours to seconds."""
    return _as_float(h, "h") * 3600.0


def kmh_to_ms(v: float) -> float:
    """Convert km/h to m/s."""
    return _as_float(v, "v") / KMH_PER_MS


def ms_to_kmh(v: float) -> float:
    """Convert m/s to km/h."""
    return _as_float(v, "v") * KMH_PER_MS


# ===========================================================================
# 1. FUNDAMENTAL DIAGRAMS
# ===========================================================================
class FundamentalDiagram:
    """Concave equilibrium flow-density relationship with demand/supply split."""

    def __init__(self, v_free: float, k_jam: float,
                 speed_fn: Callable[[float], float],
                 name: str = "generic",
                 dq_dk_fn: Optional[Callable[[float], float]] = None,
                 capacity_fn: Optional[Callable[[], Tuple[float, float]]] = None
                 ) -> None:
        self.v_free = _positive(v_free, "v_free")
        self.k_jam = _positive(k_jam, "k_jam")
        if not callable(speed_fn):
            raise TrafficFlowError("speed_fn must be callable.")
        self._speed_fn = speed_fn
        self._dq_dk_fn = dq_dk_fn
        self._capacity_fn = capacity_fn
        self.name = str(name)
        self._cap_pair = None            # type: Optional[Tuple[float, float]]
        self._max_wave = None            # type: Optional[float]

    # -- core surface -------------------------------------------------------
    def speed(self, k: float) -> float:
        """Equilibrium speed v(k) in km/h, clipped to [0, v_free]."""
        kk = _clip(_nonnegative(k, "k"), 0.0, self.k_jam)
        return _clip(float(self._speed_fn(kk)), 0.0, self.v_free)

    def flow(self, k: float) -> float:
        """Equilibrium flow q(k) = k v(k) in veh/h per lane."""
        kk = _clip(_nonnegative(k, "k"), 0.0, self.k_jam)
        return kk * self.speed(kk)

    def wave_speed(self, k: float) -> float:
        """Kinematic wave (characteristic) speed dq/dk at density ``k``, km/h."""
        kk = _clip(_nonnegative(k, "k"), 0.0, self.k_jam)
        if self._dq_dk_fn is not None:
            return float(self._dq_dk_fn(kk))
        h = 1e-7 * max(1.0, self.k_jam)
        lo = max(0.0, kk - h)
        hi = min(self.k_jam, kk + h)
        if hi <= lo:
            return 0.0
        return (self.flow(hi) - self.flow(lo)) / (hi - lo)

    def _capacity_pair(self) -> Tuple[float, float]:
        """Cached (critical density, capacity) pair."""
        if self._cap_pair is None:
            if self._capacity_fn is not None:
                k_c, q_c = self._capacity_fn()
            else:
                k_c, q_c = _argmax_unimodal(self.flow, 0.0, self.k_jam)
            self._cap_pair = (float(k_c), float(q_c))
        return self._cap_pair

    @property
    def k_critical(self) -> float:
        """Density at which flow is maximised, veh/km per lane."""
        return self._capacity_pair()[0]

    @property
    def capacity(self) -> float:
        """Maximum flow q(k_critical), veh/h per lane."""
        return self._capacity_pair()[1]

    @property
    def v_critical(self) -> float:
        """Speed at capacity, km/h."""
        k_c = self.k_critical
        return self.capacity / k_c if k_c > _EPS else self.v_free

    # -- Godunov demand / supply -------------------------------------------
    def demand(self, k: float) -> float:
        """Sending function: q(k) below capacity density, capacity above."""
        kk = _clip(_nonnegative(k, "k"), 0.0, self.k_jam)
        return self.flow(kk) if kk <= self.k_critical else self.capacity

    def supply(self, k: float) -> float:
        """Receiving function: capacity below capacity density, q(k) above."""
        kk = _clip(_nonnegative(k, "k"), 0.0, self.k_jam)
        return self.capacity if kk <= self.k_critical else self.flow(kk)

    # -- inverses & tables --------------------------------------------------
    def density_from_flow(self, q: float, congested: bool = False) -> float:
        """Smallest free-flow (default) / largest congested density carrying flow ``q``."""
        # On a plateau every density in the flat range carries q and the end of
        # the branch is returned.  Diagrams whose speed never reaches zero
        # (Underwood, Drake) cannot carry small flows at all: they return k_jam.
        qq = _nonnegative(q, "q")
        cap = self.capacity
        if qq > cap * (1.0 + 1e-9):
            raise TrafficFlowError("flow %g exceeds the capacity %g of '%s'."
                                   % (qq, cap, self.name))
        qq = min(qq, cap)
        k_c = self.k_critical
        tol = 1e-13 * max(1.0, self.k_jam)
        # a plateau makes q(k) only weakly monotone and k*(q_max/k) is not
        # bit-exact, so the branch is located with a tolerant predicate
        qtol = 1e-12 * max(1.0, cap)
        if congested:
            # largest density on the congested branch that still carries q
            lo, hi = k_c, self.k_jam
            if self.flow(hi) >= qq - qtol:
                return hi
            for _ in range(300):
                mid = 0.5 * (lo + hi)
                if self.flow(mid) >= qq - qtol:
                    lo = mid
                else:
                    hi = mid
                if hi - lo <= tol:
                    break
            return lo
        # smallest density on the free-flow branch that carries q
        lo, hi = 0.0, k_c
        if self.flow(lo) >= qq - qtol:
            return lo
        for _ in range(300):
            mid = 0.5 * (lo + hi)
            if self.flow(mid) >= qq - qtol:
                hi = mid
            else:
                lo = mid
            if hi - lo <= tol:
                break
        return hi

    def speed_from_density(self, k: float) -> float:
        """Alias of :meth:`speed` kept for API parity with the literature."""
        return self.speed(k)

    def max_wave_speed(self) -> float:
        """Largest |dq/dk| over the diagram -- the CFL wave speed, km/h."""
        if self._max_wave is None:
            best = abs(self.v_free)
            n = 1024
            for i in range(n + 1):
                k = self.k_jam * i / float(n)
                c = abs(self.wave_speed(k))
                if c > best:
                    best = c
            self._max_wave = best
        return self._max_wave

    def summary(self) -> Dict[str, float]:
        """Headline parameters of the diagram as a plain dict."""
        return {
            "name": self.name,
            "vFree": self.v_free,
            "kJam": self.k_jam,
            "capacity": self.capacity,
            "kCritical": self.k_critical,
            "vCritical": self.v_critical,
            "maxWaveSpeed": self.max_wave_speed(),
        }

    def __repr__(self) -> str:
        return ("<%s v_free=%.1f k_jam=%.1f capacity=%.1f k_crit=%.2f>"
                % (self.__class__.__name__, self.v_free, self.k_jam,
                   self.capacity, self.k_critical))


class Greenshields(FundamentalDiagram):
    """Greenshields (1935) linear speed-density law: v = v_f (1 - k/k_j)."""

    def __init__(self, v_free: float = 100.0, k_jam: float = 180.0) -> None:
        super().__init__(v_free, k_jam, self._speed_model, "greenshields",
                         self._dq_dk, self._capacity_closed_form)

    def _speed_model(self, k: float) -> float:
        return self.v_free * (1.0 - k / self.k_jam)

    def _dq_dk(self, k: float) -> float:
        return self.v_free * (1.0 - 2.0 * k / self.k_jam)

    def _capacity_closed_form(self) -> Tuple[float, float]:
        return 0.5 * self.k_jam, 0.25 * self.v_free * self.k_jam


class Greenberg(FundamentalDiagram):
    """Greenberg (1959) logarithmic law: v = v_opt ln(k_j / k), capped at v_free."""

    def __init__(self, v_opt: float = 30.0, k_jam: float = 180.0,
                 v_free: Optional[float] = None) -> None:
        self.v_opt = _positive(v_opt, "v_opt")
        vf = self.v_opt * 2.0 if v_free is None else _positive(v_free, "v_free")
        super().__init__(vf, k_jam, self._speed_model, "greenberg",
                         self._dq_dk, self._capacity_closed_form)

    def _k_cap(self) -> float:
        """Density below which the free-speed cap binds."""
        return self.k_jam * math.exp(-self.v_free / self.v_opt)

    def _speed_model(self, k: float) -> float:
        if k <= _TINY:
            return self.v_free
        return min(self.v_free, self.v_opt * math.log(self.k_jam / k))

    def _dq_dk(self, k: float) -> float:
        if k <= self._k_cap():
            return self.v_free
        if k <= _TINY:
            return self.v_free
        return self.v_opt * (math.log(self.k_jam / k) - 1.0)

    def _capacity_closed_form(self) -> Tuple[float, float]:
        if self.v_free >= self.v_opt:
            k_c = self.k_jam / math.e
            return k_c, self.v_opt * k_c
        k_c = self._k_cap()
        return k_c, self.v_free * k_c


class Underwood(FundamentalDiagram):
    """Underwood (1961) exponential law: v = v_f exp(-k / k_c)."""

    def __init__(self, v_free: float = 100.0, k_critical: float = 35.0,
                 k_jam: float = 180.0) -> None:
        self._k_c = _positive(k_critical, "k_critical")
        super().__init__(v_free, k_jam, self._speed_model, "underwood",
                         self._dq_dk, self._capacity_closed_form)

    def _speed_model(self, k: float) -> float:
        return self.v_free * math.exp(-k / self._k_c)

    def _dq_dk(self, k: float) -> float:
        return self.v_free * math.exp(-k / self._k_c) * (1.0 - k / self._k_c)

    def _capacity_closed_form(self) -> Tuple[float, float]:
        return self._k_c, self.v_free * self._k_c / math.e


class Drake(FundamentalDiagram):
    """Drake / Northwestern (1967) bell law: v = v_f exp(-(k/k_c)^2 / 2)."""

    def __init__(self, v_free: float = 100.0, k_critical: float = 35.0,
                 k_jam: float = 180.0) -> None:
        self._k_c = _positive(k_critical, "k_critical")
        super().__init__(v_free, k_jam, self._speed_model, "drake",
                         self._dq_dk, self._capacity_closed_form)

    def _speed_model(self, k: float) -> float:
        r = k / self._k_c
        return self.v_free * math.exp(-0.5 * r * r)

    def _dq_dk(self, k: float) -> float:
        r = k / self._k_c
        return self.v_free * math.exp(-0.5 * r * r) * (1.0 - r * r)

    def _capacity_closed_form(self) -> Tuple[float, float]:
        return self._k_c, self.v_free * self._k_c / math.sqrt(math.e)


class NewellDiagram(FundamentalDiagram):
    """Newell (1961) exponential FD: v = v_f [1 - exp(-(w/v_f)(k_j/k - 1))]."""

    def __init__(self, v_free: float = 100.0, k_jam: float = 180.0,
                 w: float = 20.0) -> None:
        self.w = _positive(w, "w")
        super().__init__(v_free, k_jam, self._speed_model, "newell",
                         self._dq_dk, self._capacity_closed_form)

    def _a(self) -> float:
        return self.w / self.v_free

    def _speed_model(self, k: float) -> float:
        if k <= _TINY:
            return self.v_free
        arg = -self._a() * (self.k_jam / k - 1.0)
        return self.v_free * (1.0 - math.exp(arg))

    def _dq_dk(self, k: float) -> float:
        if k <= _TINY:
            return self.v_free
        a = self._a()
        e = math.exp(-a * (self.k_jam / k - 1.0))
        return self.v_free * (1.0 - e * (1.0 + a * self.k_jam / k))

    def _capacity_closed_form(self) -> Tuple[float, float]:
        lo = self.k_jam * 1e-9
        k_c = _bisect(self._dq_dk, lo, self.k_jam)
        return k_c, self.flow(k_c)


class DelCastillo(FundamentalDiagram):
    """Del Castillo-Benitez (1995) maximum-sensitivity curve (double exponential)."""

    def __init__(self, v_free: float = 100.0, k_jam: float = 180.0,
                 w: float = 20.0) -> None:
        self.w = _positive(w, "w")
        super().__init__(v_free, k_jam, self._speed_model, "del_castillo",
                         self._dq_dk, self._capacity_closed_form)

    def _g(self, k: float) -> float:
        return (self.w / self.v_free) * (self.k_jam / k - 1.0)

    def _speed_model(self, k: float) -> float:
        if k <= _TINY:
            return self.v_free
        g = self._g(k)
        if g > 700.0:
            return self.v_free
        return self.v_free * (1.0 - math.exp(1.0 - math.exp(g)))

    def _dq_dk(self, k: float) -> float:
        if k <= _TINY:
            return self.v_free
        g = self._g(k)
        if g > 700.0:
            return self.v_free
        eg = math.exp(g)
        f = math.exp(1.0 - eg)
        return self.v_free * (1.0 - f) - f * eg * self.w * self.k_jam / k

    def _capacity_closed_form(self) -> Tuple[float, float]:
        lo = self.k_jam * 1e-6
        k_c = _bisect(self._dq_dk, lo, self.k_jam)
        return k_c, self.flow(k_c)


class TriangularDiagram(FundamentalDiagram):
    """Daganzo's triangular FD: q = min(v_f k, w (k_j - k))."""

    def __init__(self, v_free: float = 100.0, k_jam: float = 180.0,
                 w: float = 20.0) -> None:
        self.w = _positive(w, "w")
        super().__init__(v_free, k_jam, self._speed_model, "triangular",
                         self._dq_dk, self._capacity_closed_form)

    def _speed_model(self, k: float) -> float:
        if k <= _TINY:
            return self.v_free
        return min(self.v_free, self.w * (self.k_jam / k - 1.0))

    def _dq_dk(self, k: float) -> float:
        return self.v_free if k <= self.k_critical else -self.w

    def _capacity_closed_form(self) -> Tuple[float, float]:
        k_c = self.w * self.k_jam / (self.v_free + self.w)
        return k_c, self.v_free * k_c


class TrapezoidalDiagram(FundamentalDiagram):
    """Trapezoidal FD: q = min(v_f k, q_max, w (k_j - k)) -- a capacity plateau."""

    def __init__(self, v_free: float = 100.0, k_jam: float = 180.0,
                 w: float = 20.0, q_max: float = 2000.0) -> None:
        self.w = _positive(w, "w")
        self.q_max = _positive(q_max, "q_max")
        tri_cap = v_free * w * k_jam / (v_free + w)
        if self.q_max > tri_cap * (1.0 + 1e-12):
            raise TrafficFlowError(
                "q_max %g exceeds the triangular envelope %g; lower q_max or "
                "raise v_free / w / k_jam." % (self.q_max, tri_cap))
        super().__init__(v_free, k_jam, self._speed_model, "trapezoidal",
                         self._dq_dk, self._capacity_closed_form)

    @property
    def k_plateau_start(self) -> float:
        """Density at which the capacity plateau begins."""
        return self.q_max / self.v_free

    @property
    def k_plateau_end(self) -> float:
        """Density at which the congested branch takes over."""
        return self.k_jam - self.q_max / self.w

    def _speed_model(self, k: float) -> float:
        if k <= _TINY:
            return self.v_free
        return min(self.v_free, self.q_max / k, self.w * (self.k_jam / k - 1.0))

    def flow(self, k: float) -> float:
        """q = min(v_f k, q_max, w (k_j - k)) -- exact q_max on the plateau."""
        # The base class computes k * speed(k); on the plateau speed is q_max / k, so
        # the round-trip loses a bit and flow() came back 1 ulp off q_max.  That made
        # flow() disagree with demand() for the SAME state and injected a 1-ulp drift
        # into an otherwise bit-exact Godunov update.  The min-form returns q_max
        # exactly because q_max is itself one of the operands.
        kk = _clip(_nonnegative(k, "k"), 0.0, self.k_jam)
        return min(self.v_free * kk, self.q_max, self.w * (self.k_jam - kk))

    def _dq_dk(self, k: float) -> float:
        if k < self.k_plateau_start:
            return self.v_free
        if k <= self.k_plateau_end:
            return 0.0
        return -self.w

    def _capacity_closed_form(self) -> Tuple[float, float]:
        return self.k_plateau_start, self.q_max


_FD_REGISTRY = {
    "greenshields": Greenshields,
    "greenberg": Greenberg,
    "underwood": Underwood,
    "drake": Drake,
    "northwestern": Drake,
    "newell": NewellDiagram,
    "triangular": TriangularDiagram,
    "daganzo": TriangularDiagram,
    "trapezoidal": TrapezoidalDiagram,
    "del_castillo": DelCastillo,
}


def fd_from_name(name: str, **kwargs: Any) -> FundamentalDiagram:
    """Construct a fundamental diagram by name (``'greenshields'``, ``'idm'``-free)."""
    key = str(name).strip().lower()
    if key not in _FD_REGISTRY:
        raise TrafficFlowError("unknown fundamental diagram '%s' (known: %s)."
                               % (name, ", ".join(sorted(_FD_REGISTRY))))
    return _FD_REGISTRY[key](**kwargs)


def fd_capacity_numeric(fd: FundamentalDiagram, n: int = 200001
                        ) -> Tuple[float, float]:
    """Brute-force (k_critical, capacity) by scanning q(k) -- the test oracle."""
    if n < 3:
        raise TrafficFlowError("n must be >= 3.")
    best_k, best_q = 0.0, 0.0
    for i in range(n):
        k = fd.k_jam * i / float(n - 1)
        q = fd.flow(k)
        if q > best_q:
            best_q, best_k = q, k
    return best_k, best_q


def fd_table(fd: FundamentalDiagram, n: int = 51) -> List[Dict[str, float]]:
    """Sample the diagram on ``n`` densities -- ready for plotting."""
    if n < 2:
        raise TrafficFlowError("n must be >= 2.")
    out = []
    for i in range(n):
        k = fd.k_jam * i / float(n - 1)
        out.append({"k": k, "q": fd.flow(k), "v": fd.speed(k),
                    "waveSpeed": fd.wave_speed(k)})
    return out


# ===========================================================================
# 2. KINEMATIC WAVE THEORY (LWR)
# ===========================================================================
def characteristic_speed(fd: FundamentalDiagram, k: float) -> float:
    """Speed of the kinematic wave carrying density ``k``: dq/dk, km/h."""
    return fd.wave_speed(k)


def shock_speed(k1: float, q1: float, k2: float, q2: float) -> float:
    """Rankine-Hugoniot shock speed (q2 - q1) / (k2 - k1), km/h."""
    kk1 = _nonnegative(k1, "k1")
    kk2 = _nonnegative(k2, "k2")
    qq1 = _as_float(q1, "q1")
    qq2 = _as_float(q2, "q2")
    dk = kk2 - kk1
    # The "are the two states the same?" test has to be relative to the density
    # scale: an absolute 1e-12 veh/km threshold rejects a perfectly resolvable
    # density step when the caller works in veh/m, and accepts a numerically
    # meaningless one when the caller works in veh/mile.  A shock speed must
    # not depend on the unit the densities are quoted in.
    if abs(dk) <= _EPS * max(abs(kk1), abs(kk2)):
        raise TrafficFlowError("Rankine-Hugoniot needs k1 != k2 "
                               "(got k1=%g, k2=%g)." % (kk1, kk2))
    return (qq2 - qq1) / dk


def rarefaction_density(fd: FundamentalDiagram, k_left: float, k_right: float,
                        xi: float) -> float:
    """Density inside an expansion fan at similarity variable ``xi = x / t``."""
    kl = _clip(_nonnegative(k_left, "k_left"), 0.0, fd.k_jam)
    kr = _clip(_nonnegative(k_right, "k_right"), 0.0, fd.k_jam)
    if kl <= kr:
        raise TrafficFlowError("a rarefaction needs k_left > k_right "
                               "(got %g <= %g)." % (kl, kr))
    x = _as_float(xi, "xi")
    c_left = fd.wave_speed(kl)
    c_right = fd.wave_speed(kr)
    if x <= c_left:
        return kl
    if x >= c_right:
        return kr
    # dq/dk is non-increasing in k on a concave diagram -> bisect on c(k) - xi
    lo, hi = kr, kl
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if fd.wave_speed(mid) - x > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-13 * max(1.0, fd.k_jam):
            break
    return 0.5 * (lo + hi)


@dataclass
class RiemannSolution:
    """Similarity solution of the LWR Riemann problem for two uniform states."""

    k_left: float
    k_right: float
    kind: str                       # 'uniform' | 'shock' | 'rarefaction'
    shock_speed: Optional[float]
    head_speed: float
    tail_speed: float
    q_left: float
    q_right: float
    interface_flux: float
    _fd: Any = field(repr=False, default=None)

    def density_at(self, x: float, t: float) -> float:
        """Density at position ``x`` (km from the discontinuity) and time ``t`` (h)."""
        tt = _nonnegative(t, "t")
        if tt <= _EPS:
            return self.k_left if _as_float(x, "x") < 0.0 else self.k_right
        xi = _as_float(x, "x") / tt
        if self.kind == "uniform":
            return self.k_left
        if self.kind == "shock":
            return self.k_left if xi < self.shock_speed else self.k_right
        return rarefaction_density(self._fd, self.k_left, self.k_right, xi)

    def front_position(self, t: float) -> float:
        """Position of the shock (or of the fan head) at time ``t``, km."""
        tt = _nonnegative(t, "t")
        if self.kind == "shock":
            return self.shock_speed * tt
        return self.head_speed * tt


def riemann_solve(fd: FundamentalDiagram, k_left: float, k_right: float
                  ) -> RiemannSolution:
    """Solve the LWR Riemann problem: shock if k_left < k_right, else a fan."""
    kl = _clip(_nonnegative(k_left, "k_left"), 0.0, fd.k_jam)
    kr = _clip(_nonnegative(k_right, "k_right"), 0.0, fd.k_jam)
    ql, qr = fd.flow(kl), fd.flow(kr)
    flux = min(fd.demand(kl), fd.supply(kr))
    # Scale-relative for the same reason as shock_speed above: rescaling the
    # whole diagram (k_jam, k_left, k_right) by a common factor must not turn a
    # genuine shock into a "uniform" state.
    if abs(kl - kr) <= _EPS * max(kl, kr, fd.k_jam):
        return RiemannSolution(kl, kr, "uniform", None, fd.wave_speed(kl),
                               fd.wave_speed(kl), ql, qr, flux, fd)
    if kl < kr:
        s = shock_speed(kl, ql, kr, qr)
        return RiemannSolution(kl, kr, "shock", s, s, s, ql, qr, flux, fd)
    return RiemannSolution(kl, kr, "rarefaction", None,
                           fd.wave_speed(kl), fd.wave_speed(kr),
                           ql, qr, flux, fd)


# ===========================================================================
# 3. GODUNOV SCHEME FOR THE LWR EQUATION
# ===========================================================================
def godunov_flux(fd: FundamentalDiagram, k_up: float, k_down: float) -> float:
    """Godunov interface flux min(demand(upstream), supply(downstream)), veh/h."""
    return min(fd.demand(k_up), fd.supply(k_down))


def max_wave_speed(fd: FundamentalDiagram) -> float:
    """Largest characteristic speed magnitude of the diagram, km/h."""
    return fd.max_wave_speed()


def cfl_number(fd: FundamentalDiagram, dx: float, dt: float) -> float:
    """Courant number ``dt * max|dq/dk| / dx`` -- must be <= 1 for stability."""
    return _positive(dt, "dt") * fd.max_wave_speed() / _positive(dx, "dx")


def cfl_timestep(fd: FundamentalDiagram, dx: float, cfl: float = 0.9) -> float:
    """Largest stable time step for cell size ``dx`` at Courant number ``cfl``."""
    c = _positive(cfl, "cfl")
    if c > 1.0:
        raise TrafficFlowError("cfl must be <= 1 (got %g)." % c)
    return c * _positive(dx, "dx") / fd.max_wave_speed()


@dataclass
class GodunovResult:
    """Output of a conservative finite-volume LWR run."""

    times: List[float]
    x: List[float]
    densities: List[List[float]]
    inflow_cum: List[float]
    outflow_cum: List[float]
    vehicles: List[float]
    dx: float
    dt: float
    lanes: float
    cfl: float
    conservation_error: float
    max_density: float
    min_density: float

    def front_position(self, threshold: float) -> List[float]:
        """Furthest upstream cell centre whose density exceeds ``threshold``, per time."""
        out = []
        for row in self.densities:
            pos = float("nan")
            for i, k in enumerate(row):
                if k >= threshold:
                    pos = self.x[i]
                    break
            out.append(pos)
        return out

    def summary(self) -> Dict[str, Any]:
        """Headline numbers of the run."""
        return {
            "model": "godunov",
            "nCells": len(self.x),
            "nSteps": len(self.times) - 1,
            "dxKm": self.dx,
            "dtH": self.dt,
            "cfl": self.cfl,
            "vehiclesIn": self.inflow_cum[-1],
            "vehiclesOut": self.outflow_cum[-1],
            "vehiclesStored": self.vehicles[-1],
            "conservationError": self.conservation_error,
            "maxDensity": self.max_density,
            "minDensity": self.min_density,
        }


def godunov_simulate(fd: FundamentalDiagram, k_initial: Sequence[float],
                     dx: float, dt: float, n_steps: int,
                     inflow: Any = None, downstream_supply: Any = None,
                     lanes: float = 1.0, record_every: int = 1
                     ) -> GodunovResult:
    """Solve the LWR equation with the Godunov (demand/supply) finite-volume scheme."""
    k = _as_floats(k_initial, "k_initial")
    _check_len(k, 1, "k_initial")
    ddx = _positive(dx, "dx")
    ddt = _positive(dt, "dt")
    if int(n_steps) < 1:
        raise TrafficFlowError("n_steps must be >= 1.")
    n_steps = int(n_steps)
    lam = _positive(lanes, "lanes")
    every = max(1, int(record_every))
    for v in k:
        if v < -_EPS or v > fd.k_jam * (1.0 + 1e-9):
            raise TrafficFlowError("initial density %g outside [0, k_jam=%g]."
                                   % (v, fd.k_jam))
    k = [_clip(v, 0.0, fd.k_jam) for v in k]
    courant = cfl_number(fd, ddx, ddt)
    if courant > 1.0 + 1e-12:
        raise TrafficFlowError(
            "CFL violated: dt=%g needs dt <= %g for dx=%g (Courant %.3f)."
            % (ddt, cfl_timestep(fd, ddx, 1.0), ddx, courant))
    up_fn = _as_flow_fn(fd.capacity if inflow is None else inflow, "inflow")
    dn_fn = _as_flow_fn(fd.capacity if downstream_supply is None
                        else downstream_supply, "downstream_supply")

    n_cells = len(k)
    x = [(i + 0.5) * ddx for i in range(n_cells)]
    times = [0.0]
    hist = [list(k)]
    in_cum = [0.0]
    out_cum = [0.0]
    veh = [math.fsum(k) * ddx * lam]
    veh0 = veh[0]
    in_terms = []           # type: List[float]
    out_terms = []          # type: List[float]
    kmax = max(k)
    kmin = min(k)

    for step in range(n_steps):
        t = step * ddt
        # A cell can never send more than it holds nor receive more than the
        # room it has left.  For a diagram with a true jam (q(k_jam) = 0) these
        # two limits are slack under the CFL condition, so the scheme is
        # bit-identical to the textbook one; for Underwood / Drake, whose speed
        # never reaches zero, they are what stops the [0, k_jam] clip below
        # from silently destroying vehicles.
        send = [min(fd.demand(k[i]), k[i] * ddx / ddt) for i in range(n_cells)]
        recv = [min(fd.supply(k[i]), (fd.k_jam - k[i]) * ddx / ddt)
                for i in range(n_cells)]
        fluxes = [0.0] * (n_cells + 1)
        fluxes[0] = max(0.0, min(up_fn(t), recv[0]))
        for i in range(1, n_cells):
            fluxes[i] = min(send[i - 1], recv[i])
        fluxes[n_cells] = max(0.0, min(send[n_cells - 1], dn_fn(t)))
        ratio = ddt / ddx
        k = [_clip(k[i] + ratio * (fluxes[i] - fluxes[i + 1]), 0.0, fd.k_jam)
             for i in range(n_cells)]
        in_terms.append(fluxes[0] * ddt * lam)
        out_terms.append(fluxes[n_cells] * ddt * lam)
        kmax = max(kmax, max(k))
        kmin = min(kmin, min(k))
        if (step + 1) % every == 0 or step == n_steps - 1:
            times.append((step + 1) * ddt)
            hist.append(list(k))
            in_cum.append(math.fsum(in_terms))
            out_cum.append(math.fsum(out_terms))
            veh.append(math.fsum(k) * ddx * lam)

    err = abs((veh[-1] - veh0) - (in_cum[-1] - out_cum[-1]))
    return GodunovResult(times, x, hist, in_cum, out_cum, veh, ddx, ddt, lam,
                         courant, err, kmax, kmin)


# ===========================================================================
# 4. CELL TRANSMISSION MODEL (DAGANZO)
# ===========================================================================
def merge_flows(demands: Sequence[float], supply: float,
                priorities: Optional[Sequence[float]] = None) -> List[float]:
    """Daganzo priority merge: share a scarce supply between competing demands."""
    d = _as_floats(demands, "demands")
    _check_len(d, 1, "demands")
    for v in d:
        if v < 0.0:
            raise TrafficFlowError("demands must be >= 0.")
    r = _nonnegative(supply, "supply")
    n = len(d)
    if priorities is None:
        p = [1.0 / n] * n
    else:
        p = _as_floats(priorities, "priorities")
        if len(p) != n:
            raise TrafficFlowError("priorities must match demands in length.")
        for v in p:
            if v < 0.0:
                raise TrafficFlowError("priorities must be >= 0.")
        tot = math.fsum(p)
        if tot <= _EPS:
            raise TrafficFlowError("priorities must not all be zero.")
        p = [v / tot for v in p]
    if math.fsum(d) <= r:
        return list(d)
    out = [0.0] * n
    active = list(range(n))
    remaining = r
    while active:
        p_sum = math.fsum(p[i] for i in active)
        if p_sum <= _EPS:                       # zero-priority leftovers
            share = remaining / float(len(active))
            for i in active:
                out[i] = min(d[i], share)
            break
        satisfied = []
        for i in active:
            share = remaining * p[i] / p_sum
            if d[i] <= share:
                satisfied.append(i)
        if not satisfied:
            for i in active:
                out[i] = remaining * p[i] / p_sum
            break
        for i in satisfied:
            out[i] = d[i]
            remaining -= d[i]
            active.remove(i)
        if remaining <= _EPS:
            break
    return out


def diverge_flows(demand: float, supplies: Sequence[float],
                  splits: Sequence[float]) -> Tuple[float, List[float]]:
    """FIFO diverge: the most restrictive downstream branch throttles the whole flow."""
    dd = _nonnegative(demand, "demand")
    s = _as_floats(supplies, "supplies")
    b = _as_floats(splits, "splits")
    if len(s) != len(b):
        raise TrafficFlowError("supplies and splits must have equal length.")
    _check_len(s, 1, "supplies")
    for v in s:
        if v < 0.0:
            raise TrafficFlowError("supplies must be >= 0.")
    tot = math.fsum(b)
    if tot <= _EPS:
        raise TrafficFlowError("split ratios must not all be zero.")
    b = [v / tot for v in b]
    limit = dd
    for j in range(len(s)):
        if b[j] > _EPS:
            limit = min(limit, s[j] / b[j])
    limit = max(0.0, limit)
    return limit, [limit * bj for bj in b]


@dataclass
class CTMCell:
    """One CTM cell: a length of road holding ``n`` vehicles."""

    cell_id: str
    length_km: float
    lanes: float
    fd: FundamentalDiagram
    capacity_drop: float = 0.0
    n: float = 0.0

    def __post_init__(self) -> None:
        self.length_km = _positive(self.length_km, "length_km")
        self.lanes = _positive(self.lanes, "lanes")
        self.capacity_drop = _fraction(self.capacity_drop, "capacity_drop")
        self.n = _nonnegative(self.n, "n")
        if not isinstance(self.fd, FundamentalDiagram):
            raise TrafficFlowError("cell '%s' needs a FundamentalDiagram."
                                   % self.cell_id)

    @property
    def lane_km(self) -> float:
        """Lane-kilometres of storage."""
        return self.length_km * self.lanes

    @property
    def storage(self) -> float:
        """Jam storage of the cell in vehicles."""
        return self.fd.k_jam * self.lane_km

    @property
    def density(self) -> float:
        """Current density, veh/km per lane."""
        return self.n / self.lane_km

    @property
    def speed(self) -> float:
        """Current equilibrium speed, km/h."""
        return self.fd.speed(self.density)

    def demand_flow(self) -> float:
        """Sending flow in veh/h (with the capacity drop applied when congested)."""
        k = self.density
        d = self.fd.demand(k) * self.lanes
        if self.capacity_drop > 0.0 and k > self.fd.k_critical:
            d *= (1.0 - self.capacity_drop)
        return d

    def supply_flow(self) -> float:
        """Receiving flow in veh/h."""
        return self.fd.supply(self.density) * self.lanes

    def demand_veh(self, dt: float) -> float:
        """Vehicles the cell can send during ``dt`` hours."""
        return min(self.demand_flow() * dt, self.n)

    def supply_veh(self, dt: float) -> float:
        """Vehicles the cell can accept during ``dt`` hours."""
        return min(self.supply_flow() * dt, max(0.0, self.storage - self.n))


@dataclass
class CTMNode:
    """A CTM node: source, sink, ordinary link, priority merge or FIFO diverge."""

    node_id: str
    upstream: List[str] = field(default_factory=list)
    downstream: List[str] = field(default_factory=list)
    priorities: Optional[List[float]] = None
    splits: Optional[List[float]] = None
    demand_fn: Optional[FlowFn] = None
    supply_fn: Optional[FlowFn] = None

    @property
    def kind(self) -> str:
        """Node classification used by the solver."""
        nu, nd = len(self.upstream), len(self.downstream)
        if nu == 0 and nd == 1:
            return "source"
        if nu == 1 and nd == 0:
            return "sink"
        if nu == 1 and nd == 1:
            return "link"
        if nu > 1 and nd == 1:
            return "merge"
        if nu == 1 and nd > 1:
            return "diverge"
        raise TrafficFlowError("node '%s' has an unsupported %d->%d topology; "
                               "split it into a diverge followed by a merge."
                               % (self.node_id, nu, nd))


@dataclass
class CTMResult:
    """Output of a CTM run."""

    times: List[float]
    cell_ids: List[str]
    densities: List[List[float]]
    vehicles: List[List[float]]
    speeds: List[List[float]]
    node_flows: List[Dict[str, float]]
    inflow_cum: List[float]
    outflow_cum: List[float]
    stored: List[float]
    dt: float
    conservation_error: float
    total_travel_time: float
    total_travel_distance: float
    total_delay: float
    max_density: float

    def density_of(self, cell_id: str) -> List[float]:
        """Density history of one cell."""
        if cell_id not in self.cell_ids:
            raise TrafficFlowError("unknown cell '%s'." % cell_id)
        j = self.cell_ids.index(cell_id)
        return [row[j] for row in self.densities]

    def summary(self) -> Dict[str, Any]:
        """Headline numbers of the run."""
        return {
            "model": "ctm",
            "nCells": len(self.cell_ids),
            "nSteps": len(self.times) - 1,
            "dtH": self.dt,
            "vehiclesIn": self.inflow_cum[-1],
            "vehiclesOut": self.outflow_cum[-1],
            "vehiclesStored": self.stored[-1],
            "conservationError": self.conservation_error,
            "totalTravelTimeVehH": self.total_travel_time,
            "totalTravelDistanceVehKm": self.total_travel_distance,
            "totalDelayVehH": self.total_delay,
            "maxDensity": self.max_density,
        }


class CTMNetwork:
    """Daganzo's Cell Transmission Model on an arbitrary cell/node network.

    The optional ``dt`` recorded at construction is the time step the network
    was *designed* for; ``simulate`` falls back to it when none is passed.
    """

    def __init__(self, dt: Optional[float] = None) -> None:
        self.cells = {}          # type: Dict[str, CTMCell]
        self.order = []          # type: List[str]
        self.nodes = []          # type: List[CTMNode]
        self.dt = None if dt is None else _positive(dt, "dt")

    # -- construction -------------------------------------------------------
    def add_cell(self, cell_id: str, length_km: float, fd: FundamentalDiagram,
                 lanes: float = 1.0, capacity_drop: float = 0.0,
                 n: float = 0.0) -> CTMCell:
        """Append a cell to the network."""
        key = str(cell_id)
        if key in self.cells:
            raise TrafficFlowError("duplicate cell id '%s'." % key)
        cell = CTMCell(key, length_km, lanes, fd, capacity_drop, n)
        self.cells[key] = cell
        self.order.append(key)
        return cell

    def _require(self, cell_id: str) -> CTMCell:
        if cell_id not in self.cells:
            raise TrafficFlowError("unknown cell '%s'." % cell_id)
        return self.cells[cell_id]

    def add_node(self, node: CTMNode) -> CTMNode:
        """Append a pre-built node (its topology is validated immediately)."""
        for cid in list(node.upstream) + list(node.downstream):
            self._require(cid)
        node.kind                                  # validates the topology
        self.nodes.append(node)
        return node

    def connect(self, up: str, down: str, node_id: Optional[str] = None
                ) -> CTMNode:
        """Join two cells with an ordinary link node."""
        nid = node_id or ("%s->%s" % (up, down))
        return self.add_node(CTMNode(nid, [up], [down]))

    def add_merge(self, ups: Sequence[str], down: str,
                  priorities: Optional[Sequence[float]] = None,
                  node_id: Optional[str] = None) -> CTMNode:
        """Join several cells to one with Daganzo's priority merge."""
        nid = node_id or ("merge->%s" % down)
        pr = None if priorities is None else _as_floats(priorities, "priorities")
        return self.add_node(CTMNode(nid, list(ups), [down], priorities=pr))

    def add_diverge(self, up: str, downs: Sequence[str],
                    splits: Sequence[float],
                    node_id: Optional[str] = None) -> CTMNode:
        """Split one cell into several with FIFO split ratios."""
        nid = node_id or ("%s->diverge" % up)
        sp = _as_floats(splits, "splits")
        if len(sp) != len(list(downs)):
            raise TrafficFlowError("splits must match downs in length.")
        return self.add_node(CTMNode(nid, [up], list(downs), splits=sp))

    def add_source(self, down: str, demand: Any,
                   node_id: Optional[str] = None) -> CTMNode:
        """Attach an inflow boundary (constant or callable veh/h) to a cell."""
        nid = node_id or ("source->%s" % down)
        return self.add_node(CTMNode(nid, [], [down],
                                     demand_fn=_as_flow_fn(demand, "demand")))

    def add_sink(self, up: str, supply: Any = None,
                 node_id: Optional[str] = None) -> CTMNode:
        """Attach an outflow boundary (constant or callable veh/h) to a cell."""
        nid = node_id or ("%s->sink" % up)
        if supply is None:
            cell = self._require(up)
            supply = cell.fd.capacity * cell.lanes
        return self.add_node(CTMNode(nid, [up], [],
                                     supply_fn=_as_flow_fn(supply, "supply")))

    def validate(self) -> None:
        """Check that every cell has at most one inbound and one outbound node."""
        seen_in = {}     # type: Dict[str, str]
        seen_out = {}    # type: Dict[str, str]
        for node in self.nodes:
            for cid in node.downstream:
                if cid in seen_in:
                    raise TrafficFlowError(
                        "cell '%s' receives from both '%s' and '%s'."
                        % (cid, seen_in[cid], node.node_id))
                seen_in[cid] = node.node_id
            for cid in node.upstream:
                if cid in seen_out:
                    raise TrafficFlowError(
                        "cell '%s' sends to both '%s' and '%s'."
                        % (cid, seen_out[cid], node.node_id))
                seen_out[cid] = node.node_id

    def max_timestep(self, cfl: float = 0.9) -> float:
        """Largest CFL-safe time step across every cell."""
        best = float("inf")
        for cid in self.order:
            cell = self.cells[cid]
            best = min(best, cfl_timestep(cell.fd, cell.length_km, cfl))
        if not math.isfinite(best):
            raise TrafficFlowError("network has no cells.")
        return best

    # -- dynamics -----------------------------------------------------------
    def step(self, t: float, dt: float) -> Tuple[Dict[str, float], float, float]:
        """Advance one synchronous CTM step; returns (node flows, entered, exited)."""
        ddt = _positive(dt, "dt")
        demand = {cid: self.cells[cid].demand_veh(ddt) for cid in self.order}
        supply = {cid: self.cells[cid].supply_veh(ddt) for cid in self.order}
        inflow = {cid: 0.0 for cid in self.order}
        outflow = {cid: 0.0 for cid in self.order}
        flows = {}       # type: Dict[str, float]
        entered = []     # type: List[float]
        exited = []      # type: List[float]

        for node in self.nodes:
            kind = node.kind
            if kind == "source":
                dn = node.downstream[0]
                y = min(node.demand_fn(t) * ddt, supply[dn])
                y = max(0.0, y)
                inflow[dn] += y
                entered.append(y)
                flows[node.node_id] = y / ddt
            elif kind == "sink":
                up = node.upstream[0]
                y = min(demand[up], node.supply_fn(t) * ddt)
                y = max(0.0, y)
                outflow[up] += y
                exited.append(y)
                flows[node.node_id] = y / ddt
            elif kind == "link":
                up, dn = node.upstream[0], node.downstream[0]
                y = max(0.0, min(demand[up], supply[dn]))
                outflow[up] += y
                inflow[dn] += y
                flows[node.node_id] = y / ddt
            elif kind == "merge":
                dn = node.downstream[0]
                ds = [demand[u] for u in node.upstream]
                ys = merge_flows(ds, supply[dn], node.priorities)
                for u, y in zip(node.upstream, ys):
                    outflow[u] += y
                    inflow[dn] += y
                flows[node.node_id] = math.fsum(ys) / ddt
            else:                                     # diverge
                up = node.upstream[0]
                sup = [supply[d] for d in node.downstream]
                total, ys = diverge_flows(demand[up], sup, node.splits)
                outflow[up] += total
                for d, y in zip(node.downstream, ys):
                    inflow[d] += y
                flows[node.node_id] = total / ddt

        for cid in self.order:
            cell = self.cells[cid]
            cell.n = max(0.0, cell.n + inflow[cid] - outflow[cid])
        return flows, math.fsum(entered), math.fsum(exited)

    def simulate(self, dt: Optional[float] = None, n_steps: int = 1,
                 record_every: int = 1) -> CTMResult:
        """Run the network for ``n_steps`` steps of ``dt`` hours.

        ``dt`` defaults to the step the network was built with (see
        ``ctm_corridor``); pass it explicitly to override.

        **This stepper is stateful.**  ``CTMNetwork.step`` mutates ``cell.n``
        in place and ``simulate`` does NOT restore the constructed
        occupancies, so a second ``simulate`` on the same object continues
        from where the first one stopped while replaying the demand profile
        from ``t = 0``.  That is deliberate -- ``step`` is a public
        single-step API and ``cell.n`` is a public mutable attribute -- but it
        means a run is not repeatable from a reused network.  Rebuild the
        network (or reassign ``cell.n``) between independent runs.
        """
        self.validate()
        if dt is None and self.dt is None:
            raise TrafficFlowError("dt was not given and the network was not "
                                   "built with one.")
        ddt = _positive(self.dt if dt is None else dt, "dt")
        if int(n_steps) < 1:
            raise TrafficFlowError("n_steps must be >= 1.")
        n_steps = int(n_steps)
        every = max(1, int(record_every))
        safe = self.max_timestep(1.0)
        if ddt > safe * (1.0 + 1e-12):
            raise TrafficFlowError("dt=%g violates the CFL limit %g."
                                   % (ddt, safe))

        ids = list(self.order)
        times = [0.0]
        dens = [[self.cells[c].density for c in ids]]
        vehs = [[self.cells[c].n for c in ids]]
        spds = [[self.cells[c].speed for c in ids]]
        node_hist = []      # type: List[Dict[str, float]]
        in_terms = []       # type: List[float]
        out_terms = []      # type: List[float]
        in_cum = [0.0]
        out_cum = [0.0]
        stored = [math.fsum(vehs[0])]
        stored0 = stored[0]
        ttt_terms = []      # type: List[float]
        ttd_terms = []      # type: List[float]
        kmax = max(dens[0]) if dens[0] else 0.0
        v_free_max = max(self.cells[c].fd.v_free for c in ids)

        for step in range(n_steps):
            t = step * ddt
            ttt_terms.append(math.fsum(self.cells[c].n for c in ids) * ddt)
            ttd_terms.append(math.fsum(self.cells[c].n * self.cells[c].speed
                                       for c in ids) * ddt)
            flows, ent, exi = self.step(t, ddt)
            in_terms.append(ent)
            out_terms.append(exi)
            node_hist.append(flows)
            row = [self.cells[c].density for c in ids]
            kmax = max(kmax, max(row) if row else 0.0)
            if (step + 1) % every == 0 or step == n_steps - 1:
                times.append((step + 1) * ddt)
                dens.append(row)
                vehs.append([self.cells[c].n for c in ids])
                spds.append([self.cells[c].speed for c in ids])
                in_cum.append(math.fsum(in_terms))
                out_cum.append(math.fsum(out_terms))
                stored.append(math.fsum(vehs[-1]))

        ttt = math.fsum(ttt_terms)
        ttd = math.fsum(ttd_terms)
        delay = ttt - ttd / v_free_max
        err = abs((stored[-1] - stored0) - (in_cum[-1] - out_cum[-1]))
        return CTMResult(times, ids, dens, vehs, spds, node_hist, in_cum,
                         out_cum, stored, ddt, err, ttt, ttd, delay, kmax)


def ctm_corridor(fd: FundamentalDiagram, length_km: float, n_cells: int,
                 dt: float, demand: Any, lanes: float = 1.0,
                 exit_supply: Any = None, k_initial: float = 0.0,
                 bottleneck_cells: Optional[Sequence[int]] = None,
                 bottleneck_lanes: Optional[float] = None,
                 capacity_drop: float = 0.0) -> CTMNetwork:
    """Build a single-corridor CTM (optionally with a lane-drop bottleneck)."""
    if int(n_cells) < 1:
        raise TrafficFlowError("n_cells must be >= 1.")
    n_cells = int(n_cells)
    L = _positive(length_km, "length_km") / n_cells
    net = CTMNetwork(dt)
    bn = set(int(i) for i in (bottleneck_cells or []))
    for i in range(n_cells):
        lan = lanes
        if i in bn and bottleneck_lanes is not None:
            lan = bottleneck_lanes
        cell = net.add_cell("c%d" % i, L, fd, lan,
                            capacity_drop if i in bn else 0.0)
        cell.n = _nonnegative(k_initial, "k_initial") * cell.lane_km
    net.add_source("c0", demand)
    for i in range(n_cells - 1):
        net.connect("c%d" % i, "c%d" % (i + 1))
    net.add_sink("c%d" % (n_cells - 1), exit_supply)
    return net


# ===========================================================================
# 5. LINK TRANSMISSION MODEL (YPERMAN)
# ===========================================================================
@dataclass
class LTMLink:
    """One LTM link, described by the two Newell wave speeds and its storage."""

    link_id: str
    length_km: float
    lanes: float = 1.0
    v_free: float = 100.0
    w: float = 20.0
    k_jam: float = 180.0
    capacity: float = 2000.0

    def __post_init__(self) -> None:
        self.length_km = _positive(self.length_km, "length_km")
        self.lanes = _positive(self.lanes, "lanes")
        self.v_free = _positive(self.v_free, "v_free")
        self.w = _positive(self.w, "w")
        self.k_jam = _positive(self.k_jam, "k_jam")
        self.capacity = _positive(self.capacity, "capacity")
        tri = self.v_free * self.w * self.k_jam / (self.v_free + self.w)
        if self.capacity > tri * (1.0 + 1e-9):
            raise TrafficFlowError(
                "link '%s': capacity %g exceeds the triangular envelope %g."
                % (self.link_id, self.capacity, tri))

    @classmethod
    def from_fd(cls, link_id: str, length_km: float, fd: FundamentalDiagram,
                lanes: float = 1.0) -> "LTMLink":
        """Build a link from any fundamental diagram (uses its jam wave speed)."""
        w = abs(fd.wave_speed(fd.k_jam))
        if w <= _EPS:
            w = abs(fd.wave_speed(fd.k_jam * 0.999))
        if w <= _EPS:
            raise TrafficFlowError("diagram '%s' has no backward wave speed."
                                   % fd.name)
        return cls(link_id, length_km, lanes, fd.v_free, w, fd.k_jam,
                   fd.capacity)

    @property
    def free_flow_time(self) -> float:
        """Free-flow traversal time, hours."""
        return self.length_km / self.v_free

    @property
    def backward_time(self) -> float:
        """Backward (shockwave) traversal time, hours."""
        return self.length_km / self.w

    @property
    def storage(self) -> float:
        """Jam storage of the link, vehicles."""
        return self.k_jam * self.length_km * self.lanes


@dataclass
class LTMResult:
    """Cumulative-curve output of a Link Transmission Model run."""

    times: List[float]
    link_ids: List[str]
    upstream: List[List[float]]
    downstream: List[List[float]]
    vehicles: List[List[float]]
    dt: float
    inflow_cum: List[float]
    outflow_cum: List[float]
    conservation_error: float

    def curve(self, link_id: str, end: str = "downstream") -> "CumulativeCurve":
        """Cumulative curve at the upstream or downstream end of a link."""
        if link_id not in self.link_ids:
            raise TrafficFlowError("unknown link '%s'." % link_id)
        j = self.link_ids.index(link_id)
        src = self.upstream if end == "upstream" else self.downstream
        return CumulativeCurve(list(self.times), list(src[j]))

    def link_travel_time(self, link_id: str, t: float) -> float:
        """Experienced travel time of a vehicle entering ``link_id`` at time ``t``."""
        u = self.curve(link_id, "upstream")
        v = self.curve(link_id, "downstream")
        return travel_time_from_curves(u, v, t)

    def summary(self) -> Dict[str, Any]:
        """Headline numbers of the run."""
        return {
            "model": "ltm",
            "nLinks": len(self.link_ids),
            "nSteps": len(self.times) - 1,
            "dtH": self.dt,
            "vehiclesIn": self.inflow_cum[-1],
            "vehiclesOut": self.outflow_cum[-1],
            "vehiclesStored": math.fsum(row[-1] for row in self.vehicles),
            "conservationError": self.conservation_error,
        }


def _interp_uniform(values: Sequence[float], dt: float, t: float,
                    n_known: int) -> float:
    """Linear interpolation of a uniform-grid curve, clamped at both ends."""
    if t <= 0.0:
        return values[0]
    pos = t / dt
    i = int(math.floor(pos))
    if i >= n_known:
        return values[n_known]
    frac = pos - i
    if frac <= 0.0:
        return values[i]
    return values[i] + frac * (values[i + 1] - values[i])


def ltm_simulate(links: Sequence[LTMLink], dt: float, n_steps: int,
                 inflow: Any, exit_supply: Any = None) -> LTMResult:
    """Yperman's Link Transmission Model on a serial corridor of links."""
    lk = list(links)
    _check_len(lk, 1, "links")
    ddt = _positive(dt, "dt")
    if int(n_steps) < 1:
        raise TrafficFlowError("n_steps must be >= 1.")
    n_steps = int(n_steps)
    for link in lk:
        if ddt > link.free_flow_time * (1.0 + 1e-12):
            raise TrafficFlowError(
                "link '%s': dt=%g exceeds the free-flow time %g; refine dt."
                % (link.link_id, ddt, link.free_flow_time))
        if ddt > link.backward_time * (1.0 + 1e-12):
            raise TrafficFlowError(
                "link '%s': dt=%g exceeds the backward time %g; refine dt."
                % (link.link_id, ddt, link.backward_time))
    in_fn = _as_flow_fn(inflow, "inflow")
    out_fn = _as_flow_fn(lk[-1].capacity * lk[-1].lanes if exit_supply is None
                         else exit_supply, "exit_supply")

    m = len(lk)
    times = [i * ddt for i in range(n_steps + 1)]
    U = [[0.0] * (n_steps + 1) for _ in range(m)]
    V = [[0.0] * (n_steps + 1) for _ in range(m)]

    for step in range(n_steps):
        t = step * ddt
        send = [0.0] * m
        recv = [0.0] * m
        for i, link in enumerate(lk):
            cap_step = link.capacity * link.lanes * ddt
            u_lag = _interp_uniform(U[i], ddt, t + ddt - link.free_flow_time, step)
            send[i] = max(0.0, min(u_lag - V[i][step], cap_step))
            v_lag = _interp_uniform(V[i], ddt, t + ddt - link.backward_time, step)
            recv[i] = max(0.0, min(v_lag + link.storage - U[i][step], cap_step))
        transfers = [0.0] * (m + 1)
        transfers[0] = max(0.0, min(in_fn(t) * ddt, recv[0]))
        for i in range(1, m):
            transfers[i] = max(0.0, min(send[i - 1], recv[i]))
        transfers[m] = max(0.0, min(send[m - 1], out_fn(t) * ddt))
        for i in range(m):
            U[i][step + 1] = U[i][step] + transfers[i]
            V[i][step + 1] = V[i][step] + transfers[i + 1]

    vehicles = [[U[i][s] - V[i][s] for s in range(n_steps + 1)] for i in range(m)]
    in_cum = list(U[0])
    out_cum = list(V[m - 1])
    stored_end = math.fsum(vehicles[i][-1] for i in range(m))
    err = abs(in_cum[-1] - out_cum[-1] - stored_end)
    return LTMResult(times, [l.link_id for l in lk], U, V, vehicles, ddt,
                     in_cum, out_cum, err)


# ===========================================================================
# 6. CUMULATIVE COUNT CURVES & NEWELL'S THREE-DETECTOR PROBLEM
# ===========================================================================
@dataclass
class CumulativeCurve:
    """A non-decreasing cumulative count N(t) sampled on an arbitrary time grid."""

    times: List[float]
    counts: List[float]

    def __post_init__(self) -> None:
        self.times = _as_floats(self.times, "times")
        self.counts = _as_floats(self.counts, "counts")
        if len(self.times) != len(self.counts):
            raise TrafficFlowError("times and counts must have equal length.")
        _check_len(self.times, 2, "times")
        for i in range(1, len(self.times)):
            if self.times[i] <= self.times[i - 1]:
                raise TrafficFlowError("times must be strictly increasing.")

    def at(self, t: float) -> float:
        """Cumulative count at time ``t`` (linear interpolation, clamped)."""
        tt = _as_float(t, "t")
        if tt <= self.times[0]:
            return self.counts[0]
        if tt >= self.times[-1]:
            return self.counts[-1]
        lo, hi = 0, len(self.times) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.times[mid] <= tt:
                lo = mid
            else:
                hi = mid
        span = self.times[hi] - self.times[lo]
        frac = (tt - self.times[lo]) / span
        return self.counts[lo] + frac * (self.counts[hi] - self.counts[lo])

    def inverse(self, n: float) -> float:
        """First time at which the count reaches ``n`` (linear interpolation)."""
        nn = _as_float(n, "n")
        if nn <= self.counts[0]:
            return self.times[0]
        # Strict ">" on purpose: at nn == counts[-1] the loop below finds the
        # FIRST sample that reaches the value, which is what the docstring
        # promises.  With ">=" a terminal plateau returned the LAST recorded
        # time instead, so travel_time_from_curves grew with the length of the
        # recording window rather than staying a property of the traffic.
        if nn > self.counts[-1]:
            return self.times[-1]
        for i in range(1, len(self.counts)):
            if self.counts[i] >= nn:
                d = self.counts[i] - self.counts[i - 1]
                if d <= _EPS:
                    return self.times[i - 1]
                frac = (nn - self.counts[i - 1]) / d
                return self.times[i - 1] + frac * (self.times[i] - self.times[i - 1])
        return self.times[-1]

    def area(self, t0: Optional[float] = None, t1: Optional[float] = None
             ) -> float:
        """Trapezoidal area under the curve between ``t0`` and ``t1``, veh-hours."""
        a = self.times[0] if t0 is None else _as_float(t0, "t0")
        b = self.times[-1] if t1 is None else _as_float(t1, "t1")
        if b < a:
            raise TrafficFlowError("t1 must be >= t0.")
        pts = [a] + [t for t in self.times if a < t < b] + [b]
        terms = []
        for i in range(1, len(pts)):
            dt = pts[i] - pts[i - 1]
            terms.append(0.5 * dt * (self.at(pts[i - 1]) + self.at(pts[i])))
        return math.fsum(terms)

    def flows(self) -> List[float]:
        """Average flow in each interval, veh/h."""
        return [(self.counts[i] - self.counts[i - 1]) /
                (self.times[i] - self.times[i - 1])
                for i in range(1, len(self.times))]

    def shifted(self, dt: float, dn: float = 0.0) -> "CumulativeCurve":
        """Copy translated by ``dt`` in time and ``dn`` in count (Newell's shift)."""
        d = _as_float(dt, "dt")
        c = _as_float(dn, "dn")
        return CumulativeCurve([t + d for t in self.times],
                               [n + c for n in self.counts])


def cumulative_from_flows(times: Sequence[float], flows: Sequence[float],
                          n0: float = 0.0) -> CumulativeCurve:
    """Integrate a flow profile (veh/h on ``len(times)-1`` intervals) into N(t)."""
    t = _as_floats(times, "times")
    q = _as_floats(flows, "flows")
    _check_len(t, 2, "times")
    if len(q) not in (len(t), len(t) - 1):
        raise TrafficFlowError("flows must have len(times) or len(times)-1 entries.")
    counts = [_as_float(n0, "n0")]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt <= 0.0:
            raise TrafficFlowError("times must be strictly increasing.")
        rate = q[i - 1] if len(q) == len(t) - 1 else 0.5 * (q[i - 1] + q[i])
        counts.append(counts[-1] + rate * dt)
    return CumulativeCurve(t, counts)


def delay_between_curves(upstream: CumulativeCurve, downstream: CumulativeCurve,
                         t0: Optional[float] = None, t1: Optional[float] = None
                         ) -> float:
    """Total delay as the area between two cumulative curves, veh-hours."""
    a = max(upstream.times[0], downstream.times[0]) if t0 is None \
        else _as_float(t0, "t0")
    b = min(upstream.times[-1], downstream.times[-1]) if t1 is None \
        else _as_float(t1, "t1")
    if b < a:
        raise TrafficFlowError("the curves do not overlap in time.")
    knots = sorted(set([a, b] + [t for t in upstream.times if a < t < b] +
                       [t for t in downstream.times if a < t < b]))
    terms = []
    for i in range(1, len(knots)):
        dt = knots[i] - knots[i - 1]
        g0 = upstream.at(knots[i - 1]) - downstream.at(knots[i - 1])
        g1 = upstream.at(knots[i]) - downstream.at(knots[i])
        terms.append(0.5 * dt * (g0 + g1))
    return math.fsum(terms)


def queue_from_curves(upstream: CumulativeCurve, downstream: CumulativeCurve,
                      t: float) -> float:
    """Vertical distance between the curves: vehicles queued at time ``t``."""
    return upstream.at(t) - downstream.at(t)


def travel_time_from_curves(upstream: CumulativeCurve,
                            downstream: CumulativeCurve, t: float) -> float:
    """Horizontal distance between the curves: FIFO travel time of the entrant at ``t``."""
    n = upstream.at(t)
    return downstream.inverse(n) - _as_float(t, "t")


def newell_three_detector(upstream: CumulativeCurve,
                          downstream: CumulativeCurve, x: float,
                          length_km: float, v_free: float, w: float,
                          k_jam: float, times: Sequence[float],
                          lanes: float = 1.0) -> CumulativeCurve:
    """Newell's solution N(x, t) = min(free-flow wave, congested wave) at ``x``."""
    xx = _nonnegative(x, "x")
    L = _positive(length_km, "length_km")
    if xx > L:
        raise TrafficFlowError("x=%g lies beyond the link length %g." % (xx, L))
    vf = _positive(v_free, "v_free")
    ww = _positive(w, "w")
    kj = _positive(k_jam, "k_jam")
    lam = _positive(lanes, "lanes")
    grid = _as_floats(times, "times")
    _check_len(grid, 2, "times")
    stored = kj * (L - xx) * lam
    out = []
    for t in grid:
        free = upstream.at(t - xx / vf)
        cong = downstream.at(t - (L - xx) / ww) + stored
        out.append(min(free, cong))
    return CumulativeCurve(grid, out)


# ===========================================================================
# 7. POINT (VERTICAL) AND PHYSICAL (HORIZONTAL) QUEUES
# ===========================================================================
@dataclass
class QueueResult:
    """Output of a deterministic queueing run."""

    times: List[float]
    arrivals: List[float]
    departures: List[float]
    queue_veh: List[float]
    queue_km: Optional[List[float]]
    rejected: List[float]
    dt: float
    total_delay: float
    max_queue_veh: float
    max_queue_km: float
    spillback: bool
    spillback_time: Optional[float]
    conservation_error: float

    def summary(self) -> Dict[str, Any]:
        """Headline numbers of the run."""
        return {
            "model": "queue",
            "nSteps": len(self.times) - 1,
            "dtH": self.dt,
            "totalDelayVehH": self.total_delay,
            "maxQueueVeh": self.max_queue_veh,
            "maxQueueKm": self.max_queue_km,
            "spillback": self.spillback,
            "spillbackTimeH": self.spillback_time,
            "vehiclesRejected": math.fsum(self.rejected),
            "conservationError": self.conservation_error,
        }


def vertical_queue(arrivals: Sequence[float], capacity: Any, dt: float,
                   q0: float = 0.0) -> QueueResult:
    """Point-queue model: vehicles stack at a point with no physical extent."""
    a = _as_floats(arrivals, "arrivals")
    _check_len(a, 1, "arrivals")
    for v in a:
        if v < 0.0:
            raise TrafficFlowError("arrival flows must be >= 0.")
    ddt = _positive(dt, "dt")
    cap_fn = _as_flow_fn(capacity, "capacity")
    q = _nonnegative(q0, "q0")
    times = [0.0]
    queue = [q]
    deps = []      # type: List[float]
    arrs = []      # type: List[float]
    rej = []       # type: List[float]
    delay_terms = []      # type: List[float]
    qmax = q
    for i, rate in enumerate(a):
        t = i * ddt
        cap = max(0.0, cap_fn(t))
        served = min(cap, rate + q / ddt)
        delay_terms.append(q * ddt)
        q = max(0.0, q + (rate - served) * ddt)
        qmax = max(qmax, q)
        arrs.append(rate)
        deps.append(served)
        rej.append(0.0)
        times.append(t + ddt)
        queue.append(q)
    total_in = math.fsum(v * ddt for v in arrs)
    total_out = math.fsum(v * ddt for v in deps)
    err = abs((queue[-1] - queue[0]) - (total_in - total_out))
    return QueueResult(times, arrs, deps, queue, None, rej, ddt,
                       math.fsum(delay_terms), qmax, 0.0, False, None, err)


def horizontal_queue(arrivals: Sequence[float], capacity: Any, dt: float,
                     length_km: float, lanes: float, k_jam: float,
                     k_arrival: float, q0: float = 0.0) -> QueueResult:
    """Physical-queue model: the queue occupies road space and can spill back."""
    a = _as_floats(arrivals, "arrivals")
    _check_len(a, 1, "arrivals")
    for v in a:
        if v < 0.0:
            raise TrafficFlowError("arrival flows must be >= 0.")
    ddt = _positive(dt, "dt")
    L = _positive(length_km, "length_km")
    lam = _positive(lanes, "lanes")
    kj = _positive(k_jam, "k_jam")
    ka = _nonnegative(k_arrival, "k_arrival")
    if ka >= kj:
        raise TrafficFlowError("k_arrival %g must be below k_jam %g." % (ka, kj))
    cap_fn = _as_flow_fn(capacity, "capacity")
    per_km = (kj - ka) * lam            # extra vehicles stored per km of queue
    storage = per_km * L
    q = _nonnegative(q0, "q0")
    times = [0.0]
    queue = [q]
    qkm = [q / per_km]
    arrs = []      # type: List[float]
    deps = []      # type: List[float]
    rej = []       # type: List[float]
    delay_terms = []      # type: List[float]
    qmax = q
    spill = False
    spill_t = None      # type: Optional[float]
    for i, rate in enumerate(a):
        t = i * ddt
        cap = max(0.0, cap_fn(t))
        served = min(cap, rate + q / ddt)
        delay_terms.append(q * ddt)
        room = max(0.0, storage - q)
        net = (rate - served) * ddt
        if net > 0.0:
            admitted = min(net, room)
            # ``admitted`` is a min() clamp, so when net and room are
            # mathematically equal the leftover is pure round-off in vehicles.
            # Dividing that ulp-level residue by ddt inflates it into a flow
            # rate whose noise floor grows like 1/ddt, and an ABSOLUTE test on
            # the rate then made the spillback verdict depend on the reporting
            # interval.  Compare in VEHICLES against a storage-relative
            # tolerance instead, and absorb the residue so the reported
            # rejections stay consistent with the flag.
            if net - admitted <= 1e-12 * max(storage, net):
                admitted = net
            blocked = (net - admitted) / ddt
        else:
            admitted = net                      # the queue is draining
            blocked = 0.0
        if blocked > 0.0 and not spill:
            spill = True
            spill_t = t
        q = max(0.0, q + admitted)
        qmax = max(qmax, q)
        arrs.append(rate)
        deps.append(served)
        rej.append(blocked * ddt)
        times.append(t + ddt)
        queue.append(q)
        qkm.append(q / per_km)
    total_in = math.fsum(v * ddt for v in arrs) - math.fsum(rej)
    total_out = math.fsum(v * ddt for v in deps)
    err = abs((queue[-1] - queue[0]) - (total_in - total_out))
    return QueueResult(times, arrs, deps, queue, qkm, rej, ddt,
                       math.fsum(delay_terms), qmax, qmax / per_km,
                       spill, spill_t, err)


# ===========================================================================
# 8. SIGNAL DELAY MODELS
# ===========================================================================
def deterministic_delay(flow: float, saturation_flow: float, cycle_s: float,
                        green_s: float) -> float:
    """Webster's uniform (deterministic) delay term, seconds per vehicle."""
    q = _nonnegative(flow, "flow")
    s = _positive(saturation_flow, "saturation_flow")
    C = _positive(cycle_s, "cycle_s")
    g = _positive(green_s, "green_s")
    if g > C:
        raise TrafficFlowError("green_s %g exceeds cycle_s %g." % (g, C))
    lam = g / C
    cap = s * lam
    x = min(1.0, q / cap) if cap > _EPS else 1.0
    denom = 1.0 - lam * x
    if denom <= _EPS:
        raise TrafficFlowError("degenerate signal: lambda*x = 1.")
    return C * (1.0 - lam) ** 2 / (2.0 * denom)


def webster_delay(flow: float, saturation_flow: float, cycle_s: float,
                  green_s: float) -> float:
    """Webster's (1958) average delay: uniform + stochastic + empirical correction."""
    q = _positive(flow, "flow")
    s = _positive(saturation_flow, "saturation_flow")
    C = _positive(cycle_s, "cycle_s")
    g = _positive(green_s, "green_s")
    if g > C:
        raise TrafficFlowError("green_s %g exceeds cycle_s %g." % (g, C))
    lam = g / C
    cap = s * lam
    x = q / cap
    if x >= 1.0:
        raise TrafficFlowError(
            "Webster's formula diverges at saturation (x=%.3f >= 1); use "
            "akcelik_delay for the oversaturated regime." % x)
    qs = q / 3600.0
    d1 = C * (1.0 - lam) ** 2 / (2.0 * (1.0 - lam * x))
    d2 = x * x / (2.0 * qs * (1.0 - x))
    d3 = -0.65 * (C / (qs * qs)) ** (1.0 / 3.0) * x ** (2.0 + 5.0 * lam)
    return d1 + d2 + d3


def akcelik_delay(flow: float, capacity: float, period_h: float,
                  saturation_flow: float, green_s: float) -> float:
    """Akcelik's time-dependent overflow delay, seconds per vehicle."""
    q = _nonnegative(flow, "flow")
    c = _positive(capacity, "capacity")
    T = _positive(period_h, "period_h")
    s = _positive(saturation_flow, "saturation_flow")
    g = _positive(green_s, "green_s")
    x = q / c
    x0 = 0.67 + (s / 3600.0) * g / 600.0
    if x <= x0:
        return 0.0
    inner = (x - 1.0) ** 2 + 12.0 * (x - x0) / (c * T)
    return 900.0 * T * ((x - 1.0) + math.sqrt(max(0.0, inner)))


def webster_optimal_cycle(lost_time_s: float,
                          critical_flow_ratios: Sequence[float]) -> float:
    """Webster's optimal cycle length C0 = (1.5 L + 5) / (1 - Y), seconds."""
    L = _nonnegative(lost_time_s, "lost_time_s")
    y = _as_floats(critical_flow_ratios, "critical_flow_ratios")
    _check_len(y, 1, "critical_flow_ratios")
    for v in y:
        if v < 0.0:
            raise TrafficFlowError("flow ratios must be >= 0.")
    Y = math.fsum(y)
    if Y >= 1.0:
        raise TrafficFlowError("the intersection is oversaturated (Y=%.3f >= 1); "
                               "Webster's optimum does not exist." % Y)
    return (1.5 * L + 5.0) / (1.0 - Y)


@dataclass
class DelayBreakdown:
    """Delay decomposed into its uniform, random and overflow components."""

    uniform_s: float
    random_s: float
    overflow_s: float
    total_s: float
    degree_of_saturation: float
    capacity: float
    level_of_service: str

    def summary(self) -> Dict[str, Any]:
        """The breakdown as a plain dict."""
        return {
            "uniformDelayS": self.uniform_s,
            "randomDelayS": self.random_s,
            "overflowDelayS": self.overflow_s,
            "totalDelayS": self.total_s,
            "degreeOfSaturation": self.degree_of_saturation,
            "capacityVehH": self.capacity,
            "levelOfService": self.level_of_service,
        }


def _level_of_service(delay_s: float) -> str:
    """HCM signalised-intersection level of service from control delay."""
    for bound, grade in ((10.0, "A"), (20.0, "B"), (35.0, "C"),
                         (55.0, "D"), (80.0, "E")):
        if delay_s <= bound:
            return grade
    return "F"


def signal_delay(flow: float, saturation_flow: float, cycle_s: float,
                 green_s: float, period_h: float = 0.25) -> DelayBreakdown:
    """Full control-delay breakdown that stays finite through oversaturation."""
    q = _nonnegative(flow, "flow")
    s = _positive(saturation_flow, "saturation_flow")
    C = _positive(cycle_s, "cycle_s")
    g = _positive(green_s, "green_s")
    T = _positive(period_h, "period_h")
    cap = s * (g / C)
    x = q / cap if cap > _EPS else float("inf")
    d1 = deterministic_delay(q, s, C, g)
    if x < 0.95 and q > _EPS:
        d2 = webster_delay(q, s, C, g) - d1
    else:
        d2 = 0.0
    d3 = akcelik_delay(q, cap, T, s, g)
    total = d1 + d2 + d3
    return DelayBreakdown(d1, d2, d3, total, x, cap, _level_of_service(total))


# ===========================================================================
# 9. SHOCKWAVE QUEUE ANALYSIS
# ===========================================================================
@dataclass
class ShockwaveResult:
    """Queue growth and dissipation at a signal, from three kinematic waves."""

    stopping_wave_speed: float
    starting_wave_speed: float
    recovery_wave_speed: float
    max_queue_km: float
    max_queue_veh: float
    time_of_max_queue_h: float
    clearance_time_h: float
    queue_clears: bool
    k_arrival: float
    k_jam: float
    k_critical: float

    def summary(self) -> Dict[str, Any]:
        """Headline numbers of the analysis."""
        return {
            "stoppingWaveKmh": self.stopping_wave_speed,
            "startingWaveKmh": self.starting_wave_speed,
            "recoveryWaveKmh": self.recovery_wave_speed,
            "maxQueueKm": self.max_queue_km,
            "maxQueueVeh": self.max_queue_veh,
            "timeOfMaxQueueS": self.time_of_max_queue_h * 3600.0,
            "clearanceTimeS": self.clearance_time_h * 3600.0,
            "queueClears": self.queue_clears,
        }


def shockwave_queue(fd: FundamentalDiagram, arrival_flow: float,
                    red_s: float, green_s: float, lanes: float = 1.0
                    ) -> ShockwaveResult:
    """Classic three-wave queue analysis for one red-green cycle at a stop line."""
    q_a = _nonnegative(arrival_flow, "arrival_flow")
    lam = _positive(lanes, "lanes")
    q_lane = q_a / lam
    if q_lane > fd.capacity * (1.0 + 1e-9):
        raise TrafficFlowError("arrival flow %g veh/h exceeds the %g veh/h "
                               "capacity of %g lanes." % (q_a, fd.capacity * lam, lam))
    r = seconds(_positive(red_s, "red_s"))
    g = seconds(_positive(green_s, "green_s"))
    k_a = fd.density_from_flow(min(q_lane, fd.capacity), congested=False)
    k_j = fd.k_jam
    k_c = fd.k_critical
    q_cap = fd.capacity

    w_stop = shock_speed(k_a, q_lane, k_j, 0.0)          # arrival -> jam
    w_start = shock_speed(k_j, 0.0, k_c, q_cap)          # jam -> discharge
    if k_c == k_a:
        w_rec = fd.wave_speed(k_c)
    else:
        w_rec = shock_speed(k_c, q_cap, k_a, q_lane)     # discharge -> arrival

    denom = w_start - w_stop
    if denom >= -_EPS:
        raise TrafficFlowError("the starting wave never catches the stopping "
                               "wave; check the diagram.")
    t_max = w_start * r / denom
    x_max = abs(w_stop * t_max)
    veh = x_max * (k_j - k_a) * lam
    clears = (t_max - r) <= g + _EPS
    if abs(w_rec) <= _EPS:
        t_clear = float("inf")
        clears = False
    else:
        t_clear = t_max + x_max / abs(w_rec) if w_rec > 0.0 else float("inf")
    return ShockwaveResult(w_stop, w_start, w_rec, x_max, veh, t_max, t_clear,
                           clears, k_a, k_j, k_c)


# ===========================================================================
# 10. STORE-AND-FORWARD LINK MODEL
# ===========================================================================
@dataclass
class SFLink:
    """One store-and-forward link: a queue served at a green-proportional rate."""

    link_id: str
    saturation_flow: float
    storage: float
    x0: float = 0.0

    def __post_init__(self) -> None:
        self.saturation_flow = _positive(self.saturation_flow, "saturation_flow")
        self.storage = _positive(self.storage, "storage")
        self.x0 = _nonnegative(self.x0, "x0")


@dataclass
class StoreAndForwardResult:
    """Output of a store-and-forward run."""

    times: List[float]
    link_ids: List[str]
    queues: List[List[float]]
    outflows: List[List[float]]
    rejected: List[List[float]]
    dt: float
    total_delay: float
    max_queue: float
    clamped_steps: int
    conservation_error: float

    def summary(self) -> Dict[str, Any]:
        """Headline numbers of the run."""
        return {
            "model": "store_and_forward",
            "nLinks": len(self.link_ids),
            "nSteps": len(self.times) - 1,
            "totalDelayVehH": self.total_delay,
            "maxQueueVeh": self.max_queue,
            "clampedSteps": self.clamped_steps,
            "isLinear": self.clamped_steps == 0,
            "conservationError": self.conservation_error,
        }


class StoreAndForwardModel:
    """Linear (MPC-ready) store-and-forward model of a signalised network."""

    def __init__(self, links: Sequence[SFLink],
                 turn_ratios: Optional[Dict[Tuple[str, str], float]] = None,
                 cycle_s: float = 90.0) -> None:
        self.links = list(links)
        _check_len(self.links, 1, "links")
        self.ids = [l.link_id for l in self.links]
        if len(set(self.ids)) != len(self.ids):
            raise TrafficFlowError("link ids must be unique.")
        self.index = {lid: i for i, lid in enumerate(self.ids)}
        self.cycle_s = _positive(cycle_s, "cycle_s")
        self.turn_ratios = {}    # type: Dict[Tuple[str, str], float]
        for key, val in (turn_ratios or {}).items():
            up, dn = key
            if up not in self.index or dn not in self.index:
                raise TrafficFlowError("turn ratio references unknown link(s) "
                                       "'%s'->'%s'." % (up, dn))
            self.turn_ratios[(up, dn)] = _fraction(val, "turn ratio")
        for lid in self.ids:
            tot = math.fsum(v for (u, _), v in self.turn_ratios.items() if u == lid)
            if tot > 1.0 + 1e-9:
                raise TrafficFlowError("turn ratios out of '%s' sum to %.3f > 1."
                                       % (lid, tot))

    def outflow_capacity(self, link_id: str, green_s: float) -> float:
        """Green-proportional discharge capacity of a link, veh/h."""
        i = self.index[link_id]
        g = _nonnegative(green_s, "green_s")
        if g > self.cycle_s:
            raise TrafficFlowError("green %g exceeds the cycle %g."
                                   % (g, self.cycle_s))
        return self.links[i].saturation_flow * g / self.cycle_s

    def simulate(self, greens: Any, demands: Any, dt: float, n_steps: int
                 ) -> StoreAndForwardResult:
        """Run the model with green-split and demand profiles (dicts or callables)."""
        ddt = _positive(dt, "dt")
        if int(n_steps) < 1:
            raise TrafficFlowError("n_steps must be >= 1.")
        n_steps = int(n_steps)
        green_fn = greens if callable(greens) else (lambda t: greens)
        demand_fn = demands if callable(demands) else (lambda t: demands)
        x = [l.x0 for l in self.links]
        times = [0.0]
        hist = [list(x)]
        outs = []       # type: List[List[float]]
        rejs = []       # type: List[List[float]]
        delay_terms = []      # type: List[float]
        entered = []          # type: List[float]
        left = []             # type: List[float]
        clamped = 0
        xmax = max(x) if x else 0.0

        for step in range(n_steps):
            t = step * ddt
            gmap = green_fn(t)
            dmap = demand_fn(t)
            delay_terms.append(math.fsum(x) * ddt)
            out = []
            for i, link in enumerate(self.links):
                g = float(gmap.get(link.link_id, 0.0)) if hasattr(gmap, "get") \
                    else float(gmap)
                cap = self.outflow_capacity(link.link_id, g) * ddt
                y = min(cap, x[i])
                if y < cap - _EPS:
                    clamped += 1
                out.append(y)
            inflow = [0.0] * len(self.links)
            ext = []
            for i, link in enumerate(self.links):
                d = float(dmap.get(link.link_id, 0.0)) if hasattr(dmap, "get") \
                    else float(dmap)
                ext.append(max(0.0, d) * ddt)
            for (up, dn), ratio in self.turn_ratios.items():
                inflow[self.index[dn]] += ratio * out[self.index[up]]
            rej = [0.0] * len(self.links)
            exit_net = []
            for i, link in enumerate(self.links):
                ratio_sum = math.fsum(v for (u, _), v in self.turn_ratios.items()
                                      if u == self.ids[i])
                exit_net.append(out[i] * (1.0 - ratio_sum))
                arrive = inflow[i] + ext[i]
                room = max(0.0, link.storage - (x[i] - out[i]))
                admitted = min(arrive, room)
                rej[i] = arrive - admitted
                if rej[i] > _EPS:
                    clamped += 1
                x[i] = max(0.0, x[i] - out[i] + admitted)
            entered.append(math.fsum(ext) - math.fsum(rej))
            left.append(math.fsum(exit_net))
            outs.append(out)
            rejs.append(rej)
            times.append(t + ddt)
            hist.append(list(x))
            xmax = max(xmax, max(x) if x else 0.0)

        stored0 = math.fsum(hist[0])
        stored1 = math.fsum(hist[-1])
        err = abs((stored1 - stored0) - (math.fsum(entered) - math.fsum(left)))
        return StoreAndForwardResult(times, list(self.ids), hist, outs, rejs,
                                     ddt, math.fsum(delay_terms), xmax,
                                     clamped, err)


# ===========================================================================
# 11. MACROSCOPIC FUNDAMENTAL DIAGRAM
# ===========================================================================
def _solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve a dense linear system by Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) <= 1e-14:
            raise TrafficFlowError("singular system in the least-squares fit; "
                                   "reduce the polynomial degree.")
        m[col], m[piv] = m[piv], m[col]
        inv = 1.0 / m[col][col]
        for r in range(col + 1, n):
            f = m[r][col] * inv
            if f == 0.0:
                continue
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = math.fsum(m[r][c] * x[c] for c in range(r + 1, n))
        x[r] = (m[r][n] - s) / m[r][r]
    return x


def _polyfit(xs: Sequence[float], ys: Sequence[float], degree: int,
             through_origin: bool = True) -> List[float]:
    """Least-squares polynomial coefficients (ascending powers) via normal equations."""
    x = _as_floats(xs, "xs")
    y = _as_floats(ys, "ys")
    if len(x) != len(y):
        raise TrafficFlowError("xs and ys must have equal length.")
    d = int(degree)
    if d < 1:
        raise TrafficFlowError("degree must be >= 1.")
    powers = list(range(1, d + 1)) if through_origin else list(range(0, d + 1))
    if len(x) < len(powers):
        raise TrafficFlowError("need at least %d points for degree %d."
                               % (len(powers), d))
    n = len(powers)
    # Fit in a NORMALISED variable u = x / s.  The normal-equation entries are
    # sums of x**(p+c), so their magnitude scales like s**(2*degree): a fit over
    # k in [0, 1e-4] produces entries around 1e-15 even though the system is
    # perfectly well conditioned (cond ~ 1e9).  The absolute pivot test inside
    # _solve_linear then rejected it as "singular", while a fit over large x
    # sailed past the same test with a genuinely rank-deficient design.
    # Normalising puts every entry at O(1) so the pivot test measures
    # conditioning rather than units; the coefficients are unscaled afterwards
    # via c_p = sol_p / s**p, which makes the fit exactly scale-invariant.
    s = max(abs(xi) for xi in x)
    if s <= 0.0:
        raise TrafficFlowError("xs must contain a non-zero value.")
    u = [xi / s for xi in x]
    a = [[math.fsum(ui ** (powers[r] + powers[c]) for ui in u)
          for c in range(n)] for r in range(n)]
    b = [math.fsum(y[i] * u[i] ** powers[r] for i in range(len(u)))
         for r in range(n)]
    sol = _solve_linear(a, b)
    coeffs = [0.0] * (d + 1)
    for idx, p in enumerate(powers):
        coeffs[p] = sol[idx] / (s ** p)
    return coeffs


def _clip_segment(t0: float, x0: float, t1: float, x1: float,
                  ta: float, tb: float, xa: float, xb: float
                  ) -> Optional[Tuple[float, float]]:
    """Liang-Barsky clip of a space-time segment to a rectangle; returns (u_lo, u_hi)."""
    dt = t1 - t0
    dx = x1 - x0
    u_lo, u_hi = 0.0, 1.0
    for p, q in ((-dt, t0 - ta), (dt, tb - t0), (-dx, x0 - xa), (dx, xb - x0)):
        if abs(p) <= _TINY:
            if q < 0.0:
                return None
            continue
        r = q / p
        if p < 0.0:
            if r > u_hi:
                return None
            if r > u_lo:
                u_lo = r
        else:
            if r < u_lo:
                return None
            if r < u_hi:
                u_hi = r
    if u_hi <= u_lo:
        return None
    return u_lo, u_hi


def edie_flow_density(trajectories: Sequence[Trajectory], t0: float, t1: float,
                      x0: float, x1: float) -> Dict[str, float]:
    """Edie's generalised flow, density and speed over a space-time rectangle."""
    ta, tb = _as_float(t0, "t0"), _as_float(t1, "t1")
    xa, xb = _as_float(x0, "x0"), _as_float(x1, "x1")
    if tb <= ta:
        raise TrafficFlowError("t1 must exceed t0.")
    if xb <= xa:
        raise TrafficFlowError("x1 must exceed x0.")
    area = (tb - ta) * (xb - xa)
    dist_terms = []      # type: List[float]
    time_terms = []      # type: List[float]
    for traj in trajectories:
        pts = [(float(t), float(x)) for t, x in traj]
        for i in range(1, len(pts)):
            pt0, px0 = pts[i - 1]
            pt1, px1 = pts[i]
            if pt1 <= pt0:
                continue
            clip = _clip_segment(pt0, px0, pt1, px1, ta, tb, xa, xb)
            if clip is None:
                continue
            u_lo, u_hi = clip
            frac = u_hi - u_lo
            time_terms.append(frac * (pt1 - pt0))
            dist_terms.append(frac * (px1 - px0))
    ttd = math.fsum(dist_terms)
    tts = math.fsum(time_terms)
    q = ttd / area
    k = tts / area
    v = ttd / tts if tts > _EPS else 0.0
    return {"flow": q, "density": k, "speed": v,
            "totalTravelDistance": ttd, "totalTravelTime": tts, "area": area}


def aggregate_network_state(records: Sequence[Dict[str, float]]
                            ) -> Dict[str, float]:
    """Lane-km weighted network flow, density, speed and production from link data."""
    rows = list(records)
    _check_len(rows, 1, "records")
    wsum = []       # type: List[float]
    qsum = []       # type: List[float]
    ksum = []       # type: List[float]
    for r in rows:
        L = _positive(r["length_km"], "length_km")
        lanes = _positive(r.get("lanes", 1.0), "lanes")
        q = _nonnegative(r["flow"], "flow")
        k = _nonnegative(r["density"], "density")
        w = L * lanes
        wsum.append(w)
        qsum.append(q * w)
        ksum.append(k * w)
    total_w = math.fsum(wsum)
    if total_w <= _EPS:
        raise TrafficFlowError("the network has no lane-kilometres.")
    q = math.fsum(qsum) / total_w
    k = math.fsum(ksum) / total_w
    return {"flow": q, "density": k, "speed": q / k if k > _EPS else 0.0,
            "laneKm": total_w, "production": q * total_w,
            "accumulation": k * total_w}


class MFDModel:
    """A fitted Macroscopic Fundamental Diagram: network production P(k)."""

    def __init__(self, coeffs: Sequence[float], k_max: float,
                 lane_km: float = 1.0, name: str = "mfd") -> None:
        self.coeffs = _as_floats(coeffs, "coeffs")
        _check_len(self.coeffs, 2, "coeffs")
        # A production curve MUST pass through the origin -- an empty network
        # carries no flow -- and every builder in this module (``fit_mfd``)
        # produces exactly that.  The constructor used to accept any
        # polynomial, which made ``speed(0)`` (the P(k)/k limit) diverge:
        # MFDModel([500, 60, -0.3], 200).speed(0) returned 5e14 km/h, and
        # mfd_accumulation then took trip/free_speed ~ 0 as the free-flow
        # travel time and reported raw vehicle-hours as delay.  Reject the
        # curve here rather than leaving the precondition unstated.  The
        # tolerance is relative to the coefficient scale so the check is
        # unit-independent.
        scale = max(abs(c) for c in self.coeffs)
        if abs(self.coeffs[0]) > 1e-9 * scale:
            raise TrafficFlowError(
                "an MFD must pass through the origin: coeffs[0]=%g is not "
                "negligible beside the coefficient scale %g."
                % (self.coeffs[0], scale))
        self.coeffs[0] = 0.0
        self.k_max = _positive(k_max, "k_max")
        self.lane_km = _positive(lane_km, "lane_km")
        self.name = str(name)
        self._cap = None     # type: Optional[Tuple[float, float]]

    def production(self, k: float) -> float:
        """Network production (weighted flow) at network density ``k``, veh/h/lane."""
        kk = _clip(_nonnegative(k, "k"), 0.0, self.k_max)
        acc = 0.0
        for c in reversed(self.coeffs):
            acc = acc * kk + c
        return max(0.0, acc)

    def speed(self, k: float) -> float:
        """Network space-mean speed at density ``k``, km/h.

        At ``k = 0`` the analytic limit of P(k)/k is P'(0) = ``coeffs[1]``; the
        constructor guarantees P(0) = 0, so the limit exists.  (It used to be
        evaluated as ``production(_EPS) / _EPS``, which is the same number to
        within round-off for a through-origin curve but blew up to ~1e14 for
        one that was not.)
        """
        kk = _clip(_nonnegative(k, "k"), 0.0, self.k_max)
        if kk > _EPS:
            return self.production(kk) / kk
        return max(0.0, self.coeffs[1])

    def outflow(self, n: float, trip_length_km: float) -> float:
        """Network trip-completion rate for accumulation ``n`` vehicles, veh/h."""
        k = _nonnegative(n, "n") / self.lane_km
        return self.production(k) * self.lane_km / _positive(trip_length_km,
                                                             "trip_length_km")

    def _cap_pair(self) -> Tuple[float, float]:
        if self._cap is None:
            k, q = _argmax_unimodal(self.production, 0.0, self.k_max)
            self._cap = (k, q)
        return self._cap

    @property
    def critical_density(self) -> float:
        """Density maximising network production, veh/km/lane."""
        return self._cap_pair()[0]

    @property
    def capacity(self) -> float:
        """Maximum network production, veh/h/lane."""
        return self._cap_pair()[1]

    @property
    def critical_accumulation(self) -> float:
        """Vehicle accumulation at maximum production."""
        return self.critical_density * self.lane_km

    def summary(self) -> Dict[str, Any]:
        """Headline numbers of the fitted diagram."""
        return {
            "name": self.name,
            "coefficients": list(self.coeffs),
            "kMax": self.k_max,
            "laneKm": self.lane_km,
            "capacity": self.capacity,
            "criticalDensity": self.critical_density,
            "criticalAccumulation": self.critical_accumulation,
        }


def fit_mfd(densities: Sequence[float], flows: Sequence[float],
            degree: int = 3, lane_km: float = 1.0,
            k_max: Optional[float] = None) -> MFDModel:
    """Fit a through-origin polynomial MFD to observed (density, flow) pairs."""
    k = _as_floats(densities, "densities")
    q = _as_floats(flows, "flows")
    if len(k) != len(q):
        raise TrafficFlowError("densities and flows must have equal length.")
    _check_len(k, 3, "densities")
    coeffs = _polyfit(k, q, degree, through_origin=True)
    kmax = max(k) * 1.05 if k_max is None else _positive(k_max, "k_max")
    return MFDModel(coeffs, kmax, lane_km, "mfd_deg%d" % int(degree))


@dataclass
class MFDAccumulationResult:
    """Output of an accumulation-based (reservoir) network simulation."""

    times: List[float]
    accumulation: List[float]
    inflow: List[float]
    outflow: List[float]
    dt: float
    gridlocked: bool
    gridlock_time: Optional[float]
    max_accumulation: float
    total_delay: float
    conservation_error: float

    def summary(self) -> Dict[str, Any]:
        """Headline numbers of the run."""
        return {
            "model": "mfd_reservoir",
            "nSteps": len(self.times) - 1,
            "maxAccumulation": self.max_accumulation,
            "finalAccumulation": self.accumulation[-1],
            "gridlocked": self.gridlocked,
            "gridlockTimeH": self.gridlock_time,
            "totalDelayVehH": self.total_delay,
            "conservationError": self.conservation_error,
        }


def mfd_accumulation(mfd: MFDModel, inflow: Any, trip_length_km: float,
                     dt: float, n_steps: int, n0: float = 0.0
                     ) -> MFDAccumulationResult:
    """Simulate a single-reservoir network: dn/dt = inflow - P(n)/trip length."""
    ddt = _positive(dt, "dt")
    if int(n_steps) < 1:
        raise TrafficFlowError("n_steps must be >= 1.")
    n_steps = int(n_steps)
    in_fn = _as_flow_fn(inflow, "inflow")
    trip = _positive(trip_length_km, "trip_length_km")
    n = _nonnegative(n0, "n0")
    n_max = mfd.k_max * mfd.lane_km
    times = [0.0]
    acc = [n]
    q_in = []      # type: List[float]
    q_out = []     # type: List[float]
    free_speed = mfd.speed(0.0)          # = P'(0), the free-flow speed
    gridlock = False
    gridlock_t = None     # type: Optional[float]
    delay_terms = []      # type: List[float]
    nmax = n

    for step in range(n_steps):
        t = step * ddt
        out = mfd.outflow(n, trip)
        room = max(0.0, n_max - n)
        want = max(0.0, in_fn(t)) * ddt
        served = min(out * ddt, n)
        admitted = min(want, room + served)
        free_time = trip / free_speed if free_speed > _EPS else 0.0
        delay_terms.append(n * ddt - served * free_time)
        n = max(0.0, n + admitted - served)
        nmax = max(nmax, n)
        if not gridlock and n >= n_max * 0.999:
            gridlock = True
            gridlock_t = t
        q_in.append(admitted / ddt)
        q_out.append(served / ddt)
        times.append(t + ddt)
        acc.append(n)

    total_in = math.fsum(v * ddt for v in q_in)
    total_out = math.fsum(v * ddt for v in q_out)
    err = abs((acc[-1] - acc[0]) - (total_in - total_out))
    return MFDAccumulationResult(times, acc, q_in, q_out, ddt, gridlock,
                                 gridlock_t, nmax,
                                 max(0.0, math.fsum(delay_terms)), err)


# ===========================================================================
# 12. SECOND-ORDER MODELS (METANET / PAYNE-WHITHAM / AW-RASCLE-ZHANG)
# ===========================================================================
@dataclass
class SecondOrderResult:
    """Output of a second-order (density + speed) traffic model."""

    model: str
    times: List[float]
    x: List[float]
    densities: List[List[float]]
    speeds: List[List[float]]
    flows: List[List[float]]
    dx: float
    dt: float
    lanes: float
    inflow_cum: List[float]
    outflow_cum: List[float]
    vehicles: List[float]
    conservation_error: float
    max_density: float
    min_speed: float

    def summary(self) -> Dict[str, Any]:
        """Headline numbers of the run."""
        return {
            "model": self.model,
            "nSegments": len(self.x),
            "nSteps": len(self.times) - 1,
            "dxKm": self.dx,
            "dtH": self.dt,
            "vehiclesIn": self.inflow_cum[-1],
            "vehiclesOut": self.outflow_cum[-1],
            "vehiclesStored": self.vehicles[-1],
            "conservationError": self.conservation_error,
            "maxDensity": self.max_density,
            "minSpeed": self.min_speed,
        }


@dataclass
class METANETParams:
    """Calibration constants of the METANET second-order freeway model."""

    v_free: float = 102.0
    k_critical: float = 33.5
    k_jam: float = 180.0
    a_exp: float = 1.867
    tau_h: float = 18.0 / 3600.0
    nu: float = 35.0
    kappa: float = 13.0
    delta: float = 0.0122
    lanes: float = 3.0

    def __post_init__(self) -> None:
        self.v_free = _positive(self.v_free, "v_free")
        self.k_critical = _positive(self.k_critical, "k_critical")
        self.k_jam = _positive(self.k_jam, "k_jam")
        self.a_exp = _positive(self.a_exp, "a_exp")
        self.tau_h = _positive(self.tau_h, "tau_h")
        self.nu = _nonnegative(self.nu, "nu")
        self.kappa = _positive(self.kappa, "kappa")
        self.delta = _nonnegative(self.delta, "delta")
        self.lanes = _positive(self.lanes, "lanes")
        if self.k_critical >= self.k_jam:
            raise TrafficFlowError("k_critical must be below k_jam.")

    def speed_equilibrium(self, k: float) -> float:
        """METANET's equilibrium speed V(k) = v_f exp(-(1/a)(k/k_cr)^a), km/h."""
        kk = _clip(_nonnegative(k, "k"), 0.0, self.k_jam)
        return self.v_free * math.exp(-(1.0 / self.a_exp) *
                                      (kk / self.k_critical) ** self.a_exp)

    def flow_equilibrium(self, k: float) -> float:
        """Equilibrium flow k V(k), veh/h per lane."""
        return _nonnegative(k, "k") * self.speed_equilibrium(k)

    def capacity(self) -> float:
        """Maximum equilibrium flow per lane, veh/h."""
        return _argmax_unimodal(self.flow_equilibrium, 0.0, self.k_jam)[1]


def metanet_simulate(params: METANETParams, k_initial: Sequence[float],
                     v_initial: Optional[Sequence[float]], dx: float, dt: float,
                     n_steps: int, upstream_demand: Any,
                     downstream_density: Any = None,
                     ramp_flows: Optional[Dict[int, Any]] = None,
                     record_every: int = 1) -> SecondOrderResult:
    """Simulate the METANET second-order model (Papageorgiou) on a freeway stretch."""
    k = _as_floats(k_initial, "k_initial")
    _check_len(k, 2, "k_initial")
    n = len(k)
    ddx = _positive(dx, "dx")
    ddt = _positive(dt, "dt")
    if int(n_steps) < 1:
        raise TrafficFlowError("n_steps must be >= 1.")
    n_steps = int(n_steps)
    every = max(1, int(record_every))
    if params.v_free * ddt > ddx * (1.0 + 1e-12):
        raise TrafficFlowError("CFL violated: v_free*dt=%g exceeds dx=%g."
                               % (params.v_free * ddt, ddx))
    v = [params.speed_equilibrium(ki) for ki in k] if v_initial is None \
        else _as_floats(v_initial, "v_initial")
    if len(v) != n:
        raise TrafficFlowError("v_initial must match k_initial in length.")
    k = [_clip(ki, 0.0, params.k_jam) for ki in k]
    up_fn = _as_flow_fn(upstream_demand, "upstream_demand")
    dn_fn = _as_flow_fn(0.0 if downstream_density is None else
                        downstream_density, "downstream_density")
    ramps = {}      # type: Dict[int, FlowFn]
    for idx, prof in (ramp_flows or {}).items():
        i = int(idx)
        if i < 0 or i >= n:
            raise TrafficFlowError("ramp segment index %d out of range." % i)
        ramps[i] = _as_flow_fn(prof, "ramp flow")

    lam = params.lanes
    x = [(i + 0.5) * ddx for i in range(n)]
    times = [0.0]
    kh = [list(k)]
    vh = [list(v)]
    qh = [[k[i] * v[i] * lam for i in range(n)]]
    in_terms = []       # type: List[float]
    out_terms = []      # type: List[float]
    in_cum = [0.0]
    out_cum = [0.0]
    veh = [math.fsum(k) * ddx * lam]
    veh0 = veh[0]
    kmax = max(k)
    vmin = min(v)

    for step in range(n_steps):
        t = step * ddt
        # sending limit: a segment cannot release more than it holds
        q = [min(k[i] * v[i] * lam, k[i] * ddx * lam / ddt) for i in range(n)]
        q_in = max(0.0, up_fn(t))
        k_dn = _clip(dn_fn(t), 0.0, params.k_jam)
        r = [max(0.0, ramps.get(i, lambda _t: 0.0)(t)) for i in range(n)]
        # receiving limit: a backward sweep keeps 0 <= rho <= rho_jam and makes
        # the scheme exactly conservative (no clipping ever discards vehicles)
        for i in range(n - 1, -1, -1):
            allowed = (params.k_jam - k[i]) * ddx * lam / ddt + q[i]
            arriving = (q[i - 1] if i > 0 else q_in) + r[i]
            excess = arriving - allowed
            if excess > 0.0:
                if i > 0:
                    take = min(excess, q[i - 1])
                    q[i - 1] -= take
                else:
                    take = min(excess, q_in)
                    q_in -= take
                if excess - take > 0.0:
                    r[i] = max(0.0, r[i] - (excess - take))
        k_new = [0.0] * n
        v_new = [0.0] * n
        for i in range(n):
            q_up = q_in if i == 0 else q[i - 1]
            k_new[i] = _clip(k[i] + (ddt / (ddx * lam)) * (q_up - q[i] + r[i]),
                             0.0, params.k_jam)
        for i in range(n):
            v_up = v[0] if i == 0 else v[i - 1]
            k_dn_i = k_dn if i == n - 1 else k[i + 1]
            relax = (ddt / params.tau_h) * (params.speed_equilibrium(k[i]) - v[i])
            conv = (ddt / ddx) * v[i] * (v_up - v[i])
            aniso = (params.nu * ddt / (params.tau_h * ddx)) * \
                    (k_dn_i - k[i]) / (k[i] + params.kappa)
            merge = 0.0
            if r[i] > 0.0 and params.delta > 0.0:
                merge = (params.delta * ddt * r[i] * v[i] /
                         (ddx * lam * (k[i] + params.kappa)))
            v_new[i] = _clip(v[i] + relax + conv - aniso - merge,
                             0.0, params.v_free)
        k, v = k_new, v_new
        in_terms.append((q_in + math.fsum(r)) * ddt)
        out_terms.append(q[n - 1] * ddt)
        kmax = max(kmax, max(k))
        vmin = min(vmin, min(v))
        if (step + 1) % every == 0 or step == n_steps - 1:
            times.append((step + 1) * ddt)
            kh.append(list(k))
            vh.append(list(v))
            qh.append([k[i] * v[i] * lam for i in range(n)])
            in_cum.append(math.fsum(in_terms))
            out_cum.append(math.fsum(out_terms))
            veh.append(math.fsum(k) * ddx * lam)

    err = abs((veh[-1] - veh0) - (in_cum[-1] - out_cum[-1]))
    return SecondOrderResult("metanet", times, x, kh, vh, qh, ddx, ddt, lam,
                             in_cum, out_cum, veh, err, kmax, vmin)


def payne_whitham_simulate(fd: FundamentalDiagram, k_initial: Sequence[float],
                           v_initial: Optional[Sequence[float]], dx: float,
                           dt: float, n_steps: int, tau_h: float = 0.005,
                           c0: float = 20.0, lanes: float = 1.0,
                           upstream_demand: Any = None,
                           record_every: int = 1) -> SecondOrderResult:
    """Simulate the Payne-Whitham model with an anticipation (pressure) term."""
    k = _as_floats(k_initial, "k_initial")
    _check_len(k, 3, "k_initial")
    n = len(k)
    ddx = _positive(dx, "dx")
    ddt = _positive(dt, "dt")
    if int(n_steps) < 1:
        raise TrafficFlowError("n_steps must be >= 1.")
    n_steps = int(n_steps)
    every = max(1, int(record_every))
    tau = _positive(tau_h, "tau_h")
    cc = _nonnegative(c0, "c0")
    lam = _positive(lanes, "lanes")
    if (fd.v_free + cc) * ddt > ddx * (1.0 + 1e-12):
        raise TrafficFlowError("CFL violated: (v_free+c0)*dt=%g exceeds dx=%g."
                               % ((fd.v_free + cc) * ddt, ddx))
    k = [_clip(ki, 0.0, fd.k_jam) for ki in k]
    v = [fd.speed(ki) for ki in k] if v_initial is None \
        else _as_floats(v_initial, "v_initial")
    if len(v) != n:
        raise TrafficFlowError("v_initial must match k_initial in length.")
    up_fn = _as_flow_fn(fd.capacity if upstream_demand is None
                        else upstream_demand, "upstream_demand")

    x = [(i + 0.5) * ddx for i in range(n)]
    times = [0.0]
    kh = [list(k)]
    vh = [list(v)]
    qh = [[k[i] * v[i] * lam for i in range(n)]]
    in_terms = []       # type: List[float]
    out_terms = []      # type: List[float]
    in_cum = [0.0]
    out_cum = [0.0]
    veh = [math.fsum(k) * ddx * lam]
    veh0 = veh[0]
    kmax = max(k)
    vmin = min(v)
    # Relative to k_jam: this floor only guards the 1/k in the pressure
    # gradient below, and the scheme is exactly invariant under a common
    # rescaling of (k, k_jam) -- an ABSOLUTE 1e-6 veh/km floor broke that
    # invariance as soon as the working density fell under it.
    k_floor = fd.k_jam * 1e-9

    for step in range(n_steps):
        t = step * ddt
        # upwind mass flux (speeds are non-negative), limited by what the
        # upstream cell holds and by the room left downstream: this keeps
        # 0 <= k <= k_jam and makes the mass update exactly conservative
        flux = [0.0] * (n + 1)
        room0 = (fd.k_jam - k[0]) * ddx / ddt
        flux[0] = min(max(0.0, up_fn(t)), fd.supply(k[0]), room0)
        for i in range(1, n):
            flux[i] = min(k[i - 1] * v[i - 1], k[i - 1] * ddx / ddt,
                          (fd.k_jam - k[i]) * ddx / ddt)
        flux[n] = min(k[n - 1] * v[n - 1], k[n - 1] * ddx / ddt)
        k_new = [_clip(k[i] + (ddt / ddx) * (flux[i] - flux[i + 1]),
                       0.0, fd.k_jam) for i in range(n)]
        v_new = [0.0] * n
        for i in range(n):
            v_up = v[i - 1] if i > 0 else v[0]
            k_up = k[i - 1] if i > 0 else k[0]
            k_dn = k[i + 1] if i < n - 1 else k[n - 1]
            conv = -(ddt / ddx) * v[i] * (v[i] - v_up)
            relax = (ddt / tau) * (fd.speed(k[i]) - v[i])
            press = -(ddt * cc * cc) * (k_dn - k_up) / (2.0 * ddx *
                                                        max(k[i], k_floor))
            v_new[i] = _clip(v[i] + conv + relax + press, 0.0, fd.v_free)
        k, v = k_new, v_new
        in_terms.append(flux[0] * ddt * lam)
        out_terms.append(flux[n] * ddt * lam)
        kmax = max(kmax, max(k))
        vmin = min(vmin, min(v))
        if (step + 1) % every == 0 or step == n_steps - 1:
            times.append((step + 1) * ddt)
            kh.append(list(k))
            vh.append(list(v))
            qh.append([k[i] * v[i] * lam for i in range(n)])
            in_cum.append(math.fsum(in_terms))
            out_cum.append(math.fsum(out_terms))
            veh.append(math.fsum(k) * ddx * lam)

    err = abs((veh[-1] - veh0) - (in_cum[-1] - out_cum[-1]))
    return SecondOrderResult("payne_whitham", times, x, kh, vh, qh, ddx, ddt,
                             lam, in_cum, out_cum, veh, err, kmax, vmin)


def arz_simulate(k_initial: Sequence[float], v_initial: Sequence[float],
                 dx: float, dt: float, n_steps: int, v_free: float = 100.0,
                 k_jam: float = 180.0, gamma: float = 1.0,
                 tau_h: float = 0.005, lanes: float = 1.0,
                 record_every: int = 1) -> SecondOrderResult:
    """Simulate Aw-Rascle-Zhang with a Rusanov flux (exactly mass-conservative)."""
    k = _as_floats(k_initial, "k_initial")
    v = _as_floats(v_initial, "v_initial")
    if len(k) != len(v):
        raise TrafficFlowError("k_initial and v_initial must have equal length.")
    _check_len(k, 3, "k_initial")
    n = len(k)
    ddx = _positive(dx, "dx")
    ddt = _positive(dt, "dt")
    if int(n_steps) < 1:
        raise TrafficFlowError("n_steps must be >= 1.")
    n_steps = int(n_steps)
    every = max(1, int(record_every))
    vf = _positive(v_free, "v_free")
    kj = _positive(k_jam, "k_jam")
    g = _positive(gamma, "gamma")
    tau = _positive(tau_h, "tau_h")
    lam = _positive(lanes, "lanes")
    # Relative to k_jam.  Unlike the Payne-Whitham floor this one has two jobs
    # -- it guards the pressure derivative in eigen() and doubles as the vacuum
    # detector in the update -- but it is scale-blind for the same reason, so
    # the same k_jam-relative form applies.
    k_floor = kj * 1e-12

    def pressure(kk: float) -> float:
        return vf * (max(kk, 0.0) / kj) ** g

    def equilibrium(kk: float) -> float:
        return max(0.0, vf - pressure(kk))

    def eigen(kk: float, vv: float) -> float:
        dp = 0.0 if kk <= k_floor else vf * g * (kk / kj) ** (g - 1.0) / kj
        return max(abs(vv), abs(vv - kk * dp))

    k = [_clip(ki, 0.0, kj) for ki in k]
    v = [_clip(vi, 0.0, vf) for vi in v]
    y = [k[i] * (v[i] + pressure(k[i])) for i in range(n)]
    max_eig = max(eigen(k[i], v[i]) for i in range(n))
    if max_eig * ddt > ddx * (1.0 + 1e-12):
        raise TrafficFlowError("CFL violated: max|lambda|*dt=%g exceeds dx=%g."
                               % (max_eig * ddt, ddx))

    x = [(i + 0.5) * ddx for i in range(n)]
    times = [0.0]
    kh = [list(k)]
    vh = [list(v)]
    qh = [[k[i] * v[i] * lam for i in range(n)]]
    in_terms = []       # type: List[float]
    out_terms = []      # type: List[float]
    in_cum = [0.0]
    out_cum = [0.0]
    veh = [math.fsum(k) * ddx * lam]
    veh0 = veh[0]
    kmax = max(k)
    vmin = min(v)

    for step in range(n_steps):
        kk = [k[0]] + list(k) + [k[n - 1]]          # transmissive ghost cells
        vv = [v[0]] + list(v) + [v[n - 1]]
        yy = [kk[i] * (vv[i] + pressure(kk[i])) for i in range(n + 2)]
        fk = [0.0] * (n + 1)
        fy = [0.0] * (n + 1)
        for i in range(n + 1):
            kl, vl, yl = kk[i], vv[i], yy[i]
            kr, vr, yr = kk[i + 1], vv[i + 1], yy[i + 1]
            alpha = max(eigen(kl, vl), eigen(kr, vr))
            fk[i] = 0.5 * (kl * vl + kr * vr) - 0.5 * alpha * (kr - kl)
            fy[i] = 0.5 * (yl * vl + yr * vr) - 0.5 * alpha * (yr - yl)
        # The Rusanov flux telescopes, so it is conservative by construction --
        # but the [0, k_jam] clip below is not.  Limit every interface flux to
        # what the sending cell actually holds and to the room left in the
        # receiving one, so the clip never binds and mass balances exactly.
        # Two sweeps suffice: shrinking a flux can only help the other bound.
        # Every factor is exactly 1.0 on well-resolved data, leaving the
        # textbook scheme bit-identical.
        scale = [1.0] * (n + 1)
        for _pass in (0, 1):
            factor = [1.0] * n
            for j in range(n):
                if _pass == 0:
                    moved = (max(0.0, fk[j + 1] * scale[j + 1])
                             + max(0.0, -fk[j] * scale[j]))
                    budget = k[j] * ddx / ddt
                else:
                    moved = (max(0.0, fk[j] * scale[j])
                             + max(0.0, -fk[j + 1] * scale[j + 1]))
                    budget = (kj - k[j]) * ddx / ddt
                if moved > budget and moved > _TINY:
                    factor[j] = max(0.0, budget / moved)
            for i in range(n + 1):
                f = fk[i]
                if _pass == 0:
                    owner = (i - 1) if f > 0.0 else i
                else:
                    owner = i if f > 0.0 else (i - 1)
                if 0 <= owner < n:
                    scale[i] *= factor[owner]
        for i in range(n + 1):
            fk[i] *= scale[i]
            fy[i] *= scale[i]
        for i in range(n):
            k[i] = _clip(k[i] + (ddt / ddx) * (fk[i] - fk[i + 1]), 0.0, kj)
            src = k[i] * (equilibrium(k[i]) - v[i]) / tau
            y[i] = y[i] + (ddt / ddx) * (fy[i] - fy[i + 1]) + ddt * src
            if k[i] <= k_floor:
                v[i] = vf
                y[i] = k[i] * (v[i] + pressure(k[i]))
            else:
                v[i] = _clip(y[i] / k[i] - pressure(k[i]), 0.0, vf)
                y[i] = k[i] * (v[i] + pressure(k[i]))
        in_terms.append(fk[0] * ddt * lam)
        out_terms.append(fk[n] * ddt * lam)
        kmax = max(kmax, max(k))
        vmin = min(vmin, min(v))
        if (step + 1) % every == 0 or step == n_steps - 1:
            times.append((step + 1) * ddt)
            kh.append(list(k))
            vh.append(list(v))
            qh.append([k[i] * v[i] * lam for i in range(n)])
            in_cum.append(math.fsum(in_terms))
            out_cum.append(math.fsum(out_terms))
            veh.append(math.fsum(k) * ddx * lam)

    err = abs((veh[-1] - veh0) - (in_cum[-1] - out_cum[-1]))
    return SecondOrderResult("arz", times, x, kh, vh, qh, ddx, ddt, lam,
                             in_cum, out_cum, veh, err, kmax, vmin)


# ===========================================================================
# 13. CAR-FOLLOWING MODELS        -- units here are SI: metres, seconds, m/s
# ===========================================================================
class CarFollowingModel:
    """Base car-following law: a = f(v, gap, dv) with dv = v_leader - v_follower."""

    def __init__(self, v_max: float = 33.0, s0: float = 2.0,
                 length: float = 5.0, name: str = "generic",
                 accel_fn: Optional[Callable[[float, float, float], float]] = None,
                 spacing_fn: Optional[Callable[[float], float]] = None) -> None:
        self.v_max = _positive(v_max, "v_max")
        self.s0 = _nonnegative(s0, "s0")
        self.length = _positive(length, "length")
        self.name = str(name)
        if accel_fn is not None and not callable(accel_fn):
            raise TrafficFlowError("accel_fn must be callable.")
        if spacing_fn is not None and not callable(spacing_fn):
            raise TrafficFlowError("spacing_fn must be callable.")
        self._accel_fn = accel_fn
        self._spacing_fn = spacing_fn
        if type(self) is CarFollowingModel and (accel_fn is None or
                                                spacing_fn is None):
            raise TrafficFlowError("CarFollowingModel needs both accel_fn and "
                                   "spacing_fn (or use a subclass).")

    # -- model surface ------------------------------------------------------
    def acceleration(self, v: float, gap: float, dv: float = 0.0) -> float:
        """Acceleration in m/s^2 at speed ``v``, clear gap ``gap`` and closing rate ``dv``."""
        return float(self._accel_fn(_nonnegative(v, "v"),
                                    max(_as_float(gap, "gap"), _CF_MIN_GAP),
                                    _as_float(dv, "dv")))

    def equilibrium_spacing(self, v: float) -> float:
        """Clear gap at which the model produces exactly zero acceleration."""
        return float(self._spacing_fn(_nonnegative(v, "v")))

    def equilibrium_speed(self, gap: float) -> float:
        """Steady-state speed sustained at clear gap ``gap`` (inverse of the above)."""
        g = _nonnegative(gap, "gap")
        hi = self.v_max * (1.0 - 1e-9)
        if g <= self.equilibrium_spacing(0.0):
            return 0.0
        if g >= self.equilibrium_spacing(hi):
            return self.v_max
        return _bisect(lambda v: self.equilibrium_spacing(v) - g, 0.0, hi)

    def step(self, v: float, gap: float, dv: float, dt: float,
             rng: Optional[random.Random] = None) -> Tuple[float, float]:
        """Advance one vehicle by ``dt`` seconds; returns (new speed, acceleration).

        ``rng`` is part of the shared interface but is consumed only by the
        models that are actually stochastic -- in this module that is
        ``KraussCF`` alone (its dawdling term).  Every other law here is
        deterministic and ignores it, so changing the seed changes nothing for
        them.  That is the documented contract, not an oversight: raising on a
        seed would break polymorphic code that sweeps a fixed seed across a
        mixed set of models.
        """
        ddt = _positive(dt, "dt")
        a = self.acceleration(v, gap, dv)
        v_new = _clip(v + a * ddt, 0.0, self.v_max)
        return v_new, a

    # -- derived quantities -------------------------------------------------
    def equilibrium_flow(self, v: float) -> float:
        """Steady-state flow (veh/h) implied by the equilibrium spacing at ``v``."""
        vv = _nonnegative(v, "v")
        spacing = self.equilibrium_spacing(vv) + self.length
        return 3600.0 * vv / spacing if spacing > _EPS else 0.0

    def fundamental_diagram(self, k_jam: Optional[float] = None
                            ) -> FundamentalDiagram:
        """Macroscopic FD implied by the model's equilibrium spacing (veh/km, km/h).

        The diagram returned is CONTINUOUS: it wraps the model's own
        equilibrium-speed solver rather than a sampled grid, so there is no
        resolution knob.  An ``n`` parameter used to sit in front of ``k_jam``
        here; it was validated and then never read, so ``n=2`` and ``n=100000``
        returned bit-identical diagrams.  It has been removed rather than left
        as a silent no-op -- use ``fd_table(fd, n)`` to sample the result.
        """
        jam = 1000.0 / (self.s0 + self.length) if k_jam is None \
            else _positive(k_jam, "k_jam")

        def speed_of_density(k: float) -> float:
            if k <= _EPS:
                return self.v_max * KMH_PER_MS
            gap = 1000.0 / k - self.length
            if gap <= 0.0:
                return 0.0
            return self.equilibrium_speed(gap) * KMH_PER_MS

        return FundamentalDiagram(self.v_max * KMH_PER_MS, jam,
                                  speed_of_density, "cf_" + self.name)

    def __repr__(self) -> str:
        return "<%s v_max=%.2f s0=%.2f>" % (self.__class__.__name__,
                                            self.v_max, self.s0)


class NewellCF(CarFollowingModel):
    """Newell's simplified car-following law: a translated leader trajectory."""

    def __init__(self, v_max: float = 33.0, jam_gap: float = 7.0,
                 tau: float = 1.2, tau_relax: Optional[float] = None,
                 length: float = 5.0) -> None:
        super().__init__(v_max, jam_gap, length, "newell")
        self.tau = _positive(tau, "tau")
        self.tau_relax = self.tau if tau_relax is None \
            else _positive(tau_relax, "tau_relax")

    def acceleration(self, v: float, gap: float, dv: float = 0.0) -> float:
        """Relax towards the Newell target speed min(v_max, (gap - d)/tau)."""
        vv = _nonnegative(v, "v")
        g = max(_as_float(gap, "gap"), 0.0)
        target = min(self.v_max, (g - self.s0) / self.tau)
        return (max(0.0, target) - vv) / self.tau_relax

    def equilibrium_spacing(self, v: float) -> float:
        """Newell's linear spacing d + v tau."""
        return self.s0 + _nonnegative(v, "v") * self.tau

    def next_position(self, x_leader_lagged: float) -> float:
        """Newell's exact rule: x_n(t + tau) = x_(n-1)(t) - d."""
        return _as_float(x_leader_lagged, "x_leader_lagged") - self.s0

    @property
    def wave_speed_kmh(self) -> float:
        """Backward wave speed d / tau implied by the model, km/h."""
        return (self.s0 / self.tau) * KMH_PER_MS


class GippsCF(CarFollowingModel):
    """Gipps (1981) safe-distance model: the minimum of a free and a safe branch."""

    def __init__(self, v_max: float = 20.0, a: float = 1.7, b: float = 3.0,
                 b_hat: float = 3.0, tau: float = 0.8, s0: float = 2.0,
                 length: float = 5.0) -> None:
        super().__init__(v_max, s0, length, "gipps")
        self.a = _positive(a, "a")
        self.b = _positive(b, "b")
        self.b_hat = _positive(b_hat, "b_hat")
        self.tau = _positive(tau, "tau")

    def free_branch(self, v: float) -> float:
        """Gipps' unconstrained acceleration branch, m/s."""
        r = _clip(v / self.v_max, 0.0, 1.0)
        return v + 2.5 * self.a * self.tau * (1.0 - r) * math.sqrt(0.025 + r)

    def safe_branch(self, v: float, gap: float, v_lead: float) -> float:
        """Gipps' safe-following branch, m/s."""
        g = max(0.0, gap - self.s0)
        rad = (self.b * self.tau) ** 2 + self.b * (2.0 * g - v * self.tau) \
            + self.b * v_lead * v_lead / self.b_hat
        return -self.b * self.tau + math.sqrt(max(0.0, rad))

    def next_speed(self, v: float, gap: float, v_lead: float) -> float:
        """Speed after one reaction time tau, m/s."""
        return _clip(min(self.free_branch(v), self.safe_branch(v, gap, v_lead)),
                     0.0, self.v_max)

    def acceleration(self, v: float, gap: float, dv: float = 0.0) -> float:
        """Average acceleration over one reaction time."""
        vv = _nonnegative(v, "v")
        v_lead = max(0.0, vv + _as_float(dv, "dv"))
        return (self.next_speed(vv, _as_float(gap, "gap"), v_lead) - vv) / self.tau

    def equilibrium_spacing(self, v: float) -> float:
        """Closed-form Gipps equilibrium gap s0 + 1.5 v tau + v^2 (1/b - 1/b_hat)/2."""
        vv = _nonnegative(v, "v")
        return (self.s0 + 1.5 * vv * self.tau
                + 0.5 * vv * vv * (1.0 / self.b - 1.0 / self.b_hat))

    def step(self, v: float, gap: float, dv: float, dt: float,
             rng: Optional[random.Random] = None) -> Tuple[float, float]:
        """Advance by ``dt`` using the Gipps update rescaled from tau to dt."""
        ddt = _positive(dt, "dt")
        a = self.acceleration(v, gap, dv)
        return _clip(v + a * ddt, 0.0, self.v_max), a


class IDM(CarFollowingModel):
    """Treiber's Intelligent Driver Model -- the workhorse of microsimulation."""

    def __init__(self, v0: float = 33.0, T: float = 1.5, s0: float = 2.0,
                 a: float = 1.0, b: float = 1.5, delta: float = 4.0,
                 length: float = 5.0) -> None:
        super().__init__(v0, s0, length, "idm")
        self.T = _positive(T, "T")
        self.a = _positive(a, "a")
        self.b = _positive(b, "b")
        self.delta = _positive(delta, "delta")

    def desired_gap(self, v: float, dv: float = 0.0) -> float:
        """IDM's dynamic desired gap s*(v, dv), metres."""
        vv = _nonnegative(v, "v")
        approach = -_as_float(dv, "dv")             # closing rate
        return self.s0 + max(0.0, vv * self.T + vv * approach /
                             (2.0 * math.sqrt(self.a * self.b)))

    def acceleration(self, v: float, gap: float, dv: float = 0.0) -> float:
        """IDM acceleration: free term minus the interaction term."""
        vv = _nonnegative(v, "v")
        g = max(_as_float(gap, "gap"), _CF_MIN_GAP)
        star = self.desired_gap(vv, dv)
        free = 1.0 - (vv / self.v_max) ** self.delta
        return self.a * (free - (star / g) ** 2)

    def equilibrium_spacing(self, v: float) -> float:
        """Closed-form IDM equilibrium gap (s0 + vT) / sqrt(1 - (v/v0)^delta)."""
        vv = _clip(_nonnegative(v, "v"), 0.0, self.v_max * (1.0 - 1e-15))
        root = 1.0 - (vv / self.v_max) ** self.delta
        if root <= _TINY:
            return float("inf")
        return (self.s0 + vv * self.T) / math.sqrt(root)


class OptimalVelocity(CarFollowingModel):
    """Bando's Optimal-Velocity model: a = kappa (V(gap) - v)."""

    def __init__(self, v_max: float = 33.0, kappa: float = 0.85,
                 s_c: float = 25.0, beta: float = 0.13, s0: float = 0.0,
                 length: float = 5.0) -> None:
        super().__init__(v_max, s0, length, "optimal_velocity")
        self.kappa = _positive(kappa, "kappa")
        self.s_c = _positive(s_c, "s_c")
        self.beta = _positive(beta, "beta")
        self._t0 = math.tanh(self.beta * self.s_c)

    def optimal_velocity(self, gap: float) -> float:
        """The optimal-velocity function V(gap), m/s (V(0) = 0, V(inf) = v_max)."""
        g = max(_as_float(gap, "gap"), 0.0)
        val = (math.tanh(self.beta * (g - self.s_c)) + self._t0) / (1.0 + self._t0)
        return _clip(self.v_max * val, 0.0, self.v_max)

    def optimal_velocity_slope(self, gap: float) -> float:
        """dV/d(gap), 1/s -- the quantity that decides string stability."""
        g = max(_as_float(gap, "gap"), 0.0)
        ch = math.cosh(self.beta * (g - self.s_c))
        return self.v_max * self.beta / ((1.0 + self._t0) * ch * ch)

    def acceleration(self, v: float, gap: float, dv: float = 0.0) -> float:
        """Relax towards the optimal velocity of the current gap."""
        return self.kappa * (self.optimal_velocity(gap) - _nonnegative(v, "v"))

    def equilibrium_spacing(self, v: float) -> float:
        """Analytic inverse of V: gap = s_c + atanh(.) / beta."""
        vv = _clip(_nonnegative(v, "v"), 0.0, self.v_max * (1.0 - 1e-12))
        arg = vv * (1.0 + self._t0) / self.v_max - self._t0
        arg = _clip(arg, -1.0 + 1e-15, 1.0 - 1e-15)
        return max(0.0, self.s_c + math.atanh(arg) / self.beta)


class FullVelocityDifference(OptimalVelocity):
    """Jiang's Full-Velocity-Difference model: OV plus a relative-speed term."""

    def __init__(self, v_max: float = 33.0, kappa: float = 0.41,
                 lambda_: float = 0.5, s_c: float = 25.0, beta: float = 0.13,
                 s0: float = 0.0, length: float = 5.0) -> None:
        super().__init__(v_max, kappa, s_c, beta, s0, length)
        self.name = "full_velocity_difference"
        self.lambda_ = _nonnegative(lambda_, "lambda_")

    def acceleration(self, v: float, gap: float, dv: float = 0.0) -> float:
        """OV relaxation plus lambda times the speed difference."""
        return (self.kappa * (self.optimal_velocity(gap) - _nonnegative(v, "v"))
                + self.lambda_ * _as_float(dv, "dv"))


class KraussCF(CarFollowingModel):
    """Krauss' collision-free model (the SUMO default) with seeded dawdling."""

    def __init__(self, v_max: float = 33.0, a: float = 2.6, b: float = 4.5,
                 tau_r: float = 1.0, s0: float = 2.5, epsilon: float = 0.5,
                 length: float = 5.0) -> None:
        super().__init__(v_max, s0, length, "krauss")
        self.a = _positive(a, "a")
        self.b = _positive(b, "b")
        self.tau_r = _positive(tau_r, "tau_r")
        self.epsilon = _fraction(epsilon, "epsilon")

    def safe_speed(self, v: float, gap: float, v_lead: float) -> float:
        """Krauss' safe speed, m/s."""
        g = max(0.0, _as_float(gap, "gap") - self.s0)
        denom = (v_lead + v) / (2.0 * self.b) + self.tau_r
        if denom <= _EPS:
            return 0.0
        return max(0.0, v_lead + (g - v_lead * self.tau_r) / denom)

    def desired_speed(self, v: float, gap: float, v_lead: float,
                      dt: Optional[float] = None) -> float:
        """Deterministic desired speed before dawdling, m/s."""
        step = self.tau_r if dt is None else _positive(dt, "dt")
        return _clip(min(v + self.a * step, self.safe_speed(v, gap, v_lead),
                         self.v_max), 0.0, self.v_max)

    def acceleration(self, v: float, gap: float, dv: float = 0.0) -> float:
        """Deterministic acceleration over one reaction time (no dawdling)."""
        vv = _nonnegative(v, "v")
        v_lead = max(0.0, vv + _as_float(dv, "dv"))
        return (self.desired_speed(vv, _as_float(gap, "gap"), v_lead) - vv) \
            / self.tau_r

    def equilibrium_spacing(self, v: float) -> float:
        """Krauss' equilibrium gap s0 + v tau_r."""
        return self.s0 + _nonnegative(v, "v") * self.tau_r

    def step(self, v: float, gap: float, dv: float, dt: float,
             rng: Optional[random.Random] = None) -> Tuple[float, float]:
        """Advance by ``dt``; ``rng`` (if given) applies the seeded dawdling term."""
        ddt = _positive(dt, "dt")
        vv = _nonnegative(v, "v")
        v_lead = max(0.0, vv + _as_float(dv, "dv"))
        v_des = self.desired_speed(vv, gap, v_lead, ddt)
        if rng is not None and self.epsilon > 0.0:
            v_des = max(0.0, v_des - self.epsilon * self.a * ddt * rng.random())
        return _clip(v_des, 0.0, self.v_max), (v_des - vv) / ddt


@dataclass
class PlatoonResult:
    """Trajectories of a simulated platoon (SI units: m, s, m/s)."""

    times: List[float]
    positions: List[List[float]]
    speeds: List[List[float]]
    gaps: List[List[float]]
    accelerations: List[List[float]]
    dt: float
    n_vehicles: int
    min_gap: float
    collision: bool
    collision_time: Optional[float]
    max_deceleration: float

    def speed_of(self, index: int) -> List[float]:
        """Speed history of vehicle ``index`` (0 is the leader)."""
        i = int(index)
        if i < 0 or i >= self.n_vehicles:
            raise TrafficFlowError("vehicle index %d out of range." % i)
        return [row[i] for row in self.speeds]

    def speed_range(self, index: int) -> float:
        """Peak-to-peak speed excursion of one vehicle, m/s."""
        s = self.speed_of(index)
        return max(s) - min(s)

    def summary(self) -> Dict[str, Any]:
        """Headline numbers of the platoon run."""
        return {
            "model": "platoon",
            "nVehicles": self.n_vehicles,
            "nSteps": len(self.times) - 1,
            "dtS": self.dt,
            "minGapM": self.min_gap,
            "collision": self.collision,
            "maxDecelerationMs2": self.max_deceleration,
            "leaderSpeedRange": self.speed_range(0),
            "lastSpeedRange": self.speed_range(self.n_vehicles - 1),
        }


def simulate_platoon(model: CarFollowingModel, n_vehicles: int,
                     leader_speed: Any, dt: float, n_steps: int,
                     initial_gap: Optional[float] = None,
                     seed: Optional[int] = None) -> PlatoonResult:
    """Simulate a single-lane platoon following a prescribed leader (SI units).

    ``seed`` is forwarded to ``model.step`` and only ``KraussCF`` reads it (see
    ``CarFollowingModel.step``).  For every deterministic law in this module --
    IDM, Gipps, Newell, optimal-velocity, full-velocity-difference -- two runs
    with different seeds are byte-identical, by design.
    """
    n = int(n_vehicles)
    if n < 2:
        raise TrafficFlowError("a platoon needs at least 2 vehicles.")
    ddt = _positive(dt, "dt")
    if int(n_steps) < 1:
        raise TrafficFlowError("n_steps must be >= 1.")
    n_steps = int(n_steps)
    lead_fn = leader_speed if callable(leader_speed) \
        else (lambda t: float(leader_speed))
    rng = None if seed is None else random.Random(int(seed))
    v0 = max(0.0, float(lead_fn(0.0)))
    gap0 = model.equilibrium_spacing(v0) if initial_gap is None \
        else _positive(initial_gap, "initial_gap")
    if not math.isfinite(gap0):
        raise TrafficFlowError("the equilibrium gap is unbounded at v=%g; pass "
                               "initial_gap explicitly." % v0)
    spacing = gap0 + model.length
    x = [-i * spacing for i in range(n)]
    v = [v0] * n
    times = [0.0]
    xs = [list(x)]
    vs = [list(v)]
    gaps = [[x[i - 1] - x[i] - model.length for i in range(1, n)]]
    accs = [[0.0] * n]
    min_gap = min(gaps[0]) if gaps[0] else float("inf")
    collision = False
    collision_t = None      # type: Optional[float]
    max_dec = 0.0

    for step in range(n_steps):
        t = step * ddt
        v_new = [0.0] * n
        a_new = [0.0] * n
        lead_next = max(0.0, float(lead_fn(t + ddt)))
        a_new[0] = (lead_next - v[0]) / ddt
        v_new[0] = lead_next
        for i in range(1, n):
            gap = x[i - 1] - x[i] - model.length
            dv = v[i - 1] - v[i]
            if gap <= 0.0 and not collision:
                collision = True
                collision_t = t
            vi, ai = model.step(v[i], max(gap, _CF_MIN_GAP), dv, ddt, rng)
            v_new[i] = vi
            a_new[i] = ai
        x = [x[i] + 0.5 * (v[i] + v_new[i]) * ddt for i in range(n)]
        v = v_new
        max_dec = max(max_dec, max(0.0, -min(a_new)))
        row_gaps = [x[i - 1] - x[i] - model.length for i in range(1, n)]
        if row_gaps:
            min_gap = min(min_gap, min(row_gaps))
        times.append(t + ddt)
        xs.append(list(x))
        vs.append(list(v))
        gaps.append(row_gaps)
        accs.append(a_new)

    return PlatoonResult(times, xs, vs, gaps, accs, ddt, n, min_gap,
                         collision, collision_t, max_dec)


# ===========================================================================
# 14. LANE CHANGING -- MOBIL
# ===========================================================================
@dataclass
class CFState:
    """A (speed, clear gap, speed difference) triple fed to a car-following law."""

    v: float
    gap: float
    dv: float = 0.0

    def acceleration(self, model: CarFollowingModel) -> float:
        """Acceleration this state implies under ``model``."""
        return model.acceleration(self.v, self.gap, self.dv)


@dataclass
class MobilParams:
    """MOBIL tuning: politeness, switching threshold, safety limit and bias."""

    politeness: float = 0.3
    threshold: float = 0.2
    b_safe: float = 4.0
    bias: float = 0.0

    def __post_init__(self) -> None:
        self.politeness = _as_float(self.politeness, "politeness")
        self.threshold = _nonnegative(self.threshold, "threshold")
        self.b_safe = _positive(self.b_safe, "b_safe")
        self.bias = _as_float(self.bias, "bias")


@dataclass
class MobilDecision:
    """Outcome of one MOBIL lane-change evaluation."""

    change: bool
    safe: bool
    incentive: float
    own_gain: float
    new_follower_loss: float
    old_follower_gain: float
    new_follower_acceleration: float
    threshold: float

    def summary(self) -> Dict[str, Any]:
        """The decision as a plain dict."""
        return {
            "change": self.change,
            "safe": self.safe,
            "incentive": self.incentive,
            "ownGain": self.own_gain,
            "newFollowerLoss": self.new_follower_loss,
            "oldFollowerGain": self.old_follower_gain,
            "newFollowerAccel": self.new_follower_acceleration,
            "threshold": self.threshold,
        }


def mobil_decision(model: CarFollowingModel, subject_before: CFState,
                   subject_after: CFState, new_follower_before: CFState,
                   new_follower_after: CFState, old_follower_before: CFState,
                   old_follower_after: CFState,
                   params: Optional[MobilParams] = None) -> MobilDecision:
    """MOBIL: change lane if the polite total gain beats the threshold and is safe."""
    p = params or MobilParams()
    a_c = subject_before.acceleration(model)
    a_c_new = subject_after.acceleration(model)
    a_n = new_follower_before.acceleration(model)
    a_n_new = new_follower_after.acceleration(model)
    a_o = old_follower_before.acceleration(model)
    a_o_new = old_follower_after.acceleration(model)
    own = a_c_new - a_c
    new_loss = a_n_new - a_n
    old_gain = a_o_new - a_o
    incentive = own + p.politeness * (new_loss + old_gain)
    safe = a_n_new >= -p.b_safe
    limit = p.threshold + p.bias
    return MobilDecision(bool(safe and incentive > limit), bool(safe),
                         incentive, own, new_loss, old_gain, a_n_new, limit)


# ===========================================================================
# 15. NAGEL-SCHRECKENBERG CELLULAR AUTOMATON
# ===========================================================================
@dataclass
class NaSchResult:
    """Output of a Nagel-Schreckenberg run on a ring road."""

    n_cells: int
    n_vehicles: int
    v_max: int
    p_slow: float
    positions: List[List[int]]
    speeds: List[List[int]]
    mean_speed: float
    flow_per_cell: float
    density_per_cell: float
    cell_length_m: float
    step_s: float

    @property
    def flow_veh_h(self) -> float:
        """Flow converted to veh/h."""
        return self.flow_per_cell * 3600.0 / self.step_s

    @property
    def density_veh_km(self) -> float:
        """Density converted to veh/km."""
        return self.density_per_cell * 1000.0 / self.cell_length_m

    @property
    def speed_kmh(self) -> float:
        """Mean speed converted to km/h."""
        return self.mean_speed * self.cell_length_m / self.step_s * KMH_PER_MS

    def summary(self) -> Dict[str, Any]:
        """Headline numbers of the run."""
        return {
            "model": "nagel_schreckenberg",
            "nCells": self.n_cells,
            "nVehicles": self.n_vehicles,
            "vMax": self.v_max,
            "pSlow": self.p_slow,
            "meanSpeedCells": self.mean_speed,
            "flowPerCell": self.flow_per_cell,
            "densityPerCell": self.density_per_cell,
            "flowVehH": self.flow_veh_h,
            "densityVehKm": self.density_veh_km,
            "speedKmh": self.speed_kmh,
        }


def nagel_schreckenberg(n_cells: int, n_vehicles: int, v_max: int = 5,
                        p_slow: float = 0.3, n_steps: int = 200,
                        seed: int = 0, warmup: int = 0,
                        cell_length_m: float = 7.5, step_s: float = 1.0,
                        record: bool = True) -> NaSchResult:
    """Nagel-Schreckenberg CA on a ring road -- accelerate, brake, dawdle, move."""
    L = int(n_cells)
    n = int(n_vehicles)
    vm = int(v_max)
    if L < 2:
        raise TrafficFlowError("n_cells must be >= 2.")
    if n < 1 or n > L:
        raise TrafficFlowError("n_vehicles must lie in [1, n_cells].")
    if vm < 1:
        raise TrafficFlowError("v_max must be >= 1.")
    p = _fraction(p_slow, "p_slow")
    steps = int(n_steps)
    if steps < 1:
        raise TrafficFlowError("n_steps must be >= 1.")
    warm = max(0, int(warmup))
    rng = random.Random(int(seed))
    cell_m = _positive(cell_length_m, "cell_length_m")
    dt_s = _positive(step_s, "step_s")

    pos = [int(i * L / n) for i in range(n)]
    seen = set()
    fixed = []
    for x in pos:                                   # guarantee distinct cells
        while x in seen:
            x = (x + 1) % L
        seen.add(x)
        fixed.append(x)
    pos = sorted(fixed)
    vel = [0] * n

    hist_pos = []      # type: List[List[int]]
    hist_vel = []      # type: List[List[int]]
    speed_terms = []   # type: List[float]

    for step in range(warm + steps):
        for i in range(n):
            nxt = pos[(i + 1) % n]
            gap = (nxt - pos[i] - 1) % L
            if n == 1:
                gap = L - 1
            v = min(vel[i] + 1, vm)                 # 1. accelerate
            v = min(v, gap)                         # 2. brake
            if p > 0.0 and v > 0 and rng.random() < p:
                v -= 1                              # 3. dawdle (seeded)
            vel[i] = v
        pos = [(pos[i] + vel[i]) % L for i in range(n)]
        order = sorted(range(n), key=lambda i: pos[i])
        pos = [pos[i] for i in order]
        vel = [vel[i] for i in order]
        if step >= warm:
            speed_terms.append(math.fsum(vel) / float(n))
            if record:
                hist_pos.append(list(pos))
                hist_vel.append(list(vel))

    mean_v = math.fsum(speed_terms) / float(len(speed_terms))
    density = n / float(L)
    flow = density * mean_v
    return NaSchResult(L, n, vm, p, hist_pos, hist_vel, mean_v, flow, density,
                       cell_m, dt_s)


def nasch_fundamental_diagram(n_cells: int = 200, v_max: int = 5,
                              p_slow: float = 0.3, n_steps: int = 300,
                              seed: int = 0, n_points: int = 20,
                              warmup: int = 200) -> List[Dict[str, float]]:
    """Sweep density to trace the CA's emergent fundamental diagram."""
    pts = int(n_points)
    if pts < 2:
        raise TrafficFlowError("n_points must be >= 2.")
    out = []
    for i in range(1, pts + 1):
        n_veh = max(1, int(round(n_cells * i / float(pts + 1))))
        res = nagel_schreckenberg(n_cells, n_veh, v_max, p_slow, n_steps,
                                  seed + i, warmup, record=False)
        out.append({
            "densityPerCell": res.density_per_cell,
            "flowPerCell": res.flow_per_cell,
            "meanSpeedCells": res.mean_speed,
            "densityVehKm": res.density_veh_km,
            "flowVehH": res.flow_veh_h,
            "speedKmh": res.speed_kmh,
        })
    return out


# ===========================================================================
# 16. LINEAR STABILITY OF A CAR-FOLLOWING PLATOON
# ===========================================================================
@dataclass
class StabilityResult:
    """Local and string stability of a car-following law about an equilibrium."""

    v_equilibrium: float
    gap_equilibrium: float
    f_s: float
    f_v: float
    f_dv: float
    locally_stable: bool
    string_stable: bool
    string_margin: float

    def summary(self) -> Dict[str, Any]:
        """The stability verdict as a plain dict."""
        return {
            "vEquilibrium": self.v_equilibrium,
            "gapEquilibrium": self.gap_equilibrium,
            "fGap": self.f_s,
            "fSpeed": self.f_v,
            "fSpeedDiff": self.f_dv,
            "locallyStable": self.locally_stable,
            "stringStable": self.string_stable,
            "stringMargin": self.string_margin,
        }


def string_stability(model: CarFollowingModel, v_equilibrium: float,
                     gap: Optional[float] = None) -> StabilityResult:
    """Wilson-Ward criterion: string stable iff f_v^2/2 - f_dv f_v - f_s > 0."""
    v_eq = _nonnegative(v_equilibrium, "v_equilibrium")
    s_eq = model.equilibrium_spacing(v_eq) if gap is None \
        else _positive(gap, "gap")
    if not math.isfinite(s_eq):
        raise TrafficFlowError("the equilibrium gap is unbounded at v=%g." % v_eq)
    hs = 1e-5 * max(1.0, abs(s_eq))
    hv = 1e-5 * max(1.0, abs(v_eq))
    hd = 1e-5
    # Both divided differences must be divided by the span ACTUALLY sampled.
    # ``acceleration`` floors the gap at _CF_MIN_GAP and rejects a negative
    # speed, so at an equilibrium that sits on a boundary -- v_eq = 0, which for
    # the optimal-velocity family also gives s_eq = 0 -- the low sample is
    # clipped while the nominal denominator 2h is not.  That halved both f_v and
    # f_s there (f_v came out -kappa/2) and flipped string_stable outright: a
    # marginally stable OVM reported a negative margin.  Clip the sample points
    # explicitly and divide by (hi - lo).
    s_hi = s_eq + hs
    s_lo = max(s_eq - hs, _CF_MIN_GAP)
    v_hi = v_eq + hv
    v_lo = max(v_eq - hv, 0.0)
    f_s = (model.acceleration(v_eq, s_hi, 0.0) -
           model.acceleration(v_eq, s_lo, 0.0)) / (s_hi - s_lo)
    f_v = (model.acceleration(v_hi, s_eq, 0.0) -
           model.acceleration(v_lo, s_eq, 0.0)) / (v_hi - v_lo)
    f_dv = (model.acceleration(v_eq, s_eq, hd) -
            model.acceleration(v_eq, s_eq, -hd)) / (2.0 * hd)
    local = (f_s > 0.0) and ((f_dv - f_v) > 0.0)
    margin = 0.5 * f_v * f_v - f_dv * f_v - f_s
    return StabilityResult(v_eq, s_eq, f_s, f_v, f_dv, local, margin > 0.0,
                           margin)


def platoon_amplification(result: PlatoonResult) -> float:
    """Measured speed-perturbation gain from the leader to the last vehicle.

    The "is the leader perturbation resolvable?" floor is ULP-relative to the
    speed level, not an absolute m/s figure: a gain is a dimensionless ratio
    and must not change when the same physics is expressed in a unit where the
    speeds happen to be numerically smaller.  An absolute 1e-12 m/s cut-off
    used to report ``inf`` for a 2e-13 m/s leader excursion that is still ~56
    ulps wide at 20 m/s and whose gain was accurate to ~3%.
    """
    lead = result.speed_range(0)
    last = result.speed_range(result.n_vehicles - 1)
    level = max([abs(v) for v in result.speed_of(0)] +
                [abs(v) for v in result.speed_of(result.n_vehicles - 1)] + [1.0])
    floor = 8.0 * math.ulp(level)
    if lead <= floor:
        return 0.0 if last <= floor else float("inf")
    return last / lead


# ===========================================================================
# 17. PAYLOAD BUILDERS
# ===========================================================================
def build_traffic_flow_payload(fd: FundamentalDiagram, simulation: Any = None,
                               delay: Optional[DelayBreakdown] = None,
                               shockwave: Optional[ShockwaveResult] = None,
                               mfd: Optional[MFDModel] = None,
                               name: str = "Corridor",
                               fd_samples: int = 41,
                               max_frames: int = 60) -> Dict[str, Any]:
    """Build the frontend-ready JSON payload for the Traffic Flow screen."""
    curve = [{"k": round(row["k"], 4), "q": round(row["q"], 3),
              "v": round(row["v"], 3)} for row in fd_table(fd, fd_samples)]
    payload = {
        "screen": "Traffic Flow",
        "corridorName": str(name),
        "fundamentalDiagram": {
            "name": fd.name,
            "vFreeKmh": round(fd.v_free, 4),
            "kJamVehKm": round(fd.k_jam, 4),
            "capacityVehH": round(fd.capacity, 3),
            "criticalDensityVehKm": round(fd.k_critical, 4),
            "speedAtCapacityKmh": round(fd.v_critical, 4),
            "maxWaveSpeedKmh": round(fd.max_wave_speed(), 4),
            "curve": curve,
        },
        "simulation": None,
        "delay": delay.summary() if delay is not None else None,
        "shockwave": shockwave.summary() if shockwave is not None else None,
        "mfd": mfd.summary() if mfd is not None else None,
        "kpis": [],
    }

    if simulation is not None:
        summary = simulation.summary()
        block = {"summary": summary}       # type: Dict[str, Any]
        times = getattr(simulation, "times", None)
        dens = getattr(simulation, "densities", None)
        if times is not None and dens is not None:
            stride = max(1, len(times) // max(1, int(max_frames)))
            frames = []
            for i in range(0, len(times), stride):
                frames.append({"tH": round(times[i], 6),
                               "density": [round(v, 4) for v in dens[i]]})
            block["frames"] = frames
            block["x"] = [round(v, 5) for v in getattr(simulation, "x",
                                                       list(range(len(dens[0]))))] \
                if not hasattr(simulation, "cell_ids") else None
            if hasattr(simulation, "cell_ids"):
                block["cellIds"] = list(simulation.cell_ids)
        if hasattr(simulation, "inflow_cum"):
            stride = max(1, len(simulation.inflow_cum) // max(1, int(max_frames)))
            block["cumulative"] = [
                {"tH": round(simulation.times[i], 6),
                 "in": round(simulation.inflow_cum[i], 4),
                 "out": round(simulation.outflow_cum[i], 4)}
                for i in range(0, len(simulation.inflow_cum), stride)]
        payload["simulation"] = block

        kpis = [
            {"label": "Capacity", "value": round(fd.capacity, 1),
             "unit": "veh/h/lane"},
            {"label": "Critical density", "value": round(fd.k_critical, 1),
             "unit": "veh/km/lane"},
        ]
        for key, label, unit in (("vehiclesIn", "Vehicles in", "veh"),
                                 ("vehiclesOut", "Vehicles out", "veh"),
                                 ("totalDelayVehH", "Total delay", "veh-h"),
                                 ("maxDensity", "Peak density", "veh/km/lane"),
                                 ("conservationError", "Conservation error",
                                  "veh")):
            if key in summary and summary[key] is not None:
                kpis.append({"label": label,
                             "value": round(float(summary[key]), 6),
                             "unit": unit})
        payload["kpis"] = kpis
    else:
        payload["kpis"] = [
            {"label": "Capacity", "value": round(fd.capacity, 1),
             "unit": "veh/h/lane"},
            {"label": "Critical density", "value": round(fd.k_critical, 1),
             "unit": "veh/km/lane"},
            {"label": "Free-flow speed", "value": round(fd.v_free, 1),
             "unit": "km/h"},
        ]
    return payload


def build_a2ui_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Component-schema payload for generated UI / A2UI projects."""
    if not isinstance(payload, dict) or "fundamentalDiagram" not in payload:
        raise TrafficFlowError("build_a2ui_payload expects the dict returned by "
                               "build_traffic_flow_payload.")
    fdblock = payload["fundamentalDiagram"]
    components = []     # type: List[Dict[str, Any]]
    for kpi in payload.get("kpis", []):
        components.append({
            "id": "stat-%s" % str(kpi["label"]).lower().replace(" ", "-"),
            "type": "StatCard",
            "props": {"label": kpi["label"],
                      "value": "%s %s" % (kpi["value"], kpi["unit"])},
        })
    components.append({
        "id": "fundamental-diagram",
        "type": "LineChart",
        "props": {
            "title": "Fundamental diagram (%s)" % fdblock["name"],
            "xLabel": "Density (veh/km/lane)",
            "yLabel": "Flow (veh/h/lane)",
            "series": [{"name": "q(k)",
                        "points": [[row["k"], row["q"]] for row in fdblock["curve"]]}],
            "markers": [{"name": "capacity",
                         "x": fdblock["criticalDensityVehKm"],
                         "y": fdblock["capacityVehH"]}],
        },
    })
    sim = payload.get("simulation")
    if sim and "frames" in sim:
        components.append({
            "id": "density-heatmap",
            "type": "Heatmap",
            "props": {
                "title": "Density over space and time",
                "xLabel": "Cell",
                "yLabel": "Time (h)",
                "rows": [f["density"] for f in sim["frames"]],
                "rowLabels": [f["tH"] for f in sim["frames"]],
                "colLabels": sim.get("cellIds") or sim.get("x") or [],
                "unit": "veh/km/lane",
            },
        })
    if sim and "cumulative" in sim:
        components.append({
            "id": "cumulative-curves",
            "type": "LineChart",
            "props": {
                "title": "Cumulative counts (delay = area between)",
                "xLabel": "Time (h)",
                "yLabel": "Cumulative vehicles",
                "series": [
                    {"name": "upstream N(t)",
                     "points": [[r["tH"], r["in"]] for r in sim["cumulative"]]},
                    {"name": "downstream N(t)",
                     "points": [[r["tH"], r["out"]] for r in sim["cumulative"]]},
                ],
            },
        })
    if payload.get("delay"):
        d = payload["delay"]
        components.append({
            "id": "delay-breakdown",
            "type": "BarChart",
            "props": {
                "title": "Control delay breakdown (s/veh)",
                "categories": ["uniform", "random", "overflow"],
                "values": [round(d["uniformDelayS"], 3),
                           round(d["randomDelayS"], 3),
                           round(d["overflowDelayS"], 3)],
                "badge": "LOS %s" % d["levelOfService"],
            },
        })
    if payload.get("shockwave"):
        s = payload["shockwave"]
        components.append({
            "id": "shockwave-summary",
            "type": "KeyValueList",
            "props": {"title": "Shockwave queue analysis",
                      "items": [{"key": k, "value": v} for k, v in s.items()]},
        })
    if payload.get("mfd"):
        components.append({
            "id": "mfd-summary",
            "type": "KeyValueList",
            "props": {"title": "Macroscopic fundamental diagram",
                      "items": [{"key": k, "value": v}
                                for k, v in payload["mfd"].items()
                                if k != "coefficients"]},
        })
    return {
        "screen": payload["screen"],
        "title": payload["corridorName"],
        "components": components,
        "componentCount": len(components),
    }


# ===========================================================================
# 18. DEMO
# ===========================================================================
def _demo() -> None:
    print("=" * 70)
    print("  TRAFFIC FLOW (pure Python) - demo")
    print("=" * 70)

    # -- 1. fundamental diagram zoo ----------------------------------------
    print("\n-- 1. fundamental diagrams (v_f=100 km/h, k_j=150 veh/km/lane) --")
    zoo = [
        Greenshields(100.0, 150.0),
        Greenberg(25.0, 150.0, 100.0),
        Underwood(100.0, 35.0, 150.0),
        Drake(100.0, 35.0, 150.0),
        NewellDiagram(100.0, 150.0, 20.0),
        TriangularDiagram(100.0, 150.0, 20.0),
        TrapezoidalDiagram(100.0, 150.0, 20.0, 2000.0),
        DelCastillo(100.0, 150.0, 20.0),
    ]
    print("   %-14s %10s %10s %10s %12s" %
          ("model", "k_crit", "capacity", "v_at_cap", "vs argmax"))
    for fd in zoo:
        k_num, q_num = fd_capacity_numeric(fd, 40001)
        rel = abs(fd.capacity - q_num) / q_num
        print("   %-14s %10.3f %10.1f %10.2f %12.2e"
              % (fd.name, fd.k_critical, fd.capacity, fd.v_critical, rel))
    gs = zoo[0]
    print("   Greenshields closed form: v_f k_j / 4 = %.1f at k_j/2 = %.1f "
          "(error %.2e)" % (0.25 * gs.v_free * gs.k_jam, 0.5 * gs.k_jam,
                            abs(gs.capacity - 0.25 * gs.v_free * gs.k_jam)))
    tri = zoo[5]
    print("   fd_from_name('daganzo') capacity = %.1f veh/h; table rows = %d"
          % (fd_from_name("daganzo", v_free=100.0, k_jam=150.0, w=20.0).capacity,
             len(fd_table(tri, 21))))
    print("   free-flow travel time over 5 km = %.4f h (= 5/v_f = %.4f h)"
          % (5.0 / tri.speed(0.0), 5.0 / tri.v_free))
    print("   units: 30 s = %.6f h = %.1f s; 100 km/h = %.3f m/s = %.1f km/h"
          % (seconds(30.0), hours_to_seconds(seconds(30.0)),
             kmh_to_ms(100.0), ms_to_kmh(kmh_to_ms(100.0))))
    print("   KMH_PER_MS = %.1f" % KMH_PER_MS)
    try:
        tri.density_from_flow(1e6)
    except TrafficFlowError as exc:
        print("   guard rail: %s" % str(exc)[:58])

    # -- 2. kinematic waves -------------------------------------------------
    print("\n-- 2. LWR kinematic waves (triangular FD) --")
    k_l, k_r = 20.0, 100.0
    q_l, q_r = tri.flow(k_l), tri.flow(k_r)
    s_analytic = shock_speed(k_l, q_l, k_r, q_r)
    rs = riemann_solve(tri, k_l, k_r)
    print("   states: k_l=%.0f (q=%.0f), k_r=%.0f (q=%.0f)"
          % (k_l, q_l, k_r, q_r))
    print("   Riemann kind=%s, shock speed=%.4f km/h (Rankine-Hugoniot %.4f)"
          % (rs.kind, rs.shock_speed, s_analytic))
    print("   characteristic speeds: c(k_l)=%.1f, c(k_r)=%.1f km/h"
          % (characteristic_speed(tri, k_l), characteristic_speed(tri, k_r)))
    fan = riemann_solve(tri, 120.0, 5.0)
    print("   reverse Riemann (120 -> 5) kind=%s, fan head=%.1f, tail=%.1f km/h"
          % (fan.kind, fan.head_speed, fan.tail_speed))
    print("   density inside the fan at xi=0 km/h: %.3f (k_crit=%.3f)"
          % (rarefaction_density(tri, 120.0, 5.0, 0.0), tri.k_critical))

    # -- 3. Godunov ---------------------------------------------------------
    print("\n-- 3. Godunov scheme --")
    dx = 0.05
    dt = cfl_timestep(tri, dx, 0.9)
    n_cells = 200
    k0 = [k_l if i < n_cells // 2 else k_r for i in range(n_cells)]
    n_steps = int(round(0.1 / dt))
    gres = godunov_simulate(tri, k0, dx, dt, n_steps, inflow=q_l,
                            downstream_supply=q_r, lanes=1.0, record_every=20)
    t_end = gres.times[-1]
    front = gres.front_position(0.5 * (k_l + k_r))[-1]
    expected = 0.5 * n_cells * dx + s_analytic * t_end
    print("   dx=%.3f km, dt=%.6f h, Courant=%.3f, %d cells, %d steps"
          % (dx, dt, gres.cfl, n_cells, n_steps))
    print("   shock front at t=%.4f h: numerical %.4f km, analytic %.4f km "
          "(|err|=%.4f km = %.2f cells)"
          % (t_end, front, expected, abs(front - expected),
             abs(front - expected) / dx))
    print("   conservation |dStorage - (in - out)| = %.3e veh"
          % gres.conservation_error)
    print("   density range over the run: [%.4f, %.4f] (k_jam=%.1f)"
          % (gres.min_density, gres.max_density, tri.k_jam))
    uni = godunov_simulate(tri, [15.0] * 40, dx, dt, 200, inflow=tri.flow(15.0),
                           downstream_supply=tri.capacity)
    drift = max(abs(v - 15.0) for v in uni.densities[-1])
    print("   uniform stationary state drift after 200 steps = %.3e veh/km"
          % drift)
    print("   godunov_flux(k=20 | k=100) = %.1f veh/h; max wave speed = %.1f km/h"
          % (godunov_flux(tri, 20.0, 100.0), max_wave_speed(tri)))
    print("   cfl_number(dx, dt) = %.4f" % cfl_number(tri, dx, dt))

    # -- 4. cell transmission model ----------------------------------------
    print("\n-- 4. Cell Transmission Model (lane-drop bottleneck) --")
    lane_fd = TriangularDiagram(100.0, 150.0, 20.0)
    ctm_dt = cfl_timestep(lane_fd, 0.1, 0.9)
    net = ctm_corridor(lane_fd, 4.0, 40, ctm_dt, demand=5500.0, lanes=3.0,
                       bottleneck_cells=range(30, 40), bottleneck_lanes=2.0)
    steps = int(round(0.5 / ctm_dt))
    cres = net.simulate(ctm_dt, steps, record_every=20)
    print("   3 lanes -> 2 lanes; demand 5500 veh/h, bottleneck capacity %.0f veh/h"
          % (lane_fd.capacity * 2.0))
    print("   vehicles in=%.2f out=%.2f stored=%.2f, conservation err=%.3e"
          % (cres.inflow_cum[-1], cres.outflow_cum[-1], cres.stored[-1],
             cres.conservation_error))
    print("   TTT=%.2f veh-h, TTD=%.2f veh-km, delay=%.3f veh-h, peak k=%.1f"
          % (cres.total_travel_time, cres.total_travel_distance,
             cres.total_delay, cres.max_density))
    up_k = cres.density_of("c20")[-1]
    dn_k = cres.density_of("c35")[-1]
    print("   final density upstream of the drop=%.1f, inside it=%.1f veh/km/lane"
          % (up_k, dn_k))
    print("   merge_flows([1500,1200], supply 2000, priorities 0.6/0.4) = %s"
          % [round(v, 1) for v in merge_flows([1500.0, 1200.0], 2000.0,
                                              [0.6, 0.4])])
    tot, branch = diverge_flows(2000.0, [1500.0, 300.0], [0.7, 0.3])
    print("   diverge_flows(2000, supplies 1500/300, splits 0.7/0.3) -> "
          "total %.1f, branches %s (FIFO throttling)"
          % (tot, [round(v, 1) for v in branch]))

    net2 = CTMNetwork()
    for name in ("main_up", "ramp", "main_dn"):
        net2.add_cell(name, 0.1, lane_fd, 2.0 if name != "ramp" else 1.0)
    net2.add_source("main_up", 3800.0)
    net2.add_source("ramp", 1400.0)
    net2.add_merge(["main_up", "ramp"], "main_dn", [0.7, 0.3])
    net2.add_sink("main_dn")
    mres = net2.simulate(ctm_dt, 400)
    print("   merge network: mainline+ramp -> 2 lanes; exit flow=%.0f veh/h, "
          "conservation err=%.2e"
          % (mres.outflow_cum[-1] / mres.times[-1], mres.conservation_error))

    drop_net = ctm_corridor(lane_fd, 4.0, 40, ctm_dt, demand=5500.0, lanes=3.0,
                            bottleneck_cells=range(30, 40),
                            bottleneck_lanes=2.0, capacity_drop=0.1)
    dres = drop_net.simulate(ctm_dt, steps, record_every=20)
    print("   capacity drop 10%%: discharge %.0f veh/h vs %.0f veh/h without "
          "(delay %.2f vs %.2f veh-h)"
          % (dres.outflow_cum[-1] / dres.times[-1],
             cres.outflow_cum[-1] / cres.times[-1],
             dres.total_delay, cres.total_delay))

    # -- 5. link transmission model ----------------------------------------
    print("\n-- 5. Link Transmission Model vs CTM --")
    ltm_dt = seconds(1.0)
    link = LTMLink.from_fd("L1", 1.0, lane_fd, lanes=2.0)
    n_ltm = int(round(0.4 / ltm_dt))

    def pulse(t: float) -> float:
        return 4000.0 if t < 0.1 else 500.0

    lres = ltm_simulate([link], ltm_dt, n_ltm, pulse, exit_supply=2400.0)
    ctm_net = ctm_corridor(lane_fd, 1.0, 10, ltm_dt, demand=pulse, lanes=2.0,
                           exit_supply=2400.0)
    cres2 = ctm_net.simulate(ltm_dt, n_ltm)
    out_ltm = lres.outflow_cum[-1]
    out_ctm = cres2.outflow_cum[-1]
    print("   link 1 km x 2 lanes, dt=%.1f s, demand pulse 4000 -> 500 veh/h, "
          "exit 2400 veh/h" % hours_to_seconds(ltm_dt))
    print("   cumulative out: LTM %.2f veh, CTM %.2f veh (rel diff %.3e)"
          % (out_ltm, out_ctm, abs(out_ltm - out_ctm) / max(out_ltm, 1.0)))
    print("   LTM conservation err=%.3e, stored=%.3f veh"
          % (lres.conservation_error, lres.vehicles[0][-1]))
    free_run = ltm_simulate([link], ltm_dt, n_ltm, 1000.0, exit_supply=2400.0)
    free_tt = free_run.link_travel_time("L1", 0.05)
    print("   uncongested run (1000 veh/h): travel time = %.8f h, free-flow "
          "L/v_f = %.8f h (err %.2e h)"
          % (free_tt, link.free_flow_time, abs(free_tt - link.free_flow_time)))
    print("   congested run: travel time of the entrant at t=0.05 h = %.6f h"
          % lres.link_travel_time("L1", 0.05))

    # -- 6. cumulative curves & Newell's three-detector problem -------------
    print("\n-- 6. cumulative curves and Newell's three-detector problem --")
    U = lres.curve("L1", "upstream")
    V = lres.curve("L1", "downstream")
    delay_area = delay_between_curves(U, V)
    ref = math.fsum(0.5 * (U.times[i] - U.times[i - 1]) *
                    ((U.counts[i - 1] - V.counts[i - 1]) +
                     (U.counts[i] - V.counts[i]))
                    for i in range(1, len(U.times)))
    print("   delay from area between curves = %.6f veh-h (independent "
          "trapezoid %.6f, diff %.2e)" % (delay_area, ref, abs(delay_area - ref)))
    print("   queue at t=0.1 h = %.2f veh; travel time there = %.5f h"
          % (queue_from_curves(U, V, 0.1), travel_time_from_curves(U, V, 0.1)))
    mid = newell_three_detector(U, V, 0.5, 1.0, link.v_free, link.w,
                                link.k_jam, U.times, lanes=2.0)
    t_probe = 0.12
    print("   N(x=0.5 km, t=%.2f h) = %.2f, bounded by N_up=%.2f and N_dn=%.2f"
          % (t_probe, mid.at(t_probe), U.at(t_probe), V.at(t_probe)))
    prof = cumulative_from_flows([0.0, 0.25, 0.5], [1800.0, 900.0])
    print("   cumulative_from_flows: N(0.5 h)=%.1f veh, interval flows=%s veh/h"
          % (prof.at(0.5), [round(v) for v in prof.flows()]))
    print("   shifted curve: N_shift(0.3 h) = %.1f (= N(0.25 h)=%.1f)"
          % (prof.shifted(0.05).at(0.3), prof.at(0.25)))

    # -- 7. queues ----------------------------------------------------------
    print("\n-- 7. point and physical queues --")
    q_dt = seconds(10.0)
    arrivals = [2000.0 if i * q_dt < 0.25 else 800.0
                for i in range(int(round(0.75 / q_dt)))]
    vq = vertical_queue(arrivals, 1200.0, q_dt)
    print("   vertical queue: max %.1f veh (analytic (2000-1200)*0.25 = %.1f), "
          "delay %.2f veh-h, err %.1e"
          % (vq.max_queue_veh, (2000.0 - 1200.0) * 0.25, vq.total_delay,
             vq.conservation_error))
    hq = horizontal_queue(arrivals, 1200.0, q_dt, 0.3, 2.0, 150.0, 12.0)
    print("   physical queue on 0.3 km x 2 lanes: max %.1f veh (%.3f km), "
          "spillback=%s at t=%s h, rejected %.1f veh"
          % (hq.max_queue_veh, hq.max_queue_km, hq.spillback,
             "%.4f" % hq.spillback_time if hq.spillback_time is not None else "-",
             math.fsum(hq.rejected)))
    print("   queue summary keys: %d, conservation err %.1e"
          % (len(hq.summary()), hq.conservation_error))

    # -- 8. signal delay ----------------------------------------------------
    print("\n-- 8. signal delay models --")
    q_sig, sat, cycle, green = 600.0, 1800.0, 90.0, 36.0
    d1 = deterministic_delay(q_sig, sat, cycle, green)
    dw = webster_delay(q_sig, sat, cycle, green)
    cap_sig = sat * green / cycle
    print("   q=%.0f veh/h, s=%.0f, C=%.0f s, g=%.0f s -> lambda=%.2f, "
          "capacity=%.0f veh/h, x=%.3f"
          % (q_sig, sat, cycle, green, green / cycle, cap_sig, q_sig / cap_sig))
    print("   uniform (deterministic) delay = %.2f s/veh; Webster total = %.2f "
          "s/veh (stochastic part %.2f)" % (d1, dw, dw - d1))
    print("   Akcelik overflow delay at x=1.04 (T=15 min) = %.2f s/veh"
          % akcelik_delay(750.0, cap_sig, 0.25, sat, green))
    print("   Webster optimal cycle for Y=0.65, L=12 s: %.1f s"
          % webster_optimal_cycle(12.0, [0.30, 0.35]))
    brk = signal_delay(750.0, sat, cycle, green, 0.25)
    print("   breakdown at x=%.3f: uniform %.1f + random %.1f + overflow %.1f "
          "= %.1f s/veh (LOS %s)"
          % (brk.degree_of_saturation, brk.uniform_s, brk.random_s,
             brk.overflow_s, brk.total_s, brk.level_of_service))

    # -- 9. shockwave queue analysis ---------------------------------------
    print("\n-- 9. shockwave queue estimation at a stop line --")
    sw = shockwave_queue(tri, 1200.0, 30.0, 30.0, lanes=1.0)
    print("   arrivals 1200 veh/h, red 30 s, green 30 s")
    print("   stopping wave %.3f km/h, starting wave %.3f km/h, recovery %.1f km/h"
          % (sw.stopping_wave_speed, sw.starting_wave_speed,
             sw.recovery_wave_speed))
    print("   max queue %.1f m (%.1f veh) at t=%.1f s; clears at t=%.1f s (%s)"
          % (sw.max_queue_km * 1000.0, sw.max_queue_veh,
             sw.time_of_max_queue_h * 3600.0, sw.clearance_time_h * 3600.0,
             "within green" if sw.queue_clears else "spills to the next cycle"))

    # -- 10. store-and-forward ---------------------------------------------
    print("\n-- 10. store-and-forward link model --")
    sf_links = [SFLink("A", 1800.0, 120.0, 10.0),
                SFLink("B", 1800.0, 120.0, 5.0),
                SFLink("C", 1800.0, 200.0, 0.0)]
    sf = StoreAndForwardModel(sf_links, {("A", "C"): 0.6, ("B", "C"): 0.5},
                              cycle_s=90.0)
    sf_res = sf.simulate(greens={"A": 30.0, "B": 30.0, "C": 45.0},
                         demands={"A": 700.0, "B": 500.0, "C": 100.0},
                         dt=seconds(10.0), n_steps=180)
    print("   3 links, 90 s cycle; discharge capacity of A at 30 s green = "
          "%.0f veh/h" % sf.outflow_capacity("A", 30.0))
    print("   final queues %s veh, max %.1f, delay %.2f veh-h, linear=%s, "
          "err %.1e" % ([round(v, 1) for v in sf_res.queues[-1]],
                        sf_res.max_queue, sf_res.total_delay,
                        sf_res.summary()["isLinear"], sf_res.conservation_error))

    # -- 11. macroscopic fundamental diagram --------------------------------
    print("\n-- 11. macroscopic fundamental diagram --")
    v_km_h, headway_h = 60.0, 0.01
    trajs = [[(i * headway_h + tau, v_km_h * tau) for tau in
              [0.0, 0.25, 0.5, 0.75]] for i in range(-5, 60)]
    edie = edie_flow_density(trajs, 0.0, 0.5, 0.0, 1.0)
    print("   Edie over [0,0.5 h] x [0,1 km]: q=%.2f veh/h (analytic %.2f), "
          "k=%.4f veh/km, v=%.2f km/h"
          % (edie["flow"], 1.0 / headway_h, edie["density"], edie["speed"]))
    rng = random.Random(7)
    records, ks, qs = [], [], []
    for i in range(60):
        k_net = 5.0 + 130.0 * (i % 30) / 29.0
        batch = []
        for j in range(8):
            kk = max(1.0, k_net * (1.0 + 0.15 * (rng.random() - 0.5)))
            batch.append({"length_km": 0.4 + 0.2 * j / 7.0, "lanes": 2.0,
                          "density": kk, "flow": gs.flow(kk)})
        agg = aggregate_network_state(batch)
        records.append(agg)
        ks.append(agg["density"])
        qs.append(agg["flow"])
    mfd = fit_mfd(ks, qs, degree=3, lane_km=120.0)
    print("   %d aggregated observations; fitted MFD capacity %.0f veh/h/lane "
          "at k=%.1f (Greenshields truth %.0f at %.1f)"
          % (len(ks), mfd.capacity, mfd.critical_density, gs.capacity,
             gs.k_critical))
    print("   critical accumulation over 120 lane-km = %.0f veh; "
          "network speed at k=30 is %.1f km/h"
          % (mfd.critical_accumulation, mfd.speed(30.0)))
    max_trips = mfd.capacity * mfd.lane_km / 5.0
    acc = mfd_accumulation(mfd, lambda t: 1.35 * max_trips if t < 0.5
                           else 0.35 * max_trips, 5.0, seconds(30.0), 240,
                           n0=500.0)
    print("   reservoir (120 lane-km, 5 km trips): trip capacity %.0f trips/h; "
          "overload 135%% for 30 min" % max_trips)
    print("   peak accumulation %.0f veh (critical %.0f), final %.0f, "
          "delay %.0f veh-h, gridlock=%s, err %.1e"
          % (acc.max_accumulation, mfd.critical_accumulation,
             acc.accumulation[-1], acc.total_delay, acc.gridlocked,
             acc.conservation_error))

    # -- 12. second-order models -------------------------------------------
    print("\n-- 12. second-order models --")
    mp = METANETParams()
    seg_k = [22.0] * 20
    mn_dt = seconds(5.0)
    mnres = metanet_simulate(mp, seg_k, None, 0.5, mn_dt, 720,
                             upstream_demand=5200.0,
                             downstream_density=lambda t: 95.0 if t > 0.3 else 25.0,
                             ramp_flows={10: 900.0}, record_every=60)
    print("   METANET: V(k_cr)=%.1f km/h, capacity %.0f veh/h/lane"
          % (mp.speed_equilibrium(mp.k_critical), mp.capacity()))
    print("   20 segments x 0.5 km, ramp at segment 10; peak k=%.1f, "
          "min v=%.1f km/h, conservation err=%.2e"
          % (mnres.max_density, mnres.min_speed, mnres.conservation_error))
    pw_dt = seconds(2.0)
    bump = [30.0 + 70.0 * math.exp(-((i - 25) / 4.0) ** 2) for i in range(50)]
    pw = payne_whitham_simulate(gs, bump, None, 0.1, pw_dt, 300, tau_h=0.005,
                                c0=20.0, record_every=50)
    print("   Payne-Whitham: peak k %.1f -> %.1f, min v %.1f km/h, err %.2e"
          % (max(bump), pw.max_density, pw.min_speed, pw.conservation_error))
    arz_k = [30.0 if i < 25 else 110.0 for i in range(50)]
    arz_v = [gs.speed(k) for k in arz_k]
    az = arz_simulate(arz_k, arz_v, 0.1, seconds(2.0), 300, 100.0, 150.0,
                      1.0, 0.005, record_every=50)
    print("   Aw-Rascle-Zhang (gamma=1 -> Greenshields equilibrium): peak k "
          "%.1f, min v %.1f km/h, mass err %.2e"
          % (az.max_density, az.min_speed, az.conservation_error))
    print("   ARZ stored=%.3f veh, in=%.3f, out=%.3f"
          % (az.vehicles[-1], az.inflow_cum[-1], az.outflow_cum[-1]))

    # -- 13. car-following --------------------------------------------------
    print("\n-- 13. car-following at equilibrium (SI units: m, s) --")
    v_eq = 20.0
    models = [
        NewellCF(33.0, 7.0, 1.2),
        GippsCF(33.0, 1.7, 3.0, 3.0, 0.8, 2.0),
        IDM(33.0, 1.5, 2.0, 1.5, 1.5, 4.0),
        OptimalVelocity(33.0, 0.85, 25.0, 0.13),
        FullVelocityDifference(33.0, 0.41, 0.5, 25.0, 0.13),
        KraussCF(33.0, 2.6, 4.5, 1.0, 2.5, 0.5),
    ]
    print("   %-26s %10s %14s %12s" %
          ("model", "gap_eq(m)", "accel(m/s^2)", "v(gap_eq)"))
    for m in models:
        s_eq = m.equilibrium_spacing(v_eq)
        a_eq = m.acceleration(v_eq, s_eq, 0.0)
        v_back = m.equilibrium_speed(s_eq)
        print("   %-26s %10.4f %14.2e %12.6f" % (m.name, s_eq, a_eq, v_back))
    custom = CarFollowingModel(
        33.0, 2.0, 5.0, "linear_relax",
        accel_fn=lambda v, s, dv: 0.5 * ((s - 2.0) / 1.5 - v) + 0.3 * dv,
        spacing_fn=lambda v: 2.0 + 1.5 * v)
    print("   %-26s %10.4f %14.2e %12.6f"
          % (custom.name, custom.equilibrium_spacing(v_eq),
             custom.acceleration(v_eq, custom.equilibrium_spacing(v_eq), 0.0),
             custom.equilibrium_speed(custom.equilibrium_spacing(v_eq))))
    newell_cf = models[0]
    print("   Newell FD check: implied wave speed %.1f km/h; jam density "
          "%.1f veh/km" % (newell_cf.wave_speed_kmh,
                           newell_cf.fundamental_diagram().k_jam))
    print("   Newell exact rule x_n(t+tau) = x_lead(t) - d -> %.1f m behind"
          % (0.0 - newell_cf.next_position(0.0)))
    print("   Gipps branches at v=20, gap=40: free %.3f, safe %.3f m/s"
          % (models[1].free_branch(20.0), models[1].safe_branch(20.0, 40.0, 20.0)))
    print("   IDM desired gap s*(20 m/s, closing 2 m/s) = %.2f m"
          % models[2].desired_gap(20.0, -2.0))
    print("   Krauss safe speed (v=20, gap=25, lead 18) = %.3f m/s"
          % models[5].safe_speed(20.0, 25.0, 18.0))
    print("   equilibrium flow at 20 m/s: IDM %.0f veh/h, Newell %.0f veh/h"
          % (models[2].equilibrium_flow(v_eq), newell_cf.equilibrium_flow(v_eq)))

    # -- 14. platoons and string stability ---------------------------------
    print("\n-- 14. platoon dynamics and string stability --")

    def brake_tap(t: float) -> float:
        return 20.0 - 2.0 * math.exp(-((t - 10.0) / 1.5) ** 2)

    stable_idm = IDM(33.0, 1.5, 2.0, 1.5, 1.5, 4.0)
    twitchy_idm = IDM(33.0, 1.0, 2.0, 0.3, 2.0, 4.0)
    for label, m in (("stable IDM", stable_idm), ("twitchy IDM", twitchy_idm)):
        st = string_stability(m, v_eq)
        pl = simulate_platoon(m, 12, brake_tap, 0.1, 600, seed=3)
        amp = platoon_amplification(pl)
        print("   %-12s f_s=%.5f f_v=%.5f f_dv=%.5f margin=%+.5f "
              "string_stable=%-5s amplification=%.3f min gap=%.2f m"
              % (label, st.f_s, st.f_v, st.f_dv, st.string_margin,
                 st.string_stable, amp, pl.min_gap))
    ov_unstable = OptimalVelocity(33.0, 0.85, 25.0, 0.13)
    ov_stable = OptimalVelocity(33.0, 5.0, 25.0, 0.13)
    for label, m in (("OV kappa=0.85", ov_unstable), ("OV kappa=5.0", ov_stable)):
        st = string_stability(m, v_eq)
        slope = m.optimal_velocity_slope(st.gap_equilibrium)
        print("   %-14s V'(s_e)=%.4f 1/s vs kappa/2=%.4f -> string_stable=%s "
              "(margin %+.4f)" % (label, slope, m.kappa / 2.0, st.string_stable,
                                  st.string_margin))
    fvd_plat = simulate_platoon(models[4], 10, brake_tap, 0.1, 400, seed=5)
    print("   FVD platoon: leader range %.3f m/s, last range %.3f m/s, "
          "amplification %.3f, collision=%s"
          % (fvd_plat.speed_range(0), fvd_plat.speed_range(9),
             platoon_amplification(fvd_plat), fvd_plat.collision))
    krauss_plat = simulate_platoon(models[5], 8, 18.0, 1.0, 120, seed=11)
    print("   Krauss platoon with seeded dawdling: min gap %.2f m, max decel "
          "%.2f m/s^2, collision=%s"
          % (krauss_plat.min_gap, krauss_plat.max_deceleration,
             krauss_plat.collision))

    # -- 15. MOBIL ----------------------------------------------------------
    print("\n-- 15. MOBIL lane-change decision --")
    mob_model = IDM(33.0, 1.5, 2.0, 1.5, 1.5, 4.0)
    params = MobilParams(politeness=0.3, threshold=0.2, b_safe=4.0)
    dec = mobil_decision(
        mob_model,
        CFState(22.0, 28.0, -3.0),      # subject stuck behind a slow lorry
        CFState(22.0, 70.0, 0.0),       # subject in the target lane
        CFState(24.0, 90.0, 0.0),       # new follower before
        CFState(24.0, 55.0, -2.0),      # new follower after
        CFState(20.0, 40.0, 2.0),       # old follower before
        CFState(20.0, 60.0, 2.0),       # old follower after
        params)
    print("   own gain %+.3f, new-follower loss %+.3f, old-follower gain %+.3f"
          % (dec.own_gain, dec.new_follower_loss, dec.old_follower_gain))
    print("   incentive %+.3f vs threshold %.3f, safe=%s -> change=%s"
          % (dec.incentive, dec.threshold, dec.safe, dec.change))
    tight = mobil_decision(
        mob_model, CFState(22.0, 28.0, -3.0), CFState(22.0, 70.0, 0.0),
        CFState(26.0, 90.0, 0.0), CFState(26.0, 30.0, -4.0),
        CFState(20.0, 40.0, 2.0), CFState(20.0, 60.0, 2.0), params)
    print("   same driver into a tight gap: new-follower accel %.2f m/s^2 -> "
          "safe=%s, change=%s"
          % (tight.new_follower_acceleration, tight.safe, tight.change))

    # -- 16. Nagel-Schreckenberg -------------------------------------------
    print("\n-- 16. Nagel-Schreckenberg cellular automaton --")
    na = nagel_schreckenberg(200, 30, 5, 0.2, 400, seed=42, warmup=200)
    print("   L=200 cells (7.5 m), 30 vehicles, v_max=5, p=0.2")
    print("   mean speed %.3f cells/step (%.1f km/h), flow %.4f veh/cell/step "
          "(%.0f veh/h), density %.0f veh/km"
          % (na.mean_speed, na.speed_kmh, na.flow_per_cell, na.flow_veh_h,
             na.density_veh_km))
    fdia = nasch_fundamental_diagram(200, 5, 0.2, 300, seed=1, n_points=16,
                                     warmup=200)
    peak = max(fdia, key=lambda r: r["flowPerCell"])
    print("   emergent FD peak: %.0f veh/h at %.0f veh/km "
          "(%.3f veh/cell/step at rho=%.3f); %d sweep points"
          % (peak["flowVehH"], peak["densityVehKm"], peak["flowPerCell"],
             peak["densityPerCell"], len(fdia)))
    na2 = nagel_schreckenberg(200, 30, 5, 0.0, 400, seed=42, warmup=200)
    print("   deterministic (p=0): flow %.4f veh/cell/step, mean speed %.3f"
          % (na2.flow_per_cell, na2.mean_speed))

    # -- 17. payloads -------------------------------------------------------
    print("\n-- 17. frontend payloads --")
    payload = build_traffic_flow_payload(lane_fd, cres, brk, sw, mfd,
                                         name="A40 westbound")
    a2ui = build_a2ui_payload(payload)
    print("   payload keys: %s" % ", ".join(sorted(payload.keys())))
    print("   FD curve points %d, simulation frames %d, KPIs %d"
          % (len(payload["fundamentalDiagram"]["curve"]),
             len(payload["simulation"]["frames"]), len(payload["kpis"])))
    print("   a2ui components (%d): %s"
          % (a2ui["componentCount"],
             ", ".join(c["type"] for c in a2ui["components"][:6])))
    gpay = build_traffic_flow_payload(tri, gres, name="Riemann test section")
    print("   godunov payload: %d cumulative rows, screen='%s'"
          % (len(gpay["simulation"]["cumulative"]), gpay["screen"]))

    # -- 18. exported result types ------------------------------------------
    print("\n-- 18. exported container types (every result object type-checked) --")
    surface = [
        (tri, FundamentalDiagram), (rs, RiemannSolution),
        (gres, GodunovResult), (net.cells["c0"], CTMCell),
        (net.nodes[0], CTMNode), (cres, CTMResult), (link, LTMLink),
        (lres, LTMResult), (U, CumulativeCurve), (hq, QueueResult),
        (brk, DelayBreakdown), (sw, ShockwaveResult),
        (sf_res, StoreAndForwardResult), (mfd, MFDModel),
        (acc, MFDAccumulationResult), (mnres, SecondOrderResult),
        (stable_idm, CarFollowingModel), (fvd_plat, PlatoonResult),
        (dec, MobilDecision), (na, NaSchResult),
        (string_stability(stable_idm, v_eq), StabilityResult),
        (MobilParams(), MobilParams), (CFState(20.0, 30.0), CFState),
        (METANETParams(), METANETParams), (sf_links[0], SFLink),
    ]
    bad = [cls.__name__ for obj, cls in surface if not isinstance(obj, cls)]
    print("   %d result/container types checked, mismatches: %s"
          % (len(surface), bad or "none"))
    print("   every summary() is JSON-able: %s"
          % all(isinstance(o.summary(), dict) for o, _ in surface
                if hasattr(o, "summary")))

    print("\n" + "=" * 70)
    print("  demo OK")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
