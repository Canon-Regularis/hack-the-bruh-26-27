"""
Traffic Assignment -- static user equilibrium, stochastic equilibrium and system optimum
=========================================================================================

**Given an origin-destination demand and a road network, where does the traffic
actually go?**  That is the fourth step of the classical four-step model and the
engine underneath every network-design, tolling, signal-timing and emissions study
in the urban domain.  This module is the deep version of that question.

It *deepens* ``core/optimisation/network_optimisation.traffic_assignment``, which is
a single function: one Frank-Wolfe loop over a BPR network with **one global
``alpha`` and one global ``beta`` for every link in the country**.  That is a real
capability gap, not a stylistic one -- with a single shared volume-delay parameter
set you cannot even *write down* Braess's network (whose links are ``10x``, ``50+x``
and ``10+x``: three different functional forms), you cannot mix a motorway with a
residential street, and you cannot toll one link.  Here every link carries its own
volume-delay function object with its own parameters and its own analytic integral,
which is what makes the Beckmann transformation computable and therefore what makes
every convex-programming solver below possible.

On top of that per-link foundation this module carries the full static-assignment
stack: six equilibrium solvers that must agree with one another, stochastic (logit
and probit) equilibrium, the system optimum with its marginal-cost tolls, the price
of anarchy, an automatic Braess-paradox detector, multi-class PCU assignment,
elastic demand, turn penalties on an expanded graph, select-link analysis and a
finite-difference sensitivity of the equilibrium to link capacity for downstream
bi-level network design.  Pure standard library -- no NumPy, no SciPy, no pandas.

Contents
--------
0.  **Validation** -- ``TrafficAssignmentError`` and the value guards.
1.  **Volume-delay functions** -- ``BPR``, ``Davidson``, ``ModifiedDavidson``,
    ``Conical`` (Spiess), ``Akcelik``, ``Polynomial``, ``Linear``, ``Constant``
    plus the ``MarginalCostVDF`` wrapper.  Each is a dataclass of *per-link*
    parameters exposing ``travel_time`` / ``derivative`` / ``integral``, the last
    analytic -- the Beckmann objective needs it.
2.  **Network data model** -- ``Link``, ``Network``, ``ODMatrix``.
3.  **Shortest paths & paths** -- Dijkstra trees on the hot loop, Yen's
    ``k_shortest_paths``, ``enumerate_paths``, ``Path`` / ``PathSet``.
4.  **Objectives & convergence** -- ``beckmann_objective`` / ``beckmann_gradient``,
    ``relative_gap``, ``average_excess_cost``, ``max_flow_change``,
    ``IterationRecord`` / ``AssignmentResult``.
5.  **Link-based solvers** -- ``all_or_nothing``, ``incremental_assignment``,
    ``method_of_successive_averages``.
6.  **Frank-Wolfe family** -- ``frank_wolfe`` with an exact line search,
    ``conjugate_frank_wolfe`` and ``biconjugate_frank_wolfe``
    (Mitradjieva-Lindberg), plus the ``user_equilibrium`` umbrella.
7.  **Path-based** -- ``gradient_projection`` (Jayakrishnan) with column
    generation and path equilibration.
8.  **Origin-based** -- ``algorithm_b``, Dial's bush method.
9.  **Stochastic** -- ``dial_stoch`` (efficient paths + logit weights),
    ``logit_sue`` (MSA on Fisk's objective, with C-logit and path-size overlap
    corrections) and a seeded Monte-Carlo ``probit_sue``.
10. **System optimum & tolls** -- ``system_optimum``, ``marginal_cost_tolls``,
    ``tolled_user_equilibrium``, ``price_of_anarchy``.
11. **Braess** -- ``detect_braess`` searches for a link whose *removal* lowers
    total system travel time; ``braess_network`` is the textbook instance.
12. **Multi-class** -- ``multiclass_assignment`` with PCU equivalents.
13. **Elastic demand** -- Evans' partial linearisation on the excess-demand
    objective with three inverse-demand families.
14. **Turn penalties** -- ``expand_with_turn_penalties`` builds the entry/exit
    expanded graph; ``turn_penalty_assignment`` solves on it and folds back.
15. **Decomposition & analysis** -- ``decompose_flows``, ``wardrop_check``,
    ``select_link_analysis``, ``select_zone_analysis``.
16. **Sensitivity** -- ``capacity_sensitivity`` (central differences on a
    re-solved equilibrium) for bi-level network design.
17. **Builders & closed-form oracles** -- ``two_link_network`` and its analytic
    UE / SO split, ``grid_network``, ``random_affine_network``.
18. **Payloads** -- ``build_traffic_assignment_payload`` / ``build_a2ui_payload``.
19. **Demo**.

Conventions
-----------
* Nodes are ``int`` ids ``0..n_nodes-1``; links are indexed by insertion order.
* A flow vector is a ``list`` of floats, one per link, in link order.
* ``travel_time`` is pure running time; ``cost`` is the generalised cost actually
  routed on -- travel time plus the link's ``toll``.  ``total_system_travel_time``
  deliberately excludes tolls (a toll is a transfer, not a delay).
* Demand is a ``dict`` ``{(origin, destination): trips}`` or an ``ODMatrix``.
* Convexity: every volume-delay function must be non-decreasing on ``x >= 0`` so
  that the Beckmann objective is convex and the equilibrium is unique in flows.
* Everything stochastic takes an explicit ``seed`` and uses ``random.Random(seed)``,
  so results are byte-identical across runs and independent of ``PYTHONHASHSEED``.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field, replace
from typing import (Any, Dict, Iterable, Iterator, List, Optional, Sequence,
                    Set, Tuple)

__version__ = "1.0.0"

Flow = List[float]
Cost = List[float]
Demand = Dict[Tuple[int, int], float]
NodeSeq = List[int]
LinkSeq = List[int]

_EPS = 1e-12
_FLOW_EPS = 1e-11
_INF = float("inf")
_BISECT_ITERS = 80

__all__ = [
    # errors & data model
    "TrafficAssignmentError", "Link", "Network", "ODMatrix",
    # volume-delay functions
    "VolumeDelayFunction", "BPR", "Davidson", "ModifiedDavidson", "Conical",
    "Akcelik", "Polynomial", "Linear", "Constant", "MarginalCostVDF",
    # shortest paths & paths
    "ShortestPathTree", "shortest_path_tree", "shortest_path",
    "k_shortest_paths", "enumerate_paths", "Path", "PathSet", "path_cost",
    # objectives & convergence
    "beckmann_objective", "beckmann_gradient", "total_system_travel_time",
    "relative_gap", "average_excess_cost", "max_flow_change",
    "IterationRecord", "AssignmentResult",
    # link-based solvers
    "AllOrNothingResult", "all_or_nothing", "incremental_assignment",
    "method_of_successive_averages",
    # frank-wolfe family
    "exact_line_search", "frank_wolfe", "conjugate_frank_wolfe",
    "biconjugate_frank_wolfe", "user_equilibrium",
    # path-based & origin-based
    "gradient_projection", "algorithm_b",
    # stochastic
    "dial_stoch", "logit_probabilities", "c_logit_factors",
    "path_size_factors", "fisk_objective", "logit_sue", "probit_sue",
    # system optimum, tolls, price of anarchy
    "system_optimum", "marginal_cost_tolls", "tolled_user_equilibrium",
    "PriceOfAnarchyResult", "price_of_anarchy",
    # braess
    "BraessResult", "detect_braess", "braess_network",
    # multi-class
    "VehicleClass", "MultiClassResult", "multiclass_assignment",
    # elastic demand
    "DemandFunction", "LinearDemand", "ExponentialDemand", "PowerDemand",
    "ElasticDemandResult", "elastic_demand_assignment",
    # turn penalties
    "TurnExpansion", "expand_with_turn_penalties", "turn_penalty_assignment",
    # decomposition & analysis
    "decompose_flows", "WardropReport", "wardrop_check", "SelectLinkResult",
    "select_link_analysis", "select_zone_analysis",
    # sensitivity
    "SensitivityResult", "capacity_sensitivity",
    # builders & oracles
    "two_link_network", "two_link_ue_closed_form", "two_link_so_closed_form",
    "grid_network", "random_affine_network",
    # payloads
    "build_traffic_assignment_payload", "build_a2ui_payload",
]


# ===========================================================================
# 0. VALIDATION HELPERS
# ===========================================================================
class TrafficAssignmentError(ValueError):
    """Invalid input to a traffic-assignment routine."""


def _as_float(value: Any, name: str) -> float:
    """Coerce to a finite float or raise."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise TrafficAssignmentError("%s must be a real number (got %r)."
                                     % (name, value))
    if math.isnan(out) or math.isinf(out):
        raise TrafficAssignmentError("%s must be finite (got %r)." % (name, value))
    return out


def _as_positive(value: Any, name: str) -> float:
    """Coerce to a strictly positive finite float or raise."""
    out = _as_float(value, name)
    if out <= 0.0:
        raise TrafficAssignmentError("%s must be > 0 (got %g)." % (name, out))
    return out


def _as_nonneg(value: Any, name: str) -> float:
    """Coerce to a non-negative finite float or raise."""
    out = _as_float(value, name)
    if out < 0.0:
        raise TrafficAssignmentError("%s must be >= 0 (got %g)." % (name, out))
    return out


def _as_index(value: Any, name: str) -> int:
    """Coerce to a non-negative integer index or raise."""
    try:
        out = int(value)
    except (TypeError, ValueError):
        raise TrafficAssignmentError("%s must be an integer (got %r)."
                                     % (name, value))
    if out < 0:
        raise TrafficAssignmentError("%s must be >= 0 (got %d)." % (name, out))
    return out


def _as_floats(values: Sequence[Any], name: str = "x") -> List[float]:
    """Coerce a sequence to a list of finite floats or raise."""
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError):
        raise TrafficAssignmentError("%s must be a sequence of real numbers." % name)
    for v in out:
        if math.isnan(v) or math.isinf(v):
            raise TrafficAssignmentError("%s contains a non-finite value." % name)
    return out


def _check_flows(network: "Network", flows: Sequence[float],
                 name: str = "flows") -> Flow:
    """Validate a flow vector against a network: right length, non-negative."""
    out = _as_floats(flows, name)
    if len(out) != network.n_links:
        raise TrafficAssignmentError(
            "%s has %d entries but the network has %d links."
            % (name, len(out), network.n_links))
    for i, v in enumerate(out):
        if v < -1e-9:
            raise TrafficAssignmentError("%s[%d] is negative (%g)." % (name, i, v))
        if v < 0.0:
            out[i] = 0.0
    return out


def _safe_div(num: float, den: float) -> float:
    """Divide, returning 0.0 when the denominator is exactly zero.

    The test is ``den == 0.0`` and not ``abs(den) <= _EPS``: this helper is
    general-purpose and every caller's denominator carries its own units, so an
    absolute threshold would silently return 0 for a perfectly good quotient on any
    network whose quantities happen to be small.
    """
    if den == 0.0:
        return 0.0
    return num / den


# ===========================================================================
# 1. VOLUME-DELAY FUNCTIONS -- per-link parameters, analytic integrals
# ===========================================================================
class VolumeDelayFunction(object):
    """Base class for a per-link volume-delay (congestion) function ``t(x)``.

    Subclasses are dataclasses carrying *their own* parameters and must expose a
    non-decreasing ``travel_time`` on ``x >= 0``, its ``derivative`` and its
    analytic ``integral`` -- the last is what the Beckmann transformation sums.
    ``capacity`` is a plain attribute on every subclass (a nominal reference flow
    for the purely algebraic forms).
    """

    t0: float
    capacity: float

    # -- interface ---------------------------------------------------------
    def travel_time(self, flow: float) -> float:
        """Travel time at link flow ``flow``."""
        raise NotImplementedError("subclasses must implement travel_time()")

    def derivative(self, flow: float) -> float:
        """d(travel time)/d(flow) at ``flow``."""
        raise NotImplementedError("subclasses must implement derivative()")

    def integral(self, flow: float) -> float:
        """Analytic ``integral_0^flow t(w) dw`` -- the Beckmann term for this link."""
        raise NotImplementedError("subclasses must implement integral()")

    # -- shared derived quantities ----------------------------------------
    def second_derivative(self, flow: float) -> float:
        """Second derivative; central difference unless a subclass overrides it."""
        x = max(0.0, float(flow))
        cap = abs(float(getattr(self, "capacity", 1.0)))
        # The step has to shrink with the units of the flow axis.  ``max(1.0, cap)``
        # pinned it at 1e-6 for every sub-unit capacity, which swamps the whole
        # curve when capacity itself is 1e-6.
        scale = cap if cap > 0.0 else (x if x > 0.0 else 1.0)
        h = 1e-6 * scale
        lo = max(0.0, x - h)
        hi = x + h
        return (self.derivative(hi) - self.derivative(lo)) / (hi - lo)

    def marginal_cost(self, flow: float) -> float:
        """System-optimal marginal cost ``t(x) + x t'(x)``."""
        x = max(0.0, float(flow))
        return self.travel_time(x) + x * self.derivative(x)

    def free_flow_time(self) -> float:
        """Travel time at zero flow."""
        return self.travel_time(0.0)

    def with_capacity(self, capacity: float) -> "VolumeDelayFunction":
        """A copy of this function with a different capacity (for sensitivity runs)."""
        return replace(self, capacity=_as_positive(capacity, "capacity"))

    def describe(self) -> str:
        """One-line human description of the functional form and its parameters."""
        return "%s(%s)" % (type(self).__name__, ", ".join(
            "%s=%g" % (k, v) if isinstance(v, float) else "%s=%r" % (k, v)
            for k, v in sorted(vars(self).items())
            if not k.startswith("_") and not isinstance(v, VolumeDelayFunction)))


@dataclass
class BPR(VolumeDelayFunction):
    """US Bureau of Public Roads curve ``t0 * (1 + alpha (x/c)**beta)``."""

    t0: float
    capacity: float
    alpha: float = 0.15
    beta: float = 4.0

    def __post_init__(self) -> None:
        self.t0 = _as_nonneg(self.t0, "t0")
        self.capacity = _as_positive(self.capacity, "capacity")
        self.alpha = _as_nonneg(self.alpha, "alpha")
        self.beta = _as_nonneg(self.beta, "beta")

    def travel_time(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        try:
            return self.t0 * (1.0 + self.alpha * (x / self.capacity) ** self.beta)
        except OverflowError:
            # Linear and Polynomial both return ``inf`` for an overflowing cost.
            # Raising here instead let an OverflowError escape straight out of
            # user_equilibrium, which is not even a TrafficAssignmentError.
            return _INF

    def derivative(self, flow: float) -> float:
        if self.alpha == 0.0 or self.beta == 0.0:
            return 0.0
        x = flow if flow > 0.0 else 0.0
        r = x / self.capacity
        if r <= 0.0:
            if self.beta > 1.0:
                return 0.0
            if self.beta == 1.0:
                return self.t0 * self.alpha / self.capacity
            r = 1e-12                       # keep the singular case finite
        try:
            return (self.t0 * self.alpha * self.beta / self.capacity
                    * r ** (self.beta - 1.0))
        except OverflowError:
            return _INF

    def second_derivative(self, flow: float) -> float:
        if self.alpha == 0.0 or self.beta <= 1.0:
            return 0.0
        x = flow if flow > 0.0 else 0.0
        r = x / self.capacity
        if r <= 0.0:
            if self.beta > 2.0:
                return 0.0
            r = 1e-12
        try:
            return (self.t0 * self.alpha * self.beta * (self.beta - 1.0)
                    / (self.capacity ** 2) * r ** (self.beta - 2.0))
        except OverflowError:
            return _INF

    def integral(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        r = x / self.capacity
        try:
            return self.t0 * x + (self.t0 * self.alpha * self.capacity
                                  / (self.beta + 1.0)) * r ** (self.beta + 1.0)
        except OverflowError:
            return _INF


def _davidson_pieces(t0: float, cap: float, j: float, mu: float,
                     flow: float) -> Tuple[float, float, float]:
    """Davidson time / derivative / integral with a linear tail beyond ``mu*cap``."""
    x = flow if flow > 0.0 else 0.0
    xc = mu * cap

    def _raw(v: float) -> Tuple[float, float, float]:
        gap = cap - v
        t = t0 * (1.0 + j * v / gap)
        dt = t0 * j * cap / (gap * gap)
        integ = t0 * v + t0 * j * (cap * math.log(cap / gap) - v)
        return t, dt, integ

    if x <= xc:
        return _raw(x)
    t_c, dt_c, i_c = _raw(xc)
    d = x - xc
    return t_c + dt_c * d, dt_c, i_c + t_c * d + 0.5 * dt_c * d * d


@dataclass
class Davidson(VolumeDelayFunction):
    """Davidson's queueing curve ``t0 (1 + J x / (c - x))`` (singular at ``x = c``).

    The raw form is unbounded as ``x -> c``; we continue it linearly from
    ``mu*c`` (default ``0.999``, i.e. essentially the pure curve) so that a line
    search may safely probe an infeasible overload without hitting a pole.
    """

    t0: float
    capacity: float
    j: float = 0.25
    mu: float = 0.999

    def __post_init__(self) -> None:
        self.t0 = _as_nonneg(self.t0, "t0")
        self.capacity = _as_positive(self.capacity, "capacity")
        self.j = _as_nonneg(self.j, "j")
        self.mu = _as_positive(self.mu, "mu")
        if self.mu >= 1.0:
            raise TrafficAssignmentError("Davidson mu must be < 1 (got %g)." % self.mu)

    def travel_time(self, flow: float) -> float:
        return _davidson_pieces(self.t0, self.capacity, self.j, self.mu, flow)[0]

    def derivative(self, flow: float) -> float:
        return _davidson_pieces(self.t0, self.capacity, self.j, self.mu, flow)[1]

    def integral(self, flow: float) -> float:
        return _davidson_pieces(self.t0, self.capacity, self.j, self.mu, flow)[2]

    def second_derivative(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        if x > self.mu * self.capacity:
            return 0.0
        gap = self.capacity - x
        return 2.0 * self.t0 * self.j * self.capacity / (gap ** 3)


@dataclass
class ModifiedDavidson(VolumeDelayFunction):
    """Davidson with an explicit modelling threshold ``mu`` (Rose-Daskin-Koppelman).

    Identical algebra to :class:`Davidson` but ``mu`` (typically 0.80-0.95) is a
    *deliberate* choice: beyond ``mu*c`` the curve continues with the tangent, which
    keeps over-capacity links finite, differentiable and numerically well behaved.
    """

    t0: float
    capacity: float
    j: float = 0.25
    mu: float = 0.85

    def __post_init__(self) -> None:
        self.t0 = _as_nonneg(self.t0, "t0")
        self.capacity = _as_positive(self.capacity, "capacity")
        self.j = _as_nonneg(self.j, "j")
        self.mu = _as_positive(self.mu, "mu")
        if self.mu >= 1.0:
            raise TrafficAssignmentError(
                "ModifiedDavidson mu must be < 1 (got %g)." % self.mu)

    def travel_time(self, flow: float) -> float:
        return _davidson_pieces(self.t0, self.capacity, self.j, self.mu, flow)[0]

    def derivative(self, flow: float) -> float:
        return _davidson_pieces(self.t0, self.capacity, self.j, self.mu, flow)[1]

    def integral(self, flow: float) -> float:
        return _davidson_pieces(self.t0, self.capacity, self.j, self.mu, flow)[2]

    def second_derivative(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        if x > self.mu * self.capacity:
            return 0.0
        gap = self.capacity - x
        return 2.0 * self.t0 * self.j * self.capacity / (gap ** 3)


@dataclass
class Conical(VolumeDelayFunction):
    """Spiess's conical congestion function -- BPR-shaped but with a bounded slope.

    ``t(x) = t0 (2 + sqrt(a^2 (1-v)^2 + b^2) - a (1-v) - b)`` with ``v = x/c`` and
    ``b = (2a-1)/(2a-2)``, chosen so that ``t(0) = t0`` exactly and ``t(c) = 2 t0``.
    Unlike the BPR it has a finite, well-conditioned derivative everywhere.
    """

    t0: float
    capacity: float
    alpha: float = 8.0

    def __post_init__(self) -> None:
        self.t0 = _as_nonneg(self.t0, "t0")
        self.capacity = _as_positive(self.capacity, "capacity")
        self.alpha = _as_float(self.alpha, "alpha")
        if self.alpha <= 1.0:
            raise TrafficAssignmentError(
                "Conical alpha must be > 1 (got %g)." % self.alpha)

    @property
    def _b(self) -> float:
        a = self.alpha
        return (2.0 * a - 1.0) / (2.0 * a - 2.0)

    def travel_time(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        a, b = self.alpha, self._b
        u = 1.0 - x / self.capacity
        return self.t0 * (2.0 + math.sqrt(a * a * u * u + b * b) - a * u - b)

    def derivative(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        a, b = self.alpha, self._b
        u = 1.0 - x / self.capacity
        q = math.sqrt(a * a * u * u + b * b)
        return self.t0 / self.capacity * (a - a * a * u / q)

    def integral(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        a, b = self.alpha, self._b

        def _g(u: float) -> float:
            q = math.sqrt(a * a * u * u + b * b)
            s = 0.5 * u * q + (b * b / (2.0 * a)) * math.asinh(a * u / b)
            return 2.0 * u + s - 0.5 * a * u * u - b * u

        u1 = 1.0 - x / self.capacity
        return self.t0 * self.capacity * (_g(1.0) - _g(u1))


@dataclass
class Akcelik(VolumeDelayFunction):
    """Akcelik's time-dependent overflow-delay curve (used across Australasia).

    ``t(x) = t0 + 0.25 T [ z + sqrt(z^2 + 8 J x / (c T)) ]`` with ``z = x/c - 1``.
    ``period`` is the flow period ``T`` in the same time units as ``t0``; ``j`` is
    the delay parameter.  The integral is the closed form of ``int sqrt(quadratic)``.
    """

    t0: float
    capacity: float
    j: float = 0.1
    period: float = 1.0

    def __post_init__(self) -> None:
        self.t0 = _as_nonneg(self.t0, "t0")
        self.capacity = _as_positive(self.capacity, "capacity")
        self.j = _as_nonneg(self.j, "j")
        self.period = _as_positive(self.period, "period")

    def _k(self) -> float:
        return 8.0 * self.j / (self.capacity * self.period)

    def travel_time(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        c, t = self.capacity, self.period
        z = x / c - 1.0
        q = z * z + self._k() * x
        return self.t0 + 0.25 * t * (z + math.sqrt(max(0.0, q)))

    def derivative(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        c, t, k = self.capacity, self.period, self._k()
        z = x / c - 1.0
        q = z * z + k * x
        if q <= _EPS:
            return 0.25 * t / c                     # z = 0 and k = 0
        return 0.25 * t * (1.0 / c + (2.0 * z / c + k) / (2.0 * math.sqrt(q)))

    def integral(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        c, t, k = self.capacity, self.period, self._k()
        base = self.t0 * x + 0.25 * t * (x * x / (2.0 * c) - x)
        if k <= _EPS:                               # degenerate: sqrt(z^2) = |z|
            if x <= c:
                return self.t0 * x                  # z + |z| = 0 below capacity
            d = x - c
            return self.t0 * x + 0.25 * t * d * d / c
        p = 1.0 / (c * c)
        qc = k - 2.0 / c
        r = 1.0

        def _sq(v: float) -> float:
            w = max(0.0, p * v * v + qc * v + r)
            root = math.sqrt(w)
            lin = 2.0 * p * v + qc
            arg = 2.0 * math.sqrt(p) * root + lin
            if arg <= _EPS:
                arg = _EPS
            return (lin / (4.0 * p)) * root + ((4.0 * p * r - qc * qc)
                                               / (8.0 * p ** 1.5)) * math.log(arg)

        return base + 0.25 * t * (_sq(x) - _sq(0.0))


@dataclass
class Polynomial(VolumeDelayFunction):
    """General polynomial in the volume/capacity ratio: ``sum_k a_k (x/c)^k``.

    Coefficients must be non-negative so that ``t`` is non-decreasing (and the
    Beckmann objective therefore convex).  ``Polynomial([a0, a1], c)`` is the affine
    ``a0 + a1 x / c``; a degree-4 polynomial reproduces a BPR-like curve with an
    extra low-order term that a pure BPR cannot express.
    """

    coefficients: List[float]
    capacity: float = 1.0

    def __post_init__(self) -> None:
        self.coefficients = _as_floats(self.coefficients, "coefficients")
        if not self.coefficients:
            raise TrafficAssignmentError("Polynomial needs at least one coefficient.")
        for i, a in enumerate(self.coefficients):
            if a < 0.0:
                raise TrafficAssignmentError(
                    "Polynomial coefficient %d is negative (%g); the curve would "
                    "not be monotone." % (i, a))
        self.capacity = _as_positive(self.capacity, "capacity")

    def travel_time(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        r = x / self.capacity
        acc = 0.0
        for a in reversed(self.coefficients):       # Horner
            acc = acc * r + a
        return acc

    def derivative(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        r = x / self.capacity
        terms = [k * self.coefficients[k] * r ** (k - 1)
                 for k in range(1, len(self.coefficients))]
        return math.fsum(terms) / self.capacity

    def second_derivative(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        r = x / self.capacity
        terms = [k * (k - 1) * self.coefficients[k] * r ** (k - 2)
                 for k in range(2, len(self.coefficients))]
        return math.fsum(terms) / (self.capacity ** 2)

    def integral(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        r = x / self.capacity
        terms = [self.coefficients[k] * r ** (k + 1) / (k + 1)
                 for k in range(len(self.coefficients))]
        return self.capacity * math.fsum(terms)


@dataclass
class Linear(VolumeDelayFunction):
    """Affine delay ``t0 + slope * x`` -- the form Braess-type examples are written in.

    ``capacity`` is nominal (a reporting reference); ``with_capacity`` rescales
    ``slope`` inversely so that the curve in volume/capacity space is preserved,
    which is what makes the capacity sensitivity of an affine link meaningful.
    """

    t0: float
    slope: float = 0.0
    capacity: float = 1.0

    def __post_init__(self) -> None:
        self.t0 = _as_nonneg(self.t0, "t0")
        self.slope = _as_nonneg(self.slope, "slope")
        self.capacity = _as_positive(self.capacity, "capacity")

    def travel_time(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        return self.t0 + self.slope * x

    def derivative(self, flow: float) -> float:
        return self.slope

    def second_derivative(self, flow: float) -> float:
        return 0.0

    def integral(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        return self.t0 * x + 0.5 * self.slope * x * x

    def with_capacity(self, capacity: float) -> "Linear":
        """Rescale capacity, holding the volume/capacity-space curve fixed."""
        c_new = _as_positive(capacity, "capacity")
        return Linear(self.t0, self.slope * self.capacity / c_new, c_new)


@dataclass
class Constant(VolumeDelayFunction):
    """Flow-independent travel time ``t0`` -- uncongested links, connectors, turns.

    With every link constant the user equilibrium and the system optimum coincide
    exactly and the price of anarchy is 1; that degenerate case is a first-class
    oracle, so the zero-derivative branches of every solver must survive it.
    """

    t0: float
    capacity: float = 1.0

    def __post_init__(self) -> None:
        self.t0 = _as_nonneg(self.t0, "t0")
        self.capacity = _as_positive(self.capacity, "capacity")

    def travel_time(self, flow: float) -> float:
        return self.t0

    def derivative(self, flow: float) -> float:
        return 0.0

    def second_derivative(self, flow: float) -> float:
        return 0.0

    def integral(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        return self.t0 * x


@dataclass
class MarginalCostVDF(VolumeDelayFunction):
    """Wraps a VDF as its marginal social cost ``t(x) + x t'(x)``.

    Solving a user equilibrium on the wrapped network *is* the system optimum,
    because ``integral_0^f (t + w t') dw = f t(f)`` -- the Beckmann objective of the
    marginal-cost network is exactly total system travel time.
    """

    base: VolumeDelayFunction

    def __post_init__(self) -> None:
        if not isinstance(self.base, VolumeDelayFunction):
            raise TrafficAssignmentError(
                "MarginalCostVDF needs a VolumeDelayFunction (got %r)." % (self.base,))
        self.capacity = float(getattr(self.base, "capacity", 1.0))
        self.t0 = float(self.base.travel_time(0.0))

    def travel_time(self, flow: float) -> float:
        return self.base.marginal_cost(flow)

    def derivative(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        return 2.0 * self.base.derivative(x) + x * self.base.second_derivative(x)

    def integral(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        return x * self.base.travel_time(x)

    def with_capacity(self, capacity: float) -> "MarginalCostVDF":
        """Marginal cost of the base curve re-capacitated."""
        return MarginalCostVDF(self.base.with_capacity(capacity))


# ===========================================================================
# 2. NETWORK DATA MODEL -- links, network, OD matrix
# ===========================================================================
@dataclass
class Link:
    """A directed link: its endpoints, its volume-delay function and its toll."""

    tail: int
    head: int
    vdf: VolumeDelayFunction
    toll: float = 0.0
    length: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        self.tail = _as_index(self.tail, "tail")
        self.head = _as_index(self.head, "head")
        if not isinstance(self.vdf, VolumeDelayFunction):
            raise TrafficAssignmentError(
                "link (%d,%d) needs a VolumeDelayFunction (got %r)."
                % (self.tail, self.head, self.vdf))
        self.toll = _as_nonneg(self.toll, "toll")
        self.length = _as_nonneg(self.length, "length")
        self.name = str(self.name)

    @property
    def capacity(self) -> float:
        """Nominal capacity of the underlying volume-delay function."""
        return float(getattr(self.vdf, "capacity", 1.0))

    def travel_time(self, flow: float) -> float:
        """Running time on this link at ``flow``."""
        return self.vdf.travel_time(flow)

    def cost(self, flow: float) -> float:
        """Generalised cost -- running time plus the link toll."""
        return self.vdf.travel_time(flow) + self.toll

    def label(self, index: int) -> str:
        """Display label: the given name, else ``#i (tail->head)``."""
        if self.name:
            return self.name
        return "#%d (%d->%d)" % (index, self.tail, self.head)


class Network:
    """A directed road network: an ordered list of :class:`Link` plus adjacency.

    Link *index* is insertion order and is the index into every flow / cost vector
    this module produces.  Nodes are integers; adding a link with node id ``k``
    silently grows the node set to ``k+1``.
    """

    def __init__(self, name: str = "network") -> None:
        self.name = str(name)
        self.links: List[Link] = []
        self._n_nodes = 0
        self._out: List[List[Tuple[int, int]]] = []
        self._in: List[List[Tuple[int, int]]] = []

    # -- construction ------------------------------------------------------
    def ensure_nodes(self, n_nodes: int) -> None:
        """Grow the node set so that ids ``0..n_nodes-1`` exist."""
        n = _as_index(n_nodes, "n_nodes")
        while self._n_nodes < n:
            self._out.append([])
            self._in.append([])
            self._n_nodes += 1

    def add_link(self, tail: int, head: int, vdf: VolumeDelayFunction, *,
                 toll: float = 0.0, length: float = 0.0,
                 name: str = "") -> int:
        """Append a directed link and return its index."""
        link = Link(tail, head, vdf, toll, length, name)
        if link.tail == link.head:
            raise TrafficAssignmentError(
                "self-loop at node %d is not a routable link." % link.tail)
        self.ensure_nodes(max(link.tail, link.head) + 1)
        idx = len(self.links)
        self.links.append(link)
        self._out[link.tail].append((idx, link.head))
        self._in[link.head].append((idx, link.tail))
        return idx

    # -- shape -------------------------------------------------------------
    @property
    def n_links(self) -> int:
        """Number of directed links."""
        return len(self.links)

    @property
    def n_nodes(self) -> int:
        """Number of nodes (ids ``0..n_nodes-1``)."""
        return self._n_nodes

    def out_links(self, node: int) -> List[Tuple[int, int]]:
        """Pairs (link index, head node) leaving ``node``."""
        return self._out[node]

    def in_links(self, node: int) -> List[Tuple[int, int]]:
        """Pairs (link index, tail node) entering ``node``."""
        return self._in[node]

    def zeros(self) -> Flow:
        """A zero flow vector of the right length."""
        return [0.0] * len(self.links)

    # -- link vectors ------------------------------------------------------
    def travel_times(self, flows: Sequence[float]) -> Cost:
        """Running time on every link at ``flows`` (tolls excluded)."""
        return [lk.vdf.travel_time(flows[i]) for i, lk in enumerate(self.links)]

    def costs(self, flows: Sequence[float]) -> Cost:
        """Generalised cost on every link -- running time plus toll."""
        return [lk.vdf.travel_time(flows[i]) + lk.toll
                for i, lk in enumerate(self.links)]

    def derivatives(self, flows: Sequence[float]) -> Cost:
        """First derivative on every link -- the Beckmann Hessian diagonal."""
        return [lk.vdf.derivative(flows[i]) for i, lk in enumerate(self.links)]

    def marginal_costs(self, flows: Sequence[float]) -> Cost:
        """Marginal social cost plus toll on every link."""
        return [lk.vdf.marginal_cost(flows[i]) + lk.toll
                for i, lk in enumerate(self.links)]

    def capacities(self) -> List[float]:
        """Nominal capacity of every link."""
        return [lk.capacity for lk in self.links]

    def free_flow_times(self) -> List[float]:
        """Zero-flow travel time of every link."""
        return [lk.vdf.travel_time(0.0) for lk in self.links]

    def tolls(self) -> List[float]:
        """Toll on every link."""
        return [lk.toll for lk in self.links]

    def vc_ratios(self, flows: Sequence[float]) -> List[float]:
        """Volume/capacity ratio of every link."""
        return [flows[i] / lk.capacity for i, lk in enumerate(self.links)]

    # -- transformations ---------------------------------------------------
    def copy(self) -> "Network":
        """A structural copy sharing the (immutable-by-convention) VDF objects."""
        out = Network(self.name)
        out.ensure_nodes(self._n_nodes)
        for lk in self.links:
            out.add_link(lk.tail, lk.head, lk.vdf, toll=lk.toll,
                         length=lk.length, name=lk.name)
        return out

    def with_tolls(self, tolls: Sequence[float]) -> "Network":
        """A copy whose link tolls are replaced by ``tolls``."""
        vals = _as_floats(tolls, "tolls")
        if len(vals) != len(self.links):
            raise TrafficAssignmentError(
                "tolls has %d entries but the network has %d links."
                % (len(vals), len(self.links)))
        out = Network(self.name + "+tolls")
        out.ensure_nodes(self._n_nodes)
        for i, lk in enumerate(self.links):
            out.add_link(lk.tail, lk.head, lk.vdf, toll=max(0.0, vals[i]),
                         length=lk.length, name=lk.name)
        return out

    def with_marginal_costs(self) -> "Network":
        """A copy whose VDFs are wrapped as marginal social costs (the SO network)."""
        out = Network(self.name + "+marginal")
        out.ensure_nodes(self._n_nodes)
        for lk in self.links:
            out.add_link(lk.tail, lk.head, MarginalCostVDF(lk.vdf), toll=lk.toll,
                         length=lk.length, name=lk.name)
        return out

    def without_link(self, index: int) -> "Network":
        """A copy with link ``index`` deleted; later link indices shift down by one."""
        if not 0 <= index < len(self.links):
            raise TrafficAssignmentError("link index %r out of range." % (index,))
        out = Network(self.name + "-link%d" % index)
        out.ensure_nodes(self._n_nodes)
        for i, lk in enumerate(self.links):
            if i == index:
                continue
            out.add_link(lk.tail, lk.head, lk.vdf, toll=lk.toll,
                         length=lk.length, name=lk.name)
        return out

    def with_scaled_capacity(self, index: int, factor: float) -> "Network":
        """A copy with link ``index``'s capacity multiplied by ``factor``."""
        if not 0 <= index < len(self.links):
            raise TrafficAssignmentError("link index %r out of range." % (index,))
        f = _as_positive(factor, "factor")
        out = Network(self.name + "*cap")
        out.ensure_nodes(self._n_nodes)
        for i, lk in enumerate(self.links):
            vdf = lk.vdf.with_capacity(lk.capacity * f) if i == index else lk.vdf
            out.add_link(lk.tail, lk.head, vdf, toll=lk.toll,
                         length=lk.length, name=lk.name)
        return out

    # -- queries -----------------------------------------------------------
    def reachable(self, origin: int) -> Set[int]:
        """Every node reachable from ``origin`` by directed links."""
        if not 0 <= origin < self._n_nodes:
            raise TrafficAssignmentError("node %r out of range." % (origin,))
        seen = {origin}
        stack = [origin]
        while stack:
            u = stack.pop()
            for (_e, v) in self._out[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    def link_index(self, tail: int, head: int) -> int:
        """Index of the first link from ``tail`` to ``head`` (-1 if there is none)."""
        if not 0 <= tail < self._n_nodes:
            return -1
        for (e, v) in self._out[tail]:
            if v == head:
                return e
        return -1

    def __repr__(self) -> str:
        return "Network(%r, nodes=%d, links=%d)" % (
            self.name, self._n_nodes, len(self.links))


class ODMatrix:
    """A sparse origin-destination trip table mapping (o, d) to trips."""

    def __init__(self, demands: Optional[Demand] = None) -> None:
        self._d: Demand = {}
        if demands:
            for (o, d), v in dict(demands).items():
                self.set(o, d, v)

    def set(self, origin: int, destination: int, trips: float) -> None:
        """Set (or clear, when ``trips`` is 0) the demand for one OD pair."""
        o = _as_index(origin, "origin")
        d = _as_index(destination, "destination")
        v = _as_nonneg(trips, "trips")
        if o == d:
            if v > 0.0:
                raise TrafficAssignmentError(
                    "intrazonal demand (%d,%d)=%g is not assignable." % (o, d, v))
            return
        if v <= 0.0:
            self._d.pop((o, d), None)
        else:
            self._d[(o, d)] = v

    def get(self, origin: int, destination: int) -> float:
        """Demand for one OD pair (0 when absent)."""
        return self._d.get((int(origin), int(destination)), 0.0)

    def pairs(self) -> List[Tuple[int, int]]:
        """OD pairs in a deterministic (sorted) order."""
        return sorted(self._d)

    def items(self) -> List[Tuple[Tuple[int, int], float]]:
        """OD pair / trips tuples in a deterministic order."""
        return [(k, self._d[k]) for k in sorted(self._d)]

    def origins(self) -> List[int]:
        """Distinct origins, sorted."""
        return sorted({o for (o, _d) in self._d})

    def destinations(self) -> List[int]:
        """Distinct destinations, sorted."""
        return sorted({d for (_o, d) in self._d})

    def by_origin(self) -> Dict[int, List[Tuple[int, float]]]:
        """Origin -> list of (destination, trips), in a deterministic order."""
        out: Dict[int, List[Tuple[int, float]]] = {}
        for (o, d) in sorted(self._d):
            out.setdefault(o, []).append((d, self._d[(o, d)]))
        return out

    def total(self) -> float:
        """Total trips in the table."""
        return math.fsum(self._d[k] for k in sorted(self._d))

    def scaled(self, factor: float) -> "ODMatrix":
        """A copy with every cell multiplied by ``factor``."""
        f = _as_nonneg(factor, "factor")
        return ODMatrix({k: v * f for k, v in self._d.items()})

    def copy(self) -> "ODMatrix":
        """An independent copy."""
        return ODMatrix(dict(self._d))

    def as_dict(self) -> Demand:
        """A copy of the underlying dictionary."""
        return dict(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        return iter(sorted(self._d))

    def __repr__(self) -> str:
        return "ODMatrix(%d pairs, %g trips)" % (len(self._d), self.total())


def _as_od(od: Any) -> ODMatrix:
    """Accept an ``ODMatrix`` or a plain (o, d) -> trips dict."""
    if isinstance(od, ODMatrix):
        return od
    if isinstance(od, dict):
        return ODMatrix(od)
    raise TrafficAssignmentError(
        "demand must be an ODMatrix or an (o,d)->trips dict (got %r)." % (od,))


def _check_od(network: Network, od: ODMatrix) -> None:
    """Every OD pair must reference real nodes and have a reachable destination."""
    n = network.n_nodes
    for (o, d), v in od.items():
        if not (0 <= o < n and 0 <= d < n):
            raise TrafficAssignmentError(
                "OD pair (%d,%d) references a node outside 0..%d." % (o, d, n - 1))
    by_o = od.by_origin()
    for o in od.origins():
        reach = network.reachable(o)
        for (d, trips) in by_o[o]:
            if trips > 0.0 and d not in reach:
                raise TrafficAssignmentError(
                    "destination %d is unreachable from origin %d; %g trips could "
                    "not be assigned." % (d, o, trips))


# ===========================================================================
# 3. SHORTEST PATHS, PATHS AND PATH SETS
# ===========================================================================
@dataclass
class ShortestPathTree:
    """One-to-all shortest-path tree: distances and the predecessor link per node."""

    origin: int
    dist: List[float]
    pred_link: List[int]
    tails: Tuple[int, ...] = ()

    def reached(self, node: int) -> bool:
        """Whether ``node`` is reachable from the origin."""
        return self.dist[node] < _INF

    def bind(self, network: Network) -> "ShortestPathTree":
        """Attach the network's tail array so that ``path_links`` can walk back."""
        self.tails = tuple(lk.tail for lk in network.links)
        return self

    def path_links(self, node: int) -> LinkSeq:
        """Link indices of the tree path origin -> ``node`` (empty for the origin)."""
        out: LinkSeq = []
        cur = int(node)
        guard = 0
        while cur != self.origin:
            e = self.pred_link[cur]
            if e < 0:
                raise TrafficAssignmentError(
                    "node %d is not reachable from origin %d." % (node, self.origin))
            out.append(e)
            cur = self.tails[e]
            guard += 1
            if guard > len(self.pred_link) + 1:
                raise TrafficAssignmentError("cycle in the shortest-path tree.")
        out.reverse()
        return out

    def path_nodes(self, network: Network, node: int) -> NodeSeq:
        """Node sequence of the tree path origin -> ``node``."""
        links = self.path_links(node)
        nodes = [self.origin]
        for e in links:
            nodes.append(network.links[e].head)
        return nodes


def _dijkstra(network: Network, origin: int,
              costs: Sequence[float]) -> Tuple[List[float], List[int]]:
    """Binary-heap Dijkstra from ``origin`` -- the inner loop of every solver."""
    n = network.n_nodes
    dist = [_INF] * n
    pred = [-1] * n
    dist[origin] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, origin)]
    out = network._out
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u] + _EPS * (dist[u] if dist[u] < _INF else 0.0):
            continue
        for (e, v) in out[u]:
            nd = d + costs[e]
            # Relative, not absolute: link costs carry units.  A bare ``- 1e-15``
            # rejects every genuine improvement once the cost scale drops near it,
            # and Dijkstra then returns a distance that is wrong in the 4th
            # significant figure while looking perfectly converged.
            if nd < dist[v] - 1e-15 * (dist[v] if dist[v] < _INF else 0.0):
                dist[v] = nd
                pred[v] = e
                heapq.heappush(heap, (nd, v))
    return dist, pred


def _tree_order(network: Network, dist: Sequence[float], pred: Sequence[int],
                origin: int) -> List[int]:
    """Nodes of a shortest-path tree with every child before its parent.

    Sorting by descending distance is *not* safe: a zero-cost link (a connector, a
    free turn) makes ``dist[child] == dist[parent]`` and a stable sort can then
    visit the parent first, stranding the load it was supposed to forward.  Walking
    the tree itself is both exact and linear.
    """
    n = network.n_nodes
    children: List[List[int]] = [[] for _ in range(n)]
    tails = network.links
    for v in range(n):
        e = pred[v]
        if e >= 0 and dist[v] < _INF:
            children[tails[e].tail].append(v)
    visit: List[int] = []
    stack = [origin]
    seen = [False] * n
    seen[origin] = True
    while stack:
        u = stack.pop()
        visit.append(u)
        for v in children[u]:
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    visit.reverse()
    return visit


def _check_nonneg_costs(costs: Sequence[float]) -> None:
    """Dijkstra needs non-negative arc costs; say so loudly if they are not."""
    for i, c in enumerate(costs):
        if c < -1e-9:
            raise TrafficAssignmentError(
                "link %d has a negative cost (%g); shortest paths would be wrong."
                % (i, c))


def shortest_path_tree(network: Network, origin: int,
                       costs: Sequence[float]) -> ShortestPathTree:
    """One-to-all shortest-path tree from ``origin`` under link ``costs``."""
    if not 0 <= origin < network.n_nodes:
        raise TrafficAssignmentError("origin %r out of range." % (origin,))
    c = _as_floats(costs, "costs")
    if len(c) != network.n_links:
        raise TrafficAssignmentError(
            "costs has %d entries but the network has %d links."
            % (len(c), network.n_links))
    _check_nonneg_costs(c)
    dist, pred = _dijkstra(network, origin, c)
    return ShortestPathTree(origin, dist, pred).bind(network)


def shortest_path(network: Network, origin: int, destination: int,
                  costs: Sequence[float]) -> Tuple[float, NodeSeq, LinkSeq]:
    """Cheapest path (cost, nodes, links) from ``origin`` to ``destination``."""
    tree = shortest_path_tree(network, origin, costs)
    if not tree.reached(destination):
        raise TrafficAssignmentError(
            "destination %d is unreachable from origin %d." % (destination, origin))
    links = tree.path_links(destination)
    nodes = tree.path_nodes(network, destination)
    return tree.dist[destination], nodes, links


@dataclass
class Path:
    """A route through the network: its links, its flow and its last-known cost."""

    links: Tuple[int, ...]
    nodes: Tuple[int, ...] = ()
    flow: float = 0.0
    cost: float = 0.0
    origin: int = -1
    destination: int = -1

    def __post_init__(self) -> None:
        self.links = tuple(int(e) for e in self.links)
        self.nodes = tuple(int(v) for v in self.nodes)
        self.flow = float(self.flow)
        self.cost = float(self.cost)

    def key(self) -> Tuple[int, ...]:
        """Hashable identity of the route (its link sequence)."""
        return self.links

    def length(self, network: Network) -> float:
        """Physical length of the route."""
        return math.fsum(network.links[e].length for e in self.links)

    def uses_link(self, link_index: int) -> bool:
        """Whether the route traverses ``link_index``."""
        return link_index in self.links


def path_cost(network: Network, path: Any, costs: Sequence[float]) -> float:
    """Generalised cost of a route (a :class:`Path` or a bare link sequence)."""
    links = path.links if isinstance(path, Path) else tuple(int(e) for e in path)
    for e in links:
        if not 0 <= e < network.n_links:
            raise TrafficAssignmentError("link index %r out of range." % (e,))
    return math.fsum(costs[e] for e in links)


class PathSet:
    """The route choice set: routes and their flows, grouped by OD pair."""

    def __init__(self) -> None:
        self._paths: Dict[Tuple[int, int], List[Path]] = {}

    def add(self, origin: int, destination: int, links: Sequence[int],
            flow: float = 0.0, nodes: Sequence[int] = ()) -> Path:
        """Add a route (or top up an identical existing one) for an OD pair."""
        key = (int(origin), int(destination))
        links_t = tuple(int(e) for e in links)
        bucket = self._paths.setdefault(key, [])
        for p in bucket:
            if p.links == links_t:
                p.flow += float(flow)
                return p
        p = Path(links_t, tuple(int(v) for v in nodes), float(flow), 0.0,
                 key[0], key[1])
        bucket.append(p)
        return p

    def has(self, origin: int, destination: int, links: Sequence[int]) -> bool:
        """Whether this route is already in the choice set."""
        links_t = tuple(int(e) for e in links)
        return any(p.links == links_t
                   for p in self._paths.get((int(origin), int(destination)), []))

    def paths(self, origin: int, destination: int) -> List[Path]:
        """Routes stored for one OD pair (the live list)."""
        return self._paths.setdefault((int(origin), int(destination)), [])

    def pairs(self) -> List[Tuple[int, int]]:
        """OD pairs with at least one route, sorted."""
        return sorted(self._paths)

    def all_paths(self) -> List[Path]:
        """Every route in the set, OD-sorted then insertion-ordered."""
        out: List[Path] = []
        for key in sorted(self._paths):
            out.extend(self._paths[key])
        return out

    def n_paths(self) -> int:
        """Total number of routes."""
        return sum(len(v) for v in self._paths.values())

    def prune(self, tol: float = 1e-12) -> int:
        """Drop zero-flow routes (keeping one per OD pair); return how many went.

        ``tol`` is *relative to the flow already loaded on that OD pair*, not an
        absolute veh/h literal.  An absolute threshold deletes genuinely used routes
        -- and silently loses their vehicles -- from any trip table quoted in small
        units, which is a unit choice and not a modelling statement.
        """
        removed = 0
        for key in list(self._paths):
            bucket = self._paths[key]
            scale = math.fsum(p.flow for p in bucket if p.flow > 0.0)
            cut = tol * scale
            keep = [p for p in bucket if p.flow > cut]
            if not keep and bucket:
                keep = [max(bucket, key=lambda p: p.flow)]
            removed += len(bucket) - len(keep)
            self._paths[key] = keep
        return removed

    def link_flows(self, network: Network) -> Flow:
        """Aggregate the route flows onto links."""
        flows = network.zeros()
        for p in self.all_paths():
            if p.flow == 0.0:
                continue
            for e in p.links:
                flows[e] += p.flow
        return flows

    def update_costs(self, network: Network, costs: Sequence[float]) -> None:
        """Refresh every stored route cost from a link-cost vector."""
        for p in self.all_paths():
            p.cost = path_cost(network, p, costs)

    def demand(self, origin: int, destination: int) -> float:
        """Total route flow currently loaded on one OD pair."""
        return math.fsum(p.flow for p in self.paths(origin, destination))

    def __len__(self) -> int:
        return self.n_paths()

    def __repr__(self) -> str:
        return "PathSet(%d pairs, %d paths)" % (len(self._paths), self.n_paths())


def enumerate_paths(network: Network, origin: int, destination: int, *,
                    max_paths: int = 100, max_links: int = 12,
                    max_cost: Optional[float] = None,
                    costs: Optional[Sequence[float]] = None) -> List[LinkSeq]:
    """Every simple route origin -> destination, depth-first, bounded, cost-sorted."""
    if not (0 <= origin < network.n_nodes and 0 <= destination < network.n_nodes):
        raise TrafficAssignmentError("origin/destination out of range.")
    limit = _as_index(max_paths, "max_paths")
    depth = _as_index(max_links, "max_links")
    c = list(costs) if costs is not None else network.free_flow_times()
    found: List[Tuple[float, LinkSeq]] = []
    visited = {origin}
    stack_links: LinkSeq = []

    def _dfs(node: int, acc: float) -> None:
        if len(found) >= limit:
            return
        if node == destination:
            found.append((acc, list(stack_links)))
            return
        if len(stack_links) >= depth:
            return
        for (e, v) in network.out_links(node):
            if v in visited:
                continue
            nxt = acc + c[e]
            if max_cost is not None and nxt > max_cost:
                continue
            visited.add(v)
            stack_links.append(e)
            _dfs(v, nxt)
            stack_links.pop()
            visited.discard(v)

    _dfs(origin, 0.0)
    found.sort(key=lambda t: (t[0], t[1]))
    return [links for (_cost, links) in found]


def k_shortest_paths(network: Network, origin: int, destination: int,
                     costs: Sequence[float], k: int = 3
                     ) -> List[Tuple[float, LinkSeq]]:
    """Yen's algorithm: the ``k`` cheapest loopless routes, cheapest first."""
    kk = _as_index(k, "k")
    if kk == 0:
        return []
    c = _as_floats(costs, "costs")
    _check_nonneg_costs(c)
    cost0, _nodes0, links0 = shortest_path(network, origin, destination, c)
    accepted: List[Tuple[float, LinkSeq]] = [(cost0, links0)]
    candidates: List[Tuple[float, LinkSeq]] = []
    tails = [lk.tail for lk in network.links]
    heads = [lk.head for lk in network.links]
    big = 1e18
    while len(accepted) < kk:
        prev = accepted[-1][1]
        for i in range(len(prev)):
            spur_node = tails[prev[i]]
            root = prev[:i]
            masked = list(c)
            for (_pc, p) in accepted:
                if list(p[:i]) == list(root) and len(p) > i:
                    masked[p[i]] = big
            root_nodes = {origin}
            for e in root:
                root_nodes.add(heads[e])
            root_nodes.discard(spur_node)
            for e in range(network.n_links):
                if tails[e] in root_nodes or heads[e] in root_nodes:
                    masked[e] = big
            dist, pred = _dijkstra(network, spur_node, masked)
            if dist[destination] >= big / 2.0:
                continue
            tree = ShortestPathTree(spur_node, dist, pred).bind(network)
            spur = tree.path_links(destination)
            total = list(root) + list(spur)
            if any(list(p) == total for (_pc, p) in accepted):
                continue
            if any(list(p) == total for (_pc, p) in candidates):
                continue
            candidates.append((math.fsum(c[e] for e in total), total))
        if not candidates:
            break
        candidates.sort(key=lambda t: (t[0], t[1]))
        accepted.append(candidates.pop(0))
    return accepted


# ===========================================================================
# 4. OBJECTIVES AND CONVERGENCE MEASURES
# ===========================================================================
def beckmann_objective(network: Network, flows: Sequence[float], *,
                       include_tolls: bool = True) -> float:
    """Beckmann's transformation ``sum_e integral_0^{f_e} t_e(w) dw`` (+ toll revenue).

    Minimising this convex function over the feasible flow polytope *is* the user
    equilibrium: its gradient is the link cost vector, so the KKT conditions are
    exactly Wardrop's first principle.
    """
    f = _check_flows(network, flows)
    terms = [lk.vdf.integral(f[i]) for i, lk in enumerate(network.links)]
    if include_tolls:
        terms.extend(lk.toll * f[i] for i, lk in enumerate(network.links)
                     if lk.toll)
    return math.fsum(terms)


def beckmann_gradient(network: Network, flows: Sequence[float]) -> Cost:
    """Gradient of the Beckmann objective -- the generalised link cost vector."""
    f = _check_flows(network, flows)
    return network.costs(f)


def total_system_travel_time(network: Network, flows: Sequence[float]) -> float:
    """Total system travel time ``sum_e f_e t_e(f_e)`` -- tolls excluded on purpose."""
    f = _check_flows(network, flows)
    return math.fsum(f[i] * lk.vdf.travel_time(f[i])
                     for i, lk in enumerate(network.links))


def _gap_parts(network: Network, flows: Sequence[float], od: ODMatrix,
               costs: Optional[Sequence[float]] = None) -> Tuple[float, float, float]:
    """(loaded generalised cost, shortest-path travel time, total demand)."""
    c = list(costs) if costs is not None else network.costs(flows)
    loaded = math.fsum(flows[e] * c[e] for e in range(network.n_links))
    sptt = 0.0
    for o, dests in od.by_origin().items():
        dist, _pred = _dijkstra(network, o, c)
        for (d, trips) in dests:
            if dist[d] >= _INF:
                raise TrafficAssignmentError(
                    "destination %d unreachable from origin %d." % (d, o))
            sptt += trips * dist[d]
    return loaded, sptt, od.total()


def relative_gap(network: Network, flows: Sequence[float], od: Any,
                 costs: Optional[Sequence[float]] = None) -> float:
    """Relative gap ``(f.c - sum_od d*u) / (f.c)`` -- 0 exactly at equilibrium."""
    odm = _as_od(od)
    f = _check_flows(network, flows)
    loaded, sptt, _total = _gap_parts(network, f, odm, costs)
    if loaded <= _EPS:
        return 0.0
    return max(0.0, (loaded - sptt) / loaded)


def average_excess_cost(network: Network, flows: Sequence[float], od: Any,
                        costs: Optional[Sequence[float]] = None) -> float:
    """Average excess cost: surplus generalised cost per trip over the shortest path."""
    odm = _as_od(od)
    f = _check_flows(network, flows)
    loaded, sptt, total = _gap_parts(network, f, odm, costs)
    if total <= _EPS:
        return 0.0
    return max(0.0, (loaded - sptt) / total)


def max_flow_change(previous: Sequence[float], current: Sequence[float]) -> float:
    """Largest absolute link-flow change between two iterates."""
    if len(previous) != len(current):
        raise TrafficAssignmentError("flow vectors have different lengths.")
    if not previous:
        return 0.0
    return max(abs(a - b) for a, b in zip(previous, current))


@dataclass
class IterationRecord:
    """One row of the convergence log."""

    iteration: int
    objective: float
    relative_gap: float
    average_excess_cost: float
    total_travel_time: float
    max_flow_change: float = 0.0
    step_size: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        """JSON-able camelCase view of this row."""
        return {"iteration": self.iteration, "objective": self.objective,
                "relativeGap": self.relative_gap,
                "averageExcessCost": self.average_excess_cost,
                "totalTravelTime": self.total_travel_time,
                "maxFlowChange": self.max_flow_change,
                "stepSize": self.step_size}


@dataclass
class AssignmentResult:
    """Everything an assignment run produced: flows, costs and the convergence log."""

    method: str
    flows: Flow
    travel_times: Cost
    costs: Cost
    objective: float
    total_travel_time: float
    relative_gap: float
    average_excess_cost: float
    iterations: int
    converged: bool
    history: List[IterationRecord] = field(default_factory=list)
    origin_flows: Optional[Dict[int, Flow]] = None
    paths: Optional[PathSet] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def vc_ratios(self, network: Network) -> List[float]:
        """Volume/capacity ratio of every link at the solution."""
        return network.vc_ratios(self.flows)

    def objective_trace(self) -> List[float]:
        """The Beckmann objective per iteration -- must not increase for Frank-Wolfe."""
        return [r.objective for r in self.history]

    def gap_trace(self) -> List[float]:
        """The relative gap per iteration."""
        return [r.relative_gap for r in self.history]

    def summary(self) -> str:
        """One-line human summary."""
        return ("%-28s iters=%-4d gap=%.3e aec=%.3e Z=%.6f TSTT=%.6f%s"
                % (self.method, self.iterations, self.relative_gap,
                   self.average_excess_cost, self.objective,
                   self.total_travel_time,
                   "" if self.converged else "  [NOT CONVERGED]"))


# ===========================================================================
# 5. LINK-BASED LOADING -- all-or-nothing, incremental, MSA
# ===========================================================================
@dataclass
class AllOrNothingResult:
    """An all-or-nothing loading: every trip on its cheapest route."""

    flows: Flow
    shortest_path_costs: Dict[Tuple[int, int], float]
    sptt: float
    origin_flows: Optional[Dict[int, Flow]] = None

    def total_cost(self) -> float:
        """The shortest-path travel time, i.e. the lower bound used by every gap."""
        return self.sptt


def _aon_core(network: Network, by_origin: Dict[int, List[Tuple[int, float]]],
              costs: Sequence[float], want_origin_flows: bool = False
              ) -> Tuple[Flow, float, Dict[Tuple[int, int], float],
                         Optional[Dict[int, Flow]]]:
    """The hot loop: one Dijkstra per origin, then a reverse cascade load."""
    m = network.n_links
    n = network.n_nodes
    flows = [0.0] * m
    sp_costs: Dict[Tuple[int, int], float] = {}
    origin_flows: Optional[Dict[int, Flow]] = {} if want_origin_flows else None
    sptt_terms: List[float] = []
    tails = [lk.tail for lk in network.links]
    for o in sorted(by_origin):
        dist, pred = _dijkstra(network, o, costs)
        load = [0.0] * n
        for (d, trips) in by_origin[o]:
            if dist[d] >= _INF:
                raise TrafficAssignmentError(
                    "destination %d unreachable from origin %d." % (d, o))
            sp_costs[(o, d)] = dist[d]
            if trips > 0.0:
                load[d] += trips
                sptt_terms.append(trips * dist[d])
        order = _tree_order(network, dist, pred, o)
        of = [0.0] * m if want_origin_flows else None
        for node in order:
            amt = load[node]
            if amt <= 0.0 or node == o:
                continue
            e = pred[node]
            if e < 0:
                raise TrafficAssignmentError(
                    "node %d has no predecessor from origin %d." % (node, o))
            flows[e] += amt
            if of is not None:
                of[e] += amt
            load[tails[e]] += amt
            load[node] = 0.0
        if origin_flows is not None and of is not None:
            origin_flows[o] = of
    return flows, math.fsum(sptt_terms), sp_costs, origin_flows


def all_or_nothing(network: Network, od: Any, costs: Sequence[float], *,
                   keep_origin_flows: bool = False) -> AllOrNothingResult:
    """Load every OD trip on its cheapest route under fixed link ``costs``."""
    odm = _as_od(od)
    _check_od(network, odm)
    c = _as_floats(costs, "costs")
    if len(c) != network.n_links:
        raise TrafficAssignmentError(
            "costs has %d entries but the network has %d links."
            % (len(c), network.n_links))
    _check_nonneg_costs(c)
    flows, sptt, sp_costs, of = _aon_core(network, odm.by_origin(), c,
                                          keep_origin_flows)
    return AllOrNothingResult(flows, sp_costs, sptt, of)


def _finalise(network: Network, odm: ODMatrix, flows: Flow, method: str,
              history: List[IterationRecord], iterations: int, converged: bool,
              origin_flows: Optional[Dict[int, Flow]] = None,
              paths: Optional[PathSet] = None,
              extra: Optional[Dict[str, Any]] = None) -> AssignmentResult:
    """Assemble the result object and its final convergence statistics."""
    costs = network.costs(flows)
    loaded, sptt, total = _gap_parts(network, flows, odm, costs)
    rgap = max(0.0, (loaded - sptt) / loaded) if loaded > _EPS else 0.0
    aec = max(0.0, (loaded - sptt) / total) if total > _EPS else 0.0
    return AssignmentResult(
        method=method, flows=flows, travel_times=network.travel_times(flows),
        costs=costs, objective=beckmann_objective(network, flows),
        total_travel_time=total_system_travel_time(network, flows),
        relative_gap=rgap, average_excess_cost=aec, iterations=iterations,
        converged=converged, history=history, origin_flows=origin_flows,
        paths=paths, extra=dict(extra or {}))


def incremental_assignment(network: Network, od: Any, *, n_increments: int = 10,
                           weights: Optional[Sequence[float]] = None,
                           keep_origin_flows: bool = False) -> AssignmentResult:
    """Load the demand in slices, re-timing the network after each one.

    Cheap, deterministic and historically popular, but it does **not** converge to
    the user equilibrium -- the first slice is loaded on free-flow times and is
    never allowed to move again.  Its residual gap is a useful yardstick for how
    much an equilibrium solver actually buys you.
    """
    odm = _as_od(od)
    _check_od(network, odm)
    if weights is None:
        k = _as_index(n_increments, "n_increments")
        if k <= 0:
            raise TrafficAssignmentError("n_increments must be >= 1.")
        w = [1.0 / k] * k
    else:
        w = _as_floats(weights, "weights")
        if not w or any(x < 0 for x in w):
            raise TrafficAssignmentError("weights must be non-negative and non-empty.")
        s = math.fsum(w)
        if s <= 0.0:
            raise TrafficAssignmentError("weights must sum to something positive.")
        w = [x / s for x in w]
    flows = network.zeros()
    origin_flows: Optional[Dict[int, Flow]] = ({} if keep_origin_flows else None)
    history: List[IterationRecord] = []
    for i, frac in enumerate(w):
        costs = network.costs(flows)
        slice_od = odm.scaled(frac)
        y, _sptt, _sp, of = _aon_core(network, slice_od.by_origin(), costs,
                                      keep_origin_flows)
        flows = [flows[e] + y[e] for e in range(network.n_links)]
        if origin_flows is not None and of is not None:
            for o, vec in of.items():
                acc = origin_flows.setdefault(o, network.zeros())
                for e in range(network.n_links):
                    acc[e] += vec[e]
        c2 = network.costs(flows)
        loaded, sptt, total = _gap_parts(network, flows, odm, c2)
        history.append(IterationRecord(
            iteration=i, objective=beckmann_objective(network, flows),
            relative_gap=max(0.0, (loaded - sptt) / loaded) if loaded > _EPS else 0.0,
            average_excess_cost=max(0.0, (loaded - sptt) / total)
            if total > _EPS else 0.0,
            total_travel_time=total_system_travel_time(network, flows),
            max_flow_change=max(y) if y else 0.0, step_size=frac))
    return _finalise(network, odm, flows, "incremental(%d)" % len(w), history,
                     len(w), False, origin_flows)


# ===========================================================================
# 6. FRANK-WOLFE FAMILY -- FW, conjugate FW, biconjugate FW, MSA
# ===========================================================================
def _phi_prime(network: Network, flows: Sequence[float],
               direction: Sequence[float], lam: float,
               active: Sequence[int]) -> float:
    """Derivative of the Beckmann objective along ``direction`` at step ``lam``."""
    terms = []
    for e in active:
        d = direction[e]
        x = flows[e] + lam * d
        if x < 0.0:
            x = 0.0
        lk = network.links[e]
        terms.append(d * (lk.vdf.travel_time(x) + lk.toll))
    return math.fsum(terms)


def _phi(network: Network, flows: Sequence[float], direction: Sequence[float],
         lam: float, active: Sequence[int]) -> float:
    """The Beckmann objective restricted to the ray ``flows + lam*direction``."""
    terms = []
    for e in active:
        x = flows[e] + lam * direction[e]
        if x < 0.0:
            x = 0.0
        lk = network.links[e]
        terms.append(lk.vdf.integral(x) + lk.toll * x)
    return math.fsum(terms)


def exact_line_search(network: Network, flows: Sequence[float],
                      direction: Sequence[float], *, lo: float = 0.0,
                      hi: float = 1.0, method: str = "bisect",
                      iterations: int = _BISECT_ITERS,
                      tol: float = 1e-14) -> float:
    """Exact 1-D minimiser of the convex Beckmann restriction along ``direction``.

    ``method="bisect"`` brackets the root of the (monotone) directional derivative;
    ``method="golden"`` runs a golden-section search on the objective itself.  Both
    are exact to machine precision on a strictly convex restriction.
    """
    if len(flows) != network.n_links or len(direction) != network.n_links:
        raise TrafficAssignmentError("flow/direction length mismatch.")
    a = _as_float(lo, "lo")
    b = _as_float(hi, "hi")
    if b <= a:
        return a
    active = [e for e in range(network.n_links) if direction[e] != 0.0]
    if not active:
        return a
    if method == "golden":
        inv = (math.sqrt(5.0) - 1.0) / 2.0
        c1 = b - inv * (b - a)
        c2 = a + inv * (b - a)
        f1 = _phi(network, flows, direction, c1, active)
        f2 = _phi(network, flows, direction, c2, active)
        for _ in range(max(1, iterations)):
            if b - a < tol:
                break
            if f1 <= f2:
                b, c2, f2 = c2, c1, f1
                c1 = b - inv * (b - a)
                f1 = _phi(network, flows, direction, c1, active)
            else:
                a, c1, f1 = c1, c2, f2
                c2 = a + inv * (b - a)
                f2 = _phi(network, flows, direction, c2, active)
        lam = 0.5 * (a + b)
        # the endpoints can still win on a monotone restriction
        cands = [(_phi(network, flows, direction, v, active), v)
                 for v in (lo, lam, hi)]
        cands.sort()
        return cands[0][1]
    if method != "bisect":
        raise TrafficAssignmentError(
            "unknown line-search method %r (use 'bisect' or 'golden')." % (method,))
    da = _phi_prime(network, flows, direction, a, active)
    if da >= 0.0:
        return a
    db = _phi_prime(network, flows, direction, b, active)
    if db <= 0.0:
        return b
    for _ in range(max(1, iterations)):
        mid = 0.5 * (a + b)
        if b - a < tol:
            break
        if _phi_prime(network, flows, direction, mid, active) > 0.0:
            b = mid
        else:
            a = mid
    return 0.5 * (a + b)


def _hessian_inner(hess: Sequence[float], u: Sequence[float],
                   v: Sequence[float]) -> float:
    """The H-inner product ``u^T diag(hess) v`` used by the conjugate directions."""
    return math.fsum(hess[e] * u[e] * v[e] for e in range(len(hess))
                     if u[e] != 0.0 and v[e] != 0.0)


def _solve3(a: List[List[float]], rhs: List[float]) -> Optional[List[float]]:
    """Cramer's rule on a 3x3 system; ``None`` when it is numerically singular."""
    def det3(m: List[List[float]]) -> float:
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    d = det3(a)
    scale = max(abs(a[i][j]) for i in range(3) for j in range(3)) or 1.0
    if abs(d) <= 1e-14 * scale ** 3:
        return None
    out = []
    for col in range(3):
        m = [[rhs[i] if j == col else a[i][j] for j in range(3)] for i in range(3)]
        out.append(det3(m) / d)
    return out


def _fw_driver(network: Network, od: Any, *, variant: str = "bfw",
               max_iter: int = 200, tol: float = 1e-6,
               keep_origin_flows: bool = False,
               initial_flows: Optional[Sequence[float]] = None,
               line_search: str = "bisect",
               method_name: Optional[str] = None) -> AssignmentResult:
    """Shared engine for MSA / FW / conjugate FW / biconjugate FW.

    Every variant differs only in how it builds the auxiliary point ``s`` and how it
    picks the step; because ``s`` is always a convex combination of all-or-nothing
    loadings, ``x + lam (s - x)`` with ``lam`` in [0,1] is feasible by construction.
    """
    odm = _as_od(od)
    _check_od(network, odm)
    by_origin = odm.by_origin()
    total_demand = odm.total()
    m = network.n_links
    kmax = _as_index(max_iter, "max_iter")
    eps = _as_nonneg(tol, "tol")
    track = bool(keep_origin_flows)

    if initial_flows is None:
        c0 = network.costs(network.zeros())
        x, _sptt0, _sp0, x_origin = _aon_core(network, by_origin, c0, track)
    else:
        x = _check_flows(network, initial_flows, "initial_flows")
        x_origin = None
        track = False

    s_prev: Optional[Flow] = None
    s_prev2: Optional[Flow] = None
    d_prev: Optional[Flow] = None
    d_prev2: Optional[Flow] = None
    s_prev_origin: Optional[Dict[int, Flow]] = None
    s_prev2_origin: Optional[Dict[int, Flow]] = None
    history: List[IterationRecord] = []
    converged = False
    used = 0

    for k in range(1, kmax + 1):
        used = k
        costs = network.costs(x)
        y, sptt, _sp, y_origin = _aon_core(network, by_origin, costs, track)
        loaded = math.fsum(x[e] * costs[e] for e in range(m))
        rgap = max(0.0, (loaded - sptt) / loaded) if loaded > _EPS else 0.0
        aec = max(0.0, (loaded - sptt) / total_demand) if total_demand > _EPS else 0.0
        rec = IterationRecord(
            iteration=k - 1, objective=beckmann_objective(network, x),
            relative_gap=rgap, average_excess_cost=aec,
            total_travel_time=total_system_travel_time(network, x))
        history.append(rec)
        if rgap <= eps:
            converged = True
            break

        # ---- auxiliary point s (and the origin-flow bookkeeping that mirrors it)
        s = y
        s_origin = y_origin
        alpha = 0.0
        betas: Optional[List[float]] = None
        if variant in ("cfw", "bfw") and s_prev is not None and d_prev is not None:
            hess = network.derivatives(x)
            u = [s_prev[e] - x[e] for e in range(m)]
            v = [y[e] - x[e] for e in range(m)]
            if variant == "bfw" and s_prev2 is not None and d_prev2 is not None:
                w = [s_prev2[e] - x[e] for e in range(m)]
                mat = [[_hessian_inner(hess, d_prev, v),
                        _hessian_inner(hess, d_prev, u),
                        _hessian_inner(hess, d_prev, w)],
                       [_hessian_inner(hess, d_prev2, v),
                        _hessian_inner(hess, d_prev2, u),
                        _hessian_inner(hess, d_prev2, w)],
                       [1.0, 1.0, 1.0]]
                sol = _solve3(mat, [0.0, 0.0, 1.0])
                if sol is not None and all(b >= -1e-12 for b in sol) \
                        and sol[0] > 1e-12:
                    betas = [max(0.0, b) for b in sol]
                    tot = math.fsum(betas)
                    betas = [b / tot for b in betas]
                    s = [betas[0] * y[e] + betas[1] * s_prev[e]
                         + betas[2] * s_prev2[e] for e in range(m)]
                    if track and y_origin is not None and s_prev_origin is not None \
                            and s_prev2_origin is not None:
                        s_origin = {o: [betas[0] * y_origin[o][e]
                                        + betas[1] * s_prev_origin[o][e]
                                        + betas[2] * s_prev2_origin[o][e]
                                        for e in range(m)] for o in y_origin}
            if betas is None:
                num = _hessian_inner(hess, d_prev, v)
                den = _hessian_inner(hess, d_prev, [v[e] - u[e] for e in range(m)])
                if abs(den) > _EPS:
                    ratio = num / den
                    if 0.0 <= ratio < 1.0 - 1e-9:
                        alpha = ratio
                if alpha > 0.0:
                    s = [alpha * s_prev[e] + (1.0 - alpha) * y[e] for e in range(m)]
                    if track and y_origin is not None and s_prev_origin is not None:
                        s_origin = {o: [alpha * s_prev_origin[o][e]
                                        + (1.0 - alpha) * y_origin[o][e]
                                        for e in range(m)] for o in y_origin}

        direction = [s[e] - x[e] for e in range(m)]
        # a conjugate direction must still descend; fall back to plain FW if not
        slope = math.fsum(direction[e] * costs[e] for e in range(m))
        if slope >= -1e-16 and (alpha > 0.0 or betas is not None):
            s, s_origin = y, y_origin
            direction = [s[e] - x[e] for e in range(m)]
            slope = math.fsum(direction[e] * costs[e] for e in range(m))
            alpha, betas = 0.0, None
        if slope >= 0.0:
            converged = True
            break

        if variant == "msa":
            lam = 1.0 / float(k)
        else:
            lam = exact_line_search(network, x, direction, method=line_search)
        if lam <= 0.0:
            converged = True
            break

        new_x = [x[e] + lam * direction[e] for e in range(m)]
        for e in range(m):
            if new_x[e] < 0.0:
                new_x[e] = 0.0
        rec.step_size = lam
        rec.max_flow_change = max(abs(lam * direction[e]) for e in range(m)) \
            if m else 0.0
        if track and x_origin is not None and s_origin is not None:
            x_origin = {o: [x_origin[o][e] + lam * (s_origin[o][e] - x_origin[o][e])
                            for e in range(m)] for o in x_origin}
        x = new_x
        s_prev2, s_prev = s_prev, list(s)
        s_prev2_origin, s_prev_origin = s_prev_origin, s_origin
        d_prev2, d_prev = d_prev, direction

    if not converged:
        costs = network.costs(x)
        loaded, sptt, _t = _gap_parts(network, x, odm, costs)
        history.append(IterationRecord(
            iteration=used, objective=beckmann_objective(network, x),
            relative_gap=max(0.0, (loaded - sptt) / loaded) if loaded > _EPS else 0.0,
            average_excess_cost=max(0.0, (loaded - sptt) / total_demand)
            if total_demand > _EPS else 0.0,
            total_travel_time=total_system_travel_time(network, x)))
    name = method_name or {"msa": "MSA", "fw": "Frank-Wolfe",
                           "cfw": "conjugate Frank-Wolfe",
                           "bfw": "biconjugate Frank-Wolfe"}.get(variant, variant)
    return _finalise(network, odm, x, name, history, used, converged,
                     x_origin if track else None)


def method_of_successive_averages(network: Network, od: Any, *,
                                  max_iter: int = 500, tol: float = 1e-6,
                                  keep_origin_flows: bool = False,
                                  initial_flows: Optional[Sequence[float]] = None
                                  ) -> AssignmentResult:
    """MSA: average the all-or-nothing loadings with the harmonic weights ``1/k``.

    Convergence is guaranteed but glacial, and -- unlike Frank-Wolfe -- the Beckmann
    objective is *not* monotone along the way, because the step length is fixed in
    advance rather than minimising the restriction.
    """
    return _fw_driver(network, od, variant="msa", max_iter=max_iter, tol=tol,
                      keep_origin_flows=keep_origin_flows,
                      initial_flows=initial_flows, method_name="MSA")


def frank_wolfe(network: Network, od: Any, *, max_iter: int = 200,
                tol: float = 1e-6, keep_origin_flows: bool = False,
                initial_flows: Optional[Sequence[float]] = None,
                line_search: str = "bisect") -> AssignmentResult:
    """Frank-Wolfe with an exact line search on the convex 1-D restriction.

    Each iteration linearises Beckmann at the current flow (which makes the
    linearised problem an all-or-nothing loading) and then moves as far along that
    direction as the objective keeps falling, so the objective trace is monotone
    non-increasing by construction.
    """
    return _fw_driver(network, od, variant="fw", max_iter=max_iter, tol=tol,
                      keep_origin_flows=keep_origin_flows,
                      initial_flows=initial_flows, line_search=line_search,
                      method_name="Frank-Wolfe")


def conjugate_frank_wolfe(network: Network, od: Any, *, max_iter: int = 200,
                          tol: float = 1e-6, keep_origin_flows: bool = False,
                          initial_flows: Optional[Sequence[float]] = None
                          ) -> AssignmentResult:
    """Conjugate Frank-Wolfe: the new direction is H-conjugate to the previous one.

    The auxiliary point becomes ``alpha s_{k-1} + (1-alpha) y_k`` with ``alpha``
    chosen so that ``d_{k-1}^T H d_k = 0`` for the diagonal Beckmann Hessian
    ``H = diag(t'_e)``, which kills most of Frank-Wolfe's notorious zig-zag.
    """
    return _fw_driver(network, od, variant="cfw", max_iter=max_iter, tol=tol,
                      keep_origin_flows=keep_origin_flows,
                      initial_flows=initial_flows,
                      method_name="conjugate Frank-Wolfe")


def biconjugate_frank_wolfe(network: Network, od: Any, *, max_iter: int = 200,
                            tol: float = 1e-6, keep_origin_flows: bool = False,
                            initial_flows: Optional[Sequence[float]] = None
                            ) -> AssignmentResult:
    """Biconjugate Frank-Wolfe (Mitradjieva-Lindberg).

    The auxiliary point mixes the new all-or-nothing loading with the two previous
    auxiliary points, the three weights solved from the two conjugacy conditions
    plus the convexity constraint; it degrades gracefully to conjugate FW and then
    to plain FW whenever that 3x3 system is singular or would leave the simplex.
    """
    return _fw_driver(network, od, variant="bfw", max_iter=max_iter, tol=tol,
                      keep_origin_flows=keep_origin_flows,
                      initial_flows=initial_flows,
                      method_name="biconjugate Frank-Wolfe")


# ===========================================================================
# 7. PATH-BASED EQUILIBRIUM -- gradient projection (Jayakrishnan)
# ===========================================================================
def _column_generation(network: Network, odm: ODMatrix, pathset: PathSet,
                       costs: Sequence[float]) -> Tuple[float, Dict[Tuple[int, int],
                                                                    Path]]:
    """Add each OD pair's current shortest route to the choice set; top up demand."""
    sptt_terms: List[float] = []
    best: Dict[Tuple[int, int], Path] = {}
    for o, dests in odm.by_origin().items():
        dist, pred = _dijkstra(network, o, costs)
        tree = ShortestPathTree(o, dist, pred).bind(network)
        for (d, trips) in dests:
            if dist[d] >= _INF:
                raise TrafficAssignmentError(
                    "destination %d unreachable from origin %d." % (d, o))
            sptt_terms.append(trips * dist[d])
            links = tree.path_links(d)
            if not pathset.has(o, d, links):
                pathset.add(o, d, links, 0.0, tree.path_nodes(network, d))
            bucket = pathset.paths(o, d)
            for p in bucket:
                p.cost = path_cost(network, p, costs)
            star = min(bucket, key=lambda p: (p.cost, p.links))
            shortfall = trips - math.fsum(p.flow for p in bucket)
            if shortfall > 0.0:
                star.flow += shortfall
            best[(o, d)] = star
    return math.fsum(sptt_terms), best


def gradient_projection(network: Network, od: Any, *, max_iter: int = 200,
                        tol: float = 1e-6, step: float = 1.0,
                        prune: bool = True,
                        max_paths_per_od: Optional[int] = None,
                        keep_origin_flows: bool = False) -> AssignmentResult:
    """Jayakrishnan's gradient projection: column generation + path equilibration.

    Each iteration adds the current shortest route as a new column, then moves flow
    off every more-expensive route with the quasi-Newton step
    ``delta = (c_p - c_min) / sum_{e in p XOR p*} t'_e`` -- the exact equalising
    shift when the costs on the two routes are affine.  A Beckmann-objective
    safeguard halves the step rather than ever letting the objective rise.
    """
    odm = _as_od(od)
    _check_od(network, odm)
    m = network.n_links
    kmax = _as_index(max_iter, "max_iter")
    eps = _as_nonneg(tol, "tol")
    base_step = _as_positive(step, "step")
    total_demand = odm.total()
    pathset = PathSet()
    flows = network.zeros()
    costs = network.costs(flows)
    _column_generation(network, odm, pathset, costs)
    flows = pathset.link_flows(network)
    history: List[IterationRecord] = []
    converged = False
    used = 0

    for k in range(1, kmax + 1):
        used = k
        costs = network.costs(flows)
        sptt, best = _column_generation(network, odm, pathset, costs)
        flows = pathset.link_flows(network)
        costs = network.costs(flows)
        loaded = math.fsum(flows[e] * costs[e] for e in range(m))
        rgap = max(0.0, (loaded - sptt) / loaded) if loaded > _EPS else 0.0
        aec = max(0.0, (loaded - sptt) / total_demand) if total_demand > _EPS else 0.0
        obj_before = beckmann_objective(network, flows)
        rec = IterationRecord(iteration=k - 1, objective=obj_before,
                              relative_gap=rgap, average_excess_cost=aec,
                              total_travel_time=total_system_travel_time(network,
                                                                         flows))
        history.append(rec)
        if rgap <= eps:
            converged = True
            break

        snapshot = [(p, p.flow) for p in pathset.all_paths()]
        derivs = network.derivatives(flows)
        cur_step = base_step
        for _attempt in range(7):
            for (p, f0) in snapshot:
                p.flow = f0
            moved = 0.0
            for (o, d) in odm.pairs():
                bucket = pathset.paths(o, d)
                if len(bucket) < 2:
                    continue
                for p in bucket:
                    p.cost = path_cost(network, p, costs)
                star = min(bucket, key=lambda p: (p.cost, p.links))
                star_links = set(star.links)
                for p in bucket:
                    if p is star or p.flow <= 0.0:
                        continue
                    diff = p.cost - star.cost
                    if diff <= 0.0:
                        continue
                    sym = star_links.symmetric_difference(p.links)
                    sd = math.fsum(derivs[e] for e in sym)
                    if sd <= 0.0:
                        # Genuinely constant costs: the equalising shift is
                        # unbounded, so move the whole route -- but scaled by
                        # ``cur_step`` so the Beckmann safeguard's halving actually
                        # produces a *different* trial point instead of retrying the
                        # same corner seven times.
                        delta = cur_step * p.flow
                    else:
                        # No absolute floor on ``sd``.  A lightly loaded BPR link
                        # with beta=4 and a large capacity has a genuine derivative
                        # near 1e-17; calling that "zero" dumps the entire route
                        # flow onto one corner and the solver oscillates there
                        # forever while reporting a tiny gap.
                        delta = min(p.flow, cur_step * diff / sd)
                    if delta <= 0.0:
                        continue
                    p.flow -= delta
                    star.flow += delta
                    moved += delta
            trial = pathset.link_flows(network)
            obj_after = beckmann_objective(network, trial)
            if obj_after <= obj_before + 1e-13 * max(1.0, abs(obj_before)):
                rec.max_flow_change = max_flow_change(flows, trial)
                rec.step_size = cur_step
                flows = trial
                break
            cur_step *= 0.5
        else:
            for (p, f0) in snapshot:
                p.flow = f0
            flows = pathset.link_flows(network)
        if prune:
            pathset.prune(1e-13)
        if max_paths_per_od is not None:
            cap = _as_index(max_paths_per_od, "max_paths_per_od")
            for (o, d) in pathset.pairs():
                bucket = pathset.paths(o, d)
                if len(bucket) > cap:
                    bucket.sort(key=lambda p: (-p.flow, p.cost))
                    dropped = bucket[cap:]
                    keep = bucket[:cap]
                    spill = math.fsum(p.flow for p in dropped)
                    if spill > 0.0:
                        keep[0].flow += spill
                    pathset._paths[(o, d)] = keep
            flows = pathset.link_flows(network)

    if not converged:
        costs = network.costs(flows)
        loaded, sptt, _t = _gap_parts(network, flows, odm, costs)
        history.append(IterationRecord(
            iteration=used, objective=beckmann_objective(network, flows),
            relative_gap=max(0.0, (loaded - sptt) / loaded) if loaded > _EPS else 0.0,
            average_excess_cost=max(0.0, (loaded - sptt) / total_demand)
            if total_demand > _EPS else 0.0,
            total_travel_time=total_system_travel_time(network, flows)))
    pathset.update_costs(network, network.costs(flows))
    of: Optional[Dict[int, Flow]] = None
    if keep_origin_flows:
        of = {o: network.zeros() for o in odm.origins()}
        for pth in pathset.all_paths():
            if pth.flow == 0.0:
                continue
            vec = of.setdefault(pth.origin, network.zeros())
            for e in pth.links:
                vec[e] += pth.flow
    return _finalise(network, odm, flows, "gradient projection", history, used,
                     converged, of, pathset,
                     {"n_paths": pathset.n_paths()})


# ===========================================================================
# 8. ORIGIN-BASED (BUSH) EQUILIBRIUM -- Dial's Algorithm B
# ===========================================================================
def _bush_topological_order(network: Network, bush: Set[int],
                            key: Sequence[float]) -> Tuple[List[int], List[int]]:
    """Kahn's algorithm over the bush, breaking ties by ``key`` (the min labels)."""
    n = network.n_nodes
    indeg = [0] * n
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    for e in bush:
        lk = network.links[e]
        adj[lk.tail].append((e, lk.head))
        indeg[lk.head] += 1
    heap = [(key[v] if key[v] < _INF else 1e30, v)
            for v in range(n) if indeg[v] == 0]
    heapq.heapify(heap)
    order: List[int] = []
    while heap:
        _k, u = heapq.heappop(heap)
        order.append(u)
        for (_e, v) in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (key[v] if key[v] < _INF else 1e30, v))
    if len(order) != n:
        raise TrafficAssignmentError(
            "bush is not acyclic (%d of %d nodes ordered)." % (len(order), n))
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    return order, pos


def _bush_adjacency(network: Network, bush: Set[int],
                    n_nodes: int) -> List[List[int]]:
    """Head lists for the bush, built once and then extended in place."""
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    for e in bush:
        adj[network.links[e].tail].append(network.links[e].head)
    return adj


def _bush_reaches(adj: Sequence[List[int]], src: int, dst: int) -> bool:
    """Whether ``dst`` is reachable from ``src`` over the bush adjacency ``adj``.

    Adding link ``(i, j)`` to an acyclic bush keeps it acyclic exactly when ``j``
    cannot already reach ``i``.  Comparing positions in *one* topological order is
    only a sufficient test, never a necessary one, and rejects admissible links.
    """
    if src == dst:
        return True
    seen = [False] * len(adj)
    seen[src] = True
    stack = [src]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v == dst:
                return True
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    return False


def _bush_labels(network: Network, bush: Set[int], origin: int,
                 order: Sequence[int], costs: Sequence[float],
                 bush_flow: Sequence[float], flow_eps: float = _FLOW_EPS
                 ) -> Tuple[List[float], List[int], List[float], List[int]]:
    """Min-cost and max-cost (over *used* links) labels in topological order."""
    n = network.n_nodes
    min_l = [_INF] * n
    min_p = [-1] * n
    max_l = [-_INF] * n
    max_p = [-1] * n
    min_l[origin] = 0.0
    max_l[origin] = 0.0
    incoming: List[List[int]] = [[] for _ in range(n)]
    for e in bush:
        incoming[network.links[e].head].append(e)
    for v in order:
        if v == origin:
            continue
        for e in sorted(incoming[v]):
            i = network.links[e].tail
            if min_l[i] < _INF:
                cand = min_l[i] + costs[e]
                if cand < min_l[v] - 1e-15 * (abs(min_l[v])
                                              if min_l[v] < _INF else 0.0):
                    min_l[v] = cand
                    min_p[v] = e
            if bush_flow[e] > flow_eps and max_l[i] > -_INF:
                cand = max_l[i] + costs[e]
                if cand > max_l[v] + 1e-15 * (abs(max_l[v])
                                              if max_l[v] > -_INF else 0.0):
                    max_l[v] = cand
                    max_p[v] = e
    return min_l, min_p, max_l, max_p


def _segment_cost(network: Network, seg: Sequence[int], flows: Sequence[float],
                  delta: float) -> float:
    """Generalised cost of a chain of links when ``delta`` is added to each flow."""
    terms = []
    for e in seg:
        lk = network.links[e]
        x = flows[e] + delta
        if x < 0.0:
            x = 0.0
        terms.append(lk.vdf.travel_time(x) + lk.toll)
    return math.fsum(terms)


def _shift_between_segments(network: Network, seg_min: Sequence[int],
                            seg_max: Sequence[int], flows: Flow,
                            bush_flow: Flow,
                            flow_eps: float = _FLOW_EPS) -> float:
    """Equalise two parallel segments by bisection; return the flow actually moved.

    ``flow_eps`` is a *flow-scaled* floor supplied by the caller and ``gap0`` is
    compared relatively: both quantities carry units, so a bare absolute literal
    would refuse to move anything on a sub-unit demand table and never stop on a
    huge one.
    """
    cap = min(bush_flow[e] for e in seg_max)
    if cap <= flow_eps:
        return 0.0
    c_min0 = _segment_cost(network, seg_min, flows, 0.0)
    c_max0 = _segment_cost(network, seg_max, flows, 0.0)
    gap0 = c_max0 - c_min0
    if gap0 <= 1e-15 * max(abs(c_min0), abs(c_max0)):
        return 0.0

    def g_at(delta: float) -> float:
        """Cost of the expensive segment minus the cheap one after moving ``delta``."""
        hi = math.fsum(network.links[e].vdf.travel_time(max(0.0, flows[e] - delta))
                       + network.links[e].toll for e in seg_max)
        lo = math.fsum(network.links[e].vdf.travel_time(max(0.0, flows[e] + delta))
                       + network.links[e].toll for e in seg_min)
        return hi - lo

    if g_at(cap) >= 0.0:
        delta = cap
    else:
        lo, hi = 0.0, cap
        for _ in range(_BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            if g_at(mid) > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo <= 1e-15 * cap:
                break
        delta = 0.5 * (lo + hi)
    if delta <= 0.0:
        return 0.0
    for e in seg_max:
        bush_flow[e] = max(0.0, bush_flow[e] - delta)
        flows[e] = max(0.0, flows[e] - delta)
    for e in seg_min:
        bush_flow[e] += delta
        flows[e] += delta
    return delta


def algorithm_b(network: Network, od: Any, *, max_iter: int = 100,
                tol: float = 1e-6, inner_passes: int = 4,
                keep_origin_flows: bool = True) -> AssignmentResult:
    """Dial's Algorithm B: one acyclic *bush* per origin, equilibrated internally.

    Each origin keeps an acyclic subnetwork carrying all of its flow.  Within a bush
    the minimum- and maximum-cost used paths to every node are found in topological
    order and flow is shifted from the expensive to the cheap segment until they
    equalise (exactly, by bisection, so the step can never overshoot).  The bush is
    then *improved*: stale zero-flow links are dropped and any link that would
    shorten a min path is admitted, guarded by a direct ``j``-cannot-reach-``i``
    acyclicity test.

    Dropping the zero-flow links is a *correctness* requirement, not a speed
    optimisation.  A zero-flow link left in the bush still pins the topology, and it
    can permanently veto an improving link that would close a cycle with it -- which
    freezes the bush at a feasible but non-equilibrium point, with the objective
    stuck tens of percent above the true optimum while every bush reports itself
    internally equilibrated.  Every positive-flow link and the whole min-path tree
    are always kept, so loaded routes and reachability both survive the drop.
    """
    odm = _as_od(od)
    _check_od(network, odm)
    m = network.n_links
    n = network.n_nodes
    kmax = _as_index(max_iter, "max_iter")
    eps = _as_nonneg(tol, "tol")
    passes = max(1, _as_index(inner_passes, "inner_passes"))
    by_origin = odm.by_origin()
    origins = sorted(by_origin)
    total_demand = odm.total()
    # Every flow threshold below is measured against the size of the trip table.
    # A bare veh/h literal stops a small-demand network after one pass and never
    # stops a huge one.
    flow_eps = _FLOW_EPS * total_demand if total_demand > 0.0 else _FLOW_EPS
    tails = [lk.tail for lk in network.links]

    # ---- initial bushes: free-flow shortest-path trees, all-or-nothing loaded
    flows = network.zeros()
    bushes: Dict[int, Set[int]] = {}
    bush_flows: Dict[int, Flow] = {}
    c0 = network.costs(flows)
    for o in origins:
        dist, pred = _dijkstra(network, o, c0)
        bush = {pred[v] for v in range(n) if pred[v] >= 0 and dist[v] < _INF}
        bf = [0.0] * m
        load = [0.0] * n
        for (d, trips) in by_origin[o]:
            load[d] += trips
        for v in _tree_order(network, dist, pred, o):
            amt = load[v]
            if amt <= 0.0 or v == o:
                continue
            e = pred[v]
            bf[e] += amt
            load[tails[e]] += amt
            load[v] = 0.0
        bushes[o] = bush
        bush_flows[o] = bf
        for e in range(m):
            flows[e] += bf[e]

    history: List[IterationRecord] = []
    converged = False
    used = 0
    for k in range(1, kmax + 1):
        used = k
        costs = network.costs(flows)
        loaded, sptt, _t = _gap_parts(network, flows, odm, costs)
        rgap = max(0.0, (loaded - sptt) / loaded) if loaded > _EPS else 0.0
        aec = max(0.0, (loaded - sptt) / total_demand) if total_demand > _EPS else 0.0
        prev = list(flows)
        history.append(IterationRecord(
            iteration=k - 1, objective=beckmann_objective(network, flows),
            relative_gap=rgap, average_excess_cost=aec,
            total_travel_time=total_system_travel_time(network, flows)))
        if rgap <= eps:
            converged = True
            break

        for o in origins:
            bush = bushes[o]
            bf = bush_flows[o]
            # ---- equilibrate the bush
            for _p in range(passes):
                costs = network.costs(flows)
                dist_full, _pf = _dijkstra(network, o, costs)
                order, pos = _bush_topological_order(network, bush, dist_full)
                min_l, min_p, max_l, max_p = _bush_labels(
                    network, bush, o, order, costs, bf, flow_eps)
                moved = 0.0
                for v in reversed(order):
                    if min_p[v] < 0 or max_p[v] < 0 or min_p[v] == max_p[v]:
                        continue
                    seg_min = [min_p[v]]
                    a = tails[min_p[v]]
                    seg_max = [max_p[v]]
                    b = tails[max_p[v]]
                    ok = True
                    while a != b:
                        if pos[a] > pos[b]:
                            e = min_p[a]
                            if e < 0:
                                ok = False
                                break
                            seg_min.append(e)
                            a = tails[e]
                        else:
                            e = max_p[b]
                            if e < 0:
                                ok = False
                                break
                            seg_max.append(e)
                            b = tails[e]
                    if not ok or set(seg_min) & set(seg_max):
                        continue
                    moved += _shift_between_segments(network, seg_min, seg_max,
                                                     flows, bf, flow_eps)
                if moved <= flow_eps:
                    break
            # ---- improve the bush: drop stale links, then admit improving ones
            costs = network.costs(flows)
            dist_full, _pf = _dijkstra(network, o, costs)
            order, pos = _bush_topological_order(network, bush, dist_full)
            min_l, min_p, _xl, _xp = _bush_labels(network, bush, o, order, costs,
                                                  bf, flow_eps)
            # Keep every loaded link and the whole min-path tree; drop the rest.
            # The survivors are a subgraph of an acyclic bush, so the bush stays
            # acyclic, but the topology it pins is now only the topology it is
            # actually using -- which is what lets a blocked improving link in.
            keep = set(e for e in bush if bf[e] > 0.0)
            for v in range(n):
                if min_p[v] >= 0:
                    keep.add(min_p[v])
            if keep != bush:
                bush.clear()
                bush.update(keep)
                order, pos = _bush_topological_order(network, bush, dist_full)
                min_l, min_p, _xl, _xp = _bush_labels(network, bush, o, order,
                                                      costs, bf, flow_eps)
            # Admit best-improvement first.  The cut-off is relative to the labels
            # it compares: ``max(1.0, ...)`` made it a flat 1e-12 veh-min floor that
            # hid every improving link on a sub-unit cost scale.
            cand = []
            for e in range(m):
                if e in bush:
                    continue
                i = network.links[e].tail
                j = network.links[e].head
                if min_l[i] >= _INF:
                    continue
                slack = min_l[i] + costs[e] - min_l[j]
                scale = max(abs(min_l[j]), abs(min_l[i]) + abs(costs[e]))
                # A few ulps, not ``_EPS``.  1e-12 relative is still 4500x coarser
                # than double precision, and a genuine 3e-13 improvement -- which
                # is what separates two near-identical parallel links -- was
                # rounded away, leaving the bush stuck on one of them.
                if slack < -1e-15 * scale:
                    cand.append((slack, e))
            cand.sort()
            if cand:
                adj = _bush_adjacency(network, bush, n)
                for _slack, e in cand:
                    lk = network.links[e]
                    if _bush_reaches(adj, lk.head, lk.tail):
                        continue
                    bush.add(e)
                    adj[lk.tail].append(lk.head)
        history[-1].max_flow_change = max_flow_change(prev, flows)
        history[-1].step_size = 1.0

    if not converged:
        costs = network.costs(flows)
        loaded, sptt, _t = _gap_parts(network, flows, odm, costs)
        history.append(IterationRecord(
            iteration=used, objective=beckmann_objective(network, flows),
            relative_gap=max(0.0, (loaded - sptt) / loaded) if loaded > _EPS else 0.0,
            average_excess_cost=max(0.0, (loaded - sptt) / total_demand)
            if total_demand > _EPS else 0.0,
            total_travel_time=total_system_travel_time(network, flows)))
    return _finalise(network, odm, flows, "Algorithm B (bush)", history, used,
                     converged,
                     {o: list(bush_flows[o]) for o in origins}
                     if keep_origin_flows else None, None,
                     {"bush_sizes": {o: len(bushes[o]) for o in origins}})


def user_equilibrium(network: Network, od: Any, *, method: str = "bfw",
                     max_iter: int = 200, tol: float = 1e-6,
                     keep_origin_flows: bool = False) -> AssignmentResult:
    """Solve the deterministic user equilibrium with the named solver.

    ``method`` is one of ``msa``, ``fw``, ``cfw``, ``bfw``, ``gp`` (gradient
    projection) or ``b`` (Algorithm B).  All six minimise the same strictly convex
    Beckmann objective, so on a given network they must agree on link flows; that
    cross-solver agreement is the single best correctness oracle this module has.
    """
    key = str(method).lower().replace("-", "_")
    if key in ("msa", "successive_averages"):
        return method_of_successive_averages(network, od, max_iter=max_iter,
                                             tol=tol,
                                             keep_origin_flows=keep_origin_flows)
    if key in ("fw", "frank_wolfe"):
        return frank_wolfe(network, od, max_iter=max_iter, tol=tol,
                           keep_origin_flows=keep_origin_flows)
    if key in ("cfw", "conjugate", "conjugate_frank_wolfe"):
        return conjugate_frank_wolfe(network, od, max_iter=max_iter, tol=tol,
                                     keep_origin_flows=keep_origin_flows)
    if key in ("bfw", "biconjugate", "biconjugate_frank_wolfe"):
        return biconjugate_frank_wolfe(network, od, max_iter=max_iter, tol=tol,
                                       keep_origin_flows=keep_origin_flows)
    if key in ("gp", "gradient_projection", "path"):
        return gradient_projection(network, od, max_iter=max_iter, tol=tol,
                                   keep_origin_flows=keep_origin_flows)
    if key in ("b", "algorithm_b", "bush", "origin"):
        return algorithm_b(network, od, max_iter=max_iter, tol=tol,
                           keep_origin_flows=keep_origin_flows)
    raise TrafficAssignmentError(
        "unknown method %r (use msa, fw, cfw, bfw, gp or b)." % (method,))


# ===========================================================================
# 9. STOCHASTIC ASSIGNMENT -- Dial STOCH, logit SUE, probit SUE
# ===========================================================================
def _reverse_dijkstra(network: Network, destination: int,
                      costs: Sequence[float]) -> List[float]:
    """Cost-to-go from every node to ``destination`` (Dijkstra on the reverse graph)."""
    n = network.n_nodes
    dist = [_INF] * n
    dist[destination] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, destination)]
    inn = network._in
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u] + _EPS * (dist[u] if dist[u] < _INF else 0.0):
            continue
        for (e, t) in inn[u]:
            nd = d + costs[e]
            if nd < dist[t] - 1e-15 * (dist[t] if dist[t] < _INF else 0.0):
                dist[t] = nd
                heapq.heappush(heap, (nd, t))
    return dist


def _efficient_by_rank(network: Network, order: Sequence[int]) -> List[bool]:
    """Mark links that advance along ``order`` -- Dial's efficiency test, by rank.

    Dial defines a link efficient when it takes you further from the origin.
    Testing ``r(i) < r(j)`` directly excludes every zero-cost link, and with an
    absolute ``1e-12`` slack it excludes every link on a network whose costs are
    merely *small* -- which makes the backward pass fail outright with "no efficient
    approach".  ``order`` is already a strict total order (sorted by cost then node
    id), so comparing positions in it keeps the efficient subnetwork acyclic while
    admitting zero-cost connectors and free turns.
    """
    rank = [-1] * network.n_nodes
    for idx, v in enumerate(order):
        rank[v] = idx
    eff = [False] * network.n_links
    for e, lk in enumerate(network.links):
        if rank[lk.tail] >= 0 and rank[lk.head] >= 0 and rank[lk.tail] < rank[lk.head]:
            eff[e] = True
    return eff


def _dial_load(network: Network, origin: int, targets: Sequence[Tuple[int, float]],
               costs: Sequence[float], theta: float, efficient: Sequence[bool],
               order: Sequence[int], r: Sequence[float], flows: Flow) -> None:
    """Dial's two-pass STOCH kernel: forward weights, backward proportional load."""
    n = network.n_nodes
    m = network.n_links
    w_link = [0.0] * m
    w_node = [0.0] * n
    w_node[origin] = 1.0
    for v in order:
        if v != origin:
            inc = [e for (e, _t) in network.in_links(v) if efficient[e]]
            w_node[v] = math.fsum(w_link[e] for e in inc)
        for (e, _h) in network.out_links(v):
            if not efficient[e]:
                continue
            lk = network.links[e]
            like = math.exp(theta * (r[lk.head] - r[lk.tail] - costs[e]))
            w_link[e] = like * w_node[v]
    x_node = [0.0] * n
    for (d, trips) in targets:
        if trips > 0.0:
            x_node[d] += trips
    for v in reversed(list(order)):
        amt = x_node[v]
        if amt <= 0.0 or v == origin:
            continue
        inc = sorted(e for (e, _t) in network.in_links(v) if efficient[e])
        if not inc:
            raise TrafficAssignmentError(
                "node %d has no efficient approach from origin %d." % (v, origin))
        denom = math.fsum(w_link[e] for e in inc)
        if denom <= 1e-300:
            best = min(inc, key=lambda e: (r[network.links[e].tail] + costs[e], e))
            flows[best] += amt
            x_node[network.links[best].tail] += amt
        else:
            for e in inc:
                share = amt * w_link[e] / denom
                if share == 0.0:
                    continue
                flows[e] += share
                x_node[network.links[e].tail] += share
        x_node[v] = 0.0


def dial_stoch(network: Network, od: Any, costs: Sequence[float], *,
               theta: float = 1.0,
               use_destination_condition: bool = False) -> Flow:
    """Dial's STOCH loading: a logit split over *efficient* paths, without enumeration.

    A link is efficient when it takes you further from the origin
    (``r(i) < r(j)``), and -- with ``use_destination_condition`` -- also closer to
    the destination.  The forward pass accumulates link likelihoods
    ``exp(theta (r(j) - r(i) - c_ij))`` whose product along a route telescopes to
    ``exp(-theta c_route)``, so the backward pass splits demand exactly in logit
    proportions over the efficient route set.
    """
    odm = _as_od(od)
    _check_od(network, odm)
    c = _as_floats(costs, "costs")
    if len(c) != network.n_links:
        raise TrafficAssignmentError("costs length does not match the network.")
    _check_nonneg_costs(c)
    th = _as_positive(theta, "theta")
    m = network.n_links
    flows = [0.0] * m
    by_origin = odm.by_origin()
    if not use_destination_condition:
        for o in sorted(by_origin):
            r, _pred = _dijkstra(network, o, c)
            order = sorted((v for v in range(network.n_nodes) if r[v] < _INF),
                           key=lambda v: (r[v], v))
            eff = _efficient_by_rank(network, order)
            _dial_load(network, o, by_origin[o], c, th, eff, order, r, flows)
        return flows
    cache: Dict[int, List[float]] = {}
    for o in sorted(by_origin):
        r, _pred = _dijkstra(network, o, c)
        order = sorted((v for v in range(network.n_nodes) if r[v] < _INF),
                       key=lambda v: (r[v], v))
        for (d, trips) in by_origin[o]:
            if trips <= 0.0:
                continue
            if d not in cache:
                cache[d] = _reverse_dijkstra(network, d, c)
            s = cache[d]
            sub = [v for v in order if r[v] < _INF and s[v] < _INF]
            eff = _efficient_by_rank(network, sub)
            for e, lk in enumerate(network.links):
                if not eff[e]:
                    continue
                if s[lk.head] >= _INF:
                    eff[e] = False
                    continue
                tol_s = _EPS * max(abs(s[lk.tail]), abs(s[lk.head]))
                if s[lk.tail] < s[lk.head] - tol_s:
                    eff[e] = False
            _dial_load(network, o, [(d, trips)], c, th, eff, sub, r, flows)
    return flows


def logit_probabilities(path_costs: Sequence[float], theta: float = 1.0, *,
                        corrections: Optional[Sequence[float]] = None
                        ) -> List[float]:
    """Multinomial-logit route shares ``exp(-theta(c_p + corr_p))``, max-shifted."""
    costs = _as_floats(path_costs, "path_costs")
    if not costs:
        return []
    th = _as_positive(theta, "theta")
    corr = _as_floats(corrections, "corrections") if corrections is not None \
        else [0.0] * len(costs)
    if len(corr) != len(costs):
        raise TrafficAssignmentError("corrections length does not match path_costs.")
    util = [-th * (costs[i] + corr[i]) for i in range(len(costs))]
    hi = max(util)
    ex = [math.exp(u - hi) for u in util]
    tot = math.fsum(ex)
    if tot <= 0.0:
        return [1.0 / len(costs)] * len(costs)
    return [v / tot for v in ex]


def _path_lengths(network: Network, paths: Sequence[Path],
                  weights: Optional[Sequence[float]]) -> Tuple[List[float],
                                                               List[float]]:
    """Per-link weight vector and per-path total, defaulting to free-flow time."""
    if weights is None:
        w = network.free_flow_times()
        if all(v <= 0.0 for v in w):
            w = [1.0] * network.n_links
    else:
        w = _as_floats(weights, "weights")
        if len(w) != network.n_links:
            raise TrafficAssignmentError("weights length does not match the network.")
    totals = [math.fsum(w[e] for e in p.links) for p in paths]
    if any(t <= 0.0 for t in totals):
        # A single route of zero total length makes every overlap ratio 0/0, and the
        # ``max(_EPS, ...)`` floor below then *becomes* the answer: c_logit_factors
        # returns -27.6 where its own contract demands 0, path_size_factors returns
        # 1e-12 where it demands 1.  Zero-free-flow-time centroid connectors are
        # routine -- ``braess_network()`` itself builds them -- so fall back to link
        # counts, exactly as this function already does when *every* weight is zero.
        w = [1.0] * network.n_links
        totals = [math.fsum(w[e] for e in p.links) for p in paths]
    totals = [max(_EPS, t) for t in totals]
    return w, totals


def c_logit_factors(network: Network, paths: Sequence[Path], *,
                    weights: Optional[Sequence[float]] = None,
                    beta: float = 1.0, gamma: float = 1.0) -> List[float]:
    """Cassetta's commonality factors -- add them to route costs to punish overlap.

    ``CF_p = beta * ln sum_q (L_pq / sqrt(L_p L_q))**gamma`` where ``L_pq`` is the
    shared length of routes ``p`` and ``q``.  A route that shares nothing with the
    others scores ``0``; two identical routes each score ``beta ln 2``.
    """
    plist = list(paths)
    if not plist:
        return []
    b = _as_float(beta, "beta")
    g = _as_float(gamma, "gamma")
    w, totals = _path_lengths(network, plist, weights)
    sets = [set(p.links) for p in plist]
    out = []
    for i, p in enumerate(plist):
        terms = []
        for j, _q in enumerate(plist):
            shared = math.fsum(w[e] for e in sets[i] & sets[j])
            ratio = shared / math.sqrt(totals[i] * totals[j])
            terms.append(max(0.0, ratio) ** g)
        out.append(b * math.log(max(_EPS, math.fsum(terms))))
    return out


def path_size_factors(network: Network, paths: Sequence[Path], *,
                      weights: Optional[Sequence[float]] = None) -> List[float]:
    """Ben-Akiva path-size factors; ``c_p - ln(PS_p)/theta`` corrects for overlap.

    ``PS_p = sum_{a in p} (l_a / L_p) / sum_q delta_aq (L_p / L_q)``: a route made
    entirely of links nobody else uses scores 1, and ``N`` identical routes each
    score ``1/N``.
    """
    plist = list(paths)
    if not plist:
        return []
    w, totals = _path_lengths(network, plist, weights)
    sets = [set(p.links) for p in plist]
    out = []
    for i, p in enumerate(plist):
        terms = []
        for e in sets[i]:
            den = math.fsum(totals[i] / totals[j] for j in range(len(plist))
                            if e in sets[j])
            if den <= _EPS:
                continue
            terms.append((w[e] / totals[i]) / den)
        out.append(min(1.0, max(_EPS, math.fsum(terms))))
    return out


def fisk_objective(network: Network, pathset: PathSet, theta: float = 1.0, *,
                   include_tolls: bool = True) -> float:
    """Fisk's SUE objective: Beckmann plus the entropy term ``sum h (ln h - 1)/theta``.

    Minimising it over route flows reproduces exactly the multinomial-logit route
    shares, which is what makes logit SUE a convex program rather than a fixed point
    that merely happens to converge.
    """
    th = _as_positive(theta, "theta")
    flows = pathset.link_flows(network)
    z = beckmann_objective(network, flows, include_tolls=include_tolls)
    ent = math.fsum(p.flow * (math.log(p.flow) - 1.0)
                    for p in pathset.all_paths() if p.flow > 1e-300)
    return z + ent / th


def _sue_choice_set(network: Network, odm: ODMatrix, n_paths: int,
                    costs: Sequence[float]) -> PathSet:
    """Seed the route choice set with the ``k`` cheapest free-flow routes per OD."""
    ps = PathSet()
    for (o, d), _trips in odm.items():
        for (_c, links) in k_shortest_paths(network, o, d, costs, n_paths):
            nodes = [o]
            for e in links:
                nodes.append(network.links[e].head)
            ps.add(o, d, links, 0.0, nodes)
    return ps


def logit_sue(network: Network, od: Any, *, theta: float = 1.0,
              method: str = "dial", overlap: str = "none",
              max_iter: int = 100, tol: float = 1e-6, n_paths: int = 6,
              use_destination_condition: bool = False) -> AssignmentResult:
    """Stochastic user equilibrium under a logit route-choice model, solved by MSA.

    ``method="dial"`` loads with :func:`dial_stoch` (no route enumeration, the whole
    efficient route set); ``method="path"`` works on an explicit choice set and can
    apply a ``"clogit"`` or ``"pathsize"`` overlap correction, which Dial's implicit
    enumeration cannot express.  Convergence is measured by the largest link-flow
    change between successive stochastic loadings, relative to total demand.
    """
    odm = _as_od(od)
    _check_od(network, odm)
    th = _as_positive(theta, "theta")
    kmax = _as_index(max_iter, "max_iter")
    eps = _as_nonneg(tol, "tol")
    total_demand = max(_EPS, odm.total())
    m = network.n_links
    mode = str(method).lower()
    ov = str(overlap).lower()
    if mode not in ("dial", "path"):
        raise TrafficAssignmentError("method must be 'dial' or 'path'.")
    if ov not in ("none", "clogit", "pathsize"):
        raise TrafficAssignmentError(
            "overlap must be 'none', 'clogit' or 'pathsize'.")
    pathset: Optional[PathSet] = None
    if mode == "path":
        pathset = _sue_choice_set(network, odm, max(1, _as_index(n_paths,
                                                                 "n_paths")),
                                  network.free_flow_times())

    def _load(costs: Sequence[float]) -> Flow:
        if mode == "dial":
            return dial_stoch(network, odm, costs, theta=th,
                              use_destination_condition=use_destination_condition)
        assert pathset is not None
        out = [0.0] * m
        for (o, d), trips in odm.items():
            bucket = pathset.paths(o, d)
            pcosts = [path_cost(network, p, costs) for p in bucket]
            for p, pc in zip(bucket, pcosts):
                p.cost = pc
            if ov == "clogit":
                corr = c_logit_factors(network, bucket)
            elif ov == "pathsize":
                ps = path_size_factors(network, bucket)
                corr = [-math.log(v) / th for v in ps]
            else:
                corr = [0.0] * len(bucket)
            probs = logit_probabilities(pcosts, th, corrections=corr)
            for p, pr in zip(bucket, probs):
                p.flow = trips * pr
                for e in p.links:
                    out[e] += p.flow
        return out

    flows = _load(network.costs(network.zeros()))
    history: List[IterationRecord] = []
    converged = False
    used = 0
    sue_gap = _INF
    for k in range(2, kmax + 2):
        used = k - 1
        costs = network.costs(flows)
        y = _load(costs)
        sue_gap = max(abs(y[e] - flows[e]) for e in range(m)) / total_demand \
            if m else 0.0
        step = 1.0 / float(k)
        new = [flows[e] + step * (y[e] - flows[e]) for e in range(m)]
        loaded, sptt, _t = _gap_parts(network, flows, odm, costs)
        history.append(IterationRecord(
            iteration=k - 2, objective=beckmann_objective(network, flows),
            relative_gap=max(0.0, (loaded - sptt) / loaded) if loaded > _EPS else 0.0,
            average_excess_cost=max(0.0, (loaded - sptt) / total_demand),
            total_travel_time=total_system_travel_time(network, flows),
            max_flow_change=max_flow_change(flows, new), step_size=step))
        flows = new
        if sue_gap <= eps:
            converged = True
            break
    extra: Dict[str, Any] = {"theta": th, "sue_gap": sue_gap,
                             "loading": mode, "overlap": ov}
    if pathset is not None:
        # re-split the converged costs so the stored route flows match the answer
        _load(network.costs(flows))
        extra["fisk_objective"] = fisk_objective(network, pathset, th)
        extra["n_paths"] = pathset.n_paths()
    res = _finalise(network, odm, flows,
                    "logit SUE (%s%s)" % (mode, "" if ov == "none" else "/" + ov),
                    history, used, converged, None, pathset, extra)
    return res


def probit_sue(network: Network, od: Any, *, sigma: float = 0.3,
               n_draws: int = 40, max_iter: int = 40, tol: float = 1e-6,
               seed: int = 0) -> AssignmentResult:
    """Probit SUE by seeded Monte-Carlo simulation of perceived link costs.

    Each draw multiplies every link cost by ``max(0.05, 1 + sigma z)`` with ``z``
    standard normal, loads all-or-nothing on the perceived costs, and the draws are
    averaged; the outer loop is MSA.  The *same* seeded draw matrix is reused every
    outer iteration (common random numbers), so the loading map is a deterministic
    function of the link costs and the fixed point is well defined -- without that,
    MSA would be chasing a moving target.

    Precision is bounded by simulation, not by iteration count: the loading is a
    *piecewise-constant* function of the costs (a draw's route flips all at once),
    so the residual gap plateaus at roughly one route switch, ``demand/n_draws``
    vehicles.  Ask for more precision by raising ``n_draws``, never ``max_iter``.
    """
    odm = _as_od(od)
    _check_od(network, odm)
    sg = _as_nonneg(sigma, "sigma")
    draws = max(1, _as_index(n_draws, "n_draws"))
    kmax = _as_index(max_iter, "max_iter")
    eps = _as_nonneg(tol, "tol")
    m = network.n_links
    rng = random.Random(seed)
    noise = [[rng.gauss(0.0, 1.0) for _e in range(m)] for _i in range(draws)]
    by_origin = odm.by_origin()
    total_demand = max(_EPS, odm.total())

    def _load(costs: Sequence[float]) -> Flow:
        acc = [0.0] * m
        for row in noise:
            perceived = [costs[e] * max(0.05, 1.0 + sg * row[e])
                         for e in range(m)]
            f, _sptt, _sp, _of = _aon_core(network, by_origin, perceived, False)
            for e in range(m):
                acc[e] += f[e]
        return [v / draws for v in acc]

    flows = _load(network.costs(network.zeros()))
    history: List[IterationRecord] = []
    converged = False
    used = 0
    sue_gap = _INF
    for k in range(2, kmax + 2):
        used = k - 1
        costs = network.costs(flows)
        y = _load(costs)
        sue_gap = max(abs(y[e] - flows[e]) for e in range(m)) / total_demand \
            if m else 0.0
        step = 1.0 / float(k)
        new = [flows[e] + step * (y[e] - flows[e]) for e in range(m)]
        loaded, sptt, _t = _gap_parts(network, flows, odm, costs)
        history.append(IterationRecord(
            iteration=k - 2, objective=beckmann_objective(network, flows),
            relative_gap=max(0.0, (loaded - sptt) / loaded) if loaded > _EPS else 0.0,
            average_excess_cost=max(0.0, (loaded - sptt) / total_demand),
            total_travel_time=total_system_travel_time(network, flows),
            max_flow_change=max_flow_change(flows, new), step_size=step))
        flows = new
        if sue_gap <= eps:
            converged = True
            break
    return _finalise(network, odm, flows, "probit SUE (%d draws)" % draws,
                     history, used, converged, None, None,
                     {"sigma": sg, "n_draws": draws, "seed": seed,
                      "sue_gap": sue_gap})


# ===========================================================================
# 10. SYSTEM OPTIMUM, MARGINAL-COST TOLLS AND THE PRICE OF ANARCHY
# ===========================================================================
def system_optimum(network: Network, od: Any, *, method: str = "bfw",
                   max_iter: int = 300, tol: float = 1e-9,
                   keep_origin_flows: bool = False) -> AssignmentResult:
    """Minimise total system travel time by equilibrating the marginal-cost network.

    Because ``integral_0^f (t(w) + w t'(w)) dw = f t(f)``, the Beckmann objective of
    the marginal-cost network *is* total system travel time -- so a user equilibrium
    computed on ``network.with_marginal_costs()`` is the system optimum, and every
    solver in this module can compute it unchanged.
    """
    so_net = network.with_marginal_costs()
    res = user_equilibrium(so_net, od, method=method, max_iter=max_iter, tol=tol,
                           keep_origin_flows=keep_origin_flows)
    flows = res.flows
    res.method = "system optimum (%s)" % res.method
    res.travel_times = network.travel_times(flows)
    res.costs = network.costs(flows)
    res.total_travel_time = total_system_travel_time(network, flows)
    res.objective = res.total_travel_time
    res.extra["marginal_costs"] = so_net.costs(flows)
    res.extra["beckmann_of_user_costs"] = beckmann_objective(network, flows)
    return res


def marginal_cost_tolls(network: Network, flows: Sequence[float]) -> List[float]:
    """First-best congestion tolls ``tau_e = f_e t'_e(f_e)`` at the given flows.

    Charged on the *system-optimal* flows, these are exactly the externality each
    driver imposes on everybody behind them; adding them to travel time makes the
    selfish cost equal the social marginal cost, so the tolled user equilibrium
    reproduces the system optimum.
    """
    f = _check_flows(network, flows)
    return [f[i] * lk.vdf.derivative(f[i]) for i, lk in enumerate(network.links)]


def tolled_user_equilibrium(network: Network, od: Any,
                            tolls: Sequence[float], *, method: str = "bfw",
                            max_iter: int = 300,
                            tol: float = 1e-9) -> AssignmentResult:
    """User equilibrium on the network with ``tolls`` added to every link cost."""
    tolled = network.with_tolls(tolls)
    res = user_equilibrium(tolled, od, method=method, max_iter=max_iter, tol=tol)
    res.method = "tolled UE (%s)" % res.method
    res.extra["toll_revenue"] = math.fsum(
        res.flows[e] * tolled.links[e].toll for e in range(network.n_links))
    res.extra["untolled_total_travel_time"] = total_system_travel_time(network,
                                                                      res.flows)
    return res


@dataclass
class PriceOfAnarchyResult:
    """UE versus SO: the efficiency loss caused by selfish routing."""

    price_of_anarchy: float
    ue_total_travel_time: float
    so_total_travel_time: float
    ue_flows: Flow
    so_flows: Flow
    tolls: List[float]
    ue: AssignmentResult
    so: AssignmentResult

    def excess_delay(self) -> float:
        """Vehicle-hours wasted per unit time by selfish routing."""
        return self.ue_total_travel_time - self.so_total_travel_time

    def as_dict(self) -> Dict[str, Any]:
        """JSON-able camelCase summary."""
        return {"priceOfAnarchy": self.price_of_anarchy,
                "ueTotalTravelTime": self.ue_total_travel_time,
                "soTotalTravelTime": self.so_total_travel_time,
                "excessDelay": self.excess_delay(),
                "tolls": list(self.tolls)}


def price_of_anarchy(network: Network, od: Any, *, method: str = "bfw",
                     max_iter: int = 400,
                     tol: float = 1e-10) -> PriceOfAnarchyResult:
    """Ratio of user-equilibrium to system-optimal total travel time.

    For *affine* latencies Roughgarden and Tardos proved this can never exceed
    ``4/3``, whatever the topology or demand -- a hard, assertable bound.
    """
    ue = user_equilibrium(network, od, method=method, max_iter=max_iter, tol=tol)
    so = system_optimum(network, od, method=method, max_iter=max_iter, tol=tol)
    ue_t = total_system_travel_time(network, ue.flows)
    so_t = total_system_travel_time(network, so.flows)
    poa = 1.0 if so_t <= _EPS else ue_t / so_t
    return PriceOfAnarchyResult(poa, ue_t, so_t, ue.flows, so.flows,
                                marginal_cost_tolls(network, so.flows), ue, so)


# ===========================================================================
# 11. THE BRAESS PARADOX -- detection by removal
# ===========================================================================
@dataclass
class BraessResult:
    """Which links, if any, make everybody worse off by existing."""

    paradox: bool
    base_total_travel_time: float
    best_link: int
    best_total_travel_time: float
    improvement: float
    candidates: List[Tuple[int, float, float]]
    base: AssignmentResult

    def as_dict(self) -> Dict[str, Any]:
        """JSON-able camelCase summary."""
        return {"paradox": self.paradox,
                "baseTotalTravelTime": self.base_total_travel_time,
                "bestLink": self.best_link,
                "bestTotalTravelTime": self.best_total_travel_time,
                "improvement": self.improvement,
                "candidates": [{"link": e, "totalTravelTime": t, "improvement": g}
                               for (e, t, g) in self.candidates]}


def detect_braess(network: Network, od: Any, *, method: str = "bfw",
                  max_iter: int = 300, tol: float = 1e-10,
                  links: Optional[Sequence[int]] = None,
                  rel_tol: float = 1e-9) -> BraessResult:
    """Search for a link whose **removal** lowers total system travel time.

    That is the Braess paradox: adding capacity to a selfishly-routed network can
    make every single traveller worse off, so closing the right link can make
    everyone better off.  Each candidate is deleted, the equilibrium re-solved, and
    the resulting total travel time compared; candidates whose removal would strand
    an OD pair are skipped rather than counted as infinite improvement.
    """
    odm = _as_od(od)
    _check_od(network, odm)
    base = user_equilibrium(network, odm, method=method, max_iter=max_iter, tol=tol)
    base_tstt = total_system_travel_time(network, base.flows)
    cand_ids = list(range(network.n_links)) if links is None \
        else [_as_index(e, "link") for e in links]
    results: List[Tuple[int, float, float]] = []
    for e in cand_ids:
        if not 0 <= e < network.n_links:
            raise TrafficAssignmentError("link index %d out of range." % e)
        trimmed = network.without_link(e)
        ok = True
        for o in odm.origins():
            reach = trimmed.reachable(o)
            for (d, trips) in odm.by_origin()[o]:
                if trips > 0.0 and d not in reach:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        res = user_equilibrium(trimmed, odm, method=method, max_iter=max_iter,
                               tol=tol)
        tstt = total_system_travel_time(trimmed, res.flows)
        improvement = base_tstt - tstt
        if improvement > rel_tol * max(1.0, abs(base_tstt)):
            results.append((e, tstt, improvement))
    results.sort(key=lambda t: (-t[2], t[0]))
    if results:
        best_e, best_t, best_g = results[0]
        return BraessResult(True, base_tstt, best_e, best_t, best_g, results, base)
    return BraessResult(False, base_tstt, -1, base_tstt, 0.0, [], base)


def braess_network(*, demand: float = 6.0, include_bridge: bool = True
                   ) -> Tuple[Network, ODMatrix]:
    """The textbook Braess instance on four nodes, with or without the bridge.

    Links ``0->1`` and ``2->3`` cost ``10x``; links ``0->2`` and ``1->3`` cost
    ``50+x``; the optional bridge ``1->2`` costs ``10+x``.  With demand 6 from 0 to
    3 the equilibrium **without** the bridge splits 3/3 at a cost of 83 per
    traveller; **with** it the equilibrium is 2/2/2 over the three routes and every
    traveller now pays 92.  Total system travel time rises from 498 to 552 although
    a brand-new road was opened.
    """
    d = _as_positive(demand, "demand")
    net = Network("braess" if include_bridge else "braess-no-bridge")
    net.ensure_nodes(4)
    net.add_link(0, 1, Linear(0.0, 10.0, capacity=d), name="0->1 (10x)")
    net.add_link(0, 2, Linear(50.0, 1.0, capacity=d), name="0->2 (50+x)")
    if include_bridge:
        net.add_link(1, 2, Linear(10.0, 1.0, capacity=d), name="1->2 bridge (10+x)")
    net.add_link(1, 3, Linear(50.0, 1.0, capacity=d), name="1->3 (50+x)")
    net.add_link(2, 3, Linear(0.0, 10.0, capacity=d), name="2->3 (10x)")
    return net, ODMatrix({(0, 3): d})


# ===========================================================================
# 12. MULTI-CLASS ASSIGNMENT -- cars, HGVs and buses in PCU
# ===========================================================================
@dataclass
class VehicleClass:
    """One user class: its trip table, its PCU equivalent and its value of time."""

    name: str
    demand: Demand
    pcu: float = 1.0
    value_of_time: float = 1.0

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.pcu = _as_positive(self.pcu, "pcu")
        self.value_of_time = _as_positive(self.value_of_time, "value_of_time")
        self.demand = _as_od(self.demand).as_dict()

    def od(self) -> ODMatrix:
        """This class's trip table as an :class:`ODMatrix`."""
        return ODMatrix(self.demand)


@dataclass
class MultiClassResult:
    """Per-class link flows plus the shared PCU flows they congest each other with."""

    class_flows: Dict[str, Flow]
    pcu_flows: Flow
    travel_times: Cost
    class_costs: Dict[str, Cost]
    class_gaps: Dict[str, float]
    relative_gap: float
    total_travel_time: float
    iterations: int
    converged: bool
    history: List[IterationRecord] = field(default_factory=list)

    def class_names(self) -> List[str]:
        """Class names in a deterministic order."""
        return sorted(self.class_flows)

    def summary(self) -> str:
        """One-line human summary."""
        return ("multi-class  iters=%d gap=%.3e PCU-TSTT=%.4f classes=%s"
                % (self.iterations, self.relative_gap, self.total_travel_time,
                   ",".join(self.class_names())))


def multiclass_assignment(network: Network, classes: Sequence[VehicleClass], *,
                          max_iter: int = 400, tol: float = 1e-6
                          ) -> MultiClassResult:
    """Multi-class user equilibrium in passenger-car units, solved by MSA.

    Classes interact only through the PCU flow ``x_e = sum_c pcu_c f_{c,e}`` that
    drives the volume-delay function, but each class routes on its own generalised
    cost ``t_e + toll_e / vot_c``.  With different values of time the problem is
    asymmetric and has no Beckmann-style potential, so MSA -- which never needs
    one -- is the right solver; the reported gap is the demand-weighted average
    excess cost over all classes, normalised by loaded cost.
    """
    if not classes:
        raise TrafficAssignmentError("at least one vehicle class is required.")
    names = [c.name for c in classes]
    if len(set(names)) != len(names):
        raise TrafficAssignmentError("vehicle class names must be unique.")
    m = network.n_links
    kmax = _as_index(max_iter, "max_iter")
    eps = _as_nonneg(tol, "tol")
    ods = {c.name: c.od() for c in classes}
    for c in classes:
        _check_od(network, ods[c.name])
    by_origin = {c.name: ods[c.name].by_origin() for c in classes}

    def _class_costs(times: Sequence[float], c: VehicleClass) -> Cost:
        return [times[e] + network.links[e].toll / c.value_of_time
                for e in range(m)]

    def _pcu(flows_by_class: Dict[str, Flow]) -> Flow:
        out = [0.0] * m
        for c in classes:
            fc = flows_by_class[c.name]
            for e in range(m):
                out[e] += c.pcu * fc[e]
        return out

    times0 = network.travel_times([0.0] * m)
    flows_by_class: Dict[str, Flow] = {}
    for c in classes:
        f, _s, _sp, _of = _aon_core(network, by_origin[c.name],
                                    _class_costs(times0, c), False)
        flows_by_class[c.name] = f
    pcu_flows = _pcu(flows_by_class)
    history: List[IterationRecord] = []
    converged = False
    used = 0
    class_gaps: Dict[str, float] = {name: 0.0 for name in names}
    for k in range(2, kmax + 2):
        used = k - 1
        times = network.travel_times(pcu_flows)
        targets: Dict[str, Flow] = {}
        loaded_all = 0.0
        excess_all = 0.0
        for c in classes:
            cc = _class_costs(times, c)
            y, sptt, _sp, _of = _aon_core(network, by_origin[c.name], cc, False)
            targets[c.name] = y
            loaded = math.fsum(flows_by_class[c.name][e] * cc[e] for e in range(m))
            loaded_all += loaded
            excess_all += max(0.0, loaded - sptt)
            class_gaps[c.name] = max(0.0, (loaded - sptt) / loaded) \
                if loaded > _EPS else 0.0
        gap = excess_all / loaded_all if loaded_all > _EPS else 0.0
        prev_pcu = list(pcu_flows)
        history.append(IterationRecord(
            iteration=k - 2, objective=beckmann_objective(network, pcu_flows),
            relative_gap=gap, average_excess_cost=excess_all
            / max(_EPS, math.fsum(ods[n2].total() for n2 in names)),
            total_travel_time=total_system_travel_time(network, pcu_flows),
            step_size=1.0 / float(k)))
        if gap <= eps:
            converged = True
            break
        step = 1.0 / float(k)
        for c in classes:
            f = flows_by_class[c.name]
            y = targets[c.name]
            flows_by_class[c.name] = [f[e] + step * (y[e] - f[e])
                                      for e in range(m)]
        pcu_flows = _pcu(flows_by_class)
        history[-1].max_flow_change = max_flow_change(prev_pcu, pcu_flows)
    times = network.travel_times(pcu_flows)
    return MultiClassResult(
        class_flows=flows_by_class, pcu_flows=pcu_flows, travel_times=times,
        class_costs={c.name: _class_costs(times, c) for c in classes},
        class_gaps=class_gaps,
        relative_gap=history[-1].relative_gap if history else 0.0,
        total_travel_time=total_system_travel_time(network, pcu_flows),
        iterations=used, converged=converged, history=history)


# ===========================================================================
# 13. ELASTIC DEMAND -- Evans' partial linearisation
# ===========================================================================
class DemandFunction(object):
    """Base class for a decreasing OD demand function ``D(u)`` of travel cost."""

    def demand(self, cost: float) -> float:
        """Trips generated at OD cost ``cost``."""
        raise NotImplementedError("subclasses must implement demand()")

    def inverse(self, trips: float) -> float:
        """The cost at which exactly ``trips`` would be generated."""
        raise NotImplementedError("subclasses must implement inverse()")

    def inverse_integral(self, trips: float) -> float:
        """``integral_0^trips D^-1(w) dw`` -- the consumer-surplus term of Evans."""
        raise NotImplementedError("subclasses must implement inverse_integral()")

    def inverse_derivative(self, trips: float) -> float:
        """``d D^-1 / d trips`` (negative); central difference unless overridden."""
        h = 1e-7 * max(1.0, abs(trips))
        lo = max(0.0, trips - h)
        hi = trips + h
        return (self.inverse(hi) - self.inverse(lo)) / (hi - lo)

    def max_demand(self) -> float:
        """Demand at zero cost -- the upper bound Evans' line search must respect."""
        return self.demand(0.0)


@dataclass
class LinearDemand(DemandFunction):
    """``D(u) = max(0, d0 - slope u)`` -- the simplest elastic-demand curve."""

    d0: float
    slope: float

    def __post_init__(self) -> None:
        self.d0 = _as_nonneg(self.d0, "d0")
        self.slope = _as_positive(self.slope, "slope")

    def demand(self, cost: float) -> float:
        return max(0.0, self.d0 - self.slope * max(0.0, cost))

    def inverse(self, trips: float) -> float:
        t = min(max(0.0, trips), self.d0)
        return (self.d0 - t) / self.slope

    def inverse_integral(self, trips: float) -> float:
        t = min(max(0.0, trips), self.d0)
        return (self.d0 * t - 0.5 * t * t) / self.slope

    def inverse_derivative(self, trips: float) -> float:
        return -1.0 / self.slope


@dataclass
class ExponentialDemand(DemandFunction):
    """``D(u) = d0 exp(-beta u)`` -- the standard gravity-model elastic demand."""

    d0: float
    beta: float

    def __post_init__(self) -> None:
        self.d0 = _as_positive(self.d0, "d0")
        self.beta = _as_positive(self.beta, "beta")

    def demand(self, cost: float) -> float:
        return self.d0 * math.exp(-self.beta * max(0.0, cost))

    def inverse(self, trips: float) -> float:
        t = min(max(1e-12 * self.d0, trips), self.d0)
        return math.log(self.d0 / t) / self.beta

    def inverse_integral(self, trips: float) -> float:
        t = min(max(0.0, trips), self.d0)
        if t <= 0.0:
            return 0.0
        return (t / self.beta) * (math.log(self.d0 / t) + 1.0)

    def inverse_derivative(self, trips: float) -> float:
        t = min(max(1e-12 * self.d0, trips), self.d0)
        return -1.0 / (self.beta * t)


@dataclass
class PowerDemand(DemandFunction):
    """Constant-elasticity demand ``D(u) = d0 (u/u0)**-elasticity`` (elasticity > 1)."""

    d0: float
    u0: float
    elasticity: float = 2.0

    def __post_init__(self) -> None:
        self.d0 = _as_positive(self.d0, "d0")
        self.u0 = _as_positive(self.u0, "u0")
        self.elasticity = _as_float(self.elasticity, "elasticity")
        if self.elasticity <= 1.0:
            raise TrafficAssignmentError(
                "PowerDemand elasticity must exceed 1 so the surplus integral "
                "converges (got %g)." % self.elasticity)

    def demand(self, cost: float) -> float:
        u = max(1e-12 * self.u0, cost)
        return self.d0 * (u / self.u0) ** (-self.elasticity)

    def max_demand(self) -> float:
        """Demand at a nominally tiny cost -- the curve is unbounded at zero cost."""
        return self.demand(1e-6 * self.u0)

    def inverse(self, trips: float) -> float:
        t = max(1e-12 * self.d0, trips)
        return self.u0 * (t / self.d0) ** (-1.0 / self.elasticity)

    def inverse_integral(self, trips: float) -> float:
        t = max(0.0, trips)
        if t <= 0.0:
            return 0.0
        p = 1.0 - 1.0 / self.elasticity
        return self.u0 * (self.d0 ** (1.0 / self.elasticity)) * t ** p / p

    def inverse_derivative(self, trips: float) -> float:
        t = max(1e-12 * self.d0, trips)
        return -(self.u0 / (self.elasticity * self.d0)) \
            * (t / self.d0) ** (-1.0 / self.elasticity - 1.0)


@dataclass
class ElasticDemandResult:
    """Equilibrium flows *and* the endogenous trip table that produced them."""

    flows: Flow
    demand: Demand
    travel_times: Cost
    costs: Cost
    od_costs: Dict[Tuple[int, int], float]
    objective: float
    total_travel_time: float
    total_demand: float
    relative_gap: float
    demand_gap: float
    iterations: int
    converged: bool
    history: List[IterationRecord] = field(default_factory=list)
    method: str = "evans"

    def summary(self) -> str:
        """One-line human summary."""
        return ("elastic demand [%-6s] iters=%-4d gap=%.3e demandGap=%.3e "
                "trips=%.4f TSTT=%.4f%s"
                % (self.method, self.iterations, self.relative_gap,
                   self.demand_gap, self.total_demand, self.total_travel_time,
                   "" if self.converged else "  [NOT CONVERGED]"))


@dataclass
class _ExcessDemandVDF(VolumeDelayFunction):
    """Cost of *not* travelling: ``D^-1(d_max - x)`` on an OD pair's excess link."""

    func: DemandFunction
    d_max: float
    capacity: float = 1.0

    def __post_init__(self) -> None:
        self.d_max = _as_nonneg(self.d_max, "d_max")
        self.capacity = max(_EPS, self.d_max)
        self.t0 = self.func.inverse(self.d_max)

    def travel_time(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        return self.func.inverse(max(0.0, self.d_max - x))

    def derivative(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        return -self.func.inverse_derivative(max(0.0, self.d_max - x))

    def integral(self, flow: float) -> float:
        x = flow if flow > 0.0 else 0.0
        rest = max(0.0, self.d_max - x)
        return (self.func.inverse_integral(self.d_max)
                - self.func.inverse_integral(rest))

    def with_capacity(self, capacity: float) -> "_ExcessDemandVDF":
        """Capacity is the demand ceiling here, so a rescale is a no-op copy."""
        return _ExcessDemandVDF(self.func, self.d_max)


def _excess_demand_assignment(network: Network,
                              demand_functions: Dict[Tuple[int, int],
                                                     DemandFunction],
                              pairs: List[Tuple[int, int]], solver: str,
                              max_iter: int, tol: float) -> ElasticDemandResult:
    """Elastic demand as an exact *fixed*-demand problem on an augmented network.

    For each OD pair we add a private sink ``z``, a free link ``d -> z`` that real
    trips funnel through and an *excess* link ``o -> z`` whose cost is
    ``D^-1(d_max - x)`` -- the price at which the marginal traveller gives up.
    Fixing the demand at ``d_max = D(u_free)`` (which the realised demand provably
    can never exceed, since equilibrium costs never fall below free flow) makes the
    two options a Wardrop choice: at equilibrium ``u_od = D^-1(d_od)``, which *is*
    the elastic-demand condition.  Because ``z`` has no outgoing links, no other OD
    movement can launder itself through the excess link.
    """
    m = network.n_links
    free_costs = network.costs(network.zeros())
    aug = network.copy()
    aug.name = network.name + "+excess"
    u_free: Dict[Tuple[int, int], float] = {}
    for o in sorted({p[0] for p in pairs}):
        dist, _pred = _dijkstra(network, o, free_costs)
        for (oo, dd) in pairs:
            if oo == o:
                if dist[dd] >= _INF:
                    raise TrafficAssignmentError(
                        "destination %d unreachable from origin %d." % (dd, o))
                u_free[(oo, dd)] = dist[dd]
    d_max = {k: max(0.0, demand_functions[k].demand(max(1e-12, u_free[k])))
             for k in pairs}
    sink: Dict[Tuple[int, int], int] = {}
    real_link: Dict[Tuple[int, int], int] = {}
    excess_link: Dict[Tuple[int, int], int] = {}
    od_aug = ODMatrix()
    next_node = network.n_nodes
    for key in pairs:
        if d_max[key] <= 0.0:
            continue
        (o, d) = key
        z = next_node
        next_node += 1
        aug.ensure_nodes(next_node)
        sink[key] = z
        real_link[key] = aug.add_link(d, z, Constant(0.0),
                                      name="sink %d->%d" % (d, z))
        excess_link[key] = aug.add_link(
            o, z, _ExcessDemandVDF(demand_functions[key], d_max[key]),
            name="excess %d->%d" % (o, z))
        od_aug.set(o, z, d_max[key])
    res = user_equilibrium(aug, od_aug, method=solver, max_iter=max_iter,
                           tol=tol)
    flows = res.flows[:m]
    realised = {k: (res.flows[real_link[k]] if k in real_link else 0.0)
                for k in pairs}
    costs = network.costs(flows)
    od_costs: Dict[Tuple[int, int], float] = {}
    for o in sorted({p[0] for p in pairs}):
        dist, _pred = _dijkstra(network, o, costs)
        for (oo, dd) in pairs:
            if oo == o:
                od_costs[(oo, dd)] = dist[dd]
    total_now = math.fsum(realised[k] for k in pairs)
    dgap = max(abs(demand_functions[k].demand(od_costs[k]) - realised[k])
               for k in pairs) / max(1.0, total_now)
    loaded = math.fsum(flows[e] * costs[e] for e in range(m))
    sptt = math.fsum(realised[k] * od_costs[k] for k in pairs)
    rgap = max(0.0, (loaded - sptt) / loaded) if loaded > _EPS else 0.0
    obj = (beckmann_objective(network, flows)
           - math.fsum(demand_functions[k].inverse_integral(realised[k])
                       for k in pairs))
    return ElasticDemandResult(
        flows=flows, demand=realised, travel_times=network.travel_times(flows),
        costs=costs, od_costs=od_costs, objective=obj,
        total_travel_time=total_system_travel_time(network, flows),
        total_demand=total_now, relative_gap=rgap, demand_gap=dgap,
        iterations=res.iterations,
        converged=(res.converged and dgap <= max(tol, 1e-9)),
        history=res.history, method="excess")


def elastic_demand_assignment(network: Network,
                              demand_functions: Dict[Tuple[int, int],
                                                     DemandFunction], *,
                              method: str = "excess", solver: str = "gp",
                              max_iter: int = 2000, tol: float = 1e-7
                              ) -> ElasticDemandResult:
    """Equilibrium when the trip table itself responds to congestion.

    Two routes to the same answer, which makes each a check on the other:

    * ``method="excess"`` (default) reduces the problem to an ordinary *fixed*
      demand equilibrium on an augmented network -- see
      :func:`_excess_demand_assignment`.  Note the default ``solver="gp"``: the
      demand ceiling is usually far above the realised demand, so the
      all-or-nothing subproblem swings the whole OD movement on and off the network
      at once and the Frank-Wolfe family zig-zags badly.  Gradient projection moves
      route flows smoothly and does not.
    * ``method="evans"`` is Evans' partial linearisation, minimising
      ``sum_e integral t_e - sum_od integral_0^{d_od} D_od^-1(w) dw`` over flows
      *and* trips at once with an exact line search along both.  Textbook, but it
      inherits Frank-Wolfe's slow tail.

    At the optimum every OD pair satisfies ``d_od = D_od(u_od)`` and Wardrop's
    principle holds on the realised trips.
    """
    if not demand_functions:
        raise TrafficAssignmentError("at least one OD demand function is required.")
    pairs = sorted(demand_functions)
    for (o, d) in pairs:
        if not (0 <= o < network.n_nodes and 0 <= d < network.n_nodes):
            raise TrafficAssignmentError(
                "OD pair (%d,%d) references a node outside the network." % (o, d))
        if not isinstance(demand_functions[(o, d)], DemandFunction):
            raise TrafficAssignmentError(
                "demand_functions[(%d,%d)] must be a DemandFunction." % (o, d))
    mode = str(method).lower()
    if mode not in ("excess", "evans"):
        raise TrafficAssignmentError("method must be 'excess' or 'evans'.")
    if mode == "excess":
        return _excess_demand_assignment(network, demand_functions, pairs,
                                         solver, max_iter, tol)
    m = network.n_links
    kmax = _as_index(max_iter, "max_iter")
    eps = _as_nonneg(tol, "tol")
    caps = {k: demand_functions[k].max_demand() for k in pairs}

    def _od_costs(costs: Sequence[float]) -> Dict[Tuple[int, int], float]:
        out: Dict[Tuple[int, int], float] = {}
        origins = sorted({o for (o, _d) in pairs})
        for o in origins:
            dist, _pred = _dijkstra(network, o, costs)
            for (oo, dd) in pairs:
                if oo != o:
                    continue
                if dist[dd] >= _INF:
                    raise TrafficAssignmentError(
                        "destination %d unreachable from origin %d." % (dd, o))
                out[(oo, dd)] = dist[dd]
        return out

    costs = network.costs(network.zeros())
    u = _od_costs(costs)
    dem = {k: min(caps[k], demand_functions[k].demand(u[k])) for k in pairs}
    flows, _sptt, _sp, _of = _aon_core(network, ODMatrix(dem).by_origin(), costs,
                                       False)
    history: List[IterationRecord] = []
    converged = False
    used = 0
    rgap = _INF
    dgap = _INF
    for k in range(1, kmax + 1):
        used = k
        costs = network.costs(flows)
        u = _od_costs(costs)
        target = {kk: min(caps[kk], demand_functions[kk].demand(u[kk]))
                  for kk in pairs}
        y, _sptt, _sp, _of = _aon_core(network, ODMatrix(target).by_origin(),
                                       costs, False)
        loaded = math.fsum(flows[e] * costs[e] for e in range(m))
        sptt = math.fsum(dem[kk] * u[kk] for kk in pairs)
        rgap = max(0.0, (loaded - sptt) / loaded) if loaded > _EPS else 0.0
        total_now = math.fsum(dem[kk] for kk in pairs)
        dgap = max(abs(target[kk] - dem[kk]) for kk in pairs) \
            / max(1.0, total_now)
        obj = (beckmann_objective(network, flows)
               - math.fsum(demand_functions[kk].inverse_integral(dem[kk])
                           for kk in pairs))
        history.append(IterationRecord(
            iteration=k - 1, objective=obj, relative_gap=rgap,
            average_excess_cost=max(0.0, (loaded - sptt) / max(_EPS, total_now)),
            total_travel_time=total_system_travel_time(network, flows)))
        if rgap <= eps and dgap <= eps:
            converged = True
            break
        df = [y[e] - flows[e] for e in range(m)]
        dd = {kk: target[kk] - dem[kk] for kk in pairs}
        active = [e for e in range(m) if df[e] != 0.0]

        def _dz(lam: float) -> float:
            a = math.fsum(df[e] * (network.links[e].vdf.travel_time(
                max(0.0, flows[e] + lam * df[e])) + network.links[e].toll)
                for e in active)
            b = math.fsum(dd[kk] * demand_functions[kk].inverse(
                max(0.0, dem[kk] + lam * dd[kk])) for kk in pairs
                if dd[kk] != 0.0)
            return a - b

        if _dz(0.0) >= 0.0:
            lam = 0.0
        elif _dz(1.0) <= 0.0:
            lam = 1.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(_BISECT_ITERS):
                mid = 0.5 * (lo + hi)
                if _dz(mid) > 0.0:
                    hi = mid
                else:
                    lo = mid
                if hi - lo < 1e-15:
                    break
            lam = 0.5 * (lo + hi)
        if lam <= 0.0:
            converged = True
            break
        history[-1].step_size = lam
        history[-1].max_flow_change = max(abs(lam * df[e]) for e in range(m)) \
            if m else 0.0
        flows = [max(0.0, flows[e] + lam * df[e]) for e in range(m)]
        dem = {kk: max(0.0, dem[kk] + lam * dd[kk]) for kk in pairs}
    costs = network.costs(flows)
    u = _od_costs(costs)
    return ElasticDemandResult(
        flows=flows, demand=dict(dem), travel_times=network.travel_times(flows),
        costs=costs, od_costs=u,
        objective=(beckmann_objective(network, flows)
                   - math.fsum(demand_functions[kk].inverse_integral(dem[kk])
                               for kk in pairs)),
        total_travel_time=total_system_travel_time(network, flows),
        total_demand=math.fsum(dem[kk] for kk in pairs),
        relative_gap=rgap, demand_gap=dgap, iterations=used,
        converged=converged, history=history)


# ===========================================================================
# 14. TURN PENALTIES -- assignment on the expanded (link-to-link) graph
# ===========================================================================
@dataclass
class TurnExpansion:
    """The entry/exit expansion of a network, where turns are links of their own."""

    network: Network
    n_original_links: int
    n_original_nodes: int
    link_of: List[int]
    turn_link: Dict[Tuple[int, int], int]
    origin_node: Dict[int, int]
    dest_node: Dict[int, int]

    def expand_od(self, od: Any) -> ODMatrix:
        """Re-index a trip table onto the expanded graph's terminal nodes."""
        odm = _as_od(od)
        out = ODMatrix()
        for (o, d), trips in odm.items():
            if o not in self.origin_node or d not in self.dest_node:
                raise TrafficAssignmentError(
                    "OD pair (%d,%d) has no terminal in the expansion." % (o, d))
            out.set(self.origin_node[o], self.dest_node[d], trips)
        return out

    def fold_flows(self, expanded_flows: Sequence[float]) -> Flow:
        """Project expanded-graph flows back onto the original links."""
        return [expanded_flows[self.link_of[a]]
                for a in range(self.n_original_links)]

    def turn_volumes(self, expanded_flows: Sequence[float]
                     ) -> Dict[Tuple[int, int], float]:
        """Flow through every permitted turn, keyed by (incoming link, outgoing link)."""
        return {ab: expanded_flows[e] for ab, e in sorted(self.turn_link.items())}


def expand_with_turn_penalties(network: Network,
                               penalties: Optional[Dict[Tuple[int, int],
                                                        float]] = None, *,
                               banned: Optional[Iterable[Tuple[int, int]]] = None,
                               default_penalty: float = 0.0) -> TurnExpansion:
    """Build the expanded graph in which every turn is an explicit, priced link.

    Each original link ``a`` becomes an *entry* node and an *exit* node joined by a
    link carrying ``a``'s volume-delay function -- so ``a``'s flow is still
    accumulated in exactly one place, which naive link-to-link expansions get wrong.
    Turns become zero-capacity constant-cost links between an exit and an entry, and
    per-node origin / destination terminals stop through traffic from laundering a
    banned turn through a zone.
    """
    m = network.n_links
    n = network.n_nodes
    pen = dict(penalties or {})
    ban = set(banned or ())
    for (a, b) in list(pen) + list(ban):
        if not (0 <= a < m and 0 <= b < m):
            raise TrafficAssignmentError("turn (%r,%r) references a missing link."
                                         % (a, b))
    dp = _as_nonneg(default_penalty, "default_penalty")
    exp = Network(network.name + "+turns")
    exp.ensure_nodes(2 * m + 2 * n)
    link_of: List[int] = []
    for a, lk in enumerate(network.links):
        link_of.append(exp.add_link(2 * a, 2 * a + 1, lk.vdf, toll=lk.toll,
                                    length=lk.length,
                                    name=lk.name or "link%d" % a))
    turn_link: Dict[Tuple[int, int], int] = {}
    for a, la in enumerate(network.links):
        for (b, _h) in network.out_links(la.head):
            if (a, b) in ban:
                continue
            p = _as_nonneg(pen.get((a, b), dp), "turn penalty")
            turn_link[(a, b)] = exp.add_link(
                2 * a + 1, 2 * b, Constant(p), name="turn %d->%d" % (a, b))
    origin_node = {o: 2 * m + o for o in range(n)}
    dest_node = {d: 2 * m + n + d for d in range(n)}
    for o in range(n):
        for (a, _h) in network.out_links(o):
            exp.add_link(origin_node[o], 2 * a, Constant(0.0),
                         name="access o%d->link%d" % (o, a))
    for d in range(n):
        for (a, _t) in network.in_links(d):
            exp.add_link(2 * a + 1, dest_node[d], Constant(0.0),
                         name="egress link%d->d%d" % (a, d))
    return TurnExpansion(exp, m, n, link_of, turn_link, origin_node, dest_node)


def turn_penalty_assignment(network: Network, od: Any,
                            penalties: Optional[Dict[Tuple[int, int],
                                                     float]] = None, *,
                            banned: Optional[Iterable[Tuple[int, int]]] = None,
                            default_penalty: float = 0.0,
                            method: str = "bfw", max_iter: int = 300,
                            tol: float = 1e-9) -> AssignmentResult:
    """User equilibrium with explicit turn penalties and banned turns.

    Solves on the expanded graph and folds the answer back onto the original links;
    the per-turn volumes (the thing a junction designer actually wants) come back in
    ``result.extra["turn_volumes"]``.
    """
    odm = _as_od(od)
    exp = expand_with_turn_penalties(network, penalties, banned=banned,
                                     default_penalty=default_penalty)
    exp_od = exp.expand_od(odm)
    res = user_equilibrium(exp.network, exp_od, method=method, max_iter=max_iter,
                           tol=tol)
    flows = exp.fold_flows(res.flows)
    turns = exp.turn_volumes(res.flows)
    out = AssignmentResult(
        method="turn-penalty UE (%s)" % res.method, flows=flows,
        travel_times=network.travel_times(flows), costs=network.costs(flows),
        objective=res.objective,
        total_travel_time=total_system_travel_time(network, flows),
        relative_gap=res.relative_gap,
        average_excess_cost=res.average_excess_cost,
        iterations=res.iterations, converged=res.converged, history=res.history,
        extra={"turn_volumes": turns, "expansion": exp,
               "expanded_result": res,
               "turn_delay": math.fsum(
                   turns[ab] * exp.network.links[e].vdf.travel_time(turns[ab])
                   for ab, e in sorted(exp.turn_link.items()))})
    return out


# ===========================================================================
# 15. FLOW DECOMPOSITION, WARDROP CHECK AND SELECT-LINK ANALYSIS
# ===========================================================================
def decompose_flows(network: Network, od: Any, flows: Sequence[float], *,
                    origin_flows: Optional[Dict[int, Flow]] = None,
                    costs: Optional[Sequence[float]] = None,
                    tol: float = 1e-9, strict: bool = False) -> PathSet:
    """Turn a link-flow solution back into route flows.

    With ``origin_flows`` (which every solver here can keep) the decomposition is
    *exact*: each origin's own flow vector satisfies conservation with that origin
    as the only source, so repeatedly peeling off the cheapest positive-flow route
    reproduces a valid route solution.  Without it the aggregate flow is peeled OD
    pair by OD pair, which conserves vehicles but can attribute a route to the wrong
    origin on networks where two origins' flows overlap -- set ``strict`` to have
    that surface as an error instead of a silent fallback.

    In the aggregate case the residual is carried *across* origins.  Resetting it
    per origin lets several origins each spend the same link flow, so the route set
    re-loads to something that is not the input at all -- 30% out on a link, and
    ``wardrop_check`` then reports a true equilibrium as violated.  ``strict``
    verifies the re-loading and raises when it does not reproduce ``flows``.
    """
    odm = _as_od(od)
    _check_od(network, odm)
    f = _check_flows(network, flows)
    c = list(costs) if costs is not None else network.costs(f)
    m = network.n_links
    ps = PathSet()
    by_origin = odm.by_origin()
    big = 1e18
    shared = list(f)
    for o in sorted(by_origin):
        residual = (list(origin_flows[o])
                    if (origin_flows is not None and o in origin_flows)
                    else shared)
        remaining = {d: trips for (d, trips) in by_origin[o] if trips > 0.0}
        guard = 0
        while remaining:
            guard += 1
            if guard > 20 * (m + len(remaining) + 1):
                raise TrafficAssignmentError(
                    "flow decomposition from origin %d did not terminate." % o)
            masked = [c[e] if residual[e] > tol else big for e in range(m)]
            dist, pred = _dijkstra(network, o, masked)
            tree = ShortestPathTree(o, dist, pred).bind(network)
            target = None
            for d in sorted(remaining):
                if dist[d] < big / 2.0:
                    target = d
                    break
            if target is None:
                if strict:
                    raise TrafficAssignmentError(
                        "cannot decompose origin %d: %g trips have no positive-flow "
                        "route left." % (o, math.fsum(remaining.values())))
                dist, pred = _dijkstra(network, o, c)
                tree = ShortestPathTree(o, dist, pred).bind(network)
                target = sorted(remaining)[0]
                links = tree.path_links(target)
                amt = remaining.pop(target)
                ps.add(o, target, links, amt, tree.path_nodes(network, target))
                continue
            links = tree.path_links(target)
            cap = min(residual[e] for e in links) if links else remaining[target]
            amt = min(remaining[target], cap)
            if amt <= tol:
                amt = remaining[target]
            for e in links:
                residual[e] = max(0.0, residual[e] - amt)
            ps.add(o, target, links, amt, tree.path_nodes(network, target))
            left = remaining[target] - amt
            if left > tol:
                remaining[target] = left
            else:
                remaining.pop(target)
    ps.update_costs(network, c)
    if strict:
        reloaded = ps.link_flows(network)
        worst = max((abs(reloaded[e] - f[e]) for e in range(m)), default=0.0)
        scale = odm.total()
        if worst > 1e-9 * scale:
            raise TrafficAssignmentError(
                "flow decomposition does not reproduce the input: link flow is out "
                "by %g veh on a %g-veh table (pass origin_flows for an exact "
                "decomposition)." % (worst, scale))
    return ps


@dataclass
class WardropReport:
    """Direct test of Wardrop's first principle on the decomposed route flows."""

    satisfied: bool
    max_used_gap: float
    max_unused_violation: float
    tolerance: float
    per_pair: Dict[Tuple[int, int], Tuple[float, float, float]]

    def worst_pair(self) -> Optional[Tuple[int, int]]:
        """The OD pair with the widest spread of used-route costs."""
        if not self.per_pair:
            return None
        return max(sorted(self.per_pair),
                   key=lambda k: self.per_pair[k][1] - self.per_pair[k][0])

    def as_dict(self) -> Dict[str, Any]:
        """JSON-able camelCase summary."""
        return {"satisfied": self.satisfied, "maxUsedGap": self.max_used_gap,
                "maxUnusedViolation": self.max_unused_violation,
                "tolerance": self.tolerance,
                "pairs": [{"origin": o, "destination": d, "minUsed": v[0],
                           "maxUsed": v[1], "minAny": v[2]}
                          for (o, d), v in sorted(self.per_pair.items())]}


def wardrop_check(network: Network, pathset: PathSet,
                  costs: Optional[Sequence[float]] = None, *,
                  tol: float = 1e-6, flow_tol: float = 1e-9) -> WardropReport:
    """Assert Wardrop's first principle directly on route flows.

    Every *used* route of an OD pair must cost the same, and no *unused* route in
    the choice set may be cheaper than the used ones.  Both are reported as
    absolute cost gaps, scaled against the cheapest route so the tolerance means the
    same thing on a 5-minute network and a 5-hour one.

    ``flow_tol`` decides which routes count as used and is likewise *relative*, to
    the flow already loaded on that OD pair.  An absolute veh/h literal filtered
    every route out of a small trip table, leaving the check with nothing to examine
    and returning a confident ``satisfied=True`` for a network that plainly violates
    Wardrop.
    """
    c = list(costs) if costs is not None else network.costs(
        pathset.link_flows(network))
    per: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    worst_used = 0.0
    worst_unused = 0.0
    for (o, d) in pathset.pairs():
        bucket = pathset.paths(o, d)
        if not bucket:
            continue
        pc = [path_cost(network, p, c) for p in bucket]
        loaded = math.fsum(p.flow for p in bucket if p.flow > 0.0)
        cut = flow_tol * loaded
        used = [pc[i] for i, p in enumerate(bucket) if p.flow > cut]
        if not used:
            continue
        lo_used, hi_used, lo_any = min(used), max(used), min(pc)
        per[(o, d)] = (lo_used, hi_used, lo_any)
        scale = max(1.0, abs(lo_used))
        worst_used = max(worst_used, (hi_used - lo_used) / scale)
        worst_unused = max(worst_unused, (lo_used - lo_any) / scale)
    ok = worst_used <= tol and worst_unused <= tol
    return WardropReport(ok, worst_used, worst_unused, tol, per)


@dataclass
class SelectLinkResult:
    """Which trips use a chosen link (or zone), and where else they go."""

    selector: str
    total_flow: float
    od_flows: Dict[Tuple[int, int], float]
    link_flows: Flow
    link: int = -1

    def top_pairs(self, n: int = 5) -> List[Tuple[Tuple[int, int], float]]:
        """The ``n`` heaviest OD pairs through the selection."""
        return sorted(self.od_flows.items(), key=lambda kv: (-kv[1], kv[0]))[:n]

    def as_dict(self) -> Dict[str, Any]:
        """JSON-able camelCase summary."""
        return {"selector": self.selector, "link": self.link,
                "totalFlow": self.total_flow,
                "odFlows": [{"origin": o, "destination": d, "flow": v}
                            for (o, d), v in sorted(self.od_flows.items())],
                "linkFlows": list(self.link_flows)}


def select_link_analysis(network: Network, pathset: PathSet,
                         link_index: int) -> SelectLinkResult:
    """Every OD movement that uses one link, and the rest of the network it loads."""
    e0 = _as_index(link_index, "link_index")
    if e0 >= network.n_links:
        raise TrafficAssignmentError("link index %d out of range." % e0)
    od_flows: Dict[Tuple[int, int], float] = {}
    link_flows = network.zeros()
    total = 0.0
    for p in pathset.all_paths():
        if p.flow <= 0.0 or e0 not in p.links:
            continue
        total += p.flow
        key = (p.origin, p.destination)
        od_flows[key] = od_flows.get(key, 0.0) + p.flow
        for e in p.links:
            link_flows[e] += p.flow
    return SelectLinkResult("link %d" % e0, total, od_flows, link_flows, e0)


def select_zone_analysis(network: Network, pathset: PathSet, zone: int, *,
                         role: str = "either") -> SelectLinkResult:
    """Everything generated by, attracted to, or touching a zone."""
    z = _as_index(zone, "zone")
    r = str(role).lower()
    if r not in ("origin", "destination", "either"):
        raise TrafficAssignmentError(
            "role must be 'origin', 'destination' or 'either'.")
    od_flows: Dict[Tuple[int, int], float] = {}
    link_flows = network.zeros()
    total = 0.0
    for p in pathset.all_paths():
        if p.flow <= 0.0:
            continue
        hit = ((r == "origin" and p.origin == z)
               or (r == "destination" and p.destination == z)
               or (r == "either" and (p.origin == z or p.destination == z)))
        if not hit:
            continue
        total += p.flow
        key = (p.origin, p.destination)
        od_flows[key] = od_flows.get(key, 0.0) + p.flow
        for e in p.links:
            link_flows[e] += p.flow
    return SelectLinkResult("zone %d (%s)" % (z, r), total, od_flows, link_flows)


# ===========================================================================
# 16. SENSITIVITY OF THE EQUILIBRIUM TO CAPACITY
# ===========================================================================
@dataclass
class SensitivityResult:
    """Finite-difference derivatives of the equilibrium with respect to capacity."""

    links: List[int]
    delta: float
    base_total_travel_time: float
    base_flows: Flow
    dtstt_dcapacity: List[float]
    dflow_dcapacity: List[List[float]]
    dtime_dcapacity: List[List[float]]

    def best_link(self) -> int:
        """The link whose capacity buys the largest reduction in total travel time."""
        if not self.links:
            return -1
        i = min(range(len(self.links)), key=lambda j: self.dtstt_dcapacity[j])
        return self.links[i]

    def as_dict(self) -> Dict[str, Any]:
        """JSON-able camelCase summary."""
        return {"delta": self.delta,
                "baseTotalTravelTime": self.base_total_travel_time,
                "links": [{"link": e, "dTstt_dCapacity": self.dtstt_dcapacity[i]}
                          for i, e in enumerate(self.links)]}


def capacity_sensitivity(network: Network, od: Any, *,
                         links: Optional[Sequence[int]] = None,
                         delta: float = 0.01, method: str = "bfw",
                         max_iter: int = 400,
                         tol: float = 1e-11) -> SensitivityResult:
    """Central-difference derivative of the equilibrium with respect to capacity.

    Perturbs one link's capacity up and down by a relative ``delta``, re-solves the
    equilibrium each time, and differences the results.  ``dTSTT/dcapacity`` is the
    marginal benefit of widening that link and is exactly the gradient a bi-level
    network-design search needs; it is normally *negative* (more capacity, less
    delay) but on a Braess link it can be positive, which is the paradox showing up
    in the derivative rather than in a re-solve.
    """
    odm = _as_od(od)
    _check_od(network, odm)
    d = _as_positive(delta, "delta")
    if d >= 1.0:
        raise TrafficAssignmentError("delta must be < 1 (got %g)." % d)
    base = user_equilibrium(network, odm, method=method, max_iter=max_iter, tol=tol)
    base_tstt = total_system_travel_time(network, base.flows)
    targets = list(range(network.n_links)) if links is None \
        else [_as_index(e, "link") for e in links]
    dtstt: List[float] = []
    dflow: List[List[float]] = []
    dtime: List[List[float]] = []
    for e in targets:
        if not 0 <= e < network.n_links:
            raise TrafficAssignmentError("link index %d out of range." % e)
        cap = network.links[e].capacity
        up = network.with_scaled_capacity(e, 1.0 + d)
        dn = network.with_scaled_capacity(e, 1.0 - d)
        r_up = user_equilibrium(up, odm, method=method, max_iter=max_iter, tol=tol)
        r_dn = user_equilibrium(dn, odm, method=method, max_iter=max_iter, tol=tol)
        h = 2.0 * d * cap
        dtstt.append((total_system_travel_time(up, r_up.flows)
                      - total_system_travel_time(dn, r_dn.flows)) / h)
        dflow.append([(r_up.flows[i] - r_dn.flows[i]) / h
                      for i in range(network.n_links)])
        t_up = up.travel_times(r_up.flows)
        t_dn = dn.travel_times(r_dn.flows)
        dtime.append([(t_up[i] - t_dn[i]) / h for i in range(network.n_links)])
    return SensitivityResult(targets, d, base_tstt, base.flows, dtstt, dflow,
                             dtime)


# ===========================================================================
# 17. NETWORK BUILDERS AND CLOSED-FORM ORACLES
# ===========================================================================
def two_link_network(a1: float, b1: float, a2: float, b2: float, *,
                     demand: float = 1.0) -> Tuple[Network, ODMatrix]:
    """Two affine links in parallel from node 0 to node 1 -- the analytic test case."""
    net = Network("two-link")
    net.ensure_nodes(2)
    net.add_link(0, 1, Linear(_as_nonneg(a1, "a1"), _as_nonneg(b1, "b1"),
                              capacity=max(1e-9, demand)), name="upper")
    net.add_link(0, 1, Linear(_as_nonneg(a2, "a2"), _as_nonneg(b2, "b2"),
                              capacity=max(1e-9, demand)), name="lower")
    return net, ODMatrix({(0, 1): _as_nonneg(demand, "demand")})


def two_link_ue_closed_form(a1: float, b1: float, a2: float, b2: float,
                            demand: float) -> Tuple[float, float, float, float]:
    """Exact user equilibrium of the two-link network: ``(x1, x2, c1, c2)``.

    With both links used the costs equalise, ``a1 + b1 x1 = a2 + b2 (D - x1)``, so
    ``x1 = (a2 - a1 + b2 D) / (b1 + b2)`` -- clamped to ``[0, D]`` for the corner
    solutions where one link is simply never worth using.
    """
    d = _as_nonneg(demand, "demand")
    s = b1 + b2
    if s <= 0.0:
        if a1 < a2:
            x1 = d
        elif a2 < a1:
            x1 = 0.0
        else:
            x1 = 0.5 * d
    else:
        x1 = (a2 - a1 + b2 * d) / s
        x1 = min(d, max(0.0, x1))
    x2 = d - x1
    return x1, x2, a1 + b1 * x1, a2 + b2 * x2


def two_link_so_closed_form(a1: float, b1: float, a2: float, b2: float,
                            demand: float) -> Tuple[float, float, float]:
    """Exact system optimum of the two-link network: ``(x1, x2, total travel time)``.

    The marginal costs equalise, ``a1 + 2 b1 x1 = a2 + 2 b2 (D - x1)``.
    """
    d = _as_nonneg(demand, "demand")
    s = 2.0 * (b1 + b2)
    if s <= 0.0:
        if a1 < a2:
            x1 = d
        elif a2 < a1:
            x1 = 0.0
        else:
            x1 = 0.5 * d
    else:
        x1 = (a2 - a1 + 2.0 * b2 * d) / s
        x1 = min(d, max(0.0, x1))
    x2 = d - x1
    return x1, x2, x1 * (a1 + b1 * x1) + x2 * (a2 + b2 * x2)


def grid_network(rows: int, cols: int, *, free_flow_time: float = 1.0,
                 capacity: float = 200.0, alpha: float = 0.15,
                 beta: float = 4.0, demand: float = 0.0
                 ) -> Tuple[Network, ODMatrix]:
    """A ``rows x cols`` bidirectional BPR grid, optionally with corner-to-corner demand."""
    r = _as_index(rows, "rows")
    c = _as_index(cols, "cols")
    if r < 1 or c < 1:
        raise TrafficAssignmentError("grid needs at least one row and one column.")
    t0 = _as_positive(free_flow_time, "free_flow_time")
    cap = _as_positive(capacity, "capacity")
    net = Network("grid%dx%d" % (r, c))
    net.ensure_nodes(r * c)

    def nid(i: int, j: int) -> int:
        return i * c + j

    for i in range(r):
        for j in range(c):
            if j + 1 < c:
                net.add_link(nid(i, j), nid(i, j + 1), BPR(t0, cap, alpha, beta),
                             length=1.0)
                net.add_link(nid(i, j + 1), nid(i, j), BPR(t0, cap, alpha, beta),
                             length=1.0)
            if i + 1 < r:
                net.add_link(nid(i, j), nid(i + 1, j), BPR(t0, cap, alpha, beta),
                             length=1.0)
                net.add_link(nid(i + 1, j), nid(i, j), BPR(t0, cap, alpha, beta),
                             length=1.0)
    od = ODMatrix()
    dem = _as_nonneg(demand, "demand")
    if dem > 0.0 and r * c > 1:
        od.set(0, r * c - 1, dem)
    return net, od


def random_affine_network(n_nodes: int = 8, n_links: int = 18, *, seed: int = 0,
                          n_od: int = 3, demand_scale: float = 10.0,
                          max_intercept: float = 10.0, max_slope: float = 2.0
                          ) -> Tuple[Network, ODMatrix]:
    """A seeded random strongly-connected network with purely affine link costs.

    Affine costs are the case in which the price of anarchy provably cannot exceed
    ``4/3``, so a stream of these networks is the natural fuzz target for that
    bound.  A directed ring guarantees every destination is reachable from every
    origin before any random links are added.
    """
    n = _as_index(n_nodes, "n_nodes")
    if n < 3:
        raise TrafficAssignmentError("random_affine_network needs >= 3 nodes.")
    rng = random.Random(seed)
    net = Network("random-affine-%d" % seed)
    net.ensure_nodes(n)
    existing: Set[Tuple[int, int]] = set()
    for i in range(n):
        j = (i + 1) % n
        net.add_link(i, j, Linear(rng.uniform(0.5, max_intercept),
                                  rng.uniform(0.0, max_slope),
                                  capacity=max(1.0, demand_scale)))
        existing.add((i, j))
    extra = max(0, _as_index(n_links, "n_links") - n)
    tries = 0
    while extra > 0 and tries < 40 * (extra + 1):
        tries += 1
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j or (i, j) in existing:
            continue
        net.add_link(i, j, Linear(rng.uniform(0.5, max_intercept),
                                  rng.uniform(0.0, max_slope),
                                  capacity=max(1.0, demand_scale)))
        existing.add((i, j))
        extra -= 1
    od = ODMatrix()
    pairs = max(1, _as_index(n_od, "n_od"))
    guard = 0
    while len(od) < pairs and guard < 200:
        guard += 1
        o = rng.randrange(n)
        d = rng.randrange(n)
        if o == d or od.get(o, d) > 0.0:
            continue
        od.set(o, d, round(rng.uniform(0.2, 1.0) * demand_scale, 6))
    return net, od


# ===========================================================================
# 18. PAYLOADS -- plain JSON-able dictionaries for a frontend
# ===========================================================================
def _json_safe(obj: Any) -> Any:
    """Coerce a nested structure into strictly JSON-able primitives."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return str(obj)


def _congestion_band(vc: float) -> str:
    """Level-of-service band from a volume/capacity ratio."""
    if vc < 0.6:
        return "free"
    if vc < 0.8:
        return "busy"
    if vc < 1.0:
        return "congested"
    return "over-capacity"


def build_traffic_assignment_payload(network: Network, result: AssignmentResult, *,
                                     od: Any = None,
                                     pathset: Optional[PathSet] = None,
                                     top_links: int = 0) -> Dict[str, Any]:
    """Strict-JSON description of one assignment run: links, convergence, totals."""
    flows = result.flows
    times = result.travel_times
    costs = result.costs
    vc = network.vc_ratios(flows)
    links = []
    for i, lk in enumerate(network.links):
        links.append({"id": i, "label": lk.label(i), "tail": lk.tail,
                      "head": lk.head, "flow": flows[i], "travelTime": times[i],
                      "cost": costs[i], "toll": lk.toll,
                      "capacity": lk.capacity,
                      "freeFlowTime": lk.vdf.travel_time(0.0),
                      "volumeCapacityRatio": vc[i],
                      "band": _congestion_band(vc[i]),
                      "vehicleHours": flows[i] * times[i],
                      "vdf": type(lk.vdf).__name__})
    if top_links > 0:
        links = sorted(links, key=lambda d: -d["flow"])[:int(top_links)]
    payload: Dict[str, Any] = {
        "method": result.method,
        "converged": result.converged,
        "iterations": result.iterations,
        "relativeGap": result.relative_gap,
        "averageExcessCost": result.average_excess_cost,
        "beckmannObjective": result.objective,
        "totalSystemTravelTime": result.total_travel_time,
        "totalFlow": math.fsum(flows),
        "maxVolumeCapacityRatio": max(vc) if vc else 0.0,
        "links": links,
        "convergence": [r.as_dict() for r in result.history],
    }
    if od is not None:
        odm = _as_od(od)
        payload["demand"] = {"pairs": len(odm), "trips": odm.total(),
                             "cells": [{"origin": o, "destination": d,
                                        "trips": v}
                                       for (o, d), v in odm.items()]}
    ps = pathset if pathset is not None else result.paths
    if ps is not None:
        payload["paths"] = [{"origin": p.origin, "destination": p.destination,
                             "links": list(p.links), "flow": p.flow,
                             "cost": p.cost} for p in ps.all_paths()]
    return _json_safe(payload)


def build_a2ui_payload(network: Network, result: AssignmentResult, *,
                       od: Any = None,
                       comparison: Optional[Dict[str, AssignmentResult]] = None,
                       poa: Optional[PriceOfAnarchyResult] = None,
                       braess: Optional[BraessResult] = None,
                       title: str = "Traffic assignment") -> Dict[str, Any]:
    """A2UI view-model: headline cards, a convergence series, a link table, a map."""
    flows = result.flows
    vc = network.vc_ratios(flows)
    cards = [
        {"label": "Total system travel time", "value": result.total_travel_time,
         "format": "0.00", "unit": "veh-h"},
        {"label": "Relative gap", "value": result.relative_gap,
         "format": "0.00e+0"},
        {"label": "Iterations", "value": result.iterations, "format": "0"},
        {"label": "Max V/C", "value": max(vc) if vc else 0.0, "format": "0.000"},
    ]
    if poa is not None:
        cards.append({"label": "Price of anarchy",
                      "value": poa.price_of_anarchy, "format": "0.0000"})
        cards.append({"label": "Excess delay vs SO", "value": poa.excess_delay(),
                      "format": "0.00", "unit": "veh-h"})
    if braess is not None:
        cards.append({"label": "Braess link",
                      "value": braess.best_link if braess.paradox else None,
                      "format": "0"})
    rows = []
    for i, lk in enumerate(network.links):
        rows.append([i, lk.label(i), lk.tail, lk.head, flows[i],
                     result.travel_times[i], vc[i], _congestion_band(vc[i])])
    view: Dict[str, Any] = {
        "component": "traffic-assignment",
        "title": title,
        "subtitle": "%s on %s" % (result.method, network.name),
        "cards": cards,
        "series": {
            "iteration": [r.iteration for r in result.history],
            "beckmannObjective": [r.objective for r in result.history],
            "relativeGap": [r.relative_gap for r in result.history],
            "averageExcessCost": [r.average_excess_cost for r in result.history],
            "totalTravelTime": [r.total_travel_time for r in result.history],
        },
        "table": {
            "columns": ["id", "link", "tail", "head", "flow", "time", "v/c",
                        "band"],
            "rows": rows,
        },
        "map": {
            "nodes": [{"id": v} for v in range(network.n_nodes)],
            "links": [{"id": i, "from": lk.tail, "to": lk.head,
                       "flow": flows[i], "vc": vc[i],
                       "band": _congestion_band(vc[i])}
                      for i, lk in enumerate(network.links)],
        },
        "legend": [{"band": b, "upperVolumeCapacity": u}
                   for (b, u) in (("free", 0.6), ("busy", 0.8),
                                  ("congested", 1.0), ("over-capacity", None))],
    }
    if od is not None:
        odm = _as_od(od)
        view["demand"] = {"pairs": len(odm), "trips": odm.total()}
    if comparison:
        view["comparison"] = [
            {"solver": name, "totalSystemTravelTime": r.total_travel_time,
             "beckmannObjective": r.objective, "relativeGap": r.relative_gap,
             "iterations": r.iterations, "converged": r.converged}
            for name, r in sorted(comparison.items())]
        base = result.flows
        view["agreement"] = {
            name: max(abs(r.flows[e] - base[e]) for e in range(network.n_links))
            if network.n_links else 0.0
            for name, r in sorted(comparison.items())}
    if braess is not None:
        view["braess"] = braess.as_dict()
    if poa is not None:
        view["priceOfAnarchy"] = poa.as_dict()
    return _json_safe(view)


# ===========================================================================
# 19. DEMO
# ===========================================================================
def _numeric_integral(vdf: VolumeDelayFunction, x: float, n: int = 2000) -> float:
    """Composite Simpson quadrature -- an independent check on analytic integrals."""
    if x <= 0.0:
        return 0.0
    if n % 2:
        n += 1
    h = x / n
    total = vdf.travel_time(0.0) + vdf.travel_time(x)
    total += 4.0 * math.fsum(vdf.travel_time(h * i) for i in range(1, n, 2))
    total += 2.0 * math.fsum(vdf.travel_time(h * i) for i in range(2, n, 2))
    return total * h / 3.0


def _demo() -> None:
    sep = "=" * 70
    print(sep)
    print("  TRAFFIC ASSIGNMENT (pure Python) - demo")
    print(sep)

    # ---------------------------------------------------------------- VDFs
    print("\n-- 1. volume-delay functions (per-link parameters, analytic integrals)")
    cap = 100.0
    family: List[VolumeDelayFunction] = [
        BPR(10.0, cap, 0.15, 4.0),
        Davidson(10.0, cap, 0.25),
        ModifiedDavidson(10.0, cap, 0.25, 0.85),
        Conical(10.0, cap, 8.0),
        Akcelik(10.0, cap, 0.1, 1.0),
        Polynomial([10.0, 2.0, 0.0, 0.0, 6.0], cap),
        Linear(10.0, 0.05, capacity=cap),
        Constant(10.0),
    ]
    print("  %-18s %8s %8s %8s %10s %11s" % ("function", "t(0)", "t(0.8c)",
                                             "t(c)", "t(1.2c)", "int err"))
    worst_int = 0.0
    for vdf in family:
        assert isinstance(vdf, VolumeDelayFunction)
        x = 0.9 * cap                      # below the Davidson pole, so Simpson
        exact = vdf.integral(x)            # is itself accurate enough to judge
        approx = _numeric_integral(vdf, x)
        err = abs(exact - approx) / max(1.0, abs(approx))
        worst_int = max(worst_int, err)
        print("  %-18s %8.4f %8.4f %8.4f %10.4f %11.2e"
              % (type(vdf).__name__, vdf.travel_time(0.0),
                 vdf.travel_time(0.8 * cap), vdf.travel_time(cap),
                 vdf.travel_time(1.2 * cap), err))
    print("  worst analytic-vs-Simpson integral error at 0.9c: %.3e" % worst_int)
    mc = MarginalCostVDF(BPR(10.0, cap, 0.15, 4.0))
    print("  MarginalCostVDF: t(80)=%.4f  mc(80)=%.4f  integral(80)=%.4f "
          "(= 80*t(80)=%.4f)"
          % (BPR(10.0, cap, 0.15, 4.0).travel_time(80.0), mc.travel_time(80.0),
             mc.integral(80.0), 80.0 * BPR(10.0, cap, 0.15, 4.0).travel_time(80.0)))
    print("  Davidson.describe(): %s" % family[1].describe())

    # ------------------------------------------------- two-link closed form
    print("\n-- 2. two parallel links: closed-form UE and SO")
    a1, b1, a2, b2, dem = 10.0, 1.0, 20.0, 0.5, 30.0
    net2, od2 = two_link_network(a1, b1, a2, b2, demand=dem)
    assert isinstance(net2.links[0], Link) and isinstance(od2, ODMatrix)
    x1, x2, c1, c2 = two_link_ue_closed_form(a1, b1, a2, b2, dem)
    s1, s2, so_tstt = two_link_so_closed_form(a1, b1, a2, b2, dem)
    ue2 = user_equilibrium(net2, od2, method="bfw", tol=1e-14, max_iter=200)
    so2 = system_optimum(net2, od2, tol=1e-14, max_iter=200)
    print("  UE closed form  x=(%.9f, %.9f)  costs=(%.9f, %.9f)"
          % (x1, x2, c1, c2))
    print("  UE solver       x=(%.9f, %.9f)  |err|=%.3e"
          % (ue2.flows[0], ue2.flows[1],
             max(abs(ue2.flows[0] - x1), abs(ue2.flows[1] - x2))))
    print("  SO closed form  x=(%.9f, %.9f)  TSTT=%.9f" % (s1, s2, so_tstt))
    print("  SO solver       x=(%.9f, %.9f)  TSTT=%.9f  |err|=%.3e"
          % (so2.flows[0], so2.flows[1], so2.total_travel_time,
             max(abs(so2.flows[0] - s1), abs(so2.flows[1] - s2))))
    poa2 = price_of_anarchy(net2, od2)
    assert isinstance(poa2, PriceOfAnarchyResult)
    print("  price of anarchy = %.9f  (UE %.6f vs SO %.6f veh-h)"
          % (poa2.price_of_anarchy, poa2.ue_total_travel_time,
             poa2.so_total_travel_time))
    tolls2 = marginal_cost_tolls(net2, so2.flows)
    tue2 = tolled_user_equilibrium(net2, od2, tolls2, tol=1e-14)
    print("  marginal-cost tolls = (%.6f, %.6f)" % (tolls2[0], tolls2[1]))
    print("  tolled UE flows     = (%.9f, %.9f)  max|toll-UE - SO| = %.3e"
          % (tue2.flows[0], tue2.flows[1],
             max_flow_change(tue2.flows, so2.flows)))

    # ---------------------------------------------------------- constant costs
    print("\n-- 3. degenerate check: all costs constant => UE == SO, PoA == 1")
    netc = Network("constant")
    netc.ensure_nodes(2)
    netc.add_link(0, 1, Constant(5.0), name="a")
    netc.add_link(0, 1, Constant(9.0), name="b")
    odc = ODMatrix({(0, 1): 12.0})
    poac = price_of_anarchy(netc, odc)
    print("  flows UE=%s SO=%s  PoA=%.12f"
          % (["%.4f" % v for v in poac.ue_flows],
             ["%.4f" % v for v in poac.so_flows], poac.price_of_anarchy))

    # -------------------------------------------------------------- Braess
    print("\n-- 4. Braess's paradox (the flagship oracle)")
    nb, odb = braess_network(demand=6.0, include_bridge=False)
    wb, odw = braess_network(demand=6.0, include_bridge=True)
    ue_nb = user_equilibrium(nb, odb, method="bfw", tol=1e-14, max_iter=400)
    ue_wb = user_equilibrium(wb, odw, method="bfw", tol=1e-14, max_iter=400)
    print("  no bridge : flows=%s" % ["%.6f" % v for v in ue_nb.flows])
    print("              per-traveller cost = %.9f   TSTT = %.6f"
          % (ue_nb.total_travel_time / 6.0, ue_nb.total_travel_time))
    print("  bridge    : flows=%s" % ["%.6f" % v for v in ue_wb.flows])
    print("              per-traveller cost = %.9f   TSTT = %.6f"
          % (ue_wb.total_travel_time / 6.0, ue_wb.total_travel_time))
    print("              opening the bridge COSTS %.6f veh-h (83 -> 92 per trip)"
          % (ue_wb.total_travel_time - ue_nb.total_travel_time))
    br = detect_braess(wb, odw)
    assert isinstance(br, BraessResult)
    print("  detect_braess: paradox=%s  link=%d (%s)  TSTT %.4f -> %.4f"
          % (br.paradox, br.best_link,
             wb.links[br.best_link].label(br.best_link) if br.paradox else "none",
             br.base_total_travel_time, br.best_total_travel_time))
    poab = price_of_anarchy(wb, odw)
    print("  Braess PoA = %.9f  (affine bound 4/3 = %.9f)"
          % (poab.price_of_anarchy, 4.0 / 3.0))

    # ------------------------------------------------- cross-solver agreement
    print("\n-- 5. six solvers, one equilibrium (cross-solver oracle)")
    # a *heterogeneous* grid: every link gets its own free-flow time, its own
    # capacity and -- for three of them -- a different functional form entirely.
    # That is precisely what the single-global-alpha core routine cannot express.
    grid, _ = grid_network(3, 3, free_flow_time=2.0, capacity=40.0)
    rng_net = random.Random(3)
    for i, lk in enumerate(grid.links):
        t0 = round(rng_net.uniform(1.5, 3.5), 3)
        cp = round(rng_net.uniform(25.0, 60.0), 1)
        if i == 5:
            lk.vdf = Conical(t0, cp, 6.0)
        elif i == 11:
            lk.vdf = Davidson(t0, cp, 0.2)
        elif i == 17:
            lk.vdf = Polynomial([t0, 0.4 * t0, 0.0, 0.0, 0.9 * t0], cp)
        else:
            lk.vdf = BPR(t0, cp, 0.15, 4.0)
    grid.links[4].toll = 0.5
    print("  heterogeneous grid: %d links, %d distinct VDF classes, one tolled"
          % (grid.n_links,
             len({type(lk.vdf).__name__ for lk in grid.links})))
    odg = ODMatrix({(0, 8): 60.0, (2, 6): 35.0, (8, 0): 25.0})
    aon = all_or_nothing(grid, odg, grid.free_flow_times(),
                         keep_origin_flows=True)
    assert isinstance(aon, AllOrNothingResult)
    print("  all-or-nothing at free flow: SPTT=%.4f  loaded links=%d"
          % (aon.sptt, sum(1 for v in aon.flows if v > 0)))
    inc = incremental_assignment(grid, odg, n_increments=20)
    print("  " + inc.summary())
    runs: Dict[str, AssignmentResult] = {}
    runs["MSA"] = method_of_successive_averages(grid, odg, max_iter=8000,
                                                tol=1e-9)
    runs["FW"] = frank_wolfe(grid, odg, max_iter=4000, tol=1e-12)
    runs["cFW"] = conjugate_frank_wolfe(grid, odg, max_iter=2000, tol=1e-12)
    runs["bFW"] = biconjugate_frank_wolfe(grid, odg, max_iter=2000, tol=1e-12,
                                          keep_origin_flows=True)
    runs["GP"] = gradient_projection(grid, odg, max_iter=2000, tol=1e-12)
    runs["AlgB"] = algorithm_b(grid, odg, max_iter=2000, tol=1e-12)
    for name in ("MSA", "FW", "cFW", "bFW", "GP", "AlgB"):
        print("  %-5s %s" % (name, runs[name].summary()))
    ref = runs["bFW"].flows
    print("  max |flow - bFW| across solvers:")
    for name in ("MSA", "FW", "cFW", "GP", "AlgB"):
        print("    %-5s %.3e   (Beckmann delta %.3e)"
              % (name, max_flow_change(runs[name].flows, ref),
                 abs(runs[name].objective - runs["bFW"].objective)))
    first = runs["bFW"].history[0]
    assert isinstance(first, IterationRecord)
    print("  first IterationRecord: %s"
          % {k: (round(v, 6) if isinstance(v, float) else v)
             for k, v in sorted(first.as_dict().items())})
    fw_trace = runs["FW"].objective_trace()
    rises = sum(1 for i in range(1, len(fw_trace))
                if fw_trace[i] > fw_trace[i - 1] + 1e-12)
    print("  Frank-Wolfe objective monotone: %s (%d rises over %d iterations,"
          " final gap %.2e)" % (rises == 0, rises, len(fw_trace),
                                runs["FW"].gap_trace()[-1]))
    print("  NOTE: MSA and plain FW have famously slow tails -- their gaps are")
    print("        small but their flows still differ from the exact answer.")
    print("  the same six on Braess (a well-conditioned instance):")
    braess_ref: Optional[Flow] = None
    for name in ("msa", "fw", "cfw", "bfw", "gp", "b"):
        rb = user_equilibrium(wb, odw, method=name, max_iter=20000, tol=1e-12)
        if braess_ref is None:
            braess_ref = rb.flows
        print("    %-4s iters=%-6d gap=%.2e   max|flow - MSA| = %.2e"
              % (name, rb.iterations, rb.relative_gap,
                 max_flow_change(rb.flows, braess_ref)))
    print("  beckmann_objective=%.6f  total_system_travel_time=%.6f"
          % (beckmann_objective(grid, ref), total_system_travel_time(grid, ref)))
    grad = beckmann_gradient(grid, ref)
    print("  beckmann_gradient == link costs: %s"
          % (max(abs(grad[e] - runs["bFW"].costs[e])
                 for e in range(grid.n_links)) < 1e-15))
    print("  relative_gap=%.3e  average_excess_cost=%.3e"
          % (relative_gap(grid, ref, odg), average_excess_cost(grid, ref, odg)))
    direction = [aon.flows[e] - ref[e] for e in range(grid.n_links)]
    lam_b = exact_line_search(grid, ref, direction, method="bisect")
    lam_g = exact_line_search(grid, ref, direction, method="golden")
    print("  exact_line_search from the equilibrium: bisect=%.3e golden=%.3e"
          " (should both be ~0)" % (lam_b, lam_g))

    # ----------------------------------------------------- paths and analysis
    print("\n-- 6. paths, decomposition, Wardrop and select-link")
    costs_g = grid.costs(ref)
    tree = shortest_path_tree(grid, 0, costs_g)
    assert isinstance(tree, ShortestPathTree)
    cost08, nodes08, links08 = shortest_path(grid, 0, 8, costs_g)
    print("  shortest path 0->8: cost=%.6f nodes=%s" % (cost08, nodes08))
    print("  tree distance to 8 agrees: %s" % (abs(tree.dist[8] - cost08) < 1e-12))
    ksp = k_shortest_paths(grid, 0, 8, costs_g, 4)
    print("  Yen k-shortest 0->8 costs: %s"
          % ["%.6f" % c for (c, _p) in ksp])
    allp = enumerate_paths(grid, 0, 8, max_paths=40, max_links=6, costs=costs_g)
    print("  enumerate_paths 0->8 (<=6 links): %d routes, cheapest cost %.6f"
          % (len(allp), path_cost(grid, allp[0], costs_g)))
    ps = decompose_flows(grid, odg, ref, origin_flows=runs["bFW"].origin_flows,
                         costs=costs_g)
    assert isinstance(ps, PathSet) and isinstance(ps.all_paths()[0], Path)
    rebuilt = ps.link_flows(grid)
    print("  decompose_flows: %d routes, max|rebuilt - flows| = %.3e"
          % (ps.n_paths(), max_flow_change(rebuilt, ref)))
    ps_agg = decompose_flows(grid, odg, ref, costs=costs_g)
    print("  decompose_flows without origin flows (aggregate peel): %d routes, "
          "max|rebuilt - flows| = %.3e"
          % (ps_agg.n_paths(),
             max_flow_change(ps_agg.link_flows(grid), ref)))
    wr = wardrop_check(grid, ps, costs_g, tol=1e-6)
    assert isinstance(wr, WardropReport)
    print("  Wardrop: satisfied=%s  max used-route spread=%.3e  "
          "cheapest-unused violation=%.3e"
          % (wr.satisfied, wr.max_used_gap, wr.max_unused_violation))
    busiest = max(range(grid.n_links), key=lambda e: ref[e])
    sl = select_link_analysis(grid, ps, busiest)
    assert isinstance(sl, SelectLinkResult)
    print("  select-link %d: %.4f veh from %d OD pairs, top=%s"
          % (busiest, sl.total_flow, len(sl.od_flows),
             ["(%d,%d)=%.3f" % (o, d, v) for (o, d), v in sl.top_pairs(2)]))
    sz = select_zone_analysis(grid, ps, 0, role="origin")
    print("  select-zone 0 (origin): %.4f veh over %d OD pairs"
          % (sz.total_flow, len(sz.od_flows)))

    # -------------------------------------------------------- stochastic SUE
    print("\n-- 7. stochastic user equilibrium")
    st = dial_stoch(grid, odg, grid.free_flow_times(), theta=0.4)
    print("  Dial STOCH loading at free flow: total veh-links=%.4f  max link=%.4f"
          % (math.fsum(st), max(st)))
    sue_d = logit_sue(grid, odg, theta=0.4, method="dial", max_iter=120, tol=1e-8)
    sue_p = logit_sue(grid, odg, theta=0.4, method="path", overlap="none",
                      n_paths=5, max_iter=120, tol=1e-8)
    sue_c = logit_sue(grid, odg, theta=0.4, method="path", overlap="clogit",
                      n_paths=5, max_iter=120, tol=1e-8)
    sue_s = logit_sue(grid, odg, theta=0.4, method="path", overlap="pathsize",
                      n_paths=5, max_iter=120, tol=1e-8)
    for tag, r in (("dial", sue_d), ("path", sue_p), ("c-logit", sue_c),
                   ("path-size", sue_s)):
        print("  %-10s TSTT=%.4f  sue gap=%.2e  iters=%d"
              % (tag, r.total_travel_time, r.extra["sue_gap"], r.iterations))
    print("  Fisk objective (path/logit) = %.6f"
          % sue_p.extra["fisk_objective"])
    bucket = sue_p.paths.paths(0, 8) if sue_p.paths is not None else []
    pc = [p.cost for p in bucket]
    probs = logit_probabilities(pc, 0.4)
    cf = c_logit_factors(grid, bucket)
    psf = path_size_factors(grid, bucket)
    print("  route choice 0->8: costs=%s" % ["%.3f" % v for v in pc])
    print("                     logit p=%s" % ["%.4f" % v for v in probs])
    print("                     c-logit CF=%s" % ["%.4f" % v for v in cf])
    print("                     path size=%s" % ["%.4f" % v for v in psf])
    pr = probit_sue(grid, odg, sigma=0.25, n_draws=16, max_iter=120, tol=1e-6,
                    seed=7)
    pr2 = probit_sue(grid, odg, sigma=0.25, n_draws=16, max_iter=120, tol=1e-6,
                     seed=7)
    print("  probit SUE: TSTT=%.6f  gap=%.2e  reproducible=%s"
          % (pr.total_travel_time, pr.extra["sue_gap"],
             max_flow_change(pr.flows, pr2.flows) == 0.0))
    print("              (the gap floor is one route switch, demand/n_draws"
          " = %.2f veh)" % (odg.total() / pr.extra["n_draws"]))

    # ------------------------------------------------------------ multi-class
    print("\n-- 8. multi-class assignment in PCU")
    classes = [
        VehicleClass("car", {(0, 8): 40.0, (2, 6): 20.0}, pcu=1.0,
                     value_of_time=12.0),
        VehicleClass("hgv", {(0, 8): 8.0, (2, 6): 4.0}, pcu=2.5,
                     value_of_time=30.0),
        VehicleClass("bus", {(0, 8): 3.0}, pcu=2.0, value_of_time=20.0),
    ]
    mcr = multiclass_assignment(grid, classes, max_iter=3000, tol=1e-7)
    assert isinstance(mcr, MultiClassResult)
    print("  " + mcr.summary())
    for name in mcr.class_names():
        print("    %-4s veh on network = %8.4f   per-class gap = %.2e"
              % (name, math.fsum(mcr.class_flows[name]), mcr.class_gaps[name]))
    print("    PCU on the busiest link = %.4f (cars %.4f, hgv %.4f, bus %.4f)"
          % (max(mcr.pcu_flows),
             mcr.class_flows["car"][mcr.pcu_flows.index(max(mcr.pcu_flows))],
             mcr.class_flows["hgv"][mcr.pcu_flows.index(max(mcr.pcu_flows))],
             mcr.class_flows["bus"][mcr.pcu_flows.index(max(mcr.pcu_flows))]))

    # --------------------------------------------------------- elastic demand
    print("\n-- 9. elastic demand (excess-demand reduction vs Evans)")
    dfuncs: Dict[Tuple[int, int], DemandFunction] = {
        (0, 8): ExponentialDemand(120.0, 0.02),
        (2, 6): LinearDemand(60.0, 0.8),
        (8, 0): PowerDemand(40.0, 20.0, 2.0),
    }
    ed = elastic_demand_assignment(grid, dfuncs, method="excess", solver="gp",
                                   tol=1e-6, max_iter=4000)
    ev = elastic_demand_assignment(grid, dfuncs, method="evans", max_iter=300,
                                   tol=1e-6)
    assert isinstance(ed, ElasticDemandResult)
    print("  " + ed.summary())
    print("  " + ev.summary())
    print("  two formulations agree to %.3e veh on link flows, %.3e on trips"
          % (max_flow_change(ed.flows, ev.flows),
             max(abs(ed.demand[k] - ev.demand[k]) for k in sorted(dfuncs))))
    for key in sorted(dfuncs):
        want = dfuncs[key].demand(ed.od_costs[key])
        print("    OD %s: cost=%8.4f  trips=%9.4f  D(cost)=%9.4f  |err|=%.2e"
              % (str(key), ed.od_costs[key], ed.demand[key], want,
                 abs(want - ed.demand[key])))

    # ---------------------------------------------------------- turn penalties
    print("\n-- 10. turn penalties on the expanded graph")
    tn = Network("turns")
    tn.ensure_nodes(4)
    e_a = tn.add_link(0, 1, BPR(5.0, 50.0), name="a")
    e_b = tn.add_link(1, 2, BPR(5.0, 50.0), name="b")
    e_c = tn.add_link(1, 3, BPR(5.0, 50.0), name="c")
    e_d = tn.add_link(3, 2, BPR(5.0, 50.0), name="d")
    od_t = ODMatrix({(0, 2): 40.0})
    free = turn_penalty_assignment(tn, od_t, {}, tol=1e-12)
    pen = turn_penalty_assignment(tn, od_t, {(e_a, e_b): 20.0}, tol=1e-12)
    ban = turn_penalty_assignment(tn, od_t, {}, banned=[(e_a, e_b)], tol=1e-12)
    exp = expand_with_turn_penalties(tn, {(e_a, e_b): 20.0})
    assert isinstance(exp, TurnExpansion)
    assert isinstance(free.extra["expansion"], TurnExpansion)
    print("  expanded graph: %d nodes, %d links (original %d/%d), %d turns"
          % (exp.network.n_nodes, exp.network.n_links, tn.n_nodes, tn.n_links,
             len(exp.turn_link)))
    print("  expanded OD %s -> %s; link %d carries original link %d's flow"
          % (od_t.pairs(), exp.expand_od(od_t).pairs(), exp.link_of[e_c], e_c))
    for tag, r in (("no penalty", free), ("+20 on a->b", pen),
                   ("a->b banned", ban)):
        print("    %-12s flows a,b,c,d = %s  TSTT=%.4f"
              % (tag, ["%.4f" % v for v in r.flows], r.total_travel_time))
    print("    turn volumes with the penalty: %s"
          % {"link%d->link%d" % k: round(v, 4)
             for k, v in sorted(pen.extra["turn_volumes"].items())})

    # ------------------------------------------------------------ sensitivity
    print("\n-- 11. sensitivity of the equilibrium to capacity")
    sens = capacity_sensitivity(grid, odg, links=[0, 1, 2, 3], delta=0.02,
                                max_iter=400, tol=1e-12)
    assert isinstance(sens, SensitivityResult)
    print("  base TSTT=%.6f  best link to widen = %d" % (sens.base_total_travel_time,
                                                         sens.best_link()))
    for i, e in enumerate(sens.links):
        print("    link %d: dTSTT/dcap = %+.6f   largest dflow/dcap = %+.6f"
              % (e, sens.dtstt_dcapacity[i],
                 max(sens.dflow_dcapacity[i], key=abs)))
    sensb = capacity_sensitivity(wb, odw, links=[2], delta=0.05, max_iter=400,
                                 tol=1e-12)
    print("  Braess bridge dTSTT/dcap = %+.6f (positive => widening it HURTS)"
          % sensb.dtstt_dcapacity[0])

    # -------------------------------------------- price-of-anarchy fuzz (4/3)
    print("\n-- 12. price of anarchy on seeded random affine networks (<= 4/3)")
    worst = 0.0
    for seed in range(8):
        rnet, rod = random_affine_network(7, 16, seed=seed, n_od=3,
                                          demand_scale=12.0)
        rp = price_of_anarchy(rnet, rod, max_iter=600, tol=1e-12)
        worst = max(worst, rp.price_of_anarchy)
        print("    seed %d: PoA=%.9f  UE=%.4f SO=%.4f  ok=%s"
              % (seed, rp.price_of_anarchy, rp.ue_total_travel_time,
                 rp.so_total_travel_time,
                 rp.price_of_anarchy <= 4.0 / 3.0 + 1e-9))
    print("  worst PoA over 8 seeds = %.9f  (bound 4/3 = %.9f)"
          % (worst, 4.0 / 3.0))

    # ---------------------------------------------------------------- payloads
    print("\n-- 13. payloads")
    payload = build_traffic_assignment_payload(grid, runs["bFW"], od=odg,
                                               pathset=ps, top_links=5)
    ui = build_a2ui_payload(grid, runs["bFW"], od=odg, comparison=runs,
                            poa=poab, braess=br)
    print("  assignment payload keys: %s" % sorted(payload)[:8])
    print("  payload links=%d convergence rows=%d paths=%d"
          % (len(payload["links"]), len(payload["convergence"]),
             len(payload["paths"])))
    print("  a2ui cards: %s" % [c["label"] for c in ui["cards"]])
    print("  a2ui agreement vs bFW: %s"
          % {k: "%.1e" % v for k, v in sorted(ui["agreement"].items())})

    # ------------------------------------------------------------- validation
    print("\n-- 14. validation guards")
    for label, fn in (
            ("negative demand", lambda: ODMatrix({(0, 1): -5.0})),
            ("unreachable OD", lambda: user_equilibrium(
                two_link_network(1.0, 1.0, 1.0, 1.0)[0], {(1, 0): 5.0})),
            ("bad VDF alpha", lambda: Conical(10.0, 100.0, 0.5)),
            ("negative polynomial", lambda: Polynomial([1.0, -2.0], 10.0)),
            ("wrong cost length", lambda: all_or_nothing(
                net2, od2, [1.0, 2.0, 3.0]))):
        try:
            fn()
        except TrafficAssignmentError as exc:
            print("  %-20s -> TrafficAssignmentError: %s"
                  % (label, str(exc)[:58]))

    print("\n" + "=" * 70)
    print("  demo OK")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
