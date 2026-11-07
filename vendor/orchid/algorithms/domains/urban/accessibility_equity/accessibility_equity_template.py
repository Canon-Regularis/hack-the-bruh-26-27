"""
Accessibility & Equity -- who can reach what, and whether that is fair
=====================================================================

**Accessibility is the point of a transport system; equity is the point of a
city.**  This module takes the two things the rest of the urban stack produces
-- a travel-cost matrix and a distribution of people and opportunities -- and
answers the two questions a plan is actually judged on: *how much of the city
can each place reach?* and *is that reach distributed fairly across the people
who live there?*

Where it sits among its siblings:

* ``street_network`` builds the graph and computes the travel times.  It is a
  supply-side, purely descriptive layer: metres, minutes, betweenness.
* ``travel_demand`` predicts the trips people will make, given those times.
  It is behavioural and predictive: trips per hour, by mode, by purpose.
* This module *consumes* both and asks a NORMATIVE question.  Nothing here
  predicts a single trip.  Cumulative opportunity, gravity potential and the
  floating-catchment family all measure the *opportunity set* a location
  commands whether or not anybody uses it; the inequality half then takes that
  surface, lays it against population, need and deprivation, and reports who
  wins.  A Gini coefficient is not a forecast -- it is a verdict.

Everything is pure standard-library Python.  The Lorenz curve, the Theil
decomposition, the Atkinson index, the normal CDF, the floating-catchment
solvers and the Moran's-I permutation test are all built by hand.

Contents
--------
1.  **Impedance / distance decay** -- ``power``, ``exponential``, ``gaussian``,
    ``log_logistic``, ``rectangular`` and ``cumulative_gaussian`` kernels behind
    one ``DecayFunction`` object, with curves, analytic half-costs and
    ``calibrate_decay`` (bisection onto an observed mean trip cost).
2.  **Travel-cost matrices** -- ``CostMatrix`` plus builders from coordinates
    (great-circle or planar), from a weighted graph (Dijkstra) and from raw
    times, with scaling, transposition and intrazonal costs.
3.  **Cumulative opportunity** -- counts at one or many thresholds, isochrone
    bands, and the *dual* measure: the cost of reaching the N-th opportunity
    (``time_to_nth_opportunity`` / ``place_rank_accessibility``).
4.  **Gravity / potential** -- Hansen potential, generic gravity with any
    kernel, and Shen's competition-adjusted accessibility.
5.  **Floating catchments** -- ``two_step_fca``, ``enhanced_2sfca`` (kernel
    zones), ``three_step_fca`` (selection-probability competition), ``m2sfca``
    (Delamater's squared-impedance variant) and variable catchments that grow
    until they capture enough population / supply.
6.  **Utility-based accessibility** -- destination-choice logit, the logsum
    (the only accessibility measure with a welfare interpretation) and the
    consumer-surplus benefit of a scheme.
7.  **Time of day & reliability** -- period-weighted accessibility, travel-time
    percentiles, buffer / planning time indices and reliability-adjusted
    accessibility.
8.  **Inequality** -- Lorenz curve and area, Gini (direct *and* from the
    curve), Theil T and L with an exact between/within decomposition,
    generalised entropy, Atkinson, Palma, 20:20, CV, concentration curve and
    index, Kolm (absolute) and Hoover, plus a Pigou-Dalton transfer helper.
9.  **Horizontal vs vertical equity** -- the within/between split read as
    "like treated alike" versus "unlike treated differently", and a signed
    progressivity index against need.
10. **Need-weighted accessibility** -- population-, need- and
    deprivation-weighted means and between-group gaps.
11. **Transport poverty** -- accessibility poverty lines, the
    Foster-Greer-Thorbecke family and a composite deprivation index.
12. **Spatial autocorrelation** -- kNN / distance-band / inverse-distance
    weights, global and local Moran's I, Geary's C and a permutation test
    that can be run EXHAUSTIVELY for small n.
13. **Impact incidence** -- who gains and who loses between two designs, by
    group, with the change in inequality.
14. **Rawlsian priority** -- max-min, leximin comparison and ranking.
15. **Disruption** -- scenario accessibility, loss, one-at-a-time
    vulnerability ranking and a resilience index.
16. **Payloads** -- JSON-able dicts for a frontend.

Conventions
-----------
* A *cost* is a generalised travel cost in consistent units (minutes by default
  in the docstrings; the code never assumes a unit).  An infinite cost means
  unreachable and is legal everywhere a cost is accepted.
* ``cost[i][j]`` runs origins (people) down the rows and destinations
  (opportunities) across the columns.  Supply-side vectors are indexed by ``j``,
  demand-side vectors by ``i``.
* Any routine producing one number per origin returns a plain ``list`` of
  floats in origin order; anything richer returns a ``@dataclass``.
* All sums go through ``math.fsum`` and every tie is broken by index or by
  label, so results are byte-identical across runs and independent of
  ``PYTHONHASHSEED``.  Group ordering is *first appearance*, never set order.
* Relative inequality indices (Gini, Theil, Atkinson, Palma, 20:20, CV, Hoover,
  concentration) are SCALE-invariant: multiply every accessibility by 1e6 and
  they do not move.  The Kolm index is the odd one out on purpose -- it is the
  *absolute* (leftist) index and is TRANSLATION-invariant instead.
* British spelling in prose; the literature's names for the measures.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import (Any, Dict, List, Optional, Sequence, Tuple)

__version__ = "1.0.0"

Vector = List[float]
Matrix = List[List[float]]
Coord = Tuple[float, float]
Group = Any

_EPS = 1e-12
_INF = float("inf")
_SQRT2 = math.sqrt(2.0)
_EARTH_RADIUS_M = 6371008.8

# ---------------------------------------------------------------------------
# Optional core interop.  The module is fully standalone; when the orchid core
# graph toolkit happens to be importable the demo cross-checks its own
# great-circle distance and its own Dijkstra against the core implementations.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only when the core package is on the path
    from algorithms.core.graph.geospatial_graphs.geospatial_graphs_template import (
        GeoIndex as _CoreGeoIndex,
        haversine as _core_haversine,
    )
    _HAVE_CORE_GEO = True
except ImportError:                            # pragma: no cover
    _CoreGeoIndex = None
    _core_haversine = None
    _HAVE_CORE_GEO = False

try:  # pragma: no cover - exercised only when the core package is on the path
    from algorithms.core.graph.shortest_paths.shortest_paths_template import (
        PathGraph as _CorePathGraph,
        shortest_path as _core_shortest_path,
    )
    _HAVE_CORE_PATHS = True
except ImportError:                            # pragma: no cover
    _CorePathGraph = None
    _core_shortest_path = None
    _HAVE_CORE_PATHS = False


__all__ = [
    # errors
    "AccessibilityError",
    # impedance / distance decay
    "DecayFunction", "DECAY_NAMES", "get_decay", "decay_power",
    "decay_exponential", "decay_gaussian", "decay_log_logistic",
    "decay_rectangular", "decay_cumulative_gaussian", "decay_curve",
    "decay_half_cost", "mean_decay_cost", "calibrate_decay", "normal_cdf",
    # travel-cost matrices
    "CostMatrix", "haversine_m", "euclidean_m", "travel_time",
    "cost_matrix_from_coords", "cost_matrix_from_graph", "intrazonal_cost",
    # cumulative opportunity
    "CumulativeResult", "cumulative_opportunity",
    "cumulative_opportunity_curve", "isochrone_bands",
    "time_to_nth_opportunity", "PlaceRankResult", "place_rank_accessibility",
    # gravity / potential
    "AccessibilityResult", "gravity_accessibility", "hansen_accessibility",
    "potential_accessibility", "shen_accessibility", "normalise_accessibility",
    # floating catchments
    "FCAResult", "supply_demand_ratio", "two_step_fca", "e2sfca_zone_weights",
    "enhanced_2sfca", "three_step_fca", "m2sfca", "variable_catchment_2sfca",
    # utility-based accessibility
    "utility_matrix", "destination_choice_probabilities",
    "logsum_accessibility", "logsum_benefit",
    # time of day & reliability
    "TimeOfDayResult", "time_of_day_accessibility", "travel_time_percentile",
    "buffer_time_index", "planning_time_index", "reliability_adjusted_cost",
    "reliability_adjusted_accessibility",
    # inequality
    "LorenzCurve", "lorenz_curve", "lorenz_area", "gini", "gini_from_lorenz",
    "theil_t", "theil_l", "generalised_entropy", "TheilDecomposition",
    "theil_decomposition", "atkinson", "palma_ratio", "quintile_ratio",
    "coefficient_of_variation", "concentration_curve", "concentration_index",
    "kolm_index", "hoover_index", "InequalityProfile", "inequality_profile",
    "pigou_dalton_transfer",
    # horizontal vs vertical equity
    "EquityDecomposition", "horizontal_vertical_equity", "horizontal_equity",
    "vertical_equity",
    # need-weighted accessibility
    "population_weighted_mean", "equity_weights",
    "need_weighted_accessibility", "deprivation_weighted_accessibility",
    "GroupGapResult", "accessibility_gap",
    # transport poverty
    "accessibility_poverty_line", "fgt_index", "TransportPovertyResult",
    "transport_poverty_indicators", "transport_poverty_index",
    # spatial autocorrelation
    "spatial_weights_knn", "spatial_weights_distance_band",
    "spatial_weights_inverse_distance", "row_standardise", "MoranResult",
    "morans_i", "morans_i_expectation", "morans_i_permutation_test",
    "local_morans_i", "gearys_c",
    # impact incidence
    "IncidenceResult", "impact_incidence", "winners_losers",
    "distributional_summary",
    # rawlsian priority
    "rawlsian_index", "leximin_key", "leximin_compare", "leximin_sort",
    "MaximinResult", "rawlsian_maxmin",
    # disruption
    "DisruptionResult", "disruption_scenario", "accessibility_loss",
    "vulnerability_ranking", "resilience_index",
    # payloads
    "build_accessibility_equity_payload", "build_a2ui_payload",
]


# ===========================================================================
# 0. VALIDATION HELPERS
# ===========================================================================
class AccessibilityError(ValueError):
    """Invalid input to an accessibility or equity routine."""


def _as_float(x: Any, name: str = "value") -> float:
    """Coerce to a finite float or raise."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise AccessibilityError("%s must be a real number (got %r)." % (name, x))
    if math.isnan(v) or math.isinf(v):
        raise AccessibilityError("%s must be finite (got %r)." % (name, x))
    return v


def _as_cost(x: Any, name: str = "cost") -> float:
    """Coerce to a non-negative cost; an infinite (unreachable) cost is allowed."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise AccessibilityError("%s must be a real number (got %r)." % (name, x))
    if math.isnan(v):
        raise AccessibilityError("%s must not be NaN." % name)
    if v < 0.0:
        raise AccessibilityError("%s must be >= 0 (got %.6g)." % (name, v))
    return v


def _as_floats(xs: Any, name: str = "values") -> Vector:
    """Coerce a sequence to a list of finite floats or raise."""
    if isinstance(xs, (str, bytes)) or not hasattr(xs, "__iter__"):
        raise AccessibilityError("%s must be a sequence of real numbers." % name)
    try:
        out = [float(v) for v in xs]
    except (TypeError, ValueError):
        raise AccessibilityError("%s must be a sequence of real numbers." % name)
    for v in out:
        if math.isnan(v) or math.isinf(v):
            raise AccessibilityError("%s contains non-finite values." % name)
    return out


def _positive(x: Any, name: str = "value") -> float:
    """Coerce to a strictly positive finite float or raise."""
    v = _as_float(x, name)
    if v <= 0.0:
        raise AccessibilityError("%s must be > 0 (got %.6g)." % (name, v))
    return v


def _non_negative(x: Any, name: str = "value") -> float:
    """Coerce to a non-negative finite float or raise."""
    v = _as_float(x, name)
    if v < 0.0:
        raise AccessibilityError("%s must be >= 0 (got %.6g)." % (name, v))
    return v


def _in_unit(x: Any, name: str = "value") -> float:
    """Coerce to a float in [0, 1] or raise."""
    v = _as_float(x, name)
    if v < 0.0 or v > 1.0:
        raise AccessibilityError("%s must lie in [0, 1] (got %.6g)." % (name, v))
    return v


def _positive_int(x: Any, name: str = "n") -> int:
    """Coerce to a strictly positive int or raise."""
    try:
        v = int(x)
    except (TypeError, ValueError):
        raise AccessibilityError("%s must be an integer (got %r)." % (name, x))
    if v <= 0:
        raise AccessibilityError("%s must be > 0 (got %d)." % (name, v))
    return v


def _non_negative_vector(xs: Any, name: str = "values") -> Vector:
    """Coerce to a list of finite, non-negative floats or raise."""
    out = _as_floats(xs, name)
    for v in out:
        if v < 0.0:
            raise AccessibilityError("%s must be >= 0 everywhere (got %.6g)."
                                     % (name, v))
    return out


def _non_empty(xs: Sequence[Any], name: str = "values") -> None:
    """Raise if the sequence is empty."""
    if len(xs) == 0:
        raise AccessibilityError("%s must not be empty." % name)


def _as_cost_matrix(cost: Any, name: str = "cost") -> Matrix:
    """Coerce to a rectangular matrix of non-negative costs (infinities allowed)."""
    if isinstance(cost, CostMatrix):
        return [row[:] for row in cost.cost]
    if isinstance(cost, (str, bytes)) or not hasattr(cost, "__iter__"):
        raise AccessibilityError("%s must be a matrix of costs." % name)
    rows = list(cost)
    if not rows:
        raise AccessibilityError("%s must have at least one origin." % name)
    out: Matrix = []
    width = None
    for i, row in enumerate(rows):
        if isinstance(row, (str, bytes)) or not hasattr(row, "__iter__"):
            raise AccessibilityError("%s row %d must be a sequence." % (name, i))
        vals = [_as_cost(v, "%s[%d]" % (name, i)) for v in row]
        if width is None:
            width = len(vals)
            if width == 0:
                raise AccessibilityError(
                    "%s must have at least one destination." % name)
        elif len(vals) != width:
            raise AccessibilityError(
                "%s is ragged: row 0 has %d entries, row %d has %d."
                % (name, width, i, len(vals)))
        out.append(vals)
    return out


def _check_length(xs: Sequence[Any], n: int, name: str, what: str) -> None:
    """Raise unless the sequence has exactly ``n`` entries."""
    if len(xs) != n:
        raise AccessibilityError("%s must have one entry per %s (%d != %d)."
                                 % (name, what, len(xs), n))


def _default_weights(weights: Optional[Sequence[Any]], n: int,
                     name: str = "weights") -> Vector:
    """Validated non-negative weights, or ``n`` ones when None."""
    if weights is None:
        return [1.0] * n
    w = _non_negative_vector(weights, name)
    _check_length(w, n, name, "value")
    total = math.fsum(w)
    if total <= 0.0:
        raise AccessibilityError("%s must not sum to zero." % name)
    return w


def _group_order(groups: Sequence[Group], name: str = "groups") -> List[Group]:
    """Unique group labels in FIRST-APPEARANCE order (never set order)."""
    seen: Dict[Group, int] = {}
    order: List[Group] = []
    for g in groups:
        try:
            hash(g)
        except TypeError:
            raise AccessibilityError("%s entries must be hashable (got %r)."
                                     % (name, g))
        if g not in seen:
            seen[g] = len(order)
            order.append(g)
    return order


def _mean(values: Sequence[float], weights: Optional[Sequence[float]] = None
          ) -> float:
    """Arithmetic (optionally weighted) mean, summed by ``math.fsum``."""
    if weights is None:
        return math.fsum(values) / float(len(values))
    total = math.fsum(weights)
    return math.fsum(w * v for w, v in zip(weights, values)) / total


def _sorted_with_index(values: Sequence[float]) -> List[Tuple[float, int]]:
    """``(value, index)`` pairs ascending -- the canonical deterministic order."""
    return sorted(((float(v), i) for i, v in enumerate(values)),
                  key=lambda p: (p[0], p[1]))


def _scaled_moments(values: Sequence[float],
                    weights: Optional[Sequence[float]] = None
                    ) -> Tuple[float, float, float]:
    """Weighted mean and population sd, computed in MAX-SCALED units.

    Returns ``(mean, sd, scale)`` where ``scale`` is the largest magnitude in
    the sample and ``mean``/``sd`` are expressed in units of it, so the true
    moments are ``mean * scale`` and ``sd * scale``.

    Squaring a deviation is the one operation in this module that can leave
    double range on data that is otherwise perfectly representable: a surface
    around 1e-170 has ``(x - mu) ** 2`` underflow to exactly 0.0 (reporting
    perfect equality for a surface that plainly varies) and a surface around
    1e+200 has it overflow and leak a raw ``OverflowError``.  Dividing by the
    peak first puts every deviation in [-2, 2], so the sum of squares stays in
    range at every scale.  Dispersion RATIOS (sd/mean, z-scores, Moran's I)
    are then exact at any scale because both parts carry the same ``scale``.
    """
    n = len(values)
    w = [1.0] * n if weights is None else list(weights)
    scale = max(abs(float(x)) for x in values)
    if scale == 0.0 or not math.isfinite(scale):
        scale = 1.0
    total_w = math.fsum(w)
    mu = math.fsum(w[i] * (values[i] / scale) for i in range(n)) / total_w
    var = math.fsum(w[i] * (values[i] / scale - mu) ** 2
                    for i in range(n)) / total_w
    return mu, math.sqrt(var), scale


def _json_safe(x: Any) -> Any:
    """Replace non-finite floats with None so the dict survives json.dumps."""
    if isinstance(x, bool):
        return x
    if isinstance(x, float):
        if math.isinf(x) or math.isnan(x):
            return None
        return round(x, 8)
    return x


# ===========================================================================
# 1. IMPEDANCE -- DISTANCE-DECAY KERNELS
# ===========================================================================
DECAY_NAMES = ("power", "exponential", "gaussian", "log_logistic",
               "rectangular", "cumulative_gaussian")

_DECAY_DEFAULTS: Dict[str, Dict[str, float]] = {
    "power": {"beta": 1.5, "cost_min": 1.0},
    "exponential": {"beta": 0.05},
    "gaussian": {"beta": 20.0},
    "log_logistic": {"alpha": 20.0, "beta": 3.0},
    "rectangular": {"threshold": 30.0},
    "cumulative_gaussian": {"mu": 30.0, "sigma": 10.0},
}

# Which parameter ``calibrate_decay`` moves, and whether raising it raises the
# implied mean trip cost (+1) or lowers it (-1).
_DECAY_SCALE_PARAM: Dict[str, Tuple[str, int]] = {
    "power": ("beta", -1),
    "exponential": ("beta", -1),
    "gaussian": ("beta", +1),
    "log_logistic": ("alpha", +1),
    "cumulative_gaussian": ("mu", +1),
}


def normal_cdf(z: float) -> float:
    """Standard normal CDF via ``math.erfc`` (full double precision)."""
    return 0.5 * math.erfc(-_as_float(z, "z") / _SQRT2)


def decay_power(cost: float, beta: float, cost_min: float = 1.0) -> float:
    """Inverse-power decay ``(max(c, c_min) / c_min) ** -beta``; 1 at ``c_min``."""
    c = _as_cost(cost)
    b = _non_negative(beta, "beta")
    cm = _positive(cost_min, "cost_min")
    if c == _INF:
        return 0.0
    if c <= cm:
        return 1.0
    return (c / cm) ** (-b)


def decay_exponential(cost: float, beta: float) -> float:
    """Negative-exponential decay ``exp(-beta c)`` -- the entropy-model kernel."""
    c = _as_cost(cost)
    b = _non_negative(beta, "beta")
    if c == _INF:
        return 0.0 if b > 0.0 else 1.0
    return math.exp(-b * c)


def decay_gaussian(cost: float, beta: float) -> float:
    """Gaussian decay ``exp(-c^2 / (2 beta^2))``; ``beta`` is a cost-unit width."""
    c = _as_cost(cost)
    b = _positive(beta, "beta")
    if c == _INF:
        return 0.0
    z = c / b
    return math.exp(-0.5 * z * z)


def decay_log_logistic(cost: float, alpha: float, beta: float) -> float:
    """Log-logistic decay ``1 / (1 + (c / alpha) ** beta)``; half weight at alpha."""
    a = _positive(alpha, "alpha")
    b = _positive(beta, "beta")
    c = _as_cost(cost)
    if c == _INF:
        return 0.0
    if c == 0.0:
        return 1.0
    try:
        odds = (c / a) ** b
    except OverflowError:
        # ``(c/a) ** b`` is the only ``**`` in the kernel zoo whose result can
        # leave double range (c > alpha with a large beta, or a tiny alpha).
        # The denominator 1 + odds then diverges, so the mathematical limit of
        # the kernel is exactly 0 -- return it rather than leaking a raw
        # OverflowError past the AccessibilityError contract.
        return 0.0
    return 1.0 / (1.0 + odds)


def decay_rectangular(cost: float, threshold: float) -> float:
    """Hard cut-off: 1 within the threshold, 0 beyond it (cumulative opportunity)."""
    c = _as_cost(cost)
    t = _non_negative(threshold, "threshold")
    return 1.0 if c <= t else 0.0


def decay_cumulative_gaussian(cost: float, mu: float, sigma: float) -> float:
    """Soft cut-off ``1 - Phi((c - mu) / sigma)``: a threshold with fuzzy edges."""
    m = _non_negative(mu, "mu")
    s = _positive(sigma, "sigma")
    c = _as_cost(cost)
    if c == _INF:
        return 0.0
    return 0.5 * math.erfc((c - m) / (s * _SQRT2))


@dataclass
class DecayFunction:
    """A named, parameterised impedance kernel: cost in, weight in [0, 1] out."""

    name: str
    params: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in _DECAY_DEFAULTS:
            raise AccessibilityError("unknown decay %r (choose from %s)."
                                     % (self.name, ", ".join(DECAY_NAMES)))
        merged = dict(_DECAY_DEFAULTS[self.name])
        for key in sorted(self.params):
            if key not in merged:
                raise AccessibilityError(
                    "decay %r takes no parameter %r (only %s)."
                    % (self.name, key, ", ".join(sorted(merged))))
            merged[key] = _as_float(self.params[key], key)
        self.params = merged
        self(0.0)                               # validate the parameter values

    def __call__(self, cost: float) -> float:
        """Evaluate the kernel at one cost."""
        p = self.params
        if self.name == "power":
            return decay_power(cost, p["beta"], p["cost_min"])
        if self.name == "exponential":
            return decay_exponential(cost, p["beta"])
        if self.name == "gaussian":
            return decay_gaussian(cost, p["beta"])
        if self.name == "log_logistic":
            return decay_log_logistic(cost, p["alpha"], p["beta"])
        if self.name == "rectangular":
            return decay_rectangular(cost, p["threshold"])
        return decay_cumulative_gaussian(cost, p["mu"], p["sigma"])

    def weights(self, costs: Sequence[float]) -> Vector:
        """The kernel applied to a sequence of costs."""
        return [self(c) for c in costs]

    def matrix(self, cost: Any) -> Matrix:
        """The kernel applied to a whole cost matrix."""
        return [[self(c) for c in row] for row in _as_cost_matrix(cost)]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able description of the kernel."""
        return {"name": self.name,
                "params": {k: _json_safe(self.params[k])
                           for k in sorted(self.params)}}


def get_decay(name: str, **params: float) -> DecayFunction:
    """Build a DecayFunction by name, filling in the published defaults."""
    if not isinstance(name, str):
        raise AccessibilityError("decay name must be a string (got %r)." % (name,))
    return DecayFunction(name, dict(params))


def _as_decay(decay: Any, default: Optional[DecayFunction] = None
              ) -> DecayFunction:
    """Accept a DecayFunction, a kernel name, or None -> the supplied default."""
    if decay is None:
        if default is None:
            raise AccessibilityError("a decay function is required.")
        return default
    if isinstance(decay, DecayFunction):
        return decay
    if isinstance(decay, str):
        return get_decay(decay)
    raise AccessibilityError("decay must be a DecayFunction or a name (got %r)."
                             % (decay,))


def decay_curve(decay: Any, max_cost: float, points: int = 60
                ) -> List[Tuple[float, float]]:
    """Evenly spaced ``(cost, weight)`` pairs on ``[0, max_cost]``."""
    f = _as_decay(decay)
    hi = _positive(max_cost, "max_cost")
    n = _positive_int(points, "points")
    if n == 1:
        return [(0.0, f(0.0))]
    step = hi / float(n - 1)
    return [(step * k, f(step * k)) for k in range(n)]


def decay_half_cost(decay: Any, upper: Optional[float] = None,
                    iterations: int = 200) -> float:
    """The cost at which the kernel has fallen to half its value at zero cost."""
    f = _as_decay(decay)
    it = _positive_int(iterations, "iterations")
    f0 = f(0.0)
    target = 0.5 * f0
    if f0 <= 0.0:
        return 0.0
    hi = 1.0 if upper is None else _positive(upper, "upper")
    if upper is None:
        for _ in range(200):
            if f(hi) <= target:
                break
            hi *= 2.0
        else:                                   # pragma: no cover
            raise AccessibilityError("the kernel never halves below %g." % hi)
    elif f(hi) > target:
        raise AccessibilityError("the kernel has not halved by cost %g." % hi)
    lo = 0.0
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:              # exhausted the float grid
            break
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mean_decay_cost(costs: Sequence[float],
                    weights: Optional[Sequence[float]] = None,
                    decay: Any = None) -> float:
    """Mean trip cost implied by a kernel: ``sum(w f(c) c) / sum(w f(c))``."""
    cs = [_as_cost(c, "costs") for c in costs]
    _non_empty(cs, "costs")
    w = _default_weights(weights, len(cs))
    f = _as_decay(decay)
    num = math.fsum(w[k] * f(cs[k]) * cs[k] for k in range(len(cs))
                    if cs[k] != _INF)
    den = math.fsum(w[k] * f(cs[k]) for k in range(len(cs)))
    if den == 0.0:
        raise AccessibilityError("the kernel gives zero weight to every trip.")
    return num / den


def calibrate_decay(costs: Sequence[float], target_mean_cost: float,
                    name: str = "exponential",
                    weights: Optional[Sequence[float]] = None,
                    iterations: int = 200, **fixed: float) -> DecayFunction:
    """Bisect a kernel's scale parameter onto an observed mean trip cost.

    ``mean_decay_cost`` is ``sum(w f(c) c) / sum(w f(c))``, a CONVEX
    COMBINATION of the costs, so its value is mathematically confined to
    ``[min(costs), max(costs)]`` -- and each kernel's tail narrows that
    further.  A negative-exponential can be driven all the way down to the
    cheapest trip, but a log-logistic with ``beta`` fixed at 3 bottoms out at
    ``sum(c^-2) / sum(c^-3)``, and a cumulative-Gaussian with ``sigma`` fixed
    cannot collapse onto the nearest trip at all.  No closed-form range check
    covers that, so the fit is VERIFIED after bisecting: either the kernel
    returned has an implied mean equal to ``target_mean_cost`` to 1e-9
    relative, or an ``AccessibilityError`` names the closest value the kernel
    reaches.  It never returns a parameter that has merely underflowed the
    kernel to nothing.
    """
    if name not in _DECAY_SCALE_PARAM:
        raise AccessibilityError(
            "%r is not calibratable by bisection (its mean cost is piecewise "
            "constant); choose from %s."
            % (name, ", ".join(sorted(_DECAY_SCALE_PARAM))))
    cs = [_as_cost(c, "costs") for c in costs]
    _non_empty(cs, "costs")
    w = _default_weights(weights, len(cs))
    target = _positive(target_mean_cost, "target_mean_cost")
    it = _positive_int(iterations, "iterations")
    key, direction = _DECAY_SCALE_PARAM[name]
    reachable = [cs[k] for k in range(len(cs)) if cs[k] != _INF and w[k] > 0.0]
    _non_empty(reachable, "finite costs")
    floor_cost = min(reachable)

    def mean_at(value: float) -> float:
        params = dict(fixed)
        params[key] = value
        return mean_decay_cost(cs, w, DecayFunction(name, params))

    def mean_or_floor(value: float) -> float:
        """The implied mean, or the floor once every weight has underflowed.

        A fully collapsed kernel sees only the very nearest trip, so its
        implied mean cost is the SMALLEST reachable cost -- never 0, which no
        convex combination of the costs can reach.  Reporting 0 here fabricated
        a bracket end that the kernel cannot attain and let the bisection
        converge on the underflow boundary, silently returning a parameter
        whose implied mean was nowhere near the requested target.
        """
        try:
            return mean_at(value)
        except AccessibilityError:
            return floor_cost

    def signed_gap(value: float) -> float:
        """``direction * (mean - target)``; a collapsed kernel sits at the floor."""
        return direction * (mean_or_floor(value) - target)

    def unreachable(bound: float) -> AccessibilityError:
        """The honest refusal: what the kernel actually reaches, and why."""
        return AccessibilityError(
            "cannot calibrate %s onto a mean trip cost of %.6g: the implied "
            "mean is a WEIGHTED AVERAGE of the costs, so it can never leave "
            "[%.6g, %.6g], and this kernel's tail narrows that further -- the "
            "closest it reaches is %.6g."
            % (key, target, floor_cost, max(reachable), bound))

    lo = hi = 1.0                               # bracket by doubling / halving
    for _ in range(400):
        if signed_gap(hi) >= 0.0:
            break
        hi *= 2.0
    else:
        raise unreachable(mean_or_floor(hi))
    for _ in range(400):
        if signed_gap(lo) <= 0.0:
            break
        lo *= 0.5
    else:
        raise unreachable(mean_or_floor(lo))
    for _ in range(it):
        mid = math.sqrt(lo * hi)                # geometric bisection
        if mid <= lo or mid >= hi:
            break
        if signed_gap(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    params = dict(fixed)
    params[key] = math.sqrt(lo * hi)
    fitted = DecayFunction(name, params)
    # VERIFY the fit.  Bisection on a continuous, strictly monotone mean
    # converges to machine precision inside the attainable range, so a miss
    # this large means the target lay OUTSIDE it: the bracket collapsed onto
    # the boundary where the kernel underflows rather than onto a solution.
    # The tail shape can put that boundary above ``min(costs)`` -- a
    # log-logistic with beta fixed at 3 bottoms out at
    # ``sum(c^-2) / sum(c^-3)``, not at the cheapest trip -- so no closed-form
    # range check can replace checking the answer.  Returning a kernel whose
    # implied mean is nowhere near the request is the one outcome this routine
    # must never have.
    achieved = mean_or_floor(params[key])
    if abs(achieved - target) > 1e-9 * max(1.0, abs(target)):
        raise unreachable(achieved)
    return fitted


# ===========================================================================
# 2. TRAVEL-COST MATRICES
# ===========================================================================
def haversine_m(a: Coord, b: Coord, radius: float = _EARTH_RADIUS_M) -> float:
    """Great-circle distance in metres between two ``(lat, lon)`` pairs."""
    lat1, lon1 = _as_float(a[0], "lat"), _as_float(a[1], "lon")
    lat2, lon2 = _as_float(b[0], "lat"), _as_float(b[1], "lon")
    r = _positive(radius, "radius")
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    h = (math.sin(dlat * 0.5) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlon * 0.5) ** 2)
    return 2.0 * r * math.asin(min(1.0, math.sqrt(h)))


def euclidean_m(a: Coord, b: Coord) -> float:
    """Planar distance between two projected ``(x, y)`` points, same units in."""
    return math.hypot(_as_float(a[0], "x") - _as_float(b[0], "x"),
                      _as_float(a[1], "y") - _as_float(b[1], "y"))


def travel_time(distance_m: float, speed_kph: float,
                circuity: float = 1.0) -> float:
    """Minutes to cover a distance at a speed, inflated by a circuity factor."""
    d = _non_negative(distance_m, "distance_m")
    v = _positive(speed_kph, "speed_kph")
    c = _as_float(circuity, "circuity")
    if c < 1.0:
        raise AccessibilityError("circuity must be >= 1 (got %.6g)." % c)
    return (d * c / 1000.0) / v * 60.0


def intrazonal_cost(area: float, speed_kph: Optional[float] = None,
                    factor: float = 2.0 / 3.0) -> float:
    """Self-cost of a zone: ``factor * sqrt(area / pi)``, optionally as minutes.

    The default factor is the exact mean distance from the centre of a disc to a
    uniformly drawn point inside it, ``2r/3``.
    """
    a = _non_negative(area, "area")
    f = _non_negative(factor, "factor")
    d = f * math.sqrt(a / math.pi)
    if speed_kph is None:
        return d
    return travel_time(d, speed_kph)


@dataclass
class CostMatrix:
    """Origin-by-destination generalised travel costs, with labels and units."""

    cost: Matrix
    origins: List[str] = field(default_factory=list)
    destinations: List[str] = field(default_factory=list)
    units: str = "minutes"

    def __post_init__(self) -> None:
        rows = list(self.cost)
        if not rows:
            raise AccessibilityError("cost must have at least one origin.")
        out: Matrix = []
        width = None
        for i, row in enumerate(rows):
            if isinstance(row, (str, bytes)) or not hasattr(row, "__iter__"):
                raise AccessibilityError("cost row %d must be a sequence." % i)
            vals = [_as_cost(v, "cost[%d]" % i) for v in row]
            if width is None:
                width = len(vals)
                if width == 0:
                    raise AccessibilityError(
                        "cost must have at least one destination.")
            elif len(vals) != width:
                raise AccessibilityError(
                    "cost is ragged: row 0 has %d entries, row %d has %d."
                    % (width, i, len(vals)))
            out.append(vals)
        self.cost = out
        if not self.origins:
            self.origins = ["O%d" % i for i in range(len(out))]
        if not self.destinations:
            self.destinations = ["D%d" % j for j in range(width or 0)]
        self.origins = [str(s) for s in self.origins]
        self.destinations = [str(s) for s in self.destinations]
        _check_length(self.origins, len(out), "origins", "row")
        _check_length(self.destinations, width or 0, "destinations", "column")

    @property
    def n_origins(self) -> int:
        """Number of origin rows."""
        return len(self.cost)

    @property
    def n_destinations(self) -> int:
        """Number of destination columns."""
        return len(self.cost[0])

    def row(self, i: int) -> Vector:
        """A copy of origin ``i``'s cost row."""
        return list(self.cost[i])

    def column(self, j: int) -> Vector:
        """A copy of destination ``j``'s cost column."""
        return [row[j] for row in self.cost]

    def get(self, i: int, j: int) -> float:
        """The cost from origin ``i`` to destination ``j``."""
        return self.cost[i][j]

    def scaled(self, factor: float) -> "CostMatrix":
        """Every cost multiplied by a positive factor (a change of units)."""
        f = _positive(factor, "factor")
        return CostMatrix([[c * f for c in row] for row in self.cost],
                          list(self.origins), list(self.destinations), self.units)

    def transposed(self) -> "CostMatrix":
        """The destination-by-origin matrix."""
        return CostMatrix([[self.cost[i][j] for i in range(self.n_origins)]
                           for j in range(self.n_destinations)],
                          list(self.destinations), list(self.origins), self.units)

    def with_diagonal(self, values: Any) -> "CostMatrix":
        """A copy whose diagonal is replaced by an intrazonal cost (square only)."""
        if self.n_origins != self.n_destinations:
            raise AccessibilityError("a diagonal needs a square cost matrix.")
        n = self.n_origins
        if isinstance(values, (int, float)):
            diag = [_non_negative(values, "values")] * n
        else:
            diag = _non_negative_vector(values, "values")
            _check_length(diag, n, "values", "zone")
        out = [list(row) for row in self.cost]
        for i in range(n):
            out[i][i] = diag[i]
        return CostMatrix(out, list(self.origins), list(self.destinations),
                          self.units)

    def max_finite(self) -> float:
        """The largest finite cost (0.0 if every cost is infinite)."""
        finite = [c for row in self.cost for c in row if c != _INF]
        return max(finite) if finite else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able summary (labels, units and shape, not the whole matrix)."""
        return {"origins": list(self.origins),
                "destinations": list(self.destinations),
                "units": self.units,
                "shape": [self.n_origins, self.n_destinations],
                "maxFinite": _json_safe(self.max_finite())}


def cost_matrix_from_coords(origins: Sequence[Coord],
                            destinations: Optional[Sequence[Coord]] = None,
                            speed_kph: Optional[float] = None,
                            metric: str = "haversine",
                            circuity: float = 1.0,
                            intrazonal: Optional[float] = None) -> CostMatrix:
    """Build a cost matrix from coordinates, as metres or (with a speed) minutes.

    ``circuity`` inflates the straight-line distance to account for the road
    network and must therefore be ``>= 1``, exactly as in ``travel_time``.  The
    check lives here rather than only inside ``travel_time`` because the metres
    branch never reaches ``travel_time``: a circuity of 0.5 used to return a
    network distance SHORTER than the great circle, and a non-numeric circuity
    used to leak a raw ``TypeError``.
    """
    if metric not in ("haversine", "euclidean"):
        raise AccessibilityError("metric must be 'haversine' or 'euclidean' "
                                 "(got %r)." % (metric,))
    circ = _as_float(circuity, "circuity")
    if circ < 1.0:
        raise AccessibilityError("circuity must be >= 1 (got %.6g)." % circ)
    orig = [(_as_float(p[0], "coord"), _as_float(p[1], "coord"))
            for p in origins]
    _non_empty(orig, "origins")
    dest = orig if destinations is None else [
        (_as_float(p[0], "coord"), _as_float(p[1], "coord")) for p in destinations]
    _non_empty(dest, "destinations")
    dist = haversine_m if metric == "haversine" else euclidean_m
    rows: Matrix = []
    for a in orig:
        row = []
        for b in dest:
            d = dist(a, b)
            row.append(d * circ if speed_kph is None
                       else travel_time(d, speed_kph, circ))
        rows.append(row)
    cm = CostMatrix(rows, [], [], "metres" if speed_kph is None else "minutes")
    if intrazonal is not None:
        cm = cm.with_diagonal(intrazonal)
    return cm


def _dijkstra(adjacency: List[List[Tuple[int, float]]], source: int) -> Vector:
    """Shortest-path costs from one source over an index-based adjacency list."""
    n = len(adjacency)
    dist = [_INF] * n
    dist[source] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, source)]
    seen = [False] * n
    while heap:
        d, u = heappop(heap)
        if seen[u]:
            continue
        seen[u] = True
        for v, w in adjacency[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heappush(heap, (nd, v))
    return dist


def cost_matrix_from_graph(edges: Sequence[Tuple[Any, Any, float]],
                           origins: Sequence[Any],
                           destinations: Optional[Sequence[Any]] = None,
                           directed: bool = False) -> CostMatrix:
    """Shortest-path cost matrix over a weighted graph (from-scratch Dijkstra)."""
    index: Dict[Any, int] = {}
    order: List[Any] = []

    def node_id(node: Any) -> int:
        if node not in index:
            index[node] = len(order)
            order.append(node)
        return index[node]

    clean: List[Tuple[int, int, float]] = []
    for e in edges:
        if len(e) != 3:
            raise AccessibilityError("each edge must be (u, v, weight).")
        u, v, w = e
        clean.append((node_id(u), node_id(v), _non_negative(w, "edge weight")))
    orig = list(origins)
    _non_empty(orig, "origins")
    dest = list(orig if destinations is None else destinations)
    _non_empty(dest, "destinations")
    for node in orig + dest:
        if node not in index:
            raise AccessibilityError("node %r does not appear in any edge."
                                     % (node,))
    adjacency: List[List[Tuple[int, float]]] = [[] for _ in order]
    for u, v, w in clean:
        adjacency[u].append((v, w))
        if not directed:
            adjacency[v].append((u, w))
    rows: Matrix = []
    for o in orig:
        dist = _dijkstra(adjacency, index[o])
        rows.append([dist[index[d]] for d in dest])
    return CostMatrix(rows, [str(o) for o in orig], [str(d) for d in dest],
                      "cost")


# ===========================================================================
# 3. CUMULATIVE OPPORTUNITY AND ITS DUAL
# ===========================================================================
def cumulative_opportunity(cost: Any, opportunities: Sequence[float],
                           threshold: float, inclusive: bool = True) -> Vector:
    """Opportunities reachable within a cost threshold, one number per origin."""
    c = _as_cost_matrix(cost)
    o = _non_negative_vector(opportunities, "opportunities")
    _check_length(o, len(c[0]), "opportunities", "destination")
    t = _as_cost(threshold, "threshold")
    out: Vector = []
    for row in c:
        if inclusive:
            out.append(math.fsum(o[j] for j in range(len(o)) if row[j] <= t))
        else:
            out.append(math.fsum(o[j] for j in range(len(o)) if row[j] < t))
    return out


@dataclass
class CumulativeResult:
    """Cumulative-opportunity curves: one row per origin, one column per cut-off."""

    thresholds: List[float]
    access: List[List[float]]
    mean: List[float]
    total_opportunities: float
    labels: List[str] = field(default_factory=list)

    def curve(self, origin: int) -> List[Tuple[float, float]]:
        """``(threshold, opportunities)`` pairs for one origin."""
        return [(self.thresholds[k], self.access[origin][k])
                for k in range(len(self.thresholds))]

    def at(self, threshold: float) -> Vector:
        """The column for an exact threshold (raises if it was not computed)."""
        for k, t in enumerate(self.thresholds):
            if t == threshold:
                return [row[k] for row in self.access]
        raise AccessibilityError("threshold %g was not evaluated." % threshold)

    def share(self, origin: int, k: int) -> float:
        """Share of all opportunities reachable by one origin at cut-off ``k``."""
        if self.total_opportunities == 0.0:
            return 0.0
        return self.access[origin][k] / self.total_opportunities


def cumulative_opportunity_curve(cost: Any, opportunities: Sequence[float],
                                 thresholds: Sequence[float],
                                 population: Optional[Sequence[float]] = None,
                                 labels: Optional[Sequence[str]] = None
                                 ) -> CumulativeResult:
    """Cumulative opportunity at several cut-offs, with population-weighted means."""
    c = _as_cost_matrix(cost)
    o = _non_negative_vector(opportunities, "opportunities")
    _check_length(o, len(c[0]), "opportunities", "destination")
    ts = [_as_cost(t, "thresholds") for t in thresholds]
    _non_empty(ts, "thresholds")
    pop = _default_weights(population, len(c), "population")
    access = [[0.0] * len(ts) for _ in range(len(c))]
    for k, t in enumerate(ts):
        column = cumulative_opportunity(c, o, t)
        for i in range(len(c)):
            access[i][k] = column[i]
    means = [_mean([access[i][k] for i in range(len(c))], pop)
             for k in range(len(ts))]
    names = ["O%d" % i for i in range(len(c))] if labels is None else \
        [str(s) for s in labels]
    _check_length(names, len(c), "labels", "origin")
    return CumulativeResult(ts, access, means, math.fsum(o), names)


def isochrone_bands(cost_row: Sequence[float], opportunities: Sequence[float],
                    edges: Sequence[float]) -> List[Tuple[float, float, float]]:
    """Opportunities per ``(lower, upper]`` band; the bands sum to the last cut-off."""
    row = [_as_cost(c, "cost_row") for c in cost_row]
    o = _non_negative_vector(opportunities, "opportunities")
    _check_length(o, len(row), "opportunities", "destination")
    es = [_as_cost(e, "edges") for e in edges]
    _non_empty(es, "edges")
    for k in range(1, len(es)):
        if es[k] <= es[k - 1]:
            raise AccessibilityError("edges must be strictly increasing.")
    bands: List[Tuple[float, float, float]] = []
    lower = 0.0
    for k, upper in enumerate(es):
        total = math.fsum(o[j] for j in range(len(o))
                          if row[j] <= upper and (k == 0 or row[j] > lower))
        bands.append((lower, upper, total))
        lower = upper
    return bands


def time_to_nth_opportunity(cost_row: Sequence[float],
                            opportunities: Sequence[float], n: float) -> float:
    """Cost of reaching the ``n``-th opportunity -- the dual accessibility measure."""
    row = [_as_cost(c, "cost_row") for c in cost_row]
    o = _non_negative_vector(opportunities, "opportunities")
    _check_length(o, len(row), "opportunities", "destination")
    target = _non_negative(n, "n")
    if target == 0.0:
        return 0.0
    taken: Vector = []
    for c, j in _sorted_with_index(row):
        taken.append(o[j])
        # fsum (not a running total) so this agrees with cumulative_opportunity
        # to the last bit, which makes the two measures exact inverses.
        if math.fsum(taken) >= target:
            return c
    return _INF


@dataclass
class PlaceRankResult:
    """The dual measure: the cost at which each origin reaches N opportunities."""

    targets: List[float]
    costs: List[List[float]]
    labels: List[str] = field(default_factory=list)

    def reachable(self, origin: int, k: int) -> bool:
        """Whether origin ``origin`` ever reaches target ``k``."""
        return self.costs[origin][k] != _INF

    def median_cost(self, k: int) -> float:
        """Median cost across origins for target ``k`` (infinities included)."""
        column = sorted(row[k] for row in self.costs)
        m = len(column)
        if m % 2 == 1:
            return column[m // 2]
        lo, hi = column[m // 2 - 1], column[m // 2]
        if lo == _INF or hi == _INF:
            return _INF
        return 0.5 * (lo + hi)


def place_rank_accessibility(cost: Any, opportunities: Sequence[float],
                             targets: Sequence[float],
                             labels: Optional[Sequence[str]] = None
                             ) -> PlaceRankResult:
    """Cost to reach N opportunities, for several N -- the inverse of cumulative."""
    c = _as_cost_matrix(cost)
    o = _non_negative_vector(opportunities, "opportunities")
    _check_length(o, len(c[0]), "opportunities", "destination")
    ts = [_non_negative(t, "targets") for t in targets]
    _non_empty(ts, "targets")
    costs = [[time_to_nth_opportunity(row, o, t) for t in ts] for row in c]
    names = ["O%d" % i for i in range(len(c))] if labels is None else \
        [str(s) for s in labels]
    _check_length(names, len(c), "labels", "origin")
    return PlaceRankResult(ts, costs, names)


# ===========================================================================
# 4. GRAVITY / POTENTIAL ACCESSIBILITY
# ===========================================================================
def gravity_accessibility(cost: Any, opportunities: Sequence[float],
                          decay: Any, exponent: float = 1.0) -> Vector:
    """Hansen-type potential ``A_i = sum_j o_j^alpha f(c_ij)``, one per origin."""
    c = _as_cost_matrix(cost)
    o = _non_negative_vector(opportunities, "opportunities")
    _check_length(o, len(c[0]), "opportunities", "destination")
    f = _as_decay(decay)
    alpha = _non_negative(exponent, "exponent")
    size = o if alpha == 1.0 else [v ** alpha if v > 0.0 else 0.0 for v in o]
    return [math.fsum(size[j] * f(row[j]) for j in range(len(o))) for row in c]


@dataclass
class AccessibilityResult:
    """A named accessibility surface: one value per origin plus its provenance."""

    measure: str
    values: List[float]
    labels: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    units: str = "opportunities"

    def __post_init__(self) -> None:
        self.values = [float(v) for v in self.values]
        if not self.labels:
            self.labels = ["O%d" % i for i in range(len(self.values))]
        self.labels = [str(s) for s in self.labels]
        _check_length(self.labels, len(self.values), "labels", "origin")

    def mean(self, weights: Optional[Sequence[float]] = None) -> float:
        """Arithmetic or population-weighted mean accessibility."""
        w = _default_weights(weights, len(self.values), "weights")
        return _mean(self.values, w)

    def normalised(self, method: str = "max") -> Vector:
        """The surface rescaled to a dimensionless index."""
        return normalise_accessibility(self.values, method)

    def ranking(self) -> List[int]:
        """Origin indices best first, ties broken by index."""
        return [i for (_v, i) in
                sorted(((-v, i) for i, v in enumerate(self.values)),
                       key=lambda p: (p[0], p[1]))]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able description of the surface."""
        return {"measure": self.measure, "units": self.units,
                "labels": list(self.labels),
                "values": [_json_safe(v) for v in self.values],
                "mean": _json_safe(self.mean()),
                "params": {k: _json_safe(self.params[k])
                           for k in sorted(self.params, key=str)}}


def hansen_accessibility(cost: Any, opportunities: Sequence[float],
                         beta: float = 1.5, alpha: float = 1.0,
                         cost_min: float = 1.0,
                         labels: Optional[Sequence[str]] = None
                         ) -> AccessibilityResult:
    """Hansen (1959) potential with the classic inverse-power impedance."""
    f = get_decay("power", beta=beta, cost_min=cost_min)
    values = gravity_accessibility(cost, opportunities, f, alpha)
    return AccessibilityResult("hansen", values, list(labels or []),
                               {"beta": float(beta), "alpha": float(alpha),
                                "costMin": float(cost_min)})


def potential_accessibility(cost: Any, opportunities: Sequence[float],
                            decay: Any, exclude_self: bool = False,
                            self_cost: Optional[float] = None,
                            labels: Optional[Sequence[str]] = None
                            ) -> AccessibilityResult:
    """Potential accessibility with explicit control of the self-potential term."""
    c = _as_cost_matrix(cost)
    o = _non_negative_vector(opportunities, "opportunities")
    _check_length(o, len(c[0]), "opportunities", "destination")
    f = _as_decay(decay)
    if exclude_self or self_cost is not None:
        if len(c) != len(c[0]):
            raise AccessibilityError(
                "self-potential handling needs a square cost matrix.")
    if self_cost is not None:
        sc = _non_negative(self_cost, "self_cost")
        c = [list(row) for row in c]
        for i in range(len(c)):
            c[i][i] = sc
    values: Vector = []
    for i, row in enumerate(c):
        terms = [o[j] * f(row[j]) for j in range(len(o))
                 if not (exclude_self and i == j)]
        values.append(math.fsum(terms))
    return AccessibilityResult("potential", values, list(labels or []),
                               {"decay": f.to_dict(),
                                "excludeSelf": bool(exclude_self),
                                "selfCost": (None if self_cost is None
                                             else float(self_cost))})


def shen_accessibility(cost: Any, supply: Sequence[float],
                       demand: Sequence[float], decay: Any) -> Vector:
    """Shen (1998): gravity accessibility divided by the competing demand."""
    c = _as_cost_matrix(cost)
    s = _non_negative_vector(supply, "supply")
    d = _non_negative_vector(demand, "demand")
    _check_length(s, len(c[0]), "supply", "destination")
    _check_length(d, len(c), "demand", "origin")
    f = _as_decay(decay)
    w = [[f(v) for v in row] for row in c]
    pressure = [math.fsum(d[i] * w[i][j] for i in range(len(c)))
                for j in range(len(s))]
    out: Vector = []
    for i in range(len(c)):
        out.append(math.fsum(s[j] * w[i][j] / pressure[j]
                             for j in range(len(s)) if pressure[j] > 0.0))
    return out


def normalise_accessibility(values: Sequence[float], method: str = "max"
                            ) -> Vector:
    """Rescale a surface: 'max', 'mean', 'sum', 'minmax' or 'zscore'."""
    v = _as_floats(values, "values")
    _non_empty(v, "values")
    if method == "max":
        top = max(v)
        return [0.0] * len(v) if top == 0.0 else [x / top for x in v]
    if method == "mean":
        mu = _mean(v)
        return [0.0] * len(v) if mu == 0.0 else [x / mu for x in v]
    if method == "sum":
        total = math.fsum(v)
        return [0.0] * len(v) if total == 0.0 else [x / total for x in v]
    if method == "minmax":
        lo, hi = min(v), max(v)
        if hi == lo:
            return [0.0] * len(v)
        return [(x - lo) / (hi - lo) for x in v]
    if method == "zscore":
        # Max-scaled moments: a z-score is a RATIO of deviations, so working in
        # units of the peak leaves it unchanged while keeping the sum of squares
        # inside double range at every scale (see ``_scaled_moments``).
        mu, sd, scale = _scaled_moments(v)
        if sd == 0.0:
            return [0.0] * len(v)
        return [(x / scale - mu) / sd for x in v]
    raise AccessibilityError("unknown normalisation %r (max, mean, sum, minmax "
                             "or zscore)." % (method,))


# ===========================================================================
# 5. FLOATING CATCHMENT AREA METHODS
# ===========================================================================
@dataclass
class FCAResult:
    """A floating-catchment solution: supply-to-demand ratios and accessibility."""

    method: str
    access: List[float]
    ratios: List[float]
    catchment_demand: List[float]
    catchment_supply: List[float]
    params: Dict[str, Any] = field(default_factory=dict)
    supply_radii: List[float] = field(default_factory=list)
    demand_radii: List[float] = field(default_factory=list)

    @property
    def conserves_supply(self) -> bool:
        """Whether this variant redistributes every unit of reachable supply."""
        return bool(self.params.get("conserves", False))

    def allocated_supply(self, demand: Sequence[float]) -> float:
        """Total supply implied by the surface, ``sum_i P_i A_i``."""
        d = _non_negative_vector(demand, "demand")
        _check_length(d, len(self.access), "demand", "origin")
        return math.fsum(d[i] * self.access[i] for i in range(len(d)))

    def reachable_supply(self) -> float:
        """Total supply sitting in a catchment that contains somebody."""
        return math.fsum(self.ratios[j] * self.catchment_demand[j]
                         for j in range(len(self.ratios)))

    def weighted_mean(self, demand: Sequence[float]) -> float:
        """Population-weighted mean accessibility."""
        d = _default_weights(demand, len(self.access), "demand")
        return _mean(self.access, d)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able summary of the solution."""
        return {"method": self.method,
                "access": [_json_safe(v) for v in self.access],
                "ratios": [_json_safe(v) for v in self.ratios],
                "conservesSupply": self.conserves_supply,
                "params": {k: _json_safe(self.params[k])
                           for k in sorted(self.params, key=str)}}


def _as_weight_matrix(weights: Any, n_rows: int, n_cols: int,
                      name: str = "weights") -> Matrix:
    """Validate a non-negative, finite, rectangular kernel-weight matrix."""
    rows = list(weights)
    if len(rows) != n_rows:
        raise AccessibilityError("%s must have %d rows (got %d)."
                                 % (name, n_rows, len(rows)))
    out: Matrix = []
    for i, row in enumerate(rows):
        vals = _non_negative_vector(row, "%s[%d]" % (name, i))
        if len(vals) != n_cols:
            raise AccessibilityError("%s row %d must have %d entries (got %d)."
                                     % (name, i, n_cols, len(vals)))
        out.append(vals)
    return out


def supply_demand_ratio(weights: Any, supply: Sequence[float],
                        demand: Sequence[float]) -> Vector:
    """Step one of every catchment method: ``R_j = S_j / sum_i P_i w_ij``."""
    s = _non_negative_vector(supply, "supply")
    d = _non_negative_vector(demand, "demand")
    w = _as_weight_matrix(weights, len(d), len(s))
    out: Vector = []
    for j in range(len(s)):
        pressure = math.fsum(d[i] * w[i][j] for i in range(len(d)))
        out.append(s[j] / pressure if pressure > 0.0 else 0.0)
    return out


def _fca_core(method: str, w1: Matrix, w2: Matrix, supply: Vector,
              demand: Vector, params: Dict[str, Any],
              supply_radii: Optional[Vector] = None,
              demand_radii: Optional[Vector] = None) -> FCAResult:
    """Shared two-step machinery: ratios from ``w1``, accessibility from ``w2``."""
    n_i, n_j = len(demand), len(supply)
    pressure = [math.fsum(demand[i] * w1[i][j] for i in range(n_i))
                for j in range(n_j)]
    ratios = [supply[j] / pressure[j] if pressure[j] > 0.0 else 0.0
              for j in range(n_j)]
    access = [math.fsum(ratios[j] * w2[i][j] for j in range(n_j))
              for i in range(n_i)]
    catchment_supply = [math.fsum(supply[j] for j in range(n_j) if w2[i][j] > 0.0)
                        for i in range(n_i)]
    return FCAResult(method, access, ratios, pressure, catchment_supply,
                     dict(params), list(supply_radii or []),
                     list(demand_radii or []))


def _truncated_weights(cost: Matrix, decay: DecayFunction,
                       radius: Optional[float]) -> Matrix:
    """Kernel weights, zeroed beyond a catchment radius when one is given."""
    if radius is None:
        return [[decay(c) for c in row] for row in cost]
    r = _as_cost(radius, "radius")
    return [[decay(c) if c <= r else 0.0 for c in row] for row in cost]


def _fca_inputs(cost: Any, supply: Sequence[float], demand: Sequence[float]
                ) -> Tuple[Matrix, Vector, Vector]:
    """Validate the common (cost, supply, demand) triple."""
    c = _as_cost_matrix(cost)
    s = _non_negative_vector(supply, "supply")
    d = _non_negative_vector(demand, "demand")
    _check_length(s, len(c[0]), "supply", "destination")
    _check_length(d, len(c), "demand", "origin")
    return c, s, d


def two_step_fca(cost: Any, supply: Sequence[float], demand: Sequence[float],
                 radius: Optional[float] = None, decay: Any = None) -> FCAResult:
    """Luo & Wang's 2SFCA: catchment ratios, then catchment sums."""
    c, s, d = _fca_inputs(cost, supply, demand)
    if radius is None and decay is None:
        raise AccessibilityError("2SFCA needs a catchment radius or a decay.")
    f = _as_decay(decay, get_decay("rectangular", threshold=radius if radius
                                   is not None else 0.0))
    w = _truncated_weights(c, f, radius if decay is not None else None)
    return _fca_core("2SFCA", w, w, s, d,
                     {"radius": (None if radius is None else float(radius)),
                      "decay": f.to_dict(), "conserves": True})


def e2sfca_zone_weights(zones: Sequence[Tuple[float, float]]
                        ) -> List[Tuple[float, float]]:
    """Validate E2SFCA travel-time zones: ``[(upper_cost, weight), ...]``."""
    out: List[Tuple[float, float]] = []
    for z in zones:
        if len(z) != 2:
            raise AccessibilityError("each zone must be (upper_cost, weight).")
        upper = _as_cost(z[0], "zone upper cost")
        weight = _non_negative(z[1], "zone weight")
        out.append((upper, weight))
    _non_empty(out, "zones")
    for k in range(1, len(out)):
        if out[k][0] <= out[k - 1][0]:
            raise AccessibilityError("zone upper costs must strictly increase.")
    return out


def _zone_weight(cost: float, zones: Sequence[Tuple[float, float]]) -> float:
    """The step weight of a cost under a list of E2SFCA zones (0 beyond them)."""
    for upper, weight in zones:
        if cost <= upper:
            return weight
    return 0.0


def enhanced_2sfca(cost: Any, supply: Sequence[float],
                   demand: Sequence[float],
                   zones: Sequence[Tuple[float, float]]) -> FCAResult:
    """Luo & Qi's E2SFCA: the catchment is banded and each band gets a weight."""
    c, s, d = _fca_inputs(cost, supply, demand)
    zs = e2sfca_zone_weights(zones)
    w = [[_zone_weight(v, zs) for v in row] for row in c]
    return _fca_core("E2SFCA", w, w, s, d,
                     {"zones": [[float(u), float(g)] for (u, g) in zs],
                      "conserves": True})


def three_step_fca(cost: Any, supply: Sequence[float], demand: Sequence[float],
                   radius: Optional[float] = None, decay: Any = None
                   ) -> FCAResult:
    """Wan's 3SFCA: a selection probability removes the demand double-counting."""
    c, s, d = _fca_inputs(cost, supply, demand)
    if radius is None and decay is None:
        raise AccessibilityError("3SFCA needs a catchment radius or a decay.")
    f = _as_decay(decay, get_decay("rectangular", threshold=radius if radius
                                   is not None else 0.0))
    w = _truncated_weights(c, f, radius if decay is not None else None)
    g: Matrix = []
    for row in w:
        total = math.fsum(row)
        g.append([v / total if total > 0.0 else 0.0 for v in row])
    gw = [[g[i][j] * w[i][j] for j in range(len(s))] for i in range(len(d))]
    return _fca_core("3SFCA", gw, gw, s, d,
                     {"radius": (None if radius is None else float(radius)),
                      "decay": f.to_dict(), "conserves": True})


def m2sfca(cost: Any, supply: Sequence[float], demand: Sequence[float],
           radius: Optional[float] = None, decay: Any = None) -> FCAResult:
    """Delamater's M2SFCA: the impedance enters twice, penalising long catchments."""
    c, s, d = _fca_inputs(cost, supply, demand)
    if radius is None and decay is None:
        raise AccessibilityError("M2SFCA needs a catchment radius or a decay.")
    f = _as_decay(decay, get_decay("rectangular", threshold=radius if radius
                                   is not None else 0.0))
    w = _truncated_weights(c, f, radius if decay is not None else None)
    w2 = [[v * v for v in row] for row in w]
    return _fca_core("M2SFCA", w, w2, s, d,
                     {"radius": (None if radius is None else float(radius)),
                      "decay": f.to_dict(), "conserves": False})


def variable_catchment_2sfca(cost: Any, supply: Sequence[float],
                             demand: Sequence[float],
                             population_target: float,
                             supply_target: float,
                             decay: Any = None,
                             max_cost: Optional[float] = None) -> FCAResult:
    """2SFCA whose catchments grow until they capture enough population / supply."""
    c, s, d = _fca_inputs(cost, supply, demand)
    pt = _non_negative(population_target, "population_target")
    st = _non_negative(supply_target, "supply_target")
    cap = _INF if max_cost is None else _as_cost(max_cost, "max_cost")
    f = None if decay is None else _as_decay(decay)

    def base(value: float) -> float:
        """Weight inside the catchment: flat when no kernel was supplied."""
        return 1.0 if f is None else f(value)

    supply_radii: Vector = []
    for j in range(len(s)):
        column = _sorted_with_index([c[i][j] for i in range(len(d))])
        captured, radius = 0.0, 0.0
        for cost_ij, i in column:
            if cost_ij > cap or cost_ij == _INF:
                break
            radius = cost_ij
            captured += d[i]
            if captured >= pt:
                break
        supply_radii.append(radius)

    demand_radii: Vector = []
    for i in range(len(d)):
        row = _sorted_with_index(c[i])
        captured, radius = 0.0, 0.0
        for cost_ij, j in row:
            if cost_ij > cap or cost_ij == _INF:
                break
            radius = cost_ij
            captured += s[j]
            if captured >= st:
                break
        demand_radii.append(radius)

    w1 = [[base(c[i][j]) if c[i][j] <= supply_radii[j] else 0.0
           for j in range(len(s))] for i in range(len(d))]
    w2 = [[base(c[i][j]) if c[i][j] <= demand_radii[i] else 0.0
           for j in range(len(s))] for i in range(len(d))]
    return _fca_core("V2SFCA", w1, w2, s, d,
                     {"populationTarget": pt, "supplyTarget": st,
                      "maxCost": (None if max_cost is None else float(max_cost)),
                      "decay": (None if f is None else f.to_dict()),
                      "conserves": False},
                     supply_radii, demand_radii)


# ===========================================================================
# 6. UTILITY-BASED (LOGSUM) ACCESSIBILITY
# ===========================================================================
def utility_matrix(cost: Any, opportunities: Sequence[float],
                   beta_cost: float, size_coefficient: float = 1.0,
                   constants: Optional[Sequence[float]] = None) -> Matrix:
    """Destination utilities ``V_ij = theta ln(o_j) - beta c_ij + k_j``."""
    c = _as_cost_matrix(cost)
    o = _non_negative_vector(opportunities, "opportunities")
    _check_length(o, len(c[0]), "opportunities", "destination")
    beta = _non_negative(beta_cost, "beta_cost")
    theta = _as_float(size_coefficient, "size_coefficient")
    k = [0.0] * len(o) if constants is None else _as_floats(constants,
                                                            "constants")
    _check_length(k, len(o), "constants", "destination")
    out: Matrix = []
    for row in c:
        vals: Vector = []
        for j in range(len(o)):
            if o[j] <= 0.0 or row[j] == _INF:
                vals.append(-_INF)
            else:
                vals.append(theta * math.log(o[j]) - beta * row[j] + k[j])
        out.append(vals)
    return out


def _logsumexp(values: Sequence[float]) -> float:
    """Numerically safe ``ln sum exp(v)``; ``-inf`` when nothing is available."""
    finite = [v for v in values if v != -_INF]
    if not finite:
        return -_INF
    top = max(finite)
    return top + math.log(math.fsum(math.exp(v - top) for v in finite))


def destination_choice_probabilities(utilities: Sequence[float],
                                     scale: float = 1.0) -> Vector:
    """Multinomial-logit shares of one origin's destinations (they sum to 1)."""
    try:
        v = [float(x) for x in utilities]
    except (TypeError, ValueError):
        raise AccessibilityError("utilities must be a sequence of numbers.")
    _non_empty(v, "utilities")
    for x in v:
        if math.isnan(x) or x == _INF:
            raise AccessibilityError("utilities must be finite or -inf.")
    mu = _positive(scale, "scale")
    scaled = [mu * x if x != -_INF else -_INF for x in v]
    lse = _logsumexp(scaled)
    if lse == -_INF:
        raise AccessibilityError("no destination is available to this origin.")
    return [0.0 if x == -_INF else math.exp(x - lse) for x in scaled]


def logsum_accessibility(cost: Any, opportunities: Sequence[float],
                         beta_cost: float, scale: float = 1.0,
                         size_coefficient: float = 1.0,
                         constants: Optional[Sequence[float]] = None) -> Vector:
    """The logit logsum ``(1/mu) ln sum_j exp(mu V_ij)`` -- accessibility as welfare."""
    mu = _positive(scale, "scale")
    v = utility_matrix(cost, opportunities, beta_cost, size_coefficient,
                       constants)
    out: Vector = []
    for row in v:
        lse = _logsumexp([mu * x if x != -_INF else -_INF for x in row])
        out.append(-_INF if lse == -_INF else lse / mu)
    return out


def logsum_benefit(before: Sequence[float], after: Sequence[float],
                   marginal_utility_of_income: float,
                   population: Optional[Sequence[float]] = None
                   ) -> Dict[str, Any]:
    """Consumer surplus of a scheme: the logsum change over the income utility."""
    b = _as_floats(before, "before")
    a = _as_floats(after, "after")
    _check_length(a, len(b), "after", "origin")
    _non_empty(b, "before")
    lam = _positive(marginal_utility_of_income, "marginal_utility_of_income")
    pop = _default_weights(population, len(b), "population")
    per_person = [(a[i] - b[i]) / lam for i in range(len(b))]
    total = math.fsum(pop[i] * per_person[i] for i in range(len(b)))
    return {"perPerson": per_person,
            "total": total,
            "mean": total / math.fsum(pop),
            "gainers": sum(1 for x in per_person if x > 0.0),
            "losers": sum(1 for x in per_person if x < 0.0)}


# ===========================================================================
# 7. TIME OF DAY AND RELIABILITY
# ===========================================================================
@dataclass
class TimeOfDayResult:
    """Accessibility by period, and the day as a weighted whole."""

    periods: List[str]
    weights: List[float]
    access: List[List[float]]
    weighted: List[float]
    best_period: List[str] = field(default_factory=list)
    worst_period: List[str] = field(default_factory=list)
    variability: List[float] = field(default_factory=list)

    def period_mean(self, k: int,
                    population: Optional[Sequence[float]] = None) -> float:
        """Population-weighted mean accessibility in one period."""
        column = [row[k] for row in self.access]
        w = _default_weights(population, len(column), "population")
        return _mean(column, w)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able summary."""
        return {"periods": list(self.periods),
                "weights": [_json_safe(w) for w in self.weights],
                "weighted": [_json_safe(v) for v in self.weighted],
                "variability": [_json_safe(v) for v in self.variability]}


def time_of_day_accessibility(periods: Sequence[Tuple[str, Any, float]],
                              opportunities: Sequence[float], decay: Any,
                              exponent: float = 1.0) -> TimeOfDayResult:
    """Accessibility in each ``(name, cost_matrix, weight)`` period and overall."""
    entries = list(periods)
    _non_empty(entries, "periods")
    names: List[str] = []
    mats: List[Matrix] = []
    raw: Vector = []
    for e in entries:
        if len(e) != 3:
            raise AccessibilityError("each period must be (name, cost, weight).")
        names.append(str(e[0]))
        mats.append(_as_cost_matrix(e[1], "period cost"))
        raw.append(_non_negative(e[2], "period weight"))
    n_i = len(mats[0])
    n_j = len(mats[0][0])
    for m in mats[1:]:
        if len(m) != n_i or len(m[0]) != n_j:
            raise AccessibilityError("every period needs the same matrix shape.")
    total = math.fsum(raw)
    if total <= 0.0:
        raise AccessibilityError("period weights must not sum to zero.")
    w = [v / total for v in raw]
    columns = [gravity_accessibility(m, opportunities, decay, exponent)
               for m in mats]
    access = [[columns[k][i] for k in range(len(names))] for i in range(n_i)]
    weighted = [math.fsum(w[k] * access[i][k] for k in range(len(names)))
                for i in range(n_i)]
    best: List[str] = []
    worst: List[str] = []
    variability: Vector = []
    for i in range(n_i):
        row = access[i]
        best.append(names[max(range(len(row)), key=lambda k: (row[k], -k))])
        worst.append(names[min(range(len(row)), key=lambda k: (row[k], k))])
        # Max-scaled moments so the day-to-day coefficient of variation is the
        # same on a surface of 1e-170 opportunities as on one of 1e+200; the raw
        # sum of squares underflowed to 0.0 (a false "perfectly stable day") on
        # the first and overflowed on the second.
        mu, sd, _scale = _scaled_moments(row)
        if mu == 0.0:
            variability.append(0.0)
        else:
            variability.append(sd / mu)
    return TimeOfDayResult(names, w, access, weighted, best, worst, variability)


def travel_time_percentile(samples: Sequence[float], percentile: float) -> float:
    """Linear-interpolated percentile of a travel-time sample (NumPy convention)."""
    xs = sorted(_as_floats(samples, "samples"))
    _non_empty(xs, "samples")
    p = _as_float(percentile, "percentile")
    if p < 0.0 or p > 100.0:
        raise AccessibilityError("percentile must lie in [0, 100] (got %.6g)." % p)
    if len(xs) == 1:
        return xs[0]
    h = (len(xs) - 1) * (p / 100.0)
    lo = int(math.floor(h))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (h - lo) * (xs[hi] - xs[lo])


def buffer_time_index(samples: Sequence[float], percentile: float = 95.0
                      ) -> float:
    """``(p95 - mean) / mean``: the extra time budget a traveller must keep."""
    xs = _as_floats(samples, "samples")
    _non_empty(xs, "samples")
    mu = _mean(xs)
    if mu <= 0.0:
        raise AccessibilityError("the mean travel time must be > 0.")
    return (travel_time_percentile(xs, percentile) - mu) / mu


def planning_time_index(samples: Sequence[float], free_flow: float,
                        percentile: float = 95.0) -> float:
    """``p95 / free-flow time``: how much of a trip is schedule padding."""
    xs = _as_floats(samples, "samples")
    _non_empty(xs, "samples")
    ff = _positive(free_flow, "free_flow")
    return travel_time_percentile(xs, percentile) / ff


def reliability_adjusted_cost(mean_cost: float, std_cost: float,
                              reliability_ratio: float = 0.8) -> float:
    """Generalised cost with a reliability term: ``mean + lambda * sd``."""
    m = _non_negative(mean_cost, "mean_cost")
    sd = _non_negative(std_cost, "std_cost")
    lam = _non_negative(reliability_ratio, "reliability_ratio")
    return m + lam * sd


def reliability_adjusted_accessibility(cost_samples: Sequence[Sequence[
        Sequence[float]]], opportunities: Sequence[float], decay: Any,
        percentile: Optional[float] = None,
        reliability_ratio: float = 0.8) -> Vector:
    """Gravity accessibility on reliability-adjusted (percentile or mean+sd) costs."""
    rows = list(cost_samples)
    _non_empty(rows, "cost_samples")
    effective: Matrix = []
    width = None
    for i, row in enumerate(rows):
        cells = list(row)
        if width is None:
            width = len(cells)
            if width == 0:
                raise AccessibilityError("cost_samples needs a destination.")
        elif len(cells) != width:
            raise AccessibilityError("cost_samples is ragged at row %d." % i)
        out_row: Vector = []
        for cell in cells:
            xs = _as_floats(cell, "cost_samples")
            _non_empty(xs, "cost samples")
            for v in xs:
                if v < 0.0:
                    raise AccessibilityError("costs must be >= 0.")
            if percentile is None:
                # Max-scaled moments, then back to cost units: squaring a raw
                # deviation of 1e200 overflowed and leaked an OverflowError.
                mu, sd, scale = _scaled_moments(xs)
                out_row.append(reliability_adjusted_cost(mu * scale,
                                                         sd * scale,
                                                         reliability_ratio))
            else:
                out_row.append(travel_time_percentile(xs, percentile))
        effective.append(out_row)
    return gravity_accessibility(effective, opportunities, decay)


# ===========================================================================
# 8. INEQUALITY -- LORENZ, GINI, THEIL, ATKINSON AND FRIENDS
# ===========================================================================
@dataclass
class LorenzCurve:
    """Cumulative population share against cumulative value share, from (0,0)."""

    population: List[float]
    share: List[float]
    ranked_by: str = "value"

    def points(self) -> List[Tuple[float, float]]:
        """The curve as ``(population_share, value_share)`` pairs."""
        return [(self.population[k], self.share[k])
                for k in range(len(self.population))]

    def area(self) -> float:
        """Area under the curve by exact trapezoidal quadrature."""
        return lorenz_area(self)

    def share_below(self, q: float) -> float:
        """Value share held by the poorest ``q`` of the population (interpolated)."""
        p = _in_unit(q, "q")
        xs, ys = self.population, self.share
        for k in range(1, len(xs)):
            if p <= xs[k]:
                span = xs[k] - xs[k - 1]
                if span == 0.0:
                    return ys[k]
                t = (p - xs[k - 1]) / span
                return ys[k - 1] + t * (ys[k] - ys[k - 1])
        return ys[-1]

    def max_gap(self) -> float:
        """The greatest vertical distance to the diagonal (the Hoover index)."""
        return max(self.population[k] - self.share[k]
                   for k in range(len(self.population)))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able curve."""
        return {"rankedBy": self.ranked_by,
                "points": [{"population": _json_safe(x), "share": _json_safe(y)}
                           for (x, y) in self.points()]}


def _ranked_shares(values: Sequence[float], weights: Optional[Sequence[float]],
                   order: Sequence[int]) -> Tuple[Vector, Vector]:
    """Cumulative population and value shares along a given ranking."""
    v = _non_negative_vector(values, "values")
    _non_empty(v, "values")
    w = _default_weights(weights, len(v), "weights")
    total_w = math.fsum(w)
    total_v = math.fsum(w[i] * v[i] for i in range(len(v)))
    pop = [0.0]
    share = [0.0]
    cum_w = 0.0
    cum_v = 0.0
    for i in order:
        cum_w += w[i]
        cum_v += w[i] * v[i]
        pop.append(cum_w / total_w)
        share.append(cum_v / total_v if total_v > 0.0 else cum_w / total_w)
    pop[-1] = 1.0
    share[-1] = 1.0
    return pop, share


def lorenz_curve(values: Sequence[float],
                 weights: Optional[Sequence[float]] = None) -> LorenzCurve:
    """The Lorenz curve of a non-negative distribution (convex, (0,0) to (1,1))."""
    order = [i for (_v, i) in _sorted_with_index(
        _non_negative_vector(values, "values"))]
    pop, share = _ranked_shares(values, weights, order)
    return LorenzCurve(pop, share, "value")


def lorenz_area(curve: LorenzCurve) -> float:
    """Exact area under a piecewise-linear Lorenz curve (trapezoidal rule)."""
    if not isinstance(curve, LorenzCurve):
        raise AccessibilityError("lorenz_area needs a LorenzCurve.")
    xs, ys = curve.population, curve.share
    return math.fsum((xs[k] - xs[k - 1]) * (ys[k] + ys[k - 1]) * 0.5
                     for k in range(1, len(xs)))


def gini_from_lorenz(curve: LorenzCurve) -> float:
    """Gini as twice the area between the diagonal and the curve."""
    return 1.0 - 2.0 * lorenz_area(curve)


def gini(values: Sequence[float],
         weights: Optional[Sequence[float]] = None) -> float:
    """Gini coefficient: 0 at perfect equality, ``(n-1)/n`` when one takes all."""
    v = _non_negative_vector(values, "values")
    _non_empty(v, "values")
    n = len(v)
    if weights is None:
        xs = sorted(v)
        total = math.fsum(xs)
        if total == 0.0:
            return 0.0
        num = math.fsum((2.0 * (k + 1) - n - 1) * xs[k] for k in range(n))
        return num / (n * total)
    w = _default_weights(weights, n, "weights")
    order = [i for (_x, i) in _sorted_with_index(v)]
    total_w = math.fsum(w)
    total_v = math.fsum(w[i] * v[i] for i in range(n))
    if total_v == 0.0:
        return 0.0
    prev_share = 0.0
    terms: Vector = []
    cum_v = 0.0
    for i in order:
        cum_v += w[i] * v[i]
        share = cum_v / total_v
        terms.append((w[i] / total_w) * (share + prev_share))
        prev_share = share
    return 1.0 - math.fsum(terms)


def theil_t(values: Sequence[float],
            weights: Optional[Sequence[float]] = None) -> float:
    """Theil's T (GE(1)): 0 at equality, ``ln(n)`` when one unit takes everything."""
    v = _non_negative_vector(values, "values")
    _non_empty(v, "values")
    w = _default_weights(weights, len(v), "weights")
    total_w = math.fsum(w)
    mu = math.fsum(w[i] * v[i] for i in range(len(v))) / total_w
    if mu == 0.0:
        return 0.0
    terms: Vector = []
    for i in range(len(v)):
        if v[i] > 0.0 and w[i] > 0.0:
            r = v[i] / mu
            terms.append((w[i] / total_w) * r * math.log(r))
    return math.fsum(terms)


def theil_l(values: Sequence[float],
            weights: Optional[Sequence[float]] = None) -> float:
    """Theil's L / mean log deviation (GE(0)); infinite if anything is zero."""
    v = _non_negative_vector(values, "values")
    _non_empty(v, "values")
    w = _default_weights(weights, len(v), "weights")
    total_w = math.fsum(w)
    mu = math.fsum(w[i] * v[i] for i in range(len(v))) / total_w
    if mu == 0.0:
        return 0.0
    for i in range(len(v)):
        if v[i] == 0.0 and w[i] > 0.0:
            return _INF
    return math.fsum((w[i] / total_w) * math.log(mu / v[i])
                     for i in range(len(v)) if w[i] > 0.0)


def generalised_entropy(values: Sequence[float], alpha: float = 1.0,
                        weights: Optional[Sequence[float]] = None) -> float:
    """The GE(alpha) family: GE(0) is Theil L, GE(1) is Theil T, GE(2) is CV^2/2."""
    a = _as_float(alpha, "alpha")
    if a == 0.0:
        return theil_l(values, weights)
    if a == 1.0:
        return theil_t(values, weights)
    v = _non_negative_vector(values, "values")
    _non_empty(v, "values")
    w = _default_weights(weights, len(v), "weights")
    total_w = math.fsum(w)
    mu = math.fsum(w[i] * v[i] for i in range(len(v))) / total_w
    if mu == 0.0:
        return 0.0
    terms: Vector = []
    for i in range(len(v)):
        if w[i] == 0.0:
            continue
        p = w[i] / total_w
        if v[i] == 0.0:
            if a < 0.0:
                return _INF
            terms.append(p * (0.0 - 1.0))
        else:
            terms.append(p * ((v[i] / mu) ** a - 1.0))
    return math.fsum(terms) / (a * (a - 1.0))


@dataclass
class TheilDecomposition:
    """An exact between/within split of a Theil index over a grouping."""

    kind: str
    total: float
    between: float
    within: float
    groups: List[Group] = field(default_factory=list)
    population_shares: List[float] = field(default_factory=list)
    value_shares: List[float] = field(default_factory=list)
    group_means: List[float] = field(default_factory=list)
    group_indices: List[float] = field(default_factory=list)

    @property
    def residual(self) -> float:
        """``total - (between + within)`` -- exactly zero up to rounding."""
        return self.total - (self.between + self.within)

    @property
    def within_share(self) -> float:
        """Fraction of the inequality that sits inside groups."""
        if self.total == 0.0:
            return 0.0
        return self.within / self.total

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able decomposition."""
        return {"kind": self.kind, "total": _json_safe(self.total),
                "between": _json_safe(self.between),
                "within": _json_safe(self.within),
                "withinShare": _json_safe(self.within_share),
                "groups": [str(g) for g in self.groups],
                "populationShares": [_json_safe(p)
                                     for p in self.population_shares],
                "valueShares": [_json_safe(s) for s in self.value_shares],
                "groupMeans": [_json_safe(m) for m in self.group_means]}


def theil_decomposition(values: Sequence[float], groups: Sequence[Group],
                        kind: str = "T",
                        weights: Optional[Sequence[float]] = None
                        ) -> TheilDecomposition:
    """Split a Theil index into an exact between-group and within-group part."""
    v = _non_negative_vector(values, "values")
    _non_empty(v, "values")
    gs = list(groups)
    _check_length(gs, len(v), "groups", "value")
    w = _default_weights(weights, len(v), "weights")
    if kind not in ("T", "L"):
        raise AccessibilityError("kind must be 'T' or 'L' (got %r)." % (kind,))
    order = _group_order(gs)
    members: Dict[Group, List[int]] = {g: [] for g in order}
    for i, g in enumerate(gs):
        members[g].append(i)
    total_w = math.fsum(w)
    total_v = math.fsum(w[i] * v[i] for i in range(len(v)))
    mu = total_v / total_w
    pop_shares: Vector = []
    val_shares: Vector = []
    means: Vector = []
    indices: Vector = []
    between_terms: Vector = []
    within_terms: Vector = []
    for g in order:
        idx = members[g]
        gw = math.fsum(w[i] for i in idx)
        gv = math.fsum(w[i] * v[i] for i in idx)
        p = gw / total_w
        s = gv / total_v if total_v > 0.0 else 0.0
        gmu = gv / gw if gw > 0.0 else 0.0
        sub = [v[i] for i in idx]
        sub_w = [w[i] for i in idx]
        gi = (theil_t(sub, sub_w) if kind == "T" else theil_l(sub, sub_w))
        pop_shares.append(p)
        val_shares.append(s)
        means.append(gmu)
        indices.append(gi)
        if kind == "T":
            if s > 0.0 and p > 0.0:
                between_terms.append(s * math.log(s / p))
                within_terms.append(s * gi)
        else:
            if p > 0.0:
                if s <= 0.0:
                    between_terms.append(_INF)
                else:
                    between_terms.append(p * math.log(p / s))
                    within_terms.append(p * gi)
    total = theil_t(v, w) if kind == "T" else theil_l(v, w)
    between = math.fsum(between_terms) if _INF not in between_terms else _INF
    within = math.fsum(within_terms) if _INF not in within_terms else _INF
    return TheilDecomposition(kind, total, between, within, order, pop_shares,
                              val_shares, means, indices)


def atkinson(values: Sequence[float], epsilon: float = 0.5,
             weights: Optional[Sequence[float]] = None) -> float:
    """Atkinson's index at inequality aversion ``epsilon``; 0 at equality, <= 1."""
    v = _non_negative_vector(values, "values")
    _non_empty(v, "values")
    eps = _non_negative(epsilon, "epsilon")
    w = _default_weights(weights, len(v), "weights")
    total_w = math.fsum(w)
    mu = math.fsum(w[i] * v[i] for i in range(len(v))) / total_w
    if mu == 0.0:
        return 0.0
    if eps == 0.0:
        return 0.0
    zeros = any(v[i] == 0.0 and w[i] > 0.0 for i in range(len(v)))
    if eps >= 1.0 and zeros:
        return 1.0
    if eps == 1.0:
        log_mean = math.fsum((w[i] / total_w) * math.log(v[i] / mu)
                             for i in range(len(v)) if w[i] > 0.0)
        return 1.0 - math.exp(log_mean)
    power = 1.0 - eps
    bracket = math.fsum((w[i] / total_w) * (v[i] / mu) ** power
                        for i in range(len(v)) if w[i] > 0.0)
    if bracket <= 0.0:
        return 1.0
    return 1.0 - bracket ** (1.0 / power)


def palma_ratio(values: Sequence[float],
                weights: Optional[Sequence[float]] = None) -> float:
    """Palma ratio: the richest tenth's share over the poorest 40 per cent's."""
    curve = lorenz_curve(values, weights)
    bottom = curve.share_below(0.4)
    top = 1.0 - curve.share_below(0.9)
    if bottom == 0.0:
        return _INF if top > 0.0 else 0.0
    return top / bottom


def quintile_ratio(values: Sequence[float],
                   weights: Optional[Sequence[float]] = None) -> float:
    """The 20:20 ratio: the top quintile's share over the bottom quintile's."""
    curve = lorenz_curve(values, weights)
    bottom = curve.share_below(0.2)
    top = 1.0 - curve.share_below(0.8)
    if bottom == 0.0:
        return _INF if top > 0.0 else 0.0
    return top / bottom


def coefficient_of_variation(values: Sequence[float],
                             weights: Optional[Sequence[float]] = None
                             ) -> float:
    """Population standard deviation over the mean -- scale-free dispersion.

    Both moments are taken in max-scaled units, so the ratio is the same at
    1e-170 and 1e+200 as it is at 1.  Computing them in raw units made the sum
    of squares underflow to exactly 0.0 on a tiny surface (reporting perfect
    equality for one that plainly varies) and overflow on a huge one.
    """
    v = _as_floats(values, "values")
    _non_empty(v, "values")
    w = _default_weights(weights, len(v), "weights")
    mu, sd, _scale = _scaled_moments(v, w)
    if mu == 0.0:
        raise AccessibilityError("the coefficient of variation needs a "
                                 "non-zero mean.")
    return sd / abs(mu)


def concentration_curve(values: Sequence[float], ranking: Sequence[float],
                        weights: Optional[Sequence[float]] = None
                        ) -> LorenzCurve:
    """Like a Lorenz curve, but the population is ranked by a NEED variable.

    LIMITATION -- ties in ``ranking`` are broken by original index (the module's
    global determinism convention), NOT by mid-rank averaging.  Zones that are
    equally needy are therefore ordered by the order they were passed in, and
    because the curve integrates ``values`` along that order the result depends
    on it.  ``concentration_index([1, 5], [2, 2])`` is ``+1/3`` while
    ``concentration_index([5, 1], [2, 2])`` is ``-1/3``, although a row
    permutation carries no information; the mid-rank convention would give 0
    for both.  Break ties in ``ranking`` yourself (e.g. by adding a
    deterministic secondary key) when the ranking variable is coarse.
    """
    v = _non_negative_vector(values, "values")
    r = _as_floats(ranking, "ranking")
    _check_length(r, len(v), "ranking", "value")
    order = [i for (_x, i) in _sorted_with_index(r)]
    pop, share = _ranked_shares(v, weights, order)
    return LorenzCurve(pop, share, "ranking")


def concentration_index(values: Sequence[float], ranking: Sequence[float],
                        weights: Optional[Sequence[float]] = None) -> float:
    """Concentration index in [-1, 1]; negative means the measure is pro-poor.

    Inherits ``concentration_curve``'s tie convention: ties in ``ranking`` are
    broken by index rather than by mid-rank, so the index is ORDER-DEPENDENT
    when the ranking variable has ties.  See that docstring for the worked case.
    """
    return 1.0 - 2.0 * lorenz_area(concentration_curve(values, ranking, weights))


def kolm_index(values: Sequence[float], alpha: float = 1.0,
               weights: Optional[Sequence[float]] = None) -> float:
    """Kolm's ABSOLUTE index: translation-invariant, not scale-invariant."""
    v = _as_floats(values, "values")
    _non_empty(v, "values")
    a = _positive(alpha, "alpha")
    w = _default_weights(weights, len(v), "weights")
    total_w = math.fsum(w)
    mu = math.fsum(w[i] * v[i] for i in range(len(v))) / total_w
    exponents = [a * (mu - v[i]) for i in range(len(v))]
    top = max(exponents)
    inner = math.fsum((w[i] / total_w) * math.exp(exponents[i] - top)
                      for i in range(len(v)))
    if inner <= 0.0:                            # pragma: no cover - unreachable
        return 0.0
    return (top + math.log(inner)) / a


def hoover_index(values: Sequence[float],
                 weights: Optional[Sequence[float]] = None) -> float:
    """Hoover / Robin Hood index: the share that must move to level everything."""
    v = _non_negative_vector(values, "values")
    _non_empty(v, "values")
    w = _default_weights(weights, len(v), "weights")
    total_w = math.fsum(w)
    mu = math.fsum(w[i] * v[i] for i in range(len(v))) / total_w
    if mu == 0.0:
        return 0.0
    return 0.5 * math.fsum((w[i] / total_w) * abs(v[i] / mu - 1.0)
                           for i in range(len(v)))


@dataclass
class InequalityProfile:
    """Every inequality index of one distribution, computed once."""

    n: int
    mean: float
    minimum: float
    maximum: float
    gini: float
    theil_t: float
    theil_l: float
    atkinson: List[Tuple[float, float]]
    palma: float
    quintile: float
    cv: float
    hoover: float
    kolm: float

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able profile."""
        return {"n": self.n, "mean": _json_safe(self.mean),
                "min": _json_safe(self.minimum),
                "max": _json_safe(self.maximum),
                "gini": _json_safe(self.gini),
                "theilT": _json_safe(self.theil_t),
                "theilL": _json_safe(self.theil_l),
                "atkinson": [{"epsilon": _json_safe(e), "value": _json_safe(a)}
                             for (e, a) in self.atkinson],
                "palma": _json_safe(self.palma),
                "quintile": _json_safe(self.quintile),
                "cv": _json_safe(self.cv), "hoover": _json_safe(self.hoover),
                "kolm": _json_safe(self.kolm)}


def inequality_profile(values: Sequence[float],
                       weights: Optional[Sequence[float]] = None,
                       epsilons: Sequence[float] = (0.5, 1.0, 2.0),
                       kolm_alpha: float = 1.0) -> InequalityProfile:
    """Compute the whole battery of indices for one accessibility surface."""
    v = _non_negative_vector(values, "values")
    _non_empty(v, "values")
    w = _default_weights(weights, len(v), "weights")
    eps = [_non_negative(e, "epsilons") for e in epsilons]
    mu = _mean(v, w)
    return InequalityProfile(
        len(v), mu, min(v), max(v), gini(v, w), theil_t(v, w), theil_l(v, w),
        [(e, atkinson(v, e, w)) for e in eps], palma_ratio(v, w),
        quintile_ratio(v, w),
        coefficient_of_variation(v, w) if mu != 0.0 else 0.0,
        hoover_index(v, w), kolm_index(v, kolm_alpha, w))


def pigou_dalton_transfer(values: Sequence[float], richer: int, poorer: int,
                          amount: float) -> Vector:
    """Move ``amount`` from a richer unit to a poorer one, preserving their order."""
    v = _as_floats(values, "values")
    _non_empty(v, "values")
    if not (0 <= richer < len(v)) or not (0 <= poorer < len(v)):
        raise AccessibilityError("transfer indices must be inside the vector.")
    if richer == poorer:
        raise AccessibilityError("a transfer needs two different units.")
    gap = v[richer] - v[poorer]
    if gap <= 0.0:
        raise AccessibilityError("values[%d] must exceed values[%d] to transfer."
                                 % (richer, poorer))
    a = _positive(amount, "amount")
    if a > 0.5 * gap:
        raise AccessibilityError(
            "a rank-preserving transfer is at most half the gap (%.6g > %.6g)."
            % (a, 0.5 * gap))
    out = list(v)
    out[richer] -= a
    out[poorer] += a
    return out


# ===========================================================================
# 9. HORIZONTAL VERSUS VERTICAL EQUITY
# ===========================================================================
@dataclass
class EquityDecomposition:
    """Inequality split into 'unlike treated differently' and 'like treated alike'."""

    kind: str
    total: float
    between: float
    within: float
    horizontal_index: float
    vertical_index: Optional[float]
    groups: List[Group] = field(default_factory=list)
    group_means: List[float] = field(default_factory=list)
    group_populations: List[float] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """A one-word reading of the vertical index."""
        if self.vertical_index is None:
            return "unassessed"
        if self.vertical_index > 0.01:
            return "progressive"
        if self.vertical_index < -0.01:
            return "regressive"
        return "neutral"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able decomposition."""
        return {"kind": self.kind, "total": _json_safe(self.total),
                "between": _json_safe(self.between),
                "within": _json_safe(self.within),
                "horizontalIndex": _json_safe(self.horizontal_index),
                "verticalIndex": _json_safe(self.vertical_index),
                "verdict": self.verdict,
                "groups": [str(g) for g in self.groups],
                "groupMeans": [_json_safe(m) for m in self.group_means]}


def horizontal_vertical_equity(access: Sequence[float],
                               groups: Sequence[Group],
                               need: Optional[Sequence[float]] = None,
                               weights: Optional[Sequence[float]] = None,
                               kind: str = "T") -> EquityDecomposition:
    """Read the Theil split as horizontal (within) and vertical (between) equity."""
    dec = theil_decomposition(access, groups, kind, weights)
    if dec.total == 0.0:
        horizontal = 1.0
    elif dec.total == _INF or dec.within == _INF:
        horizontal = 0.0
    else:
        horizontal = 1.0 - dec.within / dec.total
    vertical = None if need is None else vertical_equity(access, need, weights)
    total_w = math.fsum(_default_weights(weights, len(list(access)), "weights"))
    pops = [p * total_w for p in dec.population_shares]
    return EquityDecomposition(kind, dec.total, dec.between, dec.within,
                               horizontal, vertical, dec.groups,
                               dec.group_means, pops)


def horizontal_equity(access: Sequence[float], groups: Sequence[Group],
                      weights: Optional[Sequence[float]] = None,
                      kind: str = "T") -> float:
    """1 when people in the same group all get the same accessibility, else < 1."""
    return horizontal_vertical_equity(access, groups, None, weights,
                                      kind).horizontal_index


def vertical_equity(access: Sequence[float], need: Sequence[float],
                    weights: Optional[Sequence[float]] = None) -> float:
    """Progressivity: > 0 when accessibility is concentrated on the neediest."""
    return concentration_index(access, need, weights)


# ===========================================================================
# 10. NEED-WEIGHTED ACCESSIBILITY AND GROUP GAPS
# ===========================================================================
def population_weighted_mean(values: Sequence[float],
                             population: Sequence[float]) -> float:
    """The mean accessibility experienced by a person, not by a zone."""
    v = _as_floats(values, "values")
    _non_empty(v, "values")
    p = _default_weights(population, len(v), "population")
    return _mean(v, p)


def equity_weights(population: Sequence[float], deprivation: Sequence[float],
                   elasticity: float = 1.0) -> Vector:
    """Population weights tilted towards deprived places; they sum to exactly 1."""
    p = _non_negative_vector(population, "population")
    d = _non_negative_vector(deprivation, "deprivation")
    _check_length(d, len(p), "deprivation", "zone")
    _non_empty(p, "population")
    e = _non_negative(elasticity, "elasticity")
    raw = [p[i] * (1.0 if e == 0.0 else (d[i] ** e if d[i] > 0.0 else 0.0))
           for i in range(len(p))]
    total = math.fsum(raw)
    if total <= 0.0:
        raise AccessibilityError("the equity weights sum to zero.")
    return [r / total for r in raw]


def need_weighted_accessibility(access: Sequence[float], need: Sequence[float],
                                population: Optional[Sequence[float]] = None
                                ) -> float:
    """Mean accessibility with each place weighted by its population times need."""
    a = _as_floats(access, "access")
    _non_empty(a, "access")
    n = _non_negative_vector(need, "need")
    _check_length(n, len(a), "need", "zone")
    p = _default_weights(population, len(a), "population")
    w = [p[i] * n[i] for i in range(len(a))]
    if math.fsum(w) <= 0.0:
        raise AccessibilityError("the need weights sum to zero.")
    return _mean(a, w)


def deprivation_weighted_accessibility(access: Sequence[float],
                                       population: Sequence[float],
                                       deprivation: Sequence[float],
                                       elasticity: float = 1.0) -> float:
    """Equity-weighted mean accessibility; at elasticity 0 it is the plain mean."""
    a = _as_floats(access, "access")
    _non_empty(a, "access")
    w = equity_weights(population, deprivation, elasticity)
    _check_length(w, len(a), "population", "zone")
    return math.fsum(w[i] * a[i] for i in range(len(a)))


@dataclass
class GroupGapResult:
    """Population-weighted accessibility by group, and the gap between them."""

    groups: List[Group]
    means: List[float]
    populations: List[float]
    overall: float
    best: Group = None
    worst: Group = None
    absolute_gap: float = 0.0
    ratio_gap: float = 1.0

    def as_dict(self) -> Dict[str, float]:
        """Group label (as a string) to mean accessibility."""
        return {str(self.groups[k]): self.means[k]
                for k in range(len(self.groups))}

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able summary."""
        return {"groups": [str(g) for g in self.groups],
                "means": [_json_safe(m) for m in self.means],
                "populations": [_json_safe(p) for p in self.populations],
                "overall": _json_safe(self.overall),
                "best": None if self.best is None else str(self.best),
                "worst": None if self.worst is None else str(self.worst),
                "absoluteGap": _json_safe(self.absolute_gap),
                "ratioGap": _json_safe(self.ratio_gap)}


def accessibility_gap(access: Sequence[float], groups: Sequence[Group],
                      population: Optional[Sequence[float]] = None
                      ) -> GroupGapResult:
    """Compare population-weighted accessibility between population groups."""
    a = _as_floats(access, "access")
    _non_empty(a, "access")
    gs = list(groups)
    _check_length(gs, len(a), "groups", "zone")
    pop = _default_weights(population, len(a), "population")
    order = _group_order(gs)
    means: Vector = []
    pops: Vector = []
    for g in order:
        idx = [i for i in range(len(a)) if gs[i] == g]
        gp = math.fsum(pop[i] for i in idx)
        pops.append(gp)
        means.append(math.fsum(pop[i] * a[i] for i in idx) / gp
                     if gp > 0.0 else 0.0)
    overall = _mean(a, pop)
    ranked = sorted(range(len(order)), key=lambda k: (means[k], k))
    worst_k, best_k = ranked[0], ranked[-1]
    gap = means[best_k] - means[worst_k]
    ratio = (means[best_k] / means[worst_k]) if means[worst_k] > 0.0 else _INF
    if means[best_k] == means[worst_k]:
        ratio = 1.0
    return GroupGapResult(order, means, pops, overall, order[best_k],
                          order[worst_k], gap, ratio)


# ===========================================================================
# 11. TRANSPORT POVERTY
# ===========================================================================
def _weighted_median(values: Sequence[float], weights: Sequence[float]
                     ) -> float:
    """Weighted median; reduces to the classic median when weights are equal."""
    order = _sorted_with_index(values)
    total = math.fsum(weights)
    half = 0.5 * total
    cum = 0.0
    for k, (v, i) in enumerate(order):
        cum += weights[i]
        if cum > half:
            return v
        if cum == half:
            if k + 1 < len(order):
                return 0.5 * (v + order[k + 1][0])
            return v
    return order[-1][0]                         # pragma: no cover


def accessibility_poverty_line(access: Sequence[float], fraction: float = 0.5,
                               statistic: str = "median",
                               weights: Optional[Sequence[float]] = None
                               ) -> float:
    """A relative poverty line: a fraction of the median (or mean) accessibility."""
    a = _as_floats(access, "access")
    _non_empty(a, "access")
    f = _positive(fraction, "fraction")
    w = _default_weights(weights, len(a), "weights")
    if statistic == "median":
        centre = _weighted_median(a, w)
    elif statistic == "mean":
        centre = _mean(a, w)
    else:
        raise AccessibilityError("statistic must be 'median' or 'mean' "
                                 "(got %r)." % (statistic,))
    return f * centre


def fgt_index(values: Sequence[float], line: float, alpha: float = 0.0,
              weights: Optional[Sequence[float]] = None) -> float:
    """Foster-Greer-Thorbecke index: headcount (0), gap (1) or severity (2)."""
    v = _as_floats(values, "values")
    _non_empty(v, "values")
    z = _positive(line, "line")
    a = _non_negative(alpha, "alpha")
    w = _default_weights(weights, len(v), "weights")
    total = math.fsum(w)
    terms: Vector = []
    for i in range(len(v)):
        if v[i] < z:
            gap = (z - v[i]) / z
            terms.append(w[i] * (1.0 if a == 0.0 else gap ** a))
    return math.fsum(terms) / total


@dataclass
class TransportPovertyResult:
    """Who falls below an accessibility poverty line, and by how far."""

    line: float
    headcount: float
    headcount_ratio: float
    gap_index: float
    severity_index: float
    mean_shortfall: float
    poor: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able summary."""
        return {"line": _json_safe(self.line),
                "headcount": _json_safe(self.headcount),
                "headcountRatio": _json_safe(self.headcount_ratio),
                "gapIndex": _json_safe(self.gap_index),
                "severityIndex": _json_safe(self.severity_index),
                "meanShortfall": _json_safe(self.mean_shortfall),
                "poor": list(self.poor)}


def transport_poverty_indicators(access: Sequence[float],
                                 line: Optional[float] = None,
                                 population: Optional[Sequence[float]] = None,
                                 fraction: float = 0.5
                                 ) -> TransportPovertyResult:
    """The FGT battery against an accessibility poverty line."""
    a = _as_floats(access, "access")
    _non_empty(a, "access")
    pop = _default_weights(population, len(a), "population")
    z = (accessibility_poverty_line(a, fraction, "median", pop)
         if line is None else _positive(line, "line"))
    poor = [i for i in range(len(a)) if a[i] < z]
    head = math.fsum(pop[i] for i in poor)
    total = math.fsum(pop)
    shortfall = (math.fsum(pop[i] * (z - a[i]) for i in poor) / head
                 if head > 0.0 else 0.0)
    return TransportPovertyResult(z, head, head / total,
                                  fgt_index(a, z, 1.0, pop),
                                  fgt_index(a, z, 2.0, pop), shortfall, poor)


def transport_poverty_index(access: Sequence[float], poverty_line: float,
                            car_availability: Sequence[float],
                            affordability: Sequence[float],
                            affordability_threshold: float = 0.1,
                            weights: Sequence[float] = (0.4, 0.3, 0.3)
                            ) -> Vector:
    """A composite 0-1 index over accessibility deficit, car deprivation and cost."""
    a = _as_floats(access, "access")
    _non_empty(a, "access")
    z = _positive(poverty_line, "poverty_line")
    car = [_in_unit(x, "car_availability") for x in car_availability]
    aff = [_non_negative(x, "affordability") for x in affordability]
    _check_length(car, len(a), "car_availability", "zone")
    _check_length(aff, len(a), "affordability", "zone")
    thr = _positive(affordability_threshold, "affordability_threshold")
    ws = _non_negative_vector(weights, "weights")
    if len(ws) != 3:
        raise AccessibilityError("weights must have three entries "
                                 "(accessibility, car, affordability).")
    total = math.fsum(ws)
    if total <= 0.0:
        raise AccessibilityError("weights must not sum to zero.")
    ws = [w / total for w in ws]
    out: Vector = []
    for i in range(len(a)):
        deficit = max(0.0, (z - a[i]) / z)
        deficit = min(1.0, deficit)
        car_dep = 1.0 - car[i]
        burden = min(1.0, aff[i] / thr)
        out.append(ws[0] * deficit + ws[1] * car_dep + ws[2] * burden)
    return out


# ===========================================================================
# 12. SPATIAL AUTOCORRELATION OF THE ACCESSIBILITY SURFACE
# ===========================================================================
def _distance_matrix(coords: Sequence[Coord], metric: str = "euclidean"
                     ) -> Matrix:
    """All-pairs distances between coordinates, planar or great-circle."""
    pts = [(_as_float(p[0], "coord"), _as_float(p[1], "coord")) for p in coords]
    _non_empty(pts, "coords")
    if metric == "euclidean":
        dist = euclidean_m
    elif metric == "haversine":
        dist = haversine_m
    else:
        raise AccessibilityError("metric must be 'euclidean' or 'haversine' "
                                 "(got %r)." % (metric,))
    return [[dist(a, b) for b in pts] for a in pts]


def row_standardise(weights: Any) -> Matrix:
    """Scale each row to sum to 1; all-zero rows are left alone (islands)."""
    rows = list(weights)
    _non_empty(rows, "weights")
    w = _as_weight_matrix(rows, len(rows), len(list(rows[0])))
    out: Matrix = []
    for row in w:
        total = math.fsum(row)
        out.append([v / total for v in row] if total > 0.0 else list(row))
    return out


def spatial_weights_knn(coords: Sequence[Coord], k: int = 4,
                        metric: str = "euclidean",
                        row_standardised: bool = True) -> Matrix:
    """Binary (or row-standardised) k-nearest-neighbour weights, ties by index."""
    d = _distance_matrix(coords, metric)
    n = len(d)
    kk = _positive_int(k, "k")
    if kk >= n:
        raise AccessibilityError("k must be < the number of points (%d)." % n)
    w = [[0.0] * n for _ in range(n)]
    for i in range(n):
        order = sorted(((d[i][j], j) for j in range(n) if j != i),
                       key=lambda p: (p[0], p[1]))
        for (_dist, j) in order[:kk]:
            w[i][j] = 1.0
    return row_standardise(w) if row_standardised else w


def spatial_weights_distance_band(coords: Sequence[Coord], threshold: float,
                                  metric: str = "euclidean",
                                  row_standardised: bool = False) -> Matrix:
    """Binary contiguity: neighbours are the points within a distance band."""
    d = _distance_matrix(coords, metric)
    n = len(d)
    t = _positive(threshold, "threshold")
    w = [[1.0 if (i != j and d[i][j] <= t) else 0.0 for j in range(n)]
         for i in range(n)]
    return row_standardise(w) if row_standardised else w


def spatial_weights_inverse_distance(coords: Sequence[Coord],
                                     power: float = 1.0,
                                     cutoff: Optional[float] = None,
                                     metric: str = "euclidean",
                                     row_standardised: bool = False) -> Matrix:
    """Inverse-distance weights ``d ** -power``, optionally cut off at a radius."""
    d = _distance_matrix(coords, metric)
    n = len(d)
    p = _non_negative(power, "power")
    cut = _INF if cutoff is None else _positive(cutoff, "cutoff")
    w = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j or d[i][j] > cut:
                continue
            if d[i][j] == 0.0:
                raise AccessibilityError(
                    "points %d and %d coincide; inverse distance is undefined."
                    % (i, j))
            w[i][j] = d[i][j] ** (-p)
    return row_standardise(w) if row_standardised else w


def _moran_parts(values: Sequence[float], weights: Any
                 ) -> Tuple[Vector, Matrix, float, float]:
    """Deviations, validated weights, ``S0`` and the sum of squared deviations."""
    v = _as_floats(values, "values")
    _non_empty(v, "values")
    n = len(v)
    w = _as_weight_matrix(weights, n, n)
    for i in range(n):
        if w[i][i] != 0.0:
            raise AccessibilityError("spatial weights must have a zero diagonal.")
    s0 = math.fsum(math.fsum(row) for row in w)
    if s0 <= 0.0:
        raise AccessibilityError("the spatial weights are all zero.")
    # Every statistic built from these parts is a RATIO of quadratic forms in
    # ``z`` (Moran's cross/m2, Geary's num/m2, the LISA's z_i w z / (m2/n)), so
    # dividing the deviations by the peak magnitude leaves all of them exactly
    # unchanged while keeping ``m2`` inside double range.  In raw units a
    # surface around 1e-170 had m2 underflow to exactly 0.0 -- and the old
    # scale-relative guard ``(_EPS * spread) ** 2`` underflowed to 0.0 FIRST,
    # so it degenerated to ``m2 <= 0`` and every statistic wrongly reported a
    # constant surface; a surface around 1e+160 leaked ValueError('-inf + inf
    # in fsum') from Moran and OverflowError from Geary.  Working scaled makes
    # the guard a bare ``n * _EPS ** 2``, which cannot itself underflow.
    mu, _sd, scale = _scaled_moments(v)
    z = [x / scale - mu for x in v]
    m2 = math.fsum(x * x for x in z)
    if m2 <= float(n) * _EPS ** 2:
        raise AccessibilityError("a constant surface has no autocorrelation.")
    return z, w, s0, m2


def morans_i(values: Sequence[float], weights: Any) -> float:
    """Global Moran's I: spatial autocorrelation of an accessibility surface."""
    z, w, s0, m2 = _moran_parts(values, weights)
    n = len(z)
    cross = math.fsum(math.fsum(w[i][j] * z[i] * z[j] for j in range(n))
                      for i in range(n))
    return (n / s0) * (cross / m2)


def morans_i_expectation(n: int) -> float:
    """The exact permutation expectation of Moran's I: ``-1 / (n - 1)``."""
    m = _positive_int(n, "n")
    if m < 2:
        raise AccessibilityError("Moran's I needs at least two observations.")
    return -1.0 / (m - 1)


def gearys_c(values: Sequence[float], weights: Any) -> float:
    """Geary's C: 1 under no autocorrelation, < 1 clustered, > 1 dispersed."""
    z, w, s0, m2 = _moran_parts(values, weights)
    n = len(z)
    num = math.fsum(math.fsum(w[i][j] * (z[i] - z[j]) ** 2 for j in range(n))
                    for i in range(n))
    return ((n - 1) * num) / (2.0 * s0 * m2)


def local_morans_i(values: Sequence[float], weights: Any) -> Vector:
    """Anselin's LISA; the local values sum to exactly ``S0`` times global I."""
    z, w, s0, m2 = _moran_parts(values, weights)
    n = len(z)
    scale = m2 / n
    return [z[i] * math.fsum(w[i][j] * z[j] for j in range(n)) / scale
            for i in range(n)]


@dataclass
class MoranResult:
    """A Moran's I test: the statistic, its null and a permutation p-value."""

    i: float
    expectation: float
    permutations: int
    exhaustive: bool
    permuted_mean: float
    permuted_sd: float
    z_score: float
    p_value: float

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able summary."""
        return {"I": _json_safe(self.i),
                "expectation": _json_safe(self.expectation),
                "permutations": self.permutations,
                "exhaustive": self.exhaustive,
                "permutedMean": _json_safe(self.permuted_mean),
                "permutedSd": _json_safe(self.permuted_sd),
                "zScore": _json_safe(self.z_score),
                "pValue": _json_safe(self.p_value)}


def morans_i_permutation_test(values: Sequence[float], weights: Any,
                              permutations: int = 999, seed: int = 0,
                              exhaustive: bool = False) -> MoranResult:
    """Permutation test for Moran's I -- sampled, or EXHAUSTIVE for small n."""
    v = _as_floats(values, "values")
    _non_empty(v, "values")
    n = len(v)
    observed = morans_i(v, weights)
    expectation = morans_i_expectation(n)
    draws: Vector = []
    if exhaustive:
        if n > 8:
            raise AccessibilityError(
                "an exhaustive test enumerates n! orderings; n = %d is too "
                "large (limit 8)." % n)
        for perm in itertools.permutations(range(n)):
            draws.append(morans_i([v[p] for p in perm], weights))
    else:
        m = _positive_int(permutations, "permutations")
        rng = random.Random(seed)
        idx = list(range(n))
        for _ in range(m):
            rng.shuffle(idx)
            draws.append(morans_i([v[p] for p in idx], weights))
    mean = _mean(draws)
    var = math.fsum((d - mean) ** 2 for d in draws) / float(len(draws))
    sd = math.sqrt(var)
    z = 0.0 if sd == 0.0 else (observed - mean) / sd
    extreme = sum(1 for d in draws if abs(d - mean) >= abs(observed - mean))
    p = (extreme / float(len(draws)) if exhaustive
         else (extreme + 1) / float(len(draws) + 1))
    return MoranResult(observed, expectation, len(draws), bool(exhaustive),
                       mean, sd, z, p)


# ===========================================================================
# 13. IMPACT INCIDENCE -- WHO GAINS AND WHO LOSES
# ===========================================================================
@dataclass
class IncidenceResult:
    """The distributional incidence of a scheme, by population group."""

    groups: List[Group]
    populations: List[float]
    mean_before: List[float]
    mean_after: List[float]
    mean_change: List[float]
    relative_change: List[float]
    share_gaining: float
    share_losing: float
    total_gain: float
    total_loss: float
    net_change: float
    gini_before: float
    gini_after: float
    bottom_half_change: float
    top_half_change: float

    @property
    def progressive(self) -> bool:
        """True when the worse-off half gains at least as much as the better-off."""
        return self.bottom_half_change >= self.top_half_change

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able summary."""
        return {"groups": [str(g) for g in self.groups],
                "populations": [_json_safe(p) for p in self.populations],
                "meanBefore": [_json_safe(v) for v in self.mean_before],
                "meanAfter": [_json_safe(v) for v in self.mean_after],
                "meanChange": [_json_safe(v) for v in self.mean_change],
                "relativeChange": [_json_safe(v) for v in self.relative_change],
                "shareGaining": _json_safe(self.share_gaining),
                "shareLosing": _json_safe(self.share_losing),
                "totalGain": _json_safe(self.total_gain),
                "totalLoss": _json_safe(self.total_loss),
                "netChange": _json_safe(self.net_change),
                "giniBefore": _json_safe(self.gini_before),
                "giniAfter": _json_safe(self.gini_after),
                "bottomHalfChange": _json_safe(self.bottom_half_change),
                "topHalfChange": _json_safe(self.top_half_change),
                "progressive": self.progressive}


def _half_split_change(before: Vector, after: Vector, population: Vector
                       ) -> Tuple[float, float]:
    """Mean change for the worse-off and better-off halves, split at the median."""
    median = _weighted_median(before, population)
    bottom = [i for i in range(len(before)) if before[i] <= median]
    top = [i for i in range(len(before)) if before[i] > median]
    if not top:                                 # every zone at or below the median
        top = [i for i in range(len(before)) if before[i] == median]
    change = [after[i] - before[i] for i in range(len(before))]

    def weighted(idx: Sequence[int]) -> float:
        total = math.fsum(population[i] for i in idx)
        if total <= 0.0:
            return 0.0
        return math.fsum(population[i] * change[i] for i in idx) / total

    return weighted(bottom), weighted(top)


def impact_incidence(before: Sequence[float], after: Sequence[float],
                     groups: Optional[Sequence[Group]] = None,
                     population: Optional[Sequence[float]] = None,
                     tolerance: float = 0.0) -> IncidenceResult:
    """Compare two accessibility surfaces and report who gains and who loses."""
    b = _as_floats(before, "before")
    a = _as_floats(after, "after")
    _non_empty(b, "before")
    _check_length(a, len(b), "after", "zone")
    pop = _default_weights(population, len(b), "population")
    tol = _non_negative(tolerance, "tolerance")
    gs = ["all"] * len(b) if groups is None else list(groups)
    _check_length(gs, len(b), "groups", "zone")
    order = _group_order(gs)
    means_b: Vector = []
    means_a: Vector = []
    means_c: Vector = []
    rel: Vector = []
    pops: Vector = []
    for g in order:
        idx = [i for i in range(len(b)) if gs[i] == g]
        gp = math.fsum(pop[i] for i in idx)
        pops.append(gp)
        mb = math.fsum(pop[i] * b[i] for i in idx) / gp if gp > 0.0 else 0.0
        ma = math.fsum(pop[i] * a[i] for i in idx) / gp if gp > 0.0 else 0.0
        means_b.append(mb)
        means_a.append(ma)
        means_c.append(ma - mb)
        rel.append((ma - mb) / mb if mb != 0.0 else _INF if ma > mb else 0.0)
    total_pop = math.fsum(pop)
    gaining = math.fsum(pop[i] for i in range(len(b)) if a[i] - b[i] > tol)
    losing = math.fsum(pop[i] for i in range(len(b)) if b[i] - a[i] > tol)
    gain = math.fsum(pop[i] * (a[i] - b[i]) for i in range(len(b))
                     if a[i] - b[i] > tol)
    loss = math.fsum(pop[i] * (b[i] - a[i]) for i in range(len(b))
                     if b[i] - a[i] > tol)
    bottom, top = _half_split_change(b, a, pop)
    return IncidenceResult(
        order, pops, means_b, means_a, means_c, rel,
        gaining / total_pop, losing / total_pop, gain, loss,
        math.fsum(pop[i] * (a[i] - b[i]) for i in range(len(b))),
        gini(b, pop) if min(b) >= 0.0 else float("nan"),
        gini(a, pop) if min(a) >= 0.0 else float("nan"),
        bottom, top)


def winners_losers(before: Sequence[float], after: Sequence[float],
                   population: Optional[Sequence[float]] = None,
                   tolerance: float = 0.0) -> Dict[str, Any]:
    """Zone indices and population shares that gain, lose or stay put."""
    b = _as_floats(before, "before")
    a = _as_floats(after, "after")
    _non_empty(b, "before")
    _check_length(a, len(b), "after", "zone")
    pop = _default_weights(population, len(b), "population")
    tol = _non_negative(tolerance, "tolerance")
    winners = [i for i in range(len(b)) if a[i] - b[i] > tol]
    losers = [i for i in range(len(b)) if b[i] - a[i] > tol]
    unchanged = [i for i in range(len(b)) if i not in set(winners + losers)]
    total = math.fsum(pop)
    return {"winners": winners, "losers": losers, "unchanged": unchanged,
            "winnerShare": math.fsum(pop[i] for i in winners) / total,
            "loserShare": math.fsum(pop[i] for i in losers) / total,
            "unchangedShare": math.fsum(pop[i] for i in unchanged) / total,
            "biggestWinner": (max(winners, key=lambda i: (a[i] - b[i], -i))
                              if winners else None),
            "biggestLoser": (max(losers, key=lambda i: (b[i] - a[i], -i))
                             if losers else None)}


def distributional_summary(values: Sequence[float],
                           weights: Optional[Sequence[float]] = None,
                           quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75,
                                                         0.9)) -> Dict[str, Any]:
    """Mean, extremes, quantile shares and the headline indices in one dict."""
    v = _non_negative_vector(values, "values")
    _non_empty(v, "values")
    w = _default_weights(weights, len(v), "weights")
    qs = [_in_unit(q, "quantiles") for q in quantiles]
    curve = lorenz_curve(v, w)
    return {"n": len(v), "mean": _mean(v, w), "min": min(v), "max": max(v),
            "median": _weighted_median(v, w),
            "shares": [{"q": q, "shareBelow": curve.share_below(q)}
                       for q in qs],
            "gini": gini(v, w), "theilT": theil_t(v, w),
            "hoover": hoover_index(v, w), "palma": palma_ratio(v, w)}


# ===========================================================================
# 14. RAWLSIAN PRIORITY -- MAX-MIN AND LEXIMIN
# ===========================================================================
def rawlsian_index(values: Sequence[float],
                   weights: Optional[Sequence[float]] = None) -> float:
    """The Rawlsian verdict: the accessibility of the worst-off unit."""
    v = _as_floats(values, "values")
    _non_empty(v, "values")
    if weights is None:
        return min(v)
    w = _default_weights(weights, len(v), "weights")
    live = [v[i] for i in range(len(v)) if w[i] > 0.0]
    _non_empty(live, "values with positive weight")
    return min(live)


def leximin_key(values: Sequence[float]) -> Tuple[float, ...]:
    """The leximin comparison key: the outcomes sorted worst-first."""
    v = _as_floats(values, "values")
    _non_empty(v, "values")
    return tuple(sorted(v))


def leximin_compare(a: Sequence[float], b: Sequence[float]) -> int:
    """+1 if ``a`` is leximin-better than ``b``, -1 if worse, 0 if equivalent."""
    ka, kb = leximin_key(a), leximin_key(b)
    if len(ka) != len(kb):
        raise AccessibilityError("leximin compares vectors of equal length "
                                 "(%d != %d)." % (len(ka), len(kb)))
    for x, y in zip(ka, kb):
        if x > y:
            return 1
        if x < y:
            return -1
    return 0


def leximin_sort(options: Sequence[Tuple[str, Sequence[float]]]) -> List[str]:
    """Option names ordered best-first by leximin, ties broken by name."""
    entries = list(options)
    _non_empty(entries, "options")
    keyed: List[Tuple[str, Tuple[float, ...]]] = []
    for e in entries:
        if len(e) != 2:
            raise AccessibilityError("each option must be (name, values).")
        keyed.append((str(e[0]), leximin_key(e[1])))
    lengths = set(len(k) for (_n, k) in keyed)
    if len(lengths) != 1:
        raise AccessibilityError("every option must cover the same units.")
    return [name for (name, _k) in
            sorted(keyed, key=lambda p: (tuple(-x for x in p[1]), p[0]))]


@dataclass
class MaximinResult:
    """Max-min and leximin rankings of competing designs."""

    names: List[str]
    minima: List[float]
    means: List[float]
    best: str
    maximin_ranking: List[str]
    leximin_ranking: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able summary."""
        return {"names": list(self.names),
                "minima": [_json_safe(m) for m in self.minima],
                "means": [_json_safe(m) for m in self.means],
                "best": self.best,
                "maximinRanking": list(self.maximin_ranking),
                "leximinRanking": list(self.leximin_ranking)}


def rawlsian_maxmin(options: Sequence[Tuple[str, Sequence[float]]]
                    ) -> MaximinResult:
    """Rank designs by the worst-off unit, then refine the ties by leximin."""
    entries = list(options)
    _non_empty(entries, "options")
    names: List[str] = []
    minima: Vector = []
    means: Vector = []
    for e in entries:
        if len(e) != 2:
            raise AccessibilityError("each option must be (name, values).")
        v = _as_floats(e[1], "values")
        _non_empty(v, "values")
        names.append(str(e[0]))
        minima.append(min(v))
        means.append(_mean(v))
    maximin = [names[k] for k in
               sorted(range(len(names)), key=lambda k: (-minima[k], names[k]))]
    leximin = leximin_sort(entries)
    return MaximinResult(names, minima, means, leximin[0], maximin, leximin)


# ===========================================================================
# 15. DISRUPTION -- ACCESSIBILITY UNDER FAILURE
# ===========================================================================
def disruption_scenario(cost: Any, opportunities: Sequence[float], decay: Any,
                        closed_destinations: Sequence[int] = (),
                        cost_multiplier: float = 1.0,
                        cost_penalty: float = 0.0,
                        blocked_pairs: Sequence[Tuple[int, int]] = ()
                        ) -> Vector:
    """Accessibility after closures, detours and delays -- the 'after' surface."""
    c = _as_cost_matrix(cost)
    o = _non_negative_vector(opportunities, "opportunities")
    _check_length(o, len(c[0]), "opportunities", "destination")
    mult = _positive(cost_multiplier, "cost_multiplier")
    pen = _non_negative(cost_penalty, "cost_penalty")
    closed = set()
    for j in closed_destinations:
        jj = int(j)
        if not (0 <= jj < len(o)):
            raise AccessibilityError("closed destination %d is out of range."
                                     % jj)
        closed.add(jj)
    blocked = set()
    for pair in blocked_pairs:
        if len(pair) != 2:
            raise AccessibilityError("each blocked pair must be (origin, dest).")
        i, j = int(pair[0]), int(pair[1])
        if not (0 <= i < len(c)) or not (0 <= j < len(o)):
            raise AccessibilityError("blocked pair (%d, %d) is out of range."
                                     % (i, j))
        blocked.add((i, j))
    supply = [0.0 if j in closed else o[j] for j in range(len(o))]
    scenario: Matrix = []
    for i in range(len(c)):
        row: Vector = []
        for j in range(len(o)):
            if (i, j) in blocked or c[i][j] == _INF:
                row.append(_INF)
            else:
                row.append(c[i][j] * mult + pen)
        scenario.append(row)
    return gravity_accessibility(scenario, supply, decay)


@dataclass
class DisruptionResult:
    """The size, spread and fairness of an accessibility loss."""

    loss: List[float]
    relative_loss: List[float]
    mean_loss: float
    weighted_mean_loss: float
    max_loss: float
    worst_zone: int
    resilience: float
    loss_gini: float
    baseline_gini: float
    disrupted_gini: float
    share_losing: float

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able summary."""
        return {"meanLoss": _json_safe(self.mean_loss),
                "weightedMeanLoss": _json_safe(self.weighted_mean_loss),
                "maxLoss": _json_safe(self.max_loss),
                "worstZone": self.worst_zone,
                "resilience": _json_safe(self.resilience),
                "lossGini": _json_safe(self.loss_gini),
                "baselineGini": _json_safe(self.baseline_gini),
                "disruptedGini": _json_safe(self.disrupted_gini),
                "shareLosing": _json_safe(self.share_losing),
                "loss": [_json_safe(v) for v in self.loss],
                "relativeLoss": [_json_safe(v) for v in self.relative_loss]}


def accessibility_loss(baseline: Sequence[float], disrupted: Sequence[float],
                       population: Optional[Sequence[float]] = None
                       ) -> DisruptionResult:
    """Quantify what a disruption costs, and whether the pain is evenly spread."""
    b = _non_negative_vector(baseline, "baseline")
    d = _non_negative_vector(disrupted, "disrupted")
    _non_empty(b, "baseline")
    _check_length(d, len(b), "disrupted", "zone")
    pop = _default_weights(population, len(b), "population")
    loss = [b[i] - d[i] for i in range(len(b))]
    rel = [(loss[i] / b[i]) if b[i] > 0.0 else 0.0 for i in range(len(b))]
    positive = [max(0.0, x) for x in loss]
    worst = max(range(len(b)), key=lambda i: (loss[i], -i))
    total_pop = math.fsum(pop)
    return DisruptionResult(
        loss, rel, _mean(loss), _mean(loss, pop), loss[worst], worst,
        resilience_index(b, d, pop), gini(positive), gini(b, pop),
        gini(d, pop),
        math.fsum(pop[i] for i in range(len(b)) if loss[i] > 0.0) / total_pop)


def vulnerability_ranking(cost: Any, opportunities: Sequence[float],
                          decay: Any,
                          population: Optional[Sequence[float]] = None,
                          candidates: Optional[Sequence[int]] = None
                          ) -> List[Tuple[int, float]]:
    """Close each destination in turn; rank them by the accessibility they carry."""
    c = _as_cost_matrix(cost)
    o = _non_negative_vector(opportunities, "opportunities")
    _check_length(o, len(c[0]), "opportunities", "destination")
    pop = _default_weights(population, len(c), "population")
    base = gravity_accessibility(c, o, decay)
    base_mean = _mean(base, pop)
    idx = list(range(len(o))) if candidates is None else [int(j) for j
                                                          in candidates]
    for j in idx:
        if not (0 <= j < len(o)):
            raise AccessibilityError("candidate %d is out of range." % j)
    scored: List[Tuple[int, float]] = []
    for j in idx:
        after = disruption_scenario(c, o, decay, closed_destinations=(j,))
        scored.append((j, base_mean - _mean(after, pop)))
    return sorted(scored, key=lambda p: (-p[1], p[0]))


def resilience_index(baseline: Sequence[float], disrupted: Sequence[float],
                     weights: Optional[Sequence[float]] = None) -> float:
    """Share of the weighted accessibility that survives a disruption."""
    b = _non_negative_vector(baseline, "baseline")
    d = _non_negative_vector(disrupted, "disrupted")
    _non_empty(b, "baseline")
    _check_length(d, len(b), "disrupted", "zone")
    w = _default_weights(weights, len(b), "weights")
    total = math.fsum(w[i] * b[i] for i in range(len(b)))
    if total == 0.0:
        return 1.0
    return math.fsum(w[i] * d[i] for i in range(len(b))) / total


# ===========================================================================
# 16. FRONTEND PAYLOADS
# ===========================================================================
def build_accessibility_equity_payload(
        access: Sequence[float], *,
        labels: Optional[Sequence[str]] = None,
        population: Optional[Sequence[float]] = None,
        groups: Optional[Sequence[Group]] = None,
        decay: Any = None,
        cumulative: Optional[CumulativeResult] = None,
        fca: Optional[FCAResult] = None,
        profile: Optional[InequalityProfile] = None,
        decomposition: Optional[TheilDecomposition] = None,
        poverty: Optional[TransportPovertyResult] = None,
        incidence: Optional[IncidenceResult] = None,
        disruption: Optional[DisruptionResult] = None,
        max_cost: float = 60.0, curve_points: int = 40) -> Dict[str, Any]:
    """Frontend-ready JSON dict: the surface, its Lorenz curve and every index."""
    a = _as_floats(access, "access")
    _non_empty(a, "access")
    pop = _default_weights(population, len(a), "population")
    names = (["Z%d" % i for i in range(len(a))] if labels is None
             else [str(s) for s in labels])
    _check_length(names, len(a), "labels", "zone")
    curve = lorenz_curve([max(0.0, v) for v in a], pop)
    payload: Dict[str, Any] = {
        "zones": [{"id": names[i], "access": _json_safe(a[i]),
                   "population": _json_safe(pop[i]),
                   "group": (None if groups is None else str(groups[i]))}
                  for i in range(len(a))],
        "summary": {
            "mean": _json_safe(_mean(a)),
            "populationWeightedMean": _json_safe(_mean(a, pop)),
            "min": _json_safe(min(a)), "max": _json_safe(max(a)),
            "worstOff": _json_safe(rawlsian_index(a)),
        },
        "lorenz": curve.to_dict(),
    }
    if decay is not None:
        f = _as_decay(decay)
        payload["decay"] = dict(
            f.to_dict(),
            curve=[{"cost": _json_safe(c), "weight": _json_safe(wv)}
                   for (c, wv) in decay_curve(f, max_cost, curve_points)],
            halfCost=_json_safe(decay_half_cost(f)))
    if cumulative is not None:
        payload["cumulative"] = {
            "thresholds": [_json_safe(t) for t in cumulative.thresholds],
            "mean": [_json_safe(m) for m in cumulative.mean],
            "totalOpportunities": _json_safe(cumulative.total_opportunities),
            "curves": [{"id": cumulative.labels[i],
                        "points": [{"threshold": _json_safe(t),
                                    "opportunities": _json_safe(v)}
                                   for (t, v) in cumulative.curve(i)]}
                       for i in range(len(cumulative.access))],
        }
    if fca is not None:
        payload["floatingCatchment"] = fca.to_dict()
    if profile is not None:
        payload["inequality"] = profile.to_dict()
    if decomposition is not None:
        payload["theilDecomposition"] = decomposition.to_dict()
    if poverty is not None:
        payload["transportPoverty"] = poverty.to_dict()
    if incidence is not None:
        payload["incidence"] = incidence.to_dict()
    if disruption is not None:
        payload["disruption"] = disruption.to_dict()
    return payload


def build_a2ui_payload(access: Sequence[float], *,
                       population: Optional[Sequence[float]] = None,
                       profile: Optional[InequalityProfile] = None,
                       gap: Optional[GroupGapResult] = None,
                       poverty: Optional[TransportPovertyResult] = None,
                       moran: Optional[MoranResult] = None,
                       disruption: Optional[DisruptionResult] = None,
                       decay: Any = None,
                       bins: int = 10) -> Dict[str, Any]:
    """Visualisation-ready dict: stat cards, Lorenz curve, histogram and bands."""
    a = _as_floats(access, "access")
    _non_empty(a, "access")
    pop = _default_weights(population, len(a), "population")
    nb = _positive_int(bins, "bins")
    cards: List[Dict[str, Any]] = [
        {"label": "Mean access", "value": round(_mean(a, pop), 2),
         "unit": "opportunities"},
        {"label": "Worst off", "value": round(min(a), 2),
         "unit": "opportunities"},
        {"label": "Best served", "value": round(max(a), 2),
         "unit": "opportunities"},
    ]
    if profile is not None:
        cards.append({"label": "Gini", "value": round(profile.gini, 4),
                      "unit": "0-1"})
        cards.append({"label": "Palma", "value": round(profile.palma, 3)
                      if profile.palma != _INF else None, "unit": "ratio"})
    if gap is not None:
        cards.append({"label": "Group gap",
                      "value": round(gap.absolute_gap, 2),
                      "unit": "opportunities"})
    if poverty is not None:
        cards.append({"label": "Transport poor",
                      "value": round(100.0 * poverty.headcount_ratio, 1),
                      "unit": "%"})
    if moran is not None:
        cards.append({"label": "Moran's I", "value": round(moran.i, 4),
                      "unit": "p=%.3f" % moran.p_value})
    if disruption is not None:
        cards.append({"label": "Resilience",
                      "value": round(100.0 * disruption.resilience, 1),
                      "unit": "%"})
    lo, hi = min(a), max(a)
    width = (hi - lo) / nb if hi > lo else 1.0
    counts = [0.0] * nb
    for i in range(len(a)):
        k = nb - 1 if hi <= lo else min(nb - 1, int((a[i] - lo) / width))
        counts[k] += pop[i]
    curve = lorenz_curve([max(0.0, v) for v in a], pop)
    charts: Dict[str, Any] = {
        "lorenz": [{"population": _json_safe(x), "share": _json_safe(y)}
                   for (x, y) in curve.points()],
        "equality": [{"population": 0.0, "share": 0.0},
                     {"population": 1.0, "share": 1.0}],
        "histogram": [{"lower": _json_safe(lo + k * width),
                       "upper": _json_safe(lo + (k + 1) * width),
                       "population": _json_safe(counts[k])}
                      for k in range(nb)],
    }
    if decay is not None:
        f = _as_decay(decay)
        charts["decay"] = [{"cost": _json_safe(c), "weight": _json_safe(w)}
                           for (c, w) in decay_curve(f, 60.0, 40)]
    if gap is not None:
        charts["groupMeans"] = [{"group": str(gap.groups[k]),
                                 "mean": _json_safe(gap.means[k]),
                                 "population": _json_safe(gap.populations[k])}
                                for k in range(len(gap.groups))]
    return {"title": "Accessibility and equity",
            "statCards": cards,
            "charts": charts,
            "verdict": ("unassessed" if profile is None else
                        ("even" if profile.gini < 0.2 else
                         "uneven" if profile.gini < 0.4 else "polarised"))}


# ===========================================================================
# 17. DEMO
# ===========================================================================
def _demo() -> None:
    print("=" * 70)
    print("  ACCESSIBILITY & EQUITY (pure Python) - demo")
    print("=" * 70)

    # -- 1. distance decay -------------------------------------------------
    print("\n-- 1. impedance kernels --")
    kernels = [
        get_decay("power", beta=1.8, cost_min=2.0),
        get_decay("exponential", beta=0.08),
        get_decay("gaussian", beta=15.0),
        get_decay("log_logistic", alpha=18.0, beta=3.0),
        get_decay("rectangular", threshold=30.0),
        get_decay("cumulative_gaussian", mu=25.0, sigma=5.0),
    ]
    closed = [2.0 * 2.0 ** (1.0 / 1.8), math.log(2.0) / 0.08,
              15.0 * math.sqrt(2.0 * math.log(2.0)), 18.0, 30.0, 25.0]
    for k, f in enumerate(kernels):
        half = decay_half_cost(f)
        print("   %-20s f(10)=%.4f  f(30)=%.4f  half-cost %8.4f "
              "(closed form %8.4f, err %.2e)"
              % (f.name, f(10.0), f(30.0), half, closed[k],
                 abs(half - closed[k])))
    print("   (the cumulative-Gaussian kernel starts at Phi(mu/sigma) < 1, so "
          "its half-cost is mu only in the limit)")
    print("   DECAY_NAMES = %s" % (DECAY_NAMES,))
    print("   raw kernels: power %.5f  exp %.5f  gauss %.5f  loglog %.5f  "
          "rect %.1f  cumgauss %.5f"
          % (decay_power(20.0, 1.8, 2.0), decay_exponential(20.0, 0.08),
             decay_gaussian(20.0, 15.0), decay_log_logistic(20.0, 18.0, 3.0),
             decay_rectangular(20.0, 30.0),
             decay_cumulative_gaussian(20.0, 25.0, 5.0)))
    print("   normal_cdf(0) = %.6f, normal_cdf(1.96) = %.6f"
          % (normal_cdf(0.0), normal_cdf(1.96)))
    rng = random.Random(7)
    trips = [round(2.0 + 58.0 * rng.random(), 3) for _ in range(400)]
    observed = mean_decay_cost(trips, None, get_decay("exponential", beta=0.06))
    fitted = calibrate_decay(trips, observed, "exponential")
    print("   calibrate: mean trip cost %.4f min -> beta %.6f (true 0.060000, "
          "err %.2e)" % (observed, fitted.params["beta"],
                         abs(fitted.params["beta"] - 0.06)))
    print("   decay_curve points: %s"
          % [(round(c, 1), round(w, 3))
             for (c, w) in decay_curve(kernels[1], 40.0, 5)])

    # -- 2. the city and its cost matrix -----------------------------------
    print("\n-- 2. a synthetic city and its travel-cost matrix --")
    rows, cols = 5, 5
    coords = [(51.50 + 0.010 * r, -0.120 + 0.016 * c)
              for r in range(rows) for c in range(cols)]
    zone_names = ["Z%02d" % i for i in range(len(coords))]
    rng = random.Random(11)
    population = [round(900 + 2600 * rng.random(), 1) for _ in coords]
    jobs = []
    for r in range(rows):
        for c in range(cols):
            pull = math.exp(-0.5 * (((r - 2) / 1.2) ** 2 + ((c - 2) / 1.2) ** 2))
            jobs.append(round(200.0 + 9000.0 * pull * (0.6 + 0.8 * rng.random()),
                              1))
    cm = cost_matrix_from_coords(coords, speed_kph=24.0, circuity=1.3,
                                 intrazonal=intrazonal_cost(1.2e6, 24.0))
    print("   %d zones, %d x %d grid; %s" % (len(coords), rows, cols,
                                             cm.to_dict()["shape"]))
    print("   centre-to-corner %.2f min, intrazonal %.3f min, max %.2f min"
          % (cm.get(12, 0), cm.get(0, 0), cm.max_finite()))
    print("   haversine (0 -> 1) = %.1f m, euclidean on a metric grid = %.1f m"
          % (haversine_m(coords[0], coords[1]),
             euclidean_m((0.0, 0.0), (3.0, 4.0))))
    print("   travel_time(1200 m, 24 kph) = %.4f min; transposed shape %s"
          % (travel_time(1200.0, 24.0), cm.transposed().to_dict()["shape"]))
    if _HAVE_CORE_GEO:
        err = max(abs(haversine_m(coords[i], coords[j])
                      - _core_haversine(coords[i], coords[j]))
                  for i in range(len(coords)) for j in range(len(coords)))
        print("   core geospatial_graphs on the path: max haversine "
              "disagreement %.3e m" % err)
        index = _CoreGeoIndex(dict(zip(zone_names, coords)))
        near, metres = index.nearest((51.523, -0.081))
        print("   core GeoIndex: nearest zone to (51.523, -0.081) is %s at "
              "%.0f m" % (near, metres))
    else:
        print("   core geospatial_graphs not importable -- using the local "
              "great-circle formula")
    edges = []
    for r in range(rows):
        for c in range(cols):
            here = r * cols + c
            if c + 1 < cols:
                edges.append((zone_names[here], zone_names[here + 1], 3.0))
            if r + 1 < rows:
                edges.append((zone_names[here], zone_names[here + cols], 3.4))
    net = cost_matrix_from_graph(edges, zone_names)
    print("   network matrix by Dijkstra: corner-to-corner %.2f (grid distance "
          "%.2f)" % (net.get(0, 24), 4 * 3.0 + 4 * 3.4))
    if _HAVE_CORE_PATHS:
        pg = _CorePathGraph()
        pg.add_edges_from(edges)
        core_dist = _core_shortest_path(pg, zone_names[0])
        gap = max(abs(net.get(0, j)
                      - core_dist.distance_to(zone_names[j]))
                  for j in range(len(zone_names)))
        print("   core shortest_paths on the path: max Dijkstra disagreement "
              "%.3e min" % gap)
    else:
        print("   core shortest_paths not importable -- using the local "
              "Dijkstra")

    # -- 3. cumulative opportunity and its dual ----------------------------
    print("\n-- 3. cumulative opportunity --")
    thresholds = [5.0, 10.0, 15.0, 20.0, 30.0]
    cum = cumulative_opportunity_curve(cm, jobs, thresholds, population,
                                       zone_names)
    for i in (0, 12):
        print("   %s: %s" % (zone_names[i],
                             "  ".join("<=%dmin %7.0f" % (t, v)
                                       for (t, v) in cum.curve(i))))
    print("   population-weighted mean by threshold: %s"
          % ["%.0f" % m for m in cum.mean])
    print("   total jobs %.0f; centre reaches %.1f%% of them within 15 min"
          % (cum.total_opportunities, 100.0 * cum.share(12, 2)))
    print("   at exactly 30 min, zone 0 reaches %.0f jobs (single-threshold "
          "call %.0f)" % (cum.at(30.0)[0],
                          cumulative_opportunity(cm, jobs, 30.0)[0]))
    bands = isochrone_bands(cm.row(0), jobs, thresholds)
    print("   isochrone bands from %s: %s" % (zone_names[0],
                                              ["%.0f" % b[2] for b in bands]))
    print("   bands sum %.4f == cumulative at 30 min %.4f"
          % (math.fsum(b[2] for b in bands), cum.at(30.0)[0]))
    pr = place_rank_accessibility(cm, jobs, [20000.0, 60000.0], zone_names)
    print("   cost to reach 20k / 60k jobs: %s = %.2f / %.2f min, %s = "
          "%.2f / %.2f min"
          % (zone_names[12], pr.costs[12][0], pr.costs[12][1], zone_names[0],
             pr.costs[0][0], pr.costs[0][1]))
    print("   median cost to 20k jobs %.2f min; centre reaches 60k: %s"
          % (pr.median_cost(0), pr.reachable(12, 1)))
    print("   nearest-job cost from zone 0 = %.3f min"
          % time_to_nth_opportunity(cm.row(0), jobs, 1.0))

    # -- 4. gravity, Hansen, potential and Shen ----------------------------
    print("\n-- 4. gravity / potential accessibility --")
    decay = get_decay("exponential", beta=0.08)
    grav = gravity_accessibility(cm, jobs, decay)
    hans = hansen_accessibility(cm, jobs, beta=1.6, alpha=0.9, cost_min=2.0,
                                labels=zone_names)
    pot = potential_accessibility(cm, jobs, decay, exclude_self=True,
                                  labels=zone_names)
    shen = shen_accessibility(cm, jobs, population, decay)
    print("   gravity  centre %.1f  corner %.1f  (ratio %.2f)"
          % (grav[12], grav[0], grav[12] / grav[0]))
    print("   hansen   centre %.1f  corner %.1f  | ranking top three %s"
          % (hans.values[12], hans.values[0],
             [zone_names[i] for i in hans.ranking()[:3]]))
    print("   potential without the self term: centre %.1f (with it %.1f)"
          % (pot.values[12], grav[12]))
    print("   shen (jobs per worker): centre %.4f corner %.4f"
          % (shen[12], shen[0]))
    print("   normalised (max) gravity, first five: %s"
          % ["%.3f" % v for v in normalise_accessibility(grav, "max")[:5]])
    print("   hansen mean %.2f, population-weighted %.2f, payload keys %s"
          % (hans.mean(), hans.mean(population), sorted(hans.to_dict())))

    # -- 5. floating catchments --------------------------------------------
    print("\n-- 5. floating catchment methods (GP surgeries) --")
    surgeries = [0.0] * len(coords)
    for j, capacity in ((6, 4.0), (12, 9.0), (18, 5.0), (22, 3.0)):
        surgeries[j] = capacity
    radius = 9.0
    two = two_step_fca(cm, surgeries, population, radius)
    e2 = enhanced_2sfca(cm, surgeries, population,
                        [(10.0, 1.0), (20.0, 0.68), (30.0, 0.22)])
    three = three_step_fca(cm, surgeries, population, radius,
                           get_decay("gaussian", beta=15.0))
    m2 = m2sfca(cm, surgeries, population, radius,
                get_decay("gaussian", beta=15.0))
    var = variable_catchment_2sfca(cm, surgeries, population, 20000.0, 8.0)
    total_supply = math.fsum(surgeries)
    for name, res in (("2SFCA", two), ("E2SFCA", e2), ("3SFCA", three),
                      ("M2SFCA", m2), ("V2SFCA", var)):
        print("   %-7s centre %.6f  corner %.6f  allocated %.6f / %.1f "
              "doctors%s"
              % (name, res.access[12] * 1000.0, res.access[0] * 1000.0,
                 res.allocated_supply(population), total_supply,
                 "  [conserves]" if res.conserves_supply else ""))
    print("   (accessibility quoted per 1000 residents)")
    print("   2SFCA supply-to-demand ratios x1000: %s"
          % ["%.4f" % (r * 1000.0) for r in two.ratios if r > 0.0])
    print("   variable catchment radii at the four surgeries: %s min"
          % ["%.1f" % var.supply_radii[j] for j in (6, 12, 18, 22)])
    print("   E2SFCA zone weights: %s"
          % e2sfca_zone_weights([(10.0, 1.0), (20.0, 0.68), (30.0, 0.22)]))
    flat_w = [[1.0] * len(surgeries) for _ in population]
    flat_ratio = supply_demand_ratio(flat_w, surgeries, population)
    print("   supply_demand_ratio with flat weights (the four surgeries): %s"
          % ["%.3e" % flat_ratio[j] for j in (6, 12, 18, 22)])
    uniform_cost = [[1.0] * 9 for _ in range(9)]
    uni = two_step_fca(uniform_cost, [2.0] * 9, [50.0] * 9, 5.0)
    print("   ORACLE uniform grid, uniform supply and demand: every zone gets "
          "%.12f, the global ratio is %.12f"
          % (uni.access[0], 2.0 / 50.0))

    # -- 6. utility-based accessibility ------------------------------------
    print("\n-- 6. logsum (utility-based) accessibility --")
    logsum = logsum_accessibility(cm, jobs, beta_cost=0.08, scale=1.0,
                                  size_coefficient=1.0)
    print("   logsum centre %.4f, corner %.4f (utils); ln(sum jobs) = %.4f"
          % (logsum[12], logsum[0], math.log(math.fsum(jobs))))
    util = utility_matrix(cm, jobs, 0.08)
    probs = destination_choice_probabilities(util[12])
    top = sorted(range(len(probs)), key=lambda j: (-probs[j], j))[:3]
    print("   centre's destination shares: %s (they sum to %.12f)"
          % (["%s %.3f" % (zone_names[j], probs[j]) for j in top],
             math.fsum(probs)))
    after_logsum = logsum_accessibility(cm.scaled(0.85), jobs, 0.08)
    benefit = logsum_benefit(logsum, after_logsum, 0.12, population)
    print("   15%% faster network: %.4f utils per person -> GBP %.2f per "
          "person per trip, GBP %.0f total"
          % (after_logsum[0] - logsum[0], benefit["mean"], benefit["total"]))
    print("   gainers %d, losers %d" % (benefit["gainers"], benefit["losers"]))

    # -- 7. time of day and reliability ------------------------------------
    print("\n-- 7. time of day and reliability --")
    tod = time_of_day_accessibility(
        [("am_peak", cm.scaled(1.35), 0.25),
         ("interpeak", cm, 0.45),
         ("evening", cm.scaled(0.9), 0.30)], jobs, decay)
    print("   periods %s with weights %s"
          % (tod.periods, ["%.2f" % w for w in tod.weights]))
    print("   centre: am %.0f, interpeak %.0f, evening %.0f -> weighted %.0f"
          % (tod.access[12][0], tod.access[12][1], tod.access[12][2],
             tod.weighted[12]))
    print("   best/worst period for the centre: %s / %s; variability %.4f"
          % (tod.best_period[12], tod.worst_period[12], tod.variability[12]))
    print("   interpeak population-weighted mean %.0f"
          % tod.period_mean(1, population))
    rng = random.Random(3)
    samples = [[[cm.get(i, j) * (0.85 + 0.5 * rng.random())
                 for _ in range(12)] for j in range(len(jobs))]
               for i in range(len(coords))]
    rel = reliability_adjusted_accessibility(samples, jobs, decay,
                                             reliability_ratio=0.8)
    rel95 = reliability_adjusted_accessibility(samples, jobs, decay,
                                               percentile=95.0)
    print("   centre: mean-cost gravity %.0f, mean+0.8sd %.0f, 95th "
          "percentile %.0f" % (grav[12], rel[12], rel95[12]))
    print("   one OD sample: p50 %.3f, p95 %.3f, buffer index %.4f, planning "
          "index %.4f"
          % (travel_time_percentile(samples[12][13], 50.0),
             travel_time_percentile(samples[12][13], 95.0),
             buffer_time_index(samples[12][13]),
             planning_time_index(samples[12][13], cm.get(12, 13))))
    print("   reliability_adjusted_cost(20, 4, 0.8) = %.2f min"
          % reliability_adjusted_cost(20.0, 4.0, 0.8))

    # -- 8. inequality of the accessibility surface ------------------------
    print("\n-- 8. inequality --")
    curve = lorenz_curve(grav, population)
    prof = inequality_profile(grav, population,
                              kolm_alpha=1.0 / population_weighted_mean(
                                  grav, population))
    print("   Lorenz: %d points, area %.6f -> Gini %.6f (direct %.6f, "
          "diff %.2e)"
          % (len(curve.population), lorenz_area(curve), gini_from_lorenz(curve),
             gini(grav, population),
             abs(gini_from_lorenz(curve) - gini(grav, population))))
    print("   poorest 40%% hold %.4f of the access, richest 10%% hold %.4f"
          % (curve.share_below(0.4), 1.0 - curve.share_below(0.9)))
    print("   Gini %.6f  Theil T %.6f  Theil L %.6f  Hoover %.6f (= max Lorenz "
          "gap %.6f)"
          % (prof.gini, prof.theil_t, prof.theil_l, prof.hoover,
             curve.max_gap()))
    print("   Atkinson: %s" % ["eps=%.1f %.6f" % (e, a)
                               for (e, a) in prof.atkinson])
    print("   Palma %.4f  20:20 %.4f  CV %.4f  Kolm(alpha=1/mean) %.4f "
          "jobs" % (prof.palma, prof.quintile, prof.cv, prof.kolm))
    print("   GE(2) %.6f == CV^2/2 %.6f"
          % (generalised_entropy(grav, 2.0, population),
             0.5 * coefficient_of_variation(grav, population) ** 2))
    print("   direct calls: Palma %.4f, 20:20 %.4f, Hoover %.4f, Kolm %.4f"
          % (palma_ratio(grav, population), quintile_ratio(grav, population),
             hoover_index(grav, population),
             kolm_index(grav, 1.0 / population_weighted_mean(grav, population),
                        population)))
    cc = concentration_curve(grav, population, population)
    print("   concentration curve ranked by population: %d points, area "
          "%.6f -> index %.6f"
          % (len(cc.population), lorenz_area(cc), 1.0 - 2.0 * lorenz_area(cc)))
    equal = [40.0] * 8
    winner = [0.0] * 7 + [1.0]
    print("   ORACLE equality: Gini %.1f, Theil T %.1f | one-takes-all: Gini "
          "%.12f == (n-1)/n %.12f, Theil T %.12f == ln(n) %.12f"
          % (gini(equal), theil_t(equal), gini(winner), 7.0 / 8.0,
             theil_t(winner), math.log(8.0)))
    moved = pigou_dalton_transfer(grav, 12, 0, 0.25 * (grav[12] - grav[0]))
    print("   Pigou-Dalton transfer centre -> corner: Gini %.6f -> %.6f "
          "(falls: %s)" % (gini(grav), gini(moved), gini(moved) < gini(grav)))
    print("   concentration index of access ranked by population %.6f"
          % concentration_index(grav, population))

    # -- 9. Theil decomposition --------------------------------------------
    print("\n-- 9. between / within decomposition --")
    ring = ["inner" if max(abs(i // cols - 2), abs(i % cols - 2)) <= 1
            else "outer" for i in range(len(coords))]
    dec_t = theil_decomposition(grav, ring, "T", population)
    dec_l = theil_decomposition(grav, ring, "L", population)
    for dec in (dec_t, dec_l):
        print("   Theil %s: total %.9f = between %.9f + within %.9f "
              "(residual %.2e)"
              % (dec.kind, dec.total, dec.between, dec.within, dec.residual))
    print("   groups %s, population shares %s, means %s"
          % (dec_t.groups, ["%.3f" % p for p in dec_t.population_shares],
             ["%.0f" % m for m in dec_t.group_means]))
    print("   %.1f%% of the inequality is WITHIN the rings"
          % (100.0 * dec_t.within_share))

    # -- 10. horizontal and vertical equity --------------------------------
    print("\n-- 10. horizontal vs vertical equity --")
    deprivation = [round(0.15 + 0.7 * rng.random(), 4) for _ in coords]
    eq = horizontal_vertical_equity(grav, ring, deprivation, population)
    print("   horizontal index %.6f (1 = like treated alike), vertical index "
          "%.6f -> %s" % (eq.horizontal_index, eq.vertical_index, eq.verdict))
    print("   horizontal_equity() %.6f, vertical_equity() %.6f"
          % (horizontal_equity(grav, ring, population),
             vertical_equity(grav, deprivation, population)))
    print("   group means %s over populations %s"
          % (["%.0f" % m for m in eq.group_means],
             ["%.0f" % p for p in eq.group_populations]))

    # -- 11. need-weighted accessibility -----------------------------------
    print("\n-- 11. need- and deprivation-weighted accessibility --")
    print("   plain mean %.1f, population-weighted %.1f, need-weighted %.1f"
          % (_mean(grav), population_weighted_mean(grav, population),
             need_weighted_accessibility(grav, deprivation, population)))
    for e in (0.0, 1.0, 2.0):
        print("   deprivation elasticity %.1f -> weighted access %.1f"
              % (e, deprivation_weighted_accessibility(grav, population,
                                                       deprivation, e)))
    ew = equity_weights(population, deprivation, 1.0)
    print("   equity weights sum to %.12f, max %.4f" % (math.fsum(ew), max(ew)))
    gap = accessibility_gap(grav, ring, population)
    print("   %s" % {k: round(v, 1) for k, v in sorted(gap.as_dict().items())})
    print("   best %s, worst %s, absolute gap %.1f, ratio %.3f"
          % (gap.best, gap.worst, gap.absolute_gap, gap.ratio_gap))

    # -- 12. transport poverty ---------------------------------------------
    print("\n-- 12. transport poverty --")
    line = accessibility_poverty_line(grav, 0.5, "median", population)
    tp = transport_poverty_indicators(grav, None, population, 0.5)
    print("   poverty line (half the median) = %.1f jobs; %.1f%% of people "
          "below it" % (line, 100.0 * tp.headcount_ratio))
    print("   FGT: headcount %.4f, gap %.6f, severity %.6f, mean shortfall "
          "%.1f" % (tp.headcount_ratio, tp.gap_index, tp.severity_index,
                    tp.mean_shortfall))
    print("   poor zones: %s" % [zone_names[i] for i in tp.poor])
    print("   fgt_index(alpha=0) %.6f == headcount ratio %.6f"
          % (fgt_index(grav, line, 0.0, population), tp.headcount_ratio))
    cars = [round(0.35 + 0.55 * rng.random(), 3) for _ in coords]
    spend = [round(0.06 + 0.10 * rng.random(), 4) for _ in coords]
    tpi = transport_poverty_index(grav, line, cars, spend)
    worst = max(range(len(tpi)), key=lambda i: (tpi[i], -i))
    print("   composite index: mean %.4f, worst zone %s at %.4f"
          % (_mean(tpi), zone_names[worst], tpi[worst]))

    # -- 13. spatial autocorrelation ---------------------------------------
    print("\n-- 13. spatial autocorrelation of the surface --")
    xy = [(1200.0 * (i % cols), 1200.0 * (i // cols)) for i in range(len(coords))]
    w_knn = spatial_weights_knn(xy, 4)
    w_band = spatial_weights_distance_band(xy, 1500.0)
    w_idw = spatial_weights_inverse_distance(xy, 1.0, 3000.0)
    print("   Moran's I: kNN %.6f, distance band %.6f, inverse distance %.6f"
          % (morans_i(grav, w_knn), morans_i(grav, w_band),
             morans_i(grav, w_idw)))
    print("   Geary's C (kNN) %.6f; expectation of I = %.6f"
          % (gearys_c(grav, w_knn), morans_i_expectation(len(grav))))
    lisa = local_morans_i(grav, w_band)
    s0 = math.fsum(math.fsum(row) for row in w_band)
    print("   LISA sum %.9f == S0 * I = %.9f"
          % (math.fsum(lisa), s0 * morans_i(grav, w_band)))
    mor = morans_i_permutation_test(grav, w_band, 999, seed=42)
    print("   permutation test: I %.6f, permuted mean %.6f (theory %.6f), "
          "z %.3f, p %.4f"
          % (mor.i, mor.permuted_mean, mor.expectation, mor.z_score,
             mor.p_value))
    small = grav[:6]
    w_small = row_standardise(spatial_weights_distance_band(xy[:6], 1500.0))
    ex = morans_i_permutation_test(small, w_small, exhaustive=True)
    print("   EXHAUSTIVE (n=6, %d orderings): mean I %.15f, theory %.15f"
          % (ex.permutations, ex.permuted_mean, ex.expectation))

    # -- 14. impact incidence ----------------------------------------------
    print("\n-- 14. who gains and who loses --")
    faster = [[cm.cost[i][j] * (0.7 if (i % cols) <= 1 or (j % cols) <= 1
                                 else 1.0) for j in range(len(jobs))]
              for i in range(len(coords))]
    after = gravity_accessibility(faster, jobs, decay)
    inc = impact_incidence(grav, after, ring, population)
    print("   west-side bus lane: net change %.0f job-reaches, %.1f%% of "
          "people gain, %.1f%% lose"
          % (inc.net_change, 100.0 * inc.share_gaining,
             100.0 * inc.share_losing))
    for k, g in enumerate(inc.groups):
        print("      %-6s %.0f -> %.0f  (%+.1f%%, population %.0f)"
              % (g, inc.mean_before[k], inc.mean_after[k],
                 100.0 * inc.relative_change[k], inc.populations[k]))
    print("   Gini %.6f -> %.6f; bottom half %+.1f vs top half %+.1f -> "
          "progressive: %s"
          % (inc.gini_before, inc.gini_after, inc.bottom_half_change,
             inc.top_half_change, inc.progressive))
    wl = winners_losers(grav, after, population)
    print("   winners %d zones (%.1f%% of people), biggest winner %s"
          % (len(wl["winners"]), 100.0 * wl["winnerShare"],
             zone_names[wl["biggestWinner"]]))
    summary = distributional_summary(after, population)
    print("   distributional summary: median %.0f, Gini %.4f, Palma %.3f"
          % (summary["median"], summary["gini"], summary["palma"]))

    # -- 15. Rawlsian priority ---------------------------------------------
    print("\n-- 15. Rawlsian max-min and leximin --")
    options = [("do-nothing", grav), ("west-bus-lane", after),
               ("orbital-link", [v * 1.04 + 250.0 for v in grav])]
    mm = rawlsian_maxmin(options)
    print("   worst-off zone by option: %s"
          % ["%s %.0f" % (mm.names[k], mm.minima[k])
             for k in range(len(mm.names))])
    print("   max-min ranking %s; leximin ranking %s; best = %s"
          % (mm.maximin_ranking, mm.leximin_ranking, mm.best))
    print("   rawlsian_index(after) = %.1f; leximin key (first three) %s"
          % (rawlsian_index(after, population),
             ["%.1f" % v for v in leximin_key(after)[:3]]))
    print("   leximin_compare(after, baseline) = %+d; leximin_sort = %s"
          % (leximin_compare(after, grav), leximin_sort(options)))

    # -- 16. disruption ----------------------------------------------------
    print("\n-- 16. disruption and resilience --")
    hit = disruption_scenario(cm, jobs, decay, closed_destinations=(12,),
                              cost_multiplier=1.25, cost_penalty=2.0)
    dis = accessibility_loss(grav, hit, population)
    print("   central closure + 25%% slower + 2 min: mean loss %.0f "
          "(weighted %.0f), worst zone %s loses %.0f (%.1f%%)"
          % (dis.mean_loss, dis.weighted_mean_loss, zone_names[dis.worst_zone],
             dis.max_loss, 100.0 * dis.relative_loss[dis.worst_zone]))
    print("   resilience %.4f, loss Gini %.4f, Gini %.4f -> %.4f, %.1f%% of "
          "people lose" % (dis.resilience, dis.loss_gini, dis.baseline_gini,
                           dis.disrupted_gini, 100.0 * dis.share_losing))
    vuln = vulnerability_ranking(cm, jobs, decay, population)
    print("   most critical destinations: %s"
          % ["%s %.0f" % (zone_names[j], loss) for (j, loss) in vuln[:4]])
    print("   resilience_index (weighted) = %.6f"
          % resilience_index(grav, hit, population))

    # -- 17. payloads -------------------------------------------------------
    print("\n-- 17. frontend payloads --")
    payload = build_accessibility_equity_payload(
        grav, labels=zone_names, population=population, groups=ring,
        decay=decay, cumulative=cum, fca=two, profile=prof,
        decomposition=dec_t, poverty=tp, incidence=inc, disruption=dis)
    ui = build_a2ui_payload(grav, population=population, profile=prof, gap=gap,
                            poverty=tp, moran=mor, disruption=dis, decay=decay)
    print("   payload keys: %s" % sorted(payload))
    print("   zones %d | lorenz points %d | histogram bins %d | stat cards %d"
          % (len(payload["zones"]), len(payload["lorenz"]["points"]),
             len(ui["charts"]["histogram"]), len(ui["statCards"])))
    print("   cards: %s" % ["%s=%s" % (c["label"], c["value"])
                            for c in ui["statCards"][:5]])
    print("   verdict: %s" % ui["verdict"])
    produced = [(CumulativeResult, cum), (PlaceRankResult, pr),
                (AccessibilityResult, hans), (FCAResult, two),
                (TimeOfDayResult, tod), (LorenzCurve, curve),
                (TheilDecomposition, dec_t), (EquityDecomposition, eq),
                (GroupGapResult, gap), (TransportPovertyResult, tp),
                (MoranResult, mor), (IncidenceResult, inc),
                (MaximinResult, mm), (DisruptionResult, dis),
                (InequalityProfile, prof), (CostMatrix, cm),
                (DecayFunction, decay)]
    print("   %d result dataclasses, all of the declared type: %s"
          % (len(produced), all(isinstance(o, t) for (t, o) in produced)))

    # -- 18. validation -----------------------------------------------------
    print("\n-- 18. input validation --")
    for bad in (lambda: get_decay("teleport"),
                lambda: gini([1.0, -2.0]),
                lambda: cumulative_opportunity([[1.0, 2.0]], [1.0], 5.0),
                lambda: two_step_fca(cm.cost, surgeries, population),
                lambda: atkinson([1.0, 2.0], -1.0),
                lambda: morans_i_permutation_test(grav, w_band,
                                                  exhaustive=True)):
        try:
            bad()
            print("   ERROR: no exception raised")
        except AccessibilityError as exc:
            print("   AccessibilityError: %s" % exc)

    print("\n" + "=" * 70)
    print("  demo OK")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
