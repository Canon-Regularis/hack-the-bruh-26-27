"""
Travel Demand -- trip generation, distribution, mode choice and matrix estimation
=================================================================================

**Where the trips come from, and where they go.**  Everything downstream of this
module takes an origin-destination matrix as a given: ``traffic_assignment``
routes one over a network, ``traffic_flow`` pushes the routed flow through a
conservation law, ``emissions_environment`` burns the fuel.  This module is the
place the matrix is actually *made* -- the first three steps of the classical
four-step model, plus the two things practitioners spend most of their time on
and textbooks skip: fitting the count models that produce the trip ends, and
pulling a matrix back out of a handful of roadside counts.

Where it sits among its siblings:

* ``land_use`` produces the INPUTS -- households, jobs, floorspace per zone.
  It answers "what is in zone 7".
* ``travel_demand`` (here) turns those land-use totals into trips: how many
  leave zone 7, which zone they choose, by which mode, in which hour.  It
  answers "how many trips from 7 to 12 by bus between 08:00 and 09:00".
* ``traffic_assignment`` takes that matrix as GIVEN and finds the equilibrium
  route flows.  It answers "which road do they use".

The four steps are therefore: generation, distribution and mode choice HERE,
assignment THERE, and the feedback loop that closes them lives here too (a
model whose distribution never sees congested costs is a model of nothing).

Everything is pure standard-library Python.  The IRLS engine for the Poisson and
negative-binomial trip-rate models, the Gaussian elimination behind the
generalised-least-squares matrix estimator, the golden-section profile
likelihood for the NB dispersion and the Gumbel sampler behind the logsum
Monte-Carlo check are all built by hand: no NumPy, no SciPy, no statsmodels.

Contents
--------
0.  **Validation & numerics** -- ``TravelDemandError``, the value guards, a
    dense linear solver / inverse, ``logsumexp``, golden-section search and the
    log-gamma / digamma pair used by the count models.
1.  **Zones, trip matrices and matrix algebra** -- ``Zone`` / ``ZoneSystem`` /
    ``TripMatrix``, margins, transposes, intrazonals, scaling.
2.  **Trip generation** -- cross-classification trip rates, and linear /
    Poisson / negative-binomial GLMs fitted by iteratively reweighted least
    squares FROM SCRATCH, with the NB dispersion found by profile likelihood.
    Plus production-attraction balancing.
3.  **Deterrence functions** -- power, exponential, combined, Tanner, top-log
    and log-normal, each with an analytic derivative, plus calibration of the
    parameter against an observed mean trip cost.
4.  **Trip distribution** -- the gravity model in all three constraint flavours,
    Furness / IPF (and its entropy-maximising derivation), the growth-factor
    family (uniform, average, Fratar, Detroit), intervening opportunities and
    competing destinations, and trip-length distributions.
5.  **Matrix operations** -- sectoring, aggregation and disaggregation,
    K-factors, intrazonal estimation and the matrix comparison statistics
    (RMSE, %RMSE, R^2, slope, GEH) every matrix audit reports.
6.  **Mode choice** -- multinomial logit, nested logit with proper logsums and
    scale parameters (including the mu/theta consistency check), incremental /
    pivot-point logit, direct and cross elasticities, value of time, composite
    cost and logsum accessibility, plus a seeded Gumbel Monte-Carlo that
    demonstrates the logsum really is the expected maximum utility.
7.  **Departure-time choice and peak spreading** -- Vickrey / Small schedule
    delay, a logit over time slices, peak-hour factors, time-of-day splits and
    capacity-driven peak spreading.
8.  **The four-step loop** -- generation -> distribution -> mode choice ->
    (external assignment) -> congested costs -> back again, with method of
    successive averages and an explicit convergence gap.
9.  **OD estimation from link counts** -- assignment proportions, generalised
    least squares, entropy maximisation (ME2), a path-flow estimator, Bayesian
    updating and the Spiess gradient method, with count-fit statistics.
10. **Demand scenarios** -- seeded commuting-peak, school, stadium, weather and
    incident generators.
11. **Synthetic population and trip lists** -- IPF population synthesis,
    largest-remainder integerisation and expansion of a matrix into a list of
    individual trips (which sums back to the matrix exactly).
12. **Payloads** -- JSON-able dicts for a frontend.
13. **Demo**.

Conventions
-----------
* A matrix is a dense ``List[List[float]]`` of TRIPS (never rates), indexed
  positionally; row ``i`` is the origin / production zone, column ``j`` the
  destination / attraction zone.  ``TripMatrix`` adds zone labels on top.
* Costs are *generalised* costs in a single consistent unit (minutes, pence or
  utils -- pick one and stay with it).  Deterrence parameters carry the inverse
  unit, so every distribution result is invariant under a change of cost unit
  as long as beta is rescaled with it.
* Utilities are dimensionless; the logit ``scale`` (mu) multiplies them.  The
  probability of an alternative therefore depends only on ``scale * utility``,
  which is the scale invariance the suite checks.
* Productions and attractions are trip ENDS; a doubly-constrained model needs
  their totals to agree, so ``balance_trip_ends`` is called for you and reports
  the factor it applied.
* Everything stochastic takes an explicit ``seed`` and uses ``random.Random``,
  so results are byte-identical across runs and independent of PYTHONHASHSEED.
  Every tie is broken on an explicit key, never on dict or set order.
* British spelling in prose; conventional modelling names for the functions.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Sequence, Tuple)

__version__ = "1.0.0"

Vector = List[float]
Matrix = List[List[float]]
CostMatrix = List[List[float]]
Pairs = Dict[Tuple[int, int], float]

# _EPS guards DIMENSIONLESS quantities only -- probabilities, shares, scale
# ratios.  Anything carrying a unit (trips, costs, utilities) is compared on a
# RELATIVE tolerance against its own magnitude, never against this constant.
_EPS = 1e-12
_INF = float("inf")
EULER_MASCHERONI = 0.5772156649015328606

# Furness stall escape.  ``_FURNESS_FLOOR`` is the RELATIVE margin error (the
# absolute miss divided by the trip total) below which a non-improving sweep is
# the rounding floor rather than a transient; ``_FURNESS_STALL`` is how many
# consecutive non-improving sweeps must land there before the loop gives up.
# Both are deliberately generous: the cost of one extra sweep is trivial and the
# cost of stopping on a plateau is a silently wrong trip table.
_FURNESS_FLOOR = 1e-9
_FURNESS_STALL = 20

# How far ``peak_spreading`` may double its level c looking for a bracket.  The
# total it has to reach grows only logarithmically in c, so the budget has to be
# the float range rather than any iteration count the caller happens to pass.
_LEVEL_DOUBLINGS = 1100

# ---------------------------------------------------------------------------
# Optional core interop.  The module is fully standalone; when the orchid core
# packages happen to be importable we defer to them so that mixed pipelines agree
# to the last bit.  Both imports are guarded and both have local implementations
# immediately below (``_digamma`` and ``_latin_hypercube``).
#
# ONE deferral is deliberately NOT taken.  Deferring to the core digamma made the
# module's accuracy depend on sys.path and made it WORSE: measured against
# scipy.special.digamma over x in [0.05, 1e6] and through the reflection branch,
# the worst relative error is 7.6e-10 with the core version live against 3.9e-12
# with the series below -- digamma(1) comes back as -0.57721566494246179 instead
# of -0.57721566490153287, so the "last bit" the two agreed on was the wrong one.
# ``negbin_alpha_score`` is the only consumer and it now carries the same 1e-12
# accuracy whether or not the core package is importable.  ``_HAVE_CORE_PROB``
# stays so a caller can still see which state it is in.  ``_latin_hypercube``
# genuinely is bit-identical either way and that deferral is taken.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only when the core package is on the path
    from algorithms.core.probabilistic.probabilistic_template import (
        digamma as _core_digamma,
    )
    _HAVE_CORE_PROB = True
except ImportError:                            # pragma: no cover
    _core_digamma = None
    _HAVE_CORE_PROB = False

try:  # pragma: no cover - exercised only when the core package is on the path
    from algorithms.core.optimisation.stochastic_optimisation.\
        stochastic_optimisation_template import latin_hypercube as _core_lhs
    _HAVE_CORE_LHS = True
except ImportError:                            # pragma: no cover
    _core_lhs = None
    _HAVE_CORE_LHS = False


__all__ = [
    # errors & numerics
    "TravelDemandError", "solve_linear", "invert_matrix", "logsumexp",
    "EULER_MASCHERONI",
    # zones & matrix algebra
    "Zone", "ZoneSystem", "TripMatrix", "zeros_matrix", "uniform_matrix",
    "matrix_from_pairs", "matrix_to_pairs", "row_sums", "col_sums",
    "matrix_total", "matrix_scale", "matrix_add", "matrix_subtract",
    "matrix_transpose", "matrix_symmetrise", "matrix_round",
    "intrazonal_share", "set_intrazonals", "strip_intrazonals",
    "MatrixMargins", "matrix_margins",
    # trip generation
    "CrossClassTable", "cross_classification_trips", "trip_rate_model",
    "design_matrix", "GLMResult", "linear_glm", "poisson_glm", "negbin_glm",
    "poisson_loglik", "negbin_loglik", "negbin_alpha_score",
    "negbin_dispersion_profile", "glm_predict",
    "TripEnds", "trip_ends", "BalanceResult", "balance_trip_ends",
    # deterrence
    "DeterrenceFunction", "PowerDeterrence", "ExponentialDeterrence",
    "CombinedDeterrence", "TannerDeterrence", "TopLogDeterrence",
    "LogNormalDeterrence", "get_deterrence", "DETERRENCE_NAMES",
    "deterrence_matrix", "CalibrationResult", "calibrate_deterrence",
    # trip distribution
    "GravityResult", "gravity_model", "unconstrained_gravity",
    "singly_constrained_gravity", "doubly_constrained_gravity",
    "FurnessResult", "furness", "ipf", "furness_to_targets",
    "GrowthFactorResult", "uniform_growth", "average_growth_factor",
    "fratar", "detroit",
    "wilson_entropy", "relative_entropy", "entropy_maximising_matrix",
    "intervening_opportunities", "competing_destinations",
    "TripLengthDistribution", "trip_length_distribution", "mean_trip_cost",
    # matrix operations
    "SectorSystem", "sector_matrix", "aggregate_matrix", "disaggregate_matrix",
    "k_factors", "apply_k_factors", "estimate_intrazonals",
    "MatrixComparison", "compare_matrices", "geh", "geh_matrix",
    # mode choice
    "Alternative", "linear_utility", "logit_probabilities", "logsum",
    "ModeChoiceResult", "multinomial_logit", "Nest", "NestingReport",
    "nesting_consistency", "NestedLogitResult", "nested_logit",
    "incremental_logit", "pivot_point", "ElasticityResult",
    "direct_elasticity", "cross_elasticity", "logit_elasticities",
    "arc_elasticity", "value_of_time", "composite_cost",
    "logsum_accessibility", "monte_carlo_logit",
    # departure time & peak
    "TimeSlice", "schedule_delay_utility", "DepartureTimeResult",
    "departure_time_choice", "peak_hour_factor", "time_of_day_split",
    "apply_time_of_day", "peak_spreading",
    # four-step loop
    "FourStepConfig", "FourStepIteration", "FourStepResult", "four_step_model",
    "msa_blend", "feedback_gap",
    # od estimation
    "AssignmentProportions", "assign_matrix_to_links", "link_flows",
    "ODEstimationResult", "gls_matrix_estimation", "me2_estimation",
    "path_flow_estimator", "bayesian_matrix_update", "spiess_gradient",
    "CountFit", "count_fit",
    # scenarios
    "ScenarioResult", "SCENARIO_NAMES", "commuting_peak_scenario",
    "school_scenario", "stadium_scenario", "weather_adjustment",
    "incident_adjustment", "generate_scenario",
    # synthetic population & trip lists
    "Household", "Person", "Trip", "SyntheticPopulation",
    "synthesise_population", "integerise_matrix", "expand_to_trip_list",
    "trip_list_to_matrix",
    # payloads
    "build_travel_demand_payload", "build_a2ui_payload",
]


# ===========================================================================
# 0. VALIDATION HELPERS AND SMALL NUMERICS
# ===========================================================================
class TravelDemandError(ValueError):
    """Invalid input to a travel-demand routine."""


def _as_float(x: Any, name: str = "value") -> float:
    """Coerce to a finite float or raise."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise TravelDemandError("%s must be a real number (got %r)." % (name, x))
    if math.isnan(v) or math.isinf(v):
        raise TravelDemandError("%s must be finite (got %r)." % (name, x))
    return v


def _as_floats(xs: Any, name: str = "values") -> Vector:
    """Coerce a sequence to a list of finite floats or raise."""
    if isinstance(xs, (str, bytes)) or not isinstance(xs, Iterable):
        raise TravelDemandError("%s must be a sequence of real numbers." % name)
    try:
        out = [float(v) for v in xs]
    except (TypeError, ValueError):
        raise TravelDemandError("%s must be a sequence of real numbers." % name)
    for v in out:
        if math.isnan(v) or math.isinf(v):
            raise TravelDemandError("%s contains non-finite values." % name)
    return out


def _positive(x: Any, name: str = "value") -> float:
    """Coerce to a strictly positive finite float or raise."""
    v = _as_float(x, name)
    if v <= 0.0:
        raise TravelDemandError("%s must be > 0 (got %.6g)." % (name, v))
    return v


def _non_negative(x: Any, name: str = "value") -> float:
    """Coerce to a non-negative finite float or raise."""
    v = _as_float(x, name)
    if v < 0.0:
        raise TravelDemandError("%s must be >= 0 (got %.6g)." % (name, v))
    return v


def _in_unit(x: Any, name: str = "value") -> float:
    """Coerce to a float in [0, 1] or raise."""
    v = _as_float(x, name)
    if v < 0.0 or v > 1.0:
        raise TravelDemandError("%s must lie in [0, 1] (got %.6g)." % (name, v))
    return v


def _positive_int(x: Any, name: str = "count") -> int:
    """Coerce to a strictly positive int or raise."""
    try:
        v = int(x)
    except (TypeError, ValueError):
        raise TravelDemandError("%s must be an integer (got %r)." % (name, x))
    if v <= 0:
        raise TravelDemandError("%s must be > 0 (got %d)." % (name, v))
    return v


def _non_negative_int(x: Any, name: str = "count") -> int:
    """Coerce to a non-negative int or raise."""
    try:
        v = int(x)
    except (TypeError, ValueError):
        raise TravelDemandError("%s must be an integer (got %r)." % (name, x))
    if v < 0:
        raise TravelDemandError("%s must be >= 0 (got %d)." % (name, v))
    return v


def _as_matrix(m: Any, name: str = "matrix", *,
               non_negative: bool = False) -> Matrix:
    """Coerce to a rectangular, finite, non-empty dense matrix or raise."""
    if isinstance(m, TripMatrix):
        m = m.values
    if isinstance(m, (str, bytes)) or not isinstance(m, Iterable):
        raise TravelDemandError("%s must be a sequence of rows." % name)
    rows = list(m)
    if not rows:
        raise TravelDemandError("%s must have at least one row." % name)
    out: Matrix = []
    width = -1
    for r, row in enumerate(rows):
        vals = _as_floats(row, "%s row %d" % (name, r))
        if width < 0:
            width = len(vals)
            if width == 0:
                raise TravelDemandError("%s must have at least one column." % name)
        elif len(vals) != width:
            raise TravelDemandError(
                "%s is ragged: row 0 has %d columns, row %d has %d."
                % (name, width, r, len(vals)))
        if non_negative:
            for v in vals:
                if v < 0.0:
                    raise TravelDemandError(
                        "%s must be non-negative (row %d holds %.6g)."
                        % (name, r, v))
        out.append(vals)
    return out


def _same_shape(a: Matrix, b: Matrix, name_a: str = "a",
                name_b: str = "b") -> None:
    """Raise unless two matrices have identical dimensions."""
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise TravelDemandError(
            "%s is %dx%d but %s is %dx%d." % (name_a, len(a), len(a[0]),
                                              name_b, len(b), len(b[0])))


def _rng(seed: Any) -> random.Random:
    """A seeded ``random.Random`` (``None`` still seeds deterministically)."""
    if isinstance(seed, random.Random):
        return seed
    return random.Random(0 if seed is None else seed)


def _mean(xs: Sequence[float]) -> float:
    """Arithmetic mean using fsum (0.0 for an empty sample)."""
    if not xs:
        return 0.0
    return math.fsum(xs) / len(xs)


def solve_linear(a: Matrix, b: Vector) -> Vector:
    """Solve ``A x = b`` by Gaussian elimination with partial pivoting."""
    A = [list(row) for row in _as_matrix(a, "A")]
    rhs = _as_floats(b, "b")
    n = len(A)
    if n != len(A[0]):
        raise TravelDemandError("A must be square (got %dx%d)." % (n, len(A[0])))
    if len(rhs) != n:
        raise TravelDemandError("b has length %d but A is %dx%d."
                                % (len(rhs), n, n))
    aug = [A[i] + [rhs[i]] for i in range(n)]
    scale = max(abs(v) for row in A for v in row) or 1.0
    for k in range(n):
        piv = max(range(k, n), key=lambda r: (abs(aug[r][k]), -r))
        if abs(aug[piv][k]) <= 1e-14 * scale:
            raise TravelDemandError("matrix is singular at column %d." % k)
        if piv != k:
            aug[k], aug[piv] = aug[piv], aug[k]
        pk = aug[k][k]
        for i in range(k + 1, n):
            f = aug[i][k] / pk
            if f == 0.0:
                continue
            row_i, row_k = aug[i], aug[k]
            for j in range(k, n + 1):
                row_i[j] -= f * row_k[j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = aug[i][n] - math.fsum(aug[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / aug[i][i]
    return x


def invert_matrix(a: Matrix) -> Matrix:
    """Inverse of a square matrix by Gauss-Jordan elimination."""
    A = [list(row) for row in _as_matrix(a, "A")]
    n = len(A)
    if n != len(A[0]):
        raise TravelDemandError("A must be square (got %dx%d)." % (n, len(A[0])))
    aug = [A[i] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    scale = max(abs(v) for row in A for v in row) or 1.0
    for k in range(n):
        piv = max(range(k, n), key=lambda r: (abs(aug[r][k]), -r))
        if abs(aug[piv][k]) <= 1e-14 * scale:
            raise TravelDemandError("matrix is singular at column %d." % k)
        if piv != k:
            aug[k], aug[piv] = aug[piv], aug[k]
        pk = aug[k][k]
        aug[k] = [v / pk for v in aug[k]]
        for i in range(n):
            if i == k:
                continue
            f = aug[i][k]
            if f == 0.0:
                continue
            row_i, row_k = aug[i], aug[k]
            for j in range(k, 2 * n):
                row_i[j] -= f * row_k[j]
    return [row[n:] for row in aug]


def logsumexp(values: Sequence[float], scale: float = 1.0) -> float:
    """``log(sum(exp(scale * v)))`` computed without overflow."""
    vals = _as_floats(values, "values")
    if not vals:
        raise TravelDemandError("logsumexp of an empty sequence.")
    mu = _as_float(scale, "scale")
    scaled = [mu * v for v in vals]
    top = max(scaled)
    return top + math.log(math.fsum(math.exp(s - top) for s in scaled))


def _golden_section_max(fn: Callable[[float], float], lo: float, hi: float,
                        iters: int = 300,
                        tol: float = 1e-12) -> Tuple[float, float]:
    """Golden-section maximiser of a unimodal ``fn`` on ``[lo, hi]``."""
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(iters):
        if b - a <= tol * max(1.0, abs(a) + abs(b)):
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = fn(d)
    x = 0.5 * (a + b)
    return x, fn(x)


def _bisect_root(fn: Callable[[float], float], lo: float, hi: float,
                 iters: int = 300) -> float:
    """Bisection root of a sign-changing continuous function on [lo, hi]."""
    flo, fhi = fn(lo), fn(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise TravelDemandError("root is not bracketed on [%.6g, %.6g]."
                                % (lo, hi))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = fn(mid)
        if fm == 0.0:
            return mid
        if flo * fm < 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
        if hi - lo <= 1e-15 * max(1.0, abs(hi)):
            break
    return 0.5 * (lo + hi)


def _digamma(x: float) -> float:
    """Digamma (psi) via recurrence plus the asymptotic series.

    Deliberately NOT deferred to ``algorithms.core.probabilistic`` even when that
    package is importable -- see the interop note at the top of the module: the
    core version is ~200x less accurate here, and a sys.path-dependent answer is
    worse than a standalone one.
    """
    v = _as_float(x, "x")
    if v <= 0.0 and v == math.floor(v):
        raise TravelDemandError("digamma is undefined at %.6g." % v)
    if v < 0.0:                                # reflection formula
        return _digamma(1.0 - v) - math.pi / math.tan(math.pi * v)
    result = 0.0
    while v < 8.0:
        result -= 1.0 / v
        v += 1.0
    inv = 1.0 / v
    inv2 = inv * inv
    series = (inv2 * (-1.0 / 12.0 + inv2 * (1.0 / 120.0 + inv2 *
              (-1.0 / 252.0 + inv2 * (1.0 / 240.0 + inv2 * (-1.0 / 132.0))))))
    return result + math.log(v) - 0.5 * inv + series


def _lgamma_ratio(y: float, r: float) -> float:
    """``lgamma(y + r) - lgamma(r)``, summed exactly for a small integer ``y``."""
    if y == math.floor(y) and 0.0 <= y <= 2000.0:
        n = int(y)
        if n == 0:
            return 0.0
        return math.fsum(math.log(r + k) for k in range(n))
    return math.lgamma(y + r) - math.lgamma(r)


def _latin_hypercube(n: int, dim: int, seed: Any) -> List[Vector]:
    """Latin-hypercube design on the unit cube (core version when available)."""
    if _HAVE_CORE_LHS:                         # pragma: no cover - core present
        return [list(p) for p in _core_lhs(n, dim, seed=seed)]
    rng = _rng(seed)
    pts = [[0.0] * dim for _ in range(n)]
    for d in range(dim):
        perm = list(range(n))
        rng.shuffle(perm)
        for i in range(n):
            pts[i][d] = (perm[i] + rng.random()) / n
    return pts


# ===========================================================================
# 1. ZONES, TRIP MATRICES AND MATRIX ALGEBRA
# ===========================================================================
# A zone system is an ordered list of zones; every matrix in this module is
# indexed by POSITION in that list, so the order is part of the data and is
# never re-derived from a set or a dict.  ``TripMatrix`` carries the labels
# alongside the numbers for the moments when you need them (payloads, audits,
# sector reports); the numerical routines take and return plain nested lists.
# ---------------------------------------------------------------------------
@dataclass
class Zone:
    """One model zone: an identifier, land-use totals and a centroid."""

    ident: str
    households: float = 0.0
    population: float = 0.0
    employment: float = 0.0
    retail: float = 0.0
    income: float = 0.0
    area: float = 1.0
    x: float = 0.0
    y: float = 0.0
    extras: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ident = str(self.ident)
        self.households = _non_negative(self.households, "households")
        self.population = _non_negative(self.population, "population")
        self.employment = _non_negative(self.employment, "employment")
        self.retail = _non_negative(self.retail, "retail")
        self.income = _non_negative(self.income, "income")
        self.area = _positive(self.area, "area")
        self.x = _as_float(self.x, "x")
        self.y = _as_float(self.y, "y")
        self.extras = {str(k): _as_float(v, "extras[%s]" % k)
                       for k, v in sorted(dict(self.extras).items())}

    def attribute(self, name: str) -> float:
        """Named land-use attribute, looking in the fields then in ``extras``."""
        key = str(name)
        if key in ("households", "population", "employment", "retail",
                   "income", "area", "x", "y"):
            return float(getattr(self, key))
        if key in self.extras:
            return self.extras[key]
        raise TravelDemandError("zone %s has no attribute %r." % (self.ident, name))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the zone."""
        return {"id": self.ident, "households": self.households,
                "population": self.population, "employment": self.employment,
                "retail": self.retail, "income": self.income,
                "area": self.area, "x": self.x, "y": self.y,
                "extras": dict(self.extras)}


class ZoneSystem:
    """An ordered collection of zones with centroid geometry and attributes."""

    def __init__(self, zones: Sequence[Zone]) -> None:
        items = list(zones)
        if not items:
            raise TravelDemandError("a zone system needs at least one zone.")
        out: List[Zone] = []
        for z in items:
            out.append(z if isinstance(z, Zone) else Zone(**dict(z)))
        seen: Dict[str, int] = {}
        for k, z in enumerate(out):
            if z.ident in seen:
                raise TravelDemandError("duplicate zone id %r (positions %d and %d)."
                                        % (z.ident, seen[z.ident], k))
            seen[z.ident] = k
        self.zones: List[Zone] = out
        self._index: Dict[str, int] = seen

    def __len__(self) -> int:
        return len(self.zones)

    def __iter__(self) -> Iterator[Zone]:
        return iter(self.zones)

    def __getitem__(self, key: Any) -> Zone:
        if isinstance(key, int):
            return self.zones[key]
        return self.zones[self.index(key)]

    @property
    def n(self) -> int:
        """Number of zones."""
        return len(self.zones)

    def ids(self) -> List[str]:
        """Zone identifiers in model order."""
        return [z.ident for z in self.zones]

    def index(self, ident: Any) -> int:
        """Position of a zone identifier in model order."""
        key = str(ident)
        if key not in self._index:
            raise TravelDemandError("unknown zone %r." % ident)
        return self._index[key]

    def vector(self, attribute: str) -> Vector:
        """One land-use attribute as a vector in model order."""
        return [z.attribute(attribute) for z in self.zones]

    def distance_matrix(self, intrazonal_factor: float = 0.5) -> Matrix:
        """Euclidean centroid distances; intrazonals are ``factor * sqrt(area)``."""
        f = _non_negative(intrazonal_factor, "intrazonal_factor")
        n = self.n
        out = zeros_matrix(n)
        for i in range(n):
            zi = self.zones[i]
            for j in range(n):
                if i == j:
                    out[i][j] = f * math.sqrt(zi.area)
                else:
                    zj = self.zones[j]
                    out[i][j] = math.hypot(zi.x - zj.x, zi.y - zj.y)
        return out

    def cost_matrix(self, speed: float = 1.0, fixed: float = 0.0,
                    intrazonal_factor: float = 0.5) -> Matrix:
        """Generalised cost ``fixed + distance / speed`` over the zone system."""
        v = _positive(speed, "speed")
        c0 = _non_negative(fixed, "fixed")
        dist = self.distance_matrix(intrazonal_factor)
        return [[c0 + d / v for d in row] for row in dist]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the zone system."""
        return {"n": self.n, "zones": [z.to_dict() for z in self.zones]}


@dataclass
class TripMatrix:
    """A labelled dense trip table: zone ids plus an ``n x m`` array of trips."""

    values: Matrix
    origins: Optional[List[str]] = None
    destinations: Optional[List[str]] = None
    name: str = "trips"

    def __post_init__(self) -> None:
        self.values = _as_matrix(self.values, "values")
        n, m = len(self.values), len(self.values[0])
        if self.origins is None:
            self.origins = ["%d" % i for i in range(n)]
        else:
            self.origins = [str(v) for v in self.origins]
        if self.destinations is None:
            self.destinations = list(self.origins) if m == n else \
                ["%d" % j for j in range(m)]
        else:
            self.destinations = [str(v) for v in self.destinations]
        if len(self.origins) != n:
            raise TravelDemandError("%d origin labels for %d rows."
                                    % (len(self.origins), n))
        if len(self.destinations) != m:
            raise TravelDemandError("%d destination labels for %d columns."
                                    % (len(self.destinations), m))
        self.name = str(self.name)

    @property
    def shape(self) -> Tuple[int, int]:
        """(rows, columns)."""
        return (len(self.values), len(self.values[0]))

    def _oi(self, ident: Any) -> int:
        key = str(ident)
        if key not in self.origins:
            raise TravelDemandError("unknown origin %r." % ident)
        return self.origins.index(key)

    def _di(self, ident: Any) -> int:
        key = str(ident)
        if key not in self.destinations:
            raise TravelDemandError("unknown destination %r." % ident)
        return self.destinations.index(key)

    def get(self, origin: Any, destination: Any) -> float:
        """Trips between two labelled zones."""
        return self.values[self._oi(origin)][self._di(destination)]

    def set(self, origin: Any, destination: Any, trips: float) -> None:
        """Set the trips between two labelled zones."""
        self.values[self._oi(origin)][self._di(destination)] = \
            _non_negative(trips, "trips")

    def row_sums(self) -> Vector:
        """Origin (production) totals."""
        return row_sums(self.values)

    def col_sums(self) -> Vector:
        """Destination (attraction) totals."""
        return col_sums(self.values)

    def total(self) -> float:
        """Grand total of the table."""
        return matrix_total(self.values)

    def copy(self) -> "TripMatrix":
        """Deep copy."""
        return TripMatrix([list(r) for r in self.values], list(self.origins),
                          list(self.destinations), self.name)

    def scaled(self, factor: float) -> "TripMatrix":
        """A copy with every cell multiplied by ``factor``."""
        return TripMatrix(matrix_scale(self.values, factor),
                          list(self.origins), list(self.destinations), self.name)

    def to_pairs(self, drop_zeros: bool = True) -> List[Tuple[Tuple[str, str], float]]:
        """Labelled ``((origin, destination), trips)`` pairs in a sorted order."""
        out: List[Tuple[Tuple[str, str], float]] = []
        for i, o in enumerate(self.origins):
            for j, d in enumerate(self.destinations):
                v = self.values[i][j]
                if drop_zeros and v == 0.0:
                    continue
                out.append(((o, d), v))
        return out

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the trip table."""
        return {"name": self.name, "origins": list(self.origins),
                "destinations": list(self.destinations),
                "values": [list(r) for r in self.values],
                "total": self.total()}


def zeros_matrix(n: int, m: Optional[int] = None) -> Matrix:
    """An ``n x m`` matrix of zeros (square when ``m`` is omitted)."""
    rows = _positive_int(n, "n")
    cols = rows if m is None else _positive_int(m, "m")
    return [[0.0] * cols for _ in range(rows)]


def uniform_matrix(n: int, m: Optional[int] = None, value: float = 1.0) -> Matrix:
    """An ``n x m`` matrix filled with a constant."""
    rows = _positive_int(n, "n")
    cols = rows if m is None else _positive_int(m, "m")
    v = _as_float(value, "value")
    return [[v] * cols for _ in range(rows)]


def matrix_from_pairs(pairs: Pairs, n: int, m: Optional[int] = None) -> Matrix:
    """Dense matrix from an ``(i, j) -> trips`` dict, in sorted key order."""
    out = zeros_matrix(n, m)
    rows, cols = len(out), len(out[0])
    for (i, j) in sorted(dict(pairs)):
        v = _as_float(pairs[(i, j)], "pairs[%r]" % ((i, j),))
        if not (0 <= int(i) < rows and 0 <= int(j) < cols):
            raise TravelDemandError("pair (%r, %r) is outside a %dx%d matrix."
                                    % (i, j, rows, cols))
        out[int(i)][int(j)] = v
    return out


def matrix_to_pairs(matrix: Matrix, drop_zeros: bool = True) -> Pairs:
    """``(i, j) -> trips`` dict built in a deterministic (row-major) order."""
    mat = _as_matrix(matrix, "matrix")
    out: Pairs = {}
    for i, row in enumerate(mat):
        for j, v in enumerate(row):
            if drop_zeros and v == 0.0:
                continue
            out[(i, j)] = v
    return out


def row_sums(matrix: Matrix) -> Vector:
    """Row totals computed with ``math.fsum``."""
    return [math.fsum(row) for row in _as_matrix(matrix, "matrix")]


def col_sums(matrix: Matrix) -> Vector:
    """Column totals computed with ``math.fsum``."""
    mat = _as_matrix(matrix, "matrix")
    return [math.fsum(mat[i][j] for i in range(len(mat)))
            for j in range(len(mat[0]))]


def matrix_total(matrix: Matrix) -> float:
    """Grand total of a matrix."""
    mat = _as_matrix(matrix, "matrix")
    return math.fsum(math.fsum(row) for row in mat)


def matrix_scale(matrix: Matrix, factor: float) -> Matrix:
    """Every cell multiplied by a constant."""
    f = _as_float(factor, "factor")
    return [[v * f for v in row] for row in _as_matrix(matrix, "matrix")]


def matrix_add(a: Matrix, b: Matrix) -> Matrix:
    """Elementwise sum of two matrices of identical shape."""
    ma, mb = _as_matrix(a, "a"), _as_matrix(b, "b")
    _same_shape(ma, mb)
    return [[ma[i][j] + mb[i][j] for j in range(len(ma[0]))]
            for i in range(len(ma))]


def matrix_subtract(a: Matrix, b: Matrix) -> Matrix:
    """Elementwise difference of two matrices of identical shape."""
    ma, mb = _as_matrix(a, "a"), _as_matrix(b, "b")
    _same_shape(ma, mb)
    return [[ma[i][j] - mb[i][j] for j in range(len(ma[0]))]
            for i in range(len(ma))]


def matrix_transpose(matrix: Matrix) -> Matrix:
    """Transpose (destinations become origins)."""
    mat = _as_matrix(matrix, "matrix")
    return [[mat[i][j] for i in range(len(mat))] for j in range(len(mat[0]))]


def matrix_symmetrise(matrix: Matrix, weight: float = 0.5) -> Matrix:
    """``w * T + (1 - w) * T'`` -- the tour-based symmetry every PA table wants."""
    mat = _as_matrix(matrix, "matrix")
    if len(mat) != len(mat[0]):
        raise TravelDemandError("only a square matrix can be symmetrised.")
    w = _in_unit(weight, "weight")
    n = len(mat)
    return [[w * mat[i][j] + (1.0 - w) * mat[j][i] for j in range(n)]
            for i in range(n)]


def matrix_round(matrix: Matrix, decimals: int = 0) -> Matrix:
    """Round every cell (a display convenience -- it does NOT preserve totals)."""
    d = int(decimals)
    return [[round(v, d) for v in row] for row in _as_matrix(matrix, "matrix")]


def intrazonal_share(matrix: Matrix) -> float:
    """Fraction of all trips that stay inside their origin zone."""
    mat = _as_matrix(matrix, "matrix")
    if len(mat) != len(mat[0]):
        raise TravelDemandError("intrazonal share needs a square matrix.")
    total = matrix_total(mat)
    if total == 0.0:
        return 0.0
    return math.fsum(mat[i][i] for i in range(len(mat))) / total


def set_intrazonals(matrix: Matrix, values: Any) -> Matrix:
    """A copy with the diagonal replaced by a scalar or a per-zone vector."""
    mat = [list(r) for r in _as_matrix(matrix, "matrix")]
    if len(mat) != len(mat[0]):
        raise TravelDemandError("intrazonals need a square matrix.")
    n = len(mat)
    if isinstance(values, (int, float)):
        vec = [_non_negative(values, "values")] * n
    else:
        vec = _as_floats(values, "values")
        if len(vec) != n:
            raise TravelDemandError("%d intrazonal values for %d zones."
                                    % (len(vec), n))
    for i in range(n):
        mat[i][i] = vec[i]
    return mat


def strip_intrazonals(matrix: Matrix) -> Matrix:
    """A copy with a zero diagonal (the assignable part of the table)."""
    return set_intrazonals(matrix, 0.0)


@dataclass
class MatrixMargins:
    """Row / column totals and the headline shape statistics of a matrix."""

    rows: Vector
    cols: Vector
    total: float
    n_rows: int
    n_cols: int
    nonzero: int
    max_cell: float
    max_cell_index: Tuple[int, int]
    mean_cell: float
    intrazonal_total: float

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the margins."""
        return {"rows": list(self.rows), "cols": list(self.cols),
                "total": self.total, "nRows": self.n_rows,
                "nCols": self.n_cols, "nonzero": self.nonzero,
                "maxCell": self.max_cell,
                "maxCellIndex": list(self.max_cell_index),
                "meanCell": self.mean_cell,
                "intrazonalTotal": self.intrazonal_total}


def matrix_margins(matrix: Matrix) -> MatrixMargins:
    """Row sums, column sums and the shape statistics of a trip table."""
    mat = _as_matrix(matrix, "matrix")
    n, m = len(mat), len(mat[0])
    rs, cs = row_sums(mat), col_sums(mat)
    total = math.fsum(rs)
    nonzero = sum(1 for row in mat for v in row if v != 0.0)
    best = (0, 0)
    best_v = mat[0][0]
    for i in range(n):
        for j in range(m):
            if mat[i][j] > best_v:                       # ties keep the first
                best_v, best = mat[i][j], (i, j)
    intra = math.fsum(mat[i][i] for i in range(min(n, m))) if n == m else 0.0
    return MatrixMargins(rows=rs, cols=cs, total=total, n_rows=n, n_cols=m,
                         nonzero=nonzero, max_cell=best_v, max_cell_index=best,
                         mean_cell=total / (n * m), intrazonal_total=intra)


# ===========================================================================
# 2. TRIP GENERATION -- CROSS-CLASSIFICATION AND COUNT GLMs (IRLS)
# ===========================================================================
# Two traditions live here.  CROSS-CLASSIFICATION (category analysis) is the
# non-parametric one: sort households into cells of a contingency table -- cars
# owned by household size by income band -- and attach an observed trip rate to
# each cell.  It cannot extrapolate but it never argues with the data.
#
# The REGRESSION tradition fits a rate surface instead.  Trips are counts, so
# the honest link is logarithmic and the honest error is Poisson; when the data
# are overdispersed (and household travel data always are -- the variance of
# trips per household comfortably exceeds the mean) the negative binomial NB2
# with variance ``mu + alpha mu^2`` is the standard repair.
#
# Both GLMs are fitted here by ITERATIVELY REWEIGHTED LEAST SQUARES, which is
# Fisher scoring on the exponential-family likelihood: at each step build the
# working response ``z = eta + (y - mu) / (dmu/deta)`` and the working weight
# ``w = (dmu/deta)^2 / V(mu)``, then solve one weighted least-squares problem.
# For a log link that is ``w = mu`` (Poisson) or ``w = mu / (1 + alpha mu)``
# (NB2).  The NB dispersion itself is NOT a GLM parameter, so it is found by
# PROFILE LIKELIHOOD: for a grid / golden section over alpha, re-fit beta by
# IRLS and keep the alpha with the highest log-likelihood.
# ---------------------------------------------------------------------------
@dataclass
class CrossClassTable:
    """Trip rates per household category: ``(car, size, income) -> rate``."""

    rates: Dict[Tuple[Any, ...], float]
    dimensions: Tuple[str, ...] = ()
    purpose: str = "hbw"

    def __post_init__(self) -> None:
        clean: Dict[Tuple[Any, ...], float] = {}
        for key, value in dict(self.rates).items():
            tup = tuple(key) if isinstance(key, tuple) else (key,)
            clean[tup] = _non_negative(value, "rate%r" % (tup,))
        if not clean:
            raise TravelDemandError("a cross-classification table needs a cell.")
        widths = sorted({len(k) for k in clean})
        if len(widths) != 1:
            raise TravelDemandError("category keys have mixed widths %s." % widths)
        self.rates = clean
        self.dimensions = tuple(str(d) for d in self.dimensions) or \
            tuple("dim%d" % k for k in range(widths[0]))
        if len(self.dimensions) != widths[0]:
            raise TravelDemandError("%d dimension names for %d-part categories."
                                    % (len(self.dimensions), widths[0]))
        self.purpose = str(self.purpose)

    def categories(self) -> List[Tuple[Any, ...]]:
        """Category keys in a deterministic (string-sorted) order."""
        return sorted(self.rates, key=lambda k: tuple(str(v) for v in k))

    def rate(self, category: Any) -> float:
        """Trip rate of one category (raises when the cell is absent)."""
        tup = tuple(category) if isinstance(category, tuple) else (category,)
        if tup not in self.rates:
            raise TravelDemandError("no trip rate for category %r." % (tup,))
        return self.rates[tup]

    def marginal(self, axis: int) -> Dict[Any, float]:
        """Mean rate along one axis of the table (unweighted)."""
        k = _non_negative_int(axis, "axis")
        if k >= len(self.dimensions):
            raise TravelDemandError("axis %d is outside a %d-way table."
                                    % (k, len(self.dimensions)))
        buckets: Dict[Any, List[float]] = {}
        for key in self.categories():
            buckets.setdefault(key[k], []).append(self.rates[key])
        return {lab: _mean(vals) for lab, vals in
                sorted(buckets.items(), key=lambda kv: str(kv[0]))}

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the table."""
        return {"purpose": self.purpose, "dimensions": list(self.dimensions),
                "cells": [{"category": [str(v) for v in k],
                           "rate": self.rates[k]} for k in self.categories()]}


def cross_classification_trips(table: CrossClassTable, households: Any) -> Any:
    """Trips from a category table: a float for one zone, a vector for many."""
    if not isinstance(table, CrossClassTable):
        raise TravelDemandError("table must be a CrossClassTable.")
    if isinstance(households, dict):
        total = 0.0
        terms: List[float] = []
        for key in sorted(households, key=lambda k: tuple(str(v) for v in
                                                          (k if isinstance(k, tuple)
                                                           else (k,)))):
            count = _non_negative(households[key], "households%r" % (key,))
            terms.append(count * table.rate(key))
        total = math.fsum(terms)
        return total
    if isinstance(households, (str, bytes)) or not isinstance(households, Iterable):
        raise TravelDemandError("households must be a mapping or a sequence of them.")
    out: Vector = []
    for z, per_zone in enumerate(households):
        if not isinstance(per_zone, dict):
            raise TravelDemandError("households[%d] must be a mapping." % z)
        out.append(cross_classification_trips(table, per_zone))
    if not out:
        raise TravelDemandError("households must hold at least one zone.")
    return out


def trip_rate_model(zones: ZoneSystem, rates: Dict[str, float],
                    intercept: float = 0.0) -> Vector:
    """Linear trip ends ``intercept + sum_k rate_k * attribute_k`` per zone."""
    if not isinstance(zones, ZoneSystem):
        raise TravelDemandError("zones must be a ZoneSystem.")
    c0 = _as_float(intercept, "intercept")
    keys = sorted(dict(rates))
    coefs = [_as_float(rates[k], "rates[%s]" % k) for k in keys]
    out: Vector = []
    for zone in zones:
        out.append(c0 + math.fsum(coefs[k] * zone.attribute(keys[k])
                                  for k in range(len(keys))))
    return out


def design_matrix(rows: Sequence[Sequence[float]],
                  intercept: bool = True) -> Matrix:
    """Stack feature rows into a design matrix, optionally with a 1.0 column."""
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Iterable):
        raise TravelDemandError("rows must be a sequence of feature vectors.")
    items = list(rows)
    if not items:
        raise TravelDemandError("rows must hold at least one observation.")
    out: Matrix = []
    width = -1
    for i, r in enumerate(items):
        if isinstance(r, (int, float)):
            vals = [float(r)]
        else:
            vals = _as_floats(r, "rows[%d]" % i)
        if width < 0:
            width = len(vals)
        elif len(vals) != width:
            raise TravelDemandError("row 0 has %d features, row %d has %d."
                                    % (width, i, len(vals)))
        out.append(([1.0] + vals) if intercept else list(vals))
    if not out[0]:
        raise TravelDemandError("a design matrix needs at least one column.")
    return out


@dataclass
class GLMResult:
    """A fitted generalised linear model: coefficients, fit and diagnostics."""

    family: str
    link: str
    coefficients: Vector
    names: List[str]
    std_errors: Vector
    z_values: Vector
    fitted: Vector
    loglik: float
    deviance: float
    null_deviance: float
    aic: float
    bic: float
    dispersion: float
    iterations: int
    converged: bool
    n_obs: int
    covariance: Matrix
    offset: Vector = field(default_factory=list)

    def __post_init__(self) -> None:
        """Check the per-coefficient vectors line up, as every sibling does."""
        k = len(self.coefficients)
        for label, vec in (("names", self.names),
                           ("std_errors", self.std_errors),
                           ("z_values", self.z_values)):
            if len(vec) != k:
                raise TravelDemandError(
                    "%s has %d entries for %d coefficients."
                    % (label, len(vec), k))
        if len(self.covariance) != k or any(len(r) != k for r in self.covariance):
            raise TravelDemandError(
                "covariance is not %dx%d." % (k, k))

    def predict(self, x: Sequence[Sequence[float]],
                offset: Optional[Sequence[float]] = None) -> Vector:
        """Fitted means for new design rows."""
        return glm_predict(self, x, offset)

    def pseudo_r2(self) -> float:
        """McFadden-style deviance R^2 ``1 - D / D_null``."""
        if self.null_deviance == 0.0:
            return 0.0
        return 1.0 - self.deviance / self.null_deviance

    def summary(self) -> str:
        """A one-line-per-coefficient text summary."""
        lines = ["%s GLM (%s link), n=%d, loglik=%.6f, AIC=%.4f"
                 % (self.family, self.link, self.n_obs, self.loglik, self.aic)]
        for k, nm in enumerate(self.names):
            lines.append("  %-14s %+12.6f  se %10.6f  z %+8.3f"
                         % (nm, self.coefficients[k], self.std_errors[k],
                            self.z_values[k]))
        if self.family == "negbin":
            lines.append("  alpha (dispersion) = %.6f" % self.dispersion)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the fit."""
        return {"family": self.family, "link": self.link,
                "names": list(self.names),
                "coefficients": list(self.coefficients),
                "stdErrors": list(self.std_errors),
                "zValues": list(self.z_values), "loglik": self.loglik,
                "deviance": self.deviance, "nullDeviance": self.null_deviance,
                "aic": self.aic, "bic": self.bic,
                "dispersion": self.dispersion, "pseudoR2": self.pseudo_r2(),
                "iterations": self.iterations, "converged": self.converged,
                "nObs": self.n_obs}


def _glm_inputs(x: Any, y: Any, offset: Any,
                weights: Any) -> Tuple[Matrix, Vector, Vector, Vector]:
    """Validate and align a design matrix, response, offset and weights."""
    X = _as_matrix(x, "X")
    yy = _as_floats(y, "y")
    if len(X) != len(yy):
        raise TravelDemandError("X has %d rows but y has %d values."
                                % (len(X), len(yy)))
    k = len(X[0])
    if len(X) < k:
        raise TravelDemandError("%d observations cannot identify %d coefficients."
                                % (len(X), k))
    off = [0.0] * len(yy) if offset is None else _as_floats(offset, "offset")
    if len(off) != len(yy):
        raise TravelDemandError("offset has %d values for %d observations."
                                % (len(off), len(yy)))
    w = [1.0] * len(yy) if weights is None else _as_floats(weights, "weights")
    if len(w) != len(yy):
        raise TravelDemandError("weights has %d values for %d observations."
                                % (len(w), len(yy)))
    for v in w:
        if v < 0.0:
            raise TravelDemandError("weights must be non-negative.")
    return X, yy, off, w


def _weighted_ls(X: Matrix, z: Vector, w: Vector) -> Tuple[Vector, Matrix]:
    """Solve ``(X' W X) b = X' W z`` and return ``b`` and ``(X' W X)^-1``."""
    n, k = len(X), len(X[0])
    xtwx = [[0.0] * k for _ in range(k)]
    xtwz = [0.0] * k
    for a in range(k):
        for b in range(a, k):
            s = math.fsum(w[i] * X[i][a] * X[i][b] for i in range(n))
            xtwx[a][b] = s
            xtwx[b][a] = s
        xtwz[a] = math.fsum(w[i] * X[i][a] * z[i] for i in range(n))
    cov = invert_matrix(xtwx)
    beta = [math.fsum(cov[a][b] * xtwz[b] for b in range(k)) for a in range(k)]
    return beta, cov


def _irls(X: Matrix, y: Vector, off: Vector, w: Vector, family: str,
          alpha: float, max_iter: int,
          tol: float) -> Tuple[Vector, Vector, Matrix, int, bool]:
    """Fisher scoring for a Gaussian / Poisson / NB2 GLM; returns beta and mu."""
    n, k = len(X), len(X[0])
    if family == "gaussian":
        beta, cov = _weighted_ls(X, [y[i] - off[i] for i in range(n)], w)
        mu = [off[i] + math.fsum(X[i][a] * beta[a] for a in range(k))
              for i in range(n)]
        return beta, mu, cov, 1, True
    eta = [math.log(max(y[i], 0.1) + 0.1) - off[i] for i in range(n)]
    beta = [0.0] * k
    cov = [[0.0] * k for _ in range(k)]
    converged = False
    used = 0
    for it in range(1, max_iter + 1):
        used = it
        mu = [math.exp(min(700.0, eta[i] + off[i])) for i in range(n)]
        if family == "poisson":
            ww = [w[i] * mu[i] for i in range(n)]
        else:
            ww = [w[i] * mu[i] / (1.0 + alpha * mu[i]) for i in range(n)]
        z = [eta[i] + (y[i] - mu[i]) / mu[i] for i in range(n)]
        if all(v <= 0.0 for v in ww):
            raise TravelDemandError("IRLS collapsed: every working weight is zero.")
        new_beta, cov = _weighted_ls(X, z, ww)
        step = max(abs(new_beta[a] - beta[a]) for a in range(k))
        scale = max(1.0, max(abs(v) for v in new_beta))
        beta = new_beta
        eta = [math.fsum(X[i][a] * beta[a] for a in range(k)) for i in range(n)]
        if step <= tol * scale:
            converged = True
            break
    mu = [math.exp(min(700.0, eta[i] + off[i])) for i in range(n)]
    return beta, mu, cov, used, converged


def poisson_loglik(beta: Sequence[float], x: Any, y: Any,
                   offset: Optional[Sequence[float]] = None) -> float:
    """Poisson log-likelihood of a coefficient vector under a log link."""
    X, yy, off, _w = _glm_inputs(x, y, offset, None)
    b = _as_floats(beta, "beta")
    if len(b) != len(X[0]):
        raise TravelDemandError("beta has %d entries for %d columns."
                                % (len(b), len(X[0])))
    terms: Vector = []
    for i in range(len(X)):
        eta = off[i] + math.fsum(X[i][a] * b[a] for a in range(len(b)))
        mu = math.exp(min(700.0, eta))
        terms.append(yy[i] * eta - mu - math.lgamma(yy[i] + 1.0))
    return math.fsum(terms)


def negbin_loglik(beta: Sequence[float], x: Any, y: Any, alpha: float,
                  offset: Optional[Sequence[float]] = None) -> float:
    """NB2 log-likelihood with variance ``mu + alpha * mu^2`` under a log link."""
    X, yy, off, _w = _glm_inputs(x, y, offset, None)
    b = _as_floats(beta, "beta")
    if len(b) != len(X[0]):
        raise TravelDemandError("beta has %d entries for %d columns."
                                % (len(b), len(X[0])))
    a = _positive(alpha, "alpha")
    r = 1.0 / a
    terms: Vector = []
    for i in range(len(X)):
        eta = off[i] + math.fsum(X[i][c] * b[c] for c in range(len(b)))
        mu = math.exp(min(700.0, eta))
        terms.append(_lgamma_ratio(yy[i], r) - math.lgamma(yy[i] + 1.0)
                     + r * math.log(r / (r + mu))
                     + (yy[i] * math.log(mu / (r + mu)) if yy[i] > 0.0 else 0.0))
    return math.fsum(terms)


def negbin_alpha_score(beta: Sequence[float], x: Any, y: Any, alpha: float,
                       offset: Optional[Sequence[float]] = None) -> float:
    """Analytic derivative of the NB2 log-likelihood with respect to alpha."""
    X, yy, off, _w = _glm_inputs(x, y, offset, None)
    b = _as_floats(beta, "beta")
    a = _positive(alpha, "alpha")
    r = 1.0 / a
    terms: Vector = []
    for i in range(len(X)):
        eta = off[i] + math.fsum(X[i][c] * b[c] for c in range(len(b)))
        mu = math.exp(min(700.0, eta))
        terms.append(_digamma(yy[i] + r) - _digamma(r)
                     + math.log(r / (r + mu)) + 1.0 - (r + yy[i]) / (r + mu))
    return -(r * r) * math.fsum(terms)          # chain rule dr/dalpha = -r^2


def _poisson_deviance(y: Vector, mu: Vector, w: Vector) -> float:
    """Poisson deviance ``2 sum w (y log(y/mu) - (y - mu))``."""
    terms: Vector = []
    for i in range(len(y)):
        t = -(y[i] - mu[i])
        if y[i] > 0.0:
            t += y[i] * math.log(y[i] / mu[i])
        terms.append(w[i] * t)
    return 2.0 * math.fsum(terms)


def _negbin_deviance(y: Vector, mu: Vector, w: Vector, alpha: float) -> float:
    """NB2 deviance against the saturated model."""
    r = 1.0 / alpha
    terms: Vector = []
    for i in range(len(y)):
        t = 0.0
        if y[i] > 0.0:
            t += y[i] * math.log(y[i] / mu[i])
        t -= (y[i] + r) * math.log((y[i] + r) / (mu[i] + r))
        terms.append(w[i] * t)
    return 2.0 * math.fsum(terms)


def _null_means(family: str, y: Vector, off: Vector, w: Vector,
                alpha: float) -> Vector:
    """Fitted means of the INTERCEPT-ONLY model, honouring the offset.

    The null model of a GLM with an offset is ``g(mu_i) = b0 + off_i``, NOT
    ``mu_i = ybar``.  With a log link and a log-exposure offset -- which is how
    trip-rate models in this module are normally specified -- the two are wildly
    different, so reading the null deviance off ``ybar`` made ``null_deviance``
    and ``pseudo_r2`` wrong for every offset model while leaving the no-offset
    answer exactly right, which is how it went unnoticed.

    Gaussian (identity link) and Poisson (log link) have exact closed forms.  The
    negative binomial's score ``sum w (y - mu) / (1 + alpha mu)`` does not
    separate, so its single unknown is bracketed and bisected.  With a zero
    offset all three collapse to the weighted mean of y, bit for bit.
    """
    n = len(y)
    sw = math.fsum(w)
    if sw <= 0.0:
        return [0.0] * n
    if family == "gaussian":                   # identity link: mu_i = b0 + off_i
        b0 = math.fsum(w[i] * (y[i] - off[i]) for i in range(n)) / sw
        return [b0 + off[i] for i in range(n)]
    ex = [math.exp(off[i]) for i in range(n)]  # log link: mu_i = exp(b0) * ex_i
    sy = math.fsum(w[i] * y[i] for i in range(n))
    se = math.fsum(w[i] * ex[i] for i in range(n))
    if sy <= 0.0 or se <= 0.0:
        return [max(0.0, sy / sw)] * n
    r = sy / se                                # exp(b0): exact for the Poisson
    if family == "poisson" or alpha <= 0.0:
        return [r * ex[i] for i in range(n)]

    def score(t: float) -> float:
        """NB2 intercept score, strictly decreasing in t = exp(b0)."""
        return math.fsum(w[i] * (y[i] - t * ex[i]) / (1.0 + alpha * t * ex[i])
                         for i in range(n))

    s = score(r)
    if s == 0.0:                               # the zero-offset case, exactly
        return [r * ex[i] for i in range(n)]
    lo_t, hi_t = r, r
    if s > 0.0:
        hi_t = r * 2.0
        while score(hi_t) > 0.0 and hi_t < 1e300:
            lo_t, hi_t = hi_t, hi_t * 2.0
    else:
        lo_t = r * 0.5
        while score(lo_t) < 0.0 and lo_t > 1e-300:
            lo_t, hi_t = lo_t * 0.5, lo_t
    t = _bisect_root(score, lo_t, hi_t)
    return [t * ex[i] for i in range(n)]


def _finish_glm(family: str, link: str, X: Matrix, y: Vector, off: Vector,
                w: Vector, beta: Vector, mu: Vector, cov: Matrix,
                iterations: int, converged: bool, alpha: float,
                names: Optional[Sequence[str]]) -> GLMResult:
    """Assemble standard errors, deviances and information criteria."""
    n, k = len(X), len(X[0])
    if names is None:
        labels = ["x%d" % a for a in range(k)]
        if all(abs(X[i][0] - 1.0) <= 0.0 for i in range(n)):
            labels[0] = "intercept"
    else:
        labels = [str(v) for v in names]
        if len(labels) != k:
            raise TravelDemandError("%d names for %d columns." % (len(labels), k))
    mu0 = _null_means(family, y, off, w, alpha)
    if family == "gaussian":
        rss = math.fsum(w[i] * (y[i] - mu[i]) ** 2 for i in range(n))
        dof = max(1, n - k)
        phi = rss / dof
        deviance = rss
        null_dev = math.fsum(w[i] * (y[i] - mu0[i]) ** 2 for i in range(n))
        sigma2 = rss / n
        # WEIGHTS here are VARIANCE weights (var_i = sigma^2 / w_i), which is the
        # convention the standard errors already follow -- they match sm.WLS.bse
        # to 3e-16.  The density of y_i then carries a sqrt(w_i) Jacobian, so the
        # log-likelihood needs the ``+ 0.5 sum log w`` term; without it the answer
        # matched NEITHER the variance-weight nor the frequency-weight
        # convention, and aic and bic inherited the error.  Unweighted, the term
        # is exactly 0.0 and nothing moves.
        jac = 0.5 * math.fsum(math.log(w[q]) for q in range(n) if w[q] > 0.0)
        loglik = -0.5 * n * (math.log(2.0 * math.pi * sigma2) + 1.0) + jac
        n_par = k + 1
    elif family == "poisson":
        phi = 1.0
        deviance = _poisson_deviance(y, mu, w)
        null_dev = _poisson_deviance(y, mu0, w)
        loglik = math.fsum(w[i] * (y[i] * math.log(mu[i]) - mu[i]
                                   - math.lgamma(y[i] + 1.0)) for i in range(n))
        n_par = k
    else:
        phi = 1.0
        deviance = _negbin_deviance(y, mu, w, alpha)
        null_dev = _negbin_deviance(y, mu0, w, alpha)
        loglik = negbin_loglik(beta, X, y, alpha, off)
        n_par = k + 1
    ses = [math.sqrt(max(0.0, phi * cov[a][a])) for a in range(k)]
    zs = [(beta[a] / ses[a]) if ses[a] > 0.0 else 0.0 for a in range(k)]
    aic = -2.0 * loglik + 2.0 * n_par
    bic = -2.0 * loglik + n_par * math.log(n)
    return GLMResult(family=family, link=link, coefficients=beta, names=labels,
                     std_errors=ses, z_values=zs, fitted=mu, loglik=loglik,
                     deviance=deviance, null_deviance=null_dev, aic=aic,
                     bic=bic, dispersion=(phi if family == "gaussian" else alpha),
                     iterations=iterations, converged=converged, n_obs=n,
                     covariance=[[phi * cov[a][b] for b in range(k)]
                                 for a in range(k)],
                     offset=list(off))


def linear_glm(x: Any, y: Any, *, weights: Optional[Sequence[float]] = None,
               offset: Optional[Sequence[float]] = None,
               names: Optional[Sequence[str]] = None) -> GLMResult:
    """Ordinary (or weighted) least squares as a Gaussian GLM with an identity link."""
    X, yy, off, w = _glm_inputs(x, y, offset, weights)
    beta, mu, cov, it, conv = _irls(X, yy, off, w, "gaussian", 0.0, 1, 0.0)
    return _finish_glm("gaussian", "identity", X, yy, off, w, beta, mu, cov,
                       it, conv, 0.0, names)


def poisson_glm(x: Any, y: Any, *, offset: Optional[Sequence[float]] = None,
                weights: Optional[Sequence[float]] = None,
                max_iter: int = 100, tol: float = 1e-12,
                names: Optional[Sequence[str]] = None) -> GLMResult:
    """Poisson regression with a log link, fitted by IRLS (Fisher scoring)."""
    X, yy, off, w = _glm_inputs(x, y, offset, weights)
    for v in yy:
        if v < 0.0:
            raise TravelDemandError("Poisson counts must be non-negative.")
    beta, mu, cov, it, conv = _irls(X, yy, off, w, "poisson", 0.0,
                                    _positive_int(max_iter, "max_iter"),
                                    _non_negative(tol, "tol"))
    return _finish_glm("poisson", "log", X, yy, off, w, beta, mu, cov, it,
                       conv, 0.0, names)


def negbin_glm(x: Any, y: Any, *, alpha: Optional[float] = None,
               offset: Optional[Sequence[float]] = None,
               weights: Optional[Sequence[float]] = None,
               max_iter: int = 100, tol: float = 1e-12,
               alpha_bounds: Tuple[float, float] = (1e-8, 1000.0),
               names: Optional[Sequence[str]] = None) -> GLMResult:
    """NB2 regression; ``alpha=None`` finds the dispersion by profile likelihood."""
    X, yy, off, w = _glm_inputs(x, y, offset, weights)
    for v in yy:
        if v < 0.0:
            raise TravelDemandError("negative-binomial counts must be non-negative.")
    lo = _positive(alpha_bounds[0], "alpha_bounds[0]")
    hi = _positive(alpha_bounds[1], "alpha_bounds[1]")
    if hi <= lo:
        raise TravelDemandError("alpha_bounds must be increasing.")
    iters = _positive_int(max_iter, "max_iter")
    ttol = _non_negative(tol, "tol")

    def profile(log_a: float) -> float:
        a = math.exp(log_a)
        b, _mu, _cov, _it, _conv = _irls(X, yy, off, w, "negbin", a, iters, ttol)
        return negbin_loglik(b, X, yy, a, off)

    if alpha is None:
        log_a, _best = _golden_section_max(profile, math.log(lo), math.log(hi))
        a_hat = math.exp(log_a)
        if a_hat <= lo * 1.000001 or a_hat >= hi * 0.999999:
            a_hat = min(max(a_hat, lo), hi)     # dispersion pinned at a bound
    else:
        a_hat = _positive(alpha, "alpha")
    beta, mu, cov, it, conv = _irls(X, yy, off, w, "negbin", a_hat, iters, ttol)
    return _finish_glm("negbin", "log", X, yy, off, w, beta, mu, cov, it, conv,
                       a_hat, names)


def negbin_dispersion_profile(x: Any, y: Any, alphas: Sequence[float], *,
                              offset: Optional[Sequence[float]] = None,
                              weights: Optional[Sequence[float]] = None,
                              max_iter: int = 100,
                              tol: float = 1e-12) -> List[Tuple[float, float]]:
    """Profile log-likelihood of the NB dispersion: ``(alpha, loglik)`` pairs."""
    X, yy, off, w = _glm_inputs(x, y, offset, weights)
    grid = _as_floats(alphas, "alphas")
    if not grid:
        raise TravelDemandError("alphas must hold at least one value.")
    out: List[Tuple[float, float]] = []
    for a in grid:
        av = _positive(a, "alpha")
        beta, _mu, _cov, _it, _conv = _irls(X, yy, off, w, "negbin", av,
                                            _positive_int(max_iter, "max_iter"),
                                            _non_negative(tol, "tol"))
        out.append((av, negbin_loglik(beta, X, yy, av, off)))
    return out


def glm_predict(model: GLMResult, x: Any,
                offset: Optional[Sequence[float]] = None) -> Vector:
    """Predicted means for new design rows under a fitted GLM."""
    if not isinstance(model, GLMResult):
        raise TravelDemandError("model must be a GLMResult.")
    X = _as_matrix(x, "X")
    k = len(model.coefficients)
    if len(X[0]) != k:
        raise TravelDemandError("X has %d columns for %d coefficients."
                                % (len(X[0]), k))
    off = [0.0] * len(X) if offset is None else _as_floats(offset, "offset")
    if len(off) != len(X):
        raise TravelDemandError("offset has %d values for %d rows."
                                % (len(off), len(X)))
    out: Vector = []
    for i in range(len(X)):
        eta = off[i] + math.fsum(X[i][a] * model.coefficients[a] for a in range(k))
        out.append(eta if model.link == "identity" else math.exp(min(700.0, eta)))
    return out


@dataclass
class TripEnds:
    """Productions and attractions for one purpose."""

    productions: Vector
    attractions: Vector
    purpose: str = "hbw"

    def __post_init__(self) -> None:
        self.productions = _as_floats(self.productions, "productions")
        self.attractions = _as_floats(self.attractions, "attractions")
        for v in self.productions + self.attractions:
            if v < 0.0:
                raise TravelDemandError("trip ends must be non-negative.")
        if not self.productions or not self.attractions:
            raise TravelDemandError("trip ends need at least one zone.")
        self.purpose = str(self.purpose)

    def production_total(self) -> float:
        """Sum of the productions."""
        return math.fsum(self.productions)

    def attraction_total(self) -> float:
        """Sum of the attractions."""
        return math.fsum(self.attractions)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the trip ends."""
        return {"purpose": self.purpose,
                "productions": list(self.productions),
                "attractions": list(self.attractions),
                "productionTotal": self.production_total(),
                "attractionTotal": self.attraction_total()}


def trip_ends(zones: ZoneSystem, production_rates: Dict[str, float],
              attraction_rates: Dict[str, float], *, purpose: str = "hbw",
              production_intercept: float = 0.0,
              attraction_intercept: float = 0.0) -> TripEnds:
    """Productions and attractions from per-attribute rates over a zone system."""
    p = trip_rate_model(zones, production_rates, production_intercept)
    a = trip_rate_model(zones, attraction_rates, attraction_intercept)
    return TripEnds(productions=[max(0.0, v) for v in p],
                    attractions=[max(0.0, v) for v in a], purpose=purpose)


@dataclass
class BalanceResult:
    """Balanced trip ends plus the factor that was applied and to which side."""

    productions: Vector
    attractions: Vector
    factor: float
    method: str
    total: float
    # ``factor`` is the ATTRACTIONS factor, which is the only one that moves for
    # the default method.  ``method='average'`` applies two, and reporting only
    # one of them left the productions factor unrecoverable, so both are carried.
    production_factor: float = 1.0
    attraction_factor: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the balance."""
        return {"productions": list(self.productions),
                "attractions": list(self.attractions),
                "factor": self.factor, "method": self.method,
                "productionFactor": self.production_factor,
                "attractionFactor": self.attraction_factor,
                "total": self.total}


def balance_trip_ends(productions: Sequence[float],
                      attractions: Sequence[float],
                      method: str = "attractions") -> BalanceResult:
    """Scale one (or both) sets of trip ends so the two totals agree exactly."""
    p = _as_floats(productions, "productions")
    a = _as_floats(attractions, "attractions")
    for v in p + a:
        if v < 0.0:
            raise TravelDemandError("trip ends must be non-negative.")
    if not p or not a:
        raise TravelDemandError("trip ends need at least one zone.")
    how = str(method).lower()
    tp, ta = math.fsum(p), math.fsum(a)
    if how == "none":
        return BalanceResult(list(p), list(a), 1.0, how, tp, 1.0, 1.0)
    if how == "attractions":
        if ta == 0.0:
            raise TravelDemandError("cannot scale zero attractions to match.")
        f = tp / ta
        return BalanceResult(list(p), [v * f for v in a], f, how, tp, 1.0, f)
    if how == "productions":
        if tp == 0.0:
            raise TravelDemandError("cannot scale zero productions to match.")
        f = ta / tp
        return BalanceResult([v * f for v in p], list(a), f, how, ta, f, 1.0)
    if how == "average":
        if tp == 0.0 or ta == 0.0:
            raise TravelDemandError("cannot average against a zero total.")
        target = 0.5 * (tp + ta)
        fp, fa = target / tp, target / ta
        return BalanceResult([v * fp for v in p], [v * fa for v in a],
                             fa, how, target, fp, fa)
    raise TravelDemandError(
        "unknown balance method %r (use attractions / productions / average / none)."
        % method)


# ===========================================================================
# 3. DETERRENCE FUNCTIONS
# ===========================================================================
# A deterrence function f(c) is the weight a gravity model gives to a trip of
# generalised cost c.  Every family here is decreasing; they differ in the TAIL,
# which is the only thing that matters once the balancing factors have absorbed
# the level.  The power family has a fat tail (too many long trips), the
# exponential a thin one (too few), and the combined / Tanner form -- a power
# times an exponential -- is the compromise that actually fits observed trip
# length distributions, because it is short-trip-suppressing at small c and
# exponentially decaying at large c.
#
# UNITS.  beta carries the inverse unit of the cost.  If you switch from minutes
# to seconds you must divide beta by 60, and then every distribution result is
# bit-for-bit unchanged.  That invariance is checked by the suite and it is the
# reason nothing in this module compares a cost to an absolute epsilon.
#
# The top-log form used here is f(c) = (1 + c)^(-beta) = exp(-beta ln(1 + c)):
# a power law in the LOG of cost, which -- unlike the pure power -- stays finite
# at c = 0 and so can carry an intrazonal cell.
# ---------------------------------------------------------------------------
class DeterrenceFunction:
    """Deterrence weight ``f(cost)``: the base wraps a callable, subclasses are exact."""

    name = "custom"

    def __init__(self, func: Optional[Callable[[float], float]] = None, *,
                 name: Optional[str] = None,
                 parameters: Optional[Dict[str, float]] = None) -> None:
        if func is not None and not callable(func):
            raise TravelDemandError("func must be callable.")
        self._func = func
        if name is not None:
            self.name = str(name)
        self.parameters: Dict[str, float] = {
            str(k): _as_float(v, str(k))
            for k, v in sorted(dict(parameters or {}).items())}

    def value(self, cost: float) -> float:
        """Deterrence weight at one generalised cost."""
        c = _non_negative(cost, "cost")
        if self._func is None:
            raise TravelDemandError("%s defines no deterrence weight." % self.name)
        v = _as_float(self._func(c), "f(cost)")
        if v < 0.0:
            raise TravelDemandError("deterrence must be non-negative (got %.6g)." % v)
        return v

    def __call__(self, cost: float) -> float:
        return self.value(cost)

    def derivative(self, cost: float) -> float:
        """d f / d cost -- central difference on a step relative to the cost."""
        c = _non_negative(cost, "cost")
        h = 1e-6 * abs(c) if c != 0.0 else 1e-6
        lo = max(0.0, c - h)
        hi = c + h
        return (self.value(hi) - self.value(lo)) / (hi - lo)

    def elasticity(self, cost: float) -> float:
        """Cost elasticity ``c f'(c) / f(c)`` of the deterrence weight."""
        c = _non_negative(cost, "cost")
        f = self.value(c)
        if f == 0.0:
            return 0.0
        return c * self.derivative(c) / f

    def curve(self, lo: float, hi: float, points: int = 50) -> List[Tuple[float, float]]:
        """``(cost, f(cost))`` samples on a uniform grid."""
        a = _non_negative(lo, "lo")
        b = _non_negative(hi, "hi")
        if b <= a:
            raise TravelDemandError("hi must exceed lo.")
        n = _positive_int(points, "points")
        if n == 1:
            return [(a, self.value(a))]
        step = (b - a) / (n - 1)
        return [(a + k * step, self.value(a + k * step)) for k in range(n)]

    def params(self) -> Dict[str, float]:
        """The fitted / supplied parameters of this deterrence function."""
        return dict(self.parameters)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the deterrence function."""
        return {"name": self.name, "parameters": dict(self.parameters)}


class PowerDeterrence(DeterrenceFunction):
    """``f(c) = c ** -beta`` -- the classical inverse-power (fat-tailed) law."""

    name = "power"

    def __init__(self, beta: float = 2.0) -> None:
        super().__init__(name="power",
                         parameters={"beta": _as_float(beta, "beta")})
        self.beta = self.parameters["beta"]

    def value(self, cost: float) -> float:
        """Inverse power of the cost (undefined at zero cost)."""
        c = _non_negative(cost, "cost")
        if c == 0.0:
            raise TravelDemandError(
                "the power deterrence diverges at cost 0; give the intrazonal "
                "cell a positive cost or use the top-log form.")
        return c ** (-self.beta)

    def derivative(self, cost: float) -> float:
        """Analytic derivative ``-beta c ** (-beta - 1)``."""
        c = _positive(cost, "cost")
        return -self.beta * c ** (-self.beta - 1.0)


class ExponentialDeterrence(DeterrenceFunction):
    """``f(c) = exp(-beta c)`` -- the entropy-maximising (thin-tailed) law."""

    name = "exponential"

    def __init__(self, beta: float = 0.1) -> None:
        super().__init__(name="exponential",
                         parameters={"beta": _as_float(beta, "beta")})
        self.beta = self.parameters["beta"]

    def value(self, cost: float) -> float:
        """Exponential decay in the generalised cost."""
        c = _non_negative(cost, "cost")
        return math.exp(min(700.0, -self.beta * c))

    def derivative(self, cost: float) -> float:
        """Analytic derivative ``-beta exp(-beta c)``."""
        return -self.beta * self.value(cost)


class CombinedDeterrence(DeterrenceFunction):
    """``f(c) = c ** alpha * exp(-beta c)`` -- power times exponential."""

    name = "combined"

    def __init__(self, alpha: float = -0.5, beta: float = 0.1) -> None:
        super().__init__(name="combined",
                         parameters={"alpha": _as_float(alpha, "alpha"),
                                     "beta": _as_float(beta, "beta")})
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]

    def value(self, cost: float) -> float:
        """Gamma-shaped deterrence: power at short cost, exponential at long."""
        c = _non_negative(cost, "cost")
        if c == 0.0:
            if self.alpha < 0.0:
                raise TravelDemandError(
                    "the combined deterrence diverges at cost 0 when alpha < 0.")
            return 0.0 if self.alpha > 0.0 else 1.0
        return (c ** self.alpha) * math.exp(min(700.0, -self.beta * c))

    def derivative(self, cost: float) -> float:
        """Analytic derivative ``(alpha / c - beta) f(c)``."""
        c = _positive(cost, "cost")
        return (self.alpha / c - self.beta) * self.value(c)


class TannerDeterrence(CombinedDeterrence):
    """``f(c) = c ** -n * exp(-beta c)`` -- the Tanner sign convention."""

    name = "tanner"

    def __init__(self, n: float = 0.5, beta: float = 0.1) -> None:
        super().__init__(alpha=-_as_float(n, "n"), beta=beta)
        self.name = "tanner"
        self.n = _as_float(n, "n")
        self.parameters = {"beta": self.beta, "n": self.n}


class TopLogDeterrence(DeterrenceFunction):
    """``f(c) = (1 + c) ** -beta`` -- a power law in log cost, finite at c = 0."""

    name = "top_log"

    def __init__(self, beta: float = 1.5) -> None:
        super().__init__(name="top_log",
                         parameters={"beta": _as_float(beta, "beta")})
        self.beta = self.parameters["beta"]

    def value(self, cost: float) -> float:
        """Top-log deterrence, equal to ``exp(-beta ln(1 + c))``."""
        c = _non_negative(cost, "cost")
        return (1.0 + c) ** (-self.beta)

    def derivative(self, cost: float) -> float:
        """Analytic derivative ``-beta (1 + c) ** (-beta - 1)``."""
        c = _non_negative(cost, "cost")
        return -self.beta * (1.0 + c) ** (-self.beta - 1.0)


class LogNormalDeterrence(DeterrenceFunction):
    """``f(c) = exp(-beta ln(1 + c / c0) ** 2)`` -- a log-normal shaped decay."""

    name = "log_normal"

    def __init__(self, beta: float = 0.3, c0: float = 1.0) -> None:
        super().__init__(name="log_normal",
                         parameters={"beta": _as_float(beta, "beta"),
                                     "c0": _positive(c0, "c0")})
        self.beta = self.parameters["beta"]
        self.c0 = self.parameters["c0"]

    def value(self, cost: float) -> float:
        """Log-normal deterrence weight."""
        c = _non_negative(cost, "cost")
        u = math.log(1.0 + c / self.c0)
        return math.exp(min(700.0, -self.beta * u * u))

    def derivative(self, cost: float) -> float:
        """Analytic derivative of the log-normal form."""
        c = _non_negative(cost, "cost")
        u = math.log(1.0 + c / self.c0)
        return self.value(c) * (-2.0 * self.beta * u) / (self.c0 + c)


DETERRENCE_NAMES = ("power", "exponential", "combined", "tanner", "top_log",
                    "log_normal")


def get_deterrence(name: str, **params: float) -> DeterrenceFunction:
    """Build one of DETERRENCE_NAMES by name, passing the parameters through."""
    key = str(name).lower().replace("-", "_").replace(" ", "_")
    if key == "power":
        return PowerDeterrence(**params)
    if key in ("exponential", "exp"):
        return ExponentialDeterrence(**params)
    if key == "combined":
        return CombinedDeterrence(**params)
    if key == "tanner":
        return TannerDeterrence(**params)
    if key in ("top_log", "toplog"):
        return TopLogDeterrence(**params)
    if key in ("log_normal", "lognormal"):
        return LogNormalDeterrence(**params)
    raise TravelDemandError("unknown deterrence %r (use one of %s)."
                            % (name, ", ".join(DETERRENCE_NAMES)))


def deterrence_matrix(costs: CostMatrix, function: DeterrenceFunction) -> Matrix:
    """Apply a deterrence function cell by cell to a cost matrix."""
    cm = _as_matrix(costs, "costs", non_negative=True)
    if not isinstance(function, DeterrenceFunction):
        if callable(function):
            function = DeterrenceFunction(function)
        else:
            raise TravelDemandError("function must be a DeterrenceFunction.")
    return [[function.value(c) for c in row] for row in cm]


@dataclass
class CalibrationResult:
    """A deterrence parameter fitted to reproduce an observed mean trip cost."""

    function: DeterrenceFunction
    beta: float
    mean_cost: float
    target_mean_cost: float
    iterations: int
    converged: bool
    matrix: Matrix
    kind: str

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the calibration."""
        return {"kind": self.kind, "beta": self.beta,
                "meanCost": self.mean_cost,
                "targetMeanCost": self.target_mean_cost,
                "iterations": self.iterations, "converged": self.converged,
                "parameters": self.function.params()}


def calibrate_deterrence(costs: Matrix, productions: Sequence[float],
                         attractions: Sequence[float], target_mean_cost: float,
                         *, kind: str = "exponential",
                         bounds: Tuple[float, float] = (1e-6, 5.0),
                         extra: Optional[Dict[str, float]] = None,
                         tol: float = 1e-12, fit_max_iter: int = 500,
                         max_expand: int = 60) -> CalibrationResult:
    """Fit beta so a doubly-constrained gravity model hits an observed mean cost.

    ``fit_max_iter`` is the Furness budget given to each trial beta.  It matters
    because the mean cost reachable by ANY beta is bounded below by the
    optimal-transport cost of the margins, and a target under that floor can only
    be told apart from a slow fit by whether the fit converges -- so a target very
    close to the floor may need a larger budget than the default before it is
    accepted rather than rejected as unreachable.
    """
    cm = _as_matrix(costs, "costs", non_negative=True)
    p = _as_floats(productions, "productions")
    a = _as_floats(attractions, "attractions")
    target = _positive(target_mean_cost, "target_mean_cost")
    lo = _positive(bounds[0], "bounds[0]")
    hi = _positive(bounds[1], "bounds[1]")
    if hi <= lo:
        raise TravelDemandError("bounds must be increasing.")
    kw = dict(extra or {})
    calls = [0]
    fit_iters = _positive_int(fit_max_iter, "fit_max_iter")
    # The reachable mean cost is DECREASING in beta and bounded below by the
    # optimal-transport cost of the margins, so a target under that floor has no
    # solution at all.  The only honest signal that we have run past the floor is
    # the distribution's own ``converged`` flag: a Furness that has NOT hit its
    # margins reports whatever interior it happened to reach, and at a large beta
    # that interior is systematically too CHEAP.  Taking it at face value used to
    # let the bisection bracket a target no feasible trip table can produce and
    # return a matrix whose row margins were the COLUMN targets.  A failed fit is
    # therefore reported as +inf -- out of range, never a cheaper answer -- which
    # keeps the search inside the region where the distribution actually solves
    # and makes an unreachable target raise instead of silently succeeding.
    best = [-_INF, _INF]                       # widest converged mean cost seen

    def mean_at(beta: float) -> Optional[float]:
        """Mean cost of the fitted table at beta, or None if the fit did not solve."""
        calls[0] += 1
        fn = get_deterrence(kind, beta=beta, **kw)
        try:
            res = doubly_constrained_gravity(p, a, cm, fn, max_iter=fit_iters)
        except TravelDemandError:
            return None                        # deterrence underflowed a whole row
        if not res.converged:
            return None
        mc = mean_trip_cost(res.matrix, cm)
        best[0] = max(best[0], mc)
        best[1] = min(best[1], mc)
        return mc

    def signed(beta: float) -> float:
        """``mean_at - target``, with an unsolvable beta read as "beta too large"."""
        mc = mean_at(beta)
        return -_INF if mc is None else mc - target

    m_lo = mean_at(lo)
    if m_lo is None:
        raise TravelDemandError(
            "the distribution does not solve even at beta = %.6g; the cost matrix "
            "or the trip ends are degenerate." % lo)
    f_lo = m_lo - target
    limit = _positive_int(max_expand, "max_expand")
    # Walk ``hi`` up by doubling until either the mean cost drops through the
    # target (bracketed) or the distribution stops solving.  ``hi_ok`` is the
    # largest beta at which it DID solve, so the bracket handed to the bisection
    # never contains a beta whose answer we do not trust.
    hi_ok, m_ok, probe, expands, stopped = lo, m_lo, hi, 0, False
    while expands <= limit:
        m = mean_at(probe)
        if m is None:
            stopped = True
            break
        hi_ok, m_ok = probe, m
        if m <= target:
            break
        probe *= 2.0
        expands += 1
    if stopped:
        # The solvable range ends in (hi_ok, probe]; find its edge geometrically.
        blo, bhi = hi_ok, probe
        for _ in range(14):                    # geometric: 2**(1/2**14) is ample
            mid = math.sqrt(blo * bhi)
            m = mean_at(mid)
            if m is None:
                bhi = mid
            else:
                blo, m_ok = mid, m
                if m <= target:
                    break
        hi_ok = blo
    hi = hi_ok
    f_hi = m_ok - target
    while f_lo < 0.0 and expands < limit:
        lo *= 0.5
        m = mean_at(lo)
        if m is None:                          # cannot happen for a smaller beta
            break                              # pragma: no cover
        f_lo = m - target
        expands += 1
    if f_lo * f_hi > 0.0 or hi <= lo:
        raise TravelDemandError(
            "target mean cost %.6g is unreachable on beta in [%.6g, %.6g] "
            "(mean cost spans %.6g to %.6g)."
            % (target, lo, hi, best[1], best[0]))
    beta = _bisect_root(signed, lo, hi)
    fn = get_deterrence(kind, beta=beta, **kw)
    res = doubly_constrained_gravity(p, a, cm, fn, max_iter=fit_iters)
    mc = mean_trip_cost(res.matrix, cm)
    return CalibrationResult(function=fn, beta=beta, mean_cost=mc,
                             target_mean_cost=target, iterations=calls[0],
                             converged=(res.converged and
                                        abs(mc - target) <= max(tol, 1e-9) * target),
                             matrix=res.matrix, kind=str(kind))


# ===========================================================================
# 4. TRIP DISTRIBUTION -- GRAVITY, FURNESS AND THE GROWTH-FACTOR FAMILY
# ===========================================================================
# Everything in this section is one idea seen from four sides.
#
# WILSON derived the gravity model by entropy maximisation: of all the trip
# tables that reproduce the observed productions, attractions and total travel
# cost, take the one that can be produced in the largest number of ways.  The
# Lagrangian gives T_ij = A_i O_i B_j D_j exp(-beta c_ij) -- a deterrence
# function times two balancing factor vectors, one per constraint.
#
# FURNESS (iterative proportional fitting) is the algorithm that finds those
# factors: scale the rows to their targets, scale the columns to theirs, repeat.
# It converges to the UNIQUE non-negative matrix that has the required margins
# and the same cross-product ratios (odds ratios) as the seed -- which is
# exactly the entropy-maximising solution relative to that seed.  Two facts fall
# straight out and both are pinned by the suite: on a UNIFORM seed the answer is
# T_ij = O_i D_j / T in one pass and no iteration, and on any seed the 2x2 odds
# ratio T11 T22 / (T12 T21) is invariant.
#
# The GROWTH-FACTOR family (uniform, average, Fratar, Detroit) does the same job
# without a cost matrix: it grows an observed table to new margins.  When the
# growth is UNIFORM all four reproduce ``g * T`` exactly.  Under non-uniform
# growth they part company, and the reason is worth knowing: Detroit multiplies
# each cell by ``E_i F_j / G``, which is SEPARABLE into a row factor times a
# column factor, so it preserves the odds ratios and lands on exactly the same
# entropy-maximising matrix as Furness (the suite pins the two together to
# 1e-9).  Fratar and the average-factor method multiply by ``(L_i + L_j) / 2``
# and ``(E_i + F_j) / 2``, which are NOT separable: they reach the target
# margins but a different interior, several tenths of a trip away on a small
# table.  If you want the entropy-maximising answer, use Furness or Detroit.
#
# INTERVENING OPPORTUNITIES replaces distance with rank: a traveller considers
# destinations in order of closeness and stops with probability L at each one,
# so T_ij is the probability mass between the cumulative opportunities before
# and after j.  COMPETING DESTINATIONS keeps the gravity form but adds a
# centrality term M_j, correcting the gravity model for the fact that a
# destination surrounded by rivals is chosen less often than an isolated one.
# ---------------------------------------------------------------------------
@dataclass
class FurnessResult:
    """A matrix fitted to row and column targets by iterative proportional fitting."""

    matrix: Matrix
    row_factors: Vector
    col_factors: Vector
    iterations: int
    converged: bool
    row_error: float
    col_error: float
    max_error: float
    total: float
    entropy: float

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the fit."""
        return {"iterations": self.iterations, "converged": self.converged,
                "rowError": self.row_error, "colError": self.col_error,
                "maxError": self.max_error, "total": self.total,
                "entropy": self.entropy,
                "rowFactors": list(self.row_factors),
                "colFactors": list(self.col_factors)}


def _margin_error(mat: Matrix, targets_r: Vector, targets_c: Vector,
                  scale: float) -> Tuple[float, float]:
    """Relative row / column margin errors of a matrix against its targets."""
    rs, cs = row_sums(mat), col_sums(mat)
    re = max(abs(rs[i] - targets_r[i]) for i in range(len(rs))) / scale
    ce = max(abs(cs[j] - targets_c[j]) for j in range(len(cs))) / scale
    return re, ce


def furness(seed: Matrix, row_targets: Sequence[float],
            col_targets: Sequence[float], *, max_iter: int = 500,
            tol: float = 1e-14, check_totals: bool = True) -> FurnessResult:
    """Iterative proportional fitting of a seed matrix to row and column targets.

    IPF needs the two margins to agree on the trip total, so by default a
    mismatch is an error.  ``check_totals=False`` resolves it by scaling the
    COLUMN targets onto the row total -- the row targets win and the returned
    ``total`` is the row total.  That asymmetry is deliberate (productions are
    usually the better-surveyed side) but it is a choice, so it is stated here;
    to let the columns win, scale the rows yourself or call
    ``balance_trip_ends``.
    """
    s = _as_matrix(seed, "seed", non_negative=True)
    n, m = len(s), len(s[0])
    R = _as_floats(row_targets, "row_targets")
    C = _as_floats(col_targets, "col_targets")
    if len(R) != n:
        raise TravelDemandError("%d row targets for %d rows." % (len(R), n))
    if len(C) != m:
        raise TravelDemandError("%d column targets for %d columns." % (len(C), m))
    for v in R + C:
        if v < 0.0:
            raise TravelDemandError("targets must be non-negative.")
    iters = _positive_int(max_iter, "max_iter")
    ttol = _non_negative(tol, "tol")
    tr, tc = math.fsum(R), math.fsum(C)
    if tr == 0.0 and tc == 0.0:
        return FurnessResult(zeros_matrix(n, m), [1.0] * n, [1.0] * m, 0, True,
                             0.0, 0.0, 0.0, 0.0, 0.0)
    if check_totals:
        if abs(tr - tc) > 1e-9 * max(tr, tc):
            raise TravelDemandError(
                "row targets total %.10g but column targets total %.10g; "
                "balance them first or pass check_totals=False." % (tr, tc))
    elif tc > 0.0:
        C = [v * tr / tc for v in C]
        tc = tr
    mat = [[0.0 if (R[i] == 0.0 or C[j] == 0.0) else s[i][j]
            for j in range(m)] for i in range(n)]
    for i in range(n):
        if R[i] > 0.0 and math.fsum(mat[i]) == 0.0:
            raise TravelDemandError(
                "row %d wants %.6g trips but every seed cell in it is zero." % (i, R[i]))
    csum = col_sums(mat)
    for j in range(m):
        if C[j] > 0.0 and csum[j] == 0.0:
            raise TravelDemandError(
                "column %d wants %.6g trips but every seed cell in it is zero."
                % (j, C[j]))
    scale = tr if tr > 0.0 else 1.0
    a = [1.0] * n
    b = [1.0] * m
    prev = _INF
    stalls = 0
    converged = False
    used = 0
    for it in range(1, iters + 1):
        used = it
        rs = row_sums(mat)
        for i in range(n):
            if rs[i] > 0.0:
                f = R[i] / rs[i]
                if f != 1.0:
                    a[i] *= f
                    mat[i] = [v * f for v in mat[i]]
        cs = col_sums(mat)
        for j in range(m):
            if cs[j] > 0.0:
                f = C[j] / cs[j]
                if f != 1.0:
                    b[j] *= f
                    for i in range(n):
                        mat[i][j] *= f
        re, ce = _margin_error(mat, R, C, scale)
        err = max(re, ce)
        if err <= ttol:
            converged = True
            break
        # A FLAT error is NOT the floating-point floor.  An ill-conditioned seed
        # (one whose cells span many orders of magnitude) sits on a plateau where
        # the margin error is bit-identical for several sweeps while the balancing
        # factors grind through the range, and then converges.  Giving up on the
        # first non-improving sweep truncated those runs and returned a matrix
        # whose margins were wrong by tens of per cent -- on a 2x2 with a 1e-18
        # off-diagonal it returned the row margins transposed.  So the stall
        # escape is qualified BY MAGNITUDE: bail out only once the error has
        # stopped improving for many consecutive sweeps AND is already far below
        # anything a caller could care about, which is what "at the floor"
        # actually means.  A genuinely infeasible seed now runs to ``max_iter``
        # and reports ``converged=False`` honestly.
        if err >= prev:
            stalls += 1
        else:
            stalls = 0
        if stalls >= _FURNESS_STALL and err <= _FURNESS_FLOOR:
            break
        prev = err
    re, ce = _margin_error(mat, R, C, scale)
    return FurnessResult(matrix=mat, row_factors=a, col_factors=b,
                         iterations=used, converged=converged, row_error=re,
                         col_error=ce, max_error=max(re, ce),
                         total=matrix_total(mat),
                         entropy=relative_entropy(mat, s))


def _flatten_table(table: Any, name: str = "table") -> Tuple[Vector, List[int]]:
    """Flatten a rectangular nested list into a flat vector plus its shape."""
    shape: List[int] = []
    probe = table
    while isinstance(probe, (list, tuple)):
        shape.append(len(probe))
        if len(probe) == 0:
            raise TravelDemandError("%s has an empty axis." % name)
        probe = probe[0]
    if not shape:
        raise TravelDemandError("%s must be a nested list." % name)
    flat: Vector = []

    def walk(node: Any, depth: int) -> None:
        if depth == len(shape):
            flat.append(_non_negative(node, "%s cell" % name))
            return
        if not isinstance(node, (list, tuple)) or len(node) != shape[depth]:
            raise TravelDemandError("%s is ragged at axis %d." % (name, depth))
        for child in node:
            walk(child, depth + 1)

    walk(table, 0)
    return flat, shape


def _unflatten_table(flat: Vector, shape: List[int]) -> Any:
    """Rebuild a nested list of the given shape from a flat vector."""
    if len(shape) == 1:
        return list(flat)
    block = 1
    for d in shape[1:]:
        block *= d
    return [_unflatten_table(flat[k * block:(k + 1) * block], shape[1:])
            for k in range(shape[0])]


def ipf(seed: Any, margins: Sequence[Sequence[float]], *, max_iter: int = 500,
        tol: float = 1e-14, strict: bool = False) -> Any:
    """N-dimensional iterative proportional fitting of a nested-list table.

    This is the ONE fitting routine here that returns a bare nested list rather
    than a result object, so there is no convergence flag to read: with the
    default ``strict=False`` a table whose zero pattern cannot reach the margins
    comes back silently, missing them.  Pass ``strict=True`` to have that raise,
    or use ``furness`` for the 2-D case, which reports ``converged``,
    ``row_error`` and ``col_error``.
    """
    flat, shape = _flatten_table(seed, "seed")
    axes = len(shape)
    targets = [_as_floats(mg, "margins[%d]" % d) for d, mg in enumerate(margins)]
    if len(targets) != axes:
        raise TravelDemandError("%d margins for a %d-dimensional table."
                                % (len(targets), axes))
    for d in range(axes):
        if len(targets[d]) != shape[d]:
            raise TravelDemandError("margin %d has %d entries for axis length %d."
                                    % (d, len(targets[d]), shape[d]))
        for v in targets[d]:
            if v < 0.0:
                raise TravelDemandError("margins must be non-negative.")
    totals = [math.fsum(t) for t in targets]
    if max(totals) > 0.0:
        for d in range(1, axes):
            if abs(totals[d] - totals[0]) > 1e-9 * max(totals[d], totals[0]):
                raise TravelDemandError(
                    "margin totals disagree: axis 0 sums to %.10g, axis %d to %.10g."
                    % (totals[0], d, totals[d]))
    strides = [1] * axes
    for d in range(axes - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    size = len(flat)
    scale = totals[0] if totals[0] > 0.0 else 1.0
    work = list(flat)
    iters = _positive_int(max_iter, "max_iter")
    ttol = _non_negative(tol, "tol")
    err = _INF
    for _it in range(iters):
        for d in range(axes):
            sums = [0.0] * shape[d]
            for k in range(size):
                sums[(k // strides[d]) % shape[d]] += work[k]
            factors = [(targets[d][q] / sums[q]) if sums[q] > 0.0 else 0.0
                       for q in range(shape[d])]
            for k in range(size):
                work[k] *= factors[(k // strides[d]) % shape[d]]
        err = 0.0
        for d in range(axes):
            sums = [0.0] * shape[d]
            for k in range(size):
                sums[(k // strides[d]) % shape[d]] += work[k]
            err = max(err, max(abs(sums[q] - targets[d][q])
                               for q in range(shape[d])) / scale)
        if err <= ttol:
            break
    if strict and err > ttol:
        raise TravelDemandError("IPF did not converge: margin error %.3e > %.3e."
                                % (err, ttol))
    return _unflatten_table(work, shape)


def furness_to_targets(matrix: Matrix,
                       row_targets: Optional[Sequence[float]] = None,
                       col_targets: Optional[Sequence[float]] = None,
                       **kwargs: Any) -> FurnessResult:
    """Fit a matrix to whichever margins are supplied (one of them, or both)."""
    mat = _as_matrix(matrix, "matrix", non_negative=True)
    n, m = len(mat), len(mat[0])
    if row_targets is None and col_targets is None:
        raise TravelDemandError("give at least one set of targets.")
    if row_targets is None:
        row_targets = row_sums(mat)
        ct = _as_floats(col_targets, "col_targets")
        scale = math.fsum(ct) / math.fsum(row_targets) \
            if math.fsum(row_targets) > 0.0 else 0.0
        row_targets = [v * scale for v in row_targets]
    elif col_targets is None:
        col_targets = col_sums(mat)
        rt = _as_floats(row_targets, "row_targets")
        scale = math.fsum(rt) / math.fsum(col_targets) \
            if math.fsum(col_targets) > 0.0 else 0.0
        col_targets = [v * scale for v in col_targets]
    return furness(mat, row_targets, col_targets, **kwargs)


@dataclass
class GravityResult:
    """A distributed trip matrix plus its balancing factors and diagnostics."""

    matrix: Matrix
    origin_factors: Vector
    destination_factors: Vector
    constraint: str
    deterrence: str
    iterations: int
    converged: bool
    row_error: float
    col_error: float
    mean_cost: float
    total: float

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the distribution."""
        return {"constraint": self.constraint, "deterrence": self.deterrence,
                "iterations": self.iterations, "converged": self.converged,
                "rowError": self.row_error, "colError": self.col_error,
                "meanCost": self.mean_cost, "total": self.total,
                "originFactors": list(self.origin_factors),
                "destinationFactors": list(self.destination_factors)}


def _gravity_inputs(productions: Any, attractions: Any,
                    costs: Any) -> Tuple[Vector, Vector, Matrix]:
    """Validate and align productions, attractions and a cost matrix."""
    p = _as_floats(productions, "productions")
    a = _as_floats(attractions, "attractions")
    for v in p + a:
        if v < 0.0:
            raise TravelDemandError("trip ends must be non-negative.")
    cm = _as_matrix(costs, "costs", non_negative=True)
    if len(cm) != len(p):
        raise TravelDemandError("costs has %d rows for %d productions."
                                % (len(cm), len(p)))
    if len(cm[0]) != len(a):
        raise TravelDemandError("costs has %d columns for %d attractions."
                                % (len(cm[0]), len(a)))
    return p, a, cm


def unconstrained_gravity(productions: Sequence[float],
                          attractions: Sequence[float], costs: Matrix,
                          function: DeterrenceFunction,
                          k: float = 1.0) -> GravityResult:
    """``T_ij = k O_i D_j f(c_ij)`` with no balancing at all."""
    p, a, cm = _gravity_inputs(productions, attractions, costs)
    kk = _non_negative(k, "k")
    f = deterrence_matrix(cm, function)
    mat = [[kk * p[i] * a[j] * f[i][j] for j in range(len(a))]
           for i in range(len(p))]
    scale = math.fsum(p) if math.fsum(p) > 0.0 else 1.0
    re, ce = _margin_error(mat, p, a, scale)
    return GravityResult(matrix=mat, origin_factors=[kk] * len(p),
                         destination_factors=[1.0] * len(a),
                         constraint="none", deterrence=function.name,
                         iterations=0, converged=True, row_error=re,
                         col_error=ce, mean_cost=mean_trip_cost(mat, cm),
                         total=matrix_total(mat))


def singly_constrained_gravity(productions: Sequence[float],
                               attractions: Sequence[float], costs: Matrix,
                               function: DeterrenceFunction,
                               constraint: str = "origin") -> GravityResult:
    """Gravity model held to the origin margin (or the destination margin) exactly."""
    p, a, cm = _gravity_inputs(productions, attractions, costs)
    side = str(constraint).lower()
    f = deterrence_matrix(cm, function)
    n, m = len(p), len(a)
    mat = zeros_matrix(n, m)
    if side in ("origin", "production", "row"):
        A: Vector = [1.0] * n
        for i in range(n):
            denom = math.fsum(a[j] * f[i][j] for j in range(m))
            if denom == 0.0:
                if p[i] > 0.0:
                    raise TravelDemandError(
                        "row %d has %.6g productions but no attractive mass."
                        % (i, p[i]))
                continue
            A[i] = 1.0 / denom
            for j in range(m):
                mat[i][j] = A[i] * p[i] * a[j] * f[i][j]
        B: Vector = [1.0] * m
    elif side in ("destination", "attraction", "column", "col"):
        B = [1.0] * m
        for j in range(m):
            denom = math.fsum(p[i] * f[i][j] for i in range(n))
            if denom == 0.0:
                if a[j] > 0.0:
                    raise TravelDemandError(
                        "column %d has %.6g attractions but zero productive mass."
                        % (j, a[j]))
                continue
            B[j] = 1.0 / denom
            for i in range(n):
                mat[i][j] = p[i] * B[j] * a[j] * f[i][j]
        A = [1.0] * n
    else:
        raise TravelDemandError("constraint must be origin or destination.")
    scale = max(math.fsum(p), math.fsum(a), 1.0)
    re, ce = _margin_error(mat, p, a, scale)
    return GravityResult(matrix=mat, origin_factors=A, destination_factors=B,
                         constraint=("origin" if side in
                                     ("origin", "production", "row")
                                     else "destination"),
                         deterrence=function.name, iterations=0,
                         converged=True, row_error=re, col_error=ce,
                         mean_cost=mean_trip_cost(mat, cm),
                         total=matrix_total(mat))


def doubly_constrained_gravity(productions: Sequence[float],
                               attractions: Sequence[float], costs: Matrix,
                               function: DeterrenceFunction, *,
                               max_iter: int = 500, tol: float = 1e-14,
                               balance: str = "attractions") -> GravityResult:
    """Gravity model balanced to BOTH margins -- Furness on the deterrence seed."""
    p, a, cm = _gravity_inputs(productions, attractions, costs)
    bal = balance_trip_ends(p, a, balance)
    p2, a2 = bal.productions, bal.attractions
    f = deterrence_matrix(cm, function)
    seed = [[p2[i] * a2[j] * f[i][j] for j in range(len(a2))]
            for i in range(len(p2))]
    res = furness(seed, p2, a2, max_iter=max_iter, tol=tol)
    return GravityResult(matrix=res.matrix, origin_factors=res.row_factors,
                         destination_factors=res.col_factors,
                         constraint="doubly", deterrence=function.name,
                         iterations=res.iterations, converged=res.converged,
                         row_error=res.row_error, col_error=res.col_error,
                         mean_cost=mean_trip_cost(res.matrix, cm),
                         total=res.total)


def gravity_model(productions: Sequence[float], attractions: Sequence[float],
                  costs: Matrix, function: DeterrenceFunction, *,
                  constraint: str = "doubly", **kwargs: Any) -> GravityResult:
    """Gravity distribution, any flavour: none / origin / destination / doubly."""
    how = str(constraint).lower()
    if how in ("none", "unconstrained", "total"):
        return unconstrained_gravity(productions, attractions, costs, function,
                                     **kwargs)
    if how in ("origin", "production", "row", "singly", "destination",
               "attraction", "column", "col"):
        side = "destination" if how in ("destination", "attraction", "column",
                                        "col") else "origin"
        return singly_constrained_gravity(productions, attractions, costs,
                                          function, side, **kwargs)
    if how in ("doubly", "double", "both"):
        return doubly_constrained_gravity(productions, attractions, costs,
                                          function, **kwargs)
    raise TravelDemandError("unknown constraint %r." % constraint)


@dataclass
class GrowthFactorResult:
    """A matrix grown to new margins without a cost matrix."""

    matrix: Matrix
    method: str
    iterations: int
    converged: bool
    row_error: float
    col_error: float
    total: float

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the growth."""
        return {"method": self.method, "iterations": self.iterations,
                "converged": self.converged, "rowError": self.row_error,
                "colError": self.col_error, "total": self.total}


def uniform_growth(matrix: Matrix, growth: float) -> GrowthFactorResult:
    """Scale an entire matrix by one growth factor."""
    mat = _as_matrix(matrix, "matrix", non_negative=True)
    g = _non_negative(growth, "growth")
    out = matrix_scale(mat, g)
    return GrowthFactorResult(matrix=out, method="uniform", iterations=1,
                              converged=True, row_error=0.0, col_error=0.0,
                              total=matrix_total(out))


def _growth_inputs(matrix: Any, row_targets: Any,
                   col_targets: Any) -> Tuple[Matrix, Vector, Vector]:
    """Validate a base matrix and its future margins."""
    mat = [list(r) for r in _as_matrix(matrix, "matrix", non_negative=True)]
    R = _as_floats(row_targets, "row_targets")
    C = _as_floats(col_targets, "col_targets")
    if len(R) != len(mat):
        raise TravelDemandError("%d row targets for %d rows." % (len(R), len(mat)))
    if len(C) != len(mat[0]):
        raise TravelDemandError("%d column targets for %d columns."
                                % (len(C), len(mat[0])))
    for v in R + C:
        if v < 0.0:
            raise TravelDemandError("targets must be non-negative.")
    tr, tc = math.fsum(R), math.fsum(C)
    if abs(tr - tc) > 1e-9 * max(tr, tc, 1.0):
        raise TravelDemandError("row targets total %.10g but column targets "
                                "total %.10g." % (tr, tc))
    return mat, R, C


def average_growth_factor(matrix: Matrix, row_targets: Sequence[float],
                          col_targets: Sequence[float], *, max_iter: int = 200,
                          tol: float = 1e-13) -> GrowthFactorResult:
    """Average-factor growth: multiply each cell by ``(E_i + F_j) / 2`` and repeat."""
    mat, R, C = _growth_inputs(matrix, row_targets, col_targets)
    n, m = len(mat), len(mat[0])
    scale = max(math.fsum(R), 1.0)
    iters = _positive_int(max_iter, "max_iter")
    ttol = _non_negative(tol, "tol")
    converged = False
    used = 0
    for it in range(1, iters + 1):
        used = it
        rs, cs = row_sums(mat), col_sums(mat)
        E = [(R[i] / rs[i]) if rs[i] > 0.0 else 1.0 for i in range(n)]
        F = [(C[j] / cs[j]) if cs[j] > 0.0 else 1.0 for j in range(m)]
        for i in range(n):
            for j in range(m):
                mat[i][j] *= 0.5 * (E[i] + F[j])
        re, ce = _margin_error(mat, R, C, scale)
        if max(re, ce) <= ttol:
            converged = True
            break
    re, ce = _margin_error(mat, R, C, scale)
    return GrowthFactorResult(matrix=mat, method="average", iterations=used,
                              converged=converged, row_error=re, col_error=ce,
                              total=matrix_total(mat))


def fratar(matrix: Matrix, row_targets: Sequence[float],
           col_targets: Sequence[float], *, max_iter: int = 200,
           tol: float = 1e-13) -> GrowthFactorResult:
    """Fratar growth with the classical locational (L) correction factors."""
    mat, R, C = _growth_inputs(matrix, row_targets, col_targets)
    n, m = len(mat), len(mat[0])
    scale = max(math.fsum(R), 1.0)
    iters = _positive_int(max_iter, "max_iter")
    ttol = _non_negative(tol, "tol")
    converged = False
    used = 0
    for it in range(1, iters + 1):
        used = it
        rs, cs = row_sums(mat), col_sums(mat)
        E = [(R[i] / rs[i]) if rs[i] > 0.0 else 1.0 for i in range(n)]
        F = [(C[j] / cs[j]) if cs[j] > 0.0 else 1.0 for j in range(m)]
        Li: Vector = [1.0] * n
        for i in range(n):
            denom = math.fsum(mat[i][j] * F[j] for j in range(m))
            if denom > 0.0:
                Li[i] = rs[i] / denom
        Lj: Vector = [1.0] * m
        for j in range(m):
            denom = math.fsum(mat[i][j] * E[i] for i in range(n))
            if denom > 0.0:
                Lj[j] = cs[j] / denom
        for i in range(n):
            for j in range(m):
                mat[i][j] *= E[i] * F[j] * 0.5 * (Li[i] + Lj[j])
        re, ce = _margin_error(mat, R, C, scale)
        if max(re, ce) <= ttol:
            converged = True
            break
    re, ce = _margin_error(mat, R, C, scale)
    return GrowthFactorResult(matrix=mat, method="fratar", iterations=used,
                              converged=converged, row_error=re, col_error=ce,
                              total=matrix_total(mat))


def detroit(matrix: Matrix, row_targets: Sequence[float],
            col_targets: Sequence[float], *, max_iter: int = 200,
            tol: float = 1e-13) -> GrowthFactorResult:
    """Detroit growth: ``t_ij E_i F_j / G`` with G the global growth factor."""
    mat, R, C = _growth_inputs(matrix, row_targets, col_targets)
    n, m = len(mat), len(mat[0])
    scale = max(math.fsum(R), 1.0)
    iters = _positive_int(max_iter, "max_iter")
    ttol = _non_negative(tol, "tol")
    converged = False
    used = 0
    for it in range(1, iters + 1):
        used = it
        rs, cs = row_sums(mat), col_sums(mat)
        total = math.fsum(rs)
        if total == 0.0:
            break
        G = math.fsum(R) / total
        E = [(R[i] / rs[i]) if rs[i] > 0.0 else 1.0 for i in range(n)]
        F = [(C[j] / cs[j]) if cs[j] > 0.0 else 1.0 for j in range(m)]
        for i in range(n):
            for j in range(m):
                mat[i][j] *= E[i] * F[j] / G
        re, ce = _margin_error(mat, R, C, scale)
        if max(re, ce) <= ttol:
            converged = True
            break
    re, ce = _margin_error(mat, R, C, scale)
    return GrowthFactorResult(matrix=mat, method="detroit", iterations=used,
                              converged=converged, row_error=re, col_error=ce,
                              total=matrix_total(mat))


def wilson_entropy(matrix: Matrix) -> float:
    """Stirling entropy ``T ln T - sum T_ij ln T_ij`` of a trip table."""
    mat = _as_matrix(matrix, "matrix", non_negative=True)
    total = matrix_total(mat)
    if total == 0.0:
        return 0.0
    cells = math.fsum(v * math.log(v) for row in mat for v in row if v > 0.0)
    return total * math.log(total) - cells


def relative_entropy(matrix: Matrix, prior: Matrix) -> float:
    """``-sum T ln(T / prior)`` -- the objective Furness maximises over the seed."""
    mat = _as_matrix(matrix, "matrix", non_negative=True)
    pri = _as_matrix(prior, "prior", non_negative=True)
    _same_shape(mat, pri, "matrix", "prior")
    terms: Vector = []
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            t, q = mat[i][j], pri[i][j]
            if t == 0.0:
                continue
            if q == 0.0:
                return -_INF
            terms.append(-t * math.log(t / q))
    return math.fsum(terms)


def entropy_maximising_matrix(prior: Matrix, row_targets: Sequence[float],
                              col_targets: Sequence[float], *,
                              costs: Optional[Matrix] = None,
                              beta: Optional[float] = None,
                              **kwargs: Any) -> FurnessResult:
    """Wilson entropy maximisation: the prior times ``exp(-beta c)``, then Furness."""
    pri = _as_matrix(prior, "prior", non_negative=True)
    if (costs is None) != (beta is None):
        raise TravelDemandError("give costs and beta together, or neither.")
    if costs is not None:
        cm = _as_matrix(costs, "costs", non_negative=True)
        _same_shape(pri, cm, "prior", "costs")
        b = _as_float(beta, "beta")
        pri = [[pri[i][j] * math.exp(min(700.0, -b * cm[i][j]))
                for j in range(len(cm[0]))] for i in range(len(cm))]
    return furness(pri, row_targets, col_targets, **kwargs)


def intervening_opportunities(productions: Sequence[float],
                              attractions: Sequence[float], costs: Matrix,
                              l: float) -> GravityResult:
    """Schneider intervening-opportunities distribution; rows match exactly."""
    p, a, cm = _gravity_inputs(productions, attractions, costs)
    lam = _positive(l, "l")
    n, m = len(p), len(a)
    mat = zeros_matrix(n, m)
    for i in range(n):
        order = sorted(range(m), key=lambda j: (cm[i][j], j))
        cumulative = 0.0
        weights: Vector = [0.0] * m
        for j in order:
            nxt = cumulative + a[j]
            weights[j] = (math.exp(min(700.0, -lam * cumulative))
                          - math.exp(min(700.0, -lam * nxt)))
            cumulative = nxt
        denom = math.fsum(weights)
        if denom <= 0.0:
            if p[i] > 0.0:
                raise TravelDemandError(
                    "row %d has %.6g productions but no opportunities." % (i, p[i]))
            continue
        for j in range(m):
            mat[i][j] = p[i] * weights[j] / denom
    scale = max(math.fsum(p), 1.0)
    re, ce = _margin_error(mat, p, a, scale)
    return GravityResult(matrix=mat, origin_factors=[1.0] * n,
                         destination_factors=[1.0] * m, constraint="origin",
                         deterrence="intervening_opportunities", iterations=0,
                         converged=True, row_error=re, col_error=ce,
                         mean_cost=mean_trip_cost(mat, cm),
                         total=matrix_total(mat))


def competing_destinations(productions: Sequence[float],
                           attractions: Sequence[float], costs: Matrix,
                           function: DeterrenceFunction, sigma: float = -0.2,
                           *, constraint: str = "origin",
                           **kwargs: Any) -> GravityResult:
    """Competing destinations: gravity times a centrality term ``M_j^sigma``."""
    p, a, cm = _gravity_inputs(productions, attractions, costs)
    if len(cm) != len(cm[0]):
        raise TravelDemandError(
            "competing destinations needs a square cost matrix (it measures "
            "destination-to-destination centrality).")
    s = _as_float(sigma, "sigma")
    f = deterrence_matrix(cm, function)
    m = len(a)
    centrality: Vector = [
        math.fsum(a[k] * f[j][k] for k in range(m) if k != j) for j in range(m)]
    weights = [(c ** s) if c > 0.0 else 0.0 for c in centrality]
    adjusted = [[f[i][j] * weights[j] for j in range(m)] for i in range(len(p))]
    label = "%s+competing(sigma=%.4g)" % (function.name, s)
    return _gravity_from_weights(p, a, cm, adjusted, constraint, label, **kwargs)


def _gravity_from_weights(p: Vector, a: Vector, cm: Matrix, f: Matrix,
                          constraint: str, label: str,
                          **kwargs: Any) -> GravityResult:
    """Run a gravity model from a pre-computed deterrence weight matrix."""
    weights = [[v for v in row] for row in f]
    n, m = len(p), len(a)
    how = str(constraint).lower()
    if how in ("doubly", "double", "both"):
        bal = balance_trip_ends(p, a, kwargs.pop("balance", "attractions"))
        seed = [[bal.productions[i] * bal.attractions[j] * weights[i][j]
                 for j in range(m)] for i in range(n)]
        res = furness(seed, bal.productions, bal.attractions, **kwargs)
        return GravityResult(matrix=res.matrix, origin_factors=res.row_factors,
                             destination_factors=res.col_factors,
                             constraint="doubly", deterrence=label,
                             iterations=res.iterations, converged=res.converged,
                             row_error=res.row_error, col_error=res.col_error,
                             mean_cost=mean_trip_cost(res.matrix, cm),
                             total=res.total)
    mat = zeros_matrix(n, m)
    A = [1.0] * n
    for i in range(n):
        denom = math.fsum(a[j] * weights[i][j] for j in range(m))
        if denom == 0.0:
            if p[i] > 0.0:
                raise TravelDemandError("row %d has no attractive mass." % i)
            continue
        A[i] = 1.0 / denom
        for j in range(m):
            mat[i][j] = A[i] * p[i] * a[j] * weights[i][j]
    scale = max(math.fsum(p), 1.0)
    re, ce = _margin_error(mat, p, a, scale)
    return GravityResult(matrix=mat, origin_factors=A,
                         destination_factors=[1.0] * m, constraint="origin",
                         deterrence=label, iterations=0, converged=True,
                         row_error=re, col_error=ce,
                         mean_cost=mean_trip_cost(mat, cm),
                         total=matrix_total(mat))


def mean_trip_cost(matrix: Matrix, costs: CostMatrix) -> float:
    """Trip-weighted mean generalised cost of a matrix."""
    mat = _as_matrix(matrix, "matrix", non_negative=True)
    cm = _as_matrix(costs, "costs", non_negative=True)
    _same_shape(mat, cm, "matrix", "costs")
    total = matrix_total(mat)
    if total == 0.0:
        return 0.0
    weighted = math.fsum(mat[i][j] * cm[i][j]
                         for i in range(len(mat)) for j in range(len(mat[0])))
    return weighted / total


@dataclass
class TripLengthDistribution:
    """Trips banded by generalised cost, with the moments of the distribution."""

    edges: Vector
    trips: Vector
    shares: Vector
    mean: float
    std: float
    median: float
    total: float

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the distribution."""
        return {"edges": list(self.edges), "trips": list(self.trips),
                "shares": list(self.shares), "mean": self.mean,
                "std": self.std, "median": self.median, "total": self.total}


def trip_length_distribution(matrix: Matrix, costs: CostMatrix,
                             bins: Any = 10) -> TripLengthDistribution:
    """Band a trip matrix by generalised cost into a trip-length distribution."""
    mat = _as_matrix(matrix, "matrix", non_negative=True)
    cm = _as_matrix(costs, "costs", non_negative=True)
    _same_shape(mat, cm, "matrix", "costs")
    flat = sorted(((cm[i][j], mat[i][j])
                   for i in range(len(mat)) for j in range(len(mat[0]))),
                  key=lambda t: (t[0], -t[1]))
    total = math.fsum(v for (_c, v) in flat)
    if isinstance(bins, int):
        nb = _positive_int(bins, "bins")
        lo = flat[0][0]
        hi = flat[-1][0]
        if hi <= lo:
            hi = lo + max(1.0, abs(lo)) * 1e-9
        step = (hi - lo) / nb
        edges = [lo + k * step for k in range(nb + 1)]
    else:
        edges = _as_floats(bins, "bins")
        if len(edges) < 2:
            raise TravelDemandError("bins needs at least two edges.")
        for k in range(1, len(edges)):
            if edges[k] <= edges[k - 1]:
                raise TravelDemandError("bin edges must be increasing.")
        nb = len(edges) - 1
    counts = [0.0] * nb
    for (c, v) in flat:
        if v == 0.0:
            continue
        if c < edges[0] or c > edges[-1]:
            continue
        k = nb - 1
        for b in range(nb):
            if c < edges[b + 1]:
                k = b
                break
        counts[k] += v
    shares = [(v / total) if total > 0.0 else 0.0 for v in counts]
    mean = mean_trip_cost(mat, cm)
    if total > 0.0:
        var = math.fsum(v * (c - mean) ** 2 for (c, v) in flat) / total
    else:
        var = 0.0
    median = 0.0
    if total > 0.0:
        seen = 0.0
        for (c, v) in flat:
            seen += v
            if seen >= 0.5 * total:
                median = c
                break
    return TripLengthDistribution(edges=edges, trips=counts, shares=shares,
                                  mean=mean, std=math.sqrt(max(0.0, var)),
                                  median=median, total=total)


# ===========================================================================
# 5. MATRIX OPERATIONS -- SECTORING, K-FACTORS AND MATRIX COMPARISON
# ===========================================================================
# The day-to-day work of a demand modeller is not solving the gravity model;
# it is comparing two matrices and explaining the difference.  SECTORING
# collapses a 500-zone table to a 6-sector one you can read on a page.
# K-FACTORS are the confession that the model is wrong in a way you cannot
# explain -- a per-cell multiplier that forces the base year to match, to be
# applied (with a straight face) to the forecast year too.  The COMPARISON
# statistics are the standard audit set; the GEH is the one the highway
# profession actually argues about, because it is the only one whose threshold
# (GEH < 5 on 85 % of counts) is written into scheme approval guidance.
# ---------------------------------------------------------------------------
class SectorSystem:
    """A many-to-one map from zones to sectors, with a deterministic label order."""

    def __init__(self, sectors: Sequence[Any],
                 names: Optional[Sequence[str]] = None) -> None:
        items = [str(s) for s in sectors]
        if not items:
            raise TravelDemandError("a sector system needs at least one zone.")
        order: List[str] = []
        for s in items:
            if s not in order:
                order.append(s)
        self.assignment: List[str] = items
        self.sector_names: List[str] = sorted(order) if names is None else \
            [str(v) for v in names]
        if names is not None:
            missing = sorted(set(order) - set(self.sector_names))
            if missing:
                raise TravelDemandError("sector names omit %s." % ", ".join(missing))
        self._pos = {name: k for k, name in enumerate(self.sector_names)}

    def __len__(self) -> int:
        return len(self.assignment)

    @property
    def n_sectors(self) -> int:
        """Number of sectors."""
        return len(self.sector_names)

    def index(self, label: Any) -> int:
        """Position of a sector label."""
        key = str(label)
        if key not in self._pos:
            raise TravelDemandError("unknown sector %r." % label)
        return self._pos[key]

    def sector_of(self, zone: int) -> str:
        """Sector label of one zone position."""
        k = _non_negative_int(zone, "zone")
        if k >= len(self.assignment):
            raise TravelDemandError("zone %d is outside a %d-zone system."
                                    % (k, len(self.assignment)))
        return self.assignment[k]

    def members(self, label: Any) -> List[int]:
        """Zone positions belonging to a sector, in ascending order."""
        key = str(label)
        self.index(key)
        return [i for i, s in enumerate(self.assignment) if s == key]

    def groups(self) -> List[List[int]]:
        """Zone positions per sector in ``sector_names`` order."""
        return [self.members(name) for name in self.sector_names]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the sector system."""
        return {"sectors": list(self.sector_names),
                "assignment": list(self.assignment),
                "groups": self.groups()}


def aggregate_matrix(matrix: Matrix, row_groups: Sequence[Sequence[int]],
                     col_groups: Optional[Sequence[Sequence[int]]] = None) -> Matrix:
    """Sum a matrix over explicit row and column index groups."""
    mat = _as_matrix(matrix, "matrix")
    n, m = len(mat), len(mat[0])
    rg = [[_non_negative_int(i, "row index") for i in grp] for grp in row_groups]
    cg = rg if col_groups is None else \
        [[_non_negative_int(j, "col index") for j in grp] for grp in col_groups]
    if not rg or not cg:
        raise TravelDemandError("aggregation needs at least one group per axis.")
    for grp in rg:
        for i in grp:
            if i >= n:
                raise TravelDemandError("row index %d is outside %d rows." % (i, n))
    for grp in cg:
        for j in grp:
            if j >= m:
                raise TravelDemandError("column index %d is outside %d columns."
                                        % (j, m))
    out = zeros_matrix(len(rg), len(cg))
    for s, rows in enumerate(rg):
        for t, cols in enumerate(cg):
            out[s][t] = math.fsum(mat[i][j] for i in rows for j in cols)
    return out


def sector_matrix(matrix: Matrix, sectors: Any,
                  col_sectors: Optional[Any] = None) -> TripMatrix:
    """Collapse a zone matrix to a labelled sector-to-sector trip table."""
    mat = _as_matrix(matrix, "matrix")
    rs = sectors if isinstance(sectors, SectorSystem) else SectorSystem(sectors)
    if col_sectors is None:
        cs = rs
    else:
        cs = col_sectors if isinstance(col_sectors, SectorSystem) \
            else SectorSystem(col_sectors)
    if len(rs) != len(mat):
        raise TravelDemandError("%d row sectors for %d rows." % (len(rs), len(mat)))
    if len(cs) != len(mat[0]):
        raise TravelDemandError("%d column sectors for %d columns."
                                % (len(cs), len(mat[0])))
    values = aggregate_matrix(mat, rs.groups(), cs.groups())
    return TripMatrix(values, list(rs.sector_names), list(cs.sector_names),
                      name="sector")


def disaggregate_matrix(sector_totals: Matrix, sectors: Any, seed: Matrix,
                        col_sectors: Optional[Any] = None) -> Matrix:
    """Split a sector matrix back to zones in proportion to a seed matrix."""
    st = _as_matrix(sector_totals, "sector_totals", non_negative=True)
    sd = _as_matrix(seed, "seed", non_negative=True)
    rs = sectors if isinstance(sectors, SectorSystem) else SectorSystem(sectors)
    if col_sectors is None:
        cs = rs
    else:
        cs = col_sectors if isinstance(col_sectors, SectorSystem) \
            else SectorSystem(col_sectors)
    if len(rs) != len(sd) or len(cs) != len(sd[0]):
        raise TravelDemandError("the seed is %dx%d but the sector system covers "
                                "%dx%d zones." % (len(sd), len(sd[0]), len(rs), len(cs)))
    if len(st) != rs.n_sectors or len(st[0]) != cs.n_sectors:
        raise TravelDemandError("sector_totals is %dx%d but there are %d x %d "
                                "sectors." % (len(st), len(st[0]), rs.n_sectors,
                                              cs.n_sectors))
    out = zeros_matrix(len(sd), len(sd[0]))
    rgroups, cgroups = rs.groups(), cs.groups()
    for s, rows in enumerate(rgroups):
        for t, cols in enumerate(cgroups):
            block = math.fsum(sd[i][j] for i in rows for j in cols)
            target = st[s][t]
            if target == 0.0:
                continue
            if block == 0.0:
                if not rows or not cols:
                    raise TravelDemandError("sector block (%d, %d) has no cells."
                                            % (s, t))
                share = target / (len(rows) * len(cols))
                for i in rows:
                    for j in cols:
                        out[i][j] = share
                continue
            for i in rows:
                for j in cols:
                    out[i][j] = target * sd[i][j] / block
    return out


def k_factors(observed: Matrix, modelled: Matrix, *, floor: float = 0.2,
              cap: float = 5.0, default: float = 1.0) -> Matrix:
    """Per-cell correction factors ``observed / modelled``, clipped to a band."""
    obs = _as_matrix(observed, "observed", non_negative=True)
    mod = _as_matrix(modelled, "modelled", non_negative=True)
    _same_shape(obs, mod, "observed", "modelled")
    lo = _positive(floor, "floor")
    hi = _positive(cap, "cap")
    if hi <= lo:
        raise TravelDemandError("cap must exceed floor.")
    d = _positive(default, "default")
    out = zeros_matrix(len(obs), len(obs[0]))
    for i in range(len(obs)):
        for j in range(len(obs[0])):
            if mod[i][j] == 0.0 or obs[i][j] == 0.0:
                out[i][j] = d
            else:
                out[i][j] = min(hi, max(lo, obs[i][j] / mod[i][j]))
    return out


def apply_k_factors(matrix: Matrix, k: Matrix) -> Matrix:
    """Multiply a matrix cell by cell by a K-factor matrix."""
    mat = _as_matrix(matrix, "matrix", non_negative=True)
    kk = _as_matrix(k, "k", non_negative=True)
    _same_shape(mat, kk, "matrix", "k")
    return [[mat[i][j] * kk[i][j] for j in range(len(mat[0]))]
            for i in range(len(mat))]


def estimate_intrazonals(matrix: Matrix, *, method: str = "share",
                         share: float = 0.1, costs: Optional[Matrix] = None,
                         neighbours: int = 3, factor: float = 0.5) -> Matrix:
    """Fill the diagonal of a matrix by a row share or by the nearest-neighbour rule."""
    mat = [list(r) for r in _as_matrix(matrix, "matrix", non_negative=True)]
    if len(mat) != len(mat[0]):
        raise TravelDemandError("intrazonals need a square matrix.")
    n = len(mat)
    how = str(method).lower()
    if how == "share":
        f = _in_unit(share, "share")
        if f >= 1.0:
            raise TravelDemandError("share must be below 1.")
        for i in range(n):
            interzonal = math.fsum(mat[i][j] for j in range(n) if j != i)
            mat[i][i] = interzonal * f / (1.0 - f)
        return mat
    if how == "nearest":
        if costs is None:
            raise TravelDemandError("the nearest-neighbour rule needs costs.")
        cm = _as_matrix(costs, "costs", non_negative=True)
        _same_shape(mat, cm, "matrix", "costs")
        k = _positive_int(neighbours, "neighbours")
        fac = _non_negative(factor, "factor")
        for i in range(n):
            order = sorted((j for j in range(n) if j != i),
                           key=lambda j: (cm[i][j], j))
            picked = order[:min(k, len(order))]
            if not picked:
                mat[i][i] = 0.0
                continue
            mat[i][i] = fac * _mean([mat[i][j] for j in picked])
        return mat
    raise TravelDemandError("unknown intrazonal method %r (use share or nearest)."
                            % method)


def geh(observed: float, modelled: float) -> float:
    """GEH statistic ``sqrt(2 (m - o)^2 / (m + o))``; zero when both are zero."""
    o = _non_negative(observed, "observed")
    m = _non_negative(modelled, "modelled")
    if o == 0.0 and m == 0.0:
        return 0.0
    return math.sqrt(2.0 * (m - o) ** 2 / (m + o))


def geh_matrix(observed: Matrix, modelled: Matrix) -> Matrix:
    """Cell-by-cell GEH of a modelled matrix against an observed one."""
    obs = _as_matrix(observed, "observed", non_negative=True)
    mod = _as_matrix(modelled, "modelled", non_negative=True)
    _same_shape(obs, mod, "observed", "modelled")
    return [[geh(obs[i][j], mod[i][j]) for j in range(len(obs[0]))]
            for i in range(len(obs))]


@dataclass
class MatrixComparison:
    """The standard audit statistics of a modelled matrix against an observed one."""

    n: int
    rmse: float
    pct_rmse: float
    mae: float
    r2: float
    correlation: float
    slope: float
    intercept: float
    mean_observed: float
    mean_modelled: float
    total_observed: float
    total_modelled: float
    max_abs_diff: float
    mean_geh: float
    pct_geh_under_5: float

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the comparison."""
        return {"n": self.n, "rmse": self.rmse, "pctRmse": self.pct_rmse,
                "mae": self.mae, "r2": self.r2,
                "correlation": self.correlation, "slope": self.slope,
                "intercept": self.intercept,
                "meanObserved": self.mean_observed,
                "meanModelled": self.mean_modelled,
                "totalObserved": self.total_observed,
                "totalModelled": self.total_modelled,
                "maxAbsDiff": self.max_abs_diff, "meanGeh": self.mean_geh,
                "pctGehUnder5": self.pct_geh_under_5}


def compare_matrices(observed: Matrix, modelled: Matrix, *,
                     ignore_zeros: bool = False) -> MatrixComparison:
    """RMSE, %RMSE, R^2, regression slope and GEH of one matrix against another.

    ``ignore_zeros`` drops a cell only when it is zero in BOTH matrices -- the
    structurally empty pairs, an origin-destination pair no one can make.  A cell
    that is zero in one matrix and not the other is a real error and is KEPT, so
    the statistics still see every cell the model got wrong.  Note the contrast
    with count-validation practice, which drops cells on the observed side alone.
    """
    obs = _as_matrix(observed, "observed", non_negative=True)
    mod = _as_matrix(modelled, "modelled", non_negative=True)
    _same_shape(obs, mod, "observed", "modelled")
    pairs: List[Tuple[float, float]] = []
    for i in range(len(obs)):
        for j in range(len(obs[0])):
            if ignore_zeros and obs[i][j] == 0.0 and mod[i][j] == 0.0:
                continue
            pairs.append((obs[i][j], mod[i][j]))
    if not pairs:
        raise TravelDemandError("no cells left to compare.")
    n = len(pairs)
    o = [p[0] for p in pairs]
    m = [p[1] for p in pairs]
    mo, mm = _mean(o), _mean(m)
    sse = math.fsum((m[k] - o[k]) ** 2 for k in range(n))
    rmse = math.sqrt(sse / n)
    pct = (100.0 * rmse / mo) if mo > 0.0 else 0.0
    mae = math.fsum(abs(m[k] - o[k]) for k in range(n)) / n
    sso = math.fsum((o[k] - mo) ** 2 for k in range(n))
    ssm = math.fsum((m[k] - mm) ** 2 for k in range(n))
    cross = math.fsum((o[k] - mo) * (m[k] - mm) for k in range(n))
    r2 = 1.0 - sse / sso if sso > 0.0 else (1.0 if sse == 0.0 else 0.0)
    corr = cross / math.sqrt(sso * ssm) if sso > 0.0 and ssm > 0.0 else 0.0
    slope = cross / sso if sso > 0.0 else 0.0
    intercept = mm - slope * mo
    gehs = [geh(o[k], m[k]) for k in range(n)]
    return MatrixComparison(n=n, rmse=rmse, pct_rmse=pct, mae=mae, r2=r2,
                            correlation=corr, slope=slope, intercept=intercept,
                            mean_observed=mo, mean_modelled=mm,
                            total_observed=math.fsum(o),
                            total_modelled=math.fsum(m),
                            max_abs_diff=max(abs(m[k] - o[k]) for k in range(n)),
                            mean_geh=_mean(gehs),
                            pct_geh_under_5=100.0 * sum(1 for g in gehs
                                                        if g < 5.0) / n)


# ===========================================================================
# 6. MODE CHOICE -- MULTINOMIAL AND NESTED LOGIT
# ===========================================================================
# The logit model is the workhorse and it earns its place by being derivable:
# assume each alternative has a systematic utility V_i plus an independent
# Gumbel error, and the probability of choosing i is exactly
# exp(mu V_i) / sum_j exp(mu V_j).  Three consequences matter in practice and
# all three are pinned by the suite:
#
# 1.  TRANSLATION INVARIANCE.  Adding a constant to every utility changes
#     nothing.  Only differences are identified, which is why one alternative
#     specific constant must always be fixed to zero.
# 2.  SCALE.  The probabilities depend on mu and V only through their product,
#     so mu is not separately identified from the coefficients inside V.
#     Doubling every utility and halving mu leaves the model unchanged.
# 3.  IIA.  The ratio P_i / P_j does not depend on any other alternative.  That
#     is the famous red-bus/blue-bus defect: a new alternative steals share from
#     everything in proportion, even from its own near-twin.
#
# NESTED LOGIT is the standard repair.  Correlated alternatives share a nest
# with its own scale mu_m; within the nest the choice is a logit at scale mu_m,
# and the nests compete at the (lower) scale mu on their LOGSUMS -- the expected
# maximum utility of their members.  Consistency with random utility
# maximisation requires mu <= mu_m for every nest, i.e. theta = mu / mu_m in
# (0, 1].  When every mu_m equals mu the nesting is vacuous and the model
# collapses EXACTLY to the MNL; when it does not, alternatives in DIFFERENT
# nests violate IIA while alternatives in the SAME nest still obey it.  The
# suite checks both sides of that statement.
#
# The LOGSUM is not a normalising constant, it is the expected maximum utility
# of the choice set (up to Euler-Mascheroni / mu), which is why it is the right
# measure of accessibility and the right composite cost to feed back into the
# distribution step.  ``monte_carlo_logit`` demonstrates this by drawing Gumbel
# errors and averaging the realised maximum.
# ---------------------------------------------------------------------------
@dataclass
class Alternative:
    """One choice alternative: a name, its attributes and its constant."""

    name: str
    attributes: Dict[str, float] = field(default_factory=dict)
    constant: float = 0.0
    available: bool = True

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.attributes = {str(k): _as_float(v, "attributes[%s]" % k)
                           for k, v in sorted(dict(self.attributes).items())}
        self.constant = _as_float(self.constant, "constant")
        self.available = bool(self.available)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the alternative."""
        return {"name": self.name, "attributes": dict(self.attributes),
                "constant": self.constant, "available": self.available}


def linear_utility(alternative: Alternative,
                   coefficients: Dict[str, float]) -> float:
    """``constant + sum_k beta_k x_k`` -- attributes the mode lacks contribute 0."""
    if not isinstance(alternative, Alternative):
        raise TravelDemandError("alternative must be an Alternative.")
    keys = sorted(dict(coefficients))
    terms = [_as_float(coefficients[k], "coefficients[%s]" % k)
             * alternative.attributes.get(k, 0.0) for k in keys]
    return alternative.constant + math.fsum(terms)


def _utilities_from(alternatives: Any,
                    coefficients: Optional[Dict[str, float]]
                    ) -> Tuple[List[str], Vector, List[bool]]:
    """Normalise alternatives / raw utilities into names, utilities, availability."""
    if isinstance(alternatives, dict):
        names = [str(k) for k in sorted(alternatives)]
        utils = [_as_float(alternatives[k], "utility[%s]" % k) for k in names]
        return names, utils, [True] * len(names)
    if isinstance(alternatives, (str, bytes)) or \
            not isinstance(alternatives, Iterable):
        raise TravelDemandError("alternatives must be a sequence or a mapping.")
    items = list(alternatives)
    if not items:
        raise TravelDemandError("alternatives must hold at least one option.")
    if all(isinstance(a, Alternative) for a in items):
        if coefficients is None:
            raise TravelDemandError("Alternative objects need coefficients.")
        names = [a.name for a in items]
        if len(set(names)) != len(names):
            raise TravelDemandError("alternative names must be unique.")
        return (names, [linear_utility(a, coefficients) for a in items],
                [a.available for a in items])
    utils = _as_floats(items, "utilities")
    return (["alt%d" % k for k in range(len(utils))], utils, [True] * len(utils))


def logit_probabilities(utilities: Sequence[float], scale: float = 1.0,
                        availability: Optional[Sequence[bool]] = None) -> Vector:
    """MNL choice probabilities ``exp(mu V_i) / sum_j exp(mu V_j)``; they sum to 1."""
    v = _as_floats(utilities, "utilities")
    if not v:
        raise TravelDemandError("utilities must hold at least one value.")
    mu = _positive(scale, "scale")
    avail = [True] * len(v) if availability is None else \
        [bool(a) for a in availability]
    if len(avail) != len(v):
        raise TravelDemandError("%d availability flags for %d utilities."
                                % (len(avail), len(v)))
    live = [k for k in range(len(v)) if avail[k]]
    if not live:
        raise TravelDemandError("every alternative is unavailable.")
    top = max(mu * v[k] for k in live)
    ex = [0.0] * len(v)
    for k in live:
        ex[k] = math.exp(mu * v[k] - top)
    denom = math.fsum(ex)
    return [e / denom for e in ex]


def logsum(utilities: Sequence[float], scale: float = 1.0,
           availability: Optional[Sequence[bool]] = None) -> float:
    """Expected maximum utility ``(1 / mu) ln sum_j exp(mu V_j)`` of a choice set."""
    v = _as_floats(utilities, "utilities")
    if not v:
        raise TravelDemandError("utilities must hold at least one value.")
    mu = _positive(scale, "scale")
    if availability is not None:
        live = [v[k] for k in range(len(v)) if bool(availability[k])]
        if len(availability) != len(v):
            raise TravelDemandError("%d availability flags for %d utilities."
                                    % (len(availability), len(v)))
        if not live:
            raise TravelDemandError("every alternative is unavailable.")
        v = live
    return logsumexp(v, mu) / mu


@dataclass
class ModeChoiceResult:
    """Utilities, probabilities and the logsum of a multinomial logit model."""

    names: List[str]
    utilities: Vector
    probabilities: Vector
    logsum: float
    best: str
    scale: float
    availability: List[bool]

    def as_dict(self) -> Dict[str, float]:
        """Name -> probability."""
        return {self.names[k]: self.probabilities[k]
                for k in range(len(self.names))}

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the choice model."""
        return {"names": list(self.names), "utilities": list(self.utilities),
                "probabilities": list(self.probabilities),
                "logsum": self.logsum, "best": self.best, "scale": self.scale,
                "availability": list(self.availability)}


def multinomial_logit(alternatives: Any,
                      coefficients: Optional[Dict[str, float]] = None,
                      scale: float = 1.0) -> ModeChoiceResult:
    """Multinomial logit over Alternatives, raw utilities or a name -> utility map."""
    names, utils, avail = _utilities_from(alternatives, coefficients)
    mu = _positive(scale, "scale")
    probs = logit_probabilities(utils, mu, avail)
    live = [k for k in range(len(utils)) if avail[k]]
    ls = logsum([utils[k] for k in live], mu)
    best = max(live, key=lambda k: (probs[k], -k))
    return ModeChoiceResult(names=names, utilities=utils, probabilities=probs,
                            logsum=ls, best=names[best], scale=mu,
                            availability=avail)


@dataclass
class Nest:
    """A nest of correlated alternatives with its own (higher) logit scale."""

    name: str
    members: Tuple[str, ...]
    scale: float = 1.0
    constant: float = 0.0

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.members = tuple(str(v) for v in self.members)
        if not self.members:
            raise TravelDemandError("nest %s has no members." % self.name)
        if len(set(self.members)) != len(self.members):
            raise TravelDemandError("nest %s repeats a member." % self.name)
        self.scale = _positive(self.scale, "scale")
        self.constant = _as_float(self.constant, "constant")

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the nest."""
        return {"name": self.name, "members": list(self.members),
                "scale": self.scale, "constant": self.constant}


@dataclass
class NestingReport:
    """Whether a nesting structure is consistent with random utility maximisation."""

    consistent: bool
    thetas: Dict[str, float]
    violations: List[str]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the report."""
        return {"consistent": self.consistent, "thetas": dict(self.thetas),
                "violations": list(self.violations), "message": self.message}


def nesting_consistency(nests: Sequence[Nest], scale: float = 1.0,
                        tol: float = _EPS) -> NestingReport:
    """Check ``theta_m = mu / mu_m`` lies in (0, 1] for every nest."""
    mu = _positive(scale, "scale")
    items = list(nests)
    if not items:
        raise TravelDemandError("give at least one nest.")
    thetas: Dict[str, float] = {}
    violations: List[str] = []
    ttol = _non_negative(tol, "tol")
    for nest in items:
        if not isinstance(nest, Nest):
            raise TravelDemandError("nests must hold Nest objects.")
        theta = mu / nest.scale
        thetas[nest.name] = theta
        if theta > 1.0 + ttol:
            violations.append(
                "nest %s has scale %.6g below the upper scale %.6g "
                "(theta = %.6g > 1)" % (nest.name, nest.scale, mu, theta))
    ok = not violations
    msg = ("consistent with random utility maximisation"
           if ok else "; ".join(sorted(violations)))
    return NestingReport(consistent=ok, thetas=thetas,
                         violations=sorted(violations), message=msg)


@dataclass
class NestedLogitResult:
    """Marginal, conditional and joint probabilities of a two-level nested logit."""

    names: List[str]
    utilities: Vector
    probabilities: Vector
    nest_names: List[str]
    nest_of: List[str]
    nest_probabilities: Vector
    conditional_probabilities: Vector
    nest_logsums: Vector
    logsum: float
    best: str
    scale: float
    consistent: bool

    def as_dict(self) -> Dict[str, float]:
        """Name -> probability."""
        return {self.names[k]: self.probabilities[k]
                for k in range(len(self.names))}

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the nested model."""
        return {"names": list(self.names), "utilities": list(self.utilities),
                "probabilities": list(self.probabilities),
                "nestNames": list(self.nest_names),
                "nestOf": list(self.nest_of),
                "nestProbabilities": list(self.nest_probabilities),
                "conditionalProbabilities": list(self.conditional_probabilities),
                "nestLogsums": list(self.nest_logsums), "logsum": self.logsum,
                "best": self.best, "scale": self.scale,
                "consistent": self.consistent}


def nested_logit(alternatives: Any, nests: Sequence[Nest],
                 coefficients: Optional[Dict[str, float]] = None,
                 scale: float = 1.0, *, strict: bool = True) -> NestedLogitResult:
    """Two-level nested logit; with every nest scale equal to mu it IS the MNL."""
    names, utils, avail = _utilities_from(alternatives, coefficients)
    mu = _positive(scale, "scale")
    items = list(nests)
    report = nesting_consistency(items, mu)
    if strict and not report.consistent:
        raise TravelDemandError(report.message)
    pos = {nm: k for k, nm in enumerate(names)}
    nest_of = [""] * len(names)
    for nest in items:
        for member in nest.members:
            if member not in pos:
                raise TravelDemandError("nest %s names unknown alternative %r."
                                        % (nest.name, member))
            if nest_of[pos[member]]:
                raise TravelDemandError("alternative %s is in two nests." % member)
            nest_of[pos[member]] = nest.name
    for k, nm in enumerate(names):
        if not nest_of[k]:
            raise TravelDemandError("alternative %s is in no nest." % nm)
    nest_names = [nest.name for nest in items]
    if len(set(nest_names)) != len(nest_names):
        raise TravelDemandError("nest names must be unique.")
    cond = [0.0] * len(names)
    nest_ls: Vector = []
    upper: Vector = []
    live_nests: List[int] = []
    for nx, nest in enumerate(items):
        idx = [pos[m] for m in nest.members if avail[pos[m]]]
        if not idx:
            nest_ls.append(0.0)
            upper.append(-_INF)
            continue
        live_nests.append(nx)
        ls = logsumexp([utils[k] for k in idx], nest.scale) / nest.scale
        nest_ls.append(ls)
        upper.append(nest.constant + ls)
        top = max(nest.scale * utils[k] for k in idx)
        ex = [math.exp(nest.scale * utils[k] - top) for k in idx]
        denom = math.fsum(ex)
        for q, k in enumerate(idx):
            cond[k] = ex[q] / denom
    if not live_nests:
        raise TravelDemandError("every alternative is unavailable.")
    live_upper = [upper[nx] for nx in live_nests]
    nest_probs_live = logit_probabilities(live_upper, mu)
    nest_probs = [0.0] * len(items)
    for q, nx in enumerate(live_nests):
        nest_probs[nx] = nest_probs_live[q]
    probs = [0.0] * len(names)
    for nx, nest in enumerate(items):
        for member in nest.members:
            k = pos[member]
            probs[k] = nest_probs[nx] * cond[k]
    total_ls = logsumexp(live_upper, mu) / mu
    live_alts = [k for k in range(len(names)) if avail[k]]
    best = max(live_alts, key=lambda k: (probs[k], -k))
    return NestedLogitResult(names=names, utilities=utils, probabilities=probs,
                             nest_names=nest_names, nest_of=nest_of,
                             nest_probabilities=nest_probs,
                             conditional_probabilities=cond,
                             nest_logsums=nest_ls, logsum=total_ls,
                             best=names[best], scale=mu,
                             consistent=report.consistent)


def incremental_logit(base_shares: Sequence[float],
                      delta_utilities: Sequence[float],
                      scale: float = 1.0) -> Vector:
    """Pivot-point logit: base shares reweighted by ``exp(mu dV)`` and renormalised."""
    p0 = _as_floats(base_shares, "base_shares")
    dv = _as_floats(delta_utilities, "delta_utilities")
    if len(p0) != len(dv):
        raise TravelDemandError("%d base shares for %d utility changes."
                                % (len(p0), len(dv)))
    for v in p0:
        if v < 0.0:
            raise TravelDemandError("base shares must be non-negative.")
    mu = _positive(scale, "scale")
    total = math.fsum(p0)
    if total <= 0.0:
        raise TravelDemandError("base shares must not all be zero.")
    top = max(mu * dv[k] for k in range(len(dv)) if p0[k] > 0.0)
    w = [p0[k] * math.exp(mu * dv[k] - top) for k in range(len(p0))]
    denom = math.fsum(w)
    return [v / denom for v in w]


def pivot_point(base_trips: Sequence[float], delta_utilities: Sequence[float],
                scale: float = 1.0, *, total: Optional[float] = None) -> Vector:
    """Incremental logit applied to absolute trips, holding (or resetting) the total."""
    t0 = _as_floats(base_trips, "base_trips")
    for v in t0:
        if v < 0.0:
            raise TravelDemandError("base trips must be non-negative.")
    shares = incremental_logit(t0, delta_utilities, scale)
    target = math.fsum(t0) if total is None else _non_negative(total, "total")
    return [s * target for s in shares]


@dataclass
class ElasticityResult:
    """Direct and cross elasticities of logit shares with respect to an attribute."""

    attribute: str
    direct: Vector
    cross: Matrix
    probabilities: Vector
    values: Vector
    coefficient: float
    scale: float

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the elasticities."""
        return {"attribute": self.attribute, "direct": list(self.direct),
                "cross": [list(r) for r in self.cross],
                "probabilities": list(self.probabilities),
                "values": list(self.values), "coefficient": self.coefficient,
                "scale": self.scale}


def direct_elasticity(probability: float, coefficient: float, value: float,
                      scale: float = 1.0) -> float:
    """Own elasticity of a logit share: ``mu beta x (1 - P)``."""
    p = _in_unit(probability, "probability")
    b = _as_float(coefficient, "coefficient")
    x = _as_float(value, "value")
    mu = _positive(scale, "scale")
    return mu * b * x * (1.0 - p)


def cross_elasticity(probability_j: float, coefficient: float, value_j: float,
                     scale: float = 1.0) -> float:
    """Cross elasticity of any other share: ``-mu beta x_j P_j``."""
    p = _in_unit(probability_j, "probability_j")
    b = _as_float(coefficient, "coefficient")
    x = _as_float(value_j, "value_j")
    mu = _positive(scale, "scale")
    return -mu * b * x * p


def logit_elasticities(probabilities: Sequence[float], coefficient: float,
                       values: Sequence[float], scale: float = 1.0,
                       attribute: str = "cost") -> ElasticityResult:
    """The full direct / cross elasticity matrix of an MNL for one attribute."""
    p = _as_floats(probabilities, "probabilities")
    x = _as_floats(values, "values")
    if len(p) != len(x):
        raise TravelDemandError("%d probabilities for %d values."
                                % (len(p), len(x)))
    b = _as_float(coefficient, "coefficient")
    mu = _positive(scale, "scale")
    direct = [direct_elasticity(p[k], b, x[k], mu) for k in range(len(p))]
    cross = [[direct[i] if i == j else cross_elasticity(p[j], b, x[j], mu)
              for j in range(len(p))] for i in range(len(p))]
    return ElasticityResult(attribute=str(attribute), direct=direct,
                            cross=cross, probabilities=p, values=x,
                            coefficient=b, scale=mu)


def arc_elasticity(q0: float, q1: float, c0: float, c1: float) -> float:
    """Midpoint (arc) elasticity of demand with respect to cost."""
    a0 = _non_negative(q0, "q0")
    a1 = _non_negative(q1, "q1")
    b0 = _positive(c0, "c0")
    b1 = _positive(c1, "c1")
    if a0 + a1 == 0.0:
        raise TravelDemandError("both demands are zero.")
    if b1 == b0:
        raise TravelDemandError("the two costs are identical.")
    return ((a1 - a0) / (a1 + a0)) / ((b1 - b0) / (b1 + b0))


def value_of_time(beta_time: float, beta_cost: float,
                  time_units_per_hour: float = 60.0) -> float:
    """Value of time ``(beta_time / beta_cost) * units per hour``, in money per hour."""
    bt = _as_float(beta_time, "beta_time")
    bc = _as_float(beta_cost, "beta_cost")
    if bc == 0.0:
        raise TravelDemandError("beta_cost must not be zero.")
    per_hour = _positive(time_units_per_hour, "time_units_per_hour")
    return (bt / bc) * per_hour


def composite_cost(utilities: Sequence[float], scale: float = 1.0,
                   cost_coefficient: float = -1.0) -> float:
    """Composite generalised cost of a choice set: ``logsum / beta_cost``."""
    bc = _as_float(cost_coefficient, "cost_coefficient")
    if bc == 0.0:
        raise TravelDemandError("cost_coefficient must not be zero.")
    return logsum(utilities, scale) / bc


def logsum_accessibility(utilities: Any, scale: float = 1.0) -> Any:
    """Logsum accessibility: a float for one choice set, a vector per matrix row."""
    if isinstance(utilities, (str, bytes)) or not isinstance(utilities, Iterable):
        raise TravelDemandError("utilities must be a sequence.")
    items = list(utilities)
    if items and isinstance(items[0], (list, tuple)):
        mat = _as_matrix(items, "utilities")
        return [logsum(row, scale) for row in mat]
    return logsum(items, scale)


def monte_carlo_logit(utilities: Sequence[float], scale: float = 1.0, *,
                      draws: int = 20000, seed: Any = 0) -> Dict[str, Any]:
    """Seeded Gumbel simulation of a logit: shares and the realised mean maximum."""
    v = _as_floats(utilities, "utilities")
    if not v:
        raise TravelDemandError("utilities must hold at least one value.")
    mu = _positive(scale, "scale")
    n = _positive_int(draws, "draws")
    rng = _rng(seed)
    counts = [0] * len(v)
    max_sum = 0.0
    for _ in range(n):
        best_k = 0
        best_u = -_INF
        for k in range(len(v)):
            u = rng.random()
            while u <= 0.0:
                u = rng.random()
            eps = -math.log(-math.log(u)) / mu
            val = v[k] + eps
            if val > best_u:
                best_u, best_k = val, k
        counts[best_k] += 1
        max_sum += best_u
    ls = logsum(v, mu)
    return {"shares": [c / n for c in counts],
            "probabilities": logit_probabilities(v, mu),
            "meanMax": max_sum / n, "logsum": ls,
            "expectedMax": ls + EULER_MASCHERONI / mu,
            "draws": n, "scale": mu}


# ===========================================================================
# 7. DEPARTURE-TIME CHOICE AND PEAK SPREADING
# ===========================================================================
# The fourth choice dimension, and the one that absorbs the pressure when the
# other three cannot.  Vickrey and Small model the cost of arriving at the
# wrong time explicitly: a traveller pays alpha per unit of travel time, beta
# per unit of EARLY schedule delay and gamma per unit of LATE schedule delay,
# with the empirical ordering beta < alpha < gamma -- being late hurts more
# than being early, which is why the peak has a long leading shoulder and a
# short trailing one.  Put those utilities through a logit over time slices and
# you have departure-time choice; feed a capacity penalty back into it and you
# have peak spreading, the mechanism by which a congested network flattens its
# own profile without anyone changing mode or destination.
# ---------------------------------------------------------------------------
@dataclass
class TimeSlice:
    """One modelled time band, in hours from midnight, with an optional capacity."""

    name: str
    start: float
    end: float
    capacity: Optional[float] = None

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.start = _as_float(self.start, "start")
        self.end = _as_float(self.end, "end")
        if self.end <= self.start:
            raise TravelDemandError("time slice %s ends before it starts." % self.name)
        if self.capacity is not None:
            self.capacity = _positive(self.capacity, "capacity")

    def duration(self) -> float:
        """Length of the slice in hours."""
        return self.end - self.start

    def midpoint(self) -> float:
        """Mid-point of the slice in hours from midnight."""
        return 0.5 * (self.start + self.end)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the time slice."""
        return {"name": self.name, "start": self.start, "end": self.end,
                "duration": self.duration(), "midpoint": self.midpoint(),
                "capacity": self.capacity}


def schedule_delay_utility(arrival: float, preferred_arrival: float,
                           travel_time: float, *, alpha: float = 1.0,
                           beta: float = 0.5, gamma: float = 2.0) -> float:
    """Small schedule-delay utility ``-(alpha T + beta SDE + gamma SDL)``."""
    a = _as_float(arrival, "arrival")
    pa = _as_float(preferred_arrival, "preferred_arrival")
    tt = _non_negative(travel_time, "travel_time")
    al = _non_negative(alpha, "alpha")
    be = _non_negative(beta, "beta")
    ga = _non_negative(gamma, "gamma")
    early = max(0.0, pa - a)
    late = max(0.0, a - pa)
    return -(al * tt + be * early + ga * late)


@dataclass
class DepartureTimeResult:
    """Shares and trips by departure slice under a schedule-delay logit."""

    names: List[str]
    utilities: Vector
    shares: Vector
    trips: Vector
    arrivals: Vector
    logsum: float
    peak: str
    total: float

    def as_dict(self) -> Dict[str, float]:
        """Slice name -> share."""
        return {self.names[k]: self.shares[k] for k in range(len(self.names))}

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the departure-time model."""
        return {"names": list(self.names), "utilities": list(self.utilities),
                "shares": list(self.shares), "trips": list(self.trips),
                "arrivals": list(self.arrivals), "logsum": self.logsum,
                "peak": self.peak, "total": self.total}


def departure_time_choice(slices: Sequence[TimeSlice],
                          travel_times: Sequence[float],
                          preferred_arrival: float, *, demand: float = 1.0,
                          alpha: float = 1.0, beta: float = 0.5,
                          gamma: float = 2.0,
                          scale: float = 1.0) -> DepartureTimeResult:
    """Logit departure-time choice over slices with Small schedule-delay costs."""
    items = list(slices)
    if not items:
        raise TravelDemandError("give at least one time slice.")
    for s in items:
        if not isinstance(s, TimeSlice):
            raise TravelDemandError("slices must hold TimeSlice objects.")
    tt = _as_floats(travel_times, "travel_times")
    if len(tt) != len(items):
        raise TravelDemandError("%d travel times for %d slices."
                                % (len(tt), len(items)))
    for v in tt:
        if v < 0.0:
            raise TravelDemandError("travel times must be non-negative.")
    pa = _as_float(preferred_arrival, "preferred_arrival")
    total = _non_negative(demand, "demand")
    arrivals = [items[k].midpoint() + tt[k] for k in range(len(items))]
    utils = [schedule_delay_utility(arrivals[k], pa, tt[k], alpha=alpha,
                                    beta=beta, gamma=gamma)
             for k in range(len(items))]
    probs = logit_probabilities(utils, scale)
    ls = logsum(utils, scale)
    peak = max(range(len(items)), key=lambda k: (probs[k], -k))
    return DepartureTimeResult(names=[s.name for s in items], utilities=utils,
                               shares=probs, trips=[p * total for p in probs],
                               arrivals=arrivals, logsum=ls,
                               peak=items[peak].name, total=total)


def peak_hour_factor(volumes: Sequence[float]) -> float:
    """PHF: hourly volume divided by ``n * peak sub-period volume``."""
    v = _as_floats(volumes, "volumes")
    if not v:
        raise TravelDemandError("volumes must hold at least one sub-period.")
    for x in v:
        if x < 0.0:
            raise TravelDemandError("volumes must be non-negative.")
    peak = max(v)
    if peak == 0.0:
        raise TravelDemandError("every sub-period volume is zero.")
    return math.fsum(v) / (len(v) * peak)


def time_of_day_split(daily_total: float,
                      factors: Dict[str, float]) -> Dict[str, float]:
    """Split a daily total across named periods in proportion to their factors."""
    total = _non_negative(daily_total, "daily_total")
    keys = sorted(dict(factors))
    if not keys:
        raise TravelDemandError("give at least one period factor.")
    weights = [_non_negative(factors[k], "factors[%s]" % k) for k in keys]
    denom = math.fsum(weights)
    if denom == 0.0:
        raise TravelDemandError("period factors must not all be zero.")
    return {keys[k]: total * weights[k] / denom for k in range(len(keys))}


def apply_time_of_day(matrix: Matrix, factor: float,
                      direction_split: Optional[float] = None) -> Matrix:
    """Scale a daily PA matrix to a period, optionally converting PA to OD."""
    mat = _as_matrix(matrix, "matrix", non_negative=True)
    f = _non_negative(factor, "factor")
    scaled = matrix_scale(mat, f)
    if direction_split is None:
        return scaled
    if len(mat) != len(mat[0]):
        raise TravelDemandError("a PA-to-OD conversion needs a square matrix.")
    return matrix_symmetrise(scaled, _in_unit(direction_split, "direction_split"))


def peak_spreading(demand: Sequence[float], capacities: Sequence[float], *,
                   penalty: float = 2.0, scale: float = 1.0,
                   max_iter: int = 200, tol: float = 1e-12) -> Dict[str, Any]:
    """Spread demand out of over-capacity slices with a congestion-penalty logit.

    ``max_iter`` and ``tol`` do NOT steer the solver.  The two-level bisection
    below is exact to the float resolution whatever they say, so ``tol`` only sets
    the threshold the returned ``converged`` flag is judged against and
    ``max_iter`` is validated and otherwise unused (it is kept so the signature
    matches the other iterative routines).  Both are documented here rather than
    quietly removed because a caller reading the signature would reasonably
    expect them to bite.
    """
    # The fixed point is d_k proportional to base_k * exp(-mu pen (d_k/cap_k - 1)_+),
    # and the penalty on slice k depends ONLY on slice k, so the whole system
    # collapses to ONE scalar unknown: the level c = total / Z.  For a given c
    # each slice solves a monotone scalar equation (bisection, exact), and the
    # sum over slices is increasing in c, so a second bisection on c closes it.
    # Damped iteration on the shares was tried first and was WRONG: with a
    # penalty stiff enough to matter it settles into a two-cycle and the gap
    # stalls at a couple of tenths instead of falling to zero.
    d = _as_floats(demand, "demand")
    cap = _as_floats(capacities, "capacities")
    if len(d) != len(cap):
        raise TravelDemandError("%d demands for %d capacities."
                                % (len(d), len(cap)))
    for v in d:
        if v < 0.0:
            raise TravelDemandError("demand must be non-negative.")
    for v in cap:
        if v <= 0.0:
            raise TravelDemandError("capacities must be positive.")
    pen = _non_negative(penalty, "penalty")
    mu = _positive(scale, "scale")
    total = math.fsum(d)
    if total == 0.0:
        return {"demand": list(d), "shares": [0.0] * len(d), "iterations": 0,
                "converged": True, "maxVolumeCapacity": 0.0, "gap": 0.0,
                "total": 0.0, "level": 0.0}
    base = [v / total for v in d]
    n = len(d)
    _positive_int(max_iter, "max_iter")        # validated only; see the docstring
    ttol = _non_negative(tol, "tol")

    def slice_demand(k: int, c: float) -> float:
        """Demand in slice k at level c: the root of d = c b_k exp(-penalty)."""
        top = c * base[k]
        if top <= 0.0 or pen == 0.0:
            return top
        if top <= cap[k]:                      # never over capacity: no penalty
            return top

        def gapfn(x: float) -> float:
            return x - top * math.exp(min(700.0,
                                          -mu * pen * max(0.0, x / cap[k] - 1.0)))

        return _bisect_root(gapfn, 0.0, top)

    def level_total(c: float) -> float:
        return math.fsum(slice_demand(k, c) for k in range(n))

    # BRACKETING the level c is not the caller's iteration budget.  sum_k d_k(c)
    # grows only like log(c) once every slice is over capacity, so a stiff case
    # (demand/capacity 50 at penalty 5) needs c ~ exp(5e7), and no fixed number
    # of doublings will do.  Bounding the doublings by ``max_iter`` made an
    # entirely ordinary 3-slice problem raise a raw "root is not bracketed" out
    # of the bisection at max_iter=1 or 2 while solving fine at 3.  So the search
    # now runs on its own budget to the edge of the float range, and a level that
    # still cannot be bracketed is reported as ``converged: False`` at the best
    # level found -- which is what the result dict was built to say.
    lo, hi = total, total
    used = 0
    reached = level_total(lo) >= total
    at_lo = reached
    while not reached and used < _LEVEL_DOUBLINGS:
        nxt = hi * 2.0
        if nxt == _INF or nxt == hi:
            break
        hi = nxt
        used += 1
        reached = level_total(hi) >= total
    if at_lo or not reached:
        c_star = lo if at_lo else hi
    else:
        c_star = _bisect_root(lambda c: level_total(c) - total, lo, hi)
    current = [slice_demand(k, c_star) for k in range(n)]
    got = math.fsum(current)
    gap = abs(got - total) / total
    return {"demand": current,
            "shares": [v / got if got > 0.0 else 0.0 for v in current],
            "iterations": used + 1, "converged": gap <= max(ttol, 1e-12),
            "maxVolumeCapacity": max(current[k] / cap[k] for k in range(n)),
            "gap": gap, "total": got, "level": c_star}


# ===========================================================================
# 8. THE FOUR-STEP LOOP WITH FEEDBACK
# ===========================================================================
# Generation gives the trip ends, distribution the matrix, mode choice the
# split, assignment the routes -- and assignment changes the costs that
# distribution and mode choice used in the first place.  A model run without
# that loop closed is internally inconsistent: it distributes trips on free-flow
# costs and then reports congested ones.
#
# The loop here is deliberately open at the assignment step: pass any
# ``cost_function`` that maps a flow matrix to a cost matrix (a real assignment
# from the ``traffic_assignment`` module, or the built-in BPR-on-OD-capacity
# default).  Costs -- not flows -- are averaged between iterations by the METHOD
# OF SUCCESSIVE AVERAGES, c_(k+1) = c_k + (c_new - c_k) / k, which is the
# standard damping that stops the loop oscillating between free-flow and
# jammed.  With no congestion at all the loop reproduces itself and the gap is
# EXACTLY zero -- the suite pins that.
#
# MSA buys a guarantee, not speed: its step falls like 1/k, so the gap falls
# like 1/k too and a tolerance of 1e-9 costs hundreds of passes.  Plain
# fixed-point iteration (``msa=False``) reaches the same answer in about a
# dozen passes WHEN the cost map is a contraction, and diverges when it is not.
# Both are offered, the gap is traced per iteration, and ``converged`` is never
# claimed merely because ``max_feedback`` ran out.
#
# The composite cost that distribution sees is the mode-choice LOGSUM divided by
# the cost coefficient, so an improvement to any mode makes every OD pair
# closer, which is the whole point of doing the steps in this order.
# ---------------------------------------------------------------------------
@dataclass
class FourStepConfig:
    """Everything the four-step loop needs: trip ends, costs, modes and tolerances."""

    productions: Vector
    attractions: Vector
    free_cost: Matrix
    deterrence: DeterrenceFunction
    mode_costs: Dict[str, Matrix] = field(default_factory=dict)
    mode_constants: Dict[str, float] = field(default_factory=dict)
    cost_coefficient: float = -0.1
    mode_scale: float = 1.0
    congested_mode: str = "car"
    capacity: Optional[Matrix] = None
    bpr_alpha: float = 0.15
    bpr_beta: float = 4.0
    max_feedback: int = 50
    tol: float = 1e-9
    msa: bool = True
    balance: str = "attractions"
    use_composite_cost: bool = True

    def __post_init__(self) -> None:
        self.productions = _as_floats(self.productions, "productions")
        self.attractions = _as_floats(self.attractions, "attractions")
        self.free_cost = _as_matrix(self.free_cost, "free_cost",
                                    non_negative=True)
        if not isinstance(self.deterrence, DeterrenceFunction):
            raise TravelDemandError("deterrence must be a DeterrenceFunction.")
        if not self.mode_costs:
            self.mode_costs = {self.congested_mode:
                               [list(r) for r in self.free_cost]}
        self.mode_costs = {str(k): _as_matrix(v, "mode_costs[%s]" % k,
                                              non_negative=True)
                           for k, v in sorted(dict(self.mode_costs).items())}
        for name, cm in self.mode_costs.items():
            _same_shape(self.free_cost, cm, "free_cost", "mode_costs[%s]" % name)
        self.mode_constants = {str(k): _as_float(v, "mode_constants[%s]" % k)
                               for k, v in sorted(dict(self.mode_constants).items())}
        for name in sorted(self.mode_constants):
            if name not in self.mode_costs:
                raise TravelDemandError("mode_constants names unknown mode %r." % name)
        self.cost_coefficient = _as_float(self.cost_coefficient,
                                          "cost_coefficient")
        if self.cost_coefficient >= 0.0:
            raise TravelDemandError("cost_coefficient must be negative.")
        self.mode_scale = _positive(self.mode_scale, "mode_scale")
        self.congested_mode = str(self.congested_mode)
        if self.congested_mode not in self.mode_costs:
            raise TravelDemandError("congested_mode %r is not a modelled mode."
                                    % self.congested_mode)
        if self.capacity is not None:
            self.capacity = _as_matrix(self.capacity, "capacity")
            _same_shape(self.free_cost, self.capacity, "free_cost", "capacity")
            for row in self.capacity:
                for v in row:
                    if v <= 0.0:
                        raise TravelDemandError("capacity cells must be positive.")
        self.bpr_alpha = _non_negative(self.bpr_alpha, "bpr_alpha")
        self.bpr_beta = _positive(self.bpr_beta, "bpr_beta")
        self.max_feedback = _positive_int(self.max_feedback, "max_feedback")
        self.tol = _non_negative(self.tol, "tol")

    def modes(self) -> List[str]:
        """Modelled mode names in a deterministic order."""
        return sorted(self.mode_costs)


@dataclass
class FourStepIteration:
    """One pass of the feedback loop and how far it moved."""

    index: int
    gap: float
    cost_gap: float
    mean_cost: float
    total: float
    mode_shares: Dict[str, float]
    max_volume_capacity: float
    # The DISTRIBUTION's own report, carried out of the loop.  ``gap`` only says
    # whether the matrix STOPPED MOVING between passes, and a matrix that is
    # wrong the same way every pass is perfectly stable -- so a stable loop over
    # a distribution that never hit its trip ends used to report gap 0.0 and
    # converged True while missing the productions by 8 per cent.
    distribution_converged: bool = True
    distribution_row_error: float = 0.0
    distribution_col_error: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the iteration."""
        return {"index": self.index, "gap": self.gap, "costGap": self.cost_gap,
                "meanCost": self.mean_cost, "total": self.total,
                "modeShares": dict(self.mode_shares),
                "maxVolumeCapacity": self.max_volume_capacity,
                "distributionConverged": self.distribution_converged,
                "distributionRowError": self.distribution_row_error,
                "distributionColError": self.distribution_col_error}


@dataclass
class FourStepResult:
    """The converged matrix, the mode matrices, the costs and the whole trace."""

    matrix: Matrix
    mode_matrices: Dict[str, Matrix]
    costs: Dict[str, Matrix]
    composite_cost: Matrix
    composite_offset: float
    iterations: List[FourStepIteration]
    converged: bool
    gap: float
    mean_cost: float
    total: float
    mode_shares: Dict[str, float]
    # Did the final distribution actually hit the trip ends?  ``converged`` above
    # is ANDed with this, so it can no longer be True over a matrix that misses
    # its own margins.
    distribution_converged: bool = True
    distribution_row_error: float = 0.0
    distribution_col_error: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the loop."""
        return {"converged": self.converged, "gap": self.gap,
                "meanCost": self.mean_cost, "total": self.total,
                "compositeOffset": self.composite_offset,
                "modeShares": dict(self.mode_shares),
                "distributionConverged": self.distribution_converged,
                "distributionRowError": self.distribution_row_error,
                "distributionColError": self.distribution_col_error,
                "iterations": [it.to_dict() for it in self.iterations],
                "modes": sorted(self.mode_matrices)}


def msa_blend(old: Any, new: Any, k: int) -> Any:
    """Method of successive averages ``old + (new - old) / k`` on vectors or matrices."""
    step = _positive_int(k, "k")
    if isinstance(old, (list, tuple)) and old and \
            isinstance(old[0], (list, tuple)):
        a = _as_matrix(old, "old")
        b = _as_matrix(new, "new")
        _same_shape(a, b, "old", "new")
        return [[a[i][j] + (b[i][j] - a[i][j]) / step for j in range(len(a[0]))]
                for i in range(len(a))]
    a1 = _as_floats(old, "old")
    b1 = _as_floats(new, "new")
    if len(a1) != len(b1):
        raise TravelDemandError("old has %d entries but new has %d."
                                % (len(a1), len(b1)))
    return [a1[i] + (b1[i] - a1[i]) / step for i in range(len(a1))]


def feedback_gap(old: Matrix, new: Matrix) -> float:
    """Relative convergence gap ``sum |new - old| / sum new`` between two matrices."""
    a = _as_matrix(old, "old")
    b = _as_matrix(new, "new")
    _same_shape(a, b, "old", "new")
    denom = math.fsum(abs(v) for row in b for v in row)
    if denom == 0.0:
        return 0.0
    return math.fsum(abs(b[i][j] - a[i][j])
                     for i in range(len(a)) for j in range(len(a[0]))) / denom


def _mode_split(config: FourStepConfig, costs: Dict[str, Matrix],
                matrix: Matrix) -> Tuple[Dict[str, Matrix], Matrix,
                                         Dict[str, float], float]:
    """Per-OD logit mode split, the composite (logsum) cost matrix and its offset."""
    modes = config.modes()
    n, m = len(matrix), len(matrix[0])
    out = {name: zeros_matrix(n, m) for name in modes}
    composite = zeros_matrix(n, m)
    for i in range(n):
        for j in range(m):
            utils = [config.mode_constants.get(name, 0.0)
                     + config.cost_coefficient * costs[name][i][j]
                     for name in modes]
            probs = logit_probabilities(utils, config.mode_scale)
            for q, name in enumerate(modes):
                out[name][i][j] = matrix[i][j] * probs[q]
            composite[i][j] = logsum(utils, config.mode_scale) / \
                config.cost_coefficient
    # The composite cost of a choice set lies BELOW the cheapest member -- that
    # is the logsum bonus of having a choice at all -- so it can fall below
    # zero.  Shift the whole matrix by one constant so the deterrence function
    # sees a non-negative cost.  Under an exponential deterrence a uniform
    # shift multiplies every seed cell by the same exp(-beta k), which the
    # doubly-constrained balancing factors absorb exactly: the fitted matrix is
    # bit-identical.  The offset is reported rather than hidden.
    lowest = min(composite[i][j] for i in range(n) for j in range(m))
    offset = -lowest if lowest < 0.0 else 0.0
    if offset > 0.0:
        composite = [[v + offset for v in row] for row in composite]
    total = matrix_total(matrix)
    shares = {name: (matrix_total(out[name]) / total if total > 0.0 else 0.0)
              for name in modes}
    return out, composite, shares, offset


def _bpr_costs(config: FourStepConfig, flow: Matrix) -> Matrix:
    """Built-in congestion: free-flow cost times the BPR volume/capacity penalty."""
    if config.capacity is None:
        return [list(r) for r in config.mode_costs[config.congested_mode]]
    free = config.mode_costs[config.congested_mode]
    cap = config.capacity
    return [[free[i][j] * (1.0 + config.bpr_alpha
                           * (flow[i][j] / cap[i][j]) ** config.bpr_beta)
             for j in range(len(free[0]))] for i in range(len(free))]


def four_step_model(config: FourStepConfig, *,
                    cost_function: Optional[Callable[[Matrix], Matrix]] = None
                    ) -> FourStepResult:
    """Generation, distribution, mode choice and congestion feedback to convergence."""
    if not isinstance(config, FourStepConfig):
        raise TravelDemandError("config must be a FourStepConfig.")
    modes = config.modes()
    bal = balance_trip_ends(config.productions, config.attractions,
                            config.balance)
    p, a = bal.productions, bal.attractions
    costs = {name: [list(r) for r in config.mode_costs[name]] for name in modes}
    n, m = len(config.free_cost), len(config.free_cost[0])
    matrix = zeros_matrix(n, m)
    mode_matrices: Dict[str, Matrix] = {name: zeros_matrix(n, m) for name in modes}
    composite = [list(r) for r in costs[config.congested_mode]]
    offset = 0.0
    shares: Dict[str, float] = {name: 0.0 for name in modes}
    trace: List[FourStepIteration] = []
    converged = False
    gap = _INF
    cost_gap = _INF
    dist_ok, dist_re, dist_ce = False, _INF, _INF
    for k in range(1, config.max_feedback + 1):
        distribution_cost = composite if config.use_composite_cost \
            else costs[config.congested_mode]
        grav = doubly_constrained_gravity(p, a, distribution_cost,
                                          config.deterrence,
                                          balance="none")
        new_matrix = grav.matrix
        mode_matrices, composite, shares, offset = _mode_split(
            config, costs, new_matrix)
        flow = mode_matrices[config.congested_mode]
        if cost_function is None:
            proposed = _bpr_costs(config, flow)
        else:
            proposed = _as_matrix(cost_function(flow), "cost_function output",
                                  non_negative=True)
            _same_shape(proposed, config.free_cost, "cost_function output",
                        "free_cost")
        old_cost = costs[config.congested_mode]
        blended = msa_blend(old_cost, proposed, k) if config.msa else proposed
        cost_gap = feedback_gap(old_cost, blended)
        gap = feedback_gap(matrix, new_matrix)
        matrix = new_matrix
        costs[config.congested_mode] = blended
        vc = 0.0
        if config.capacity is not None:
            vc = max(flow[i][j] / config.capacity[i][j]
                     for i in range(n) for j in range(m))
        # The distribution's own tolerance (1e-14) is tighter than the loop's,
        # so judge it against THIS loop's ``tol``: the question is whether the
        # matrix hit the trip ends to the accuracy the caller asked for.
        dist_re, dist_ce = grav.row_error, grav.col_error
        dist_ok = grav.converged or max(dist_re, dist_ce) <= config.tol
        trace.append(FourStepIteration(
            index=k, gap=gap, cost_gap=cost_gap,
            mean_cost=mean_trip_cost(matrix, costs[config.congested_mode]),
            total=matrix_total(matrix), mode_shares=dict(shares),
            max_volume_capacity=vc, distribution_converged=dist_ok,
            distribution_row_error=dist_re, distribution_col_error=dist_ce))
        if k > 1 and dist_ok and max(gap, cost_gap) <= config.tol:
            converged = True
            break
    mode_matrices, composite, shares, offset = _mode_split(config, costs, matrix)
    return FourStepResult(matrix=matrix, mode_matrices=mode_matrices,
                          costs=costs, composite_cost=composite,
                          composite_offset=offset,
                          iterations=trace, converged=converged, gap=gap,
                          mean_cost=mean_trip_cost(
                              matrix, costs[config.congested_mode]),
                          total=matrix_total(matrix), mode_shares=shares,
                          distribution_converged=dist_ok,
                          distribution_row_error=dist_re,
                          distribution_col_error=dist_ce)


# ===========================================================================
# 9. OD MATRIX ESTIMATION FROM LINK COUNTS
# ===========================================================================
# The inverse problem.  An assignment turns a matrix into link flows; matrix
# estimation turns link flows back into a matrix.  It is underdetermined and
# always will be: a 100-zone model has ~10 000 unknowns and a good survey has
# 200 counts, so the counts alone can never identify the answer.  Every method
# here therefore combines the counts with a PRIOR matrix, and they differ only
# in how they trade the two off:
#
# * GENERALISED LEAST SQUARES minimises a weighted sum of squared deviations
#   from the prior and from the counts.  With the prior weight set to zero and
#   an invertible assignment it inverts the counts exactly -- which is the
#   oracle the suite uses: assign a known matrix, keep the counts, invert, and
#   demand the original matrix back to machine precision.
# * ME2 (Van Zuylen and Willumsen) maximises entropy subject to the counts, so
#   the estimate has the multiplicative form t_ij * prod_a X_a^(p_ij^a) -- the
#   same family as Furness, with one balancing factor per COUNT instead of one
#   per margin.
# * The PATH-FLOW ESTIMATOR works one level down, on path flows, and adds them
#   back up; it is the only one here that can tell two routes apart.
# * BAYESIAN updating is GLS with the weights read as inverse variances, which
#   also gives a posterior variance per cell -- the honest answer to "how much
#   of this matrix did the counts actually tell me".
# * The SPIESS GRADIENT method never forms a normal-equation matrix at all: it
#   descends the count-error objective multiplicatively, with the analytic
#   optimal step, and so scales to matrices the direct solvers cannot touch.
#
# ASSIGNMENT PROPORTIONS p_ij^a are the bridge: the fraction of the trips from
# i to j that use link a.  They come from an assignment (see the sibling
# ``traffic_assignment`` module) and are held fixed here -- the proportional
# assumption, which is exactly true for a fixed route choice and approximately
# true otherwise.
# ---------------------------------------------------------------------------
@dataclass
class AssignmentProportions:
    """``p_ij^a``: the fraction of the trips from i to j that use each link."""

    links: List[str]
    shape: Tuple[int, int]
    proportions: Dict[str, Dict[Tuple[int, int], float]]

    def __post_init__(self) -> None:
        self.links = [str(v) for v in self.links]
        if not self.links:
            raise TravelDemandError("give at least one link.")
        if len(set(self.links)) != len(self.links):
            raise TravelDemandError("link names must be unique.")
        n = _positive_int(self.shape[0], "shape[0]")
        m = _positive_int(self.shape[1], "shape[1]")
        self.shape = (n, m)
        clean: Dict[str, Dict[Tuple[int, int], float]] = {}
        for name in self.links:
            if name not in self.proportions:
                raise TravelDemandError("link %s has no proportions." % name)
            cells: Dict[Tuple[int, int], float] = {}
            for key in sorted(dict(self.proportions[name])):
                i, j = int(key[0]), int(key[1])
                if not (0 <= i < n and 0 <= j < m):
                    raise TravelDemandError(
                        "link %s names cell (%d, %d) outside a %dx%d matrix."
                        % (name, i, j, n, m))
                p = _non_negative(self.proportions[name][key],
                                  "proportion[%s][%r]" % (name, key))
                if p > 0.0:
                    cells[(i, j)] = p
            clean[name] = cells
        self.proportions = clean

    def cells(self) -> List[Tuple[int, int]]:
        """Every matrix cell in row-major order -- the unknown vector ordering."""
        n, m = self.shape
        return [(i, j) for i in range(n) for j in range(m)]

    def dense(self) -> Matrix:
        """The proportions as a dense ``n_links x (n * m)`` matrix."""
        cells = self.cells()
        pos = {c: k for k, c in enumerate(cells)}
        out = [[0.0] * len(cells) for _ in self.links]
        for a, name in enumerate(self.links):
            for cell, p in sorted(self.proportions[name].items()):
                out[a][pos[cell]] = p
        return out

    def flows(self, matrix: Matrix) -> Vector:
        """Link flows of a matrix, in ``links`` order."""
        mat = _as_matrix(matrix, "matrix")
        if (len(mat), len(mat[0])) != self.shape:
            raise TravelDemandError("matrix is %dx%d but the proportions cover "
                                    "%dx%d." % (len(mat), len(mat[0]),
                                                self.shape[0], self.shape[1]))
        out: Vector = []
        for name in self.links:
            out.append(math.fsum(p * mat[i][j] for (i, j), p in
                                 sorted(self.proportions[name].items())))
        return out

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the proportions."""
        return {"links": list(self.links), "shape": list(self.shape),
                "proportions": {name: [{"cell": list(cell), "p": p}
                                       for cell, p in
                                       sorted(self.proportions[name].items())]
                                for name in self.links}}


def assign_matrix_to_links(matrix: Matrix,
                           assignment: AssignmentProportions) -> Dict[str, float]:
    """Link flows of a trip matrix as a ``link -> flow`` mapping."""
    if not isinstance(assignment, AssignmentProportions):
        raise TravelDemandError("assignment must be AssignmentProportions.")
    vals = assignment.flows(matrix)
    return {assignment.links[a]: vals[a] for a in range(len(assignment.links))}


def link_flows(matrix: Matrix, assignment: AssignmentProportions) -> Vector:
    """Link flows of a trip matrix as a vector in ``assignment.links`` order."""
    if not isinstance(assignment, AssignmentProportions):
        raise TravelDemandError("assignment must be AssignmentProportions.")
    return assignment.flows(matrix)


@dataclass
class CountFit:
    """How well a set of modelled link flows reproduces the observed counts."""

    n: int
    rmse: float
    pct_rmse: float
    r2: float
    slope: float
    mean_geh: float
    pct_geh_under_5: float
    max_abs_diff: float
    total_observed: float
    total_modelled: float

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the count fit."""
        return {"n": self.n, "rmse": self.rmse, "pctRmse": self.pct_rmse,
                "r2": self.r2, "slope": self.slope, "meanGeh": self.mean_geh,
                "pctGehUnder5": self.pct_geh_under_5,
                "maxAbsDiff": self.max_abs_diff,
                "totalObserved": self.total_observed,
                "totalModelled": self.total_modelled}


def count_fit(observed: Sequence[float], modelled: Sequence[float]) -> CountFit:
    """RMSE, R^2, slope and GEH of modelled link flows against observed counts."""
    o = _as_floats(observed, "observed")
    m = _as_floats(modelled, "modelled")
    if len(o) != len(m):
        raise TravelDemandError("%d observed counts for %d modelled flows."
                                % (len(o), len(m)))
    if not o:
        raise TravelDemandError("give at least one count.")
    for v in o + m:
        if v < 0.0:
            raise TravelDemandError("counts must be non-negative.")
    n = len(o)
    mo, mm = _mean(o), _mean(m)
    sse = math.fsum((m[k] - o[k]) ** 2 for k in range(n))
    sso = math.fsum((o[k] - mo) ** 2 for k in range(n))
    cross = math.fsum((o[k] - mo) * (m[k] - mm) for k in range(n))
    rmse = math.sqrt(sse / n)
    gehs = [geh(o[k], m[k]) for k in range(n)]
    return CountFit(n=n, rmse=rmse,
                    pct_rmse=(100.0 * rmse / mo) if mo > 0.0 else 0.0,
                    r2=(1.0 - sse / sso) if sso > 0.0
                    else (1.0 if sse == 0.0 else 0.0),
                    slope=(cross / sso) if sso > 0.0 else 0.0,
                    mean_geh=_mean(gehs),
                    pct_geh_under_5=100.0 * sum(1 for g in gehs if g < 5.0) / n,
                    max_abs_diff=max(abs(m[k] - o[k]) for k in range(n)),
                    total_observed=math.fsum(o), total_modelled=math.fsum(m))


@dataclass
class ODEstimationResult:
    """An estimated matrix, how it was found and how well it fits the counts."""

    matrix: Matrix
    method: str
    iterations: int
    converged: bool
    objective: float
    fit: CountFit
    modelled_counts: Vector
    prior_distance: float
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the estimation."""
        return {"method": self.method, "iterations": self.iterations,
                "converged": self.converged, "objective": self.objective,
                "priorDistance": self.prior_distance,
                "modelledCounts": list(self.modelled_counts),
                "fit": self.fit.to_dict(), "total": matrix_total(self.matrix),
                "detail": dict(self.detail)}


def _estimation_inputs(prior: Any, counts: Any, assignment: Any
                       ) -> Tuple[Matrix, Vector, AssignmentProportions]:
    """Validate a prior matrix, a count vector and the assignment proportions."""
    if not isinstance(assignment, AssignmentProportions):
        raise TravelDemandError("assignment must be AssignmentProportions.")
    pri = _as_matrix(prior, "prior", non_negative=True)
    if (len(pri), len(pri[0])) != assignment.shape:
        raise TravelDemandError("prior is %dx%d but the proportions cover %dx%d."
                                % (len(pri), len(pri[0]), assignment.shape[0],
                                   assignment.shape[1]))
    y = _as_floats(counts, "counts")
    if len(y) != len(assignment.links):
        raise TravelDemandError("%d counts for %d links."
                                % (len(y), len(assignment.links)))
    for v in y:
        if v < 0.0:
            raise TravelDemandError("counts must be non-negative.")
    return pri, y, assignment


def _wrap_estimate(flat: Vector, assignment: AssignmentProportions,
                   prior: Matrix, counts: Vector, method: str, iterations: int,
                   converged: bool, detail: Dict[str, Any]) -> ODEstimationResult:
    """Package a flat cell vector into an ODEstimationResult."""
    n, m = assignment.shape
    mat = [[flat[i * m + j] for j in range(m)] for i in range(n)]
    modelled = assignment.flows(mat)
    obj = 0.5 * math.fsum((modelled[a] - counts[a]) ** 2
                          for a in range(len(counts)))
    dist = math.fsum(abs(mat[i][j] - prior[i][j])
                     for i in range(n) for j in range(m))
    return ODEstimationResult(matrix=mat, method=method, iterations=iterations,
                              converged=converged, objective=obj,
                              fit=count_fit(counts, modelled),
                              modelled_counts=modelled, prior_distance=dist,
                              detail=detail)


def gls_matrix_estimation(prior: Matrix, counts: Sequence[float],
                          assignment: AssignmentProportions, *,
                          prior_weight: float = 1.0, count_weight: float = 1.0,
                          prior_variance: Optional[Matrix] = None,
                          count_variance: Optional[Sequence[float]] = None,
                          non_negative: bool = True) -> ODEstimationResult:
    """Generalised least squares against the prior and the counts together."""
    pri, y, asg = _estimation_inputs(prior, counts, assignment)
    n, m = asg.shape
    cells = asg.cells()
    k = len(cells)
    A = asg.dense()
    w0 = _non_negative(prior_weight, "prior_weight")
    w1 = _non_negative(count_weight, "count_weight")
    if w0 == 0.0 and w1 == 0.0:
        raise TravelDemandError("give a positive weight to the prior or the counts.")
    if prior_variance is None:
        pw = [w0] * k
    else:
        pv = _as_matrix(prior_variance, "prior_variance", non_negative=True)
        _same_shape(pri, pv, "prior", "prior_variance")
        pw = [(w0 / pv[i][j]) if pv[i][j] > 0.0 else 0.0 for (i, j) in cells]
    if count_variance is None:
        cw = [w1] * len(y)
    else:
        cv = _as_floats(count_variance, "count_variance")
        if len(cv) != len(y):
            raise TravelDemandError("%d count variances for %d counts."
                                    % (len(cv), len(y)))
        cw = [(w1 / v) if v > 0.0 else 0.0 for v in cv]
    # A cell with no prior weight that no observed link touches is not
    # identified by anything; it is held at its prior value and reported,
    # rather than being allowed to make the normal equations singular.
    touched = [False] * k
    for a in range(len(y)):
        if cw[a] == 0.0:
            continue
        for c in range(k):
            if A[a][c] != 0.0:
                touched[c] = True
    active = [c for c in range(k) if pw[c] > 0.0 or touched[c]]
    fixed = [c for c in range(k) if c not in set(active)]
    x = [pri[cells[c][0]][cells[c][1]] for c in range(k)]
    if active:
        na = len(active)
        pos = {c: q for q, c in enumerate(active)}
        lhs = [[0.0] * na for _ in range(na)]
        rhs = [0.0] * na
        for q, c in enumerate(active):
            lhs[q][q] += pw[c]
            rhs[q] += pw[c] * pri[cells[c][0]][cells[c][1]]
        for a in range(len(y)):
            if cw[a] == 0.0:
                continue
            support = [c for c in active if A[a][c] != 0.0]
            offset = math.fsum(A[a][c] * x[c] for c in fixed if A[a][c] != 0.0)
            for c in support:
                rhs[pos[c]] += cw[a] * A[a][c] * (y[a] - offset)
                for d in support:
                    lhs[pos[c]][pos[d]] += cw[a] * A[a][c] * A[a][d]
        try:
            sol = solve_linear(lhs, rhs)
        except TravelDemandError:
            raise TravelDemandError(
                "the normal equations are singular: %d counts cannot identify "
                "%d cells on their own -- give prior_weight > 0."
                % (len(y), na))
        for q, c in enumerate(active):
            x[c] = sol[q]
    clipped = 0
    if non_negative:
        for c in range(k):
            if x[c] < 0.0:
                x[c] = 0.0
                clipped += 1
    return _wrap_estimate(x, asg, pri, y, "gls", 1, True,
                          {"priorWeight": w0, "countWeight": w1,
                           "clippedCells": clipped, "unknowns": k,
                           "identifiedCells": len(active),
                           "heldAtPrior": len(fixed), "counts": len(y)})


def me2_estimation(prior: Matrix, counts: Sequence[float],
                   assignment: AssignmentProportions, *, max_iter: int = 500,
                   tol: float = 1e-13) -> ODEstimationResult:
    """Entropy maximisation (ME2): one multiplicative balancing factor per count."""
    pri, y, asg = _estimation_inputs(prior, counts, assignment)
    n, m = asg.shape
    x = [[pri[i][j] for j in range(m)] for i in range(n)]
    links = asg.links
    iters = _positive_int(max_iter, "max_iter")
    ttol = _non_negative(tol, "tol")
    scale = max(math.fsum(y), 1.0)
    converged = False
    used = 0
    for it in range(1, iters + 1):
        used = it
        for a, name in enumerate(links):
            cells = sorted(asg.proportions[name].items())
            if not cells:
                continue
            v = math.fsum(p * x[i][j] for (i, j), p in cells)
            if y[a] == 0.0:
                for (i, j), _p in cells:
                    x[i][j] = 0.0
                continue
            if v <= 0.0:
                raise TravelDemandError(
                    "link %s wants %.6g trips but its prior cells are all zero."
                    % (name, y[a]))
            if all(p == 1.0 for (_c, p) in cells):
                delta = math.log(y[a] / v)     # exact for a 0/1 incidence row
            else:
                def gap(d: float, cs: Any = cells) -> float:
                    return math.fsum(p * x[i][j] * math.exp(min(700.0, p * d))
                                     for (i, j), p in cs) - y[a]
                lo, hi = -50.0, 50.0
                if gap(lo) > 0.0 or gap(hi) < 0.0:
                    delta = 0.0
                else:
                    delta = _bisect_root(gap, lo, hi)
            for (i, j), p in cells:
                x[i][j] *= math.exp(min(700.0, p * delta))
        flows = asg.flows(x)
        err = max(abs(flows[a] - y[a]) for a in range(len(y))) / scale
        if err <= ttol:
            converged = True
            break
    flat = [x[i][j] for i in range(n) for j in range(m)]
    return _wrap_estimate(flat, asg, pri, y, "me2", used, converged,
                          {"unknowns": n * m, "counts": len(y)})


def path_flow_estimator(paths: Sequence[Tuple[Any, Any, Sequence[str], float]],
                        counts: Dict[str, float], shape: Tuple[int, int], *,
                        max_iter: int = 500,
                        tol: float = 1e-13) -> ODEstimationResult:
    """Multiplicative path-flow estimator: balance path flows onto the counts."""
    items = list(paths)
    if not items:
        raise TravelDemandError("give at least one path.")
    n = _positive_int(shape[0], "shape[0]")
    m = _positive_int(shape[1], "shape[1]")
    obs = {str(k): _non_negative(v, "counts[%s]" % k)
           for k, v in sorted(dict(counts).items())}
    if not obs:
        raise TravelDemandError("give at least one count.")
    link_names = sorted(obs)
    flows: Vector = []
    od: List[Tuple[int, int]] = []
    incidence: List[List[str]] = []
    for q, item in enumerate(items):
        i, j, links, f0 = int(item[0]), int(item[1]), list(item[2]), item[3]
        if not (0 <= i < n and 0 <= j < m):
            raise TravelDemandError("path %d leaves the %dx%d matrix." % (q, n, m))
        od.append((i, j))
        incidence.append([str(v) for v in links])
        flows.append(_non_negative(f0, "path %d prior flow" % q))
    prior = zeros_matrix(n, m)
    for q in range(len(items)):
        prior[od[q][0]][od[q][1]] += flows[q]
    users = {name: [q for q in range(len(items)) if name in incidence[q]]
             for name in link_names}
    iters = _positive_int(max_iter, "max_iter")
    ttol = _non_negative(tol, "tol")
    scale = max(math.fsum(obs[name] for name in link_names), 1.0)
    converged = False
    used = 0
    for it in range(1, iters + 1):
        used = it
        for name in link_names:
            idx = users[name]
            if not idx:
                continue
            v = math.fsum(flows[q] * incidence[q].count(name) for q in idx)
            if obs[name] == 0.0:
                for q in idx:
                    flows[q] = 0.0
                continue
            if v <= 0.0:
                raise TravelDemandError(
                    "link %s wants %.6g trips but no path carries any."
                    % (name, obs[name]))
            f = obs[name] / v
            for q in idx:
                flows[q] *= f
        err = 0.0
        for name in link_names:
            v = math.fsum(flows[q] * incidence[q].count(name)
                          for q in users[name])
            err = max(err, abs(v - obs[name]) / scale)
        if err <= ttol:
            converged = True
            break
    mat = zeros_matrix(n, m)
    for q in range(len(items)):
        mat[od[q][0]][od[q][1]] += flows[q]
    modelled = [math.fsum(flows[q] * incidence[q].count(name)
                          for q in users[name]) for name in link_names]
    y = [obs[name] for name in link_names]
    obj = 0.5 * math.fsum((modelled[a] - y[a]) ** 2 for a in range(len(y)))
    dist = math.fsum(abs(mat[i][j] - prior[i][j])
                     for i in range(n) for j in range(m))
    return ODEstimationResult(matrix=mat, method="path_flow", iterations=used,
                              converged=converged, objective=obj,
                              fit=count_fit(y, modelled),
                              modelled_counts=modelled, prior_distance=dist,
                              detail={"paths": len(items),
                                      "pathFlows": list(flows),
                                      "links": link_names})


def bayesian_matrix_update(prior: Matrix, prior_variance: Matrix,
                           counts: Sequence[float],
                           count_variance: Sequence[float],
                           assignment: AssignmentProportions, *,
                           non_negative: bool = True) -> ODEstimationResult:
    """Gaussian Bayesian update of a prior matrix by counts, with posterior variances."""
    pri, y, asg = _estimation_inputs(prior, counts, assignment)
    pv = _as_matrix(prior_variance, "prior_variance", non_negative=True)
    _same_shape(pri, pv, "prior", "prior_variance")
    cv = _as_floats(count_variance, "count_variance")
    if len(cv) != len(y):
        raise TravelDemandError("%d count variances for %d counts."
                                % (len(cv), len(y)))
    for v in cv:
        if v <= 0.0:
            raise TravelDemandError("count variances must be positive.")
    for row in pv:
        for v in row:
            if v <= 0.0:
                raise TravelDemandError("prior variances must be positive.")
    res = gls_matrix_estimation(pri, y, asg, prior_weight=1.0, count_weight=1.0,
                                prior_variance=pv, count_variance=cv,
                                non_negative=non_negative)
    cells = asg.cells()
    A = asg.dense()
    k = len(cells)
    precision = [[0.0] * k for _ in range(k)]
    for c in range(k):
        i, j = cells[c]
        precision[c][c] += (1.0 / pv[i][j]) if pv[i][j] > 0.0 else 0.0
    for a in range(len(y)):
        support = [c for c in range(k) if A[a][c] != 0.0]
        for c in support:
            for d in support:
                precision[c][d] += A[a][c] * A[a][d] / cv[a]
    cov = invert_matrix(precision)
    post_var = [cov[c][c] for c in range(k)]
    n, m = asg.shape
    res.method = "bayesian"
    # MERGE, do not replace: the GLS step already reported how many cells it had
    # to clip, how many the counts identified and how many were held at the
    # prior, and those diagnostics are exactly what tells a user how much of this
    # posterior is data rather than prior.  Overwriting ``detail`` threw them away.
    res.detail = dict(res.detail or {})
    res.detail.update(
        {"posteriorVariance": [[post_var[i * m + j] for j in range(m)]
                               for i in range(n)],
         "varianceReduction": [
             [1.0 - post_var[i * m + j] / pv[i][j]
              if pv[i][j] > 0.0 else 0.0 for j in range(m)]
             for i in range(n)],
         "unknowns": k, "counts": len(y)})
    return res


def spiess_gradient(prior: Matrix, counts: Sequence[float],
                    assignment: AssignmentProportions, *, max_iter: int = 500,
                    tol: float = 1e-12,
                    max_step: float = 1.0) -> ODEstimationResult:
    """Spiess gradient descent on the count error with the analytic optimal step."""
    pri, y, asg = _estimation_inputs(prior, counts, assignment)
    n, m = asg.shape
    x = [[pri[i][j] for j in range(m)] for i in range(n)]
    iters = _positive_int(max_iter, "max_iter")
    ttol = _non_negative(tol, "tol")
    cap = _positive(max_step, "max_step")
    scale = max(math.fsum(y), 1.0)
    converged = False
    used = 0
    obj = _INF
    for it in range(1, iters + 1):
        used = it
        v = asg.flows(x)
        resid = [v[a] - y[a] for a in range(len(y))]
        obj = 0.5 * math.fsum(r * r for r in resid)
        if math.sqrt(2.0 * obj) / scale <= ttol:
            converged = True
            break
        grad = zeros_matrix(n, m)
        for a, name in enumerate(asg.links):
            if resid[a] == 0.0:
                continue
            for (i, j), p in sorted(asg.proportions[name].items()):
                grad[i][j] += resid[a] * p
        direction = [[-x[i][j] * grad[i][j] for j in range(m)] for i in range(n)]
        w = [math.fsum(p * direction[i][j] for (i, j), p in
                       sorted(asg.proportions[name].items()))
             for name in asg.links]
        denom = math.fsum(val * val for val in w)
        if denom <= 0.0:
            converged = True
            break
        lam = -math.fsum(resid[a] * w[a] for a in range(len(y))) / denom
        lam = max(-cap, min(cap, lam))
        moved = 0.0
        for i in range(n):
            for j in range(m):
                new = x[i][j] + lam * direction[i][j]
                if new < 0.0:
                    new = 0.0
                moved = max(moved, abs(new - x[i][j]))
                x[i][j] = new
        if moved == 0.0:
            break
    flat = [x[i][j] for i in range(n) for j in range(m)]
    return _wrap_estimate(flat, asg, pri, y, "spiess", used, converged,
                          {"finalObjective": obj, "unknowns": n * m,
                           "counts": len(y)})


# ===========================================================================
# 10. SEEDED DEMAND SCENARIOS
# ===========================================================================
# Five generators that turn a zone system into a plausible matrix for a named
# situation.  They exist so that the rest of the urban stack has something to
# eat that is neither a toy 3x3 nor a confidential survey: a morning commute
# concentrated on the employment zones, a school run that is short and sharp, a
# stadium emptying into a release profile, and the two demand shocks that
# dominate operational planning -- weather and incidents.
#
# Every generator takes a ``seed`` and every random draw goes through
# ``random.Random`` or a Latin-hypercube design, so a given seed reproduces a
# scenario byte for byte on any interpreter and under any PYTHONHASHSEED.  The
# zone-level variability is drawn from a STRATIFIED design rather than plain
# Monte Carlo, so a 20-zone draw actually covers the range instead of clumping.
# ---------------------------------------------------------------------------
SCENARIO_NAMES = ("commuting_peak", "school", "stadium", "weather", "incident")

_WEATHER = {
    # condition: (demand multiplier, car shift, walk/cycle multiplier)
    "dry": (1.00, 0.00, 1.00),
    "rain": (0.97, 0.04, 0.80),
    "heavy_rain": (0.92, 0.08, 0.55),
    "snow": (0.78, 0.05, 0.40),
    "heat": (0.95, 0.03, 0.75),
    "fog": (0.93, 0.02, 0.70),
}

_ACTIVE_MODES = ("walk", "cycle", "active")


@dataclass
class ScenarioResult:
    """A generated demand scenario: the matrix, its trip ends and what was assumed."""

    name: str
    matrix: Matrix
    productions: Vector
    attractions: Vector
    seed: Any
    notes: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    def total(self) -> float:
        """Total trips in the scenario."""
        return matrix_total(self.matrix)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the scenario."""
        return {"name": self.name, "total": self.total(),
                "productions": list(self.productions),
                "attractions": list(self.attractions),
                "seed": self.seed, "notes": list(self.notes),
                "detail": dict(self.detail)}


def _lognormal_multipliers(n: int, sigma: float, seed: Any) -> Vector:
    """Stratified lognormal multipliers with mean 1, from a Latin hypercube."""
    if sigma == 0.0:
        return [1.0] * n
    pts = _latin_hypercube(n, 1, seed)
    out: Vector = []
    for k in range(n):
        u = min(1.0 - 1e-12, max(1e-12, pts[k][0]))
        # inverse standard normal by the Beasley-Springer-Moro tail-free form
        z = _norm_ppf(u)
        out.append(math.exp(sigma * z - 0.5 * sigma * sigma))
    return out


def _norm_ppf(p: float) -> float:
    """Standard-normal quantile (Acklam rational approximation, ~1e-9 absolute)."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                 + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) \
        * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def commuting_peak_scenario(zones: ZoneSystem, *, seed: Any = 0,
                            trip_rate: float = 1.15, beta: float = 0.08,
                            variability: float = 0.18, speed: float = 0.5,
                            fixed_cost: float = 2.0) -> ScenarioResult:
    """Morning commute: households produce, employment attracts, gravity distributes."""
    if not isinstance(zones, ZoneSystem):
        raise TravelDemandError("zones must be a ZoneSystem.")
    rate = _positive(trip_rate, "trip_rate")
    sigma = _non_negative(variability, "variability")
    costs = zones.cost_matrix(speed=speed, fixed=fixed_cost)
    mult = _lognormal_multipliers(zones.n, sigma, seed)
    hh = zones.vector("households")
    emp = zones.vector("employment")
    if math.fsum(hh) == 0.0 or math.fsum(emp) == 0.0:
        raise TravelDemandError(
            "the commuting scenario needs positive households and employment.")
    p = [hh[k] * rate * mult[k] for k in range(zones.n)]
    a = [emp[k] for k in range(zones.n)]
    fn = get_deterrence("exponential", beta=beta)
    grav = doubly_constrained_gravity(p, a, costs, fn)
    return ScenarioResult(
        name="commuting_peak", matrix=grav.matrix,
        productions=row_sums(grav.matrix), attractions=col_sums(grav.matrix),
        seed=seed,
        notes=["productions = households x %.3f x a stratified lognormal "
               "(sigma = %.3f)" % (rate, sigma),
               "attractions = employment, balanced to the production total",
               "doubly-constrained gravity, exponential deterrence beta = %.4f"
               % beta],
        detail={"meanCost": grav.mean_cost, "iterations": grav.iterations,
                "converged": grav.converged, "multipliers": mult,
                "speed": speed})


def school_scenario(zones: ZoneSystem, *, seed: Any = 0,
                    child_share: float = 0.18, trip_rate: float = 0.9,
                    beta: float = 0.35, school_zones: Optional[Sequence[int]] = None,
                    speed: float = 0.25) -> ScenarioResult:
    """The school run: short trips from every zone to a handful of school zones."""
    if not isinstance(zones, ZoneSystem):
        raise TravelDemandError("zones must be a ZoneSystem.")
    share = _in_unit(child_share, "child_share")
    rate = _positive(trip_rate, "trip_rate")
    pop = zones.vector("population")
    if math.fsum(pop) == 0.0:
        raise TravelDemandError("the school scenario needs a positive population.")
    if school_zones is None:
        take = max(1, zones.n // 3)
        picked = sorted(sorted(range(zones.n),
                               key=lambda k: (-pop[k], k))[:take])
    else:
        picked = sorted({_non_negative_int(k, "school zone") for k in school_zones})
        for k in picked:
            if k >= zones.n:
                raise TravelDemandError("school zone %d is outside the system." % k)
        if not picked:
            raise TravelDemandError("give at least one school zone.")
    rng = _rng(seed)
    capacity = [0.0] * zones.n
    for k in picked:
        capacity[k] = max(1.0, pop[k]) * (0.8 + 0.4 * rng.random())
    p = [pop[k] * share * rate for k in range(zones.n)]
    costs = zones.cost_matrix(speed=speed, fixed=1.0)
    fn = get_deterrence("exponential", beta=beta)
    grav = doubly_constrained_gravity(p, capacity, costs, fn)
    return ScenarioResult(
        name="school", matrix=grav.matrix, productions=row_sums(grav.matrix),
        attractions=col_sums(grav.matrix), seed=seed,
        notes=["productions = population x %.3f children x %.3f trips"
               % (share, rate),
               "attractions confined to %d school zones" % len(picked),
               "short-trip deterrence beta = %.4f (%.1fx the commute)"
               % (beta, beta / 0.08)],
        detail={"schoolZones": picked, "meanCost": grav.mean_cost,
                "capacity": capacity})


def stadium_scenario(zones: ZoneSystem, stadium_zone: int, attendance: float,
                     *, seed: Any = 0, release_minutes: float = 45.0,
                     step_minutes: float = 5.0, beta: float = 0.03,
                     speed: float = 0.42, home_share: float = 0.85
                     ) -> ScenarioResult:
    """A stadium emptying: one origin, a triangular release, gravity destinations."""
    if not isinstance(zones, ZoneSystem):
        raise TravelDemandError("zones must be a ZoneSystem.")
    src = _non_negative_int(stadium_zone, "stadium_zone")
    if src >= zones.n:
        raise TravelDemandError("stadium zone %d is outside the system." % src)
    crowd = _positive(attendance, "attendance")
    release = _positive(release_minutes, "release_minutes")
    step = _positive(step_minutes, "step_minutes")
    if step > release:
        raise TravelDemandError("step_minutes must not exceed release_minutes.")
    share = _in_unit(home_share, "home_share")
    pop = zones.vector("population")
    rng = _rng(seed)
    pull = [(pop[k] if k != src else 0.0) * (0.85 + 0.3 * rng.random())
            for k in range(zones.n)]
    if math.fsum(pull) == 0.0:
        raise TravelDemandError("the stadium scenario needs population elsewhere.")
    costs = zones.cost_matrix(speed=speed, fixed=1.0)
    p = [0.0] * zones.n
    p[src] = crowd * share
    fn = get_deterrence("exponential", beta=beta)
    grav = singly_constrained_gravity(p, pull, costs, fn, "origin")
    steps = max(1, int(round(release / step)))
    weights: Vector = []
    for k in range(steps):
        t = (k + 0.5) / steps
        weights.append(1.0 - abs(2.0 * t - 1.0))     # triangular release
    wsum = math.fsum(weights)
    profile = [((k + 1) * step, crowd * share * weights[k] / wsum)
               for k in range(steps)]
    return ScenarioResult(
        name="stadium", matrix=grav.matrix, productions=row_sums(grav.matrix),
        attractions=col_sums(grav.matrix), seed=seed,
        notes=["%.0f spectators, %.0f%% heading home" % (crowd, 100.0 * share),
               "singly constrained on the stadium zone only",
               "triangular release over %.0f minutes in %d steps"
               % (release, steps)],
        detail={"stadiumZone": src, "profile": profile,
                "peakFlowPerMinute": max(f for (_t, f) in profile) / step,
                "meanCost": grav.mean_cost})


def weather_adjustment(matrix: Matrix, condition: str = "rain", *,
                       mode_shares: Optional[Dict[str, float]] = None,
                       seed: Any = 0, jitter: float = 0.04) -> ScenarioResult:
    """Weather shock: a demand multiplier plus a shift out of the active modes."""
    mat = _as_matrix(matrix, "matrix", non_negative=True)
    key = str(condition).lower().replace(" ", "_")
    if key not in _WEATHER:
        raise TravelDemandError("unknown weather %r (use one of %s)."
                                % (condition, ", ".join(sorted(_WEATHER))))
    demand_mult, car_shift, active_mult = _WEATHER[key]
    jit = _non_negative(jitter, "jitter")
    rng = _rng(seed)
    factor = demand_mult * (1.0 + jit * (2.0 * rng.random() - 1.0))
    out = matrix_scale(mat, factor)
    shares = {"car": 0.62, "public_transport": 0.20, "walk": 0.12, "cycle": 0.06} \
        if mode_shares is None else \
        {str(k): _non_negative(v, "mode_shares[%s]" % k)
         for k, v in sorted(dict(mode_shares).items())}
    total = math.fsum(shares[k] for k in sorted(shares))
    if total <= 0.0:
        raise TravelDemandError("mode shares must not all be zero.")
    shares = {k: shares[k] / total for k in sorted(shares)}
    moved = 0.0
    new_shares: Dict[str, float] = {}
    for name in sorted(shares):
        if name in _ACTIVE_MODES:
            kept = shares[name] * active_mult
            moved += shares[name] - kept
            new_shares[name] = kept
        else:
            new_shares[name] = shares[name]
    receivers = [k for k in sorted(new_shares) if k not in _ACTIVE_MODES]
    if receivers and moved > 0.0:
        car_first = min(moved, car_shift) if "car" in receivers else 0.0
        if car_first > 0.0:
            new_shares["car"] += car_first
            moved -= car_first
        base = math.fsum(new_shares[k] for k in receivers)
        if base > 0.0:
            for k in receivers:
                new_shares[k] += moved * new_shares[k] / base
        else:
            for k in receivers:
                new_shares[k] += moved / len(receivers)
    renorm = math.fsum(new_shares[k] for k in sorted(new_shares))
    new_shares = {k: new_shares[k] / renorm for k in sorted(new_shares)}
    return ScenarioResult(
        name="weather:%s" % key, matrix=out, productions=row_sums(out),
        attractions=col_sums(out), seed=seed,
        notes=["%s cuts demand by %.1f%% (jittered to %.1f%%)"
               % (key, 100.0 * (1.0 - demand_mult), 100.0 * (1.0 - factor)),
               "active-mode shares multiplied by %.2f, the loss moved to the "
               "motorised modes" % active_mult],
        detail={"condition": key, "demandFactor": factor,
                "modeShares": new_shares, "baseModeShares": shares,
                "activeMultiplier": active_mult})


def incident_adjustment(matrix: Matrix, affected_zones: Sequence[int], *,
                        severity: float = 0.5, suppression: float = 0.35,
                        seed: Any = 0) -> ScenarioResult:
    """An incident: trips touching the affected zones are cut, some redistributed."""
    mat = _as_matrix(matrix, "matrix", non_negative=True)
    n, m = len(mat), len(mat[0])
    hit = sorted({_non_negative_int(k, "affected zone") for k in affected_zones})
    if not hit:
        raise TravelDemandError("give at least one affected zone.")
    for k in hit:
        if k >= max(n, m):
            raise TravelDemandError("affected zone %d is outside the matrix." % k)
    sev = _in_unit(severity, "severity")
    sup = _in_unit(suppression, "suppression")
    rng = _rng(seed)
    hitset = set(hit)
    out = [list(r) for r in mat]
    removed = [0.0] * n
    for i in range(n):
        for j in range(m):
            if i in hitset or j in hitset:
                local = sev * (0.85 + 0.3 * rng.random())
                local = min(1.0, max(0.0, local))
                lost = out[i][j] * local
                out[i][j] -= lost
                removed[i] += lost
    redistributed = 0.0
    for i in range(n):
        spare = removed[i] * (1.0 - sup)
        if spare <= 0.0:
            continue
        open_cells = [j for j in range(m) if j not in hitset and i not in hitset]
        base = math.fsum(out[i][j] for j in open_cells)
        if not open_cells or base <= 0.0:
            continue
        for j in open_cells:
            out[i][j] += spare * out[i][j] / base
        redistributed += spare
    lost_total = math.fsum(removed)
    return ScenarioResult(
        name="incident", matrix=out, productions=row_sums(out),
        attractions=col_sums(out), seed=seed,
        notes=["%d zone(s) affected at severity %.2f" % (len(hit), sev),
               "%.1f trips removed, %.1f redistributed, %.1f suppressed"
               % (lost_total, redistributed, lost_total - redistributed)],
        detail={"affectedZones": hit, "removed": lost_total,
                "redistributed": redistributed,
                "suppressed": lost_total - redistributed,
                "severity": sev, "suppression": sup})


def generate_scenario(name: str, *args: Any, **kwargs: Any) -> ScenarioResult:
    """Dispatch to one of SCENARIO_NAMES by name, passing the arguments through."""
    key = str(name).lower().replace("-", "_").replace(" ", "_")
    if key in ("commuting_peak", "commute", "commuting"):
        return commuting_peak_scenario(*args, **kwargs)
    if key in ("school", "school_run"):
        return school_scenario(*args, **kwargs)
    if key == "stadium":
        return stadium_scenario(*args, **kwargs)
    if key == "weather":
        return weather_adjustment(*args, **kwargs)
    if key == "incident":
        return incident_adjustment(*args, **kwargs)
    raise TravelDemandError("unknown scenario %r (use one of %s)."
                            % (name, ", ".join(SCENARIO_NAMES)))


# ===========================================================================
# 11. SYNTHETIC POPULATION AND TRIP LISTS
# ===========================================================================
# The bridge from an aggregate matrix to an agent-based model.  POPULATION
# SYNTHESIS fits a household contingency table -- zone by household size by car
# ownership -- to known marginals by IPF, then INTEGERISES it, because you
# cannot simulate 3.7 households.  Largest-remainder integerisation is used
# rather than naive rounding because it preserves the grand total EXACTLY and
# never moves a cell by more than one household.
#
# TRIP-LIST EXPANSION goes the other way from a matrix: every whole trip in
# every cell becomes an individual ``Trip`` with a mode and a departure time
# drawn from a seeded generator.  Summing that list back up reproduces the
# matrix EXACTLY -- an exact round-trip identity, which is the oracle the suite
# uses on this section, and the reason the expansion refuses a matrix whose
# cells are not whole numbers instead of quietly rounding them.
# ---------------------------------------------------------------------------
@dataclass
class Household:
    """One synthetic household: where it lives, how big it is, what it drives."""

    ident: int
    zone: int
    size: int
    cars: int
    income: float = 0.0
    persons: Tuple[int, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the household."""
        return {"id": self.ident, "zone": self.zone, "size": self.size,
                "cars": self.cars, "income": self.income,
                "persons": list(self.persons)}


@dataclass
class Person:
    """One synthetic person, attached to a household."""

    ident: int
    household: int
    zone: int
    age: int
    employed: bool = False
    licence: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the person."""
        return {"id": self.ident, "household": self.household,
                "zone": self.zone, "age": self.age,
                "employed": self.employed, "licence": self.licence}


@dataclass
class Trip:
    """One individual trip in an expanded trip list."""

    ident: int
    origin: int
    destination: int
    purpose: str = "hbw"
    mode: str = "car"
    departure: float = 8.0
    person: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the trip."""
        return {"id": self.ident, "origin": self.origin,
                "destination": self.destination, "purpose": self.purpose,
                "mode": self.mode, "departure": self.departure,
                "person": self.person}


@dataclass
class SyntheticPopulation:
    """A fitted, integerised household table expanded into households and persons."""

    households: List[Household]
    persons: List[Person]
    table: Any
    zone_totals: Vector
    seed: Any
    detail: Dict[str, Any] = field(default_factory=dict)

    def size(self) -> int:
        """Number of people in the population."""
        return len(self.persons)

    def households_in(self, zone: int) -> List[Household]:
        """Households living in one zone, in id order."""
        k = _non_negative_int(zone, "zone")
        return [h for h in self.households if h.zone == k]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able view of the population."""
        return {"households": len(self.households), "persons": len(self.persons),
                "zoneTotals": list(self.zone_totals), "seed": self.seed,
                "detail": dict(self.detail)}


def integerise_matrix(matrix: Matrix, *, method: str = "largest_remainder",
                      seed: Any = 0) -> Matrix:
    """Round a matrix to whole trips; largest remainder preserves the total exactly."""
    mat = _as_matrix(matrix, "matrix", non_negative=True)
    n, m = len(mat), len(mat[0])
    how = str(method).lower()
    if how in ("largest_remainder", "largest", "lr"):
        total = matrix_total(mat)
        target = int(round(total))
        floors = [[math.floor(mat[i][j]) for j in range(m)] for i in range(n)]
        placed = int(math.fsum(math.fsum(r) for r in floors))
        spare = target - placed
        if spare > 0:
            order = sorted(((mat[i][j] - floors[i][j], i, j)
                            for i in range(n) for j in range(m)),
                           key=lambda t: (-t[0], t[1], t[2]))
            for k in range(min(spare, len(order))):
                _frac, i, j = order[k]
                floors[i][j] += 1.0
        elif spare < 0:
            order = sorted(((mat[i][j] - floors[i][j], i, j)
                            for i in range(n) for j in range(m)),
                           key=lambda t: (t[0], t[1], t[2]))
            taken = 0
            for (_frac, i, j) in order:
                if taken >= -spare:
                    break
                if floors[i][j] > 0.0:
                    floors[i][j] -= 1.0
                    taken += 1
        return [[float(v) for v in row] for row in floors]
    if how in ("stochastic", "random"):
        rng = _rng(seed)
        out = zeros_matrix(n, m)
        for i in range(n):
            for j in range(m):
                base = math.floor(mat[i][j])
                frac = mat[i][j] - base
                out[i][j] = float(base + (1 if rng.random() < frac else 0))
        return out
    raise TravelDemandError("unknown integerisation %r (use largest_remainder "
                            "or stochastic)." % method)


def synthesise_population(zone_households: Sequence[float],
                          size_distribution: Sequence[float],
                          car_distribution: Sequence[float], *, seed: Any = 0,
                          employment_rate: float = 0.55,
                          licence_rate: float = 0.75,
                          max_iter: int = 200) -> SyntheticPopulation:
    """IPF a zone x size x cars household table to its marginals, then expand it."""
    zh = _as_floats(zone_households, "zone_households")
    sd = _as_floats(size_distribution, "size_distribution")
    cd = _as_floats(car_distribution, "car_distribution")
    for v in zh + sd + cd:
        if v < 0.0:
            raise TravelDemandError("distributions must be non-negative.")
    if not zh or not sd or not cd:
        raise TravelDemandError("every dimension needs at least one category.")
    total = math.fsum(zh)
    if total <= 0.0:
        raise TravelDemandError("zone_households must not all be zero.")
    sd_total, cd_total = math.fsum(sd), math.fsum(cd)
    if sd_total <= 0.0 or cd_total <= 0.0:
        raise TravelDemandError("the size and car distributions must not be zero.")
    size_marg = [v * total / sd_total for v in sd]
    car_marg = [v * total / cd_total for v in cd]
    nz, ns, nc = len(zh), len(sd), len(cd)
    seed_table = [[[1.0] * nc for _ in range(ns)] for _ in range(nz)]
    fitted = ipf(seed_table, [zh, size_marg, car_marg], max_iter=max_iter)
    # Integerise each ZONE separately so that every zone total is preserved to
    # the nearest household; integerising the whole table at once would only
    # pin the grand total and would let a zone drift by a couple of units.
    whole: Matrix = []
    for z in range(nz):
        block = [[fitted[z][s][c] for c in range(nc)] for s in range(ns)]
        whole.extend(integerise_matrix(block, method="largest_remainder"))
    rng = _rng(seed)
    households: List[Household] = []
    persons: List[Person] = []
    hid = 0
    pid = 0
    zone_totals = [0.0] * nz
    emp_rate = _in_unit(employment_rate, "employment_rate")
    lic_rate = _in_unit(licence_rate, "licence_rate")
    for z in range(nz):
        for s in range(ns):
            for c in range(nc):
                count = int(round(whole[z * ns + s][c]))
                for _ in range(count):
                    size = s + 1
                    members: List[int] = []
                    for _q in range(size):
                        age = 6 + int(rng.random() * 74)
                        persons.append(Person(
                            ident=pid, household=hid, zone=z, age=age,
                            employed=(18 <= age < 67 and rng.random() < emp_rate),
                            licence=(age >= 17 and rng.random() < lic_rate)))
                        members.append(pid)
                        pid += 1
                    households.append(Household(
                        ident=hid, zone=z, size=size, cars=c,
                        income=round(18000.0 + 42000.0 * rng.random(), 2),
                        persons=tuple(members)))
                    zone_totals[z] += 1.0
                    hid += 1
    return SyntheticPopulation(
        households=households, persons=persons, table=fitted,
        zone_totals=zone_totals, seed=seed,
        detail={"zones": nz, "sizes": ns, "carCategories": nc,
                "targetHouseholds": total, "synthesised": len(households),
                "employmentRate": emp_rate, "licenceRate": lic_rate})


def expand_to_trip_list(matrix: Matrix, *, purpose: str = "hbw",
                        mode_shares: Optional[Dict[str, float]] = None,
                        time_shares: Optional[Dict[str, float]] = None,
                        slices: Optional[Sequence[TimeSlice]] = None,
                        seed: Any = 0, start_id: int = 0) -> List[Trip]:
    """Expand a WHOLE-NUMBER matrix into individual trips with modes and departures."""
    mat = _as_matrix(matrix, "matrix", non_negative=True)
    n, m = len(mat), len(mat[0])
    for i in range(n):
        for j in range(m):
            if abs(mat[i][j] - round(mat[i][j])) > 1e-9 * max(1.0, mat[i][j]):
                raise TravelDemandError(
                    "cell (%d, %d) holds %.6g trips, which is not a whole "
                    "number -- call integerise_matrix first." % (i, j, mat[i][j]))
    modes = {"car": 1.0} if mode_shares is None else \
        {str(k): _non_negative(v, "mode_shares[%s]" % k)
         for k, v in sorted(dict(mode_shares).items())}
    mode_names = sorted(modes)
    mode_total = math.fsum(modes[k] for k in mode_names)
    if mode_total <= 0.0:
        raise TravelDemandError("mode shares must not all be zero.")
    mode_cum: Vector = []
    run = 0.0
    for k in mode_names:
        run += modes[k] / mode_total
        mode_cum.append(run)
    if slices is None:
        bands = [TimeSlice("all-day", 0.0, 24.0)]
    else:
        bands = list(slices)
        for s in bands:
            if not isinstance(s, TimeSlice):
                raise TravelDemandError("slices must hold TimeSlice objects.")
    if time_shares is None:
        weights = [1.0] * len(bands)
    else:
        ts = {str(k): _non_negative(v, "time_shares[%s]" % k)
              for k, v in sorted(dict(time_shares).items())}
        weights = [ts.get(s.name, 0.0) for s in bands]
    wtotal = math.fsum(weights)
    if wtotal <= 0.0:
        raise TravelDemandError("time shares must not all be zero.")
    time_cum: Vector = []
    run = 0.0
    for w in weights:
        run += w / wtotal
        time_cum.append(run)
    rng = _rng(seed)
    out: List[Trip] = []
    tid = _non_negative_int(start_id, "start_id")
    for i in range(n):
        for j in range(m):
            count = int(round(mat[i][j]))
            for _ in range(count):
                u = rng.random()
                mk = 0
                while mk < len(mode_cum) - 1 and u > mode_cum[mk]:
                    mk += 1
                v = rng.random()
                bk = 0
                while bk < len(time_cum) - 1 and v > time_cum[bk]:
                    bk += 1
                band = bands[bk]
                depart = band.start + rng.random() * band.duration()
                out.append(Trip(ident=tid, origin=i, destination=j,
                                purpose=str(purpose), mode=mode_names[mk],
                                departure=depart))
                tid += 1
    return out


def trip_list_to_matrix(trips: Sequence[Trip], shape: Tuple[int, int], *,
                        mode: Optional[str] = None,
                        purpose: Optional[str] = None) -> Matrix:
    """Aggregate a trip list into a matrix, optionally filtered by mode or purpose."""
    n = _positive_int(shape[0], "shape[0]")
    m = _positive_int(shape[1], "shape[1]")
    out = zeros_matrix(n, m)
    want_mode = None if mode is None else str(mode)
    want_purpose = None if purpose is None else str(purpose)
    for t in trips:
        if not isinstance(t, Trip):
            raise TravelDemandError("trips must hold Trip objects.")
        if want_mode is not None and t.mode != want_mode:
            continue
        if want_purpose is not None and t.purpose != want_purpose:
            continue
        if not (0 <= t.origin < n and 0 <= t.destination < m):
            raise TravelDemandError("trip %d leaves a %dx%d matrix."
                                    % (t.ident, n, m))
        out[t.origin][t.destination] += 1.0
    return out


# ===========================================================================
# 12. FRONTEND PAYLOADS
# ===========================================================================
def _json_safe(x: Any) -> Any:
    """Replace non-finite floats with None so the dict survives json.dumps."""
    if isinstance(x, float):
        if math.isinf(x) or math.isnan(x):
            return None
        return round(x, 8)
    return x


def _matrix_json(mat: Matrix, max_cells: int) -> List[Dict[str, Any]]:
    """The largest cells of a matrix as a JSON-able list, biggest first."""
    cells = [(mat[i][j], i, j) for i in range(len(mat))
             for j in range(len(mat[0])) if mat[i][j] != 0.0]
    cells.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [{"origin": i, "destination": j, "trips": _json_safe(v)}
            for (v, i, j) in cells[:max(1, int(max_cells))]]


def build_travel_demand_payload(
        matrix: Matrix, *, zones: Optional[ZoneSystem] = None,
        costs: Optional[Matrix] = None,
        deterrence: Optional[DeterrenceFunction] = None,
        generation: Optional[GLMResult] = None,
        distribution: Optional[GravityResult] = None,
        mode_choice: Optional[Any] = None,
        departure: Optional[DepartureTimeResult] = None,
        four_step: Optional[FourStepResult] = None,
        estimation: Optional[ODEstimationResult] = None,
        comparison: Optional[MatrixComparison] = None,
        scenario: Optional[ScenarioResult] = None,
        bins: int = 12, curve_points: int = 40,
        max_cells: int = 300) -> Dict[str, Any]:
    """Frontend-ready JSON dict: matrix, margins, trip lengths, choices, fit."""
    mat = _as_matrix(matrix, "matrix", non_negative=True)
    margins = matrix_margins(mat)
    payload: Dict[str, Any] = {
        "matrix": {"rows": len(mat), "cols": len(mat[0]),
                   "total": _json_safe(margins.total),
                   "cells": _matrix_json(mat, max_cells)},
        "margins": {k: (_json_safe(v) if isinstance(v, float) else v)
                    for k, v in margins.to_dict().items()},
    }
    if zones is not None:
        payload["zones"] = zones.to_dict()
    if costs is not None:
        cm = _as_matrix(costs, "costs", non_negative=True)
        tld = trip_length_distribution(mat, cm, bins)
        payload["tripLengths"] = {
            "edges": [_json_safe(v) for v in tld.edges],
            "trips": [_json_safe(v) for v in tld.trips],
            "shares": [_json_safe(v) for v in tld.shares],
            "mean": _json_safe(tld.mean), "std": _json_safe(tld.std),
            "median": _json_safe(tld.median), "total": _json_safe(tld.total)}
    if deterrence is not None:
        pts = _positive_int(curve_points, "curve_points")
        hi = 1.0
        if costs is not None:
            hi = max(max(row) for row in _as_matrix(costs, "costs"))
        lo = 0.0 if deterrence.name in ("exponential", "top_log", "log_normal") \
            else max(1e-6, hi / (10.0 * pts))
        payload["deterrence"] = dict(
            deterrence.to_dict(),
            curve=[{"cost": _json_safe(c), "weight": _json_safe(f)}
                   for (c, f) in deterrence.curve(lo, max(hi, lo * 2.0), pts)])
    if generation is not None:
        payload["generation"] = {k: (_json_safe(v) if isinstance(v, float) else v)
                                 for k, v in generation.to_dict().items()}
    if distribution is not None:
        payload["distribution"] = {
            k: (_json_safe(v) if isinstance(v, float) else v)
            for k, v in distribution.to_dict().items()}
    if mode_choice is not None:
        payload["modeChoice"] = {
            k: (_json_safe(v) if isinstance(v, float) else v)
            for k, v in mode_choice.to_dict().items()}
    if departure is not None:
        payload["departureTime"] = {
            k: (_json_safe(v) if isinstance(v, float) else v)
            for k, v in departure.to_dict().items()}
    if four_step is not None:
        payload["fourStep"] = {
            "converged": four_step.converged, "gap": _json_safe(four_step.gap),
            "meanCost": _json_safe(four_step.mean_cost),
            "total": _json_safe(four_step.total),
            "compositeOffset": _json_safe(four_step.composite_offset),
            "modeShares": {k: _json_safe(v)
                           for k, v in sorted(four_step.mode_shares.items())},
            "trace": [{"index": it.index, "gap": _json_safe(it.gap),
                       "costGap": _json_safe(it.cost_gap),
                       "meanCost": _json_safe(it.mean_cost),
                       "maxVolumeCapacity": _json_safe(it.max_volume_capacity)}
                      for it in four_step.iterations]}
    if estimation is not None:
        payload["estimation"] = {
            "method": estimation.method, "converged": estimation.converged,
            "iterations": estimation.iterations,
            "objective": _json_safe(estimation.objective),
            "priorDistance": _json_safe(estimation.prior_distance),
            "modelledCounts": [_json_safe(v) for v in estimation.modelled_counts],
            "fit": {k: (_json_safe(v) if isinstance(v, float) else v)
                    for k, v in estimation.fit.to_dict().items()}}
    if comparison is not None:
        payload["comparison"] = {
            k: (_json_safe(v) if isinstance(v, float) else v)
            for k, v in comparison.to_dict().items()}
    if scenario is not None:
        payload["scenario"] = {"name": scenario.name,
                               "total": _json_safe(scenario.total()),
                               "notes": list(scenario.notes),
                               "seed": scenario.seed}
    return payload


def build_a2ui_payload(matrix: Matrix, *, zones: Optional[ZoneSystem] = None,
                       costs: Optional[Matrix] = None,
                       distribution: Optional[GravityResult] = None,
                       mode_choice: Optional[Any] = None,
                       four_step: Optional[FourStepResult] = None,
                       estimation: Optional[ODEstimationResult] = None,
                       comparison: Optional[MatrixComparison] = None,
                       bins: int = 12, top_cells: int = 10) -> Dict[str, Any]:
    """Compact agent-to-UI payload: stat cards, one chart per panel, a top-cell table."""
    mat = _as_matrix(matrix, "matrix", non_negative=True)
    margins = matrix_margins(mat)
    cards: List[Dict[str, Any]] = [
        {"label": "Trips", "value": "%.0f" % margins.total, "unit": "trips"},
        {"label": "Zones", "value": "%d x %d" % (margins.n_rows, margins.n_cols),
         "unit": "OD"},
        {"label": "Non-zero cells",
         "value": "%d" % margins.nonzero,
         "unit": "of %d" % (margins.n_rows * margins.n_cols)},
    ]
    charts: List[Dict[str, Any]] = []
    if costs is not None:
        cm = _as_matrix(costs, "costs", non_negative=True)
        tld = trip_length_distribution(mat, cm, bins)
        cards.append({"label": "Mean trip cost", "value": "%.2f" % tld.mean,
                      "unit": "cost units"})
        charts.append({"id": "tripLength", "title": "Trip length distribution",
                       "type": "bar",
                       "points": [{"x": _json_safe(0.5 * (tld.edges[k]
                                                          + tld.edges[k + 1])),
                                   "y": _json_safe(tld.trips[k])}
                                  for k in range(len(tld.trips))]})
    if margins.n_rows == margins.n_cols:
        cards.append({"label": "Intrazonal share",
                      "value": "%.1f%%" % (100.0 * intrazonal_share(mat)),
                      "unit": "of trips"})
    if distribution is not None:
        cards.append({"label": "Distribution",
                      "value": "%s / %s" % (distribution.constraint,
                                            distribution.deterrence),
                      "unit": "%d iterations" % distribution.iterations})
    if mode_choice is not None:
        shares = mode_choice.as_dict()
        charts.append({"id": "modeShares", "title": "Mode shares", "type": "pie",
                       "points": [{"label": k, "y": _json_safe(shares[k])}
                                  for k in sorted(shares)]})
        top = max(sorted(shares), key=lambda k: (shares[k], k))
        cards.append({"label": "Leading mode", "value": top,
                      "unit": "%.1f%%" % (100.0 * shares[top])})
    if four_step is not None:
        charts.append({"id": "feedback", "title": "Feedback convergence",
                       "type": "line",
                       "points": [{"x": it.index, "y": _json_safe(it.gap)}
                                  for it in four_step.iterations]})
        cards.append({"label": "Feedback gap",
                      "value": "%.3e" % four_step.gap,
                      "unit": "converged" if four_step.converged else "open"})
    if estimation is not None:
        cards.append({"label": "Count fit (GEH<5)",
                      "value": "%.0f%%" % estimation.fit.pct_geh_under_5,
                      "unit": "%d counts" % estimation.fit.n})
        charts.append({"id": "counts", "title": "Modelled vs observed counts",
                       "type": "scatter",
                       "points": [{"x": _json_safe(v), "y": _json_safe(v)}
                                  for v in estimation.modelled_counts]})
    if comparison is not None:
        cards.append({"label": "Matrix R2", "value": "%.4f" % comparison.r2,
                      "unit": "%%RMSE %.1f" % comparison.pct_rmse})
    table = {"id": "topCells", "title": "Largest OD movements",
             "columns": ["origin", "destination", "trips"],
             "rows": []}
    labels = zones.ids() if zones is not None else None
    for cell in _matrix_json(mat, top_cells):
        o, d = cell["origin"], cell["destination"]
        table["rows"].append([labels[o] if labels else o,
                              labels[d] if labels else d, cell["trips"]])
    return {"statCards": cards, "charts": charts, "tables": [table]}


# ===========================================================================
# 13. DEMO
# ===========================================================================
def _demo() -> None:
    print("=" * 74)
    print("  TRAVEL DEMAND (pure Python) - demo")
    print("=" * 74)

    # -- 0. numerics --------------------------------------------------------
    print("\n-- 0. numerics: linear solve, inverse, logsumexp --")
    A = [[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]]
    b = [7.0, 4.0, 7.0]
    x = solve_linear(A, b)
    resid = max(abs(math.fsum(A[i][j] * x[j] for j in range(3)) - b[i])
                for i in range(3))
    inv = invert_matrix(A)
    ident = max(abs(math.fsum(A[i][k] * inv[k][j] for k in range(3))
                    - (1.0 if i == j else 0.0))
                for i in range(3) for j in range(3))
    print("   solve residual %.3e | A A^-1 - I = %.3e" % (resid, ident))
    print("   logsumexp([0,1,2]) = %.10f (exact %.10f)"
          % (logsumexp([0.0, 1.0, 2.0]),
             math.log(1.0 + math.e + math.e ** 2)))
    print("   Euler-Mascheroni = %.16f" % EULER_MASCHERONI)

    # -- 1. zones and matrices ---------------------------------------------
    print("\n-- 1. zones, trip matrices and matrix algebra --")
    zones = ZoneSystem([
        Zone("CBD", households=1200, population=2600, employment=9800,
             retail=2400, income=34000, area=1.8, x=0.0, y=0.0),
        Zone("Inner-N", households=3100, population=7400, employment=2600,
             retail=900, income=29000, area=3.4, x=1.2, y=3.0),
        Zone("Inner-S", households=2800, population=6900, employment=2100,
             retail=750, income=27000, area=3.1, x=-0.8, y=-3.4),
        Zone("Suburb-E", households=4200, population=10800, employment=1500,
             retail=1100, income=41000, area=9.6, x=7.5, y=1.1),
        Zone("Suburb-W", households=3600, population=9100, employment=1200,
             retail=600, income=38000, area=8.8, x=-7.9, y=0.6),
        Zone("Town", households=1900, population=4700, employment=3400,
             retail=1600, income=31000, area=5.2, x=2.4, y=-8.5),
    ])
    costs = zones.cost_matrix(speed=0.5, fixed=2.0)       # km/min -> minutes
    print("   %d zones: %s" % (zones.n, ", ".join(zones.ids())))
    print("   households %s" % ["%.0f" % v for v in zones.vector("households")])
    print("   cost CBD -> Suburb-E = %.3f min, intrazonal CBD = %.3f min"
          % (costs[0][3], costs[0][0]))
    seedm = uniform_matrix(zones.n, zones.n, 1.0)
    empty = zeros_matrix(2, 3)
    pairs = matrix_from_pairs({(0, 1): 40.0, (1, 0): 25.0}, 2, 3)
    print("   zeros_matrix(2,3) total %.0f | matrix_from_pairs -> %s"
          % (matrix_total(empty), matrix_to_pairs(pairs)))
    demo_mat = [[0.0, 120.0, 80.0], [95.0, 0.0, 60.0], [70.0, 55.0, 0.0]]
    tm = TripMatrix(demo_mat, ["A", "B", "C"], ["A", "B", "C"], name="base")
    print("   TripMatrix %s: total %.0f, A->B = %.0f, row sums %s"
          % (tm.name, tm.total(), tm.get("A", "B"),
             ["%.0f" % v for v in tm.row_sums()]))
    print("   col sums %s | transpose A<-B = %.0f | scaled x2 total %.0f"
          % (["%.0f" % v for v in col_sums(demo_mat)],
             matrix_transpose(demo_mat)[0][1], tm.scaled(2.0).total()))
    summed = matrix_add(demo_mat, matrix_transpose(demo_mat))
    print("   T + T' is symmetric: %s | T - T is zero: %.1f"
          % (all(summed[i][j] == summed[j][i] for i in range(3)
                 for j in range(3)),
             matrix_total(matrix_subtract(demo_mat, demo_mat))))
    sym = matrix_symmetrise(demo_mat, 0.5)
    print("   symmetrise(0.5) keeps the total: %.1f -> %.1f"
          % (matrix_total(demo_mat), matrix_total(sym)))
    withintra = set_intrazonals(demo_mat, [20.0, 15.0, 10.0])
    print("   intrazonal share after set_intrazonals = %.4f, stripped = %.4f"
          % (intrazonal_share(withintra), intrazonal_share(
              strip_intrazonals(withintra))))
    print("   matrix_round(1 dp) row 0: %s" % matrix_round(sym, 1)[0])
    marg = matrix_margins(withintra)
    print("   margins: total %.0f, non-zero %d, biggest cell %.0f at %s"
          % (marg.total, marg.nonzero, marg.max_cell, marg.max_cell_index))

    # -- 2. trip generation -------------------------------------------------
    print("\n-- 2. trip generation: cross-classification and count GLMs --")
    cct = CrossClassTable(
        rates={(0, 1): 0.9, (0, 2): 1.4, (0, 3): 1.8,
               (1, 1): 1.6, (1, 2): 2.4, (1, 3): 3.1,
               (2, 1): 2.0, (2, 2): 3.0, (2, 3): 4.0},
        dimensions=("cars", "size"), purpose="hbw")
    households = [{(0, 1): 180.0, (1, 2): 260.0, (2, 3): 90.0},
                  {(0, 1): 420.0, (1, 2): 610.0, (2, 3): 140.0},
                  {(0, 1): 330.0, (1, 2): 540.0, (2, 3): 120.0},
                  {(0, 1): 210.0, (1, 2): 980.0, (2, 3): 640.0},
                  {(0, 1): 250.0, (1, 2): 840.0, (2, 3): 510.0},
                  {(0, 1): 300.0, (1, 2): 420.0, (2, 3): 150.0}]
    per_zone = cross_classification_trips(cct, households)
    print("   cross-classification productions: %s"
          % ["%.1f" % v for v in per_zone])
    print("   one zone alone = %.1f | mean rate by cars: %s"
          % (cross_classification_trips(cct, households[0]),
             {k: round(v, 3) for k, v in cct.marginal(0).items()}))
    rng = random.Random(20240915)
    rows: List[List[float]] = []
    counts: Vector = []
    for _ in range(240):
        cars = rng.randint(0, 2)
        size = rng.randint(1, 4)
        mu = math.exp(-0.25 + 0.42 * cars + 0.28 * size)
        # negative-binomial draw: Poisson mixed over a gamma
        shape = 1.0 / 0.35
        lam = mu * rng.gammavariate(shape, 1.0 / shape)
        k = 0
        limit = math.exp(-lam) if lam < 700.0 else 0.0
        prod = rng.random()
        while prod > limit and k < 200:
            prod *= rng.random()
            k += 1
        rows.append([float(cars), float(size)])
        counts.append(float(k))
    X = design_matrix(rows)
    pois = poisson_glm(X, counts, names=["intercept", "cars", "size"])
    print("   %s" % pois.summary().replace("\n", "\n   "))
    nb = negbin_glm(X, counts, names=["intercept", "cars", "size"])
    print("   NB2 alpha = %.5f (true 0.35), loglik %.4f vs Poisson %.4f"
          % (nb.dispersion, nb.loglik, pois.loglik))
    print("   NB beats Poisson on AIC: %.2f < %.2f -> %s"
          % (nb.aic, pois.aic, nb.aic < pois.aic))
    print("   alpha score at the optimum = %.3e (should be ~0)"
          % negbin_alpha_score(nb.coefficients, X, counts, nb.dispersion))
    profile = negbin_dispersion_profile(X, counts, [0.05, 0.2, nb.dispersion,
                                                    0.8, 2.0])
    print("   dispersion profile: %s"
          % ["a=%.3f ll=%.3f" % (a, ll) for (a, ll) in profile])
    only = poisson_glm([[1.0]] * len(counts), counts)
    print("   intercept-only Poisson beta0 = %.15f, log(mean) = %.15f"
          % (only.coefficients[0], math.log(_mean(counts))))
    print("   poisson_loglik at the fit = %.6f (matches the result: %s)"
          % (poisson_loglik(pois.coefficients, X, counts),
             abs(poisson_loglik(pois.coefficients, X, counts)
                 - pois.loglik) < 1e-9))
    print("   negbin_loglik at the fit  = %.6f"
          % negbin_loglik(nb.coefficients, X, counts, nb.dispersion))
    lin = linear_glm(X, counts, names=["intercept", "cars", "size"])
    print("   OLS comparison: coefficients %s, residual variance %.4f"
          % (["%+.4f" % v for v in lin.coefficients], lin.dispersion))
    print("   predicted trips for (1 car, 3 people) = %.4f"
          % glm_predict(pois, [[1.0, 1.0, 3.0]])[0])
    prods = trip_rate_model(zones, {"households": 0.62, "employment": 0.04})
    attrs = trip_rate_model(zones, {"employment": 0.58, "retail": 0.22,
                                    "households": 0.05})
    ends = trip_ends(zones, {"households": 0.62, "employment": 0.04},
                     {"employment": 0.58, "retail": 0.22, "households": 0.05})
    print("   productions %s (total %.0f)"
          % (["%.0f" % v for v in prods], math.fsum(prods)))
    print("   attractions %s (total %.0f)"
          % (["%.0f" % v for v in attrs], math.fsum(attrs)))
    bal = balance_trip_ends(ends.productions, ends.attractions, "attractions")
    print("   balanced by %s: factor %.6f, totals now %.4f and %.4f"
          % (bal.method, bal.factor, math.fsum(bal.productions),
             math.fsum(bal.attractions)))
    P, A = bal.productions, bal.attractions

    # -- 3. deterrence functions -------------------------------------------
    print("\n-- 3. deterrence functions --")
    print("   %-12s %10s %10s %10s %12s" % ("family", "f(5)", "f(20)", "f(45)",
                                            "elasticity(20)"))
    for name in DETERRENCE_NAMES:
        fn = get_deterrence(name)
        print("   %-12s %10.6f %10.6f %10.6f %12.4f"
              % (name, fn.value(5.0), fn.value(20.0), fn.value(45.0),
                 fn.elasticity(20.0)))
    exp_fn = ExponentialDeterrence(0.08)
    pow_fn = PowerDeterrence(1.6)
    comb = CombinedDeterrence(-0.4, 0.05)
    tann = TannerDeterrence(0.4, 0.05)
    toplog = TopLogDeterrence(1.4)
    lnorm = LogNormalDeterrence(0.30, 8.0)
    custom = DeterrenceFunction(lambda c: 1.0 / (1.0 + 0.05 * c * c),
                                name="inverse-square", parameters={"k": 0.05})
    print("   Tanner(n=0.4) == Combined(alpha=-0.4): %s"
          % all(abs(tann.value(c) - comb.value(c)) < 1e-15
                for c in (3.0, 12.0, 40.0)))
    print("   analytic vs numerical derivative at c=15: %s"
          % ["%s %.3e" % (f.name, abs(f.derivative(15.0)
                                      - DeterrenceFunction.derivative(f, 15.0)))
             for f in (exp_fn, pow_fn, comb, toplog, lnorm)])
    print("   custom callable %s: f(10) = %.6f, params %s"
          % (custom.name, custom.value(10.0), custom.params()))
    dmat = deterrence_matrix(costs, exp_fn)
    print("   deterrence matrix: CBD->CBD %.6f, CBD->Suburb-E %.6f"
          % (dmat[0][0], dmat[0][3]))
    target = 12.0
    cal = calibrate_deterrence(costs, P, A, target, kind="exponential")
    print("   calibrated beta = %.6f -> mean cost %.6f (target %.4f, %s)"
          % (cal.beta, cal.mean_cost, cal.target_mean_cost,
             "converged" if cal.converged else "open"))

    # -- 4. trip distribution ----------------------------------------------
    print("\n-- 4. trip distribution: gravity, Furness and growth factors --")
    hand = furness([[1.0, 2.0], [3.0, 4.0]], [10.0, 20.0], [12.0, 18.0])
    t11 = 0.5 * (-68.0 + math.sqrt(68.0 * 68.0 + 4.0 * 240.0))
    print("   2x2 hand solve: T11 = %.12f, Furness gives %.12f (err %.2e)"
          % (t11, hand.matrix[0][0], abs(t11 - hand.matrix[0][0])))
    print("   odds ratio preserved: seed %.10f -> fitted %.10f"
          % (4.0 / 6.0, (hand.matrix[0][0] * hand.matrix[1][1])
             / (hand.matrix[0][1] * hand.matrix[1][0])))
    print("   margins exact to %.2e in %d iterations (entropy %.6f)"
          % (hand.max_error, hand.iterations, hand.entropy))
    flat_costs = uniform_matrix(zones.n, zones.n, 9.0)
    flat = doubly_constrained_gravity(P, A, flat_costs, exp_fn)
    total = math.fsum(P)
    worst = max(abs(flat.matrix[i][j] - P[i] * A[j] / total)
                for i in range(zones.n) for j in range(zones.n))
    print("   uniform costs -> T_ij = O_i D_j / T exactly: max error %.3e "
          "in %d iteration(s)" % (worst, flat.iterations))
    grav = doubly_constrained_gravity(P, A, costs, cal.function)
    print("   doubly constrained: total %.2f, mean cost %.4f min, %d iterations"
          % (grav.total, grav.mean_cost, grav.iterations))
    print("   row error %.2e, column error %.2e | A_1 = %.6e, B_1 = %.6e"
          % (grav.row_error, grav.col_error, grav.origin_factors[0],
             grav.destination_factors[0]))
    single = singly_constrained_gravity(P, A, costs, cal.function, "origin")
    dest = singly_constrained_gravity(P, A, costs, cal.function, "destination")
    uncon = unconstrained_gravity(P, A, costs, cal.function, k=1e-4)
    print("   singly (origin): rows exact %.2e, columns off %.4f"
          % (single.row_error, single.col_error))
    print("   singly (destination): columns exact %.2e | unconstrained total %.1f"
          % (dest.col_error, uncon.total))
    via_umbrella = gravity_model(P, A, costs, cal.function, constraint="doubly")
    print("   gravity_model umbrella agrees with the direct call: %.3e"
          % max(abs(via_umbrella.matrix[i][j] - grav.matrix[i][j])
                for i in range(zones.n) for j in range(zones.n)))
    base = grav.matrix
    growth_rows = [v * 1.22 for v in row_sums(base)]
    growth_cols = [v * 1.22 for v in col_sums(base)]
    uni = uniform_growth(base, 1.22)
    fr = fratar(base, growth_rows, growth_cols)
    de = detroit(base, growth_rows, growth_cols)
    av = average_growth_factor(base, growth_rows, growth_cols)
    print("   uniform growth 1.22: Fratar %.3e, Detroit %.3e, average %.3e "
          "from g*T" % tuple(
              max(abs(r.matrix[i][j] - uni.matrix[i][j])
                  for i in range(zones.n) for j in range(zones.n))
              for r in (fr, de, av)))
    skew_rows = [v * (1.0 + 0.1 * k) for k, v in enumerate(row_sums(base))]
    scale_f = math.fsum(skew_rows) / math.fsum(col_sums(base))
    skew_cols = [v * scale_f for v in col_sums(base)]
    fr2 = fratar(base, skew_rows, skew_cols)
    de2 = detroit(base, skew_rows, skew_cols)
    fu2 = furness(base, skew_rows, skew_cols)
    print("   skewed growth: Detroit == Furness to %.2e, but Fratar differs "
          "by %.4f trips"
          % (max(abs(de2.matrix[i][j] - fu2.matrix[i][j])
                 for i in range(zones.n) for j in range(zones.n)),
             max(abs(fr2.matrix[i][j] - fu2.matrix[i][j])
                 for i in range(zones.n) for j in range(zones.n))))
    cal_weights = deterrence_matrix(costs, cal.function)
    ent = entropy_maximising_matrix(cal_weights, P, A)
    print("   entropy-maximising matrix == doubly-constrained gravity: %.3e"
          % max(abs(ent.matrix[i][j] - grav.matrix[i][j])
                for i in range(zones.n) for j in range(zones.n)))
    print("   Wilson entropy %.4f | relative entropy vs the seed %.6f"
          % (wilson_entropy(grav.matrix),
             relative_entropy(grav.matrix, cal_weights)))
    three_d = ipf([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                  [[30.0, 70.0], [40.0, 60.0], [50.0, 50.0]])
    print("   3-D IPF margins: axis0 %s axis1 %s axis2 %s"
          % ([round(math.fsum(math.fsum(r) for r in blk), 6) for blk in three_d],
             [round(math.fsum(three_d[z][s][c] for z in range(2)
                              for c in range(2)), 6) for s in range(2)],
             [round(math.fsum(three_d[z][s][c] for z in range(2)
                              for s in range(2)), 6) for c in range(2)]))
    one_sided = furness_to_targets(base, row_targets=[v * 1.1
                                                      for v in row_sums(base)])
    print("   furness_to_targets with rows only: total %.2f (was %.2f)"
          % (one_sided.total, matrix_total(base)))
    io = intervening_opportunities(P, A, costs, 1.0 / math.fsum(A))
    print("   intervening opportunities: rows exact %.2e, mean cost %.4f"
          % (io.row_error, io.mean_cost))
    cd = competing_destinations(P, A, costs, cal.function, sigma=-0.25)
    print("   competing destinations (%s): mean cost %.4f vs gravity %.4f"
          % (cd.deterrence, cd.mean_cost, single.mean_cost))
    tld = trip_length_distribution(grav.matrix, costs, 8)
    print("   trip lengths: mean %.3f, median %.3f, sd %.3f min"
          % (tld.mean, tld.median, tld.std))
    print("   bands %s" % ["%.0f-%.0f: %.1f%%"
                          % (tld.edges[k], tld.edges[k + 1],
                             100.0 * tld.shares[k])
                          for k in range(len(tld.trips))])
    print("   mean_trip_cost cross-check %.6f == %.6f"
          % (mean_trip_cost(grav.matrix, costs), tld.mean))

    # -- 5. matrix operations ----------------------------------------------
    print("\n-- 5. matrix operations: sectors, K-factors and comparison --")
    sectors = SectorSystem(["core", "inner", "inner", "outer", "outer", "core"])
    sm = sector_matrix(grav.matrix, sectors)
    print("   %d zones -> %d sectors %s, total preserved %.4f -> %.4f"
          % (zones.n, sectors.n_sectors, sm.origins, matrix_total(grav.matrix),
             sm.total()))
    print("   sector groups %s" % sectors.groups())
    agg = aggregate_matrix(grav.matrix, [[0, 5], [1, 2], [3, 4]])
    print("   aggregate_matrix agrees with sector_matrix: %.3e"
          % max(abs(agg[s][t] - sm.values[s][t]) for s in range(3)
                for t in range(3)))
    disagg = disaggregate_matrix(sm.values, sectors, grav.matrix)
    print("   disaggregate back to zones: max error %.3e"
          % max(abs(disagg[i][j] - grav.matrix[i][j])
                for i in range(zones.n) for j in range(zones.n)))
    observed = [[v * (1.08 if i == j else 0.96) for j, v in enumerate(row)]
                for i, row in enumerate(grav.matrix)]
    kf = k_factors(observed, grav.matrix)
    fixed = apply_k_factors(grav.matrix, kf)
    print("   K-factors span %.4f to %.4f; applying them reproduces the "
          "observed matrix to %.3e"
          % (min(min(r) for r in kf), max(max(r) for r in kf),
             max(abs(fixed[i][j] - observed[i][j]) for i in range(zones.n)
                 for j in range(zones.n))))
    intra = estimate_intrazonals(strip_intrazonals(grav.matrix),
                                 method="nearest", costs=costs, neighbours=2,
                                 factor=0.6)
    share_intra = estimate_intrazonals(strip_intrazonals(grav.matrix),
                                       method="share", share=0.12)
    print("   intrazonals: nearest-neighbour share %.4f, fixed-share rule %.4f"
          % (intrazonal_share(intra), intrazonal_share(share_intra)))
    comp = compare_matrices(observed, grav.matrix)
    print("   comparison: RMSE %.4f (%.2f%%), R2 %.6f, slope %.4f, GEH<5 on "
          "%.0f%% of cells" % (comp.rmse, comp.pct_rmse, comp.r2, comp.slope,
                               comp.pct_geh_under_5))
    print("   geh(1000, 1050) = %.4f | worst cell GEH %.4f"
          % (geh(1000.0, 1050.0),
             max(max(r) for r in geh_matrix(observed, grav.matrix))))

    # -- 6. mode choice -----------------------------------------------------
    print("\n-- 6. mode choice: multinomial and nested logit --")
    modes = [Alternative("car", {"time": 22.0, "cost": 3.40, "wait": 0.0}),
             Alternative("bus", {"time": 38.0, "cost": 1.80, "wait": 6.0},
                         constant=-0.55),
             Alternative("rail", {"time": 29.0, "cost": 2.60, "wait": 4.0},
                         constant=-0.25),
             Alternative("cycle", {"time": 34.0, "cost": 0.10, "wait": 0.0},
                         constant=-0.90)]
    coef = {"time": -0.045, "cost": -0.28, "wait": -0.075}
    mnl = multinomial_logit(modes, coef, scale=1.0)
    print("   utilities %s" % ["%s %+.4f" % (mnl.names[k], mnl.utilities[k])
                               for k in range(4)])
    print("   MNL shares %s (sum %.15f), best %s, logsum %.6f"
          % ({k: round(v, 5) for k, v in sorted(mnl.as_dict().items())},
             math.fsum(mnl.probabilities), mnl.best, mnl.logsum))
    print("   linear_utility(car) = %.6f | logit_probabilities agree %.2e"
          % (linear_utility(modes[0], coef),
             max(abs(logit_probabilities(mnl.utilities)[k]
                     - mnl.probabilities[k]) for k in range(4))))
    shifted = logit_probabilities([v + 12.5 for v in mnl.utilities])
    rescaled = logit_probabilities([v * 4.0 for v in mnl.utilities], 0.25)
    print("   translation invariance %.2e | scale invariance %.2e"
          % (max(abs(shifted[k] - mnl.probabilities[k]) for k in range(4)),
             max(abs(rescaled[k] - mnl.probabilities[k]) for k in range(4))))
    nests = [Nest("private", ("car", "cycle"), scale=1.0),
             Nest("public", ("bus", "rail"), scale=2.4)]
    report = nesting_consistency(nests, 1.0)
    nl = nested_logit(modes, nests, coef, 1.0)
    print("   nesting: %s, thetas %s"
          % (report.message,
             {k: round(v, 4) for k, v in sorted(report.thetas.items())}))
    print("   nested shares %s (nest shares %s)"
          % ({k: round(v, 5) for k, v in sorted(nl.as_dict().items())},
             [round(v, 5) for v in nl.nest_probabilities]))
    print("   nest logsums %s, overall %.6f"
          % ([round(v, 5) for v in nl.nest_logsums], nl.logsum))
    flat_nests = [Nest("private", ("car", "cycle"), scale=1.0),
                  Nest("public", ("bus", "rail"), scale=1.0)]
    collapsed = nested_logit(modes, flat_nests, coef, 1.0)
    print("   every nest scale = mu -> collapses to the MNL: %.3e"
          % max(abs(collapsed.probabilities[k] - mnl.probabilities[k])
                for k in range(4)))
    ratio_before = nl.probabilities[0] / nl.probabilities[1]
    modes5 = modes + [Alternative("tram", {"time": 31.0, "cost": 2.2,
                                           "wait": 5.0}, constant=-0.35)]
    nests5 = [Nest("private", ("car", "cycle"), scale=1.0),
              Nest("public", ("bus", "rail", "tram"), scale=2.4)]
    nl5 = nested_logit(modes5, nests5, coef, 1.0)
    ratio_after = nl5.probabilities[0] / nl5.probabilities[1]
    mnl5 = multinomial_logit(modes5, coef, 1.0)
    print("   IIA: MNL car/bus ratio %.10f -> %.10f (unchanged)"
          % (mnl.probabilities[0] / mnl.probabilities[1],
             mnl5.probabilities[0] / mnl5.probabilities[1]))
    print("   nested car/bus ratio %.6f -> %.6f (IIA violated ACROSS nests "
          "by %.4f)" % (ratio_before, ratio_after,
                        abs(ratio_after - ratio_before)))
    print("   bus/rail ratio inside one nest is still IIA: %.10f -> %.10f"
          % (nl.probabilities[1] / nl.probabilities[2],
             nl5.probabilities[1] / nl5.probabilities[2]))
    mc = monte_carlo_logit(mnl.utilities, 1.0, draws=40000, seed=2024)
    print("   Gumbel Monte-Carlo (%d draws): mean max utility %.5f vs logsum "
          "+ gamma = %.5f (err %.4f)"
          % (mc["draws"], mc["meanMax"], mc["expectedMax"],
             abs(mc["meanMax"] - mc["expectedMax"])))
    print("   simulated shares %s vs analytic %s"
          % ([round(v, 4) for v in mc["shares"]],
             [round(v, 4) for v in mc["probabilities"]]))
    el = logit_elasticities(mnl.probabilities, coef["cost"],
                            [m.attributes["cost"] for m in modes],
                            attribute="cost")
    print("   cost elasticities: direct %s"
          % ["%s %.4f" % (mnl.names[k], el.direct[k]) for k in range(4)])
    print("   cross elasticity of bus wrt car cost = %.4f (= -beta x_car P_car)"
          % el.cross[1][0])
    bump = list(mnl.utilities)
    bump[0] += coef["cost"] * 0.01 * modes[0].attributes["cost"]
    after = logit_probabilities(bump)
    fd = ((after[0] - mnl.probabilities[0]) / mnl.probabilities[0]) / 0.01
    print("   finite-difference check of the direct elasticity: %.6f vs %.6f"
          % (fd, direct_elasticity(mnl.probabilities[0], coef["cost"],
                                   modes[0].attributes["cost"])))
    print("   cross_elasticity helper agrees: %.6f"
          % cross_elasticity(mnl.probabilities[0], coef["cost"],
                             modes[0].attributes["cost"]))
    print("   arc elasticity (demand 1000 -> 940, cost 3.40 -> 3.74) = %.4f"
          % arc_elasticity(1000.0, 940.0, 3.40, 3.74))
    print("   value of time = %.2f money units per hour"
          % value_of_time(coef["time"], coef["cost"]))
    print("   composite cost of the choice set = %.4f (cheapest mode %.4f)"
          % (composite_cost(mnl.utilities, 1.0, coef["cost"]),
             min(linear_utility(m, coef) / coef["cost"] for m in modes)))
    print("   logsum %.6f == logsum_accessibility %.6f"
          % (logsum(mnl.utilities), logsum_accessibility(mnl.utilities)))
    access = logsum_accessibility([[-0.1 * c for c in row] for row in costs])
    print("   zone accessibilities %s" % ["%.4f" % v for v in access])
    incr = incremental_logit(mnl.probabilities, [0.0, 0.30, 0.0, 0.0])
    piv = pivot_point([620.0, 180.0, 150.0, 50.0], [0.0, 0.30, 0.0, 0.0])
    print("   a +0.30 utility bus improvement: shares %s -> %s"
          % ([round(v, 4) for v in mnl.probabilities],
             [round(v, 4) for v in incr]))
    print("   pivot point on observed trips: %s (total held at %.1f)"
          % (["%.1f" % v for v in piv], math.fsum(piv)))

    # -- 7. departure time and peak spreading -------------------------------
    print("\n-- 7. departure-time choice and peak spreading --")
    slices = [TimeSlice("07:00-07:30", 7.0, 7.5, capacity=2200.0),
              TimeSlice("07:30-08:00", 7.5, 8.0, capacity=2200.0),
              TimeSlice("08:00-08:30", 8.0, 8.5, capacity=2200.0),
              TimeSlice("08:30-09:00", 8.5, 9.0, capacity=2200.0)]
    travel = [0.42, 0.50, 0.62, 0.55]
    dep = departure_time_choice(slices, travel, 9.0, demand=8000.0,
                                alpha=1.0, beta=0.5, gamma=2.0, scale=2.5)
    print("   arrivals %s" % ["%.3f" % v for v in dep.arrivals])
    print("   shares %s, peak slice %s, logsum %.5f"
          % ({k: round(v, 4) for k, v in dep.as_dict().items()}, dep.peak,
             dep.logsum))
    print("   schedule delay at a 30 min early arrival = %.4f"
          % schedule_delay_utility(8.5, 9.0, 0.5, alpha=1.0, beta=0.5,
                                   gamma=2.0))
    print("   PHF of [520, 610, 700, 580] 15-minute counts = %.4f"
          % peak_hour_factor([520.0, 610.0, 700.0, 580.0]))
    tod = time_of_day_split(matrix_total(grav.matrix),
                            {"AM": 0.28, "IP": 0.34, "PM": 0.30, "OP": 0.08})
    print("   time-of-day split %s"
          % {k: "%.1f" % v for k, v in sorted(tod.items())})
    am = apply_time_of_day(grav.matrix, 0.28, direction_split=0.75)
    print("   AM peak matrix total %.2f (%.1f%% of daily), PA->OD at 75/25"
          % (matrix_total(am), 100.0 * matrix_total(am)
             / matrix_total(grav.matrix)))
    spread = peak_spreading(dep.trips, [s.capacity for s in slices],
                            penalty=3.0, scale=1.0)
    print("   peak spreading: %s -> %s"
          % (["%.0f" % v for v in dep.trips],
             ["%.0f" % v for v in spread["demand"]]))
    print("   worst volume/capacity %.4f after %d iterations (%s)"
          % (spread["maxVolumeCapacity"], spread["iterations"],
             "converged" if spread["converged"] else "open"))

    # -- 8. the four-step loop ---------------------------------------------
    print("\n-- 8. the four-step loop with feedback --")
    bus_cost = [[1.7 * v + 6.0 for v in row] for row in costs]
    capacity = [[900.0 if i != j else 4000.0 for j in range(zones.n)]
                for i in range(zones.n)]
    cfg = FourStepConfig(
        productions=P, attractions=A, free_cost=costs,
        deterrence=get_deterrence("exponential", beta=cal.beta),
        mode_costs={"car": costs, "bus": bus_cost},
        mode_constants={"bus": -0.6}, cost_coefficient=-0.09,
        congested_mode="car", capacity=capacity, max_feedback=40,
        tol=1e-9, msa=False)
    loop = four_step_model(cfg)
    print("   congested loop: %d passes, converged %s, gap %.3e"
          % (len(loop.iterations), loop.converged, loop.gap))
    print("   mode shares %s, mean car cost %.4f min (free-flow %.4f)"
          % ({k: round(v, 4) for k, v in sorted(loop.mode_shares.items())},
             loop.mean_cost, mean_trip_cost(loop.matrix, costs)))
    print("   worst volume/capacity %.4f, composite-cost offset %.4f"
          % (loop.iterations[-1].max_volume_capacity, loop.composite_offset))
    print("   gap trace %s"
          % ["%.2e" % it.gap for it in loop.iterations[:6]])
    free_cfg = FourStepConfig(
        productions=P, attractions=A, free_cost=costs,
        deterrence=get_deterrence("exponential", beta=cal.beta),
        mode_costs={"car": costs, "bus": bus_cost},
        mode_constants={"bus": -0.6}, cost_coefficient=-0.09,
        congested_mode="car", max_feedback=12, tol=1e-12)
    free_loop = four_step_model(free_cfg)
    print("   with no congestion the loop is a fixed point: gap %.1e after "
          "%d passes (converged %s)"
          % (free_loop.gap, len(free_loop.iterations), free_loop.converged))
    blended = msa_blend([[1.0, 2.0], [3.0, 4.0]], [[3.0, 4.0], [5.0, 6.0]], 2)
    print("   msa_blend at k=2 is the midpoint: %s" % blended)
    print("   feedback_gap between two identical matrices = %.1f"
          % feedback_gap(grav.matrix, grav.matrix))

    # -- 9. OD estimation from link counts ----------------------------------
    print("\n-- 9. OD matrix estimation from link counts --")
    truth = [[round(v, 3) for v in row] for row in strip_intrazonals(grav.matrix)]
    props: Dict[str, Dict[Tuple[int, int], float]] = {}
    for i in range(zones.n):
        for j in range(zones.n):
            if i != j:
                props["L%d-%d" % (i, j)] = {(i, j): 1.0}
    props["screenline-N"] = {(i, 1): 1.0 for i in range(zones.n) if i != 1}
    asg = AssignmentProportions(links=sorted(props), shape=(zones.n, zones.n),
                                proportions=props)
    counts_v = link_flows(truth, asg)
    as_dict = assign_matrix_to_links(truth, asg)
    print("   %d links; screenline-N carries %.2f trips"
          % (len(asg.links), as_dict["screenline-N"]))
    prior = matrix_scale(truth, 0.65)
    gls = gls_matrix_estimation(prior, counts_v, asg, prior_weight=0.0)
    print("   GLS with complete counts recovers the matrix: max error %.3e "
          "(%d cells held at the prior)"
          % (max(abs(gls.matrix[i][j] - truth[i][j]) for i in range(zones.n)
                 for j in range(zones.n)), gls.detail["heldAtPrior"]))
    me2 = me2_estimation(prior, counts_v, asg)
    print("   ME2 entropy solution: max error %.3e in %d sweeps (%s)"
          % (max(abs(me2.matrix[i][j] - truth[i][j]) for i in range(zones.n)
                 for j in range(zones.n)), me2.iterations,
             "converged" if me2.converged else "open"))
    spiess = spiess_gradient(prior, counts_v, asg, max_iter=400)
    print("   Spiess gradient: objective %.3e after %d iterations, count "
          "RMSE %.3e" % (spiess.objective, spiess.iterations, spiess.fit.rmse))
    partial = AssignmentProportions(
        links=["screenline-N", "L0-3", "L3-0"], shape=(zones.n, zones.n),
        proportions={k: props[k] for k in ("screenline-N", "L0-3", "L3-0")})
    pcounts = link_flows(truth, partial)
    soft = gls_matrix_estimation(prior, pcounts, partial, prior_weight=1.0,
                                 count_weight=25.0)
    print("   with only %d counts GLS leans on the prior: count RMSE %.4f, "
          "matrix distance from the prior %.1f trips"
          % (len(partial.links), soft.fit.rmse, soft.prior_distance))
    pvar = [[max(1.0, prior[i][j]) for j in range(zones.n)]
            for i in range(zones.n)]
    bayes = bayesian_matrix_update(prior, pvar, pcounts, [4.0] * len(pcounts),
                                   partial)
    reduction = bayes.detail["varianceReduction"]
    print("   Bayesian update: posterior variance falls by up to %.1f%% and "
          "counts fit to RMSE %.4f"
          % (100.0 * max(max(r) for r in reduction), bayes.fit.rmse))
    paths = []
    for i in range(zones.n):
        for j in range(zones.n):
            if i != j:
                links_used = ["L%d-%d" % (i, j)]
                if j == 1:
                    links_used.append("screenline-N")
                paths.append((i, j, links_used, prior[i][j]))
    pfe = path_flow_estimator(paths, {k: v for k, v in zip(asg.links, counts_v)},
                              (zones.n, zones.n), max_iter=400)
    print("   path-flow estimator over %d paths: count RMSE %.3e, matrix "
          "error %.3e"
          % (pfe.detail["paths"], pfe.fit.rmse,
             max(abs(pfe.matrix[i][j] - truth[i][j]) for i in range(zones.n)
                 for j in range(zones.n))))
    fit = count_fit(counts_v, link_flows(gls.matrix, asg))
    print("   count fit of the GLS answer: R2 %.6f, slope %.6f, GEH<5 on "
          "%.0f%% of links" % (fit.r2, fit.slope, fit.pct_geh_under_5))

    # -- 10. demand scenarios ------------------------------------------------
    print("\n-- 10. seeded demand scenarios --")
    print("   available: %s" % ", ".join(SCENARIO_NAMES))
    commute = commuting_peak_scenario(zones, seed=17, trip_rate=1.15,
                                      beta=0.06)
    print("   commuting peak: %.1f trips, mean cost %.3f min, CBD attracts "
          "%.1f%%" % (commute.total(), commute.detail["meanCost"],
                      100.0 * commute.attractions[0] / commute.total()))
    print("   note: %s" % commute.notes[0])
    school = school_scenario(zones, seed=17)
    print("   school run: %.1f trips to zones %s, mean cost %.3f min"
          % (school.total(), school.detail["schoolZones"],
             school.detail["meanCost"]))
    stadium = stadium_scenario(zones, 5, 24000.0, seed=17,
                               release_minutes=40.0, step_minutes=5.0)
    print("   stadium: %.0f trips out of %s over %d steps, peak %.1f "
          "trips/min" % (stadium.total(), zones.ids()[5],
                         len(stadium.detail["profile"]),
                         stadium.detail["peakFlowPerMinute"]))
    wet = weather_adjustment(commute.matrix, "heavy_rain", seed=17)
    print("   heavy rain: demand x %.4f, mode shares %s"
          % (wet.detail["demandFactor"],
             {k: round(v, 4) for k, v in sorted(
                 wet.detail["modeShares"].items())}))
    inc = incident_adjustment(commute.matrix, [0], severity=0.6,
                              suppression=0.4, seed=17)
    print("   incident in the CBD: %.1f trips removed, %.1f redistributed, "
          "%.1f suppressed" % (inc.detail["removed"],
                               inc.detail["redistributed"],
                               inc.detail["suppressed"]))
    again = generate_scenario("commuting_peak", zones, seed=17, trip_rate=1.15,
                              beta=0.06)
    print("   generate_scenario reproduces it byte for byte: %s"
          % all(again.matrix[i][j] == commute.matrix[i][j]
                for i in range(zones.n) for j in range(zones.n)))

    # -- 11. synthetic population and trip lists -----------------------------
    print("\n-- 11. synthetic population and trip lists --")
    pop = synthesise_population(
        [int(v) for v in zones.vector("households")],
        [0.28, 0.33, 0.22, 0.17], [0.22, 0.48, 0.30], seed=99)
    print("   %d households, %d people, %.1f people per household"
          % (len(pop.households), pop.size(),
             pop.size() / max(1, len(pop.households))))
    print("   zone totals %s (targets %s)"
          % ([int(v) for v in pop.zone_totals],
             [int(v) for v in zones.vector("households")]))
    hh0 = pop.households[0]
    p0 = pop.persons[0]
    print("   first household: zone %d, size %d, %d cars, income %.0f, "
          "members %s" % (hh0.zone, hh0.size, hh0.cars, hh0.income,
                          list(hh0.persons)))
    print("   first person: age %d, employed %s, licence %s | %d households "
          "in zone 0" % (p0.age, p0.employed, p0.licence,
                         len(pop.households_in(0))))
    whole = integerise_matrix(am)
    print("   integerised AM matrix: %.4f -> %.0f trips (largest remainder "
          "preserves the total: %s)"
          % (matrix_total(am), matrix_total(whole),
             abs(matrix_total(whole) - round(matrix_total(am))) < 1e-9))
    print("   no cell moves by more than one trip: %s"
          % (max(abs(whole[i][j] - am[i][j]) for i in range(zones.n)
                 for j in range(zones.n)) < 1.0))
    stoch = integerise_matrix(am, method="stochastic", seed=5)
    print("   stochastic rounding total %.0f (unbiased, not exact)"
          % matrix_total(stoch))
    trip_list = expand_to_trip_list(
        whole, purpose="hbw",
        mode_shares={"car": 0.62, "bus": 0.16, "rail": 0.14, "cycle": 0.08},
        slices=slices,
        time_shares={s.name: w for s, w in zip(slices,
                                               [0.22, 0.28, 0.32, 0.18])},
        seed=99)
    rebuilt = trip_list_to_matrix(trip_list, (zones.n, zones.n))
    print("   expanded to %d trips; summing them back reproduces the matrix "
          "exactly: %s"
          % (len(trip_list), all(rebuilt[i][j] == whole[i][j]
                                 for i in range(zones.n)
                                 for j in range(zones.n))))
    by_mode = {m: matrix_total(trip_list_to_matrix(trip_list,
                                                   (zones.n, zones.n), mode=m))
               for m in ("car", "bus", "rail", "cycle")}
    print("   trips by mode %s (sum %.0f)"
          % ({k: int(v) for k, v in sorted(by_mode.items())},
             math.fsum(by_mode[k] for k in sorted(by_mode))))
    t0 = trip_list[0]
    print("   first trip: %d %d->%d by %s at %.3f h (purpose %s)"
          % (t0.ident, t0.origin, t0.destination, t0.mode, t0.departure,
             t0.purpose))

    # -- 12. payloads --------------------------------------------------------
    print("\n-- 12. frontend payloads --")
    produced = [MatrixMargins, GLMResult, TripEnds, BalanceResult,
                CalibrationResult, GravityResult, FurnessResult,
                GrowthFactorResult, TripLengthDistribution, MatrixComparison,
                ModeChoiceResult, NestingReport, NestedLogitResult,
                ElasticityResult, DepartureTimeResult, FourStepIteration,
                FourStepResult, ODEstimationResult, CountFit, ScenarioResult,
                Household, Person, Trip, SyntheticPopulation]
    live = [marg, pois, ends, bal, cal, grav, hand, uni, tld, comp, mnl,
            report, nl, el, dep, loop.iterations[0], loop, gls, fit, commute,
            hh0, p0, t0, pop]
    print("   %d result types produced and type-checked: %s"
          % (len(produced),
             all(isinstance(live[k], produced[k]) for k in range(len(produced)))))
    payload = build_travel_demand_payload(
        grav.matrix, zones=zones, costs=costs, deterrence=cal.function,
        generation=pois, distribution=grav, mode_choice=mnl, departure=dep,
        four_step=loop, estimation=gls, comparison=comp, scenario=commute)
    ui = build_a2ui_payload(grav.matrix, zones=zones, costs=costs,
                            distribution=grav, mode_choice=mnl,
                            four_step=loop, estimation=gls, comparison=comp)
    print("   payload keys: %s" % sorted(payload))
    print("   %d matrix cells, %d trip-length bands, %d deterrence points, "
          "%d feedback passes"
          % (len(payload["matrix"]["cells"]),
             len(payload["tripLengths"]["trips"]),
             len(payload["deterrence"]["curve"]),
             len(payload["fourStep"]["trace"])))
    print("   a2ui: %d stat cards, %d charts, %d table rows"
          % (len(ui["statCards"]), len(ui["charts"]),
             len(ui["tables"][0]["rows"])))
    print("   cards: %s" % ["%s=%s" % (c["label"], c["value"])
                            for c in ui["statCards"][:5]])
    print("   payload is JSON-serialisable: %d characters"
          % len(json.dumps(payload)))

    # -- 13. input validation ------------------------------------------------
    print("\n-- 13. input validation --")
    for bad in (lambda: furness([[1.0, 1.0], [1.0, 1.0]], [10.0, 10.0],
                                [5.0, 5.0]),
                lambda: get_deterrence("teleporter"),
                lambda: PowerDeterrence(1.5).value(0.0),
                lambda: nested_logit(modes, [Nest("all", ("car", "bus", "rail",
                                                          "cycle"), 0.4)],
                                     coef, 1.0),
                lambda: logit_probabilities([1.0, 2.0], scale=0.0),
                lambda: expand_to_trip_list([[1.5, 0.0], [0.0, 2.0]]),
                lambda: balance_trip_ends([1.0], [0.0], "attractions")):
        try:
            bad()
            print("   ERROR: no exception raised")
        except TravelDemandError as exc:
            print("   TravelDemandError: %s" % exc)

    print("\n" + "=" * 74)
    print("  demo OK")
    print("=" * 74)


if __name__ == "__main__":
    _demo()
