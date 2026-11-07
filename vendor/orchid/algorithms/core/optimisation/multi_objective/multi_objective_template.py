"""
Multi-Objective Optimisation -- a self-contained, zero-dependency toolkit
=========================================================================

Optimisation with *several conflicting objectives at once*, where there is no
single optimum but a **Pareto front** of trade-offs.  This module is the
dedicated multi-objective (and many-objective) counterpart to the sibling
``optimisation`` packages -- it supplies the Pareto machinery, quality
indicators, scalarisations, a fleet of state-of-the-art MOEAs, and the
multi-criteria decision-making tools needed to *pick* a solution off the front.

Pure Python, standard library only (``math`` / ``random`` / ``dataclasses`` /
``typing``) -- no NumPy, no SciPy, no pymoo/DEAP/jMetal.  Copy this one file into
a competition / hackathon / interview / embedded sandbox and go.

Layers
------
1.  **Pareto machinery** -- dominance (strict / weak / epsilon / constraint),
    fast non-dominated sort, crowding distance, front extraction.
2.  **Quality indicators** -- exact hypervolume (2-D sweep + WFG for any
    dimension) and a Monte-Carlo estimator, IGD / IGD+ / GD / GD+, additive and
    multiplicative epsilon, spacing, spread, R2, set coverage, MPFE.
3.  **Reference directions & scalarisation** -- Das-Dennis simplex lattice
    (single and two-layer), weighted sum, (augmented) Tchebycheff, PBI, ASF.
4.  **Normalisation** -- ideal / nadir estimation, extreme points, [0,1] scaling.
5.  **Operators** -- SBX crossover, polynomial mutation, DE, binary tournament,
    real-coded sampling (all overridable, so combinatorial problems drop in too).
6.  **Algorithms** -- NSGA-II, NSGA-III, MOEA/D, SPEA2, SMS-EMOA, IBEA.
7.  **Decision making (MCDM)** -- TOPSIS, compromise programming, pseudo-weights,
    weighted selection, knee-point / high-trade-off detection, an epsilon archive.
8.  **Benchmarks** -- ZDT1-3, DTLZ1-2, Schaffer, Kursawe, Fonseca-Fleming,
    Binh-Korn (constrained), each with analytic Pareto fronts where known.

Conventions
-----------
* Everything is internally **minimised**.  A :class:`MOProblem` may declare per
  objective ``senses`` of ``'min'`` / ``'max'``; maximised objectives are negated
  internally and un-negated on the way out, so the low-level indicator/dominance
  API can always assume minimisation.
* Every solver returns a uniform :class:`MOResult` (the non-dominated ``X`` / ``F``
  plus evaluation counts, an optional indicator ``history`` and a ``meta`` dict).
* Every randomised routine takes a ``seed`` (or ``rng``); a given seed reproduces
  the run exactly, independent of ``PYTHONHASHSEED``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple)

__version__ = "1.0.0"

__all__ = [
    # data structures
    "MOProblem", "Individual", "MOResult",
    # dominance & Pareto machinery
    "dominates", "weakly_dominates", "epsilon_dominates", "constraint_dominates",
    "non_dominated_sort", "crowding_distance", "find_pareto_front", "pareto_ranks",
    # indicators
    "hypervolume", "hypervolume_monte_carlo", "hypervolume_contributions",
    "generational_distance", "inverted_generational_distance", "igd_plus",
    "gd_plus", "epsilon_indicator", "epsilon_indicator_mult", "spacing", "spread",
    "r2_indicator", "set_coverage", "maximum_pareto_front_error",
    # reference directions & scalarisation
    "das_dennis", "reference_directions", "weighted_sum", "tchebycheff",
    "augmented_tchebycheff", "pbi", "asf",
    # normalisation
    "ideal_point", "nadir_point", "normalize", "extreme_points",
    # operators
    "random_sampling", "sbx_crossover", "polynomial_mutation", "de_operator",
    "binary_tournament",
    # algorithms
    "nsga2", "nsga3", "moead", "spea2", "sms_emoa", "ibea",
    # decision making
    "topsis", "compromise_programming", "pseudo_weights", "weighted_selection",
    "knee_point", "high_tradeoff_points", "ParetoArchive",
    # benchmarks
    "zdt1", "zdt2", "zdt3", "dtlz1", "dtlz2", "schaffer", "kursawe",
    "fonseca_fleming", "binh_korn",
    "zdt1_front", "zdt2_front", "zdt3_front", "dtlz2_front",
]

Vector = List[float]
_EPS = 1e-12
_INF = float("inf")


# ===========================================================================
# 0. SMALL HELPERS
# ===========================================================================
def _as_rng(seed: Any) -> random.Random:
    """Coerce ``seed`` into a ``random.Random``.  Pass an int/str for a fresh
    reproducible stream, or an existing ``random.Random`` to thread state."""
    if isinstance(seed, random.Random):
        return seed
    return random.Random(seed)


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(a, b)))


def _dist_plus(a: Sequence[float], z: Sequence[float]) -> float:
    """IGD+/GD+ modified distance from solution ``a`` to reference ``z`` (both
    minimised): only the components where ``a`` is *worse* than ``z`` count, which
    makes the indicator weakly Pareto-compliant."""
    return math.sqrt(math.fsum(max(ai - zi, 0.0) ** 2 for ai, zi in zip(a, z)))


def _mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return math.fsum(xs) / len(xs) if xs else 0.0


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# ===========================================================================
# 1. DOMINANCE & PARETO MACHINERY
# ===========================================================================
def dominates(a: Sequence[float], b: Sequence[float],
              senses: Optional[Sequence[int]] = None) -> bool:
    """Return ``True`` if ``a`` Pareto-dominates ``b`` (minimisation by default):
    ``a`` is no worse in every objective and strictly better in at least one.

    ``senses`` optionally gives a per-objective sign (+1 minimise, -1 maximise);
    without it pure minimisation is assumed."""
    if senses is None:
        better = False
        for ai, bi in zip(a, b):
            if ai > bi:
                return False
            if ai < bi:
                better = True
        return better
    if len(senses) != len(a):
        # a short senses list would silently zip-truncate and ignore objectives
        raise ValueError("senses must have one entry per objective.")
    better = False
    for ai, bi, s in zip(a, b, senses):
        ai, bi = s * ai, s * bi
        if ai > bi:
            return False
        if ai < bi:
            better = True
    return better


def weakly_dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """``a`` is no worse than ``b`` in every objective (minimisation)."""
    return all(ai <= bi for ai, bi in zip(a, b))


def epsilon_dominates(a: Sequence[float], b: Sequence[float],
                      epsilon: float) -> bool:
    """Additive epsilon-dominance: ``a`` dominates ``b`` once ``a`` is shifted to
    be ``epsilon`` better in every objective (used for epsilon-archives)."""
    better = False
    for ai, bi in zip(a, b):
        if ai - epsilon > bi:
            return False
        if ai - epsilon < bi:
            better = True
    return better


def constraint_dominates(fa: Sequence[float], cva: float,
                         fb: Sequence[float], cvb: float) -> bool:
    """Deb's constraint-domination between two solutions with objective vectors
    ``fa``/``fb`` and total constraint violations ``cva``/``cvb`` (>= 0, 0 means
    feasible): a feasible solution beats any infeasible one; among infeasibles the
    smaller violation wins; among feasibles ordinary Pareto dominance applies."""
    if cva <= 0.0 and cvb <= 0.0:
        return dominates(fa, fb)
    if cva <= 0.0:
        return True
    if cvb <= 0.0:
        return False
    return cva < cvb


def non_dominated_sort(F: Sequence[Sequence[float]],
                       cv: Optional[Sequence[float]] = None
                       ) -> List[List[int]]:
    """Deb's *fast non-dominated sort* (minimisation).  Partition the indices of
    ``F`` into successive Pareto fronts; ``fronts[0]`` is the non-dominated set,
    ``fronts[1]`` the set non-dominated once ``fronts[0]`` is removed, and so on.

    ``cv`` optionally supplies per-solution constraint violations, switching to
    constraint-domination.  Runs in O(M n^2)."""
    n = len(F)
    if n == 0:
        return []
    if any(v != v for f in F for v in f):
        # NaN compares False both ways, so a NaN objective would falsely "dominate"
        # finite points and be ranked best -- reject it rather than corrupt the sort.
        raise ValueError("objective vectors must be finite (found NaN).")

    def dom(i: int, j: int) -> bool:
        if cv is None:
            return dominates(F[i], F[j])
        return constraint_dominates(F[i], cv[i], F[j], cv[j])

    dominated: List[List[int]] = [[] for _ in range(n)]
    n_dominated = [0] * n
    fronts: List[List[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dom(p, q):
                dominated[p].append(q)
            elif dom(q, p):
                n_dominated[p] += 1
        if n_dominated[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt: List[int] = []
        for p in fronts[i]:
            for q in dominated[p]:
                n_dominated[q] -= 1
                if n_dominated[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts


def crowding_distance(F: Sequence[Sequence[float]],
                      indices: Optional[Sequence[int]] = None) -> Dict[int, float]:
    """NSGA-II crowding distance for one front: the sum over objectives of the
    normalised gap between each point's neighbours.  Boundary points get infinite
    distance so the extremes are always preserved.  Returns ``{index: distance}``."""
    if indices is None:
        indices = list(range(len(F)))
    indices = list(indices)
    dist = {i: 0.0 for i in indices}
    if len(indices) <= 2:
        for i in indices:
            dist[i] = _INF
        return dist
    m = len(F[indices[0]])
    for k in range(m):
        order = sorted(indices, key=lambda i: F[i][k])
        fmin = F[order[0]][k]
        fmax = F[order[-1]][k]
        dist[order[0]] = _INF
        dist[order[-1]] = _INF
        span = fmax - fmin
        if span <= _EPS or not math.isfinite(span):
            # a zero range gives no ordering info; an infinite range (an inf
            # objective in the front) would make (inf - finite)/inf == NaN and
            # silently corrupt the crowding-based selection.
            continue
        for j in range(1, len(order) - 1):
            if dist[order[j]] == _INF:
                continue
            dist[order[j]] += (F[order[j + 1]][k] - F[order[j - 1]][k]) / span
    return dist


def find_pareto_front(F: Sequence[Sequence[float]],
                      cv: Optional[Sequence[float]] = None) -> List[int]:
    """Indices of the non-dominated (first) front of ``F``."""
    fronts = non_dominated_sort(F, cv)
    return fronts[0] if fronts else []


def pareto_ranks(F: Sequence[Sequence[float]],
                 cv: Optional[Sequence[float]] = None) -> List[int]:
    """Per-solution Pareto rank (0 = non-dominated front, 1 = next, ...)."""
    fronts = non_dominated_sort(F, cv)
    ranks = [0] * len(F)
    for r, front in enumerate(fronts):
        for i in front:
            ranks[i] = r
    return ranks


# ===========================================================================
# 2. QUALITY INDICATORS
# ===========================================================================
def _filter_for_hv(front: Sequence[Sequence[float]],
                   ref: Sequence[float]) -> List[Vector]:
    """Keep only points that dominate the reference point, reduced to their
    non-dominated subset (the hypervolume of a set equals that of its front)."""
    if front and len(front[0]) != len(ref):
        # a dimension mismatch would silently zip-truncate the filter (front < ref)
        # or crash cryptically in the 2-D sweep (front > ref).
        raise ValueError("each point must have the same dimension as the "
                         "reference point.")
    pts = [list(p) for p in front if all(pi <= ri for pi, ri in zip(p, ref))]
    keep: List[Vector] = []
    for i, p in enumerate(pts):
        if any(j != i and weakly_dominates(q, p) and q != p for j, q in enumerate(pts)):
            continue
        keep.append(p)
    # de-duplicate exact repeats that survived the strict-inequality filter
    unique: List[Vector] = []
    seen = set()
    for p in keep:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _hv_2d(points: Sequence[Sequence[float]], ref: Sequence[float]) -> float:
    """Exact 2-D hypervolume by an O(n log n) sweep of the staircase."""
    pts = sorted(points, key=lambda p: p[0])
    hv = 0.0
    prev_f1 = ref[1]
    for f0, f1 in pts:
        if f1 >= prev_f1:                       # dominated by an earlier point
            continue
        hv += (ref[0] - f0) * (prev_f1 - f1)
        prev_f1 = f1
    return hv


def _inclhv(p: Sequence[float], ref: Sequence[float]) -> float:
    vol = 1.0
    for pi, ri in zip(p, ref):
        vol *= (ri - pi)
    return vol


def _limit_set(pl: Sequence[Sequence[float]], k: int) -> List[Vector]:
    """WFG ``limitset``: worst-case (component max) of point ``k`` against each
    later point, reduced to its non-dominated subset."""
    d = len(pl[k])
    limited = [[max(pl[k][i], pl[j][i]) for i in range(d)]
               for j in range(k + 1, len(pl))]
    keep: List[Vector] = []
    for i, p in enumerate(limited):
        if any(j != i and weakly_dominates(q, p) and q != p
               for j, q in enumerate(limited)):
            continue
        keep.append(p)
    return keep


def _hv_wfg(pl: List[Vector], ref: Sequence[float]) -> float:
    """Exact hypervolume for any dimension via the WFG recursion (While et al.
    2012): HV = sum of the *exclusive* hypervolumes of each point."""
    total = 0.0
    for k in range(len(pl)):
        excl = _inclhv(pl[k], ref)
        limited = _limit_set(pl, k)
        if limited:
            excl -= _hv_wfg(limited, ref)
        total += excl
    return total


def hypervolume(front: Sequence[Sequence[float]],
                ref: Sequence[float]) -> float:
    """Exact **hypervolume** dominated by ``front`` and bounded by the reference
    point ``ref`` (minimisation; ``ref`` must be worse than every point kept).
    The single most informative Pareto-set quality indicator -- it captures both
    convergence and spread and is Pareto-compliant.

    Uses an exact O(n log n) sweep for 2-D and the exact WFG recursion for any
    higher dimension; for large high-dimensional fronts prefer
    :func:`hypervolume_monte_carlo`."""
    d = len(ref)
    pts = _filter_for_hv(front, ref)
    if not pts:
        return 0.0
    if d == 1:
        return ref[0] - min(p[0] for p in pts)
    if d == 2:
        return _hv_2d(pts, ref)
    return _hv_wfg(sorted(pts, key=lambda p: p[0]), ref)


def hypervolume_monte_carlo(front: Sequence[Sequence[float]],
                            ref: Sequence[float], *,
                            lower: Optional[Sequence[float]] = None,
                            n_samples: int = 100_000, seed: Any = 0) -> float:
    """Monte-Carlo estimate of the hypervolume -- draw uniform points in the box
    ``[lower, ref]`` and measure the fraction dominated by the front.  Scales to
    many objectives where exact WFG becomes expensive; ``lower`` defaults to the
    componentwise minimum of the front."""
    pts = _filter_for_hv(front, ref)
    if not pts:
        return 0.0
    d = len(ref)
    if lower is None:
        lower = [min(p[i] for p in pts) for i in range(d)]
    # check every axis: an even number of inverted (lower > ref) axes would make
    # the product box positive and slip past a plain ``box <= 0`` test.
    if any(ref[i] <= lower[i] for i in range(d)):
        return 0.0
    box = 1.0
    for i in range(d):
        box *= (ref[i] - lower[i])
    rng = _as_rng(seed)
    hits = 0
    for _ in range(n_samples):
        s = [rng.uniform(lower[i], ref[i]) for i in range(d)]
        for p in pts:
            if all(p[i] <= s[i] for i in range(d)):
                hits += 1
                break
    return box * hits / n_samples


def hypervolume_contributions(front: Sequence[Sequence[float]],
                              ref: Sequence[float]) -> List[float]:
    """Exclusive hypervolume contribution of each point -- how much total
    hypervolume is lost if that point is removed.  The basis of SMS-EMOA
    selection; points are returned in the order of ``front``."""
    total = hypervolume(front, ref)
    contribs = []
    for i in range(len(front)):
        without = [p for j, p in enumerate(front) if j != i]
        contribs.append(total - hypervolume(without, ref))
    return contribs


def generational_distance(approx: Sequence[Sequence[float]],
                          reference: Sequence[Sequence[float]],
                          p: float = 2.0) -> float:
    """Generational Distance: the ``p``-averaged distance from each solution to
    the nearest point of a reference Pareto front.  Measures *convergence* (how
    close the set is to the true front); lower is better."""
    if not approx:
        return _INF
    d = [min(_dist(a, r) for r in reference) for a in approx]
    return (math.fsum(di ** p for di in d) / len(approx)) ** (1.0 / p)


def inverted_generational_distance(approx: Sequence[Sequence[float]],
                                   reference: Sequence[Sequence[float]],
                                   p: float = 2.0) -> float:
    """Inverted Generational Distance: the ``p``-averaged distance from each
    *reference* point to the nearest solution.  Captures convergence *and*
    diversity in one number (lower is better) -- the standard offline metric."""
    if not approx:
        return _INF
    d = [min(_dist(r, a) for a in approx) for r in reference]
    return (math.fsum(di ** p for di in d) / len(reference)) ** (1.0 / p)


def igd_plus(approx: Sequence[Sequence[float]],
             reference: Sequence[Sequence[float]]) -> float:
    """IGD+ -- the weakly Pareto-compliant variant of IGD using the modified
    distance that only penalises objectives on which a solution is dominated."""
    if not approx:
        return _INF
    d = [min(_dist_plus(a, r) for a in approx) for r in reference]
    return _mean(d)


def gd_plus(approx: Sequence[Sequence[float]],
            reference: Sequence[Sequence[float]]) -> float:
    """GD+ -- Pareto-compliant Generational Distance (modified distance)."""
    if not approx:
        return _INF
    d = [min(_dist_plus(a, r) for r in reference) for a in approx]
    return _mean(d)


def epsilon_indicator(approx: Sequence[Sequence[float]],
                      reference: Sequence[Sequence[float]]) -> float:
    """Additive (unary) epsilon indicator: the smallest ``eps`` such that every
    reference point is additively eps-dominated by some solution.  A hard,
    Pareto-compliant convergence measure; lower is better."""
    worst = -_INF
    for r in reference:
        best = _INF
        for a in approx:
            gap = max(ai - ri for ai, ri in zip(a, r))
            best = min(best, gap)
        worst = max(worst, best)
    return worst


def epsilon_indicator_mult(approx: Sequence[Sequence[float]],
                           reference: Sequence[Sequence[float]]) -> float:
    """Multiplicative epsilon indicator: the smallest factor ``eps`` such that
    ``eps * a`` covers each reference point.  As a ratio it is only defined for
    **strictly positive** objectives; a zero or negative objective raises rather
    than dividing by zero (use :func:`epsilon_indicator` for signed objectives)."""
    if any(ri <= 0.0 for r in reference for ri in r) or \
            any(ai <= 0.0 for a in approx for ai in a):
        raise ValueError("multiplicative epsilon requires strictly positive "
                         "objectives; use epsilon_indicator for signed ones.")
    worst = -_INF
    for r in reference:
        best = _INF
        for a in approx:
            ratio = max(ai / ri for ai, ri in zip(a, r))
            best = min(best, ratio)
        worst = max(worst, best)
    return worst


def spacing(front: Sequence[Sequence[float]]) -> float:
    """Schott's spacing metric: the standard deviation of nearest-neighbour L1
    distances within the front.  Measures how *evenly* points are distributed
    (0 = perfectly uniform); says nothing about convergence."""
    n = len(front)
    if n < 2:
        return 0.0
    d = []
    for i in range(n):
        nn = _INF
        for j in range(n):
            if i == j:
                continue
            l1 = math.fsum(abs(a - b) for a, b in zip(front[i], front[j]))
            nn = min(nn, l1)
        d.append(nn)
    dbar = _mean(d)
    return math.sqrt(math.fsum((dbar - di) ** 2 for di in d) / (n - 1))


def spread(front: Sequence[Sequence[float]],
           extremes: Optional[Sequence[Sequence[float]]] = None) -> float:
    """Deb's spread / diversity metric ``Delta`` for a 2-D front: combines the
    spacing of consecutive points with how well the extremes are covered.  Lower
    is better (0 = perfectly uniform and covering the true extremes).

    Delta's sort-into-a-chain construction is only meaningful for two objectives;
    for more, use :func:`spacing` (dimension-agnostic)."""
    if front and len(front[0]) != 2:
        raise ValueError("spread (Deb's Delta) is a 2-objective metric; "
                         "use spacing for more objectives.")
    pts = sorted(front, key=lambda p: p[0])
    n = len(pts)
    if n < 2:
        return 0.0
    consec = [_dist(pts[i], pts[i + 1]) for i in range(n - 1)]
    dbar = _mean(consec)
    if extremes is not None and len(extremes) >= 2:
        df = _dist(pts[0], extremes[0])
        dl = _dist(pts[-1], extremes[-1])
    else:
        df = dl = 0.0
    num = df + dl + math.fsum(abs(di - dbar) for di in consec)
    den = df + dl + (n - 1) * dbar
    return num / den if den > _EPS else 0.0


def r2_indicator(approx: Sequence[Sequence[float]],
                 weights: Sequence[Sequence[float]],
                 ideal: Optional[Sequence[float]] = None) -> float:
    """Unary R2 indicator: the mean over a set of weight vectors of the best
    (minimum) weighted-Tchebycheff utility achieved by the set.  A cheap,
    weakly Pareto-compliant alternative to the hypervolume; lower is better.

    ``ideal`` is the FIXED reference (utopian) point and defaults to the origin.
    It must be the *same* across sets being compared -- deriving it from ``approx``
    would make R2 translation-invariant and unable to tell a converged front from
    a shifted one, so pass the common/true ideal for a meaningful comparison."""
    if not approx:
        return _INF
    if not weights:
        raise ValueError("r2_indicator needs at least one weight vector.")
    m = len(approx[0])
    if ideal is None:
        ideal = [0.0] * m
    total = 0.0
    for w in weights:
        best = min(tchebycheff(a, w, ideal) for a in approx)
        total += best
    return total / len(weights)


def set_coverage(A: Sequence[Sequence[float]],
                 B: Sequence[Sequence[float]]) -> float:
    """C-metric ``C(A, B)``: the fraction of ``B`` weakly dominated by at least
    one member of ``A``.  ``C(A,B)=1`` means ``A`` covers all of ``B``; report
    both ``C(A,B)`` and ``C(B,A)`` since the measure is not symmetric."""
    if not B:
        return 0.0
    covered = 0
    for b in B:
        if any(weakly_dominates(a, b) for a in A):
            covered += 1
    return covered / len(B)


def maximum_pareto_front_error(approx: Sequence[Sequence[float]],
                               reference: Sequence[Sequence[float]]) -> float:
    """Maximum Pareto-front error: the largest distance from any solution to the
    reference front (the worst-case convergence gap)."""
    if not approx:
        return _INF
    return max(min(_dist(a, r) for r in reference) for a in approx)


# ===========================================================================
# 3. REFERENCE DIRECTIONS & SCALARISATION
# ===========================================================================
def das_dennis(n_obj: int, n_partitions: int) -> List[Vector]:
    """Das-Dennis structured weight vectors: all points on the unit simplex whose
    coordinates are multiples of ``1/n_partitions``.  Produces
    ``C(n_partitions + n_obj - 1, n_obj - 1)`` evenly spread directions -- the
    backbone of NSGA-III and MOEA/D."""
    if n_obj < 1:
        raise ValueError("n_obj must be >= 1.")
    if n_partitions < 1:
        return [[1.0 / n_obj] * n_obj]
    result: List[Vector] = []

    def recurse(depth: int, remaining: int, current: List[float]) -> None:
        if depth == n_obj - 1:
            result.append(current + [remaining / n_partitions])
            return
        for i in range(remaining + 1):
            recurse(depth + 1, remaining - i, current + [i / n_partitions])

    recurse(0, n_partitions, [])
    return result


def reference_directions(n_obj: int, n_partitions: int, *,
                         inner_partitions: Optional[int] = None,
                         scaling: float = 0.5) -> List[Vector]:
    """Reference directions on the unit simplex.  For many objectives a single
    Das-Dennis layer leaves the interior empty, so passing ``inner_partitions``
    adds a second, shrunk layer (Deb & Jain's two-layer construction)."""
    outer = das_dennis(n_obj, n_partitions)
    if inner_partitions is None or n_obj < 3:
        return outer
    inner = das_dennis(n_obj, inner_partitions)
    centre = 1.0 / n_obj
    shifted = [[scaling * w + (1.0 - scaling) * centre for w in v] for v in inner]
    return outer + shifted


def _check_weights(f: Sequence[float], w: Sequence[float]) -> None:
    if len(w) != len(f):
        # a weight vector shorter/longer than f would silently zip-truncate and
        # scalarise only some objectives.
        raise ValueError("weight vector must match the objective dimension.")


def weighted_sum(f: Sequence[float], w: Sequence[float]) -> float:
    """Weighted-sum scalarisation ``sum w_i f_i`` (minimisation).  Simple and
    fast but can only reach the convex part of a Pareto front."""
    _check_weights(f, w)
    return math.fsum(wi * fi for wi, fi in zip(w, f))


def tchebycheff(f: Sequence[float], w: Sequence[float],
                ideal: Sequence[float]) -> float:
    """Weighted Tchebycheff scalarisation ``max_i w_i |f_i - z*_i|`` about the
    ideal point ``z*``.  Reaches non-convex parts of the front that weighted-sum
    misses; zero weights are floored to a tiny value."""
    _check_weights(f, w)
    return max((max(wi, _EPS) * abs(fi - zi))
               for fi, wi, zi in zip(f, w, ideal))


def augmented_tchebycheff(f: Sequence[float], w: Sequence[float],
                          ideal: Sequence[float], rho: float = 0.05) -> float:
    """Augmented Tchebycheff: the Tchebycheff term plus a small ``rho``-weighted
    sum term that removes weakly-Pareto-optimal ties."""
    _check_weights(f, w)
    diffs = [max(wi, _EPS) * abs(fi - zi)
             for fi, wi, zi in zip(f, w, ideal)]
    return max(diffs) + rho * math.fsum(diffs)


def pbi(f: Sequence[float], w: Sequence[float], ideal: Sequence[float],
        theta: float = 5.0) -> float:
    """Penalty-based Boundary Intersection: ``d1 + theta * d2`` where ``d1`` is
    the distance along the weight line from the ideal point and ``d2`` the
    perpendicular distance to it.  MOEA/D's most effective aggregation for
    well-spread fronts."""
    _check_weights(f, w)
    wnorm = math.sqrt(math.fsum(wi * wi for wi in w))
    if wnorm <= _EPS:
        wnorm = _EPS
    diff = [fi - zi for fi, zi in zip(f, ideal)]
    d1 = abs(math.fsum(di * wi for di, wi in zip(diff, w))) / wnorm
    proj = [d1 * wi / wnorm for wi in w]
    d2 = math.sqrt(math.fsum((di - pi) ** 2 for di, pi in zip(diff, proj)))
    return d1 + theta * d2


def asf(f: Sequence[float], w: Sequence[float],
        ideal: Sequence[float]) -> float:
    """Wierzbicki achievement scalarising function ``max_i (f_i - z_i) / w_i``,
    used to locate extreme points during NSGA-III normalisation."""
    _check_weights(f, w)
    return max((fi - zi) / max(wi, _EPS)
               for fi, wi, zi in zip(f, w, ideal))


# ===========================================================================
# 4. NORMALISATION
# ===========================================================================
def ideal_point(F: Sequence[Sequence[float]]) -> Vector:
    """Componentwise minimum (the ideal / utopian objective vector)."""
    if not F:
        raise ValueError("ideal_point needs at least one objective vector.")
    m = len(F[0])
    return [min(f[k] for f in F) for k in range(m)]


def nadir_point(F: Sequence[Sequence[float]]) -> Vector:
    """Componentwise maximum over the *non-dominated* front (the nadir vector)."""
    if not F:
        raise ValueError("nadir_point needs at least one objective vector.")
    front = [F[i] for i in find_pareto_front(F)]
    if not front:
        front = list(F)
    m = len(front[0])
    return [max(f[k] for f in front) for k in range(m)]


def extreme_points(F: Sequence[Sequence[float]],
                   ideal: Sequence[float]) -> List[Vector]:
    """The ``m`` extreme points that minimise the axis-aligned ASF -- used to
    build the linear hyperplane / intercepts for NSGA-III normalisation."""
    m = len(ideal)
    extremes: List[Vector] = []
    for k in range(m):
        w = [1e-6] * m
        w[k] = 1.0
        best_val = _INF
        best_f: Vector = list(F[0])
        for f in F:
            val = asf(f, w, ideal)
            if val < best_val:
                best_val = val
                best_f = list(f)
        extremes.append(best_f)
    return extremes


def normalize(F: Sequence[Sequence[float]],
              ideal: Optional[Sequence[float]] = None,
              nadir: Optional[Sequence[float]] = None
              ) -> List[Vector]:
    """Scale objectives into ``[0, 1]`` using the ideal and nadir points
    (degenerate axes with zero range collapse to 0)."""
    if not F:
        return []
    if ideal is None:
        ideal = ideal_point(F)
    if nadir is None:
        nadir = nadir_point(F)
    m = len(ideal)
    out = []
    for f in F:
        row = []
        for k in range(m):
            span = nadir[k] - ideal[k]
            row.append((f[k] - ideal[k]) / span if span > _EPS else 0.0)
        out.append(row)
    return out


# ===========================================================================
# 5. GENETIC OPERATORS
# ===========================================================================
def random_sampling(bounds: Sequence[Tuple[float, float]], n: int,
                    rng: random.Random) -> List[Vector]:
    """``n`` real-coded individuals sampled uniformly inside ``bounds``."""
    return [[rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(n)]


def _betaq(rand: float, alpha: float, eta: float) -> float:
    if rand <= 1.0 / alpha:
        return (rand * alpha) ** (1.0 / (eta + 1.0))
    return (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))


def sbx_crossover(p1: Sequence[float], p2: Sequence[float],
                  bounds: Sequence[Tuple[float, float]], *,
                  eta: float = 15.0, prob: float = 0.9, prob_var: float = 0.5,
                  rng: Optional[random.Random] = None) -> Tuple[Vector, Vector]:
    """Simulated Binary Crossover (Deb & Agrawal): the real-coded analogue of
    single-point binary crossover, with a spread controlled by the distribution
    index ``eta`` (larger = children closer to parents).  Respects box bounds."""
    rng = rng or _as_rng(0)
    c1, c2 = list(p1), list(p2)
    if rng.random() > prob:
        return c1, c2
    for i in range(len(p1)):
        if rng.random() > prob_var:
            continue
        x1, x2 = p1[i], p2[i]
        if abs(x1 - x2) <= 1e-14:
            continue
        lo, hi = bounds[i]
        if x2 < x1:
            x1, x2 = x2, x1
        rand = rng.random()
        beta = 1.0 + 2.0 * (x1 - lo) / (x2 - x1)
        alpha = 2.0 - beta ** (-(eta + 1.0))
        y1 = 0.5 * ((x1 + x2) - _betaq(rand, alpha, eta) * (x2 - x1))
        beta = 1.0 + 2.0 * (hi - x2) / (x2 - x1)
        alpha = 2.0 - beta ** (-(eta + 1.0))
        y2 = 0.5 * ((x1 + x2) + _betaq(rand, alpha, eta) * (x2 - x1))
        y1 = _clip(y1, lo, hi)
        y2 = _clip(y2, lo, hi)
        if rng.random() < 0.5:
            c1[i], c2[i] = y2, y1
        else:
            c1[i], c2[i] = y1, y2
    return c1, c2


def polynomial_mutation(x: Sequence[float],
                        bounds: Sequence[Tuple[float, float]], *,
                        eta: float = 20.0, prob: Optional[float] = None,
                        rng: Optional[random.Random] = None) -> Vector:
    """Polynomial mutation (Deb): perturb each variable by a polynomial-shaped
    kernel bounded to its box, with per-variable probability ``prob`` (default
    ``1/n``) and sharpness ``eta``."""
    rng = rng or _as_rng(0)
    n = len(x)
    if prob is None:
        prob = 1.0 / n
    y = list(x)
    mut_pow = 1.0 / (eta + 1.0)
    for i in range(n):
        if rng.random() > prob:
            continue
        lo, hi = bounds[i]
        if hi - lo <= _EPS:
            continue
        xi = x[i]
        d1 = (xi - lo) / (hi - lo)
        d2 = (hi - xi) / (hi - lo)
        r = rng.random()
        if r < 0.5:
            val = 2.0 * r + (1.0 - 2.0 * r) * (1.0 - d1) ** (eta + 1.0)
            deltaq = val ** mut_pow - 1.0
        else:
            val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (1.0 - d2) ** (eta + 1.0)
            deltaq = 1.0 - val ** mut_pow
        y[i] = _clip(xi + deltaq * (hi - lo), lo, hi)
    return y


def de_operator(target: Sequence[float], a: Sequence[float], b: Sequence[float],
                c: Sequence[float], bounds: Sequence[Tuple[float, float]], *,
                F: float = 0.5, cr: float = 0.9,
                rng: Optional[random.Random] = None) -> Vector:
    """Differential-evolution ``rand/1/bin`` trial vector ``a + F (b - c)`` with
    binomial crossover against ``target`` -- the reproduction operator of
    MOEA/D-DE, robust on rotated and non-separable problems."""
    rng = rng or _as_rng(0)
    n = len(target)
    jrand = rng.randrange(n)
    y = list(target)
    for i in range(n):
        if rng.random() < cr or i == jrand:
            lo, hi = bounds[i]
            y[i] = _clip(a[i] + F * (b[i] - c[i]), lo, hi)
    return y


def binary_tournament(pop: Sequence["Individual"],
                      rng: random.Random,
                      key: Optional[Callable[["Individual", "Individual"], "Individual"]]
                      = None) -> "Individual":
    """Binary tournament selection: pick two individuals at random and return the
    better.  The default comparator is NSGA-II's (feasibility, then Pareto rank,
    then larger crowding distance)."""
    i = rng.randrange(len(pop))
    j = rng.randrange(len(pop))
    a, b = pop[i], pop[j]
    if key is not None:
        return key(a, b)
    return _crowded_compare(a, b)


def _crowded_compare(a: "Individual", b: "Individual") -> "Individual":
    if (a.cv > 0.0) or (b.cv > 0.0):
        if a.cv != b.cv:
            return a if a.cv < b.cv else b
    if a.rank != b.rank:
        return a if a.rank < b.rank else b
    return a if a.crowding > b.crowding else b


# ===========================================================================
# 6. DATA STRUCTURES
# ===========================================================================
@dataclass
class Individual:
    """One candidate solution: decision vector ``x``, minimisation-normalised
    objective vector ``f``, constraint violation ``cv`` (0 = feasible) plus
    per-algorithm bookkeeping (Pareto ``rank`` and ``crowding`` distance)."""
    x: Vector
    f: Vector
    cv: float = 0.0
    rank: int = 0
    crowding: float = 0.0


@dataclass
class MOProblem:
    """A multi-objective problem.

    ``evaluate(x)`` returns the raw objective vector in the user's own senses;
    ``constraints(x)`` (optional) returns violations in ``g(x) <= 0`` form.  With
    ``senses`` any objective may be maximised -- it is negated internally so the
    whole library can assume minimisation, and un-negated in reported results."""
    n_var: int
    n_obj: int
    evaluate: Callable[[Vector], Sequence[float]]
    bounds: Optional[List[Tuple[float, float]]] = None
    senses: Optional[List[str]] = None
    n_constr: int = 0
    constraints: Optional[Callable[[Vector], Sequence[float]]] = None
    name: str = "MOProblem"
    _signs: List[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.senses is None:
            self.senses = ["min"] * self.n_obj
        if len(self.senses) != self.n_obj:
            raise ValueError("senses must have one entry per objective.")
        for s in self.senses:
            if s not in ("min", "max"):
                raise ValueError("each sense must be 'min' or 'max'.")
        self._signs = [1.0 if s == "min" else -1.0 for s in self.senses]
        if self.bounds is not None and len(self.bounds) != self.n_var:
            raise ValueError("bounds must have one (lo, hi) pair per variable.")

    def evaluate_min(self, x: Vector) -> Vector:
        """Objective vector normalised to minimisation (maximised terms negated)."""
        raw = list(self.evaluate(x))
        if len(raw) != self.n_obj:
            raise ValueError("evaluate must return n_obj objectives.")
        if any(r != r for r in raw):        # NaN would be ranked best in the sort
            raise ValueError("evaluate returned a NaN objective; "
                             "check the objective function (a +/-inf penalty is fine).")
        return [s * r for s, r in zip(self._signs, raw)]

    def to_original(self, f_min: Sequence[float]) -> Vector:
        """Undo :meth:`evaluate_min` -- recover the user's own objective senses."""
        return [s * fm for s, fm in zip(self._signs, f_min)]

    def violation(self, x: Vector) -> float:
        """Total constraint violation ``sum max(0, g_j(x))`` (0 means feasible)."""
        if self.constraints is None:
            return 0.0
        return math.fsum(max(0.0, g) for g in self.constraints(x))

    def sample(self, rng: random.Random) -> Vector:
        """A single uniform random decision vector (requires ``bounds``)."""
        if self.bounds is None:
            raise ValueError("sample requires real-coded bounds.")
        return [rng.uniform(lo, hi) for lo, hi in self.bounds]

    def make_individual(self, x: Vector) -> Individual:
        return Individual(list(x), self.evaluate_min(x), self.violation(x))


@dataclass
class MOResult:
    """Outcome of a multi-objective run: the non-dominated decision vectors ``X``
    and their objective vectors ``F`` (in the problem's *original* senses), the
    number of evaluations / generations, an optional indicator ``history`` and a
    ``meta`` dict.  ``F_min`` holds the minimisation-normalised objectives."""
    X: List[Vector]
    F: List[Vector]
    n_eval: int
    n_gen: int
    history: List[float] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    F_min: List[Vector] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.X)


# ===========================================================================
# 7. OPERATOR PLUMBING FOR THE ALGORITHMS
# ===========================================================================
def _make_operators(problem: MOProblem, sampling, crossover, mutation,
                    sbx_eta, sbx_prob, mut_eta, mut_prob):
    """Assemble (sample, crossover, mutate) callables, defaulting to real-coded
    SBX + polynomial mutation when the caller does not override them."""
    if sampling is None:
        def sampling(prob, n, rng):
            return random_sampling(prob.bounds, n, rng)
    if crossover is None:
        def crossover(prob, a, b, rng):
            return sbx_crossover(a, b, prob.bounds, eta=sbx_eta, prob=sbx_prob,
                                 rng=rng)
    if mutation is None:
        def mutation(prob, x, rng):
            return polynomial_mutation(x, prob.bounds, eta=mut_eta, prob=mut_prob,
                                       rng=rng)
    return sampling, crossover, mutation


def _assign_ranks_and_crowding(pop: List[Individual]) -> List[List[int]]:
    """Fill in ``rank`` and ``crowding`` for a population, returning the fronts."""
    F = [ind.f for ind in pop]
    cv = [ind.cv for ind in pop] if any(ind.cv > 0 for ind in pop) else None
    fronts = non_dominated_sort(F, cv)
    for r, front in enumerate(fronts):
        cd = crowding_distance(F, front)
        for i in front:
            pop[i].rank = r
            pop[i].crowding = cd[i]
    return fronts


def _result_from_population(problem: MOProblem, pop: List[Individual],
                            n_eval: int, n_gen: int, history: List[float],
                            meta: Dict[str, Any]) -> MOResult:
    F = [ind.f for ind in pop]
    cv = [ind.cv for ind in pop] if any(ind.cv > 0 for ind in pop) else None
    front = find_pareto_front(F, cv)
    X = [pop[i].x for i in front]
    F_min = [pop[i].f for i in front]
    F_orig = [problem.to_original(fm) for fm in F_min]
    return MOResult(X, F_orig, n_eval, n_gen, history, meta, F_min)


# ===========================================================================
# 8. ALGORITHMS
# ===========================================================================
def nsga2(problem: MOProblem, *, pop_size: int = 100, n_gen: int = 200,
          sbx_eta: float = 15.0, sbx_prob: float = 0.9, mut_eta: float = 20.0,
          mut_prob: Optional[float] = None, sampling=None, crossover=None,
          mutation=None, seed: Any = 0, ref_point: Optional[Sequence[float]] = None,
          track_hv: bool = False,
          callback: Optional[Callable[[int, List[Individual]], None]] = None
          ) -> MOResult:
    """**NSGA-II** (Deb et al. 2002): elitist non-dominated sorting with crowding
    distance -- the default work-horse for 2-3 objectives.  Binary-tournament
    parent selection, (SBX + polynomial mutation) reproduction, and environmental
    selection by (rank, crowding) on the combined parent+offspring pool.

    Handles constraints via constraint-domination automatically when the problem
    defines them.  Set ``track_hv`` (with a ``ref_point``) to record the
    hypervolume each generation in ``result.history``."""
    if pop_size < 2:
        raise ValueError("pop_size must be >= 2.")
    rng = _as_rng(seed)
    sample, cross, mutate = _make_operators(problem, sampling, crossover, mutation,
                                            sbx_eta, sbx_prob, mut_eta, mut_prob)
    pop = [problem.make_individual(x) for x in sample(problem, pop_size, rng)]
    n_eval = len(pop)
    _assign_ranks_and_crowding(pop)
    history: List[float] = []

    for gen in range(n_gen):
        offspring: List[Individual] = []
        while len(offspring) < pop_size:
            p1 = binary_tournament(pop, rng)
            p2 = binary_tournament(pop, rng)
            c1, c2 = cross(problem, p1.x, p2.x, rng)
            for child in (c1, c2):
                child = mutate(problem, child, rng)
                offspring.append(problem.make_individual(child))
                n_eval += 1
                if len(offspring) >= pop_size:
                    break
        combined = pop + offspring
        pop = _environmental_selection_nsga2(combined, pop_size)
        if track_hv and ref_point is not None:
            history.append(hypervolume([ind.f for ind in pop], ref_point))
        if callback is not None:
            callback(gen, pop)

    return _result_from_population(problem, pop, n_eval, n_gen, history,
                                   {"algorithm": "NSGA-II"})


def _environmental_selection_nsga2(pop: List[Individual],
                                   n: int) -> List[Individual]:
    fronts = _assign_ranks_and_crowding(pop)
    selected: List[Individual] = []
    for front in fronts:
        if len(selected) + len(front) <= n:
            selected.extend(pop[i] for i in front)
        else:
            remaining = n - len(selected)
            by_crowd = sorted(front, key=lambda i: pop[i].crowding, reverse=True)
            selected.extend(pop[i] for i in by_crowd[:remaining])
            break
    return selected


def nsga3(problem: MOProblem, *, pop_size: Optional[int] = None,
          n_partitions: int = 12, n_gen: int = 200, sbx_eta: float = 30.0,
          sbx_prob: float = 1.0, mut_eta: float = 20.0,
          mut_prob: Optional[float] = None, ref_dirs: Optional[List[Vector]] = None,
          sampling=None, crossover=None, mutation=None, seed: Any = 0) -> MOResult:
    """**NSGA-III** (Deb & Jain 2014): non-dominated sorting with *reference-point*
    niching instead of crowding distance -- built for **many objectives** (4+),
    where crowding distance loses its grip.  Adaptively normalises objectives to
    the ideal point and hyperplane intercepts, then associates and niches
    solutions to the Das-Dennis reference directions."""
    rng = _as_rng(seed)
    if ref_dirs is None:
        ref_dirs = das_dennis(problem.n_obj, n_partitions)
    if ref_dirs and len(ref_dirs[0]) != problem.n_obj:
        # a dimension mismatch would silently zip-truncate the association
        raise ValueError("ref_dirs dimension must equal the number of objectives.")
    if pop_size is None:
        pop_size = len(ref_dirs) + (len(ref_dirs) % 2)   # next even number
    if pop_size < 2:
        raise ValueError("pop_size must be >= 2.")
    sample, cross, mutate = _make_operators(problem, sampling, crossover, mutation,
                                            sbx_eta, sbx_prob, mut_eta, mut_prob)
    pop = [problem.make_individual(x) for x in sample(problem, pop_size, rng)]
    n_eval = len(pop)

    for _ in range(n_gen):
        offspring: List[Individual] = []
        while len(offspring) < pop_size:
            p1 = binary_tournament(pop, rng, key=lambda a, b:
                                   a if a.rank <= b.rank else b)
            p2 = binary_tournament(pop, rng, key=lambda a, b:
                                   a if a.rank <= b.rank else b)
            c1, c2 = cross(problem, p1.x, p2.x, rng)
            for child in (c1, c2):
                child = mutate(problem, child, rng)
                offspring.append(problem.make_individual(child))
                n_eval += 1
                if len(offspring) >= pop_size:
                    break
        combined = pop + offspring
        pop = _environmental_selection_nsga3(combined, pop_size, ref_dirs)

    return _result_from_population(problem, pop, n_eval, n_gen, [],
                                   {"algorithm": "NSGA-III",
                                    "n_ref_dirs": len(ref_dirs)})


def _environmental_selection_nsga3(pop: List[Individual], n: int,
                                   ref_dirs: List[Vector]) -> List[Individual]:
    F = [ind.f for ind in pop]
    cv = [ind.cv for ind in pop] if any(ind.cv > 0 for ind in pop) else None
    fronts = non_dominated_sort(F, cv)

    selected_idx: List[int] = []
    last_front: List[int] = []
    for front in fronts:
        if len(selected_idx) + len(front) <= n:
            selected_idx.extend(front)
        else:
            last_front = front
            break
    if not last_front:
        return [pop[i] for i in selected_idx]

    candidate_idx = selected_idx + last_front
    z_ideal = ideal_point([F[i] for i in candidate_idx])
    extremes = extreme_points([F[i] for i in candidate_idx], z_ideal)
    intercepts = _hyperplane_intercepts(extremes, z_ideal)
    m = len(z_ideal)
    norm = {}
    for i in candidate_idx:
        norm[i] = [(F[i][k] - z_ideal[k]) / intercepts[k] if intercepts[k] > _EPS
                   else (F[i][k] - z_ideal[k]) for k in range(m)]

    assoc, dist = _associate(norm, candidate_idx, ref_dirs)
    niche_count: Dict[int, int] = {j: 0 for j in range(len(ref_dirs))}
    for i in selected_idx:
        niche_count[assoc[i]] += 1

    remaining = n - len(selected_idx)
    last_set = set(last_front)
    chosen = _niching(remaining, niche_count, assoc, dist, last_set, ref_dirs)
    return [pop[i] for i in selected_idx + chosen]


def _hyperplane_intercepts(extremes: List[Vector],
                           ideal: Sequence[float]) -> Vector:
    """Intercepts of the hyperplane through the (translated) extreme points; falls
    back to the componentwise max when the system is degenerate."""
    m = len(ideal)
    trans = [[e[k] - ideal[k] for k in range(m)] for e in extremes]
    try:
        b = [1.0] * m
        coeff = _gaussian_solve([row[:] for row in trans], b)
        intercepts = [1.0 / c if abs(c) > _EPS else _INF for c in coeff]
        if any(ic <= _EPS or ic == _INF for ic in intercepts):
            raise ValueError
        return intercepts
    except (ValueError, ZeroDivisionError):
        worst = [max(e[k] - ideal[k] for e in extremes) for k in range(m)]
        return [w if w > _EPS else 1.0 for w in worst]


def _gaussian_solve(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve the small dense system ``A x = b`` by Gaussian elimination with
    partial pivoting (used only for the m x m NSGA-III intercept system)."""
    n = len(A)
    M = [A[i] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < _EPS:
            raise ValueError("singular")
        M[col], M[pivot] = M[pivot], M[col]
        piv = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= piv
        for r in range(n):
            if r != col and abs(M[r][col]) > 0.0:
                factor = M[r][col]
                for j in range(col, n + 1):
                    M[r][j] -= factor * M[col][j]
    return [M[i][n] for i in range(n)]


def _associate(norm: Dict[int, Vector], indices: Sequence[int],
               ref_dirs: List[Vector]) -> Tuple[Dict[int, int], Dict[int, float]]:
    """Associate each solution with the nearest reference direction (smallest
    perpendicular distance to the line through the origin)."""
    assoc: Dict[int, int] = {}
    dist: Dict[int, float] = {}
    ref_norm = [math.sqrt(math.fsum(w * w for w in d)) or _EPS for d in ref_dirs]
    for i in indices:
        s = norm[i]
        best_j, best_d = 0, _INF
        for j, w in enumerate(ref_dirs):
            dot = math.fsum(si * wi for si, wi in zip(s, w))
            proj = dot / ref_norm[j]
            perp = math.sqrt(max(0.0, math.fsum(si * si for si in s) - proj * proj))
            if perp < best_d:
                best_d, best_j = perp, j
        assoc[i] = best_j
        dist[i] = best_d
    return assoc, dist


def _niching(k: int, niche_count: Dict[int, int], assoc: Dict[int, int],
             dist: Dict[int, float], last_set: set,
             ref_dirs: List[Vector]) -> List[int]:
    """NSGA-III niche-preserving fill: repeatedly pick the least-crowded reference
    direction and take from its associated last-front members."""
    chosen: List[int] = []
    available: Dict[int, List[int]] = {j: [] for j in range(len(ref_dirs))}
    for i in last_set:
        available[assoc[i]].append(i)
    while len(chosen) < k:
        candidate_niches = [j for j in range(len(ref_dirs)) if available[j]]
        if not candidate_niches:
            break
        min_count = min(niche_count[j] for j in candidate_niches)
        tied = [j for j in candidate_niches if niche_count[j] == min_count]
        j = tied[0]
        members = available[j]
        if niche_count[j] == 0:
            pick = min(members, key=lambda i: dist[i])
        else:
            pick = members[0]
        chosen.append(pick)
        members.remove(pick)
        niche_count[j] += 1
    return chosen


def moead(problem: MOProblem, *, n_partitions: int = 12, n_gen: int = 250,
          neighbourhood: int = 20, aggregation: str = "tchebycheff",
          theta: float = 5.0, cr: float = 1.0, de_f: float = 0.5,
          mut_eta: float = 20.0, mut_prob: Optional[float] = None,
          ref_dirs: Optional[List[Vector]] = None, delta: float = 0.9,
          n_replace: int = 2, seed: Any = 0) -> MOResult:
    """**MOEA/D** (Zhang & Li 2007): decompose the problem into scalar
    subproblems along a set of weight vectors and solve them *cooperatively*,
    each subproblem improved using solutions from its ``neighbourhood``.  Uses the
    DE ``rand/1`` operator plus polynomial mutation.

    ``aggregation`` selects the scalarisation: ``'tchebycheff'`` (default),
    ``'pbi'`` or ``'weighted_sum'``."""
    rng = _as_rng(seed)
    if ref_dirs is None:
        ref_dirs = das_dennis(problem.n_obj, n_partitions)
    if ref_dirs and len(ref_dirs[0]) != problem.n_obj:
        # a dimension mismatch would silently zip-truncate the aggregation
        raise ValueError("ref_dirs dimension must equal the number of objectives.")
    N = len(ref_dirs)
    T = min(neighbourhood, N)
    agg = _aggregation_fn(aggregation, theta)

    neighbours = _weight_neighbours(ref_dirs, T)
    pop = [problem.make_individual(problem.sample(rng)) for _ in range(N)]
    n_eval = N
    z = ideal_point([ind.f for ind in pop])

    for _ in range(n_gen):
        for i in range(N):
            pool = neighbours[i] if rng.random() < delta else list(range(N))
            r = [pool[rng.randrange(len(pool))] for _ in range(3)]
            trial = de_operator(pop[i].x, pop[r[0]].x, pop[r[1]].x, pop[r[2]].x,
                                problem.bounds, F=de_f, cr=cr, rng=rng)
            trial = polynomial_mutation(trial, problem.bounds, eta=mut_eta,
                                        prob=mut_prob, rng=rng)
            child = problem.make_individual(trial)
            n_eval += 1
            for k in range(problem.n_obj):
                z[k] = min(z[k], child.f[k])
            # update at most n_replace neighbours the child improves
            perm = pool[:]
            rng.shuffle(perm)
            replaced = 0
            for j in perm:
                if replaced >= n_replace:
                    break
                if _moead_improves(child, pop[j], ref_dirs[j], z, agg):
                    pop[j] = child
                    replaced += 1

    return _result_from_population(problem, pop, n_eval, n_gen, [],
                                   {"algorithm": "MOEA/D",
                                    "aggregation": aggregation})


def _moead_improves(child: Individual, incumbent: Individual, w: Vector,
                    z: Sequence[float], agg) -> bool:
    """Constrained MOEA/D replacement rule: a feasible child replaces an infeasible
    incumbent; between two feasibles the smaller aggregated value wins; between two
    infeasibles the smaller constraint violation wins.  Without this, MOEA/D would
    optimise the objectives and silently ignore feasibility."""
    if child.cv <= 0.0 and incumbent.cv <= 0.0:
        return agg(child.f, w, z) <= agg(incumbent.f, w, z)
    if child.cv <= 0.0:
        return True
    if incumbent.cv <= 0.0:
        return False
    return child.cv <= incumbent.cv


def _aggregation_fn(name: str, theta: float):
    if name == "tchebycheff":
        return lambda f, w, z: tchebycheff(f, w, z)
    if name == "weighted_sum":
        return lambda f, w, z: weighted_sum(f, w)
    if name == "pbi":
        return lambda f, w, z: pbi(f, w, z, theta)
    raise ValueError("aggregation must be 'tchebycheff', 'pbi' or 'weighted_sum'.")


def _weight_neighbours(ref_dirs: List[Vector], T: int) -> List[List[int]]:
    N = len(ref_dirs)
    neighbours = []
    for i in range(N):
        dists = sorted(range(N), key=lambda j: _dist(ref_dirs[i], ref_dirs[j]))
        neighbours.append(dists[:T])
    return neighbours


def spea2(problem: MOProblem, *, pop_size: int = 100,
          archive_size: Optional[int] = None, n_gen: int = 200,
          sbx_eta: float = 15.0, sbx_prob: float = 0.9, mut_eta: float = 20.0,
          mut_prob: Optional[float] = None, sampling=None, crossover=None,
          mutation=None, seed: Any = 0) -> MOResult:
    """**SPEA2** (Zitzler et al. 2001): strength-Pareto evolutionary algorithm.
    Fitness combines a *strength* term (how many others a solution dominates /
    is dominated by) with a k-th-nearest-neighbour *density* term, and the archive
    is trimmed by a distance-based truncation that preserves boundary points.

    ``archive_size`` defaults to ``pop_size`` (the usual convention); making the
    archive much larger than the offspring pool weakens convergence pressure."""
    if pop_size < 2:
        raise ValueError("pop_size must be >= 2.")
    if archive_size is None:
        archive_size = pop_size
    rng = _as_rng(seed)
    sample, cross, mutate = _make_operators(problem, sampling, crossover, mutation,
                                            sbx_eta, sbx_prob, mut_eta, mut_prob)
    pop = [problem.make_individual(x) for x in sample(problem, pop_size, rng)]
    archive: List[Individual] = []
    n_eval = len(pop)

    for _ in range(n_gen):
        union = pop + archive
        fitness = _spea2_fitness(union)
        archive = _spea2_environmental(union, fitness, archive_size)
        mating = [_spea2_tournament(archive, fitness, union, rng)
                  for _ in range(pop_size)]
        offspring: List[Individual] = []
        while len(offspring) < pop_size:
            p1 = mating[rng.randrange(len(mating))]
            p2 = mating[rng.randrange(len(mating))]
            c1, c2 = cross(problem, p1.x, p2.x, rng)
            for child in (c1, c2):
                child = mutate(problem, child, rng)
                offspring.append(problem.make_individual(child))
                n_eval += 1
                if len(offspring) >= pop_size:
                    break
        pop = offspring

    return _result_from_population(problem, archive or pop, n_eval, n_gen, [],
                                   {"algorithm": "SPEA2"})


def _spea2_fitness(union: List[Individual]) -> List[float]:
    n = len(union)
    F = [ind.f for ind in union]
    cv = [ind.cv for ind in union]
    strength = [0] * n
    dominated_by: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and constraint_dominates(F[i], cv[i], F[j], cv[j]):
                strength[i] += 1
                dominated_by[j].append(i)
    raw = [math.fsum(strength[i] for i in dominated_by[j]) for j in range(n)]
    k = int(math.sqrt(n)) or 1
    density = []
    for j in range(n):
        dists = sorted(_dist(F[j], F[l]) for l in range(n) if l != j)
        sigma = dists[k - 1] if len(dists) >= k else (dists[-1] if dists else 0.0)
        density.append(1.0 / (sigma + 2.0))
    return [raw[j] + density[j] for j in range(n)]


def _spea2_environmental(union: List[Individual], fitness: List[float],
                         size: int) -> List[Individual]:
    idx = [i for i in range(len(union)) if fitness[i] < 1.0]     # non-dominated
    if len(idx) < size:
        rest = sorted((i for i in range(len(union)) if i not in set(idx)),
                      key=lambda i: fitness[i])
        idx += rest[:size - len(idx)]
    elif len(idx) > size:
        idx = _spea2_truncate(union, idx, size)
    return [union[i] for i in idx]


def _spea2_truncate(union: List[Individual], idx: List[int],
                    size: int) -> List[int]:
    """Distance-based archive truncation: iteratively drop the individual whose
    sorted vector of distances to the others is lexicographically smallest (the
    most crowded), preserving the boundary.  The pairwise distance matrix is
    computed once so each removal is a table lookup, not an O(n d) recompute."""
    F = [union[i].f for i in idx]
    n = len(idx)
    D = [[0.0] * n for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            d = _dist(F[a], F[b])
            D[a][b] = D[b][a] = d
    keep = list(range(n))
    while len(keep) > size:
        best = None
        best_key: Optional[List[float]] = None
        for a in keep:
            key = sorted(D[a][b] for b in keep if b != a)
            if best_key is None or key < best_key:
                best_key = key
                best = a
        keep.remove(best)
    return [idx[i] for i in keep]


def _spea2_tournament(archive: List[Individual], fitness: List[float],
                      union: List[Individual], rng: random.Random) -> Individual:
    if not archive:
        return union[rng.randrange(len(union))]
    fit_map = {id(union[i]): fitness[i] for i in range(len(union))}
    a = archive[rng.randrange(len(archive))]
    b = archive[rng.randrange(len(archive))]
    fa = fit_map.get(id(a), _INF)
    fb = fit_map.get(id(b), _INF)
    return a if fa <= fb else b


def sms_emoa(problem: MOProblem, *, pop_size: int = 100, n_gen: int = 200,
             ref_point: Optional[Sequence[float]] = None, sbx_eta: float = 15.0,
             sbx_prob: float = 0.9, mut_eta: float = 20.0,
             mut_prob: Optional[float] = None, sampling=None, crossover=None,
             mutation=None, seed: Any = 0) -> MOResult:
    """**SMS-EMOA** (Beume et al. 2007): a steady-state (mu+1) algorithm whose
    environmental selection removes the individual contributing the *least
    hypervolume* to the worst front.  Directly optimises the hypervolume, giving
    excellent, well-spread fronts for a modest number of objectives.

    ``n_gen`` here counts single-offspring iterations x ``pop_size`` (one
    generation = ``pop_size`` steady-state steps)."""
    if pop_size < 2:
        raise ValueError("pop_size must be >= 2.")
    rng = _as_rng(seed)
    sample, cross, mutate = _make_operators(problem, sampling, crossover, mutation,
                                            sbx_eta, sbx_prob, mut_eta, mut_prob)
    pop = [problem.make_individual(x) for x in sample(problem, pop_size, rng)]
    n_eval = len(pop)

    for _ in range(n_gen * pop_size):
        p1 = pop[rng.randrange(pop_size)]
        p2 = pop[rng.randrange(pop_size)]
        c1, _ = cross(problem, p1.x, p2.x, rng)
        child = problem.make_individual(mutate(problem, c1, rng))
        n_eval += 1
        pop.append(child)
        pop = _sms_reduce(pop, ref_point)

    return _result_from_population(problem, pop, n_eval, n_gen, [],
                                   {"algorithm": "SMS-EMOA"})


def _sms_reduce(pop: List[Individual],
                ref_point: Optional[Sequence[float]]) -> List[Individual]:
    F = [ind.f for ind in pop]
    cv = [ind.cv for ind in pop] if any(ind.cv > 0 for ind in pop) else None
    fronts = non_dominated_sort(F, cv)
    worst = fronts[-1]
    if len(worst) == 1:
        drop = worst[0]
    else:
        worst_F = [F[i] for i in worst]
        ref = list(ref_point) if ref_point is not None else \
            [max(f[k] for f in worst_F) + 1.0 for k in range(len(worst_F[0]))]
        contribs = hypervolume_contributions(worst_F, ref)
        drop = worst[min(range(len(worst)), key=lambda t: contribs[t])]
    return [ind for t, ind in enumerate(pop) if t != drop]


def ibea(problem: MOProblem, *, pop_size: int = 100, n_gen: int = 200,
         kappa: float = 0.05, sbx_eta: float = 15.0, sbx_prob: float = 0.9,
         mut_eta: float = 20.0, mut_prob: Optional[float] = None,
         sampling=None, crossover=None, mutation=None, seed: Any = 0) -> MOResult:
    """**IBEA** (Zitzler & Kunzli 2004): indicator-based EA that dispenses with
    diversity heuristics entirely and lets a binary quality indicator (here the
    additive epsilon indicator) define fitness,
    ``F(i) = sum_{j != i} -exp(-I(j, i) / (kappa * c))``.  Environmental selection
    removes the worst individual and incrementally updates the remaining fitness."""
    if pop_size < 2:
        raise ValueError("pop_size must be >= 2.")
    rng = _as_rng(seed)
    sample, cross, mutate = _make_operators(problem, sampling, crossover, mutation,
                                            sbx_eta, sbx_prob, mut_eta, mut_prob)
    pop = [problem.make_individual(x) for x in sample(problem, pop_size, rng)]
    n_eval = len(pop)

    for _ in range(n_gen):
        offspring: List[Individual] = []
        pop = _ibea_select(pop, pop_size, kappa)
        while len(offspring) < pop_size:
            p1 = pop[rng.randrange(len(pop))]
            p2 = pop[rng.randrange(len(pop))]
            c1, c2 = cross(problem, p1.x, p2.x, rng)
            for child in (c1, c2):
                child = mutate(problem, child, rng)
                offspring.append(problem.make_individual(child))
                n_eval += 1
                if len(offspring) >= pop_size:
                    break
        pop = pop + offspring

    pop = _ibea_select(pop, pop_size, kappa)
    return _result_from_population(problem, pop, n_eval, n_gen, [],
                                   {"algorithm": "IBEA"})


def _ibea_select(pop: List[Individual], n: int, kappa: float) -> List[Individual]:
    """Fitness-assign by pairwise additive-epsilon indicator and iteratively drop
    the worst individual, updating the remaining fitnesses."""
    if len(pop) <= n:
        return pop
    m = len(pop)
    F = [ind.f for ind in pop]
    # scale objectives to [0,1] for a well-behaved indicator
    ideal = [min(f[k] for f in F) for k in range(len(F[0]))]
    nad = [max(f[k] for f in F) for k in range(len(F[0]))]
    norm = [[(f[k] - ideal[k]) / (nad[k] - ideal[k]) if nad[k] - ideal[k] > _EPS
             else 0.0 for k in range(len(f))] for f in F]
    # I(a, b) = max_k (norm_a[k] - norm_b[k]) (additive epsilon; a onto b)
    I = [[max(norm[a][k] - norm[b][k] for k in range(len(norm[a])))
          for b in range(m)] for a in range(m)]
    c = max(abs(I[a][b]) for a in range(m) for b in range(m)) or 1.0
    fitness = [math.fsum(-math.exp(-I[j][i] / (kappa * c))
                         for j in range(m) if j != i) for i in range(m)]
    alive = list(range(m))
    while len(alive) > n:
        # feasibility first: discard the worst constraint violator before any
        # feasible individual, otherwise IBEA's objective-only indicator would keep
        # infeasible solutions and could return an all-infeasible front.
        infeasible = [i for i in alive if pop[i].cv > 0.0]
        if infeasible:
            worst = max(infeasible, key=lambda i: pop[i].cv)
        else:
            worst = min(alive, key=lambda i: fitness[i])
        alive.remove(worst)
        for i in alive:
            fitness[i] -= -math.exp(-I[worst][i] / (kappa * c))
    return [pop[i] for i in alive]


# ===========================================================================
# 9. DECISION MAKING (MCDM)
# ===========================================================================
def topsis(F: Sequence[Sequence[float]], weights: Optional[Sequence[float]] = None,
           senses: Optional[Sequence[int]] = None) -> Tuple[int, List[float]]:
    """TOPSIS ranking (Hwang & Yoon): score each solution by its relative
    closeness to the ideal and distance from the anti-ideal after vector
    normalisation.  Returns ``(best_index, scores)`` where higher is better;
    minimisation is assumed unless ``senses`` (+1 min / -1 max) is given."""
    F = [list(f) for f in F]
    if not F:
        raise ValueError("topsis needs a non-empty set of solutions.")
    n, m = len(F), len(F[0])
    if weights is None:
        weights = [1.0 / m] * m
    if senses is None:
        senses = [1] * m
    denom = [math.sqrt(math.fsum(F[i][k] ** 2 for i in range(n))) or _EPS
             for k in range(m)]
    V = [[weights[k] * F[i][k] / denom[k] for k in range(m)] for i in range(n)]
    best = [min(V[i][k] for i in range(n)) if senses[k] > 0
            else max(V[i][k] for i in range(n)) for k in range(m)]
    worst = [max(V[i][k] for i in range(n)) if senses[k] > 0
             else min(V[i][k] for i in range(n)) for k in range(m)]
    scores = []
    for i in range(n):
        d_best = math.sqrt(math.fsum((V[i][k] - best[k]) ** 2 for k in range(m)))
        d_worst = math.sqrt(math.fsum((V[i][k] - worst[k]) ** 2 for k in range(m)))
        scores.append(d_worst / (d_best + d_worst) if (d_best + d_worst) > _EPS
                      else 0.0)
    best_idx = max(range(n), key=lambda i: scores[i])
    return best_idx, scores


def compromise_programming(F: Sequence[Sequence[float]],
                           weights: Optional[Sequence[float]] = None,
                           p: float = 2.0,
                           ideal: Optional[Sequence[float]] = None) -> int:
    """Compromise programming: pick the solution minimising the weighted ``L_p``
    distance to the ideal point (``p=1`` Manhattan, ``p=2`` Euclidean,
    ``p=inf`` Tchebycheff).  Returns the chosen index (minimisation)."""
    F = [list(f) for f in F]
    if not F:
        raise ValueError("compromise_programming needs a non-empty set.")
    if not p >= 1.0:
        # p < 1 is not a valid norm order; p = 0 hits 1/p and p < 0 hits 0**-|p|
        raise ValueError("p must be >= 1 (or inf).")
    m = len(F[0])
    if weights is None:
        weights = [1.0 / m] * m
    if ideal is None:
        ideal = ideal_point(F)
    nad = nadir_point(F)
    spans = [nad[k] - ideal[k] if nad[k] - ideal[k] > _EPS else 1.0
             for k in range(m)]

    def lp(f):
        terms = [weights[k] * abs(f[k] - ideal[k]) / spans[k] for k in range(m)]
        if math.isinf(p):
            return max(terms)
        return math.fsum(t ** p for t in terms) ** (1.0 / p)

    return min(range(len(F)), key=lambda i: lp(F[i]))


def pseudo_weights(F: Sequence[Sequence[float]]) -> List[Vector]:
    """Pseudo-weight of each solution: the normalised distance from the nadir in
    every objective, ``(nadir_k - f_k)/(nadir_k - ideal_k)`` re-normalised to sum
    to 1.  Tells you which trade-off (which effective weight vector) a point
    represents."""
    F = [list(f) for f in F]
    if not F:
        return []
    m = len(F[0])
    z = ideal_point(F)
    nad = nadir_point(F)
    spans = [nad[k] - z[k] if nad[k] - z[k] > _EPS else 1.0 for k in range(m)]
    out = []
    for f in F:
        raw = [(nad[k] - f[k]) / spans[k] for k in range(m)]
        s = math.fsum(raw)
        if abs(s) <= _EPS:
            # degenerate (single point / all objectives at the nadir): the trade-off
            # is undefined, so report an equal (uniform) weighting that sums to 1.
            out.append([1.0 / m] * m)
        else:
            out.append([r / s for r in raw])
    return out


def weighted_selection(F: Sequence[Sequence[float]],
                       weights: Sequence[float]) -> int:
    """Index of the solution with the smallest weighted-sum objective (a fast
    a-posteriori pick when the decision maker can state linear weights)."""
    if not F:
        raise ValueError("weighted_selection needs a non-empty set.")
    return min(range(len(F)), key=lambda i: weighted_sum(F[i], weights))


def knee_point(F: Sequence[Sequence[float]]) -> int:
    """Find the *knee* of a 2-D front -- the point of maximum trade-off, i.e.
    farthest below the line joining the two extreme solutions.  Beyond a knee you
    sacrifice a lot in one objective for little gain in the other."""
    if not F:
        raise ValueError("knee_point needs a non-empty front.")
    if len(F[0]) != 2:
        # the chord construction only sees objectives 0 and 1; for more objectives
        # it would silently ignore the rest.
        raise ValueError("knee_point is defined for 2 objectives; "
                         "use high_tradeoff_points for more.")
    pts = sorted(range(len(F)), key=lambda i: F[i][0])
    a, b = F[pts[0]], F[pts[-1]]
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    length = math.hypot(bx - ax, by - ay) or _EPS
    best_idx, best_d = pts[0], -_INF
    for i in pts:
        # perpendicular distance below the chord (positive = knee side)
        d = ((bx - ax) * (ay - F[i][1]) - (ax - F[i][0]) * (by - ay)) / length
        if d > best_d:
            best_d, best_idx = d, i
    return best_idx


def high_tradeoff_points(F: Sequence[Sequence[float]],
                         epsilon: float = 1e-3) -> List[int]:
    """Rachmawati-Srinivasan high-trade-off points: solutions where moving to a
    neighbour forces a sharp worsening in some objective for a small gain in
    another (a generalisation of the knee to any dimension).  Returns their
    indices, most pronounced first."""
    n = len(F)
    if n < 3:
        return list(range(n))
    z = ideal_point(F)
    nad = nadir_point(F)
    m = len(F[0])
    spans = [nad[k] - z[k] if nad[k] - z[k] > _EPS else 1.0 for k in range(m)]
    norm = [[(F[i][k] - z[k]) / spans[k] for k in range(m)] for i in range(n)]

    scores = []
    for i in range(n):
        worst_ratio = _INF
        for j in range(n):
            if i == j:
                continue
            gain = math.fsum(max(norm[j][k] - norm[i][k], 0.0) for k in range(m))
            loss = math.fsum(max(norm[i][k] - norm[j][k], 0.0) for k in range(m))
            if gain > epsilon:
                worst_ratio = min(worst_ratio, loss / gain)
        scores.append((worst_ratio, i))
    scores = [(s, i) for s, i in scores if s != _INF]
    scores.sort(reverse=True)
    return [i for _, i in scores]


class ParetoArchive:
    """An unbounded non-dominated archive with optional additive
    **epsilon-dominance** (Laumanns et al. 2002).  Feeding a stream of solutions
    keeps only the mutually non-dominated ones; a positive ``epsilon`` coarsens
    the front onto a grid, bounding the archive size while guaranteeing
    convergence and spread."""

    def __init__(self, epsilon: float = 0.0):
        if epsilon < 0.0:
            raise ValueError("epsilon must be >= 0.")
        self.epsilon = float(epsilon)
        self.items: List[Tuple[Vector, Vector]] = []      # (x, f) minimisation

    def add(self, x: Sequence[float], f: Sequence[float]) -> bool:
        """Offer ``(x, f)`` to the archive; return ``True`` if it was accepted."""
        x, f = list(x), list(f)
        if self.epsilon > 0.0:
            box = tuple(math.floor(fi / self.epsilon) for fi in f)
            kept: List[Tuple[Vector, Vector]] = []
            corner = [b * self.epsilon for b in box]
            for (xi, fi) in self.items:
                bi = tuple(math.floor(v / self.epsilon) for v in fi)
                if bi == box:
                    if weakly_dominates(fi, f):
                        return False                      # existing dominates new
                    if not weakly_dominates(f, fi):
                        # mutually non-dominated inside the box: keep the point
                        # closest to the box's lower corner (lexicographic tiebreak)
                        # so the representative is INDEPENDENT of insertion order.
                        if (_dist(fi, corner), fi) <= (_dist(f, corner), f):
                            return False
                    continue                              # else new replaces box
                if dominates(fi, f):
                    return False
                if not dominates(f, fi):
                    kept.append((xi, fi))
            kept.append((x, f))
            self.items = kept
            return True
        for (xi, fi) in self.items:
            if weakly_dominates(fi, f) and fi != f:
                return False
        self.items = [(xi, fi) for (xi, fi) in self.items if not dominates(f, fi)]
        if any(fi == f for (_, fi) in self.items):
            return False
        self.items.append((x, f))
        return True

    @property
    def F(self) -> List[Vector]:
        return [f for (_, f) in self.items]

    @property
    def X(self) -> List[Vector]:
        return [x for (x, _) in self.items]

    def __len__(self) -> int:
        return len(self.items)


# ===========================================================================
# 10. BENCHMARK PROBLEMS
# ===========================================================================
def _zdt(f1: Callable[[Vector], float], g_and_h, n_var: int, name: str) -> MOProblem:
    def evaluate(x):
        a = f1(x)
        b = g_and_h(x, a)
        return [a, b]
    return MOProblem(n_var, 2, evaluate, bounds=[(0.0, 1.0)] * n_var, name=name)


def zdt1(n_var: int = 30) -> MOProblem:
    """ZDT1: convex Pareto front ``f2 = 1 - sqrt(f1)``.  30 variables in [0,1]."""
    def evaluate(x):
        f1 = x[0]
        g = 1.0 + 9.0 * math.fsum(x[1:]) / (len(x) - 1)
        f2 = g * (1.0 - math.sqrt(f1 / g))
        return [f1, f2]
    return MOProblem(n_var, 2, evaluate, bounds=[(0.0, 1.0)] * n_var, name="ZDT1")


def zdt2(n_var: int = 30) -> MOProblem:
    """ZDT2: non-convex Pareto front ``f2 = 1 - f1^2``."""
    def evaluate(x):
        f1 = x[0]
        g = 1.0 + 9.0 * math.fsum(x[1:]) / (len(x) - 1)
        f2 = g * (1.0 - (f1 / g) ** 2)
        return [f1, f2]
    return MOProblem(n_var, 2, evaluate, bounds=[(0.0, 1.0)] * n_var, name="ZDT2")


def zdt3(n_var: int = 30) -> MOProblem:
    """ZDT3: disconnected Pareto front (five discontinuous segments)."""
    def evaluate(x):
        f1 = x[0]
        g = 1.0 + 9.0 * math.fsum(x[1:]) / (len(x) - 1)
        r = f1 / g
        f2 = g * (1.0 - math.sqrt(r) - r * math.sin(10.0 * math.pi * f1))
        return [f1, f2]
    return MOProblem(n_var, 2, evaluate, bounds=[(0.0, 1.0)] * n_var, name="ZDT3")


def dtlz1(n_var: int = 7, n_obj: int = 3) -> MOProblem:
    """DTLZ1: a scalable linear front on the simplex ``sum f_i = 0.5`` with a
    multimodal ``g`` (many local fronts)."""
    k = n_var - n_obj + 1

    def evaluate(x):
        xm = x[n_obj - 1:]
        g = 100.0 * (k + math.fsum((xi - 0.5) ** 2
                     - math.cos(20.0 * math.pi * (xi - 0.5)) for xi in xm))
        f = []
        for i in range(n_obj):
            val = 0.5 * (1.0 + g)
            for j in range(n_obj - 1 - i):
                val *= x[j]
            if i > 0:
                val *= (1.0 - x[n_obj - 1 - i])
            f.append(val)
        return f
    return MOProblem(n_var, n_obj, evaluate, bounds=[(0.0, 1.0)] * n_var,
                     name="DTLZ1")


def dtlz2(n_var: int = 12, n_obj: int = 3) -> MOProblem:
    """DTLZ2: a scalable spherical front ``sum f_i^2 = 1`` -- the standard
    many-objective convergence/spread test."""
    k = n_var - n_obj + 1

    def evaluate(x):
        xm = x[n_obj - 1:]
        g = math.fsum((xi - 0.5) ** 2 for xi in xm)
        f = []
        for i in range(n_obj):
            val = (1.0 + g)
            for j in range(n_obj - 1 - i):
                val *= math.cos(x[j] * math.pi / 2.0)
            if i > 0:
                val *= math.sin(x[n_obj - 1 - i] * math.pi / 2.0)
            f.append(val)
        return f
    return MOProblem(n_var, n_obj, evaluate, bounds=[(0.0, 1.0)] * n_var,
                     name="DTLZ2")


def schaffer() -> MOProblem:
    """Schaffer N.1: the classic 1-variable bi-objective problem ``[x^2, (x-2)^2]``
    with a convex front."""
    def evaluate(x):
        return [x[0] ** 2, (x[0] - 2.0) ** 2]
    return MOProblem(1, 2, evaluate, bounds=[(-5.0, 5.0)], name="Schaffer")


def kursawe(n_var: int = 3) -> MOProblem:
    """Kursawe: a hard, non-convex, disconnected front over 3 variables."""
    def evaluate(x):
        f1 = math.fsum(-10.0 * math.exp(-0.2 * math.sqrt(x[i] ** 2 + x[i + 1] ** 2))
                       for i in range(len(x) - 1))
        f2 = math.fsum(abs(xi) ** 0.8 + 5.0 * math.sin(xi ** 3) for xi in x)
        return [f1, f2]
    return MOProblem(n_var, 2, evaluate, bounds=[(-5.0, 5.0)] * n_var,
                     name="Kursawe")


def fonseca_fleming(n_var: int = 3) -> MOProblem:
    """Fonseca-Fleming: a symmetric non-convex front; both objectives in [0,1]."""
    def evaluate(x):
        n = len(x)
        s = 1.0 / math.sqrt(n)
        f1 = 1.0 - math.exp(-math.fsum((xi - s) ** 2 for xi in x))
        f2 = 1.0 - math.exp(-math.fsum((xi + s) ** 2 for xi in x))
        return [f1, f2]
    return MOProblem(n_var, 2, evaluate, bounds=[(-4.0, 4.0)] * n_var,
                     name="Fonseca-Fleming")


def binh_korn() -> MOProblem:
    """Binh-Korn: a **constrained** bi-objective problem with two inequality
    constraints -- exercises the constraint-domination machinery."""
    def evaluate(x):
        f1 = 4.0 * x[0] ** 2 + 4.0 * x[1] ** 2
        f2 = (x[0] - 5.0) ** 2 + (x[1] - 5.0) ** 2
        return [f1, f2]

    def constraints(x):
        g1 = (x[0] - 5.0) ** 2 + x[1] ** 2 - 25.0            # <= 0
        g2 = 7.7 - ((x[0] - 8.0) ** 2 + (x[1] + 3.0) ** 2)   # <= 0
        return [g1, g2]
    return MOProblem(2, 2, evaluate, bounds=[(0.0, 5.0), (0.0, 3.0)],
                     n_constr=2, constraints=constraints, name="Binh-Korn")


def zdt1_front(n: int = 100) -> List[Vector]:
    """Analytic ZDT1 Pareto front ``{(f1, 1 - sqrt(f1))}`` for IGD/GD oracles."""
    return [[f1, 1.0 - math.sqrt(f1)] for f1 in (i / (n - 1) for i in range(n))]


def zdt2_front(n: int = 100) -> List[Vector]:
    """Analytic ZDT2 Pareto front ``{(f1, 1 - f1^2)}``."""
    return [[f1, 1.0 - f1 ** 2] for f1 in (i / (n - 1) for i in range(n))]


def zdt3_front(n: int = 200) -> List[Vector]:
    """Analytic ZDT3 Pareto front (the five non-dominated segments)."""
    pts = []
    for i in range(n):
        f1 = i / (n - 1)
        f2 = 1.0 - math.sqrt(f1) - f1 * math.sin(10.0 * math.pi * f1)
        pts.append([f1, f2])
    front = [pts[i] for i in find_pareto_front(pts)]
    return sorted(front, key=lambda p: p[0])


def dtlz2_front(n_obj: int = 3, n_partitions: int = 12) -> List[Vector]:
    """Analytic DTLZ2 front: the unit sphere sampled on the Das-Dennis simplex."""
    pts = []
    for w in das_dennis(n_obj, n_partitions):
        norm = math.sqrt(math.fsum(wi * wi for wi in w)) or _EPS
        pts.append([wi / norm for wi in w])
    return pts


# ===========================================================================
# DEMO
# ===========================================================================
def _demo() -> None:
    print("=" * 70)
    print("  MULTI-OBJECTIVE OPTIMISATION (pure Python) - demo")
    print("=" * 70)

    ref2 = [1.1, 1.1]
    pf = zdt1_front(200)

    print("\n-- NSGA-II on ZDT1 (convex 2-obj front) --")
    r = nsga2(zdt1(10), pop_size=40, n_gen=60, seed=0)
    print(f"   |front|={len(r)}  HV={hypervolume(r.F_min, ref2):.4f}  "
          f"IGD={inverted_generational_distance(r.F_min, pf):.4f}")

    print("-- SPEA2 on ZDT2 (non-convex) --")
    r2 = spea2(zdt2(10), pop_size=50, n_gen=100, seed=0)
    print(f"   |front|={len(r2)}  HV={hypervolume(r2.F_min, ref2):.4f}  "
          f"IGD={inverted_generational_distance(r2.F_min, zdt2_front(200)):.4f}")

    print("-- MOEA/D (Tchebycheff) on ZDT1 --")
    r3 = moead(zdt1(10), n_partitions=39, n_gen=80, seed=0)
    print(f"   |front|={len(r3)}  IGD={inverted_generational_distance(r3.F_min, pf):.4f}")

    print("-- SMS-EMOA (hypervolume-driven) on ZDT1 --")
    r4 = sms_emoa(zdt1(10), pop_size=40, n_gen=40, ref_point=ref2, seed=0)
    print(f"   |front|={len(r4)}  HV={hypervolume(r4.F_min, ref2):.4f}")

    print("-- NSGA-III on DTLZ2 (3 objectives) --")
    r5 = nsga3(dtlz2(12, 3), n_partitions=8, n_gen=60, seed=0)
    igd3 = inverted_generational_distance(r5.F_min, dtlz2_front(3, 12))
    print(f"   |front|={len(r5)}  IGD={igd3:.4f}")

    print("-- IBEA on ZDT3 (disconnected) --")
    r6 = ibea(zdt3(10), pop_size=40, n_gen=50, seed=0)
    print(f"   |front|={len(r6)}  HV={hypervolume(r6.F_min, [1.1, 6.0]):.4f}")

    print("-- constrained NSGA-II on Binh-Korn --")
    r7 = nsga2(binh_korn(), pop_size=40, n_gen=50, seed=0)
    feasible = all(all(g <= 1e-6 for g in binh_korn().constraints(x)) for x in r7.X)
    print(f"   |front|={len(r7)}  all feasible={feasible}")

    print("-- decision making on the ZDT1 front --")
    best_topsis, _ = topsis(r.F, weights=[0.5, 0.5])
    knee = knee_point(r.F)
    print(f"   TOPSIS pick f={[round(v, 3) for v in r.F[best_topsis]]}  "
          f"knee f={[round(v, 3) for v in r.F[knee]]}")

    print("\n" + "=" * 70)
    print("  demo OK")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
