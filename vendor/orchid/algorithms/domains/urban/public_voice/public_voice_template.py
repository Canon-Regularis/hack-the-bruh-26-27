"""
Public Voice -- consultation feedback as a field over the street network
=======================================================================

**A consultation is a spatial signal before it is a pile of comments.**  People
pin worries to places -- the crossing outside the school, the rat-run past the
corner shop, the bus station that makes a child's asthma worse -- and a planner
needs that signal in the same coordinates as everything else the optimiser
sees: a value per street per theme.  This module turns located comments into
the paper's concern field ``s_e^j = sum_i sev_i K(dist(e, loc_i)) 1[theme_i = j]``,
pairs each theme with the network metric that answers it, and assembles the
public-voice objective ``J_voice(x) = sum_j sum_e s_e^j m_e^j(x)`` with
baseline normalisation, so that "did this design respond to what people said?"
becomes a number on the Pareto vector.

It is deliberately not ``urban/policy_compiler``: that module binds the
*written* constraints of a plan (TROs, policy clauses) to network elements and
compiles them into fixings.  This one handles the *spoken* voice -- many
short, located, noisy, duplicated, self-selected comments -- and owns the
statistics that make them trustworthy: near-duplicate collapse so a copy-paste
campaign is not mistaken for a groundswell, hotspot tests with false-discovery
control so a heat map's red blob is not noise, and self-selection diagnostics
so the loudest postcode is not mistaken for the whole town.  Nor is it
``urban/accessibility_equity``, which also has a Moran's I: there the values
are accessibility scores; here they are concern intensities, and the local
statistics, the Getis-Ord family and the hotspot-to-comment bookkeeping are
this module's own.

Everything is pure standard library.  Dijkstra, MinHash with LSH banding,
SimHash with pigeonhole blocking, Fisher-Jenks dynamic programming, the
Cliff-Ord / Getis-Ord / Anselin null moments, Ripley's isotropic edge
correction and the incomplete-gamma chi-square tail are all built by hand.
No builtin ``hash()`` is used anywhere -- text fingerprints are FNV-1a,
SHA-1/SHA-256 or HMAC -- every stochastic routine takes an explicit ``seed``,
and every tie is broken on an explicit key, so output is byte-identical across
runs and independent of ``PYTHONHASHSEED``.

Contents
--------
0.  **Validation** -- ``PublicVoiceError`` and the ``_as_floats`` family.
1.  **Themes and concern records** -- ``THEMES`` / ``THEME_TAXONOMY``
    (crossing delay, road safety, rat-running, air quality, missing cycle
    link, accessibility, parking, severance, noise, other), ``ConcernRecord``,
    ``make_concern``, the keyword ``classify_theme``, ``theme_counts`` and
    ``theme_summary``.
2.  **Anonymisation at ingestion** -- ``redact_text``, ``pseudonymise``
    (HMAC-SHA256), ``coarsen_location``, donut ``jitter_location``,
    ``k_anonymity_cells`` and ``ingest_concerns``.
3.  **Street network & point-to-edge attachment** -- ``VoiceNetwork``,
    ``network_from_grid``, ``project_point_to_segment``, ``attach_point``,
    ``attach_concerns``, an own ``dijkstra``, ``network_distances_to_edges``
    and ``network_distance_between``.
4.  **Kernels** -- Gaussian, Epanechnikov, triangular, quartic, uniform and
    exponential, as peak profiles, 1-D densities and 2-D radial densities,
    with closed-form moments and canonical bandwidths.
5.  **Bandwidth selection** -- Silverman (1-D and spatial),
    leave-one-out likelihood and least-squares cross-validation, Abramson
    adaptive and k-nearest-neighbour bandwidths.
6.  **The concern field** -- ``concern_field`` with Euclidean or network
    distance, fixed or adaptive bandwidth; ``concern_contributions`` and the
    raster ``concern_surface``.
7.  **Theme-metric pairing & J_voice** -- ``ThemeMetricPair``,
    ``default_pairing`` / ``make_pairing``, ``normalise_metric``,
    ``voice_objective``, ``voice_improvement`` and ``heatmap_layers``.
8.  **Near-duplicate detection** -- FNV-1a, ``canonical_form``, character
    and word shingles, exact Jaccard, ``MinHasher`` with LSH banding,
    ``simhash``, ``find_near_duplicates`` and ``collapse_duplicates``.
9.  **Spatial weights** -- distance band, k-nearest (asymmetric), kernel,
    network contiguity, explicit matrices and row standardisation.
10. **Global autocorrelation** -- Moran's I, Geary's C, Getis-Ord General
    G with exact randomisation moments, permutation distributions, Ripley's
    K / L with three edge corrections and a CSR envelope.
11. **Hotspots** -- Benjamini-Hochberg FDR, local Getis-Ord ``G_i`` and
    ``G_i*``, local Moran's I with HH / LL / HL / LH classification, hotspot
    clusters and comment counts per hotspot.
12. **Heat-map classification** -- Jenks natural breaks (exact DP), equal
    interval and quantile breaks, square and hexagonal binning.
13. **Self-selection diagnostics** -- representativeness (dissimilarity,
    chi-square), post-stratification and raking weights, Kish effective
    sample size and design effect, weighted theme shares, the R-indicator and
    spatial representation.
14. **Payloads** -- ``build_public_voice_payload`` and ``build_a2ui_payload``.
15. **Demo**.

Conventions
-----------
* Coordinates are projected metres; distances and bandwidths are metres.
  Severity is any non-negative score (1-5 in the demo); ``count`` is how many
  submissions a record stands for.
* **Kernel normalisation.**  The concern field uses PEAK-normalised kernels by
  default: ``K_h(d) = k(d / h)`` with ``k(0) = 1``, so a single comment at
  distance 0 contributes exactly its severity, and the field is in severity
  units.  The ``"density"`` (1-D, ``1/m``) and ``"area"`` (2-D, ``1/m^2``)
  forms integrate to 1 and are used for the raster surface and for
  bandwidth cross-validation.
* A theme whose pairing has no metric (``parking`` and ``other`` by default)
  is shown on the heat map but EXCLUDED from ``J_voice``.
* Weights matrices have a zero diagonal and may be asymmetric; every null
  moment uses the general ``S0`` / ``S1`` / ``S2``.
* Guards are scale-relative (``_EPS * scale`` or exact zero); rescaling
  coordinates, bandwidths and cut-offs together by any factor leaves every
  dimensionless output unchanged.
* Ids are strings and every ordering that could reach a number is an explicit
  sorted order, never ``set`` or ``dict`` iteration.
"""

from __future__ import annotations

import bisect
import hashlib
import heapq
import hmac
import itertools
import json
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__version__ = "1.0.0"

# --------------------------------------------------------------------------
# Optional core import.  The core probabilistic library's ``p_adjust`` is the
# repo's reference Benjamini-Hochberg implementation; ``benjamini_hochberg``
# validates its input here FIRST (so a NaN or an out-of-range p-value raises
# PublicVoiceError on either path) and only then delegates.  The local
# fallback is the same step-up recursion in the same order, so the two paths
# return bit-identical values (the suite asserts this when the core imports).
# --------------------------------------------------------------------------
try:
    from algorithms.core.probabilistic.probabilistic_template import (
        p_adjust as _core_p_adjust,
    )
    _HAVE_CORE_STATS = True
except ImportError:
    _HAVE_CORE_STATS = False

Vector = List[float]
Matrix = List[List[float]]
Point = Tuple[float, float]

_EPS = 1e-12
_SQRT2 = math.sqrt(2.0)

__all__ = [
    # errors
    "PublicVoiceError",
    # themes and concern records
    "THEMES", "THEME_TAXONOMY", "ThemeInfo", "MODES", "SOURCES", "ConcernRecord",
    "make_concern", "ThemeScore", "classify_theme", "theme_counts", "ThemeSummary",
    "theme_summary",
    # anonymisation at ingestion
    "PII_FIELDS", "RedactionResult", "redact_text", "pseudonymise", "coarsen_location",
    "jitter_location", "KAnonymityReport", "k_anonymity_cells", "IngestReport",
    "IngestResult", "ingest_concerns",
    # street network and attachment
    "VoiceNetwork", "network_from_grid", "Projection", "project_point_to_segment",
    "point_segment_distance", "Attachment", "attach_point", "attach_concerns",
    "dijkstra", "network_distances_to_edges", "network_distance_between",
    # kernels
    "KERNELS", "KERNEL_NORMALISATIONS", "kernel_profile", "kernel_density_1d",
    "kernel_density_2d", "kernel_weight", "kernel_support", "KernelMoments",
    "kernel_moments",
    # bandwidth selection
    "BandwidthResult", "silverman_bandwidth", "spatial_silverman_bandwidth",
    "loo_log_likelihood", "lscv_score", "cv_bandwidth", "AdaptiveBandwidths",
    "kde_at_points", "abramson_bandwidths", "knn_bandwidths",
    # concern field
    "DISTANCE_MODES", "EDGE_REFERENCES", "ConcernField", "concern_field",
    "concern_contributions", "ConcernSurface", "concern_surface",
    # theme-metric pairing and J_voice
    "NORMALISATION_MODES", "ThemeMetricPair", "default_pairing", "make_pairing",
    "normalise_metric", "VoiceObjective", "voice_objective", "voice_improvement",
    "HeatmapCell", "heatmap_layers",
    # near-duplicate detection
    "fnv1a_64", "fnv1a_32", "stable_hash64", "canonical_form", "char_shingles",
    "word_shingles", "jaccard", "MinHasher", "lsh_probability", "lsh_threshold",
    "lsh_candidate_pairs", "simhash", "hamming_distance", "DUPLICATE_METHODS",
    "DuplicateReport", "find_near_duplicates", "COLLAPSE_POLICIES",
    "collapse_duplicates",
    # spatial weights
    "SpatialWeights", "weights_from_matrix", "row_standardise", "weights_distance_band",
    "weights_knn", "weights_kernel", "weights_from_network",
    # global autocorrelation
    "GlobalStatistic", "morans_i", "gearys_c", "general_g", "permutation_distribution",
    "RIPLEY_CORRECTIONS", "RipleyResult", "ripley_edge_weight", "ripleys_k",
    "Envelope", "csr_envelope",
    # hotspots
    "benjamini_hochberg", "bh_reject", "LocalGResult", "local_getis_ord",
    "LocalMoranResult", "local_morans_i", "Hotspot", "hotspot_clusters",
    "comment_counts_per_hotspot",
    # heat-map classification
    "JenksResult", "jenks_breaks", "equal_interval_breaks", "quantile_breaks",
    "classify_values", "HeatmapClasses", "heatmap_classes", "bin_points_square",
    "hex_of_point", "hex_centre", "bin_points_hex",
    # self-selection diagnostics
    "chi2_sf", "RepresentativenessResult", "representativeness",
    "post_stratification_weights", "RakingResult", "rake_weights",
    "kish_effective_sample_size", "design_effect", "weighted_theme_shares",
    "r_indicator", "SpatialRepresentation", "spatial_representation",
    # payloads
    "build_public_voice_payload", "build_a2ui_payload",
]


# ===========================================================================
# 0. VALIDATION HELPERS
# ===========================================================================
class PublicVoiceError(ValueError):
    """Invalid input to a public-voice routine."""


def _is_number_like(x: Any) -> bool:
    """True for real numbers that are not bools or strings."""
    return not isinstance(x, (bool, str, bytes)) and x is not None


def _as_float(x: Any, name: str = "x") -> float:
    """Coerce a scalar to a finite float or raise."""
    if not _is_number_like(x):
        raise PublicVoiceError("%s must be a real number (got %r)." % (name, x))
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):
        raise PublicVoiceError("%s must be a real number (got %r)." % (name, x))
    if math.isnan(v) or math.isinf(v):
        raise PublicVoiceError("%s must be finite (got %r)." % (name, x))
    return v


def _as_floats(xs: Any, name: str = "x", allow_empty: bool = False) -> Vector:
    """Coerce a sequence to a list of finite floats or raise."""
    if isinstance(xs, (str, bytes)) or xs is None:
        raise PublicVoiceError("%s must be a sequence of real numbers." % name)
    try:
        items = list(xs)
    except TypeError:
        raise PublicVoiceError("%s must be a sequence of real numbers." % name)
    if not items and not allow_empty:
        raise PublicVoiceError("%s must not be empty." % name)
    return [_as_float(v, "%s[%d]" % (name, i)) for i, v in enumerate(items)]


def _as_positive(x: Any, name: str = "x") -> float:
    """Coerce to a strictly positive finite float or raise."""
    v = _as_float(x, name)
    if v <= 0.0:
        raise PublicVoiceError("%s must be > 0 (got %r)." % (name, x))
    return v


def _as_nonneg(x: Any, name: str = "x") -> float:
    """Coerce to a non-negative finite float or raise."""
    v = _as_float(x, name)
    if v < 0.0:
        raise PublicVoiceError("%s must be >= 0 (got %r)." % (name, x))
    return v


def _as_unit(x: Any, name: str = "x", open_low: bool = False,
             open_high: bool = False) -> float:
    """Coerce to a float in [0, 1] (optionally open at either end) or raise."""
    v = _as_float(x, name)
    lo_bad = v <= 0.0 if open_low else v < 0.0
    hi_bad = v >= 1.0 if open_high else v > 1.0
    if lo_bad or hi_bad:
        raise PublicVoiceError("%s must lie in %s0, 1%s (got %r)."
                               % (name, "(" if open_low else "[", ")" if open_high else "]",
                                  x))
    return v


def _as_int(x: Any, name: str = "x", lo: Optional[int] = None,
            hi: Optional[int] = None) -> int:
    """Coerce an integral value to int, enforcing optional bounds, or raise."""
    if isinstance(x, bool) or not _is_number_like(x):
        raise PublicVoiceError("%s must be an integer (got %r)." % (name, x))
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x) or x != math.floor(x):
            raise PublicVoiceError("%s must be an integer (got %r)." % (name, x))
    try:
        v = int(x)
    except (TypeError, ValueError, OverflowError):
        raise PublicVoiceError("%s must be an integer (got %r)." % (name, x))
    if not isinstance(x, float) and v != x:
        raise PublicVoiceError("%s must be an integer (got %r)." % (name, x))
    if lo is not None and v < lo:
        raise PublicVoiceError("%s must be >= %d (got %d)." % (name, lo, v))
    if hi is not None and v > hi:
        raise PublicVoiceError("%s must be <= %d (got %d)." % (name, hi, v))
    return v


def _as_bool(x: Any, name: str = "flag") -> bool:
    """A flag must be a real bool: ``"no"``, ``nan`` or ``[]`` are not silently
    read through their truthiness."""
    if not isinstance(x, bool):
        raise PublicVoiceError("%s must be a bool (got %r)." % (name, x))
    return x


def _as_seed(x: Any, name: str = "seed") -> int:
    """A seed is any integer (bools rejected)."""
    return _as_int(x, name)


def _as_id(x: Any, name: str = "id") -> str:
    """Identifiers are non-empty strings."""
    if not isinstance(x, str) or not x:
        raise PublicVoiceError("%s must be a non-empty string (got %r)." % (name, x))
    return x


def _as_text(x: Any, name: str = "text") -> str:
    """Free text must be a str (may be empty)."""
    if not isinstance(x, str):
        raise PublicVoiceError("%s must be a string (got %r)." % (name, type(x).__name__))
    return x


def _check_choice(x: Any, choices: Sequence[str], name: str = "option") -> str:
    """Validate an enumerated string option."""
    if not isinstance(x, str) or x not in choices:
        raise PublicVoiceError("%s must be one of %s (got %r)."
                               % (name, ", ".join(choices), x))
    return x


def _as_point(p: Any, name: str = "point") -> Point:
    """Coerce an (x, y) pair to a tuple of finite floats or raise."""
    if isinstance(p, (str, bytes)) or p is None:
        raise PublicVoiceError("%s must be an (x, y) pair." % name)
    try:
        items = list(p)
    except TypeError:
        raise PublicVoiceError("%s must be an (x, y) pair." % name)
    if len(items) != 2:
        raise PublicVoiceError("%s must have exactly two coordinates (got %d)."
                               % (name, len(items)))
    return (_as_float(items[0], name + ".x"), _as_float(items[1], name + ".y"))


def _as_points(ps: Any, name: str = "points", allow_empty: bool = False) -> List[Point]:
    """Coerce a sequence of (x, y) pairs or raise."""
    if isinstance(ps, (str, bytes)) or ps is None:
        raise PublicVoiceError("%s must be a sequence of (x, y) pairs." % name)
    try:
        items = list(ps)
    except TypeError:
        raise PublicVoiceError("%s must be a sequence of (x, y) pairs." % name)
    if not items and not allow_empty:
        raise PublicVoiceError("%s must not be empty." % name)
    return [_as_point(p, "%s[%d]" % (name, i)) for i, p in enumerate(items)]


def _as_matrix(m: Any, n: int, name: str = "matrix") -> Matrix:
    """Coerce an n x n matrix of finite floats or raise."""
    if isinstance(m, (str, bytes)) or m is None:
        raise PublicVoiceError("%s must be a square matrix." % name)
    try:
        rows = list(m)
    except TypeError:
        raise PublicVoiceError("%s must be a square matrix." % name)
    if len(rows) != n:
        raise PublicVoiceError("%s must have %d rows (got %d)." % (name, n, len(rows)))
    out = []
    for i, row in enumerate(rows):
        r = _as_floats(row, "%s[%d]" % (name, i), allow_empty=(n == 0))
        if len(r) != n:
            raise PublicVoiceError("%s row %d must have %d entries (got %d)."
                                   % (name, i, n, len(r)))
        out.append(r)
    return out


def _str_keys(d: Any, name: str = "dict") -> List[str]:
    """Sorted keys of a dict whose keys must all be non-empty strings.

    Validating BEFORE sorting matters: ``sorted`` on mixed key types raises a
    bare ``TypeError``."""
    if not isinstance(d, dict):
        raise PublicVoiceError("%s must be a dict." % name)
    for k in d:
        if not isinstance(k, str) or not k:
            raise PublicVoiceError("%s keys must be non-empty strings (got %r)." % (name, k))
    return sorted(d)


def _as_seq(xs: Any, name: str = "sequence") -> List[Any]:
    """A non-string iterable as a list, or raise."""
    if isinstance(xs, (str, bytes, dict)) or xs is None:
        raise PublicVoiceError("%s must be a sequence." % name)
    try:
        return list(xs)
    except TypeError:
        raise PublicVoiceError("%s must be a sequence." % name)


def _cell_index(v: float, name: str = "coordinate / cell") -> int:
    """``floor(v)`` as an int; a ratio that overflowed to infinity is bad input."""
    if math.isinf(v) or math.isnan(v):
        raise PublicVoiceError("%s is out of range (cell too small for the coordinate)."
                               % name)
    return int(math.floor(v))


def _fsum(xs: Iterable[float]) -> float:
    """Correctly rounded sum."""
    return math.fsum(xs)


def _mean(xs: Sequence[float]) -> float:
    """Arithmetic mean through ``math.fsum``."""
    return math.fsum(xs) / len(xs)


def _safe_mean(xs: Sequence[float]) -> float:
    """``_mean``, bit for bit, but a sum that overflows (values near 1e308)
    is taken over values divided by an exact power of two instead."""
    try:
        return math.fsum(xs) / len(xs)
    except OverflowError:
        e = math.frexp(max(abs(v) for v in xs))[1]
        return math.ldexp(math.fsum(math.ldexp(v, -e) for v in xs) / len(xs), e)


def _sample_var(xs: Sequence[float]) -> float:
    """Unbiased sample variance (two-pass, centred)."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return math.fsum((v - m) * (v - m) for v in xs) / (n - 1)


def _sample_sd(xs: Sequence[float]) -> float:
    """Sample standard deviation, overflow- and underflow-proof.

    The centred deviations are divided by an exact POWER OF TWO before being
    squared (squares overflow above ~1e154 and underflow below ~1e-162), and
    the root is scaled back by the same power: because ``sqrt(x 4^-e) =
    sqrt(x) 2^-e`` exactly, the result is bit-identical to
    ``sqrt(_sample_var(xs))`` wherever that does not over/underflow."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    z = [v - m for v in xs]
    big = max(abs(v) for v in z)
    if big == 0.0:
        return 0.0
    e = math.frexp(big)[1]
    z = [math.ldexp(v, -e) for v in z]
    return math.ldexp(math.sqrt(math.fsum(v * v for v in z) / (n - 1)), e)


def _quantile(sorted_xs: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (Hyndman-Fan type 7, numpy's default)."""
    n = len(sorted_xs)
    if n == 1:
        return sorted_xs[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * frac


def _normal_sf_two_sided(z: float) -> float:
    """Two-sided normal tail probability ``P(|Z| >= |z|)``."""
    return math.erfc(abs(z) / _SQRT2)


def _derived_seed(seed: int, key: str) -> int:
    """A 64-bit integer seed derived from (seed, key) with SHA-256 -- hash-seed proof."""
    digest = hashlib.sha256(("%d|%s" % (seed, key)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


# ===========================================================================
# 1. THEMES AND CONCERN RECORDS -- taxonomy, records, keyword classifier
# ===========================================================================
@dataclass(frozen=True)
class ThemeInfo:
    """One entry of the theme taxonomy."""
    theme: str
    label: str
    description: str
    keywords: Tuple[str, ...]
    default_metric: Optional[str]
    direction: str = "lower"

    @property
    def measurable(self) -> bool:
        """True when the theme has a measurable network counterpart."""
        return self.default_metric is not None


THEMES: Tuple[str, ...] = (
    "crossing_delay", "road_safety", "rat_running", "air_quality",
    "missing_cycle_link", "accessibility", "parking", "severance", "noise",
    "other",
)

THEME_TAXONOMY: Dict[str, ThemeInfo] = {
    "crossing_delay": ThemeInfo(
        "crossing_delay", "Crossing delay",
        "Waiting too long to cross; no crossing where one is needed.",
        ("crossing", "cross", "wait", "waiting", "zebra", "pelican", "puffin",
         "toucan", "traffic light", "pedestrian light", "at the lights", "green man",
         "button"),
        "ped_crossing_delay_s"),
    "road_safety": ThemeInfo(
        "road_safety", "Road safety",
        "Speeding, near misses, collisions, dangerous junctions.",
        ("speed", "speeding", "dangerous", "danger", "accident", "collision",
         "near miss", "unsafe", "crash", "hit", "fast"),
        "collision_risk"),
    "rat_running": ThemeInfo(
        "rat_running", "Rat-running",
        "Through traffic cutting along residential streets.",
        ("rat run", "rat-run", "rat running", "rat-running", "cut through",
         "cut-through", "shortcut", "through traffic", "satnav"),
        "through_traffic_veh_h"),
    "air_quality": ThemeInfo(
        "air_quality", "Air quality",
        "Fumes, idling, pollution, asthma.",
        ("pollution", "polluted", "fumes", "air quality", "exhaust", "idling",
         "asthma", "no2", "smog", "emissions"),
        "no2_ug_m3"),
    "missing_cycle_link": ThemeInfo(
        "missing_cycle_link", "Missing cycle link",
        "A gap in the cycle network; a lane that ends abruptly.",
        ("cycle lane", "bike lane", "cycle path", "cycling", "cyclist",
         "bike", "bicycle", "cycle route", "segregated"),
        "cycle_route_gap"),
    "accessibility": ThemeInfo(
        "accessibility", "Accessibility",
        "Step-free access, dropped kerbs, wheelchair and buggy access.",
        ("wheelchair", "dropped kerb", "kerb", "buggy", "pushchair",
         "step-free", "disabled", "mobility", "blind", "tactile"),
        "accessibility_score", "higher"),
    "parking": ThemeInfo(
        "parking", "Parking",
        "Pavement parking, permit pressure, loading.",
        ("parking", "parked", "car park", "to park", "park on", "parks on", "permit",
         "loading", "double yellow", "pavement parking", "spaces"),
        None),
    "severance": ThemeInfo(
        "severance", "Severance",
        "A road that cuts a community in two.",
        ("severance", "barrier", "cut off", "divides", "no way across",
         "wall of traffic", "isolated", "split"),
        "severance_index"),
    "noise": ThemeInfo(
        "noise", "Noise",
        "Traffic noise, HGVs at night, motorbikes.",
        ("noise", "noisy", "loud", "rumble", "hgv", "lorry", "motorbike",
         "engine", "sleep"),
        "noise_lden_db"),
    "other": ThemeInfo(
        "other", "Other",
        "Anything the taxonomy does not cover.",
        (), None),
}

MODES: Tuple[str, ...] = ("walk", "cycle", "bus", "car", "freight", "all")

SOURCES: Tuple[str, ...] = ("map_pin", "survey", "email", "workshop", "letter",
                            "social_media", "petition")


@dataclass
class ConcernRecord:
    """A located concern: one consultation comment pinned to a place.

    ``count`` is the number of identical submissions the record stands for
    (after duplicate collapse it may be any positive real -- see
    ``collapse_duplicates``).  ``respondent`` is a PSEUDONYM, never a name."""
    cid: str
    theme: str
    severity: float
    x: float
    y: float
    mode: str = "all"
    source: str = "map_pin"
    count: float = 1.0
    text: str = ""
    respondent: str = ""
    stratum: str = ""

    def __post_init__(self) -> None:
        self.cid = _as_id(self.cid, "cid")
        self.theme = _check_choice(self.theme, THEMES, "theme")
        self.severity = _as_nonneg(self.severity, "severity")
        self.x = _as_float(self.x, "x")
        self.y = _as_float(self.y, "y")
        self.mode = _check_choice(self.mode, MODES, "mode")
        self.source = _check_choice(self.source, SOURCES, "source")
        self.count = _as_positive(self.count, "count")
        if math.isinf(self.severity * self.count):
            raise PublicVoiceError("severity x count overflows (%r x %r)."
                                   % (self.severity, self.count))
        self.text = _as_text(self.text, "text")
        self.respondent = _as_text(self.respondent, "respondent")
        self.stratum = _as_text(self.stratum, "stratum")

    @property
    def point(self) -> Point:
        """The (x, y) location."""
        return (self.x, self.y)

    @property
    def weight(self) -> float:
        """Severity times count -- the record's mass in the concern field."""
        return self.severity * self.count

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able camelCase dict."""
        return {"id": self.cid, "theme": self.theme, "severity": self.severity,
                "x": self.x, "y": self.y, "mode": self.mode, "source": self.source,
                "count": self.count, "text": self.text,
                "respondent": self.respondent, "stratum": self.stratum}


def make_concern(cid: str, theme: str, severity: float, x: float, y: float,
                 **kwargs: Any) -> ConcernRecord:
    """Validated ``ConcernRecord`` constructor (keyword pass-through)."""
    allowed = ("mode", "source", "count", "text", "respondent", "stratum")
    for k in sorted(kwargs):
        if k not in allowed:
            raise PublicVoiceError("make_concern: unknown field %r." % k)
    return ConcernRecord(cid, theme, severity, x, y, **kwargs)


def _check_concerns(concerns: Any, name: str = "concerns",
                    allow_empty: bool = False) -> List[ConcernRecord]:
    """Validate a list of ConcernRecord with unique ids; return sorted by id."""
    if isinstance(concerns, (str, bytes)) or concerns is None:
        raise PublicVoiceError("%s must be a sequence of ConcernRecord." % name)
    try:
        items = list(concerns)
    except TypeError:
        raise PublicVoiceError("%s must be a sequence of ConcernRecord." % name)
    if not items and not allow_empty:
        raise PublicVoiceError("%s must not be empty." % name)
    seen = set()
    for c in items:
        if not isinstance(c, ConcernRecord):
            raise PublicVoiceError("%s must contain only ConcernRecord (got %s)."
                                   % (name, type(c).__name__))
        if c.cid in seen:
            raise PublicVoiceError("%s has a duplicate id %r." % (name, c.cid))
        seen.add(c.cid)
    return sorted(items, key=lambda c: c.cid)


@dataclass
class ThemeScore:
    """Keyword-classifier output for one text."""
    theme: str
    scores: Dict[str, int]
    matched: Dict[str, List[str]]
    confident: bool


def _keyword_hits(text_l: str, kw: str) -> int:
    """Count whole-word (or whole-phrase) occurrences of ``kw`` in ``text_l``,
    its regular plural (``-s`` / ``-es``, ``-y`` -> ``-ies``) included: "cycle
    lanes", "wheelchairs", "buggies" and "near misses" are the same concern as
    the singular, and missing them tipped mixed comments into the other theme."""
    if kw.endswith("y") and len(kw) > 1 and kw[-2] not in "aeiou":
        body = re.escape(kw[:-1]) + r"(?:y|ies)"
    else:
        body = re.escape(kw) + r"(?:e?s)?"
    pattern = r"(?<![a-z0-9])" + body + r"(?![a-z0-9])"
    return len(re.findall(pattern, text_l))


# A proper street name -- capitalised words ending in a thoroughfare type --
# is a PLACE, not a concern: "Park Road" is not about parking and "Cross
# Street" not about crossing.  Masked before keyword matching.
_RE_STREET_NAME = re.compile(
    r"\b(?:[A-Z][a-z]+\s+){1,3}"
    r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Close|Drive|Dr|Way|Gardens|Place|"
    r"Terrace|Crescent|Court|Grove|Hill|Row|Square|Walk)\b")

# A hyphen (or Unicode hyphen / en dash) inside a word pair, or any run of
# whitespace, is one space: "near-miss", "cycle-lane", "air-quality" and a
# line-wrapped "cycle<newline>lane" are the phrases "near miss", "cycle lane".
_RE_JOINER = re.compile(r"(?<=[a-z0-9])[-\u2010\u2011\u2012\u2013](?=[a-z0-9])|\s+")
# ... but a line break before a CAPITALISED line starts a new list item
# ("Too much traffic<newline>Lights out on the path" is not "traffic
# lights"); a wrapped phrase continues in lower case.
_RE_NEW_LINE_ITEM = re.compile(r"[ \t]*[\r\n]\s*(?=[A-Z])")


def _norm_phrase(t: str) -> str:
    """Lower-cased text with hyphens and whitespace runs as single spaces."""
    return _RE_JOINER.sub(" ", t.lower())


def _has_keyword(t: str) -> bool:
    """True when normalised text ``t`` holds any taxonomy keyword."""
    return any(_keyword_hits(t, _norm_phrase(kw)) for th in THEMES
               for kw in THEME_TAXONOMY[th].keywords)


def _mask_street(m: Any) -> str:
    """Mask a matched street name but keep leading capitalised words that hold
    a keyword: the match can start at a sentence-initial verb ("Crossing
    Station Road takes ages", "Cycling Mill Lane ..."), and masking it filed
    the comment under "other" -- out of J_voice.  The last specific word and
    the thoroughfare type are always masked ("Cross Street" stays a place)."""
    words = m.group(0).split()
    for j in range(len(words) - 2, 0, -1):
        if _has_keyword(_norm_phrase(" ".join(words[:j]))):
            return " ".join(words[:j]) + " "
    return " "


def classify_theme(text: str) -> ThemeScore:
    """Deterministic keyword classifier over the taxonomy.

    The theme with the most keyword hits wins; ties are broken by the fixed
    order of ``THEMES`` (never by dict order), and a text with no hit is
    ``"other"``.  ``confident`` is False when the top score is tied.
    Keywords match whole words with their regular plural; proper street
    names ("Park Road", "Cross Street") are masked first, and the bare noun
    "park" (a green space as often as a car park) is not a keyword.  Hyphens
    and line breaks inside a phrase are spaces; spellings that then coincide
    ("rat run" / "rat-run") count once."""
    t = _RE_NEW_LINE_ITEM.sub(" . ", _as_text(text, "text"))
    t = _norm_phrase(_RE_STREET_NAME.sub(_mask_street, t))
    scores: Dict[str, int] = {}
    matched: Dict[str, List[str]] = {}
    for theme in THEMES:
        hits = []
        seen: Dict[str, int] = {}
        for kw in THEME_TAXONOMY[theme].keywords:
            nk = _norm_phrase(kw)
            if nk not in seen:
                seen[nk] = _keyword_hits(t, nk)
            if seen[nk]:
                hits.append(kw)
        scores[theme] = sum(seen[k] for k in sorted(seen))
        matched[theme] = hits
    best = max(scores[th] for th in THEMES)
    if best == 0:
        return ThemeScore("other", scores, matched, False)
    winners = [th for th in THEMES if scores[th] == best]
    return ThemeScore(winners[0], scores, matched, len(winners) == 1)


def theme_counts(concerns: Sequence[ConcernRecord],
                 weighted: bool = False) -> Dict[str, float]:
    """Records (or total ``count``) per theme, every theme present, in THEMES order."""
    cs = _check_concerns(concerns, allow_empty=True)
    _as_bool(weighted, "weighted")
    out: Dict[str, float] = {}
    for th in THEMES:
        if weighted:
            out[th] = math.fsum(c.count for c in cs if c.theme == th)
        else:
            out[th] = float(sum(1 for c in cs if c.theme == th))
    return out


@dataclass
class ThemeSummary:
    """Per-theme descriptive statistics."""
    theme: str
    n_records: int
    total_count: float
    mean_severity: float
    max_severity: float
    total_weight: float
    measurable: bool


def theme_summary(concerns: Sequence[ConcernRecord]) -> List[ThemeSummary]:
    """One ``ThemeSummary`` per theme with at least one record, in THEMES order."""
    cs = _check_concerns(concerns)
    out = []
    for th in THEMES:
        sub = [c for c in cs if c.theme == th]
        if not sub:
            continue
        tot = math.fsum(c.count for c in sub)
        out.append(ThemeSummary(
            th, len(sub), tot,
            math.fsum(c.severity * c.count for c in sub) / tot,
            max(c.severity for c in sub),
            math.fsum(c.weight for c in sub),
            THEME_TAXONOMY[th].measurable))
    return out


# ===========================================================================
# 2. ANONYMISATION AT INGESTION -- redaction, pseudonyms, geomasking
# ===========================================================================
_RE_URL = re.compile(r"(?:https?://|www\.)[^\s]+", re.IGNORECASE)
_RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_RE_HANDLE = re.compile(r"(?<![\w@])@[A-Za-z0-9_]{2,}")
_RE_POSTCODE = re.compile(r"\b[A-Z]{1,2}[0-9][A-Z0-9]?\s?[0-9][A-Z]{2}\b", re.IGNORECASE)
_RE_PHONE = re.compile(r"(?:\+44\s?\(?0?\)?\s?\d{2,4}|\b0\d{2,4}|\(0\d{2,4}\))"
                       r"[\s-]?\d{3,4}[\s-]?\d{3,4}\b"
                       r"|\b\d{10,}\b")
_RE_ADDRESS = re.compile(
    r"\b\d{1,4}[A-Za-z]?,?\s+(?:[A-Z][a-z]+\s+){1,3}"
    r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Close|Drive|Dr|Way|Gardens|Place|"
    r"Terrace|Crescent|Court|Grove|Hill|Row|Square|Walk)\b")
_RE_NAME = re.compile(r"\b((?:[Mm]y name is|[Tt]his is|[Ss]igned|[Rr]egards,?)\s+)"
                      r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")

# (label, compiled pattern, replacement) in application order.  URLs and
# e-mails go first so that their '@' and digits are not re-matched.
_REDACTIONS: Tuple[Tuple[str, Any, str], ...] = (
    ("url", _RE_URL, "[URL]"),
    ("email", _RE_EMAIL, "[EMAIL]"),
    ("handle", _RE_HANDLE, "[HANDLE]"),
    ("address", _RE_ADDRESS, "[ADDRESS]"),
    ("postcode", _RE_POSTCODE, "[POSTCODE]"),
    ("phone", _RE_PHONE, "[PHONE]"),
)

PII_FIELDS: Tuple[str, ...] = ("author", "name", "email", "phone", "address",
                               "postcode", "ip", "user_id", "username")


@dataclass
class RedactionResult:
    """Redacted text and per-category counts."""
    text: str
    counts: Dict[str, int]

    @property
    def total(self) -> int:
        """Total number of redactions."""
        return sum(self.counts[k] for k in sorted(self.counts))


def redact_text(text: str) -> RedactionResult:
    """Strip personal data from free text (URLs, e-mails, handles, street
    addresses with a house number, UK postcodes, phone numbers, self-named
    signatures).  Idempotent: redacting redacted text changes nothing."""
    t = _as_text(text, "text")
    counts: Dict[str, int] = {}
    for label, pat, repl in _REDACTIONS:
        t, k = pat.subn(repl, t)
        counts[label] = k
    t, k = _RE_NAME.subn(lambda m: m.group(1) + "[NAME]", t)
    counts["name"] = k
    return RedactionResult(t, counts)


def pseudonymise(identifier: str, salt: str, length: int = 16) -> str:
    """Keyed one-way pseudonym: the first ``length`` hex digits of
    HMAC-SHA256(salt, identifier).  Stable across runs and hash seeds, not
    reversible without the salt, and distinct salts give unlinkable ids."""
    ident = _as_text(identifier, "identifier")
    s = _as_text(salt, "salt")
    if not s:
        raise PublicVoiceError("salt must be a non-empty string.")
    ln = _as_int(length, "length", 8, 64)
    return hmac.new(s.encode("utf-8"), ident.encode("utf-8"),
                    hashlib.sha256).hexdigest()[:ln]


def coarsen_location(x: float, y: float, cell: float,
                     origin: Point = (0.0, 0.0)) -> Point:
    """Snap a point to the centre of its ``cell``-sized grid square (geomasking)."""
    xv = _as_float(x, "x")
    yv = _as_float(y, "y")
    c = _as_positive(cell, "cell")
    ox, oy = _as_point(origin, "origin")
    i = _cell_index((xv - ox) / c)
    j = _cell_index((yv - oy) / c)
    out = (ox + (i + 0.5) * c, oy + (j + 0.5) * c)
    if not (math.isfinite(out[0]) and math.isfinite(out[1])):
        raise PublicVoiceError("coarsened location of (%r, %r) at cell %r is beyond the "
                               "float range." % (xv, yv, c))
    return out


def jitter_location(x: float, y: float, r_min: float, r_max: float,
                    seed: int, key: str) -> Point:
    """Donut geomask: displace by a distance in ``[r_min, r_max]`` (uniform in
    AREA over the annulus) at a uniform angle.  The draw is keyed on
    ``(seed, key)`` through SHA-256, so it is reproducible per record and
    independent of record order and ``PYTHONHASHSEED``."""
    xv = _as_float(x, "x")
    yv = _as_float(y, "y")
    lo = _as_nonneg(r_min, "r_min")
    hi = _as_positive(r_max, "r_max")
    if lo > hi:
        raise PublicVoiceError("r_min must not exceed r_max (%r > %r)." % (lo, hi))
    rng = random.Random(_derived_seed(_as_seed(seed), _as_text(key, "key")))
    u = rng.random()
    r = math.sqrt(lo * lo + u * (hi * hi - lo * lo))
    theta = 2.0 * math.pi * rng.random()
    out = (xv + r * math.cos(theta), yv + r * math.sin(theta))
    if not (math.isfinite(out[0]) and math.isfinite(out[1])):
        raise PublicVoiceError("jittered location of (%r, %r) is beyond the float range."
                               % (xv, yv))
    return out


@dataclass
class KAnonymityReport:
    """Cells of a coarsening grid holding fewer than k respondents."""
    k: int
    cell: float
    cell_counts: Dict[Tuple[int, int], int]
    violating_cells: List[Tuple[int, int]]

    @property
    def satisfied(self) -> bool:
        """True when every occupied cell holds at least k records."""
        return not self.violating_cells


def k_anonymity_cells(points: Sequence[Point], cell: float, k: int,
                      origin: Point = (0.0, 0.0)) -> KAnonymityReport:
    """Count points per grid cell and list cells with fewer than ``k``."""
    pts = _as_points(points, "points", allow_empty=True)
    c = _as_positive(cell, "cell")
    kk = _as_int(k, "k", 1)
    ox, oy = _as_point(origin, "origin")
    counts: Dict[Tuple[int, int], int] = {}
    for (px, py) in pts:
        key = (_cell_index((px - ox) / c), _cell_index((py - oy) / c))
        counts[key] = counts.get(key, 0) + 1
    bad = sorted(kc for kc in counts if counts[kc] < kk)
    return KAnonymityReport(kk, c, {kc: counts[kc] for kc in sorted(counts)}, bad)


@dataclass
class IngestReport:
    """What ingestion did to the raw submissions."""
    n_in: int
    n_out: int
    rejected: List[Tuple[int, str]]
    pii_fields_dropped: Dict[str, int]
    redactions: Dict[str, int]
    auto_classified: int
    coarsened: int
    jittered: int


@dataclass
class IngestResult:
    """Anonymised concern records plus the ingestion report."""
    records: List[ConcernRecord]
    report: IngestReport


def ingest_concerns(raw: Sequence[Dict[str, Any]], *, salt: str,
                    location_cell: Optional[float] = None,
                    jitter: Optional[Tuple[float, float]] = None,
                    seed: int = 0, strict: bool = True,
                    default_severity: float = 1.0) -> IngestResult:
    """Anonymise raw consultation submissions at the door.

    Every PII field (``PII_FIELDS``) is dropped before anything else sees it;
    the author (``author``, else ``email``, else ``name``) survives only as a
    salted HMAC pseudonym in ``respondent``; free text is redacted; a missing
    or ``"auto"`` theme is classified from the REDACTED text; the location is
    optionally snapped to a ``location_cell`` grid and/or donut-jittered.
    Records are returned sorted by id.  With ``strict=False`` a malformed
    record is listed in ``report.rejected`` instead of raising."""
    if isinstance(raw, (str, bytes, dict)) or raw is None:
        raise PublicVoiceError("raw must be a sequence of dicts.")
    try:
        items = list(raw)
    except TypeError:
        raise PublicVoiceError("raw must be a sequence of dicts.")
    s = _as_text(salt, "salt")
    if not s:
        raise PublicVoiceError("salt must be a non-empty string.")
    cell = None if location_cell is None else _as_positive(location_cell, "location_cell")
    jit = None
    if jitter is not None:
        jp = _as_floats(jitter, "jitter")
        if len(jp) != 2 or jp[0] < 0.0 or jp[1] <= 0.0 or jp[0] > jp[1]:
            raise PublicVoiceError("jitter must be (r_min, r_max) with 0 <= r_min <= r_max, "
                                   "r_max > 0.")
        jit = (jp[0], jp[1])
    sd = _as_seed(seed)
    dsev = _as_nonneg(default_severity, "default_severity")
    if not isinstance(strict, bool):
        raise PublicVoiceError("strict must be a bool.")
    out: List[ConcernRecord] = []
    rejected: List[Tuple[int, str]] = []
    dropped = {f: 0 for f in PII_FIELDS}
    red_tot: Dict[str, int] = {}
    n_auto = n_coarse = n_jit = 0
    seen_ids = set()
    for idx, rec in enumerate(items):
        try:
            if not isinstance(rec, dict):
                raise PublicVoiceError("record %d is not a dict." % idx)
            author = ""
            for f in ("author", "email", "name"):
                v = rec.get(f)
                if isinstance(v, str) and v:
                    author = v
                    break
            for f in PII_FIELDS:
                if f in rec:
                    dropped[f] += 1
            cid = rec.get("id", "c%05d" % idx)
            if isinstance(cid, int) and not isinstance(cid, bool):
                cid = str(cid)
            cid = _as_id(cid, "id")
            if cid in seen_ids:
                raise PublicVoiceError("duplicate id %r." % cid)
            red = redact_text(rec.get("text", ""))
            theme = rec.get("theme", "auto")
            if theme is None or theme == "auto":
                theme = classify_theme(red.text).theme
                auto = 1
            else:
                auto = 0
            if "x" not in rec or "y" not in rec:
                raise PublicVoiceError("record %d has no location." % idx)
            px = _as_float(rec["x"], "x")
            py = _as_float(rec["y"], "y")
            coarse = jit_n = 0
            if cell is not None:
                px, py = coarsen_location(px, py, cell)
                coarse = 1
            if jit is not None:
                px, py = jitter_location(px, py, jit[0], jit[1], sd, cid)
                jit_n = 1
            respondent = pseudonymise(author, s) if author else ""
            c = ConcernRecord(cid, theme, rec.get("severity", dsev), px, py,
                              mode=rec.get("mode", "all"),
                              source=rec.get("source", "map_pin"),
                              count=rec.get("count", 1.0), text=red.text,
                              respondent=respondent,
                              stratum=rec.get("stratum", ""))
        except PublicVoiceError as exc:
            if strict:
                raise
            rejected.append((idx, str(exc)))
            continue
        seen_ids.add(cid)
        for k in sorted(red.counts):
            red_tot[k] = red_tot.get(k, 0) + red.counts[k]
        n_auto += auto
        n_coarse += coarse
        n_jit += jit_n
        out.append(c)
    out.sort(key=lambda c: c.cid)
    rep = IngestReport(len(items), len(out), rejected, dropped,
                       {k: red_tot[k] for k in sorted(red_tot)}, n_auto, n_coarse, n_jit)
    return IngestResult(out, rep)


# ===========================================================================
# 3. STREET NETWORK & POINT-TO-EDGE ATTACHMENT -- projection, own Dijkstra
# ===========================================================================
class VoiceNetwork:
    """A minimal undirected street graph for attaching and spreading concerns.

    ``nodes`` maps node id -> (x, y); ``edges`` is a sequence of
    ``(edge_id, u, v)`` or ``(edge_id, u, v, length)``.  When no length is
    given the straight-line length is used; a given length must be at least
    the chord (a street cannot be shorter than the straight line between its
    ends).  Edge geometry is the straight segment u-v.  Everything is stored
    in sorted-id order so no answer depends on insertion order."""

    def __init__(self, nodes: Dict[str, Any], edges: Sequence[Any],
                 name: str = "network") -> None:
        if not isinstance(nodes, dict) or not nodes:
            raise PublicVoiceError("nodes must be a non-empty dict id -> (x, y).")
        self.name = _as_text(name, "name")
        self._xy: Dict[str, Point] = {}
        for nid in sorted(nodes, key=lambda k: str(k)):
            self._xy[_as_id(nid, "node id")] = _as_point(nodes[nid], "node %r" % nid)
        if isinstance(edges, (str, bytes)) or edges is None:
            raise PublicVoiceError("edges must be a sequence of (id, u, v[, length]).")
        try:
            raw_edges = list(edges)
        except TypeError:
            raise PublicVoiceError("edges must be a sequence of (id, u, v[, length]).")
        if not raw_edges:
            raise PublicVoiceError("edges must not be empty.")
        self._edges: Dict[str, Tuple[str, str, float]] = {}
        for e in raw_edges:
            if isinstance(e, (str, bytes)):
                raise PublicVoiceError("each edge must be (id, u, v[, length]).")
            try:
                parts = list(e)
            except TypeError:
                raise PublicVoiceError("each edge must be (id, u, v[, length]).")
            if len(parts) not in (3, 4):
                raise PublicVoiceError("each edge must be (id, u, v[, length]).")
            eid = _as_id(parts[0], "edge id")
            u = _as_id(parts[1], "edge %s u" % eid)
            v = _as_id(parts[2], "edge %s v" % eid)
            if eid in self._edges:
                raise PublicVoiceError("duplicate edge id %r." % eid)
            if u not in self._xy or v not in self._xy:
                raise PublicVoiceError("edge %r references an unknown node." % eid)
            if u == v:
                raise PublicVoiceError("edge %r is a self-loop." % eid)
            (x1, y1), (x2, y2) = self._xy[u], self._xy[v]
            chord = math.hypot(x2 - x1, y2 - y1)
            if chord == 0.0:
                raise PublicVoiceError("edge %r has coincident end nodes." % eid)
            if len(parts) == 4:
                length = _as_positive(parts[3], "edge %s length" % eid)
                if length < chord * (1.0 - 1e-9):
                    raise PublicVoiceError("edge %r length %r is shorter than its chord %r."
                                           % (eid, length, chord))
            else:
                length = chord
            self._edges[eid] = (u, v, length)
        self._edge_ids = sorted(self._edges)
        self._node_ids = sorted(self._xy)
        adj: Dict[str, List[Tuple[str, str, float]]] = {n: [] for n in self._node_ids}
        for eid in self._edge_ids:
            u, v, ln = self._edges[eid]
            adj[u].append((v, eid, ln))
            adj[v].append((u, eid, ln))
        for n in self._node_ids:
            adj[n].sort(key=lambda t: (t[0], t[1]))
        self._adj = adj
        xs = [p[0] for p in self._xy.values()]
        ys = [p[1] for p in self._xy.values()]
        self.scale = max(max(xs) - min(xs), max(ys) - min(ys),
                         max(self._edges[e][2] for e in self._edge_ids))

    @property
    def node_ids(self) -> List[str]:
        """Sorted node ids."""
        return list(self._node_ids)

    @property
    def edge_ids(self) -> List[str]:
        """Sorted edge ids."""
        return list(self._edge_ids)

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return len(self._node_ids)

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return len(self._edge_ids)

    def has_edge(self, eid: str) -> bool:
        """True when ``eid`` is an edge id."""
        return isinstance(eid, str) and eid in self._edges

    def node_xy(self, nid: str) -> Point:
        """Coordinates of a node."""
        if not isinstance(nid, str) or nid not in self._xy:
            raise PublicVoiceError("unknown node %r." % (nid,))
        return self._xy[nid]

    def endpoints(self, eid: str) -> Tuple[str, str]:
        """(u, v) of an edge."""
        if not self.has_edge(eid):
            raise PublicVoiceError("unknown edge %r." % (eid,))
        u, v, _ = self._edges[eid]
        return (u, v)

    def length(self, eid: str) -> float:
        """Length of an edge in metres."""
        if not self.has_edge(eid):
            raise PublicVoiceError("unknown edge %r." % (eid,))
        return self._edges[eid][2]

    def segment(self, eid: str) -> Tuple[Point, Point]:
        """Straight-line geometry of an edge."""
        u, v = self.endpoints(eid)
        return (self._xy[u], self._xy[v])

    def midpoint(self, eid: str) -> Point:
        """Geometric midpoint of an edge."""
        (x1, y1), (x2, y2) = self.segment(eid)
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def neighbours(self, nid: str) -> List[Tuple[str, str, float]]:
        """Sorted (neighbour, edge id, length) triples of a node."""
        if not isinstance(nid, str) or nid not in self._adj:
            raise PublicVoiceError("unknown node %r." % (nid,))
        return list(self._adj[nid])

    def edge_adjacency(self) -> Dict[str, List[str]]:
        """Edges sharing a node with each edge (line-graph adjacency), sorted."""
        out: Dict[str, List[str]] = {}
        for eid in self._edge_ids:
            u, v, _ = self._edges[eid]
            nb = set()
            for n in (u, v):
                for (_, e2, _) in self._adj[n]:
                    if e2 != eid:
                        nb.add(e2)
            out[eid] = sorted(nb)
        return out

    def summary(self) -> Dict[str, Any]:
        """Headline numbers."""
        return {"name": self.name, "nNodes": self.n_nodes, "nEdges": self.n_edges,
                "totalLengthM": math.fsum(self._edges[e][2] for e in self._edge_ids)}


def network_from_grid(nx: int, ny: int, spacing: float, name: str = "grid",
                      origin: Point = (0.0, 0.0)) -> VoiceNetwork:
    """A rectangular street grid of ``nx`` x ``ny`` nodes.

    Node ids are ``"n<i>_<j>"``; horizontal edges ``"h<i>_<j>"`` join
    ``(i, j)`` to ``(i+1, j)`` and vertical edges ``"v<i>_<j>"`` join
    ``(i, j)`` to ``(i, j+1)``."""
    a = _as_int(nx, "nx", 1)
    b = _as_int(ny, "ny", 1)
    if a * b < 2:
        raise PublicVoiceError("a grid needs at least two nodes.")
    s = _as_positive(spacing, "spacing")
    ox, oy = _as_point(origin, "origin")
    nodes = {}
    for i in range(a):
        for j in range(b):
            nodes["n%d_%d" % (i, j)] = (ox + i * s, oy + j * s)
    edges = []
    for i in range(a):
        for j in range(b):
            if i + 1 < a:
                edges.append(("h%d_%d" % (i, j), "n%d_%d" % (i, j), "n%d_%d" % (i + 1, j)))
            if j + 1 < b:
                edges.append(("v%d_%d" % (i, j), "n%d_%d" % (i, j), "n%d_%d" % (i, j + 1)))
    return VoiceNetwork(nodes, edges, name=name)


@dataclass
class Projection:
    """Closest point on a segment."""
    distance: float
    t: float
    point: Point


def project_point_to_segment(p: Point, a: Point, b: Point) -> Projection:
    """Orthogonal projection of ``p`` onto segment ``ab`` clamped to it.

    ``t`` is the fractional position along ``ab`` (0 at ``a``, 1 at ``b``);
    an end-point projection returns the end point EXACTLY, so a point at a
    node has distance exactly 0."""
    px, py = _as_point(p, "p")
    ax, ay = _as_point(a, "a")
    bx, by = _as_point(b, "b")
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        raise PublicVoiceError("segment has zero length.")
    # ``t`` is a ratio of squared lengths: above ~1e154 they overflow to inf
    # (t = inf/inf = NaN poisoned the whole field) and below ~1e-162 they
    # underflow.  Dividing every difference by the same exact POWER OF TWO
    # makes the ratio scale-free and bit-identical to the unscaled one.
    e = math.frexp(max(abs(dx), abs(dy)))[1]
    sdx, sdy = math.ldexp(dx, -e), math.ldexp(dy, -e)
    try:
        spx, spy = math.ldexp(px - ax, -e), math.ldexp(py - ay, -e)
        t = (spx * sdx + spy * sdy) / (sdx * sdx + sdy * sdy)
    except OverflowError:
        t = math.nan
    if math.isnan(t) or math.isinf(t):
        # the point is > 1e300 segment-lengths away: it projects to an end
        da, db = math.hypot(px - ax, py - ay), math.hypot(px - bx, py - by)
        t = 0.0 if da <= db else 1.0
    if t <= 0.0:
        return Projection(math.hypot(px - ax, py - ay), 0.0, (ax, ay))
    if t >= 1.0:
        return Projection(math.hypot(px - bx, py - by), 1.0, (bx, by))
    qx, qy = ax + t * dx, ay + t * dy
    return Projection(math.hypot(px - qx, py - qy), t, (qx, qy))


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    """Euclidean distance from ``p`` to segment ``ab``."""
    return project_point_to_segment(p, a, b).distance


@dataclass
class Attachment:
    """A point attached to its nearest edge."""
    edge_id: str
    t: float
    offset_m: float
    distance_m: float
    snapped: Point

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able dict."""
        return {"edgeId": self.edge_id, "t": self.t, "offsetM": self.offset_m,
                "distanceM": self.distance_m,
                "snapped": [self.snapped[0], self.snapped[1]]}


def attach_point(network: VoiceNetwork, p: Point,
                 max_distance: Optional[float] = None) -> Optional[Attachment]:
    """Attach a point to the nearest edge (ties broken on edge id).

    ``offset_m`` is measured along the edge from its ``u`` end as
    ``t * length``.  Returns None when the nearest edge is farther than
    ``max_distance``."""
    if not isinstance(network, VoiceNetwork):
        raise PublicVoiceError("network must be a VoiceNetwork.")
    q = _as_point(p, "p")
    md = None if max_distance is None else _as_nonneg(max_distance, "max_distance")
    best: Optional[Tuple[float, str, Projection]] = None
    for eid in network.edge_ids:
        a, b = network.segment(eid)
        pr = project_point_to_segment(q, a, b)
        if best is None or (pr.distance, eid) < (best[0], best[1]):
            best = (pr.distance, eid, pr)
    assert best is not None
    d, eid, pr = best
    if md is not None and d > md:
        return None
    return Attachment(eid, pr.t, pr.t * network.length(eid), d, pr.point)


def attach_concerns(network: VoiceNetwork, concerns: Sequence[ConcernRecord],
                    max_distance: Optional[float] = None
                    ) -> Tuple[Dict[str, Attachment], List[str]]:
    """Attach every concern; returns (cid -> Attachment, unattached cids)."""
    cs = _check_concerns(concerns, allow_empty=True)
    att: Dict[str, Attachment] = {}
    lost: List[str] = []
    for c in cs:
        a = attach_point(network, c.point, max_distance)
        if a is None:
            lost.append(c.cid)
        else:
            att[c.cid] = a
    return att, lost


def dijkstra(network: VoiceNetwork, sources: Dict[str, float],
             cutoff: Optional[float] = None) -> Dict[str, float]:
    """Multi-source Dijkstra over edge lengths.

    ``sources`` maps node id -> initial distance (>= 0).  Returns node ->
    shortest distance for every node reached within ``cutoff`` (all nodes
    when None).  The heap is keyed on ``(distance, node id)`` so the visiting
    order -- and therefore every tie -- is deterministic."""
    if not isinstance(network, VoiceNetwork):
        raise PublicVoiceError("network must be a VoiceNetwork.")
    if not isinstance(sources, dict) or not sources:
        raise PublicVoiceError("sources must be a non-empty dict node -> distance.")
    lim = None if cutoff is None else _as_nonneg(cutoff, "cutoff")
    dist: Dict[str, float] = {}
    heap: List[Tuple[float, str]] = []
    for nid in _str_keys(sources, "sources"):
        network.node_xy(nid)
        d0 = _as_nonneg(sources[nid], "sources[%s]" % nid)
        heap.append((d0, nid))
    heapq.heapify(heap)
    while heap:
        d, n = heapq.heappop(heap)
        if n in dist:
            continue
        if lim is not None and d > lim:
            break
        dist[n] = d
        for (m, _, ln) in network.neighbours(n):
            if m not in dist:
                heapq.heappush(heap, (d + ln, m))
    return {k: dist[k] for k in sorted(dist)}


def network_distances_to_edges(network: VoiceNetwork, attachment: Attachment,
                               cutoff: Optional[float] = None,
                               include_snap: bool = False) -> Dict[str, float]:
    """Shortest along-network distance from an attached point to the NEAREST
    point of every edge.

    The attachment edge itself is at distance 0; any other edge is reached
    through one of its end nodes, so its distance is ``min(d(u), d(v))`` from
    a two-source Dijkstra seeded at ``t*L`` and ``(1-t)*L``.  With
    ``include_snap`` the off-network snap distance is added to every value.
    Edges beyond ``cutoff`` are omitted."""
    if not isinstance(network, VoiceNetwork):
        raise PublicVoiceError("network must be a VoiceNetwork.")
    if not isinstance(attachment, Attachment):
        raise PublicVoiceError("attachment must be an Attachment.")
    if not network.has_edge(attachment.edge_id):
        raise PublicVoiceError("attachment edge %r is not in the network."
                               % attachment.edge_id)
    lim = None if cutoff is None else _as_nonneg(cutoff, "cutoff")
    u, v = network.endpoints(attachment.edge_id)
    ln = network.length(attachment.edge_id)
    srcs: Dict[str, float] = {u: attachment.t * ln}
    dv = (1.0 - attachment.t) * ln
    srcs[v] = min(srcs.get(v, math.inf), dv)
    nd = dijkstra(network, srcs, cutoff=lim)
    extra = attachment.distance_m if include_snap else 0.0
    out: Dict[str, float] = {}
    for eid in network.edge_ids:
        if eid == attachment.edge_id:
            d = 0.0
        else:
            a, b = network.endpoints(eid)
            d = min(nd.get(a, math.inf), nd.get(b, math.inf))
        if math.isinf(d):
            continue
        d += extra
        if lim is not None and d > lim:
            continue
        out[eid] = d
    return out


def network_distance_between(network: VoiceNetwork, a: Attachment,
                             b: Attachment) -> float:
    """Shortest along-network distance between two attached points."""
    if not isinstance(network, VoiceNetwork):
        raise PublicVoiceError("network must be a VoiceNetwork.")
    for nm, at in (("a", a), ("b", b)):
        if not isinstance(at, Attachment) or not network.has_edge(at.edge_id):
            raise PublicVoiceError("%s must be an Attachment on this network." % nm)
    ua, va = network.endpoints(a.edge_id)
    la = network.length(a.edge_id)
    srcs = {ua: a.t * la}
    srcs[va] = min(srcs.get(va, math.inf), (1.0 - a.t) * la)
    nd = dijkstra(network, srcs)
    ub, vb = network.endpoints(b.edge_id)
    lb = network.length(b.edge_id)
    best = min(nd.get(ub, math.inf) + b.t * lb, nd.get(vb, math.inf) + (1.0 - b.t) * lb)
    if a.edge_id == b.edge_id:
        best = min(best, abs(a.t - b.t) * la)
    return best


# ===========================================================================
# 4. KERNELS -- profiles, 1-D and 2-D densities, moments, canonical bandwidth
# ===========================================================================
KERNELS: Tuple[str, ...] = ("gaussian", "epanechnikov", "triangular", "quartic",
                            "uniform", "exponential")

KERNEL_NORMALISATIONS: Tuple[str, ...] = ("peak", "density", "area")

# 1-D constants c1 so that  integral over R of c1 * k(u) du = 1.
_K1_CONST: Dict[str, float] = {
    "gaussian": 1.0 / math.sqrt(2.0 * math.pi),
    "epanechnikov": 0.75,
    "triangular": 1.0,
    "quartic": 15.0 / 16.0,
    "uniform": 0.5,
    "exponential": 0.5,
}

# 2-D radial constants c2 so that  integral over R^2 of c2 * k(|x|) dx = 1,
# i.e.  integral_0^inf 2 pi r c2 k(r) dr = 1.
_K2_CONST: Dict[str, float] = {
    "gaussian": 1.0 / (2.0 * math.pi),
    "epanechnikov": 2.0 / math.pi,
    "triangular": 3.0 / math.pi,
    "quartic": 3.0 / math.pi,
    "uniform": 1.0 / math.pi,
    "exponential": 1.0 / (2.0 * math.pi),
}

# Second moment mu2 = int u^2 K(u) du and roughness R = int K(u)^2 du of the
# 1-D density forms (closed forms; verified by quadrature in the suite).
_K1_MU2: Dict[str, float] = {
    "gaussian": 1.0, "epanechnikov": 0.2, "triangular": 1.0 / 6.0,
    "quartic": 1.0 / 7.0, "uniform": 1.0 / 3.0, "exponential": 2.0,
}
_K1_ROUGH: Dict[str, float] = {
    "gaussian": 1.0 / (2.0 * math.sqrt(math.pi)), "epanechnikov": 0.6,
    "triangular": 2.0 / 3.0, "quartic": 5.0 / 7.0, "uniform": 0.5,
    "exponential": 0.25,
}
# Per-coordinate second moment and roughness of the 2-D radial densities.
_K2_MU2: Dict[str, float] = {
    "gaussian": 1.0, "epanechnikov": 1.0 / 6.0, "triangular": 3.0 / 20.0,
    "quartic": 1.0 / 8.0, "uniform": 0.25, "exponential": 3.0,
}
_K2_ROUGH: Dict[str, float] = {
    "gaussian": 1.0 / (4.0 * math.pi), "epanechnikov": 4.0 / (3.0 * math.pi),
    "triangular": 3.0 / (2.0 * math.pi), "quartic": 9.0 / (5.0 * math.pi),
    "uniform": 1.0 / math.pi, "exponential": 1.0 / (8.0 * math.pi),
}


def _kernel_name(kernel: Any) -> str:
    """Validate a kernel name."""
    return _check_choice(kernel, KERNELS, "kernel")


def kernel_profile(kernel: str, u: float) -> float:
    """Unnormalised kernel profile ``k(u)`` with ``k(0) = 1`` exactly.

    gaussian ``exp(-u^2/2)``; epanechnikov ``1-u^2``; triangular ``1-|u|``;
    quartic ``(1-u^2)^2``; uniform ``1``; exponential ``exp(-|u|)``.  The
    four compact kernels vanish for ``|u| > 1`` (the uniform includes
    ``|u| = 1``)."""
    k = _kernel_name(kernel)
    a = abs(_as_float(u, "u"))
    return _profile(k, a)


def _profile(k: str, a: float) -> float:
    """Unchecked profile at ``a = |u| >= 0``."""
    if k == "gaussian":
        return math.exp(-0.5 * a * a)
    if k == "exponential":
        return math.exp(-a)
    if k == "uniform":
        return 1.0 if a <= 1.0 else 0.0
    if a >= 1.0:
        return 0.0
    if k == "epanechnikov":
        return 1.0 - a * a
    if k == "triangular":
        return 1.0 - a
    w = 1.0 - a * a
    return w * w


def kernel_density_1d(kernel: str, u: float) -> float:
    """1-D kernel density ``K(u) = c1 k(u)``; integrates to 1 over the real line."""
    k = _kernel_name(kernel)
    return _K1_CONST[k] * _profile(k, abs(_as_float(u, "u")))


def kernel_density_2d(kernel: str, r: float) -> float:
    """2-D radial kernel density ``K2(r) = c2 k(r)``; integrates to 1 over the plane."""
    k = _kernel_name(kernel)
    rv = _as_nonneg(r, "r")
    return _K2_CONST[k] * _profile(k, rv)


def kernel_weight(kernel: str, distance: float, bandwidth: float,
                  normalisation: str = "peak") -> float:
    """Kernel weight at ``distance`` for bandwidth ``h``.

    * ``"peak"`` (the concern-field default): ``k(d/h)`` -- dimensionless,
      in ``[0, 1]``, exactly 1 at ``d = 0``.  A single comment at distance 0
      therefore contributes EXACTLY its severity to the field.
    * ``"density"``: ``c1 k(d/h) / h`` -- a 1-D density in 1/m (integrates to
      1 over a signed line distance; the right form along a single street).
    * ``"area"``: ``c2 k(d/h) / h^2`` -- a 2-D density in 1/m^2 (integrates
      to 1 over the plane; the right form for a heat-map surface)."""
    k = _kernel_name(kernel)
    d = _as_nonneg(distance, "distance")
    h = _as_positive(bandwidth, "bandwidth")
    mode = _check_choice(normalisation, KERNEL_NORMALISATIONS, "normalisation")
    return _finite_density(_kernel_weight_unchecked(k, d, h, mode), "kernel_weight")


def _kernel_weight_unchecked(k: str, d: float, h: float, mode: str) -> float:
    """``kernel_weight`` without validation (hot loop).

    The area form divides by ``h`` twice rather than by ``h * h``: ``h * h``
    underflows to 0 below h ~ 1e-162 (a bare ZeroDivisionError) although
    ``c2 k / h^2`` is still representable down to h ~ 1e-154."""
    v = _profile(k, d / h)
    if mode == "peak":
        return v
    if mode == "density":
        return _K1_CONST[k] * v / h
    return _K2_CONST[k] * v / h / h


def _finite_sum(xs: Sequence[float], what: str) -> float:
    """``math.fsum`` that reports overflow (an inf term, or fsum's own
    intermediate OverflowError) as the module error, never a bare exception
    or a silent ``inf`` in a field."""
    try:
        v = math.fsum(xs)
    except OverflowError:
        v = math.inf
    if math.isinf(v) or math.isnan(v):
        raise PublicVoiceError("%s: a value overflows a float; rescale the weights, "
                               "severities or units." % what)
    return v


def _finite_density(v: float, what: str) -> float:
    """A density that overflowed (a bandwidth below ~1e-154 m) is unrepresentable."""
    if math.isinf(v) or math.isnan(v):
        raise PublicVoiceError("%s: the density overflows a float at this bandwidth; "
                               "rescale the units (e.g. metres -> kilometres)." % what)
    return v


def kernel_support(kernel: str, bandwidth: float) -> float:
    """Radius beyond which the kernel is exactly zero (``inf`` if unbounded)."""
    k = _kernel_name(kernel)
    h = _as_positive(bandwidth, "bandwidth")
    if k in ("gaussian", "exponential"):
        return math.inf
    return h


@dataclass
class KernelMoments:
    """Closed-form moments of a kernel's 1-D and 2-D density forms."""
    kernel: str
    mu2_1d: float
    roughness_1d: float
    mu2_2d: float
    roughness_2d: float
    canonical_1d: float
    canonical_2d: float
    efficiency_1d: float
    compact: bool


def kernel_moments(kernel: str) -> KernelMoments:
    """Second moment, roughness, canonical bandwidth and efficiency.

    The canonical bandwidth ``delta = (R / mu2^2)^(1/(d+4))`` converts a
    bandwidth between kernels at equal AMISE: ``h_K = h_G * delta_K /
    delta_G``.  ``efficiency_1d`` is relative to the Epanechnikov kernel
    (1.0 for Epanechnikov, 0.951 Gaussian, 0.986 triangular,
    0.994 quartic, 0.930 uniform)."""
    k = _kernel_name(kernel)
    c1 = (_K1_ROUGH[k] / (_K1_MU2[k] ** 2)) ** 0.2
    c2 = (_K2_ROUGH[k] / (_K2_MU2[k] ** 2)) ** (1.0 / 6.0)
    # Efficiency (Silverman 1986, table 3.1): the sample size the Epanechnikov
    # kernel needs for the same optimal AMISE, i.e. (C(Epan) / C(K))^(5/4) with
    # C(K) = (sqrt(mu2) R)^(4/5) -- so simply sqrt(mu2_E) R_E / (sqrt(mu2_K) R_K).
    ck = math.sqrt(_K1_MU2[k]) * _K1_ROUGH[k]
    ce = math.sqrt(_K1_MU2["epanechnikov"]) * _K1_ROUGH["epanechnikov"]
    return KernelMoments(k, _K1_MU2[k], _K1_ROUGH[k], _K2_MU2[k], _K2_ROUGH[k],
                         c1, c2, ce / ck, k not in ("gaussian", "exponential"))


# ===========================================================================
# 5. BANDWIDTH SELECTION -- Silverman, cross-validation, adaptive
# ===========================================================================
@dataclass
class BandwidthResult:
    """A selected bandwidth and the evidence for it."""
    bandwidth: float
    method: str
    kernel: str
    candidates: List[float]
    scores: List[float]


def silverman_bandwidth(values: Sequence[float], kernel: str = "gaussian",
                        robust: bool = True) -> float:
    """Silverman's rule of thumb in 1-D.

    ``h_G = 0.9 * A * n^(-1/5)`` with ``A = min(sd, IQR/1.34)`` (``robust``)
    or ``h_G = 1.06 * sd * n^(-1/5)`` (Scott's normal reference, not
    robust), converted to ``kernel`` through the canonical-bandwidth ratio.
    When the IQR is zero the standard deviation is used alone; a constant
    sample has no scale and raises."""
    xs = _as_floats(values, "values")
    k = _kernel_name(kernel)
    n = len(xs)
    if n < 2:
        raise PublicVoiceError("silverman_bandwidth needs at least two values.")
    sd = _sample_sd(xs)
    if sd == 0.0:
        raise PublicVoiceError("silverman_bandwidth: values are constant.")
    if _as_bool(robust, "robust"):
        s = sorted(xs)
        iqr = _quantile(s, 0.75) - _quantile(s, 0.25)
        a = min(sd, iqr / 1.34) if iqr > 0.0 else sd
        hg = 0.9 * a * n ** (-0.2)
    else:
        hg = 1.06 * sd * n ** (-0.2)
    return hg * kernel_moments(k).canonical_1d / kernel_moments("gaussian").canonical_1d


def spatial_silverman_bandwidth(points: Sequence[Point], kernel: str = "gaussian",
                                robust: bool = False) -> float:
    """2-D normal-reference bandwidth for an isotropic kernel.

    For ``d = 2`` the AMISE-optimal Gaussian bandwidth under a normal
    reference is ``h = sigma * n^(-1/6)`` where ``sigma = sqrt((s_x^2 +
    s_y^2) / 2)``; ``robust`` replaces each ``s`` by ``min(s, IQR/1.349)``.
    Converted to ``kernel`` through the 2-D canonical-bandwidth ratio."""
    pts = _as_points(points, "points")
    k = _kernel_name(kernel)
    _as_bool(robust, "robust")
    n = len(pts)
    if n < 2:
        raise PublicVoiceError("spatial_silverman_bandwidth needs at least two points.")
    sds = []
    for axis in (0, 1):
        vals = [p[axis] for p in pts]
        sd = _sample_sd(vals)
        if robust:
            s = sorted(vals)
            iqr = _quantile(s, 0.75) - _quantile(s, 0.25)
            if iqr > 0.0:
                sd = min(sd, iqr / 1.349)
        sds.append(sd)
    sigma = math.hypot(sds[0], sds[1]) / _SQRT2      # no overflow of s^2
    if sigma == 0.0:
        raise PublicVoiceError("spatial_silverman_bandwidth: all points coincide.")
    hg = sigma * n ** (-1.0 / 6.0)
    return hg * kernel_moments(k).canonical_2d / kernel_moments("gaussian").canonical_2d


def _pair_distances(pts: Sequence[Point]) -> Matrix:
    """Dense Euclidean distance matrix."""
    n = len(pts)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            v = math.hypot(xi - pts[j][0], yi - pts[j][1])
            d[i][j] = v
            d[j][i] = v
    return d


def _default_candidates(h0: float, n: int = 25) -> List[float]:
    """Geometric grid from h0/4 to 4*h0 (n points)."""
    return [h0 * 4.0 ** (-1.0 + 2.0 * i / (n - 1)) for i in range(n)]


def _loo_ll(d: Matrix, k: str, h: float) -> float:
    """Leave-one-out log-likelihood from a distance matrix, computed in
    BANDWIDTH UNITS: ``f_i h^2 = (1/(n-1)) sum_j c2 k(d_ij/h)`` is
    dimensionless, so nothing over- or underflows whatever the scale, and
    ``log f_i = log(f_i h^2) - 2 log h`` is added back once."""
    n = len(d)
    terms = []
    for i in range(n):
        f = math.fsum(_K2_CONST[k] * _profile(k, d[i][j] / h)
                      for j in range(n) if j != i) / (n - 1)
        if f <= 0.0:
            return -math.inf
        terms.append(math.log(f))
    return math.fsum(terms) - 2.0 * n * math.log(h)


def loo_log_likelihood(points: Sequence[Point], kernel: str, bandwidth: float) -> float:
    """Leave-one-out log-likelihood of a 2-D fixed-bandwidth KDE.

    Returns ``-inf`` when some point has zero leave-one-out density (a compact
    kernel with a bandwidth below the nearest-neighbour distance).  Evaluated
    in bandwidth units (``_loo_ll``), so it is finite at any coordinate scale."""
    pts = _as_points(points, "points")
    k = _kernel_name(kernel)
    h = _as_positive(bandwidth, "bandwidth")
    n = len(pts)
    if n < 2:
        raise PublicVoiceError("loo_log_likelihood needs at least two points.")
    return _loo_ll(_pair_distances(pts), k, h)


def _lscv_h2(d: Matrix, h: float) -> float:
    """``h^2 LSCV(h)`` -- the dimensionless Gaussian LSCV score.

    With ``u = d/h``: ``int f^2 h^2 = (1/n^2) sum_ij c2 exp(-u^2/4) / 2`` and
    the leave-one-out term ``h^2 f_{-i} = sum_{j != i} c2 exp(-u^2/2) / (n-1)``."""
    n = len(d)
    c2 = _K2_CONST["gaussian"]
    conv = math.fsum(0.5 * c2 * math.exp(-0.25 * (d[i][j] / h) ** 2)
                     for i in range(n) for j in range(n)) / (n * n)
    loo = math.fsum(c2 * math.exp(-0.5 * (d[i][j] / h) ** 2)
                    for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    return conv - 2.0 * loo


def lscv_score(points: Sequence[Point], bandwidth: float) -> float:
    """Least-squares cross-validation score of a 2-D GAUSSIAN KDE.

    ``LSCV(h) = int f_h^2 - (2/n) sum_i f_{h,-i}(x_i)`` with the closed form
    ``int f_h^2 = (1/n^2) sum_ij phi_{h sqrt 2}(x_i - x_j)``; minimising it
    minimises the integrated squared error up to a constant.  The score is
    1/m^2: it is computed dimensionless (``_lscv_h2``) and divided by ``h``
    twice, so it is exact wherever the result itself is representable."""
    pts = _as_points(points, "points")
    h = _as_positive(bandwidth, "bandwidth")
    n = len(pts)
    if n < 2:
        raise PublicVoiceError("lscv_score needs at least two points.")
    return _lscv_h2(_pair_distances(pts), h) / h / h


def cv_bandwidth(points: Sequence[Point], kernel: str = "gaussian",
                 candidates: Optional[Sequence[float]] = None,
                 method: str = "likelihood") -> BandwidthResult:
    """Cross-validated bandwidth over a candidate grid.

    ``"likelihood"`` maximises the leave-one-out log-likelihood (any kernel);
    ``"lscv"`` minimises least-squares CV (Gaussian only -- the closed-form
    convolution is what makes it exact).  The default grid is 25 geometric
    steps from 1/4 to 4 times the spatial Silverman bandwidth.  Ties go to
    the SMALLER bandwidth."""
    pts = _as_points(points, "points")
    k = _kernel_name(kernel)
    m = _check_choice(method, ("likelihood", "lscv"), "method")
    if m == "lscv" and k != "gaussian":
        raise PublicVoiceError("lscv is implemented for the gaussian kernel only.")
    if candidates is None:
        cands = _default_candidates(spatial_silverman_bandwidth(pts, k))
    else:
        cands = [_as_positive(c, "candidates[%d]" % i)
                 for i, c in enumerate(_as_floats(candidates, "candidates", allow_empty=True))]
        if not cands:
            raise PublicVoiceError("candidates must not be empty.")
    cands = sorted(set(cands))
    d = _pair_distances(pts)
    scores = []
    keys = []
    for h in cands:
        if m == "likelihood":
            v = _loo_ll(d, k, h)
            scores.append(v)
            keys.append(v)
        else:
            # Rank on h^2 LSCV(h) / h^2 against a REFERENCE bandwidth: the
            # ratio (h_ref / h)^2 is O(1) on any grid, so the comparison never
            # over/underflows even where LSCV itself (1/m^2) would.
            v = _lscv_h2(d, h)
            keys.append(v * (cands[0] / h) * (cands[0] / h))
            scores.append(v / h / h)
    if m == "likelihood":
        best_i = max(range(len(cands)), key=lambda i: (keys[i], -i))
        if math.isinf(keys[best_i]):
            raise PublicVoiceError("every candidate bandwidth leaves a point with zero "
                                   "leave-one-out density; widen the grid.")
    else:
        best_i = min(range(len(cands)), key=lambda i: (keys[i], i))
    return BandwidthResult(cands[best_i], m, k, cands, scores)


@dataclass
class AdaptiveBandwidths:
    """Per-point bandwidths from an adaptive rule."""
    method: str
    kernel: str
    pilot: float
    bandwidths: List[float]
    factors: List[float]
    geometric_mean_density: float


def _kde_h2(pts: Sequence[Point], k: str, h: float,
            targets: Sequence[Point]) -> Vector:
    """``h^2 f(t)`` -- the 2-D KDE in bandwidth units (dimensionless)."""
    n = len(pts)
    return [math.fsum(_K2_CONST[k] * _profile(k, math.hypot(tx - px, ty - py) / h)
                      for (px, py) in pts) / n for (tx, ty) in targets]


def kde_at_points(points: Sequence[Point], kernel: str, bandwidth: float,
                  at: Optional[Sequence[Point]] = None) -> Vector:
    """2-D fixed-bandwidth KDE (area normalisation) evaluated at ``at``
    (default: the sample points themselves, INCLUDING the self term).
    A density that overflows a float (bandwidth below ~1e-154) raises."""
    pts = _as_points(points, "points")
    k = _kernel_name(kernel)
    h = _as_positive(bandwidth, "bandwidth")
    targets = pts if at is None else _as_points(at, "at")
    return [_finite_density(v / h / h, "kde_at_points") for v in _kde_h2(pts, k, h, targets)]


def abramson_bandwidths(points: Sequence[Point], kernel: str = "gaussian",
                        pilot: Optional[float] = None,
                        alpha: float = 0.5) -> AdaptiveBandwidths:
    """Abramson's adaptive (square-root-law) bandwidths.

    A pilot KDE ``f~`` at fixed bandwidth ``h0`` gives local factors
    ``lambda_i = (f~(x_i) / g)^(-alpha)`` with ``g`` the geometric mean of
    ``f~(x_i)``, and ``h_i = h0 * lambda_i``.  Because ``g`` is the
    geometric mean, the geometric mean of the ``lambda_i`` is exactly 1 and
    the geometric mean of the ``h_i`` is exactly ``h0`` -- dense clusters get
    sharper kernels without changing the overall smoothing level.  The
    factors depend only on density RATIOS, so the pilot KDE is evaluated in
    bandwidth units (``h0^2 f~``, always in ``(0, c2]``): the factors are
    exact at any coordinate scale, and only ``geometric_mean_density``
    (1/m^2) can over/underflow."""
    pts = _as_points(points, "points")
    k = _kernel_name(kernel)
    a = _as_unit(alpha, "alpha")
    h0 = spatial_silverman_bandwidth(pts, k) if pilot is None else _as_positive(pilot, "pilot")
    f = _kde_h2(pts, k, h0, pts)
    logs = [math.log(v) for v in f]
    lg = math.fsum(logs) / len(logs)
    lam = [math.exp(-a * (lv - lg)) for lv in logs]
    g = math.exp(lg) / h0 / h0
    return AdaptiveBandwidths("abramson", k, h0, [h0 * v for v in lam], lam, g)


def knn_bandwidths(points: Sequence[Point], k: int = 5,
                   kernel: str = "gaussian") -> AdaptiveBandwidths:
    """Balloon-style adaptive bandwidths: distance to the k-th nearest neighbour.

    A point whose k-th neighbour coincides with it (duplicates) falls back to
    its nearest POSITIVE neighbour distance; if every point coincides the
    sample has no scale and raises."""
    pts = _as_points(points, "points")
    kn = _as_int(k, "k", 1)
    kname = _kernel_name(kernel)
    n = len(pts)
    if kn > n - 1:
        raise PublicVoiceError("k must be at most n - 1 = %d (got %d)." % (n - 1, kn))
    d = _pair_distances(pts)
    hs = []
    for i in range(n):
        row = sorted(d[i][j] for j in range(n) if j != i)
        h = row[kn - 1]
        if h == 0.0:
            pos = [v for v in row if v > 0.0]
            if not pos:
                raise PublicVoiceError("knn_bandwidths: all points coincide.")
            h = pos[0]
        hs.append(h)
    gm = math.exp(math.fsum(math.log(h) for h in hs) / n)
    return AdaptiveBandwidths("knn", kname, gm, hs, [h / gm for h in hs], float("nan"))


# ===========================================================================
# 6. THE CONCERN FIELD -- s_e^j = sum_i sev_i K(dist(e, loc_i)) 1[theme_i = j]
# ===========================================================================
DISTANCE_MODES: Tuple[str, ...] = ("euclidean", "network")
EDGE_REFERENCES: Tuple[str, ...] = ("segment", "midpoint")


@dataclass
class ConcernField:
    """Concern intensity per theme per edge.

    ``values[theme][k]`` is ``s_e^j`` for ``e = edge_ids[k]``.  Every theme in
    THEMES is present (all-zero when no comment carries it) so downstream
    code never has to special-case a missing theme."""
    kernel: str
    bandwidth: Optional[float]
    normalisation: str
    distance: str
    edge_reference: str
    edge_ids: List[str]
    values: Dict[str, Vector]
    n_concerns: int
    unattached: List[str]

    def theme_vector(self, theme: str) -> Vector:
        """``s_e^j`` over edges for one theme."""
        th = _check_choice(theme, THEMES, "theme")
        return list(self.values[th])

    def edge_value(self, edge_id: str, theme: Optional[str] = None) -> float:
        """Field at one edge for one theme, or summed over themes."""
        if not isinstance(edge_id, str) or edge_id not in self._index():
            raise PublicVoiceError("unknown edge %r." % (edge_id,))
        k = self._index()[edge_id]
        if theme is None:
            return math.fsum(self.values[th][k] for th in THEMES)
        return self.values[_check_choice(theme, THEMES, "theme")][k]

    def total_vector(self) -> Vector:
        """Field summed over every theme, per edge."""
        return [math.fsum(self.values[th][k] for th in THEMES)
                for k in range(len(self.edge_ids))]

    def theme_mass(self) -> Dict[str, float]:
        """Total field mass per theme."""
        return {th: math.fsum(self.values[th]) for th in THEMES}

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        """theme -> edge id -> value."""
        return {th: {e: self.values[th][k] for k, e in enumerate(self.edge_ids)}
                for th in THEMES}

    def _index(self) -> Dict[str, int]:
        """Edge id -> position."""
        return {e: k for k, e in enumerate(self.edge_ids)}


def _check_field(field: Any, network: Optional["VoiceNetwork"] = None) -> "ConcernField":
    """A ConcernField whose ``values`` hold every theme as a finite vector over
    its ``edge_ids`` (a hand-built or deserialised field is checked here, not
    trusted); with ``network``, its edges must be exactly that network's."""
    if not isinstance(field, ConcernField):
        raise PublicVoiceError("field must be a ConcernField.")
    if not isinstance(field.edge_ids, list) or not isinstance(field.values, dict):
        raise PublicVoiceError("field.edge_ids must be a list and field.values a dict.")
    for e in field.edge_ids:
        _as_id(e, "field edge id")
    for th in THEMES:
        if th not in field.values:
            raise PublicVoiceError("field.values has no theme %r." % th)
        v = _as_floats(field.values[th], "field.values[%s]" % th, allow_empty=True)
        if len(v) != len(field.edge_ids):
            raise PublicVoiceError("field.values[%s] has %d entries for %d edges."
                                   % (th, len(v), len(field.edge_ids)))
    if network is not None and field.edge_ids != network.edge_ids:
        raise PublicVoiceError("field was computed over different edges than this network.")
    return field


def _concern_weights(cs: List[ConcernRecord], weights: Optional[Dict[str, float]],
                     use_counts: bool) -> Dict[str, float]:
    """Mass of each concern: severity x (count) x (external weight)."""
    out: Dict[str, float] = {}
    if weights is not None and not isinstance(weights, dict):
        raise PublicVoiceError("weights must be a dict cid -> weight.")
    for c in cs:
        w = c.severity * (c.count if use_counts else 1.0)
        if weights is not None:
            if c.cid not in weights:
                raise PublicVoiceError("weights has no entry for concern %r." % c.cid)
            w *= _as_nonneg(weights[c.cid], "weights[%s]" % c.cid)
        out[c.cid] = w
    return out


def _resolve_bandwidths(cs: List[ConcernRecord], bandwidth: Optional[float],
                        bandwidths: Optional[Dict[str, float]],
                        kernel: str) -> Tuple[Optional[float], Dict[str, float]]:
    """Per-concern bandwidths from a global value, a dict, or Silverman."""
    if bandwidths is not None:
        if not isinstance(bandwidths, dict):
            raise PublicVoiceError("bandwidths must be a dict cid -> bandwidth.")
        if bandwidth is not None:
            raise PublicVoiceError("pass bandwidth OR bandwidths, not both.")
        out = {}
        for c in cs:
            if c.cid not in bandwidths:
                raise PublicVoiceError("bandwidths has no entry for concern %r." % c.cid)
            out[c.cid] = _as_positive(bandwidths[c.cid], "bandwidths[%s]" % c.cid)
        return None, out
    if bandwidth is None:
        h = spatial_silverman_bandwidth([c.point for c in cs], kernel)
    else:
        h = _as_positive(bandwidth, "bandwidth")
    return h, {c.cid: h for c in cs}


def concern_field(network: VoiceNetwork, concerns: Sequence[ConcernRecord], *,
                  kernel: str = "gaussian", bandwidth: Optional[float] = None,
                  bandwidths: Optional[Dict[str, float]] = None,
                  normalisation: str = "peak", distance: str = "euclidean",
                  edge_reference: str = "segment", weights: Optional[Dict[str, float]] = None,
                  use_counts: bool = True, cutoff: Optional[float] = None,
                  max_attach_distance: Optional[float] = None,
                  include_snap: bool = False) -> ConcernField:
    """The paper's concern field over network edges.

    ``s_e^j = sum_i w_i * K_h(dist(e, loc_i)) * 1[theme_i = j]`` with
    ``w_i = severity_i * count_i * weights[i]`` (``count`` only when
    ``use_counts``; ``weights`` e.g. post-stratification weights).

    * ``distance="euclidean"``: ``dist`` is the straight-line distance from
      the comment to the edge's ``segment`` (closest point) or ``midpoint``.
    * ``distance="network"``: the comment is attached to its nearest edge and
      ``dist`` is the shortest along-network distance to the edge's nearest
      point (own Dijkstra); comments farther than ``max_attach_distance``
      from every edge are listed in ``unattached`` and contribute nothing.

    Kernel normalisation convention (``kernel_weight``): the default
    ``"peak"`` makes ``K_h(0) = 1``, so a single comment at distance 0
    contributes EXACTLY its severity.  The field is exactly linear in each
    severity and additive over comments: each value is the correctly rounded
    ``math.fsum`` of the per-comment terms.  ``bandwidth=None`` uses the
    spatial Silverman rule on the comment locations; ``bandwidths`` gives a
    per-comment (adaptive) bandwidth.  ``cutoff`` drops contributions beyond
    that distance (exact for the compact kernels when ``cutoff >= h``)."""
    if not isinstance(network, VoiceNetwork):
        raise PublicVoiceError("network must be a VoiceNetwork.")
    cs = _check_concerns(concerns)
    k = _kernel_name(kernel)
    mode = _check_choice(normalisation, KERNEL_NORMALISATIONS, "normalisation")
    dmode = _check_choice(distance, DISTANCE_MODES, "distance")
    eref = _check_choice(edge_reference, EDGE_REFERENCES, "edge_reference")
    if not isinstance(use_counts, bool) or not isinstance(include_snap, bool):
        raise PublicVoiceError("use_counts and include_snap must be bools.")
    # Options that only one distance mode reads are rejected in the other
    # rather than silently ignored.
    if dmode == "network" and eref != "segment":
        raise PublicVoiceError("edge_reference=%r applies to euclidean distance only; "
                               "network distance is always to the edge's nearest point."
                               % eref)
    if dmode == "euclidean" and (include_snap or max_attach_distance is not None):
        raise PublicVoiceError("include_snap and max_attach_distance apply to "
                               "distance='network' only.")
    lim = None if cutoff is None else _as_positive(cutoff, "cutoff")
    w = _concern_weights(cs, weights, use_counts)
    hglob, hs = _resolve_bandwidths(cs, bandwidth, bandwidths, k)
    eids = network.edge_ids
    terms: Dict[str, List[List[float]]] = {th: [[] for _ in eids] for th in THEMES}
    unattached: List[str] = []
    if dmode == "network":
        md = None if max_attach_distance is None else _as_nonneg(max_attach_distance,
                                                                 "max_attach_distance")
    for c in cs:
        h = hs[c.cid]
        wc = w[c.cid]
        if dmode == "euclidean":
            dists: Dict[str, float] = {}
            for e in eids:
                if eref == "segment":
                    a, b = network.segment(e)
                    dists[e] = project_point_to_segment(c.point, a, b).distance
                else:
                    mx, my = network.midpoint(e)
                    dists[e] = math.hypot(c.x - mx, c.y - my)
        else:
            att = attach_point(network, c.point, md)
            if att is None:
                unattached.append(c.cid)
                continue
            dists = network_distances_to_edges(network, att, cutoff=lim,
                                               include_snap=include_snap)
        bucket = terms[c.theme]
        for kk, e in enumerate(eids):
            if e not in dists:
                continue
            d = dists[e]
            if lim is not None and d > lim:
                continue
            kv = _kernel_weight_unchecked(k, d, h, mode)
            if kv != 0.0:
                bucket[kk].append(wc * kv)
    values = {th: [_finite_sum(terms[th][kk], "concern_field") for kk in range(len(eids))]
              for th in THEMES}
    return ConcernField(k, hglob, mode, dmode, eref, list(eids), values, len(cs), unattached)


_FIELD_KWARGS: Tuple[str, ...] = (
    "kernel", "bandwidth", "bandwidths", "normalisation", "distance", "edge_reference",
    "weights", "use_counts", "cutoff", "max_attach_distance", "include_snap")


def concern_contributions(network: VoiceNetwork, concerns: Sequence[ConcernRecord],
                          edge_id: str, top: int = 10, **field_kwargs: Any
                          ) -> List[Tuple[str, str, float]]:
    """Which comments drive the field on one edge: ``(cid, theme, contribution)``
    sorted by contribution descending then id, at most ``top`` of them.  The
    contributions sum (over all comments) to ``field.edge_value(edge_id)``."""
    if not isinstance(network, VoiceNetwork) or not network.has_edge(edge_id):
        raise PublicVoiceError("edge_id must be an edge of the network.")
    cs = _check_concerns(concerns)
    t = _as_int(top, "top", 1)
    for key in sorted(field_kwargs):
        if key not in _FIELD_KWARGS:
            raise PublicVoiceError("concern_contributions: unknown field option %r." % key)
    h_kw = field_kwargs.get("bandwidth")
    if h_kw is None and field_kwargs.get("bandwidths") is None:
        field_kwargs = dict(field_kwargs)
        field_kwargs["bandwidth"] = spatial_silverman_bandwidth(
            [c.point for c in cs], _kernel_name(field_kwargs.get("kernel", "gaussian")))
    # Validate every option (weights / bandwidths complete, enums, ...) on the
    # full set FIRST, so the per-comment loop below cannot hit a bare KeyError.
    concern_field(network, cs, **field_kwargs)
    out = []
    for c in cs:
        sub = dict(field_kwargs)
        if sub.get("bandwidths") is not None:
            sub["bandwidths"] = {c.cid: sub["bandwidths"][c.cid]}
        if sub.get("weights") is not None:
            sub["weights"] = {c.cid: sub["weights"][c.cid]}
        f = concern_field(network, [c], **sub)
        v = f.edge_value(edge_id)
        if v != 0.0:
            out.append((c.cid, c.theme, v))
    out.sort(key=lambda r: (-r[2], r[0]))
    return out[:t]


@dataclass
class ConcernSurface:
    """A raster heat surface of concern intensity (cell-centre samples)."""
    x0: float
    y0: float
    cell: float
    nx: int
    ny: int
    grid: Matrix
    kernel: str
    bandwidth: float
    normalisation: str

    def cell_centre(self, i: int, j: int) -> Point:
        """Centre of column ``i``, row ``j``."""
        return (self.x0 + (i + 0.5) * self.cell, self.y0 + (j + 0.5) * self.cell)

    def integral(self) -> float:
        """Midpoint-rule integral of the surface over its extent."""
        return math.fsum(v for row in self.grid for v in row) * self.cell * self.cell


def concern_surface(concerns: Sequence[ConcernRecord], x0: float, y0: float,
                    cell: float, nx: int, ny: int, *, kernel: str = "gaussian",
                    bandwidth: Optional[float] = None, theme: Optional[str] = None,
                    normalisation: str = "area", use_counts: bool = True,
                    weights: Optional[Dict[str, float]] = None) -> ConcernSurface:
    """Heat-map raster: kernel intensity sampled at cell centres.

    ``grid[j][i]`` is the value at column ``i`` (x) and row ``j`` (y).  With
    the default ``"area"`` normalisation and unit weights the surface of one
    comment integrates to 1 over the plane, so the midpoint-rule integral of
    a window that contains the kernel's support equals the total mass."""
    cs = _check_concerns(concerns)
    if theme is not None:
        th = _check_choice(theme, THEMES, "theme")
        cs = [c for c in cs if c.theme == th]
        if not cs:
            raise PublicVoiceError("no concern carries theme %r." % th)
    k = _kernel_name(kernel)
    mode = _check_choice(normalisation, KERNEL_NORMALISATIONS, "normalisation")
    ox = _as_float(x0, "x0")
    oy = _as_float(y0, "y0")
    c_ = _as_positive(cell, "cell")
    nxx = _as_int(nx, "nx", 1)
    nyy = _as_int(ny, "ny", 1)
    if not isinstance(use_counts, bool):
        raise PublicVoiceError("use_counts must be a bool.")
    if bandwidth is None:
        if len(cs) < 2:
            raise PublicVoiceError("a single comment needs an explicit bandwidth.")
        h = spatial_silverman_bandwidth([c.point for c in cs], k)
    else:
        h = _as_positive(bandwidth, "bandwidth")
    w = _concern_weights(cs, weights, use_counts)
    grid: Matrix = []
    for j in range(nyy):
        cy = oy + (j + 0.5) * c_
        row = []
        for i in range(nxx):
            cx = ox + (i + 0.5) * c_
            row.append(_finite_sum([w[c.cid] * _kernel_weight_unchecked(
                k, math.hypot(cx - c.x, cy - c.y), h, mode) for c in cs], "concern_surface"))
        grid.append(row)
    return ConcernSurface(ox, oy, c_, nxx, nyy, grid, k, h, mode)


# ===========================================================================
# 7. THEME-METRIC PAIRING & THE PUBLIC-VOICE OBJECTIVE J_voice
# ===========================================================================
NORMALISATION_MODES: Tuple[str, ...] = ("edge", "network")


@dataclass(frozen=True)
class ThemeMetricPair:
    """Which network metric answers a theme, and which way is better.

    ``metric=None`` marks a theme with NO measurable counterpart: it is drawn
    on the heat map but excluded from ``J_voice`` (the paper's explicit
    rule).  ``direction="lower"`` means a smaller metric is better (delay,
    NO2, noise); ``"higher"`` means a larger metric is better
    (accessibility) and the normalised burden is the inverse ratio."""
    theme: str
    metric: Optional[str]
    direction: str = "lower"

    @property
    def measurable(self) -> bool:
        """True when a metric is paired."""
        return self.metric is not None


def default_pairing() -> Dict[str, ThemeMetricPair]:
    """The taxonomy's default theme -> metric pairing (parking and other unmeasurable)."""
    return {th: ThemeMetricPair(th, THEME_TAXONOMY[th].default_metric,
                                THEME_TAXONOMY[th].direction) for th in THEMES}


def make_pairing(overrides: Optional[Dict[str, Optional[str]]] = None,
                 directions: Optional[Dict[str, str]] = None
                 ) -> Dict[str, ThemeMetricPair]:
    """Default pairing with per-theme metric / direction overrides.

    ``overrides[theme] = None`` makes a theme unmeasurable."""
    base = default_pairing()
    ov = {} if overrides is None else overrides
    dr = {} if directions is None else directions
    if not isinstance(ov, dict) or not isinstance(dr, dict):
        raise PublicVoiceError("overrides and directions must be dicts.")
    for th in _str_keys(ov, "overrides"):
        _check_choice(th, THEMES, "theme")
        m = ov[th]
        if m is not None:
            _as_id(m, "metric for %s" % th)
    for th in _str_keys(dr, "directions"):
        _check_choice(th, THEMES, "theme")
        _check_choice(dr[th], ("lower", "higher"), "direction")
    out = {}
    for th in THEMES:
        m = ov[th] if th in ov else base[th].metric
        d = dr[th] if th in dr else base[th].direction
        out[th] = ThemeMetricPair(th, m, d)
    return out


def _check_pairing(pairing: Optional[Dict[str, ThemeMetricPair]]
                   ) -> Dict[str, ThemeMetricPair]:
    """Validate a pairing dict (every theme present)."""
    if pairing is None:
        return default_pairing()
    if not isinstance(pairing, dict):
        raise PublicVoiceError("pairing must be a dict theme -> ThemeMetricPair.")
    for th in THEMES:
        p = pairing.get(th)
        if not isinstance(p, ThemeMetricPair) or p.theme != th:
            raise PublicVoiceError("pairing has no valid entry for theme %r." % th)
        _check_choice(p.direction, ("lower", "higher"), "direction")
    return pairing


def _metric_vector(metrics: Any, metric: str, edge_ids: List[str],
                   name: str) -> Vector:
    """Extract one metric over the field's edges."""
    if not isinstance(metrics, dict):
        raise PublicVoiceError("%s must be a dict metric -> {edge id -> value}." % name)
    if metric not in metrics:
        raise PublicVoiceError("%s has no metric %r." % (name, metric))
    m = metrics[metric]
    if not isinstance(m, dict):
        raise PublicVoiceError("%s[%r] must be a dict edge id -> value." % (name, metric))
    out = []
    for e in edge_ids:
        if e not in m:
            raise PublicVoiceError("%s[%r] has no value for edge %r." % (name, metric, e))
        out.append(_as_nonneg(m[e], "%s[%s][%s]" % (name, metric, e)))
    return out


def normalise_metric(values: Sequence[float], baseline: Sequence[float],
                     direction: str = "lower", mode: str = "edge") -> Vector:
    """Baseline-normalised burden ``m_e`` so that the baseline maps to a known
    reference.

    ``mode="edge"`` (per-edge ratio): lower-is-better ``m_e = x_e / b_e``;
    an edge whose baseline is exactly zero uses ``1 + x_e / b_bar`` (``b_bar``
    the mean positive baseline), so the baseline is EXACTLY 1 on every edge.
    Higher-is-better ``m_e = b_e / x_e`` (both must be > 0).
    ``mode="network"``: lower ``m_e = x_e / b_bar``, higher ``m_e = b_bar /
    x_e`` with ``b_bar`` the network mean baseline.  Ratios are scale-free,
    so rescaling a metric's units leaves every ``m_e`` unchanged."""
    xs = _as_floats(values, "values")
    bs = _as_floats(baseline, "baseline")
    d = _check_choice(direction, ("lower", "higher"), "direction")
    md = _check_choice(mode, NORMALISATION_MODES, "mode")
    if len(xs) != len(bs):
        raise PublicVoiceError("values and baseline differ in length.")
    if any(v < 0.0 for v in xs) or any(v < 0.0 for v in bs):
        raise PublicVoiceError("metric values must be non-negative.")
    pos = [b for b in bs if b > 0.0]
    if d == "higher":
        if any(v == 0.0 for v in xs) or not pos:
            raise PublicVoiceError("a higher-is-better metric must be strictly positive.")
        if md == "edge":
            if len(pos) != len(bs):
                raise PublicVoiceError("a higher-is-better metric needs a positive baseline "
                                       "on every edge in edge mode.")
            out = [b / v for v, b in zip(xs, bs)]
        else:
            bbar = _safe_mean(bs)
            out = [bbar / v for v in xs]
    elif not pos:
        if any(v > 0.0 for v in xs):
            raise PublicVoiceError("baseline is zero everywhere; cannot normalise a "
                                   "non-zero metric.")
        out = [1.0 if md == "edge" else 0.0 for _ in xs]
    elif md == "edge":
        bpos = _safe_mean(pos)
        out = [v / b if b > 0.0 else 1.0 + v / bpos for v, b in zip(xs, bs)]
    else:
        bbar = _safe_mean(bs)
        out = [v / bbar for v in xs]
    # a ratio beyond the float range (x / b > ~1e308) is no burden at all:
    # it used to come back as inf, and J_voice = inf, improvement = -inf
    if any(math.isinf(v) for v in out):
        raise PublicVoiceError("a normalised metric overflows a float (ratio > ~1e308); "
                               "check the metric's units against its baseline.")
    return out


@dataclass
class VoiceObjective:
    """``J_voice`` and its decomposition."""
    total: float
    by_theme: Dict[str, float]
    included_themes: List[str]
    excluded_themes: List[str]
    excluded_mass: Dict[str, float]
    normalisation: str


def voice_objective(field: ConcernField, metrics: Dict[str, Dict[str, float]],
                    baseline_metrics: Dict[str, Dict[str, float]],
                    pairing: Optional[Dict[str, ThemeMetricPair]] = None,
                    normalisation: str = "edge") -> VoiceObjective:
    """``J_voice(x) = sum_j sum_e s_e^j * m_e^j(x)`` over MEASURABLE themes.

    ``m_e^j`` is the baseline-normalised burden of theme ``j``'s paired
    metric (``normalise_metric``).  A theme with no measurable counterpart is
    EXCLUDED from the sum -- its field mass is reported in ``excluded_mass``
    so the heat map can still show it.  A measurable theme with zero field
    mass contributes nothing and needs no metric."""
    _check_field(field)
    pr = _check_pairing(pairing)
    md = _check_choice(normalisation, NORMALISATION_MODES, "normalisation")
    by: Dict[str, float] = {}
    inc: List[str] = []
    exc: List[str] = []
    exc_mass: Dict[str, float] = {}
    for th in THEMES:
        s = field.values[th]
        p = pr[th]
        if not p.measurable:
            exc.append(th)
            exc_mass[th] = _finite_sum(s, "voice_objective")
            continue
        if all(v == 0.0 for v in s):
            continue
        assert p.metric is not None
        xv = _metric_vector(metrics, p.metric, field.edge_ids, "metrics")
        bv = _metric_vector(baseline_metrics, p.metric, field.edge_ids, "baseline_metrics")
        m = normalise_metric(xv, bv, p.direction, md)
        by[th] = _finite_sum([sv * mv for sv, mv in zip(s, m)], "voice_objective")
        inc.append(th)
    total = _finite_sum([by[th] for th in inc], "voice_objective")
    return VoiceObjective(total, by, inc, exc, exc_mass, md)


def voice_improvement(field: ConcernField, metrics: Dict[str, Dict[str, float]],
                      baseline_metrics: Dict[str, Dict[str, float]],
                      pairing: Optional[Dict[str, ThemeMetricPair]] = None,
                      normalisation: str = "edge") -> float:
    """Public-voice improvement ``1 - J_voice(x) / J_voice(baseline)``.

    Positive is better.  The baseline against itself is EXACTLY 0 (the two
    totals are the same floating-point computation).  When both
    ``J_voice(baseline)`` and ``J_voice(x)`` are 0 (no measurable concern
    anywhere) the improvement is defined as 0.  When ``J_voice(baseline) = 0``
    but ``J_voice(x) > 0`` -- possible in ``"network"`` mode when every
    concern sits on edges whose baseline metric is zero -- the design made
    things strictly worse from nothing, the ratio is ``-inf``, and the call
    raises rather than report "no change"."""
    jx = voice_objective(field, metrics, baseline_metrics, pairing, normalisation).total
    jb = voice_objective(field, baseline_metrics, baseline_metrics, pairing,
                         normalisation).total
    if jb == 0.0:
        if jx == 0.0:
            return 0.0
        raise PublicVoiceError("J_voice(baseline) is 0 but J_voice(x) = %r > 0: the "
                               "improvement is unbounded (-inf); compare the totals "
                               "directly or use normalisation='edge'." % jx)
    return 1.0 - jx / jb


@dataclass
class HeatmapCell:
    """One edge x theme cell of the concern heat map."""
    edge_id: str
    theme: str
    value: float
    in_objective: bool
    metric: Optional[str]


def heatmap_layers(field: ConcernField,
                   pairing: Optional[Dict[str, ThemeMetricPair]] = None,
                   include_zero: bool = False) -> List[HeatmapCell]:
    """Every (edge, theme) value for the heat map, flagged ``in_objective``.

    Unmeasurable themes ARE shown (with ``in_objective=False``) -- the
    exclusion rule applies to ``J_voice``, not to what the public sees."""
    _check_field(field)
    pr = _check_pairing(pairing)
    _as_bool(include_zero, "include_zero")
    out = []
    for th in THEMES:
        for k, e in enumerate(field.edge_ids):
            v = field.values[th][k]
            if v == 0.0 and not include_zero:
                continue
            out.append(HeatmapCell(e, th, v, pr[th].measurable, pr[th].metric))
    return out


# ===========================================================================
# 8. NEAR-DUPLICATE DETECTION -- canonical form, shingles, MinHash+LSH, SimHash
# ===========================================================================
_FNV64_OFFSET = 0xCBF29CE484222325
_FNV64_PRIME = 0x100000001B3
_FNV32_OFFSET = 0x811C9DC5
_FNV32_PRIME = 0x01000193
_MASK64 = (1 << 64) - 1
_MERSENNE61 = (1 << 61) - 1

_ABBREVIATIONS: Dict[str, str] = {
    "st": "street", "rd": "road", "ave": "avenue", "ln": "lane", "dr": "drive",
    "pl": "place", "sq": "square", "cl": "close", "jct": "junction",
    "xing": "crossing", "ped": "pedestrian", "peds": "pedestrians",
    "hgvs": "hgv", "lorrys": "lorries", "pls": "please", "plz": "please",
    "u": "you", "r": "are", "thru": "through", "nr": "near", "opp": "opposite",
}

_STOPWORDS: Tuple[str, ...] = (
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has",
    "have", "i", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our",
    "so", "that", "the", "their", "there", "this", "to", "too", "very", "was",
    "we", "were", "with", "you",
)


def fnv1a_64(data: bytes) -> int:
    """64-bit FNV-1a hash (``fnv1a_64(b"") == 0xcbf29ce484222325``)."""
    if not isinstance(data, (bytes, bytearray)):
        raise PublicVoiceError("fnv1a_64 needs bytes.")
    h = _FNV64_OFFSET
    for byte in bytes(data):
        h ^= byte
        h = (h * _FNV64_PRIME) & _MASK64
    return h


def fnv1a_32(data: bytes) -> int:
    """32-bit FNV-1a hash (``fnv1a_32(b"") == 0x811c9dc5``)."""
    if not isinstance(data, (bytes, bytearray)):
        raise PublicVoiceError("fnv1a_32 needs bytes.")
    h = _FNV32_OFFSET
    for byte in bytes(data):
        h ^= byte
        h = (h * _FNV32_PRIME) & 0xFFFFFFFF
    return h


def _mix64(z: int) -> int:
    """SplitMix64 finaliser: a bijective avalanche mix of a 64-bit integer.

    FNV-1a of SHORT inputs is nearly affine in the last byte (single-character
    shingles hash to an arithmetic progression), and a linear permutation
    ``(a x + b) mod p`` of an arithmetic progression is far from min-wise
    independent -- MinHash over such keys was biased by ~7 standard errors.
    Mixing first restores the unbiasedness the estimator relies on."""
    z = (z + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def stable_hash64(text: str) -> int:
    """FNV-1a 64 of the UTF-8 bytes -- independent of ``PYTHONHASHSEED``."""
    return fnv1a_64(_as_text(text, "text").encode("utf-8"))


def canonical_form(text: str, expand_abbreviations: bool = True,
                   drop_stopwords: bool = False) -> str:
    """Canonical text for duplicate collapse.

    Unicode-normalised (NFKC), case-folded, punctuation replaced by spaces,
    redaction tokens kept as words, abbreviations expanded (``st`` ->
    ``street``), optional stop-word removal, whitespace collapsed.  Two
    comments differing only in case, punctuation, spacing or those
    abbreviations map to the same string."""
    _as_bool(expand_abbreviations, "expand_abbreviations")
    _as_bool(drop_stopwords, "drop_stopwords")
    t = unicodedata.normalize("NFKC", _as_text(text, "text")).casefold()
    t = re.sub(r"[^\w\s]", " ", t)
    t = t.replace("_", " ")
    words = t.split()
    if expand_abbreviations:
        words = [_ABBREVIATIONS.get(w, w) for w in words]
    if drop_stopwords:
        stop = set(_STOPWORDS)
        words = [w for w in words if w not in stop]
    return " ".join(words)


def char_shingles(text: str, k: int = 5, canonical: bool = True) -> List[str]:
    """Sorted unique character k-shingles (of the canonical form by default).

    A text shorter than ``k`` yields itself as its single shingle (never an
    empty set, so two short identical texts still match)."""
    kk = _as_int(k, "k", 1)
    t = canonical_form(text) if _as_bool(canonical, "canonical") else _as_text(text, "text")
    if not t:
        return []
    if len(t) < kk:
        return [t]
    return sorted(set(t[i:i + kk] for i in range(len(t) - kk + 1)))


def word_shingles(text: str, w: int = 2, canonical: bool = True) -> List[str]:
    """Sorted unique word w-shingles (word n-grams joined by a space)."""
    ww = _as_int(w, "w", 1)
    t = canonical_form(text) if _as_bool(canonical, "canonical") else _as_text(text, "text")
    words = t.split()
    if not words:
        return []
    if len(words) < ww:
        return [" ".join(words)]
    return sorted(set(" ".join(words[i:i + ww]) for i in range(len(words) - ww + 1)))


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """Exact Jaccard similarity of two shingle sets (1.0 for two empty sets)."""
    sa = set(_as_text(v, "shingle") for v in _as_seq(a, "a"))
    sb = set(_as_text(v, "shingle") for v in _as_seq(b, "b"))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


class MinHasher:
    """MinHash signatures with universal hashing modulo the Mersenne prime 2^61-1.

    Shingles are hashed with FNV-1a 64 (never the salted builtin), passed
    through the SplitMix64 finaliser (``_mix64``), reduced modulo ``p``, and permuted by ``(a x + b) mod p`` with ``(a, b)`` drawn
    from ``random.Random(seed)``.  The fraction of agreeing coordinates is an
    unbiased estimator of the Jaccard similarity of the shingle sets."""

    def __init__(self, num_perm: int = 128, seed: int = 1, shingle: str = "char",
                 k: int = 5) -> None:
        self.num_perm = _as_int(num_perm, "num_perm", 1)
        self.seed = _as_seed(seed)
        self.shingle = _check_choice(shingle, ("char", "word"), "shingle")
        self.k = _as_int(k, "k", 1)
        rng = random.Random(self.seed)
        self._a = [rng.randrange(1, _MERSENNE61) for _ in range(self.num_perm)]
        self._b = [rng.randrange(0, _MERSENNE61) for _ in range(self.num_perm)]

    def shingles(self, text: str) -> List[str]:
        """The shingle set this hasher uses."""
        if self.shingle == "char":
            return char_shingles(text, self.k)
        return word_shingles(text, self.k)

    def signature_of_shingles(self, shingles: Sequence[str]) -> List[int]:
        """Signature of an explicit shingle set (empty set -> all ``p``)."""
        items = _as_seq(shingles, "shingles")
        hs = sorted(set(_mix64(fnv1a_64(_as_text(s, "shingle").encode("utf-8"))) % _MERSENNE61
                        for s in items))
        if not hs:
            return [_MERSENNE61] * self.num_perm
        return [min((a * x + b) % _MERSENNE61 for x in hs)
                for a, b in zip(self._a, self._b)]

    def signature(self, text: str) -> List[int]:
        """MinHash signature of a text."""
        return self.signature_of_shingles(self.shingles(text))

    @staticmethod
    def estimate(sig_a: Sequence[int], sig_b: Sequence[int]) -> float:
        """Estimated Jaccard similarity from two signatures."""
        sa = _as_seq(sig_a, "sig_a")
        sb = _as_seq(sig_b, "sig_b")
        if len(sa) != len(sb) or not sa:
            raise PublicVoiceError("signatures must be non-empty and of equal length.")
        return sum(1 for x, y in zip(sa, sb) if x == y) / len(sa)


def lsh_probability(similarity: float, bands: int, rows: int) -> float:
    """Probability that a pair of Jaccard ``s`` shares at least one LSH band:
    ``1 - (1 - s^r)^b``."""
    s = _as_unit(similarity, "similarity")
    b = _as_int(bands, "bands", 1)
    r = _as_int(rows, "rows", 1)
    return 1.0 - (1.0 - s ** r) ** b


def lsh_threshold(bands: int, rows: int) -> float:
    """The S-curve's approximate threshold ``(1/b)^(1/r)``."""
    b = _as_int(bands, "bands", 1)
    r = _as_int(rows, "rows", 1)
    return (1.0 / b) ** (1.0 / r)


def lsh_candidate_pairs(signatures: Dict[str, Sequence[int]], bands: int,
                        rows: int) -> List[Tuple[str, str]]:
    """Banded LSH: ids whose signatures agree on every row of some band.

    Band keys are SHA-1 digests of the band's integers (no builtin hash), and
    the output is a sorted list of ``(a, b)`` with ``a < b``."""
    if not isinstance(signatures, dict):
        raise PublicVoiceError("signatures must be a dict id -> signature.")
    b = _as_int(bands, "bands", 1)
    r = _as_int(rows, "rows", 1)
    ids = _str_keys(signatures, "signatures")
    sigs: Dict[str, List[int]] = {}
    for i in ids:
        sg = _as_seq(signatures[i], "signature of %s" % i)
        for v in sg:
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise PublicVoiceError("signature of %r must hold non-negative ints." % i)
        if len(sg) < b * r:
            raise PublicVoiceError("signature of %r is shorter than bands * rows = %d."
                                   % (i, b * r))
        sigs[i] = sg
    pairs = set()
    for band in range(b):
        buckets: Dict[str, List[str]] = {}
        for i in ids:
            chunk = sigs[i][band * r:(band + 1) * r]
            key = hashlib.sha1((",".join("%d" % v for v in chunk)).encode("ascii")).hexdigest()
            buckets.setdefault(key, []).append(i)
        for key in sorted(buckets):
            members = buckets[key]
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    pairs.add((members[x], members[y]))
    return sorted(pairs)


def simhash(text: str, bits: int = 64, features: str = "word") -> int:
    """Charikar SimHash fingerprint of the canonical text.

    Features (words, or character 3-shingles) are weighted by term frequency
    and hashed with FNV-1a 64 (extended by re-hashing for ``bits > 64``);
    bit ``i`` of the result is 1 when the weighted vote on that bit is
    positive.  Near-identical texts have a small Hamming distance."""
    nb = _as_int(bits, "bits", 8, 256)
    fm = _check_choice(features, ("word", "char"), "features")
    t = canonical_form(text)
    feats = t.split() if fm == "word" else (
        [t[i:i + 3] for i in range(len(t) - 2)] if len(t) >= 3 else ([t] if t else []))
    tf: Dict[str, int] = {}
    for f in feats:
        tf[f] = tf.get(f, 0) + 1
    votes = [0] * nb
    for f in sorted(tf):
        h = 0
        chunk = 0
        while chunk * 64 < nb:
            h |= fnv1a_64(("%d:" % chunk).encode("ascii") + f.encode("utf-8")) << (64 * chunk)
            chunk += 1
        wgt = tf[f]
        for i in range(nb):
            votes[i] += wgt if (h >> i) & 1 else -wgt
    out = 0
    for i in range(nb):
        if votes[i] > 0:
            out |= 1 << i
    return out


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits of two non-negative integers."""
    for nm, v in (("a", a), ("b", b)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise PublicVoiceError("%s must be a non-negative int." % nm)
    return bin(a ^ b).count("1")


def _simhash_candidates(prints: Dict[str, int], bits: int, max_distance: int
                        ) -> List[Tuple[str, str]]:
    """Pigeonhole blocking: two prints within ``max_distance`` bits agree
    EXACTLY on at least one of ``max_distance + 1`` disjoint bit blocks."""
    nblocks = max_distance + 1
    edges = [bits * i // nblocks for i in range(nblocks + 1)]
    ids = sorted(prints)
    pairs = set()
    for bi in range(nblocks):
        lo, hi = edges[bi], edges[bi + 1]
        mask = ((1 << (hi - lo)) - 1) << lo
        buckets: Dict[int, List[str]] = {}
        for i in ids:
            buckets.setdefault(prints[i] & mask, []).append(i)
        for key in sorted(buckets):
            m = buckets[key]
            for x in range(len(m)):
                for y in range(x + 1, len(m)):
                    pairs.add((m[x], m[y]))
    return sorted(pairs)


class _UnionFind:
    """Union-find whose representative is the lexicographically smallest id."""

    def __init__(self, ids: Sequence[str]) -> None:
        self.parent = {i: i for i in ids}

    def find(self, x: str) -> str:
        """Root with path halving."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        """Merge, keeping the smaller id as root."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if rb < ra:
            ra, rb = rb, ra
        self.parent[rb] = ra

    def groups(self) -> List[List[str]]:
        """Sorted groups of size >= 2, sorted by first member."""
        g: Dict[str, List[str]] = {}
        for i in sorted(self.parent):
            g.setdefault(self.find(i), []).append(i)
        return sorted([sorted(v) for v in g.values() if len(v) > 1], key=lambda v: v[0])


DUPLICATE_METHODS: Tuple[str, ...] = ("exact", "canonical", "minhash", "simhash")


@dataclass
class DuplicateReport:
    """Near-duplicate pairs and the groups they form."""
    method: str
    threshold: float
    pairs: List[Tuple[str, str, float]]
    groups: List[List[str]]
    n_candidates: int
    n_texts: int

    def representative_of(self) -> Dict[str, str]:
        """Member id -> group representative (smallest id)."""
        out = {}
        for g in self.groups:
            for m in g:
                out[m] = g[0]
        return out


def find_near_duplicates(texts: Dict[str, str], method: str = "minhash", *,
                         threshold: Optional[float] = None,
                         num_perm: Optional[int] = None, bands: Optional[int] = None,
                         k: Optional[int] = None, seed: Optional[int] = None,
                         bits: Optional[int] = None,
                         verify: Optional[str] = None) -> DuplicateReport:
    """Find near-duplicate texts.

    * ``"exact"``: identical raw strings (SHA-256 bucket).
    * ``"canonical"``: identical ``canonical_form`` -- collapses case,
      punctuation, whitespace and abbreviations.
    * ``"minhash"``: LSH banding on MinHash signatures of character
      k-shingles, then each candidate pair is kept when its similarity
      (``verify="estimate"``: signature agreement; ``"exact"``: true Jaccard
      of the shingle sets) is at least ``threshold`` (default 0.8).
      Defaults ``num_perm=128``, ``k=5``, ``seed=1``, ``verify="estimate"``;
      ``bands`` defaults to the divisor of ``num_perm`` whose S-curve
      threshold is closest to (just below) ``threshold``.
    * ``"simhash"``: pigeonhole-blocked SimHash, pairs within ``threshold``
      bits (default 3) of a ``bits``-bit print (default 64).
    ``exact`` and ``canonical`` take no threshold (identity is the test); any
    value other than None / 1 is rejected rather than silently ignored, and
    so is any option that only another method reads (``num_perm`` / ``bands``
    / ``k`` / ``seed`` / ``verify`` are MinHash's, ``bits`` is SimHash's).  A
    text whose canonical form is EMPTY (no words -- e.g. a map pin with no
    comment) is never paired: there is no text evidence of duplication.
    Groups are the connected components of the accepted pairs, each sorted,
    with the smallest id as representative."""
    if not isinstance(texts, dict):
        raise PublicVoiceError("texts must be a dict id -> text.")
    m = _check_choice(method, DUPLICATE_METHODS, "method")
    ids = _str_keys(texts, "texts")
    for i in ids:
        _as_text(texts[i], "texts[%s]" % i)
    mh_opts = (("num_perm", num_perm), ("bands", bands), ("k", k), ("seed", seed),
               ("verify", verify))
    for nm, val in mh_opts + (("bits", bits),):
        owner = "simhash" if nm == "bits" else "minhash"
        if val is not None and m != owner:
            raise PublicVoiceError("%s applies to method=%r only (got method=%r)."
                                   % (nm, owner, m))
    num_perm = 128 if num_perm is None else num_perm
    k = 5 if k is None else k
    seed = 1 if seed is None else seed
    bits = 64 if bits is None else bits
    vf = _check_choice("estimate" if verify is None else verify, ("estimate", "exact"),
                       "verify")
    live = [i for i in ids if canonical_form(texts[i])]
    pairs: List[Tuple[str, str, float]] = []
    n_cand = 0
    if m in ("exact", "canonical"):
        if threshold is not None and _as_unit(threshold, "threshold") != 1.0:
            raise PublicVoiceError("method %r takes no threshold (identity is the test)."
                                   % m)
        thr = 1.0
        buckets: Dict[str, List[str]] = {}
        for i in live:
            s = texts[i] if m == "exact" else canonical_form(texts[i])
            key = hashlib.sha256(s.encode("utf-8")).hexdigest()
            buckets.setdefault(key, []).append(i)
        for key in sorted(buckets):
            mem = buckets[key]
            for x in range(len(mem)):
                for y in range(x + 1, len(mem)):
                    pairs.append((mem[x], mem[y], 1.0))
        n_cand = len(pairs)
    elif m == "minhash":
        thr = 0.8 if threshold is None else _as_unit(threshold, "threshold", open_low=True)
        mh = MinHasher(num_perm, seed, "char", k)
        if bands is None:
            divs = [b for b in range(1, mh.num_perm + 1) if mh.num_perm % b == 0]
            below = [b for b in divs if lsh_threshold(b, mh.num_perm // b) <= thr]
            nb = min(below, key=lambda b: (thr - lsh_threshold(b, mh.num_perm // b), b)) \
                if below else divs[-1]
        else:
            nb = _as_int(bands, "bands", 1, mh.num_perm)
        rows = mh.num_perm // nb
        shing = {i: mh.shingles(texts[i]) for i in live}
        sigs = {i: mh.signature_of_shingles(shing[i]) for i in live}
        cands = lsh_candidate_pairs(sigs, nb, rows)
        n_cand = len(cands)
        for a, b in cands:
            sim = (MinHasher.estimate(sigs[a], sigs[b]) if vf == "estimate"
                   else jaccard(shing[a], shing[b]))
            if sim >= thr:
                pairs.append((a, b, sim))
    else:
        thr_i = 3 if threshold is None else _as_int(threshold, "threshold", 0)
        nbits = _as_int(bits, "bits", 8, 256)
        if thr_i >= nbits:
            raise PublicVoiceError("threshold must be below bits.")
        thr = float(thr_i)
        prints = {i: simhash(texts[i], nbits) for i in live}
        cands = _simhash_candidates(prints, nbits, thr_i)
        n_cand = len(cands)
        for a, b in cands:
            hd = hamming_distance(prints[a], prints[b])
            if hd <= thr_i:
                pairs.append((a, b, 1.0 - hd / nbits))
    uf = _UnionFind(ids)
    for a, b, _ in pairs:
        uf.union(a, b)
    pairs.sort(key=lambda p: (p[0], p[1]))
    return DuplicateReport(m, thr, pairs, uf.groups(), n_cand, len(ids))


COLLAPSE_POLICIES: Tuple[str, ...] = ("sum", "one", "sqrt")


def collapse_duplicates(concerns: Sequence[ConcernRecord], report: DuplicateReport,
                        policy: str = "sum", severity: str = "max",
                        max_distance: Optional[float] = None) -> List[ConcernRecord]:
    """Merge each duplicate group into its representative record.

    ``policy`` sets the merged ``count`` from the members' counts: ``"sum"``
    (every submission still counts -- the field is unchanged when severities
    agree), ``"one"`` (a campaign is one voice) or ``"sqrt"`` (square root of
    the summed count -- a compromise).  ``severity`` is the members' ``max`` or
    count-weighted ``mean``.  With ``max_distance`` a member farther than that
    from the representative is NOT merged (the same words about a different
    place are a different concern).  A member whose THEME differs from the
    representative's is never merged either: folding it in would silently move
    its field mass into another theme (and in or out of ``J_voice``).
    Members left out of a merge are NOT orphaned: the group is processed as
    a leader clustering in id order -- the smallest unmerged id leads, absorbs
    every remaining member of its theme (within ``max_distance`` of it), and
    the rest repeat -- so two same-theme duplicates still merge when a
    differently-themed (or distant) duplicate happens to hold the smaller id.
    Output is sorted by id."""
    cs = _check_concerns(concerns, allow_empty=True)
    if not isinstance(report, DuplicateReport):
        raise PublicVoiceError("report must be a DuplicateReport.")
    pol = _check_choice(policy, COLLAPSE_POLICIES, "policy")
    sev = _check_choice(severity, ("max", "mean"), "severity")
    md = None if max_distance is None else _as_nonneg(max_distance, "max_distance")
    by_id = {c.cid: c for c in cs}
    absorbed = set()
    merged: Dict[str, ConcernRecord] = {}
    clusters: List[List[ConcernRecord]] = []
    for g in report.groups:
        remaining = sorted(i for i in g if i in by_id)
        while len(remaining) >= 2:
            rep = by_id[remaining[0]]
            members = [rep]
            rest = []
            for i in remaining[1:]:
                c = by_id[i]
                if c.theme != rep.theme or (
                        md is not None and math.hypot(c.x - rep.x, c.y - rep.y) > md):
                    rest.append(i)
                else:
                    members.append(c)
            if len(members) >= 2:
                clusters.append(members)
            remaining = rest
    for members in clusters:
        rep = members[0]
        tot = math.fsum(c.count for c in members)
        if pol == "sum":
            cnt = tot
        elif pol == "one":
            cnt = 1.0
        else:
            cnt = math.sqrt(tot)
        if sev == "max":
            sv = max(c.severity for c in members)
        else:
            sv = math.fsum(c.severity * c.count for c in members) / tot
        merged[rep.cid] = ConcernRecord(rep.cid, rep.theme, sv, rep.x, rep.y, rep.mode,
                                        rep.source, cnt, rep.text, rep.respondent,
                                        rep.stratum)
        for c in members[1:]:
            absorbed.add(c.cid)
    out = []
    for c in cs:
        if c.cid in absorbed:
            continue
        out.append(merged.get(c.cid, c))
    return out


# ===========================================================================
# 9. SPATIAL WEIGHTS -- distance band, k-nearest, inverse distance, network
# ===========================================================================
@dataclass
class SpatialWeights:
    """A dense spatial-weights matrix over identified units.

    ``matrix[i][j]`` is the weight of unit ``j`` in unit ``i``'s
    neighbourhood; the diagonal is zero (self-weights for Getis-Ord ``G*``
    are added by that routine, never stored here).  The matrix need NOT be
    symmetric -- k-nearest-neighbour weights are not -- and every global
    statistic below uses the general (asymmetric) Cliff-Ord constants."""
    ids: List[str]
    matrix: Matrix
    transform: str = "binary"

    def __post_init__(self) -> None:
        if not isinstance(self.ids, list) or not self.ids:
            raise PublicVoiceError("ids must be a non-empty list.")
        for i in self.ids:
            _as_id(i, "unit id")
        if len(set(self.ids)) != len(self.ids):
            raise PublicVoiceError("unit ids must be unique.")
        n = len(self.ids)
        self.matrix = _as_matrix(self.matrix, n, "weights")
        for i in range(n):
            if self.matrix[i][i] != 0.0:
                raise PublicVoiceError("weights diagonal must be zero (unit %r)."
                                       % self.ids[i])
            for j in range(n):
                if self.matrix[i][j] < 0.0:
                    raise PublicVoiceError("weights must be non-negative.")
        self.transform = _as_text(self.transform, "transform")
        self.ids = list(self.ids)          # never alias the caller's list

    @property
    def n(self) -> int:
        """Number of units."""
        return len(self.ids)

    @property
    def s0(self) -> float:
        """``S0 = sum_ij w_ij``."""
        return math.fsum(v for row in self.matrix for v in row)

    @property
    def s1(self) -> float:
        """``S1 = 1/2 sum_ij (w_ij + w_ji)^2``."""
        n = self.n
        w = self.matrix
        return 0.5 * math.fsum((w[i][j] + w[j][i]) ** 2 for i in range(n) for j in range(n))

    @property
    def s2(self) -> float:
        """``S2 = sum_i (w_i. + w_.i)^2``."""
        n = self.n
        w = self.matrix
        return math.fsum((math.fsum(w[i]) + math.fsum(w[j][i] for j in range(n))) ** 2
                         for i in range(n))

    def row_sums(self) -> Vector:
        """``w_i.`` per unit."""
        return [math.fsum(r) for r in self.matrix]

    def neighbours(self, i: int) -> List[int]:
        """Indices with a positive weight in row ``i``."""
        return [j for j, v in enumerate(self.matrix[i]) if v > 0.0]

    def is_symmetric(self) -> bool:
        """True when ``w_ij == w_ji`` for all pairs."""
        n = self.n
        return all(self.matrix[i][j] == self.matrix[j][i]
                   for i in range(n) for j in range(i + 1, n))

    def islands(self) -> List[str]:
        """Units with no neighbour."""
        return [self.ids[i] for i in range(self.n) if not self.neighbours(i)]


def weights_from_matrix(ids: Sequence[str], matrix: Sequence[Sequence[float]],
                        transform: str = "custom") -> SpatialWeights:
    """Wrap an explicit matrix (validated by ``SpatialWeights``)."""
    if isinstance(ids, (str, bytes)) or ids is None:
        raise PublicVoiceError("ids must be a sequence of strings.")
    try:
        idl = list(ids)
    except TypeError:
        raise PublicVoiceError("ids must be a sequence of strings.")
    return SpatialWeights(idl, matrix, transform)


def row_standardise(w: SpatialWeights) -> SpatialWeights:
    """Row-standardised copy (each non-island row sums to 1; islands stay zero)."""
    if not isinstance(w, SpatialWeights):
        raise PublicVoiceError("w must be SpatialWeights.")
    rows = []
    for r in w.matrix:
        s = math.fsum(r)
        rows.append([v / s for v in r] if s > 0.0 else list(r))
    return SpatialWeights(list(w.ids), rows, "row")


def _check_ids_points(ids: Sequence[str], points: Sequence[Point]
                      ) -> Tuple[List[str], List[Point]]:
    """Validate parallel ids / points lists."""
    pts = _as_points(points, "points")
    idl = [_as_id(i, "id") for i in _as_seq(ids, "ids")]
    if len(idl) != len(pts):
        raise PublicVoiceError("ids and points differ in length.")
    return idl, pts


def weights_distance_band(ids: Sequence[str], points: Sequence[Point],
                          threshold: float, binary: bool = True,
                          power: Optional[float] = None) -> SpatialWeights:
    """Neighbours within ``threshold`` (inclusive): binary 1, or ``d^-power``
    (``power`` default 1; it applies to ``binary=False`` only, and passing
    one with binary weights is an error rather than silently ignored)."""
    idl, pts = _check_ids_points(ids, points)
    t = _as_positive(threshold, "threshold")
    _as_bool(binary, "binary")
    if binary and power is not None:
        raise PublicVoiceError("power applies to inverse-distance weights (binary=False) "
                               "only.")
    pw = 1.0 if power is None else _as_positive(power, "power")
    d = _pair_distances(pts)
    n = len(pts)
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and d[i][j] <= t:
                if binary:
                    m[i][j] = 1.0
                else:
                    if d[i][j] == 0.0:
                        raise PublicVoiceError("coincident points %r and %r have no "
                                               "inverse distance." % (idl[i], idl[j]))
                    try:
                        m[i][j] = d[i][j] ** (-pw)
                    except OverflowError:
                        m[i][j] = math.inf
                    if math.isinf(m[i][j]):
                        raise PublicVoiceError("inverse-distance weight %r^-%r of %r-%r "
                                               "exceeds the float range."
                                               % (d[i][j], pw, idl[i], idl[j]))
    return SpatialWeights(idl, m, "binary" if binary else "inverse_distance")


def weights_knn(ids: Sequence[str], points: Sequence[Point], k: int) -> SpatialWeights:
    """k-nearest-neighbour binary weights (ASYMMETRIC in general).

    Distance ties are broken on the neighbour's index, so the matrix is
    deterministic."""
    idl, pts = _check_ids_points(ids, points)
    n = len(pts)
    kk = _as_int(k, "k", 1)
    if kk > n - 1:
        raise PublicVoiceError("k must be at most n - 1 = %d (got %d)." % (n - 1, kk))
    d = _pair_distances(pts)
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        order = sorted((j for j in range(n) if j != i), key=lambda j: (d[i][j], j))
        for j in order[:kk]:
            m[i][j] = 1.0
    return SpatialWeights(idl, m, "knn")


def weights_kernel(ids: Sequence[str], points: Sequence[Point], bandwidth: float,
                   kernel: str = "quartic") -> SpatialWeights:
    """Kernel weights ``w_ij = k(d_ij / h)`` (peak-normalised), zero diagonal."""
    idl, pts = _check_ids_points(ids, points)
    h = _as_positive(bandwidth, "bandwidth")
    kn = _kernel_name(kernel)
    d = _pair_distances(pts)
    n = len(pts)
    m = [[0.0 if i == j else _kernel_weight_unchecked(kn, d[i][j], h, "peak")
          for j in range(n)] for i in range(n)]
    return SpatialWeights(idl, m, "kernel")


def weights_from_network(network: VoiceNetwork) -> SpatialWeights:
    """Binary edge-contiguity weights: two edges are neighbours when they share
    a node (the line graph).  Units are the network's sorted edge ids."""
    if not isinstance(network, VoiceNetwork):
        raise PublicVoiceError("network must be a VoiceNetwork.")
    eids = network.edge_ids
    pos = {e: k for k, e in enumerate(eids)}
    adj = network.edge_adjacency()
    n = len(eids)
    m = [[0.0] * n for _ in range(n)]
    for e in eids:
        for f in adj[e]:
            m[pos[e]][pos[f]] = 1.0
    return SpatialWeights(list(eids), m, "network_contiguity")


# ===========================================================================
# 10. GLOBAL AUTOCORRELATION -- Moran's I, Geary's C, General G, Ripley's K
# ===========================================================================
@dataclass
class GlobalStatistic:
    """A global spatial-autocorrelation statistic with its null moments."""
    name: str
    value: float
    expected: float
    var_normal: float
    var_random: float
    z_normal: float
    z_random: float
    p_normal: float
    p_random: float
    p_sim: Optional[float]
    n: int
    permutations: int


def _check_values_w(values: Sequence[float], w: SpatialWeights, minimum: int = 3
                    ) -> Vector:
    """Validate values against a weights object."""
    if not isinstance(w, SpatialWeights):
        raise PublicVoiceError("w must be SpatialWeights.")
    x = _as_floats(values, "values")
    if len(x) != w.n:
        raise PublicVoiceError("values has %d entries but weights has %d units."
                               % (len(x), w.n))
    if len(x) < minimum:
        raise PublicVoiceError("need at least %d units (got %d)." % (minimum, len(x)))
    if w.s0 == 0.0:
        raise PublicVoiceError("weights have no links (S0 = 0).")
    # Every statistic built on these values is invariant to their scale, but
    # the fourth moments (kurtosis b2) overflow above |x| ~ 1e77 and squares
    # underflow below ~ 1e-154.  Rescale by an exact POWER OF TWO so the
    # largest |x| lies in [0.5, 1): every product, sum and ratio downstream is
    # then the bit-identical scaled image of the unscaled computation.
    big = max(abs(v) for v in x)
    if big > 0.0:
        e = math.frexp(big)[1]
        x = [math.ldexp(v, -e) for v in x]
    return x


def _centred(x: Vector) -> Tuple[Vector, float]:
    """Deviations from the mean and their sum of squares; constant -> error.

    A sample whose spread is below ~1e-14 of its magnitude is treated as
    constant (its deviations are rounding noise, not signal)."""
    m = _mean(x)
    z = [v - m for v in x]
    ss = math.fsum(v * v for v in z)
    scale = max(abs(v) for v in x)
    if ss == 0.0 or math.sqrt(ss / len(x)) <= 1e-14 * scale:
        raise PublicVoiceError("values are constant; spatial autocorrelation undefined.")
    return z, ss


def _moran_value(z: Vector, ss: float, w: Matrix, s0: float) -> float:
    """Moran's I from centred values."""
    n = len(z)
    num = math.fsum(w[i][j] * z[i] * z[j] for i in range(n) for j in range(n) if w[i][j] != 0.0)
    return (n / s0) * num / ss


_ULP = 2.0 ** -52          # double-precision unit roundoff (2 x 2^-53)
_TIE_SAFETY = 4.0          # margin over the first-order rounding bounds below


def _p_sim(obs: float, sims: Sequence[float], tol: Optional[float] = None) -> float:
    """Folded one-sided pseudo p-value ``(1 + #extreme) / (1 + R)``.

    ``#extreme = min(#{s >= obs}, #{s <= obs})``: a simulated value EQUAL to
    the observed one is as extreme as it in BOTH tails.  The usual shortcut
    ``R - #{s >= obs}`` for the lower tail counts only ``s < obs`` and drops
    the ties -- harmless for continuous data, but sparse comment counts (most
    streets 0) make ties the norm, and a statistic that every permutation
    reproduces exactly (an island under ``G*``, complete-graph weights) then
    got ``p = 1/(R+1)`` -- "significant" -- instead of 1.

    "Equal" means equal in exact arithmetic: integer counts make many
    permutations tie with the observation through DIFFERENT neighbour sets
    (``0 + 2 = 1 + 1``), and the centring ``x - xbar`` rounds those sums
    differently, so a bitwise test split real ties at random into one tail or
    the other.  ``tol`` is therefore an absolute bound on the rounding error
    of the statistic, derived by each caller from its own formula
    (``_tie_tol_*``): two values within ``tol`` are ties.  It is a few ulps
    of the terms actually summed -- never a fixed fraction of the statistic,
    which (at 1e-9) merged GENUINELY distinct values once one count was
    ~1e5 times the others (one petition street among single comments) and
    moved Geary's p from 0.025 to 0.075.  ``tol=None`` uses
    ``64 ulp`` of the largest magnitude in play."""
    r = len(sims)
    big = max([abs(obs)] + [abs(s) for s in sims])
    t = 64.0 * _ULP * big if tol is None else tol + 8.0 * _ULP * big
    ge = sum(1 for s in sims if s >= obs - t)
    le = sum(1 for s in sims if s <= obs + t)
    return (min(ge, le) + 1.0) / (r + 1.0)


def _row_col_max(w: Matrix) -> Tuple[float, float]:
    """Largest row sum and largest column sum of a non-negative matrix."""
    n = len(w)
    rmax = max(math.fsum(row) for row in w)
    cmax = max(math.fsum(w[i][j] for i in range(n)) for j in range(n))
    return rmax, cmax


def _tie_tol_quadratic_centred(x: Vector, z: Vector, w: Matrix) -> float:
    """Rounding bound for ``sum_ij w_ij z_pi(i) z_pi(j)`` over ANY permutation
    ``pi`` of the centred values ``z = fl(x - fl(mean))``.

    Each ``z_i`` is off by at most ``ulp (2|m| + |z_i|)``; a product term by
    ``ulp |w| (2|m| (|z_i| + |z_j|) + 6 |z_i z_j|)`` (centring, two products,
    the weight's own representation and the final sum).  Over every
    permutation, ``sum_ij w_ij |z_pi(i)| <= R sum|z|`` (``R`` the largest row
    sum, ``C`` the largest column sum) and ``sum_ij w_ij |z_i||z_j| <=
    sqrt(R C) sum z^2`` (Schur's test).  Two exact ties differ by at most
    twice the bound."""
    m = abs(_mean(x))
    rmax, cmax = _row_col_max(w)
    sabs = math.fsum(abs(v) for v in z)
    ss = math.fsum(v * v for v in z)
    bound = _ULP * (2.0 * m * (rmax + cmax) * sabs + 6.0 * math.sqrt(rmax * cmax) * ss)
    return 2.0 * _TIE_SAFETY * bound


def _tie_tol_linear(ws: Vector, xs: Vector, centre: float) -> float:
    """Rounding bound for the DIFFERENCE between ``sum_j a_j (x_pi(j) - c)``
    at two permutations, ``c = fl(centre)``.  The error of ``c`` itself
    shifts every permutation by the same ``(c - centre) sum_j a_j`` and
    cancels; what remains is the subtraction, the product, the weight's own
    representation and the sum, ``<= 4 ulp sum_j |a_j| max_j |x_j - c|`` --
    plus each VALUE's own representation (``ulp/2 |x_j|``: 55.3 dB is not a
    binary float), which moves with the permutation.  Without it decimals
    on an offset (55.0 ... 55.4 dB) split their exact ties at random: a
    G* / LISA p of 0.25 for an exact 0.44 (and 0.44 at offset 0).

    The same bound serves the tests against an EXACT 0 (a G / G* z, a LISA
    lag): there the centre's error does not cancel, but it is at most
    ``ulp |c| sum|a| <= ulp max|x| sum|a|``, inside the representation term
    (a zero z on counts + 1000 used to come out -7e-14: "ns" -> "cold")."""
    dmax = max(abs(v - centre) for v in xs)
    xmax = max(abs(v) for v in xs)
    bound = _ULP * math.fsum(abs(a) for a in ws) * (4.0 * dmax + xmax)
    return 2.0 * _TIE_SAFETY * bound


def morans_i(values: Sequence[float], w: SpatialWeights, permutations: int = 0,
             seed: int = 0) -> GlobalStatistic:
    """Global Moran's I with normal and randomisation inference.

    ``I = (n / S0) sum_ij w_ij z_i z_j / sum_i z_i^2``; ``E[I] = -1/(n-1)``
    exactly under both nulls; the randomisation variance (Cliff-Ord, general
    asymmetric ``S1``/``S2``) is the EXACT variance of ``I`` over all ``n!``
    permutations of the values.  ``permutations > 0`` adds a seeded
    conditional-permutation pseudo p-value."""
    x = _check_values_w(values, w, 4)
    r = _as_int(permutations, "permutations", 0)
    sd = _as_seed(seed)
    n = len(x)
    z, ss = _centred(x)
    s0, s1, s2 = w.s0, w.s1, w.s2
    i_val = _moran_value(z, ss, w.matrix, s0)
    e = -1.0 / (n - 1)
    vn = (n * n * s1 - n * s2 + 3.0 * s0 * s0) / ((n * n - 1.0) * s0 * s0) - e * e
    b2 = n * math.fsum(v ** 4 for v in z) / (ss * ss)
    vr = ((n * ((n * n - 3 * n + 3) * s1 - n * s2 + 3.0 * s0 * s0)
           - b2 * ((n * n - n) * s1 - 2.0 * n * s2 + 6.0 * s0 * s0))
          / ((n - 1.0) * (n - 2.0) * (n - 3.0) * s0 * s0) - e * e)
    zn = (i_val - e) / math.sqrt(vn) if vn > 0.0 else 0.0
    zr = (i_val - e) / math.sqrt(vr) if vr > 0.0 else 0.0
    psim = None
    if r:
        rng = random.Random(sd)
        sims = []
        perm = list(z)
        for _ in range(r):
            rng.shuffle(perm)
            sims.append(_moran_value(perm, ss, w.matrix, s0))
        # every term is a product of centred values: bound the rounding of
        # the numerator over all permutations, then scale it like I
        tol = _tie_tol_quadratic_centred(x, z, w.matrix) * (n / s0) / ss
        psim = _p_sim(i_val, sims, tol)
    return GlobalStatistic("moran_i", i_val, e, vn, vr, zn, zr,
                           _normal_sf_two_sided(zn), _normal_sf_two_sided(zr), psim, n, r)


def _geary_value(x: Vector, ss: float, w: Matrix, s0: float) -> float:
    """Geary's C from raw values."""
    n = len(x)
    num = math.fsum(w[i][j] * (x[i] - x[j]) ** 2 for i in range(n) for j in range(n)
                    if w[i][j] != 0.0)
    return (n - 1.0) * num / (2.0 * s0 * ss)


def gearys_c(values: Sequence[float], w: SpatialWeights, permutations: int = 0,
             seed: int = 0) -> GlobalStatistic:
    """Geary's C: ``(n-1) sum_ij w_ij (x_i - x_j)^2 / (2 S0 sum z^2)``.

    ``E[C] = 1`` exactly; below 1 is positive autocorrelation.  Normal and
    randomisation variances are Cliff-Ord's (the latter exact over all
    permutations)."""
    x = _check_values_w(values, w, 4)
    r = _as_int(permutations, "permutations", 0)
    sd = _as_seed(seed)
    n = len(x)
    z, ss = _centred(x)
    s0, s1, s2 = w.s0, w.s1, w.s2
    c = _geary_value(x, ss, w.matrix, s0)
    vn = ((2.0 * s1 + s2) * (n - 1.0) - 4.0 * s0 * s0) / (2.0 * (n + 1.0) * s0 * s0)
    b2 = n * math.fsum(v ** 4 for v in z) / (ss * ss)
    vr = (((n - 1.0) * s1 * (n * n - 3.0 * n + 3.0 - (n - 1.0) * b2)
           - 0.25 * (n - 1.0) * s2 * (n * n + 3.0 * n - 6.0 - (n * n - n + 2.0) * b2)
           + s0 * s0 * (n * n - 3.0 - (n - 1.0) ** 2 * b2))
          / (n * (n - 2.0) * (n - 3.0) * s0 * s0))
    zn = (c - 1.0) / math.sqrt(vn) if vn > 0.0 else 0.0
    zr = (c - 1.0) / math.sqrt(vr) if vr > 0.0 else 0.0
    psim = None
    if r:
        rng = random.Random(sd)
        sims = []
        perm = list(x)
        for _ in range(r):
            rng.shuffle(perm)
            sims.append(_geary_value(perm, ss, w.matrix, s0))
        # sum of NON-NEGATIVE terms w (x_i - x_j)^2 on raw values: its rounding
        # is a few ulps of the statistic (the default 64 ulp) -- plus the
        # values' own representation (decimals: ulp/2 |x|), which moves each
        # difference by <= ulp max|x| and the sum by <= 2 ulp max|x| sum
        # w |x_i - x_j| <= 2 ulp max|x| (R + C) sum|z| over any permutation
        rmax, cmax = _row_col_max(w.matrix)
        rep = (2.0 * _TIE_SAFETY * 4.0 * _ULP * max(abs(v) for v in x) * (rmax + cmax)
               * math.fsum(abs(v) for v in z) * (n - 1.0) / (2.0 * s0 * ss))
        psim = _p_sim(c, sims, rep + 56.0 * _ULP * max([abs(c)] + [abs(v) for v in sims]))
    return GlobalStatistic("geary_c", c, 1.0, vn, vr, zn, zr,
                           _normal_sf_two_sided(zn), _normal_sf_two_sided(zr), psim, n, r)


def _general_g_value(x: Vector, w: Matrix) -> float:
    """General G from raw (non-negative) values."""
    n = len(x)
    num = math.fsum(w[i][j] * x[i] * x[j] for i in range(n) for j in range(n)
                    if i != j and w[i][j] != 0.0)
    sx = math.fsum(x)
    den = sx * sx - math.fsum(v * v for v in x)
    return num / den


def general_g(values: Sequence[float], w: SpatialWeights, permutations: int = 0,
              seed: int = 0) -> GlobalStatistic:
    """Getis-Ord General G: ``sum_{i!=j} w_ij x_i x_j / sum_{i!=j} x_i x_j``.

    Values must be non-negative.  ``E[G] = S0 / (n(n-1))`` and the variance
    is Getis & Ord's (1992) randomisation formula, exact over all
    permutations.  There is no normality variant; ``var_normal`` repeats the
    randomisation variance."""
    x = _check_values_w(values, w, 4)
    if any(v < 0.0 for v in x):
        raise PublicVoiceError("general_g needs non-negative values.")
    r = _as_int(permutations, "permutations", 0)
    sd = _as_seed(seed)
    n = len(x)
    _centred(x)
    s0, s1, s2 = w.s0, w.s1, w.s2
    m1 = math.fsum(x)
    m2 = math.fsum(v * v for v in x)
    m3 = math.fsum(v ** 3 for v in x)
    m4 = math.fsum(v ** 4 for v in x)
    if m1 * m1 - m2 <= 0.0:
        raise PublicVoiceError("general_g needs at least two positive values.")
    g = _general_g_value(x, w.matrix)
    e = s0 / (n * (n - 1.0))
    b0 = (n * n - 3.0 * n + 3.0) * s1 - n * s2 + 3.0 * s0 * s0
    b1 = -((n * n - n) * s1 - 2.0 * n * s2 + 6.0 * s0 * s0)
    b2 = -(2.0 * n * s1 - (n + 3.0) * s2 + 6.0 * s0 * s0)
    b3 = 4.0 * (n - 1.0) * s1 - 2.0 * (n + 1.0) * s2 + 8.0 * s0 * s0
    b4 = s1 - s2 + s0 * s0
    eg2 = ((b0 * m2 * m2 + b1 * m4 + b2 * m1 * m1 * m2 + b3 * m1 * m3 + b4 * m1 ** 4)
           / ((m1 * m1 - m2) ** 2 * n * (n - 1.0) * (n - 2.0) * (n - 3.0)))
    var = eg2 - e * e
    zz = (g - e) / math.sqrt(var) if var > 0.0 else 0.0
    psim = None
    if r:
        rng = random.Random(sd)
        sims = []
        perm = list(x)
        for _ in range(r):
            rng.shuffle(perm)
            sims.append(_general_g_value(perm, w.matrix))
        # sum of non-negative products of raw values: a few ulps of G itself
        psim = _p_sim(g, sims)
    p = _normal_sf_two_sided(zz)
    return GlobalStatistic("general_g", g, e, var, var, zz, zz, p, p, psim, n, r)


def permutation_distribution(values: Sequence[float], w: SpatialWeights,
                             statistic: str = "moran_i", exhaustive: bool = False,
                             permutations: int = 999, seed: int = 0) -> Vector:
    """The statistic over permutations of the values.

    ``exhaustive=True`` enumerates all ``n!`` orderings (``n <= 8``) -- the
    exact null distribution; otherwise ``permutations`` seeded shuffles."""
    stat = _check_choice(statistic, ("moran_i", "geary_c", "general_g"), "statistic")
    x = _check_values_w(values, w, 4)
    n = len(x)
    z, ss = _centred(x)
    s0 = w.s0
    if stat == "general_g" and any(v < 0.0 for v in x):
        raise PublicVoiceError("general_g needs non-negative values.")

    def f(v: Vector) -> float:
        if stat == "moran_i":
            m = _mean(v)
            return _moran_value([a - m for a in v], ss, w.matrix, s0)
        if stat == "geary_c":
            return _geary_value(v, ss, w.matrix, s0)
        return _general_g_value(v, w.matrix)

    if _as_bool(exhaustive, "exhaustive"):
        if n > 8:
            raise PublicVoiceError("exhaustive enumeration is limited to n <= 8.")
        return [f(list(p)) for p in itertools.permutations(x)]
    r = _as_int(permutations, "permutations", 1)
    rng = random.Random(_as_seed(seed))
    perm = list(x)
    out = []
    for _ in range(r):
        rng.shuffle(perm)
        out.append(f(list(perm)))
    return out


@dataclass
class RipleyResult:
    """Ripley's K, Besag's L and ``L(r) - r``."""
    radii: Vector
    k: Vector
    l: Vector
    h: Vector
    n: int
    area: float
    correction: str


RIPLEY_CORRECTIONS: Tuple[str, ...] = ("none", "border", "isotropic")


def _check_window(window: Any) -> Tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) with positive extent."""
    w = _as_floats(window, "window")
    if len(w) != 4 or w[2] <= w[0] or w[3] <= w[1]:
        raise PublicVoiceError("window must be (xmin, ymin, xmax, ymax) with positive extent.")
    return (w[0], w[1], w[2], w[3])


def ripley_edge_weight(point: Point, r: float,
                       window: Tuple[float, float, float, float]) -> float:
    """Ripley's isotropic edge correction ``1 / e`` where ``e`` is the fraction
    of the circle of radius ``r`` about ``point`` lying inside the rectangle.

    The outside arc is ``sum_sides 2 acos(d_s / r)`` minus, at each corner the
    circle contains, the double-counted overlap ``acos(d_a/r) + acos(d_b/r) -
    pi/2``.  Valid for ``r`` up to half the shorter side."""
    px, py = _as_point(point, "point")
    rr = _as_positive(r, "r")
    x0, y0, x1, y1 = _check_window(window)
    if not (x0 <= px <= x1 and y0 <= py <= y1):
        raise PublicVoiceError("point must lie inside the window.")
    if rr > 0.5 * min(x1 - x0, y1 - y0):
        raise PublicVoiceError("isotropic correction needs r <= half the shorter side.")
    d = {"l": px - x0, "r": x1 - px, "b": py - y0, "t": y1 - py}
    ang = {s: (math.acos(min(1.0, d[s] / rr)) if d[s] < rr else 0.0) for s in ("l", "r", "b", "t")}
    outside = 2.0 * math.fsum(ang[s] for s in ("l", "r", "b", "t"))
    for a, b in (("l", "b"), ("l", "t"), ("r", "b"), ("r", "t")):
        if d[a] < rr and d[b] < rr and d[a] * d[a] + d[b] * d[b] < rr * rr:
            outside -= ang[a] + ang[b] - 0.5 * math.pi
    e = 1.0 - outside / (2.0 * math.pi)
    return 1.0 / e


def ripleys_k(points: Sequence[Point], radii: Sequence[float],
              window: Tuple[float, float, float, float],
              correction: str = "isotropic") -> RipleyResult:
    """Ripley's K function with ``none`` / ``border`` / ``isotropic`` edge correction.

    ``K(r) = A / (n (n-1)) sum_{i != j} e_ij 1[d_ij <= r]`` (``e_ij = 1`` with
    no correction, Ripley's isotropic weight otherwise); the border estimator
    uses only points at least ``r`` from the boundary,
    ``K(r) = A sum_{i: b_i >= r} #{j : d_ij <= r} / ((n-1) m_r)``.  Under
    complete spatial randomness ``K(r) = pi r^2`` and ``L(r) = r``."""
    pts = _as_points(points, "points")
    rs = _as_floats(radii, "radii")
    if any(v <= 0.0 for v in rs):
        raise PublicVoiceError("radii must be positive.")
    win = _check_window(window)
    corr = _check_choice(correction, RIPLEY_CORRECTIONS, "correction")
    x0, y0, x1, y1 = win
    for (px, py) in pts:
        if not (x0 <= px <= x1 and y0 <= py <= y1):
            raise PublicVoiceError("every point must lie inside the window.")
    n = len(pts)
    if n < 2:
        raise PublicVoiceError("ripleys_k needs at least two points.")
    area = (x1 - x0) * (y1 - y0)
    d = _pair_distances(pts)
    ks = []
    for r in rs:
        if corr == "border":
            terms = []
            m = 0
            for i in range(n):
                bi = min(pts[i][0] - x0, x1 - pts[i][0], pts[i][1] - y0, y1 - pts[i][1])
                if bi >= r:
                    m += 1
                    terms.append(float(sum(1 for j in range(n) if j != i and d[i][j] <= r)))
            kv = area * math.fsum(terms) / ((n - 1.0) * m) if m else math.nan
        else:
            terms = []
            for i in range(n):
                for j in range(n):
                    if i != j and d[i][j] <= r:
                        if corr == "none":
                            terms.append(1.0)
                        else:
                            terms.append(ripley_edge_weight(pts[i], d[i][j], win)
                                         if d[i][j] > 0.0 else 1.0)
            kv = area * math.fsum(terms) / (n * (n - 1.0))
        ks.append(kv)
    ls = [math.sqrt(v / math.pi) if not math.isnan(v) else math.nan for v in ks]
    hs = [lv - r for lv, r in zip(ls, rs)]
    return RipleyResult(rs, ks, ls, hs, n, area, corr)


@dataclass
class Envelope:
    """Pointwise simulation envelope of ``L(r) - r`` under CSR."""
    radii: Vector
    lower: Vector
    upper: Vector
    simulations: int


def csr_envelope(n: int, radii: Sequence[float], window: Tuple[float, float, float, float],
                 simulations: int = 99, seed: int = 0,
                 correction: str = "isotropic") -> Envelope:
    """Min/max of ``L(r) - r`` over seeded CSR (binomial) patterns of ``n`` points."""
    nn = _as_int(n, "n", 2)
    rs = _as_floats(radii, "radii")
    win = _check_window(window)
    sims = _as_int(simulations, "simulations", 1)
    rng = random.Random(_as_seed(seed))
    x0, y0, x1, y1 = win
    lo = [math.inf] * len(rs)
    hi = [-math.inf] * len(rs)
    for _ in range(sims):
        pts = [(x0 + (x1 - x0) * rng.random(), y0 + (y1 - y0) * rng.random())
               for _ in range(nn)]
        res = ripleys_k(pts, rs, win, correction)
        for k, v in enumerate(res.h):
            if math.isnan(v):
                continue
            lo[k] = min(lo[k], v)
            hi[k] = max(hi[k], v)
    # A radius no simulation could estimate (border correction with no point
    # at least r from the edge) has NO envelope: NaN, like ``ripleys_k``
    # itself -- never the inverted [inf, -inf] that every finite observed
    # L(r) - r would lie "outside".
    lo = [v if not math.isinf(v) else math.nan for v in lo]
    hi = [v if not math.isinf(v) else math.nan for v in hi]
    return Envelope(rs, lo, hi, sims)


# ===========================================================================
# 11. HOTSPOTS -- local Getis-Ord G / G*, local Moran's I, BH-FDR, clusters
# ===========================================================================
def _bh_local(p: Vector) -> Vector:
    """Benjamini-Hochberg step-up adjusted p-values (local implementation)."""
    n = len(p)
    order = sorted(range(n), key=lambda i: p[i])
    adj = [0.0] * n
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, min(1.0, p[i] * n / (rank + 1)))
        adj[i] = prev
    return adj


def benjamini_hochberg(pvalues: Sequence[float]) -> Vector:
    """Benjamini-Hochberg FDR-adjusted p-values (q-values).

    ``q_(k) = min_{m >= k} min(1, p_(m) n / m)`` over the ascending order;
    ties are ordered by input position.  Delegates to the core library's
    ``p_adjust(method="bh")`` when importable, AFTER validating here, so the
    values and the exception type are identical on either path."""
    p = _as_floats(pvalues, "pvalues", allow_empty=True)
    for v in p:
        if v < 0.0 or v > 1.0:
            raise PublicVoiceError("p-values must lie in [0, 1] (got %r)." % v)
    if not p:
        return []
    if _HAVE_CORE_STATS:
        return [float(v) for v in _core_p_adjust(p, method="bh")]
    return _bh_local(p)


def bh_reject(pvalues: Sequence[float], q: float = 0.05) -> List[bool]:
    """Benjamini-Hochberg step-up decisions at FDR level ``q``."""
    qq = _as_unit(q, "q", open_low=True, open_high=True)
    adj = benjamini_hochberg(pvalues)
    return [a <= qq for a in adj]


@dataclass
class LocalGResult:
    """Local Getis-Ord statistics for every unit."""
    ids: List[str]
    star: bool
    g: Vector
    z: Vector
    p_normal: Vector
    p_sim: Optional[Vector]
    p_adjusted: Vector
    labels: List[str]
    alpha: float
    fdr: bool


def _local_g_terms(x: Vector, w: Matrix, i: int, star: bool, self_weight: float
                   ) -> Tuple[Vector, Vector]:
    """(weights, values) entering G_i or G_i* for unit i."""
    n = len(x)
    if star:
        ws = [self_weight if j == i else w[i][j] for j in range(n)]
        return ws, list(x)
    return [w[i][j] for j in range(n) if j != i], [x[j] for j in range(n) if j != i]


def _g_z(ws: Vector, xs: Vector) -> Tuple[float, float]:
    """(raw G ratio, standardised z) for one unit from its weights and values.

    ``z = (sum w x - xbar W) / (s sqrt((m S1 - W^2) / (m - 1)))`` with ``m``
    the number of values, ``s^2 = sum x^2 / m - xbar^2`` -- the exact
    permutation moments of a weighted sum sampled without replacement.
    A degenerate row (all weights equal, or zero spread) gets z = 0."""
    m = len(xs)
    sx = math.fsum(xs)
    gnum = math.fsum(a * b for a, b in zip(ws, xs))
    graw = gnum / sx if sx != 0.0 else 0.0
    xbar = sx / m
    var = math.fsum((v - xbar) ** 2 for v in xs) / m
    wsum = math.fsum(ws)
    s1 = math.fsum(a * a for a in ws)
    spread = m * s1 - wsum * wsum
    scale = max(abs(v) for v in xs) if xs else 0.0
    if var <= (1e-14 * scale) ** 2 or spread <= 1e-12 * m * s1:
        return graw, 0.0
    num = math.fsum(a * (b - xbar) for a, b in zip(ws, xs))
    return graw, num / (math.sqrt(var) * math.sqrt(spread / (m - 1.0)))


def _g_tie_tol(ws: Vector, xs: Vector) -> float:
    """Tie tolerance for the conditional-permutation ``z`` of ``_g_z``: its
    numerator ``sum w (x - xbar)`` is linear in centred values and its
    denominator is the same for every permutation of ``xs``; it also bounds
    a z against an exact 0."""
    m = len(xs)
    xbar = math.fsum(xs) / m
    var = math.fsum((v - xbar) ** 2 for v in xs) / m
    wsum = math.fsum(ws)
    spread = m * math.fsum(a * a for a in ws) - wsum * wsum
    if var <= 0.0 or spread <= 0.0:
        return 0.0
    return _tie_tol_linear(ws, xs, xbar) / (math.sqrt(var) * math.sqrt(spread / (m - 1.0)))


def local_getis_ord(values: Sequence[float], w: SpatialWeights, star: bool = True,
                    self_weight: Optional[float] = None, permutations: int = 0,
                    seed: int = 0, alpha: float = 0.05, fdr: bool = True) -> LocalGResult:
    """Local Getis-Ord ``G_i`` (``star=False``) or ``G_i*`` (``star=True``).

    Returned ``z`` is the standardised statistic of Ord & Getis (1995):
    under random permutation of the values (all ``n`` for ``G*``, the other
    ``n-1`` with ``x_i`` held for ``G``) it has mean exactly 0 and variance
    exactly 1.  ``G*`` includes unit ``i`` with weight ``self_weight``
    (default 1; passing one with ``star=False`` is an error, never silently
    ignored).  An ISLAND (no neighbour in row ``i``) carries no spatial
    information: its ``G*`` z would only standardise its own value, so it
    gets ``z = 0``, ``p = 1``, ``"ns"`` -- as under ``G`` and as its
    conditional-permutation p-value always was.  So does a unit whose z is 0
    in exact arithmetic (its local sum equals its expectation): a ``|z|``
    within the rounding bound of its numerator is set to exactly 0, never
    left as a +-1e-16 whose sign would make it "hot" or "cold".
    Labels are ``"hot"`` / ``"cold"`` / ``"ns"`` from two-sided p-values --
    the seeded conditional-permutation pseudo p-values when
    ``permutations > 0``, else the normal ones -- after Benjamini-Hochberg
    adjustment when ``fdr``."""
    x = _check_values_w(values, w, 3)
    if not isinstance(star, bool) or not isinstance(fdr, bool):
        raise PublicVoiceError("star and fdr must be bools.")
    if self_weight is not None and not star:
        raise PublicVoiceError("self_weight applies to G* (star=True) only.")
    sw = 1.0 if self_weight is None else _as_positive(self_weight, "self_weight")
    r = _as_int(permutations, "permutations", 0)
    rng = random.Random(_as_seed(seed))
    a = _as_unit(alpha, "alpha", open_low=True, open_high=True)
    _centred(x)
    n = len(x)
    gs: Vector = []
    zs: Vector = []
    psim: Optional[Vector] = [] if r else None
    for i in range(n):
        ws, xs = _local_g_terms(x, w.matrix, i, star, sw)
        g, z = _g_z(ws, xs)
        tol = _g_tie_tol(ws, xs)
        island = not any(w.matrix[i][j] > 0.0 for j in range(n) if j != i)
        if island or abs(z) <= tol:
            # An island carries no spatial information; and a z that is 0 in
            # exact arithmetic (the local sum equals its expectation -- e.g. a
            # street with 4 comments and two silent neighbours when the mean is
            # 4/3) comes out as +-1e-16 of rounding noise whose SIGN made the
            # street "hot", "ns" or "cold" depending on a units factor.
            z = 0.0
        gs.append(g)
        zs.append(z)
        if r:
            assert psim is not None
            if star:
                others = [x[j] for j in range(n) if j != i]
                sims = []
                for _ in range(r):
                    rng.shuffle(others)
                    perm = others[:i] + [x[i]] + others[i:]
                    sims.append(_g_z(ws, perm)[1])
            else:
                others = list(xs)
                sims = []
                for _ in range(r):
                    rng.shuffle(others)
                    sims.append(_g_z(ws, others)[1])
            psim.append(_p_sim(z, sims, tol) if z != 0.0 else 1.0)
    pn = [_normal_sf_two_sided(z) for z in zs]
    base = [min(1.0, 2.0 * v) for v in psim] if psim is not None else pn
    padj = benjamini_hochberg(base) if fdr else list(base)
    labels = []
    for z, pa in zip(zs, padj):
        if pa <= a and z > 0.0:
            labels.append("hot")
        elif pa <= a and z < 0.0:
            labels.append("cold")
        else:
            labels.append("ns")
    return LocalGResult(list(w.ids), star, gs, zs, pn, psim, padj, labels, a, fdr)


@dataclass
class LocalMoranResult:
    """Local Moran's I (LISA) for every unit."""
    ids: List[str]
    i: Vector
    expected: Vector
    variance: Vector
    z: Vector
    p_normal: Vector
    p_sim: Optional[Vector]
    p_adjusted: Vector
    quadrant: List[str]
    labels: List[str]
    alpha: float
    fdr: bool


def _local_moran_value(z: Vector, m2: float, w: Matrix, i: int) -> float:
    """``I_i = (z_i / m2) sum_j w_ij z_j``."""
    return z[i] / m2 * math.fsum(w[i][j] * z[j] for j in range(len(z)) if w[i][j] != 0.0)


def local_morans_i(values: Sequence[float], w: SpatialWeights, permutations: int = 0,
                   seed: int = 0, alpha: float = 0.05, fdr: bool = True
                   ) -> LocalMoranResult:
    """Anselin's (1995) local Moran's I with cluster/outlier classification.

    ``I_i = (z_i / m2) sum_j w_ij z_j`` with ``m2 = sum z^2 / n``, so
    ``sum_i I_i = S0 * I_global`` exactly.  Under total randomisation
    ``E[I_i] = -w_i. / (n-1)`` and the variance is Anselin's formula (exact
    over all permutations).  ``quadrant`` is HH / LL / HL / LH from the signs
    of ``z_i`` and its spatial lag; ``labels`` keep the quadrant where the
    (FDR-adjusted) p-value is at most ``alpha`` and are ``"ns"`` elsewhere --
    HH and LL are clusters, HL and LH are spatial outliers.

    A unit whose value IS the mean (``|z_i| <= 1e-14 max|x|``, the same
    rounding-noise floor as the constant-values check) has ``I_i = 0`` under
    every conditional permutation and neither a high nor a low value: it
    gets ``z = 0``, ``p = 1``, ``"ns"``.  Anselin's total-randomisation
    moments (which also move ``x_i``) would otherwise make it "significant"
    on tied, two-valued counts -- e.g. the centre of a 3 x 3 block holding
    the mean between 0s and 2s came out a ``"LL"`` cluster at p = 0.005.
    Likewise a spatial lag that is 0 in exact arithmetic (neighbours
    averaging exactly the mean) is "L", whatever the sign of its rounding."""
    x = _check_values_w(values, w, 3)
    if not isinstance(fdr, bool):
        raise PublicVoiceError("fdr must be a bool.")
    r = _as_int(permutations, "permutations", 0)
    rng = random.Random(_as_seed(seed))
    a = _as_unit(alpha, "alpha", open_low=True, open_high=True)
    n = len(x)
    z, ss = _centred(x)
    mean = _mean(x)
    at_mean_tol = 1e-14 * max(abs(v) for v in x)
    m2 = ss / n
    b2 = (math.fsum(v ** 4 for v in z) / n) / (m2 * m2)
    ivals, ex, var, zs, quad = [], [], [], [], []
    psim: Optional[Vector] = [] if r else None
    for i in range(n):
        row = w.matrix[i]
        wi = math.fsum(row)
        wi2 = math.fsum(v * v for v in row)
        ii = _local_moran_value(z, m2, w.matrix, i)
        e = -wi / (n - 1.0)
        v = (wi2 * (n - b2) / (n - 1.0)
             + (wi * wi - wi2) * (2.0 * b2 - n) / ((n - 1.0) * (n - 2.0))
             - wi * wi / ((n - 1.0) ** 2))
        at_mean = abs(z[i]) <= at_mean_tol
        ivals.append(ii)
        ex.append(e)
        var.append(v)
        zs.append((ii - e) / math.sqrt(v) if v > 0.0 and not at_mean else 0.0)
        lag = math.fsum(row[j] * z[j] for j in range(n))
        # A lag that is 0 in exact arithmetic (neighbours averaging exactly the
        # mean, e.g. 3 neighbours summing to 4 when the mean is 4/3) comes out
        # as +-rounding noise of fl(mean); its sign is not a direction.  Within
        # the rounding bound it counts as 0 ("L", like an exact 0), so the
        # quadrant no longer flips when every count is multiplied by 3.
        lag_tol = _tie_tol_linear(row, x, mean)
        quad.append(("H" if z[i] > at_mean_tol else "L")
                    + ("H" if lag > lag_tol else "L"))
        if r:
            assert psim is not None
            # I_i = (z_i / m2) * lag and the lag is linear in centred values
            tol = abs(z[i] / m2) * _tie_tol_linear(row, x, mean)
            others = [z[j] for j in range(n) if j != i]
            sims = []
            for _ in range(r):
                rng.shuffle(others)
                perm = others[:i] + [z[i]] + others[i:]
                sims.append(_local_moran_value(perm, m2, w.matrix, i))
            psim.append(_p_sim(ii, sims, tol) if not at_mean else 1.0)
    pn = [_normal_sf_two_sided(v) for v in zs]
    base = [min(1.0, 2.0 * v) for v in psim] if psim is not None else pn
    padj = benjamini_hochberg(base) if fdr else list(base)
    labels = [q if pa <= a else "ns" for q, pa in zip(quad, padj)]
    return LocalMoranResult(list(w.ids), ivals, ex, var, zs, pn, psim, padj, quad, labels,
                            a, fdr)


@dataclass
class Hotspot:
    """A contiguous cluster of significant units and the comments behind it."""
    hid: str
    kind: str
    unit_ids: List[str]
    n_records: int
    n_comments: float
    n_respondents: int
    by_theme: Dict[str, float]
    mean_severity: float
    peak_value: float


def hotspot_clusters(ids: Sequence[str], labels: Sequence[str], w: SpatialWeights,
                     target: str = "hot") -> List[List[str]]:
    """Connected components (through positive weights in either direction) of
    the units labelled ``target``; each sorted, ordered by first id."""
    if not isinstance(w, SpatialWeights):
        raise PublicVoiceError("w must be SpatialWeights.")
    idl = [_as_id(i, "id") for i in _as_seq(ids, "ids")]
    lab = [_as_text(v, "label") for v in _as_seq(labels, "labels")]
    if idl != w.ids or len(lab) != len(idl):
        raise PublicVoiceError("ids and labels must align with the weights' ids.")
    tg = _as_text(target, "target")
    n = len(idl)
    member = [lab[i] == tg for i in range(n)]
    seen = [False] * n
    out = []
    for s in range(n):
        if not member[s] or seen[s]:
            continue
        comp = []
        stack = [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(idl[u])
            for v in range(n):
                if member[v] and not seen[v] and (w.matrix[u][v] > 0.0 or w.matrix[v][u] > 0.0):
                    seen[v] = True
                    stack.append(v)
        out.append(sorted(comp))
    return sorted(out, key=lambda c: c[0])


def comment_counts_per_hotspot(clusters: Sequence[Sequence[str]],
                               concerns: Sequence[ConcernRecord],
                               unit_of_concern: Dict[str, str],
                               unit_values: Optional[Dict[str, float]] = None,
                               kind: str = "hot") -> List[Hotspot]:
    """How many comments (records, summed ``count``, distinct respondents,
    per theme) fall in each hotspot cluster.

    ``unit_of_concern`` maps concern id -> unit id (e.g. the attachment edge
    from ``attach_concerns``); concerns mapped to no cluster are ignored.
    ``unit_values`` (e.g. the field total per edge) supplies ``peak_value``."""
    cs = _check_concerns(concerns, allow_empty=True)
    kind = _as_id(kind, "kind")
    if not isinstance(unit_of_concern, dict):
        raise PublicVoiceError("unit_of_concern must be a dict cid -> unit id.")
    vals = {} if unit_values is None else unit_values
    if not isinstance(vals, dict):
        raise PublicVoiceError("unit_values must be a dict unit id -> value.")
    for cid in unit_of_concern:
        _as_id(cid, "unit_of_concern key")
        _as_id(unit_of_concern[cid], "unit of %r" % (cid,))
    out = []
    for k, cl in enumerate(_as_seq(clusters, "clusters")):
        units = sorted(_as_id(u, "unit id") for u in _as_seq(cl, "cluster"))
        uset = set(units)
        sub = [c for c in cs if unit_of_concern.get(c.cid) in uset]
        tot = math.fsum(c.count for c in sub)
        by = {th: math.fsum(c.count for c in sub if c.theme == th) for th in THEMES}
        resp = sorted(set(c.respondent for c in sub if c.respondent))
        msev = math.fsum(c.severity * c.count for c in sub) / tot if tot > 0.0 else 0.0
        peak = max([_as_float(vals[u], "unit_values[%s]" % u) for u in units if u in vals],
                   default=0.0)
        out.append(Hotspot("%s%d" % (kind, k + 1), kind, units, len(sub), tot, len(resp),
                           by, msev, peak))
    return out


# ===========================================================================
# 12. HEATMAP CLASSIFICATION -- Jenks natural breaks, bins, hexagons
# ===========================================================================
@dataclass
class JenksResult:
    """Optimal k-class partition of sorted values."""
    k: int
    breaks: Vector
    class_sizes: List[int]
    sdcm: float
    sdam: float
    gvf: float


def _ssd_column(xs: Vector, j: int) -> Vector:
    """``out[i]`` = sum of squared deviations of sorted ``xs[i:j]`` for every
    ``i < j``, accumulated from ``j - 1`` downwards in deviations from the
    ANCHOR ``xs[j - 1]``.

    A global prefix-sum form ``S2 - S1^2/m`` cancels catastrophically for a
    tight class far from the global mean (a cluster at 1e8 with a 1e-4
    spread came out as 48 instead of 4e-10 and picked the wrong breaks).
    Anchoring each class at its own end bounds the cancellation: with the
    span ``R`` of the class, ``sum d^2 <= SSD + m R^2 <= (1 + 2m) SSD``, so the
    relative error is ``O(m eps)`` whatever the offset."""
    out = [0.0] * j
    anchor = xs[j - 1]
    s1 = 0.0
    s2 = 0.0
    for i in range(j - 1, -1, -1):
        d = xs[i] - anchor
        s1 += d
        s2 += d * d
        m = j - i
        v = s2 - s1 * s1 / m
        out[i] = v if v > 0.0 else 0.0
    return out


def _pow2_scaled(xs: Sequence[float]) -> Tuple[Vector, int]:
    """``xs`` divided by the power of two ``2^e`` that puts ``max |x|`` in
    ``[0.5, 1)``, and ``e``.  Partitions, ratios and the GVF are invariant
    to a common factor and the rescale is exact, so they come out
    bit-identical wherever nothing over/underflows -- and correct far beyond
    that: unscaled, squared deviations overflow above ~1e154 (an ``inf - inf``
    NaN cost made Jenks pick arbitrary breaks) and underflow below ~1e-162."""
    big = max((abs(v) for v in xs), default=0.0)
    if big == 0.0:
        return list(xs), 0
    e = math.frexp(big)[1]
    return [math.ldexp(v, -e) for v in xs], e


def _unscale_ssd(v: float, e: int) -> float:
    """A sum of squares of ``2^-e``-scaled values, scaled back; ``inf`` when the
    true value exceeds the float range (it is then not representable)."""
    try:
        return math.ldexp(v, 2 * e)
    except OverflowError:
        return math.inf


def _exact_ssd(seg: Sequence[float]) -> float:
    """Sum of squared deviations of one class: two-pass on deviations from the
    class's own last value.  A plain two-pass is NOT enough far from zero:
    the rounded mean of values near 1e9 is off by up to half an ulp (6e-8),
    which alone is a 1e-4 relative error for a 1e-5 spread.  Deviations from
    a member are exact (Sterbenz), so their mean is accurate to the spread."""
    anchor = seg[-1]
    d = [v - anchor for v in seg]
    mu = _mean(d)
    return math.fsum((v - mu) * (v - mu) for v in d)


def jenks_breaks(values: Sequence[float], k: int) -> JenksResult:
    """Jenks natural breaks by exact dynamic programming (Fisher 1958).

    Minimises the within-class sum of squared deviations (SDCM) over every
    partition of the SORTED values into ``k`` contiguous non-empty classes.
    ``breaks`` has ``k + 1`` entries: the minimum, then each class's upper
    value (class ``c`` is ``(breaks[c], breaks[c+1]]``, the first closed at
    the minimum).  Each candidate class's SSD is accumulated relative to its
    own end value (``_ssd_column``), so the cancellation error is relative to
    the class's own spread, never to the data's magnitude or offset; the
    reported SDCM / SDAM are two-pass sums.  On an exact SDCM tie the earliest
    break wins.  ``gvf = 1 - SDCM/SDAM``.  The DP runs on the values divided
    by an exact power of two (``_pow2_scaled``), so the partition and the GVF
    are exact at any magnitude; SDCM / SDAM are scaled back and are ``inf``
    only when the true sum of squares exceeds the float range."""
    xs = sorted(_as_floats(values, "values"))
    kk = _as_int(k, "k", 1)
    n = len(xs)
    distinct = len(set(xs))
    if kk > distinct:
        raise PublicVoiceError("k = %d exceeds the number of distinct values (%d)."
                               % (kk, distinct))
    raw = xs
    xs, ex = _pow2_scaled(raw)
    inf = math.inf
    # cost[m][j]: best SDCM of the first j values in m classes; arg[m][j]: start of last class
    cost = [[inf] * (n + 1) for _ in range(kk + 1)]
    arg = [[0] * (n + 1) for _ in range(kk + 1)]
    cost[0][0] = 0.0
    for j in range(1, n + 1):
        col = _ssd_column(xs, j)
        for m in range(1, min(kk, j) + 1):
            if j > n - (kk - m):
                continue
            best = inf
            bi = m - 1
            for i in range(m - 1, j):
                if cost[m - 1][i] == inf:
                    continue
                v = cost[m - 1][i] + col[i]
                if v < best:
                    best = v
                    bi = i
            cost[m][j] = best
            arg[m][j] = bi
    cuts = []
    j = n
    for m in range(kk, 0, -1):
        i = arg[m][j]
        cuts.append((i, j))
        j = i
    cuts.reverse()
    sizes = [b - a for a, b in cuts]
    breaks = [raw[0]] + [raw[b - 1] for _, b in cuts]
    sdcm = math.fsum(_exact_ssd(xs[a:b]) for a, b in cuts)
    sdam = _exact_ssd(xs)
    gvf = 1.0 - sdcm / sdam if sdam > 0.0 else 1.0
    return JenksResult(kk, breaks, sizes, _unscale_ssd(sdcm, ex), _unscale_ssd(sdam, ex),
                       gvf)


def equal_interval_breaks(values: Sequence[float], k: int) -> Vector:
    """``k + 1`` equally spaced breaks from min to max."""
    xs = _as_floats(values, "values")
    kk = _as_int(k, "k", 1)
    lo, hi = min(xs), max(xs)
    return [lo + (hi - lo) * i / kk for i in range(kk)] + [hi]


def quantile_breaks(values: Sequence[float], k: int) -> Vector:
    """``k + 1`` breaks at the type-7 quantiles ``0, 1/k, ..., 1``."""
    xs = sorted(_as_floats(values, "values"))
    kk = _as_int(k, "k", 1)
    return [_quantile(xs, i / kk) for i in range(kk)] + [xs[-1]]


def classify_values(values: Sequence[float], breaks: Sequence[float]) -> List[int]:
    """Class index (0-based) of each value: class ``c`` is ``(b_c, b_{c+1}]``,
    with values at or below ``b_0`` in class 0 and above the last break in
    the last class."""
    xs = _as_floats(values, "values", allow_empty=True)
    br = _as_floats(breaks, "breaks")
    if len(br) < 2:
        raise PublicVoiceError("breaks needs at least two entries.")
    for a, b in zip(br, br[1:]):
        if b < a:
            raise PublicVoiceError("breaks must be non-decreasing.")
    inner = br[1:-1]
    return [bisect.bisect_left(inner, v) for v in xs]


@dataclass
class HeatmapClasses:
    """Values classified for a choropleth / heat map."""
    method: str
    breaks: Vector
    classes: List[int]
    counts: List[int]
    gvf: float


def heatmap_classes(values: Sequence[float], k: int = 5,
                    method: str = "jenks") -> HeatmapClasses:
    """Classify values by ``"jenks"``, ``"quantile"`` or ``"equal"`` breaks.

    ``k`` is reduced to the number of distinct values when larger; the
    goodness of variance fit is reported for every method."""
    xs = _as_floats(values, "values")
    kk = _as_int(k, "k", 1)
    md = _check_choice(method, ("jenks", "quantile", "equal"), "method")
    kk = min(kk, len(set(xs)))
    if md == "jenks":
        br = jenks_breaks(xs, kk).breaks
    elif md == "quantile":
        br = quantile_breaks(xs, kk)
    else:
        br = equal_interval_breaks(xs, kk)
    cls = classify_values(xs, br)
    counts = [sum(1 for c in cls if c == i) for i in range(len(br) - 1)]
    # The GVF the same way ``jenks_breaks`` computes it (power-of-two scaled,
    # each class anchored at its own sorted end), so it is exact at any
    # magnitude or offset and, for "jenks", bit-identical to JenksResult.gvf.
    ys, _ = _pow2_scaled(xs)
    sdam = _exact_ssd(sorted(ys))
    groups: Dict[int, List[float]] = {}
    for v, c in zip(ys, cls):
        groups.setdefault(c, []).append(v)
    sdcm = math.fsum(_exact_ssd(sorted(g)) for c, g in sorted(groups.items()))
    gvf = 1.0 - sdcm / sdam if sdam > 0.0 else 1.0
    return HeatmapClasses(md, br, cls, counts, gvf)


def bin_points_square(points: Sequence[Point], cell: float,
                      weights: Optional[Sequence[float]] = None,
                      origin: Point = (0.0, 0.0)) -> Dict[Tuple[int, int], float]:
    """Square-grid binning: cell ``(i, j)`` covers ``[x0 + i c, x0 + (i+1) c)``.
    Returns occupied cells only, sorted; the bin totals sum to the total weight."""
    pts = _as_points(points, "points", allow_empty=True)
    c = _as_positive(cell, "cell")
    ws = [1.0] * len(pts) if weights is None else _as_floats(weights, "weights",
                                                              allow_empty=True)
    if len(ws) != len(pts):
        raise PublicVoiceError("weights and points differ in length.")
    ox, oy = _as_point(origin, "origin")
    acc: Dict[Tuple[int, int], List[float]] = {}
    for (px, py), wv in zip(pts, ws):
        key = (_cell_index((px - ox) / c), _cell_index((py - oy) / c))
        acc.setdefault(key, []).append(wv)
    return {key: math.fsum(acc[key]) for key in sorted(acc)}


def hex_of_point(point: Point, size: float) -> Tuple[int, int]:
    """Axial (q, r) coordinates of the pointy-top hexagon of circumradius
    ``size`` containing ``point`` (cube rounding; the hexagon centred at the
    origin is (0, 0))."""
    px, py = _as_point(point, "point")
    s = _as_positive(size, "size")
    qf = (math.sqrt(3.0) / 3.0 * px - py / 3.0) / s
    rf = (2.0 / 3.0 * py) / s
    _cell_index(qf, "hex coordinate")
    _cell_index(rf, "hex coordinate")
    xf, zf = qf, rf
    yf = -xf - zf
    rx, ry, rz = round(xf), round(yf), round(zf)
    dx, dy, dz = abs(rx - xf), abs(ry - yf), abs(rz - zf)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return (int(rx), int(rz))


def hex_centre(q: int, r: int, size: float) -> Point:
    """Centre of axial hexagon (q, r) (pointy-top)."""
    qq = _as_int(q, "q")
    rr = _as_int(r, "r")
    s = _as_positive(size, "size")
    try:
        cx, cy = s * math.sqrt(3.0) * (qq + rr / 2.0), s * 1.5 * rr
    except OverflowError:
        cx = cy = math.inf
    if math.isinf(cx) or math.isinf(cy):
        raise PublicVoiceError("hex_centre: (q, r) = (%d, %d) at size %r is beyond the "
                               "float range." % (qq, rr, s))
    return (cx, cy)


def bin_points_hex(points: Sequence[Point], size: float,
                   weights: Optional[Sequence[float]] = None) -> Dict[Tuple[int, int], float]:
    """Hexagonal binning (pointy-top, circumradius ``size``), sorted by (q, r)."""
    pts = _as_points(points, "points", allow_empty=True)
    s = _as_positive(size, "size")
    ws = [1.0] * len(pts) if weights is None else _as_floats(weights, "weights",
                                                              allow_empty=True)
    if len(ws) != len(pts):
        raise PublicVoiceError("weights and points differ in length.")
    acc: Dict[Tuple[int, int], List[float]] = {}
    for p, wv in zip(pts, ws):
        acc.setdefault(hex_of_point(p, s), []).append(wv)
    return {key: math.fsum(acc[key]) for key in sorted(acc)}


# ===========================================================================
# 13. SELF-SELECTION DIAGNOSTICS -- representativeness, reweighting, ESS
# ===========================================================================
def _gamma_q(a: float, x: float) -> float:
    """Regularised upper incomplete gamma ``Q(a, x)`` (series / Lentz CF)."""
    if x <= 0.0:
        return 1.0
    lg = math.lgamma(a)
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        ap = a
        for _ in range(10000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-17:
                break
        p = total * math.exp(-x + a * math.log(x) - lg)
        return max(0.0, 1.0 - p)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-16:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - lg) * h)


def chi2_sf(x: float, dof: int) -> float:
    """Chi-square survival function ``P(X >= x)`` for ``dof`` degrees of freedom."""
    xv = _as_nonneg(x, "x")
    k = _as_int(dof, "dof", 1)
    return _gamma_q(0.5 * k, 0.5 * xv)


@dataclass
class RepresentativenessResult:
    """Sample composition against the population, stratum by stratum."""
    strata: List[str]
    sample_counts: Vector
    sample_share: Vector
    population_share: Vector
    representation_ratio: Vector
    dissimilarity: float
    chi2: float
    dof: int
    p_value: float
    uncovered: List[str]
    max_abs_gap: float


def _check_count_dict(d: Any, name: str) -> Dict[str, float]:
    """Validate a dict stratum -> non-negative count."""
    if not isinstance(d, dict) or not d:
        raise PublicVoiceError("%s must be a non-empty dict stratum -> count." % name)
    return {k: _as_nonneg(d[k], "%s[%s]" % (name, k)) for k in _str_keys(d, name)}


def representativeness(sample_counts: Dict[str, float],
                       population: Dict[str, float]) -> RepresentativenessResult:
    """How far the respondents' composition is from the population's.

    ``representation_ratio = sample share / population share`` per stratum
    (1 = proportional), the Duncan dissimilarity index ``1/2 sum |p - q|``
    (the share of respondents who would have to change stratum), and
    Pearson's chi-square goodness of fit of the sample counts to the
    population shares with ``k - 1`` degrees of freedom.  Strata present in
    the population but absent from the sample are ``uncovered``; a sample
    stratum missing from the population is an error."""
    s = _check_count_dict(sample_counts, "sample_counts")
    p = _check_count_dict(population, "population")
    for k in s:
        if k not in p:
            raise PublicVoiceError("sample stratum %r is not in the population." % k)
    strata = sorted(p)
    ns = math.fsum(s.values())
    npop = math.fsum(p.values())
    if ns <= 0.0 or npop <= 0.0:
        raise PublicVoiceError("sample and population totals must be positive.")
    sc = [s.get(k, 0.0) for k in strata]
    ss = [v / ns for v in sc]
    ps = [p[k] / npop for k in strata]
    ratio = [a / b if b > 0.0 else (math.inf if a > 0.0 else 1.0) for a, b in zip(ss, ps)]
    for k, a, b in zip(strata, ss, ps):
        if b == 0.0 and a > 0.0:
            raise PublicVoiceError("stratum %r has respondents but zero population." % k)
    diss = 0.5 * math.fsum(abs(a - b) for a, b in zip(ss, ps))
    # Pearson's sum((o - e)^2 / e) with o = n p_s, e = n p_pop is n times a
    # sum over SHARES: computed that way it cannot overflow on huge counts
    # ((o - e)^2 overflowed above ~1e154, a bare OverflowError).
    live = [(a, b) for a, b in zip(ss, ps) if b > 0.0]
    chi2 = ns * math.fsum((a - b) * (a - b) / b for a, b in live)
    if math.isinf(chi2):
        raise PublicVoiceError("chi-square statistic overflows (sample total %r)." % ns)
    dof = max(1, len(live) - 1)
    unc = [k for k, v in zip(strata, sc) if v == 0.0 and p[k] > 0.0]
    gap = max(abs(a - b) for a, b in zip(ss, ps))
    return RepresentativenessResult(strata, sc, ss, ps, ratio, diss, chi2, dof,
                                    chi2_sf(chi2, dof), unc, gap)


def post_stratification_weights(respondent_strata: Dict[str, str],
                                population: Dict[str, float]) -> Dict[str, float]:
    """Post-stratification weights ``w_h = (N_h / N) / (n_h / n)``.

    Every respondent in stratum ``h`` gets ``w_h``; the weights have mean 1
    (sum ``n``), and the WEIGHTED sample shares equal the population shares
    of the covered strata (renormalised over them) exactly up to rounding."""
    if not isinstance(respondent_strata, dict) or not respondent_strata:
        raise PublicVoiceError("respondent_strata must be a non-empty dict id -> stratum.")
    p = _check_count_dict(population, "population")
    rs = {}
    for rid in _str_keys(respondent_strata, "respondent_strata"):
        st = _as_id(respondent_strata[rid], "stratum of %s" % rid)
        if st not in p or p[st] <= 0.0:
            raise PublicVoiceError("stratum %r has no population." % st)
        rs[rid] = st
    n = len(rs)
    nh: Dict[str, int] = {}
    for rid in sorted(rs):
        nh[rs[rid]] = nh.get(rs[rid], 0) + 1
    covered = sorted(nh)
    npop = math.fsum(p[h] for h in covered)
    wh = {h: (p[h] / npop) / (nh[h] / n) for h in covered}
    return {rid: wh[rs[rid]] for rid in sorted(rs)}


@dataclass
class RakingResult:
    """Iterative proportional fitting (raking) weights."""
    weights: Dict[str, float]
    iterations: int
    converged: bool
    max_margin_error: float


def rake_weights(respondents: Dict[str, Dict[str, str]],
                 margins: Dict[str, Dict[str, float]], max_iter: int = 200,
                 tol: float = 1e-12) -> RakingResult:
    """Raking (IPF) to several population margins at once.

    ``respondents[id][dimension] = level``; ``margins[dimension][level]`` are
    population shares (or counts -- each margin is normalised).  Weights
    start at 1 and are scaled dimension by dimension (in sorted order) until
    every weighted margin matches to ``tol`` (absolute, on shares).  Final
    weights have mean 1."""
    if not isinstance(respondents, dict) or not respondents:
        raise PublicVoiceError("respondents must be a non-empty dict.")
    if not isinstance(margins, dict) or not margins:
        raise PublicVoiceError("margins must be a non-empty dict.")
    it_max = _as_int(max_iter, "max_iter", 1)
    tl = _as_positive(tol, "tol")
    dims = _str_keys(margins, "margins")
    targets: Dict[str, Dict[str, float]] = {}
    for d in dims:
        m = _check_count_dict(margins[d], "margins[%s]" % d)
        tot = math.fsum(m.values())
        if tot <= 0.0:
            raise PublicVoiceError("margin %r has zero total." % d)
        targets[d] = {k: v / tot for k, v in m.items()}
    ids = _str_keys(respondents, "respondents")
    for rid in ids:
        rec = respondents[rid]
        if not isinstance(rec, dict):
            raise PublicVoiceError("respondents[%s] must be a dict." % rid)
        for d in dims:
            lv = rec.get(d)
            if not isinstance(lv, str) or lv not in targets[d]:
                raise PublicVoiceError("respondent %r has no valid level for %r." % (rid, d))
            if targets[d][lv] <= 0.0:
                raise PublicVoiceError("level %r of %r has zero population share." % (lv, d))
    for d in dims:
        present = set(respondents[rid][d] for rid in ids)
        for lv in sorted(targets[d]):
            if targets[d][lv] > 0.0 and lv not in present:
                raise PublicVoiceError("margin %r level %r has no respondent; raking "
                                       "cannot reach it." % (d, lv))
    w = {rid: 1.0 for rid in ids}

    def margin_error() -> float:
        tot = math.fsum(w.values())
        worst = 0.0
        for d in dims:
            for lv in sorted(targets[d]):
                share = math.fsum(w[r] for r in ids if respondents[r][d] == lv) / tot
                worst = max(worst, abs(share - targets[d][lv]))
        return worst

    it = 0
    err = margin_error()
    while err > tl and it < it_max:
        it += 1
        for d in dims:
            tot = math.fsum(w.values())
            for lv in sorted(targets[d]):
                members = [r for r in ids if respondents[r][d] == lv]
                cur = math.fsum(w[r] for r in members) / tot
                if cur > 0.0:
                    f = targets[d][lv] / cur
                    for r in members:
                        w[r] *= f
        err = margin_error()
    tot = math.fsum(w.values())
    n = len(ids)
    w = {rid: w[rid] * n / tot for rid in ids}
    return RakingResult(w, it, err <= tl, err)


def kish_effective_sample_size(weights: Sequence[float]) -> float:
    """Kish's effective sample size ``(sum w)^2 / sum w^2`` (``n`` for equal weights)."""
    ws = _as_floats(weights, "weights")
    if any(v < 0.0 for v in ws):
        raise PublicVoiceError("weights must be non-negative.")
    s = math.fsum(ws)
    if s == 0.0:
        raise PublicVoiceError("weights are all zero.")
    m = max(ws)
    s2 = math.fsum((v / m) ** 2 for v in ws)
    return (s / m) ** 2 / s2


def design_effect(weights: Sequence[float]) -> float:
    """Kish's design effect due to weighting ``n / n_eff = 1 + CV^2``."""
    ws = _as_floats(weights, "weights")
    return len(ws) / kish_effective_sample_size(ws)


def weighted_theme_shares(concerns: Sequence[ConcernRecord],
                          respondent_weights: Optional[Dict[str, float]] = None
                          ) -> Dict[str, float]:
    """Theme shares of the comment ``count`` mass, optionally weighted by each
    record's respondent weight (unknown respondents weigh 1)."""
    cs = _check_concerns(concerns)
    rw = {} if respondent_weights is None else respondent_weights
    if not isinstance(rw, dict):
        raise PublicVoiceError("respondent_weights must be a dict.")
    mass = {th: [] for th in THEMES}  # type: Dict[str, List[float]]
    for c in cs:
        wv = _as_nonneg(rw[c.respondent], "weight") if c.respondent in rw else 1.0
        mass[c.theme].append(c.count * wv)
    tot = math.fsum(v for th in THEMES for v in mass[th])
    if tot <= 0.0:
        raise PublicVoiceError("total weighted mass is zero.")
    return {th: math.fsum(mass[th]) / tot for th in THEMES}


def r_indicator(propensities: Sequence[float]) -> float:
    """Schouten's R-indicator ``1 - 2 S(rho)`` of response propensities
    (1 = perfectly representative response, sample standard deviation)."""
    ps = _as_floats(propensities, "propensities")
    for v in ps:
        if v < 0.0 or v > 1.0:
            raise PublicVoiceError("propensities must lie in [0, 1].")
    if len(ps) < 2:
        raise PublicVoiceError("r_indicator needs at least two propensities.")
    return 1.0 - 2.0 * math.sqrt(_sample_var(ps))


@dataclass
class SpatialRepresentation:
    """Responses per head of population across zones."""
    zones: List[str]
    rate_per_1000: Vector
    ratio_to_overall: Vector
    gini: float
    silent_zones: List[str]


def spatial_representation(zone_counts: Dict[str, float],
                           zone_population: Dict[str, float]) -> SpatialRepresentation:
    """Who spoke, where: response rate per 1000 residents per zone, the ratio to
    the overall rate, the population-weighted Gini of the rates (0 = every
    zone equally heard) and the zones with no response at all."""
    c = _check_count_dict(zone_counts, "zone_counts")
    p = _check_count_dict(zone_population, "zone_population")
    for k in c:
        if k not in p:
            raise PublicVoiceError("zone %r has responses but no population entry." % k)
        if c[k] > 0.0 and p[k] == 0.0:
            raise PublicVoiceError("zone %r has responses but zero population; its "
                                   "responses would silently vanish from the rates." % k)
    zones = sorted(z for z in p if p[z] > 0.0)
    if not zones:
        raise PublicVoiceError("no zone has a positive population.")
    rates = [1000.0 * c.get(z, 0.0) / p[z] for z in zones]
    if any(math.isinf(r) for r in rates):
        raise PublicVoiceError("a response rate overflows (responses / population).")
    overall = 1000.0 * math.fsum(c.get(z, 0.0) for z in zones) / math.fsum(p[z] for z in zones)
    ratio = [r / overall if overall > 0.0 else 0.0 for r in rates]
    ptot = math.fsum(p[z] for z in zones)
    # population SHARES, not raw populations: q1 * q2 * ptot^2 underflowed to
    # 0 for tiny (or overflowed for huge) populations -- the Gini is scale-free
    pops = [p[z] / ptot for z in zones]
    mean_rate = math.fsum(r * q for r, q in zip(rates, pops))
    if mean_rate > 0.0:
        g = math.fsum(q1 * q2 * abs(r1 - r2) for r1, q1 in zip(rates, pops)
                      for r2, q2 in zip(rates, pops)) / (2.0 * mean_rate)
    else:
        g = 0.0
    silent = [z for z, r in zip(zones, rates) if r == 0.0]
    return SpatialRepresentation(zones, rates, ratio, g, silent)


# ===========================================================================
# 14. FRONTEND PAYLOADS
# ===========================================================================
def _json_safe(x: Any) -> Any:
    """Replace non-finite floats with ``None`` so the dict survives ``json.dumps``,
    and round finite floats to 10 SIGNIFICANT digits.

    Significant, not decimal, places: a fixed ``round(x, 8)`` is an absolute
    quantum, so an ``"area"``-normalised field (1/m^2, ~1e-7 at a 1 km
    bandwidth) or any data rescaled by 1e-6 would be flattened to 0 in the
    payload.  Relative rounding keeps every scale equally precise."""
    if isinstance(x, float):
        if math.isinf(x) or math.isnan(x):
            return None
        return float("%.10g" % x)
    return x


def _check_optional(x: Any, cls: Any, name: str) -> None:
    """An optional payload block must be None or an instance of ``cls``."""
    if x is not None and not isinstance(x, cls):
        raise PublicVoiceError("%s must be a %s (got %s)."
                               % (name, cls.__name__, type(x).__name__))


def _check_hotspots(hotspots: Any) -> Optional[List[Hotspot]]:
    """None, or a sequence of Hotspot whose ``by_theme`` maps known themes to
    finite non-negative masses (a theme it omits has mass 0) and whose
    ``n_comments`` is a finite number.  Returned with ``by_theme`` completed
    over THEMES (copies; the caller's objects are untouched)."""
    if hotspots is None:
        return None
    hs = _as_seq(hotspots, "hotspots")
    out = []
    for h in hs:
        _check_optional(h, Hotspot, "hotspots item")
        if h is None:
            raise PublicVoiceError("hotspots must not contain None.")
        if not isinstance(h.by_theme, dict):
            raise PublicVoiceError("hotspot %r: by_theme must be a dict." % (h.hid,))
        for th in _str_keys(h.by_theme, "by_theme"):
            _check_choice(th, THEMES, "by_theme theme")
        by = {th: (_as_nonneg(h.by_theme[th], "by_theme[%s]" % th)
                   if th in h.by_theme else 0.0) for th in THEMES}
        _as_float(h.n_comments, "hotspot n_comments")
        out.append(Hotspot(_as_id(h.hid, "hotspot id"), h.kind,
                           [_as_id(u, "hotspot unit") for u in _as_seq(h.unit_ids, "units")],
                           h.n_records, h.n_comments, h.n_respondents, by,
                           h.mean_severity, h.peak_value))
    return out


def _js_list(xs: Sequence[Any]) -> List[Any]:
    """``_json_safe`` over a sequence."""
    return [_json_safe(v) for v in xs]


def build_public_voice_payload(
        network: VoiceNetwork, concerns: Sequence[ConcernRecord], *,
        field: Optional[ConcernField] = None,
        objective: Optional[VoiceObjective] = None,
        pairing: Optional[Dict[str, ThemeMetricPair]] = None,
        duplicates: Optional[DuplicateReport] = None,
        moran: Optional[GlobalStatistic] = None,
        local_g: Optional[LocalGResult] = None,
        local_moran: Optional[LocalMoranResult] = None,
        hotspots: Optional[Sequence[Hotspot]] = None,
        heatmap: Optional[HeatmapClasses] = None,
        represent: Optional[RepresentativenessResult] = None,
        ingest: Optional[IngestReport] = None,
        max_items: int = 200) -> Dict[str, Any]:
    """Frontend-ready JSON dict: the network, the (anonymised) concerns and
    whichever analyses were run.  Nothing is recomputed; every block is
    optional; lists are truncated to ``max_items`` in a fixed order."""
    if not isinstance(network, VoiceNetwork):
        raise PublicVoiceError("build_public_voice_payload needs a VoiceNetwork.")
    cs = _check_concerns(concerns, allow_empty=True)
    cap = _as_int(max_items, "max_items", 1)
    pr = _check_pairing(pairing)
    if field is not None:
        _check_field(field, network)
    for blk, cls, nm in ((field, ConcernField, "field"), (objective, VoiceObjective, "objective"),
                         (duplicates, DuplicateReport, "duplicates"),
                         (moran, GlobalStatistic, "moran"), (local_g, LocalGResult, "local_g"),
                         (local_moran, LocalMoranResult, "local_moran"),
                         (heatmap, HeatmapClasses, "heatmap"),
                         (represent, RepresentativenessResult, "represent"),
                         (ingest, IngestReport, "ingest")):
        _check_optional(blk, cls, nm)
    hotspots = _check_hotspots(hotspots)
    payload: Dict[str, Any] = {
        "network": {k: _json_safe(v) for k, v in network.summary().items()},
        "edges": [{"id": e, "u": network.endpoints(e)[0], "v": network.endpoints(e)[1],
                   "lengthM": _json_safe(network.length(e))}
                  for e in network.edge_ids[:cap]],
        "concerns": [{"id": c.cid, "theme": c.theme, "severity": _json_safe(c.severity),
                      "x": _json_safe(c.x), "y": _json_safe(c.y), "mode": c.mode,
                      "source": c.source, "count": _json_safe(c.count)}
                     for c in cs[:cap]],
        "themes": [{"theme": th, "label": THEME_TAXONOMY[th].label,
                    "metric": pr[th].metric, "inObjective": pr[th].measurable,
                    "records": sum(1 for c in cs if c.theme == th)} for th in THEMES],
    }
    if field is not None:
        payload["field"] = {
            "kernel": field.kernel, "bandwidthM": _json_safe(field.bandwidth),
            "normalisation": field.normalisation, "distance": field.distance,
            "edgeIds": list(field.edge_ids[:cap]),
            "values": {th: _js_list(field.values[th][:cap]) for th in THEMES},
            "total": _js_list(field.total_vector()[:cap]),
            "unattached": list(field.unattached[:cap]),
        }
        payload["heatmapLayers"] = [
            {"edgeId": h.edge_id, "theme": h.theme, "value": _json_safe(h.value),
             "inObjective": h.in_objective, "metric": h.metric}
            for h in heatmap_layers(field, pr)[:cap]]
    if objective is not None:
        payload["objective"] = {
            "jVoice": _json_safe(objective.total),
            "byTheme": {k: _json_safe(objective.by_theme[k]) for k in sorted(objective.by_theme)},
            "included": list(objective.included_themes),
            "excluded": list(objective.excluded_themes),
            "excludedMass": {k: _json_safe(objective.excluded_mass[k])
                             for k in sorted(objective.excluded_mass)},
            "normalisation": objective.normalisation,
        }
    if duplicates is not None:
        payload["duplicates"] = {
            "method": duplicates.method, "threshold": _json_safe(duplicates.threshold),
            "nTexts": duplicates.n_texts, "nCandidates": duplicates.n_candidates,
            "groups": [list(g) for g in duplicates.groups[:cap]],
            "pairs": [{"a": a, "b": b, "similarity": _json_safe(s)}
                      for a, b, s in duplicates.pairs[:cap]],
        }
    if moran is not None:
        payload["globalAutocorrelation"] = {
            "statistic": moran.name, "value": _json_safe(moran.value),
            "expected": _json_safe(moran.expected), "zRandom": _json_safe(moran.z_random),
            "pRandom": _json_safe(moran.p_random), "pSim": _json_safe(moran.p_sim),
            "n": moran.n}
    if local_g is not None:
        payload["getisOrd"] = [
            {"id": i, "z": _json_safe(z), "pAdjusted": _json_safe(p), "label": lb}
            for i, z, p, lb in list(zip(local_g.ids, local_g.z, local_g.p_adjusted,
                                        local_g.labels))[:cap]]
    if local_moran is not None:
        payload["localMoran"] = [
            {"id": i, "I": _json_safe(v), "z": _json_safe(z), "quadrant": q, "label": lb}
            for i, v, z, q, lb in list(zip(local_moran.ids, local_moran.i, local_moran.z,
                                           local_moran.quadrant, local_moran.labels))[:cap]]
    if hotspots is not None:
        payload["hotspots"] = [
            {"id": h.hid, "kind": h.kind, "units": list(h.unit_ids),
             "records": h.n_records, "comments": _json_safe(h.n_comments),
             "respondents": h.n_respondents,
             "byTheme": {k: _json_safe(h.by_theme[k]) for k in THEMES if h.by_theme[k] > 0.0},
             "meanSeverity": _json_safe(h.mean_severity), "peak": _json_safe(h.peak_value)}
            for h in list(hotspots)[:cap]]
    if heatmap is not None:
        payload["classification"] = {
            "method": heatmap.method, "breaks": _js_list(heatmap.breaks),
            "counts": list(heatmap.counts), "gvf": _json_safe(heatmap.gvf)}
    if represent is not None:
        payload["representativeness"] = {
            "strata": list(represent.strata),
            "sampleShare": _js_list(represent.sample_share),
            "populationShare": _js_list(represent.population_share),
            "ratio": _js_list(represent.representation_ratio),
            "dissimilarity": _json_safe(represent.dissimilarity),
            "chi2": _json_safe(represent.chi2), "dof": represent.dof,
            "pValue": _json_safe(represent.p_value), "uncovered": list(represent.uncovered)}
    if ingest is not None:
        payload["ingest"] = {
            "nIn": ingest.n_in, "nOut": ingest.n_out, "nRejected": len(ingest.rejected),
            "piiFieldsDropped": {k: ingest.pii_fields_dropped[k]
                                 for k in sorted(ingest.pii_fields_dropped)},
            "redactions": {k: ingest.redactions[k] for k in sorted(ingest.redactions)},
            "autoClassified": ingest.auto_classified, "jittered": ingest.jittered,
            "coarsened": ingest.coarsened}
    return payload


def build_a2ui_payload(network: VoiceNetwork, concerns: Sequence[ConcernRecord], *,
                       field: Optional[ConcernField] = None,
                       objective: Optional[VoiceObjective] = None,
                       improvement: Optional[float] = None,
                       hotspots: Optional[Sequence[Hotspot]] = None,
                       represent: Optional[RepresentativenessResult] = None,
                       effective_n: Optional[float] = None,
                       top_n: int = 10) -> Dict[str, Any]:
    """Visualisation-ready dict: stat cards, chart series and the hotspot table."""
    if not isinstance(network, VoiceNetwork):
        raise PublicVoiceError("build_a2ui_payload needs a VoiceNetwork.")
    cs = _check_concerns(concerns, allow_empty=True)
    n = _as_int(top_n, "top_n", 1)
    if field is not None:
        _check_field(field, network)
    for blk, cls, nm in ((field, ConcernField, "field"), (objective, VoiceObjective, "objective"),
                         (represent, RepresentativenessResult, "represent")):
        _check_optional(blk, cls, nm)
    hotspots = _check_hotspots(hotspots)
    total = math.fsum(c.count for c in cs)
    cards: List[Dict[str, Any]] = [
        {"label": "Comments", "value": _json_safe(total), "unit": ""},
        {"label": "Records", "value": len(cs), "unit": ""},
        {"label": "Respondents", "value": len(set(c.respondent for c in cs if c.respondent)),
         "unit": ""},
        {"label": "Streets", "value": network.n_edges, "unit": ""},
    ]
    if objective is not None:
        cards.append({"label": "J_voice", "value": _json_safe(objective.total),
                      "unit": ""})
    if improvement is not None:
        cards.append({"label": "Public-voice improvement",
                      "value": _json_safe(round(100.0 * _as_float(improvement,
                                                                  "improvement"), 2)),
                      "unit": "%"})
    if hotspots is not None:
        cards.append({"label": "Hotspots", "value": len(list(hotspots)), "unit": ""})
    if represent is not None:
        cards.append({"label": "Dissimilarity",
                      "value": _json_safe(round(100.0 * represent.dissimilarity, 2)),
                      "unit": "%"})
    if effective_n is not None:
        cards.append({"label": "Effective n",
                      "value": _json_safe(round(_as_nonneg(effective_n, "effective_n"), 2)),
                      "unit": ""})
    counts = theme_counts(cs, weighted=True)
    charts: Dict[str, Any] = {
        "themeCounts": [{"theme": th, "label": THEME_TAXONOMY[th].label,
                         "count": _json_safe(counts[th])} for th in THEMES],
    }
    if field is not None:
        tot = field.total_vector()
        order = sorted(range(len(tot)), key=lambda k: (-tot[k], field.edge_ids[k]))[:n]
        charts["topEdges"] = [{"edgeId": field.edge_ids[k], "value": _json_safe(tot[k])}
                              for k in order]
    table = []
    if hotspots is not None:
        for h in sorted(hotspots, key=lambda h: (-h.n_comments, h.hid))[:n]:
            top = sorted(((h.by_theme[th], th) for th in THEMES if h.by_theme[th] > 0.0),
                         key=lambda t: (-t[0], t[1]))
            table.append({"id": h.hid, "streets": len(h.unit_ids),
                          "comments": _json_safe(h.n_comments),
                          "topTheme": top[0][1] if top else None})
    return {"statCards": cards, "charts": charts, "hotspotTable": table}


# ===========================================================================
# 15. DEMO
# ===========================================================================
_DEMO_TEXTS: Tuple[Tuple[str, Optional[str], float, Tuple[float, float]], ...] = (
    # (text, theme or None for auto, severity, cluster centre)
    ("Please put a zebra crossing outside the school on Mill Rd, kids wait ages.",
     None, 4.0, (240.0, 120.0)),
    ("please put a zebra crossing outside the school on Mill Road -- kids wait ages!",
     None, 4.0, (240.0, 120.0)),
    ("Please put a zebra crossing outside the school on Mill Rd; kids wait ages",
     None, 5.0, (240.0, 120.0)),
    ("Waiting at the lights for the green man takes forever, I gave up and crossed.",
     None, 3.0, (250.0, 110.0)),
    ("Cars speeding past the school gate, near miss last week. Call 07700 900123.",
     None, 5.0, (230.0, 130.0)),
    ("Dangerous junction, a cyclist was hit here. My name is Jane Doe.",
     None, 5.0, (240.0, 140.0)),
    ("Rat running through our street every morning, satnav sends lorries this way.",
     None, 4.0, (0.0, 180.0)),
    ("Through traffic cut through on the residential road -- see www.example.org/x",
     None, 3.0, (10.0, 200.0)),
    ("Shortcut traffic from the A road at 60 mph. email me jo@example.com",
     None, 4.0, (5.0, 240.0)),
    ("Fumes and idling buses outside 12 Station Road make my asthma worse.",
     None, 4.0, (420.0, 360.0)),
    ("Air quality is terrible by the bus station, pollution all day.",
     None, 3.0, (440.0, 340.0)),
    ("The cycle lane just ends at the bridge, no safe bike route to town.",
     None, 4.0, (360.0, 60.0)),
    ("Missing cycle route between the park and the station. Segregated please.",
     None, 3.0, (380.0, 40.0)),
    ("No dropped kerb for wheelchair users at the corner, buggy impossible.",
     None, 4.0, (120.0, 360.0)),
    ("Pavement parking blocks the footway every evening.", None, 3.0, (300.0, 300.0)),
    ("Not enough permit parking spaces for residents.", None, 2.0, (310.0, 280.0)),
    ("The dual carriageway is a wall of traffic, it divides the estate.",
     None, 4.0, (480.0, 180.0)),
    ("HGV noise at night, the rumble wakes us up, can't sleep. Postcode CB1 2AB.",
     None, 3.0, (470.0, 200.0)),
    ("Lovely new benches, thank you.", None, 1.0, (200.0, 200.0)),
    ("Kids wait ages to cross outside the school, please add a crossing.",
     "crossing_delay", 4.0, (235.0, 125.0)),
)

_DEMO_STRATA = ("18-34", "35-64", "65+")
_DEMO_POPULATION = {"18-34": 3200.0, "35-64": 4100.0, "65+": 2700.0}


def _demo_raw(seed: int = 11) -> List[Dict[str, Any]]:
    """Raw submissions with PII, duplicates and jittered repeats."""
    rng = random.Random(seed)
    raw: List[Dict[str, Any]] = []
    k = 0
    for rep in range(2):
        for text, theme, sev, (cx, cy) in _DEMO_TEXTS:
            if rep == 1 and rng.random() < 0.5:
                continue
            k += 1
            rec: Dict[str, Any] = {
                "id": "c%03d" % k, "text": text, "severity": sev,
                "x": cx + rng.uniform(-15.0, 15.0), "y": cy + rng.uniform(-15.0, 15.0),
                "author": "resident-%02d" % rng.randrange(14),
                "email": "r%d@example.com" % k,
                "source": "map_pin" if k % 3 else "survey",
                "stratum": _DEMO_STRATA[rng.randrange(3) if k % 4 else 0],
            }
            if theme is not None:
                rec["theme"] = theme
            raw.append(rec)
    return raw


def _demo_metrics(network: VoiceNetwork, factor: float) -> Dict[str, Dict[str, float]]:
    """Synthetic per-edge metrics; ``factor`` scales the crossing / safety burden
    near the school (the intervention lowers it)."""
    out: Dict[str, Dict[str, float]] = {}
    base = {"ped_crossing_delay_s": 25.0, "collision_risk": 0.4,
            "through_traffic_veh_h": 300.0, "no2_ug_m3": 32.0, "cycle_route_gap": 0.5,
            "accessibility_score": 60.0, "severance_index": 0.3, "noise_lden_db": 62.0}
    for metric in sorted(base):
        vals = {}
        for k, e in enumerate(network.edge_ids):
            mx, my = network.midpoint(e)
            v = base[metric] * (1.0 + 0.05 * (k % 7))
            near_school = math.hypot(mx - 240.0, my - 120.0) < 130.0
            if near_school and metric in ("ped_crossing_delay_s", "collision_risk"):
                v *= factor
            vals[e] = v
        out[metric] = vals
    return out


def _demo() -> None:
    """Walk the whole public-voice pipeline on a synthetic district."""
    print("=" * 70)
    print("  public_voice -- consultation feedback as a field over the network")
    print("=" * 70)

    # -- 1-2. ingestion and anonymisation ----------------------------------
    print("\n-- 1-2. taxonomy, ingestion and anonymisation --")
    print("   %d themes; measurable: %s" % (len(THEMES), ", ".join(
        th for th in THEMES if THEME_TAXONOMY[th].measurable)))
    info = THEME_TAXONOMY["parking"]
    assert isinstance(info, ThemeInfo) and not info.measurable
    print("   modes %s | sources %s | PII fields %d"
          % ("/".join(MODES), "/".join(SOURCES[:3]), len(PII_FIELDS)))
    ts = classify_theme("Cars speeding and a near miss at the junction")
    assert isinstance(ts, ThemeScore)
    print("   classify_theme -> %s (confident=%s)" % (ts.theme, ts.confident))
    red = redact_text("Call 07700 900123 or jo@example.com, I live at 12 Mill Road.")
    assert isinstance(red, RedactionResult)
    print("   redact_text -> %r (%d redactions)" % (red.text, red.total))
    print("   pseudonymise -> %s" % pseudonymise("resident-01", "district-salt"))
    print("   coarsen_location(123.4, 87.6, 50) -> %s"
          % (coarsen_location(123.4, 87.6, 50.0),))
    jx, jy = jitter_location(100.0, 100.0, 5.0, 15.0, seed=3, key="c001")
    print("   jitter_location -> (%.3f, %.3f)" % (jx, jy))
    raw = _demo_raw()
    ing = ingest_concerns(raw, salt="district-salt", jitter=(2.0, 6.0), seed=5)
    assert isinstance(ing, IngestResult) and isinstance(ing.report, IngestReport)
    concerns = ing.records
    print("   ingested %d/%d records, %d auto-classified, redactions %s"
          % (ing.report.n_out, ing.report.n_in, ing.report.auto_classified,
             ing.report.redactions))
    kan = k_anonymity_cells([c.point for c in concerns], 100.0, 2)
    assert isinstance(kan, KAnonymityReport)
    print("   k-anonymity (k=2, 100 m cells): %d cells, %d violating"
          % (len(kan.cell_counts), len(kan.violating_cells)))
    extra = make_concern("z001", "noise", 2.0, 480.0, 190.0, mode="freight",
                         source="letter", text="Lorries at night")
    assert isinstance(extra, ConcernRecord)
    concerns = sorted(concerns + [extra], key=lambda c: c.cid)
    tc = theme_counts(concerns)
    summ = theme_summary(concerns)
    assert all(isinstance(s, ThemeSummary) for s in summ)
    print("   theme counts: %s" % {k: int(v) for k, v in tc.items() if v})

    # -- 8. near duplicates -------------------------------------------------
    print("\n-- 8. near-duplicate detection --")
    print("   fnv1a_64('') = %#x, fnv1a_32('a') = %#x, stable_hash64('x') = %d"
          % (fnv1a_64(b""), fnv1a_32(b"a"), stable_hash64("x")))
    a_txt = concerns[0].text
    print("   canonical_form -> %r" % canonical_form(a_txt))
    sh1 = char_shingles(a_txt, 5)
    sh2 = char_shingles(concerns[1].text, 5)
    print("   char shingles %d / word shingles %d, exact jaccard %.3f"
          % (len(sh1), len(word_shingles(a_txt, 2)), jaccard(sh1, sh2)))
    mh = MinHasher(num_perm=128, seed=1)
    print("   minhash estimate %.3f | lsh P(0.8; 16x8) = %.3f | threshold %.3f"
          % (MinHasher.estimate(mh.signature(concerns[0].text),
                                mh.signature(concerns[1].text)),
             lsh_probability(0.8, 16, 8), lsh_threshold(16, 8)))
    sigs = {c.cid: mh.signature(c.text) for c in concerns[:6]}
    print("   lsh candidate pairs among 6: %d" % len(lsh_candidate_pairs(sigs, 32, 4)))
    print("   simhash hamming %d"
          % hamming_distance(simhash(concerns[0].text), simhash(concerns[1].text)))
    texts = {c.cid: c.text for c in concerns}
    for meth in DUPLICATE_METHODS:
        rep = find_near_duplicates(texts, meth, threshold=0.6 if meth == "minhash" else None)
        print("   %-9s -> %d groups" % (meth, len(rep.groups)))
    dup = find_near_duplicates(texts, "minhash", threshold=0.6)
    assert isinstance(dup, DuplicateReport)
    collapsed = collapse_duplicates(concerns, dup, policy="sqrt", max_distance=80.0)
    print("   collapse (%s): %d -> %d records" % ("/".join(COLLAPSE_POLICIES), len(concerns),
                                                 len(collapsed)))

    # -- 3. network ---------------------------------------------------------
    print("\n-- 3. network and point-to-edge attachment --")
    net = network_from_grid(5, 4, 120.0, name="demo district")
    assert isinstance(net, VoiceNetwork)
    pr = project_point_to_segment((50.0, 10.0), (0.0, 0.0), (120.0, 0.0))
    assert isinstance(pr, Projection)
    print("   %s; projection t=%.3f d=%.1f; point-segment %.1f"
          % (net.summary(), pr.t, pr.distance,
             point_segment_distance((50.0, 10.0), (0.0, 0.0), (120.0, 0.0))))
    att = attach_point(net, (250.0, 115.0))
    assert isinstance(att, Attachment)
    print("   attach (250, 115) -> edge %s at %.1f m (snap %.1f m)"
          % (att.edge_id, att.offset_m, att.distance_m))
    atts, lost = attach_concerns(net, collapsed, max_distance=60.0)
    nd = dijkstra(net, {"n0_0": 0.0})
    d_edges = network_distances_to_edges(net, att)
    print("   attached %d, unattached %d; dijkstra to n4_3 = %.0f m; edges reached %d"
          % (len(atts), len(lost), nd["n4_3"], len(d_edges)))
    att2 = attach_point(net, (10.0, 330.0))
    assert att2 is not None
    print("   network distance between two pins: %.1f m"
          % network_distance_between(net, att, att2))

    # -- 4-5. kernels and bandwidths ---------------------------------------
    print("\n-- 4-5. kernels and bandwidth selection --")
    for kn in KERNELS:
        km = kernel_moments(kn)
        assert isinstance(km, KernelMoments)
        print("   %-12s k(0.5)=%.4f K1(0)=%.4f K2(0)=%.4f eff=%.4f support=%s"
              % (kn, kernel_profile(kn, 0.5), kernel_density_1d(kn, 0.0),
                 kernel_density_2d(kn, 0.0), km.efficiency_1d, kernel_support(kn, 50.0)))
    print("   normalisations %s; weight(d=40, h=80, density) = %.6f"
          % (KERNEL_NORMALISATIONS, kernel_weight("quartic", 40.0, 80.0, "density")))
    pts = [c.point for c in collapsed]
    print("   silverman 1-D %.2f | spatial %.2f | LOO-LL(80) %.2f | LSCV(80) %.3e"
          % (silverman_bandwidth([p[0] for p in pts]), spatial_silverman_bandwidth(pts),
             loo_log_likelihood(pts, "gaussian", 80.0), lscv_score(pts, 80.0)))
    cvb = cv_bandwidth(pts, "gaussian", method="likelihood")
    cvl = cv_bandwidth(pts, "gaussian", method="lscv")
    assert isinstance(cvb, BandwidthResult)
    print("   cv (likelihood) %.2f m, cv (lscv) %.2f m" % (cvb.bandwidth, cvl.bandwidth))
    ab = abramson_bandwidths(pts, "gaussian", pilot=cvb.bandwidth)
    kb = knn_bandwidths(pts, 4)
    assert isinstance(ab, AdaptiveBandwidths)
    print("   abramson h in [%.1f, %.1f], knn h in [%.1f, %.1f]; kde max %.3e"
          % (min(ab.bandwidths), max(ab.bandwidths), min(kb.bandwidths), max(kb.bandwidths),
             max(kde_at_points(pts, "gaussian", 80.0))))

    # -- 6. concern field ---------------------------------------------------
    print("\n-- 6. the concern field --")
    fe = concern_field(net, collapsed, kernel="quartic", bandwidth=120.0)
    fn = concern_field(net, collapsed, kernel="quartic", bandwidth=120.0,
                       distance="network", max_attach_distance=60.0)
    fa = concern_field(net, collapsed, kernel="gaussian",
                       bandwidths={c.cid: h for c, h in zip(collapsed, ab.bandwidths)})
    assert isinstance(fe, ConcernField)
    tot = fe.total_vector()
    top_e = max(range(len(tot)), key=lambda k: (tot[k], -k))
    print("   modes %s / refs %s" % (DISTANCE_MODES, EDGE_REFERENCES))
    print("   euclidean peak edge %s = %.2f; network field mass %.2f; adaptive mass %.2f"
          % (fe.edge_ids[top_e], tot[top_e], math.fsum(fn.total_vector()),
             math.fsum(fa.total_vector())))
    contrib = concern_contributions(net, collapsed, fe.edge_ids[top_e], top=3,
                                    kernel="quartic", bandwidth=120.0)
    print("   top contributors there: %s" % [(c, t, round(v, 2)) for c, t, v in contrib])
    surf = concern_surface(collapsed, -100.0, -100.0, 20.0, 35, 28, kernel="gaussian",
                           bandwidth=40.0, use_counts=False,
                           weights={c.cid: 1.0 / c.severity if c.severity else 0.0
                                    for c in collapsed})
    assert isinstance(surf, ConcernSurface)
    print("   surface %dx%d integrates to %.3f (records: %d)"
          % (surf.nx, surf.ny, surf.integral(), len(collapsed)))

    # -- 7. J_voice ---------------------------------------------------------
    print("\n-- 7. theme-metric pairing and J_voice --")
    pairing = default_pairing()
    custom = make_pairing({"noise": None}, {"accessibility": "higher"})
    assert isinstance(custom["noise"], ThemeMetricPair) and not custom["noise"].measurable
    baseline = _demo_metrics(net, 1.0)
    design = _demo_metrics(net, 0.6)
    print("   normalise_metric([2, 6], [4, 3]) = %s; modes %s"
          % (normalise_metric([2.0, 6.0], [4.0, 3.0]), NORMALISATION_MODES))
    jb = voice_objective(fe, baseline, baseline, pairing)
    jx = voice_objective(fe, design, baseline, pairing)
    assert isinstance(jb, VoiceObjective)
    print("   J_voice(baseline) = %.3f, J_voice(design) = %.3f" % (jb.total, jx.total))
    print("   excluded from J_voice (shown on heat map): %s"
          % {k: round(v, 2) for k, v in jx.excluded_mass.items()})
    imp0 = voice_improvement(fe, baseline, baseline, pairing)
    imp = voice_improvement(fe, design, baseline, pairing)
    print("   improvement: baseline %.1f, design %.4f" % (imp0, imp))
    layers = heatmap_layers(fe, pairing)
    assert all(isinstance(h, HeatmapCell) for h in layers)
    print("   heat-map cells %d (%d outside the objective)"
          % (len(layers), sum(1 for h in layers if not h.in_objective)))

    # -- 9-10. weights and global autocorrelation --------------------------
    print("\n-- 9-10. spatial weights and global autocorrelation --")
    w_net = weights_from_network(net)
    mids = [net.midpoint(e) for e in net.edge_ids]
    w_band = weights_distance_band(net.edge_ids, mids, 130.0)
    w_knn = weights_knn(net.edge_ids, mids, 3)
    w_ker = weights_kernel(net.edge_ids, mids, 250.0)
    w_row = row_standardise(w_net)
    w_mat = weights_from_matrix(["a", "b", "c"], [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert isinstance(w_net, SpatialWeights)
    print("   network S0=%g, band S0=%g, knn symmetric=%s, kernel S0=%.2f, row S0=%g, "
          "matrix S0=%g" % (w_net.s0, w_band.s0, w_knn.is_symmetric(), w_ker.s0, w_row.s0,
                            w_mat.s0))
    mi = morans_i(tot, w_net, permutations=199, seed=3)
    gc = gearys_c(tot, w_net)
    gg = general_g(tot, w_band)
    assert isinstance(mi, GlobalStatistic)
    print("   Moran I = %.4f (E = %.4f, z = %.2f, p_sim = %.3f)"
          % (mi.value, mi.expected, mi.z_random, mi.p_sim))
    print("   Geary C = %.4f, General G = %.4f (E = %.4f)" % (gc.value, gg.value, gg.expected))
    dist = permutation_distribution(tot[:6], weights_knn(net.edge_ids[:6], mids[:6], 2),
                                    exhaustive=True)
    print("   exhaustive permutations n=6: %d values, mean %.6f (exact -1/5)"
          % (len(dist), math.fsum(dist) / len(dist)))
    window = (-60.0, -60.0, 540.0, 420.0)
    rk = ripleys_k(pts, [40.0, 80.0, 120.0], window, "isotropic")
    rb = ripleys_k(pts, [40.0, 80.0, 120.0], window, "border")
    env = csr_envelope(len(pts), [40.0, 80.0, 120.0], window, simulations=19, seed=2)
    assert isinstance(rk, RipleyResult) and isinstance(env, Envelope)
    print("   Ripley L(r)-r iso %s border %s, CSR upper %s (%s)"
          % ([round(v, 1) for v in rk.h], [round(v, 1) for v in rb.h],
             [round(v, 1) for v in env.upper], "/".join(RIPLEY_CORRECTIONS)))
    print("   isotropic edge weight at a corner: %.4f"
          % ripley_edge_weight((-60.0, -60.0), 50.0, window))

    # -- 11. hotspots -------------------------------------------------------
    print("\n-- 11. hotspots --")
    pv = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    print("   BH adjusted: %s" % [round(v, 4) for v in benjamini_hochberg(pv)])
    print("   BH rejects at q=0.05: %d" % sum(bh_reject(pv, 0.05)))
    lg = local_getis_ord(tot, w_band, star=True, alpha=0.1)
    lgp = local_getis_ord(tot, w_band, star=True, permutations=199, seed=4)
    lg0 = local_getis_ord(tot, w_band, star=False)
    lm = local_morans_i(tot, w_band, permutations=199, seed=4)
    assert isinstance(lg, LocalGResult) and isinstance(lm, LocalMoranResult)
    print("   G*: %d hot, %d cold (permutation: %d hot); G: %d hot; LISA labels %s"
          % (lg.labels.count("hot"), lg.labels.count("cold"), lgp.labels.count("hot"),
             lg0.labels.count("hot"),
             {q: lm.labels.count(q) for q in ("HH", "LL", "HL", "LH", "ns")}))
    clusters = hotspot_clusters(lg.ids, lg.labels, w_band, "hot")
    unit_of = {cid: a.edge_id for cid, a in atts.items()}
    hs = comment_counts_per_hotspot(clusters, collapsed, unit_of,
                                    {e: v for e, v in zip(fe.edge_ids, tot)})
    for h in hs:
        assert isinstance(h, Hotspot)
        print("   %s: %d streets, %d records, %.1f comments, top theme %s"
              % (h.hid, len(h.unit_ids), h.n_records, h.n_comments,
                 max(THEMES, key=lambda th: (h.by_theme[th], -THEMES.index(th)))))

    # -- 12. heat-map classes -----------------------------------------------
    print("\n-- 12. heat-map classification --")
    jr = jenks_breaks(tot, 4)
    assert isinstance(jr, JenksResult)
    print("   jenks breaks %s (GVF %.3f)" % ([round(b, 2) for b in jr.breaks], jr.gvf))
    print("   equal %s | quantile %s"
          % ([round(b, 2) for b in equal_interval_breaks(tot, 4)],
             [round(b, 2) for b in quantile_breaks(tot, 4)]))
    hc = heatmap_classes(tot, 4, "jenks")
    assert isinstance(hc, HeatmapClasses)
    print("   classes %s counts %s" % (classify_values(tot[:6], jr.breaks), hc.counts))
    sq = bin_points_square(pts, 100.0)
    hx = bin_points_hex(pts, 60.0)
    q0, r0 = hex_of_point(pts[0], 60.0)
    print("   %d square bins, %d hex bins; first pin in hex (%d, %d) centred %s"
          % (len(sq), len(hx), q0, r0, tuple(round(v, 1) for v in hex_centre(q0, r0, 60.0))))

    # -- 13. self-selection -------------------------------------------------
    print("\n-- 13. self-selection diagnostics --")
    resp_stratum: Dict[str, str] = {}
    for c in collapsed:
        if c.respondent and c.stratum:
            resp_stratum.setdefault(c.respondent, c.stratum)
    counts: Dict[str, float] = {}
    for rid in sorted(resp_stratum):
        counts[resp_stratum[rid]] = counts.get(resp_stratum[rid], 0.0) + 1.0
    rp = representativeness(counts, _DEMO_POPULATION)
    assert isinstance(rp, RepresentativenessResult)
    print("   sample shares %s vs population %s" % ([round(v, 3) for v in rp.sample_share],
                                                    [round(v, 3) for v in rp.population_share]))
    print("   dissimilarity %.3f, chi2 %.2f on %d dof, p = %.3f (chi2_sf check %.3f)"
          % (rp.dissimilarity, rp.chi2, rp.dof, rp.p_value, chi2_sf(rp.chi2, rp.dof)))
    psw = post_stratification_weights(resp_stratum, _DEMO_POPULATION)
    ess = kish_effective_sample_size(list(psw.values()))
    print("   post-stratified: n = %d, Kish n_eff = %.2f, deff = %.3f"
          % (len(psw), ess, design_effect(list(psw.values()))))
    rk_in = {rid: {"age": resp_stratum[rid],
                   "area": "north" if int(rid[:2], 16) % 2 else "south"}
             for rid in sorted(resp_stratum)}
    rr = rake_weights(rk_in, {"age": _DEMO_POPULATION, "area": {"north": 0.5, "south": 0.5}})
    assert isinstance(rr, RakingResult)
    print("   raking converged=%s in %d iterations (max margin error %.1e)"
          % (rr.converged, rr.iterations, rr.max_margin_error))
    shares = weighted_theme_shares(collapsed, psw)
    print("   weighted crossing-delay share %.3f; R-indicator %.3f"
          % (shares["crossing_delay"], r_indicator([0.12, 0.25, 0.31, 0.08, 0.2])))
    zone_counts = {"north": 14.0, "south": 9.0, "east": 0.0}
    sr = spatial_representation(zone_counts, {"north": 4000.0, "south": 5000.0,
                                              "east": 1000.0})
    assert isinstance(sr, SpatialRepresentation)
    print("   response per 1000: %s, Gini %.3f, silent %s"
          % ([round(v, 2) for v in sr.rate_per_1000], sr.gini, sr.silent_zones))

    # -- 14. payloads -------------------------------------------------------
    print("\n-- 14. payloads --")
    payload = build_public_voice_payload(
        net, collapsed, field=fe, objective=jx, pairing=pairing, duplicates=dup,
        moran=mi, local_g=lg, local_moran=lm, hotspots=hs, heatmap=hc, represent=rp,
        ingest=ing.report)
    ui = build_a2ui_payload(net, collapsed, field=fe, objective=jx, improvement=imp,
                            hotspots=hs, represent=rp, effective_n=ess)
    blob = json.dumps(payload, allow_nan=False, sort_keys=True)
    json.dumps(ui, allow_nan=False)
    print("   payload blocks: %s" % ", ".join(sorted(payload)))
    print("   json.dumps survives: %d characters, %d a2ui cards"
          % (len(blob), len(ui["statCards"])))

    # -- 0. validation ------------------------------------------------------
    print("\n-- 0. input validation --")
    for bad in (lambda: make_concern("x", "potholes", 1.0, 0.0, 0.0),
                lambda: kernel_weight("gaussian", 1.0, 0.0),
                lambda: morans_i([1.0, 1.0, 1.0, 1.0], w_mat),
                lambda: jenks_breaks([1.0, 2.0], 3),
                lambda: concern_field(net, [], bandwidth=10.0)):
        try:
            bad()
            print("   ERROR: no exception raised")
        except PublicVoiceError as exc:
            print("   PublicVoiceError: %s" % exc)

    print("\n" + "=" * 70)
    print("demo OK")


if __name__ == "__main__":
    _demo()
