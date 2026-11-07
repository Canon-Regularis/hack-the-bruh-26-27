"""
Policy Compiler -- deterministic binding of policy text to search-space fixings
===============================================================================

**Reading is out of scope by design.**  This module is the MOSAIC paper's
section 5.5 made deterministic: an upstream reader (a person, or a model whose
output is treated as untrusted) proposes *typed clauses* -- a verbatim quote,
a page, a named place and one ruling per search operation -- and everything
from there on is lookup, binding and assignment.  The quote is verified
against the source character for character, the place is bound to network
identifiers by rules that refuse rather than guess, and the rulings are
compiled into fixings and bounds on the hybrid search vector
``x = (z, y, w, k)`` that ``urban/urban_optimisation`` explores.  Anything the
compiler cannot verify or bind unambiguously goes to an **abstention queue**
for a human; nothing is ever silently dropped and no identifier is ever made
up.

The module sits between ``urban/public_voice`` (which turns consultation comments into a concern
field -- soft, located, statistical) and ``urban/urban_optimisation`` (which consumes the
``CompiledFixings`` dict produced here as plain JSON, without importing this file).  It owns the
HARD side of policy: what a traffic order prohibits, protects or bounds, when it is in force,
and how a candidate design stands against it clause by clause.

The highest-risk failure is a WRONG BINDING -- a clause about Fitzwilliam Street compiled onto
Fitzwilliam Gate -- because it is invisible in the output.  Whole-name fuzzy scores cannot
prevent it (Jaro-Winkler puts the wrong street at 0.9306 and the right abbreviation at 0.9556),
so the street resolver splits every name into (specific, thoroughfare type), requires the
normalised type class to match EXACTLY and fuzzy-matches only the specific.

Everything is pure standard library.  Every iteration that can reach an output is over a sorted
sequence, every tie is broken on an explicit id, the salted builtin string hash is never used
(digests use ``hashlib``), and the optional core imports are validated-then-delegated with
bit-identical local fallbacks, so output is byte-identical across runs, working directories and
``PYTHONHASHSEED``.

Contents
--------
0.  **Validation** -- ``PolicyCompilerError`` and the coercion helpers.
1.  **String metrics** -- Levenshtein, OSA Damerau, Jaro(-Winkler), Jaccard, LCS, R-O, codes.
2.  **Text normalisation** -- ``normalise_text``, ordinals, compass words.
3.  **Span verification** -- ``verify_span``: exact / recovered / whitespace / fuzzy / rejected.
4.  **JSON-schema-lite** -- ``check_schema``, ``validate_json``, ``is_valid_json``.
5.  **Constraint schema** -- ``OPERATIONS``, ``RULINGS`` with ``ABSTAIN``, ``PolicyClause``.
6.  **Street-name resolver** -- ``split_street_name``, ``resolve_street_name`` (exact type class).
7.  **Gazetteer** -- override table, ``enumerate_candidates``, ``resolve``, BM25, RRF.
8.  **Geometric disambiguation** -- bearings, distances, ``chain_contiguity``, ``disambiguate``.
9.  **Ambiguity and abstention** -- ``ambiguity_score``, ``AbstentionQueue``.
10. **Predicates** -- ``TimeWindow``, ``Scenario``, ``Predicate``, ``parse_predicate``.
11. **In-force resolution** -- Allen's 13 relations, the amendment DAG, ``resolve_in_force``.
12. **Directional wording** -- ``classify_wording``, ``weight_prior`` (``w * (1 + eta * p)``).
13. **Compilation** -- ``CompiledFixings`` (the contract), ``compile_clause``, ``compile_policy``.
14. **Conflict detection** -- ``detect_conflicts``, ``merge_fixings``.
15. **Projection and reduction** -- ``DesignSpace``, ``project_design``, ``free_variables``,
    ``enumerate_designs`` (the candidate generator).
16. **Relaxation analysis** -- ``non_dominated_indices``, ``hypervolume``, ``relaxation_analysis``.
17. **Policy-check report** -- ``policy_check``.
18. **Provenance and agreement** -- ``ProvenanceGraph``, ``agreement_score``, ``cohen_kappa``.
19. **Payloads** -- ``build_policy_compiler_payload`` and ``build_a2ui_payload``.
20. **Example district** -- ``build_example_district``.
21. **Demo**.

Conventions
-----------
* Every identifier (entry, edge, node, clause, instrument) is a ``str``; a
  bare string where a list of ids is expected is refused, never iterated.
* ``CompiledFixings.to_dict()`` is the contract with ``urban_optimisation``:
  ``fixedZero`` (``z_e := 0``, prohibited) and ``fixedOne`` (``z_e := 1``,
  protected) are sorted edge ids, ``bounds`` maps a ``quantity:edge``
  variable to ``[lo, hi]``, ``crossingFixings`` maps a node to 0 or 1,
  ``junctionDomains`` maps a node to the sorted allowed subset of
  ``JUNCTION_TYPES``, ``conditional`` holds predicate-gated fixings and
  ``provenance`` maps each fixing key to the clauses behind it.
* Rulings: ``PROHIBIT`` / ``REQUIRE`` compile to fixings, ``PERMIT`` records
  permission without restricting anything (the one exception: a REPARTITION
  permit that carries explicit ``bounds`` is a permission scoped to that range
  and compiles those bounds), ``ABSTAIN`` says the clause is silent about that
  operation.  Only ``strength='hard'`` clauses compile;
  soft clauses feed the directional weight prior.
* Dates are ISO strings or integer day numbers; validity periods are
  half-open ``[start, end)`` intervals.
* Thresholds on similarity scores are dimensionless; geometric decisions are
  RELATIVE (nearest by a factor of two), so they are unit-free.  Nothing is
  compared against an absolute epsilon.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import heapq
import itertools
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

__version__ = "1.0.0"

# --------------------------------------------------------------------------
# Optional cross-checks from the orchid core library.  Each import is guarded and has a complete
# local fallback that reproduces the core algorithm operation for operation.  Every public
# wrapper validates its arguments BEFORE delegating, so the exception type (always
# PolicyCompilerError) and the returned value are identical whichever import succeeded.
# --------------------------------------------------------------------------
try:
    from algorithms.core.nlp.similarity.similarity_template import (
        JaccardSimilarity as _CoreJaccardSimilarity,
        JaroWinklerSimilarity as _CoreJaroWinklerSimilarity,
        LCSSimilarity as _CoreLCSSimilarity,
        LevenshteinSimilarity as _CoreLevenshteinSimilarity,
        RatcliffObershelpSimilarity as _CoreRatcliffObershelpSimilarity,
        damerau_levenshtein_distance as _core_damerau_levenshtein_distance,
        jaro_similarity as _core_jaro_similarity,
        levenshtein_distance as _core_levenshtein_distance,
        metaphone_code as _core_metaphone_code,
        soundex_code as _core_soundex_code,
    )
    _HAVE_CORE_SIMILARITY = True
except ImportError:
    _HAVE_CORE_SIMILARITY = False

try:
    from algorithms.core.nlp.search.search_template import (
        BM25SearchEngine as _CoreBM25SearchEngine,
        InvertedIndex as _CoreInvertedIndex,
        RRFRanker as _CoreRRFRanker,
    )
    _HAVE_CORE_SEARCH = True
except ImportError:
    _HAVE_CORE_SEARCH = False

Point = Tuple[float, float]

__all__ = [
    # errors
    "PolicyCompilerError",
    # string metrics
    "levenshtein", "damerau_levenshtein", "levenshtein_similarity",
    "levenshtein_normalised_distance", "jaro", "jaro_winkler", "jaccard_tokens",
    "lcs_similarity", "ratcliff_obershelp", "soundex", "metaphone", "STRING_METRICS",
    "string_similarity", "StringComparison", "compare_strings",
    # text normalisation
    "ABBREVIATIONS", "COMPASS_POINTS", "normalise_whitespace", "ordinal_suffix",
    "normalise_text", "compass_bearing",
    # span verification
    "SPAN_STATUSES", "SpanVerification", "find_all_occurrences", "verify_span",
    # json-schema-lite
    "check_schema", "validate_json", "is_valid_json",
    # constraint schema
    "OPERATIONS", "ABSTAIN", "RULINGS", "JUNCTION_TYPES", "TARGET_KINDS", "OBJECTIVES",
    "STRENGTHS", "DAYS", "VEHICLE_CLASSES", "PREDICATE_SCHEMA", "HINT_SCHEMA", "CLAUSE_SCHEMA",
    "make_rulings", "PolicyClause",
    # street-name resolver
    "THOROUGHFARE_TYPES", "STREET_QUALIFIERS", "STREET_STATUSES", "StreetNameParts",
    "thoroughfare_class", "split_street_name", "StreetMatch", "StreetResolution",
    "resolve_street_name",
    # gazetteer
    "ENTRY_KINDS", "bm25_scores", "phrase_match_counts", "rrf_fuse", "GazetteerEntry",
    "Candidate", "BINDING_STATUSES", "Binding", "Gazetteer",
    # geometric disambiguation
    "bearing_deg", "angular_difference", "axial_difference", "point_segment_distance",
    "point_polyline_distance", "polyline_length", "polyline_bearing", "ChainContiguity",
    "chain_contiguity", "GeometricHint", "Disambiguation", "disambiguate",
    # ambiguity and abstention
    "ABSTENTION_REASONS", "ambiguity_score", "normalised_entropy", "AbstentionItem",
    "AbstentionQueue",
    # predicates
    "TimeWindow", "parse_time_window", "Scenario", "Predicate", "parse_predicate",
    # in-force resolution
    "ALLEN_RELATIONS", "Interval", "allen_relation", "allen_inverse", "allen_composition",
    "intervals_overlap", "date_ordinal", "Instrument", "amendment_edges",
    "find_amendment_cycle", "topological_order", "InForceState", "resolve_in_force",
    "validity_relation",
    # directional wording
    "WORDING_CUES", "MODALITIES", "WordingClass", "classify_wording", "directional_priors",
    "weight_prior",
    # compilation
    "FIXING_PART_SCHEMA", "FIXINGS_SCHEMA", "ConditionalFixing", "CompiledFixings",
    "TargetElements", "target_elements", "compile_clause", "CompilationResult",
    "compile_policy",
    # conflict detection
    "CONFLICT_KINDS", "Conflict", "detect_conflicts", "merge_fixings",
    # projection and free-variable reduction
    "Design", "DesignSpace", "applicable_fixings", "violations", "is_feasible",
    "ProjectionResult", "project_design", "FreeVariableReport", "free_variables",
    "default_y_levels", "enumerate_designs", "count_designs",
    # relaxation analysis
    "dominates", "non_dominated_indices", "hypervolume", "ParetoSet", "pareto_set",
    "RelaxationEntry", "RelaxationReport", "relaxation_analysis",
    # policy check
    "CHECK_STATUSES", "PolicyCheckRow", "PolicyCheckReport", "policy_check",
    # provenance and agreement
    "PROVENANCE_NODE_TYPES", "ProvenanceGraph", "build_provenance_graph", "AgreementReport",
    "agreement_score", "binding_agreement", "cohen_kappa",
    # payloads
    "build_policy_compiler_payload", "build_a2ui_payload",
    # example district
    "ExampleDistrict", "build_example_district",
]


# ===========================================================================
# 0. VALIDATION HELPERS
# ===========================================================================
class PolicyCompilerError(ValueError):
    """Invalid input to a policy-compiler routine."""


def _as_float(x: Any, name: str = "x") -> float:
    """Coerce a scalar to a finite float or raise."""
    if isinstance(x, bool):
        raise PolicyCompilerError("%s must be a real number, not a bool." % name)
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):
        raise PolicyCompilerError("%s must be a real number (got %r)." % (name, x))
    if math.isnan(v) or math.isinf(v):
        raise PolicyCompilerError("%s must be finite (got %r)." % (name, x))
    return v


def _as_nonneg(x: Any, name: str = "x") -> float:
    """Coerce to a non-negative finite float or raise."""
    v = _as_float(x, name)
    if v < 0.0:
        raise PolicyCompilerError("%s must be non-negative (got %g)." % (name, v))
    return v


def _as_positive(x: Any, name: str = "x") -> float:
    """Coerce to a strictly positive finite float or raise."""
    v = _as_float(x, name)
    if v <= 0.0:
        raise PolicyCompilerError("%s must be strictly positive (got %g)." % (name, v))
    return v


def _as_fraction(x: Any, name: str = "x") -> float:
    """Coerce to a float in ``[0, 1]`` or raise."""
    v = _as_float(x, name)
    if not (0.0 <= v <= 1.0):
        raise PolicyCompilerError("%s must lie in [0, 1] (got %g)." % (name, v))
    return v


def _as_int(x: Any, name: str = "x", minimum: Optional[int] = None) -> int:
    """Coerce an integral value (not a bool, not a fractional float) to int."""
    if isinstance(x, bool):
        raise PolicyCompilerError("%s must be an integer, not a bool." % name)
    if isinstance(x, int):
        v = int(x)
    elif isinstance(x, float):
        if math.isnan(x) or math.isinf(x) or x != math.floor(x):
            raise PolicyCompilerError("%s must be an integer (got %r)." % (name, x))
        v = int(x)
    else:
        raise PolicyCompilerError("%s must be an integer (got %r)." % (name, x))
    if minimum is not None and v < minimum:
        raise PolicyCompilerError("%s must be >= %d (got %d)." % (name, minimum, v))
    return v


def _as_str(x: Any, name: str = "x", allow_empty: bool = True) -> str:
    """Require a ``str`` (never coerce: a number is not a street name)."""
    if not isinstance(x, str):
        raise PolicyCompilerError("%s must be a string (got %s)." % (name, type(x).__name__))
    if not allow_empty and not x.strip():
        raise PolicyCompilerError("%s must be a non-empty string." % name)
    return x


def _as_id(x: Any, name: str = "id") -> str:
    """An identifier: a non-empty string with no surrounding whitespace."""
    s = _as_str(x, name, allow_empty=False)
    if s != s.strip():
        raise PolicyCompilerError("%s must not carry surrounding whitespace (%r)." % (name, s))
    return s


def _as_id_list(xs: Any, name: str = "ids") -> List[str]:
    """A list of identifiers (a bare string is rejected, not iterated)."""
    if isinstance(xs, (str, bytes)) or xs is None:
        raise PolicyCompilerError("%s must be a sequence of identifiers." % name)
    try:
        items = list(xs)
    except TypeError:
        raise PolicyCompilerError("%s must be a sequence of identifiers." % name)
    return [_as_id(v, "%s[%d]" % (name, i)) for i, v in enumerate(items)]


def _as_seq(xs: Any, name: str = "items") -> List[Any]:
    """A finite sequence as a list (strings, bytes, None and scalars refused)."""
    if isinstance(xs, (str, bytes, dict)) or xs is None:
        raise PolicyCompilerError("%s must be a sequence." % name)
    try:
        return list(xs)
    except TypeError:
        raise PolicyCompilerError("%s must be a sequence." % name)


def _as_bool(x: Any, name: str = "flag") -> bool:
    """Require a real bool."""
    if not isinstance(x, bool):
        raise PolicyCompilerError("%s must be a bool (got %r)." % (name, x))
    return x


def _as_choice(x: Any, choices: Sequence[str], name: str = "choice") -> str:
    """Require one of a fixed set of string options."""
    if not isinstance(x, str) or x not in choices:
        raise PolicyCompilerError("%s must be one of %s (got %r)."
                                  % (name, ", ".join(choices), x))
    return x


def _as_point(p: Any, name: str = "point") -> Point:
    """A finite planar point ``(x, y)``."""
    try:
        seq = list(p)
    except TypeError:
        raise PolicyCompilerError("%s must be an (x, y) pair." % name)
    if len(seq) != 2:
        raise PolicyCompilerError("%s must be an (x, y) pair (got %d values)." % (name, len(seq)))
    return (_as_float(seq[0], name + ".x"), _as_float(seq[1], name + ".y"))


def _as_vector(xs: Any, name: str = "x") -> List[float]:
    """A list of finite floats."""
    if isinstance(xs, (str, bytes)):
        raise PolicyCompilerError("%s must be a sequence of numbers." % name)
    try:
        items = list(xs)
    except TypeError:
        raise PolicyCompilerError("%s must be a sequence of numbers." % name)
    return [_as_float(v, "%s[%d]" % (name, i)) for i, v in enumerate(items)]


def _json_safe(x: Any) -> Any:
    """Map non-finite floats to ``None`` so ``json.dumps(allow_nan=False)`` passes."""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _stable_digest(obj: Any) -> str:
    """SHA-256 of a canonical JSON rendering -- never the salted builtin hash."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ===========================================================================
# 1. STRING METRICS -- edit distances, Jaro(-Winkler), set and phonetic codes
# ===========================================================================
# Each public metric validates its arguments FIRST and then delegates to the core implementation
# when it is importable, otherwise to the local copy below.  The local copies reproduce the core
# algorithms operation for operation (same recurrences, same floating-point expression order),
# so the two paths return bit-identical values -- the suite checks this on seeded random strings
# with the core both available and forced off.

def _local_levenshtein(a: str, b: str) -> int:
    """Two-row Levenshtein DP (insert / delete / substitute, unit costs)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    n, m = len(a), len(b)
    prev = list(range(n + 1))
    for j in range(1, m + 1):
        curr = [j] + [0] * n
        bj = b[j - 1]
        for i in range(1, n + 1):
            cost = 0 if a[i - 1] == bj else 1
            curr[i] = min(curr[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost)
        prev = curr
    return prev[n]


def _local_damerau_levenshtein(a: str, b: str) -> int:
    """Optimal-string-alignment distance (adjacent transpositions cost 1)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[n][m]


def _local_jaro(a: str, b: str) -> float:
    """Jaro similarity with the classic ``max(|a|,|b|)//2 - 1`` match window."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    match_window = max(len(a), len(b)) // 2 - 1
    match_window = max(0, match_window)
    a_flags = [False] * len(a)
    b_flags = [False] * len(b)
    matches = 0
    for i, ca in enumerate(a):
        lo = max(0, i - match_window)
        hi = min(len(b), i + match_window + 1)
        for j in range(lo, hi):
            if not b_flags[j] and ca == b[j]:
                a_flags[i] = True
                b_flags[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    a_matched = [a[i] for i in range(len(a)) if a_flags[i]]
    b_matched = [b[j] for j in range(len(b)) if b_flags[j]]
    transpositions = sum(ca != cb for ca, cb in zip(a_matched, b_matched)) // 2
    return (matches / len(a) + matches / len(b)
            + (matches - transpositions) / matches) / 3.0


def _local_jaro_winkler(a: str, b: str, p: float, max_prefix: int) -> float:
    """Jaro-Winkler: Jaro boosted by ``l * p * (1 - jaro)`` for a shared prefix."""
    jaro = _local_jaro(a, b)
    if jaro == 0.0:
        return 0.0
    prefix = 0
    for ca, cb in zip(a[:max_prefix], b[:max_prefix]):
        if ca == cb:
            prefix += 1
        else:
            break
    return min(1.0, jaro + prefix * p * (1.0 - jaro))


def _local_lcs_length(a: str, b: str) -> int:
    """Longest common subsequence length (two-row DP)."""
    if not a or not b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    n, m = len(a), len(b)
    prev = [0] * (n + 1)
    for j in range(1, m + 1):
        curr = [0] * (n + 1)
        for i in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[i] = prev[i - 1] + 1
            else:
                curr[i] = max(prev[i], curr[i - 1])
        prev = curr
    return prev[n]


def _local_lcs_similarity(a: str, b: str) -> float:
    """``2 * LCS(a, b) / (|a| + |b|)``."""
    total = len(a) + len(b)
    if total == 0:
        return 1.0
    return 2.0 * _local_lcs_length(a, b) / total


def _local_ro_matching(a: str, b: str) -> int:
    """Ratcliff-Obershelp matching-character count (leftmost longest block)."""
    if not a or not b:
        return 0
    best_len = 0
    best_ai = best_bi = 0
    for ai in range(len(a)):
        for bi in range(len(b)):
            length = 0
            while (ai + length < len(a) and bi + length < len(b)
                   and a[ai + length] == b[bi + length]):
                length += 1
            if length > best_len:
                best_len = length
                best_ai, best_bi = ai, bi
    if best_len == 0:
        return 0
    return (best_len
            + _local_ro_matching(a[:best_ai], b[:best_bi])
            + _local_ro_matching(a[best_ai + best_len:], b[best_bi + best_len:]))


def _local_ratcliff_obershelp(a: str, b: str) -> float:
    """Gestalt pattern-matching similarity ``2 * M / (|a| + |b|)``."""
    total = len(a) + len(b)
    if total == 0:
        return 1.0
    return 2.0 * _local_ro_matching(a, b) / total


def _local_word_tokens(text: str) -> List[str]:
    """NFC, lower-case, punctuation to spaces, split -- the core tokeniser."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _local_jaccard(a: str, b: str) -> float:
    """Jaccard coefficient of the two token SETS."""
    sa = set(_local_word_tokens(a))
    sb = set(_local_word_tokens(b))
    if not sa and not sb:
        return 1.0
    intersection = len(sa & sb)
    union = len(sa | sb)
    return intersection / union if union else 0.0


_SOUNDEX_TABLE = str.maketrans("AEHIOUWYBFPVCGJKQSXZDTLMNR",
                               "00000000111122222222334556")


def _local_soundex(word: str) -> str:
    """American Soundex (H/W transparent, vowels reset the run)."""
    word = word.upper()
    if not word:
        return "0000"
    first = word[0]
    result = first
    prev = first.translate(_SOUNDEX_TABLE) or "0"
    for orig_ch in word[1:]:
        code = orig_ch.translate(_SOUNDEX_TABLE)
        if not code:
            continue
        if code == "0":
            if orig_ch not in "HW":
                prev = "0"
            continue
        if code != prev:
            result += code
            if len(result) == 4:
                break
        prev = code
    return result.ljust(4, "0")


# letters whose Metaphone code never depends on their neighbours
_METAPHONE_PLAIN = {"B": "B", "F": "F", "J": "J", "L": "L", "M": "M", "N": "N", "R": "R",
                    "Q": "K", "V": "F", "X": "KS", "Z": "S"}


def _local_metaphone(word: str) -> str:
    """Simplified single-pass Metaphone (the core variant, rule for rule)."""
    word = re.sub(r"[^A-Z]", "", word.upper())
    if not word:
        return ""
    vowels = set("AEIOU")
    word = word[1:] if word[:2] in ("AE", "GN", "KN", "PN", "WR") else word
    word = word[1:] if word and word[0] in vowels else word
    code: List[str] = []
    i = 0
    while i < len(word):
        ch = word[i]
        nxt = word[i + 1] if i + 1 < len(word) else ""
        nxt2 = word[i + 2] if i + 2 < len(word) else ""
        if ch in vowels:
            pass
        elif ch in _METAPHONE_PLAIN:
            code.append(_METAPHONE_PLAIN[ch])
        elif ch == "C":
            soft, cia = nxt in ("I", "E", "Y"), nxt == "I" and nxt2 == "A"
            code += ["S", "X"] if soft and cia else ["S"] if soft else ["X" if nxt == "H" else "K"]
            i += (soft and cia) or (not soft and nxt == "H")
        elif ch == "D":
            dg = nxt == "G" and nxt2 in ("E", "I", "Y")
            code.append("J" if dg else "T")
            i += dg
        elif ch == "G":
            if (nxt == "H" and nxt2 not in vowels) or (nxt == "N" and (not nxt2 or nxt2 not in
                                                                       "AEIOU")):
                i += 2
                continue
            code.append("J" if nxt in ("E", "I", "Y") else "K")
            i += nxt == "H"
        elif ch in ("H", "W", "Y"):
            code += [ch] if nxt in vowels else []
        elif ch == "K":
            code += ["K"] if nxt != "C" else []
        elif ch == "P":
            code.append("F" if nxt == "H" else "P")
            i += nxt == "H"
        elif ch == "S":
            sch, sh = nxt == "C" and nxt2 == "H", nxt == "H" or (nxt == "I" and nxt2 in ("O", "A"))
            code.append("SK" if sch else "X" if sh else "S")
            i += 2 if sch else 1 if sh else 0
        elif ch == "T":
            code.append("0" if nxt == "H" else "X" if nxt == "I" and nxt2 in ("A", "O") else "T")
            i += nxt == "H"
        i += 1
    result = [c for k, c in enumerate(code) if k == 0 or c != code[k - 1]]
    return ("".join(result) or word[0]) if word else ""


def levenshtein(a: str, b: str) -> int:
    """Levenshtein edit distance (unit insert / delete / substitute)."""
    a = _as_str(a, "a")
    b = _as_str(b, "b")
    if _HAVE_CORE_SIMILARITY:
        return int(_core_levenshtein_distance(a, b))
    return _local_levenshtein(a, b)


def damerau_levenshtein(a: str, b: str) -> int:
    """Optimal-string-alignment distance (NOT a metric: no triangle inequality)."""
    a = _as_str(a, "a")
    b = _as_str(b, "b")
    if _HAVE_CORE_SIMILARITY:
        return int(_core_damerau_levenshtein_distance(a, b))
    return _local_damerau_levenshtein(a, b)


def levenshtein_similarity(a: str, b: str) -> float:
    """``1 - levenshtein(a, b) / max(|a|, |b|)`` (1.0 for two empty strings)."""
    a = _as_str(a, "a")
    b = _as_str(b, "b")
    if _HAVE_CORE_SIMILARITY:
        return float(_CoreLevenshteinSimilarity().similarity(a, b))
    if a == b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - _local_levenshtein(a, b) / max_len


def levenshtein_normalised_distance(a: str, b: str) -> float:
    """``levenshtein(a, b) / max(|a|, |b|)`` -- the core ``.distance()``."""
    a = _as_str(a, "a")
    b = _as_str(b, "b")
    if _HAVE_CORE_SIMILARITY:
        return float(_CoreLevenshteinSimilarity().distance(a, b))
    if a == b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    return _local_levenshtein(a, b) / max_len


def jaro(a: str, b: str) -> float:
    """Jaro similarity in ``[0, 1]``."""
    a = _as_str(a, "a")
    b = _as_str(b, "b")
    if _HAVE_CORE_SIMILARITY:
        return float(_core_jaro_similarity(a, b))
    return _local_jaro(a, b)


def _check_jw_params(p: Any, max_prefix: Any) -> Tuple[float, int]:
    """Validate the Winkler scaling factor and prefix cap before any delegation."""
    pv = _as_nonneg(p, "p")
    mp = _as_int(max_prefix, "max_prefix", minimum=0)
    if pv * mp > 1.0:
        raise PolicyCompilerError("p * max_prefix must be <= 1 (got %g * %d)." % (pv, mp))
    return pv, mp


def jaro_winkler(a: str, b: str, p: float = 0.1, max_prefix: int = 4) -> float:
    """Jaro-Winkler similarity (standard ``p = 0.1``, prefix cap 4)."""
    a = _as_str(a, "a")
    b = _as_str(b, "b")
    pv, mp = _check_jw_params(p, max_prefix)
    if _HAVE_CORE_SIMILARITY:
        return float(_CoreJaroWinklerSimilarity(pv, mp).similarity(a, b))
    return _local_jaro_winkler(a, b, pv, mp)


def jaccard_tokens(a: str, b: str) -> float:
    """Jaccard coefficient of the word-token sets of two strings."""
    a = _as_str(a, "a")
    b = _as_str(b, "b")
    if _HAVE_CORE_SIMILARITY:
        return float(_CoreJaccardSimilarity().similarity(a, b))
    return _local_jaccard(a, b)


def lcs_similarity(a: str, b: str) -> float:
    """``2 * LCS / (|a| + |b|)`` on characters."""
    a = _as_str(a, "a")
    b = _as_str(b, "b")
    if _HAVE_CORE_SIMILARITY:
        return float(_CoreLCSSimilarity().similarity(a, b))
    return _local_lcs_similarity(a, b)


def ratcliff_obershelp(a: str, b: str) -> float:
    """Ratcliff-Obershelp gestalt similarity."""
    a = _as_str(a, "a")
    b = _as_str(b, "b")
    if _HAVE_CORE_SIMILARITY:
        return float(_CoreRatcliffObershelpSimilarity().similarity(a, b))
    return _local_ratcliff_obershelp(a, b)


def _soundex_letters(word: Any) -> str:
    """The ASCII letters of a word; refuse a word with none."""
    w = _as_str(word, "word")
    letters = "".join(ch for ch in w if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    if not letters:
        raise PolicyCompilerError("soundex needs at least one ASCII letter (got %r)." % w)
    return letters


def soundex(word: str) -> str:
    """American Soundex code of the ASCII letters of ``word`` (``Robert`` -> ``R163``)."""
    letters = _soundex_letters(word)
    if _HAVE_CORE_SIMILARITY:
        return str(_core_soundex_code(letters))
    return _local_soundex(letters)


def metaphone(word: str) -> str:
    """Simplified Metaphone code (empty string when ``word`` has no letters)."""
    w = _as_str(word, "word")
    if _HAVE_CORE_SIMILARITY:
        return str(_core_metaphone_code(w))
    return _local_metaphone(w)


STRING_METRICS = ("levenshtein", "damerau_levenshtein", "jaro", "jaro_winkler",
                  "jaccard", "lcs", "ratcliff_obershelp")


def string_similarity(a: str, b: str, method: str = "jaro_winkler") -> float:
    """Similarity in ``[0, 1]`` by name; edit distances are max-length normalised."""
    method = _as_choice(method, STRING_METRICS, "method")
    a = _as_str(a, "a")
    b = _as_str(b, "b")
    if method == "levenshtein":
        return levenshtein_similarity(a, b)
    if method == "damerau_levenshtein":
        if a == b:
            return 1.0
        return 1.0 - damerau_levenshtein(a, b) / max(len(a), len(b))
    if method == "jaro":
        return jaro(a, b)
    if method == "jaro_winkler":
        return jaro_winkler(a, b)
    if method == "jaccard":
        return jaccard_tokens(a, b)
    if method == "lcs":
        return lcs_similarity(a, b)
    return ratcliff_obershelp(a, b)


@dataclass
class StringComparison:
    """Every metric for one string pair, plus the JW ``compare()`` score/distance."""
    a: str
    b: str
    levenshtein: int
    damerau_levenshtein: int
    levenshtein_similarity: float
    jaro: float
    jaro_winkler: float
    jaro_winkler_distance: float
    jaccard: float
    lcs: float
    ratcliff_obershelp: float
    soundex_equal: Optional[bool]
    metaphone_equal: bool

    def as_dict(self) -> Dict[str, Any]:
        """JSON-able camelCase dict."""
        return {"a": self.a, "b": self.b, "levenshtein": self.levenshtein,
                "damerauLevenshtein": self.damerau_levenshtein,
                "levenshteinSimilarity": self.levenshtein_similarity,
                "jaro": self.jaro, "jaroWinkler": self.jaro_winkler,
                "jaroWinklerDistance": self.jaro_winkler_distance,
                "jaccard": self.jaccard, "lcs": self.lcs,
                "ratcliffObershelp": self.ratcliff_obershelp,
                "soundexEqual": self.soundex_equal,
                "metaphoneEqual": self.metaphone_equal}


def compare_strings(a: str, b: str) -> StringComparison:
    """All metrics for one pair.  Uses the core ``.compare()`` API when present."""
    a, b = _as_str(a, "a"), _as_str(b, "b")
    if _HAVE_CORE_SIMILARITY:
        res = _CoreJaroWinklerSimilarity().compare(a, b)
        jw, jwd = float(res.score), float(res.distance)
    else:
        jw = max(0.0, min(1.0, _local_jaro_winkler(a, b, 0.1, 4)))
        jwd = max(0.0, 1.0 - jw)
    sx = (soundex(a) == soundex(b)) if all(any(ch.isascii() and ch.isalpha() for ch in x)
                                           for x in (a, b)) else None
    return StringComparison(
        a=a, b=b, levenshtein=levenshtein(a, b),
        damerau_levenshtein=damerau_levenshtein(a, b),
        levenshtein_similarity=levenshtein_similarity(a, b),
        jaro=jaro(a, b), jaro_winkler=jw, jaro_winkler_distance=jwd,
        jaccard=jaccard_tokens(a, b), lcs=lcs_similarity(a, b),
        ratcliff_obershelp=ratcliff_obershelp(a, b), soundex_equal=sx,
        metaphone_equal=metaphone(a) == metaphone(b))


# ===========================================================================
# 2. TEXT NORMALISATION -- case, punctuation, abbreviations, ordinals, compass
# ===========================================================================
# General abbreviations used in traffic-order prose.  Street-type abbreviations live in
# THOROUGHFARE_TYPES (section 6) and are applied by the resolver, NOT here, because "St" is
# "Street" at the end of a name and "Saint" at the start of one -- a context the street parser
# owns.
ABBREVIATIONS: Dict[str, str] = {
    "approx": "approximately", "bldg": "building", "cttee": "committee",
    "dept": "department", "eg": "for example", "etc": "et cetera",
    "hgv": "heavy goods vehicle", "hgvs": "heavy goods vehicles",
    "ie": "that is", "jct": "junction", "jcn": "junction", "lgv": "light goods vehicle",
    "mph": "miles per hour", "kph": "kilometres per hour",
    "opp": "opposite", "para": "paragraph", "reg": "regulation",
    "sched": "schedule", "sect": "section", "tro": "traffic regulation order",
    "veh": "vehicle", "vehs": "vehicles",
}

_ORDINAL_WORDS: Dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "thirtieth": 30,
}
_ORDINAL_TENS: Dict[str, int] = {"twenty": 20, "thirty": 30}

COMPASS_POINTS: Dict[str, str] = {
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "nth": "north", "sth": "south",
    "north": "north", "south": "south", "east": "east", "west": "west",
    "northeast": "northeast", "northwest": "northwest",
    "southeast": "southeast", "southwest": "southwest",
}
_COMPASS_BEARING: Dict[str, float] = {
    "north": 0.0, "northeast": 45.0, "east": 90.0, "southeast": 135.0,
    "south": 180.0, "southwest": 225.0, "west": 270.0, "northwest": 315.0,
}


def normalise_whitespace(text: str) -> str:
    """Collapse every whitespace run to one space and strip the ends."""
    return " ".join(_as_str(text, "text").split())


def ordinal_suffix(n: int) -> str:
    """``1 -> '1st'``, ``2 -> '2nd'``, ``11 -> '11th'``, ``23 -> '23rd'``."""
    v = _as_int(n, "n", minimum=0)
    if 10 <= v % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(v % 10, "th")
    return "%d%s" % (v, suf)


def _ordinal_value(token: str) -> Optional[int]:
    """Integer value of an ordinal token ('3rd', 'third', 'twenty-first')."""
    m = re.match(r"^(\d+)(st|nd|rd|th)$", token)
    if m:
        v = int(m.group(1))
        if ordinal_suffix(v) == token:
            return v
        return None
    if token in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[token]
    parts = token.split("-")
    if len(parts) == 2 and parts[0] in _ORDINAL_TENS and parts[1] in _ORDINAL_WORDS:
        unit = _ORDINAL_WORDS[parts[1]]
        if 1 <= unit <= 9:
            return _ORDINAL_TENS[parts[0]] + unit
    return None


def _strip_accents(text: str) -> str:
    """NFKD-decompose and drop combining marks (Sean -> Sean with no fada)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalise_text(text: str, *, expand_abbreviations: bool = True,
                   ordinals: bool = True, compass: bool = True) -> str:
    """Canonical form: accents and case folded, punctuation dropped, tokens rewritten.
    ``Stephen's -> stephens``, ``twenty-first -> 21st``, ``NE -> northeast``, other punctuation
    to spaces, then the abbreviation table token by token."""
    s = _strip_accents(_as_str(text, "text")).lower()
    expand_abbreviations = _as_bool(expand_abbreviations, "expand_abbreviations")
    ordinals = _as_bool(ordinals, "ordinals")
    compass = _as_bool(compass, "compass")
    s = re.sub("['`\u2018\u2019\u02bc]", "", s)          # every apostrophe form: isn't, isn`t
    # dotted single-letter runs are ONE token: "N.E." -> "ne" (-> northeast), "e.g." -> "eg",
    # "i.e." -> "ie" -- never two separate compass words.
    s = re.sub(r"\b(?:[a-z]\.){2,}",
               lambda mo: " %s " % mo.group(0).replace(".", ""), s)
    s = re.sub(r"(?<=[a-z])-(?=[a-z])", "_", s)
    s = re.sub(r"[^\w\s]", " ", s)
    out: List[str] = []
    for tok in s.split():
        tok = tok.replace("_", "-")
        if ordinals:
            v = _ordinal_value(tok)
            if v is not None:
                out.append(ordinal_suffix(v))
                continue
        if compass and tok.replace("-", "") in COMPASS_POINTS and len(tok) <= 10:
            out.append(COMPASS_POINTS[tok.replace("-", "")])
            continue
        if "-" in tok:
            pieces = [p for p in tok.split("-") if p]
        else:
            pieces = [tok]
        for p in pieces:
            if expand_abbreviations and p in ABBREVIATIONS:
                out.extend(ABBREVIATIONS[p].split())
            else:
                out.append(p)
    return " ".join(out)


def compass_bearing(word: str) -> float:
    """Bearing in degrees clockwise from north for a compass word or abbreviation."""
    w = _as_str(word, "word").strip().lower().replace("-", "").replace(" ", "")
    for suffix in ("bound", "erly", "ern", "ward", "wards"):
        if w.endswith(suffix) and w[:-len(suffix)] in COMPASS_POINTS:
            w = w[:-len(suffix)]
            break
    if w not in COMPASS_POINTS:
        raise PolicyCompilerError("not a compass direction: %r" % word)
    return _COMPASS_BEARING[COMPASS_POINTS[w]]


# ===========================================================================
# 3. SPAN VERIFICATION -- literal substring, offset recovery, fuzzy fallback
# ===========================================================================
SPAN_STATUSES = ("exact", "recovered", "whitespace", "fuzzy", "rejected")


@dataclass
class SpanVerification:
    """Outcome of checking that a quote really occurs in its source text.

    ``exact`` (claimed offsets hold), ``recovered`` (literal substring at the returned offsets),
    ``whitespace`` (substring once whitespace runs collapse; offsets bracket the ORIGINAL text,
    in ``matched_text``), ``fuzzy`` (best window at least the threshold -- NOT verified),
    ``rejected`` (-1, -1)."""
    status: str
    start: int
    end: int
    matched_text: str
    score: float
    occurrences: int
    quote: str

    @property
    def verified(self) -> bool:
        """True only for the three literal outcomes (never for fuzzy)."""
        return self.status in ("exact", "recovered", "whitespace")

    def as_dict(self) -> Dict[str, Any]:
        """JSON-able camelCase dict."""
        return {"status": self.status, "start": self.start, "end": self.end,
                "matchedText": self.matched_text, "score": _json_safe(self.score),
                "occurrences": self.occurrences, "verified": self.verified}


def find_all_occurrences(source: str, quote: str) -> List[int]:
    """Every start offset of ``quote`` in ``source`` (overlapping matches included)."""
    source, quote = _as_str(source, "source"), _as_str(quote, "quote", allow_empty=False)
    return [mo.start() for mo in re.finditer("(?=%s)" % re.escape(quote), source)]


def _whitespace_map(text: str) -> Tuple[str, List[int]]:
    """Whitespace-collapsed text plus, per kept character, its original index."""
    chars: List[str] = []
    index: List[int] = []
    pending_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            pending_space = bool(chars)         # a run of whitespace becomes one space
            continue
        chars += [" ", ch] if pending_space else [ch]
        index += [i - 1, i] if pending_space else [i]
        pending_space = False
    return "".join(chars), index


def _closest(offsets: Sequence[int], hint: Optional[int]) -> int:
    """The offset nearest a hint (ties to the earlier), or the first."""
    return offsets[0] if hint is None else min(offsets, key=lambda o: (abs(o - hint), o))


def _token_spans(text: str) -> List[Tuple[int, int]]:
    """Character spans of the whitespace-separated tokens of ``text``."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def verify_span(source: str, quote: str, start: Optional[int] = None,
                end: Optional[int] = None, *, fuzzy_threshold: float = 0.9,
                max_fuzzy_windows: int = 12) -> SpanVerification:
    """Verify a quoted span against its source: exact, recovered, whitespace, fuzzy or rejected.
    A verified span satisfies ``source[start:end] == quote`` literally or after whitespace
    collapsing.  The fuzzy branch scores token-aligned windows by normalised Levenshtein; a hit
    is returned for a human, never trusted."""
    source = _as_str(source, "source")
    quote = _as_str(quote, "quote", allow_empty=False)
    thr = _as_fraction(fuzzy_threshold, "fuzzy_threshold")
    nwin = _as_int(max_fuzzy_windows, "max_fuzzy_windows", minimum=1)
    hint: Optional[int] = None
    if start is not None or end is not None:
        if start is None or end is None:
            raise PolicyCompilerError("start and end must be given together.")
        s0, e0 = _as_int(start, "start", minimum=0), _as_int(end, "end", minimum=0)
        if e0 < s0:
            raise PolicyCompilerError("end (%d) precedes start (%d)." % (e0, s0))
        hint = s0
        if e0 <= len(source) and source[s0:e0] == quote:
            n_occ = len(find_all_occurrences(source, quote))
            return SpanVerification("exact", s0, e0, quote, 1.0, n_occ, quote)
    occ = find_all_occurrences(source, quote)
    if occ:
        s = _closest(occ, hint)
        return SpanVerification("recovered", s, s + len(quote), quote, 1.0, len(occ), quote)
    norm_src, idx = _whitespace_map(source)
    norm_q = normalise_whitespace(quote)
    if norm_q:
        hits = [mo.start() for mo in re.finditer("(?=%s)" % re.escape(norm_q), norm_src)]
        if hits:
            mapped = [idx[h] for h in hits]
            s = _closest(mapped, hint)
            h = hits[mapped.index(s)]
            e = idx[h + len(norm_q) - 1] + 1
            return SpanVerification("whitespace", s, e, source[s:e], 1.0, len(hits), quote)
    # fuzzy fallback over token-aligned windows of the collapsed source
    toks = _token_spans(norm_src)
    q_toks = norm_q.split(" ") if norm_q else []
    nq = len(q_toks)
    if nq == 0 or not toks:
        return SpanVerification("rejected", -1, -1, "", 0.0, 0, quote)
    q_bag = sorted(q_toks)
    windows: List[Tuple[int, int, int]] = []
    # widths are capped at the source length: a quote longer than the whole source is still
    # scored (against all of it), never left with no window.
    for width in sorted(set(min(len(toks), w) for w in (max(1, nq - 1), nq, nq + 1))):
        for i in range(0, len(toks) - width + 1):
            win = norm_src[toks[i][0]:toks[i + width - 1][1]].split(" ")
            overlap = _bag_overlap(q_bag, sorted(win))
            windows.append((-overlap, i, width))
    windows.sort()
    # The overlap ranking only shortlists ANCHORS: the true window of a typo'd quote can rank
    # below a wider or shifted neighbour that picks up a stray matching word, so every window
    # (of any width) lying inside a shortlisted anchor widened by one token on each side is
    # scored too.
    widths = sorted(set(w for _o, _i, w in windows))
    pool = set()
    for _neg_ov, i, width in windows[:nwin]:
        for w2 in widths:
            for j in range(i - 1, i + width - w2 + 2):
                if 0 <= j and j + w2 <= len(toks):
                    pool.add((j, w2))
    best: Optional[Tuple[float, int, int, int]] = None
    for i, width in sorted(pool):
        a, b = toks[i][0], toks[i + width - 1][1]
        cand = norm_src[a:b]
        score = 1.0 - _local_levenshtein(cand, norm_q) / max(len(cand), len(norm_q))
        key = (-score, a, b)
        if best is None or key < (-best[0], best[1], best[2]):
            best = (score, a, b, width)
    assert best is not None
    score, a, b, _w = best
    if score < thr:
        return SpanVerification("rejected", -1, -1, "", score, 0, quote)
    s, e = idx[a], idx[b - 1] + 1
    return SpanVerification("fuzzy", s, e, source[s:e], score, 1, quote)


def _bag_overlap(a_sorted: Sequence[str], b_sorted: Sequence[str]) -> int:
    """Multiset intersection size of two sorted token lists."""
    i = j = n = 0
    while i < len(a_sorted) and j < len(b_sorted):
        if a_sorted[i] == b_sorted[j]:
            n += 1
            i += 1
            j += 1
        elif a_sorted[i] < b_sorted[j]:
            i += 1
        else:
            j += 1
    return n


# ===========================================================================
# 4. JSON-SCHEMA-LITE VALIDATOR
# ===========================================================================
# A strict subset of JSON Schema (draft-6 numeric keywords).  Unknown keywords are a SCHEMA
# error, not silently ignored: a typo such as "requried" would otherwise accept everything.
_SCHEMA_KEYWORDS = frozenset([
    "type", "enum", "const", "required", "properties", "additionalProperties",
    "items", "minItems", "maxItems", "uniqueItems", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength", "pattern",
    "anyOf", "oneOf", "allOf", "not", "description", "title",
])
_SCHEMA_TYPES = ("object", "array", "string", "number", "integer", "boolean", "null")


def check_schema(schema: Any, path: str = "#") -> None:
    """Raise unless ``schema`` uses only the supported keywords with legal values."""
    if not isinstance(schema, dict):
        raise PolicyCompilerError("schema at %s must be an object." % path)
    for key in sorted(schema, key=str):
        if key not in _SCHEMA_KEYWORDS:
            raise PolicyCompilerError("unsupported schema keyword %r at %s." % (key, path))
    if "type" in schema:
        t = schema["type"]
        ts = t if isinstance(t, list) else [t]
        for v in ts:
            if not isinstance(v, str) or v not in _SCHEMA_TYPES:
                raise PolicyCompilerError("unknown type %r at %s." % (v, path))
        if not ts or len(set(ts)) != len(ts):
            raise PolicyCompilerError("type list at %s must be non-empty and unique." % path)
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise PolicyCompilerError("enum at %s must be a list." % path)
    if "required" in schema:
        req = schema["required"]
        if not isinstance(req, list) or not all(isinstance(r, str) for r in req) \
                or len(set(req)) != len(req):
            raise PolicyCompilerError("required at %s must be a list of unique strings." % path)
    # a non-bool uniqueItems would be a SILENT no-op, like a misspelt keyword
    if "uniqueItems" in schema and not isinstance(schema["uniqueItems"], bool):
        raise PolicyCompilerError("uniqueItems at %s must be a bool." % path)
    for key in ("title", "description"):
        if key in schema and not isinstance(schema[key], str):
            raise PolicyCompilerError("%s at %s must be a string." % (key, path))
    for key in ("properties",):
        if key in schema:
            if not isinstance(schema[key], dict):
                raise PolicyCompilerError("%s at %s must be an object." % (key, path))
            for name in sorted(schema[key], key=str):
                check_schema(schema[key][name], "%s/%s/%s" % (path, key, name))
    if "additionalProperties" in schema:
        ap = schema["additionalProperties"]
        if not isinstance(ap, bool):
            check_schema(ap, path + "/additionalProperties")
    if "items" in schema:
        check_schema(schema["items"], path + "/items")
    if "not" in schema:
        check_schema(schema["not"], path + "/not")
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            if not isinstance(schema[key], list) or not schema[key]:
                raise PolicyCompilerError("%s at %s must be a non-empty list." % (key, path))
            for i, sub in enumerate(schema[key]):
                check_schema(sub, "%s/%s/%d" % (path, key, i))
    for key in ("minItems", "maxItems", "minLength", "maxLength"):
        if key in schema:
            v = schema[key]
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise PolicyCompilerError("%s at %s must be a non-negative integer."
                                          % (key, path))
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if key in schema:
            _as_float(schema[key], "%s at %s" % (key, path))
    if "pattern" in schema:
        try:
            re.compile(schema["pattern"])
        except (re.error, TypeError):
            raise PolicyCompilerError("pattern at %s does not compile." % path)


def _json_type_ok(x: Any, t: str) -> bool:
    """Does ``x`` have JSON type ``t`` (bools are never numbers)?"""
    if t == "null":
        return x is None
    if t == "boolean":
        return isinstance(x, bool)
    if t == "integer":
        # draft-6 and later: any number with a zero fractional part (1.0 too)
        if isinstance(x, float):
            return math.isfinite(x) and x == math.floor(x)
        return isinstance(x, int) and not isinstance(x, bool)
    if t == "number":
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    if t == "string":
        return isinstance(x, str)
    if t == "array":
        return isinstance(x, (list, tuple))
    if t == "object":
        return isinstance(x, dict) and all(isinstance(k, str) for k in x)
    return False


def _json_equal(a: Any, b: Any) -> bool:
    """JSON value equality: ``True != 1`` and ``[1] == (1,)``."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        # key SETS, never sorted(): a stray non-string key must not raise TypeError
        return (len(a) == len(b) and all(k in b for k in a)
                and all(_json_equal(a[k], b[k]) for k in a))
    return type(a) == type(b) and a == b


def _validate(x: Any, schema: Dict[str, Any], path: str, errors: List[str]) -> None:
    """Accumulate validation errors for one instance against one schema node."""
    where = path or "/"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        errors.append("%s: non-finite number" % where)
        return
    if "type" in schema:
        t = schema["type"]
        ts = t if isinstance(t, list) else [t]
        if not any(_json_type_ok(x, v) for v in ts):
            errors.append("%s: expected type %s, got %s" % (where, "/".join(ts), type(x).__name__))
            return
    if "enum" in schema and not any(_json_equal(x, e) for e in schema["enum"]):
        errors.append("%s: value %r not in enum" % (where, x))
    if "const" in schema and not _json_equal(x, schema["const"]):
        errors.append("%s: value %r != const %r" % (where, x, schema["const"]))
    is_num = isinstance(x, (int, float)) and not isinstance(x, bool)
    if is_num:
        if "minimum" in schema and x < schema["minimum"]:
            errors.append("%s: %r < minimum %r" % (where, x, schema["minimum"]))
        if "maximum" in schema and x > schema["maximum"]:
            errors.append("%s: %r > maximum %r" % (where, x, schema["maximum"]))
        if "exclusiveMinimum" in schema and x <= schema["exclusiveMinimum"]:
            errors.append("%s: %r <= exclusiveMinimum %r" % (where, x, schema["exclusiveMinimum"]))
        if "exclusiveMaximum" in schema and x >= schema["exclusiveMaximum"]:
            errors.append("%s: %r >= exclusiveMaximum %r" % (where, x, schema["exclusiveMaximum"]))
    if isinstance(x, str):
        if "minLength" in schema and len(x) < schema["minLength"]:
            errors.append("%s: string shorter than %d" % (where, schema["minLength"]))
        if "maxLength" in schema and len(x) > schema["maxLength"]:
            errors.append("%s: string longer than %d" % (where, schema["maxLength"]))
        if "pattern" in schema and re.search(schema["pattern"], x) is None:
            errors.append("%s: %r does not match pattern %r" % (where, x, schema["pattern"]))
    if isinstance(x, (list, tuple)):
        if "minItems" in schema and len(x) < schema["minItems"]:
            errors.append("%s: fewer than %d items" % (where, schema["minItems"]))
        if "maxItems" in schema and len(x) > schema["maxItems"]:
            errors.append("%s: more than %d items" % (where, schema["maxItems"]))
        if schema.get("uniqueItems") is True:
            for i in range(len(x)):
                for j in range(i + 1, len(x)):
                    if _json_equal(x[i], x[j]):
                        errors.append("%s: items %d and %d are equal" % (where, i, j))
        if "items" in schema:
            for i, item in enumerate(x):
                _validate(item, schema["items"], "%s/%d" % (path, i), errors)
    if isinstance(x, dict):
        for r in schema.get("required", []):
            if r not in x:
                errors.append("%s: missing required property %r" % (where, r))
        props = schema.get("properties", {})
        for k in sorted(x, key=str):
            if not isinstance(k, str):
                errors.append("%s: non-string key %r" % (where, k))
                continue
            sub = "%s/%s" % (path, k)
            if k in props:
                _validate(x[k], props[k], sub, errors)
            else:
                ap = schema.get("additionalProperties", True)
                if ap is False:
                    errors.append("%s: additional property %r not allowed" % (where, k))
                elif isinstance(ap, dict):
                    _validate(x[k], ap, sub, errors)
    if "allOf" in schema:
        for sub in schema["allOf"]:
            _validate(x, sub, path, errors)
    if "anyOf" in schema:
        if not any(not validate_json(x, sub, _checked=True) for sub in schema["anyOf"]):
            errors.append("%s: matches none of anyOf" % where)
    if "oneOf" in schema:
        n_ok = sum(1 for sub in schema["oneOf"] if not validate_json(x, sub, _checked=True))
        if n_ok != 1:
            errors.append("%s: matches %d of oneOf (need exactly 1)" % (where, n_ok))
    if "not" in schema and not validate_json(x, schema["not"], _checked=True):
        errors.append("%s: matches a forbidden 'not' schema" % where)


def _nonfinite_paths(x: Any, path: str, errors: List[str]) -> None:
    """Flag every NaN / inf anywhere in an instance -- JSON has no such numbers."""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        errors.append("%s: non-finite number" % (path or "/"))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            _nonfinite_paths(v, "%s/%d" % (path, i), errors)
    elif isinstance(x, dict):
        for k in sorted(x, key=str):
            _nonfinite_paths(x[k], "%s/%s" % (path, k), errors)


def validate_json(instance: Any, schema: Dict[str, Any], *, _checked: bool = False) -> List[str]:
    """All errors of ``instance`` against a JSON-schema-lite ``schema`` (sorted; [] = valid).
    Non-finite numbers are reported wherever they sit, including under properties the schema
    leaves unconstrained."""
    if not _checked:
        check_schema(schema)
    errors: List[str] = []
    _nonfinite_paths(instance, "", errors)
    _validate(instance, schema, "", errors)
    return sorted(set(errors))


def is_valid_json(instance: Any, schema: Dict[str, Any]) -> bool:
    """True iff ``validate_json`` reports no error."""
    return not validate_json(instance, schema)


# ===========================================================================
# 5. CONSTRAINT SCHEMA -- typed clauses with one ruling per search operation
# ===========================================================================
# The six search operations of the MOSAIC topology search.  Every clause carries exactly one
# ruling for EACH of them; a clause that is silent about an operation says so explicitly with
# ABSTAIN rather than by omission.
OPERATIONS = ("ADD_EDGE", "REMOVE_EDGE", "REPARTITION", "ADD_CROSSING",
              "GREEN_CORRIDOR", "JUNCTION_REDESIGN")
ABSTAIN = "ABSTAIN"
RULINGS = ("PERMIT", "PROHIBIT", "REQUIRE", ABSTAIN)
# The junction-type set K shared with urban_optimisation.
JUNCTION_TYPES = ("priority", "signalised", "signalised_banned_turns",
                  "roundabout", "mini_roundabout")
TARGET_KINDS = ("street", "junction", "area", "edge", "node")
OBJECTIVES = ("car", "ped", "green", "emit", "voice", "cost", "risk")
STRENGTHS = ("hard", "soft")
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
VEHICLE_CLASSES = ("car", "taxi", "bus", "coach", "lgv", "hgv", "motorcycle",
                   "cycle", "emergency", "pedestrian")

def _array(items: Any, **kw: Any) -> Dict[str, Any]:
    return dict(type="array", items=items, **kw)


def _object(props: Dict[str, Any], *required: str) -> Dict[str, Any]:
    return dict([("type", "object")] + ([("required", list(required))] if required else [])
                + [("properties", props), ("additionalProperties", False)])


def _ids(**kw: Any) -> Dict[str, Any]:
    return _array({"type": "string", "minLength": 1}, **kw)


_PAIR = _array({"type": "number"}, minItems=2, maxItems=2)
PREDICATE_SCHEMA: Dict[str, Any] = _object({
    "timeWindows": _array(_array({"type": "integer", "minimum": 0, "maximum": 1440},
                                 minItems=2, maxItems=2)),
    "days": _array({"enum": list(DAYS)}, uniqueItems=True),
    "vehicleClasses": _array({"enum": list(VEHICLE_CLASSES)}, uniqueItems=True),
    "exemptClasses": _array({"enum": list(VEHICLE_CLASSES)}, uniqueItems=True),
    "scenarioTags": _ids(uniqueItems=True)})
HINT_SCHEMA: Dict[str, Any] = _object({
    "point": _PAIR, "maxDistance": {"type": "number", "exclusiveMinimum": 0},
    "bearing": {"type": "number", "minimum": 0, "exclusiveMaximum": 360},
    "bearingTolerance": {"type": "number", "minimum": 0, "maximum": 180},
    "adjacentTo": _ids(uniqueItems=True), "district": {"type": "string", "minLength": 1}})
CLAUSE_SCHEMA: Dict[str, Any] = _object(dict(
    [(k, {"type": "string", "minLength": 1}) for k in ("clauseId", "documentId")]
    + [("page", {"type": "integer", "minimum": 1}), ("quote", {"type": "string", "minLength": 1})]
    + [(k, {"type": "integer", "minimum": 0}) for k in ("start", "end")] + [
        ("target", _object({"kind": {"enum": list(TARGET_KINDS)},
                            "name": {"type": "string", "minLength": 1},
                            "nameB": {"type": "string", "minLength": 1}}, "kind", "name")),
        ("rulings", _object(dict((op, {"enum": list(RULINGS)}) for op in OPERATIONS),
                            *OPERATIONS)),
        ("bounds", {"type": "object", "additionalProperties": _PAIR}),
        ("junctionTypes", _array({"enum": list(JUNCTION_TYPES)}, uniqueItems=True, minItems=1)),
        ("predicate", PREDICATE_SCHEMA), ("hint", HINT_SCHEMA),
        ("strength", {"enum": list(STRENGTHS)}), ("objective", {"enum": list(OBJECTIVES)})]
    + [(k, {"type": "string", "minLength": 1}) for k in ("wording", "instrumentId")]),
    "clauseId", "documentId", "page", "quote", "target", "rulings")


def make_rulings(**given: str) -> Dict[str, str]:
    """A full ruling map: ABSTAIN for every operation not named explicitly."""
    out = dict((op, ABSTAIN) for op in OPERATIONS)
    for op in sorted(given):
        if op not in OPERATIONS:
            raise PolicyCompilerError("unknown operation %r." % op)
        out[op] = _as_choice(given[op], RULINGS, "ruling for %s" % op)
    return out


@dataclass
class PolicyClause:
    """One typed policy clause: a quote, a place, and a ruling per operation."""
    clause_id: str
    document_id: str
    page: int
    quote: str
    target_kind: str
    target_name: str
    rulings: Dict[str, str]
    target_name_b: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    junction_types: Optional[Tuple[str, ...]] = None
    predicate: Optional["Predicate"] = None
    hint: Optional["GeometricHint"] = None
    strength: str = "hard"
    objective: Optional[str] = None
    wording: Optional[str] = None
    instrument_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.rulings, dict) or not isinstance(self.bounds, dict):
            raise PolicyCompilerError("rulings and bounds must be dicts.")
        # own copies: a caller editing its dict afterwards must not bypass validation
        self.rulings = dict(self.rulings)
        self.bounds = dict(self.bounds)
        if self.predicate is not None and not isinstance(self.predicate, Predicate):
            raise PolicyCompilerError("predicate must be a Predicate or None.")
        if self.hint is not None and not isinstance(self.hint, GeometricHint):
            raise PolicyCompilerError("hint must be a GeometricHint or None.")
        if self.junction_types is not None:
            if isinstance(self.junction_types, str):
                raise PolicyCompilerError("junction_types must be a sequence, not a string.")
            self.junction_types = tuple(_as_seq(self.junction_types, "junction_types"))
        for q in sorted(self.bounds, key=str):
            b = self.bounds[q]
            if isinstance(b, (str, bytes)) or not hasattr(b, "__len__") or len(b) != 2:
                raise PolicyCompilerError("bounds[%r] must be a (lo, hi) pair." % (q,))
            if not isinstance(q, str) or ":" in q:
                # variables are "quantity:edge"; a colon in the quantity would make the edge id
                # unrecoverable (edge ids MAY contain ':')
                raise PolicyCompilerError("bounds quantity %r must be a colon-free string." % (q,))
        try:
            snapshot = self.to_dict()
        except (TypeError, ValueError, AttributeError) as exc:
            raise PolicyCompilerError("invalid clause: %s" % exc)
        errs = validate_json(snapshot, CLAUSE_SCHEMA)
        if errs:
            raise PolicyCompilerError("invalid clause %r: %s" % (self.clause_id, "; ".join(errs)))
        _as_id(self.clause_id, "clause_id")
        _as_id(self.document_id, "document_id")
        # the schema admits integral floats (draft-6 "integer"); store real ints
        self.page = _as_int(self.page, "page", minimum=1)
        if self.start is not None:
            self.start = _as_int(self.start, "start", minimum=0)
        if self.end is not None:
            self.end = _as_int(self.end, "end", minimum=0)
        for q, (lo, hi) in sorted(self.bounds.items()):
            if _as_float(lo, "bounds[%s].lo" % q) > _as_float(hi, "bounds[%s].hi" % q):
                raise PolicyCompilerError("bounds[%s] has lo > hi (%g > %g)." % (q, lo, hi))
        if (self.start is None) != (self.end is None):
            raise PolicyCompilerError("clause %s: start and end go together." % self.clause_id)
        if self.start is not None and self.end is not None and self.end < self.start:
            raise PolicyCompilerError("clause %s: end precedes start." % self.clause_id)
        if self.target_kind == "junction" and self.target_name_b is None:
            raise PolicyCompilerError("a junction target needs both street names "
                                      "(target_name and target_name_b).")
        if self.target_kind != "junction" and self.target_name_b is not None:
            raise PolicyCompilerError("target_name_b is only meaningful for a junction.")
        self.bounds = dict((q, (float(lo), float(hi)))
                           for q, (lo, hi) in sorted(self.bounds.items()))
        if self.junction_types is not None:
            self.junction_types = tuple(sorted(self.junction_types))

    @property
    def abstains_everywhere(self) -> bool:
        """True when every operation's ruling is ABSTAIN."""
        return all(self.rulings[op] == ABSTAIN for op in OPERATIONS)

    @property
    def text_for_wording(self) -> str:
        """The text the directional classifier reads (``wording`` or the quote)."""
        return self.wording if self.wording is not None else self.quote

    def to_dict(self) -> Dict[str, Any]:
        """Schema-shaped camelCase dict (round-trips through ``from_dict``)."""
        target: Dict[str, Any] = {"kind": self.target_kind, "name": self.target_name}
        if self.target_name_b is not None:
            target["nameB"] = self.target_name_b
        d: Dict[str, Any] = {
            "clauseId": self.clause_id, "documentId": self.document_id,
            "page": self.page, "quote": self.quote, "target": target,
            "rulings": dict((op, self.rulings.get(op)) for op in sorted(self.rulings)),
        }
        if self.start is not None:
            d["start"] = self.start
        if self.end is not None:
            d["end"] = self.end
        if self.bounds:
            d["bounds"] = dict((q, [lo, hi]) for q, (lo, hi) in sorted(self.bounds.items()))
        if self.junction_types is not None:
            d["junctionTypes"] = list(self.junction_types)
        if self.predicate is not None:
            d["predicate"] = self.predicate.to_dict()
        if self.hint is not None:
            d["hint"] = self.hint.to_dict()
        d["strength"] = self.strength
        if self.objective is not None:
            d["objective"] = self.objective
        if self.wording is not None:
            d["wording"] = self.wording
        if self.instrument_id is not None:
            d["instrumentId"] = self.instrument_id
        return d

    @classmethod
    def from_dict(cls, d: Any) -> "PolicyClause":
        """Validate a clause dict against ``CLAUSE_SCHEMA`` and build the clause."""
        errs = validate_json(d, CLAUSE_SCHEMA)
        if errs:
            raise PolicyCompilerError("invalid clause dict: %s" % "; ".join(errs))
        tgt = d["target"]
        pred = Predicate.from_dict(d["predicate"]) if "predicate" in d else None
        hint = GeometricHint.from_dict(d["hint"]) if "hint" in d else None
        return cls(
            clause_id=d["clauseId"], document_id=d["documentId"], page=d["page"],
            quote=d["quote"], target_kind=tgt["kind"], target_name=tgt["name"],
            target_name_b=tgt.get("nameB"), rulings=dict(d["rulings"]),
            start=d.get("start"), end=d.get("end"),
            bounds=dict((q, (v[0], v[1])) for q, v in d.get("bounds", {}).items()),
            junction_types=(tuple(d["junctionTypes"]) if "junctionTypes" in d else None),
            predicate=pred, hint=hint, strength=d.get("strength", "hard"),
            objective=d.get("objective"), wording=d.get("wording"),
            instrument_id=d.get("instrumentId"))


# ===========================================================================
# 6. STREET-NAME RESOLVER -- (specific, thoroughfare type), exact type class
# ===========================================================================
# THE LOAD-BEARING RULE.  A street name is split into its SPECIFIC part ("fitzwilliam") and its
# THOROUGHFARE TYPE ("street").  The type is mapped through this table to a canonical class and
# must match EXACTLY; only the specific is fuzzy-matched.  Jaro-Winkler on whole names scores
# "Fitzwilliam Street" vs "Fitzwilliam Gate" at 0.9306 and vs the legitimate "Fitzwilliam St" at
# 0.9556 -- any whole-name threshold that admits the abbreviation also admits a DIFFERENT
# STREET.  No threshold is safe; the type class is.  Qualifiers (Upper / Lower / North ...) also
# distinguish streets and must match exactly.
THOROUGHFARE_TYPES: Dict[str, Tuple[str, ...]] = {
    "alley": ("alley", "aly"), "avenue": ("avenue", "ave", "av", "avn"),
    "boulevard": ("boulevard", "blvd"), "bridge": ("bridge", "br", "brg"),
    "circus": ("circus", "circ"), "close": ("close", "cl"),
    "court": ("court", "ct"), "crescent": ("crescent", "cres", "cresc", "cr"),
    "drive": ("drive", "dr", "drv"), "embankment": ("embankment", "emb"),
    "gardens": ("gardens", "gdns", "gdn"), "gate": ("gate", "gte"),
    "green": ("green", "grn"), "grove": ("grove", "gr", "gro", "grv"),
    "highway": ("highway", "hwy"), "hill": ("hill",), "lane": ("lane", "ln"),
    "mews": ("mews",), "parade": ("parade", "pde"), "park": ("park", "pk"),
    "place": ("place", "pl"), "quay": ("quay", "qy"), "rise": ("rise",),
    "road": ("road", "rd"), "row": ("row",), "square": ("square", "sq", "sqr"),
    "street": ("street", "st", "str"), "terrace": ("terrace", "terr", "ter", "tce"),
    "walk": ("walk", "wlk"), "way": ("way",), "wharf": ("wharf", "whf"),
    "yard": ("yard", "yd"),
}
_TYPE_LOOKUP: Dict[str, str] = {}
for _canon in sorted(THOROUGHFARE_TYPES):
    for _variant in THOROUGHFARE_TYPES[_canon]:
        _TYPE_LOOKUP[_variant] = _canon
STREET_QUALIFIERS = ("upper", "lower", "north", "south", "east", "west",
                     "great", "little", "middle", "inner", "outer")
STREET_STATUSES = ("bound", "ambiguous", "unresolved")


@dataclass
class StreetNameParts:
    """A street name split into qualifier(s), specific and thoroughfare-type class."""
    raw: str
    specific: str
    type_class: Optional[str]
    qualifiers: Tuple[str, ...]

    def canonical(self) -> str:
        """Canonical rendering: qualifiers, specific, type class."""
        parts = list(self.qualifiers) + [self.specific]
        if self.type_class is not None:
            parts.append(self.type_class)
        return " ".join(p for p in parts if p)


def thoroughfare_class(token: str) -> Optional[str]:
    """Canonical type class of one token (``'St' -> 'street'``), or None."""
    t = _as_str(token, "token").strip().lower().rstrip(".")
    return _TYPE_LOOKUP.get(t)


def split_street_name(name: str) -> StreetNameParts:
    """Split a street name into ``(specific, type_class, qualifiers)``. ``"Fitzwilliam St" ->
    ("fitzwilliam", "street")``; a leading ``St`` is Saint; qualifiers before the specific or
    after the type are pulled out, so ``Upper X Street`` and ``X Street Upper`` agree -- and
    differ from Lower."""
    raw = _as_str(name, "name", allow_empty=False)
    toks = normalise_text(raw, expand_abbreviations=False, ordinals=True,
                          compass=False).split()
    if toks and toks[0] == "the" and len(toks) > 1:
        toks = toks[1:]
    if not toks:
        raise PolicyCompilerError("street name %r has no words." % raw)
    if len(toks) > 1 and toks[0] in ("st", "saint"):
        toks[0] = "saint"
    trailing: List[str] = []
    while len(toks) > 2 and toks[-1] in STREET_QUALIFIERS and toks[-2] in _TYPE_LOOKUP:
        trailing.insert(0, toks.pop())
    type_class: Optional[str] = None
    if len(toks) >= 2 and toks[-1] in _TYPE_LOOKUP:
        type_class = _TYPE_LOOKUP[toks.pop()]
    leading: List[str] = []
    while len(toks) > 1 and toks[0] in STREET_QUALIFIERS:
        leading.append(toks.pop(0))
    specific = " ".join(toks)
    return StreetNameParts(raw, specific, type_class, tuple(sorted(leading + trailing)))


@dataclass
class StreetMatch:
    """One gazetteer name examined by the resolver, with the verdict on it."""
    entry_id: str
    name: str
    specific_score: float
    type_match: bool
    qualifier_match: bool
    accepted: bool
    reason: str


@dataclass
class StreetResolution:
    """Resolver verdict: bound to one entry, ambiguous, or unresolved."""
    status: str
    entry_id: Optional[str]
    query: StreetNameParts
    matches: List[StreetMatch]
    accepted: List[Tuple[str, float]]

    @property
    def candidate_ids(self) -> List[str]:
        """Entry ids that passed the type gate and the specific threshold."""
        return [eid for eid, _s in self.accepted]


def resolve_street_name(query: str, names: Sequence[Tuple[str, str]], *,
                        threshold: float = 0.9, margin: float = 0.04,
                        metric: str = "jaro_winkler") -> StreetResolution:
    """Bind a street name to one ``(entry_id, name)`` pair, or refuse. Admissible only with
    EXACTLY the query's type class and qualifiers; the specifics are compared by ``metric``
    (kept at ``>= threshold``, aliases folded to their entry's best). One entry, or a leader
    ahead by ``margin``, is ``bound``; otherwise ``ambiguous``; none is ``unresolved``."""
    q = split_street_name(query)
    thr = _as_fraction(threshold, "threshold")
    mg = _as_fraction(margin, "margin")
    metric = _as_choice(metric, STRING_METRICS, "metric")
    pairs: List[Tuple[str, str]] = []
    for i, pr in enumerate(_as_seq(names, "names")):
        try:
            eid, nm = pr
        except (TypeError, ValueError):
            raise PolicyCompilerError("names[%d] must be an (entry_id, name) pair." % i)
        if isinstance(pr, (str, bytes, dict)):
            raise PolicyCompilerError("names[%d] must be an (entry_id, name) pair." % i)
        pairs.append((_as_id(eid, "entry_id"), _as_str(nm, "name", allow_empty=False)))
    pairs.sort()
    matches: List[StreetMatch] = []
    best: Dict[str, float] = {}
    for eid, nm in pairs:
        c = split_street_name(nm)
        type_ok = c.type_class == q.type_class
        qual_ok = c.qualifiers == q.qualifiers
        if not type_ok:
            matches.append(StreetMatch(eid, nm, 0.0, False, qual_ok, False, "type_mismatch"))
            continue
        if not qual_ok:
            matches.append(StreetMatch(eid, nm, 0.0, True, False, False, "qualifier_mismatch"))
            continue
        score = 1.0 if c.specific == q.specific else string_similarity(q.specific, c.specific, metric)
        short, long_ = sorted((q.specific, c.specific), key=len)
        # a specific that is another's proper prefix plus 2+ letters is a DIFFERENT name, not a
        # typo: Church/Churchill, Kings/Kingston score >= 0.9 under JW
        ext = long_.startswith(short) and len(long_) - len(short) >= 2
        ok = score >= thr and not ext
        matches.append(StreetMatch(eid, nm, score, True, True, ok, "accepted" if ok else
                                   "prefix_extension" if ext else "below_threshold"))
        if ok and score > best.get(eid, -1.0):
            best[eid] = score
    accepted = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    if not accepted:
        return StreetResolution("unresolved", None, q, matches, [])
    if len(accepted) == 1 or accepted[0][1] - accepted[1][1] >= mg:
        return StreetResolution("bound", accepted[0][0], q, matches, accepted)
    return StreetResolution("ambiguous", None, q, matches, accepted)


# ===========================================================================
# 7. GAZETTEER -- entries, manual overrides, lexical search, candidates
# ===========================================================================
ENTRY_KINDS = ("street", "junction", "area", "landmark")


def _search_tokens(text: str) -> List[str]:
    """The core search tokeniser: NFC, ``\\b\\w+\\b`` runs, lower-cased."""
    return [t.lower() for t in re.findall(r"(?u)\b\w+\b", unicodedata.normalize("NFC", text))]


def _local_bm25_token_scores(token: str, docs: Sequence[str], k1: float, b: float,
                             epsilon: float) -> List[float]:
    """Single-token Okapi BM25 per document (the core formula, term for term)."""
    tokenised = [_search_tokens(d) for d in docs]
    n = len(docs)
    if n == 0:
        return []
    avgdl = sum(len(t) for t in tokenised) / n
    c = sum(1 for t in tokenised if token in t)
    out: List[float] = []
    if c == 0:
        return [0.0] * n
    idf = max(epsilon, math.log((n - c + 0.5) / (c + 0.5) + 1.0))
    for toks in tokenised:
        f_qd = toks.count(token)
        if f_qd == 0:
            out.append(0.0)
            continue
        dl = len(toks)
        denom = f_qd + k1 * (1 - b + b * dl / avgdl) if avgdl > 0 else f_qd + k1
        out.append(0.0 + idf * (f_qd * (k1 + 1) / denom + 0.0))
    return out


def bm25_scores(query: str, documents: Sequence[str], *, k1: float = 1.5,
                b: float = 0.75, epsilon: float = 0.25) -> List[float]:
    """Okapi BM25 score of ``query`` against every document. Per-token scores from the core
    ``score_all`` (or the local copy), summed here over SORTED tokens -- the core sums a
    hash-seed-ordered ``set``."""
    query = _as_str(query, "query")
    docs = [_as_str(d, "documents[%d]" % i)
            for i, d in enumerate(_as_seq(documents, "documents"))]
    k1v = _as_nonneg(k1, "k1")
    bv = _as_fraction(b, "b")
    ev = _as_nonneg(epsilon, "epsilon")
    toks = sorted(set(_search_tokens(query)))
    per_doc: List[List[float]] = [[] for _ in docs]
    if not docs or not toks:
        return [0.0] * len(docs)
    engine = None
    if _HAVE_CORE_SEARCH:
        engine = _CoreBM25SearchEngine(k1=k1v, b=bv, epsilon=ev, store_docs=False)
        engine.index(docs)
    for tok in toks:
        if engine is not None:
            col = [float(v) for v in engine.score_all(tok)]
        else:
            col = _local_bm25_token_scores(tok, docs, k1v, bv, ev)
        for i, v in enumerate(col):
            per_doc[i].append(v)
    return [math.fsum(v) for v in per_doc]


def _local_phrase_counts(phrase: str, docs: Sequence[str]) -> List[int]:
    """Occurrences of the token sequence of ``phrase`` in each document."""
    terms = _search_tokens(phrase)
    out: List[int] = []
    for d in docs:
        toks = _search_tokens(d)
        if not terms:
            out.append(0)
            continue
        n = 0
        for i in range(len(toks) - len(terms) + 1):
            if toks[i:i + len(terms)] == terms:
                n += 1
        out.append(n)
    return out


def phrase_match_counts(phrase: str, documents: Sequence[str]) -> List[int]:
    """Per document, how often the token sequence of ``phrase`` occurs in it. Delegates to the
    core ``InvertedIndex.phrase_search`` positional intersection when importable."""
    phrase = _as_str(phrase, "phrase")
    docs = [_as_str(d, "documents[%d]" % i)
            for i, d in enumerate(_as_seq(documents, "documents"))]
    if not docs:
        return []
    if _HAVE_CORE_SEARCH:
        idx = _CoreInvertedIndex(store_docs=False)
        idx.index(docs)
        res = idx.phrase_search(phrase, top_k=len(docs))
        counts = res.metadata.get("phrase_match_counts", {})
        return [int(counts.get(str(i), 0)) for i in range(len(docs))]
    return _local_phrase_counts(phrase, docs)


def rrf_fuse(ranked_lists: Sequence[Sequence[str]], *, k: int = 60) -> List[Tuple[str, float]]:
    """Reciprocal-rank fusion ``sum_r 1/(k + rank_r(d))``, ties broken on id. Uses the core
    ``RRFRanker`` when importable; string ids are mapped to integers through their sorted order
    and back."""
    kv = _as_int(k, "k", minimum=1)
    lists = [_as_id_list(r, "ranked_lists[%d]" % i)
             for i, r in enumerate(_as_seq(ranked_lists, "ranked_lists"))]
    for i, r in enumerate(lists):
        if len(set(r)) != len(r):
            raise PolicyCompilerError("ranked_lists[%d] repeats an id." % i)
    ids = sorted(set(x for r in lists for x in r))
    pos = dict((x, i) for i, x in enumerate(ids))
    if _HAVE_CORE_SEARCH:
        fused = _CoreRRFRanker(k=kv).fuse([[pos[x] for x in r] for r in lists])
        pairs = [(ids[int(d)], float(s)) for d, s in fused]
    else:
        scores: Dict[str, float] = {}
        order: List[str] = []
        for r in lists:
            for rank, x in enumerate(r, start=1):
                if x not in scores:
                    scores[x] = 0.0
                    order.append(x)
                scores[x] += 1.0 / (kv + rank)
        pairs = [(x, scores[x]) for x in order]
    return sorted(pairs, key=lambda p: (-p[1], p[0]))


@dataclass
class GazetteerEntry:
    """A named place: its existing edges, candidate edges, nodes and geometry."""
    entry_id: str
    name: str
    kind: str = "street"
    edge_ids: Tuple[str, ...] = ()
    candidate_edge_ids: Tuple[str, ...] = ()
    node_ids: Tuple[str, ...] = ()
    coords: Tuple[Point, ...] = ()
    aliases: Tuple[str, ...] = ()
    district: Optional[str] = None

    def __post_init__(self) -> None:
        self.entry_id = _as_id(self.entry_id, "entry_id")
        self.name = _as_str(self.name, "name", allow_empty=False)
        self.kind = _as_choice(self.kind, ENTRY_KINDS, "kind")
        self.edge_ids = tuple(sorted(set(_as_id_list(self.edge_ids, "edge_ids"))))
        self.candidate_edge_ids = tuple(sorted(set(_as_id_list(self.candidate_edge_ids,
                                                               "candidate_edge_ids"))))
        both = set(self.edge_ids) & set(self.candidate_edge_ids)
        if both:
            raise PolicyCompilerError("entry %s lists %s as existing AND candidate."
                                      % (self.entry_id, sorted(both)[0]))
        self.node_ids = tuple(sorted(set(_as_id_list(self.node_ids, "node_ids"))))
        if isinstance(self.coords, (str, bytes)):
            raise PolicyCompilerError("coords must be a sequence of points.")
        self.coords = tuple(_as_point(p, "coords[%d]" % i) for i, p in enumerate(self.coords))
        if isinstance(self.aliases, (str, bytes)):
            raise PolicyCompilerError("aliases must be a sequence of names.")
        self.aliases = tuple(_as_str(a, "alias", allow_empty=False) for a in self.aliases)
        if self.district is not None:
            self.district = _as_id(self.district, "district")

    @property
    def all_names(self) -> Tuple[str, ...]:
        """Official name followed by every alias."""
        return (self.name,) + self.aliases

    def as_dict(self) -> Dict[str, Any]:
        """JSON-able camelCase dict."""
        return {"id": self.entry_id, "name": self.name, "kind": self.kind,
                "edgeIds": list(self.edge_ids),
                "candidateEdgeIds": list(self.candidate_edge_ids),
                "nodeIds": list(self.node_ids),
                "coords": [[x, y] for (x, y) in self.coords],
                "aliases": list(self.aliases), "district": self.district}


@dataclass
class Candidate:
    """A gazetteer candidate for a query -- always an EXISTING entry id."""
    entry_id: str
    name: str
    score: float
    sources: Tuple[str, ...]


BINDING_STATUSES = ("bound", "ambiguous", "unresolved")


@dataclass
class Binding:
    """Result of binding a clause target to the network."""
    status: str
    target_type: str
    target_id: Optional[str]
    candidates: List[str]
    scores: List[float]
    ambiguity: float
    method: str
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        """JSON-able camelCase dict."""
        return {"status": self.status, "targetType": self.target_type,
                "targetId": self.target_id, "candidates": list(self.candidates),
                "scores": [_json_safe(s) for s in self.scores],
                "ambiguity": _json_safe(self.ambiguity), "method": self.method,
                "reason": self.reason}


class Gazetteer:
    """Named places of the network, a manual override table and candidate search. Overrides pin
    a normalised phrase to an EXISTING entry (the planner's escape hatch); no API here ever
    invents an identifier."""

    def __init__(self, entries: Sequence[GazetteerEntry], *,
                 edge_endpoints: Optional[Dict[str, Tuple[str, str]]] = None,
                 node_coords: Optional[Dict[str, Point]] = None,
                 overrides: Optional[Dict[str, str]] = None) -> None:
        ents = _as_seq(entries, "entries")
        if not ents:
            raise PolicyCompilerError("a gazetteer needs at least one entry.")
        self._entries: Dict[str, GazetteerEntry] = {}
        for e in ents:
            if not isinstance(e, GazetteerEntry):
                raise PolicyCompilerError("entries must be GazetteerEntry instances.")
            if e.entry_id in self._entries:
                raise PolicyCompilerError("duplicate gazetteer id %r." % e.entry_id)
            self._entries[e.entry_id] = e
        both = set(x for e in ents for x in e.edge_ids) & set(x for e in ents
                                                              for x in e.candidate_edge_ids)
        if both:        # the GazetteerEntry rule, across entries: z := 1 would BUILD a candidate
            raise PolicyCompilerError("edge %r is existing in one entry and a candidate in "
                                      "another." % sorted(both)[0])
        for nm, tbl in (("edge_endpoints", edge_endpoints), ("node_coords", node_coords),
                        ("overrides", overrides)):
            if tbl is not None and not isinstance(tbl, dict):
                raise PolicyCompilerError("%s must be a dict." % nm)
        self._edge_endpoints: Dict[str, Tuple[str, str]] = {}
        for eid in sorted(edge_endpoints or {}, key=str):
            uv = (edge_endpoints or {})[eid]
            try:
                if isinstance(uv, (str, bytes)):
                    raise TypeError("pair")
                u, v = uv
            except (TypeError, ValueError):
                raise PolicyCompilerError("edge_endpoints[%r] must be a (u, v) pair." % eid)
            self._edge_endpoints[_as_id(eid, "edge id")] = (_as_id(u, "u"), _as_id(v, "v"))
        self._node_coords: Dict[str, Point] = {}
        for nid in sorted(node_coords or {}, key=str):
            self._node_coords[_as_id(nid, "node id")] = _as_point((node_coords or {})[nid],
                                                                  "node_coords[%s]" % nid)
        self._overrides: Dict[str, str] = {}
        for text in sorted(overrides or {}, key=str):
            self.add_override(text, (overrides or {})[text])

    # -- basic access ------------------------------------------------------
    @property
    def ids(self) -> List[str]:
        """Sorted entry ids."""
        return sorted(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, entry_id: object) -> bool:
        return isinstance(entry_id, str) and entry_id in self._entries

    def get(self, entry_id: str) -> GazetteerEntry:
        """The entry with this id (raises for an unknown id)."""
        eid = _as_id(entry_id, "entry_id")
        if eid not in self._entries:
            raise PolicyCompilerError("unknown gazetteer id %r." % eid)
        return self._entries[eid]

    def entries(self, kind: Optional[str] = None) -> List[GazetteerEntry]:
        """Entries (optionally of one kind) in id order."""
        if kind is not None:
            kind = _as_choice(kind, ENTRY_KINDS, "kind")
        return [self._entries[i] for i in self.ids
                if kind is None or self._entries[i].kind == kind]

    @property
    def edge_endpoints(self) -> Dict[str, Tuple[str, str]]:
        """Copy of the edge -> (u, v) table."""
        return dict(self._edge_endpoints)

    @property
    def node_coords(self) -> Dict[str, Point]:
        """Copy of the node -> (x, y) table."""
        return dict(self._node_coords)

    @property
    def all_edge_ids(self) -> List[str]:
        """Every existing and candidate edge id named by any entry (sorted)."""
        return sorted(set(x for e in self._entries.values() for x in e.edge_ids
                          + e.candidate_edge_ids))

    # -- overrides ---------------------------------------------------------
    @staticmethod
    def key(text: str) -> str:
        """Normalised lookup key for a phrase."""
        return normalise_text(_as_str(text, "text", allow_empty=False))

    def add_override(self, text: str, entry_id: str) -> None:
        """Pin a phrase to an EXISTING entry (unknown ids are refused)."""
        k = self.key(text)
        self.get(entry_id)
        self._overrides[k] = entry_id

    def remove_override(self, text: str) -> None:
        """Drop an override (raises if absent)."""
        k = self.key(text)
        if k not in self._overrides:
            raise PolicyCompilerError("no override for %r." % text)
        del self._overrides[k]

    @property
    def overrides(self) -> Dict[str, str]:
        """Copy of the override table."""
        return dict(self._overrides)

    # -- candidate enumeration ----------------------------------------------
    def _name_rows(self, kind: Optional[str]) -> List[Tuple[str, str]]:
        """(entry_id, name) for every official name and alias, in a fixed order."""
        rows: List[Tuple[str, str]] = []
        for e in self.entries(kind):
            for nm in e.all_names:
                rows.append((e.entry_id, nm))
        return rows

    def enumerate_candidates(self, query: str, *, kind: Optional[str] = None,
                             top_k: int = 10, threshold: float = 0.9) -> List[Candidate]:
        """Ranked candidates for a phrase -- RRF over override, exact, resolver and BM25.
        Returns existing entries only; an empty list means nothing plausible, and it is never
        padded with a made-up id.  An applicable override ranks FIRST (``resolve`` binds it);
        the rest by fused score, then id."""
        q = _as_str(query, "query", allow_empty=False)
        k = _as_int(top_k, "top_k", minimum=1)
        qk = self.key(q)
        rows = self._name_rows(kind)
        lists: List[List[str]] = []
        sources: Dict[str, List[str]] = {}

        def note(eid: str, src: str) -> None:
            sources.setdefault(eid, [])
            if src not in sources[eid]:
                sources[eid].append(src)

        pinned: Optional[str] = None
        if qk in self._overrides:
            eid = self._overrides[qk]
            if kind is None or self._entries[eid].kind == kind:
                lists.append([eid])
                note(eid, "override")
                pinned = eid
        exact = sorted(set(eid for eid, nm in rows if self.key(nm) == qk))
        if exact:
            lists.append(exact)
            for eid in exact:
                note(eid, "exact")
        if kind in (None, "street") and rows:
            street_rows = [(eid, nm) for eid, nm in rows
                           if self._entries[eid].kind == "street"]
            if street_rows:
                res = resolve_street_name(q, street_rows, threshold=threshold)
                if res.accepted:
                    lists.append([eid for eid, _s in res.accepted])
                    for eid, _s in res.accepted:
                        note(eid, "resolver")
        if rows:
            docs = [normalise_text(nm) for _eid, nm in rows]
            sc = bm25_scores(qk, docs)
            best: Dict[str, float] = {}
            for (eid, _nm), s in zip(rows, sc):
                if s > 0.0 and s > best.get(eid, 0.0):
                    best[eid] = s
            lex = [eid for eid, _s in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))]
            if lex:
                lists.append(lex)
                for eid in lex:
                    note(eid, "lexical")
        if not lists:
            return []
        fused = rrf_fuse(lists)
        if pinned is not None:
            fused = sorted(fused, key=lambda p: (p[0] != pinned, -p[1], p[0]))
        out = [Candidate(eid, self._entries[eid].name, s, tuple(sources[eid]))
               for eid, s in fused[:k]]
        for c in out:
            if c.entry_id not in self._entries:   # pragma: no cover - invariant
                raise PolicyCompilerError("fabricated identifier %r." % c.entry_id)
        return out

    # -- binding -----------------------------------------------------------
    def resolve(self, query: str, *, kind: str = "street",
                hint: Optional["GeometricHint"] = None, threshold: float = 0.9,
                margin: float = 0.04) -> Binding:
        """Bind a place name to one entry: override, then exact class rules, then geometry."""
        q = _as_str(query, "query", allow_empty=False)
        kind = _as_choice(kind, ENTRY_KINDS, "kind")
        qk = self.key(q)
        if qk in self._overrides and self._entries[self._overrides[qk]].kind == kind:
            eid = self._overrides[qk]
            return Binding("bound", "entry", eid, [eid], [1.0], 0.0, "override", "manual override")
        if kind == "street":
            rows = self._name_rows("street")
            if not rows:
                return Binding("unresolved", "entry", None, [], [], 1.0, "resolver",
                               "no street entries")
            res = resolve_street_name(q, rows, threshold=threshold, margin=margin)
            cands = [eid for eid, _s in res.accepted]
            scores = [s for _eid, s in res.accepted]
            amb = ambiguity_score(scores)
            if res.status == "bound":
                return self._vetted(Binding(
                    "bound", "entry", res.entry_id, cands, scores, amb, "resolver",
                    "type class %s, specific %r" % (res.query.type_class,
                                                    res.query.specific)), hint)
            if res.status == "unresolved":
                nearest = sorted(set(m.reason for m in res.matches))
                return Binding("unresolved", "entry", None, [], [], 1.0, "resolver",
                               "no admissible street (%s)" % ", ".join(nearest))
        else:
            hits = sorted(set(eid for eid, nm in self._name_rows(kind) if self.key(nm) == qk))
            cands = hits
            scores = [1.0] * len(hits)
            amb = ambiguity_score(scores)
            if len(hits) == 1:
                return self._vetted(Binding("bound", "entry", hits[0], hits, scores, 0.0,
                                            "exact", "exact normalised name"), hint)
            if not hits:
                return Binding("unresolved", "entry", None, [], [], 1.0, "exact",
                               "no %s entry with that name" % kind)
        if hint is not None:
            dis = disambiguate(cands, self, hint)
            if dis.selected is not None:
                return Binding("bound", "entry", dis.selected, cands, scores, amb, "geometric",
                               dis.reason)
            return Binding("ambiguous", "entry", None, dis.kept or cands, scores, amb,
                           "geometric", dis.reason)
        return Binding("ambiguous", "entry", None, cands, scores, amb, "resolver",
                       "%d admissible entries" % len(cands))

    def _vetted(self, b: Binding, hint: Optional["GeometricHint"]) -> Binding:
        """A unique name match must still pass the clause's own geometric hint. A hint ruling
        out the only admissible entry is a contradiction: it is refused (the entry kept as the
        reviewer's candidate), never bound."""
        if hint is None or b.target_id is None:
            return b
        dis = disambiguate([b.target_id], self, hint)
        if dis.selected is not None:
            return b
        return Binding("unresolved", b.target_type, None, [b.target_id], list(b.scores),
                       1.0, "geometric", "the only admissible entry fails the hint (%s)"
                       % dis.reasons.get(b.target_id, dis.reason))

    def resolve_junction(self, name_a: str, name_b: str, *,
                         hint: Optional["GeometricHint"] = None,
                         threshold: float = 0.9, margin: float = 0.04) -> Binding:
        """Bind 'the junction of A and B' to the node(s) both streets share. An ambiguous street
        makes the junction ambiguous (its candidates are the nodes shared by any admissible pair
        of streets), an unresolved one makes it unresolved."""
        ba = self.resolve(name_a, kind="street", threshold=threshold, margin=margin)
        bb = self.resolve(name_b, kind="street", threshold=threshold, margin=margin)
        if ba.status == "unresolved" or bb.status == "unresolved":
            which = name_a if ba.status == "unresolved" else name_b
            return Binding("unresolved", "node", None, [], [], 1.0, "junction",
                           "street %r is unresolved" % which)
        if ba.status != "bound" or bb.status != "bound":
            # a BOUND street contributes only its entry, never margin-rejected runners-up
            sa = [ba.target_id] if ba.status == "bound" else ba.candidates
            sb = [bb.target_id] if bb.status == "bound" else bb.candidates
            pool = set()
            for ea in sa:
                for eb in sb:
                    pool.update(set(self._entries[ea].node_ids) & set(self._entries[eb].node_ids))
            nodes = sorted(pool)
            which = name_a if ba.status != "bound" else name_b
            if not nodes:
                return Binding("unresolved", "node", None, [], [], 1.0, "junction",
                               "street %r is ambiguous and no candidate pair meets" % which)
            return Binding("ambiguous", "node", None, nodes, [1.0] * len(nodes),
                           ambiguity_score([1.0] * len(nodes)), "junction",
                           "street %r is ambiguous" % which)
        na = set(self._entries[ba.target_id or ""].node_ids)
        nb = set(self._entries[bb.target_id or ""].node_ids)
        shared = sorted(na & nb)
        if not shared:
            return Binding("unresolved", "node", None, [], [], 1.0, "junction",
                           "the two streets share no node")
        if len(shared) == 1:
            return Binding("bound", "node", shared[0], shared, [1.0], 0.0, "junction",
                           "unique shared node")
        if hint is not None and hint.point is not None:
            dists = []
            for n in shared:
                if n in self._node_coords:
                    dists.append((math.hypot(self._node_coords[n][0] - hint.point[0],
                                             self._node_coords[n][1] - hint.point[1]), n))
            dists.sort()
            # two nodes both AT the hint point are a dead heat, never an id-order win
            if len(dists) == len(shared) and dists[1][0] > 0.0 \
                    and dists[0][0] <= 0.5 * dists[1][0]:
                return Binding("bound", "node", dists[0][1], shared, [1.0] * len(shared),
                               ambiguity_score([1.0] * len(shared)), "junction",
                               "nearest shared node to the hint point")
        return Binding("ambiguous", "node", None, shared, [1.0] * len(shared),
                       ambiguity_score([1.0] * len(shared)), "junction",
                       "%d shared nodes" % len(shared))


# ===========================================================================
# 8. GEOMETRIC DISAMBIGUATION -- bearing, proximity, contiguity of a chain
# ===========================================================================
def bearing_deg(p: Point, q: Point) -> float:
    """Planar bearing from p to q, degrees clockwise from north (+y), in [0, 360)."""
    a = _as_point(p, "p")
    b = _as_point(q, "q")
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == 0.0 and dy == 0.0:
        raise PolicyCompilerError("bearing of a zero-length segment is undefined.")
    v = math.degrees(math.atan2(dx, dy)) % 360.0
    return 0.0 if v >= 360.0 else v


def angular_difference(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = abs(_as_float(a, "a") - _as_float(b, "b")) % 360.0
    return min(d, 360.0 - d)


def axial_difference(a: float, b: float) -> float:
    """Difference between two UNDIRECTED line orientations, in [0, 90]."""
    d = angular_difference(a, b)
    return min(d, 180.0 - d)


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    """Euclidean distance from p to the closed segment ab."""
    px, py = _as_point(p, "p")
    ax, ay = _as_point(a, "a")
    bx, by = _as_point(b, "b")
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    # in units of the segment's size: |d|^2 neither overflows nor underflows
    sc = max(abs(dx), abs(dy))
    ux, uy = dx / sc, dy / sc
    t = (((px - ax) / sc) * ux + ((py - ay) / sc) * uy) / (ux * ux + uy * uy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def point_polyline_distance(p: Point, coords: Sequence[Point]) -> float:
    """Distance from p to a polyline (a single point counts as a polyline)."""
    p = _as_point(p, "p")
    pts = [_as_point(c, "coords") for c in _as_seq(coords, "coords")]
    if not pts:
        raise PolicyCompilerError("polyline needs at least one point.")
    if len(pts) == 1:
        return math.hypot(p[0] - pts[0][0], p[1] - pts[0][1])
    return min(point_segment_distance(p, pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def polyline_length(coords: Sequence[Point]) -> float:
    """Total length of a polyline."""
    pts = [_as_point(c, "coords") for c in _as_seq(coords, "coords")]
    return math.fsum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                     for i in range(len(pts) - 1))


def polyline_bearing(coords: Sequence[Point]) -> float:
    """Bearing of the chord from the first to the last vertex."""
    pts = [_as_point(c, "coords") for c in _as_seq(coords, "coords")]
    if len(pts) < 2:
        raise PolicyCompilerError("a bearing needs at least two points.")
    return bearing_deg(pts[0], pts[-1])


@dataclass
class ChainContiguity:
    """Is a named chain of edges one connected run, and in what order?"""
    n_edges: int
    n_components: int
    components: List[List[str]]
    is_contiguous: bool
    is_simple_path: bool
    ordered_nodes: List[str]
    gaps: int


def chain_contiguity(edge_ids: Sequence[str],
                     endpoints: Dict[str, Tuple[str, str]]) -> ChainContiguity:
    """Connected components of the edges of a named street, and its walk order if a path."""
    eids = sorted(set(_as_id_list(edge_ids, "edge_ids")))
    if not isinstance(endpoints, dict):
        raise PolicyCompilerError("endpoints must map edge id -> (u, v).")
    for e in eids:
        uv = endpoints.get(e)
        if isinstance(uv, (str, bytes)) or not isinstance(uv, (tuple, list)) or len(uv) != 2:
            raise PolicyCompilerError("edge %r has no (u, v) endpoints." % e)
        _as_id(uv[0], "u")
        _as_id(uv[1], "v")
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in eids:
        u, v = endpoints[e]
        for n in (u, v):
            parent.setdefault(n, n)
        ru, rv = find(u), find(v)
        parent[max(ru, rv)] = min(ru, rv)
    groups: Dict[str, List[str]] = {}
    for e in eids:
        groups.setdefault(find(endpoints[e][0]), []).append(e)
    comps = sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
    deg: Dict[str, int] = {}
    for n in (n for e in eids for n in endpoints[e]):
        deg[n] = deg.get(n, 0) + 1
    contiguous = len(comps) <= 1
    ends = sorted(n for n in deg if deg[n] == 1)
    simple = (contiguous and bool(eids) and all(d <= 2 for d in deg.values())
              and len(ends) == 2 and all(endpoints[e][0] != endpoints[e][1] for e in eids))
    order: List[str] = []
    if simple:
        adj: Dict[str, List[Tuple[str, str]]] = {}
        for e in eids:
            u, v = endpoints[e]
            adj.setdefault(u, []).append((v, e))
            adj.setdefault(v, []).append((u, e))
        cur, used = ends[0], set()
        order.append(cur)
        while True:
            nxt = [(n, e) for n, e in sorted(adj[cur], key=lambda t: (t[1], t[0])) if e not in used]
            if not nxt:
                break
            n, e = nxt[0]
            used.add(e)
            order.append(n)
            cur = n
    return ChainContiguity(len(eids), len(comps), comps, contiguous, simple, order,
                           max(0, len(comps) - 1))


@dataclass(frozen=True)
class GeometricHint:
    """Geometric context in a clause: a point, a direction, neighbours, a district (frozen)."""
    point: Optional[Point] = None
    max_distance: Optional[float] = None
    bearing: Optional[float] = None
    bearing_tolerance: float = 30.0
    adjacent_to: Tuple[str, ...] = ()
    district: Optional[str] = None

    def __post_init__(self) -> None:
        put = object.__setattr__.__get__(self)  # frozen: bind the setter once
        if self.point is not None:
            put("point", _as_point(self.point, "point"))
        if self.max_distance is not None:
            put("max_distance", _as_positive(self.max_distance, "max_distance"))
            if self.point is None:
                raise PolicyCompilerError("max_distance needs a point.")
        if self.bearing is not None:
            b = _as_float(self.bearing, "bearing")
            if not (0.0 <= b < 360.0):
                raise PolicyCompilerError("bearing must lie in [0, 360) (got %g)." % b)
            put("bearing", b)
        tol = _as_float(self.bearing_tolerance, "bearing_tolerance")
        if not (0.0 <= tol <= 180.0):
            raise PolicyCompilerError("bearing_tolerance must lie in [0, 180].")
        put("bearing_tolerance", tol)
        put("adjacent_to", tuple(sorted(set(_as_id_list(self.adjacent_to, "adjacent_to")))))
        if self.district is not None:
            put("district", _as_id(self.district, "district"))

    def to_dict(self) -> Dict[str, Any]:
        """Schema-shaped camelCase dict."""
        d: Dict[str, Any] = {}
        if self.point is not None:
            d["point"] = [self.point[0], self.point[1]]
        if self.max_distance is not None:
            d["maxDistance"] = self.max_distance
        if self.bearing is not None:
            d["bearing"] = self.bearing
            d["bearingTolerance"] = self.bearing_tolerance
        if self.adjacent_to:
            d["adjacentTo"] = list(self.adjacent_to)
        if self.district is not None:
            d["district"] = self.district
        return d

    @classmethod
    def from_dict(cls, d: Any) -> "GeometricHint":
        """Validate against ``HINT_SCHEMA`` and build."""
        errs = validate_json(d, HINT_SCHEMA)
        if errs:
            raise PolicyCompilerError("invalid hint: %s" % "; ".join(errs))
        return cls(point=(tuple(d["point"]) if "point" in d else None),
                   max_distance=d.get("maxDistance"), bearing=d.get("bearing"),
                   bearing_tolerance=d.get("bearingTolerance", 30.0),
                   adjacent_to=tuple(d.get("adjacentTo", ())), district=d.get("district"))


@dataclass
class Disambiguation:
    """Which candidates survive the geometric tests, and the one chosen (if any)."""
    selected: Optional[str]
    kept: List[str]
    distances: Dict[str, float]
    reasons: Dict[str, str]
    reason: str


def disambiguate(candidate_ids: Sequence[str], gazetteer: Gazetteer,
                 hint: GeometricHint) -> Disambiguation:
    """Filter candidates by district, distance, axial bearing and adjacency; pick if unique.
    When several survive and a point is given, the nearest is chosen only if it is at most HALF
    as far as the runner-up -- a relative rule, so the decision is the same in metres,
    kilometres or degrees."""
    if not isinstance(gazetteer, Gazetteer):
        raise PolicyCompilerError("disambiguate needs a Gazetteer.")
    if not isinstance(hint, GeometricHint):
        raise PolicyCompilerError("disambiguate needs a GeometricHint.")
    ids = sorted(set(_as_id_list(candidate_ids, "candidate_ids")))
    for i in ids:
        gazetteer.get(i)
    adj_nodes = set(n for a in hint.adjacent_to for n in gazetteer.get(a).node_ids)
    kept: List[str] = []
    reasons: Dict[str, str] = {}
    dist: Dict[str, float] = {}
    for i in ids:
        e = gazetteer.get(i)
        if hint.district is not None and e.district != hint.district:
            reasons[i] = "other district"
            continue
        if hint.point is not None and e.coords:
            dist[i] = point_polyline_distance(hint.point, e.coords)
            if hint.max_distance is not None and dist[i] > hint.max_distance:
                reasons[i] = "too far"
                continue
        if hint.bearing is not None and len(e.coords) >= 2 and e.coords[0] != e.coords[-1]:
            if axial_difference(polyline_bearing(e.coords), hint.bearing) > hint.bearing_tolerance:
                reasons[i] = "wrong orientation"
                continue
        if hint.adjacent_to and not (set(e.node_ids) & adj_nodes):
            reasons[i] = "not contiguous with %s" % ", ".join(hint.adjacent_to)
            continue
        reasons[i] = "kept"
        kept.append(i)
    if len(kept) == 1:
        return Disambiguation(kept[0], kept, dist, reasons, "unique survivor of geometric tests")
    if not kept:
        return Disambiguation(None, kept, dist, reasons, "no candidate survives the geometric tests")
    if hint.point is not None and all(k in dist for k in kept):
        ranked = sorted(kept, key=lambda k: (dist[k], k))
        # Two candidates both AT the point (distance 0) are a dead heat, not a factor-of-two
        # win; the id order must never decide a binding.
        if dist[ranked[1]] > 0.0 and dist[ranked[0]] <= 0.5 * dist[ranked[1]]:
            return Disambiguation(ranked[0], kept, dist, reasons,
                                  "nearest by a factor of at least two")
    return Disambiguation(None, kept, dist, reasons, "%d candidates remain" % len(kept))


# ===========================================================================
# 9. AMBIGUITY SCORING AND THE ABSTENTION QUEUE
# ===========================================================================
ABSTENTION_REASONS = ("span_unverified", "span_fuzzy", "binding_unresolved",
                      "binding_ambiguous", "no_target_elements", "missing_baseline",
                      "conflict", "wording_ambiguous", "unrepresentable")


def ambiguity_score(scores: Sequence[float]) -> float:
    """``s2 / s1`` of the two best candidate scores; 0 for one, 1 for none. Scale-free by
    construction (a common factor cancels), 0 when the leader is unchallenged and 1 at a dead
    heat."""
    vals = sorted(_as_vector(scores, "scores"), reverse=True)
    if any(v < 0.0 for v in vals):
        raise PolicyCompilerError("scores must be non-negative.")
    if not vals or vals[0] == 0.0:
        return 1.0
    if len(vals) == 1:
        return 0.0
    return vals[1] / vals[0]


def normalised_entropy(scores: Sequence[float]) -> float:
    """Shannon entropy of the normalised scores divided by ``log n`` (0 for n <= 1)."""
    vals = _as_vector(scores, "scores")
    if any(v < 0.0 for v in vals):
        raise PolicyCompilerError("scores must be non-negative.")
    tot = math.fsum(vals)
    n = len(vals)
    if n <= 1 or tot == 0.0:
        return 0.0
    h = -math.fsum((v / tot) * math.log(v / tot) for v in vals if v > 0.0)
    return h / math.log(n)


@dataclass
class AbstentionItem:
    """One clause the compiler refused to act on, and why."""
    clause_id: str
    reason: str
    ambiguity: float
    candidates: Tuple[str, ...]
    quote: str
    page: int
    document_id: str
    detail: str = ""

    def __post_init__(self) -> None:
        self.clause_id = _as_id(self.clause_id, "clause_id")
        self.reason = _as_choice(self.reason, ABSTENTION_REASONS, "reason")
        self.ambiguity = _as_fraction(self.ambiguity, "ambiguity")
        self.candidates = tuple(sorted(set(_as_id_list(self.candidates, "candidates"))))

    def as_dict(self) -> Dict[str, Any]:
        """JSON-able camelCase dict."""
        return {"clauseId": self.clause_id, "reason": self.reason,
                "ambiguity": self.ambiguity, "candidates": list(self.candidates),
                "quote": self.quote, "page": self.page, "documentId": self.document_id,
                "detail": self.detail}


class AbstentionQueue:
    """Human-review queue: most ambiguous first, ties on (clause id, reason). ``resolve``
    accepts only one of the item's own candidates, so a human reviewer can choose but the queue
    can never mint a new identifier."""

    def __init__(self) -> None:
        self._items: Dict[Tuple[str, str], AbstentionItem] = {}
        self._resolved: Dict[str, Optional[str]] = {}

    def add(self, item: AbstentionItem) -> None:
        """Enqueue (a repeat of the same clause and reason replaces the old one)."""
        if not isinstance(item, AbstentionItem):
            raise PolicyCompilerError("add needs an AbstentionItem.")
        self._items[(item.clause_id, item.reason)] = item

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> List[AbstentionItem]:
        """Queue order: ambiguity descending, then clause id, then reason."""
        return sorted(self._items.values(), key=lambda it: (-it.ambiguity, it.clause_id, it.reason))

    def clause_ids(self) -> List[str]:
        """Distinct clause ids awaiting review (sorted)."""
        return sorted(set(cid for cid, _r in self._items))

    def resolve(self, clause_id: str, choice: str) -> str:
        """Record a reviewer's choice among a clause's own candidates; dequeue it."""
        cid = _as_id(clause_id, "clause_id")
        ch = _as_id(choice, "choice")
        mine = [it for key, it in sorted(self._items.items()) if key[0] == cid]
        if not mine:
            raise PolicyCompilerError("clause %r is not in the queue." % cid)
        allowed = set(c for it in mine for c in it.candidates)
        if ch not in allowed:
            raise PolicyCompilerError("%r is not one of the candidates for %r: %s"
                                      % (ch, cid, sorted(allowed)))
        for key in [k for k in sorted(self._items) if k[0] == cid]:
            del self._items[key]
        self._resolved[cid] = ch
        return ch

    def reject(self, clause_id: str) -> None:
        """A reviewer decides the clause binds nothing; dequeue it."""
        cid = _as_id(clause_id, "clause_id")
        keys = [k for k in sorted(self._items) if k[0] == cid]
        if not keys:
            raise PolicyCompilerError("clause %r is not in the queue." % cid)
        for key in keys:
            del self._items[key]
        self._resolved[cid] = None

    @property
    def resolved(self) -> Dict[str, Optional[str]]:
        """Copy of the reviewer decisions so far."""
        return dict(self._resolved)

    def abstention_rate(self, n_clauses: int) -> float:
        """Distinct queued clauses over the number of clauses compiled."""
        n = _as_int(n_clauses, "n_clauses", minimum=1)
        return len(self.clause_ids()) / n

    def as_list(self) -> List[Dict[str, Any]]:
        """JSON-able queue in review order."""
        return [it.as_dict() for it in self.items()]


# ===========================================================================
# 10. CONDITIONAL AND TEMPORAL PREDICATES -- time windows, days, vehicles
# ===========================================================================
_MINUTES_PER_DAY = 1440


@dataclass(frozen=True)
class TimeWindow:
    """A daily window ``[start, end)`` in minutes after midnight; start > end wraps."""
    start_min: int
    end_min: int

    def __post_init__(self) -> None:
        s = _as_int(self.start_min, "start_min", minimum=0)
        e = _as_int(self.end_min, "end_min", minimum=0)
        if s > _MINUTES_PER_DAY or e > _MINUTES_PER_DAY:
            raise PolicyCompilerError("time-window minutes must lie in [0, 1440].")
        if s == e:
            raise PolicyCompilerError("an empty time window (start == end) is not allowed.")
        if s == _MINUTES_PER_DAY and e == 0:
            # 24:00 -> 00:00 is midnight to midnight read the wrong way round: it would wrap
            # into a window holding no minute at all.
            raise PolicyCompilerError("the time window 24:00-00:00 is empty.")
        object.__setattr__(self, "start_min", s)
        object.__setattr__(self, "end_min", e)

    @property
    def wraps(self) -> bool:
        """True when the window runs past midnight."""
        return self.start_min > self.end_min

    def segments(self) -> List[Tuple[int, int]]:
        """Non-wrapping half-open pieces of the window."""
        if not self.wraps:
            return [(self.start_min, self.end_min)]
        out = [(self.start_min, _MINUTES_PER_DAY)]
        if self.end_min > 0:
            out.insert(0, (0, self.end_min))
        return out

    def duration(self) -> int:
        """Length in minutes."""
        return sum(b - a for a, b in self.segments())

    def contains(self, minute: int) -> bool:
        """Is ``minute`` (0..1439) inside the window?"""
        m = _as_int(minute, "minute", minimum=0)
        if m >= _MINUTES_PER_DAY:
            raise PolicyCompilerError("minute must lie in [0, 1440).")
        return any(a <= m < b for a, b in self.segments())

    def overlaps(self, other: "TimeWindow") -> bool:
        """Do two windows share at least one minute?"""
        if not isinstance(other, TimeWindow):
            raise PolicyCompilerError("overlaps needs a TimeWindow.")
        return any(a0 < b1 and b0 < a1 for a0, a1 in self.segments()
                   for b0, b1 in other.segments())

    def to_list(self) -> List[int]:
        """``[start_min, end_min]``."""
        return [self.start_min, self.end_min]


def _parse_clock(text: str) -> int:
    """'07:30', '0730', '7.30', '24:00' -> minutes after midnight."""
    m = re.match(r"^\s*(\d{1,2})[:.]?(\d{2})\s*$", text)
    if not m:
        raise PolicyCompilerError("not a clock time: %r" % text)
    h, mi = int(m.group(1)), int(m.group(2))
    if mi >= 60 or h > 24 or (h == 24 and mi != 0):
        raise PolicyCompilerError("not a valid clock time: %r" % text)
    return h * 60 + mi


def parse_time_window(text: str) -> TimeWindow:
    """Parse ``'07:00-10:00'`` (also ``0700-1000``, ``'7.00 to 10.00'``)."""
    t = _as_str(text, "text", allow_empty=False)
    m = re.match(r"^\s*(\d{1,2}[:.]?\d{2})\s*(?:-|to)\s*(\d{1,2}[:.]?\d{2})\s*$", t)
    if not m:
        raise PolicyCompilerError("not a time window: %r" % t)
    return TimeWindow(_parse_clock(m.group(1)), _parse_clock(m.group(2)))


@dataclass(frozen=True)
class Scenario:
    """One evaluation context: minute of day, day of week, vehicle class, tags."""
    minute: int
    day: str
    vehicle_class: str = "car"
    tags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        m = _as_int(self.minute, "minute", minimum=0)
        if m >= _MINUTES_PER_DAY:
            raise PolicyCompilerError("minute must lie in [0, 1440).")
        object.__setattr__(self, "minute", m)
        _as_choice(self.day, DAYS, "day")
        _as_choice(self.vehicle_class, VEHICLE_CLASSES, "vehicle_class")
        object.__setattr__(self, "tags", tuple(sorted(set(_as_id_list(self.tags, "tags")))))


@dataclass(frozen=True)
class Predicate:
    """Scenario-conditional applicability of a clause (immutable). Holds iff the minute is in
    ANY window, the day and the class are listed (empty = all) and the class is not exempt, and
    every named tag is present. Frozen: compiled fixings SHARE it, so a later edit would rewrite
    them."""
    time_windows: Tuple[TimeWindow, ...] = ()
    days: Tuple[str, ...] = ()
    vehicle_classes: Tuple[str, ...] = ()
    exempt_classes: Tuple[str, ...] = ()
    scenario_tags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        tws = self.time_windows
        if isinstance(tws, TimeWindow):
            tws = (tws,)
        tws = _as_seq(tws, "time_windows")
        for tw in tws:
            if not isinstance(tw, TimeWindow):
                raise PolicyCompilerError("time_windows must hold TimeWindow objects.")
        object.__setattr__(self, "time_windows",
                           tuple(sorted(tws, key=lambda w: (w.start_min, w.end_min))))
        for nm, allowed in (("days", DAYS), ("vehicle_classes", VEHICLE_CLASSES),
                            ("exempt_classes", VEHICLE_CLASSES)):
            vals = getattr(self, nm)
            if isinstance(vals, str):
                raise PolicyCompilerError("%s must be a sequence, not a string." % nm)
            vals = _as_seq(vals, nm)
            for v in vals:
                _as_choice(v, allowed, nm)
            object.__setattr__(self, nm, tuple(sorted(set(vals), key=list(allowed).index)))
        object.__setattr__(self, "scenario_tags",
                           tuple(sorted(set(_as_id_list(self.scenario_tags, "scenario_tags")))))

    @property
    def is_trivial(self) -> bool:
        """True when the predicate restricts nothing (always holds)."""
        return not (self.time_windows or self.days or self.vehicle_classes
                    or self.exempt_classes or self.scenario_tags)

    def allowed_classes(self) -> Tuple[str, ...]:
        """Vehicle classes the predicate can hold for."""
        base = self.vehicle_classes or VEHICLE_CLASSES
        return tuple(v for v in base if v not in self.exempt_classes)

    def allowed_days(self) -> Tuple[str, ...]:
        """Days the predicate can hold on."""
        return self.days or DAYS

    def holds(self, scenario: Scenario) -> bool:
        """Evaluate against one scenario."""
        if not isinstance(scenario, Scenario):
            raise PolicyCompilerError("holds needs a Scenario.")
        if self.time_windows and not any(w.contains(scenario.minute) for w in self.time_windows):
            return False
        if scenario.day not in self.allowed_days():
            return False
        if scenario.vehicle_class not in self.allowed_classes():
            return False
        return all(t in scenario.tags for t in self.scenario_tags)

    def is_satisfiable(self) -> bool:
        """Can ANY scenario satisfy the predicate?"""
        return bool(self.allowed_classes()) and bool(self.allowed_days())

    def intersects(self, other: "Predicate") -> bool:
        """Is there a scenario in which BOTH predicates hold?  (Exact.)"""
        if not isinstance(other, Predicate):
            raise PolicyCompilerError("intersects needs a Predicate.")
        if not (set(self.allowed_days()) & set(other.allowed_days())):
            return False
        if not (set(self.allowed_classes()) & set(other.allowed_classes())):
            return False
        if self.time_windows and other.time_windows:
            if not any(a.overlaps(b) for a in self.time_windows for b in other.time_windows):
                return False
        return True

    def minutes_per_week(self) -> int:
        """Minutes per week the time/day part holds (vehicle class aside)."""
        per_day = (_MINUTES_PER_DAY if not self.time_windows else
                   len(set(m for w in self.time_windows for a, b in w.segments()
                           for m in range(a, b))))
        return per_day * len(self.allowed_days())

    def to_dict(self) -> Dict[str, Any]:
        """Stable camelCase dict -- all five keys always present."""
        return {"timeWindows": [w.to_list() for w in self.time_windows],
                "days": list(self.days), "vehicleClasses": list(self.vehicle_classes),
                "exemptClasses": list(self.exempt_classes),
                "scenarioTags": list(self.scenario_tags)}

    @classmethod
    def from_dict(cls, d: Any) -> "Predicate":
        """Validate against ``PREDICATE_SCHEMA`` and build."""
        errs = validate_json(d, PREDICATE_SCHEMA)
        if errs:
            raise PolicyCompilerError("invalid predicate: %s" % "; ".join(errs))
        return cls(time_windows=tuple(TimeWindow(a, b) for a, b in d.get("timeWindows", [])),
                   days=tuple(d.get("days", ())),
                   vehicle_classes=tuple(d.get("vehicleClasses", ())),
                   exempt_classes=tuple(d.get("exemptClasses", ())),
                   scenario_tags=tuple(d.get("scenarioTags", ())))


def _words(spec: str) -> Dict[str, Any]:
    """'value: word word; value: word' -> {word: value} (values split on ',' into tuples)."""
    return dict((w, tuple(v.split(",")) if "," in v else v) for part in spec.split(";")
                for v, ws in [part.strip().split(":")] for w in ws.split())


_DAY_WORDS = _words("mon: monday mon mondays; tue: tuesday tue tues tuesdays; wed: wednesday "
                    "wed wednesdays; thu: thursday thu thur thurs thursdays; fri: friday fri "
                    "fridays; sat: saturday sat saturdays; sun: sunday sun sundays")
_VEHICLE_WORDS = _words("car: car cars; taxi: taxi taxis; bus: bus buses; coach: coach coaches; "
                        "lgv: lgv lgvs van vans; hgv: hgv hgvs lorry lorries; motorcycle: "
                        "motorcycle motorcycles; cycle: cycle cycles bicycle bicycles cyclists; "
                        "emergency: emergency; pedestrian: pedestrian pedestrians")
_MOTOR = tuple(v for v in VEHICLE_CLASSES if v not in ("cycle", "pedestrian"))
# The predicate grammar is CLOSED: every token must be consumed by a rule below, or the phrase
# is refused.  Range separators, list joiners (a comma joins items of one kind only), exemption
# markers (longest first), the activities a leading "No" may name, and the week words.
_SEPS = tuple((w,) for w in "- to until till through thru".split())
_JOINS = (("and",), ("or",), ("&",), (",",))
_MARKERS = tuple(tuple(m.split()) for m in (
    "with the exception of", "except for", "other than", "save for", "apart from", "but not",
    "except", "excepting", "excluding"))
_ACTIVITIES = tuple(tuple(a.split()) for a in (
    "entry", "loading", "unloading", "waiting", "stopping", "parking", "left turn",
    "right turn", "u - turn", "u turn"))
_WEEK = _words("mon,tue,wed,thu,fri: weekday weekdays; sat,sun: weekend weekends")
_ON_NEXT = set(_DAY_WORDS) | set(_WEEK) | {"school", "schooldays"}      # "on" precedes these
_AT_NEXT = {"weekend", "weekends", "any", "anytime", "all"}              # "at" precedes these
_Item = Tuple[str, Any]          # (kind, value): 'c' classes, 'd' days, 'w' window, 's' school


class _Cursor:
    """A position in a token list; ``take`` consumes the first matching sequence or nothing,
    ``word`` consumes one token found in a table and returns its value."""

    def __init__(self, toks: List[str]) -> None:
        self.t, self.i = toks, 0

    def peek(self, k: int = 0) -> str:
        return self.t[self.i + k] if self.i + k < len(self.t) else ""

    def take(self, *seqs: Tuple[str, ...]) -> bool:
        for seq in seqs:
            if tuple(self.t[self.i:self.i + len(seq)]) == seq:
                self.i += len(seq)
                return True
        return False

    def word(self, table: Dict[str, Any]) -> Any:
        v = table.get(self.peek())
        self.i += v is not None
        return v


def _clock(g: _Cursor) -> Optional[Tuple[Optional[int], Optional[str], int, str, bool]]:
    """(hour, 'a' / 'p' / None, minutes or -1 for a bare hour, the token, is midnight)."""
    g.i += g.peek() == "12" and g.peek(1) in ("noon", "midday", "midnight")
    w = g.peek()
    m = re.match(r"([0-9]{2})([0-9]{2})$|([0-9]{1,2})(?:[:.]([0-9]{2}))?$", w)
    if w in ("noon", "midday", "midnight") or m:
        g.i += 1
    if not m:
        return (None, None, 0 if w == "midnight" else 720, w, w == "midnight") if w in (
            "noon", "midday", "midnight") else None
    h, mi = int(m.group(1) or m.group(3)), int(m.group(2) or m.group(4) or 0)
    if g.take(("am",), ("pm",)):
        mer = g.t[g.i - 1][0]
        if m.group(1) or not 1 <= h <= 12 or mi >= 60:
            raise PolicyCompilerError("not a 12-hour clock time: %r" % w)
        return h, mer, (h % 12) * 60 + mi + (720 if mer == "p" else 0), w, False
    if m.group(3) and m.group(4) is None:
        return h, None, -1, w, False
    g.take(("hrs",), ("hours",), ("hr",))
    return h, None, _parse_clock(w), w, False


def _window(c1: Tuple[Any, ...], c2: Tuple[Any, ...]) -> TimeWindow:
    """A clock range; one meridiem serves a bare (or dotted, unpadded) end, the same one
    unless that empties or wraps the range ("10-2pm" = 10am-2pm, "11-1am" = 11pm-1am).  With
    no meridiem at all, one end must be plainly 24-hour (zero-padded, 4-digit or past 12)."""
    (h1, m1, a, w1, _x), (h2, m2, b, w2, mid) = c1, c2
    plain = [h is not None and (w[0] == "0" or len(w) == 4 and w.isdigit() or h > 12)
             for h, w in ((h1, w1), (h2, w2))]
    if m1 is None and m2 is None and (min(a, b) < 0 or (h1, h2) != (None, None)
                                      and not any(plain)):     # "7-10", "8.00-6.30"
        raise PolicyCompilerError("a range with no am / pm and no 24-hour clock is ambiguous.")
    # a plainly 24-hour end ('08:00', '1000') never borrows: '1000-11pm' is 10:00-23:00
    bare, raw, mer, fixed = (h1, a, m2, plain[0]) if m1 is None else (h2, b, m1, plain[1])
    if (m1 is None) != (m2 is None) and 1 <= (bare or 0) <= 12 and not fixed:
        v = [(bare % 12) * 60 + max(raw, 0) % 60 + d for d in (0, 720)]
        sm, ot = (v[1], v[0]) if mer == "p" else (v[0], v[1])    # same / other meridiem
        a, b = (sm if sm < b else ot, b) if m1 is None else (a, sm if a < sm else ot)
    return TimeWindow(a, _MINUTES_PER_DAY if mid and b == 0 else b)   # "... to midnight"


def _span(g: _Cursor, read: Callable[[_Cursor], Any], single: bool) -> Optional[Tuple[Any, Any]]:
    """'X SEP Y', 'from X SEP Y', 'between (the hours of) X and Y' -- or a lone X if
    ``single`` -- where ``read`` reads one end; the cursor is restored on failure."""
    s, between = g.i, g.take(("between",))
    if between and read is _clock:
        g.take(("the", "hours", "of"))
    frm = not between and g.take(("from",))
    x = read(g)
    if x is not None and g.take(*((("and",),) if between else _SEPS)):
        y = read(g)
        if y is not None:
            return x, y
    elif x is not None and single and not (between or frm):
        return x, None
    g.i = s
    return None


def _class_item(g: _Cursor) -> Optional[_Item]:
    """A vehicle word, '(light|heavy|large) goods vehicles' (plain: vans AND lorries, as in
    road-traffic law; 'over 3.5 tonnes' -- the legal LGV / HGV line -- makes it HGV), 'motor
    vehicles' (no cycles or pedestrians), 'motor cycles', 'emergency vehicles'."""
    s, veh, size = g.i, (("vehicles",), ("vehicle",)), g.peek()
    if g.take(("motor",)):
        if g.take(*veh):
            return "c", _MOTOR
        g.take(("-",))
        if g.take(("cycles",), ("cycle",)):
            return "c", ("motorcycle",)
    g.i = s + (size in ("light", "heavy", "large"))
    if g.take(("goods",)) and g.take(*veh):
        cls = ("lgv",) if size == "light" else ("lgv", "hgv") if size == "goods" else ("hgv",)
        if g.take(("over",), ("exceeding",), ("above",)):
            if size == "light" or not g.take(("3.5", "tonnes"), ("3.5", "tonne"), ("3.5", "t")):
                raise PolicyCompilerError("a weight limit other than 3.5 t is not a class.")
            cls = ("hgv",)
        return "c", cls
    g.i = s
    c = g.word(_VEHICLE_WORDS)
    if c == "emergency":
        g.take(*veh)
    return None if c is None else ("c", (c,))


def _item(g: _Cursor) -> Optional[_Item]:
    """One time, day, school-day or (unprefixed) class item; 'on' may precede a day or school
    days, 'at' only weekends or any time ("on Mondays", "at weekends", "at any time")."""
    s = g.i
    pre = g.t[s] if g.take(("on",), ("at",)) else ""
    if pre and g.peek() not in (_AT_NEXT if pre == "at" else _ON_NEXT):
        g.i = s
        return None
    if g.take(("school", "days"), ("school", "day"), ("schooldays",)):
        return "s", None
    if g.take(("any", "time"), ("anytime",), ("all", "times")):
        return "w", None
    wk = g.word(_WEEK)
    sp = None if wk else _span(g, _clock, False)
    if wk or sp:
        return ("d", wk) if wk else ("w", _window(*sp))
    sp = _span(g, lambda c: c.word(_DAY_WORDS), True)
    if sp is not None and sp[0] == sp[1]:
        raise PolicyCompilerError("the day range %s-%s is ambiguous." % sp)
    if sp is not None:
        i = DAYS.index(sp[0])
        return "d", tuple(DAYS[(i + k) % 7] for k in range(
            (DAYS.index(sp[1] or sp[0]) - i) % 7 + 1))
    it = None if pre else _class_item(g)
    g.i = g.i if it else s
    return it


def _group(g: _Cursor, first: _Item) -> List[_Item]:
    """``first`` plus every further item OF THE SAME KIND joined by and / or / & / comma."""
    out = [first]
    while True:
        s = g.i
        it = _item(g) if g.take(*_JOINS) else None
        if it is None or it[0] != first[0]:
            g.i = s
            return out
        out.append(it)


def _window_union(ws: Sequence[TimeWindow], drop: Sequence[TimeWindow]) -> List[TimeWindow]:
    """Minutes in ``ws`` (all day when empty) but not in ``drop``, as windows."""
    keep = [(not ws or any(w.contains(m) for w in ws)) and not any(w.contains(m) for w in drop)
            for m in range(_MINUTES_PER_DAY)]
    runs, m = [], 0
    for flag, grp in itertools.groupby(keep):
        n = len(list(grp))
        if flag:
            runs.append([m, m + n])
        m += n
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == _MINUTES_PER_DAY:
        runs[-1][1] = runs.pop(0)[1]                  # rejoin the run across midnight
    if not runs:
        raise PolicyCompilerError("the exempt times cover every restricted minute.")
    return [TimeWindow(s, e) for s, e in runs]


def _refuse(text: str, tok: str) -> bool:
    raise PolicyCompilerError("%r: cannot read %r here." % (text, tok))


def parse_predicate(text: str) -> Predicate:
    """Read order prose with a CLOSED grammar: ``[no (activity | classes)] group* [only]
    [marker exemption (joiner exemption)*]`` where a group is days, times, classes or school
    days (each kind at most once; 'only' may follow a final day / time group) and an exemption
    is one class, day or time item ("Mon-Fri 7am-10am except buses and Saturdays").  Every
    token must be consumed; anything unread, ambiguous or unrepresentable raises."""
    t = re.sub(r"[\u2010-\u2015\u2212]", "-", _as_str(text, "text").lower())
    t = re.sub(r"(?<![a-z])([ap])\.\s?m(?![a-z])\.?", r"\1m", t)          # a.m. -> am
    g = _Cursor(re.findall(r"[0-9]+(?:[:.][0-9]+)?|[a-z]+|\S", t))
    g.t = g.t[:-1] if g.t[-1:] == ["."] else g.t
    head: Dict[str, List[_Item]] = {}
    if g.take(("no",)) and not g.take(*_ACTIVITIES):
        head["c"] = _group(g, _class_item(g) or ("c", _refuse(text, g.peek())))
    marked = False
    while g.peek() and not marked:
        s, marked = g.i, g.take(*_MARKERS)
        it = None if marked else _item(g)
        if it is not None and it[0] not in head:
            head[it[0]] = _group(g, it)
            if it[0] != "c" and g.take(("only",)) and g.peek():      # 'only' ends the head
                marked = g.take(*_MARKERS) or _refuse(text, g.peek())
        elif not marked:
            _refuse(text, g.t[s])
    tail: List[_Item] = []
    while marked:                       # one-kind exemptions; a comma joins items of one kind
        sep, it = g.t[g.i - 1], _item(g)
        if it is None or it[1] is None or (sep == "," and tail[-1][0] != it[0]):  # no school
            _refuse(text, g.peek() or "the end")
        tail.append(it)
        marked = g.take(*_JOINS) or bool(g.peek()) and _refuse(text, g.peek())
    wins = [w for _k, w in head.get("w", [])]
    if None in wins and len(wins) > 1:
        raise PolicyCompilerError("%r: 'any time' and a time range together." % text)
    ex = dict((k, [v for kk, v in tail if kk == k]) for k in "cdw")
    wins = [w for w in wins if w is not None]
    wins = _window_union(wins, ex["w"]) if ex["w"] else wins
    days = [d for _k, v in head.get("d", []) for d in v]
    days = [d for d in (days or DAYS) if not any(d in v for v in ex["d"])] if ex["d"] else days
    if not days and ex["d"]:
        raise PolicyCompilerError("%r exempts every day it names." % text)
    return Predicate(time_windows=tuple(wins), days=tuple(days),
                     vehicle_classes=tuple(c for _k, v in head.get("c", []) for c in v),
                     exempt_classes=tuple(c for v in ex["c"] for c in v),
                     scenario_tags=("school_days",) if "s" in head else ())


# ===========================================================================
# 11. IN-FORCE RESOLUTION -- amendment DAG, Allen's interval algebra
# ===========================================================================
ALLEN_RELATIONS = ("before", "meets", "overlaps", "starts", "during", "finishes",
                   "equals", "finished_by", "contains", "started_by",
                   "overlapped_by", "met_by", "after")
_ALLEN_INVERSE = dict(zip(ALLEN_RELATIONS, tuple(reversed(ALLEN_RELATIONS))))
_ALLEN_DISJOINT = frozenset(["before", "meets", "met_by", "after"])


@dataclass(frozen=True)
class Interval:
    """A half-open validity period ``[start, end)``; ``end`` may be ``inf`` (open-ended)."""
    start: float
    end: float

    def __post_init__(self) -> None:
        s = _as_float(self.start, "start")
        e = float(self.end) if isinstance(self.end, (int, float)) and not isinstance(self.end, bool) else None
        if e is None or math.isnan(e) or e == -math.inf:
            raise PolicyCompilerError("end must be a real number or +inf.")
        if not e > s:
            raise PolicyCompilerError("an interval needs end > start (got [%g, %g))." % (s, e))
        object.__setattr__(self, "start", s)
        object.__setattr__(self, "end", e)

    def contains_point(self, t: float) -> bool:
        """``start <= t < end``."""
        v = _as_float(t, "t")
        return self.start <= v < self.end


def allen_relation(a: Interval, b: Interval) -> str:
    """The unique Allen relation holding between intervals a and b."""
    if not isinstance(a, Interval) or not isinstance(b, Interval):
        raise PolicyCompilerError("allen_relation needs two Intervals.")
    if a.end < b.start:
        return "before"
    if a.end == b.start:
        return "meets"
    if b.end < a.start:
        return "after"
    if b.end == a.start:
        return "met_by"
    if a.start == b.start and a.end == b.end:
        return "equals"
    if a.start == b.start:
        return "starts" if a.end < b.end else "started_by"
    if a.end == b.end:
        return "finishes" if a.start > b.start else "finished_by"
    if a.start > b.start and a.end < b.end:
        return "during"
    if a.start < b.start and a.end > b.end:
        return "contains"
    return "overlaps" if a.start < b.start else "overlapped_by"


def allen_inverse(relation: str) -> str:
    """The converse relation (``before <-> after``, ``during <-> contains``...)."""
    return _ALLEN_INVERSE[_as_choice(relation, ALLEN_RELATIONS, "relation")]


_ALLEN_TABLE: Dict[Tuple[str, str], Tuple[str, ...]] = {}


def _build_allen_table() -> None:
    """Composition table by exhaustive enumeration over integer endpoints 0..5. A relation
    depends only on the ORDER of the endpoints, and three intervals have at most six distinct
    endpoints, so every configuration is realised on {0..5}; the table is therefore exact, not
    sampled."""
    ivs = [Interval(s, e) for s in range(6) for e in range(s + 1, 6)]
    acc: Dict[Tuple[str, str], set] = {}
    for x in ivs:
        for y in ivs:
            r1 = allen_relation(x, y)
            for z in ivs:
                key = (r1, allen_relation(y, z))
                acc.setdefault(key, set()).add(allen_relation(x, z))
    for key in acc:
        _ALLEN_TABLE[key] = tuple(r for r in ALLEN_RELATIONS if r in acc[key])


def allen_composition(r1: str, r2: str) -> Tuple[str, ...]:
    """Possible relations of (x, z) given x r1 y and y r2 z, in canonical order."""
    a = _as_choice(r1, ALLEN_RELATIONS, "r1")
    b = _as_choice(r2, ALLEN_RELATIONS, "r2")
    if not _ALLEN_TABLE:
        _build_allen_table()
    return _ALLEN_TABLE[(a, b)]


def intervals_overlap(a: Interval, b: Interval) -> bool:
    """Do two half-open periods share a time point?"""
    return allen_relation(a, b) not in _ALLEN_DISJOINT


def date_ordinal(value: Any) -> int:
    """An ISO date string (``'2024-03-01'``) or an integer day number -> day number."""
    if isinstance(value, str):
        try:
            return _dt.date.fromisoformat(value.strip()).toordinal()
        except ValueError:
            raise PolicyCompilerError("not an ISO date: %r" % value)
    if isinstance(value, _dt.date):
        return value.toordinal()
    return _as_int(value, "day")


@dataclass
class Instrument:
    """A legal instrument (order, plan, amendment) with its validity and its edits. ``revokes``
    / ``supersedes`` end a target; ``amends`` + ``clause_changes`` rewrite its clauses (``None``
    deletes).  Effects start on ``start``."""
    instrument_id: str
    start: Any
    end: Any = None
    clause_ids: Tuple[str, ...] = ()
    amends: Tuple[str, ...] = ()
    revokes: Tuple[str, ...] = ()
    supersedes: Tuple[str, ...] = ()
    clause_changes: Dict[str, Optional[str]] = field(default_factory=dict)
    made: Any = None

    def __post_init__(self) -> None:
        self.instrument_id = _as_id(self.instrument_id, "instrument_id")
        self.start = date_ordinal(self.start)
        if self.end is not None:
            self.end = date_ordinal(self.end)
            if self.end <= self.start:
                raise PolicyCompilerError("instrument %s ends before it starts." % self.instrument_id)
        self.made = self.start if self.made is None else date_ordinal(self.made)
        self.clause_ids = tuple(sorted(set(_as_id_list(self.clause_ids, "clause_ids"))))
        for nm in ("amends", "revokes", "supersedes"):
            vals = tuple(sorted(set(_as_id_list(getattr(self, nm), nm))))
            if self.instrument_id in vals:
                raise PolicyCompilerError("instrument %s cannot %s itself." % (self.instrument_id, nm))
            setattr(self, nm, vals)
        if not isinstance(self.clause_changes, dict):
            raise PolicyCompilerError("clause_changes must be a dict.")
        cc: Dict[str, Optional[str]] = {}
        for old in sorted(self.clause_changes, key=str):
            new = self.clause_changes[old]
            cc[_as_id(old, "clause_changes key")] = None if new is None else _as_id(new, "new clause")
        if cc and not self.amends:
            raise PolicyCompilerError("clause_changes need an 'amends' target.")
        for new in cc.values():
            if new is not None and new not in self.clause_ids:
                raise PolicyCompilerError("replacement clause %r must be one of %s's clauses."
                                          % (new, self.instrument_id))
        self.clause_changes = cc

    @property
    def interval(self) -> Interval:
        """Nominal validity ``[start, end)`` (open-ended when ``end`` is None)."""
        return Interval(self.start, math.inf if self.end is None else self.end)

    def targets(self) -> List[Tuple[str, str]]:
        """``(target_id, kind)`` for every instrument this one modifies."""
        out = [(t, "amends") for t in self.amends]
        out += [(t, "revokes") for t in self.revokes]
        out += [(t, "supersedes") for t in self.supersedes]
        return sorted(out)


def _instrument_map(instruments: Sequence[Instrument]) -> Dict[str, Instrument]:
    """Validate a collection of instruments and index it by id."""
    out: Dict[str, Instrument] = {}
    for ins in _as_seq(instruments, "instruments"):
        if not isinstance(ins, Instrument):
            raise PolicyCompilerError("instruments must be Instrument objects.")
        if ins.instrument_id in out:
            raise PolicyCompilerError("duplicate instrument id %r." % ins.instrument_id)
        out[ins.instrument_id] = ins
    for iid in sorted(out):
        for t, kind in out[iid].targets():
            if t not in out:
                raise PolicyCompilerError("%s %s unknown instrument %r." % (iid, kind, t))
    return out


def amendment_edges(instruments: Sequence[Instrument]) -> List[Tuple[str, str, str]]:
    """DAG edges ``(target, modifier, kind)``: the modifier must come after its target."""
    m = _instrument_map(instruments)
    return sorted((t, iid, kind) for iid in sorted(m) for t, kind in m[iid].targets())


def find_amendment_cycle(instruments: Sequence[Instrument]) -> Optional[List[str]]:
    """A cycle in the amendment graph as ``[a, b, ..., a]``, or None if it is a DAG."""
    instruments = _as_seq(instruments, "instruments")    # one pass only: may be an iterator
    m = _instrument_map(instruments)
    succ: Dict[str, List[str]] = dict((i, []) for i in m)
    for t, mod, _k in amendment_edges(instruments):
        succ[t].append(mod)
    for i in succ:
        succ[i] = sorted(set(succ[i]))
    colour = dict((i, 0) for i in m)
    for root in sorted(m):
        if colour[root]:
            continue
        stack: List[Tuple[str, int]] = [(root, 0)]
        path: List[str] = [root]
        colour[root] = 1
        while stack:
            node, k = stack[-1]
            if k < len(succ[node]):
                stack[-1] = (node, k + 1)
                nxt = succ[node][k]
                if colour[nxt] == 1:
                    return path[path.index(nxt):] + [nxt]
                if colour[nxt] == 0:
                    colour[nxt] = 1
                    stack.append((nxt, 0))
                    path.append(nxt)
            else:
                colour[node] = 2
                stack.pop()
                path.pop()
    return None


def topological_order(instruments: Sequence[Instrument]) -> List[str]:
    """Kahn order of the amendment DAG, ties on (start, made, id); raises on a cycle."""
    instruments = _as_seq(instruments, "instruments")
    m = _instrument_map(instruments)
    cyc = find_amendment_cycle(instruments)
    if cyc is not None:
        raise PolicyCompilerError("amendment cycle: %s" % " -> ".join(cyc))
    indeg = dict((i, 0) for i in m)
    succ: Dict[str, List[str]] = dict((i, []) for i in m)
    for t, mod, _k in amendment_edges(instruments):
        if mod not in succ[t]:
            succ[t].append(mod)
            indeg[mod] += 1
    heap = [(m[i].start, m[i].made, i) for i in m if indeg[i] == 0]
    heapq.heapify(heap)
    order: List[str] = []
    while heap:
        _s, _md, i = heapq.heappop(heap)
        order.append(i)
        for j in sorted(succ[i]):
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(heap, (m[j].start, m[j].made, j))
    return order


@dataclass
class InForceState:
    """The resolved legal state on one day."""
    at: int
    order: List[str]
    in_force: List[str]
    not_commenced: List[str]
    expired: List[str]
    revoked: Dict[str, str]
    superseded: Dict[str, str]
    effective: Dict[str, Interval]
    clause_owner: Dict[str, str]
    clauses: List[str]
    deleted_clauses: List[str]

    def is_in_force(self, instrument_id: str) -> bool:
        """Membership test."""
        return instrument_id in self.in_force

    def as_dict(self) -> Dict[str, Any]:
        """JSON-able camelCase dict."""
        return {"at": self.at, "order": list(self.order), "inForce": list(self.in_force),
                "notCommenced": list(self.not_commenced), "expired": list(self.expired),
                "revoked": dict(self.revoked), "superseded": dict(self.superseded),
                "effective": dict((k, [v.start, _json_safe(v.end)])
                                  for k, v in sorted(self.effective.items())),
                "clauses": list(self.clauses), "deletedClauses": list(self.deleted_clauses)}


def resolve_in_force(instruments: Sequence[Instrument], at: Any) -> InForceState:
    """Topologically apply revocations, supersessions and amendments up to day ``at``. Effective
    period: nominal ``[start, end)`` cut at the earliest COMMENCED revoker / superseder; one cut
    on or before its own start never commences (none of its effects happen). Commenced
    amendments are permanent."""
    instruments = _as_seq(instruments, "instruments")
    m = _instrument_map(instruments)
    t = date_ordinal(at)
    order = topological_order(instruments)
    eff_end: Dict[str, float] = dict((i, m[i].interval.end) for i in m)
    revoked: Dict[str, str] = {}
    superseded: Dict[str, str] = {}
    owned: Dict[str, List[str]] = {}
    replacement_ids = set((i, new) for i in m for new in m[i].clause_changes.values()
                          if new is not None)
    for i in order:
        owned[i] = [c for c in m[i].clause_ids if (i, c) not in replacement_ids]
    deleted: List[str] = []
    # An instrument revoked or superseded on or before its own start day never comes into force,
    # so NONE of its effects (revocations, supersessions, amendments) ever happen.  Whether a
    # terminator is itself void depends only on instruments later in the topological order, so
    # one reverse sweep settles every instrument.
    terminators: Dict[str, List[str]] = dict((i, []) for i in m)
    for tgt, mod, kind in amendment_edges(instruments):
        if kind in ("revokes", "supersedes"):
            terminators[tgt].append(mod)
    void: Dict[str, bool] = {}
    for i in reversed(order):
        cuts = [m[j].start for j in terminators[i] if m[j].start <= t and not void[j]]
        void[i] = bool(cuts) and min(cuts) <= m[i].start
    cut_by: Dict[str, Tuple[int, int, str, str]] = {}
    for i in order:
        ins = m[i]
        if ins.start > t or void[i]:
            continue
        for kind, tgts in (("revoked", ins.revokes), ("superseded", ins.supersedes)):
            for tgt in tgts:
                key = (ins.start, 0 if kind == "revoked" else 1, i, kind)
                if tgt not in cut_by or key < cut_by[tgt]:
                    cut_by[tgt] = key
        for old in sorted(ins.clause_changes):
            new = ins.clause_changes[old]
            home = [tg for tg in ins.amends if old in owned[tg]]
            if not home:
                raise PolicyCompilerError("%s amends clause %r, which none of %s holds."
                                          % (i, old, ", ".join(ins.amends)))
            tg = home[0]
            owned[tg] = [c for c in owned[tg] if c != old]
            deleted.append(old)
            if new is not None:
                owned[tg] = sorted(owned[tg] + [new])
    for tgt in sorted(cut_by):
        day, _rank, by, kind = cut_by[tgt]
        if day < eff_end[tgt]:
            eff_end[tgt] = day
            (revoked if kind == "revoked" else superseded)[tgt] = by
    in_force, pending, expired = [], [], []
    effective: Dict[str, Interval] = {}
    for i in sorted(m):
        s = m[i].start
        if eff_end[i] > s:
            effective[i] = Interval(s, eff_end[i])
        if s > t:
            pending.append(i)
        elif t < eff_end[i]:
            in_force.append(i)
        else:
            expired.append(i)
    owner = dict((c, i) for i in in_force for c in owned[i])
    return InForceState(t, order, in_force, pending, expired, revoked, superseded,
                        effective, owner, sorted(owner), sorted(set(deleted)))


def validity_relation(instruments: Sequence[Instrument], a: str, b: str) -> str:
    """Allen relation between the NOMINAL validity periods of two instruments."""
    instruments = _as_seq(instruments, "instruments")
    m = _instrument_map(instruments)
    for x in (_as_id(a, "a"), _as_id(b, "b")):
        if x not in m:
            raise PolicyCompilerError("unknown instrument %r." % x)
    return allen_relation(m[a].interval, m[b].interval)


# ===========================================================================
# 12. DIRECTIONAL WORDING -- modality, polarity and the weight prior
# ===========================================================================
# (cue, modality, strength p).  The LONGEST cue found wins, ties to the earliest position, so
# "must not" beats "must" and "should not" beats "should".
WORDING_CUES: Tuple[Tuple[str, str, float], ...] = (
    ("must not", "prohibition", -1.0), ("shall not", "prohibition", -1.0),
    ("will not be permitted", "prohibition", -1.0), ("not permitted", "prohibition", -1.0),
    ("is prohibited", "prohibition", -1.0), ("are prohibited", "prohibition", -1.0),
    ("prohibited", "prohibition", -1.0), ("may not", "prohibition", -1.0),
    ("must", "obligation", 1.0), ("shall", "obligation", 1.0),
    ("is required", "obligation", 1.0), ("are required", "obligation", 1.0),
    ("will be required", "obligation", 1.0),
    ("strongly encourage", "recommendation", 0.75), ("will prioritise", "recommendation", 0.75),
    ("should", "recommendation", 0.5), ("encourage", "recommendation", 0.5),
    ("prioritise", "recommendation", 0.5), ("promote", "recommendation", 0.5),
    ("support", "recommendation", 0.5), ("seek to", "recommendation", 0.5),
    ("should not", "discouragement", -0.5), ("discourage", "discouragement", -0.5),
    ("resist", "discouragement", -0.5), ("strongly discourage", "discouragement", -0.75),
    ("may", "permission", 0.0), ("can be considered", "permission", 0.0),
    ("ought to", "recommendation", 0.5), ("ought not to", "discouragement", -0.5),
)
MODALITIES = ("prohibition", "obligation", "recommendation", "discouragement",
              "permission", "neutral")


@dataclass
class WordingClass:
    """Modality and signed strength ``p in [-1, 1]`` of a piece of policy wording."""
    modality: str
    p: float
    cue: Optional[str]
    position: int

    @property
    def is_hard(self) -> bool:
        """Obligations and prohibitions are hard constraints."""
        return self.modality in ("prohibition", "obligation")


_NEGATORS = ("not", "never", "no")
# Every word that negates (the rules below resolve only "not" / "never" / "no"), including the
# words that carry a negation of their own: with a negator they make a double negative the cue
# rules do not model ("shall not be prohibited", "It shall be unlawful ...").
_NEG_WORDS = _NEGATORS + tuple("nor neither none nothing nobody without prohibit prohibits "
                               "prohibited prevent prevents prevented forbid forbids forbidden "
                               "ban bans banned discourage discourages resist resists unlawful "
                               "illegal fail fails refuse refuses".split())
# A second deontic word governed by the cue ("shall not be REQUIRED", "shall be PERMITTED")
# changes the modality in ways the cue table does not model: refused, except "... required"
# after a positive obligation cue (which agrees with it).
_DEONTIC = ("required", "permitted", "allowed", "obliged", "entitled", "authorised",
            "authorized")
_CONTRACTED = dict([(v + "nt", v + " not") for v in "must should need would could might ought may "
                    "dare does did is are was were has have had".split()] + [
                    ("shant", "shall not"), ("cant", "can not"), ("cannot", "can not"),
                    ("wont", "will not"), ("dont", "do not"), ("aint", "is not")])
# A negation the cue lacks flips it deontically: just before the cue it negates the modality
# ("not prohibited" = permitted, "will not support" = oppose), a sentence- or clause-opening
# "No" negates the subject ("No vehicle shall" = prohibited), and "never" after a modal negates
# what it governs ("must never" = must not).  Any OTHER negation in the cue's comma clause is
# a configuration these rules do not model, so the wording is refused (neutral, p = 0).
_NEGATED_PRE = {"prohibition": ("permission", 0.0), "obligation": ("prohibition", -1.0),
                "permission": ("prohibition", -1.0)}
_NEGATED_POST = {"obligation": ("prohibition", -1.0), "permission": ("prohibition", -1.0)}
_FLIP = {"recommendation": "discouragement", "discouragement": "recommendation"}
# inflected soft verbs read as their cue ("will be supported", "encourages", "prioritizing",
# "seeks / sought to")
_INFLECT = dict(((v[:-1] if s == "ing" and v[-1] == "e" else v) + s, w) for w in (
    "encourage", "discourage", "promote", "support", "resist", "prioritise", "seek")
    for v in (w, w.replace("ise", "ize")) for s in ("", "s", "d" if w[-1] == "e" else "ed", "ing"))
_INFLECT["sought"] = "seek"
_NO_IDIOMS = ("doubt", "matter", "exceeding")       # "no doubt", "not exceeding": no negation
_COMPARATIVES = ("later", "earlier", "sooner", "less", "more", "fewer", "longer", "greater")
_MONTH_BEFORE = ("in", "of", "from", "since", "until", "till", "by", "during", "before", "after",
                 "early", "late", "mid")


def classify_wording(text: str) -> WordingClass:
    """Classify wording by its longest modal cue (earliest on a tie); see ``_NEGATED_PRE``.
    Wording whose negation the rules cannot resolve is ``neutral`` with p = 0 but keeps its
    cue, so ``compile_policy`` can still see a soft (should / encourage) cue on a hard clause."""
    t = re.sub(r"(?i)\b(at no time|(?:in|under) no circumstances?|on no account|in no case)\b",
               "never", _as_str(text, "text").replace("\0", " "))
    t = re.sub(r"\b((?:[A-Z]\.){2,}|(?i:st|rd|ave|dept|etc|approx|mrs?|ms|e\.g|i\.e)\.)(?=\s)|\.(?="
               r"\s+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)(?:s|rs?|day|sday|nesday|rsday|urday)?s?\b)",
               lambda mo: (mo.group(1) or "").replace(".", ""), t)   # H.G.V., St., "p.m. Monday"
    # after an initial, a.m. / p.m., a capitalised word of <= 4 letters (Co., Sq.), a listed
    # abbreviation (opp.) or "...", a full stop before a capital may or may not end the sentence
    # ("Zone A. No" / "J. Smith"): read both ways, refused (neutral) if the readings differ
    t = re.sub(r"(\b(?:[A-Za-z]\.)+|\b(?:[A-Z][A-Za-z]{1,3}|(?i:%s))\.|\.\.\.)(?=\s+[^a-z\s\d])"
               % "|".join(ABBREVIATIONS), lambda mo: mo.group(1).replace(".", "") + "\0", t)
    a = _wording(t.replace("\0", ""))
    b = _wording(t.replace("\0", ".")) if "\0" in t else a
    return a if (a.modality, a.p) == (b.modality, b.p) else WordingClass("neutral", 0.0, a.cue,
                                                                          a.position)


def _wording(t: str) -> WordingClass:
    """``classify_wording`` on prepared text: no full stop before a digit or clock colon ends
    a sentence ("No. 5", "Art. 5", "08:00")."""
    words: List[str] = []
    where: List[Tuple[int, int, int]] = []          # (sentence, comma clause, index in clause)
    for si, s in enumerate(re.split(r";|:(?!\d)|(?<!\d):|\.(?=\s*$|\s+[^a-z\s\d])", t)):
        for ci, c in enumerate(s.split(",")):
            ws = " ".join(_CONTRACTED.get(w, _INFLECT.get(w, w)) for w in normalise_text(
                c, expand_abbreviations=False, compass=False).split()).split()
            words += ws
            where += [(si, ci, k) for k in range(len(ws))]
    nxt = words[1:] + [""]
    best: Optional[Tuple[int, int, int, str, str, float]] = None
    found: List[Tuple[int, str]] = []
    for cue, modality, p in WORDING_CUES:   # never across a comma ("shall, not later than")
        cw = cue.split()
        for i in range(len(words) - len(cw) + 1):
            if words[i:i + len(cw)] != cw or where[i][:2] != where[i + len(cw) - 1][:2] or (
                    cue == "may" and ((i and words[i - 1] in _MONTH_BEFORE)   # May, the month
                                      or nxt[i][:1].isdigit())):
                continue
            found.append((i, cue))
            key = (-len(cue), sum(len(w) + 1 for w in words[:i]))
            if best is None or key < best[:2]:
                best = key + (i, cue, modality, p)
    if best is None:
        return WordingClass("neutral", 0.0, None, -1)
    _nl, pos, i, cue, modality, p = best
    n, (si, ci, k) = len(cue.split()), where[best[2]]

    def negates(j: int) -> bool:          # comparatives are idioms only before "than"
        return words[j] in _NEG_WORDS and not (words[j] in ("no", "not") and (
            nxt[j] in _NO_IDIOMS or nxt[j] in _COMPARATIVES and words[j + 2:j + 3] == ["than"]
            or nxt[j][:1].isdigit() and words[j] == "no"))

    # a cue opening a later comma clause continues an earlier one ("will not, however, support",
    # "shall not, in any case, be prohibited"): the earlier clauses' negations reach it too
    cont = ci > 0 and (k == 0 or k == 1 and words[i - 1] in ("be", "is", "are"))
    # a modal ENDING its comma clause continues into the later ones ("shall, in any event,
    # never commence"): their negations and deontic words reach it too
    fwd = cue.split()[-1] in ("must", "shall", "may", "should", "not", "to") and (
        i + n == len(words) or where[i + n][:2] != (si, ci))
    scope = set(j for j in range(len(words)) if where[j][0] == si and (where[j][1] == ci or (
        cont and where[j][1] < ci) or (fwd and where[j][1] > ci)))
    gov = sorted([(g, q) for g, q in found if q in ("should", "may", "ought to")] + [
        (g, words[g]) for g in scope if words[g] in ("might", "could", "would", "can")])
    gov = [x for x in gov if x[0] < i and x[0] in scope]
    if gov and cue == "prohibited":     # "should be prohibited" is no prohibition; a cue with its
        return WordingClass("neutral", 0.0, gov[-1][1],        # own verb ("is prohibited") is
                            sum(len(w) + 1 for w in words[:gov[-1][0]]))   # not governed by it
    handled, table = -1, None
    if "not" not in cue.split():
        subj = [j for j in range(i) if words[j] == "no" and where[j][0] == si and (where[j][2] == 0
                or words[j - 1] in ("am", "pm")) and where[j][1] in (0, ci) and negates(j) and nxt[j]]
        if i and words[i - 1] in _NEGATORS and where[i - 1][:2] == (si, ci):
            handled, table = i - 1, _NEGATED_PRE
        elif subj and where[subj[-1]][1] != ci and k:   # "No objection being made, the Council
            return WordingClass("neutral", 0.0, cue, pos)  # shall": is "No ..." its subject?
        elif subj:
            handled, table = subj[-1], _NEGATED_PRE
        elif nxt[i + n - 1] == "never":
            handled, table = i + n, _NEGATED_POST
    if any(j in scope and not i <= j < i + n and ((negates(j) and j != handled) or (
            where[j][1] >= ci and words[j] in _DEONTIC and not (
                words[j] == "required" and modality == "obligation" and table is None)))
           for j in range(len(words))):
        return WordingClass("neutral", 0.0, cue, pos)
    if table is not None and modality in table:     # "No permit is required": not obligatory
        modality, p = ("permission", 0.0) if cue.endswith("required") else table[modality]
    elif table is not None and modality in _FLIP:
        modality, p = _FLIP[modality], -p
    return WordingClass(modality, p, cue, pos)


def directional_priors(clauses: Sequence[PolicyClause]) -> Dict[str, float]:
    """Mean wording strength per objective over the clauses that name one."""
    acc: Dict[str, List[float]] = {}
    for c in sorted(_clause_list(clauses), key=lambda c: c.clause_id):
        if c.objective is None:
            continue
        acc.setdefault(c.objective, []).append(classify_wording(c.text_for_wording).p)
    return dict((obj, math.fsum(v) / len(v)) for obj, v in sorted(acc.items()))


def weight_prior(w_planner: Dict[str, float], p: Dict[str, float],
                 eta: float = 0.5) -> Dict[str, float]:
    """``w = w_planner * (1 + eta * p)`` elementwise; objectives without a p keep their weight."""
    if not isinstance(w_planner, dict) or not isinstance(p, dict):
        raise PolicyCompilerError("w_planner and p must be dicts keyed by objective.")
    ev = _as_nonneg(eta, "eta")
    out: Dict[str, float] = {}
    for k in sorted(p, key=str):
        if k not in w_planner:
            raise PolicyCompilerError("p names objective %r with no planner weight." % k)
    for k in sorted(w_planner, key=str):
        w = _as_nonneg(w_planner[k], "w_planner[%s]" % k)
        pk = _as_float(p.get(k, 0.0), "p[%s]" % k)
        if not (-1.0 <= pk <= 1.0):
            raise PolicyCompilerError("p[%s] must lie in [-1, 1] (got %g)." % (k, pk))
        factor = 1.0 + ev * pk
        if factor < 0.0:
            raise PolicyCompilerError("eta * p[%s] < -1 would make a weight negative." % k)
        out[k] = w * factor
    return out


# ===========================================================================
# 13. COMPILATION TO FIXINGS AND BOUNDS
# ===========================================================================
FIXING_PART_SCHEMA: Dict[str, Any] = _object({
    "fixedZero": _ids(uniqueItems=True), "fixedOne": _ids(uniqueItems=True),
    "bounds": {"type": "object", "additionalProperties": _PAIR},
    "crossingFixings": {"type": "object", "additionalProperties": {"enum": [0, 1]}},
    "junctionDomains": {"type": "object", "additionalProperties": _array(
        {"enum": list(JUNCTION_TYPES)}, uniqueItems=True)}},
    "fixedZero", "fixedOne", "bounds", "crossingFixings", "junctionDomains")
FIXINGS_SCHEMA: Dict[str, Any] = _object(dict(list(FIXING_PART_SCHEMA["properties"].items()) + [
    ("conditional", _array(_object({"predicate": PREDICATE_SCHEMA, "fixing": FIXING_PART_SCHEMA},
                                   "predicate", "fixing"))),
    ("provenance", {"type": "object", "additionalProperties": _ids()})]),
    *(FIXING_PART_SCHEMA["required"] + ["conditional", "provenance"]))


@dataclass
class ConditionalFixing:
    """A fixing that applies only in scenarios where ``predicate`` holds."""
    predicate: Predicate
    fixing: "CompiledFixings"


@dataclass
class CompiledFixings:
    """Fixings and bounds for the search variables ``x = (z, y, w, k)``. ``fixed_zero``: edges
    with ``z_e := 0`` (prohibited); ``fixed_one``: edges with ``z_e := 1`` (protected);
    ``bounds``: ``y_v in [lo, hi]``; ``crossing_fixings``: ``w_v := 0 | 1``;
    ``junction_domains``: the allowed subset of the junction-type set K for ``k_v``;
    ``conditional``: fixings gated on a scenario predicate; ``provenance``: fixing key -> clause
    ids."""
    fixed_zero: List[str] = field(default_factory=list)
    fixed_one: List[str] = field(default_factory=list)
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    crossing_fixings: Dict[str, int] = field(default_factory=dict)
    junction_domains: Dict[str, List[str]] = field(default_factory=dict)
    conditional: List[ConditionalFixing] = field(default_factory=list)
    provenance: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.fixed_zero = sorted(set(_as_id_list(self.fixed_zero, "fixed_zero")))
        self.fixed_one = sorted(set(_as_id_list(self.fixed_one, "fixed_one")))
        both = set(self.fixed_zero) & set(self.fixed_one)
        if both:
            raise PolicyCompilerError("edge %r is fixed to both 0 and 1." % sorted(both)[0])
        if not isinstance(self.bounds, dict) or not isinstance(self.crossing_fixings, dict) \
                or not isinstance(self.junction_domains, dict) \
                or not isinstance(self.provenance, dict):
            raise PolicyCompilerError("bounds, crossing_fixings, junction_domains and "
                                      "provenance must be dicts.")
        b: Dict[str, Tuple[float, float]] = {}
        for k in sorted(self.bounds, key=str):
            v = self.bounds[k]
            if isinstance(v, (str, bytes)) or not hasattr(v, "__len__") or len(v) != 2:
                raise PolicyCompilerError("bounds[%r] must be a (lo, hi) pair." % (k,))
            lo, hi = _as_float(v[0], "bounds lo"), _as_float(v[1], "bounds hi")
            if lo > hi:
                raise PolicyCompilerError("bounds[%s] has lo > hi." % k)
            b[_as_id(k, "bound key")] = (lo, hi)
        self.bounds = b
        cf: Dict[str, int] = {}
        for k in sorted(self.crossing_fixings, key=str):
            v = self.crossing_fixings[k]
            if isinstance(v, bool) or v not in (0, 1):
                raise PolicyCompilerError("crossing_fixings[%r] must be 0 or 1." % (k,))
            cf[_as_id(k, "crossing node")] = int(v)
        self.crossing_fixings = cf
        jd: Dict[str, List[str]] = {}
        for k in sorted(self.junction_domains, key=str):
            vals = _as_seq(self.junction_domains[k], "junction_domains[%r]" % (k,))
            for t in vals:
                _as_choice(t, JUNCTION_TYPES, "junction type")
            jd[_as_id(k, "junction node")] = sorted(set(vals))
        self.junction_domains = jd
        self.conditional = _as_seq(self.conditional, "conditional")
        for c in self.conditional:
            if not isinstance(c, ConditionalFixing):
                raise PolicyCompilerError("conditional entries must be ConditionalFixing.")
            if c.fixing.conditional:
                raise PolicyCompilerError("conditional fixings cannot nest.")
        pv: Dict[str, List[str]] = {}
        for k in sorted(self.provenance, key=str):
            pv[_as_id(k, "provenance key")] = sorted(set(_as_id_list(self.provenance[k],
                                                                     "provenance")))
        self.provenance = pv

    @property
    def is_empty(self) -> bool:
        """No fixing of any kind."""
        return not (self.fixed_zero or self.fixed_one or self.bounds or self.crossing_fixings
                    or self.junction_domains or self.conditional)

    def n_fixings(self) -> int:
        """Number of unconditional fixings plus conditional entries."""
        return (len(self.fixed_zero) + len(self.fixed_one) + len(self.bounds)
                + len(self.crossing_fixings) + len(self.junction_domains)
                + len(self.conditional))

    def part_dict(self) -> Dict[str, Any]:
        """The five unconditional blocks only."""
        return {"fixedZero": list(self.fixed_zero), "fixedOne": list(self.fixed_one),
                "bounds": dict((k, [lo, hi]) for k, (lo, hi) in sorted(self.bounds.items())),
                "crossingFixings": dict(sorted(self.crossing_fixings.items())),
                "junctionDomains": dict((k, list(v)) for k, v in
                                        sorted(self.junction_domains.items()))}

    def to_dict(self) -> Dict[str, Any]:
        """The cross-module contract consumed by urban_optimisation (plain JSON)."""
        d = self.part_dict()
        d["conditional"] = [{"predicate": c.predicate.to_dict(), "fixing": c.fixing.part_dict()}
                            for c in self.conditional]
        d["provenance"] = dict((k, list(v)) for k, v in sorted(self.provenance.items()))
        return d

    @classmethod
    def from_dict(cls, d: Any) -> "CompiledFixings":
        """Validate against ``FIXINGS_SCHEMA`` and rebuild (round-trips ``to_dict``)."""
        errs = validate_json(d, FIXINGS_SCHEMA)
        if errs:
            raise PolicyCompilerError("invalid fixings dict: %s" % "; ".join(errs))

        def part(p: Dict[str, Any]) -> "CompiledFixings":
            return cls(fixed_zero=list(p["fixedZero"]), fixed_one=list(p["fixedOne"]),
                       bounds=dict((k, (v[0], v[1])) for k, v in p["bounds"].items()),
                       crossing_fixings=dict(p["crossingFixings"]),
                       junction_domains=dict((k, list(v)) for k, v in
                                             p["junctionDomains"].items()))
        base = part(d)
        base.conditional = [ConditionalFixing(Predicate.from_dict(c["predicate"]),
                                              part(c["fixing"])) for c in d["conditional"]]
        base.provenance = dict((k, sorted(set(v))) for k, v in sorted(d["provenance"].items()))
        return base

    def atoms(self) -> List[Tuple[str, str, Any]]:
        """Unconditional fixings as ``(kind, key, value)`` with kind in z / y / w / k."""
        out: List[Tuple[str, str, Any]] = []
        out += [("z", e, 0) for e in self.fixed_zero]
        out += [("z", e, 1) for e in self.fixed_one]
        out += [("y", v, b) for v, b in sorted(self.bounds.items())]
        out += [("w", n, x) for n, x in sorted(self.crossing_fixings.items())]
        out += [("k", n, tuple(d)) for n, d in sorted(self.junction_domains.items())]
        return out


def _fixing_key(kind: str, key: str, value: Any = None) -> str:
    """Provenance key of an atom: fixedZero:e / fixedOne:e / bounds:v / ..."""
    if kind == "z":
        return ("fixedZero:%s" if value == 0 else "fixedOne:%s") % key
    return {"y": "bounds:%s", "w": "crossingFixings:%s", "k": "junctionDomains:%s"}[kind] % key


def _from_atoms(atoms: Sequence[Tuple[str, str, Any]]) -> CompiledFixings:
    """Rebuild an unconditional fixing from consistent atoms."""
    fz, fo, b, cf, jd = [], [], {}, {}, {}
    for kind, key, val in atoms:
        if kind == "z":
            (fz if val == 0 else fo).append(key)
        else:
            {"y": b, "w": cf, "k": jd}[kind][key] = list(val) if kind == "k" else val
    return CompiledFixings(fz, fo, b, cf, jd)


def _clause_list(clauses: Any) -> List[PolicyClause]:
    """Validate a clause collection (list or dict) and return a list."""
    if isinstance(clauses, dict):
        clauses = [clauses[k] for k in sorted(clauses, key=str)]
    out = _as_seq(clauses, "clauses")
    seen = set()
    for c in out:
        if not isinstance(c, PolicyClause):
            raise PolicyCompilerError("clauses must be PolicyClause objects.")
        if c.clause_id in seen:
            raise PolicyCompilerError("duplicate clause id %r." % c.clause_id)
        seen.add(c.clause_id)
    return out


@dataclass
class TargetElements:
    """The network elements a binding resolves to."""
    edges: List[str]
    candidate_edges: List[str]
    nodes: List[str]


def target_elements(binding: Binding, gazetteer: Gazetteer) -> TargetElements:
    """Existing edges, candidate edges and nodes behind a bound target."""
    if not isinstance(binding, Binding) or binding.status != "bound" or binding.target_id is None:
        raise PolicyCompilerError("target_elements needs a bound Binding.")
    if binding.target_type == "entry":
        e = gazetteer.get(binding.target_id)
        return TargetElements(list(e.edge_ids), list(e.candidate_edge_ids), list(e.node_ids))
    cand_all = set(x for e in gazetteer.entries() for x in e.candidate_edge_ids)
    exist_all = set(x for e in gazetteer.entries() for x in e.edge_ids)
    if binding.target_type == "edge":
        eid = binding.target_id
        if eid in cand_all:
            return TargetElements([], [eid], [])
        return TargetElements([eid], [], [])
    node = binding.target_id
    ends = gazetteer.edge_endpoints
    inc = sorted(e for e, (u, v) in ends.items() if node in (u, v))
    return TargetElements([e for e in inc if e in exist_all], [e for e in inc if e in cand_all],
                          [node])


def _check_baselines(baseline_y: Any, baseline_junctions: Any
                     ) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Validate the baseline tables (dicts of finite values / junction types)."""
    for nm, tbl in (("baseline_y", baseline_y), ("baseline_junctions", baseline_junctions)):
        if tbl is not None and not isinstance(tbl, dict):
            raise PolicyCompilerError("%s must be a dict." % nm)
    by = dict((_as_id(k, "baseline_y key"), _as_float(v, "baseline_y[%s]" % k))
              for k, v in sorted((baseline_y or {}).items(), key=lambda kv: str(kv[0])))
    bj = dict((_as_id(k, "baseline_junctions key"),
               _as_choice(v, JUNCTION_TYPES, "baseline junction"))
              for k, v in sorted((baseline_junctions or {}).items(), key=lambda kv: str(kv[0])))
    return by, bj


def compile_clause(clause: PolicyClause, binding: Binding, gazetteer: Gazetteer, *,
                   baseline_y: Optional[Dict[str, float]] = None,
                   baseline_junctions: Optional[Dict[str, str]] = None
                   ) -> Tuple[CompiledFixings, List[str]]:
    """One clause's rulings to fixings on its bound target; also the issues found. ADD_EDGE acts
    on CANDIDATE edges, REMOVE_EDGE on EXISTING ones (PROHIBIT = protect, ``z := 1``);
    REPARTITION bounds ``quantity:edge`` (PROHIBIT pins the baseline; REQUIRE needs bounds);
    GREEN_CORRIDOR pins ``green:edge``; JUNCTION_REDESIGN restricts ``k_v`` (PROHIBIT removes
    ``junction_types`` or freezes the baseline; REQUIRE allows only them or forbids the
    baseline). PERMIT / ABSTAIN fix nothing, except a REPARTITION PERMIT with bounds."""
    if not isinstance(clause, PolicyClause):
        raise PolicyCompilerError("compile_clause needs a PolicyClause.")
    if not isinstance(gazetteer, Gazetteer):
        raise PolicyCompilerError("compile_clause needs a Gazetteer.")
    by, bj = _check_baselines(baseline_y, baseline_junctions)
    tgt = target_elements(binding, gazetteer)
    r = clause.rulings
    fz, fo, issues = [], [], []  # type: List[str], List[str], List[str]
    bounds, cross, doms = {}, {}, {}  # type: Dict[str, Tuple[float, float]], Dict[str, int], Dict

    def need(elems: Sequence[str]) -> bool:
        issues.extend([] if elems else ["no_target_elements"])
        return bool(elems)

    if r["ADD_EDGE"] in ("PROHIBIT", "REQUIRE") and need(tgt.candidate_edges):
        (fz if r["ADD_EDGE"] == "PROHIBIT" else fo).extend(tgt.candidate_edges)
    if r["REMOVE_EDGE"] in ("PROHIBIT", "REQUIRE") and need(tgt.edges):
        (fo if r["REMOVE_EDGE"] == "PROHIBIT" else fz).extend(tgt.edges)
    if r["REPARTITION"] == "REQUIRE" and not clause.bounds:
        # y != baseline is no interval: compiling nothing would pass every check
        issues.append("unrepresentable")
    if r["REPARTITION"] in ("PERMIT", "REQUIRE") and clause.bounds and need(tgt.edges):
        for q, (lo, hi) in sorted(clause.bounds.items()):
            for e in tgt.edges:
                bounds["%s:%s" % (q, e)] = (lo, hi)
    if r["REPARTITION"] == "PROHIBIT" and need(tgt.edges):
        if clause.bounds:
            want = ["%s:%s" % (q, e) for q in sorted(clause.bounds) for e in tgt.edges]
        else:
            # "quantity:edge", colon-free quantity: the edge id may itself hold ':'
            want = sorted(v for v in by if ":" in v and v.split(":", 1)[1] in tgt.edges)
        if not want:
            issues.append("missing_baseline")
        for v in want:
            if v in by:
                bounds[v] = (by[v], by[v])
            else:
                issues.append("missing_baseline")
    if r["GREEN_CORRIDOR"] in ("PROHIBIT", "REQUIRE") and need(tgt.edges):
        val = 0.0 if r["GREEN_CORRIDOR"] == "PROHIBIT" else 1.0
        for e in tgt.edges:
            bounds["green:%s" % e] = (val, val)
    if r["ADD_CROSSING"] in ("PROHIBIT", "REQUIRE") and need(tgt.nodes):
        for n in tgt.nodes:
            cross[n] = 0 if r["ADD_CROSSING"] == "PROHIBIT" else 1
    jr = r["JUNCTION_REDESIGN"]
    jt = clause.junction_types
    if jr in ("PROHIBIT", "REQUIRE") and need(tgt.nodes):
        for n in tgt.nodes:
            if jr == "PROHIBIT" and jt is not None:
                doms[n] = [t for t in JUNCTION_TYPES if t not in jt]
            elif jr == "REQUIRE" and jt is not None:
                doms[n] = list(jt)
            elif n not in bj:
                issues.append("missing_baseline")
            elif jr == "PROHIBIT":
                doms[n] = [bj[n]]
            else:
                doms[n] = [t for t in JUNCTION_TYPES if t != bj[n]]
            if n in doms and not doms[n]:
                # prohibiting every type in K empties k_v's domain: the clause contradicts
                # itself and would silently empty the design space
                del doms[n]
                issues.append("conflict")
    if clause.predicate is not None and not clause.predicate.is_satisfiable():
        # every class it names is exempt: a dead fixing would silently drop the clause
        return CompiledFixings(), sorted(set(issues + ["conflict"]))
    base = CompiledFixings(fz, fo, bounds, cross, doms)
    prov = dict((_fixing_key(k, key, v), [clause.clause_id]) for k, key, v in base.atoms())
    if clause.predicate is not None and not clause.predicate.is_trivial and not base.is_empty:
        out = CompiledFixings(conditional=[ConditionalFixing(clause.predicate, base)],
                              provenance={"conditional:0": [clause.clause_id]})
    else:
        base.provenance = prov
        out = base
    return out, sorted(set(issues))


@dataclass
class CompilationResult:
    """Everything the compiler decided, clause by clause, and the merged fixings."""
    fixings: CompiledFixings
    per_clause: Dict[str, CompiledFixings]
    clauses: Dict[str, PolicyClause]
    spans: Dict[str, SpanVerification]
    bindings: Dict[str, Binding]
    wording: Dict[str, WordingClass]
    queue: AbstentionQueue
    conflicts: List["Conflict"]
    skipped: Dict[str, str]
    priors: Dict[str, float]
    in_force: Optional[InForceState]

    @property
    def n_clauses(self) -> int:
        """Clauses submitted."""
        return len(self.clauses)

    @property
    def compiled_clause_ids(self) -> List[str]:
        """Clauses that contributed at least one surviving fixing."""
        ids = set()
        for v in self.fixings.provenance.values():
            ids.update(v)
        return sorted(ids)

    def abstention_rate(self) -> float:
        """Share of submitted clauses sitting in the abstention queue."""
        return self.queue.abstention_rate(max(1, self.n_clauses))


def compile_policy(clauses: Any, gazetteer: Gazetteer, sources: Dict[str, str], *,
                   at: Any = None, instruments: Optional[Sequence[Instrument]] = None,
                   baseline_y: Optional[Dict[str, float]] = None,
                   baseline_junctions: Optional[Dict[str, str]] = None,
                   choices: Optional[Dict[str, str]] = None, threshold: float = 0.9,
                   margin: float = 0.04, fuzzy_threshold: float = 0.9) -> CompilationResult:
    """Verify, bind, compile and merge a set of clauses into ``CompiledFixings``. Per clause in
    id order: in-force filter -> span -> wording -> binding (reviewer ``choices`` first) ->
    ``compile_clause``.  Anything unverifiable, unbound or ambiguous is queued; conflicting
    clauses are withheld and queued."""
    cl = _clause_list(clauses)
    if not isinstance(gazetteer, Gazetteer):
        raise PolicyCompilerError("compile_policy needs a Gazetteer.")
    if not isinstance(sources, dict):
        raise PolicyCompilerError("sources must map document id -> text.")
    for k in sorted(sources, key=str):
        _as_str(sources[k], "sources[%r]" % (k,))
    # validate EVERY argument up front, even if no clause reaches its stage
    _as_fraction(threshold, "threshold")
    _as_fraction(margin, "margin")
    _as_fraction(fuzzy_threshold, "fuzzy_threshold")
    _check_baselines(baseline_y, baseline_junctions)
    if at is not None:
        date_ordinal(at)
    if choices is not None and not isinstance(choices, dict):
        raise PolicyCompilerError("choices must map clause id -> chosen id.")
    ch = dict(choices or {})
    for k in sorted(ch, key=str):
        _as_id(k, "choices key")
        _as_id(ch[k], "choices[%s]" % k)
    state: Optional[InForceState] = None
    listed = set()
    if instruments is not None:
        if at is None:
            raise PolicyCompilerError("instruments need an 'at' date.")
        instruments = _as_seq(instruments, "instruments")    # iterated twice below
        # a clause's instrument_id lists it there: a clause of a revoked order must never
        # compile as in force just because no clause_ids name it
        imap, own = _instrument_map(instruments), {}  # type: Dict[str, Instrument], Dict
        for c in (c for c in sorted(cl, key=lambda c: c.clause_id) if c.instrument_id):
            by = [i for i in sorted(imap) if c.clause_id in imap[i].clause_ids]
            if c.instrument_id not in imap or (by and c.instrument_id not in by):
                raise PolicyCompilerError("clause %s names instrument %r, which is unknown or "
                                          "lists it elsewhere." % (c.clause_id, c.instrument_id))
            own.setdefault(c.instrument_id, []).extend([] if by else [c.clause_id])
        instruments = [replace(imap[i], clause_ids=imap[i].clause_ids + tuple(own.get(i, ())))
                       for i in sorted(imap)]
        state = resolve_in_force(instruments, at)
        for ins in instruments:
            listed.update(ins.clause_ids)
    queue = AbstentionQueue()
    per_clause: Dict[str, CompiledFixings] = {}
    spans: Dict[str, SpanVerification] = {}
    bindings: Dict[str, Binding] = {}
    wording: Dict[str, WordingClass] = {}
    skipped: Dict[str, str] = {}
    cmap: Dict[str, PolicyClause] = {}

    def abstain(c: PolicyClause, reason: str, amb: float, cands: Sequence[str], detail: str) -> None:
        queue.add(AbstentionItem(c.clause_id, reason, amb, tuple(cands), c.quote, c.page,
                                 c.document_id, detail))

    for c in sorted(cl, key=lambda c: c.clause_id):
        cid = c.clause_id
        cmap[cid] = c
        wording[cid] = classify_wording(c.text_for_wording)
        if state is not None and cid in listed and cid not in state.clauses:
            skipped[cid] = "not_in_force"
            continue
        src = sources.get(c.document_id)
        if src is None:
            abstain(c, "span_unverified", 1.0, (), "source document %r missing" % c.document_id)
            continue
        sv = verify_span(src, c.quote, c.start, c.end, fuzzy_threshold=fuzzy_threshold)
        spans[cid] = sv
        if sv.status == "fuzzy":
            abstain(c, "span_fuzzy", 1.0 - sv.score, (), "fuzzy span score %.4f" % sv.score)
            continue
        if not sv.verified:
            abstain(c, "span_unverified", 1.0, (), "quote not found in source")
            continue
        if c.abstains_everywhere:
            skipped[cid] = "no_operation"
            continue
        if c.strength == "soft":
            skipped[cid] = "soft_clause"
            continue
        soft = ("recommendation", "discouragement")       # an unresolved soft cue counts too
        if wording[cid].modality in soft or (wording[cid].modality == "neutral" and dict(
                (q, m) for q, m, _p in WORDING_CUES).get(wording[cid].cue) in soft):
            abstain(c, "wording_ambiguous", 0.5, (), "hard clause with %s wording %r"
                    % (wording[cid].modality, wording[cid].cue))
            continue
        b = _bind_clause(c, gazetteer, ch.get(cid), threshold, margin)
        bindings[cid] = b
        if b.status != "bound":
            abstain(c, "binding_ambiguous" if b.status == "ambiguous" else "binding_unresolved",
                    b.ambiguity, b.candidates, b.reason)
            continue
        fx, issues = compile_clause(c, b, gazetteer, baseline_y=baseline_y,
                                    baseline_junctions=baseline_junctions)
        for iss in issues:
            abstain(c, iss, 1.0 if iss in ("no_target_elements", "conflict") else 0.5, (), iss)
        per_clause[cid] = fx
    merged, conflicts = merge_fixings(per_clause)
    for cf in conflicts:
        for cid in cf.clause_ids:
            c = cmap[cid]
            abstain(c, "conflict", 1.0, (), "%s on %s" % (cf.kind, cf.key))
    active = [c for c in cl if skipped.get(c.clause_id) != "not_in_force"]
    return CompilationResult(merged, per_clause, cmap, spans, bindings, wording, queue,
                             conflicts, skipped, directional_priors(active), state)


def _bind_clause(c: PolicyClause, gazetteer: Gazetteer, choice: Optional[str],
                 threshold: float, margin: float) -> Binding:
    """Bind one clause target, honouring a reviewer's choice when given."""
    hint = c.hint
    if choice is not None:
        ch = _as_id(choice, "choice")
        if c.target_kind in ("junction", "node"):
            known = set(gazetteer.node_coords) | set(n for e in gazetteer.entries() for n in e.node_ids)
            if ch not in known:
                raise PolicyCompilerError("reviewer choice %r is not a node." % ch)
            return Binding("bound", "node", ch, [ch], [1.0], 0.0, "reviewer", "reviewer choice")
        gazetteer.get(ch)
        return Binding("bound", "entry", ch, [ch], [1.0], 0.0, "reviewer", "reviewer choice")
    if c.target_kind in ("street", "area"):
        kind = c.target_kind
        b = gazetteer.resolve(c.target_name, kind=kind, hint=hint, threshold=threshold,
                              margin=margin)
        if b.status == "unresolved" and kind == "area":
            b2 = gazetteer.resolve(c.target_name, kind="landmark", hint=hint)
            if b2.status != "unresolved":
                return b2
        return b
    if c.target_kind == "junction":
        return gazetteer.resolve_junction(c.target_name, c.target_name_b or "", hint=hint,
                                          threshold=threshold, margin=margin)
    if c.target_kind == "edge":
        if c.target_name in gazetteer.all_edge_ids:
            return Binding("bound", "edge", c.target_name, [c.target_name], [1.0], 0.0,
                           "identifier", "explicit edge id")
        return Binding("unresolved", "edge", None, [], [], 1.0, "identifier",
                       "no edge %r in the gazetteer" % c.target_name)
    known_nodes = set(gazetteer.node_coords) | set(n for e in gazetteer.entries() for n in e.node_ids)
    if c.target_name in known_nodes:
        return Binding("bound", "node", c.target_name, [c.target_name], [1.0], 0.0,
                       "identifier", "explicit node id")
    return Binding("unresolved", "node", None, [], [], 1.0, "identifier",
                   "no node %r in the gazetteer" % c.target_name)


# ===========================================================================
# 14. CONFLICT DETECTION
# ===========================================================================
CONFLICT_KINDS = ("zero_vs_one", "crossing_clash", "empty_bounds", "empty_junction_domain")


@dataclass
class Conflict:
    """Two or more clauses that cannot all hold in some common scenario."""
    kind: str
    key: str
    clause_ids: Tuple[str, ...]
    detail: str

    def as_dict(self) -> Dict[str, Any]:
        """JSON-able camelCase dict."""
        return {"kind": self.kind, "key": self.key, "clauseIds": list(self.clause_ids),
                "detail": self.detail}


def _atoms_with_context(per_clause: Dict[str, CompiledFixings]
                        ) -> List[Tuple[str, str, Any, str, Optional[Predicate], int]]:
    """``(kind, key, value, clause_id, predicate, conditional_index)`` for every atom."""
    out = []
    for cid in sorted(per_clause, key=str):
        fx = per_clause[_as_id(cid, "per_clause key")]
        if not isinstance(fx, CompiledFixings):
            raise PolicyCompilerError("per_clause values must be CompiledFixings.")
        for kind, key, val in fx.atoms():
            out.append((kind, key, val, cid, None, -1))
        for i, cf in enumerate(fx.conditional):
            for kind, key, val in cf.fixing.atoms():
                out.append((kind, key, val, cid, cf.predicate, i))
    return out


def _predicates_meet(p: Optional[Predicate], q: Optional[Predicate]) -> bool:
    """None means 'always'; two predicates meet if some scenario satisfies both. 'Always' meets
    a predicate only if that predicate can hold at all -- a never-satisfiable predicate (every
    listed class exempt) meets nothing."""
    if p is None and q is None:
        return True
    if p is None or q is None:
        return (q if p is None else p).is_satisfiable()
    return p.intersects(q)


def _atom_clash(kind: str, a: Any, b: Any) -> bool:
    """Do two atoms on the same key contradict each other?"""
    if kind in ("z", "w"):
        return a != b
    if kind == "y":
        return max(a[0], b[0]) > min(a[1], b[1])
    return not (set(a) & set(b))


def _joint_domain_clash(g: Sequence[Tuple[Any, str, Optional[Predicate], int]]) -> List[str]:
    """Clauses in a MINIMAL jointly-empty family of junction domains, in some scenario. Exact:
    one scenario per (day, class, window cell) decides every predicate (tags can always
    co-occur).  A clause is involved iff it is in a MINIMAL family sharing no type -- at most
    |K| = 5 members, each the only one missing some type -- so a clause compatible with every
    culprit is not."""
    cuts, tags = set([0]), set()
    for p in (x[2] for x in g if x[2] is not None):
        cuts.update(c for w in p.time_windows for seg in w.segments() for c in seg)
        tags.update(p.scenario_tags)
    out, seen = set(), set()
    for day, vc, minute in itertools.product(DAYS, VEHICLE_CLASSES,
                                             sorted(c for c in cuts if c < _MINUTES_PER_DAY)):
        sc = Scenario(minute, day, vc, tuple(sorted(tags)))
        dom: Dict[str, frozenset] = {}
        for x in g:
            if x[2] is None or x[2].holds(sc):
                dom[x[1]] = dom.get(x[1], frozenset(JUNCTION_TYPES)) & frozenset(x[0])
        state = tuple(sorted((c, tuple(sorted(d))) for c, d in dom.items()))
        if state in seen:
            continue
        seen.add(state)
        out.update(c for c in dom if not dom[c])
        live = sorted(c for c in dom if dom[c])
        for size in range(2, min(len(JUNCTION_TYPES), len(live)) + 1):
            for fam in itertools.combinations(live, size):
                if not frozenset.intersection(*[dom[c] for c in fam]) and all(
                        frozenset.intersection(*[dom[c] for c in fam if c != d]) for d in fam):
                    out.update(fam)
    return sorted(out)


def detect_conflicts(per_clause: Dict[str, CompiledFixings]) -> List[Conflict]:
    """Every contradiction between clause fixings that can co-occur in a scenario. Pairs clash
    when their predicates intersect (exactly); junction domains are also checked JOINTLY per
    scenario cell.  The clauses named are those in some MINIMAL contradictory family, so a
    clause compatible with every culprit is never blamed -- nor, in ``merge_fixings``,
    withheld."""
    if not isinstance(per_clause, dict):
        raise PolicyCompilerError("per_clause must map clause id -> CompiledFixings.")
    atoms = _atoms_with_context(per_clause)
    groups: Dict[Tuple[str, str], List[Tuple[Any, str, Optional[Predicate], int]]] = {}
    for kind, key, val, cid, pred, ci in atoms:
        groups.setdefault((kind, key), []).append((val, cid, pred, ci))
    out: List[Conflict] = []
    names = {"z": "zero_vs_one", "w": "crossing_clash", "y": "empty_bounds",
             "k": "empty_junction_domain"}
    for (kind, key) in sorted(groups):
        g = groups[(kind, key)]
        involved = set()
        for a, b in itertools.combinations(g, 2):
            if a[1] != b[1] and _predicates_meet(a[2], b[2]) and _atom_clash(kind, a[0], b[0]):
                involved.update((a[1], b[1]))
        # always run (a pairwise clash elsewhere must not hide a joint one)
        if kind == "k":
            involved.update(_joint_domain_clash(g))
        if involved:
            out.append(Conflict(names[kind], key, tuple(sorted(involved)),
                                "%d clauses disagree on %s %s" % (len(involved), kind, key)))
    return out


def merge_fixings(per_clause: Dict[str, CompiledFixings]
                  ) -> Tuple[CompiledFixings, List[Conflict]]:
    """Merge per-clause fixings; every CONFLICTING clause's atom is withheld. Agreeing fixings
    combine (bounds and domains intersect) with provenance naming every contributor.  The atoms
    of the clauses a conflict names are dropped everywhere (no winner is picked); an UNDISPUTED
    clause on the same key still binds -- withholding it would silently drop a valid
    constraint."""
    conflicts = detect_conflicts(per_clause)
    kind_of = {"zero_vs_one": "z", "crossing_clash": "w", "empty_bounds": "y",
               "empty_junction_domain": "k"}
    blocked_cl = set((kind_of[c.kind], c.key, cid) for c in conflicts for cid in c.clause_ids)
    merged: Dict[Tuple[str, str], Any] = {}
    prov: Dict[str, List[str]] = {}
    cond: List[Tuple[str, int, ConditionalFixing]] = []
    for kind, key, val, cid, pred, ci in _atoms_with_context(per_clause):
        if (kind, key, cid) in blocked_cl or pred is not None:
            continue
        if (kind, key) in merged:
            cur = merged[(kind, key)]
            if kind == "y":
                val = (max(cur[0], val[0]), min(cur[1], val[1]))
            elif kind == "k":
                val = tuple(sorted(set(cur) & set(val)))
        merged[(kind, key)] = val
    for (kind, key), val in sorted(merged.items()):
        prov[_fixing_key(kind, key, val)] = []
    for kind, key, val, cid, pred, ci in _atoms_with_context(per_clause):
        if pred is None and (kind, key) in merged and (kind, key, cid) not in blocked_cl:
            pk = _fixing_key(kind, key, merged[(kind, key)])
            if cid not in prov[pk]:
                prov[pk].append(cid)
    for cid in sorted(per_clause, key=str):
        for i, cf in enumerate(per_clause[cid].conditional):
            kept = [a for a in cf.fixing.atoms() if (a[0], a[1], cid) not in blocked_cl]
            if kept:
                cond.append((cid, i, ConditionalFixing(cf.predicate, _from_atoms(kept))))
    out = _from_atoms([(k, key, v) for (k, key), v in sorted(merged.items())])
    out.conditional = [c for _cid, _i, c in cond]
    for j, (cid, _i, _c) in enumerate(cond):
        prov["conditional:%d" % j] = [cid]
    out.provenance = prov
    out.__post_init__()
    return out, conflicts


# ===========================================================================
# 15. DESIGN SPACE, FEASIBLE-SET PROJECTION AND FREE-VARIABLE REDUCTION
# ===========================================================================
@dataclass
class Design:
    """One point ``x = (z, y, w, k)`` of the search space."""
    z: Dict[str, int]
    y: Dict[str, float]
    w: Dict[str, int]
    k: Dict[str, str]

    def key(self) -> Tuple[Any, ...]:
        """Canonical hashable form (sorted by variable name)."""
        return (tuple(sorted(self.z.items())), tuple(sorted(self.y.items())),
                tuple(sorted(self.w.items())), tuple(sorted(self.k.items())))

    def copy(self) -> "Design":
        """Independent copy."""
        return Design(dict(self.z), dict(self.y), dict(self.w), dict(self.k))

    def as_dict(self) -> Dict[str, Any]:
        """JSON-able dict."""
        return {"z": dict(sorted(self.z.items())), "y": dict(sorted(self.y.items())),
                "w": dict(sorted(self.w.items())), "k": dict(sorted(self.k.items()))}


def _midpoint(lo: float, hi: float) -> float:
    """``(lo + hi) / 2`` without overflow when ``lo + hi`` exceeds the float range."""
    m = 0.5 * (lo + hi)
    if math.isinf(m):
        m = 0.5 * lo + 0.5 * hi
    return min(max(m, lo), hi)


class DesignSpace:
    """The variables a search may set: edges, continuous quantities, crossings, junctions."""

    def __init__(self, edges: Sequence[str], *, baseline_edges: Sequence[str] = (),
                 continuous: Optional[Dict[str, Tuple[float, float]]] = None,
                 crossing_nodes: Sequence[str] = (), junction_nodes: Sequence[str] = (),
                 baseline_y: Optional[Dict[str, float]] = None,
                 baseline_junctions: Optional[Dict[str, str]] = None) -> None:
        for nm, tbl in (("continuous", continuous), ("baseline_y", baseline_y),
                        ("baseline_junctions", baseline_junctions)):
            if tbl is not None and not isinstance(tbl, dict):
                raise PolicyCompilerError("%s must be a dict." % nm)
        self.edges = sorted(set(_as_id_list(edges, "edges")))
        self.baseline_edges = sorted(set(_as_id_list(baseline_edges, "baseline_edges")))
        if not set(self.baseline_edges) <= set(self.edges):
            raise PolicyCompilerError("baseline_edges must be a subset of edges.")
        self.continuous: Dict[str, Tuple[float, float]] = {}
        for v in sorted(continuous or {}, key=str):
            pair = (continuous or {})[v]
            if isinstance(pair, (str, bytes)) or not hasattr(pair, "__len__") or len(pair) != 2:
                raise PolicyCompilerError("continuous[%r] must be a (lo, hi) pair." % v)
            lo, hi = _as_float(pair[0], "lo"), _as_float(pair[1], "hi")
            if lo > hi:
                raise PolicyCompilerError("continuous[%s] has lo > hi." % v)
            self.continuous[_as_id(v, "variable")] = (lo, hi)
        self.crossing_nodes = sorted(set(_as_id_list(crossing_nodes, "crossing_nodes")))
        self.junction_nodes = sorted(set(_as_id_list(junction_nodes, "junction_nodes")))
        self.baseline_y: Dict[str, float] = {}
        for v in sorted(baseline_y or {}, key=str):
            if v not in self.continuous:
                raise PolicyCompilerError("baseline_y names unknown variable %r." % v)
            x = _as_float((baseline_y or {})[v], "baseline_y[%s]" % v)
            lo, hi = self.continuous[v]
            if not (lo <= x <= hi):
                raise PolicyCompilerError("baseline_y[%s] outside its range." % v)
            self.baseline_y[v] = x
        self.baseline_junctions: Dict[str, str] = {}
        for n in sorted(baseline_junctions or {}, key=str):
            if n not in self.junction_nodes:
                raise PolicyCompilerError("baseline_junctions names unknown node %r." % n)
            self.baseline_junctions[n] = _as_choice((baseline_junctions or {})[n],
                                                    JUNCTION_TYPES, "junction type")

    def baseline_design(self) -> Design:
        """The do-nothing design (missing baselines: range midpoint, first type in K)."""
        z = dict((e, 1 if e in self.baseline_edges else 0) for e in self.edges)
        y = dict((v, self.baseline_y.get(v, _midpoint(lo, hi)))
                 for v, (lo, hi) in sorted(self.continuous.items()))
        w = dict((n, 0) for n in self.crossing_nodes)
        k = dict((n, self.baseline_junctions.get(n, JUNCTION_TYPES[0])) for n in self.junction_nodes)
        return Design(z, y, w, k)

    def validate(self, design: Design) -> None:
        """Raise unless ``design`` sets exactly this space's variables to legal values."""
        if not isinstance(design, Design):
            raise PolicyCompilerError("expected a Design.")
        if sorted(design.z, key=str) != self.edges or sorted(design.w, key=str) != \
                self.crossing_nodes or sorted(design.k, key=str) != self.junction_nodes \
                or sorted(design.y, key=str) != sorted(self.continuous):
            raise PolicyCompilerError("design variables do not match the design space.")
        for e in self.edges:
            if isinstance(design.z[e], bool) or design.z[e] not in (0, 1):
                raise PolicyCompilerError("z[%s] must be 0 or 1." % e)
        for n in self.crossing_nodes:
            if isinstance(design.w[n], bool) or design.w[n] not in (0, 1):
                raise PolicyCompilerError("w[%s] must be 0 or 1." % n)
        for n in self.junction_nodes:
            _as_choice(design.k[n], JUNCTION_TYPES, "k[%s]" % n)
        for v, (lo, hi) in self.continuous.items():
            x = _as_float(design.y[v], "y[%s]" % v)
            if not (lo <= x <= hi):
                raise PolicyCompilerError("y[%s] = %g outside [%g, %g]." % (v, x, lo, hi))


def applicable_fixings(fixings: CompiledFixings, scenario: Optional[Scenario] = None
                       ) -> CompiledFixings:
    """Unconditional fixings plus every conditional one whose predicate holds. With
    ``scenario=None`` the conditional fixings are ignored (they are scenario-specific by
    definition).  Contradictory active fixings raise."""
    if not isinstance(fixings, CompiledFixings):
        raise PolicyCompilerError("expected CompiledFixings.")
    parts = {"base": CompiledFixings(fixings.fixed_zero, fixings.fixed_one, fixings.bounds,
                                     fixings.crossing_fixings, fixings.junction_domains)}
    if scenario is not None:
        for i, c in enumerate(fixings.conditional):
            if c.predicate.holds(scenario):
                parts["cond%04d" % i] = c.fixing
    merged, conflicts = merge_fixings(parts)
    if conflicts:
        raise PolicyCompilerError("active fixings contradict each other on %s."
                                  % ", ".join(c.key for c in conflicts))
    return merged


def _check_design_values(design: Design) -> None:
    """Each value a design sets must be legal (0/1 binaries, finite y, a type in K).
    ``violations`` accepts a design over any subset of variables, so it cannot use
    ``DesignSpace.validate``; without this a NaN width on an unbounded variable, or ``z = 2`` on
    a free edge, would pass as feasible."""
    for nm in ("z", "y", "w", "k"):
        if not isinstance(getattr(design, nm), dict):
            raise PolicyCompilerError("design.%s must be a dict." % nm)
    for nm in ("z", "w"):
        for v, x in sorted(getattr(design, nm).items(), key=lambda kv: str(kv[0])):
            if isinstance(x, bool) or x not in (0, 1):
                raise PolicyCompilerError("%s[%s] must be 0 or 1 (got %r)." % (nm, v, x))
    for v in sorted(design.y, key=str):
        _as_float(design.y[v], "y[%s]" % v)
    for v in sorted(design.k, key=str):
        _as_choice(design.k[v], JUNCTION_TYPES, "k[%s]" % v)


def violations(design: Design, fixings: CompiledFixings,
               scenario: Optional[Scenario] = None) -> List[str]:
    """Provenance-style keys of every active fixing the design breaks (sorted)."""
    if not isinstance(design, Design):
        raise PolicyCompilerError("violations needs a Design.")
    _check_design_values(design)
    act = applicable_fixings(fixings, scenario)
    out = ["fixedZero:%s" % e for e in act.fixed_zero if design.z.get(e, 0) != 0]
    out += ["fixedOne:%s" % e for e in act.fixed_one if e in design.z and design.z[e] != 1]
    out += ["bounds:%s" % v for v, (lo, hi) in act.bounds.items()
            if v in design.y and not (lo <= design.y[v] <= hi)]
    out += ["crossingFixings:%s" % n for n, x in act.crossing_fixings.items()
            if n in design.w and design.w[n] != x]
    out += ["junctionDomains:%s" % n for n, dom in act.junction_domains.items()
            if n in design.k and design.k[n] not in dom]
    return sorted(out)


def is_feasible(design: Design, fixings: CompiledFixings,
                scenario: Optional[Scenario] = None) -> bool:
    """No active fixing is violated."""
    return not violations(design, fixings, scenario)


@dataclass
class ProjectionResult:
    """The projected design and every change the projection made."""
    design: Design
    changes: List[Tuple[str, Any, Any]]


def _reduced_range(space: DesignSpace, act: CompiledFixings, v: str) -> Tuple[float, float]:
    """Intersection of a variable's space range with its active bound."""
    lo, hi = space.continuous[v]
    if v in act.bounds:
        blo, bhi = act.bounds[v]
        lo, hi = max(lo, blo), min(hi, bhi)
        if lo > hi:
            raise PolicyCompilerError("bound on %s does not meet its range: infeasible." % v)
    return lo, hi


def project_design(design: Design, fixings: CompiledFixings, space: DesignSpace,
                   scenario: Optional[Scenario] = None) -> ProjectionResult:
    """Nearest feasible design: fixed variables set, continuous ones clipped, k_v re-domained.
    Idempotent (projecting twice changes nothing more) and always feasible.  A junction outside
    its domain moves to the baseline type if that is allowed, otherwise to the first allowed
    type in K's order."""
    if not isinstance(space, DesignSpace):
        raise PolicyCompilerError("project_design needs a DesignSpace.")
    space.validate(design)
    act = applicable_fixings(fixings, scenario)
    out = design.copy()
    changes: List[Tuple[str, Any, Any]] = []
    for e in act.fixed_zero:
        if e in out.z and out.z[e] != 0:
            changes.append(("z:%s" % e, out.z[e], 0))
            out.z[e] = 0
    for e in act.fixed_one:
        if e in out.z and out.z[e] != 1:
            changes.append(("z:%s" % e, out.z[e], 1))
            out.z[e] = 1
    for v in sorted(out.y):
        lo, hi = _reduced_range(space, act, v)
        x = min(max(out.y[v], lo), hi)
        if x != out.y[v]:
            changes.append(("y:%s" % v, out.y[v], x))
            out.y[v] = x
    for n, x in sorted(act.crossing_fixings.items()):
        if n in out.w and out.w[n] != x:
            changes.append(("w:%s" % n, out.w[n], x))
            out.w[n] = x
    for n, dom in sorted(act.junction_domains.items()):
        if n in out.k and out.k[n] not in dom:
            if not dom:
                raise PolicyCompilerError("empty junction domain at %s: infeasible." % n)
            base = space.baseline_junctions.get(n)
            new = base if base in dom else [t for t in JUNCTION_TYPES if t in dom][0]
            changes.append(("k:%s" % n, out.k[n], new))
            out.k[n] = new
    return ProjectionResult(out, changes)


@dataclass
class FreeVariableReport:
    """What is left to search after the fixings are applied."""
    free_edges: List[str]
    fixed_edges: Dict[str, int]
    free_crossings: List[str]
    fixed_crossings: Dict[str, int]
    junction_domains: Dict[str, List[str]]
    y_ranges: Dict[str, Tuple[float, float]]
    pinned_y: List[str]
    unknown_keys: List[str]
    n_discrete_before: int
    n_discrete_after: int

    @property
    def reduction_factor(self) -> float:
        """Discrete designs before / after (>= 1)."""
        return self.n_discrete_before / self.n_discrete_after if self.n_discrete_after else math.inf


def free_variables(space: DesignSpace, fixings: CompiledFixings,
                   scenario: Optional[Scenario] = None) -> FreeVariableReport:
    """Free-variable reduction: which binaries remain free and how far domains shrink."""
    if not isinstance(space, DesignSpace):
        raise PolicyCompilerError("free_variables needs a DesignSpace.")
    act = applicable_fixings(fixings, scenario)
    fixed_e: Dict[str, int] = dict([(e, 0) for e in act.fixed_zero if e in space.edges]
                                   + [(e, 1) for e in act.fixed_one if e in space.edges])
    fixed_w = dict((n, x) for n, x in act.crossing_fixings.items() if n in space.crossing_nodes)
    doms = dict((n, list(act.junction_domains.get(n, JUNCTION_TYPES)))
                for n in space.junction_nodes)
    doms = dict((n, [t for t in JUNCTION_TYPES if t in d]) for n, d in doms.items())
    yr = dict((v, _reduced_range(space, act, v)) for v in sorted(space.continuous))
    pinned = sorted(v for v, (lo, hi) in yr.items() if lo == hi)
    # known PER KIND: a crossing fixing on a junction-only node is unknown
    known = {"z": set(space.edges), "w": set(space.crossing_nodes),
             "k": set(space.junction_nodes), "y": set(space.continuous)}
    unknown = sorted(set(k for kind, k, _v in act.atoms() if k not in known[kind]))
    before = (2 ** len(space.edges)) * (2 ** len(space.crossing_nodes)) * \
        (len(JUNCTION_TYPES) ** len(space.junction_nodes))
    after = (2 ** (len(space.edges) - len(fixed_e))) * \
        (2 ** (len(space.crossing_nodes) - len(fixed_w)))
    for n in space.junction_nodes:
        after *= len(doms[n])
    return FreeVariableReport(sorted(e for e in space.edges if e not in fixed_e), fixed_e,
                              sorted(n for n in space.crossing_nodes if n not in fixed_w),
                              fixed_w, doms, yr, pinned, unknown, before, after)


def default_y_levels(space: DesignSpace) -> Dict[str, List[float]]:
    """Three levels per continuous variable: lo, midpoint, hi (deduplicated)."""
    if not isinstance(space, DesignSpace):
        raise PolicyCompilerError("default_y_levels needs a DesignSpace.")
    out: Dict[str, List[float]] = {}
    for v, (lo, hi) in sorted(space.continuous.items()):
        out[v] = sorted(set([lo, _midpoint(lo, hi), hi]))
    return out


def enumerate_designs(space: DesignSpace, fixings: Optional[CompiledFixings] = None, *,
                      scenario: Optional[Scenario] = None,
                      y_levels: Optional[Dict[str, Sequence[float]]] = None
                      ) -> Iterator[Design]:
    """The candidate generator: every design over the FREE variables only. Fixed binaries at
    their value, junction types over the restricted domain, levels projected into the reduced
    range: a violating design is unrepresentable, not filtered."""
    if not isinstance(space, DesignSpace):
        raise PolicyCompilerError("enumerate_designs needs a DesignSpace.")
    fx = fixings if fixings is not None else CompiledFixings()
    rep = free_variables(space, fx, scenario)
    levels = default_y_levels(space) if y_levels is None else y_levels
    if not isinstance(levels, dict):
        raise PolicyCompilerError("y_levels must map variable -> levels.")
    ylv: List[Tuple[str, List[float]]] = []
    for v in sorted(space.continuous):
        if v not in levels:
            raise PolicyCompilerError("y_levels lacks variable %r." % v)
        raw = _as_vector(levels[v], "y_levels[%s]" % v)
        if not raw:
            raise PolicyCompilerError("y_levels[%s] is empty." % v)
        lo, hi = rep.y_ranges[v]
        ylv.append((v, sorted(set(min(max(x, lo), hi) for x in raw))))
    e_free = rep.free_edges
    w_free = rep.free_crossings
    k_nodes = space.junction_nodes
    axes: List[Sequence[Any]] = ([(0, 1)] * len(e_free) + [(0, 1)] * len(w_free)
                                 + [rep.junction_domains[n] for n in k_nodes]
                                 + [lv for _v, lv in ylv])
    ne, nw, nk = len(e_free), len(w_free), len(k_nodes)
    for combo in itertools.product(*axes):
        z = dict(rep.fixed_edges)
        for i, e in enumerate(e_free):
            z[e] = combo[i]
        w = dict(rep.fixed_crossings)
        for i, n in enumerate(w_free):
            w[n] = combo[ne + i]
        k = dict((n, combo[ne + nw + i]) for i, n in enumerate(k_nodes))
        y = dict((v, combo[ne + nw + nk + i]) for i, (v, _lv) in enumerate(ylv))
        yield Design(dict(sorted(z.items())), y, dict(sorted(w.items())), k)


def count_designs(space: DesignSpace, fixings: Optional[CompiledFixings] = None, *,
                  scenario: Optional[Scenario] = None,
                  y_levels: Optional[Dict[str, Sequence[float]]] = None) -> int:
    """Size of the generator's output (computed by enumeration)."""
    return sum(1 for _ in enumerate_designs(space, fixings, scenario=scenario, y_levels=y_levels))


# ===========================================================================
# 16. NON-DOMINATED SETS AND RELAXATION ANALYSIS
# ===========================================================================
def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """Pareto dominance for MINIMISATION: no worse everywhere, better somewhere."""
    av = _as_vector(a, "a")
    bv = _as_vector(b, "b")
    if len(av) != len(bv) or not av:
        raise PolicyCompilerError("dominance needs two vectors of the same positive length.")
    return all(x <= y for x, y in zip(av, bv)) and any(x < y for x, y in zip(av, bv))


def non_dominated_indices(points: Sequence[Sequence[float]]) -> List[int]:
    """Indices of the non-dominated points (minimisation); duplicates all survive."""
    pts = [_as_vector(p, "points[%d]" % i) for i, p in enumerate(_as_seq(points, "points"))]
    if pts and len(set(len(p) for p in pts)) != 1:
        raise PolicyCompilerError("points must share one dimension.")
    if pts and not pts[0]:
        raise PolicyCompilerError("points must have at least one objective.")
    out = []
    for i, p in enumerate(pts):
        if not any(dominates(q, p) for j, q in enumerate(pts) if j != i):
            out.append(i)
    return out


def _hv(points: List[Tuple[float, ...]], ref: Tuple[float, ...]) -> float:
    """Hypervolume by slicing on the last objective (exact, small sets)."""
    d = len(ref)
    if not points:
        return 0.0
    if d == 1:
        return ref[0] - min(p[0] for p in points)
    pts = sorted(points, key=lambda p: (p[-1],) + p)
    terms: List[float] = []
    for i, p in enumerate(pts):
        nxt = pts[i + 1][-1] if i + 1 < len(pts) else ref[-1]
        width = nxt - p[-1]
        if width > 0.0:
            terms.append(width * _hv([q[:-1] for q in pts[:i + 1]], ref[:-1]))
    return math.fsum(terms)


def hypervolume(points: Sequence[Sequence[float]], reference: Sequence[float]) -> float:
    """Exact dominated hypervolume w.r.t. ``reference`` (minimisation). Points not strictly
    better than the reference in every objective add nothing.  Exponential in the dimension,
    intended for the small fronts of the relaxation analysis."""
    ref = tuple(_as_vector(reference, "reference"))
    if not ref:
        raise PolicyCompilerError("reference must have at least one coordinate.")
    pts = []
    for i, p in enumerate(_as_seq(points, "points")):
        v = tuple(_as_vector(p, "points[%d]" % i))
        if len(v) != len(ref):
            raise PolicyCompilerError("points[%d] has the wrong dimension." % i)
        if all(x < r for x, r in zip(v, ref)):
            pts.append(v)
    return _hv(pts, ref)


@dataclass
class ParetoSet:
    """Feasible designs and the non-dominated subset under one fixing set."""
    n_feasible: int
    designs: List[Design]
    objectives: List[Tuple[float, ...]]

    def vectors(self) -> List[Tuple[float, ...]]:
        """Sorted objective vectors of the non-dominated designs."""
        return sorted(self.objectives)

    def keys(self) -> List[Tuple[Any, ...]]:
        """Sorted canonical keys of the non-dominated designs."""
        return sorted(d.key() for d in self.designs)


def pareto_set(space: DesignSpace, fixings: Optional[CompiledFixings],
               objective: Callable[[Design], Sequence[float]], *,
               scenario: Optional[Scenario] = None,
               y_levels: Optional[Dict[str, Sequence[float]]] = None,
               cache: Optional[Dict[Tuple[Any, ...], Tuple[float, ...]]] = None) -> ParetoSet:
    """Exhaustively enumerate the feasible designs and keep the non-dominated ones."""
    if not callable(objective):
        raise PolicyCompilerError("objective must be callable.")
    memo = cache if cache is not None else {}
    designs = list(enumerate_designs(space, fixings, scenario=scenario, y_levels=y_levels))
    vecs: List[Tuple[float, ...]] = []
    for d in designs:
        k = d.key()
        if k not in memo:
            memo[k] = tuple(_as_vector(objective(d), "objective(design)"))
        vecs.append(memo[k])
    nd = non_dominated_indices(vecs)
    order = sorted(nd, key=lambda i: (vecs[i], designs[i].key()))
    return ParetoSet(len(designs), [designs[i] for i in order], [vecs[i] for i in order])


@dataclass
class RelaxationEntry:
    """Effect of dropping ONE compiled constraint on the non-dominated set."""
    constraint_id: str
    binding: bool
    n_feasible_before: int
    n_feasible_after: int
    added: List[Tuple[float, ...]]
    removed: List[Tuple[float, ...]]
    identical_designs: bool
    hv_before: Optional[float]
    hv_after: Optional[float]

    @property
    def hv_gain(self) -> Optional[float]:
        """Hypervolume gained by relaxing (None without a reference)."""
        if self.hv_before is None or self.hv_after is None:
            return None
        return self.hv_after - self.hv_before

    def as_dict(self) -> Dict[str, Any]:
        """JSON-able camelCase dict."""
        return {"constraintId": self.constraint_id, "binding": self.binding,
                "nFeasibleBefore": self.n_feasible_before,
                "nFeasibleAfter": self.n_feasible_after,
                "added": [list(v) for v in self.added],
                "removed": [list(v) for v in self.removed],
                "identicalDesigns": self.identical_designs,
                "hvBefore": _json_safe(self.hv_before), "hvAfter": _json_safe(self.hv_after),
                "hvGain": _json_safe(self.hv_gain)}


@dataclass
class RelaxationReport:
    """Relaxation analysis over every compiled constraint."""
    baseline: ParetoSet
    entries: List[RelaxationEntry]

    @property
    def binding_ids(self) -> List[str]:
        """Constraints whose removal changes the non-dominated set."""
        return [e.constraint_id for e in self.entries if e.binding]


def relaxation_analysis(space: DesignSpace, per_constraint: Dict[str, CompiledFixings],
                        objective: Callable[[Design], Sequence[float]], *,
                        scenario: Optional[Scenario] = None,
                        y_levels: Optional[Dict[str, Sequence[float]]] = None,
                        reference: Optional[Sequence[float]] = None) -> RelaxationReport:
    """Drop each constraint in turn and measure the change in the non-dominated set. Exact, by
    enumeration (small or reduced spaces).  Binding iff the SET of distinct non-dominated
    objective vectors changes; designs that only tie a front vector are alternative optima,
    reported by ``identical_designs``."""
    if not isinstance(per_constraint, dict):
        raise PolicyCompilerError("per_constraint must map id -> CompiledFixings.")
    for k in sorted(per_constraint, key=str):
        _as_id(k, "constraint id")
    cache: Dict[Tuple[Any, ...], Tuple[float, ...]] = {}
    full, conflicts = merge_fixings(per_constraint)
    if conflicts:
        raise PolicyCompilerError("constraints conflict (%s); resolve before relaxing."
                                  % ", ".join(c.key for c in conflicts))
    base = pareto_set(space, full, objective, scenario=scenario, y_levels=y_levels, cache=cache)
    ref = None if reference is None else _as_vector(reference, "reference")
    hv0 = None if ref is None else hypervolume(base.objectives, ref)
    entries: List[RelaxationEntry] = []
    for cid in sorted(per_constraint):
        rest = dict((k, v) for k, v in per_constraint.items() if k != cid)
        fx, _c = merge_fixings(rest)
        ps = pareto_set(space, fx, objective, scenario=scenario, y_levels=y_levels, cache=cache)
        before, after = set(base.vectors()), set(ps.vectors())
        added = sorted(after - before)
        removed = sorted(before - after)
        entries.append(RelaxationEntry(
            cid, before != after, base.n_feasible, ps.n_feasible, added, removed,
            base.keys() == ps.keys(), hv0,
            None if ref is None else hypervolume(ps.objectives, ref)))
    return RelaxationReport(base, entries)


# ===========================================================================
# 17. POLICY-CHECK REPORT
# ===========================================================================
CHECK_STATUSES = ("satisfied", "in_tension", "abstained", "not_applicable")


@dataclass
class PolicyCheckRow:
    """One clause's verdict on one design, with its citation."""
    clause_id: str
    status: str
    quote: str
    page: int
    document_id: str
    violated: List[str]
    detail: str

    def as_dict(self) -> Dict[str, Any]:
        """JSON-able camelCase dict."""
        return {"clauseId": self.clause_id, "status": self.status, "quote": self.quote,
                "page": self.page, "documentId": self.document_id,
                "violated": list(self.violated), "detail": self.detail}


@dataclass
class PolicyCheckReport:
    """Clause-by-clause satisfied / in tension report for a design."""
    rows: List[PolicyCheckRow]
    counts: Dict[str, int]
    compliance_ratio: float

    def as_dict(self) -> Dict[str, Any]:
        """JSON-able camelCase dict."""
        return {"rows": [r.as_dict() for r in self.rows], "counts": dict(self.counts),
                "complianceRatio": self.compliance_ratio}


def policy_check(design: Design, result: CompilationResult,
                 scenario: Optional[Scenario] = None) -> PolicyCheckReport:
    """Check a design clause by clause against what each clause compiled to. Conditional clauses
    count as active without a scenario; queued ones are ``abstained`` unless a fixing they still
    contribute is broken. Compliance = satisfied / (satisfied + in tension), or 1."""
    if not isinstance(design, Design):
        raise PolicyCompilerError("policy_check needs a Design.")
    if not isinstance(result, CompilationResult):
        raise PolicyCompilerError("policy_check needs a CompilationResult.")
    if scenario is not None and not isinstance(scenario, Scenario):
        raise PolicyCompilerError("scenario must be a Scenario or None.")
    queued = set(result.queue.clause_ids())
    rows: List[PolicyCheckRow] = []
    for cid in sorted(result.clauses):
        c = result.clauses[cid]
        # a queued clause's atoms that survived the merge still bind: a conflict withholds only
        # the clashing keys (as merge_fixings does), so check the rest
        cut = set(("zwyk"[CONFLICT_KINDS.index(cf.kind)], cf.key) for cf in result.conflicts
                  if cid in cf.clause_ids)
        keep = lambda f: _from_atoms([a for a in f.atoms() if a[:2] not in cut])  # noqa: E731

        def row(status: str, viol: Sequence[str], detail: str) -> None:
            rows.append(PolicyCheckRow(cid, status, c.quote, c.page, c.document_id,
                                       list(viol), detail))
        why = ", ".join(sorted(set(it.reason for it in result.queue.items() if it.clause_id == cid)))
        fx = result.per_clause.get(cid, CompiledFixings())
        fx = replace(keep(fx), conditional=[ConditionalFixing(f.predicate, keep(f.fixing))
                                            for f in fx.conditional if keep(f.fixing).atoms()])
        if cid in queued and fx.is_empty:
            row("abstained", [], why)
            continue
        if cid in result.skipped or cid not in result.per_clause:
            row("not_applicable", [], result.skipped.get(cid, "nothing compiled"))
            continue
        if fx.is_empty:
            # a bare permission can never be in tension: not a checkable clause
            row("not_applicable", [], "compiled to no fixing")
            continue
        parts = {"u": CompiledFixings(fx.fixed_zero, fx.fixed_one, fx.bounds,
                                      fx.crossing_fixings, fx.junction_domains)}
        detail = "unconditional"
        for i, cf in enumerate(fx.conditional):
            if scenario is not None and not cf.predicate.holds(scenario):
                continue
            parts["c%d" % i] = cf.fixing
            detail = "conditional (checked as active)" if scenario is None else "conditional (active)"
        if len(parts) == 1 and fx.conditional:
            row("abstained" if why else "not_applicable", [],
                why or "predicate does not hold in this scenario")
            continue
        viol = violations(design, merge_fixings(parts)[0])
        row("in_tension" if viol else "abstained" if why else "satisfied", viol,
            "%s; queued: %s" % (detail, why) if why else detail)
    counts = dict((s, sum(1 for r in rows if r.status == s)) for s in CHECK_STATUSES)
    den = counts["satisfied"] + counts["in_tension"]
    ratio = counts["satisfied"] / den if den else 1.0
    return PolicyCheckReport(rows, counts, ratio)


# ===========================================================================
# 18. PROVENANCE GRAPH, CITATION CHAINS AND AGREEMENT SCORING
# ===========================================================================
PROVENANCE_NODE_TYPES = ("document", "page", "span", "clause", "binding", "fixing")


class ProvenanceGraph:
    """Typed DAG document -> page -> span -> clause -> binding -> fixing."""

    def __init__(self) -> None:
        self._nodes: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        self._parents: Dict[str, List[Tuple[str, str]]] = {}

    def add_node(self, node_id: str, node_type: str, **attrs: Any) -> None:
        """Add (or update) a typed node."""
        nid = _as_id(node_id, "node_id")
        _as_choice(node_type, PROVENANCE_NODE_TYPES, "node_type")
        if nid in self._nodes and self._nodes[nid][0] != node_type:
            raise PolicyCompilerError("node %r already has type %s." % (nid, self._nodes[nid][0]))
        self._nodes[nid] = (node_type, dict(attrs))
        self._parents.setdefault(nid, [])

    def add_edge(self, src: str, dst: str, relation: str) -> None:
        """Add a provenance edge ``src -> dst`` (both nodes must exist)."""
        for n in (_as_id(src, "src"), _as_id(dst, "dst")):
            if n not in self._nodes:
                raise PolicyCompilerError("unknown provenance node %r." % n)
        pair = (src, _as_id(relation, "relation"))
        if pair not in self._parents[dst]:
            self._parents[dst].append(pair)
            self._parents[dst].sort()

    def node_type(self, node_id: str) -> str:
        """Type of a node."""
        if _as_id(node_id, "node_id") not in self._nodes:
            raise PolicyCompilerError("unknown provenance node %r." % node_id)
        return self._nodes[node_id][0]

    def attrs(self, node_id: str) -> Dict[str, Any]:
        """Attributes of a node (copy)."""
        self.node_type(node_id)
        return dict(self._nodes[node_id][1])

    @property
    def nodes(self) -> List[str]:
        """Sorted node ids."""
        return sorted(self._nodes)

    @property
    def edges(self) -> List[Tuple[str, str, str]]:
        """Sorted ``(src, dst, relation)`` triples."""
        return sorted((s, d, r) for d in self._parents for s, r in self._parents[d])

    def citation_chains(self, node_id: str) -> List[List[str]]:
        """Every path from a node back to a document (node first, document last)."""
        self.node_type(node_id)
        out: List[List[str]] = []

        def walk(n: str, path: List[str]) -> None:
            if self._nodes[n][0] == "document":
                out.append(path)
                return
            for p, _r in self._parents[n]:
                if p not in path:
                    walk(p, path + [p])
        walk(node_id, [node_id])
        return sorted(out)

    def verify(self) -> List[str]:
        """Problems: a fixing with no chain to a document, or a chain through an unverified span."""
        problems: List[str] = []
        for n in self.nodes:
            if self._nodes[n][0] != "fixing":
                continue
            chains = self.citation_chains(n)
            if not chains:
                problems.append("%s has no citation chain" % n)
            for ch in chains:
                spans = [x for x in ch if self._nodes[x][0] == "span"]
                if not spans or not all(self._nodes[s][1].get("verified") for s in spans):
                    problems.append("%s cites an unverified span" % n)
        return sorted(set(problems))

    def as_dict(self) -> Dict[str, Any]:
        """JSON-able dict of nodes and edges."""
        return {"nodes": [{"id": n, "type": self._nodes[n][0],
                           "attrs": dict((k, _json_safe(v)) for k, v in
                                         sorted(self._nodes[n][1].items()))}
                          for n in self.nodes],
                "edges": [{"source": s, "target": d, "relation": r} for s, d, r in self.edges]}


def build_provenance_graph(result: CompilationResult) -> ProvenanceGraph:
    """The provenance graph of a compilation: every surviving fixing back to its page."""
    if not isinstance(result, CompilationResult):
        raise PolicyCompilerError("build_provenance_graph needs a CompilationResult.")
    g = ProvenanceGraph()
    for cid in sorted(result.clauses):
        c = result.clauses[cid]
        doc = "doc:%s" % c.document_id
        page = "page:%s:%d" % (c.document_id, c.page)
        g.add_node(doc, "document", documentId=c.document_id)
        g.add_node(page, "page", page=c.page)
        g.add_edge(doc, page, "contains")
        g.add_node("clause:%s" % cid, "clause", quote=c.quote)
        if cid in result.spans:
            sv = result.spans[cid]
            g.add_node("span:%s" % cid, "span", start=sv.start, end=sv.end, status=sv.status,
                       verified=sv.verified)
            g.add_edge(page, "span:%s" % cid, "contains")
            g.add_edge("span:%s" % cid, "clause:%s" % cid, "quoted_by")
        if cid in result.bindings and result.bindings[cid].status == "bound":
            b = result.bindings[cid]
            g.add_node("bind:%s" % cid, "binding", target=b.target_id, method=b.method)
            g.add_edge("clause:%s" % cid, "bind:%s" % cid, "binds_to")
    for key in sorted(result.fixings.provenance):
        g.add_node("fix:%s" % key, "fixing", key=key)
        for cid in result.fixings.provenance[key]:
            if "bind:%s" % cid in g.nodes:
                g.add_edge("bind:%s" % cid, "fix:%s" % key, "compiles_to")
    return g


@dataclass
class AgreementReport:
    """Agreement of predictions with a gold standard, abstentions counted apart."""
    n_gold: int
    n_answered: int
    n_correct: int
    n_abstained: int
    precision: float
    recall: float
    f1: float
    abstention_rate: float

    def as_dict(self) -> Dict[str, Any]:
        """JSON-able camelCase dict."""
        return {"nGold": self.n_gold, "nAnswered": self.n_answered,
                "nCorrect": self.n_correct, "nAbstained": self.n_abstained,
                "precision": self.precision, "recall": self.recall, "f1": self.f1,
                "abstentionRate": self.abstention_rate}


def agreement_score(predicted: Dict[str, Optional[str]], gold: Dict[str, str]) -> AgreementReport:
    """Precision over answered items, recall over all gold items, abstention rate. ``None`` or
    ``ABSTAIN`` (and a missing key) is an abstention.  Precision is 1.0 when nothing was
    answered -- an abstainer is never wrong, which is exactly why recall and the abstention rate
    are reported beside it."""
    if not isinstance(predicted, dict) or not isinstance(gold, dict):
        raise PolicyCompilerError("predicted and gold must be dicts.")
    if not gold:
        raise PolicyCompilerError("gold must not be empty.")
    for k in sorted(predicted, key=str):
        if k not in gold:
            raise PolicyCompilerError("prediction for %r has no gold label." % (k,))
    answered = correct = abstained = 0
    for k in sorted(gold, key=str):
        _as_str(gold[k], "gold[%s]" % k, allow_empty=False)
        p = predicted.get(k)
        if p is None or p == ABSTAIN:
            abstained += 1
            continue
        answered += 1
        if p == gold[k]:
            correct += 1
    n = len(gold)
    prec = correct / answered if answered else 1.0
    rec = correct / n
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0.0 else 0.0
    return AgreementReport(n, answered, correct, abstained, prec, rec, f1, abstained / n)


def binding_agreement(result: CompilationResult, gold: Dict[str, str]) -> AgreementReport:
    """Agreement of the compiler's bindings with gold target ids (queued = abstained)."""
    if not isinstance(result, CompilationResult):
        raise PolicyCompilerError("binding_agreement needs a CompilationResult.")
    queued = set(result.queue.clause_ids())
    pred: Dict[str, Optional[str]] = {}
    for cid in sorted(gold, key=str):
        b = result.bindings.get(cid)
        pred[cid] = (b.target_id if (b is not None and b.status == "bound" and cid not in queued)
                     else None)
    return agreement_score(pred, gold)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa between two raters' labels (1.0 when chance agreement is total)."""
    la, lb = _as_seq(a, "a"), _as_seq(b, "b")
    if len(la) != len(lb) or not la:
        raise PolicyCompilerError("label sequences must be non-empty and equally long.")
    for x in la + lb:
        _as_str(x, "label")
    n = len(la)
    po = sum(1 for x, y in zip(la, lb) if x == y) / n
    labels = sorted(set(la) | set(lb))
    pe = math.fsum((la.count(k) / n) * (lb.count(k) / n) for k in labels)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


# ===========================================================================
# 19. FRONTEND PAYLOADS
# ===========================================================================
def build_policy_compiler_payload(result: CompilationResult, *,
                                  report: Optional[PolicyCheckReport] = None,
                                  relaxation: Optional[RelaxationReport] = None,
                                  provenance: Optional[ProvenanceGraph] = None,
                                  agreement: Optional[AgreementReport] = None,
                                  max_items: int = 200) -> Dict[str, Any]:
    """Frontend-ready JSON dict: fixings, clause table, queue, conflicts and analyses."""
    if not isinstance(result, CompilationResult):
        raise PolicyCompilerError("build_policy_compiler_payload needs a CompilationResult.")
    cap = _as_int(max_items, "max_items", minimum=1)
    queued = set(result.queue.clause_ids())
    clauses = []
    for cid in sorted(result.clauses)[:cap]:
        c = result.clauses[cid]
        sv = result.spans.get(cid)
        b = result.bindings.get(cid)
        wd = result.wording[cid]
        clauses.append({
            "clauseId": cid, "documentId": c.document_id, "page": c.page, "quote": c.quote,
            "target": c.to_dict()["target"], "rulings": dict(sorted(c.rulings.items())),
            "span": None if sv is None else sv.as_dict(),
            "binding": None if b is None else b.as_dict(),
            "wording": {"modality": wd.modality, "p": wd.p, "cue": wd.cue},
            "status": ("abstained" if cid in queued else
                       result.skipped.get(cid, "compiled" if cid in result.per_clause
                                          else "not_compiled")),
        })
    payload: Dict[str, Any] = {
        "summary": {"nClauses": result.n_clauses, "nCompiled": len(result.compiled_clause_ids),
                    "nAbstained": len(queued), "nConflicts": len(result.conflicts),
                    "nFixings": result.fixings.n_fixings(),
                    "abstentionRate": result.abstention_rate()},
        "fixings": result.fixings.to_dict(),
        "clauses": clauses,
        "abstentionQueue": result.queue.as_list()[:cap],
        "conflicts": [c.as_dict() for c in result.conflicts[:cap]],
        "skipped": dict(sorted(result.skipped.items())),
        "priors": dict(sorted(result.priors.items())),
        "inForce": None if result.in_force is None else result.in_force.as_dict(),
    }
    if report is not None:
        payload["policyCheck"] = report.as_dict()
    if relaxation is not None:
        payload["relaxation"] = {
            "baselineFront": [list(v) for v in relaxation.baseline.vectors()[:cap]],
            "nFeasible": relaxation.baseline.n_feasible,
            "entries": [e.as_dict() for e in relaxation.entries[:cap]],
            "bindingIds": relaxation.binding_ids}
    if provenance is not None:
        payload["provenance"] = provenance.as_dict()
    if agreement is not None:
        payload["agreement"] = agreement.as_dict()
    return payload


def build_a2ui_payload(result: CompilationResult, *,
                       report: Optional[PolicyCheckReport] = None,
                       relaxation: Optional[RelaxationReport] = None,
                       top_n: int = 10) -> Dict[str, Any]:
    """Visualisation-ready dict: stat cards, chart series and the review table."""
    if not isinstance(result, CompilationResult):
        raise PolicyCompilerError("build_a2ui_payload needs a CompilationResult.")
    n = _as_int(top_n, "top_n", minimum=1)
    fx = result.fixings
    cards: List[Dict[str, Any]] = [
        {"label": "Clauses", "value": result.n_clauses, "unit": ""},
        {"label": "Compiled", "value": len(result.compiled_clause_ids), "unit": ""},
        {"label": "Abstained", "value": len(result.queue.clause_ids()), "unit": ""},
        {"label": "Abstention rate", "value": round(100.0 * result.abstention_rate(), 1),
         "unit": "%"},
        {"label": "Conflicts", "value": len(result.conflicts), "unit": ""},
        {"label": "Prohibited edges", "value": len(fx.fixed_zero), "unit": ""},
        {"label": "Protected edges", "value": len(fx.fixed_one), "unit": ""},
    ]
    if report is not None:
        cards.append({"label": "Policy compliance",
                      "value": round(100.0 * report.compliance_ratio, 1), "unit": "%"})
    if relaxation is not None:
        cards.append({"label": "Binding constraints",
                      "value": len(relaxation.binding_ids), "unit": ""})
    reasons: Dict[str, int] = {}
    for it in result.queue.items():
        reasons[it.reason] = reasons.get(it.reason, 0) + 1
    charts: Dict[str, Any] = {
        "fixingsByKind": [
            {"kind": "fixedZero", "count": len(fx.fixed_zero)},
            {"kind": "fixedOne", "count": len(fx.fixed_one)},
            {"kind": "bounds", "count": len(fx.bounds)},
            {"kind": "crossingFixings", "count": len(fx.crossing_fixings)},
            {"kind": "junctionDomains", "count": len(fx.junction_domains)},
            {"kind": "conditional", "count": len(fx.conditional)}],
        "abstentionReasons": [{"reason": r, "count": reasons[r]} for r in sorted(reasons)],
    }
    if relaxation is not None:
        charts["relaxationGain"] = [
            {"constraintId": e.constraint_id, "binding": e.binding,
             "added": len(e.added), "removed": len(e.removed),
             "hvGain": _json_safe(e.hv_gain)} for e in relaxation.entries[:n]]
    review = [{"clauseId": it.clause_id, "reason": it.reason,
               "ambiguity": round(it.ambiguity, 4), "candidates": list(it.candidates),
               "page": it.page} for it in result.queue.items()[:n]]
    return {"title": "Policy compiler -- %d clauses" % result.n_clauses,
            "statCards": cards, "charts": charts, "reviewTable": review}


# ===========================================================================
# 20. EXAMPLE DISTRICT -- a small TRO, gazetteer and design space with no data
# ===========================================================================
_EXAMPLE_TRO = (
    "DUBLIN CITY COUNCIL -- FITZWILLIAM AREA TRAFFIC ORDER 2024\n"
    "Article 3. No new road shall be constructed through Merrion Square Park.\n"
    "Article 4. Fitzwilliam Street shall not be closed to\n    through traffic.\n"
    "Article 5. A signal-controlled pedestrian crossing must be provided at the "
    "junction of Fitzwilliam Street and Lower Baggot Street.\n"
    "Article 6. A roundabout shall not be constructed at the junction of "
    "Fitzwilliam Street and Lower Baggot Street.\n"
    "Article 7. The footway on Lower Baggot Street must be not less than 2.5 metres "
    "and not more than 4.0 metres wide.\n"
    "Article 8. Loading is prohibited on Main Street 07:00-10:00 Mon-Fri.\n"
    "Article 9. Fitzwilliam Gate shall be closed to vehicles 22:00-06:00 except "
    "emergency vehicles.\n"
    "Article 10. The Council should prioritise pedestrian movement throughout the area.\n"
)
_EXAMPLE_AMENDMENT = (
    "FITZWILLIAM AREA TRAFFIC (AMENDMENT) ORDER 2025\n"
    "Article 2. For Article 7 substitute: the footway on Lower Baggot Street must be "
    "not less than 3.0 metres and not more than 4.0 metres wide.\n"
)


@dataclass
class ExampleDistrict:
    """Everything a pipeline needs to run the compiler end to end without data."""
    gazetteer: Gazetteer
    sources: Dict[str, str]
    clauses: List[PolicyClause]
    instruments: List[Instrument]
    at: str
    space: DesignSpace
    baseline_y: Dict[str, float]
    baseline_junctions: Dict[str, str]
    gold_bindings: Dict[str, str]

    def objective(self, d: Design) -> Tuple[float, float, float]:
        """(J_car, J_ped, J_cost), all minimised, for the example design space."""
        fw = d.y.get("footway_width:e5", 3.0)
        kt = d.k.get("n2", "priority")
        j_car = math.fsum([10.0, -3.0 * d.z["e1"], -3.0 * d.z["e2"], -2.0 * d.z["e5"],
                           -2.0 * d.z["c1"], -2.0 * d.z["c2"], 1.5 * d.w["n2"], 1.5 * d.w["n3"],
                           {"priority": 0.0, "signalised": 0.8, "signalised_banned_turns": 1.0,
                            "roundabout": -0.5, "mini_roundabout": 0.2}[kt],
                           0.25 * (fw - 2.0)])
        j_ped = math.fsum([10.0, -2.0 * d.w["n2"], -2.0 * d.w["n3"], -1.0 * fw,
                           1.5 * d.z["c1"], 1.5 * d.z["c2"],
                           -1.0 if kt in ("signalised", "signalised_banned_turns") else 0.0])
        j_cost = math.fsum([5.0 * d.z["c1"], 5.0 * d.z["c2"], 2.0 * d.w["n2"], 2.0 * d.w["n3"],
                            {"priority": 0.0, "signalised": 4.0, "signalised_banned_turns": 4.5,
                             "roundabout": 6.0, "mini_roundabout": 3.0}[kt],
                            abs(fw - 3.0), 8.0 * (1 - d.z["e1"]) + 8.0 * (1 - d.z["e2"])])
        return (j_car, j_ped, j_cost)


def build_example_district() -> ExampleDistrict:
    """A Fitzwilliam-area fixture: TRO + amendment, gazetteer, clauses, design space."""
    entries = [
        GazetteerEntry("s_fitzwilliam_st", "Fitzwilliam St", "street", edge_ids=("e1", "e2"),
                       node_ids=("n1", "n2", "n3"), coords=((0, 0), (0, 100), (0, 200)),
                       district="central"),
        GazetteerEntry("s_fitzwilliam_gate", "Fitzwilliam Gate", "street", edge_ids=("e3",),
                       node_ids=("n3", "n4"), coords=((0, 200), (100, 200)), district="central"),
        GazetteerEntry("s_fitzalan_sq", "Fitzalan Square", "street", edge_ids=("e4",),
                       node_ids=("n4", "n5"), coords=((100, 200), (200, 200)), district="central"),
        GazetteerEntry("s_baggot_lower", "Lower Baggot Street", "street", edge_ids=("e5",),
                       node_ids=("n2", "n6"), coords=((0, 100), (100, 100)), district="central"),
        GazetteerEntry("s_baggot_upper", "Baggot Street Upper", "street", edge_ids=("e6",),
                       node_ids=("n6", "n7"), coords=((100, 100), (200, 100)), district="central"),
        GazetteerEntry("s_main_north", "Main Street", "street", edge_ids=("e7",),
                       node_ids=("n9", "n10"), coords=((1000, 0), (1000, 100)), district="north"),
        GazetteerEntry("s_main_south", "Main St", "street", edge_ids=("e8",),
                       node_ids=("n11", "n12"), coords=((-1000, 0), (-1000, 100)),
                       district="south"),
        GazetteerEntry("a_merrion_park", "Merrion Square Park", "area",
                       candidate_edge_ids=("c1", "c2"), node_ids=("n4", "n5", "n8"),
                       coords=((100, 200), (200, 200), (150, 300)), aliases=("Merrion Square",)),
    ]
    endpoints = {"e1": ("n1", "n2"), "e2": ("n2", "n3"), "e3": ("n3", "n4"),
                 "e4": ("n4", "n5"), "e5": ("n2", "n6"), "e6": ("n6", "n7"),
                 "e7": ("n9", "n10"), "e8": ("n11", "n12"), "c1": ("n4", "n8"),
                 "c2": ("n5", "n8")}
    coords = {"n1": (0, 0), "n2": (0, 100), "n3": (0, 200), "n4": (100, 200),
              "n5": (200, 200), "n6": (100, 100), "n7": (200, 100), "n8": (150, 300),
              "n9": (1000, 0), "n10": (1000, 100), "n11": (-1000, 0), "n12": (-1000, 100)}
    gaz = Gazetteer(entries, edge_endpoints=endpoints, node_coords=coords,
                    overrides={"the Square": "a_merrion_park"})
    sources = {"tro-2024": _EXAMPLE_TRO, "amd-2025": _EXAMPLE_AMENDMENT}

    def clause(cid: str, doc: str, page: int, quote: str, kind: str, name: str,
               **kw: Any) -> PolicyClause:
        rul = kw.pop("rulings")
        return PolicyClause(clause_id=cid, document_id=doc, page=page, quote=quote,
                            target_kind=kind, target_name=name, rulings=make_rulings(**rul), **kw)

    clauses = [
        clause("A03", "tro-2024", 1, "No new road shall be constructed through Merrion Square Park.",
               "area", "Merrion Square Park", rulings={"ADD_EDGE": "PROHIBIT"}),
        clause("A04", "tro-2024", 1, "Fitzwilliam Street shall not be closed to through traffic.",
               "street", "Fitzwilliam Street", rulings={"REMOVE_EDGE": "PROHIBIT"}),
        clause("A05", "tro-2024", 1, "A signal-controlled pedestrian crossing must be provided at "
               "the junction of Fitzwilliam Street and Lower Baggot Street.", "junction",
               "Fitzwilliam Street", target_name_b="Lower Baggot Street",
               rulings={"ADD_CROSSING": "REQUIRE"}, objective="ped"),
        clause("A06", "tro-2024", 2, "A roundabout shall not be constructed at the junction of "
               "Fitzwilliam Street and Lower Baggot Street.", "junction", "Fitzwilliam Street",
               target_name_b="Lower Baggot Street", rulings={"JUNCTION_REDESIGN": "PROHIBIT"},
               junction_types=("roundabout",)),
        clause("A07", "tro-2024", 2, "The footway on Lower Baggot Street must be not less than "
               "2.5 metres and not more than 4.0 metres wide.", "street", "Lower Baggot Street",
               rulings={"REPARTITION": "REQUIRE"}, bounds={"footway_width": (2.5, 4.0)},
               instrument_id="TRO-2024"),
        clause("A07a", "amd-2025", 1, "the footway on Lower Baggot Street must be not less than "
               "3.0 metres and not more than 4.0 metres wide.", "street", "Lower Baggot Street",
               rulings={"REPARTITION": "REQUIRE"}, bounds={"footway_width": (3.0, 4.0)},
               instrument_id="AMD-2025"),
        clause("A08", "tro-2024", 2, "Loading is prohibited on Main Street 07:00-10:00 Mon-Fri.",
               "street", "Main Street", rulings={"REPARTITION": "REQUIRE"},
               bounds={"loading": (0.0, 0.0)},
               predicate=parse_predicate("07:00-10:00 Mon-Fri")),
        clause("A09", "tro-2024", 2, "Fitzwilliam Gate shall be closed to vehicles 22:00-06:00 "
               "except emergency vehicles.", "street", "Fitzwilliam Gate",
               rulings={"REMOVE_EDGE": "REQUIRE"},
               predicate=parse_predicate("22:00-06:00 except emergency")),
        clause("A10", "tro-2024", 3, "The Council should prioritise pedestrian movement "
               "throughout the area.", "area", "Merrion Square Park", rulings={},
               strength="soft", objective="ped"),
        clause("A11", "tro-2024", 3, "Cycle lanes must be provided on Pembroke Road.",
               "street", "Pembroke Road", rulings={"ADD_EDGE": "REQUIRE"}),
    ]
    instruments = [
        Instrument("TRO-2019", "2019-01-01", clause_ids=("OLD-01",)),
        Instrument("TRO-2024", "2024-03-01", supersedes=("TRO-2019",),
                   clause_ids=("A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11")),
        Instrument("AMD-2025", "2025-01-01", amends=("TRO-2024",), clause_ids=("A07a",),
                   clause_changes={"A07": "A07a"}),
    ]
    baseline_y = {"footway_width:e5": 3.0}
    baseline_junctions = {"n2": "priority"}
    space = DesignSpace(["e1", "e2", "e5", "c1", "c2"], baseline_edges=["e1", "e2", "e5"],
                        continuous={"footway_width:e5": (2.0, 6.0)},
                        crossing_nodes=["n2", "n3"], junction_nodes=["n2"],
                        baseline_y=baseline_y, baseline_junctions=baseline_junctions)
    gold = {"A03": "a_merrion_park", "A04": "s_fitzwilliam_st", "A05": "n2", "A06": "n2",
            "A07a": "s_baggot_lower", "A08": "s_main_north", "A09": "s_fitzwilliam_gate"}
    return ExampleDistrict(gaz, sources, clauses, instruments, "2025-06-01", space,
                           baseline_y, baseline_junctions, gold)


# ===========================================================================
# 21. DEMO
# ===========================================================================
def _demo() -> None:
    print("=" * 70 + "\n  POLICY COMPILER (pure Python) - demo\n" + "=" * 70)
    fs, fg = "Fitzwilliam Street", "Fitzwilliam Gate"
    cmp_ = compare_strings(fs, fg)
    print("\n-- 1. string metrics (core similarity: %s) --" % _HAVE_CORE_SIMILARITY)
    print("   lev / damerau / soundex / metaphone / JW / jaro:", levenshtein("kitten", "sitting"),
          damerau_levenshtein("CA", "AC"), soundex("Robert"), soundex("Rupert"),
          metaphone("Thompson"), round(jaro_winkler("MARTHA", "MARHTA"), 6),
          jaro_winkler("DWAYNE", "DUANE"), round(jaro("MARTHA", "MARHTA"), 6))
    print("   Street vs Gate (the trap) JW / lev sim / norm dist / jaccard / lcs / R-O:",
          ["%.4f" % v for v in (cmp_.jaro_winkler, levenshtein_similarity(fs, fg),
                                levenshtein_normalised_distance(fs, fg), jaccard_tokens(fs, fg),
                                lcs_similarity(fs, fg), ratcliff_obershelp(fs, fg))],
          sorted(cmp_.as_dict())[:3], STRING_METRICS,
          ["%.3f" % string_similarity("Baggot St", "Bagot St", m) for m in STRING_METRICS])
    ex = build_example_district()
    src = ex.sources["tro-2024"]
    print("\n-- 2. text normalisation and span verification --")
    print("  ", normalise_text("St. Stephen's Green, NE side (Twenty-First Jct.)"),
          normalise_whitespace("  a \n b  "), ordinal_suffix(23), compass_bearing("southbound"),
          len(ABBREVIATIONS), len(COMPASS_POINTS), SPAN_STATUSES)
    print("   spans", [(v.status, v.start, v.end, round(v.score, 3)) for v in (
        verify_span(src, q) for q in ("Fitzwilliam Street shall not be closed to through traffic.",
                                      "Loading is prohibitted on Main Street 07:00-10:00 Mon-Fri.",
                                      "Cycle lanes."))],
          find_all_occurrences(src, "Lower Baggot Street"))
    print("\n-- 3. typed schema and JSON-schema-lite --")
    d = ex.clauses[0].to_dict()
    check_schema(CLAUSE_SCHEMA)
    bad = validate_json(dict(d, rulings={"ADD_EDGE": "MAYBE"}), CLAUSE_SCHEMA)
    print("   A03 valid / round trip / bad clause:", is_valid_json(d, CLAUSE_SCHEMA),
          PolicyClause.from_dict(d).to_dict() == d, bad[0], len(OPERATIONS), RULINGS, ABSTAIN)
    print("   schema keys", [len(x["properties"]) for x in (PREDICATE_SCHEMA, HINT_SCHEMA,
                                                         FIXINGS_SCHEMA)],
          ex.clauses[8].abstains_everywhere, TARGET_KINDS, OBJECTIVES, STRENGTHS)
    print("\n-- 4. street-name resolver (the load-bearing constraint) --")
    gaz = ex.gazetteer
    names = [(e.entry_id, e.name) for e in gaz.entries("street")]
    only_wrong = [n for n in names if n[0] != "s_fitzwilliam_st"]
    print("  ", [(q, resolve_street_name(q, names).entry_id, split_street_name(q).canonical())
                 for q in ("Fitzwilliam Street", "Upper Baggot Street", "Main Street")],
          "without Fitzwilliam St:", resolve_street_name(fs, only_wrong).status,
          thoroughfare_class("Sq"), len(THOROUGHFARE_TYPES), STREET_QUALIFIERS[:2], STREET_STATUSES)
    print("   candidates", [(c.entry_id, round(c.score, 4), c.sources)
                         for c in gaz.enumerate_candidates("Merrion Square")],
          gaz.resolve("the Square", kind="area").target_id, ENTRY_KINDS, BINDING_STATUSES)
    print("   bm25", ["%.2f" % v for v in bm25_scores("baggot st", [e.name for e in gaz.entries()])],
          phrase_match_counts("baggot street", ["Lower Baggot Street", "Street Baggot"]),
          rrf_fuse([["a", "b"], ["b", "c"]]))
    print("\n-- 5. geometric disambiguation --")
    b = gaz.resolve("Main Street", hint=GeometricHint(point=(990.0, 50.0), max_distance=200.0))
    dis = disambiguate(["s_main_north", "s_main_south"], gaz, GeometricHint(district="south"))
    cc = chain_contiguity(["e1", "e2", "e5"], gaz.edge_endpoints)
    print("   Main Street near (990, 50) ->", b.target_id, b.reason, "; south ->", dis.selected,
          bearing_deg((0, 0), (1, 1)), angular_difference(350, 10), axial_difference(0, 170))
    print("   chain e1,e2,e5:", cc.is_contiguous, cc.is_simple_path, cc.gaps,
          point_segment_distance((1, 1), (0, 0), (2, 0)),
          point_polyline_distance((5, 5), [(0, 0), (10, 0)]),
          polyline_length([(0, 0), (3, 4), (6, 8)]), polyline_bearing([(0, 0), (0, -5)]),
          gaz.resolve_junction(fs, "Lower Baggot Street").target_id)
    print("\n-- 6. ambiguity scoring --\n  ", ambiguity_score([0.9, 0.3]), ambiguity_score([1, 1]),
          normalised_entropy([1, 1, 1, 1]), len(ABSTENTION_REASONS))
    print("\n-- 7. predicates and in-force resolution --")
    pr = parse_predicate("Mon-Fri 7am-10am except buses")
    print("  ", pr.to_dict(), pr.holds(Scenario(minute=480, day="tue", vehicle_class="car")),
          pr.holds(Scenario(480, "tue", "bus")), parse_time_window("22:00-06:00").segments(),
          pr.is_satisfiable(), len(DAYS), len(VEHICLE_CLASSES), TimeWindow(1320, 360).duration())
    st = resolve_in_force(ex.instruments, ex.at)
    print("   in force on", ex.at, st.in_force, st.superseded, st.clauses,
          topological_order(ex.instruments), amendment_edges(ex.instruments),
          find_amendment_cycle(ex.instruments))
    print("   Allen", allen_relation(Interval(0, 2), Interval(1, 3)),
          validity_relation(ex.instruments, "TRO-2019", "TRO-2024"), allen_inverse("during"),
          allen_composition("overlaps", "overlaps"), len(ALLEN_RELATIONS),
          intervals_overlap(Interval(0, 1), Interval(1, 2)), date_ordinal("2025-06-01"))
    print("\n-- 8. directional wording -> weight prior --")
    pri = directional_priors(ex.clauses)
    print("  ", [(classify_wording(t).modality, classify_wording(t).is_hard) for t in (
        "shall not be closed", "Save as provided, no person shall wait", "may be considered")],
          pri, weight_prior({"car": 1.0, "ped": 1.0, "cost": 1.0}, pri, eta=0.5),
          len(WORDING_CUES), len(MODALITIES))
    print("\n-- 9. compile the district --")
    res = compile_policy(ex.clauses, gaz, ex.sources, at=ex.at, instruments=ex.instruments,
                         baseline_y=ex.baseline_y, baseline_junctions=ex.baseline_junctions)
    fxd = res.fixings.to_dict()
    print("  ", fxd, "\n   round trip:", CompiledFixings.from_dict(fxd).to_dict() == fxd,
          res.skipped, [(i.clause_id, i.reason, i.candidates) for i in res.queue.items()])
    park = gaz.resolve("Merrion Square Park", kind="area")
    one, issues = compile_clause(ex.clauses[0], park, gaz)
    q2 = AbstentionQueue()
    q2.add(AbstentionItem("X1", "binding_ambiguous", 1.0, ("s_main_north", "s_x"), "q", 1, "d"))
    print("  ", JUNCTION_TYPES, res.abstention_rate(), one.fixed_zero, issues,
          target_elements(park, gaz).candidate_edges, q2.resolve("X1", "s_main_north"))
    print("\n-- 10. conflict detection --")
    clash = PolicyClause("B01", "consult", 1, "footway at least 4.5 metres", "street",
                         "Lower Baggot Street", make_rulings(REPARTITION="REQUIRE"),
                         bounds={"footway_width": (4.5, 6.0)})
    res2 = compile_policy(ex.clauses + [clash], gaz,
                          dict(ex.sources, consult="The footway at least 4.5 metres wide."),
                          at=ex.at, instruments=ex.instruments, baseline_y=ex.baseline_y,
                          baseline_junctions=ex.baseline_junctions)
    merged, cfs = merge_fixings(res2.per_clause)
    print("  ", [(cf.kind, cf.key, cf.clause_ids) for cf in res2.conflicts], "withheld:",
          "bounds:footway_width:e5" not in merged.provenance, CONFLICT_KINDS,
          [c.key for c in detect_conflicts(res2.per_clause)] == [c.key for c in cfs])
    print("\n-- 11. feasible-set projection and free-variable reduction --")
    sp = ex.space
    fv = free_variables(sp, res.fixings)
    bad_design = sp.baseline_design()
    bad_design.z["c1"] = 1
    pj = project_design(bad_design, res.fixings, sp)
    print("   designs %d -> %d (x%.1f)" % (fv.n_discrete_before, fv.n_discrete_after,
                                         fv.reduction_factor), fv.free_edges, fv.pinned_y,
          violations(bad_design, res.fixings), pj.changes, is_feasible(pj.design, res.fixings))
    print("   generator:", count_designs(sp, y_levels=default_y_levels(sp)),
          count_designs(sp, res.fixings), "c1 ever built:",
          any(dd.z["c1"] == 1 for dd in enumerate_designs(sp, res.fixings)), "night closure:",
          applicable_fixings(res.fixings, Scenario(23 * 60, "sat", "car")).fixed_zero)
    print("\n-- 12. relaxation analysis --")
    unc = dict((k, v) for k, v in res.per_clause.items() if not v.conditional)
    rel = relaxation_analysis(sp, unc, ex.objective, reference=(20.0, 20.0, 40.0))
    ps = pareto_set(sp, res.fixings, ex.objective)
    print("   front %d of %d, HV %.3f" % (len(ps.designs), ps.n_feasible,
                                         hypervolume(ps.objectives, (20.0, 20.0, 40.0))),
          rel.binding_ids, [(e.constraint_id, len(e.added), len(e.removed), round(e.hv_gain, 3))
                            for e in rel.entries],
          dominates((1, 2), (2, 2)), non_dominated_indices([(1, 2), (2, 1), (2, 2)]))
    print("\n-- 13. policy check, provenance and agreement --")
    rep = policy_check(pj.design, res)
    g = build_provenance_graph(res)
    ProvenanceGraph().add_node("doc:x", "document")
    agr = binding_agreement(res, ex.gold_bindings)
    print("  ", [(r.clause_id, r.status) for r in rep.rows], rep.compliance_ratio, CHECK_STATUSES)
    print("  ", " <- ".join(g.citation_chains("fix:fixedOne:e1")[0]), g.verify(), len(g.nodes),
          len(PROVENANCE_NODE_TYPES), (agr.precision, agr.recall, agr.abstention_rate, agr.f1),
          agreement_score({"a": "x", "b": None}, {"a": "x", "b": "y"}).as_dict(),
          cohen_kappa(["a", "a", "b", "b"], ["a", "b", "b", "b"]))
    print("\n-- 14. payloads --")
    payload = build_policy_compiler_payload(res, report=rep, relaxation=rel, provenance=g,
                                            agreement=agr)
    ui = build_a2ui_payload(res, report=rep, relaxation=rel)
    print("  ", sorted(payload), len(json.dumps(payload, allow_nan=False, sort_keys=True)),
          len(json.dumps(ui, allow_nan=False)), _stable_digest(payload)[:12])
    print("\n-- 15. input validation --")
    for bad_call in (lambda: levenshtein("a", 3), lambda: jaro_winkler("a", "b", p=0.3),
                     lambda: TimeWindow(600, 600), lambda: parse_predicate("after 7pm"),
                     lambda: weight_prior({"car": 1.0}, {"car": float("nan")}),
                     lambda: gaz.add_override("Nowhere Lane", "s_does_not_exist"),
                     lambda: topological_order([Instrument("X", 1, amends=("Y",)),
                                                Instrument("Y", 2, amends=("X",))])):
        try:
            bad_call()
            print("   ERROR: no exception raised")
        except PolicyCompilerError as exc:
            print("   PolicyCompilerError: %s" % exc)
    print("\n" + "=" * 70 + "\n  demo OK\n" + "=" * 70)


if __name__ == "__main__":
    _demo()
