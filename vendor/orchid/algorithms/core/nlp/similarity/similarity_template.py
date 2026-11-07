"""
algorithms/core/nlp/similarity/similarity_template.py

Pure-Python, zero-dependency NLP similarity toolkit.

Implements twelve similarity / distance strategies from scratch:
  LevenshteinSimilarity    - edit distance (insertions, deletions, substitutions)
  DamerauLevenshtein       - edit distance + transpositions (adjacent swap)
  HammingSimilarity        - position-wise character differences (equal-length strings)
  JaroSimilarity           - character match + transposition ratio
  JaroWinklerSimilarity    - Jaro boosted for common prefix
  JaccardSimilarity        - set intersection / union on tokens
  DiceSimilarity           - 2|A∩B| / (|A|+|B|) on token sets
  OverlapSimilarity        - |A∩B| / min(|A|,|B|) on token sets
  CosineSimilarity         - TF-weighted bag-of-words cosine
  LCSSimilarity            - longest common subsequence ratio
  NgramSimilarity          - character n-gram overlap (default trigrams)
  RatcliffObershelpSimilarity - gestalt pattern matching (recursion on matching blocks)

Plus:
  SoundexSimilarity        - phonetic bucket equality (Soundex coding)
  MetaphoneSimilarity      - simplified phonetic similarity
  BM25Similarity           - Okapi BM25 scoring (query vs. single document)

Utilities:
  levenshtein_distance     - raw edit distance (int)
  lcs_length               - raw LCS length (int)
  soundex_code             - Soundex code for a word
  metaphone_code           - Simplified Metaphone code for a word
  ngram_set                - generate character n-grams
  tokenize                 - whitespace + punctuation tokeniser

Result types:
  SimilarityResult         - rich result object with metadata
  build_similarity_payload / build_a2ui_payload  - frontend-ready JSON

Run as a script for a quick demo:
  python similarity_template.py
"""

from __future__ import annotations

import math
import re
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List

__version__ = "1.0.0"

__all__ = [
    # Result type
    "SimilarityResult",
    # Abstract base
    "BaseSimilarity",
    # String-edit metrics
    "LevenshteinSimilarity",
    "DamerauLevenshtein",
    "HammingSimilarity",
    "JaroSimilarity",
    "JaroWinklerSimilarity",
    # Token-set metrics
    "JaccardSimilarity",
    "DiceSimilarity",
    "OverlapSimilarity",
    "CosineSimilarity",
    # Sequence metrics
    "LCSSimilarity",
    "NgramSimilarity",
    "RatcliffObershelpSimilarity",
    # Phonetic metrics
    "SoundexSimilarity",
    "MetaphoneSimilarity",
    # IR metric
    "BM25Similarity",
    # Raw utilities
    "levenshtein_distance",
    "damerau_levenshtein_distance",
    "hamming_distance",
    "jaro_similarity",
    "lcs_length",
    "soundex_code",
    "metaphone_code",
    "ngram_set",
    "tokenize",
    # Payload builders
    "build_similarity_payload",
    "build_a2ui_payload",
]


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SimilarityResult:
    """Returned by every BaseSimilarity.similarity() call."""
    score: float        # [0, 1], higher = more similar
    distance: float     # [0, ∞), lower = more similar; may be normalised
    strategy: str       # class name
    text_a: str
    text_b: str
    metadata: Dict = field(default_factory=dict)

    @cached_property
    def is_identical(self) -> bool:
        return self.score >= 1.0

    @cached_property
    def is_dissimilar(self) -> bool:
        return self.score <= 0.0

    def __repr__(self) -> str:
        return (
            f"SimilarityResult(strategy={self.strategy!r}, score={self.score:.4f}, "
            f"distance={self.distance:.4f})"
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseSimilarity(ABC):
    """Abstract base for all similarity strategies.

    Subclasses implement `similarity(a, b) -> float` returning a value in [0, 1].
    `distance(a, b)` defaults to `1 - similarity(a, b)` but can be overridden
    when a metric-space distance is well-defined (e.g., normalised edit distance).
    `compare(a, b) -> SimilarityResult` wraps both into a rich result object.
    """

    @abstractmethod
    def similarity(self, a: str, b: str) -> float:
        ...

    def distance(self, a: str, b: str) -> float:
        return 1.0 - self.similarity(a, b)

    def compare(self, a: str, b: str) -> SimilarityResult:
        score = max(0.0, min(1.0, self.similarity(a, b)))
        dist = max(0.0, 1.0 - score)  # derive from clamped score; avoids calling similarity twice
        return SimilarityResult(
            score=score,
            distance=dist,
            strategy=type(self).__name__,
            text_a=a,
            text_b=b,
        )


# ---------------------------------------------------------------------------
# Raw utility functions
# ---------------------------------------------------------------------------

def levenshtein_distance(a: str, b: str) -> int:
    """Classic DP edit distance — O(|a|·|b|) time, O(min(|a|,|b|)) space."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Ensure `a` is the shorter string for the rolling-row optimisation.
    if len(a) > len(b):
        a, b = b, a
    n, m = len(a), len(b)
    prev = list(range(n + 1))
    for j in range(1, m + 1):
        curr = [j] + [0] * n
        for i in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[i] = min(curr[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost)
        prev = curr
    return prev[n]


def damerau_levenshtein_distance(a: str, b: str) -> int:
    """Optimal string alignment distance (includes adjacent transpositions)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    n, m = len(a), len(b)
    # OSA (restricted edit distance) — simpler than full Damerau-Levenshtein
    # but handles the common transposition case correctly.
    d: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,        # deletion
                d[i][j - 1] + 1,        # insertion
                d[i - 1][j - 1] + cost, # substitution
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)  # transposition always costs 1
    return d[n][m]


def hamming_distance(a: str, b: str) -> int:
    """Number of positions where characters differ.  Strings must be equal length."""
    if len(a) != len(b):
        raise ValueError(
            f"Hamming distance requires equal-length strings; got {len(a)} vs {len(b)}."
        )
    return sum(ca != cb for ca, cb in zip(a, b))


def jaro_similarity(a: str, b: str) -> float:
    """Jaro similarity in [0, 1]."""
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
    return (
        matches / len(a)
        + matches / len(b)
        + (matches - transpositions) / matches
    ) / 3.0


def lcs_length(a: str, b: str) -> int:
    """Length of the longest common subsequence (DP, two-row rolling array)."""
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


def ngram_set(text: str, n: int = 3) -> Counter:
    """Return a Counter of character n-grams (padded with '#')."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    padded = "#" * (n - 1) + text + "#" * (n - 1)
    return Counter(padded[i:i + n] for i in range(len(padded) - n + 1))


# ---------------------------------------------------------------------------
# Soundex
# ---------------------------------------------------------------------------

_SOUNDEX_TABLE = str.maketrans(
    "AEHIOUWYBFPVCGJKQSXZDTLMNR",
    "00000000111122222222334556",
)

def soundex_code(word: str) -> str:
    """Return the Soundex code for `word` (e.g. 'Robert' → 'R163').

    Implements American Soundex (U.S. Census rules):
    - Vowels (A E I O U Y) between two consonants of the same code allow BOTH to be coded.
    - H and W between same-code consonants are transparent (treated as adjacent, only coded once).
    """
    word = word.upper()
    if not word:
        return "0000"
    first = word[0]
    result = first
    # Seed prev with the Soundex code of the first letter so that, e.g., "Pfister"
    # (P=1, F=1) correctly drops the F as a duplicate of the opening P.
    prev = first.translate(_SOUNDEX_TABLE) or "0"
    for orig_ch in word[1:]:
        code = orig_ch.translate(_SOUNDEX_TABLE)
        if not code:
            continue
        if code == "0":
            # H and W are transparent — they do NOT reset same-code suppression.
            # Vowels (A E I O U Y) DO reset it, allowing a new sequence to start.
            if orig_ch not in "HW":
                prev = "0"
            continue
        if code != prev:
            result += code
            if len(result) == 4:
                break
        prev = code
    return result.ljust(4, "0")


# ---------------------------------------------------------------------------
# Simplified Metaphone
# ---------------------------------------------------------------------------

def metaphone_code(word: str) -> str:
    """Simplified Metaphone encoding — single consonant-cluster pass.

    This is a lightweight approximation, not full Double Metaphone.
    Covers the most common English phonetic equivalences.
    """
    word = word.upper()
    word = re.sub(r"[^A-Z]", "", word)
    if not word:
        return ""
    # Drop trailing vowel except first position
    vowels = set("AEIOU")
    # Handle initial special cases
    for pair, _ in [
        ("AE", "E"), ("GN", "N"), ("KN", "N"), ("PN", "N"), ("WR", "R"),
    ]:
        if word.startswith(pair):
            word = word[1:]
            break
    # Drop initial vowel
    if word and word[0] in vowels:
        word = word[1:]
    code: List[str] = []
    i = 0
    while i < len(word):
        ch = word[i]
        if ch in vowels:
            i += 1
            continue
        nxt = word[i + 1] if i + 1 < len(word) else ""
        nxt2 = word[i + 2] if i + 2 < len(word) else ""
        if ch == "B":
            code.append("B")
        elif ch == "C":
            if nxt in ("I", "E", "Y"):
                code.append("S")
                if nxt == "I" and nxt2 == "A":
                    code.append("X")
                    i += 1
            elif nxt == "H":
                code.append("X")
                i += 1
            else:
                code.append("K")
        elif ch == "D":
            if nxt == "G" and nxt2 in ("E", "I", "Y"):
                code.append("J")
                i += 1
            else:
                code.append("T")
        elif ch == "F":
            code.append("F")
        elif ch == "G":
            if nxt == "H" and nxt2 not in vowels:
                i += 2
                continue
            if nxt == "N" and (not nxt2 or nxt2 not in "AEIOU"):
                i += 2
                continue
            if nxt in ("E", "I", "Y"):
                code.append("J")
            else:
                code.append("K")
            # GH before a vowel: G is voiced (K), H is silent — skip the H
            if nxt == "H":
                i += 1
        elif ch == "H":
            if nxt in vowels:
                code.append("H")
        elif ch == "J":
            code.append("J")
        elif ch == "K":
            if nxt != "C":
                code.append("K")
        elif ch == "L":
            code.append("L")
        elif ch == "M":
            code.append("M")
        elif ch == "N":
            code.append("N")
        elif ch == "P":
            if nxt == "H":
                code.append("F")
                i += 1
            else:
                code.append("P")
        elif ch == "Q":
            code.append("K")
        elif ch == "R":
            code.append("R")
        elif ch == "S":
            if nxt == "C" and nxt2 == "H":  # SCH → SK (school, scheme)
                code.append("SK")
                i += 2
            elif nxt == "H" or (nxt == "I" and nxt2 in ("O", "A")):
                code.append("X")
                i += 1
            else:
                code.append("S")
        elif ch == "T":
            if nxt == "H":
                code.append("0")
                i += 1
            elif nxt == "I" and nxt2 in ("A", "O"):
                code.append("X")
            else:
                code.append("T")
        elif ch == "V":
            code.append("F")
        elif ch == "W":
            if nxt in vowels:
                code.append("W")
        elif ch == "X":
            code.append("KS")
        elif ch == "Y":
            if nxt in vowels:
                code.append("Y")
        elif ch == "Z":
            code.append("S")
        i += 1
    # Deduplicate adjacent identical codes
    result: List[str] = []
    prev = ""
    for c in code:
        if c != prev:
            result.append(c)
        prev = c
    return "".join(result) or word[0] if word else ""


# ---------------------------------------------------------------------------
# String-edit similarity classes
# ---------------------------------------------------------------------------

class LevenshteinSimilarity(BaseSimilarity):
    """Normalised Levenshtein similarity: 1 - edit_distance / max(|a|, |b|)."""

    def similarity(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 1.0
        return 1.0 - levenshtein_distance(a, b) / max_len

    def distance(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 0.0
        return levenshtein_distance(a, b) / max_len


class DamerauLevenshtein(BaseSimilarity):
    """Normalised OSA (Damerau-Levenshtein) similarity including transpositions."""

    def similarity(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 1.0
        return 1.0 - damerau_levenshtein_distance(a, b) / max_len

    def distance(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 0.0
        return damerau_levenshtein_distance(a, b) / max_len


class HammingSimilarity(BaseSimilarity):
    """Hamming similarity: fraction of matching positions.

    For unequal-length strings, the shorter string is right-padded with NUL
    characters so comparison always succeeds (distance counts padding mismatches).
    """

    def similarity(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        length = max(len(a), len(b))
        if length == 0:
            return 1.0
        a = a.ljust(length, "\x00")
        b = b.ljust(length, "\x00")
        matches = sum(ca == cb for ca, cb in zip(a, b))
        return matches / length

    def distance(self, a: str, b: str) -> float:
        return 1.0 - self.similarity(a, b)


class JaroSimilarity(BaseSimilarity):
    """Jaro similarity — best for short strings / proper names."""

    def similarity(self, a: str, b: str) -> float:
        return jaro_similarity(a, b)


class JaroWinklerSimilarity(BaseSimilarity):
    """Jaro-Winkler similarity — boosts Jaro score for common prefix (up to 4 chars).

    Parameters
    ----------
    p : float
        Scaling factor per matching prefix character.  Standard value is 0.1.
        Must be >= 0.  The combined constraint p * max_prefix_length <= 1.0
        must hold to guarantee the result stays in [0, 1].
    max_prefix_length : int
        Maximum prefix length considered, default 4 (Winkler standard).
    """

    def __init__(self, p: float = 0.1, max_prefix_length: int = 4) -> None:
        if p < 0.0:
            raise ValueError(f"p must be >= 0, got {p!r}.")
        if max_prefix_length < 0:
            raise ValueError(f"max_prefix_length must be >= 0, got {max_prefix_length!r}.")
        if p * max_prefix_length > 1.0:
            raise ValueError(
                f"p * max_prefix_length must be <= 1.0 to keep similarity in [0, 1]; "
                f"got {p} * {max_prefix_length} = {p * max_prefix_length:.4f}."
            )
        self._p = p
        self._max_prefix = max_prefix_length

    def similarity(self, a: str, b: str) -> float:
        jaro = jaro_similarity(a, b)
        if jaro == 0.0:
            return 0.0
        prefix = 0
        for ca, cb in zip(a[:self._max_prefix], b[:self._max_prefix]):
            if ca == cb:
                prefix += 1
            else:
                break
        return min(1.0, jaro + prefix * self._p * (1.0 - jaro))


# ---------------------------------------------------------------------------
# Token-set similarity classes
# ---------------------------------------------------------------------------

class JaccardSimilarity(BaseSimilarity):
    """Jaccard coefficient on token sets: |A ∩ B| / |A ∪ B|."""

    def similarity(self, a: str, b: str) -> float:
        sa = set(tokenize(a))
        sb = set(tokenize(b))
        if not sa and not sb:
            return 1.0
        intersection = len(sa & sb)
        union = len(sa | sb)
        return intersection / union if union else 0.0


class DiceSimilarity(BaseSimilarity):
    """Sørensen-Dice coefficient: 2|A ∩ B| / (|A| + |B|)."""

    def similarity(self, a: str, b: str) -> float:
        sa = set(tokenize(a))
        sb = set(tokenize(b))
        total = len(sa) + len(sb)
        if total == 0:
            return 1.0
        return 2.0 * len(sa & sb) / total


class OverlapSimilarity(BaseSimilarity):
    """Overlap coefficient: |A ∩ B| / min(|A|, |B|)."""

    def similarity(self, a: str, b: str) -> float:
        sa = set(tokenize(a))
        sb = set(tokenize(b))
        if not sa or not sb:
            return 1.0 if not sa and not sb else 0.0
        return len(sa & sb) / min(len(sa), len(sb))


class CosineSimilarity(BaseSimilarity):
    """Cosine similarity on TF-weighted token bags."""

    def similarity(self, a: str, b: str) -> float:
        ca = Counter(tokenize(a))
        cb = Counter(tokenize(b))
        if not ca and not cb:
            return 1.0
        if not ca or not cb:
            return 0.0
        dot = sum(ca[t] * cb[t] for t in ca if t in cb)
        mag_a = math.sqrt(sum(v * v for v in ca.values()))
        mag_b = math.sqrt(sum(v * v for v in cb.values()))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Sequence similarity
# ---------------------------------------------------------------------------

class LCSSimilarity(BaseSimilarity):
    """LCS similarity: 2 * lcs_length(a, b) / (|a| + |b|)."""

    def similarity(self, a: str, b: str) -> float:
        total = len(a) + len(b)
        if total == 0:
            return 1.0
        return 2.0 * lcs_length(a, b) / total


class NgramSimilarity(BaseSimilarity):
    """Character n-gram overlap similarity (default: trigrams).

    Uses the Dice coefficient over n-gram Counter bags:
        2 * Σ min(count_a[g], count_b[g]) / (Σ count_a + Σ count_b)
    """

    def __init__(self, n: int = 3) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n!r}.")
        self._n = n

    def similarity(self, a: str, b: str) -> float:
        ca = ngram_set(a, self._n)
        cb = ngram_set(b, self._n)
        total_a = sum(ca.values())
        total_b = sum(cb.values())
        total = total_a + total_b
        if total == 0:
            return 1.0
        shared = sum(min(ca[g], cb[g]) for g in ca if g in cb)
        return 2.0 * shared / total


class RatcliffObershelpSimilarity(BaseSimilarity):
    """Gestalt pattern matching (Ratcliff/Obershelp algorithm).

    Score = 2 * total_matching_chars / (|a| + |b|).
    Recursively finds the longest common substring, then recurses on both
    non-matching flanks until no matching substrings remain.
    """

    def _matching_chars(self, a: str, b: str) -> int:
        if not a or not b:
            return 0
        # Find longest common substring
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
        return (
            best_len
            + self._matching_chars(a[:best_ai], b[:best_bi])
            + self._matching_chars(a[best_ai + best_len:], b[best_bi + best_len:])
        )

    def similarity(self, a: str, b: str) -> float:
        total = len(a) + len(b)
        if total == 0:
            return 1.0
        return 2.0 * self._matching_chars(a, b) / total


# ---------------------------------------------------------------------------
# Phonetic similarity
# ---------------------------------------------------------------------------

class SoundexSimilarity(BaseSimilarity):
    """Soundex code equality: 1.0 if codes match, 0.0 otherwise.

    For multi-word strings, the first token is used (standard Soundex behaviour).
    """

    def similarity(self, a: str, b: str) -> float:
        ta = tokenize(a)
        tb = tokenize(b)
        wa = ta[0] if ta else ""
        wb = tb[0] if tb else ""
        if not wa and not wb:
            return 1.0
        if not wa or not wb:
            return 0.0
        return 1.0 if soundex_code(wa) == soundex_code(wb) else 0.0

    def compare(self, a: str, b: str) -> SimilarityResult:
        ta = tokenize(a)
        tb = tokenize(b)
        wa = ta[0] if ta else ""
        wb = tb[0] if tb else ""
        ca = soundex_code(wa) if wa else ""
        cb = soundex_code(wb) if wb else ""
        if not wa and not wb:
            score = 1.0
        elif not wa or not wb:
            score = 0.0
        else:
            score = 1.0 if ca == cb else 0.0
        return SimilarityResult(
            score=score,
            distance=1.0 - score,
            strategy=type(self).__name__,
            text_a=a,
            text_b=b,
            metadata={"soundex_a": ca, "soundex_b": cb},
        )


class MetaphoneSimilarity(BaseSimilarity):
    """Simplified Metaphone code equality.

    Multi-word strings: first token used.
    """

    def similarity(self, a: str, b: str) -> float:
        ta = tokenize(a)
        tb = tokenize(b)
        wa = ta[0] if ta else ""
        wb = tb[0] if tb else ""
        if not wa and not wb:
            return 1.0
        if not wa or not wb:
            return 0.0
        ca = metaphone_code(wa)
        cb = metaphone_code(wb)
        if not ca and not cb:
            return 1.0
        return 1.0 if ca == cb else 0.0

    def compare(self, a: str, b: str) -> SimilarityResult:
        ta = tokenize(a)
        tb = tokenize(b)
        wa = ta[0] if ta else ""
        wb = tb[0] if tb else ""
        ca = metaphone_code(wa) if wa else ""
        cb = metaphone_code(wb) if wb else ""
        if not wa and not wb:
            score = 1.0
        elif not wa or not wb:
            score = 0.0
        elif not ca and not cb:
            score = 1.0
        else:
            score = 1.0 if ca == cb else 0.0
        return SimilarityResult(
            score=score,
            distance=1.0 - score,
            strategy=type(self).__name__,
            text_a=a,
            text_b=b,
            metadata={"metaphone_a": ca, "metaphone_b": cb},
        )


# ---------------------------------------------------------------------------
# BM25 similarity (query vs. single document)
# ---------------------------------------------------------------------------

class BM25Similarity(BaseSimilarity):
    """Okapi BM25 relevance score (query vs. single document string).

    Returns a normalised score in [0, 1] by dividing by an approximate maximum
    attainable score (all query terms present once each, document length = avgdl).
    Use as a relative ranking metric; the normalised value is approximate.

    Parameters
    ----------
    k1 : float  Term saturation parameter, default 1.5.
    b  : float  Length normalisation parameter, default 0.75.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0.0:
            raise ValueError(f"k1 must be >= 0, got {k1!r}.")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be in [0, 1], got {b!r}.")
        self._k1 = k1
        self._b = b

    def _bm25_score(self, query_tokens: List[str], doc_tokens: List[str], avgdl: float) -> float:
        doc_len = len(doc_tokens)
        doc_freq = Counter(doc_tokens)
        score = 0.0
        for qt in set(query_tokens):  # deduplicate: each query term counted once
            tf = doc_freq.get(qt, 0)
            if tf == 0:
                continue
            # IDF approximation for single-document use: log((N+1)/(df+0.5)) with N=df=1
            # = log(2/1.5) ≈ 0.288.  We use log(2.0) ≈ 0.693 instead so the maximum
            # attainable score is non-trivially positive and the normalised result stays [0,1].
            idf = math.log(2.0)
            tf_norm = (tf * (self._k1 + 1)) / (
                tf + self._k1 * (1 - self._b + self._b * doc_len / max(avgdl, 1))
            )
            score += idf * tf_norm
        return score

    def similarity(self, a: str, b: str) -> float:
        """Treat `a` as the query and `b` as the document."""
        qt = tokenize(a)
        dt = tokenize(b)
        if not qt and not dt:
            return 1.0
        if not qt or not dt:
            return 0.0
        raw = self._bm25_score(qt, dt, avgdl=float(len(dt)))
        # Approximate max: every query term present once, avgdl = doc_len
        max_score = len(set(qt)) * math.log(2.0) * (self._k1 + 1) / (
            1 + self._k1 * (1 - self._b + self._b)
        )
        if max_score <= 0.0:
            return 0.0
        return min(1.0, raw / max_score)

    def distance(self, a: str, b: str) -> float:
        return 1.0 - self.similarity(a, b)


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def build_similarity_payload(result: SimilarityResult) -> Dict:
    """Return a JSON-serialisable dict for API / frontend consumption."""
    return {
        "strategy": result.strategy,
        "score": round(result.score, 6),
        "distance": round(result.distance, 6),
        "isIdentical": result.is_identical,
        "isDissimilar": result.is_dissimilar,
        "textA": result.text_a,
        "textB": result.text_b,
        "metadata": result.metadata,
        "summary": _build_summary(result),
    }


def _build_summary(result: SimilarityResult) -> str:
    pct = result.score * 100
    if result.is_identical:
        label = "identical"
    elif pct >= 80:
        label = "highly similar"
    elif pct >= 50:
        label = "moderately similar"
    elif pct >= 20:
        label = "weakly similar"
    else:
        label = "dissimilar"
    return (
        f"{result.strategy}: '{result.text_a[:40]}' vs '{result.text_b[:40]}' "
        f"-> {pct:.1f}% ({label})"
    )


def build_a2ui_payload(result: SimilarityResult) -> Dict:
    """Return a UI-component-ready payload (4 StatCards + SimilarityCard)."""
    return {
        "statCards": [
            {"label": "Strategy",  "value": result.strategy,            "unit": ""},
            {"label": "Score",     "value": round(result.score, 4),     "unit": ""},
            {"label": "Distance",  "value": round(result.distance, 4),  "unit": ""},
            {"label": "Identical", "value": str(result.is_identical),   "unit": ""},
        ],
        "similarityCard": {
            "textA":    result.text_a,
            "textB":    result.text_b,
            "score":    round(result.score, 6),
            "distance": round(result.distance, 6),
            "summary":  _build_summary(result),
        },
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    pairs = [
        ("kitten",        "sitting"),
        ("Saturday",      "Sunday"),
        ("the cat sat",   "the cat sat on the mat"),
        ("colour",        "color"),
        ("Robert",        "Rupert"),
        ("",              ""),
        ("hello world",   ""),
        ("fuzzy wuzzy",   "fuzzy wuzzy was a bear"),
    ]

    strategies: List[BaseSimilarity] = [
        LevenshteinSimilarity(),
        DamerauLevenshtein(),
        HammingSimilarity(),
        JaroSimilarity(),
        JaroWinklerSimilarity(),
        JaccardSimilarity(),
        DiceSimilarity(),
        OverlapSimilarity(),
        CosineSimilarity(),
        LCSSimilarity(),
        NgramSimilarity(n=3),
        RatcliffObershelpSimilarity(),
        SoundexSimilarity(),
        MetaphoneSimilarity(),
        BM25Similarity(),
    ]

    print("=" * 72)
    print(f"  Similarity Template  v{__version__}")
    print("=" * 72)

    for a, b in pairs:
        print(f"\n  '{a}'  vs  '{b}'")
        print(f"  {'Strategy':<30}  {'Score':>7}  {'Distance':>8}")
        print(f"  {'-' * 50}")
        for strat in strategies:
            try:
                r = strat.compare(a, b)
                print(f"  {r.strategy:<30}  {r.score:>7.4f}  {r.distance:>8.4f}")
            except Exception as exc:
                print(f"  {type(strat).__name__:<30}  {'N/A':>7}  ({exc})")

    # Payload demo
    r = LevenshteinSimilarity().compare("kitten", "sitting")
    print("\n\n  build_similarity_payload →")
    payload = build_similarity_payload(r)
    for k, v in payload.items():
        print(f"    {k}: {v!r}")

    print("\n  build_a2ui_payload →")
    a2ui = build_a2ui_payload(r)
    for card in a2ui["statCards"]:
        print(f"    StatCard: {card}")
    print(f"    similarityCard: {a2ui['similarityCard']}")

    # Utility demos
    print("\n\n  Utilities:")
    print(f"    soundex_code('Robert')    = {soundex_code('Robert')!r}")
    print(f"    soundex_code('Rupert')    = {soundex_code('Rupert')!r}")
    print(f"    metaphone_code('Knight')  = {metaphone_code('Knight')!r}")
    print(f"    metaphone_code('Night')   = {metaphone_code('Night')!r}")
    print(f"    levenshtein_distance('kitten','sitting') = {levenshtein_distance('kitten','sitting')}")
    print(f"    lcs_length('ABCBDAB','BDCABA')           = {lcs_length('ABCBDAB','BDCABA')}")
    print(f"    ngram_set('abc', 2)                      = {dict(ngram_set('abc', 2))}")
    print("=" * 72)


if __name__ == "__main__":
    _demo()
