"""
algorithms/core/nlp/search/search_template.py

Pure-Python, zero-dependency search engine toolkit.

Implements nine search strategies and supporting infrastructure from scratch:

  InvertedIndex          - postings-list full-text index; TF-IDF scoring, phrase
                           search, and boolean AND / OR / NOT retrieval
  BM25SearchEngine       - Okapi BM25 and BM25+ relevance ranking with configurable
                           k1, b, and delta parameters; sparse candidate optimisation
  TFIDFSearchEngine      - TF-IDF cosine similarity retrieval; smooth / standard IDF,
                           raw / log / freq / bool TF, and L2 normalisation
  FuzzySearchEngine      - edit-distance approximate string search; expands each query
                           term to all vocabulary terms within max_edits and re-scores
                           by weighted term frequency
  TrieSearch             - trie-based prefix search and frequency-ranked autocomplete;
                           O(|prefix|) lookup, configurable exact / prefix-final mode
  LinearVectorSearch     - brute-force exact nearest-neighbour (cosine, Euclidean, dot);
                           guaranteed optimal top-k; use as ANN baseline
  LSHSearch              - Locality Sensitive Hashing approximate nearest-neighbour via
                           random hyperplane projection; configurable tables / hash bits;
                           exact re-ranking of candidate set
  KDTreeSearch           - KD-tree exact nearest-neighbour; O(log n) average query time;
                           recommended for dim ≤ 20
  MultiSearchEngine      - ensemble score fusion (Reciprocal Rank Fusion or weighted
                           linear combination) over any set of text search engines

Plus:
  QueryNode / QueryParser  - recursive-descent boolean query parser supporting
                             AND / OR / NOT and parenthesised sub-expressions
  RRFRanker                - standalone Reciprocal Rank Fusion over multiple ranked lists
  ScoreNormaliser          - min-max, z-score, softmax, and rank-reciprocal normalisation
  SearchHit                - rich per-document result object with score and metadata
  SearchResult             - ordered collection of SearchHit objects with engine metadata
  build_search_payload / build_a2ui_payload  - frontend-ready JSON

Run as a script for a quick demo:
  python search_template.py
"""

from __future__ import annotations

import heapq
import math
import random
import re
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import (
    Any, Dict, Iterable, List, Optional,
    Sequence, Set, Tuple, Union,
)

__version__ = "1.0.0"

__all__ = [
    # Result types
    "SearchHit",
    "SearchResult",
    # Query parsing
    "QueryNode",
    "QueryParser",
    # Abstract bases
    "BaseSearchEngine",
    "BaseVectorSearchEngine",
    # Text search engines
    "InvertedIndex",
    "BM25SearchEngine",
    "TFIDFSearchEngine",
    "FuzzySearchEngine",
    "TrieSearch",
    "MultiSearchEngine",
    # Vector search engines
    "LinearVectorSearch",
    "LSHSearch",
    "KDTreeSearch",
    # Ranking / normalisation utilities
    "RRFRanker",
    "ScoreNormaliser",
    # Payload builders
    "build_search_payload",
    "build_a2ui_payload",
]


# ════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════════════════

_DEFAULT_TOKEN_PATTERN = r"(?u)\b\w+\b"
_TOKEN_RE = re.compile(_DEFAULT_TOKEN_PATTERN)


def _tokenize(
    text: str,
    *,
    lowercase: bool = True,
    pattern: "re.Pattern[str]" = _TOKEN_RE,
) -> List[str]:
    text = unicodedata.normalize("NFC", text)
    tokens = pattern.findall(text)
    return [t.lower() for t in tokens] if lowercase else tokens


def _levenshtein(a: str, b: str) -> int:
    """Two-row rolling DP edit distance — O(|a|·|b|) time, O(min(|a|,|b|)) space."""
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
        for i in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[i] = min(curr[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost)
        prev = curr
    return prev[n]


def _cosine(u: Sequence[float], v: Sequence[float]) -> float:
    dot  = sum(a * b for a, b in zip(u, v))
    nu   = math.sqrt(sum(a * a for a in u))
    nv   = math.sqrt(sum(b * b for b in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)


def _euclidean(u: Sequence[float], v: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))


def _l2_normalize(v: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0.0:
        return list(v)
    return [x / norm for x in v]


# ════════════════════════════════════════════════════════════════════════════
# Result types
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SearchHit:
    """A single document result returned by a search engine.

    Attributes
    ----------
    doc_id   : Zero-based document index within the indexed corpus.
    score    : Relevance score; higher is always more relevant.
    document : Original document text if store_docs=True was set on the engine.
    metadata : Engine-specific extras (expanded terms, distances, BM25 stats, etc.).
    """
    doc_id:   int
    score:    float
    document: Optional[str]             = None
    metadata: Dict[str, Any]            = field(default_factory=dict)

    def __lt__(self, other: "SearchHit") -> bool:
        return self.score < other.score

    def __repr__(self) -> str:
        snippet = (self.document or "")[:40].replace("\n", " ")
        return (
            f"SearchHit(doc_id={self.doc_id}, score={self.score:.4f}, "
            f"doc={snippet!r})"
        )


@dataclass
class SearchResult:
    """Ordered collection of hits produced by a single search engine call.

    Attributes
    ----------
    query               : The original query string (or vector repr for ANN).
    hits                : Ranked list of SearchHit objects (highest score first).
    engine              : Class name of the producing search engine.
    total_docs_searched : Number of documents evaluated (may be < corpus for ANN).
    metadata            : Engine-level extras (index stats, fusion mode, etc.).
    """
    query:               str
    hits:                List[SearchHit]
    engine:              str
    total_docs_searched: int
    metadata:            Dict[str, Any] = field(default_factory=dict)

    @cached_property
    def n_hits(self) -> int:
        return len(self.hits)

    @cached_property
    def top_hit(self) -> Optional[SearchHit]:
        return self.hits[0] if self.hits else None

    @cached_property
    def scores(self) -> List[float]:
        return [h.score for h in self.hits]

    def get_doc_ids(self) -> List[int]:
        return [h.doc_id for h in self.hits]

    def __repr__(self) -> str:
        top = f"{self.top_hit.score:.4f}" if self.top_hit else "N/A"
        return (
            f"SearchResult(engine={self.engine!r}, query={self.query!r}, "
            f"n_hits={self.n_hits}, top_score={top})"
        )


# ════════════════════════════════════════════════════════════════════════════
# Abstract bases
# ════════════════════════════════════════════════════════════════════════════

class BaseSearchEngine(ABC):
    """Abstract base for text-document search engines.

    Subclasses implement index() (learn from a text corpus) and search()
    (return ranked hits for a string query).  fit-then-transform follows the
    same convention as the NLP vectorisers in this kit.
    """

    _fitted: bool = False
    # True for engines that APPEND to an existing index rather than replacing it.
    # HybridSearchEngine enforces _accumulates=False on its text engine so that
    # doc_ids in the text and vector legs always stay aligned.
    _accumulates: bool = False

    @abstractmethod
    def index(self, documents: Iterable[str]) -> "BaseSearchEngine":
        """Index a collection of text documents.  Returns self."""

    @abstractmethod
    def search(self, query: str, *, top_k: int = 10) -> SearchResult:
        """Return the top_k documents most relevant to query."""

    def _require_indexed(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been indexed. "
                "Call index() first."
            )


class BaseVectorSearchEngine(ABC):
    """Abstract base for dense-vector nearest-neighbour search engines."""

    _fitted: bool = False

    @abstractmethod
    def index(
        self,
        vectors:   Iterable[List[float]],
        documents: Optional[Iterable[str]] = None,
    ) -> "BaseVectorSearchEngine":
        """Index a collection of dense vectors.  Returns self."""

    @abstractmethod
    def search_vector(
        self, query: List[float], *, top_k: int = 10
    ) -> SearchResult:
        """Return the top_k vectors nearest to query."""

    def _require_indexed(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been indexed. "
                "Call index() first."
            )


# ════════════════════════════════════════════════════════════════════════════
# QueryParser
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class QueryNode:
    """Node in a parsed boolean query tree.

    op       : "TERM" | "AND" | "OR" | "NOT"
    value    : Term string for TERM nodes; None for operator nodes.
    children : Sub-nodes for AND / OR (two children) and NOT (one child).
    """
    op:       str
    value:    Optional[str]       = None
    children: List["QueryNode"]   = field(default_factory=list)

    def __repr__(self) -> str:
        if self.op == "TERM":
            return f"TERM({self.value!r})"
        return f"{self.op}({', '.join(repr(c) for c in self.children)})"


class QueryParser:
    """Recursive-descent boolean query parser.

    Grammar
    -------
    query    := or_expr
    or_expr  := and_expr  ( 'OR'  and_expr  )*
    and_expr := not_expr  ( ('AND' | implicit) not_expr )*
    not_expr := 'NOT' atom | atom
    atom     := '(' query ')' | TERM

    Operator tokens (case-insensitive): AND, OR, NOT.
    Adjacent terms without an explicit operator are treated as implicit AND.

    Examples
    --------
    >>> qp = QueryParser()
    >>> qp.parse("cat AND dog")
    AND(TERM('cat'), TERM('dog'))
    >>> qp.parse("(cat OR kitten) AND NOT mat")
    AND(OR(TERM('cat'), TERM('kitten')), NOT(TERM('mat')))
    """

    _OPS      = {"AND", "OR", "NOT"}
    _TOKEN_RE = re.compile(r"\(|\)|[A-Za-z0-9_]+")

    def parse(self, query: str) -> QueryNode:
        """Parse a boolean query string into a QueryNode tree."""
        self._tokens: List[str] = self._TOKEN_RE.findall(query)
        self._pos = 0
        if not self._tokens:
            return QueryNode(op="TERM", value="")
        return self._parse_or()

    # ------------------------------------------------------------------
    # Private recursive descent

    def _peek(self) -> Optional[str]:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> str:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _parse_or(self) -> QueryNode:
        left = self._parse_and()
        while self._peek() and self._peek().upper() == "OR":
            self._consume()
            right = self._parse_and()
            left  = QueryNode(op="OR", children=[left, right])
        return left

    def _parse_and(self) -> QueryNode:
        left = self._parse_not()
        while True:
            peek = self._peek()
            if peek is None or peek == ")":
                break
            if peek.upper() == "OR":
                break
            if peek.upper() == "AND":
                self._consume()
            right = self._parse_not()
            left  = QueryNode(op="AND", children=[left, right])
        return left

    def _parse_not(self) -> QueryNode:
        if self._peek() and self._peek().upper() == "NOT":
            self._consume()
            return QueryNode(op="NOT", children=[self._parse_not()])
        return self._parse_atom()

    def _parse_atom(self) -> QueryNode:
        tok = self._peek()
        if tok == "(":
            self._consume()
            node = self._parse_or()
            if self._peek() == ")":
                self._consume()
            return node
        if tok is not None and tok.upper() not in self._OPS:
            self._consume()
            return QueryNode(op="TERM", value=tok.lower())
        return QueryNode(op="TERM", value="")

    # ------------------------------------------------------------------

    def extract_terms(self, node: QueryNode) -> List[str]:
        """Return all TERM values in the tree (including those under NOT)."""
        if node.op == "TERM":
            return [node.value] if node.value else []
        terms: List[str] = []
        for child in node.children:
            terms.extend(self.extract_terms(child))
        return terms


# ════════════════════════════════════════════════════════════════════════════
# InvertedIndex
# ════════════════════════════════════════════════════════════════════════════

class InvertedIndex(BaseSearchEngine):
    """Postings-list inverted index with TF-IDF ranking.

    Core data structure: for each term, a dict mapping doc_id → list of
    character positions.  Supports three retrieval modes:

    search()          — ranked TF-IDF retrieval (log-TF × smooth-IDF)
    phrase_search()   — exact phrase matching via positional intersection
    boolean_search()  — AND / OR / NOT Boolean retrieval using QueryParser

    Parameters
    ----------
    lowercase    : Fold text to lowercase before tokenising.
    store_docs   : Cache original document text in each SearchHit.
    token_pattern: Regex used to extract tokens.

    Note: InvertedIndex *appends* on each index() call (incremental by design).
    Use BM25SearchEngine or TFIDFSearchEngine if you need replace-on-reindex
    semantics (e.g. inside HybridSearchEngine).
    """

    _accumulates: bool = True

    def __init__(
        self,
        *,
        lowercase:     bool = True,
        store_docs:    bool = True,
        token_pattern: str  = _DEFAULT_TOKEN_PATTERN,
    ) -> None:
        self._lowercase  = lowercase
        self._store_docs = store_docs
        self._pattern    = re.compile(token_pattern)
        # term -> {doc_id: [position, ...]}
        self._postings:     Dict[str, Dict[int, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._doc_lengths:  List[int]           = []
        self._documents:    List[Optional[str]] = []
        self._n_docs:       int = 0

    # ------------------------------------------------------------------

    def index(self, documents: Iterable[str]) -> "InvertedIndex":
        docs    = list(documents)
        offset  = self._n_docs
        for i, text in enumerate(docs):
            doc_id = offset + i
            tokens = _tokenize(text, lowercase=self._lowercase, pattern=self._pattern)
            self._doc_lengths.append(len(tokens))
            self._documents.append(text if self._store_docs else None)
            for pos, token in enumerate(tokens):
                self._postings[token][doc_id].append(pos)
        self._n_docs += len(docs)
        self._fitted  = True
        return self

    # ------------------------------------------------------------------
    # IDF / TF helpers

    def _df(self, term: str) -> int:
        return len(self._postings.get(term, {}))

    def _smooth_idf(self, term: str) -> float:
        return math.log((1 + self._n_docs) / (1 + self._df(term))) + 1.0

    # ------------------------------------------------------------------
    # Ranked TF-IDF retrieval

    def search(self, query: str, *, top_k: int = 10) -> SearchResult:
        """Log-TF × smooth-IDF ranked retrieval."""
        self._require_indexed()
        terms = _tokenize(query, lowercase=self._lowercase, pattern=self._pattern)
        if not terms:
            return SearchResult(
                query=query, hits=[], engine="InvertedIndex",
                total_docs_searched=self._n_docs,
            )
        scores: Dict[int, float] = defaultdict(float)
        for term in set(terms):
            idf = self._smooth_idf(term)
            for doc_id, positions in self._postings.get(term, {}).items():
                tf = len(positions)
                scores[doc_id] += (1.0 + math.log(tf)) * idf
        hits = self._topk_hits(scores, top_k)
        return SearchResult(
            query=query,
            hits=hits,
            engine="InvertedIndex",
            total_docs_searched=self._n_docs,
            metadata={"n_matched_docs": len(scores)},
        )

    # ------------------------------------------------------------------
    # Phrase search

    def phrase_search(self, phrase: str, *, top_k: int = 10) -> SearchResult:
        """Exact phrase search using positional intersection."""
        self._require_indexed()
        terms = _tokenize(phrase, lowercase=self._lowercase, pattern=self._pattern)
        if not terms:
            return SearchResult(
                query=phrase, hits=[], engine="InvertedIndex",
                total_docs_searched=self._n_docs,
            )
        # Candidate docs: intersection of posting sets for every term.
        candidate_docs: Optional[Set[int]] = None
        for term in terms:
            posting = set(self._postings.get(term, {}).keys())
            candidate_docs = posting if candidate_docs is None else candidate_docs & posting
        if not candidate_docs:
            return SearchResult(
                query=phrase, hits=[], engine="InvertedIndex",
                total_docs_searched=self._n_docs,
            )
        # Positional check: for each anchor position of terms[0], verify
        # that terms[i] appears at anchor + i for every i > 0.
        matched: Dict[int, float] = {}
        for doc_id in candidate_docs:
            anchors = set(self._postings[terms[0]][doc_id])
            for offset, term in enumerate(terms[1:], start=1):
                pos_set = set(self._postings.get(term, {}).get(doc_id, []))
                anchors = {p for p in anchors if p + offset in pos_set}
                if not anchors:
                    break
            if anchors:
                matched[doc_id] = float(len(anchors))
        hits = self._topk_hits(matched, top_k)
        return SearchResult(
            query=phrase,
            hits=hits,
            engine="InvertedIndex",
            total_docs_searched=self._n_docs,
            metadata={"phrase_match_counts": {str(d): int(s) for d, s in matched.items()}},
        )

    # ------------------------------------------------------------------
    # Boolean retrieval

    def boolean_search(self, query: str, *, top_k: int = 10) -> SearchResult:
        """AND / OR / NOT boolean retrieval with TF-IDF ranking over matched docs."""
        self._require_indexed()
        ast   = QueryParser().parse(query)
        doc_ids = self._eval_node(ast)
        terms   = QueryParser().extract_terms(ast)
        scores: Dict[int, float] = {}
        for doc_id in doc_ids:
            s = 0.0
            for term in terms:
                tf = len(self._postings.get(term, {}).get(doc_id, []))
                if tf > 0:
                    s += (1.0 + math.log(tf)) * self._smooth_idf(term)
            scores[doc_id] = s
        hits = self._topk_hits(scores, top_k)
        return SearchResult(
            query=query,
            hits=hits,
            engine="InvertedIndex",
            total_docs_searched=self._n_docs,
            metadata={"boolean_op": ast.op, "n_matched": len(doc_ids)},
        )

    def _eval_node(self, node: QueryNode) -> Set[int]:
        if node.op == "TERM":
            return set(self._postings.get(node.value or "", {}).keys())
        if node.op == "AND":
            sets = [self._eval_node(c) for c in node.children]
            if not sets:
                return set()
            result = sets[0]
            for s in sets[1:]:
                result &= s
            return result
        if node.op == "OR":
            result_or: Set[int] = set()
            for c in node.children:
                result_or |= self._eval_node(c)
            return result_or
        if node.op == "NOT":
            all_docs = set(range(self._n_docs))
            excluded = self._eval_node(node.children[0]) if node.children else set()
            return all_docs - excluded
        return set()

    # ------------------------------------------------------------------

    def _topk_hits(
        self, scores: Dict[int, float], top_k: int
    ) -> List[SearchHit]:
        if not scores:
            return []
        top = heapq.nlargest(top_k, scores.items(), key=lambda kv: kv[1])
        return [
            SearchHit(
                doc_id=doc_id,
                score=score,
                document=self._documents[doc_id]
                if doc_id < len(self._documents) else None,
            )
            for doc_id, score in top
        ]

    # ------------------------------------------------------------------
    # Introspection

    @property
    def vocabulary(self) -> List[str]:
        return sorted(self._postings.keys())

    @property
    def n_docs(self) -> int:
        return self._n_docs

    def doc_frequency(self, term: str) -> int:
        t = term.lower() if self._lowercase else term
        return self._df(t)


# ════════════════════════════════════════════════════════════════════════════
# BM25SearchEngine
# ════════════════════════════════════════════════════════════════════════════

class BM25SearchEngine(BaseSearchEngine):
    """Okapi BM25 and BM25+ search engine.

    BM25 is the de-facto industry standard for sparse text retrieval.
    BM25+ (Robertson & Zaragoza 2009) adds a floor delta to each matched
    term's contribution, improving recall on short queries and long documents.

    score(D, Q) = Σ_i  IDF(q_i) ×
                  [f(q_i,D)×(k1+1)] / [f(q_i,D) + k1×(1−b + b×|D|/avgdl)]
                  + delta

    IDF(t) = log((N − df(t) + 0.5) / (df(t) + 0.5) + 1)  (floored at epsilon)

    Candidate documents are found via an internal inverted index so that only
    documents containing at least one query term are scored (sparse bypass).

    Parameters
    ----------
    k1      : Term-saturation parameter.  Typical range 1.2–2.0 (default 1.5).
    b       : Length-normalisation weight.  0 = none, 1 = full (default 0.75).
    delta   : BM25+ floor.  0 = standard BM25, 1.0 = BM25+ (default 0.0).
    epsilon : IDF floor to prevent negative scores for ultra-common terms.
    """

    def __init__(
        self,
        *,
        k1:            float = 1.5,
        b:             float = 0.75,
        delta:         float = 0.0,
        epsilon:       float = 0.25,
        lowercase:     bool  = True,
        store_docs:    bool  = True,
        token_pattern: str   = _DEFAULT_TOKEN_PATTERN,
    ) -> None:
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be in [0, 1]; got {b}")
        if k1 < 0.0:
            raise ValueError(f"k1 must be >= 0; got {k1}")
        self._k1      = k1
        self._b       = b
        self._delta   = delta
        self._epsilon = epsilon
        self._lower   = lowercase
        self._keep    = store_docs
        self._pattern = re.compile(token_pattern)
        # Set by index()
        self._corpus_tokens: List[List[str]]    = []
        self._documents:     List[Optional[str]] = []
        self._postings:      Dict[str, Set[int]] = defaultdict(set)
        self._avgdl:         float               = 0.0
        self._idf:           Dict[str, float]    = {}
        self._n_docs:        int                 = 0

    def index(self, documents: Iterable[str]) -> "BM25SearchEngine":
        self._postings.clear()
        docs = list(documents)
        tokenised = [
            _tokenize(d, lowercase=self._lower, pattern=self._pattern) for d in docs
        ]
        self._corpus_tokens = tokenised
        self._documents     = list(docs) if self._keep else [None] * len(docs)
        self._n_docs        = len(docs)
        if self._n_docs == 0:
            self._avgdl = 0.0
            self._idf   = {}
            self._fitted = True
            return self
        self._avgdl = sum(len(t) for t in tokenised) / self._n_docs
        # Document-frequency counter and sparse postings.
        df: Counter = Counter()
        for doc_idx, tokens in enumerate(tokenised):
            unique = set(tokens)
            df.update(unique)
            for tok in unique:
                self._postings[tok].add(doc_idx)
        N = self._n_docs
        self._idf = {
            t: max(self._epsilon, math.log((N - c + 0.5) / (c + 0.5) + 1.0))
            for t, c in df.items()
        }
        self._fitted = True
        return self

    def _score_doc(self, query_tokens: List[str], doc_idx: int) -> float:
        doc_tokens = self._corpus_tokens[doc_idx]
        dl  = len(doc_tokens)
        tf  = Counter(doc_tokens)
        k1, b, avgdl = self._k1, self._b, self._avgdl
        score = 0.0
        for qt in set(query_tokens):
            f_qd = tf.get(qt, 0)
            if f_qd == 0:
                continue
            idf   = self._idf.get(qt, 0.0)
            denom = f_qd + k1 * (1 - b + b * dl / avgdl) if avgdl > 0 else f_qd + k1
            score += idf * (f_qd * (k1 + 1) / denom + self._delta)
        return score

    def search(self, query: str, *, top_k: int = 10) -> SearchResult:
        self._require_indexed()
        q_tokens = _tokenize(query, lowercase=self._lower, pattern=self._pattern)
        if not q_tokens or self._n_docs == 0:
            return SearchResult(
                query=query, hits=[], engine="BM25SearchEngine",
                total_docs_searched=self._n_docs,
            )
        # Sparse bypass: only score docs containing at least one query term.
        candidate_docs: Set[int] = set()
        for qt in set(q_tokens):
            candidate_docs |= self._postings.get(qt, set())
        if not candidate_docs:
            return SearchResult(
                query=query, hits=[], engine="BM25SearchEngine",
                total_docs_searched=self._n_docs,
            )
        scored = [(i, self._score_doc(q_tokens, i)) for i in candidate_docs]
        top    = heapq.nlargest(top_k, scored, key=lambda x: x[1])
        hits   = [
            SearchHit(doc_id=i, score=s, document=self._documents[i])
            for i, s in top
        ]
        return SearchResult(
            query=query,
            hits=hits,
            engine="BM25SearchEngine",
            total_docs_searched=self._n_docs,
            metadata={
                "n_candidates": len(candidate_docs),
                "k1": self._k1,
                "b":  self._b,
                "bm25_plus": self._delta > 0,
            },
        )

    def score_all(self, query: str) -> List[float]:
        """BM25 score for every indexed document (use for external re-ranking)."""
        self._require_indexed()
        q = _tokenize(query, lowercase=self._lower, pattern=self._pattern)
        return [self._score_doc(q, i) for i in range(self._n_docs)]

    @property
    def n_docs(self) -> int:
        return self._n_docs

    @property
    def idf_(self) -> Dict[str, float]:
        self._require_indexed()
        return dict(self._idf)


# ════════════════════════════════════════════════════════════════════════════
# TFIDFSearchEngine
# ════════════════════════════════════════════════════════════════════════════

class TFIDFSearchEngine(BaseSearchEngine):
    """TF-IDF cosine similarity retrieval.

    Each document is stored as an L2-normalised TF-IDF vector built at index
    time.  A query is converted to the same vector space at search time and
    ranked by cosine similarity against all document vectors.

    TF schemes  : "raw" (default) | "log" | "freq" | "bool"
    IDF schemes : "smooth" (default) | "standard"
    Norm        : "l2" (default) | "l1" | None

    Parameters
    ----------
    min_df / max_df : Minimum / maximum document-frequency thresholds.
                      int → absolute count; float → fraction of corpus size.
    sublinear_tf    : Apply 1 + log(tf) before the tf_scheme (sklearn default).
    """

    def __init__(
        self,
        *,
        lowercase:     bool                  = True,
        min_df:        Union[int, float]     = 1,
        max_df:        Union[int, float]     = 1.0,
        max_features:  Optional[int]         = None,
        tf_scheme:     str                   = "raw",
        idf_scheme:    str                   = "smooth",
        sublinear_tf:  bool                  = False,
        norm:          Optional[str]         = "l2",
        store_docs:    bool                  = True,
        token_pattern: str                   = _DEFAULT_TOKEN_PATTERN,
    ) -> None:
        _tf   = {"raw", "log", "freq", "bool"}
        _idf  = {"smooth", "standard"}
        _norm = {None, "l1", "l2"}
        if tf_scheme not in _tf:
            raise ValueError(f"tf_scheme must be one of {_tf}; got {tf_scheme!r}")
        if idf_scheme not in _idf:
            raise ValueError(f"idf_scheme must be one of {_idf}; got {idf_scheme!r}")
        if norm not in _norm:
            raise ValueError(f"norm must be one of {_norm}; got {norm!r}")
        self._lower       = lowercase
        self._min_df      = min_df
        self._max_df      = max_df
        self._max_feat    = max_features
        self._tf_scheme   = tf_scheme
        self._idf_scheme  = idf_scheme
        self._sublinear   = sublinear_tf
        self._norm        = norm
        self._keep        = store_docs
        self._pattern     = re.compile(token_pattern)
        # Set by index()
        self._vocab:      Dict[str, int]     = {}
        self._terms:      List[str]          = []
        self._idf_vec:    List[float]        = []
        self._matrix:     List[List[float]]  = []
        self._documents:  List[Optional[str]] = []
        self._n_docs:     int                = 0

    # ------------------------------------------------------------------

    def _df_bound(self, p: Union[int, float], n: int, *, lo: bool) -> int:
        if isinstance(p, float):
            val = math.floor(p * n)
            return max(1, val) if lo else val
        return int(p)

    def _tf(self, count: int, doc_len: int) -> float:
        c = (1.0 + math.log(count)) if (self._sublinear and count > 0) else float(count)
        if self._tf_scheme == "raw":
            return c
        if self._tf_scheme == "log":
            return 1.0 + math.log(c) if c > 0 else 0.0
        if self._tf_scheme == "freq":
            return c / doc_len if doc_len else 0.0
        return 1.0 if count > 0 else 0.0  # "bool"

    def _idf(self, df: int, n: int) -> float:
        if self._idf_scheme == "smooth":
            return math.log((1 + n) / (1 + df)) + 1.0
        return math.log(n / df) + 1.0 if df > 0 else 0.0

    def index(self, documents: Iterable[str]) -> "TFIDFSearchEngine":
        docs   = list(documents)
        n      = len(docs)
        self._n_docs    = n
        self._documents = list(docs) if self._keep else [None] * n
        if n == 0:
            self._vocab = {}; self._terms = []; self._idf_vec = []; self._matrix = []
            self._fitted = True; return self
        tokenised = [
            _tokenize(d, lowercase=self._lower, pattern=self._pattern) for d in docs
        ]
        df: Counter = Counter()
        for tokens in tokenised:
            df.update(set(tokens))
        lo = self._df_bound(self._min_df, n, lo=True)
        hi = self._df_bound(self._max_df, n, lo=False)
        terms = sorted(
            (t for t, c in df.items() if lo <= c <= hi),
            key=lambda t: (-df[t], t),
        )
        if self._max_feat is not None:
            terms = terms[: self._max_feat]
        self._terms   = terms
        self._vocab   = {t: i for i, t in enumerate(terms)}
        self._idf_vec = [self._idf(df[t], n) for t in terms]
        V = len(terms)
        matrix: List[List[float]] = []
        for tokens in tokenised:
            counts  = Counter(tokens)
            doc_len = len(tokens)
            vec     = [0.0] * V
            for term, cnt in counts.items():
                idx = self._vocab.get(term)
                if idx is not None:
                    vec[idx] = self._tf(cnt, doc_len) * self._idf_vec[idx]
            if self._norm == "l2":
                vec = _l2_normalize(vec)
            elif self._norm == "l1":
                s = sum(abs(x) for x in vec)
                vec = [x / s if s else x for x in vec]
            matrix.append(vec)
        self._matrix  = matrix
        self._fitted  = True
        return self

    def search(self, query: str, *, top_k: int = 10) -> SearchResult:
        self._require_indexed()
        q_tokens = _tokenize(query, lowercase=self._lower, pattern=self._pattern)
        if not q_tokens or not self._vocab:
            return SearchResult(
                query=query, hits=[], engine="TFIDFSearchEngine",
                total_docs_searched=self._n_docs,
            )
        V        = len(self._vocab)
        q_counts = Counter(q_tokens)
        q_len    = len(q_tokens)
        q_vec    = [0.0] * V
        for term, cnt in q_counts.items():
            idx = self._vocab.get(term)
            if idx is not None:
                q_vec[idx] = self._tf(cnt, q_len) * self._idf_vec[idx]
        if self._norm == "l2":
            q_vec = _l2_normalize(q_vec)
        scored = [(i, _cosine(q_vec, dv)) for i, dv in enumerate(self._matrix)]
        top    = heapq.nlargest(top_k, scored, key=lambda x: x[1])
        top    = [(i, s) for i, s in top if s > 0.0]
        hits   = [
            SearchHit(doc_id=i, score=s, document=self._documents[i])
            for i, s in top
        ]
        return SearchResult(
            query=query, hits=hits, engine="TFIDFSearchEngine",
            total_docs_searched=self._n_docs,
            metadata={"vocab_size": V, "tf_scheme": self._tf_scheme},
        )

    @property
    def vocabulary(self) -> Dict[str, int]:
        self._require_indexed(); return dict(self._vocab)

    @property
    def idf_(self) -> Dict[str, float]:
        self._require_indexed(); return dict(zip(self._terms, self._idf_vec))


# ════════════════════════════════════════════════════════════════════════════
# FuzzySearchEngine
# ════════════════════════════════════════════════════════════════════════════

class FuzzySearchEngine(BaseSearchEngine):
    """Edit-distance approximate string search.

    Each query token is expanded to all vocabulary terms within max_edits
    Levenshtein distance.  Expanded terms are weighted by 1/(1+edit_distance)
    so exact matches score 1.0, one-edit matches score 0.5, etc.  Document
    scores accumulate weighted log-TF contributions across all expanded terms.

    A shared-prefix guard (prefix_len) prunes the vocabulary before computing
    full edit distance, keeping expansion fast even for large vocabularies.

    Parameters
    ----------
    max_edits    : Maximum Levenshtein edit distance for expansion (default 1).
    min_term_len : Minimum query-token length to trigger fuzzy expansion;
                   shorter tokens are matched exactly to avoid over-expansion.
    prefix_len   : Require this many leading characters to match before the
                   full edit-distance computation (0 disables the guard).

    Note: FuzzySearchEngine *appends* on each index() call (it wraps an
    InvertedIndex internally).  Use inside HybridSearchEngine with caution —
    see InvertedIndex note.
    """

    _accumulates: bool = True

    def __init__(
        self,
        *,
        max_edits:     int  = 1,
        min_term_len:  int  = 4,
        prefix_len:    int  = 1,
        lowercase:     bool = True,
        store_docs:    bool = True,
        token_pattern: str  = _DEFAULT_TOKEN_PATTERN,
    ) -> None:
        if max_edits < 0:
            raise ValueError(f"max_edits must be >= 0; got {max_edits}")
        self._max_edits    = max_edits
        self._min_term_len = min_term_len
        self._prefix_len   = prefix_len
        self._lower        = lowercase
        self._keep         = store_docs
        self._pattern      = re.compile(token_pattern)
        self._inverted     = InvertedIndex(
            lowercase=lowercase, store_docs=store_docs, token_pattern=token_pattern
        )
        self._vocab:     Set[str]           = set()
        self._documents: List[Optional[str]] = []
        self._n_docs:    int                = 0

    def index(self, documents: Iterable[str]) -> "FuzzySearchEngine":
        docs = list(documents)
        self._inverted.index(docs)
        self._vocab     = set(self._inverted.vocabulary)
        self._n_docs    = self._inverted.n_docs
        self._documents = list(self._inverted._documents)
        self._fitted    = True
        return self

    def _expand_term(self, term: str) -> List[Tuple[str, int]]:
        """Return (vocab_term, edit_distance) pairs within max_edits of term."""
        if len(term) < self._min_term_len or self._max_edits == 0:
            return [(term, 0)] if term in self._vocab else []
        results: List[Tuple[str, int]] = []
        prefix  = term[: self._prefix_len] if self._prefix_len > 0 else None
        for vt in self._vocab:
            if prefix and not vt.startswith(prefix):
                continue
            if abs(len(vt) - len(term)) > self._max_edits:
                continue
            d = _levenshtein(term, vt)
            if d <= self._max_edits:
                results.append((vt, d))
        return results

    def search(self, query: str, *, top_k: int = 10) -> SearchResult:
        self._require_indexed()
        q_tokens = _tokenize(query, lowercase=self._lower, pattern=self._pattern)
        if not q_tokens:
            return SearchResult(
                query=query, hits=[], engine="FuzzySearchEngine",
                total_docs_searched=self._n_docs,
            )
        # Expand query tokens; keep max weight per vocab term across all tokens.
        expanded: Dict[str, float] = {}
        for token in q_tokens:
            for vt, dist in self._expand_term(token):
                w = 1.0 / (1.0 + dist)
                expanded[vt] = max(expanded.get(vt, 0.0), w)
        if not expanded:
            return SearchResult(
                query=query, hits=[], engine="FuzzySearchEngine",
                total_docs_searched=self._n_docs,
            )
        scores: Dict[int, float] = defaultdict(float)
        for vt, weight in expanded.items():
            for doc_id, positions in self._inverted._postings.get(vt, {}).items():
                tf = len(positions)
                scores[doc_id] += weight * (1.0 + math.log(tf))
        top  = heapq.nlargest(top_k, scores.items(), key=lambda kv: kv[1])
        hits = [
            SearchHit(
                doc_id=doc_id, score=score,
                document=self._documents[doc_id]
                if doc_id < len(self._documents) else None,
                metadata={"expanded_terms": list(expanded.keys())[:10]},
            )
            for doc_id, score in top
        ]
        return SearchResult(
            query=query, hits=hits, engine="FuzzySearchEngine",
            total_docs_searched=self._n_docs,
            metadata={
                "max_edits":        self._max_edits,
                "n_expanded_terms": len(expanded),
                "expanded_sample":  list(expanded.keys())[:10],
            },
        )

    @property
    def vocabulary(self) -> List[str]:
        return sorted(self._vocab)

    @property
    def n_docs(self) -> int:
        return self._n_docs


# ════════════════════════════════════════════════════════════════════════════
# TrieSearch
# ════════════════════════════════════════════════════════════════════════════

class _TrieNode:
    __slots__ = ("children", "is_end", "frequency", "doc_ids")

    def __init__(self) -> None:
        self.children:  Dict[str, "_TrieNode"] = {}
        self.is_end:    bool                    = False
        self.frequency: int                     = 0  # number of docs containing this word
        self.doc_ids:   Set[int]                = set()


class TrieSearch(BaseSearchEngine):
    """Trie-based prefix search and frequency-ranked autocomplete.

    Every unique token from the indexed corpus is inserted into a character
    trie.  search() tokenises the query and performs an exact match on all
    but the final token; the final token is matched as a prefix (autocomplete
    style) when prefix_only=True.

    autocomplete() returns the top-k words sharing a given prefix, ranked by
    document frequency (most widely-seen word first).

    Parameters
    ----------
    prefix_only : Match the final query token as a prefix (default True).
    """

    def __init__(
        self,
        *,
        lowercase:     bool = True,
        store_docs:    bool = True,
        prefix_only:   bool = True,
        token_pattern: str  = _DEFAULT_TOKEN_PATTERN,
    ) -> None:
        self._lower      = lowercase
        self._keep       = store_docs
        self._pfx_only   = prefix_only
        self._pattern    = re.compile(token_pattern)
        self._root       = _TrieNode()
        self._documents: List[Optional[str]] = []
        self._n_docs:    int = 0

    def _insert(self, word: str, doc_id: int) -> None:
        node = self._root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = _TrieNode()
            node = node.children[ch]
        if doc_id not in node.doc_ids:
            node.is_end     = True
            node.frequency += 1
            node.doc_ids.add(doc_id)

    def index(self, documents: Iterable[str]) -> "TrieSearch":
        self._root = _TrieNode()
        docs = list(documents)
        for doc_id, text in enumerate(docs):
            for token in _tokenize(text, lowercase=self._lower, pattern=self._pattern):
                self._insert(token, doc_id)
        self._documents = list(docs) if self._keep else [None] * len(docs)
        self._n_docs    = len(docs)
        self._fitted    = True
        return self

    def _find_node(self, prefix: str) -> Optional[_TrieNode]:
        node = self._root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def _collect(
        self, node: _TrieNode, prefix: str
    ) -> List[Tuple[str, int, Set[int]]]:
        """DFS from node; collect (word, frequency, doc_ids) for all end nodes."""
        out: List[Tuple[str, int, Set[int]]] = []
        stack = [(node, prefix)]
        while stack:
            cur, pfx = stack.pop()
            if cur.is_end:
                out.append((pfx, cur.frequency, cur.doc_ids))
            for ch, child in cur.children.items():
                stack.append((child, pfx + ch))
        return out

    def autocomplete(self, prefix: str, *, top_k: int = 10) -> List[str]:
        """Top-k completions for prefix, ranked by document frequency."""
        if self._lower:
            prefix = prefix.lower()
        node = self._find_node(prefix)
        if node is None:
            return []
        completions = self._collect(node, prefix)
        completions.sort(key=lambda x: -x[1])
        return [w for w, _, _ in completions[:top_k]]

    def search(self, query: str, *, top_k: int = 10) -> SearchResult:
        self._require_indexed()
        tokens = _tokenize(query, lowercase=self._lower, pattern=self._pattern)
        if not tokens:
            return SearchResult(
                query=query, hits=[], engine="TrieSearch",
                total_docs_searched=self._n_docs,
            )
        scores: Dict[int, float] = defaultdict(float)
        for i, token in enumerate(tokens):
            is_last = (i == len(tokens) - 1)
            if is_last and self._pfx_only:
                node = self._find_node(token)
                if node is not None:
                    for _, freq, doc_ids in self._collect(node, token):
                        for doc_id in doc_ids:
                            scores[doc_id] += freq * 0.5
            else:
                node = self._find_node(token)
                if node is not None and node.is_end:
                    for doc_id in node.doc_ids:
                        scores[doc_id] += node.frequency
        top  = heapq.nlargest(top_k, scores.items(), key=lambda kv: kv[1])
        hits = [
            SearchHit(
                doc_id=doc_id, score=score,
                document=self._documents[doc_id]
                if doc_id < len(self._documents) else None,
            )
            for doc_id, score in top
        ]
        return SearchResult(
            query=query, hits=hits, engine="TrieSearch",
            total_docs_searched=self._n_docs,
        )


# ════════════════════════════════════════════════════════════════════════════
# LinearVectorSearch  (exact brute-force ANN baseline)
# ════════════════════════════════════════════════════════════════════════════

class LinearVectorSearch(BaseVectorSearchEngine):
    """Exact brute-force nearest-neighbour search over dense vectors.

    Scores every indexed vector against the query and returns the top_k.
    Guarantees exact results; use as a correctness baseline against LSH / KD-tree.

    Parameters
    ----------
    metric : "cosine" (default) | "euclidean" | "dot"
    """

    def __init__(
        self,
        *,
        metric:     str  = "cosine",
        store_docs: bool = True,
    ) -> None:
        if metric not in ("cosine", "euclidean", "dot"):
            raise ValueError(f"metric must be 'cosine', 'euclidean', or 'dot'; got {metric!r}")
        self._metric    = metric
        self._keep      = store_docs
        self._vectors:   List[List[float]]  = []
        self._documents: List[Optional[str]] = []
        self._n_vectors: int = 0
        self._dim:       Optional[int] = None

    def index(
        self,
        vectors:   Iterable[List[float]],
        documents: Optional[Iterable[str]] = None,
    ) -> "LinearVectorSearch":
        vecs = [list(v) for v in vectors]
        if vecs:
            self._dim = len(vecs[0])
        docs = list(documents) if documents is not None else [None] * len(vecs)
        if len(docs) != len(vecs):
            docs = (docs + [None] * len(vecs))[: len(vecs)]
        self._vectors   = vecs
        self._documents = docs
        self._n_vectors = len(vecs)
        self._fitted    = True
        return self

    def _score(self, u: List[float], v: List[float]) -> float:
        if self._metric == "cosine":
            return _cosine(u, v)
        if self._metric == "dot":
            return sum(a * b for a, b in zip(u, v))
        return -_euclidean(u, v)  # negate so higher = better

    def search_vector(self, query: List[float], *, top_k: int = 10) -> SearchResult:
        self._require_indexed()
        if not self._vectors:
            return SearchResult(
                query=repr(query[:4]), hits=[],
                engine="LinearVectorSearch", total_docs_searched=0,
            )
        if self._dim is not None and len(query) != self._dim:
            raise ValueError(
                f"Query dimension {len(query)} does not match indexed dimension {self._dim}"
            )
        scored = [(i, self._score(query, v)) for i, v in enumerate(self._vectors)]
        top    = heapq.nlargest(top_k, scored, key=lambda x: x[1])
        hits   = [
            SearchHit(doc_id=i, score=s, document=self._documents[i])
            for i, s in top
        ]
        return SearchResult(
            query=repr(query[:4]) + "...", hits=hits,
            engine="LinearVectorSearch",
            total_docs_searched=self._n_vectors,
            metadata={"metric": self._metric},
        )

    @property
    def n_vectors(self) -> int:
        return self._n_vectors

    @property
    def dim(self) -> Optional[int]:
        return self._dim


# ════════════════════════════════════════════════════════════════════════════
# LSHSearch  (random-hyperplane approximate nearest-neighbour)
# ════════════════════════════════════════════════════════════════════════════

class LSHSearch(BaseVectorSearchEngine):
    """Approximate nearest-neighbour via Locality Sensitive Hashing.

    Uses the SimHash / random-hyperplane technique for cosine similarity:
    each bit in the hash is 1 if the dot product with a random unit vector
    is ≥ 0, and 0 otherwise.  Vectors that are close in cosine space are
    likely to share the same binary hash code.

    Multiple independent hash tables increase recall: a candidate is included
    if it matches the query in any table's bucket.  The full candidate set is
    then re-ranked by exact cosine similarity, so precision is not sacrificed.

    Parameters
    ----------
    n_tables   : Number of independent hash tables.  More tables → better
                 recall at the cost of more memory and slightly slower indexing.
    n_hashes   : Bits per hash table.  More bits → smaller buckets (better
                 precision) but fewer candidates per query (lower recall).
    random_seed: Integer seed for reproducible hyperplanes; None = entropy.
    """

    def __init__(
        self,
        *,
        n_tables:    int          = 8,
        n_hashes:    int          = 8,
        dim:         Optional[int] = None,
        random_seed: Optional[int] = None,
        store_docs:  bool          = True,
    ) -> None:
        if n_tables < 1:
            raise ValueError(f"n_tables must be >= 1; got {n_tables}")
        if n_hashes < 1:
            raise ValueError(f"n_hashes must be >= 1; got {n_hashes}")
        self._n_tables   = n_tables
        self._n_hashes   = n_hashes
        self._dim        = dim
        self._seed       = random_seed
        self._keep       = store_docs
        self._vectors:   List[List[float]]  = []
        self._documents: List[Optional[str]] = []
        self._n_vectors: int = 0
        # Built by index(): list of (hyperplanes, bucket_map)
        self._tables: List[Tuple[List[List[float]], Dict[int, List[int]]]] = []

    # ------------------------------------------------------------------

    def _random_unit_vector(self, dim: int, rng: random.Random) -> List[float]:
        v    = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def _hash(self, v: List[float], planes: List[List[float]]) -> int:
        bits = 0
        for plane in planes:
            bits = (bits << 1) | (1 if sum(a * b for a, b in zip(v, plane)) >= 0 else 0)
        return bits

    def index(
        self,
        vectors:   Iterable[List[float]],
        documents: Optional[Iterable[str]] = None,
    ) -> "LSHSearch":
        vecs = [list(v) for v in vectors]
        if not vecs:
            self._fitted = True
            return self
        dim        = len(vecs[0])
        self._dim  = dim
        docs = list(documents) if documents is not None else [None] * len(vecs)
        if len(docs) != len(vecs):
            docs = (docs + [None] * len(vecs))[: len(vecs)]
        self._vectors   = vecs
        self._documents = docs
        self._n_vectors = len(vecs)
        rng = random.Random(self._seed)
        self._tables = []
        for _ in range(self._n_tables):
            planes  = [self._random_unit_vector(dim, rng) for _ in range(self._n_hashes)]
            buckets: Dict[int, List[int]] = defaultdict(list)
            for idx, v in enumerate(vecs):
                buckets[self._hash(v, planes)].append(idx)
            self._tables.append((planes, dict(buckets)))
        self._fitted = True
        return self

    def search_vector(self, query: List[float], *, top_k: int = 10) -> SearchResult:
        self._require_indexed()
        if not self._vectors:
            return SearchResult(
                query=repr(query[:4]), hits=[],
                engine="LSHSearch", total_docs_searched=0,
            )
        if self._dim is not None and len(query) != self._dim:
            raise ValueError(
                f"Query dimension {len(query)} does not match indexed dimension {self._dim}"
            )
        candidates: Set[int] = set()
        for planes, buckets in self._tables:
            h = self._hash(query, planes)
            candidates.update(buckets.get(h, []))
        # Fallback to full scan if no candidates found (guarantees non-empty results).
        search_set = candidates if candidates else set(range(self._n_vectors))
        scored     = [(i, _cosine(query, self._vectors[i])) for i in search_set]
        top        = heapq.nlargest(top_k, scored, key=lambda x: x[1])
        hits       = [
            SearchHit(doc_id=i, score=s, document=self._documents[i])
            for i, s in top
        ]
        return SearchResult(
            query=repr(query[:4]) + "...", hits=hits,
            engine="LSHSearch",
            total_docs_searched=len(search_set),
            metadata={
                "n_candidates": len(candidates),
                "n_tables":     self._n_tables,
                "n_hashes":     self._n_hashes,
            },
        )

    @property
    def n_vectors(self) -> int:
        return self._n_vectors


# ════════════════════════════════════════════════════════════════════════════
# KDTreeSearch
# ════════════════════════════════════════════════════════════════════════════

class _KDNode:
    __slots__ = ("point", "doc_id", "axis", "left", "right")

    def __init__(
        self,
        point:  List[float],
        doc_id: int,
        axis:   int,
        left:   Optional["_KDNode"] = None,
        right:  Optional["_KDNode"] = None,
    ) -> None:
        self.point  = point
        self.doc_id = doc_id
        self.axis   = axis
        self.left   = left
        self.right  = right


class KDTreeSearch(BaseVectorSearchEngine):
    """KD-tree exact nearest-neighbour for low-dimensional dense vectors.

    Constructs a balanced KD-tree by recursively splitting on the median of the
    current axis (cycling through all dimensions).  Query traversal uses the
    standard backtracking algorithm that prunes subtrees when the splitting-plane
    distance exceeds the current best known distance.

    Complexity : O(n log n) build, O(log n) average query — degrades toward O(n)
                 as dimensionality grows.  Recommended for dim ≤ 20; prefer
                 LSHSearch for higher dimensions.

    Parameters
    ----------
    metric : "euclidean" (default) | "cosine"
    """

    def __init__(
        self,
        *,
        metric:     str  = "euclidean",
        store_docs: bool = True,
    ) -> None:
        if metric not in ("euclidean", "cosine"):
            raise ValueError(f"metric must be 'euclidean' or 'cosine'; got {metric!r}")
        self._metric    = metric
        self._keep      = store_docs
        self._root:      Optional[_KDNode]   = None
        self._documents: List[Optional[str]] = []
        self._n_vectors: int                 = 0
        self._dim:       Optional[int]       = None

    def _build(
        self,
        points: List[Tuple[List[float], int]],
        depth:  int = 0,
    ) -> Optional[_KDNode]:
        if not points:
            return None
        dim  = len(points[0][0])
        axis = depth % dim
        points.sort(key=lambda p: p[0][axis])
        mid  = len(points) // 2
        pt, doc_id = points[mid]
        return _KDNode(
            point=pt, doc_id=doc_id, axis=axis,
            left =self._build(points[:mid],       depth + 1),
            right=self._build(points[mid + 1:],   depth + 1),
        )

    def index(
        self,
        vectors:   Iterable[List[float]],
        documents: Optional[Iterable[str]] = None,
    ) -> "KDTreeSearch":
        vecs = [list(v) for v in vectors]
        if not vecs:
            self._fitted = True
            return self
        self._dim = len(vecs[0])
        docs = list(documents) if documents is not None else [None] * len(vecs)
        if len(docs) != len(vecs):
            docs = (docs + [None] * len(vecs))[: len(vecs)]
        self._documents  = docs
        self._n_vectors  = len(vecs)
        self._root       = self._build([(v, i) for i, v in enumerate(vecs)])
        self._fitted     = True
        return self

    def _dist(self, u: List[float], v: List[float]) -> float:
        return _euclidean(u, v) if self._metric == "euclidean" else 1.0 - _cosine(u, v)

    def _nn_search(
        self,
        node:  Optional[_KDNode],
        query: List[float],
        heap:  List[Tuple[float, int]],  # max-heap of (neg_dist, doc_id)
        k:     int,
    ) -> None:
        if node is None:
            return
        d = self._dist(query, node.point)
        if len(heap) < k:
            heapq.heappush(heap, (-d, node.doc_id))
        elif d < -heap[0][0]:
            heapq.heapreplace(heap, (-d, node.doc_id))
        axis  = node.axis
        delta = query[axis] - node.point[axis]
        near, far = (node.left, node.right) if delta <= 0 else (node.right, node.left)
        self._nn_search(near, query, heap, k)
        if len(heap) < k or abs(delta) < -heap[0][0]:
            self._nn_search(far, query, heap, k)

    def search_vector(self, query: List[float], *, top_k: int = 10) -> SearchResult:
        self._require_indexed()
        if self._root is None or top_k <= 0:
            return SearchResult(
                query=repr(query[:4]), hits=[],
                engine="KDTreeSearch", total_docs_searched=self._n_vectors,
            )
        if self._dim is not None and len(query) != self._dim:
            raise ValueError(
                f"Query dimension {len(query)} does not match indexed dimension {self._dim}"
            )
        heap: List[Tuple[float, int]] = []
        self._nn_search(self._root, query, heap, top_k)
        # Sort ascending by neg_dist (= descending distance = closest first).
        heap.sort()
        hits: List[SearchHit] = []
        for neg_dist, doc_id in heap:
            dist  = -neg_dist
            score = (1.0 - dist) if self._metric == "cosine" else 1.0 / (1.0 + dist)
            hits.append(SearchHit(
                doc_id=doc_id, score=score,
                document=self._documents[doc_id],
                metadata={"distance": dist},
            ))
        hits.sort(key=lambda h: -h.score)
        return SearchResult(
            query=repr(query[:4]) + "...", hits=hits[:top_k],
            engine="KDTreeSearch",
            total_docs_searched=self._n_vectors,
            metadata={"metric": self._metric},
        )

    @property
    def n_vectors(self) -> int:
        return self._n_vectors


# ════════════════════════════════════════════════════════════════════════════
# MultiSearchEngine  (ensemble score fusion)
# ════════════════════════════════════════════════════════════════════════════

class MultiSearchEngine(BaseSearchEngine):
    """Ensemble search engine that fuses results from multiple text engines.

    Fusion modes
    ------------
    "rrf"      : Reciprocal Rank Fusion (default, Cormack et al. 2009).
                 Score-free and robust to differences in score scale between engines.
    "weighted" : Weighted linear combination of min-max normalised scores.
                 Requires tuned weights for best results.

    Parameters
    ----------
    engines : List of BaseSearchEngine instances to combine.  All engines share
              the same corpus — index() calls index() on each sub-engine.
    weights : Per-engine weights for "weighted" mode.  Normalised to sum to 1.
    rrf_k   : RRF smoothing constant (default 60).
    """

    def __init__(
        self,
        engines: List[BaseSearchEngine],
        *,
        weights: Optional[List[float]] = None,
        mode:    str                   = "rrf",
        rrf_k:   int                   = 60,
    ) -> None:
        if not engines:
            raise ValueError("MultiSearchEngine requires at least one engine.")
        if mode not in ("rrf", "weighted"):
            raise ValueError(f"mode must be 'rrf' or 'weighted'; got {mode!r}")
        self._engines = engines
        self._mode    = mode
        self._rrf_k   = rrf_k
        if weights is not None:
            if len(weights) != len(engines):
                raise ValueError(
                    f"len(weights)={len(weights)} != len(engines)={len(engines)}"
                )
            if any(w < 0 for w in weights):
                raise ValueError("weights must all be non-negative")
            total = sum(weights)
            if total <= 0.0:
                raise ValueError(
                    f"weights must sum to a positive value; got sum={total}"
                )
            self._weights = [w / total for w in weights]
        else:
            self._weights = [1.0 / len(engines)] * len(engines)

    def index(self, documents: Iterable[str]) -> "MultiSearchEngine":
        docs = list(documents)
        for engine in self._engines:
            engine.index(docs)
        self._fitted = True
        return self

    def search(self, query: str, *, top_k: int = 10) -> SearchResult:
        self._require_indexed()
        per_engine = [e.search(query, top_k=top_k) for e in self._engines]
        fused      = (
            self._rrf_fuse(per_engine)
            if self._mode == "rrf"
            else self._weighted_fuse(per_engine)
        )
        top = heapq.nlargest(top_k, fused.items(), key=lambda kv: kv[1])
        doc_lookup: Dict[int, Optional[str]] = {}
        for result in per_engine:
            for hit in result.hits:
                if hit.document is not None and hit.doc_id not in doc_lookup:
                    doc_lookup[hit.doc_id] = hit.document
        hits = [
            SearchHit(doc_id=doc_id, score=score, document=doc_lookup.get(doc_id))
            for doc_id, score in top
        ]
        return SearchResult(
            query=query, hits=hits, engine="MultiSearchEngine",
            total_docs_searched=max(
                (r.total_docs_searched for r in per_engine), default=0
            ),
            metadata={
                "mode":         self._mode,
                "n_engines":    len(self._engines),
                "engine_names": [type(e).__name__ for e in self._engines],
            },
        )

    def _rrf_fuse(self, results: List[SearchResult]) -> Dict[int, float]:
        scores: Dict[int, float] = defaultdict(float)
        k = self._rrf_k
        for result in results:
            for rank, hit in enumerate(result.hits, start=1):
                scores[hit.doc_id] += 1.0 / (k + rank)
        return scores

    def _weighted_fuse(self, results: List[SearchResult]) -> Dict[int, float]:
        scores: Dict[int, float] = defaultdict(float)
        for weight, result in zip(self._weights, results):
            if not result.hits:
                continue
            max_s = max(h.score for h in result.hits) or 1.0
            for hit in result.hits:
                scores[hit.doc_id] += weight * (hit.score / max_s)
        return scores


# ════════════════════════════════════════════════════════════════════════════
# RRFRanker  (standalone utility)
# ════════════════════════════════════════════════════════════════════════════

class RRFRanker:
    """Standalone Reciprocal Rank Fusion over multiple ranked lists.

    RRF(d) = Σ_r  1 / (k + rank_r(d))

    where rank_r(d) is d's 1-based rank in result list r and k is a smoothing
    constant (default 60, Cormack, Clarke & Buettcher 2009).

    Usage
    -----
    >>> ranker = RRFRanker(k=60)
    >>> fused = ranker.fuse([[1, 2, 3], [2, 1, 4]])
    >>> # [(doc_id, rrf_score), ...] sorted descending
    """

    def __init__(self, *, k: int = 60) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}")
        self._k = k

    def fuse(
        self,
        ranked_lists: List[List[int]],
        *,
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Return (doc_id, rrf_score) sorted descending.

        Parameters
        ----------
        ranked_lists : Each inner list is doc IDs ranked best-to-worst.
        top_k        : Truncate output; None returns all.
        """
        scores: Dict[int, float] = defaultdict(float)
        for ranked in ranked_lists:
            for rank, doc_id in enumerate(ranked, start=1):
                scores[doc_id] += 1.0 / (self._k + rank)
        result = sorted(scores.items(), key=lambda kv: -kv[1])
        return result if top_k is None else result[:top_k]

    def fuse_results(
        self,
        search_results: List[SearchResult],
        *,
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Convenience: fuse a list of SearchResult objects."""
        return self.fuse([r.get_doc_ids() for r in search_results], top_k=top_k)


# ════════════════════════════════════════════════════════════════════════════
# ScoreNormaliser
# ════════════════════════════════════════════════════════════════════════════

class ScoreNormaliser:
    """Collection of score normalisation functions.  All methods return new lists."""

    @staticmethod
    def min_max(scores: List[float]) -> List[float]:
        """Scale scores to [0, 1]."""
        if not scores:
            return []
        lo, hi = min(scores), max(scores)
        span   = hi - lo
        if span == 0.0:
            return [1.0 if s > 0 else 0.0 for s in scores]
        return [(s - lo) / span for s in scores]

    @staticmethod
    def z_score(scores: List[float]) -> List[float]:
        """Standardise to zero mean, unit standard deviation."""
        if not scores:
            return []
        n    = len(scores)
        mean = sum(scores) / n
        var  = sum((s - mean) ** 2 for s in scores) / n
        std  = math.sqrt(var) if var > 0.0 else 1.0
        return [(s - mean) / std for s in scores]

    @staticmethod
    def softmax(scores: List[float], *, temperature: float = 1.0) -> List[float]:
        """Convert scores to a probability distribution via softmax."""
        if not scores:
            return []
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0; got {temperature}")
        scaled = [s / temperature for s in scores]
        max_s  = max(scaled)
        exps   = [math.exp(s - max_s) for s in scaled]
        total  = sum(exps)
        return [e / total for e in exps]

    @staticmethod
    def rank_reciprocal(scores: List[float]) -> List[float]:
        """Map each score to 1/rank where rank 1 = highest score."""
        if not scores:
            return []
        order  = sorted(range(len(scores)), key=lambda i: -scores[i])
        result = [0.0] * len(scores)
        for rank, idx in enumerate(order, start=1):
            result[idx] = 1.0 / rank
        return result


# ════════════════════════════════════════════════════════════════════════════
# HybridSearchEngine  (sparse + dense fusion)
# ════════════════════════════════════════════════════════════════════════════

class HybridSearchEngine:
    """Hybrid sparse-plus-dense retrieval engine.

    Combines a text engine (sparse retrieval) with a vector engine (dense retrieval)
    using either Reciprocal Rank Fusion or weighted linear combination of normalised
    scores.  This is the dominant production-search architecture used in Elasticsearch
    8.x, Weaviate, Pinecone, and most modern RAG / semantic-search systems.

    Parameters
    ----------
    text_engine   : Any BaseSearchEngine (BM25SearchEngine, InvertedIndex, etc.).
    vector_engine : Any BaseVectorSearchEngine (LinearVectorSearch, LSHSearch, etc.).
    mode          : "rrf" (default) — rank-based, scale-agnostic; or "weighted" —
                    linear combination of min-max-normalised scores.
    rrf_k         : RRF smoothing constant (default 60, Cormack et al. 2009).
    text_weight   : Weight for text scores in "weighted" mode (default 0.5).
    vector_weight : Weight for vector scores in "weighted" mode (default 0.5).

    Usage
    -----
    >>> bm25   = BM25SearchEngine()
    >>> lin    = LinearVectorSearch(metric="cosine")
    >>> hybrid = HybridSearchEngine(bm25, lin, mode="rrf")
    >>> hybrid.index(documents, vectors)
    >>> result = hybrid.search("query text", query_vector=[...], top_k=5)
    """

    def __init__(
        self,
        text_engine:   BaseSearchEngine,
        vector_engine: BaseVectorSearchEngine,
        *,
        mode:          str   = "rrf",
        rrf_k:         int   = 60,
        text_weight:   float = 0.5,
        vector_weight: float = 0.5,
    ) -> None:
        if mode not in ("rrf", "weighted"):
            raise ValueError(f"mode must be 'rrf' or 'weighted'; got {mode!r}")
        if rrf_k < 1:
            raise ValueError(f"rrf_k must be >= 1; got {rrf_k}")
        if getattr(text_engine, "_accumulates", False):
            raise TypeError(
                f"{type(text_engine).__name__} appends to its index on every index() call "
                "and cannot be used with HybridSearchEngine — doc_ids would become "
                "misaligned with the vector engine after re-indexing. "
                "Use BM25SearchEngine, TFIDFSearchEngine, TrieSearch, or MultiSearchEngine."
            )
        if mode == "weighted":
            total = text_weight + vector_weight
            if total <= 0.0:
                raise ValueError("text_weight + vector_weight must be positive")
            text_weight   = text_weight   / total
            vector_weight = vector_weight / total
        self._text    = text_engine
        self._vec     = vector_engine
        self._mode    = mode
        self._rrf_k   = rrf_k
        self._tw      = text_weight
        self._vw      = vector_weight
        self._n_docs: int                        = 0
        self._fitted: bool                       = False

    def index(
        self,
        documents: Iterable[str],
        vectors:   Iterable[List[float]],
    ) -> "HybridSearchEngine":
        """Index documents and their corresponding dense vectors.

        ``len(documents)`` must equal ``len(vectors)`` — each vector is the
        dense embedding of the document at the same position.
        """
        docs = list(documents)
        vecs = list(vectors)
        if len(docs) != len(vecs):
            raise ValueError(
                f"len(documents)={len(docs)} != len(vectors)={len(vecs)}"
            )
        self._text.index(docs)
        self._vec.index(vecs, docs)
        self._n_docs = len(docs)
        self._fitted = True
        return self

    def search(
        self,
        query:        str,
        query_vector: List[float],
        *,
        top_k: int = 10,
    ) -> SearchResult:
        """Return top_k results fusing sparse-text and dense-vector retrieval.

        Parameters
        ----------
        query        : Text query for the sparse engine.
        query_vector : Dense embedding of the query for the vector engine.
        top_k        : Number of final results to return.
        """
        if not self._fitted:
            raise RuntimeError(
                "HybridSearchEngine has not been indexed. Call index() first."
            )
        if self._n_docs == 0:
            return SearchResult(
                query=query, hits=[], engine="HybridSearchEngine",
                total_docs_searched=0,
            )
        fetch_k      = min(self._n_docs, max(top_k * 3, 50))
        text_result  = self._text.search(query,        top_k=fetch_k)
        vec_result   = self._vec.search_vector(query_vector, top_k=fetch_k)
        scores       = (
            self._rrf_fuse(text_result, vec_result)
            if self._mode == "rrf"
            else self._weighted_fuse(text_result, vec_result)
        )
        top          = heapq.nlargest(top_k, scores.items(), key=lambda kv: kv[1])
        doc_lookup: Dict[int, Optional[str]] = {}
        for hit in (*text_result.hits, *vec_result.hits):
            if hit.doc_id not in doc_lookup:
                doc_lookup[hit.doc_id] = hit.document
        hits = [
            SearchHit(doc_id=doc_id, score=score, document=doc_lookup.get(doc_id))
            for doc_id, score in top
        ]
        return SearchResult(
            query=query, hits=hits, engine="HybridSearchEngine",
            total_docs_searched=self._n_docs,
            metadata={
                "mode":              self._mode,
                "text_engine":       type(self._text).__name__,
                "vector_engine":     type(self._vec).__name__,
                "text_candidates":   text_result.n_hits,
                "vector_candidates": vec_result.n_hits,
            },
        )

    def _rrf_fuse(
        self,
        text_result: SearchResult,
        vec_result:  SearchResult,
    ) -> Dict[int, float]:
        scores: Dict[int, float] = defaultdict(float)
        k = self._rrf_k
        for rank, hit in enumerate(text_result.hits, start=1):
            scores[hit.doc_id] += 1.0 / (k + rank)
        for rank, hit in enumerate(vec_result.hits, start=1):
            scores[hit.doc_id] += 1.0 / (k + rank)
        return scores

    def _weighted_fuse(
        self,
        text_result: SearchResult,
        vec_result:  SearchResult,
    ) -> Dict[int, float]:
        scores: Dict[int, float] = defaultdict(float)
        for result, weight in ((text_result, self._tw), (vec_result, self._vw)):
            if not result.hits:
                continue
            max_s = max(h.score for h in result.hits) or 1.0
            for hit in result.hits:
                scores[hit.doc_id] += weight * (hit.score / max_s)
        return scores

    @property
    def n_docs(self) -> int:
        return self._n_docs


# ════════════════════════════════════════════════════════════════════════════
# BM25F  (field-weighted BM25 for structured documents)
# ════════════════════════════════════════════════════════════════════════════

class BM25F:
    """BM25F: field-weighted BM25 for structured multi-field documents.

    Documents are provided as ``{field_name: text}`` dicts.  Each field
    contributes independently to a per-term weighted TF, which is then fed into
    a single shared BM25 saturation curve.

    Robertson-Zaragoza BM25F formula (uniform k1 across fields):

        WTF(t, d)  = Σ_f  boost_f · tf(t, d_f)
                            ─────────────────────────────
                            1 + b_f · (|d_f| / avg_len_f)

        score(q, d) = Σ_{t ∈ q}  IDF(t) · WTF(t, d) · (k1 + 1)
                                         ─────────────────────────
                                         WTF(t, d) + k1

    Parameters
    ----------
    field_config : Mapping ``{field_name: {"b": float, "boost": float}}``.
                   Defaults: ``b=0.75``, ``boost=1.0``.  Fields absent from
                   a document are treated as empty text.
    k1           : BM25 saturation parameter (default 1.5).
    epsilon      : IDF floor to prevent negative scores (default 0.25).

    Usage
    -----
    >>> docs = [
    ...     {"title": "Python search engines", "body": "BM25 ranking formulas"},
    ...     {"title": "Machine learning",       "body": "neural retrieval models"},
    ... ]
    >>> bm25f = BM25F(
    ...     field_config={"title": {"b": 0.4, "boost": 2.5},
    ...                   "body":  {"b": 0.75, "boost": 1.0}},
    ... )
    >>> bm25f.index(docs)
    >>> bm25f.search("search ranking", top_k=5)
    """

    def __init__(
        self,
        field_config:  Optional[Dict[str, Dict[str, float]]] = None,
        *,
        k1:            float = 1.5,
        epsilon:       float = 0.25,
        lowercase:     bool  = True,
        token_pattern: str   = _DEFAULT_TOKEN_PATTERN,
    ) -> None:
        if k1 < 0.0:
            raise ValueError(f"k1 must be >= 0; got {k1}")
        self._field_cfg  = field_config or {}
        self._k1         = k1
        self._epsilon    = epsilon
        self._lower      = lowercase
        self._pattern    = re.compile(token_pattern)
        self._fields:     List[str]                    = []
        self._avglen:     Dict[str, float]             = {}
        self._field_tfs:  Dict[str, List[Counter]]    = {}
        self._idf:        Dict[str, float]             = {}
        self._postings:   Dict[str, Set[int]]          = defaultdict(set)
        self._documents:  List[str]                    = []
        self._n_docs:     int                          = 0
        self._fitted:     bool                         = False

    def index(
        self,
        documents: Iterable[Dict[str, str]],
    ) -> "BM25F":
        """Index a collection of structured documents.

        Parameters
        ----------
        documents : Iterable of ``{field_name: text}`` dicts.  All documents
                    should share the same set of field names (missing fields are
                    treated as empty strings).
        """
        self._postings.clear()
        docs = list(documents)
        if not docs:
            self._n_docs = 0; self._fitted = True; return self

        # Determine the field order: explicit config first, then from docs.
        all_fields: List[str] = list(self._field_cfg.keys()) or list(
            dict.fromkeys(k for d in docs for k in d)
        )
        self._fields = all_fields

        # Tokenise each field independently.
        field_tokens: Dict[str, List[List[str]]] = {
            fld: [
                _tokenize(d.get(fld, ""), lowercase=self._lower, pattern=self._pattern)
                for d in docs
            ]
            for fld in all_fields
        }

        # Per-field average lengths.
        for fld in all_fields:
            lengths = [len(t) for t in field_tokens[fld]]
            self._avglen[fld] = sum(lengths) / len(lengths) if lengths else 1.0

        # Per-field per-doc term-frequency Counters.
        self._field_tfs = {
            fld: [Counter(toks) for toks in field_tokens[fld]]
            for fld in all_fields
        }

        # Global document-frequency (any field containing the term counts once).
        df: Counter = Counter()
        for fld, tok_lists in field_tokens.items():
            for doc_idx, toks in enumerate(tok_lists):
                unique = set(toks)
                df.update(unique)
                for tok in unique:
                    self._postings[tok].add(doc_idx)

        N = len(docs)
        self._idf = {
            t: max(self._epsilon, math.log((N - c + 0.5) / (c + 0.5) + 1.0))
            for t, c in df.items()
        }
        # Store a flat text representation for SearchHit snippets.
        self._documents = [" | ".join(str(v) for v in d.values()) for d in docs]
        self._n_docs    = N
        self._fitted    = True
        return self

    def _wtf(self, term: str, doc_idx: int) -> float:
        """Weighted term frequency across all fields for one document."""
        wtf = 0.0
        for fld in self._fields:
            cfg    = self._field_cfg.get(fld, {})
            b      = float(cfg.get("b",     0.75))
            boost  = float(cfg.get("boost", 1.0))
            tf     = self._field_tfs[fld][doc_idx].get(term, 0)
            if tf == 0:
                continue
            avglen = self._avglen.get(fld, 1.0)
            dl     = sum(self._field_tfs[fld][doc_idx].values())
            norm   = 1.0 + b * (dl / avglen if avglen > 0 else 0.0)
            wtf   += boost * tf / norm
        return wtf

    def _score_doc(self, query_tokens: List[str], doc_idx: int) -> float:
        k1    = self._k1
        score = 0.0
        for qt in set(query_tokens):
            idf = self._idf.get(qt, 0.0)
            wtf = self._wtf(qt, doc_idx)
            if wtf == 0.0:
                continue
            score += idf * (wtf * (k1 + 1.0) / (wtf + k1))
        return score

    def search(self, query: str, *, top_k: int = 10) -> SearchResult:
        """Rank documents by BM25F relevance score."""
        if not self._fitted:
            raise RuntimeError("BM25F has not been indexed. Call index() first.")
        q_tokens = _tokenize(query, lowercase=self._lower, pattern=self._pattern)
        if not q_tokens or self._n_docs == 0:
            return SearchResult(
                query=query, hits=[], engine="BM25F",
                total_docs_searched=self._n_docs,
            )
        candidates: Set[int] = set()
        for qt in set(q_tokens):
            candidates |= self._postings.get(qt, set())
        if not candidates:
            return SearchResult(
                query=query, hits=[], engine="BM25F",
                total_docs_searched=self._n_docs,
            )
        scored = [(i, self._score_doc(q_tokens, i)) for i in candidates]
        top    = heapq.nlargest(top_k, scored, key=lambda x: x[1])
        hits   = [
            SearchHit(doc_id=i, score=s, document=self._documents[i])
            for i, s in top
        ]
        return SearchResult(
            query=query, hits=hits, engine="BM25F",
            total_docs_searched=self._n_docs,
            metadata={"k1": self._k1, "fields": self._fields},
        )

    @property
    def n_docs(self) -> int:
        return self._n_docs

    @property
    def idf_(self) -> Dict[str, float]:
        """IDF values for all indexed terms."""
        if not self._fitted:
            raise RuntimeError("BM25F has not been indexed. Call index() first.")
        return dict(self._idf)


# ════════════════════════════════════════════════════════════════════════════
# MaximalMarginalRelevance  (diversity-aware reranking)
# ════════════════════════════════════════════════════════════════════════════

class MaximalMarginalRelevance:
    """Greedy MMR reranking: balance relevance against result diversity.

    At each selection step picks the candidate that maximises:

        MMR(d) = λ · sim(q, d)  −  (1 − λ) · max_{d' ∈ S} sim(d, d')

    where S is the set of already-selected documents, λ ∈ [0, 1] controls the
    relevance-diversity tradeoff, and sim is cosine similarity.

    Use cases
    ---------
    * Removing near-duplicate results from dense-retrieval outputs.
    * Ensuring topical diversity in multi-document summaries.
    * Snippet selection for long-document search results.

    Parameters
    ----------
    lambda_ : Relevance weight (default 0.5).
              λ = 1 → pure relevance ranking; λ = 0 → maximum diversity.

    Reference
    ---------
    Carbonell & Goldstein, "The use of MMR, diversity-based reranking for
    reordering documents and producing summaries", SIGIR 1998.
    """

    def __init__(self, *, lambda_: float = 0.5) -> None:
        if not (0.0 <= lambda_ <= 1.0):
            raise ValueError(f"lambda_ must be in [0, 1]; got {lambda_}")
        self._lambda = lambda_

    def rerank(
        self,
        query_vector:      List[float],
        candidate_vectors: List[List[float]],
        candidate_ids:     Optional[List[int]] = None,
        *,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Rerank candidate vectors by MMR score.

        Parameters
        ----------
        query_vector      : Dense query embedding.
        candidate_vectors : Dense vectors for each candidate document.
        candidate_ids     : Doc IDs parallel to candidate_vectors; defaults to
                            ``[0, 1, 2, ...]``.
        top_k             : Number of results to return.

        Returns
        -------
        ``[(doc_id, mmr_score), ...]`` sorted descending.
        """
        if not candidate_vectors:
            return []
        n     = len(candidate_vectors)
        ids   = list(candidate_ids) if candidate_ids is not None else list(range(n))
        if len(ids) != n:
            raise ValueError(
                f"len(candidate_ids)={len(ids)} != len(candidate_vectors)={n}"
            )
        top_k     = min(top_k, n)
        lam       = self._lambda
        one_minus = 1.0 - lam

        # Pre-compute query-candidate relevance scores once.
        rel = [_cosine(query_vector, v) for v in candidate_vectors]

        selected_vecs: List[List[float]] = []
        remaining                        = list(range(n))
        results: List[Tuple[int, float]] = []

        for _ in range(top_k):
            best_idx   = -1
            best_score = -float("inf")
            for i in remaining:
                relevance  = lam * rel[i]
                redundancy = (
                    one_minus * max(_cosine(candidate_vectors[i], sv) for sv in selected_vecs)
                    if selected_vecs else 0.0
                )
                mmr = relevance - redundancy
                if mmr > best_score:
                    best_score = mmr
                    best_idx   = i
            if best_idx == -1:
                break
            selected_vecs.append(candidate_vectors[best_idx])
            remaining.remove(best_idx)
            results.append((ids[best_idx], best_score))

        return results

    def rerank_results(
        self,
        query_vector:     List[float],
        result:           SearchResult,
        document_vectors: List[List[float]],
        *,
        top_k: int = 10,
    ) -> SearchResult:
        """Rerank a SearchResult using MMR diversity reranking.

        Parameters
        ----------
        query_vector     : Dense query embedding.
        result           : SearchResult from any engine (the candidates).
        document_vectors : Dense vectors for the full corpus, indexed by doc_id.
        top_k            : Number of re-ranked results to return.
        """
        if not result.hits:
            return SearchResult(
                query=result.query, hits=[], engine="MaximalMarginalRelevance",
                total_docs_searched=result.total_docs_searched,
                metadata={"lambda": self._lambda},
            )
        cand_ids  = [h.doc_id for h in result.hits]
        cand_vecs = [document_vectors[doc_id] for doc_id in cand_ids]
        doc_map   = {h.doc_id: h for h in result.hits}
        reranked  = self.rerank(query_vector, cand_vecs, cand_ids, top_k=top_k)
        hits      = [
            SearchHit(
                doc_id=doc_id, score=mmr_score,
                document=doc_map[doc_id].document,
                metadata={**doc_map[doc_id].metadata, "original_score": doc_map[doc_id].score},
            )
            for doc_id, mmr_score in reranked
        ]
        return SearchResult(
            query=result.query, hits=hits, engine="MaximalMarginalRelevance",
            total_docs_searched=result.total_docs_searched,
            metadata={"lambda": self._lambda, "original_engine": result.engine},
        )


# ════════════════════════════════════════════════════════════════════════════
# SparseVectorIndex  (learned sparse / explicit feature-weight retrieval)
# ════════════════════════════════════════════════════════════════════════════

class SparseVectorIndex:
    """Dot-product retrieval over explicit sparse term-weight vectors.

    Documents are stored as ``{term: weight}`` dicts rather than raw text.
    Retrieval score is the dot product between query and document sparse vectors,
    computed efficiently via an inverted index over terms.

    Use cases
    ---------
    * Learned sparse representations: SPLADE, UniCOIL, DeepCT — where a neural
      model produces a ``{expanded_term: weight}`` dict for each document.
    * Custom term-weighting schemes where IDF or BM25 is not appropriate.
    * Feature-based retrieval for structured data converted to sparse vectors.
    * Testing / evaluating sparse retrieval vs dense retrieval.

    Parameters
    ----------
    store_docs : Cache document strings in SearchHit results (default True).
    """

    def __init__(self, *, store_docs: bool = True) -> None:
        self._keep:     bool                              = store_docs
        self._index:    Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self._documents: List[Optional[str]]              = []
        self._n_vectors: int                              = 0
        self._fitted:    bool                             = False

    def index(
        self,
        vectors:   Iterable[Dict[str, float]],
        documents: Optional[Iterable[str]] = None,
    ) -> "SparseVectorIndex":
        """Index sparse vectors.

        Parameters
        ----------
        vectors   : Iterable of ``{term: weight}`` dicts.  Zero-weight entries
                    are skipped.
        documents : Optional text labels stored in each SearchHit.
        """
        self._index.clear()
        vecs = list(vectors)
        docs = list(documents) if documents is not None else [None] * len(vecs)
        if len(docs) != len(vecs):
            docs = (docs + [None] * len(vecs))[: len(vecs)]
        for doc_id, vec in enumerate(vecs):
            for term, weight in vec.items():
                if weight != 0.0:
                    self._index[term].append((doc_id, weight))
        self._documents  = docs
        self._n_vectors  = len(vecs)
        self._fitted     = True
        return self

    def search(
        self,
        query: Dict[str, float],
        *,
        top_k: int = 10,
    ) -> SearchResult:
        """Retrieve documents by sparse dot-product score.

        Parameters
        ----------
        query : ``{term: weight}`` dict representing the sparse query vector.
        top_k : Number of top-scoring documents to return.
        """
        if not self._fitted:
            raise RuntimeError(
                "SparseVectorIndex has not been indexed. Call index() first."
            )
        q_repr = repr(list(query.items())[:4])
        if not query or self._n_vectors == 0:
            return SearchResult(
                query=q_repr, hits=[], engine="SparseVectorIndex",
                total_docs_searched=self._n_vectors,
            )
        scores: Dict[int, float] = defaultdict(float)
        for qterm, qweight in query.items():
            if qweight == 0.0:
                continue
            for doc_id, dweight in self._index.get(qterm, []):
                scores[doc_id] += qweight * dweight
        top  = heapq.nlargest(top_k, scores.items(), key=lambda kv: kv[1])
        hits = [
            SearchHit(
                doc_id=doc_id, score=score,
                document=self._documents[doc_id],
                metadata={"n_query_terms": len(query)},
            )
            for doc_id, score in top
        ]
        return SearchResult(
            query=q_repr + "...", hits=hits, engine="SparseVectorIndex",
            total_docs_searched=len(scores),
        )

    @property
    def n_vectors(self) -> int:
        return self._n_vectors

    @property
    def vocabulary(self) -> List[str]:
        """Sorted list of all indexed terms."""
        return sorted(self._index.keys())


# ════════════════════════════════════════════════════════════════════════════
# Payload builders
# ════════════════════════════════════════════════════════════════════════════

def build_search_payload(result: SearchResult) -> Dict:
    """Return a JSON-serialisable dict for API / frontend consumption."""
    return {
        "screen":            "Search Results",
        "engine":            result.engine,
        "query":             result.query,
        "nHits":             result.n_hits,
        "totalDocsSearched": result.total_docs_searched,
        "topScore":          round(result.top_hit.score, 6) if result.top_hit else None,
        "hits": [
            {
                "rank":     rank,
                "docId":    hit.doc_id,
                "score":    round(hit.score, 6),
                "snippet":  (hit.document or "")[:120],
                "metadata": hit.metadata,
            }
            for rank, hit in enumerate(result.hits, start=1)
        ],
        "metadata": result.metadata,
    }


def build_a2ui_payload(result: SearchResult) -> Dict:
    """Return a UI-component-ready payload (4 StatCards + ResultList)."""
    return {
        "screen": "Search Results",
        "components": [
            {"type": "StatCard", "label": "Engine",        "value": result.engine},
            {"type": "StatCard", "label": "Hits",          "value": result.n_hits},
            {"type": "StatCard", "label": "Docs Searched", "value": result.total_docs_searched},
            {
                "type":  "StatCard",
                "label": "Top Score",
                "value": round(result.top_hit.score, 4) if result.top_hit else "N/A",
            },
            {
                "type":  "ResultList",
                "label": f"Top results for '{result.query}'",
                "items": [
                    {
                        "rank":    rank,
                        "docId":   hit.doc_id,
                        "score":   round(hit.score, 4),
                        "snippet": (hit.document or "")[:80],
                    }
                    for rank, hit in enumerate(result.hits[:10], start=1)
                ],
            },
        ],
    }


# ════════════════════════════════════════════════════════════════════════════
# Demo
# ════════════════════════════════════════════════════════════════════════════

def _demo() -> None:
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "a fast red fox leaped across the sleeping hound",
        "machine learning and natural language processing",
        "search engines rank documents by relevance",
        "the dog barked at the fox in the garden",
        "natural language understanding drives modern AI",
        "information retrieval with BM25 and TF-IDF",
        "the brown bear and the black bear share habitat",
        "approximate nearest neighbour search via LSH hashing",
        "trie data structures enable fast prefix search",
    ]

    print("=" * 70)
    print(f"  Search Template  v{__version__}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # InvertedIndex -- ranked, phrase, boolean
    # ------------------------------------------------------------------
    print("\n-- InvertedIndex -------------------------------------------------")
    idx = InvertedIndex()
    idx.index(corpus)
    r = idx.search("fox dog", top_k=3)
    print(f"  Ranked   'fox dog'          -> {r}")
    for h in r.hits:
        print(f"    rank {r.hits.index(h)+1}: [{h.doc_id}] score={h.score:.3f}  {(h.document or '')[:55]!r}")

    r_ph = idx.phrase_search("brown fox", top_k=3)
    print(f"\n  Phrase   'brown fox'        -> {r_ph.n_hits} hit(s)")
    for h in r_ph.hits:
        print(f"    [{h.doc_id}] count={int(h.score)} {(h.document or '')[:55]!r}")

    r_bool = idx.boolean_search("fox AND NOT bear", top_k=3)
    print(f"\n  Boolean  'fox AND NOT bear' -> {r_bool.n_hits} hit(s)")
    for h in r_bool.hits:
        print(f"    [{h.doc_id}] score={h.score:.3f}  {(h.document or '')[:55]!r}")

    # ------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------
    print("\n-- BM25SearchEngine ----------------------------------------------")
    bm25 = BM25SearchEngine(k1=1.5, b=0.75)
    bm25.index(corpus)
    r_bm = bm25.search("language processing", top_k=3)
    for h in r_bm.hits:
        print(f"  [{h.doc_id}] {h.score:.4f}  {(h.document or '')[:60]!r}")

    # BM25+
    bm25p = BM25SearchEngine(k1=1.5, b=0.75, delta=1.0)
    bm25p.index(corpus)
    r_bmp = bm25p.search("language processing", top_k=2)
    print(f"  BM25+ top hit: [{r_bmp.top_hit.doc_id}] {r_bmp.top_hit.score:.4f}")

    # ------------------------------------------------------------------
    # TFIDFSearchEngine
    # ------------------------------------------------------------------
    print("\n-- TFIDFSearchEngine ---------------------------------------------")
    tfidf = TFIDFSearchEngine(sublinear_tf=True)
    tfidf.index(corpus)
    r_tf = tfidf.search("search retrieval ranking", top_k=3)
    for h in r_tf.hits:
        print(f"  [{h.doc_id}] {h.score:.4f}  {(h.document or '')[:60]!r}")

    # ------------------------------------------------------------------
    # FuzzySearchEngine
    # ------------------------------------------------------------------
    print("\n-- FuzzySearchEngine ---------------------------------------------")
    fuzzy = FuzzySearchEngine(max_edits=1)
    fuzzy.index(corpus)
    r_fz = fuzzy.search("laguage procesing", top_k=3)  # intentional typos
    print(f"  Query 'laguage procesing' (typos) -> {r_fz.n_hits} hit(s)")
    print(f"  Expanded terms: {r_fz.metadata.get('expanded_sample', [])[:6]}")
    for h in r_fz.hits:
        print(f"  [{h.doc_id}] {h.score:.4f}  {(h.document or '')[:60]!r}")

    # ------------------------------------------------------------------
    # TrieSearch + autocomplete
    # ------------------------------------------------------------------
    print("\n-- TrieSearch ----------------------------------------------------")
    trie = TrieSearch(prefix_only=True)
    trie.index(corpus)
    completions = trie.autocomplete("nat", top_k=5)
    print(f"  autocomplete('nat') -> {completions}")
    r_tr = trie.search("natur lang", top_k=3)
    for h in r_tr.hits:
        print(f"  [{h.doc_id}] {h.score:.3f}  {(h.document or '')[:60]!r}")

    # ------------------------------------------------------------------
    # MultiSearchEngine (RRF ensemble)
    # ------------------------------------------------------------------
    print("\n-- MultiSearchEngine (RRF) ---------------------------------------")
    multi = MultiSearchEngine(
        [InvertedIndex(), BM25SearchEngine(), TFIDFSearchEngine()],
        mode="rrf",
    )
    multi.index(corpus)
    r_mul = multi.search("fox jumps", top_k=3)
    for h in r_mul.hits:
        print(f"  [{h.doc_id}] rrf={h.score:.5f}  {(h.document or '')[:55]!r}")

    # ------------------------------------------------------------------
    # Vector search engines
    # ------------------------------------------------------------------
    print("\n-- Vector Search Engines -----------------------------------------")
    rng = random.Random(42)
    vectors = [[rng.gauss(0.0, 1.0) for _ in range(8)] for _ in range(len(corpus))]
    vectors = [_l2_normalize(v) for v in vectors]
    query_vec = _l2_normalize([rng.gauss(0.0, 1.0) for _ in range(8)])

    lin = LinearVectorSearch(metric="cosine")
    lin.index(vectors, corpus)
    r_lin = lin.search_vector(query_vec, top_k=3)
    print(f"  LinearVectorSearch top-3:  {r_lin.get_doc_ids()}")

    lsh = LSHSearch(n_tables=4, n_hashes=6, random_seed=0)
    lsh.index(vectors, corpus)
    r_lsh = lsh.search_vector(query_vec, top_k=3)
    print(f"  LSHSearch      candidates: {r_lsh.metadata['n_candidates']}  "
          f"top-3: {r_lsh.get_doc_ids()}")

    kdt = KDTreeSearch(metric="euclidean")
    kdt.index(vectors, corpus)
    r_kd = kdt.search_vector(query_vec, top_k=3)
    print(f"  KDTreeSearch   top-3:      {r_kd.get_doc_ids()}")

    # ------------------------------------------------------------------
    # QueryParser
    # ------------------------------------------------------------------
    print("\n-- QueryParser ---------------------------------------------------")
    qp = QueryParser()
    for q in [
        "fox AND dog",
        "fox OR bear",
        "(fox OR bear) AND NOT hound",
        "language processing AI",
    ]:
        print(f"  {q!r:42s} -> {qp.parse(q)}")

    # ------------------------------------------------------------------
    # RRFRanker
    # ------------------------------------------------------------------
    print("\n-- RRFRanker -----------------------------------------------------")
    ranker = RRFRanker(k=60)
    fused  = ranker.fuse([[0, 1, 2], [2, 0, 3], [1, 2, 0]], top_k=4)
    print(f"  fuse([[0,1,2],[2,0,3],[1,2,0]]) -> {fused}")

    # ------------------------------------------------------------------
    # ScoreNormaliser
    # ------------------------------------------------------------------
    print("\n-- ScoreNormaliser -----------------------------------------------")
    raw_scores = [3.5, 1.0, 7.2, 0.0, 5.1]
    print(f"  raw         : {raw_scores}")
    print(f"  min_max     : {[round(s, 3) for s in ScoreNormaliser.min_max(raw_scores)]}")
    print(f"  z_score     : {[round(s, 3) for s in ScoreNormaliser.z_score(raw_scores)]}")
    print(f"  softmax(T=1): {[round(s, 3) for s in ScoreNormaliser.softmax(raw_scores)]}")
    print(f"  rank_recip  : {ScoreNormaliser.rank_reciprocal(raw_scores)}")

    # ------------------------------------------------------------------
    # HybridSearchEngine (sparse BM25 + dense cosine)
    # ------------------------------------------------------------------
    print("\n-- HybridSearchEngine -------------------------------------------")
    h_bm25 = BM25SearchEngine()
    h_lin  = LinearVectorSearch(metric="cosine")
    hybrid = HybridSearchEngine(h_bm25, h_lin, mode="rrf")
    hybrid.index(corpus, vectors)
    r_hyb  = hybrid.search("language processing", query_vec, top_k=3)
    print(f"  RRF hybrid 'language processing' -> {r_hyb.n_hits} hit(s)")
    for h in r_hyb.hits:
        print(f"  [{h.doc_id}] rrf={h.score:.5f}  {(h.document or '')[:55]!r}")

    # ------------------------------------------------------------------
    # BM25F (field-weighted BM25)
    # ------------------------------------------------------------------
    print("\n-- BM25F --------------------------------------------------------")
    structured_docs = [
        {"title": "BM25 ranking formula",       "body": "probabilistic relevance model"},
        {"title": "TF-IDF cosine similarity",   "body": "vector space model retrieval"},
        {"title": "neural information retrieval","body": "dense embeddings and transformers"},
        {"title": "inverted index structures",  "body": "postings lists skip pointers"},
        {"title": "language model retrieval",   "body": "query likelihood smoothing"},
    ]
    bm25f = BM25F(
        field_config={"title": {"b": 0.4, "boost": 2.5},
                      "body":  {"b": 0.75, "boost": 1.0}},
    )
    bm25f.index(structured_docs)
    r_f = bm25f.search("retrieval ranking", top_k=3)
    print(f"  Query 'retrieval ranking' -> {r_f.n_hits} hit(s)")
    for h in r_f.hits:
        print(f"  [{h.doc_id}] {h.score:.4f}  {(h.document or '')[:60]!r}")

    # ------------------------------------------------------------------
    # MaximalMarginalRelevance (diversity reranking)
    # ------------------------------------------------------------------
    print("\n-- MaximalMarginalRelevance -------------------------------------")
    mmr     = MaximalMarginalRelevance(lambda_=0.5)
    r_lin3  = lin.search_vector(query_vec, top_k=6)
    r_mmr   = mmr.rerank_results(query_vec, r_lin3, vectors, top_k=3)
    print(f"  Linear top-6 ids : {r_lin3.get_doc_ids()}")
    print(f"  MMR    top-3 ids : {r_mmr.get_doc_ids()}  (reranked for diversity)")

    # ------------------------------------------------------------------
    # SparseVectorIndex (SPLADE-style sparse retrieval)
    # ------------------------------------------------------------------
    print("\n-- SparseVectorIndex --------------------------------------------")
    sparse_docs = [
        {"fox": 1.2, "quick": 0.8, "jumps": 1.0},
        {"learning": 1.5, "language": 1.3, "model": 1.1},
        {"search": 1.4, "retrieval": 1.2, "ranking": 0.9},
        {"fox": 0.6, "dog": 1.0, "garden": 0.8},
    ]
    svi = SparseVectorIndex()
    svi.index(sparse_docs, ["fox corpus", "NLP corpus", "IR corpus", "fox+dog corpus"])
    r_sv = svi.search({"fox": 1.0, "quick": 0.5}, top_k=3)
    print(f"  Query {{fox:1.0, quick:0.5}} -> {r_sv.n_hits} hit(s)")
    for h in r_sv.hits:
        print(f"  [{h.doc_id}] {h.score:.4f}  {h.document!r}")

    # ------------------------------------------------------------------
    # Payload builders
    # ------------------------------------------------------------------
    print("\n-- Payload builders ----------------------------------------------")
    payload = build_search_payload(r_bm)
    print(f"  build_search_payload keys : {list(payload.keys())}")
    a2ui    = build_a2ui_payload(r_bm)
    print(f"  build_a2ui_payload types  : {[c['type'] for c in a2ui['components']]}")

    print("\n" + "=" * 70)
    print("  OK  All demos completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
