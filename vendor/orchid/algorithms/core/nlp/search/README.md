# NLP Search

Pure-Python, zero-dependency search engine toolkit — thirteen strategies, boolean query parsing, hybrid retrieval, diversity reranking, and ANN vector search.

## Pick

| Goal | Use |
|---|---|
| Ranked full-text retrieval (TF-IDF) | `InvertedIndex` |
| Exact phrase search | `InvertedIndex.phrase_search` |
| Boolean AND / OR / NOT retrieval | `InvertedIndex.boolean_search` |
| Industry-standard relevance ranking | `BM25SearchEngine` |
| BM25+ (long-document boost) | `BM25SearchEngine(delta=1.0)` |
| Field-weighted BM25 (title/body/tags) | `BM25F` |
| Cosine TF-IDF retrieval | `TFIDFSearchEngine` |
| Typo-tolerant approximate matching | `FuzzySearchEngine` |
| Prefix search / autocomplete | `TrieSearch` |
| Combine multiple text engines | `MultiSearchEngine` |
| Sparse+dense hybrid retrieval (RAG / semantic search) | `HybridSearchEngine` |
| Exact nearest-neighbour (dense vectors) | `LinearVectorSearch` |
| Approximate nearest-neighbour (large corpora) | `LSHSearch` |
| Exact nearest-neighbour (low-dim vectors) | `KDTreeSearch` |
| Diversity-aware reranking (MMR) | `MaximalMarginalRelevance` |
| Learned sparse / feature-weight retrieval | `SparseVectorIndex` |
| Parse boolean query strings | `QueryParser` |
| Fuse ranked lists from multiple sources | `RRFRanker` |
| Normalise relevance scores | `ScoreNormaliser` |
| Frontend JSON | `build_search_payload`, `build_a2ui_payload` |

## Install

```bash
# No runtime deps — copy the file.
pip install pytest  # tests only
```

## Quick start

```python
from search_template import BM25SearchEngine

corpus = [
    "the quick brown fox jumps over the lazy dog",
    "machine learning and natural language processing",
    "search engines rank documents by relevance",
]

engine = BM25SearchEngine(k1=1.5, b=0.75)
engine.index(corpus)

result = engine.search("language processing", top_k=3)
for hit in result.hits:
    print(hit.doc_id, f"{hit.score:.4f}", hit.document[:60])
```

## API

### Text search engines — `index / search`

All text engines share `BaseSearchEngine` and the same two-step API.

| Class | Strategy | Key params |
|---|---|---|
| `InvertedIndex` | Log-TF × smooth-IDF | `lowercase`, `store_docs` |
| `BM25SearchEngine` | Okapi BM25 / BM25+ | `k1`, `b`, `delta`, `epsilon` |
| `TFIDFSearchEngine` | Cosine TF-IDF | `tf_scheme`, `idf_scheme`, `sublinear_tf`, `norm`, `min_df`, `max_df` |
| `FuzzySearchEngine` | Edit-distance expansion | `max_edits`, `min_term_len`, `prefix_len` |
| `TrieSearch` | Trie prefix match | `prefix_only` |
| `MultiSearchEngine` | Ensemble fusion | `engines`, `mode` ("rrf"\|"weighted"), `weights`, `rrf_k` |

```python
from search_template import InvertedIndex

idx = InvertedIndex()
idx.index(corpus)

# Ranked retrieval
idx.search("fox dog", top_k=5)

# Exact phrase
idx.phrase_search("brown fox", top_k=5)

# Boolean
idx.boolean_search("fox AND NOT bear", top_k=5)
```

```python
from search_template import TFIDFSearchEngine

tfidf = TFIDFSearchEngine(
    tf_scheme="log",    # "raw" | "log" | "freq" | "bool"
    idf_scheme="smooth", # "smooth" | "standard"
    sublinear_tf=True,
    norm="l2",          # "l2" | "l1" | None
    min_df=1,
    max_df=0.95,
)
tfidf.index(corpus)
tfidf.search("retrieval ranking", top_k=5)
tfidf.vocabulary    # Dict[str, int]
tfidf.idf_          # Dict[str, float]
```

```python
from search_template import FuzzySearchEngine

fuzzy = FuzzySearchEngine(max_edits=1, min_term_len=4)
fuzzy.index(corpus)
fuzzy.search("laguage procesing", top_k=5)  # tolerates typos
```

```python
from search_template import TrieSearch

trie = TrieSearch(prefix_only=True)
trie.index(corpus)
trie.autocomplete("nat", top_k=5)   # ["natural", ...]
trie.search("natur lang", top_k=5)  # prefix on last token
```

```python
from search_template import MultiSearchEngine, InvertedIndex, BM25SearchEngine

multi = MultiSearchEngine(
    [InvertedIndex(), BM25SearchEngine()],
    mode="rrf",   # Reciprocal Rank Fusion
)
multi.index(corpus)
multi.search("fox jumps", top_k=5)
```

### Vector search engines — `index / search_vector`

All vector engines share `BaseVectorSearchEngine`.

| Class | Strategy | Key params |
|---|---|---|
| `LinearVectorSearch` | Exact brute-force | `metric` ("cosine"\|"euclidean"\|"dot") |
| `LSHSearch` | Random-hyperplane ANN | `n_tables`, `n_hashes`, `random_seed` |
| `KDTreeSearch` | KD-tree exact NN | `metric` ("euclidean"\|"cosine") |

```python
from search_template import LinearVectorSearch, LSHSearch, KDTreeSearch

vectors = [[0.1, 0.9, 0.3], [0.8, 0.2, 0.5], [0.4, 0.6, 0.1]]
query   = [0.2, 0.8, 0.2]

# Exact (baseline)
lin = LinearVectorSearch(metric="cosine")
lin.index(vectors, documents=corpus)
lin.search_vector(query, top_k=2)

# Approximate (large corpora)
lsh = LSHSearch(n_tables=8, n_hashes=8, random_seed=42)
lsh.index(vectors)
lsh.search_vector(query, top_k=2)

# Exact, low-dim (dim <= 20)
kdt = KDTreeSearch(metric="euclidean")
kdt.index(vectors)
kdt.search_vector(query, top_k=2)
```

### SearchResult / SearchHit

```python
result.query               # str
result.hits                # List[SearchHit], sorted descending by score
result.engine              # str — class name
result.n_hits              # int (cached)
result.top_hit             # SearchHit | None (cached)
result.scores              # List[float] (cached)
result.get_doc_ids()       # List[int]
result.total_docs_searched # int
result.metadata            # Dict

hit.doc_id   # int
hit.score    # float  (higher = more relevant)
hit.document # str | None
hit.metadata # Dict  (engine-specific extras)
```

### QueryParser

```python
from search_template import QueryParser

qp   = QueryParser()
tree = qp.parse("(fox OR hound) AND NOT bear")
# AND(OR(TERM('fox'), TERM('hound')), NOT(TERM('bear')))

qp.extract_terms(tree)  # ['fox', 'hound', 'bear']
```

Operators (case-insensitive): `AND`, `OR`, `NOT`. Adjacent terms without an operator are treated as implicit AND.

### RRFRanker

```python
from search_template import RRFRanker

ranker = RRFRanker(k=60)

# From raw ranked lists of doc IDs
ranker.fuse([[0, 1, 2], [2, 0, 3]], top_k=3)
# [(doc_id, rrf_score), ...]

# From SearchResult objects
ranker.fuse_results([result_a, result_b], top_k=5)
```

### ScoreNormaliser

```python
from search_template import ScoreNormaliser

scores = [3.5, 1.0, 7.2, 0.0, 5.1]

ScoreNormaliser.min_max(scores)         # [0, 1]
ScoreNormaliser.z_score(scores)         # zero mean, unit std
ScoreNormaliser.softmax(scores, temperature=1.0)
ScoreNormaliser.rank_reciprocal(scores) # 1/rank
```

### Payload builders

```python
from search_template import build_search_payload, build_a2ui_payload

payload = build_search_payload(result)
# {"screen", "engine", "query", "nHits", "totalDocsSearched",
#  "topScore", "hits": [{"rank", "docId", "score", "snippet", "metadata"}],
#  "metadata"}

a2ui = build_a2ui_payload(result)
# 4 StatCards (Engine, Hits, Docs Searched, Top Score) + ResultList
```

### HybridSearchEngine — `index / search`

Combines a sparse text engine with a dense vector engine. The dominant architecture for production semantic search and RAG systems.

```python
from search_template import HybridSearchEngine, BM25SearchEngine, LinearVectorSearch

hybrid = HybridSearchEngine(
    BM25SearchEngine(),
    LinearVectorSearch(metric="cosine"),
    mode="rrf",           # "rrf" (default) or "weighted"
    text_weight=0.5,      # used only in "weighted" mode
    vector_weight=0.5,
)
hybrid.index(corpus, vectors)         # len(corpus) == len(vectors)
result = hybrid.search("query text", query_vector=[...], top_k=10)
```

### BM25F — `index / search`

Field-weighted BM25 for structured documents. Each field contributes independently with its own length normalisation (`b`) and importance boost.

```python
from search_template import BM25F

docs = [
    {"title": "Python search engines", "body": "BM25 ranking formulas explained"},
    {"title": "Machine learning",       "body": "neural retrieval with transformers"},
]
bm25f = BM25F(
    field_config={
        "title": {"b": 0.4, "boost": 2.5},   # title matches count 2.5× more
        "body":  {"b": 0.75, "boost": 1.0},
    },
    k1=1.5,
)
bm25f.index(docs)
bm25f.search("search ranking", top_k=5)
bm25f.idf_          # Dict[str, float]
bm25f.n_docs        # int
```

### MaximalMarginalRelevance — `rerank / rerank_results`

Greedy diversity-aware reranking. Prevents returning near-duplicate results from dense retrieval outputs.

```python
from search_template import MaximalMarginalRelevance, LinearVectorSearch

mmr = MaximalMarginalRelevance(
    lambda_=0.5,   # 0 = max diversity, 1 = max relevance
)

# Low-level: pass vectors directly
reranked = mmr.rerank(query_vector, candidate_vectors, candidate_ids, top_k=5)
# [(doc_id, mmr_score), ...]

# High-level: wrap a SearchResult
lin = LinearVectorSearch(metric="cosine")
lin.index(vectors)
result = lin.search_vector(query_vector, top_k=20)
diverse = mmr.rerank_results(query_vector, result, vectors, top_k=5)
```

### SparseVectorIndex — `index / search`

Dot-product retrieval over explicit `{term: weight}` sparse vectors. Use for learned sparse representations (SPLADE, UniCOIL) or custom term-weighting.

```python
from search_template import SparseVectorIndex

# Documents as sparse weight dicts (e.g. from a SPLADE model)
sparse_docs = [
    {"fox": 1.2, "quick": 0.8, "animal": 0.6},
    {"neural": 1.5, "language": 1.3, "model": 1.1},
]
svi = SparseVectorIndex()
svi.index(sparse_docs, documents=corpus)
svi.search({"fox": 1.0, "animal": 0.5}, top_k=5)
svi.vocabulary    # List[str] — all indexed terms
svi.n_vectors     # int
```

## Tests / Demo

```bash
pytest _stress_test.py -v
python search_template.py
```
