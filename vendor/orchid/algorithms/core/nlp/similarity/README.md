# NLP Similarity

Pure-Python, zero-dependency text similarity — fifteen strategies, phonetic matching, BM25 relevance, and composable result types.

## Pick

| Goal | Use |
|---|---|
| Typo / edit distance (insert, delete, substitute) | `LevenshteinSimilarity` |
| Edit distance + adjacent transpositions | `DamerauLevenshtein` |
| Position-wise character diff (equal-length strings) | `HammingSimilarity` |
| Character match + transposition ratio | `JaroSimilarity` |
| Jaro boosted for common prefix | `JaroWinklerSimilarity` |
| Token set overlap (Jaccard) | `JaccardSimilarity` |
| Token set overlap (Sørensen-Dice) | `DiceSimilarity` |
| Subset containment | `OverlapSimilarity` |
| TF bag-of-words cosine | `CosineSimilarity` |
| Longest common subsequence ratio | `LCSSimilarity` |
| Character n-gram overlap (default trigrams) | `NgramSimilarity` |
| Gestalt pattern matching (recursive blocks) | `RatcliffObershelpSimilarity` |
| Phonetic bucket equality (Soundex) | `SoundexSimilarity` |
| Phonetic similarity (Simplified Metaphone) | `MetaphoneSimilarity` |
| IR relevance score (a = query, b = document) | `BM25Similarity` |
| Frontend JSON | `build_similarity_payload`, `build_a2ui_payload` |

## Install

```bash
# No runtime deps — copy the file.
pip install pytest  # tests only
```

## Quick start

```python
from similarity_template import LevenshteinSimilarity, JaroWinklerSimilarity, CosineSimilarity

# Plain float
LevenshteinSimilarity().similarity("kitten", "sitting")   # 0.5714...

# Rich result object
result = CosineSimilarity().compare("the cat sat on the mat", "the dog sat on the log")
print(result.score)        # 0.6666...
print(result.is_identical) # False

# Every strategy shares the same interface
for cls in [LevenshteinSimilarity, JaroWinklerSimilarity, CosineSimilarity]:
    s = cls().similarity("hello", "helo")
    print(f"{cls.__name__:30s} {s:.4f}")
```

## API

### All strategies — shared interface

```python
strat.similarity(a, b) -> float          # [0, 1], higher = more similar
strat.distance(a, b)   -> float          # [0, 1], lower = more similar
strat.compare(a, b)    -> SimilarityResult
```

### String-edit metrics

| Class | Notes |
|---|---|
| `LevenshteinSimilarity()` | Normalised by `max(len(a), len(b))` |
| `DamerauLevenshtein()` | OSA; normalised same as Levenshtein |
| `HammingSimilarity()` | Returns 0.0 if lengths differ |
| `JaroSimilarity()` | Match window = `floor(max_len/2) - 1` |
| `JaroWinklerSimilarity(p=0.1, max_prefix_length=4)` | Constraint: `p * max_prefix_length <= 1.0` |

```python
from similarity_template import JaroWinklerSimilarity

jw = JaroWinklerSimilarity(p=0.1, max_prefix_length=4)
jw.similarity("MARTHA", "MARHTA")   # 0.9611...

# Custom prefix length — constraint still enforced
jw2 = JaroWinklerSimilarity(p=0.1, max_prefix_length=8)  # 0.1*8=0.8 ≤ 1.0
# JaroWinklerSimilarity(p=0.2, max_prefix_length=6)      # raises — 0.2*6=1.2 > 1.0
```

### Token-set metrics

All tokenise with `tokenize()` (lowercase, strip punctuation, split on whitespace).

| Class | Formula |
|---|---|
| `JaccardSimilarity()` | `\|A∩B\| / \|A∪B\|` |
| `DiceSimilarity()` | `2\|A∩B\| / (\|A\|+\|B\|)` |
| `OverlapSimilarity()` | `\|A∩B\| / min(\|A\|,\|B\|)` |
| `CosineSimilarity()` | TF-weighted bag cosine |

```python
from similarity_template import JaccardSimilarity

JaccardSimilarity().similarity("the cat sat", "the dog sat")  # 0.5
```

### Sequence metrics

| Class | Notes |
|---|---|
| `LCSSimilarity()` | Normalised by `max(len(a), len(b))` |
| `NgramSimilarity(n=3)` | Dice over character n-gram bags; `n >= 1` |
| `RatcliffObershelpSimilarity()` | `2 * matching_chars / total_chars` |

```python
from similarity_template import NgramSimilarity

NgramSimilarity(n=2).similarity("night", "nacht")  # bigram overlap
NgramSimilarity(n=1).similarity("abc", "cba")      # 1.0 — unigrams ignore order
```

### Phonetic metrics

Both operate on the **first token** of each input; `compare()` exposes the codes in `metadata`.

```python
from similarity_template import SoundexSimilarity, MetaphoneSimilarity

SoundexSimilarity().similarity("Robert", "Rupert")    # 1.0 — both R163
MetaphoneSimilarity().similarity("phone", "fone")     # 1.0 — both FN

r = SoundexSimilarity().compare("Smith", "Smythe")
r.metadata   # {"soundex_a": "S530", "soundex_b": "S530"}
```

### BM25Similarity

BM25 is **asymmetric**: `a` is the query, `b` is the document. `similarity(a, b) != similarity(b, a)` in general.

```python
from similarity_template import BM25Similarity

bm25 = BM25Similarity(k1=1.5, b=0.75)

# Short query against longer document scores higher than the reverse
bm25.similarity("cat", "the cat sat on the mat")       # > 0
bm25.similarity("the cat sat on the mat", "cat")       # < above
```

### SimilarityResult

```python
result.score        # float  [0, 1]
result.distance     # float  [0, 1]
result.strategy     # str    class name
result.text_a       # str
result.text_b       # str
result.metadata     # Dict   strategy-specific extras
result.is_identical # bool   score >= 1.0 (cached)
result.is_dissimilar# bool   score <= 0.0 (cached)
```

### Raw utilities

```python
levenshtein_distance(a, b)         -> int    # raw edit ops
damerau_levenshtein_distance(a, b) -> int    # OSA edit ops
hamming_distance(a, b)             -> int    # position mismatches
jaro_similarity(a, b)              -> float  # [0, 1]
lcs_length(a, b)                   -> int    # longest common subsequence
soundex_code(word)                 -> str    # e.g. "R163"
metaphone_code(word)               -> str    # e.g. "FN"
ngram_set(text, n)                 -> Counter
tokenize(text)                     -> List[str]
```

### Payload builders

```python
payload = build_similarity_payload(result)
# {"strategy", "score", "distance", "isIdentical", "isDissimilar",
#  "textA", "textB", "metadata", "summary"}

a2ui = build_a2ui_payload(result)
# 4 StatCards (Strategy, Score, Distance, Identical) + SimilarityCard
```

## Tests / Demo

```bash
pytest _stress_test.py -v
python similarity_template.py
```
