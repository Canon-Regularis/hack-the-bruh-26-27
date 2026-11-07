# Public Voice

A consultation is a spatial signal before it is a pile of comments. People pin
worries to places — the crossing outside the school, the rat-run past the corner
shop, the bus station that makes a child's asthma worse — and a planner needs that
signal in the same coordinates the optimiser uses: **a value per street per theme**.
This module turns located comments into the MOSAIC concern field

```
s_e^j = Σ_i sev_i · K(dist(e, loc_i)) · 1[theme_i = j]
```

pairs each theme with the network metric that answers it, and assembles the
public-voice objective

```
J_voice(x) = Σ_j Σ_e s_e^j · m_e^j(x)
```

with baseline normalisation, so "did this design respond to what people said?" is a
number on the Pareto vector. Themes with no measurable counterpart (parking, by
default) are drawn on the heat map but **excluded from `J_voice`** — the paper's
explicit rule.

It also owns the statistics that make comments trustworthy: **near-duplicate
collapse** so a copy-paste campaign is not mistaken for a groundswell, **hotspot
tests with false-discovery control** so a red blob on the heat map is not noise, and
**self-selection diagnostics** so the loudest postcode is not mistaken for the whole
town — plus **anonymisation at ingestion**, so none of it ever touches a name.

This is deliberately not `urban/policy_compiler`, which binds the *written*
constraints of a plan (TROs, policy clauses) to network elements; this module handles
the *spoken* voice — many short, located, noisy, duplicated, self-selected comments.
Nor is it `urban/accessibility_equity`, whose Moran's I runs over accessibility
scores; here the values are concern intensities, and the Getis-Ord family, LISA
classes and hotspot-to-comment bookkeeping are this module's own.

```python
from public_voice_template import (network_from_grid, ingest_concerns,
    find_near_duplicates, collapse_duplicates, concern_field, voice_objective,
    voice_improvement)

net = network_from_grid(5, 4, 120.0)                 # 20 junctions, 31 streets
raw = [
    {"id": "a", "text": "Kids wait ages at the zebra crossing by the school",
     "severity": 4, "x": 240, "y": 125, "author": "Ann"},
    {"id": "b", "text": "kids wait AGES at the zebra crossing by the school!!",
     "severity": 4, "x": 236, "y": 118, "author": "Bob"},
    {"id": "c", "text": "Cars speeding past the school, near miss. Call 07700 900123",
     "severity": 5, "x": 250, "y": 130, "author": "Cy"},
    {"id": "d", "text": "Pavement parking blocks the footway",
     "severity": 3, "x": 300, "y": 300, "author": "Di"},
]
ing = ingest_concerns(raw, salt="district-salt")      # PII dropped, text redacted
print([(c.cid, c.theme) for c in ing.records])
# [('a', 'crossing_delay'), ('b', 'crossing_delay'), ('c', 'road_safety'), ('d', 'parking')]
print(ing.records[2].text)    # Cars speeding past the school, near miss. Call [PHONE]

dup = find_near_duplicates({c.cid: c.text for c in ing.records}, "canonical")
print(dup.groups)                                      # [['a', 'b']]
cs = collapse_duplicates(ing.records, dup, policy="sqrt")

f = concern_field(net, cs, kernel="quartic", bandwidth=120.0)
print(f.edge_value("h2_1"), f.edge_value("h2_1", "crossing_delay"))   # 10.568 5.6372

base = {m: {e: 10.0 for e in net.edge_ids}
        for m in ("ped_crossing_delay_s", "collision_risk")}
design = {m: {e: (6.0 if e == "h2_1" else 10.0) for e in net.edge_ids} for m in base}
j = voice_objective(f, design, base)
print(j.total, j.excluded_themes)                      # 38.762 ['parking', 'other']
print(voice_improvement(f, base, base))                # 0.0  (exactly)
print(voice_improvement(f, design, base))              # 0.0983
```

Coordinates are projected metres, bandwidths metres, severity any non-negative score.
Everything stochastic takes a `seed`; no builtin `hash()` is used anywhere, so output
is byte-identical and independent of `PYTHONHASHSEED`.

---

## Pick

| Goal | Use |
| --- | --- |
| Take raw submissions in without leaking a name | `ingest_concerns` (+ `redact_text`, `pseudonymise`) |
| Blur pins so a front door can't be found | `coarsen_location` / `jitter_location` / `k_anonymity_cells` |
| Put an untagged comment in a theme | `classify_theme` |
| A copy-paste campaign — count it how? | `find_near_duplicates` + `collapse_duplicates(policy=...)` |
| Concern intensity per street per theme | `concern_field` |
| …along the network rather than as the crow flies | `concern_field(distance="network")` |
| Which comments drive this street's score | `concern_contributions` |
| A smooth heat-map raster | `concern_surface` |
| A bandwidth that isn't a guess | `spatial_silverman_bandwidth` / `cv_bandwidth` / `abramson_bandwidths` |
| "Did the design respond to the public?" | `voice_objective` / `voice_improvement` |
| Is the concern clustered at all? | `morans_i` / `gearys_c` / `general_g` / `ripleys_k` |
| Where are the hotspots, honestly? | `local_getis_ord` / `local_morans_i` (BH-FDR on by default) |
| How many comments sit behind each hotspot | `hotspot_clusters` + `comment_counts_per_hotspot` |
| Colour classes for the map | `jenks_breaks` / `heatmap_classes` / `bin_points_hex` |
| Who answered, and who didn't | `representativeness` / `spatial_representation` |
| Reweight to the population | `post_stratification_weights` / `rake_weights` + `kish_effective_sample_size` |
| JSON a browser can render | `build_public_voice_payload` / `build_a2ui_payload` |

---

## Install

Nothing to install. `public_voice_template.py` is a single zero-dependency file —
copy it next to your code and import it. CPython 3.8+, standard library only. The one
optional import, the core library's `p_adjust` for Benjamini-Hochberg, is guarded and
has a bit-identical local fallback.

```bash
python public_voice_template.py     # the demo, prints "demo OK"
pytest _stress_test.py -q           # 481 oracle-based tests
```

---

## What's inside

### 0 — Validation

Every entry point coerces its arguments through shared finite / positive /
non-negative / unit-interval / integer helpers and raises `PublicVoiceError` (a
`ValueError` subclass) naming the offender — including NaN, infinities, bools passed
as numbers, empty inputs and unknown enum values — so nothing malformed reaches a
solver and no bare `TypeError` or `ZeroDivisionError` escapes.

### 1 — Themes and concern records

`THEMES` holds the paper's nine themes — crossing delay, road safety, rat-running, air
quality, missing cycle link, accessibility, parking, severance, noise — plus `other`.
`THEME_TAXONOMY` gives each a label, keywords, its default metric and a direction
(accessibility is higher-is-better). `ConcernRecord` is the located concern (theme,
mode, severity, location, source, count, text, pseudonymous respondent, stratum).
`classify_theme` is a deterministic whole-word keyword classifier with ties broken in
`THEMES` order; every keyword classifies to its own theme. Keywords match with their
regular plural (`-s`, `-es`, `-y` → `-ies`); proper street names (capitalised words
ending in a thoroughfare type: "Park Road", "Cross Street") are masked before matching,
except leading capitalised words that hold a keyword ("Crossing Station Road takes
ages" keeps "Crossing"; the last name word and the type are always masked); hyphens
and line breaks inside a phrase are spaces ("near-miss", "cycle-lane", a wrapped
"cycle⏎lane") — but a line break before a capitalised line starts a new list item
("traffic⏎Lights out" is not "traffic lights") — and spellings that then coincide
("rat run" / "rat-run") count once;
the bare noun "park" is not a keyword (a green space as often as a car park — "car
park", "to park", "park on" are) and street *lighting* is not a crossing signal ("traffic
light", "at the lights" are).

### 2 — Anonymisation at ingestion

`ingest_concerns` drops every PII field before anything else sees it, keeps the author
only as an HMAC-SHA256 pseudonym, redacts URLs, e-mails, handles, house-number
addresses, UK postcodes, phone numbers and self-named signatures from the text,
classifies untagged comments from the **redacted** text, and optionally snaps
locations to a grid or donut-jitters them (uniform in area, keyed per record so the
result does not depend on order or hash seed). `k_anonymity_cells` audits a grid.
Phone numbers are caught with `+44`, a leading `0` or a bracketed `(020)` area code.
Redaction is regex-based and therefore **incomplete by construction** (pinned by a
test): lower-case self-introductions (`my name is jane doe`), lower-case street
addresses, IP addresses, bare domains and dates of birth are not recognised. Text from
an uncontrolled source needs a human or a trained NER pass before publication.

### 3 — Network and attachment

`VoiceNetwork` (or `network_from_grid`) is a small undirected street graph. One-way
streets are deliberately not modelled: a concern spreads along a street to the people
who walk past it, whichever way the traffic flows, so network kernels use the
undirected distance (a directed graph would make the field depend on traffic rules
rather than on where people are).
`attach_point` projects a pin onto its nearest edge (ties on edge id) and an end-point
projection is exact, so a pin on a node is at distance exactly 0. `dijkstra` is
multi-source with a `(distance, node id)` heap; `network_distances_to_edges` gives
the along-network distance from an attached pin to the nearest point of every edge.
With `distance="network"` the field always measures to an edge's nearest point, so
`edge_reference` (segment / midpoint) applies to the Euclidean mode only — and
`concern_field` REJECTS `edge_reference="midpoint"` with network distance, and
`include_snap` / `max_attach_distance` with Euclidean distance, rather than silently
ignoring them.

### 4 — Kernels

Gaussian, Epanechnikov, triangular, quartic, uniform and exponential, each as a peak
profile `k(u)` with `k(0) = 1`, a 1-D density and a 2-D radial density, with
closed-form second moments, roughness, canonical bandwidths and Silverman
efficiencies.

### 5 — Bandwidth selection

1-D and 2-D Silverman (converted between kernels by the canonical-bandwidth ratio),
leave-one-out likelihood CV for any kernel, closed-form least-squares CV for the
Gaussian, Abramson's square-root-law adaptive bandwidths (geometric mean exactly the
pilot) and k-nearest-neighbour bandwidths. The k-NN bandwidth is a distance and does
not depend on the kernel; `knn_bandwidths(kernel=...)` only records which kernel the
bandwidths are meant for. The 1-D rule uses Silverman's `IQR/1.34`; the 2-D
`robust` option uses the standard-normal IQR `1.349` — both published constants, a
0.7 % difference. Leave-one-out likelihood, LSCV, the KDE and the Abramson pilot are
evaluated in bandwidth units (`h²f`, dimensionless), so cross-validated bandwidths and
Abramson factors are scale-free from 1e-160 to 1e160 m; only a density that itself
overflows a float (bandwidth below ~1e-154 m) raises `PublicVoiceError`.

### 6 — The concern field

`concern_field` evaluates `s_e^j` on every edge, Euclidean (to the segment or the
midpoint) or along the network, with a global or per-comment bandwidth, optional
weights (e.g. post-stratification) and an optional cut-off. Every theme is present in
the output, all-zero when nobody raised it. `concern_contributions` explains one
street; `concern_surface` rasterises for the heat map.

### 7 — Theme-metric pairing and `J_voice`

`ThemeMetricPair` says which metric answers a theme and which way is better;
`make_pairing` overrides the defaults. `normalise_metric` turns a metric into a
baseline-normalised burden — per edge (`x/b`, exactly 1 at baseline) or per network
(`x/b̄`). `voice_objective` sums `s · m` over measurable themes and reports the
excluded themes' field mass separately; `voice_improvement` is `1 − J(x)/J(baseline)`,
exactly 0 at baseline. `heatmap_layers` emits every (street, theme) cell flagged
`in_objective`.

### 8 — Near-duplicate detection

FNV-1a 32/64 (published test vectors), `canonical_form` (NFKC, case-fold, punctuation,
abbreviations), character and word shingles, exact Jaccard, `MinHasher` (universal
hashing mod 2^61−1 of FNV-hashed shingles passed through the SplitMix64 finaliser) with
banded LSH and its S-curve, SimHash with pigeonhole blocking, and `find_near_duplicates`
over four methods. `exact` and `canonical` take no threshold (passing one other than 1
is an error, not a silent no-op), and a text with no words — a map pin without a
comment — is never paired with anything. `collapse_duplicates` merges a group into its
representative with a `sum` / `one` / `sqrt` count policy and an optional spatial
guard, and never merges members of a different theme (that would move field mass
between themes, and in or out of `J_voice`).

### 9 — Spatial weights

Distance band (binary or inverse-distance), k-nearest neighbours (asymmetric, ties on
index), kernel weights, network edge contiguity and explicit matrices; zero diagonal;
general `S0`, `S1`, `S2`; row standardisation leaves islands at zero.

### 10 — Global autocorrelation

Moran's I and Geary's C with normal and randomisation moments, Getis-Ord General G
with its exact randomisation variance, seeded permutation p-values, exhaustive
permutation distributions for `n ≤ 8`, Ripley's K / L / `L − r` with no, border or
isotropic edge correction, and a CSR envelope.

### 11 — Hotspots

Benjamini-Hochberg (`benjamini_hochberg`, `bh_reject`), local Getis-Ord `G_i` and
`G_i*` (Ord-Getis standardised z), local Moran's I with HH / LL / HL / LH quadrants,
conditional-permutation pseudo p-values, `hotspot_clusters` (components of
significant units) and `comment_counts_per_hotspot` (records, summed counts, distinct
respondents, per-theme counts, mean severity, peak field value).

### 12 — Heat-map classification

Fisher-Jenks natural breaks by exact dynamic programming (each candidate class's
sum of squares accumulated relative to its own end value, so a tight class far from
the global mean is as exact as one near it),
equal-interval and quantile breaks, `classify_values`, `heatmap_classes` with the
goodness of variance fit, square binning and pointy-top hexagonal binning with cube
rounding.

### 13 — Self-selection diagnostics

`representativeness` (shares, representation ratios, Duncan dissimilarity, Pearson
chi-square with an incomplete-gamma tail), `post_stratification_weights`,
`rake_weights` (IPF over several margins), `kish_effective_sample_size` and
`design_effect`, `weighted_theme_shares`, Schouten's `r_indicator` and
`spatial_representation` (responses per 1000 residents and their Gini). A zone with
responses but zero population is an error in both `representativeness` and
`spatial_representation` — its responses would otherwise vanish from every rate.

### 14 — Frontend payloads

`build_public_voice_payload` serialises whichever analyses were run (camelCase keys,
`json.dumps(..., allow_nan=False)`-safe, truncated in a fixed order);
`build_a2ui_payload` emits stat cards, chart series and the hotspot table. Floats are
rounded to **10 significant digits**, not a fixed number of decimals: an `"area"`
field is in 1/m² (~1e-7 at a 1 km bandwidth) and a fixed `round(x, 8)` flattened it,
and anything rescaled by 1e-6, to zero.

---

## Recipes

**Find hotspots that survive a false-discovery correction, and count the people
behind them.**

```python
from public_voice_template import (weights_distance_band, local_getis_ord,
    hotspot_clusters, attach_concerns, comment_counts_per_hotspot)

tot = f.total_vector()
mids = [net.midpoint(e) for e in net.edge_ids]
w = weights_distance_band(net.edge_ids, mids, 130.0)
g = local_getis_ord(tot, w, star=True, alpha=0.1)          # BH-FDR by default
clusters = hotspot_clusters(g.ids, g.labels, w, "hot")
att, _ = attach_concerns(net, cs)
spots = comment_counts_per_hotspot(clusters, cs, {k: a.edge_id for k, a in att.items()})
```

**Reweight a self-selected sample.** Post-stratify on one margin, or rake on several,
then report what the weights cost:

```python
from public_voice_template import (representativeness, post_stratification_weights,
    kish_effective_sample_size)

rep = representativeness({"18-34": 6, "35-64": 1, "65+": 2},
                         {"18-34": 3200, "35-64": 4100, "65+": 2700})
# resp_stratum: respondent pseudonym -> age band, e.g. from ConcernRecord.stratum
w = post_stratification_weights(resp_stratum, {"18-34": 3200, "35-64": 4100, "65+": 2700})
n_eff = kish_effective_sample_size(list(w.values()))
field = concern_field(net, cs, bandwidth=120.0,
                      weights={c.cid: w.get(c.respondent, 1.0) for c in cs})
```

**Choose a bandwidth by cross-validation, then go adaptive.**

```python
from public_voice_template import cv_bandwidth, abramson_bandwidths, concern_field

h = cv_bandwidth([c.point for c in cs], "gaussian").bandwidth
ab = abramson_bandwidths([c.point for c in cs], "gaussian", pilot=h)
f = concern_field(net, cs, kernel="gaussian",
                  bandwidths={c.cid: hi for c, hi in zip(sorted(cs, key=lambda c: c.cid),
                                                         ab.bandwidths)})
```

---

## Design notes

- **Kernel normalisation is a convention, stated once.** The field uses the
  *peak* form `K_h(d) = k(d/h)`, so a comment at distance 0 contributes exactly its
  severity and the field is in severity units — what a planner reads off a map. The
  *density* (`c1·k(d/h)/h`, 1/m) and *area* (`c2·k(d/h)/h²`, 1/m²) forms integrate to
  1 and are used for the raster surface and for cross-validation. The suite checks
  both facts for all six kernels.
- **Exclusion is from the objective, not from the map.** `voice_objective` skips a
  theme whose pairing has no metric and reports its field mass in `excluded_mass`;
  `heatmap_layers` still emits it with `in_objective=False`. Adding parking comments
  changes neither `J_voice` nor the improvement — bit for bit.
- **Baseline normalisation.** In `edge` mode a lower-is-better burden is `x_e/b_e`,
  exactly 1 at baseline; an edge whose baseline is zero uses `1 + x_e/b̄₊` so the
  baseline is still exactly 1. Higher-is-better metrics use `b_e/x_e` and must be
  positive. Ratios are unit-free, so metric units never matter. When nothing
  measurable was raised, `J_voice(baseline) = 0` and the improvement is defined as 0.
- **Duplicates: the count policy is a decision, not a fact.** `sum` keeps every
  submission (and, for co-located members of equal severity, leaves the field exactly
  unchanged); `one` makes a campaign one voice; `sqrt` is a compromise. The suite pins
  all three. `max_distance` stops the same words about a different place from merging.
- **No builtin `hash()`.** Python salts `hash(str)` per process; any MinHash or LSH
  built on it changes with `PYTHONHASHSEED`. Shingles are hashed with FNV-1a, LSH band
  keys with SHA-1, buckets with SHA-256, jitter seeds with SHA-256, pseudonyms with
  HMAC. A suite test walks the template's AST to keep it that way.
- **Weights may be asymmetric.** k-nearest-neighbour weights are; every global moment
  uses the general `S1 = ½ΣΣ(w_ij + w_ji)²` and `S2 = Σ(w_i· + w_·i)²`, and the exact
  permutation checks use deliberately asymmetric weights and skewed values.
- **Local p-values.** `p_normal` is two-sided; `p_sim` is the folded *one-sided*
  conditional-permutation pseudo p-value `(1 + #extreme)/(1 + R)`, doubled before the
  BH adjustment that drives the labels. A unit whose z is degenerate (an island for
  `G_i` **or `G_i*`**, or equal weights everywhere) gets `z = 0`, `p = 1`, `"ns"`:
  a `G*` island's z only standardised its own value, so an isolated cul-de-sac with
  many comments was a "hot spot" under the normal moments while its permutation
  p-value was 1. Likewise a LISA unit whose value **is the mean** (`|z_i| ≤ 1e-14 ·
  max|x|`) has `I_i = 0` under every conditional permutation and is neither high nor
  low; it gets `z = 0`, `p = 1`, `"ns"` (Anselin's total-randomisation moments, which
  also move `x_i`, made the centre of a 3 × 3 block of 0s and 2s holding the mean 1 a
  significant `"LL"` cluster at p = 0.005). `I_i`, `expected` and `variance` stay
  Anselin's.
- **Permutation ties count in both tails.** `#extreme = min(#{sim ≥ obs}, #{sim ≤
  obs})` for every global and local pseudo p-value. The common shortcut `R − #{sim ≥
  obs}` (PySAL's) drops ties from the lower tail; with sparse comment counts (most
  streets 0) ties are the norm, and that shortcut gave every zero street with zero
  neighbours `p = 1/(R+1)` — 18 spurious "cold spots" on a 60-street grid — and gave a
  statistic every shuffle reproduces exactly (complete-graph weights, a `G*` island)
  `p = 0.01` instead of 1. For continuous data the two definitions coincide.
  "Equal" means equal in **exact arithmetic**: integer counts tie through different
  neighbour sets (`0 + 2 = 1 + 1`), the centring `x − x̄` rounds such sums
  differently, and a bitwise test split real ties at random between the tails (a
  `G*` p of 0.35 for an exact 0.44, a Moran p of 0.025 for an exact 0.04, and a
  different p after multiplying every count by 3). Two values count as tied when they
  differ by less than a **rounding bound of the terms actually summed**, derived per
  statistic: Moran's from the centred products (including the error of the rounded
  mean, which does not cancel across permutations on irregular weights), the local
  statistics' from their linear numerators, Geary's and General G's (sums of
  non-negative terms) a few ulps of the statistic. Every bound also carries the
  **values' own representation** (`ulp/2·|x|`): a decimal such as 55.3 dB is not a
  binary float, and on levels 55.0–55.4 dB that error is ~100× the spread, so exact
  decimal ties (55.4 + 55.2 = 55.3 + 55.3) split at random without it — a `G*`, `G_i`
  and LISA p of 0.25 for an exact 0.44 and a Geary p of 0.15 for 0.17, where the same
  levels minus 55 dB gave the exact values (round 7). The same linear bound serves the
  tests against an exact 0 below: the rounded mean's own error, `ulp·|x̄|·Σ|w|`, lies
  inside its representation term. An earlier fixed tolerance of 1e-9
  *of the statistic* merged genuinely distinct values as soon as one count was ~1e5
  times the others — one petition street among single comments moved Geary's p from
  0.09 to 0.17 and Moran's from 0.44 to 0.69. The suite checks every local and global
  pseudo p-value against an exact rational mirror of the same shuffles, on sparse
  counts, wide-range counts (up to 3e5 : 1), counts on a large offset and decimals
  (tenths, thirds) on offsets up to 1000.
- **Ties at a dynamic range beyond ~1e7 : 1 (inherent limit).** Distinct values that
  differ by less than a few ulps of the terms summed cannot be told from rounding
  noise: Geary's C or Moran's I of single comments next to a count of 1e7 merge
  values that are distinct in exact arithmetic (making p larger, never smaller).
  General G multiplies values, so it squares the range: counts of 0–4 on an offset of
  1e7 (or 1e6 under row-standardised weights) already merge distinct G values. The
  local statistics were exact in the round-7 replay up to counts on an offset of 1e12.
- **A z or a lag that is 0 in exact arithmetic is 0.** A `G_i` / `G_i*` z within the
  rounding bound of its numerator is set to exactly 0 (`p = 1`, `"ns"`): a street with
  4 comments and two silent neighbours, mean 4/3, has a local sum exactly equal to its
  expectation, yet rounding left z = ±3e-16 and its conditional p was 0.018, so it was
  `"hot"` at counts ×1, `"ns"` at ×3 and `"cold"` at ×7. Likewise a LISA spatial lag
  within its rounding bound of 0 is `"L"` (an exact 0 always was), so a high street
  whose neighbours average exactly the mean no longer flips between `"HH"` (cluster)
  and `"HL"` (outlier) with the units. Against an exact 0 (unlike between two
  permutations) the rounded mean's own error does NOT cancel: it shifts the sum by up
  to `ulp·|x̄|·Σ|w|`, so the zero bound includes it — without it counts on a common
  offset (every value + 1000) turned an exact-zero `G*` z into a `"cold"` spot at
  p = 0.018 and flipped the LISA quadrant `"HL"` → `"HH"`.
- **Keyword classifier: title case is ambiguous (inherent limit).** A one-word name
  before a thoroughfare type is always masked, so in title-case text "Parking Close To
  The School" reads as the street "Parking Close". Lower-case prose is unaffected.
  ALL-CAPS text has no case cue at all, so no street name is masked: "CROSS STREET IS
  A RAT RUN" ties crossing delay with rat-running and takes crossing delay (`THEMES`
  order, `confident=False`).
- **Keyword classifier: vocabulary (inherent limit).** A keyword list cannot read
  idiom or synonyms: "an accident waiting to happen" ties road safety with crossing
  delay ("waiting"), "sat nav" is not "satnav", "exceed the 20mph limit" and the verb
  "cycle" match nothing. On 30 realistic comments written for the round-7 hunt, 12
  went to another theme or to `"other"` (most with `confident=False`). Tag the theme
  at source where the form allows it, and send `confident=False` and `"other"`
  records to a human before they enter `J_voice`.
- **A JSON null is not a missing field (convention).** `ingest_concerns` treats a
  missing `theme` or `theme: null` as "classify", but every other field that is
  present must be valid: `"text": null` or `"severity": null` is a malformed record
  (raised, or listed in `report.rejected` with `strict=False`), never silently
  defaulted. Drop null keys before ingestion if the export writes them.
- **Pseudonyms are exact (convention).** `pseudonymise` keys the identifier as given:
  `Alice@Example.com` and `alice@example.com` (or the same with a trailing space)
  are two respondents. Normalise author identifiers (strip, case-fold e-mails)
  before ingestion if submissions are matched across channels.
- **`G*` self-weight is in the weights' units.** `self_weight` (default 1) is added
  to row `i` as is, which matches binary and row-standardised weights. With
  inverse-distance or kernel weights it is NOT rescaled with the coordinates (1/m
  weights against a fixed 1), so pick it relative to the row (e.g. the row maximum);
  the suite's scale-invariance tests use binary and kernel-peak weights.
- **Keyword classifier counts phrase hits separately.** A phrase keyword that contains
  a single-word keyword of the same theme (`"bike lane"` / `"bike"`, `"dropped kerb"` /
  `"kerb"`, `"pavement parking"` / `"parking"`) scores two hits for one mention: the
  more specific phrase weighs double. It is a convention of the heuristic; tag themes
  explicitly when it matters.
- **Result objects are validated, not trusted.** `voice_objective`, `heatmap_layers` and
  both payload builders check a `ConcernField` (every theme present, one finite value per
  edge, and — in a payload — exactly the network's edges); a `Hotspot.by_theme` may list
  only the themes it has (the rest are 0) but must map known themes to non-negative
  numbers. A hand-built or deserialised object used to leak `KeyError`.
- **Options of another mode raise.** `weights_distance_band(power=...)` applies to
  inverse-distance weights only and `local_getis_ord(self_weight=...)` to `G*` only;
  both default to `None` (meaning 1) and passing one to the other mode raises.
- **CSR envelope gaps.** A radius that no simulation can estimate (border correction
  beyond half the window) has a NaN envelope, like `ripleys_k` itself — never the
  inverted `[inf, −inf]` that every finite observed `L(r) − r` would lie outside.
- **Method-specific duplicate options.** `num_perm`, `bands`, `k`, `seed` and `verify`
  are MinHash's and `bits` is SimHash's; passing one to another method raises rather
  than being silently ignored. Every flag argument must be a real `bool`.
- **Near-constant values are rejected relative to magnitude.** A spread below ~1e-14
  of the values' size is rounding noise, so Moran / Geary / LISA raise instead of
  reporting autocorrelation of noise; a genuinely small spread at a small magnitude is
  fine.
- **Extreme magnitudes.** Every autocorrelation statistic is scale-free, but its
  kurtosis term `Σz⁴` overflows above |x| ≈ 1e77 and squares underflow below ≈ 1e-154.
  Values are therefore rescaled by an exact power of two before any arithmetic, which
  leaves every result bit-identical and works from 1e-300 to 1e300.
- **Core BH vs fallback.** `benjamini_hochberg` validates first, then delegates to the
  core `p_adjust(method="bh")` when importable (from the repo root) and otherwise runs
  the same step-up recursion locally (from the module directory); values are
  bit-identical and the exception type is `PublicVoiceError` on both paths.
- **Binning boundaries (inherent limit).** `bin_points_square`, `k_anonymity_cells`
  and `coarsen_location` all bin by `floor(fl((x − x0)/c))`. A point within about one
  ulp of a cell boundary can land in the neighbouring cell relative to exact real
  arithmetic (`12.3 + 40·30` lands in cell 40, a hair below the exact boundary); the
  three functions always agree with each other, totals are conserved, and dyadic
  coordinates and cells are exact. The suite pins the rule.
- **Projection far from a short segment.** Squared lengths are formed after an exact
  power-of-two rescale, so attachment and the field are bit-identical up to 1e300; a
  point more than ~1e300 segment-lengths away projects to the nearer end, and when
  both end distances round to the same float the tie goes to `a`.
- **`voice_improvement` from a zero baseline.** In `network` mode `J(baseline)` can be
  0 while `J(x) > 0` (every concern on edges whose baseline metric is 0); the ratio is
  −∞ and the call raises instead of reporting "no change". `0/0` is still 0.
- **Payload precision.** Payload floats carry 10 significant digits; `J_voice` read
  back from a payload resolves improvements only to ~1e-10 relative, and the a2ui
  improvement card is a percentage to 2 decimals. Compare full-precision objects, not
  payloads, when exactness matters. A cluster unit absent from `unit_values` gets
  `peak_value = 0`.
- **Jenks is exact but quadratic.** The Fisher DP is `O(k n²)`; it is meant for the
  per-street values of a district (hundreds), not millions of pixels — bin first. It
  runs on the values divided by an exact power of two, so the breaks and the GVF are
  exact from 1e-300 to 1e300 (unscaled, squared deviations overflowed above ~1e154 and
  silently produced wrong breaks); `sdcm` / `sdam` are reported as `inf` when the true
  sum of squares exceeds the float range. `heatmap_classes` computes its GVF the same
  way, so for `"jenks"` it equals `JenksResult.gvf` bit for bit.

---

## Testing

`_stress_test.py` is an oracle-based suite — **481 tests**, no golden outputs, in four
tiers:

1. **Closed forms.** Each kernel's 1-D and 2-D normalising constant, second moment and
   roughness by Simpson quadrature; Silverman's efficiency table (0.951 / 0.986 / 0.994
   / 0.930) and the canonical bandwidths; a single comment at distance 0 gives exactly
   its severity (every kernel, both distance modes); hand values of the field, of
   `J_voice` and of the normalised burdens; Moran's I of an alternating path is exactly
   −1 and of a trend exactly 0.6; Ripley's isotropic weight is exactly 4 at a corner and
   2 on a side, and matches a 200 000-point circle quadrature elsewhere; the chi-square
   tail with 2 dof is `exp(−x/2)`; FNV-1a test vectors; Kish's ESS of `[1, 3]` is 1.6;
   BH against a hand computation; a conditional-permutation p-value against its exact
   hypergeometric probability.
2. **Exact identities.** The field is linear in severity (bit-exact for a factor of 2)
   and additive over comments; `J_voice(baseline)/J_voice(baseline) = 1` and the
   improvement at baseline is exactly 0; **Moran's I has mean exactly −1/(n−1) over all
   n! permutations for n = 4…7** with asymmetric weights and values, and its Cliff-Ord
   randomisation variance equals the exhaustive permutation variance; likewise Geary's C
   (mean 1), General G, local Moran's I (Anselin's mean and variance) and local
   Getis-Ord `G_i` / `G_i*` (**mean 0, variance 1**); Jenks equals brute force over every
   partition; `Σ I_i = S0 · I`; post-stratified and raked margins match the population;
   Abramson's geometric mean is the pilot.
3. **Metamorphic.** Scale invariance at 1e+6 and 1e−6 with value assertions
   (coordinates and bandwidths, metric units, severities, statistic values, Ripley's
   `K ∝ s²`); affine invariance of Moran's I and Geary's C; unmeasurable comments leave
   `J_voice` unchanged; canonical duplicates are always MinHash and SimHash duplicates;
   the LSH candidate rate follows the S-curve; Jenks dominates quantile and equal-interval
   GVF; determinism across repeat calls and across `PYTHONHASHSEED` 0 / 1 / 1010 in
   subprocesses.
4. **Differential** (behind `pytest.importorskip`). The 2-D KDE and leave-one-out
   likelihood against `sklearn.neighbors.KernelDensity` for five kernels; Dijkstra and
   attached-point network distances against `networkx`; BH against
   `scipy.stats.false_discovery_control` and against the core library's `p_adjust`
   (bit-identical, both import paths, same exception types); the chi-square tail and
   goodness of fit against `scipy.stats`; the Silverman factor against
   `scipy.stats.gaussian_kde`; Ripley pair counts against `scipy.spatial.cKDTree`;
   quantiles, Moran's I, LISA and `G_i*` against `numpy`. The Cliff-Ord normality
   variances are checked by a 20 000-draw Gaussian simulation to within 4 standard errors.

```bash
pytest _stress_test.py -q -n 8
for s in 0 1 1010; do PYTHONHASHSEED=$s pytest _stress_test.py -q -n 8; done
cd ../../../.. && python -m pytest algorithms/domains/urban/public_voice/ -q
```

**What the suite caught while it was being written.** The Silverman efficiency was
first computed with an extra 4/5 power (Gaussian 0.961 instead of the published
0.951); a taxonomy keyword (`"impossible to cross"`) classified to the wrong theme
because it contains `"cross"`; and `weights_from_matrix` leaked a bare `TypeError` on a
non-iterable matrix. All three are fixed and pinned.

**What the adversarial bug hunt caught** (section 29 of the suite; every test there
fails against the pre-hunt template): payload floats rounded to an absolute 1e-8,
which flattened area-normalised fields and 1e-6-scaled data to zero; 44 bad-input
paths that leaked bare `TypeError` / `KeyError` / `AttributeError` / `OverflowError`
(mixed-type dict keys hitting `sorted`, unvalidated optional payload blocks,
`concern_contributions` options, cell indices overflowing to infinity, ...); Moran /
Geary / General G / LISA crashing with `OverflowError` above |x| ~ 1e77 and
`ZeroDivisionError` below ~ 1e-150; `SpatialWeights` aliasing the caller's id list;
`exact` / `canonical` duplicate detection silently ignoring a threshold; MinHash
over-estimating Jaccard by ~7 standard errors on low-entropy (single-character)
shingles; bracketed UK area codes escaping phone redaction; `spatial_representation`
dropping responses from a zero-population zone; `collapse_duplicates` merging across
themes; and every comment-less map pin being flagged a duplicate of every other.
Section 30 kills the hand mutants that survived mutation testing (31 of 33 killed;
the other two are equivalent mutants).

**What the second bug hunt caught** (section 31; every test there fails against the
round-1 template). Jenks lost the optimum for a tight class far from the global mean
(prefix sums `S2 − S1²/m` cancelled: SDCM reported as 48 for a true 4e-10, wrong
breaks); `collapse_duplicates` orphaned same-theme duplicates whenever a
differently-themed or distant duplicate held the smallest id, so renaming one id
changed a theme's mass; `voice_improvement` reported 0 ("no change") for a design that
raised `J_voice` from a zero network-mode baseline; `project_point_to_segment`
returned NaN above ~1e154, silently turning the whole field into NaN; the KDE,
leave-one-out likelihood, LSCV, Silverman and Abramson leaked bare
`ZeroDivisionError` / `OverflowError` / `ValueError` (or picked a bandwidth from all-inf
scores) at 1e±160; `representativeness` overflowed on huge counts and
`spatial_representation` divided by an underflowed `ptot²`; and `concern_field`
silently ignored options of the other distance mode. Section 32 kills the two of 32
round-2 hand mutants that survived (a dropped SplitMix64 round, pinned to the published
output stream, and a 1-D exponent in the 2-D canonical bandwidth, pinned by
quadrature).

**What the third bug hunt caught** (section 33; every test there fails against the
round-2 template). Permutation pseudo p-values dropped ties from the lower tail, so on
sparse comment counts every zero street with zero neighbours got `p = 1/(R+1)` and was
labelled a cold spot, and a statistic every shuffle reproduces exactly (complete-graph
weights, a `G*` island) was "significant" at `p = 0.01` instead of `p = 1` (an
existing test had pinned the wrong `1/(R+1)`; it now pins 1); Jenks silently returned
wrong breaks above ~1e154 and below ~1e-154 and `heatmap_classes` leaked
`OverflowError` (both now run on power-of-two-scaled values) and re-derived a GVF that
differed from Jenks' far from zero; `hex_centre` returned `inf` or leaked
`OverflowError` beyond the float range; ten flag arguments read any object through its
truthiness; and `find_near_duplicates` silently ignored MinHash / SimHash options
passed to another method. Section 34 kills five of seven surviving round-3 hand
mutants (30 of 32 killed); the other two are equivalent (the union-find root is never
observable, and the hex rounding branch differs only for points equidistant from two
centres).

**What the fourth bug hunt caught** (section 35; every test there fails against the
round-3 template). Pseudo p-values still split exact-arithmetic ties at random:
integer counts tie through different neighbour sets, the centring rounds those sums
differently, so `G*`, LISA, Moran and Geary p-values differed from the exact ones
(0.35 for 0.44, 0.025 for 0.04) and changed when every count was multiplied by 3; a
`G*` island was a normal-theory "hot spot" (its z only standardised its own value)
while its permutation p was 1; a LISA unit holding exactly the mean was labelled a
significant `"LL"` cluster; the CSR envelope of a radius no simulation could estimate
was the inverted `[inf, −inf]`; and `weights_distance_band(power=)` /
`local_getis_ord(self_weight=)` were silently ignored in the other mode. One existing
test pinned the old island z (`> 0`); it now pins `z = 0`. Section 36 kills the three
of 29 round-4 hand mutants that survived the suite or could not be applied as first
written (a kernel-weights builder that always used the quartic, a signed "max
absolute gap", and y binned against the x origin); all 29 are killed.

**What the fifth bug hunt caught** (section 37; every regression there fails against
the round-4 template, alongside a hand-computed ingest-to-`J_voice` district and the
classifier's positive cases, which guard the fixes). The 1e-9 relative tie tolerance
merged genuinely distinct permutation values once one count was ~1e5 times the others
(Geary 0.17 for an exact 0.09, Moran 0.69 for 0.44; 54 of 1953 fuzzed pseudo p-values wrong) — replaced by
per-statistic rounding bounds; a `G*` z that is exactly 0 kept a rounding sign and was
labelled `"hot"` / `"cold"` depending on a units factor; a LISA lag that is exactly 0
flipped the quadrant between cluster and outlier the same way; the keyword classifier
filed "noise near the park" and "lorries along Park Road" under parking (silently out
of `J_voice`), "street lights" under crossing delay, and missed every plural
("wheelchairs", "accidents", "cycle lanes"); a hand-built `ConcernField` or `Hotspot`
leaked `KeyError` from four entry points and a field from another network was put in a
payload unchecked; and float-range extremes leaked `OverflowError` from inverse-distance
weights or returned `inf` / NaN from the geomasks. The round-4 exact mirror for local
p-values computed its observed lag in floats (`Fraction × float` is a float), so with
row-standardised weights it could miss an exact tie; it now stays rational. Section 38
kills the six of 30 round-5 hand mutants that survived: five with new tests, and one
(dropping the rounded-centre term from the local bound) by deleting that term — the
centre's error shifts every permutation equally and cancels, so the bound is now
tighter and still sound.

**What the sixth bug hunt caught** (section 39; every regression there fails against
the round-5 template, except two classifier cases that guard the street-name fix
and one inherent-limit pin — 5 template bugs). An independent exact-fraction
replay of every pseudo p-value (Moran, Geary, General G, LISA, `G_i`, `G_i*`; 700
fuzzed districts, ~30 000 local p-values; sparse, 3e5 : 1, offset, thirds and
1e-6-scaled counts) agreed with the round-5 tie bounds everywhere — but the *zero* tests did not. Round 5's deletion
of the rounded-centre term was right for ties between two permutations and wrong
for a comparison against an exact 0, where that error does not cancel: on counts
with a common offset (every value + 1000) an exact-zero `G*` / `G_i` z came out
as −7e-14 and a street was a `"cold"` spot at p = 0.018 instead of `"ns"` at
p = 1, and a LISA lag that is exactly 0 flipped the quadrant from `"HL"`
(outlier) to `"HH"` (cluster) for 146 of 429 offsets — both statistics are
invariant to the offset. The keyword classifier masked a capitalised verb with
the street name after it ("Crossing Station Road takes ages", "Cycling Mill Lane
…", "Parking Church Street …" were all `"other"`, i.e. out of `J_voice`), and
missed every phrase keyword spelled with a hyphen or broken across a line or
double space ("near-miss", "cycle-lane", "air-quality", "traffic-light",
"through-traffic", "step free", "cycle⏎lane"), which also tipped
"Pavement-parking blocks my buggy" to accessibility. `voice_objective` /
`voice_improvement` returned `J_voice = inf` and an improvement of −∞ silently
when a normalised burden overflowed, and leaked a bare `OverflowError` for a
baseline near 1e308. Section 40 kills the five of 24 round-6 hand mutants that
survived (an empty pseudonym counted as a respondent, a theme skipped when its
first edge is 0, the equal-weights guard of `G_i` tightened to an exact 0, the
LISA own-value sign read without its at-mean floor, `-es` plurals dropped); the
sixth (ties in the LSH band choice) is equivalent — the S-curve threshold
`(1/b)^(b/N)` is strictly decreasing in `b`, checked for every `num_perm` up to
1024. Two findings are conventions, not fixed: pseudonyms are exact (see Design
notes), and `collapse_duplicates` returns the caller's own record objects for
the records it does not merge.

**What the seventh bug hunt caught** (section 41; every regression there fails against
the round-6 template, except three classifier cases that guard the wrapped-phrase
behaviour — 4 template bugs). An independent exact-fraction replay of every pseudo
p-value, zero test and LISA quadrant (Moran, Geary, General G, LISA, `G_i`, `G_i*`;
over 5 000 fuzzed districts; sparse, 3e5 : 1, offset up to 1e12, thirds, 1e±6-scaled
counts) confirmed the round-5 tie bounds and the round-6 zero bounds on integer and
dyadic data — no exact zero missed, no small genuine z or lag swallowed — but not on
**decimals**. (1) The local tie bound left out each value's own representation error,
so on levels like 55.0–55.4 dB exact decimal ties (55.4 + 55.2 = 55.3 + 55.3) split at
random: `G*` / `G_i` p = 0.25 for an exact 0.44, and 0.44 for the same levels minus
55 dB. (2) `local_morans_i` derived that bound per unit and never passed it to the
pseudo p-value (the round-6 open question: it was a latent bug, not dead code) — it fell
back to 64 ulp of the statistic, which the same data beat even with (1) fixed.
(3) Geary's 64-ulp test had the same gap (p = 0.15 for an exact 0.17). (4) Round 6's
"every whitespace run is one space" joined separate list items: "Too much
traffic⏎Lights out on the path" became the crossing keyword "traffic light" — the
street-lights confusion round 5 fixed; a line break before a capitalised line is now
a boundary, while a wrapped lower-case "cycle⏎lane" still matches. With the
representation term inside the linear bound, round 6's separate zero bound
(`_zero_tol_linear`) became redundant — the rounded mean's error is inside it — and its
mutant survived, so it was deleted (the zero tests now use the tie bound itself,
still sound with a 5x margin). Section 42 kills three of the five survivors of 25 round-7 hand mutants
(`G_i`'s noise-floor guard tightened to an exact 0, the chi-square dof over
zero-population strata, the en dash dropped from the phrase joiner); the other two
are equivalent — the deleted zero term above, and dropping `_p_sim`'s `8 ulp·|stat|`
slack, which every derived bound already dominates (`≥ 32 ulp` of the statistic).
Inherent limits and conventions, documented not fixed: General G squares the dynamic
range (counts on an offset of 1e7 merge distinct values); ALL-CAPS text masks no street
names; keyword vocabulary (12 of 30 realistic comments misrouted, mostly with
`confident=False`); a JSON `null` field is a malformed record; and a huge `k` or
`num_perm` (1e30) exhausts memory rather than raising.

---

## References

- Silverman (1986) — *Density Estimation for Statistics and Data Analysis*: the rule of
  thumb, kernel efficiencies, adaptive kernels. Abramson (1982) — the square-root law.
  Marron & Nolan (1988) — canonical kernels. Wand & Jones (1995) — *Kernel Smoothing*.
  Okabe & Sugihara (2012) — *Spatial Analysis Along Networks*.
- Moran (1950); Geary (1954); Cliff & Ord (1981) — *Spatial Processes*: the
  normality and randomisation moments. Getis & Ord (1992); Ord & Getis (1995) — the
  General G and local `G_i`, `G_i*`. Anselin (1995) — *Local Indicators of Spatial
  Association — LISA*. Benjamini & Hochberg (1995) — the false discovery rate.
- Ripley (1976, 1977) — the K function and the isotropic edge correction; Besag (1977)
  — the L function. Jenks (1967); Fisher (1958) — optimal grouping.
- Broder (1997) — MinHash; Indyk & Motwani (1998), Leskovec, Rajaraman & Ullman —
  *Mining of Massive Datasets*, ch. 3 (LSH banding); Charikar (2002) — SimHash; Manku,
  Jain & Das Sarma (2007) — pigeonhole blocking. Fowler, Noll & Vo — FNV.
- Kish (1965) — *Survey Sampling*, effective sample size; Deming & Stephan (1940) —
  raking; Schouten, Cobben & Bethlehem (2009) — the R-indicator. Duncan & Duncan (1955)
  — the dissimilarity index.
- Kwan, Casas, Schmitz (2004); Allshouse et al. (2010) — geographic masking and donut
  jitter.

---

*Pure standard library: `bisect`, `hashlib`, `hmac`, `heapq`, `itertools`, `json`,
`math`, `random`, `re`, `unicodedata`, `dataclasses`, `typing`.
`public_voice_template.py` — `__version__ = "1.0.0"`, 141 public names in `__all__`,
481 oracle-based tests.*
