# Accessibility & Equity

A self-contained, **zero-dependency** toolkit for the two questions a transport
plan is actually judged on: **how much of the city can each place reach**, and
**is that reach fairly distributed across the people who live there?** It takes
what the rest of the urban stack produces — a travel-cost matrix from
`street_network`, a population and opportunity distribution from `land_use` /
`travel_demand` — and turns it into cumulative-opportunity counts, gravity and
floating-catchment surfaces, logsum welfare measures, and then the whole
inequality battery: Lorenz curves, Gini, Theil with an exact between/within
decomposition, Atkinson, Palma, concentration indices, transport-poverty
headcounts and Rawlsian max-min.

Nothing here predicts a trip. `travel_demand` forecasts behaviour and
`street_network` measures the network; this module *consumes* both and asks a
**normative** question. Accessibility measures the opportunity set a location
commands whether or not anybody uses it; the equity half lays that surface
against population, need and deprivation and reports who wins. A Gini
coefficient is not a forecast — it is a verdict. Everything is pure standard
library: the Lorenz curve, the Theil decomposition, the normal CDF, the
floating-catchment solvers, Dijkstra and the exhaustive Moran's-I permutation
test are all built by hand.

```python
from accessibility_equity_template import (
    cost_matrix_from_coords, get_decay, gravity_accessibility,
    cumulative_opportunity, two_step_fca, inequality_profile,
    theil_decomposition, accessibility_gap,
)

cost = cost_matrix_from_coords(zone_centroids, speed_kph=24.0, circuity=1.3)
decay = get_decay("exponential", beta=0.08)          # 8.7-minute half-cost

jobs_30min = cumulative_opportunity(cost, jobs, 30.0)        # isochrone count
potential  = gravity_accessibility(cost, jobs, decay)        # Hansen potential
doctors    = two_step_fca(cost, gp_capacity, population, 20.0)   # 2SFCA

profile = inequality_profile(potential, population)          # Gini, Theil, ...
split   = theil_decomposition(potential, borough, "T", population)
gap     = accessibility_gap(potential, ethnicity, population)

print(profile.gini, split.between + split.within == split.total, gap.ratio_gap)
```

`cost[i][j]` runs origins (people) down the rows and destinations
(opportunities) across the columns; an infinite cost means unreachable and is
legal everywhere. Anything producing one number per origin returns a plain
`list` of floats; anything richer returns a `@dataclass`.

---

## Pick

| You want to know… | Use |
| --- | --- |
| how many jobs are within 30 minutes | `cumulative_opportunity` |
| …at several cut-offs, with a population-weighted mean | `cumulative_opportunity_curve` |
| how long it takes to reach 20 000 jobs (the *dual*) | `place_rank_accessibility` / `time_to_nth_opportunity` |
| a smooth, decay-weighted opportunity potential | `gravity_accessibility`, `hansen_accessibility` |
| the same, adjusted for competition for those jobs | `shen_accessibility` |
| doctors per resident, allowing for cross-boundary use | `two_step_fca`, `enhanced_2sfca`, `three_step_fca`, `m2sfca` |
| a catchment that grows until it captures enough people | `variable_catchment_2sfca` |
| accessibility with a welfare interpretation (£) | `logsum_accessibility`, `logsum_benefit` |
| what the peak does to all of the above | `time_of_day_accessibility` |
| accessibility when the network is unreliable | `reliability_adjusted_accessibility` |
| how unequal the surface is | `inequality_profile`, `gini`, `atkinson` |
| whether the inequality is *between* groups or *within* them | `theil_decomposition` |
| whether like is treated alike, and need is treated better | `horizontal_vertical_equity` |
| how many people fall below an accessibility poverty line | `transport_poverty_indicators` |
| whether low access clusters in space | `morans_i`, `local_morans_i`, `gearys_c` |
| who gains and who loses from a scheme | `impact_incidence`, `winners_losers` |
| which option is best for the worst-off | `rawlsian_maxmin`, `leximin_sort` |
| what a closure or a flood costs, and to whom | `disruption_scenario`, `accessibility_loss`, `vulnerability_ranking` |

---

## Install

```bash
# nothing to install -- copy the file
cp accessibility_equity_template.py your_project/

python accessibility_equity_template.py      # runs the demo, prints "demo OK"
pytest _stress_test.py -q                    # 180 oracle-based tests
```

Python 3.8+, standard library only. `pytest>=7.0` is needed only to run the
test suite; NumPy / SciPy / esda are needed only for the optional differential
tier, which skips cleanly when they are absent.

---

## What's inside

### 1 — Impedance (distance decay)

Six kernels behind one `DecayFunction` object — `power`, `exponential`,
`gaussian`, `log_logistic`, `rectangular` and `cumulative_gaussian` — plus
`decay_curve`, `decay_half_cost` (bisection, agreeing with the analytic root to
1e-12) and `calibrate_decay`, which bisects a kernel's scale parameter until its
implied mean trip cost matches an observed one. `mean_decay_cost` is the
calibration target; `normal_cdf` is the hand-rolled `erfc` wrapper behind the
cumulative-Gaussian kernel.

### 2 — Travel-cost matrices

`CostMatrix` (labels, units, `scaled`, `transposed`, `with_diagonal`,
`max_finite`) with builders from coordinates (`haversine_m` or `euclidean_m`,
speed and circuity), from a weighted graph (`cost_matrix_from_graph`, a
from-scratch binary-heap Dijkstra) and `intrazonal_cost`, whose default `2r/3`
is the exact mean centre-to-point distance in a disc.

### 3 — Cumulative opportunity and its dual

`cumulative_opportunity` at one cut-off, `cumulative_opportunity_curve` at
many, `isochrone_bands` for the ring-by-ring breakdown, and the *dual* measure —
`time_to_nth_opportunity` / `place_rank_accessibility`, the cost of reaching the
N-th opportunity. The two are exact inverses of each other, by construction.

### 4 — Gravity and potential

`gravity_accessibility` (any kernel, any size exponent), `hansen_accessibility`
(the 1959 inverse-power original), `potential_accessibility` (explicit control
of the self-potential term) and `shen_accessibility` (opportunity divided by
the demand competing for it). `normalise_accessibility` rescales a surface to a
dimensionless index five ways.

### 5 — Floating catchments

`two_step_fca` (Luo & Wang), `enhanced_2sfca` (Luo & Qi's banded kernel
weights), `three_step_fca` (Wan's selection probability, which removes the
demand double-counting), `m2sfca` (Delamater's squared impedance) and
`variable_catchment_2sfca`, whose catchments grow until they capture a target
population and a target supply. `FCAResult` carries the supply-to-demand
ratios, both catchment totals, the fitted radii and whether the variant
conserves supply.

### 6 — Utility-based accessibility

`utility_matrix`, `destination_choice_probabilities` (a numerically safe
multinomial logit) and `logsum_accessibility` — the only accessibility measure
with a welfare interpretation — plus `logsum_benefit`, which converts a logsum
change into consumer surplus at a given marginal utility of income.

### 7 — Time of day and reliability

`time_of_day_accessibility` weights per-period surfaces into a day and reports
each zone's best and worst period and its temporal variability.
`travel_time_percentile` (NumPy's linear-interpolation convention),
`buffer_time_index`, `planning_time_index`, `reliability_adjusted_cost` and
`reliability_adjusted_accessibility` handle the unreliable network.

### 8 — Inequality

`lorenz_curve` / `lorenz_area` / `gini_from_lorenz` and the rank-formula `gini`;
`theil_t`, `theil_l`, `generalised_entropy` and `theil_decomposition` (exact
between/within); `atkinson` at any aversion; `palma_ratio`, `quintile_ratio`,
`coefficient_of_variation`, `hoover_index`, `kolm_index`,
`concentration_curve` / `concentration_index`; `inequality_profile` computes the
lot at once, and `pigou_dalton_transfer` makes the defining axiom a first-class
operation.

### 9 — Horizontal versus vertical equity

`horizontal_vertical_equity` reads the Theil split as *like treated alike*
(within) versus *unlike treated differently* (between), returning a horizontal
index in `[0, 1]` and a signed progressivity index against need, with a
one-word `verdict`.

### 10 — Need-weighted accessibility

`population_weighted_mean`, `equity_weights` (population tilted by deprivation
with an explicit elasticity), `need_weighted_accessibility`,
`deprivation_weighted_accessibility` and `accessibility_gap` for
between-group comparisons.

### 11 — Transport poverty

`accessibility_poverty_line` (a fraction of the weighted median),
`fgt_index` (the Foster-Greer-Thorbecke family: headcount, gap, severity),
`transport_poverty_indicators` and a composite `transport_poverty_index` over
accessibility deficit, car deprivation and cost burden.

### 12 — Spatial autocorrelation

`spatial_weights_knn`, `spatial_weights_distance_band`,
`spatial_weights_inverse_distance`, `row_standardise`, then `morans_i`,
`gearys_c`, `local_morans_i` (LISA) and `morans_i_permutation_test`, which can
be run **exhaustively** over every ordering for small n instead of sampling.

### 13 — Impact incidence

`impact_incidence` compares two surfaces group by group — mean change, relative
change, share of people gaining and losing, the change in Gini and whether the
worse-off half gained more than the better-off half. `winners_losers` gives the
zone-level partition; `distributional_summary` the one-dict overview.

### 14 — Rawlsian priority

`rawlsian_index` (the worst-off zone), `leximin_key` / `leximin_compare` /
`leximin_sort`, and `rawlsian_maxmin`, which ranks competing designs by max-min
and then refines the ties by leximin.

### 15 — Disruption

`disruption_scenario` (closures, detour multipliers, delays, blocked pairs),
`accessibility_loss` (per-zone loss, relative loss, the Gini of the loss and
the change in the Gini of the surface), `vulnerability_ranking` (one-at-a-time
destination removal) and `resilience_index`.

### 16 — Payloads

`build_accessibility_equity_payload` and `build_a2ui_payload` return JSON-able
dicts — zones, Lorenz curve, decay curve, histogram, stat cards — with every
non-finite float replaced by `null`.

---

## Recipes

**A 30-minute jobs isochrone, population-weighted.**

```python
counts = cumulative_opportunity(cost, jobs, 30.0)
print(population_weighted_mean(counts, population))
```

**Is the GP surface equitable, and where is the shortfall?**

```python
fca  = two_step_fca(cost, gp_capacity, population, radius=20.0)
prof = inequality_profile(fca.access, population)
line = accessibility_poverty_line(fca.access, 0.5, "median", population)
poor = transport_poverty_indicators(fca.access, line, population)
print(prof.gini, poor.headcount_ratio, [zones[i] for i in poor.poor])
```

**Between boroughs, or within them?**

```python
dec = theil_decomposition(access, borough, "T", population)
print("%.1f%% of the inequality is WITHIN boroughs" % (100 * dec.within_share))
assert abs(dec.residual) < 1e-12          # between + within == total, exactly
```

**Who gains from the new bus lane?**

```python
after = gravity_accessibility(faster_cost, jobs, decay)
inc   = impact_incidence(before, after, income_band, population)
print(inc.net_change, inc.progressive, inc.gini_before, inc.gini_after)
```

**Pick the option that is best for the worst-off.**

```python
res = rawlsian_maxmin([("do-nothing", a), ("bus-lane", b), ("orbital", c)])
print(res.best, res.leximin_ranking)
```

**What does closing the interchange cost, and to whom?**

```python
hit  = disruption_scenario(cost, jobs, decay, closed_destinations=(12,),
                           cost_multiplier=1.25, cost_penalty=2.0)
loss = accessibility_loss(base, hit, population)
print(loss.resilience, loss.loss_gini, zones[loss.worst_zone])
```

---

## Design notes

**One number per origin is a list; anything richer is a dataclass.** Every
measure composes: `gini(gravity_accessibility(...), population)` works because
the accessibility half returns plain floats and the equity half accepts them.
The dataclasses (`FCAResult`, `TheilDecomposition`, `IncidenceResult`, …) carry
provenance — parameters, group order, catchment totals — and all have
`to_dict()` for the payloads.

**Exact-zero guards, never an absolute epsilon — and scale the DATA, not the
threshold.** A denominator is tested against `0.0` exactly; where a tolerance is
genuinely needed it is made relative to the data. The only use of `_EPS` is the
constant-surface check in Moran's I, and that check now runs on max-scaled
deviations, so the comparison is a bare `n * _EPS ** 2`.

The earlier form, `m2 <= n * (_EPS * spread) ** 2`, *looked* scale-relative and
was still wrong at the extremes: on a surface around 1e-170 the guard's own
expression underflowed to exactly 0.0 **before** `m2` did, so it silently
degenerated into `m2 <= 0` and every statistic reported "a constant surface" for
a surface that plainly varied. Squaring a deviation is the one operation here
that can leave double range, so every routine that needs a second moment
(`coefficient_of_variation`, `normalise_accessibility('zscore')`, the Moran
family, `time_of_day_accessibility`'s variability and
`reliability_adjusted_accessibility`) goes through the shared `_scaled_moments`
helper, which divides by the peak magnitude first. Dispersion *ratios* are
unchanged by that division, so they are now bit-identical across 400 orders of
magnitude — the suite asserts exactly that, at 1e-200 through 1e+200, and not
merely at 1e-6 / 1e6. **Making a threshold relative to the data is not enough
if the threshold itself can underflow: scale the data instead.**

**Determinism is a design constraint, not a hope.** Every sum goes through
`math.fsum` (which is order-independent, so an index-ordered sum and a
cost-ordered sum agree to the last bit), every tie is broken by `(value,
index)` or by label, and group ordering is *first appearance* — never set or
dict iteration order. The permutation test takes an explicit seed. The demo is
byte-identical under `PYTHONHASHSEED` 0, 1 and 1010.

**The Kolm index is deliberately not scale-invariant.** Gini, Theil, Atkinson,
Palma, 20:20, CV, Hoover and the concentration index are *relative* measures:
multiply every accessibility by a million and they do not move. Kolm is the
*absolute* (leftist) index and is translation-invariant instead — add a
constant to everybody and it does not move. Both properties are asserted.

**Supply conservation separates the catchment variants.** 2SFCA, E2SFCA and
3SFCA redistribute every reachable unit of supply exactly: `sum_i P_i A_i`
equals the supply sitting in a live catchment. M2SFCA deliberately does not —
squaring the impedance is how it penalises a poorly configured catchment — and
the variable-catchment variant cannot, because its two steps use different
catchments. `FCAResult.conserves_supply` says which you have.

**Theil L is infinite when anything is zero, and says so.** The mean log
deviation is undefined at zero; rather than silently dropping zones (which
would quietly change the population base) the function returns `+inf` and the
decomposition propagates it. Theil T, Gini, Atkinson below `epsilon = 1` and
the whole cumulative family handle zeros normally.

**`calibrate_decay` verifies its own answer.** The implied mean trip cost
`sum(w f(c) c) / sum(w f(c))` is a **convex combination of the costs**, so it can
never leave `[min(costs), max(costs)]` — and the kernel's tail narrows that
further. A negative-exponential can be driven down to the cheapest trip, but a
log-logistic with `beta` fixed at 3 bottoms out at `sum(c^-2) / sum(c^-3)`
(12.09 for costs `[10, 20, 30, 40]`, not 10), and a cumulative-Gaussian with
`sigma` fixed at 10 bottoms out at 11.40. No closed-form range check can cover
that, so after bisecting, the routine **re-evaluates the mean at the parameter it
found** and raises if it missed by more than 1e-9 relative. Either you get a
kernel whose implied mean *is* the target, or you get an error naming the closest
value the kernel reaches. It never returns a parameter that merely underflowed.

**`circuity` means `>= 1` everywhere it appears.** It inflates a straight line
into a road journey, so a value below 1 would make the journey shorter than the
crow flight. `travel_time` always enforced it; `cost_matrix_from_coords` now
enforces it on **both** branches, because the metres branch multiplied it in raw
and never reached `travel_time`.

**Tie-breaking by index is a determinism guarantee with one statistical cost.**
`concentration_curve` and `concentration_index` rank the population by a *need*
variable, and ties in that variable break by original index rather than by
mid-rank averaging. Two zones with identical need therefore keep their input
order, and because the curve integrates the values along that order the index
moves with it: `concentration_index([1, 5], [2, 2])` is `+1/3` while
`concentration_index([5, 1], [2, 2])` is `-1/3`, though the row swap carries no
information (mid-ranks would give 0 for both). This is documented behaviour, not
a defect — the module's determinism rule is deliberate — and both docstrings now
warn about it. Break ties in `ranking` yourself when the need variable is coarse.

**The dual measure is bit-consistent with the primal.** `time_to_nth_opportunity`
accumulates with `math.fsum` over the sorted prefix rather than with a running
`+=`, precisely so that `cumulative_opportunity(t) >= n` holds *if and only if*
`t >= place_rank(n)`. A running total differs from `fsum` by an ulp, which is
enough to flip the boundary case.

---

## Testing

`_stress_test.py` holds **251 tests**, **all of which run** — there is no
conditional skip anywhere in the file — green under `PYTHONHASHSEED` 0, 1 and
1010, from the module directory and from the repository root, in five tiers:

* **Closed forms** — every kernel's half-cost against its analytic root; Gini
  exactly `0` at equality and exactly `(n-1)/n` when one takes all; Theil
  exactly `0` and exactly `ln(n)`; Atkinson at `epsilon = 1` exactly
  `1 - gmean/mean`; the Moran permutation expectation exactly `-1/(n-1)` and
  Geary's `1`, both by **exhaustive enumeration of all n! orderings** for
  `n <= 7`; `2r/3` for the intrazonal cost; a degree of latitude.
* **Exact identities** — gravity under a rectangular kernel *is* cumulative
  opportunity (list equality, not approximation); place-rank inverts cumulative
  opportunity; isochrone bands sum to the last cut-off; E2SFCA with one zone
  reduces bit-exactly to 2SFCA; Shen is 2SFCA; 2SFCA on a uniform grid returns
  `S/P` **exactly** everywhere, and so does a circulant ring; Theil
  between + within == total over 160 seeded decompositions; Hoover == the widest
  Lorenz gap; local Moran sums to `S0 * I`; Dijkstra against Floyd-Warshall.
* **Metamorphic / invariant** — cumulative opportunity monotone in the
  threshold; all relative indices invariant under 1e-6 and 1e6 rescaling while
  Kolm is invariant under translation; the Pigou-Dalton transfer strictly
  lowers Gini, Theil and Atkinson; the logsum dominates the best utility and
  shifts by exactly an added constant; closing a destination removes exactly
  its own gravity term; determinism across repeated calls and label orderings.
* **Longhand definitions, to break the circularity.** The kernel zoo used to
  be pinned only by `decay_x(...) == get_decay("x", ...)(...)` — both sides from
  the same module, so an exponent that was wrong in the raw function *and* in the
  registry agreed with itself. Every kernel is now checked against its
  mathematical definition written out in the test (`1/(1 + (c/alpha)^beta)`,
  `exp(-c^2/(2 beta^2))`, …) and the registry dispatch is checked separately,
  including that no two kernels agree at a probe cost. Same treatment for
  2SFCA / E2SFCA / 3SFCA / M2SFCA / the variable-catchment variant and for Shen,
  each against an independent longhand two-step reference rather than against
  each other.
* **Differential** — `numpy.trapezoid` quadrature of the Lorenz curve and a
  vectorised outer-difference Gini, `numpy.percentile`, `numpy.average`;
  `scipy.stats.norm` for the cumulative-Gaussian kernel and `scipy.stats.gmean`
  for Atkinson; **`scipy.spatial.cKDTree`** for the kNN and distance-band
  neighbour sets; **`networkx.single_source_dijkstra_path_length`** for the
  from-scratch Dijkstra, on a deliberately disconnected graph so the
  unreachable-pair branch is compared too; **`scipy.stats.entropy`** for the
  Theil family (Theil T and L are exactly the two Kullback-Leibler divergences
  between the value shares and the uniform); **`scipy.stats.kendalltau`** for the
  vulnerability ranking and **`spearmanr`** for the winner/loser split; and the
  spherical **law of cosines** — algebraically independent of the half-angle
  haversine form — over 2000 seeded global pairs.
* **Analytic population values.** Stronger than re-implementing the same sum in
  NumPy: the target is the closed-form index of a *named distribution*, so an
  error in the estimator cannot hide behind a matching re-implementation.
  `Gini(Pareto(alpha)) = 1/(2 alpha - 1)`,
  `Gini(lognormal(sigma)) = 2 Phi(sigma/sqrt 2) - 1`,
  `Theil_T(lognormal) = sigma^2/2`,
  `Atkinson(eps, lognormal) = 1 - exp(-eps sigma^2/2)`,
  `Gini(exponential) = 1/2`, `Gini(uniform) = 1/3`, plus the textbook
  `1, 2, 3, 4, 5 -> 4/15` by hand. Sampled on deterministic midpoint-quantile
  grids so the tests can assert the *convergence* as well as the tolerance, with
  one seeded 200 000-draw Monte-Carlo run alongside.

**Why the esda test is gone.** `test_morans_i_matches_esda` was gated on
`esda` + `libpysal`, neither of which is installable in this environment, so it
could never run and was the file's only skip. It is replaced by the same
definition written in explicit NumPy linear algebra —
`I = (n/S0) (z'Wz)/(z'z)` with `z = x - mean(x)` and `S0 = sum(W)` — plus the
matching forms for **Geary's C** (which previously had no direct value oracle at
all, only the exact `E[C] = 1` permutation expectation) and the **LISA** (which
had only the `sum == S0 * I` identity). Measured over 60 seeded configurations
spanning all three weight builders crossed with row-standardisation, the worst
relative disagreement is 1.1e-14 for Moran, 6.3e-16 for Geary and 5.0e-13 for
the LISA. No new dependency, and strictly more coverage than the skip provided.

**Defects the suite caught, round two.** (5) `calibrate_decay` treated a fully
underflowed kernel as "implied mean cost 0". Since the implied mean is a convex
combination of the costs its infimum is `min(costs)`, never 0, so that
fabricated a bracket end the kernel cannot reach: bisection then converged on
the underflow boundary and `calibrate_decay([10,20,30,40], 5.0, 'exponential')`
returned `beta = 74.513` with weights `[5e-324, 0, 0, 0]` and an implied mean of
10.0 — silently 100% away from the request, and the *same* beta for every target
below 10. Fixed twice over: the collapsed-kernel fallback is now `min(costs)`,
and the fit is verified afterwards. (6) `decay_log_logistic` leaked a raw
`OverflowError` from `(c/alpha) ** beta` for a small alpha or a large beta,
breaking the `AccessibilityError` contract; the mathematical limit is 0 and it
now returns it. (7) `cost_matrix_from_coords` never validated `circuity` on the
metres branch, so `circuity=0.5` returned a road distance *shorter* than the
great circle and a non-numeric value leaked a `TypeError`. (8) Five routines
squared raw deviations and left double range — see the scale note above.

**What was pinned rather than changed.** The `concentration_index` tie
convention (documented determinism, now also warned about in the docstring); the
half-open `(lower, upper]` isochrone bands, which *provably* partition where a
closed-left convention would double-count a boundary cost; the closed-on-the-
right distance band and E2SFCA zones; the strict `< line` FGT poverty test; the
Hope (1968) `+1` on the sampled permutation p-value, absent from the exhaustive
branch because that branch is exact; and the median (not mean) split in
`_half_split_change`, which is what "worse-off half versus better-off half"
means.

**Mutants that used to survive and now do not.** Twenty-seven, verified one at a
time by patching the template and re-running: the log-logistic exponent and its
reciprocal; `mean_decay_cost` dropping its weight vector; the haversine kernel
without `cos(phi1) cos(phi2)` (the old test probed only the equator and
`dlon == 0`, where the factor is invisible); the variable-catchment radius keying
swapped, both `>=` capture boundaries, and `max_cost` ignored; the exclusive
`cumulative_opportunity` arm; `row_standardised` dropped from both weight
builders; the distance band's inclusivity; FGT ignoring `alpha` and its poverty
boundary; the composite poverty index's weight normalisation and its
affordability threshold; the poverty line's `mean` arm; Kolm's `/alpha`; the
20:20 cut-points moved to 25:75; `utility_matrix` dropping `theta`;
`intrazonal_cost`'s factor; `impact_incidence` ignoring its tolerance; the
median/mean half-split; E2SFCA's first-matching band; the isochrone bands'
half-openness; the Hope correction; `LorenzCurve.area`; and
`concentration_curve` ranking by value instead of by need.

**Defects the suite caught, round one.** (1) `calibrate_decay` bracketed its scale
parameter downwards from a hard-coded `1e-9`; for the Gaussian and log-logistic
kernels that collapses every weight to zero and the calibration died inside
`mean_decay_cost`. It now brackets from `1.0` by halving and treats a collapsed
kernel as the extreme end of the bracket. (2) Writing the primal/dual oracle
exposed that `time_to_nth_opportunity` used a running total where
`cumulative_opportunity` used `math.fsum`, so the two could disagree by an ulp
at the boundary; the prefix is now summed with `fsum`. (3) `isochrone_bands`
double-counted zero-cost destinations when the first band edge was `0.0` (a
`lower == 0.0` sentinel standing in for "first band"). (4) The Moran
constant-surface guard was an absolute `m2 == 0.0`; it is now scale-relative.
Two further failures turned out to be **test** bugs, not template bugs — the
Palma ratio of an all-zero surface is `0.25` (an all-zero surface is perfectly
equal, so the share ratios take their equality values) and a "checkerboard"
that was really vertical stripes — and the tests were corrected rather than the
module weakened.

```bash
python accessibility_equity_template.py     # demo, prints "demo OK"
pytest _stress_test.py -q -rs               # 251 passed, no skips
PYTHONHASHSEED=1010 pytest _stress_test.py -q
cd ../../../.. && pytest algorithms/domains/urban/accessibility_equity/ -q
```

The last line matters: from the repository root `algorithms.core` *is*
importable, so the two optional core-interop branches go live. The demo's
cross-checks against `algorithms.core.graph.geospatial_graphs` (max haversine
disagreement 1.8e-12 m over the demo city) and
`algorithms.core.graph.shortest_paths` (max Dijkstra disagreement exactly 0.0)
used to be asserted only by the demo's own `print` statements, which no test
read. They are tests now, and they are written so that they assert the local
formula against the spherical law of cosines when the core is absent — so they
stay meaningful in both environments without skipping.

---

## References

* Hansen, W. G. (1959). *How accessibility shapes land use.* JAIP 25(2).
* Geurs, K. & van Wee, B. (2004). *Accessibility evaluation of land-use and
  transport strategies.* Journal of Transport Geography 12(2).
* Luo, W. & Wang, F. (2003). *Measures of spatial accessibility to health care
  in a GIS environment.* Environment and Planning B 30(6). (2SFCA)
* Luo, W. & Qi, Y. (2009). *An enhanced two-step floating catchment area
  (E2SFCA) method.* Health & Place 15(4).
* Wan, N., Zou, B. & Sternberg, T. (2012). *A three-step floating catchment
  area method.* International Journal of GIS 26(6).
* Delamater, P. L. (2013). *Spatial accessibility in suboptimally configured
  health care systems: a modified 2SFCA method.* Health & Place 24.
* Shen, Q. (1998). *Location characteristics of inner-city neighborhoods and
  employment accessibility of low-wage workers.* Environment and Planning B 25.
* Ben-Akiva, M. & Lerman, S. (1985). *Discrete Choice Analysis.* MIT Press.
  (the logsum as consumer surplus)
* Theil, H. (1967). *Economics and Information Theory.* North-Holland.
* Atkinson, A. B. (1970). *On the measurement of inequality.* JET 2(3).
* Kolm, S.-C. (1976). *Unequal inequalities.* JET 12(3).
* Foster, J., Greer, J. & Thorbecke, E. (1984). *A class of decomposable
  poverty measures.* Econometrica 52(3).
* Lucas, K., Mattioli, G., Verlinghieri, E. & Guzman, A. (2016). *Transport
  poverty and its adverse social consequences.* Transport 169(6).
* Anselin, L. (1995). *Local indicators of spatial association -- LISA.*
  Geographical Analysis 27(2).
* Rawls, J. (1971). *A Theory of Justice.* Harvard University Press.
* Levinson, D. (1998). *Accessibility and the journey to work.* Journal of
  Transport Geography 6(1). (the dual, place-rank view)
* Hope, A. C. A. (1968). *A simplified Monte Carlo significance test
  procedure.* JRSS B 30(3). (the `+1` on a sampled permutation p-value)

---

*Pure standard library: `math`, `random`, `itertools`, `heapq`, `dataclasses`,
`typing`. `accessibility_equity_template.py` — `__version__ = "1.0.0"`, 117
public names in `__all__`, no third-party imports anywhere.*
