# Travel Demand

Where the trips come from, and where they go. Everything downstream of this module
takes an origin–destination matrix as a given: `traffic_assignment` routes one over a
network, `traffic_flow` pushes the routed flow through a conservation law,
`emissions_environment` burns the fuel. This module is the place the matrix is
actually **made** — trip generation, trip distribution and mode choice, the first
three steps of the classical four-step model, plus the feedback loop that closes them
and the inverse problem that pulls a matrix back out of a handful of roadside counts.

It is a **zero-dependency**, pure standard-library build. The IRLS engine behind the
Poisson and negative-binomial trip-rate models, the Gaussian elimination behind the
generalised-least-squares matrix estimator, the golden-section profile likelihood for
the NB dispersion, the Gumbel sampler that demonstrates the logsum really is an
expected maximum — all of it is written by hand. No NumPy, no SciPy, no statsmodels,
no pandas. Copy `travel_demand_template.py` into a competition, a hackathon, an
interview or an embedded sandbox and it runs.

```python
from travel_demand_template import (
    Zone, ZoneSystem, trip_ends, calibrate_deterrence,
    doubly_constrained_gravity, multinomial_logit, Alternative,
)

zones = ZoneSystem([
    Zone("CBD",      households=1200, employment=9800, x=0.0,  y=0.0),
    Zone("Suburb-E", households=4200, employment=1500, x=7.5,  y=1.1),
    Zone("Suburb-W", households=3600, employment=1200, x=-7.9, y=0.6),
])
costs = zones.cost_matrix(speed=0.5, fixed=2.0)            # km/min -> minutes

ends = trip_ends(zones, {"households": 0.62}, {"employment": 0.58})
cal  = calibrate_deterrence(costs, ends.productions, ends.attractions,
                            target_mean_cost=12.0)          # fit beta to a survey
trips = doubly_constrained_gravity(ends.productions, ends.attractions,
                                   costs, cal.function)     # the OD matrix

split = multinomial_logit(
    [Alternative("car", {"time": 22.0, "cost": 3.40}),
     Alternative("bus", {"time": 38.0, "cost": 1.80}, constant=-0.55)],
    {"time": -0.045, "cost": -0.28})
print(trips.mean_cost, split.as_dict(), split.logsum)
```

---

## Pick

| You want to… | Reach for | Notes |
|---|---|---|
| Turn land use into trip ends | `trip_ends`, `cross_classification_trips` | Rates per attribute, or a category table |
| Fit trip rates to survey counts | `poisson_glm`, `negbin_glm` | IRLS from scratch; NB dispersion by profile likelihood |
| Distribute trips over a cost matrix | `doubly_constrained_gravity` | Both margins exact; the default choice |
| Grow an observed matrix to new margins | `detroit`, `furness` | Both land on the entropy-maximising answer |
| …the way a 1950s report did | `fratar`, `average_growth_factor` | Margin-correct, different interior — see Design notes |
| Fit any matrix to any margins | `furness`, `ipf` | 2-D with diagnostics, or N-D nested lists |
| Split trips between modes | `multinomial_logit`, `nested_logit` | Nesting fixes IIA; scales are checked for consistency |
| Measure accessibility | `logsum_accessibility`, `composite_cost` | The logsum *is* the expected maximum utility |
| Choose a departure time | `departure_time_choice`, `peak_spreading` | Small schedule delay; exact capacity fixed point |
| Close the loop on congestion | `four_step_model` | MSA damping or plain iteration, gap traced per pass |
| Get a matrix out of link counts | `gls_matrix_estimation`, `me2_estimation` | Exact when the counts are complete |
| …on a matrix too big to invert | `spiess_gradient` | Multiplicative descent, analytic optimal step |
| Audit two matrices against each other | `compare_matrices`, `geh_matrix` | RMSE, %RMSE, R², slope, GEH |
| Invent a plausible demand scenario | `generate_scenario` | Commute, school, stadium, weather, incident — all seeded |
| Go from matrix to agents | `integerise_matrix`, `expand_to_trip_list` | Sums back to the matrix exactly |

---

## Install

Nothing to install. The module is a single file on the Python standard library:

```bash
python travel_demand_template.py     # runs the demo, prints "demo OK"
pytest _stress_test.py -q            # runs the oracle-based test suite
```

`pytest>=7.0` is the only uncommented line in `requirements.txt`, and it is needed
only to run the suite. NumPy, SciPy and statsmodels are optional: when they happen to
be importable the suite adds a differential tier that pins this implementation to
them, and when they are absent that tier skips cleanly.

---

## What's inside

### 0 — Validation and numerics

`TravelDemandError` and the value guards, a dense `solve_linear` / `invert_matrix`
(partial pivoting, Gauss–Jordan inverse), an overflow-safe `logsumexp`,
golden-section search, bisection, log-gamma and digamma.

### 1 — Zones, trip matrices and matrix algebra

`Zone` / `ZoneSystem` (land use, centroid geometry, distance and generalised-cost
matrices) and `TripMatrix` (labelled dense table), plus margins, transposes,
symmetrisation, intrazonals and scaling.

### 2 — Trip generation

Cross-classification (`CrossClassTable`) for the non-parametric tradition, and the
regression tradition done properly: `linear_glm`, `poisson_glm` and `negbin_glm`
fitted by **iteratively reweighted least squares** — Fisher scoring on the
exponential-family likelihood, with working weights `mu` (Poisson) and
`mu / (1 + alpha mu)` (NB2). The NB dispersion is not a GLM parameter, so it is found
by **profile likelihood**: golden section over `log alpha`, re-fitting beta at every
step. Standard errors, deviance, AIC/BIC, an analytic `negbin_alpha_score` and a
`negbin_dispersion_profile` for plotting. Plus production–attraction balancing.

### 3 — Deterrence functions

`PowerDeterrence`, `ExponentialDeterrence`, `CombinedDeterrence`,
`TannerDeterrence`, `TopLogDeterrence`, `LogNormalDeterrence` and a callable-wrapping
base class, each with an **analytic derivative** and a cost elasticity.
`calibrate_deterrence` fits beta so that a doubly-constrained model reproduces an
observed mean trip cost, bracketing and expanding automatically.

### 4 — Trip distribution

The gravity model in all three constraint flavours; `furness` (2-D IPF with
diagnostics) and `ipf` (N-dimensional, used by the population synthesiser);
`entropy_maximising_matrix` and the Wilson / relative entropies; the growth-factor
family (`uniform_growth`, `average_growth_factor`, `fratar`, `detroit`);
`intervening_opportunities` (Schneider) and `competing_destinations`
(Fotheringham); trip-length distributions and the mean trip cost.

### 5 — Matrix operations

`SectorSystem` and sectoring / aggregation / disaggregation, `k_factors`, intrazonal
estimation (fixed share or nearest-neighbour), and the standard audit set:
`compare_matrices` (RMSE, %RMSE, MAE, R², correlation, slope, intercept) and
`geh` / `geh_matrix`.

### 6 — Mode choice

`multinomial_logit` and `nested_logit` with proper logsums and scale parameters,
including the `nesting_consistency` check that `theta = mu / mu_m` lies in `(0, 1]`.
`incremental_logit` / `pivot_point` for base-year-anchored forecasting, direct and
cross elasticities, `value_of_time`, `composite_cost`, `logsum_accessibility`, and
`monte_carlo_logit` — a seeded Gumbel simulation that demonstrates the logsum is the
expected maximum utility.

### 7 — Departure-time choice and peak spreading

`TimeSlice`, the Vickrey/Small `schedule_delay_utility` (early is cheaper than late,
which is why peaks have a long leading shoulder), a logit over time slices,
`peak_hour_factor`, time-of-day splits, PA-to-OD conversion and capacity-driven
`peak_spreading`.

### 8 — The four-step loop

Generation → distribution → mode choice → *(your assignment)* → congested costs →
round again, with method-of-successive-averages damping, an explicit convergence gap
per pass, and a `cost_function` hook so a real assignment can be dropped in where the
built-in BPR default sits.

### 9 — OD estimation from link counts

`AssignmentProportions` as the bridge, then `gls_matrix_estimation`, `me2_estimation`
(Van Zuylen–Willumsen entropy maximisation), `path_flow_estimator`,
`bayesian_matrix_update` (with posterior variances per cell) and `spiess_gradient`,
all reported through one `ODEstimationResult` with `count_fit` statistics.

### 10 — Demand scenarios

Seeded generators for a commuting peak, a school run, a stadium emptying, a weather
shock and an incident. Zone-level variability is drawn from a **stratified** Latin
hypercube, not plain Monte Carlo, so a 20-zone draw covers the range.

### 11 — Synthetic population and trip lists

`synthesise_population` (IPF to zone / size / car marginals, then largest-remainder
integerisation **per zone** so every zone control total is honoured), and
`expand_to_trip_list` / `trip_list_to_matrix`, which round-trip exactly.

### 12 — Payloads

`build_travel_demand_payload` and `build_a2ui_payload` — JSON-able dicts with the
matrix, margins, trip-length bands, deterrence curve, choice model, feedback trace
and count fit, ready for a frontend.

---

## Recipes

**Fit trip rates to a household survey and forecast.**

```python
X = design_matrix([[cars, hh_size] for cars, hh_size in survey])
fit = negbin_glm(X, trips, names=["intercept", "cars", "size"])
print(fit.summary(), fit.dispersion)          # alpha by profile likelihood
future = glm_predict(fit, design_matrix(forecast_households))
```

**Calibrate a gravity model against an observed trip-length distribution.**

```python
target = mean_trip_cost(observed_matrix, costs)
cal = calibrate_deterrence(costs, P, A, target, kind="combined",
                           extra={"alpha": -0.4})
tld = trip_length_distribution(cal.matrix, costs, bins=12)
```

**Split by mode with a nest that actually binds.**

```python
nests = [Nest("private", ("car", "cycle"), scale=1.0),
         Nest("public",  ("bus", "rail"),  scale=2.4)]
print(nesting_consistency(nests, scale=1.0).message)
nl = nested_logit(modes, nests, coefficients, scale=1.0)
```

**Recover a matrix from roadside counts.**

```python
asg   = AssignmentProportions(links=links, shape=(n, n), proportions=props)
est   = gls_matrix_estimation(prior, counts, asg, prior_weight=1.0,
                              count_weight=25.0)
print(est.fit.pct_geh_under_5, est.prior_distance)
```

**Hand the matrix to an agent-based model.**

```python
whole = integerise_matrix(apply_time_of_day(daily, 0.28, direction_split=0.75))
trips = expand_to_trip_list(whole, mode_shares=shares, slices=slices, seed=99)
assert trip_list_to_matrix(trips, (n, n)) == whole      # exactly
```

---

## Design notes

**The matrix is the product, not the input.** Every function here produces a trip
matrix, consumes one to produce another, or measures one. Routing is somebody else's
job — `traffic_assignment` takes this module's output as given, and the
`cost_function` hook in `four_step_model` is where the two meet.

**Units live in beta, not in the code.** Costs are generalised costs in whatever
single unit you pick. Deterrence parameters carry the inverse unit, so multiplying
every cost by 1e6 and dividing beta by 1e6 leaves the distributed matrix
bit-identical. The suite asserts exactly that at 1e-6 and 1e6. Nothing in the module
compares a cost, a trip count or a utility to an absolute epsilon; `_EPS` guards
dimensionless quantities only, and every other tolerance is relative to the magnitude
of the thing being measured.

**Furness and Detroit agree; Fratar does not.** All four growth-factor methods
reproduce `g * T` exactly under uniform growth. Under non-uniform growth Detroit
multiplies each cell by `E_i F_j / G`, which is **separable** into a row factor times
a column factor, so it preserves the odds ratios and lands on exactly the same
entropy-maximising matrix as Furness (pinned to 1e-9 by the suite). Fratar and the
average-factor method multiply by `(L_i + L_j) / 2` and `(E_i + F_j) / 2`, which are
not separable: they reach the target margins but a different interior — a few tenths
of a trip on a 5×5 table, more on a real one. The suite pins *both* sides of that.

**The nested logit is checked in both directions.** With every nest scale equal to the
upper scale, the nested model collapses onto the MNL to 1e-12 — the nesting is
vacuous and the code had better prove it. With a binding nest, alternatives in
*different* nests must **violate** IIA while alternatives in the *same* nest still
obey it to 1e-12. A model that only ever passed the first test would be an MNL with
extra steps.

**The logsum is not a normalising constant.** It is the expected maximum utility of
the choice set, which is why it is the right accessibility measure and the right
composite cost to feed back into distribution; `monte_carlo_logit` draws Gumbel errors
and averages the realised maximum, converging to `logsum + gamma / mu`. It also means
the composite cost of a choice set lies *below* its cheapest member and can go
negative, so the four-step loop shifts the whole composite matrix by one **reported**
constant — which, under an exponential deterrence, the doubly-constrained balancing
factors absorb exactly.

**MSA buys a guarantee, not speed.** Its step falls like `1/k`, so a feedback gap of
1e-9 costs hundreds of passes. Plain fixed-point iteration (`msa=False`) gets there in
about a dozen when the cost map is a contraction, and diverges when it is not. Both
are offered, the gap is traced per iteration, and `converged` is never claimed merely
because `max_feedback` ran out.

**Ties are broken on explicit keys, everywhere.** No number in this module depends on
dict or set iteration order: category tables sort on the string form of their keys,
intervening opportunities ranks destinations by `(cost, index)`, largest-remainder
integerisation sorts on `(-fraction, i, j)`, and every stochastic generator takes a
seed and uses `random.Random`. The demo output is byte-identical under
`PYTHONHASHSEED` 0, 1 and 1010.

---

## Testing

`_stress_test.py` is an oracle-based suite — **147 tests**, all green, and green under
`PYTHONHASHSEED` 0, 1 and 1010. There is not a single golden output in it.

- **Closed forms** — a doubly-constrained gravity model on uniform costs returns
  exactly `T_ij = O_i D_j / T` (the flagship, in one pass, no iteration); a 2×2 Furness
  is hand-solved through the quadratic implied by its invariant odds ratio; an
  intercept-only Poisson GLM returns exactly `log(mean(y))`; the logsum of two
  alternatives is `V1 + log(1 + exp(V2 - V1))`; the Gumbel expected maximum is
  `logsum + gamma/mu`; GEH, the peak-hour factor, the value of time and the Small
  schedule-delay utility all reproduce their formulae; every deterrence function agrees
  with its own analytic derivative to 1e-6 relative.
- **Exact identities** — Furness hits both margins to machine precision and preserves
  the seed odds ratios; a nested logit whose scales all equal `mu` collapses onto the
  MNL to 1e-12; logit probabilities are invariant under translating every utility and
  under rescaling utility and scale together; GLS, ME2, Spiess and the path-flow
  estimator each invert a known matrix from complete counts; a trip list expanded from
  a whole-number matrix sums back to it exactly; sectoring and disaggregation
  round-trip; a four-step loop with no congestion has a gap of exactly 0.0.
- **Metamorphic** — cost-unit and trip-scale invariance at 1e-6 and 1e6; Detroit ≡
  Furness while Fratar ≢ Furness; IRLS reaches the same optimum as a direct hill-climb
  on the likelihood; a 1 % cost change moves demand by the analytic elasticity (direct
  *and* cross); the NB dispersion profile is unimodal and peaks at the fitted alpha;
  congestion raises costs and capacity lowers them; more counts improve an estimate;
  Bayesian updating shrinks every posterior variance.
- **Differential** (behind `pytest.importorskip`) — IPF against a NumPy loop;
  `solve_linear` / `invert_matrix` against `numpy.linalg`; `linear_glm` against
  `numpy.linalg.lstsq`; `poisson_glm` against `statsmodels` (coefficients, standard
  errors, log-likelihood, deviance and AIC to 1e-8); `negbin_glm` against the
  statsmodels NB2 GLM at a fixed alpha and against the full `NegativeBinomial` MLE for
  the profile-likelihood alpha; the logit against `scipy.special.softmax`; calibration
  against `scipy.optimize.brentq`; and the trip-length density against
  `numpy.trapezoid` (NumPy 2.x removed `trapz`).

```bash
python travel_demand_template.py                 # demo OK
pytest _stress_test.py -q                        # 147 passed
PYTHONHASHSEED=1010 pytest _stress_test.py -q    # 147 passed
```

**Real defects this work caught and fixed.** Four, all in the template:

1. **`peak_spreading` never converged.** The damped share iteration settled into a
   two-cycle and the gap stalled at ~0.25 forever — 200 iterations and 20 000
   iterations returned the identical wrong answer. Replaced with an exact two-level
   bisection: the congestion penalty on a slice depends only on that slice, so the
   whole system collapses to one scalar level, solved to a fixed-point residual of
   9e-13. Pinned by `test_peak_spreading_preserves_the_total_and_solves_its_fixed_point`.
2. **GLS blew up on unidentified cells.** With `prior_weight=0`, any cell that no count
   touches (every intrazonal, typically) left a zero row in the normal equations and
   the solve raised "matrix is singular". Such cells are now held at their prior value,
   counted in `detail["heldAtPrior"]`, and the reduced system is solved. Pinned by
   `test_gls_recovers_a_known_matrix_from_complete_counts`.
3. **The four-step loop crashed on its own composite cost.** `logsum / beta_cost` is
   below the cheapest mode by construction and went negative, which the deterrence
   function correctly refused. Now shifted by a single reported constant that the
   balancing factors absorb exactly.
4. **Population synthesis missed its zone control totals.** Integerising the whole
   three-way table at once pinned only the grand total and let zones drift by ±2
   households. Now integerised zone by zone. Pinned by
   `test_synthetic_population_matches_its_zone_control_totals`.

A fifth finding was not a bug but a correction to the module's own prose: the initial
comment claimed Fratar and Detroit were "accelerations, not different models". Detroit
is; Fratar is not, and the suite now proves the difference in both directions.

---

## References

- Ortúzar & Willumsen — *Modelling Transport* (the four-step model end to end).
- Wilson (1967) — *A statistical theory of spatial distribution models* (entropy
  maximisation and the derivation of the gravity model).
- Furness (1965); Fratar (1954) — iterative proportional fitting and the Fratar method.
- Ben-Akiva & Lerman — *Discrete Choice Analysis*; McFadden (1974); Williams (1977) —
  the logsum as expected maximum utility.
- Small (1982) — *The scheduling of consumer activities: work trips* (schedule delay).
- Cameron & Trivedi — *Regression Analysis of Count Data* (Poisson/NB2 GLMs, IRLS).
- Cascetta & Nguyen (1988) — a unified framework for OD estimation from traffic counts.
- Van Zuylen & Willumsen (1980) — the most likely trip matrix from traffic counts (ME2).
- Spiess (1990) — a gradient approach for the OD matrix adjustment problem.
- Fotheringham (1983) — competing destinations; Stouffer (1940) / Schneider —
  intervening opportunities.
- DfT TAG Unit M2 — *Variable Demand Modelling* (deterrence families, realism testing,
  pivot-point forecasting).

---

*Pure Python, standard library only (`json`, `math`, `random`, `dataclasses`,
`typing`). `__version__ = "1.0.0"`, 153 public names in `__all__`.*
