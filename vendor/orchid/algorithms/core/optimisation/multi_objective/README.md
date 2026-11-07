# Multi-Objective Optimisation

A self-contained, **zero-dependency** toolkit for **optimisation with several
conflicting objectives at once** — on the Python standard library alone. When you
cannot collapse "cheap *and* fast *and* reliable" into a single number, there is
no lone optimum but a whole **Pareto front** of trade-offs; this module gives you
the machinery to *find* that front and the decision tools to *pick* a point off
it.

Where the sibling `nonlinear_programming` / `stochastic_optimisation` packages
optimise one scalar objective, here every tool is built around **dominance,
Pareto fronts and quality indicators**. No NumPy, no SciPy, no pymoo / DEAP /
jMetal — it drops straight into a competition / hackathon / interview / embedded
sandbox: just copy `multi_objective_template.py`.

```python
from multi_objective_template import nsga2, zdt1, hypervolume, inverted_generational_distance, zdt1_front

res = nsga2(zdt1(30), pop_size=100, n_gen=250, seed=0)   # a real-coded 2-objective problem
print(len(res), "non-dominated solutions")
print("hypervolume :", hypervolume(res.F_min, [1.1, 1.1]))
print("IGD         :", inverted_generational_distance(res.F_min, zdt1_front(500)))
```

Every solver returns a uniform **`MOResult`** (the non-dominated decision vectors
`X` and objective vectors `F`, evaluation / generation counts, an optional
indicator `history` and a `meta` dict), and **every randomised routine takes a
`seed`** (or an `rng`) so runs are exactly reproducible, independent of
`PYTHONHASHSEED`.

---

## What's inside

Ten composable layers, from the raw Pareto primitives up to six state-of-the-art
evolutionary algorithms and the decision-making tools that follow them.

### 1. Pareto machinery

The primitives every multi-objective method is built on:

- `dominates` / `weakly_dominates` / `epsilon_dominates` — the dominance relations
  (with optional per-objective `senses`).
- `constraint_dominates` — Deb's constraint-domination (feasible beats infeasible;
  among infeasibles the smaller violation wins).
- `non_dominated_sort` — Deb's fast non-dominated sort into successive fronts
  (optionally constraint-aware), `crowding_distance`, `find_pareto_front`,
  `pareto_ranks`.

### 2. Quality indicators

How good *is* a Pareto-set approximation? — measured against a reference front or
a reference point:

- `hypervolume` — the gold-standard indicator (captures convergence **and**
  spread, Pareto-compliant). Exact `O(n log n)` sweep in 2-D and the exact **WFG**
  recursion in any higher dimension; `hypervolume_monte_carlo` estimates it for
  large, high-dimensional fronts, and `hypervolume_contributions` gives each
  point's exclusive share.
- `inverted_generational_distance` / `generational_distance` and their
  Pareto-compliant `igd_plus` / `gd_plus` variants.
- `epsilon_indicator` (additive) and `epsilon_indicator_mult` (multiplicative).
- `spacing`, `spread`, `r2_indicator`, `set_coverage`, `maximum_pareto_front_error`.

### 3. Reference directions & scalarisation

Turn a vector objective into scalars along structured directions:

- `das_dennis` — the simplex-lattice weight vectors that underpin NSGA-III and
  MOEA/D; `reference_directions` adds Deb & Jain's two-layer construction for many
  objectives.
- `weighted_sum`, `tchebycheff`, `augmented_tchebycheff`, `pbi` (penalty boundary
  intersection), `asf` (achievement scalarising function).

### 4. Normalisation

- `ideal_point`, `nadir_point`, `extreme_points`, `normalize` — estimate the ideal
  / nadir corners and scale objectives into the unit cube (as NSGA-III does each
  generation).

### 5. Genetic operators

Real-coded by default, but every one is **overridable**, so a combinatorial
problem drops in by passing your own `sampling` / `crossover` / `mutation`:

- `sbx_crossover` (simulated binary crossover), `polynomial_mutation`,
  `de_operator` (differential evolution `rand/1/bin`), `binary_tournament`,
  `random_sampling`.

### 6. Algorithms

Six MOEAs behind one consistent signature, each taking a `MOProblem` and returning
an `MOResult`:

- `nsga2` — **NSGA-II**: elitist non-dominated sorting + crowding distance, the
  default work-horse for 2–3 objectives (constraint-domination automatic).
- `nsga3` — **NSGA-III**: reference-point niching for **many** objectives (4+),
  with adaptive ideal/hyperplane normalisation.
- `moead` — **MOEA/D**: decomposition into cooperating scalar subproblems
  (`tchebycheff` / `pbi` / `weighted_sum`), DE + polynomial-mutation reproduction.
- `spea2` — **SPEA2**: strength + k-th-nearest-neighbour density, distance-based
  archive truncation.
- `sms_emoa` — **SMS-EMOA**: steady-state, selects on **hypervolume
  contribution** (directly maximises HV).
- `ibea` — **IBEA**: indicator-based fitness from the additive epsilon indicator.

```python
from multi_objective_template import moead, dtlz2, das_dennis
res = moead(dtlz2(12, 3), n_partitions=12, aggregation="pbi", n_gen=300, seed=0)
```

### 7. Decision making (MCDM)

Once you *have* the front, choose from it:

- `topsis` — closeness to the ideal / distance from the anti-ideal.
- `compromise_programming` — nearest to the ideal in a weighted `L_p` norm.
- `pseudo_weights` — which effective trade-off each solution represents.
- `weighted_selection` — the best solution for a stated linear weighting.
- `knee_point` (2-D) and `high_tradeoff_points` (any dimension) — the "biggest
  bang for the buck" solutions.
- `ParetoArchive` — a streaming non-dominated archive with optional additive
  **epsilon-dominance** to bound its size.

### 8. Benchmarks

Ready-made problems for templating and testing, each returning a `MOProblem`, with
analytic Pareto fronts where they are known:

- `zdt1` / `zdt2` / `zdt3` (convex / non-convex / disconnected 2-objective),
  `dtlz1` / `dtlz2` (scalable to any number of objectives), `schaffer`, `kursawe`,
  `fonseca_fleming`, and the **constrained** `binh_korn`.
- `zdt1_front`, `zdt2_front`, `zdt3_front`, `dtlz2_front` — analytic fronts to use
  as IGD / GD oracles.

---

## Recipes

**Many-objective optimisation** (4+ objectives) with reference-point niching:

```python
from multi_objective_template import nsga3, dtlz2, reference_directions
res = nsga3(dtlz2(n_var=14, n_obj=5), n_partitions=6, n_gen=400, seed=0)
```

**Constrained** multi-objective optimisation — constraints are handled by
constraint-domination automatically:

```python
from multi_objective_template import nsga2, binh_korn
res = nsga2(binh_korn(), pop_size=100, n_gen=200, seed=0)   # every X is feasible
```

**Your own problem**, mixing minimised and maximised objectives:

```python
from multi_objective_template import MOProblem, nsga2
problem = MOProblem(
    n_var=3, n_obj=2,
    evaluate=lambda x: [cost(x), throughput(x)],
    bounds=[(0, 10)] * 3,
    senses=["min", "max"],           # maximise throughput; negated internally
)
res = nsga2(problem, pop_size=80, n_gen=150, seed=1)
```

**A combinatorial problem** — pass custom operators (bounds may be `None`):

```python
res = nsga2(problem, sampling=my_sampler, crossover=my_crossover,
            mutation=my_mutation, pop_size=100, n_gen=200, seed=0)
```

**Track convergence** and then **pick a solution**:

```python
res = nsga2(zdt1(30), pop_size=100, n_gen=250, ref_point=[1.1, 1.1],
            track_hv=True, seed=0)
print(res.history[-1])                        # hypervolume per generation
best, scores = topsis(res.F, weights=[0.5, 0.5])
knee = knee_point(res.F)
```

---

## Design notes

- **Everything is minimised internally.** A `MOProblem` may declare per-objective
  `senses` of `'min'` / `'max'`; maximised objectives are negated on the way in and
  un-negated on the way out, so the whole indicator / dominance API can assume
  minimisation. `MOResult.F` is in your original senses; `MOResult.F_min` is the
  minimisation-normalised form to feed straight into `hypervolume` / IGD.
- **Uniform result & pluggable operators.** Swap `nsga2` for `moead` for `spea2`
  without touching the surrounding code, and swap the real-coded operators for your
  own to attack combinatorial problems.
- **Constraints everywhere.** All six algorithms respect a problem's constraints:
  the sorting-based methods (NSGA-II/III, SPEA2, SMS-EMOA) via constraint-domination,
  MOEA/D via a feasibility-first replacement rule, and IBEA by discarding the worst
  violators before any feasible individual — a feasible solution always beats an
  infeasible one, and among infeasibles the smaller total violation wins.
- **Exact where it counts.** Hypervolume is exact (2-D sweep + WFG); the reference
  directions are the exact Das-Dennis lattice; the scalarisations and dominance
  relations are textbook-faithful.
- **Reproducible.** All randomness flows through a seeded `random.Random`; a given
  `seed` reproduces the run exactly, independent of `PYTHONHASHSEED`.
- **Validated inputs.** A bad objective sense, a bounds/variable mismatch, a
  population size below 2, an unknown MOEA/D aggregation, a degenerate Das-Dennis
  request or a negative archive epsilon raises a clear `ValueError`.

---

## Tests / Demo

```bash
python multi_objective_template.py    # runs the demo, prints "demo OK"
pytest _stress_test.py -q             # 67 oracle-based tests
```

The suite is **oracle-based** — no hand-typed magic number where a closed form,
a brute-force reference or an invariant exists:

- **Closed forms** — hypervolume against hand-computed rectangles / boxes, GD/IGD
  against analytically nearest points, Das-Dennis counts against
  `C(p + m - 1, m - 1)`, the scalarisations against pencil-and-paper values.
- **Brute force** — the fast non-dominated sort against the naive definition on
  random clouds; the exact WFG hypervolume against an independent Monte-Carlo
  estimate in 3-D and 4-D.
- **Convergence oracles** — every one of the six MOEAs is driven to a benchmark
  whose true Pareto front is known analytically (ZDT1/2/3, DTLZ2) and must reach
  it (IGD below a threshold), and the set it returns must be internally
  non-dominated. The constrained run must return only feasible solutions.
- **Invariants** — hypervolume is order-invariant and monotone under improvement;
  `IGD(front, front) = 0`; IGD+ is Pareto-compliant; a maximisation problem
  round-trips exactly to its minimisation twin; the epsilon-archive bounds the
  front size; every algorithm reproduces exactly from its seed.

Runs clean under several `PYTHONHASHSEED`s.

## API summary

| Area | Names |
|------|-------|
| Data | `MOProblem`, `Individual`, `MOResult` |
| Dominance | `dominates`, `weakly_dominates`, `epsilon_dominates`, `constraint_dominates`, `non_dominated_sort`, `crowding_distance`, `find_pareto_front`, `pareto_ranks` |
| Indicators | `hypervolume`, `hypervolume_monte_carlo`, `hypervolume_contributions`, `generational_distance`, `inverted_generational_distance`, `igd_plus`, `gd_plus`, `epsilon_indicator`, `epsilon_indicator_mult`, `spacing`, `spread`, `r2_indicator`, `set_coverage`, `maximum_pareto_front_error` |
| Reference / scalarisation | `das_dennis`, `reference_directions`, `weighted_sum`, `tchebycheff`, `augmented_tchebycheff`, `pbi`, `asf` |
| Normalisation | `ideal_point`, `nadir_point`, `normalize`, `extreme_points` |
| Operators | `random_sampling`, `sbx_crossover`, `polynomial_mutation`, `de_operator`, `binary_tournament` |
| Algorithms | `nsga2`, `nsga3`, `moead`, `spea2`, `sms_emoa`, `ibea` |
| Decision making | `topsis`, `compromise_programming`, `pseudo_weights`, `weighted_selection`, `knee_point`, `high_tradeoff_points`, `ParetoArchive` |
| Benchmarks | `zdt1`, `zdt2`, `zdt3`, `dtlz1`, `dtlz2`, `schaffer`, `kursawe`, `fonseca_fleming`, `binh_korn`, `zdt1_front`, `zdt2_front`, `zdt3_front`, `dtlz2_front` |

## References

- Deb, Pratap, Agarwal & Meyarivan (2002) — *NSGA-II*.
- Deb & Jain (2014) — *NSGA-III* (reference-point many-objective).
- Zhang & Li (2007) — *MOEA/D*.
- Zitzler, Laumanns & Thiele (2001) — *SPEA2*.
- Beume, Naujoks & Emmerich (2007) — *SMS-EMOA* (hypervolume selection).
- Zitzler & Künzli (2004) — *IBEA* (indicator-based).
- While, Bradstreet & Barone (2012) — the *WFG* hypervolume algorithm.
- Das & Dennis (1998) — structured reference points on the simplex.
- Deb & Agrawal (1995) — simulated binary crossover.
- Zitzler, Deb & Thiele (2000) — *ZDT*; Deb, Thiele, Laumanns & Zitzler (2005) — *DTLZ*.
- Hwang & Yoon (1981) — *TOPSIS*.
