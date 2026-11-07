# Stochastic Optimisation

A self-contained, **zero-dependency** toolkit for **optimisation under
uncertainty** and optimisation of **noisy / stochastic objectives** — on the
Python standard library alone. Where the sibling `nonlinear_programming` /
`convex_optimisation` modules assume a *deterministic* oracle, here the
objective, its gradient, or the data itself is **random**, so every tool is built
around sampling, noise, variance reduction and risk.

No NumPy, no SciPy, no PyTorch/TensorFlow/JAX, no cvxpy — not even for the
optimisers or the CMA-ES covariance update (the separable variant needs no matrix
factorisation). It drops straight into a competition / hackathon / interview /
embedded sandbox — just copy `stochastic_optimisation_template.py`.

```python
from stochastic_optimisation_template import Adam, minimize_sgd

target = [3.0, -2.0, 1.0]
noisy_grad = lambda x: [(xi - ti) + __import__("random").gauss(0, 0.5)
                        for xi, ti in zip(x, target)]        # a stochastic gradient
res = minimize_sgd(noisy_grad, [0, 0, 0], optimizer=Adam(lr=0.1),
                   n_iter=2000, averaging=True)
print(res.x)          # ~[3, -2, 1] — Polyak-Ruppert averaging kills the noise
```

Every solver returns a uniform **`OptResult`** (best point `x`, objective
estimate `fun`, iteration / evaluation counts, a `converged` flag, a `history`
trace and a `meta` dict), and **every randomised routine takes a `seed`** (or an
`rng`) so runs are exactly reproducible.

---

## What's inside

Eight composable layers, from ML-style gradient methods to risk-averse
programming.

### 1. Stochastic first-order optimisers

The machine-learning workhorses, as stateful `.step(params, grad) -> params`
objects (the PyTorch / Keras pattern — they never mutate their input):

- `SGD` — with (Nesterov) momentum and decoupled weight decay.
- `AdaGrad`, `RMSProp`, `AdaDelta` — adaptive per-coordinate rates.
- `Adam`, `AdaMax`, `AMSGrad`, `Nadam` — the Adam family (bias-corrected moments,
  infinity-norm, non-decreasing second moment, Nesterov-accelerated).

Plus **learning-rate schedules** (`constant_lr`, `step_decay`,
`exponential_decay`, `inverse_time_decay`, `cosine_annealing`,
`polynomial_decay`, `warmup_cosine`), **gradient clipping** (`clip_by_norm`,
`clip_by_value`), and a driver `minimize_sgd` with **Polyak-Ruppert averaging**
and early stopping.

```python
from stochastic_optimisation_template import SGD, cosine_annealing, minimize_sgd
res = minimize_sgd(grad, x0, optimizer=SGD(lr=0.1, momentum=0.9),
                   lr_schedule=cosine_annealing(0.1, t_max=5000),
                   clip_norm=1.0, n_iter=5000, objective=f)
```

### 2. Finite-sum / variance reduction

For `F(x) = (1/n) Σ fᵢ(x)` with a per-component gradient `grad_i(x, i)`:

- `minibatch_sgd` — averaged mini-batch gradients.
- `svrg` — Stochastic Variance-Reduced Gradient (periodic full-gradient anchor).
- `saga` — a per-component gradient memory table.

The variance-reduced pair converge **linearly** on strongly-convex sums, unlike
plain SGD.

### 3. Gradient-free stochastic approximation

When you only have a noisy *function* oracle:

- `spsa` — Simultaneous Perturbation SA: the whole gradient from just **two**
  evaluations per step, regardless of dimension.
- `kiefer_wolfowitz` — coordinate-wise finite-difference SA.
- `robbins_monro` — root finding of a noisy function (quantile tracking, dosage).
- `finite_difference_gradient` — forward / central differences.

### 4. Distribution-based global search

Derivative-free optimisers for noisy, multi-modal black boxes:

- `cross_entropy_method` — fit a diagonal Gaussian to the elite samples, with
  decaying **noise injection** to avoid premature collapse.
- `sep_cma_es` — **separable CMA-ES**: evolution-path adaptation of a
  per-coordinate scale and a global step size, with *no* O(n²) covariance or
  eigendecomposition, so it scales to high dimension.
- `evolution_strategy` — a Natural Evolution Strategy (the "OpenAI-ES"
  estimator) with antithetic sampling and fitness-rank normalisation.

### 5. Sampling & scenarios

Represent and integrate uncertainty:

- `monte_carlo_estimate` — mean, variance, standard error and a confidence
  interval; `control_variate_estimate` and `importance_sampling_estimate` for
  variance reduction.
- `latin_hypercube`, `stratified_uniform`, `antithetic_normal` — variance-reduced
  designs.
- `van_der_corput`, `halton_sequence`, `sobol_sequence` — low-discrepancy
  quasi-Monte-Carlo sequences (Sobol via Bratley-Fox direction numbers and the
  Gray-code construction).
- `scenario_reduction` — Kantorovich (Wasserstein) fast-forward selection: shrink
  a discrete scenario set to `k` representatives, redistributing probability mass.

### 6. Stochastic programming

- `sample_average_approximation` — approximate `min_x E[F(x, ξ)]` by the
  sample-average over `N` scenarios, solved by a derivative-free inner optimiser;
  reports an **independent out-of-sample** objective estimate.
- `two_stage_saa` — two-stage problems `min c·x + E[Q(x, ξ)]` with a caller-supplied
  recourse oracle `Q`.
- `newsvendor` — the canonical single-stage program (critical-fractile order).
- `stochastic_solution_metrics` — the classic value metrics **EVPI** (price of
  perfect information) and **VSS** (value of the stochastic solution).

### 7. Risk-averse optimisation

- `value_at_risk`, `conditional_value_at_risk` (expected shortfall),
  `entropic_risk`, `mean_variance_objective` — coherent and convex risk measures.
- `minimize_cvar` — minimise CVaR via the **Rockafellar-Uryasev** formulation,
  which is jointly convex in `(x, η)` and solved here by stochastic subgradient.

### 8. Stochastic bandits

Online stochastic optimisation of an unknown reward:

- `EpsilonGreedy`, `UCB1`, `ThompsonSampling` (Beta or Normal posterior).
- `run_bandit` — drive a policy against a stochastic environment and trace the
  cumulative regret.

---

## Recipes

**Risk-averse portfolio** — minimise the 95% CVaR of the loss:

```python
from stochastic_optimisation_template import minimize_cvar
res = minimize_cvar(loss, x0, scenario_sampler, alpha=0.95,
                    n_scenarios=1000, bounds=[(0, 1)] * n_assets)
```

**Two-stage stochastic program** with recourse:

```python
from stochastic_optimisation_template import two_stage_saa
res = two_stage_saa(first_stage_cost=c, recourse=Q, x0=x0,
                    scenario_sampler=sample_demand, n_scenarios=500)
```

**Quasi-Monte-Carlo integration** (far fewer samples than plain MC):

```python
from stochastic_optimisation_template import sobol_sequence
pts = sobol_sequence(4096, dim)
integral = sum(f(p) for p in pts) / len(pts)
```

**Tune a noisy simulator** you cannot differentiate (SPSA — two evals per step):

```python
from stochastic_optimisation_template import spsa
res = spsa(noisy_simulation, x0, n_iter=2000, a=0.1, c=0.1, bounds=box)
```

---

## Design notes

- **Uniform result.** Every solver returns an `OptResult`, so you can swap SGD
  for CMA-ES for SPSA without touching the surrounding code.
- **Reproducible.** All randomness flows through a seeded `random.Random`; a given
  `seed` reproduces the run exactly (independent of `PYTHONHASHSEED`).
- **Immutable steps.** Optimiser `.step()` returns a *new* parameter vector and
  never mutates its argument, so a parameter history is trivial to keep.
- **Numerically careful.** `math.fsum` for reductions, log-sum-exp for the
  entropic risk, Acklam's rational inverse-normal-CDF for exact z-scores, and
  antithetic / control-variate / stratified estimators for variance reduction.
- **Validated inputs.** A non-positive learning rate, a momentum outside `[0,1)`,
  a risk level outside `(0,1)`, a population size below 2, a non-positive SPSA /
  Kiefer-Wolfowitz / Robbins-Monro gain, an empty risk sample, too many
  Halton/Sobol dimensions, or a newsvendor price below cost raises a clear
  `ValueError` rather than returning a silently-wrong answer.

---

## Tests / Demo

```bash
python stochastic_optimisation_template.py   # runs the demo, prints "demo OK"
pytest _stress_test.py -q                     # 57 oracle-based tests
```

The suite is **oracle-based** — no hand-typed magic number where a closed form
exists:

- **Closed-form optima** — every optimiser is driven to a known quadratic /
  least-squares minimum (from the normal equations) and the sphere minimum at 0.
- **Distribution oracles** — VaR, CVaR and entropic risk are checked against their
  exact Gaussian formulas, and the newsvendor order against the analytic
  critical-fractile quantile.
- **Low-discrepancy** — Halton and Sobol integrate a smooth function to a tight
  absolute-error bound that plain Monte-Carlo cannot match at the same sample
  count.
- **Invariants** — CVaR ≥ VaR, VaR monotone in `α`, scenario-reduction
  probabilities sum to 1, EVPI / VSS ≥ 0, control variates never raise variance,
  and the smart bandits beat uniform-random with sublinear regret.
- **Determinism** — every randomised routine reproduces exactly from its seed.

Runs clean under several `PYTHONHASHSEED`s.

## API summary

| Area | Names |
|------|-------|
| Result | `OptResult` |
| Optimisers | `SGD`, `AdaGrad`, `RMSProp`, `AdaDelta`, `Adam`, `AdaMax`, `AMSGrad`, `Nadam` |
| Schedules / utils | `constant_lr`, `step_decay`, `exponential_decay`, `inverse_time_decay`, `cosine_annealing`, `polynomial_decay`, `warmup_cosine`, `clip_by_norm`, `clip_by_value`, `minimize_sgd` |
| Variance reduction | `minibatch_sgd`, `svrg`, `saga` |
| Gradient-free | `spsa`, `kiefer_wolfowitz`, `robbins_monro`, `finite_difference_gradient` |
| Global search | `cross_entropy_method`, `sep_cma_es`, `evolution_strategy` |
| Sampling | `monte_carlo_estimate`, `control_variate_estimate`, `importance_sampling_estimate`, `latin_hypercube`, `van_der_corput`, `halton_sequence`, `sobol_sequence`, `antithetic_normal`, `stratified_uniform`, `scenario_reduction` |
| Stochastic programming | `sample_average_approximation`, `two_stage_saa`, `newsvendor`, `stochastic_solution_metrics` |
| Risk | `value_at_risk`, `conditional_value_at_risk`, `entropic_risk`, `mean_variance_objective`, `minimize_cvar` |
| Bandits | `EpsilonGreedy`, `UCB1`, `ThompsonSampling`, `run_bandit` |

## References

- Robbins & Monro (1951) — *A Stochastic Approximation Method*.
- Kiefer & Wolfowitz (1952) — stochastic estimation of the maximum of a function.
- Spall (1992) — Simultaneous Perturbation Stochastic Approximation (SPSA).
- Kingma & Ba (2015) — *Adam*; Reddi et al. (2018) — *AMSGrad*.
- Johnson & Zhang (2013) — SVRG; Defazio, Bach & Lacoste-Julien (2014) — SAGA.
- Rubinstein & Kroese (2004) — *The Cross-Entropy Method*.
- Hansen (2016) — *The CMA Evolution Strategy: A Tutorial*; Ros & Hansen (2008) —
  separable CMA-ES.
- Rockafellar & Uryasev (2000) — *Optimization of Conditional Value-at-Risk*.
- Birge & Louveaux (2011) — *Introduction to Stochastic Programming* (SAA, EVPI,
  VSS, newsvendor).
- Dupačová, Gröwe-Kuska & Römisch (2003) — scenario reduction.
- Auer, Cesa-Bianchi & Fischer (2002) — UCB1; Thompson (1933) — Thompson sampling.
