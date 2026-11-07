# Traffic Assignment — where the traffic actually goes, in one file

Give it an origin-destination trip table and a road network and it tells you the
flow on every link. That is the fourth step of the classical four-step model and
the engine underneath every network-design, tolling, signal-timing and emissions
study in the urban domain — and it is the step where the interesting behaviour
lives, because drivers *choose*. Close a road and everyone reroutes. Open one
and, sometimes, everyone gets slower. This module is the deep version of that
question: six equilibrium solvers that must agree with one another, stochastic
route choice, the system optimum with its marginal-cost tolls, the price of
anarchy, an automatic Braess-paradox detector, multi-class PCU assignment,
elastic demand, turn penalties, select-link analysis and a finite-difference
sensitivity of the equilibrium to link capacity.

It *deepens* `core/optimisation/network_optimisation.traffic_assignment`, a
single function: one Frank-Wolfe loop over a BPR network with **one global
`alpha` and one global `beta` for every link in the country**. That is a real
capability gap, not a stylistic one — with one shared volume-delay parameter set
you cannot even write down Braess's network (whose links are `10x`, `50+x` and
`10+x`: three different functional forms), you cannot mix a motorway with a
residential street, and you cannot toll one link. Here every link carries its own
volume-delay function object with its own parameters and its own **analytic
integral**, which is what makes the Beckmann transformation computable and
therefore what makes every convex solver below possible. Pure standard library —
no NumPy, no SciPy, no pandas, no aequilibrae.

```python
from traffic_assignment_template import (Network, ODMatrix, BPR,
    user_equilibrium, price_of_anarchy, detect_braess, braess_network)

net = Network("corridor")
net.add_link(0, 1, BPR(t0=5.0, capacity=30.0, alpha=0.15, beta=4.0))
net.add_link(0, 2, BPR(t0=7.0, capacity=25.0))
net.add_link(1, 3, BPR(t0=6.0, capacity=20.0))
net.add_link(2, 3, BPR(t0=4.0, capacity=28.0))
net.add_link(1, 2, BPR(t0=1.5, capacity=15.0))          # the sneaky connector
od = ODMatrix({(0, 3): 35.0})

res = user_equilibrium(net, od, method="bfw", tol=1e-12)
print(res.summary())          # iters=8 gap=2.8e-16 Z=382.039818 TSTT=399.019654
print(price_of_anarchy(net, od).price_of_anarchy)        # 1.0141

wb, wb_od = braess_network(demand=6.0, include_bridge=True)
br = detect_braess(wb, wb_od)                            # the textbook paradox
print(br.paradox, br.best_link, br.improvement)          # True 2 54.0
```

Every solver returns the same `AssignmentResult` — link `flows`, `travel_times`,
`costs`, the Beckmann `objective`, `total_travel_time`, `relative_gap`,
`average_excess_cost` and a per-iteration `history` — so `method=` swaps without
touching the surrounding code, and everything stochastic takes an explicit
`seed`, so runs are byte-identical independent of `PYTHONHASHSEED`.

---

## Pick

| You have… | …and want | Reach for |
| --- | --- | --- |
| A network and a trip table (beginner) | the equilibrium flows | `user_equilibrium(net, od)` |
| A network and a trip table (beginner) | a fast, rough loading | `all_or_nothing` / `incremental_assignment` |
| A tight convergence budget (intermediate) | the *exact* equilibrium | `algorithm_b` / `gradient_projection` |
| Two answers that disagree (intermediate) | to know which is right | `relative_gap` / `wardrop_check` |
| A congested corridor (intermediate) | the socially best flows | `system_optimum` |
| A tolling study (intermediate) | first-best charges | `marginal_cost_tolls` + `tolled_user_equilibrium` |
| A policy argument (intermediate) | the cost of selfishness | `price_of_anarchy` |
| A new link on the drawing board (advanced) | to check it does not backfire | `detect_braess` / `capacity_sensitivity` |
| Imperfect information (advanced) | dispersed route choice | `logit_sue` / `dial_stoch` / `probit_sue` |
| Cars, HGVs and buses (advanced) | one loading in PCU | `multiclass_assignment` |
| Demand that responds to congestion (advanced) | an endogenous trip table | `elastic_demand_assignment` |
| A junction design (advanced) | per-turn volumes | `turn_penalty_assignment` |
| A screenline count (advanced) | who is using this link | `decompose_flows` + `select_link_analysis` |
| A dashboard (advanced) | strict JSON | `build_traffic_assignment_payload` / `build_a2ui_payload` |

---

## Install

Nothing to install. Copy `traffic_assignment_template.py` next to your code and
import it — CPython 3.8 or newer, standard library only. `requirements.txt`
lists a single uncommented line, `pytest`, for the test suite; every reference
package named in it is optional and used exclusively for cross-checking.

---

## What's inside

### 1 — Volume-delay functions

The foundation. Each is a dataclass of *per-link* parameters exposing
`travel_time(x)`, `derivative(x)` and — crucially — an **analytic** `integral(x)`,
because the Beckmann objective is the sum of those integrals. `BPR`
(`t0 (1 + alpha (x/c)^beta)`); `Davidson` / `ModifiedDavidson` (`t0 (1 + Jx/(c-x))`,
continued linearly from `mu*c` so a line search can probe an overload without
hitting the pole); `Conical` (Spiess: BPR-shaped but with a bounded derivative
everywhere, exactly `t0` at zero flow and `2 t0` at capacity); `Akcelik` (the
Australasian overflow-delay curve); and the algebraic `Polynomial` / `Linear` /
`Constant`. `MarginalCostVDF` wraps any curve as `t(x) + x t'(x)`: because
`∫₀^f (t + w t') dw = f t(f)`, a user equilibrium on the wrapped network **is**
the system optimum, which is how `system_optimum` is implemented.

### 2-5 — Network model, paths, objectives and link-based loading

`Link` (endpoints, VDF, toll, length, name), `Network` (ordered links plus
adjacency, with `copy` / `with_tolls` / `with_marginal_costs` / `without_link` /
`with_scaled_capacity` transformations) and `ODMatrix` (a sparse trip table whose
every accessor returns a **sorted** order, so no number depends on dict
iteration). On top: binary-heap Dijkstra trees (`shortest_path_tree`,
`shortest_path`), Yen's `k_shortest_paths`, a bounded depth-first
`enumerate_paths`, and `Path` / `PathSet` / `path_cost`. Then
`beckmann_objective` and its gradient (which *is* the link-cost vector),
`total_system_travel_time` (tolls deliberately excluded — a toll is a transfer,
not a delay), `relative_gap`, `average_excess_cost`, `max_flow_change`, the
`IterationRecord` / `AssignmentResult` containers, and the cheap loadings:
`all_or_nothing`, `incremental_assignment`, `method_of_successive_averages`.

### 6-8 — The equilibrium solvers

`frank_wolfe` with an `exact_line_search` (bisection on the monotone directional
derivative, or golden section), `conjugate_frank_wolfe` (the new direction is
H-conjugate to the last one) and `biconjugate_frank_wolfe`
(Mitradjieva-Lindberg, mixing the two previous auxiliary points);
`gradient_projection` (Jayakrishnan: column generation plus path equilibration
with a quasi-Newton equalising step and a Beckmann safeguard); and `algorithm_b`
(Dial's bush method: one acyclic subnetwork per origin, equilibrated internally
by exact bisection between the min- and max-cost used segments).
`user_equilibrium` is the umbrella over all six.

### 9-11 — Stochastic, system optimum, Braess

`dial_stoch` (Dial's STOCH: a logit split over *efficient* paths with no
enumeration at all), `logit_sue` (MSA on Fisk's objective, with C-logit and
path-size overlap corrections), `probit_sue` (seeded Monte Carlo with common
random numbers), plus `logit_probabilities`, `c_logit_factors`,
`path_size_factors` and `fisk_objective`. Then `system_optimum`,
`marginal_cost_tolls`, `tolled_user_equilibrium` and `price_of_anarchy`; and
`detect_braess`, which searches for a link whose *removal* lowers total system
travel time, with `braess_network` building the textbook instance.

### 12-19 — Extensions, analysis, builders and payloads

`multiclass_assignment` (PCU equivalents and per-class values of time);
`elastic_demand_assignment` (an excess-demand reduction *and* Evans' partial
linearisation, each a check on the other, with `LinearDemand` /
`ExponentialDemand` / `PowerDemand`); `expand_with_turn_penalties` /
`turn_penalty_assignment` on the entry/exit expanded graph; `decompose_flows`
(link flows back into routes), `wardrop_check`, `select_link_analysis` and
`select_zone_analysis`; `capacity_sensitivity`, central differences on a
re-solved equilibrium — exactly the gradient a bi-level network-design search
needs; `two_link_network` with its analytic `two_link_ue_closed_form` /
`two_link_so_closed_form`, plus `grid_network` and `random_affine_network`; and
two strict-JSON payload builders with a demo that exercises every export.

---

## Recipes

**Compare solvers on one network** — they minimise the same convex objective, so
disagreement is a bug — then **price the corridor so that selfish routing becomes
socially optimal:**

```python
runs = {m: user_equilibrium(net, od, method=m, tol=1e-12, max_iter=3000)
        for m in ("fw", "cfw", "bfw", "gp", "b")}
ref = runs["bfw"].flows
for name, r in sorted(runs.items()):
    print(name, r.iterations, r.relative_gap, max_flow_change(r.flows, ref))

so = system_optimum(net, od, tol=1e-12)
tolls = marginal_cost_tolls(net, so.flows)
tue = tolled_user_equilibrium(net, od, tolls, tol=1e-12)
assert max_flow_change(tue.flows, so.flows) < 1e-6         # exactly the SO
print(tue.extra["toll_revenue"], tue.extra["untolled_total_travel_time"])
```

**Ask whether a proposed link will backfire** before anybody pours concrete:

```python
braess = detect_braess(net, od)
if braess.paradox:
    print("closing link", braess.best_link, "saves", braess.improvement, "veh-h")
sens = capacity_sensitivity(net, od, delta=0.02)
print("widen link", sens.best_link())     # dTSTT/dcap > 0 means widening HURTS
```

**Mix vehicle classes, trace a screenline, let demand respond to delay:**

```python
classes = [VehicleClass("car", {(0, 8): 40.0}, pcu=1.0, value_of_time=12.0),
           VehicleClass("hgv", {(0, 8): 8.0},  pcu=2.5, value_of_time=30.0)]
print(multiclass_assignment(net, classes, tol=1e-7).summary())

ue = user_equilibrium(net, od, method="bfw", keep_origin_flows=True, tol=1e-12)
routes = decompose_flows(net, od, ue.flows, origin_flows=ue.origin_flows)
assert wardrop_check(net, routes, tol=1e-6).satisfied
print(select_link_analysis(net, routes, busiest_link).top_pairs(5))

res = elastic_demand_assignment(net, {(0, 8): ExponentialDemand(120.0, 0.02)},
                                method="excess", solver="gp", tol=1e-8)
print(res.total_demand, res.od_costs)      # d_od == D_od(u_od) at the optimum
```

---

## Design notes

- **Per-link volume-delay functions are the whole point.** The parent routine in
  `network_optimisation` takes one global `(alpha, beta)`; here the parameters
  live on the link, each curve carries its own *analytic* integral, and that
  integral is what makes the Beckmann transformation — and therefore every
  convex solver, the system optimum and the price of anarchy — computable.
  `AssignmentResult` is identical whichever `method=` you pass, and all six
  solvers minimise that same strictly convex objective; their agreement is the
  module's best correctness oracle.
- **They are not equally good, and the docs say so.** Plain Frank-Wolfe has a
  famously slow tail: it will report a gap of 1e-5 while its *flows* are still
  visibly wrong. `algorithm_b` and `gradient_projection` do not. Never read a
  small gap as a converged flow vector — check both.
- **Tolls are generalised cost, not delay.** `Link.cost` is travel time plus toll
  (what drivers route on), while `total_system_travel_time` deliberately excludes
  tolls: a toll is a transfer between road users and the exchequer, not a delay
  anybody experiences.
- **The system optimum is a user equilibrium in disguise.** `MarginalCostVDF`
  wraps `t` as `t + x t'`, whose integral is exactly `x t(x)`, so the Beckmann
  objective of the wrapped network *is* total system travel time. Every solver
  computes the SO unchanged — no second algorithm, no second set of bugs.
- **Every tie is broken deterministically.** OD pairs, origins, route buckets,
  bush links and Braess candidates are iterated in sorted order and ranked with
  explicit `(value, index)` keys, so no output number depends on `set` or `dict`
  iteration order; the suite builds one trip table in three insertion orders and
  demands bit-identical flows.
- **Guards are relative or exact-zero, never absolute.** The relative gap is
  clamped at zero (cancellation genuinely drives the raw `loaded - sptt`
  difference negative on ~6% of exactly-equilibrated two-link networks);
  `wardrop_check` scales its tolerance against the cheapest route, so it means
  the same thing on a five-minute network and a five-hour one. Multiply the time
  axis or the vehicle axis by 1e-6 or 1e6 and the flows, the v/c ratios and the
  price of anarchy do not move.
- **Reproducible, and validated.** `probit_sue` and `random_affine_network` take
  a `seed` and flow through a private `random.Random`; probit reuses the *same*
  seeded draw matrix every outer iteration (common random numbers), so the
  loading is a deterministic function of the link costs and the fixed point is
  well defined. A negative toll, a non-monotone polynomial, a `Conical` alpha
  below 1, a self-loop, an intrazonal trip, a cost vector of the wrong length, a
  negative link cost or an unreachable destination all raise a clear
  `TrafficAssignmentError` rather than silently returning a wrong number.

---

## Testing

```bash
python traffic_assignment_template.py             # the demo, ends "demo OK"
pytest _stress_test.py -q                         # 140 tests, ~6 s
for h in 0 1 1010; do PYTHONHASHSEED=$h pytest _stress_test.py -q; done
```

**140 oracle-based tests** in four tiers, green under every hash seed — no
hand-typed magic number where a closed form, an exact identity or an invariant
exists, and no golden outputs anywhere.

- **Closed forms.** **Braess** is the flagship: demand 6 costs exactly 498.0
  veh-h (83.0 per traveller) without the bridge and exactly 552.0 (92.0 per
  traveller) with it, at link flows (4,2,2,2,4), and `detect_braess` must *find*
  the bridge. **Pigou** (latencies `x` and `1`, unit demand) has UE cost exactly
  1, SO cost exactly 0.75 and a price of anarchy of exactly 4/3, to 1e-9. The
  two-link UE split `(a2-a1+b2 D)/(b1+b2)` and SO split
  `(a2-a1+2 b2 D)/(2(b1+b2))` are re-derived in the test. Every VDF's analytic
  integral is checked against composite-Simpson quadrature and its derivative
  against a central difference; `exact_line_search` on affine links reproduces
  `lam* = -d·(t0 + Bx + toll) / d·Bd`; Dial's STOCH split on parallel constant
  links is the multinomial logit exactly.
- **Exact identities.** `MarginalCostVDF.integral(x) == x·t(x)` to the bit;
  `beckmann_gradient == network.costs`; a route decomposition re-aggregates to
  the link flows it came from; marginal-cost tolls turn the UE into the SO; a
  zero-penalty turn expansion reproduces the plain equilibrium; Yen's *k*
  cheapest routes equal the *k* cheapest of a brute-force enumeration (and the
  3×3 grid's classic 12 corner-to-corner simple paths are all found).
- **Metamorphic / invariant.** The cross-solver oracle (five solvers, one set of
  link flows, to 1e-5); Frank-Wolfe's objective trace is monotone non-increasing
  while MSA's demonstrably is not; the price of anarchy of an affine network
  never exceeds 4/3 across 24 seeded random networks (Roughgarden-Tardos);
  constant costs collapse UE onto SO with PoA exactly 1; logit SUE approaches the
  UE monotonically as `theta` grows; an incremental loading can never beat the
  Beckmann minimum; **scale invariance** under a 1e-6 and 1e6 rescaling of both
  the time axis and the vehicle axis; **determinism** across repeated runs, three
  insertion orders of one trip table and three `PYTHONHASHSEED`s.
- **Differential**, behind `pytest.importorskip`. `numpy.trapezoid` (NumPy 2.x
  removed `np.trapz`) on a 200 001-point grid re-derives every analytic VDF
  integral and `numpy.gradient` its derivative; `scipy.optimize.minimize`
  (SLSQP) solves the Beckmann and total-travel-time programs on an explicit route
  parametrisation and lands on the same flows (objectives to 1e-9; SLSQP's own
  flow precision binds at ~1e-5). A fifth, in-repo differential runs against
  `network_optimisation.traffic_assignment` and agrees to 9e-9 on link flows —
  but **only on homogeneous networks**, because that function takes a single
  global `alpha`/`beta`, so per-link VDFs, tolls and Braess are not
  representable in it at all.

**What the suite caught.** The template needed **no fixes** — every oracle above
passes against it unmodified. The suite did catch three real traps in *itself*
while it was being written, all now pinned:

1. **A silently degenerate price-of-anarchy fixture.** The first cross-solver
   network had free-flow times satisfying `t0[2] == t0[3] + t0[4]`. Because a BPR
   link obeys `mc - t = beta (t - t0)`, that coincidence makes the user
   equilibrium *exactly* system-optimal: PoA came out at 1.0 to sixteen digits
   and every tolling and anarchy assertion would have been vacuously true. The
   builder now documents and avoids it.
2. **The relative-gap clamp is load-bearing.** Across 400 seeded two-link
   networks evaluated at their closed-form equilibrium, the raw `loaded - sptt`
   difference is *negative* 24 times from pure float cancellation. The suite
   asserts both that the clamp fires *and* that the cancellation it exists for
   actually happens — without the second half the test would still pass on a
   build where the clamp had been deleted.
3. **Oracle precision limits, correctly attributed.** Golden section is held to
   1e-6 and bisection to 1e-12 on the *same* line search, because a flat
   quadratic minimum caps golden section at √ε; Davidson near its pole limits
   *Simpson* and *`numpy.gradient`*, not the analytic integral under test, so
   those comparisons are sampled below the knee; and Frank-Wolfe is held to 1e-3
   while `algorithm_b` and `gradient_projection` are held to 1e-10, because one
   shared bar would have meant weakening it for the solvers that deserve the
   hard one.

---

## References

- Beckmann, McGuire & Winsten (1956) — *Studies in the Economics of
  Transportation*, the transformation that makes user equilibrium convex;
  Wardrop (1952) — the two principles.
- Frank & Wolfe (1956); Mitradjieva & Lindberg (2013) — conjugate-direction
  Frank-Wolfe; Jayakrishnan et al. (1994) — path-based gradient projection;
  Dial (2006) — Algorithm B; Dial (1971) — the STOCH multipath model.
- Fisk (1980) — the logit-SUE objective; Cascetta et al. (1996) — C-logit;
  Ben-Akiva & Bierlaire (1999) — path-size logit; Evans (1976) — combined
  distribution and assignment (elastic demand); Yen (1971) — *k* shortest paths.
- Braess (1968) — *Über ein Paradoxon aus der Verkehrsplanung* (Braess,
  Nagurney & Wakolbinger, 2005, for the translation); Pigou (1920) — *The
  Economics of Welfare*; Roughgarden & Tardos (2002) — *How bad is selfish
  routing?* (the 4/3 bound).
- Spiess (1990) — conical volume-delay functions; Davidson (1966); Akçelik
  (1991). Sheffi (1985) — *Urban Transportation Networks*; Patriksson (1994) —
  *The Traffic Assignment Problem*.

---

*Pure standard library: `heapq`, `math`, `random`, `dataclasses`, `typing`.
`traffic_assignment_template.py` — `__version__ = "1.0.0"`, 83 public names in
`__all__`, 140 oracle-based tests.*
