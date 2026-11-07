# Signal Control — the stopline, from saturation flow to green waves

Green time is the scarcest currency a city has. Every junction is a tiny auction in
which cars, buses, cyclists and people on foot bid for the same seconds, and the
clearing price is delay. This module is the *static timing theory* of that auction,
in one dependency-free file: how much traffic a stopline can discharge, how much
delay a given split causes, which movements may legally run together, how long the
clearance periods must be, and — given all of that — which plan is best.

It is deliberately the theory half of the signals stack: `realtime_operations` does
receding-horizon control against live detectors, `traffic_flow` owns link dynamics
and `pedestrian_flow` owns crowd movement; this module owns the stopline. The
simplex behind Allsop's LP and MAXBAND, the Bron–Kerbosch enumerator behind stage
generation, the Robertson recursion behind the offsets and the queue simulations
behind max-pressure and Lämmer are all built here from scratch — no SciPy, no
NetworkX, nothing to `pip install`.

```python
from signal_control_template import (standard_four_arm_movements, conflict_matrix,
                                     design_fixed_time_plan, plan_delay)

mvs  = standard_four_arm_movements(turns=("through", "right"))
conf = conflict_matrix(mvs, drive_on_left=True, opposing_turns_conflict=False)

demand = [520., 170., 430., 140., 500., 160., 460., 130.]   # veh/h per movement
sat    = [1900., 1650.] * 4                                 # veh/h of green

plan = design_fixed_time_plan(demand, sat, conflict=conf,    # stages are INFERRED
                              intergreens=5.0, min_green=7.0)
print(plan.stages)                       # [[0, 4], [1, 5], [2, 6], [3, 7]]
print(round(plan.cycle, 1))              # 97.9 s  (Webster's optimum, clamped)
print([round(g, 1) for g in plan.effective_greens()])   # [31.8, 12.0, 28.2, 9.9]

delays, avg, los = plan_delay(plan, demand, sat)
print(round(avg, 1), los)                # 50.9 s/veh -> LOS D
print(plan.validate(conflict=conf))      # []  -- every rule holds
```

Give it demands, saturation flows and a conflict matrix and it generates the
stages, picks the cycle, splits the green, times the intergreens and hands back a
plan whose effective greens sum to exactly `C − L`.

---

## Pick

| You have… | …and want | Reach for |
| --- | --- | --- |
| A lane group's geometry | its **saturation flow** | `SaturationFlowFactors` → `saturation_flow_rate` |
| Critical flow ratios + lost time | the **optimum cycle** | `webster_optimal_cycle` (closed form) / `webster_optimal_cycle_numeric` (swept) |
| A movement's timing | **delay and LOS** | `hcm_control_delay` / `webster_delay` / `akcelik_delay` |
| A junction layout | a legal **staging** | `conflict_matrix` → `maximal_compatible_stages` → `minimum_stage_cover` |
| A cycle to divide | a **split** | `equisaturation_split` / `allsop_lp_split` / `smith_p0_split` / `delay_minimising_split` |
| Demands and a matrix | the whole **plan** | `design_fixed_time_plan` |
| A US-style controller | **dual-ring** structure | `nema_eight_phase` / `validate_ring_barrier` |
| An approach speed and width | **intergreen** timings | `intergreen_time` / `pedestrian_crossing_time` / `minimum_green_time` |
| A corridor of signals | **offsets** and **bandwidth** | `optimise_offsets_dp` / `optimise_offsets` / `arterial_bandwidth` / `maxband` |
| Detectors, or queues and turns | an **adaptive policy** | `simulate_actuated_phase` / `max_pressure_stage` / `laemmer_select_stage` |
| A give-way junction or roundabout | **entry capacity** | `hcm_potential_capacity` / `tanner_capacity` / `kimber_capacity` |
| A frontend to feed | a **JSON payload** | `build_signal_control_payload` / `build_a2ui_payload` |

---

## Install

```
# nothing to install -- Python >= 3.8 and the standard library
import signal_control_template as sc
```

`requirements.txt` has one uncommented line, `pytest>=7.0`. SciPy, NumPy and
NetworkX appear only as commented-out referees for the suite's differential tier;
the module itself never imports them.

---

## What's inside

### 1 — Saturation flow & capacity

The HCM base rate with the full adjustment chain — lane width, grade, parking, bus
blockage, area type, heavy vehicles, lane utilisation, turning radius and the
Appendix-D pedestrian/bicycle factor — collected in `SaturationFlowFactors` and
multiplied out by `saturation_flow_rate`. Then `movement_capacity` (`c = s·g/C`),
`flow_ratio` (`y = q/s`), `degree_of_saturation` (`X = v/c`) and the cycle-sizing
`critical_flow_ratios` / `practical_cycle_length`.

### 2 — Delay, queues & level of service

Webster's three-term delay and his optimum cycle `C0 = (1.5L + 5)/(1 − Y)`, plus a
sweep that minimises the *total* delay directly. The HCM 2010/2016 chain — `d1`
uniform, `d2` incremental, `d3` initial queue, progression factor `PF` — assembled
by `hcm_control_delay` into a `DelayResult`. Akçelik's overflow delay, the average
and 95th-percentile back of queue, and the signalised / unsignalised LOS bands.

### 3 — Conflicts, compatibility & staging

`conflict_matrix` decides which movement pairs clash by intersecting their paths
*geometrically* — with switches for merge conflicts, permitted turns, offside-to-
offside opposed turns and either hand of drive. `compatibility_graph` complements
it, `bron_kerbosch` enumerates every maximal clique (the candidate stages),
`minimum_stage_cover` picks the smallest family serving every movement, and
`validate_stages` / `is_conflict_free` police the result.

### 4 — Ring-barrier structure

`nema_eight_phase` builds the canonical dual-ring, two-barrier, eight-phase NEMA
structure; `validate_ring_barrier` checks that each ring fills the cycle and each
barrier aligns across rings; `ring_barrier_sequence` lays it out on the clock.

### 5 — Timing constraints

The ITE change-and-clearance formulae (`yellow_change_interval`,
`all_red_clearance`, `intergreen_time` with its clamp and rounding),
`pedestrian_crossing_time`, `minimum_green_time` and the lost time.

### 6 — The signal plan

`SignalPlan` holds stages, displayed greens, intergreens and lost times, exposing
`effective_greens`, `green_shares`, `movement_green`, `capacities` and a `validate`
that re-checks `Σg = C − L`, minimum greens and conflict-freedom.
`make_signal_plan` builds one from *effective* greens, `design_fixed_time_plan`
runs the whole pipeline and `plan_delay` scores it.

### 7 — Linear-programming kernel

A two-phase simplex with Bland's rule (`linear_program` → `LPResult`) handling
inequalities, equalities, negative right-hand sides, redundant rows, infeasibility
and unboundedness — the engine under Allsop, MAXBAND, MULTIBAND and max-pressure.

### 8 — Fixed-time split optimisation

`equisaturation_split` / `webster_split` (green proportional to the critical flow
ratios), `allsop_lp_split` (maximise the common capacity multiplier `μ`),
`smith_p0_split` and `delay_minimising_split` — all on top of `apportion_green`,
which respects minimum and maximum greens and closes the sum exactly.

### 9 — Platoon dispersion, offsets & bandwidth

`cyclic_flow_profile` builds an arrival profile, `robertson_dispersion` runs
Robertson's recursion to a cyclic steady state, `departure_profile` discharges it
at a stopline and `link_delay_table` scores every offset; `optimise_offsets_dp`
(exact DP) or `optimise_offsets` (TRANSYT hill climb) choose them.
`arterial_bandwidth` gives the exact through band by intersecting circular green
arcs, and `maxband` / `multiband` maximise the band itself.

### 10 — Adaptive policies

Actuated control (gap-out / max-out / volume density, plus the closed-form expected
green), Varaiya's max-pressure (movement and stage pressure, the policy, a
store-and-forward simulation, an LP stability margin) and the Lämmer–Helbing
self-organising index with its stabilising regime.

### 11 — Priority, people on foot, give-way junctions

Transit priority by green extension, early start or an inserted bus phase, all
preserving the cycle; pedestrian phasing (exclusive stages, concurrent checks,
leading pedestrian intervals, the Barnes dance, uniform delay); HCM two-way-stop
gap acceptance with rank impedance and four critical-gap estimators; roundabout
capacity by the HCM exponential, Brilon–Wu and the TRL/Kimber linear model.

### 12 — Payloads

`build_signal_control_payload` and `build_a2ui_payload`, both strict JSON.

---

## Recipes

**Time a junction from scratch.** Build the movements, get the conflict matrix and
let the designer infer everything else:

```python
plan = design_fixed_time_plan(demand, sat, conflict=conf, intergreens=5.0,
                              min_green=7.0, method="delay")   # or webster/allsop/smith_p0
```

**Audit a cycle you were handed.** Sweep the delay curve and see how far off it is:

```python
c_star, d_star = webster_optimal_cycle_numeric(demand, sat, plan.stages, L)
print(webster_total_delay(90.0, demand, sat, plan.stages, L) / d_star)  # 1.004 -> 0.4 % worse
```

**Progress a corridor**, exactly or by simulation:

```python
band, start = arterial_bandwidth(offsets, greens, travel_times, C, "outbound")
res   = maxband(greens, travel_out, travel_in, C, inbound_weight=1.0, m_range=2)
tuned = optimise_offsets(demands, sats, greens, travel_times, C, step=2.0)
```

**Prioritise a late bus without stretching the cycle**, then size a give-way entry
or run a policy instead of a plan:

```python
res = apply_transit_priority(plan, TSPRequest(stage=0, arrival=t, passengers=55))
assert abs(res.plan.cycle - plan.cycle) < 1e-9        # the cycle never moves

c_p = hcm_potential_capacity(600.0, critical_gap=6.5, follow_up=3.5)   # 459 veh/h
q_e = kimber_capacity(600.0, KimberGeometry())                         # 1187 pcu/h

rho, shares = max_pressure_stability_margin(movements, stages, sat, demand)
sim = simulate_max_pressure(movements, stages, sat, demand, horizon_s=3600.0)
print(rho, sim.is_bounded())                          # 1.607 True
```

---

## Design notes

**Effective green is the currency, displayed green is the output.** Everything
internal is expressed in effective greens, so `Σg = C − L` holds by construction;
`make_signal_plan` converts back (`G = g − I + l`) and refuses a negative displayed
green. `SignalPlan.validate()` re-checks the residual and *returns* the problems
rather than raising, so a caller can show them all at once.

**A stage is an independent set, and nothing is allowed to forget that.** The
conflict matrix is built geometrically — entry and exit points on the junction
circle, proper segment crossing — not from a lookup table, so it adapts to either
hand of drive, to permitted turns and to pedestrian pseudo-movements. Stage
generation is then exactly maximal-clique enumeration on the complement graph, and
`validate_stages` re-checks every pair afterwards.

**The delay families are kept honestly apart.** Webster's is a steady-state model:
it diverges as `v/c → 1` and returns `inf` at saturation, which is correct and
useful. The HCM and Akçelik forms are time-dependent over a finite analysis period
and stay finite at `X = 1`. Nothing silently substitutes one for the other;
`delay_minimising_split` takes the family by name.

**Ties are broken deterministically, everywhere.** Clique enumeration sorts its
pivots and output; the set cover orders candidates by `(−size, cost, members)`;
max-pressure keeps the lowest stage index on a tie. No number here depends on set
or dict iteration order, so results are byte-identical under any `PYTHONHASHSEED`
— and every stochastic routine takes a `seed` and uses a private `random.Random`.

**Guards are relative or exact, never absolutely scaled.** Capacity and pressure
comparisons are made against zero or a relative tolerance, so multiplying every
flow by 10⁻⁶ or 10⁶ leaves every dimensionless output — flow ratios, `v/c`, splits,
the chosen stage, the stability margin, `μ` — unchanged, as the suite asserts.

**Errors are one type.** Every invalid input raises `SignalControlError`, a
`ValueError` subclass, naming the offending quantity and its value — including the
interesting cases, like a junction whose critical flow ratios sum past one and
therefore cannot be signalised at any cycle.

---

## Testing

```bash
python signal_control_template.py        # the demo, ending in "demo OK"
pytest _stress_test.py -q                # 139 tests, ~3 s
PYTHONHASHSEED=1010 pytest _stress_test.py -q
```

**139 oracle-based tests, all green**, and green identically under
`PYTHONHASHSEED=0`, `1` and `1010`. No golden outputs: every expected value is a
closed form, an exact identity, a brute-force enumeration or a published example.

*Closed forms.* Webster's `C0 = (1.5L + 5)/(1 − Y)` is asserted exactly **and**
shown to sit in the flat minimum of a fine sweep of the total-delay curve (within
10 % of the swept optimum on every test junction, always better than half or double
the cycle). The HCM uniform delay `d1 = 0.5C(1 − g/C)²/(1 − min(1,X)·g/C)` is
asserted exactly and re-derived by integrating the deterministic vertical queue.
The ITE intervals, the adjustment chain, the Lämmer index and the Kimber
coefficients reproduce their formulae.

*Exact identities.* Effective greens sum to `C − L` on 200 random generated plans;
an equisaturation split equalises every degree of saturation at `Y·C/(C − L)`; a
symmetric two-phase junction splits the green *exactly* in half; Allsop's LP
reproduces that split and the multiplier `μ = (C − L)/(C·Y)`; Robertson's recursion
and the stopline discharge conserve vehicles; priority and the leading pedestrian
interval preserve the cycle; Tanner's capacity inverts back to its critical gap.

*Metamorphic and brute force.* Stages are checked to be independent sets of the
conflict matrix **exhaustively**, pair by pair; Bron–Kerbosch against enumeration
of all subsets for every n ≤ 12; a four-arm all-movements crossroads has exactly
**32 conflict points (16 crossing, 8 merging, 8 diverging)** and a T-junction
exactly **9 (3/3/3)**. Max-pressure picks the brute-force maximum-pressure stage
over 200 random queue states and keeps queues bounded strictly inside its stability
region — unbounded outside it. The exact offset DP matches brute force; the
arterial band matches a fine scan of the cycle; Siegloch, Tanner and Harders
converge to exactly `3600/t_f` as the conflicting flow vanishes; Kimber is exactly
`k·F` at `Q_c = 0`.

*Differential.* There is no pure-Python signal-control library to differentiate
against, so this tier is split: published worked examples — HCM 2010/2016, Webster
(1958), ITE change-interval practice, Kimber (TRL LR942), Tanner (1962) — are
hard-coded as ground truth, and where a *general-purpose* library can referee a
general-purpose kernel it does, behind `pytest.importorskip`: the simplex against
`scipy.optimize.linprog`, the Siegloch regression against `numpy.polyfit`, the
queue integral against `numpy.trapezoid` (numpy 2.x removed `np.trapz`) and the
clique enumeration against `networkx.find_cliques`.

**What the suite caught.** Two real defects, both fixed: `tsp_green_extension` and
`tsp_early_start` indexed the stage list before checking it, so an out-of-range
stage escaped as a raw `IndexError` instead of a `SignalControlError` like every
sibling entry point. It also pinned two behaviours that read like bugs but are not
— the LOS bands are closed on the right (10.0 s is still an A), and the
cardinality-only `minimum_stage_cover` can return a cover whose critical flow
ratios sum past one, so cost the candidates with `critical_flow_ratios` as
`design_fixed_time_plan` does.

---

## References

- Webster, F. V. (1958), *Traffic Signal Settings*, RRL Technical Paper 39; Webster
  & Cobbe (1966), *Traffic Signals*, RRL Technical Paper 56.
- *Highway Capacity Manual* (TRB, 2010 and 2016) — signalised intersections
  (saturation-flow adjustments, `d1`/`d2`/`d3`, LOS), two-way-stop control,
  roundabouts; ITE, *Determining Vehicle Change Intervals*.
- Allsop (1971), "Delay-minimising settings for fixed-time traffic signals"; Smith
  (1980), the P0 policy; Akçelik (1981), *Traffic Signals*, ARRB 123.
- Bron, C. & Kerbosch, J. (1973), "Algorithm 457: finding all cliques of an
  undirected graph", *CACM*; Robertson, D. I. (1969), *TRANSYT*, RRL Report 253;
  Little, Kelson & Gartner (1981), MAXBAND; Gartner et al. (1991), MULTIBAND.
- Varaiya, P. (2013), "Max pressure control of a network of signalized
  intersections"; Lämmer & Helbing (2008), "Self-control of traffic lights and
  vehicle flows in urban road networks".
- Kimber, R. M. (1980), *The Traffic Capacity of Roundabouts*, TRRL LR942; Tanner
  (1962); Siegloch (1973); Raff (1950); Troutbeck (1992); Brilon & Wu.

---

*Pure standard library — `itertools`, `math`, `random`, `dataclasses`, `typing` —
with no third-party imports anywhere. `signal_control_template.py`,
`__version__ = "1.0.0"`, 131 public names in `__all__`, 139 oracle-based tests.*
