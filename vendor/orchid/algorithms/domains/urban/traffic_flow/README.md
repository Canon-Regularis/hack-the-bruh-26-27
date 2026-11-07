# Traffic Flow

`traffic_assignment` answers *which route*; `microsimulation` answers *what this
particular driver did*. This module answers the question in between, the one a
corridor study actually asks: **once the vehicles are on the road, what
happens?** Where the shock forms and how fast it travels backwards, how far the
queue reaches before the signal turns green, what a bottleneck really discharges
once it has congested, and whether a brake tap in the twelfth car of a platoon
dies out or grows into a jam nobody caused.

Both halves of the theory live here and are pinned to each other. Macroscopic:
eight fundamental diagrams behind one surface, LWR by Godunov in the
sending/receiving form, Daganzo's Cell Transmission Model as a network with
priority merges, FIFO diverges, spillback and capacity drop, Yperman's Link
Transmission Model on cumulative curves, Newell's three-detector construction,
the Webster/Akçelik delay family, the Macroscopic Fundamental Diagram, and
METANET / Payne-Whitham / Aw-Rascle-Zhang. Microscopic: Newell, Gipps, IDM,
Optimal Velocity, Full Velocity Difference and Krauss — each with a closed-form
equilibrium spacing — plus MOBIL, Nagel-Schreckenberg and Wilson-Ward string
stability. Pure standard library: the Riemann solver, the finite-volume schemes
and the cellular automaton are all built by hand.

```python
from traffic_flow_template import (TriangularDiagram, riemann_solve,
                                   cfl_timestep, godunov_simulate, signal_delay)

fd = TriangularDiagram(v_free=100.0, k_jam=150.0, w=20.0)   # Daganzo's triangle
print(fd.capacity, fd.k_critical)      # 2500.0 veh/h/lane at 25.0 veh/km/lane

rs = riemann_solve(fd, 20.0, 100.0)    # free flow runs into a standing queue
print(rs.kind, rs.shock_speed)         # 'shock' -12.5 km/h — the back of the queue

dx = 0.05                              # 50 m cells
dt = cfl_timestep(fd, dx, 0.9)         # largest stable step at Courant 0.9
res = godunov_simulate(fd, [20.0] * 100 + [100.0] * 100, dx, dt,
                       n_steps=int(round(0.1 / dt)),
                       inflow=fd.flow(20.0), downstream_supply=fd.flow(100.0))
print(res.front_position(60.0)[-1])    # 3.775 km  (analytic 5.0 − 12.5 × 0.1 = 3.75)
print(res.conservation_error)          # 7.1e-14 veh — vehicles are not invented

d = signal_delay(flow=750.0, saturation_flow=1800.0, cycle_s=90.0, green_s=36.0)
print(d.degree_of_saturation, d.total_s, d.level_of_service)   # 1.04  71.6 s/veh  'E'
```

Everything stochastic (Krauss dawdling, Nagel-Schreckenberg) takes an explicit
`seed` and drives a private `random.Random`, so a run is byte-identical and
independent of `PYTHONHASHSEED`.

---

## Pick

| Goal | Use |
| --- | --- |
| A q(k) curve fitted to a real road | `Greenshields`, `Greenberg`, `Underwood`, `Drake`, `NewellDiagram`, `DelCastillo`, `fd_from_name` |
| A kink-free triangle, or a capacity plateau | `TriangularDiagram` / `TrapezoidalDiagram` |
| Where the front goes when two states meet | `riemann_solve`, `shock_speed`, `rarefaction_density` |
| Density over space and time on one link | `godunov_simulate`, sized by `cfl_timestep` |
| A corridor with merges, diverges, spillback | `ctm_corridor` / `CTMNetwork` |
| The same corridor from its boundary curves | `ltm_simulate` |
| Delay, queue, travel time from counts alone | `delay_between_curves`, `queue_from_curves`, `travel_time_from_curves` |
| `N(x, t)` between two detectors | `newell_three_detector` |
| A queue with (or without) physical length | `horizontal_queue` / `vertical_queue` |
| Control delay and an HCM level of service | `signal_delay`, `webster_delay`, `akcelik_delay`, `webster_optimal_cycle` |
| How far back one red pushes the queue | `shockwave_queue` |
| A linear, MPC-ready network model | `StoreAndForwardModel` |
| A network-wide MFD, and gridlock risk | `edie_flow_density`, `fit_mfd`, `mfd_accumulation` |
| Capacity drop, hysteresis, stop-and-go waves | `metanet_simulate`, `payne_whitham_simulate`, `arz_simulate` |
| One car following another | `IDM`, `GippsCF`, `NewellCF`, `KraussCF`, `OptimalVelocity`, `FullVelocityDifference` |
| Whether a brake tap grows or dies | `string_stability` / `platoon_amplification` |
| A lane-change decision with politeness | `mobil_decision` |
| An emergent FD from the simplest rules | `nagel_schreckenberg` / `nasch_fundamental_diagram` |
| JSON a browser can render | `build_traffic_flow_payload` / `build_a2ui_payload` |

---

## Install

Nothing to install. Copy `traffic_flow_template.py` next to your code and import
it: **zero runtime dependencies**, standard library only, CPython 3.8+. The one
uncommented line in `requirements.txt` is `pytest>=7.0`, for the suite.

```bash
python traffic_flow_template.py     # the demo, prints "demo OK"
pytest _stress_test.py -q           # 237 oracle-based tests
```

---

## What's inside

### 0 — Units and validation

Macroscopic units are **(km, h)**, microscopic **SI (m, s)**. `seconds` turns a
signal timing into hours, `hours_to_seconds` returns, and `kmh_to_ms` /
`ms_to_kmh` / `KMH_PER_MS` bridge them. Every entry point range-checks its
arguments and raises `TrafficFlowError`, a `ValueError` subclass.

### 1 — Fundamental diagrams

Eight `q(k)` laws behind one `FundamentalDiagram` surface — `Greenshields`,
`Greenberg`, `Underwood`, `Drake`, `NewellDiagram`, `DelCastillo`,
`TriangularDiagram`, `TrapezoidalDiagram` — each with analytic `capacity`,
`k_critical`, `v_critical`, `wave_speed`, `max_wave_speed`, `density_from_flow`
and the `demand` / `supply` split every solver is built on. Plus `fd_from_name`,
`fd_table` and `fd_capacity_numeric`, the suite's brute-force scan oracle.

### 2 — Kinematic wave theory (LWR)

`characteristic_speed` is `dq/dk`, `shock_speed` the Rankine-Hugoniot quotient
`(q₂ − q₁)/(k₂ − k₁)`, `rarefaction_density` the density at `ξ = x/t` inside a
fan. `riemann_solve` returns the whole similarity solution as a `RiemannSolution`
— `'shock'`, `'rarefaction'` or `'uniform'` — with `density_at` and
`front_position`.

### 3 — Godunov scheme

`godunov_flux` is `min(demand(up), supply(down))` and that is the whole scheme;
`cfl_number` / `cfl_timestep` / `max_wave_speed` size the step, and
`godunov_simulate` returns a `GodunovResult` with the density history, cumulative
counts, `conservation_error` and `front_position(threshold)`.

### 4 — Cell Transmission Model

`merge_flows` is Daganzo's priority merge (slack from a satisfied branch
reallocated), `diverge_flows` the FIFO diverge. `CTMCell` holds vehicles and
applies `capacity_drop`, `CTMNode` classifies itself as source / sink / link /
merge / diverge, and `CTMNetwork` wires them (`add_cell`, `connect`, `add_merge`,
`add_diverge`, `add_source`, `add_sink`, `validate`, `max_timestep`, `step`,
`simulate`). `ctm_corridor` builds the common case in a line; `CTMResult` carries
densities, speeds, node flows, travel time/distance/delay and `density_of`.

### 5 — Link Transmission Model

`LTMLink` is a link described by its two Newell wave speeds and its storage
(`from_fd` derives them from any diagram); `ltm_simulate` runs Yperman's LTM from
*lagged* cumulative curves rather than cell states. `LTMResult` returns each
link's two curves plus `link_travel_time` — no discretisation, so free-flow time
is exact.

### 6 — Cumulative curves

`CumulativeCurve` is `N(t)` with `at`, `inverse`, `area`, `flows` and Newell's
`shifted`; `cumulative_from_flows` integrates a flow profile into one. Then
`delay_between_curves` (area, veh-h), `queue_from_curves` (vertical gap) and
`travel_time_from_curves` (horizontal gap). `newell_three_detector` builds
`N(x, t)` inside a link as the min of the forward and backward waves.

### 7 — Queues

`vertical_queue` stacks vehicles at a point; `horizontal_queue` gives the queue
real road space, so it can fill the link, reject arrivals and report `spillback`
/ `spillback_time`. Both return a `QueueResult` with the profiles, the queue in
vehicles and km, total delay and a `conservation_error`.

### 8 — Signal delay

`deterministic_delay` (Webster's uniform term), `webster_delay` (the full 1958
formula), `akcelik_delay` (time-dependent overflow) and `webster_optimal_cycle`.
`signal_delay` combines them into a `DelayBreakdown` — uniform, random and
overflow components, degree of saturation, capacity, an HCM `level_of_service` —
that stays finite as `x → 1`.

### 9 — Shockwave analysis

`shockwave_queue` is the classical three-wave construction at a stop line:
stopping, starting and recovery waves from Rankine-Hugoniot, their intersection
giving the maximum queue in km and vehicles, when it occurs, the clearance time
and whether it clears within the green. Returns a `ShockwaveResult`.

### 10 — Store-and-forward

`SFLink` and `StoreAndForwardModel`: the linear link-queue model MPC signal
control is built on. Outflow is green-proportional (`outflow_capacity`), turn
ratios move vehicles between links, storage rejects the overflow.
`StoreAndForwardResult` reports `clampedSteps` and an `isLinear` flag, so you can
see whether the run stayed inside the regime where the model really is linear.

### 11 — Macroscopic Fundamental Diagram

`edie_flow_density` is Edie's generalised definition over a space-time rectangle;
`aggregate_network_state` does the lane-km weighted aggregation; `fit_mfd` fits a
through-origin polynomial into an `MFDModel` (`production`, `speed`, `outflow`,
`capacity`, `critical_density`, `critical_accumulation`); and `mfd_accumulation`
runs the reservoir dynamics into an `MFDAccumulationResult` flagging gridlock.

### 12 — Second-order models

`METANETParams` + `metanet_simulate` (Papageorgiou, with ramp and anisotropy
terms), `payne_whitham_simulate` (anticipation / pressure) and `arz_simulate`
(Aw-Rascle-Zhang on a Rusanov flux), each returning a `SecondOrderResult` with
density, speed and flow fields, and each flux-limited so mass balances exactly.

### 13 — Car-following

`NewellCF` (with the exact `next_position` rule and an implied `wave_speed_kmh`),
`GippsCF`, `IDM`, `OptimalVelocity`, `FullVelocityDifference` and `KraussCF` (the
SUMO default, seeded dawdling). Each exposes a closed-form `equilibrium_spacing`,
its inverse `equilibrium_speed`, an `equilibrium_flow` and a
`fundamental_diagram()` lifting the law back to the macroscopic world;
`CarFollowingModel` takes two callables for your own. `simulate_platoon` returns
a `PlatoonResult` with trajectories, gaps, accelerations and collisions.

### 14 — Lane changing

`mobil_decision` evaluates Kesting's MOBIL rule from six `CFState` triples
(subject, new follower, old follower — before and after). `MobilParams` carries
politeness, threshold, safety deceleration and bias; `MobilDecision` splits the
incentive into own gain, new-follower loss and old-follower gain, with the safety
verdict kept separate from the change decision.

### 15 — Cellular automata

`nagel_schreckenberg` is the seeded four-rule CA on a ring road — accelerate,
brake, dawdle, move — returning a `NaSchResult` with the raw cell/step numbers
plus `flow_veh_h` / `density_veh_km` / `speed_kmh`. `nasch_fundamental_diagram`
sweeps density to trace the emergent FD.

### 16 — Linear stability

`string_stability` differentiates any car-following law about its own equilibrium
and applies the Wilson-Ward criterion — stable iff `f_v²/2 − f_dv·f_v − f_s > 0`
— returning a `StabilityResult` with the sensitivities, the local verdict and a
signed `string_margin`. `platoon_amplification` measures the same thing
empirically.

### 17 — Payload builders

`build_traffic_flow_payload` packs a diagram, any simulation result, a delay
breakdown, a shockwave analysis and an MFD into one JSON-able dict (frames
strided to a budget); `build_a2ui_payload` turns that into a typed component list
— stat cards, an FD line chart, a density heatmap, cumulative curves, a delay bar.

---

## Recipes

**Size the queue behind a lane drop.** The bottleneck discharges its own
capacity, not the demand, and the queue stands upstream of it:

```python
from traffic_flow_template import TriangularDiagram, ctm_corridor, seconds

fd = TriangularDiagram(v_free=100.0, k_jam=150.0, w=20.0)
dt = seconds(1.0)                                     # 1 s steps
net = ctm_corridor(fd, length_km=4.0, n_cells=40, dt=dt, demand=6000.0,
                   lanes=3.0, bottleneck_cells=range(30, 40),
                   bottleneck_lanes=2.0)              # three lanes drop to two
res = net.simulate(dt, n_steps=1800)                  # half an hour

rate = (res.outflow_cum[-1] - res.outflow_cum[-2]) / (res.times[-1] - res.times[-2])
print(round(rate), round(fd.capacity * 2.0))          # 5000 5000 — exactly the drop
print(round(res.total_delay, 1), res.conservation_error)   # 113.1 veh-h, 3e-12 veh
print(sum(0.1 for c in res.cell_ids
          if res.density_of(c)[-1] > fd.k_critical))  # 3.0 km of standing queue
```

**Get delay, queue and travel time out of counts alone** — every count-based
answer is a distance between the two cumulative curves:

```python
from traffic_flow_template import (LTMLink, ltm_simulate, delay_between_curves,
                                   queue_from_curves, travel_time_from_curves,
                                   newell_three_detector, seconds)

link = LTMLink.from_fd("L1", length_km=1.0, fd=fd, lanes=2.0)
step = seconds(1.0)
pulse = lambda t: 4000.0 if t < 0.1 else 500.0

ltm = ltm_simulate([link], step, n_steps=int(round(0.4 / step)),
                   inflow=pulse, exit_supply=2400.0)
up, down = ltm.curve("L1", "upstream"), ltm.curve("L1", "downstream")

print(delay_between_curves(up, down))                  # 19.83 veh-h (the area)
print(queue_from_curves(up, down, 0.1))                # 180.0 veh (vertical gap)
print(travel_time_from_curves(up, down, 0.1) * 3600.0) # 270.0 s (horizontal gap)

mid = newell_three_detector(up, down, x=0.5, length_km=1.0, v_free=link.v_free,
                            w=link.w, k_jam=link.k_jam, times=up.times, lanes=2.0)
print(mid.at(0.12), up.at(0.12), down.at(0.12))        # 354.0, between 406 and 264
```

**Time a signal, then check the queue fits.** Webster gives the cycle, the
breakdown grades it, the shockwave construction says how far back the red pushes:

```python
from traffic_flow_template import (webster_optimal_cycle, signal_delay,
                                   shockwave_queue)

cycle = webster_optimal_cycle(lost_time_s=12.0, critical_flow_ratios=[0.30, 0.35])
green = (cycle - 12.0) * 0.30 / 0.65                   # proportional split
d = signal_delay(650.0, 1800.0, cycle, green, period_h=0.25)
print(round(cycle, 1), round(green, 1))                # 65.7 s cycle, 24.8 s green
print(round(d.total_s, 1), d.level_of_service)         # 42.7 s/veh, LOS 'D'

sw = shockwave_queue(fd, arrival_flow=1200.0, red_s=cycle - green, green_s=green)
print(round(sw.stopping_wave_speed, 2))                # −8.70 km/h, going backwards
print(round(sw.max_queue_km * 1000.0), sw.queue_clears)   # 175 m, False — it spills
```

**Ask whether a brake tap becomes a phantom jam** — the analytic verdict about
the equilibrium, and the measured gain, agree:

```python
import math
from traffic_flow_template import (OptimalVelocity, string_stability,
                                   simulate_platoon, platoon_amplification)

def brake_tap(t):                        # leader dips 2 m/s around t = 10 s
    return 20.0 - 2.0 * math.exp(-((t - 10.0) / 1.5) ** 2)

for label, m in (("sluggish", OptimalVelocity(33.0, kappa=0.85, s_c=25.0, beta=0.13)),
                 ("responsive", OptimalVelocity(33.0, kappa=5.0, s_c=25.0, beta=0.13))):
    st = string_stability(m, v_equilibrium=20.0)
    pl = simulate_platoon(m, n_vehicles=12, leader_speed=brake_tap,
                          dt=0.1, n_steps=900, seed=3)
    print(label, round(st.string_margin, 3), st.string_stable,
          round(platoon_amplification(pl), 2), round(pl.min_gap, 2))
# sluggish   -1.382 False 14.76  3.37   <- a 2 m/s tap becomes a 30 m/s swing
# responsive  2.248 True   0.84 25.72   <- the same tap is damped away
```

---

## Design notes

- **Two unit systems, named at every boundary.** Macro is (km, h, veh/km/lane,
  veh/h/lane); car-following, MOBIL and the platoon simulator are SI (m, s,
  m/s²), and every micro banner repeats it. There `gap` is the **clear** distance
  and `spacing = gap + length` — never one word for both.

- **Per lane on the diagram, total on the cell.** `fd.capacity` is per lane; a
  `CTMCell` multiplies by its own `lanes`, which is what makes a lane drop a
  one-argument change (`bottleneck_lanes=2.0`) rather than a second diagram.

- **Demand/supply is the entire engine.** Every model downstream is a different
  way of sharing `min(demand(up), supply(down))`: the CTM's merge is a priority
  split of one cell's supply, its diverge a FIFO throttle by the tightest branch,
  the LTM the same rule on lagged cumulative curves.

- **Underwood and Drake are not concave, and the module says so.** Both turn
  convex above an inflection density (`k = 2·k_c`, `k = √3·k_c`), so their
  characteristic speed is *not* monotone; neither reaches zero speed, so neither
  has a true jam and `density_from_flow` saturates at `k_jam` for small flows.
  Pinned by tests — use a triangle or trapezoid if you need strict concavity.

- **The trapezoid's plateau is exactly `q_max`.** `TrapezoidalDiagram.flow` is
  computed in min-form, not as `k · speed(k)`; see **Testing**.

- **CFL is enforced, not suggested.** `godunov_simulate` refuses an unstable step
  and names the legal one, `CTMNetwork.simulate` checks `max_timestep()`, and
  `ltm_simulate` needs `dt` below both the free-flow *and* the backward traversal
  time of every link. Use `cfl_timestep(fd, dx, 0.9)` and stop worrying.

- **Conservation is measured, never assumed.** Every conservative solver returns
  a `conservation_error`, `~1e-12` or better throughout; the flux limiters exist
  so the physical `[0, k_jam]` clip can never silently destroy a vehicle.

- **Capacity drop only bites where a cell is congested — mind where you put it.**
  It fires when `k > k_critical`, but a lane-drop bottleneck holds *its own* cells
  near capacity and queues upstream, so the parameter can sit inert there. Put it
  on cells that congest (`bottleneck_lanes == lanes` selects without dropping).

- **Webster diverges at saturation and raises rather than returning nonsense.**
  `webster_delay` refuses `x ≥ 1`; reach for `signal_delay`, which drops the
  stochastic term above `x = 0.95` and adds Akçelik's overflow term instead. CTM
  topology is likewise checked at construction — one inbound and one outbound
  node per cell, so an `n → m` junction must be a diverge followed by a merge.

- **Two micro conventions worth knowing.** `IDM.desired_gap` uses the canonical
  Treiber clamp — the *whole* bracket floored at `s0`, so a strongly receding
  leader gives `s* = s0`, not the variant clamping only the closing term. And
  Krauss dawdles only when handed an rng: it slows a vehicle *within a step*, but
  a follower that dawdles opens a gap and can finish the run faster.

---

## Testing

`_stress_test.py` is an oracle-based suite: **237 tests**, green under
`PYTHONHASHSEED` 0, 1 and 1010, with no hand-typed golden number anywhere a
closed form, an identity or an invariant exists. Four tiers:

1. **Closed forms.** Greenshields' capacity is exactly `v_f·k_j/4` at `k_j/2`;
   Underwood's `v_f·k_c/e`, Drake's `v_f·k_c/√e`, Greenberg's critical density
   `k_j/e`; Daganzo's triangle peaks at `k_c = w·k_j/(v_f + w)` with wave speeds
   exactly `v_f` and `−w`. Rankine-Hugoniot, Webster's uniform delay and optimal
   cycle, Akçelik's threshold and the HCM LOS bands are all asserted against the
   published formulas; a deterministic (`p = 0`) NaSch ring has flow exactly
   `min(ρ·v_max, 1 − ρ)`, peaking at `v_max/(v_max + 1)`; the Optimal-Velocity
   string margin is exactly `κ²/2 − κ·V′(s_e)`.
2. **Exact identities.** Godunov, CTM, LTM, METANET, Payne-Whitham and ARZ all
   satisfy `storage(T) − storage(0) == in − out` to machine precision; a uniform
   state stays **bit-exactly** uniform for every diagram; every car-following law
   gives *exactly* zero acceleration at its own equilibrium spacing and
   `equilibrium_speed(equilibrium_spacing(v)) == v`; the delay between two curves
   equals an independent trapezoid.
3. **Metamorphic / invariant.** Demand is non-decreasing and supply
   non-increasing on every concave diagram; densities stay in `[0, k_jam]` under
   CFL; the shock-front error does not grow under refinement; halving `dt` halves
   the CTM's gap to the LTM's exact storage (first order, to 10 %); a
   string-unstable platoon amplifies the tap a stable one damps.
4. **Differential.** There is no importable reference for a kinematic-wave solver
   — SUMO and Aimsun are simulators, not numerics — so this tier is **internal
   cross-model agreement**: the CTM reproduces Godunov cell for cell across the
   whole zoo, the LTM agrees with the CTM under refinement, Newell's
   three-detector construction reproduces the LTM's own boundary curves at `x = 0`
   and `x = L`, the CTM's steady state inverts the fundamental diagram, and the
   deterministic CA reproduces Daganzo's triangle. A small optional NumPy tier
   (`linalg.lstsq` for the MFD fit, `trapezoid` for the curve area) sits behind
   `pytest.importorskip` and skips cleanly.

```bash
pytest _stress_test.py -q
PYTHONHASHSEED=0 pytest _stress_test.py -q     # also 1 and 1010
```

**What the suite caught.** One real template defect: `TrapezoidalDiagram.flow`
inherited the base `k · speed(k)`, but on the capacity plateau `speed` is
`q_max / k`, so the round trip came back one ulp off `q_max`. That made `flow()`
disagree with `demand()` for the same state and injected a one-ulp drift into an
otherwise bit-exact Godunov update. Computing it in min-form returns `q_max`
exactly — `q_max` is itself one of the operands — and a uniform state now stays
bit-exactly uniform for every diagram in the zoo over 4000 steps.

Several suite **oracles were corrected rather than weakened**, which is the more
interesting half of the record. Wave-speed monotonicity only holds for *concave*
diagrams, so Underwood and Drake are now excluded and pinned instead by a new
test asserting their analytic inflection points (`k = 2·k_c`, `k = √3·k_c`). The
CTM discharge test divided cumulative outflow by total elapsed time while
ignoring the corridor fill time. The capacity-drop test never congested the cells
carrying the parameter. The IDM test encoded the non-canonical clamping variant.
The Krauss dawdling invariant was asserted on a platoon end state where it does
not hold, rather than per step where it does. And the CTM-versus-point-queue
comparison asserted an equality that over-counts, because a queue at jam density
displaces the free-flowing vehicles that stretch would otherwise hold — it is now
bracketed, with exact conservation tying the three numbers together.

---

## References

- Lighthill & Whitham (1955); Richards (1956) — the kinematic-wave (LWR) model.
  Godunov (1959); LeVeque, *Finite Volume Methods for Hyperbolic Problems*.
- Greenshields (1935); Greenberg (1959); Underwood (1961); Drake, Schofer & May
  (1967); Newell (1961); Del Castillo & Benítez (1995) — the speed-density laws.
- Daganzo (1994, 1995) — *The Cell Transmission Model*, I and II; Daganzo (1997),
  *Fundamentals of Transportation and Traffic Operations*; Lebacque (1996) — the
  demand/supply formulation. Newell (1993) — *A simplified theory of kinematic
  waves*, I-III; Yperman (2007) — the Link Transmission Model.
- Webster (1958), *Traffic Signal Settings*; Akçelik (1981), *Traffic Signals:
  Capacity and Timing Analysis*; TRB *Highway Capacity Manual* — the LOS bands;
  Diakaki, Papageorgiou & Aboudolas (2002) — store-and-forward (TUC).
- Edie (1963) — generalised flow, density and speed; Geroliminis & Daganzo (2008)
  — the MFD; Daganzo (2007) — urban gridlock and accumulation dynamics.
- Papageorgiou, Blosseville & Hadj-Salem (1990) — METANET; Payne (1971), Whitham
  (1974); Daganzo (1995) — *Requiem for second-order fluid approximations*; Aw &
  Rascle (2000), Zhang (2002) — ARZ.
- Gipps (1981); Bando et al. (1995) — Optimal Velocity; Treiber, Hennecke &
  Helbing (2000) — IDM; Jiang, Wu & Zhu (2001) — Full Velocity Difference; Krauß
  (1998) — the model behind SUMO; Kesting, Treiber & Helbing (2007) — MOBIL.
- Nagel & Schreckenberg (1992) — the cellular automaton; Wilson & Ward (2011) —
  car-following models and string stability; Treiber & Kesting (2013), *Traffic
  Flow Dynamics* — the standard modern reference for most of the above.

---

*Pure Python, standard library only (`math`, `random`, `dataclasses`, `typing`).
`traffic_flow_template.py` — `__version__ = "1.0.0"`, 91 public names in
`__all__`, 237 oracle-based tests.*
