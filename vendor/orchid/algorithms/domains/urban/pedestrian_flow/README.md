# Pedestrian Flow

People are a density field long before they are a crowd. This module treats
walkers the way hydraulics treats water: a density ρ in persons per square metre,
a speed that falls as ρ rises, and a conservation law that never loses a single
person. Six published pedestrian fundamental diagrams — Weidmann's Kladek law,
Older, Fruin, SFPE, Predtechenskii-Milinskii, Virkler — each with an **analytic**
capacity; the 1-D conservation PDE `∂ρ/∂t + ∂(ρv)/∂x = s` solved by **Godunov** in
the sending/receiving formulation with an exact people ledger; Hughes' continuum
model, where an **eikonal** potential solved by fast marching bends around the
congestion ahead; the Helbing-Molnár **social force** integrator; a floor-field
**cellular automaton**; Fruin's A-F level of service; Adams' gap-acceptance and
signalised crossing delay; and flow-limited evacuation egress.

Pure standard library — the Godunov flux, the fast marching, the symplectic
integrators and the lattice gas are all built by hand. This is deliberately not
`urban/traffic_flow`, the *vehicle* conservation layer: cars have a triangular
lane-based diagram and couple through turning movements and signal capacity,
while pedestrians have a smooth concave diagram in persons per metre of width per
second, walk in two dimensions, form counterflow lanes, and route themselves by a
potential field. Nor is it `urban/active_travel`, which audits whether a walking
network is *good*; this module asks how many people fit on it, how fast they
move, and when it fails.

```python
from pedestrian_flow_template import (fd_weidmann, WalkwayLink, simulate_walkway,
    fruin_los_from_density, crossing_delay, egress_time)

fd = fd_weidmann()                     # Kladek law: v0 1.34 m/s, jam 5.4 P/m²
print(fd.capacity(), fd.critical_density())      # 1.2249 P/m/s at 1.751 P/m²

footway = WalkwayLink(40.0, 3.0, fd, n_cells=200, name="high-street")
sim = simulate_walkway(footway, 120.0, inflow=2.4, outflow_capacity=1.5)
print(sim.max_density, sim.conservation_error)   # 4.298 P/m², 5e-12 people
print(fruin_los_from_density(sim.max_density).grade)             # F

x = crossing_delay(7.3, control="signalised", cycle=90.0, walk_green=25.0)
print(x.delay, x.level_of_service)               # 23.47 s, LOS D

eg = egress_time(450.0, 1.8, 35.0, pre_movement=60.0)
print(eg.total_time, eg.controlling)             # 290.8 s, flow-controlled
```

ρ is P/m², `v` is m/s, and the *specific flow* `q = ρv` is persons per metre of
width per second (P/m/s) — multiply by the width for a total flow in P/s. Time is
seconds, length metres, delay seconds per person. Everything stochastic takes a
`seed` and drives a private `random.Random`, so output is byte-identical and
independent of `PYTHONHASHSEED`.

---

## Pick

| Goal | Use |
| --- | --- |
| A speed-density law with a real capacity | `fd_weidmann` / `get_fd` / `FD_NAMES` |
| Grade a footway, a concourse or a stair | `fruin_los_from_density` / `fruin_los` |
| "Is this crowd dangerous?" | `crowd_density_band` / `crush_risk` |
| Delay at an uncontrolled crossing | `crossing_delay` / `adams_delay` |
| Size a pedestrian signal phase | `pedestrian_phase_times` / `pedestrian_minimum_green` |
| A hand-calculation evacuation time | `egress_time` / `exit_capacity` |
| A crowding wave down a pavement | `WalkwayLink` + `simulate_walkway` |
| The exact speed of a congestion front | `riemann_solution` / `riemann_shock_speed` |
| A merge, a diverge or a whole network | `merge_flows` / `node_transfer` / `simulate_network` |
| Two opposing streams on one pavement | `bidirectional_capacity` / `simulate_lane_formation` |
| The least-travel-time field around an obstacle | `fast_marching` / `potential_descent_path` |
| Which walking route people actually take | `multinomial_logit` / `path_size_logit` |
| A stand, a station or a stadium emptying | `stadium_egress_profile` / `accumulation_profile` |
| A 2-D crowd routing itself out of a hall | `HughesGrid` + `simulate_hughes` |
| Individual walkers, contact forces, a bottleneck | `simulate_social_force` / `FloorFieldCA` |
| A life-safety verdict | `predtechenskii_milinskii_egress` + `safety_margin` |
| JSON a browser can render | `build_pedestrian_flow_payload` / `build_a2ui_payload` |

---

## Install

Nothing to install. `pedestrian_flow_template.py` is a single zero-dependency
file — copy it next to your code and import it; the only entry in
`requirements.txt` is `pytest>=7.0`, for the suite. No NumPy, no SciPy, no GIS,
CPython 3.8+.

```bash
python pedestrian_flow_template.py     # the demo, prints "demo OK"
pytest _stress_test.py -q              # 140 oracle-based tests
```

---

## What's inside

### 0 — Validation

Every entry point coerces its arguments through shared finite / positive /
non-negative / unit-interval helpers and raises `PedestrianFlowError` (a
`ValueError` subclass) naming the offender, so a `NaN` never reaches a solver.

### 1 — Pedestrian fundamental diagrams

`PedestrianFD` is one speed-density law with `speed`, `flow`, an **analytic**
`dflow` (the kinematic wave speed), `capacity`, `critical_density`,
`capacity_speed`, `density_from_flow` (invertible on either branch) and
`to_dict`. Six families — `fd_weidmann` (Kladek), `fd_older`, `fd_fruin`
(Greenshields), `fd_sfpe`, `fd_predtechenskii_milinskii` (tabulated),
`fd_virkler` (Pipes-Munjal) — via `get_fd`, listed in `FD_NAMES`. Every capacity
is closed-form except Weidmann's, bisected on the **analytic** derivative rather
than a sampled grid. `fd_speed_curve` / `fd_flow_curve` sample for plotting;
`max_speed_slope` guards monotonicity.

### 2 — Speed-density modulation

The same laws as bare multipliers: `phi_linear`, `phi_exponential` (Kladek) and
`phi_power` (Pipes-Munjal), clipped to `[0, 1]` and combined by `speed_from_phi`.
`tobler_walking_speed` applies Tobler's hiking function to a gradient, normalised
so a flat path returns exactly the free speed (its true maximum is at −5 %).

### 3 — The conservation PDE (Godunov)

`demand_flow` (sending) saturates at capacity above ρ_c, `supply_flow` below it,
and `godunov_flux` is their minimum — the exact Riemann flux for a concave
diagram. `godunov_step` is one explicit update, `cfl_timestep` / `max_wave_speed`
set a stable step, `WalkwayLink` (`set_uniform`, `set_profile`, `people`,
`capacity_pps`, `cell_centres`, `copy`) holds the state, and `simulate_walkway`
returns a `WalkwaySimulation` with the ledger and its `conservation_error`.
Analytically: `riemann_solution` classifies shock / rarefaction / constant,
`riemann_shock_speed` is Rankine-Hugoniot, `front_position` finds a level
crossing, `walkway_travel_time` is `L/v(ρ)`.

### 4 — Node coupling

`merge_flows` shares a merge capacity by priority water-filling, `diverge_flows`
is the FIFO diverge blocked by the tightest downstream supply, and
`node_transfer` is the general junction — FIFO-consistent supply rationing
iterated to a fixed point, returning a `NodeFlowResult` with the
origin-destination matrix and reduction factors. `Junction`, `PedestrianNetwork`
and `simulate_network` couple links up, `NetworkSimulation` keeping one ledger
across every junction and exit.

### 5 — Bidirectional flow and lane formation

`counterflow_penalty` removes speed in proportion to the *opposing* fraction,
`bidirectional_speeds` applies it to both streams, `bidirectional_capacity`
sweeps for the two-way maximum. `lane_formation_index` is the order parameter
over lateral strips (1 = segregated) and `simulate_lane_formation` a seeded
sorting model returning a `LaneFormationResult` — at index 1 the penalty
vanishes.

### 6 — Eikonal potential and Hughes' model

`fast_marching` solves `|∇T| = 1/F` by a Dijkstra-like sweep on the upwind
quadratic, so diagonals cost what they should; `eikonal_descent_direction` gives
the unit walking direction, `potential_descent_path` the steepest-descent route.
`HughesGrid` (`fill`, `people`, `copy`) plus `simulate_hughes` closes the 2-D
continuity equation with that potential — congestion ahead bends the field — into
a `HughesResult` with the egress series, clearance time and exact ledger.

### 7 — Social force (Helbing-Molnár)

`SocialForceParams` holds the published constants and `Pedestrian` / `Wall` the
agents and geometry. `social_forces` assembles the driving, social, wall, body
and sliding-friction terms, `social_force_step` advances by semi-implicit Euler
or velocity-Verlet, `simulate_social_force` runs a crowd into a
`SocialForceResult` (trajectories, exit times, minimum separation), and
`social_force_equilibrium_separation` is the closed-form two-agent equilibrium
the suite pins the integrator against.

### 8 — Floor-field cellular automaton

`static_floor_field` is the distance-to-exit field (fast marching at unit speed).
`FloorFieldCA` is Burstedde-Klauck-Schadschneider-Zittartz: a parallel-update
lattice gas driven by that static field plus a decaying **dynamic** field of
virtual traces, conflicts resolved by the friction parameter μ. `step` advances
once; `run` returns a `CAResult` with egress times, flow and occupancy series.

### 9 — Level of service and crowd risk

`fruin_los` grades an area module (m²/person) or a unit flow (P/min/m) for
walkways, queueing areas and stairways, `fruin_los_from_density` takes a density,
`los_bands` returns the table — the imperial originals (35/25/15/10/5 ft²,
7/10/15/20/25 P/min/ft) converted once, here. `crowd_density_band` maps a density
onto a named `CrowdBand` with a 0-5 risk score, `crowd_pressure` is Helbing's
`ρ·Var(v)` turbulence measure, and `crush_risk` combines both.

### 10 — Crossings, delay and signal timing

`critical_gap` is width/speed plus margins, `gap_acceptance_probability` is
`exp(−q t_c)`, `adams_delay` is `(e^(q t_c) − q t_c − 1)/q`,
`signalised_pedestrian_delay` is `(C − g)²/(2C)`, `two_stage_crossing_delay` adds
a refuge, and `crossing_delay` wraps all three into a `CrossingDelayResult`.
`pedestrian_delay_los` grades a delay on the HCM bands; `pedestrian_minimum_green`
and `pedestrian_phase_times` (→ `PedestrianPhase`) size the phase;
`platoon_dispersion` runs Robertson's filter over an arrival profile into a
mass-conserving `PlatoonResult`.

### 11 — Route choice

`WalkRoute` carries what a pedestrian trades off — length, crossings, crossing
delay, stress, comfort, gradient — falling back to a Tobler-corrected walking
time. `RouteCoefficients` holds the linear weights, `route_utility` evaluates
them, `multinomial_logit` and `path_size_logit` (the overlap correction for
shared links) give the probabilities in a `RouteChoiceResult`, and
`logsum_accessibility` is the expected maximum utility.

### 12 — Evacuation, egress and accumulation

`effective_width` applies the SFPE boundary layers (a stream never uses the full
geometric width), `exit_capacity` turns that into P/s, `egress_time` returns an
`EgressResult` saying whether travel or flow controls, and
`predtechenskii_milinskii_egress` is the staged Russian method — per-section
densities, speeds, flows and merge delays (`PMEgressResult`). `evacuation_curve`
is the cumulative egress curve, `accumulation_profile` the input-output diagram
with queue, delay and clearance (`AccumulationResult`), `stadium_egress_profile`
shapes a stand's release into it, `safety_margin` gives the ASET−RSET verdict.

### 13 — Cross-model comparison

`cross_model_egress` traverses one corridor three ways — the Godunov PDE (with a
tracer), the floor-field CA and the social-force integrator — returning a
`CrossModelResult` with all three traversal times and their spread. At free-flow
density three independent solvers sharing one `v0` must agree, which is what
makes this the module's differential tier.

### 14 — Frontend payloads

`build_pedestrian_flow_payload` is the full JSON-able study — diagram curve,
density profile, ledger, LOS, crossing, egress, route choice, accumulation — and
`build_a2ui_payload` the visualisation cut: stat cards, chart series and
colour-coded bands. Non-finite floats become `None`, so both survive
`json.dumps`.

---

## Recipes

**Track a congestion front and check it against Rankine-Hugoniot.** A step from
1.0 to 4.5 P/m² is a shock, and it travels *upstream*:

```python
from pedestrian_flow_template import (fd_weidmann, WalkwayLink, simulate_walkway,
    riemann_solution, riemann_shock_speed, front_position)

fd = fd_weidmann()
rl, rr = 1.0, 4.5
print(riemann_solution(fd, rl, rr).kind)     # shock

link = WalkwayLink(40.0, 3.0, fd, 400, "shock")
link.set_profile(lambda x: rl if x < 20.0 else rr)
run = simulate_walkway(link, 20.0, inflow=fd.flow(rl) * 3.0,
                       outflow_capacity=fd.flow(rr) * 3.0)
front = front_position(run.density, run.dx, 0.5 * (rl + rr))
print((front - 20.0) / run.time, riemann_shock_speed(fd, rl, rr))  # -0.1846 vs -0.18446
```

**Cost the counterflow on a two-way footway, then let lanes form.** Even
counterflow throws away 12.5 % of the one-way capacity; self-organised lanes hand
it back:

```python
from pedestrian_flow_template import (bidirectional_capacity, bidirectional_speeds,
    simulate_lane_formation, crowd_pressure, crush_risk, fruin_los)

r1, q1 = bidirectional_capacity(fd, split=1.0)        # one-way
r2, q2 = bidirectional_capacity(fd, split=0.5)        # even counterflow
print(q1, q2, 100.0 * (1.0 - q2 / q1))                # 1.225, 1.072, 12.5 % lost

lanes = simulate_lane_formation(8, 40, 40, 150, beta=2.0, seed=3)
vp, vm = bidirectional_speeds(0.8, 0.8, fd, friction=0.25,
                              lane_index=lanes.final_index)
print(lanes.initial_index, lanes.final_index, vp)     # 0.125 -> 1.0, 0.762 m/s

vel = [(0.2, 0.1), (-0.3, 0.4), (0.1, -0.5), (0.4, 0.2), (-0.1, -0.2)]
print(crush_risk(4.5, crowd_pressure(4.5, vel))["level"])   # danger
print(fruin_los(1.0 / 4.5, "queueing").grade)               # E
```

**Compare three ways of controlling one 7.3 m crossing.** Adams' delay is convex,
so a refuge splitting the carriageway beats one whole crossing — and both beat a
90 s signal cycle:

```python
from pedestrian_flow_template import (critical_gap, adams_delay, crossing_delay,
    gap_acceptance_probability, pedestrian_phase_times)

tc = critical_gap(7.3, walking_speed=1.2, safety_margin=1.5)
print(tc, gap_acceptance_probability(900.0, tc), adams_delay(900.0, tc))
# 7.583 s critical gap | 15.0 % of gaps acceptable | 15.05 s mean wait

one = crossing_delay(7.3, vehicle_flow=900.0)
two = crossing_delay(7.3, vehicle_flow=450.0, vehicle_flow_2=450.0,
                     refuge=True, refuge_dwell=2.0)
sig = crossing_delay(7.3, control="signalised", cycle=90.0, walk_green=25.0)
for r in (one, two, sig):
    print(r.control, round(r.delay, 2), r.level_of_service)
# unsignalised-one-stage 15.05 C | two-stage 5.14 B | signalised 23.47 D

phase = pedestrian_phase_times(7.2, 1.2, walk_interval=7.0, cycle=90.0)
print(phase.walk_interval, phase.clearance_interval, phase.delay)   # 7.0 + 6.0 s
```

**Empty a stand and pass the life-safety test** — a shaped release into
capacity-limited exits, then ASET versus RSET:

```python
from pedestrian_flow_template import (stadium_egress_profile, exit_capacity,
    safety_margin, build_a2ui_payload)

exits = [3.0, 3.0, 2.4, 2.4]
print(sum(exit_capacity(w) for w in exits))          # 12.48 P/s through four doors

stand = stadium_egress_profile(9000.0, exits, release_duration=420.0, dt=15.0,
                               shape="triangular")
print(stand.max_queue, stand.clearance_time, stand.mean_delay)  # 4521, 765 s, 181 s
print(safety_margin(900.0, stand.clearance_time)["verdict"])    # marginal

ui = build_a2ui_payload(fd, accumulation=stand)      # stat cards + chart series
```

---

## Design notes

- **Boundary and junction flows are TOTAL, not specific.** `simulate_walkway`'s
  `inflow` / `outflow_capacity` and every junction transfer are in **P/s**.
  `godunov_step` takes the width explicitly and carries those units into the cell
  update, so no division by the width is ever needed and a uniform state is held
  **bit-exactly** rather than drifting an ulp per step; with `width=1.0` every
  flux is a specific flow instead.
- **Nothing is clamped inside the PDE.** Positivity and the jam bound fall out of
  the demand/supply flux itself under the CFL condition — which is exactly what
  makes the ledger exact: `stored_initial + inflow + source − outflow ==
  stored_final` to ~1e-12, with no clipping to hide a leak.
- **The Predtechenskii-Milinskii diagram is not globally concave, on purpose.**
  It is empirical, and `D = 0.40-0.50` carries one interior kink (a local maximum
  of `q` at ρ = 4.857 P/m², 0.087 % above the next table node), so its sending
  function dips by 2.4e-3 P/m/s over a 0.14 P/m² window before rising to the true
  capacity at ρ = 5.8. Published data, not a transcription error: the ledger and
  `godunov_flux(ρ, ρ) == q(ρ)` stay exact regardless. PM is also the *fast*
  model — its high-density flows sit well above Weidmann and SFPE.
- **`fruin_los` wants an area module, not a density.** The `'area'` basis is m²
  **per person** — pass `1/ρ`, or use `fruin_los_from_density`, which does it for
  you and fills in the `density` field. An empty facility is an unbounded area
  module and always grades A.
- **The social-force body spring is stiff.** Its period is ~0.16 s at the
  published constants, so contact-dominated runs (bottlenecks, panic) need
  `dt ≤ 0.005 s`; 0.01-0.02 s is ample while nobody is touching. Both integrators
  are symplectic and reproduce free walking exactly.
- **The CA's free speed is `cell_size / time_step` only if `k_s` is large.** At
  the classic `k_s = 2.5` with 0.4 m cells an agent advances ~0.67 cells per step,
  the walk still being partly a downhill-biased random walk, and the parallel
  update passes at most one person per exit cell per step — so
  `cross_model_egress` compares **free-flow** traversal, where all three models
  share one `v0` and must agree.
- **Vehicle flows are veh/h, and `mobility_impaired` wins.** `adams_delay` and
  `gap_acceptance_probability` divide by 3600 themselves; above `q t_c = 700`
  Adams refuses rather than overflowing, because the crossing never clears. And
  `mobility_impaired=True` pins 1.0 m/s, ignoring whatever `walking_speed` was
  passed — it is a design standard, not a suggestion.

---

## Testing

`_stress_test.py` is an oracle-based suite — **140 tests** across all fourteen
sections, no golden outputs anywhere, in four tiers:

1. **Closed forms.** Every diagram's capacity against a fine numerical argmax of
   `q(ρ)`; Older and Fruin reproduce `v0 ρ_jam / 4` at `ρ_jam / 2`, Virkler its
   power-law capacity, SFPE its plateau; the signalised delay is `(C − g)²/(2C)`
   (exactly `C/2` at `g = 0`); Adams' is `(e^(q t_c) − q t_c − 1)/q`; the minimum
   green is `distance / speed`; the two-agent social-force equilibrium is
   `2r − B ln(m v0 / (τ A))`; a queue's delay is its trapezoidal area.
2. **Exact identities.** The Godunov people ledger closes to ~1e-10; a uniform
   state stays **bit-exactly** uniform; free-flow traversal is exactly `L / v0`;
   `godunov_flux(ρ, ρ) == q(ρ)`; `density_from_flow` inverts `flow` on both
   branches; junction transfers conserve people exactly and never violate a
   downstream supply; the CA advances exactly one cell per step at large `k_s`.
3. **Metamorphic / invariant.** Speed falls with density and the demand/supply
   flows are monotone; refining the grid shrinks the tracked shock-speed error
   against Rankine-Hugoniot; a two-way pavement's capacity is below a one-way one
   and lane formation restores it; fast marching scales as `1/F`; Adams' delay
   rises with flow; the logit is IIA and the logsum dominates the best utility;
   the path size of a disjoint route is exactly `L_k / L_shortest`.
4. **Differential — internal, by design.** There is no third-party
   pedestrian-flow library to differentiate against, so the fourth tier is an
   *internal* differential: one corridor solved by the Godunov PDE, the
   floor-field CA and the Helbing-Molnár integrator, which must agree on the
   free-flow traversal time to a few per cent. The usual `pytest.importorskip`
   tier is therefore empty — the suite is zero-dependency and skips nothing.

```bash
pytest _stress_test.py -q
PYTHONHASHSEED=0    pytest _stress_test.py -q
PYTHONHASHSEED=1010 pytest _stress_test.py -q
```

**What the suite caught.** None — the suite was green on the first full run,
which is itself worth recording: the conservation, free-flow and closed-form
crossing-delay oracles all held without a single template change. The conventions
that usually generate the bugs here — total-versus-specific flows at the
boundary, the unclamped update, the analytic rather than sampled capacity — were
built in from the start, and the ledger proved them rather than caught them out.

---

## References

- Lighthill & Whitham (1955), Richards (1956) — the kinematic-wave (LWR) model.
  Godunov (1959) — the upwind flux. Daganzo (1994, 1995) — the cell transmission
  model and the sending/receiving decomposition used at every face and junction.
  Newell (1993) — simplified kinematic-wave theory and cumulative curves. Treiber
  & Kesting (2013) — *Traffic Flow Dynamics*.
- Weidmann (1993), after Kladek (1966) — the exponential speed-density law and
  the `v0 = 1.34 m/s`, `ρ_jam = 5.4 P/m²` calibration. Older (1968) — shopping
  streets. Fruin (1971) — *Pedestrian Planning and Design*: the walkway, queueing
  and stairway LOS bands. Nelson & Mowrer, *SFPE Handbook of Fire Protection
  Engineering* — the plateau law, boundary layers and effective width.
  Predtechenskii & Milinskii (1978) — *Planning for Foot Traffic Flow in
  Buildings*. Virkler & Elayadath (1994) — the Pipes-Munjal power form.
- Hughes (2002) — *A continuum theory for the flow of pedestrians*. Sethian
  (1996), Tsitsiklis (1995) — fast marching for the eikonal equation. Helbing &
  Molnár (1995) — the social force model; Helbing, Farkas & Vicsek (2000) —
  escape panic (body and sliding-friction terms); Helbing, Johansson &
  Al-Abideen (2007) — crowd turbulence and the pressure `ρ·Var(v)`. Burstedde,
  Klauck, Schadschneider & Zittartz (2001) — the floor-field cellular automaton;
  Schadschneider, Chowdhury & Nishinari (2010) — *Stochastic Transport*.
- Adams (1936) — *Road Traffic Considered as a Random Series*. Robertson (1969) —
  TRANSYT platoon dispersion. *Highway Capacity Manual* (TRB, 6th ed.) —
  pedestrian crossing delay bands and signal timing. Ben-Akiva & Lerman (1985) —
  discrete choice and the logsum; Ben-Akiva & Bierlaire (1999) and Cascetta et
  al. (1996) — the path-size correction. Tobler (1993) — the hiking function.

---

*Pure standard library: `bisect`, `heapq`, `math`, `random`, `dataclasses`,
`typing`. `pedestrian_flow_template.py` — `__version__ = "1.0.0"`, 106 public
names in `__all__`, 140 oracle-based tests.*
