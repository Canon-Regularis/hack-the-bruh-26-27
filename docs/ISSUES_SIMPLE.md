# HackTheBruh -- algorithmic system issues

These issues cover the algorithmic pipeline only:

- road graph, map matching, speed prediction and validation
- rule binding and the OSM mismatch
- demand estimation and traffic assignment
- pedestrian delay and the concern field
- scenario reduction, robust search, Pareto selection and policy check

Data collection, the API/map UI and the pitch are out of scope here; data-handling steps that an
algorithm needs are folded into that algorithm's issue.

The issues are in order of importance: the MVP algorithm set first, then Stages 1-4. Each stage
builds on the one before. Every algorithm calls ORCHID through `from engine.orchid import ...`.
Modules that are still held (policy_compiler, public_voice, network_design) have glue stand-ins
until their ORCHID suites converge.

One GitHub issue per heading: the heading is the issue title. "Depends on" references use the ID
at the start of each title.

## MVP -- the core algorithm set

### [ALG-MVP-0] Vendor the ORCHID algorithms behind engine.orchid (done)

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah

**Description:** Done on branch `feat/vendor-orchid`, but not yet committed or merged.

- **Vendored:** urban street_network, traffic_assignment, travel_demand, pedestrian_flow,
  emissions_environment, traffic_flow, accessibility_equity and signal_control, plus core
  data_splitting, ensembles, multi_objective and stochastic_optimisation. All are copied byte for
  byte from ORCHID `a2e5e88`.
- **Held:** policy_compiler, public_voice, network_design, resilience, green_infrastructure and
  probabilistic. Importing one raises ImportError with the reason.

### [ALG-MVP-1] Road graph: OSM to a cleaned RoadGraph with frozen edge ids

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah

**Description:** Build `load_graph()` in `src/graph/build_graph.py` using `street_network.RoadGraph`.

- `add_node` with lat/lon. The first node fixes the projection origin.
- `add_edge` with `oneway=False` for two-way ways (the default is `oneway=True`). Swap u/v for
  `oneway=-1`.
- Take lanes and speeds from `HIGHWAY_DEFAULTS`. Keep the way id, name and access tags.
- Run `clean_network`. Its simplify step removes crossing and signal nodes, so disable it or
  preserve those nodes.
- Freeze edge ids after cleaning, then build `EdgeIndex` once.

Every downstream algorithm keys on these edge ids: map matching, features, assignment and rules.
Depends on: cached OSM and the study bbox.

### [ALG-MVP-2] Gazetteer: bind street names to edge ids

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah

**Description:** Build `src/graph/gazetteer.py`.

- Normalise OSM names (case, punctuation, abbreviations) but keep qualifiers (Upper, Lower,
  North...) exact.
- Map each name to its edge ids in both directions.
- Report any names that don't resolve.

It binds legal clauses and complaints to edges. This is a glue stand-in for the held
`policy_compiler.Gazetteer` (see ALG-S3-2). Depends on ALG-MVP-1.

### [ALG-MVP-3] Map matching and traversal speeds per edge

**Assignees:** Matthew Li, Adish Boodhun

**Description:**

1. Split each bus's pings into time-ordered trips, breaking on gaps.
2. Run `street_network.map_match(graph, [(lat, lon), ...], geographic=True, index=idx,
   mode="bus")`. This is HMM/Viterbi matching, and latitude comes first.
3. `map_match` returns matched positions only, not speeds. Derive each edge's traversal speed
   from consecutive matched positions: (offset_frac x length + routed distance) / time between
   pings. Strip dwell time at stops.
4. Aggregate to edge x 15-min bins (speed_kmh, n_traversals, n_vehicles) and report coverage.

Fallback: nearest-edge `snap_point` on bus corridors only. Depends on ALG-MVP-1.

### [ALG-MVP-4] Rule binding and OSM-mismatch detection

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah

**Description:** Input: ~15 hand-authored clauses. Each has a verbatim quote, page, streets,
restriction type, times and exemptions.

1. Check that each quote is an exact substring of its cached source.
2. Mark any clause that can't be encoded honestly (e.g. "with exemptions") as abstained, with a
   reason.
3. Bind each clause to edges through the gazetteer.
4. Compare each bound restriction with the OSM access, motor_vehicle, oneway and
   highway=pedestrian tags.
5. Output per-edge restriction flags for the speed model.

Done when a script prints "N of M legal restrictions missing or wrong on OSM" (or that there are
none), plus the list of abstained clauses. This is a glue stand-in for the held policy_compiler
(see ALG-S3-2). Depends on ALG-MVP-2.

### [ALG-MVP-5] Feature engineering for the speed model, with leakage firewall

**Assignees:** Matthew Li, Adish Boodhun

**Description:** Features per edge x bin:

- road class (numeric), lanes, length, speed limit;
- `street_network.edge_betweenness_centrality(graph, mode="car", weight="free_flow_time_s")`;
- restriction flags from ALG-MVP-4;
- hour and weekday;
- spatial lag from neighbouring edges' recent speeds.

Leakage firewall: features may use only other edges and earlier bins. Impute missing values and
add an indicator column for each, because the ORCHID GBR needs finite 2-D lists. Depends on
ALG-MVP-3 (ALG-MVP-4 is optional at first).

### [ALG-MVP-6] Speed prediction: two baselines, GBM, GroupKFold by street

**Assignees:** Matthew Li, Adish Boodhun

**Description:**

- **Baselines:** (1) mean by road class x hour; (2) nearest observed edge in the same bin.
- **Model:** GBM via sklearn/LightGBM, with `engine.orchid.ensembles.GradientBoostingRegressor`
  as the fallback.
- **Validation:** `data_splitting.GroupKFold(n_splits=5)` grouped by street, so held-out streets
  are genuinely unobserved.
- **Report:** MAE (km/h) and skill = 1 - MAE/MAE_baseline against both baselines, labelled
  "bus-observed speed".

If the GBM doesn't beat baseline 2, report that honestly and ship baseline 2. Depends on
ALG-MVP-5.

### [ALG-MVP-7] Predict unobserved edges and quantify coverage

**Assignees:** Matthew Li, Adish Boodhun

**Description:** Fit on all observed data, then predict every edge x bin that has no bus
observation. Flag each value as observed or predicted. Compute: "X% of street-time in the study
area has no direct observation; we predict it with error Y", taking Y from ALG-MVP-6. Output
per-edge observed/predicted speeds for the rest of the pipeline. Depends on ALG-MVP-6.

## Stage 1 -- validation and the traffic model

### [ALG-S1-1] Time-split validation ("next hour")

**Assignees:** Matthew Li, Adish Boodhun

**Description:** Run `data_splitting.TimeSeriesSplit` over the unique 15-min bins. It splits by
position, so split the list of bins and then expand back to rows. Report the time-split MAE and
skill next to the GroupKFold numbers. Confirm the leakage firewall holds: no feature uses a later
bin or the same street. Depends on ALG-MVP-6.

### [ALG-S1-2] Match DfT count points to edges

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah

**Description:** Attach each DfT Edinburgh count point in or near the bbox to its nearest edge.
`snap_point` takes projected metres, so convert lat/lon first, using either:

- `map_match` on the single point with `geographic=True`, or
- `graph._project`.

Then call `snap_point(..., index=idx)` and check each match by eye. Output AADF per matched edge.
It is a model feature (ALG-MVP-5) and the GLS target (ALG-S1-4). Depends on ALG-MVP-1.

### [ALG-S1-3] RoadGraph to traffic_assignment Network adapter + Braess sanity check

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah

**Description:** ORCHID has no adapter, so write one:

- dense node ids 0..n-1;
- only car edges with capacity > 0;
- `net.add_link(u, v, BPR(...))` with free-flow time (s) and capacity (veh/h);
- edge-to-link maps for both directions.

Test: on the textbook Braess network, `detect_braess` must give 83/498 -> 92/552. Pass this
before trusting any assignment number. Depends on ALG-MVP-1.

### [ALG-S1-4] Demand estimation: gravity prior + GLS fit to counts

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah

**Description:**

1. Use at most 20-25 zones, because GLS is `O((n*m)^3)`. Give each zone its own centroid node.
2. Costs are free-flow times, all in one unit.
3. Build the prior with `travel_demand.gravity_model(productions, attractions, costs,
   ExponentialDeterrence(beta))`. beta is in inverse cost units.
4. Fit with `gls_matrix_estimation(prior, counts, AssignmentProportions(...))` against the matched
   counts.

Report modelled vs counted flows, and label the OD matrix as estimated. Depends on ALG-S1-2,
ALG-S1-3.

### [ALG-S1-5] Baseline user equilibrium

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah

**Description:** Run `traffic_assignment.user_equilibrium(net, od, method="b")`. Avoid the
Frank-Wolfe variants, which converge slowly at the end. Assert `converged` and report
`relative_gap`. Output link flows, times and v/c for both directions. Depends on ALG-S1-3,
ALG-S1-4.

### [ALG-S1-6] Validate modelled link times against bus-observed speeds

**Assignees:** Matthew Li, Adish Boodhun

**Description:** Compare UE link times with observed and predicted bus speeds on the same edges.
Report the MAE and bias as the simulator's measured error. State the caveat: buses are not cars
(stops, bus lanes). Depends on ALG-S1-5, ALG-MVP-7.

### [ALG-S1-7] Pedestrian crossing delay and level of service (J_ped)

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah

**Description:** For each crossing:

- `pedestrian_flow.crossing_delay(width_m, control="signalised"|"unsignalised",
  vehicle_flow=<UE veh/h>, ...)`. Signalised crossings also need `cycle` and `walk_green`.
- `fruin_los_from_density(rho)` with assumed densities, labelled synthetic.

Sum the delays to get the baseline J_ped. Depends on ALG-S1-5.

### [ALG-S1-8] Concern field per edge (glue)

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah

**Description:** Inputs are the published consultation records: theme, location or named street,
and count. Compute each edge's intensity per theme with the plan's kernel:

`s_e^j = sum_i sev_i * K(dist(e, loc_i)) * 1[theme_i = j]`

If the comments can't be located, use counts per named street instead and label them that way.
Never invent records. The output feeds the speed model (concern-intensity feature) and later
J_voice. This is a glue stand-in for the held public_voice (see ALG-S3-4). Depends on ALG-MVP-2.

## Stage 2 -- the decision algorithms

### [ALG-S2-1] Candidate interventions and rule filter before enumeration

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah

**Description:** Pick 6-8 binary interventions and encode each as a Network transform:

- close or filter a street: drop both twin links;
- reallocate road space: `with_scaled_capacity`;
- add a crossing: a change to J_ped, plus an optional capacity factor.

Before enumeration, remove any candidate that a bound clause forbids, so illegal designs are never
generated. Record each exclusion with its clause. Depends on ALG-MVP-4, ALG-S1-3.

### [ALG-S2-2] Scenario set and scenario reduction

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah

**Description:** Scenarios are demand multipliers (low, normal, peak, event) plus one closure,
e.g. Johnston Terrace. Reduce them with
`stochastic_optimisation.scenario_reduction(scenarios, None, k)`, scaling the components so
they're comparable. Keep the resulting probabilities. Depends on ALG-S1-4.

### [ALG-S2-3] Exhaustive robust search over designs x scenarios

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah

**Description:** For every candidate mask (at most 2^8) and every scenario:

- transform the network;
- run `user_equilibrium(method="b")` and check `converged`;
- J_car = TSTT;
- J_ped from `crossing_delay`, using the UE flows.

Per design, compute mean J_car and J_risk = `conditional_value_at_risk(losses, alpha=0.9)`. The
default alpha is 0.95, so pass 0.9 explicitly. The function only does equal weights, so add
weighting in glue if you use the reduced probabilities.

Checkpoint so the run can resume, and run it overnight. Fallback if too slow: k=6, or 4
scenarios. This is a glue stand-in for the held network_design. Depends on ALG-S2-1, ALG-S2-2,
ALG-S1-5, ALG-S1-7.

### [ALG-S2-4] Exact Pareto front, hypervolume and the three designs

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah

**Description:**

1. Run `multi_objective.find_pareto_front(F)` over (mean J_car, J_ped, J_risk). It returns
   indices, not vectors.
2. Normalise the objectives, then compute `hypervolume(front, ref)`.
3. `knee_point` takes only 2 objectives, so find the knee on the J_car vs J_risk projection.
4. Select the two extremes plus the knee, and output their design masks and diffs.

The plan's `pareto_set` lives in the held policy_compiler; this replaces it. Depends on ALG-S2-3.

### [ALG-S2-5] Tail-risk comparison: mean-optimal vs CVaR-optimal

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah, Matthew Li, Adish Boodhun

**Description:** From a single search run, compare how much the mean-optimal design and the
CVaR-optimal design each lose in the worst 10% of scenarios: "mean-optimal loses X%, the CVaR
design loses Y%". Depends on ALG-S2-3.

### [ALG-S2-6] Policy check per design (glue)

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah

**Description:** For each design x clause, compute satisfied / in_tension / abstained, with the
clause's quote and page. Report it as "consistent with N verified clauses; M abstained for
review". Never say "legally compliant". This is a glue stand-in for the held
`policy_compiler.policy_check` (see ALG-S3-2). Depends on ALG-MVP-4, ALG-S2-4.

### [ALG-S2-7] Retrain and re-evaluate on the full dataset

**Assignees:** Matthew Li, Adish Boodhun

**Description:** Re-run features, model, GroupKFold and time-split validation on all the collected
data, then freeze the final numbers. Depends on ALG-MVP-6, ALG-S1-1.

## Stage 3 -- ORCHID swap-ins and stretch algorithms

### [ALG-S3-1] ORCHID: policy_compiler confirmation bug-hunt round

**Assignees:** Unassigned (ORCHID maintainers)

**Description:** Run one confirmation round on the closed grammar. If it finds 0 template bugs:

1. Move policy_compiler from `HELD` to `URBAN_READY` in `scripts/sync_orchid.py`.
2. Add its import to `engine/orchid.py`.
3. Re-sync and run `--check`.

### [ALG-S3-2] Swap rule binding and policy check to ORCHID policy_compiler

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah

**Description:** Blocked until ALG-S3-1.

- Build a `Gazetteer` from `GazetteerEntry` items made from OSM names.
- Build each `PolicyClause` with `make_rulings`; all six operations are required.
- Call `compile_policy(clauses, gazetteer, sources={doc_id: text})` with no instruments. It
  verifies the quotes.
- Add `AbstentionItem(cid, "unrepresentable", ...)` for each clause the grammar refuses. This is
  the "advert abstains" finding.
- `CompiledFixings.to_dict()` replaces the ALG-S2-1 filter. REMOVE_EDGE PROHIBIT means "may not
  be closed".
- `policy_check` replaces ALG-S2-6.

Done when the counts match the glue, or the differences are explained.

### [ALG-S3-3] ORCHID: converge public_voice

**Assignees:** Unassigned (ORCHID maintainers)

**Description:** Keep running bug-hunt rounds until one finds 0 template bugs, then vendor the
module.

### [ALG-S3-4] Swap the concern field to ORCHID public_voice (FDR hotspots)

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah

**Description:** Blocked until ALG-S3-3.

1. `ingest_concerns(raw, salt=..., strict=False)`. It needs projected x/y, so geocode
   street-named items first.
2. `classify_theme`, then `find_near_duplicates`.
3. `concern_field` with an explicit 100-150 m bandwidth.
4. `local_getis_ord`, then Benjamini-Hochberg FDR, then `hotspot_clusters`.
5. Count comments per hotspot.

Done when the hotspots that survive FDR are reported.

### [ALG-S3-5] J_voice objective in the search

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah

**Description:** Compute J_voice, either with `voice_objective` / `voice_improvement` from
public_voice, or with glue for `J_voice(x) = sum_j sum_e s_e^j m_e^j(x)` using ALG-S1-8. Add it as a
fourth objective in ALG-S2-4. Drop this first if we're behind. Depends on ALG-S2-3, ALG-S1-8.

### [ALG-S3-6] ORCHID: converge network_design

**Assignees:** Unassigned (ORCHID maintainers)

**Description:** Keep running bug-hunt rounds until one finds 0 template bugs, then vendor the
module.

### [ALG-S3-7] Cross-check the search with ORCHID network_design

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah

**Description:** Blocked until ALG-S3-6. Exhaustive `robust_ndp` on a sub-case must reproduce the
glue optimum. Use `genetic_algorithm` when k > 8. `DesignProblem` can only add projects, so encode
closures and filters as projects. Depends on ALG-S2-3.

### [ALG-S3-8] Prediction intervals

**Assignees:** Matthew Li, Adish Boodhun

**Description:** Build intervals in one of two ways:

- split-conformal, from the GroupKFold out-of-fold residuals; or
- sklearn/LightGBM quantile loss.

The ORCHID GBR has no quantile loss. Report interval coverage on held-out streets. Depends on
ALG-MVP-6.

### [ALG-S3-9] Emissions estimates for the three designs (display only)

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah

**Description:** For each edge, call
`emissions_environment.fleet_hot_emissions(FleetComposition(...), "NOx" or "CO2", speed_kmh,
length_km, vehicles=...)` using each design's UE flows and speeds. This is display only and not
part of the optimiser. Depends on ALG-S2-4.

## Stage 4 -- after the hackathon (ORCHID)

### [ALG-S4-1] ORCHID: fix the green_min_cut_tree adapter for the fixed core gomory_hu_tree

**Assignees:** Unassigned (ORCHID maintainers)

**Description:** The fixed core can return a parent greater than its child. The adapter
(L5047-5052) assumes parent < child, which corrupts `.parent` and `.cut_value`. Fix it, and add a
regression test comparing `engine='core'` with `'local'`. Then un-hold green_infrastructure.

### [ALG-S4-2] ORCHID: fix core probabilistic erf and the digamma accuracy doc

**Assignees:** Unassigned (ORCHID maintainers)

**Description:** `erf = gammainc(0.5, x*x)` overflows to nan above about 1e154 (URBAN_HANDOVER
section 5). The README's digamma accuracy claim contradicts the measured 7.6e-10. Fix both, run
fix rounds, then un-hold the module.

### [ALG-S4-3] ORCHID: converge resilience

**Assignees:** Unassigned (ORCHID maintainers)

**Description:** Keep running bug-hunt rounds until one finds 0 template bugs.

### [ALG-S4-4] ORCHID: zero-bug confirmation round for the 9 baseline urban modules

**Assignees:** Unassigned (ORCHID maintainers)

**Description:** No zero-bug round is on record since these modules' single nine-axis campaign.
That is the risk we accepted when vendoring them.

### [ALG-S4-5] ORCHID: build urban_optimisation and mosaic_pipeline

**Assignees:** Unassigned (ORCHID maintainers)

**Description:** These are Waves B-D in URBAN_PICKUP.md. They would replace the team's ALG-S2-3
search glue.

### [ALG-S4-6] Re-sync vendor/ to the converged ORCHID and re-run every algorithm

**Assignees:** Matthew Miezaniec, Darryl Liu Liong Wah, Matthew Li, Adish Boodhun

**Description:** After the Stage 3-4 ORCHID work:

1. Move the fixed modules to `URBAN_READY`.
2. Re-sync and run `--check`.
3. Re-run every algorithm.
4. Compare the new numbers with the glue versions.
