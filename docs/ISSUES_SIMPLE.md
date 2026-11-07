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

GitHub issue numbers are in each heading, e.g. `(#18)`; headings without a number are tracked in the Signal issue named on their **Tracked in** line. This file mirrors the GitHub issues as of the plan update on 7 Nov (bustimes.org, dwell kept apart, speed = total metres / total seconds); the issues are the source of truth.

"Depends on" and "Unblocks" lines link GitHub issues directly, with the ALG ID in brackets.

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

**Tracked in:** PR #16 (merged)

### [ALG-MVP-1] Road graph: OSM to a cleaned RoadGraph with frozen edge ids (#18)

**Assignees:** Darryl Liu Liong Wah

**Description:** Build `load_graph()` in `src/graph/build_graph.py` using `street_network.RoadGraph`.

- `add_node` with lat/lon. The first node fixes the projection origin.
- `add_edge` with `oneway=False` for two-way ways (the default is `oneway=True`). Swap u/v for `oneway=-1`.
- Take lanes and speeds from `HIGHWAY_DEFAULTS`. Keep the way id, name and access tags.
- Run `clean_network`. Its simplify step removes crossing and signal nodes, so disable it or preserve those nodes.
- Freeze edge ids after cleaning, then build `EdgeIndex` once.

Every downstream algorithm keys on these edge ids: map matching, features, assignment and rules. RoadGraph mints int ids, so publish them as `str(eid)` everywhere and convert back with `int()` before calling `graph.edge()`. Freeze them once, on the generous OSM bbox. When the study bbox is fixed, crop with `graph.subgraph(...)`, which keeps ids. Rebuilding on a new bbox, or changing the build, would mint new ids and break Signal's matched pings.

**Depends on:** the OSM cache for a generous bbox (Engine, T+0-0.5). The study bbox, fixed from the first ping plot in #9, only crops the graph.

**Unblocks:** #19 (ALG-MVP-2), #21 (ALG-S1-2), #22 (ALG-S1-3), and Signal's critical path: #10 (map matching) and #3 (edge speeds on real data).

**Hand-off to Signal:** the frozen graph from `load_graph()`, which #10 passes to `map_match`, plus three JSON files. Signal uses this graph and does not build its own. Edge ids are `str(eid)`, the same ids as in `PingPair.edges` (`data/matching/table.py`). Anyone reading ids from a file must keep them as strings (pandas: `dtype={"edge_id": str}`), because `str(eid)` looks numeric. Scripts run from `data/` need the repo root on `sys.path` to import `load_graph()` (the root `pytest.ini` already adds it for tests), as #14 needs for `engine.orchid`.

- `street_names.json`: `{edge_id: name}` for every directed edge, covering both directions of a two-way street, with `""` for unnamed ways. `edge_speed_table` (`data/speeds/edge_speeds.py`) raises on a missing id. Don't build it from `to_geojson`, which lists only one id per two-way pair.
- `bus_stops.json`: `[[lat, lon], ...]`, latitude first, for `is_dwell` (`data/speeds/dwell.py`). Take every OSM node tagged `highway=bus_stop`, or `public_transport=platform` or `stop_position` with `bus=yes`. Put them in the OSM cache, because an osmnx street-graph download leaves out stops beside the road.
- `bbox.json`: the study bbox, once fixed from #9, as west, south, east, north in degrees. The collector (#8) then narrows its bustimes.org query to this box (`xmin`/`xmax` are longitudes, `ymin`/`ymax` latitudes); until then it polls its city-wide default. The generous OSM bbox must contain it.

**Plan update (7 Nov):**

- Live positions now come from bustimes.org (TfE is closed and BODS has no Lothian buses), one request per poll for the whole box. The study bbox is that query box, which is why it is shared.
- Dwell is kept apart, not stripped: a ping pair is dwell when the bus moved under 10 m and was within 20 m of an OSM bus stop. That is why Signal needs `bus_stops.json`.
- Speed is total metres / total seconds, so edge lengths must be true metres. Let `add_edge` compute them from lat/lon nodes, or pass `geometry` projected with `graph._project`, never raw lat/lon.
- On the DataShare 2014-15 fallback, today's OSM stops may have moved since then, so dwell near moved stops is less reliable.

### [ALG-MVP-2] Gazetteer: bind street names to edge ids (#19)

**Assignees:** Darryl Liu Liong Wah

**Description:** Build `src/graph/gazetteer.py`.

- Normalise OSM names (case, punctuation, abbreviations) but keep qualifiers (Upper, Lower, North...) exact.
- Map each name to its edge ids in both directions, using the string ids from #18 (ALG-MVP-1).
- Report any names that don't resolve.

It binds legal clauses and complaints to edges. Read names from `RoadEdge.street_name`, the same source as #18's `street_names.json`, so rules, complaints and Signal's speed table agree on street names. This is a glue stand-in for the held `policy_compiler.Gazetteer` (see #33 (ALG-S3-2)).

**Depends on:** #18 (ALG-MVP-1).

**Unblocks:** #20 (ALG-MVP-4), #25 (ALG-S1-8).

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

**Tracked in:** #10 (map matching) and #3 (dwell + edge-speed table)

### [ALG-MVP-4] Rule binding and OSM-mismatch detection (#20)

**Assignees:** Darryl Liu Liong Wah

**Description:** Input: ~15 hand-authored clauses. Each has a verbatim quote, page, streets, restriction type, times and exemptions.

1. Check that each quote is an exact substring of its cached source.
2. Mark any clause that can't be encoded honestly (e.g. "with exemptions") as abstained, with a reason.
3. Bind each clause to edges through the gazetteer.
4. Compare each bound restriction with the OSM access, motor_vehicle, oneway and highway=pedestrian tags.
5. Output per-edge restriction flags for the speed model.

Done when a script prints "N of M legal restrictions missing or wrong on OSM" (or that there are none), plus the list of abstained clauses. This is a glue stand-in for the held policy_compiler (see #33 (ALG-S3-2)).

Map matching, which this issue's old title named by mistake, is Signal's work and is tracked in #10 (edge speeds in #3).

**Depends on:** #19 (ALG-MVP-2).

**Unblocks:** #26 (ALG-S2-1), #31 (ALG-S2-6), and the restriction-flag feature in #4 (ALG-MVP-5).

**Hand-off to Signal:** the step 5 flags for #4 (ALG-MVP-5) as a CSV: `edge_id` (string, as in #18 (ALG-MVP-1)) plus one column per restriction type, with one row per directed edge, because one-way rules differ by direction.

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

**Tracked in:** #4

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

**Tracked in:** #4

### [ALG-MVP-7] Predict unobserved edges and quantify coverage

**Assignees:** Matthew Li, Adish Boodhun

**Description:** Fit on all observed data, then predict every edge x bin that has no bus
observation. Flag each value as observed or predicted. Compute: "X% of street-time in the study
area has no direct observation; we predict it with error Y", taking Y from ALG-MVP-6. Output
per-edge observed/predicted speeds for the rest of the pipeline. Depends on ALG-MVP-6.

**Tracked in:** #11

## Stage 1 -- validation and the traffic model

### [ALG-S1-1] Time-split validation ("next hour")

**Assignees:** Matthew Li, Adish Boodhun

**Description:** Run `data_splitting.TimeSeriesSplit` over the unique 15-min bins. It splits by
position, so split the list of bins and then expand back to rows. Report the time-split MAE and
skill next to the GroupKFold numbers. Confirm the leakage firewall holds: no feature uses a later
bin or the same street. Depends on ALG-MVP-6.

**Tracked in:** #4

### [ALG-S1-2] Match DfT count points to edges (#21)

**Assignees:** Darryl Liu Liong Wah

**Description:** Attach each DfT Edinburgh count point in or near the bbox to its nearest edge. `snap_point` takes projected metres, so convert lat/lon first, using either:

- `map_match` on the single point with `geographic=True`, or
- `graph._project`.

Then call `snap_point(..., index=idx)` and check each match by eye. Output AADF per matched edge, keyed by string edge id. DfT AADF is a two-way total, so record both directions of a two-way road (the edge and its `reverse_eid`). It is a model feature (#4 (ALG-MVP-5)) and the GLS target (#23 (ALG-S1-4)).

**Depends on:** #18 (ALG-MVP-1).

**Unblocks:** #23 (ALG-S1-4), #4 (ALG-MVP-5).

**Hand-off to Signal:** for the "DfT AADF where matched" feature in #4 (ALG-MVP-5), a CSV with `edge_id` (string, as in #18 (ALG-MVP-1)), `count_point_id` and `aadf`, one row per direction. `aadf` is the two-way total repeated on both rows, so never sum it across directions. Leave out edges with no count point rather than writing 0.

### [ALG-S1-3] RoadGraph to traffic_assignment Network adapter + Braess sanity check (#22)

**Assignees:** Darryl Liu Liong Wah

**Description:** ORCHID has no adapter, so write one:

- dense node ids 0..n-1;
- only car edges with capacity > 0;
- `net.add_link(u, v, BPR(...), length=<metres>, name=str(eid))` with free-flow time (s) and capacity (veh/h);
- edge-to-link maps for both directions, keyed by the string edge ids from #18 (ALG-MVP-1). Each link's `name` holds its id, so the maps can be rebuilt after `without_link`, which shifts later link indices.

Test: on the textbook Braess network, `detect_braess` must give 83/498 -> 92/552. Pass this before trusting any assignment number.

The seconds, metres and string ids here let #24 (ALG-S1-5) report UE results to Signal's #5 (ALG-S1-6) under the edge ids Signal uses.

**Depends on:** #18 (ALG-MVP-1).

**Unblocks:** #23 (ALG-S1-4), #24 (ALG-S1-5), #26 (ALG-S2-1).

### [ALG-S1-4] Demand estimation: gravity prior + GLS fit to counts (#23)

**Assignees:** Matthew Miezaniec

**Description:** Estimate the OD matrix:

1. Use at most 20-25 zones, because GLS is `O((n*m)^3)`. Give each zone its own centroid node.
2. Costs are free-flow times, all in one unit.
3. Build the prior with `travel_demand.gravity_model(productions, attractions, costs, ExponentialDeterrence(beta))`. beta is in inverse cost units.
4. Fit with `gls_matrix_estimation(prior, counts, AssignmentProportions(...))` against the matched counts from #21 (ALG-S1-2). A two-way AADF covers both directions, so give each count point one entry, named by its `count_point_id` (link names must be unique), whose proportions are the sum of its two edges' proportions.

Report modelled vs counted flows, and label the OD matrix as estimated. Keep trips in veh/h to match the BPR capacities from #22 (ALG-S1-3). DfT AADF is vehicles per day, so convert the counts to the modelled hour first and record which hour it is.

**Depends on:** #21 (ALG-S1-2), #22 (ALG-S1-3).

**Unblocks:** #24 (ALG-S1-5), #27 (ALG-S2-2).

### [ALG-S1-5] Baseline user equilibrium (#24)

**Assignees:** Darryl Liu Liong Wah

**Description:** Run `traffic_assignment.user_equilibrium(net, od, method="b")`. Avoid the Frank-Wolfe variants, which converge slowly at the end. Assert `converged` and report `relative_gap`. Output link flows, times and v/c for both directions, keyed by string edge id through the maps from #22 (ALG-S1-3).

**Depends on:** #22 (ALG-S1-3), #23 (ALG-S1-4).

**Unblocks:** #5 (ALG-S1-6), #45 (ALG-S1-7), #28 (ALG-S2-3).

**Hand-off to Signal:** for #5 (ALG-S1-6), a CSV with one row per directed car edge: `edge_id` (string, as in #18 (ALG-MVP-1)), `length_m`, `free_flow_time_s`, `ue_time_s`, `flow_veh_h`, `vc`. Leave out the centroid connectors from #23 (ALG-S1-4), which have no edge id. Say which hour the demand represents, so #5 compares it with the matching 15-min bins.

**Plan update (7 Nov):** the speed target is now total metres / total seconds per edge per 15-min bin (the travel-time index was dropped). #5 compares `length_m / ue_time_s` with `total_metres / total_seconds` from `edge_speed_table`, so times must be in seconds and lengths in metres.

### [ALG-S1-6] Validate modelled link times against bus-observed speeds

**Assignees:** Matthew Li, Adish Boodhun

**Description:** Compare UE link times with observed and predicted bus speeds on the same edges.
Report the MAE and bias as the simulator's measured error. State the caveat: buses are not cars
(stops, bus lanes). Depends on ALG-S1-5, ALG-MVP-7.

**Tracked in:** #5

### [ALG-S1-7] Pedestrian crossing delay and level of service (J_ped) (#45)

**Assignees:** Matthew Miezaniec

**Description:** Compute pedestrian crossing delay and level of service, giving the baseline J_ped that the design search (#28) minimises.

- Take crossing locations from the cached OSM: nodes tagged `highway=crossing`, with `crossing=traffic_signals` meaning signalised. `clean_network` in #18 (ALG-MVP-1) removes crossing nodes when it simplifies, so read them from the OSM cache, or keep them in the graph, and attach each to its string edge id.
- For each crossing, call `pedestrian_flow.crossing_delay(width_m, control="signalised"|"unsignalised", vehicle_flow=<UE veh/h>, ...)`. Take the vehicle flow from #24 (ALG-S1-5) by edge id, summing both directions for a two-way street. Signalised crossings also need `cycle` and `walk_green` (`walk_green <= cycle`, both in seconds). Note the British spelling of `control`.
- Call `pedestrian_flow.fruin_los_from_density(rho)` with assumed densities (P/m^2), labelled synthetic: the plan collects no pedestrian counts.
- Sum the delays to get the baseline J_ped, and expose a function that recomputes J_ped from a new set of UE flows, so #28 can call it per design and scenario.

**Depends on:** #18 (ALG-MVP-1), #24 (ALG-S1-5).

**Unblocks:** #26 (ALG-S2-1), whose crossing interventions change J_ped, and #28 (ALG-S2-3).

### [ALG-S1-8] Concern field per edge (glue) (#25)

**Assignees:** Darryl Liu Liong Wah

**Description:** Inputs are the published consultation records: theme, location or named street, and count. Compute each edge's intensity per theme with the plan's kernel:

`s_e^j = sum_i sev_i * K(dist(e, loc_i)) * 1[theme_i = j]`

If the comments can't be located, use counts per named street instead and label them that way. Never invent records. The output feeds the speed model (the concern-intensity feature in #4 (ALG-MVP-5)) and later J_voice (#36 (ALG-S3-5)). This is a glue stand-in for the held public_voice (see #35 (ALG-S3-4)).

**Depends on:** #19 (ALG-MVP-2).

**Unblocks:** #4 (ALG-MVP-5), #36 (ALG-S3-5).

**Hand-off to Signal:** for #4 (ALG-MVP-5), a CSV with one row per directed edge: `edge_id` (string, as in #18 (ALG-MVP-1)), one intensity column per theme, and a `basis` column saying whether the values come from located comments or per-street counts.

## Stage 2 -- the decision algorithms

### [ALG-S2-1] Candidate interventions and rule filter before enumeration (#26)

**Assignees:** Matthew Miezaniec

**Description:** Pick 6-8 binary interventions and encode each as a Network transform, keyed by the same string edge ids as #18 (ALG-MVP-1):

- close or filter a street: drop the edge and its `reverse_eid` twin with `net.without_link(i)`, using the edge-to-link maps. Each call shifts every later link index down by one: drop from the highest index first, and keep a link-to-edge-id list in step so UE results map back to edge ids;
- reallocate road space: `with_scaled_capacity`, taking the index from that same list;
- add a crossing: a change to J_ped, plus an optional capacity factor.

Before enumeration, remove any candidate that a bound clause forbids, so illegal designs are never generated. Record each exclusion with its clause.

**Depends on:** #20 (ALG-MVP-4) for the bound clauses, #22 (ALG-S1-3) for the Network and its edge-to-link maps, #45 (ALG-S1-7) for the crossing candidates.

**Unblocks:** #28 (ALG-S2-3). Later, #33 (ALG-S3-2) replaces the rule filter with `CompiledFixings.to_dict()`.

### [ALG-S2-2] Scenario set and scenario reduction (#27)

**Assignees:** Matthew Miezaniec

**Description:** Scenarios are demand multipliers (low, normal, peak, event) plus one closure, e.g. Johnston Terrace. Store the closure as its string edge ids, so #28 (ALG-S2-3) can drop them with the #26 (ALG-S2-1) close transform. Reduce the set with `stochastic_optimisation.scenario_reduction(scenarios, None, k)`, scaling the components so they're comparable (the default metric is Euclidean). Here k is the number of scenarios kept, not the number of interventions. Keep the returned `probabilities`: #28 (ALG-S2-3) weights the mean and the CVaR with them.

**Depends on:** #23 (ALG-S1-4) for the OD matrix, #19 (ALG-MVP-2) for the closure street's edge ids.

**Unblocks:** #28 (ALG-S2-3).

### [ALG-S2-3] Exhaustive robust search over designs x scenarios (#28)

**Assignees:** Darryl Liu Liong Wah

**Description:** For every candidate mask (at most 2^8) and every scenario:

- apply the #26 (ALG-S2-1) transforms for the mask plus the scenario's closure (drop the union of their edge ids once), and scale the OD by the scenario's demand multiplier;
- run `user_equilibrium(method="b")` and check `converged`;
- J_car = TSTT (`total_travel_time` on the result);
- J_ped from `crossing_delay`, using the UE flows looked up by edge id.

Per design, compute mean J_car and J_risk = `conditional_value_at_risk(losses, alpha=0.9)`. The default alpha is 0.95, so pass 0.9 explicitly. The function only does equal weights, so add weighting in glue if you use the reduced probabilities, and weight the mean the same way.

A closure that cuts a zone off makes `user_equilibrium` raise `TrafficAssignmentError` (destination unreachable). Catch it, mark that mask infeasible with the reason, and carry on.

Checkpoint each (mask, scenario) result so the run can resume, and run it overnight. Fallback if too slow: k=6 interventions, or 4 scenarios. This is a glue stand-in for the held network_design.

**Depends on:** #26 (ALG-S2-1), #27 (ALG-S2-2), #24 (ALG-S1-5), #45 (ALG-S1-7).

**Unblocks:** #29 (ALG-S2-4), #30 (ALG-S2-5), #36 (ALG-S3-5). #38 (ALG-S3-7) cross-checks the optimum with network_design once that is vendored.

### [ALG-S2-4] Exact Pareto front, hypervolume and the three designs (#29)

**Assignees:** Darryl Liu Liong Wah

**Description:**

1. Run `multi_objective.find_pareto_front(F)` over (mean J_car, J_ped, J_risk) for the feasible masks. It returns indices, not vectors.
2. Normalise the objectives, then compute `hypervolume(front, ref)`. Set ref just past the nadir (e.g. 1.1 on every axis): with ref at the nadir, any design that is worst on some objective adds no volume.
3. `knee_point` takes only 2 objectives, so find the knee on the J_car vs J_risk projection. Its chord runs between the lowest and highest J_car points, so first cut the projection down to its own front with `find_pareto_front`. It returns an index into the list you pass; map that back to the design.
4. Select the two extremes plus the knee, and output their design masks and diffs as string edge ids, plus each design's per-edge UE flows and link times in the #24 (ALG-S1-5) format.

The plan's `pareto_set` lives in the held policy_compiler; this replaces it.

**Depends on:** #28 (ALG-S2-3).

**Unblocks:** #31 (ALG-S2-6), #40 (ALG-S3-9), and the Pareto scatter and map diff on screen. #36 (ALG-S3-5) later adds J_voice here as a fourth objective.

### [ALG-S2-5] Tail-risk comparison: mean-optimal vs CVaR-optimal (#30)

**Assignees:** Darryl Liu Liong Wah

**Description:** From a single search run, compare how much the mean-optimal design and the CVaR-optimal design each lose in the worst 10% of scenarios: "mean-optimal loses X%, the CVaR design loses Y%". Use the same scenario weights as #28 (ALG-S2-3). Whenever the worst scenario carries at least 10% of the probability (always true with 10 or fewer equally likely scenarios), CVaR at alpha=0.9 is just that scenario's J_car, so say "in the worst scenario" unless the tail spans more than one.

**Depends on:** #28 (ALG-S2-3).

**Hand-off to Signal:** X, Y and the scenario count, written by the same run as a small JSON, for the results slide (#6).

### [ALG-S2-6] Policy check per design (glue) (#31)

**Assignees:** Matthew Miezaniec

**Description:** For each design x clause, compute satisfied / in_tension / abstained, with the clause's quote and page. Report it as "consistent with N verified clauses; M abstained for review". Never say "legally compliant". This is a glue stand-in for the held `policy_compiler.policy_check` (see #33 (ALG-S3-2)).

**Depends on:** #20 (ALG-MVP-4) for the bound clauses and the abstained list, #29 (ALG-S2-4) for the three designs.

**Unblocks:** the policy-check panel on screen. #33 (ALG-S3-2) later swaps in `policy_check` once #32 (ALG-S3-1) un-holds policy_compiler.

### [ALG-S2-7] Retrain and re-evaluate on the full dataset

**Assignees:** Matthew Li, Adish Boodhun

**Description:** Re-run features, model, GroupKFold and time-split validation on all the collected
data, then freeze the final numbers. Depends on ALG-MVP-6, ALG-S1-1.

**Tracked in:** #6

## Stage 3 -- ORCHID swap-ins and stretch algorithms

### [ALG-S3-1] ORCHID: policy_compiler confirmation bug-hunt round (#32)

**Assignees:** Matthew Miezaniec

**Description:** Run one confirmation round on the closed grammar. If it finds 0 template bugs, enable the module (same steps as README.md, "Held back"):

1. Move policy_compiler from `HELD` to `URBAN_READY` in `scripts/sync_orchid.py`.
2. Add its import to `engine/orchid.py` under its short name (`from algorithms.domains.urban.policy_compiler import policy_compiler_template as policy_compiler`; `--check` rejects any other alias), and add it to `__all__`.
3. Re-sync with `python scripts/sync_orchid.py --orchid ../orchid` and run `--check`. If the round lands on a later ORCHID commit, bump `ORCHID_COMMIT` too; that re-syncs every vendored module, so re-run the Braess check in #22 (ALG-S1-3).
4. Check a known answer before showing any number from it.

**Unblocks:** #33 (ALG-S3-2).

### [ALG-S3-2] Swap rule binding and policy check to ORCHID policy_compiler (#33)

**Assignees:** Matthew Miezaniec

**Description:** Blocked until #32 (ALG-S3-1).

- Build a `Gazetteer` from `GazetteerEntry` items made from OSM names; this replaces the glue in #19 (ALG-MVP-2). Use the published string edge ids from #18 (ALG-MVP-1): policy_compiler never coerces, so int eids raise.
- Build each `PolicyClause` with `make_rulings`; all six operations are required.
- Call `compile_policy(clauses, gazetteer, sources={doc_id: text})` with no instruments. It verifies the quotes.
- Add `AbstentionItem(cid, "unrepresentable", ...)` for each clause the grammar refuses. This is the "advert abstains" finding.
- `CompiledFixings.to_dict()` replaces the #26 (ALG-S2-1) filter. REMOVE_EDGE PROHIBIT means "may not be closed" (`fixedOne`).
- `policy_check` replaces #31 (ALG-S2-6). It takes a policy_compiler `Design`, so convert each #29 (ALG-S2-4) mask into one, keyed by the same string edge ids.

Done when the OSM-mismatch count (now from `CompiledFixings.to_dict()`), the abstained clauses and the policy-check rows match the glue in #20 (ALG-MVP-4) and #31 (ALG-S2-6), or the differences are explained.

**Depends on:** #32 (ALG-S3-1), #18 (ALG-MVP-1), #19 (ALG-MVP-2), #20 (ALG-MVP-4), #26 (ALG-S2-1), #29 (ALG-S2-4), #31 (ALG-S2-6).

**Hand-off to Signal:** the per-edge restriction flags that #20 (ALG-MVP-4) publishes for the features in #4 now come from the compiled fixings. Keep the same string edge ids and format, so #4 does not change.

### [ALG-S3-3] ORCHID: converge public_voice (#34)

**Assignees:** Matthew Miezaniec

**Description:** Keep running bug-hunt rounds until one finds 0 template bugs (at `a2e5e88`, round 5 still found 5), then vendor the module with the same steps as #32 (ALG-S3-1). Converging means new ORCHID commits, so bump `ORCHID_COMMIT`; that re-syncs every vendored module, so re-run `--check` and the Braess check in #22 (ALG-S1-3) before trusting any number.

**Unblocks:** #35 (ALG-S3-4), and the `voice_objective` route in #36 (ALG-S3-5).

### [ALG-S3-4] Swap the concern field to ORCHID public_voice (FDR hotspots) (#35)

**Assignees:** Matthew Miezaniec

**Description:** Blocked until #34 (ALG-S3-3). Replaces the glue concern field in #25 (ALG-S1-8).

1. `ingest_concerns(raw, salt=..., strict=False)`. It needs projected x/y, so geocode street-named items first, through the gazetteer from #19 (ALG-MVP-2) and in the RoadGraph's own projection.
2. `classify_theme`, then `find_near_duplicates`.
3. Build a `VoiceNetwork` from the #18 (ALG-MVP-1) RoadGraph, with string node and edge ids, projected x/y and length in metres. It is undirected, so add one edge per twin pair and copy its value to both twins.
4. `concern_field` with an explicit 100-150 m bandwidth.
5. `local_getis_ord` with Benjamini-Hochberg FDR (on by default, `fdr=True`, so do not adjust a second time), then `hotspot_clusters`.
6. Count comments per hotspot.

Done when the hotspots that survive FDR are reported.

**Depends on:** #34 (ALG-S3-3), #18 (ALG-MVP-1), #19 (ALG-MVP-2), #25 (ALG-S1-8).

**Unblocks:** #36 (ALG-S3-5), whose `voice_objective` route needs this field.

**Hand-off to Signal:** per-edge, per-theme concern intensity in the same format and string edge ids as #25 (ALG-S1-8), so the concern feature in #4 does not change.

### [ALG-S3-5] J_voice objective in the search (#36)

**Assignees:** Matthew Miezaniec

**Description:** Compute J_voice, either with public_voice's `voice_objective` / `voice_improvement` once #34 (ALG-S3-3) and #35 (ALG-S3-4) are done, or with glue for `J_voice(x) = sum_j sum_e s_e^j m_e^j(x)` using #25 (ALG-S1-8). Take m_e^j(x) from each design's UE and pedestrian outputs in #28 (ALG-S2-3), against the baseline from #24 (ALG-S1-5) and #45 (ALG-S1-7). `voice_objective` wants both as `{metric: {edge_id: value}}`, with a non-negative value for every field edge, keyed by the same string edge ids. Add it as a fourth objective in #29 (ALG-S2-4). Drop this first if we're behind.

**Depends on:** #28 (ALG-S2-3), #25 (ALG-S1-8), #24 (ALG-S1-5), #45 (ALG-S1-7); for the public_voice route, also #34 (ALG-S3-3) and #35 (ALG-S3-4).

### [ALG-S3-6] ORCHID: converge network_design (#37)

**Assignees:** Matthew Miezaniec

**Description:** Keep running bug-hunt rounds until one finds 0 template bugs (at `a2e5e88`, round 3 still found 4), then vendor the module with the same steps as #32 (ALG-S3-1). Converging means new ORCHID commits, so bump `ORCHID_COMMIT`; that re-syncs every vendored module, so re-run `--check` and the Braess check in #22 (ALG-S1-3) before trusting any number. network_design's own `braess_network("A")` is a ready known answer: 83/498 without the bridge, 92/552 with it.

**Unblocks:** #38 (ALG-S3-7).

### [ALG-S3-7] Cross-check the search with ORCHID network_design (#38)

**Assignees:** Matthew Miezaniec

**Description:** Blocked until #37 (ALG-S3-6). Exhaustive `robust_ndp` on a sub-case must reproduce the glue optimum from #28 (ALG-S2-3). Use `genetic_algorithm` when k > 8. `DesignProblem` can only add projects, so encode closures and filters as projects.

- It takes a `DesignNetwork`, not the #22 (ALG-S1-3) `Network`. Use the same string ids (`str(eid)` for links, the node ids as strings), so results map back to the glue masks and the #24 (ALG-S1-5) link outputs.
- `fixings` use the policy_compiler form (`{"fixedZero": [...], "fixedOne": [...]}`) but by project or existing-link id: unknown ids and `fixedZero` on an existing link raise, and `fixedOne` on one is ignored. Translate the #26 (ALG-S2-1) exclusions onto the closure and filter projects.
- Match the glue objective: `alpha=0.9`, `risk_weight` = lambda (0 for the mean-optimal design), and the #27 (ALG-S2-2) scenarios with the same probabilities the glue used (network_design weights its CVaR by them). Scenarios are demand maps only, so leave the closure scenario out of the sub-case. The objective has no J_ped.

**Depends on:** #37 (ALG-S3-6), #28 (ALG-S2-3), #26 (ALG-S2-1), #27 (ALG-S2-2).

### [ALG-S3-8] Prediction intervals (#39)

**Assignees:** Matthew Miezaniec

**Description:** Build intervals in one of two ways:

- split-conformal, from the GroupKFold out-of-fold residuals of #4 (ALG-MVP-6); or
- sklearn quantile loss (the pinned scikit-learn 1.5.2 has `GradientBoostingRegressor(loss="quantile")`). LightGBM is not in `data/requirements.txt` yet, so add it first if you want it.

The ORCHID GBR has no quantile loss. Report interval coverage on held-out streets, on `source == "observed"` rows only. The code sits in Signal's `data/` root, so take GroupKFold from `data/orchid_api.py` (a placeholder until #14) or use sklearn's.

**Plan update (7 Nov):** the target is now mean bus traversal speed on edge e in 15-min bin t, as total metres / total seconds; the travel-time-index alternative is dropped. Build the intervals on `mean_bus_observed_speed_kmh` from `data/speeds/edge_speeds.py` (#3), in km/h, with dwell left out of the speed. Rows with only dwell (`total_seconds == 0`) have an empty speed, so drop them first.

**Depends on:** #4 (ALG-MVP-6).

**Hand-off to Signal:** lower and upper bounds in km/h per string edge id and `bin_start_utc`, for the prediction JSON in #11 / #12 (ALG-MVP-7).

### [ALG-S3-9] Emissions estimates for the three designs (display only) (#40)

**Assignees:** Matthew Miezaniec

**Description:** For each edge, call `emissions_environment.fleet_hot_emissions(FleetComposition(...), "NOx" or "CO2", speed_kmh, distance_km, vehicles=...)` using each design's UE flows and speeds, plus the #24 (ALG-S1-5) baseline for comparison. UE outputs are keyed by string edge ids, with link time in seconds and length in metres, so `speed_kmh = 3.6 * length_m / time_s` and `distance_km = length_m / 1000`. With `vehicles` = the UE flow in veh/h, the result is grams per hour. Speeds outside a curve's range are clamped. This is display only and not part of the optimiser.

**Depends on:** #29 (ALG-S2-4) for the three designs, #24 (ALG-S1-5) for the UE outputs.

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

