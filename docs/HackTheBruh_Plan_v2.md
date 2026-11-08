# HackTheBruh v2: Edinburgh's Invisible Streets

**HackTheBurgh XIII · Team Civilian\$ · Adish Boodhun · Darryl Liu Liong Wah · Matthew Li · Matthew Miezaniec**

> **One line:** Edinburgh's streets have three layers nobody can see — the **rules** (buried in traffic-order PDFs), the **complaints** (scattered across consultations) and the **actual traffic** (most streets have no counter). We find all three, **predict** the one that's missing, and **decide** with them: street designs that obey the real rules, answer real complaints, and still hold up in the worst 10% of days.

One pitch, three rubrics:

| Act | Urban track | G-Research ("make the invisible visible") | HRT ("data → predictions → decisions") |
|---|---|---|---|
| 1. Hidden | Real Old Town orders + consultations | *What* data is hidden | Diverse, unstructured inputs |
| 2. Found | Verified, cited, mapped | *How* we found it (incl. a tool we built to collect our own data) | Live data stream, real scale |
| 3. Predicted | Traffic on streets with no sensor | The invisible layer, filled in with an error bar | Out-of-sample accuracy vs a baseline |
| 4. Decided | Policy-safe designs, explained | *What we can do* with it | Robust decisions under tail risk |

---

## 0. What changed from v1 (be ruthless)

**The library already owns the maths.** `orchid/algorithms/domains/urban` has 13 hardened, tested modules (~3,500 tests) covering street graphs, map matching, assignment, network design, policy compiling and public voice. **We don't write algorithms this weekend. We write glue, data and a demo.**

**Not built in the library — so we don't depend on them:** `urban_optimisation`, `mosaic_pipeline`, `realtime_operations`, `microsimulation`, `transit`, `active_travel`, `safety`, `land_use`, `parking_curbside`, `geospatial_urban`. We replace the first two with ~300 lines of our own glue.

### Cut list

| v1 feature | Verdict | Why |
|---|---|---|
| LLM agent loop (observe / formalise / propose / critique / iterate) | **CUT** | No judge scores it; it's the riskiest part to demo. Replace with a scripted pipeline. |
| LLM reading of policy PDFs | **CUT** | We hand-author ~15 clauses for one district; `policy_compiler` verifies every quote character-for-character. Faster and more trustworthy. Hand-authoring time doubles as our "manual effort" number. |
| GPU / PyTorch / CuPy / CUDA | **CUT** | Library is pure Python; Old Town graph is a few hundred edges. Exhaustive search fits on a laptop overnight. |
| 2-D pedestrian density field, Hughes, social force | **CUT** | Use `pedestrian_flow` crossing delay + Fruin level of service only. |
| Signal optimisation, MPC / receding horizon (§8) | **CUT** | `realtime_operations` not built. "Real-time" = the live speed layer. |
| Node variables, junction redesign, green corridors, new links | **CUT** | Keep three operations: close/filter a street, reallocate road space, add a crossing. |
| Green connectivity λ₂ objective | **CUT** | Nobody on any panel scores it in an urban core. |
| Emissions objective | **DISPLAY ONLY** | One call to `emissions_environment` if time allows; not in the optimiser. |
| In-force amendment chains, conditional/temporal reconciliation | **CUT** (library has it; we don't spend time) | Advert summaries abstain; that is the honest answer. |
| Self-selection reweighting, representativeness | **CUT** | Needs population data we won't have. Show comment counts instead. |
| Future demand, roadworks, tourism scenarios, City Plan allocations | **CUT** | Scenario set = demand multipliers + one closure. |
| §5.5.5–5.5.6 compiler upgrades, §15 research extensions | **CUT** | One "next steps" line in the pitch, max. |
| Appendix A formulation table | **CUT** to the four equations we actually run (§6). |

### Kept, and why it wins

| Kept | Track it wins |
|---|---|
| **Bus-as-sensor collector + prediction of unobserved streets** (new) | HRT, G-Research |
| Policy compiler with abstention queue | Urban, G-Research |
| **OSM vs legal-order mismatch count** (headline finding) | G-Research |
| Public-voice concern field + FDR-controlled hotspots | Urban, G-Research |
| Exhaustive robust search → **exact** Pareto front, CVaR vs mean | HRT, Urban |
| "Numbers only come from code; nothing called legally approved" | All (credibility) |

---

## 1. Data (all real, all public)

| Layer | Source | Notes |
|---|---|---|
| Street graph | OpenStreetMap (osmnx/Overpass, cached once) | Converted into `street_network.RoadGraph` via `HIGHWAY_DEFAULTS` |
| **Bus positions (live)** | bustimes.org `GET https://bustimes.org/vehicles.json?xmin=&xmax=&ymin=&ymax=` | **Third-party, unauthenticated, no published terms found: ask the site owner before polling overnight.** Tested 7 Nov: 290 vehicles around Edinburgh, 244 of them Lothian, with coordinates, fix time, heading and service. One request per poll for the whole box, append raw JSON + parsed rows, and credit the source. |
| Bus positions (fallback) | Edinburgh DataShare: Lothian Buses full-fleet GPS traces 2014–15 (CC BY 4.0) | Historical, but real. Lose "live", keep the prediction story. OSM shows today's stops, so some may have moved since 2014–15. |
| Ruled out | TfE Open Data API; Bus Open Data Service (BODS) | TfE's docs say the API is closed. BODS had 25,625 vehicles nationally on 7 Nov, none from Lothian, and 8 around Edinburgh. |
| Vehicle volumes | DfT road traffic statistics: 156 Edinburgh count points, CSV | Calibrates demand on the boundary roads. |
| Rules | Lawnmarket ETRO/25/23 advert + council traffic-order pages | Cached as text with page numbers. |
| Complaints | Consultation Hub / Improving Old Town Streets published material | If no located comments exist: themes by named street, labelled as such. Never invent. |

**Study area:** Old Town + its approaches. Fix the bounding box **after** plotting the first hour of bus pings — the restricted streets will have little or no bus data; the signal lives on the surrounding roads (which is exactly where diverted traffic goes).

---

## 2. Pipeline → library mapping

| Step | Library call | Glue we write |
|---|---|---|
| Graph | `street_network.RoadGraph`, `clean_network`, `EdgeIndex` | OSM → RoadGraph adapter (~80 lines) |
| Bus pings → edge speeds | `street_network.map_match` (Newson–Krumm HMM, Viterbi) | Collector; per-vehicle ping sequences with repeated fixes removed; traversal speed per edge per 15-min bin; separate dwell at stops (moved under 10 m and within 20 m of an OSM bus stop): keep every ping pair, store dwell time apart, leave it out of the speed |
| Rules | `policy_compiler.compile_policy`, `verify_span`, `Gazetteer`, `parse_predicate`, `AbstentionQueue`, `policy_check` | Gazetteer from OSM street names; ~15 hand-authored `PolicyClause`s |
| OSM mismatch | `CompiledFixings.to_dict()` | Compare compiled prohibitions with OSM `access` / `motor_vehicle` tags on the bound edges |
| Complaints | `public_voice.ingest_concerns`, `classify_theme`, `find_near_duplicates`, `concern_field`, `local_getis_ord`, `hotspot_clusters` | Loader for published material |
| Prediction | `core/ml/data_splitting.GroupKFold`, `TimeSeriesSplit` (or sklearn equivalents) + gradient boosting (sklearn / LightGBM for speed; `core/ml/ensembles.GradientBoostingRegressor` as fallback) | Feature table, baselines, eval script |
| Demand | `travel_demand` gravity + `gls_matrix_estimation` against DfT counts | Count-point → edge matching |
| Vehicle response | `traffic_assignment.user_equilibrium`, `detect_braess` | Validate modelled link times against observed bus speeds |
| Search | `network_design.DesignProblem` + `exhaustive_ndp` / `robust_ndp` (scenarios + CVaR), honouring compiled fixings | Candidate list (6–8 interventions); scenario set |
| Pedestrians | `pedestrian_flow` crossing delay, Fruin LOS | Crossing locations |
| Public-voice objective | `public_voice.voice_objective`, `voice_improvement` | Theme → metric map from UE/ped outputs |
| Pareto + pick | `core/optimisation/multi_objective.hypervolume`, `knee_point` (or exact `pareto_set`) | Extremes + knee = 3 designs on screen |
| Map layers | `to_geojson`, `build_*_payload` | FastAPI serving precomputed JSON; MapLibre front end |

**Library caveats (from its own pickup notes):** `public_voice`, `network_design` and `resilience` did not fully converge in their last bug hunts. Mitigation: sanity-pin every result we show (e.g. `traffic_assignment`'s Braess constants: 83/498 → 92/552) and stick to the main entry points listed above.

---

## 3. The prediction (the HRT core)

| | |
|---|---|
| **Question** | What is the traffic speed on a street, right now, where no bus has been in the last 15 minutes? |
| **Target** | Mean bus traversal speed on edge *e* in 15-min bin *t*, as total metres ÷ total seconds |
| **Features** | Road class, lanes, length, limit (OSM) · edge betweenness (`street_network`) · restriction flags (from the compiled rules) · concern intensity per theme (from `public_voice`) · hour, weekday · spatial lag: recent speeds on neighbouring edges · DfT AADF where matched |
| **Baselines** | (1) mean by road class × hour; (2) nearest observed edge, same bin |
| **Validation** | **GroupKFold by street** = held-out streets are genuinely unobserved (the "invisible" claim). **Time split** (train early hours, test last hours) for the "next hour" claim. |
| **Leakage firewall** | Features use only *other* edges and *earlier* bins; group split guarantees no same-street leakage. One slide. |
| **Metric** | MAE (km/h) + skill = 1 − MAE / MAE_baseline. Prediction intervals (quantile loss) only if time. |
| **Caveat we say out loud** | Buses ≠ cars (stops, bus lanes). We report "bus-observed speed", strip dwell, and don't pretend otherwise. |

**Then it feeds the decision:** the assignment model's link times are checked against observed + predicted speeds. The simulator is no longer "synthetic, trust us"; it has a measured error.

---

## 4. The decision (all four tracks see it)

- **Candidates:** 6–8 binary interventions in the study area (closures/filters, road-space reallocation, new crossings), chosen after the first map plot. Compiled fixings remove anything the rules forbid **before** enumeration, so illegal designs are never generated.
- **Scenarios:** demand multipliers (low/normal/peak/event) + one closure (e.g. Johnston Terrace), reduced with `scenario_reduction`.
- **Search:** exhaustive over ≤ 2^8 designs × scenarios → the **exact** Pareto front, not a heuristic one. If too slow: k = 6, or `genetic_algorithm` with the exhaustive result on a sub-case as proof.
- **Objectives (all minimised):** J_car (total vehicle travel time), J_ped (crossing delay), J_voice (public voice), J_risk (CVaR₀.₉ of J_car).
- **On screen:** three designs (the two extremes + the knee) with a map diff, a policy check (`satisfied / in_tension / abstained` per clause, with quote) and the concern hotspots they address.
- **The HRT slide:** "The mean-optimal design loses X% in the worst 10% of days; the CVaR design loses Y%." Both numbers come from the same run.

---

## 5. The headline findings (G-Research)

1. **"N of M legal restrictions in our area are missing or wrong on OpenStreetMap"** — the maps everyone routes on don't match the law. *Check before claiming; if there are no mismatches, say that.*
2. **"Assembling the rules for 1 km² took us N person-hours"** — measured on ourselves while hand-authoring clauses, then scaled to the city.
3. **"The advertised ETRO summary abstains"** — "(with exemptions)" can't be parsed by the closed grammar, so the compiler refuses rather than guesses. The queue shows what is still hidden and why.
4. **"X% of street-time in the study area has no direct observation; we predict it with error Y"**.

---

## 6. The only equations we run

$$t_e(f_e) = t_e^0\left[1 + \alpha\left(\tfrac{f_e}{c_e}\right)^{\beta}\right] \qquad J_{\text{robust}}(x) = \mathbb{E}[J(x,D)] + \lambda\,\mathrm{CVaR}_{0.9}\big(J(x,D)\big)$$

$$s_e^j = \sum_i \mathrm{sev}_i\, K\big(\mathrm{dist}(e,\mathrm{loc}_i)\big)\,\mathbb{1}[\mathrm{theme}_i = j] \qquad J_{\text{voice}}(x) = \sum_j \sum_e s_e^j\, m_e^j(x)$$

---

## 7. Team and timeline

Pairs, per our rule (never solo). **T+0 was Sat 16:40**, so T+3 is 19:40, T+6 is 22:40 and T+9 is 01:40. Hacking ends Sun 12:00, we submit by 11:30, and feature freeze is 09:00.

| Pair | Owns |
|---|---|
| **Signal** — Matthew Li + Adish | Collector, map matching, edge speeds, features, model, validation, results slide |
| **Engine** — Matthew Miezaniec + Darryl | OSM graph, gazetteer, clauses, compiler, mismatch, concerns, demand, search, API, UI |

| When | Signal | Engine |
|---|---|---|
| **T+0 – 0.5** (both) | **Get the OK from bustimes.org. Start the collector on a machine that will stay awake all night.** Every hour lost here is data lost. | Pull + cache OSM for a generous bbox |
| T+0.5 – 3 | Ping → map match → edge-speed table on the first hour of data | OSM → RoadGraph; gazetteer from street names; fix bbox from first ping plot |
| T+3 – 6 | Features, two baselines, gradient boosting, GroupKFold + time split | ~15 clauses + compile + abstention queue + OSM mismatch; concern records + field + hotspots |
| T+6 – 9 | Prediction layer → JSON; check assignment link times vs observed speeds | Demand (gravity + GLS vs DfT); UE; candidate list; scenarios; **launch exhaustive robust run overnight** |
| T+9 – 12 | Sleep in shifts; collector keeps running | FastAPI over precomputed JSON; MapLibre: rules, heatmap, speed (observed solid / predicted hatched), Pareto scatter → map diff |
| Morning | Retrain on the full night's data; final numbers | Integrate results; policy-check panel |
| **Deadline − 3h** | Results slide, pitch, rehearse ×3 (no feature freeze) | Same |

### Signal tasks

Status at Sun 07:30, from the repo and the final run on 21:00 to 05:00 UTC. No more windows are coming, so these are the final numbers.

| Owner | Task | Target | Status |
|---|---|---|---|
| Matthew Li | Collector | 17:10 | Working: 44,063 pings plotted by 21:12, including repeated fixes, and a live test at 22:54 returned 203 vehicles. We have the OK for overnight polling. Asked to keep it running all night. |
| Matthew Li | Ping plot and 20-minute sample | 18:00 | Plot merged in PR #54. Speeds received for 21:00 to 05:00 UTC in 2-hour windows, all matched with the fixed matcher. That is the whole night: no morning windows. |
| Matthew Li | Map matching | 19:40 | Merged (#74), with the fixes for breaks, pairs above 80 km/h and backward drift under 10 m. `scripts/run_matching.py` and `scripts/run_matching_hourly.py` turn raw pings into edge-speed windows. |
| Matthew Li | Prediction JSON for the frontend | 01:40 | Covered by the prediction table and summary from `scripts/run_signal.py`, sent to Darryl as `speed_prediction.csv` and `speed_summary.json`. No separate JSON unless the frontend asks for one. |
| Adish | Repo setup and library check | T+0 | Done. |
| Adish | Dwell stripping and edge-speed table | 19:40 | Done. Final run on 21:00 to 05:00: 73,551 bus passes over 1,866 edges on 122 streets in 32 bins. Coverage: 81.9% of bus-road street-time in the study bbox (4,463 edges) has no bus observation. Slow rows: 0.9% of rows on edges within 20 m of a stop are under 2 km/h, against 0.4% elsewhere (median 14.1 vs 15.4 km/h), so dwell is not leaking in noticeably. Known limit: 2.1% of 03:00 to 05:00 rows read 50 km/h or more, mostly on The Mound, Waverley Bridge and Market Street, which looks like one matched route rather than real driving. |
| Adish | Features, leakage firewall, baselines, model, validation | 22:40 | Done (#61, #62, #63, #72, #76). Final, on 21:00 to 05:00 UTC bus-observed speeds (26,061 rows): MAE of model / baseline 1 / baseline 2 is 7.04 / 7.37 / 7.13 km/h on held-out streets and 6.47 / 7.58 / 8.98 km/h on later bins. The model beats both baselines on both splits, by a small margin on held-out streets. The leakage check passes (shuffled-target skill -0.005, tolerance 0.02). Conformal intervals (#78): plus or minus 15.3 km/h, measured coverage 0.896 against a 0.9 target (0.74 to 0.96 across the five street folds). Concern intensity is read from Engine's `data/concerns/concern_field.csv` once it exists. |
| Adish | Assignment vs observed speeds | 01:40 | Done, final. On 1,866 shared edges Engine's modelled speed is 35.3 km/h above the bus-observed speed on average (MAE 35.6); free-flow speed gives the same gap (+35.8), because the equilibrium adds little delay (median v/c 0.09). On 2,637 edges with predicted speeds only, the gap is +18.4 km/h. The equilibrium is for 08:00 to 09:00 and all bus data is night-time, so the result uses every bin and is labelled as an hour mismatch. Buses are not cars, but dwell is already stripped, so stops do not explain a gap this size: modelled link speeds are road-class speed-limit defaults. |
| Adish | Morning retrain, final numbers, results slide | Sun morning | Retrain and final numbers done (above), from one run of `scripts/run_signal.py` on the four windows. Results slide next. |

**Next for Adish:** the results slide. The final run is `scripts/run_signal.py --speeds` with the 21:00, 23:00, 01:00 and 03:00 windows; rerun it only if Engine's concern field or clauses file arrives.

**What Matthew Li needs from Adish:** nothing. Matching is merged (#74) and the frontend files are with Darryl.

**What Adish needs from Matthew Li:** the total ping count from `pings.csv`, for the "every bus is a sensor" line. Without it, the slide uses bus passes per edge (73,551), labelled as such.

Adish sent the OSM cache, `data/raw/osm.json`, directly because it is not in git. Everyone must build from this same file, because a fresh download can give different edge ids. Run `python -m src.graph.build_graph` from the repo root to build the graph from it. No code in the repo creates `osm.json`: it was downloaded once from the Overpass API on 7 Nov at 21:12 (SHA-256 starts `e6c3c0dfd255d4b3`), and the query was not saved. `src/graph/build_graph.py` only reads it, and it expects the query to use recursion (`>;`) so every road's nodes are included. Keep this file rather than downloading it again.

**Feature inputs:** DfT AADF comes from the City of Edinburgh CSV (local authority 29), saved as `data/raw/dft_aadf.csv` and matched to edges with `python -m src.graph.count_points`. 120 of 158 count points matched, and Engine should check `data/graph/dft_review.csv` by eye. Restriction flags need `data/rules/clauses.json`, which Engine writes by hand. In Engine's graph, lanes and speed limit are road-class defaults, because OSM `lanes` and `maxspeed` are not read, so they repeat road class.

**Model setup:** the speed model uses LightGBM 4.6.0, pinned in `data/requirements.txt`. On macOS it also needs `brew install libomp`. Concern intensity code is on `main` (#25) and writes `data/concerns/concern_field.csv` from `data/raw/concerns.csv`.

**Map matching speed:** on synthetic bus traces over the real graph, `map_match` took about 0.1 to 0.16 s per ping, roughly 4 hours per 100,000 pings on one core. It picked the right edge 89 to 92% of the time. Matching must remove repeated fixes, keep only pings in the study area, run buses in parallel and run in batches as data arrives. Timing it on the 20-minute sample will replace this estimate. Fallback: `snap_point` on bus corridors only.

**Data needed:** a 20-minute sample is enough to build and test map matching, and the first hour gives the first edge-speed table. The model needs the overnight data.

**Prediction JSON:** fit on all observed data, predict every edge and 15-minute bin with no bus observation, and flag each value as observed or predicted. It also gives the headline figure: X% of street-time has no direct observation, predicted with error Y. Signal and Engine still need to agree the JSON format, including how the map gets each street's geometry.

### Kill switches

| If… | By | Then |
|---|---|---|
| bustimes.org refuses, blocks us or goes down | As soon as it happens | DataShare 2014–15 traces. Drop "live", keep everything else. |
| Map matching too slow or wrong | T+3 | `snap_point` nearest-edge on bus corridors only |
| Model doesn't beat baseline 2 | T+9 | Report it honestly; ship baseline 2 + the leakage slide. Rigour still scores. |
| Exhaustive run too slow | T+8 | k = 6 or 4 scenarios; GA on the full set |
| No locatable complaints | T+5 | Themes by named street; J_voice shown, not optimised |
| Behind overall | T+10 | Drop J_voice from the search, drop UI polish. **Never drop the prediction or the mismatch finding.** |

---

## 8. Demo (5 minutes, every panel at once)

| Time | Beat | On screen |
|---|---|---|
| 0:00 | Hook: "Three layers of Edinburgh nobody can see." | Blank Old Town map |
| 0:30 | **Rules.** ETRO PDF → same rule on the map. Abstention queue. "N disagreements with OSM." | Restrictions layer + queue |
| 1:15 | **Complaints.** Heatmap; hotspots that survive FDR; comment counts. | Concern layer |
| 2:00 | **Traffic.** "Every bus is a sensor: X readings this weekend." Observed vs predicted streets. "Held-out error Y vs Z baseline." | Speed layer + one chart |
| 3:15 | **Decide.** Pareto scatter → click knee → map diff + policy check. "Mean-optimal loses X% on bad days; ours loses Y%." | Pareto + diff |
| 4:15 | Results slide; leakage firewall; "numbers only come from code; nothing here is legally approved." | One slide |
| 4:45 | Close: the one-liner. | |

---

## 9. Claims discipline (non-negotiable)

- Every number on a slide comes from a script output we can rerun. Label **observed / predicted / synthetic** everywhere.
- Never say "legally compliant". Say "consistent with N verified clauses; M abstained for review".
- An advertised summary is not the signed order; it abstains.
- "Bus-observed speed", never "traffic speed".
- No claim of calibrated city-wide forecasts.

**Sources:** [bustimes.org](https://bustimes.org/) · [TfE API closure notice](https://tfe-opendata.readme.io/) · [Lothian Buses GPS traces 2014–15](https://datashare.ed.ac.uk/handle/10283/2089) · [DfT counts, City of Edinburgh](https://roadtraffic.dft.gov.uk/local-authorities/29) · [ETRO/25/23 advert](https://www.edinburgh.gov.uk/downloads/file/38911/advert) · [Edinburgh traffic orders](https://www.edinburgh.gov.uk/trafficorders) · [Improving Old Town Streets](https://www.edinburgh.gov.uk/public-transport/improving-old-town-streets) · [Consultation Hub](https://consultationhub.edinburgh.gov.uk/)
