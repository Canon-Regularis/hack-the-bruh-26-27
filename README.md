<div align="center">

# Edinburgh's Invisible Streets

**The rules, the complaints and the traffic of Edinburgh's Old Town, put on one map and used to choose better street designs.**

HackTheBurgh XIII · Team Civilian$

![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![React 19](https://img.shields.io/badge/react-19-149ECA?logo=react&logoColor=white)
![MapLibre](https://img.shields.io/badge/map-MapLibre-396CB2)
![LightGBM](https://img.shields.io/badge/model-LightGBM-2E7D32)

</div>

---

Edinburgh's Old Town streets have three layers you can't see on a map: the **rules** (traffic orders buried in PDFs), the **complaints** (scattered across council consultations) and the **actual traffic** (most streets have no counter). We collect all three, fill in the missing traffic using live bus positions, and use the lot to pick street designs that follow the real rules, respond to real complaints and still hold up on bad days.

<p align="center">
  <img src="data/out/ping_plot.png" width="640" alt="Live bus pings collected across Edinburgh">
  <br>
  <sub>Bus pings from our live collector, plotted on the first evening.</sub>
</p>

## Contents

- [What it does](#what-it-does)
- [Current numbers](#current-numbers)
- [Quick start](#quick-start)
- [How it works: the maths](#how-it-works-the-maths)
- [Repo layout](#repo-layout)
- [ORCHID library](#orchid-library-vendored)
- [Team](#team)

## What it does

| | Data | What we do with it |
| --- | --- | --- |
| **Traffic** | Live bus positions from [bustimes.org](https://bustimes.org/) | Match each ping to its street, work out bus-observed speeds, then predict the streets no bus covered, with an error bar on each |
| **Rules** | [Edinburgh traffic orders](https://www.edinburgh.gov.uk/trafficorders) | Check every quote against the order, compare each rule with OpenStreetMap, and flag clauses the parser can't read instead of guessing |
| **Complaints** | [Council consultations](https://consultationhub.edinburgh.gov.uk/) | Spread comments onto the streets they mention and find where they cluster more than chance would explain |
| **Designs** | All of the above | Try every legal mix of closures, bus lanes and signalised crossings under several demand scenarios, keep the designs nothing else beats, and show four on the map |

The site (React + MapLibre) has one view for each of these.

## Current numbers

| | |
| --- | --- |
| Street-time with no bus reading | **66%** in the 08:00–08:45 window, 80% overnight |
| Speed model error | **6.9 km/h**, against 7.8 km/h for the best simple baseline |
| Prediction interval | **±14.1 km/h**, covering 90% of held-out readings |
| Rules missing or wrong on OpenStreetMap | **10 of 25** |
| Clauses flagged for review | **46** |
| Chosen designs against the rules | consistent with 16 verified clauses, in tension with 9 |

<sub>From the data the site ships with, in `frontend/public/data/`. Speeds are bus-observed, not general traffic speeds, and nothing here has been legally approved.</sub>

## Quick start

### 1. See the site

The site's data is committed, so it runs straight from a clone. You need Node.js.

```bash
cd frontend
npm install
npm run dev
```

### 2. Set up Python

Python 3.12 is what we tested on (3.10 or newer works).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r data/requirements.txt scipy networkx
```

> [!NOTE]
> On macOS, LightGBM also needs OpenMP: `brew install libomp`. Don't `pip install orchid`, it's an unrelated package. The ORCHID code we use is already in `vendor/orchid/`.

### 3. Get the raw data

None of this is in git.

| File | Where it comes from |
| --- | --- |
| `data/raw/osm.json` | The team's OpenStreetMap download. Get it from Adish rather than downloading it again, or the edge ids won't match. |
| `data/raw/dft_aadf.csv` | [DfT traffic counts](https://roadtraffic.dft.gov.uk/local-authorities/29) for the City of Edinburgh |
| `data/raw/concerns.csv` | Built by hand from published consultation material: theme, lat, lon, street, count, severity |
| `data/rules/clauses.json`<br>`data/raw/sources/<document_id>.txt` | Traffic-order clauses copied by hand, and the text of each order |
| `data/raw/pings.csv` | Written by the bus collector (step 4) |

### 4. Run the pipelines

<details>
<summary><b>Engine pipeline</b>: rules, complaints, demand and the design search</summary>

<br>

From the repo root, in this order:

```bash
python -m src.graph.build_graph              # OSM -> street graph
python -m src.graph.count_points             # DfT counts -> streets
python -m src.rules.bind_rules               # traffic orders -> rules, OSM mismatches
python -m src.rules.compile_rules            # the same rules through ORCHID's policy compiler
python -m src.concerns.concern_field         # complaints -> concern score per street
python -m src.concerns.hotspots              # complaint hotspots
python -m src.decide.od_matrix_estimation    # morning-peak trips, fitted to the counts
python -m src.decide.baseline_user_equilibrium
python -m src.decide.pedestrian_crossing_delay
python -m src.decide.scenarios
python -m src.decide.propose_candidates
python -m src.decide.interventions           # drops options the rules forbid
python -m src.decide.design_search           # every design under every scenario
python -m src.decide.best_trade_offs         # Pareto front, J_voice and the picks
python -m src.rules.policy_check
python -m src.decide.tail_risk_comparison
python -m src.decide.emissions               # display only
```

The design search saves its progress as it goes, so you can stop and restart it. To split it across machines, use `--shard K/N`, then join the parts with `--merge N`.

</details>

<details>
<summary><b>Signal pipeline</b>: live bus positions to street speeds</summary>

<br>

```bash
cd data && python collector.py collect --interval 30    # leave it running; writes raw/pings.csv
cd ..
python scripts/run_matching.py --start 2026-11-08T08:00 --end 2026-11-08T09:00 --out data/derived/speeds_0800.csv
PYTHONPATH=.:data python scripts/run_signal.py --speeds data/derived/speeds_0800.csv
```

`run_signal.py` joins the speed windows, scores the model against both baselines, predicts the streets with no bus data and writes a summary to `data/derived/`. On Windows, use `;` instead of `:` in `PYTHONPATH`.

</details>

### 5. Update the site and run the tests

```bash
python -m src.api.export     # writes frontend/public/data/ (only the street graph is required)
pytest                       # everything under tests/
cd frontend && npm test
```

The speed model tests need LightGBM installed.

## How it works: the maths

```mermaid
flowchart LR
    bus["Bus positions"] --> speeds["Map matching<br/>Speed prediction"] --> map["Speed map"]
    osm["OSM map + DfT counts"] --> traffic["Demand<br/>Traffic assignment<br/>Crossing delay"] --> search
    consult["Consultations"] --> concerns["Concern field<br/>and hotspots"] --> search
    orders["Traffic orders"] --> rules["Rule checking"] --> search
    search["Design search"] --> picks["Pareto front + J_voice<br/>four designs"]
```

| Step | Method | In one line |
| --- | --- | --- |
| Map matching | Hidden Markov Model + Viterbi | Picks the street each ping was really on; speed is metres ÷ seconds |
| Speed prediction | LightGBM + conformal intervals | Fills in streets with no bus, tested on streets and hours it never saw |
| Travel demand | Gravity model + GLS | Morning-peak trips between 20 zones, fitted to the DfT counts |
| Traffic assignment | User equilibrium, Algorithm B | Where every driver goes once a street changes |
| Crossing delay | Adams' delay, signal timing | How long pedestrians wait, given the traffic |
| Complaints | Gaussian kernel, binomial test, Benjamini–Hochberg | Where complaints cluster more than chance |
| Design search | Exhaustive search, CVaR | Every legal design under every scenario, including the worst days |
| Picking designs | Pareto front, hypervolume, knee | Keeps the designs nothing beats and picks four |
| Rule checking | Quote check, closed grammar | Turns traffic orders into rules, and never guesses |

### Key formulas

Congestion on each street (BPR curve):

```math
t_e(f_e) = t^{0}_e \left[ 1 + 0.15 \left( \frac{f_e}{c_e} \right)^{4} \right]
```

First guess at trips between zones (gravity model), before fitting to the counts:

```math
T^{0}_{ij} = k \, O_i \, D_j \, e^{-\beta c_{ij}}, \qquad \beta = \frac{1}{\bar{c}}
```

Pedestrian wait at an unmarked crossing (Adams):

```math
d = \frac{e^{q t_c} - q t_c - 1}{q}
```

Bad-day risk: the average of the worst 10% of scenarios (CVaR):

```math
\text{CVaR}_{0.9} = \text{VaR}_{0.9} + \frac{1}{0.1} \sum_{s} p_s \max\left(0, \; J_s - \text{VaR}_{0.9}\right)
```

How much a design changes conditions where people complained (J_voice):

```math
J_{\text{voice}}(x) = \sum_{j} \sum_{e} s_e^{j} \, m_e^{j}(x)
```

Every formula, with what each symbol means, is in **[docs/MATHEMATICS.md](docs/MATHEMATICS.md)**. Most steps use the vendored ORCHID library below; the speed model, hotspots and J_voice are built on LightGBM, numpy, scipy and networkx.

## Repo layout

```text
src/        Engine pipeline: graph, rules, concerns, decide, and the export for the site
data/       Signal pipeline: bus collector, map matching, speeds, features, speed model
frontend/   the site (React, Vite, MapLibre)
engine/     the one import point for ORCHID
vendor/     the vendored ORCHID library (don't edit by hand)
scripts/    runners for the Signal pipeline, plus sync_orchid.py
tests/      pytest suite
docs/       the plan, the issue list and the maths
```

## ORCHID library (vendored)

<details>
<summary>Which modules we vendor, which are held back, and the gotchas</summary>

The maths comes from ORCHID: 14 modules copied byte for byte into `vendor/orchid/` from ORCHID
commit `81022c7`, which is pinned in `scripts/sync_orchid.py`. They use only the standard
library and need Python 3.8+, so there is nothing to install and no ORCHID clone to set up.
Import them through one wrapper, and run code from the repo root (e.g.
`python -m src.graph.build_graph`):

```python
from engine.orchid import street_network as sn, traffic_assignment as ta
```

| Module | Plan step | Key functions |
| --- | --- | --- |
| `street_network` | §2 graph, map matching, map layers; §3 features | `RoadGraph`, `HIGHWAY_DEFAULTS`, `clean_network`, `EdgeIndex`, `map_match`, `snap_point`, `to_geojson`, `edge_betweenness_centrality` |
| `traffic_assignment` | §2 vehicle response | `user_equilibrium`, `detect_braess` |
| `travel_demand` | §2 demand | `gravity_model`, `gls_matrix_estimation` |
| `pedestrian_flow` | §2 pedestrians | `crossing_delay`, `fruin_los` |
| `public_voice` | §2 complaints; §4 J_voice (**not converged**, see below) | `ingest_concerns`, `classify_theme`, `find_near_duplicates`, `concern_field`, `local_getis_ord`, `hotspot_clusters`, `voice_objective`, `voice_improvement` |
| `emissions_environment` | display only | `network_emission_proxy` |
| `data_splitting` | §3 validation | `GroupKFold`, `TimeSeriesSplit` |
| `ensembles` | §3 model fallback | `GradientBoostingRegressor` (no quantile loss) |
| `multi_objective` | §4 Pareto + pick | `find_pareto_front`, `hypervolume`, `knee_point` |
| `stochastic_optimisation` | §4 scenarios + risk | `scenario_reduction`, `conditional_value_at_risk` |
| `policy_compiler` | §2 rules, OSM mismatch, policy check | `PolicyClause`, `Gazetteer`, `compile_policy`, `verify_span`, `policy_check`, `pareto_set` |
| `traffic_flow`, `accessibility_equity`, `signal_control` | not used by the plan | |

Each module's API docs are in its `README.md` under `vendor/orchid/algorithms/`. The test
counts in those READMEs are out of date.

**Held back.** These raise `ImportError`, with the reason, if you import them from
`engine.orchid`:

- `network_design`, `resilience`: ORCHID's bug hunts on them haven't converged.
- `green_infrastructure`: a min-cut-tree bug against the current core.
- core `probabilistic`: a known erf defect.

**Vendored before convergence.** `public_voice` and `policy_compiler` are enabled although
their hunts haven't converged:

- `public_voice`: rounds 6 and 7 found and fixed 5 and 4 template bugs, so a later round may
  find more. Sanity-check every number from it before showing it. Our concern hotspots (#35,
  `src/concerns/hotspots.py`) and J_voice (#36, `src/decide/voice_objective.py`) don't use it;
  they are built on numpy, scipy and networkx.
- `policy_compiler`: see the caveat below.

To enable one once it's fixed:

1. Move its line from `HELD` to `URBAN_READY` in `scripts/sync_orchid.py`.
2. Add its import to `engine/orchid.py`.
3. Run `python scripts/sync_orchid.py --orchid ../orchid`.
4. Check a known answer (e.g. Braess 83/498 -> 92/552) before showing any number from it.

**`policy_compiler` caveat.** It is enabled with hunt rounds 6-8 merged (ORCHID #4, #6, #7), but its
hunts haven't converged. Almost every bug found was in `classify_wording`, which reads
English wording for the `wording_ambiguous` check. Set each clause's `strength` ("hard" or
"soft") by hand rather than trusting the wording reader, and check the abstention queue by eye.

**Rules.** Never edit `vendor/`. Never put an ORCHID checkout on `sys.path` / `PYTHONPATH`.
Import only through `engine.orchid`. `python scripts/sync_orchid.py --check` verifies the
copy.

**Gotchas.**

- Plan §2's `pareto_set` lives in `policy_compiler`. `multi_objective.find_pareto_front` +
  `knee_point` work too.
- There is no `RoadGraph` -> `traffic_assignment` network adapter, so we write that glue
  ourselves.
- `street_network`'s node-graph routers ignore banned turns. Use `edge_expanded_graph` /
  `turn_aware_shortest_path` for turn bans from traffic orders.

</details>

## Team

Yakshdevsing (Adish) Boodhun · Darryl Liu Liong Wah · Matthew Li · Matthew Miezaniec
