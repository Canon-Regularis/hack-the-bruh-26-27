```text
Yakshdevsing (Adish) Boodhun
Darryl Liu Liong Wah
Matthew Li
Matthew Miezaniec
```

## ORCHID library (vendored)

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
  find more. Sanity-check every number from it before showing it.
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
