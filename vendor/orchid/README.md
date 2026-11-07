# vendor/orchid -- generated, do not edit

Verbatim copies of ORCHID (https://github.com/Orthant-Research-Group/orchid) at commit `156ca53033a18775afc669929220e3f035dda5f0`, written by `scripts/sync_orchid.py`.
Import them through `engine/orchid.py`, e.g. `from engine.orchid import street_network`.

To change what is vendored, edit the lists at the top of `scripts/sync_orchid.py`, re-run it,
then run `python scripts/sync_orchid.py --check`. `MANIFEST.json` records every module, why it
is here, what is held back and why, and the git blob id of every file.

| Module | Role | Why |
|---|---|---|
| `geospatial_graphs` | dependency | optional core import of vendored modules |
| `shortest_paths` | dependency | optional core import of vendored modules |
| `data_splitting` | core | plan s3: GroupKFold, TimeSeriesSplit |
| `ensembles` | core | plan s3: GradientBoostingRegressor fallback (no quantile loss) |
| `search` | dependency | optional core import of vendored modules |
| `similarity` | dependency | optional core import of vendored modules |
| `multi_objective` | core | plan s4: hypervolume, knee_point, find_pareto_front |
| `stochastic_optimisation` | core | plan s4: scenario_reduction, conditional_value_at_risk |
| `accessibility_equity` | urban | hardened; not called by the plan |
| `emissions_environment` | urban | plan s0: display-only emissions layer |
| `pedestrian_flow` | urban | plan s2: crossing_delay, fruin_los |
| `policy_compiler` | urban | plan s2 rules: compile_policy, verify_span, Gazetteer, AbstentionQueue, policy_check |
| `signal_control` | urban | hardened; not called by the plan |
| `street_network` | urban | plan s2: RoadGraph, HIGHWAY_DEFAULTS, clean_network, EdgeIndex, map_match, snap_point, to_geojson |
| `traffic_assignment` | urban | plan s2: user_equilibrium, detect_braess (suite pins Braess 83/498 -> 92/552) |
| `traffic_flow` | urban | hardened; not called by the plan |
| `travel_demand` | urban | plan s2: gravity_model, gls_matrix_estimation |

Held back (not copied):

- `probabilistic` -- held: documented, unfixed erf defect (URBAN_HANDOVER.md s5); emissions_environment and travel_demand fall back to their own stdlib code
- `green_infrastructure` -- held: green_min_cut_tree's adapter assumes parent < child, but core gomory_hu_tree (fixed in 07a4271) can return parent > child, so .parent/.cut_value are wrong whenever core network_optimisation is importable; the plan does not use green_infrastructure
- `network_design` -- pending: not converged, bug-hunt round 3 still found 4 template bugs (URBAN_PICKUP.md)
- `public_voice` -- pending: not converged, bug-hunt round 5 still found 5 template bugs (URBAN_PICKUP.md)
- `resilience` -- pending: not converged, bug-hunt round 7 still found 4 template bugs (URBAN_PICKUP.md)
