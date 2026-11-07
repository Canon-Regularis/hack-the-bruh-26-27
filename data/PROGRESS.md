# Signal progress

Stop reports, one per task, newest at the bottom.

### Task 0: Workflow setup and library check, BLOCKED

Blocked on the library check only. The workflow setup is finished. Tasks 1 and 2 do not need `orchid`, so they are not blocked.

**What was achieved:**
- Created the `signal/` layout, `check.sh`, `PROGRESS.md`, `DECISIONS.md`, `.gitignore` and `requirements.txt`.
- Could not confirm any `orchid` signature, because the library is not on this machine, in the repo or on the remote.
- Added `check_orchid.py`, so the confirmation takes one command once the library is available.

**Files created or changed:** all new, all under `signal/`, none committed.
- `.gitignore`, `pytest.ini`, `requirements.txt`, `PROGRESS.md`, `DECISIONS.md`
- `scripts/check.sh`, `scripts/check_orchid.py`, `tests/test_layout.py`
- Empty `__init__.py` in `collector/`, `matching/`, `speeds/`, `features/`, `model/`, `export/`
- `.gitkeep` in `data/raw/`, `data/derived/`, `data/sample/`, `reports/`

**How to run it:**
```
signal/scripts/check.sh
python3 signal/scripts/check_orchid.py [DIR_CONTAINING_THE_ORCHID_PACKAGE]
```

**Checks:**

`signal/scripts/check.sh` gives lint passed, 17 tests passed, 0 failed, smoke skipped (no pipeline stages yet), exit 0.

`check.sh` was also shown to fail when it should, using throwaway files that were then removed:

| Injected fault | Exit code |
|---|---|
| Lint error | 1 |
| Failing test | 1 |
| Failing smoke script | 3 |

`python3 signal/scripts/check_orchid.py`, actual output, exit 1:
```
MISSING MODULE street_network: tried ['orchid.algorithms.domains.urban.{}']
MISSING MODULE data_splitting: tried ['orchid.algorithms.core.ml.{}', 'orchid.core.ml.{}']
MISSING MODULE ensembles: tried ['orchid.algorithms.core.ml.{}', 'orchid.core.ml.{}']
MISSING MODULE traffic_assignment: tried ['orchid.algorithms.domains.urban.{}']
missing entry points: 9
```

`check_orchid.py` was tested against a fake library in the scratchpad, so it has not yet seen the real one. It exited 1 with one name removed and 0 with all nine present.

**Library check:**

| Entry point (v2 section 2) | Status | Signature |
|---|---|---|
| `street_network.RoadGraph` | Not confirmed | Library not available |
| `street_network.clean_network` | Not confirmed | Library not available |
| `street_network.EdgeIndex` | Not confirmed | Library not available |
| `street_network.map_match` | Not confirmed | Library not available |
| `street_network.snap_point` | Not confirmed | Library not available |
| `core/ml/data_splitting.GroupKFold` | Not confirmed | Library not available |
| `core/ml/data_splitting.TimeSeriesSplit` | Not confirmed | Library not available |
| `core/ml/ensembles.GradientBoostingRegressor` | Not confirmed | Library not available |
| `traffic_assignment.user_equilibrium` | Not confirmed | Library not available |

Places searched for `orchid`:
- Home folder, by folder name and by module filename (`street_network`, `traffic_assignment`, `data_splitting`, `policy_compiler`, `map_match`)
- Spotlight (`mdfind`)
- The active Python (`find_spec('orchid')` returns None)
- Every file ever in this repo's git history
- The remote: only `main`, at `c8038b1`
- The old white paper `.docx`, which never mentions it

The PyPI package `Orchid` 1.1 is a "Generic Multi Threaded Web Crawler", which is a different project. Do not install it.

**Engine adapter (OSM to RoadGraph):** not found. The repo and remote contain no code at all, only docs. It could still exist unpushed on an Engine machine.

**Review pass findings:** not required until Task 3.

**Numbers produced:**
- 17 tests passed: `signal/scripts/check.sh`
- 9 missing entry points: `python3 signal/scripts/check_orchid.py`

**Assumptions made:** none of these are v2 design choices, but please correct any that are wrong.
- `.gitignore` is at `signal/.gitignore`, not the repo root, to stay inside `signal/`.
- `signal/` is the import root and must never contain `__init__.py`. If it did, it would shadow Python's built-in `signal` module for anything run from the repo root. `tests/test_layout.py` guards this.
- The linter is `ruff`, because the kickoff only says "a linter" and it is already installed.
- `requirements.txt` pins the versions installed on this machine. `pip install --dry-run` confirms they are all satisfied.
- `check_orchid.py` was not in the task list. It is a small helper for this task's confirmation step.
- Its import paths are inferred from v2's wording and may need adjusting for the real library.

**Questions or blockers waiting on a person:**
- Q1 in `DECISIONS.md`: where is `orchid`? `git log` shows `dawilliucodes` committed v2, so that is the first person to ask.

**Risks noticed:**
- Task 3 cannot start without `orchid`, and v2 has no fallback for it. The `snap_point` fallback is in `orchid` too. Only the splits and gradient boosting have sklearn equivalents.
- `tmux` is not installed on this machine. `nohup` and `caffeinate` are, which is enough to run the collector overnight here.
- No commits means work exists only in this working tree until someone says otherwise.
- v2 section 3 words the question as "traffic speed", which breaks its own section 9 rule. This is left as written in v2. Our outputs will say "bus-observed speed".

**Next task per v2 (not started):** Task 1, TfE endpoint test and collector. T+0.5 is 17:10.

### Task 0 update (17:15): orchid skeleton added, DONE with the library check deferred

**What was achieved:**
- Adish confirmed `orchid` will be set up later and a sub-issue covers it, which clears the Task 0 blocker.
- Added `signal/orchid_api.py`, the one file Signal uses to reach `orchid`. Every entry point raises `OrchidNotAvailable` when called and never returns a value.
- Committed the Task 0 setup on branch `signal` and pushed it, as Adish asked.

**Files created or changed:** `orchid_api.py`, `tests/test_orchid_api.py`, `DECISIONS.md`, `PROGRESS.md`.

**How to run it:** `signal/scripts/check.sh`

**Checks:**
- `check.sh`: lint passed, 27 tests passed, 0 failed, smoke skipped (no pipeline stages yet), exit 0.
- `python3 signal/scripts/check_orchid.py`: still 9 missing entry points, exit 1. This is correct, because the skeleton must not pass as the real library.

**Numbers produced:**
- 27 tests passed: `signal/scripts/check.sh`
- 9 missing entry points: `python3 signal/scripts/check_orchid.py`

**Assumptions made:**
- `orchid_api.py` sits at the top of `signal/`, because the kickoff layout has no folder for it.
- "The workflow part" includes the skeleton, since the logs and tests depend on it.
- The go-ahead to commit covers this setup only. Later work stays uncommitted until Adish says otherwise.

**Questions or blockers waiting on a person:**
- Q2 in `DECISIONS.md`, not blocking: stay in `signal/`, or move into `src/traffic/` from PR 7?

**Risks noticed:**
- The skeleton unblocks writing code, not producing results. Any step that calls `map_match` or `snap_point` stops with `OrchidNotAvailable`, so Task 3 cannot pass its synthetic test or produce edge speeds until the sub-issue lands. The kill-switch deadline for map matching is 19:40.
- All 9 signatures are still unconfirmed, so code written against them may need changes after the swap.

**Next task per v2 (not started):** Task 1, TfE endpoint test and collector. T+0.5 (17:10) has passed.

### Task 3, speeds part (owner: Adish), BLOCKED

Waiting on the answers the kickoff requires before building. Nothing has been built. Matthew Li owns the matching half of this task.

**What was achieved:**
- Adish clarified that this work covers Adish's tasks only. Task 1 belongs to Matthew Li, so it was undone.
- Checked what the speeds stage needs from other people.
- Logged the two questions the kickoff requires, plus the RoadGraph question, as Q3 to Q5 in `DECISIONS.md`.

**Files created or changed:** `DECISIONS.md`, `PROGRESS.md`. None committed.

**How to run it:** nothing to run yet.

**Checks:** `signal/scripts/check.sh`: lint passed, 27 tests passed, 0 failed, smoke skipped (no pipeline stages yet), exit 0. No new tests, because the kickoff says to ask before building.

**Review pass findings:** no pipeline code exists yet, so there are no stages to compare.

**What the speeds stage needs:**

| Needed | Owner | Status at 17:40 |
|---|---|---|
| RoadGraph of the agreed study area | Engine | Not on the remote. `src/graph/` holds only a placeholder. |
| Real `orchid` (`map_match`) | Sub-issue | Not available. `orchid_api.py` raises an error. |
| Matched traces | Matthew Li | Not started as far as the remote shows |
| Agreed study area and fixed sample | Matthew Li (Task 2), then the team | Not started as far as the remote shows |

**Numbers produced:** none.

**Assumptions made:** none.

**Questions or blockers waiting on a person:** Q3, Q4 and Q5 in `DECISIONS.md`.

**Risks noticed:**
- The kickoff's full synthetic test adds GPS noise, which needs real map matching. Until `map_match` exists, a synthetic test can only cover the speeds stage.
- Thresholds for dwell depend on the ping interval, which is unknown until the bus data source is settled.

**Next task per v2 (not started):** the Task 3 speeds build once Q3 to Q5 are answered, then Task 4.

### Layout update (18:00): Signal work moved into `data/`

Adish asked for `data/` to hold what `signal/` held. Older entries above still say `signal/`, so read those paths as `data/`. `signal/data/raw`, `derived` and `sample` are now `data/raw`, `data/derived` and `data/sample`. Run checks with `data/scripts/check.sh`. Q3 to Q6 are answered in `DECISIONS.md`.

### Task 3, speeds part (owner: Adish), DONE on synthetic data

Built and tested on synthetic rows only. Real speeds are blocked on Matthew Li's matching, the real `orchid`, the RoadGraph and a bus data source.

**What was achieved:**
- Added the hand-off table `PingPair` and a `match_pings` skeleton that raises until Matthew Li's code replaces it.
- Added the dwell rule from Q3: moved under 10 m and within 20 m of a stop.
- Added the edge-speed table, using total metres over total seconds from Q4, with entry-time 15-minute bins in UTC.

**Files created or changed:** `matching/table.py`, `speeds/dwell.py`, `speeds/edge_speeds.py`, `speeds/synthetic.py`, `scripts/synthetic_speed_check.py`, `tests/test_edge_speeds.py`, `PROGRESS.md`. None committed.

**How to run it:**
```
data/scripts/check.sh
python3 data/scripts/synthetic_speed_check.py
```

**Checks:** `data/scripts/check.sh`: lint passed, 59 passed, 0 failed, smoke skipped (only one working stage), exit 0.

Synthetic check, true speed 18.00 km/h, stop on `edge_3`:

| Case | Edges without the stop | Edge with the stop |
|---|---|---|
| Dwell lines up with pings, no noise | 18.00 km/h | 18.00 km/h |
| Dwell falls between pings, 2 m GPS noise | 18.00 km/h | 12.00 km/h (300 m in 90 s) |

**Proposed tolerance:** within 1% on edges without a stop. On an edge with a stop, up to one ping interval of dwell can remain, so the test allows 12 to 18 km/h in this case.

**Not done, because the inputs do not exist:** the run on the fixed sample, time per 1,000 pings, GeoJSON traces, the full GPS-noise test through `map_match`, and fetching OSM stops, which needs the agreed study area.

**Review pass findings:**
- Units: metres and seconds throughout, converted to km/h once at the end. No problems.
- Timestamps: `PingPair` accepted any timezone. Fixed: it now rejects anything not in UTC, with a test. Its docstring says times are GPS fix times, not poll times.
- Edge IDs: an edge missing from the street lookup now raises. Matching must use the same IDs.
- Bin boundaries: one function, `bin_start`, sets every bin. Later stages must reuse it.

**Numbers produced:** all synthetic, from `python3 data/scripts/synthetic_speed_check.py`, plus 59 passed from `data/scripts/check.sh`.

**Assumptions made:**
- `PingPair` is our assumed format until Matthew Li's code exists (Q5).
- A pair counts as near a stop if either ping is within 20 m.
- Consecutive pairs join only if one starts exactly when the last ended, so a missing ping starts a new traversal.

**Questions or blockers waiting on a person:** Matthew Li to confirm the `PingPair` format when matching is ready.

**Risks noticed:**
- Edges with stops can read low. When a dwell falls between pings, part of it sits inside pairs where the bus also moved, so the rule cannot remove it. Here that took the stop's edge from 18 to 12 km/h.
- A dwell shorter than the ping interval never gives a stationary pair, so none of it is removed.
- The 10 m rule is per ping pair, so its meaning depends on the ping interval, which is unknown until the data source is chosen.

**Next task per v2 (not started):** Task 4, feature table and leakage firewall.

### Task 3 speeds update (18:15): no ping pairs dropped

Adish asked to keep every ping pair. Dwell pairs now stay in the output, and the table has a new `dwell_seconds` column. The speed still leaves dwell out, and `total_seconds + dwell_seconds` gives the full time. A row with only dwell has an empty speed, which Task 4 must handle. `data/scripts/check.sh`: lint passed, 61 passed, 0 failed, smoke skipped, exit 0. Not committed.

### Task 3 speeds review (18:40): 4 bugs fixed, design decisions checked

`data/scripts/check.sh`: lint passed, 64 tests passed, 0 failed, smoke skipped, exit 0. Each fix below had a test that failed first.

**Bugs fixed:**
- A bus that waited at a stop and then moved was binned at arrival time, so a 40-minute wait put its speed in the wrong bin. It now bins by when the bus first moved.
- A bus seen only dwelling counted as a traversal and a vehicle. Both counts now cover moving buses only.
- One vehicle with two source labels could merge into one traversal, so `synthetic` and `observed` could mix. Sources now stay apart.
- Negative edge distances were accepted. `PingPair` now rejects them.

**Decisions checked and kept:** total metres over total seconds; time shared by distance within a pair; dwell metres left out along with dwell time, so distance and time come from the same pairs; a gap in pings starts a new traversal; bins by entry time; an unknown edge raises an error.

**Risks left open, none changed without a decision:**
- GPS noise weakens the 10 m rule. Share of stationary pairs recognised as dwell, from `python3 data/scripts/synthetic_speed_check.py` (synthetic, 200 runs each): 2 m noise 100.0%, 5 m noise 79.2%, 10 m noise 34.5%. A missed dwell counts as moving time, so the edge reads low. Map-matched positions should help, but that is untested.
- A parked bus away from a stop, such as at a depot or layover, counts as queuing. With 0 metres and long time, it can pull a row's speed close to 0.
- A 20 m radius also catches buses queuing at lights near a stop, or opposite a stop across the road.
- A long dwell partway along an edge keeps both moving parts in the entry bin.

**Status:** the Task 3 speeds code is complete on synthetic data. The real-data checks wait on matching, `orchid`, the RoadGraph and a data source.
