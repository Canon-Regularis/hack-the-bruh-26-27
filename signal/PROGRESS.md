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
