# Signal decisions log

Every question asked and the answer given, with the time. Read this at the start of every task.
Times are local (GMT) on Saturday 7 November 2026, unless stated.

## Standing instructions

| Time | From | Instruction |
|---|---|---|
| 16:59 | Adish | **Do not commit anything until told to.** This overrides kickoff rule 8 (commit at the end of every task and hourly). |
| 17:14 | Adish | Commit the workflow part and push. Read as covering the Task 0 setup only, on branch `signal`. Later work stays uncommitted until Adish says otherwise. |
| 17:40 | Adish | **Only do tasks owned by Adish:** Task 0, the speeds half of Task 3, and Tasks 4, 5 and 7. Task 1 was started by mistake and undone, because Matthew Li owns it. |
| 18:15 | Adish | **Do not drop ping pairs.** Keep every pair. Dwell pairs stay in the output, with their time stored in `dwell_seconds`. The speed still leaves dwell out, as v2 requires. This changes how Q3 is applied, not its 20 m and 10 m rule. |

## Questions

### Q1 (17:01, Task 0): where is the `orchid` library? ANSWERED

`orchid` is not on Adish's machine, in this repo or its history, or on the remote. Without it the signatures cannot be confirmed, and Task 3 (`map_match`) cannot start.

Options:

- **A (recommended):** whoever has `orchid` shares a local path or git URL. We run `python3 signal/scripts/check_orchid.py <dir>` and record the signatures.
- **B:** whoever has `orchid` runs that script on their own machine and pastes the output. This confirms the signatures, but Task 3 would then have to run on that machine.
- **C:** nobody can produce it. v2 then has no path for map matching, and the team has to decide what replaces it.

Answer (17:14, Adish): none of these for now. `orchid` will be set up later, so use a very abstract skeleton. A sub-issue already exists to replace it with the real library.

Done as `signal/orchid_api.py`. Every entry point raises `OrchidNotAvailable` when called and never returns a value. The signatures stay unconfirmed until the sub-issue lands.

### Q2 (17:15, Task 0): should Signal stay in `signal/` now that `main` has `src/traffic/`? ANSWERED

PR 7 merged a team repo skeleton with empty placeholders under `src/`, `frontend/`, `tests/` and `data/`, including `src/traffic/`. The kickoff puts all Signal work in `signal/`, so that is what we are doing.

Options:

- **A (recommended):** stay in `signal/`, as the kickoff says. The Engine pair reads Signal's output as JSON, so nothing needs to move this weekend.
- **B:** move into `src/traffic/` now, while there is almost no code to move. `check.sh`, the import root and the `.gitignore` would all need changing.

Answer (17:18, Adish): stay in `signal/`.

### Q3 (17:41, Task 3 speeds): how is dwell stripped, and where do stop locations come from? ANSWERED

v2 says "strip dwell at stops" but not how. This question is required by the kickoff before building.

Options:

- **A (recommended):** drop a ping pair only when the bus was near-stationary close to a known stop. Stop locations come from OSM bus stop tags, which keeps us inside v2's data sources. Time queuing in traffic away from a stop stays in the speed.
- **B:** drop all near-stationary pings and use no stop list. This is the simplest option, but it also removes traffic queues, so speeds read too high.
- **C:** drop every ping pair whose path passes a stop. This removes dwell most fully, but loses a lot of data on bus corridors.

With A or B, a dwell shorter than the ping interval will not show as stationary, so some dwell stays in the speed.

Answer (18:00, Adish): A. Drop only time a bus is stationary near a stop, and keep queues at lights.

- "Near" means within 20 m of a stop.
- "Stationary" means moved less than 10 m between pings.
- Stops come from OSM. If the Engine OSM pull has no bus stops, the speeds stage fetches them from OSM itself, which stays within v2's data sources.
- Caveat: if the bus data is the 2014 to 2015 archive, OSM shows today's stops. Most main-corridor stops have not moved, but some have, for example along the tram routes.

### Q4 (17:41, Task 3 speeds): which target do we use? ANSWERED

v2 section 3 gives "mean bus traversal speed ... (or travel-time index vs free-flow)". This question is required by the kickoff before building.

Options:

- **A (recommended):** mean bus traversal speed, as the plain average of each traversal's speed. It matches the km/h metric in v2 and the output table in the kickoff.
- **B:** mean bus traversal speed, as total distance over total time. It weights slow traversals more, which suits the travel-time comparison in Task 7.
- **C:** travel-time index. This needs a definition of free-flow that v2 does not give, and MAE would no longer be in km/h.

Answer (18:00, Adish): B, total distance over total time, not A.

- Two buses on a 300 m street at 30 and 10 km/h take 36 s and 108 s. A gives 20 km/h. B gives 600 m / 144 s = 15 km/h, which matches the typical 72 s trip and what the Task 7 traffic model produces.
- B also protects very short edges: a 20 m edge crossed in an estimated 1 s reads as 72 km/h, but carries almost no weight under B.
- It is still a mean bus traversal speed, so it stays within v2's wording.
- Each output row also stores total metres and total seconds with the counts, so either version can be recomputed.

### Q5 (17:41, Task 3 speeds): can the speeds stage start before the RoadGraph, `orchid` and matching exist? ANSWERED

The kickoff says to stop and ask if the Engine adapter is not available. It is not.

Options:

- **A (recommended):** start now from an agreed input table. Each row is one ping pair from Matthew Li's matching: vehicle ID, start and end time in UTC, start and end position, the edges passed with the metres travelled on each, and a source label. Positions are needed to tell when a bus is near a stop. A lookup gives each edge's street name. Tests run on synthetic rows, labelled as synthetic. Matthew Li needs to agree this table.
- **B:** wait until real matched traces exist, then write the speeds stage against them.

With A, each traversal would go in the 15-minute bin of its entry time, in UTC, with bins starting at :00, :15, :30 and :45.

Answer (17:52, Adish): A, using a skeleton. Assume Matthew Li's matching code will arrive later, write the speeds code now, and Matthew Li fills in that part afterwards. The table format stays our assumption until then.

### Q6 (17:54): should Signal move into the team skeleton from PR 7? ANSWERED

Adish asked to use the team skeleton instead of `signal/`, which reopens Q2, and asked how to amend PR 15 (branch `signal`). `data/` on `main` holds only `raw/` and `out/`, so it cannot hold code. Nothing on `main` says which folder is Signal's. `src/traffic/` is the likely home for the code, but that is an inference.

Options:

- **A (recommended):** move everything: code to `src/traffic/`, tests to `tests/traffic/`, data to `traffic/` subfolders of `data/`, and logs to `docs/signal/`. Tried in a throwaway clone: lint passed, 27 tests passed, no `signal/` paths left in the PR diff, no conflicts with `main`.
- **B:** move only the data into `data/` and keep the code in `signal/`.

Answer (18:00, Adish): neither. Move the content of `signal/` into `data/`, which replaces Q2's answer. `signal/data/raw`, `derived` and `sample` became `data/raw`, `data/derived` and `data/sample`, so there is no `data/data/`. `data/out/` is left alone for the Engine pair. `data/` is now the import root.

## Resolved without a question

| Time | Item | Resolution |
|---|---|---|
| 16:59 | Kickoff left the v2 path as `<PUT PATH HERE>` | Used `docs/HackTheBruh_Plan_v2.md`. It is identical to the copy in Downloads (checked with `diff`). |
