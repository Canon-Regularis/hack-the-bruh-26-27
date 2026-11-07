# Signal decisions log

Every question asked and the answer given, with the time. Read this at the start of every task.
Times are local (GMT) on Saturday 7 November 2026, unless stated.

## Standing instructions

| Time | From | Instruction |
|---|---|---|
| 16:59 | Adish | **Do not commit anything until told to.** This overrides kickoff rule 8 (commit at the end of every task and hourly). |
| 17:14 | Adish | Commit the workflow part and push. Read as covering the Task 0 setup only, on branch `signal`. Later work stays uncommitted until Adish says otherwise. |

## Questions

### Q1 (17:01, Task 0): where is the `orchid` library? ANSWERED

`orchid` is not on Adish's machine, in this repo or its history, or on the remote. Without it the signatures cannot be confirmed, and Task 3 (`map_match`) cannot start.

Options:

- **A (recommended):** whoever has `orchid` shares a local path or git URL. We run `python3 signal/scripts/check_orchid.py <dir>` and record the signatures.
- **B:** whoever has `orchid` runs that script on their own machine and pastes the output. This confirms the signatures, but Task 3 would then have to run on that machine.
- **C:** nobody can produce it. v2 then has no path for map matching, and the team has to decide what replaces it.

Answer (17:14, Adish): none of these for now. `orchid` will be set up later, so use a very abstract skeleton. A sub-issue already exists to replace it with the real library.

Done as `signal/orchid_api.py`. Every entry point raises `OrchidNotAvailable` when called and never returns a value. The signatures stay unconfirmed until the sub-issue lands.

### Q2 (17:15, Task 0): should Signal stay in `signal/` now that `main` has `src/traffic/`? OPEN, not blocking

PR 7 merged a team repo skeleton with empty placeholders under `src/`, `frontend/`, `tests/` and `data/`, including `src/traffic/`. The kickoff puts all Signal work in `signal/`, so that is what we are doing.

Options:

- **A (recommended):** stay in `signal/`, as the kickoff says. The Engine pair reads Signal's output as JSON, so nothing needs to move this weekend.
- **B:** move into `src/traffic/` now, while there is almost no code to move. `check.sh`, the import root and the `.gitignore` would all need changing.

Answer: waiting.

## Resolved without a question

| Time | Item | Resolution |
|---|---|---|
| 16:59 | Kickoff left the v2 path as `<PUT PATH HERE>` | Used `docs/HackTheBruh_Plan_v2.md`. It is identical to the copy in Downloads (checked with `diff`). |
