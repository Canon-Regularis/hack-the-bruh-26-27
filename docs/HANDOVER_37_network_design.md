# Handover: #37 (ALG-S3-6) converge ORCHID `network_design`, then vendor it

For an AI coding agent picking this up cold. State as of 2026-11-08. Read all of it before you run anything.

## 1. Goal and definition of done

Issue: <https://github.com/Canon-Regularis/hack-the-bruh-26-27/issues/37>. It unblocks #38 (ALG-S3-7).

1. **ORCHID side** (`../orchid`, i.e. `C:\Users\Matth\Projects\orchid`, GitHub `Orthant-Research-Group/orchid`).
   Run bug-hunt rounds on `algorithms/domains/urban/network_design/` until one round finds **0 template bugs**.
   That means converged.
2. **hack-the-bruh side** (this repo).
   - Re-pin `ORCHID_COMMIT` to the final ORCHID `main` merge.
   - If converged, vendor `network_design`.
   - Re-run `--check` and the Braess known answers.
   - Merge the PR.

### Decisions the owner (Matthew) has already made. Don't re-ask.

- **Round cap: rounds 4-8 only**, five rounds in total. Rounds 4 and 5 are done, so **6, 7 and 8 remain**. Never start a round 9.
- **If round 8 still finds bugs:**
  - Merge the fixes anyway and re-pin.
  - **Keep `network_design` in `HELD`**, with an updated reason giving the round history.
  - Leave #37 open: the PR says "Refs #37", not "Closes".
- **ORCHID round PRs:** open one PR per round and **merge it yourself** (`gh pr merge --merge`) once verification passes. The next round branches from the new `origin/main`.
- **The final hack-the-bruh PR:** **merge it yourself** after `--check` and the known answers pass and the full `pytest` is green.
- Hunt agents should use the strongest model available. The owner explicitly reverted a downgrade to a cheaper model.

## 2. Where things stand

### Round history

| round | template bugs | ORCHID PR | merge commit |
|---|---|---|---|
| 1-3 (earlier session) | 10, 5, 4 | in #1 | `a2e5e88` |
| 4 | 5 | #13 | `0fb59d3` |
| 5 | 16 (first round with the strongest model at every stage) | #14 | `9309f9d` = current `origin/main` |

- The suite has **300 tests**.
- The template is **5415/5500 lines** on `main`. The 5500 cap is hard.

### Round 6: started, interrupted

- **The ORCHID clone** is checked out on branch `fix/urban/network-design-round6`, cut from `origin/main` at `9309f9d`.
- **The one uncommitted change** is a condensation of `network_design_template.py`, from 5415 down to **5237 lines**. Its whole purpose is to make room for fixes.
- **It is NOT verified yet.** The session ended while its equivalence battery was running.
- **Before hunting, do one of these:**
  - **(a) Verify it.** Run the full suite from both locations (commands in s4). Then show the condensed file behaves identically to `git show origin/main:algorithms/domains/urban/network_design/network_design_template.py`. Load both under different module names and compare `repr` of results and of exception type and message, over a broad seeded battery that calls every name in `__all__`, including bad inputs. Aim for at least 1,000 calls. Also review `git diff` hunk by hunk. Look for changed defaults, dropped validation, a changed exception type, reordered tie-breaks, and a changed float summation order.
  - **(b) Discard it** with `git -C ../orchid checkout -- algorithms/domains/urban/network_design/network_design_template.py`, then condense afresh to about 5200-5250 lines.

### hack-the-bruh

- `origin/main` still pins ORCHID `81022c7`.
- `network_design` is in `HELD` in `scripts/sync_orchid.py`, with the reason "bug-hunt round 3 still found 4".
- Nothing for #37 has been committed here yet.

## 3. One hunt round: the procedure

The method is ORCHID's own. **Read these first:**

- `../orchid/URBAN_PICKUP.md` s3 (method) and s5 (the full shared brief, the BUG-HUNT protocol and the triage rule)
- `../orchid/URBAN_HANDOVER.md` s2 (conventions) and s5 (traps)
- `network_design/README.md`, "Testing", especially every "What the Nth hunt caught" paragraph (the fix history) and the "Conventions and limits recorded" lists

Hard rules for the module:

- **Language and encoding:** stdlib only, ASCII-only template, parses with `ast.parse(src, feature_version=(3, 8))`, and **5500 lines or fewer**.
- **Errors:** one `NetworkDesignError(ValueError)` for all bad input. No bare `TypeError`, `KeyError` and so on may escape.
- **Determinism:** seeds are explicit, ties break on `(value, id)`, and the builtin `hash()` of a str is never used.
- **Scale:** guards are relative, never an absolute epsilon against data-scaled values.
- **Files:** only the module's 5 files may change. Delete any `__pycache__`, `.benchmarks` or `.hypothesis` you create in the module dir.

Round 4 and 5 shape:

1. **Branch.** `git -C ../orchid fetch origin && git -C ../orchid switch -c fix/urban/network-design-roundN origin/main`.
2. **Snapshot.** Copy the module dir to a scratch location as the **pre-fix snapshot**. Every repro and every "fails on old code" check runs against it.
3. **Find.** Make independent passes, ideally separate fresh agents, one per lens:
   - **differential and identities:** UE/SO against scipy on the Beckmann programme and against the sibling `traffic_assignment`; exhaustive vs B&B vs every metaheuristic vs brute force; LP vs `scipy.optimize.linprog`; planarity, blocks, SCC and Dijkstra vs networkx; sensitivity vs finite differences
   - **scale and NaN/inf:** rescale money, flow, time and length by 1e+6 and 1e-6, alone and jointly; hunt absolute tolerances; check exact-rational budgets
   - **contract:** dead parameters, aliasing and mutation of caller inputs, determinism under `PYTHONHASHSEED` 0/1/1010, payloads through `json.dumps(allow_nan=False)`, a reflective fuzz for leaked exceptions
   - **adversarial:** zero-capacity closures everywhere, `second_best_tolls` from multiple starts, ties, degenerate networks
   - **regression:** re-attack every earlier fix, especially round 5's, and look for sibling instances of the same bug class

   Do not re-report anything the README already records as a convention or limit, unless you show it gives a silent wrong answer on realistic inputs.
4. **Verify.** Give each finding **two independent attempts to refute it** on the pre-fix snapshot.
   - A finding counts as a TEMPLATE_BUG only if both confirm it.
   - On a 1-1 split, get a third opinion.
   - Precedent: "a heuristic silently returns a design that violates a constraint the caller set" counts as a TEMPLATE_BUG.
   - Triage each finding as TEMPLATE_BUG / TEST_BUG / CONVENTION / INHERENT_LIMIT, per URBAN_PICKUP s3.
5. **Fix.**
   - For each TEMPLATE_BUG, fix the root cause and every sibling instance of it.
   - Add value-asserting tests in a new numbered section at the end of `_stress_test.py`, before the `__main__` block. Round 5 used sections 37-39, so round 6 starts at 40.
   - **Demonstrate each new test fails against the pre-fix snapshot.**
   - Never weaken an existing oracle.
   - Record CONVENTION and INHERENT_LIMIT findings in the README.
6. **Mutate.** Run about 25 fresh hand-made mutants against a copy of the module.
   - For each surviving mutant, either explain why it is equivalent or add a test that kills it, and verify the test kills it.
   - A survivor that shows the template itself is wrong is another TEMPLATE_BUG.
7. **Review.** Get an independent read of `git diff origin/main`: no weakened oracles, no banned test patterns (URBAN_PICKUP s5), and a README that matches the diff.
   **Watch for regressions the fixes themselves cause.** Round 5's VDF validation broke `VolumeDelay` subclasses that implement only the public methods, and needed fallback hooks.
8. **Docs.**
   - In `network_design/README.md`, add a "**What the Nth hunt caught**" paragraph in the existing style.
   - Update the test count in **three places**: the `# N oracle-based tests` line near the top, `**N tests**` under "## Testing", and the version footer.
   - In `../orchid/URBAN_PICKUP.md`, update the `network_design` table row (test count, append the round's bug count, converged?) and the line count in s2.
9. **Verify, commit, PR, merge.** Run s4. Then make two commits, matching the existing history:
   - `fix(urban): network_design bug-hunt round N (k template bugs)`. The body lists each bug, the mutation numbers, the suite growth and the line count.
   - `docs: record network_design hunt round N in URBAN_PICKUP`

   End both commit messages with the repo's usual `Co-Authored-By` trailer if you use one. Then push, `gh pr create --base main`, and `gh pr merge --merge`.
10. **Decide.** If k = 0, ORCHID has converged: go to s5. If k > 0 and N < 8, run round N+1. If N = 8, go to s5 in the not-converged variant.

Cost so far: one round is about 3-4.5M subagent tokens and 70-90 minutes when fanned out over about 60 agents. Most of that is the two-refuter verification.

## 4. Verification block (Windows, Git Bash)

```bash
cd ../orchid/algorithms/domains/urban/network_design
python network_design_template.py                      # last line: demo OK
python -c "import network_design_template as M; print([n for n in M.__all__ if not hasattr(M,n)])"   # []
python -c "import ast;s=open('network_design_template.py','rb').read();assert max(s)<128;ast.parse(s,feature_version=(3,8));print('ok')"
wc -l network_design_template.py                       # <= 5500
for s in "" 0 1 1010; do PYTHONHASHSEED=$s timeout 900 python -m pytest _stress_test.py -q -n 4 -p no:cacheprovider -rs | tail -1; done
cd ../../../..   # ORCHID repo root
for s in "" 1010; do PYTHONHASHSEED=$s timeout 900 python -m pytest algorithms/domains/urban/network_design/ -q -n 4 -p no:cacheprovider -rs | tail -1; done
find algorithms/domains/urban/network_design \( -name __pycache__ -o -name .benchmarks \) -exec rm -rf {} +
```

Every run must report **N passed, 0 failed, 0 skipped**. The full suite takes about 25 s with `-n 4`.

## 5. Vendoring into hack-the-bruh

This working tree may hold the owner's unrelated work on another branch, so **use a worktree**:

```bash
git -C C:/Users/Matth/Projects/hack-the-bruh-26-27 fetch origin
git -C C:/Users/Matth/Projects/hack-the-bruh-26-27 worktree add ../hack-the-bruh-nd -b feat/vendor-network-design origin/main
```

`../orchid` still resolves from `../hack-the-bruh-nd`.

### Changes

1. **`scripts/sync_orchid.py`**
   - Set `ORCHID_COMMIT` to the final ORCHID `main` merge SHA, with a comment naming the rounds.
   - **Converged:** move the `network_design` entry from `HELD` to `URBAN_READY`, e.g. `"plan s4 cross-check (#38): DesignNetwork, DesignProblem, exhaustive_ndp, robust_ndp, genetic_algorithm, braess_network"`.
   - **Not converged:** only rewrite its `HELD` reason, e.g. "pending: not converged, hunt rounds 1-8 found 10, 5, 4, 5, 16, ... template bugs (URBAN_PICKUP.md)".
2. **`engine/orchid.py`** (converged only)
   - Add `from algorithms.domains.urban.network_design import network_design_template as network_design  # noqa: E402`. `--check` rejects any other alias.
   - Add `"network_design"` to `__all__`.
3. **Re-sync and check:**

   ```bash
   python scripts/sync_orchid.py --orchid ../orchid
   python scripts/sync_orchid.py --check
   ```

   The re-pin re-syncs **every** vendored module. Read `git diff --stat vendor/` and confirm that nothing outside `network_design` changed unexpectedly. Between `81022c7` and `9309f9d` only `network_design` and `URBAN_PICKUP.md` changed. Never edit `vendor/` by hand.
4. **Known answers. Pass these before trusting any number.**
   - Existing: `pytest tests/decide/test_network.py::test_braess_textbook_numbers` (#22: `traffic_assignment` Braess 83/498 to 92/552).
   - Converged only: add a `network_design` test to `tests/decide/test_network.py`.
     - `braess_network("A", include_bridge=False)`: user equilibrium TSTT **498**, **83** per trip (demand 6).
     - `braess_network("A")`: TSTT **552**, every used path costs **92**, flows (2, 2, 2) on the three paths.
     - Note: `DesignProblem.evaluate` takes a **mask** (an int), not a design tuple. Use `pb.evaluate(pb.mask_of(design))`.
5. **`README.md`**
   - **Converged:** add a `network_design` row to the module table (`DesignNetwork`, `DesignProblem`, `robust_ndp`, `genetic_algorithm`, for the #38 cross-check), and remove it from "Held back".
   - **Not converged:** update its round count.
6. **Finish.**
   - Run the full `python -m pytest -q`.
   - Commit, push, and open the PR. Its body says "Closes #37" if converged, otherwise "Refs #37", and gives the round history and the check results.
   - **Merge it** (`gh pr merge --merge`).
   - Remove the worktree.

## 6. Pitfalls already hit

- **Line cap.** The template sits near its 5500 cap. Condense first, behaviour-preserving and proven, whenever there are fewer than about 150 lines of headroom.
- **Duplicate findings.** Several lenses report the same root cause. Merge duplicates before verification.
- **Settled items.** Many findings each round are conventions or limits already recorded in the README. Skip them unless there is new evidence.
- **Overreaching fixes.** "Fix every sibling" fixes can overreach. Round 5's `toll_sensitivity` now raises on inactive links in a group. That was accepted as a convention, but check that fixes don't break callers.
- **Sibling module.** The suite loads the sibling `traffic_assignment` by flat name for differential tests. Run from both the module dir and the repo root, because the `algorithms.*` core import is only available from the root.
- **Long runs.** Wrap long runs in `timeout`, and use `-p no:cacheprovider`.
- **Don't wander.** Don't touch other ORCHID modules, `URBAN_HANDOVER.md`, or `algorithms/core`.
