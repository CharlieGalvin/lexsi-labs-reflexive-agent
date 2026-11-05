# Review PR: plan for once #37, #38 and #39 are merged

**Branch:** `charliegalvin/review-4nov`. **Written:** Wed 4 Nov, for Thu 5 Nov.
**Delete this file in the PR's last commit.** It's a working plan, not documentation.

## Already done on the branch (4 commits)

- [x] `.gitignore`: everything in `results/` is ignored except `.gitkeep` and `*.metrics.json`
  (the "Claude outputs" leak)
- [x] `student_review.py`: prints the totals (56 unknown-tool calls, 56 "not found" lookups), not
  only the 8 commonest
- [x] `teacher_probe.py` and `teacher_check.py` moved to `scripts/archive/` (with a README);
  `setup_github_project.sh` deleted. Their `just` commands still work.
- [x] BDD specifications: `tests/features/splits.feature` and `golden_trajectories.feature`
  (10 scenarios), typed steps, `just features`, `pytest-bdd` in the dev extra

## 0. Merge order (Charlie)

1. ~~**#38**~~ merged 5 Nov.
2. ~~**#37**~~ merged 5 Nov.
3. **#39** (docs), then **#40** (Chiara's classifier). **Before merging it**, Claude updates #39 for #37:
   - add `arms.py` and `scripts/rollout.py` to the new README repo-layout block;
   - put Chiara's Arm A row in the "Training and evaluation runs" table of `docs/EXPERIMENTS.md`.

Then run the SEL-48 upload (`just golden-publish`) and hand over to Katie. That's separate from
this PR and shouldn't wait for it.

## 1. Bring the branch up to date

- [ ] `git merge main` into `charliegalvin/review-4nov`
- [ ] Resolve `pyproject.toml`: the dev extra needs `huggingface_hub` (#38) **and** `pytest-bdd`
  (this branch)
- [ ] `just check` passes before any refactor starts

## 2. Capture "before" outputs (the refactor's regression check)

The refactor must not change any result. Save each output **before** touching code, then diff
**after**:

```bash
mkdir -p /tmp/before
just teacher-fixes --report            > /tmp/before/fixes_report.txt
just bakeoff-report                    > /tmp/before/bakeoff.txt
python scripts/phase2_report.py        > /tmp/before/phase2.txt
just student-review                    > /tmp/before/student.txt
just golden                            > /tmp/before/golden.txt
sha256sum data/golden/v1/*.jsonl       > /tmp/before/golden.sha
```

## 3. Restructure `scripts/` (with `git mv`, so history follows each file)

```text
scripts/
  tau2/       inspect, first_run.sh, report, cli, check_scorer, episode_check
  teacher/    bakeoff (was teacher_sample), fixes (was teacher_fixes), bakeoff_report, bakeoff_plot, phase2_report
  student/    review (was student_review)
  arms/       rollout (Chiara's, from #37)
  data/       make_splits, build_golden, publish_golden
  archive/    teacher_probe, teacher_check (done)
  README.md   one line per script: what it does, its `just` command, whether it costs money
```

- [ ] **Every `just` command keeps its name.** Only the paths inside the `justfile` change.
- [ ] Update the paths in docs, the README and docstrings
  (`grep -rn "scripts/" README.md docs src scripts tests justfile`)

## 4. Split `teacher_sample.py` (735 lines) into a library and a short CLI

- [ ] **`src/reflexive/runner.py`** (library): `NativeAgent`, `OpenRouterAgent`, `AyaAgent`,
  `make_agent`, `run_one`, `SLATE`/`VARIANTS`, credit checks, `openrouter_info`, `_broken`,
  `wilson`
- [ ] **`scripts/teacher/bakeoff.py`** (CLI): argument parsing, `main()`, printing
- [ ] `teacher_fixes`, `phase2_report`, `student_review` and the 3 test files import
  `reflexive.runner` **instead of loading scripts by file path** (`importlib` tricks)
- [ ] Remove the dead running paths (`tiny`, `aya8`, `aya-steps`, the old `qwen`/`gpt`
  reference models). **Keep their names readable by the reports**: the phase 1 records
  `sample-tiny.jsonl` and others must still load.
- [ ] Rename `TeacherClient` → `CohereClient` (it's now the *student's* client), and **keep
  `TeacherClient = CohereClient` as an alias**: Chiara's `arms.py` imports it.
- [ ] Fix stale docstrings: `AyaAgent` "the 32B teacher", `teacher_fixes` "train tasks not
  used in phase 1", `teacher_sample` "Is Aya a usable teacher?", the `just teacher-sample` help

## 5. Readability

- [ ] Break up the four most tangled functions (ruff `C901` complexity) into named steps, each
  under 15:
  - `teacher_fixes.main` (25);
  - `teacher_sample.main` (22);
  - `build_golden.main` (18);
  - `run_one` (14).
- [ ] Remove the 8 hand-written `sys.path.insert(...)` lines. Scripts rely on the installed
  package (check that `just setup-dev` installs `reflexive` in editable mode).
- [ ] `DEFAULT_MAX_ERRORS` in `episode.py`: unused (vulture). Either pass it to τ²-bench or
  delete it (after #37, which edits that block).

## 6. Type hints, then ty in CI

- [ ] Type hints on every function in `src/` and `scripts/`. Records are `dict[str, Any]`
  aliases (as in `tests/test_features_golden.py`); use dataclasses where a dict is really a
  fixed shape.
- [ ] `ty check src scripts tests` is clean. Today it reports 50 warnings, all false positives
  from missing hints; τ²-bench's `Optional` loaders need a narrowing `assert` or a `cast`.
- [ ] Add `ty` to CI, **pinned to one version** (it's beta, 0.0.x, so its rules change).

## 7. Formatting and quality gates

- [ ] `ruff format` the whole repo as **its own commit** (25 of 32 files change, style only)
- [ ] Add that commit's hash to `.git-blame-ignore-revs`, so `git blame` skips it
- [ ] CI: add `ruff format --check`, and the ruff complexity rule (`C901`, max 15) in
  `pyproject.toml`
- [ ] `just quality`: ty, vulture (dead code) and deptry (dependencies), run locally, not
  blocking. Optional: a `pre-commit` hook running ruff and ruff format.
- [ ] Declare `matplotlib` in a small `reports` extra (the bake-off charts use it; deptry
  found it undeclared)

## 8. Three more BDD feature files

- [ ] `agent_view.feature`: the agent never sees the customer's phone actions; the view survives
  the end of the conversation, including when the agent ends it (Chiara's fix in #37)
- [ ] `scoring.feature`: a pass means τ²-bench's own evaluator passed it; a broken run is never
  counted as a model failure
- [ ] `publishing.feature`: what `golden-publish` checks before upload; it refuses a public repo

## 8b. Follow-ups from the #38 review (5 Nov)

- [ ] Make `FakeApi` in `tests/test_publish_golden.py` keep tag state, like the real Hub
  (`create_tag` raises if the tag exists; `delete_tag` removes it), so the re-tag test proves
  the behaviour rather than the order of calls
- [ ] Tag the commit that was uploaded: `info = api.upload_folder(...)`, then
  `create_tag(..., revision=info.oid)`
- [ ] `--dry-run` reports whether `v1` already exists (today it returns before contacting the Hub)
- [ ] `--retag` help: after the hand-over, publish `v1.1`/`v2` instead of moving `v1`
- [ ] A half-written `.jsonl` line gives a "⛔" message, not a traceback (`load()` in `main`)
- [ ] `build_golden.py`'s console summary: "of 142 / of 36" derived, not typed
- [ ] Move `load`/`sha256` from `build_golden.py` into the package (the publish script imports a script)
- [ ] `Tau2Episode`'s default `max_context=8192` (the Tiny Aya leftover) → 128K, or make it required
  (a GRPO loop building episodes directly would cut telecom short)
- [ ] Data card (`publish_golden.card`, for v1.1): add the size facts now in `docs/DATA_CONTRACT.md`
  (file sizes, target tokens per record and in total, ~13% of each record is a target)
- [ ] Wait for #40 (Chiara's classifier): it adds `scripts/dump_tool_schemas.py`,
  `scripts/spot_check_hallucination.py` and `docs/experiments/HALLUCINATION_CLASSIFIER.md`.
  Move, type and format them with the rest, and add the page to `docs/experiments/README.md`
- [ ] `TeacherClient` records the API's `finish_reason`, so a reply cut by `max_tokens` is visible

## 9. Verify, then raise the PR

- [ ] `just check` (lint, format check, tests) and `just features`
- [ ] The tests also pass **without τ²-bench** (as in CI)
- [ ] Re-run step 2 into `/tmp/after/`, then `diff -r /tmp/before /tmp/after`. **Must be
  identical** apart from timings, and the golden `.jsonl` checksums must match exactly.
- [ ] `just splits-check`; `just golden` still reports 30/30, 127 of 142, 31 of 36
- [ ] Push, open the PR (list what changed and the before/after check), and link it on Linear
- [ ] Delete this file in the last commit

## Not in this PR

- No new model runs and no spend
- No changes to the data contract or the golden record format
- Chiara's and Katie's code is only touched to update import paths
