# Golden trajectories v1 (4–5 Nov 2026, SEL-36, SEL-9, SEL-48)

> **Status: published 5 Nov** as a private Hugging Face dataset,
> [`reagent-lexsi/tau2-golden-trajectories`](https://huggingface.co/datasets/reagent-lexsi/tau2-golden-trajectories), tag `v1`, and handed over to the Trainer
> and Architect. The hard subset follows as `v1.1` (SEL-60).

**What the Curator hands the Trainer:** 127 conversations in which the teacher, GLM-5.3-Flash,
solved a τ²-bench **train** task, verified by τ²-bench's own evaluator. They're formatted
exactly as Aya Expanse 32B (the student) sees a conversation, with its turns marked as the
learning targets. There are also 31 dev conversations, for reference only and never trained on.
The format is fixed by the [data contract v1](../DATA_CONTRACT.md), agreed on 4 Nov.

**Rebuild it yourself (no API calls, ~10 s).** The raw run files are backed up in the same
dataset under `raw/` (team members only):

```bash
hf download reagent-lexsi/tau2-golden-trajectories --repo-type dataset --include "raw/fixes/*" --local-dir hf_raw
mkdir -p results/fixes && cp hf_raw/raw/fixes/*.jsonl results/fixes/
just golden --sample 5      # expect 30/30 · 127 of 142 · 31 of 36
```

## From teacher run to dataset

| Stage | Count | Where |
|---|---:|---|
| τ²-bench train tasks | 178 | official `split_tasks.json` |
| Teacher conversations (GLM-5.3-Flash, judges on) | 217 | `results/fixes/glm53f.jsonl` (phase 2) |
| Clean passes (passed, no retried failed call, no unknown tool) | 193 | 16 failed a check, 8 hit the step limit |
| **One per task:** the first clean pass | **158** | 20 tasks never passed cleanly |
| → **train** (`golden_train.jsonl`) | **127 of 142** | retail 53/59, airline 23/24, telecom 51/59 |
| → **dev reference** (`golden_dev_reference.jsonl`) | **31 of 36** | retail 11/15, airline 5/6, telecom 15/15 |
| → test | 0 | no test task was ever run by the teacher |

## The four steps (`scripts/data/build_golden.py`, `src/reflexive/golden.py`)

Each step stops the build if it fails.

1. **Split by task.** `configs/splits.json` (SEL-36): τ²-bench's test split unchanged, and a
   dev set of 15 retail, 6 airline and 15 telecom drawn at random, with a fixed seed, from the
   official train split. `just splits-check` re-draws it and confirms the file. Any test task
   in the input stops the build (`splits.assert_not_test`).
2. **The agent's view.** In telecom the customer operates their own phone. Those actions and
   their results are in the saved conversation, but the agent never saw them, so they're
   removed (`golden.agent_view`). τ²-bench's system prompt is rebuilt offline and put first.
   **Check:** on 30 Aya conversations that saved what the agent really saw, the rebuilt view
   matches it **exactly in 30/30**.
3. **Pick and convert.** The first clean pass per task (`golden.pick`) is converted to
   Cohere's `Action:` format with the same `teacher.to_prompted` that built Aya's baseline
   prompt, so training and evaluation prompts are identical. `train_on` lists every
   assistant turn.
4. **Read it back.** `golden.check_record` parses each converted record back and requires
   exactly the original turns, tool calls, arguments and tool results, alternating roles,
   `train_on` = every assistant turn, and reward 1.0. **All 158 pass.**
   `tests/test_golden.py` breaks a record in 8 ways and checks that each is caught.

## Publishing (`just golden-publish`, SEL-48)

`scripts/data/publish_golden.py` refuses to upload unless all of these hold:

- every record is well formed (a malformed record is reported, never a crash), IDs are
  unique, each file holds only its own split, and no task is in both files;
- the counts match the build report;
- **the files are byte-for-byte the ones `just golden` checked** (SHA-256 checksums in
  `build_report.json`), and the uploaded `splits.json` is the live `configs/splits.json`;
- the repo is private;
- tag `v1` doesn't exist yet. `--retag` moves it, but only use that before the hand-over. Once
  someone has trained on `v1`, publish a new version (`v1.1`) instead.

It then writes the data card, with every number derived from the build report and the splits,
and uploads with each person's own HF token. `--dry-run` runs every check and writes the card
without uploading. Chiara's review of #38 added most of these checks.

**Published 5 Nov:** `v1` holds exactly `golden_train.jsonl`, `golden_dev_reference.jsonl`,
`splits.json` and `README.md`, as checked on the Hub.

## What a record looks like

```text
id        glm53f/airline/0/1        (teacher / domain / task / attempt)
messages  [0] system     τ²-bench instructions + airline policy + tools as Python stubs + Action: rules
          [1] assistant  Hi! How can I help you today?                      ◀ target
          [2] user       … I need to cancel my reservation …
          [3] assistant  … could you please provide me with your user ID …  ◀ target
          [4] user       Sure, it's emma_kim_9957.
          [5] assistant  Action: ```json [{"tool_name": "get_user_details", …}]```  ◀ target
          [6] user       <results> Document: 0 … </results>
          …
train_on  [1, 3, 5, …]               every assistant turn
outcome   {"reward": 1.0, "checks": {"DB": 1.0, "COMMUNICATE": 1.0}}
teacher   {"model": "z-ai/glm-5.3-flash", "provider": "InferenceNet", "temperature": 0.0, "attempt": 1}
```

Training records run from 9 to 73 turns (median 31), and everything fits a 16K context.

## Questions sent to the Trainer at hand-over (5 Nov; answers pending)

1. Does GRPO need the live τ²-bench environment (`Tau2Episode`), or only these files?
2. Which chat template turns `messages` into tokens for Aya Expanse?
3. Does BCO need failed (negative) examples? If so, ~$5–15 for Aya runs on retail and airline
   train (telecom is done).
4. Every record opens with the agent's fixed greeting before any customer message. Aya's
   baseline sees the same, but some chat templates expect a user turn first.

## The raw run files (backed up 5 Nov)

The raw teacher and student runs are gitignored (`results/`), so they're backed up in the same
private dataset, under `raw/` (in later commits on `main`, so they're not part of `v1`):

| File | Why it matters |
|---|---|
| `results/fixes/glm53f.jsonl` | The source of this dataset: `just golden` rebuilds everything from it |
| `results/fixes/aya.jsonl` | The 30/30 agent's-view check, and the [student review](STUDENT_REVIEW_AYA.md) |
| `results/fixes/qwen38nt.jsonl` | The phase 2 comparison behind choosing GLM |
| `results/episodes/sample-*.jsonl` | Phase 1 bake-off |

They're at `raw/fixes/` and `raw/episodes/` in the dataset.

## Limitations

- **One teacher**, at temperature 0 for 156 of the 158 tasks. The records carry GLM's style,
  e.g. markdown in replies.
- **20 of τ²-bench's train tasks have no trajectory:** 15 in our train split (retail 41, 46,
  59, 83, 105, 110; airline 7; 8 telecom) and 5 in dev (retail 4, 76, 82, 112; airline 39).
  The 15 train ones are the hard subset's "teacher-unsolved" group (SEL-60).
- **The customer turns are GPT-4.1 outputs** (OpenAI's terms). They're context only and never a
  learning target. Settle the licence before any public release.
