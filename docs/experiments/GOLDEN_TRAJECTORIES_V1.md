# Golden trajectories v1 (4 Nov 2026, SEL-36, SEL-9, SEL-48)

**What the Curator hands the Trainer:** 127 conversations in which the teacher, GLM-5.3-Flash,
solved a τ²-bench **train** task, verified by τ²-bench's own evaluator. They're formatted
exactly as Aya Expanse 32B (the student) sees a conversation, with its turns marked as the
learning targets. There are also 31 dev conversations, for reference only and never trained on.
The format is fixed by the [data contract v1](https://claude.ai/artifact/Ew1amyTBrTHbdiW92DQzNw),
agreed on 4 Nov.

**Rebuild it yourself (no API calls, ~10 s):** `just golden --sample 5`. It needs the teacher
run's raw file `results/fixes/glm53f.jsonl`, which is gitignored; ask Charlie for it.

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

## The four steps (`scripts/build_golden.py`, `src/reflexive/golden.py`)

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

`scripts/publish_golden.py` re-checks every record first:

- IDs are unique;
- each file holds only its own split, and no task is in both files;
- the counts match the build report.

It then writes the data card and uploads **privately** to
`reagent-lexsi/tau2-golden-trajectories`, tagged `v1`. It refuses a public repo, and uses
each person's own HF token. `--dry-run` checks the records and writes the card without
uploading.

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

Training records run from 9 to 73 turns (median 31), and the longest is ~45K characters,
roughly 11–13K tokens. Everything fits a 16K context.

## Open questions for the Trainer (raise at hand-over)

1. Does GRPO need the live τ²-bench environment (`Tau2Episode`), or only these files?
2. Which chat template turns `messages` into tokens for Aya Expanse?
3. Does BCO need failed (negative) examples? If so, ~$20 for an Aya run on the train tasks.
4. Every record opens with the agent's fixed greeting before any customer message. Aya's
   baseline sees the same, but some chat templates expect a user turn first.

## To do: back up the raw run files

The raw teacher and student runs exist only on Charlie's laptop (`results/`, gitignored):

| File | Why it matters |
|---|---|
| `results/fixes/glm53f.jsonl` | The source of this dataset: `just golden` rebuilds everything from it |
| `results/fixes/aya.jsonl` | The 30/30 agent's-view check, and the [student review](STUDENT_REVIEW_AYA.md) |
| `results/fixes/qwen38nt.jsonl` | The phase 2 comparison behind choosing GLM |
| `results/episodes/sample-*.jsonl` | Phase 1 bake-off |

**Plan:** upload them to the same private HF dataset under `raw/`, so Chiara and Katie can
re-run `just golden` and the reports (SEL-48).

## Limitations

- **One teacher**, at temperature 0 for 156 of the 158 tasks. The records carry GLM's style,
  e.g. markdown in replies.
- **20 unsolved train tasks** (retail 4, 41, 46, 59, 76, 82, 83, 105, 110, 112; airline 7, 39;
  8 telecom) have no trajectory. They're the core of the hard subset (SEL-7).
- **The customer turns are GPT-4.1 outputs** (OpenAI's terms). They're context only and never a
  learning target. Settle the licence before any public release.
