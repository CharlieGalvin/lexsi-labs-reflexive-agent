# Teacher bake-off: which model should write the fixes? (SEL-58)

> **Status: the design, written before running (3 Nov). Kept as the pre-registered plan.**
> Phase 1 ran as designed: [results](TEACHER_BAKEOFF_RESULTS.md). Phase 2 ran differently
> from the plan below, by decision during the project: 2 finalists (GLM-5.3-Flash and
> Qwen3.8 27B with thinking off) instead of 3–4, **all** train tasks instead of 70, 1 attempt
> instead of 3, and providers not pinned. See [phase 2 results](TEACHER_PHASE2_RESULTS.md).
> Phase 3 hasn't run. **Outcome: GLM-5.3-Flash is the teacher.**

**Question.** The student will be Aya Expanse 32B. Which API model makes the best *teacher*,
meaning the most verified, clean fixes per dollar on tasks Aya fails, under a licence that lets
us train on its outputs? And does an open-weight teacher cost us anything in quality?

**What we know (3 Nov, 30 train tasks, database checks only):**

| Model | Weights | Passed | 95% range | Convs with invented IDs | Fixes for Aya 32B |
|---|---|---:|---|---:|---:|
| GLM-4.7-Flash | open | 17/30 | 39–73% | 3 | 17 |
| Qwen3 32B (thinking off) | open | 14/30 | 30–64% | 12 | 14 |
| Aya Expanse 32B (student) | open | 1/30 | 1–17% | 21 | – |
| Tiny Aya | open | 0/30 | 0–11% | 28 | – |

GLM already clears the bar: 17 fixes from 30 tasks, almost no invented IDs, $0.66 for the lot.
The bake-off asks whether a different model is clearly better, and whether the best one is open
or closed.

## Phase 1: screen (30 tasks, 1 attempt each, ~$30, ~1 hour)

- **Tasks:** the same fixed 30 (20 retail, 10 airline, seed 0). Comparisons are paired: every
  model sees the same tasks.
- **Models (`--models bakeoff`):** 12 picked from OpenRouter's τ²-bench airline board to span
  open vs closed weights and cheap vs strong (board score in brackets):
  - open: GLM-4.7-Flash (66.7%), GLM-5.3-Flash (75.6%), DeepSeek V4 Flash (73.9%),
    MiMo-V2.6-Flash (77.9%), Qwen3.8 27B (77.5%), Gemma 4 31B (75.3%), gpt-oss-120b (65.2%)
  - closed: Claude Haiku 5.5 (72.8%), Claude Sonnet 5.5
    (78.8%), GPT-5.1 (80.0%, the ceiling), Gemini 3.8 Flash (71.3%), Claude Opus 5.5 (75.3%). Gemini 3.5 Flash Lite was dropped on 3 Nov
- **Fixed for every model:** native tool calling through OpenRouter, with providers that
  support every parameter we send (`require_parameters`); temperature 0; the providers' default
  thinking settings; GPT-4.1 as the simulated customer; the same policy prompt and tools.
- **Measured per model:** pass rate with a 95% range; per-domain passes; conversations with
  invented IDs, unknown tools or retried failed calls; the agent's cost per pass; open weights
  or not (read live from OpenRouter's Hugging Face link); the fix count against Aya 32B.

**Shortlist rule (agreed before running):**

1. **Eligible:** at least 15/30 passed, at most 5/30 conversations with invented IDs, and no
   broken runs left (redo them with `--retry-broken`).
2. **Top tier:** eligible models whose pass count is within the best model's 95% range.
3. **Shortlist for phase 2:** the cheapest-per-pass *open* model in the top tier, the
   cheapest-per-pass *closed* model in the top tier, and GLM-4.7-Flash as the baseline.
   If the top tier has no open model, say so: that's the cost of going open.

With 30 tasks, models within about 5 passes of each other can't be told apart. Phase 1 only
removes the clearly weaker or clearly wasteful ones.

## Phase 2: confirm on the teacher's real job (~$40–80)

- **Tasks:** 60 *new* train tasks that change the database (40 retail, 20 airline), plus
  10 telecom train tasks, since Aya 32B's 128K window fits telecom. None of them overlap phase 1.
- **3 attempts per task**, as in production, where the teacher gets best-of-N.
- **Judges on** for retail and airline. This needs an OpenAI key (see SEL-26), and it also
  measures how much the database-only score overstates.
- **Metrics:**
  - **Fix yield:** share of tasks with at least one verified pass in 3 attempts (pass@3).
  - **Clean fix rate:** passing conversations with no invented IDs and no retried failed calls.
    These are the fixes we'd actually train on.
  - **Consistency:** pass^3, i.e. all 3 attempts pass.
  - **Cost per clean fix**, the agent and the customer together.
  - **Judged vs database-only gap.**
- **Decision:** the most clean fixes per dollar, among teachers whose licence allows training
  on their outputs. If the best open and best closed models are within 10 points of fix yield,
  choose the open one.

## Phase 3 (optional, needs the Trainer's GPU): does the teacher choice change the student?

Train two small adapters on Aya 32B from the same number of clean fixes (e.g. 100), one from
the best open teacher and one from the best closed teacher. Compare them on the dev split. This
is the real question, but it's only worth doing if phase 2 leaves two close candidates.

## Open vs closed: what's actually at stake

- **Licence.** Open-weight teachers (MIT, Apache-2.0) let us train on their outputs and
  publish the fix dataset (SEL-48). The terms of the big closed APIs typically forbid using
  outputs to build competing models. **Check each provider's terms before using a closed
  teacher's outputs for training, and record the result on SEL-58.**
- **Staying available.** Open weights can't be withdrawn. Cohere removed Aya Expanse 8B from
  its API on 4 Apr 2026, and we found out mid-run. With open weights, judges and other teams can
  rerun the exact teacher.
- **Self-hosting.** If we need thousands of fixes, an open teacher can run on our own GPU at a
  fixed cost, with no rate limits.
- **More signal for training.** Open weights give us token log-probabilities, which help to
  weight or filter fixes. True logit distillation would need a teacher with Aya's tokenizer,
  so that advantage is smaller for us.
- **Story.** "A fully open pipeline" is a stronger hackathon pitch.
- **Against open:** on OpenRouter an open model is served by several third-party providers,
  some with heavier quantisation or weaker tool-call parsing. Results can vary by provider, so
  record which one served each run, and pin it in phase 2 if the numbers look noisy. Closed
  models are usually more reliable at tool calling and, at the very top, slightly stronger.

## How to run

```bash
python scripts/teacher/bakeoff.py --models haiku --limit 2               # smoke test one model
just teacher-sample aya bakeoff                                         # phase 1, resumes
python scripts/teacher/bakeoff.py --models aya bakeoff --json results/bakeoff.json
python scripts/teacher/bakeoff.py --models or:moonshotai/kimi-k2.6       # any other OpenRouter model
```

**Phase 1 results:** [`TEACHER_BAKEOFF_RESULTS.md`](TEACHER_BAKEOFF_RESULTS.md) (3 Nov).
