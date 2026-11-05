# Experiments: the Curator's story so far

Read these in order. Each page says at the top whether it's final, and what superseded it.

| # | Page | Date | Question | Answer | Status |
|---|---|---|---|---|---|
| 1 | [`TEACHER_BAKEOFF.md`](TEACHER_BAKEOFF.md) | 3 Nov | How should we pick the teacher? | The pre-registered design: phases, shortlist rule, measures | Design; phase 2 ran differently (noted at the top) |
| 2 | [`TEACHER_BAKEOFF_RESULTS.md`](TEACHER_BAKEOFF_RESULTS.md) | 3 Nov | Which of 12 models are good enough, and at what cost? | A statistical tie at the top; GLM-5.3-Flash and Qwen3.8 27B are the cheapest there | Final |
| 3 | [`TEACHER_PHASE2_RESULTS.md`](TEACHER_PHASE2_RESULTS.md) | 3 Nov | GLM-5.3-Flash or Qwen3.8 27B (thinking off), on every train task? | **GLM-5.3-Flash**: 88% vs 68%, decided by telecom (89% vs 49%); 158 of 178 tasks solved for $11 | Final |
| 3b | [`STUDENT_REVIEW_AYA.md`](STUDENT_REVIEW_AYA.md) ([log page](reports/aya_student_review.html)) | 4 Nov | Where does the student, Aya Expanse 32B, go wrong? Is it the format? | 7/30 passed; 0 unreadable calls. It's behaviour: phone tools, batching, invented IDs, retries | Final (30-task sample) |
| 4 | [`GOLDEN_TRAJECTORIES_V1.md`](GOLDEN_TRAJECTORIES_V1.md) | 4 Nov | What exactly does the Trainer get? | **127 training + 31 dev reference records**, checked end to end | Final; **published 5 Nov** ([private HF dataset](https://huggingface.co/datasets/reagent-lexsi/tau2-golden-trajectories), tag `v1`) |

**Decisions along the way**, and where each is recorded:

- **Teacher = GLM-5.3-Flash; student = Aya Expanse 32B** (3 Nov). Aya solved 1/30 as a
  teacher. Pages 2–3 and [`../TEACHER_SETUP.md`](../TEACHER_SETUP.md).
- **Telecom fully in scope** (3 Nov). Aya Expanse's 128K context fits it.
  [`../TAU2_SETUP.md`](../TAU2_SETUP.md).
- **Data contract v1** (4 Nov): Cohere `Action:` format, the agent's view, loss on assistant
  turns, one trajectory per solved task, dev split by task.
  [Contract doc](https://claude.ai/artifact/Ew1amyTBrTHbdiW92DQzNw); page 4.
- **Splits frozen** (4 Nov): train 142, dev 36, test 100. `configs/splits.json`.
- **Prompting alone doesn't fix it** (4 Nov, Chiara): on all 74 telecom train tasks, Arm C
  (a fixed "be careful not to hallucinate" instruction) passed 2 vs Arm A's 5 (exact McNemar
  p = 0.45), with no drop in hallucination flags. [`../EXPERIMENTS.md`](../EXPERIMENTS.md).
- **Golden trajectories v1 handed over** (5 Nov) to the Trainer and Architect.

**Not yet done:**
- **The hard subset**, as golden trajectories v1.1 (SEL-7 builds it, SEL-60 publishes it):
  healable tasks (Arm A fails, a golden fix exists), teacher-unsolved tasks and tasks Aya
  already solves. It needs Arm A on retail and airline train (SEL-55) and the failure
  classifier (SEL-12, PR #40).
- Aya on the test split for Arm A (the final four-arm runs, SEL-26).
- The bake-off's phase 3: does the teacher choice change the student?

Every run, with cost: [`../EXPERIMENTS.md`](../EXPERIMENTS.md).
