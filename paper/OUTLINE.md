# Paper outline: Reflexive Agents (Stage 2)

EMNLP 2023 template (`paper/template/`), long paper: **8 pages** of main text. The Limitations
section is mandatory; Limitations, Ethics, references and appendices don't count toward the 8 pages.
Final (non-review) mode with the authors named (Katie Kitchen, Charlie Galvin, Chiara Picardi; Lloyds
Banking Group), since the hackathon write-up isn't blind. Submission due **Tue 24 Nov** (SEL-80).

## PaperOrchestra verdict (SEL-90): yes for Stage 1 prose, no for related work

We keep its draft, [`paper/draft/main.tex`](draft/main.tex), as the skeleton for the Stage 1
sections (SEL-93). We don't use its bibliography or its related-work section.

**What it produced** (20 Nov, about 50 minutes on `gemini-4-neon`, plus `gemini-3.8-flash` for
search): an EMNLP-format draft whose main text fits 8 pages. It has an introduction, related work,
method, experiments (setup, the bake-off, the test results, the heal cycles, the ablations, GRPO,
error analysis), the conclusion, Limitations, Ethics (with an AI-assistance declaration) and a
3-part appendix. It compiles with no errors. Its own 3-reviewer ensemble gave it 5/10, Reject.

**Why it's worth keeping:**
- The numbers are faithful. Of 308 distinct numbers in the draft, 300 appear verbatim in its
  inputs (the README, `docs/EXPERIMENTS.md`, the blog post and 7 experiment pages).
- The other 8 are correct too:
  - 7 match committed `results/*.metrics.json` values: the 95% intervals on hallucination and loop
    rates, from `arm-A-test-vs-arm-S2-test.compare.metrics.json`; 252 = 104 + 74 + 74 replayed
    rollouts; and the gate's 1.10× bound;
  - 67.3% is 37/55.
- `check_numbers.py` still has to pass on the final `.tex` (SEL-112).
- The method and experiment sections are detailed and accurate: the classifier's kinds, the two
  fix granularities, the masked SFT objective, the gate's four checks and decision budget, and
  GRPO arms G and H.
- The peer reviews are the most useful output: [`paperorchestra/reviews/`](paperorchestra/reviews/).

**Why not for related work:**
- Only 9 references survived, all real but too few. τ²-bench, LoRA, QLoRA, GRPO, DAgger and the
  self-correction literature are missing.
- The cause is Semantic Scholar. PaperOrchestra checks each found paper against it, and without
  an API key Semantic Scholar answers most requests with 429. PaperOrchestra treats that as "not
  found" and silently drops the paper.
- So SEL-94 writes related work by hand and checks every entry. The one bib fix so far: a Unicode
  character in the Aya Expanse author list.

**Known gaps in the draft:**
- ~~**No figures.**~~ Done 21 Nov: the 4 images are in Appendix D (Figures), referenced from §3.2.
- **No Stage 2 content** (section 6, SEL-111) and **no threat model** (section 7, SEL-99).
- ~~**The AI-assistance declaration**~~ Reworded 21 Nov to say only what was done: an automated check
  found every number in the committed docs, configs or results. SEL-112's `check_numbers.py` on the
  `.tex` is still open.
- **Section names:** it calls the system "ReAgent" and the arms A/C/B, as the README does.

**What the reviewers asked for:** these are worth answering in the paper; Stage 2 covers some.
- The classifier as a runtime guard, compared with SFT: the guard, SEL-111.
- Retention of general ability: SEL-111.
- Classifier precision and recall in every domain: SEL-98.
- The serving confound: A and C ran on Cohere with 128K context, B on vLLM with 32K, and P wasn't
  run on test. State it plainly in Limitations.
- SL on dev, to separate steps from data.
- The turn-level vs. trajectory-level comparison is confounded: say so.
- The validation split is the complement of the healable tasks, which biases it.
- "Self-healing" overclaims what did the work: in the run that won, the oracle, not the
  classifier, chose the data.

**Running it:** PaperOrchestra's README says OpenAI works, but three agents call Gemini directly.
With OpenAI only, the search step also hit gpt-5-search-api's 45k tokens-per-minute limit, and 40
of 60 searches were dropped. On a paid Gemini key it ran unmodified except for two hard-coded
model names (`gemini-3-flash-preview` → `gemini-3.8-flash`). The upstream commit and our patch
(which also includes an unused OpenAI fallback) are in [`paperorchestra/`](paperorchestra/), with
the EMNLP template adapted for it and its outline. Inputs: the README sections, the experiment
log, the blog post and 7 pages from `docs/experiments/`, plus 4 figures.

**Rules (apply to every section):**

- A person checks every citation: the paper exists, the authors and year are right, and it says
  what we cite it for (SEL-94). Nothing an automated tool suggested goes in unchecked.
- Every number comes from a committed `results/*.metrics.json` file and passes
  `scripts/blog/check_numbers.py`, extended to the `.tex` (SEL-112).
- The Ethics statement declares AI-assisted drafting.

## Owners

| § | Section | ~Pages | Owner | Ticket | Sources |
|---|---|---:|---|---|---|
| — | Title, abstract | 0.25 | Chiara | SEL-93 | README Abstract, blog title |
| 1 | Introduction | 0.75 | Chiara | SEL-93 | README Overview ("Why this problem"), blog "The idea" |
| 2 | Related work | 0.75 | Chiara | SEL-94 | new; topics below |
| 3 | Method: the reflexive loop | 1.25 | Chiara | SEL-93 (Stage 1), SEL-111 (Stage 2 parts) | README "How it works", `docs/experiments/HALLUCINATION_CLASSIFIER.md`, `TEACHER_BAKEOFF_RESULTS.md`, `SFT.md`, `configs/gate.yaml` |
| 4 | Experimental setup | 0.75 | Chiara | SEL-93 | README "Data, splits and metrics", "Training setup", `configs/splits.json` |
| 5 | Results, Stage 1 | 1.5 | Chiara (check: Katie) | SEL-93 | README Results, `docs/EXPERIMENTS.md`, `GRPO_HEAL.md`, `HEAL_FIRST_CYCLE.md`, `HEAL_FROM_V2.md` |
| 6 | Results, Stage 2 | 1.5 | Chiara (tables: Katie) | SEL-111, SEL-106, SEL-108, SEL-98 | Katie's Stage 2 tables |
| 7 | Threat model | 0.5 | Charlie | SEL-99 | Charlie's threat-model section |
| 8 | Discussion and conclusion | 0.5 | Chiara | SEL-111 | README "Discussion and implications", blog "What we learned" |
| — | Limitations | (free) | Chiara | SEL-111 | README Limitations, blog Limitations |
| — | Ethics statement and AI-assistance declaration | (free) | Chiara | SEL-112 | README Declaration |
| — | Appendix: prompts, judging rules, per-domain tables, adapters | (free) | Chiara; Katie for tables | SEL-93, SEL-108 | README Appendix, "Models and data" |
| — | Number check, final read, submission | — | Chiara; all three read | SEL-112 | — |
| — | Video script, from this outline | — | Charlie | SEL-113 | — |

## Section by section

### Abstract (≤ 200 words)
Agents act, so a hallucination is a wrong action. We present a detect → fix → retrain → gate loop.
Headline (test, 100 tasks × 2): Tool or Arg hallucination 95% → 10% of conversations; success
14.5% → 51.5% (95% CI 43.5–59.5%); the "be careful" prompt gives 13.0%. The loop also ran
automatically and the gate promoted it (dev 6 → 22 of 36). Then the Stage 2 headline (SEL-111). We
release the code, the dataset and the adapters.

### 1 Introduction
- Agents with write access: hallucination means a wrong record changed, not a wrong sentence.
- Open models on your own hardware need to be reliable without a frontier model in the loop, and
  need a safe way to keep improving after deployment.
- Contributions:
  1. the loop, with a gate whose rules were fixed before any results;
  2. the test result against a prompt-only control;
  3. the automated cycle, and the finding that the grain of the fix matters (turn vs. whole conversation);
  4. a negative result: GRPO's rewards were gamed;
  5. Stage 2: the made-up intent finding, the guard, and meta-healing (SEL-111);
  6. released artefacts.
- Figure 1: `docs/blog/img/heal-loop.png`.

### 2 Related work (SEL-94; every entry checked by hand)
- Self-correction and reflection in LLM agents.
- Agentic distillation, and fine-tuning on teacher trajectories.
- τ²-bench and other tool-use benchmarks.
- Hallucination in tool use: invented tools and invented arguments.
- RL for agents, and reward hacking.
- Runtime guardrails and monitors.

### 3 Method
- 3.1 **Detect:** the Tool / Arg / Loop classifier and the first failing turn
  (`HALLUCINATION_CLASSIFIER.md`). Figure: `failure-types.png`.
- 3.2 **Fix:** the teacher, picked from 12 models in a bake-off; GLM-5.3-Flash; fixes verified by
  τ²-bench's evaluator; turn-level vs. whole-conversation fixes.
- 3.3 **Retrain:** QLoRA rank 16 on Aya Expanse 32B, agent turns only.
- 3.4 **Gate:** ≥ +2 of 36 dev tasks, no more hallucinations, no domain −2; rules fixed in advance
  (`configs/gate.yaml`).
- 3.5 **Stage 2 additions** (SEL-111): the made-up intent checker, meta-healing, the guard.

### 4 Experimental setup
- τ²-bench v2.0.0: retail, airline, telecom; simulated customer GPT-4.1.
- Frozen splits: train 142 (30 of them validation), dev 36, test 100.
- Arms: A (untuned), C (A + "be careful" prompt), B (SFT v2), declared before the test run.
- Metrics: success with 95% bootstrap CIs over tasks; Tool/Arg hallucination; loops; tool-call
  validity; latency, mean and median (SEL-108); AuditKIT's oracle re-score.
- Compute and cost. A note that B ran self-hosted at 32K context, and A and C on Cohere's API.

### 5 Results, Stage 1
- Table 1: test arms A / C / B, per domain (README Results table). Figure: `test-results.png`.
- B − A +37.0 pts (+28.0 to +45.5); C − A −1.5 (−7.0 to +4.0).
- The SFT ablations on validation: P / S / S2 / SL / S3 / S4. Data matters more than steps.
- The automated heal cycles: the turn-level cycle (findings −71%, success 4 → 4, rejected) against
  the trajectory-level cycle (6 → 22, promoted). Say clearly that the second used v2's data and was
  a demonstration.
- The negative result: GRPO (arms G and H). The rewards were gamed, and arm H was rejected (24 → 19).
- Error analysis: 37 of 55 remaining failures have no hallucination; they are decision errors
  (SEL-28).

### 6 Results, Stage 2 (SEL-111; numbers from SEL-106 / SEL-108 / SEL-98)
- The guard and the retrained model on test; the fair race; c1 on test.
- Retention of general ability; ROI; the human audit.
- The frozen checker against τ²-bench's verdicts (SEL-98).

### 7 Threat model (Charlie, SEL-99)
- The harms, and which component stops each; what isn't covered.

### 8 Discussion and conclusion
- Healing is mostly about data; asking nicely doesn't work; the grain of the fix matters; what's
  left are decisions.

### Limitations (mandatory)
- 100 test tasks; one student model; a simulated customer; the dev set shaped v2's data; the
  32K-context difference between arms; GPU cost counted as $0 in successes per dollar.

### Ethics statement
- AI-assisted drafting declared; synthetic customers only, no personal data; dual-use notes.
