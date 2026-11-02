# Work plan

Thesis: **RL on tool-calling trajectories makes a small Cohere open-weights agent use fewer,
better tool calls and hallucinate less — without lowering task success.**

Replace the suggested tasks below with your own breakdown, then run
`scripts/setup_github_project.sh` to turn them into GitHub issues on a project board.

## Key dates (hackathon)

| Date | Event | Our milestone |
|---|---|---|
| Tue 4 Nov | — | **M1 Baseline**: pipeline runs end to end, baseline numbers committed |
| Sun 9 Nov | — | **M2 First win**: one trained adapter beats baseline on calls/success *and* hallucination |
| Thu 13 Nov | — | **M3 Submission-ready**: ablations done, model + data on HF, README results table |
| **Fri 14 Nov** | **Submission deadline** | Submit (buffer day) |
| Sun 16 Nov | Shortlist announced | — |
| Fri 21 Nov | Shortlist deadline | **M4 Thesis proven**: second task / larger eval, demo rehearsed |
| **Sat 22 Nov** | **Demo Day** | Live demo |

## Workstreams and suggested tasks

Owner initials go in the issue assignee. One owner per task; pair if blocked.

### W1 — Environment & data (`area:data`)
- [ ] Get gated access to `CohereLabs/tiny-aya-global` for every teammate; create team HF org
- [ ] GPU access sorted (Colab / Kaggle / lab / HF credits) and `make smoke` passes on it
- [ ] Verify BioGRID schema; fix `UNANSWERABLE_TEMPLATES` so they are truly unanswerable
- [ ] Decide on the second task (optional, M4): another DB/API agent from the problem list
- [ ] Push eval split + unanswerable probes to the Hub as a dataset with a card

### W2 — Rewards (`area:rewards`)
- [ ] Unit-test every reward on hand-written good/bad trajectories (extend `tests/`)
- [ ] Reward-hacking review: try to get high reward with bad behaviour; patch the hole
- [ ] Calibrate weights: log each reward separately on 50 rollouts before combining
- [ ] (Stretch) LLM-judge or process reward for argument quality

### W3 — Training (`area:training`)
- [ ] `make smoke` (10 steps) green on the team GPU
- [ ] Baseline GRPO run (`configs/grpo_efficiency.yaml`)
- [ ] Ablation: correctness-only reward vs. full reward (shows *our* rewards matter)
- [ ] Ablation: no unanswerable probes (shows the hallucination term matters)
- [ ] (Stretch) RLOO or DPO-on-ranked-rollouts comparison
- [ ] Push best adapter to the Hub with model card + `lexsi_provenance.json`

### W4 — Evaluation (`area:eval`)
- [ ] `make baseline` → commit `results/base.metrics.json`
- [ ] Evaluate each trained adapter; commit `results/<run>.metrics.json`
- [ ] Confidence intervals (bootstrap over episodes) for the headline metrics
- [ ] Error analysis: 20 failure transcripts categorised (loop / bad args / guess / refusal)

### W5 — Story & demo (`area:demo`)
- [ ] README results table (from `scripts/compare.py`) + one plot (calls vs. success)
- [ ] Side-by-side transcript demo: base agent loops/hallucinates, tuned agent doesn't
- [ ] Submission write-up; feedback/feature requests for Lexsi (they asked for it)
- [ ] Demo Day slides + rehearsal

## Definition of done for the headline claim

Across the held-out eval split, tuned vs. base:
- `calls_per_success` **down**, `redundant_call_rate` **down**
- `hallucination_rate` **down**, `correct_abstention_unanswerable` **up**
- `task_success` **not lower** (within CI)
- AuditKIT's independent scorers agree in direction
