# Work plan — Reflexive Agents

Claim: **a Detect → Synthesise → Retrain → Verify loop raises success rate and cuts
hallucination, loops, steps and redundant calls versus a naive agent (A), a "be careful" prompt
(C) and a DSPy-optimised prompt (D).**

**Live tracking is in Linear** ([Reflexive Agents project](https://linear.app/self-healing-hallucination/project/reflexive-agents-e8769e1623aa));
this file is the original plan the Linear issues (`SEL-5`…`SEL-30`) were created from.

> **Kept as the original 2 Nov plan; not updated.** What changed since:
> - Public benchmarks instead of a synthetic task suite: **τ²-bench** (train 142 / dev 36 /
>   test 100) and **BFCL v3 multi-turn**, not 200 SQL + API tasks.
> - **Teacher: GLM-5.3-Flash** (bake-off, 3 Nov), not Aya Expanse 32B.
>   **Student: Aya Expanse 32B**, not Tiny Aya.
> - Golden trajectories are verified by τ²-bench's own evaluator, not a sandbox replay.
> - Milestone wording below (v0 suite of 50, 200-task test split) refers to the old plan.

## Key dates

| Date | Event | Our milestone |
|---|---|---|
| Thu 5 Nov | — | **M1 Baseline**: schema + GPU smoke test + Arm A numbers on the starter task; data contract agreed |
| Tue 10 Nov | — | **M2 First heal**: v0 task suite (50) → one heal cycle beats Arm A on the dev split |
| Fri 13 Nov | Internal deadline | **M3 Submission-ready**: all 4 arms on the 200-task test split, README results |
| **Sat 14 Nov** | **Submission deadline** | Submit (buffer day; aim for Fri evening) |
| Mon 16 Nov | Shortlist announced | — |
| Sat 21 Nov | Shortlist deadline | **M4 Thesis proven**: multiple heal cycles, B′ ablation, demo rehearsed |
| **Sun 22 Nov** | **Demo Day** | Live demo |

The live, day-by-day plan (with sub-issues SEL-31…SEL-55) is the *Delivery plan* document in the Linear project.

## Role 1 — Data & Synthesis, "the Curator" (`area:data`)
- [ ] Design the SQL + API task suite schema (DB tables, mock APIs incl. sandboxed email) and verifiable outcomes
- [ ] Build 200 tasks; split dev/train vs. held-out test; push to the team HF org with a dataset card
- [ ] Build the "hard hallucination" subset: tasks that trigger Tool / Arg / Loop failures in Arm A
- [ ] Teacher setup: serve Aya Expanse 32B (HF endpoint / vLLM) and generate golden trajectories for failed tasks
- [ ] Verify every golden trajectory by executing it in the sandbox; drop the ones that fail (CuratorKIT quality gate)
- [ ] AuditKIT verification suite: outcome oracles per task + agent scorers; wire into the evaluation script
- [ ] Final ROI and success-per-token charts across the 4 arms

## Role 2 — Systems & Logic, "the Architect" (`area:systems`)
- [ ] Implement `classify_hallucination` (Tool / Arg / Loop) for τ²-bench and BFCL; add tests for each failure type
- [ ] Measure classifier agreement with AgentTune's LLM `FailureClassifier` on 50 failures
- [ ] Build the `ReflexiveAgent` class (rollout → classify → queue failure → swap adapter when the gate passes)
- [ ] DECIDE YAML workflow orchestrating Detect → Synthesise → Retrain → Verify (reuse `agenttune.decide.closed_loop`)
- [ ] Arm C (prompt-only) and Arm D (DSPy-optimised prompt on dev/train) harnesses, same rollout code as A
- [ ] Public demo: live UI or notebook showing a failure being detected, healed and re-run
- [ ] (Stretch) Online mode: heal cycles triggered during serving, with `DeploymentGate`

## Role 3 — Model & Training, "the Trainer" (`area:training`)
- [x] GPU access sorted; a 10-step smoke training run passes with `CohereLabs/tiny-aya-global` (gated licence accepted by all) — `just train-smoke`, 5 Nov (licence: Trainer done; others to confirm)
- [x] Baseline GRPO run (first config in `configs/`) — `configs/train/messy_grpo.yaml` on the starter task, 6 Nov: [MESSY_LOOKUP.md](experiments/MESSY_LOOKUP.md)
- [ ] HEAL training step: retrain the student on the Curator's golden trajectories (GRPO with rewards, and BCO)
- [ ] Adapter lifecycle: version every adapter on the Hub with model card + `lexsi_provenance.json`
- [ ] Multiple heal cycles: success / hallucination vs. cycle number (does it keep improving or overfit?)
- [ ] Ablation B′: heal without the teacher (student's own successful rollouts only)
- [ ] (Stretch) GRPO vs. BCO vs. RLOO comparison on the same heal data

## Shared — Evaluation & story (`area:eval`)
- [ ] Run all 4 arms on the held-out test split; commit `results/<arm>.metrics.json`
- [ ] Bootstrap confidence intervals for SR, hallucination rate, loop rate, steps/task
- [ ] Error analysis: 20 remaining failures after healing, categorised
- [ ] README results table + plots; submission write-up incl. feedback for Lexsi Labs
- [ ] Demo Day slides + rehearsal

## Definition of done for the headline claim

On the held-out test split, Arm B vs. Arms A, C and D:
- Success rate **up** (CIs not overlapping A; ideally not overlapping D)
- Hallucination rate and loop rate **down**
- Steps per task and redundant call rate **down**
- AuditKIT's independent scorers agree in direction
- No test-split task or teacher output for a test task was used in training or prompt optimisation
