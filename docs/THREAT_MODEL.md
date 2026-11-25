# Threat model

ReAgent heals an agent that has **write access**: it cancels orders, changes flights and edits
accounts in τ²-bench's retail, airline and telecom databases. The asset at risk is the
customer's records. The main threat isn't an attacker. It's the agent itself taking a wrong
action, and the healing loop making that worse while appearing to make it better.

## Harms, and what in the system stops each

| Harm | What stops it | Evidence |
|---|---|---|
| **A made-up action:** a tool the agent doesn't have, an invented ID or argument, a retried failed call | Healing on verified conversations; the classifier (Tool / Arg / Loop) and the gate's third check, which refuses a model whose findings rise more than 10% | Conversations with a made-up tool or argument: 95% → 10% on test; schema-invalid calls 86 → 12 of 200 |
| **A change the customer never agreed to** | The "no yes" rule (SEL-88, frozen Sun 22 Nov, before any test run), which counts changes made without the customer's explicit yes, applied to every test conversation; optionally the guard, which blocks such a change before it runs | 0.152 → 0.010 per conversation on test (95% of the change −0.310 to −0.035). The guard blocked 5 replies in 2 of 60 validation conversations and never a good change, but didn't move the score |
| **A wrong teacher:** the loop learning a mistake from the stronger model | Every teacher conversation is kept only if τ²-bench's own evaluator passes it | 158 of 178 train and dev tasks verified; 21 of 39 full-conversation fixes kept |
| **Reward hacking:** a reward that is satisfied without the task being done | Fine-tuning on verified conversations, not a hallucination reward; the guard is a filter, never a reward | Our GRPO runs rewarded the absence of hallucination and learned to act confidently, not correctly: they cancelled reservations nobody asked to cancel. None was adopted |
| **Overfitting the gate:** retrying until a model passes by luck | A fixed gate (`configs/gate.yaml`) on 36 dev tasks, at most 3 decisions on them; every rule dated before its run (`docs/experiments/STAGE2.md`); claims only on 100 untouched test tasks, 8 samples each, with paired 95% ranges | The gate rejected a cycle that cut hallucinations 71% but solved no more tasks. The same model scores 12 to 19 of 30 on validation from run to run, which is why single runs decide nothing |
| **A swapped or tampered adapter** | Every published adapter has `lexsi_provenance.json` (training config, data SHA-256, git commit, base model); the deployed one is pinned to an exact Hub revision in `results/adapters/state.json` | All Stage 1 and Stage 2 adapters published with provenance |
| **Something else getting worse:** general ability lost while healing | Retention benchmarks with a rule fixed in advance: a drop counts only if the paired 95% range lies entirely below zero | No drop for v2 or S2E4 against base, or for FP8 against 16-bit, on IFEval, GSM8K and Global-MMLU-Lite |

## Not covered

- **Prompt injection.** τ²-bench's tool outputs are honest database lookups. We didn't test
  instructions hidden in tool results or documents, the main attack on agents with write access.
- **A hostile customer.** The simulated customer (GPT-4.1) is cooperative. We didn't test social
  engineering, or customers pressing the agent to break policy beyond what τ²-bench's tasks do.
- **Decisions the benchmark can't verify.** Healing keeps only what τ²-bench can check. Wrong
  decisions are still the main failure: the final model fails 40% of test tasks, and our detectors
  find few of those failures at the turn where they go wrong.
- **Beyond one benchmark.** Three customer-service domains in English, one model (Aya Expanse
  32B), one teacher.
- **Deployment.** No human review of high-value changes, rate limits or rollback; those belong to
  the system around the agent.
