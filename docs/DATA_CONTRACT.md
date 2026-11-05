# Data contract v1: golden trajectories (agreed 4 Nov 2026, SEL-35)

What the Curator hands the Trainer and the Architect, and the rules it follows. Agreed by
Charlie and Chiara on 4 Nov. Implemented in `src/reflexive/golden.py` and
`scripts/build_golden.py` (SEL-9), and published as the private Hugging Face dataset
[`reagent-lexsi/tau2-golden-trajectories`](https://huggingface.co/datasets/reagent-lexsi/tau2-golden-trajectories),
tag `v1` (SEL-48). How it was built and checked: [`experiments/GOLDEN_TRAJECTORIES_V1.md`](experiments/GOLDEN_TRAJECTORIES_V1.md).

A **golden trajectory** is one complete conversation in which the teacher, GLM-5.3-Flash, solved a
τ²-bench train task, verified by τ²-bench's own evaluator with judges on.

## The five rules

| # | Rule | In one line |
|---|---|---|
| 1 | **Format** | Cohere's `Action:` tool-call format, with the same prompt as Aya's Arm A baseline, for training *and* evaluation |
| 2 | **Input** | Each record is the conversation **as the agent saw it**, system prompt included |
| 3 | **Learning target** | The agent's turns only; everything else is context |
| 4 | **What goes in** | One clean teacher pass per solved task, from train tasks outside the dev split |
| 5 | **GRPO** | Fine-tuning needs only these files; GRPO also needs the live environment (to confirm with the Trainer) |

### 1. Format: Cohere's `Action:` format

The teacher's conversations were saved with OpenAI-style native tool calls. They're converted to
the format Aya was trained on and is evaluated with (`teacher.to_prompted(..., style="cohere")`).
Only the tool parts change; chat text passes through untouched.

| Part | As the teacher saved it (OpenAI) | In the training record (Cohere) |
|---|---|---|
| Tool list | A separate `tools` field | Written into the system prompt as Python stubs |
| Agent calls a tool | A structured `tool_calls` field | Agent text: ``Action: ```json [{"tool_name": …, "parameters": …}]``` `` |
| Tool result | Its own `tool` turn | A `user` turn: `<results>\nDocument: N\ntool_name: …\nresult: …\n</results>` |
| Customer and agent text | Unchanged | Unchanged |

The prompt stays identical across arms. Changing the prompt is Arm C's and Arm D's job.

### 2. Input: the conversation as the agent saw it

In telecom the customer operates their own phone. The saved conversation records those phone
actions (e.g. `check_status_bar()` and its result), but the agent never sees them: it only hears
what the customer says. The records drop them (`golden.agent_view`). Checked against 30 real
saved agent views: an exact match in 30/30.

### 3. Learning target: the agent's turns only

| Turn | In the record | Learning target |
|---|---|---|
| `system`: policy and tools | yes | no |
| `user`: the customer's message | yes | no |
| `user`: `<results>` from a tool | yes | no |
| `assistant`: an `Action:` | yes | **yes** |
| `assistant`: a reply to the customer | yes | **yes** |

`train_on` lists the target turns explicitly: every `assistant` turn, including the fixed
opening greeting. In training-library terms: loss on assistant tokens only (TRL's
`assistant_only_loss=True`, or labels of −100 elsewhere).

### 4. What goes in

| Conversation | In? | Why |
|---|---|---|
| Passed every check, no retried failed calls, no unknown tools | yes | A clean pass |
| Database right but a judged check failed | no | It told the customer something wrong |
| A second clean solution of the same task | no | One trajectory per task; only the first clean pass is kept |
| A dev task | no (reference file only) | Dev is never trained on |
| A test task | never | Test is for the final results only; the teacher never ran on it |
| Teacher failures, student conversations | not yet | Possible negatives, if BCO needs them; the hard subset adds the student's failures (v1.1, SEL-60) |

### 5. GRPO: files or the live environment

Fine-tuning on golden trajectories needs only the files. GRPO learns from the student's own fresh
attempts, so the training loop runs conversations itself and scores each one:

```python
ep = Tau2Episode("airline", task_id, max_context=128_000)   # a train task; GPT-4.1 as the customer
ep.reset()
while not ep.done:
    ep.step(student.turn(ep.prompt(), ep.tool_schemas))     # Aya, Cohere format
reward = ep.record()["outcome"]["reward"]                   # τ²-bench's evaluator: 1.0 pass, 0.0 fail
```

That needs `Tau2Episode`, the outcome checker and customer API budget (about $0.01–0.12 per
conversation; telecom costs most). Pass `max_context` explicitly: the default is still 8,192.

## Splits

Golden trajectories come only from τ²-bench's **train** tasks. Dev is carved out of them by task
(random, fixed seed `reagent-dev-v1`); test is τ²-bench's own, unchanged. Task IDs:
`configs/splits.json` (SEL-36).

| Split | Retail | Airline | Telecom | Total | Used for |
|---|---:|---:|---:|---:|---|
| Train | 59 | 24 | 59 | **142** | Golden trajectories; the student learns from these |
| Dev | 15 | 6 | 15 | **36** | Decisions during the project: checkpoints, the deployment gate, heal cycles, DSPy. Never trained on |
| Test | 40 | 20 | 40 | **100** | The final four-arm results only (SEL-26) |

## Files

| File | Records | Size | Contents |
|---|---:|---:|---|
| `golden_train.jsonl` | 127 | 5.9 MB | One training record per solved **train** task. Train on these. |
| `golden_dev_reference.jsonl` | 31 | 1.4 MB | Records for solved **dev** tasks, for reference only. Never train on these. |
| `splits.json` | | 17 KB | The train, dev and test task IDs |
| `README.md` | | | The data card: source, counts, version, how to load |

## How big a record is

Token counts are estimates at 3.5–4 characters per token. Check them with Aya's tokenizer before
choosing a maximum sequence length.

| | Characters | ≈ Tokens |
|---|---:|---:|
| Typical record (median) | 33,600 | 8.4K–9.6K |
| Longest record | 44,600 (telecom) | 11.2K–12.7K |
| System prompt (policy + tool stubs), in every record | 17,400 | 4.3K–5.0K |
| **Learning targets** per record (median) | 4,200 | 1.0K–1.2K |
| **Learning targets**, all 127 training records | 563,000 | **≈ 140K–160K** |

Two consequences for training:

1. **Everything fits a 16K context,** even the longest telecom record.
2. **Only about 13% of each record is learning target.** About half of every record is the
   system prompt, and most of the rest is customer text and tool results. So the whole
   training set is roughly 150K target tokens: enough for a LoRA fine-tune and a first heal
   cycle, but small. Expect later heal cycles (fixes for the student's own failures) to matter
   more than this set alone.

By domain, a record is typically 24K (airline), 30K (retail) or 39K (telecom) characters.

## One record

| Field | Type | Contents |
|---|---|---|
| `id` | string | `<teacher>/<domain>/<task_id>/<attempt>`, e.g. `glm53f/airline/40/1` |
| `format` | string | `reagent.golden/v1` |
| `domain`, `task_id` | string | The τ²-bench task |
| `split` | string | `train` (or `dev` in the reference file) |
| `messages` | list | The conversation as the agent saw it, in Cohere format: `system`, then `user` and `assistant` turns strictly alternating |
| `train_on` | list of int | Indexes in `messages` that are learning targets: every `assistant` turn |
| `tools` | list | The agent's tool definitions (OpenAI schema), for reference or another format later |
| `outcome` | object | `reward` (1.0) and the score per check, e.g. `{"DB": 1.0, "COMMUNICATE": 1.0}` |
| `teacher` | object | `model`, `provider`, `temperature`, `attempt` |
| `customer` | string | The customer model, `openrouter/openai/gpt-4.1` |
| `tau2_version` | string | `2.0.0` |
| `source` | object | The original saved conversation, for audits |

## Changing the contract

A change to any rule above is a new contract version (v2), agreed by the Curator, Trainer and
Architect, and a new dataset version. Never re-tag `v1` once someone has trained on it.
