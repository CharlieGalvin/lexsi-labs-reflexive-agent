# τ²-bench setup and decisions (SEL-31)

**Decision: pin τ²-bench v2.0.0 and keep all three domains (retail, airline, telecom).**

> **Update 3–4 Nov:** the student is now **Aya Expanse 32B (128K context)**, not Tiny Aya
> (8K), so **telecom is fully back in scope** and the drop rule below no longer applies. The
> longest golden trajectory is ~13K tokens. The Tiny Aya context-fit sections are kept as the
> record of why telecom was at risk.

Originally, telecom was **at risk**: its prompt alone fills ~91% of Tiny Aya's 8,192-token
context window. Everything here is reproducible with `python scripts/tau2_inspect.py` (no API calls).

**Telecom drop rule (retired 3 Nov with the move to Aya Expanse 32B):** drop telecom if, in
Arm A rollouts on its train tasks, **more than 25% hit the context limit**, or its pass rate is
near zero with context overflow as the main cause. The overflow rate becomes the evidence in
the write-up. If dropped, we fall back to retail + airline (104 train / 60 test).

## Install

```bash
just setup-dev                   # .venv (Python 3.12) + τ²-bench v2.0.0 WITH its data folder
just tau2-inspect                # splits, prompt sizes, context fit (no API calls)
just tau2-first-run telecom      # first live run on a domain (default: retail)
```

- Pinned to the **v2.0.0** release tag. `just tau2` does a sparse clone into `third_party/tau2-bench`
  (~350 MB; skips the 577 MB of past leaderboard results) and installs it editable.
- **Don't `pip install` τ²-bench from git.** That installs the code but not its `data/` folder
  (tasks, databases, language packs), and runs then fail with
  `ValueError: no published language pack for 'en'` and a "Data directory does not exist" warning.
- Requires **Python 3.12 or 3.13** (3.14 is rejected). `just setup-dev` creates a 3.12 environment.
  On 3.13, τ²-bench also needs the `audioop-lts` backport, which the `bench` extra installs.

## API keys and judges

| Domain | How τ²-bench scores it | Keys needed |
| --- | --- | --- |
| retail | Final database **plus an LLM judge** (`gpt-5.4-mini`) on 112 of 114 tasks | `OPENAI_API_KEY` (judge is pinned to OpenAI) |
| airline | Final database **plus an LLM judge** (`gpt-5.5`) on what the agent told the customer | `OPENAI_API_KEY` |
| telecom | Environment state checks only, no LLM judge | `OPENROUTER_API_KEY` alone is enough |

The agent and simulated customer can run through either OpenAI or OpenRouter. With both keys in
`.env`, the script uses OpenAI for everything; set `AGENT_LLM` / `USER_LLM` to
`openrouter/openai/gpt-4.1` to bill those to OpenRouter while the judges use OpenAI.

## Splits (from τ²-bench's own `split_tasks.json`)

| Domain | Train | Test | Notes |
| --- | ---: | ---: | --- |
| retail | 74 | 40 | 16 agent tools |
| airline | 30 | 20 | 14 agent tools |
| telecom | 74 | 40 | In scope; also a 2,285-task generated pool for extra training |
| **Total** | **178** | **100** | all three domains |

**Frozen 4 Nov (SEL-36) in [`configs/splits.json`](../configs/splits.json).** A dev set is carved
from train **by task**, at random with a fixed seed (`reagent-dev-v1`):

| Domain | Train (for training) | Dev | Test |
| --- | ---: | ---: | ---: |
| retail | 59 | 15 | 40 |
| airline | 24 | 6 | 20 |
| telecom | 59 | 15 | 40 |
| **Total** | **142** | **36** | **100** |

Dev is for DSPy, gate checks and heal-cycle checks; a dev task's trajectory is never trained on.
`python scripts/make_splits.py --check` confirms the file matches a fresh rebuild from the seed.

Healing (hard subset, teacher fixes, GRPO) and the DSPy prompt (Arm D) use **train only** (and dev). All
reported numbers come from the 100 test tasks, run with 3–4 trials
each (τ²-bench's pass^k) so the confidence intervals stay usable on a small test set.

## Context fit for Tiny Aya (8,192 tokens): historical, the student is now Aya Expanse 32B (128K)

| Domain | Policy | Tool schemas | Fixed prompt | Gold-path tool I/O (median / p90) | Tasks over 8K on the gold path |
| --- | ---: | ---: | ---: | ---: | ---: |
| airline | ~1,900 | ~2,600 | ~4,500 | ~320 / ~1,300 | 1 of 50 |
| retail | ~1,700 | ~2,900 | ~4,600 | ~970 / ~2,700 | 7 of 114 |
| telecom | ~5,800 | ~1,600 | ~7,400 | ~120 / ~270 | 56 of 114 |

Approximate tokens (characters / 4). "Gold path" = replaying each task's reference actions,
so real rollouts (extra lookups, retries, chat) will be longer: treat these as lower bounds.

**What this means for the harness (SEL-32):**
- Serialise tool results as **compact JSON** (`separators=(",", ":")`). τ²-bench returns
  indented JSON; compacting alone cut retail overflows from 10 to 7 tasks.
- Add a context guard: when the conversation nears 8K, truncate the oldest tool outputs
  first, and log a `context_overflow` flag so it is reported, not hidden.
- Count context overflow as its own failure type in the error analysis; it is not a
  hallucination and should not be "healed".

## Simulated customer (user simulator)

- **Choice:** `gpt-4.1-2025-04-14`, τ²-bench's default user simulator. Using the default keeps
  our Arm A numbers comparable with published τ²-bench results.
- **OpenRouter works too:** set `OPENROUTER_API_KEY` instead of `OPENAI_API_KEY` and the
  script uses `openrouter/openai/gpt-4.1`.
- Pick **one** route (OpenAI direct or OpenRouter) and keep it for every arm and every run, so
  the simulated customer is identical across the experiment.
- Optional for cheap development runs only: `gpt-4.1-mini`. Never mix simulators between arms.
- **Cost per task:** τ²-bench records `agent_cost` and `user_cost` for every run.
  `bash scripts/tau2_first_run.sh` runs 2 retail train tasks and prints both, plus token counts
  (`scripts/tau2_report.py`). LiteLLM may not price OpenRouter model IDs, so the reported cost
  can be empty; the script then estimates it from tokens at GPT-4.1 list prices. Your
  OpenRouter activity page shows the actual spend. Post the numbers
  in SEL-31 / SEL-37 and multiply: (178 train + 100 test × 3 trials × 4 arms) × user cost per task.

## Version note

v2.0.0's English retail and telecom tasks differ from v1.0.1, so v2.0 scores are **not**
comparable with the 1.x leaderboard. Our headline comparison is between our own four arms on
the same test split, so this doesn't affect the experiment. If we want a leaderboard reference
point for Arm A, use the v2.0 text leaderboard.

## Outcome checker (SEL-39)

Pass/fail for any trajectory comes from τ²-bench's own scoring, through one function:

```python
from reflexive.outcome import score
out = score("telecom", task_id, messages, termination="user_stop")   # judges=True by default
out.passed, out.reward, out.checks, out.skipped
```

- **No grading code of our own.** `score` wraps the trajectory as a τ²-bench `SimulationRun` and
  calls `tau2.evaluator.evaluate_simulation` with the same task definitions, language-pack
  database, policy and judge settings that `tau2 run` uses.
- **Pass = reward 1.0.** This is τ²-bench's rule: the reward is the product of the checks in the
  task's `reward_basis`. A conversation that ends any way other than the agent or the customer
  stopping (step limit, too many errors, our context guard) scores 0.
- **Message format.** Takes τ²-bench messages or OpenAI-style chat dicts (`assistant` / `user` /
  `tool`, with `tool_calls`). **The customer's tool calls must be kept:** in telecom, 393 of 516
  expected actions are the customer's (e.g. toggling airplane mode on their phone), and they go
  on `user` messages. This matters for the data contract (SEL-35) and the harness (SEL-32).
- **Judges.** `judges=False` scores only the deterministic checks (DB, ENV_ASSERTION, ACTION)
  and lists what it left out in `out.skipped`. Use it wherever cost or repeatability matters,
  e.g. training rewards. Telecom never uses a judge, so it is unaffected.

| Domain | Graded by | Deterministic? |
|---|---|---|
| Telecom | ENV_ASSERTION (94 tasks), plus ACTION (20 tasks) | Yes |
| Retail | DB + NL_ASSERTION (112 tasks), DB only (2 tasks) | DB part only |
| Airline | DB + COMMUNICATE (all 50 tasks) | DB part only |

**Checks.**

| Command | What it shows | Cost |
|---|---|---|
| `just check` | 16 tests: correct actions pass and a missing change fails (one task per domain), premature stops score 0, both message formats agree | free |
| `just tau2-check-gold` | Replays every task's correct actions (must pass) and the same with the last state-changing action removed (should fail) | free, ~10 min |
| `just tau2-scorer-run` | Runs 5 fixed telecom tasks (3 state-check only, 2 that also check actions) and compares our score with the official one, in both message formats | ~$0.70 |
| `just tau2-check-scorer <run>` | The same comparison on any saved run. Run it on every real run (e.g. Arm A). | free |

Results of the full replay (2 Nov):

| Domain | Correct actions pass | Last change removed → fail |
|---|---|---|
| Retail | 114/114 | 103/104 |
| Airline | 50/50 | 26/26 |
| Telecom | 114/114 | 94/94 |

The one exception is **retail task 105**, and it's a τ²-bench data quirk. Its only expected
action exchanges the same item twice, which errors inside τ²-bench, so the expected database
never changes and "doing nothing" matches it. Its NL assertions (judged) still apply. Tasks
without any state-changing action (10 retail, 24 airline, 20 telecom) can't be tested this way.

## Conversation wrapper (SEL-32)

`reflexive.episode.Tau2Episode` is one τ²-bench conversation that our code drives turn by turn.
τ²-bench runs everything else, exactly as in `tau2 run`: the simulated customer (GPT-4.1,
temperature 0), the tools, the database and the customer's own tool calls.

```python
ep = Tau2Episode("retail", task_id)     # optional: count_tokens=<Tiny Aya tokenizer>, max_context=8192
opener = ep.reset()                     # τ²-bench's greeting + the customer's first message
while not ep.done:
    messages = ep.prompt()              # system prompt + conversation, fitted to the window
    if ep.done:                         # didn't fit even after trimming → ended as overflow
        break
    ep.step(model(messages, tools=ep.tool_schemas))   # {"role": "assistant", "content" | "tool_calls"}
record = ep.record()                    # trajectory + outcome + flags: one JSON line
```

- **Built on τ²-bench's Gym environment** (`tau2.gym.AgentGymEnv`), with two changes:
  - **No in-run scoring.** Its built-in scoring runs the LLM judges and, without an OpenAI key,
    loses the conversation. `record()` scores with `reflexive.outcome` instead (`judges=`).
  - **No `done` tool.** The Gym agent's extra `done` tool is hidden, so the model sees exactly
    the official agent's tools.
- **Same prompt and tools as the official agent.** `system_prompt` is τ²-bench's agent
  instructions plus the policy, including τ²-bench's language line. `tool_schemas` is the
  official agent's tool list. A test checks both against what `tau2 run` gives its agent.
- **Tool results go to the model as compact JSON.** It's the same data without indentation.
  This is the format `tau2_inspect.py` measured: retail tasks whose correct path exceeds 8K go
  from 10 to 7.
- **Context guard.**
  - `prompt()` keeps the conversation plus tool definitions within `max_context - reserve`
    tokens (defaults 8,192 and 1,024). It does this by replacing the oldest tool results with a
    placeholder, in the model's view only: the saved trajectory keeps them in full.
  - If the prompt still doesn't fit, the conversation ends as `context_window_exceeded`. That
    scores 0 and is flagged `context_overflow`, so it's reported separately and never healed.
  - `flags.peak_prompt_tokens` records the largest untrimmed prompt.
  - Token counting is pluggable. The default is characters ÷ 4; plug in Tiny Aya's tokenizer
    once everyone has access.
- **Tools as Python functions.** `tool_functions()` gives each official tool, with the same name,
  arguments and description, as a function that runs that tool call in the conversation. This is
  for AgentTune's `tools=` route. AgentTune's own loop has no simulated customer, so training
  rollouts need a `rollout_func` that drives `Tau2Episode` (the Trainer's side).
- **Saved record** (`reagent.tau2.episode/v0`, provisional until the data contract, SEL-35):
  domain, task, termination, outcome (passed / reward / checks / skipped), flags, models, cost,
  and `messages`. The messages are the full trajectory in OpenAI style, with the customer's tool
  calls on `user` messages.

**Prompt size at the first agent turn** (estimate, characters ÷ 4; re-check with Tiny Aya's
tokenizer):

| Domain | Policy + tools + opener | Left for the conversation (8,192 − 1,024 reserve) |
|---|---:|---:|
| Retail | ~4,900 | ~2,270 |
| Airline | ~4,860 | ~2,300 |
| Telecom | ~7,690 | **none (−520)** |

At 8K, telecom doesn't fit before the agent's first reply unless the reserve drops below ~500
tokens, and that leaves almost no room for the conversation. This is the evidence the SEL-31
drop rule asks for. Confirm it with the real tokenizer and Tiny Aya's actual prompt format before
the team decides.

**Checks**

| Command | What it shows | Cost |
|---|---|---|
| `just check` | 11 episode tests, among them: correct actions pass through the wrapper (retail, airline); prompt and tools match the official agent (all 3 domains); results are compact; the guard trims oldest-first and keeps the full trajectory; an over-long conversation ends as `context_window_exceeded` | free |
| `just tau2-episode-check` | GPT-4.1 plays the agent through `Tau2Episode` on the 5 SEL-39 telecom tasks: each conversation saves and scores, the customer's tool calls are recorded, and the peak prompt size is reported | ~$0.70 |

**Tool names and ID formats** (for the classifier, SEL-12). Write tools change the database;
the rest only read.

| Domain | Agent tools | Writes | Customer tools | ID formats (examples) |
|---|---:|---:|---:|---|
| Retail | 16 | 7 | 0 | user `yusuf_rossi_9620`, order `#W2378156`, product / item 10 digits `1656367028`, payment `credit_card_3124723` / `paypal_7644869` / `gift_card_…`, email `mia.garcia2723@example.com` |
| Airline | 14 | 6 | 0 | user `anya_garcia_5901`, reservation 6 characters `4OG6T3`, flight `HAT005`, payment `credit_card_2408938` / `gift_card_1642017` |
| Telecom | 13 | 6 | 30 | customer `C1001`, line `L1002`, bill `B1234321` |

- **Retail writes:** cancel_pending_order, exchange_delivered_order_items,
  modify_pending_order_address, modify_pending_order_items, modify_pending_order_payment,
  modify_user_address, return_delivered_order_items.
- **Airline writes:** book_reservation, cancel_reservation, send_certificate,
  update_reservation_baggages, update_reservation_flights, update_reservation_passengers.
- **Telecom writes:** suspend_line, resume_line, send_payment_request, enable_roaming,
  disable_roaming, refuel_data.
- **Telecom customer tools** run on the customer's phone and are called by the simulated
  customer, never by the agent. Examples: toggle_airplane_mode, toggle_data, toggle_roaming,
  set_network_mode_preference, reseat_sim_card, make_payment.
- `transfer_to_human_agents` exists in every domain.
- A call to a tool name that doesn't exist is a Tool hallucination candidate. τ²-bench returns
  an error and replays it as a no-op.
