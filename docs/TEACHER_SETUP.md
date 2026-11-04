# Teacher setup (SEL-37, SEL-8)

**Current setup (decided 3 Nov): the teacher is GLM-5.3-Flash** (`z-ai/glm-5.3-flash`, MIT
licence) via OpenRouter, with native tool calling. The student is **Aya Expanse 32B**.

| What | How |
|---|---|
| Teacher | `openrouter/z-ai/glm-5.3-flash`, native tool calls, temperature 0 (then 0.7 for retries) |
| Simulated customer | `openrouter/openai/gpt-4.1`, replies capped at 2,048 tokens |
| Judges | τ²-bench's own: retail `gpt-5.4-mini`, airline `gpt-5.5`, via OpenAI |
| Keys (`.env`) | `OPENROUTER_API_KEY` (teacher + customer), `OPENAI_API_KEY` (judges) |
| Run | `just teacher-fixes` (see `--help`); `--report` summarises what's saved, for free |
| Cost | ~$11 for all 178 train tasks, mostly the telecom customer |
| Why GLM | [`experiments/TEACHER_PHASE2_RESULTS.md`](experiments/TEACHER_PHASE2_RESULTS.md) |

**The rest of this page is the 3 Nov record of Aya Expanse 32B as the teacher**, before it
became the student. It's still the reference for **how Aya is prompted**: Cohere's `Action:`
tool-call format (`teacher.to_prompted`, `style="cohere"`) is the format the student is
evaluated and trained in (data contract, 4 Nov).

---

# History: Aya Expanse 32B via the Cohere API

The teacher model (Role 1, Curator) takes the agent seat in τ²-bench conversations on tasks the
student failed. Its verified wins become golden-fix training data (SEL-8, SEL-45, SEL-9).
These are Cohere's documented facts as of 3 Nov 2026, plus what they mean for our plan.

## The model

| | |
|---|---|
| API model ID | `c4ai-aya-expanse-32b` ([Cohere: Aya Expanse](https://docs.cohere.com/docs/aya-expanse)) |
| Size | 32B parameters, text in → text out |
| Context window | 128K tokens. Telecom's ~10K-token conversations fit easily, unlike Tiny Aya's 8K. |
| Maximum output | 4K tokens per reply |
| Languages | 23, including English |
| Weights / licence | Open weights on Hugging Face, **CC-BY-NC 4.0** (non-commercial) plus Cohere Labs' Acceptable Use Policy ([model card](https://huggingface.co/CohereLabs/aya-expanse-32b)). Fine for a research hackathon. |
| Price | Third-party listing: ~$0.50 / 1M input and ~$1.50 / 1M output tokens ([Maxim model library](https://www.getmaxim.ai/bifrost/model-library/compare/cohere/c4ai-aya-expanse-32b.md)). Confirm on [cohere.com/pricing](https://cohere.com/pricing) before relying on it. |

## API key and limits

1. **Get a key.** Create one at [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys).
2. **Add it to `.env`.** Put it in as `COHERE_API_KEY=` and never commit it.

**Trial-key limits**

- **20 requests per minute.** This is what our trial key shows. Cohere's [rate-limits page](https://docs.cohere.com/docs/rate-limits) lists 20/min for trial keys on its Chat models.
- **1,000 API calls per month.** Same page: trial keys are "limited to 1,000 API calls a month."
- **Trial usage is free.** See [How does Cohere pricing work](https://docs.cohere.com/docs/how-does-cohere-pricing-work).

**Production key:** pay-as-you-go, created on the same API keys page. Production limits are listed for Command models (500 requests/minute), but **not for Aya**. Ask support@cohere.com if we need more than 20/min.

**What the limits mean for us**

- **Each agent turn is one API call.** A teacher conversation is about 10–20 agent turns. The simulated customer runs on GPT-4.1 via OpenRouter, so its calls don't count against Cohere's limits.
- **Rate (trial key):** 20/min allows about **1 teacher conversation per minute**, so 200 fixes would need at least ~3½ hours of calls. **We use a production key instead (decided 3 Nov).** The driver doesn't throttle unless `TEACHER_RPM` is set, and it retries on HTTP 429.
- **Monthly cap:** 1,000 calls is about **50–100 conversations**. That's enough to set up, measure 3 tasks (SEL-37) and run a small pilot, but **not ~200 fixes** (~3,000 calls). Options:
  1. **A production key** (pay-as-you-go). This is the simplest: at the prices above, 200 fixes cost roughly $10–15 of teacher tokens.
  2. **Self-host the open weights with vLLM** (`vllm serve CohereLabs/aya-expanse-32b`). This needs ~64 GB of GPU memory at 16-bit, or ~20 GB 4-bit compressed, and would compete with training for GPU time.

## How to call it

- **Cohere SDK, Chat API v2** ([guide](https://docs.cohere.com/v2/docs/chat-api)):

  ```python
  import cohere
  co = cohere.ClientV2()          # reads CO_API_KEY; or pass os.environ["COHERE_API_KEY"]
  r = co.chat(model="c4ai-aya-expanse-32b", messages=[{"role": "user", "content": "Hi"}])
  print(r.message.content[0].text)
  ```

- **OpenAI-compatible endpoint** ([Compatibility API](https://docs.cohere.com/docs/compatibility-api)): base URL `https://api.cohere.ai/compatibility/v1`, with the Cohere key as the API key.
  - It supports `tools`, `temperature`, `max_tokens`, `seed`, `stop` and `response_format`.
  - It does **not** support `n` or `parallel_tool_calls`.
  - This is the easiest route for us: LiteLLM, which τ²-bench already uses, can call it as an OpenAI-compatible model.
- **Playground**, for trying prompts by hand: [dashboard.cohere.com/playground/chat](https://dashboard.cohere.com/playground/chat).

## Tool calling: prompted (native isn't supported)

**Tested 3 Nov (`just teacher-probe`).** A request with `tools=` is rejected by Cohere:
`invalid request: tool use is not supported by the provided model: c4ai-aya-expanse-32b`.
The prompted format worked first time: the teacher replied with a valid JSON tool call that
parsed. So the teacher runs in **prompted mode** (`TeacherClient("prompted")`, the default):

- **Use Aya's own tool format (`style="cohere"`, the default).** Aya Expanse comes from
  Cohere's Command R family and was trained on its tool-use prompt. Its model template, which
  Ollama also uses ([Ollama: aya-expanse:32b](https://ollama.com/library/aya-expanse:32b-q4_0/blobs/4f426f0c38cf)),
  lists the tools as Python stubs under "## Available Tools" and asks the model to "Write
  'Action:' followed by a json-formatted list of actions", each `{"tool_name", "parameters"}`,
  with a `directly-answer` pseudo-tool. Ollama's tool support for Aya is this same prompting,
  so we use the same wording through Cohere's API, with no self-hosting.
- **Our additions for τ²-bench:**
  - A turn is an Action *or* a message, never both.
  - Results come back in `<results>` tags.
  - "Never write tool results yourself."
  - Stop sequences (`<results>`, `Tool result`, `Document: 0`) end the reply where Aya started
    inventing results on 3 Nov.
- **Safety nets:**
  - A call written inside prose is still found, and the prose is dropped.
  - `directly-answer` becomes a message to the customer.
- **Alternative (`style="agenttune"`):** AgentTune's generic fallback wording, which is what
  Tiny Aya gets. Compare both with `just teacher-check cohere` / `just teacher-check agenttune`.
  Both write the same `{"tool_name", "parameters"}` objects, and the saved trajectory is
  OpenAI-style either way, so teacher fixes convert to the student's format. Which format the
  *student* uses is for the data contract (SEL-35). Tiny Aya is also a Cohere model, so the
  `cohere` format may suit it too.

## Cost and budget (measured 3 Nov, `just teacher-check`, prompted mode)

| Task (first train task) | Result | Teacher turns | Teacher cost | Customer cost | Time |
|---|---|---:|---:|---:|---:|
| Retail | ❌ (DB check only, no judge) | 6 | $0.017 | $0.018 | 34 s |
| Airline | ⚠️ "passed" the DB check, but see below | 7 | $0.019 | $0.016 | 36 s |
| Telecom | ❌ | 19 (182K input tokens) | $0.097 | $0.129 | 138 s |

**Mean per conversation:** teacher $0.044 + customer $0.054, ~70 s.

**Correction (after reading the transcripts):** these outcomes don't reflect the teacher's
ability, because of a parser bug, now fixed.

- **Tool calls inside prose were missed.** Aya Expanse wrote its JSON tool calls *inside*
  normal replies ("I will now call the tool: ```json [...]``` Please confirm"). The parser then
  only accepted replies that *start* with the JSON, so these went to the customer as text. **No
  agent tool ran in retail or airline.**
- **The teacher also hallucinated.** It invented IDs (a user ID and item IDs), wrote its own
  "Tool result: ...", and confirmed actions that never happened.
- **The airline "pass" is false.** Nothing changed in the database, and this task's expected
  end state evidently matches the unchanged database. Its judged check (COMMUNICATE), which
  needs an OpenAI key, wasn't run.
- **Telecom's first turn did call tools, with real hallucinations:** a placeholder argument
  (`"user's phone number"`, an Arg error) and 4 customer-side tools the agent doesn't have
  (Tool errors).
- **The fix** (`find_prompted_calls`) takes a tool call written anywhere in the reply, keeps only
  the call, and drops the prose, including any invented result. `teacher-check` now reports
  how many agent tool calls actually ran, and warns when none did.

**Re-measure before relying on the pass rate.** Verifying teacher fixes (SEL-9) needs the
judges (an OpenAI key): database-only scoring can pass a conversation in which nothing happened.

**Budget**

| Item | Estimate |
|---|---:|
| ~200 verified fixes (~600 attempts if ~1/3 pass; the real rate is still to be measured) | ~$60 |
| Test pass, all 4 arms × 3 samples (1,200 conversations, customer only) | ~$65 |
| Arm A baseline on train × 3 samples (customer only) | ~$30 |
| **Total** | **~$155** |

The teacher's pass rate drives the first line, and 3 tasks is too few to know it; SEL-45
measures it properly. ~600 attempts at ~70 s is ~11 h one after another, so run several at once.

**Backup teacher:** Cohere's Command models support native tool calling, including parallel
calls, multi-step calls and `tool_choice` ([tool-use usage patterns](https://docs.cohere.com/v2/docs/tool-use-usage-patterns),
examples with `command-a-plus-05-2026`). `TeacherClient("native", model=...)` already
supports this. Switching would depart from the proposal's teacher (Aya Expanse 32B), so it's a
team decision if Aya's pass rate turns out too low.

## Second run (3 Nov, both prompt styles, before the empty-reply fix)

The transcripts show the remaining problem is **Aya's task-solving, not the tool-call format**.
Both styles now get tool calls through, and every failure is a hallucination of the kind we
study:

| Run | What happened |
|---|---|
| AgentTune, retail | **Invented item IDs** (`1012345678`) and a payment method (`credit_card_0000000`) without looking up the order; called the exchange **5 times with the same invented IDs** (a loop); told the customer "I have resolved the issue". |
| AgentTune, airline ("pass") | **False pass.** Wrote a pseudo-call `[(tool_name: …)]` (not JSON, so not run), invented a "$250" refund, then transferred. The database was untouched. |
| AgentTune, telecom | Called 4 customer-side tools plus `get_customer_by_phone("user's phone number")`; never identified the customer. |
| Cohere, airline | Tool calling worked, but it **cancelled immediately**, without checking the policy or asking for confirmation, and promised a full refund. The correct outcome is to refuse. |
| Cohere, retail and telecom | Crashed on an empty reply (fixed since). |

## Hints: τ²-bench's resolution steps (teacher only)

τ²-bench ships a "ground truth" agent setting, `llm_agent_gt`. It adds `<resolution_steps>`,
generated from the task's expected actions, to the agent's prompt. `Tau2Episode(hints=...)`
uses it:

- **`steps` (recommended):** tool names in order, e.g. "[Step 1] Assistant action:
  find_user_id_by_name_zip. [Step 2] Assistant action: get_order_details. …". Aya still has to
  find every ID through the lookups, so its transcripts stay grounded.
- **`steps_with_args`:** the exact calls, IDs included. This risks teaching the student to use
  IDs it never looked up, the very hallucination we're fighting. Use it only to diagnose.
- **Guard rails:**
  - Hints are refused on test-split tasks (they come from the answer key), and every record
    stores `hints` / `hinted`.
  - Tasks whose correct outcome is "do nothing", such as refusals, get no steps.
  - The student is trained on the conversation, not on this prompt.
- **Other diagnostics:**
  - `unseen_arguments` lists ID-like tool arguments that never appeared earlier in the
    conversation, i.e. probable inventions. `teacher-check` prints them, and SEL-9 can use
    them to reject fixes.
  - Replies that mention `tool_name` but don't parse are now counted as `unparsed`.

## Is Aya a usable teacher? (`just teacher-sample`)

A fixed test to settle it, agreed 3 Nov:

- **Tasks:** 30 *train* tasks whose correct answer changes the database (20 retail, 10 airline,
  sampled with seed 0, the same every run), so doing nothing can't pass.
- **Scoring:** τ²-bench's database checks only. That is a **best case**: a fail is certain, a
  pass may still fail the judged checks. No OpenAI key needed.
- **Models:** Aya Expanse 32B (our teacher, prompted in its own format) against **Qwen3 32B**
  (`qwen/qwen3-32b` on OpenRouter, ~$0.08 / $0.28 per 1M tokens), a same-size model trained for
  tool use, with native tool calling and thinking off. That separates "Aya is weak" from "32B is
  too small". Add `gpt` for GPT-4.1, to show the tasks are solvable.
- **Also counted** per conversation: invented IDs, calls to tools the agent doesn't have, and
  identical retries of a failed call. Broken runs (API failures) are reported separately; redo
  them with `python scripts/teacher_sample.py --retry-broken`.
- **95% ranges for 30 tasks:** 0 wins → 0–11%; 3 → 3–26%; 6 → 10–37%; 9 → 17–48%.
- **Decision rule:**
  - **Aya ≤ 3/30 while the reference ≥ ~15/30:** unaided Aya is a poor teacher. Test hints
    next; if they don't rescue it, make the teacher-free variant (B′) the main path.
  - **Aya ≥ ~9/30 with few invented IDs:** a usable teacher.
  - **In between:** add ~20 tasks, or go to the hinted test.
- **Cost:** ~$3 for Aya plus the customer, ~$2 for Qwen plus the customer, +$4 with GPT-4.1.
  It runs 4 conversations at a time and resumes if interrupted.

## Result (3 Nov): unaided Aya is a poor teacher

Same 30 train tasks, database checks only (a best case):

| Model | Passed | 95% range | Conversations with invented IDs | With retried failed calls | Over 8K |
|---|---:|---|---:|---:|---:|
| Aya Expanse 32B (teacher) | 1/30 | 1–17% | 21 | 7 | – |
| Qwen3 32B (same size, native tools) | 14/30 | 30–64% | 12 | 0 | – |
| Tiny Aya (student, 8K window) | 0/30 | 0–11% | 28 | 19 | 8 |

- **The decision rule is met.** Aya ≤ 3/30 while the same-size reference solved about half. So
  it's Aya, not 32B being too small.
- **The fix pool is 1 task** (airline 42): unaided, the teacher can't supply ~200 verified
  fixes. Next is the hinted test (`just teacher-sample aya-steps tiny`). If hints don't rescue
  it, teacher-free healing (B′, SEL-24) becomes the main path.
- **Caveats:**
  - 21 runs broke on bugs since fixed: a bundle of tool results the converter didn't handle,
    Qwen's empty replies, an 8K overflow by 35 tokens, and a `None` response. They were counted
    as fails, so Qwen's true rate is probably higher. Re-run them with `--retry-broken`.
  - The invented-ID check is a heuristic, so some flags may be false positives. Check a few
    transcripts.
- **Re-plan (3 Nov): a stronger teacher, Aya as the student.** Leading option: **GLM-4.7-Flash**
  as teacher (`just teacher-sample glm`, `z-ai/glm-4.7-flash` on OpenRouter, 200K context,
  ~$0.07 / $0.40 per 1M tokens, native tools, thinking on, MIT-licensed weights) and **Aya
  Expanse 32B** as student. The summary prints a fix-pool line per teacher → student pair
  (e.g. `glm → aya`): the tasks the teacher solves and the student doesn't.
- **GLM-4.7-Flash result (3 Nov): 17/30 (39–73%)**, 3 conversations with invented IDs, $0.66 for
  all 30. That's 17 possible fixes for Aya 32B (Qwen3 32B: 14). The unaided-Aya problem is solved
  by changing the teacher. Which teacher, open or closed, is the **teacher bake-off** (SEL-58,
  `docs/experiments/TEACHER_BAKEOFF.md`, `just teacher-sample aya bakeoff`).
- **Aya Expanse 8B is no longer on Cohere's API** (`c4ai-aya-expanse-8b` removed 4 Apr 2026;
  [deprecations](https://docs.cohere.com/docs/deprecations)). To test it, serve the open weights
  (`CohereLabs/aya-expanse-8b` on Hugging Face, CC-BY-NC) on any OpenAI-compatible server and
  point the script at it:
  `AYA8_API_BASE=http://localhost:11434/v1 AYA8_MODEL=aya-expanse:8b just teacher-sample aya8`
  (Ollama shown; vLLM on a GPU is much faster). Without those settings the script stops at once
  instead of logging 30 broken runs. Any removed or misnamed model now stops after the first
  failure too.
- **Tiny Aya's probe:** no native tool calling (it answered in prose). Prompted, it wrote
  `Action: get_order_details({...})` in function-call style, which the parser now also reads.

## Decision (3 Nov): the teacher is GLM-5.3-Flash

- **Teacher:** `z-ai/glm-5.3-flash` via OpenRouter, native tool calling, MIT licence.
- **Performance:** on the whole τ²-bench train split (178 tasks, judges on), 88% on the first
  try, with a fix for 158 tasks. Against Qwen3.8 27B (thinking off) on the same 138 tasks, it
  ties on retail and airline and wins telecom 89% to 49%. Details:
  [`experiments/TEACHER_PHASE2_RESULTS.md`](experiments/TEACHER_PHASE2_RESULTS.md).
- **Fix run commands:** `just teacher-fixes` (148 unused tasks), `--tasks phase1` / `--tasks all`,
  `--attempts N`, `--report`, `--rejudge`. Results go to `results/fixes/<model>.jsonl`.
- **Accounts:** the teacher and the simulated customer go through OpenRouter, the judges
  through OpenAI. A full 178-task run costs ~$11, mostly the telecom customer.

## Commands

| Command | What it does | Cohere calls |
|---|---|---|
| `just teacher-probe` | Checks the key and model ID, whether native tool calling works, and whether prompted JSON calls parse. Ends with a recommended `mode`. | 3 |
| `just teacher-check [cohere\|agenttune] [none\|steps\|steps_with_args]` | Teacher on one train task per domain, in that prompt style (default `cohere`) and with those hints (default `none`). Reports cost, latency, score, the tool calls that actually ran and any invented IDs, then the budget. Saves to `results/episodes/teacher-check-<style>[-<hints>].jsonl`. | ~30–60 |

- **Rate:** we use a **production key** (decided 3 Nov), so the driver doesn't throttle by default. If calls start failing with rate-limit errors (HTTP 429), they're retried; to space them out, set `TEACHER_RPM` in `.env` (e.g. `20` for a trial key).
- **Code:** `reflexive.teacher.TeacherClient`, used as the agent in `reflexive.episode.Tau2Episode`.
