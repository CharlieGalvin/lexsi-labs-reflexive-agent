# Teacher setup: Aya Expanse 32B via the Cohere API (SEL-37)

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

## Commands

| Command | What it does | Cohere calls |
|---|---|---|
| `just teacher-probe` | Checks the key and model ID, whether native tool calling works, and whether prompted JSON calls parse. Ends with a recommended `mode`. | 3 |
| `just teacher-check [cohere\|agenttune] [none\|steps\|steps_with_args]` | Teacher on one train task per domain, in that prompt style (default `cohere`) and with those hints (default `none`). Reports cost, latency, score, the tool calls that actually ran and any invented IDs, then the budget. Saves to `results/episodes/teacher-check-<style>[-<hints>].jsonl`. | ~30–60 |

- **Rate:** we use a **production key** (decided 3 Nov), so the driver doesn't throttle by default. If calls start failing with rate-limit errors (HTTP 429), they're retried; to space them out, set `TEACHER_RPM` in `.env` (e.g. `20` for a trial key).
- **Code:** `reflexive.teacher.TeacherClient`, used as the agent in `reflexive.episode.Tau2Episode`.
