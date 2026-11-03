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

## Tool calling: to be tested

Cohere's docs don't say whether Aya Expanse supports native tool calling. Its model card describes it as text in, text out, and Cohere's [tool-use guide](https://docs.cohere.com/docs/tools) doesn't list it. So:

1. **Probe first.** Send one request with `tools=` through the Compatibility API, and see whether it returns structured `tool_calls`.
2. **Fallback: prompted JSON tool calls.** Describe the tools in the prompt and ask for a JSON tool call, then parse it. This is the same approach Tiny Aya needs (AgentTune's JSON fallback prompt), so **the teacher and the student should share one tool-call format and parser.** Agree it with the Architect (data contract, SEL-35), so the teacher's fixes are directly usable as training data.

## Cost and budget (to be measured in SEL-37)

| Item | Estimate before measuring |
|---|---|
| Teacher, per conversation | ~100K input tokens (as GPT-4.1 used on telecom) → ~$0.05–0.07 |
| Simulated customer (GPT-4.1, OpenRouter), per conversation | ~$0.05 (measured in SEL-31) |
| 200 teacher fixes, plus retries for failed attempts | ~$20–30 |
| One 4-arm pass on the test split (100 tasks × 4 arms × 3 samples), customer only | ~$60 |

SEL-37 replaces these with numbers measured on 3 tasks: cost, latency and calls per conversation.

## Commands

| Command | What it does | Cohere calls |
|---|---|---|
| `just teacher-probe` | Checks the key and model ID, whether native tool calling works, and whether prompted JSON calls parse. Ends with a recommended `mode`. | 3 |
| `just teacher-check` | Teacher on one train task per domain (prompted mode). Reports cost, latency and score, then the budget estimate. `just teacher-check native` if the probe says native works. | ~30–60 |

- **Rate:** we use a **production key** (decided 3 Nov), so the driver doesn't throttle by default. If calls start failing with rate-limit errors (HTTP 429), they're retried; to space them out, set `TEACHER_RPM` in `.env` (e.g. `20` for a trial key).
- **Code:** `reflexive.teacher.TeacherClient`, used as the agent in `reflexive.episode.Tau2Episode`.
