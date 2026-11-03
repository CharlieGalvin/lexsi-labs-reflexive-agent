"""Is Aya Expanse 32B a usable teacher? A fixed 30-task test against a reference model (SEL-37).

    python scripts/teacher_sample.py                      # Aya vs Qwen3 32B (same size)
    python scripts/teacher_sample.py --models aya qwen gpt
    python scripts/teacher_sample.py --models aya --workers 2

**Tasks.** A fixed sample of 30 τ²-bench *train* tasks whose correct answer changes the database:
20 retail + 10 airline (seed 0, the same every run). On these, doing nothing can't pass.

**Scoring.** τ²-bench's database / state checks only (no LLM judges, so no OpenAI key needed).
That is a *best case*: failing it is a definite fail, but passing it doesn't guarantee the
judged checks would pass too. If even the best case is low, the model is a weak teacher.

**Models** (all as the agent through `Tau2Episode`, GPT-4.1 as the simulated customer):
- ``aya``: Aya Expanse 32B via Cohere, prompted tool calls (Aya's own format); our teacher.
- ``qwen``: Qwen3 32B via OpenRouter, native tool calling, thinking off (``/no_think``): a
  same-size model trained for tool use, so it separates "Aya is weak" from "32B is too small".
- ``gpt``: GPT-4.1 via OpenRouter (or OpenAI), native tool calling: shows the tasks are solvable.
- ``glm``: GLM-4.7-Flash (``z-ai/glm-4.7-flash``) via OpenRouter, native tool calling, thinking
  left on: the candidate *teacher* (MIT-licensed weights, so its outputs can be trained on).
- ``aya-steps``: the teacher again, with τ²-bench's resolution steps (tool names in order, no
  IDs) added to its prompt, the hinted test from the decision rule. Train tasks only.
- ``aya8``: Aya Expanse 8B, a candidate student, prompted like the others inside its real 8K
  window. Cohere removed it from its API on 4 Apr 2026, so it runs on a server you start
  yourself (Ollama, vLLM...): set ``AYA8_API_BASE`` (an OpenAI-compatible ``/v1`` URL) and
  ``AYA8_MODEL`` (the server's own name for it, e.g. ``aya-expanse:8b`` on Ollama).
- ``tiny``: Tiny Aya (``tiny-aya-global``) via Cohere, our *student*, prompted exactly like the
  teacher, inside its real 8,192-token window (our context guard on; overflows are counted). Its
  API price isn't published, so its cost line is the simulated customer only.

**Per conversation** it also counts tool hallucinations: ID-like arguments never seen earlier
(invented IDs), calls to tools the agent doesn't have, and identical retries of a failed call.

Records go to `results/episodes/sample-<model>.jsonl`; an interrupted run resumes where it
stopped (finished tasks are skipped). Prints pass rates with 95% confidence ranges (Wilson).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import sqrt
from typing import Any

from loguru import logger

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from reflexive.episode import Tau2Episode, estimate_tokens  # noqa: E402
from reflexive.teacher import (  # noqa: E402
    ContextTooLong,
    EmptyTeacherReply,
    TeacherClient,
    tool_errors,
    unseen_arguments,
)

logger.remove()
logger.add(sys.stderr, level="ERROR")

SAMPLE = {"retail": 20, "airline": 10}
SEED = 0
# USD per token, for models whose cost LiteLLM doesn't report (listed prices, 3 Nov 2026)
PRICES = {"qwen": (0.08e-6, 0.28e-6), "gpt": (2.0e-6, 8.0e-6), "glm": (0.07e-6, 0.40e-6)}


# --- Task sample --------------------------------------------------------------------------------


def sample_tasks(sizes: dict[str, int] = SAMPLE, seed: int = SEED) -> list[tuple[str, str]]:
    """The fixed sample: train tasks whose expected actions change state, per domain."""
    from tau2.registry import registry

    from reflexive.outcome import scorer

    out = []
    for domain, n in sizes.items():
        sc = scorer(domain)
        pool = sorted(t for t in registry.get_task_splits_loader(domain)()["train"]
                      if sc.write_action_indices(t))
        out += [(domain, t) for t in random.Random(seed).sample(pool, min(n, len(pool)))]
    return out


# --- Agents -------------------------------------------------------------------------------------


class NativeAgent:
    """An API model with native tool calling, as the agent (one call per turn)."""

    def __init__(self, model: str, prices: tuple[float, float], no_think: bool = False):
        self.model, self.prices, self.no_think = model, prices, no_think
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    def turn(self, messages: list[dict], tools: list[dict]) -> dict:
        import litellm

        litellm.suppress_debug_info = True
        if self.no_think and messages and messages[0]["role"] == "system":
            messages = [{**messages[0], "content": messages[0]["content"] + "\n\n/no_think"},
                        *messages[1:]]
        for _attempt in range(2):  # Qwen3 sometimes returns an empty reply: ask once more
            start = time.monotonic()
            resp = litellm.completion(model=self.model, messages=messages, tools=tools,
                                      temperature=0.0, num_retries=3)
            self.usage["seconds"] += time.monotonic() - start
            self.usage["calls"] += 1
            u = getattr(resp, "usage", None)
            self.usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            self.usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            msg = resp.choices[0].message
            if msg.tool_calls or (msg.content or "").replace("<think>", "").replace("</think>", "").strip():
                break
            self.usage["empty"] = self.usage.get("empty", 0) + 1
            messages = [*messages, {"role": "user", "content":
                                    "(Write your next turn now: a tool call, or a message to the customer.)"}]
        calls = [{"id": tc.id, "type": "function",
                  "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                 for tc in (msg.tool_calls or [])]
        if calls:  # τ²-bench's rule: a tool call or a message, not both
            return {"role": "assistant", "content": None, "tool_calls": calls}
        text = (msg.content or "").replace("<think>", "").replace("</think>", "").strip()
        if not text:
            raise EmptyTeacherReply(f"{self.model} returned an empty reply")
        return {"role": "assistant", "content": text}

    def usage_dict(self) -> dict:
        pin, pout = self.prices
        return {**self.usage, "seconds": round(self.usage["seconds"], 2),
                "est_cost": round(self.usage["prompt_tokens"] * pin
                                  + self.usage["completion_tokens"] * pout, 4)}


class AyaAgent:
    """A Cohere Aya model, prompted tool calls in Aya's own format (default: the 32B teacher)."""

    def __init__(self, model: str = "c4ai-aya-expanse-32b", prices: tuple[float, float] | None = None,
                 **server):  # api_base / api_key for a self-hosted model
        self.client = TeacherClient("prompted", style="cohere", model=model, prices=prices, **server)
        self.model = f"cohere/{self.client.model}"

    def turn(self, messages, tools):
        return self.client.turn(messages, tools)

    def usage_dict(self) -> dict:
        return self.client.usage.as_dict()


def make_agent(name: str) -> Callable[[], Any]:
    from reflexive.llm import default_model

    if name == "aya":
        return AyaAgent
    if name == "qwen":
        return lambda: NativeAgent("openrouter/qwen/qwen3-32b", PRICES["qwen"], no_think=True)
    if name == "gpt":
        return lambda: NativeAgent(default_model(), PRICES["gpt"])
    if name == "tiny":
        return lambda: AyaAgent("tiny-aya-global", prices=(0.0, 0.0))  # price not published
    if name == "aya-steps":
        return AyaAgent
    if name == "glm":
        return lambda: NativeAgent("openrouter/z-ai/glm-4.7-flash", PRICES["glm"])
    if name == "aya8":  # self-hosted: Cohere retired c4ai-aya-expanse-8b on 4 Apr 2026
        return _self_hosted_aya8
    raise ValueError(name)


class ModelUnavailable(RuntimeError):
    """The model can't be reached at all (removed, wrong ID, server not set): stop the run."""


def _self_hosted_aya8() -> AyaAgent:
    import os

    base, model = os.environ.get("AYA8_API_BASE"), os.environ.get("AYA8_MODEL")
    if not base or not model:
        raise ModelUnavailable(
            "Aya Expanse 8B is no longer on Cohere's API (removed 4 Apr 2026). Serve it yourself "
            "and set AYA8_API_BASE and AYA8_MODEL; see docs/TEACHER_SETUP.md.")
    return AyaAgent(model, prices=(0.0, 0.0),  # self-hosted: no API price
                    api_base=base, api_key=os.environ.get("AYA8_API_KEY", "none"))

# (max_context, reserve) per model: the student gets its real 8K window, the others don't need one
CONTEXT = {"tiny": (8192, 1024), "aya8": (8192, 1024)}
# characters ÷ 4 undercounted Tiny Aya's tokenizer (3 Nov: Cohere refused a prompt by 35 tokens)
COUNT = {"tiny": lambda msgs: int(estimate_tokens(msgs) * 1.15),
         "aya8": lambda msgs: int(estimate_tokens(msgs) * 1.15)}
# τ²-bench resolution-step hints (train tasks only; see Tau2Episode(hints=...))
HINTS = {"aya-steps": "steps"}
MODELS = ["aya", "qwen", "gpt", "glm", "tiny", "aya8", "aya-steps"]


# --- One conversation ---------------------------------------------------------------------------


def run_one(model: str, domain: str, task_id: str) -> dict:
    agent = make_agent(model)()
    max_context, reserve = CONTEXT.get(model, (128_000, 4_096))
    ep = Tau2Episode(domain, task_id, max_context=max_context, reserve=reserve,
                     count_tokens=COUNT.get(model, estimate_tokens), hints=HINTS.get(model, "none"))
    start = time.monotonic()
    error = None
    try:
        ep.reset()
        while not ep.done:
            messages = ep.prompt()
            if ep.done:
                break
            try:
                ep.step(agent.turn(messages, ep.tool_schemas))
            except ContextTooLong:  # the API's own token count was over the window: an overflow
                ep._end_for_context()
    except Exception as e:  # noqa: BLE001 - one bad conversation shouldn't stop the sample
        error = f"{type(e).__name__}: {str(e)[:300]}"
        if ep._env is not None and not ep.done:
            ep.finish()
    if not ep.done:  # failed before the conversation started: record it as a failed attempt
        rec = {"domain": domain, "task_id": task_id, "termination": "unexpected_error",
               "outcome": {"passed": False, "reward": 0.0, "checks": {}, "judged": False,
                           "skipped": [], "note": error}, "messages": [], "user_cost": None,
               "sample_model": model, "agent_llm": agent.model,
               "agent_usage": agent.usage_dict(), "error": error}
    else:
        rec = ep.record(judges=False, sample_model=model, agent_llm=agent.model,
                        agent_usage=agent.usage_dict(), error=error)
    msgs = rec["messages"]
    rec["diagnostics"] = {
        "tools_ran": sum(len(m.get("tool_calls") or []) for m in msgs if m["role"] == "assistant"),
        "unseen_arguments": unseen_arguments(msgs),
        **tool_errors(msgs),
    }
    rec["wall_s"] = round(time.monotonic() - start, 1)
    return rec


# --- Summary ------------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence range for a pass rate of k out of n (Wilson score interval)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d), min(1.0, (c + r) / d)


BROKEN_ENDINGS = {"unexpected_error", "infrastructure_error", "agent_error", "user_error"}


def _broken(rec: dict) -> bool:
    """The run broke (an API failure, a crash), so it says nothing about the model's ability."""
    return bool(rec.get("error")) or rec.get("termination") in BROKEN_ENDINGS


def summarise(model: str, recs: list[dict]) -> dict:
    n = len(recs)
    k = sum(r["outcome"]["passed"] for r in recs)
    lo, hi = wilson(k, n)
    diag = [r["diagnostics"] for r in recs]
    by_domain = {d: (sum(r["outcome"]["passed"] for r in recs if r["domain"] == d),
                     sum(1 for r in recs if r["domain"] == d)) for d in SAMPLE}
    return {
        "model": model, "n": n, "passed": k, "rate": k / n if n else 0.0, "ci": (lo, hi),
        "by_domain": by_domain,
        "invented_ids": sum(bool(x["unseen_arguments"]) for x in diag),
        "unknown_tools": sum(bool(x["unknown_tools"]) for x in diag),
        "repeated_failures": sum(bool(x["repeated_failures"]) for x in diag),
        "no_tools": sum(x["tools_ran"] == 0 for x in diag),
        "overflow": sum(bool((r.get("flags") or {}).get("context_overflow")) for r in recs),
        "errors": sum(_broken(r) for r in recs),
        "cost": sum(r["agent_usage"]["est_cost"] + (r["user_cost"] or 0) for r in recs),
        "mean_s": sum(r["wall_s"] for r in recs) / n if n else 0.0,
    }


def print_summary(rows: list[dict]) -> None:
    print("\n=== Summary: 30 train tasks that need a state change (DB checks only: a best case) ===")
    print(f"{'model':<6} {'passed':>9} {'95% range':>12} {'retail':>7} {'airline':>8} "
          f"{'invented IDs':>13} {'unknown tool':>13} {'retried fail':>13} {'no tools':>9} "
          f"{'over 8K':>8} {'errors':>7} {'cost':>7}")
    for s in rows:
        lo, hi = s["ci"]
        dom = s["by_domain"]
        print(f"{s['model']:<6} {s['passed']:>3}/{s['n']:<3} ({s['rate']:>4.0%}) "
              f"{lo:>4.0%}–{hi:<4.0%} {dom['retail'][0]:>3}/{dom['retail'][1]:<3} "
              f"{dom['airline'][0]:>3}/{dom['airline'][1]:<4} "
              f"{s['invented_ids']:>9}/{s['n']:<3} {s['unknown_tools']:>9}/{s['n']:<3} "
              f"{s['repeated_failures']:>9}/{s['n']:<3} {s['no_tools']:>5}/{s['n']:<3} "
              f"{s['overflow']:>4}/{s['n']:<3} {s['errors']:>7} ${s['cost']:>6.2f}")
    print("Counts in the hallucination columns are conversations with at least one such event.")
    if any(s["model"] == "tiny" for s in rows):
        print("tiny: cost is the simulated customer only (Tiny Aya's API price isn't published); "
              "'over 8K' = ended by the context guard (estimated tokens).")
    if any(s["errors"] for s in rows):
        print("⚠ 'errors' are broken runs (API failures, crashes): they count as fails above but say "
              "nothing about the model. Fix the cause and re-run with --retry-broken.")
    aya = next((s for s in rows if s["model"] == "aya"), None)
    refs = [s for s in rows if s["model"] != "aya"]
    if aya and refs:
        print("\nDecision rule (agreed before running):")
        print("  Aya ≤ 3/30 while a reference ≥ ~15/30  → unaided Aya is a poor teacher; test hints next.")
        print("  Aya ≥ ~9/30 with few invented IDs      → usable teacher; best-of-N will give enough fixes.")
        print("  In between                             → add ~20 tasks, or go straight to the hinted test.")


TEACHERS = ("gpt", "glm", "qwen", "aya-steps", "aya")  # candidate teachers, strongest first
STUDENTS = ("aya", "aya8", "tiny")              # candidate students


def _gone(rec: dict) -> bool:
    err = rec.get("error") or ""
    return "NotFoundError" in err or "was removed" in err


def print_overlap(passed_by: dict[str, set]) -> None:
    """Teacher vs student: the tasks only the teacher solves are the possible fixes."""
    pairs = [(t, s) for t in TEACHERS for s in STUDENTS if t != s and t in passed_by and s in passed_by]
    if not pairs:
        return
    print("\nPossible teacher fixes (tasks the teacher solves and the student doesn't):")
    for t, s in pairs:
        teacher, student = passed_by[t], passed_by[s]
        fix = sorted(teacher - student)
        print(f"  {t:>9} → {s:<5} {len(fix):>2}   ({len(student - teacher)} the other way, "
              f"{len(teacher & student)} both)  {', '.join(f'{d}:{i}' for d, i in fix) or '-'}")


# --- Main ---------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=MODELS, default=["aya", "qwen"])
    ap.add_argument("--workers", type=int, default=4, help="conversations run at once per model")
    ap.add_argument("--fresh", action="store_true", help="ignore earlier results and start again")
    ap.add_argument("--list", action="store_true", help="only print the task sample")
    ap.add_argument("--limit", type=int, help="run only the first N tasks (a quick smoke test)")
    ap.add_argument("--retry-broken", action="store_true",
                    help="re-run conversations that broke (API failures, crashes)")
    a = ap.parse_args()

    tasks = sample_tasks()  # also loads each domain's tasks once, before the worker threads
    tasks = tasks[:a.limit] if a.limit else tasks
    from tau2.gym import register_gym_agent

    import reflexive.episode as episode

    register_gym_agent()  # once, here, rather than raced by the workers
    episode._REGISTERED = True
    if a.list:
        for d, t in tasks:
            print(d, t)
        return 0
    rows = []
    passed_by: dict[str, set] = {}
    for model in a.models:
        out = pathlib.Path(f"results/episodes/sample-{model}.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        done: dict[tuple[str, str], dict] = {}
        if out.exists() and not a.fresh:
            for line in out.read_text().splitlines():
                r = json.loads(line)
                if a.retry_broken and _broken(r):
                    continue
                done[(r["domain"], r["task_id"])] = r
        todo = [t for t in tasks if t not in done]
        print(f"\n== {model}: {len(tasks)} tasks ({len(done)} already done) → {out}")
        if a.retry_broken and not a.fresh:
            out.write_text("".join(json.dumps(r) + "\n" for r in done.values()))
        try:
            if todo:
                make_agent(model)()  # fail now, not 30 times, if the model can't be set up
        except ModelUnavailable as e:
            print(f"   ⛔ {e}")
            continue
        with out.open("w" if a.fresh else "a") as f, ThreadPoolExecutor(a.workers) as pool:
            futures = {pool.submit(run_one, model, d, t): (d, t) for d, t in todo}
            for fut in as_completed(futures):
                d, t = futures[fut]
                rec = fut.result()
                if _gone(rec):  # removed model / wrong ID: every other task would fail the same way
                    for other in futures:
                        other.cancel()
                    print(f"   ⛔ {model}: model not available, stopping ({rec['error'][:160]})")
                    break
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[(d, t)] = rec
                x = rec["diagnostics"]
                flags = [s for s, on in (("invented IDs", x["unseen_arguments"]),
                                         ("unknown tool", x["unknown_tools"]),
                                         ("retried a failed call", x["repeated_failures"]),
                                         ("no tools ran", not x["tools_ran"]),
                                         ("over 8K", (rec.get("flags") or {}).get("context_overflow")),
                                         (f"BROKEN RUN ({rec['termination']}): {rec.get('error') or ''}",
                                          _broken(rec))) if on]
                print(f"   {'✅' if rec['outcome']['passed'] else '❌'} {d}:{t:<6} "
                      f"{x['tools_ran']:>2} tools  {rec['wall_s']:>5.0f}s  {'; '.join(flags)}")
        rows.append(summarise(model, [done[t] for t in tasks if t in done]))
        passed_by[model] = {t for t in tasks if t in done and done[t]["outcome"]["passed"]}
    print_summary(rows)
    print_overlap(passed_by)
    return 0


if __name__ == "__main__":
    sys.exit(main())
