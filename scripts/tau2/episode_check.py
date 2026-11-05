"""Drive `Tau2Episode` with an API model as the agent, to check the plumbing end to end (SEL-32).

    python scripts/tau2/episode_check.py telecom --task-ids '<id> <id>' [--out results/x.jsonl]

This is a test driver, not our harness: the real agents (Tiny Aya, the teacher) are driven by
the Architect's harness and the Trainer's rollout. Here GPT-4.1 plays the agent with native
function calling, so the run is comparable with the official `tau2 run` of the same tasks.

For each task it prints how the conversation ended, the score (from `reflexive.outcome`), how
many tool calls the agent and the customer made, and the peak prompt size, i.e. how close the
conversation came to Tiny Aya's 8,192-token window (this driver doesn't enforce it).
Each conversation is saved as one JSON line (default `results/episodes/<domain>-check.jsonl`).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from loguru import logger

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "src"))
from reflexive.episode import Tau2Episode  # noqa: E402
from reflexive.llm import default_model  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="ERROR")


def agent_turn(model: str, messages: list[dict], tools: list[dict]) -> dict:
    import litellm

    resp = litellm.completion(model=model, messages=messages, tools=tools, temperature=0.0)
    msg = resp.choices[0].message
    calls = [{"id": tc.id, "type": "function",
              "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
             for tc in (msg.tool_calls or [])]
    if calls:  # τ²-bench's rule: a tool call or a message, not both
        return {"role": "assistant", "content": None, "tool_calls": calls}
    return {"role": "assistant", "content": msg.content or ""}


def run(domain: str, task_id: str, model: str) -> dict:
    ep = Tau2Episode(domain, task_id, max_context=1_000_000)  # measure size, don't enforce it
    ep.reset()
    while not ep.done:
        messages = ep.prompt()
        if ep.done:
            break
        ep.step(agent_turn(model, messages, ep.tool_schemas))
    return ep.record(agent_llm=model)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("domain")
    ap.add_argument("task_ids", nargs="*")
    ap.add_argument("--task-ids", dest="task_ids_str", default="",
                    help="space-separated task IDs in one string (IDs contain | and [ ])")
    ap.add_argument("--agent-llm", default=None, help="default: GPT-4.1 via OpenAI or OpenRouter")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.task_ids = [*a.task_ids, *a.task_ids_str.split()]
    if not a.task_ids:
        ap.error("give at least one task ID")
    model = a.agent_llm or default_model()
    out = pathlib.Path(a.out or f"results/episodes/{a.domain}-check.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"agent={model}  domain={a.domain}  → {out}")
    print(f"{'task':<62} {'ended':<24} {'reward':>6} {'agent/cust calls':>16} {'peak tok':>9}")
    rows = []
    with out.open("w") as f:
        for tid in a.task_ids:
            rec = run(a.domain, tid, model)
            f.write(json.dumps(rec) + "\n")
            msgs = rec["messages"]
            n_agent = sum(len(m.get("tool_calls") or []) for m in msgs if m["role"] == "assistant")
            n_cust = sum(len(m.get("tool_calls") or []) for m in msgs if m["role"] == "user")
            rows.append(rec)
            print(f"{tid[:62]:<62} {rec['termination']:<24} {rec['outcome']['reward']:>6.1f} "
                  f"{n_agent:>8}/{n_cust:<7} {rec['flags']['peak_prompt_tokens']:>9,}")
    passed = sum(r["outcome"]["passed"] for r in rows)
    over = sum(r["flags"]["peak_prompt_tokens"] > 8192 for r in rows)
    print(f"\n{passed}/{len(rows)} passed · {over}/{len(rows)} would exceed Tiny Aya's 8,192 tokens")


if __name__ == "__main__":
    main()
