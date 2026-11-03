"""Teacher cost and latency on 3 τ²-bench tasks, plus a budget estimate (SEL-37).

    python scripts/teacher_check.py [--mode prompted|native] [--tasks retail:0 airline:3 ...]

Runs the teacher (Aya Expanse 32B via Cohere) as the agent through `Tau2Episode`, one train
task per domain by default, with GPT-4.1 as the simulated customer (as in every run). For each
conversation it reports how it ended, the score, the teacher's calls / tokens / estimated cost,
the customer's cost and the wall time. It then projects the SEL-37 budget. Conversations are
saved to `results/episodes/teacher-check.jsonl`.

Scoring: telecom needs no judge. Retail and airline are scored with judges only when
OPENAI_API_KEY is set; otherwise their deterministic checks only (marked "no judge").
Trial keys allow 20 requests/min; set TEACHER_RPM higher for a production key.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

from loguru import logger

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from reflexive.episode import Tau2Episode  # noqa: E402
from reflexive.teacher import TeacherClient  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="ERROR")

N_FIXES = 200          # SEL-37: ~200 teacher fixes
TEST_ROLLOUTS = 100 * 4 * 3  # 100 test tasks x 4 arms x 3 samples (customer cost only)


def first_train_task(domain: str) -> str:
    from tau2.registry import registry

    return registry.get_task_splits_loader(domain)()["train"][0]


def run(domain: str, task_id: str, mode: str, judges: bool) -> dict:
    teacher = TeacherClient(mode)
    ep = Tau2Episode(domain, task_id, max_context=128_000, reserve=4_096)  # Aya: 128K / 4K out
    start = time.monotonic()
    ep.reset()
    while not ep.done:
        messages = ep.prompt()
        if ep.done:
            break
        ep.step(teacher.turn(messages, ep.tool_schemas))
    rec = ep.record(judges=judges, agent_llm=f"cohere/{teacher.model}", teacher_mode=mode,
                    teacher_usage=teacher.usage.as_dict())
    rec["wall_s"] = round(time.monotonic() - start, 1)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["prompted", "native"], default="prompted")
    ap.add_argument("--tasks", nargs="*", help="domain:task_id (default: first train task per domain)")
    ap.add_argument("--out", default="results/episodes/teacher-check.jsonl")
    a = ap.parse_args()
    tasks = [t.split(":", 1) for t in a.tasks] if a.tasks else [
        (d, first_train_task(d)) for d in ("retail", "airline", "telecom")]
    have_judges = bool(os.environ.get("OPENAI_API_KEY"))
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"teacher=c4ai-aya-expanse-32b ({a.mode})  rpm={os.environ.get('TEACHER_RPM', 20)}  → {out}")
    print(f"{'task':<46} {'ended':<14} {'reward':>6} {'calls':>5} {'tok in/out':>14} "
          f"{'teacher$':>8} {'cust$':>6} {'time':>6}")
    rows = []
    with out.open("w") as f:
        for domain, tid in tasks:
            judges = have_judges or domain == "telecom"
            rec = run(domain, tid, a.mode, judges)
            f.write(json.dumps(rec) + "\n")
            rows.append(rec)
            u = rec["teacher_usage"]
            tag = "" if rec["outcome"]["judged"] or domain == "telecom" else " (no judge)"
            print(f"{(domain + ':' + tid)[:46]:<46} {rec['termination'][:14]:<14} "
                  f"{rec['outcome']['reward']:>6.1f} {u['calls']:>5} "
                  f"{u['prompt_tokens']:>7,}/{u['completion_tokens']:<6,} {u['est_cost']:>8.4f} "
                  f"{(rec['user_cost'] or 0):>6.3f} {rec['wall_s']:>5.0f}s{tag}")
            if u["unparsed"]:
                print(f"   ⚠ {u['unparsed']} reply(ies) looked like a tool call but didn't parse")

    n = len(rows)
    teacher_c = sum(r["teacher_usage"]["est_cost"] for r in rows) / n
    cust_c = sum(r["user_cost"] or 0 for r in rows) / n
    calls = sum(r["teacher_usage"]["calls"] for r in rows) / n
    secs = sum(r["wall_s"] for r in rows) / n
    print(f"\nPer conversation (mean of {n}): teacher ${teacher_c:.4f} · customer ${cust_c:.4f} · "
          f"{calls:.0f} teacher calls · {secs:.0f}s")
    print("Budget estimate (SEL-37):")
    print(f"  {N_FIXES} teacher fixes:        ${N_FIXES * (teacher_c + cust_c):7.2f}  "
          f"(~{N_FIXES * calls:,.0f} Cohere calls; at 20/min ≥ {N_FIXES * calls / 20 / 60:.1f} h)")
    print(f"  Test pass, all arms ×3: ${TEST_ROLLOUTS * cust_c:7.2f}  "
          f"(customer only: {TEST_ROLLOUTS:,} conversations; the student runs on our GPU)")
    print("  Teacher prices are estimates (listed $0.50/$1.50 per 1M tokens); check the Cohere dashboard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
