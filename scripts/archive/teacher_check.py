"""Teacher cost and latency on 3 τ²-bench tasks, plus a budget estimate (SEL-37).

    python scripts/archive/teacher_check.py [--mode prompted|native] [--tasks retail:0 airline:3 ...]

Runs the teacher (Aya Expanse 32B via Cohere) as the agent through `Tau2Episode`, one train
task per domain by default, with GPT-4.1 as the simulated customer (as in every run). For each
conversation it reports how it ended, the score, the teacher's calls / tokens / estimated cost,
the customer's cost and the wall time. "turns" are teacher API calls; "tools" are the agent tool
calls τ²-bench actually ran (a turn can make several, and a message makes none). It then
projects the SEL-37 budget. Conversations are saved to `results/episodes/teacher-check.jsonl`.

Scoring: telecom needs no judge. Retail and airline are scored with judges only when
OPENAI_API_KEY is set; otherwise their deterministic checks only (marked "no judge").
No client-side rate limit by default (production key); set TEACHER_RPM to add one (trial keys: 20).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

from loguru import logger

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "src"))
from reflexive.episode import Tau2Episode  # noqa: E402
from reflexive.teacher import EmptyTeacherReply, TeacherClient, unseen_arguments  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="ERROR")

N_FIXES = 200          # SEL-37: ~200 teacher fixes
TEST_ROLLOUTS = 100 * 4 * 3  # 100 test tasks x 4 arms x 3 samples (customer cost only)


def first_train_task(domain: str) -> str:
    from tau2.registry import registry

    return registry.get_task_splits_loader(domain)()["train"][0]


def run(domain: str, task_id: str, mode: str, style: str, judges: bool, hints: str) -> dict:
    teacher = TeacherClient(mode, style=style)
    ep = Tau2Episode(domain, task_id, max_context=128_000, reserve=4_096,  # Aya: 128K / 4K out
                     hints=hints)
    start = time.monotonic()
    ep.reset()
    error = None
    while not ep.done:
        messages = ep.prompt()
        if ep.done:
            break
        try:
            ep.step(teacher.turn(messages, ep.tool_schemas))
        except EmptyTeacherReply as e:  # end cleanly and record why, rather than crash
            error = str(e)
            ep.finish()
    rec = ep.record(judges=judges, agent_llm=f"cohere/{teacher.model}", teacher_mode=mode,
                    teacher_style=style, teacher_usage=teacher.usage.as_dict(), teacher_error=error)
    rec["wall_s"] = round(time.monotonic() - start, 1)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["prompted", "native"], default="prompted")
    ap.add_argument("--style", choices=["cohere", "agenttune"], default="cohere",
                    help="prompted tool-call format: Aya's own (cohere) or AgentTune's fallback")
    ap.add_argument("--tasks", nargs="*", help="domain:task_id (default: first train task per domain)")
    ap.add_argument("--hints", choices=["none", "steps", "steps_with_args"], default="none",
                    help="add tau2's resolution steps to the teacher's prompt (train tasks only)")
    ap.add_argument("--out", default=None,
                    help="default: results/episodes/teacher-check-<style>[-<hints>].jsonl")
    a = ap.parse_args()
    tasks = [t.split(":", 1) for t in a.tasks] if a.tasks else [
        (d, first_train_task(d)) for d in ("retail", "airline", "telecom")]
    have_judges = bool(os.environ.get("OPENAI_API_KEY"))
    tag = a.style if a.hints == "none" else f"{a.style}-{a.hints}"
    out = pathlib.Path(a.out or f"results/episodes/teacher-check-{tag}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    rpm = os.environ.get("TEACHER_RPM") or "no limit"
    print(f"teacher=c4ai-aya-expanse-32b ({a.mode}, {a.style} style, hints={a.hints})  rpm={rpm}  "
          f"judges={'on' if have_judges else 'telecom only (no OPENAI_API_KEY)'}  → {out}")
    print(f"{'task':<40} {'ended':<14} {'reward':>6} {'turns':>5} {'tools':>5} {'tok in/out':>14} "
          f"{'teacher$':>8} {'cust$':>6} {'time':>6}")
    rows = []
    with out.open("w") as f:
        for domain, tid in tasks:
            judges = have_judges or domain == "telecom"
            rec = run(domain, tid, a.mode, a.style, judges, a.hints)
            rec["unseen_arguments"] = unseen_arguments(rec["messages"])
            f.write(json.dumps(rec) + "\n")
            rows.append(rec)
            u = rec["teacher_usage"]
            tag = "" if rec["outcome"]["judged"] or domain == "telecom" else " (no judge)"
            tools = sum(len(m.get("tool_calls") or []) for m in rec["messages"]
                        if m["role"] == "assistant")  # agent tool calls τ²-bench actually ran
            print(f"{(domain + ':' + tid)[:40]:<40} {rec['termination'][:14]:<14} "
                  f"{rec['outcome']['reward']:>6.1f} {u['calls']:>5} {tools:>5} "
                  f"{u['prompt_tokens']:>7,}/{u['completion_tokens']:<6,} {u['est_cost']:>8.4f} "
                  f"{(rec['user_cost'] or 0):>6.3f} {rec['wall_s']:>5.0f}s{tag}")
            if u["unparsed"]:
                print(f"   ⚠ {u['unparsed']} reply(ies) looked like a tool call but didn't parse")
            if u.get("embedded"):
                print(f"   ⚠ {u['embedded']} tool call(s) were written inside prose (prose dropped)")
            if u.get("empty"):
                print(f"   ⚠ {u['empty']} empty reply(ies) re-asked")
            if rec.get("teacher_error"):
                print(f"   ⚠ ended early: {rec['teacher_error']}")
            unseen = rec["unseen_arguments"]
            if unseen:
                print(f"   ⚠ ID-like arguments never seen earlier (likely invented): {unseen[:5]}")
            if rec.get("hints") != "none" and not rec.get("hinted"):
                print("   (no resolution steps for this task: it expects no actions)")
            if not tools:
                print("   ⚠ no agent tool calls ran: a pass here can only mean 'leave the database as is'")

    n = len(rows)
    teacher_c = sum(r["teacher_usage"]["est_cost"] for r in rows) / n
    cust_c = sum(r["user_cost"] or 0 for r in rows) / n
    calls = sum(r["teacher_usage"]["calls"] for r in rows) / n
    secs = sum(r["wall_s"] for r in rows) / n
    print(f"\nPer conversation (mean of {n}): teacher ${teacher_c:.4f} · customer ${cust_c:.4f} · "
          f"{calls:.0f} teacher calls · {secs:.0f}s")
    print("Budget estimate (SEL-37):")
    print(f"  {N_FIXES} teacher fixes:        ${N_FIXES * (teacher_c + cust_c):7.2f}  "
          f"(~{N_FIXES * calls:,.0f} Cohere calls)")
    print(f"  Test pass, all arms ×3: ${TEST_ROLLOUTS * cust_c:7.2f}  "
          f"(customer only: {TEST_ROLLOUTS:,} conversations; the student runs on our GPU)")
    print("  Teacher prices are estimates (listed $0.50/$1.50 per 1M tokens); check the Cohere dashboard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
