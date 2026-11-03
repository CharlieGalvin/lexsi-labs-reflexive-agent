"""Teacher fix run: the chosen teacher on every train task not used in phase 1 (SEL-58 phase 2, SEL-45).

    python scripts/teacher_fixes.py --list                       # the 148 tasks, no API calls
    python scripts/teacher_fixes.py --limit 3                    # smoke test: 3 tasks x 3 attempts
    python scripts/teacher_fixes.py                              # the full run (GLM-5.3-Flash)
    python scripts/teacher_fixes.py --report                     # summary of what's saved

**Tasks.** Every τ²-bench *train* task in retail, airline and telecom that phase 1 didn't use:
54 retail + 20 airline + 74 telecom = 148. Train tasks only, so every verified pass is a real fix.

**Attempts.** 3 per task: attempt 1 at temperature 0, attempts 2-3 at 0.7 so retries actually
differ. All 3 always run, which gives pass@3 (at least one pass: a fix exists) and pass^3
(all three pass: the teacher is consistent), plus up to 3 different fixes per task.

**Scoring.** τ²-bench's full evaluator, judges on: retail's NL assertions and airline's
communicate check need ``OPENAI_API_KEY``. Telecom needs no judges. The database-only result is
kept too, to show how much the judges change. A failed judge call keeps the conversation and
marks it ``judge_error`` for re-judging, rather than losing it.

**Accounts.** The teacher goes through OpenRouter; the simulated customer (GPT-4.1) and the judges
through OpenAI when ``OPENAI_API_KEY`` is set. Estimated ~$13 for GLM-5.3-Flash on all 148 tasks.

Records go to ``results/fixes/<model>.jsonl``, one line per (task, attempt). An interrupted run
resumes where it stopped; ``--retry-broken`` redoes broken conversations.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location("teacher_sample", HERE / "teacher_sample.py")
ts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ts)

TEMPERATURES = [0.0, 0.7, 0.7]  # attempt 1, 2, 3
CUSTOMER = {"openai": "gpt-4.1-2025-04-14",       # the snapshot τ²-bench pins (needs OPENAI_API_KEY)
            "openrouter": "openrouter/openai/gpt-4.1"}


def judge_models() -> list[str]:
    """The judge models τ²-bench pins: retail NL assertions and airline communicate."""
    from tau2.config import DEFAULT_LLM_NL_ASSERTIONS, DEFAULT_SEMANTIC_COMMUNICATE_MODEL

    return [DEFAULT_LLM_NL_ASSERTIONS, DEFAULT_SEMANTIC_COMMUNICATE_MODEL]


def preflight(models: list[str]) -> list[str]:
    """One tiny call per model before the run: problems found now cost nothing.

    Catches an empty account or a bad key up front. Otherwise every conversation fails within
    seconds and τ²-bench hides the reason inside the simulated customer.
    """
    import litellm

    litellm.suppress_debug_info = True
    problems = []
    for m in dict.fromkeys(models):
        try:
            litellm.completion(model=m, messages=[{"role": "user", "content": "Say ok."}],
                               max_completion_tokens=16, num_retries=1)
        except Exception as e:  # noqa: BLE001 - report every problem at once
            msg = str(e).split("\n")[0][:220]
            hint = (" → add OpenAI credits: https://platform.openai.com/settings/organization/billing"
                    if "credits" in msg.lower() or "quota" in msg.lower() else "")
            problems.append(f"{m}: {type(e).__name__}: {msg}{hint}")
    return problems
JUDGED = {"NL_ASSERTION", "COMMUNICATE"}
DOMAINS = ("retail", "airline", "telecom")


def remaining_train_tasks() -> list[tuple[str, str]]:
    """Every train task in the three domains, minus phase 1's fixed 30, in a stable order."""
    from tau2.registry import registry

    used = set(ts.sample_tasks())
    out = []
    for d in DOMAINS:
        out += [(d, t) for t in sorted(registry.get_task_splits_loader(d)()["train"],
                                       key=lambda x: (len(x), x)) if (d, t) not in used]
    return out


def db_only_pass(rec: dict) -> bool:
    """Would it pass on the deterministic checks alone? (The phase 1 measure.)"""
    if rec.get("termination") not in ("agent_stop", "user_stop"):
        return False
    checks = {k: v for k, v in (rec["outcome"].get("checks") or {}).items() if k not in JUDGED}
    return bool(checks) and all(v == 1.0 for v in checks.values())


# --- Report -------------------------------------------------------------------------------------


def report(recs: list[dict], attempts: int = 3) -> dict:
    by_task: dict[tuple[str, str], list[dict]] = {}
    for r in recs:
        by_task.setdefault((r["domain"], r["task_id"]), []).append(r)
    rows = {}
    for d in (*DOMAINS, "all"):
        tasks = {k: v for k, v in by_task.items() if d in ("all", k[0])}
        if not tasks:
            continue
        convs = [r for v in tasks.values() for r in v]
        complete = {k: v for k, v in tasks.items() if len(v) >= attempts}
        rows[d] = {
            "tasks": len(tasks), "complete_tasks": len(complete), "conversations": len(convs),
            "pass1": _rate([r["outcome"]["passed"] for k, v in tasks.items()
                            for r in v if r.get("attempt") == 1]),
            "pass_at_3": _rate([any(r["outcome"]["passed"] for r in v) for v in complete.values()]),
            "pass_all_3": _rate([all(r["outcome"]["passed"] for r in v) for v in complete.values()]),
            "db_only_pass1": _rate([db_only_pass(r) for v in tasks.values() for r in v
                                    if r.get("attempt") == 1]),
            "fixes": sum(r["outcome"]["passed"] and not ts._broken(r) for r in convs),
            "clean_fixes": sum(_clean(r) for r in convs),
            "broken": sum(ts._broken(r) for r in convs),
            "judge_errors": sum(bool(r.get("judge_error")) for r in convs),
            "cost": sum((r["agent_usage"].get("est_cost") or 0) + (r.get("user_cost") or 0) for r in convs),
        }
    return rows


def _rate(xs: list[bool]) -> tuple[int, int]:
    return sum(xs), len(xs)


def _clean(rec: dict) -> bool:
    """A passing conversation with no retried failed calls and no unknown tools.

    Invented-ID flags aren't used yet: in phase 1 every flag for the top teachers was a date
    worked out from the conversation (SEL-9 will make the check ignore dates).
    """
    x = rec["diagnostics"]
    return (rec["outcome"]["passed"] and not ts._broken(rec) and not rec.get("judge_error")
            and not x["repeated_failures"] and not x["unknown_tools"])


def print_report(model: str, rows: dict) -> None:
    def pct(kn):
        k, n = kn
        lo, hi = ts.wilson(k, n)
        return f"{k:>3}/{n:<3} ({k / n:>4.0%}, {lo:.0%}–{hi:.0%})" if n else "      -       "

    print(f"\n=== {model}: train tasks not used in phase 1 (judges on) ===")
    print(f"{'domain':<8} {'pass@1 (judged)':>24} {'pass@1 (DB only)':>24} {'pass@3: a fix exists':>24} "
          f"{'pass^3: all 3 pass':>24} {'clean fixes':>11} {'broken':>6} {'cost':>7}")
    for d, s in rows.items():
        print(f"{d:<8} {pct(s['pass1']):>24} {pct(s['db_only_pass1']):>24} {pct(s['pass_at_3']):>24} "
              f"{pct(s['pass_all_3']):>24} {s['clean_fixes']:>11} {s['broken']:>6} ${s['cost']:>6.2f}")
    a = rows.get("all", {})
    if a.get("judge_errors"):
        print(f"⚠ {a['judge_errors']} conversations have a failed judge call: rerun with --rejudge later.")
    if a.get("broken"):
        print(f"⚠ {a['broken']} broken conversations (API failures): rerun with --retry-broken.")
    print("pass@1 = attempt 1 (temperature 0). pass@3 = at least one of 3 attempts passed. "
          "Cost = teacher + simulated customer (judges not included).")


# --- Main ---------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="glm53f", help="a teacher_sample model name (default glm53f)")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8, help="conversations at once (default 8)")
    ap.add_argument("--provider", help="pin one OpenRouter provider, e.g. 'z-ai' (default: any)")
    ap.add_argument("--no-judges", action="store_true", help="database checks only (no OpenAI key)")
    ap.add_argument("--customer", choices=sorted(CUSTOMER), default=None,
                    help="where the simulated customer (GPT-4.1) runs: openai (default when "
                         "OPENAI_API_KEY is set) or openrouter")
    ap.add_argument("--limit", type=int, help="only the first N tasks (smoke test)")
    ap.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    ap.add_argument("--retry-broken", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the tasks and exit")
    ap.add_argument("--report", action="store_true", help="summarise saved results and exit")
    a = ap.parse_args()

    out = pathlib.Path(f"results/fixes/{ts._file_name(a.model)}.jsonl")
    if a.report:
        recs = [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
        print_report(a.model, report(recs, a.attempts))
        return 0

    tasks = [t for t in remaining_train_tasks() if t[0] in a.domains]
    tasks = tasks[:a.limit] if a.limit else tasks
    if a.list:
        for d in DOMAINS:
            ids = [t for dd, t in tasks if dd == d]
            print(f"{d} ({len(ids)}): {' '.join(ids)}")
        print(f"total: {len(tasks)}")
        return 0
    if not a.no_judges and not __import__("os").environ.get("OPENAI_API_KEY"):
        ap.error("judges need OPENAI_API_KEY in .env (or pass --no-judges)")

    from tau2.gym import register_gym_agent

    import reflexive.episode as episode

    register_gym_agent()
    episode._REGISTERED = True

    out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[tuple[str, str, int], dict] = {}
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if a.retry_broken and ts._broken(r):
                continue
            done[(r["domain"], r["task_id"], r["attempt"])] = r
        if a.retry_broken:
            out.write_text("".join(json.dumps(r) + "\n" for r in done.values()))
    todo = [(d, t, i) for d, t in tasks for i in range(1, a.attempts + 1) if (d, t, i) not in done]
    print(f"== {a.model}: {len(tasks)} tasks × {a.attempts} attempts; {len(done)} done, {len(todo)} to run "
          f"→ {out}")
    left = ts.openrouter_credits()
    if left is not None:
        print(f"   OpenRouter credits left: ${left:.2f}")
        if left < ts.MIN_CREDITS:
            print("   ⛔ add credits at https://openrouter.ai/settings/credits first")
            return 1
    import os

    customer = CUSTOMER[a.customer or ("openai" if os.environ.get("OPENAI_API_KEY") else "openrouter")]
    print(f"   simulated customer: {customer}; judges: {'off' if a.no_judges else ', '.join(judge_models())}")
    if todo:
        ts.make_agent(a.model)()  # fail now if the model can't be set up
        problems = preflight([customer, *([] if a.no_judges else judge_models())])
        if problems:
            print("   ⛔ not started; a test call failed:")
            for p in problems:
                print(f"      {p}")
            return 1

    with out.open("a") as f, ThreadPoolExecutor(a.workers) as pool:
        futures = {pool.submit(ts.run_one, a.model, d, t, temperature=TEMPERATURES[min(i, 3) - 1],
                               judges=not a.no_judges, provider=a.provider,
                               user_llm=customer): (d, t, i)
                   for d, t, i in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            d, t, i = futures[fut]
            rec = fut.result()
            if ts._no_credits(rec) or ts._gone(rec):
                for other in futures:
                    other.cancel()
                print(f"   ⛔ stopping: {rec['error'][:200]}. Fix it, then run the same command again.")
                break
            rec["attempt"] = i
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[(d, t, i)] = rec
            o, judged = rec["outcome"], rec["outcome"].get("judged")
            note = "; ".join(s for s, on in (
                ("judge error", rec.get("judge_error")),
                (f"BROKEN ({rec.get('error') or rec['termination']})"[:120], ts._broken(rec)),
                ("DB ok, judge failed", judged and not o["passed"] and db_only_pass(rec))) if on)
            short = t if len(t) <= 40 else t[:37] + "..."  # telecom IDs are long
            print(f"   [{n:>3}/{len(todo)}] {'✅' if o['passed'] else '❌'} {d}:{short:<5} #{i}  "
                  f"{rec['diagnostics']['tools_ran']:>2} tools {rec['wall_s']:>5.0f}s  {note}")
    print_report(a.model, report(list(done.values()), a.attempts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
