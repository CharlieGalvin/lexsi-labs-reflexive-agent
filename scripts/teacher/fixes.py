"""Teacher fix run: the chosen teacher on the train tasks, up to 3 attempts each (SEL-58 phase 2, SEL-45).

    python scripts/teacher/fixes.py --list                       # the 148 tasks, no API calls
    python scripts/teacher/fixes.py --limit 3                    # smoke test: 3 tasks x 3 attempts
    python scripts/teacher/fixes.py --tasks all                  # all 178 (the SEL-45 run, GLM-5.3-Flash)
    python scripts/teacher/fixes.py --report                     # summary of what's saved

**Tasks.** τ²-bench *train* tasks in retail, airline and telecom: by default the 148 that phase 1
didn't use (54 retail + 20 airline + 74 telecom); ``--tasks all`` adds phase 1's 30, for all 178.
Train tasks only, so every verified pass is a real fix.

**Attempts.** Up to 3 per task, stopping at the first pass: attempt 1 at temperature 0, retries
at 0.7 so they actually differ. pass@3 = a fix exists. ``--all-attempts`` runs all 3 even after
a pass (~2.5x the cost), which also gives pass^3 (consistency) and up to 3 fixes per task.

**Scoring.** τ²-bench's full evaluator, judges on: retail's NL assertions and airline's
communicate check need ``OPENAI_API_KEY``. Telecom needs no judges. The database-only result is
kept too, to show how much the judges change. A failed judge call keeps the conversation and
marks it ``judge_error`` for re-judging, rather than losing it.

**Accounts.** The teacher and the simulated customer (GPT-4.1) go through OpenRouter (~$11); the
judges through OpenAI (~$2), as τ²-bench pins them. ``--customer openai`` uses OpenAI's exact
GPT-4.1 snapshot instead, but a new OpenAI account's rate limit (10K tokens/min) is far too low.
If judge calls hit a rate limit, the conversation is kept: run ``--rejudge`` afterwards.

Records go to ``results/fixes/<model>.jsonl``, one line per (task, attempt). An interrupted run
resumes where it stopped; ``--retry-broken`` redoes broken conversations.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from reflexive import runner
from reflexive.runner import Record, Task, is_broken
from reflexive.stats import wilson

TEMPERATURES = [0.0, 0.7, 0.7]  # attempt 1, 2, 3
# The customer's replies are a few sentences. Without a cap, OpenRouter reserves GPT-4.1's full
# 32K-token output price for every request in flight, and 8 long telecom chats at once use up a
# small balance's in-flight budget (HTTP 402 "in_flight_budget_exhausted").
CUSTOMER_ARGS = {"temperature": 0.0, "max_tokens": 2048}
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


def remaining_train_tasks() -> list[Task]:
    """Every train task in the three domains, minus phase 1's fixed 30, in a stable order."""
    from tau2.registry import registry

    used = set(runner.sample_tasks())
    out = []
    for d in DOMAINS:
        out += [(d, t) for t in sorted(registry.get_task_splits_loader(d)()["train"],
                                       key=lambda x: (len(x), x)) if (d, t) not in used]
    return out


def db_only_pass(rec: Record) -> bool:
    """Would it pass on the deterministic checks alone? (The phase 1 measure.)"""
    if rec.get("termination") not in ("agent_stop", "user_stop"):
        return False
    checks = {k: v for k, v in (rec["outcome"].get("checks") or {}).items() if k not in JUDGED}
    return bool(checks) and all(v == 1.0 for v in checks.values())


# --- Report -------------------------------------------------------------------------------------


def report(recs: list[Record], attempts: int = 3) -> dict[str, dict[str, Any]]:
    """Pass rates, fixes, broken runs and cost per domain, plus \"all\"."""
    by_task: dict[Task, list[Record]] = {}
    for r in recs:
        by_task.setdefault((r["domain"], r["task_id"]), []).append(r)
    rows = {}
    for d in (*DOMAINS, "all"):
        tasks = {k: v for k, v in by_task.items() if d in ("all", k[0])}
        if not tasks:
            continue
        convs = [r for v in tasks.values() for r in v]
        finished = {k: v for k, v in tasks.items()
                    if len(v) >= attempts or any(r["outcome"]["passed"] for r in v)}
        complete = {k: v for k, v in tasks.items() if len(v) >= attempts
                    and all(r.get("attempt_mode", "all") == "all" for r in v)}
        rows[d] = {
            "tasks": len(tasks), "complete_tasks": len(complete), "conversations": len(convs),
            "pass1": _rate([r["outcome"]["passed"] for k, v in tasks.items()
                            for r in v if r.get("attempt") == 1]),
            "pass_at_3": _rate([any(r["outcome"]["passed"] for r in v) for v in finished.values()]),
            "pass_all_3": _rate([all(r["outcome"]["passed"] for r in v) for v in complete.values()]),
            "db_only_pass1": _rate([db_only_pass(r) for v in tasks.values() for r in v
                                    if r.get("attempt") == 1]),
            "fixes": sum(r["outcome"]["passed"] and not is_broken(r) for r in convs),
            "clean_fixes": sum(_clean(r) for r in convs),
            "broken": sum(is_broken(r) for r in convs),
            "judge_errors": sum(bool(r.get("judge_error")) for r in convs),
            "cost": sum((r["agent_usage"].get("est_cost") or 0) + (r.get("user_cost") or 0) for r in convs),
        }
    return rows


def _rate(xs: list[bool]) -> tuple[int, int]:
    return sum(xs), len(xs)


def _clean(rec: Record) -> bool:
    """A passing conversation with no retried failed calls and no unknown tools.

    Invented-ID flags aren't used yet: in phase 1 every flag for the top teachers was a date
    worked out from the conversation (SEL-9 will make the check ignore dates).
    """
    x = rec["diagnostics"]
    return (rec["outcome"]["passed"] and not is_broken(rec) and not rec.get("judge_error")
            and not x["repeated_failures"] and not x["unknown_tools"])


def needs_judging(rec: Record) -> bool:
    """Scored without its judged checks (a judge call failed, or the run used --no-judges)."""
    if is_broken(rec) and not rec.get("judge_error"):
        return False
    return bool(rec.get("judge_error")) or bool(rec["outcome"].get("skipped"))


def rejudge(path: pathlib.Path, workers: int = 2, tries: int = 6) -> tuple[int, int]:
    """Score again, judges on, every saved conversation that's missing its judged checks.

    Judges run one call per conversation, so a couple at a time stays inside small OpenAI
    rate limits; rate-limit errors are retried with a growing wait.
    """
    import time

    from reflexive.outcome import score

    recs = [json.loads(x) for x in path.read_text().splitlines()]
    todo = [r for r in recs if needs_judging(r)]

    def one(r: Record) -> bool:
        for k in range(tries):
            try:
                o = score(r["domain"], r["task_id"], r["messages"], r["termination"], judges=True)
            except Exception as e:  # noqa: BLE001 - retry rate limits, give up on the rest
                if "rate" in str(e).lower() and k < tries - 1:
                    time.sleep(5 * 2 ** k)
                    continue
                r["judge_error"] = f"{type(e).__name__}: {str(e)[:300]}"
                return False
            r["outcome"] = {"passed": o.passed, "reward": o.reward, "checks": o.checks,
                            "judged": o.judged, "skipped": o.skipped, "note": o.note}
            r.pop("judge_error", None)
            return True
        return False

    with ThreadPoolExecutor(workers) as pool:
        ok = sum(pool.map(one, todo))
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in recs))
    tmp.replace(path)
    return ok, len(todo)


def print_report(model: str, rows: dict[str, dict[str, Any]]) -> None:
    def pct(kn: tuple[int, int]) -> str:
        k, n = kn
        lo, hi = wilson(k, n)
        return f"{k:>3}/{n:<3} ({k / n:>4.0%}, {lo:.0%}–{hi:.0%})" if n else "      -       "

    print(f"\n=== {model}: train tasks (judges on) ===")
    print(f"{'domain':<8} {'pass@1 (judged)':>24} {'pass@1 (DB only)':>24} {'pass@3: a fix exists':>24} "
          f"{'pass^3: all 3 pass':>24} {'clean fixes':>11} {'broken':>6} {'cost':>7}")
    for d, s in rows.items():
        print(f"{d:<8} {pct(s['pass1']):>24} {pct(s['db_only_pass1']):>24} {pct(s['pass_at_3']):>24} "
              f"{pct(s['pass_all_3']):>24} {s['clean_fixes']:>11} {s['broken']:>6} ${s['cost']:>6.2f}")
    a = rows.get("all", {})
    if a.get("judge_errors"):
        print(f"⚠ {a['judge_errors']} conversations have a failed judge call: run with --rejudge.")
    if a.get("broken"):
        print(f"⚠ {a['broken']} broken conversations (API failures): rerun with --retry-broken.")
    print("pass@1 = attempt 1 (temperature 0). pass@3 = at least one of 3 attempts passed. "
          "Cost = teacher + simulated customer (judges not included).")


# --- Running ------------------------------------------------------------------------------------

Attempt = tuple[str, str, int]  # (domain, task_id, attempt number)


def pick_tasks(which: str, domains: list[str], *, sample: int | None = None,
               limit: int | None = None) -> list[Task]:
    """The tasks to run: phase 1's rest, its 30, or all; then a sample per domain, then a limit."""
    pool = {"rest": remaining_train_tasks, "phase1": runner.sample_tasks,
            "all": lambda: runner.sample_tasks() + remaining_train_tasks()}[which]()
    tasks = [t for t in pool if t[0] in domains]
    if sample:
        tasks = [t for d in DOMAINS for t in random.Random(0).sample(
            [x for x in tasks if x[0] == d], min(sample, sum(x[0] == d for x in tasks)))]
    return tasks[:limit] if limit else tasks


def load_done(out: pathlib.Path, *, retry_broken: bool) -> dict[Attempt, Record]:
    """The attempts already saved (all of them, or only unbroken ones with --retry-broken)."""
    done: dict[Attempt, Record] = {}
    if not out.exists():
        return done
    for line in out.read_text().splitlines():
        r = json.loads(line)
        if not (retry_broken and is_broken(r)):
            done[(r["domain"], r["task_id"], r["attempt"])] = r
    if retry_broken:  # rewrite the file without the broken attempts about to be redone
        out.write_text("".join(json.dumps(r) + "\n" for r in done.values()))
    return done


def _passed(done: dict[Attempt, Record], d: str, t: str) -> bool:
    return any(r["outcome"]["passed"] for (dd, tt, _), r in done.items() if (dd, tt) == (d, t))


def ready_to_run(model: str, customer: str, *, judges: bool, todo: bool) -> bool:
    """Credits, the model, the customer and the judges all work: checked before any spend."""
    left = runner.openrouter_credits()
    if left is not None:
        print(f"   OpenRouter credits left: ${left:.2f}")
        if left < runner.MIN_CREDITS:
            print("   ⛔ add credits at https://openrouter.ai/settings/credits first")
            return False
    print(f"   simulated customer: {customer}; judges: {', '.join(judge_models()) if judges else 'off'}")
    if not todo:
        return True
    runner.make_agent(model)()  # fail now if the model can't be set up
    problems = preflight([customer, *(judge_models() if judges else [])])
    if problems:
        print("   ⛔ not started; a test call failed:")
        for p in problems:
            print(f"      {p}")
        return False
    return True


def _note(rec: Record) -> str:
    o = rec["outcome"]
    return "; ".join(s for s, on in (
        ("judge error", rec.get("judge_error")),
        (f"BROKEN ({rec.get('error') or rec['termination']})"[:120], is_broken(rec)),
        ("DB ok, judge failed", o.get("judged") and not o["passed"] and db_only_pass(rec))) if on)


def run_attempts(a: argparse.Namespace, tasks: list[Task], done: dict[Attempt, Record],
                 out: pathlib.Path, customer: str) -> None:
    """One round per attempt number; a task's retries run only if it hasn't passed yet."""
    mode = "all" if a.all_attempts else "until_pass"
    n = 0
    with out.open("a") as f, ThreadPoolExecutor(a.workers) as pool:
        for i in range(1, a.attempts + 1):
            round_todo = [(d, t) for d, t in tasks if (d, t, i) not in done
                          and (a.all_attempts or not _passed(done, d, t))]
            if not round_todo:
                continue
            temperature = TEMPERATURES[min(i, 3) - 1]
            print(f"-- attempt {i} (temperature {temperature}): {len(round_todo)} tasks")
            futures = {pool.submit(runner.run_one, a.model, d, t, temperature=temperature,
                                   judges=not a.no_judges, provider=a.provider,
                                   user_llm=customer, user_llm_args=CUSTOMER_ARGS):
                       (d, t) for d, t in round_todo}
            for fut in as_completed(futures):
                d, t = futures[fut]
                rec = fut.result()
                if runner.out_of_credits(rec) or runner.model_gone(rec):
                    for other in futures:
                        other.cancel()
                    print(f"   ⛔ stopping: {rec['error'][:200]}. Fix it, then run the same command again.")
                    return
                n += 1
                rec["attempt"], rec["attempt_mode"] = i, mode
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[(d, t, i)] = rec
                short = t if len(t) <= 40 else t[:37] + "..."  # telecom IDs are long
                print(f"   [{n:>3}] {'✅' if rec['outcome']['passed'] else '❌'} {d}:{short:<5} #{i}  "
                      f"{rec['diagnostics']['tools_ran']:>2} tools {rec['wall_s']:>5.0f}s  {_note(rec)}")


# --- Main ---------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="glm53f", help="a model name from reflexive.runner (default glm53f)")
    ap.add_argument("--attempts", type=int, default=3, help="most attempts per task (default 3)")
    ap.add_argument("--all-attempts", action="store_true",
                    help="run every attempt even after a pass (gives pass^3; ~2.5x the cost)")
    ap.add_argument("--workers", type=int, default=6, help="conversations at once (default 6)")
    ap.add_argument("--provider", help="pin one OpenRouter provider, e.g. 'z-ai' (default: any)")
    ap.add_argument("--no-judges", action="store_true", help="database checks only (no OpenAI key)")
    ap.add_argument("--customer", choices=sorted(CUSTOMER), default=None,
                    help="where the simulated customer (GPT-4.1) runs: openrouter (default) or "
                         "openai (the exact pinned snapshot; needs a high OpenAI rate limit)")
    ap.add_argument("--limit", type=int, help="only the first N tasks (smoke test)")
    ap.add_argument("--sample", type=int,
                    help="a random N tasks per domain (seed 0), e.g. a cheap student check")
    ap.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    ap.add_argument("--tasks", choices=("rest", "phase1", "all"), default="rest",
                    help="rest = the 148 phase 1 didn't use (default); phase1 = its 30; all = 178")
    ap.add_argument("--retry-broken", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the tasks and exit")
    ap.add_argument("--report", action="store_true", help="summarise saved results and exit")
    ap.add_argument("--rejudge", action="store_true",
                    help="score saved conversations that are missing their judged checks, then exit")
    a = ap.parse_args()

    out = pathlib.Path(f"results/fixes/{runner.file_name(a.model)}.jsonl")
    if a.rejudge:
        if not out.exists():
            ap.error(f"nothing saved yet at {out}")
        ok, n = rejudge(out)
        print(f"re-judged {ok}/{n} conversations" + ("" if ok == n else "; run --rejudge again for the rest"))
        a.report = True
    if a.report:
        recs = [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
        print_report(a.model, report(recs, a.attempts))
        return 0

    tasks = pick_tasks(a.tasks, a.domains, sample=a.sample, limit=a.limit)
    if a.list:
        for d in DOMAINS:
            ids = [t for dd, t in tasks if dd == d]
            print(f"{d} ({len(ids)}): {' '.join(ids)}")
        print(f"total: {len(tasks)}")
        return 0
    if not a.no_judges and not os.environ.get("OPENAI_API_KEY"):
        ap.error("judges need OPENAI_API_KEY in .env (or pass --no-judges)")

    from tau2.gym import register_gym_agent

    import reflexive.episode as episode

    register_gym_agent()
    episode._REGISTERED = True

    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out, retry_broken=a.retry_broken)
    todo = [(d, t, i) for d, t in tasks for i in range(1, a.attempts + 1) if (d, t, i) not in done]
    if not a.all_attempts:  # the most it could be; retries only run for tasks without a pass
        todo = [(d, t, i) for d, t, i in todo if i == 1 or not _passed(done, d, t)]
    print(f"== {a.model}: {len(tasks)} tasks × {a.attempts} attempts; {len(done)} done, {len(todo)} to run "
          f"→ {out}")
    customer = CUSTOMER[a.customer or "openrouter"]  # OpenAI's low new-account rate limits stall runs
    if not ready_to_run(a.model, customer, judges=not a.no_judges, todo=bool(todo)):
        return 1
    run_attempts(a, tasks, done, out, customer)
    print_report(a.model, report(list(done.values()), a.attempts))
    return 0

if __name__ == "__main__":
    sys.exit(main())
