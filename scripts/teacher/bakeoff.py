"""Teacher bake-off, phase 1: candidate teachers on a fixed 30-task sample (SEL-37, SEL-58).

    python scripts/teacher/bakeoff.py --models bakeoff        # the whole slate
    python scripts/teacher/bakeoff.py --models aya glm53f     # chosen models
    python scripts/teacher/bakeoff.py --models or:moonshotai/kimi-k2.6 --workers 2
    python scripts/teacher/bakeoff.py --list                  # the task sample, no API calls

**Tasks.** A fixed sample of 30 τ²-bench *train* tasks whose correct answer changes the database:
20 retail + 10 airline (seed 0, the same every run). On these, doing nothing can't pass.

**Scoring.** τ²-bench's database / state checks only (no LLM judges, so no OpenAI key needed).
That is a *best case*: failing it is a definite fail, but passing it doesn't guarantee the
judged checks would pass too. Phase 2 (``fixes.py``) scores with the judges on.

**Models** (see ``reflexive.runner``; GPT-4.1 is the simulated customer): the bake-off slate
(``--models bakeoff``), any OpenRouter model as ``or:<slug>``, and the phase 1 references: Aya
Expanse 32B (``aya``, our student), Tiny Aya (``tiny``) and Qwen3 32B (``qwen``).

**Per conversation** it also counts tool hallucinations: ID-like arguments never seen earlier
(invented IDs), calls to tools the agent doesn't have, and identical retries of a failed call.

Records go to `results/episodes/sample-<model>.jsonl`; an interrupted run resumes where it
stopped (finished tasks are skipped). Prints pass rates with 95% confidence ranges (Wilson).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from reflexive import runner
from reflexive.files import resume_jsonl
from reflexive.runner import Record, Task, is_broken
from reflexive.stats import wilson

# --- Summary ------------------------------------------------------------------------------------


def summarise(model: str, recs: list[Record]) -> dict[str, Any]:
    """Pass rate, its range, and conversations with each kind of problem, for one model."""
    n = len(recs)
    k = sum(r["outcome"]["passed"] for r in recs)
    lo, hi = wilson(k, n)
    diag = [r["diagnostics"] for r in recs]
    by_domain = {
        d: (
            sum(r["outcome"]["passed"] for r in recs if r["domain"] == d),
            sum(1 for r in recs if r["domain"] == d),
        )
        for d in runner.SAMPLE
    }
    return {
        "model": model,
        "n": n,
        "passed": k,
        "rate": k / n if n else 0.0,
        "ci": (lo, hi),
        "by_domain": by_domain,
        "invented_ids": sum(bool(x["unseen_arguments"]) for x in diag),
        "unknown_tools": sum(bool(x["unknown_tools"]) for x in diag),
        "repeated_failures": sum(bool(x["repeated_failures"]) for x in diag),
        "no_tools": sum(x["tools_ran"] == 0 for x in diag),
        "overflow": sum(bool((r.get("flags") or {}).get("context_overflow")) for r in recs),
        "errors": sum(is_broken(r) for r in recs),
        "cost": sum(r["agent_usage"]["est_cost"] + (r["user_cost"] or 0) for r in recs),
        "agent_cost": sum(r["agent_usage"]["est_cost"] for r in recs),
        "open_weights": runner.open_weights(model),
        "mean_s": sum(r["wall_s"] for r in recs) / n if n else 0.0,
    }


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\n=== Summary: 30 train tasks that need a state change (DB checks only: a best case) ===")
    print(
        f"{'model':<9} {'open':>4} {'passed':>9} {'95% range':>12} {'retail':>7} {'airline':>8} "
        f"{'invented IDs':>13} {'unknown tool':>13} {'retried fail':>13} {'no tools':>9} "
        f"{'over 8K':>8} {'errors':>7} {'cost':>7} {'$/pass':>7}"
    )
    for s in rows:
        if not s["n"]:
            print(f"{s['model'][:9]:<9}  (no finished runs)")
            continue
        lo, hi = s["ci"]
        dom = s["by_domain"]
        ow = {True: "yes", False: "no", None: "?"}[s["open_weights"]]
        per = f"${s['agent_cost'] / s['passed']:.3f}" if s["passed"] else "-"
        print(
            f"{s['model'][:9]:<9} {ow:>4} {s['passed']:>3}/{s['n']:<3} ({s['rate']:>4.0%}) "
            f"{lo:>4.0%}–{hi:<4.0%} {dom['retail'][0]:>3}/{dom['retail'][1]:<3} "
            f"{dom['airline'][0]:>3}/{dom['airline'][1]:<4} "
            f"{s['invented_ids']:>9}/{s['n']:<3} {s['unknown_tools']:>9}/{s['n']:<3} "
            f"{s['repeated_failures']:>9}/{s['n']:<3} {s['no_tools']:>5}/{s['n']:<3} "
            f"{s['overflow']:>4}/{s['n']:<3} {s['errors']:>7} ${s['cost']:>6.2f} {per:>7}"
        )
    print("Counts in the hallucination columns are conversations with at least one such event.")
    print(
        "open = weights published (OpenRouter links them on Hugging Face); $/pass = the agent's "
        "own cost per passed task (customer excluded)."
    )
    if any(s["model"] == "tiny" for s in rows):
        print(
            "tiny: cost is the simulated customer only (Tiny Aya's API price isn't published); "
            "'over 8K' = ended by the context guard (estimated tokens)."
        )
    if any(s["errors"] for s in rows):
        print(
            "⚠ 'errors' are broken runs (API failures, crashes): they count as fails above but say "
            "nothing about the model. Fix the cause and re-run with --retry-broken."
        )


def print_overlap(passed_by: dict[str, set[Task]]) -> None:
    """Teacher vs student: the tasks only the teacher solves are the possible fixes."""
    teachers = sorted(
        (m for m in passed_by if m not in runner.STUDENTS or m == "aya"), key=lambda m: -len(passed_by[m])
    )
    pairs = [
        (t, s) for t in teachers for s in runner.STUDENTS if t != s and t in passed_by and s in passed_by
    ]
    if not pairs:
        return
    print("\nPossible teacher fixes (tasks the teacher solves and the student doesn't):")
    for t, s in pairs:
        teacher, student = passed_by[t], passed_by[s]
        fix = sorted(teacher - student)
        print(
            f"  {t[:9]:>9} → {s:<5} {len(fix):>2}   ({len(student - teacher)} the other way, "
            f"{len(teacher & student)} both)  {', '.join(f'{d}:{i}' for d, i in fix) or '-'}"
        )


# --- Running one model --------------------------------------------------------------------------


def load_done(out: pathlib.Path, *, fresh: bool, retry_broken: bool) -> dict[Task, Record]:
    """The conversations already saved for this model (all of them, or only unbroken ones)."""
    if fresh or not out.exists():
        return {}
    done = {}
    for r in resume_jsonl(out):  # survives a half-written last line from a stopped run
        if not (retry_broken and is_broken(r)):
            done[(r["domain"], r["task_id"])] = r
    if retry_broken:  # rewrite the file without the broken conversations about to be redone
        out.write_text("".join(json.dumps(r) + "\n" for r in done.values()))
    return done


def enough_credits(model: str) -> bool:
    """False (with a message) when an OpenRouter model would run out of credits partway."""
    left = runner.openrouter_credits() if runner.uses_openrouter(model) else None
    if left is None:
        return True
    print(f"   OpenRouter credits left: ${left:.2f}")
    if left < runner.MIN_CREDITS:
        print(
            f"   ⛔ under ${runner.MIN_CREDITS:.0f}: add credits at "
            "https://openrouter.ai/settings/credits, then run the same command again"
        )
        return False
    return True


def _flags(rec: Record) -> list[str]:
    x = rec["diagnostics"]
    return [
        s
        for s, on in (
            ("invented IDs", x["unseen_arguments"]),
            ("unknown tool", x["unknown_tools"]),
            ("retried a failed call", x["repeated_failures"]),
            ("no tools ran", not x["tools_ran"]),
            ("over 8K", (rec.get("flags") or {}).get("context_overflow")),
            (f"BROKEN RUN ({rec['termination']}): {rec.get('error') or ''}", is_broken(rec)),
        )
        if on
    ]


def run_model(
    model: str, todo: list[Task], done: dict[Task, Record], out: pathlib.Path, *, workers: int, fresh: bool
) -> bool:
    """Run the missing conversations, saving each as it finishes. False: stop the whole run.

    On a stop, conversations not started yet are cancelled, but the ones already running are
    waited for anyway: those that finish cleanly are saved, so they aren't paid for twice.
    """
    carry_on: bool | None = None  # set when stopping: whether the next model should still run
    with out.open("w" if fresh else "a") as f, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(runner.run_one, model, d, t): (d, t) for d, t in todo}
        for fut in as_completed(futures):
            if fut.cancelled():
                continue
            d, t = futures[fut]
            rec = fut.result()
            if runner.out_of_credits(rec) or runner.model_gone(rec):
                if carry_on is None:
                    for other in futures:
                        other.cancel()
                    if runner.out_of_credits(rec):  # every other call would fail too, for every model
                        print(
                            "   ⛔ OpenRouter is out of credits: stopping. Add credits, then run the "
                            "same command with --retry-broken (finished tasks are kept)"
                        )
                        carry_on = False
                    else:
                        print(f"   ⛔ {model}: model not available, stopping ({rec['error'][:160]})")
                        carry_on = True
                continue  # not saved: re-run it once fixed
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[(d, t)] = rec
            print(
                f"   {'✅' if rec['outcome']['passed'] else '❌'} {d}:{t:<6} "
                f"{rec['diagnostics']['tools_ran']:>2} tools  {rec['wall_s']:>5.0f}s  "
                f"{'; '.join(_flags(rec))}"
            )
    return True if carry_on is None else carry_on


# --- Main ---------------------------------------------------------------------------------------


def _quiet_tau2() -> None:
    """τ²-bench logs every step at INFO through loguru; keep errors only."""
    try:
        from loguru import logger
    except ImportError:  # pragma: no cover - only without τ²-bench
        return
    logger.remove()
    logger.add(sys.stderr, level="ERROR")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["aya", "qwen"],
        help=f"any of {', '.join(runner.MODELS)}; 'bakeoff' for the whole slate; or:<openrouter slug>",
    )
    ap.add_argument("--json", type=pathlib.Path, help="also write the summary rows to this file")
    ap.add_argument("--workers", type=int, default=4, help="conversations run at once per model")
    ap.add_argument("--fresh", action="store_true", help="ignore earlier results and start again")
    ap.add_argument("--list", action="store_true", help="only print the task sample")
    ap.add_argument("--limit", type=int, help="run only the first N tasks (a quick smoke test)")
    ap.add_argument(
        "--retry-broken", action="store_true", help="re-run conversations that broke (API failures, crashes)"
    )
    a = ap.parse_args()
    a.models = runner.resolve_models(a.models)
    bad = [m for m in a.models if m not in runner.MODELS and not m.startswith("or:")]
    if bad:
        ap.error(f"unknown model(s) {bad}; choose from {runner.MODELS}, 'bakeoff' or or:<slug>")

    _quiet_tau2()
    tasks = runner.sample_tasks()  # also loads each domain's tasks once, before the worker threads
    tasks = tasks[: a.limit] if a.limit else tasks
    from tau2.gym import register_gym_agent

    import reflexive.episode as episode

    register_gym_agent()  # once, here, rather than raced by the workers
    episode._REGISTERED = True
    if a.list:
        for d, t in tasks:
            print(d, t)
        return 0

    rows, passed_by = [], {}
    for model in a.models:
        out = pathlib.Path(f"results/episodes/sample-{runner.file_name(model)}.jsonl")
        if (model in runner.SLATE or model.startswith("or:")) and runner._openrouter_models():
            if not runner.openrouter_info(runner.SLATE.get(model) or model[3:])["listed"]:
                print(f"\n== {model}: ⛔ not on OpenRouter's model list, skipped")
                continue
        out.parent.mkdir(parents=True, exist_ok=True)
        done = load_done(out, fresh=a.fresh, retry_broken=a.retry_broken)
        todo = [t for t in tasks if t not in done]
        print(f"\n== {model}: {len(tasks)} tasks ({len(done)} already done) → {out}")
        if todo and not enough_credits(model):
            break
        try:
            if todo:
                runner.make_agent(model)()  # fail now, not 30 times, if the model can't be set up
        except runner.ModelUnavailable as e:
            print(f"   ⛔ {e}")
            continue
        keep_going = run_model(model, todo, done, out, workers=a.workers, fresh=a.fresh)
        rows.append(summarise(model, [done[t] for t in tasks if t in done]))
        passed_by[model] = {t for t in tasks if t in done and done[t]["outcome"]["passed"]}
        if not keep_going:
            break
    print_summary(rows)
    print_overlap(passed_by)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
