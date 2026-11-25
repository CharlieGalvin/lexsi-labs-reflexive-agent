"""Full-conversation fixes (STAGE2.md, amended Sun 22 Nov 14:10): each mismatch fix, played to the end.

    python scripts/heal/continue_fixes.py --limit 2          # try two first
    python scripts/heal/continue_fixes.py                    # all 39 (≈ an hour, no pod)

For every single-turn fix in ``results/heal/mismatch-fixes.jsonl`` (from ``onpolicy_fixes.py
--points mismatch``), the saved conversation it came from is resumed in τ²-bench just before the
flagged turn (τ²-bench replays the earlier tool calls to rebuild the database). The teacher's fixed
turn is sent as the agent's first action, and the teacher (GLM-5.3-Flash, temperature 0) carries the
conversation on with the same customer (GPT-4.1) to its end. One continuation per fix.

A continuation is kept only if τ²-bench passes the **whole** conversation and it is clean
(`golden.why_not_clean`). It becomes a ``reagent.sft/v1`` example in Aya's format (as the
single-turn fixes are), group ``mismatch``, trained on the fixed turn and every agent turn after it,
never on the student's own earlier turns. Train tasks only, never a validation task
(``P-*-val.jsonl``, which must be whole) or a test task.

Writes ``results/heal/mismatch-full-fixes.jsonl`` (kept examples), ``mismatch-full-runs.jsonl``
(every continuation, kept or not, for the audit) and a ``.report.json``. Each result is written as
soon as it's done, so a crash or Ctrl-C keeps what was paid for; re-running skips fixes already run.
"""

from __future__ import annotations

import argparse
import collections
import glob
import importlib.util
import json
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from reflexive.files import ROOT, latest_records, read_jsonl, shown

FIXES = ROOT / "results" / "heal" / "mismatch-fixes.jsonl"
EPISODES = ROOT / "results" / "episodes" / "arms"
OUT = ROOT / "results" / "heal" / "mismatch-full-fixes.jsonl"
RUNS = ROOT / "results" / "heal" / "mismatch-full-runs.jsonl"
Record = dict[str, Any]


def parse_id(fix_id: str) -> dict[str, Any]:
    """``mismatch/<teacher>/<domain>/<task>/<source file>/s<sample>/t<turn>/a<attempt>``."""
    parts = fix_id.split("/")
    if len(parts) < 8 or parts[0] != "mismatch":
        raise ValueError(f"not a mismatch fix id: {fix_id!r}")
    # a task id may itself contain "/": everything between the domain and the source file is the task
    domain, source = parts[2], parts[-4]
    return {
        "teacher": parts[1],
        "domain": domain,
        "task_id": "/".join(parts[3:-4]),
        "source": source,
        "sample": int(parts[-3][1:]),
        "turn": int(parts[-2][1:]),
    }


def cut(rec: Record, turn: int) -> tuple[list[dict], int]:
    """The saved conversation (OpenAI-style ``messages``) up to the agent's turn at ``turn`` in its
    view, and how many agent turns come before it. Raises if the two don't line up."""
    from reflexive.hallucination import agent_view_of

    view = agent_view_of(rec)
    if not 0 < turn < len(view) or view[turn].get("role") != "assistant":
        raise ValueError(f"turn {turn} isn't an agent turn of the saved view")
    k = sum(1 for m in view[:turn] if m.get("role") == "assistant")
    seen = 0
    for i, m in enumerate(rec["messages"]):
        if m.get("role") == "assistant":
            if seen == k:
                names = [tc["function"]["name"] for tc in m.get("tool_calls") or []]
                want = [tc["function"]["name"] for tc in view[turn].get("tool_calls") or []]
                if names != want:
                    raise ValueError(f"agent turn {k} differs between messages and view: {names} vs {want}")
                return rec["messages"][:i], k
            seen += 1
    raise ValueError(f"no agent turn {k} in the saved messages")


def train_from(msgs: list[dict], k: int) -> list[int]:
    """The indices of the agent's turns from its ``k``-th (0-based) on: the fixed turn and after."""
    agent = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    if k >= len(agent):
        raise ValueError(f"only {len(agent)} agent turns, so no turn {k} to train from")
    return agent[k:]


def to_example(fix: Record, info: dict, run: Record, system_prompt: str, k: int) -> Record:
    """The continued conversation as a training example, as the single-turn fixes are formatted."""
    from reflexive import golden
    from reflexive.heal import SFT_FORMAT, domain_tools
    from reflexive.teacher import to_prompted

    tools, _ = domain_tools(info["domain"])
    view = golden.agent_view(run["messages"], system_prompt)
    msgs = [{"role": m["role"], "content": m["content"]} for m in to_prompted(view, tools, "cohere")]
    if sum(m["role"] == "assistant" for m in msgs) != sum(m["role"] == "assistant" for m in view):
        raise ValueError("agent turns don't line up after formatting")
    train_on = train_from(msgs, k)
    return {
        "id": fix["id"].replace("mismatch/", "mismatch-full/", 1),
        "format": SFT_FORMAT,
        "domain": info["domain"],
        "task_id": info["task_id"],
        "group": "mismatch",
        "labels": ["stage2-mismatch-full"],
        "messages": msgs,
        "train_on": train_on,
        "source": {
            "kind": "mismatch_full_fix",
            "fix_id": fix["id"],
            "teacher": info["teacher"],
            "fixed_agent_turn": k,
            "agent_turns_trained": len(train_on),
            "user_llm": run.get("user_llm"),
            "outcome": run["outcome"],
        },
    }


def continue_one(
    fix: Record,
    rec: Record,
    make_teacher: Callable[[], Any],
    episode: Callable[..., Any] | None = None,
) -> tuple[Record | None, Record | None, str]:
    """Play one fix to the end: (the example if kept, the continued run, why)."""
    from reflexive import golden
    from reflexive.hallucination import agent_view_of

    info = parse_id(fix["id"])
    history, k = cut(rec, info["turn"])
    if episode is None:
        from reflexive.episode import Tau2Episode as episode
    ep = episode(
        info["domain"],
        info["task_id"],
        user_llm=rec.get("user_llm"),
        prompt_variant=rec.get("prompt_variant") or "default",
        history=history,
    )
    ep.reset()
    teacher = make_teacher()
    fixed = fix["source"]["fixed_turn_message"]
    ep.step({k2: v for k2, v in fixed.items() if k2 in ("role", "content", "tool_calls")})
    while not ep.done:
        messages = ep.prompt()
        if ep.done:
            break
        reply = teacher.turn(messages, ep.tool_schemas)
        if not reply.get("tool_calls") and not (reply.get("content") or "").strip():
            reply = {"role": "assistant", "content": "Is there anything else I can help you with?"}
        ep.step(reply)
    run = ep.record(judges=True, fix_id=fix["id"], continued_from_agent_turn=k)
    if not run["outcome"]["passed"]:
        return None, run, "the whole conversation didn't pass"
    if (why := golden.why_not_clean(run)) is not None:
        return None, run, f"not clean: {why}"
    system_prompt = agent_view_of(rec)[0]["content"]
    return to_example(fix, info, run, system_prompt, k), run, "kept"


def pending(fixes: list[Record], banned: set[tuple[str, str]], done: set) -> list[Record]:
    """The fixes still to continue; raises if any is on a task that can't be trained on."""
    from reflexive.splits import assert_not_test, split_of

    todo = []
    for fix in fixes:
        info = parse_id(fix["id"])
        d, t = info["domain"], info["task_id"]
        assert_not_test(d, t)
        if (d, t) in banned or split_of(d, t) != "train":
            raise ValueError(f"{fix['id']}: not a train task outside validation")
        if fix["id"] not in done:
            todo.append(fix)
    return todo


def _onpolicy() -> Any:
    spec = importlib.util.spec_from_file_location(
        "onpolicy_fixes", ROOT / "scripts" / "heal" / "onpolicy_fixes.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("onpolicy_fixes", mod)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fixes", type=Path, default=FIXES)
    p.add_argument("--episodes", type=Path, default=EPISODES)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--runs", type=Path, default=RUNS)
    p.add_argument("--exclude", nargs="*", default=sorted(glob.glob(str(EPISODES / "P-*-val.jsonl"))))
    p.add_argument("--limit", type=int, help="only the first N fixes (a trial)")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)

    from loguru import logger

    from reflexive import heal, runner

    logger.remove()
    banned, problems = _onpolicy().val_ids(a.exclude)
    if problems:
        print("⛔ the validation exclusion isn't complete, refusing to risk training on validation tasks:")
        for why in problems:
            print(f"   {why}")
        return 1
    fixes = read_jsonl(a.fixes)[: a.limit] if a.limit else read_jsonl(a.fixes)
    done = {r.get("fix_id") for r in read_jsonl(a.runs, partial_last_ok=True)} if a.runs.exists() else set()
    records: dict[str, dict[tuple[str, int], Record]] = {}

    def record_for(info: dict) -> Record:
        if info["source"] not in records:
            path = a.episodes / f"{info['source']}.jsonl"
            records[info["source"]] = {
                (str(r["task_id"]), r.get("sample", 0)): r for r in latest_records(read_jsonl(path))
            }
        return records[info["source"]][(info["task_id"], info["sample"])]

    def make_teacher() -> Any:
        agent = runner.make_agent(heal.TEACHER)()
        if isinstance(agent, runner.NativeAgent):
            agent.temperature = 0.0
        return agent

    try:
        todo = pending(fixes, banned, done)
    except ValueError as e:
        print(f"⛔ {e}")
        return 1
    print(f"{len(fixes)} fixes, {len(fixes) - len(todo)} already run, {len(todo)} to continue")
    why: collections.Counter = collections.Counter()
    lock = threading.Lock()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    stopped = None
    with a.out.open("a") as out, a.runs.open("a") as runs, ThreadPoolExecutor(a.workers) as pool:

        def one(fix: Record) -> tuple[Record, Record | None, Record | None, str]:
            start = time.monotonic()
            try:
                ex, run, reason = continue_one(fix, record_for(parse_id(fix["id"])), make_teacher)
            except (runner.OutOfCredits, runner.ModelUnavailable):
                raise
            except Exception as e:  # noqa: BLE001 - one broken continuation isn't the run
                ex, run, reason = None, None, f"crashed: {type(e).__name__}: {str(e)[:200]}"
            fix["_seconds"] = round(time.monotonic() - start)
            return fix, ex, run, reason

        futures = [pool.submit(one, f) for f in todo]
        try:
            for n, fut in enumerate(as_completed(futures), 1):
                fix, ex, run, reason = fut.result()
                with lock:
                    why[reason.split(":")[0]] += 1
                    if ex is not None:
                        out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                        out.flush()
                    runs.write(
                        json.dumps(run or {"fix_id": fix["id"], "crashed": reason}, ensure_ascii=False) + "\n"
                    )
                    runs.flush()
                print(
                    f"[{n}/{len(todo)}] {fix['id'].split('/', 2)[2]} → {reason} ({fix['_seconds']}s)",
                    flush=True,
                )
        except (runner.OutOfCredits, runner.ModelUnavailable) as e:
            stopped = f"{type(e).__name__}: {e}"[:300]
            print(f"⛔ {stopped}\n   stopping: what was kept so far is saved; re-run to carry on")
            for f in futures:
                f.cancel()
    kept = len(read_jsonl(a.out)) if a.out.exists() else 0
    report = {
        "fixes": len(fixes),
        "continued_now": len(todo),
        "outcomes_now": dict(why),
        "kept_total": kept,
        "stopped": stopped,
    }
    a.out.with_suffix(".report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"outcomes this run: {dict(why)}\nfull-conversation fixes kept in all: {kept} → {shown(a.out)}")
    return 1 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
