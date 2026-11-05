"""Check our outcome checker (src/reflexive/outcome.py) against τ²-bench (SEL-39).

    python scripts/tau2/check_scorer.py run <save-to name or results.json>   # vs an official run
    python scripts/tau2/check_scorer.py gold [retail airline telecom]        # no API, all tasks

`run` re-scores every conversation in a saved `tau2 run` with our wrapper, in both message
formats (τ²-bench's own and the OpenAI-style one our harness writes), and compares with the
official score. Checks graded by an LLM judge (retail NL assertions, airline communicate) are
not re-run unless --judges is given: they cost money and can vary between calls, so only the
deterministic checks are compared for those tasks. Exits 1 on any mismatch.

`gold` replays each task's correct actions (must pass) and the same with the last
state-changing action removed (should fail). Takes ~10 minutes for all three domains.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from loguru import logger

from reflexive.outcome import JUDGED, scorer, to_openai

logger.remove()
logger.add(sys.stderr, level="ERROR")


def load_results(name: str):
    from tau2.data_model.simulation import Results
    from tau2.utils.utils import DATA_DIR

    for p in (pathlib.Path(name), pathlib.Path(DATA_DIR) / "simulations" / name):
        f = p / "results.json" if p.is_dir() else p
        if f.is_file():
            return Results.model_validate_json(f.read_text()), f
    sys.exit(f"No results.json for {name!r} (looked in . and {DATA_DIR}/simulations/)")


def deterministic(info) -> dict:
    """The parts of a RewardInfo that don't depend on an LLM judge."""
    db = info.db_check.db_match if info.db_check else None
    return {
        "db_match": db,
        "env_assertions": [c.met for c in info.env_assertions or []],
        "action_checks": [c.action_match for c in info.action_checks or []],
        "breakdown": {getattr(k, "value", k): v for k, v in (info.reward_breakdown or {}).items()
                      if getattr(k, "value", k) not in JUDGED},
    }


def check_run(name: str, judges: bool) -> int:
    results, path = load_results(name)
    domain = results.info.environment_info.domain_name
    sc = scorer(domain, results.info.language or "en")
    print(f"{path}  domain={domain}  {len(results.simulations)} conversation(s)")
    print(f"{'task':<70} {'official':>8} {'ours':>6} {'openai':>6}  match")
    bad = 0
    for sim in results.simulations:
        official = sim.reward_info
        task = sc.task(sim.task_id)
        basis = {b.value for b in task.evaluation_criteria.reward_basis}
        judged = bool(basis & JUDGED) and not judges
        ours = sc.score(sim.task_id, sim.messages, sim.termination_reason.value, judges=judges)
        oai = sc.score(sim.task_id, to_openai(sim.messages), sim.termination_reason.value,
                       judges=judges)
        same = all(deterministic(o.reward_info) == deterministic(official) for o in (ours, oai))
        if not judged:
            same = same and ours.reward == oai.reward == official.reward
        bad += not same
        tag = "✓" if same else "✗ MISMATCH"
        if judged:
            tag += f"  (deterministic checks only; skipped {', '.join(ours.skipped)})"
        print(f"{sim.task_id[:70]:<70} {official.reward:>8.1f} {ours.reward:>6.1f} "
              f"{oai.reward:>6.1f}  {tag}")
        if not same:
            print(f"   official: {deterministic(official)}\n   ours:     {deterministic(ours.reward_info)}")
    print(f"\n{len(results.simulations) - bad}/{len(results.simulations)} match the official score")
    return 1 if bad else 0


def check_gold(domains: list[str]) -> int:
    failures = 0
    for d in domains:
        sc = scorer(d)
        gold_pass = drop_fail = drop_total = 0
        notes = []
        for tid in sc.tasks:
            o = sc.score(tid, sc.gold_trajectory(tid), "user_stop", judges=False)
            gold_pass += o.passed
            if not o.passed:
                notes.append(f"correct actions FAILED: task {tid} {o.checks}")
            writes = sc.write_action_indices(tid)
            if writes:
                drop_total += 1
                o2 = sc.score(tid, sc.gold_trajectory(tid, drop=writes[-1]), "user_stop",
                              judges=False)
                drop_fail += not o2.passed
                if o2.passed:
                    notes.append(f"still passes without its last change: task {tid}")
        n = len(sc.tasks)
        failures += n - gold_pass
        print(f"{d:<8} correct actions pass: {gold_pass}/{n}   "
              f"last change removed → fail: {drop_fail}/{drop_total}")
        for line in notes:
            print(f"         {line}")
    return 1 if failures else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="compare with a saved official run")
    r.add_argument("name")
    r.add_argument("--judges", action="store_true", help="also re-run LLM judges (OpenAI key, costs money)")
    g = sub.add_parser("gold", help="correct-action replay on every task (no API)")
    g.add_argument("domains", nargs="*", default=["retail", "airline", "telecom"])
    a = ap.parse_args()
    sys.exit(check_run(a.name, a.judges) if a.cmd == "run" else check_gold(a.domains))


if __name__ == "__main__":
    main()
