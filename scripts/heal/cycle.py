"""Run heal cycles end to end: rollout → classify → teacher fix → retrain → gate → swap (SEL-47).

    python scripts/heal/cycle.py --fixes-only --domains telecom --limit 5   # today: Cohere's Aya
    python scripts/heal/cycle.py --fixes-only --sample 3                     # 3 random per domain
    python scripts/heal/cycle.py --config configs/sft/arms.base.yaml \\
        --train-cmd "<your training command> {data} {out} {base}" --load-adapter
    python scripts/heal/cycle.py ... --cycles 3                              # SEL-23

Run it with ``just heal ...``, which loads the API keys from ``.env``.

**What it runs.** The student from ``--config``'s arm (default Arm A, `configs/arms.yaml`) on the
train tasks of ``--domains``: all of them, the first ``--limit`` per domain, or a random
``--sample`` per domain (seed ``--seed``, default 0, so a rerun picks the same tasks; the first
tasks of a split are often one family, e.g. telecom's MMS issue). Failures go to `heal.PrefixTeacher`
(GLM-5.3-Flash via OpenRouter, fixing the failing turn from the agent's view), the accepted fixes
to `heal.CommandTrainer` and the candidate adapter to the agreed gate (`gate.DevGate`,
`configs/gate.yaml`): both adapters on **every dev task** (36), oracle-scored with judges on.

**Two modes.**

* ``--fixes-only``: rollout, classify and fix, then stop: nothing is trained or gated. Works with
  the student behind Cohere's API, which can't load an adapter. Checks the teacher half of the
  loop and writes the fixes (``fixes.jsonl``) a training run can use.
* Full cycle: needs the student served by vLLM (SEL-63, `scripts/pod/serve.sh`) with a config
  pointing ``api_base`` at it. ``--train-cmd`` trains the adapter: ``{data}`` (the fixes, as
  ``reagent.sft/v1``), ``{out}`` (where the adapter must be written), ``{name}`` and ``{base}``
  (the adapter it starts from; empty for the base student) are filled in. One GPU can't train and
  serve at once, so the command is expected to stop vLLM, train, and start it again.
  ``--load-adapter`` then registers ``{out}`` with the running vLLM under ``{name}``
  (``/v1/load_lora_adapter``; vLLM needs ``--enable-lora`` and
  ``VLLM_ALLOW_RUNTIME_LORA_UPDATING=True``), and the gate runs the student as model ``{name}``.

**What it writes**, in ``--out`` (default ``results/heal/<time>``):

* ``audit.jsonl``: one record per cycle (`heal.audit_record`): failures, every teacher attempt,
  fixes accepted, training, gate decision, adapter before / candidate / after;
* ``cycle-<n>.episodes.jsonl``: every conversation of the cycle, tagged ``heal_role`` =
  ``heal``, ``gate_baseline`` or ``gate_candidate``;
* ``runs/``: one folder per candidate adapter, with its ``fixes.jsonl`` and ``train.log``.

The gate's decisions are also appended to ``--gate-history`` (default
``results/heal/gate_history.jsonl``), so its budget of ``max_cycles`` holds across runs.

**Cost.** The student and the customer (GPT-4.1) per conversation, as in `just rollout` (Arm A on
telecom train, 74 tasks: $11.25); the teacher, about $0.001 a failure per attempt; the judges.
A full cycle runs the dev tasks twice.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reflexive import splits
from reflexive.agent import Adapter, CycleReport, Record, ReflexiveAgent, Task, arm_rollout
from reflexive.arms import get_arm, make_student
from reflexive.heal import CommandTrainer, PrefixTeacher, write_audit

DOMAINS = ("retail", "airline", "telecom")
COHERE = "api.cohere.ai"


def heal_tasks(
    domains: list[str], limit: int | None = None, sample: int | None = None, seed: int = 0
) -> list[Task]:
    """The train tasks to heal on: all, the first ``limit``, or a random ``sample`` per domain.

    A sample keeps the split's order and is the same for the same seed.
    """
    if limit is not None and sample is not None:
        raise ValueError("pass --limit or --sample, not both")
    out = []
    for d in domains:
        ids = splits.task_ids(d, "train")
        if sample is not None:
            picked = set(random.Random(f"{seed}:{d}").sample(ids, min(sample, len(ids))))
            ids = [t for t in ids if t in picked]
        out += [(d, t) for t in ids[:limit]]
    return out


def gate_tasks() -> list[Task]:
    """Every dev task: the gate's rule needs all of them (``min_tasks``)."""
    return [(d, t) for d in DOMAINS for t in splits.task_ids(d, "dev")]


def student_factory(arm: Any) -> Any:
    """``make_agent(adapter)`` for `arm_rollout`: the arm's student, as model ``adapter`` if given."""
    cohere = COHERE in str(arm.student.get("api_base"))

    def make(adapter: Adapter) -> Any:
        if adapter is None:
            return make_student(arm.student)
        if cohere:
            raise RuntimeError(
                f"adapter {adapter!r}: the student behind Cohere's API can't load an adapter; "
                "serve it with vLLM (scripts/pod/serve.sh) and pass that config, or use --fixes-only"
            )
        return make_student({**arm.student, "model": adapter})

    return make


@dataclass
class LoadingTrainer(CommandTrainer):
    """A `CommandTrainer` that also registers each new adapter with the running vLLM server."""

    api_base: str = ""  # the vLLM server's OpenAI-compatible base, e.g. http://localhost:8000/v1

    def train(self, fixes: Any, base: Adapter) -> Adapter:
        name = super().train(fixes, base)
        out = Path(self.root) / str(name)
        if not (out / "adapter_config.json").exists():
            raise RuntimeError(f"the training command wrote no adapter to {out} (no adapter_config.json)")
        body = json.dumps({"lora_name": name, "lora_path": str(out.resolve())}).encode()
        req = urllib.request.Request(
            f"{self.api_base.rstrip('/')}/load_lora_adapter",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310 - our own server
            if resp.status != 200:
                raise RuntimeError(f"vLLM refused adapter {name}: HTTP {resp.status}")
        return name


def save_episodes(report: CycleReport, path: Path) -> None:
    with path.open("w") as f:
        for role, recs in (
            ("heal", report.rollouts),
            ("gate_baseline", report.gate_baseline),
            ("gate_candidate", report.gate_candidate),
        ):
            for r in recs:
                f.write(json.dumps({**r, "heal_role": role}, default=str) + "\n")


def print_cycle(n: int, rec: Record) -> None:
    h, fx, g = rec["heal"], rec["fixes"], rec["gate"]
    types: dict[str, int] = {}
    for f in rec["failures"]:
        types[f["failure"]] = types.get(f["failure"], 0) + 1
    print(f"\n== cycle {n}: adapter {rec['adapter']['before'] or 'base'}")
    print(f"   heal: {h['passed']}/{len(h['tasks'])} passed, {len(h['broken'])} broken; failures {types}")
    print(f"   fixes: {fx['accepted']}/{fx['of']} accepted")
    for les, k in sorted(fx.get("lessons", {}).items(), key=lambda kv: -kv[1]):
        print(f"      {k} × {les}")
    if g is not None:
        print(f"   gate: {'PASS' if g.get('passed') else 'no'}: {g.get('reason')}")
    if rec["skipped"]:
        print(f"   stopped: {rec['skipped']}")
    a = rec["adapter"]
    print(f"   adapter now: {a['after'] or 'base'}" + (" (swapped)" if a["swapped"] else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="A", help="the arm whose student heals (default A)")
    ap.add_argument(
        "--config", help="arms config (default configs/arms.yaml); point it at vLLM for a full cycle"
    )
    ap.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS), help="heal on these")
    ap.add_argument("--limit", type=int, help="the first N train tasks per domain (default all)")
    ap.add_argument("--sample", type=int, help="a random N train tasks per domain (see --seed)")
    ap.add_argument("--seed", type=int, default=0, help="the --sample seed (default 0)")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--fixes-only", action="store_true", help="stop after the fixes: no training, no gate")
    ap.add_argument("--train-cmd", help="training command; {data} {out} {name} {base} are filled in")
    ap.add_argument("--load-adapter", action="store_true", help="register each new adapter with vLLM")
    ap.add_argument("--adapter", help="the adapter the student starts with (default: the base student)")
    ap.add_argument("--workers", type=int, default=4, help="conversations at once (default 4)")
    ap.add_argument(
        "--judges",
        default="auto",
        choices=["auto", "on", "off"],
        help="τ²-bench's LLM judges for the heal rollouts (auto: on if OPENAI_API_KEY is set); "
        "a full cycle needs them on",
    )
    ap.add_argument("--attempts", type=int, default=3, help="teacher attempts per failure (default 3)")
    ap.add_argument("--out", type=Path, help="default results/heal/<time>")
    ap.add_argument("--gate-history", type=Path, default=Path("results/heal/gate_history.jsonl"))
    a = ap.parse_args()
    if a.limit is not None and a.sample is not None:
        ap.error("pass --limit or --sample, not both")

    if not a.fixes_only and not a.train_cmd:
        ap.error("a full cycle needs --train-cmd (or pass --fixes-only)")
    if a.load_adapter and a.fixes_only:
        ap.error("--load-adapter needs a full cycle")
    if not os.environ.get("OPENROUTER_API_KEY"):
        ap.error("the teacher needs OPENROUTER_API_KEY in .env")
    judges = {"on": True, "off": False}.get(a.judges, bool(os.environ.get("OPENAI_API_KEY")))
    if not a.fixes_only and not (judges and os.environ.get("OPENAI_API_KEY")):
        ap.error("the gate scores with judges on: set OPENAI_API_KEY in .env (and don't pass --judges off)")
    if judges and not os.environ.get("OPENAI_API_KEY"):
        ap.error("--judges on needs OPENAI_API_KEY in .env")

    arm = get_arm(a.arm, a.config)
    if not a.fixes_only and COHERE in str(arm.student.get("api_base")):
        ap.error("a full cycle needs the student served by vLLM: pass its --config (or use --fixes-only)")

    from tau2.gym import register_gym_agent

    import reflexive.episode as episode

    register_gym_agent()
    episode._REGISTERED = True

    out = a.out or Path("results/heal") / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    teacher = PrefixTeacher(attempts=a.attempts)
    command = shlex.split(a.train_cmd) if a.train_cmd else None
    if a.load_adapter:
        trainer: CommandTrainer = LoadingTrainer(
            out / "runs", command=command, api_base=arm.student["api_base"]
        )
    else:
        trainer = CommandTrainer(out / "runs", command=command)

    if a.fixes_only:
        from reflexive.agent import PassRateGate

        gate: Any = PassRateGate()  # never reached: the cycle stops before the gate
    else:
        from reflexive.gate import DevGate

        gate = DevGate.load(history_file=a.gate_history)

    agent = ReflexiveAgent(
        arm_rollout(arm, student_factory(arm), judges=judges),
        teacher=teacher,
        trainer=trainer,
        gate=gate,
        adapter=a.adapter,
        workers=a.workers,
    )
    heal, dev = heal_tasks(a.domains, a.limit, a.sample, a.seed), gate_tasks()
    print(f"== heal {len(heal)} train tasks ({', '.join(a.domains)}); gate on {len(dev)} dev tasks → {out}")
    if a.fixes_only:
        print("   --fixes-only: nothing will be trained or gated")
    if not judges:
        print("   judges off: retail and airline are scored on their deterministic checks only")

    for n in range(1, a.cycles + 1):
        report = agent.run_cycle(heal, dev, stop_after_fixes=a.fixes_only)
        rec = write_audit(report, out / "audit.jsonl", cycle=n, teacher=teacher, trainer=trainer)
        save_episodes(report, out / f"cycle-{n}.episodes.jsonl")
        if report.fixes:
            (out / f"cycle-{n}.fixes.jsonl").write_text("".join(json.dumps(f) + "\n" for f in report.fixes))
        print_cycle(n, rec)
        if report.skipped and n < a.cycles:
            print("   (the next cycle would see the same student: stopping)")
            break
    print(f"\naudit: {out / 'audit.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
