"""Run heal cycles end to end: rollout → classify → teacher fix → retrain → gate → swap (SEL-47).

    python scripts/heal/cycle.py --fixes-only --domains telecom --limit 5   # today: Cohere's Aya
    python scripts/heal/cycle.py --fixes-only --sample 3                     # 3 random per domain
    python scripts/heal/cycle.py --config configs/sft/arms.base.yaml \\
        --train-cmd "bash scripts/heal/train_and_serve.sh {data} {out} {base}" --load-adapter
    python scripts/heal/cycle.py ... --cycles 3                              # SEL-23
    python scripts/heal/cycle.py --report results/heal/<run>                 # metrics, no API calls

Run it with ``just heal ...``, which loads the API keys from ``.env``.

**What it runs.** The student from ``--config``'s arm (default Arm A, `configs/arms.yaml`) on the
train tasks of ``--domains``: all of them, the first ``--limit`` per domain, or a random
``--sample`` per domain (seed ``--seed``, default 0, so a rerun picks the same tasks; the first
tasks of a split are often one family, e.g. telecom's MMS issue). The 30 **validation** tasks (train
tasks no arm trains on: the hard subset's ``solved_by_aya`` and ``teacher_unsolved`` groups) are
left out, so a healed adapter can still be scored on them; ``--keep-validation`` keeps them.
Failures go to `heal.PrefixTeacher`
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
  serve at once, so the command is expected to stop vLLM, train, and start it again: on the pod,
  ``scripts/heal/train_and_serve.sh`` (SEL-68) does exactly that.
  ``--load-adapter`` then registers ``{out}`` with the running vLLM under ``{name}``
  (``/v1/load_lora_adapter``; vLLM needs ``--enable-lora`` and
  ``VLLM_ALLOW_RUNTIME_LORA_UPDATING=True``), and the gate runs the student as model ``{name}``.

* Full cycle **through the pod's endpoint** (SEL-68): the cycle runs here, and the student is
  reached only by its public vLLM URL (`configs/sft/arms.pod.yaml`: ``POD_STUDENT_URL``,
  ``POD_API_KEY``). ``--train-url`` (default ``$HEAL_SERVICE_URL``, with ``HEAL_TOKEN``) sends each
  cycle's fixes to the pod's heal service (`scripts/pod/heal_service.py`), which trains and
  restarts vLLM with the new adapter served (`RemoteTrainer`); no ``--train-cmd`` or
  ``--load-adapter``. ``just heal --config configs/sft/arms.pod.yaml --arm P --sample 10``.

**What it writes**, in ``--out`` (default ``results/heal/<time>``):

* ``audit.jsonl``: one record per cycle (`heal.audit_record`): failures, every teacher attempt,
  fixes accepted, training, gate decision, adapter before / candidate / after;
* ``cycle-<n>.episodes.jsonl``: every conversation of the cycle, tagged ``heal_role`` =
  ``heal``, ``gate_baseline`` or ``gate_candidate``;
* ``runs/``: one folder per candidate adapter, with its ``fixes.jsonl`` and ``train.log``;
* ``run.json``: the command, commit and settings.

``--report <folder>`` sums a finished run up as ``results/heal-<name>.metrics.json``
(`heal.run_metrics`): the small file that is committed, as the arms' metrics are.

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
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reflexive import splits
from reflexive.agent import Adapter, CycleReport, Record, ReflexiveAgent, Task, arm_rollout
from reflexive.arms import get_arm, make_student, student_api_base
from reflexive.files import ROOT, read_jsonl
from reflexive.heal import TEACHER, CommandTrainer, PrefixTeacher, run_metrics, write_audit

DOMAINS = ("retail", "airline", "telecom")
COHERE = "api.cohere.ai"


VALIDATION_GROUPS = ("solved_by_aya", "teacher_unsolved")  # docs/experiments/SFT.md: the 30 validation tasks
HARD_SUBSET = (  # golden v1.1's hard subset: where setup.sh puts it on the pod, and `just train-data`'s copy
    ROOT / "data" / "golden" / "hub-v1.1" / "hard_subset.jsonl",
    ROOT / "data" / "golden" / "hard" / "hard_subset.jsonl",
)


def validation_tasks(paths: tuple[Path, ...] = HARD_SUBSET) -> set[Task]:
    """The validation tasks: train tasks no arm trains on, so heal cycles must not train on them either."""
    for path in paths:
        if path.exists():
            rows = read_jsonl(path)
            return {(r["domain"], str(r["task_id"])) for r in rows if r.get("group") in VALIDATION_GROUPS}
    raise FileNotFoundError(
        f"no hard subset in {', '.join(str(p) for p in paths)}: it lists the validation tasks "
        "(run scripts/pod/setup.sh, or pass --keep-validation)"
    )


def heal_tasks(
    domains: list[str],
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 0,
    exclude: set[Task] | None = None,
) -> list[Task]:
    """The train tasks to heal on: all, the first ``limit``, or a random ``sample`` per domain,
    never one in ``exclude`` (the validation tasks).

    A sample keeps the split's order and is the same for the same seed.
    """
    if limit is not None and sample is not None:
        raise ValueError("pass --limit or --sample, not both")
    out = []
    for d in domains:
        ids = [t for t in splits.task_ids(d, "train") if (d, str(t)) not in (exclude or set())]
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
    cohere = COHERE in str(arm.student.get("api_base") or "")

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


@dataclass
class RemoteTrainer(CommandTrainer):
    """Trains each adapter on the pod, through its heal service (`scripts/pod/heal_service.py`).

    For a cycle that reaches the student only through the pod's endpoint. The fixes are written
    here as for any `CommandTrainer` (``runs/<name>/fixes.jsonl``), sent to the service, and the job
    is polled until it ends. The service restarts vLLM with the new adapter served under its name;
    before the gate runs, the endpoint's model list must show it.
    """

    url: str = ""  # the heal service, e.g. https://<pod>-8001.proxy.runpod.net
    token: str = ""
    api_base: str = ""  # the student's vLLM endpoint, to check the adapter is served
    api_key: str | None = None
    poll_s: float = 30.0
    timeout_s: float = 4 * 3600.0
    max_refused: int = 3  # polls refused in a row (401/403/404) before giving up
    served_tries: int = 3  # the final model-list check, through the same proxy

    def train(self, fixes: Any, base: Adapter) -> Adapter:
        name = super().train(fixes, base)  # no command: writes the data, returns the name
        call = self.calls[-1]
        data = Path(call["data"])
        sent = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
        start = time.monotonic()
        code, body = self._request("POST", "/heal/train", {"name": name, "base": base, "fixes": sent})
        if code != 202:
            raise RuntimeError(f"the heal service refused {name}: HTTP {code} {body.get('error', body)}")
        status: dict[str, Any] = {}
        refused = 0  # consecutive 401/403/404: the service restarted with another token or --root
        while time.monotonic() - start < self.timeout_s:
            time.sleep(self.poll_s)
            code, status = self._request("GET", f"/heal/jobs/{name}")
            if code == 200 and status.get("status") != "running":
                break
            refused = refused + 1 if code in (401, 403, 404) else 0
            if refused >= self.max_refused:
                raise RuntimeError(
                    f"the heal service answers HTTP {code} for {name} ({refused} polls in a row): "
                    f"{status.get('error', status)}; was it restarted with another HEAL_TOKEN or --root?"
                )
        else:
            raise RuntimeError(f"{name}: still training after {self.timeout_s / 3600:.1f} h")
        call.update(
            remote=self.url,
            returncode=status.get("exit_code"),
            seconds=round(time.monotonic() - start, 1),
        )
        (data.parent / "train.log").write_text("\n".join(status.get("log_tail") or []) + "\n")
        if status.get("status") != "done":
            raise RuntimeError(
                f"training {name} on the pod failed (exit {status.get('exit_code')}); "
                f"see {data.parent / 'train.log'} (the last lines) or the pod's runs/heal-service/{name}/"
            )
        if name not in self._served():
            raise RuntimeError(f"the pod trained {name} but its vLLM doesn't serve it: check the pod")
        return name

    def _request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        req = urllib.request.Request(
            self.url.rstrip("/") + path,
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - our own service
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read() or b"{}")
            except json.JSONDecodeError:
                return e.code, {}
        except (urllib.error.URLError, TimeoutError) as e:  # the pod's proxy can drop a poll
            return 0, {"error": str(e)}

    def _served(self) -> set[str]:
        """The endpoint's model list, retried: the pod's proxy can drop a request (502, timeout)."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        req = urllib.request.Request(self.api_base.rstrip("/") + "/models", headers=headers)
        for attempt in range(1, self.served_tries + 1):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - the student's server
                    return {m.get("id") for m in json.loads(resp.read()).get("data", [])}
            except urllib.error.HTTPError as e:
                if e.code < 500 or attempt == self.served_tries:  # 401/404 won't fix themselves
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.served_tries:
                    raise
            time.sleep(min(self.poll_s, 10.0))
        raise AssertionError("unreachable")


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


def git_commit() -> str | None:
    import subprocess

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")


def report_run(folder: Path, name: str | None = None) -> Path:
    """``results/heal-<name>.metrics.json`` from a run folder's audit, conversations and run.json."""
    audits = read_jsonl(folder / "audit.jsonl")
    episodes = []
    for a in audits:
        path = folder / f"cycle-{a['cycle']}.episodes.jsonl"
        episodes += [{**e, "cycle": a["cycle"]} for e in read_jsonl(path)] if path.exists() else []
    run_file = folder / "run.json"
    run = json.loads(run_file.read_text()) if run_file.exists() else {}
    m = run_metrics(audits, episodes, {**run, "folder": str(folder)})
    out = Path("results") / f"heal-{name or folder.name}.metrics.json"
    out.write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n")
    for c in m["cycles"]:
        lo, hi = c["task_success_wilson95"]
        print(
            f"cycle {c['cycle']}: {c['passed']}/{c['conversations']} passed ({lo:.0%}–{hi:.0%}); "
            f"failures {c['failure_types']}; fixed {c['fixed']}/{c['failures']} "
            f"({c['fixable']} had a turn to fix)"
        )
    print(f"cost ${m['cost_usd']['total']:.2f} (judges not included) → {out}")
    return out


def check_args(ap: argparse.ArgumentParser, a: argparse.Namespace) -> bool:
    """Refuse settings that can't work, before anything is spent. Returns whether judges are on."""
    if a.limit is not None and a.sample is not None:
        ap.error("pass --limit or --sample, not both")
    if not a.fixes_only and not (a.train_cmd or a.train_url):
        ap.error("a full cycle needs --train-cmd or --train-url (or pass --fixes-only)")
    if a.train_cmd and a.train_url:
        ap.error("pass --train-cmd (training here) or --train-url (on the pod), not both")
    if a.load_adapter and a.fixes_only:
        ap.error("--load-adapter needs a full cycle")
    if a.load_adapter and a.train_url:
        ap.error("--train-url: the pod serves each new adapter itself, no --load-adapter")
    if a.train_url and not os.environ.get("HEAL_TOKEN"):
        ap.error("--train-url needs HEAL_TOKEN in .env (the heal service's token)")
    if not os.environ.get("OPENROUTER_API_KEY"):
        ap.error("the teacher needs OPENROUTER_API_KEY in .env")
    judges = {"on": True, "off": False}.get(a.judges, bool(os.environ.get("OPENAI_API_KEY")))
    if not a.fixes_only and not (judges and os.environ.get("OPENAI_API_KEY")):
        ap.error("the gate scores with judges on: set OPENAI_API_KEY in .env (and don't pass --judges off)")
    if judges and not os.environ.get("OPENAI_API_KEY"):
        ap.error("--judges on needs OPENAI_API_KEY in .env")
    return judges


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
    ap.add_argument(
        "--keep-validation",
        action="store_true",
        help="also heal on the 30 validation tasks (left out by default, so they stay unseen)",
    )
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--fixes-only", action="store_true", help="stop after the fixes: no training, no gate")
    ap.add_argument("--train-cmd", help="training command; {data} {out} {name} {base} are filled in")
    ap.add_argument("--load-adapter", action="store_true", help="register each new adapter with vLLM")
    ap.add_argument(
        "--train-url",
        default=os.environ.get("HEAL_SERVICE_URL"),
        help="train on the pod through its heal service (scripts/pod/heal_service.py; "
        "default $HEAL_SERVICE_URL), for a cycle that reaches the student only by its endpoint",
    )
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
    ap.add_argument(
        "--report",
        type=Path,
        metavar="RUN_FOLDER",
        help="no run: write results/heal-<folder name>.metrics.json from a finished run (no API calls)",
    )
    ap.add_argument("--name", help="with --report: the metrics file's name (default: the folder's)")
    a = ap.parse_args()
    if a.report:
        if not (a.report / "audit.jsonl").exists():
            ap.error(f"no audit.jsonl in {a.report}")
        report_run(a.report, a.name)
        return 0
    judges = check_args(ap, a)

    arm = get_arm(a.arm, a.config)
    if not a.fixes_only and COHERE in str(arm.student.get("api_base")):
        ap.error("a full cycle needs the student served by vLLM: pass its --config (or use --fixes-only)")

    from tau2.gym import register_gym_agent

    import reflexive.episode as episode

    register_gym_agent()
    episode._REGISTERED = True

    out = a.out or Path("results/heal") / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    run = {
        "command": "python scripts/heal/cycle.py " + " ".join(shlex.quote(x) for x in sys.argv[1:]),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": git_commit(),
        "arm": a.arm,
        "config": a.config or "configs/arms.yaml",
        "student": arm.student.get("model"),
        "customer": arm.customer,
        "teacher": TEACHER,
        "domains": a.domains,
        "limit": a.limit,
        "sample": a.sample,
        "seed": a.seed,
        "judges": judges,
        "fixes_only": a.fixes_only,
        "attempts": a.attempts,
    }
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    teacher = PrefixTeacher(attempts=a.attempts)
    command = shlex.split(a.train_cmd) if a.train_cmd else None
    if a.train_url and not a.fixes_only:
        key_env = arm.student.get("api_key_env")
        trainer: CommandTrainer = RemoteTrainer(
            out / "runs",
            url=a.train_url,
            token=os.environ["HEAL_TOKEN"],
            api_base=student_api_base(arm.student),
            api_key=os.environ.get(key_env) if key_env else None,
        )
    elif a.load_adapter:
        trainer = LoadingTrainer(out / "runs", command=command, api_base=student_api_base(arm.student))
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
    exclude = set() if a.keep_validation else validation_tasks()
    heal, dev = heal_tasks(a.domains, a.limit, a.sample, a.seed, exclude=exclude), gate_tasks()
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
    print(f"\naudit: {out / 'audit.jsonl'}; metrics: just heal --report {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
