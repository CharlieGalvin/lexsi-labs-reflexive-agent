"""Walk through the blog post's experiments and reproduce their results (SEL-77).

    just reproduce                  # every experiment, then a check of the post's numbers
    just reproduce 3                # one experiment
    just reproduce --list           # the experiments, without running them

No GPU, API keys or downloads. Each experiment is introduced (the question, what we ran), then its
results are recomputed from the metrics files those runs saved in `results/`, with the same rules
the post uses (95% bootstrap ranges over tasks, the agreed gate). Numbers whose raw results were
never committed are shown as "from the write-up", with the page they come from.

Every experiment also prints the commands that rerun it from scratch. Those need the student on a
GPU (Aya Expanse 32B, ~80 GB) and API keys for the customer simulator and the teacher.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_numbers as cn  # noqa: E402  (the post's number checks: same loaders, same rules)

BOLD, DIM, GREEN, RED, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[0m"
if not sys.stdout.isatty():
    BOLD = DIM = GREEN = RED = RESET = ""
DOMAINS = cn.DOMAINS
TEST_AB = "arm-A-test-vs-arm-S2-test.compare.metrics.json"
TEST_AC = "arm-A-test-vs-arm-C-test.compare.metrics.json"


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def ci(lo_hi: list[float]) -> str:
    lo, hi = lo_hi
    return f"({lo * 100:.0f}–{hi * 100:.0f}%)"


def diff(r: dict) -> str:
    lo, hi = r["diff_ci95"]
    d = r["diff_b_minus_a"] * 100
    return f"{cn.signed(d)} pts ({cn.signed(lo * 100)} to {cn.signed(hi * 100)})"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]

    def line(r: list[str]) -> str:  # first column left-aligned, numbers right-aligned
        cells = zip(r, widths, strict=True)
        return "  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w) for i, (c, w) in enumerate(cells))

    return [line(header), "  ".join("─" * w for w in widths), *(line(r) for r in rows)]


@dataclass
class Experiment:
    title: str
    question: str
    method: str
    run: Callable[[], tuple[list[str], str, bool]]  # → (result lines, verdict, verdict is a "yes")
    sources: list[str]
    rerun: list[str] = field(default_factory=list)
    needs: str = "the student on a GPU and API keys"


# ---- 1. The baseline -------------------------------------------------------------------------


def baseline() -> tuple[list[str], str, bool]:
    a = cn.load("arm-A-test.eval.metrics.json")
    rows = [["All 100 tasks", pct(a["overall"]["success_rate"]), ci(a["overall"]["success_ci95"])]]
    for d in DOMAINS:
        v = a["by_domain"][f"A-{d}-test"]
        rows.append([d.capitalize(), pct(v["success_rate"]), ci(v["success_ci95"])])
    h, loop = cn.row(TEST_AB, "hallucination_rate"), cn.row(TEST_AB, "loop_rate")
    lines = table(["Untuned student, test split", "Success", "95% range"], rows)
    lines += [
        "",
        f"Conversations with a Tool or Arg failure: {pct(h['a'])} {ci(h['a_ci95'])}",
        f"Conversations with a loop:                {pct(loop['a'])} {ci(loop['a_ci95'])}",
    ]
    return (
        lines,
        f"The untuned agent hallucinates in {h['a']:.0%} of conversations "
        f"and solves {a['overall']['success_rate']:.1%} of tasks.",
        False,
    )  # noqa: E501


# ---- 2. The careful prompt -------------------------------------------------------------------


def careful() -> tuple[list[str], str, bool]:
    rows = []
    for metric, label in [
        ("success_rate", "Success"),
        ("hallucination_rate", "Tool or Arg failure"),
        ("loop_rate", "Loop"),
    ]:
        r = cn.row(TEST_AC, metric)
        rows.append([label, pct(r["a"]), pct(r["b"]), diff(r)])
    lines = table(["Test split, 100 tasks × 2", "Arm A", "Arm C", "C − A (95% range)"], rows)
    r = cn.row(TEST_AC, "success_rate")
    clear = r["diff_ci95"][0] > 0 or r["diff_ci95"][1] < 0
    return (
        lines,
        (
            "The prompt made a clear difference."
            if clear
            else "No clear difference: the 95% range of C − A includes zero. Asking nicely doesn't work."
        ),
        clear,
    )


# ---- 3. Fine-tuning on the teacher's verified fixes ------------------------------------------


def sft() -> tuple[list[str], str, bool]:
    rows = []
    for arm, label in [("P", "Base model"), ("S", "SFT v1"), ("S2", "SFT v2  (Arm B)")]:
        m = [cn.load(f"arm-{arm}-{d}-dev.metrics.json") for d in DOMAINS]
        rows.append(
            [
                label,
                *(f"{x['passed']}/{x['conversations']}" for x in m),
                f"{cn.mean_passes(arm, 'dev') / 36:.0%}",
            ]
        )
    lines = ["Step 1: the dev split (36 tasks × 2 samples), passed conversations", ""]
    lines += table(["Student", "Retail", "Airline", "Telecom", "Success"], rows)
    lines += [
        "",
        f"{DIM}Gate: v1 rejected (its airline result dropped: changes nobody asked for); "
        f"v2 promoted [docs/experiments/SFT.md]{RESET}",
        "",
        "Step 2: the declared test, 100 unseen tasks × 2 samples",
        "",
    ]
    rows = []
    for metric, label in [
        ("success_rate", "Success"),
        ("hallucination_rate", "Tool or Arg failure"),
        ("loop_rate", "Loop"),
    ]:
        r = cn.row(TEST_AB, metric)
        rows.append([label, pct(r["a"]), pct(r["b"]), diff(r)])
    lines += table(["Test split", "Arm A", "Arm B", "B − A (95% range)"], rows)
    r = cn.row(TEST_AB, "success_rate")
    return (
        lines,
        (
            f"Arm B beats the untuned student by {r['diff_b_minus_a'] * 100:.0f} points, and the whole 95% "
            f"range is above zero. Healing works."
        ),
        r["diff_ci95"][0] > 0,
    )


# ---- 4. Reinforcement learning (GRPO) --------------------------------------------------------


def grpo() -> tuple[list[str], str, bool]:
    s2_val = cn.mean_passes("S2", "val")
    (s2_dev, s2_loops), (h_dev, h_loops) = cn.dev_from_compares("a"), cn.dev_from_compares("b")
    tel = cn.row("S2-telecom-dev-vs-H-telecom-dev.compare.metrics.json", "redundant_call_rate")
    try:  # the gate's own record, where it's present: results/*.json other than metrics isn't in git
        g = cn.load("gate-S2-vs-H-s0.json")
        gate_line = f"  Gate (Arm B vs heal GRPO, sample 0): {g['verdict'].upper()}: {g['reason']}"
    except FileNotFoundError:
        gate_line = (
            f"  {DIM}from the write-up [docs/EXPERIMENTS.md, grpo-heal-aya32b-v2]:{RESET} "
            "gate (Arm B vs heal GRPO, sample 0): REJECT, 24 → 19 dev tasks"
        )
    lines = [
        f"Starting point: Arm B solves {s2_val:.1f} of 30 held-out validation tasks (mean of 2 samples).",
        "",
        "Run 1: reward only the teacher's tool calls",
        f"  {DIM}from the write-up [docs/EXPERIMENTS.md, grpo-aya32b-v2]:{RESET} 0.5 of 30 on validation. "
        "It stopped replying to customers.",
        "",
        "Run 2: heal GRPO, one reward per failure type (Tool, Arg, Loop) + agreement with the teacher",
        "",
    ]
    lines += [
        "  " + x
        for x in table(
            ["Dev split, 36 tasks × 2", "Arm B", "Heal GRPO"],
            [
                ["Success", f"{s2_dev / 36:.0%}", f"{h_dev / 36:.0%}"],
                ["Conversations with a loop", f"{s2_loops:.0f} of 72", f"{h_loops:.0f} of 72"],
                ["Repeated calls, telecom", f"{tel['a']:.0%}", f"{tel['b']:.0%}"],
            ],
        )
    ]
    lines += [
        "",
        gate_line,
        "",
        "Run 3: + a reward for the right decision",
        f"  {DIM}from the write-up [GRPO_DECISION.md, branch charliegalvin/h2-decision-grpo]:{RESET} "
        f"14.5 of 30 on validation, against Arm B's {s2_val:.1f}. Not adopted.",
    ]
    return (
        lines,
        (
            "Half the loops, far fewer repeated calls, but fewer tasks solved: the failures left are "
            "decisions, not hallucinations. Fine-tuning stays the champion."
        ),
        False,
    )


# ---- 5. The runtime monitor ------------------------------------------------------------------


def monitor() -> tuple[list[str], str, bool]:
    looped, caught, passed, flagged = cn.monitor()
    lines = table(
        ["Telecom conversations, replayed", "Count", "Flagged"],
        [
            ["Looped until the step limit", str(looped), str(caught)],
            ["Passed", str(passed), str(flagged)],
        ],
    )
    ok = caught == looped and flagged == 0
    return lines, f"It flagged all {looped} that looped, and none that passed.", ok


EXPERIMENTS = [
    Experiment(
        "The baseline: how often does an untuned agent hallucinate?",
        "Before healing anything: how bad is the problem?",
        "Aya Expanse 32B, untuned (Arm A), on the 100 test tasks of τ²-bench, 2 samples each.\n"
        "A conversation passes only if the database ends in exactly the right state. A classifier\n"
        "labels each failure Tool (a tool it doesn't have), Arg (an invented argument) or Loop\n"
        "(repeating itself).",
        baseline,
        ["results/arm-A-test.eval.metrics.json", f"results/{TEST_AB}"],
        [
            "just rollout A <domain> --split test --samples 2        # each of retail, airline, telecom",
            "just arm-eval results/episodes/arms/A-*-test.jsonl",
        ],
    ),
    Experiment(
        "Can you just tell it to be careful?",
        "Does an instruction not to hallucinate fix it, for free?",
        'The same student with a fixed "be careful not to hallucinate" instruction (Arm C), on the same\n'
        "test tasks, compared with Arm A task by task (paired bootstrap, 10,000 resamples).",
        careful,
        [f"results/{TEST_AC}"],
        [
            "just rollout C <domain> --split test --samples 2",
            "just arm-compare results/episodes/arms/A-<domain>-test.jsonl \\",
            "  results/episodes/arms/C-<domain>-test.jsonl",
        ],
    ),
    Experiment(
        "Healing with fine-tuning on the teacher's verified fixes",
        "If a stronger model rewrites the turns where the student failed, can the student learn from them?",
        "The teacher (GLM-5.3-Flash) solved the train tasks; we kept only conversations the benchmark\n"
        "verified. We fine-tuned the student on them (QLoRA, rank 16), gated each version on dev, then\n"
        "ran the promoted version (Arm B) once on the test split, declared in advance.",
        sft,
        ["results/arm-{P,S,S2}-<domain>-dev.metrics.json", f"results/{TEST_AB}"],
        [
            "just teacher-fixes && just train-data && just sft-v2-data           # API keys, no GPU",
            "just sft-train configs/train/sft_aya32b_v2.yaml --push                # GPU",
            "just rollout S2 <domain> --split dev --samples 2 --config configs/sft/arms.sft2.yaml",
            "just gate-check P S2",
        ],
    ),
    Experiment(
        "Healing with reinforcement learning (GRPO)",
        "Can a reward for each kind of failure heal the student further than fine-tuning did?",
        "Starting from Arm B, each prompt is a conversation up to one agent turn. The model writes that\n"
        "turn 8 times and a reward scores each attempt. Three runs, each with a different reward.",
        grpo,
        [
            "results/arm-S2-<domain>-val.metrics.json",
            "results/S2-<domain>-dev-vs-H-<domain>-dev.compare.metrics.json",
            "results/gate-S2-vs-H-s0.json",
        ],
        [
            "just heal-fixes && just grpo-heal-data && just grpo-heal-reward-check   # API key, no GPU",
            "just grpo-train configs/train/grpo_heal_aya32b.yaml --push               # GPU, ~4 h",
            "just gate-check S2 H",
        ],
    ),
    Experiment(
        "A runtime monitor for loops",
        "Can we stop a loop while the conversation is still happening?",
        "A monitor watches for repeated, no-progress turns. We replayed Arm A's saved telecom\n"
        "conversations through it and counted what it would have flagged.",
        monitor,
        ["results/monitor-no-progress-replay.metrics.json"],
        [
            "python scripts/arms/monitor_replay.py \\",
            "  results/episodes/arms/A-{telecom,retail,airline}-*.jsonl \\",
            "  results/episodes/arms/C-telecom-train.jsonl \\",
            "  --metrics results/monitor-no-progress-replay.metrics.json",
        ],
        needs="only Arm A's saved conversations, which aren't in git; no GPU or keys",
    ),
]


def show(i: int, e: Experiment) -> bool:
    n = len(EXPERIMENTS)
    print(f"\n{BOLD}━━ Experiment {i} of {n}: {e.title}{RESET}\n")
    print(f"{BOLD}Question:{RESET} {e.question}")
    print(f"{BOLD}What we ran:{RESET}\n  " + e.method.replace("\n", "\n  "))
    print(f"\n{DIM}▶ running: recomputing from {', '.join(e.sources)}{RESET}\n")
    try:
        lines, verdict, good = e.run()
    except (FileNotFoundError, KeyError, AssertionError) as err:
        print(f"  {RED}✗ couldn't recompute: {type(err).__name__}: {err}{RESET}")
        return False
    for line in lines:
        print("  " + line)
    print(f"\n{BOLD}Result:{RESET} {GREEN if good else ''}{verdict}{RESET}")
    print(f"\n{DIM}Rerun from scratch (needs {e.needs}; see docs/experiments/):")
    for c in e.rerun:
        print(f"  {c}")
    print(RESET, end="")
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", nargs="?", type=int, help=f"run only this one (1–{len(EXPERIMENTS)})")
    p.add_argument("--list", action="store_true", help="list the experiments")
    a = p.parse_args(argv)
    if a.list:
        for i, e in enumerate(EXPERIMENTS, 1):
            print(f"{i}. {e.title}")
        return 0
    if a.experiment is not None and not 1 <= a.experiment <= len(EXPERIMENTS):
        p.error(f"experiment must be 1–{len(EXPERIMENTS)}")
    chosen = [a.experiment] if a.experiment else range(1, len(EXPERIMENTS) + 1)
    print(f"{BOLD}Reflexive Agents: reproducing the blog post's results{RESET}")
    print(f"{DIM}From the metrics each run saved in results/. No GPU or API keys needed.{RESET}")
    ok = all([show(i, EXPERIMENTS[i - 1]) for i in chosen])
    if a.experiment is None:
        print(f"\n{BOLD}━━ Checking the post{RESET}\n")
        post = " ".join(cn.POST.read_text().split())
        claims = cn.claims()
        hits = sum(" ".join(f().split()) in post for _, f, _ in claims)
        print(
            f"{hits} of {len(claims)} recomputed numbers appear in docs/blog/reflexive-agents.md exactly as "
            f"written{'.' if hits == len(claims) else f'. {RED}Run `just blog-check` for details.{RESET}'}"
        )
        ok = ok and hits == len(claims)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
