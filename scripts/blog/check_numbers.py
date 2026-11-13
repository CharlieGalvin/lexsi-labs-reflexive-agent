"""Check every number in the blog post against the committed results (SEL-77).

    just blog-check                 # recompute, compare with docs/blog/reflexive-agents.md
    just reproduce                  # the walkthrough of every experiment, which ends with this check

Each claim is recomputed from a committed metrics file and must appear in the post **exactly as
written** (a whole table row or phrase, not just the number). Exits 1 if any doesn't. Claims whose
raw results were never committed (the teacher bake-off, the student review) or that live on a
teammate's branch are listed with the page they come from, so a reader can follow them up.

Re-running the experiments themselves needs a GPU and API keys; each experiment page in
`docs/experiments/` gives its commands. This reproduces the *numbers* from what those runs saved.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
POST = ROOT / "docs" / "blog" / "reflexive-agents.md"
DOMAINS = ("retail", "airline", "telecom")
MINUS = "−"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def row(compare: str, metric: str) -> dict:
    return next(r for r in load(compare)["rows"] if r["metric"] == metric)


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def signed(x: float, digits: int = 1) -> str:
    s = f"{abs(x):.{digits}f}"
    return f"+{s}" if x >= 0 else f"{MINUS}{s}"


def mean_passes(arm: str, split: str) -> float:
    """Passed conversations over the three domains, divided by 2 samples: the mean of 2 samples."""
    return sum(load(f"arm-{arm}-{d}-{split}.metrics.json")["passed"] for d in DOMAINS) / 2


def dev_from_compares(side: str) -> tuple[float, float]:
    """(mean passes of 2 samples, loop conversations) for side 'a' (S2) or 'b' (H), over dev."""
    passes = loops = 0.0
    for d in DOMAINS:
        c = load(f"S2-{d}-dev-vs-H-{d}-dev.compare.metrics.json")
        n = c[f"{side}_scored"]
        rows = {r["metric"]: r for r in c["rows"]}
        passes += rows["success_rate"][side] * n
        loops += rows["loop_rate"][side] * n
    return round(passes) / 2, round(loops)


def monitor() -> tuple[int, int, int, int]:
    """Telecom replay: (loops to the step limit, of them caught, passed conversations, of them flagged)."""
    looped = caught = passed = flagged = 0
    for name, f in load("monitor-no-progress-replay.metrics.json")["files"].items():
        if "telecom" not in name:
            continue
        looped += f["step_limit"]
        passed += f["passed"]
        caught += sum(1 for x in f.get("fired", []) if x.get("termination") == "max_steps")
        flagged += sum(1 for x in f.get("fired", []) if x.get("passed"))
    return looped, caught, passed, flagged


def claims() -> list[tuple[str, Callable[[], str], str]]:
    """(what, a function giving the text the post must contain, source file)."""
    ab, ac = "arm-A-test-vs-arm-S2-test.compare.metrics.json", "arm-A-test-vs-arm-C-test.compare.metrics.json"
    s = {a: load(f"arm-{a}-test.eval.metrics.json")["overall"]["success_rate"] for a in ("A", "C", "S2")}

    def test_row(metric: str, label: str) -> str:
        a, b, c = row(ab, metric)["a"], row(ab, metric)["b"], row(ac, metric)["b"]
        return f"{label}: Arm A {pct(a)}, Arm C {pct(c)}, Arm B {pct(b)}"  # the test-table image's alt text

    def hall_b() -> float:
        return row(ab, "hallucination_rate")["b"]

    def b_minus_a() -> str:
        r = row(ab, "success_rate")
        lo, hi = r["diff_ci95"]
        return f"by {r['diff_b_minus_a'] * 100:.0f} points** (95% range +{lo * 100:.0f} to +{hi * 100:.0f})"

    def c_minus_a() -> str:
        r = row(ac, "success_rate")
        lo, hi = r["diff_ci95"]
        diff = signed(r["diff_b_minus_a"] * 100)
        return f"a difference of {diff} points, 95% range {signed(lo * 100)} to {signed(hi * 100)}"

    def dev_h_vs_s2() -> str:
        s2, h = dev_from_compares("a")[0], dev_from_compares("b")[0]
        return f"({h / 36:.0%} against {s2 / 36:.0%} on dev)"

    def loops_halved() -> str:
        s2, h = dev_from_compares("a")[1], dev_from_compares("b")[1]
        assert h * 2 == s2, f"loops {s2} → {h} isn't half"
        return "halved"

    def telecom_repeats() -> str:
        r = row("S2-telecom-dev-vs-H-telecom-dev.compare.metrics.json", "redundant_call_rate")
        return f"from {r['a']:.0%} to {r['b']:.0%}"

    def sft_dev() -> str:
        return f"from {mean_passes('P', 'dev') / 36:.0%} to {mean_passes('S', 'dev') / 36:.0%}"

    def monitor_claim() -> str:
        looped, caught, passed, flagged = monitor()
        assert caught == looped and flagged == 0, (looped, caught, passed, flagged)
        return f"flagged all {looped} that looped"

    return [
        (
            "Test success, arms A / C / B",
            lambda: test_row("success_rate", "Success"),
            "arm-A-test-vs-arm-*-test.compare",
        ),
        (
            "Test Tool/Arg rate",
            lambda: test_row("hallucination_rate", "Conversations with a Tool or Arg failure"),
            "same",
        ),
        ("Test loop rate", lambda: test_row("loop_rate", "Conversations with a loop"), "same"),
        ("Headline success", lambda: f"from **{pct(s['A'])} to {pct(s['S2'])}**", "arm-*-test.eval"),
        (
            "Headline hallucination",
            lambda: f"from **{row(ab, 'hallucination_rate')['a']:.0%} to {hall_b():.0%}**",
            "arm-A-test-vs-arm-S2-test.compare",
        ),
        ("Arm B − Arm A", b_minus_a, "arm-A-test-vs-arm-S2-test.compare"),
        ("Careful prompt − Arm A", c_minus_a, "arm-A-test-vs-arm-C-test.compare"),
        ("SFT v1 on dev (base → v1)", sft_dev, "arm-P-*-dev, arm-S-*-dev"),
        ("SFT v2 on dev", lambda: f"{mean_passes('S2', 'dev') / 36:.0%} on dev", "arm-S2-*-dev"),
        ("Arm B on validation", lambda: f"Arm B's {mean_passes('S2', 'val'):.1f}", "arm-S2-*-val"),
        ("GRPO (H) vs Arm B on dev", dev_h_vs_s2, "S2-*-dev-vs-H-*-dev.compare"),
        ("GRPO (H) halves loops (12 → 6 of 72)", loops_halved, "S2-*-dev-vs-H-*-dev.compare"),
        ("GRPO (H) telecom repeated calls", telecom_repeats, "S2-telecom-dev-vs-H-telecom-dev.compare"),
        ("Runtime monitor on telecom", monitor_claim, "monitor-no-progress-replay"),
    ]


DOCUMENTED = [
    (
        "Teacher: 12 models screened; 88% first try; 158 of 178 fixed; $11",
        "docs/experiments/TEACHER_PHASE2_RESULTS.md",
    ),
    (
        "Student review: 7 of 30 passed; 0 unreadable calls in 266 replies",
        "docs/experiments/STUDENT_REVIEW_AYA.md",
    ),
    ("Teacher fixes cost about $0.001 each", "docs/experiments/HEAL_FIXES_ONLY.md"),
    ("First GRPO run: 0.5 of 30 on validation", "docs/EXPERIMENTS.md (grpo-aya32b-v2)"),
    ("Second GRPO run: 177 teacher fixes", "docs/experiments/GRPO_HEAL.md"),
    (
        "Full heal cycle: 24 of 24 fixed, 71% fewer errors, 4 against 4",
        "HEAL_FIRST_CYCLE.md (branch charliegalvin/sel-68-first-cycle-results)",
    ),
    ("Third GRPO run: 14.5 of 30 on validation", "GRPO_DECISION.md (branch charliegalvin/h2-decision-grpo)"),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--post", type=Path, default=POST)
    a = p.parse_args(argv)
    post = a.post.read_text()
    flat = " ".join(post.split())  # line breaks in the post don't matter
    bad = 0
    print(f"Recomputed from results/ and checked against {a.post.relative_to(ROOT)}:\n")
    for what, compute, source in claims():
        try:
            text = compute()
        except (AssertionError, FileNotFoundError, KeyError) as e:
            print(f"  ✗ {what}: couldn't recompute ({type(e).__name__}: {e})")
            bad += 1
            continue
        ok = " ".join(text.split()) in flat
        bad += not ok
        print(
            f"  {'✓' if ok else '✗'} {what}: {text!r}"
            + ("" if ok else "  ← not in the post as written")
            + f"  [{source}]"
        )
    print("\nFrom the experiment write-ups (raw results not committed, or on a teammate's branch):\n")
    for what, source in DOCUMENTED:
        print(f"  · {what}  [{source}]")
    print(f"\n{'All recomputed numbers match the post.' if not bad else f'{bad} claim(s) differ.'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
