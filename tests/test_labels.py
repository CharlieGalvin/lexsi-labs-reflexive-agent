"""reflexive.labels and scripts/labels/label_metrics.py (SEL-85, SEL-98): hand-checkable answers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _scripts import load_script

from reflexive.labels import Confusion, agreement_report, as_bool, bootstrap, cohen_kappa, confusion_report

label_metrics = load_script("labels/label_metrics")


def pairs(tp: int, fp: int, fn: int, tn: int) -> list[tuple[bool, bool]]:
    return [(True, True)] * tp + [(False, True)] * fp + [(True, False)] * fn + [(False, False)] * tn


def test_confusion_metrics_by_hand() -> None:
    c = Confusion.of(pairs(tp=8, fp=2, fn=4, tn=6))
    assert (c.tp, c.fp, c.fn, c.tn, c.n) == (8, 2, 4, 6, 20)
    m = c.metrics()
    assert m["precision"] == pytest.approx(0.8)  # 8 / 10 flagged
    assert m["recall"] == pytest.approx(8 / 12)  # 8 / 12 real
    assert m["f1"] == pytest.approx(2 * 0.8 * (8 / 12) / (0.8 + 8 / 12))
    assert m["false_positive_rate"] == pytest.approx(0.25)  # 2 / 8 negatives
    assert m["accuracy"] == pytest.approx(0.7)


def test_undefined_rates_are_none_not_zero() -> None:
    c = Confusion.of(pairs(tp=0, fp=0, fn=3, tn=5))  # it flagged nothing
    assert c.rate("precision") is None
    assert c.f1() is None
    assert c.rate("recall") == 0.0


def test_kappa_textbook_example() -> None:
    # 50 items: both yes 20, A yes B no 5, A no B yes 10, both no 15 -> p_o 0.70, p_e 0.50, kappa 0.40
    a = ["yes"] * 25 + ["no"] * 25
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert cohen_kappa(a, a) == pytest.approx(1.0)
    assert cohen_kappa(["x", "x"], ["x", "x"]) is None  # one category only: undefined


def test_as_bool() -> None:
    assert as_bool("Yes") and as_bool("1") and as_bool(True)
    assert not as_bool("no") and not as_bool("0")
    assert as_bool("FAIL", positive="fail") and not as_bool("pass", positive="fail")
    with pytest.raises(ValueError):
        as_bool("maybe")


def test_report_intervals_contain_the_estimate() -> None:
    rep = confusion_report(pairs(tp=30, fp=5, fn=10, tn=55), reps=2000)
    for k, v in rep["metrics"].items():
        lo, hi = rep["ci95"][k]
        assert lo <= v <= hi, k
    assert rep["interval_method"].startswith("Wilson")


def test_cluster_bootstrap_is_wider_than_treating_samples_as_independent() -> None:
    # Each task's two samples agree with each other: clustering must not shrink the range
    p = pairs(tp=10, fp=4, fn=6, tn=20) * 2
    tasks = [f"t{i}" for i in range(40)] * 2
    flat = confusion_report(p, reps=2000)["ci95"]["recall"]
    clustered = confusion_report(p, clusters=tasks, reps=2000)["ci95"]["recall"]
    assert clustered[1] - clustered[0] >= (flat[1] - flat[0]) * 0.9


def test_bootstrap_none_when_mostly_undefined() -> None:
    assert bootstrap([[1]], lambda xs: None, reps=100) is None
    assert bootstrap([], lambda xs: 1.0) is None


def test_agreement_report_lists_disagreements() -> None:
    a = {"c1": "no yes", "c2": "reasoning", "c3": "nobody asked", "c4": "against policy"}
    b = {"c1": "no yes", "c2": "against policy", "c3": "nobody asked", "c5": "reasoning"}
    rep = agreement_report(a, b, reps=500)
    assert rep["n"] == 3 and rep["only_in_a"] == 1 and rep["only_in_b"] == 1
    assert rep["agreement"] == pytest.approx(2 / 3)
    assert rep["disagreements"] == [{"id": "c2", "a": "reasoning", "b": "against policy"}]


def test_script_confusion_and_agreement(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = [
        {
            "id": f"c{i}",
            "task_id": f"t{i // 2}",
            "failed": "fail" if i < 10 else "pass",
            "flagged": "yes" if i in (0, 1, 2, 3, 4, 5, 12) else "no",
        }
        for i in range(20)
    ]
    f = tmp_path / "checker.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "out.json"
    assert (
        label_metrics.main(
            [
                "confusion",
                str(f),
                "--truth",
                "failed",
                "--positive",
                "fail",
                "--pred",
                "flagged",
                "--cluster",
                "task_id",
                "--reps",
                "500",
                "--json",
                str(out),
            ]
        )
        == 0
    )
    rep = json.loads(out.read_text())
    assert rep["counts"] == {"tp": 6, "fp": 1, "fn": 4, "tn": 9}
    assert rep["metrics"]["recall"] == pytest.approx(0.6)
    assert "cluster bootstrap" in capsys.readouterr().out

    ka, kb = tmp_path / "katie.csv", tmp_path / "charlie.csv"
    ka.write_text("id,label\nc1,Tool\nc2,Arg\nc3,not a hallucination\n")
    kb.write_text("id,label\nc1,tool\nc2,Loop\nc3,not a hallucination\n")
    assert label_metrics.main(["agreement", str(ka), str(kb), "--label", "label", "--reps", "200"]) == 0
    printed = capsys.readouterr().out
    assert "c2: arg / loop" in printed  # labels are compared case-insensitively
