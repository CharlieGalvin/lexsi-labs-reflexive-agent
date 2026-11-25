"""The consent check (scripts/arms/consent_check.py): counting and the paired comparison. Stand-in
findings; no API calls."""

from __future__ import annotations

from _scripts import load_script

cc = load_script("arms/consent_check")


def rec(d, t, s, flags=0, broken=False):
    r = {
        "domain": d,
        "task_id": t,
        "sample": s,
        "termination": "user_stop",
        "outcome": {"passed": True},
        "_flags": flags,
    }
    if broken:
        r.update(termination="error", error="RateLimitError")
    return r


def findings(r):
    return [{"live": True}] * r["_flags"] + [{"live": False}] * (r["task_id"] == "9")


def test_counts_per_arm_leave_out_crashes_and_keep_errors_apart():
    rows = cc.per_conversation(
        [
            rec("retail", "1", 0, 1),
            rec("retail", "1", 1),
            rec("retail", "9", 0),
            rec("airline", "2", 0, broken=True),
        ],
        findings,
        count=lambda r: 2,
    )
    s = cc.summary(rows)["all"]
    assert s["conversations"] == 3 and s["flags"] == 1 and s["conversations_flagged"] == 1
    assert s["changes"] == 6 and s["flags_per_change"] == round(1 / 6, 4) and s["checker_errors"] == 1


def test_the_paired_change_and_whether_it_rises():
    base = [{"domain": "retail", "task_id": str(t), "flags": 1} for t in range(10)]
    same = [dict(r) for r in base]
    more = [dict(r, flags=2) for r in base]
    assert cc.paired(base, same, 500)["does_not_rise"] is True
    up = cc.paired(base, more, 500)
    assert up["change"] == 1.0 and up["does_not_rise"] is False and up["ci95"] == [1.0, 1.0]
