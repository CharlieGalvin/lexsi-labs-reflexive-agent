"""SEL-98's script (scripts/labels/checker_vs_verdicts.py): counting, the error stop, the test guard.

Stand-in classifier and rule findings; no API calls.
"""

from __future__ import annotations

import json

import pytest
from _scripts import load_script

from reflexive import hallucination, intent
from reflexive.splits import task_ids

cvv = load_script("labels/checker_vs_verdicts")


def rec(task, sample, passed, tal=False, live=0, errors=0, broken=False):
    r = {
        "task_id": task,
        "sample": sample,
        "termination": "user_stop",
        "outcome": {"passed": passed},
        "_tal": tal,
        "_live": live,
        "_errors": errors,
    }
    if broken:
        r.update(termination="error", error="RateLimitError")
    return r


@pytest.fixture
def stand_ins(monkeypatch):
    """Tool/Arg/Loop and the rules read from each record's ``_tal``, ``_live`` and ``_errors``."""

    class Classified:
        def __init__(self, r):
            self.findings = ["tal"] if r["_tal"] else []

    def findings(r, rules=None):
        return [{"live": True, "rule": "mismatch", "reason": ""}] * r["_live"] + [
            {"live": False, "rule": "mismatch", "reason": "judge call failed"}
        ] * r["_errors"]

    monkeypatch.setattr(hallucination, "classify_record", Classified)
    monkeypatch.setattr(intent, "intent_findings", findings)


def write(tmp_path, domain, split, recs):
    f = tmp_path / f"S2-{domain}-{split}.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_counts_the_37_passes_and_failures_and_keeps_the_latest_copy(tmp_path, stand_ins):
    dev = task_ids("retail", "dev")
    write(
        tmp_path,
        "retail",
        "dev",
        [
            rec(dev[0], 0, False, live=1),  # one of the 37, flagged by a rule
            rec(dev[0], 1, False),  # one of the 37, not flagged
            rec(dev[1], 0, False, tal=True),  # a failure the classifier catches
            rec(dev[1], 1, True, live=1),  # a pass a rule flags
            rec(dev[2], 0, True, tal=True),  # a pass only the classifier flags
            rec(dev[2], 1, True, live=1),  # a pass whose first copy a rule flagged...
            rec(dev[2], 1, True),  # ...superseded by a clean copy: the latest counts
            rec(dev[3], 0, False, broken=True),  # a crash: left out
        ],
    )
    out = tmp_path / "m.json"
    assert cvv.main(["--episodes", str(tmp_path), "--json", str(out)]) == 0
    a = json.loads(out.read_text())["domains"]["all"]
    assert (a["the_37_flagged"]["k"], a["the_37_flagged"]["n"]) == (1, 2)
    assert (a["failures_flagged_intent"]["k"], a["failures_flagged_intent"]["n"]) == (1, 3)
    assert a["failures_flagged_checker"]["k"] == 2
    assert (a["passes_flagged_intent"]["k"], a["passes_flagged_intent"]["n"]) == (1, 3)
    assert a["passes_flagged_checker"]["k"] == 2
    assert json.loads(out.read_text())["findings_by_rule"] == {"mismatch": 2}


def test_a_rule_error_stops_without_writing(tmp_path, stand_ins):
    write(tmp_path, "retail", "dev", [rec(task_ids("retail", "dev")[0], 0, False, errors=1)])
    out = tmp_path / "m.json"
    assert cvv.main(["--episodes", str(tmp_path), "--json", str(out)]) == 1
    assert not out.exists()


def test_a_test_task_is_refused(tmp_path, stand_ins):
    write(tmp_path, "retail", "dev", [rec(task_ids("retail", "test")[0], 0, True)])
    with pytest.raises(ValueError, match="TEST task"):
        cvv.main(["--episodes", str(tmp_path)])


def test_no_conversations_writes_nothing(tmp_path, stand_ins):
    out = tmp_path / "m.json"
    assert cvv.main(["--episodes", str(tmp_path), "--json", str(out)]) == 1
    assert not out.exists()
