"""The overnight runner's completeness check (scripts/pod/overnight.py): what counts as finished."""

from __future__ import annotations

import json

from _scripts import load_script

from reflexive.splits import task_ids

ov = load_script("pod/overnight")


def write(folder, arm, domain, recs):
    (folder / f"{arm}-{domain}-test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def rec(d, t, s, broken=False):
    r = {"domain": d, "task_id": t, "sample": s, "termination": "user_stop", "outcome": {"passed": True}}
    if broken:
        r.update(termination="error", error="RateLimitError")
    return r


def test_complete_only_when_every_task_has_every_sample_scored(tmp_path):
    for d in ov.DOMAINS:
        write(tmp_path, "X", d, [rec(d, str(t), s) for t in task_ids(d, "test") for s in range(2)])
    assert ov.complete(ov.status("X", "test", 2, tmp_path))
    d = "telecom"
    ts = [str(t) for t in task_ids(d, "test")]
    recs = [rec(d, t, s) for t in ts for s in range(2)]
    recs.append(rec(d, ts[0], 1, broken=True))  # the last record for that conversation crashed
    write(tmp_path, "X", d, recs)
    st = ov.status("X", "test", 2, tmp_path)
    assert not ov.complete(st) and st[d]["short"] == [ts[0]] and st[d]["crashed"] == 1


def test_missing_files_are_incomplete_and_no_pod_id_means_no_stop(tmp_path, monkeypatch):
    assert not ov.complete(ov.status("Y", "test", 2, tmp_path))
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    assert ov.stop_pod().startswith("not stopped")
    monkeypatch.setenv("RUNPOD_POD_ID", "abc")
    assert ov.stop_pod(dry=True) == "would stop pod abc"
