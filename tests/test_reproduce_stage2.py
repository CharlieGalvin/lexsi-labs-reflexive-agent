"""Stage 2's reproduction (scripts/blog/reproduce_stage2.py): the check, offline."""

from __future__ import annotations

from _scripts import load_script

rs = load_script("blog/reproduce_stage2")


def test_a_match_to_the_reported_rounding_passes():
    res = {"v2": 50.12, "candidate": 60.0, "gain": 9.88, "ci95": [3.88, 16.0]}
    assert rs.check(res, {"gain": 9.88, "ci95": [3.88, 16.0], "v2": 50.12}) == []


def test_any_difference_is_reported():
    res = {"gain": 9.5, "ci95": [3.9, 16.0]}
    off = rs.check(res, {"gain": 9.88, "ci95": [3.88, 16.0]})
    assert len(off) == 2 and "gain" in off[0] and "95% range" in off[1]


def test_every_claim_names_two_arms_a_split_and_what_it_reports():
    for claim, base, cand, split, samples, min_sample, expected in rs.CLAIMS:
        assert base != cand and split in {"test", "val"} and samples + min_sample == 8
        assert "gain" in expected and "ci95" in expected, claim
