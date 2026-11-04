"""The frozen splits (SEL-36): sizes, no overlaps, and τ²-bench's own splits kept intact."""

import json
import sys
from pathlib import Path

import pytest

from reflexive import splits

ROOT = Path(__file__).resolve().parents[1]
TAU2 = ROOT / "third_party" / "tau2-bench" / "data" / "tau2" / "domains"
SIZES = {"retail": (59, 15, 40), "airline": (24, 6, 20), "telecom": (59, 15, 40)}


@pytest.mark.parametrize("domain", splits.DOMAINS)
def test_sizes_and_disjoint(domain):
    tr, dv, te = (set(splits.task_ids(domain, s)) for s in splits.SPLITS)
    assert (len(tr), len(dv), len(te)) == SIZES[domain]
    assert not (tr & dv) and not (tr & te) and not (dv & te)


def test_totals():
    n = {s: sum(len(splits.task_ids(d, s)) for d in splits.DOMAINS) for s in splits.SPLITS}
    assert n == {"train": 142, "dev": 36, "test": 100}


def test_split_of_and_guard():
    test_id = splits.task_ids("retail", "test")[0]
    assert splits.split_of("retail", test_id) == "test"
    assert splits.split_of("airline", splits.task_ids("airline", "dev")[0]) == "dev"
    with pytest.raises(ValueError, match="TEST"):
        splits.assert_not_test("retail", test_id)
    splits.assert_not_test("retail", splits.task_ids("retail", "train")[0])
    with pytest.raises(KeyError):
        splits.split_of("retail", "no-such-task")


@pytest.mark.skipif(not TAU2.exists(), reason="needs the τ²-bench checkout in third_party/")
@pytest.mark.parametrize("domain", splits.DOMAINS)
def test_matches_official(domain):
    """train + dev is exactly τ²-bench's train; test is exactly τ²-bench's test."""
    off = json.loads((TAU2 / domain / "split_tasks.json").read_text())
    ours = set(splits.task_ids(domain, "train")) | set(splits.task_ids(domain, "dev"))
    assert ours == set(map(str, off["train"]))
    assert set(splits.task_ids(domain, "test")) == set(map(str, off["test"]))


@pytest.mark.skipif(not TAU2.exists(), reason="needs the τ²-bench checkout in third_party/")
def test_rebuild_matches_committed():
    """The dev pick is reproducible from the seed: nobody hand-edited it."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_splits

    assert make_splits.build()["domains"] == splits.load()["domains"]
