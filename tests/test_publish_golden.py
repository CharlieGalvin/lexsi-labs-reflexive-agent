"""SEL-48: the checks that must pass before golden trajectories are uploaded, and the data card."""

import copy
import sys

import pytest
from _scripts import load_script
from test_golden import SP, TOOLS, record

from reflexive import golden, splits

pg = load_script("data/publish_golden")

TRAIN = splits.task_ids("retail", "train")[0]
DEV = splits.task_ids("retail", "dev")[0]


def rows():
    tr = golden.to_record(record(domain="retail", task_id=TRAIN), SP, TOOLS, "train")
    dv = golden.to_record(record(domain="retail", task_id=DEV), SP, TOOLS, "dev")
    return {"train": [tr], "dev": [dv]}


def report(n_train=1, n_dev=1, unsolved_train=(), unsolved_dev=()):
    z = {"tasks": 0, "trajectories": 0, "unsolved": []}
    retail = {
        "train": {**z, "trajectories": n_train, "unsolved": list(unsolved_train)},
        "dev": {**z, "trajectories": n_dev, "unsolved": list(unsolved_dev)},
    }
    return {
        "domains": {"retail": retail},
        "train_sizes": {k: {"min": 1, "median": 1, "max": 1} for k in ("turns", "targets", "chars")},
    }


def test_clean_rows_pass():
    assert pg.validate(rows(), report()) == []


def test_catches_wrong_file_duplicates_and_counts():
    r = rows()
    r["train"].append(copy.deepcopy(r["dev"][0]))  # a dev record in the train file, also a duplicate id
    probs = pg.validate(r, report())
    assert any("belongs to dev" in p for p in probs)
    assert any("duplicated" in p for p in probs)
    assert any("both files" in p for p in probs)
    assert any("build report says" in p for p in probs)


def test_catches_bad_reward_and_targets():
    r = rows()
    r["train"][0]["outcome"]["reward"] = 0.0
    r["train"][0]["train_on"] = r["train"][0]["train_on"][1:]
    probs = pg.validate(r, report())
    assert any("reward" in p for p in probs) and any("train_on" in p for p in probs)


def test_malformed_record_is_a_problem_not_a_crash():
    r = rows()
    del r["train"][0]["outcome"]
    r["dev"][0]["task_id"] = "no-such-task"
    probs = pg.validate(r, report())
    assert any("malformed" in p for p in probs)
    assert any("belongs to no split" in p for p in probs)


@pytest.mark.parametrize("key", ["domain", "task_id", "id", "split", "messages"])
def test_record_missing_any_key_is_a_problem_not_a_crash(key):
    """Every field validate() reads, including the ones used to spot tasks in both files."""
    r = rows()
    del r["train"][0][key]
    probs = pg.validate(r, report())
    assert any(f"train record 1: malformed (KeyError: '{key}')" in p for p in probs), probs


def test_record_that_is_not_an_object_is_a_problem_not_a_crash():
    r = rows()
    r["train"][0] = ["not", "a", "record"]
    assert any("train record 1: malformed" in p for p in pg.validate(r, report()))


def test_card_counts_unsolved_train_and_dev_separately():
    tr, dv = splits.task_ids("retail", "train")[1:3], splits.task_ids("retail", "dev")[1:2]
    text = pg.card(rows(), report(unsolved_train=tr, unsolved_dev=dv), "org/name", {})
    assert "**2 train tasks have no trajectory**" in text
    assert "**1 dev tasks have no reference trajectory" in text
    assert "The format is set by the same prompt builder" in text


def test_card_has_front_matter_and_never_train_on_dev():
    text = pg.card(rows(), report(), "org/name", {"golden_train.jsonl": "abc"})
    assert text.startswith("---\n") and "split: dev_reference" in text
    assert "NEVER train on" in text and "`abc`" in text


class FakeApi:
    """Stands in for huggingface_hub.HfApi, keeping state like the Hub; never touches the network.

    Like the real Hub, creating a tag that exists fails (HTTP 409), deleting one removes it, and
    each upload is a new commit with its own ID.
    """

    existing: dict = {}  # repo → private?
    calls: list = []
    tags: list = []
    tagged_at: dict = {}  # tag → the commit it points to
    commits: int = 0
    offline: bool = False  # True: every call fails, as with no token or no network

    def repo_info(self, repo_id, repo_type):
        from huggingface_hub.utils import RepositoryNotFoundError

        if self.offline:
            raise ConnectionError("no network")
        if repo_id not in self.existing:
            # built without __init__, which needs an HTTP response object whose type varies by version
            raise RepositoryNotFoundError.__new__(RepositoryNotFoundError)
        return type("Info", (), {"private": self.existing[repo_id]})()

    def create_repo(self, repo_id, repo_type, private):
        self.calls.append(("create", repo_id, private))
        type(self).existing = {**self.existing, repo_id: private}

    def upload_folder(self, **kw):
        self.calls.append(("upload", sorted(kw["allow_patterns"])))
        type(self).commits += 1
        return type("CommitInfo", (), {"oid": f"commit-{self.commits}"})()

    def list_repo_refs(self, repo_id, repo_type):
        return type("Refs", (), {"tags": [type("Ref", (), {"name": t})() for t in self.tags]})()

    def delete_tag(self, repo_id, repo_type, tag):
        self.calls.append(("untag", tag))
        self.tags.remove(tag)  # ValueError if it isn't there, as the Hub refuses
        self.tagged_at.pop(tag, None)

    def create_tag(self, repo_id, repo_type, tag, revision=None):
        if tag in self.tags:
            raise RuntimeError(f"409 Conflict: tag {tag} already exists")
        self.calls.append(("tag", tag))
        self.tags.append(tag)
        self.tagged_at[tag] = revision


def _setup(tmp_path, monkeypatch, existing, tags=(), args=()):
    import json
    import shutil

    import huggingface_hub

    for split, recs in rows().items():
        (tmp_path / pg.FILES[split]).write_text("".join(json.dumps(r) + "\n" for r in recs))
    shutil.copy(splits.SPLITS_PATH, tmp_path / "splits.json")
    files = {n: pg.sha256(tmp_path / n) for n in [*pg.FILES.values(), "splits.json"]}
    (tmp_path / "build_report.json").write_text(json.dumps({**report(), "files": files}))
    FakeApi.existing, FakeApi.calls, FakeApi.tags = dict(existing), [], list(tags)
    FakeApi.tagged_at, FakeApi.commits, FakeApi.offline = {}, 0, False
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(sys, "argv", ["publish_golden.py", "--dir", str(tmp_path), "--repo", "org/ds", *args])


def test_upload_creates_private_repo_and_tags(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, existing={})
    assert pg.main() == 0
    assert FakeApi.calls[0] == ("create", "org/ds", True)
    assert FakeApi.calls[1] == ("upload", ["README.md", *sorted([*pg.FILES.values(), "splits.json"])])
    assert FakeApi.calls[2] == ("tag", "v1")


def test_refuses_public_repo(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, existing={"org/ds": False})
    assert pg.main() == 1 and FakeApi.calls == []


def test_refuses_to_reupload_over_existing_tag(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, existing={"org/ds": True}, tags=["v1"])
    assert pg.main() == 1 and FakeApi.calls == []


def test_retag_moves_existing_tag(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, existing={"org/ds": True}, tags=["v1"], args=["--retag"])
    assert pg.main() == 0
    assert [c[0] for c in FakeApi.calls] == ["upload", "untag", "tag"]
    assert FakeApi.tags == ["v1"] and FakeApi.tagged_at["v1"] == "commit-1"  # one v1, on the new data


def test_the_tag_points_at_the_commit_just_uploaded(tmp_path, monkeypatch):
    """Not "whatever is newest": an upload by someone else in between can't take the tag."""
    _setup(tmp_path, monkeypatch, existing={})
    FakeApi.commits = 41  # the repo's history before this upload
    assert pg.main() == 0
    assert FakeApi.tagged_at == {"v1": "commit-42"}


def test_dry_run_says_the_real_run_would_stop_on_an_existing_tag(tmp_path, monkeypatch, capsys):
    _setup(tmp_path, monkeypatch, existing={"org/ds": True}, tags=["v1"], args=["--dry-run"])
    assert pg.main() == 0 and FakeApi.calls == []  # a dry run uploads nothing
    out = capsys.readouterr().out
    assert "the real run would stop" in out and "already has tag v1" in out


def test_dry_run_still_checks_the_files_when_the_hub_cant_be_reached(tmp_path, monkeypatch, capsys):
    _setup(tmp_path, monkeypatch, existing={}, args=["--dry-run"])
    FakeApi.offline = True
    assert pg.main() == 0
    out = capsys.readouterr().out
    assert "no problems" in out and "couldn't check the Hub" in out


def test_refuses_files_changed_since_build(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, existing={})
    with (tmp_path / pg.FILES["train"]).open("a") as f:
        f.write("\n")
    assert pg.main() == 1 and FakeApi.calls == []


def test_refuses_stale_splits_copy(tmp_path):
    (tmp_path / "splits.json").write_text("{}")
    for n in pg.FILES.values():
        (tmp_path / n).write_text("")
    files = {n: pg.sha256(tmp_path / n) for n in [*pg.FILES.values(), "splits.json"]}
    probs = pg.check_files(tmp_path, {"files": files})
    assert any("differs from" in p for p in probs)
    assert any("no file checksums" in p for p in pg.check_files(tmp_path, {}))


def test_splits_order_keeps_file_order_across_train_and_dev():
    tr, dv = splits.task_ids("telecom", "train"), splits.task_ids("telecom", "dev")
    mixed = [dv[0], tr[3], tr[0], dv[3]]
    assert pg.splits_order("telecom", mixed) == [tr[0], tr[3], dv[0], dv[3]]
