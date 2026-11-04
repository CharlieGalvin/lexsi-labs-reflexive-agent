"""SEL-48: the checks that must pass before golden trajectories are uploaded, and the data card."""

import copy
import sys
from pathlib import Path

from reflexive import golden, splits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import publish_golden as pg  # noqa: E402
from test_golden import SP, TOOLS, record  # noqa: E402

TRAIN = splits.task_ids("retail", "train")[0]
DEV = splits.task_ids("retail", "dev")[0]


def rows():
    tr = golden.to_record(record(domain="retail", task_id=TRAIN), SP, TOOLS, "train")
    dv = golden.to_record(record(domain="retail", task_id=DEV), SP, TOOLS, "dev")
    return {"train": [tr], "dev": [dv]}


def report(n_train=1, n_dev=1):
    z = {"tasks": 0, "trajectories": 0, "unsolved": []}
    retail = {"train": {**z, "trajectories": n_train}, "dev": {**z, "trajectories": n_dev}}
    return {"domains": {"retail": retail},
            "train_sizes": {k: {"min": 1, "median": 1, "max": 1} for k in ("turns", "targets", "chars")}}


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


def test_card_has_front_matter_and_never_train_on_dev():
    text = pg.card(rows(), report(), "org/name", {"golden_train.jsonl": "abc"})
    assert text.startswith("---\n") and "split: dev_reference" in text
    assert "NEVER train on" in text and "`abc`" in text


class FakeApi:
    """Stands in for huggingface_hub.HfApi: records calls, never touches the network."""

    existing: dict = {}
    calls: list = []

    def repo_info(self, repo_id, repo_type):
        from huggingface_hub.utils import RepositoryNotFoundError

        if repo_id not in self.existing:
            # built without __init__, which needs an HTTP response object whose type varies by version
            raise RepositoryNotFoundError.__new__(RepositoryNotFoundError)
        return type("Info", (), {"private": self.existing[repo_id]})()

    def create_repo(self, repo_id, repo_type, private):
        self.calls.append(("create", repo_id, private))

    def upload_folder(self, **kw):
        self.calls.append(("upload", sorted(kw["allow_patterns"])))

    def create_tag(self, repo_id, repo_type, tag, exist_ok):
        self.calls.append(("tag", tag))


def _setup(tmp_path, monkeypatch, existing):
    import json

    import huggingface_hub

    for split, recs in rows().items():
        (tmp_path / pg.FILES[split]).write_text("".join(json.dumps(r) + "\n" for r in recs))
    (tmp_path / "splits.json").write_text("{}")
    (tmp_path / "build_report.json").write_text(json.dumps(report()))
    FakeApi.existing, FakeApi.calls = existing, []
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(sys, "argv", ["publish_golden.py", "--dir", str(tmp_path), "--repo", "org/ds"])


def test_upload_creates_private_repo_and_tags(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, existing={})
    assert pg.main() == 0
    assert FakeApi.calls[0] == ("create", "org/ds", True)
    assert FakeApi.calls[1] == ("upload", ["README.md", *sorted([*pg.FILES.values(), "splits.json"])])
    assert FakeApi.calls[2] == ("tag", "v1")


def test_refuses_public_repo(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, existing={"org/ds": False})
    assert pg.main() == 1 and FakeApi.calls == []
