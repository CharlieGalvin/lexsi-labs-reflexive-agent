"""Steps for features/publishing.feature: `just golden-publish` checks everything before uploading.

Uses the fake Hugging Face API and folder set-up from test_publish_golden.py: nothing is sent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when
from test_publish_golden import FakeApi, _setup, pg, report, rows

from reflexive import splits

scenarios("features/publishing.feature")


@dataclass
class Upload:
    """The folder to publish, what the Hub already has, and what happened."""

    folder: Path
    monkeypatch: pytest.MonkeyPatch
    existing: dict[str, bool] = field(default_factory=dict)  # repo → private?
    tags: list[str] = field(default_factory=list)
    edit_after_build: bool = False
    exit_code: int | None = None
    records: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


@given("a golden trajectories folder that passed its build checks", target_fixture="up")
def built_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Upload:
    return Upload(tmp_path, monkeypatch, records=rows())


@given("the dataset doesn't exist yet on Hugging Face")
def no_dataset(up: Upload) -> None:
    up.existing = {}


@given("the dataset already exists on Hugging Face and is public")
def public_dataset(up: Upload) -> None:
    up.existing = {"org/ds": False}


@given(parsers.parse('the dataset already exists on Hugging Face, privately, with the tag "{tag}"'))
def tagged_dataset(up: Upload, tag: str) -> None:
    up.existing, up.tags = {"org/ds": True}, [tag]


@given("a record file was edited after the build")
def edited(up: Upload) -> None:
    up.edit_after_build = True


@when("it is published")
def publish(up: Upload) -> None:
    _setup(up.folder, up.monkeypatch, existing=up.existing, tags=up.tags)
    if up.edit_after_build:
        with (up.folder / pg.FILES["train"]).open("a") as f:
            f.write("\n")
    up.exit_code = pg.main()


@then("the upload succeeds")
def succeeded(up: Upload) -> None:
    assert up.exit_code == 0


@then("the dataset is created private")
def created_private(up: Upload) -> None:
    assert ("create", "org/ds", True) in FakeApi.calls


@then("the data card, both record files and the splits are uploaded")
def uploaded_files(up: Upload) -> None:
    assert ("upload", sorted(["README.md", *pg.FILES.values(), "splits.json"])) in FakeApi.calls


@then(parsers.parse('the upload is tagged "{tag}"'))
def tagged(up: Upload, tag: str) -> None:
    assert FakeApi.calls[-1] == ("tag", tag)


@then("nothing is uploaded")
def nothing(up: Upload) -> None:
    assert up.exit_code == 1 and FakeApi.calls == []


# --- The record checks ----------------------------------------------------------------------------


@given("one of the training records is from the test split")
def a_test_split_record(up: Upload) -> None:
    rec = up.records["train"][0]
    rec["task_id"] = splits.task_ids(rec["domain"], "test")[0]  # what a mistake upstream could do


@when("the records are checked")
def check(up: Upload) -> None:
    up.problems = pg.validate(up.records, report())


@then(parsers.parse('the check reports a problem mentioning "{word}"'))
def reported(up: Upload, word: str) -> None:
    assert any(word in p for p in up.problems), up.problems
