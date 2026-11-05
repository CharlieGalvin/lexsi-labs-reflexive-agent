"""reflexive.files: JSON Lines reading, checksums and printable paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from reflexive.files import ROOT, BadJsonLine, read_jsonl, sha256, shown


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    f.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert read_jsonl(f) == [{"a": 1}, {"a": 2}]


def test_a_half_written_line_names_the_file_and_line(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    f.write_text('{"a": 1}\n{"a": 2, "b"\n')
    with pytest.raises(BadJsonLine, match=r"x\.jsonl, line 2: not valid JSON"):
        read_jsonl(f)


def test_sha256(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("abc")
    assert sha256(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_shown_is_relative_inside_the_repo_and_absolute_outside(tmp_path: Path) -> None:
    assert shown(ROOT / "data" / "golden" / "v1") == "data/golden/v1"
    assert shown(Path("configs/splits.json")) == "configs/splits.json"  # relative input (was a crash)
    assert shown(tmp_path) == str(tmp_path.resolve())
