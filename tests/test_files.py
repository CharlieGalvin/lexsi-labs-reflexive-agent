"""reflexive.files: JSON Lines reading, checksums and printable paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from reflexive import files
from reflexive.files import ROOT, BadJsonLine, read_jsonl, resume_jsonl, sha256, shown


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    f.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert read_jsonl(f) == [{"a": 1}, {"a": 2}]


def test_a_half_written_line_names_the_file_and_line(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    f.write_text('{"a": 1}\n{"a": 2, "b"\n')
    with pytest.raises(BadJsonLine, match=r"x\.jsonl, line 2: not valid JSON"):
        read_jsonl(f)


def test_a_half_written_last_line_can_be_skipped_but_not_one_in_the_middle(tmp_path: Path, capsys) -> None:
    f = tmp_path / "x.jsonl"
    f.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "b"')  # a run stopped while writing record 3
    assert read_jsonl(f, partial_last_ok=True) == [{"a": 1}, {"a": 2}]
    assert "line 3 is half-written" in capsys.readouterr().err
    f.write_text('{"a": 1}\n{"a": 2, "b"\n{"a": 3}\n')
    with pytest.raises(BadJsonLine, match="line 2"):
        read_jsonl(f, partial_last_ok=True)


def test_resume_cuts_the_half_written_line_so_appending_carries_on(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    f.write_text('{"a": 1, "é": "x"}\n{"a": 2}\n{"a": 3, "b"')
    assert resume_jsonl(f) == [{"a": 1, "é": "x"}, {"a": 2}]
    assert f.read_text() == '{"a": 1, "é": "x"}\n{"a": 2}\n'  # the rest byte for byte
    with f.open("a") as out:
        out.write('{"a": 3}\n')
    assert read_jsonl(f) == [{"a": 1, "é": "x"}, {"a": 2}, {"a": 3}]  # not glued onto the bad line
    assert resume_jsonl(f) == read_jsonl(f)  # a clean file is left alone


def test_root_is_found_from_the_working_directory_when_not_installed_editable(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "splits.json").write_text("{}")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(files, "__file__", str(tmp_path / "venv" / "lib" / "reflexive" / "files.py"))
    monkeypatch.chdir(tmp_path / "scripts")
    assert files._repo_root() == tmp_path.resolve()
    assert ROOT == Path(__file__).resolve().parents[1]  # this checkout: installed editable


def test_sha256(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("abc")
    assert sha256(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_shown_is_relative_inside_the_repo_and_absolute_outside(tmp_path: Path) -> None:
    assert shown(ROOT / "data" / "golden" / "v1") == "data/golden/v1"
    assert shown(Path("configs/splits.json")) == "configs/splits.json"  # relative input (was a crash)
    assert shown(tmp_path) == str(tmp_path.resolve())
