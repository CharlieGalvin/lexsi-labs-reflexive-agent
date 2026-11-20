"""scripts/labels/make_sheets.py (SEL-85, SEL-87): blind sheets from saved conversations (synthetic)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from _scripts import load_script

ms = load_script("labels/make_sheets")


def rec(task: str, passed: bool, *, arg: bool = False, loop: bool = False, tool: bool = False) -> dict:
    """A tiny retail conversation; optionally with an invented email (ARG), a repeated failed call
    (LOOP) or a call to a tool the agent doesn't have (TOOL)."""
    view = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "Hi, I want to return an order. My name is Ann Lee, zip 12345."},
    ]
    if arg:
        call = {
            "id": "c1",
            "type": "function",
            "function": {"name": "find_user_id_by_email", "arguments": json.dumps({"email": "made@up.com"})},
        }
        view += [
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "content": "Error: user not found"},
        ]
    if tool:
        call = {"id": "t1", "type": "function", "function": {"name": "teleport_order", "arguments": "{}"}}
        view += [
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "t1", "content": "Error: no such tool"},
        ]
    if loop:
        for k in range(2):
            call = {
                "id": f"l{k}",
                "type": "function",
                "function": {
                    "name": "find_user_id_by_name_zip",
                    "arguments": json.dumps({"first_name": "Ann", "last_name": "Lee", "zip": "12345"}),
                },
            }
            view += [
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": f"l{k}", "content": "Error: user not found"},
            ]
    view.append({"role": "assistant", "content": "Done."})
    return {
        "domain": "retail",
        "task_id": task,
        "sample": 0,
        "termination": "user_stop",
        "outcome": {"passed": passed, "checks": {"db": 0.0 if not passed else 1.0}},
        "agent_view": view,
        "messages": view,
    }


def write(path: Path, recs: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_sheets(tmp_path: Path) -> None:
    ep = tmp_path / "episodes"
    ep.mkdir()
    write(
        ep / "P-retail-dev.jsonl",
        [rec(f"p{i}", False, arg=True) for i in range(5)]
        + [rec("p-loop", False, loop=True), rec("p-tool", False, tool=True), rec("p-ok", True)],
    )
    write(
        ep / "S2-retail-dev.jsonl",
        [rec("s-none-1", False), rec("s-none-2", False), rec("s-pass", True), rec("s-arg", False, arg=True)],
    )
    write(ep / "S2-retail-val.jsonl", [rec("s-none-3", False), {**rec("s-crash", False), "error": "boom"}])
    out = tmp_path / "labels"
    assert (
        ms.main(
            [
                "--episodes",
                str(ep),
                "--out",
                str(out),
                "--per-type",
                "3",
                "--recall",
                "4",
                "--expect-decisions",
                "3",
            ]
        )
        == 0
    )

    # Decisions: v2's failed conversations with no finding (crashes left out), blind sheet + key
    dec = list(csv.DictReader((out / "decisions-katie.csv").open()))
    assert len(dec) == 3 and set(dec[0]) == {"id", "category", "notes"}
    key = {r["id"]: r for r in csv.DictReader((out / "decisions-key.csv").open())}
    assert {k["task_id"] for k in key.values()} == {"s-none-1", "s-none-2", "s-none-3"}
    assert all(k["model"] == "sft2" and k["classifier"] == "no" for k in key.values())
    assert "FAILED" in (out / "decisions" / "items" / "D01.md").read_text()

    # Precision: the claim is visible, the model isn't
    prec = list(csv.DictReader((out / "audit-precision-charlie.csv").open()))
    types = sorted(r["claimed_type"] for r in prec)
    assert types.count("ARG") == 3 and types.count("LOOP") == 1 and types.count("TOOL") == 1
    assert "model" not in prec[0]
    item = (out / "audit-precision" / "items" / f"{prec[0]['id']}.md").read_text()
    assert "FLAGGED TURN" in item and "sft2" not in item and "S2-" not in item

    # Recall: no hint of the classifier's verdict on the labeller's sheet
    rows = list(csv.DictReader((out / "audit-recall-katie.csv").open()))
    assert len(rows) == 4 and set(rows[0]) == {"id", "label", "types_seen", "notes"}
    assert "classifier" in next(iter(csv.DictReader((out / "audit-recall-key.csv").open())))


def test_rerun_keeps_filled_sheets_and_is_reproducible(tmp_path: Path) -> None:
    ep = tmp_path / "episodes"
    ep.mkdir()
    write(ep / "S2-retail-dev.jsonl", [rec(f"s{i}", False) for i in range(4)])
    out = tmp_path / "labels"
    args = [
        "--episodes",
        str(ep),
        "--out",
        str(out),
        "--expect-decisions",
        "4",
        "--per-type",
        "1",
        "--recall",
        "2",
    ]
    ms.main(args)
    first_key = (out / "decisions-key.csv").read_text()
    sheet = out / "decisions-katie.csv"
    rows = list(csv.DictReader(sheet.open()))
    rows[0]["category"] = "no yes"
    with sheet.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["id", "category", "notes"])
        w.writeheader()
        w.writerows(rows)
    ms.main(args)
    assert "no yes" in sheet.read_text()  # a labelled sheet is never overwritten
    assert (out / "decisions-key.csv").read_text() == first_key  # same seed, same sheets
