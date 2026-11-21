"""scripts/retention (SEL-84): the item-by-item comparison of two models' saved runs. No GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _scripts import load_script

compare = load_script("retention/compare")
run = load_script("retention/run")

CONFIG = """
models: {base: CohereLabs/aya-expanse-32b, sft2: sft2}
benchmarks:
  - {name: Instruction following (IFEval), tasks: [ifeval], score: prompt_level_strict_acc}
  - {name: Maths (GSM8K), tasks: [gsm8k], score: "exact_match,strict-match"}
  - {name: General knowledge (Global-MMLU-Lite), tasks: [global_mmlu_en, global_mmlu_fr], score: acc}
reps: 500
"""


def save_run(path: Path, rows: list[tuple[str, str, bool | None]], headline: dict | None = None) -> None:
    """A minimal AuditKIT RunResult JSON: the predictions and headline the comparison reads."""
    preds = [{"task": t, "sample_id": i, "correct": c} for t, i, c in rows]
    path.write_text(json.dumps({"run_id": path.stem, "predictions": preds, "headline": headline or {}}))


def gsm8k_two_copies(strict: list[bool], flexible: list[bool]) -> list[tuple[str, str, bool | None]]:
    """As AuditKIT saves GSM8K: every item once per filter, same IDs, strict-match first."""
    return [("gsm8k", str(i), c) for i, c in enumerate(strict)] + [
        ("gsm8k", str(i), c) for i, c in enumerate(flexible)
    ]


def test_the_copy_matching_the_headline_is_kept(tmp_path: Path) -> None:
    import yaml

    cfg = yaml.safe_load(CONFIG)
    strict = [True] * 6 + [False] * 4  # 60%
    flexible = [True] * 8 + [False] * 2  # 80%
    f = tmp_path / "base.json"
    save_run(f, gsm8k_two_copies(strict, flexible), {"gsm8k:exact_match,strict-match": 0.6})
    checks: list[str] = []
    items = compare.load_items(f, cfg, checks)
    assert len(items) == 10 and sum(items.values()) == 6  # strict kept, one copy per item
    assert "copy 1 of 2" in checks[0] and "✓" in checks[0]

    # If the strict copies came second, the second copy is the one kept
    save_run(f, gsm8k_two_copies(flexible, strict), {"gsm8k:exact_match,strict-match": 0.6})
    assert sum(compare.load_items(f, cfg).values()) == 6

    # Neither copy reproduces the headline: stop rather than guess
    save_run(f, gsm8k_two_copies(strict, flexible), {"gsm8k:exact_match,strict-match": 0.7})
    with pytest.raises(SystemExit):
        compare.load_items(f, cfg)


def test_groups_split_global_mmlu_by_language() -> None:
    import yaml

    labels = [label for label, _ in compare.groups(yaml.safe_load(CONFIG))]
    assert labels == [
        "Instruction following (IFEval)",
        "Maths (GSM8K)",
        "General knowledge (Global-MMLU-Lite)",
        "  en",
        "  fr",
    ]
    assert compare.matches("global_mmlu_fr_stem", ["global_mmlu_fr"])
    assert not compare.matches("global_mmlu_frx", ["global_mmlu_fr"])


def test_compare_verdicts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "retention.yaml"
    cfg.write_text(CONFIG)
    base, sft2 = [], []
    for i in range(200):
        base.append(("gsm8k", str(i), i < 160))  # maths: 80% → 50%, a clear drop
        sft2.append(("gsm8k", str(i), i < 100))
        base.append(("ifeval", str(i), i % 2 == 0))  # instructions: identical, no clear change
        sft2.append(("ifeval", str(i), i % 2 == 0))
    for lang in ("en", "fr"):  # knowledge: one sub-task per language, 40% → 90%, a clear rise
        for i in range(100):
            base.append((f"global_mmlu_{lang}_stem", str(i), i < 40))
            sft2.append((f"global_mmlu_{lang}_stem", str(i), i < 90))
    base.append(("gsm8k", "only-in-base", True))  # unpaired: left out
    sft2.append(("ifeval", "unscored", None))  # unscored: left out
    save_run(tmp_path / "base.json", base)
    save_run(tmp_path / "sft2.json", sft2)
    out = tmp_path / "out.json"
    assert (
        compare.main(["base", "sft2", "--config", str(cfg), "--dir", str(tmp_path), "--out", str(out)]) == 0
    )
    rep = json.loads(out.read_text())["benchmarks"]
    assert rep["Maths (GSM8K)"]["verdict"] == "drop"
    assert rep["Maths (GSM8K)"]["items"] == 200
    assert rep["Maths (GSM8K)"]["diff"] == pytest.approx(-0.3)
    assert rep["Maths (GSM8K)"]["only_first_right"] == 60
    assert rep["Instruction following (IFEval)"]["verdict"] == "no clear change"
    assert rep["General knowledge (Global-MMLU-Lite)"]["verdict"] == "rise"
    assert rep["General knowledge (Global-MMLU-Lite)"]["items"] == 200
    assert rep["en"]["items"] == 100
    printed = capsys.readouterr().out
    assert "drop" in printed and "rise" in printed


def test_run_rejects_an_unknown_model(tmp_path: Path) -> None:
    cfg = tmp_path / "retention.yaml"
    cfg.write_text(CONFIG)
    with pytest.raises(SystemExit):
        run.main(["sft9", "--config", str(cfg)])
