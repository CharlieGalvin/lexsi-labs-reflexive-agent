"""SEL-65: SFT tokenising (only the agent's turns are targets), the training script's dry run,
and the pod arm configs. No GPU, no model: a fake tokenizer renders turns like Aya's template."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml
from _scripts import load_script

from reflexive import splits
from reflexive.arms import get_arm
from reflexive.files import ROOT
from reflexive.sft import IGNORE, NotPrefixStable, encode, encode_all, pad_batch

ts = load_script("sft/train_sft")

ROLE = {"system": "SYSTEM", "user": "USER", "assistant": "CHATBOT"}


class FakeTokenizer:
    """One token per word, with turn markers like Cohere's template:
    <BOS> <S> SYSTEM ... <E> <S> CHATBOT ... <E> ...; the generation prompt is <S> CHATBOT."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def tid(self, w: str) -> int:
        return self.vocab.setdefault(w, len(self.vocab) + 2)

    def render(self, conversation: list[dict[str, str]], add_generation_prompt: bool) -> list[str]:
        words = ["<BOS>"]
        for m in conversation:
            words += ["<S>", ROLE[m["role"]], *m["content"].split(), "<E>"]
        return words + (["<S>", "CHATBOT"] if add_generation_prompt else [])

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):  # noqa: ANN001, ANN201
        return [self.tid(w) for w in self.render(conversation, add_generation_prompt)]

    def words(self, ids: list[int]) -> list[str]:
        back = {v: k for k, v in self.vocab.items()}
        return [back[i] for i in ids]


def example(**kw: Any) -> dict[str, Any]:
    msgs = [
        {"role": "system", "content": "the policy"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "change my order"},
        {"role": "assistant", "content": "Action: get_order"},
        {"role": "user", "content": "results here"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "bye ###STOP###"},
    ]
    e = {"id": "glm53f/retail/0/1", "messages": msgs, "train_on": [1, 3, 5]}
    e.update(kw)
    return e


def targets(tok: FakeTokenizer, enc: dict[str, list[int]]) -> list[str]:
    return tok.words([i for i, lab in zip(enc["input_ids"], enc["labels"], strict=True) if lab != IGNORE])


def test_only_the_agents_words_and_end_of_turn_are_targets():
    tok = FakeTokenizer()
    enc = encode(example(), tok)
    assert len(enc["input_ids"]) == len(enc["labels"])
    assert targets(tok, enc) == ["hi", "there", "<E>", "Action:", "get_order", "<E>", "done", "<E>"]
    words = tok.words(enc["input_ids"])
    assert words == tok.render(example()["messages"], add_generation_prompt=False)  # the whole chat is input
    for w in ("policy", "change", "results", "###STOP###", "CHATBOT"):  # context and headers aren't targets
        assert w not in targets(tok, enc)


def test_a_turn_left_out_of_train_on_is_context():
    tok = FakeTokenizer()
    enc = encode(example(train_on=[5]), tok)
    assert targets(tok, enc) == ["done", "<E>"]


def test_train_on_naming_a_customer_turn_is_refused():
    with pytest.raises(ValueError, match="isn't the agent's"):
        encode(example(train_on=[1, 2]), FakeTokenizer())


def test_a_template_that_rewrites_earlier_turns_is_refused():
    class Unstable(FakeTokenizer):
        def render(self, conversation, add_generation_prompt):  # noqa: ANN001, ANN201
            return [f"<turns={len(conversation)}>", *super().render(conversation, add_generation_prompt)[1:]]

    with pytest.raises(NotPrefixStable, match="rewrote earlier tokens"):
        encode(example(), Unstable())


def test_a_conversation_over_the_limit_is_left_out_whole_not_cut():
    tok = FakeTokenizer()
    short = example()
    long = example(id="long", messages=[*example()["messages"][:-1], {"role": "user", "content": "x " * 50}])
    kept, stats = encode_all([short, long], tok, max_len=40)
    assert stats["examples"] == 1 and [t["id"] for t in stats["too_long"]] == ["long"]
    assert len(kept[0]["input_ids"]) <= 40 and 0 < stats["target_share"] < 1


def test_padding_masks_attention_and_loss():
    b = pad_batch(
        [{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}, {"input_ids": [5], "labels": [5]}], pad_id=0
    )
    assert (
        b["input_ids"][1] == [5, 0, 0]
        and b["attention_mask"][1] == [1, 0, 0]
        and b["labels"][1] == [5, -100, -100]
    )


# --- just sft-train --dry-run ----------------------------------------------------------------------


def _data(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    f = tmp_path / "sft-all.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return f


def _rows() -> list[dict[str, Any]]:
    ids = splits.task_ids("retail", "train")
    return [
        {**example(id=f"g/{t}"), "domain": "retail", "task_id": t, "group": g}
        for t, g in zip(
            ids[:5], ["healable", "healable", "healable", "solved_by_aya", "healable"], strict=True
        )
    ]


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]], *args: str) -> int:
    data = _data(tmp_path, rows)
    cfg = yaml.safe_load((ROOT / "configs" / "train" / "sft_aya32b.yaml").read_text())
    cfg.update(data=str(data), name="test-run")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    fake = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda _: FakeTokenizer())
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(ts, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_sft.py", str(path), *args])
    return ts.main()


def test_the_dry_run_tokenises_the_healable_examples_and_holds_out_the_rest(tmp_path, monkeypatch, capsys):
    assert _run(tmp_path, monkeypatch, _rows(), "--dry-run") == 0
    out = capsys.readouterr().out
    assert "train: 4 conversations" in out and "held out: 1" in out
    assert not (tmp_path / "runs").exists()  # a dry run writes nothing


def test_the_training_script_refuses_a_dev_or_test_task(tmp_path, monkeypatch):
    rows = _rows()
    rows[0]["task_id"] = splits.task_ids("retail", "test")[0]
    with pytest.raises(ValueError, match="TEST task"):
        _run(tmp_path, monkeypatch, rows, "--dry-run")
    rows[0]["task_id"] = splits.task_ids("retail", "dev")[0]
    with pytest.raises(SystemExit, match="isn't a train task"):
        _run(tmp_path, monkeypatch, rows, "--dry-run")


def test_the_training_script_refuses_to_overwrite_a_finished_run(tmp_path, monkeypatch, capsys):
    (tmp_path / "runs" / "test-run").mkdir(parents=True)
    (tmp_path / "runs" / "test-run" / "adapter_config.json").write_text("{}")
    assert _run(tmp_path, monkeypatch, _rows(), "--dry-run") == 1
    assert "--force" in capsys.readouterr().out


def test_the_config_trains_on_healable_and_holds_out_solved_by_aya():
    cfg = yaml.safe_load((ROOT / "configs" / "train" / "sft_aya32b.yaml").read_text())
    assert (cfg["train_group"], cfg["eval_group"]) == ("healable", "solved_by_aya")
    assert cfg["max_len"] >= 12_800  # the longest golden fix is ~12.7K tokens: none is cut


# --- The pod arms --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "file", "model"),
    [("P", "arms.base.yaml", "CohereLabs/aya-expanse-32b"), ("S", "arms.sft.yaml", "sft")],
)
def test_the_pod_arms_match_arm_a_except_for_the_student_endpoint(key, file, model):
    pod, a = get_arm(key, ROOT / "configs" / "sft" / file), get_arm("A")
    assert pod.student["model"] == model and pod.student["api_base"] == "http://localhost:8000/v1"
    assert pod.student["api_key_env"] is None and pod.prompt_variant == a.prompt_variant == "default"
    same = ("mode", "style", "temperature", "max_tokens", "reserve")
    assert {k: pod.student[k] for k in same} == {k: a.student[k] for k in same} and pod.customer == a.customer
