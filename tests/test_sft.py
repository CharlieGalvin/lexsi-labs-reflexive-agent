"""SEL-65: SFT tokenising (only the agent's turns are targets), the training script's dry run,
and the pod arm configs. No GPU, no model: a fake tokenizer renders turns like Aya's template."""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path
from typing import Any

import jinja2
import jinja2.sandbox
import pytest
import yaml
from _scripts import load_script

from reflexive import splits
from reflexive.arms import get_arm
from reflexive.files import ROOT
from reflexive.sft import (
    IGNORE,
    NotPrefixStable,
    default_template,
    encode,
    encode_all,
    pad_batch,
    relax_chat_template,
    relaxed_template,
)

ts = load_script("sft/train_sft")

ROLE = {"system": "SYSTEM", "user": "USER", "assistant": "CHATBOT"}


class FakeTokenizer:
    """One token per word, with turn markers like Cohere's template:
    <BOS> <S> SYSTEM ... <E> <S> CHATBOT ... <E> ...; the generation prompt is <S> CHATBOT."""

    pad_token_id = 0
    eos_token_id = 1
    chat_template = (  # carries the check, so the training script's relaxing has something to remove
        "{% if (m == 1) != (loop.index0 % 2 == 0) %}"
        "{{ raise_exception('Conversation roles must alternate') }}{% endif %}"
    )

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
    kept, stats = encode_all([short, long], tok, max_len=40, min_tokens=1)
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
    cfg.update(data=str(data), name="test-run", min_tokens=1)
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
    [
        ("P", "arms.base.yaml", "CohereLabs/aya-expanse-32b"),
        ("S", "arms.sft.yaml", "sft"),
        ("S2", "arms.sft2.yaml", "sft2"),
        ("SL", "arms.sftlong.yaml", "sftlong"),
        ("S3", "arms.sft3.yaml", "sft3"),
    ],
)
def test_the_pod_arms_match_arm_a_except_for_the_student_endpoint(key, file, model):
    pod, a = get_arm(key, ROOT / "configs" / "sft" / file), get_arm("A")
    assert pod.student["model"] == model and pod.student["api_base"] == "http://localhost:8000/v1"
    assert pod.student["api_key_env"] is None and pod.prompt_variant == a.prompt_variant == "default"
    same = ("mode", "style", "temperature", "max_tokens", "reserve")
    assert {k: pod.student[k] for k in same} == {k: a.student[k] for k in same} and pod.customer == a.customer


def _train_cfg(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / "train" / name).read_text())


def _without(cfg: dict, *keys: str) -> dict:
    """The config with these keys left out (``trainer.max_steps`` for a nested one)."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
    for k in keys:
        top, _, sub = k.partition(".")
        if sub:
            out[top].pop(sub, None)
        else:
            out.pop(top, None)
    return out


def test_v3_differs_from_v2_only_in_name_data_and_its_fixed_74_steps():
    v2, v3 = _train_cfg("sft_aya32b_v2.yaml"), _train_cfg("sft_aya32b_v3.yaml")
    assert v3["data"] == "data/train/sft-v3.jsonl" and v3["trainer"]["max_steps"] == 74
    assert _without(v3, "name", "data", "trainer.max_steps") == _without(v2, "name", "data")


def test_sl_is_v1_trained_for_v2s_74_steps():
    v1, sl = _train_cfg("sft_aya32b.yaml"), _train_cfg("sft_aya32b_long.yaml")
    assert sl["trainer"]["max_steps"] == 74
    assert _without(sl, "name", "trainer.max_steps") == _without(v1, "name")


# --- Aya's chat template: the agent speaks first --------------------------------------------------


# Cohere's chat template as it ships with Aya Expanse (tokenizer_config.json): system prompt, then
# turns that must alternate user / assistant
COHERE = (
    "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}"
    "{% set system_message = messages[0]['content'] %}{% elif false == true %}"
    "{% set loop_messages = messages %}"
    "{% set system_message = 'You are Aya.' %}{% else %}{% set loop_messages = messages %}"
    "{% set system_message = false %}{% endif %}{% if system_message != false %}"
    "{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}{% endif %}"
    "{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}"
    "{% set content = message['content'] %}{% if message['role'] == 'user' %}"
    "{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}"
)


def _raise(msg: str) -> None:
    raise jinja2.exceptions.TemplateError(msg)


def render(template: str, messages: list[dict[str, str]], gen: bool = False) -> str:
    env = jinja2.sandbox.ImmutableSandboxedEnvironment()
    return env.from_string(template).render(
        messages=messages, bos_token="<BOS_TOKEN>", add_generation_prompt=gen, raise_exception=_raise
    )


class JinjaTokenizer:
    """Renders with a real chat template; one token per special token or word."""

    def __init__(self, template: str) -> None:
        self.chat_template = template
        self.vocab: dict[str, int] = {}

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):  # noqa: ANN001, ANN201
        text = render(self.chat_template, conversation, add_generation_prompt)
        pieces = re.findall(r"<\|[A-Z_]+\|>|<[A-Z_]+>|[^\s<]+", text)  # special tokens split out first
        return [self.vocab.setdefault(w, len(self.vocab)) for w in pieces]


AGENT_FIRST = example()["messages"]  # system, then the agent's greeting: as every τ²-bench conversation


def test_the_shipped_template_refuses_a_conversation_that_starts_with_the_agent():
    with pytest.raises(jinja2.exceptions.TemplateError, match="must alternate"):
        render(COHERE, AGENT_FIRST)


def test_the_relaxed_template_renders_it_and_changes_nothing_else():
    relaxed = relaxed_template(COHERE)
    text = render(relaxed, AGENT_FIRST)
    assert text.startswith(
        "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>the policy<|END_OF_TURN_TOKEN|>"
    )
    assert (
        "<|CHATBOT_TOKEN|>hi there<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>change my order"
        in text
    )
    alternating = [AGENT_FIRST[0], *AGENT_FIRST[2:]]  # user first: the shipped template accepts it
    for gen in (False, True):
        assert render(relaxed, alternating, gen) == render(COHERE, alternating, gen)
    assert "raise_exception" not in relaxed and len(COHERE) - len(relaxed) < 200


def test_masking_works_with_the_relaxed_cohere_template():
    tok = JinjaTokenizer(relaxed_template(COHERE))
    enc = encode(example(), tok)
    back = {v: k for k, v in tok.vocab.items()}
    words = [back[i] for i, lab in zip(enc["input_ids"], enc["labels"], strict=True) if lab != IGNORE]
    assert words == [
        "hi",
        "there",
        "<|END_OF_TURN_TOKEN|>",
        "Action:",
        "get_order",
        "<|END_OF_TURN_TOKEN|>",
        "done",
        "<|END_OF_TURN_TOKEN|>",
    ]


def test_a_template_without_the_check_or_with_two_is_refused():
    with pytest.raises(ValueError, match="found 0"):
        relaxed_template("{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}")
    with pytest.raises(ValueError, match="found 2"):
        relaxed_template(COHERE + COHERE)


def test_named_templates_are_relaxed_where_they_have_the_check():
    named = {"default": COHERE, "tool_use": "{{ bos_token }}tools"}
    out = relax_chat_template(named)
    assert out["tool_use"] == named["tool_use"] and "raise_exception" not in out["default"]
    listed = relax_chat_template([{"name": "default", "template": COHERE}])
    assert default_template(listed) == relaxed_template(COHERE) == default_template(relaxed_template(COHERE))


# --- What apply_chat_template returns, by transformers version -----------------------------------

from collections import UserDict  # noqa: E402


class BatchEncodingTokenizer(FakeTokenizer):
    """transformers 5: a BatchEncoding (a mapping, not a dict) of a batch of one."""

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):  # noqa: ANN001, ANN201
        ids = super().apply_chat_template(
            conversation, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )
        return UserDict({"input_ids": [ids], "attention_mask": [[1] * len(ids)]})


def test_a_batch_encoding_is_read_as_the_token_ids_not_its_keys():
    plain, batch = encode(example(), FakeTokenizer()), encode(example(), BatchEncodingTokenizer())
    assert batch == plain and len(batch["input_ids"]) > 20


def test_a_batch_of_more_than_one_conversation_is_refused():
    class Two(FakeTokenizer):
        def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):  # noqa: ANN001, ANN201
            return {"input_ids": [[1, 2], [3, 4]]}

    with pytest.raises(ValueError, match="batch of 2"):
        encode(example(), Two())


def test_implausibly_short_conversations_or_no_targets_stop_the_run():
    """What the pod's first dry run showed (2 tokens a conversation): now refused, not reported."""
    with pytest.raises(ValueError, match="implausible tokenising"):
        encode_all([example()], FakeTokenizer(), max_len=10_000, min_tokens=64)
    assert encode_all([example()], FakeTokenizer(), max_len=10_000, min_tokens=8)[1]["examples"] == 1


def test_named_templates_without_the_check_are_refused():
    plain = "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}"
    with pytest.raises(ValueError, match="none of the named chat templates"):
        relax_chat_template({"default": plain, "tool_use": plain})
    with pytest.raises(ValueError, match="none of the named chat templates"):
        relax_chat_template([{"name": "default", "template": plain}])


def test_a_missing_default_template_is_named_not_a_bare_key_error():
    with pytest.raises(ValueError, match=r"no chat template named 'default'.*\['rag', 'tool_use'\]"):
        default_template({"tool_use": COHERE, "rag": COHERE})


def test_every_example_over_max_len_stops_the_run():
    with pytest.raises(ValueError, match=r"all 1 conversations are over max_len 5 tokens"):
        encode_all([example()], FakeTokenizer(), max_len=5, min_tokens=1)
