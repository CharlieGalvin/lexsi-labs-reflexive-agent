"""SEL-21: the held-out reward check for single-turn GRPO.

Everything but the sampling itself, which needs the GPU: the sampler is replaced by one that
returns fixed replies, so the scoring, the summary and the comparison run on real reward code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from _scripts import load_script

from reflexive import grpo_data, grpo_rewards

hc = load_script("grpo/heldout_check")

FIND = {
    "name": "find_user_id_by_name_zip",
    "arguments": {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"},
}
PROMPT = [
    {"role": "system", "content": "policy and tools"},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "I'm Yusuf Rossi, zip 19122."},
]
RIGHT = (
    'Action: ```json\n[{"tool_name": "find_user_id_by_name_zip", "parameters": '
    '{"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}}]\n```'
)
WRONG_ARGS = RIGHT.replace("19122", "10001")
GIVE_UP = "Sorry."


def _row(task_id: str, turn: int = 3, **kw: Any) -> dict[str, Any]:
    return {
        "id": f"x/retail/{task_id}/1#t{turn}",
        "format": grpo_data.FORMAT,
        "domain": "retail",
        "task_id": task_id,
        "group": "healable",
        "turn": turn,
        "kind": "call",
        "prompt": PROMPT,
        "target": RIGHT,
        "target_calls": [FIND],
        "source": {},
        **kw,
    }


def _write(path: Path, *rows: dict[str, Any]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


@pytest.fixture
def cfg(tmp_path: Path) -> dict[str, Any]:
    """The real config; training file: task 2; held-out file: tasks 0 and 3 (all train tasks)."""
    train = _write(tmp_path / "grpo-x.jsonl", _row("2"))
    _write(tmp_path / "grpo-x.heldout.jsonl", _row("0", 3), _row("0", 5), _row("3", 3))
    return {**hc.ts.load(hc.ts.CONFIG), "data": train, "name": "grpo-test"}


def test_the_held_out_file_sits_beside_the_training_file():
    assert (
        hc.heldout_data({"data": "data/train/grpo-healable.jsonl"})
        == "data/train/grpo-healable.heldout.jsonl"
    )
    assert hc.heldout_data(hc.ts.load(hc.ts.CONFIG)) == "data/train/grpo-healable.heldout.jsonl"


def test_held_out_prompts_load_and_limit(cfg: dict[str, Any]):
    rows = hc.read_eval_rows(cfg, hc.heldout_data(cfg))
    assert [r["id"] for r in rows] == ["x/retail/0/1#t3", "x/retail/0/1#t5", "x/retail/3/1#t3"]
    assert len(hc.read_eval_rows(cfg, hc.heldout_data(cfg), limit=2)) == 2


def test_a_trained_task_in_the_held_out_file_is_refused(cfg: dict[str, Any], tmp_path: Path):
    leaky = _write(tmp_path / "leaky.jsonl", _row("0"), _row("2", 7))
    with pytest.raises(SystemExit, match="trained on"):
        hc.read_eval_rows(cfg, leaky)


def test_the_training_file_itself_can_be_checked(cfg: dict[str, Any]):
    assert [r["task_id"] for r in hc.read_eval_rows(cfg, cfg["data"])] == ["2"]


def test_a_dev_task_is_refused(cfg: dict[str, Any], tmp_path: Path):
    with pytest.raises(SystemExit, match="isn't a train task"):
        hc.read_eval_rows(cfg, _write(tmp_path / "dev.jsonl", _row("1")))


def test_default_adapters_are_sft_v2_then_the_grpo_run_once_trained(
    cfg: dict[str, Any], tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(hc, "ROOT", tmp_path)
    assert hc.parse_adapters(None, cfg) == {"sft-v2": "reagent-lexsi/sft-aya32b-r16-v2"}
    run = tmp_path / "runs" / "grpo-test"
    run.mkdir(parents=True)
    (run / "adapter_config.json").write_text("{}")
    assert list(hc.parse_adapters(None, cfg)) == ["sft-v2", "grpo"]


def test_adapters_given_as_name_equals_path_keep_their_order(cfg: dict[str, Any]):
    got = hc.parse_adapters(["b=runs/b", "a=org/a"], cfg)
    assert list(got.items()) == [("b", "runs/b"), ("a", "org/a")]
    with pytest.raises(SystemExit, match="name=path"):
        hc.parse_adapters(["runs/b"], cfg)
    with pytest.raises(SystemExit, match="twice"):
        hc.parse_adapters(["a=x", "a=y"], cfg)


def test_samples_are_scored_with_the_training_reward(cfg: dict[str, Any]):
    rows = hc.read_eval_rows(cfg, hc.heldout_data(cfg), limit=1)
    w = grpo_rewards.DEFAULT_WEIGHTS
    right, wrong = hc.score_samples(rows, [[RIGHT, WRONG_ARGS]], w)
    assert right["reward"] == pytest.approx(sum(w.values())) and right["exact"]
    assert wrong["args_match"] < 1 and not wrong["exact"] and wrong["reward"] < right["reward"]
    assert (right["task_id"], right["sample"], wrong["sample"]) == ("0", 0, 1)
    with pytest.raises(ValueError, match="sample lists"):
        hc.score_samples(rows, [], w)


def _fake_sampler(replies: dict[str, list[str]]):
    seen = []

    def sample(rows: list[dict[str, Any]], name: str, path: str) -> list[list[str]]:
        seen.append((name, path, len(rows)))
        return [list(replies[name]) for _ in rows]

    return sample, seen


def test_run_compares_each_adapter_with_the_first_on_the_same_prompts(cfg: dict[str, Any], tmp_path: Path):
    rows = hc.read_eval_rows(cfg, hc.heldout_data(cfg))
    sampler, seen = _fake_sampler({"sft-v2": [RIGHT, WRONG_ARGS], "grpo": [RIGHT, RIGHT]})
    w = grpo_rewards.DEFAULT_WEIGHTS
    out = tmp_path / "heldout"
    summary = hc.run(rows, {"sft-v2": "org/sft", "grpo": "runs/g"}, sampler, w, out)

    assert seen == [("sft-v2", "org/sft", 3), ("grpo", "runs/g", 3)]
    sft, grpo = summary["adapters"]["sft-v2"], summary["adapters"]["grpo"]
    assert (sft["prompts"], sft["tasks"], sft["samples"]) == (3, 2, 6)
    assert sft["exact"] == 0.5 and grpo["exact"] == 1.0
    assert sft["all_same"] == 0.0 and grpo["all_same"] == 1.0  # every grpo group scores the same
    assert grpo["reward"] == pytest.approx(summary["max_reward"]) and grpo["reward_share"] == 1.0
    d = summary["differences"]["grpo"]
    assert d["against"] == "sft-v2" and d["reward_diff"] > 0 and d["tasks"] == 2
    assert d["ci95"][0] == pytest.approx(d["reward_diff"], abs=1e-3)  # the same gain on every task

    logged = [json.loads(line) for line in (out / "grpo.jsonl").read_text().splitlines()]
    assert len(logged) == 6 and set(grpo_rewards.COMPONENTS) <= set(logged[0])
    md = hc.to_markdown(summary)
    assert "| sft-v2 | 3 |" in md and "**grpo − sft-v2:** +" in md


def test_one_adapter_summarises_without_a_comparison(cfg: dict[str, Any], tmp_path: Path):
    rows = hc.read_eval_rows(cfg, hc.heldout_data(cfg))
    sampler, _ = _fake_sampler({"sft-v2": [GIVE_UP]})
    summary = hc.run(rows, {"sft-v2": "org/sft"}, sampler, grpo_rewards.DEFAULT_WEIGHTS, tmp_path)
    assert summary["differences"] == {} and summary["adapters"]["sft-v2"]["parts"]["no_give_up"] == 0.0
    assert "−" not in hc.to_markdown(summary)


class _Tok:
    eos_token_id, unk_token_id = 3, 0

    def __init__(self, eot: int):
        self.eot = eot

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.eot


def test_replies_stop_at_aya_s_end_of_turn_too():
    assert hc.stop_ids(_Tok(eot=255001)) == [3, 255001]
    assert hc.stop_ids(_Tok(eot=0)) == [3]  # no such token: unknown
    assert hc.stop_ids(_Tok(eot=3)) == [3]


def test_dry_run_checks_and_reports_without_loading_anything(
    cfg: dict[str, Any], tmp_path: Path, monkeypatch, capsys
):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["heldout_check", str(path), "--dry-run", "--limit", "2"])
    monkeypatch.setattr(hc, "make_sampler", lambda *a, **k: pytest.fail("loaded a model in a dry run"))
    assert hc.main() == 0
    out = capsys.readouterr().out
    assert "2 prompts from" in out and "(1 tasks), 4 samples each at temperature 1.0" in out
    assert "sft-v2=reagent-lexsi/sft-aya32b-r16-v2" in out and "no comparison" in out


def test_the_sampler_switches_adapters_on_one_model_and_is_reproducible(
    cfg: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The real sampling code on a tiny random Llama, two LoRA adapters, on CPU (skipped without torch)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    bytes_ = pre_tokenizers.ByteLevel.alphabet()  # a byte-level tokenizer built here: no download
    vocab = {t: i for i, t in enumerate(["<pad>", "</s>", *sorted(bytes_)])}
    raw = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", eos_token="</s>", padding_side="left"
    )
    tok.chat_template = (
        "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}</s>{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )
    small = transformers.LlamaConfig(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    torch.manual_seed(0)
    base = transformers.AutoModelForCausalLM.from_config(small)
    for name in ("a", "b"):  # two different, non-zero adapters
        lora = peft.LoraConfig(r=4, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
        peft.get_peft_model(transformers.AutoModelForCausalLM.from_config(small), lora).save_pretrained(
            tmp_path / name
        )
    monkeypatch.setattr(hc.ts, "load_tokenizer", lambda model: tok)
    monkeypatch.setattr(hc.ts, "load_base", lambda c: base)

    rows = hc.read_eval_rows(cfg, hc.heldout_data(cfg), limit=2)
    sample = hc.make_sampler(cfg, samples=3, temperature=1.0, max_new=6, seed=0)
    first = sample(rows, "a", str(tmp_path / "a"))
    other = sample(rows, "b", str(tmp_path / "b"))
    again = sample(rows, "a", str(tmp_path / "a"))  # loaded already: switched back to, not reloaded

    assert [len(x) for x in first] == [3, 3] and all(isinstance(t, str) for x in first for t in x)
    assert again == first  # the same seed for every adapter: reruns repeat exactly
    assert other != first
    summary = hc.run(rows, {"a": str(tmp_path / "a")}, sample, grpo_rewards.DEFAULT_WEIGHTS, tmp_path / "out")
    assert summary["adapters"]["a"]["samples"] == 6
