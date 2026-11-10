"""SEL-21: the single-turn GRPO trainer's checks, and one real TRL step on a tiny model.

The checks (config, data, batch arithmetic) need no GPU. The last test runs TRL's GRPOTrainer
for real: a tiny random Llama with a LoRA adapter on it (as SFT v2's sits on Aya), our reward,
conversational prompts. Skipped without torch, transformers, peft and trl (the GPU pod has them).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _scripts import load_script

from reflexive import grpo_data, grpo_rewards
from reflexive.files import ROOT

ts = load_script("grpo/train_single_turn")

FIND = {
    "name": "find_user_id_by_name_zip",
    "arguments": {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"},
}
PROMPT = [
    {"role": "system", "content": "policy and tools"},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "I'm Yusuf Rossi, zip 19122."},
]


def _row(task_id: str = "2", **kw: Any) -> dict[str, Any]:
    return {
        "id": f"x/retail/{task_id}/1#t3",
        "format": grpo_data.FORMAT,
        "domain": "retail",
        "task_id": task_id,
        "group": "healable",
        "turn": 3,
        "kind": "call",
        "prompt": PROMPT,
        "target": 'Action: ```json\n[{"tool_name": "find_user_id_by_name_zip", "parameters": '
        '{"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}}]\n```',
        "target_calls": [FIND],
        "source": {},
        **kw,
    }


def _data(tmp_path: Path, *rows: dict[str, Any]) -> dict[str, Any]:
    path = tmp_path / "grpo.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {**ts.load(ts.CONFIG), "data": str(path)}


def test_the_config_loads_and_its_batches_hold_whole_groups(tmp_path: Path):
    cfg = ts.load(ts.CONFIG)
    args = ts.grpo_args(cfg, tmp_path)
    assert args["remove_unused_columns"] is False and args["num_generations"] == 8
    assert (args["per_device_train_batch_size"] * args["gradient_accumulation_steps"]) % 8 == 0
    # one group per generation round: 16 x ~8K-token prompts at once ran the H100 out of memory
    assert args["per_device_train_batch_size"] * args["steps_per_generation"] == args["num_generations"]


def test_batches_that_split_a_group_are_refused_before_loading(tmp_path: Path):
    with pytest.raises(SystemExit, match="multiple of num_generations"):
        ts.grpo_args(ts.load(ts.CONFIG), tmp_path, {"gradient_accumulation_steps": 12})


def test_unknown_reward_components_are_refused(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text("name: x\nmodel: m\nadapter: a\ndata: d\ntrainer: {}\nreward_weights: {correct: 1}\n")
    with pytest.raises(SystemExit, match="unknown reward component"):
        ts.load(path)


def test_rows_become_the_columns_the_reward_reads(tmp_path: Path):
    (r,) = ts.prompts(_data(tmp_path, _row()))
    assert set(r) == set(ts.COLUMNS) and json.loads(r["target_calls"]) == [FIND]


@pytest.mark.parametrize(
    ("row", "error"),
    [(_row("1"), "isn't a train task"), (_row(format="reagent.sft/v1"), "rebuild")],
)
def test_a_dev_task_or_another_format_is_refused(tmp_path: Path, row: dict[str, Any], error: str):
    with pytest.raises(SystemExit, match=error):
        ts.prompts(_data(tmp_path, row))


def test_a_missing_data_file_says_how_to_build_it():
    with pytest.raises(SystemExit, match="just grpo-data"):
        ts.prompts({**ts.load(ts.CONFIG), "data": str(ROOT / "nowhere.jsonl")})


def test_the_run_records_its_commit():
    commit = ts.git_commit()
    assert commit and len(commit.split("+")[0]) >= 7


def test_sdpa_is_never_asked_for_gqa_so_float32_attention_stays_memory_efficient():
    sdpa = pytest.importorskip("transformers.integrations.sdpa_attention")
    torch = pytest.importorskip("torch")
    key = torch.zeros(1, 8, 4, 128)
    original = sdpa.use_gqa_in_sdpa
    try:
        ts.no_gqa_in_sdpa()
        assert sdpa.use_gqa_in_sdpa(None, key, key) is False  # no mask: transformers would have said True
    finally:
        sdpa.use_gqa_in_sdpa = original


def test_one_real_grpo_step_on_a_tiny_model(tmp_path: Path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    trl = pytest.importorskip("trl")
    from datasets import Dataset
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
    torch.manual_seed(0)
    small = transformers.LlamaConfig(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    base = transformers.AutoModelForCausalLM.from_config(small)
    lora = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    peft.get_peft_model(base, lora).save_pretrained(tmp_path / "sft")  # stands in for SFT v2's adapter
    model = peft.PeftModel.from_pretrained(
        transformers.AutoModelForCausalLM.from_config(small), tmp_path / "sft", is_trainable=True
    )

    cfg = _data(tmp_path, _row(), _row())
    rows = ts.prompts(cfg)
    args = ts.grpo_args(
        cfg,
        tmp_path / "run",
        {
            "max_steps": 1,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "num_generations": 4,
            "max_completion_length": 8,
            "optim": "adamw_torch",
            "report_to": [],
            "bf16": False,
            "save_strategy": "no",
            "use_cpu": True,
        },
    )
    seen: dict[str, list[float]] = {}
    reward = grpo_rewards.make_reward_fn(log_path=tmp_path / "rewards.jsonl")

    def spy(**kw: Any) -> list[float]:
        log_metric = kw.pop("log_metric")
        return reward(**kw, log_metric=lambda k, v: (seen.setdefault(k, []).append(v), log_metric(k, v)))

    spy.__name__ = "single_turn_reward"
    trainer = trl.GRPOTrainer(
        model=model,
        reward_funcs=[spy],
        args=trl.GRPOConfig(**args),
        train_dataset=Dataset.from_list(rows),
        processing_class=tok,
    )
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    trainer.train()

    logged = [json.loads(line) for line in (tmp_path / "rewards.jsonl").read_text().splitlines()]
    assert len(logged) == 4 and all(r["id"] == rows[0]["id"] for r in logged)  # one group of 4 samples
    assert set(seen) == {f"reward_parts/{k}" for k in grpo_rewards.COMPONENTS}
    assert trainable and all("lora" in n for n in trainable)  # the SFT adapter is what trains
