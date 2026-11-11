"""A tiny tokenizer and a tiny Llama with a LoRA adapter, for real TRL steps on CPU. No download.

Used by the GRPO trainer tests (SEL-21, SEL-75); they skip without torch, transformers and peft.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def tokenizer() -> Any:
    """A byte-level tokenizer built here, with a minimal chat template."""
    import transformers
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    bytes_ = pre_tokenizers.ByteLevel.alphabet()
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
    return tok


def adapter_model(tmp_path: Path, tok: Any, prepare: Any) -> Any:
    """A bfloat16 tiny Llama, prepared as the trainer prepares Aya (``prepare``), with a trainable
    LoRA adapter loaded from disk: it stands in for SFT v2's."""
    import peft
    import torch
    import transformers

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
    peft.get_peft_model(base, lora).save_pretrained(tmp_path / "sft")
    tiny = transformers.AutoModelForCausalLM.from_config(small).to(torch.bfloat16)
    prepare(tiny)
    assert all(p.dtype == torch.bfloat16 and not p.requires_grad for p in tiny.parameters())
    return peft.PeftModel.from_pretrained(tiny, tmp_path / "sft", is_trainable=True)
