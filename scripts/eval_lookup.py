"""Evaluate a model on Customer Lookup questions (runs 2 and 4 of docs/experiments/GRPO_WARMUP.md).

    python scripts/eval_lookup.py                                      # run 2: base Tiny Aya
    python scripts/eval_lookup.py --adapter runs/lookup-grpo           # run 4: trained adapter
    python scripts/eval_lookup.py --split train --limit 40 --l2-share 0.3   # tune the task mix

Greedy decoding, the same rollout code and tool prompt as training (AgentTune's
``create_rollout_fn``), and the same scoring as the reward (``reflexive.toy_lookup.score``).
The eval split's 40 customers never appear in training.

Writes one CSV row per question (``--out``, default ``<adapter or runs/base>/eval_<split>.csv``)
and prints success rates by level.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from reflexive import toy_lookup


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--adapter", type=pathlib.Path, help="LoRA adapter folder from train_grpo.py")
    ap.add_argument("--split", default="eval", choices=["eval", "train"])
    ap.add_argument("--l2-share", type=float, default=0.5, help="must match the training config")
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--max-tool-steps", type=int, default=3, help="same as max_steps_per_turn")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    import pandas as pd
    import torch
    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = toy_lookup.questions(a.split, l2_share=a.l2_share)[: a.limit]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(a.adapter))
    model.eval()
    rollout = create_rollout_fn(model=model, tokenizer=tok, rollout_backend="transformers",
                                tools=toy_lookup.TOOLS, max_steps=a.max_tool_steps,
                                **toy_lookup.HARNESS)  # the same tool wording as training

    records, t0 = [], time.time()
    for i, row in enumerate(rows, 1):
        out = rollout([toy_lookup.prompt(row)], do_sample=False, max_new_tokens=a.max_new_tokens)
        conv, text = out["conversations"][0], out["responses"][0] or ""
        s = toy_lookup.score(conv, text, row["answer"], row["field"], row["customer_id"])
        records.append({**row, **s, "final_text": text[:300],
                        "conversation": json.dumps(conv, default=str)})
        if i % 10 == 0 or i == len(rows):
            done = pd.DataFrame(records)
            print(f"{i}/{len(rows)}  correct so far {done.correct.mean():.0%}  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(records)
    dest = a.out or (a.adapter or pathlib.Path("runs/base")) / f"eval_{a.split}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)

    cols = ["correct", "tool_ok", "format", "n_tool_calls", "hallucinated_ids", "wrong_customer"]
    print("\nBy level (means):")
    print(df.groupby("level")[cols].mean().round(2).to_string())
    print("\nOverall:")
    print(df[cols].mean().round(2).to_string())
    summary = {"model": a.model, "adapter": str(a.adapter) if a.adapter else None, "split": a.split,
               "n": len(df), "seconds": round(time.time() - t0), **df[cols].mean().round(3).to_dict()}
    dest.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {dest} and {dest.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
