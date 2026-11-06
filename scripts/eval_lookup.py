"""Evaluate a model on Messy-ID Lookup (docs/experiments/MESSY_LOOKUP.md).

    python scripts/eval_lookup.py                                   # base model
    python scripts/eval_lookup.py --adapter runs/messy-grpo         # trained adapter
    python scripts/eval_lookup.py --split train --limit 40          # a quick look
    python scripts/eval_lookup.py --record base --arm A             # = just baseline

The question mix, model and max_steps_per_turn come from the training config (``--config``,
default configs/train/messy_grpo.yaml), so the eval matches training.

Greedy decoding, the same rollout code and tool prompt as training (AgentTune's
``create_rollout_fn``), and the same scoring as the reward (``reflexive.messy_lookup.score``).
The eval split's 40 customers never appear in training.

Writes one CSV row per question (``--out``, default ``<adapter or runs/base-messy>/eval_<split>.csv``)
and prints success rates by level (question type). ``--record RUN`` then writes
``results/RUN.metrics.json`` and a docs/EXPERIMENTS.md row (scripts/experiment_log.py).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import yaml

from reflexive import messy_lookup


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("configs/train/messy_grpo.yaml"),
        help="training config: question mix, model, max_steps_per_turn",
    )
    ap.add_argument("--model", help="default: the config's model, else CohereLabs/tiny-aya-global")
    ap.add_argument("--adapter", type=pathlib.Path, help="LoRA adapter folder from train_grpo.py")
    ap.add_argument("--split", default="eval", choices=["eval", "train"])
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--max-tool-steps", type=int, help="default: the config's max_steps_per_turn, else 3")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 = greedy (default, what --record uses); e.g. 0.8 to sample like training",
    )
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--record", metavar="RUN", help="also record results/RUN.metrics.json + a log row")
    ap.add_argument("--arm", help="with --record: the experiment arm, e.g. A")
    a = ap.parse_args()

    if a.record and (a.split != "eval" or a.temperature):
        ap.error("--record only records the eval split with greedy decoding")
    if a.record and (pathlib.Path("results") / f"{a.record}.metrics.json").exists():
        ap.error(f"results/{a.record}.metrics.json exists; pick a new run name")

    import experiment_log  # scripts/experiment_log.py (this script's folder is on sys.path)
    import pandas as pd
    import torch
    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = yaml.safe_load(a.config.read_text())
    if "base" in cfg:
        raise SystemExit("--config: pass the config the 'base:' line points to")
    task = cfg.get("task")
    if task != "messy_lookup":
        raise SystemExit(f"--config: task is {task!r}, expected messy_lookup")
    mod = messy_lookup
    a.model = a.model or cfg.get("model", "CohereLabs/tiny-aya-global")
    max_tool_steps = a.max_tool_steps or cfg.get("trainer", {}).get("max_steps_per_turn", 3)
    rows = mod.questions(a.split, styles=cfg.get("task_options", {}).get("styles"))[: a.limit]
    print(f"task={task} model={a.model} adapter={a.adapter} questions={len(rows)}")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(a.adapter))
    model.eval()
    rollout = create_rollout_fn(
        model=model,
        tokenizer=tok,
        rollout_backend="transformers",
        tools=mod.TOOLS,
        max_steps=max_tool_steps,
        **mod.HARNESS,
    )  # the same tool wording as training

    sampling = {"do_sample": True, "temperature": a.temperature} if a.temperature else {"do_sample": False}
    records, t0 = [], time.time()
    for i, row in enumerate(rows, 1):
        out = rollout([mod.prompt(row)], max_new_tokens=a.max_new_tokens, **sampling)
        conv, text = out["conversations"][0], out["responses"][0] or ""
        s = mod.score(conv, text, row["answer"], row["field"], row["customer_id"])
        records.append({**row, **s, "final_text": text[:300], "conversation": json.dumps(conv, default=str)})
        if i % 10 == 0 or i == len(rows):
            done = pd.DataFrame(records)
            print(f"{i}/{len(rows)}  correct so far {done.correct.mean():.0%}  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(records)
    suffix = f"_t{a.temperature:g}" if a.temperature else ""
    dest = a.out or (a.adapter or pathlib.Path("runs/base-messy")) / f"eval_{a.split}{suffix}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)

    cols = ["correct", "tool_ok", "format", "n_tool_calls", "hallucinated_ids", "wrong_customer"]
    print("\nBy level / question type (means):")
    print(df.groupby("level")[cols].mean().round(2).to_string())
    print("\nOverall:")
    print(df[cols].mean().round(2).to_string())
    summary = {
        "model": a.model,
        "adapter": str(a.adapter) if a.adapter else None,
        "split": a.split,
        "commit": experiment_log._git("rev-parse", "--short", "HEAD"),
        "n": len(df),
        "seconds": round(time.time() - t0),
        **df[cols].mean().round(3).to_dict(),
    }
    dest.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {dest} and {dest.with_suffix('.json')}")
    if a.record:
        rec = argparse.Namespace(
            eval_csv=dest,
            run=a.record,
            arm=a.arm,
            config=str(a.config),
            adapter=str(a.adapter) if a.adapter else None,
            commit=None,
            owner=None,
            change=None,
            notes="",
        )
        return experiment_log.record(rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
