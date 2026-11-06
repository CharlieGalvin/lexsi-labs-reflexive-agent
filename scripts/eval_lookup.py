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
and prints success rates by level (question type). A run with any override (``--limit``,
``--temperature``, ``--model``, ``--max-tool-steps``, ``--max-new-tokens``) gets its own file
name, e.g. ``eval_eval_n10.csv``, so it never replaces the full eval. The settings are saved
next to the CSV (``.json``). ``--record RUN`` then writes ``results/RUN.metrics.json`` and a
docs/EXPERIMENTS.md row (scripts/experiment_log.py); it refuses, before loading the model,
anything but the official eval (``experiment_log.official_problems``).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import experiment_log  # scripts/experiment_log.py (this script's folder is on sys.path); stdlib only
from train_grpo import load_config  # the same config reading as training, `base:` included

from reflexive import messy_lookup


def suffix(s: dict) -> str:
    """The CSV name's tag for every setting that differs from the official eval ("" for it)."""
    tags = []
    if s["temperature"]:
        tags.append(f"t{s['temperature']:g}")
    if s["limit"] is not None:
        tags.append(f"n{s['limit']}")
    if s["max_tool_steps"] != s["config_max_tool_steps"]:
        tags.append(f"steps{s['max_tool_steps']}")
    if s["max_new_tokens"] != experiment_log.MAX_NEW_TOKENS:
        tags.append(f"tok{s['max_new_tokens']}")
    if s["model"] != s["config_model"]:
        tags.append("model-" + "".join(c if c.isalnum() or c in "-." else "-" for c in s["model"]))
    return "".join(f"_{t}" for t in tags)


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
    ap.add_argument("--max-new-tokens", type=int, default=experiment_log.MAX_NEW_TOKENS)
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

    cfg = load_config(a.config)
    task = cfg.get("task")
    if task != "messy_lookup":
        raise SystemExit(f"--config: task is {task!r}, expected messy_lookup")
    mod = messy_lookup
    config_model = cfg.get("model", "CohereLabs/tiny-aya-global")
    config_steps = cfg.get("trainer", {}).get("max_steps_per_turn", 3)
    a.model = a.model or config_model
    max_tool_steps = a.max_tool_steps or config_steps
    split_rows = mod.questions(a.split, styles=cfg.get("task_options", {}).get("styles"))
    rows = split_rows[: a.limit]
    settings = {
        "model": a.model,
        "config_model": config_model,
        "adapter": str(a.adapter) if a.adapter else None,
        "config": str(a.config),
        "split": a.split,
        "temperature": a.temperature,
        "limit": a.limit,
        "n": len(rows),
        "questions_in_split": len(split_rows),
        "max_tool_steps": max_tool_steps,
        "config_max_tool_steps": config_steps,
        "max_new_tokens": a.max_new_tokens,
    }
    # Refuse before loading the model: a wrong --record would only fail after the whole eval
    if a.record:
        out = experiment_log.RESULTS / f"{a.record}.metrics.json"
        if out.exists():
            ap.error(f"{out.relative_to(experiment_log.ROOT)} exists; pick a new run name")
        if problems := experiment_log.official_problems(settings):
            ap.error("--record only records the official eval: " + "; ".join(problems))
    dest = a.out or (a.adapter or pathlib.Path("runs/base-messy")) / f"eval_{a.split}{suffix(settings)}.csv"

    import pandas as pd
    import torch
    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"task={task} model={a.model} adapter={a.adapter} questions={len(rows)} -> {dest}")
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
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)

    cols = ["correct", "tool_ok", "format", "n_tool_calls", "hallucinated_ids", "wrong_customer"]
    print("\nBy level / question type (means):")
    print(df.groupby("level")[cols].mean().round(2).to_string())
    print("\nOverall:")
    print(df[cols].mean().round(2).to_string())
    summary = {
        **settings,
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
