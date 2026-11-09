"""Held-out reward check for single-turn GRPO (SEL-21): does the reward rise on turns not trained on?

The GRPO trainer only sees the training prompts, so its reward curve can rise by memorising
them. This samples the **held-out** prompts (``grpo-healable.heldout.jsonl``: the 15
``solved_by_aya`` conversations `just grpo-data` keeps apart) with each adapter, scores every
sample with the training reward (`reflexive.grpo_rewards`, the config's weights) and compares
the adapters on the same prompts. On the GPU pod:

    just grpo-heldout --limit 5                      # smoke: SFT v2 (+ GRPO if trained), 5 prompts
    just grpo-heldout                                # SFT v2 against runs/<name> (the GRPO adapter)
    just grpo-heldout --adapter sft-v2=reagent-lexsi/sft-aya32b-r16-v2 \\
        --adapter step50=runs/grpo-aya32b-v2/checkpoints/checkpoint-50   # any adapters, by name=path
    just grpo-heldout --data data/train/grpo-healable.jsonl --limit 40 --out runs/x/train-check
                                                     # the same on training prompts: the gap is memorising

The first adapter is the reference: every other one gets the paired difference against it, with
a 95% range from resampling the 15 **tasks** (turns of one conversation aren't independent), via
`reflexive.compare.paired_bootstrap_diff`. Before GRPO has run, SFT v2 alone still tells you how
often its samples all score the same (``all_same``): those groups give GRPO nothing to learn from.

Writes ``<out>/<adapter>.jsonl`` (every sample with each reward part: ``pd.read_json(path,
lines=True)``), ``<out>/summary.json`` and ``<out>/summary.md``. Default ``<out>``:
``runs/<name>/heldout``. Plain transformers ``generate``, no vLLM: roughly 10–20 minutes an adapter
for the 87 held-out prompts on an H100 (a guess).
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_single_turn as ts  # noqa: E402 - the trainer's config, data checks and model loading

from reflexive import grpo_rewards  # noqa: E402
from reflexive.compare import paired_bootstrap_diff  # noqa: E402
from reflexive.files import ROOT, sha256, shown  # noqa: E402

SAMPLES = 4
Sampler = Callable[[list[dict[str, Any]], str, str], list[list[str]]]  # rows, name, path → texts per row


def heldout_data(cfg: dict[str, Any]) -> str:
    """The held-out file `just grpo-data` writes beside the config's training file."""
    data = cfg["data"]
    return data[: -len(".jsonl")] + ".heldout.jsonl" if data.endswith(".jsonl") else data + ".heldout"


def read_eval_rows(cfg: dict[str, Any], data: str, limit: int | None = None) -> list[dict[str, Any]]:
    """The prompts to check: train-split tasks only, and (unless it *is* the training file) no
    task the trainer saw. ``limit`` keeps the first ``limit`` prompts."""
    rows = ts.read_rows(data)
    if data != cfg["data"]:
        trained = {(r["domain"], str(r["task_id"])) for r in ts.read_rows(cfg["data"])}
        if both := sorted({(r["domain"], str(r["task_id"])) for r in rows} & trained):
            raise SystemExit(
                f"⛔ {len(both)} task(s) in {data} were trained on ({cfg['data']}), "
                f"e.g. {both[0][0]}/{both[0][1]}: not a held-out check"
            )
    return rows[:limit] if limit else rows


def parse_adapters(specs: Sequence[str] | None, cfg: dict[str, Any]) -> dict[str, str]:
    """``name=path`` specs (a local folder or a Hub repo) in order. Default: SFT v2 (the config's
    ``adapter``), then the GRPO run's adapter if ``runs/<name>`` has one."""
    if not specs:
        out = {"sft-v2": cfg["adapter"]}
        run = ROOT / "runs" / cfg["name"]
        if (run / "adapter_config.json").exists():
            out["grpo"] = shown(run)
        return out
    out: dict[str, str] = {}
    for s in specs:
        name, sep, path = s.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"⛔ --adapter {s!r} isn't name=path")
        if name in out:
            raise SystemExit(f"⛔ adapter name {name!r} given twice")
        out[name] = path
    return out


def score_samples(
    rows: list[dict[str, Any]], texts: list[list[str]], weights: dict[str, float]
) -> list[dict[str, Any]]:
    """One record per sample: the prompt's id and task, each reward part, the total, the text."""
    if len(texts) != len(rows):
        raise ValueError(f"{len(texts)} sample lists for {len(rows)} prompts")
    out = []
    for r, samples in zip(rows, texts, strict=True):
        for k, text in enumerate(samples):
            s = grpo_rewards.score(text, r["target_calls"], r["domain"], r["prompt"], r["target"])
            out.append(
                {
                    "id": r["id"],
                    "domain": r["domain"],
                    "task_id": str(r["task_id"]),
                    "sample": k,
                    "reward": grpo_rewards.total(s, weights),
                    "exact": s["tool_match"] == 1.0 and s["args_match"] == 1.0,
                    **s,
                    "completion": text[:2000],
                }
            )
    return out


def _by_prompt(scored: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for s in scored:
        groups[s["id"]].append(s)
    return groups


def _by_task(scored: list[dict[str, Any]]) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """Each task's samples as (reward, 1) pairs: the bootstrap's unit is the task."""
    tasks: dict[tuple[str, str], list[tuple[float, float]]] = collections.defaultdict(list)
    for s in scored:
        tasks[(s["domain"], s["task_id"])].append((s["reward"], 1.0))
    return tasks


def _mean(xs: Sequence[float]) -> float:
    return round(statistics.fmean(xs), 4) if xs else 0.0


def summarise(scored: dict[str, list[dict[str, Any]]], max_reward: float) -> dict[str, Any]:
    """Per adapter: mean reward, each part's mean, exact-match share, the share of prompts whose
    samples all scored the same, the mean by domain; and each adapter against the first."""
    adapters: dict[str, Any] = {}
    for name, rows in scored.items():
        groups = _by_prompt(rows)
        domains = sorted({r["domain"] for r in rows})
        adapters[name] = {
            "prompts": len(groups),
            "tasks": len({(r["domain"], r["task_id"]) for r in rows}),
            "samples": len(rows),
            "reward": _mean([r["reward"] for r in rows]),
            "reward_share": round(_mean([r["reward"] for r in rows]) / max_reward, 4) if max_reward else None,
            "exact": _mean([float(r["exact"]) for r in rows]),
            "all_same": _mean([float(len({g["reward"] for g in grp}) == 1) for grp in groups.values()]),
            "parts": {k: _mean([r[k] for r in rows]) for k in grpo_rewards.COMPONENTS},
            "by_domain": {d: _mean([r["reward"] for r in rows if r["domain"] == d]) for d in domains},
        }
    names = list(scored)
    diffs = {}
    for name in names[1:]:
        ref = names[0]
        a, b = _by_task(scored[ref]), _by_task(scored[name])
        ci = paired_bootstrap_diff(a, b)
        diffs[name] = {
            "against": ref,
            "reward_diff": round(adapters[name]["reward"] - adapters[ref]["reward"], 4),
            "ci95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
            "tasks": len(a.keys() & b.keys()),
        }
    return {"max_reward": max_reward, "adapters": adapters, "differences": diffs}


def to_markdown(summary: dict[str, Any]) -> str:
    parts = list(grpo_rewards.COMPONENTS)
    head = ["adapter", "prompts", "reward", "exact", "all_same", *parts]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, a in summary["adapters"].items():
        cells = [name, str(a["prompts"]), f"{a['reward']:.3f} / {summary['max_reward']:g}"]
        cells += [f"{a['exact']:.0%}", f"{a['all_same']:.0%}", *(f"{a['parts'][k]:.2f}" for k in parts)]
        lines.append("| " + " | ".join(cells) + " |")
    for name, d in summary["differences"].items():
        ci = f"95% range {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}" if d["ci95"] else "no range"
        diff = f"{d['reward_diff']:+.3f} reward ({ci}, resampling {d['tasks']} tasks)"
        lines.append(f"\n**{name} − {d['against']}:** {diff}")
    return "\n".join(lines) + "\n"


def stop_ids(tok: Any) -> list[int]:
    """Where a reply ends: the tokenizer's end token and Aya's end-of-turn token, if it has one."""
    ids = [tok.eos_token_id] if tok.eos_token_id is not None else []
    eot = tok.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>")
    if isinstance(eot, int) and eot != tok.unk_token_id and eot not in ids:
        ids.append(eot)
    return ids


def make_sampler(cfg: dict[str, Any], samples: int, temperature: float, max_new: int, seed: int) -> Sampler:
    """Loads the model once, then each adapter on top in turn (GPU only)."""
    import torch
    from peft import PeftModel

    tok = ts.load_tokenizer(cfg["model"])
    base = ts.load_base(cfg)
    state: dict[str, Any] = {"model": None}
    stops = stop_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else stops[0]

    def sample(rows: list[dict[str, Any]], name: str, path: str) -> list[list[str]]:
        if state["model"] is None:
            state["model"] = PeftModel.from_pretrained(base, path, adapter_name=name)
        elif name not in state["model"].peft_config:  # already loaded: just switch to it
            state["model"].load_adapter(path, adapter_name=name)
        model = state["model"]
        model.set_adapter(name)
        model.eval()
        torch.manual_seed(seed)  # the same seed for every adapter
        out, start = [], time.time()
        for i, r in enumerate(rows, 1):
            text = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                gen = model.generate(
                    **enc,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=1.0,
                    num_return_sequences=samples,
                    max_new_tokens=max_new,
                    eos_token_id=stops,
                    pad_token_id=pad,
                )
            n = enc["input_ids"].shape[1]
            out.append([tok.decode(g[n:], skip_special_tokens=True) for g in gen])
            if i % 10 == 0 or i == len(rows):
                print(f"   {name}: {i}/{len(rows)} prompts, {(time.time() - start) / 60:.1f} min", flush=True)
        return out

    return sample


def run(
    rows: list[dict[str, Any]],
    adapters: dict[str, str],
    sampler: Sampler,
    weights: dict[str, float],
    out: Path,
) -> dict[str, Any]:
    """Sample and score every adapter, writing each one's samples as it finishes; then the summary."""
    out.mkdir(parents=True, exist_ok=True)
    scored = {}
    for name, path in adapters.items():
        print(f"{name} ({path}): sampling {len(rows)} prompts")
        scored[name] = score_samples(rows, sampler(rows, name, path), weights)
        (out / f"{name}.jsonl").write_text(
            "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in scored[name])
        )
    return summarise(scored, max_reward=sum(weights.values()))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path, nargs="?", default=ts.CONFIG, help=f"default: {shown(ts.CONFIG)}")
    p.add_argument("--adapter", action="append", help="name=path (repeat; the first is the reference)")
    p.add_argument("--data", help="the prompts to check (default: the config's held-out file)")
    p.add_argument("--limit", type=int, help="only the first N prompts (a smoke run: 5)")
    p.add_argument("--samples", type=int, default=SAMPLES, help=f"samples per prompt (default {SAMPLES})")
    p.add_argument("--temperature", type=float, help="default: the config's; 0 = greedy (use --samples 1)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, help="default: runs/<name>/heldout")
    p.add_argument("--dry-run", action="store_true", help="check the data and adapters only (no GPU)")
    a = p.parse_args()

    cfg = ts.load(a.config)
    data = a.data or heldout_data(cfg)
    rows = read_eval_rows(cfg, data, a.limit)
    adapters = parse_adapters(a.adapter, cfg)
    weights = {**grpo_rewards.DEFAULT_WEIGHTS, **(cfg.get("reward_weights") or {})}
    t = cfg["trainer"]
    temperature = a.temperature if a.temperature is not None else t.get("temperature", 1.0)
    out = a.out or ROOT / "runs" / cfg["name"] / "heldout"
    print(
        f"{len(rows)} prompts from {data} ({len({(r['domain'], r['task_id']) for r in rows})} tasks), "
        f"{a.samples} samples each at temperature {temperature}; adapters: "
        + ", ".join(f"{n}={p}" for n, p in adapters.items())
    )
    if len(adapters) == 1:
        print("   (one adapter: no comparison. Train first, or pass --adapter twice)")
    if a.dry_run:
        return 0

    sampler = make_sampler(cfg, a.samples, temperature, t.get("max_completion_length", 384), a.seed)
    summary = run(rows, adapters, sampler, weights, out)
    summary["inputs"] = {
        "config": shown(a.config),
        "data": {"file": data, "sha256": sha256(ROOT / data), "prompts": len(rows)},
        "adapters": adapters,
        "samples": a.samples,
        "temperature": temperature,
        "seed": a.seed,
        "weights": weights,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    md = to_markdown(summary)
    (out / "summary.md").write_text(md)
    print("\n" + md + f"\nsamples and summary in {shown(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
