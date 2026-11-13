"""Single-turn GRPO from SFT v2 (SEL-21): one sampled agent turn, scored against the teacher's.

On the GPU pod, after ``scripts/pod/setup.sh`` (and ``just grpo-data``):

    just grpo-train --dry-run                        # tokenise and report (no GPU): check first
    just grpo-train --max-steps 2 --report-to none   # smoke run: loads, samples, steps, saves
    just grpo-train                                  # configs/train/grpo_aya32b_v2.yaml
    just grpo-train --push                           # also upload the adapter, private
    just grpo-train other.yaml                       # another config

Loads the config's ``model`` in 4-bit with SFT v2's LoRA ``adapter`` on it (trainable: GRPO
continues it), and trains with TRL's ``GRPOTrainer``: for each prompt it samples
``num_generations`` replies, scores each with `reflexive.grpo_rewards` and moves the adapter
towards the replies that score above their group's mean. Plain TRL, not AgentTune: a sample is
one reply, with no tool run and no customer to answer it.

Writes ``runs/<name>/``: the adapter, ``rewards.jsonl`` (every scored sample, with each reward
component: ``pd.read_json(path, lines=True)``), ``log.jsonl`` (TRL's metrics per step) and
``run_summary.json``. Refuses to overwrite a finished run without ``--force``. Trains on train
tasks only, and checks again.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import yaml

from reflexive import grpo_data, grpo_rewards, heal_rewards, splits
from reflexive.files import ROOT, read_jsonl, sha256, shown
from reflexive.sft import relax_chat_template

HUB_ORG = "reagent-lexsi"
CONFIG = ROOT / "configs" / "train" / "grpo_aya32b_v2.yaml"
REQUIRED = ("name", "model", "adapter", "data", "trainer")
COLUMNS = ("prompt", "target", "target_calls", "domain", "id")  # what the reward reads
HEAL_COLUMNS = (*COLUMNS, "failure_type", "failed", "history")  # `rewards: heal` (SEL-75) reads these too
JSON_COLUMNS = ("target_calls", "failed", "history")  # as JSON text: Arrow can't hold dicts of varying shape
REWARD_SETS = {"single_turn": grpo_rewards, "heal": heal_rewards}


def reward_set(cfg: dict[str, Any]) -> Any:
    """The reward module the config names (``rewards:``): the first run's single-turn reward by
    default, or ``heal``: one reward per failure type plus the teacher-fix agreement (SEL-75)."""
    return REWARD_SETS[cfg.get("rewards", "single_turn")]


def load(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    for key in REQUIRED:
        if key not in cfg:
            raise SystemExit(f"⛔ {shown(path)} has no {key!r}")
    if cfg.get("rewards", "single_turn") not in REWARD_SETS:
        raise SystemExit(f"⛔ unknown rewards {cfg['rewards']!r}: expected one of {sorted(REWARD_SETS)}")
    components = reward_set(cfg).COMPONENTS
    if bad := sorted(set(cfg.get("reward_weights") or {}) - set(components)):
        raise SystemExit(f"⛔ unknown reward component(s) {bad}: expected some of {components}")
    return cfg


def read_rows(data: str) -> list[dict[str, Any]]:
    """The GRPO prompts in ``data`` (a path under the repo), checked: our format, train tasks only."""
    path = ROOT / data
    if not path.exists():
        raise SystemExit(f"⛔ {data} doesn't exist: run `just grpo-data` first")
    rows = read_jsonl(path)
    for r in rows:
        if r.get("format") != grpo_data.FORMAT:
            raise SystemExit(f"⛔ {r.get('id')} isn't {grpo_data.FORMAT}: rebuild with `just grpo-data`")
        splits.assert_not_test(r["domain"], r["task_id"])  # a test task is never trained on
        if splits.split_of(r["domain"], r["task_id"]) != "train":
            raise SystemExit(f"⛔ {r['id']} isn't a train task: rebuild with `just grpo-data`")
    if not rows:
        raise SystemExit(f"⛔ {data} is empty")
    return rows


def prompts(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """The training rows, as the dataset's columns. ``target_calls`` as JSON text: Arrow can't
    hold argument dicts of different shapes in one column."""
    rows = read_rows(cfg["data"])
    columns = HEAL_COLUMNS if cfg.get("rewards") == "heal" else COLUMNS
    if missing := sorted({c for r in rows for c in columns if c not in r}):
        raise SystemExit(f"⛔ {cfg['data']} has no {missing}: build it with `just grpo-heal-data`")
    return [{k: json.dumps(r[k]) if k in JSON_COLUMNS else r[k] for k in columns} for r in rows]


def grpo_args(cfg: dict[str, Any], out: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """GRPOConfig's keywords: the config's ``trainer`` section (and any overrides) over the fixed ones.

    Checks the batch arithmetic here, so a bad config stops before the 32B model loads: the
    samples generated together (``per_device_train_batch_size`` x ``steps_per_generation``,
    which defaults to ``gradient_accumulation_steps``) and the samples in one training step must
    both hold whole groups."""
    t = {**cfg["trainer"], **(overrides or {})}
    batch = t.get("per_device_train_batch_size", 8)
    accumulate = t.get("gradient_accumulation_steps", 1)
    groups = t.get("num_generations", 8)
    for what, n in (
        (
            "generation batch (per_device_train_batch_size x steps_per_generation)",
            batch * (t.get("steps_per_generation") or accumulate),
        ),
        ("training step (per_device_train_batch_size x gradient_accumulation_steps)", batch * accumulate),
    ):
        if n % groups:
            raise SystemExit(
                f"⛔ {n} samples per {what} don't divide into groups of {groups}: "
                "make it a multiple of num_generations"
            )
    return {
        "output_dir": str(out / "checkpoints"),
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "remove_unused_columns": False,  # the reward reads target, target_calls, domain, id
        "report_to": ["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        "run_name": cfg["name"],
        **t,
    }


def prompt_tokens(rows: list[dict[str, Any]], tok: Any) -> list[int]:
    """Each prompt's length in tokens, rendered as the trainer renders it (with the turn header).

    Renders to text, then tokenises: with ``tokenize=True``, transformers 5 returns a dict (its
    keys, not its tokens, are what ``len`` counts), as SFT found."""
    out = []
    for r in rows:
        text = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
        out.append(len(tok(text, add_special_tokens=False)["input_ids"]))
    return out


def cap_prompts(
    rows: list[dict[str, Any]], lengths: list[int], cap: int | None
) -> tuple[list[dict[str, Any]], list[int]]:
    """Drops the prompts longer than ``cap`` tokens (the config's ``max_prompt_tokens``): GPU memory
    during generation grows with the prompt's length. ``None`` keeps them all."""
    if not cap:
        return rows, lengths
    kept = [(r, n) for r, n in zip(rows, lengths, strict=True) if n <= cap]
    if not kept:
        raise SystemExit(f"⛔ no prompt is {cap:,} tokens or shorter: raise max_prompt_tokens")
    return [r for r, _ in kept], [n for _, n in kept]


def length_report(lengths: list[int]) -> str:
    """Quantiles and how many prompts are over a few lengths: to choose ``max_prompt_tokens``."""
    xs = sorted(lengths)
    q = {p: xs[min(int(p / 100 * len(xs)), len(xs) - 1)] for p in (50, 90, 99)}
    over = ", ".join(f">{k // 1000}K: {sum(n > k for n in xs)}" for k in (6000, 8000, 9000, 10000, 12000))
    return f"tokens: median {q[50]:,}, p90 {q[90]:,}, p99 {q[99]:,}, longest {xs[-1]:,}; over {over}"


def load_tokenizer(model: str) -> Any:
    """Aya's tokenizer, padding on the left, with the chat template that lets the agent speak first."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, padding_side="left", truncation_side="left")
    tok.chat_template = relax_chat_template(tok.chat_template)  # see reflexive.sft
    return tok


def load_base(cfg: dict[str, Any]) -> Any:
    """The config's model in 4-bit on GPU 0, as SFT trained the adapter on it."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    q = cfg.get("quantization", {})
    return AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=q.get("load_in_4bit", True),
            bnb_4bit_quant_type=q.get("quant_type", "nf4"),
            bnb_4bit_use_double_quant=q.get("double_quant", True),
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": 0},
    )


def git_commit() -> str | None:
    """The repo's commit, with ``+dirty`` if files have changed since: the code this run used."""
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
        ).stdout

    try:
        return git("rev-parse", "--short", "HEAD").strip() + (
            "+dirty" if git("status", "--porcelain", "-uno") else ""
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def warm_cublas() -> None:
    """Create cuBLAS's handles now, on this thread and on autograd's backward thread.

    cuBLAS allocates its handle outside PyTorch, the first time a thread multiplies on the GPU.
    For the backward thread that's the first backward pass, after generation has filled PyTorch's
    memory cache: the first trial failed there with ``CUBLAS_STATUS_ALLOC_FAILED``."""
    import torch

    x = torch.ones(16, 16, device="cuda", requires_grad=True)
    (x @ x).sum().backward()
    torch.cuda.synchronize()


def gb(n: int) -> float:
    return round(n / 2**30, 1)


def releasing_trainer(base: type) -> type:
    """TRL's trainer class, giving generation's cached GPU memory back before the backward pass.

    Prints the first round's peak memory: what generating one group of samples takes."""
    import torch

    class Trainer(base):
        rounds = 0

        def _generate_and_score_completions(self, *args: Any, **kwargs: Any) -> Any:
            out = super()._generate_and_score_completions(*args, **kwargs)
            self.rounds += 1
            if torch.cuda.is_available():
                used, cached = torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved()
                torch.cuda.empty_cache()
                if self.rounds == 1:
                    print(
                        f"first generation round: peak {gb(used)} GB, {gb(cached)} GB cached (now released)"
                    )
            return out

    return Trainer


def prepare_for_training(base: Any) -> None:
    """Freeze the base and enable gradient checkpointing, keeping its activations in bfloat16.

    Not ``peft.prepare_model_for_kbit_training``: it casts the norms and embeddings to float32, so
    every activation, the KV cache and the generation buffers are float32 too. In the 3-step trial
    that peaked at 77.9 of the H100's 79 GB on a median-length prompt, and every telecom prompt is
    longer. bfloat16 halves that memory. (SFT v2 trained with the float32 norms; within GRPO, the
    policy and the reference adapter both run in bfloat16.)"""
    for p in base.parameters():
        p.requires_grad_(False)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()


class ReplyDrift:
    """The heal run's drift guard (SEL-75, Chiara's review): stop training if the model starts
    replying much more, or much less, than the teacher did on the same prompts.

    Watches ``heal/reply_excess`` (the samples' share of replies minus the teacher's, per step,
    `reflexive.heal_rewards`). The first ``window`` logged steps are the baseline: SFT v2's own
    habit. From ``after_step`` on, if the mean of the last ``window`` steps is more than
    ``max_drift`` away from the baseline, either way, training stops: G drifted to replying
    almost never; a reward that rewarded generic questions would drift the other way."""

    def __init__(self, max_drift: float = 0.10, window: int = 10, after_step: int = 20) -> None:
        self.max_drift, self.window, self.after_step = max_drift, window, after_step
        self.values: list[float] = []
        self.baseline: float | None = None
        self.stopped: str | None = None

    def update(self, step: int, logs: dict[str, Any]) -> str | None:
        """Feed one step's logs; returns why to stop, or None."""
        if "heal/reply_excess" not in logs or self.stopped:
            return self.stopped
        self.values.append(float(logs["heal/reply_excess"]))
        if self.baseline is None:
            if len(self.values) >= self.window:
                self.baseline = statistics.fmean(self.values[: self.window])
            return None
        if step < self.after_step:
            return None
        recent = statistics.fmean(self.values[-self.window :])
        if abs(recent - self.baseline) > self.max_drift:
            self.stopped = (
                f"step {step}: reply share minus the teacher's averaged {recent:+.2f} over the last "
                f"{self.window} steps, against {self.baseline:+.2f} at the start "
                f"(limit ±{self.max_drift:g} more)"
            )
        return self.stopped


def no_gqa_in_sdpa() -> None:
    """Make transformers expand Aya's key/value heads itself instead of asking SDPA for GQA.

    With no padding mask (every sequence in a generation round is the same prompt), transformers
    passes ``enable_gqa=True`` to SDPA. On float32 activations (as ``prepare_model_for_kbit_training``
    gave), only SDPA's math kernel supports that, and it builds the full attention matrix: 8 x 64
    heads x ~8.5K x ~8.5K floats asked for 137 GiB on the H100. Expanded heads take the
    memory-efficient kernel instead, as they did when padding made a mask (~5 GB per layer)."""
    from transformers.integrations import sdpa_attention

    sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False


def main() -> int:  # noqa: PLR0915 - one linear script: check, load, train, save
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path, nargs="?", default=CONFIG, help=f"default: {shown(CONFIG)}")
    p.add_argument("--dry-run", action="store_true", help="tokenise and report only (no GPU)")
    p.add_argument("--max-steps", type=int, help="override trainer.max_steps (a smoke run: 2)")
    p.add_argument("--report-to", help="override where metrics go (none, wandb)")
    p.add_argument("--force", action="store_true", help="replace a finished run")
    p.add_argument("--push", action="store_true", help=f"upload the adapter to {HUB_ORG}/<name>, private")
    a = p.parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation
    os.environ.setdefault("WANDB_PROJECT", "reagent-trainer")  # all our runs in one W&B project

    cfg = load(a.config)
    out = ROOT / "runs" / cfg["name"]
    if (out / "run_summary.json").exists() and not a.force and not a.dry_run:
        print(f"⛔ {shown(out)} already has a finished run: pass --force to train again")
        return 1
    overrides: dict[str, Any] = {}
    if a.max_steps:
        overrides["max_steps"] = a.max_steps
    if a.report_to:
        overrides["report_to"] = [] if a.report_to == "none" else [a.report_to]
    args = grpo_args(cfg, out, overrides)
    rows = prompts(cfg)

    tok = load_tokenizer(cfg["model"])
    lengths = prompt_tokens(rows, tok)
    print(f"{len(rows)} prompts; {length_report(lengths)}")
    rows, lengths = cap_prompts(rows, lengths, cfg.get("max_prompt_tokens"))
    groups = args.get("num_generations", 8)
    if (steps := args.get("max_steps") or 0) > 0:
        batch = args.get("per_device_train_batch_size", 8) * args.get("gradient_accumulation_steps", 1)
        n_prompts = steps * batch
        seen = f"{steps} steps see {n_prompts // groups} of them"
    else:
        seen = f"{args.get('num_train_epochs', 3)} epoch(s)"
    print(
        f"training on {len(rows)} prompts (max_prompt_tokens: {cfg.get('max_prompt_tokens') or 'none'}); "
        f"{seen}, {groups} samples each"
    )
    if a.dry_run:
        return 0

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    base = load_base(cfg)
    prepare_for_training(base)  # bfloat16 throughout: the float32 upcast nearly filled the H100
    no_gqa_in_sdpa()
    model = PeftModel.from_pretrained(base, cfg["adapter"], is_trainable=True)
    model.print_trainable_parameters()

    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("w")

    guard = ReplyDrift(**cfg["drift_guard"]) if cfg.get("drift_guard") else None

    class Log(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw) -> None:  # noqa: ANN001, ANN003
            if logs:
                log.write(json.dumps({"step": state.global_step, **logs}) + "\n")
                log.flush()
                if (
                    guard
                    and (why := guard.update(state.global_step, logs))
                    and not control.should_training_stop
                ):
                    print(f"⛔ drift guard: {why}; stopping and saving the adapter")
                    control.should_training_stop = True

    Trainer = releasing_trainer(GRPOTrainer)
    weights = {**reward_set(cfg).DEFAULT_WEIGHTS, **(cfg.get("reward_weights") or {})}
    if cfg.get("rewards") == "heal":  # one TRL reward function per component, weighted by TRL
        reward_funcs = heal_rewards.RewardFns(out / "rewards.jsonl", weights).fns
        args["reward_weights"] = [weights[c] for c in heal_rewards.COMPONENTS]
    else:
        reward_funcs = [grpo_rewards.make_reward_fn(weights, log_path=out / "rewards.jsonl")]
    trainer = Trainer(
        model=model,
        reward_funcs=reward_funcs,
        args=GRPOConfig(**args),
        train_dataset=Dataset.from_list(rows),
        processing_class=tok,
        callbacks=[Log()],
    )
    warm_cublas()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    result = trainer.train()
    minutes = (time.time() - start) / 60
    model.save_pretrained(out)
    tok.save_pretrained(out)
    log.close()

    summary = {
        "name": cfg["name"],
        "commit": git_commit(),
        "model": cfg["model"],
        "adapter": cfg["adapter"],
        "config": shown(a.config),
        "data": {"file": cfg["data"], "sha256": sha256(ROOT / cfg["data"]), "prompts": len(rows)},
        "prompt_tokens": {
            "median": statistics.median(lengths),
            "max": max(lengths),
            "cap": cfg.get("max_prompt_tokens"),
        },
        "rewards": cfg.get("rewards", "single_turn"),
        "reward_weights": weights,
        "trainer": {k: v for k, v in args.items() if isinstance(v, str | int | float | bool)},
        "train_loss": result.training_loss,
        "stopped_by_drift_guard": guard.stopped if guard else None,
        "steps": result.global_step,
        "minutes": round(minutes, 1),
        "minutes_per_step": round(minutes / max(result.global_step, 1), 2),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "gpu": torch.cuda.get_device_name(0),
    }
    if a.push:
        repo = f"{HUB_ORG}/{cfg['name']}"
        model.push_to_hub(repo, private=True)
        summary["hub"] = repo
    (out / "run_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(
        f"done in {minutes:.0f} min ({summary['minutes_per_step']} min/step); "
        f"peak {summary['peak_gpu_gb']} GB; "
        f"adapter in {shown(out)}; per-sample rewards in {shown(out / 'rewards.jsonl')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
