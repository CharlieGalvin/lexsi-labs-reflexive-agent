"""The adapter lifecycle (SEL-22): which adapters exist, what each is, and which one is Arm B.

* ``configs/adapters.yaml``: the registry, one entry per adapter, written by hand.
* ``results/adapters/state.json``: which adapter is current, the Hub revision each was published
  at, and every switch (``history``). Written by ``scripts/adapters/*.py``, committed.

Pure functions here; the scripts do the I/O with the Hub and the vLLM server.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from reflexive.files import ROOT

REGISTRY = ROOT / "configs" / "adapters.yaml"
STATE = ROOT / "results" / "adapters" / "state.json"
STATUSES = {"current", "candidate", "superseded", "not_adopted", "control"}
PROVENANCE_FORMAT = "reagent.provenance/v1"


def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def adapters_by_id(reg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {a["id"]: a for a in reg["adapters"]}


def check_registry(reg: dict[str, Any], root: Path = ROOT) -> list[str]:
    """What's wrong with the registry (empty if nothing)."""
    problems = []
    ids = [a["id"] for a in reg["adapters"]]
    if len(ids) != len(set(ids)):
        problems.append("an id appears twice")
    repos = [a["repo"] for a in reg["adapters"]]
    if len(repos) != len(set(repos)):
        problems.append("a Hub repo appears twice")
    for a in reg["adapters"]:
        if a["status"] not in STATUSES:
            problems.append(f"{a['id']}: status {a['status']!r} isn't one of {sorted(STATUSES)}")
        if a.get("parent") and a["parent"] not in ids:
            problems.append(f"{a['id']}: parent {a['parent']!r} isn't in the registry")
        for key in ("train_config", "arm_config"):
            if a.get(key) and not (root / a[key]).exists():
                problems.append(f"{a['id']}: {key} {a[key]} doesn't exist")
    if sum(a["status"] == "current" for a in reg["adapters"]) != 1:
        problems.append("exactly one adapter must have status current (Arm B)")
    return problems


def hub_repo(reg: dict[str, Any], adapter: dict[str, Any]) -> str:
    return f"{reg['hub_org']}/{adapter['repo']}"


# --- state: the current adapter, published revisions, the switch history ----------------------


def load_state(path: Path = STATE) -> dict[str, Any]:
    if not path.exists():
        return {"current": None, "revisions": {}, "history": []}
    return json.loads(path.read_text())


def save_state(state: dict[str, Any], path: Path = STATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=1, ensure_ascii=False) + "\n")


def switch(state: dict[str, Any], to: str, why: str, by: str = "", now: str | None = None) -> dict[str, Any]:
    """A new state with ``to`` current and the switch recorded (the old state is unchanged)."""
    entry = {
        "at": now or dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "from": state.get("current"),
        "to": to,
        "revision": state.get("revisions", {}).get(to),
        "why": why,
        "by": by,
    }
    return {**state, "current": to, "history": [*state.get("history", []), entry]}


def previous(state: dict[str, Any]) -> str | None:
    """The adapter that was current before the latest switch (what a rollback returns to)."""
    history = state.get("history", [])
    return history[-1]["from"] if history else None


# --- what gets published with an adapter ------------------------------------------------------


def git_commit(root: Path = ROOT) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def provenance(
    reg: dict[str, Any], adapter: dict[str, Any], summary: dict[str, Any] | None, commit: str | None
) -> dict[str, Any]:
    """``lexsi_provenance.json``: where the adapter came from, enough to rebuild or audit it.

    ``summary`` is train_sft.py's ``run_summary.json`` when it exists (data checksum, settings,
    losses, steps, GPU); without it the card says so instead of guessing."""
    return {
        "format": PROVENANCE_FORMAT,
        "id": adapter["id"],
        "hub_repo": hub_repo(reg, adapter),
        "base_model": reg["base"],
        "parent": adapter.get("parent"),
        "method": adapter["method"],
        "data": adapter["data"],
        "train_config": adapter["train_config"],
        "arm_config": adapter["arm_config"],
        "ticket": adapter["ticket"],
        "status": adapter["status"],
        "code_commit": commit,
        "training": summary,  # None when the run's summary wasn't kept
        "eval": adapter.get("eval", {}),
        "metrics_files": adapter.get("metrics", []),
        "written": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }


def _eval_lines(ev: dict[str, Any]) -> list[str]:
    lines = []
    if "val" in ev:
        v = ev["val"]
        samples = ", ".join(map(str, v["samples"]))
        lines.append(f"| Validation (30 unseen train tasks) | {v['mean']} / 30 | samples {samples} |")
    if "dev" in ev:
        lines.append(
            f"| Dev ({ev['dev']['of']} tasks) | {ev['dev']['mean']} / {ev['dev']['of']} | mean of 2 |"
        )
    if "test" in ev:
        t = ev["test"]
        lines.append(
            f"| Test (100 tasks) | {t['mean']} / 100 | samples {', '.join(map(str, t.get('samples', [])))} |"
        )
    return lines


STATUS_WORDS = {
    "current": "**Arm B**: the adapter served as the project's healed student.",
    "candidate": "A candidate, not yet decided.",
    "superseded": "Superseded: it was the current adapter and has been replaced.",
    "not_adopted": "Not adopted: its pre-registered rule didn't hold. Kept for the record.",
    "control": "A control (an ablation), never a candidate.",
}


def model_card(reg: dict[str, Any], adapter: dict[str, Any], prov: dict[str, Any]) -> str:
    """README.md for the adapter's Hub repo."""
    s = prov.get("training") or {}
    tr = s.get("trainer", {})
    lora = s.get("lora", {})
    loss = s.get("held_out_loss", {})
    training = (
        [
            f"- Steps: {s.get('steps')}, learning rate {tr.get('learning_rate')}, "
            f"LoRA r{lora.get('r')} / alpha {lora.get('lora_alpha')}",
            f"- Train loss {s.get('train_loss'):.3f}; "
            f"held-out loss {loss.get('before')} → {loss.get('after')}"
            if s.get("train_loss") is not None
            else "- Losses: see the run log",
            f"- {s.get('minutes')} min on {s.get('gpu')}, peak {s.get('peak_gpu_gb')} GB",
            f"- Data `{s.get('data', {}).get('file')}`, sha256 `{s.get('data', {}).get('sha256')}`",
        ]
        if s
        else ["- The run's `run_summary.json` wasn't kept; settings are in the config above."]
    )
    ev = _eval_lines(adapter.get("eval", {}))
    lines = [
        "---",
        f"base_model: {reg['base']}",
        "library_name: peft",
        "tags: [lora, qlora, tau2-bench, reagent]",
        "---",
        "",
        f"# {adapter['id']}: {adapter['method']}",
        "",
        STATUS_WORDS[adapter["status"]],
        "",
        f"A LoRA adapter for [{reg['base']}](https://huggingface.co/{reg['base']}), "
        "trained by the ReAgent team (Lexsi Labs hackathon) to make the student agent "
        "hallucinate less on τ²-bench.",
        "",
        "## What it was trained on",
        "",
        f"- {adapter['data']}",
        f"- Config: `{adapter['train_config']}` (ticket {adapter['ticket']})",
        f"- Parent adapter: {adapter.get('parent') or 'none (from the base model)'}",
        "",
        "## Training",
        "",
        *training,
        "",
        "## Evaluation",
        "",
        "Tasks solved, mean of 2 samples per task, oracle-scored (τ²-bench v2.0.0, customer GPT-4.1).",
        "",
        "| Split | Solved | |",
        "|---|---:|---|",
        *(ev or ["| — | not evaluated | |"]),
        "",
        "Indicative: 30 or 36 tasks are small, and two samples of the same arm differ by up to 4 tasks.",
        "Only arms declared in advance run on test.",
        "",
        "## Use",
        "",
        "```bash",
        f'EXTRA_ADAPTERS="{adapter["served_as"]}=runs/{adapter["repo"]}" just sft-serve',
        f"just rollout <arm> <domain> --config {adapter['arm_config']}",
        "```",
        "",
        "Served at 32K context on one H100 (Aya Expanse 32B allows 128K).",
        "",
        f"Provenance: `lexsi_provenance.json` (code commit `{prov.get('code_commit')}`).",
        "",
    ]
    return "\n".join(lines)
