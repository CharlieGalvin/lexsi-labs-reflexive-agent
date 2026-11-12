"""The adapter lifecycle (SEL-22): which adapters exist, what each is, and which one is Arm B.

* ``configs/adapters.yaml``: the registry, one entry per adapter, written by hand: what each is
  and what was decided about it (``status``).
* ``results/adapters/state.json``: which adapter is current (Arm B), the Hub revision each was
  published at, and every switch (``history``). Written by ``scripts/adapters/*.py``, committed.

**One source of truth for Arm B:** only the state says which adapter is current; the registry
never does. `status_of` gives an adapter's status as a card shows it: ``current`` for the state's
current adapter, ``superseded`` for an adopted one that was current and was switched away from,
else the registry's.

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
STATUSES = {"adopted", "candidate", "superseded", "not_adopted", "control"}  # the registry's
SWITCHABLE = {"adopted", "candidate", "superseded"}  # can be made current without --force
PROVENANCE_FORMAT = "reagent.provenance/v1"


def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def adapters_by_id(reg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {a["id"]: a for a in reg["adapters"]}


class RegistryError(ValueError):
    """The registry can't be used as it is, or has no such adapter: the message says why."""


def get_adapter(reg: dict[str, Any], adapter_id: str, root: Path = ROOT) -> dict[str, Any]:
    """The adapter's registry entry, after checking the registry. Raises `RegistryError`."""
    problems = check_registry(reg, root)
    if problems:
        raise RegistryError("fix configs/adapters.yaml first:\n  " + "\n  ".join(problems))
    by_id = adapters_by_id(reg)
    if adapter_id not in by_id:
        raise RegistryError(f"no adapter {adapter_id!r}; the registry has {sorted(by_id)}")
    return by_id[adapter_id]


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
    return problems


def check_state(reg: dict[str, Any], state: dict[str, Any]) -> list[str]:
    """What's wrong with the state against the registry (empty if nothing)."""
    ids = set(adapters_by_id(reg))
    problems = []
    if state.get("current") is not None and state["current"] not in ids:
        problems.append(f"state.json's current adapter {state['current']!r} isn't in the registry")
    for h in state.get("history", []):
        for k in ("from", "to"):
            if h.get(k) is not None and h[k] not in ids:
                problems.append(f"state.json's history names {h[k]!r}, which isn't in the registry")
    return problems


def status_of(adapter: dict[str, Any], state: dict[str, Any]) -> str:
    """The adapter's status as published: the state decides which one is current (see the top)."""
    if adapter["id"] == state.get("current"):
        return "current"
    was_current = any(h.get("to") == adapter["id"] for h in state.get("history", []))
    if adapter["status"] == "adopted" and was_current:
        return "superseded"
    return adapter["status"]


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


def git_state(root: Path = ROOT) -> dict[str, Any]:
    """The checked-out commit and whether tracked files have uncommitted changes (None if unknown)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        dirty: bool | None = bool(out.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    return {"commit": git_commit(root), "dirty": dirty}


def provenance(
    reg: dict[str, Any],
    adapter: dict[str, Any],
    summary: dict[str, Any] | None,
    published_from: dict[str, Any] | None,
    status: str | None = None,
) -> dict[str, Any]:
    """``lexsi_provenance.json``: where the adapter came from, enough to rebuild or audit it.

    ``summary`` is train_sft.py's ``run_summary.json`` when it exists (data checksum, settings,
    losses, steps, GPU, and since SEL-22 the code it was trained with); without it the card says so
    instead of guessing. ``code_commit`` is the **training** commit from the summary, None when it
    wasn't recorded; ``published_from`` (`git_state` at publish time) is only where the card was
    written from. ``status``: `status_of` (default: the registry's)."""
    code = (summary or {}).get("code") or {}
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
        "status": status or adapter["status"],
        "code_commit": code.get("commit"),
        "code_dirty": code.get("dirty"),
        "published_from": published_from,
        "training": summary,  # None when the run's summary wasn't kept
        "eval": adapter.get("eval", {}),
        "metrics_files": adapter.get("metrics", []),
        "written": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }


EVAL_ROWS = (  # (key, label, task count when the entry gives no ``of``)
    ("val", "Validation (unseen train tasks)", 30),
    ("dev", "Dev", 36),
    ("test", "Test", 100),
)


def _eval_lines(ev: dict[str, Any]) -> list[str]:
    """One row per split: the entry's own task count (``of``) and samples, never assumed."""
    lines = []
    for key, label, default_of in EVAL_ROWS:
        if key not in ev:
            continue
        e = ev[key]
        of = e.get("of", default_of)
        samples = e.get("samples") or []
        n = e.get("n_samples", len(samples) or None)
        how = f"mean of {n}" if n else "mean"
        how += f"; samples {', '.join(map(str, samples))}" if samples else ""
        lines.append(f"| {label}, {of} tasks | {e['mean']} / {of} | {how} |")
    return lines


STATUS_WORDS = {
    "current": "**Arm B**: the adapter served as the project's healed student.",
    "candidate": "A candidate, not yet decided.",
    "superseded": "Superseded: it was the current adapter and has been replaced.",
    "not_adopted": "Not adopted: its pre-registered rule didn't hold. Kept for the record.",
    "control": "A control (an ablation), never a candidate.",
}


def _loss_line(s: dict[str, Any]) -> str:
    loss = s.get("held_out_loss") or {}
    if s.get("train_loss") is None:
        return "- Losses: see the run log"
    line = f"- Train loss {s['train_loss']:.3f}"
    if loss.get("before") is not None and loss.get("after") is not None:
        line += f"; held-out loss {loss['before']:.3f} → {loss['after']:.3f}"
    else:
        line += "; no held-out set"
    return line


def model_card(reg: dict[str, Any], adapter: dict[str, Any], prov: dict[str, Any]) -> str:
    """README.md for the adapter's Hub repo. The status is the provenance's (`status_of`)."""
    s = prov.get("training") or {}
    tr = s.get("trainer", {})
    lora = s.get("lora", {})
    training = (
        [
            f"- Steps: {s.get('steps')}, learning rate {tr.get('learning_rate')}, "
            f"LoRA r{lora.get('r')} / alpha {lora.get('lora_alpha')}",
            _loss_line(s),
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
        STATUS_WORDS[prov.get("status") or adapter["status"]],
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
        "Tasks solved per run, oracle-scored (τ²-bench v2.0.0, customer GPT-4.1).",
        "",
        "| Split | Solved | |",
        "|---|---:|---|",
        *(ev or ["| — | not evaluated | |"]),
        "",
        "Indicative: these task sets are small, and two samples of the same arm differ by up to 4 tasks.",
        "Only arms declared in advance run on test.",
        "",
        "## Use",
        "",
        "```bash",
        f"huggingface-cli download {hub_repo(reg, adapter)} --local-dir runs/hub/{adapter['repo']}",
        f'ADAPTER= EXTRA_ADAPTERS="{adapter["served_as"]}=runs/hub/{adapter["repo"]}" just sft-serve',
        f"just rollout <arm> <domain> --config {adapter['arm_config']}",
        "```",
        "",
        "Served at 32K context on one H100 (Aya Expanse 32B allows 128K).",
        "",
        "Provenance: `lexsi_provenance.json`; "
        + (
            f"trained from code commit `{prov['code_commit']}`"
            + (" with uncommitted changes" if prov.get("code_dirty") else "")
            if prov.get("code_commit")
            else "the training commit wasn't recorded (trained before SEL-22)"
        )
        + ".",
        "",
    ]
    return "\n".join(lines)
