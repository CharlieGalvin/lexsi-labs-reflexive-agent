# ruff: noqa: E501  (the data card template has long markdown lines)
"""Publish golden trajectories v1 to a PRIVATE Hugging Face dataset (SEL-48).

    just golden                      # build data/golden/v1/ first (no API calls)
    just golden-publish --dry-run    # check the files and write the data card; uploads nothing
    just golden-publish              # upload, private, and tag v1

Uses your own Hugging Face token: ``HF_TOKEN`` in ``.env``, or ``hf auth login``. Before uploading
it re-checks every record (unique IDs, format, no test task, each file holds only its split, no
task in both files, counts match the build report) and refuses to upload to a public repo.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reflexive import golden, splits  # noqa: E402

DIR = ROOT / "data" / "golden" / "v1"
REPO = "reagent-lexsi/tau2-golden-trajectories"
FILES = {"train": "golden_train.jsonl", "dev": "golden_dev_reference.jsonl"}
CONTRACT = "https://claude.ai/artifact/Ew1amyTBrTHbdiW92DQzNw"


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(rows: dict[str, list[dict]], report: dict) -> list[str]:
    """Problems that must stop an upload."""
    problems, ids, tasks = [], collections.Counter(), {}
    for split, recs in rows.items():
        tasks[split] = {(r["domain"], r["task_id"]) for r in recs}
        for r in recs:
            ids[r["id"]] += 1
            if r["format"] != golden.FORMAT:
                problems.append(f"{r['id']}: format {r['format']}")
            if r["split"] != split or splits.split_of(r["domain"], r["task_id"]) != split:
                problems.append(f"{r['id']}: in the {split} file but belongs to {r['split']}")
            if r["outcome"]["reward"] != 1.0:
                problems.append(f"{r['id']}: reward {r['outcome']['reward']}")
            if r["train_on"] != [i for i, m in enumerate(r["messages"]) if m["role"] == "assistant"]:
                problems.append(f"{r['id']}: train_on")
    problems += [f"{i}: duplicated {n} times" for i, n in ids.items() if n > 1]
    if tasks["train"] & tasks["dev"]:
        problems.append(f"{len(tasks['train'] & tasks['dev'])} tasks in both files")
    for split, recs in rows.items():
        want = sum(v[split]["trajectories"] for v in report["domains"].values())
        if len(recs) != want:
            problems.append(f"{split}: {len(recs)} records, build report says {want}")
    return problems


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def splits_order(domain: str, ids: list[str]) -> list[str]:
    """Task IDs in the frozen file's order (numeric for retail and airline)."""
    order = {t: i for s in splits.SPLITS for i, t in enumerate(splits.task_ids(domain, s))}
    return sorted(ids, key=lambda t: (int(t) if t.isdigit() else 0, order.get(t, 0), t))


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def card(rows: dict[str, list[dict]], report: dict, repo: str, shas: dict[str, str]) -> str:
    d = report["domains"]
    sizes = report["train_sizes"]
    providers = collections.Counter(r["teacher"]["provider"] for rs in rows.values() for r in rs)
    customers = collections.Counter(r["customer"] for rs in rows.values() for r in rs)
    tau2_commit = json.loads(splits.SPLITS_PATH.read_text())["source"].get("commit")
    unsolved = {k: splits_order(k, v["train"]["unsolved"] + v["dev"]["unsolved"]) for k, v in d.items()}
    unsolved_md = "\n".join(
        f"  - {k} ({len(v)}): " + (", ".join(v) if k != "telecom" else
                                   "".join(f"\n    - `{t}`" for t in v)) for k, v in unsolved.items())
    temps = collections.Counter((r["teacher"]["attempt"], r["teacher"]["temperature"])
                                for rs in rows.values() for r in rs)
    attempts_md = "; ".join(f"{n} from attempt {a} (temperature {t})" for (a, t), n in sorted(temps.items()))
    count_rows = "\n".join(
        f"| {k} | {v['train']['trajectories']} of {v['train']['tasks']} | "
        f"{v['dev']['trajectories']} of {v['dev']['tasks']} |" for k, v in d.items())
    n_train, n_dev = len(rows["train"]), len(rows["dev"])
    return f"""---
pretty_name: ReAgent golden trajectories (τ²-bench) v1
language: [en]
license: other
license_name: mixed-see-card
tags: [agents, tool-use, tau2-bench, distillation, golden-trajectories]
task_categories: [text-generation]
size_categories: [n<1K]
configs:
- config_name: default
  data_files:
  - split: train
    path: {FILES['train']}
  - split: dev_reference
    path: {FILES['dev']}
---

# ReAgent golden trajectories (τ²-bench), v1

**Private, for the ReAgent team (Lexsi Labs hackathon, Track 3).** {n_train} training conversations,
plus {n_dev} dev conversations for reference, in which a teacher model solved a
[τ²-bench](https://github.com/sierra-research/tau2-bench) v2.0.0 **train** task, verified by
τ²-bench's own evaluator. They're formatted for fine-tuning **Aya Expanse 32B** in Cohere's
`Action:` tool-call format. The format is set by the
[ReAgent data contract v1]({CONTRACT}), agreed 4 Nov 2026.

## Load it

```python
from datasets import load_dataset

train = load_dataset("{repo}", split="train", revision="v1")              # {n_train} records: train on these
dev = load_dataset("{repo}", split="dev_reference", revision="v1")        # {n_dev} records: NEVER train on these
rec = train[0]
rec["messages"]   # [{{"role": "system" | "user" | "assistant", "content": str}}, ...]
rec["train_on"]   # indexes of the assistant turns: the learning targets
```

**Loss on assistant turns only.** Every assistant turn is a target and nothing else is: not
the system prompt, the customer's messages or the tool results. Either mask labels to −100
outside the `train_on` turns, or use TRL's `assistant_only_loss=True` with a chat template that
marks assistant turns.

## Files

| File | Records | Contents |
|---|---:|---|
| `{FILES['train']}` | {n_train} | One golden trajectory per solved **train** task. Train on these. |
| `{FILES['dev']}` | {n_dev} | Trajectories for **dev** tasks: for reference and comparison, **never trained on** |
| `splits.json` | | The frozen task IDs: train 142, dev 36, test 100 (seed `reagent-dev-v1`) |

| Domain | Train: tasks with a trajectory | Dev: tasks with a trajectory |
|---|---:|---:|
{count_rows}
| **Total** | **{n_train} of 142** | **{n_dev} of 36** |

The test split (100 tasks) is τ²-bench's own test split, unchanged. No test task was ever run by
the teacher or appears here.

## One record

| Field | Contents |
|---|---|
| `id` | `<teacher>/<domain>/<task_id>/<attempt>` |
| `format` | `{golden.FORMAT}` |
| `domain`, `task_id`, `split` | The τ²-bench task, and `train` or `dev` |
| `messages` | The conversation **as the agent saw it**, in Cohere format: the `system` prompt (τ²-bench's agent instructions, the domain policy and the tools as Python stubs), then alternating `assistant` and `user` turns |
| `train_on` | Indexes in `messages` of every `assistant` turn |
| `tools` | The agent's tools as OpenAI function schemas, for reference or another format later |
| `outcome` | `reward` (always 1.0) and the score per check, e.g. `{{"DB": 1.0, "COMMUNICATE": 1.0}}` |
| `teacher` | `model`, `provider`, `temperature`, `attempt` |
| `customer` | The simulated customer model |
| `tau2_version`, `source` | τ²-bench version; the saved run this came from, for audits |

In `messages`:
- **A tool call** is an assistant turn: ``Action: ```json [{{"tool_name": ..., "parameters": ...}}]``` ``.
- **A tool result** comes back as a user turn: `<results>\\nDocument: N\\ntool_name: ...\\nresult: ...\\n</results>`.
- **Anything else** is a message to or from the customer.

Sizes per training record: {sizes['turns']['min']}–{sizes['turns']['max']} turns (median
{sizes['turns']['median']}), {sizes['targets']['min']}–{sizes['targets']['max']} targets
(median {sizes['targets']['median']}), and at most {sizes['chars']['max']:,} characters
(about 11–13K tokens). Everything fits a 16K context.

## How it was made

- **Teacher:** GLM-5.3-Flash (`z-ai/glm-5.3-flash` via OpenRouter, MIT licence), chosen in a
  bake-off (it beat Qwen3.8 27B with thinking off decisively on telecom). It ran as the agent with
  native tool calls; each kept trajectory is the task's first clean pass: {attempts_md}.
  Providers: {", ".join(f"{k} {v}" for k, v in providers.most_common())}.
- **Customer:** τ²-bench's user simulator, played by {', '.join(f'`{k}`' for k in customers)}.
- **Scoring:** τ²-bench's own evaluator with judges on. Retail and airline add LLM judges for
  what the agent told the customer; telecom checks the phone's final state.
- **Selection:** the first clean pass per task. Clean means it passed every check, never retried
  a failed call and never called a tool it doesn't have.
- **The agent's view:** in telecom the customer operates their own phone. Those actions and
  their results are removed, because the agent never saw them. Checked to match the real saved
  view exactly in 30 of 30 conversations.
- **Conversion:** the same prompt builder Aya's baseline was measured with, so training and
  evaluation prompts are identical. Every record was parsed back and gives exactly the
  original turns, tool calls, arguments and tool results.
- **Cost:** $11.20 for the teacher run on all 178 train tasks.

Code: `src/reflexive/golden.py`, `scripts/build_golden.py`, `scripts/publish_golden.py` in the
ReAgent repo (commit `{git_commit()}`). τ²-bench commit `{tau2_commit}`.

## Limitations

- **{sum(map(len, unsolved.values()))} train tasks have no trajectory** (the teacher never passed them cleanly).
  They're the core of the hard subset:
{unsolved_md}
- **One teacher, almost always one attempt.** The trajectories reflect GLM-5.3-Flash's style, e.g. markdown in
  replies and an occasional parallel call.
- **Every conversation opens with the agent's fixed greeting** ("Hi! How can I help you today?"),
  before the first customer message. That's how τ²-bench starts a conversation, and Aya's baseline
  saw the same thing. It's a target like any other assistant turn.
- **Providers weren't pinned,** and the customer is OpenRouter's `openai/gpt-4.1`, not the dated
  snapshot τ²-bench pins.
- **English only.** No failed or negative examples yet.

## Licence (to settle before any public release)

- τ²-bench tasks, policies and tools: MIT.
- Teacher (assistant) turns: GLM-5.3-Flash outputs (MIT-licensed model).
- Customer turns: generated by GPT-4.1 and subject to OpenAI's terms. They're context only and
  never a learning target.

This v1 is private to the team.

## Checksums (SHA-256)

{chr(10).join(f'- `{k}`: `{v}`' for k, v in shas.items())}

*Version v1, built {dt.date.today().isoformat()}.*
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=REPO)
    p.add_argument("--dir", type=Path, default=DIR)
    p.add_argument("--dry-run", action="store_true", help="check and write the card; upload nothing")
    a = p.parse_args()

    if not (a.dir / FILES["train"]).exists():
        print(f"⛔ {a.dir} has no {FILES['train']}: run `just golden` first")
        return 1
    rows = {s: load(a.dir / f) for s, f in FILES.items()}
    report = json.loads((a.dir / "build_report.json").read_text())
    problems = validate(rows, report)
    if problems:
        for x in problems[:20]:
            print(f"⛔ {x}")
        print(f"⛔ {len(problems)} problems; nothing uploaded")
        return 1
    print(f"checked: {len(rows['train'])} train + {len(rows['dev'])} dev records, no problems")

    names = [*FILES.values(), "splits.json"]
    shas = {n: sha256(a.dir / n) for n in names}
    (a.dir / "README.md").write_text(card(rows, report, a.repo, shas))
    print(f"wrote {_rel(a.dir)}/README.md (the data card)")
    if a.dry_run:
        print(f"dry run: would upload {', '.join(names)} and README.md to {a.repo} (private), tag v1")
        return 0

    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi()
    try:
        info = api.repo_info(a.repo, repo_type="dataset")
        if not info.private:
            print(f"⛔ {a.repo} exists and is PUBLIC; not uploading. Make it private first.")
            return 1
    except RepositoryNotFoundError:
        api.create_repo(a.repo, repo_type="dataset", private=True)
        print(f"created {a.repo} (private)")
    api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=str(a.dir),
                      allow_patterns=[*names, "README.md"],
                      commit_message=f"Golden trajectories v1 ({len(rows['train'])} train, "
                                     f"{len(rows['dev'])} dev reference)")
    api.create_tag(a.repo, repo_type="dataset", tag="v1", exist_ok=True)
    print(f"uploaded and tagged v1: https://huggingface.co/datasets/{a.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
