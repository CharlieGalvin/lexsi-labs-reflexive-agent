# ruff: noqa: E501  (the data card template has long markdown lines)
"""Publish golden trajectories to a PRIVATE Hugging Face dataset: v1 (SEL-48), v1.1 with the hard subset (SEL-60).

    just golden                                  # build data/golden/v1/ first (no API calls)
    just hard-subset                             # build data/golden/hard/ (for v1.1; no API calls)
    just golden-publish --tag v1.1 --dry-run     # check the files and write the data card; uploads nothing
    just golden-publish --tag v1.1               # upload, private, and tag v1.1

``--tag v1`` (the default) publishes the golden trajectories alone, as v1 was. Any other tag
adds the hard subset (``hard_subset.jsonl``) and a section about it in the data card. Its records
are checked too (format, train tasks only, each group consistent with Arm A's result and the
golden fixes, counts match its build report), and it must have been built from this golden v1.

Uses your own Hugging Face token: ``HF_TOKEN`` in ``.env``, or ``hf auth login``. Before uploading
it re-checks every record (unique IDs, format, no test task, each file holds only its split, no
task in both files, counts match the build report), checks the files are byte-for-byte the ones
`just golden` verified, and refuses to upload to a public repo. If the tag already exists it
stops, unless you pass ``--retag`` to move it to the new upload. Everything goes up in one commit,
and the tag points at that commit.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reflexive import golden, hard, splits
from reflexive.files import ROOT, BadJsonLine, read_jsonl, sha256, shown

DIR = ROOT / "data" / "golden" / "v1"
REPO = "reagent-lexsi/tau2-golden-trajectories"
FILES = {"train": "golden_train.jsonl", "dev": "golden_dev_reference.jsonl"}
TAG = "v1"
HARD_DIR = ROOT / "data" / "golden" / "hard"
HARD_FILE = "hard_subset.jsonl"
HARD_REPORT = "hard_report.json"


def validate(rows: dict[str, list[dict]], report: dict) -> list[str]:
    """Problems that must stop an upload."""
    problems, ids = [], collections.Counter()
    tasks: dict[str, set[tuple[str, str]]] = {split: set() for split in rows}
    for split, recs in rows.items():
        for n, r in enumerate(recs):
            try:  # every key is read inside the try: a malformed record is a problem, never a crash
                tasks[split].add((r["domain"], r["task_id"]))
                ids[r["id"]] += 1
                if r["format"] != golden.FORMAT:
                    problems.append(f"{r['id']}: format {r['format']}")
                try:
                    belongs = splits.split_of(r["domain"], r["task_id"])
                except KeyError:
                    belongs = "no split"
                if r["split"] != split or belongs != split:
                    problems.append(f"{r['id']}: in the {split} file but belongs to {belongs}")
                if r["outcome"]["reward"] != 1.0:
                    problems.append(f"{r['id']}: reward {r['outcome']['reward']}")
                if r["train_on"] != [i for i, m in enumerate(r["messages"]) if m["role"] == "assistant"]:
                    problems.append(f"{r['id']}: train_on")
            except (KeyError, TypeError, AttributeError) as e:
                problems.append(f"{split} record {n + 1}: malformed ({type(e).__name__}: {e})")
    problems += [f"{i}: duplicated {n} times" for i, n in ids.items() if n > 1]
    if tasks["train"] & tasks["dev"]:
        problems.append(f"{len(tasks['train'] & tasks['dev'])} tasks in both files")
    for split, recs in rows.items():
        try:
            want = sum(v[split]["trajectories"] for v in report["domains"].values())
        except (KeyError, TypeError):
            problems.append(f"{split}: the build report has no trajectory count")
            continue
        if len(recs) != want:
            problems.append(f"{split}: {len(recs)} records, build report says {want}")
    return problems


def check_files(folder: Path, report: dict) -> list[str]:
    """The files must be the ones `just golden` built and round-trip checked, and splits.json the live one."""
    problems = []
    built = report.get("files")
    if not built:
        return ["the build report has no file checksums: rebuild with `just golden`"]
    shas = {name: sha256(folder / name) for name in [*FILES.values(), "splits.json"]}
    for name, sha in shas.items():
        if built.get(name) != sha:
            problems.append(f"{name} changed since `just golden` checked it: rebuild")
    if shas["splits.json"] != sha256(splits.SPLITS_PATH):
        problems.append(f"splits.json differs from {shown(splits.SPLITS_PATH)}: rebuild with `just golden`")
    return problems


def validate_hard(recs: list[dict], report: dict, golden_train: list[dict]) -> list[str]:
    """Problems with the hard subset that must stop an upload."""
    problems, ids = [], collections.Counter()
    fixes = {(g["domain"], g["task_id"]): g["id"] for g in golden_train}
    for n, r in enumerate(recs):
        try:
            rid, key = r["id"], (r["domain"], r["task_id"])
            ids[rid] += 1
            if r["format"] != hard.FORMAT:
                problems.append(f"{rid}: format {r['format']}")
            try:
                belongs = splits.split_of(*key)
            except KeyError:
                belongs = "no split"
            if r["split"] != "train" or belongs != "train":
                problems.append(f"{rid}: in the hard subset but belongs to {belongs}")
            passed, fix = r["arm_a"]["passed"], fixes.get(key)
            want = hard.group_of(bool(passed), fix is not None)
            if r["group"] != want:
                problems.append(
                    f"{rid}: group {r['group']}, but Arm A passed={passed} and fix={fix} make it {want}"
                )
            if r["golden_id"] != fix:
                problems.append(f"{rid}: golden_id {r['golden_id']}, golden v1 has {fix}")
            if not r["labels"] or set(r["labels"]) - set(hard.LABELS):
                problems.append(f"{rid}: labels {r['labels']}")
            if not r["messages"]:
                problems.append(f"{rid}: no Arm A conversation")
        except (KeyError, TypeError, AttributeError) as e:
            problems.append(f"hard subset record {n + 1}: malformed ({type(e).__name__}: {e})")
    problems += [f"{i}: duplicated {n} times" for i, n in ids.items() if n > 1]
    present = {(r.get("domain"), r.get("task_id")) for r in recs if isinstance(r, dict)}
    domains = [d for d in splits.DOMAINS if any(k[0] == d for k in present)]
    missing = [f"{d} task {t}" for d in domains for t in splits.task_ids(d, "train") if (d, t) not in present]
    if missing:
        problems.append(f"{len(missing)} train tasks missing from the hard subset, e.g. {missing[0]}")
    if problems:
        return problems
    # The card is built from the report, so every number in it must be what the records give
    dev = report.get("dev_left_out") if isinstance(report, dict) else None
    want = hard.counts(recs, collections.Counter(dev if isinstance(dev, dict) else {}))
    for key in ("domains", "total", "thin", "dev_left_out", "min_examples"):
        if report.get(key) != want[key]:
            problems.append(
                f"the hard subset's build report: {key} doesn't match its records; "
                "rebuild with `just hard-subset --force`"
            )
    return problems


def check_hard_files(hard_dir: Path, hard_report: dict, golden_dir: Path) -> list[str]:
    """The hard subset must be the file `just hard-subset` wrote, built from this golden v1."""
    problems = []
    if (hard_report.get("files") or {}).get(HARD_FILE) != sha256(hard_dir / HARD_FILE):
        problems.append(f"{HARD_FILE} changed since `just hard-subset` built it: rebuild")
    code = hard_report.get("code") if isinstance(hard_report.get("code"), dict) else {}
    if not code.get("commit"):
        problems.append(
            f"{HARD_REPORT} doesn't say which commit labelled the subset: rebuild with `just hard-subset --force`"
        )
    elif not isinstance(code.get("files"), dict) or not code["files"]:
        problems.append(
            f"{HARD_REPORT} doesn't list the label code's checksums: rebuild with `just hard-subset --force`"
        )
    elif code.get("uncommitted") is None or code["uncommitted"]:
        what = ", ".join(code.get("uncommitted") or []) or "the label code (not checked)"
        problems.append(f"{HARD_FILE} was labelled by uncommitted code ({what}): commit it and rebuild")
    built_from = ((hard_report.get("inputs") or {}).get("golden") or {}).get("sha256")
    if built_from != sha256(golden_dir / FILES["train"]):
        problems.append(
            f"{HARD_FILE} was built from another {FILES['train']}: rebuild with `just hard-subset --force`"
        )
    return problems


def splits_order(domain: str, ids: list[str]) -> list[str]:
    """Task IDs numerically (retail, airline), else in the frozen file's order (train, dev, test)."""
    file_order = [t for s in splits.SPLITS for t in splits.task_ids(domain, s)]
    order = {t: i for i, t in enumerate(file_order)}
    return sorted(ids, key=lambda t: (0, int(t), t) if t.isdigit() else (1, order.get(t, len(order)), t))


def _ids_md(by_domain: dict[str, list[str]]) -> str:
    return "\n".join(
        f"  - {k} ({len(v)}): " + (", ".join(v) if k != "telecom" else "".join(f"\n    - `{t}`" for t in v))
        for k, v in by_domain.items()
        if v
    )


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def hard_section(recs: list[dict], report: dict, repo: str, tag: str) -> str:
    """The data card's section on the hard subset (SEL-60)."""
    rows_md = "\n".join(
        f"| {d} | " + " | ".join(str(v["groups"][g]) for g in hard.GROUPS) + f" | {v['tasks']} |"
        for d, v in report["domains"].items()
    )
    total = report["total"]
    healable = total["labels"]["healable"]
    customer = total["customer_tool"]["healable"]
    aya_only = [
        f"{r['domain']} `{r['task_id']}`"
        for r in recs
        if r["group"] == "solved_by_aya" and not r["golden_id"]
    ]
    dev = {d: n for d, n in report.get("dev_left_out", {}).items() if n}
    dev_md = (
        "- **Dev tasks are left out** ("
        + ", ".join(f"{d} {n}" for d, n in dev.items())
        + "): never trained on.\n"
        if dev
        else ""
    )
    aya_md = (
        f"- **Aya passes {len(aya_only)} tasks the teacher never solved:** {', '.join(aya_only)}.\n"
        if aya_only
        else ""
    )
    thin_md = (
        f"- **Thin types:** fewer than {report['min_examples']} healable tasks with {', '.join(report['thin'])}.\n"
        if report["thin"]
        else ""
    )
    code_md = ", ".join(f"`{Path(f).name}` `{s}`" for f, s in report["code"]["files"].items())
    return f"""
## Hard subset (added in v1.1)

`{HARD_FILE}`: {len(recs)} records, one per frozen **train** task. Each one records how the
untuned student did (**Arm A**: Aya Expanse 32B with τ²-bench's own agent prompt, temperature
0, one conversation per task), what went wrong, and whether a golden fix exists.

```python
hard = load_dataset("{repo}", "hard_subset", split="train", revision="{tag}")
fixes = {{r["id"]: r for r in train}}                   # the golden trajectories above
healable = [(h, fixes[h["golden_id"]]) for h in hard if h["group"] == "healable"]
```

| Group | Meaning | Use |
|---|---|---|
| `healable` | Arm A failed, and the teacher's verified fix is in `{FILES["train"]}` (`golden_id`) | What healing trains on: the fix, weighted by failure type |
| `teacher_unsolved` | Arm A failed, and the teacher never solved it | Beyond the teacher for now; not trainable |
| `solved_by_aya` | Arm A passed | Checks that healing doesn't break what already works |

| Domain | Healable | Teacher-unsolved | Solved by Aya | Tasks |
|---|---:|---:|---:|---:|
{rows_md}
| **Total** | **{total["groups"]["healable"]}** | **{total["groups"]["teacher_unsolved"]}** | **{total["groups"]["solved_by_aya"]}** | **{total["tasks"]}** |

**Failure labels** (`labels`): every failure type the classifier finds in Arm A's conversation,
so a task can have several. Among the {total["groups"]["healable"]} healable tasks: TOOL {healable["TOOL"]}
({customer} of them calls to the customer's own phone tools, `customer_tool`), ARG {healable["ARG"]},
LOOP {healable["LOOP"]}, and NONE {healable["NONE"]} (no structural slip: it failed on reasoning or
policy, which the classifier doesn't judge). `primary` is the earliest one, since later
failures usually follow from it.

| Field | Contents |
|---|---|
| `id` | `armA/<domain>/<task_id>` |
| `format` | `{hard.FORMAT}` |
| `domain`, `task_id`, `split` | The τ²-bench task; `split` is always `train` |
| `group`, `golden_id` | As above; `golden_id` is the fix's `id` in `{FILES["train"]}`, or null |
| `labels`, `primary`, `first`, `kinds`, `customer_tool` | The classifier's findings: all types, the earliest, its detail, counts per kind |
| `arm_a` | `passed` (0 or 1), `reward`, `checks`, `termination`, `student`, `samples` (1), and the source run file |
| `messages` | Arm A's conversation **as it saw it**, in OpenAI format: context for the failure, **never a learning target** |

{dev_md}{aya_md}{thin_md}- **One conversation per task.** Arm A and the customer both run at temperature 0, so another
  run would mostly repeat it. Still, one run is one run: a task it failed once might pass on another.
- **Labels are structural.** The classifier is deterministic and checks tool names, arguments
  against the schemas and the conversation, and repetition. Code: `src/reflexive/hard.py`,
  `src/reflexive/hallucination.py`, `scripts/data/build_hard_subset.py`.
- **Labelled by** ReAgent commit `{report["code"]["commit"][:12]}`: {code_md}. A rebuild with
  the same files gives the same subset.
"""


def card(
    rows: dict[str, list[dict]],
    report: dict,
    repo: str,
    shas: dict[str, str],
    tag: str = TAG,
    hard_part: tuple[list[dict], dict] | None = None,
) -> str:
    d = report["domains"]
    sizes = report["train_sizes"]
    providers = collections.Counter(r["teacher"]["provider"] for rs in rows.values() for r in rs)
    customers = collections.Counter(r["customer"] for rs in rows.values() for r in rs)
    meta = splits.load()  # check_files has made sure it's the same file as the uploaded splits.json
    tau2_commit = meta["source"].get("commit")
    split_sizes = {s: sum(len(v[s]) for v in meta["domains"].values()) for s in splits.SPLITS}
    seed = meta.get("dev_method", {}).get("seed")
    unsolved = {k: splits_order(k, v["train"]["unsolved"]) for k, v in d.items()}
    unsolved_dev = {k: splits_order(k, v["dev"]["unsolved"]) for k, v in d.items()}
    n_unsolved, n_unsolved_dev = sum(map(len, unsolved.values())), sum(map(len, unsolved_dev.values()))
    unsolved_md = _ids_md(unsolved)
    unsolved_dev_md = (
        (f"- **{n_unsolved_dev} dev tasks have no reference trajectory either:**\n{_ids_md(unsolved_dev)}\n")
        if n_unsolved_dev
        else ""
    )
    tasks_train = sum(v["train"]["tasks"] for v in d.values())
    tasks_dev = sum(v["dev"]["tasks"] for v in d.values())
    max_chars = sizes["chars"]["max"]
    tok_lo, tok_hi = round(max_chars / 4 / 1000), round(max_chars / 3.5 / 1000)  # 3.5–4 chars per token
    fits = (
        "Everything fits a 16K context."
        if max_chars / 3.5 < 16_000
        else "**The longest records may not fit a 16K context:** check token counts before choosing a max length."
    )
    temps = collections.Counter(
        (r["teacher"]["attempt"], r["teacher"]["temperature"]) for rs in rows.values() for r in rs
    )
    attempts_md = "; ".join(f"{n} from attempt {a} (temperature {t})" for (a, t), n in sorted(temps.items()))
    count_rows = "\n".join(
        f"| {k} | {v['train']['trajectories']} of {v['train']['tasks']} | "
        f"{v['dev']['trajectories']} of {v['dev']['tasks']} |"
        for k, v in d.items()
    )
    n_train, n_dev = len(rows["train"]), len(rows["dev"])
    hard_config = (
        f"""
- config_name: hard_subset
  data_files:
  - split: train
    path: {HARD_FILE}"""
        if hard_part
        else ""
    )
    hard_md = hard_section(*hard_part, repo, tag) if hard_part else ""
    unsolved_note = (
        "With no fix, none of them is healable: the hard subset above shows how Arm A did on each."
        if hard_part
        else "They're the core of the hard subset:"
    )
    negatives = (
        "The hard subset's Arm A conversations are failures, kept as context: never learning targets."
        if hard_part
        else "No failed or negative examples yet."
    )
    return f"""---
pretty_name: ReAgent golden trajectories (τ²-bench) {tag}
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
    path: {FILES["train"]}
  - split: dev_reference
    path: {FILES["dev"]}{hard_config}
---

# ReAgent golden trajectories (τ²-bench), {tag}

**Private, for the ReAgent team (Lexsi Labs hackathon, Track 3).** {n_train} training conversations,
plus {n_dev} dev conversations for reference, in which a teacher model solved a
[τ²-bench](https://github.com/sierra-research/tau2-bench) v2.0.0 **train** task, verified by
τ²-bench's own evaluator. They're formatted for fine-tuning **Aya Expanse 32B** in Cohere's
`Action:` tool-call format. The format is set by the same prompt builder Aya's baseline was
measured with, so training and evaluation prompts are identical.

## Load it

```python
from datasets import load_dataset

train = load_dataset("{repo}", split="train", revision="{tag}")              # {n_train} records: train on these
dev = load_dataset("{repo}", split="dev_reference", revision="{tag}")        # {n_dev} records: NEVER train on these
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
| `{FILES["train"]}` | {n_train} | One golden trajectory per solved **train** task. Train on these. |
| `{FILES["dev"]}` | {n_dev} | Trajectories for **dev** tasks: for reference and comparison, **never trained on** |
| `splits.json` | | The frozen task IDs: train {split_sizes["train"]}, dev {split_sizes["dev"]}, test {split_sizes["test"]} (seed `{seed}`) |

| Domain | Train: tasks with a trajectory | Dev: tasks with a trajectory |
|---|---:|---:|
{count_rows}
| **Total** | **{n_train} of {tasks_train}** | **{n_dev} of {tasks_dev}** |

The test split ({split_sizes["test"]} tasks) is τ²-bench's own test split, unchanged. No test task was ever run by
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

Sizes per training record: {sizes["turns"]["min"]}–{sizes["turns"]["max"]} turns (median
{sizes["turns"]["median"]}), {sizes["targets"]["min"]}–{sizes["targets"]["max"]} targets
(median {sizes["targets"]["median"]}), and at most {max_chars:,} characters
(about {tok_lo}–{tok_hi}K tokens). {fits}

## How it was made

- **Teacher:** GLM-5.3-Flash (`z-ai/glm-5.3-flash` via OpenRouter, MIT licence), chosen in a
  bake-off (it beat Qwen3.8 27B with thinking off decisively on telecom). It ran as the agent with
  native tool calls; each kept trajectory is the task's first clean pass: {attempts_md}.
  Providers: {", ".join(f"{k} {v}" for k, v in providers.most_common())}.
- **Customer:** τ²-bench's user simulator, played by {", ".join(f"`{k}`" for k in customers)}.
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
{hard_md}
Code: `src/reflexive/golden.py`, `scripts/data/build_golden.py`, `scripts/data/publish_golden.py` in the
ReAgent repo (commit `{git_commit()}`). τ²-bench commit `{tau2_commit}`.

## Limitations

- **{n_unsolved} train tasks have no trajectory** (the teacher never passed them cleanly).
  {unsolved_note}
{unsolved_md}
{unsolved_dev_md}- **One teacher, almost always one attempt.** The trajectories reflect GLM-5.3-Flash's style, e.g. markdown in
  replies and an occasional parallel call.
- **Every conversation opens with the agent's fixed greeting** ("Hi! How can I help you today?"),
  before the first customer message. That's how τ²-bench starts a conversation, and Aya's baseline
  saw the same thing. It's a target like any other assistant turn.
- **Providers weren't pinned,** and the customer is OpenRouter's `openai/gpt-4.1`, not the dated
  snapshot τ²-bench pins.
- **English only.** {negatives}

## Licence (to settle before any public release)

- τ²-bench tasks, policies and tools: MIT.
- Teacher (assistant) turns: GLM-5.3-Flash outputs (MIT-licensed model).
- Customer turns: generated by GPT-4.1 and subject to OpenAI's terms. They're context only and
  never a learning target.

This {tag} is private to the team.

## Checksums (SHA-256)

{chr(10).join(f"- `{k}`: `{v}`" for k, v in shas.items())}

*Version {tag}, built {dt.date.today().isoformat()}.*
"""


@dataclass
class HubState:
    """What the Hub already has for the dataset (read-only)."""

    exists: bool
    private: bool = True
    tagged: bool = False


def hub_state(api: Any, repo: str, tag: str = TAG) -> HubState:
    """Does the dataset exist, is it private, and does it already have the tag?"""
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        info = api.repo_info(repo, repo_type="dataset")
    except RepositoryNotFoundError:
        return HubState(exists=False)
    tags = api.list_repo_refs(repo, repo_type="dataset").tags
    return HubState(exists=True, private=bool(info.private), tagged=any(t.name == tag for t in tags))


def refusal(state: HubState, repo: str, *, retag: bool, tag: str = TAG) -> str | None:
    """Why the upload must not go ahead, or None."""
    if state.exists and not state.private:
        return f"{repo} exists and is PUBLIC; not uploading. Make it private first."
    if retag and tag == TAG:
        return f"{TAG} is frozen: the Trainer has trained on it, so --retag can't move it. Publish a new tag."
    if state.tagged and not retag:
        return (
            f"{repo} already has tag {tag}; not uploading. Anyone who trained on {tag} expects it "
            f"never to change: publish the new data under a new tag instead (see --retag)."
        )
    return None


def _problems(problems: list[str]) -> None:
    for x in problems[:20]:
        print(f"⛔ {x}")
    print(f"⛔ {len(problems)} problems; nothing uploaded")


def load_checked(folder: Path) -> tuple[dict[str, list[dict[str, Any]]], dict] | None:
    """The records and the build report, if every check passes; else print the problems, return None."""
    if not (folder / FILES["train"]).exists():
        print(f"⛔ {folder} has no {FILES['train']}: run `just golden` first")
        return None
    try:
        rows = {s: read_jsonl(folder / f) for s, f in FILES.items()}
    except BadJsonLine as e:
        print(f"⛔ {e}")
        return None
    report = json.loads((folder / "build_report.json").read_text())
    problems = validate(rows, report) + check_files(folder, report)
    if problems:
        _problems(problems)
        return None
    print(f"checked: {len(rows['train'])} train + {len(rows['dev'])} dev records, no problems")
    return rows, report


def load_checked_hard(
    hard_dir: Path, golden_dir: Path, golden_train: list[dict]
) -> tuple[list[dict], dict] | None:
    """The hard subset and its build report, if every check passes; else print the problems, return None."""
    if not (hard_dir / HARD_FILE).exists():
        print(f"⛔ {shown(hard_dir)} has no {HARD_FILE}: run `just hard-subset` first")
        return None
    try:
        recs = read_jsonl(hard_dir / HARD_FILE)
    except BadJsonLine as e:
        print(f"⛔ {e}")
        return None
    try:
        report = json.loads((hard_dir / HARD_REPORT).read_text())
    except (OSError, ValueError) as e:
        print(
            f"⛔ can't read {shown(hard_dir / HARD_REPORT)} ({type(e).__name__}): rebuild with `just hard-subset --force`"
        )
        return None
    if not isinstance(report, dict):
        print(
            f"⛔ {shown(hard_dir / HARD_REPORT)} isn't a build report: rebuild with `just hard-subset --force`"
        )
        return None
    problems = validate_hard(recs, report, golden_train) + check_hard_files(hard_dir, report, golden_dir)
    if problems:
        _problems(problems)
        return None
    print(f"checked: {len(recs)} hard-subset records, no problems")
    return recs, report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=REPO)
    p.add_argument("--dir", type=Path, default=DIR)
    p.add_argument(
        "--tag",
        default=TAG,
        help=f"the version to publish (default {TAG}). Any tag but {TAG} adds the hard subset, e.g. v1.1",
    )
    p.add_argument(
        "--hard", type=Path, default=HARD_DIR, help="the hard subset's folder (`just hard-subset`)"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="check, write the card and look at the Hub; upload nothing"
    )
    p.add_argument(
        "--retag",
        action="store_true",
        help="move an existing tag to this upload. Only before anyone has trained on it; "
        "after the hand-over, publish under a new tag instead",
    )
    a = p.parse_args()
    tag = a.tag
    if not re.fullmatch(r"v\d+(\.\d+)*", tag):
        print(f"⛔ --tag {tag!r}: a version like v1.1 or v2")
        return 1
    if a.retag and tag == TAG:
        print(
            f"⛔ {TAG} is frozen: the Trainer has trained on it, so --retag can't move it. Publish a new tag."
        )
        return 1
    if tag != TAG and tag.rstrip("0").rstrip(".") == TAG:
        print(f"⛔ --tag {tag} is the same version as {TAG}, which is frozen: publish v1.1 or later")
        return 1

    checked = load_checked(a.dir)
    if checked is None:
        return 1
    rows, report = checked
    names = [*FILES.values(), "splits.json"]
    shas = {n: report["files"][n] for n in names}  # check_files just confirmed they match the files
    paths = {n: a.dir / n for n in names}
    hard_part = None
    if tag != TAG:
        hard_part = load_checked_hard(a.hard, a.dir, rows["train"])
        if hard_part is None:
            return 1
        shas[HARD_FILE] = hard_part[1]["files"][HARD_FILE]
        paths[HARD_FILE] = a.hard / HARD_FILE
    (a.dir / "README.md").write_text(card(rows, report, a.repo, shas, tag, hard_part))
    paths["README.md"] = a.dir / "README.md"
    print(f"wrote {shown(a.dir)}/README.md (the data card)")

    if a.dry_run:
        print(f"dry run: would upload {', '.join(paths)} to {a.repo} (private), tag {tag}")
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            state = hub_state(api, a.repo, tag)
        except Exception as e:  # noqa: BLE001 - no token or offline: the dry run still checked the files
            print(f"   (couldn't check the Hub: {type(e).__name__})")
            return 0
        why = refusal(state, a.repo, retag=a.retag, tag=tag)
        print(
            f"   ⛔ the real run would stop: {why}"
            if why
            else f"   the Hub: {'exists' if state.exists else 'new'}, ok"
        )
        return 0

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        print("⛔ huggingface_hub isn't installed: it comes with the dev tools (`uv sync --extra dev`)")
        return 1
    api = HfApi()
    state = hub_state(api, a.repo, tag)
    why = refusal(state, a.repo, retag=a.retag, tag=tag)
    if why:
        print(f"⛔ {why}")
        return 1
    if not state.exists:
        api.create_repo(a.repo, repo_type="dataset", private=True)
        print(f"created {a.repo} (private)")
    extra = f", {len(hard_part[0])} hard subset" if hard_part else ""
    commit = api.create_commit(  # one commit for every file, so the tag can't land on a half upload
        repo_id=a.repo,
        repo_type="dataset",
        operations=[CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(f)) for n, f in paths.items()],
        commit_message=f"Golden trajectories {tag} ({len(rows['train'])} train, {len(rows['dev'])} dev reference{extra})",
    )
    if state.tagged:
        api.delete_tag(a.repo, repo_type="dataset", tag=tag)
    api.create_tag(a.repo, repo_type="dataset", tag=tag, revision=commit.oid)  # the commit just uploaded
    print(
        f"uploaded and {'re' if state.tagged else ''}tagged {tag}: https://huggingface.co/datasets/{a.repo}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
