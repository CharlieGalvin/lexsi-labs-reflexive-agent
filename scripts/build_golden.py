"""Build golden trajectories v1 (SEL-9) from the teacher's fix run. No API calls, no spend.

    python scripts/build_golden.py                    # results/fixes/glm53f.jsonl → data/golden/v1/
    python scripts/build_golden.py --sample 5         # also write 5 records to read (markdown)

Steps, each one stops the build if it fails:

1. **System prompt and tools** per domain, rebuilt offline from τ²-bench (no conversation runs).
2. **Agent's-view check:** on the conversations that saved the real agent's view
   (``results/fixes/aya.jsonl``), the rebuilt view must match it exactly.
3. **Pick** the first clean pass per task (`golden.pick`); dev tasks go to the reference file.
   A test task anywhere is an error.
4. **Convert** to Cohere format and **check** every record (`golden.check_record`): the text
   reads back to exactly the original turns, roles alternate, ``train_on`` is every assistant turn.

Writes ``golden_train.jsonl``, ``golden_dev_reference.jsonl``, ``splits.json`` and
``build_report.json`` to ``data/golden/v1/`` (gitignored: the dataset goes to Hugging Face).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reflexive import golden, splits  # noqa: E402
from reflexive.files import BadJsonLine, read_jsonl, sha256, shown  # noqa: E402

DEFAULT_IN = ROOT / "results" / "fixes" / "glm53f.jsonl"
VIEW_CHECK = ROOT / "results" / "fixes" / "aya.jsonl"
OUT = ROOT / "data" / "golden" / "v1"


def prompts_and_tools(domain: str, task_id: str) -> tuple[str, list[dict]]:
    """τ²-bench's agent system prompt and tool schemas for a task, without starting a conversation."""
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT

    from reflexive.episode import _NoScoreAgentGymEnv

    env = _NoScoreAgentGymEnv(domain=domain, task_id=task_id, max_steps=200, user_llm="gpt-4.1",
                              user_llm_args={}, language="en")
    agent = env._get_orchestrator().agent
    prompt = SYSTEM_PROMPT.format(agent_instruction=AGENT_INSTRUCTION, domain_policy=agent.domain_policy)
    return prompt, [t.openai_schema for t in agent.tools if t.name != "done"]


def sha(path: Path) -> str:
    return sha256(path)[:16]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_IN)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--sample", type=int, default=0, help="write N random train records to read")
    a = p.parse_args()
    from loguru import logger

    logger.remove()
    try:
        recs = read_jsonl(a.input)
        saved_views = read_jsonl(VIEW_CHECK) if VIEW_CHECK.exists() else []
    except BadJsonLine as e:
        print(f"⛔ {e}")
        return 1
    print(f"{len(recs)} conversations from {shown(a.input)}")

    # 1. System prompt and tools, one per domain
    first = {}
    for r in recs:
        first.setdefault(r["domain"], str(r["task_id"]))
    system = {d: prompts_and_tools(d, t) for d, t in first.items()}

    # 2. The agent's view must match what the agent really saw
    checked = 0
    for r in saved_views:
        if "agent_view" not in r:
            continue
        sp, _ = system.get(r["domain"]) or prompts_and_tools(r["domain"], r["task_id"])
        if golden.agent_view(r["messages"], sp) != r["agent_view"]:
            print(f"⛔ agent's view differs from the saved one: {r['domain']} {r['task_id']}")
            return 1
        checked += 1
    if not checked:
        print("⛔ no saved agent's view to check against (results/fixes/aya.jsonl)")
        return 1
    print(f"1-2. system prompts rebuilt; agent's view matches the saved one exactly in {checked}/{checked}")

    # 3. Pick one clean pass per task
    for r in recs:
        splits.assert_not_test(r["domain"], r["task_id"])
    reasons = collections.Counter(golden.why_not_clean(r) or "clean" for r in recs)
    picked = golden.pick(recs)
    print(f"3. {len(picked)} tasks with a clean pass; conversations: {dict(reasons)}")

    # 4. Convert and check
    a.out.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict]] = {"train": [], "dev": []}
    bad = []
    src = {"file": shown(a.input), "sha256_16": sha(a.input)}
    for (d, t), r in sorted(picked.items(), key=lambda kv: (kv[0][0], splits.task_ids(kv[0][0], "train")
                                                           .index(kv[0][1]) if kv[0][1] in
                                                           splits.task_ids(kv[0][0], "train") else -1)):
        sp, tools = system[d]
        split = splits.split_of(d, t)
        rec = golden.to_record(r, sp, tools, split, source={**src, "attempt": r.get("attempt") or 1})
        problems = golden.check_record(rec, golden.agent_view(r["messages"], sp))
        if problems:
            bad.append((rec["id"], problems))
        rows[split].append(rec)
    if bad:
        for i, ps in bad[:10]:
            print(f"⛔ {i}: {'; '.join(ps)}")
        print(f"⛔ {len(bad)} records failed their checks; nothing written")
        return 1
    print(f"4. all {len(picked)} records converted to Cohere format and read back exactly")

    for split, name in (("train", "golden_train.jsonl"), ("dev", "golden_dev_reference.jsonl")):
        with (a.out / name).open("w") as f:
            for rec in rows[split]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    shutil.copy(splits.SPLITS_PATH, a.out / "splits.json")

    # Checksums of what was just checked: publish_golden.py refuses files that changed since
    files = {n: sha256(a.out / n)
             for n in ("golden_train.jsonl", "golden_dev_reference.jsonl", "splits.json")}
    report = {"input": src, "view_check": checked, "conversations": dict(reasons), "files": files,
              "domains": {}}
    for d in splits.DOMAINS:
        for split in ("train", "dev"):
            ids = splits.task_ids(d, split)
            have = {x["task_id"] for x in rows[split] if x["domain"] == d}
            report["domains"].setdefault(d, {})[split] = {
                "tasks": len(ids), "trajectories": len(have),
                "unsolved": [t for t in ids if t not in have]}
    stats = collections.defaultdict(list)
    for rec in rows["train"]:
        stats["turns"].append(len(rec["messages"]))
        stats["targets"].append(len(rec["train_on"]))
        stats["chars"].append(sum(len(m["content"]) for m in rec["messages"]))
    report["train_sizes"] = {k: {"min": min(v), "median": sorted(v)[len(v) // 2], "max": max(v)}
                             for k, v in stats.items()}
    (a.out / "build_report.json").write_text(json.dumps(report, indent=1) + "\n")

    print("\n          train            dev")
    for d, v in report["domains"].items():
        print(f"{d:8s}  {v['train']['trajectories']:3d} of {v['train']['tasks']:3d}       "
              f"{v['dev']['trajectories']:2d} of {v['dev']['tasks']:2d}")
    print(f"{'total':8s}  {len(rows['train']):3d} of 142       {len(rows['dev']):2d} of 36")
    print(f"\nwrote {shown(a.out)}/: golden_train.jsonl, golden_dev_reference.jsonl, "
          "splits.json, build_report.json")

    if a.sample:
        pick_ = random.Random("reagent-read-v1").sample(rows["train"], a.sample)
        (a.out / "sample_to_read.md").write_text(render(pick_))
        print(f"wrote {shown(a.out)}/sample_to_read.md ({a.sample} records to read)")
    return 0


def render(records: list[dict]) -> str:
    """Records as markdown for a person to read: each turn labelled, targets marked."""
    out = ["# Golden trajectories v1: records to read\n",
           "Every turn as the student will see it. **▶ TARGET** marks the turns it learns to write "
           "(`train_on`). The system prompt is cut to its first and last lines.\n"]
    for rec in records:
        out.append(f"\n---\n\n## {rec['id']}\n\nOutcome: `{json.dumps(rec['outcome']['checks'])}` · "
                   f"teacher {rec['teacher']['model']} via {rec['teacher']['provider']} · "
                   f"{len(rec['messages'])} turns, {len(rec['train_on'])} targets\n")
        for i, m in enumerate(rec["messages"]):
            text = m["content"]
            if m["role"] == "system":
                lines = text.splitlines()
                text = "\n".join(lines[:3] + [f"… ({len(lines)} lines) …"] + lines[-3:])
            tag = " ▶ TARGET" if i in rec["train_on"] else ""
            out.append(f"\n**[{i}] {m['role']}{tag}**\n\n```text\n{text}\n```\n")
    return "".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
