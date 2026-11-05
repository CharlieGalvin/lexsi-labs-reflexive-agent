"""Save τ²-bench's tool schemas per domain to configs/tool_schemas.json (SEL-12).

    python scripts/tau2/dump_tool_schemas.py           # rewrite the file (needs τ²-bench installed)
    python scripts/tau2/dump_tool_schemas.py --check   # confirm the file matches τ²-bench

The hallucination classifier (``reflexive.hallucination``) reads this file, so it runs without
τ²-bench. Per domain: the agent's tools as OpenAI function schemas (without τ²-bench's internal
``done``), and in telecom the names of the customer's own phone tools, which the agent can't call.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from reflexive import splits  # noqa: E402
from reflexive.hallucination import SCHEMAS_PATH  # noqa: E402


def build() -> dict:
    from loguru import logger

    from reflexive.episode import _NoScoreAgentGymEnv

    logger.remove()
    out = {"source": "tau2-bench v2.0.0", "domains": {}}
    for domain in splits.DOMAINS:
        task = splits.task_ids(domain, "train")[0]
        env = _NoScoreAgentGymEnv(
            domain=domain, task_id=task, max_steps=200, user_llm="gpt-4.1", user_llm_args={}, language="en"
        )
        orch = env._get_orchestrator()
        out["domains"][domain] = {
            "agent_tools": [t.openai_schema for t in orch.agent.tools if t.name != "done"],
            "customer_tools": sorted(t.name for t in (getattr(orch.user, "tools", None) or [])),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="compare with τ²-bench; write nothing")
    a = p.parse_args()
    text = json.dumps(build(), indent=1) + "\n"
    if a.check:
        same = SCHEMAS_PATH.exists() and SCHEMAS_PATH.read_text() == text
        print("tool schemas match τ²-bench" if same else "⛔ configs/tool_schemas.json is out of date")
        return 0 if same else 1
    SCHEMAS_PATH.write_text(text)
    print(f"wrote {SCHEMAS_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
