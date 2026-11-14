"""Golden trajectories v1 (SEL-9): teacher passes → training records in the data-contract format.

The contract (agreed 4 Nov, https://claude.ai/artifact/Ew1amyTBrTHbdiW92DQzNw):

1. **Format:** Cohere's ``Action:`` format, built by the same `teacher.to_prompted` that gave
   Aya its baseline prompt, so training and evaluation prompts are identical.
2. **Input:** the conversation **as the agent saw it**: the system prompt, then everything
   except the customer's own phone actions (telecom) and their results.
3. **Learning target:** every assistant turn (``train_on``).
4. **What goes in:** one clean pass per solved task, the first by attempt number; dev tasks
   go to a separate reference file; test tasks never appear.

Every record is checked by `check_record` before it's written: the Cohere text is parsed back
and must give exactly the original agent turns (same text, same tool calls and arguments,
same tool results), the roles must alternate and ``train_on`` must be every assistant turn.

Pure functions; τ²-bench is only needed to make the system prompt (`scripts/data/build_golden.py`).
"""

from __future__ import annotations

import json
import re

from reflexive.episode import compact_json
from reflexive.teacher import parse_prompted, to_prompted

FORMAT = "reagent.golden/v1"
TAU2_VERSION = "2.0.0"
TEACHER_MODELS = {"glm53f": "z-ai/glm-5.3-flash"}
GOOD_ENDINGS = ("agent_stop", "user_stop")
_RESULT = re.compile(r"<results>\nDocument: (\d+)\ntool_name: (.*?)\nresult: (.*?)\n</results>", re.DOTALL)


# --- 2. The agent's view ----------------------------------------------------------------------


def agent_view(messages: list[dict], system_prompt: str) -> list[dict]:
    """The saved full conversation reduced to what the agent saw, system prompt first.

    Drops the customer's tool calls (a ``user`` turn with ``tool_calls``) and the results of
    those calls; compacts JSON tool results the way `Tau2Episode` showed them. Checked to give
    exactly `Tau2Episode.messages()` on the 30 Aya conversations that saved both.
    """
    customer_calls = {tc["id"] for m in messages if m["role"] == "user" for tc in m.get("tool_calls") or []}
    view = [{"role": "system", "content": system_prompt}]
    for m in messages:
        if m["role"] == "user" and m.get("tool_calls"):
            continue
        if m["role"] == "tool":
            if m.get("tool_call_id") in customer_calls:
                continue
            m = {**m, "content": compact_json(m["content"])}
        view.append(m)
    return view


# --- 4. What goes in --------------------------------------------------------------------------


def tool_problems(view: list[dict]) -> dict:
    """Retried failed calls and calls to tools the agent doesn't have, recounted on the view."""
    from reflexive.teacher import tool_errors

    return tool_errors(view)


def why_not_clean(rec: dict) -> str | None:
    """None for a clean pass; otherwise the first reason it isn't one."""
    if rec.get("termination") == "monitor_stop":  # the harness ended it, not the agent (SEL-69)
        return "ended by the runtime monitor's guard"
    if rec.get("error") or rec.get("termination") not in GOOD_ENDINGS:
        return f"run broke ({rec.get('termination')})"
    if not rec["outcome"]["passed"]:
        return "failed a check"
    if rec.get("judge_error") or rec["outcome"].get("skipped"):
        return "not fully judged"
    probs = tool_problems(rec["messages"])
    if probs["repeated_failures"]:
        return "retried a failed call"
    if probs["unknown_tools"]:
        return "called an unknown tool"
    return None


def pick(records: list[dict]) -> dict[tuple[str, str], dict]:
    """The first clean pass per (domain, task), by attempt number."""
    best: dict[tuple[str, str], dict] = {}
    for r in sorted(records, key=lambda r: r.get("attempt") or 1):
        key = (r["domain"], str(r["task_id"]))
        if key not in best and why_not_clean(r) is None:
            best[key] = r
    return best


# --- 1 + 3. Cohere format and learning targets -----------------------------------------------


def to_record(
    rec: dict, system_prompt: str, tools: list[dict], split: str, source: dict | None = None
) -> dict:
    """One training record (see the contract's field table)."""
    if split not in ("train", "dev"):
        raise ValueError(f"golden records are train or dev, never {split!r}")
    view = agent_view(rec["messages"], system_prompt)
    msgs = to_prompted(view, tools, "cohere")
    teacher = rec.get("sample_model", "teacher")
    attempt = rec.get("attempt") or 1
    providers = (rec.get("agent_usage") or {}).get("providers") or {}
    return {
        "id": f"{teacher}/{rec['domain']}/{rec['task_id']}/{attempt}",
        "format": FORMAT,
        "domain": rec["domain"],
        "task_id": str(rec["task_id"]),
        "split": split,
        "messages": msgs,
        "train_on": [i for i, m in enumerate(msgs) if m["role"] == "assistant"],
        "tools": tools,
        "outcome": {"reward": rec["outcome"]["reward"], "checks": rec["outcome"]["checks"]},
        "teacher": {
            "model": TEACHER_MODELS.get(teacher, rec.get("agent_llm")),
            "provider": max(providers, key=lambda p: providers[p]) if providers else None,
            "temperature": rec.get("temperature"),
            "attempt": attempt,
        },
        "customer": rec.get("user_llm"),
        "tau2_version": TAU2_VERSION,
        "source": source or {},
    }


def from_cohere(msgs: list[dict]) -> list[dict]:
    """Read a Cohere-format conversation back into agent turns, for the round-trip check.

    Returns ``[{"role", "content" | "calls" | "results"}]`` with the system turn dropped:
    assistant turns as text or a list of (tool, arguments); result turns as a list of
    (tool, result text).
    """
    out = []
    for m in msgs[1:]:
        text = m["content"]
        if m["role"] == "assistant" and text.startswith("Action: ```json"):
            calls = parse_prompted(text) or []
            out.append(
                {
                    "role": "assistant",
                    "calls": [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls],
                }
            )
        elif m["role"] == "user" and text.startswith("<results>"):
            found = _RESULT.findall(text)
            rebuilt = "\n\n".join(
                f"<results>\nDocument: {i}\ntool_name: {n}\nresult: {r}\n</results>" for i, n, r in found
            )
            if rebuilt != text:  # something besides results was merged into this turn
                out.append({"role": "user", "content": text})
            else:
                out.append({"role": "user", "results": [(n, r) for _, n, r in found]})
        else:
            out.append({"role": m["role"], "content": text})
    return out


def _expected(view: list[dict]) -> list[dict]:
    """The agent's view in `from_cohere`'s shape: what the round trip must give back."""
    names, out = {}, []
    for m in view[1:]:
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", tc)
                args = fn.get("arguments") or {}
                args = json.loads(args) if isinstance(args, str) and args.strip() else (args or {})
                names[tc.get("id")] = fn["name"]
                calls.append((fn["name"], args))
            out.append({"role": "assistant", "calls": calls})
        elif m["role"] == "tool":
            item = (names.get(m.get("tool_call_id"), "tool"), m["content"])
            if out and "results" in out[-1]:
                out[-1]["results"].append(item)
            else:
                out.append({"role": "user", "results": [item]})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def check_record(golden: dict, view: list[dict]) -> list[str]:
    """Everything wrong with a training record, compared with the agent's view it came from."""
    msgs, problems = golden["messages"], []
    if (
        not msgs
        or msgs[0]["role"] != "system"
        or msgs[0]["content"] != (to_prompted(view[:1], golden["tools"], "cohere")[0]["content"])
    ):
        problems.append("system prompt isn't the baseline prompt")
    roles = [m["role"] for m in msgs[1:]]
    if any(a == b for a, b in zip(roles, roles[1:], strict=False)) or set(roles) - {"user", "assistant"}:
        problems.append("roles don't alternate user/assistant")
    if golden["train_on"] != [i for i, m in enumerate(msgs) if m["role"] == "assistant"]:
        problems.append("train_on isn't every assistant turn")
    if not golden["train_on"]:
        problems.append("no assistant turns")
    back, want = from_cohere(msgs), _expected(view)
    if back != want:
        pairs = enumerate(zip(back, want, strict=False))
        n = next((i for i, (a, b) in pairs if a != b), min(len(back), len(want)))
        problems.append(f"round trip differs at turn {n + 1}: {len(back)} turns back, {len(want)} expected")
    if golden["outcome"]["reward"] != 1.0:
        problems.append("reward isn't 1.0")
    return problems
