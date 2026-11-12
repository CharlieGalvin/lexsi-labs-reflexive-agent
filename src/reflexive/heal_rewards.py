"""Heal GRPO rewards (SEL-75): one reward per failure type, plus agreement with the teacher's fix.

Every sampled turn is scored by five functions, each 0–1, passed to TRL separately (so W&B
charts each one, ``rewards/<name>/mean``) with the weights from the config:

* ``tool``: **no Tool failure** (Chiara's ``TOOL``): every call names one of the agent's own
  tools. A call to a customer-only tool (telecom's phone tools) or an unknown tool scores 0.
* ``arg``: **no Arg failure** (``ARG``): the share of calls whose arguments have no
  placeholder, no invented (ungrounded) identifier, and fit the schema; a reply with an
  ``Action:`` the parser can't read (``malformed``) scores 0. Uses the classifier's own checks.
* ``loop``: **no Loop failure** (``LOOP``): the turn doesn't repeat a call that already failed,
  doesn't make a call for the third time, doesn't send the same reply a third time, and isn't
  the failed turn again. Unless the teacher's fix does the same.
* ``fix``: **agreement with the teacher's turn**. Where it called: a call (0.4), the same tools
  (0.3) and arguments (0.3). Where it replied: a reply (0.2) and what it says (0.8, the F1 of its
  content words against the teacher's: IDs, names, amounts, not filler), so a generic reply
  ("Could you provide more details?") scores little.
* ``guard``: the call is the whole reply (``format``) and the turn doesn't give up (a stop, a
  hand-off or a few words) unless the teacher did.
* ``decision`` (arm H2, `reflexive.decision_data`): every state-changing call matches an expected
  change still pending at this turn, checked against τ²-bench's expected actions; where the
  teacher made the change, the sample makes it too. Weight 0 by default and 1.0 for rows without
  a ``decision`` column, so H's rewards are unchanged.

**The failure rewards count only for the teacher's kind of turn** (Chiara's review, 11 Nov). A
reply makes no call, so it can't fail ``tool``, ``arg`` or ``loop``; without this rule a reply
would score 3.0 of 4.0 wherever the teacher called a tool, more than any call with one invented
argument, and the model could learn "always ask" as G learned "always call". So a reply where
the teacher called, or a call where it replied, scores 0 on all three (only ``guard`` remains).
`just grpo-heal-reward-check` scores canned replies to show the reward can't be gamed that way.

Every failure check applies to every prompt, whatever its label: a Loop prompt can still be
answered with an invented argument. The label decides which prompts are in the data (the
Tool / Arg / Loop mix), and the weights decide what's optimised (Tool and Arg above Loop).

**Dataset columns** read: ``target``, ``target_calls`` (JSON text), ``domain``, ``id``,
``failure_type``, ``failed`` and ``history`` (JSON text, `reflexive.heal_data`); and TRL's
``prompts``.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from reflexive import grpo_rewards as g
from reflexive.decision_data import decision_score
from reflexive.hallucination import ACTION_TEXT, _arg_findings, domain_tools
from reflexive.heal_data import call_key, norm_text

COMPONENTS = ("tool", "arg", "loop", "fix", "guard", "decision")
# max 4.0; decision is H2's (0 here, so H is unchanged)
DEFAULT_WEIGHTS = {"tool": 1.0, "arg": 1.0, "loop": 0.5, "fix": 1.0, "guard": 0.5, "decision": 0.0}
ARG_KINDS = {"placeholder", "ungrounded", "unknown_param", "missing_param", "wrong_type"}
LOOP_THRESHOLD = 3  # as the classifier's: the third identical call or reply is a loop
REPLY_KIND = 0.2  # a reply where the teacher replied: 0.2 for the kind, 0.8 for what it says
_WORD = re.compile(r"[#\w][\w.@#-]*\w|\w")
STOPWORDS = frozenset(
    "a an and are as at be but by can could do for from have how i if in is it its let me my "
    "no not of on or our please so that the their them then there this to us was we what when "
    "which will with would you your yes thank thanks sure hello hi help "
    # customer-service filler: says nothing about this customer or task
    "details information more provide look into moment anything else today assist understand "
    "happy glad apologize sorry certainly okay ok just one".split()
)


def content_words(text: str) -> set[str]:
    """The words that carry the reply's content: IDs, names, amounts, nouns; not filler."""
    return {w for w in (m.group().casefold() for m in _WORD.finditer(text)) if w not in STOPWORDS}


def word_overlap(text: str, target: str) -> float:
    """F1 of the content words of a reply against the teacher's: 1.0 for the teacher's own reply,
    low for a generic one ("Could you provide more details?") that names nothing it named."""
    a, b = content_words(text), content_words(target)
    if not b:
        return 1.0 if not a else 0.5
    hits = len(a & b)
    if not hits:
        return 0.0
    precision, recall = hits / len(a), hits / len(b)
    return 2 * precision * recall / (precision + recall)


Record = dict[str, Any]


def _json(v: Any, default: Any) -> Any:
    if isinstance(v, str):
        return json.loads(v) if v else default
    return default if v is None else v


TOOLS_BLOCK = "## Available Tools"  # where `reflexive.teacher.COHERE_TOOLS_PROMPT` starts


def seen_text(prompt: Sequence[Record]) -> tuple[str, ...]:
    """What counts as already seen for the grounding check: the conversation, without the tool
    descriptions in the system prompt. Their examples (``something@example.com``) aren't the
    customer's data, and Chiara's classifier doesn't count them either: it reads the
    conversation with the tools given separately. Copying one is an invented argument."""
    out = []
    for m in prompt:
        content = str(m.get("content") or "")
        if m.get("role") == "system":
            content = content.split(TOOLS_BLOCK, 1)[0]
        out.append(content)
    return tuple(out)


def score(
    text: str,
    *,
    target: str,
    target_calls: Sequence[Record],
    domain: str,
    prompt: Sequence[Record],
    failed: Record | None = None,
    history: Record | None = None,
    decision: Record | None = None,
) -> Record:
    """Score one sampled turn. Returns each component (0–1) plus what it found, for the log."""
    failed = failed or {"calls": [], "text": ""}
    history = history or {"calls": [], "texts": []}
    calls, embedded = g.parse_calls(text)
    tools, customer = domain_tools(domain)
    schemas = {t["function"]["name"]: t["function"] for t in tools}
    ctx = g._context(seen_text(prompt))

    # Tool: only the agent's own tools
    wrong_tools = [c["name"] for c in calls if c["name"] not in schemas]
    tool = 0.0 if wrong_tools else 1.0

    # Arg: the classifier's argument checks, per call to a known tool
    arg_kinds: list[str] = []
    known = [(i, c) for i, c in enumerate(calls) if c["name"] in schemas]
    clean = 0
    for i, c in known:
        kinds = {f.kind for f in _arg_findings(0, i, c["name"], c["arguments"], schemas[c["name"]], ctx)}
        kinds &= ARG_KINDS
        arg_kinds += sorted(kinds)
        clean += not kinds
    malformed = not calls and bool(ACTION_TEXT.search(text))
    if malformed:
        arg, arg_kinds = 0.0, ["malformed"]
    else:
        arg = clean / len(known) if known else 1.0

    # Loop: repeats of what the prompt already holds, or of the failed turn
    target_keys = {call_key(c["name"], c["arguments"]) for c in target_calls}
    seen = Counter(c["key"] for c in history["calls"])
    failed_before = {c["key"] for c in history["calls"] if c["failed"]}
    loop_kinds = []
    for c in calls:
        key = call_key(c["name"], c["arguments"])
        if key in target_keys:
            continue  # the teacher made this call here too
        if key in failed_before:
            loop_kinds.append("repeated_failed_call")
        elif seen[key] + 1 >= LOOP_THRESHOLD:
            loop_kinds.append("repeated_call")
        elif key in failed["calls"]:
            loop_kinds.append("failed_turn_again")
    said = norm_text(text)
    if not calls and said:
        if Counter(history["texts"])[said] + 1 >= LOOP_THRESHOLD:
            loop_kinds.append("repeated_message")
        elif failed["text"] and said == failed["text"]:
            loop_kinds.append("failed_turn_again")
    loop = 0.0 if loop_kinds else 1.0

    # Fix: the teacher's choice (reply or call), then its tools and arguments
    if target_calls:
        if calls:
            pairs = g._pair(target_calls, calls)
            args = sum(g.args_score(t["arguments"], s["arguments"]) if s else 0.0 for t, s in pairs) / len(
                pairs
            )
            fix = 0.4 + 0.3 * g._overlap(target_calls, calls) + 0.3 * args
        else:
            fix = 0.0
    else:  # the teacher replied: a reply, scored on what it says
        fix = 0.0 if calls else REPLY_KIND + (1 - REPLY_KIND) * word_overlap(text, target)

    # Guard: the call is the whole reply; no giving up unless the teacher did
    teacher_gave_up = g._gives_up(target, target_calls)
    gives_up = g._gives_up(text, calls)
    guard = 0.5 * (0.0 if calls and embedded else 1.0) + 0.5 * (
        1.0 if teacher_gave_up or not gives_up else 0.0
    )

    # The failure rewards count only for the teacher's kind of turn (Chiara's review): otherwise a
    # reply would score full marks on all three wherever the teacher called, and beat any call
    # with one argument error.
    # Decision: the state changes against the task's expected actions still pending (H2)
    decision_value, decision_found = decision_score(calls, target_calls, decision)

    kind_match = bool(calls) == bool(target_calls)
    if not kind_match:
        tool = arg = loop = 0.0

    return {
        "tool": tool,
        "arg": arg,
        "loop": loop,
        "fix": round(fix, 4),
        "guard": guard,
        "decision": decision_value,
        "decision_found": decision_found,
        "kind": "call" if calls else "text",
        "kind_match": kind_match,
        "tools": [c["name"] for c in calls],
        "wrong_tools": wrong_tools,
        "customer_tools": [n for n in wrong_tools if n in set(customer)],
        "arg_kinds": arg_kinds,
        "loop_kinds": loop_kinds,
    }


def total(s: Record, weights: dict[str, float]) -> float:
    return sum(w * s[k] for k, w in weights.items())


def score_row(text: str, row: Record) -> Record:
    """`score` from a dataset row (`reflexive.heal_data` columns, JSON or parsed)."""
    return score(
        text,
        target=row.get("target") or "",
        target_calls=_json(row.get("target_calls"), []),
        domain=row["domain"],
        prompt=row.get("prompt") or [],
        failed=_json(row.get("failed"), None),
        history=_json(row.get("history"), None),
        decision=_json(row.get("decision"), None),
    )


class RewardFns:
    """The five rewards as TRL reward functions, scoring each batch once.

    TRL calls every reward function on the same completions in turn; the first call scores the
    batch (and logs every sample to ``log_path``), the others reuse it. ``fns`` is in
    `COMPONENTS` order: pass ``[weights[c] for c in COMPONENTS]`` as GRPOConfig's
    ``reward_weights``."""

    def __init__(self, log_path: str | Path | None = None, weights: dict[str, float] | None = None) -> None:
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        if bad := sorted(set(self.weights) - set(COMPONENTS)):
            raise ValueError(f"unknown reward component(s) {bad}: expected some of {COMPONENTS}")
        self.log = Path(log_path) if log_path else None
        if self.log and self.log.exists() and self.log.stat().st_size:
            self.log.rename(
                self.log.with_name(f"{self.log.stem}.stopped-{int(time.time())}{self.log.suffix}")
            )
        self._lock = threading.Lock()
        self._key: tuple | None = None
        self._scores: list[Record] = []
        self.calls = 0
        self.fns = [self._fn(c) for c in COMPONENTS]

    def _batch(self, completions: list[Any], kwargs: dict[str, Any]) -> list[Record]:
        texts = [g.completion_text(c) for c in completions]
        key = (tuple(texts), tuple(kwargs.get("id") or ()))
        with self._lock:
            if key == self._key:
                return self._scores
        prompts = kwargs.get("prompts")
        n = len(texts)
        scores, rows = [], []
        for i in range(n):
            row = {
                k: kwargs[k][i]
                for k in ("target", "target_calls", "domain", "failed", "history", "decision")
                if k in kwargs
            }
            row["prompt"] = prompts[i] if prompts is not None and isinstance(prompts[i], list) else []
            s = score_row(texts[i], row)
            scores.append(s)
            rows.append(
                {
                    "step": getattr(kwargs.get("trainer_state"), "global_step", None),
                    "call": self.calls + 1,
                    "id": (kwargs.get("id") or [None] * n)[i],
                    "failure_type": (kwargs.get("failure_type") or [None] * n)[i],
                    "reward": total(s, self.weights),
                    **s,
                    "completion": texts[i][:2000],
                }
            )
        with self._lock:
            self.calls += 1
            self._key, self._scores = key, scores
            if self.log:
                with self.log.open("a") as f:
                    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        if callable(log_metric := kwargs.get("log_metric")):  # TRL ≥ 1.x: extra charts
            teacher_replies = sum(not _json(c, []) for c in kwargs["target_calls"]) / n
            replies = sum(s["kind"] == "text" for s in scores) / n
            log_metric("heal/kind_match", sum(s["kind_match"] for s in scores) / n)
            log_metric("heal/reply_share", replies)
            log_metric("heal/reply_excess", replies - teacher_replies)  # the drift guard watches this
        return scores

    def _fn(self, component: str) -> Callable[..., list[float]]:
        def reward(completions: list[Any], **kwargs: Any) -> list[float]:
            if "target_calls" not in kwargs or "domain" not in kwargs:
                raise ValueError("the dataset needs the heal columns: build it with `just grpo-heal-data`")
            return [float(s[component]) for s in self._batch(completions, kwargs)]

        reward.__name__ = component  # TRL names the chart after it: rewards/<component>/mean
        return reward
