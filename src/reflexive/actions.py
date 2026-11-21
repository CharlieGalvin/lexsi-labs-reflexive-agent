"""``check_action``: the one function the checker and the guard both code against (SEL-83).

    finding = check_action(conversation, call, policy)   # before the call runs
    if finding is not None and finding.live: ...          # the guard blocks it (SEL-89)

The **checker** (SEL-88's "no yes" rule, and meta-healing rules later) supplies rules; the
**guard** (`reflexive.monitor.ActionGuard`) calls `check_action` on every state-changing call the
agent proposes, before τ²-bench runs it. Neither side knows the other's internals, so a new rule
plugs in by joining `RULES`, with no change to the guard.

A rule reads only what a live deployment has: the conversation as the agent saw it, the proposed
call and the domain policy. It never reads τ²-bench's expected actions or the task.

**Live or after the fact.** Every rule says which it is, in the `ActionFinding` it returns:

* ``live=True``: checkable at the moment of the change, and fast. The guard may block on it.
* ``live=False``: needs more than the moment of the change (later turns, the outcome), or is too
  slow to run on every call. Used by the gate and in reports only; the guard logs it, never blocks.

This is the agreed interface. ``RULES`` holds SEL-88's ``no_yes`` (`reflexive.intent`), the hand-written
``mismatch`` (`reflexive.mismatch`), then the
meta-healing rules (SEL-91, `reflexive.meta_heal`) a person accepted into ``configs/meta_rules.json``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

Message = dict[str, Any]


@dataclass(frozen=True)
class ActionFinding:
    """Why a proposed call shouldn't run (or, when ``live`` is False, should be looked at later)."""

    type: str  # the finding type, e.g. "INTENT" (made-up intent), beside TOOL / ARG / LOOP
    rule: str  # the rule that fired, e.g. "no_yes"
    reason: str  # one sentence a person can read in the log
    live: bool  # True: checkable at the moment of the change, so the guard may block on it
    message: str = ""  # what the guard tells the agent when it blocks; "" means its BLOCK_MESSAGE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


Rule = Callable[[Sequence[Message], dict, str], "ActionFinding | None"]
"""``rule(conversation, call, policy) -> ActionFinding | None``, the same arguments as `check_action`."""

RULES: list[Rule] = []
"""The rules `check_action` runs, in order: SEL-88's ``no_yes``, registered at the end of this module."""


def check_action(
    conversation: Sequence[Message],
    call: dict,
    policy: str,
    rules: Sequence[Rule] | None = None,
) -> ActionFinding | None:
    """The first finding for ``call``, live findings before after-the-fact ones; None if it may run.

    ``conversation``: the agent's view so far (OpenAI-style messages, system prompt first), not
    including the turn that proposes ``call``.
    ``call``: one OpenAI tool call, ``{"id", "type": "function", "function": {"name", "arguments"}}``,
    with ``arguments`` a JSON string or a dict.
    ``policy``: the domain policy text, as the agent's system prompt carries it.
    ``rules``: the rules to run; `RULES` when omitted.
    """
    later: ActionFinding | None = None
    for rule in RULES if rules is None else rules:
        finding = rule(conversation, call, policy)
        if finding is None:
            continue
        if finding.live:
            return finding
        later = later or finding
    return later


# The rules, registered last because reflexive.intent builds its findings from this module.
from reflexive.intent import no_yes  # noqa: E402

RULES.append(no_yes)

# Hand-written after meta-healing: a change that doesn't match the request or the lookups.
from reflexive.mismatch import mismatch  # noqa: E402

RULES.append(mismatch)

# SEL-91: the meta-healing rules a person accepted (configs/meta_rules.json), after "no yes".
from reflexive.meta_heal import meta_rules  # noqa: E402

RULES.extend(meta_rules())
