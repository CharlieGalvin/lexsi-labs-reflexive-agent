"""Detect & Classify (SEL-12): label how an agent's conversation went wrong as Tool, Arg or Loop.

``classify_hallucination`` reads a conversation **as the agent saw it** (an episode record's
``agent_view``, OpenAI message format) and finds every structural failure in it:

* ``TOOL``: a call to a tool the agent doesn't have. In telecom most of these are the
  **customer's** phone tools (``check_apn_settings``, ``can_send_mms`` …): the agent should
  ask the customer to run them, not call them itself (kind ``customer_tool``). Anything else is
  ``unknown_tool``.
* ``ARG``: the right tool, wrong arguments.
    - ``placeholder``: a value that describes what should go there instead of being it, e.g.
      ``"customer_id from previous tool result"``, ``"<line_id>"``, ``"YYYY-MM-DD"``.
    - ``ungrounded``: an identifier, name, phone number, email or date of birth that appears
      nowhere in the conversation **before** the call: invented.
    - ``unknown_param`` / ``missing_param`` / ``wrong_type``: the call doesn't fit the schema.
    - ``malformed``: arguments that aren't JSON, or an ``Action:`` block the parser couldn't read.
* ``LOOP``: going round in circles.
    - ``repeated_failed_call``: the same call (tool and arguments) as one that already failed.
    - ``repeated_call``: the same call ``loop_threshold`` times, even if it worked.
    - ``repeated_message``: the same reply to the customer ``loop_threshold`` times.
    - ``step_limit``: τ²-bench stopped the conversation at its step limit.
    Neither of the last two counts after a successful ``transfer_to_human_agents``: the agent's
    job is over, and a conversation that runs on is the customer simulator's doing.

The **primary** failure is the earliest finding, because later ones usually follow from it
(an invented ID gets an error, the agent retries it, and the retry is a loop). Ties within
one turn go TOOL, then ARG, then LOOP. ``NONE`` means no structural failure: a conversation can
still fail τ²-bench for reasoning or policy reasons, which this classifier doesn't judge.

It is deterministic and cheap, so it runs on every conversation. It needs no τ²-bench install:
the tool schemas come from ``configs/tool_schemas.json`` (``scripts/dump_tool_schemas.py``).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMAS_PATH = Path(__file__).resolve().parents[2] / "configs" / "tool_schemas.json"


class FailureType(StrEnum):
    TOOL = "TOOL"
    ARG = "ARG"
    LOOP = "LOOP"
    NONE = "NONE"


_ORDER = {FailureType.TOOL: 0, FailureType.ARG: 1, FailureType.LOOP: 2}

# Our labels in AgentTune's closed-loop FailureClassifier taxonomy, for SEL-13.
TO_AGENTTUNE = {
    FailureType.TOOL: "wrong_tool",
    FailureType.ARG: "hallucinated_output",
    FailureType.LOOP: "loop_collapse",
}


@dataclass(frozen=True)
class Finding:
    type: FailureType
    kind: str
    turn: int  # index in the messages of the assistant turn it's in
    tool: str | None = None
    detail: str = ""
    call: int = 0  # position of the call within that turn

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "type": self.type.value}


@dataclass
class Diagnosis:
    failure: FailureType
    findings: list[Finding] = field(default_factory=list)

    @property
    def first(self) -> Finding | None:
        return self.findings[0] if self.findings else None

    @property
    def kinds(self) -> Counter:
        return Counter(f"{f.type.value}/{f.kind}" for f in self.findings)

    @property
    def types(self) -> set[FailureType]:
        return {f.type for f in self.findings}

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure": self.failure.value,
            "first": self.first.to_dict() if self.first else None,
            "kinds": dict(self.kinds),
            "findings": [f.to_dict() for f in self.findings],
        }


# --- Tool schemas --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _schemas() -> dict:
    return json.loads(SCHEMAS_PATH.read_text())["domains"]


def domain_tools(domain: str) -> tuple[list[dict], list[str]]:
    """The agent's tools (OpenAI function schemas) and the customer's tool names, for a domain."""
    d = _schemas()[domain]
    return d["agent_tools"], d["customer_tools"]


# --- Argument checks -----------------------------------------------------------------------

# Arguments the agent must copy from the conversation, never make up.
_GROUNDED_KEY = re.compile(r"(?:^|_)ids?$|phone|email|^zip$|^dob$|(?:^|_)name$|flight_number")
_ID_KEY = re.compile(r"(?:^|_)ids?$|phone|email|flight_number")
# Not checked: free text, and values the agent may legitimately work out (airport codes, dates).
_FREE_TEXT = {"summary", "reason", "expression", "app_name"}
_PLACEHOLDER = re.compile(
    r"[<>{}]|\b(?:from (?:the )?(?:previous|prior|above|last|earlier|search|results?|lookup|tool|"
    r"customer|data|response|output)|not provided|placeholder|insert|replace|"
    r"goes here|your[_ ]|customer'?s|user'?s|the customer|the user|unknown|"
    r"e\.g\.|for example|tbd|n/a|x{3,})\b"
    r"|_here\b|\byyyy\b|\bmm-dd\b|^(?:id|line[_ ]id|customer[_ ]id|user[_ ]id|order[_ ]id|"
    r"reservation[_ ]id|phone[_ ]number)$",
    re.I,
)
_SQUASHED = re.compile(  # the same phrases with the spaces taken out: CustomerIDfromPreviousToolResult
    r"from(?:the)?(?:previous|prior|above|last|earlier|search|result|lookup|tool|customer|data|response|output)"
    r"|placeholder|notprovided|goeshere|your(?:phone|customer|line|user|order|reservation|id|email)"
)
_UNFILLED = re.compile(r"\d{2}xx\b|\bxx\d|^x+$", re.I)  # 19XX-05-14, XXX


def _is_placeholder(key: str, value: str) -> bool:
    """A description of the value instead of the value: also with underscores for spaces."""
    v = value.strip()
    words = re.sub(r"(?<=[a-z])(?=[A-Z])|_", " ", v)  # customer_id_from_search, YourPhoneNumber
    return bool(
        _PLACEHOLDER.search(v)
        or _PLACEHOLDER.search(words)
        or _SQUASHED.search(re.sub(r"[\W_]", "", v).lower())
        or _UNFILLED.search(v)
        or (_ID_KEY.search(key) and " " in v)
    )


_JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "array": list,
    "boolean": bool,
    "object": dict,
}
_MONTHS = {
    m: i
    for i, m in enumerate(
        "january february march april may june july august september october november december".split(), 1
    )
}
_MONTH_RE = "|".join(_MONTHS) + "|" + "|".join(m[:3] for m in _MONTHS)


def _values(args: Any, key: str = "") -> list[tuple[str, str]]:
    """(key, string value) for every string in the arguments, nested ones under their own key."""
    if isinstance(args, dict):
        return [kv for k, v in args.items() for kv in _values(v, k)]
    if isinstance(args, list):
        return [kv for v in args for kv in _values(v, key)]
    return [(key, args)] if isinstance(args, str) else []


def _month(name: str) -> int:
    return next(i for m, i in _MONTHS.items() if m.startswith(name.lower()[:3]))


def _iso_dates(text: str) -> list[str]:
    """Dates written out in the text, as YYYY-MM-DD, so a dob given as 'April 17, 1985' grounds."""
    out = [
        f"{y}-{_month(m):02d}-{int(d):02d}"
        for m, d, y in re.findall(
            rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})", text, re.I
        )
    ]
    out += [
        f"{y}-{_month(m):02d}-{int(d):02d}"
        for d, m, y in re.findall(
            rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_RE})\.?,?\s+(\d{{4}})", text, re.I
        )
    ]
    out += [
        f"{_month(m):02d}-{int(d):02d}"
        for m, d in re.findall(  # no year given: "March 14th"
            rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", text, re.I
        )
    ]
    for a, b, y in re.findall(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", text):
        out += [f"{y}-{int(a):02d}-{int(b):02d}", f"{y}-{int(b):02d}-{int(a):02d}"]
    return out


class _Context:
    """What the agent had been told before a call: system prompt, customer turns, tool results."""

    def __init__(self) -> None:
        self.text = ""
        self.numbers: set[str] = set()  # each number in the text, separators removed
        self.dates: set[str] = set()

    def add(self, text: str) -> None:
        self.text += " " + text.lower()
        joined = re.sub(r"(?<=\d)[ ().+-]{1,3}(?=\d)", "", text)  # (555) 123-4567 -> 5551234567
        self.numbers.update(re.findall(r"\d{7,}", joined))
        self.dates.update(_iso_dates(text))

    def has(self, key: str, value: str) -> bool:
        v = value.strip().lower()
        if not v or v in self.text or v.lstrip("#") in self.text:
            return True
        if key == "dob" and v in self.dates:
            return True
        digits = re.sub(r"\D", "", v)
        if (
            key != "dob"
            and len(digits) >= 7
            and len(digits) >= len(v) - 4
            and any(digits in n for n in self.numbers)
        ):
            return True  # a phone number written differently: (555) 123-4567 vs 5551234567
        if key.endswith("name"):
            return all(w in self.text for w in v.split())  # "Smith John" or extra spaces
        return False


def _type_ok(value: Any, want: str | None) -> bool:
    if want is None or want not in _JSON_TYPES:
        return True
    if want in ("integer", "number") and isinstance(value, str):
        return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", value.strip()))  # τ²-bench coerces "5"
    if want == "integer" and isinstance(value, float):
        return value.is_integer()
    if isinstance(value, bool) and want != "boolean":
        return False
    return isinstance(value, _JSON_TYPES[want])


def _arg_findings(turn: int, pos: int, name: str, args: dict, schema: dict, ctx: _Context) -> list[Finding]:
    params = schema.get("parameters") or {}
    props, required = params.get("properties") or {}, set(params.get("required") or [])
    out = []

    def add(kind: str, detail: str) -> None:
        out.append(Finding(FailureType.ARG, kind, turn, name, detail, pos))

    if extra := sorted(set(args) - set(props)):
        add("unknown_param", f"no parameter {', '.join(extra)}")
    if missing := sorted(required - set(args)):
        add("missing_param", f"missing {', '.join(missing)}")
    for k, v in args.items():
        if k in props and not _type_ok(v, props[k].get("type")):
            add("wrong_type", f"{k}={v!r} should be {props[k].get('type')}")
    for k, v in _values(args):
        if k in _FREE_TEXT or not _GROUNDED_KEY.search(k):
            continue
        if _is_placeholder(k, v):
            add("placeholder", f"{k}={v!r}")
        elif not ctx.has(k, v):
            add("ungrounded", f"{k}={v!r} never appeared before this call")
    return out


# --- The classifier ------------------------------------------------------------------------

_ACTION_TEXT = re.compile(r"(?:^|\n)\s*Action:\s*(?:```|\[)", re.I)


def _calls(msg: dict) -> list[tuple[str, Any, str | None]]:
    out = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", tc)
        out.append((fn.get("name") or "", fn.get("arguments"), tc.get("id")))
    return out


def _parse_args(raw: Any) -> dict | None:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
        except ValueError:
            return None
        return v if isinstance(v, dict) else None
    return None


def classify_hallucination(
    messages: list[dict],
    tools: list[dict],
    *,
    customer_tools: list[str] | set[str] = (),
    termination: str | None = None,
    loop_threshold: int = 3,
) -> Diagnosis:
    """Every Tool / Arg / Loop failure in a conversation, and the primary (earliest) one.

    ``messages`` is the conversation as the agent saw it (OpenAI format, system prompt first),
    ``tools`` the agent's tools as OpenAI function schemas, ``customer_tools`` the names of tools
    only the customer can use, and ``termination`` τ²-bench's reason for ending it.
    """
    schemas = {s.get("function", s)["name"]: s.get("function", s) for s in tools}
    customer = set(customer_tools)
    results = {m.get("tool_call_id"): m.get("content") or "" for m in messages if m.get("role") == "tool"}
    ctx = _Context()
    findings: list[Finding] = []
    failed: set[str] = set()
    call_counts: Counter = Counter()
    said: Counter = Counter()
    handed_off = False  # after a successful hand-off the agent has no more work: don't count its chat

    for i, m in enumerate(messages):
        role = m.get("role")
        if role != "assistant":
            ctx.add(str(m.get("content") or ""))
            continue
        text = str(m.get("content") or "")
        calls = _calls(m)
        if not calls and _ACTION_TEXT.search(text):
            findings.append(
                Finding(FailureType.ARG, "malformed", i, None, "an Action: block the parser couldn't read")
            )
        if not calls and len(text.strip()) > 20 and not handed_off:
            said[" ".join(text.split())] += 1
            if said[" ".join(text.split())] == loop_threshold:
                findings.append(
                    Finding(
                        FailureType.LOOP,
                        "repeated_message",
                        i,
                        None,
                        f"the same reply {loop_threshold} times: {text.strip()[:80]!r}",
                    )
                )
        for pos, (name, raw, cid) in enumerate(calls):
            args = _parse_args(raw)
            key = f"{name}:{json.dumps(args if args is not None else raw, sort_keys=True)}"
            if name not in schemas:
                kind = "customer_tool" if name in customer else "unknown_tool"
                findings.append(
                    Finding(FailureType.TOOL, kind, i, name, f"the agent has no tool {name!r}", pos)
                )
            elif args is None:
                findings.append(
                    Finding(
                        FailureType.ARG,
                        "malformed",
                        i,
                        name,
                        f"arguments aren't a JSON object: {str(raw)[:80]!r}",
                        pos,
                    )
                )
            else:
                findings += _arg_findings(i, pos, name, args, schemas[name], ctx)
            if key in failed:
                findings.append(
                    Finding(
                        FailureType.LOOP, "repeated_failed_call", i, name, "the same call already failed", pos
                    )
                )
            call_counts[key] += 1
            if call_counts[key] == loop_threshold:
                findings.append(
                    Finding(
                        FailureType.LOOP,
                        "repeated_call",
                        i,
                        name,
                        f"the same call {loop_threshold} times",
                        pos,
                    )
                )
            if results.get(cid, "").startswith("Error"):
                failed.add(key)
            elif name == "transfer_to_human_agents" and cid in results:
                handed_off = True

    if termination == "max_steps" and not handed_off:
        last = max(
            (i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=len(messages) - 1
        )
        findings.append(
            Finding(FailureType.LOOP, "step_limit", last, None, "stopped at τ²-bench's step limit", 99)
        )

    findings.sort(key=lambda f: (f.turn, f.call, _ORDER[f.type]))
    return Diagnosis(findings[0].type if findings else FailureType.NONE, findings)


def agent_view_of(rec: dict) -> list[dict]:
    """The conversation as the agent saw it: the saved ``agent_view``, or one rebuilt from ``messages``.

    Older records (e.g. the teacher runs) have no ``agent_view``. Their ``messages`` include the
    customer's own phone actions and the results, which the agent never saw; counting them as
    "seen" would hide invented values. So those are dropped, as `reflexive.golden.agent_view` does.
    """
    if rec.get("agent_view"):
        return rec["agent_view"]
    messages = rec["messages"]
    customer_calls = {
        tc.get("id") for m in messages if m.get("role") == "user" for tc in m.get("tool_calls") or []
    }
    return [
        m
        for m in messages
        if not (m.get("role") == "user" and m.get("tool_calls"))
        and not (m.get("role") == "tool" and m.get("tool_call_id") in customer_calls)
    ]


def classify_record(rec: dict, **kw: Any) -> Diagnosis:
    """Classify a saved episode record (``reagent.tau2.episode``): its agent's view, domain and ending."""
    tools, customer = domain_tools(rec["domain"])
    return classify_hallucination(
        agent_view_of(rec), tools, customer_tools=customer, termination=rec.get("termination"), **kw
    )
