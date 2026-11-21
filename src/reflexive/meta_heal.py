"""SEL-91, meta-healing: the checker proposes its own rules from the failures it can't yet explain.

    propose   an LLM (GPT-4.1, pinned) reads v2's train-task failures that no current rule explains,
              plus some passes, and writes at most 10 candidate rules, each a fixed judge prompt
    score     each candidate is run, by a second pinned model, on the held-out third of the tasks
    accept    STAGE2.md's bar: flags at least 2, and at least 20%, of the held-out unexplained
              failures, and at most 1 held-out pass; at most 3 accepted, after a person reads 5
              conversations each one flags (``scripts/heal/meta_heal.py review``)

Accepted rules are saved in ``configs/meta_rules.json`` and join `reflexive.actions.RULES`.

**What each side reads.** The *proposer* is the developer here: on the propose two thirds it sees
each conversation and which of τ²-bench's checks failed, as we did when we wrote the "no yes" rule
by hand. A *rule* reads only what a live deployment has: the conversation (tool results included),
the proposed call and the domain policy. Never the task, the expected actions or the outcome
(tests/test_meta_heal.py checks this). Held-out conversations are never shown to the proposer.

**Live or after the fact.** A ``call`` rule is checked at the moment of a state-changing call, so
the guard may block on it if the proposer marked it live; it then re-asks with the rule's own
``block_message``. A ``conversation`` rule reads the whole conversation (an action the agent never
took, a wrong closing statement), so it is after the fact: the gate and reports use it, the guard
never blocks on it.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from reflexive.intent import ALL_WRITE_TOOLS, WRITE_TOOLS

Message = dict[str, Any]

PROPOSER_MODEL = "gpt-4.1-2025-04-14"
RULE_MODEL = "gpt-4.1-mini-2025-04-14"  # the same pinned model as the "no yes" judge
MAX_CANDIDATES = 10
MAX_ACCEPTED = 3
REVIEW_EXAMPLES = 5
MIN_FLAGGED, MIN_SHARE, MAX_PASSES = 2, 0.20, 1  # STAGE2.md's acceptance bar, on the held-out third
HOLDOUT_SHARE = 1 / 3
MAX_CHARS = 1200  # per message or tool result in a transcript
MAX_POLICY_CHARS = 30_000
ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "configs" / "meta_rules.json"

EXISTING_RULES = """\
- TOOL: the agent calls a tool that doesn't exist, or one only the customer can use.
- ARG: a call argument is a placeholder, never appeared in the conversation (made up), is an
  unknown or missing parameter, has the wrong type, or the arguments are malformed.
- LOOP: the same failing call repeated, the same call or message 3 times, or the step limit hit.
- no_yes: a state-changing call made before the customer explicitly agreed to that change."""

PROPOSER_SYSTEM = f"""You improve an automatic checker for a customer-service AI agent.

The agent serves customers of an airline, a retail shop or a phone company (τ²-bench), following
the domain policy, by calling tools. The checker already has these rules:
{EXISTING_RULES}

You will see conversations that FAILED although none of these rules fired (each with the checks
τ²-bench says it failed), and some that PASSED. Find the patterns of error that explain the
failures, and write each one as a new rule a second model can check.

Each rule must:
- be decidable from the conversation so far, the agent's proposed call and the domain policy only.
  The checker never sees the task, the expected answer, the customer's hidden instructions or
  whether the conversation passed. Don't write a rule that needs them.
- not repeat the existing rules above.
- be general (it should catch the same kind of error in conversations you haven't seen), and
  rarely fire on the passing conversations.

Write at most {MAX_CANDIDATES} rules, the most useful first. Answer with JSON only:
{{"rules": [{{
  "name": "snake_case_name",
  "what": "one sentence: the error this rule catches",
  "applies_to": "call" or "conversation",
  "question": "a yes/no question for the checker; YES means the error is present",
  "live": true or false,
  "block_message": "one sentence the agent is told when the change is blocked, or \\"\\"",
  "seen_in": ["the example ids it explains"]
}}]}}

"call": checked at each state-changing call, on the conversation before it and the call itself.
"conversation": checked once on the whole conversation (e.g. a change the agent should have made
and never did); always live=false. "live": true only if, at the moment of the call, the checker
can be confident enough to stop the change and send block_message to the agent."""

ROUND_GUIDANCE = """

THIS IS ROUND {round}. Earlier rounds already proposed the rules under ALREADY TRIED: don't repeat
them or write close variants. The failures you'll see, by domain: {by_domain}. Spread your rules
across the domains in proportion to those failures, so every domain with failures gets at least
one rule. Two kinds of error to look for in each domain:
- a change that doesn't match what the customer asked for or agreed to, or what the agent's own
  lookups showed (wrong item, wrong order or reservation, wrong amount, more or less than asked):
  a "call" rule;
- something the policy required that the agent never did, or a wrong or missing statement to the
  customer: a "conversation" rule, never a "call" rule, since there may be no call to check.
Write at most {limit} rules (this overrides the number above)."""

RULE_SYSTEM = """You check one rule for a customer-service AI agent, using only the conversation and
the domain policy below. The rule describes an ERROR. Decide whether the error is present.

Answer with JSON only: {"violated": true or false, "evidence": "...", "reason": "..."}
"evidence": when violated, a short exact quote (a few words) from the conversation that shows the
error; otherwise "". "reason": one sentence. If you aren't sure the error is present, answer false."""


# --- Candidates -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    name: str
    what: str
    applies_to: str  # "call" or "conversation"
    question: str
    live: bool = False
    block_message: str = ""
    seen_in: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()  # the state-changing tools it checks; () = every one in the domain
    tool_chars: int = 0  # how much of each tool result the checker sees; 0 = MAX_CHARS
    model: str = ""  # the checker's model; "" = RULE_MODEL

    @classmethod
    def from_dict(cls, d: dict) -> Candidate:
        name = re.sub(r"[^a-z0-9_]", "_", str(d.get("name", "")).strip().lower()).strip("_")
        applies_to = str(d.get("applies_to", "call")).strip().lower()
        if not name or not str(d.get("question", "")).strip():
            raise ValueError(f"a candidate needs a name and a question: {d!r}")
        if applies_to not in ("call", "conversation"):
            raise ValueError(f"applies_to must be 'call' or 'conversation', not {applies_to!r}")
        live = (
            d.get("live") is True and applies_to == "call" and bool(str(d.get("block_message", "")).strip())
        )
        return cls(
            name=name,
            what=str(d.get("what", "")).strip(),
            applies_to=applies_to,
            question=str(d["question"]).strip(),
            live=live,
            block_message=str(d.get("block_message", "")).strip() if live else "",
            seen_in=tuple(str(x) for x in d.get("seen_in") or ()),
            tools=tuple(str(x) for x in d.get("tools") or ()),
            tool_chars=int(d.get("tool_chars") or 0),
            model=str(d.get("model") or ""),
        )

    def to_dict(self) -> dict:
        return {**asdict(self), "seen_in": list(self.seen_in), "tools": list(self.tools)}

    def checks(self, tool: str) -> bool:
        return not self.tools or tool in self.tools


# --- The LLM, cached --------------------------------------------------------------------------


class LLM:
    """One pinned model at temperature 0, JSON answers, cached by the exact request (a re-run is free)."""

    def __init__(
        self, model: str, completion: Callable[..., Any] | None = None, cache_path: Path | str | None = None
    ) -> None:
        self.model = model
        self._completion = completion
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.calls = 0
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    self.cache[row["key"]] = row["answer"]
                except (json.JSONDecodeError, KeyError):
                    continue

    def __call__(self, messages: list[Message]) -> dict:
        key = hashlib.sha256(json.dumps([self.model, messages], sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        if self._completion is None:
            import litellm

            litellm.suppress_debug_info = True  # ty: ignore[invalid-assignment]
            self._completion = litellm.completion
        resp = self._completion(
            model=self.model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
            num_retries=2,
            timeout=180,
        )
        self.calls += 1
        answer = json.loads(resp.choices[0].message.content or "{}")
        if not isinstance(answer, dict):
            raise ValueError(f"the model answered {answer!r}, not a JSON object")
        with self._lock:
            self.cache[key] = answer
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a") as f:
                    f.write(json.dumps({"key": key, "model": self.model, "answer": answer}) + "\n")
        return answer


def judges_off() -> bool:
    return os.environ.get("REFLEXIVE_INTENT_JUDGE", "on").lower() in ("off", "0", "false", "no")


_rule_llms: dict[str, LLM] = {}


def default_rule_llm(model: str = "") -> LLM | None:
    """The shared rule checker for ``model`` (default `RULE_MODEL`), or None when
    ``REFLEXIVE_INTENT_JUDGE=off`` (then these rules never fire)."""
    if judges_off():
        return None
    model = model or RULE_MODEL
    if model not in _rule_llms:
        _rule_llms[model] = LLM(model, cache_path=os.environ.get("REFLEXIVE_META_CACHE") or None)
    return _rule_llms[model]


# --- Transcripts: what a rule reads -----------------------------------------------------------


def _text(content: object) -> str:
    if content is None:
        return ""
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def _call_text(call: dict) -> str:
    fn = call.get("function") or call
    args = fn.get("arguments")
    args = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    return f"{fn.get('name')}({args})"


def _name(call: dict) -> str:
    return str((call.get("function") or call).get("name"))


def transcript(conversation: Sequence[Message], max_chars: int = MAX_CHARS, tool_chars: int = 0) -> str:
    """Customer and agent turns, the agent's calls and the tool results; no system prompt.

    ``tool_chars``: how much of each tool result to keep (0: ``max_chars``)."""
    lines = []
    for m in conversation:
        role = m.get("role")
        text = _text(m.get("content")).strip()
        if role == "user" and text:
            lines.append(f"CUSTOMER: {text[:max_chars]}")
        elif role == "assistant":
            if text:
                lines.append(f"AGENT: {text[:max_chars]}")
            lines += [f"AGENT CALLS: {_call_text(c)[:max_chars]}" for c in m.get("tool_calls") or []]
        elif role == "tool":
            lines.append(f"TOOL RESULT: {text[: tool_chars or max_chars]}")
    return "\n".join(lines)


def policy_of(view: Sequence[Message]) -> str:
    return _text(view[0].get("content")) if view and view[0].get("role") == "system" else ""


def _norm(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'")
    return " ".join(re.findall(r"[a-z0-9'#_-]+", text))


def evidence_found(evidence: str, shown: str) -> bool:
    """The quote is in what the checker was shown (case, spacing and punctuation aside)."""
    q = _norm(evidence)
    return bool(q) and f" {q} " in f" {_norm(shown)} "


def rule_request(
    c: Candidate, conversation: Sequence[Message], policy: str, call: dict | None
) -> list[Message]:
    shown = transcript(conversation, tool_chars=c.tool_chars)
    parts = [
        f"DOMAIN POLICY:\n{policy[:MAX_POLICY_CHARS]}",
        f"RULE: {c.what}\nQUESTION: {c.question}",
        f"CONVERSATION{' SO FAR' if call else ''}:\n{shown}",
    ]
    if call is not None:
        parts.append(f"PROPOSED CHANGE (not yet run):\n{_call_text(call)}")
    return [{"role": "system", "content": RULE_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


@dataclass(frozen=True)
class Hit:
    flagged: bool
    evidence: str = ""
    reason: str = ""
    error: str | None = None


def check(
    c: Candidate, conversation: Sequence[Message], policy: str, call: dict | None, llm: Callable[[list], dict]
) -> Hit:
    """One candidate on one moment. A model error never flags; nor does evidence it made up."""
    messages = rule_request(c, conversation, policy, call)
    try:
        a = llm(messages)
    except Exception as e:  # noqa: BLE001 - a checker error must never block a change
        return Hit(False, error=f"{type(e).__name__}: {str(e)[:200]}")
    evidence, reason = str(a.get("evidence") or ""), str(a.get("reason") or "")
    if a.get("violated") is not True:
        return Hit(False, evidence, reason)
    shown = messages[1]["content"].split("CONVERSATION", 1)[1]
    if not evidence_found(evidence, shown):
        return Hit(False, evidence, f"its evidence isn't in the conversation ({reason})")
    return Hit(True, evidence, reason)


def candidate_findings(c: Candidate, rec: dict, llm: Callable[[list], dict]) -> list[dict]:
    """Every place ``c`` fires in a saved conversation; judge errors come back with ``error`` set."""
    from reflexive.hallucination import agent_view_of

    view = agent_view_of(rec)
    policy = policy_of(view)
    out: list[dict] = []
    if c.applies_to == "conversation":
        h = check(c, view, policy, None, llm)
        if h.flagged or h.error:
            out.append({"turn": len(view), "tool": None, **asdict(h)})
        return out
    writes = WRITE_TOOLS.get(rec.get("domain", ""), ALL_WRITE_TOOLS)
    for i, m in enumerate(view):
        if m.get("role") != "assistant":
            continue
        for call in m.get("tool_calls") or []:
            if _name(call) not in writes or not c.checks(_name(call)):
                continue
            h = check(c, view[:i], policy, call, llm)
            if h.flagged or h.error:
                out.append({"turn": i, "tool": _name(call), **asdict(h)})
    return out


# --- As a rule in reflexive.actions.RULES -----------------------------------------------------


def as_rule(
    c: Candidate, llm: Any = "default", kind: str = "META", prefix: str = "meta_"
) -> Callable[[Sequence[Message], dict, str], Any]:
    """A ``call`` candidate as a `reflexive.actions.Rule` (conversation rules are after the fact only).

    ``kind``: the finding type (``META`` for meta-healing's rules; a hand-written rule names its own)."""

    def rule(conversation: Sequence[Message], call: dict, policy: str) -> Any:
        from reflexive.actions import ActionFinding

        checker = default_rule_llm(c.model) if llm == "default" else llm
        if checker is None or not c.checks(_name(call)):
            return None
        h = check(c, conversation, policy, call, checker)
        if h.error:
            return ActionFinding(
                kind,
                f"{c.name}_error",
                f"Not blocked: the checker couldn't answer ({h.error}).",
                live=False,
            )
        if not h.flagged:
            return None
        message = f"Note from the system, not the customer: {c.block_message}" if c.live else ""
        return ActionFinding(
            kind, c.name, f'{c.what} Evidence: "{h.evidence}". {h.reason}', live=c.live, message=message
        )

    rule.__name__ = f"{prefix}{c.name}"
    return rule


def accepted(path: Path | str = RULES_PATH) -> list[Candidate]:
    p = Path(path)
    if not p.exists():
        return []
    return [Candidate.from_dict(r) for r in json.loads(p.read_text()).get("accepted", [])]


def meta_rules(path: Path | str = RULES_PATH) -> list[Callable]:
    """The accepted call rules, ready for `reflexive.actions.RULES`."""
    return [as_rule(c) for c in accepted(path) if c.applies_to == "call"]


def conversation_findings(rec: dict, path: Path | str = RULES_PATH, llm: Any = "default") -> list[dict]:
    """The accepted conversation rules (after the fact) on a whole saved conversation."""
    if judges_off() and llm == "default":
        return []
    out = []
    for c in accepted(path):
        if c.applies_to == "conversation":
            checker = default_rule_llm(c.model) if llm == "default" else llm
            if checker is None:
                continue
            out += [{"rule": c.name, "live": False, **f} for f in candidate_findings(c, rec, checker)]
    return out


# --- The split, what's unexplained, the proposer ----------------------------------------------


def task_key(rec: dict) -> str:
    return f"{rec['domain']}:{rec['task_id']}"


def example_id(rec: dict) -> str:
    return f"{task_key(rec)}#{rec.get('sample', 0)}"


def passed(rec: dict) -> bool:
    return bool((rec.get("outcome") or {}).get("passed"))


def split(recs: Sequence[dict], seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Two thirds to propose from, one third held out, by task (every sample of a task on one side),
    stratified by domain and by whether the task failed."""
    by_task: dict[str, list[dict]] = {}
    for r in recs:
        by_task.setdefault(task_key(r), []).append(r)
    strata: dict[tuple[str, bool], list[str]] = {}
    for k, rs in by_task.items():
        strata.setdefault((rs[0]["domain"], all(passed(r) for r in rs)), []).append(k)
    rng = random.Random(seed)
    held: set[str] = set()
    for s in sorted(strata):
        keys = sorted(strata[s])
        rng.shuffle(keys)
        held |= set(keys[: round(len(keys) * HOLDOUT_SHARE)])
    propose = [r for k in sorted(by_task) if k not in held for r in by_task[k]]
    holdout = [r for k in sorted(by_task) if k in held for r in by_task[k]]
    return propose, holdout


def explained(rec: dict) -> bool:
    """A current rule explains it: a Tool/Arg/Loop finding or a live "no yes" finding."""
    from reflexive.hallucination import FailureType, classify_record
    from reflexive.intent import intent_findings, no_yes

    if classify_record(rec).failure != FailureType.NONE:
        return True
    found = intent_findings(rec, rules=[no_yes])
    if errors := [f for f in found if f["rule"] == "no_yes_judge_error"]:
        # counting it unexplained would quietly hand the proposer failures "no yes" already covers
        raise RuntimeError(f"the no-yes judge failed on {example_id(rec)}: {errors[0]['reason']}")
    return any(f["live"] for f in found)


def failed_checks(rec: dict) -> list[str]:
    checks = (rec.get("outcome") or {}).get("checks") or {}
    return sorted(k for k, v in checks.items() if isinstance(v, (int, float)) and v < 1)


def proposer_request(
    failures: Sequence[dict],
    passes: Sequence[dict],
    round: int = 1,
    tried: Sequence[Candidate] = (),
    limit: int = MAX_CANDIDATES,
) -> list[Message]:
    """Round 1's request is unchanged, so its cached answer still replays; later rounds add guidance."""
    from reflexive.hallucination import agent_view_of

    policies: dict[str, str] = {}
    blocks = []
    for label, recs in (("FAILED", failures), ("PASSED", passes)):
        for r in recs:
            view = agent_view_of(r)
            policies.setdefault(r["domain"], policy_of(view)[:MAX_POLICY_CHARS])
            checks = (
                f" (failed checks: {', '.join(failed_checks(r)) or 'unknown'})" if label == "FAILED" else ""
            )
            blocks.append(f"=== {example_id(r)}: {label}{checks}\n{transcript(view, 600)}")
    pol = "\n\n".join(f"=== {d.upper()} POLICY\n{p}" for d, p in sorted(policies.items()))
    user = pol + "\n\n" + "\n\n".join(blocks)
    system = PROPOSER_SYSTEM
    if round > 1:
        by_domain = {}
        for r in failures:
            by_domain[r["domain"]] = by_domain.get(r["domain"], 0) + 1
        counts = ", ".join(f"{d} {n}" for d, n in sorted(by_domain.items())) or "none"
        system += ROUND_GUIDANCE.format(round=round, by_domain=counts, limit=limit)
        tried_text = "\n".join(f"- {c.name} ({c.applies_to}): {c.what}" for c in tried) or "- none"
        user = f"ALREADY TRIED:\n{tried_text}\n\n{user}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def propose(
    recs: Sequence[dict],
    llm: Callable[[list], dict],
    n_passes: int = 12,
    seed: int = 0,
    round: int = 1,
    tried: Sequence[Candidate] = (),
    limit: int = MAX_CANDIDATES,
) -> tuple[list[Candidate], dict]:
    """Candidates from the propose set's unexplained failures (and ``n_passes`` passes).

    ``round`` > 1 adds `ROUND_GUIDANCE` and the ``tried`` rules; ``limit`` is what is left of the
    `MAX_CANDIDATES` scored across all rounds."""
    failures = [r for r in recs if not passed(r) and not explained(r)]
    pool = [r for r in recs if passed(r)]
    rng = random.Random(seed)
    passes = rng.sample(pool, min(n_passes, len(pool)))
    answer = llm(proposer_request(failures, passes, round, tried, limit))
    out: list[Candidate] = []
    for d in answer.get("rules") or []:
        try:
            c = Candidate.from_dict(d)
        except (ValueError, TypeError, KeyError):
            continue
        if c.name not in {x.name for x in out} | {x.name for x in tried}:
            out.append(c)
    meta = {
        "unexplained_failures": [example_id(r) for r in failures],
        "passes_shown": [example_id(r) for r in passes],
        "proposed": len(answer.get("rules") or []),
        "round": round,
    }
    return out[:limit], meta


# --- Scoring and the bar ----------------------------------------------------------------------


@dataclass
class Score:
    name: str
    unexplained_failures: list[int] = field(default_factory=lambda: [0, 0])  # flagged, all
    other_failures: list[int] = field(default_factory=lambda: [0, 0])  # failures a rule explains already
    passes: list[int] = field(default_factory=lambda: [0, 0])
    errors: int = 0
    first_error: str = ""
    flagged: list[dict] = field(default_factory=list)  # one per flagged conversation, for review

    @property
    def share(self) -> float:
        k, n = self.unexplained_failures
        return k / n if n else 0.0

    @property
    def meets_bar(self) -> bool:
        return (
            self.errors == 0
            and self.unexplained_failures[0] >= MIN_FLAGGED
            and self.share >= MIN_SHARE
            and self.passes[0] <= MAX_PASSES
        )

    def to_dict(self) -> dict:
        return {**asdict(self), "share": self.share, "meets_bar": self.meets_bar}


def score(
    candidates: Sequence[Candidate], holdout: Sequence[dict], llm: Callable[[list], dict]
) -> list[Score]:
    """Each candidate on the held-out conversations. Explained-ness is worked out once per conversation."""
    kinds = {}
    for r in holdout:
        kinds[example_id(r)] = (
            "passes" if passed(r) else ("other_failures" if explained(r) else "unexplained_failures")
        )
    out = []
    for c in candidates[:MAX_CANDIDATES]:
        s = Score(c.name)
        for r in holdout:
            fs = candidate_findings(c, r, llm)
            s.errors += sum(1 for f in fs if f["error"])
            s.first_error = s.first_error or next((f["error"] for f in fs if f["error"]), "")
            hits = [f for f in fs if f["flagged"]]
            row = getattr(s, kinds[example_id(r)])
            row[0] += bool(hits)
            row[1] += 1
            if hits:
                s.flagged.append({"id": example_id(r), "kind": kinds[example_id(r)], "findings": hits})
        out.append(s)
    return out


def shortlist(scores: Sequence[Score]) -> list[str]:
    """The candidates that meet the bar, best first, at most `MAX_ACCEPTED` (a person still reviews them)."""
    ok = [s for s in scores if s.meets_bar]
    ok.sort(key=lambda s: (-s.unexplained_failures[0], s.passes[0], s.name))
    return [s.name for s in ok[:MAX_ACCEPTED]]
