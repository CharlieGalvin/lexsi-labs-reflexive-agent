"""Meta-healing v2 (STAGE2.md, amended Sat 21 Nov 22:01): the checker proposes, tests and refines its
own rules on the learning set, then is tested once on fresh held-out conversations.

v1 (`reflexive.meta_heal`) had three gaps, each fixed here:

1. **What "right" was.** On the learning set, the proposer sees each failure's expected actions
   (τ²-bench's). The rules it writes still read only the conversation, the call and the policy.
2. **Cut-off evidence.** Tool results are shown in full to the checker (6,000 characters) and
   mostly in full to the proposer (3,000), and the checker is full GPT-4.1.
3. **No feedback.** Up to 3 rounds: propose, score every candidate on the learning set, show the
   proposer what each caught and wrongly flagged, refine. Only the last round is held out.

The proposer's prompt names no kind of error. "Mismatch" isn't among the rules it's told about, so
we can see whether it is rediscovered.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace

from reflexive import meta_heal as mh
from reflexive.intent import WRITE_TOOLS

PROPOSER_MODEL = "gpt-4.1-2025-04-14"
CHECKER_MODEL = "gpt-4.1-2025-04-14"
ROUNDS = 3
PER_ROUND = 8
CHECKER_TOOL_CHARS = 6000
PROPOSER_TOOL_CHARS = 3000
MIN_ADDED, MIN_ADDED_SHARE, MAX_PASS_SHARE = 3, 0.10, 0.10  # the held-out bar
REVIEW_EXAMPLES, MAX_ACCEPTED = 5, 3

EXISTING = """\
- TOOL: the agent calls a tool that doesn't exist, or one only the customer can use.
- ARG: a call argument is a placeholder, never appeared in the conversation (made up), is an
  unknown or missing parameter, has the wrong type, or the arguments are malformed.
- LOOP: the same failing call repeated, the same call or message 3 times, or the step limit hit.
- no_yes: a state-changing call made before the customer explicitly agreed to that change."""

SYSTEM = f"""You improve an automatic checker for a customer-service AI agent.

The agent serves customers of an airline, a retail shop or a phone company (τ²-bench), following
the domain policy, by calling tools. The checker already has these rules:
{EXISTING}

You will see conversations that FAILED although none of these rules fired. For each, you are shown
the actions τ²-bench expected the agent to take (the answer key). Compare what the agent did with
what it should have done, find the kinds of error that explain the failures, and write each kind
as a rule a second model can check. You will also see some conversations that PASSED: a good rule
rarely fires on them.

Each rule must:
- be decidable WITHOUT the answer key: from the conversation so far (customer and agent turns,
  the agent's calls and the tool results), the agent's proposed call and the domain policy. The
  answer key is only for you, to find the errors.
- not repeat the existing rules above.
- be general: it should catch the same kind of error in conversations about other customers,
  orders, products and flights. Never name a specific customer, order, item, flight or task.
- name the exact tools it applies to.

Answer with JSON only:
{{"rules": [{{
  "name": "snake_case_name",
  "what": "one sentence: the error this rule catches",
  "applies_to": "call" or "conversation",
  "tools": ["the exact state-changing tool names it checks; [] for a conversation rule"],
  "question": "a yes/no question for the checker; YES means the error is present",
  "live": true or false,
  "block_message": "one sentence the agent is told when the change is blocked, or \\"\\""
}}]}}

"call": checked at each state-changing call (of the tools listed), on the conversation before it
and the call itself. "conversation": checked once on the whole conversation (e.g. a change that
should have been made and never was); always live=false. "live": true only if, at the moment of
the call, the checker can be confident enough to stop the change and send block_message."""

FEEDBACK = """Here is how each of your rules did on these conversations (the checker answered each
rule's question at every place it applies; it only sees the conversation and the policy).

{rows}

Failures no rule caught: {missed}

Refine: keep what works, fix rules that flag passing conversations or catch nothing, drop the rest,
and add rules for the kinds of failure still missed. At most {n} rules. Same JSON format."""


# --- What the proposer sees -------------------------------------------------------------------


def expected_actions(rec: dict) -> list[str]:
    """τ²-bench's expected actions for a train task (shown to the proposer only)."""
    from reflexive.outcome import scorer

    task = scorer(rec["domain"]).task(str(rec["task_id"]))
    ec = task.evaluation_criteria
    out = []
    for a in (ec.actions if ec and ec.actions else []) or []:
        args = json.dumps(a.arguments or {}, ensure_ascii=False, sort_keys=True)
        out.append(f"{a.name}({args})")
    return out


def proposer_request(
    failures: Sequence[dict], passes: Sequence[dict], expected: Callable[[dict], list[str]] = expected_actions
) -> list[mh.Message]:
    from reflexive.hallucination import agent_view_of

    policies: dict[str, str] = {}
    blocks = []
    for r in failures:
        view = agent_view_of(r)
        policies.setdefault(r["domain"], mh.policy_of(view)[: mh.MAX_POLICY_CHARS])
        exp = "\n".join(f"  {x}" for x in expected(r)) or "  (no state-changing action expected)"
        blocks.append(
            f"=== {mh.example_id(r)}: FAILED\nEXPECTED ACTIONS (answer key, for you only):\n{exp}\n"
            f"CONVERSATION:\n{mh.transcript(view, mh.MAX_CHARS, PROPOSER_TOOL_CHARS)}"
        )
    for r in passes:
        view = agent_view_of(r)
        policies.setdefault(r["domain"], mh.policy_of(view)[: mh.MAX_POLICY_CHARS])
        blocks.append(
            f"=== {mh.example_id(r)}: PASSED\n{mh.transcript(view, mh.MAX_CHARS, PROPOSER_TOOL_CHARS)}"
        )
    pol = "\n\n".join(f"=== {d.upper()} POLICY\n{p}" for d, p in sorted(policies.items()))
    user = pol + "\n\n" + "\n\n".join(blocks) + f"\n\nWrite at most {PER_ROUND} rules."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def with_domains(c: mh.Candidate) -> mh.Candidate:
    """``c`` with the domains it was written for, when it doesn't say: from its name's prefix
    (``retail_...``), else from its tools; neither means every domain. The proposer's prompt is
    unchanged, so its cached answers still replay."""
    if c.domains:
        return c
    domains = tuple(d for d in WRITE_TOOLS if c.name.startswith(f"{d}_"))
    if not domains and c.tools:
        domains = tuple(d for d, tools in WRITE_TOOLS.items() if set(c.tools) & tools)
    return replace(c, domains=domains)


def parse(answer: dict, limit: int = PER_ROUND) -> list[mh.Candidate]:
    out: list[mh.Candidate] = []
    for d in answer.get("rules") or []:
        try:
            c = mh.Candidate.from_dict(d)
        except (ValueError, TypeError, KeyError):
            continue
        # No domains here: learning runs each rule as the cached rounds did (see `learn`).
        c = replace(c, model=CHECKER_MODEL, tool_chars=CHECKER_TOOL_CHARS)
        if c.name not in {x.name for x in out}:
            out.append(c)
    return out[:limit]


# --- Scoring ----------------------------------------------------------------------------------


def findings_by_record(
    cands: Sequence[mh.Candidate],
    recs: Sequence[dict],
    llm: Callable[[list], dict],
    workers: int = 8,
    on_found: Callable[[dict], None] | None = None,
) -> dict[str, dict[str, list[dict]]]:
    """{candidate: {example id: findings}}: every place each candidate fires, errors included.
    ``on_found`` sees the result first (the script stops there on a missing key)."""
    jobs = [(c, r) for c in cands for r in recs]

    def one(job: tuple[mh.Candidate, dict]) -> tuple[str, str, list[dict]]:
        c, r = job
        return c.name, mh.example_id(r), mh.candidate_findings(c, r, llm)

    out: dict[str, dict[str, list[dict]]] = {c.name: {} for c in cands}
    with ThreadPoolExecutor(workers) as pool:
        for name, rid, fs in pool.map(one, jobs):
            out[name][rid] = fs
    if on_found:
        on_found(out)
    return out


def feedback(
    cands: Sequence[mh.Candidate], found: dict, failures: Sequence[dict], passes: Sequence[dict], n: int
) -> str:
    rows, caught_any = [], set()
    for c in cands:
        f_ids = [
            mh.example_id(r) for r in failures if any(f["flagged"] for f in found[c.name][mh.example_id(r)])
        ]
        p_hits = [(mh.example_id(r), found[c.name][mh.example_id(r)]) for r in passes]
        p_hits = [(i, [f for f in fs if f["flagged"]]) for i, fs in p_hits]
        p_hits = [(i, fs) for i, fs in p_hits if fs]
        caught_any |= set(f_ids)
        lines = [
            f"- {c.name}: caught {len(f_ids)}/{len(failures)} failures ({', '.join(f_ids) or 'none'}); "
            f"wrongly flagged {len(p_hits)}/{len(passes)} passes"
        ]
        for i, fs in p_hits[:3]:
            lines.append(f'    wrong flag in {i}: "{fs[0]["evidence"][:120]}" ({fs[0]["reason"][:200]})')
        rows.append("\n".join(lines))
    missed = [mh.example_id(r) for r in failures if mh.example_id(r) not in caught_any]
    return FEEDBACK.format(rows="\n".join(rows), missed=", ".join(missed) or "none", n=n)


def learn(
    recs: Sequence[dict],
    proposer: Callable[[list], dict],
    checker: Callable[[list], dict],
    rounds: int = ROUNDS,
    n_passes_shown: int = 8,
    n_passes_scored: int = 24,
    seed: int = 0,
    expected: Callable[[dict], list[str]] = expected_actions,
    explained: Callable[[dict], bool] = mh.explained,
    log: Callable[[str], None] = print,
    on_found: Callable[[dict], None] | None = None,
) -> dict:
    """The learning rounds. Returns every round's candidates and learning-set counts, and the last
    round that produced rules.

    The passes scored for feedback are drawn independently of those shown, so some overlap
    (``passes_overlap``): the learning-set pass counts are partly on conversations the proposer
    read. Learning-set counts run each rule on every domain; the held-out test (`held_out`) runs it
    only on its own domains. Both are kept as they were so the cached proposer answers replay."""
    failures = [r for r in recs if not mh.passed(r) and not explained(r)]
    pool = sorted((r for r in recs if mh.passed(r)), key=mh.example_id)
    rng = random.Random(seed)
    shown = rng.sample(pool, min(n_passes_shown, len(pool)))
    scored = rng.sample(pool, min(n_passes_scored, len(pool)))
    messages = proposer_request(failures, shown, expected)
    history = []
    cands: list[mh.Candidate] = []
    for k in range(1, rounds + 1):
        answer = proposer(messages)
        parsed = parse(answer)
        if not parsed:
            log(
                f"round {k}: no usable rules; keeping round {k - 1}'s"
                if cands
                else f"round {k}: no usable rules"
            )
            break
        cands = parsed
        found = findings_by_record(cands, [*failures, *scored], checker, on_found=on_found)
        errors = sum(1 for c in found.values() for fs in c.values() for f in fs if f.get("error"))
        row = {"round": k, "candidates": [c.to_dict() for c in cands], "errors": errors, "counts": {}}
        for c in cands:
            fc = sum(any(f["flagged"] for f in found[c.name][mh.example_id(r)]) for r in failures)
            pc = sum(any(f["flagged"] for f in found[c.name][mh.example_id(r)]) for r in scored)
            row["counts"][c.name] = {"failures": [fc, len(failures)], "passes": [pc, len(scored)]}
            log(f"round {k}: {c.name:45s} failures {fc}/{len(failures)}  passes {pc}/{len(scored)}")
        history.append(row)
        if k < rounds:
            messages = [
                *messages,
                {"role": "assistant", "content": json.dumps(answer)},
                {"role": "user", "content": feedback(cands, found, failures, scored, PER_ROUND)},
            ]
    return {
        "unexplained_failures": [mh.example_id(r) for r in failures],
        "passes_shown": [mh.example_id(r) for r in shown],
        "passes_scored": [mh.example_id(r) for r in scored],
        "passes_overlap": sorted({mh.example_id(r) for r in shown} & {mh.example_id(r) for r in scored}),
        "rounds": history,
        "final": [c.to_dict() for c in cands],
    }


# --- The held-out test ------------------------------------------------------------------------


@dataclass
class HeldOut:
    name: str
    added: list[int] = field(default_factory=lambda: [0, 0])  # failures no frozen rule catches
    failures: list[int] = field(default_factory=lambda: [0, 0])
    passes: list[int] = field(default_factory=lambda: [0, 0])
    passes_errored: int = 0  # passes with a checker error and no flag: flagged at worst
    rediscovers: list[int] = field(default_factory=lambda: [0, 0])  # of the failures "mismatch" flags
    errors: int = 0
    flagged: list[dict] = field(default_factory=list)

    @property
    def added_share(self) -> float:
        return self.added[0] / self.added[1] if self.added[1] else 0.0

    @property
    def pass_share_at_worst(self) -> float:
        return (self.passes[0] + self.passes_errored) / self.passes[1] if self.passes[1] else 0.0

    @property
    def meets_bar(self) -> bool:
        return (
            self.added[0] >= MIN_ADDED
            and self.added_share >= MIN_ADDED_SHARE
            and self.pass_share_at_worst <= MAX_PASS_SHARE
        )

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "added_share": self.added_share,
            "pass_share_at_worst": self.pass_share_at_worst,
            "meets_bar": self.meets_bar,
        }


def held_out(
    cands: Sequence[mh.Candidate],
    recs: Sequence[dict],
    frozen: dict[str, dict[str, bool]],
    checker: Callable[[list], dict],
    workers: int = 8,
    on_found: Callable[[dict], None] | None = None,
) -> list[HeldOut]:
    """``frozen``: per example id, whether Tool/Arg/Loop, "no yes" and "mismatch" catch it
    (keys ``tal``, ``no_yes``, ``mismatch``).

    Each rule is scored only on the conversations of its own domains (`with_domains`): a retail
    rule's flags on airline passes say nothing about it, and the guard never runs it there."""
    cands = [with_domains(c) for c in cands]
    found: dict[str, dict[str, list[dict]]] = {c.name: {} for c in cands}
    for c in cands:
        mine = [r for r in recs if c.covers(r["domain"])]
        found[c.name] = findings_by_record([c], mine, checker, workers, on_found)[c.name]
    out = []
    for c in cands:
        s = HeldOut(c.name)
        for r in (r for r in recs if c.covers(r["domain"])):
            rid = mh.example_id(r)
            fs = found[c.name][rid]
            err = any(f.get("error") for f in fs)
            s.errors += sum(1 for f in fs if f.get("error"))
            hit = any(f["flagged"] for f in fs)
            fz = frozen[rid]
            if mh.passed(r):
                s.passes[0] += hit
                s.passes[1] += 1
                s.passes_errored += err and not hit
                kind = "pass"
            else:
                s.failures[0] += hit
                s.failures[1] += 1
                uncaught = not (fz["tal"] or fz["no_yes"] or fz["mismatch"])
                if uncaught:
                    s.added[0] += hit
                    s.added[1] += 1
                if fz["mismatch"]:
                    s.rediscovers[0] += hit
                    s.rediscovers[1] += 1
                kind = "added" if uncaught else "failure"
            if hit:
                s.flagged.append({"id": rid, "kind": kind, "findings": [f for f in fs if f["flagged"]]})
        out.append(s)
    return out


def layers(
    ids: dict[str, bool], frozen: dict[str, dict[str, bool]], scores: Sequence[dict], names: Sequence[str]
) -> list[dict]:
    """What the loop can see without an answer key, one layer of the checker at a time.

    ``ids``: example id -> passed. Each row is cumulative: failures caught and passes flagged by
    Tool/Arg/Loop, then + "no yes", then + "mismatch", then + the meta-healing rules in ``names``."""
    hits: dict[str, set[str]] = {s["name"]: {x["id"] for x in s["flagged"]} for s in scores}
    fails = [i for i, ok in ids.items() if not ok]
    passes = [i for i, ok in ids.items() if ok]
    steps = [
        ("Tool / Arg / Loop (Stage 1)", lambda i: frozen[i]["tal"]),
        ('+ "no yes" (hand-written)', lambda i: frozen[i]["no_yes"]),
        ('+ "mismatch" (hand-written after meta-healing v1)', lambda i: frozen[i]["mismatch"]),
    ]
    if names:
        steps.append(
            (f"+ meta-healing v2 ({', '.join(names)})", lambda i: any(i in hits.get(n, ()) for n in names))
        )
    rows, seen = [], set()
    for label, caught in steps:
        seen |= {i for i in ids if caught(i)}
        rows.append(
            {
                "layer": label,
                "failures": [sum(i in seen for i in fails), len(fails)],
                "passes": [sum(i in seen for i in passes), len(passes)],
            }
        )
    return rows


def shortlist(scores: Sequence[HeldOut]) -> list[str]:
    ok = sorted((s for s in scores if s.meets_bar), key=lambda s: (-s.added[0], s.passes[0], s.name))
    return [s.name for s in ok[:MAX_ACCEPTED]]
