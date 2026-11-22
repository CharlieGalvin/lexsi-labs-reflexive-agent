"""The hybrid judge: an LLM reads the whole conversation; code supplies the facts it gets wrong
(STAGE2.md, Stage 2b; the organisers' "LLM-as-a-judge with rule-based exceptions").

    python scripts/labels/hybrid_judge.py --split trainpool          # the development run (format only)
    python scripts/labels/hybrid_judge.py --split fresh              # the held-out run, once
    python scripts/labels/hybrid_judge.py --split fresh --no-judge   # the two baselines only (free)

Three detectors, scored against the answer key on v2's train-task conversations
(``results/episodes/arms/S2-<domain>-<split>.jsonl``):

1. **Rules** (Stage 1): the Tool / Arg / Loop classifier (`classify_record`).
2. **Checker** (Stage 2, frozen): "no yes" and "mismatch", one state-changing call at a time.
3. **Hybrid**: one call to ``gpt-4.1-2025-04-14`` per conversation, at temperature 0, cached. It reads
   the policy and the whole conversation, never the answer key, plus facts computed by code. Each
   fact answers an error the training-data audit found in the checker's judge:

   - every change the agent made, in order, with the customer's last message before it and
     whether the tool accepted it (the checker judged one call of a multi-call change as the whole);
   - every product variant the lookups showed, with its price and options (it misread prices);
   - the Stage 1 classifier's findings, as evidence.

   It answers in a fixed structure, reasoning first: ``analysis``, ``error``, ``turn``, ``kind``,
   ``evidence``. Code checks it: an error counts only if the evidence is quoted from the
   conversation and the turn is an agent turn; otherwise it is counted as invalid and not flagged
   (the checker's flags contradicted their own reasoning).

**The answer key** (never shown to any detector): τ²-bench's verdict, and for a failure, the turn
where the agent's changes first leave the task's expected ones (`onpolicy_fixes.Domains.fix_point`:
an unexpected change, or the last turn when an expected change was never made).

**Reported per detector, overall and per domain:** failures flagged (recall), passes flagged (false
flags), and, for failures with an answer-key turn, the flagged turn at the true turn and within 4
messages of it. Writes ``results/judge/hybrid-<split>.jsonl`` (one line per conversation) and
``results/judge/hybrid-<split>.json``. Train tasks only; validation tasks are left out.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from reflexive.files import ROOT, latest_records, read_jsonl
from reflexive.splits import assert_not_test, split_of

DOMAINS = ("retail", "airline", "telecom")
ARMS = ROOT / "results" / "episodes" / "arms"
OUT = ROOT / "results" / "judge"
CACHE = ROOT / "results" / "cache"
MODEL = "gpt-4.1-2025-04-14"
NEAR = 4  # messages: about two agent turns
MAX_PRICES = 120
KINDS = (
    "unrequested_change",
    "unconfirmed_change",
    "wrong_arguments",
    "against_policy",
    "missed_action",
    "stopped_short",
    "wrong_information",
)

Record = dict[str, Any]
Message = dict[str, Any]

SYSTEM = """You review a customer-service conversation between an AI agent and a customer. The agent
follows the POLICY below and acts through tools. Decide whether the agent made an error that would
make the task fail, and if so, at which agent turn it first went wrong.

Error kinds:
- unrequested_change: the agent changed something the customer never asked for.
- unconfirmed_change: the agent made a change before the customer explicitly agreed to it, where the
  policy requires agreement.
- wrong_arguments: the change acts on the wrong order, reservation, item, option, amount, passenger,
  payment method or date, compared with what the customer asked for and the lookups showed.
- against_policy: the agent did something the policy forbids, or refused something it allows.
- missed_action: the customer asked for something the policy allows, and the agent never did it.
- stopped_short: the agent ended, transferred or gave up while a requested, allowed action remained.
- wrong_information: the agent told the customer something its lookups contradict.

Rules for you:
- Use the COMPUTED FACTS. They were produced by code from the conversation and are correct. In
  particular, a change carried out in several calls is one change: judge it against the whole list
  of CHANGES MADE, not call by call. Read prices only from PRICES IN THE LOOKUPS; do not do arithmetic
  the facts already give.
- Flag an error only if you can quote the exact words or call from the conversation that show it.
- If the agent's actions match what the customer asked for and the policy allows, there is no error.

Answer with one JSON object, in this order:
{"analysis": "<your step-by-step reasoning, at most 12 sentences>",
 "error": true or false,
 "turn": <the [number] of the agent turn where it first went wrong, or null>,
 "kind": "<one error kind, or null>",
 "evidence": "<an exact quote from the conversation, or null>"}"""


# --- Facts computed by code (the rule-based exceptions) -----------------------------------------


def _text(content: object) -> str:
    if content is None:
        return ""
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def _fn(call: dict) -> tuple[str, Any]:
    fn = call.get("function") or call
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            pass
    return str(fn.get("name")), args


def changes(view: Sequence[Message], writes: Iterable[str]) -> list[dict]:
    """Every state-changing call in order: its turn, the call, the tool's answer, and the customer's
    last message before it."""
    writes = set(writes)
    results = {m.get("tool_call_id"): _text(m.get("content")) for m in view if m.get("role") == "tool"}
    out, last_customer = [], ""
    for i, m in enumerate(view):
        if m.get("role") == "user":
            last_customer = _text(m.get("content")).strip()
        if m.get("role") != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            name, args = _fn(c)
            if name in writes:
                res = results.get(c.get("id"), "")
                out.append(
                    {
                        "turn": i,
                        "call": f"{name}({json.dumps(args, ensure_ascii=False)})",
                        "accepted": not res.startswith("Error"),
                        "customer_before": last_customer[:300],
                    }
                )
    return out


def _walk(x: Any) -> Iterable[dict]:
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from _walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from _walk(v)


def prices(view: Sequence[Message]) -> list[str]:
    """Each product variant the lookups showed: item ID, price, options (retail's ``get_product_details``
    and order lookups), sorted by product then price; at most `MAX_PRICES`."""
    seen: dict[str, tuple[str, float, str]] = {}
    for m in view:
        if m.get("role") != "tool":
            continue
        try:
            data = json.loads(_text(m.get("content")))
        except json.JSONDecodeError:
            continue
        for top in _walk(data):
            name = str(top.get("name") or "")
            variants = top.get("variants")
            if isinstance(variants, dict):
                for vid, v in variants.items():
                    if isinstance(v, dict) and "price" in v:
                        opts = json.dumps(v.get("options") or {}, ensure_ascii=False)
                        seen[str(vid)] = (
                            name,
                            float(v["price"]),
                            opts + ("" if v.get("available", True) else " (unavailable)"),
                        )
            if "item_id" in top and "price" in top:
                opts = json.dumps(top.get("options") or {}, ensure_ascii=False)
                seen.setdefault(
                    str(top["item_id"]), (name or str(top.get("name", "")), float(top["price"]), opts)
                )
    rows = sorted(seen.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    return [f"{p} | item {i} | ${pr:.2f} | {o}" for i, (p, pr, o) in rows[:MAX_PRICES]]


def numbered(view: Sequence[Message], tool_chars: int = 1500) -> str:
    """The conversation with each message's index, as the judge cites turns (no system prompt)."""
    lines = []
    for i, m in enumerate(view):
        role, text = m.get("role"), _text(m.get("content")).strip()
        if role == "user" and text:
            lines.append(f"[{i}] CUSTOMER: {text[:1500]}")
        elif role == "assistant":
            calls = "; ".join(
                f"{_fn(c)[0]}({json.dumps(_fn(c)[1], ensure_ascii=False)})" for c in m.get("tool_calls") or []
            )
            lines.append(f"[{i}] AGENT: {text[:1500]}" + (f" CALLS: {calls}" if calls else ""))
        elif role == "tool":
            lines.append(f"[{i}] TOOL RESULT: {text[:tool_chars]}")
    return "\n".join(lines)


def request(view: Sequence[Message], domain: str, classifier: Sequence[dict]) -> list[Message]:
    """The judge's messages: the policy, the numbered conversation and the computed facts."""
    from reflexive.intent import WRITE_TOOLS

    policy = _text(view[0].get("content")) if view and view[0].get("role") == "system" else ""
    made = changes(view, WRITE_TOOLS.get(domain, ()))
    facts = ["CHANGES MADE, IN ORDER (code):"]
    facts += [
        f"- [{c['turn']}] {c['call']} | tool {'accepted' if c['accepted'] else 'REJECTED'}"
        f' | customer\'s last message before it: "{c["customer_before"]}"'
        for c in made
    ] or ["- none"]
    if domain == "retail":
        facts += [
            "",
            "PRICES IN THE LOOKUPS (code; product | item | price | options):",
            *(prices(view) or ["- none"]),
        ]
    facts += ["", "STAGE 1 CLASSIFIER FINDINGS (code; made-up tools, arguments, loops):"]
    facts += [
        f"- [{f['turn']}] {f['type']}/{f['kind']}: {f.get('detail', '')[:200]}" for f in classifier
    ] or ["- none"]
    user = f"POLICY:\n{policy}\n\nCONVERSATION:\n{numbered(view)}\n\nCOMPUTED FACTS:\n" + "\n".join(facts)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def verdict(answer: dict, view: Sequence[Message]) -> dict:
    """The judge's answer, checked: an error counts only with a quoted piece of the conversation and
    an agent turn."""
    from reflexive.meta_heal import evidence_found

    err = answer.get("error") is True
    turn = answer.get("turn")
    shown = numbered(view, tool_chars=100000)
    if not err:
        return {"flagged": False, "turn": None, "kind": None, "invalid": None}
    if not isinstance(turn, int) or not 0 <= turn < len(view) or view[turn].get("role") != "assistant":
        return {
            "flagged": False,
            "turn": None,
            "kind": answer.get("kind"),
            "invalid": "turn isn't an agent turn",
        }
    if not evidence_found(str(answer.get("evidence") or ""), shown):
        return {
            "flagged": False,
            "turn": None,
            "kind": answer.get("kind"),
            "invalid": "evidence not in the conversation",
        }
    return {"flagged": True, "turn": turn, "kind": answer.get("kind"), "invalid": None}


# --- The answer key and the baselines -----------------------------------------------------------


def _onpolicy() -> Any:
    path = ROOT / "scripts" / "heal" / "onpolicy_fixes.py"
    spec = importlib.util.spec_from_file_location("onpolicy_fixes", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("onpolicy_fixes", mod)
    spec.loader.exec_module(mod)
    return mod


def score(rows: Sequence[dict], detectors: Sequence[str]) -> dict:
    """Per detector and domain: failures flagged, passes flagged, and localisation."""
    out: dict = {}
    for det in detectors:
        out[det] = {}
        for d in (*DOMAINS, "all"):
            rs = [r for r in rows if d == "all" or r["domain"] == d]
            if not rs:
                continue
            fails = [r for r in rs if not r["passed"]]
            passes = [r for r in rs if r["passed"]]
            located = [r for r in fails if r["true_turn"] is not None and r[det]["flagged"]]
            out[det][d] = {
                "failures_flagged": [sum(r[det]["flagged"] for r in fails), len(fails)],
                "passes_flagged": [sum(r[det]["flagged"] for r in passes), len(passes)],
                "at_true_turn": [sum(r[det]["turn"] == r["true_turn"] for r in located), len(located)],
                "within_near": [
                    sum(
                        r[det]["turn"] is not None and abs(r[det]["turn"] - r["true_turn"]) <= NEAR
                        for r in located
                    ),
                    len(located),
                ],
                "invalid": sum(bool(r[det].get("invalid")) for r in rs),
            }
    return out


def label(recs: Iterable[Record], val: set[tuple[str, str]]) -> list[dict]:
    """Every conversation with its answer key and the two baselines' verdicts (no API calls but the
    checker's, which is cached)."""
    from reflexive.hallucination import FailureType, agent_view_of
    from reflexive.intent import intent_findings, no_yes
    from reflexive.metrics import is_broken
    from reflexive.mismatch import mismatch

    dom = _onpolicy().Domains()
    rows = []
    for r in latest_records(recs):
        d, t = r["domain"], str(r["task_id"])
        assert_not_test(d, t)
        if split_of(d, t) != "train" or (d, t) in val or is_broken(r):
            continue
        passed = bool((r.get("outcome") or {}).get("passed"))
        point, diag, _ = dom.fix_point(r)
        true_turn = point.turn if (not passed and point is not None and point.kind != "classifier") else None
        found = [f for f in intent_findings(r, rules=[no_yes, mismatch]) if f.get("live")]
        rows.append(
            {
                "domain": d,
                "task_id": t,
                "sample": r.get("sample", 0),
                "passed": passed,
                "true_turn": true_turn,
                "true_kind": point.kind if point is not None and not passed else None,
                "rules": {
                    "flagged": diag.failure != FailureType.NONE,
                    "turn": diag.first.turn if diag.first else None,
                    "kind": f"{diag.first.type.value}/{diag.first.kind}" if diag.first else None,
                },
                "checker": {
                    "flagged": bool(found),
                    "turn": min((f["turn"] for f in found), default=None),
                    "kind": found[0]["rule"] if found else None,
                },
                "_view": agent_view_of(r),
                "_classifier": [f.to_dict() for f in diag.findings],
            }
        )
    return rows


def combine(row: dict) -> dict:
    """The rules and the checker together, as the guard and the gate used them."""
    a, b = row["rules"], row["checker"]
    turns = [x["turn"] for x in (a, b) if x["flagged"] and x["turn"] is not None]
    return {"flagged": a["flagged"] or b["flagged"], "turn": min(turns) if turns else None, "kind": None}


def judge_all(rows: list[dict], llm: Callable[[list], dict], workers: int) -> None:
    def one(row: dict) -> dict:
        try:
            answer = llm(request(row["_view"], row["domain"], row["_classifier"]))
        except Exception as e:  # noqa: BLE001 - an error never flags, and is counted
            return {"flagged": False, "turn": None, "kind": None, "invalid": f"judge error: {e}"[:200]}
        v = verdict(answer, row["_view"])
        v["analysis"] = str(answer.get("analysis") or "")[:1500]
        return v

    with cf.ThreadPoolExecutor(workers) as pool:
        for row, v in zip(rows, pool.map(one, rows), strict=True):
            row["hybrid"] = v


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--split", required=True, help="the file suffix: trainpool (development) or fresh (held out)"
    )
    p.add_argument("--from", dest="source", type=Path, default=ARMS, help="where P-<domain>-val.jsonl are")
    p.add_argument("--no-judge", action="store_true", help="the baselines only")
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    os.environ["REFLEXIVE_INTENT_JUDGE"] = "on"
    os.environ.setdefault("REFLEXIVE_JUDGE_CACHE", str(CACHE / "intent-judge.jsonl"))
    os.environ.setdefault("REFLEXIVE_META_CACHE", str(CACHE / "meta-rules.jsonl"))
    val: set[tuple[str, str]] = set()
    for d in DOMAINS:
        f = a.source / f"P-{d}-val.jsonl"
        if not f.exists():
            raise SystemExit(f"⛔ no {f}: pass --from with the folder of P-*-val.jsonl")
        val |= {(d, str(r["task_id"])) for r in read_jsonl(f, partial_last_ok=True)}
    recs: list[Record] = []
    for d in DOMAINS:
        f = ARMS / f"S2-{d}-{a.split}.jsonl"
        if f.exists():
            recs += read_jsonl(f, partial_last_ok=True)
    if not recs:
        raise SystemExit(f"⛔ no S2-<domain>-{a.split}.jsonl in {ARMS}")
    rows = label(recs, val)
    for row in rows:
        row["rules+checker"] = combine(row)
    detectors = ["rules", "checker", "rules+checker"]
    if not a.no_judge:
        from reflexive.meta_heal import LLM

        judge_all(rows, LLM(MODEL, cache_path=CACHE / "hybrid-judge.jsonl", max_tokens=2000), a.workers)
        detectors.append("hybrid")
    rep = score(rows, detectors)
    for det in detectors:
        x = rep[det]["all"]
        f, pz, at, near = x["failures_flagged"], x["passes_flagged"], x["at_true_turn"], x["within_near"]
        print(
            f"{det:<14} failures flagged {f[0]}/{f[1]} ({f[0] / max(f[1], 1):.0%})"
            f" | passes flagged {pz[0]}/{pz[1]} ({pz[0] / max(pz[1], 1):.0%})"
            f" | at the true turn {at[0]}/{at[1]}, within {NEAR} {near[0]}/{near[1]}"
            + (f" | invalid {x['invalid']}" if x["invalid"] else "")
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"hybrid-{a.split}.jsonl").write_text(
        "".join(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n" for r in rows)
    )
    (OUT / f"hybrid-{a.split}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(f"saved results/judge/hybrid-{a.split}.json and .jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
