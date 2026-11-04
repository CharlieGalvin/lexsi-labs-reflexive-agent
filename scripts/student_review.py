# ruff: noqa: E501  (long HTML/CSS template lines)
"""Where does the student go wrong? Failure categories plus a browsable log page (SEL-55, SEL-7).

    python scripts/student_review.py                                  # results/fixes/aya.jsonl
    python scripts/student_review.py results/episodes/sample-aya.jsonl
    open results/reports/aya_review.html                              # every conversation, readable

Each conversation is tagged with every problem found, and given one *main reason* (the first
match, in the order below):

- **broken**: an API failure or crash (says nothing about the model)
- **format**: Aya wrote a tool call our parser couldn't read (a call written inside prose is
  noted as "call inside prose" but still worked, so it's never the main reason)
- **no tools**: never called a tool
- **unknown tool**: called a tool the agent doesn't have
- **invented ID**: an ID-like argument (user, order, reservation…) that appeared nowhere earlier
  in the conversation (dates excluded)
- **repeated failed call**: retried a call that had just failed, unchanged (a loop)
- **handed to a human**: transferred instead of solving it
- **ran out of steps / context**: hit τ²-bench's step limit or our context guard
- **no change made**: the task needed a database change and Aya never attempted one
- **wrong change**: attempted changes, but the database ended up wrong
- **told the customer wrongly**: the database was right but a judged check failed
- **device not fixed** (telecom): the customer's phone didn't end up working
- **other**

Standard library only.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import pathlib

WRITE_PREFIXES = ("cancel_", "modify_", "exchange_", "return_", "book_", "update_", "send_",
                  "enable_", "disable_", "refuel_", "suspend_", "resume_", "toggle_")
ORDER = ["broken", "format", "no tools", "unknown tool", "invented ID", "repeated failed call",
         "handed to a human", "ran out of steps / context", "no change made", "wrong change",
         "told the customer wrongly", "device not fixed", "other"]
NOTES = ["call inside prose"]  # tagged, but not a reason for failing
JUDGED = {"NL_ASSERTION", "COMMUNICATE"}


def agent_calls(rec: dict) -> list[dict]:
    return [tc for m in rec["messages"] if m["role"] == "assistant" for tc in (m.get("tool_calls") or [])]


def tool_name(tc: dict) -> str:
    return (tc.get("function") or tc).get("name", "")


def _current_unseen(rec: dict) -> list[str]:
    """Re-run today's invented-ID check (it now skips dates); fall back to what was saved."""
    try:
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
        from reflexive.teacher import unseen_arguments

        return unseen_arguments(rec["messages"])
    except Exception:  # noqa: BLE001 - the saved value is still a fine answer
        return (rec.get("diagnostics") or {}).get("unseen_arguments") or []


def tags_for(rec: dict) -> list[str]:
    """Every problem found in one conversation (empty for a clean pass)."""
    o, x = rec["outcome"], rec.get("diagnostics", {})
    usage = rec.get("agent_usage") or {}
    kinds = collections.Counter(t["kind"] for t in rec.get("raw_turns") or [])
    calls = agent_calls(rec)
    checks = o.get("checks") or {}
    tags = []
    if rec.get("error") or rec.get("termination") in ("unexpected_error", "infrastructure_error"):
        tags.append("broken")
    if usage.get("unparsed") or kinds["unparsed_call"]:
        tags.append("format")
    if usage.get("embedded") or kinds["tool_call_in_prose"]:
        tags.append("call inside prose")  # parsed fine, prose dropped: noted, never the main reason
    if not calls:
        tags.append("no tools")
    if x.get("unknown_tools"):
        tags.append("unknown tool")
    if _current_unseen(rec):
        tags.append("invented ID")
    if x.get("repeated_failures"):
        tags.append("repeated failed call")
    if any(tool_name(tc) == "transfer_to_human_agents" for tc in calls):
        tags.append("handed to a human")
    if rec.get("termination") in ("max_steps", "context_window_exceeded") or (rec.get("flags") or {}).get(
            "context_overflow"):
        tags.append("ran out of steps / context")
    if not o.get("passed"):
        writes = [tc for tc in calls if tool_name(tc).startswith(WRITE_PREFIXES)]
        if checks.get("DB") == 0.0:
            tags.append("wrong change" if writes else "no change made")
        if checks.get("DB", 1.0) == 1.0 and any(checks.get(k) == 0.0 for k in JUDGED):
            tags.append("told the customer wrongly")
        if checks.get("ENV_ASSERTION") == 0.0 or (rec["domain"] == "telecom" and not checks):
            tags.append("device not fixed")
        if not tags:
            tags.append("other")
    return tags


def main_reason(tags: list[str]) -> str | None:
    return next((t for t in ORDER if t in tags), None)


def summarise(recs: list[dict]) -> dict:
    out = {"n": len(recs), "passed": sum(r["outcome"]["passed"] for r in recs), "by_domain": {},
           "main": collections.Counter(), "any": collections.Counter(), "invented_examples": [],
           "unknown_tools": collections.Counter(), "failed_lookups": collections.Counter()}
    for d in ("retail", "airline", "telecom"):
        rr = [r for r in recs if r["domain"] == d]
        if rr:
            out["by_domain"][d] = (sum(r["outcome"]["passed"] for r in rr), len(rr))
    for r in recs:
        tags = tags_for(r)
        for t in tags:
            out["any"][t] += 1
        if not r["outcome"]["passed"]:
            out["main"][main_reason(tags)] += 1
        for v in _current_unseen(r):
            if len(out["invented_examples"]) < 12:
                out["invented_examples"].append((r["domain"], r["task_id"], v))
        results = {m.get("tool_call_id"): str(m.get("content") or "") for m in r["messages"] if m["role"] == "tool"}
        for tc in agent_calls(r):
            res = results.get(tc.get("id"), "")
            if res.startswith("Error: Tool"):
                out["unknown_tools"][tool_name(tc)] += 1
            elif "not found" in res.lower():
                out["failed_lookups"][tool_name(tc)] += 1
    return out


def view_check(recs: list[dict]) -> tuple[int, int, int]:
    """Did the agent see the customer's own phone actions? (telecom, records with agent_view)

    Returns (records checked, records whose full conversation has customer tool calls,
    records where any of those calls also appear in what the agent was shown).
    """
    checked = with_user_calls = leaked = 0
    for r in recs:
        view = r.get("agent_view")
        if view is None or r["domain"] != "telecom":
            continue
        checked += 1
        ids = {tc.get("id") for m in r["messages"] if m["role"] == "user" for tc in (m.get("tool_calls") or [])}
        if ids:
            with_user_calls += 1
            seen = {tc.get("id") for m in view for tc in (m.get("tool_calls") or [])}
            seen |= {m.get("tool_call_id") for m in view if m["role"] == "tool"}
            leaked += bool(ids & seen)
    return checked, with_user_calls, leaked


def print_summary(name: str, s: dict) -> None:
    print(f"\n=== {name}: {s['passed']}/{s['n']} passed "
          + " · ".join(f"{d} {k}/{n}" for d, (k, n) in s["by_domain"].items()) + " ===")
    fails = s["n"] - s["passed"]
    print(f"\nMain reason for each of the {fails} failures:")
    for t in ORDER:
        if s["main"][t]:
            print(f"  {t:<28} {s['main'][t]:>4}  ({s['main'][t] / max(fails, 1):.0%})")
    print("\nConversations showing each problem (a conversation can show several):")
    for t in ORDER + NOTES:
        if s["any"][t]:
            print(f"  {t:<28} {s['any'][t]:>4}  ({s['any'][t] / s['n']:.0%})")
    if s["invented_examples"]:
        print("\nExamples of invented IDs:", ", ".join(f"{d}:{t[:20]} → {v}" for d, t, v in s["invented_examples"]))
    if s["unknown_tools"]:
        print(f"Tools called that the agent doesn't have ({sum(s['unknown_tools'].values())} calls; "
              "8 commonest):", dict(s["unknown_tools"].most_common(8)))
    if s["failed_lookups"]:
        print("Lookups that came back 'not found' (usually a wrong or invented ID):",
              dict(s["failed_lookups"].most_common(8)))


# --- HTML log page -------------------------------------------------------------------------------

CSS = """
:root{--bg:#fbfbfa;--card:#fff;--ink:#1f1f1e;--muted:#6b6a63;--line:#e6e5e0;--agent:#e8f1fb;--cust:#f3f2ee;
--tool:#f6f6f2;--bad:#c0392b;--good:#1e8449;--warn:#b9770e;--code:#f0efe9}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--card:#232322;--ink:#f0efec;--muted:#a8a79f;--line:#3a3a37;
--agent:#1f2f43;--cust:#2b2b29;--tool:#262624;--bad:#ef6f5e;--good:#58c27d;--warn:#e0a84a;--code:#2d2d2a}}
body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px;max-width:1100px;margin:auto}
h1{font-size:24px;margin:0 0 4px} h2{font-size:18px;margin:28px 0 8px} .muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;margin:8px 0} td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}
details.conv{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:8px 12px}
summary{cursor:pointer} .tag{display:inline-block;font-size:12px;border:1px solid var(--line);border-radius:10px;padding:0 8px;margin:0 4px 0 0}
.pass{color:var(--good);font-weight:600} .fail{color:var(--bad);font-weight:600}
.msg{border-radius:8px;padding:6px 10px;margin:6px 0;white-space:pre-wrap;word-break:break-word}
.assistant{background:var(--agent);margin-right:12%} .user{background:var(--cust);margin-left:12%} .tool{background:var(--tool);font-size:13px}
.role{font-size:12px;color:var(--muted);display:block} code,pre{background:var(--code);border-radius:4px;font-size:13px}
pre{padding:6px 8px;white-space:pre-wrap;word-break:break-word;margin:4px 0}
.raw{border-left:3px solid var(--line);padding-left:8px;margin:4px 0} input{font:inherit;padding:4px 8px;width:260px}
"""


def _msg_html(m: dict) -> str:
    role = m["role"]
    who = {"assistant": "Aya (agent)", "user": "Customer", "tool": "Tool result"}.get(role, role)
    parts = []
    if m.get("content"):
        txt = m["content"] if role != "tool" else m["content"][:1500]
        parts.append(html.escape(str(txt)))
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function") or tc
        parts.append(f"<pre>{html.escape(fn.get('name', ''))}({html.escape(str(fn.get('arguments', '')))})</pre>")
    return f'<div class="msg {role}"><span class="role">{who}</span>{"".join(parts)}</div>'


def write_html(name: str, recs: list[dict], s: dict, dst: pathlib.Path) -> None:
    rows = "".join(f"<tr><td>{t}</td><td>{s['main'][t]}</td><td>{s['any'][t]}</td></tr>" for t in ORDER + NOTES if s["any"][t])
    convs = []
    for r in sorted(recs, key=lambda r: (r["outcome"]["passed"], r["domain"], r["task_id"])):
        tags = tags_for(r)
        main = main_reason(tags) if not r["outcome"]["passed"] else None
        badge = '<span class="pass">PASS</span>' if r["outcome"]["passed"] else '<span class="fail">FAIL</span>'
        tagh = "".join(f'<span class="tag">{html.escape(t)}</span>' for t in tags)
        checks = html.escape(json.dumps(r["outcome"].get("checks")))
        inv = _current_unseen(r)
        raw = "".join(f'<div class="raw"><span class="role">turn {t["turn"]}: {html.escape(t["kind"])}</span>'
                      f'<pre>{html.escape(t["reply"])}</pre></div>' for t in r.get("raw_turns") or [])
        convs.append(
            f'<details class="conv" data-text="{html.escape(" ".join([r["domain"], r["task_id"], *tags]).lower())}">'
            f'<summary>{badge} <b>{r["domain"]}:{html.escape(r["task_id"][:70])}</b> '
            f'{"— main reason: <b>" + html.escape(main) + "</b> " if main else ""}{tagh}</summary>'
            f'<p class="muted">checks {checks} · ended: {html.escape(str(r.get("termination")))} · '
            f'{(r.get("diagnostics") or {}).get("tools_ran", 0)} tool calls'
            f'{" · invented: " + html.escape(", ".join(inv)) if inv else ""}</p>'
            + "".join(_msg_html(m) for m in r["messages"])
            + (f"<details><summary>Aya's raw replies ({len(r['raw_turns'])})</summary>{raw}</details>" if raw else "")
            + "</details>")
    doms = " · ".join(f"{d} {k}/{n}" for d, (k, n) in s["by_domain"].items())
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(name)} review</title><style>{CSS}</style></head><body>
<h1>{html.escape(name)}: where it goes wrong</h1>
<p class="muted">{s['passed']}/{s['n']} passed · {doms}. Failures first. Click a conversation to read it; "Aya's raw
replies" shows exactly what the model wrote before our parser read it.</p>
<h2>Problems</h2><table><tr><th>Problem</th><th>Main reason for (failures)</th><th>Seen in (all conversations)</th></tr>{rows}</table>
<h2>Conversations</h2><input id="q" placeholder="Filter: e.g. invented, telecom, retail:41">
{''.join(convs)}
<script>document.getElementById('q').addEventListener('input',e=>{{const q=e.target.value.toLowerCase();
document.querySelectorAll('details.conv').forEach(d=>d.style.display=d.dataset.text.includes(q)?'':'none')}})</script>
</body></html>"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(page)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="results/fixes/aya.jsonl", type=pathlib.Path)
    ap.add_argument("--html", type=pathlib.Path, help="default: results/reports/<name>_review.html")
    a = ap.parse_args()
    recs = [json.loads(x) for x in a.path.read_text().splitlines() if x.strip()]
    recs = [r for r in recs if r.get("attempt", 1) == 1]
    name = a.path.stem.removeprefix("sample-")
    s = summarise(recs)
    print_summary(name, s)
    checked, with_calls, leaked = view_check(recs)
    if checked:
        print(f"\nAgent's view check (telecom): {checked} conversations, {with_calls} with customer phone "
              f"actions; the agent was shown them in {leaked}. "
              + ("✅ the agent never sees them: train on agent_view, not messages."
                 if not leaked else "⚠ the agent does see them: check before converting."))
    dst = a.html or pathlib.Path(f"results/reports/{name}_review.html")
    write_html(name, recs, s, dst)
    print(f"\nLog page: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
