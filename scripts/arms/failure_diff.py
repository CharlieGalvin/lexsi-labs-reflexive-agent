"""What the agent did against what τ²-bench expected, for every failed conversation in rollout files.

    just arm-failures results/episodes/arms/S-*-dev.jsonl
    just arm-failures results/episodes/arms/S-retail-dev.jsonl --all     # passes too

For each failed task: the checks it missed, the customer's request, then the expected actions that
change state (the ones the DB check compares): done, done with different arguments, or never done;
and the agent's own state-changing calls that weren't expected. Reads (lookups) are left out.

Telecom: the customer's own actions on the phone aren't in the record (only the agent's side
is), so for those the report says whether the agent's messages mention the action, a hint, not a
check; the ENV assertions are what τ²-bench tested on the phone at the end.

A summary across all failures counts the actions most often missed, done wrongly, or added.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from reflexive.files import read_jsonl, shown


def latest(path: Path) -> list[dict]:
    """The last record per (task, sample): a re-run of a crashed conversation replaces it."""
    keep: dict[tuple, dict] = {}
    for r in read_jsonl(path, partial_last_ok=True):
        keep[(r["task_id"], r.get("sample", 0))] = r
    return list(keep.values())


def agent_calls(messages: list[dict]) -> list[tuple[str, dict]]:
    """The agent's tool calls, in order, as (name, arguments)."""
    out = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc)
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"<unparsed>": args}
            out.append((fn.get("name", "?"), args))
    return out


def agent_text(messages: list[dict]) -> str:
    return "\n".join(m.get("content") or "" for m in messages if m.get("role") == "assistant").lower()


# Words an agent uses when it asks the customer to do a phone action (telecom), for the hint
_SAYS = {
    "toggle_airplane_mode": ("airplane", "flight mode"),
    "set_network_mode_preference": ("network mode", "network preference", "4g", "5g"),
    "toggle_wifi_calling": ("wi-fi calling", "wifi calling"),
    "reset_apn_settings": ("apn",),
    "reboot_device": ("restart", "reboot", "turn your phone off"),
    "grant_app_permission": ("permission",),
    "toggle_data": ("mobile data", "cellular data", "data on", "turn on data", "toggle data"),
    "toggle_data_saver_mode": ("data saver",),
    "reseat_sim_card": ("sim",),
    "toggle_roaming": ("roaming",),
}


def mentioned(name: str, text: str) -> bool:
    """Whether the agent's messages seem to ask for this customer action: a hint, not a check."""
    words = _SAYS.get(name, (name.replace("_", " "), " ".join(name.split("_")[1:])))
    return name in text or any(w and w in text for w in words)


def compare(expected: list[tuple[str, dict]], done: list[tuple[str, dict]]) -> dict[str, list]:
    """Match expected state-changing calls to the agent's: exact first, then same tool, other args.

    Returns ``ok`` (expected, done as expected), ``wrong`` ((expected, what was done instead)),
    ``missing`` (expected, never called) and ``extra`` (called, not expected)."""
    left = list(done)
    ok, unmatched = [], []
    for e in expected:
        if e in left:
            left.remove(e)
            ok.append(e)
        else:
            unmatched.append(e)
    wrong, missing = [], []
    for e in unmatched:
        same = next((d for d in left if d[0] == e[0]), None)
        if same is not None:
            left.remove(same)
            wrong.append((e, same))
        else:
            missing.append(e)
    return {"ok": ok, "wrong": wrong, "missing": missing, "extra": left}


def args_diff(want: dict, got: dict) -> str:
    keys = sorted(set(want) | set(got))
    parts = [f"{k}: {want.get(k)!r} → {got.get(k)!r}" for k in keys if want.get(k) != got.get(k)]
    return "; ".join(parts) or "(same arguments, different order)"


def call(name: str, args: dict, width: int = 110) -> str:
    s = f"{name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"
    return s if len(s) <= width else s[: width - 1] + "…"


def report(path: Path, show_all: bool, totals: dict[str, Counter]) -> list[str]:  # noqa: C901
    from tau2.registry import registry

    from reflexive.outcome import scorer

    rows = latest(path)
    if not rows:
        return [f"{shown(path)}: no records"]
    domain = rows[0]["domain"]
    sc = scorer(domain)
    env = registry.get_env_constructor(domain)(**(sc.env_kwargs or {}))

    def mutating(name: str) -> bool:
        try:
            return bool(env._is_mutating_tool(name))
        except Exception:  # noqa: BLE001 - an unknown tool (a hallucination) changes nothing
            return False

    failed = [r for r in rows if show_all or not r["outcome"].get("passed")]
    lines = [f"# {shown(path)}: {len(failed)} of {len(rows)} shown ({domain})", ""]
    for r in sorted(failed, key=lambda r: str(r["task_id"])):
        task = sc.task(r["task_id"])
        ec = task.evaluation_criteria
        missed = [k for k, v in (r["outcome"].get("checks") or {}).items() if v < 1] or ["none recorded"]
        verdict = "PASS" if r["outcome"].get("passed") else "fail"
        lines.append(
            f"## {r['task_id']} (sample {r.get('sample', 0)}): {verdict}, {r.get('termination')}; "
            f"missed: {', '.join(missed)}"
        )
        instr = getattr(task.user_scenario, "instructions", None)
        why = getattr(instr, "reason_for_call", None) or str(instr or "")
        lines.append(f"Customer: {' '.join(why.split())[:300]}")

        exp_agent = [(a.name, a.arguments or {}) for a in ec.actions or [] if a.requestor != "user"]
        exp_user = [(a.name, a.arguments or {}) for a in ec.actions or [] if a.requestor == "user"]
        want = [e for e in exp_agent if mutating(e[0])]
        did = [d for d in agent_calls(r["messages"]) if mutating(d[0])]
        c = compare(want, did)
        for e in c["ok"]:
            lines.append(f"  ✓ {call(*e)}")
        for e, d in c["wrong"]:
            lines.append(f"  ≠ {call(*e)}\n      did: {args_diff(e[1], d[1])}")
            totals["wrong"][e[0]] += 1
        for e in c["missing"]:
            lines.append(f"  ✗ never done: {call(*e)}")
            totals["missing"][e[0]] += 1
        for d in c["extra"]:
            lines.append(f"  + not expected: {call(*d)}")
            totals["extra"][d[0]] += 1
        if not (want or did):
            lines.append("  (no state-changing agent actions expected or taken)")
        if exp_user:
            text = agent_text(r["messages"])
            for name, args in exp_user:
                hint = "mentioned" if mentioned(name, text) else "NOT mentioned"
                lines.append(f"  customer should: {call(name, args)}: {hint} by the agent")
                if hint == "NOT mentioned":
                    totals["customer step not mentioned"][name] += 1
        for a in ec.env_assertions or []:
            lines.append(
                f"  phone check: {a.func_name}({', '.join(f'{k}={v!r}' for k, v in a.arguments.items())})"
            )
        lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path, help="rollout files (results/episodes/arms/*.jsonl)")
    ap.add_argument("--all", action="store_true", help="show passed conversations too")
    ap.add_argument("--out", type=Path, help="also write the report here (Markdown)")
    a = ap.parse_args()
    for f in a.files:  # e.g. `S2` instead of a path: say so instead of a traceback
        if not f.exists():
            ap.error(f"no such file: {f} (pass rollout files: results/episodes/arms/*.jsonl)")

    totals: dict[str, Counter] = {
        k: Counter() for k in ("missing", "wrong", "extra", "customer step not mentioned")
    }
    lines: list[str] = []
    for f in a.files:
        lines += report(f, a.all, totals)
    lines.append("# Across the files")
    for kind, c in totals.items():
        if c:
            lines.append(f"- {kind}: " + ", ".join(f"{n} ×{k}" for n, k in c.most_common(8)))
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")
        print(f"\n→ {shown(a.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
