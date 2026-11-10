"""Switch which adapter is Arm B, or roll back to the one before (SEL-22).

    just adapter-use sft-v2 --why "declared Arm B, 10 Nov"   # record it, fetch its pinned revision
    just adapter-rollback --why "G failed its test rule"      # back to the adapter before the last switch
    ... --server http://localhost:8000                        # also load it into the running vLLM

The switch is recorded in ``results/adapters/state.json`` (``current`` and ``history``: from, to,
revision, why, when), the only record of which adapter is Arm B (the registry says what was
decided about each, never which is current). The adapter is downloaded at the revision
``just adapter-publish`` recorded (else the repo's latest) into ``runs/hub/<repo>``, apart from
``runs/<repo>``, where train_sft.py writes, so a fetch never overwrites a local training run. With
``--server`` it's loaded into the running vLLM as ``current`` without a restart (serve.sh enables
vLLM's runtime LoRA loading), so ``configs/sft/arms.current.yaml`` always runs whichever adapter
is current.

**Refused:** switching to the adapter already current (a rollback would then go nowhere), and to
a ``not_adopted`` or ``control`` adapter unless ``--force`` (e.g. a drill).

**Loading into vLLM** checks the new adapter first, under a temporary name, so a bad path or a
rank above the server's ``--max-lora-rank`` fails with the old ``current`` still served. Only then
is ``current`` swapped; if that last load still fails, the old adapter is loaded back. Nothing is
recorded unless the switch fully worked.

**The server must have LoRA on:** serve.sh enables it (and sets the largest rank) only when it
starts with at least one adapter, so start it with ``EXTRA_ADAPTERS="current=runs/<repo>"``; a
base-only server refuses the load. **Rollback** undoes the last switch, so two rollbacks in a row
return to where you started; ``just adapter-use <id>`` goes to any adapter. The state file is
written where this runs: on a pod, commit ``results/adapters/state.json`` from there.
"""

from __future__ import annotations

import argparse
import getpass
from pathlib import Path
from typing import Any

from reflexive import adapters as ad
from reflexive.files import ROOT, shown

SERVED_NAME = "current"
STAGING_NAME = "current-check"


def local_dir(adapter: dict[str, Any]) -> Path:
    """Where an adapter's weights are on this machine: the Hub download, else the training run."""
    hub = ROOT / "runs" / "hub" / adapter["repo"]
    return hub if hub.exists() else ROOT / "runs" / adapter["repo"]


def _post(server: str, what: str, body: dict[str, Any]) -> Any:
    import httpx

    return httpx.post(f"{server}/v1/{what}_lora_adapter", json=body, timeout=600)


def _unload(server: str, name: str) -> None:
    """Unload ``name``; an adapter that isn't loaded is fine, any other refusal stops the swap."""
    r = _post(server, "unload", {"lora_name": name})
    text = r.text.lower()
    missing = r.status_code == 404 or "not found" in text or "cannot be found" in text
    if r.status_code != 200 and not missing:
        raise SystemExit(f"⛔ vLLM refused to unload {name!r} ({r.status_code}): {r.text[:300]}")


def load_into_server(server: str, path: str, old_path: str | None = None) -> None:
    """Serve ``path`` as ``current``: check it under a temporary name, then swap (see the top)."""
    r = _post(server, "load", {"lora_name": STAGING_NAME, "lora_path": path})
    if r.status_code != 200:
        raise SystemExit(
            f"⛔ vLLM refused the adapter ({r.status_code}): {r.text[:300]}\n   {SERVED_NAME!r} is unchanged"
        )
    _unload(server, STAGING_NAME)
    _unload(server, SERVED_NAME)
    r = _post(server, "load", {"lora_name": SERVED_NAME, "lora_path": path})
    if r.status_code == 200:
        return
    back = _post(server, "load", {"lora_name": SERVED_NAME, "lora_path": old_path}) if old_path else None
    ok = back is not None and back.status_code == 200
    restored = "the old adapter is loaded back" if ok else "NOTHING is served as current"
    raise SystemExit(f"⛔ vLLM refused the adapter ({r.status_code}): {r.text[:300]}\n   {restored}")


def target(a: argparse.Namespace, reg: dict[str, Any], state: dict[str, Any]) -> tuple[dict | None, str]:
    """The adapter to make current, or None and why not."""
    to = ad.previous(state) if a.rollback else a.id
    if to is None:
        return (
            None,
            "nothing to roll back to: no earlier switch is recorded" if a.rollback else "name an adapter",
        )
    try:
        adapter = ad.get_adapter(reg, to)
    except ad.RegistryError as e:
        return None, str(e)
    problems = ad.check_state(reg, state)
    if problems:
        return None, "fix results/adapters/state.json first:\n  " + "\n  ".join(problems)
    if to == state.get("current"):
        return None, f"{to} is already current: nothing to switch"
    if adapter["status"] not in ad.SWITCHABLE and not a.force:
        return None, f"{to} is {adapter['status']}, not a candidate for Arm B (--force for a drill)"
    return adapter, ""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("id", nargs="?", help="the adapter to make current (omit with --rollback)")
    p.add_argument(
        "--rollback", action="store_true", help="back to the adapter current before the last switch"
    )
    p.add_argument("--why", required=True, help="the reason, kept in the history")
    p.add_argument("--server", help="also load it into this running vLLM, e.g. http://localhost:8000")
    p.add_argument("--no-download", action="store_true", help="record the switch only")
    p.add_argument("--force", action="store_true", help="allow a not_adopted or control adapter")
    p.add_argument("--by", default=getpass.getuser(), help="who switched (default: the machine's user)")
    a = p.parse_args()

    reg, state = ad.load_registry(), ad.load_state()
    adapter, why = target(a, reg, state)
    if adapter is None:
        print(f"⛔ {why}")
        return 1
    to = adapter["id"]
    revision = state.get("revisions", {}).get(to)
    local = ROOT / "runs" / "hub" / adapter["repo"]
    if not a.no_download:
        from huggingface_hub import snapshot_download

        snapshot_download(ad.hub_repo(reg, adapter), revision=revision, local_dir=local)
        print(f"fetched {ad.hub_repo(reg, adapter)} at {revision or 'latest'} into {shown(local)}")
    else:
        local = local_dir(adapter)
    if a.server:
        old = ad.adapters_by_id(reg).get(state.get("current") or "")
        load_into_server(a.server, str(local), str(local_dir(old)) if old else None)
        print(f"vLLM at {a.server} now serves {to} as {SERVED_NAME!r}")
    ad.save_state(ad.switch(state, to, a.why, by=a.by))
    print(f"current: {state.get('current')} → {to}; recorded in {shown(ad.STATE)} (commit it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
