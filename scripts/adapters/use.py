"""Switch which adapter is Arm B, or roll back to the one before (SEL-22).

    just adapter-use sft-v2 --why "declared Arm B, 10 Nov"   # record it, fetch its pinned revision
    just adapter-rollback --why "G failed its test rule"      # back to the adapter before the last switch
    ... --server http://localhost:8000                        # also load it into the running vLLM

The switch is recorded in ``results/adapters/state.json`` (``current`` and ``history``: from, to,
revision, why, when). The adapter is downloaded at the revision ``just adapter-publish`` recorded
(else the repo's latest) into ``runs/<repo>``. With ``--server`` it's loaded into the running vLLM
as ``current`` without a restart (serve.sh enables vLLM's runtime LoRA loading), so the arm config
``configs/sft/arms.current.yaml`` always runs whichever adapter is current.
"""

from __future__ import annotations

import argparse
import getpass

from reflexive import adapters as ad
from reflexive.files import ROOT, shown

SERVED_NAME = "current"


def load_into_server(server: str, path: str) -> None:
    """Swap the adapter vLLM serves as ``current`` (unload, then load: same name, new weights)."""
    import httpx

    httpx.post(
        f"{server}/v1/unload_lora_adapter", json={"lora_name": SERVED_NAME}, timeout=60
    )  # may not exist yet
    r = httpx.post(
        f"{server}/v1/load_lora_adapter", json={"lora_name": SERVED_NAME, "lora_path": path}, timeout=600
    )
    if r.status_code != 200:
        raise SystemExit(f"⛔ vLLM refused the adapter ({r.status_code}): {r.text[:300]}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("id", nargs="?", help="the adapter to make current (omit with --rollback)")
    p.add_argument(
        "--rollback", action="store_true", help="back to the adapter current before the last switch"
    )
    p.add_argument("--why", required=True, help="the reason, kept in the history")
    p.add_argument("--server", help="also load it into this running vLLM, e.g. http://localhost:8000")
    p.add_argument("--no-download", action="store_true", help="record the switch only")
    a = p.parse_args()

    reg, state = ad.load_registry(), ad.load_state()
    by_id = ad.adapters_by_id(reg)
    to = ad.previous(state) if a.rollback else a.id
    if to is None:
        print(
            "⛔ nothing to roll back to: no earlier switch is recorded"
            if a.rollback
            else "⛔ name an adapter"
        )
        return 1
    if to not in by_id:
        print(f"⛔ no adapter {to!r}; the registry has {sorted(by_id)}")
        return 1
    adapter = by_id[to]
    revision = state.get("revisions", {}).get(to)
    local = ROOT / "runs" / adapter["repo"]
    if not a.no_download:
        from huggingface_hub import snapshot_download

        snapshot_download(ad.hub_repo(reg, adapter), revision=revision, local_dir=local)
        print(f"fetched {ad.hub_repo(reg, adapter)} at {revision or 'latest'} into {shown(local)}")
    if a.server:
        load_into_server(a.server, str(local))
        print(f"vLLM at {a.server} now serves {to} as {SERVED_NAME!r}")
    ad.save_state(ad.switch(state, to, a.why, by=getpass.getuser()))
    print(f"current: {state.get('current')} → {to}; recorded in {shown(ad.STATE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
