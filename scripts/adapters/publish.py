"""Publish an adapter's model card, provenance and eval metrics to its Hub repo (SEL-22).

    just adapter-publish sft-v2 --dry-run                # writes results/adapters/sft-v2/, uploads nothing
    just adapter-publish sft-v2                          # uploads them and records the revision
    just adapter-publish sft-v2 --summary runs/sft-aya32b-r16-v2/run_summary.json

Adds to the repo the adapter weights are already in (`train_sft.py --push`): ``README.md`` (the
model card), ``lexsi_provenance.json`` (registry entry, code commit, the run's summary) and the
adapter's ``metrics`` files under ``eval/`` (cleared first, so only the registry's current list
goes up). The card's status comes from `adapters.status_of`: ``results/adapters/state.json``
decides which adapter is Arm B. The upload's commit is recorded in ``state.json`` as the
adapter's revision, which ``just adapter-use`` pins.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from reflexive import adapters as ad
from reflexive.files import ROOT, shown


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("id", help="the adapter's id in configs/adapters.yaml, e.g. sft-v2")
    p.add_argument(
        "--summary", type=Path, help="its run_summary.json (default: runs/<repo>/run_summary.json)"
    )
    p.add_argument("--dry-run", action="store_true", help="write the files locally, upload nothing")
    a = p.parse_args()

    reg = ad.load_registry()
    try:
        adapter = ad.get_adapter(reg, a.id)
    except ad.RegistryError as e:
        print(f"⛔ {e}")
        return 1
    summary_path = a.summary or ROOT / "runs" / adapter["repo"] / "run_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    if summary is None:
        print(f"   ! no {shown(summary_path)}: the card says the run's summary wasn't kept")

    out = ROOT / "results" / "adapters" / a.id
    out.mkdir(parents=True, exist_ok=True)
    prov = ad.provenance(reg, adapter, summary, ad.git_state(), ad.status_of(adapter, ad.load_state()))
    if prov["code_commit"] is None:
        print("   ! the run's summary has no training commit: the card says it wasn't recorded")
    (out / "lexsi_provenance.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False) + "\n")
    (out / "README.md").write_text(ad.model_card(reg, adapter, prov))
    shutil.rmtree(out / "eval", ignore_errors=True)  # only this publish's metrics files go up
    (out / "eval").mkdir()
    for m in adapter.get("metrics", []):
        if (ROOT / m).exists():
            shutil.copy(ROOT / m, out / "eval" / Path(m).name)
        else:
            print(f"   ! {m} isn't here: not uploaded")
    n_eval = len(list((out / "eval").iterdir()))
    print(f"wrote {shown(out)}: README.md, lexsi_provenance.json, eval/ ({n_eval} files)")
    if a.dry_run:
        print("(dry run: nothing uploaded)")
        return 0

    from huggingface_hub import HfApi

    repo = ad.hub_repo(reg, adapter)
    info = HfApi().upload_folder(
        folder_path=str(out), repo_id=repo, commit_message=f"Model card and provenance for {a.id} (SEL-22)"
    )
    state = ad.load_state()
    state.setdefault("revisions", {})[a.id] = info.oid
    ad.save_state(state)
    print(f"uploaded to {repo} at revision {info.oid[:10]}; recorded in {shown(ad.STATE)} (commit it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
