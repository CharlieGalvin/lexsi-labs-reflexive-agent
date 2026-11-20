"""Run the retention benchmarks on one model (SEL-84), through AuditKIT's lm-eval engine.

    just retention-run base                 # every benchmark in configs/retention.yaml
    just retention-run sft2 --limit 5       # a pilot: 5 items per task, to check the plumbing

Needs, on the pod: our vLLM server holding the model (see configs/retention.yaml), and
``uv pip install "auditkit[lmeval]" "lm-eval[api,ifeval]"``. Writes
results/retention/<label>.json (AuditKIT's full RunResult: every item's answer and score), or
<label>-pilot.json with --limit. Then `just retention-compare base sft2`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("label", help="a model label from the config's models: (e.g. base, sft2)")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "retention.yaml")
    p.add_argument("--limit", type=int, help="items per task: a pilot, saved apart from the real run")
    p.add_argument("--base-url", help="override the config's server URL")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "retention")
    a = p.parse_args(argv)

    cfg = yaml.safe_load(a.config.read_text())
    if a.label not in cfg["models"]:
        p.error(f"no model {a.label!r} in {a.config} (have: {', '.join(cfg['models'])})")
    from auditkit.lmeval_engine import run_benchmark  # the pod's env only (lm-eval is heavy)
    from auditkit.runspec import RunConfig

    d = cfg["decoding"]
    tasks = [t for b in cfg["benchmarks"] for t in b["tasks"]]
    out = a.out_dir / f"{a.label}{'-pilot' if a.limit else ''}.json"
    print(
        f"{a.label} ({cfg['models'][a.label]}): {len(tasks)} tasks"
        f"{f', {a.limit} items each' if a.limit else ''} → {out}"
    )
    t0 = time.time()
    result = run_benchmark(
        tasks,
        f"api:{cfg['models'][a.label]}",
        config=RunConfig(limit=a.limit, seed=d["seed"], temperature=d["temperature"], batch_size=1),
        run_name=f"retention-{a.label}",
        base_url=a.base_url or cfg["base_url"],
        tokenizer=cfg["tokenizer"],
        apply_chat_template=d["apply_chat_template"],
        fewshot_as_multiturn=d["fewshot_as_multiturn"],
        gen_kwargs=f"temperature={d['temperature']},do_sample=False",
        model_args={"num_concurrent": d["concurrent"]},  # requests in flight to our server
        lmeval_kwargs={
            "numpy_random_seed": d["seed"],
            "torch_random_seed": d["seed"],
        },  # AuditKIT sets random_seed
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(out), write_provenance=False)
    scored = sum(x.correct is not None for x in result.predictions)
    print(f"done in {(time.time() - t0) / 60:.1f} min: {len(result.predictions)} items, {scored} scored")
    for k, v in sorted(result.headline.items()):
        print(f"  {k:<50} {v:.3f}")
    if result.errors:
        print(f"⚠ {len(result.errors)} errors (see {out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
