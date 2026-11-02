CONFIG ?= configs/grpo_efficiency.yaml
RUN    ?= grpo-eff-v0

.PHONY: setup setup-dev test lint smoke baseline train after evaluate compare

setup:            ## full GPU environment (Python 3.12+)
	pip install -e ".[train,dev]"
	[ -d third_party/AgentTune ] || git clone https://github.com/Lexsi-Labs/AgentTune.git third_party/AgentTune
	pip install -e third_party/AgentTune

setup-dev:        ## laptop: rewards, metrics, tests, evaluation only (no torch)
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src scripts tests

smoke:            ## 10-step training run to check the plumbing end to end
	python scripts/train.py --config $(CONFIG) train.max_steps=10 train.output_dir=runs/smoke train.report_to=none

baseline:
	python scripts/collect.py --config $(CONFIG) --out results/base.jsonl

train:
	python scripts/train.py --config $(CONFIG)

after:
	python scripts/collect.py --config $(CONFIG) --adapter runs/$(RUN) --out results/$(RUN).jsonl

evaluate:
	python scripts/evaluate.py results/*.jsonl

compare:
	python scripts/compare.py results/base.metrics.json results/$(RUN).metrics.json
