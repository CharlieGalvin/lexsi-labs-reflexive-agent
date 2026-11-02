# ReAgent task runner. Install just once: `brew install just` (or `uv tool install rust-just`).
# Run `just` to list every command.

set dotenv-load := true                       # loads API keys from .env automatically
export PATH := justfile_directory() + "/.venv/bin:" + env_var("PATH")

config := "configs/grpo_efficiency.yaml"
run := "grpo-eff-v0"

# List all commands
default:
    @just --list --unsorted

# --- Setup -------------------------------------------------------------------

# Create .venv with Python 3.12 (needs uv: `brew install uv`)
venv:
    @command -v uv >/dev/null || { echo "Install uv first: brew install uv"; exit 1; }
    @test -x .venv/bin/python || uv venv --python 3.12 .venv
    @python -V

# Laptop setup: tests, classifier, metrics and τ²-bench. No GPU needed.
setup-dev: venv
    uv pip install --python .venv/bin/python -e ".[dev,bench]"
    @test -f .env || { cp .env.example .env; echo "Created .env: add OPENAI_API_KEY or OPENROUTER_API_KEY"; }

# GPU box: everything in setup-dev, plus training libraries and AgentTune
setup: setup-dev
    uv pip install --python .venv/bin/python -e ".[train]"
    @test -d third_party/AgentTune || git clone --depth 1 https://github.com/Lexsi-Labs/AgentTune.git third_party/AgentTune
    uv pip install --python .venv/bin/python -e third_party/AgentTune

# --- Checks ------------------------------------------------------------------

# Run the test suite
test:
    python -m pytest -q

# Lint the code
lint:
    python -m ruff check src scripts tests

# Lint + tests (what CI runs)
check: lint test

# --- τ²-bench ----------------------------------------------------------------

# Splits, prompt sizes and context fit per domain (no API calls)
tau2-inspect:
    python scripts/tau2_inspect.py

# First live run: 2 train tasks + score and cost (domain: retail, airline or telecom)
tau2-first-run domain="retail":
    DOMAIN={{domain}} bash scripts/tau2_first_run.sh

# --- Experiment --------------------------------------------------------------

# 10 GRPO steps to check the training plumbing end to end
smoke config=config:
    python scripts/train.py --config {{config}} train.max_steps=10 train.output_dir=runs/smoke train.report_to=none

# Arm A: baseline rollouts on the held-out split
baseline config=config:
    python scripts/collect.py --config {{config}} --out results/base.jsonl

# GRPO training with the reward stack
train config=config:
    python scripts/train.py --config {{config}}

# Rollouts with a trained adapter (run = output folder under runs/)
after run=run config=config:
    python scripts/collect.py --config {{config}} --adapter runs/{{run}} --out results/{{run}}.jsonl

# Score every results file (our metrics + AuditKIT)
evaluate:
    python scripts/evaluate.py results/*.jsonl

# Before / after table for a run
compare run=run:
    python scripts/compare.py results/base.metrics.json results/{{run}}.metrics.json
