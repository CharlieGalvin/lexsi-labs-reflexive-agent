# ReAgent task runner. Install just once: `brew install just` (or `uv tool install rust-just`).
# Not `pip install just`: that is an unrelated Python package with the same name.
# Run `just` to list every command.

set dotenv-load := true                       # loads API keys from .env automatically
export PATH := justfile_directory() + "/.venv/bin:" + env_var("PATH")

tau2_version := "v2.0.0"

# List all commands
default:
    @just --list --unsorted

# --- Setup -------------------------------------------------------------------

# Create .venv with Python 3.12 (needs uv: `brew install uv`)
venv:
    @command -v uv >/dev/null || { echo "Install uv first: brew install uv"; exit 1; }
    @test -x .venv/bin/python || uv venv --python 3.12 .venv
    @python -V

# Laptop setup: τ²-bench (with data) + lint and test tools. No GPU needed.
setup-dev: venv tau2
    uv pip install --python .venv/bin/python -e ".[dev,bench]"
    @test -f .env || { cp .env.example .env; echo "Created .env: add OPENAI_API_KEY (needed for retail/airline scoring) or OPENROUTER_API_KEY"; }

# τ²-bench, pinned, WITH its data folder (sparse clone into third_party/, ~340 MB, skips past results)
tau2:
    @test -d third_party/tau2-bench || { \
      git clone -q --depth 1 --branch {{tau2_version}} --filter=blob:none --no-checkout \
        https://github.com/sierra-research/tau2-bench.git third_party/tau2-bench && \
      git -C third_party/tau2-bench sparse-checkout set --no-cone '/*' '!/data/tau2/results/' '!/web/' '!/figs/' && \
      git -C third_party/tau2-bench checkout -q {{tau2_version}}; }
    uv pip uninstall --python .venv/bin/python tau2 >/dev/null 2>&1 || true
    uv pip install --python .venv/bin/python -e "third_party/tau2-bench[gym]"

# GPU box: everything in setup-dev, plus training libraries and AgentTune (environment only)
setup: setup-dev
    uv pip install --python .venv/bin/python -e ".[train]"
    @test -d third_party/AgentTune || git clone --depth 1 https://github.com/Lexsi-Labs/AgentTune.git third_party/AgentTune
    uv pip install --python .venv/bin/python -e third_party/AgentTune

# --- Checks ------------------------------------------------------------------

# Run the test suite
test:
    python -m pytest -q

# The plain-English specifications (tests/features/*.feature), with each scenario's result
features:
    python -m pytest tests/test_features_*.py --gherkin-terminal-reporter -v -p no:cacheprovider

# Lint the code
lint:
    python -m ruff check src scripts tests

# Lint + tests (what CI runs)
check: lint test

# --- τ²-bench ----------------------------------------------------------------

# Build golden trajectories v1 from the teacher's fix run (SEL-9; no API calls)
golden *args:
    python scripts/data/build_golden.py {{args}}

# Check data/golden/v1/, write the data card and upload it PRIVATE to the team HF org, tag v1 (SEL-48; --dry-run first)
golden-publish *args:
    python scripts/data/publish_golden.py {{args}}

# Confirm configs/splits.json matches a rebuild from its seed (SEL-36)
splits-check:
    python scripts/data/make_splits.py --check

# Spot-check the Tool/Arg/Loop classifier on saved rollouts (SEL-12; no API calls)
hallucination-spot-check *files="results/episodes/arms/A-telecom-train.jsonl results/episodes/arms/C-telecom-train.jsonl":
    python scripts/arms/spot_check_hallucination.py {{files}} --labels docs/experiments/hallucination_labels.json

# Re-save τ²-bench's tool schemas for the classifier (configs/tool_schemas.json; --check to verify)
tool-schemas *args:
    python scripts/tau2/dump_tool_schemas.py {{args}}

# Splits, prompt sizes and context fit per domain (no API calls)
tau2-inspect:
    python scripts/tau2/inspect_tasks.py

# First live run: 2 train tasks + score and cost (domain: retail, airline or telecom)
tau2-first-run domain="retail":
    DOMAIN={{domain}} bash scripts/tau2/first_run.sh

# 5 fixed telecom tasks (3 state-check, 2 also check actions) with GPT-4.1, ~$0.70
scorer_tasks := "[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None] [mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None] [mobile_data_issue]airplane_mode_on|bad_network_preference[PERSONA:Hard] [service_issue]lock_sim_card_pin|overdue_bill_suspension[PERSONA:Easy] [service_issue]contract_end_suspension|lock_sim_card_pin[PERSONA:Hard]"

# SEL-39: run 5 telecom tasks (~$0.70), then check our scorer agrees with τ²-bench
tau2-scorer-run:
    DOMAIN=telecom TASK_IDS='{{scorer_tasks}}' OUT=tau2-scorer-check bash scripts/tau2/first_run.sh
    python scripts/tau2/check_scorer.py run tau2-scorer-check

# Re-score a saved run with our scorer and compare with the official score (free)
tau2-check-scorer name="tau2-scorer-check":
    python scripts/tau2/check_scorer.py run {{name}}

# SEL-32: GPT-4.1 drives our conversation wrapper on the same 5 telecom tasks (~$0.70)
tau2-episode-check:
    python scripts/tau2/episode_check.py telecom --task-ids '{{scorer_tasks}}'

# Replay every task's correct actions through our scorer: all must pass (no API, ~10 min)
tau2-check-gold *domains:
    python scripts/tau2/check_scorer.py gold {{domains}}

# --- Archived: Aya Expanse 32B as the teacher (3 Nov; Aya is now the student) ---
# Kept for the record (docs/TEACHER_SETUP.md, history). Needs COHERE_API_KEY.

# Does a Cohere model do native tool calling? 3 API calls (default: the teacher; or tiny-aya-global)
teacher-probe model="c4ai-aya-expanse-32b":
    python scripts/archive/teacher_probe.py --model {{model}}

# Teacher on 3 tasks (one per domain). style: cohere (Aya's own) | agenttune; hints: none | steps | steps_with_args
teacher-check style="cohere" hints="none":
    python scripts/archive/teacher_check.py --style {{style}} --hints {{hints}}

# Is Aya a usable teacher? Same 30 train tasks for Aya and a reference (default: Qwen3 32B). ~$5
teacher-sample *models="aya qwen":
    python scripts/teacher/bakeoff.py --models {{models}}

# Fix run: the teacher (default GLM-5.3-Flash) on the 148 train tasks phase 1 didn't use, 3 attempts,
# judges on. ~$13. Try `just teacher-fixes --limit 2` first; `--report` summarises what's saved.
teacher-fixes *args:
    python scripts/teacher/fixes.py {{args}}

# Where does a model go wrong? Failure categories + a browsable log page (results/reports/)
student-review path="results/fixes/aya.jsonl":
    python scripts/student/review.py {{path}}

# Bake-off statistics and projected cost from the saved runs in results/episodes (no API calls)
bakeoff-report:
    python scripts/teacher/bakeoff_report.py

# --- Experiment arms (configs/arms.yaml) ----------------------------------------

# Run an arm on a domain's split, e.g. `just rollout A retail --limit 5` or `just rollout C telecom`
rollout arm domain *args:
    python scripts/arms/rollout.py {{arm}} {{domain}} {{args}}

# Experiment commands (train, evaluate…) are added by the Architect and Trainer.
