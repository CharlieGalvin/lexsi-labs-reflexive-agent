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

# Lint the code and check its formatting
lint:
    python -m ruff check src scripts tests
    python -m ruff format --check src scripts tests

# Format the code (ruff format), then fix what lint can fix by itself
fmt:
    python -m ruff format src scripts tests
    python -m ruff check --fix src scripts tests

# Type checks with ty (Chiara's arms and classifier show as warnings; only errors fail)
types:
    python -m ty check --exit-zero-on-warning

# Lint + types + tests (what CI runs)
check: lint types test

# Advisory, not blocking: types, dead code (vulture) and undeclared dependencies (deptry)
quality: types
    -uvx vulture@2.14 src scripts --min-confidence 80 --exclude scripts/archive
    -uvx deptry@0.23.1 .

# --- τ²-bench ----------------------------------------------------------------

# Build golden trajectories v1 from the teacher's fix run (SEL-9; no API calls)
golden *args:
    python scripts/data/build_golden.py {{args}}

# Check data/golden/v1/ (+ the hard subset unless --tag v1), write the card, upload PRIVATE to the team HF org and tag it (SEL-48, SEL-60; --dry-run first)
golden-publish *args:
    python scripts/data/publish_golden.py {{args}}

# Build the hard subset from Arm A's train runs and golden v1 → data/golden/hard/ (SEL-7; no API calls)
hard-subset *args:
    python scripts/data/build_hard_subset.py {{args}}

# Training examples from golden v1.1: the healable fixes as Aya reads them → data/train/ (SEL-66)
train-data *args:
    python scripts/data/build_train_data.py {{args}}

# Indicative SFT run on the GPU pod (SEL-65): QLoRA on Aya 32B, the agent's turns as targets → runs/
sft-train *args:
    python scripts/sft/train_sft.py {{args}}

# Serve Aya 32B (+ the SFT adapter) with vLLM on the pod, OpenAI-compatible on :8000 (SEL-65)
sft-serve:
    bash scripts/pod/serve.sh

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

# Teacher bake-off, phase 1: models on 30 fixed train tasks (`--models bakeoff`: the slate, ~$26; `--list`: free)
teacher-sample *models="aya qwen":
    python scripts/teacher/bakeoff.py --models {{models}}

# Fix run: GLM-5.3-Flash, up to 3 attempts per train task, judges on (`--tasks all`: 178, $11.20; `--report`: free)
teacher-fixes *args:
    python scripts/teacher/fixes.py {{args}}

# Where does a model go wrong? Failure categories + a browsable log page (results/reports/)
student-review path="results/fixes/aya.jsonl":
    python scripts/student/review.py {{path}}

# Bake-off statistics and projected cost from the saved runs in results/episodes (no API calls)
bakeoff-report:
    python scripts/teacher/bakeoff_report.py

# --- Experiment arms (configs/arms.yaml) ----------------------------------------

# Re-score saved rollouts with the AuditKIT oracle: verified success rate (judges on; `--only dev`, `--judges off`)
oracle-score *files:
    python scripts/arms/oracle_score.py {{files}}
# Summarise a rollout file as results/arm-<arm>-<domain>-<split>.metrics.json (no API calls; `--only dev`)
arm-metrics file *args:
    python scripts/arms/metrics.py {{file}} {{args}}
# Failed conversations: the expected state-changing actions against what the agent did (`--all`, `--out`)
arm-failures +args:
    python scripts/arms/failure_diff.py {{args}}
# Passes per arm, domain and sample: the run-to-run spread (`--split val`)
arm-samples +args:
    python scripts/arms/samples.py {{args}}
# The agreed DevGate on saved dev runs: `just gate-check P S` (a check), `--record` (a heal cycle)
gate-check baseline candidate *args:
    python scripts/arms/gate_check.py {{baseline}} {{candidate}} {{args}}
# SFT v2 data (SEL-71): v1's examples + the teacher's extra verified passes for these domains
sft-v2-data *args="--domains airline":
    python scripts/data/build_extra_fixes.py {{args}}

# Adapter lifecycle (SEL-22): configs/adapters.yaml; state and history in results/adapters/state.json
adapter-publish id *args:
    python scripts/adapters/publish.py {{id}} {{args}}
adapter-use id *args:
    python scripts/adapters/use.py {{id}} {{args}}
adapter-rollback *args:
    python scripts/adapters/use.py --rollback {{args}}

# On-policy teacher fixes (SEL-73): where the student's changes leave the expected ones on train tasks
# (v2's train-pool rollouts are always the input; extra arguments are added, e.g. --dry-run)
onpolicy-fixes *args:
    python scripts/heal/onpolicy_fixes.py results/episodes/arms/S2-retail-trainpool.jsonl results/episodes/arms/S2-airline-trainpool.jsonl results/episodes/arms/S2-telecom-trainpool.jsonl {{args}}

# SFT v4 data (SEL-73): v2's file + the on-policy fixes (group "onpolicy")
sft-v4-data fixes="results/heal/onpolicy-fixes.jsonl":
    cat data/train/sft-v2.jsonl {{fixes}} > data/train/sft-v4.jsonl
    @echo "data/train/sft-v4.jsonl: $(wc -l < data/train/sft-v4.jsonl) examples ($(wc -l < {{fixes}}) on-policy)"

# SFT v3 data (SEL-72): v1's examples + the extra verified retail and airline passes
sft-v3-data *args:
    python scripts/data/build_extra_fixes.py --domains retail airline --out data/train/sft-v3.jsonl {{args}}

# Compare two rollout files with 95% bootstrap CIs (SEL-27): success, hallucination, loops, steps, redundant calls
arm-compare a b *args:
    python scripts/arms/compare.py {{a}} {{b}} {{args}}

# Run an arm on a domain's split, e.g. `just rollout A retail --limit 5` or `just rollout C telecom`
[positional-arguments]  # quoted: telecom task IDs contain | and [ ], which a shell would split
rollout arm domain *args:
    python scripts/arms/rollout.py "$@"

# Experiment commands (train, evaluate…) are added by the Architect and Trainer.

# Heal cycles (SEL-47): rollout → teacher fix → retrain → gate → swap, with an audit record.
# Today `just heal --fixes-only --sample 3` (3 random train tasks per domain); a full cycle needs the vLLM student
heal *args:
    python scripts/heal/cycle.py {{args}}

# --- Training (Trainer, GPU box; docs/experiments/MESSY_LOOKUP.md) --------------

# Run 1 (SEL-34): 10-step Tiny Aya GRPO smoke run with one tool, ~10 min
train-smoke *args:
    python scripts/train_grpo.py configs/train/smoke_add.yaml {{args}}

# GRPO from a config, e.g. `just train configs/train/messy_grpo.yaml --max-steps 3`
train config *args:
    python scripts/train_grpo.py {{config}} {{args}}

# Messy-ID Lookup eval (base model; add `--adapter runs/messy-grpo` for the trained one)
eval-lookup *args:
    python scripts/eval_lookup.py {{args}}

# --- Starter task: Arm A vs one GRPO run (docs/experiments/MESSY_LOOKUP.md) ------

# Arm A: eval the untuned student -> results/base.metrics.json + docs/EXPERIMENTS.md row
baseline *args:
    python scripts/eval_lookup.py --record base --arm A {{args}}

# One GRPO run on the starter task -> runs/<run> (W&B on), e.g. `just after messy-grpo-2`
after run="messy-grpo" *args:
    python scripts/train_grpo.py configs/train/messy_grpo.yaml --output-dir runs/{{run}} {{args}}

# Eval the trained adapter -> results/<run>.metrics.json + docs/EXPERIMENTS.md row
evaluate run="messy-grpo" *args:
    python scripts/eval_lookup.py --adapter runs/{{run}} --record {{run}} {{args}}

# Markdown table of two recorded runs, to post on the ticket (saved to results/<a>-vs-<b>.compare.md)
compare a="base" b="messy-grpo":
    python scripts/experiment_log.py compare {{a}} {{b}}

# Record an eval that already ran, e.g. `just record runs/base-messy/eval_eval.csv base --arm A`
record csv run *args:
    python scripts/experiment_log.py record {{csv}} {{run}} {{args}}
