"""SEL-67: scripts/pod/setup.sh, the RunPod setup. It can't run here (no GPU); these check it's sound."""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from reflexive.files import ROOT

SCRIPT = ROOT / "scripts" / "pod" / "setup.sh"


SCRIPTS = [SCRIPT, ROOT / "scripts" / "pod" / "serve.sh"]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda s: s.name)
def test_the_scripts_are_valid_bash_and_stop_on_the_first_error(script):
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert "set -euo pipefail" in script.read_text()


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck isn't installed")
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda s: s.name)
def test_shellcheck_finds_nothing(script):
    subprocess.run(["shellcheck", str(script)], check=True)


def test_serve_uses_the_context_the_pod_arms_expect():
    import yaml

    text = (ROOT / "scripts" / "pod" / "serve.sh").read_text()
    found = re.search(r'MAX_LEN="\$\{MAX_LEN:-(\d+)\}"', text)
    assert found, "serve.sh sets no default MAX_LEN"
    default = int(found.group(1))
    for f in (
        "arms.base.yaml",
        "arms.sft.yaml",
        "arms.sft2.yaml",
        "arms.sftlong.yaml",
        "arms.sft3.yaml",
        "arms.sft4.yaml",
    ):
        cfg = yaml.safe_load((ROOT / "configs" / "sft" / f).read_text())
        assert cfg["student"]["max_context"] == default


def test_the_just_recipes_and_extras_it_uses_exist():
    text = SCRIPT.read_text()
    justfile = (ROOT / "justfile").read_text()
    for recipe in set(re.findall(r"^\s*just ([a-z-]+)", text, flags=re.M)):
        assert re.search(rf"^{recipe}\b", justfile, flags=re.M), f"no `just {recipe}`"
    assert re.search(r"^sft = \[", (ROOT / "pyproject.toml").read_text(), flags=re.M)


def test_it_refuses_a_mig_slice_and_a_small_gpu_before_installing_anything():
    text = SCRIPT.read_text()
    gpu, install = text.index('step "GPU"'), text.index("\njust setup\n")  # the command, not the header
    assert gpu < install and "MIG" in text[gpu:install] and "MIN_GPU_GB" in text[gpu:install]


def _block(script: str, start: str, end: str) -> str:
    """The lines of a pod script from ``start`` up to (not including) ``end``, to run on their own."""
    lines = (ROOT / "scripts" / "pod" / script).read_text().splitlines()
    i = next(n for n, line in enumerate(lines) if line.startswith(start))
    j = next(n for n, line in enumerate(lines) if n > i and line.startswith(end))
    return "\n".join(lines[i:j])


def _bash(code: str, env: dict[str, str], cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + code],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", **env},
        cwd=cwd,
    )


def test_serve_loads_every_adapter_at_the_largest_rank(tmp_path):
    for name, r in (("v1", 16), ("v2", 32)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "adapter_config.json").write_text(f'{{"r": {r}}}')
    code = _block("serve.sh", "# Every adapter to load", "# Aya Expanse 32B") + '\necho "ARGS ${lora[*]}"'
    base = {"MODEL": "m", "PORT": "8000", "MAX_LEN": "32768"}

    both = _bash(code, {**base, "ADAPTER": f"{tmp_path}/v1", "EXTRA_ADAPTERS": f"sft2={tmp_path}/v2"})
    assert both.returncode == 0, both.stderr
    assert "--max-lora-rank 32 --max-loras 2" in both.stdout  # not the first adapter's 16

    extra_only = _bash(code, {**base, "ADAPTER": "", "EXTRA_ADAPTERS": f"sft2={tmp_path}/v2"})
    assert extra_only.returncode == 0, extra_only.stderr
    assert f"--lora-modules sft2={tmp_path}/v2 " in extra_only.stdout  # not dropped

    missing = _bash(code, {**base, "ADAPTER": f"{tmp_path}/v1", "EXTRA_ADAPTERS": f"sft2={tmp_path}/nope"})
    assert missing.returncode == 1 and "no adapter in" in missing.stderr


@pytest.mark.skipif(shutil.which("git") is None, reason="git isn't installed")
def test_setup_stays_on_a_detached_head_with_a_clear_note(tmp_path):
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x")
    git("checkout", "-q", "--detach")
    code = (
        'step() { :; }; fail() { echo "FAIL $*" >&2; exit 1; }; ok() { echo "ok $*"; }\n'
        'BRANCH="$(git branch --show-current)"\n'
        + _block("setup.sh", 'if [[ -n "$(git status --porcelain', 'step "Training environment')
    )
    res = _bash(code, {"HOME": str(tmp_path)}, cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "on a commit, not a branch" in res.stdout


def test_serve_can_start_with_runtime_lora_and_no_adapter(tmp_path):
    # SEL-68: a heal cycle loads its adapters while serving; vLLM needs LoRA on from the start
    code = _block("serve.sh", "# Every adapter to load", "# Aya Expanse 32B") + (
        '\necho "ARGS ${lora[*]:-}"\necho "RUNTIME ${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-unset}"'
    )
    base = {"MODEL": "m", "PORT": "8000", "MAX_LEN": "32768", "ADAPTER": "", "EXTRA_ADAPTERS": ""}
    heal = _bash(code, {**base, "RUNTIME_LORA": "1"})
    assert heal.returncode == 0, heal.stderr
    assert "ARGS --enable-lora --max-lora-rank 16 --max-loras 4" in heal.stdout
    assert "RUNTIME True" in heal.stdout and "--lora-modules" not in heal.stdout

    (tmp_path / "c1").mkdir()
    (tmp_path / "c1" / "adapter_config.json").write_text('{"r": 32}')
    kept = _bash(code, {**base, "RUNTIME_LORA": "1", "EXTRA_ADAPTERS": f"heal-c1={tmp_path}/c1"})
    assert f"--lora-modules heal-c1={tmp_path}/c1 --max-lora-rank 32 --max-loras 4" in kept.stdout

    plain = _bash(code, base)  # without it: as before, the base model only
    assert plain.returncode == 0, plain.stderr
    assert "ARGS \n" in plain.stdout and "RUNTIME unset" in plain.stdout
