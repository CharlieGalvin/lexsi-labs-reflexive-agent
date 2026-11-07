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
    for f in ("arms.base.yaml", "arms.sft.yaml"):
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
