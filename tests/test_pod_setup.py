"""SEL-67: scripts/pod/setup.sh, the RunPod setup. It can't run here (no GPU); these check it's sound."""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from reflexive.files import ROOT

SCRIPT = ROOT / "scripts" / "pod" / "setup.sh"


def test_the_script_is_valid_bash_and_stops_on_the_first_error():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assert "set -euo pipefail" in SCRIPT.read_text()


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck isn't installed")
def test_shellcheck_finds_nothing():
    subprocess.run(["shellcheck", str(SCRIPT)], check=True)


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
