"""SEL-68: scripts/heal/train_and_serve.sh, a heal cycle's training command, with stubs for vLLM and
the trainer: it stops the server, trains on the cycle's fixes, and starts the server again.

The stub server answers ``/v1/models`` like vLLM; its command line carries a marker the script is
told to match (``VLLM_MATCH``). The script is run as `heal.CommandTrainer` runs it: output
captured, so a server left holding the pipes would hang the test.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid

import pytest

from reflexive.files import ROOT

SCRIPT = ROOT / "scripts" / "heal" / "train_and_serve.sh"
pytestmark = pytest.mark.skipif(
    shutil.which("curl") is None or shutil.which("pgrep") is None, reason="needs curl and pgrep"
)

SERVER = """
import http.server, json, os, sys
marker, port, env_file = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if env_file != "-":
    keys = ("ADAPTER", "EXTRA_ADAPTERS", "RUNTIME_LORA", "PORT")
    json.dump({k: os.environ.get(k) for k in keys}, open(env_file, "w"))
key = os.environ.get("VLLM_API_KEY")
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if key and self.headers.get("Authorization") != f"Bearer {key}":
            self.send_response(401)  # as vLLM does with VLLM_API_KEY
        else:
            self.send_response(200 if self.path == "/v1/models" else 404)
        self.end_headers()
        self.wfile.write(b'{"data": []}')
    def log_message(self, *a):
        pass
http.server.HTTPServer(("localhost", port), H).serve_forever()
"""

TRAINER = """
import json, os, sys
from pathlib import Path
argv = sys.argv[1:]
json.dump(argv, open(os.environ["TRAIN_ARGS_FILE"], "w"))
open(os.environ["TRAIN_ARGS_FILE"] + ".hf_home", "w").write(os.environ.get("HF_HOME", ""))
if os.environ.get("TRAIN_FAILS"):
    sys.exit(3)
out = Path(argv[argv.index("--out") + 1])
(out / "adapter_config.json").write_text('{"r": 16}')
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def answers(port: int) -> bool:
    return (
        subprocess.run(["curl", "-sf", f"http://localhost:{port}/v1/models"], capture_output=True).returncode
        == 0
    )


@pytest.fixture
def pod(tmp_path):
    """A running stub vLLM, the stubs, and the cycle's folders, as on the pod."""
    marker = f"stub-vllm-{uuid.uuid4().hex[:8]}"
    port = free_port()
    (tmp_path / "server.py").write_text(SERVER)
    (tmp_path / "trainer.py").write_text(TRAINER)
    runs = tmp_path / "results" / "heal" / "run" / "runs"
    out = runs / "heal-20261110-120000-c1"
    out.mkdir(parents=True)
    (out / "fixes.jsonl").write_text('{"id": "a"}\n{"id": "b"}\n')
    old = subprocess.Popen([sys.executable, str(tmp_path / "server.py"), marker, str(port), "-"])
    for _ in range(50):
        if answers(port):
            break
        time.sleep(0.1)
    env = {
        **os.environ,
        "PORT": str(port),
        "VLLM_MATCH": marker,
        "HEAL_TRAIN": f"{sys.executable} {tmp_path / 'trainer.py'}",
        "HEAL_SERVE": f"{sys.executable} {tmp_path / 'server.py'} {marker} {port} {tmp_path / 'env.json'}",
        "SERVE_LOG": str(tmp_path / "vllm.log"),
        "TRAIN_ARGS_FILE": str(tmp_path / "train_args.json"),
        "WORKSPACE": str(tmp_path),
        "POLL": "0.2",
        "READY_TIMEOUT": "30",
        "STOP_TIMEOUT": "10",
    }
    env.pop("HEAL_INIT_ADAPTER", None)
    env.pop("VLLM_API_KEY", None)
    env.pop("HF_HOME", None)
    env.pop("HEAL_SERVE_ADAPTERS", None)
    yield {"tmp": tmp_path, "port": port, "old": old, "runs": runs, "out": out, "env": env, "marker": marker}
    old.kill()
    subprocess.run(["pkill", "-f", marker], check=False)


def run(pod, base: str = "", **env) -> subprocess.CompletedProcess:
    out = pod["out"]
    return subprocess.run(
        ["bash", str(SCRIPT), str(out / "fixes.jsonl"), str(out), base],
        capture_output=True,
        text=True,
        timeout=60,
        env={**pod["env"], **env},
    )


def read(pod, name: str):
    return json.loads((pod["tmp"] / name).read_text())


def test_it_is_valid_bash_and_stops_on_the_first_error():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assert "set -euo pipefail" in SCRIPT.read_text()


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck isn't installed")
def test_shellcheck_finds_nothing():
    subprocess.run(["shellcheck", str(SCRIPT)], check=True)


def test_first_cycle_stops_the_server_trains_a_new_adapter_and_serves_again(pod):
    done = run(pod)
    assert done.returncode == 0, done.stderr
    assert pod["old"].wait(timeout=5) is not None  # the old server was stopped
    args = read(pod, "train_args.json")
    out = str(pod["out"])
    assert args[0] == "configs/train/sft_heal.yaml"
    assert args[1:] == ["--data", f"{out}/fixes.jsonl", "--out", out, "--name", pod["out"].name]
    assert answers(pod["port"])  # a new server, started by the script, is up
    assert read(pod, "env.json") == {
        "ADAPTER": "",  # no "sft" adapter at start
        "EXTRA_ADAPTERS": "",  # nothing to keep: the base student is the gate's baseline
        "RUNTIME_LORA": "1",  # the cycle loads the new adapter itself
        "PORT": str(pod["port"]),
    }
    assert "vLLM is up" in done.stdout


def test_a_later_cycle_continues_the_adapter_being_healed_and_serves_it_for_the_gate(pod):
    prev = pod["runs"] / "heal-20261110-110000-c1"
    prev.mkdir()
    (prev / "adapter_config.json").write_text('{"r": 16}')
    done = run(pod, base=prev.name, HEAL_SERVE_ADAPTERS="sft2=runs/sft-aya32b-r16-v2")
    assert done.returncode == 0, done.stderr
    args = read(pod, "train_args.json")
    assert args[args.index("--init-adapter") + 1] == str(prev)
    assert read(pod, "env.json")["EXTRA_ADAPTERS"] == f"sft2=runs/sft-aya32b-r16-v2 {prev.name}={prev}"


def test_a_first_cycle_on_an_adapter_starts_from_heal_init_adapter(pod):
    v2 = pod["tmp"] / "sft-v2"
    v2.mkdir()
    (v2 / "adapter_config.json").write_text('{"r": 16}')
    done = run(pod, HEAL_INIT_ADAPTER=str(v2), HEAL_SERVE_ADAPTERS=f"sft2={v2}")
    assert done.returncode == 0, done.stderr
    args = read(pod, "train_args.json")
    assert args[args.index("--init-adapter") + 1] == str(v2)
    assert read(pod, "env.json")["EXTRA_ADAPTERS"] == f"sft2={v2}"


def test_a_failed_training_still_brings_the_server_back_and_fails(pod):
    done = run(pod, TRAIN_FAILS="1")
    assert done.returncode == 3  # training's exit code
    assert "training failed (exit 3)" in done.stdout
    assert answers(pod["port"])


def test_an_adapter_it_cant_find_stops_everything_before_the_server_is_touched(pod):
    done = run(pod, base="heal-missing-c9")
    assert done.returncode == 1 and "no adapter for base heal-missing-c9" in done.stderr
    assert pod["old"].poll() is None and answers(pod["port"])  # still serving
    assert not (pod["tmp"] / "train_args.json").exists()


def test_a_server_that_dies_while_starting_is_reported(pod):
    (pod["tmp"] / "dies.py").write_text("import sys\nprint('CUDA error: out of memory')\nsys.exit(1)\n")
    done = run(pod, HEAL_SERVE=f"{sys.executable} {pod['tmp'] / 'dies.py'}")
    assert done.returncode == 1 and "vLLM exited while starting" in done.stderr
    assert "CUDA error: out of memory" in done.stderr  # the log's tail, to see why


def test_for_the_heal_service_the_new_adapter_is_served_from_the_restart(pod):
    done = run(pod, HEAL_SERVE_NEW="1")
    assert done.returncode == 0, done.stderr
    assert read(pod, "env.json")["EXTRA_ADAPTERS"] == f"{pod['out'].name}={pod['out']}"
    failed = run(pod, HEAL_SERVE_NEW="1", TRAIN_FAILS="1")  # nothing new to serve
    assert failed.returncode == 3 and read(pod, "env.json")["EXTRA_ADAPTERS"] == ""


def test_a_server_that_requires_a_key_is_checked_with_it(pod):
    done = run(pod, VLLM_API_KEY="sk-pod", READY_TIMEOUT="5")
    assert done.returncode == 0, done.stderr
    assert "vLLM is up" in done.stdout


def test_it_loads_the_pods_caches_as_serve_sh_does(pod):
    hf_home = pod["tmp"] / "hf_cache"
    (pod["tmp"] / ".pod_env").write_text(f"export HF_HOME={hf_home}\n")
    done = run(pod)
    assert done.returncode == 0, done.stderr
    assert (pod["tmp"] / "train_args.json.hf_home").read_text() == str(hf_home)  # training saw it


def test_a_gpu_that_doesnt_free_up_skips_training_but_still_brings_the_server_back(pod):
    bin_dir = pod["tmp"] / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi").write_text("#!/bin/sh\necho 30000\n")  # something else holds 30 GB
    (bin_dir / "nvidia-smi").chmod(0o755)
    done = run(pod, PATH=f"{bin_dir}:{os.environ['PATH']}", STOP_TIMEOUT="1")
    assert done.returncode == 1 and "Not training" in done.stdout
    assert not (pod["tmp"] / "train_args.json").exists()  # never trained
    assert answers(pod["port"]) and "vLLM is up" in done.stdout  # but the endpoint is back


def test_an_adapter_already_served_isnt_passed_to_vllm_twice(pod):
    prev = pod["runs"] / "heal-20261110-110000-c1"
    prev.mkdir()
    (prev / "adapter_config.json").write_text('{"r": 16}')
    done = run(pod, base=prev.name, HEAL_SERVE_ADAPTERS=f"{prev.name}={prev} sft2=runs/v2")
    assert done.returncode == 0, done.stderr
    assert read(pod, "env.json")["EXTRA_ADAPTERS"] == f"{prev.name}={prev} sft2=runs/v2"
