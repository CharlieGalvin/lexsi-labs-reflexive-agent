"""SEL-68: the pod's heal service and the cycle's `RemoteTrainer`, over real HTTP on localhost.

The service runs a stub instead of `train_and_serve.sh` (it writes an adapter, or fails), and a
stub vLLM lists the models it serves, as the pod's endpoint would after the restart.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from _scripts import load_script

svc = load_script("pod/heal_service")
cycle = load_script("heal/cycle")

TOKEN = "t" * 20
STUB = """
import json, os, sys, time
from pathlib import Path
data, out, base = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
new = os.environ.get("HEAL_SERVE_NEW")
print("training", out.name, "from", base or "the base student", "serve new:", new)
time.sleep(float(os.environ.get("STUB_SECONDS", "0.2")))
if os.environ.get("STUB_FAILS"):
    print("CUDA out of memory"); sys.exit(2)
(out / "adapter_config.json").write_text('{"r": 16}')
"""


def fix(i: int = 0) -> dict:
    return {
        "id": f"heal/glm53f/retail/{i}/t3/a1",
        "format": "reagent.sft/v1",
        "domain": "retail",
        "task_id": str(i),
        "group": "heal",
        "labels": ["ARG"],
        "messages": [{"role": "system", "content": "p"}, {"role": "assistant", "content": "Which order?"}],
        "train_on": [1],
        "source": {"kind": "runtime_fix", "finding": {"type": "ARG", "kind": "placeholder", "tool": "x"}},
    }


def serve(handler) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("localhost", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://localhost:{server.server_address[1]}"


@pytest.fixture
def service(tmp_path, monkeypatch):
    (tmp_path / "stub.py").write_text(STUB)
    s = svc.HealService(tmp_path / "jobs", TOKEN, wrapper=[sys.executable, str(tmp_path / "stub.py")])
    server, url = serve(svc.handler(s))
    yield s, url, tmp_path
    server.shutdown()


def call(url: str, method: str = "GET", body=None, token: str = TOKEN) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait(url: str, name: str) -> dict:
    for _ in range(100):
        code, st = call(f"{url}/heal/jobs/{name}")
        if code == 200 and st["status"] != "running":
            return st
        time.sleep(0.05)
    raise AssertionError("the job never finished")


# --- The service -------------------------------------------------------------------------------


def test_every_request_needs_the_token(service):
    _, url, _ = service
    assert call(f"{url}/heal/health", token="wrong")[0] == 401
    assert call(f"{url}/heal/train", "POST", {"name": "x"}, token="")[0] == 401
    assert call(f"{url}/heal/health") == (200, {"ok": True, "busy": None})


def test_a_job_trains_once_and_reports(service):
    s, url, tmp = service
    code, body = call(
        f"{url}/heal/train", "POST", {"name": "heal-c1", "base": None, "fixes": [fix(0), fix(1)]}
    )
    assert (code, body) == (202, {"name": "heal-c1", "status": "running"})
    assert call(f"{url}/heal/train", "POST", {"name": "heal-c2", "fixes": [fix()]})[0] == 409  # busy
    st = wait(url, "heal-c1")
    assert (
        st["status"] == "done" and st["adapter"] == "heal-c1" and st["exit_code"] == 0 and st["examples"] == 2
    )
    assert "training heal-c1 from the base student serve new: 1" in st["log_tail"]  # HEAL_SERVE_NEW=1
    folder = tmp / "jobs" / "heal-c1"
    assert len((folder / "fixes.jsonl").read_text().splitlines()) == 2
    assert call(f"{url}/heal/train", "POST", {"name": "heal-c1", "fixes": [fix()]})[0] == 409  # used once
    assert call(f"{url}/heal/health")[1]["busy"] is None


def test_a_failed_training_is_reported_with_its_log(service, monkeypatch):
    _, url, _ = service
    monkeypatch.setenv("STUB_FAILS", "1")
    call(f"{url}/heal/train", "POST", {"name": "heal-c1", "base": "heal-c0", "fixes": [fix()]})
    st = wait(url, "heal-c1")
    assert st["status"] == "failed" and st["exit_code"] == 2 and st["adapter"] is None
    assert "CUDA out of memory" in st["log_tail"] and "from heal-c0" in st["log_tail"][0]


@pytest.mark.parametrize(
    "body, error",
    [
        ({"name": "../etc", "fixes": [{"format": "reagent.sft/v1", "messages": []}]}, "plain folder name"),
        ({"name": "ok", "base": "a/../b", "fixes": [fix()]}, "adapter name"),
        ({"name": "ok", "fixes": []}, "non-empty"),
        ({"name": "ok", "fixes": [{"id": "x", "format": "other"}]}, "not reagent.sft/v1"),
    ],
)
def test_bad_requests_are_refused_before_anything_runs(service, body, error):
    _, url, tmp = service
    code, out = call(f"{url}/heal/train", "POST", body)
    assert code == 400 and error in out["error"]
    assert not any((tmp / "jobs").iterdir())


def test_an_unknown_job_is_404_and_a_job_name_is_never_a_path(service):
    _, url, _ = service
    assert call(f"{url}/heal/jobs/nope")[0] == 404
    assert call(f"{url}/heal/jobs/..%2F..%2Fetc")[0] == 404


def test_the_service_wont_start_without_a_long_token(monkeypatch, capsys):
    monkeypatch.setenv("HEAL_TOKEN", "short")
    assert svc.main(["--port", "0"]) == 1 and "HEAL_TOKEN" in capsys.readouterr().err


# --- The cycle's side ---------------------------------------------------------------------------


def vllm(served: list[str]):
    class Models(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            assert self.headers.get("Authorization") == "Bearer pod-key"
            raw = json.dumps({"data": [{"id": m} for m in served]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (BaseHTTPRequestHandler's name)
            pass

    return serve(Models)


def test_the_remote_trainer_trains_on_the_pod_and_checks_the_endpoint_serves_it(service, tmp_path):
    s, url, _ = service
    served: list[str] = ["CohereLabs/aya-expanse-32b"]
    server, api = vllm(served)
    t = cycle.RemoteTrainer(tmp_path / "runs", url=url, token=TOKEN, api_base=f"{api}/v1", api_key="pod-key")
    t.poll_s = 0.05

    # the stub "restarts vLLM" with the new adapter listed (before the job reports done, as on the pod)
    real_run = s._run
    s._run = lambda name, base, data: (served.append(name), real_run(name, base, data))
    name = t.train([fix(0), fix(1)], None)
    assert name.startswith("heal-") and t.calls[-1]["returncode"] == 0 and t.calls[-1]["remote"] == url
    assert (tmp_path / "runs" / name / "fixes.jsonl").exists()  # kept here too, for the audit
    assert "training" in (tmp_path / "runs" / name / "train.log").read_text()
    server.shutdown()


def test_the_remote_trainer_fails_loudly(service, tmp_path, monkeypatch):
    _, url, _ = service
    server, api = vllm(["CohereLabs/aya-expanse-32b"])  # the adapter never shows up
    t = cycle.RemoteTrainer(tmp_path / "runs", url=url, token=TOKEN, api_base=f"{api}/v1", api_key="pod-key")
    t.poll_s = 0.05
    with pytest.raises(RuntimeError, match="doesn't serve it"):
        t.train([fix()], None)
    monkeypatch.setenv("STUB_FAILS", "1")
    with pytest.raises(RuntimeError, match="failed"):
        t.train([fix(1)], None)
    bad = cycle.RemoteTrainer(tmp_path / "runs2", url=url, token="wrong", api_base=f"{api}/v1")
    with pytest.raises(RuntimeError, match="HTTP 401"):
        bad.train([fix()], None)
    server.shutdown()
