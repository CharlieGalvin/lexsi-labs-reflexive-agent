"""The pod's heal service (SEL-68): a heal cycle running elsewhere trains its adapter here.

The cycle (`just heal --train-url`) runs on a laptop and reaches the student only through the
pod's vLLM endpoint. Training needs the pod's GPU, so this small service, next to vLLM, takes a
cycle's fixes and runs `scripts/heal/train_and_serve.sh` on them: stop vLLM, train, start vLLM
again with the new adapter served under its name (``HEAL_SERVE_NEW=1``). The cycle polls until the
job is done, then gates the adapter through the endpoint, as any other model.

On the pod, after ``scripts/pod/setup.sh`` and with vLLM running (``RUNTIME_LORA=1 ADAPTER=
just sft-serve``):

    HEAL_TOKEN=<a secret> VLLM_API_KEY=<the pod's vLLM key> .venv/bin/python scripts/pod/heal_service.py
                                                                            # port 8001

``VLLM_API_KEY`` is required: each job restarts vLLM with this service's environment, and vLLM
only asks for a key when it's set, so without it the public endpoint would come back open.

Expose port 8001 (RunPod: "Expose HTTP ports"), and give the cycle the URL and the token
(``HEAL_SERVICE_URL``, ``HEAL_TOKEN`` in its ``.env``). Every request needs
``Authorization: Bearer <HEAL_TOKEN>``: the port is public.

    POST /heal/train       {"name", "base" (or null), "fixes": [reagent.sft/v1 examples]} → 202
    GET  /heal/jobs/<name> {"status": running | done | failed, "exit_code", "log_tail", ...}
    GET  /heal/health      {"ok": true, "busy": <name> or null}

One job at a time (one GPU). A name is used once: its folder, ``runs/heal-service/<name>/``, holds
the fixes, ``train.log``, ``status.json`` and the adapter, so a restarted service still answers for
earlier jobs. Settings for the wrapper (``HEAL_INIT_ADAPTER``, ``HEAL_SERVE_ADAPTERS``, ``PORT`` …)
are read from this service's environment.
"""

from __future__ import annotations

import argparse
import contextlib
import hmac
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ["bash", str(ROOT / "scripts" / "heal" / "train_and_serve.sh")]
SFT_FORMAT = "reagent.sft/v1"
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")  # a folder name: never a path
MAX_BODY = 50 * 2**20


class HealService:
    """The jobs: one folder each under ``root``; one running at a time."""

    def __init__(self, root: Path, token: str, wrapper: Sequence[str] = WRAPPER) -> None:
        self.root, self.token, self.wrapper = Path(root), token, list(wrapper)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.busy: str | None = None
        self._interrupted()

    def _interrupted(self) -> None:
        """Jobs still "running" from before a restart: their thread is gone, so they never finish."""
        for path in sorted(self.root.glob("*/status.json")):
            try:
                status = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if status.get("status") == "running":
                self._status(
                    path.parent.name, status="failed", exit_code=None, adapter=None, interrupted=True
                )

    # --- Requests -------------------------------------------------------------------------------

    def authorised(self, header: str | None) -> bool:
        # bytes: compare_digest raises TypeError on a str with non-ASCII characters
        want = f"Bearer {self.token}".encode()
        return bool(header) and hmac.compare_digest((header or "").encode("utf-8", "replace"), want)

    def submit(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        name, base, fixes = body.get("name"), body.get("base") or None, body.get("fixes")
        if not isinstance(name, str) or not NAME.match(name):
            return 400, {"error": "name must be a plain folder name (letters, digits, . _ -)"}
        if base is not None and (not isinstance(base, str) or not NAME.match(base)):
            return 400, {"error": "base must be an adapter name, or null for the base student"}
        if not isinstance(fixes, list) or not fixes:
            return 400, {"error": "fixes must be a non-empty list"}
        bad = [f.get("id") if isinstance(f, dict) else repr(f)[:40] for f in fixes if not _is_sft(f)]
        if bad:
            return 400, {"error": f"not {SFT_FORMAT} examples: {bad[:5]}"}
        with self.lock:
            if self.busy is not None:
                return 409, {"error": f"busy with {self.busy}: one job at a time (one GPU)"}
            folder = self.root / name
            if folder.exists():
                return 409, {"error": f"{name} already exists: a name is used once"}
            folder.mkdir()
            self.busy = name
        data = folder / "fixes.jsonl"
        try:
            data.write_text("".join(json.dumps(f) + "\n" for f in fixes))
            self._status(name, status="running", base=base, examples=len(fixes), started=time.time())
            threading.Thread(target=self._run, args=(name, base, data), daemon=True).start()
        except Exception as e:  # noqa: BLE001 - e.g. the disk quota: the job fails, the service stays free
            error = f"{type(e).__name__}: {e}"
            with contextlib.suppress(Exception):
                self._status(name, status="failed", exit_code=None, adapter=None, error=error)
            with self.lock:
                self.busy = None
            return 500, {"error": f"couldn't start {name}: {error}"}
        return 202, {"name": name, "status": "running"}

    def job(self, name: str) -> tuple[int, dict[str, Any]]:
        path = self.root / name / "status.json" if NAME.match(name) else None
        if path is None or not path.exists():
            return 404, {"error": f"no job {name}"}
        status = json.loads(path.read_text())
        log = self.root / name / "train.log"
        if log.exists():
            status["log_tail"] = log.read_text(errors="replace").splitlines()[-40:]
        return 200, status

    def health(self) -> dict[str, Any]:
        return {"ok": True, "busy": self.busy}

    # --- The job --------------------------------------------------------------------------------

    def _run(self, name: str, base: str | None, data: Path) -> None:
        folder = self.root / name
        argv = [*self.wrapper, str(data), str(folder), base or ""]
        env = {**os.environ, "HEAL_SERVE_NEW": "1"}
        start = time.monotonic()
        try:
            with (folder / "train.log").open("w") as log:
                done = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)  # noqa: S603
            code = done.returncode
        except Exception as e:  # noqa: BLE001 - the job fails; the service keeps answering
            code = -1
            with (folder / "train.log").open("a") as log:
                log.write(f"\n⛔ {type(e).__name__}: {e}\n")
        adapter = (folder / "adapter_config.json").exists()
        self._status(
            name,
            status="done" if code == 0 and adapter else "failed",
            exit_code=code,
            adapter=name if code == 0 and adapter else None,
            seconds=round(time.monotonic() - start, 1),
        )
        with self.lock:
            self.busy = None

    def _status(self, name: str, **fields: Any) -> None:
        path = self.root / name / "status.json"
        current = json.loads(path.read_text()) if path.exists() else {"name": name}
        path.write_text(json.dumps({**current, **fields}, indent=1) + "\n")


def _is_sft(f: Any) -> bool:
    return isinstance(f, dict) and f.get("format") == SFT_FORMAT and isinstance(f.get("messages"), list)


def handler(service: HealService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict[str, Any]) -> None:
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _allowed(self) -> bool:
            if service.authorised(self.headers.get("Authorization")):
                return True
            self._send(401, {"error": "Authorization: Bearer <HEAL_TOKEN> needed"})
            return False

        def do_GET(self) -> None:  # noqa: N802 - http.server's name
            if not self._allowed():
                return
            if self.path == "/heal/health":
                self._send(200, service.health())
            elif self.path.startswith("/heal/jobs/"):
                self._send(*service.job(self.path.removeprefix("/heal/jobs/")))
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._allowed():
                return
            if self.path != "/heal/train":
                self._send(404, {"error": "unknown path"})
                return
            size = int(self.headers.get("Content-Length") or 0)
            if size > MAX_BODY:
                self._send(413, {"error": f"over {MAX_BODY // 2**20} MiB"})
                return
            try:
                body = json.loads(self.rfile.read(size) or b"{}")
            except json.JSONDecodeError:
                self._send(400, {"error": "the body isn't JSON"})
                return
            self._send(*service.submit(body if isinstance(body, dict) else {}))

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (BaseHTTPRequestHandler's name)
            sys.stderr.write(f"[heal_service] {self.address_string()} {format % args}\n")

    return Handler


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=int(os.environ.get("HEAL_SERVICE_PORT", 8001)))
    p.add_argument("--host", default="0.0.0.0")  # noqa: S104 - reached through the pod's proxy
    p.add_argument("--root", type=Path, default=ROOT / "runs" / "heal-service")
    a = p.parse_args(argv)
    token = os.environ.get("HEAL_TOKEN", "")
    if len(token) < 16:
        print("⛔ set HEAL_TOKEN (16 characters or more): the service's port is public", file=sys.stderr)
        return 1
    if not os.environ.get("VLLM_API_KEY"):
        print(
            "⛔ set VLLM_API_KEY (the key vLLM is serving with): each job restarts vLLM with this "
            "service's environment, and without it vLLM comes back on its public port with no key",
            file=sys.stderr,
        )
        return 1
    service = HealService(a.root, token)
    server = ThreadingHTTPServer((a.host, a.port), handler(service))
    print(f"heal service on :{a.port}, jobs in {a.root}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
