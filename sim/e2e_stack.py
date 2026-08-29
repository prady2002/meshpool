"""Boot a real MeshPool mesh for the browser end-to-end tests.

Playwright's ``webServer`` starts a bare control plane, which is enough to prove
the console boots, routes and paints.  It is not enough to prove the *flagship*
screen works.  The Playground can only stream tokens when a model is genuinely
deployed across genuinely separate agent processes; against a bare control plane
the model list is empty, the Send button stays disabled, and the one path the
product is actually sold on never executes.  Every existing browser test passes
in that state.

This starts the same :class:`sim.harness.Harness` the acceptance gates use -- a
control plane plus N agents as real OS processes -- deploys a model, waits for it
to report ``ready``, then prints exactly one line::

    MESHPOOL_E2E_READY <base-url> stages=<n>

and blocks.  The parent (a Playwright spec) waits for that line, drives a browser
against the URL, and then closes this process's stdin.

Closing stdin is the shutdown signal on purpose.  It is the only mechanism that
behaves the same on Windows and POSIX: Node's ``proc.kill()`` terminates this
process alone, which would orphan the control plane and every agent it spawned,
leaving ports held and the next run mysteriously broken.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from sim.harness import Harness  # noqa: E402

READY_PREFIX = "MESHPOOL_E2E_READY"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _deploy(cp: str, model: str, timeout: float) -> int:
    """Deploy *model* and block until the control plane reports it ready.

    Returns the number of pipeline stages so the caller can refuse to declare
    readiness for a "deployment" that was never actually sharded.
    """
    r = httpx.post(f"{cp}/api/v1/models", json={"model": model}, timeout=120)
    r.raise_for_status()
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        view = httpx.get(f"{cp}/api/v1/models", timeout=10).json()
        last = view.get("status")
        if last == "ready":
            return len(view.get("stages") or [])
        time.sleep(1.0)
    raise TimeoutError(f"{model} never became ready (last status={last!r})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--port", type=int, default=0, help="0 picks a free port")
    ap.add_argument("--model", default="nano-gpt")
    ap.add_argument("--scale", default="home")
    ap.add_argument("--deploy-timeout", type=float, default=240.0)
    args = ap.parse_args()

    data_dir = str(Path(os.environ.get("TEMP", "/tmp")) / "meshpool-e2e-stack")
    harness = Harness(n_agents=args.agents, cp_port=args.port or _free_port(),
                      scale=args.scale, data_dir=data_dir)

    stop = threading.Event()

    def _watch_stdin() -> None:
        # Blocks until the parent closes the pipe (or a terminal sends EOF).
        try:
            for _ in sys.stdin:
                pass
        except Exception:  # noqa: BLE001
            pass
        stop.set()

    started = False
    try:
        harness.start()
        started = True
        cp = harness.cp_url()
        harness.wait_nodes_online(args.agents, timeout=90)
        stages = _deploy(cp, args.model, args.deploy_timeout)
        if stages < 1:
            raise RuntimeError(
                f"{args.model} reported ready with {stages} stages -- nothing is sharded")

        threading.Thread(target=_watch_stdin, daemon=True).start()
        # One machine-readable line, flushed, only once everything the browser
        # needs is genuinely serving.  Anything printed before this is diagnostic
        # noise and the parent ignores it.
        print(f"{READY_PREFIX} {cp} stages={stages}", flush=True)

        while not stop.wait(0.5):
            if harness.cp_proc is not None and harness.cp_proc.poll() is not None:
                print("control plane exited unexpectedly", file=sys.stderr, flush=True)
                return 1
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"e2e stack failed to start: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if started:
            try:
                harness.stop()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    raise SystemExit(main())
