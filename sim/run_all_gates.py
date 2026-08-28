"""Run every acceptance gate in sequence and report a pass/fail table.

Gates start real OS processes on fixed ports, so they must run one at a time.
Exit 0 only when every gate exits 0.

Three properties matter more than they look, because a full run takes hours:

* **A hung gate must not destroy the run.** ``subprocess.run(timeout=...)``
  raises, which would abort the whole sweep and throw away every result
  collected so far. Here a timeout is just another failure.
* **A timed-out gate must not poison the next one.** Gates spawn control
  planes and agents on *fixed* ports; killing only the direct child leaves
  those grandchildren holding the ports, so every later gate fails for a
  reason that has nothing to do with the code under test. We kill the whole
  process tree and then wait for the ports to be released.
* **The run must be resumable.** Results are appended to
  ``_gatelogs/results.json`` as they happen, and ``--resume`` skips gates
  that already passed, so a machine reboot costs minutes rather than hours.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = sorted((ROOT / "sim" / "scenarios").glob("verify_*.py"))
LOGS = ROOT / "_gatelogs"
RESULTS = LOGS / "results.json"

DEFAULT_TIMEOUT = 1800.0
SETTLE_SECONDS = 2.0
TIMEOUT_EXIT = -9


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the gate *and everything it started*.

    A gate that hangs has almost always left a control plane and a handful of
    agents behind it. Those hold the fixed ports the next gate needs.
    """
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def _load(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save(path: Path, data: dict[str, dict]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _ports_busy(ports: list[int]) -> list[int]:
    busy = []
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                busy.append(port)
    return busy


def _wait_for_quiet_ports(deadline: float = 30.0) -> None:
    """Block until the well-known gate ports are bindable again."""
    ports = [8080, 8081, 8090, 8100, 9000, 9001, 9002, 9003]
    end = time.time() + deadline
    while time.time() < end:
        if not _ports_busy(ports):
            return
        time.sleep(1.0)


def main() -> int:
    args = list(sys.argv[1:])
    resume = "--resume" in args
    if resume:
        args.remove("--resume")
    timeout = DEFAULT_TIMEOUT
    if "--timeout" in args:
        i = args.index("--timeout")
        timeout = float(args[i + 1])
        del args[i:i + 2]
    only = set(args)

    LOGS.mkdir(exist_ok=True)
    recorded = _load(RESULTS) if resume else {}
    results: list[tuple[str, int, float]] = []

    selected = [p for p in SCENARIOS if not only or p.stem in only]
    unknown = only - {p.stem for p in SCENARIOS}
    if unknown:
        print(f"unknown gate(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    for index, path in enumerate(selected, start=1):
        name = path.stem
        prior = recorded.get(name)
        if resume and prior and prior.get("code") == 0:
            print(f"{'SKIP':<9} {name:<40} (already passed)", flush=True)
            results.append((name, 0, float(prior.get("seconds", 0.0))))
            continue

        print(f"{'RUN':<9} {name:<40} [{index}/{len(selected)}]", flush=True)
        started = time.time()
        with open(LOGS / f"{name}.log", "w", encoding="utf-8", errors="replace") as fh:
            proc = subprocess.Popen([sys.executable, str(path)], cwd=str(ROOT),
                                    stdout=fh, stderr=subprocess.STDOUT)
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_tree(proc)
                code = TIMEOUT_EXIT
                fh.write(f"\n\n[run_all_gates] TIMEOUT after {timeout:.0f}s; "
                         f"process tree killed\n")
        elapsed = time.time() - started

        results.append((name, code, elapsed))
        recorded[name] = {"code": code, "seconds": round(elapsed, 1),
                          "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _save(RESULTS, recorded)

        if code == 0:
            flag = "PASS"
        elif code == TIMEOUT_EXIT:
            flag = "TIMEOUT"
        else:
            flag = f"FAIL({code})"
        print(f"{flag:<9} {name:<40} {elapsed:6.1f}s", flush=True)

        time.sleep(SETTLE_SECONDS)
        if code != 0:
            _wait_for_quiet_ports()

    bad = [r for r in results if r[1] != 0]
    print(f"\n{len(results) - len(bad)}/{len(results)} gates passed "
          f"in {sum(r[2] for r in results):.0f}s")
    for name, code, _ in bad:
        label = "timed out" if code == TIMEOUT_EXIT else f"exit {code}"
        print(f"  FAILED {name} ({label}) -> _gatelogs/{name}.log")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
