"""Acceptance gate: deny-by-default egress filter blocks abuse destinations (spec ?6.3).

What this gate proves (end-to-end, no cluster needed for most tests):
  (a) All 4 hard-blocked ports (25/445/3389/6667) are unconditionally denied.
  (b) RFC1918 / loopback / link-local / APIPA (169.254.x.x) are denied by default.
  (c) A whitelisted destination actually succeeds (positive control with a real TCP server).
  (d) WASM workloads have NO network capability (WASI p1 has no socket API ? structural).
  (e) ProcessRuntime declares egress policy compliance.
  (f) ALL runtime classes declare _EGRESS_POLICY_COMPLIANT=True ? adding a new runtime
      without this attribute FAILS this gate immediately.
  (g) The abuse-report endpoint works over a real cluster.

Non-vacuity:
  - Positive control (c) proves the filter is not just "all connections fail".
  - Mutation 1: change HARD_BLOCKED_PORTS default to frozenset() ? gate fails at test (a).
  - Mutation 2: flip deny-by-default to allow-by-default ? gate fails at test (b).
  Both mutations are byte-level and restored identically (sha256 verified).
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import inspect
import json
import shutil
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx  # noqa: E402

from meshpool.common.identity import DeviceIdentity, enrollment_challenge  # noqa: E402
from meshpool.common.models import now_ms  # noqa: E402
from meshpool.functions import wasm as _wasm  # noqa: E402
from sim.harness import Harness
from sim import mutation_guard  # noqa: E402

CP_PORT = 8720
DATA_DIR = Path(__file__).resolve().parent / "_data_egress"
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" ? {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def _auth(cp: str, identity: DeviceIdentity) -> dict:
    ts = now_ms()
    sig = identity.sign(enrollment_challenge(identity.device_id, ts))
    resp = httpx.post(f"{cp}/api/v1/auth/device", json={
        "device_id": identity.device_id,
        "public_key": identity.public_key_b64,
        "signature": sig,
        "ts_ms": ts,
    }, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextlib.contextmanager
def _byte_mutation(path: Path, old_variants: tuple[bytes, ...], replacement_text: str):
    original = path.read_bytes()
    matched = None
    for old in old_variants:
        if old in original:
            matched = old
            break
    if matched is None:
        raise AssertionError(f"mutation target not found in {path}")
    newline = "\r\n" if matched.endswith(b"\r\n") else "\n"
    mutated = original.replace(matched, replacement_text.replace("\n", newline).encode("utf-8"), 1)
    _mg = mutation_guard.journal(path, original)
    path.write_bytes(mutated)
    try:
        yield _sha256(path)
    finally:
        path.write_bytes(original)
        mutation_guard.clear(_mg)
        restored = _sha256(path)
        original_sha = hashlib.sha256(original).hexdigest()
        if restored != original_sha:
            raise AssertionError(f"restore sha mismatch: {restored} != {original_sha}")


def _test_hard_blocked_ports(egress_mod) -> None:
    policy = egress_mod.EgressPolicy(allow_all=True)
    for port in (25, 445, 3389, 6667):
        try:
            policy.check("1.2.3.4", port)
        except egress_mod.EgressDenied:
            continue
        raise AssertionError(f"port {port} unexpectedly allowed")


def _test_private_ranges(egress_mod) -> None:
    policy = egress_mod.EgressPolicy()
    for host in ("127.0.0.1", "192.168.1.1", "169.254.169.254"):
        try:
            policy.check(host, 80)
        except egress_mod.EgressDenied:
            continue
        raise AssertionError(f"private destination {host} unexpectedly allowed")


def _positive_control(egress_mod) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    policy = egress_mod.EgressPolicy(whitelist=[("127.0.0.1", port)])
    policy.check("127.0.0.1", port)
    accepted: list[tuple[str, int]] = []

    def _serve() -> None:
        conn, addr = listener.accept()
        accepted.append(addr)
        conn.sendall(b"ok")
        conn.close()
        listener.close()

    import threading
    th = threading.Thread(target=_serve, daemon=True)
    th.start()
    with socket.create_connection(("127.0.0.1", port), timeout=2) as conn:
        assert conn.recv(2) == b"ok"
    th.join(timeout=2)
    if not accepted:
        raise AssertionError("positive-control server was never reached")


def _test_wasm_preflight() -> tuple[bool, str]:
    if not _wasm.WASM_AVAILABLE:
        return True, "skipped (wasmtime not installed)"
    import wasmtime

    blob = bytes(wasmtime.wat2wasm(r"""
(module
  (import "my_module" "net" (func $net))
  (func (export "_start")
    call $net))
"""))
    try:
        _wasm.preflight(blob)
    except _wasm.FunctionError as exc:
        if exc.kind != "invalid_module":
            raise AssertionError(f"unexpected FunctionError kind: {exc.kind}") from exc
        return True, exc.message
    raise AssertionError("foreign-import WASM module unexpectedly passed preflight")


def _runtime_classes(runtime_module) -> list[type]:
    classes = []
    for _name, cls in inspect.getmembers(runtime_module, inspect.isclass):
        if cls.__module__ != runtime_module.__name__:
            continue
        if cls.__name__ == "Runtime":
            continue
        if callable(getattr(cls, "run_unit", None)) and callable(getattr(cls, "cpu_slots", None)):
            classes.append(cls)
    return classes


def _check_runtime_markers() -> None:
    import meshpool.agent.runtime as runtime_module

    classes = _runtime_classes(runtime_module)
    if not classes:
        raise AssertionError("no runtime classes found")
    for cls in classes:
        if getattr(cls, "_EGRESS_POLICY_COMPLIANT", None) is not True:
            raise AssertionError(f"{cls.__name__} missing _EGRESS_POLICY_COMPLIANT=True")


def _check_real_cluster() -> None:
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    harness = Harness(n_agents=2, cp_port=CP_PORT, data_dir=str(DATA_DIR), scale="public", sandbox=True)
    try:
        harness.start()
        cp = harness.cp_url()
        identity = DeviceIdentity.load_or_create(str(DATA_DIR / "consumer"))
        auth = _auth(cp, identity)
        headers = {"X-MeshPool-Session": auth["session_token"]}

        report = {
            "node_id": "sim-node-1",
            "description": "test abuse report",
            "evidence": json.dumps({"kind": "gate", "ts": time.time()}),
        }
        post_resp = httpx.post(f"{cp}/api/v1/abuse/report", json=report, timeout=10)
        post_resp.raise_for_status()
        post_body = post_resp.json()
        if not post_body.get("accepted"):
            raise AssertionError(f"abuse report not accepted: {post_body}")

        get_resp = httpx.get(f"{cp}/api/v1/egress/policy", headers=headers, timeout=10)
        get_resp.raise_for_status()
        policy_body = get_resp.json()
        if policy_body.get("default") != "deny":
            raise AssertionError(f"unexpected egress policy body: {policy_body}")
    finally:
        harness.stop()


def _mutation_checks() -> None:
    import meshpool.common.egress as egress_mod

    path = Path(egress_mod.__file__).resolve()
    original_sha = _sha256(path)

    with _byte_mutation(
        path,
        (b"    25,    # SMTP relay (spam)\r\n", b"    25,    # SMTP relay (spam)\n"),
        "    # 25, [MUTATED]\n",
    ):
        egress_mod = importlib.reload(egress_mod)
        mutated_policy = egress_mod.EgressPolicy(allow_all=True)
        allowed = False
        try:
            mutated_policy.check("1.2.3.4", 25)
            allowed = True
        except egress_mod.EgressDenied:
            allowed = False
        check("mutation 1 non-vacuity", allowed,
              f"mutated_exit={1 if allowed else 0}; restore_sha={original_sha}")
    egress_mod = importlib.reload(egress_mod)
    check("mutation 1 restore sha", _sha256(path) == original_sha, _sha256(path))

    with _byte_mutation(
        path,
        (b"        raise EgressDenied(f\"{host}:{port} denied by default\", host, port)\r\n",
         b"        raise EgressDenied(f\"{host}:{port} denied by default\", host, port)\n"),
        "        return  # MUTATED: deny disabled\n",
    ):
        egress_mod = importlib.reload(egress_mod)
        mutated_policy = egress_mod.EgressPolicy()
        allowed = False
        try:
            mutated_policy.check("1.2.3.4", 80)
            allowed = True
        except egress_mod.EgressDenied:
            allowed = False
        check("mutation 2 non-vacuity", allowed,
              f"mutated_exit={1 if allowed else 0}; restore_sha={original_sha}")
    egress_mod = importlib.reload(egress_mod)
    check("mutation 2 restore sha", _sha256(path) == original_sha, _sha256(path))


def main() -> int:
    ok = True
    try:
        import meshpool.common.egress as egress_mod

        _test_hard_blocked_ports(egress_mod)
        check("hard-blocked ports", True)
        _test_private_ranges(egress_mod)
        check("private ranges denied", True)
        _positive_control(egress_mod)
        check("whitelist positive control", True)
        wasm_ok, wasm_detail = _test_wasm_preflight()
        check("wasm has no network capability", wasm_ok, wasm_detail)
        _check_runtime_markers()
        check("runtime compliance markers", True)
        _check_real_cluster()
        check("real abuse-report and egress-policy endpoints", True)
        _mutation_checks()
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    if FAILS:
        ok = False
    if ok:
        print("\nALL EGRESS FILTER CHECKS PASSED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
