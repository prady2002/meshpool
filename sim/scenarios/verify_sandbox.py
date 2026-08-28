"""Acceptance gate: the locked-down sandbox protects a donor end-to-end.

On the public (Tier-2) mesh every curated unit from a stranger runs inside the
locked-down sandbox on the donor's machine. This gate boots a small public cluster
with sandboxing on and proves, over real processes, the two properties a donor
relies on:

  1. a legitimate unit runs *correctly* end-to-end through the sandbox (the sandbox
     is transparent on success — distributed results equal the local computation);
  2. an *abusive* unit (a cost parameter far past the curated cap, e.g. a
     ``sha_pow`` with 10**12 rounds) is rejected by in-sandbox input validation
     rather than being allowed to burn the donor's CPU — the job finishes fast with
     failed units instead of hanging the machine.

Keeping this fast (2 agents, tiny jobs) makes it a cheap always-on guard that the
global safety path stays intact.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402
from meshpool.batch.functions import run_function  # noqa: E402
from meshpool.common.identity import DeviceIdentity, enrollment_challenge  # noqa: E402
from meshpool.common.models import now_ms  # noqa: E402


def _auth(cp: str, identity: DeviceIdentity) -> dict:
    ts = now_ms()
    sig = identity.sign(enrollment_challenge(identity.device_id, ts))
    r = httpx.post(f"{cp}/api/v1/auth/device", json={
        "device_id": identity.device_id, "public_key": identity.public_key_b64,
        "signature": sig, "ts_ms": ts}, timeout=10)
    r.raise_for_status()
    return r.json()


def _wait_job(cp: str, job_id: str, timeout: float = 90) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = httpx.get(f"{cp}/api/v1/jobs/{job_id}", timeout=10).json()
        if j["status"].startswith("completed"):
            return j
        time.sleep(0.4)
    # Report what the job was actually doing. Polling only for a "completed"
    # prefix means a job stalled with units outstanding, a job whose units all
    # failed, and a job that was never dispatched at all all produce the SAME
    # message -- so the detail that would identify the cause is precisely the
    # one this helper throws away. Fetch it once more on the way out.
    try:
        _last = httpx.get(f"{cp}/api/v1/jobs/{job_id}", timeout=10).json()
        _detail = f" (last seen: {_last})"[:600]
    except Exception as _exc:  # noqa: BLE001 - diagnostics must not mask the timeout
        _detail = f" (control plane unreachable while diagnosing: {_exc})"
    raise TimeoutError(f"job {job_id} did not finish in {timeout}s{_detail}")


def main() -> int:
    data = Path(__file__).resolve().parent / "_data_sandbox"
    shutil.rmtree(data, ignore_errors=True)   # hermetic run
    h = Harness(n_agents=2, cp_port=8165, scale="public", sandbox=True,
                data_dir=str(data))
    ok = True
    try:
        print("== starting PUBLIC cluster (2 sandboxed agents) ==")
        h.start()
        cp = h.cp_url()
        health = httpx.get(f"{cp}/api/v1/health", timeout=10).json()
        assert health["scale"] == "public", health
        consumer = DeviceIdentity.load_or_create(str(data / "consumer"))
        auth = _auth(cp, consumer)
        hdr = {"X-MeshPool-Session": auth["session_token"]}
        print(f"  cluster up at scale={health['scale']} (Tier 2); consumer authenticated")

        # ---- 1) a legitimate unit is correct end-to-end through the sandbox --- #
        print("\n== test 1: a curated job runs correctly through the sandbox ==")
        payloads = [{"x": i} for i in range(8)]
        r = httpx.post(f"{cp}/api/v1/jobs", json={
            "name": "sq", "func": "square", "payloads": payloads,
            "cpu_pct": 10, "ram_mb": 64}, headers=hdr, timeout=10)
        assert r.status_code == 200, r.text
        job = _wait_job(cp, r.json()["job_id"])
        assert job["status"] == "completed", job["status"]
        for u in job["units"]:
            assert u["result"] == u["payload"]["x"] ** 2, f"wrong sandbox result: {u}"
        # transparent on success: also matches a legit embed against the local oracle
        r = httpx.post(f"{cp}/api/v1/jobs", json={
            "name": "emb", "func": "embed",
            "payloads": [{"text": "mesh pool compute", "dim": 16}],
            "cpu_pct": 10, "ram_mb": 64}, headers=hdr, timeout=10)
        assert r.status_code == 200, r.text
        ejob = _wait_job(cp, r.json()["job_id"])
        assert ejob["units"][0]["result"] == run_function(
            "embed", {"text": "mesh pool compute", "dim": 16})
        print(f"  PASS: {len(job['units'])} sandboxed units correct; embed matches oracle")

        # ---- 2) an abusive unit is rejected, not run ------------------------- #
        print("\n== test 2: an abusive over-cap unit is rejected (donor protected) ==")
        t0 = time.time()
        abusive = [{"seed": f"a{i}", "rounds": 10 ** 12} for i in range(4)]
        r = httpx.post(f"{cp}/api/v1/jobs", json={
            "name": "abuse", "func": "sha_pow", "payloads": abusive,
            "cpu_pct": 50, "ram_mb": 64}, headers=hdr, timeout=10)
        assert r.status_code == 200, r.text
        job = _wait_job(cp, r.json()["job_id"], timeout=60)
        elapsed = time.time() - t0
        assert job["status"] == "completed_with_failures", job["status"]
        assert all(u["status"] == "failed" for u in job["units"]), job["units"]
        # rejected by validation, so it finished quickly rather than burning CPU for
        # 10**12 hash rounds (which would take effectively forever if it had run)
        assert elapsed < 30, f"abusive job took too long ({elapsed:.1f}s) — did it run?"
        print(f"  PASS: {len(job['units'])} abusive units failed fast in {elapsed:.1f}s "
              f"(rejected before running)")

        print("\nALL SANDBOX E2E CHECKS PASSED")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        h.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
