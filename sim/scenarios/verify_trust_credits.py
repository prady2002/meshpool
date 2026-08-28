"""Acceptance gate: the PUBLIC (Tier-2) economy — credits, gating, and the trust
spot-check that catches a lying node.

Boots a public-scale cluster of 4 sandboxed agents, one of which is deliberately
DISHONEST (returns wrong results). A separate consumer device authenticates and then
must obey the earn-to-spend rules. This proves, end-to-end over real processes:

  0. gating is ON at Tier 2, and a new device gets a one-time starter grant;
  1. a consumer with no session cannot spend (401);
  2. a batch fan-out submitted WITH a session completes and every accepted result is
     correct -- the dishonest node's answers are caught and never accepted;
  3. the dishonest node earns nothing and its reputation drops below the honest ones;
  4. the credit ledger conserves: total donated == total consumed;
  5. once a consumer overspends its balance, the next submit is refused (402).

Curated compute (the only thing that runs) is executed inside the locked-down
sandbox on every agent, so this also exercises the global safety path end-to-end.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402
from meshpool.common.identity import DeviceIdentity, enrollment_challenge  # noqa: E402
from meshpool.common.models import now_ms  # noqa: E402

STARTER_GRANT = 60.0   # ANON_STARTER_GRANT (accounts.py)


def _auth(cp: str, identity: DeviceIdentity) -> dict:
    ts = now_ms()
    sig = identity.sign(enrollment_challenge(identity.device_id, ts))
    r = httpx.post(f"{cp}/api/v1/auth/device", json={
        "device_id": identity.device_id, "public_key": identity.public_key_b64,
        "signature": sig, "ts_ms": ts}, timeout=10)
    r.raise_for_status()
    return r.json()


def _wait_job(cp: str, job_id: str, timeout: float = 120) -> dict:
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


def main() -> int:  # noqa: C901 — a deliberately end-to-end gate
    data = Path(__file__).resolve().parent / "_data_trust"
    # Start hermetic: this gate asserts on exact credit balances, so a previous
    # run's persisted ledger/identities (e.g. an overspent consumer) must not leak
    # into this one.
    shutil.rmtree(data, ignore_errors=True)
    h = Harness(n_agents=4, cp_port=8155, scale="public", sandbox=True,
                liar_indices={3}, data_dir=str(data))
    ok = True
    try:
        print("== starting PUBLIC cluster (4 sandboxed agents; node-4 is dishonest) ==")
        h.start()
        cp = h.cp_url()
        health = httpx.get(f"{cp}/api/v1/health", timeout=10).json()
        assert health["scale"] == "public", health
        nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=10).json()["nodes"]
        assert len(nodes) == 4, f"expected 4 nodes, got {len(nodes)}"
        print(f"  {len(nodes)} nodes online at scale={health['scale']} (Tier 2)")

        # ---- 0) device auth + starter grant ----------------------------- #
        print("\n== test 0: a new consumer device authenticates and is granted credits ==")
        consumer = DeviceIdentity.load_or_create(str(data / "consumer"))
        auth = _auth(cp, consumer)
        assert auth["gating"] is True, "gating must be ON at Tier 2"
        assert auth["balance"] == STARTER_GRANT, auth
        token = auth["session_token"]
        owner = auth["owner"]
        hdr = {"X-MeshPool-Session": token}
        print(f"  owner={owner} balance={auth['balance']} gating={auth['gating']}")

        # ---- 1) no session cannot spend --------------------------------- #
        print("\n== test 1: consuming without a session is rejected (401) ==")
        r = httpx.post(f"{cp}/api/v1/jobs", json={
            "name": "sq", "func": "square", "payloads": [{"x": 1}],
            "cpu_pct": 10, "ram_mb": 64}, timeout=10)
        assert r.status_code == 401, f"expected 401, got {r.status_code}: {r.text}"
        print("  PASS: anonymous (session-less) consume denied with 401")

        # ---- 2) authenticated fan-out completes with only correct results  #
        print("\n== test 2: authenticated batch fan-out; dishonest results never accepted ==")
        payloads = [{"x": i} for i in range(40)]
        r = httpx.post(f"{cp}/api/v1/jobs", json={
            "name": "sq", "func": "square", "payloads": payloads,
            "cpu_pct": 10, "ram_mb": 64, "owner": "someone-else"}, headers=hdr, timeout=10)
        assert r.status_code == 200, r.text
        assert r.json()["owner"] == owner, "server must resolve owner from the session"
        job = _wait_job(cp, r.json()["job_id"])
        assert job["status"] == "completed", job["status"]
        for u in job["units"]:
            assert u["result"] == u["payload"]["x"] ** 2, f"WRONG result accepted: {u}"
        print(f"  PASS: {len(job['units'])} units done, every accepted result correct")

        # ---- 3) the liar earned nothing + lost reputation --------------- #
        print("\n== test 3: the dishonest node is caught, unpaid, and down-scored ==")
        # NON-VACUITY FIRST. `MESHPOOL_LIE=1` only takes effect if the agent's
        # fault injector opted in (meshpool/common/faultinject.py `_opt_in()`
        # requires MESHPOOL_SIM_HARNESS=1, which sim/harness.py now sets). If the
        # opt-in ever regresses, node-4 silently returns *correct* answers, every
        # assertion below still passes, and this gate proves nothing. So assert
        # the fraud path actually executed before believing its conclusions.
        summary = httpx.get(f"{cp}/api/v1/cluster", timeout=10).json()["summary"]
        assert summary["spot_checks"] > 0, (
            f"no results were spot-checked at all (spot_checks={summary['spot_checks']}) "
            f"— the verification path never ran, so this gate is vacuous")
        assert summary["spot_check_rejections"] > 0, (
            f"{summary['spot_checks']} results were spot-checked but NONE were "
            f"rejected — node-4 never actually lied (is MESHPOOL_SIM_HARNESS=1 set "
            f"in sim/harness.py::_base_env, and is MESHPOOL_LIE reaching the agent?). "
            f"This gate would pass even with fraud detection removed.")
        print(f"  non-vacuity: {summary['spot_checks']} spot-check(s) ran, "
              f"{summary['spot_check_rejections']} caught a wrong result")
        nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=10).json()["nodes"]
        liar = next(n for n in nodes if n["name"] == "sim-node-4")
        honest = [n for n in nodes if n["name"] != "sim-node-4"]
        assert liar["trust_score"] < min(n["trust_score"] for n in honest), \
            f"liar score {liar['trust_score']} not below honest {[n['trust_score'] for n in honest]}"
        led = httpx.get(f"{cp}/api/v1/ledger", timeout=10).json()
        by_owner = {o["owner"]: o for o in led["owners"]}
        liar_donated = by_owner.get(liar["owner"], {}).get("donated", 0.0)
        assert liar_donated == 0.0, f"liar was paid {liar_donated} for rejected work"
        honest_earn = sum(by_owner.get(n["owner"], {}).get("donated", 0.0) for n in honest)
        assert honest_earn > 0, "honest nodes should have earned for real work"
        print(f"  PASS: liar score={liar['trust_score']} donated={liar_donated}; "
              f"honest earned={honest_earn:.3f}")

        # ---- 4) credit conservation ------------------------------------- #
        print("\n== test 4: the ledger conserves (total donated == total consumed) ==")
        s = led["summary"]
        assert s["consumed"] > 0 and abs(s["donated"] - s["consumed"]) < 1e-6, s
        print(f"  PASS: donated={s['donated']} == consumed={s['consumed']}")

        # ---- 5) overspend -> 402 ---------------------------------------- #
        print("\n== test 5: once the balance is spent, the next submit is refused (402) ==")
        wallet = httpx.get(f"{cp}/api/v1/wallet", headers=hdr, timeout=10).json()
        if wallet["balance"] > 0:
            # Still in credit: spend the rest on a heavy job (submitted while the
            # balance is positive, so it is admitted, then runs the balance down).
            heavy = [{"seed": f"h{i}", "rounds": 200000} for i in range(16)]
            r = httpx.post(f"{cp}/api/v1/jobs", json={
                "name": "burn", "func": "sha_pow", "payloads": heavy,
                "cpu_pct": 90, "ram_mb": 64}, headers=hdr, timeout=10)
            assert r.status_code == 200, r.text
            _wait_job(cp, r.json()["job_id"])
            wallet = httpx.get(f"{cp}/api/v1/wallet", headers=hdr, timeout=10).json()
        assert wallet["balance"] <= 0, f"expected a spent-out balance, got {wallet['balance']}"
        r = httpx.post(f"{cp}/api/v1/jobs", json={
            "name": "sq2", "func": "square", "payloads": [{"x": 2}],
            "cpu_pct": 10, "ram_mb": 64}, headers=hdr, timeout=10)
        assert r.status_code == 402, f"expected 402 when broke, got {r.status_code}"
        print(f"  PASS: balance={wallet['balance']:.2f} -> further consumption denied (402)")

        print("\nALL PUBLIC TRUST/CREDITS E2E CHECKS PASSED")
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
