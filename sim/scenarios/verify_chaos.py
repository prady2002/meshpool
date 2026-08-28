"""Acceptance gate: the mesh self-heals after the coordinator restarts.

Reliability is the whole promise of "works like a machine." Phase 1 already proves work
survives an *agent* dying (verify_global). This gate proves the harder case: the
**coordinator itself** is killed and restarted while tunnel-only agents are connected,
and the mesh comes back on its own — no manual steps, no re-onboarding.

Over real processes (public/Tier-2, agents reachable ONLY via their outbound tunnel):

  1. baseline: sign in, run a batch job to completion, and note the credit balance;
  2. hard-kill the coordinator process, then start a fresh one on the same data dir;
  3. both agents reconnect on their own (the tunnel client re-registers on a stale
     token and retries with backoff) and appear online again;
  4. state persisted across the restart: the credit ledger (durable SQLite) still shows
     the balance earned/spent before the crash - the starter grant is not re-issued;
  5. recovery: a brand-new job submitted after the restart completes correctly, and the
     ledger still conserves (donated == consumed).
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness, _kill_tree  # noqa: E402

# The harness gives agent i the port cp_port + 1 + i, so a 2-agent cluster owns
# CP_PORT..CP_PORT+2 -- the whole range, not just the control-plane port. The
# old value 8180 therefore claimed 8181, which is verify_console's CP_PORT: the
# gates run serially, so it only bit when one gate's processes had not fully
# released their sockets before the next started, and it then failed as an
# unrelated-looking "service not ready at 127.0.0.1:8181" timeout.
# 8700-8702 is clear of every other gate.
CP_PORT = 8700


def _auth(cp: str) -> dict:
    r = httpx.post(f"{cp}/api/v1/auth/local", timeout=10)
    r.raise_for_status()
    return r.json()


def _submit_square(cp: str, hdr: dict, n: int = 8) -> dict:
    r = httpx.post(f"{cp}/api/v1/jobs", json={
        "name": "chaos", "func": "square",
        "payloads": [{"x": i} for i in range(n)],
        "cpu_pct": 15, "ram_mb": 64}, headers=hdr, timeout=10)
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


def _wait_nodes(cp: str, n: int, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    last = -1
    while time.time() < deadline:
        try:
            nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=3).json().get("nodes", [])
            last = len([x for x in nodes if x.get("status") == "online"])
            if last >= n:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    raise TimeoutError(f"only {last}/{n} nodes online after {timeout}s")


def _wait_tunnels(cp: str, n: int, timeout: float = 60) -> None:
    """Wait until >= n nodes are reachable *through their tunnel* — the real "ready to
    receive work" signal for a global node, stronger than heartbeat-online."""
    deadline = time.time() + timeout
    last = -1
    while time.time() < deadline:
        try:
            nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=3).json().get("nodes", [])
            last = len([x for x in nodes if x.get("reachable_via") == "tunnel"])
            if last >= n:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    raise TimeoutError(f"only {last}/{n} tunnels attached after {timeout}s")


def main() -> int:
    data = Path(__file__).resolve().parent / "_data_chaos"
    shutil.rmtree(data, ignore_errors=True)
    h = Harness(n_agents=2, cp_port=CP_PORT, scale="public", sandbox=True,
                tunnel=True, data_dir=str(data))
    ok = True
    try:
        print("== booting PUBLIC tunnel-only cluster (2 agents) ==")
        h.start()
        cp = h.cp_url()

        print("\n== 1) baseline: sign in and run a job over the tunnel ==")
        sess = _auth(cp)
        hdr = {"X-MeshPool-Session": sess["session_token"]}
        owner = sess["owner"]
        job = _wait_job(cp, _submit_square(cp, hdr)["job_id"])
        assert job["status"] == "completed", job["status"]
        for u in job["units"]:
            assert u["result"] == u["payload"]["x"] ** 2, u
        bal_before = httpx.get(f"{cp}/api/v1/wallet", headers=hdr, timeout=10).json()["balance"]
        assert bal_before > 0, bal_before
        print(f"  PASS: baseline job done; balance after = {bal_before}")

        print("\n== 2) hard-kill the coordinator, then restart on the same data dir ==")
        _kill_tree(h.cp_proc)
        time.sleep(2.0)                       # ensure the socket is fully down
        h.start_control_plane()               # fresh process, same MESHPOOL_DATA_DIR
        print("  coordinator restarted")

        print("\n== 3) agents reconnect on their own (no re-onboarding) ==")
        _wait_nodes(cp, 2, timeout=60)
        # Heartbeat-online is not enough for a global node: assert the reverse tunnel
        # itself re-attached, which is what actually makes the node able to receive
        # work again. (Before the re-register fix this never happened.)
        _wait_tunnels(cp, 2, timeout=60)
        print("  PASS: both agents are online again and their tunnels re-attached")

        print("\n== 4) state persisted across the restart (durable ledger) ==")
        sess2 = _auth(cp)                     # sessions are in-memory -> re-auth
        assert sess2["owner"] == owner, (sess2["owner"], owner)
        hdr2 = {"X-MeshPool-Session": sess2["session_token"]}
        w = httpx.get(f"{cp}/api/v1/wallet", headers=hdr2, timeout=10).json()
        # the balance from before the crash survives; the grant is not re-issued
        assert abs(w["balance"] - bal_before) < 1e-6, (w["balance"], bal_before)
        assert w["granted"] > 0 and w["consumed"] > 0, w
        print(f"  PASS: balance preserved at {w['balance']} (grant not re-issued)")

        print("\n== 5) recovery: a new job after the restart completes correctly ==")
        job2 = _wait_job(cp, _submit_square(cp, hdr2, n=10)["job_id"])
        assert job2["status"] == "completed", job2["status"]
        for u in job2["units"]:
            assert u["result"] == u["payload"]["x"] ** 2, u
        led = httpx.get(f"{cp}/api/v1/ledger", timeout=10).json()
        assert abs(led["summary"]["donated"] - led["summary"]["consumed"]) < 1e-6, led["summary"]
        print("  PASS: post-restart job done; ledger still conserves "
              "(donated == consumed)")

        print("\nALL CHAOS CHECKS PASSED")
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
