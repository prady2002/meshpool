"""Acceptance gate: the global "unzip -> run -> contribute/use" experience works.

Phase 3 turns MeshPool into a product anyone can run and use with zero signup. This
gate boots a public (Tier-2) coordinator + agents exactly as a shipped build would and
proves, over real processes, the UX-critical paths the redesigned dashboard depends on:

  1. the dashboard and its static assets are actually served;
  2. onboarding advertises a one-command join for every OS;
  3. zero-signup local sign-in (``POST /api/v1/auth/local``) mints a consumer session
     and the one-time starter grant on the machine running MeshPool -- no keys in the
     browser, no account required;
  4. the Wallet API reflects that grant for the signed-in owner;
  5. the "Use" flow works: a batch job submitted *with the session* is accepted (not
     401/402), completes, and debits the consumer while crediting the donors -- and the
     earn-to-spend gate still rejects a session-less submit with 401;
  6. the ledger endpoint returns BOTH the per-owner and per-node shapes from one route
     (regression guard: these used to be two routes on one path, silently shadowing).

Kept small (2 agents, tiny jobs) so it is a cheap always-on guard on the product path.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402


def _wait_job(cp: str, job_id: str, timeout: float = 90) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = httpx.get(f"{cp}/api/v1/jobs/{job_id}", timeout=10).json()
        if j["status"].startswith("completed"):
            return j
        time.sleep(0.4)
    raise TimeoutError(f"job {job_id} did not finish in {timeout}s")


def main() -> int:
    data = Path(__file__).resolve().parent / "_data_ux"
    shutil.rmtree(data, ignore_errors=True)   # hermetic run
    h = Harness(n_agents=2, cp_port=8170, scale="public", sandbox=True,
                data_dir=str(data))
    ok = True
    try:
        print("== starting PUBLIC cluster (2 agents) as a shipped build would ==")
        h.start()
        cp = h.cp_url()

        # ---- 1) the dashboard + assets are served --------------------------- #
        print("\n== test 1: dashboard and static assets are served ==")
        root = httpx.get(f"{cp}/", timeout=10)
        assert root.status_code == 200, root.status_code
        assert "<!DOCTYPE html>" in root.text[:80], root.text[:80]
        appjs = httpx.get(f"{cp}/static/app.js", timeout=10)
        assert appjs.status_code == 200 and len(appjs.text) > 200, appjs.status_code
        print("  PASS: index.html + /static/app.js served")

        # ---- 2) onboarding advertises a one-command join -------------------- #
        print("\n== test 2: onboarding gives a one-command join for every OS ==")
        onb = httpx.get(f"{cp}/api/v1/onboarding", timeout=10).json()
        assert onb.get("join_windows") and onb.get("join_unix"), onb
        assert onb.get("invite_token"), onb
        print("  PASS: onboarding advertises Windows + macOS/Linux join commands")

        # ---- 3) zero-signup local sign-in mints a session + starter grant --- #
        print("\n== test 3: zero-signup local sign-in (no keys in the browser) ==")
        s = httpx.post(f"{cp}/api/v1/auth/local", timeout=10)
        assert s.status_code == 200, s.text
        sess = s.json()
        assert sess["gating"] is True and sess["tier"] == 2, sess
        assert sess["session_token"] and sess["owner"].startswith("dev:"), sess
        grant = sess["balance"]
        assert grant > 0, f"expected a starter grant, got balance {grant}"
        hdr = {"X-MeshPool-Session": sess["session_token"]}
        print(f"  PASS: signed in as {sess['owner']} with starter balance {grant}")

        # ---- 4) the Wallet API reflects the grant --------------------------- #
        print("\n== test 4: Wallet reflects the signed-in owner's balance ==")
        w = httpx.get(f"{cp}/api/v1/wallet", headers=hdr, timeout=10).json()
        assert w["owner"] == sess["owner"], w
        assert abs(w["balance"] - grant) < 1e-6 and w["granted"] > 0, w
        assert w["consumed"] == 0, w
        # a wallet request with no session must be rejected at Tier 2
        assert httpx.get(f"{cp}/api/v1/wallet", timeout=10).status_code == 401
        print(f"  PASS: wallet balance {w['balance']} granted {w['granted']}; "
              f"anonymous wallet blocked (401)")

        # ---- 5) the Use flow works, and the earn-to-spend gate holds -------- #
        print("\n== test 5: 'Use' works with a session; blocked without one ==")
        # session-less submit is rejected (earn-to-spend gate)
        anon = httpx.post(f"{cp}/api/v1/jobs", json={
            "name": "sq", "func": "square", "payloads": [{"x": 1}],
            "cpu_pct": 10, "ram_mb": 64}, timeout=10)
        assert anon.status_code == 401, f"session-less submit should 401, got {anon.status_code}"
        # with the session it is accepted and completes
        r = httpx.post(f"{cp}/api/v1/jobs", json={
            "name": "sq", "func": "square", "payloads": [{"x": i} for i in range(12)],
            "cpu_pct": 10, "ram_mb": 64}, headers=hdr, timeout=10)
        assert r.status_code == 200, r.text
        job = _wait_job(cp, r.json()["job_id"])
        assert job["status"] == "completed", job["status"]
        for u in job["units"]:
            assert u["result"] == u["payload"]["x"] ** 2, u
        w2 = httpx.get(f"{cp}/api/v1/wallet", headers=hdr, timeout=10).json()
        assert w2["consumed"] > 0, w2
        assert w2["balance"] < grant, (w2["balance"], grant)
        print(f"  PASS: 12 units done; consumer debited "
              f"{round(grant - w2['balance'], 3)} credits; anon submit blocked (401)")

        # ---- 6) ledger returns both shapes from one route ------------------- #
        print("\n== test 6: ledger exposes owner + node views from one route ==")
        led = httpx.get(f"{cp}/api/v1/ledger", timeout=10).json()
        for key in ("owners", "summary", "by_node", "credits_by_owner", "recent"):
            assert key in led, f"ledger missing '{key}': {list(led)}"
        assert len(led["by_node"]) == 2, led["by_node"]
        assert led["summary"]["consumed"] > 0 and led["summary"]["granted"] > 0, led["summary"]
        # conservation: what consumers spent equals what donors earned
        assert abs(led["summary"]["donated"] - led["summary"]["consumed"]) < 1e-6, led["summary"]
        print("  PASS: ledger has owners+by_node; donated == consumed (conserved)")

        print("\nALL UX E2E CHECKS PASSED")
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
