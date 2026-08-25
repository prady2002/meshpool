"""End-to-end scenario: scoped, single-use secret brokering (spec §5, §9.8).

Proves that a workload can use a credential that is **never** in its job definition:
the control plane stores the value in memory, mints a single-use node-scoped lease
per unit, the agent redeems it just-in-time, and the unit signs a request with it.
The raw secret never appears in any API response; a unit that didn't request the
secret never receives it; and leases are single-use (consumed after redemption).
"""
from __future__ import annotations

import hashlib
import hmac
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402

TOKEN = "sk-meshpool-demo-C0ffee-Sekret"


def _submit(cp: str, func: str, payloads: list[dict], secret_names=None) -> str:
    body = {"name": func, "func": func, "payloads": payloads, "cpu_pct": 15}
    if secret_names is not None:
        body["secret_names"] = secret_names
    r = httpx.post(f"{cp}/api/v1/jobs", json=body, timeout=10)
    r.raise_for_status()
    return r.json()["job_id"]


def _wait_job(cp: str, job_id: str, timeout: float = 60) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = httpx.get(f"{cp}/api/v1/jobs/{job_id}", timeout=10).json()
        if j["status"].startswith("completed"):
            return j
        time.sleep(0.3)
    raise TimeoutError(f"job {job_id} did not finish")


def _expected_sig(resource: str) -> str:
    return hmac.new(TOKEN.encode(), resource.encode(), hashlib.sha256).hexdigest()[:32]


def main() -> int:
    h = Harness(n_agents=2, cp_port=8140,
                data_dir=str(Path(__file__).resolve().parent / "_data_secrets"))
    ok = True
    try:
        print("== starting cluster (control plane + 2 agents) ==")
        h.start()
        cp = h.cp_url()
        h.wait_nodes_online(2)

        # ---- store a secret; confirm the value is never exposed --------- #
        print("\n== test 1: store secret; value must never leave the control plane ==")
        httpx.post(f"{cp}/api/v1/secrets",
                   json={"name": "API_TOKEN", "value": TOKEN}, timeout=10).raise_for_status()
        listing = httpx.get(f"{cp}/api/v1/secrets", timeout=10).json()
        assert "API_TOKEN" in listing["names"], listing
        assert TOKEN not in httpx.get(f"{cp}/api/v1/secrets", timeout=10).text
        print(f"  listing exposes names only: {listing['names']} (value hidden) PASS")

        # ---- run a job that USES the brokered secret -------------------- #
        print("\n== test 2: job signs requests with a secret it never sees in its spec ==")
        resources = [f"/api/resource/{i}" for i in range(8)]
        jid = _submit(cp, "secure_fetch",
                      [{"resource": r} for r in resources], secret_names=["API_TOKEN"])
        job = _wait_job(cp, jid)
        assert job["status"] == "completed", job["status"]
        # every unit produced the correct HMAC -> it really received the token value
        by_res = {u["result"]["resource"]: u["result"] for u in job["units"]}
        for r in resources:
            res = by_res[r]
            assert res["ok"] is True, res
            assert res["signature"] == _expected_sig(r), (r, res["signature"])
        print(f"  {len(resources)} units signed correctly with the brokered token PASS")

        # the raw secret must NOT appear anywhere in the job record
        raw = httpx.get(f"{cp}/api/v1/jobs/{jid}", timeout=10).text
        assert TOKEN not in raw, "secret leaked into job record!"
        print("  secret value absent from the job record (only signatures) PASS")

        # ---- a unit that didn't request the secret never gets it -------- #
        print("\n== test 3: unit without secret_names is not given the secret ==")
        jid2 = _submit(cp, "secure_fetch", [{"resource": "/nope"}])  # no secret_names
        job2 = _wait_job(cp, jid2)
        r2 = job2["units"][0]["result"]
        assert r2["ok"] is False and "no API_TOKEN" in r2["error"], r2
        print(f"  unresolved unit result: {r2} PASS")

        # ---- leases are single-use (all consumed / swept) --------------- #
        after = httpx.get(f"{cp}/api/v1/secrets", timeout=10).json()
        assert after["active_leases"] == 0, after
        print(f"  active leases after run: {after['active_leases']} (single-use) PASS")

        # ---- unknown secret fails the unit cleanly (job still terminates) - #
        print("\n== test 4: a job needing a missing secret fails cleanly (no hang) ==")
        jid3 = _submit(cp, "secure_fetch", [{"resource": "/x"}], secret_names=["MISSING"])
        job3 = _wait_job(cp, jid3)  # must reach a terminal state, not hang
        u3 = job3["units"][0]
        assert job3["status"] == "completed_with_failures", job3["status"]
        assert u3["status"] == "failed" and "secret error" in (u3["error"] or ""), u3
        print(f"  job terminal={job3['status']!r}, unit error={u3['error']!r} PASS")

        print("\nALL SECRETS E2E CHECKS PASSED")
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
