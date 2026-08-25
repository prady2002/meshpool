"""ACCEPTANCE GATE — a hard spend cap STOPS work, it does not merely warn.

Plan of record §6.3 ("Anti-AWS by construction") commits to the one billing
guarantee no hyperscaler offers: **the cap stops the work.**  AWS Budgets, GCP
budget alerts and Azure cost alerts are all retrospective — they email you after
the money is gone.  If MeshPool's cap were the same thing wearing a different
label, the product would be selling a safety property it does not have.

A unit test cannot prove this.  ``tests/test_spend_caps.py`` proves the
arithmetic; only a live control plane with real agents executing real units can
prove the enforcement is actually wired into **both** places it has to be:

    Test 1  ADMISSION   a capped owner is refused new work with HTTP 402.
    Test 2  IN FLIGHT   lowering the cap under an already-running job halts it.
                        This is the load-bearing one: a cap enforced only at
                        submission lets a single large job overrun it without
                        limit, which is precisely the failure caps exist to stop.
    Test 3  NON-VACUITY the identical job with no cap runs to completion, so
                        test 2's halt is attributable to the cap and nothing else.
    Test 4  RESUME      raising the cap lets the halted job finish -- a cap is a
                        brake, not a kill switch.
    Test 5  HONESTY     the API reports enforced/spent/remaining truthfully, and
                        reports ``enforced: false`` when no cap exists rather
                        than displaying a number nothing enforces.

Exit code 0 = all five hold.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sim.harness import Harness  # noqa: E402

CP_PORT = 8520
DATA_DIR = str(Path(__file__).resolve().parent / "_data_spendcap")

CAP_URL = "/api/v1/account/spend-cap"


def _fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def _submit(cp: str, payloads: list[dict]) -> httpx.Response:
    return httpx.post(f"{cp}/api/v1/jobs",
                      json={"name": "capjob", "func": "sleep_echo",
                            "payloads": payloads, "cpu_pct": 10},
                      timeout=15)


def _job(cp: str, job_id: str) -> dict:
    return httpx.get(f"{cp}/api/v1/jobs/{job_id}", timeout=10).json()


def _done(job: dict) -> int:
    return sum(1 for u in job["units"] if u["status"] == "done")


def _get_cap(cp: str) -> dict:
    r = httpx.get(f"{cp}{CAP_URL}", timeout=10)
    r.raise_for_status()
    return r.json()


def _set_cap(cp: str, cap, expect_ok: bool = True) -> dict:
    r = httpx.put(f"{cp}{CAP_URL}", json={"cap": cap}, timeout=10)
    if expect_ok:
        r.raise_for_status()
    return r.json()


def _reset_cap(cp: str) -> dict:
    r = httpx.post(f"{cp}{CAP_URL}/reset", timeout=10)
    r.raise_for_status()
    return r.json()


# A job must be big enough that the mesh cannot swallow it in a single dispatch
# round, or there is no "mid-flight" for test 2 to interrupt and the test would
# silently prove nothing.  Sized against total donatable slots (~7 across the
# three sim nodes), so ~LIVE_UNITS x LIVE_SECONDS / 7 seconds of wall clock.
LIVE_UNITS = 60
LIVE_SECONDS = 1.0


def _wait_for_progress(cp: str, job_id: str, n: int, timeout: float = 60.0) -> int:
    """Block until at least ``n`` units are done; return the count actually seen.

    Time-based waits ("sleep 1.5s then assume the job is mid-flight") are a lie
    on a machine faster or busier than the author's -- the first version of this
    gate slammed the cap two seconds AFTER the job had already finished and then
    blamed the cap.  Waiting on observed progress makes the test mean the same
    thing at any speed.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = _job(cp, job_id)
        d = _done(j)
        if d >= n:
            return d
        if j["status"].startswith("completed"):
            _fail(f"the job finished ({d} units) before the cap could be applied "
                  f"-- there was no mid-flight to interrupt, so this gate would "
                  f"prove nothing. Raise LIVE_UNITS/LIVE_SECONDS.")
        time.sleep(0.2)
    _fail(f"job made no meaningful progress within {timeout}s; expected >= {n} "
          f"units done before capping")
    return 0


def main() -> int:
    h = Harness(n_agents=3, cp_port=CP_PORT, data_dir=DATA_DIR, scale="home")
    h.start()
    cp = h.cp_url()
    try:
        h.wait_nodes_online(3)
        print("== spend cap hard-stop gate ==")

        # ---- Test 5a: honest reporting when uncapped -------------------- #
        st = _get_cap(cp)
        if st.get("enforced") is not False:
            _fail(f"a fresh account must report enforced=false, got {st}")
        if st.get("cap") is not None:
            _fail(f"uncapped account must report cap=null, not a fabricated "
                  f"number: {st}")
        print(f"  ok  uncapped account reports honestly: {st}")

        # Establish a baseline of real spend so the cap has something to bite on.
        r = _submit(cp, [{"seconds": 0.2, "echo": i} for i in range(3)])
        r.raise_for_status()
        warm = r.json()["job_id"]
        deadline = time.time() + 90
        while time.time() < deadline:
            if _job(cp, warm)["status"].startswith("completed"):
                break
            time.sleep(0.5)
        else:
            _fail("warm-up job never completed; cannot measure spend")
        spent_after_warm = _get_cap(cp)["spent"]
        if spent_after_warm <= 0:
            _fail(f"executing real units must accrue spend, got {spent_after_warm}")
        print(f"  ok  real work accrues measurable spend: {spent_after_warm}")

        # ---- Test 1: admission is refused once the cap is reached -------- #
        # Cap exactly at current spend => already exceeded => new work refused.
        _set_cap(cp, spent_after_warm)
        st = _get_cap(cp)
        if not st["exceeded"]:
            _fail(f"cap set at current spend must read as exceeded: {st}")
        r = _submit(cp, [{"seconds": 0.1, "echo": 0}])
        if r.status_code != 402:
            _fail(f"a capped owner must be refused new work with HTTP 402, "
                  f"got {r.status_code}: {r.text[:200]}")
        print("  ok  test 1: capped owner refused new work (402)")

        # ---- Test 4 (setup) / cap raised so work is admitted again ------- #
        _set_cap(cp, None)
        r = _submit(cp, [{"seconds": 0.1, "echo": 0}])
        if r.status_code != 200:
            _fail(f"clearing the cap must re-admit work, got {r.status_code}")
        print("  ok  clearing the cap re-admits work")

        # ---- Test 3 (non-vacuity): uncapped job completes ---------------- #
        _reset_cap(cp)
        _set_cap(cp, None)
        big = [{"seconds": LIVE_SECONDS, "echo": i} for i in range(LIVE_UNITS)]
        r = _submit(cp, big)
        r.raise_for_status()
        control_job = r.json()["job_id"]
        deadline = time.time() + 150
        completed = False
        while time.time() < deadline:
            if _job(cp, control_job)["status"].startswith("completed"):
                completed = True
                break
            time.sleep(0.5)
        if not completed:
            _fail("NON-VACUITY: the identical job must complete when uncapped; "
                  "if it cannot, test 2 proves nothing")
        print(f"  ok  test 3: uncapped control job completed "
              f"({_done(_job(cp, control_job))}/{len(big)} units)")

        # ---- Test 2: lowering the cap halts an in-flight job ------------- #
        _reset_cap(cp)
        _set_cap(cp, None)
        r = _submit(cp, [{"seconds": LIVE_SECONDS, "echo": i}
                         for i in range(LIVE_UNITS)])
        r.raise_for_status()
        live_job = r.json()["job_id"]

        # Wait for *observed* progress rather than a fixed sleep, then slam the
        # cap down while the job is provably still running.
        done_at_cap = _wait_for_progress(cp, live_job, 3)
        _set_cap(cp, _get_cap(cp)["spent"])
        st = _get_cap(cp)
        if not st["exceeded"]:
            _fail(f"cap must read exceeded immediately after being lowered: {st}")
        print(f"  ..  cap slammed mid-flight at {done_at_cap} units done")

        # Give the dispatcher ample time to keep going if it is going to.
        time.sleep(12)
        j = _job(cp, live_job)
        done_after = _done(j)
        remaining = sum(1 for u in j["units"]
                        if u["status"] in ("pending", "assigned", "running"))

        # In-flight units are allowed to finish; what must NOT happen is the
        # dispatcher continuing to hand out the rest of the backlog.
        if remaining == 0:
            _fail(f"the capped job ran to completion -- the cap did not stop "
                  f"work, it only warned. done={done_after}/{len(j['units'])}")
        # Progress must have essentially stopped, not merely slowed.
        settled = _done(_job(cp, live_job))
        time.sleep(6)
        if _done(_job(cp, live_job)) > settled:
            _fail("work kept progressing after the cap was reached; a cap that "
                  "only slows spending is an alert, not a cap")
        print(f"  ok  test 2: job halted at {settled}/{len(j['units'])} units "
              f"({remaining} never dispatched)")

        # ---- Test 4: raising the cap resumes the halted job -------------- #
        _set_cap(cp, None)
        deadline = time.time() + 150
        resumed = False
        while time.time() < deadline:
            if _job(cp, live_job)["status"].startswith("completed"):
                resumed = True
                break
            time.sleep(0.5)
        if not resumed:
            _fail("raising the cap must let the halted job finish -- a cap is a "
                  "brake, not a kill switch")
        print("  ok  test 4: halted job resumed and completed after cap raised")

        # ---- Test 5b: reporting stays truthful under a real cap ---------- #
        _reset_cap(cp)
        _set_cap(cp, 1000.0)
        st = _get_cap(cp)
        if not st["enforced"]:
            _fail(f"an explicit cap must report enforced=true: {st}")
        if st["remaining"] is None or st["remaining"] > 1000.0:
            _fail(f"remaining must be reported and bounded by the cap: {st}")
        if st["spent"] < 0:
            _fail(f"spent must never be negative: {st}")
        print(f"  ok  test 5: cap reported truthfully: cap={st['cap']} "
              f"spent={st['spent']} remaining={st['remaining']}")

        print("\nPASS: spend caps stop work at admission AND in flight, "
              "resume when raised, and report honestly.")
        return 0
    finally:
        h.stop()


if __name__ == "__main__":
    raise SystemExit(main())
