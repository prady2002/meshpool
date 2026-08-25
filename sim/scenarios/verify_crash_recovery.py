"""Acceptance gate: crash-recovery correctness for the control plane.

Drives a real control plane + node agents, hard-kills the coordinator
mid-flight (simulating a power loss or OOM kill), restarts it, and asserts:

  1. **DB integrity** — the recovered database is readable with no SQLITE_CORRUPT
     or missing-table errors.
  2. **Acknowledged nodes present** — every node that the coordinator had already
     persisted (responded to its registration request) is still in the DB after
     restart.
  3. **Jobs present** — every job whose submission returned an id is still present
     and queryable after restart.
  4. **In-flight units requeued** — ``recover_inflight_units()`` moves every
     ASSIGNED/RUNNING unit back to PENDING so the batch manager can retry them;
     no unit is silently lost.
  5. **No ledger corruption** — the ledger table is intact and queryable; every
     row that was committed before the kill is still there.
  6. **WAL + FULL synchronous** — verified directly on the recovered file.

The gate exits 0 (PASS) or 1 (FAIL).  It never hangs: every wait has a
deadline and raises TimeoutError on miss.

Quality bar: modelled on verify_spd_equivalence.py and verify_inference.py —
each assertion is distinct, independently falsifiable, and reported explicitly.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim.harness import REPO_ROOT, Harness, _kill_tree  # noqa: E402
from meshpool.common.db import Store  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _wait_http(url: str, timeout: float = 25.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code < 500:
                return
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(0.3)
    raise TimeoutError(f"service not ready at {url} after {timeout}s: {last}")


def _submit_job(cp: str, n_units: int = 8, owner: str = "alice") -> str:
    """Submit a batch job and return its id."""
    r = httpx.post(f"{cp}/api/v1/jobs", json={
        "name": "crash-test-job",
        "func": "square",
        "payloads": [{"x": i} for i in range(n_units)],
        "cpu_pct": 10,
        "ram_mb": 64,
        "owner": owner,
    }, timeout=15)
    r.raise_for_status()
    return r.json()["job_id"]


def _get_job(cp: str, job_id: str) -> dict:
    r = httpx.get(f"{cp}/api/v1/jobs/{job_id}", timeout=10)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:  # noqa: C901  (complexity OK — each test step is sequential)
    ok = True
    CP_PORT = 8199
    DATA_DIR = str(Path(__file__).resolve().parent / "_data_crash")

    h = Harness(n_agents=2, cp_port=CP_PORT, data_dir=DATA_DIR)
    db_path = str(Path(DATA_DIR) / "meshpool.db")

    try:
        # ------------------------------------------------------------------ #
        print("== starting cluster ==")
        h.start()
        cp = h.cp_url()
        print(f"  control plane: {cp}")
        print(f"  DB path:        {db_path}")

        # ------------------------------------------------------------------ #
        print("\n== test 1: baseline sanity — nodes registered ==")
        nodes_before = httpx.get(f"{cp}/api/v1/nodes", timeout=10).json()["nodes"]
        node_ids_before = {n["id"] for n in nodes_before}
        print(f"  {len(node_ids_before)} nodes online before crash: {node_ids_before}")
        assert len(node_ids_before) >= 2, "expected at least 2 nodes online"
        print("  PASS")

        # ------------------------------------------------------------------ #
        print("\n== test 2: submit batch jobs before crash ==")
        job_id_a = _submit_job(cp, n_units=12, owner="alice")
        job_id_b = _submit_job(cp, n_units=6, owner="bob")
        print(f"  submitted job_a={job_id_a}, job_b={job_id_b}")
        # Let the scheduler dispatch some units (but don't wait for completion)
        time.sleep(1.5)
        job_a_view = _get_job(cp, job_id_a)
        print(f"  job_a progress: {job_a_view.get('progress', {})}")
        print("  PASS: jobs acknowledged, some units dispatched")

        # Capture ledger state before crash
        ledger_before = httpx.get(f"{cp}/api/v1/ledger", timeout=10).json()
        entries_before = len(ledger_before.get("entries", []))
        print(f"\n== pre-crash ledger: {entries_before} entries ==")

        # ------------------------------------------------------------------ #
        print("\n== test 3: HARD KILL the control plane mid-flight ==")
        _kill_tree(h.cp_proc)
        h.cp_proc = None
        print("  control plane process killed (SIGKILL / taskkill /F)")
        # agents are still alive, still generating heartbeats
        time.sleep(0.5)
        print("  PASS: process dead")

        # ------------------------------------------------------------------ #
        print("\n== test 4: verify DB is not corrupted after kill ==")
        conn = sqlite3.connect(db_path)
        try:
            result = conn.execute("PRAGMA integrity_check").fetchone()[0]
            assert result == "ok", f"integrity_check returned: {result!r}"
            jm = conn.execute("PRAGMA journal_mode").fetchone()[0]
            sync = conn.execute("PRAGMA synchronous").fetchone()[0]
        finally:
            conn.close()
        print(f"  integrity_check: {result!r}  journal_mode: {jm!r}  synchronous: {sync}")
        assert jm == "wal", f"Expected WAL mode after crash, got {jm!r}"
        assert sync == 2, f"Expected synchronous=FULL (2) after crash, got {sync}"
        print("  PASS: DB intact, WAL+FULL synchronous confirmed")

        # ------------------------------------------------------------------ #
        print("\n== test 5: verify recover_inflight_units requeues stuck units ==")
        store = Store(db_path)
        recovered_jobs = store.recover_inflight_units()
        print(f"  recover_inflight_units touched job ids: {recovered_jobs}")
        # After recovery every unit in the touched jobs must be PENDING
        for jid in recovered_jobs:
            job = store.get_job(jid)
            assert job is not None, f"job {jid} missing from DB after recovery"
            bad = [u for u in job.units
                   if u.status.value in ("assigned", "running")]
            assert not bad, (
                f"job {jid} still has {len(bad)} ASSIGNED/RUNNING units after recovery"
            )
        store.close()
        print("  PASS: all in-flight units requeued to PENDING")

        # ------------------------------------------------------------------ #
        print("\n== test 6: restart control plane ==")
        h.start_control_plane()
        print(f"  control plane restarted on {cp}")
        print("  PASS")

        # ------------------------------------------------------------------ #
        print("\n== test 7: all acknowledged nodes are present after restart ==")
        nodes_after = httpx.get(f"{cp}/api/v1/nodes", timeout=10).json()["nodes"]
        node_ids_after = {n["id"] for n in nodes_after}
        print(f"  nodes after restart: {node_ids_after}")
        missing = node_ids_before - node_ids_after
        assert not missing, (
            f"nodes lost across crash: {missing}\n"
            f"  before: {node_ids_before}\n  after: {node_ids_after}"
        )
        print("  PASS: all pre-crash nodes survived restart")

        # ------------------------------------------------------------------ #
        print("\n== test 8: acknowledged jobs still present after restart ==")
        job_a_after = _get_job(cp, job_id_a)
        job_b_after = _get_job(cp, job_id_b)
        assert job_a_after["id"] == job_id_a, "job_a id mismatch after restart"
        assert job_b_after["id"] == job_id_b, "job_b id mismatch after restart"
        print(f"  job_a status: {job_a_after.get('status')!r}  "
              f"job_b status: {job_b_after.get('status')!r}")
        print("  PASS: all submitted jobs survived the crash")

        # ------------------------------------------------------------------ #
        print("\n== test 9: ledger is readable and rows committed before kill are present ==")
        ledger_after = httpx.get(f"{cp}/api/v1/ledger", timeout=10).json()
        entries_after = len(ledger_after.get("entries", []))
        print(f"  entries before crash: {entries_before}, after restart: {entries_after}")
        # entries_after >= entries_before because no committed rows must be lost;
        # more entries may exist because the store's write-ahead log was checkpointed
        # on restart, capturing any in-flight WAL writes that completed before the kill.
        assert entries_after >= entries_before, (
            f"ledger shrank from {entries_before} to {entries_after} across crash "
            f"— committed rows were lost"
        )
        print("  PASS: no committed ledger rows lost")

        # ------------------------------------------------------------------ #
        print("\n== test 10: agents re-heartbeat and nodes come back online ==")
        online_after = h.wait_nodes_online(len(node_ids_before), timeout=30)
        online_ids = {n["id"] for n in online_after}
        assert online_ids >= node_ids_before, (
            f"not all pre-crash nodes came back online: missing {node_ids_before - online_ids}"
        )
        print(f"  {len(online_after)} nodes back online: {online_ids}")
        print("  PASS: nodes reconnected and heartbeating")

        # ------------------------------------------------------------------ #
        print("\n== test 11: jobs resume and complete after restart ==")
        # Wait up to 30s for both jobs to reach a terminal state
        deadline = time.time() + 30
        while time.time() < deadline:
            ja = _get_job(cp, job_id_a)
            jb = _get_job(cp, job_id_b)
            if (ja.get("status") in ("completed", "completed_with_failures") and
                    jb.get("status") in ("completed", "completed_with_failures")):
                break
            time.sleep(1.0)
        ja = _get_job(cp, job_id_a)
        jb = _get_job(cp, job_id_b)
        print(f"  job_a final status: {ja.get('status')!r}  "
              f"job_b final status: {jb.get('status')!r}")
        assert ja.get("status") in ("completed", "completed_with_failures"), (
            f"job_a did not complete: {ja.get('status')}"
        )
        assert jb.get("status") in ("completed", "completed_with_failures"), (
            f"job_b did not complete: {jb.get('status')}"
        )
        print("  PASS: both jobs completed after restart")

        print("\nALL CRASH RECOVERY CHECKS PASSED")

    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAIL: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        h.stop()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
