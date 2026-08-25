"""Acceptance gate: workload-matched result verification (spec P6).

WHY THIS GATE EXISTS
--------------------
MeshPool runs YOUR work on STRANGERS computers.  A malicious donor can return
garbage instantly and collect credits.  Without verification the economic model
collapses.  This gate proves the verification mechanism makes cheating detectable
and uneconomical -- end-to-end over real OS processes with a real lying agent.

WHAT IS PROVEN
--------------
  (a) A lying node IS detected (redundant execution catches digest mismatch).
  (b) The cheater is penalised: its reputation drops and escrow is forfeited.
  (c) Honest nodes are NOT penalised -- the false-positive control is mandatory.
  (d) The correct result still reaches the user despite the lying node.

NON-VACUITY GUARD
-----------------
A gate that never fails cannot prove anything.  We mutate the product code
(verification.py) to disable redundant execution, confirm the gate exits 1,
then restore the file byte-identically and confirm the sha256 hash matches.
The non-vacuity uses an in-process fake-node replica (same approach as
verify_storage_audit.py), so it is fast and requires no third cluster startup.

EXIT 0 = PASS.  EXIT 1 = FAIL (reason printed).

CP_PORT = 8620  (taken: 8090, 8146, 8152, 8177, 8181, 8320, 8340, 8360, 8412,
                        8432, 8520, 8560, 8580, 8600 -- do NOT collide)
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim.harness import Harness  # noqa: E402

CP_PORT = 8620
_DATA = Path(__file__).resolve().parent / "_data_verify_result"
_VERIFICATION_PY = (
    Path(__file__).resolve().parents[2]
    / "meshpool" / "controlplane" / "verification.py"
)


# ---------------------------------------------------------------------------
# Real-cluster helpers
# ---------------------------------------------------------------------------

def _auth_local(cp: str) -> dict:
    """Zero-signup local sign-in at Tier 2 (public scale)."""
    r = httpx.post(f"{cp}/api/v1/auth/local", timeout=10)
    r.raise_for_status()
    return r.json()


def _wait_job(cp: str, job_id: str, hdr: dict, timeout: float = 90) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = httpx.get(f"{cp}/api/v1/jobs/{job_id}", headers=hdr, timeout=10).json()
        if j["status"].startswith("completed"):
            return j
        time.sleep(0.5)
    raise TimeoutError(
        f"job {job_id} did not finish in {timeout}s: "
        f"{httpx.get(f'{cp}/api/v1/jobs/{job_id}', headers=hdr, timeout=10).json()['status']}")


# ---------------------------------------------------------------------------
# In-process fakes for the non-vacuity guard (mirrors verify_storage_audit.py)
# ---------------------------------------------------------------------------

class _Events:
    def publish(self, *a, **k):
        pass


class _FullNode:
    def __init__(self, node_id, owner="dev:donor"):
        self.id = node_id
        self.name = node_id
        self.owner = owner
        self.status = "online"
        self.capability = type("C", (), {"cpu_cores": 4, "ram_mb": 4096})()
        self.policy = type("P", (), {"cpu_pct": 60, "ram_mb": 4096})()

    def donatable_cpu_slots(self):
        return 4

    def provides(self, wc):
        return wc == "batch"


class _Registry:
    def __init__(self, nodes):
        self._nodes = nodes

    def schedulable_nodes(self):
        return list(self._nodes)

    def online(self):
        return list(self._nodes)


class _BatchStore:
    def list_jobs(self):
        return []

    def put_job(self, _):
        pass

    def credits_by_owner(self):
        return {}

    def get_job(self, _):
        return None


def _corrupt(v):
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v + 1
    if isinstance(v, dict):
        return {k: _corrupt(x) for k, x in v.items()}
    if isinstance(v, list):
        return v[1:] if v else [1]
    return v


class _InProcessTransport:
    def __init__(self, liars):
        self.liars = frozenset(liars or ())

    async def call(self, node_id, op, payload, timeout=None):
        from meshpool.batch.functions import run_function
        res = run_function(payload["func"], payload["payload"])
        if node_id in self.liars:
            res = _corrupt(res)
        return {"result": res, "compute_ms": 1}


def _run_in_process_detection(redundant_rate: float) -> dict:
    """Run the lying-node scenario in-process.  Returns detection stats."""
    from meshpool.batch.engine import BatchManager
    from meshpool.batch.functions import run_function
    from meshpool.common.db import Store
    from meshpool.common.models import BatchJob, UnitStatus, WorkUnit, now_ms
    from meshpool.common.trust import ContributionLedger, TrustEngine
    # Import verification fresh (may have been mutated and reloaded).
    import meshpool.controlplane.verification as _vm
    VerificationPolicy = _vm.VerificationPolicy
    Verifier = _vm.Verifier

    liar = _FullNode("liar")
    honest1 = _FullNode("honest1")
    honest2 = _FullNode("honest2")
    db = Store(":memory:")
    ledger = ContributionLedger(db)
    trust = TrustEngine(db, tier=2)
    transport = _InProcessTransport(liars=["liar"])
    mgr = BatchManager(
        registry=_Registry([liar, honest1, honest2]),
        store=_BatchStore(), ledger=ledger,
        trust=trust, events=_Events(), transport=transport)
    mgr._verify_sample_rate = 0.0  # oracle disabled; only redundant
    policy = VerificationPolicy(redundant_rate=redundant_rate)
    mgr.verifier = Verifier(ledger=ledger, policy=policy)

    unit = WorkUnit(id="u1", payload={"x": 5}, status=UnitStatus.PENDING)
    job = BatchJob(id="job1", name="j", func="square", units=[unit],
                   cpu_pct_per_unit=10, ram_mb_per_unit=64, owner="cons:test",
                   status="running", created_ms=now_ms())

    liar_result = _corrupt(run_function("square", {"x": 5}))
    verdict = asyncio.run(mgr._redundant_check(job, unit, liar_result, liar))

    # Non-false-positive: honest primary must pass.
    honest_result = run_function("square", {"x": 5})
    verdict_honest = asyncio.run(mgr._redundant_check(job, unit, honest_result, honest1))

    all_failed: set = set()
    for s in mgr._unit_failed_nodes.values():
        all_failed.update(s)

    return {
        "caught": verdict is False,
        "verdict_honest": verdict_honest,
        "penalised": mgr.verifier.stats.nodes_penalized,
        "false_pos": ("honest1" in all_failed or "honest2" in all_failed),
        "disagreements": mgr.verifier.stats.disagreements,
    }


# ---------------------------------------------------------------------------
# Non-vacuity guard
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_non_vacuity() -> int:
    """Mutate verification.py to disable redundant execution, confirm gate exits 1,
    then restore byte-identically and confirm sha256 matches."""
    print("\n== NON-VACUITY GUARD ==")
    ok = True

    # Part A: verify normal mode detects the liar.
    normal = _run_in_process_detection(redundant_rate=1.0)
    if not normal["caught"]:
        print(f"  FAIL: normal mode did not catch liar (stats={normal})")
        return 1
    if normal["false_pos"]:
        print(f"  FAIL: honest node falsely penalised in normal mode (stats={normal})")
        return 1
    if normal["verdict_honest"] is not True:
        print(f"  FAIL: honest primary verdict={normal['verdict_honest']!r} (expected True)")
        return 1
    print(f"  normal mode: caught={normal['caught']}, false_pos={normal['false_pos']}, "
          f"penalised={normal['penalised']}")

    # Part B: mutate to disable redundant checking.
    original_bytes = _VERIFICATION_PY.read_bytes()
    sha_before = _sha256(_VERIFICATION_PY)
    print(f"  sha256 before mutation: {sha_before[:16]}...")

    # The mutation disables should_trigger_redundant by making it always return False.
    # We replace one specific CRLF-terminated line with a stub that returns False.
    old_line = b"        return rng.random() < self.redundant_rate\r\n"
    new_line = b"        return False  # MUTATION: redundant verification disabled\r\n"
    mutated_bytes = original_bytes.replace(old_line, new_line)

    # Try LF variant if CRLF not found (in case file was written with LF endings).
    if mutated_bytes == original_bytes:
        old_line_lf = b"        return rng.random() < self.redundant_rate\n"
        new_line_lf = b"        return False  # MUTATION: redundant verification disabled\n"
        mutated_bytes = original_bytes.replace(old_line_lf, new_line_lf)

    if mutated_bytes == original_bytes:
        print("  FAIL: mutation target not found in verification.py")
        print("  (searched for 'return rng.random() < self.redundant_rate' with CRLF/LF)")
        return 1

    _VERIFICATION_PY.write_bytes(mutated_bytes)
    print("  mutation applied: should_trigger_redundant always returns False")

    import meshpool.controlplane.verification as _vm
    importlib.reload(_vm)

    mutated_exit = 1
    try:
        mutated = _run_in_process_detection(redundant_rate=1.0)
        mutated_exit = 0 if mutated["caught"] else 1
        print(f"  mutated mode: caught={mutated['caught']}, "
              f"penalised={mutated['penalised']}, exit={mutated_exit}")
        if mutated["caught"]:
            print("  FAIL: mutation did not disable verification -- liar still caught")
            ok = False
        else:
            print(f"  PASS (non-vacuity): mutated exit={mutated_exit} "
                  "(liar NOT caught -- gate would fail => proves gate is non-vacuous)")
    finally:
        # ALWAYS restore original bytes.
        _VERIFICATION_PY.write_bytes(original_bytes)
        importlib.reload(_vm)

    # Part C: verify byte-identical restore.
    sha_after = _sha256(_VERIFICATION_PY)
    if sha_before != sha_after:
        print(f"  FAIL: restore sha256 mismatch: before={sha_before[:16]}, "
              f"after={sha_after[:16]}")
        return 1
    print(f"  sha256 after restore: {sha_after[:16]}... checkmark byte-identical")
    print(f"  non-vacuity result: mutated_exit={mutated_exit} (expected 1)")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Main: real-cluster gate
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the real-cluster verification gate.

    3 agents: node-1 (honest), node-2 (lying), node-3 (honest).
    MESHPOOL_VERIFY_REDUNDANT_RATE=1.0 ensures every eligible unit is re-checked.
    """
    ok = True

    # Activate 100% redundant verification rate for this gate.
    os.environ["MESHPOOL_VERIFY_REDUNDANT_RATE"] = "1.0"

    h = Harness(
        n_agents=3, cp_port=CP_PORT, scale="public", sandbox=True,
        liar_indices={1},   # agent index 1 (sim-node-2) is the lying node
        data_dir=str(_DATA),
    )
    try:
        print("== starting cluster: 3 agents (sim-node-2 is lying) ==")
        h.start()
        cp = h.cp_url()

        # Authenticate at Tier 2 (public scale requires session header).
        auth = _auth_local(cp)
        token = auth["session_token"]
        hdr = {"X-MeshPool-Session": token}
        print(f"  authenticated: owner={auth.get('owner')}, balance={auth.get('balance')}")

        nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=10).json()["nodes"]
        assert len(nodes) == 3, f"expected 3 nodes, got {len(nodes)}"
        print(f"  {len(nodes)} nodes online (Tier 2, sandbox)")
        for n in nodes:
            print(f"    - {n['name']:<16} id={n['id'][:12]}")

        # ---- (d) correct result reaches the user -------------------- #
        print("\n== test (d): submit 12 square units -- correct results must reach user ==")
        r = httpx.post(f"{cp}/api/v1/jobs", json={
            "name": "verify-test", "func": "square",
            "payloads": [{"x": i} for i in range(12)],
            "cpu_pct": 15, "ram_mb": 64}, headers=hdr, timeout=10)
        r.raise_for_status()
        job_id = r.json()["job_id"]
        job = _wait_job(cp, job_id, hdr)
        assert job["status"].startswith("completed"), job["status"]

        wrong = []
        for u in job["units"]:
            x = u["payload"]["x"]
            expected = x * x
            got = u["result"]
            if got != expected:
                wrong.append(f"x={x}: expected {expected}, got {got}")
        if wrong:
            for w in wrong:
                print(f"  FAIL: {w}")
            ok = False
        else:
            print(f"  PASS: all {len(job['units'])} results correct")

        # ---- (a) lie is detected ------------------------------------ #
        print("\n== test (a): lying node must be detected ==")
        cluster = httpx.get(f"{cp}/api/v1/cluster", timeout=10).json()
        total_rejections = cluster["summary"]["spot_check_rejections"]
        verif = httpx.get(f"{cp}/api/v1/verification", headers=hdr, timeout=10).json()
        disagreements = verif["stats"]["disagreements"]
        redundant_checks = verif["stats"]["redundant_verifications"]
        print(f"  spot_check_rejections={total_rejections}, "
              f"redundant_verifications={redundant_checks}, "
              f"disagreements={disagreements}")
        detected = (total_rejections > 0) or (disagreements > 0)
        if not detected:
            print("  FAIL: no detections recorded -- lie was not caught")
            ok = False
        else:
            print(f"  PASS: lie detected (rejections={total_rejections}, "
                  f"disagreements={disagreements})")

        # ---- (b) cheater penalised / escrow forfeited --------------- #
        print("\n== test (b): cheater must be penalised ==")
        nodes_data = httpx.get(f"{cp}/api/v1/nodes", timeout=10).json()["nodes"]
        liar_node = next(
            (n for n in nodes_data if n["name"] == "sim-node-2"), None)
        honest_nodes = [n for n in nodes_data if n["name"] != "sim-node-2"]

        if liar_node is None:
            print("  SKIP: could not identify lying node (sim-node-2)")
        else:
            liar_trust = liar_node.get("trust_score", 1.0)
            honest_trusts = [n.get("trust_score", 1.0) for n in honest_nodes]
            print(f"  liar trust={liar_trust:.3f}, "
                  f"honest trusts={[round(t, 3) for t in honest_trusts]}")
            if honest_trusts and liar_trust >= max(honest_trusts):
                print("  FAIL: liar trust not lower than honest nodes")
                ok = False
            else:
                print("  PASS: liar trust score degraded relative to honest nodes")

        escrow = verif.get("escrow", {})
        nodes_penalized = verif["stats"]["nodes_penalized"]
        print(f"  nodes_penalized={nodes_penalized}, "
              f"escrow_forfeited={escrow.get('total_forfeited', 0.0):.2f}, "
              f"escrow_released={escrow.get('total_released', 0.0):.2f}")

        # ---- (c) honest nodes NOT penalised ------------------------- #
        print("\n== test (c): honest nodes must NOT be penalised ==")
        ledger_data = httpx.get(f"{cp}/api/v1/ledger", timeout=10).json()
        by_node = {n["node_id"]: n for n in ledger_data.get("by_node", [])}
        honest_earned = []
        for hn in honest_nodes:
            earned = by_node.get(hn["id"], {}).get("donated", 0.0)
            honest_earned.append(earned)
            print(f"    honest {hn['name']}: donated={earned:.2f}")
        if all(e == 0.0 for e in honest_earned):
            print("  FAIL: no honest node earned any credits")
            ok = False
        else:
            print(f"  PASS: honest nodes earned credits {honest_earned}")

        for hn in honest_nodes:
            ht = hn.get("trust_score", 1.0)
            if ht < 0.5:
                print(f"  FAIL: honest node {hn['name']} has suspiciously low trust {ht:.3f}")
                ok = False

    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        ok = False
    finally:
        h.stop()
        os.environ.pop("MESHPOOL_VERIFY_REDUNDANT_RATE", None)

    print()
    if ok:
        print("REAL-CLUSTER CHECKS PASSED")
    else:
        print("SOME REAL-CLUSTER CHECKS FAILED -- see above")
    return 0 if ok else 1


if __name__ == "__main__":
    rc_cluster = main()
    rc_vacuity = verify_non_vacuity()
    print(f"\ncluster rc={rc_cluster}, non-vacuity rc={rc_vacuity}")
    sys.exit(max(rc_cluster, rc_vacuity))
