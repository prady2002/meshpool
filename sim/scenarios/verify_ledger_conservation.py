"""Acceptance gate: the credit ledger conserves — credits in ≡ credits out.

Plan §Phase 6 lists this gate ("credits in ≡ credits out, always") but it was
never written, and its absence hid a real defect for as long as it was missing.

The ledger has exactly three entry kinds and only two of them are a transfer:

    donated   -- a node performed work for somebody
    consumed  -- that somebody paid for it
    grant     -- the platform *issued* credits out of nothing

``donated`` and ``consumed`` are the two halves of one movement, so in aggregate
they must be equal. ``grant`` is the single, deliberate issuance channel and is
therefore excluded from the invariant — but *only* grants may be excluded. The
moment anything else writes a ``donated`` row without a matching ``consumed``
row, the ledger is forging the appearance of work, and every number built on top
of it (balances, receipts, payouts, the contributor's earnings ticker) is wrong.

That is not hypothetical. Both escrow engines -- ``StorageAuditEngine`` and
``ComputeEscrowEngine`` -- used to book their bonus release via
``record_donation``. A node that passed N consecutive verifications was credited
for work no consumer ever bought, so ``donated`` drifted above ``consumed``
without limit: an inflation bug in a product whose plan explicitly rules out
inflationary issuance. It survived because the default pytest run deselects the
slow end-to-end tests, so the one assertion that covered it almost never ran.

This gate makes that invariant a first-class, always-run check, and it exercises
the escrow path deliberately rather than incidentally:

  1. run enough verified batch units to drive at least one escrow release;
  2. assert the release actually happened (otherwise the gate is vacuous and
     proves nothing -- this is the non-vacuity guard);
  3. assert donated == consumed across the whole ledger anyway;
  4. assert the escrow bonus is present, and present as ``grant``;
  5. assert every owner's reported balance equals (donated + grant) - consumed;
  6. assert the per-owner rows sum to the summary roll-up, so the invariant
     cannot be satisfied on one view while the UI reads a different one.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402

CP_PORT = 8680  # unused by any other gate

# The compute escrow releases after CONSECUTIVE_PASSES_TO_RELEASE verified units
# per node. Submit comfortably more than that across the fleet so at least one
# release is guaranteed rather than merely likely.
UNITS = 40
TOL = 1e-6


def _auth(cp: str) -> dict:
    r = httpx.post(f"{cp}/api/v1/auth/local", timeout=10)
    r.raise_for_status()
    return r.json()


def _submit(cp: str, hdr: dict, n: int) -> str:
    r = httpx.post(f"{cp}/api/v1/jobs", json={
        "name": "conservation", "func": "square",
        "payloads": [{"x": i} for i in range(n)],
        "cpu_pct": 15, "ram_mb": 64}, headers=hdr, timeout=15)
    r.raise_for_status()
    return r.json()["job_id"]


def _wait_job(cp: str, job_id: str, timeout: float = 180) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = httpx.get(f"{cp}/api/v1/jobs/{job_id}", timeout=10).json()
        if j["status"].startswith("completed"):
            return j
        time.sleep(0.4)
    raise TimeoutError(f"job {job_id} did not finish in {timeout}s")


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


def main() -> int:
    data = Path(__file__).resolve().parent / "_data_ledger_conservation"
    shutil.rmtree(data, ignore_errors=True)
    # Tier 2 ("public") is required: the Verifier -- and therefore the compute
    # escrow whose release used to break this invariant -- is only wired in at
    # tier >= 2. At tier 0/1 this gate would pass without ever exercising the
    # code path it exists to protect.
    h = Harness(n_agents=3, cp_port=CP_PORT, scale="public", sandbox=True,
                tunnel=True, data_dir=str(data))
    ok = True
    try:
        print("== booting PUBLIC cluster (3 agents, verifier active) ==")
        h.start()
        cp = h.cp_url()
        _wait_nodes(cp, 3)

        print(f"\n== 1) run {UNITS} verified units to drive an escrow release ==")
        sess = _auth(cp)
        hdr = {"X-MeshPool-Session": sess["session_token"]}
        job_id = _submit(cp, hdr, UNITS)
        job = _wait_job(cp, job_id)
        assert job["status"] == "completed", job["status"]
        for u in job["units"]:
            assert u["result"] == u["payload"]["x"] ** 2, u
        print(f"  PASS: {UNITS}/{UNITS} units completed correctly")

        led = httpx.get(f"{cp}/api/v1/ledger", timeout=10).json()
        summary = led["summary"]
        donated = float(summary["donated"])
        consumed = float(summary["consumed"])
        granted = float(summary["granted"])
        print(f"  ledger: donated={donated} consumed={consumed} granted={granted}")

        print("\n== 2) NON-VACUITY: an escrow release must actually have occurred ==")
        ver = httpx.get(f"{cp}/api/v1/verification", timeout=10).json()
        released = float(ver["escrow"]["total_released"])
        assert ver["policy"]["active"] is True, (
            "the verifier is not active, so this gate never exercised the escrow "
            "path it exists to protect: " + repr(ver["policy"]))
        assert released > 0.0, (
            "no escrow was ever released, so donated==consumed would hold "
            "trivially and this gate would prove nothing. Raise UNITS or check "
            "CONSECUTIVE_PASSES_TO_RELEASE. verification=" + repr(ver["escrow"]))
        print(f"  PASS: escrow released = {released} (the path IS exercised)")

        print("\n== 3) conservation: donated == consumed ==")
        assert abs(donated - consumed) < TOL, (
            f"ledger does not conserve: donated={donated} consumed={consumed} "
            f"(delta={donated - consumed}). Something credited a node for work "
            f"no consumer paid for -- check for record_donation calls that have "
            f"no matching record_consumption. summary={summary}")
        assert donated > 0.0, f"no work was recorded at all: {summary}"
        print(f"  PASS: donated == consumed == {donated}")

        print("\n== 4) the escrow bonus is booked as a grant, not as donated ==")
        assert granted >= released - TOL, (
            f"escrow released {released} but total grants are only {granted}: "
            f"the bonus was not booked through the issuance channel. summary={summary}")
        print(f"  PASS: granted={granted} covers the {released} of released escrow")

        print("\n== 5) per-owner balances agree with the ledger ==")
        owners = led.get("owners") or []
        assert owners, f"the ledger reported no owners at all: {led.keys()}"
        # Values are rounded to 3dp server-side, so compare at that resolution.
        for row in owners:
            expect = (float(row["donated"]) + float(row["granted"])
                      - float(row["consumed"]))
            bal = float(row["balance"])
            assert abs(bal - expect) < 1e-3, (
                f"owner {row['owner']}: reported balance {bal} != "
                f"(donated + granted) - consumed = {expect}; row={row}")
        print(f"  PASS: {len(owners)} owner balance(s) reconcile exactly")

        print("\n== 6) per-owner rows sum to the summary roll-up ==")
        # A summary that disagrees with its own rows is how a conservation check
        # gets quietly bypassed: the invariant is asserted on one view while the
        # UI and receipts read the other.
        for field in ("donated", "consumed", "granted"):
            rows = sum(float(r[field]) for r in owners)
            assert abs(rows - float(summary[field])) < 1e-3, (
                f"summary.{field}={summary[field]} but the per-owner rows "
                f"sum to {rows}")
        print("  PASS: summary and per-owner rows agree on all three kinds")

        print("\n== 7) every receipt line: total_charged == gross + fee ==")
        # Sections 3-6 all reconcile recorded numbers against other recorded
        # numbers, so a books-wide error that is *consistently* wrong still
        # balances perfectly. Double the platform fee on every charge and the
        # ledger stays self-consistent while consumers are silently overbilled.
        # This is the one check that compares the money to arithmetic instead of
        # to itself.
        r = httpx.get(f"{cp}/api/v1/jobs/{job_id}/receipt", headers=hdr, timeout=10)
        assert r.status_code == 200, f"receipt fetch failed: {r.status_code} {r.text[:200]}"
        receipt = r.json()
        lines = receipt.get("lines") or []
        assert lines, f"receipt carries no line items, so nothing was checked: {receipt}"
        fee_rate = float(receipt["fee_rate"])
        assert fee_rate > 0.0, (
            f"platform fee rate is {fee_rate}, so fee errors are unobservable here "
            "and this check would pass on any arithmetic at all")
        for ln in lines:
            gross, fee = float(ln["gross_earnings"]), float(ln["platform_fee"])
            total = float(ln["total_charged"])
            assert abs(total - (gross + fee)) < TOL, (
                f"line {ln.get('line_id')}: consumer charged {total} but the donor "
                f"earns {gross} and the platform takes {fee} (= {gross + fee}). "
                f"{total - (gross + fee):+.6f} appears from nowhere.")
            assert abs(fee - gross * fee_rate) < TOL, (
                f"line {ln.get('line_id')}: fee {fee} != gross {gross} x rate "
                f"{fee_rate} = {gross * fee_rate}")
        tot = receipt["totals"]
        for field, want in (("gross_earnings", sum(float(x["gross_earnings"]) for x in lines)),
                            ("platform_fee", sum(float(x["platform_fee"]) for x in lines)),
                            ("total_charged", sum(float(x["total_charged"]) for x in lines))):
            assert abs(float(tot[field]) - want) < TOL, (
                f"receipt totals.{field}={tot[field]} but its own lines sum to {want}")
        assert abs(float(receipt["platform_fee_line_item"]["amount"])
                   - float(tot["platform_fee"])) < TOL, (
            "the platform fee shown as a separate line item disagrees with the "
            "fee actually charged across the receipt's lines")
        print(f"  PASS: {len(lines)} line(s) at fee rate {fee_rate}; charges equal "
              f"gross + fee exactly, and totals match their own lines")

        print("\nALL LEDGER CONSERVATION CHECKS PASSED")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        h.stop()
        shutil.rmtree(data, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
