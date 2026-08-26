"""Acceptance gate: fraud detection slashes a node that returns wrong results.

Scope: in-process control-plane machinery, no fake green path.  A deterministic
batch unit is run through the real redundant-verification code with one lying
node.  The gate proves detection, slashing as a balanced ledger transfer, and
two non-vacuity controls: an honest node is not slashed, and disabling the
detector lets the lie through.

LIMITATION: this validates the fraud-detection and slashing logic in-process.
It does not yet prove end-to-end detection across real agent processes and the
real transport.  A multi-process version is required before anonymous public
supply is opened.
"""
from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from meshpool.batch.engine import BatchManager  # noqa: E402
from meshpool.batch.functions import run_function  # noqa: E402
from meshpool.common.db import Store  # noqa: E402
from meshpool.common.models import (  # noqa: E402
    BatchJob,
    Capability,
    DonationPolicy,
    Node,
    NodeStatus,
    UnitStatus,
    WorkUnit,
    now_ms,
)
from meshpool.common.trust import ContributionLedger, TrustEngine  # noqa: E402
from meshpool.controlplane.registry import NodeRegistry  # noqa: E402
from meshpool.controlplane.verification import VerificationPolicy, Verifier  # noqa: E402


class _Events:
    def publish(self, *_, **__):
        pass


def _signature_without_self(method) -> tuple:
    sig = inspect.signature(method)
    return tuple(
        (p.kind, p.default is inspect.Signature.empty)
        for name, p in sig.parameters.items()
        if name != "self"
    )


def _assert_conformance() -> None:
    """Fail fast if this in-process gate drifts from real model/registry APIs."""
    for method in ("provides", "donatable_cpu_slots"):
        if not hasattr(Node, method):
            raise AssertionError(f"real Node is missing {method}()")
    assert _signature_without_self(Node.provides) == (
        (inspect.Parameter.POSITIONAL_OR_KEYWORD, True),
    ), "Node.provides(service) signature changed"
    assert _signature_without_self(Node.donatable_cpu_slots) == (), (
        "Node.donatable_cpu_slots() signature changed")

    for method in ("schedulable_nodes", "online", "list", "get"):
        if not hasattr(NodeRegistry, method):
            raise AssertionError(f"real NodeRegistry is missing {method}()")
        if not hasattr(_Registry, method):
            raise AssertionError(f"_Registry fake is missing {method}()")
        assert _signature_without_self(getattr(_Registry, method)) == (
            _signature_without_self(getattr(NodeRegistry, method))
        ), f"_Registry.{method} signature drifted from NodeRegistry.{method}"


def _make_node(node_id: str) -> Node:
    return Node(
        id=node_id,
        name=node_id,
        url=f"http://127.0.0.1/{node_id}",
        owner=f"dev:{node_id}",
        status=NodeStatus.ONLINE,
        capability=Capability(cpu_cores=4, ram_mb=4096),
        policy=DonationPolicy(cpu_pct=100, ram_mb=4096),
    )


class _Registry:
    def __init__(self, nodes):
        self.nodes = nodes

    def schedulable_nodes(self):
        return list(self.nodes)

    def online(self):
        return list(self.nodes)

    def list(self):
        return list(self.nodes)

    def get(self, node_id: str):
        return next((n for n in self.nodes if n.id == node_id), None)


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
    return v + 1 if isinstance(v, (int, float)) else v


class _Transport:
    def __init__(self, liars):
        self.liars = set(liars)
        self.calls = 0

    async def call(self, node_id, op, payload, timeout=None):
        self.calls += 1
        res = run_function(payload["func"], payload["payload"])
        if node_id in self.liars:
            res = _corrupt(res)
        return {"result": res, "compute_ms": 1}


def _make_job():
    unit = WorkUnit(id="u1", payload={"x": 9}, status=UnitStatus.PENDING)
    job = BatchJob(id="job1", name="fraud", func="square", units=[unit],
                   cpu_pct_per_unit=10, ram_mb_per_unit=64, owner="cons:alice",
                   status="running", created_ms=now_ms())
    return job, unit


def _run(liars: set[str], primary_id: str, redundant_rate: float) -> dict:
    _assert_conformance()
    nodes = [_make_node("liar"), _make_node("honest1"), _make_node("honest2")]
    store = Store(":memory:")
    ledger = ContributionLedger(store)
    trust = TrustEngine(store, tier=2)
    verifier = Verifier(ledger=ledger, policy=VerificationPolicy(redundant_rate),
                        trust=trust)
    mgr = BatchManager(_Registry(nodes), _BatchStore(), ledger, trust, _Events(),
                       _Transport(liars))
    mgr._verify_sample_rate = 0.0
    mgr.verifier = verifier
    primary = next(n for n in nodes if n.id == primary_id)
    job, unit = _make_job()
    expected_value = 5.0
    verifier.ensure_stake(primary.id, primary.owner, expected_value)
    result = run_function("square", unit.payload)
    if primary.id in liars:
        result = _corrupt(result)
    verdict = asyncio.run(mgr._redundant_check(
        job, unit, result, primary, expected_job_value=expected_value))
    totals = store.ledger_totals_by_owner()
    store.close()
    return {
        "verdict": verdict,
        "calls": mgr.transport.calls,
        "penalized": verifier.stats.nodes_penalized,
        "slashed": verifier.stats.stake_slashed,
        "totals": totals,
    }


def main() -> int:
    ok = True
    try:
        print("== test 0: gate fakes conform to real Node/NodeRegistry interfaces ==")
        _assert_conformance()
        print("  PASS: Node and registry fake signatures match the real interfaces")

        print("== test 1: dishonest primary is detected and slashed ==")
        bad = _run({"liar"}, "liar", redundant_rate=1.0)
        assert bad["calls"] >= 2, f"redundant path did not execute: {bad}"
        assert bad["verdict"] is False, bad
        assert bad["penalized"] == 1, bad
        assert bad["slashed"] > 0.0, bad
        assert bad["totals"]["meshpool:slashing"]["donated"] == bad["slashed"], bad
        print(f"  PASS: liar detected, penalized={bad['penalized']}, "
              f"slashed={bad['slashed']}")

        print("\n== test 2: NON-VACUITY honest primary is not slashed ==")
        honest = _run(set(), "honest1", redundant_rate=1.0)
        assert honest["calls"] >= 1, f"redundant path did not execute: {honest}"
        assert honest["verdict"] is True, honest
        assert honest["penalized"] == 0, honest
        assert honest["slashed"] == 0.0, honest
        print("  PASS: honest primary verified without penalty")

        print("\n== test 3: NON-VACUITY detector can fail when checks are disabled ==")
        disabled = _run({"liar"}, "liar", redundant_rate=0.0)
        assert disabled["calls"] == 0, disabled
        assert disabled["verdict"] is None, disabled
        assert disabled["penalized"] == 0 and disabled["slashed"] == 0.0, disabled
        print("  PASS: disabled detector misses the liar, so test 1 is meaningful")

        print("\nALL FRAUD DETECTION CHECKS PASSED")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
