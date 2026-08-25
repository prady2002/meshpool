"""A stalled coordinator must not evict a healthy mesh.

``NodeRegistry.reap`` decides a node is dead by comparing its last heartbeat
against a wall-clock cutoff. That is sound only while the reaper is actually
running. If the control plane's event loop is starved for longer than
``node_timeout_s`` -- a GC pause, a blocking call on the loop thread, a VM
suspend, or an oversubscribed host -- then on the next tick every heartbeat in
the registry looks stale simultaneously and a single pass evicts the entire
mesh.

Observed on a loaded machine during a gate run::

    22:04:06 WARNING registry  node timed out -> offline node=... sim-node-1
    22:04:06 WARNING registry  node timed out -> offline node=... sim-node-2
    22:04:06 WARNING registry  node timed out -> offline node=... sim-node-3
    22:04:06 WARNING inference cannot re-partition: insufficient survivor capacity

Three nodes in the same second is the signature of one coordinator stall, not
three independent failures. The blast radius is everything the reaper drives:
tunnels dropped, batch units requeued, hosting replicas torn down, model
partitions abandoned. Recovery then fails too, because re-partitioning needs
survivors and the sweep left none.
"""
from __future__ import annotations

import time

import pytest

from meshpool.common.db import Store
from meshpool.common.models import (
    Capability, DonationPolicy, Heartbeat, NodeLoad, NodeStatus, RegisterRequest,
)
from meshpool.controlplane.registry import EventBus, NodeRegistry


NODE_TIMEOUT_S = 12


@pytest.fixture()
def registry(tmp_path):
    store = Store(str(tmp_path / "cp.db"))
    reg = NodeRegistry(store, NODE_TIMEOUT_S, EventBus())
    reg.reap_interval_s = 2.0
    return reg


def _register(reg: NodeRegistry, name: str):
    resp = reg.register(
        RegisterRequest(
            name=name,
            url=f"http://127.0.0.1:9/{name}",
            capability=Capability(cpu_cores=4, ram_mb=8192, disk_mb=10000,
                                  flops_estimate=10.0),
        ),
        heartbeat_interval_s=2,
    )
    node = reg.get(resp.node_id)
    assert node is not None
    return node


def _beat(reg: NodeRegistry, node_id: str) -> None:
    reg.heartbeat(Heartbeat(node_id=node_id, load=NodeLoad(),
                            policy=DonationPolicy()))


def _steady_state(reg: NodeRegistry, ticks: int = 3) -> None:
    """Establish a normal cadence so the first real call is not seen as a stall."""
    for _ in range(ticks):
        reg.reap()


class TestNormalOperation:
    def test_a_genuinely_stale_node_is_still_evicted(self, registry):
        """The guard must not become a blanket amnesty.

        If a stall check swallowed every eviction, dead nodes would linger
        forever and work would keep being scheduled onto them -- a far worse
        failure than the one being fixed.
        """
        node = _register(registry, "dead-node")
        _steady_state(registry)

        node.last_heartbeat_ms -= (NODE_TIMEOUT_S + 5) * 1000
        gone = registry.reap()

        assert gone == [node.id]
        assert registry.get(node.id).status is NodeStatus.OFFLINE

    def test_a_healthy_node_is_left_alone(self, registry):
        node = _register(registry, "live-node")
        _steady_state(registry)

        assert registry.reap() == []
        assert registry.get(node.id).status is not NodeStatus.OFFLINE


class TestCoordinatorStall:
    def test_a_stalled_reaper_evicts_nobody(self, registry, monkeypatch):
        """The headline case: the loop was gone longer than the timeout.

        Every heartbeat is stale, but that is evidence about the coordinator,
        not about the nodes.
        """
        nodes = [_register(registry, f"node-{i}") for i in range(3)]
        _steady_state(registry)

        # Simulate the loop being descheduled for well past the eviction timeout.
        base = time.monotonic()
        monkeypatch.setattr(time, "monotonic",
                            lambda: base + NODE_TIMEOUT_S * 3)
        for n in nodes:
            n.last_heartbeat_ms -= (NODE_TIMEOUT_S + 5) * 1000

        gone = registry.reap()

        assert gone == [], "a coordinator stall wiped out the whole mesh"
        for n in nodes:
            assert registry.get(n.id).status is not NodeStatus.OFFLINE

    def test_nodes_that_re_assert_during_the_grace_period_survive(
            self, registry, monkeypatch):
        """The grace period exists so live nodes can prove they are alive.

        Without it the deferral would only postpone the mass eviction by one
        tick, which fixes nothing.
        """
        nodes = [_register(registry, f"node-{i}") for i in range(3)]
        _steady_state(registry)

        base = time.monotonic()
        stalled = base + NODE_TIMEOUT_S * 3
        monkeypatch.setattr(time, "monotonic", lambda: stalled)
        for n in nodes:
            n.last_heartbeat_ms -= (NODE_TIMEOUT_S + 5) * 1000
        assert registry.reap() == []

        # The nodes check in again, as they would once the loop resumes.
        for n in nodes:
            _beat(registry, n.id)

        monkeypatch.setattr(time, "monotonic", lambda: stalled + 2.0)
        assert registry.reap() == []
        for n in nodes:
            assert registry.get(n.id).status is not NodeStatus.OFFLINE

    def test_a_node_still_silent_after_the_grace_period_is_evicted(
            self, registry, monkeypatch):
        """Deferral, not amnesty: a truly dead node is still collected.

        This is the property that keeps the fix from trading a mass-eviction bug
        for a never-evict bug.
        """
        live, dead = _register(registry, "live"), _register(registry, "dead")
        _steady_state(registry)

        base = time.monotonic()
        stalled = base + NODE_TIMEOUT_S * 3
        monkeypatch.setattr(time, "monotonic", lambda: stalled)
        for n in (live, dead):
            n.last_heartbeat_ms -= (NODE_TIMEOUT_S + 5) * 1000
        assert registry.reap() == []

        _beat(registry, live.id)

        # Walk past the grace window with a normal cadence.
        t = stalled
        gone: list[str] = []
        for _ in range(60):
            t += 2.0
            monkeypatch.setattr(time, "monotonic", lambda t=t: t)
            gone = registry.reap()
            if gone:
                break

        assert gone == [dead.id], f"expected only the silent node, got {gone}"
        assert registry.get(live.id).status is not NodeStatus.OFFLINE

    def test_a_brief_scheduling_jitter_is_not_treated_as_a_stall(
            self, registry, monkeypatch):
        """A slightly late tick is normal and must not defer evictions.

        Treating ordinary jitter as a stall would delay every failover, which is
        the cost side of this trade-off; it has to stay narrow.
        """
        node = _register(registry, "dead-node")
        _steady_state(registry)

        base = time.monotonic()
        monkeypatch.setattr(time, "monotonic", lambda: base + 3.0)
        node.last_heartbeat_ms -= (NODE_TIMEOUT_S + 5) * 1000

        assert registry.reap() == [node.id]


class TestEvictionEventsAreConsistent:
    def test_no_offline_event_when_the_under_lock_recheck_rescues_a_node(
            self, registry, monkeypatch):
        """No ``node_offline`` for a node that was not actually taken offline.

        ``reap`` re-checks under the lock in case a heartbeat landed between the
        snapshot and the write, but the store write, the log line and the event
        publish all sat *outside* that check. A node rescued by the re-check was
        therefore announced offline to every SSE subscriber while remaining
        online in the registry -- the console would paint it dead, and anything
        reacting to ``node_offline`` (tunnel drop, unit requeue, replica
        teardown, model re-partition) would tear down its work for no reason.
        """
        node = _register(registry, "racing-node")
        _steady_state(registry)

        published: list[tuple[str, dict]] = []
        monkeypatch.setattr(registry.events, "publish",
                            lambda kind, data: published.append((kind, data)))

        # Make the node look stale to the snapshot, then let a heartbeat land
        # before the under-lock re-check -- exactly the race the re-check exists
        # for.
        node.last_heartbeat_ms -= (NODE_TIMEOUT_S + 5) * 1000
        real_lock = registry._lock

        class _RescuingLock:
            """Land a heartbeat between the snapshot and the re-check.

            ``reap`` takes the lock twice: once to snapshot, then again per
            stale node to re-check. Rescuing on the *first* acquisition would
            make the node look fresh to the outer ``if`` and the loop body would
            never run at all -- the test would pass without ever reaching the
            code it is about. Rescue on the second acquisition only.
            """

            def __init__(self):
                self.acquisitions = 0

            def __enter__(self):
                real_lock.acquire()
                self.acquisitions += 1
                if self.acquisitions == 2:
                    node.last_heartbeat_ms = int(time.time() * 1000)
                return self

            def __exit__(self, *exc):
                real_lock.release()
                return False

        lock = _RescuingLock()
        monkeypatch.setattr(registry, "_lock", lock)

        gone = registry.reap()

        assert lock.acquisitions >= 2, (
            "reap never reached the under-lock re-check, so this test proved "
            "nothing about it")
        assert gone == [], "a node rescued by the re-check was reported evicted"
        assert not [k for k, _ in published if k == "node_offline"], (
            f"published a node_offline event for a node that stayed online: "
            f"{published}")

    def test_a_real_eviction_does_publish_exactly_one_event(self, registry,
                                                            monkeypatch):
        """The inverse control: suppressing the spurious event must not
        suppress the legitimate one, or failover would never be triggered."""
        node = _register(registry, "dead-node")
        _steady_state(registry)

        published: list[tuple[str, dict]] = []
        monkeypatch.setattr(registry.events, "publish",
                            lambda kind, data: published.append((kind, data)))

        node.last_heartbeat_ms -= (NODE_TIMEOUT_S + 5) * 1000
        assert registry.reap() == [node.id]

        offline = [d for k, d in published if k == "node_offline"]
        assert len(offline) == 1, f"expected one node_offline event, got {published}"
        assert offline[0]["id"] == node.id
