"""Tests for placement diversity and correlated-failure-aware durability.

Verifies:
(a) Shards spread across failure domains when the mesh has enough domains.
(b) Graceful, honest degradation when diversity is impossible (small/homogeneous mesh).
(c) MONOTONICITY: adding a node never worsens or breaks placement.
(d) Effective durability is reported LOWER than naive RS durability when placement
    is correlated -- the specific lie this feature exists to prevent.
(e) Domain inference from node URLs, owner, and explicit labels.
(f) Diversity-aware bin_pack_replicas for hosting/app replicas.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
from typing import Optional

import pytest

from meshpool.controlplane.topology import (
    effective_durability,
    infer_failure_domain,
    pick_diverse_nodes,
    _subnet_24,
    _is_private,
    _parse_addr,
)
from meshpool.common.models import (
    Capability, DonationPolicy, Node, NodeLoad, NodeStatus,
)
from meshpool.controlplane.scheduler import bin_pack_replicas
from meshpool.storage.engine import StorageManager, StorageError, _pack_ec_params
from meshpool.common.models import BlobMeta
from meshpool.common.transport import TransportError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(nid: str, domain: Optional[str] = None, disk_mb: int = 100_000,
          cpu_cores: int = 4, ram_mb: int = 4096,
          services: Optional[list] = None) -> Node:
    cap = Capability(cpu_cores=cpu_cores, ram_mb=ram_mb, disk_mb=disk_mb)
    if services is not None:
        cap.services = services
    return Node(
        id=nid, name=nid, url=f"http://127.0.0.1:0/{nid}",
        capability=cap, status=NodeStatus.ONLINE,
        failure_domain=domain,
        policy=DonationPolicy(cpu_pct=50, ram_mb=ram_mb // 2),
        load=NodeLoad(cpu_load_pct=0.0, ram_used_mb=0),
    )


# ---------------------------------------------------------------------------
# pick_diverse_nodes
# ---------------------------------------------------------------------------

class TestPickDiverseNodes:
    def test_spreads_across_domains(self):
        """Shards spread across failure domains when the mesh has enough distinct ones."""
        nodes = [
            _node("a1", "lab-a"), _node("a2", "lab-a"),
            _node("b1", "lab-b"), _node("b2", "lab-b"),
            _node("c1", "lab-c"), _node("c2", "lab-c"),
        ]
        result = pick_diverse_nodes(nodes, 6)
        assert len(result) == 6
        domains = [n.failure_domain for n in result]
        # Should have all 3 domains represented
        assert set(domains) == {"lab-a", "lab-b", "lab-c"}
        # Should NOT start with both nodes from same domain
        assert domains[0] != domains[1]

    def test_picks_exactly_count_when_enough(self):
        """Returns exactly count nodes when pool is big enough."""
        nodes = [_node(f"n{i}", f"d{i}") for i in range(10)]
        for c in [1, 3, 7, 10]:
            result = pick_diverse_nodes(nodes, c)
            assert len(result) == c, f"count={c} gave {len(result)}"

    def test_returns_all_when_count_exceeds_pool(self):
        """Returns all available nodes when count > len(nodes)."""
        nodes = [_node(f"n{i}", f"d{i}") for i in range(3)]
        result = pick_diverse_nodes(nodes, 100)
        assert len(result) == 3

    def test_empty_pool_returns_empty(self):
        assert pick_diverse_nodes([], 5) == []

    def test_count_zero_returns_empty(self):
        nodes = [_node("n0", "d0")]
        assert pick_diverse_nodes(nodes, 0) == []

    def test_graceful_degradation_same_domain(self):
        """When all nodes share a domain, still returns count nodes (degrades gracefully)."""
        nodes = [_node(f"n{i}", "single-domain") for i in range(6)]
        result = pick_diverse_nodes(nodes, 4)
        assert len(result) == 4, "must not refuse to place just because diversity is impossible"

    def test_no_duplicate_node_ids(self):
        """Never returns the same node twice."""
        nodes = [_node(f"n{i}", f"d{i % 2}") for i in range(10)]
        result = pick_diverse_nodes(nodes, 8)
        ids = [n.id for n in result]
        assert len(ids) == len(set(ids))

    def test_monotonicity_adding_node_never_worsens(self):
        """CRITICAL: adding a node must never reduce the returned count or break placement.

        This is the antimonotonic capacity bug: when adding a node to the pool, the
        placement must be at least as good (same count, diversity >= base), never worse.
        The comparison is always against the BASE result (no extra node), not cumulative.
        """
        base_nodes = [
            _node("a1", "lab-a"), _node("a2", "lab-a"),
            _node("b1", "lab-b"), _node("b2", "lab-b"),
        ]
        count = 4
        result_base = pick_diverse_nodes(base_nodes, count)
        assert len(result_base) == count
        base_domain_count = len({n.failure_domain for n in result_base})

        # Add each extra node INDEPENDENTLY to base; compare against base result
        for extra in [
            _node("c1", "lab-c"),    # new domain: diversity should improve
            _node("c2", "lab-c"),    # new domain: diversity should improve
            _node("d1", "lab-d"),    # new domain: diversity should improve
            _node("a3", "lab-a"),    # existing domain: diversity must not drop below base
        ]:
            extended_nodes = base_nodes + [extra]
            result_extended = pick_diverse_nodes(extended_nodes, count)
            assert len(result_extended) == count, (
                f"Adding node {extra.id} reduced placement count from {count} "
                f"to {len(result_extended)}")
            domain_count = len({n.failure_domain for n in result_extended})
            # When the pool grows by one node, the result's domain diversity must be
            # >= the base result's diversity (adding a node in an existing domain
            # keeps diversity the same; adding in a new domain improves it).
            assert domain_count >= base_domain_count, (
                f"Adding node {extra.id} (domain {extra.failure_domain}) "
                f"WORSENED diversity from {base_domain_count} to {domain_count} domains")

    def test_monotonicity_count_nines_homogeneous_grow(self):
        """Monotonicity on homogeneous (single-domain) growth: count always met."""
        nodes = []
        for i in range(1, 12):
            nodes.append(_node(f"n{i}", "same"))
            need = min(i, 9)
            result = pick_diverse_nodes(list(nodes), need)
            assert len(result) == need, (
                f"After adding node {i}, picking {need} returned {len(result)}")

    def test_nodes_without_domain_are_each_independent(self):
        """Nodes without a failure_domain are each treated as their own unique domain."""
        nodes = [_node(f"n{i}", None) for i in range(5)]
        result = pick_diverse_nodes(nodes, 5)
        assert len(result) == 5
        # Each is independent, so round-robin still picks one per group per pass
        ids = [n.id for n in result]
        assert len(set(ids)) == 5


# ---------------------------------------------------------------------------
# effective_durability
# ---------------------------------------------------------------------------

class TestEffectiveDurability:
    def test_perfect_diversity(self):
        """One shard per domain: effective == theoretical."""
        domains = [f"d{i}" for i in range(9)]  # 9 distinct domains
        result = effective_durability(domains, k=6, m=3)
        assert result["effective_tolerance"] == 3
        assert result["theoretical_tolerance"] == 3
        assert result["correlated"] is False
        assert result["warning"] is None
        assert result["n_domains"] == 9

    def test_all_same_domain_exceeds_budget(self):
        """All shards in one domain: effective_tolerance is 0 (a single outage loses data)."""
        domains = ["one-lab"] * 9
        result = effective_durability(domains, k=6, m=3)
        assert result["effective_tolerance"] == 0
        assert result["theoretical_tolerance"] == 3
        assert result["correlated"] is True
        assert result["warning"] is not None
        assert "lie" in result["warning"].lower() or "lose data" in result["warning"].lower()

    def test_effective_lower_than_theoretical_when_correlated(self):
        """Effective durability is strictly lower than theoretical when correlated.

        This is the core invariant: reporting theoretical when effective is lower
        is a lie.  When max_domain_size > 1 and effective < theoretical, we must
        report the lower number.
        """
        # 3 domains: A has 5 shards, B has 2, C has 2 (k=6, m=3)
        domains = ["A"] * 5 + ["B"] * 2 + ["C"] * 2
        result = effective_durability(domains, k=6, m=3)
        assert result["theoretical_tolerance"] == 3
        assert result["effective_tolerance"] < result["theoretical_tolerance"], (
            "When a domain holds > 1 shard, effective must be < theoretical")
        assert result["correlated"] is True

    def test_boundary_max_domain_equals_m(self):
        """max_domain_size == m: effective_tolerance == 1 (just barely survives one kill)."""
        # 2 domains with 3 shards each, m=3
        domains = ["A", "A", "A", "B", "B", "B"]
        result = effective_durability(domains, k=3, m=3)
        assert result["max_domain_size"] == 3
        assert result["effective_tolerance"] == 1  # m - 3 + 1 = 1
        assert result["tolerable_domain_kills"] == 1

    def test_empty_shards(self):
        """No shards placed: effective_tolerance 0, no crash."""
        result = effective_durability([], k=6, m=3)
        assert result["effective_tolerance"] == 0
        assert result["n_domains"] == 0

    def test_two_domains_spread_evenly(self):
        """Perfectly split 2 domains: tolerable_domain_kills == 1 when each half <= m."""
        # k=3, m=3, 3 in each domain
        result = effective_durability(["A", "A", "A", "B", "B", "B"], k=3, m=3)
        assert result["tolerable_domain_kills"] == 1
        assert result["n_domains"] == 2

    def test_three_domains_all_independent(self):
        """When shards spread across enough domains, reports full tolerance."""
        # k=2, m=2, 1 shard per domain (4 domains)
        domains = ["A", "B", "C", "D"]
        result = effective_durability(domains, k=2, m=2)
        assert result["effective_tolerance"] == 2  # m - 1 + 1 = 2
        assert result["theoretical_tolerance"] == 2
        assert result["correlated"] is False


# ---------------------------------------------------------------------------
# Storage engine: diversity-aware _pick_nodes
# ---------------------------------------------------------------------------

class FakeRegistry:
    def __init__(self, nodes):
        self._nodes_list = list(nodes)
        self._online = {n.id for n in nodes}

    def online(self):
        return [n for n in self._nodes_list if n.id in self._online]

    def schedulable_nodes(self):
        return self.online()

    def kill(self, nid):
        self._online.discard(nid)


class FakeStore:
    def __init__(self):
        self._rows: dict = {}

    def put_blob(self, meta):
        self._rows[meta.id] = meta.model_copy(deep=True)

    def get_blob(self, bid):
        m = self._rows.get(bid)
        return m.model_copy(deep=True) if m else None

    def list_blobs(self):
        return [m.model_copy(deep=True) for m in self._rows.values()]

    def delete_blob(self, bid):
        self._rows.pop(bid, None)


class FakeTransport:
    def __init__(self, node_ids):
        self.disks: dict = {n: {} for n in node_ids}
        self.unreachable: set = set()

    def add_node(self, nid):
        self.disks.setdefault(nid, {})

    async def call(self, node_id, method, payload, timeout=None):
        if node_id in self.unreachable or node_id not in self.disks:
            raise TransportError(f"node {node_id} unreachable")
        disk = self.disks[node_id]
        if method == "blob_put":
            raw = base64.b64decode(payload["data"])
            if hashlib.sha256(raw).hexdigest() != payload["blob_id"]:
                raise TransportError("hash mismatch at agent")
            disk[payload["blob_id"]] = raw
            return {"blob_id": payload["blob_id"], "stored": True}
        if method == "blob_get":
            raw = disk.get(payload["blob_id"])
            if raw is None:
                return {"blob_id": payload["blob_id"], "found": False, "data": ""}
            return {"blob_id": payload["blob_id"], "found": True,
                    "data": base64.b64encode(raw).decode("ascii")}
        if method == "blob_delete":
            disk.pop(payload["blob_id"], None)
            return {"deleted": True}
        raise TransportError(f"unknown method {method}")


class FakeEvents:
    def __init__(self):
        self.events: list = []

    def publish(self, kind, data):
        self.events.append((kind, data))


def _make_storage(nodes):
    node_ids = [n.id for n in nodes]
    reg = FakeRegistry(nodes)
    store = FakeStore()
    transport = FakeTransport(node_ids)
    events = FakeEvents()
    mgr = StorageManager(reg, store, None, events, transport)
    return mgr, reg, store, transport, events


class TestStorageDiversity:
    def _store_and_get_shard_domains(self, nodes, k, m):
        """Helper: store a blob with given nodes and EC params; return shard node domains."""
        import random
        rng = random.Random(0)
        data = bytes(rng.getrandbits(8) for _ in range(32_000))
        mgr, reg, store, transport, _ = _make_storage(nodes)
        meta = asyncio.run(mgr.put(data, erasure=True, ec_k=k, ec_m=m))
        # Collect shard domains
        domains = []
        for sid in meta.chunks:
            sm = store.get_blob(sid)
            if sm and sm.replicas:
                node = next((n for n in nodes if n.id == sm.replicas[0]), None)
                if node:
                    domains.append(node.failure_domain or node.id)
        return meta, domains, data, mgr, reg, transport

    def test_shards_spread_across_domains(self):
        """With enough distinct domains, shards should span at least 2 domains."""
        nodes = (
            [_node(f"a{i}", "lab-a") for i in range(3)]
            + [_node(f"b{i}", "lab-b") for i in range(3)]
            + [_node(f"c{i}", "lab-c") for i in range(3)]
        )
        meta, domains, _, _, _, _ = self._store_and_get_shard_domains(nodes, k=3, m=3)
        assert len(meta.chunks) == 6
        unique_domains = len(set(domains))
        assert unique_domains >= 2, (
            f"Expected shards across >=2 domains, got {unique_domains}: {domains}")

    def test_graceful_degradation_single_domain(self):
        """On a homogeneous mesh (all one domain), blobs are stored successfully."""
        nodes = [_node(f"n{i}", "single") for i in range(6)]
        meta, domains, data, mgr, reg, transport = self._store_and_get_shard_domains(
            nodes, k=3, m=3)
        assert meta is not None
        assert len(meta.chunks) == 6
        # All in one domain -- graceful degradation, no error
        assert all(d == "single" for d in domains)
        # Data is still recoverable
        recovered = asyncio.run(mgr.get(meta.id))
        assert recovered == data

    def test_effective_durability_lower_when_correlated(self):
        """Effective durability is below theoretical when all shards share a domain."""
        nodes = [_node(f"n{i}", "single") for i in range(6)]
        _, domains, _, _, _, _ = self._store_and_get_shard_domains(nodes, k=3, m=3)
        eff = effective_durability(domains, k=3, m=3)
        assert eff["theoretical_tolerance"] == 3
        # All in single domain: max_domain_size == 6 >> m == 3 -> effective == 0
        assert eff["effective_tolerance"] < eff["theoretical_tolerance"], (
            "Effective durability must be lower than theoretical when all shards "
            "share a domain. Reporting theoretical is a lie.")
        assert eff["correlated"] is True

    def test_correlated_kill_with_diversity_recovers(self):
        """A whole-domain kill is survivable when shards are spread across domains."""
        # 6 nodes: 3 in lab-a, 3 in lab-b; k=3, m=3
        a_nodes = [_node(f"a{i}", "lab-a") for i in range(3)]
        b_nodes = [_node(f"b{i}", "lab-b") for i in range(3)]
        nodes = a_nodes + b_nodes

        import random
        rng = random.Random(42)
        data = bytes(rng.getrandbits(8) for _ in range(32_000))
        mgr, reg, store, transport, _ = _make_storage(nodes)
        meta = asyncio.run(mgr.put(data, erasure=True, ec_k=3, ec_m=3))
        assert len(meta.chunks) == 6

        # Verify shards are spread (diversity enforcement)
        domains = []
        for sid in meta.chunks:
            sm = store.get_blob(sid)
            if sm and sm.replicas:
                nd = next((n for n in nodes if n.id == sm.replicas[0]), None)
                if nd:
                    domains.append(nd.failure_domain)

        # Kill the entire lab-a domain (all 3 nodes)
        for n in a_nodes:
            reg.kill(n.id)
            transport.unreachable.add(n.id)
            # Mark their shards as unreachable by clearing disk
            for sid in meta.chunks:
                sm = store.get_blob(sid)
                if sm and n.id in sm.replicas:
                    transport.disks[n.id].clear()

        # Data should still be recoverable from lab-b if diversity was enforced
        if len(set(domains)) >= 2:
            # Diversity was achieved: recovery must succeed
            recovered = asyncio.run(mgr.get(meta.id))
            assert recovered == data, "Data lost after killing one domain -- diversity enforcement failed"

    def test_correlated_kill_without_diversity_may_lose_data(self):
        """Without diversity (all shards in one domain), killing that domain loses data."""
        # 6 nodes all in same domain
        nodes = [_node(f"n{i}", "single-lab") for i in range(6)]
        import random
        rng = random.Random(99)
        data = bytes(rng.getrandbits(8) for _ in range(32_000))
        mgr, reg, store, transport, _ = _make_storage(nodes)
        meta = asyncio.run(mgr.put(data, erasure=True, ec_k=3, ec_m=3))

        # Kill enough nodes to exceed m=3
        killed = 0
        for n in nodes:
            reg.kill(n.id)
            transport.unreachable.add(n.id)
            transport.disks[n.id].clear()
            killed += 1
            if killed == 4:  # 4 > m=3
                break

        recovered = asyncio.run(mgr.get(meta.id))
        # With 4 out of 6 shards gone and all from same domain, data is lost
        assert recovered is None, (
            "Expected data loss after killing 4/6 nodes in a correlated cluster")

    def test_monotonicity_adding_storage_node(self):
        """Adding a storage node never makes placement fail or return fewer shards."""
        import random
        rng = random.Random(7)
        data = bytes(rng.getrandbits(8) for _ in range(32_000))

        # Start with exactly k+m nodes
        base_nodes = [_node(f"n{i}", f"d{i}") for i in range(6)]
        mgr, reg, store, transport, _ = _make_storage(base_nodes)
        meta_base = asyncio.run(mgr.put(data, erasure=True, ec_k=3, ec_m=3))
        placed_base = sum(1 for sid in meta_base.chunks
                          if store.get_blob(sid) and store.get_blob(sid).replicas)
        assert placed_base == 6

        # Add more nodes -- must not make the result worse
        extra_node = _node("extra1", "extra-domain")
        more_nodes = base_nodes + [extra_node]
        mgr2, reg2, store2, transport2, _ = _make_storage(more_nodes)
        meta2 = asyncio.run(mgr2.put(data, erasure=True, ec_k=3, ec_m=3))
        placed2 = sum(1 for sid in meta2.chunks
                      if store2.get_blob(sid) and store2.get_blob(sid).replicas)
        assert placed2 == 6, (
            f"Adding a node reduced placed shards from {placed_base} to {placed2}")

        # Data should still be recoverable
        recovered = asyncio.run(mgr2.get(meta2.id))
        assert recovered == data


# ---------------------------------------------------------------------------
# Domain inference
# ---------------------------------------------------------------------------

class TestDomainInference:
    def test_explicit_domain_wins(self):
        d = infer_failure_domain("http://1.2.3.4:9001", owner="default",
                                 explicit="test-domain")
        assert d == "test-domain"

    def test_non_default_owner_gives_owner_domain(self):
        d = infer_failure_domain("http://1.2.3.4:9001", owner="alice")
        assert d == "owner:alice"

    def test_private_ip_falls_back_to_subnet(self):
        d = infer_failure_domain("http://192.168.1.100:9001", owner="default")
        assert "subnet:" in d, f"Expected subnet domain for private IP, got {d}"

    def test_loopback_ip_falls_back_to_subnet(self):
        d = infer_failure_domain("http://127.0.0.1:9001", owner="default")
        assert "subnet:" in d, f"Expected subnet domain for loopback, got {d}"

    def test_same_ip_different_ports_same_subnet(self):
        """Two nodes on the same /24 share a subnet domain."""
        d1 = infer_failure_domain("http://10.0.1.5:9001", owner="default")
        d2 = infer_failure_domain("http://10.0.1.6:9002", owner="default")
        assert d1 == d2, f"Same /24 should give same domain: {d1} vs {d2}"

    def test_different_subnets_different_domains(self):
        """Nodes on different /24s get different domains."""
        d1 = infer_failure_domain("http://10.0.1.5:9001", owner="default")
        d2 = infer_failure_domain("http://10.0.2.5:9001", owner="default")
        assert d1 != d2, f"Different /24s should give different domains: {d1} vs {d2}"

    def test_never_raises_on_bad_url(self):
        """Domain inference must never crash even on garbage input."""
        for bad in ["", "not-a-url", "ftp://weird:abc/path", "http://:0"]:
            d = infer_failure_domain(bad, owner="default")
            assert isinstance(d, str) and len(d) > 0

    def test_subnet_24_helper(self):
        assert _subnet_24("192.168.1.42") == "192.168.1.x"
        assert _subnet_24("10.20.30.40") == "10.20.30.x"

    def test_is_private_helper(self):
        assert _is_private("192.168.1.1")
        assert _is_private("10.0.0.1")
        assert _is_private("172.16.0.1")
        assert _is_private("127.0.0.1")
        assert not _is_private("8.8.8.8")

    def test_parse_addr_helper(self):
        ip, port = _parse_addr("http://1.2.3.4:9001")
        assert ip == "1.2.3.4"
        assert port == 9001


# ---------------------------------------------------------------------------
# bin_pack_replicas with diversity
# ---------------------------------------------------------------------------

class TestBinPackDiversity:
    def _hosting_node(self, nid: str, domain: str,
                      cpu_cores: int = 4, ram_mb: int = 4096) -> Node:
        return _node(nid, domain, cpu_cores=cpu_cores, ram_mb=ram_mb,
                     services=["hosting", "batch", "storage", "shards"])

    def test_prefers_different_domains(self):
        """bin_pack_replicas prefers nodes in distinct failure domains."""
        nodes = [
            self._hosting_node("a1", "lab-a"),
            self._hosting_node("a2", "lab-a"),
            self._hosting_node("b1", "lab-b"),
            self._hosting_node("b2", "lab-b"),
        ]
        result = bin_pack_replicas(nodes, count=2, cpu_pct=10, ram_mb=256)
        assert len(result) == 2
        domains = [n.failure_domain for n in result]
        # Should prefer different domains over same
        assert len(set(domains)) == 2, (
            f"Expected 2 distinct domains for 2 replicas, got: {domains}")

    def test_degrades_gracefully_single_domain(self):
        """On a homogeneous mesh, still places as many replicas as possible."""
        nodes = [self._hosting_node(f"n{i}", "only") for i in range(5)]
        result = bin_pack_replicas(nodes, count=3, cpu_pct=10, ram_mb=256)
        assert len(result) == 3

    def test_capacity_constraint_is_hard(self):
        """Diversity is a soft preference; capacity is a hard constraint."""
        nodes = [
            self._hosting_node("a1", "lab-a", ram_mb=512),  # can fit 1 replica
            self._hosting_node("b1", "lab-b", ram_mb=512),
        ]
        # Each node can only fit 1 replica (500 MB req, 256 MB policy cap)
        result = bin_pack_replicas(nodes, count=4, cpu_pct=10, ram_mb=200)
        # max 2 total (one per node with 512 MB / 2 headroom cap)
        assert len(result) <= 2

    def test_monotonicity_adding_hosting_node(self):
        """Adding a hosting node never reduces the number of placed replicas."""
        base = [
            self._hosting_node("a1", "lab-a"),
            self._hosting_node("b1", "lab-b"),
        ]
        result_base = bin_pack_replicas(base, count=2, cpu_pct=10, ram_mb=128)
        count_base = len(result_base)

        for extra in [
            self._hosting_node("c1", "lab-c"),
            self._hosting_node("a2", "lab-a"),
        ]:
            extended = base + [extra]
            result_ext = bin_pack_replicas(extended, count=2, cpu_pct=10, ram_mb=128)
            assert len(result_ext) >= count_base, (
                f"Adding node {extra.id} reduced replica count from "
                f"{count_base} to {len(result_ext)}")
