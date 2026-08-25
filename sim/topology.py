"""Declarative topology description for MeshPool simulation (Phase 0.4).

A Topology is a set of NodeSpecs and a function ``link(a, b) -> LinkProfile`` that
returns the emulated WAN characteristics between any two nodes. Everything is
deterministically seeded: same seed + same topology ⇒ byte-identical behaviour.

Link presets are grounded in real-world measurements where possible and cited:
  - RTT values from Cloudflare Radar / CAIDA / WonderNetwork global latency maps (2023).
  - Residential bandwidth: FCC Broadband Data Collection (2023) median US figures.
  - CGNAT prevalence: APNIC "IPv4 address exhaustion" blog posts and RIPE NCC reports
    put global residential CGNAT at 30–50%; we use 35% as a mid-estimate.
  - Loss: typical values from various ISP / CDN operational blogs; 1% IXP is generous
    for intercontinental but matches what overlay networks like Petals report.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

NatClass = Literal["full_cone", "restricted", "port_restricted", "symmetric", "cgnat", "public"]
NodeClass = Literal["anchor", "relay", "constellation", "drifter"]


@dataclass
class NodeSpec:
    """A single simulated peer in the mesh."""

    id: str
    region: str                       # e.g. "us-east", "eu-west", "ap-southeast"
    asn: str                          # e.g. "AS15169"
    subnet: str                       # /24 CIDR, e.g. "192.168.1.0/24"
    nat_class: NatClass = "port_restricted"
    cpu_scale: float = 1.0            # compute multiplier; 1.0 = reference workstation
    ram_mb: int = 16000
    vram_mb: int = 0
    uptime_profile: str = "consumer"  # "consumer" | "server" | "always-on"
    node_class: NodeClass = "constellation"


@dataclass
class JitterDistribution:
    """Gaussian base jitter with a heavy tail.

    With probability ``heavy_tail_prob``, the sample is multiplied by
    ``heavy_tail_factor`` (modelling occasional router-queue spikes).
    This combination is described in RFC 3393 (IPPM delay variation) and is
    a simple but practical approximation of a real ISP jitter distribution.
    """

    sigma_ms: float = 2.0            # gaussian σ
    heavy_tail_prob: float = 0.01    # 1 % of packets
    heavy_tail_factor: float = 10.0  # those packets get 10× the jitter


@dataclass
class LinkProfile:
    """Emulated WAN characteristics for one (src, dst) pair.

    Bandwidth is intentionally asymmetric (residential up is far slower than down).
    """

    rtt_ms: float                         # base one-way RTT × 2 (round-trip)
    jitter: JitterDistribution = field(default_factory=JitterDistribution)
    bw_down_kbps: float = 500_000         # 500 Mbps downstream
    bw_up_kbps: float = 20_000            # 20 Mbps upstream
    loss_prob: float = 0.0                # per-packet drop probability [0, 1]
    reorder_prob: float = 0.0             # per-packet reorder probability


# ---------------------------------------------------------------------------
# Named presets (one-way latency; realistic asymmetric residential bandwidth)
# ---------------------------------------------------------------------------

def _lan() -> LinkProfile:
    """Local area network / Thunderbolt / same-datacenter rack."""
    return LinkProfile(
        rtt_ms=1.0,
        jitter=JitterDistribution(sigma_ms=0.1, heavy_tail_prob=0.001, heavy_tail_factor=5.0),
        bw_down_kbps=1_000_000,   # 1 Gbps
        bw_up_kbps=1_000_000,
        loss_prob=0.0,
        reorder_prob=0.0,
    )


def _metro() -> LinkProfile:
    """Same metro area, residential ISP.  RTT ~15 ms (e.g. NYC–NJ).
    Bandwidth: FCC BDC 2023 median US residential 500/20 Mbps."""
    return LinkProfile(
        rtt_ms=15.0,
        jitter=JitterDistribution(sigma_ms=3.0, heavy_tail_prob=0.01, heavy_tail_factor=10.0),
        bw_down_kbps=500_000,
        bw_up_kbps=20_000,
        loss_prob=0.001,
        reorder_prob=0.0,
    )


def _national() -> LinkProfile:
    """Cross-country, same continent.  RTT ~45 ms (e.g. NYC–LA).
    Bandwidth: FCC BDC 2023 mid-tier plan after contention."""
    return LinkProfile(
        rtt_ms=45.0,
        jitter=JitterDistribution(sigma_ms=5.0, heavy_tail_prob=0.01, heavy_tail_factor=10.0),
        bw_down_kbps=200_000,
        bw_up_kbps=10_000,
        loss_prob=0.003,
        reorder_prob=0.001,
    )


def _intercontinental() -> LinkProfile:
    """Cross-ocean.  RTT ~150 ms (e.g. NA–EU, US–JP).
    Loss 1 %: empirical from Petals overlay-network paper (arXiv 2209.01188 §6)."""
    return LinkProfile(
        rtt_ms=150.0,
        jitter=JitterDistribution(sigma_ms=10.0, heavy_tail_prob=0.02, heavy_tail_factor=10.0),
        bw_down_kbps=100_000,
        bw_up_kbps=5_000,
        loss_prob=0.01,
        reorder_prob=0.002,
    )


PRESETS: dict[str, Callable[[], LinkProfile]] = {
    "lan": _lan,
    "metro": _metro,
    "national": _national,
    "intercontinental": _intercontinental,
}


class Topology:
    """A set of nodes and the link profile between any pair.

    All random decisions (jitter, link assignment, etc.) use a seeded RNG so
    the same ``seed`` always produces identical topology and link assignments.

    Args:
        nodes:      List of :class:`NodeSpec` objects.
        link_fn:    Callable ``(node_a_id, node_b_id) -> LinkProfile``.
                    Defaults to ``_intercontinental`` for every pair.
        seed:       Integer seed for the internal RNG (determinism guarantee).
    """

    def __init__(
        self,
        nodes: list[NodeSpec],
        link_fn: Optional[Callable[[str, str], LinkProfile]] = None,
        seed: int = 42,
    ) -> None:
        self.nodes: dict[str, NodeSpec] = {n.id: n for n in nodes}
        self._link_fn = link_fn or (lambda a, b: _intercontinental())
        self.seed = seed
        # Deterministic per-pair RNG — each pair gets its own seeded state so
        # evaluating link(a,b) does not affect link(c,d).
        self._rng_seed: dict[tuple[str, str], int] = {}
        master = random.Random(seed)
        for nid in sorted(self.nodes):
            for mid in sorted(self.nodes):
                if nid != mid:
                    self._rng_seed[(nid, mid)] = master.randint(0, 2**31)

    def link(self, a: str, b: str) -> LinkProfile:
        """Return the :class:`LinkProfile` for the (a→b) direction.

        Note: link(a,b) and link(b,a) share the same base profile but are
        accessed in the same direction as the call, so asymmetric bandwidth
        (upload vs download) is from the perspective of the *sender*.
        """
        if a == b:
            return _lan()
        return self._link_fn(a, b)

    def rng(self, a: str, b: str) -> random.Random:
        """Return a deterministically seeded RNG for the (a,b) pair."""
        seed = self._rng_seed.get((a, b), self._rng_seed.get((b, a), self.seed))
        return random.Random(seed)

    def node(self, nid: str) -> NodeSpec:
        if nid not in self.nodes:
            raise KeyError(f"unknown node id {nid!r}")
        return self.nodes[nid]

    def relay_nodes(self) -> list[str]:
        """Return all node IDs whose NAT class is ``public``."""
        return [nid for nid, n in self.nodes.items() if n.nat_class == "public"]


# ---------------------------------------------------------------------------
# Preset topologies
# ---------------------------------------------------------------------------

def residential_mixed(n: int = 10, seed: int = 42) -> Topology:
    """Build a realistic residential topology.

    ~35% of nodes are CGNAT (mid-estimate from APNIC/RIPE reports on global
    residential IPv4 exhaustion strategy, 2022–2023).  One anchor node is
    ``public`` to serve as the relay for NAT-blocked paths.
    """
    rng = random.Random(seed)
    nodes: list[NodeSpec] = []

    nat_pool: list[NatClass] = []
    n_cgnat = max(1, round(n * 0.35))
    n_port_restr = max(1, round(n * 0.30))
    n_restricted = max(1, round(n * 0.15))
    n_full_cone = max(0, n - n_cgnat - n_port_restr - n_restricted - 1)  # -1 for public anchor

    nat_pool += ["cgnat"] * n_cgnat
    nat_pool += ["port_restricted"] * n_port_restr
    nat_pool += ["restricted"] * n_restricted
    nat_pool += ["full_cone"] * n_full_cone
    rng.shuffle(nat_pool)

    regions = ["us-east", "us-west", "eu-west", "eu-central", "ap-southeast",
               "ap-northeast", "sa-east", "af-south", "me-central", "au-east"]

    # Node 0 is the public anchor / relay
    nodes.append(NodeSpec(
        id="relay-0",
        region="us-east",
        asn="AS13335",
        subnet="10.0.0.0/24",
        nat_class="public",
        cpu_scale=2.0,
        ram_mb=32000,
        node_class="relay",
    ))

    for i in range(n - 1):
        region = regions[i % len(regions)]
        nat = nat_pool[i] if i < len(nat_pool) else "port_restricted"
        nodes.append(NodeSpec(
            id=f"node-{i + 1}",
            region=region,
            asn=f"AS{10000 + i}",
            subnet=f"192.168.{i + 1}.0/24",
            nat_class=nat,
            cpu_scale=rng.uniform(0.5, 2.0),
            ram_mb=rng.choice([4000, 8000, 16000, 32000]),
            vram_mb=rng.choice([0, 0, 0, 8000, 16000, 24000]),  # mostly CPU-only
            node_class="constellation",
        ))

    def link_fn(a: str, b: str) -> LinkProfile:
        na = next(nd for nd in nodes if nd.id == a)
        nb = next(nd for nd in nodes if nd.id == b)
        if na.region == nb.region:
            return _metro()
        # crude: same continent prefix → national; else intercontinental
        continent_a = na.region.split("-")[0]
        continent_b = nb.region.split("-")[0]
        if continent_a == continent_b:
            return _national()
        return _intercontinental()

    return Topology(nodes=nodes, link_fn=link_fn, seed=seed)
