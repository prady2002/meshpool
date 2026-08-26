"""NAT traversal planning for MeshPool's QUIC fabric.

This module models the part of NAT traversal that must be deterministic and
honest before any packet is sent: classify a node from measured probe results,
decide whether hole punching is physically possible for a pair, and select the
lowest-RTT relay when it is not.  It deliberately does **not** claim that hole
punching always succeeds when possible; the production planning number remains
libp2p DCUtR's ~73% across mixed NATs, with 25-40% permanent relay dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Protocol

from ..common.transport import AgentTransport, ProxyResult

NatClass = Literal[
    "public",
    "full_cone",
    "address_restricted",
    "port_restricted",
    "symmetric",
    "cgnat",
]
PathKind = Literal["direct", "relay"]

NAT_KEEPALIVE_S = 22.0
HOLE_PUNCH_MAX_ATTEMPTS = 3
HOLE_PUNCH_ATTEMPT_BUDGET_MS = 250
EXPECTED_DCUTR_SUCCESS = 0.73
PERMANENT_RELAY_DEPENDENCY_RANGE = (0.25, 0.40)


def _norm(cls: str) -> NatClass:
    # Back-compat with sim.topology's older name.
    if cls == "restricted":
        return "address_restricted"
    if cls not in {
        "public", "full_cone", "address_restricted", "port_restricted",
        "symmetric", "cgnat",
    }:
        raise ValueError(f"unknown NAT class {cls!r}")
    return cls  # type: ignore[return-value]


_CAN_DIRECT: dict[NatClass, dict[NatClass, bool]] = {
    "public":             {"public": True, "full_cone": True, "address_restricted": True,
                           "port_restricted": True, "symmetric": True, "cgnat": True},
    "full_cone":          {"public": True, "full_cone": True, "address_restricted": True,
                           "port_restricted": True, "symmetric": True, "cgnat": False},
    "address_restricted": {"public": True, "full_cone": True, "address_restricted": True,
                           "port_restricted": True, "symmetric": False, "cgnat": False},
    "port_restricted":    {"public": True, "full_cone": True, "address_restricted": True,
                           "port_restricted": True, "symmetric": False, "cgnat": False},
    "symmetric":          {"public": True, "full_cone": True, "address_restricted": False,
                           "port_restricted": False, "symmetric": False, "cgnat": False},
    "cgnat":              {"public": True, "full_cone": False, "address_restricted": False,
                           "port_restricted": False, "symmetric": False, "cgnat": False},
}


@dataclass(frozen=True)
class NatProbeObservation:
    """Measured output from STUN-like probes; never user-supplied configuration."""

    local_endpoint: tuple[str, int]
    mapped_endpoints: tuple[tuple[str, int], ...]
    udp_reachable: bool = True
    received_from_unknown_addr: bool = False
    received_from_known_addr_other_port: bool = False
    mapped_ip_is_private_or_shared: bool = False


def detect_nat_class(obs: NatProbeObservation) -> NatClass:
    """Classify NAT behaviour from active observations.

    The decision tree is the RFC 3489 / RFC 4787 shape: compare the local endpoint
    with the reflexive mapping, compare mappings across reflectors, then use
    alternate-source replies to distinguish full-cone from address/port-restricted.
    """
    if not obs.udp_reachable or not obs.mapped_endpoints or obs.mapped_ip_is_private_or_shared:
        return "cgnat"
    if all(ep == obs.local_endpoint for ep in obs.mapped_endpoints):
        return "public"
    if len(set(obs.mapped_endpoints)) > 1:
        return "symmetric"
    if obs.received_from_unknown_addr:
        return "full_cone"
    if obs.received_from_known_addr_other_port:
        return "address_restricted"
    return "port_restricted"


def can_hole_punch(a: str, b: str) -> bool:
    """True only when direct UDP hole punching is physically plausible."""
    ca = _norm(a)
    cb = _norm(b)
    return _CAN_DIRECT[ca][cb] and _CAN_DIRECT[cb][ca]


@dataclass(frozen=True)
class RelayCandidate:
    node_id: str
    rtt_to_src_ms: float
    rtt_to_dst_ms: float
    nat_class: NatClass = "public"

    @property
    def total_rtt_ms(self) -> float:
        return self.rtt_to_src_ms + self.rtt_to_dst_ms


@dataclass(frozen=True)
class TraversalPlan:
    kind: PathKind
    src: str
    dst: str
    src_nat: NatClass
    dst_nat: NatClass
    relay_id: Optional[str] = None
    hole_punch_attempts: int = 0
    attempt_budget_ms: int = HOLE_PUNCH_ATTEMPT_BUDGET_MS
    keepalive_s: float = NAT_KEEPALIVE_S
    reason: str = ""

    @property
    def path(self) -> list[str]:
        if self.kind == "direct":
            return [self.src, self.dst]
        if self.relay_id is None:
            return [self.src, self.dst]
        return [self.src, self.relay_id, self.dst]


def select_lowest_rtt_relay(relays: list[RelayCandidate]) -> RelayCandidate:
    usable = [r for r in relays if _norm(r.nat_class) == "public"]
    if not usable:
        raise RuntimeError("no public relay candidate available")
    return min(usable, key=lambda r: (r.total_rtt_ms, r.node_id))


def plan_traversal(
    src: str,
    dst: str,
    src_nat: str,
    dst_nat: str,
    relays: list[RelayCandidate],
    *,
    max_attempts: int = HOLE_PUNCH_MAX_ATTEMPTS,
) -> TraversalPlan:
    """Return the direct-vs-relay plan with a bounded hole-punch budget."""
    a = _norm(src_nat)
    b = _norm(dst_nat)
    if can_hole_punch(a, b):
        return TraversalPlan(
            "direct", src, dst, a, b,
            hole_punch_attempts=min(max_attempts, HOLE_PUNCH_MAX_ATTEMPTS),
            reason="hole-punch-plausible",
        )
    relay = select_lowest_rtt_relay(relays)
    return TraversalPlan(
        "relay", src, dst, a, b, relay_id=relay.node_id,
        hole_punch_attempts=0,
        reason="structural-zero-success; relay-first",
    )


class NatDirectory(Protocol):
    def nat_class(self, node_id: str) -> str: ...
    def relay_candidates(self, src: str, dst: str) -> list[RelayCandidate]: ...


class NatTraversalTransport:
    """AgentTransport wrapper that records direct vs. relayed delivery decisions.

    It deliberately delegates actual work to another AgentTransport, so engines keep
    using the same seam.  In production the relay is the coordinator/relay node
    forwarding over this seam; in simulation the path is asserted without rewriting
    callers.
    """

    def __init__(self, inner: AgentTransport, directory: NatDirectory) -> None:
        self.inner = inner
        self.directory = directory
        self.last_plan: dict[tuple[str, str], TraversalPlan] = {}

    def plan(self, src: str, dst: str) -> TraversalPlan:
        p = plan_traversal(
            src, dst,
            self.directory.nat_class(src),
            self.directory.nat_class(dst),
            self.directory.relay_candidates(src, dst),
        )
        self.last_plan[(src, dst)] = p
        return p

    def peer_pull_ok(self, node_id: str) -> bool:
        return self.inner.peer_pull_ok(node_id)

    async def call(self, node_id: str, path: str, payload: Optional[dict] = None, *,
                   method: str = "POST", timeout: float = 30.0) -> dict:
        return await self.inner.call(node_id, path, payload, method=method, timeout=timeout)

    async def proxy(self, node_id: str, method: str, url: str, *,
                    params: Any = None, content: bytes = b"",
                    headers: Optional[dict] = None, timeout: float = 15.0) -> ProxyResult:
        return await self.inner.proxy(node_id, method, url, params=params,
                                      content=content, headers=headers, timeout=timeout)

    async def proxy_open(self, node_id: str, method: str, url: str, *,
                         params: Any = None, content: bytes = b"",
                         headers: Optional[dict] = None, timeout: float = 15.0) -> Any:
        return await self.inner.proxy_open(node_id, method, url, params=params,
                                           content=content, headers=headers, timeout=timeout)

    async def close(self) -> None:
        await self.inner.close()


@dataclass
class NatClassificationDirectory:
    """Small in-memory seam for feeding NAT class into future node classification."""

    classes: dict[str, NatClass] = field(default_factory=dict)

    def set_measured(self, node_id: str, observation: NatProbeObservation) -> NatClass:
        cls = detect_nat_class(observation)
        self.classes[node_id] = cls
        return cls

    def nat_class(self, node_id: str) -> str:
        return self.classes[node_id]
