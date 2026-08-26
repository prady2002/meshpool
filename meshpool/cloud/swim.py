"""SWIM-style gossip membership for churny residential MeshPool nodes.

Scope: this is the membership engine and deterministic simulation harness used by
the control plane. It implements SWIM's load-bearing pieces: direct probes,
indirect probes through k peers, suspicion before death, and bounded probe cadence.
It is not a Raft/etcd membership store and deliberately makes no global-consensus
claim.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from .nat import NAT_KEEPALIVE_S

MemberState = Literal["alive", "suspect", "dead"]


@dataclass
class Member:
    node_id: str
    state: MemberState = "alive"
    incarnation: int = 0
    suspect_since_tick: Optional[int] = None
    last_heard_tick: int = 0


@dataclass
class SwimConfig:
    indirect_checks: int = 3
    suspicion_ticks: int = 3
    probe_interval_s: float = NAT_KEEPALIVE_S
    probe_count: int = 1


class SwimMembership:
    """One node's local SWIM membership view."""

    def __init__(self, self_id: str, peers: list[str],
                 config: Optional[SwimConfig] = None, seed: int = 1) -> None:
        self.self_id = self_id
        self.config = config or SwimConfig()
        self.rng = random.Random(seed)
        self.members: dict[str, Member] = {
            peer: Member(peer, last_heard_tick=0) for peer in sorted(set(peers + [self_id]))
        }
        self._cursor = 0

    def alive_ids(self) -> set[str]:
        return {nid for nid, m in self.members.items() if m.state == "alive"}

    def mark_alive(self, node_id: str, tick: int, incarnation: int = 0) -> None:
        m = self.members.setdefault(node_id, Member(node_id))
        if incarnation >= m.incarnation:
            m.state = "alive"
            m.incarnation = incarnation
            m.suspect_since_tick = None
            m.last_heard_tick = tick

    def suspect(self, node_id: str, tick: int) -> None:
        m = self.members.setdefault(node_id, Member(node_id))
        if m.state == "alive":
            m.state = "suspect"
            m.suspect_since_tick = tick

    def declare_dead(self, node_id: str) -> None:
        m = self.members.setdefault(node_id, Member(node_id))
        m.state = "dead"
        m.suspect_since_tick = None

    def _targets(self) -> list[str]:
        peers = [p for p in sorted(self.members) if p != self.self_id]
        if not peers:
            return []
        count = min(max(1, self.config.probe_count), len(peers))
        out = []
        for _ in range(count):
            out.append(peers[self._cursor % len(peers)])
            self._cursor += 1
        return out

    def _indirect_peers(self, target: str) -> list[str]:
        candidates = [
            p for p, m in self.members.items()
            if p not in (self.self_id, target) and m.state != "dead"
        ]
        self.rng.shuffle(candidates)
        return candidates[:self.config.indirect_checks]

    def tick(self, tick: int, probe: Callable[[str, str], bool]) -> None:
        for target in self._targets():
            if self.members[target].state == "dead":
                continue
            direct_ok = probe(self.self_id, target)
            indirect_ok = False
            if not direct_ok:
                indirect_ok = any(
                    probe(self.self_id, peer) and probe(peer, target)
                    for peer in self._indirect_peers(target)
                )
            if direct_ok or indirect_ok:
                self.mark_alive(target, tick, self.members[target].incarnation)
            else:
                self.suspect(target, tick)

        for node_id, member in list(self.members.items()):
            if node_id == self.self_id or member.state != "suspect":
                continue
            assert member.suspect_since_tick is not None
            if tick - member.suspect_since_tick >= self.config.suspicion_ticks:
                self.declare_dead(node_id)


class SwimCluster:
    """Deterministic multi-node SWIM simulation used by the acceptance gate."""

    def __init__(self, node_ids: list[str], *, config: Optional[SwimConfig] = None,
                 seed: int = 1) -> None:
        self.node_ids = list(node_ids)
        self.config = config or SwimConfig()
        self.memberships = {
            nid: SwimMembership(nid, self.node_ids, self.config, seed + i)
            for i, nid in enumerate(self.node_ids)
        }
        self.live: set[str] = set(self.node_ids)
        self.blocked_links: set[tuple[str, str]] = set()
        self.tick_count = 0

    def churn_offline(self, node_ids: set[str]) -> None:
        self.live -= set(node_ids)

    def block_one_way(self, src: str, dst: str) -> None:
        self.blocked_links.add((src, dst))

    def probe(self, src: str, dst: str) -> bool:
        return src in self.live and dst in self.live and (src, dst) not in self.blocked_links

    def tick(self) -> None:
        self.tick_count += 1
        for node_id in sorted(self.live):
            self.memberships[node_id].tick(self.tick_count, self.probe)

    def converged(self) -> bool:
        for node_id in self.live:
            view = self.memberships[node_id]
            for peer in self.node_ids:
                state = view.members[peer].state
                if peer in self.live and state == "dead":
                    return False
                if peer not in self.live and state != "dead":
                    return False
        return True
