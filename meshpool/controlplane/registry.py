"""Control-plane node registry: live capability, load, health, trust tier.

Holds authoritative live node state in memory (fast) and writes through to the
SQLite store (durable). A background reaper marks nodes offline when heartbeats
stop, returning the affected node ids so the engines can reschedule their work.

Locking model
-------------
``NodeRegistry._lock`` (threading.RLock) guards all accesses to the in-memory
``_nodes``, ``_tokens``, and ``_token_to_node`` dicts.  It must be held for any
read-modify-write on those structures.

Why a threading lock rather than asyncio.Lock?  FastAPI routes that use plain
``def`` (not ``async def``) run in Starlette's thread pool and can therefore
execute concurrently with the asyncio event loop, so only a real OS-level lock
gives the correct mutual exclusion.  ``threading.RLock`` is reentrant so a
single call chain that acquires it twice (e.g. ``register`` calling
``put_node_token`` on the store which also takes a lock) is safe.

The lock does *not* span the slow ``store.*`` write calls — those are already
serialised by ``Store._lock``.  The registry lock is released before every store
call, so the fast in-memory check and update happen atomically and the durable
write follows immediately under the store's own lock.
"""
from __future__ import annotations

import asyncio
from collections import deque
import secrets as _secrets
import threading
import time
from datetime import datetime
from typing import Callable, Optional

from ..common.db import Store
from ..common.logging import get_logger
from ..common.models import (
    DonationPolicy, Heartbeat, Node, NodeStatus, RegisterRequest, RegisterResponse,
    new_id, now_ms,
)
from ..common.vivaldi import Coordinate, distance_to, update as vivaldi_update

log = get_logger("registry")


def schedulable(node: Node, at: Optional[float] = None) -> bool:
    """Is this node eligible to receive work *right now* under its schedule?"""
    if node.status is not NodeStatus.ONLINE:
        return False
    pol = node.policy
    sched = pol.schedule
    ts = at if at is not None else time.time()
    lt = datetime.fromtimestamp(ts)
    business = (lt.weekday() < 5) and (9 <= lt.hour < 17)
    if sched == "always":
        return True
    if sched == "idle_only":
        return node.load.idle_seconds >= pol.idle_minutes * 60
    if sched == "business_hours":
        return business
    if sched == "outside_business_hours":
        return not business
    return True


class EventBus:
    """Minimal async pub/sub used to push live updates to the dashboard (SSE)."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._history = deque(maxlen=10_000)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def publish(self, kind: str, data: dict) -> None:
        evt = {"kind": kind, "data": data, "ts": now_ms()}
        self._history.append(evt)
        for q in list(self._subscribers):
            try:
                q.put_nowait(evt)
            except asyncio.QueueFull:
                pass

    def recent_for_job(self, job_id: str) -> list[dict]:
        return [
            evt for evt in self._history
            if isinstance(evt.get("data"), dict) and evt["data"].get("id") == job_id
        ]


class NodeRegistry:
    def __init__(self, store: Store, node_timeout_s: int, events: EventBus) -> None:
        self.store = store
        self.node_timeout_s = node_timeout_s
        self.events = events
        # How often reap() is expected to be called, used only to notice when the
        # coordinator itself stalled. See reap() for why that matters.
        self.reap_interval_s: float = 2.0
        self._last_reap_monotonic: Optional[float] = None
        self._eviction_grace_until: float = 0.0
        # _lock guards _nodes, _tokens, _token_to_node (see module docstring).
        self._lock = threading.RLock()
        self._nodes: dict[str, Node] = {n.id: n for n in store.list_nodes()}
        # Optional owner resolver (global mode): maps a verified device_id to the
        # subject that earns credits/reputation for its donations (its linked
        # account, else the device itself). Left None in LAN mode, where the
        # caller-supplied owner is used verbatim.
        self.owner_resolver: Optional[Callable[[str], str]] = None
        # Optional reachability predicate (global mode): returns False for a node that
        # is nominally online (heartbeating) but cannot currently be reached — e.g. a
        # tunnel-only node whose reverse tunnel is not attached. Left None in LAN mode,
        # where every online node is directly reachable. Used to keep the schedulers
        # from dispatching work to a node it cannot actually deliver to.
        self.reachability: Optional[Callable[["Node"], bool]] = None
        # Per-node session tokens. Kept in memory for fast lookup and persisted so
        # returning agents survive coordinator restarts without re-registration.
        persisted_tokens = store.load_node_tokens()
        self._tokens: dict[str, str] = {
            node_id: token for node_id, token in persisted_tokens.items()
            if node_id in self._nodes
        }
        self._token_to_node: dict[str, str] = {
            token: node_id for node_id, token in self._tokens.items()
        }
        # nodes rediscovered on restart are stale until they heartbeat again
        for n in self._nodes.values():
            n.status = NodeStatus.OFFLINE
        # ------------------------------------------------------------------ #
        # Vivaldi network coordinates.
        # _vivaldi maps node_id -> Coordinate (in-memory only; reconstructed
        # from incoming RTT observations after a restart, which is fine because
        # a cold coordinator simply starts with high-error coordinates and the
        # system converges again within a few heartbeat cycles).
        # _vivaldi_lock is separate from _lock so vivaldi updates (on every
        # heartbeat) never contend with registration / token lookups.
        # ------------------------------------------------------------------ #
        self._vivaldi: dict[str, Coordinate] = {}
        self._vivaldi_lock = threading.Lock()

    # --- registration / heartbeat --------------------------------------- #
    def _verified_device_id(self, req: RegisterRequest) -> Optional[str]:
        """Validate any device identity on the request; return the trusted device_id.

        Returns None for a plain (LAN) registration that carries no identity. Raises
        ``ValueError`` if identity fields are present but inconsistent or the
        signature does not prove possession of the private key — so a caller can
        never claim a device_id it does not own, nor collapse onto another node.
        """
        if not (req.device_id or req.public_key or req.signature):
            return None
        if not (req.device_id and req.public_key and req.signature and req.ts_ms):
            raise ValueError("incomplete device identity")
        from ..common.identity import (
            challenge_is_fresh, device_id_for_public_key, enrollment_challenge,
            verify_signature,
        )
        if device_id_for_public_key(req.public_key) != req.device_id:
            raise ValueError("device_id does not match public_key")
        if not challenge_is_fresh(req.ts_ms, now_ms()):
            raise ValueError("stale device signature (clock skew or replay); re-register")
        challenge = enrollment_challenge(req.device_id, req.ts_ms)
        if not verify_signature(req.public_key, challenge, req.signature):
            raise ValueError("invalid device signature")
        return req.device_id

    def register(self, req: RegisterRequest, heartbeat_interval_s: int) -> RegisterResponse:
        device_id = self._verified_device_id(req)
        with self._lock:
            # Identity-bearing nodes (global mode) are keyed on the stable device_id, so a
            # node keeps its id across IP/network changes and many tunnel nodes can share
            # the same sentinel URL. Plain LAN nodes are matched by URL as before.
            if device_id is not None:
                existing = next((n for n in self._nodes.values()
                                 if n.device_id == device_id), None)
            else:
                existing = next((n for n in self._nodes.values() if n.url == req.url), None)
            node_id = existing.id if existing else new_id("node")
            # Resolve the owning subject. In global mode a verified device earns for its
            # linked account (or itself); an explicitly-supplied non-default owner (LAN)
            # is always honoured.
            owner = req.owner
            if device_id is not None and self.owner_resolver is not None \
                    and req.owner in ("", "default"):
                owner = self.owner_resolver(device_id)
            # Infer (or honour the explicit) failure domain for diversity-aware placement.
            # Never raises -- infer_failure_domain degrades gracefully to the machine key.
            try:
                from .topology import infer_failure_domain
                failure_domain = infer_failure_domain(
                    req.url, owner=owner, explicit=req.failure_domain)
            except Exception:  # noqa: BLE001
                failure_domain = None
            node = Node(
                id=node_id, name=req.name, url=req.url, owner=owner,
                device_id=device_id,
                capability=req.capability, link=req.link, policy=req.policy,
                status=NodeStatus.ONLINE, last_heartbeat_ms=now_ms(),
                registered_ms=existing.registered_ms if existing else now_ms(),
                failure_domain=failure_domain,
            )
            self._nodes[node_id] = node
            # Mint (or reuse) a session token so the node can prove its identity later.
            token = self._tokens.get(node_id) or ("nt_" + _secrets.token_urlsafe(24))
            self._tokens[node_id] = token
            self._token_to_node[token] = node_id
        # Store writes happen outside the in-memory lock; Store has its own lock.
        self.store.put_node(node)
        self.store.put_node_token(node_id, token, now_ms())
        log.info("node registered", node=node_id, name=node.name,
                 cores=node.capability.cpu_cores, ram=node.capability.ram_mb)
        self.events.publish("node_registered", {"id": node_id, "name": node.name})
        return RegisterResponse(node_id=node_id, control_plane_time_ms=now_ms(),
                                heartbeat_interval_s=heartbeat_interval_s,
                                node_token=token)

    def node_id_for_token(self, token: str) -> Optional[str]:
        """Server-side node identity from a session token (None if unknown)."""
        with self._lock:
            return self._token_to_node.get(token) if token else None

    def heartbeat(self, hb: Heartbeat) -> bool:
        with self._lock:
            node = self._nodes.get(hb.node_id)
            if not node:
                return False
            node.load = hb.load
            node.policy = hb.policy
            node.status = hb.status
            node.last_heartbeat_ms = now_ms()
        # store write outside _lock so we never hold two locks simultaneously
        self.store.put_node(node)
        self.events.publish("heartbeat", {"id": node.id, "load": hb.load.model_dump()})
        # Process any observed RTTs and update Vivaldi coordinates.
        if hb.observed_rtts:
            for peer_id, rtt_ms in hb.observed_rtts.items():
                if rtt_ms > 0:
                    self.update_vivaldi(hb.node_id, peer_id, rtt_ms)
        return True

    def set_status(self, node_id: str, status: NodeStatus) -> None:
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return
            node.status = status
            node.last_heartbeat_ms = now_ms()
        self.store.put_node(node)
        self.events.publish("node_status", {"id": node_id, "status": status.value})

    def set_policy(self, node_id: str, policy: DonationPolicy) -> None:
        """Apply an owner policy change the agent has already confirmed, so the
        dashboard reflects it immediately instead of waiting for the next heartbeat."""
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return
            node.policy = policy
        self.store.put_node(node)
        self.events.publish("node_policy", {"id": node_id, "cpu_pct": policy.cpu_pct,
                                            "schedule": policy.schedule.value})

    # --- Vivaldi network coordinates ------------------------------------- #
    def update_vivaldi(self, reporter_id: str, peer_id: str, rtt_ms: float) -> None:
        """Update Vivaldi coordinates for *reporter_id* after observing *rtt_ms* to *peer_id*.

        Both nodes' coordinates are updated (symmetric RTT assumption).  Unknown
        nodes get a fresh high-error coordinate on first mention so they converge
        naturally.  The update is locked with a dedicated lock (not _lock) to keep
        the fast heartbeat path from contending with slow registration.
        """
        if rtt_ms <= 0.0:
            return
        with self._vivaldi_lock:
            c_reporter = self._vivaldi.get(reporter_id) or Coordinate.origin()
            c_peer = self._vivaldi.get(peer_id) or Coordinate.origin()
            self._vivaldi[reporter_id] = vivaldi_update(c_reporter, c_peer, rtt_ms)
            # Symmetric update: the peer observes the same RTT from its perspective.
            self._vivaldi[peer_id] = vivaldi_update(c_peer, c_reporter, rtt_ms)

    def get_coordinate(self, node_id: str) -> Optional[Coordinate]:
        """Return the current Vivaldi coordinate for *node_id*, or None if unknown."""
        with self._vivaldi_lock:
            return self._vivaldi.get(node_id)

    def predicted_rtt(self, node_a: str, node_b: str) -> Optional[float]:
        """Return the Vivaldi-predicted RTT (ms) between two nodes, or None.

        Returns None when either node has no coordinate yet, or when both nodes have
        high error (coordinates are untrustworthy) so callers can fall back to a
        non-latency-aware default.  The scheduler uses this to reorder pipeline stages.
        """
        with self._vivaldi_lock:
            ca = self._vivaldi.get(node_a)
            cb = self._vivaldi.get(node_b)
        if ca is None or cb is None:
            return None
        if ca.error >= 0.8 and cb.error >= 0.8:
            return None  # both untrustworthy; don't mislead the scheduler
        return distance_to(ca, cb)

    def vivaldi_snapshot(self) -> dict[str, Coordinate]:
        """Return a copy of all known Vivaldi coordinates (thread-safe snapshot)."""
        with self._vivaldi_lock:
            return {k: v.copy() for k, v in self._vivaldi.items()}

    # --- queries --------------------------------------------------------- #
    def get(self, node_id: str) -> Optional[Node]:
        with self._lock:
            return self._nodes.get(node_id)

    def list(self) -> list[Node]:
        with self._lock:
            return list(self._nodes.values())

    def online(self) -> list[Node]:
        with self._lock:
            return [n for n in self._nodes.values() if n.status is NodeStatus.ONLINE]

    def schedulable_nodes(self) -> list[Node]:
        nodes = [n for n in self.online() if schedulable(n)]
        if self.reachability is not None:
            nodes = [n for n in nodes if self.reachability(n)]
        return nodes

    # --- reaper ---------------------------------------------------------- #
    def reap(self) -> list[str]:
        """Mark nodes offline whose heartbeat is stale. Returns their ids.

        Guards against evicting the entire mesh because *we* stalled.

        The cutoff is wall-clock: a node is offline if its last heartbeat is
        older than ``node_timeout_s``. That is correct only while this loop is
        actually running. If the coordinator's event loop is starved for longer
        than the timeout -- a long GC pause, a blocking call on the loop thread,
        a VM suspend, or simply a badly oversubscribed host -- then on the next
        tick *every* heartbeat looks stale at once, and this method evicts a
        completely healthy mesh in a single pass.

        That is not hypothetical. A gate run on a loaded machine produced::

            22:04:06 WARNING registry  node timed out -> offline node=... sim-node-1
            22:04:06 WARNING registry  node timed out -> offline node=... sim-node-2
            22:04:06 WARNING registry  node timed out -> offline node=... sim-node-3
            22:04:06 WARNING inference cannot re-partition: insufficient survivor capacity

        Three nodes in the same second is the signature: not three independent
        failures, one coordinator stall. The blast radius is everything the
        reaper drives -- tunnels dropped, batch units requeued, hosting replicas
        torn down, model partitions abandoned -- so a hiccup inside the control
        plane becomes a mesh-wide outage, and re-partitioning cannot even
        succeed because no survivors are left to place onto.

        The fix is to distrust the clock after a gap. If far more time passed
        than the reap interval, the missing heartbeats say nothing about the
        nodes; they are evidence about us. Skip the eviction pass and grant a
        grace period so live nodes get a fair chance to re-assert before any
        judgement is made. Kubernetes' node controller and Akka's accrual
        detector both do a version of this for the same reason.

        Deliberately one-sided: a stall delays eviction of genuinely dead nodes
        by one grace period, which costs a little failover latency. Getting it
        wrong the other way tears down the whole mesh.
        """
        mono = time.monotonic()
        last, self._last_reap_monotonic = self._last_reap_monotonic, mono

        if last is not None:
            gap = mono - last
            # Normal gap is reap_interval_s. Treat anything approaching the
            # eviction timeout as a stall -- by then the cutoff is meaningless.
            stall_threshold = max(self.reap_interval_s * 3, self.node_timeout_s * 0.5)
            if gap > stall_threshold:
                self._eviction_grace_until = mono + max(gap, self.node_timeout_s)
                log.warning(
                    "reaper stalled; deferring evictions",
                    gap_s=round(gap, 1),
                    expected_s=self.reap_interval_s,
                    grace_s=round(self._eviction_grace_until - mono, 1),
                )
                return []

        if mono < self._eviction_grace_until:
            return []

        cutoff = now_ms() - self.node_timeout_s * 1000
        gone: list[str] = []
        with self._lock:
            snapshot = list(self._nodes.values())
        for node in snapshot:
            if node.status is not NodeStatus.OFFLINE and node.last_heartbeat_ms < cutoff:
                with self._lock:
                    # re-check under lock in case a concurrent heartbeat arrived
                    if node.status is not NodeStatus.OFFLINE and node.last_heartbeat_ms < cutoff:
                        node.status = NodeStatus.OFFLINE
                        gone.append(node.id)
                    else:
                        continue
                self.store.put_node(node)
                log.warning("node timed out -> offline", node=node.id, name=node.name)
                self.events.publish("node_offline", {"id": node.id, "name": node.name})
        return gone
