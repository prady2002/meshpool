"""Website / app hosting engine (spec §9.5, §4.6).

Deploy an app at a desired replica count; the manager:
  * **bin-packs replicas across distinct nodes** (anti-affinity) so a single node
    failing never takes the app fully down;
  * drives each node agent to run the replica as a sandboxed child process
    (the ``Runtime`` abstraction — a container in production);
  * **health-checks** every replica continuously and **self-heals**: a replica on a
    node that sleeps/drops is removed and re-placed elsewhere to restore the
    desired count (failover);
  * exposes a **round-robin load balancer** (see the ``/app/{id}/`` proxy route in
    the control-plane API) that only ever routes to healthy replicas — so killing a
    machine mid-traffic reroutes live requests instead of dropping them.

This mirrors "Caddy/Traefik driven via API" from the spec; here the control plane
*is* the reverse proxy so the whole behaviour is demonstrable with no extra
infrastructure.
"""
from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Optional

from ..common.logging import get_logger
from ..common.models import Deployment, Replica, new_id
from ..common.transport import AgentTransport

log = get_logger("hosting")


class HostingManager:
    def __init__(self, registry, store, ledger, events, transport: AgentTransport) -> None:
        self.registry = registry
        self.store = store
        self.ledger = ledger
        self.events = events
        self.transport = transport
        self._rr: dict[str, int] = {}   # deployment_id -> round-robin cursor
        self._lock = asyncio.Lock()     # serialize deploy/remove/reconcile mutations
        # Set post-construction (lifespan) to the pooled StorageManager. A LARGE app's
        # bundle lives in the pool (content-addressed, chunked, replicated) and is
        # hydrated from here on each replica start instead of travelling inline.
        self.storage = None

    # --- queries --------------------------------------------------------- #
    def list(self) -> list[Deployment]:
        return self.store.list_deployments()

    def get(self, dep_id: str) -> Optional[Deployment]:
        return self.store.get_deployment(dep_id)

    def pick_replica(self, dep_id: str, exclude: Optional[set[str]] = None) -> Optional[Replica]:
        """Round-robin over *healthy* replicas — the load-balancer core.

        ``exclude`` lets a retrying request skip replicas it already found dead this
        request, so failover is **deterministic**: with any surviving replica the
        retry lands on it instead of possibly re-picking the same just-killed node.
        """
        dep = self.store.get_deployment(dep_id)
        if not dep:
            return None
        healthy = [r for r in dep.replicas if r.healthy]
        pool = [r for r in healthy if r.id not in exclude] if exclude else healthy
        if not pool:
            pool = healthy          # nothing left to exclude — fall back (sole replica)
        if not pool:
            return None
        i = self._rr.get(dep_id, 0) % len(pool)
        self._rr[dep_id] = i + 1
        return pool[i]

    def sticky_replica(self, dep_id: str, replica_id: str,
                       exclude: Optional[set[str]] = None) -> Optional[Replica]:
        """Session affinity: return the client's previously-assigned replica iff it is
        still healthy (and not already tried this request), so a stateful multi-page
        app keeps hitting the same worker. ``None`` -> caller falls back to round-robin.
        """
        if not replica_id or (exclude and replica_id in exclude):
            return None
        dep = self.store.get_deployment(dep_id)
        if not dep:
            return None
        for r in dep.replicas:
            if r.id == replica_id and r.healthy:
                return r
        return None

    # --- deploy / scale -------------------------------------------------- #
    async def deploy(self, name: str, app: str, replicas: int, owner: str = "default",
                     site: Optional[dict[str, str]] = None,
                     start_cmd: Optional[str] = None,
                     env: Optional[dict[str, str]] = None,
                     site_blob_id: Optional[str] = None) -> Deployment:
        dep = Deployment(id=new_id("dep"), name=name, app=app, site=site,
                         site_blob_id=site_blob_id,
                         start_cmd=start_cmd, env=env or {},
                         replicas_desired=max(1, replicas), owner=owner, status="deploying")
        self.store.put_deployment(dep)
        self.events.publish("deploy", {"id": dep.id, "app": app, "replicas": replicas})
        async with self._lock:
            await self._scale_up(dep)
        dep.status = "running" if dep.replicas else "pending"
        self.store.put_deployment(dep)
        log.info("deployed", dep=dep.id, app=app, placed=len(dep.replicas), desired=replicas)
        return dep

    async def remove(self, dep_id: str) -> bool:
        async with self._lock:
            dep = self.store.get_deployment(dep_id)
            if not dep:
                return False
            for r in dep.replicas:
                await self._stop_replica(r)
            self.store.delete_deployment(dep_id)
            self._rr.pop(dep_id, None)
        self.events.publish("deploy_removed", {"id": dep_id})
        return True

    async def logs(self, dep_id: str, tail: int = 200) -> Optional[list[dict]]:
        """Gather each replica's most recent stdout/stderr from its node so a deployer
        can debug a hosted process app. Returns None if the deployment is unknown; a
        replica whose node is unreachable simply comes back with no lines."""
        dep = self.store.get_deployment(dep_id)
        if not dep:
            return None
        out: list[dict] = []
        for r in dep.replicas:
            lines: list[str] = []
            try:
                body = await self.transport.call(
                    r.node_id, "replica_logs",
                    {"replica_id": r.id, "tail": tail}, timeout=4.0)
                lines = body.get("lines") or []
            except Exception:  # noqa: BLE001 — an unreachable node yields empty logs
                lines = []
            out.append({"replica_id": r.id, "node_name": r.node_name,
                        "healthy": r.healthy, "lines": lines})
        return out

    async def _scale_up(self, dep: Deployment) -> int:
        """Place replicas until desired count is met, spreading across nodes."""
        missing = dep.replicas_desired - len(dep.replicas)
        if missing <= 0:
            return 0
        site = await self._hydrate_site(dep)
        placed = 0
        for node in self._placement(dep, missing):
            try:
                payload = {"deployment_id": dep.id, "replica_id": new_id("rep"),
                           "app": dep.app}
                if site:
                    payload["site"] = site
                if dep.start_cmd:
                    payload["start_cmd"] = dep.start_cmd
                if dep.env:
                    payload["env"] = dep.env
                body = await self.transport.call(node.id, "start_replica", payload, timeout=8.0)
                dep.replicas.append(Replica(
                    id=body["replica_id"], node_id=node.id, node_name=node.name,
                    url=body["url"], healthy=True))
                placed += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("replica start failed", node=node.id, err=str(exc)[:120])
        if placed:
            self.store.put_deployment(dep)
        return placed

    async def _hydrate_site(self, dep: Deployment) -> Optional[dict[str, str]]:
        """Return the site map to ship to a replica. A small app carries its bundle
        inline on the Deployment; a LARGE app's bundle is staged in the pooled blob
        store, so pull + decode it here (works for the initial deploy and self-heal)."""
        if dep.site:
            return dep.site
        if dep.site_blob_id and self.storage is not None:
            raw = await self.storage.get(dep.site_blob_id)
            if raw is None:
                log.error("deploy bundle unavailable in pool",
                          dep=dep.id, blob=dep.site_blob_id[:12])
                return None
            try:
                return json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                log.error("deploy bundle decode failed", dep=dep.id, err=str(exc)[:120])
                return None
        return None

    def _placement(self, dep: Deployment, count: int) -> list:
        """Choose `count` nodes, preferring those NOT already hosting this app
        (anti-affinity for HA), then those with the most spare CPU slots."""
        nodes = [n for n in self.registry.schedulable_nodes() if n.provides("hosting")]
        if not nodes:
            return []
        cur = Counter(r.node_id for r in dep.replicas)
        chosen = []
        for _ in range(count):
            avail = [n for n in nodes if cur[n.id] < max(1, n.donatable_cpu_slots())]
            if not avail:
                break
            avail.sort(key=lambda n: (cur[n.id], -n.donatable_cpu_slots()))
            pick = avail[0]
            chosen.append(pick)
            cur[pick.id] += 1
        return chosen

    async def _stop_replica(self, r: Replica) -> None:
        node = self.registry.get(r.node_id)
        if not node:
            return
        try:
            await self.transport.call(node.id, "stop_replica",
                                      {"replica_id": r.id}, timeout=4.0)
        except Exception:  # noqa: BLE001
            pass

    async def _healthy(self, r: Replica) -> bool:
        # A replica is "up" if its HTTP server answers at all. Built-in/static apps
        # expose /healthz (200); user-supplied process apps may not, so any response
        # (even 404) at /healthz or / proves the process is listening. Only a failed
        # connection (killed node / crashed replica) counts as unhealthy — which is
        # exactly what drives failover.
        for path in ("/healthz", "/"):
            try:
                resp = await self.transport.proxy(r.node_id, "GET", f"{r.url}{path}",
                                                  timeout=2.0)
            except Exception:  # noqa: BLE001
                return False
            if resp.status_code < 500:
                return True
        return False

    # --- failover -------------------------------------------------------- #
    def on_node_offline(self, node_id: str) -> None:
        """A node dropped: prune its replicas so the LB stops routing there and the
        reconcile loop re-places them elsewhere (fast failover)."""
        for dep in self.store.list_deployments():
            before = len(dep.replicas)
            dep.replicas = [r for r in dep.replicas if r.node_id != node_id]
            if len(dep.replicas) != before:
                dep.status = "degraded"
                self.store.put_deployment(dep)
                self.events.publish("replica_lost", {"id": dep.id, "node": node_id})

    async def _reconcile(self) -> None:
        live = {n.id for n in self.registry.online()}
        for dep in self.store.list_deployments():
            changed = False
            # prune replicas on nodes that are no longer online
            kept = [r for r in dep.replicas if r.node_id in live]
            if len(kept) != len(dep.replicas):
                dep.replicas, changed = kept, True
            # health-check survivors
            for r in dep.replicas:
                h = await self._healthy(r)
                if h != r.healthy:
                    r.healthy, changed = h, True
            # self-heal back to desired replica count (failover / autoscale)
            if len([r for r in dep.replicas if r.healthy]) < dep.replicas_desired:
                if await self._scale_up(dep):
                    changed = True
            if changed:
                healthy = sum(1 for r in dep.replicas if r.healthy)
                dep.status = ("running" if healthy >= dep.replicas_desired
                              else "degraded" if healthy else "down")
                self.store.put_deployment(dep)
                self.events.publish("replica_health",
                                    {"id": dep.id, "healthy": healthy,
                                     "desired": dep.replicas_desired})

    async def run_loop(self, interval: float = 2.0) -> None:
        while True:
            try:
                async with self._lock:
                    await self._reconcile()
            except Exception as exc:  # noqa: BLE001
                log.error("hosting reconcile error", err=str(exc)[:160])
            await asyncio.sleep(interval)
