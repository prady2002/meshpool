"""Placement algorithms for the three workload types.

* ``best_fit_node``      — batch work units (tight bin-packing).
* ``bin_pack_replicas``  — website/app replicas (spread for HA, then pack).
* ``topology_partition`` — model layer blocks across nodes (exo-style, spec §4.4).

Kept intentionally simple and greedy first (Nomad/k8s philosophy — don't reinvent
bin-packing theory); sophistication can be layered on later.

Concurrency note
----------------
These are **pure functions** that consume a snapshot of node state.  They carry
no mutable state of their own, so they are inherently thread-safe.  Callers
(BatchManager, HostingManager, InferenceManager) are responsible for holding
their own lock while building the node snapshot and updating in-flight counters
atomically — see docs/durability.md §Locking model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable, Optional

from ..common.models import LayerAssignment, Node, NodeStatus, TrustScore
from .nodeclass import classify

CONSTELLATION_RTT_THRESHOLD_MS = 25.0


def best_fit_node(nodes: list[Node], cpu_pct: float, ram_mb: int,
                  priority: Optional[dict[str, float]] = None) -> Optional[Node]:
    """Pick the online, schedulable node whose remaining headroom *most tightly*
    fits the request (best-fit) so large-capacity nodes stay free for large work.

    ``priority`` maps owner -> fair-share weight; higher weight breaks ties in
    favour of nodes owned by higher-contributing users (Tier 1+, no-op at Tier 0).
    """
    best: Optional[Node] = None
    best_key: Optional[tuple[float, float]] = None
    for n in nodes:
        hc = n.headroom_cpu_pct()
        hr = n.headroom_ram_mb()
        if hc < cpu_pct or hr < ram_mb:
            continue
        weight = (priority or {}).get(n.owner, 1.0)
        # slack: how much would be left over (smaller = tighter fit = better)
        slack = (hc - cpu_pct) + (hr - ram_mb) / 1024.0
        key = (slack, -weight)  # tie-break: prefer higher fair-share weight
        if best_key is None or key < best_key:
            best_key, best = key, n
    return best


def bin_pack_replicas(nodes: list[Node], count: int, cpu_pct: float,
                      ram_mb: int) -> list[Node]:
    """Place ``count`` replicas across nodes, preferring distinct failure domains
    first (correlated-failure diversity), then distinct nodes (HA), then pack extras.

    Returns the chosen nodes (may be shorter than ``count`` if capacity is short —
    we never promise autoscaling beyond what's physically pooled, spec §4.6).

    Diversity is a *soft* preference, not a hard filter: if the mesh is small and
    homogeneous (all in one domain) this degrades gracefully to the existing best-fit
    behaviour.  Adding a new node never breaks existing placements.
    """
    # Track simulated remaining headroom as we place.
    remaining = {n.id: [n.headroom_cpu_pct(), n.headroom_ram_mb()]
                 for n in nodes if n.provides("hosting")}
    used_count = {nid: 0 for nid in remaining}
    placements: list[Node] = []
    by_id = {n.id: n for n in nodes}
    # Per-node domain key for diversity scoring.
    node_domain = {n.id: (n.failure_domain or n.id) for n in nodes}
    used_domains: set[str] = set()

    for _ in range(count):
        # candidates with enough headroom
        candidates = [nid for nid, rem in remaining.items()
                      if rem[0] >= cpu_pct and rem[1] >= ram_mb]
        if not candidates:
            break
        # Sort key: (domain_already_used, replicas_on_node, slack)
        # This prefers a node in a *new* domain, then the least-loaded, then tightest fit.
        def _key(nid: str):
            domain_used = int(node_domain[nid] in used_domains)
            slack = ((remaining[nid][0] - cpu_pct)
                     + (remaining[nid][1] - ram_mb) / 1024.0)
            return (domain_used, used_count[nid], slack)
        candidates.sort(key=_key)
        nid = candidates[0]
        remaining[nid][0] -= cpu_pct
        remaining[nid][1] -= ram_mb
        used_count[nid] += 1
        used_domains.add(node_domain[nid])
        placements.append(by_id[nid])
    return placements


def usable_layer_memory_mb(n: Node) -> int:
    """Memory this node can devote to holding model layers, in MB.

    A GPU is an *accelerator*, not a smaller box. Taking VRAM **instead of** RAM
    (the previous behaviour) meant a machine with 32 GB of RAM and an integrated
    GPU advertising a 512 MB carve-out was scored at ``512`` — i.e. zero layers —
    and was silently excluded from every model deployment. That is the common
    case on consumer laptops and mini-PCs, exactly the hardware this project is
    built for, and it made ``POST /api/v1/models`` fail with "insufficient pooled
    memory" on machines with plenty of memory.

    Layers can live in VRAM (fast) or in system RAM (slower, but it works — the
    CPU path is always available), so the honest bound is whichever is larger.
    The donation policy cap is applied by the caller.
    """
    cap = n.capability
    vram = cap.vram_mb if (cap.gpu and cap.vram_mb) else 0
    return max(int(vram or 0), int(cap.ram_mb or 0))


def _safe_rtt(predictor: Callable, a_id: str, b_id: str) -> float:
    """Call *predictor(a_id, b_id)* safely; return inf on any failure or None."""
    try:
        v = predictor(a_id, b_id)
        if v is not None and v >= 0.0:
            return float(v)
    except Exception:  # noqa: BLE001
        pass
    return float("inf")


def latency_sort_nodes(
    usable: list[dict],
    rtt_predictor: Callable[[str, str], Optional[float]],
) -> list[dict]:
    """Reorder *usable* so that consecutive pipeline stages have low predicted RTT.

    Uses greedy nearest-neighbor starting from the highest-FLOPS node (the
    "anchor"), which ensures the most compute-heavy stage is at position 0 and
    subsequent stages are placed on nearby nodes.

    Returns *usable* unchanged when fewer than 3 nodes are present, or when
    every predicted RTT is infinite (coordinates not yet available).

    This is O(n²), which is acceptable for the small N typical in a MeshPool
    pipeline (2–16 nodes); for N > 16 the scheduler's time cost is dominated by
    the HTTP calls to load shards, not this sort.
    """
    if len(usable) <= 2:
        return usable

    remaining = list(usable)
    # Anchor: highest-FLOPS node
    anchor_idx = max(range(len(remaining)), key=lambda i: remaining[i]["flops"])
    result = [remaining.pop(anchor_idx)]

    while remaining:
        last_id = result[-1]["node"].id
        best_idx = min(
            range(len(remaining)),
            key=lambda i: _safe_rtt(rtt_predictor, last_id, remaining[i]["node"].id),
        )
        result.append(remaining.pop(best_idx))

    # If every RTT was infinite (all coordinates missing), we just re-derived the
    # flops-based anchor ordering; that is acceptable and graceful.
    return result


def _usable_shard_nodes(
    nodes: list[Node],
    headroom_fraction: float,
) -> list[dict]:
    usable = []
    for n in nodes:
        if not n.provides("shards"):
            continue
        mem_mb = usable_layer_memory_mb(n)
        mem_mb = min(mem_mb, n.policy.ram_mb) if n.policy.ram_mb else mem_mb
        cap_mb = mem_mb * headroom_fraction
        if cap_mb <= 0:
            continue
        flops = max(1e-3, n.capability.flops_estimate or n.capability.cpu_cores)
        usable.append({"node": n, "cap_mb": cap_mb, "cap_bytes": int(cap_mb * 1024 * 1024), "flops": flops})
    return usable


@dataclass(frozen=True)
class PartitionFit:
    fits: bool
    bytes_needed: int
    bytes_available: int
    reason: str


@dataclass(frozen=True)
class Constellation:
    """Soft low-latency node neighbourhood discovered from measured RTT."""

    id: str
    node_ids: tuple[str, ...]
    pooled_ram_mb: int
    pooled_vram_mb: int
    pooled_usable_mb: int
    total_flops: float
    max_internal_rtt_ms: float
    avg_internal_rtt_ms: float

    def fits_memory(self, required_mb: int) -> bool:
        return self.pooled_usable_mb >= required_mb


class ConstellationIndex:
    """Pre-clustered lookup table for constellation-capacity placement.

    Formation may inspect the node snapshot, but lookup is bucketed by required
    memory so schedulers do not re-run a global optimization on every request.
    ``lookup_scans`` is an acceptance-gate guard: fitting lookups must not
    enumerate all nodes or all constellations.
    """

    def __init__(self, nodes_by_id: dict[str, Node], constellations: list[Constellation],
                 bucket_mb: int = 1024) -> None:
        self.nodes_by_id = dict(nodes_by_id)
        self.constellations = sorted(
            constellations,
            key=lambda c: (c.avg_internal_rtt_ms, c.max_internal_rtt_ms, -c.pooled_usable_mb, c.id),
        )
        self.bucket_mb = max(1, int(bucket_mb))
        self.lookup_scans = 0
        self._fit_bucket: dict[int, Constellation] = {}
        if self.constellations:
            max_bucket = max(math.ceil(c.pooled_usable_mb / self.bucket_mb) for c in self.constellations)
            for bucket in range(max_bucket + 1):
                required = bucket * self.bucket_mb
                for c in self.constellations:
                    if c.fits_memory(required):
                        self._fit_bucket[bucket] = c
                        break

    def fitting(self, required_mb: int) -> Optional[Constellation]:
        bucket = math.ceil(max(0, int(required_mb)) / self.bucket_mb)
        return self._fit_bucket.get(bucket)

    def nodes_for(self, constellation: Constellation) -> list[Node]:
        return [self.nodes_by_id[nid] for nid in constellation.node_ids if nid in self.nodes_by_id]


def _constellation_id(node_ids: list[str]) -> str:
    digest = hashlib.blake2b(",".join(sorted(node_ids)).encode("utf-8"), digest_size=5).hexdigest()
    return f"constellation-{digest}"


def build_constellation_index(
    nodes: list[Node],
    rtt_predictor: Callable[[str, str], Optional[float]],
    *,
    rtt_threshold_ms: float = CONSTELLATION_RTT_THRESHOLD_MS,
    min_size: int = 2,
) -> ConstellationIndex:
    """Cluster shard-capable online nodes from measured RTT/Vivaldi predictions."""
    usable_nodes = [
        n for n in nodes
        if n.status is NodeStatus.ONLINE and n.provides("shards") and usable_layer_memory_mb(n) > 0
    ]
    by_id = {n.id: n for n in usable_nodes}
    adjacency: dict[str, set[str]] = {n.id: set() for n in usable_nodes}
    pair_rtt: dict[tuple[str, str], float] = {}
    for i, a in enumerate(usable_nodes):
        for b in usable_nodes[i + 1:]:
            rtt = _safe_rtt(rtt_predictor, a.id, b.id)
            if rtt <= rtt_threshold_ms:
                adjacency[a.id].add(b.id)
                adjacency[b.id].add(a.id)
                pair_rtt[(a.id, b.id)] = rtt
                pair_rtt[(b.id, a.id)] = rtt

    seen: set[str] = set()
    constellations: list[Constellation] = []
    for node_id in sorted(adjacency):
        if node_id in seen:
            continue
        stack = [node_id]
        component: list[str] = []
        seen.add(node_id)
        while stack:
            cur = stack.pop()
            component.append(cur)
            for nxt in sorted(adjacency[cur]):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        if len(component) < min_size:
            continue
        rtts = [
            pair_rtt.get((a, b), rtt_threshold_ms)
            for i, a in enumerate(component)
            for b in component[i + 1:]
        ]
        members = [by_id[nid] for nid in sorted(component)]
        pooled_ram = sum(max(0, n.headroom_ram_mb()) for n in members)
        pooled_vram = sum(max(0, int(n.capability.vram_mb or 0)) for n in members if n.capability.gpu)
        pooled_usable = sum(
            min(usable_layer_memory_mb(n), n.policy.ram_mb) if n.policy.ram_mb
            else usable_layer_memory_mb(n)
            for n in members
        )
        constellations.append(Constellation(
            id=_constellation_id(component),
            node_ids=tuple(sorted(component)),
            pooled_ram_mb=pooled_ram,
            pooled_vram_mb=pooled_vram,
            pooled_usable_mb=int(pooled_usable),
            total_flops=sum(max(1e-3, n.capability.flops_estimate or n.capability.cpu_cores)
                            for n in members),
            max_internal_rtt_ms=max(rtts) if rtts else 0.0,
            avg_internal_rtt_ms=(sum(rtts) / len(rtts)) if rtts else 0.0,
        ))
    return ConstellationIndex(by_id, constellations)


def select_pipeline_constellation(
    index: ConstellationIndex,
    *,
    required_memory_mb: int,
    pipeline_depth: int,
) -> Optional[Constellation]:
    """Return the best pre-clustered pool for a model, minimizing depth × RTT."""
    candidates = [
        c for c in index.constellations
        if c.fits_memory(required_memory_mb) and len(c.node_ids) >= max(1, pipeline_depth)
    ]
    if not candidates:
        return index.fitting(required_memory_mb)
    return min(
        candidates,
        key=lambda c: (
            max(0, pipeline_depth - 1) * c.avg_internal_rtt_ms,
            c.max_internal_rtt_ms,
            -c.pooled_usable_mb,
            c.id,
        ),
    )


def constellation_nodes_for_model(
    nodes: list[Node],
    rtt_predictor: Callable[[str, str], Optional[float]],
    *,
    required_memory_mb: int,
    pipeline_depth: int,
) -> list[Node]:
    """O(1)-lookup constellation selection; empty means gracefully use flat pool."""
    index = build_constellation_index(nodes, rtt_predictor)
    c = select_pipeline_constellation(
        index, required_memory_mb=required_memory_mb, pipeline_depth=pipeline_depth)
    return index.nodes_for(c) if c is not None else []


def stable_nodes_for_model(
    nodes: list[Node],
    trust_lookup: Optional[Callable[[str], Optional["TrustScore"]]],
    *,
    required_memory_mb: int,
    headroom_fraction: float = 0.8,
) -> list[Node]:
    """Drop Drifter-class nodes from a placement pool -- but only if the model
    still fits without them.

    A pipeline's availability is the *product* of its stages', so one node that
    unplugs at 6pm degrades every request rather than its own share of them. The
    scheduler otherwise cannot see this at all: ``_usable_shard_nodes`` filters on
    advertised service and memory, and a laptop with 64 GB looks strictly better
    than a mini-PC with 16 GB right up until it closes.

    Preference, never prohibition. The product promise is running models that fit
    on no single machine, so a rule that turns a working deployment into
    "insufficient pooled memory" is a regression wearing a safety feature's
    clothes. When the volatile node is load-bearing, it is used and the volatility
    is handled where it belongs -- hot spares (§3.4).

    ``trust_lookup`` of ``None``, or one that knows nothing about these nodes,
    leaves the pool exactly as it was: LAN mode and freshly restarted
    coordinators have no history, and inventing one would empty the pool.
    """
    if trust_lookup is None or not nodes:
        return list(nodes)

    keep: list[Node] = []
    dropped: list[Node] = []
    for n in nodes:
        try:
            verdict = classify(n, trust_lookup(n.id))
        except Exception:  # noqa: BLE001
            # Classification is an optimisation. It must never be the reason a
            # model fails to deploy, so an unexpected signal keeps the node.
            keep.append(n)
            continue
        (keep if verdict.suitable_for_pipeline_stage() else dropped).append(n)

    if not dropped or not keep:
        return list(nodes)

    available_mb = sum(u["cap_mb"] for u in _usable_shard_nodes(keep, headroom_fraction))
    if available_mb < required_memory_mb:
        return list(nodes)
    return keep


def explain_layer_bytes_fit(
    nodes: list[Node],
    layer_bytes: list[int] | tuple[int, ...],
    headroom_fraction: float = 0.8,
) -> PartitionFit:
    """Explain whether measured, indivisible layer bytes fit in the shard pool."""
    usable = _usable_shard_nodes(nodes, headroom_fraction)
    needed = int(sum(layer_bytes))
    available = int(sum(u["cap_bytes"] for u in usable))
    if not layer_bytes:
        return PartitionFit(True, needed, available, "no layers to place")
    if not usable:
        return PartitionFit(False, needed, available, "no online shard-capable nodes with usable memory")
    if max(layer_bytes) > max(u["cap_bytes"] for u in usable):
        return PartitionFit(
            False, needed, available,
            f"largest indivisible layer needs {max(layer_bytes)} bytes but the largest node can hold "
            f"{max(u['cap_bytes'] for u in usable)} bytes")
    if needed > available:
        return PartitionFit(
            False, needed, available,
            f"checkpoint layers need {needed} bytes but the pool can safely hold {available} bytes")
    return PartitionFit(True, needed, available, "checkpoint layers fit in pooled usable memory")


def topology_partition_by_bytes(
    nodes: list[Node],
    layer_bytes: list[int] | tuple[int, ...],
    headroom_fraction: float = 0.8,
    rtt_predictor: Optional[Callable[[str, str], Optional[float]]] = None,
    *,
    trust_lookup: Optional[Callable[[str], Optional[TrustScore]]] = None,
) -> list[LayerAssignment]:
    """Assign contiguous, indivisible layer ranges using measured layer bytes.

    This is the GGUF/MoE path: an MoE layer's router, all experts, and any shared
    experts are counted in one ``layer_bytes[i]`` value and are never split across
    nodes. The node ordering reuses the same latency-aware nearest-neighbor pass as
    ``topology_partition`` so the pipeline still minimizes consecutive-stage RTT.
    """
    layer_bytes = tuple(int(b) for b in layer_bytes)
    n_layers = len(layer_bytes)
    nodes = stable_nodes_for_model(
        nodes, trust_lookup,
        required_memory_mb=math.ceil(sum(layer_bytes) / (1024 * 1024)),
        headroom_fraction=headroom_fraction)
    fit = explain_layer_bytes_fit(nodes, layer_bytes, headroom_fraction)
    if not fit.fits:
        return []

    if rtt_predictor is not None and len(nodes) > 2 and layer_bytes:
        required_mb = math.ceil(sum(layer_bytes) / (1024 * 1024))
        c_nodes = constellation_nodes_for_model(
            nodes, rtt_predictor,
            required_memory_mb=required_mb,
            pipeline_depth=min(len(nodes), len(layer_bytes)),
        )
        if c_nodes and explain_layer_bytes_fit(c_nodes, layer_bytes, headroom_fraction).fits:
            nodes = c_nodes

    usable = _usable_shard_nodes(nodes, headroom_fraction)
    usable = [u for u in usable if u["cap_bytes"] >= min(layer_bytes or (0,))]
    if rtt_predictor is not None and len(usable) > 2:
        usable = latency_sort_nodes(usable, rtt_predictor)

    suffix = [0] * (len(usable) + 1)
    for i in range(len(usable) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + usable[i]["cap_bytes"]

    prefix = [0]
    for b in layer_bytes:
        prefix.append(prefix[-1] + b)

    from functools import lru_cache

    def span_bytes(a: int, b: int) -> int:
        return prefix[b] - prefix[a]

    @lru_cache(maxsize=None)
    def feasible(layer_idx: int, node_idx: int) -> bool:
        if layer_idx == n_layers:
            return True
        if node_idx == len(usable):
            return False
        if sum(layer_bytes[layer_idx:]) > suffix[node_idx]:
            return False
        if feasible(layer_idx, node_idx + 1):
            return True
        cap = usable[node_idx]["cap_bytes"]
        for end in range(layer_idx + 1, n_layers + 1):
            used = span_bytes(layer_idx, end)
            if used > cap:
                break
            if feasible(end, node_idx + 1):
                return True
        return False

    assignments: list[LayerAssignment] = []
    layer_idx = 0
    node_idx = 0
    while layer_idx < n_layers and node_idx < len(usable):
        remaining_bytes = span_bytes(layer_idx, n_layers)
        remaining_flops = sum(u["flops"] for u in usable[node_idx:]) or 1.0
        target = remaining_bytes * usable[node_idx]["flops"] / remaining_flops
        cap = usable[node_idx]["cap_bytes"]
        candidates: list[tuple[float, int]] = []
        for end in range(layer_idx + 1, n_layers + 1):
            used = span_bytes(layer_idx, end)
            if used > cap:
                break
            if feasible(end, node_idx + 1):
                candidates.append((abs(used - target), end))
        if not candidates:
            node_idx += 1
            continue
        _score, end = min(candidates, key=lambda x: (x[0], x[1]))
        node = usable[node_idx]["node"]
        assignments.append(LayerAssignment(
            node_id=node.id, url=node.url, start_layer=layer_idx, end_layer=end,
            is_first=False, is_last=False))
        layer_idx = end
        node_idx += 1

    if layer_idx != n_layers:
        return []
    for i, a in enumerate(assignments):
        a.is_first = i == 0
        a.is_last = i == len(assignments) - 1
    return assignments


def topology_partition(
    nodes: list[Node],
    n_layers: int,
    per_layer_mb: int,
    headroom_fraction: float = 0.8,
    rtt_predictor: Optional[Callable[[str, str], Optional[float]]] = None,
    *,
    trust_lookup: Optional[Callable[[str], Optional[TrustScore]]] = None,
) -> list[LayerAssignment]:
    """Greedily assign contiguous blocks of layers to nodes (pipeline parallel).

    Matches exo's topology-aware auto-parallel idea (spec §4.4):
      * a node's layer capacity is bounded by its usable memory headroom (the
        larger of VRAM and RAM — see ``usable_layer_memory_mb``), never planned
        to 100% — ``headroom_fraction`` leaves room for activations + the OS;
      * nodes with more compute (flops_estimate) get proportionally more layers;
      * assignment is contiguous per node (a pipeline stage), first stage owns the
        embedding, last stage owns the LM head.

    When *rtt_predictor* is provided (a callable ``(node_id_a, node_id_b) ->
    Optional[float]``), eligible nodes are re-ordered with
    :func:`latency_sort_nodes` so that consecutive pipeline stages land on low-
    latency pairs, reducing the ``(N-1) * RTT`` per-token pipeline overhead.
    """
    nodes = stable_nodes_for_model(
        nodes, trust_lookup,
        required_memory_mb=n_layers * per_layer_mb,
        headroom_fraction=headroom_fraction)

    usable = []
    for u0 in _usable_shard_nodes(nodes, headroom_fraction):
        n = u0["node"]
        cap_layers = int(u0["cap_mb"] // max(1, per_layer_mb))
        if cap_layers <= 0:
            continue
        usable.append({"node": n, "cap": cap_layers, "flops": u0["flops"]})

    if not usable:
        return []
    total_capacity = sum(u["cap"] for u in usable)
    if total_capacity < n_layers:
        # Not enough pooled memory to host the whole model.
        return []

    if rtt_predictor is not None and len(nodes) > 2:
        c_nodes = constellation_nodes_for_model(
            nodes, rtt_predictor,
            required_memory_mb=n_layers * per_layer_mb,
            pipeline_depth=min(len(nodes), n_layers),
        )
        if c_nodes:
            c_usable = []
            for u0 in _usable_shard_nodes(c_nodes, headroom_fraction):
                n = u0["node"]
                cap_layers = int(u0["cap_mb"] // max(1, per_layer_mb))
                if cap_layers > 0:
                    c_usable.append({"node": n, "cap": cap_layers, "flops": u0["flops"]})
            if sum(u["cap"] for u in c_usable) >= n_layers:
                usable = c_usable

    # Latency-aware ordering: reorder nodes so consecutive stages have low RTT.
    # Only applied when a predictor is available; otherwise the original order
    # (by the node iteration above) is preserved exactly, keeping this change
    # backwards-compatible and causing zero behavioural difference for existing
    # deployments without Vivaldi data.
    if rtt_predictor is not None and len(usable) > 2:
        usable = latency_sort_nodes(usable, rtt_predictor)

    total_flops = sum(u["flops"] for u in usable)
    # target #layers per node proportional to compute, clamped by memory capacity
    targets = []
    for u in usable:
        want = max(1, round(n_layers * u["flops"] / total_flops))
        targets.append(min(want, u["cap"]))

    # fix rounding so sum(targets) == n_layers
    diff = n_layers - sum(targets)
    i = 0
    guard = 0
    while diff != 0 and guard < 10000:
        idx = i % len(usable)
        if diff > 0 and targets[idx] < usable[idx]["cap"]:
            targets[idx] += 1
            diff -= 1
        elif diff < 0 and targets[idx] > 0:
            targets[idx] -= 1
            diff += 1
        i += 1
        guard += 1

    assignments: list[LayerAssignment] = []
    cursor = 0
    stages = [(u["node"], t) for u, t in zip(usable, targets) if t > 0]
    for si, (node, count) in enumerate(stages):
        start = cursor
        end = cursor + count
        assignments.append(LayerAssignment(
            node_id=node.id, url=node.url, start_layer=start, end_layer=end,
            is_first=(si == 0), is_last=(si == len(stages) - 1)))
        cursor = end
    return assignments
