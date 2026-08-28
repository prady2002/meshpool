"""Shared data contracts (mirror the spec's proto3 messages).

Every message that crosses the node<->control-plane or node<->node boundary is a
Pydantic model here. A gRPC port would translate these directly into proto3.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional, get_origin

from pydantic import BaseModel, Field, model_validator


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def now_ms() -> int:
    return int(time.time() * 1000)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Schedule(str, Enum):
    ALWAYS = "always"
    IDLE_ONLY = "idle_only"
    BUSINESS_HOURS = "business_hours"
    OUTSIDE_BUSINESS_HOURS = "outside_business_hours"


class TrustTier(int, Enum):
    # Kept architecturally separate from day one (spec §7/§13.5). Only tier 0 is
    # active in the MVP; tier 1/2 fields exist but are inert.
    SAME_OWNER = 0        # home: full mutual trust
    TRUSTED_ORG = 1       # office: SSO-gated, reputation scoring
    OPEN_PUBLIC = 2       # public: out of MVP scope, R&D only


class NodeStatus(str, Enum):
    ONLINE = "online"
    DRAINING = "draining"   # reclaim in progress, finish/evict in-flight work
    OFFLINE = "offline"


# --------------------------------------------------------------------------- #
# Node capability / policy / load
# --------------------------------------------------------------------------- #
class Capability(BaseModel):
    cpu_cores: int = 1
    cpu_arch: str = "unknown"
    ram_mb: int = 0
    gpu: bool = False
    vram_mb: int = 0
    disk_mb: int = 0
    flops_estimate: float = 0.0  # rough micro-benchmark score (higher = faster)
    # Real-LLM models this node can already serve locally via a detected Ollama
    # server (name:tag, e.g. "qwen2.5-coder:14b"). Empty when no Ollama is present.
    # Lets the control plane route inference to nodes that already hold a model
    # (spec §9.6 engine adapter: the "ollama" backend) instead of sharding a toy net.
    ollama_models: list[str] = Field(default_factory=list)
    # Execution runtimes this agent can actually serve. The scheduler filters on this,
    # so a node never receives work it will only reject.
    #
    #   "builtin" — the curated Python function registry (square, is_prime, embed, ...)
    #   "wasm"    — user-supplied wasm:<sha256> modules
    #
    # The default is both, which is exactly what the Python agent provides, so every
    # already-registered node keeps its existing behaviour. meshd advertises ["wasm"]
    # only: it deliberately does not reimplement the curated registry, because two
    # independent implementations of one function name is how a pool starts returning
    # different answers for the same input and slashing honest nodes for it.
    runtimes: list[str] = Field(default_factory=lambda: ["builtin", "wasm"])
    # Workload *classes* this agent can serve at all. ``runtimes`` is finer-grained
    # and only meaningful within "batch"; this is the coarse question the hosting
    # and inference planners have to ask before they commit to a placement plan.
    #
    #   "batch"   — run work units          (POST /run_unit)
    #   "hosting" — run app replicas        (POST /start_replica)
    #   "shards"  — hold model layer ranges (POST /shard/load, /shard/forward)
    #   "storage" — hold content-addressed blobs (POST /blob_put, ...)
    #
    # Default is everything, so every already-registered node keeps its behaviour;
    # the field can only ever *narrow* a node that narrowed its own advertisement.
    #
    # This matters more here than it does for batch. Batch retries a rejected unit
    # on another node, but hosting and inference *commit to a plan*: an unservable
    # node in a layer partition means the whole model deploy fails, and it counts
    # its RAM toward pooled capacity, so adding a node can break a deployment that
    # would have succeeded without it.
    services: list[str] = Field(
        default_factory=lambda: ["batch", "hosting", "shards", "storage"])

    # Descriptive hardware detail from the real probe. These do not gate
    # placement on their own; they feed node classification (Anchor / Relay /
    # Constellation / Drifter) and are shown to the operator so the console can
    # explain *why* a machine was or was not chosen. Every one of them defaults
    # to an explicitly unknown value rather than a plausible guess, because a
    # confident wrong number is worse for placement than a known absence.
    cpu_model: str = "unknown"
    gpu_names: list[str] = Field(default_factory=list)
    disk_total_mb: int = 0
    thermal_state: str = "unknown"      # ok | warm | throttling | unknown
    battery_powered: Optional[bool] = None
    battery_percent: Optional[float] = None
    network_class: str = "unknown"      # public | full-cone | restricted | symmetric | cgnat

    @model_validator(mode="before")
    @classmethod
    def _null_list_means_absent(cls, data):
        """Treat an explicit JSON ``null`` for a list field as "field not sent".

        This is not cosmetic — it is a fleet-bricking hazard. Go marshals a nil
        slice as ``null``, not ``[]``, so any agent build (or any third-party
        agent, in any language) that leaves one of these slices unset sends
        ``"services": null``. Pydantic rejects null for ``list[str]``, the
        control plane answers 422, and the node **cannot register at all** — it
        just retries forever with no path to recovery. Adding a list field to
        this model would otherwise be enough to lock out every older agent
        binary mid-rollout. ``ollama_models`` has had this hazard since it was
        introduced; it only escaped notice because the Python agent happens to
        always set it.

        There is no information in the distinction between "absent" and "null"
        here, so dropping the key and letting the field's own default apply is
        strictly safer, and keeps each default defined in exactly one place.
        """
        if isinstance(data, dict):
            nulled = [k for k in cls._null_tolerant_list_fields()
                      if k in data and data[k] is None]
            if nulled:
                data = {k: v for k, v in data.items() if k not in nulled}
        return data

    @classmethod
    def _null_tolerant_list_fields(cls) -> tuple[str, ...]:
        """Every list-typed field on this model, derived rather than hand-listed.

        Deriving matters: a hand-maintained tuple silently stops covering the
        newest field, which is the precise way the null-slice lockout gets
        reintroduced by someone adding a field and not knowing this exists.
        """
        return tuple(
            name for name, field in cls.model_fields.items()
            if get_origin(field.annotation) is list
        )


class LinkStats(BaseModel):
    """Measured (not guessed) link quality from this node to the control plane."""
    latency_ms: float = 0.0
    bandwidth_mbps: float = 0.0


class DonationPolicy(BaseModel):
    """User-set donation cap + schedule. The agent enforces this via the OS /
    runtime; the workload is never trusted to self-limit (spec §4.1)."""
    cpu_pct: int = 50            # donate up to this % of CPU
    ram_mb: int = 1024           # donate up to this much RAM
    gpu_allowed: bool = False
    schedule: Schedule = Schedule.ALWAYS
    idle_minutes: int = 10       # idle threshold for IDLE_ONLY schedule


class NodeLoad(BaseModel):
    cpu_load_pct: float = 0.0
    ram_used_mb: int = 0
    running_units: int = 0
    idle_seconds: float = 0.0    # seconds since last local user activity


# --------------------------------------------------------------------------- #
# Node registry record
# --------------------------------------------------------------------------- #
class Node(BaseModel):
    id: str
    name: str
    url: str                     # the agent's own HTTP base URL (peers/CP call it)
    owner: str = "default"
    device_id: Optional[str] = None  # stable cryptographic identity (global mode)
    capability: Capability = Field(default_factory=Capability)
    link: LinkStats = Field(default_factory=LinkStats)
    policy: DonationPolicy = Field(default_factory=DonationPolicy)
    load: NodeLoad = Field(default_factory=NodeLoad)
    trust_tier: TrustTier = TrustTier.SAME_OWNER
    trust_score: float = 1.0
    status: NodeStatus = NodeStatus.ONLINE
    last_heartbeat_ms: int = Field(default_factory=now_ms)
    registered_ms: int = Field(default_factory=now_ms)
    # Inferred or explicitly declared failure domain (see controlplane/topology.py).
    # Nodes sharing a domain are assumed to fail together (correlated).
    # None means "not yet determined"; placement treats it as its own unique domain.
    failure_domain: Optional[str] = None

    # Effective donatable headroom right now (cap minus current use), respecting
    # the active schedule (a node "off schedule" advertises zero headroom).
    def headroom_cpu_pct(self, schedulable: bool = True) -> float:
        if not schedulable or self.status is not NodeStatus.ONLINE:
            return 0.0
        return max(0.0, float(self.policy.cpu_pct) - self.load.cpu_load_pct)

    def headroom_ram_mb(self, schedulable: bool = True) -> int:
        if not schedulable or self.status is not NodeStatus.ONLINE:
            return 0
        return max(0, self.policy.ram_mb - self.load.ram_used_mb)

    def provides(self, service: str) -> bool:
        """True when this node advertises that it can serve ``service`` at all.

        A node registered before ``Capability.services`` existed deserializes with
        the permissive default, and an empty list is read as "did not narrow", so
        this can only ever exclude a node that explicitly narrowed itself.
        """
        advertised = getattr(self.capability, "services", None)
        return not advertised or service in advertised

    def donatable_cpu_slots(self) -> int:
        """How many concurrent CPU-bound work units this node will run = donatable
        cores = floor(cpu_pct/100 * cores). This is the unit the batch scheduler
        packs against (and exactly the runtime's worker-pool size), so scheduling
        can never over-subscribe a node's pool."""
        import math
        return max(1, int(math.floor((self.policy.cpu_pct / 100.0) * self.capability.cpu_cores)))


# --------------------------------------------------------------------------- #
# Registration / heartbeat
# --------------------------------------------------------------------------- #
class RegisterRequest(BaseModel):
    name: str
    url: str
    owner: str = "default"
    capability: Capability
    link: LinkStats = Field(default_factory=LinkStats)
    policy: DonationPolicy = Field(default_factory=DonationPolicy)
    # Enterprise enrollment is gated by this token (Tier 1+). Ignored at Tier 0.
    enroll_token: Optional[str] = None
    # Auto device identity (zero-signup global onboarding). When present the
    # coordinator verifies `signature` over enrollment_challenge(device_id, ts_ms)
    # to prove the caller holds the private key, and keys the node record on the
    # stable device_id (survives IP/network changes) instead of the URL. Absent on
    # plain LAN registrations, which continue to be keyed by URL.
    device_id: Optional[str] = None
    public_key: Optional[str] = None
    signature: Optional[str] = None
    ts_ms: Optional[int] = None
    # Explicit failure-domain label for simulation / operator override.
    # When set, overrides the auto-inferred domain (see controlplane/topology.py).
    # Absent from older agents and Go agents -- both work unchanged because None
    # triggers auto-inference in the registry.  NOT a list field; no null-list hazard.
    failure_domain: Optional[str] = None


class RegisterResponse(BaseModel):
    node_id: str
    control_plane_time_ms: int
    heartbeat_interval_s: int = 3
    # Server-minted per-node session token. The agent presents it (X-MeshPool-Node-Token)
    # so the control plane can verify node identity server-side on sensitive calls
    # (e.g. secret redemption) instead of trusting a caller-supplied node_id.
    node_token: str = ""


class Heartbeat(BaseModel):
    node_id: str
    load: NodeLoad
    policy: DonationPolicy
    status: NodeStatus = NodeStatus.ONLINE
    # Vivaldi: observed RTTs (ms) to peers measured since the last heartbeat.
    # Maps peer node_id -> rtt_ms.  Optional so that existing agents (Python and Go)
    # that omit it continue to work unchanged; ``null`` from Go nil-map is handled by
    # the Optional type (it becomes None, not a validation error).
    observed_rtts: Optional[dict[str, float]] = None


# --------------------------------------------------------------------------- #
# Batch engine
# --------------------------------------------------------------------------- #
class UnitStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class WorkUnit(BaseModel):
    id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    status: UnitStatus = UnitStatus.PENDING
    assigned_node: Optional[str] = None
    attempts: int = 0
    result: Optional[Any] = None
    error: Optional[str] = None


class BatchJob(BaseModel):
    id: str
    name: str
    func: str                    # named worker function (e.g. "square", "embed")
    units: list[WorkUnit] = Field(default_factory=list)
    cpu_pct_per_unit: int = 15
    ram_mb_per_unit: int = 128
    owner: str = "default"
    capability_token_id: Optional[str] = None
    secret_names: list[str] = Field(default_factory=list)
    status: str = "pending"
    created_ms: int = Field(default_factory=now_ms)

    def progress(self) -> dict[str, int]:
        counts = {s.value: 0 for s in UnitStatus}
        for u in self.units:
            counts[u.status.value] += 1
        counts["total"] = len(self.units)
        return counts


# --------------------------------------------------------------------------- #
# Hosting engine
# --------------------------------------------------------------------------- #
class Replica(BaseModel):
    id: str
    node_id: str
    node_name: str = ""
    url: str                     # where the running app replica is reachable
    healthy: bool = False
    started_ms: int = Field(default_factory=now_ms)


class Deployment(BaseModel):
    id: str
    name: str
    app: str                     # named built-in app or command spec
    site: Optional[dict[str, str]] = None   # {relpath: base64} for a hosted static site (inline)
    # For a LARGE app the bundle is staged in the pooled blob store instead of inline;
    # this is its content id, hydrated back on each replica start (initial + self-heal).
    site_blob_id: Optional[str] = None
    start_cmd: Optional[str] = None   # launch command for a dynamic process app (listens on $PORT)
    env: dict[str, str] = Field(default_factory=dict)  # user config for a dynamic app (12-factor)
    replicas_desired: int = 2
    replicas: list[Replica] = Field(default_factory=list)
    owner: str = "default"
    status: str = "pending"
    created_ms: int = Field(default_factory=now_ms)


class BlobMeta(BaseModel):
    """Metadata for one object in the pooled, content-addressed store.

    ``id`` is the SHA-256 hex digest of the blob's bytes, so identical uploads dedupe
    and any node's returned copy can be integrity-checked against it. ``replicas``
    lists the node ids currently known to hold a full copy; the storage manager keeps
    that at ``replication_desired`` (re-replicating after a node is lost) so an object
    survives machine failures no single box could withstand. The bytes themselves live
    on the agents' donated disk — only this metadata is persisted centrally.
    """
    id: str                      # sha256 hex digest of the content
    owner: str = "default"
    size: int = 0                # bytes
    name: str = ""               # optional human label / download filename
    replicas: list[str] = Field(default_factory=list)   # node ids holding a copy
    replication_desired: int = 2
    status: str = "stored"       # stored | degraded | lost
    # A large object that exceeds the per-chunk size is stored as an ordered list of
    # smaller content-addressed chunk blobs (each independently replicated across the
    # pool), so an object bigger than any single machine's free disk still fits. For
    # such a manifest ``chunks`` holds the ordered chunk ids and ``replicas`` is empty
    # (the bytes live in the chunks); a plain object has ``chunks == []``.
    chunks: list[str] = Field(default_factory=list)
    internal: bool = False       # True for a chunk blob (hidden from user listings)
    created_ms: int = Field(default_factory=now_ms)


# --------------------------------------------------------------------------- #
# Inference engine
# --------------------------------------------------------------------------- #
class LayerAssignment(BaseModel):
    node_id: str
    url: str
    start_layer: int             # inclusive
    end_layer: int               # exclusive
    is_first: bool = False       # owns embedding + input
    is_last: bool = False        # owns final norm + LM head


class ModelDeployment(BaseModel):
    id: str
    model: str
    # Engine adapter serving this deployment: "meshshard" (NumPy pipeline-sharded,
    # the default + correctness oracle) or a native-generate backend such as
    # "ollama" (each partition entry is a full-model replica, not a layer block).
    backend: str = "meshshard"
    n_layers: int
    partition: list[LayerAssignment] = Field(default_factory=list)
    owner: str = "default"
    # When set, this model's weights are staged in the pooled blob store — split into
    # per-layer group blobs referenced by a small manifest (this id) — rather than
    # pre-installed on every node; each stage node pulls only the group blobs its layers
    # need, so no node, nor the control plane, ever holds the whole checkpoint.
    weights_blob_id: Optional[str] = None
    # Number of per-layer weight-group blobs the pooled checkpoint was split into
    # (n_layer + the shared embedding/positional/final-norm groups); None if built-in.
    weight_groups: Optional[int] = None
    status: str = "pending"
    tokens_per_sec: Optional[float] = None
    # Warm standby nodes per stage: primary node id -> node ids holding the SAME
    # layer range, loaded and resident but idle. In a pipeline one dead stage stalls
    # everything behind it, and re-partitioning the whole deployment takes seconds
    # that an in-flight generation does not have. A spare turns that into a rotation
    # plus an O(T) single-stage replay. Empty when the pool has no capacity to spare,
    # which is the honest common case on a small mesh.
    spares: dict[str, list[str]] = Field(default_factory=dict)
    # Cumulative failover history, folded in when a batcher is retired so it is not
    # lost the moment the control plane re-partitions. A user asking "did my
    # deployment survive anything today?" is asking about the deployment's whole
    # life, not about whichever pipeline object happens to be resident right now.
    failovers_total: int = 0
    hydrations_total: int = 0
    retired_nodes: list[str] = Field(default_factory=list)
    created_ms: int = Field(default_factory=now_ms)


class InferenceRequest(BaseModel):
    deployment_id: str
    prompt: str
    max_new_tokens: int = 32


# --------------------------------------------------------------------------- #
# Trust + contribution ledger (Tier 0 active; Tier 1/2 fields inert)
# --------------------------------------------------------------------------- #
class LedgerEntry(BaseModel):
    id: str
    node_id: str
    owner: str
    kind: str                    # "donated" | "consumed"
    compute_units: float         # cpu_pct-share * seconds
    ts_ms: int = Field(default_factory=now_ms)
    detail: str = ""


class TrustScore(BaseModel):
    """Reputation signals for a node.

    The legacy EMA fields are retained for compatibility, but Tier 2 decisions use
    the decaying Beta posterior below.  ``beta_alpha`` and ``beta_beta`` are
    weighted by verified work value, not just job count, so a node cannot cheaply
    farm trust on tiny jobs and spend it on expensive ones.
    """
    node_id: str
    benchmark_pass_rate_ema: float = 1.0
    task_success_rate_ema: float = 1.0
    uptime_ratio_ema: float = 1.0
    latency_consistency_ema: float = 1.0
    # Number of recorded reputation samples. At Tier 2 a stranger with too few
    # samples is on "probation" (its results are verified) until it proves itself,
    # so a fresh identity cannot be trusted on day one just because EMAs default 1.0.
    samples: int = 0
    # Tier-2 oracle spot-check tally. verify_failures > 0 means this node has been
    # caught returning a wrong answer at least once, so it is re-verified on every
    # unit thereafter — a proven cheat is never again trusted blindly.
    verify_passes: int = 0
    verify_failures: int = 0
    beta_alpha: float = 1.0
    beta_beta: float = 1.0
    beta_last_ms: int = 0
    proven_value: float = 0.0

    def score(self) -> float:
        beta_total = self.beta_alpha + self.beta_beta
        if beta_total > 2.0:
            return self.beta_alpha / beta_total
        return (
            0.35 * self.benchmark_pass_rate_ema
            + 0.25 * self.task_success_rate_ema
            + 0.25 * self.uptime_ratio_ema
            + 0.15 * self.latency_consistency_ema
        )


# --------------------------------------------------------------------------- #
# Batch DAG / workflow engine
# --------------------------------------------------------------------------- #
class DagStageSpec(BaseModel):
    id: str
    func: str
    payloads: list[dict[str, Any]] = Field(default_factory=lambda: [{}])
    depends_on: list[str] = Field(default_factory=list)
    reduce: bool = False
    reduce_input_key: str = "items"
    cpu_pct: int = 15
    ram_mb: int = 128
    secret_names: list[str] = Field(default_factory=list)


class DagStageState(DagStageSpec):
    status: str = "pending"
    job_id: Optional[str] = None
    results: list[Any] = Field(default_factory=list)
    error: Optional[str] = None
    execution_count: int = 0
    started_ms: Optional[int] = None
    completed_ms: Optional[int] = None


class DagWorkflow(BaseModel):
    id: str
    name: str = "dag"
    stages: list[DagStageState] = Field(default_factory=list)
    owner: str = "default"
    status: str = "pending"
    created_ms: int = Field(default_factory=now_ms)
    updated_ms: int = Field(default_factory=now_ms)

    def progress(self) -> dict[str, int]:
        counts: dict[str, int] = {
            "pending": 0, "running": 0, "succeeded": 0, "failed": 0,
            "total": len(self.stages),
        }
        for stage in self.stages:
            counts[stage.status] = counts.get(stage.status, 0) + 1
        return counts
