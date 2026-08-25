"""Batch job manager: dispatch work units across nodes, collect results, retry.

Design notes (spec §4.5):
* Units are independent; each is dispatched to a node with spare *declared*
  capacity chosen best-fit.
* A failed or interrupted unit is retried on a *different* node — a failure is
  never assumed to be the job's fault (the node may simply have gone offline).
* The manager tracks in-flight assignments locally so it doesn't over-commit a
  node between (slower) heartbeat load updates.
"""
from __future__ import annotations

import asyncio
import base64
import os
import random
import statistics
from typing import Optional

from ..common.logging import get_logger
from ..common.models import BatchJob, UnitStatus, WorkUnit, new_id, now_ms
from ..common.transport import AgentTransport
from ..common.trust import (VERIFY_SAMPLE_RATE, ContributionLedger, TrustEngine,
                            fair_share_priority)
from .functions import is_wasm_function, run_function, wasm_module_id
from .registry_ref import RegistryRef

log = get_logger("batch")

MAX_ATTEMPTS = 5

# How long a job may wait for a node advertising its runtime to appear before the
# scheduler declares the pool incapable and fails it with a clear reason.
NO_RUNTIME_GRACE_MS = 30_000

# Speculative backup execution ("work stealing") for heterogeneous home fleets.
# Tunable in gates via env, disabled with MESHPOOL_BATCH_SPECULATION=0.
SPECULATION_MULTIPLIER = float(os.environ.get("MESHPOOL_BATCH_SPECULATION_MULTIPLIER", "2.0"))
SPECULATION_FLOOR_MS = int(os.environ.get("MESHPOOL_BATCH_SPECULATION_FLOOR_MS", "1500"))
SPECULATION_MIN_COMPLETED = int(os.environ.get("MESHPOOL_BATCH_SPECULATION_MIN_COMPLETED", "2"))


def required_runtime(func: str) -> str:
    """Which execution runtime a unit of ``func`` needs from a node."""
    return "wasm" if is_wasm_function(func) else "builtin"


def node_supports_runtime(node, runtime: str) -> bool:
    """True if ``node`` can actually run a unit of ``runtime``.

    Two independent advertisements have to agree for this to be true: the node
    must serve the ``batch`` workload class at all, *and* it must offer this
    specific runtime within it. Checking only the latter would let an incoherent
    advertisement through — a node that says "I do not run work units" but still
    lists a runtime would keep being handed work it rejects.

    A node registered before either field existed deserializes with the permissive
    default, so this is backwards compatible: it can only ever *exclude* a node
    that explicitly narrowed its own advertisement.
    """
    if not node.provides("batch"):
        return False
    advertised = getattr(node.capability, "runtimes", None)
    if not advertised:
        return True
    return runtime in advertised

# Deterministic curated functions whose result the control plane can cheaply
# recompute as a trusted oracle to spot-check a stranger's node (Tier 2). Functions
# that depend on brokered secrets or wall-clock (secure_fetch, sleep_echo) are not
# verifiable this way and are never spot-checked.
VERIFIABLE_FUNCS = frozenset({"square", "is_prime", "sha_pow", "embed"})


def _results_match(a, b, tol: float = 1e-6) -> bool:
    """Structural comparison of an agent result vs. the local oracle, tolerant of
    float rounding and JSON's tuple->list / int->float coercions."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_results_match(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_results_match(x, y, tol) for x, y in zip(a, b))
    return a == b


class BatchManager:
    def __init__(self, registry: RegistryRef, store, ledger: ContributionLedger,
                 trust: TrustEngine, events, transport: AgentTransport,
                 secrets=None) -> None:
        self.registry = registry
        self.store = store
        self.ledger = ledger
        self.trust = trust
        self.events = events
        self.transport = transport
        self.secrets = secrets       # SecretBroker | None (scoped secret leases)
        # SpendCapManager | None. Wired post-construction by the control plane
        # (same pattern as .storage). None means caps are not enforced, which is
        # what the standalone engine tests want.
        self.spend_caps = None
        # Jobs already announced as spend-capped, so the SSE stream carries one
        # explanation per pause instead of one per dispatch tick.
        self._capped_announced: set[str] = set()
        self.jobs: dict[str, BatchJob] = {j.id: j for j in store.list_jobs()}
        self._inflight: dict[str, int] = {}          # node_id -> running unit count
        self._unit_started: dict[str, float] = {}     # unit_id -> monotonic start
        self._unit_failed_nodes: dict[str, set] = {}   # unit_id -> nodes that failed it
        self._unit_active_attempts: dict[str, int] = {}  # unit_id -> live originals+backups
        self._speculative_nodes: dict[str, set[str]] = {}  # unit_id -> backup node ids
        self._completed_unit_ms: dict[str, list[int]] = {}  # job_id -> accepted durations
        self.speculative_attempts = 0
        self.speculative_wins = 0
        self.speculative_discards = 0
        self.speculative_pairs: list[dict[str, str]] = []
        self.requeued_units = 0
        # Ongoing random audit of already-graduated Tier-2 nodes (traitor defence).
        self._verify_sample_rate = VERIFY_SAMPLE_RATE
        self._audit_rng = random.Random()
        # Verification counters. These are the only externally visible proof that the
        # fraud-detection path actually ran: without them a gate asserting "the liar
        # was caught" cannot tell "caught nobody because nobody lied" apart from
        # "caught nobody because the check never executed".
        self.spot_checks = 0
        self.spot_check_rejections = 0
        # User WASM modules: module_id -> set(node_id) already holding the bytes.
        # Seeding is per (module, node) and permanent for the node's lifetime, so a
        # 1000-unit job over 10 nodes costs 10 transfers, not 1000.
        self.storage = None            # StorageManager, wired by build_app
        # Workload-matched result verifier (spec §P6).  Wired post-construction by
        # build_app() for Tier-2 deployments (same pattern as spend_caps / storage).
        # None means verification infrastructure is absent; all guards below fail-open
        # so the absence of a verifier never blocks dispatch.
        self.verifier = None           # controlplane.verification.Verifier | None
        self._wasm_seeded: dict[str, set] = {}
        self._wasm_cache: dict[str, bytes] = {}
        self._wasm_locks: dict[str, asyncio.Lock] = {}
        self.wasm_module_pushes = 0
        self._lock = asyncio.Lock()

    def speculation_enabled(self) -> bool:
        return os.environ.get("MESHPOOL_BATCH_SPECULATION", "1").lower() not in (
            "0", "false", "no", "off")

    async def _ensure_wasm_on_node(self, func: str, node) -> None:
        """Make sure *node* holds the module for a ``wasm:<sha256>`` job.

        Reuses the agent's content-addressed blob store — the same one that carries
        model weights — so a user's code is distributed by exactly the machinery that
        is already proven to move gigabyte checkpoints, and is verified against its
        hash on arrival.
        """
        module_id = wasm_module_id(func)
        seeded = self._wasm_seeded.setdefault(module_id, set())
        if node.id in seeded:
            return
        # Units are dispatched concurrently, so without this lock every unit that
        # reaches a cold node before the first push completes starts its own push.
        # Measured: 7 transfers of the same module for 3 nodes. The lock is per
        # (module, node) so unrelated nodes still seed in parallel.
        key = f"{module_id}:{node.id}"
        lock = self._wasm_locks.get(key)
        if lock is None:
            lock = self._wasm_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if node.id in seeded:      # won by another unit while we waited
                return
            blob = self._wasm_cache.get(module_id)
            if blob is None:
                if self.storage is None:
                    raise RuntimeError("pooled storage is not available to fetch "
                                       "the wasm module")
                blob = await self.storage.get(module_id)
                if blob is None:
                    raise RuntimeError(
                        f"wasm module {module_id[:12]} is not in the pool — upload "
                        f"it with POST /api/v1/functions first")
                self._wasm_cache[module_id] = blob
            await self.transport.call(
                node.id, "blob_put",
                {"blob_id": module_id,
                 "data": base64.b64encode(blob).decode("ascii")},
                timeout=60.0)
            seeded.add(node.id)
            self.wasm_module_pushes += 1

    def _oracle_payload(self, job, unit) -> dict:
        """The payload the trusted local oracle should re-execute.

        Identical to the unit's payload for a curated function. For a wasm function
        it additionally carries the module bytes, which the coordinator already holds
        (it distributed them) and which ``_run_wasm`` re-verifies against the hash.
        """
        if not is_wasm_function(job.func):
            return unit.payload
        module_id = wasm_module_id(job.func)
        blob = self._wasm_cache.get(module_id)
        if blob is None:
            raise RuntimeError(f"oracle has no copy of wasm module {module_id[:12]}")
        return {**unit.payload,
                "_wasm_module": base64.b64encode(blob).decode("ascii")}

    # --- submission ------------------------------------------------------ #
    def submit(self, name: str, func: str, payloads: list[dict], cpu_pct: int,
               ram_mb: int, owner: str = "default",
               secret_names: Optional[list[str]] = None) -> BatchJob:
        units = [WorkUnit(id=new_id("u"), payload=p) for p in payloads]
        job = BatchJob(id=new_id("job"), name=name, func=func, units=units,
                       cpu_pct_per_unit=cpu_pct, ram_mb_per_unit=ram_mb, owner=owner,
                       secret_names=secret_names or [],
                       status="running", created_ms=now_ms())
        self.jobs[job.id] = job
        self.store.put_job(job)
        log.info("job submitted", job=job.id, units=len(units), func=func)
        self.events.publish("job_submitted", {"id": job.id, "units": len(units)})
        return job

    def get(self, job_id: str) -> Optional[BatchJob]:
        return self.jobs.get(job_id)

    def list(self) -> list[BatchJob]:
        return sorted(self.jobs.values(), key=lambda j: j.created_ms, reverse=True)

    # --- failover -------------------------------------------------------- #
    def requeue_node_units(self, node_id: str) -> int:
        """Requeue units that were assigned/running on a now-offline node."""
        n = 0
        for job in self.jobs.values():
            for u in job.units:
                if u.assigned_node == node_id and u.status in (UnitStatus.ASSIGNED, UnitStatus.RUNNING):
                    u.status = UnitStatus.PENDING
                    u.assigned_node = None
                    self._unit_started.pop(u.id, None)
                    n += 1
            if n:
                self.store.put_job(job)
        self._inflight[node_id] = 0
        if n:
            self.requeued_units += n
            log.warning("requeued units from offline node", node=node_id, units=n)
            self.events.publish("units_requeued", {"node": node_id, "count": n})
        return n

    # --- scheduling ------------------------------------------------------ #
    def _pick_node(self, cpu_pct: int, ram_mb: int, exclude: set[str],
                   runtime: str | None = None):
        """Worst-fit by free CPU slots: pick the schedulable node with the most
        free capacity so units spread across the cluster proportional to each
        node's donatable cores (and never exceed its runtime pool size).

        ``runtime`` is the execution runtime the unit needs ("builtin" or "wasm").
        Nodes that do not advertise it are skipped. Without this filter a
        single-binary Go agent — which runs wasm modules but not the curated Python
        registry — would be handed a ``square`` unit, reject it, and burn one of the
        unit's five attempts; five such assignments in a row fail the unit outright
        on a cluster that was perfectly capable of running it.
        """
        priority = self.store.credits_by_owner() if self.trust.tier > 0 else {}
        best = None
        best_key = None
        for node in self.registry.schedulable_nodes():
            if node.id in exclude:
                continue
            if runtime and not node_supports_runtime(node, runtime):
                continue
            inflight = self._inflight.get(node.id, 0)
            free_slots = node.donatable_cpu_slots() - inflight
            if free_slots <= 0:
                continue
            # RAM is budgeted against the *declared donation cap* minus MeshPool's
            # own in-flight commitment on this node (spec §9.4 "spare declared
            # capacity"). We deliberately do NOT gate on the host's live RAM use:
            # the owner already carved out `policy.ram_mb` for donation and the
            # runtime enforces it, so counting the host's own usage here would make
            # a busy-but-willing machine look full and starve the queue.
            if (inflight + 1) * ram_mb > node.policy.ram_mb:
                continue
            weight = 1.0
            if self.trust.tier > 0:
                weight = fair_share_priority(priority.get(node.owner, 0.0), self.trust.tier)
            # most free slots first (spread); higher fair-share weight breaks ties
            key = (-free_slots, -weight)
            if best_key is None or key < best_key:
                best_key, best = key, node
        return best

    def _runtime_unservable(self, job, runtime: str) -> bool:
        """True when no schedulable node can *ever* run this job's runtime.

        Without this, a job whose runtime nothing in the pool provides sits
        PENDING forever: ``_pick_node`` keeps returning None, ``attempts`` never
        increments, so the MAX_ATTEMPTS terminal path above is never reached and
        the job hangs in "running" with no error. Failing loudly beats hanging
        silently.

        A grace period applies because a capable node may still be joining — we
        only give up once the job has had a fair chance and the pool is provably
        non-empty (an empty pool is "no capacity yet", not "wrong capacity").
        """
        if now_ms() - job.created_ms < NO_RUNTIME_GRACE_MS:
            return False
        nodes = self.registry.schedulable_nodes()
        if not nodes:
            return False
        return not any(node_supports_runtime(n, runtime) for n in nodes)

    def _capped(self, owner: str) -> bool:
        """True when ``owner`` has hit their hard spend cap and dispatch must stop.

        Fails **open** on error: a spend-cap lookup problem must never silently
        freeze the whole mesh's dispatch loop. The admission check in
        ``_require_credits`` still refuses new work in that case.
        """
        if self.spend_caps is None:
            return False
        try:
            return bool(self.spend_caps.exceeded(owner))
        except Exception as exc:  # noqa: BLE001
            log.warning("spend cap check failed; allowing dispatch",
                        owner=owner, err=str(exc)[:120])
            return False

    async def dispatch_once(self) -> None:
        async with self._lock:
            tasks = []
            for job in self.jobs.values():
                if job.status not in ("running", "pending"):
                    continue
                # HARD SPEND CAP (plan §6.3). Enforcing caps only at submission
                # would let one large job overrun the cap without limit, which is
                # the exact failure caps exist to prevent. Units already running
                # are left to finish; we simply stop handing out more.
                if self._capped(job.owner):
                    # Announce once per job so the console can explain WHY a job
                    # stalled. A job that silently stops making progress with no
                    # reason shown is a support ticket waiting to happen.
                    if job.id not in self._capped_announced:
                        self._capped_announced.add(job.id)
                        log.warning("job paused: spend cap reached",
                                    job=job.id, owner=job.owner)
                        self.events.publish("job_spend_capped", {
                            "id": job.id, "owner": job.owner,
                            "reason": "spend cap reached — raise the cap or reset "
                                      "the spend window to resume",
                            **job.progress(),
                        })
                    continue
                if job.id in self._capped_announced:
                    self._capped_announced.discard(job.id)
                    self.events.publish("job_spend_resumed",
                                        {"id": job.id, "owner": job.owner})
                for u in job.units:
                    if u.status is not UnitStatus.PENDING:
                        continue
                    if u.attempts >= MAX_ATTEMPTS:
                        u.status = UnitStatus.FAILED
                        u.error = u.error or "max attempts exceeded"
                        self._unit_failed_nodes.pop(u.id, None)
                        # This unit is now terminal. If it was the last one
                        # outstanding, finalize here — nothing else will (the
                        # _run_unit finally-block that normally finalizes never
                        # runs for a unit we refuse to dispatch), otherwise the
                        # job would hang in "running" forever.
                        self._publish_progress(job)
                        self._finalize(job)
                        self.store.put_job(job)
                        continue
                    # Prefer a node that hasn't already failed this unit (a killed
                    # node, or — at Tier 2 — one caught lying by the spot-check), so a
                    # fast-failing node can't keep grabbing the retry. Fall back to the
                    # broader set if avoiding them would leave nowhere to run, so we
                    # never stall a unit that still has attempts left.
                    last = {u.assigned_node} if u.assigned_node else set()
                    failed = self._unit_failed_nodes.get(u.id, set())
                    need = required_runtime(job.func)
                    node = self._pick_node(job.cpu_pct_per_unit, job.ram_mb_per_unit,
                                           last | failed, runtime=need)
                    if node is None and failed:
                        node = self._pick_node(job.cpu_pct_per_unit,
                                               job.ram_mb_per_unit, last,
                                               runtime=need)
                    if node is None:
                        if self._runtime_unservable(job, need):
                            u.status = UnitStatus.FAILED
                            u.error = (f"no node in the pool provides the "
                                       f"'{need}' runtime")
                            self._unit_failed_nodes.pop(u.id, None)
                            self._publish_progress(job)
                            self._finalize(job)
                            self.store.put_job(job)
                        continue  # no capacity right now; try again next tick
                    u.status = UnitStatus.ASSIGNED
                    u.assigned_node = node.id
                    u.attempts += 1
                    self._inflight[node.id] = self._inflight.get(node.id, 0) + 1
                    self._unit_active_attempts[u.id] = self._unit_active_attempts.get(u.id, 0) + 1
                    tasks.append(self._run_unit(job, u, node, speculative=False))
                if self.speculation_enabled():
                    for u in job.units:
                        if u.status is not UnitStatus.RUNNING:
                            continue
                        if not self._should_speculate(job, u):
                            continue
                        failed = self._unit_failed_nodes.get(u.id, set())
                        already = {u.assigned_node} if u.assigned_node else set()
                        already |= self._speculative_nodes.get(u.id, set())
                        need = required_runtime(job.func)
                        node = self._pick_node(job.cpu_pct_per_unit,
                                               job.ram_mb_per_unit,
                                               already | failed,
                                               runtime=need)
                        if node is None:
                            continue
                        self._speculative_nodes.setdefault(u.id, set()).add(node.id)
                        self._inflight[node.id] = self._inflight.get(node.id, 0) + 1
                        self._unit_active_attempts[u.id] = (
                            self._unit_active_attempts.get(u.id, 0) + 1)
                        self.speculative_attempts += 1
                        self.speculative_pairs.append({
                            "job_id": job.id,
                            "unit_id": u.id,
                            "original": u.assigned_node or "",
                            "backup": node.id,
                        })
                        self.speculative_pairs = self.speculative_pairs[-50:]
                        log.warning("speculative backup dispatched",
                                    job=job.id, unit=u.id,
                                    original=u.assigned_node, backup=node.id)
                        tasks.append(self._run_unit(job, u, node, speculative=True))
            for t in tasks:
                asyncio.create_task(t)

    def _should_speculate(self, job: BatchJob, unit: WorkUnit) -> bool:
        if self._speculative_nodes.get(unit.id):
            return False
        started = self._unit_started.get(unit.id)
        if started is None:
            return False
        durations = self._completed_unit_ms.get(job.id, [])
        if len(durations) < SPECULATION_MIN_COMPLETED:
            return False
        median_ms = statistics.median(durations)
        threshold = max(SPECULATION_FLOOR_MS, int(median_ms * SPECULATION_MULTIPLIER))
        return now_ms() - started > threshold

    def _simulated_straggler_delay(self, payload: dict, node) -> float:
        """Harness-only delay hook for proving backup execution is non-vacuous.

        The delay is control-plane-side, before the agent call, so agent-reported
        ``compute_ms`` and the ledger charge remain identical with speculation on
        and off. It is inert unless the simulation harness opt-in is present.
        """
        if os.environ.get("MESHPOOL_SIM_HARNESS") != "1":
            return 0.0
        target = payload.get("_meshpool_slow_node_name")
        if target and target != getattr(node, "name", ""):
            return 0.0
        try:
            return max(0.0, float(payload.get("_meshpool_slow_extra_s", 0.0)))
        except (TypeError, ValueError):
            return 0.0

    async def _run_unit(self, job: BatchJob, unit: WorkUnit, node,
                        speculative: bool = False) -> None:
        if not speculative:
            unit.status = UnitStatus.RUNNING
            self._unit_started[unit.id] = now_ms()
        started = now_ms()
        # Mint short-lived, node-scoped leases for any secrets this job needs. Only
        # lease ids travel in the dispatch; the worker redeems values just-in-time.
        secret_leases: dict[str, str] = {}
        if job.secret_names and self.secrets is not None:
            try:
                for sname in job.secret_names:
                    lease = self.secrets.mint_lease(sname, node.id, ttl_s=45.0,
                                                    owner=job.owner)
                    secret_leases[sname] = lease.id
            except Exception as exc:  # noqa: BLE001
                unit.status = UnitStatus.FAILED
                unit.error = f"secret error: {str(exc)[:150]}"
                self._inflight[node.id] = max(0, self._inflight.get(node.id, 1) - 1)
                self._unit_started.pop(unit.id, None)
                self._publish_progress(job)
                self._finalize(job)
                self.store.put_job(job)
                return
        try:
            if is_wasm_function(job.func):
                await self._ensure_wasm_on_node(job.func, node)
            delay_s = self._simulated_straggler_delay(unit.payload, node)
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            charge_started = now_ms()
            body = await self.transport.call(
                node.id, "run_unit",
                {"job_id": job.id, "unit_id": unit.id, "func": job.func,
                 "payload": unit.payload, "cpu_pct": job.cpu_pct_per_unit,
                 "ram_mb": job.ram_mb_per_unit, "secret_leases": secret_leases},
                timeout=60.0)
            result = body.get("result")
            # Prefer the agent's measured compute time (excludes sandbox-spawn /
            # network overhead) so the credit charge reflects only the work the
            # consumer actually used; fall back to wall-clock for older agents.
            compute_ms = body.get("compute_ms")
            if compute_ms is not None:
                elapsed_s = max(0.001, float(compute_ms) / 1000.0)
            else:
                elapsed_s = max(0.001, (now_ms() - charge_started) / 1000.0)
            if delay_s > 0:
                elapsed_s = max(0.001, elapsed_s - delay_s)
            if os.environ.get("MESHPOOL_SIM_HARNESS") == "1" and "_meshpool_billing_ms" in unit.payload:
                try:
                    elapsed_s = max(0.001, float(unit.payload["_meshpool_billing_ms"]) / 1000.0)
                except (TypeError, ValueError):
                    pass
            # Track redundant-check verdict across both the oracle and redundant paths.
            # None = skipped/inconclusive; True = verified pass; False = primary lied.
            _redundant_verdict = None
            # Tier 2: spot-check unproven / low-trust strangers against a trusted
            # local oracle. A node that returns a wrong answer is penalised and the
            # unit is retried elsewhere, crediting nobody — so a lying node cannot
            # get its bad results (or its pay) accepted.
            if self._should_verify(job, node.id):
                self.spot_checks += 1
                try:
                    # A wasm job's payload deliberately does not carry the module
                    # (it is content-addressed and distributed separately), so the
                    # oracle has to re-attach the verified bytes. Without this the
                    # oracle would raise, `expected` would be None, and every wasm
                    # result would be accepted unchecked — a fraud hole precisely
                    # where untrusted user code runs.
                    oracle_payload = self._oracle_payload(job, unit)
                    expected = await asyncio.to_thread(run_function, job.func,
                                                       oracle_payload)
                except Exception:  # noqa: BLE001 — oracle failed; don't penalise node
                    expected = None
                if expected is not None and not _results_match(expected, result):
                    self.spot_check_rejections += 1
                    self.trust.record_verification(node.id, False)
                    self.trust.record_task_result(node.id, False)
                    self._unit_failed_nodes.setdefault(unit.id, set()).add(node.id)
                    unit.status = UnitStatus.PENDING   # retry on a different node
                    unit.error = "failed verification spot-check"
                    log.warning("result rejected by spot-check", unit=unit.id,
                                node=node.id, func=job.func)
                    # Escrow: oracle caught a lie — forfeit held bonus credits.
                    if self.verifier is not None:
                        self.verifier.record_fail(node.id)
                    return
                self.trust.record_verification(node.id, True)
                # Escrow: oracle verified honest — accrue bonus credit toward release.
                if self.verifier is not None:
                    self.verifier.record_pass(node.id, node.owner)
                    _redundant_verdict = True   # oracle confirmed; skip redundant
            # Redundant execution (spec §P6): run same unit on a second independent
            # node and compare digests.  Degrades gracefully when the cluster is too
            # small to supply a second node — failing open, never blocking dispatch.
            # Must happen OUTSIDE the accept lock: transport calls cannot hold a lock.
            if self.verifier is not None and _redundant_verdict is None:
                _redundant_verdict = await self._redundant_check(
                    job, unit, result, node)
            if _redundant_verdict is False:
                # Primary node lied in redundant check — reject and retry elsewhere.
                unit.status = UnitStatus.PENDING
                unit.error = "failed redundant verification"
                return
            accepted = False
            async with self._lock:
                if unit.status in (UnitStatus.DONE, UnitStatus.FAILED):
                    self.speculative_discards += 1
                    log.info("discarded late speculative result",
                             job=job.id, unit=unit.id, node=node.id)
                    return
                unit.result = result
                unit.status = UnitStatus.DONE
                unit.assigned_node = node.id
                unit.error = None
                self._unit_failed_nodes.pop(unit.id, None)
                self._completed_unit_ms.setdefault(job.id, []).append(now_ms() - started)
                if speculative:
                    self.speculative_wins += 1
                accepted = True
                cost = job.cpu_pct_per_unit * elapsed_s
                # Donor earns; consumer spends the same amount -> the ledger conserves
                # (net zero when producer == consumer, e.g. a single owner). Only the
                # first accepted attempt records credit, so a backup task cannot double-pay.
                self.ledger.record_donation(node.id, node.owner, cost,
                                            detail=f"batch:{job.id}")
                self.ledger.record_consumption(node.id, job.owner, cost,
                                               detail=f"batch:{job.id}")
                self.trust.record_task_result(node.id, True)
                # Escrow: redundant check confirmed this result — accrue bonus credit.
                if self.verifier is not None and _redundant_verdict is True:
                    self.verifier.record_pass(node.id, node.owner)
        except Exception as exc:  # noqa: BLE001 — any failure -> retry elsewhere
            async with self._lock:
                remaining = max(0, self._unit_active_attempts.get(unit.id, 1) - 1)
                if unit.status not in (UnitStatus.DONE, UnitStatus.FAILED):
                    if remaining == 0:
                        unit.status = UnitStatus.PENDING   # retry on a different node
                        unit.error = str(exc)[:200]
                    else:
                        unit.error = str(exc)[:200]
                    self._unit_failed_nodes.setdefault(unit.id, set()).add(node.id)
                self.trust.record_task_result(node.id, False)
            log.warning("unit failed, will retry", unit=unit.id, node=node.id,
                        attempt=unit.attempts, err=unit.error)
        finally:
            self._inflight[node.id] = max(0, self._inflight.get(node.id, 1) - 1)
            self._unit_active_attempts[unit.id] = max(
                0, self._unit_active_attempts.get(unit.id, 1) - 1)
            if self._unit_active_attempts.get(unit.id, 0) == 0:
                self._unit_active_attempts.pop(unit.id, None)
                self._unit_started.pop(unit.id, None)
                self._speculative_nodes.pop(unit.id, None)
            else:
                self._speculative_nodes.get(unit.id, set()).discard(node.id)
            if unit.status in (UnitStatus.DONE, UnitStatus.FAILED):
                self._publish_progress(job)
            self._finalize(job)
            self.store.put_job(job)

    async def _redundant_check(self, job, unit, primary_result, primary_node) -> Optional[bool]:
        """Run the same unit on a second independent node and compare digests.

        Returns:
          True  — digests match; primary node is verified honest.
          False — tiebreak concluded primary node is the outlier; it has been
                  penalised and the unit should be retried elsewhere.
          None  — check was skipped or inconclusive; proceed as unverified
                  (fail-open: small cluster or secondary unreachable).

        This method must be called OUTSIDE any lock: it makes transport calls.
        Inflight counters are adjusted inside a try/finally to survive exceptions.
        """
        if self.verifier is None:
            return None
        if not self.verifier.should_trigger_redundant(job.func, self._audit_rng):
            return None

        need = required_runtime(job.func)
        failed_set = self._unit_failed_nodes.get(unit.id, set())

        # Pick secondary node (not primary, not already failed this unit)
        secondary = self._pick_node(
            job.cpu_pct_per_unit, job.ram_mb_per_unit,
            {primary_node.id} | failed_set, runtime=need)
        if secondary is None:
            self.verifier.stats.degraded_no_secondary += 1
            return None   # degrade gracefully

        self._inflight[secondary.id] = self._inflight.get(secondary.id, 0) + 1
        try:
            body2 = await self.transport.call(
                secondary.id, "run_unit",
                {"job_id": job.id, "unit_id": unit.id, "func": job.func,
                 "payload": unit.payload, "cpu_pct": job.cpu_pct_per_unit,
                 "ram_mb": job.ram_mb_per_unit, "secret_leases": {}},
                timeout=60.0)
            secondary_result = body2.get("result")
        except Exception:  # noqa: BLE001 — secondary unreachable, degrade
            self.verifier.stats.degraded_secondary_error += 1
            return None
        finally:
            self._inflight[secondary.id] = max(
                0, self._inflight.get(secondary.id, 1) - 1)

        self.verifier.stats.redundant_verifications += 1

        if self.verifier.digests_match(primary_result, secondary_result):
            # Agreement: both nodes returned the same result.
            return True

        # MISMATCH: one node is lying.  Escalate to a tiebreak.
        self.verifier.stats.disagreements += 1
        log.warning("redundant check: digest mismatch — escalating to tiebreak",
                    unit=unit.id, primary=primary_node.id, secondary=secondary.id,
                    func=job.func)

        tiebreak = self._pick_node(
            job.cpu_pct_per_unit, job.ram_mb_per_unit,
            {primary_node.id, secondary.id} | failed_set, runtime=need)
        if tiebreak is None:
            # Can't determine who lied — do not penalise; mark unverified.
            self.verifier.stats.degraded_no_tiebreak += 1
            return None

        self._inflight[tiebreak.id] = self._inflight.get(tiebreak.id, 0) + 1
        try:
            body3 = await self.transport.call(
                tiebreak.id, "run_unit",
                {"job_id": job.id, "unit_id": unit.id, "func": job.func,
                 "payload": unit.payload, "cpu_pct": job.cpu_pct_per_unit,
                 "ram_mb": job.ram_mb_per_unit, "secret_leases": {}},
                timeout=60.0)
            tiebreak_result = body3.get("result")
        except Exception:  # noqa: BLE001 — tiebreak unreachable, mark unverified
            self.verifier.stats.degraded_secondary_error += 1
            return None
        finally:
            self._inflight[tiebreak.id] = max(
                0, self._inflight.get(tiebreak.id, 1) - 1)

        primary_matches_tb = self.verifier.digests_match(primary_result, tiebreak_result)
        secondary_matches_tb = self.verifier.digests_match(secondary_result, tiebreak_result)

        if not primary_matches_tb and secondary_matches_tb:
            # Secondary + tiebreak agree; primary is the outlier.
            self._unit_failed_nodes.setdefault(unit.id, set()).add(primary_node.id)
            self.trust.record_verification(primary_node.id, False)
            self.trust.record_task_result(primary_node.id, False)
            self.verifier.record_fail(primary_node.id)
            log.warning("redundant check: PRIMARY is the outlier — penalised",
                        unit=unit.id, primary=primary_node.id)
            return False   # signal _run_unit to reject and retry

        if primary_matches_tb and not secondary_matches_tb:
            # Primary + tiebreak agree; secondary is the outlier.
            self.trust.record_verification(secondary.id, False)
            self.trust.record_task_result(secondary.id, False)
            self.verifier.record_fail(secondary.id)
            log.warning("redundant check: SECONDARY is the outlier — penalised",
                        unit=unit.id, secondary=secondary.id)
            return True    # primary is fine

        # All three disagree or all three match in an inconsistent way — inconclusive.
        self.verifier.stats.degraded_no_tiebreak += 1
        return None

    def _should_verify(self, job: BatchJob, node_id: str) -> bool:
        """Whether this node's result for this job should be oracle-checked."""
        if self.trust is None or self.trust.tier < 2:
            return False
        if job.func not in VERIFIABLE_FUNCS:
            return False
        if self.trust.needs_verification(node_id):
            return True
        # Graduated, trusted node: still audit a small random fraction so a node that
        # behaved through warmup and only then defects is caught. The first audited
        # lie sets verify_failures, pinning it to per-unit verification thereafter.
        return self._audit_rng.random() < self._verify_sample_rate

    def _finalize(self, job: BatchJob) -> None:
        prog = job.progress()
        if prog["done"] + prog["failed"] == prog["total"] and job.status == "running":
            job.status = "completed" if prog["failed"] == 0 else "completed_with_failures"
            log.info("job finished", job=job.id, status=job.status, **prog)
            self.events.publish("job_finished", {"id": job.id, "status": job.status})

    def _publish_progress(self, job: BatchJob) -> None:
        prog = job.progress()
        self.events.publish("job_progress", {
            "id": job.id,
            "done": prog.get("done", 0),
            "failed": prog.get("failed", 0),
            "running": prog.get("running", 0) + prog.get("assigned", 0),
            "total": prog.get("total", 0),
        })

    async def run_loop(self, interval: float = 0.5) -> None:
        while True:
            try:
                await self.dispatch_once()
            except Exception as exc:  # noqa: BLE001
                log.error("dispatch loop error", err=str(exc))
            await asyncio.sleep(interval)
