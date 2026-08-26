"""Distributed inference engine — the flagship (spec §9.6, §4.4).

Runs a single model that is **too big for any one machine** by sharding its
decoder layers into contiguous **pipeline stages**, one block per node (the
Petals pattern), placed by a **topology-aware partitioner** (the exo pattern,
``scheduler.topology_partition``). The control plane orchestrates a token-by-token
pipeline: only the hidden-state activations (a ``[T, d_model]`` tensor) travel
between stages over the network — the property (spec §0) that makes LAN/Wi-Fi
fast enough.

Guarantees / behaviours:
  * **Correctness oracle:** because every node reconstructs its slice of weights
    from the same checkpoint, the distributed logits are *identical* to a
    single-machine forward pass (verified in ``verify_inference.py``).
  * **Auto re-partitioning on churn:** when a node joins/leaves, the model is
    re-partitioned across the survivors and the shards reloaded — the served
    output is unchanged (same weights), so inference continues seamlessly as long
    as pooled memory suffices.
  * **OpenAI-compatible** chat is exposed by the control-plane API on top of
    ``chat()`` so existing clients work unmodified.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from typing import Optional

import numpy as np

from ..common.logging import get_logger
from ..common.models import ModelDeployment, NodeStatus, new_id
from ..common.transport import AgentTransport, TransportError
from . import model as M
from .adapters import BACKENDS, DEFAULT_BACKEND, get_backend
from .batching import BatchRequest, ContinuousBatcher
from .pipeline import generate_baseline
from .remote import RemotePipeline, build_remote_stages
from .spd import SelfDraft, speculative_generate

log = get_logger("inference")


class InferenceManager:
    def __init__(self, registry, store, ledger, events, transport: AgentTransport) -> None:
        self.registry = registry
        self.store = store
        self.ledger = ledger
        self.events = events
        self.transport = transport
        self._lock = asyncio.Lock()
        self._tok: dict[str, M.CharTokenizer] = {}   # model_id -> tokenizer
        self._rr: dict[str, int] = {}                # model_id -> round-robin replica cursor
        self.backend = get_backend(DEFAULT_BACKEND)  # default adapter (meshshard); spec §9.6
        # Set post-construction in the control-plane lifespan. A model too big to
        # pre-install on every node has its weights staged in the pooled blob store;
        # this handle hydrates them (cached by content-addressed blob id) at load time.
        self.storage = None
        self._manifest_cache: dict[str, tuple[dict, Optional[dict]]] = {}
        # Placement telemetry: how many weight-group blobs were put on a node by a direct
        # peer-to-peer pull vs. relayed through the control plane (fallback / tunnel) vs.
        # already resident from an earlier deployment (warm residency — the cheapest
        # outcome, and the one Weight Swarm exists to maximise).
        self.weight_group_pulls = 0
        self.weight_group_pushes = 0
        self.weight_group_cached = 0
        # One continuous batcher per deployment, shared by every concurrent request.
        #
        # This is what turns the pipeline from single-tenant into multi-tenant. Each
        # entry is keyed by (deployment id, partition fingerprint) so that a
        # re-partition retires the old batcher instead of letting it keep talking to
        # a stage that has moved.
        self._batchers: dict[str, tuple[str, ContinuousBatcher, RemotePipeline]] = {}
        self.max_batch = int(os.environ.get("MESHPOOL_MAX_BATCH", "8"))

    # --- queries --------------------------------------------------------- #
    def list(self) -> list[ModelDeployment]:
        return self.store.list_models()

    def get(self, model_id: str) -> Optional[ModelDeployment]:
        return self.store.get_model(model_id)

    def _plan_partition(self, backend, nodes, cfg, rtt_predictor=None):
        """Plan a partition, injecting the Vivaldi RTT predictor for meshshard."""
        if not backend.native_generate and rtt_predictor is not None:
            from ..controlplane.scheduler import topology_partition
            return topology_partition(
                nodes, cfg.n_layer, cfg.per_layer_mb,
                rtt_predictor=rtt_predictor,
            )
        return backend.plan_partition(nodes, cfg)

    def view(self, model_id: str) -> Optional[dict]:
        """Dashboard-shaped summary of a deployment."""
        dep = self.store.get_model(model_id)
        if not dep:
            return None
        return {
            "deployed": dep.status == "ready",
            "id": dep.id,
            "model": dep.model,
            "backend": getattr(dep, "backend", "meshshard"),
            "pooled": bool(getattr(dep, "weights_blob_id", None)),
            "weight_groups": getattr(dep, "weight_groups", None),
            "status": dep.status,
            "n_layers": dep.n_layers,
            "replicas": len(dep.partition),
            "tokens_per_sec": dep.tokens_per_sec,
            "stages": [
                {"node_id": a.node_id, "node_name": self._node_name(a.node_id),
                 "first": a.start_layer, "last": a.end_layer,
                 "is_first": a.is_first, "is_last": a.is_last}
                for a in sorted(dep.partition, key=lambda a: a.start_layer)
            ],
        }

    def _node_name(self, node_id: str) -> str:
        n = self.registry.get(node_id)
        return n.name if n else node_id

    def _tokenizer(self, cfg: M.ModelConfig, meta: Optional[dict]) -> M.CharTokenizer:
        return self.backend.tokenizer(cfg, meta)

    def _backend_for(self, dep: ModelDeployment):
        """The engine adapter that serves an existing deployment."""
        return BACKENDS.get(getattr(dep, "backend", None) or DEFAULT_BACKEND, self.backend)

    def _select_backend(self, model_name: str, nodes: list):
        """Choose the adapter for a NEW deployment of ``model_name``.

        A native-generate backend (a real LLM the pool already runs, e.g. via
        Ollama) wins whenever a scheduled node advertises the model; otherwise fall
        back to the built-in meshshard net. Raises ValueError if neither can serve.
        """
        ollama = BACKENDS.get("ollama")
        if (ollama is not None and ollama.available()
                and ollama.nodes_with_model(nodes, model_name)):
            return ollama
        if self.backend.resolve_config(model_name) is not None:
            return self.backend
        raise ValueError(
            f"unknown model {model_name!r}: not a built-in model and no online node "
            f"advertises it via Ollama (pull it on a contributor node first)")

    # --- deploy ---------------------------------------------------------- #
    async def deploy(self, model_name: str = "nano-gpt", owner: str = "default",
                     weights_blob_id: Optional[str] = None) -> ModelDeployment:
        async with self._lock:
            nodes = self.registry.schedulable_nodes()
            backend = self._select_backend(model_name, nodes)
            cfg = backend.resolve_config(model_name)
            # Use Vivaldi-predicted RTTs to reorder pipeline stages so consecutive
            # stages land on low-latency node pairs.  Falls back to the default
            # ordering when coordinates are not yet available or trusted.
            rtt_pred = getattr(self.registry, "predicted_rtt", None)
            partition = self._plan_partition(backend, nodes, cfg, rtt_predictor=rtt_pred)
            if not partition:
                if backend.native_generate:
                    raise RuntimeError(
                        f"no online node currently serves {model_name!r} via Ollama")
                pooled = self._pooled_layer_capacity(nodes, cfg)
                shard_capable = sum(1 for n in nodes if n.provides("shards"))
                detail = ""
                if shard_capable < len(nodes):
                    detail = (f" ({len(nodes) - shard_capable} of {len(nodes)} online "
                              f"nodes do not serve model shards and were not counted)")
                raise RuntimeError(
                    f"insufficient pooled memory: need {cfg.n_layer} layers "
                    f"(~{cfg.n_layer * cfg.per_layer_mb} MB), pooled capacity "
                    f"{pooled} layers{detail}")
            dep = ModelDeployment(id=new_id("mdl"), model=model_name, backend=backend.name,
                                  n_layers=cfg.n_layer, partition=partition, owner=owner,
                                  weights_blob_id=weights_blob_id, status="loading")
            self.store.put_model(dep)
            self.events.publish("model_deploying",
                                {"id": dep.id, "model": model_name, "backend": backend.name,
                                 "stages": len(partition)})
            if not backend.native_generate:
                manifest, meta = await self._resolve_manifest(dep)
                dep.weight_groups = len(manifest["groups"]) if manifest else None
                await self._load_shards(dep, cfg, manifest)
                if meta is None:
                    _, meta = M.load_params(cfg)
                self._tok[dep.id] = backend.tokenizer(cfg, meta)
            dep.status = "ready"
            self.store.put_model(dep)
        log.info("model deployed", id=dep.id, model=model_name, backend=backend.name,
                 stages=len(partition),
                 layout=[[a.start_layer, a.end_layer] for a in partition])
        self.events.publish("model_ready",
                            {"id": dep.id, "model": model_name, "backend": backend.name,
                             "stages": len(dep.partition)})
        return dep

    def _pooled_layer_capacity(self, nodes, cfg) -> int:
        from ..controlplane.scheduler import usable_layer_memory_mb
        total = 0
        for n in nodes:
            # Must use the SAME memory rule *and the same eligibility rule* as
            # topology_partition, or this diagnostic will contradict the planner
            # it is explaining ("pooled capacity 32 layers" next to a refusal).
            if not n.provides("shards"):
                continue
            mem = usable_layer_memory_mb(n)
            mem = min(mem, n.policy.ram_mb) if n.policy.ram_mb else mem
            total += int((mem * 0.8) // max(1, cfg.per_layer_mb))
        return total

    async def _resolve_manifest(self, dep: ModelDeployment
                                ) -> tuple[Optional[dict], Optional[dict]]:
        """Fetch a pooled model's weight *manifest* (group name -> blob id) and its
        small tokenizer meta from the pool, cached by id.

        Returns ``(None, None)`` for a built-in model whose weights every node already
        has on disk — the historical path, untouched. The manifest and meta are tiny
        (ids + vocab), so resolving them never pulls the whole checkpoint into the
        control plane — the property that lets a larger-than-any-machine model deploy
        and re-partition on nothing but blob ids."""
        bid = getattr(dep, "weights_blob_id", None)
        if not bid or self.storage is None:
            return None, None
        cached = self._manifest_cache.get(bid)
        if cached is not None:
            return cached
        raw = await self.storage.get(bid)
        if raw is None:
            raise RuntimeError(f"weight manifest {bid[:12]} lost from the pool")
        manifest = json.loads(raw.decode("utf-8"))
        meta: Optional[dict] = None
        mid = manifest.get("meta")
        if mid:
            mraw = await self.storage.get(mid)
            if mraw is not None:
                _, meta = M.loads_npz(mraw)
        self._manifest_cache[bid] = (manifest, meta)
        return manifest, meta

    async def _place_blobs(self, node_id: str, blob_ids: list[str]) -> None:
        """Ensure each weight-group blob is present on ``node_id``.

        Preferred path (LAN): the node pulls each blob it lacks **directly from a peer**
        that holds it, so the control plane carries only ids + peer URLs and never the
        weight bytes. Fallback (a tunnel-attached node, or no reachable peer holder): the
        control plane relays the bytes itself, holding at most one group in memory at a
        time. Either way the node ends up with the whole group under its content id."""
        gate = getattr(self.transport, "peer_pull_ok", None)
        may_pull = gate(node_id) if callable(gate) else False
        for bid in blob_ids:
            try:
                has = await self.transport.call(
                    node_id, "blob_has", {"blob_id": bid}, timeout=10.0)
                if has.get("has"):
                    self.weight_group_cached += 1
                    continue
            except TransportError:
                pass   # can't confirm -- fall through and (re)place it
            if may_pull and await self._try_peer_pull(node_id, bid):
                continue
            await self._push_blob(node_id, bid)

    async def _try_peer_pull(self, node_id: str, bid: str) -> bool:
        """Attempt a direct peer-to-peer pull of one group blob onto ``node_id``. Returns
        True on success; False (so the caller relays it via the control plane) when no
        plan is available or the pull could not be delivered."""
        plan = self._pull_plan(bid, exclude_node=node_id)
        if plan is None:
            return False
        try:
            await self.transport.call(node_id, "blob_pull", plan, timeout=120.0)
        except TransportError as exc:
            log.warning("peer pull failed, relaying via control plane",
                        node=node_id, blob=bid[:12], err=str(exc)[:120])
            return False
        self.weight_group_pulls += 1
        return True

    async def _push_blob(self, node_id: str, bid: str) -> None:
        """Relay one group blob through the control plane (the pre-P2P path): fetch it
        from the pool and push it to the node, holding at most one group in memory."""
        raw = await self.storage.get(bid)
        if raw is None:
            raise RuntimeError(f"weight blob {bid[:12]} lost from the pool")
        payload = base64.b64encode(raw).decode("ascii")
        await self.transport.call(
            node_id, "blob_put", {"blob_id": bid, "data": payload}, timeout=60.0)
        self.weight_group_pushes += 1

    def _pull_plan(self, blob_id: str, exclude_node: str) -> Optional[dict]:
        """Build a byte-free peer-pull plan for ``blob_id`` from storage metadata: the
        peer URLs (and, for a chunked object, the per-chunk peer URLs) a target node can
        pull directly, excluding the target itself and offline holders. Returns None if
        the object is unknown or any piece has no reachable holder -- the caller then
        relays the bytes through the control plane instead."""
        if self.storage is None:
            return None
        meta = self.storage.get_meta(blob_id)
        if meta is None:
            return None
        if meta.chunks:
            chunks: list[dict] = []
            for cid in meta.chunks:
                cm = self.storage.get_meta(cid)
                srcs = self._holder_urls(cm.replicas, exclude_node) if cm else []
                if not srcs:
                    return None
                chunks.append({"id": cid, "sources": srcs})
            return {"blob_id": blob_id, "chunks": chunks}
        srcs = self._holder_urls(meta.replicas, exclude_node)
        if not srcs:
            return None
        return {"blob_id": blob_id, "sources": srcs}

    def _holder_urls(self, node_ids, exclude_node: str) -> list[str]:
        """Advertised URLs of the online nodes that hold a blob, minus the target."""
        urls: list[str] = []
        for nid in node_ids:
            if nid == exclude_node:
                continue
            n = self.registry.get(nid)
            if n is not None and n.status is NodeStatus.ONLINE and getattr(n, "url", None):
                urls.append(n.url)
        return urls

    async def _load_shards(self, dep: ModelDeployment, cfg: M.ModelConfig,
                           manifest: Optional[dict] = None) -> None:
        """Instruct every stage node to build its ShardModule.

        Built-in model (no manifest): each node loads its slice from the checkpoint it
        already has on disk. Pooled model: ship each node only the *blob ids* of the
        layer groups it hosts (placing any it lacks), and let it assemble its slice from
        its local store — so the control plane never materializes the whole checkpoint
        and no node ever holds more than its own layers."""
        if manifest is None:
            manifest, _ = await self._resolve_manifest(dep)
        for a in dep.partition:
            weight_blobs: Optional[list[str]] = None
            if manifest is not None:
                names = M.stage_groups(a.start_layer, a.end_layer, a.is_first, a.is_last)
                weight_blobs = [manifest["groups"][n] for n in names]
                await self._place_blobs(a.node_id, weight_blobs)
            await self.transport.call(
                a.node_id, "shard/load",
                {"model_id": dep.id, "model_name": dep.model,
                 "start_layer": a.start_layer, "end_layer": a.end_layer,
                 "is_first": a.is_first, "is_last": a.is_last,
                 "weight_blobs": weight_blobs},
                timeout=60.0)

    async def _unload_shards(self, dep: ModelDeployment) -> None:
        for a in dep.partition:
            try:
                await self.transport.call(a.node_id, "shard/unload",
                                          {"model_id": dep.id}, timeout=5.0)
            except Exception:  # noqa: BLE001
                pass

    async def remove(self, model_id: str) -> bool:
        await self._retire_batcher(model_id)
        async with self._lock:
            dep = self.store.get_model(model_id)
            if not dep:
                return False
            if not self._backend_for(dep).native_generate:
                await self._unload_shards(dep)
            self.store.delete_model(model_id)
            self._tok.pop(model_id, None)
            self._rr.pop(model_id, None)
            bid = getattr(dep, "weights_blob_id", None)
            if bid:
                self._manifest_cache.pop(bid, None)
        self.events.publish("model_removed", {"id": model_id})
        return True

    # --- chat / generate ------------------------------------------------- #
    async def chat(self, model_id: str, prompt: str, max_new_tokens: int = 48,
                   temperature: float = 0.8, seed: int = 0,
                   speculative: bool = True, draft_k: int = 4,
                   draft_layers: int = 2, chunked_prefill: int = 0) -> dict:
        """Generate over the sharded pipeline.

        Decoding is KV-cached, so a decode step ships ``d_model`` floats per hop
        instead of re-sending the whole context every token, and prefill is
        position-parallel (one traversal for the entire prompt, not one per token).
        With ``speculative`` on, Speculative Pipeline Decoding verifies several
        drafted tokens in a single traversal, which amortises the ``(N-1) * RTT``
        term that otherwise dominates wide-area decoding. It is exact: greedy output
        is token-identical to ordinary decoding, and sampled output has the same
        distribution (proven in ``sim/scenarios/verify_spd_equivalence.py``).
        """
        dep = self.store.get_model(model_id)
        if not dep:
            raise KeyError("unknown model deployment")
        if dep.status != "ready":
            raise RuntimeError(f"model not ready (status={dep.status})")
        backend = self._backend_for(dep)
        if backend.native_generate:
            return await self._native_chat(dep, backend, prompt, max_new_tokens, temperature)
        cfg = backend.resolve_config(dep.model)
        tok = self._tok.get(model_id) or backend.tokenizer(cfg, M.load_params(cfg)[1])
        self._tok[model_id] = tok

        prompt_ids = tok.encode(prompt) or [tok.encode(" ")[0]]

        use_spd = speculative and cfg.n_layer > draft_layers >= 1
        batcher, pipe = await self._batcher_for(dep, cfg)
        session = f"{dep.id}:{new_id('sess')}"

        draft_factory = None
        if use_spd:
            # The draft runs on the *first stage's own layers*, which already live on
            # one machine, so drafting costs zero extra wide-area hops. See
            # meshpool/inference/spd.py for why this is exact and not approximate.
            full_params = M.load_params(cfg)[0]
            draft_factory = lambda: SelfDraft(cfg, full_params, draft_layers)  # noqa: E731

        t0 = time.perf_counter()
        result_b = await batcher.generate(BatchRequest(
            prompt=prompt_ids, max_new_tokens=max_new_tokens, temperature=temperature,
            seed=seed, session=session, chunked_prefill=chunked_prefill,
            draft_factory=draft_factory, draft_k=draft_k))
        elapsed = max(1e-6, time.perf_counter() - t0)

        generated = result_b.tokens
        stats = result_b.stats
        spec = result_b.spec

        tps = len(generated) / elapsed
        text = tok.decode(generated)

        dep.tokens_per_sec = round(tps, 2)
        self.store.put_model(dep)
        self._record_ledger(dep, elapsed)
        self.events.publish("inference",
                            {"id": dep.id, "tokens": len(generated),
                             "tokens_per_sec": round(tps, 2)})
        result = {
            "model": dep.model,
            "deployment_id": dep.id,
            "prompt": prompt,
            "text": text,
            "tokens": len(generated),
            "finish_reason": result_b.finish_reason,
            "tokens_per_sec": round(tps, 2),
            "elapsed_s": round(elapsed, 3),
            "pipeline": stats.as_dict(),
            # How much of the wide-area cost this request shared with other tenants.
            # mean_batch 1.0 means it ran alone; 4.0 means four requests split every
            # round trip between them.
            "batch": {
                "mean_batch": round(result_b.mean_batch, 3),
                "queued_s": round(result_b.queued_s, 4),
                "ttft_s": round(result_b.ttft_s, 4),
                "max_batch": batcher.max_batch,
            },
            "stages": [
                {"node_name": self._node_name(a.node_id),
                 "first": a.start_layer, "last": a.end_layer}
                for a in sorted(dep.partition, key=lambda a: a.start_layer)
            ],
        }
        if spec is not None:
            result["speculative"] = spec
        return result

    @staticmethod
    def _partition_fingerprint(dep: ModelDeployment) -> str:
        return "|".join(f"{a.node_id}:{a.start_layer}-{a.end_layer}"
                        for a in sorted(dep.partition, key=lambda a: a.start_layer))

    async def _batcher_for(self, dep: ModelDeployment,
                           cfg: M.ModelConfig) -> tuple[ContinuousBatcher, RemotePipeline]:
        """The batcher every concurrent request for this deployment shares.

        Sharing is the entire point: two users chatting at the same time must ride the
        same traversals rather than each paying their own ``(N-1) * RTT``. The
        fingerprint check retires a batcher whose partition has moved underneath it,
        so a re-partition can never leave requests talking to a stage that no longer
        holds those layers.
        """
        fp = self._partition_fingerprint(dep)
        existing = self._batchers.get(dep.id)
        if existing is not None:
            if existing[0] == fp:
                return existing[1], existing[2]
            await self._retire_batcher(dep.id)
        pipe = RemotePipeline(cfg, build_remote_stages(self.transport, dep.id,
                                                       dep.partition))
        batcher = ContinuousBatcher(pipe, max_batch=self.max_batch)
        await batcher.start()
        self._batchers[dep.id] = (fp, batcher, pipe)
        return batcher, pipe

    async def _retire_batcher(self, model_id: str) -> None:
        entry = self._batchers.pop(model_id, None)
        if entry is None:
            return
        _, batcher, pipe = entry
        try:
            await batcher.aclose()
        finally:
            await pipe.flush()

    async def _native_chat(self, dep: ModelDeployment, backend, prompt: str,
                           max_new_tokens: int, temperature: float) -> dict:
        """Serve one request from a whole-model replica (native-generate backends).

        Picks a replica round-robin and fails over to the next if a node errors, so
        concurrent requests spread across every node that holds the model (aggregate
        throughput) and a node dropping mid-pool never fails the caller. Only the
        prompt/response text crosses the network — the model runs entirely on one
        node, so per-request latency matches that node's local Ollama exactly.
        """
        replicas = []
        for a in dep.partition:
            n = self.registry.get(a.node_id)
            if n is not None and n.status is NodeStatus.ONLINE:
                replicas.append(a)
        if not replicas:
            raise RuntimeError("no online replica currently holds this model")
        start = self._rr.get(dep.id, 0) % len(replicas)
        self._rr[dep.id] = start + 1
        order = replicas[start:] + replicas[:start]
        last_err: Optional[Exception] = None
        for a in order:
            t0 = time.perf_counter()
            try:
                body = await self.transport.call(
                    a.node_id, "ollama/generate",
                    {"model": dep.model, "prompt": prompt,
                     "max_tokens": max_new_tokens, "temperature": temperature},
                    timeout=600.0)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                log.warning("replica generate failed, failing over",
                            node=a.node_id, err=str(exc)[:120])
                continue
            elapsed = max(1e-6, time.perf_counter() - t0)
            text = body.get("text", "")
            tokens = int(body.get("tokens", 0) or 0)
            tps = float(body.get("tokens_per_sec") or (tokens / elapsed if tokens else 0.0))
            node = self.registry.get(a.node_id)
            served_by = node.name if node else a.node_id
            dep.tokens_per_sec = round(tps, 2)
            self.store.put_model(dep)
            if node:
                self.ledger.record_donation(node.id, node.owner, elapsed,
                                            detail=f"inference:{dep.id}")
            self.events.publish("inference",
                                {"id": dep.id, "tokens": tokens,
                                 "tokens_per_sec": dep.tokens_per_sec, "node": served_by})
            return {
                "model": dep.model,
                "deployment_id": dep.id,
                "prompt": prompt,
                "text": text,
                "tokens": tokens,
                "tokens_per_sec": dep.tokens_per_sec,
                "elapsed_s": round(elapsed, 3),
                "backend": backend.name,
                "served_by": served_by,
                "stages": [
                    {"node_name": self._node_name(x.node_id),
                     "first": x.start_layer, "last": x.end_layer}
                    for x in dep.partition
                ],
            }
        raise RuntimeError(f"all replicas failed: {str(last_err)[:160]}")

    def _record_ledger(self, dep: ModelDeployment, elapsed_s: float) -> None:
        """Credit each stage node for compute donated to this inference."""
        for a in dep.partition:
            node = self.registry.get(a.node_id)
            if not node:
                continue
            layers = max(1, a.end_layer - a.start_layer)
            self.ledger.record_donation(node.id, node.owner, layers * elapsed_s,
                                        detail=f"inference:{dep.id}")

    # --- churn: re-partition across survivors ---------------------------- #
    def on_node_offline(self, node_id: str) -> None:
        """Called by the reaper when a node drops. If any live model used that node,
        schedule an async re-partition on the survivors."""
        affected = [d for d in self.store.list_models()
                    if any(a.node_id == node_id for a in d.partition)
                    and d.status in ("ready", "loading", "degraded")]
        if not affected:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        for dep in affected:
            self.events.publish("model_repartitioning", {"id": dep.id, "lost": node_id})
            if loop is not None:
                loop.create_task(self._repartition(dep.id, node_id))

    async def _repartition(self, model_id: str, lost_node_id: str) -> None:
        # A batcher holds RemoteShards pinned to the *old* stage nodes, one of which
        # has just gone. Retire it before replanning so no in-flight or subsequent
        # request can be handed a stage that no longer exists.
        await self._retire_batcher(model_id)
        async with self._lock:
            dep = self.store.get_model(model_id)
            if not dep:
                return
            backend = self._backend_for(dep)
            cfg = backend.resolve_config(dep.model)
            if cfg is None:
                return
            survivors = [n for n in self.registry.schedulable_nodes() if n.id != lost_node_id]
            rtt_pred = getattr(self.registry, "predicted_rtt", None)
            partition = self._plan_partition(backend, survivors, cfg, rtt_predictor=rtt_pred)
            if not partition:
                dep.status = "degraded"
                self.store.put_model(dep)
                log.warning("cannot re-partition: insufficient survivor capacity",
                            id=dep.id, lost=lost_node_id)
                self.events.publish("model_degraded",
                                    {"id": dep.id, "reason": "insufficient capacity"})
                return
            dep.partition = partition
            dep.status = "loading"
            self.store.put_model(dep)
            try:
                if not backend.native_generate:
                    await self._load_shards(dep, cfg)
                dep.status = "ready"
                self.store.put_model(dep)
                log.info("model re-partitioned", id=dep.id,
                         stages=len(partition),
                         layout=[[a.start_layer, a.end_layer] for a in partition])
                self.events.publish("model_ready",
                                    {"id": dep.id, "model": dep.model,
                                     "stages": len(partition), "repartitioned": True})
            except Exception as exc:  # noqa: BLE001
                dep.status = "degraded"
                self.store.put_model(dep)
                log.error("re-partition shard load failed", id=dep.id, err=str(exc)[:160])
