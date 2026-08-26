"""Drive a real, multi-node pipeline with the same code that the sim harness proves.

:mod:`meshpool.inference.pipeline` and :mod:`meshpool.inference.spd` are written
against a small stage interface — ``forward`` / ``truncate`` / ``drop`` — and know
nothing about where a stage physically lives. :class:`RemoteShard` implements that
interface over :class:`meshpool.common.transport.AgentTransport`, so KV caching,
chunked prefill and Speculative Pipeline Decoding run **unchanged** against machines
scattered across the internet.

That is deliberate: the acceptance gates
(``sim/scenarios/verify_spd_equivalence.py``) exercise exactly the algorithms used in
production, and only the link underneath them is swapped. A bug fixed in the sim is
fixed on the mesh, and a property proven in the sim is a property of the mesh.

Folding rollback into the next traversal
----------------------------------------
Speculation needs every stage's KV cache rolled back to the accepted length after each
round. Doing that as its own request would cost a full extra ``(N-1) * RTT`` per round
— which on a wide-area mesh is precisely the cost SPD exists to avoid, and would cancel
most of its benefit. So :meth:`RemoteShard.truncate` only *records* the rollback, and
the pending length rides along on the next ``shard/forward`` request, where the node
applies it before computing. **Rollback therefore costs zero extra round trips.**
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional, Sequence

import numpy as np

from . import model as M
from ..common.transport import TransportError
from .pipeline import LinkProfile, Pipeline, StageCall

# A node that does not know the batched hop answers with one of these. Anything
# else is a real failure and must not be silently downgraded to the slow path.
_BATCH_UNSUPPORTED_STATUS = frozenset({404, 405, 501})


def spare_model_id(model_id: str, start: int, end: int) -> str:
    """The key a *standby* copy of one stage is loaded under on its node.

    A node keys its loaded shards by model id alone (``state.shards[model_id]``), so
    a machine that already serves stage ``k`` of a deployment would have that shard
    silently **overwritten** if the standby copy of stage ``j`` arrived under the same
    id -- turning an attempt to add redundancy into an outage. Qualifying the id with
    the layer range lets one machine hold several stages of the same model, which is
    what makes redundancy possible at all on a mesh with no idle nodes: the planner
    spreads a model across everything it has, so "put the spare somewhere else"
    usually has nowhere else to point.
    """
    return f"{model_id}@{start}-{end}"


class StageUnavailable(RuntimeError):
    """Every candidate node for one pipeline stage has failed.

    Raised only when there is genuinely nothing left to try. It is deliberately a
    distinct type so a caller can tell "this stage is gone" from "this request was
    malformed" -- the first is worth re-partitioning the deployment over, the second
    is not.
    """

    def __init__(self, stage: "RemoteShard", last: BaseException) -> None:
        super().__init__(
            f"stage {stage.start}:{stage.end} exhausted all "
            f"{len(stage.candidates)} candidate node(s) "
            f"({', '.join(stage.candidates)}): {type(last).__name__}: {last}")
        self.stage = stage
        self.last_error = last


class RemoteShard:
    """One pipeline stage that lives on another machine.

    Mirrors :class:`meshpool.inference.pipeline.CachedShard`'s interface exactly, so
    :class:`~meshpool.inference.pipeline.Pipeline` cannot tell the difference.

    Hot spares
    ----------
    A stage holds a *list* of candidate nodes, not one. In a pipeline a single dead
    stage stalls everything behind it, and on consumer hardware nodes do not fail
    politely -- they thermally throttle, someone launches a game, Wi-Fi drops. So a
    transport failure rotates to the next candidate and the traversal continues,
    rather than the generation dying.

    The spare is **cold**: it has no KV cache for any session that was living on the
    node that just died. Sending the next decode step straight to it would be a
    request to resume at position ``t`` against an empty cache, which both the
    in-process shard and the real node correctly refuse with a cache-state error
    (HTTP 409) rather than answering from nothing. So the failure is loud -- but the
    generation still dies, which for the user is indistinguishable from the node
    having no spare at all. Hydration is what converts that into survival: every
    session is marked dirty on rotation, and the first request for each session
    replays its history through the new node first (O(T) through one stage, not a
    restart of the generation). Sessions that are never touched again are never
    hydrated, so the cost is proportional to what is actually still in flight.
    """

    def __init__(self, transport: Any, node_id: Any, model_id: str,
                 start: int, end: int, is_first: bool, is_last: bool,
                 timeout: float = 60.0,
                 spares: Optional[Sequence[str]] = None,
                 spare_model_id: Optional[str] = None) -> None:
        self.transport = transport
        candidates = [node_id] if isinstance(node_id, str) else [str(n) for n in node_id]
        for extra in (spares or ()):
            if extra not in candidates:
                candidates.append(str(extra))
        if not candidates:
            raise ValueError("a stage needs at least one candidate node")
        self.candidates: list[str] = candidates
        self._active = 0
        self._model_id = model_id
        # Standby copies may live on a machine that already serves another stage of
        # the same deployment, so they are loaded under a stage-qualified id. See
        # spare_model_id(). Defaults to the primary id, which keeps a stage built
        # outside build_remote_stages behaving exactly as before.
        self._spare_model_id = spare_model_id or model_id
        self.start, self.end = start, end
        self.is_first, self.is_last = is_first, is_last
        self.timeout = timeout
        self._truncate_to: dict[str, int] = {}
        self._dropped: set[str] = set()
        # None = not probed yet, True = node batches, False = latched to the fallback.
        self._batch_supported: Optional[bool] = None
        self.batch_fallbacks = 0
        # -- hot-spare state ------------------------------------------------- #
        self.failovers = 0
        self.dead: list[str] = []
        self.hydrations = 0
        # Sessions this stage has served, and those the *current* node has no cache
        # for. Tracked separately because a rotation invalidates every session at
        # once but hydration is paid lazily, per session, only if it is used again.
        self._sessions: set[str] = set()
        self._needs_hydration: set[str] = set()
        # Installed by RemotePipeline: rehydrate(session, upto_pos) rebuilds THIS
        # stage on whichever node is now active.
        self.rehydrate: Optional[Callable[[str, int], Awaitable[None]]] = None
        self._hydrating = False

    @property
    def node_id(self) -> str:
        """The node currently serving this stage."""
        return self.candidates[self._active]

    @property
    def model_id(self) -> str:
        """The shard key to address on the node currently serving this stage."""
        return self._model_id if self._active == 0 else self._spare_model_id

    @property
    def spares_remaining(self) -> int:
        return len(self.candidates) - self._active - 1

    def _rotate(self) -> bool:
        """Move to the next candidate. False when there is nothing left to try."""
        if self._active + 1 >= len(self.candidates):
            return False
        self.dead.append(self.candidates[self._active])
        self._active += 1
        self.failovers += 1
        # The new node has no cache for anything. Everything is suspect until it has
        # been rebuilt there -- including any rollback that was still pending for the
        # old node, which is now meaningless because there is nothing to roll back:
        # replaying an accepted history rebuilds the post-rollback state directly.
        self._truncate_to.clear()
        self._needs_hydration = set(self._sessions)
        self._batch_supported = None
        return True

    async def _ensure_hydrated(self, sessions: Sequence[Optional[str]],
                               positions: Sequence[int]) -> None:
        """Rebuild any session in ``sessions`` that the current node lacks."""
        if self._hydrating or not self._needs_hydration:
            return
        for session, pos in zip(sessions, positions):
            if session is None or session not in self._needs_hydration:
                continue
            self._needs_hydration.discard(session)
            if pos <= 0:
                # Nothing precedes this request, so there is nothing to rebuild: the
                # node is about to be handed position 0 and will build the cache itself.
                continue
            if self.rehydrate is None:
                raise StageUnavailable(self, RuntimeError(
                    f"session {session!r} needs {pos} positions replayed onto the "
                    f"spare, but this stage has no rehydrate hook: it was built "
                    f"outside RemotePipeline"))
            self._hydrating = True
            try:
                await self.rehydrate(session, pos)
                self.hydrations += 1
            finally:
                self._hydrating = False

    def _is_node_failure(self, exc: BaseException) -> bool:
        """True when ``exc`` means "this node is gone", not "this request was bad".

        Rotating on a 400 would burn every spare on a malformed payload and then
        report the stage as unavailable, hiding a client bug behind an infrastructure
        story. Only transport-level faults and server-side faults rotate.
        """
        if isinstance(exc, TransportError):
            status = getattr(exc, "status", None)
            if status is None:
                return True
            return status >= 500 or status in (408, 429)
        return isinstance(exc, (asyncio.TimeoutError, ConnectionError, OSError))

    # -- cache control: recorded now, transmitted with the next forward ------ #
    def truncate(self, session: str, length: int) -> None:
        self._truncate_to[session] = length

    def drop(self, session: str) -> None:
        self._truncate_to.pop(session, None)
        self._needs_hydration.discard(session)
        if not self._hydrating:
            # A hydration replay drops-then-rebuilds on the *current* node; that is
            # not the session going away, so it must not un-track it.
            self._sessions.discard(session)
        self._dropped.add(session)

    def cache_bytes(self) -> int:  # remote; not locally observable
        return 0

    async def flush(self) -> None:
        """Release remote cache memory for sessions dropped since the last flush.

        Called once when a generation finishes, so it never sits on a token's
        critical path.
        """
        pending, self._dropped = self._dropped, set()
        for session in pending:
            try:
                await self.transport.call(
                    self.node_id, "shard/drop",
                    {"model_id": self.model_id, "session": session}, timeout=10.0)
            except Exception:  # noqa: BLE001
                # A node that vanished has already forgotten its cache; a node that
                # is merely slow will expire the session itself. Never fail a
                # completed generation over cleanup.
                pass

    async def forward(self, data, is_tokens: bool, pos_offset: int,
                      session: Optional[str]) -> np.ndarray:
        """One hop through this stage, surviving the loss of its node.

        Retries on the next candidate rather than failing the generation. The
        hydration that precedes the retry is what makes the retry *useful*: a cold
        spare would refuse the resume with a cache-state error, so without hydration
        the failover would swap one loud failure for another.
        """
        if session is not None and not self._hydrating:
            self._sessions.add(session)
        while True:
            await self._ensure_hydrated([session], [pos_offset])
            try:
                return await self._forward_once(data, is_tokens, pos_offset, session)
            except Exception as exc:  # noqa: BLE001
                if not self._is_node_failure(exc) or not self._rotate():
                    if self._is_node_failure(exc):
                        raise StageUnavailable(self, exc) from exc
                    raise

    async def _forward_once(self, data, is_tokens: bool, pos_offset: int,
                            session: Optional[str]) -> np.ndarray:
        payload: dict[str, Any] = {
            "model_id": self.model_id,
            "data": data if is_tokens else np.asarray(data, np.float32).tolist(),
            "is_tokens": is_tokens,
        }
        if session is not None:
            payload["session"] = session
            payload["pos_offset"] = int(pos_offset)
            payload["use_cache"] = True
            trunc = self._truncate_to.pop(session, None)
            if trunc is not None:
                payload["truncate_to"] = int(trunc)
        try:
            body = await self.transport.call(self.node_id, "shard/forward", payload,
                                             timeout=self.timeout)
        except Exception:
            # Put the rollback back. If this attempt is retried on a spare the
            # truncation is discarded by _rotate (replay rebuilds the accepted state
            # directly); if it is retried on the SAME node it must still be applied,
            # and consuming it here would silently leave rejected draft positions in
            # that node's cache forever.
            if session is not None and "truncate_to" in payload:
                self._truncate_to.setdefault(session, int(payload["truncate_to"]))
            raise
        return np.asarray(body["data"], dtype=np.float32)

    async def forward_batch(self, calls: list[StageCall]) -> list[np.ndarray]:
        """Run several sessions through this stage in **one** request.

        The whole point of batching on a wide-area mesh is that B concurrent tenants
        share one set of hops instead of paying for B, so this must be a single round
        trip. Issuing one ``shard/forward`` per item would preserve correctness and
        destroy the entire benefit -- so if a node cannot batch, the fallback below is
        recorded as a fallback rather than passed off as batching.
        """
        if not calls:
            raise ValueError("forward_batch needs at least one call")
        for c in calls:
            if c.session is not None and not self._hydrating:
                self._sessions.add(c.session)
        while True:
            await self._ensure_hydrated([c.session for c in calls],
                                        [c.pos_offset for c in calls])
            try:
                return await self._forward_batch_once(calls)
            except Exception as exc:  # noqa: BLE001
                if not self._is_node_failure(exc) or not self._rotate():
                    if self._is_node_failure(exc):
                        raise StageUnavailable(self, exc) from exc
                    raise

    async def _forward_batch_once(self, calls: list[StageCall]) -> list[np.ndarray]:
        if self._batch_supported is False:
            return await self._forward_batch_serially(calls)

        is_tokens = calls[0].is_tokens
        if any(c.is_tokens != is_tokens for c in calls):
            raise ValueError("a batched hop cannot mix token and activation inputs")

        items: list[dict[str, Any]] = []
        flat: list[Any] = []
        for c in calls:
            entry: dict[str, Any] = {"rows": c.rows, "pos_offset": int(c.pos_offset)}
            if c.session is not None:
                entry["session"] = c.session
                entry["use_cache"] = True
                trunc = self._truncate_to.pop(c.session, None)
                if trunc is not None:
                    entry["truncate_to"] = int(trunc)
            items.append(entry)
            if is_tokens:
                flat.extend(int(t) for t in c.data)  # type: ignore[union-attr]
            else:
                flat.extend(np.asarray(c.data, np.float32).tolist())

        payload = {"model_id": self.model_id, "is_tokens": is_tokens,
                   "items": items, "data": flat}
        try:
            body = await self.transport.call(self.node_id, "shard/forward_batch",
                                             payload, timeout=self.timeout)
        except TransportError as exc:
            if exc.status not in _BATCH_UNSUPPORTED_STATUS:
                # A real failure. Put back the truncations this attempt consumed
                # before it propagates, for the same reason _forward_once does: a
                # retry against the same node must still apply them.
                self._restore_truncations(calls, items)
                raise
            # An older node, or meshd. Latch it so the next traversal does not pay
            # for a second discovery round trip, put back the truncations this
            # attempt consumed, and serve the batch the slow-but-correct way.
            self._batch_supported = False
            self._restore_truncations(calls, items)
            return await self._forward_batch_serially(calls)
        except Exception:
            self._restore_truncations(calls, items)
            raise

        self._batch_supported = True
        rows = [int(n) for n in body.get("rows", [c.rows for c in calls])]
        if len(rows) != len(calls):
            raise ValueError(
                f"node returned {len(rows)} row counts for {len(calls)} batch items")
        arr = np.asarray(body["data"], dtype=np.float32)
        if int(arr.shape[0]) != sum(rows):
            raise ValueError(
                f"node returned {arr.shape[0]} rows, expected {sum(rows)}")
        out, cursor = [], 0
        for n in rows:
            out.append(arr[cursor:cursor + n])
            cursor += n
        return out

    def _restore_truncations(self, calls: list[StageCall],
                             items: list[dict[str, Any]]) -> None:
        """Put back pending rollbacks that a failed attempt consumed."""
        for c, entry in zip(calls, items):
            if "truncate_to" in entry and c.session is not None:
                self._truncate_to.setdefault(c.session, int(entry["truncate_to"]))

    async def _forward_batch_serially(self, calls: list[StageCall]) -> list[np.ndarray]:
        """Correct fallback for a node that cannot batch. Costs one hop per item."""
        self.batch_fallbacks += 1
        return [await self.forward(c.data, c.is_tokens, c.pos_offset, c.session)
                for c in calls]


class RemotePipeline(Pipeline):
    """A :class:`Pipeline` whose stages are remote nodes.

    The synthetic :class:`LinkProfile` is disabled (``rtt_ms=0``): the delay here is
    the real network, and adding a modelled delay on top would double-count it. The
    stats it records — traversals, wire bytes, positions — remain exactly comparable
    to the sim, which is what makes measured and modelled runs directly relatable.
    """

    def __init__(self, cfg: M.ModelConfig, stages: list[RemoteShard]) -> None:
        super().__init__(cfg, stages, LinkProfile(rtt_ms=0.0, bandwidth_mbps=0.0))
        # Give every stage the ability to rebuild itself on a spare. The stage knows
        # *that* it failed over; only the pipeline holds the replay sources (the
        # activation cache and the token journal), so the two have to be joined here.
        for idx, stage in enumerate(stages):
            stage.rehydrate = self._make_rehydrator(idx)

    def _make_rehydrator(self, idx: int) -> Callable[[str, int], Awaitable[None]]:
        async def _rehydrate(session: str, upto_pos: int) -> None:
            await self.rehydrate_stage(session, idx, upto_pos)
        return _rehydrate

    @property
    def failovers(self) -> int:
        """Total stage failovers across this pipeline's lifetime."""
        return sum(s.failovers for s in self.stages)

    @property
    def hydrations(self) -> int:
        """Total session rebuilds performed onto spares."""
        return sum(s.hydrations for s in self.stages)

    async def flush(self) -> None:
        await asyncio.gather(*(s.flush() for s in self.stages), return_exceptions=True)


def build_remote_stages(transport: Any, model_id: str, partition,
                        timeout: float = 60.0,
                        spares: Optional[dict[str, Sequence[str]]] = None
                        ) -> list[RemoteShard]:
    """Build ordered remote stages from a deployment's partition.

    ``spares`` maps a primary node id to the standby nodes that hold the same layer
    range. A stage with no entry has no spare and behaves exactly as before -- so
    adding hot spares is opt-in per stage and never silently changes an existing
    deployment's failure semantics.

    Standby copies are addressed by a stage-qualified shard id (see
    :func:`spare_model_id`) because a standby may share a machine with a *different*
    stage of the same deployment.
    """
    assigns = sorted(partition, key=lambda a: a.start_layer)
    if not assigns:
        raise ValueError("deployment has no partition")
    return [
        RemoteShard(transport, a.node_id, model_id, a.start_layer, a.end_layer,
                    is_first=(i == 0), is_last=(i == len(assigns) - 1),
                    timeout=timeout,
                    spares=(spares or {}).get(a.node_id),
                    spare_model_id=spare_model_id(model_id, a.start_layer,
                                                  a.end_layer))
        for i, a in enumerate(assigns)
    ]
