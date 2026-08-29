"""Pipeline-parallel execution with explicit traversal accounting.

A "traversal" is one end-to-end pass through every stage of the pipeline. It is the
unit that actually costs money on a wide-area mesh, because the governing cost of
distributed decoding is

    t_token  ~=  sum(compute_i)  +  (N - 1) * RTT

and the ``(N - 1) * RTT`` term is paid **once per traversal**, no matter how many
token positions ride along inside it. Everything interesting about making
decentralised inference fast comes down to moving work *into* a traversal so that
fewer traversals are needed:

* **Chunked prefill** puts ``T`` prompt positions into one traversal instead of ``T``.
* **Speculative pipeline decoding** (see :mod:`meshpool.inference.spd`) puts ``K+1``
  candidate tokens into one traversal instead of one.

So this module deliberately counts traversals, hops and wire bytes as first-class
telemetry rather than treating them as an implementation detail. A speedup claim that
is not backed by a drop in traversal count is not a real speedup.

The pipeline here runs stages in-process. That is on purpose: it isolates the
*algorithm* from the transport so correctness can be proven exactly, and it lets a
synthetic link delay be injected to model a wide-area link on a single machine.
The distributed transport path (``inference/remote.py``) implements the same
protocol against real nodes over HTTP.
"""
from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from . import kvcache as KV
from . import model as M


@dataclass
class LinkProfile:
    """Synthetic wide-area link between two consecutive stages.

    Defaults describe a same-metro residential link. See ``sim/topology.py`` for the
    full topology model; this is the minimal slice the inference engine needs so it
    can be benchmarked without booting a cluster.
    """

    rtt_ms: float = 0.0
    bandwidth_mbps: float = 0.0   # 0 => unmodelled / infinite
    time_scale: float = 1.0       # 0.1 runs 10x faster, preserving ratios

    def transit_s(self, nbytes: int) -> float:
        # One hop costs half a round trip plus serialisation of the payload.
        secs = (self.rtt_ms / 2000.0)
        if self.bandwidth_mbps > 0:
            secs += (nbytes * 8.0) / (self.bandwidth_mbps * 1e6)
        return secs * self.time_scale


@dataclass
class PipelineStats:
    """What a run actually cost. ``traversals`` is the headline number."""

    traversals: int = 0
    hops: int = 0
    stage_calls: int = 0
    hydration_stage_calls: int = 0
    hydration_bytes: int = 0
    wire_bytes: int = 0
    link_seconds: float = 0.0
    compute_seconds: float = 0.0
    positions_processed: int = 0
    layer_positions_processed: int = 0
    hydration_positions: int = 0

    def merge(self, other: "PipelineStats") -> None:
        self.traversals += other.traversals
        self.hops += other.hops
        self.stage_calls += other.stage_calls
        self.hydration_stage_calls += other.hydration_stage_calls
        self.hydration_bytes += other.hydration_bytes
        self.wire_bytes += other.wire_bytes
        self.link_seconds += other.link_seconds
        self.compute_seconds += other.compute_seconds
        self.positions_processed += other.positions_processed
        self.layer_positions_processed += other.layer_positions_processed
        self.hydration_positions += other.hydration_positions

    def as_dict(self) -> dict:
        return {
            "traversals": self.traversals,
            "hops": self.hops,
            "stage_calls": self.stage_calls,
            "hydration_stage_calls": self.hydration_stage_calls,
            "hydration_bytes": self.hydration_bytes,
            "wire_bytes": self.wire_bytes,
            "link_seconds": round(self.link_seconds, 6),
            "compute_seconds": round(self.compute_seconds, 6),
            "positions_processed": self.positions_processed,
            "layer_positions_processed": self.layer_positions_processed,
            "hydration_positions": self.hydration_positions,
        }


@dataclass
class StageCall:
    """One session's slice of a batched traversal.

    A batched traversal carries several of these at once. They are independent —
    different sessions, different positions, possibly different row counts (a prefill
    riding alongside several one-row decodes) — and every stage keeps them apart by
    routing each to its own KV cache.
    """

    session: Optional[str]
    data: object
    is_tokens: bool
    pos_offset: int

    @property
    def rows(self) -> int:
        if self.is_tokens:
            return len(self.data)  # type: ignore[arg-type]
        return int(np.asarray(self.data, dtype=np.float32).shape[0])


def _stage_layer_span(stage: object) -> int:
    """How many transformer layers a stage owns, for work accounting.

    Both :class:`CachedShard` and :class:`~meshpool.inference.remote.RemoteShard`
    expose ``start``/``end``, but a stage may legitimately be a thin proxy that
    wraps one, so the wrapped object is consulted too. A stats counter must never
    be able to abort a generation, so an unrecognisable stage contributes 0 rather
    than raising.
    """
    for obj in (stage, getattr(stage, "inner", None)):
        start, end = getattr(obj, "start", None), getattr(obj, "end", None)
        if isinstance(start, int) and isinstance(end, int):
            return max(0, end - start)
    return 0


class CachedShard:
    """One pipeline stage: a contiguous block of layers plus a per-session KV cache.

    Holds only :func:`model.slice_params` for its own layer range, so no stage ever
    materialises the full checkpoint — the property that lets a model larger than any
    single machine be served by the pool.
    """

    def __init__(self, cfg: M.ModelConfig, start: int, end: int,
                 is_first: bool, is_last: bool,
                 params: Optional[dict] = None,
                 cache_page_size: int = 16) -> None:
        self.cfg = cfg
        self.start, self.end = start, end
        self.is_first, self.is_last = is_first, is_last
        self.cache_page_size = cache_page_size
        full = params if params is not None else M.load_params(cfg)[0]
        self.P = M.slice_params(full, start, end, is_first, is_last)
        self._caches: dict[str, KV.KVCache] = {}
        # Work actually performed by this stage. With a correct KV cache these grow
        # by one position per decode step, not by the whole sequence length --
        # which is the observable difference between caching and recomputing.
        self.positions_computed = 0
        self.layer_positions_computed = 0
        self.forward_calls = 0

    def cache(self, session: str) -> KV.KVCache:
        c = self._caches.get(session)
        if c is None:
            c = self._caches[session] = KV.KVCache(self.start, self.end, page_size=self.cache_page_size)
        return c

    def drop(self, session: str) -> None:
        self._caches.pop(session, None)

    def truncate(self, session: str, length: int) -> None:
        c = self._caches.get(session)
        if c is not None:
            c.truncate(length)

    def cache_bytes(self) -> int:
        return sum(c.nbytes() for c in self._caches.values())

    def forward(self, data, is_tokens: bool, pos_offset: int,
                session: Optional[str]) -> np.ndarray:
        """Run this stage's layers.

        ``session`` selects the KV cache; ``None`` disables caching entirely, which is
        the exact-recompute path used as the correctness oracle.
        """
        self.forward_calls += 1
        cache = self.cache(session) if (session is not None and self.start < self.end) else None
        if cache is not None:
            # Recorded BEFORE the rebuild branch below mutates the cache: once
            # reset() has run, length is 0 and a rebuild would be indistinguishable
            # from a first-ever prefill.
            cache.record_access(pos_offset)
        if cache is not None and not cache.accepts(0, pos_offset):
            # The caller jumped to a position that does not continue this cache
            # (window slide, or a fresh sequence reusing the id). Rebuilding is the
            # only correct response — see kvcache module docstring.
            if pos_offset != cache.length:
                cache.reset()
                if pos_offset != 0:
                    raise ValueError(
                        f"cache miss for session {session!r}: cannot resume at "
                        f"position {pos_offset} with an empty cache")

        if is_tokens:
            if not self.is_first:
                raise ValueError("only the first stage may be handed raw tokens")
            x = KV.embed_tokens_at(self.P, self.cfg, list(data), pos_offset)
        else:
            x = np.asarray(data, dtype=np.float32)

        x = KV.run_layers_cached(self.P, self.cfg, x, self.start, self.end, cache)
        rows = int(np.asarray(x).shape[0])
        self.positions_computed += rows
        self.layer_positions_computed += rows * (self.end - self.start)
        return M.head(self.P, self.cfg, x) if self.is_last else x

    def forward_batch(self, items: list["StageCall"]) -> list[np.ndarray]:
        """Run several independent sessions through this stage in one call.

        Each item is computed **separately**, against its own KV cache, by the very
        same :meth:`forward` the unbatched path uses. That is deliberate. The
        tempting alternative — stacking every item's rows into one big matmul — would
        change the floating-point reduction order, so a request's output would depend
        on who else happened to be in the batch with it. Batched output would then
        differ from unbatched output, and ``verify_continuous_batching`` could not
        assert token-identity. Reproducibility is worth more here than the matmul.

        The saving batching actually delivers is therefore **not** compute: it is the
        ``(N-1) * RTT`` term. B concurrent requests cost B traversals when run
        separately and ONE when batched, because all B rides share the same hops.
        On a wide-area mesh that term dominates, which is exactly why this is worth
        doing — and it is the same trick speculative decoding plays across tokens,
        played here across requests instead.
        """
        return [self.forward(it.data, it.is_tokens, it.pos_offset, it.session)
                for it in items]


def build_stages(cfg: M.ModelConfig, n_stages: int,
                 params: Optional[dict] = None,
                 cache_page_size: int = 16) -> list[CachedShard]:
    """Split ``cfg.n_layer`` layers as evenly as possible across ``n_stages``."""
    if n_stages < 1:
        raise ValueError("n_stages must be >= 1")
    if n_stages > cfg.n_layer:
        raise ValueError(f"cannot split {cfg.n_layer} layers across {n_stages} stages")
    full = params if params is not None else M.load_params(cfg)[0]
    base, extra = divmod(cfg.n_layer, n_stages)
    shards, cursor = [], 0
    for s in range(n_stages):
        width = base + (1 if s < extra else 0)
        start, end = cursor, cursor + width
        shards.append(CachedShard(cfg, start, end, s == 0, s == n_stages - 1, full,
                                  cache_page_size=cache_page_size))
        cursor = end
    return shards


class Pipeline:
    """Ordered stages, a synthetic link between them, and honest accounting."""

    def __init__(self, cfg: M.ModelConfig, stages: list[CachedShard],
                 link: Optional[LinkProfile] = None) -> None:
        self.cfg = cfg
        self.stages = stages
        self.link = link or LinkProfile()
        self.stats = PipelineStats()
        self.activation_cache = KV.ActivationCache()
        # Per-session token history, indexed by absolute position.
        #
        # ActivationCache deliberately does not store boundary 0 (raw tokens), which
        # left stage 0 as the one stage no client could rebuild: every other stage is
        # recoverable by replaying the boundary activations the client already holds,
        # but stage 0's input is tokens and nobody kept them. That asymmetry meant a
        # mesh could survive the loss of any stage EXCEPT the first, and the first is
        # no less likely to churn than the rest. Tokens are ints and a whole 8k
        # context is ~64 KB, so journalling them costs nothing measurable and makes
        # recovery uniform across the pipeline.
        self.token_journal: dict[str, list[int]] = {}

    @property
    def depth(self) -> int:
        return len(self.stages)

    def reset_stats(self) -> None:
        self.stats = PipelineStats()

    def drop_session(self, session: str) -> None:
        for s in self.stages:
            s.drop(session)
        self.activation_cache.drop(session)
        self.token_journal.pop(session, None)

    def truncate_session(self, session: str, length: int) -> None:
        """Roll every stage's cache back to ``length`` positions.

        This is what makes speculation safe: the verify traversal writes K+1
        positions into every stage's cache, and rejected drafts must vanish.
        """
        for s in self.stages:
            s.truncate(session, length)
        self.activation_cache.truncate(session, length)
        journal = self.token_journal.get(session)
        if journal is not None and length < len(journal):
            # Rejected draft tokens must vanish from the journal too, or a later
            # stage-0 rebuild would replay tokens the model never actually accepted
            # and silently reconstruct a different conversation.
            del journal[length:]

    def _journal_tokens(self, session: str, pos_offset: int,
                        tokens: Sequence[int]) -> None:
        """Append ``tokens`` to ``session``'s history at ``pos_offset``.

        Mirrors ActivationCache.record's append discipline exactly: writing at 0
        restarts the session, writing at the current length appends, and anything
        else is a positional inconsistency that must be loud rather than silently
        producing a journal that does not correspond to the model's real history.
        """
        journal = self.token_journal.get(session)
        if pos_offset == 0:
            self.token_journal[session] = [int(t) for t in tokens]
            return
        if journal is None or pos_offset != len(journal):
            raise ValueError(
                f"token journal for {session!r} cannot append at {pos_offset}; "
                f"current length is {0 if journal is None else len(journal)}")
        journal.extend(int(t) for t in tokens)

    def cache_bytes(self) -> int:
        return sum(s.cache_bytes() for s in self.stages)

    async def hydrate_stage_from_activations(
        self,
        session: str,
        stage_index: int,
        upto_pos: int,
        chunk: int = 0,
    ) -> None:
        """Rebuild one stage's KV cache from client-side boundary activations.

        This is the Petals-style recovery primitive. If stage ``j`` loses its
        server-side KV cache, the client replays the cached activations emitted by
        stage ``j-1`` through only stage ``j``. That is O(T) for one stage instead of
        restarting the full multi-stage pipeline from tokens.

        The replay must be awaited. A remote stage's ``forward`` is a coroutine, and
        dropping it on the floor here would be worse than doing nothing: the cache has
        already been cleared one line above, so the stage would be left EMPTY while
        the caller believes it was rebuilt, and the next decode step would compute
        against a missing cache at a non-zero offset. In-process stages are
        synchronous and hid this completely.
        """
        if not 0 < stage_index < len(self.stages):
            raise ValueError("activation hydration needs a non-first stage index")
        stage = self.stages[stage_index]
        stage.drop(session)
        chunk = upto_pos if chunk <= 0 else chunk
        for pos in range(0, upto_pos, chunk):
            act = self.activation_cache.get(session, stage_index, pos, min(upto_pos, pos + chunk))
            out = stage.forward(act, is_tokens=False, pos_offset=pos, session=session)
            if inspect.isawaitable(out):
                await out
            # Recovery work is counted on its own axis. Folding it into stage_calls
            # or traversals would let a churn-heavy run look like a more expensive
            # generation, and would corrupt the tok/traversal numbers the
            # speculative-decoding gates assert on.
            self.stats.hydration_stage_calls += 1
            self.stats.hydration_positions += min(upto_pos, pos + chunk) - pos
            self.stats.hydration_bytes += int(np.asarray(act, dtype=np.float32).nbytes)

    async def hydrate_stage_from_tokens(
        self,
        session: str,
        upto_pos: int,
        chunk: int = 0,
    ) -> None:
        """Rebuild **stage 0**'s KV cache by replaying the journalled tokens.

        The counterpart of :meth:`hydrate_stage_from_activations` for the one stage
        it cannot serve. Same shape, same cost class -- O(T) through a single stage
        rather than a restart of the whole generation -- so losing the first node of
        a pipeline is no more expensive to recover from than losing any other.
        """
        if not self.stages:
            raise ValueError("pipeline has no stages")
        journal = self.token_journal.get(session)
        if journal is None:
            raise ValueError(
                f"no token journal for session {session!r}: stage 0 cannot be "
                f"rebuilt without the tokens that produced its cache")
        if upto_pos > len(journal):
            raise ValueError(
                f"cannot rebuild stage 0 to position {upto_pos}; the journal holds "
                f"only {len(journal)} tokens")
        stage = self.stages[0]
        stage.drop(session)
        chunk = upto_pos if chunk <= 0 else chunk
        for pos in range(0, upto_pos, chunk):
            span = journal[pos:min(upto_pos, pos + chunk)]
            out = stage.forward(list(span), is_tokens=True, pos_offset=pos,
                                session=session)
            if inspect.isawaitable(out):
                await out
            self.stats.hydration_stage_calls += 1
            self.stats.hydration_positions += len(span)
            # Measured the same way as the activation path rather than assumed.
            # Token replay ships int32 ids, not [T, D] activations -- which is
            # precisely why stage 0 is the cheapest stage to rebuild.
            self.stats.hydration_bytes += int(
                np.asarray(list(span), dtype=np.int32).nbytes)

    async def rehydrate_stage(
        self,
        session: str,
        stage_index: int,
        upto_pos: int,
        chunk: int = 0,
    ) -> None:
        """Rebuild any stage's cache, choosing the right replay source for it.

        Callers recovering from churn should use this rather than picking between
        the two primitives themselves: which source is correct is a property of the
        stage's position, not of the caller's situation.
        """
        if stage_index == 0:
            await self.hydrate_stage_from_tokens(session, upto_pos, chunk)
        else:
            await self.hydrate_stage_from_activations(
                session, stage_index, upto_pos, chunk)

    async def forward(self, tokens: list[int], pos_offset: int = 0,
                      session: Optional[str] = None) -> np.ndarray:
        """One traversal. Returns logits ``[len(tokens), vocab]`` for the new positions.

        Every position in ``tokens`` rides along inside this single traversal, so the
        ``(N-1) * RTT`` cost is paid once regardless of how many there are. That is
        the entire basis of both chunked prefill and speculative pipeline decoding.
        """
        st = PipelineStats(traversals=1, positions_processed=len(tokens))
        data: object = list(tokens)
        is_tokens = True
        if session is not None:
            self._journal_tokens(session, pos_offset, tokens)

        for idx, stage in enumerate(self.stages):
            if idx > 0:
                nbytes = int(np.asarray(data, dtype=np.float32).nbytes)
                delay = self.link.transit_s(nbytes)
                st.hops += 1
                st.wire_bytes += nbytes
                st.link_seconds += delay
                if delay > 0:
                    await asyncio.sleep(delay)

            t0 = time.perf_counter()
            data = stage.forward(data, is_tokens, pos_offset, session)
            if inspect.isawaitable(data):
                # A remote stage (see meshpool.inference.remote). The algorithms above
                # are identical either way; only the link underneath changes.
                data = await data
            if session is not None and idx < len(self.stages) - 1:
                self.activation_cache.record(session, idx + 1, pos_offset, np.asarray(data, dtype=np.float32))
            st.compute_seconds += time.perf_counter() - t0
            st.stage_calls += 1
            st.layer_positions_processed += len(tokens) * _stage_layer_span(stage)
            is_tokens = False

        self.stats.merge(st)
        return np.asarray(data, dtype=np.float32)

    async def forward_batch(self, calls: list[StageCall]) -> list[np.ndarray]:
        """One traversal carrying several sessions at once.

        Returns one logits array per input call, in the order supplied.

        This is the multi-tenant twin of :meth:`forward`. The wide-area cost of
        decoding is ``sum(compute_i) + (N-1) * RTT``, and the RTT term is paid once
        per *traversal*, not once per row. So B concurrent users decoding separately
        pay ``B * (N-1) * RTT`` while the same B users batched pay ``(N-1) * RTT``
        between them. On a mesh where RTT dominates, throughput therefore scales
        close to linearly in B until compute becomes the binding constraint — which
        is the whole reason a decentralised pipeline can serve more than one person
        at a time without falling over.

        Sessions must be distinct. Two rows sharing one KV cache inside a single
        traversal would append both to the same cache and silently interleave two
        conversations, so that is rejected rather than tolerated.
        """
        if not calls:
            raise ValueError("forward_batch needs at least one call")
        named = [c.session for c in calls if c.session is not None]
        if len(set(named)) != len(named):
            raise ValueError(
                "every batch item needs a distinct session: two rows sharing one KV "
                "cache in a single traversal would interleave their positions")

        st = PipelineStats(traversals=1, positions_processed=sum(c.rows for c in calls))
        current = list(calls)
        for c in calls:
            if c.session is not None and c.is_tokens:
                self._journal_tokens(c.session, c.pos_offset, c.data)  # type: ignore[arg-type]

        for idx, stage in enumerate(self.stages):
            if idx > 0:
                nbytes = sum(int(np.asarray(c.data, dtype=np.float32).nbytes)
                             for c in current)
                delay = self.link.transit_s(nbytes)
                st.hops += 1
                st.wire_bytes += nbytes
                # One delay for the whole batch, not one per item: the rows are
                # concatenated onto a single hop. This is the saving being claimed.
                st.link_seconds += delay
                if delay > 0:
                    await asyncio.sleep(delay)

            t0 = time.perf_counter()
            out = stage.forward_batch(current)
            if inspect.isawaitable(out):
                out = await out
            if len(out) != len(current):
                raise RuntimeError(
                    f"stage {idx} returned {len(out)} results for {len(current)} "
                    f"batch items")
            st.compute_seconds += time.perf_counter() - t0
            st.stage_calls += 1
            st.layer_positions_processed += (
                sum(c.rows for c in calls) * _stage_layer_span(stage))

            if idx < len(self.stages) - 1:
                for c, arr in zip(current, out):
                    if c.session is not None:
                        self.activation_cache.record(
                            c.session, idx + 1, c.pos_offset,
                            np.asarray(arr, dtype=np.float32))
            current = [StageCall(c.session, arr, False, c.pos_offset)
                       for c, arr in zip(current, out)]

        self.stats.merge(st)
        return [np.asarray(c.data, dtype=np.float32) for c in current]


# --------------------------------------------------------------------------- #
# Sampling helpers
#
# Speculative decoding is only *provably* equivalent when the draft and the target
# are compared under the same distribution transform. These helpers materialise the
# full post-temperature, post-top-k probability vector so the rejection test operates
# on exactly the distribution the baseline sampler draws from.
# --------------------------------------------------------------------------- #
GREEDY_EPS = 1e-6


def dist_from_logits(logits: np.ndarray, temperature: float,
                     top_k: int = 40) -> np.ndarray:
    """Full-vocab probability vector for one position.

    Greedy (``temperature <= 0``) is represented as a one-hot distribution rather than
    special-cased, so the same rejection-sampling code path is exercised and greedy
    speculative decoding is provably identical to greedy baseline decoding.
    """
    logits = np.asarray(logits, dtype=np.float32)
    if temperature <= GREEDY_EPS:
        p = np.zeros(logits.shape[-1], dtype=np.float64)
        p[int(logits.argmax())] = 1.0
        return p
    z = (logits / temperature).astype(np.float64)
    if top_k and top_k < z.shape[-1]:
        keep = np.argpartition(z, -top_k)[-top_k:]
        masked = np.full_like(z, -np.inf)
        masked[keep] = z[keep]
        z = masked
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def sample_from_dist(p: np.ndarray, rng: np.random.Generator) -> int:
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(p))
    return int(rng.choice(p.shape[-1], p=p / total))


async def generate_baseline(pipe: Pipeline, prompt: list[int], max_new_tokens: int,
                            temperature: float = 0.8, top_k: int = 40, seed: int = 0,
                            session: str = "baseline",
                            chunked_prefill: int = 0) -> tuple[list[int], PipelineStats]:
    """Standard autoregressive decoding: one traversal per generated token.

    This is the reference implementation that speculative decoding must match
    distributionally, and match *exactly* at temperature 0.
    """
    rng = np.random.default_rng(seed)
    pipe.drop_session(session)
    pipe.reset_stats()

    logits = await prefill(pipe, prompt, session, chunk=chunked_prefill)
    pos = len(prompt)
    out: list[int] = []

    for _ in range(max_new_tokens):
        p = dist_from_logits(logits[-1], temperature, top_k)
        nxt = sample_from_dist(p, rng)
        out.append(nxt)
        if pos + 1 > pipe.cfg.block_size:
            break
        logits = await pipe.forward([nxt], pos_offset=pos, session=session)
        pos += 1

    return out, pipe.stats


async def prefill(pipe: Pipeline, prompt: list[int], session: str,
                  chunk: int = 0) -> np.ndarray:
    """Process the prompt, optionally in chunks.

    Prefill is **position-parallel**: a stage transforms all ``T`` positions at once
    and forwards a single ``[T, d_model]`` tensor to the next stage. It costs ONE
    traversal for the whole prompt, not one per prompt token — a distinction that is
    frequently and badly miscalculated. Chunking trades a few extra traversals for
    bounded peak memory and for the ability to interleave other work, so a long prompt
    cannot head-of-line block a shared pipeline.
    """
    if not prompt:
        raise ValueError("prompt must contain at least one token")
    if chunk <= 0 or chunk >= len(prompt):
        return await pipe.forward(prompt, pos_offset=0, session=session)

    logits = None
    for i in range(0, len(prompt), chunk):
        piece = prompt[i:i + chunk]
        logits = await pipe.forward(piece, pos_offset=i, session=session)
    return logits
