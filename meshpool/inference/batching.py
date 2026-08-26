"""Continuous batching for a wide-area pipeline.

Why this exists
---------------
The governing cost of decoding across machines is

    t_token  ~=  sum(compute_i)  +  (N - 1) * RTT

and the ``(N - 1) * RTT`` term is paid **once per traversal**, no matter how many
rows ride inside it. :mod:`meshpool.inference.spd` exploits that across *tokens* —
verify K drafted tokens in one traversal. This module exploits the same fact across
*requests*: B people decoding at the same time cost B traversals per step when served
one at a time, and ONE traversal per step when batched together.

That distinction is the difference between a demo and a service. A single-tenant
pipeline that hits 20 tok/s serves exactly one person at 20 tok/s; batched, the same
pipeline serves eight people at close to 20 tok/s each, because the eight of them are
sharing hops rather than queueing for them. On a mesh where RTT dominates — which is
the entire premise of the project — added tenants are close to free until compute
becomes the binding constraint.

Why it is *continuous*
----------------------
Static batching collects B requests, runs them to completion together, and only then
accepts more. That makes every request in the batch wait for the slowest one: ask for
8 tokens next to someone asking for 512 and you wait for their 512. Continuous
batching instead re-forms the batch **every step** — finished sequences are retired
immediately and their slot is refilled from the queue on the very next traversal.
``verify_continuous_batching`` asserts exactly this, because it is the property that
users actually feel.

Exactness
---------
Every sequence is computed by the same per-session call the unbatched path makes (see
:meth:`meshpool.inference.pipeline.CachedShard.forward_batch` for why the rows are
deliberately *not* stacked into one matmul), and every sequence samples from its own
seeded generator. So a request's output does not depend on who else was in the batch
with it: batched output is token-identical to running that request alone. This is a
throughput optimisation with no behavioural cost, and the gate proves it rather than
asserting it.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..common.logging import get_logger
from .pipeline import (GREEDY_EPS, Pipeline, PipelineStats, StageCall,
                       dist_from_logits, sample_from_dist)
from .spd import DraftModel, SpecStats, _residual

log = get_logger("batching")


@dataclass
class BatchRequest:
    """One generation request handed to the batcher."""

    prompt: list[int]
    max_new_tokens: int = 32
    temperature: float = 0.8
    top_k: int = 40
    seed: int = 0
    session: Optional[str] = None
    chunked_prefill: int = 0
    # Speculative pipeline decoding, per request. The factory is called once per
    # sequence because a draft carries its own KV cache and must not be shared.
    # SPD and batching amortise the same ``(N-1) * RTT`` term along different axes --
    # SPD across tokens, batching across tenants -- so composing them multiplies:
    # B tenants each verifying K+1 candidates ride ONE traversal between them.
    draft_factory: Optional[Callable[[], DraftModel]] = None
    draft_k: int = 4
    # Called with each newly produced token id, as soon as it exists, so a caller can
    # stream. Real streaming has to originate *here*: the tokens become available one
    # traversal at a time, and anything that waits for the whole answer before
    # chunking it is not streaming -- it is a progress bar drawn after the fact, with
    # a time-to-first-token equal to the total generation time.
    on_tokens: Optional[Callable[[list[int]], None]] = None

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("prompt must contain at least one token")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.draft_factory is not None and self.draft_k < 1:
            raise ValueError("draft_k must be >= 1")


@dataclass
class BatchResult:
    """What one request produced, plus what it cost."""

    session: str
    tokens: list[int]
    queued_s: float
    ttft_s: float
    elapsed_s: float
    steps: int
    # Why generation stopped. "length" = the caller's max_new_tokens was reached and
    # the answer may well continue; "context_window" = the model's positional limit
    # was hit and the answer was CUT OFF. Reporting these as one thing is how an API
    # tells a client "this reply is complete" about a truncated reply, so the reason
    # is produced here, where it is known, and never re-derived downstream from a
    # token count -- a count cannot distinguish the two.
    finish_reason: str = "length"
    spec: Optional[dict] = None
    stats: PipelineStats = field(default_factory=PipelineStats)
    mean_batch: float = 1.0

    def as_dict(self) -> dict:
        d = {
            "session": self.session,
            "tokens": len(self.tokens),
            "queued_s": round(self.queued_s, 4),
            "ttft_s": round(self.ttft_s, 4),
            "elapsed_s": round(self.elapsed_s, 4),
            "steps": self.steps,
            "finish_reason": self.finish_reason,
            "mean_batch": round(self.mean_batch, 3),
        }
        if self.spec is not None:
            d["speculative"] = self.spec
        return d


@dataclass
class BatcherStats:
    """Scheduler-level accounting.

    ``rows_per_traversal`` is the headline: it is the average number of sequences that
    shared each hop, and therefore the factor by which the wide-area RTT term was
    amortised. A batching implementation that quietly degraded to serial execution
    would report 1.0 here, which is why the gate asserts on it.
    """

    steps: int = 0
    admitted: int = 0
    completed: int = 0
    abandoned: int = 0
    rejected: int = 0
    rows_processed: int = 0
    max_batch_seen: int = 0
    batch_size_sum: int = 0

    @property
    def mean_batch(self) -> float:
        return (self.batch_size_sum / self.steps) if self.steps else 0.0

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "admitted": self.admitted,
            "completed": self.completed,
            "abandoned": self.abandoned,
            "rejected": self.rejected,
            "rows_processed": self.rows_processed,
            "max_batch_seen": self.max_batch_seen,
            "mean_batch": round(self.mean_batch, 3),
        }


@dataclass
class _Seq:
    """Live state for one in-flight sequence.

    Mirrors :func:`meshpool.inference.pipeline.generate_baseline` exactly: sample from
    the logits produced by the previous traversal, append, then feed the sampled token
    back in at the next position. Keeping the order identical is what makes batched
    output token-identical to unbatched output.

    When a draft is supplied the sequence instead mirrors
    :func:`meshpool.inference.spd.speculative_generate`, contributing ``len(pending) +
    K`` rows to the shared traversal rather than one.
    """

    req: BatchRequest
    session: str
    rng: np.random.Generator
    submitted_at: float
    future: "asyncio.Future[BatchResult]"
    started_at: float = 0.0
    ttft_at: float = 0.0
    pos: int = 0
    prefill_cursor: int = 0
    logits: Optional[np.ndarray] = None
    out: list[int] = field(default_factory=list)
    steps: int = 0
    finish: str = ""
    stats: PipelineStats = field(default_factory=PipelineStats)
    co_riders: int = 0
    # -- speculative state (unused when req.draft_factory is None) ------------ #
    draft: Optional[DraftModel] = None
    spec: SpecStats = field(default_factory=SpecStats)
    ctx: list[int] = field(default_factory=list)
    pending: list[int] = field(default_factory=list)
    cached: int = 0
    prefilled: bool = False
    drafts: list[int] = field(default_factory=list)
    q_dists: list[np.ndarray] = field(default_factory=list)
    n_pending: int = 0
    base: int = 0

    @property
    def speculative(self) -> bool:
        return self.draft is not None

    @property
    def prefilling(self) -> bool:
        return self.logits is None


class ContinuousBatcher:
    """Serve many concurrent generations over one shared pipeline.

    Usage::

        batcher = ContinuousBatcher(pipe, max_batch=8)
        await batcher.start()
        result = await batcher.submit(BatchRequest(prompt=[1, 2, 3]))
        await batcher.aclose()

    The loop admits, steps and retires; callers just await their own future.
    """

    def __init__(self, pipe: Pipeline, *, max_batch: int = 8,
                 idle_sleep_s: float = 0.001) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        self.pipe = pipe
        self.max_batch = max_batch
        self.idle_sleep_s = idle_sleep_s
        self.stats = BatcherStats()
        self._queue: list[_Seq] = []
        self._active: list[_Seq] = []
        self._admitting: Optional[_Seq] = None
        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        self._wake = asyncio.Event()
        self._seq_counter = 0

    # -- lifecycle ---------------------------------------------------------- #
    async def start(self) -> None:
        if self._task is None:
            self._stopping = False
            self._task = asyncio.create_task(self._loop())

    async def aclose(self) -> None:
        """Stop the loop once everything in flight has drained."""
        self._stopping = True
        self._wake.set()
        task, self._task = self._task, None
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:  # pragma: no cover - shutdown race
                pass
            except Exception:  # noqa: BLE001
                # _loop already logged this and handed it to every waiting caller.
                # Re-raising from aclose() would replace a caller's real error with
                # the same error thrown from a cleanup path.
                pass

    @property
    def in_flight(self) -> int:
        return len(self._active) + len(self._queue)

    @property
    def alive(self) -> bool:
        """True while the scheduler loop can still resolve a request."""
        return self._task is not None and not self._task.done()

    # -- submission --------------------------------------------------------- #
    def submit(self, req: BatchRequest) -> "asyncio.Future[BatchResult]":
        """Queue a request. The returned future resolves with its :class:`BatchResult`."""
        if self._stopping:
            raise RuntimeError("batcher is shutting down")
        task = self._task
        if task is not None and task.done():
            # Nothing will ever drain this queue again. Refusing is strictly better
            # than accepting a request that would wait forever.
            exc = task.exception() if not task.cancelled() else None
            raise RuntimeError(f"batcher loop is not running: {exc!r}")
        loop = asyncio.get_event_loop()
        fut: "asyncio.Future[BatchResult]" = loop.create_future()
        self._seq_counter += 1
        session = req.session or f"batch-{self._seq_counter}"
        seq = _Seq(req=req, session=session,
                   rng=np.random.default_rng(req.seed),
                   submitted_at=time.perf_counter(), future=fut)
        self._queue.append(seq)
        self._wake.set()
        return fut

    async def generate(self, req: BatchRequest) -> BatchResult:
        return await self.submit(req)

    # -- scheduler ---------------------------------------------------------- #
    async def _loop(self) -> None:
        try:
            await self._run_loop()
        except BaseException as exc:  # noqa: BLE001
            # The loop is the only thing that ever resolves a caller's future, so if
            # it dies every in-flight and queued request waits forever -- no error, no
            # timeout, just permanent silence on a shared deployment. Anything that
            # escapes the per-step handler must therefore take the whole batcher down
            # loudly and hand the exception to everyone who was waiting.
            log.exception("batcher loop terminated", error=str(exc))
            self._fail_admitting(exc)
            self._fail_active(exc)
            self._fail_queued(exc)
            raise

    async def _run_loop(self) -> None:
        while True:
            self._admit()
            if not self._active:
                if self._stopping and not self._queue:
                    return
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self.idle_sleep_s)
                except asyncio.TimeoutError:
                    pass
                continue
            try:
                await self._step()
            except Exception as exc:  # noqa: BLE001
                # One bad step must not wedge the scheduler or strand every other
                # tenant's future. Fail the whole active set loudly and carry on:
                # a traversal is shared, so a failure in it is genuinely shared too.
                self._fail_active(exc)
            # Yield between steps, unconditionally.
            #
            # A traversal only suspends if something inside it actually awaits, and
            # over a fast link nothing does. Without this the loop would run every
            # step of every sequence back to back without ever returning to the event
            # loop, so completed futures would not be delivered until the whole batch
            # drained, newly submitted requests could not even be enqueued, and any
            # HTTP handler sharing this process would stall behind generation. The
            # cost is one scheduler tick per step; the benefit is that "continuous"
            # is true rather than aspirational.
            await asyncio.sleep(0)

    def _admit(self) -> None:
        """Refill free slots from the queue.

        Called at the top of every step, which is precisely what makes the batching
        continuous: a slot freed by a sequence that finished on the previous step is
        reused on this one, instead of idling until the whole batch drains.
        """
        while self._queue and len(self._active) < self.max_batch:
            seq = self._queue.pop(0)
            if seq.future.cancelled():
                continue
            # A sequence that has left the queue but not yet joined the active set is
            # owned by neither, so if the loop dies here _fail_queued and _fail_active
            # both miss it and its caller waits forever. Hold it somewhere the failure
            # path can see, and clear it only on the paths that leave normally -- a
            # ``finally`` would clear it while the exception was still propagating,
            # which is exactly the case it exists for.
            self._admitting = seq
            self.pipe.drop_session(seq.session)
            if seq.req.draft_factory is not None:
                # A caller-supplied factory is arbitrary code. Letting it raise here
                # would escape into the scheduler and strand every other tenant, so
                # the blast radius is contained to the request that supplied it.
                try:
                    seq.draft = seq.req.draft_factory()
                    seq.draft.reset()
                except Exception as exc:  # noqa: BLE001
                    if not seq.future.done():
                        seq.future.set_exception(exc)
                    self.stats.rejected += 1
                    self._admitting = None
                    continue
            seq.started_at = time.perf_counter()
            self._active.append(seq)
            self.stats.admitted += 1
            self._admitting = None

    async def _step(self) -> None:
        """One shared traversal on behalf of every active sequence."""
        calls: list[StageCall] = []
        owners: list[_Seq] = []
        finished: list[_Seq] = []
        abandoned: list[_Seq] = []

        for seq in self._active:
            if seq.future.cancelled():
                abandoned.append(seq)
                continue
            call = self._plan(seq)
            if call is None:
                finished.append(seq)
                continue
            calls.append(call)
            owners.append(seq)

        for seq in abandoned:
            self._abandon(seq)
        for seq in finished:
            self._retire(seq)

        if not calls:
            return

        before = self.pipe.stats
        link0, compute0 = before.link_seconds, before.compute_seconds
        outs = await self.pipe.forward_batch(calls)
        link_before_share = self.pipe.stats.link_seconds - link0
        compute_before_share = self.pipe.stats.compute_seconds - compute0

        self.stats.steps += 1
        self.stats.batch_size_sum += len(calls)
        self.stats.rows_processed += sum(c.rows for c in calls)
        self.stats.max_batch_seen = max(self.stats.max_batch_seen, len(calls))

        # Attribute the traversal to each rider.
        #
        # A shared pipeline's own counters describe the whole batch, which is useless
        # to a single tenant asking "what did my request cost?". Traversals, hops,
        # stage calls, positions and wire bytes are attributable *exactly*: every
        # inter-stage payload is [rows, d_model] float32, so a sequence's share of the
        # wire is its own rows and nothing else. Time is genuinely shared, so it is
        # split pro rata by rows and labelled as such rather than pretended to be
        # exact. Reported this way, a solo request's numbers are identical to what the
        # unbatched path would have reported -- which is what keeps the acceptance
        # gates comparing like with like.
        depth = self.pipe.depth
        total_rows = max(1, sum(c.rows for c in calls))
        row_bytes = self.pipe.cfg.d_model * 4
        for seq, call in zip(owners, calls):
            share = call.rows / total_rows
            seq.stats.traversals += 1
            seq.stats.hops += depth - 1
            seq.stats.stage_calls += depth
            seq.stats.positions_processed += call.rows
            seq.stats.wire_bytes += call.rows * row_bytes * (depth - 1)
            seq.stats.link_seconds += link_before_share * share
            seq.stats.compute_seconds += compute_before_share * share
            seq.co_riders += len(calls)

        for seq, call, logits in zip(owners, calls, outs):
            seq.steps += 1
            self._absorb(seq, call, logits)

    def _plan(self, seq: _Seq) -> Optional[StageCall]:
        """Decide what this sequence contributes to the next traversal.

        Returns ``None`` when the sequence is finished. Sampling happens *here*,
        before the traversal, exactly as in the unbatched loop.
        """
        if seq.speculative:
            return self._plan_spd(seq)

        if seq.prefilling:
            chunk = seq.req.chunked_prefill
            prompt = seq.req.prompt
            if chunk <= 0 or chunk >= len(prompt):
                piece = prompt
            else:
                piece = prompt[seq.prefill_cursor:seq.prefill_cursor + chunk]
            return StageCall(seq.session, list(piece), True, seq.prefill_cursor)

        assert seq.logits is not None
        p = dist_from_logits(seq.logits[-1], seq.req.temperature, seq.req.top_k)
        nxt = sample_from_dist(p, seq.rng)
        seq.out.append(nxt)
        self._emit(seq, [nxt])
        if seq.ttft_at == 0.0:
            seq.ttft_at = time.perf_counter()
        if len(seq.out) >= seq.req.max_new_tokens:
            seq.finish = "length"
            return None
        if seq.pos + 1 > self.pipe.cfg.block_size:
            # Learned absolute position embeddings: sliding the window would
            # invalidate every cached key/value (see kvcache.py). Stopping is the
            # honest choice -- but the caller must be told, or a cut-off answer looks
            # like a complete one.
            seq.finish = "context_window"
            return None
        call = StageCall(seq.session, [nxt], True, seq.pos)
        seq.pos += 1
        return call

    def _absorb(self, seq: _Seq, call: StageCall, logits: np.ndarray) -> None:
        if seq.speculative:
            self._absorb_spd(seq, logits)
            return
        if seq.prefilling:
            prompt = seq.req.prompt
            seq.prefill_cursor += call.rows
            if seq.prefill_cursor >= len(prompt):
                seq.logits = logits
                seq.pos = len(prompt)
            # Intermediate prefill chunks discard their logits, matching
            # pipeline.prefill(): only the final chunk's logits are ever sampled.
            return
        seq.logits = logits

    # -- speculative pipeline decoding, batched ----------------------------- #
    #
    # This is the composition that makes multi-tenancy cheap on a mesh. SPD puts K+1
    # candidate tokens into one traversal; batching puts B tenants into one traversal.
    # Together, one traversal carries B * (K+1) rows, and the (N-1)*RTT term is
    # divided by the product rather than by either factor alone.
    #
    # The algorithm below is deliberately a line-for-line mirror of
    # spd.speculative_generate. Any divergence would make a batched request differ
    # from the same request run alone, which test_batching asserts cannot happen.

    def _plan_spd(self, seq: _Seq) -> Optional[StageCall]:
        cfg = self.pipe.cfg
        if not seq.prefilled:
            prompt = seq.req.prompt
            # Prefill all but the final prompt token: that token stays pending and
            # rides into the first verify traversal with the drafts, so no traversal
            # is ever spent on a single token.
            head = prompt[:-1]
            if head:
                chunk = seq.req.chunked_prefill
                piece = (head if chunk <= 0 or chunk >= len(head)
                         else head[seq.prefill_cursor:seq.prefill_cursor + chunk])
                return StageCall(seq.session, list(piece), True, seq.prefill_cursor)
            seq.prefilled = True
            seq.ctx = list(prompt)
            seq.cached = 0
            seq.pending = list(prompt)

        if len(seq.out) >= seq.req.max_new_tokens:
            seq.finish = "length"
            return None
        max_draft = cfg.block_size - (seq.cached + len(seq.pending))
        if max_draft < 0 or not seq.pending:
            seq.finish = "context_window"
            return None
        k_eff = min(seq.req.draft_k, max_draft, seq.req.max_new_tokens - len(seq.out))

        assert seq.draft is not None
        seq.drafts, seq.q_dists = seq.draft.propose(
            seq.ctx, seq.pending, k_eff, seq.req.temperature, seq.req.top_k, seq.rng)
        feed = seq.pending + seq.drafts
        seq.n_pending = len(seq.pending)
        seq.base = len(seq.ctx)
        return StageCall(seq.session, list(feed), True, seq.cached)

    def _absorb_spd(self, seq: _Seq, logits: np.ndarray) -> None:
        if not seq.prefilled:
            head = seq.req.prompt[:-1]
            chunk = seq.req.chunked_prefill
            step = len(head) if chunk <= 0 or chunk >= len(head) else chunk
            seq.prefill_cursor += step
            if seq.prefill_cursor >= len(head):
                seq.prefilled = True
                seq.ctx = list(seq.req.prompt)
                seq.cached = len(head)
                seq.pending = seq.ctx[seq.cached:]
            return

        req = seq.req
        n_draft = len(seq.drafts)
        m, base = seq.n_pending, seq.base
        seq.spec.rounds += 1
        seq.spec.verify_traversals += 1
        seq.spec.drafted += n_draft
        seq.cached += m + n_draft

        tgt = [dist_from_logits(logits[m - 1 + i], req.temperature, req.top_k)
               for i in range(n_draft + 1)]

        accepted = 0
        emitted: Optional[int] = None
        for i in range(n_draft):
            x = seq.drafts[i]
            p_i, q_i = tgt[i], seq.q_dists[i]
            px, qx = float(p_i[x]), float(q_i[x])
            if req.temperature <= GREEDY_EPS:
                ok = px > 0.0
            else:
                ok = qx <= 0.0 or seq.rng.random() <= min(1.0, px / qx)
            if ok:
                accepted += 1
                continue
            emitted = sample_from_dist(_residual(p_i, q_i), seq.rng)
            break
        if emitted is None:
            emitted = sample_from_dist(tgt[n_draft], seq.rng)

        seq.spec.accepted += accepted
        seq.spec.accept_run_hist[accepted] = seq.spec.accept_run_hist.get(accepted, 0) + 1

        new_tokens = (seq.drafts[:accepted] + [emitted])[:req.max_new_tokens - len(seq.out)]
        seq.out.extend(new_tokens)
        self._emit(seq, new_tokens)
        seq.spec.emitted += len(new_tokens)
        if seq.ttft_at == 0.0 and new_tokens:
            seq.ttft_at = time.perf_counter()

        retained = min(accepted, len(new_tokens))
        valid = base + retained
        # Rejected drafts must leave no trace in any stage's KV cache. On a remote
        # pipeline the rollback is only *recorded* here and rides along with this
        # sequence's slice of the next batched traversal, so it costs no round trip.
        self.pipe.truncate_session(seq.session, valid)
        seq.cached = valid
        seq.ctx = seq.ctx[:base] + new_tokens
        seq.pending = seq.ctx[seq.cached:]

    def _emit(self, seq: _Seq, tokens: list[int]) -> None:
        """Hand freshly produced tokens to a streaming caller, if there is one.

        Deliberately swallowing: this batcher is shared by every tenant of the
        deployment, so one consumer whose callback raises -- a client that hung up
        mid-stream is the ordinary case -- must not be able to take down the loop
        that is serving everyone else. The tokens are still recorded in ``seq.out``,
        so the final result is unaffected either way.
        """
        cb = seq.req.on_tokens
        if cb is None or not tokens:
            return
        try:
            cb(list(tokens))
        except Exception:  # noqa: BLE001
            seq.req.on_tokens = None

    def _abandon(self, seq: _Seq) -> None:
        """Drop a sequence whose caller gave up waiting for it.

        Streaming makes this the common case rather than an edge case: a browser tab
        closing mid-answer cancels the awaiting task, and without this the sequence
        would keep riding every traversal until it hit ``max_new_tokens``, occupying
        a batch slot and spending wide-area round trips on output nobody will read.
        ``_admit`` already skips cancelled requests still in the queue; this is the
        same check for ones already running.
        """
        self._active.remove(seq)
        self.pipe.drop_session(seq.session)
        self.stats.abandoned += 1

    def _retire(self, seq: _Seq) -> None:
        now = time.perf_counter()
        self._active.remove(seq)
        self.pipe.drop_session(seq.session)
        self.stats.completed += 1
        if not seq.future.done():
            seq.future.set_result(BatchResult(
                session=seq.session,
                tokens=list(seq.out),
                queued_s=max(0.0, seq.started_at - seq.submitted_at),
                ttft_s=max(0.0, (seq.ttft_at or now) - seq.started_at),
                elapsed_s=max(0.0, now - seq.started_at),
                steps=seq.steps,
                finish_reason=seq.finish or "length",
                spec=seq.spec.as_dict() if seq.speculative else None,
                stats=seq.stats,
                mean_batch=(seq.co_riders / seq.steps) if seq.steps else 1.0,
            ))

    def _fail_active(self, exc: BaseException) -> None:
        for seq in list(self._active):
            self._active.remove(seq)
            self.pipe.drop_session(seq.session)
            if not seq.future.done():
                seq.future.set_exception(exc)

    def _fail_admitting(self, exc: BaseException) -> None:
        """Resolve the sequence caught mid-admission, if there is one.

        _admit pops from the queue before appending to the active set. A failure in
        that window leaves the sequence referenced by neither list, so without this
        its caller is the one tenant the loud-failure path silently misses -- and it
        is precisely the tenant whose request caused the failure.
        """
        seq = self._admitting
        self._admitting = None
        if seq is not None and not seq.future.done():
            seq.future.set_exception(exc)

    def _fail_queued(self, exc: BaseException) -> None:
        for seq in list(self._queue):
            self._queue.remove(seq)
            if not seq.future.done():
                seq.future.set_exception(exc)


async def batched_generate(pipe: Pipeline, requests: list[BatchRequest], *,
                           max_batch: int = 8
                           ) -> tuple[list[BatchResult], BatcherStats, PipelineStats]:
    """Run ``requests`` concurrently over one pipeline and return everything.

    A convenience wrapper for callers that already hold the whole request set — the
    benchmark harness and the acceptance gate both use it. Live serving uses
    :class:`ContinuousBatcher` directly so requests can arrive at any time.
    """
    pipe.reset_stats()
    batcher = ContinuousBatcher(pipe, max_batch=max_batch)
    await batcher.start()
    try:
        futures = [batcher.submit(r) for r in requests]
        results = await asyncio.gather(*futures)
    finally:
        await batcher.aclose()
    return list(results), batcher.stats, pipe.stats
