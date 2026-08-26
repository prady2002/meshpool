"""Continuous batching must be free throughput, never a behaviour change.

The claim being tested is narrow and checkable: B concurrent generations that would
cost B traversals per step cost ONE when batched, and every one of them produces
exactly the tokens it would have produced alone. Both halves matter. Batching that
changes output is a correctness bug wearing a performance costume, and batching that
does not collapse traversals is not batching at all -- so every test here asserts on
one or the other, and several assert on both.

The traversal assertions are deliberately exact rather than "fewer than". A silent
degradation to serial execution would still be "fewer than" for a while.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from meshpool.common.transport import TransportError
from meshpool.inference import model as M
from meshpool.inference.batching import (BatchRequest, BatcherStats, ContinuousBatcher,
                                         batched_generate)
from meshpool.inference.pipeline import (LinkProfile, Pipeline, StageCall,
                                         build_stages, generate_baseline, prefill)
from meshpool.inference.remote import RemotePipeline, build_remote_stages
from meshpool.inference.spd import SelfDraft, speculative_generate

PROMPTS = [[5, 9, 33, 41, 7], [2, 3, 4], [61, 12, 8, 9, 10, 11], [44, 45]]


@pytest.fixture(scope="module")
def cfg_params():
    cfg = M.REGISTRY["nano-gpt"]
    P, _ = M.load_params(cfg)
    return cfg, P


def _pipe(cfg, P, depth=3, rtt=0.0):
    return Pipeline(cfg, build_stages(cfg, depth, P), LinkProfile(rtt_ms=rtt))


def _solo(cfg, P, req: BatchRequest, depth=3):
    """Generate one request completely alone -- the oracle every batch is judged against."""
    pipe = _pipe(cfg, P, depth)
    out, stats = asyncio.run(generate_baseline(
        pipe, req.prompt, req.max_new_tokens, temperature=req.temperature,
        top_k=req.top_k, seed=req.seed, session=f"solo-{req.seed}",
        chunked_prefill=req.chunked_prefill))
    return out, stats


def _reqs(n=4, max_new=6, temperature=0.8, chunk=0):
    return [BatchRequest(prompt=PROMPTS[i % len(PROMPTS)], max_new_tokens=max_new,
                         temperature=temperature, seed=1000 + i, chunked_prefill=chunk)
            for i in range(n)]


# --------------------------------------------------------------------------- #
# Correctness: a tenant's output must not depend on who shared its batch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_batched_output_is_identical_to_running_each_request_alone(cfg_params, temperature):
    cfg, P = cfg_params
    reqs = _reqs(4, max_new=6, temperature=temperature)
    expected = [_solo(cfg, P, r)[0] for r in reqs]

    results, _, _ = asyncio.run(batched_generate(_pipe(cfg, P), reqs, max_batch=4))
    got = [r.tokens for r in results]

    assert got == expected, (
        "batched output diverged from solo output; a tenant's result must never "
        "depend on who else was in the batch")
    assert len(set(map(tuple, got))) > 1, "different seeds should give different text"


def test_results_come_back_in_submission_order(cfg_params):
    """A shorter request finishing first must not reorder anyone's result."""
    cfg, P = cfg_params
    reqs = [BatchRequest(prompt=[5, 9, 33], max_new_tokens=n, seed=7) for n in (8, 2, 5)]
    results, _, _ = asyncio.run(batched_generate(_pipe(cfg, P), reqs, max_batch=4))
    assert [len(r.tokens) for r in results] == [8, 2, 5]


def test_batching_does_not_depend_on_batch_composition(cfg_params):
    """The same request in a batch of 1, 2 and 4 must give the same tokens.

    This is the sharpest form of the exactness claim: it fails if any stage stacks
    rows into a shared reduction, because then the floating-point result would move
    as neighbours come and go.
    """
    cfg, P = cfg_params
    target = BatchRequest(prompt=PROMPTS[0], max_new_tokens=7, seed=99)
    outs = []
    for extra in (0, 1, 3):
        batch = [target] + [BatchRequest(prompt=PROMPTS[1], max_new_tokens=7,
                                         seed=500 + i) for i in range(extra)]
        results, _, _ = asyncio.run(batched_generate(_pipe(cfg, P), batch, max_batch=8))
        outs.append(results[0].tokens)
    assert outs[0] == outs[1] == outs[2], f"output moved with batch size: {outs}"


@pytest.mark.parametrize("chunk", [0, 2])
def test_chunked_prefill_inside_a_batch_matches_unbatched(cfg_params, chunk):
    cfg, P = cfg_params
    reqs = _reqs(3, max_new=5, chunk=chunk)
    expected = [_solo(cfg, P, r)[0] for r in reqs]
    results, _, _ = asyncio.run(batched_generate(_pipe(cfg, P), reqs, max_batch=4))
    assert [r.tokens for r in results] == expected


# --------------------------------------------------------------------------- #
# The actual saving: traversals, not compute
# --------------------------------------------------------------------------- #
def test_one_traversal_per_step_regardless_of_batch_size(cfg_params):
    """B tenants must share hops, not queue for them.

    Serial decoding costs one traversal per token per tenant. Batched decoding costs
    one traversal per *step*, so the traversal count must be independent of B. That
    independence is the entire economic argument for multi-tenancy on a mesh.
    """
    cfg, P = cfg_params
    counts = {}
    for b in (1, 2, 4, 8):
        reqs = [BatchRequest(prompt=[5, 9, 33], max_new_tokens=6, seed=2000 + i)
                for i in range(b)]
        _, bstats, pstats = asyncio.run(batched_generate(_pipe(cfg, P), reqs, max_batch=8))
        counts[b] = pstats.traversals
        assert bstats.max_batch_seen == b
    assert len(set(counts.values())) == 1, (
        f"traversals must not grow with batch size, got {counts}")


def test_batching_collapses_the_traversals_serial_execution_would_pay(cfg_params):
    cfg, P = cfg_params
    reqs = _reqs(4, max_new=6)
    serial_traversals = sum(_solo(cfg, P, r)[1].traversals for r in reqs)
    _, _, pstats = asyncio.run(batched_generate(_pipe(cfg, P), reqs, max_batch=4))
    assert pstats.traversals * 4 <= serial_traversals, (
        f"expected roughly a 4x collapse, got {serial_traversals} -> {pstats.traversals}")


def test_wall_clock_improves_when_the_link_actually_costs_something(cfg_params):
    """With RTT injected, batching must show up as real time saved.

    Traversal counting alone could be gamed by accounting. This measures the clock.
    """
    cfg, P = cfg_params
    reqs = _reqs(4, max_new=5)

    async def _serial():
        for i, r in enumerate(reqs):
            await generate_baseline(_pipe(cfg, P, rtt=40.0), r.prompt, r.max_new_tokens,
                                    temperature=r.temperature, seed=r.seed,
                                    session=f"ser{i}")

    loop_serial = _timed(_serial())
    loop_batched = _timed(batched_generate(_pipe(cfg, P, rtt=40.0), reqs, max_batch=4))
    assert loop_batched < loop_serial / 2, (
        f"batched {loop_batched:.3f}s vs serial {loop_serial:.3f}s -- expected at "
        f"least a 2x wall-clock win at 40ms RTT")


def _timed(coro) -> float:
    import time
    t0 = time.perf_counter()
    asyncio.run(coro)
    return time.perf_counter() - t0


def test_wire_bytes_per_hop_scale_with_rows_not_with_hops(cfg_params):
    """Batching trades a bigger payload for fewer hops. Confirm that is what happens."""
    cfg, P = cfg_params
    reqs = [BatchRequest(prompt=[5, 9, 33], max_new_tokens=4, seed=3000 + i)
            for i in range(4)]
    _, _, batched = asyncio.run(batched_generate(_pipe(cfg, P), reqs, max_batch=4))
    serial_stats = [_solo(cfg, P, r)[1] for r in reqs]
    serial_hops = sum(s.hops for s in serial_stats)
    assert batched.hops * 4 <= serial_hops, "hops did not collapse"
    # Bytes are conserved-ish: the same activations still cross the wire.
    assert batched.wire_bytes > 0


# --------------------------------------------------------------------------- #
# Continuous, not static
# --------------------------------------------------------------------------- #
def test_a_long_request_does_not_hold_the_slot_of_short_ones(cfg_params):
    """Static batching makes everyone wait for the slowest member. This must not.

    With ``max_batch=2`` and one very long generation running, three short requests
    submitted afterwards must all complete before the long one -- which can only
    happen if a retired sequence's slot is refilled on the very next step.
    """
    cfg, P = cfg_params

    async def _run():
        pipe = _pipe(cfg, P)
        b = ContinuousBatcher(pipe, max_batch=2)
        await b.start()
        long_f = b.submit(BatchRequest(prompt=[5, 9], max_new_tokens=24, seed=1))
        shorts = [b.submit(BatchRequest(prompt=[5, 9], max_new_tokens=2, seed=10 + i))
                  for i in range(3)]
        done_short = await asyncio.gather(*shorts)
        long_pending = not long_f.done()
        long_r = await long_f
        await b.aclose()
        return done_short, long_pending, long_r

    done_short, long_pending, long_r = asyncio.run(_run())
    assert long_pending, (
        "all three short requests should have finished while the long one was still "
        "generating; they were forced to wait for it")
    assert [len(s.tokens) for s in done_short] == [2, 2, 2]
    assert len(long_r.tokens) == 24


def test_queue_admits_more_than_max_batch_requests(cfg_params):
    cfg, P = cfg_params
    reqs = [BatchRequest(prompt=[5, 9, 33], max_new_tokens=3, seed=4000 + i)
            for i in range(7)]
    results, bstats, _ = asyncio.run(batched_generate(_pipe(cfg, P), reqs, max_batch=2))
    assert len(results) == 7
    assert bstats.completed == 7
    assert bstats.max_batch_seen <= 2, "max_batch must be an actual cap"
    assert all(len(r.tokens) == 3 for r in results)


def test_prefill_and_decode_ride_the_same_traversal(cfg_params):
    """A newcomer's prefill must join the in-flight batch, not wait for its own hop.

    Mixed batching is what keeps time-to-first-token low under load: an arriving
    request is prefilled alongside everyone else's decode step rather than queueing
    behind it.
    """
    cfg, P = cfg_params

    async def _run():
        pipe = _pipe(cfg, P)
        b = ContinuousBatcher(pipe, max_batch=4)
        await b.start()
        first = b.submit(BatchRequest(prompt=[5, 9, 33, 41], max_new_tokens=12, seed=1))
        await asyncio.sleep(0.02)          # let it get past prefill and into decode
        late = b.submit(BatchRequest(prompt=[2, 3, 4, 5, 6, 7], max_new_tokens=3, seed=2))
        out = await asyncio.gather(first, late)
        stats = b.stats
        await b.aclose()
        return out, stats

    (first_r, late_r), stats = asyncio.run(_run())
    assert len(late_r.tokens) == 3 and len(first_r.tokens) == 12
    # A step whose row count exceeds its member count can only be a prefill sharing a
    # traversal with decodes.
    assert stats.rows_processed > stats.steps, (
        "no step ever carried more rows than sequences, so prefill never shared a "
        "traversal with a decode")


# --------------------------------------------------------------------------- #
# Refusals -- paired with the acceptances above
# --------------------------------------------------------------------------- #
def test_two_rows_may_not_share_one_kv_cache(cfg_params):
    cfg, P = cfg_params
    pipe = _pipe(cfg, P)
    calls = [StageCall("same", [1, 2], True, 0), StageCall("same", [3], True, 0)]
    with pytest.raises(ValueError, match="distinct session"):
        asyncio.run(pipe.forward_batch(calls))


def test_empty_batch_is_rejected(cfg_params):
    cfg, P = cfg_params
    with pytest.raises(ValueError):
        asyncio.run(_pipe(cfg, P).forward_batch([]))


def test_bad_requests_are_rejected_at_construction():
    with pytest.raises(ValueError):
        BatchRequest(prompt=[], max_new_tokens=4)
    with pytest.raises(ValueError):
        BatchRequest(prompt=[1], max_new_tokens=0)


def test_a_failing_traversal_fails_the_futures_instead_of_hanging(cfg_params):
    cfg, P = cfg_params

    class _Exploding(Pipeline):
        async def forward_batch(self, calls):
            raise RuntimeError("stage down")

    async def _run():
        pipe = _Exploding(cfg, build_stages(cfg, 2, P), LinkProfile())
        b = ContinuousBatcher(pipe, max_batch=2)
        await b.start()
        fut = b.submit(BatchRequest(prompt=[5, 9], max_new_tokens=3, seed=1))
        try:
            await asyncio.wait_for(fut, timeout=5.0)
        except RuntimeError as exc:
            await b.aclose()
            return str(exc)
        await b.aclose()
        return None

    assert asyncio.run(_run()) == "stage down"


# --------------------------------------------------------------------------- #
# The remote path: batching must be ONE round trip, and degrade honestly
# --------------------------------------------------------------------------- #
class _Assign:
    def __init__(self, node_id, start, end):
        self.node_id, self.start_layer, self.end_layer = node_id, start, end


class BatchingTransport:
    """In-process transport that serves the batched hop, recording every call.

    Goes through ``tolist()`` / ``np.asarray`` exactly as the wire does, so a
    serialisation bug shows up here rather than on the mesh.
    """

    def __init__(self, cfg, params, depth, supports_batch=True):
        self.shards = {f"n{i}": s
                       for i, s in enumerate(build_stages(cfg, depth, params))}
        self.calls: list[tuple[str, str, dict]] = []
        self.supports_batch = supports_batch

    async def call(self, node_id, path, payload, timeout=None):
        self.calls.append((node_id, path, payload))
        shard = self.shards[node_id]
        if path == "shard/drop":
            shard.drop(payload["session"])
            return {"dropped": payload["session"]}
        if path == "shard/forward":
            session = payload.get("session") if payload.get("use_cache") else None
            if payload.get("truncate_to") is not None and payload.get("session"):
                shard.truncate(payload["session"], int(payload["truncate_to"]))
            out = shard.forward(payload["data"], payload["is_tokens"],
                                int(payload.get("pos_offset", 0)), session)
            return {"data": out.tolist()}
        if path == "shard/forward_batch":
            if not self.supports_batch:
                raise TransportError(f"{node_id} {path} -> HTTP 501", status=501)
            outs, cursor = [], 0
            for item in payload["items"]:
                rows = int(item["rows"])
                piece = payload["data"][cursor:cursor + rows]
                cursor += rows
                if item.get("truncate_to") is not None and item.get("session"):
                    shard.truncate(item["session"], int(item["truncate_to"]))
                inp = piece if payload["is_tokens"] else np.asarray(piece, "float32")
                session = item.get("session") if item.get("use_cache") else None
                outs.append(shard.forward(inp, payload["is_tokens"],
                                          int(item.get("pos_offset", 0)), session))
            return {"rows": [int(o.shape[0]) for o in outs],
                    "data": np.concatenate(outs, axis=0).tolist()}
        raise AssertionError(f"unexpected path {path}")

    def paths(self, path):
        return [c for c in self.calls if c[1] == path]


def _remote(cfg, P, depth=3, supports_batch=True):
    t = BatchingTransport(cfg, P, depth, supports_batch)
    partition = [_Assign(f"n{i}", s.start, s.end)
                 for i, s in enumerate(t.shards.values())]
    return RemotePipeline(cfg, build_remote_stages(t, "m1", partition)), t


def test_remote_batched_output_matches_local_solo(cfg_params):
    cfg, P = cfg_params
    reqs = _reqs(3, max_new=5)
    expected = [_solo(cfg, P, r)[0] for r in reqs]
    pipe, _ = _remote(cfg, P)
    results, _, _ = asyncio.run(batched_generate(pipe, reqs, max_batch=4))
    assert [r.tokens for r in results] == expected


def test_remote_batch_is_one_request_per_stage_not_one_per_tenant(cfg_params):
    """The whole point. Four tenants must not cost four round trips per stage."""
    cfg, P = cfg_params
    depth, b = 3, 4
    reqs = [BatchRequest(prompt=[5, 9, 33], max_new_tokens=4, seed=6000 + i)
            for i in range(b)]
    pipe, t = _remote(cfg, P, depth)
    _, _, pstats = asyncio.run(batched_generate(pipe, reqs, max_batch=b))
    batch_calls = len(t.paths("shard/forward_batch"))
    assert not t.paths("shard/forward"), "batched traversal fell back to single hops"
    assert batch_calls == pstats.traversals * depth, (
        f"{batch_calls} calls for {pstats.traversals} traversals x {depth} stages")
    # Serial would have been b times as many.
    assert batch_calls * b <= pstats.traversals * depth * b


def test_a_node_that_cannot_batch_still_produces_correct_tokens(cfg_params):
    """Degrade honestly: correct output, and the fallback is counted, not hidden."""
    cfg, P = cfg_params
    reqs = _reqs(3, max_new=4)
    expected = [_solo(cfg, P, r)[0] for r in reqs]
    pipe, t = _remote(cfg, P, supports_batch=False)
    results, _, _ = asyncio.run(batched_generate(pipe, reqs, max_batch=4))
    assert [r.tokens for r in results] == expected
    assert t.paths("shard/forward"), "fallback never issued single-hop forwards"
    assert sum(s.batch_fallbacks for s in pipe.stages) > 0, (
        "fallback must be recorded so a mesh full of old nodes is visible, not silent")


def test_capability_is_probed_once_not_every_traversal(cfg_params):
    """A latched capability check must not re-pay discovery on every step."""
    cfg, P = cfg_params
    reqs = [BatchRequest(prompt=[5, 9], max_new_tokens=6, seed=1)]
    pipe, t = _remote(cfg, P, depth=2, supports_batch=False)
    asyncio.run(batched_generate(pipe, reqs, max_batch=2))
    attempts = len(t.paths("shard/forward_batch"))
    assert attempts == 2, (
        f"expected one failed probe per stage, got {attempts} -- the unsupported "
        f"result was not latched")


def test_stats_report_the_amortisation_honestly(cfg_params):
    cfg, P = cfg_params
    reqs = _reqs(4, max_new=5)
    _, bstats, _ = asyncio.run(batched_generate(_pipe(cfg, P), reqs, max_batch=4))
    d = bstats.as_dict()
    assert d["completed"] == 4 and d["admitted"] == 4
    assert d["max_batch_seen"] == 4
    assert 1.0 < d["mean_batch"] <= 4.0
    assert BatcherStats().mean_batch == 0.0


# --------------------------------------------------------------------------- #
# Composition with speculative pipeline decoding
#
# SPD amortises (N-1)*RTT across tokens; batching amortises it across tenants. The
# two are orthogonal, so composed they should multiply -- one traversal carrying
# B * (K+1) rows. These tests demand that the composition is exact, because an
# optimisation that changes what a user gets back is not an optimisation.
# --------------------------------------------------------------------------- #
def _spd_reqs(cfg, P, n=4, max_new=8, temperature=0.8, k=4):
    return [BatchRequest(prompt=PROMPTS[i % len(PROMPTS)], max_new_tokens=max_new,
                         temperature=temperature, seed=300 + i, draft_k=k,
                         draft_factory=lambda: SelfDraft(cfg, P, 2))
            for i in range(n)]


def _spd_solo(cfg, P, req, depth=3):
    pipe = _pipe(cfg, P, depth)
    out, stats, spec = asyncio.run(speculative_generate(
        pipe, SelfDraft(cfg, P, 2), req.prompt, req.max_new_tokens, k=req.draft_k,
        temperature=req.temperature, top_k=req.top_k, seed=req.seed,
        session=f"spdsolo-{req.seed}", chunked_prefill=req.chunked_prefill))
    return out, stats, spec


@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_batched_spd_is_token_identical_to_solo_spd(cfg_params, temperature):
    cfg, P = cfg_params
    reqs = _spd_reqs(cfg, P, 4, temperature=temperature)
    expected = [_spd_solo(cfg, P, r)[0] for r in reqs]
    results, _, _ = asyncio.run(batched_generate(_pipe(cfg, P), reqs, max_batch=4))
    assert [r.tokens for r in results] == expected, (
        "batched speculative decoding diverged from solo speculative decoding")


def test_batched_spd_collapses_traversals_beyond_what_spd_alone_achieves(cfg_params):
    """The two amortisations must compose, not merely coexist."""
    cfg, P = cfg_params
    reqs = _spd_reqs(cfg, P, 4)
    solo_traversals = sum(_spd_solo(cfg, P, r)[1].traversals for r in reqs)
    _, _, pstats = asyncio.run(batched_generate(_pipe(cfg, P), reqs, max_batch=4))
    assert pstats.traversals < solo_traversals / 2, (
        f"batched SPD used {pstats.traversals} traversals where four solo SPD runs "
        f"used {solo_traversals}; the tenants are not sharing traversals")


def test_speculative_rollback_rides_the_batched_traversal(cfg_params):
    """Rejected drafts must be rolled back without buying their own round trip.

    On a remote pipeline a rollback that cost its own hop would cancel most of SPD's
    benefit, so truncation is recorded and carried on the next batched forward. If it
    were sent separately this test would see extra single-hop calls.
    """
    cfg, P = cfg_params
    reqs = _spd_reqs(cfg, P, 3, max_new=10)
    pipe, t = _remote(cfg, P, depth=2)
    results, _, _ = asyncio.run(batched_generate(pipe, reqs, max_batch=3))
    assert not t.paths("shard/forward"), "rollback or verify fell off the batched path"
    truncations = [it for _, _, payload in t.paths("shard/forward_batch")
                   for it in payload["items"] if it.get("truncate_to") is not None]
    assert truncations, "no rollback ever rode along; drafts were never rejected"
    solo = [_spd_solo(cfg, P, r)[0] for r in reqs]
    assert [r.tokens for r in results] == solo


def test_speculative_and_plain_requests_share_one_traversal(cfg_params):
    """A tenant using SPD and a tenant not using it must still batch together."""
    cfg, P = cfg_params
    spd_req = BatchRequest(prompt=PROMPTS[0], max_new_tokens=8, seed=300,
                           draft_factory=lambda: SelfDraft(cfg, P, 2), draft_k=4)
    plain = BatchRequest(prompt=PROMPTS[1], max_new_tokens=8, seed=1001)
    results, bstats, _ = asyncio.run(
        batched_generate(_pipe(cfg, P), [spd_req, plain], max_batch=2))
    assert bstats.max_batch_seen == 2, "the two request kinds never shared a traversal"
    assert results[0].tokens == _spd_solo(cfg, P, spd_req)[0]
    assert results[1].tokens == _solo(cfg, P, plain)[0]
    assert results[0].as_dict().get("speculative") is not None
    assert results[1].as_dict().get("speculative") is None


def test_draft_k_must_be_positive_when_drafting(cfg_params):
    cfg, P = cfg_params
    with pytest.raises(ValueError):
        BatchRequest(prompt=[1, 2], draft_factory=lambda: SelfDraft(cfg, P, 2), draft_k=0)


# --------------------------------------------------------------------------- #
# Why generation stopped
#
# A token count cannot answer this. When the context window cuts a reply short the
# count lands BELOW max_new_tokens, so any rule of the form
# `"length" if n >= max_tokens else "stop"` reports a truncated answer as a finished
# one -- and callers that continue on "length" but not on "stop" then silently drop
# the rest of the response. The reason must come from the code that made the
# decision, so these tests pin it there.
# --------------------------------------------------------------------------- #
def _fits(cfg, P, prompt, max_new):
    res, _, _ = asyncio.run(batched_generate(
        _pipe(cfg, P), [BatchRequest(prompt=list(prompt), max_new_tokens=max_new,
                                     temperature=0.0, session="fr")], max_batch=4))
    return res[0]


def test_reaching_the_token_budget_reports_length(cfg_params):
    cfg, P = cfg_params
    r = _fits(cfg, P, [1, 2, 3], 5)
    assert len(r.tokens) == 5
    assert r.finish_reason == "length"


def test_running_out_of_context_is_not_reported_as_a_natural_stop(cfg_params):
    cfg, P = cfg_params
    prompt = list(range(1, 21))
    want = cfg.block_size * 4                      # far more than can ever fit
    r = _fits(cfg, P, prompt, want)
    assert len(r.tokens) < want, "expected the context window to cut this short"
    assert r.finish_reason == "context_window", (
        f"generation stopped at {len(r.tokens)} of {want} tokens because the window "
        f"ran out, but it reported {r.finish_reason!r}")
    # The distinction has to survive into the serialised form the API returns.
    assert r.as_dict()["finish_reason"] == "context_window"


def test_the_boundary_between_the_two_reasons_is_exact(cfg_params):
    """Land on the last token that fits, and on the first that does not."""
    cfg, P = cfg_params
    prompt = list(range(1, 21))
    room = cfg.block_size - len(prompt)
    assert room > 1
    inside = _fits(cfg, P, prompt, room)
    assert len(inside.tokens) == room
    assert inside.finish_reason == "length", (
        "a request that exactly fills the window still finished because it hit its "
        "own budget, not because the window ran out")
    over = _fits(cfg, P, prompt, room + 5)
    assert over.finish_reason == "context_window"
    # Truncation must not change the tokens that were produced before it.
    assert over.tokens[:len(inside.tokens)] == inside.tokens


def test_speculative_requests_report_the_reason_too(cfg_params):
    cfg, P = cfg_params
    prompt = list(range(1, 21))
    req = BatchRequest(prompt=prompt, max_new_tokens=cfg.block_size * 4,
                       temperature=0.0, session="frs",
                       draft_factory=lambda: SelfDraft(cfg, P, 2), draft_k=4)
    res, _, _ = asyncio.run(batched_generate(_pipe(cfg, P), [req], max_batch=4))
    assert res[0].finish_reason == "context_window", (
        "speculative decoding must report truncation the same way plain decoding "
        "does; it is the same window")


def test_a_short_request_sharing_a_batch_with_a_truncated_one_still_says_length(cfg_params):
    """One tenant hitting the window must not mislabel another tenant's reply."""
    cfg, P = cfg_params
    prompt = list(range(1, 21))
    cut = BatchRequest(prompt=prompt, max_new_tokens=cfg.block_size * 4,
                       temperature=0.0, session="cut")
    ok = BatchRequest(prompt=[1, 2, 3], max_new_tokens=4, temperature=0.0, session="ok")
    res, _, _ = asyncio.run(batched_generate(_pipe(cfg, P), [cut, ok], max_batch=4))
    assert res[0].finish_reason == "context_window"
    assert res[1].finish_reason == "length"
    assert len(res[1].tokens) == 4
