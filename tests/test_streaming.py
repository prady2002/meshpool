"""Streaming must be real, not a progress bar drawn after the fact.

This file exists because the OpenAI-compatible route once passed every streaming
test it had while doing no streaming at all: it generated the whole answer, then
sliced the finished string into Server-Sent Events. The transcript was
indistinguishable from the real thing -- correct frame shapes, correct ``[DONE]``
sentinel, many small deltas -- and every format assertion was green. Time-to-first-
token equalled total generation time.

So none of the tests here assert on shape. They assert on *ordering in time*:

  * a token reaches the caller while generation is still running, proved without a
    timing threshold by checking that the producing task is not yet done;
  * what was streamed reconstructs the final answer exactly;
  * nothing is streamed that a later decode would contradict.

The first of those is the one that a fake implementation cannot satisfy.
"""
from __future__ import annotations

import asyncio

import pytest

from meshpool.inference import model as M
from meshpool.inference.batching import BatchRequest, ContinuousBatcher
from meshpool.inference.engine import InferenceManager
from meshpool.inference.pipeline import LinkProfile, Pipeline, build_stages
from meshpool.inference.spd import SelfDraft

PROMPT = [5, 9, 33, 41, 7]


@pytest.fixture(scope="module")
def cfg_params():
    cfg = M.REGISTRY["nano-gpt"]
    P, _ = M.load_params(cfg)
    return cfg, P


def _pipe(cfg, P, depth=3, rtt=0.0):
    return Pipeline(cfg, build_stages(cfg, depth, P), LinkProfile(rtt_ms=rtt))


async def _run(pipe, req: BatchRequest, **kw):
    b = ContinuousBatcher(pipe, **kw)
    await b.start()
    try:
        return await b.generate(req)
    finally:
        await b.aclose()


# --------------------------------------------------------------------------- #
# The property that a fake cannot fake: tokens arrive before the answer is done
# --------------------------------------------------------------------------- #

class TestEmissionIsIncremental:
    def test_first_token_arrives_before_generation_finishes(self, cfg_params):
        """The whole thesis, asserted without a stopwatch.

        ``task.done()`` is a fact, not a measurement, so this cannot flake the way a
        "TTFT < 30% of total" threshold would on a loaded CI box. Under the old
        generate-then-chop implementation there was no callback at all; under a
        plausible wrong implementation that buffers tokens and flushes them at the
        end, the task would already be complete when the first one showed up.
        """
        cfg, P = cfg_params

        async def go():
            seen: list[int] = []
            first = asyncio.Event()

            def on_tokens(toks):
                seen.extend(toks)
                first.set()

            pipe = _pipe(cfg, P)
            b = ContinuousBatcher(pipe)
            await b.start()
            try:
                task = asyncio.create_task(b.generate(BatchRequest(
                    prompt=PROMPT, max_new_tokens=24, seed=7, on_tokens=on_tokens)))
                await first.wait()
                mid_flight = not task.done()
                tokens_at_first_delta = len(seen)
                res = await task
            finally:
                await b.aclose()
            return mid_flight, tokens_at_first_delta, res

        mid_flight, at_first, res = asyncio.run(go())
        assert len(res.tokens) == 24
        assert mid_flight, (
            "the first token was only delivered after generation had already "
            "finished -- that is buffering, not streaming")
        assert at_first < len(res.tokens), (
            f"the caller received {at_first} of {len(res.tokens)} tokens in a single "
            "callback, so nothing was actually incremental")

    def test_every_token_is_delivered_exactly_once_and_in_order(self, cfg_params):
        cfg, P = cfg_params
        seen: list[int] = []
        res = asyncio.run(_run(_pipe(cfg, P), BatchRequest(
            prompt=PROMPT, max_new_tokens=20, seed=11,
            on_tokens=seen.extend)))
        assert seen == res.tokens

    def test_callbacks_are_fine_grained_not_one_big_flush(self, cfg_params):
        """Exact, not "more than one".

        Non-speculative decoding produces exactly one token per traversal, so the
        number of callbacks must equal the number of tokens. A batcher that
        accumulated and flushed every few steps would still be "streaming" by a
        weaker assertion while quietly reintroducing latency.
        """
        cfg, P = cfg_params
        batches: list[list[int]] = []
        res = asyncio.run(_run(_pipe(cfg, P), BatchRequest(
            prompt=PROMPT, max_new_tokens=16, seed=3,
            on_tokens=lambda t: batches.append(list(t)))))
        assert len(batches) == len(res.tokens) == 16
        assert all(len(b) == 1 for b in batches)


# --------------------------------------------------------------------------- #
# Streaming must not perturb what is generated
# --------------------------------------------------------------------------- #

class TestEmissionIsSideEffectFree:
    def test_output_is_identical_with_and_without_a_listener(self, cfg_params):
        """The control. Observation must not change the observed.

        Sampling consumes a seeded RNG; anything that draws from it, reorders steps,
        or copies state differently when a listener is attached would silently change
        the answer a streaming client gets versus the one a non-streaming client gets
        for the same seed.
        """
        cfg, P = cfg_params
        quiet = asyncio.run(_run(_pipe(cfg, P), BatchRequest(
            prompt=PROMPT, max_new_tokens=18, seed=99)))
        loud = asyncio.run(_run(_pipe(cfg, P), BatchRequest(
            prompt=PROMPT, max_new_tokens=18, seed=99, on_tokens=lambda t: None)))
        assert quiet.tokens == loud.tokens

    def test_a_raising_callback_cannot_kill_the_shared_loop(self, cfg_params):
        """One tenant hanging up must not end everyone else's generation.

        The batcher is shared across every request on a deployment. A client that
        disconnects mid-stream is the ordinary case, not the exception, and its
        callback raising is how that shows up here.
        """
        cfg, P = cfg_params
        calls = {"n": 0}

        def angry(toks):
            calls["n"] += 1
            raise RuntimeError("client hung up")

        res = asyncio.run(_run(_pipe(cfg, P), BatchRequest(
            prompt=PROMPT, max_new_tokens=12, seed=5, on_tokens=angry)))
        assert len(res.tokens) == 12
        assert calls["n"] == 1, (
            "a callback that raised was called again -- a disconnected client would "
            "cost one exception per token for the rest of the generation")

    def test_a_raising_callback_does_not_change_the_answer(self, cfg_params):
        cfg, P = cfg_params

        def angry(toks):
            raise RuntimeError("boom")

        quiet = asyncio.run(_run(_pipe(cfg, P), BatchRequest(
            prompt=PROMPT, max_new_tokens=12, seed=5)))
        broken = asyncio.run(_run(_pipe(cfg, P), BatchRequest(
            prompt=PROMPT, max_new_tokens=12, seed=5, on_tokens=angry)))
        assert quiet.tokens == broken.tokens


# --------------------------------------------------------------------------- #
# Speculative decoding emits whole accepted runs, and only accepted ones
# --------------------------------------------------------------------------- #

class TestSpeculativeEmission:
    def test_spd_emits_accepted_runs_and_never_a_rejected_draft(self, cfg_params):
        """Rejected drafts must never reach the wire.

        SPD proposes K tokens and keeps a prefix. Emitting on proposal instead of on
        acceptance would stream text that the model then un-generates -- and unlike a
        KV-cache rollback, an SSE frame the client has already rendered cannot be
        rolled back.
        """
        cfg, P = cfg_params
        seen: list[int] = []
        batches: list[list[int]] = []

        def on_tokens(toks):
            seen.extend(toks)
            batches.append(list(toks))

        pipe = _pipe(cfg, P)
        res = asyncio.run(_run(pipe, BatchRequest(
            prompt=PROMPT, max_new_tokens=20, seed=13,
            draft_factory=lambda: SelfDraft(cfg, P, 2), draft_k=4,
            on_tokens=on_tokens)))
        assert seen == res.tokens
        assert res.spec is not None
        # One callback per verification traversal, each carrying a whole accepted run
        # plus the correction token -- that is what makes SPD cheaper than decoding.
        assert len(batches) == res.spec["verify_traversals"]
        assert max(len(b) for b in batches) > 1, (
            "every SPD traversal emitted a single token, so no draft was ever "
            "accepted and this test is not exercising what it claims to")

    def test_spd_never_emits_more_than_max_new_tokens(self, cfg_params):
        """The last traversal can accept a run that overshoots the caller's limit."""
        cfg, P = cfg_params
        seen: list[int] = []
        res = asyncio.run(_run(_pipe(cfg, P), BatchRequest(
            prompt=PROMPT, max_new_tokens=7, seed=21,
            draft_factory=lambda: SelfDraft(cfg, P, 2), draft_k=4,
            on_tokens=seen.extend)))
        assert len(seen) == 7 == len(res.tokens)
        assert seen == res.tokens


# --------------------------------------------------------------------------- #
# A shared scheduler loop must never die silently
#
# Found the hard way: a request whose draft_factory raised took the loop down with
# it, because the factory is called from _admit, outside the per-step handler. The
# loop task then held the exception, nothing ever drained the queue again, and every
# caller -- including tenants with nothing to do with the bad request -- awaited a
# future that would never resolve. No error, no timeout, just permanent silence on a
# shared deployment.
# --------------------------------------------------------------------------- #

class TestSchedulerLoopSurvival:
    def test_a_bad_draft_factory_fails_only_its_own_request(self, cfg_params):
        cfg, P = cfg_params

        async def go():
            b = ContinuousBatcher(_pipe(cfg, P))
            await b.start()
            try:
                bad = asyncio.create_task(b.generate(BatchRequest(
                    prompt=PROMPT, max_new_tokens=8, seed=1,
                    draft_factory=lambda: SelfDraft(cfg, P, cfg.n_layer + 5),
                    draft_k=4)))
                good = asyncio.create_task(b.generate(BatchRequest(
                    prompt=PROMPT, max_new_tokens=8, seed=2)))
                with pytest.raises(ValueError):
                    await bad
                res = await asyncio.wait_for(good, timeout=10)
                return res, b.alive, b.stats.rejected
            finally:
                await b.aclose()

        res, alive, rejected = asyncio.run(go())
        assert len(res.tokens) == 8, "an unrelated tenant lost its generation"
        assert alive, "one bad request killed the scheduler loop"
        assert rejected == 1

    def test_a_dead_loop_refuses_new_work_instead_of_swallowing_it(self, cfg_params):
        """Fail fast beats hanging forever.

        If the loop ever does die, the next caller must find out immediately. A
        request accepted into a queue nobody drains is the worst possible outcome:
        indistinguishable from a slow model until the client times out.
        """
        cfg, P = cfg_params

        async def go():
            b = ContinuousBatcher(_pipe(cfg, P))
            await b.start()
            b._task.cancel()
            try:
                await b._task
            except asyncio.CancelledError:
                pass
            assert not b.alive
            with pytest.raises(RuntimeError):
                b.submit(BatchRequest(prompt=PROMPT, max_new_tokens=4))

        asyncio.run(go())

    def test_loop_death_hands_the_error_to_everyone_waiting(self, cfg_params):
        """Nobody waits forever, including requests still in the queue."""
        cfg, P = cfg_params

        async def go():
            b = ContinuousBatcher(_pipe(cfg, P), max_batch=1)
            await b.start()
            # Poison the pipeline so the *next* step raises out of the loop body.
            active = asyncio.create_task(b.generate(
                BatchRequest(prompt=PROMPT, max_new_tokens=32, seed=1)))
            queued = asyncio.create_task(b.generate(
                BatchRequest(prompt=PROMPT, max_new_tokens=32, seed=2)))
            await asyncio.sleep(0)
            b._admit = _boom
            results = await asyncio.gather(active, queued, return_exceptions=True)
            await b.aclose()
            return results

        results = asyncio.run(go())
        assert all(isinstance(r, RuntimeError) for r in results), (
            f"a caller was left hanging after the loop died: {results}")


def _boom():
    raise RuntimeError("scheduler exploded")


# --------------------------------------------------------------------------- #
# chat_stream: turning token ids into text deltas safely
# --------------------------------------------------------------------------- #
class _Tok:
    """Append-only decode, like the built-in character tokenizer."""

    def decode(self, ids):
        return "".join(chr(96 + i) for i in ids)


class _LateTok:
    """A decode that only resolves once a pair is complete.

    Stands in for a byte-pair tokenizer emitting half of a multi-byte character:
    the intermediate decode contains U+FFFD and is not safe to forward.
    """

    def decode(self, ids):
        out = []
        i = 0
        while i + 1 < len(ids):
            out.append("X" if ids[i] else "Z")
            i += 2
        if i < len(ids):
            out.append("\ufffd")
        return "".join(out)


class _Streamer:
    """Minimal host for the real ``chat_stream`` implementation.

    ``chat_stream`` depends on exactly two things -- ``self.chat`` and
    ``self._tok`` -- so binding the real method here tests the real code rather
    than a copy of it, without standing up a control plane.
    """

    chat_stream = InferenceManager.chat_stream

    def __init__(self, tok, script, text=None, raises=None):
        self._tok = {"m": tok} if tok is not None else {}
        self._script = script
        self._text = text
        self._raises = raises

    async def chat(self, model_id, prompt, max_new_tokens, on_tokens=None, **kw):
        if self._raises is not None:
            raise self._raises
        ids = []
        for group in self._script:
            ids.extend(group)
            if on_tokens is not None:
                on_tokens(list(group))
            await asyncio.sleep(0)
        text = self._text
        if text is None:
            tok = self._tok.get(model_id)
            text = tok.decode(ids) if tok is not None else ""
        return {"model": "m", "text": text, "tokens": len(ids)}


async def _drain(streamer):
    deltas, final = [], None
    async for item in streamer.chat_stream("m", "hi", 8):
        if "delta" in item:
            deltas.append(item["delta"])
        else:
            final = item
    return deltas, final


class TestChatStream:
    def test_deltas_concatenate_to_the_final_text(self):
        s = _Streamer(_Tok(), [[1], [2], [3], [4]])
        deltas, final = asyncio.run(_drain(s))
        assert "".join(deltas) == "abcd" == final["result"]["text"]
        assert final["done"] is True

    def test_deltas_arrive_one_per_emission_not_one_at_the_end(self):
        s = _Streamer(_Tok(), [[1], [2], [3], [4]])
        deltas, _ = asyncio.run(_drain(s))
        assert deltas == ["a", "b", "c", "d"]

    def test_result_is_the_last_item_and_carries_the_full_chat_payload(self):
        s = _Streamer(_Tok(), [[1], [2]])
        items = []

        async def go():
            async for item in s.chat_stream("m", "hi", 8):
                items.append(item)

        asyncio.run(go())
        assert set(items[-1]) == {"done", "result"}
        assert items[-1]["result"]["tokens"] == 2
        assert all("delta" in i for i in items[:-1])

    def test_partial_multibyte_decode_is_held_back_never_streamed(self):
        """U+FFFD must never reach a client.

        Decoding tokens one at a time would emit a replacement character here and
        then the real character, so the client would render mojibake that the final
        text does not contain. Prefix-diff decoding with a hold-back does not.
        """
        s = _Streamer(_LateTok(), [[1], [1], [0], [0]])
        deltas, final = asyncio.run(_drain(s))
        assert "\ufffd" not in "".join(deltas)
        assert "".join(deltas) == final["result"]["text"]

    def test_a_backend_that_emits_no_tokens_still_yields_its_text(self):
        """Whole-model replicas behind Ollama produce no incremental tokens.

        One late delta is the honest outcome. Yielding nothing at all would drop the
        answer; faking a drip would misreport time-to-first-token.
        """
        s = _Streamer(_Tok(), [], text="from ollama")
        deltas, final = asyncio.run(_drain(s))
        assert deltas == ["from ollama"]
        assert final["result"]["text"] == "from ollama"

    def test_generation_errors_surface_on_the_first_step(self):
        """So the route can still answer with a status code instead of a broken stream."""
        s = _Streamer(_Tok(), [[1]], raises=KeyError("no such deployment"))
        agen = s.chat_stream("m", "hi", 8)

        async def go():
            with pytest.raises(KeyError):
                await agen.__anext__()

        asyncio.run(go())

    def test_an_error_after_partial_output_still_propagates(self):
        class _Halfway(_Streamer):
            async def chat(self, model_id, prompt, max_new_tokens,
                           on_tokens=None, **kw):
                on_tokens([1])
                await asyncio.sleep(0)
                raise RuntimeError("stage died")

        s = _Halfway(_Tok(), [])

        async def go():
            seen = []
            with pytest.raises(RuntimeError):
                async for item in s.chat_stream("m", "hi", 8):
                    seen.append(item)
            return seen

        seen = asyncio.run(go())
        assert seen == [{"delta": "a"}], (
            "the tokens produced before the failure must still have been streamed")

    def test_cancelling_the_consumer_cancels_the_generation(self):
        """An abandoned stream must not leave a batch slot generating forever.

        Real streaming makes this the common case: a browser tab closed halfway
        through an answer. Without propagation the mesh keeps spending wide-area
        round trips -- and someone's credits -- producing text nobody will read.
        """
        async def go():
            cancelled = asyncio.Event()

            class _Forever(_Streamer):
                async def chat(self, model_id, prompt, max_new_tokens,
                               on_tokens=None, **kw):
                    on_tokens([1])
                    try:
                        await asyncio.sleep(30)
                    except asyncio.CancelledError:
                        cancelled.set()
                        raise
                    return {"model": "m", "text": "never", "tokens": 1}

            s = _Forever(_Tok(), [])
            agen = s.chat_stream("m", "hi", 8)
            first = await agen.__anext__()
            assert first == {"delta": "a"}
            await agen.aclose()
            await asyncio.wait_for(cancelled.wait(), timeout=5)

        asyncio.run(go())

    def test_abandoned_sequences_stop_riding_traversals(self, cfg_params):
        """The batcher half of the same property, measured rather than asserted at.

        A cancelled future is only cheap if the scheduler notices it. This kills a
        generation partway through and checks the shared loop stopped spending steps
        on it, using a second live request as the clock.
        """
        cfg, P = cfg_params

        async def go():
            seen = asyncio.Event()
            victim_tokens: list[int] = []

            def on_tokens(toks):
                victim_tokens.extend(toks)
                if len(victim_tokens) >= 3:
                    seen.set()

            b = ContinuousBatcher(_pipe(cfg, P))
            await b.start()
            try:
                victim = asyncio.create_task(b.generate(BatchRequest(
                    prompt=PROMPT, max_new_tokens=200, seed=1,
                    on_tokens=on_tokens)))
                other = asyncio.create_task(b.generate(BatchRequest(
                    prompt=PROMPT, max_new_tokens=40, seed=2)))
                await seen.wait()
                victim.cancel()
                await other
                at_cancel = len(victim_tokens)
                # Let the loop run well past the point where the victim would have
                # finished had it kept going.
                await asyncio.sleep(0.05)
                return at_cancel, len(victim_tokens), b.in_flight
            finally:
                await b.aclose()

        at_cancel, after, in_flight = asyncio.run(go())
        assert at_cancel < 200, "the victim finished before it could be cancelled"
        assert after - at_cancel <= 1, (
            f"the cancelled request kept generating: {at_cancel} tokens at cancel, "
            f"{after} afterwards")
        assert in_flight == 0, "the abandoned sequence still occupies a batch slot"
