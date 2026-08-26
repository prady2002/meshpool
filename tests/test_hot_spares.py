"""Hot spares: a pipeline stage survives losing the node it lives on.

The property under test is not "it does not crash". It is much stronger, and it is
the only one worth having:

    the tokens produced by a pipeline that loses a node mid-generation are
    IDENTICAL to the tokens produced by a pipeline that loses nothing.

Anything weaker is worthless here. A spare that takes over is useless if its answers
drift from what the dead node would have said, and "the stream kept flowing" is not
evidence that it did not.

The fake transport below is deliberately built out of real in-process shards, so
what is being tested is the failover and hydration logic, not a mock of it.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from meshpool.common.transport import TransportError
from meshpool.inference import model as M
from meshpool.inference.pipeline import CachedShard, StageCall, build_stages
from meshpool.inference.remote import (
    RemotePipeline,
    RemoteShard,
    StageUnavailable,
)


class FakeNodeTransport:
    """An in-process stand-in for a mesh of nodes running ``shard/forward``.

    Every node id maps to its own CachedShard with its own KV cache, exactly as a
    real node has. Killing a node makes its calls raise, and -- crucially -- the
    spare that takes over starts with a genuinely empty cache, so the test can
    actually observe the bug that hydration exists to prevent.
    """

    def __init__(self, cfg: M.ModelConfig, layout: dict[str, tuple[int, int, bool, bool]],
                 params: dict) -> None:
        self.cfg = cfg
        self.shards: dict[str, CachedShard] = {
            node: CachedShard(cfg, start, end, first, last, params)
            for node, (start, end, first, last) in layout.items()
        }
        self.dead: set[str] = set()
        self.calls: list[tuple[str, str]] = []
        self.forward_counts: dict[str, int] = {n: 0 for n in layout}

    def kill(self, node: str) -> None:
        self.dead.add(node)

    async def call(self, node_id: str, op: str, payload: dict, timeout: float = 60.0):
        self.calls.append((node_id, op))
        if node_id in self.dead:
            raise TransportError(f"node {node_id} is gone", status=503)
        shard = self.shards[node_id]
        if op == "shard/drop":
            shard.drop(payload["session"])
            return {"ok": True}
        if op == "shard/forward":
            self.forward_counts[node_id] += 1
            session = payload.get("session")
            if session is not None and "truncate_to" in payload:
                shard.truncate(session, int(payload["truncate_to"]))
            data = payload["data"]
            if not payload["is_tokens"]:
                data = np.asarray(data, dtype=np.float32)
            out = shard.forward(data, payload["is_tokens"],
                                int(payload.get("pos_offset", 0)), session)
            return {"data": np.asarray(out, dtype=np.float32).tolist()}
        raise TransportError(f"unsupported op {op}", status=404)


def _mesh(n_stages: int = 3, spare_for: dict[str, list[str]] | None = None):
    """Build a RemotePipeline over FakeNodeTransport, plus a local reference."""
    cfg = M.ModelConfig()
    params, _ = M.load_params(cfg)
    base, extra = divmod(cfg.n_layer, n_stages)
    layout: dict[str, tuple[int, int, bool, bool]] = {}
    primaries: list[str] = []
    cursor = 0
    for s in range(n_stages):
        width = base + (1 if s < extra else 0)
        start, end = cursor, cursor + width
        node = f"n{s}"
        primaries.append(node)
        layout[node] = (start, end, s == 0, s == n_stages - 1)
        for spare in (spare_for or {}).get(node, []):
            layout[spare] = (start, end, s == 0, s == n_stages - 1)
        cursor = end
    transport = FakeNodeTransport(cfg, layout, params)
    stages = []
    for i, node in enumerate(primaries):
        start, end, first, last = layout[node]
        stages.append(RemoteShard(transport, node, "m", start, end, first, last,
                                  spares=(spare_for or {}).get(node)))
    return cfg, params, transport, RemotePipeline(cfg, stages)


async def _decode(pipe, prompt: list[int], n_new: int, session: str,
                  kill_at: int = -1, killer=None) -> list[int]:
    """Greedy decode, optionally killing a node partway through."""
    out: list[int] = []
    logits = await pipe.forward(prompt, pos_offset=0, session=session)
    nxt = int(np.asarray(logits)[-1].argmax())
    out.append(nxt)
    pos = len(prompt)
    for step in range(n_new - 1):
        if step == kill_at and killer is not None:
            killer()
        logits = await pipe.forward([nxt], pos_offset=pos, session=session)
        nxt = int(np.asarray(logits)[-1].argmax())
        out.append(nxt)
        pos += 1
    return out


# --------------------------------------------------------------------------- #
# The headline property
# --------------------------------------------------------------------------- #

def test_midstream_node_death_produces_identical_tokens():
    """Kill a middle stage's node mid-generation: the output must not change."""
    async def run():
        prompt = [1, 2, 3, 4, 5]
        _, _, _, clean = _mesh(3, {"n1": ["n1b"]})
        reference = await _decode(clean, prompt, 12, "s")

        _, _, transport, pipe = _mesh(3, {"n1": ["n1b"]})
        got = await _decode(pipe, prompt, 12, "s", kill_at=4,
                            killer=lambda: transport.kill("n1"))
        return reference, got, pipe, transport

    reference, got, pipe, transport = asyncio.run(run())
    assert pipe.failovers == 1, f"expected exactly one failover, got {pipe.failovers}"
    assert pipe.hydrations == 1, f"expected exactly one hydration, got {pipe.hydrations}"
    assert pipe.stages[1].node_id == "n1b"
    assert transport.forward_counts["n1b"] > 0, "the spare never served anything"
    assert got == reference, (
        f"failover changed the output:\n  clean={reference}\n  after={got}")


def test_first_stage_death_is_also_survivable():
    """Stage 0 is recoverable too -- via the token journal, not activations."""
    async def run():
        prompt = [7, 8, 9]
        _, _, _, clean = _mesh(3, {"n0": ["n0b"]})
        reference = await _decode(clean, prompt, 10, "s")

        _, _, transport, pipe = _mesh(3, {"n0": ["n0b"]})
        got = await _decode(pipe, prompt, 10, "s", kill_at=3,
                            killer=lambda: transport.kill("n0"))
        return reference, got, pipe

    reference, got, pipe = asyncio.run(run())
    assert pipe.stages[0].node_id == "n0b"
    assert pipe.failovers == 1 and pipe.hydrations == 1
    assert got == reference, (
        f"stage-0 failover changed the output:\n  clean={reference}\n  after={got}")


def test_last_stage_death_is_survivable():
    async def run():
        prompt = [3, 1, 4]
        _, _, _, clean = _mesh(3, {"n2": ["n2b"]})
        reference = await _decode(clean, prompt, 10, "s")
        _, _, transport, pipe = _mesh(3, {"n2": ["n2b"]})
        got = await _decode(pipe, prompt, 10, "s", kill_at=3,
                            killer=lambda: transport.kill("n2"))
        return reference, got, pipe

    reference, got, pipe = asyncio.run(run())
    assert pipe.stages[2].node_id == "n2b"
    assert got == reference


def test_every_stage_can_die_in_one_generation():
    """Not one failure -- a cascade. Output still must not move."""
    async def run():
        prompt = [2, 2, 2, 2]
        spares = {"n0": ["n0b"], "n1": ["n1b"], "n2": ["n2b"]}
        _, _, _, clean = _mesh(3, spares)
        reference = await _decode(clean, prompt, 14, "s")

        _, _, transport, pipe = _mesh(3, spares)
        out: list[int] = []
        logits = await pipe.forward(prompt, pos_offset=0, session="s")
        nxt = int(np.asarray(logits)[-1].argmax())
        out.append(nxt)
        pos = len(prompt)
        for step in range(13):
            if step == 2:
                transport.kill("n0")
            elif step == 5:
                transport.kill("n2")
            elif step == 8:
                transport.kill("n1")
            logits = await pipe.forward([nxt], pos_offset=pos, session="s")
            nxt = int(np.asarray(logits)[-1].argmax())
            out.append(nxt)
            pos += 1
        return reference, out, pipe

    reference, got, pipe = asyncio.run(run())
    assert pipe.failovers == 3, f"expected 3 failovers, got {pipe.failovers}"
    assert [s.node_id for s in pipe.stages] == ["n0b", "n1b", "n2b"]
    assert got == reference, (
        f"cascading failover changed the output:\n  clean={reference}\n  after={got}")


# --------------------------------------------------------------------------- #
# Non-vacuity: prove the test can actually SEE the bug it guards against
# --------------------------------------------------------------------------- #

def test_without_hydration_the_generation_cannot_continue():
    """The control experiment: prove hydration is load-bearing.

    Disable hydration and repeat the exact same kill. If the generation still
    completed, every assertion above would be passing for free.

    What this measured, and it is worth recording because it is *better* than
    expected: a cold spare does not invent an answer. Both CachedShard.forward and
    the real node handler (agent/daemon.py `/shard/forward`, `/shard/forward_batch`)
    detect "resume at position t with an empty cache" and refuse -- the node reports
    HTTP 409. So MeshPool was never at risk of the silent-corruption failure mode
    that usually motivates this machinery. The honest value of hydration is
    therefore AVAILABILITY, not correctness: without it a failover trades one loud
    failure for another and the user's generation dies either way.
    """
    async def run():
        prompt = [1, 2, 3, 4, 5]
        _, _, transport, pipe = _mesh(3, {"n1": ["n1b"]})
        for s in pipe.stages:
            s._ensure_hydrated = lambda *a, **k: asyncio.sleep(0)  # type: ignore[assignment]
        with pytest.raises(ValueError, match="cache miss"):
            await _decode(pipe, prompt, 12, "s", kill_at=4,
                          killer=lambda: transport.kill("n1"))
        return pipe

    pipe = asyncio.run(run())
    assert pipe.failovers == 1, "the kill must still have caused a failover"
    assert pipe.hydrations == 0, "hydration was supposed to be disabled"


def test_a_cold_spare_refuses_rather_than_guessing():
    """Pin the refusal itself: it is the reason silent corruption is impossible."""
    async def run():
        _, _, transport, pipe = _mesh(3, {"n1": ["n1b"]})
        await pipe.forward([1, 2, 3, 4], pos_offset=0, session="s")
        cold = transport.shards["n1b"]
        with pytest.raises(ValueError, match="cannot resume at position 4"):
            cold.forward(np.zeros((1, pipe.cfg.d_model), dtype=np.float32),
                         False, 4, "s")

    asyncio.run(run())


def test_no_spare_means_a_clear_error_not_a_wrong_answer():
    async def run():
        _, _, transport, pipe = _mesh(3)  # no spares anywhere
        await pipe.forward([1, 2, 3], pos_offset=0, session="s")
        transport.kill("n1")
        with pytest.raises(StageUnavailable) as ei:
            await pipe.forward([4], pos_offset=3, session="s")
        return str(ei.value), pipe

    msg, pipe = asyncio.run(run())
    assert "n1" in msg and "exhausted" in msg
    assert pipe.failovers == 0


def test_client_error_does_not_burn_spares():
    """A 400 is the caller's fault. Rotating on it would hide a bug behind churn."""
    async def run():
        _, _, transport, pipe = _mesh(3, {"n1": ["n1b", "n1c"]})

        async def bad_call(node_id, op, payload, timeout=60.0):
            raise TransportError("malformed payload", status=400)

        transport.call = bad_call  # type: ignore[assignment]
        with pytest.raises(TransportError):
            await pipe.forward([1, 2, 3], pos_offset=0, session="s")
        return pipe

    pipe = asyncio.run(run())
    assert pipe.failovers == 0, "a 4xx must not consume a spare"
    assert pipe.stages[1].node_id == "n1"


def test_spares_are_exhausted_in_order_then_reported():
    async def run():
        _, _, transport, pipe = _mesh(3, {"n1": ["n1b", "n1c"]})
        await pipe.forward([1, 2, 3], pos_offset=0, session="s")
        seen = []
        transport.kill("n1")
        await pipe.forward([4], pos_offset=3, session="s")
        seen.append(pipe.stages[1].node_id)
        transport.kill("n1b")
        await pipe.forward([5], pos_offset=4, session="s")
        seen.append(pipe.stages[1].node_id)
        transport.kill("n1c")
        with pytest.raises(StageUnavailable):
            await pipe.forward([6], pos_offset=5, session="s")
        return seen, pipe

    seen, pipe = asyncio.run(run())
    assert seen == ["n1b", "n1c"]
    assert pipe.stages[1].dead == ["n1", "n1b"]
    assert pipe.failovers == 2


def test_hydration_is_lazy_and_per_session():
    """Two sessions, one dies; only the session that is used again is rebuilt."""
    async def run():
        _, _, transport, pipe = _mesh(3, {"n1": ["n1b"]})
        await pipe.forward([1, 2, 3], pos_offset=0, session="a")
        await pipe.forward([4, 5, 6], pos_offset=0, session="b")
        transport.kill("n1")
        await pipe.forward([7], pos_offset=3, session="a")
        return pipe

    pipe = asyncio.run(run())
    assert pipe.hydrations == 1, (
        f"only session 'a' was used after the failure, so exactly one hydration "
        f"should have been paid for, got {pipe.hydrations}")
    assert "b" in pipe.stages[1]._needs_hydration, (
        "session 'b' must still be marked dirty — its cache is gone too, and "
        "serving it without a rebuild would be silently wrong")


def test_batched_traversal_survives_failover_identically():
    """Two tenants sharing hops: a node dies, neither tenant's output moves."""
    async def run():
        spares = {"n1": ["n1b"]}
        _, _, _, clean = _mesh(3, spares)
        pa = [1, 2, 3]
        pb = [4, 5, 6]

        async def batched(pipe, transport=None, kill_step=-1):
            outs: dict[str, list[int]] = {"a": [], "b": []}
            nxt = {}
            res = await pipe.forward_batch([
                StageCall("a", list(pa), True, 0),
                StageCall("b", list(pb), True, 0)])
            for name, arr in zip(("a", "b"), res):
                nxt[name] = int(np.asarray(arr)[-1].argmax())
                outs[name].append(nxt[name])
            pos = 3
            for step in range(9):
                if step == kill_step and transport is not None:
                    transport.kill("n1")
                res = await pipe.forward_batch([
                    StageCall("a", [nxt["a"]], True, pos),
                    StageCall("b", [nxt["b"]], True, pos)])
                for name, arr in zip(("a", "b"), res):
                    nxt[name] = int(np.asarray(arr)[-1].argmax())
                    outs[name].append(nxt[name])
                pos += 1
            return outs

        reference = await batched(clean)
        _, _, transport, pipe = _mesh(3, spares)
        got = await batched(pipe, transport, kill_step=3)
        return reference, got, pipe

    reference, got, pipe = asyncio.run(run())
    assert pipe.failovers == 1
    assert pipe.hydrations == 2, (
        f"both in-flight sessions ride the same traversal, so both must be "
        f"rebuilt, got {pipe.hydrations}")
    assert got == reference, (
        f"batched failover changed output:\n  clean={reference}\n  after={got}")


# --------------------------------------------------------------------------- #
# The token journal, which is what makes stage 0 recoverable at all
# --------------------------------------------------------------------------- #

def test_token_journal_tracks_the_real_history():
    async def run():
        cfg = M.ModelConfig()
        from meshpool.inference.pipeline import Pipeline
        pipe = Pipeline(cfg, build_stages(cfg, 2))
        await pipe.forward([1, 2, 3], pos_offset=0, session="s")
        await pipe.forward([4], pos_offset=3, session="s")
        await pipe.forward([5], pos_offset=4, session="s")
        return pipe

    pipe = asyncio.run(run())
    assert pipe.token_journal["s"] == [1, 2, 3, 4, 5]


def test_truncate_rolls_the_journal_back_too():
    """Rejected speculative drafts must vanish from the journal.

    If they did not, a later stage-0 rebuild would replay tokens the model never
    accepted and silently reconstruct a different conversation.
    """
    async def run():
        cfg = M.ModelConfig()
        from meshpool.inference.pipeline import Pipeline
        pipe = Pipeline(cfg, build_stages(cfg, 2))
        await pipe.forward([1, 2, 3, 4, 5], pos_offset=0, session="s")
        pipe.truncate_session("s", 3)
        return pipe

    pipe = asyncio.run(run())
    assert pipe.token_journal["s"] == [1, 2, 3]


def test_journal_rejects_a_positional_gap():
    async def run():
        cfg = M.ModelConfig()
        from meshpool.inference.pipeline import Pipeline
        pipe = Pipeline(cfg, build_stages(cfg, 2))
        await pipe.forward([1, 2, 3], pos_offset=0, session="s")
        with pytest.raises(ValueError, match="cannot append at"):
            await pipe.forward([9], pos_offset=7, session="s")

    asyncio.run(run())


def test_drop_session_clears_the_journal():
    async def run():
        cfg = M.ModelConfig()
        from meshpool.inference.pipeline import Pipeline
        pipe = Pipeline(cfg, build_stages(cfg, 2))
        await pipe.forward([1, 2, 3], pos_offset=0, session="s")
        pipe.drop_session("s")
        return pipe

    pipe = asyncio.run(run())
    assert "s" not in pipe.token_journal


def test_stage0_rebuild_matches_a_never_broken_cache():
    """Rebuilt-from-journal must equal never-lost, exactly."""
    async def run():
        cfg = M.ModelConfig()
        from meshpool.inference.pipeline import Pipeline
        pipe = Pipeline(cfg, build_stages(cfg, 3))
        await pipe.forward([1, 2, 3, 4], pos_offset=0, session="s")
        clean = await pipe.forward([5], pos_offset=4, session="s")

        pipe2 = Pipeline(cfg, build_stages(cfg, 3))
        await pipe2.forward([1, 2, 3, 4], pos_offset=0, session="s")
        pipe2.stages[0].drop("s")
        await pipe2.rehydrate_stage("s", 0, 4)
        rebuilt = await pipe2.forward([5], pos_offset=4, session="s")
        return np.asarray(clean), np.asarray(rebuilt)

    clean, rebuilt = asyncio.run(run())
    assert np.array_equal(clean, rebuilt), (
        "a stage-0 cache rebuilt from the token journal did not reproduce the "
        "logits of a cache that was never lost")


# --------------------------------------------------------------------------- #
# A standby may have to share a machine with another stage of the same model
# --------------------------------------------------------------------------- #

class KeyedNodeTransport:
    """A mesh whose nodes key their shards by ``model_id``, exactly as a real one does.

    ``agent/daemon.py`` stores ``state.shards[model_id] = shard`` -- one entry per
    model per machine. That makes co-hosting a standby dangerous in a way the
    node-keyed fake above cannot show: loading stage j's standby onto the machine
    already serving stage k, under the same model id, silently REPLACES stage k's
    shard. An attempt to add redundancy would take the deployment down.
    """

    def __init__(self, cfg: M.ModelConfig, params: dict) -> None:
        self.cfg, self.params = cfg, params
        self.shards: dict[tuple[str, str], CachedShard] = {}
        self.dead: set[str] = set()

    def load(self, node: str, model_id: str, start: int, end: int,
             first: bool, last: bool) -> None:
        key = (node, model_id)
        assert key not in self.shards, f"{node} already holds a shard for {model_id!r}"
        self.shards[key] = CachedShard(self.cfg, start, end, first, last, self.params)

    def kill(self, node: str) -> None:
        self.dead.add(node)

    async def call(self, node_id: str, op: str, payload: dict, timeout: float = 60.0):
        if node_id in self.dead:
            raise TransportError(f"node {node_id} is gone", status=503)
        shard = self.shards.get((node_id, payload["model_id"]))
        if shard is None:
            raise TransportError("shard not loaded on this node", status=404)
        if op == "shard/drop":
            shard.drop(payload["session"])
            return {"ok": True}
        if op == "shard/forward":
            session = payload.get("session")
            if session is not None and "truncate_to" in payload:
                shard.truncate(session, int(payload["truncate_to"]))
            data = payload["data"]
            if not payload["is_tokens"]:
                data = np.asarray(data, dtype=np.float32)
            out = shard.forward(data, payload["is_tokens"],
                                int(payload.get("pos_offset", 0)), session)
            return {"data": np.asarray(out, dtype=np.float32).tolist()}
        raise TransportError(f"unsupported op {op}", status=404)


def _cohosted_mesh():
    """Two machines, two stages, and each machine holds the other's standby.

    This is the shape a fully-packed mesh actually has: the planner spreads the
    model over every node it has, so there is no idle machine left to put a spare
    on. Redundancy has to be co-hosted or it does not exist.
    """
    from meshpool.inference.remote import spare_model_id
    cfg = M.ModelConfig()
    params, _ = M.load_params(cfg)
    mid = "m"
    half = cfg.n_layer // 2
    spans = [(0, half, True, False), (half, cfg.n_layer, False, True)]
    t = KeyedNodeTransport(cfg, params)
    t.load("A", mid, *spans[0])          # A: primary stage 0
    t.load("B", mid, *spans[1])          # B: primary stage 1
    # ...and each machine also holds the *other* stage as a standby, under a
    # stage-qualified id so it cannot collide with the primary it already serves.
    t.load("B", spare_model_id(mid, spans[0][0], spans[0][1]), *spans[0])
    t.load("A", spare_model_id(mid, spans[1][0], spans[1][1]), *spans[1])
    stages = [
        RemoteShard(t, "A", mid, *spans[0], spares=["B"],
                    spare_model_id=spare_model_id(mid, spans[0][0], spans[0][1])),
        RemoteShard(t, "B", mid, *spans[1], spares=["A"],
                    spare_model_id=spare_model_id(mid, spans[1][0], spans[1][1])),
    ]
    return cfg, t, RemotePipeline(cfg, stages)


def test_a_cohosted_standby_does_not_displace_the_stage_already_there():
    """Loading B's standby copy of stage 0 must not evict B's own stage 1."""
    cfg, t, pipe = _cohosted_mesh()
    keys = sorted(t.shards)
    assert len(keys) == 4, f"expected 4 distinct shards, got {keys}"
    assert ("B", "m") in t.shards and ("A", "m") in t.shards, (
        "the primaries must still be addressable by the plain model id")


def test_a_fully_packed_mesh_still_survives_a_node_death_midstream():
    """The point of co-hosting: no idle machine, yet the answer is unchanged."""
    async def run():
        prompt = [3, 1, 4, 1, 5]
        _, _, clean_pipe = _cohosted_mesh()
        clean = await _decode(clean_pipe, prompt, 8, "s")

        _, t, pipe = _cohosted_mesh()
        after = await _decode(pipe, prompt, 8, "s", kill_at=3,
                              killer=lambda: t.kill("B"))
        return clean, after, pipe

    clean, after, pipe = asyncio.run(run())
    assert after == clean, (
        f"co-hosted failover changed the answer: {after} != {clean}")
    assert pipe.failovers == 1, f"expected exactly one failover, got {pipe.failovers}"
    assert pipe.hydrations >= 1, "the standby was never rebuilt"


def test_the_standby_is_addressed_by_its_own_stage_qualified_id():
    """Non-vacuity: prove the failover really talked to the qualified shard.

    Without this the two tests above would pass just as happily if the standby had
    been loaded under the plain model id -- on a real node that load would have
    destroyed the primary, and the fake would be hiding it.
    """
    from meshpool.inference.remote import spare_model_id
    cfg = M.ModelConfig()
    half = cfg.n_layer // 2
    qualified = spare_model_id("m", half, cfg.n_layer)

    async def run():
        _, t, pipe = _cohosted_mesh()
        seen: list[tuple[str, str]] = []
        inner = t.call

        async def spy(node_id, op, payload, timeout=60.0):
            seen.append((node_id, payload.get("model_id", "")))
            return await inner(node_id, op, payload, timeout)

        t.call = spy  # type: ignore[method-assign]
        await _decode(pipe, [3, 1, 4, 1, 5], 8, "s", kill_at=3,
                      killer=lambda: t.kill("B"))
        return seen

    seen = asyncio.run(run())
    assert ("A", qualified) in seen, (
        f"stage 1 never addressed its standby by the qualified id {qualified!r}; "
        f"saw {sorted(set(seen))}")
    assert ("A", "m") in seen, "stage 0's primary should still use the plain id"
