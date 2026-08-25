"""The remote pipeline must behave exactly like the local one.

:mod:`meshpool.inference.remote` is the seam between the algorithms proven in the sim
and real machines on the internet. If it is not faithful, every guarantee established
by ``verify_spd_equivalence.py`` stops applying in production — so these tests drive
the *real* remote code path (including the JSON serialisation round trip) against an
in-process transport, and demand identical tokens.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from meshpool.inference import model as M
from meshpool.inference.pipeline import (LinkProfile, Pipeline, build_stages,
                                         generate_baseline)
from meshpool.inference.remote import (RemotePipeline, RemoteShard,
                                       build_remote_stages)
from meshpool.inference.spd import SelfDraft, speculative_generate

PROMPT = [5, 9, 33, 41, 7]


@pytest.fixture(scope="module")
def cfg_params():
    cfg = M.REGISTRY["nano-gpt"]
    P, _ = M.load_params(cfg)
    return cfg, P


class _Assign:
    """Stands in for a ModelDeployment partition entry."""

    def __init__(self, node_id, start, end):
        self.node_id, self.start_layer, self.end_layer = node_id, start, end


class FakeTransport:
    """Routes shard calls to real CachedShards, recording every request.

    Deliberately goes through ``tolist()`` / ``np.asarray`` exactly as the wire does,
    so serialisation loss would show up as a test failure rather than in production.
    """

    def __init__(self, cfg, params, depth):
        self.shards = {f"n{i}": s
                       for i, s in enumerate(build_stages(cfg, depth, params))}
        self.calls: list[tuple[str, str, dict]] = []

    async def call(self, node_id, path, payload, timeout=None):
        self.calls.append((node_id, path, payload))
        shard = self.shards[node_id]
        if path == "shard/drop":
            shard.drop(payload["session"])
            return {"dropped": payload["session"]}
        if path != "shard/forward":
            raise AssertionError(f"unexpected path {path}")
        session = payload.get("session") if payload.get("use_cache") else None
        if payload.get("truncate_to") is not None and payload.get("session"):
            shard.truncate(payload["session"], int(payload["truncate_to"]))
        out = shard.forward(payload["data"], payload["is_tokens"],
                            int(payload.get("pos_offset", 0)), session)
        return {"data": out.tolist()}

    def paths(self, path):
        return [c for c in self.calls if c[1] == path]


def _remote_pipe(cfg, params, depth):
    t = FakeTransport(cfg, params, depth)
    partition = [_Assign(f"n{i}", s.start, s.end)
                 for i, s in enumerate(t.shards.values())]
    return RemotePipeline(cfg, build_remote_stages(t, "m1", partition)), t


def _local_pipe(cfg, params, depth):
    return Pipeline(cfg, build_stages(cfg, depth, params), LinkProfile())


# --------------------------------------------------------------------------- #
# Fidelity: remote must equal local
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("depth", [1, 2, 3, 4, 8])
def test_remote_baseline_matches_local(cfg_params, depth):
    cfg, P = cfg_params
    local, _ = asyncio.run(generate_baseline(_local_pipe(cfg, P, depth), PROMPT, 20,
                                             temperature=0.0, seed=3))
    pipe, _ = _remote_pipe(cfg, P, depth)
    remote, _ = asyncio.run(generate_baseline(pipe, PROMPT, 20, temperature=0.0, seed=3))
    assert remote == local


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_remote_spd_matches_local_baseline(cfg_params, depth):
    """The headline guarantee, across the real remote code path."""
    cfg, P = cfg_params
    local, _ = asyncio.run(generate_baseline(_local_pipe(cfg, P, depth), PROMPT, 24,
                                             temperature=0.0, seed=3))
    pipe, _ = _remote_pipe(cfg, P, depth)
    remote, _, spec = asyncio.run(speculative_generate(
        pipe, SelfDraft(cfg, P, 2), PROMPT, 24, k=4, temperature=0.0, seed=3))
    assert remote == local
    assert spec.tokens_per_traversal > 1.0


def test_remote_forward_matches_single_process_logits(cfg_params):
    cfg, P = cfg_params
    reference = M.forward_full(P, cfg, PROMPT)
    pipe, _ = _remote_pipe(cfg, P, 3)
    got = asyncio.run(pipe.forward(PROMPT, 0, session="s"))
    assert np.abs(reference - got).max() < 1e-4


# --------------------------------------------------------------------------- #
# Rollback must not cost a round trip
# --------------------------------------------------------------------------- #
def test_truncate_sends_nothing_and_rides_the_next_forward(cfg_params):
    """A separate rollback request would cost (N-1)*RTT per speculation round.

    On a wide-area mesh that is precisely the cost speculation exists to remove, so
    the rollback must be folded into the next traversal instead.
    """
    cfg, P = cfg_params
    pipe, t = _remote_pipe(cfg, P, 3)
    asyncio.run(pipe.forward(PROMPT, 0, session="s"))
    before = len(t.calls)

    pipe.truncate_session("s", 3)
    assert len(t.calls) == before, "truncate must not make a network call of its own"

    asyncio.run(pipe.forward([7], pos_offset=3, session="s"))
    carried = [c for c in t.calls[before:] if c[2].get("truncate_to") == 3]
    assert len(carried) == pipe.depth, "every stage must be told to roll back"


def test_spd_makes_no_extra_round_trips_for_rollback(cfg_params):
    """Total requests must equal traversals x depth — nothing else on the wire."""
    cfg, P = cfg_params
    pipe, t = _remote_pipe(cfg, P, 3)
    _, stats, _ = asyncio.run(speculative_generate(
        pipe, SelfDraft(cfg, P, 2), PROMPT, 24, k=4, temperature=0.0, seed=3))
    forwards = t.paths("shard/forward")
    assert len(forwards) == stats.traversals * pipe.depth
    assert not t.paths("shard/drop"), "no cleanup should happen mid-generation"


def test_flush_releases_remote_sessions(cfg_params):
    cfg, P = cfg_params
    pipe, t = _remote_pipe(cfg, P, 3)
    asyncio.run(pipe.forward(PROMPT, 0, session="s"))
    pipe.drop_session("s")
    assert not t.paths("shard/drop"), "drop is deferred off the critical path"
    asyncio.run(pipe.flush())
    assert len(t.paths("shard/drop")) == pipe.depth


def test_flush_survives_a_dead_node(cfg_params):
    """Cleanup must never fail a generation that already succeeded."""
    cfg, P = cfg_params
    pipe, t = _remote_pipe(cfg, P, 2)

    async def boom(node_id, path, payload, timeout=None):
        raise ConnectionError("node vanished")

    asyncio.run(pipe.forward(PROMPT, 0, session="s"))
    pipe.drop_session("s")
    for stage in pipe.stages:
        stage.transport = type("T", (), {"call": staticmethod(boom)})()
    asyncio.run(pipe.flush())   # must not raise


# --------------------------------------------------------------------------- #
# Wire cost
# --------------------------------------------------------------------------- #
def test_decode_sends_one_activation_per_hop_not_the_whole_context(cfg_params):
    """The KV cache is what makes wide-area pipelining affordable at all."""
    cfg, P = cfg_params
    pipe, t = _remote_pipe(cfg, P, 3)
    asyncio.run(pipe.forward(PROMPT, 0, session="s"))

    sizes = []
    for pos in range(len(PROMPT), len(PROMPT) + 4):
        before = len(t.calls)
        asyncio.run(pipe.forward([7], pos_offset=pos, session="s"))
        sent = [c for c in t.calls[before:] if not c[2]["is_tokens"]]
        sizes.append(sum(len(c[2]["data"]) for c in sent))
    assert len(set(sizes)) == 1, f"decode payload must not grow with context: {sizes}"


def test_only_the_first_stage_receives_tokens(cfg_params):
    cfg, P = cfg_params
    pipe, t = _remote_pipe(cfg, P, 3)
    asyncio.run(pipe.forward(PROMPT, 0, session="s"))
    tok_calls = [c for c in t.paths("shard/forward") if c[2]["is_tokens"]]
    assert [c[0] for c in tok_calls] == ["n0"]


def test_stateless_mode_sends_no_session_fields(cfg_params):
    """Backwards compatibility: session=None must look like the original protocol."""
    cfg, P = cfg_params
    pipe, t = _remote_pipe(cfg, P, 2)
    asyncio.run(pipe.forward(PROMPT, 0, session=None))
    for _, _, payload in t.paths("shard/forward"):
        assert "session" not in payload and "use_cache" not in payload


def test_build_remote_stages_marks_first_and_last(cfg_params):
    cfg, P = cfg_params
    t = FakeTransport(cfg, P, 3)
    partition = [_Assign("n2", 5, 8), _Assign("n0", 0, 3), _Assign("n1", 3, 5)]
    stages = build_remote_stages(t, "m1", partition)
    assert [s.node_id for s in stages] == ["n0", "n1", "n2"], "must sort by layer"
    assert stages[0].is_first and stages[-1].is_last
    assert not stages[1].is_first and not stages[1].is_last


def test_build_remote_stages_rejects_empty_partition(cfg_params):
    cfg, P = cfg_params
    with pytest.raises(ValueError):
        build_remote_stages(FakeTransport(cfg, P, 2), "m1", [])
