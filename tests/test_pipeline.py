"""Pipeline traversal accounting, sharding equivalence, and chunked prefill."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from meshpool.inference import model as M
from meshpool.inference.pipeline import (LinkProfile, Pipeline, build_stages,
                                         dist_from_logits, generate_baseline, prefill)

TOL = 1e-4
PROMPT = [5, 9, 33, 41, 7]


@pytest.fixture(scope="module")
def cfg_params():
    cfg = M.REGISTRY["nano-gpt"]
    P, _ = M.load_params(cfg)
    return cfg, P


def _pipe(cfg, P, depth, rtt=0.0):
    return Pipeline(cfg, build_stages(cfg, depth, P), LinkProfile(rtt_ms=rtt))


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 8])
def test_sharded_forward_matches_single_process(cfg_params, depth):
    """Splitting layers across stages must not change the numbers."""
    cfg, P = cfg_params
    reference = M.forward_full(P, cfg, PROMPT)
    got = asyncio.run(_pipe(cfg, P, depth).forward(PROMPT, 0, session="s"))
    assert np.abs(reference - got).max() < TOL


def test_stage_partition_covers_every_layer_exactly_once(cfg_params):
    cfg, P = cfg_params
    for depth in range(1, cfg.n_layer + 1):
        stages = build_stages(cfg, depth, P)
        covered = [i for s in stages for i in range(s.start, s.end)]
        assert covered == list(range(cfg.n_layer))
        assert stages[0].is_first and stages[-1].is_last


def test_no_stage_holds_the_whole_checkpoint(cfg_params):
    """The property that lets a model exceed any single machine."""
    cfg, P = cfg_params
    stages = build_stages(cfg, 4, P)
    full_keys = set(P)
    for s in stages:
        assert set(s.P) < full_keys, "a stage must hold a strict subset of parameters"


def test_prefill_is_one_traversal_regardless_of_prompt_length(cfg_params):
    """Position-parallel prefill: T positions ride in a single traversal.

    A frequent and serious modelling error is to assume prefill costs one traversal
    per prompt token. It does not.
    """
    cfg, P = cfg_params
    for n in (1, 5, 20, 40):
        pipe = _pipe(cfg, P, 3)
        asyncio.run(prefill(pipe, list(range(1, n + 1)), session="p"))
        assert pipe.stats.traversals == 1
        assert pipe.stats.positions_processed == n


@pytest.mark.parametrize("chunk", [1, 2, 4])
def test_chunked_prefill_matches_one_shot(cfg_params, chunk):
    cfg, P = cfg_params
    tokens = [5, 9, 33, 41, 7, 60, 12, 88]
    one = asyncio.run(prefill(_pipe(cfg, P, 3), tokens, session="a"))
    many = asyncio.run(prefill(_pipe(cfg, P, 3), tokens, session="b", chunk=chunk))
    # one-shot returns all rows; chunked returns the final chunk's rows
    assert np.abs(one[-many.shape[0]:] - many).max() < TOL


def test_chunked_prefill_costs_more_traversals_but_bounds_memory(cfg_params):
    cfg, P = cfg_params
    tokens = list(range(1, 17))
    p1 = _pipe(cfg, P, 3)
    asyncio.run(prefill(p1, tokens, session="a"))
    p2 = _pipe(cfg, P, 3)
    asyncio.run(prefill(p2, tokens, session="b", chunk=4))
    assert p1.stats.traversals == 1
    assert p2.stats.traversals == 4


def test_baseline_costs_one_traversal_per_token(cfg_params):
    cfg, P = cfg_params
    pipe = _pipe(cfg, P, 3)
    out, stats = asyncio.run(generate_baseline(pipe, PROMPT, 10, temperature=0.0))
    assert len(out) == 10
    assert stats.traversals == 11, "1 prefill + 1 per generated token"


def test_decode_wire_bytes_are_constant_per_token(cfg_params):
    """With a KV cache a decode hop carries 2*d_model bytes, not T*d_model.

    This is the property that makes wide-area pipeline parallelism viable at all.
    """
    cfg, P = cfg_params
    pipe = _pipe(cfg, P, 3)
    asyncio.run(prefill(pipe, PROMPT, session="s"))
    per_token = []
    for pos in range(len(PROMPT), len(PROMPT) + 5):
        before = pipe.stats.wire_bytes
        asyncio.run(pipe.forward([7], pos_offset=pos, session="s"))
        per_token.append(pipe.stats.wire_bytes - before)
    assert len(set(per_token)) == 1, f"decode wire cost must be constant, got {per_token}"
    hops = pipe.depth - 1
    assert per_token[0] == hops * cfg.d_model * 4  # float32 activations


def test_link_profile_injects_latency(cfg_params):
    cfg, P = cfg_params
    pipe = _pipe(cfg, P, 3, rtt=40.0)
    asyncio.run(pipe.forward(PROMPT, 0, session="s"))
    # 3 stages => 2 hops => 2 * rtt/2 = one full RTT of modelled delay
    assert pipe.stats.link_seconds == pytest.approx(0.040, abs=1e-6)


def test_truncate_session_rolls_back_every_stage(cfg_params):
    cfg, P = cfg_params
    pipe = _pipe(cfg, P, 3)
    asyncio.run(pipe.forward(PROMPT, 0, session="s"))
    assert all(s.cache("s").length == len(PROMPT) for s in pipe.stages)
    pipe.truncate_session("s", 2)
    assert all(s.cache("s").length == 2 for s in pipe.stages)


def test_greedy_dist_is_one_hot():
    logits = np.array([0.1, 5.0, 0.3], dtype=np.float32)
    p = dist_from_logits(logits, temperature=0.0)
    assert p[1] == 1.0 and p.sum() == 1.0


def test_top_k_truncates_support():
    logits = np.arange(10, dtype=np.float32)
    p = dist_from_logits(logits, temperature=1.0, top_k=3)
    assert int((p > 0).sum()) == 3
    assert p.sum() == pytest.approx(1.0)


def test_build_stages_rejects_impossible_splits(cfg_params):
    cfg, P = cfg_params
    with pytest.raises(ValueError):
        build_stages(cfg, 0, P)
    with pytest.raises(ValueError):
        build_stages(cfg, cfg.n_layer + 1, P)
