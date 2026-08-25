"""KV cache correctness.

The cache is only useful if it is *exactly* equivalent to recomputing the full
forward pass. These tests assert that equivalence directly, because a subtly wrong
cache produces plausible-looking text and is very hard to notice downstream.
"""
from __future__ import annotations

import numpy as np
import pytest

from meshpool.inference import kvcache as KV
from meshpool.inference import model as M

# float32 matmul is not associative, so cached and uncached paths accumulate in a
# different order. 1e-4 is far below the gap between competing logits in practice.
TOL = 1e-4


@pytest.fixture(scope="module")
def cfg_params():
    cfg = M.REGISTRY["nano-gpt"]
    P, _ = M.load_params(cfg)
    return cfg, P


TOKENS = [5, 9, 33, 41, 7, 60, 12, 88, 2, 71]


def test_uncached_path_is_bit_identical_to_original(cfg_params):
    cfg, P = cfg_params
    a = M.forward_full(P, cfg, TOKENS)
    b = KV.forward_full_cached(P, cfg, TOKENS, cache=None, pos_offset=0)
    assert np.array_equal(a, b), "cache-free path must not change the math at all"


def test_incremental_decode_matches_full_recompute(cfg_params):
    cfg, P = cfg_params
    reference = M.forward_full(P, cfg, TOKENS)
    cache = KV.KVCache(0, cfg.n_layer)
    rows = [KV.forward_full_cached(P, cfg, [t], cache=cache, pos_offset=i)[-1]
            for i, t in enumerate(TOKENS)]
    assert np.abs(reference - np.stack(rows)).max() < TOL


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 7])
def test_chunked_prefill_matches_one_shot(cfg_params, chunk):
    """Prefill is position-parallel: chunking must not change the result."""
    cfg, P = cfg_params
    reference = M.forward_full(P, cfg, TOKENS)
    cache = KV.KVCache(0, cfg.n_layer)
    pieces = [KV.forward_full_cached(P, cfg, TOKENS[i:i + chunk], cache=cache, pos_offset=i)
              for i in range(0, len(TOKENS), chunk)]
    assert np.abs(reference - np.concatenate(pieces, axis=0)).max() < TOL


def test_truncate_rolls_back_exactly(cfg_params):
    """Rejected speculative tokens must leave no trace."""
    cfg, P = cfg_params
    reference = M.forward_full(P, cfg, TOKENS)
    cache = KV.KVCache(0, cfg.n_layer)
    KV.forward_full_cached(P, cfg, TOKENS, cache=cache, pos_offset=0)
    cache.truncate(4)
    assert cache.length == 4
    resumed = KV.forward_full_cached(P, cfg, TOKENS[4:], cache=cache, pos_offset=4)
    assert np.abs(reference[4:] - resumed).max() < TOL


def test_truncate_to_zero_clears(cfg_params):
    cfg, P = cfg_params
    cache = KV.KVCache(0, cfg.n_layer)
    KV.forward_full_cached(P, cfg, TOKENS, cache=cache, pos_offset=0)
    assert cache.length == len(TOKENS)
    cache.truncate(0)
    assert cache.length == 0
    assert cache.nbytes() == 0


def test_cache_grows_linearly_not_quadratically(cfg_params):
    cfg, P = cfg_params
    cache = KV.KVCache(0, cfg.n_layer)
    sizes = []
    for i, t in enumerate(TOKENS[:8]):
        KV.forward_full_cached(P, cfg, [t], cache=cache, pos_offset=i)
        sizes.append(cache.nbytes())
    deltas = np.diff(sizes)
    assert np.allclose(deltas, deltas[0]), "per-token cache growth must be constant"


def test_position_beyond_block_size_is_rejected(cfg_params):
    """Absolute position embeddings make out-of-window positions meaningless."""
    cfg, P = cfg_params
    with pytest.raises(ValueError, match="block_size"):
        KV.embed_tokens_at(P, cfg, [1, 2], pos_offset=cfg.block_size)


def test_accepts_detects_window_slide(cfg_params):
    cfg, _ = cfg_params
    cache = KV.KVCache(0, cfg.n_layer)
    assert cache.accepts(window_start=0, pos_offset=0)
    assert not cache.accepts(window_start=0, pos_offset=3)
    assert not cache.accepts(window_start=1, pos_offset=0)
