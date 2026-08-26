"""KV cache correctness.

The cache is only useful if it is *exactly* equivalent to recomputing the full
forward pass. These tests assert that equivalence directly, because a subtly wrong
cache produces plausible-looking text and is very hard to notice downstream.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from gguf_fixtures import build_llama_gguf
from meshpool.inference import gguf_loader as G
from meshpool.inference import kvcache as KV
from meshpool.inference import model as M
from meshpool.inference.pipeline import Pipeline, build_stages, generate_baseline, prefill

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


def test_layer_cache_is_paged_and_reuses_pages_after_truncate():
    cache = KV.LayerCache(page_size=3)
    k = np.arange(2 * 7 * 4, dtype=np.float32).reshape(2, 7, 4)
    v = -k
    cache.append(k[:, :5, :], v[:, :5, :])
    cache.append(k[:, 5:, :], v[:, 5:, :])
    assert cache.length == 7
    assert cache.page_count == 3
    assert np.array_equal(cache.k, k)
    cache.truncate(4)
    assert cache.page_count == 3
    cache.append(k[:, 4:6, :], v[:, 4:6, :])
    assert cache.page_count == 3
    assert np.array_equal(cache.k, k[:, :6, :])


def test_paged_boundary_crossing_matches_full_recompute(cfg_params):
    cfg, P = cfg_params
    reference = M.forward_full(P, cfg, TOKENS)
    cache = KV.KVCache(0, cfg.n_layer, page_size=3)
    rows = [KV.forward_full_cached(P, cfg, [t], cache=cache, pos_offset=i)[-1]
            for i, t in enumerate(TOKENS)]
    assert all(cache.layer(i).page_count >= 4 for i in range(cfg.n_layer))
    assert np.abs(reference - np.stack(rows)).max() < TOL


def test_stage_cache_contains_only_owned_layers(cfg_params):
    cfg, P = cfg_params
    cache = KV.KVCache(2, 5, page_size=2)
    x = np.random.default_rng(11).standard_normal((5, cfg.d_model)).astype(np.float32)
    KV.run_layers_cached(P, cfg, x, 2, 5, cache)
    assert sorted(cache.layers) == [2, 3, 4]
    assert all(2 <= i < 5 for i in cache.layers)


def _rope_ref(q: np.ndarray, k: np.ndarray, pos_offset: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    hd = q.shape[-1]
    half = hd // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
    pos = (np.arange(q.shape[1], dtype=np.float32) + float(pos_offset))[:, None]
    freqs = pos * inv[None, :]
    cos, sin = np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)

    def apply(x):
        first = x[..., :half]
        second = x[..., half:2 * half]
        rotated = np.concatenate(
            [first * cos[None, :, :] - second * sin[None, :, :],
             first * sin[None, :, :] + second * cos[None, :, :]], axis=-1)
        return np.concatenate([rotated, x[..., 2 * half:]], axis=-1) if hd % 2 else rotated

    return apply(q).astype(np.float32), apply(k).astype(np.float32)


def _rms_ref(x: np.ndarray, g: np.ndarray, eps: float) -> np.ndarray:
    return (x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * g).astype(np.float32)


def _softmax_ref(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _llama_block_last_ref(P: dict, cfg: M.ModelConfig, layer: int, x_full: np.ndarray) -> np.ndarray:
    import math
    t = x_full.shape[0]
    nh, nkh, hd = cfg.n_head, cfg.kv_heads, cfg.head_dim
    a = _rms_ref(x_full, P[f"h{layer}.rms1.g"], cfg.rms_norm_eps)
    q = (a @ P[f"h{layer}.attn.Wq"]).reshape(t, nh, hd).transpose(1, 0, 2)
    k = (a @ P[f"h{layer}.attn.Wk"]).reshape(t, nkh, hd).transpose(1, 0, 2)
    v = (a @ P[f"h{layer}.attn.Wv"]).reshape(t, nkh, hd).transpose(1, 0, 2)
    q, k = _rope_ref(q, k, 0, cfg.rope_theta)
    k_rep = np.repeat(k, nh // nkh, axis=0)
    v_rep = np.repeat(v, nh // nkh, axis=0)
    att = (q[:, -1:, :] @ k_rep.transpose(0, 2, 1)) / math.sqrt(hd)
    att = _softmax_ref(att, axis=-1)
    out = (att @ v_rep).transpose(1, 0, 2).reshape(1, cfg.d_model)
    x = x_full[-1:] + out @ P[f"h{layer}.attn.Wo"]
    b = _rms_ref(x, P[f"h{layer}.rms2.g"], cfg.rms_norm_eps)
    h = (b @ P[f"h{layer}.mlp.Wgate"])
    h = (h / (1.0 + np.exp(-h))) * (b @ P[f"h{layer}.mlp.Wup"])
    return (x + h @ P[f"h{layer}.mlp.Wdown"]).astype(np.float32)


def test_cached_llama_decode_matches_independent_attention_oracle():
    cfg = M.ModelConfig(name="oracle", vocab_size=32, d_model=16, n_head=4, n_kv_head=2,
                        n_layer=1, d_ff=32, architecture="llama", block_size=32,
                        seed=4242)
    P = M.random_params(cfg)
    tokens = [1, 5, 9, 13, 17]
    x_full = KV.embed_tokens_at(P, cfg, tokens, 0)
    cache = KV.KVCache(0, 1, page_size=2)
    for pos, token in enumerate(tokens[:-1]):
        x = KV.embed_tokens_at(P, cfg, [token], pos)
        KV.run_layers_cached(P, cfg, x, 0, 1, cache)
    x_last = KV.embed_tokens_at(P, cfg, [tokens[-1]], len(tokens) - 1)
    got = KV.run_layers_cached(P, cfg, x_last, 0, 1, cache)
    expected = _llama_block_last_ref(P, cfg, 0, x_full)
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_client_activation_cache_hydrates_failed_middle_stage(cfg_params):
    cfg, P = cfg_params
    prompt = TOKENS[:8]
    ref = Pipeline(cfg, build_stages(cfg, 3, P, cache_page_size=3))
    rec = Pipeline(cfg, build_stages(cfg, 3, P, cache_page_size=3))
    rec.activation_cache.page_size = 3
    session = "recover"
    asyncio.run(prefill(ref, prompt, session="ref"))
    asyncio.run(prefill(rec, prompt, session=session))
    assert rec.activation_cache.length(session, 1) == len(prompt)
    assert rec.activation_cache.page_count(session, 1) == 3
    rec.stages[1].drop(session)
    assert rec.stages[1].cache(session).length == 0
    asyncio.run(rec.hydrate_stage_from_activations(session, 1, len(prompt), chunk=3))
    assert rec.stages[1].cache(session).length == len(prompt)
    ref_logits = asyncio.run(ref.forward([7], pos_offset=len(prompt), session="ref"))
    rec_logits = asyncio.run(rec.forward([7], pos_offset=len(prompt), session=session))
    assert np.abs(ref_logits - rec_logits).max() < TOL


@pytest.fixture(scope="module")
def gguf_cached_fixture(tmp_path_factory):
    pytest.importorskip("gguf", reason="GGUF support needs the reference gguf package")
    p = str(tmp_path_factory.mktemp("gguf_kv") / "kv.gguf")
    build_llama_gguf(p, n_layer=4, d_model=32, n_head=4, n_kv_head=2,
                     d_ff=64, vocab_size=128, seed=20240820)
    info = G.read_info(p)
    return p, info, G.load_full_params(p, info)


@pytest.mark.parametrize("prompt", [
    [1, 5, 9],
    [3, 11, 27, 5, 19],
    [1, 5, 9, 13, 17, 21, 25],
])
def test_real_gguf_cached_generation_matches_uncached_across_shard_boundary(gguf_cached_fixture, prompt):
    _path, info, P = gguf_cached_fixture
    cfg = info.config
    uncached = M.generate_with(lambda ctx: M.forward_full(P, cfg, ctx), cfg,
                               prompt, max_new_tokens=6, temperature=0.0)
    pipe = Pipeline(cfg, build_stages(cfg, 2, P, cache_page_size=3))
    cached, stats = asyncio.run(generate_baseline(pipe, prompt, 6, temperature=0.0,
                                                  session=f"s{len(prompt)}"))
    assert cached == uncached
    assert len(set(cached)) > 1
    assert stats.traversals == 7
    assert all(s.cache(f"s{len(prompt)}").length == len(prompt) + 6 for s in pipe.stages)
