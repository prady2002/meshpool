"""Key/value attention cache — the single change that makes generation O(T) not O(T^2).

Why this file exists
--------------------
The original generation loop (a full pipeline sweep per token, plus
``model.generate_with``) re-ran a **full forward pass over the entire context for
every generated token**.
That is quadratic in sequence length and it also meant every pipeline hop shipped a
``[T, d_model]`` activation tensor per token instead of ``[1, d_model]``.

Both consequences matter enormously for a WAN pipeline:

* compute per token grows with the context, so long chats get slower and slower;
* bytes per hop grow with the context, when the whole reason pipeline parallelism is
  viable over the internet is that a decode step only needs ``2 * d_model`` bytes
  (16 KB for Llama-70B, ~14 KB for a Kimi-K2-shaped model).

With a KV cache a decode step feeds each stage only the *new* position, so the wire
cost becomes constant and the compute cost becomes linear.

Positional embeddings and cache validity
----------------------------------------
This model uses **learned absolute** position embeddings (``wpe``). That has a real
consequence which is easy to get wrong: if the context window slides (once the
sequence exceeds ``block_size`` the old generation loop kept only the last
``block_size`` tokens), then *every* retained token changes its position index, every
cached key/value is stale, and the cache MUST be rebuilt. We detect that explicitly
rather than silently returning wrong numbers.

Production models use rotary embeddings (RoPE), which are relative and do not suffer
from this; when the llama.cpp/GGUF backend lands it will not need the rebuild path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import model as M


@dataclass
class LayerCache:
    """Cached keys/values for one decoder layer, shaped ``[n_head, T, head_dim]``."""

    k: Optional[np.ndarray] = None
    v: Optional[np.ndarray] = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else int(self.k.shape[1])

    def append(self, k_new: np.ndarray, v_new: np.ndarray) -> None:
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = np.concatenate([self.k, k_new], axis=1)
            self.v = np.concatenate([self.v, v_new], axis=1)

    def truncate(self, length: int) -> None:
        """Drop everything at or beyond absolute position ``length``.

        Used to roll back rejected speculative tokens: the verify pass writes K+1
        positions into the cache, and whatever the rejection sampler declines has to
        disappear as if it was never computed.
        """
        if self.k is None:
            return
        if length <= 0:
            self.k = self.v = None
        elif length < self.k.shape[1]:
            self.k = self.k[:, :length, :]
            self.v = self.v[:, :length, :]

    def nbytes(self) -> int:
        return 0 if self.k is None else int(self.k.nbytes + self.v.nbytes)


@dataclass
class KVCache:
    """A generation session's cache for one pipeline stage (layers ``[start, end)``).

    ``window_start`` is the absolute position of the token stored at cache index 0.
    Any change to it invalidates the whole cache (see the module docstring).
    """

    start_layer: int
    end_layer: int
    window_start: int = 0
    layers: dict[int, LayerCache] = field(default_factory=dict)

    def layer(self, i: int) -> LayerCache:
        lc = self.layers.get(i)
        if lc is None:
            lc = self.layers[i] = LayerCache()
        return lc

    @property
    def length(self) -> int:
        """Number of cached positions (identical across layers by construction)."""
        for i in range(self.start_layer, self.end_layer):
            lc = self.layers.get(i)
            if lc is not None:
                return lc.length
        return 0

    def reset(self, window_start: int = 0) -> None:
        self.layers.clear()
        self.window_start = window_start

    def truncate(self, length: int) -> None:
        for lc in self.layers.values():
            lc.truncate(length)

    def nbytes(self) -> int:
        return sum(lc.nbytes() for lc in self.layers.values())

    def accepts(self, window_start: int, pos_offset: int) -> bool:
        """Whether an incoming chunk at ``pos_offset`` can extend this cache directly.

        It can only do so when the window has not slid and the chunk starts exactly
        where the cache ends — anything else means we are looking at a different
        sequence and must rebuild.
        """
        return window_start == self.window_start and pos_offset == self.length


def block_cached(P: dict, cfg: M.ModelConfig, i: int, x: np.ndarray,
                 cache: Optional[LayerCache]) -> np.ndarray:
    """One decoder layer over ``x`` (``[T_new, D]``), attending over cache + ``x``.

    Mathematically identical to :func:`model._block` restricted to the new rows: the
    only difference is that the keys/values for earlier positions are read from the
    cache instead of being recomputed. When ``cache`` is None this degenerates to the
    uncached path exactly.

    Mutates ``cache`` in place, appending the new keys/values.
    """
    if cfg.architecture == "llama":
        return llama_block_cached(P, cfg, i, x, cache)

    T_new, D = x.shape
    nh, hd = cfg.n_head, cfg.head_dim

    a = M._layernorm(x, P[f"h{i}.ln1.g"], P[f"h{i}.ln1.b"])
    qkv = a @ P[f"h{i}.attn.Wqkv"]
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(T_new, nh, hd).transpose(1, 0, 2)     # (nh, T_new, hd)
    k = k.reshape(T_new, nh, hd).transpose(1, 0, 2)
    v = v.reshape(T_new, nh, hd).transpose(1, 0, 2)

    if cache is not None:
        T_past = cache.length
        cache.append(k, v)
        k_all, v_all = cache.k, cache.v
    else:
        T_past = 0
        k_all, v_all = k, v

    att = (q @ k_all.transpose(0, 2, 1)) / math.sqrt(hd)  # (nh, T_new, T_past+T_new)
    # Row i is absolute position (T_past + i) and may attend to every column j with
    # j <= T_past + i. np.triu(..., k) keeps j - i >= k, so k = T_past + 1 masks
    # exactly the strictly-future columns.
    mask = np.triu(np.ones((T_new, k_all.shape[1]), np.float32), k=T_past + 1) * -1e9
    att = M._softmax(att + mask, axis=-1)

    out = (att @ v_all).transpose(1, 0, 2).reshape(T_new, D)
    x = x + out @ P[f"h{i}.attn.Wo"]
    b = M._layernorm(x, P[f"h{i}.ln2.g"], P[f"h{i}.ln2.b"])
    h = M._gelu(b @ P[f"h{i}.mlp.W1"] + P[f"h{i}.mlp.b1"])
    x = x + h @ P[f"h{i}.mlp.W2"] + P[f"h{i}.mlp.b2"]
    return x.astype(np.float32)


def llama_block_cached(P: dict, cfg: M.ModelConfig, i: int, x: np.ndarray,
                       cache: Optional[LayerCache]) -> np.ndarray:
    """Cached Llama/Qwen2 decoder block: RMSNorm + RoPE + GQA + SwiGLU."""
    T_new, D = x.shape
    nh, nkh, hd = cfg.n_head, cfg.kv_heads, cfg.head_dim
    repeats = nh // nkh

    a = M._rmsnorm(x, P[f"h{i}.rms1.g"], cfg.rms_norm_eps)
    q = (a @ P[f"h{i}.attn.Wq"]).reshape(T_new, nh, hd).transpose(1, 0, 2)
    k = (a @ P[f"h{i}.attn.Wk"]).reshape(T_new, nkh, hd).transpose(1, 0, 2)
    v = (a @ P[f"h{i}.attn.Wv"]).reshape(T_new, nkh, hd).transpose(1, 0, 2)
    T_past = cache.length if cache is not None else 0
    q, k = M._rope(q, k, T_past, cfg.rope_theta)

    if cache is not None:
        cache.append(k, v)
        k_all, v_all = cache.k, cache.v
    else:
        k_all, v_all = k, v
    k_rep = np.repeat(k_all, repeats, axis=0)
    v_rep = np.repeat(v_all, repeats, axis=0)
    att = (q @ k_rep.transpose(0, 2, 1)) / math.sqrt(hd)
    mask = np.triu(np.ones((T_new, k_all.shape[1]), np.float32), k=T_past + 1) * -1e9
    att = M._softmax(att + mask, axis=-1)
    out = (att @ v_rep).transpose(1, 0, 2).reshape(T_new, D)
    x = x + out @ P[f"h{i}.attn.Wo"]
    b = M._rmsnorm(x, P[f"h{i}.rms2.g"], cfg.rms_norm_eps)
    h = M._silu(b @ P[f"h{i}.mlp.Wgate"]) * (b @ P[f"h{i}.mlp.Wup"])
    x = x + h @ P[f"h{i}.mlp.Wdown"]
    return x.astype(np.float32)


def embed_tokens_at(P: dict, cfg: M.ModelConfig, tokens: list[int],
                    pos_offset: int) -> np.ndarray:
    """Embed ``tokens`` as if they occupied absolute positions ``pos_offset...``.

    :func:`model.embed_tokens` hardcodes ``wpe[:T]``, which is only correct for a
    chunk that starts at position 0. Incremental decode and chunked prefill both feed
    chunks that start later, so they need this variant.
    """
    T = len(tokens)
    if pos_offset + T > cfg.block_size:
        raise ValueError(
            f"position {pos_offset + T} exceeds block_size {cfg.block_size}; "
            "the caller must slide the window and rebuild the cache")
    x = P["wte"][np.asarray(tokens, dtype=np.int64)]
    if cfg.architecture != "llama":
        x = x + P["wpe"][pos_offset:pos_offset + T]
    return x.astype(np.float32)


def run_layers_cached(P: dict, cfg: M.ModelConfig, x: np.ndarray,
                      start: int, end: int, cache: Optional[KVCache]) -> np.ndarray:
    for i in range(start, end):
        x = block_cached(P, cfg, i, x, cache.layer(i) if cache is not None else None)
    return x


def forward_full_cached(P: dict, cfg: M.ModelConfig, tokens: list[int],
                        cache: Optional[KVCache] = None,
                        pos_offset: int = 0) -> np.ndarray:
    """Single-process forward over ``tokens`` at ``pos_offset``, returning logits.

    Returns ``[len(tokens), vocab]`` — logits for the *new* positions only, which is
    all any caller needs: position ``j`` predicts token ``j+1``.
    """
    x = embed_tokens_at(P, cfg, tokens, pos_offset)
    x = run_layers_cached(P, cfg, x, 0, cfg.n_layer, cache)
    return M.head(P, cfg, x)
