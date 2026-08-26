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
    """Paged cached keys/values for one decoder layer.

    Each page stores ``[n_head, page_size, head_dim]``. Appending new positions
    fills the current page before allocating another one, so a sequence grows in
    fixed-size blocks rather than by repeatedly reallocating one contiguous
    ``[n_head, T, head_dim]`` buffer. The ``k``/``v`` properties materialise the
    used prefix for the attention kernel; ownership and allocation stay paged.
    """

    page_size: int = 16
    _k_pages: list[np.ndarray] = field(default_factory=list)
    _v_pages: list[np.ndarray] = field(default_factory=list)
    _length: int = 0
    _shape: Optional[tuple[int, int]] = None

    @property
    def length(self) -> int:
        return self._length

    @property
    def page_count(self) -> int:
        return len(self._k_pages)

    @property
    def k(self) -> Optional[np.ndarray]:
        return self._materialize(self._k_pages)

    @property
    def v(self) -> Optional[np.ndarray]:
        return self._materialize(self._v_pages)

    def _materialize(self, pages: list[np.ndarray]) -> Optional[np.ndarray]:
        if self._length == 0:
            return None
        used = []
        left = self._length
        for p in pages:
            n = min(left, self.page_size)
            if n <= 0:
                break
            used.append(p[:, :n, :])
            left -= n
        return used[0].copy() if len(used) == 1 else np.concatenate(used, axis=1)

    def append(self, k_new: np.ndarray, v_new: np.ndarray) -> None:
        k_new = np.asarray(k_new, dtype=np.float32)
        v_new = np.asarray(v_new, dtype=np.float32)
        if k_new.shape != v_new.shape or k_new.ndim != 3:
            raise ValueError(f"KV pages need matching [heads, T, dim], got {k_new.shape} and {v_new.shape}")
        heads, n_new, dim = k_new.shape
        if self._shape is None:
            self._shape = (heads, dim)
        elif self._shape != (heads, dim):
            raise ValueError(f"KV shape changed from {self._shape} to {(heads, dim)}")
        copied = 0
        while copied < n_new:
            page_idx = self._length // self.page_size
            page_off = self._length % self.page_size
            if page_idx == len(self._k_pages):
                self._k_pages.append(np.zeros((heads, self.page_size, dim), np.float32))
                self._v_pages.append(np.zeros((heads, self.page_size, dim), np.float32))
            take = min(n_new - copied, self.page_size - page_off)
            self._k_pages[page_idx][:, page_off:page_off + take, :] = k_new[:, copied:copied + take, :]
            self._v_pages[page_idx][:, page_off:page_off + take, :] = v_new[:, copied:copied + take, :]
            self._length += take
            copied += take

    def truncate(self, length: int) -> None:
        """Drop everything at or beyond absolute position ``length``.

        Used to roll back rejected speculative tokens: the verify pass writes K+1
        positions into the cache, and whatever the rejection sampler declines has to
        disappear as if it was never computed.
        """
        if self._length == 0:
            return
        if length <= 0:
            self._length = 0
            self._k_pages.clear()
            self._v_pages.clear()
            self._shape = None
        elif length < self._length:
            self._length = int(length)

    def nbytes(self) -> int:
        return sum(int(k.nbytes + v.nbytes) for k, v in zip(self._k_pages, self._v_pages))


@dataclass
class KVCache:
    """A generation session's cache for one pipeline stage (layers ``[start, end)``).

    ``window_start`` is the absolute position of the token stored at cache index 0.
    Any change to it invalidates the whole cache (see the module docstring).
    """

    start_layer: int
    end_layer: int
    page_size: int = 16
    window_start: int = 0
    layers: dict[int, LayerCache] = field(default_factory=dict)

    def layer(self, i: int) -> LayerCache:
        lc = self.layers.get(i)
        if lc is None:
            lc = self.layers[i] = LayerCache(page_size=self.page_size)
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


@dataclass
class ActivationCache:
    """Client-side paged cache of pipeline-boundary activations.

    Boundary ``0`` is raw token input and is intentionally not stored here.
    Boundary ``j`` stores the hidden states emitted by stage ``j-1`` and therefore
    the activations needed to hydrate/replay stage ``j`` after churn.
    """

    page_size: int = 16
    _pages: dict[tuple[str, int], list[np.ndarray]] = field(default_factory=dict)
    _lengths: dict[tuple[str, int], int] = field(default_factory=dict)
    _dims: dict[tuple[str, int], int] = field(default_factory=dict)

    def record(self, session: str, boundary: int, pos_offset: int, x: np.ndarray) -> None:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"activation cache stores [T, D], got {arr.shape}")
        key = (session, int(boundary))
        if pos_offset != self._lengths.get(key, 0):
            if pos_offset == 0:
                self.drop(session, boundary)
            else:
                raise ValueError(
                    f"activation cache for {key} cannot append at {pos_offset}; "
                    f"current length is {self._lengths.get(key, 0)}")
        pages = self._pages.setdefault(key, [])
        dim = int(arr.shape[1])
        if key not in self._dims:
            self._dims[key] = dim
        elif self._dims[key] != dim:
            raise ValueError(f"activation dimension changed from {self._dims[key]} to {dim}")
        length = self._lengths.get(key, 0)
        copied = 0
        while copied < arr.shape[0]:
            page_idx = length // self.page_size
            page_off = length % self.page_size
            if page_idx == len(pages):
                pages.append(np.zeros((self.page_size, dim), np.float32))
            take = min(arr.shape[0] - copied, self.page_size - page_off)
            pages[page_idx][page_off:page_off + take, :] = arr[copied:copied + take, :]
            length += take
            copied += take
        self._lengths[key] = length

    def get(self, session: str, boundary: int, start: int = 0,
            end: Optional[int] = None) -> np.ndarray:
        key = (session, int(boundary))
        length = self._lengths.get(key, 0)
        end = length if end is None else int(end)
        if start < 0 or end < start or end > length:
            raise ValueError(f"activation span [{start}, {end}) outside cached length {length}")
        pages = self._pages.get(key, [])
        if start == end:
            dim = self._dims.get(key, 0)
            return np.zeros((0, dim), np.float32)
        chunks = []
        pos = start
        while pos < end:
            page_idx = pos // self.page_size
            page_off = pos % self.page_size
            take = min(end - pos, self.page_size - page_off)
            chunks.append(pages[page_idx][page_off:page_off + take, :])
            pos += take
        return chunks[0].copy() if len(chunks) == 1 else np.concatenate(chunks, axis=0)

    def length(self, session: str, boundary: int) -> int:
        return self._lengths.get((session, int(boundary)), 0)

    def page_count(self, session: str, boundary: int) -> int:
        return len(self._pages.get((session, int(boundary)), []))

    def drop(self, session: str, boundary: Optional[int] = None) -> None:
        keys = [k for k in self._pages if k[0] == session and (boundary is None or k[1] == boundary)]
        for key in keys:
            self._pages.pop(key, None)
            self._lengths.pop(key, None)
            self._dims.pop(key, None)

    def truncate(self, session: str, length: int) -> None:
        for key in [k for k in self._pages if k[0] == session]:
            if length <= 0:
                self._pages.pop(key, None)
                self._lengths.pop(key, None)
                self._dims.pop(key, None)
            elif length < self._lengths.get(key, 0):
                self._lengths[key] = int(length)


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
    if f"h{i}.mlp.Wgate_exps" in P:
        h = M._llama_moe(P, cfg, i, b)
    else:
        h = M._silu(b @ P[f"h{i}.mlp.Wgate"]) * (b @ P[f"h{i}.mlp.Wup"])
        h = h @ P[f"h{i}.mlp.Wdown"]
    x = x + h
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
