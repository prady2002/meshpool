"""A small but *real* GPT-style transformer implemented in NumPy (spec §9.6, §4.4).

Why hand-rolled? The environment is offline (no torch / transformers / Petals),
so to demonstrate genuine **pipeline-parallel sharded inference** we implement a
compact decoder-only transformer whose contiguous blocks of layers can be hosted
on different machines. Only the hidden-state activations (a ``[T, d_model]`` tensor)
cross the network between stages — exactly the property (spec §0, §4.4) that makes
LAN/Wi-Fi fast enough where full weight/gradient sync would not be.

Design points mirrored from the references:
* **Petals**: each node holds a *contiguous block of decoder layers* and computes a
  forward pass on the incoming activations.
* **exo**: which node holds which block is decided by a *topology-aware* partitioner
  (see ``scheduler.topology_partition``).

The same layer math (`_block`) is used for both the in-process full model (for
training and for the correctness oracle) and the per-node shard, so the distributed
result is provably identical to the single-machine result (verified in tests).

Weights are loaded from a trained checkpoint (``weights/<model>.npz``) when present,
otherwise deterministically seeded — either way every node reconstructs *exactly*
the slice of parameters it needs, so no weight-shipping mechanism is required here.
"""
from __future__ import annotations

import io
import json
import math
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

WEIGHTS_DIR = Path(__file__).parent / "weights"


@dataclass
class ModelConfig:
    name: str = "nano-gpt"
    vocab_size: int = 96
    d_model: int = 128
    n_head: int = 4
    n_kv_head: Optional[int] = None
    n_layer: int = 8
    block_size: int = 64
    d_ff: int = 512
    seed: int = 1337
    architecture: str = "gpt"
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    # a representative per-layer memory footprint (MB) used only by the topology
    # partitioner so it plans placement as it would for a genuinely large model;
    # the actual NumPy weights are tiny. Documented simulation choice.
    per_layer_mb: int = 1000

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def kv_heads(self) -> int:
        return int(self.n_kv_head or self.n_head)


REGISTRY: dict[str, ModelConfig] = {
    # 8 layers @ ~1GB/layer (declared) => ~8GB: too big for any single sim node,
    # fits when pooled across the cluster.
    "nano-gpt": ModelConfig(name="nano-gpt", d_model=128, n_head=4, n_layer=8,
                            block_size=64, d_ff=512, per_layer_mb=1000),
    # Metadata matches https://huggingface.co/yujiepan/qwen2-tiny-random
    # (config/tokenizer downloads are exercised by the M1 gate). Weights are
    # intentionally NOT pretrained: this environment blocks binary weight blobs, so
    # they are deterministically initialized locally. This validates architecture,
    # tokenization, sharding and transport, not model quality.
    "qwen2-tiny-random": ModelConfig(
        name="qwen2-tiny-random", vocab_size=152064, d_model=8, n_head=4,
        n_kv_head=2, n_layer=2, block_size=32768, d_ff=16, seed=20240824,
        architecture="llama", rope_theta=1_000_000.0, rms_norm_eps=1e-6,
        per_layer_mb=2500),
}


def config_from_hf_json(name: str, raw: bytes | str) -> ModelConfig:
    """Build a local NumPy config from a HuggingFace Llama/Qwen-style config.json.

    The gate uses this for honest scope: architecture metadata and tokenizer are real
    downloaded artifacts; only weight tensors are locally initialized because this
    network cannot fetch HuggingFace LFS weight binaries.
    """
    data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    model_type = str(data.get("model_type", "")).lower()
    if model_type not in {"llama", "qwen2"}:
        raise ValueError(f"unsupported architecture in config.json: {model_type!r}")
    return ModelConfig(
        name=name,
        vocab_size=int(data["vocab_size"]),
        d_model=int(data["hidden_size"]),
        n_head=int(data["num_attention_heads"]),
        n_kv_head=int(data.get("num_key_value_heads") or data["num_attention_heads"]),
        n_layer=int(data["num_hidden_layers"]),
        block_size=int(data.get("max_position_embeddings", 32768)),
        d_ff=int(data["intermediate_size"]),
        seed=REGISTRY.get(name, ModelConfig(name=name)).seed,
        architecture="llama",
        rope_theta=float(data.get("rope_theta", 10000.0)),
        rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
        per_layer_mb=REGISTRY.get(name, ModelConfig(name=name)).per_layer_mb,
    )


# --------------------------------------------------------------------------- #
# Char-level tokenizer (vocab_size chars). Byte/char level keeps the demo
# self-contained (no external BPE vocab file needed) and is what the tiny model
# is trained on. A trained checkpoint may ship its own itos mapping in its npz.
# --------------------------------------------------------------------------- #
def default_charset(vocab_size: int = 96) -> list[str]:
    """Newline + printable ASCII (32..126) == 96 symbols for vocab_size 96."""
    chars = ["\n"] + [chr(c) for c in range(32, 127)]
    return chars[:vocab_size]


class CharTokenizer:
    def __init__(self, itos: Optional[list[str]] = None, vocab_size: int = 96) -> None:
        self.itos = list(itos) if itos else default_charset(vocab_size)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self._unk = self.stoi.get(" ", 0)

    def encode(self, s: str) -> list[int]:
        return [self.stoi.get(c, self._unk) for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if 0 <= i < len(self.itos))


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class HFTokenizer:
    """Small, dependency-free HuggingFace ``tokenizer.json`` BPE reader.

    It intentionally consumes the real downloaded BPE vocab/merges. If the optional
    ``tokenizers`` wheel is installed, callers can use that instead; this fallback is
    here so the M1 gate does not need synthetic token ids when binary deps are absent.
    """

    def __init__(self, tokenizer_json: bytes | str | dict) -> None:
        data = (json.loads(tokenizer_json.decode("utf-8"))
                if isinstance(tokenizer_json, bytes)
                else json.loads(tokenizer_json) if isinstance(tokenizer_json, str)
                else tokenizer_json)
        model = data.get("model", {})
        if model.get("type") != "BPE":
            raise ValueError(f"unsupported tokenizer model: {model.get('type')!r}")
        self.vocab: dict[str, int] = {str(k): int(v) for k, v in model["vocab"].items()}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        merges = model.get("merges", [])
        pairs = [(m.split() if isinstance(m, str) else m) for m in merges]
        self.bpe_ranks = {(str(a), str(b)): i for i, (a, b) in enumerate(pairs)}
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.unk_token = model.get("unk_token")
        self.bos_token = (data.get("added_tokens") or [{}])[0].get("content")
        self._cache: dict[str, tuple[str, ...]] = {}

    def _bpe(self, token: str) -> tuple[str, ...]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        word = tuple(token)
        if len(word) <= 1:
            return word
        while True:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, 10**12))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
        self._cache[token] = word
        return word

    def encode(self, s: str) -> list[int]:
        # Byte-level BPE over simple whitespace-preserving chunks. This uses the real
        # vocab/merges; it avoids regex-package dependency while remaining deterministic.
        pieces = re.findall(r"\s+\S*|\S+", s, flags=re.UNICODE) or [s]
        ids: list[int] = []
        unk = self.vocab.get(self.unk_token or "", 0)
        for piece in pieces:
            mapped = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            ids.extend(self.vocab.get(tok, unk) for tok in self._bpe(mapped))
        return ids

    def decode(self, ids: list[int]) -> str:
        toks = "".join(self.id_to_token.get(int(i), "") for i in ids)
        raw = bytearray(self.byte_decoder[c] for c in toks if c in self.byte_decoder)
        return raw.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def random_params(cfg: ModelConfig) -> dict:
    """Deterministically-seeded parameters (used when no trained checkpoint)."""
    g = _rng(cfg.seed)
    D, F, V, B, L = cfg.d_model, cfg.d_ff, cfg.vocab_size, cfg.block_size, cfg.n_layer
    s = 0.20 if cfg.architecture == "llama" else 0.02
    P: dict[str, np.ndarray] = {"wte": (g.standard_normal((V, D)) * s).astype(np.float32)}
    if cfg.architecture == "llama":
        kv = cfg.kv_heads
        P["lnf.g"] = np.ones(D, np.float32)
        for i in range(L):
            P[f"h{i}.rms1.g"] = np.ones(D, np.float32)
            P[f"h{i}.attn.Wq"] = (g.standard_normal((D, cfg.n_head * cfg.head_dim)) * s).astype(np.float32)
            P[f"h{i}.attn.Wk"] = (g.standard_normal((D, kv * cfg.head_dim)) * s).astype(np.float32)
            P[f"h{i}.attn.Wv"] = (g.standard_normal((D, kv * cfg.head_dim)) * s).astype(np.float32)
            P[f"h{i}.attn.Wo"] = (g.standard_normal((D, D)) * s).astype(np.float32)
            P[f"h{i}.rms2.g"] = np.ones(D, np.float32)
            P[f"h{i}.mlp.Wgate"] = (g.standard_normal((D, F)) * s).astype(np.float32)
            P[f"h{i}.mlp.Wup"] = (g.standard_normal((D, F)) * s).astype(np.float32)
            P[f"h{i}.mlp.Wdown"] = (g.standard_normal((F, D)) * s).astype(np.float32)
    else:
        P["wpe"] = (g.standard_normal((B, D)) * s).astype(np.float32)
        P["lnf.g"] = np.ones(D, np.float32); P["lnf.b"] = np.zeros(D, np.float32)
        for i in range(L):
            P[f"h{i}.ln1.g"] = np.ones(D, np.float32); P[f"h{i}.ln1.b"] = np.zeros(D, np.float32)
            P[f"h{i}.attn.Wqkv"] = (g.standard_normal((D, 3 * D)) * s).astype(np.float32)
            P[f"h{i}.attn.Wo"] = (g.standard_normal((D, D)) * s).astype(np.float32)
            P[f"h{i}.ln2.g"] = np.ones(D, np.float32); P[f"h{i}.ln2.b"] = np.zeros(D, np.float32)
            P[f"h{i}.mlp.W1"] = (g.standard_normal((D, F)) * s).astype(np.float32)
            P[f"h{i}.mlp.b1"] = np.zeros(F, np.float32)
            P[f"h{i}.mlp.W2"] = (g.standard_normal((F, D)) * s).astype(np.float32)
            P[f"h{i}.mlp.b2"] = np.zeros(D, np.float32)
    return P


def load_params(cfg: ModelConfig) -> tuple[dict, Optional[dict]]:
    """Return (params, meta). Loads a trained checkpoint if available."""
    path = WEIGHTS_DIR / f"{cfg.name}.npz"
    if path.exists():
        data = np.load(path, allow_pickle=False)
        P = {k: data[k] for k in data.files if not k.startswith("__")}
        meta = {}
        if "__itos__" in data.files:
            meta["itos"] = [str(t) for t in data["__itos__"]]
        return P, meta
    return random_params(cfg), None


def dumps_npz(params: dict, itos: Optional[list[str]] = None) -> bytes:
    """Serialize a params dict (optionally a tokenizer vocab) to ``.npz`` bytes.

    Same on-the-wire format as an on-disk checkpoint, so a checkpoint staged in the
    pool round-trips through :func:`loads_npz` exactly as ``load_params`` reads a file.
    This is what lets a model's weights travel through the pooled blob store instead of
    being shipped whole to (or pre-installed on) every node.

    The tokenizer vocabulary is stored as a **unicode** array rather than an object
    array on purpose: object arrays can only be written and read through pickle, and
    these bytes are later loaded on donor machines, where honouring a pickle would let
    a crafted upload run code on somebody else's PC. Everything written here loads
    safely with ``allow_pickle=False``."""
    arrays: dict = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
    if itos is not None:
        arrays["__itos__"] = np.asarray([str(t) for t in itos], dtype=np.str_)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def loads_npz(raw: bytes) -> tuple[dict, dict]:
    """Inverse of :func:`dumps_npz`: ``.npz`` bytes -> (params, meta) like load_params.

    ``allow_pickle=False`` is a security boundary, not a preference: ``raw`` reaches
    this function from the pooled blob store, i.e. from whoever uploaded the model, and
    it is deserialised on a **donor's** machine."""
    data = np.load(io.BytesIO(raw), allow_pickle=False)
    P = {k: data[k] for k in data.files if not k.startswith("__")}
    meta: dict = {}
    if "__itos__" in data.files:
        meta["itos"] = [str(t) for t in data["__itos__"]]
    return P, meta


def _layer_suffixes(cfg: ModelConfig) -> tuple[str, ...]:
    if cfg.architecture == "llama":
        return ("rms1.g", "attn.Wq", "attn.Wk", "attn.Wv", "attn.Wo", "rms2.g",
                "mlp.Wgate", "mlp.Wup", "mlp.Wdown")
    return _LAYER_SUFFIXES


def slice_params(full: dict, start: int, end: int,
                 is_first: bool, is_last: bool) -> dict:
    """The exact subset of parameters a pipeline stage ``[start:end]`` needs.

    Mirrors :class:`ShardModule`'s own selection, so a node handed only this slice
    serves a forward pass identical to one holding the whole checkpoint. That is the
    property that lets a model's weights be sharded across the pool: each stage node
    is shipped just its slice, and no node ever needs the full (possibly larger than
    one machine) checkpoint."""
    sub: dict = {}
    cfg = _infer_cfg_from_params(full)
    suffixes = _layer_suffixes(cfg) if cfg is not None else _LAYER_SUFFIXES
    for i in range(start, end):
        for suf in suffixes:
            sub[f"h{i}.{suf}"] = full[f"h{i}.{suf}"]
    if is_first:
        sub["wte"] = full["wte"]
        if "wpe" in full:
            sub["wpe"] = full["wpe"]
    if is_last:
        sub["wte"] = full["wte"]
        sub["lnf.g"] = full["lnf.g"]
        if "lnf.b" in full:
            sub["lnf.b"] = full["lnf.b"]
    return sub


# --------------------------------------------------------------------------- #
# Per-layer weight groups — the unit of pooled, larger-than-any-machine staging.
#
# A checkpoint splits into named groups: one per decoder layer (``h{i}``) plus the
# shared embedding (``wte``), positional (``wpe``) and final-norm (``lnf``) weights.
# Each group becomes one content-addressed blob in the pool, and a stage node pulls
# only the groups its layers need (:func:`stage_groups`) and reassembles them
# (:func:`merge_groups`) into exactly :func:`slice_params`. Because a group is the
# finest slice the partitioner ever addresses, neither the control plane nor any node
# holds more than one layer's worth of weights at a time — so a model whose full
# checkpoint dwarfs any single machine still deploys and re-partitions.
# --------------------------------------------------------------------------- #
_LAYER_SUFFIXES = ("ln1.g", "ln1.b", "attn.Wqkv", "attn.Wo", "ln2.g", "ln2.b",
                   "mlp.W1", "mlp.b1", "mlp.W2", "mlp.b2")


def _infer_cfg_from_params(P: dict) -> Optional[ModelConfig]:
    if "wte" not in P:
        return None
    D = int(P["wte"].shape[1])
    for cfg in REGISTRY.values():
        if cfg.d_model == D and cfg.vocab_size == int(P["wte"].shape[0]):
            return cfg
    return None


def _layer_keys(i: int, cfg: Optional[ModelConfig] = None) -> tuple[str, ...]:
    """The parameter keys that make up decoder layer ``i``."""
    return tuple(f"h{i}.{suf}" for suf in (_layer_suffixes(cfg) if cfg else _LAYER_SUFFIXES))


def group_specs(cfg: ModelConfig) -> list[tuple[str, tuple[str, ...]]]:
    """Ordered ``(group_name, param_keys)`` pairs a checkpoint splits into.

    Iterating these lets an uploader stage one group at a time, so the whole checkpoint
    is never held at once; together the groups partition every parameter exactly once."""
    specs: list[tuple[str, tuple[str, ...]]] = [
        ("wte", ("wte",)),
    ]
    if cfg.architecture != "llama":
        specs.append(("wpe", ("wpe",)))
        specs.append(("lnf", ("lnf.g", "lnf.b")))
    else:
        specs.append(("lnf", ("lnf.g",)))
    for i in range(cfg.n_layer):
        specs.append((f"h{i}", _layer_keys(i, cfg)))
    return specs


def split_params(full: dict, cfg: ModelConfig) -> dict[str, dict]:
    """Split a full checkpoint into named per-layer/embedding groups (see
    :func:`group_specs`). Reassembling the groups a stage needs (:func:`stage_groups`
    + :func:`merge_groups`) reproduces :func:`slice_params` exactly, so sharding the
    weights this way changes nothing about the math."""
    return {name: {k: full[k] for k in keys} for name, keys in group_specs(cfg)}


def stage_groups(start: int, end: int, is_first: bool, is_last: bool) -> list[str]:
    """The group names a pipeline stage ``[start:end]`` must assemble — its pull
    manifest. Mirrors :func:`slice_params`'s selection (``wte`` is shared by the first
    and last stages); returned de-duplicated in a deterministic order."""
    names: list[str] = [f"h{i}" for i in range(start, end)]
    if is_first:
        names += ["wte", "wpe"]
    if is_last:
        names += ["wte", "lnf"]
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def merge_groups(groups: list[dict]) -> dict:
    """Concatenate weight groups (as loaded from the pool) into one stage params dict —
    the inverse of :func:`split_params` for a single stage's slice."""
    merged: dict = {}
    for g in groups:
        merged.update(g)
    return merged


# --------------------------------------------------------------------------- #
# Forward math (shared by full model + shards + training)
# --------------------------------------------------------------------------- #
def _layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    return xhat * g + b


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _rmsnorm(x, g, eps=1e-6):
    return (x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * g).astype(np.float32)


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def embed_tokens(P: dict, cfg: ModelConfig, tokens: list[int]) -> np.ndarray:
    T = len(tokens)
    x = P["wte"][np.asarray(tokens, dtype=np.int64)]
    if cfg.architecture != "llama":
        x = x + P["wpe"][:T]
    return x.astype(np.float32)


def _rope(q: np.ndarray, k: np.ndarray, pos_offset: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    # q/k are [heads, T, head_dim]. RoPE rotates pairs in the head dimension.
    hd = q.shape[-1]
    if hd < 2:
        return q, k
    half = hd // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
    pos = (np.arange(q.shape[1], dtype=np.float32) + float(pos_offset))[:, None]
    freqs = pos * inv[None, :]
    cos, sin = np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)

    def apply(x):
        a, b = x[..., :half], x[..., half:2 * half]
        rot = np.concatenate([a * cos[None, :, :] - b * sin[None, :, :],
                              a * sin[None, :, :] + b * cos[None, :, :]], axis=-1)
        return np.concatenate([rot, x[..., 2 * half:]], axis=-1) if hd % 2 else rot

    return apply(q).astype(np.float32), apply(k).astype(np.float32)


def _block(P: dict, cfg: ModelConfig, i: int, x: np.ndarray) -> np.ndarray:
    if cfg.architecture == "llama":
        return _llama_block(P, cfg, i, x, pos_offset=0)
    T, D = x.shape
    nh, hd = cfg.n_head, cfg.head_dim
    a = _layernorm(x, P[f"h{i}.ln1.g"], P[f"h{i}.ln1.b"])
    qkv = a @ P[f"h{i}.attn.Wqkv"]                     # (T,3D)
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(T, nh, hd).transpose(1, 0, 2)         # (nh,T,hd)
    k = k.reshape(T, nh, hd).transpose(1, 0, 2)
    v = v.reshape(T, nh, hd).transpose(1, 0, 2)
    att = (q @ k.transpose(0, 2, 1)) / math.sqrt(hd)    # (nh,T,T)
    mask = np.triu(np.ones((T, T), np.float32), k=1) * -1e9
    att = _softmax(att + mask, axis=-1)
    out = att @ v                                       # (nh,T,hd)
    out = out.transpose(1, 0, 2).reshape(T, D)
    x = x + out @ P[f"h{i}.attn.Wo"]
    b = _layernorm(x, P[f"h{i}.ln2.g"], P[f"h{i}.ln2.b"])
    h = _gelu(b @ P[f"h{i}.mlp.W1"] + P[f"h{i}.mlp.b1"])
    x = x + h @ P[f"h{i}.mlp.W2"] + P[f"h{i}.mlp.b2"]
    return x.astype(np.float32)


def _llama_block(P: dict, cfg: ModelConfig, i: int, x: np.ndarray,
                 pos_offset: int = 0) -> np.ndarray:
    T, D = x.shape
    nh, nkh, hd = cfg.n_head, cfg.kv_heads, cfg.head_dim
    if nh % nkh != 0:
        raise ValueError(f"n_head ({nh}) must be divisible by n_kv_head ({nkh})")
    a = _rmsnorm(x, P[f"h{i}.rms1.g"], cfg.rms_norm_eps)
    q = (a @ P[f"h{i}.attn.Wq"]).reshape(T, nh, hd).transpose(1, 0, 2)
    k = (a @ P[f"h{i}.attn.Wk"]).reshape(T, nkh, hd).transpose(1, 0, 2)
    v = (a @ P[f"h{i}.attn.Wv"]).reshape(T, nkh, hd).transpose(1, 0, 2)
    q, k = _rope(q, k, pos_offset, cfg.rope_theta)
    repeats = nh // nkh
    k_rep = np.repeat(k, repeats, axis=0)
    v_rep = np.repeat(v, repeats, axis=0)
    att = (q @ k_rep.transpose(0, 2, 1)) / math.sqrt(hd)
    mask = np.triu(np.ones((T, T), np.float32), k=1) * -1e9
    att = _softmax(att + mask, axis=-1)
    out = (att @ v_rep).transpose(1, 0, 2).reshape(T, D)
    x = x + out @ P[f"h{i}.attn.Wo"]
    b = _rmsnorm(x, P[f"h{i}.rms2.g"], cfg.rms_norm_eps)
    h = _silu(b @ P[f"h{i}.mlp.Wgate"]) * (b @ P[f"h{i}.mlp.Wup"])
    x = x + h @ P[f"h{i}.mlp.Wdown"]
    return x.astype(np.float32)


def run_layers(P, cfg, x, start, end):
    for i in range(start, end):
        x = _block(P, cfg, i, x)
    return x


def head(P: dict, cfg: ModelConfig, x: np.ndarray) -> np.ndarray:
    if cfg.architecture == "llama":
        x = _rmsnorm(x, P["lnf.g"], cfg.rms_norm_eps)
    else:
        x = _layernorm(x, P["lnf.g"], P["lnf.b"])
    return (x @ P["wte"].T).astype(np.float32)          # tied embedding


def forward_full(P, cfg, tokens) -> np.ndarray:
    x = embed_tokens(P, cfg, tokens)
    x = run_layers(P, cfg, x, 0, cfg.n_layer)
    return head(P, cfg, x)


# --------------------------------------------------------------------------- #
# Shard (one pipeline stage = a contiguous block of layers on one machine)
# --------------------------------------------------------------------------- #
class ShardModule:
    def __init__(self, cfg: ModelConfig, start: int, end: int,
                 is_first: bool, is_last: bool, params: Optional[dict] = None) -> None:
        self.cfg = cfg
        self.start, self.end = start, end
        self.is_first, self.is_last = is_first, is_last
        full = params if params is not None else load_params(cfg)[0]
        # keep only the parameters this stage needs
        self.P: dict[str, np.ndarray] = slice_params(full, start, end, is_first, is_last)

    def forward(self, inp, is_tokens: bool) -> np.ndarray:
        x = embed_tokens(self.P, self.cfg, inp) if is_tokens else np.asarray(inp, np.float32)
        x = run_layers(self.P, self.cfg, x, self.start, self.end)
        return head(self.P, self.cfg, x) if self.is_last else x


# --------------------------------------------------------------------------- #
# Generation (works for both in-process and distributed via injected `runner`)
# --------------------------------------------------------------------------- #
def sample_next(logits_last: np.ndarray, temperature: float, rng: np.random.Generator,
                top_k: int = 40) -> int:
    if temperature <= 1e-6:
        return int(logits_last.argmax())
    logits = logits_last / temperature
    if top_k and top_k < logits.shape[-1]:
        idx = np.argpartition(logits, -top_k)[-top_k:]
        probs = _softmax(logits[idx])
        return int(rng.choice(idx, p=probs))
    probs = _softmax(logits)
    return int(rng.choice(len(probs), p=probs))


def generate_with(runner: Callable[[list[int]], np.ndarray], cfg: ModelConfig,
                  prompt: list[int], max_new_tokens: int = 32, temperature: float = 0.8,
                  seed: int = 0, stop_token: Optional[int] = None) -> list[int]:
    """Autoregressively generate, calling ``runner(context_tokens) -> logits[T,V]``.

    The same loop drives the single-process model and the distributed pipeline; only
    ``runner`` differs, which is what lets us prove they produce identical logits."""
    rng = _rng(seed)
    tokens = list(prompt)
    out: list[int] = []
    for _ in range(max_new_tokens):
        ctx = tokens[-cfg.block_size:]
        logits = runner(ctx)
        nxt = sample_next(logits[-1], temperature, rng)
        tokens.append(nxt)
        out.append(nxt)
        if stop_token is not None and nxt == stop_token:
            break
    return out
