"""Load real GGUF checkpoints into MeshPool's pipeline-shardable parameter layout.

GGUF is the on-disk format llama.cpp uses, and it is what every published
quantized open-weight model ships as. Supporting it is what turns MeshPool's
inference engine from "runs our toy transformer" into "runs the checkpoint you
downloaded", without touching the engine: the parameters produced here use
exactly the naming convention :mod:`meshpool.inference.model` already expects,
so partitioning, the pipeline, KV cache, speculative decoding, failover and the
OpenAI API all work unchanged.

Two properties matter more than anything else here.

**Load only the layers this node owns.** :func:`load_params` takes a layer range
and dequantizes only the tensors in it. The file is memory-mapped, so a node
assigned layers 40-48 of a 600 GB checkpoint never pages in layers 0-39. That is
the whole premise of the product -- a machine holds a slice, not the model -- and
it would be quietly violated by any loader that materialised the file first.

**Get the RoPE convention right.** This is the subtle one. llama.cpp rotates
adjacent pairs (x[2i], x[2i+1]); HuggingFace rotates halves (x[i], x[i+hd/2]).
The two are related by a permutation of the Q and K rows, and llama.cpp's
conversion script bakes that permutation into the file so its own kernel comes
out right. MeshPool's attention implements the HuggingFace convention, so this
module inverts the permutation at load time -- see :func:`unpermute_rope`.
Getting this wrong does not crash; it produces fluent-looking garbage, which is
far worse. :mod:`tests.test_gguf_loader` pins it with a round-trip against the
reference permutation.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

from .model import ModelConfig

# Tensor-name suffix -> MeshPool parameter suffix, and whether the matrix needs
# transposing. GGUF stores linear weights as [out_features, in_features]; the
# engine multiplies x @ W, so it wants [in_features, out_features].
_LAYER_MAP: tuple[tuple[str, str, bool], ...] = (
    ("attn_norm.weight", "rms1.g", False),
    ("attn_q.weight", "attn.Wq", True),
    ("attn_k.weight", "attn.Wk", True),
    ("attn_v.weight", "attn.Wv", True),
    ("attn_output.weight", "attn.Wo", True),
    ("ffn_norm.weight", "rms2.g", False),
    ("ffn_gate.weight", "mlp.Wgate", True),
    ("ffn_up.weight", "mlp.Wup", True),
    ("ffn_down.weight", "mlp.Wdown", True),
)

# Architectures whose GGUF files carry llama.cpp's rope permutation on Q/K.
# These are the ones whose HuggingFace implementation uses rotate_half, which is
# every model llama.cpp converts through the LlamaModel/Qwen2Model paths.
_ROPE_PERMUTED_ARCHES = frozenset({"llama", "qwen2", "qwen3", "mistral", "gemma"})


class GGUFUnsupported(RuntimeError):
    """A GGUF file MeshPool can parse but cannot faithfully execute.

    Raised loudly and specifically rather than guessed around: producing
    plausible tokens from a model we are running incorrectly is a worse failure
    than refusing to run it.
    """


def _require_gguf():
    try:
        import gguf  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise GGUFUnsupported(
            "reading GGUF checkpoints needs the 'gguf' package "
            "(pip install gguf). It is the reference implementation from "
            "llama.cpp and is pure Python + numpy.") from exc
    return __import__("gguf")


def unpermute_rope(w: np.ndarray, n_head: int) -> np.ndarray:
    """Invert llama.cpp's Q/K row permutation, restoring the HF rope layout.

    llama.cpp's converter applies, per attention head::

        w.reshape(n_head, 2, hd // 2, ...).swapaxes(1, 2).reshape(w.shape)

    which interleaves the two rotation halves so that rotating adjacent pairs
    reproduces rotating halves. This is its exact inverse: split into
    ``(hd // 2, 2)`` instead of ``(2, hd // 2)`` and swap back.

    ``n_head`` must be the number of heads *this* tensor is grouped by -- the
    query head count for Q, and the key/value head count for K under GQA, since
    K has fewer rows.
    """
    rows = w.shape[0]
    if n_head <= 0 or rows % (n_head * 2) != 0:
        raise GGUFUnsupported(
            f"cannot un-permute a [{rows}, ...] tensor across {n_head} heads: "
            f"rows must be a multiple of 2*n_head")
    return (w.reshape(n_head, rows // n_head // 2, 2, *w.shape[1:])
             .swapaxes(1, 2)
             .reshape(w.shape))


@dataclass(frozen=True)
class GGUFInfo:
    """What MeshPool needs to know about a checkpoint before placing it."""

    path: str
    architecture: str
    config: ModelConfig
    tied_embeddings: bool
    quantizations: dict[str, int]
    layer_bytes: list[int]
    total_bytes: int

    @property
    def per_layer_mb(self) -> int:
        if not self.layer_bytes:
            return 0
        return max(1, int(max(self.layer_bytes) / (1024 * 1024)))


def _field(reader, key: str, default=None):
    f = reader.fields.get(key)
    if f is None:
        return default
    try:
        val = f.contents()
    except Exception:  # pragma: no cover - malformed field
        return default
    return default if val is None else val


def _arch(reader) -> str:
    arch = _field(reader, "general.architecture")
    if not arch:
        raise GGUFUnsupported("GGUF file declares no general.architecture")
    return str(arch)


def _open_reader(path: str):
    """``GGUFReader`` with one exception type on failure.

    Truncated downloads, wrong-format files and unsupported GGUF versions
    surface from the reference reader as assorted ``ValueError`` / ``OSError`` /
    ``struct.error``. Callers -- placement, the model catalog, shard load -- need
    a single "this checkpoint is not usable" signal; an unhandled ``ValueError``
    from a half-downloaded file would otherwise take down the scheduler.
    """
    gguf = _require_gguf()
    try:
        return gguf.GGUFReader(path, "r")
    except GGUFUnsupported:
        raise
    except Exception as exc:
        raise GGUFUnsupported(f"cannot read GGUF {path!r}: {exc}") from exc


def read_info(path: str) -> GGUFInfo:
    """Parse metadata and tensor geometry without dequantizing anything.

    Cheap enough to call on the control plane for placement decisions: it reads
    the header and tensor directory, never the weights.
    """
    reader = _open_reader(path)
    arch = _arch(reader)

    def a(suffix, default=None):
        return _field(reader, f"{arch}.{suffix}", default)

    n_layer = a("block_count")
    d_model = a("embedding_length")
    n_head = a("attention.head_count")
    if n_layer is None or d_model is None or n_head is None:
        raise GGUFUnsupported(
            f"GGUF '{arch}' is missing block_count/embedding_length/head_count; "
            f"MeshPool cannot determine its geometry")

    n_kv_head = a("attention.head_count_kv", n_head)
    d_ff = a("feed_forward_length")
    tensors = {t.name: t for t in reader.tensors}
    if d_ff is None:
        gate = tensors.get("blk.0.ffn_gate.weight") or tensors.get("blk.0.ffn_up.weight")
        if gate is None:
            raise GGUFUnsupported(f"GGUF '{arch}' has no feed_forward_length and no ffn tensors")
        d_ff = int(gate.data.shape[0])

    vocab = _field(reader, "tokenizer.ggml.tokens")
    if vocab is None:
        embd = tensors.get("token_embd.weight")
        if embd is None:
            raise GGUFUnsupported("GGUF has neither a tokenizer vocab nor token_embd.weight")
        vocab_size = int(embd.data.shape[0])
    else:
        vocab_size = len(reader.fields["tokenizer.ggml.tokens"].data)

    cfg = ModelConfig(
        name=str(_field(reader, "general.name") or arch),
        vocab_size=int(vocab_size),
        d_model=int(d_model),
        n_head=int(n_head),
        n_kv_head=int(n_kv_head),
        n_layer=int(n_layer),
        block_size=int(a("context_length", 2048) or 2048),
        d_ff=int(d_ff),
        architecture="llama",
        rope_theta=float(a("rope.freq_base", 10000.0) or 10000.0),
        rms_norm_eps=float(a("attention.layer_norm_rms_epsilon", 1e-5) or 1e-5),
    )

    layer_bytes = [0] * int(n_layer)
    quantizations: dict[str, int] = {}
    total = 0
    for name, t in tensors.items():
        nbytes = int(t.n_bytes)
        total += nbytes
        qname = t.tensor_type.name
        quantizations[qname] = quantizations.get(qname, 0) + nbytes
        if name.startswith("blk."):
            try:
                idx = int(name.split(".", 2)[1])
            except (IndexError, ValueError):
                continue
            if 0 <= idx < len(layer_bytes):
                layer_bytes[idx] += nbytes

    info = GGUFInfo(
        path=path,
        architecture=arch,
        config=cfg,
        tied_embeddings="output.weight" not in tensors,
        quantizations=quantizations,
        layer_bytes=layer_bytes,
        total_bytes=total,
    )
    # per_layer_mb drives placement, so derive it from the file rather than the
    # simulated default the toy model uses.
    object.__setattr__(info, "config", _with_per_layer_mb(cfg, info.per_layer_mb))
    return info


def _with_per_layer_mb(cfg: ModelConfig, per_layer_mb: int) -> ModelConfig:
    from dataclasses import replace
    return replace(cfg, per_layer_mb=max(1, int(per_layer_mb)))


def _dequant(gguf, tensor) -> np.ndarray:
    """Materialise one tensor as float32, dequantizing if needed."""
    data = tensor.data
    qtype = tensor.tensor_type
    if qtype in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        return np.asarray(data, dtype=np.float32)
    try:
        out = gguf.quants.dequantize(data, qtype)
    except Exception as exc:
        raise GGUFUnsupported(
            f"tensor {tensor.name!r} uses quantization {qtype.name}, which this "
            f"build of the gguf package cannot dequantize: {exc}") from exc
    return np.asarray(out, dtype=np.float32)


def load_params(path: str, start: int, end: int, is_first: bool, is_last: bool,
                info: Optional[GGUFInfo] = None) -> dict[str, np.ndarray]:
    """Parameters for pipeline stage ``[start, end)``, and nothing else.

    Only the tensors this stage actually multiplies are dequantized. Everything
    else stays as untouched pages of a memory-mapped file, which is what allows
    a node to host a slice of a checkpoint far larger than its RAM.
    """
    gguf = _require_gguf()
    info = info or read_info(path)
    cfg = info.config
    if not (0 <= start <= end <= cfg.n_layer):
        raise ValueError(
            f"layer range [{start}, {end}) is outside this model's 0..{cfg.n_layer}")

    reader = _open_reader(path)
    tensors = {t.name: t for t in reader.tensors}
    permute = info.architecture in _ROPE_PERMUTED_ARCHES

    def take(name: str) -> np.ndarray:
        t = tensors.get(name)
        if t is None:
            raise GGUFUnsupported(
                f"checkpoint is missing tensor {name!r}; MeshPool cannot run a "
                f"partial '{info.architecture}' model")
        return _dequant(gguf, t)

    P: dict[str, np.ndarray] = {}
    for i in range(start, end):
        for suffix, param, transpose in _LAYER_MAP:
            w = take(f"blk.{i}.{suffix}")
            if permute and suffix == "attn_q.weight":
                w = unpermute_rope(w, cfg.n_head)
            elif permute and suffix == "attn_k.weight":
                w = unpermute_rope(w, cfg.kv_heads)
            P[f"h{i}.{param}"] = np.ascontiguousarray(w.T if transpose else w,
                                                      dtype=np.float32)

    if is_first or is_last:
        P["wte"] = take("token_embd.weight")
    if is_last:
        P["lnf.g"] = take("output_norm.weight")
        if not info.tied_embeddings:
            P["out"] = take("output.weight")
    return P


def load_full_params(path: str, info: Optional[GGUFInfo] = None) -> dict[str, np.ndarray]:
    """Every parameter, for the single-process correctness oracle."""
    info = info or read_info(path)
    return load_params(path, 0, info.config.n_layer, True, True, info=info)


def read_vocab(path: str) -> dict[str, Any]:
    """Tokenizer vocabulary and merge rules straight out of the checkpoint.

    Returned raw rather than as a tokenizer object so the caller can decide what
    to do with an unsupported tokenizer model instead of failing at import time.
    """
    gguf = _require_gguf()
    reader = _open_reader(path)

    def _list(key: str) -> list:
        f = reader.fields.get(key)
        if f is None:
            return []
        return [f.contents(i) for i in range(len(f.data))]

    tokens = _list("tokenizer.ggml.tokens")
    return {
        "model": str(_field(reader, "tokenizer.ggml.model") or ""),
        "pre": str(_field(reader, "tokenizer.ggml.pre") or ""),
        "tokens": [t if isinstance(t, str) else str(t) for t in tokens],
        "scores": _list("tokenizer.ggml.scores"),
        "token_type": _list("tokenizer.ggml.token_type"),
        "merges": [m if isinstance(m, str) else str(m)
                   for m in _list("tokenizer.ggml.merges")],
        "bos_id": _field(reader, "tokenizer.ggml.bos_token_id"),
        "eos_id": _field(reader, "tokenizer.ggml.eos_token_id"),
        "add_bos": bool(_field(reader, "tokenizer.ggml.add_bos_token", True)),
        "remove_extra_whitespaces": bool(
            _field(reader, "tokenizer.ggml.remove_extra_whitespaces", False)),
        "add_space_prefix": bool(
            _field(reader, "tokenizer.ggml.add_space_prefix", True)),
    }


@functools.lru_cache(maxsize=8)
def cached_info(path: str) -> GGUFInfo:
    """``read_info`` memoized: placement asks for the same file repeatedly."""
    return read_info(path)


def layer_span_bytes(info: GGUFInfo, start: int, end: int) -> int:
    """Bytes a node must hold to serve ``[start, end)`` -- what placement sizes."""
    return int(sum(info.layer_bytes[start:end]))


def describe(info: GGUFInfo) -> str:
    """One-line human summary for logs and the console."""
    cfg = info.config
    quant = max(info.quantizations.items(), key=lambda kv: kv[1])[0] if info.quantizations else "?"
    return (f"{cfg.name} [{info.architecture}] {cfg.n_layer}L d={cfg.d_model} "
            f"heads={cfg.n_head}/{cfg.kv_heads} ff={cfg.d_ff} "
            f"vocab={cfg.vocab_size} {quant} "
            f"{info.total_bytes / 1e9:.2f}GB ~{info.per_layer_mb}MB/layer")


def iter_layer_tensor_names(cfg: ModelConfig, layers: Iterable[int]) -> Iterable[str]:
    """GGUF tensor names a given set of layers depends on (for prefetching)."""
    for i in layers:
        for suffix, _param, _t in _LAYER_MAP:
            yield f"blk.{i}.{suffix}"
