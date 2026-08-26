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

import contextlib
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

_ATTN_LAYER_MAP: tuple[tuple[str, str, bool], ...] = _LAYER_MAP[:6]
_DENSE_FFN_MAP: tuple[tuple[str, str, bool], ...] = _LAYER_MAP[6:]
_MOE_TENSOR_SUFFIXES: tuple[str, ...] = (
    "ffn_gate_exps.weight",
    "ffn_down_exps.weight",
    "ffn_up_exps.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_shexp.weight",
    "ffn_down_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_gate_inp_shexp.weight",
)
_MOE_EXPERT_SUFFIXES = {"ffn_gate_exps.weight", "ffn_down_exps.weight", "ffn_up_exps.weight"}
_MOE_LOAD_MAP: tuple[tuple[str, str, bool], ...] = (
    ("ffn_gate_inp.weight", "mlp.Wrouter", True),
    ("ffn_gate_exps.weight", "mlp.Wgate_exps", True),
    ("ffn_up_exps.weight", "mlp.Wup_exps", True),
    ("ffn_down_exps.weight", "mlp.Wdown_exps", True),
    ("ffn_gate_shexp.weight", "mlp.Wgate_shexp", True),
    ("ffn_up_shexp.weight", "mlp.Wup_shexp", True),
    ("ffn_down_shexp.weight", "mlp.Wdown_shexp", True),
)

# Which architectures store Q/K permuted is not a matter of taste: llama.cpp
# picks one of two rope conventions per architecture, and the converter permutes
# Q/K only for the "NORM" ones so that rotating adjacent pairs reproduces
# rotating halves. Our attention implements rotate_half, so we must undo the
# permutation for exactly the NORM set and leave the NEOX set alone.
#
# There is no metadata field to read: llama.cpp's llama_model_rope_type()
# switches on model->arch and nothing else, so mirroring that switch is the only
# correct approach. Both lists below are transcribed programmatically from it
# (src/llama-model.cpp) rather than by hand. [tier: cited]
#
# Getting an entry wrong is the worst failure this loader has, because it does
# not crash: the model loads, generates fluently, and is quietly wrong.
_ROPE_PERMUTED_ARCHES = frozenset({
    "arcee", "arctic", "baichuan", "bailingmoe", "bailingmoe3", "chameleon",
    "chatglm", "cohere2", "cohere2moe", "command_r", "deci", "deepseek",
    "deepseek2", "deepseek2ocr", "deepseek32", "deepseek4", "dots3note",
    "eagle3", "ernie4_5", "ernie4_5_moe", "glm_dsa", "granite",
    "granite_hybrid", "granite_moe", "granite_swa", "granite_switch",
    "internlm2", "llada", "llama", "llama4", "llama_embed", "maincoder",
    "minicpm", "mistral3", "mistral4", "muse_glimmer", "nanbeige", "neo_bert",
    "olmo", "plm", "pockettts", "smollm3", "starcoder", "xverse"
})

# Architectures that keep the HuggingFace rotate_half layout, needing no fixup.
_ROPE_HALF_ARCHES = frozenset({
    "afmoe", "apertus", "bailingmoe2", "bert", "bitnet", "codeshell",
    "cogvlm", "dbrx", "dots1", "dream", "eurobert", "exaone", "exaone4",
    "exaone_moe", "falcon", "falcon_h1", "gemma", "gemma2", "gemma3",
    "gemma3n", "gemma4", "gemma4_assistant", "gemma_embedding", "gptneox",
    "grok", "grovemoe", "hunyuan_dense", "hunyuan_moe", "hy_v3", "jais2",
    "jina_bert_v3", "laguna", "lfm2", "lfm2moe", "llada_moe", "mellum",
    "mimo2", "minicpm3", "minimax_01", "minimax_m2", "minimax_m3",
    "modern_bert", "nemotron", "nomic_bert", "nomic_bert_moe", "olmo2",
    "olmoe", "openai_moe", "openelm", "orion", "pangu_embed", "phi2", "phi3",
    "phimoe", "plamo", "plamo2", "plamo3", "qwen", "qwen2", "qwen2moe",
    "qwen3", "qwen3moe", "qwen3next", "rnd1", "seed_oss", "smallthinker",
    "stablelm", "starcoder2", "step35", "talkie"
})


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


def _rope_is_permuted(arch: str) -> bool:
    """Whether this architecture's GGUF stores Q/K in llama.cpp's paired layout.

    Deliberately refuses to guess. An architecture in neither list is one whose
    convention nobody has checked, and defaulting either way would load the
    model, generate fluent text, and be silently wrong — exactly the failure this
    module exists to prevent. A loud refusal costs someone five minutes; a wrong
    guess costs them their trust in every answer the model gives.
    """
    if arch in _ROPE_PERMUTED_ARCHES:
        return True
    if arch in _ROPE_HALF_ARCHES:
        return False
    raise GGUFUnsupported(
        f"unknown rope convention for architecture {arch!r}. MeshPool has not "
        f"verified whether llama.cpp stores this architecture's Q/K weights "
        f"permuted, and guessing would produce fluent but incorrect output. "
        f"Look up {arch!r} in llama_model_rope_type() in llama.cpp's "
        f"src/llama-model.cpp and add it to _ROPE_PERMUTED_ARCHES (for "
        f"LLAMA_ROPE_TYPE_NORM) or _ROPE_HALF_ARCHES (for LLAMA_ROPE_TYPE_NEOX).")


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
class GGUFLayerMoEInfo:
    layer: int
    is_moe: bool
    expert_count: int
    expert_used_count: int
    bytes: int
    tensor_names: tuple[str, ...] = ()


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
    moe_layers: tuple[GGUFLayerMoEInfo, ...] = ()

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


def _moe_metadata(reader, arch: str) -> dict[str, Optional[int]]:
    return {
        "expert_count": _field(reader, f"{arch}.expert_count"),
        "expert_used_count": _field(reader, f"{arch}.expert_used_count"),
        "expert_feed_forward_length": _field(reader, f"{arch}.expert_feed_forward_length"),
        "expert_shared_feed_forward_length": _field(reader, f"{arch}.expert_shared_feed_forward_length"),
    }


def _is_moe_tensor(name: str) -> bool:
    if not name.startswith("blk."):
        return False
    try:
        suffix = name.split(".", 2)[2]
    except IndexError:
        return False
    return suffix in _MOE_TENSOR_SUFFIXES


def _layer_index(name: str) -> Optional[int]:
    if not name.startswith("blk."):
        return None
    try:
        return int(name.split(".", 2)[1])
    except (IndexError, ValueError):
        return None


@contextlib.contextmanager
def _open_reader(path: str):
    """``GGUFReader`` with one exception type on failure.

    Truncated downloads, wrong-format files and unsupported GGUF versions
    surface from the reference reader as assorted ``ValueError`` / ``OSError`` /
    ``struct.error``. Callers -- placement, the model catalog, shard load -- need
    a single "this checkpoint is not usable" signal; an unhandled ``ValueError``
    from a half-downloaded file would otherwise take down the scheduler.
    """
    gguf = _require_gguf()
    reader = None
    try:
        reader = gguf.GGUFReader(path, "r")
    except GGUFUnsupported:
        raise
    except Exception as exc:
        raise GGUFUnsupported(f"cannot read GGUF {path!r}: {exc}") from exc
    try:
        yield reader
    finally:
        if reader is not None:
            mm = getattr(getattr(reader, "data", None), "_mmap", None)
            if mm is not None:
                mm.close()


def read_info(path: str) -> GGUFInfo:
    """Parse metadata and tensor geometry without dequantizing anything.

    Cheap enough to call on the control plane for placement decisions: it reads
    the header and tensor directory, never the weights.
    """
    with _open_reader(path) as reader:
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
            if gate is None and a("expert_feed_forward_length") is not None:
                d_ff = int(a("expert_feed_forward_length"))
            elif gate is None:
                raise GGUFUnsupported(f"GGUF '{arch}' has no feed_forward_length and no ffn tensors")
            else:
                d_ff = int(gate.data.shape[0])

        vocab = _field(reader, "tokenizer.ggml.tokens")
        if vocab is None:
            embd = tensors.get("token_embd.weight")
            if embd is None:
                raise GGUFUnsupported("GGUF has neither a tokenizer vocab nor token_embd.weight")
            vocab_size = int(embd.data.shape[0])
        else:
            vocab_size = len(reader.fields["tokenizer.ggml.tokens"].data)

        moe_meta = _moe_metadata(reader, arch)
        expert_count = int(moe_meta["expert_count"] or 0)
        expert_used_count = int(moe_meta["expert_used_count"] or 0)
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
            expert_count=expert_count,
            expert_used_count=expert_used_count,
            expert_feed_forward_length=(
                int(moe_meta["expert_feed_forward_length"])
                if moe_meta["expert_feed_forward_length"] is not None else None),
            expert_shared_feed_forward_length=(
                int(moe_meta["expert_shared_feed_forward_length"])
                if moe_meta["expert_shared_feed_forward_length"] is not None else None),
        )

        layer_bytes = [0] * int(n_layer)
        moe_tensor_names: list[list[str]] = [[] for _ in range(int(n_layer))]
        quantizations: dict[str, int] = {}
        total = 0
        for name, t in tensors.items():
            nbytes = int(t.n_bytes)
            total += nbytes
            qname = t.tensor_type.name
            quantizations[qname] = quantizations.get(qname, 0) + nbytes
            idx = _layer_index(name)
            if idx is not None and 0 <= idx < len(layer_bytes):
                layer_bytes[idx] += nbytes
                if _is_moe_tensor(name):
                    moe_tensor_names[idx].append(name)

        moe_layers = []
        for idx, names in enumerate(moe_tensor_names):
            actual_experts = expert_count
            for name in names:
                if any(name.endswith(s) for s in _MOE_EXPERT_SUFFIXES):
                    actual_experts = int(tensors[name].data.shape[0])
                    break
            moe_layers.append(GGUFLayerMoEInfo(
                layer=idx,
                is_moe=bool(names),
                expert_count=int(actual_experts or 0),
                expert_used_count=int(expert_used_count or (1 if names else 0)),
                bytes=int(layer_bytes[idx]),
                tensor_names=tuple(sorted(names))))
        if expert_count > 0 and not any(m.is_moe for m in moe_layers):
            raise GGUFUnsupported(
                f"GGUF '{arch}' declares expert_count={expert_count} but has no MoE expert tensors")

        tied_embeddings = "output.weight" not in tensors
        info = GGUFInfo(
            path=path,
            architecture=arch,
            config=cfg,
            tied_embeddings=tied_embeddings,
            quantizations=quantizations,
            layer_bytes=layer_bytes,
            total_bytes=total,
            moe_layers=tuple(moe_layers),
        )
        del tensors
    # per_layer_mb drives placement, so derive it from the file rather than the
    # simulated default the toy model uses.
    object.__setattr__(info, "config", _with_per_layer_mb(
        cfg, info.per_layer_mb, tuple(layer_bytes), tuple(m.is_moe for m in moe_layers)))
    return info


def _with_per_layer_mb(
    cfg: ModelConfig,
    per_layer_mb: int,
    layer_bytes: Optional[tuple[int, ...]] = None,
    moe_layers: Optional[tuple[bool, ...]] = None,
) -> ModelConfig:
    from dataclasses import replace
    return replace(cfg, per_layer_mb=max(1, int(per_layer_mb)),
                   layer_bytes=layer_bytes, moe_layers=moe_layers)


def _dequant(gguf, tensor) -> np.ndarray:
    """Materialise one tensor as float32, dequantizing if needed."""
    data = tensor.data
    qtype = tensor.tensor_type
    if qtype in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        return np.array(data, dtype=np.float32, copy=True)
    try:
        out = gguf.quants.dequantize(data, qtype)
    except Exception as exc:
        raise GGUFUnsupported(
            f"tensor {tensor.name!r} uses quantization {qtype.name}, which this "
            f"build of the gguf package cannot dequantize: {exc}") from exc
    return np.array(out, dtype=np.float32, copy=True)


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

    with _open_reader(path) as reader:
        tensors = {t.name: t for t in reader.tensors}
        permute = _rope_is_permuted(info.architecture)

        def take(name: str) -> np.ndarray:
            t = tensors.get(name)
            if t is None:
                raise GGUFUnsupported(
                    f"checkpoint is missing tensor {name!r}; MeshPool cannot run a "
                    f"partial '{info.architecture}' model")
            return _dequant(gguf, t)

        P: dict[str, np.ndarray] = {}
        for i in range(start, end):
            is_moe = bool(info.moe_layers and info.moe_layers[i].is_moe)
            for suffix, param, transpose in _ATTN_LAYER_MAP:
                w = take(f"blk.{i}.{suffix}")
                if permute and suffix == "attn_q.weight":
                    w = unpermute_rope(w, cfg.n_head)
                elif permute and suffix == "attn_k.weight":
                    w = unpermute_rope(w, cfg.kv_heads)
                P[f"h{i}.{param}"] = np.ascontiguousarray(w.T if transpose else w,
                                                          dtype=np.float32)
            if is_moe:
                router = take(f"blk.{i}.ffn_gate_inp.weight")
                P[f"h{i}.mlp.Wrouter"] = np.ascontiguousarray(router.T, dtype=np.float32)
                for suffix, param in (
                    ("ffn_gate_exps.weight", "Wgate_exps"),
                    ("ffn_up_exps.weight", "Wup_exps"),
                    ("ffn_down_exps.weight", "Wdown_exps"),
                ):
                    w = take(f"blk.{i}.{suffix}")
                    P[f"h{i}.mlp.{param}"] = np.ascontiguousarray(
                        np.swapaxes(w, -1, -2), dtype=np.float32)
                for suffix, param in (
                    ("ffn_gate_shexp.weight", "Wgate_shexp"),
                    ("ffn_up_shexp.weight", "Wup_shexp"),
                    ("ffn_down_shexp.weight", "Wdown_shexp"),
                ):
                    name = f"blk.{i}.{suffix}"
                    if name in tensors:
                        w = take(name)
                        P[f"h{i}.mlp.{param}"] = np.ascontiguousarray(w.T, dtype=np.float32)
            else:
                for suffix, param, transpose in _DENSE_FFN_MAP:
                    w = take(f"blk.{i}.{suffix}")
                    P[f"h{i}.{param}"] = np.ascontiguousarray(w.T if transpose else w,
                                                              dtype=np.float32)

        if is_first or is_last:
            P["wte"] = take("token_embd.weight")
        if is_last:
            P["lnf.g"] = take("output_norm.weight")
            if not info.tied_embeddings:
                P["out"] = take("output.weight")
        del tensors
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
    with _open_reader(path) as reader:
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
    moe = sum(1 for m in info.moe_layers if m.is_moe)
    moe_text = f" moe={moe}L/{cfg.expert_count}e" if moe else ""
    return (f"{cfg.name} [{info.architecture}] {cfg.n_layer}L d={cfg.d_model} "
            f"heads={cfg.n_head}/{cfg.kv_heads} ff={cfg.d_ff} "
            f"vocab={cfg.vocab_size}{moe_text} {quant} "
            f"{info.total_bytes / 1e9:.2f}GB ~{info.per_layer_mb}MB/layer")


def iter_layer_tensor_names(cfg: ModelConfig, layers: Iterable[int]) -> Iterable[str]:
    """GGUF tensor names a given set of layers depends on (for prefetching)."""
    for i in layers:
        is_moe = bool(cfg.moe_layers and cfg.moe_layers[i])
        maps = _ATTN_LAYER_MAP + (_MOE_LOAD_MAP if is_moe else _DENSE_FFN_MAP)
        for suffix, _param, _t in maps:
            yield f"blk.{i}.{suffix}"


def validate_moe_expert_colocation(
    info: GGUFInfo,
    assignments,
    tensor_node_ids: Optional[dict[str, str]] = None,
) -> None:
    """Raise if any MoE layer's expert tensors are split across nodes."""
    owners: dict[int, str] = {}
    for a in assignments:
        for layer in range(int(a.start_layer), int(a.end_layer)):
            if layer in owners:
                raise ValueError(f"layer {layer} is assigned to both {owners[layer]} and {a.node_id}")
            owners[layer] = a.node_id
    for m in info.moe_layers:
        if not m.is_moe:
            continue
        owner = owners.get(m.layer)
        if owner is None:
            raise ValueError(f"MoE layer {m.layer} is not assigned to any node")
        if tensor_node_ids:
            seen = {tensor_node_ids.get(t, owner) for t in m.tensor_names}
            if len(seen) != 1:
                raise ValueError(
                    f"MoE layer {m.layer} experts are split across nodes: {sorted(seen)}")
