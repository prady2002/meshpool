"""Build real GGUF checkpoints for tests.

These files are written with llama.cpp's own reference writer (the ``gguf``
package), so they are byte-exact GGUF -- same magic, same alignment, same tensor
directory, same metadata keys, same quantized block layouts as a checkpoint
downloaded from HuggingFace. What is synthetic is the *weights*, not the format.

That distinction is the whole point. This environment's TLS proxy blocks model
downloads, so a published checkpoint cannot be fetched here. Rather than pretend
otherwise, the fixtures exercise every part of the path that a real file would:
metadata parsing, tensor geometry, quantized dequantization, the rope
permutation, layer-range loading, and the tokenizer vocabulary. Point the same
loader at a downloaded Kimi-K2 GGUF and the only thing that changes is the size.

The weights are generated in **HuggingFace layout** and then pushed through
:func:`permute_rope_reference` -- a transcription of llama.cpp's
``convert_hf_to_gguf.py`` permutation -- on the way into the file. The loader
inverts that independently. Neither function is derived from the other, so the
round-trip is a genuine check rather than a tautology.
"""
from __future__ import annotations

import numpy as np


def permute_rope_reference(w: np.ndarray, n_head: int) -> np.ndarray:
    """llama.cpp's Q/K permutation, transcribed from ``convert_hf_to_gguf.py``.

    The reference reads::

        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                       .swapaxes(1, 2)
                       .reshape(weights.shape))

    Kept deliberately verbatim: it is the specification that
    ``gguf_loader.unpermute_rope`` has to invert, and paraphrasing it would
    weaken the test into a restatement of the implementation.
    """
    return (w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:])
             .swapaxes(1, 2)
             .reshape(w.shape))


def build_llama_gguf(path: str, *, n_layer: int = 4, d_model: int = 32,
                     n_head: int = 4, n_kv_head: int = 2, d_ff: int = 64,
                     vocab_size: int = 64, seed: int = 20240826,
                     tied: bool = False, quantize: str | None = None,
                     rope_theta: float = 10000.0,
                     rms_eps: float = 1e-5) -> dict[str, np.ndarray]:
    """Write a real GGUF file; return the HF-layout weights that went into it.

    The returned dict is the ground truth the loader must recover: it is in
    MeshPool/HuggingFace orientation, *before* the rope permutation and *before*
    the transpose into the file's [out, in] convention.
    """
    import gguf

    hd = d_model // n_head
    rng = np.random.default_rng(seed)

    def rand(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    truth: dict[str, np.ndarray] = {
        "token_embd.weight": rand(vocab_size, d_model),
        "output_norm.weight": (rng.random(d_model).astype(np.float32) + 0.5),
    }
    if not tied:
        truth["output.weight"] = rand(vocab_size, d_model)
    for i in range(n_layer):
        truth[f"blk.{i}.attn_norm.weight"] = rng.random(d_model).astype(np.float32) + 0.5
        truth[f"blk.{i}.attn_q.weight"] = rand(n_head * hd, d_model)
        truth[f"blk.{i}.attn_k.weight"] = rand(n_kv_head * hd, d_model)
        truth[f"blk.{i}.attn_v.weight"] = rand(n_kv_head * hd, d_model)
        truth[f"blk.{i}.attn_output.weight"] = rand(d_model, n_head * hd)
        truth[f"blk.{i}.ffn_norm.weight"] = rng.random(d_model).astype(np.float32) + 0.5
        truth[f"blk.{i}.ffn_gate.weight"] = rand(d_ff, d_model)
        truth[f"blk.{i}.ffn_up.weight"] = rand(d_ff, d_model)
        truth[f"blk.{i}.ffn_down.weight"] = rand(d_model, d_ff)

    w = gguf.GGUFWriter(path, "llama")
    w.add_block_count(n_layer)
    w.add_context_length(256)
    w.add_embedding_length(d_model)
    w.add_feed_forward_length(d_ff)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_kv_head)
    w.add_layer_norm_rms_eps(rms_eps)
    w.add_rope_freq_base(rope_theta)
    w.add_rope_dimension_count(hd)
    w.add_tokenizer_model("llama")
    # A vocabulary shaped like a real SentencePiece one: byte fallback tokens,
    # specials, then word pieces with the U+2581 space marker.
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)][:vocab_size - 3]
    tokens = tokens[:vocab_size]
    while len(tokens) < vocab_size:
        tokens.append(f"\u2581w{len(tokens)}")
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * vocab_size)
    w.add_token_types([3, 3, 3] + [6] * (vocab_size - 3))
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)

    qtype = None
    if quantize:
        qtype = getattr(gguf.GGMLQuantizationType, quantize.upper())

    for name, arr in truth.items():
        out = arr
        # Q/K go into the file permuted, exactly as llama.cpp's converter writes
        # them. The loader has to undo this.
        if name.endswith("attn_q.weight"):
            out = permute_rope_reference(out, n_head)
        elif name.endswith("attn_k.weight"):
            out = permute_rope_reference(out, n_kv_head)
        # GGUF stores linear weights [out, in]; that is already this layout.
        # Norm vectors stay float32 in real checkpoints too, and block quants
        # need whole blocks of 32 columns.
        if qtype is not None and out.ndim == 2 and out.shape[-1] % 32 == 0:
            out = np.ascontiguousarray(out)
            # raw_shape must be left to the writer: it derives the logical shape
            # from the packed byte array and the quant type. Passing the logical
            # shape makes it read the row length as a byte count and reject it.
            w.add_tensor(name, gguf.quants.quantize(out, qtype), raw_dtype=qtype)
        else:
            w.add_tensor(name, np.ascontiguousarray(out))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return truth


def build_moe_llama_gguf(path: str, *, n_layer: int = 4, d_model: int = 32,
                         n_head: int = 4, n_kv_head: int = 2, d_ff: int = 32,
                         expert_count: int = 4, expert_used_count: int = 2,
                         expert_ff: int = 128, shared_ff: int = 16,
                         vocab_size: int = 64, dense_layers: tuple[int, ...] = (0,),
                         seed: int = 20240827) -> dict[str, np.ndarray]:
    """Write a mixed dense/MoE GGUF using llama.cpp's reference tensor names."""
    import gguf

    hd = d_model // n_head
    rng = np.random.default_rng(seed)

    def rand(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    truth: dict[str, np.ndarray] = {
        "token_embd.weight": rand(vocab_size, d_model),
        "output_norm.weight": (rng.random(d_model).astype(np.float32) + 0.5),
        "output.weight": rand(vocab_size, d_model),
    }
    dense = set(dense_layers)
    for i in range(n_layer):
        truth[f"blk.{i}.attn_norm.weight"] = rng.random(d_model).astype(np.float32) + 0.5
        truth[f"blk.{i}.attn_q.weight"] = rand(n_head * hd, d_model)
        truth[f"blk.{i}.attn_k.weight"] = rand(n_kv_head * hd, d_model)
        truth[f"blk.{i}.attn_v.weight"] = rand(n_kv_head * hd, d_model)
        truth[f"blk.{i}.attn_output.weight"] = rand(d_model, n_head * hd)
        truth[f"blk.{i}.ffn_norm.weight"] = rng.random(d_model).astype(np.float32) + 0.5
        if i in dense:
            truth[f"blk.{i}.ffn_gate.weight"] = rand(d_ff, d_model)
            truth[f"blk.{i}.ffn_up.weight"] = rand(d_ff, d_model)
            truth[f"blk.{i}.ffn_down.weight"] = rand(d_model, d_ff)
        else:
            truth[f"blk.{i}.ffn_gate_inp.weight"] = rand(expert_count, d_model)
            truth[f"blk.{i}.ffn_gate_exps.weight"] = rand(expert_count, expert_ff, d_model)
            truth[f"blk.{i}.ffn_up_exps.weight"] = rand(expert_count, expert_ff, d_model)
            truth[f"blk.{i}.ffn_down_exps.weight"] = rand(expert_count, d_model, expert_ff)
            if shared_ff:
                truth[f"blk.{i}.ffn_gate_shexp.weight"] = rand(shared_ff, d_model)
                truth[f"blk.{i}.ffn_up_shexp.weight"] = rand(shared_ff, d_model)
                truth[f"blk.{i}.ffn_down_shexp.weight"] = rand(d_model, shared_ff)

    w = gguf.GGUFWriter(path, "llama")
    w.add_block_count(n_layer)
    w.add_context_length(256)
    w.add_embedding_length(d_model)
    w.add_feed_forward_length(d_ff)
    w.add_expert_count(expert_count)
    w.add_expert_used_count(expert_used_count)
    w.add_expert_feed_forward_length(expert_ff)
    if shared_ff:
        w.add_expert_shared_feed_forward_length(shared_ff)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_kv_head)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_freq_base(10000.0)
    w.add_rope_dimension_count(hd)
    w.add_tokenizer_model("llama")
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)][:vocab_size - 3]
    tokens = tokens[:vocab_size]
    while len(tokens) < vocab_size:
        tokens.append(f"\u2581w{len(tokens)}")
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * vocab_size)
    w.add_token_types([3, 3, 3] + [6] * (vocab_size - 3))
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)

    for name, arr in truth.items():
        out = arr
        if name.endswith("attn_q.weight"):
            out = permute_rope_reference(out, n_head)
        elif name.endswith("attn_k.weight"):
            out = permute_rope_reference(out, n_kv_head)
        w.add_tensor(name, np.ascontiguousarray(out))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return truth
