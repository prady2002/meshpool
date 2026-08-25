import asyncio
import json

import numpy as np

from meshpool.inference import model as M
from meshpool.inference.pipeline import CachedShard, Pipeline, generate_baseline


def test_qwen2_config_is_derived_from_hf_json():
    raw = json.dumps({
        "model_type": "qwen2",
        "vocab_size": 152064,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
    })
    cfg = M.config_from_hf_json("qwen2-tiny-random", raw)
    assert cfg.architecture == "llama"
    assert cfg.n_head == 4
    assert cfg.kv_heads == 2
    assert cfg.n_head // cfg.kv_heads == 2


def test_llama_qwen_gqa_full_equals_three_stage_pipeline_with_head_stage():
    cfg = M.REGISTRY["qwen2-tiny-random"]
    params = M.random_params(cfg)
    prompt = [10, 20, 30, 40]
    single = M.generate_with(lambda ctx: M.forward_full(params, cfg, ctx), cfg, prompt,
                             max_new_tokens=6, temperature=0.0)
    stages = [
        CachedShard(cfg, 0, 1, True, False, params),
        CachedShard(cfg, 1, 2, False, False, params),
        CachedShard(cfg, 2, 2, False, True, params),
    ]
    dist, stats = asyncio.run(generate_baseline(
        Pipeline(cfg, stages), prompt, 6, temperature=0.0, session="test"))
    assert dist == single
    assert stats.hops == 14
    assert stats.wire_bytes > 0
    assert "h1.attn.Wk" not in stages[0].P
    assert "h0.attn.Wv" not in stages[1].P
    assert all(not k.startswith("h") for k in stages[2].P)


def test_hf_bpe_tokenizer_uses_real_vocab_and_merges():
    tok_json = {
        "model": {
            "type": "BPE",
            "unk_token": "<unk>",
            "vocab": {"<unk>": 0, "M": 1, "e": 2, "s": 3, "h": 4, "Me": 5, "sh": 6, "Mesh": 7},
            "merges": ["M e", "s h", "Me sh"],
        }
    }
    tok = M.HFTokenizer(tok_json)
    ids = tok.encode("Mesh")
    assert ids == [7]
    assert tok.decode(ids) == "Mesh"


def test_llama_cached_decode_matches_full_recompute():
    cfg = M.REGISTRY["qwen2-tiny-random"]
    params = M.random_params(cfg)
    prompt = [1, 2, 3]
    pipe = Pipeline(cfg, [
        CachedShard(cfg, 0, 1, True, False, params),
        CachedShard(cfg, 1, 2, False, True, params),
    ])
    cached, _ = asyncio.run(generate_baseline(pipe, prompt, 5, temperature=0.0,
                                              session="cache"))
    full = M.generate_with(lambda ctx: M.forward_full(params, cfg, ctx), cfg, prompt,
                           max_new_tokens=5, temperature=0.0)
    assert cached == full
    assert np.isfinite(M.forward_full(params, cfg, prompt)).all()
