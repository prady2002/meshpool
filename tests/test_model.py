"""Model correctness: forward shapes, the shard==full oracle, tokenizer, sampling."""
from __future__ import annotations

import numpy as np

from meshpool.inference import model as M


def test_forward_shapes():
    cfg = M.REGISTRY["nano-gpt"]
    P = M.random_params(cfg)
    logits = M.forward_full(P, cfg, list(range(10)))
    assert logits.shape == (10, cfg.vocab_size)


def test_shard_pipeline_equals_full_model():
    """The correctness oracle: a pipeline of shards must produce logits identical
    to the single-process full model (this is what makes distributed == local)."""
    cfg = M.REGISTRY["nano-gpt"]
    P = M.random_params(cfg)
    tokens = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    full = M.forward_full(P, cfg, tokens)

    stages = [(0, 3, True, False), (3, 6, False, False), (6, cfg.n_layer, False, True)]
    x, is_tok = tokens, True
    for (s, e, first, last) in stages:
        shard = M.ShardModule(cfg, s, e, first, last, params=P)
        x = shard.forward(x, is_tok)
        is_tok = False
    assert np.allclose(full, x, atol=1e-4)
    assert float(np.max(np.abs(full - x))) < 1e-4


def test_tokenizer_roundtrip():
    tok = M.CharTokenizer()
    s = "MeshPool pools idle machines."
    assert tok.decode(tok.encode(s)) == s
    assert len(tok.itos) == 96


def test_greedy_generation_deterministic():
    cfg = M.REGISTRY["nano-gpt"]
    P = M.random_params(cfg)
    runner = lambda ctx: M.forward_full(P, cfg, ctx)  # noqa: E731
    a = M.generate_with(runner, cfg, [1, 2, 3], max_new_tokens=8, temperature=0.0)
    b = M.generate_with(runner, cfg, [1, 2, 3], max_new_tokens=8, temperature=0.0)
    assert a == b  # temperature 0 => argmax => deterministic


def test_dumps_loads_npz_roundtrip():
    """A checkpoint serialized for the pool round-trips byte-for-byte, incl. its vocab.
    This is what lets a model's weights travel through the pooled blob store."""
    cfg = M.REGISTRY["nano-gpt"]
    P = M.random_params(cfg)
    itos = M.default_charset(cfg.vocab_size)
    P2, meta = M.loads_npz(M.dumps_npz(P, itos=itos))
    assert set(P2) == set(P)
    for k in P:
        assert np.array_equal(P2[k], P[k])
    assert meta["itos"] == itos


def test_slice_params_builds_identical_shard():
    """A node handed only its slice builds the same ShardModule as one holding the
    whole checkpoint — the property that lets weights be sharded across the pool."""
    cfg = M.REGISTRY["nano-gpt"]
    full = M.random_params(cfg)
    start, end, first, last = 3, 6, False, False
    sub = M.slice_params(full, start, end, first, last)
    from_slice = M.ShardModule(cfg, start, end, first, last, params=sub)
    from_full = M.ShardModule(cfg, start, end, first, last, params=full)
    assert set(from_slice.P) == set(from_full.P)
    for k in from_full.P:
        assert np.array_equal(from_slice.P[k], from_full.P[k])


def test_pipeline_of_pooled_slices_equals_full_model():
    """Serialize -> pool-transport (bytes) -> slice per stage -> forward: the assembled
    pipeline is bit-identical to the single-process full model. Proves distributing the
    weights through the pool changes nothing about the math."""
    cfg = M.REGISTRY["nano-gpt"]
    P = M.random_params(cfg)
    tokens = [7, 0, 2, 9, 4, 4, 1, 8]
    oracle = M.forward_full(P, cfg, tokens)

    # emulate the wire: full checkpoint -> npz bytes -> reload -> per-stage slice bytes
    full, _ = M.loads_npz(M.dumps_npz(P))
    stages = [(0, 4, True, False), (4, cfg.n_layer, False, True)]
    x, is_tok = tokens, True
    for (s, e, fst, lst) in stages:
        sub, _ = M.loads_npz(M.dumps_npz(M.slice_params(full, s, e, fst, lst)))
        x = M.ShardModule(cfg, s, e, fst, lst, params=sub).forward(x, is_tok)
        is_tok = False
    assert np.allclose(oracle, x, atol=1e-4)
    assert float(np.max(np.abs(oracle - x))) < 1e-4


def test_group_specs_partition_every_parameter_once():
    """The per-layer weight groups a checkpoint is split into cover every parameter
    exactly once — so staging it group-by-group loses nothing and duplicates nothing."""
    cfg = M.REGISTRY["nano-gpt"]
    full = M.random_params(cfg)
    seen: list[str] = []
    for _name, keys in M.group_specs(cfg):
        seen += list(keys)
    assert sorted(seen) == sorted(full), "groups must cover exactly the checkpoint's keys"
    assert len(seen) == len(set(seen)), "no parameter may appear in two groups"


def test_stage_groups_reassemble_slice_params():
    """Splitting a checkpoint into per-layer groups and reassembling the groups a stage
    needs reproduces slice_params exactly — the property that lets each node pull only
    its own layers from the pool instead of the whole (larger-than-one-box) checkpoint."""
    cfg = M.REGISTRY["nano-gpt"]
    full = M.random_params(cfg)
    groups = M.split_params(full, cfg)
    cases = [(0, 3, True, False), (3, 6, False, False),
             (6, cfg.n_layer, False, True), (0, cfg.n_layer, True, True)]
    for (start, end, first, last) in cases:
        names = M.stage_groups(start, end, first, last)
        via_groups = M.merge_groups([groups[n] for n in names])
        oracle = M.slice_params(full, start, end, first, last)
        assert set(via_groups) == set(oracle), (start, end, first, last)
        for k in oracle:
            assert np.array_equal(via_groups[k], oracle[k])
        # a shard built from the reassembled slice matches one built from the full ckpt
        from_groups = M.ShardModule(cfg, start, end, first, last, params=via_groups)
        from_full = M.ShardModule(cfg, start, end, first, last, params=full)
        for k in from_full.P:
            assert np.array_equal(from_groups.P[k], from_full.P[k])
