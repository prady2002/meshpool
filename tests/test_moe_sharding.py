from __future__ import annotations

import numpy as np
import pytest

from gguf_fixtures import build_moe_llama_gguf
from meshpool.common.models import Capability, DonationPolicy, LayerAssignment, Node
from meshpool.controlplane.scheduler import (
    explain_layer_bytes_fit,
    topology_partition_by_bytes,
)
from meshpool.inference import adapters, gguf_loader as G, model as M

pytest.importorskip("gguf", reason="GGUF support needs the reference gguf package")


def _node(nid: str, ram_mb: int = 2, flops: float = 1.0) -> Node:
    return Node(
        id=nid, name=nid, url=f"http://127.0.0.1:90{nid}",
        capability=Capability(cpu_cores=4, ram_mb=ram_mb, flops_estimate=flops),
        policy=DonationPolicy(ram_mb=ram_mb),
    )


def _silu_ref(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _softmax_ref(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def _expert_ffn_ref(b: np.ndarray, gate: np.ndarray, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    return (_silu_ref(b @ gate) * (b @ up)) @ down


def _naive_moe_ref(P: dict, cfg: M.ModelConfig, layer: int, b: np.ndarray) -> np.ndarray:
    gate = P[f"h{layer}.mlp.Wgate_exps"]
    up = P[f"h{layer}.mlp.Wup_exps"]
    down = P[f"h{layer}.mlp.Wdown_exps"]
    router = P[f"h{layer}.mlp.Wrouter"]
    probs = _softmax_ref(b @ router, axis=-1)
    out = np.zeros((b.shape[0], cfg.d_model), np.float32)
    for t in range(b.shape[0]):
        for e in range(gate.shape[0]):
            out[t] += probs[t, e] * _expert_ffn_ref(b[t], gate[e], up[e], down[e])
    if f"h{layer}.mlp.Wgate_shexp" in P:
        out += _expert_ffn_ref(
            b, P[f"h{layer}.mlp.Wgate_shexp"], P[f"h{layer}.mlp.Wup_shexp"],
            P[f"h{layer}.mlp.Wdown_shexp"])
    return out.astype(np.float32)


def _moe_case(*, n_expert: int = 4, d_model: int = 6, d_ff: int = 5,
              shared_ff: int = 3, top_k: int = 2, seed: int = 7) -> tuple[dict, M.ModelConfig, np.ndarray]:
    rng = np.random.default_rng(seed)
    P = {
        "h0.mlp.Wrouter": rng.standard_normal((d_model, n_expert)).astype(np.float32) * 0.4,
        "h0.mlp.Wgate_exps": rng.standard_normal((n_expert, d_model, d_ff)).astype(np.float32) * 0.3,
        "h0.mlp.Wup_exps": rng.standard_normal((n_expert, d_model, d_ff)).astype(np.float32) * 0.3,
        "h0.mlp.Wdown_exps": rng.standard_normal((n_expert, d_ff, d_model)).astype(np.float32) * 0.3,
    }
    if shared_ff:
        P.update({
            "h0.mlp.Wgate_shexp": rng.standard_normal((d_model, shared_ff)).astype(np.float32) * 0.3,
            "h0.mlp.Wup_shexp": rng.standard_normal((d_model, shared_ff)).astype(np.float32) * 0.3,
            "h0.mlp.Wdown_shexp": rng.standard_normal((shared_ff, d_model)).astype(np.float32) * 0.3,
        })
    cfg = M.ModelConfig(d_model=d_model, n_head=2, n_layer=1, d_ff=d_ff,
                        expert_count=n_expert, expert_used_count=top_k)
    b = rng.standard_normal((4, d_model)).astype(np.float32)
    return P, cfg, b


@pytest.fixture(scope="module")
def moe_ckpt(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("moe") / "moe.gguf")
    build_moe_llama_gguf(
        path, n_layer=6, d_model=32, n_head=4, n_kv_head=2, d_ff=32,
        expert_count=4, expert_used_count=2, expert_ff=768, shared_ff=16,
        vocab_size=128, dense_layers=(0,), seed=20240820)
    return path, G.read_info(path)


def test_moe_metadata_and_layer_bytes_are_detected(moe_ckpt):
    _path, info = moe_ckpt
    assert info.config.expert_count == 4
    assert info.config.expert_used_count == 2
    assert info.config.expert_feed_forward_length == 768
    assert info.config.expert_shared_feed_forward_length == 16
    assert [m.is_moe for m in info.moe_layers] == [False, True, True, True, True, True]
    assert all(m.expert_count == 4 for m in info.moe_layers[1:])
    assert info.layer_bytes[1] > info.layer_bytes[0] * 10


def test_load_params_keeps_all_experts_of_a_layer_together(moe_ckpt):
    path, info = moe_ckpt
    p = G.load_params(path, 1, 2, False, False, info=info)
    assert {"h1.mlp.Wrouter", "h1.mlp.Wgate_exps", "h1.mlp.Wup_exps",
            "h1.mlp.Wdown_exps"}.issubset(p)
    assert p["h1.mlp.Wgate_exps"].shape == (4, 32, 768)
    assert p["h1.mlp.Wdown_exps"].shape == (4, 768, 32)
    assert not any(k.startswith("h0.") or k.startswith("h2.") for k in p)


def test_byte_partitioner_balances_non_uniform_layers(moe_ckpt):
    _path, info = moe_ckpt
    nodes = [_node("n1"), _node("n2"), _node("n3"), _node("n4"), _node("n5")]
    parts = topology_partition_by_bytes(nodes, info.layer_bytes)
    assert parts
    G.validate_moe_expert_colocation(info, parts)
    by_node = {
        p.node_id: G.layer_span_bytes(info, p.start_layer, p.end_layer)
        for p in parts
    }
    cap = int(2 * 0.8 * 1024 * 1024)
    assert all(used <= cap for used in by_node.values())
    naive = [(0, 2), (2, 4), (4, 5), (5, 6)]
    assert any(G.layer_span_bytes(info, a, b) > cap for a, b in naive)


def test_partition_loaded_params_keep_moe_layer_experts_together(moe_ckpt):
    path, info = moe_ckpt
    parts = topology_partition_by_bytes(
        [_node("n1"), _node("n2"), _node("n3"), _node("n4"), _node("n5")],
        info.layer_bytes)
    owners: dict[int, dict[str, set[str]]] = {}
    expected = {"mlp.Wrouter", "mlp.Wgate_exps", "mlp.Wup_exps", "mlp.Wdown_exps",
                "mlp.Wgate_shexp", "mlp.Wup_shexp", "mlp.Wdown_shexp"}
    for p in parts:
        params = G.load_params(path, p.start_layer, p.end_layer,
                               p.is_first, p.is_last, info=info)
        for name in params:
            if not name.startswith("h"):
                continue
            layer_text, suffix = name.split(".", 1)
            if suffix not in expected:
                continue
            owners.setdefault(int(layer_text[1:]), {}).setdefault(suffix, set()).add(p.node_id)
    for m in info.moe_layers:
        if not m.is_moe:
            continue
        assert expected.issubset(owners[m.layer])
        assert len(set().union(*owners[m.layer].values())) == 1


def test_pool_too_small_fails_honestly(moe_ckpt):
    _path, info = moe_ckpt
    nodes = [_node("n1", ram_mb=1)]
    assert topology_partition_by_bytes(nodes, info.layer_bytes) == []
    fit = explain_layer_bytes_fit(nodes, info.layer_bytes)
    assert not fit.fits
    assert fit.bytes_needed > fit.bytes_available or "largest indivisible layer" in fit.reason
    assert "bytes" in fit.reason


def test_expert_splitting_is_detected(moe_ckpt):
    _path, info = moe_ckpt
    parts = [LayerAssignment(node_id="n1", url="u", start_layer=0, end_layer=6)]
    split = {name: "n1" for name in info.moe_layers[1].tensor_names}
    split[info.moe_layers[1].tensor_names[0]] = "n2"
    with pytest.raises(ValueError, match="split across nodes"):
        G.validate_moe_expert_colocation(info, parts, split)


def test_moe_sharded_generation_is_bit_exact(moe_ckpt):
    path, info = moe_ckpt
    cfg = info.config
    full = G.load_full_params(path, info)
    prompt = [1, 5, 9, 13]
    single_logits = M.forward_full(full, cfg, prompt)
    reference = M.generate_with(lambda ctx: M.forward_full(full, cfg, ctx), cfg,
                                prompt, max_new_tokens=6, temperature=0.0)

    parts = topology_partition_by_bytes(
        [_node("n1"), _node("n2"), _node("n3"), _node("n4"), _node("n5")],
        info.layer_bytes)

    def run(ctx):
        h: object = ctx
        is_tokens = True
        for p in parts:
            params = G.load_params(path, p.start_layer, p.end_layer,
                                   p.is_first, p.is_last, info=info)
            shard = M.ShardModule(cfg, p.start_layer, p.end_layer,
                                  p.is_first, p.is_last, params=params)
            h = shard.forward(h, is_tokens=is_tokens)
            is_tokens = False
        return np.asarray(h, dtype=np.float32)

    sharded_logits = run(prompt)
    assert np.array_equal(sharded_logits, single_logits)
    sharded = M.generate_with(run, cfg, prompt, max_new_tokens=6, temperature=0.0)
    assert sharded == reference
    assert len(set(reference)) > 1


def test_gguf_backend_uses_measured_layer_bytes(moe_ckpt):
    path, info = moe_ckpt
    backend = adapters.GGUFBackend()
    cfg = backend.resolve_config(path)
    assert cfg is not None and cfg.layer_bytes == tuple(info.layer_bytes)
    parts = backend.plan_partition([_node("n1"), _node("n2"), _node("n3"), _node("n4"), _node("n5")], cfg)
    assert parts
    G.validate_moe_expert_colocation(info, parts)


class TestMoEMathOracle:
    def test_identical_experts_collapse_to_dense_ffn_for_any_router(self):
        P, cfg, b = _moe_case(shared_ff=0, top_k=2)
        dense_gate = P["h0.mlp.Wgate_exps"][0].copy()
        dense_up = P["h0.mlp.Wup_exps"][0].copy()
        dense_down = P["h0.mlp.Wdown_exps"][0].copy()
        for e in range(cfg.expert_count):
            P["h0.mlp.Wgate_exps"][e] = dense_gate
            P["h0.mlp.Wup_exps"][e] = dense_up
            P["h0.mlp.Wdown_exps"][e] = dense_down
        P["h0.mlp.Wrouter"] = np.array([
            [3.0, -1.0, 0.5, 2.0],
            [-0.5, 2.5, 1.5, 0.0],
            [1.0, 0.0, -2.0, 4.0],
            [0.25, -0.75, 1.25, 2.25],
            [2.0, 1.0, 0.0, -1.0],
            [-1.5, 0.5, 3.5, 1.0],
        ], dtype=np.float32)
        expected = _expert_ffn_ref(b, dense_gate, dense_up, dense_down).astype(np.float32)
        got = M._llama_moe(P, cfg, 0, b)
        assert np.allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_one_hot_router_selects_each_expert_exactly(self):
        P, cfg, _b = _moe_case(shared_ff=0, top_k=1)
        b = np.abs(np.random.default_rng(9).standard_normal((3, cfg.d_model))).astype(np.float32) + 0.25
        for expert in range(cfg.expert_count):
            P["h0.mlp.Wrouter"] = np.zeros((cfg.d_model, cfg.expert_count), np.float32)
            P["h0.mlp.Wrouter"][:, expert] = 100.0
            got = M._llama_moe(P, cfg, 0, b)
            expected = np.vstack([
                _expert_ffn_ref(row, P["h0.mlp.Wgate_exps"][expert],
                                P["h0.mlp.Wup_exps"][expert],
                                P["h0.mlp.Wdown_exps"][expert])
                for row in b
            ]).astype(np.float32)
            assert np.allclose(got, expected, rtol=1e-6, atol=1e-6), expert

    def test_full_routing_matches_independent_reference_loop(self):
        P, cfg, b = _moe_case(shared_ff=3, top_k=4)
        got = M._llama_moe(P, cfg, 0, b)
        expected = _naive_moe_ref(P, cfg, 0, b)
        assert np.allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_shared_expert_term_is_numerically_required(self):
        P, cfg, b = _moe_case(shared_ff=3, top_k=2)
        with_shared = M._llama_moe(P, cfg, 0, b)
        no_shared = {k: v for k, v in P.items() if "_shexp" not in k}
        without_shared = M._llama_moe(no_shared, cfg, 0, b)
        assert not np.allclose(with_shared, without_shared, rtol=1e-6, atol=1e-6)

    def test_top_k_router_weights_are_renormalized_to_one(self):
        P, cfg, b = _moe_case(shared_ff=0, top_k=2)
        dense_gate = P["h0.mlp.Wgate_exps"][0].copy()
        dense_up = P["h0.mlp.Wup_exps"][0].copy()
        dense_down = P["h0.mlp.Wdown_exps"][0].copy()
        for e in range(cfg.expert_count):
            P["h0.mlp.Wgate_exps"][e] = dense_gate
            P["h0.mlp.Wup_exps"][e] = dense_up
            P["h0.mlp.Wdown_exps"][e] = dense_down
        logits = b @ P["h0.mlp.Wrouter"]
        probs = _softmax_ref(logits, axis=-1)
        chosen = np.argsort(probs, axis=-1)[:, -cfg.expert_used_count:]
        applied = probs[np.arange(b.shape[0])[:, None], chosen]
        applied = applied / applied.sum(axis=-1, keepdims=True)
        assert np.allclose(applied.sum(axis=-1), 1.0)
        expected = _expert_ffn_ref(b, dense_gate, dense_up, dense_down).astype(np.float32)
        got = M._llama_moe(P, cfg, 0, b)
        assert np.allclose(got, expected, rtol=1e-6, atol=1e-6)
