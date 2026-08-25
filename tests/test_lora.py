"""Tests for single-machine LoRA fine-tuning (``meshpool.inference.lora``).

The correctness gate is the finite-difference gradient check (the same technique that
guards the base backward pass in ``test_train``). The behavioural gates prove a frozen
base actually *learns*: training loss falls on a fixed corpus, the base weights are
never mutated, and merged adapters reproduce the effective forward.
"""
from __future__ import annotations

import numpy as np

from meshpool.inference import lora
from meshpool.inference import model as M
from meshpool.inference.model import ModelConfig
from meshpool.inference.train import ce_loss, forward_cache


def _tiny():
    text = "meshpool shares spare compute with the whole world. " * 40
    itos = sorted(set(text))
    cfg = ModelConfig(name="tiny-lora", vocab_size=len(itos), d_model=32, n_head=4,
                      n_layer=2, block_size=16, d_ff=128, seed=0)
    return cfg, M.random_params(cfg), text, itos


def _fixed_batch(cfg, text, itos, n=6):
    data = np.asarray(M.CharTokenizer(itos=itos, vocab_size=cfg.vocab_size).encode(text),
                      dtype=np.int64)
    T = cfg.block_size
    batch = np.stack([data[i:i + T + 1] for i in range(n)])
    return batch[:, :-1], batch[:, 1:]


def _loss(P, cfg, xb, yb):
    logits, _ = forward_cache(P, cfg, xb)
    return ce_loss(logits, yb)[0]


def test_lora_gradient_check_matches_numeric():
    """The dA/dB projection matches central differences to < 1e-4 (float64)."""
    worst = lora.gradient_check_lora()
    assert worst < 1e-4, f"LoRA gradient check worst rel error too high: {worst:.2e}"


def test_zero_init_adapter_is_identity():
    """Fresh adapters (B=0) leave the model output exactly unchanged — a fine-tune
    starts from the pretrained model."""
    cfg, P, _, _ = _tiny()
    lc = lora.LoRAConfig(rank=4, alpha=8.0, targets=lora.ALL_TARGETS, seed=0)
    adapters = lora.init_adapters(P, lc, zero_b=True)
    xb = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    base_logits, _ = forward_cache(P, cfg, xb)
    eff_logits, _ = forward_cache(lora.effective_params(P, adapters, lc.scale), cfg, xb)
    assert np.allclose(base_logits, eff_logits, atol=1e-6)


def test_lora_training_reduces_loss_and_freezes_base():
    """Training drives the loss down on a fixed corpus while the base stays frozen."""
    cfg, P, text, itos = _tiny()
    before = {k: v.copy() for k, v in P.items()}
    lc = lora.LoRAConfig(rank=8, alpha=16.0, targets=lora.ALL_TARGETS, seed=1)
    adapters, history = lora.train_lora(P, cfg, text, lc, steps=250, batch_size=8,
                                        lr=8e-3, itos=itos, seed=1)

    assert len(history) == 250
    assert np.isfinite(history).all(), "loss went non-finite (training diverged)"
    assert history[-1] < history[0], f"loss did not fall: {history[0]:.3f} -> {history[-1]:.3f}"
    assert np.mean(history[-25:]) < np.mean(history[:25]) - 0.1, (
        f"insufficient improvement: {np.mean(history[:25]):.3f} -> {np.mean(history[-25:]):.3f}")
    # frozen base: train_lora must not mutate P_base
    for k in before:
        assert np.array_equal(before[k], P[k]), f"base weight {k} was mutated"


def test_merge_reproduces_effective_forward_and_beats_base():
    """Merged weights equal the effective (base+adapter) forward, and a trained adapter
    lowers loss on the corpus versus the frozen base — proof the adapter carries signal."""
    cfg, P, text, itos = _tiny()
    lc = lora.LoRAConfig(rank=8, alpha=16.0, targets=lora.ALL_TARGETS, seed=2)
    adapters, _ = lora.train_lora(P, cfg, text, lc, steps=200, batch_size=8, lr=8e-3,
                                  itos=itos, seed=2)

    merged = lora.merge_adapters(P, adapters, lc.scale)
    eff = lora.effective_params(P, adapters, lc.scale)
    for k in adapters:
        assert np.allclose(merged[k], eff[k], atol=1e-5)
        assert not np.allclose(merged[k], P[k]), f"adapter left {k} unchanged"

    xb, yb = _fixed_batch(cfg, text, itos)
    assert _loss(merged, cfg, xb, yb) < _loss(P, cfg, xb, yb)


def test_lora_save_load_roundtrip(tmp_path):
    """Adapters + config survive a save/load cycle byte-for-byte."""
    cfg, P, _, _ = _tiny()
    lc = lora.LoRAConfig(rank=4, alpha=8.0, targets=lora.DEFAULT_TARGETS, seed=3)
    adapters = lora.init_adapters(P, lc, zero_b=False)
    path = tmp_path / "adapter.npz"
    lora.save_lora(path, adapters, lc, base_model=cfg.name)

    loaded, lc2, base = lora.load_lora(path)
    assert base == cfg.name
    assert (lc2.rank, lc2.alpha, tuple(lc2.targets)) == (lc.rank, lc.alpha, tuple(lc.targets))
    assert set(loaded) == set(adapters)
    for k in adapters:
        assert np.allclose(loaded[k]["A"], adapters[k]["A"])
        assert np.allclose(loaded[k]["B"], adapters[k]["B"])


def test_targets_selection_attn_only_vs_all():
    """The attn preset adapts 2 matrices/layer; 'all' adds the 2 MLP matrices."""
    cfg, P, _, _ = _tiny()
    attn = lora.target_keys(P, lora.LoRAConfig(targets=lora.DEFAULT_TARGETS))
    allk = lora.target_keys(P, lora.LoRAConfig(targets=lora.ALL_TARGETS))
    assert len(attn) == 2 * cfg.n_layer
    assert len(allk) == 4 * cfg.n_layer
    assert set(attn).issubset(set(allk))
