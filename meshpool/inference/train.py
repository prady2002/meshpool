"""Trainer for the demo transformer (produces ``weights/<model>.npz``).

The distributed inference engine serves whatever checkpoint is on disk; this module
*creates* that checkpoint by training the tiny model on an **original, copyright-free
MeshPool-themed corpus** (see ``corpus.py``). Training is deliberately **single-process**
— per the spec/plan the MVP is *inference*-only distributed; distributed training is a
documented non-goal — so this is an offline pre-step, not a network workload.

Everything here is hand-written NumPy: forward-with-cache, a full backward pass, and an
Adam optimiser. Because hand-derived gradients are easy to get subtly wrong, the module
ships a **numerical gradient check** (``gradient_check``) that is run as a gate before we
trust any training run.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import model as M
from .model import ModelConfig, _gelu, _layernorm, _softmax


# --------------------------------------------------------------------------- #
# Forward pass that stashes the intermediates the backward pass needs.
# --------------------------------------------------------------------------- #
def forward_cache(P: dict, cfg: ModelConfig, tokens) -> tuple[np.ndarray, dict]:
    """Batched forward. ``tokens`` is an int array of shape (B, T); returns logits
    (B, T, V) and a cache of intermediates for the backward pass."""
    toks = np.asarray(tokens, dtype=np.int64)
    if toks.ndim == 1:
        toks = toks[None, :]
    B, T = toks.shape
    nh, hd, D = cfg.n_head, cfg.head_dim, cfg.d_model
    x = P["wte"][toks] + P["wpe"][:T]                        # (B,T,D)
    cache: dict = {"tokens": toks, "B": B, "T": T, "layers": []}
    mask = (np.triu(np.ones((T, T)), k=1) * -1e9).astype(x.dtype)

    for i in range(cfg.n_layer):
        lc: dict = {"x_in": x}
        a = _layernorm(x, P[f"h{i}.ln1.g"], P[f"h{i}.ln1.b"])
        lc["a"] = a
        qkv = a @ P[f"h{i}.attn.Wqkv"]                       # (B,T,3D)
        q, k, v = np.split(qkv, 3, axis=-1)
        qh = q.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)   # (B,nh,T,hd)
        kh = k.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        vh = v.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        att = (qh @ kh.transpose(0, 1, 3, 2)) / math.sqrt(hd)  # (B,nh,T,T)
        p = _softmax(att + mask, axis=-1)
        oh = p @ vh                                          # (B,nh,T,hd)
        o = oh.transpose(0, 2, 1, 3).reshape(B, T, D)
        attn_out = o @ P[f"h{i}.attn.Wo"]
        x1 = x + attn_out
        lc.update({"qh": qh, "kh": kh, "vh": vh, "p": p, "o": o, "x1": x1})
        b = _layernorm(x1, P[f"h{i}.ln2.g"], P[f"h{i}.ln2.b"])
        pre = b @ P[f"h{i}.mlp.W1"] + P[f"h{i}.mlp.b1"]
        h = _gelu(pre)
        m = h @ P[f"h{i}.mlp.W2"] + P[f"h{i}.mlp.b2"]
        x = x1 + m
        lc.update({"b": b, "pre": pre, "h": h})
        cache["layers"].append(lc)

    xf = _layernorm(x, P["lnf.g"], P["lnf.b"])
    logits = xf @ P["wte"].T                                 # (B,T,V)
    cache["x_final"] = x
    cache["xf"] = xf
    return logits, cache


def _layernorm_backward(dy, x, g, eps=1e-5):
    """LayerNorm backward over the last axis, summing param grads over all leading
    (batch/time) axes."""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * inv
    lead = tuple(range(dy.ndim - 1))
    dg = (dy * xhat).sum(lead)
    db = dy.sum(lead)
    dxhat = dy * g
    dx = inv * (dxhat - dxhat.mean(-1, keepdims=True)
                - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db


def _gelu_backward(dh, pre):
    c = 0.7978845608
    u = c * (pre + 0.044715 * pre ** 3)
    t = np.tanh(u)
    dudx = c * (1.0 + 3.0 * 0.044715 * pre ** 2)
    dgelu = 0.5 * (1.0 + t) + 0.5 * pre * (1.0 - t ** 2) * dudx
    return dh * dgelu


def backward(P: dict, cfg: ModelConfig, cache: dict, dlogits: np.ndarray) -> dict:
    B, T, nh, hd, D = cache["B"], cache["T"], cfg.n_head, cfg.head_dim, cfg.d_model
    grads = {k: np.zeros_like(v) for k, v in P.items()}

    # logits = xf @ wte.T   (sum weight grad over batch+time)
    grads["wte"] += np.einsum("btv,btd->vd", dlogits, cache["xf"])
    dxf = dlogits @ P["wte"]                                 # (B,T,D)
    dx, dgf, dbf = _layernorm_backward(dxf, cache["x_final"], P["lnf.g"])
    grads["lnf.g"] += dgf
    grads["lnf.b"] += dbf

    for i in reversed(range(cfg.n_layer)):
        lc = cache["layers"][i]
        dx1 = dx.copy()
        dm = dx
        # m = h @ W2 + b2
        grads[f"h{i}.mlp.W2"] += np.einsum("btf,btd->fd", lc["h"], dm)
        grads[f"h{i}.mlp.b2"] += dm.sum((0, 1))
        dh = dm @ P[f"h{i}.mlp.W2"].T
        dpre = _gelu_backward(dh, lc["pre"])
        grads[f"h{i}.mlp.W1"] += np.einsum("btd,btf->df", lc["b"], dpre)
        grads[f"h{i}.mlp.b1"] += dpre.sum((0, 1))
        db = dpre @ P[f"h{i}.mlp.W1"].T
        dx1b, dg2, db2 = _layernorm_backward(db, lc["x1"], P[f"h{i}.ln2.g"])
        grads[f"h{i}.ln2.g"] += dg2
        grads[f"h{i}.ln2.b"] += db2
        dx1 += dx1b
        # x1 = x_in + attn_out
        dx_in = dx1.copy()
        d_attn_out = dx1
        # attn_out = o @ Wo
        grads[f"h{i}.attn.Wo"] += np.einsum("btd,bte->de", lc["o"], d_attn_out)
        do = d_attn_out @ P[f"h{i}.attn.Wo"].T
        doh = do.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)   # (B,nh,T,hd)
        p = lc["p"]
        # oh = p @ vh
        dp = doh @ lc["vh"].transpose(0, 1, 3, 2)              # (B,nh,T,T)
        dvh = p.transpose(0, 1, 3, 2) @ doh                    # (B,nh,T,hd)
        ds = p * (dp - (dp * p).sum(-1, keepdims=True))
        ds = ds / math.sqrt(hd)
        dqh = ds @ lc["kh"]                                    # (B,nh,T,hd)
        dkh = ds.transpose(0, 1, 3, 2) @ lc["qh"]              # (B,nh,T,hd)
        dq = dqh.transpose(0, 2, 1, 3).reshape(B, T, D)
        dk = dkh.transpose(0, 2, 1, 3).reshape(B, T, D)
        dv = dvh.transpose(0, 2, 1, 3).reshape(B, T, D)
        dqkv = np.concatenate([dq, dk, dv], axis=-1)           # (B,T,3D)
        grads[f"h{i}.attn.Wqkv"] += np.einsum("btd,bte->de", lc["a"], dqkv)
        da = dqkv @ P[f"h{i}.attn.Wqkv"].T
        dx_in_b, dg1, db1 = _layernorm_backward(da, lc["x_in"], P[f"h{i}.ln1.g"])
        grads[f"h{i}.ln1.g"] += dg1
        grads[f"h{i}.ln1.b"] += db1
        dx_in += dx_in_b
        dx = dx_in

    # embedding: x0 = wte[tokens] + wpe[:T]
    np.add.at(grads["wte"], cache["tokens"].ravel(), dx.reshape(-1, D))
    grads["wpe"][:T] += dx.sum(0)
    return grads


def ce_loss(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean cross-entropy over (batch, time); returns (loss, dlogits) with dlogits
    the same shape as ``logits``."""
    logits2 = logits.reshape(-1, logits.shape[-1])
    tgt = np.asarray(targets).reshape(-1)
    probs = _softmax(logits2, axis=-1)
    N = logits2.shape[0]
    logp = np.log(probs[np.arange(N), tgt] + 1e-12)
    loss = float(-logp.mean())
    d = probs.copy()
    d[np.arange(N), tgt] -= 1.0
    d /= N
    return loss, d.reshape(logits.shape)


# --------------------------------------------------------------------------- #
# Gradient check (the correctness gate for the hand-derived backward pass)
# --------------------------------------------------------------------------- #
def gradient_check(seed: int = 0, n_samples: int = 40, eps: float = 1e-4) -> float:
    """Numerically verify the hand-derived backward pass in float64 (float32 rounding
    swamps central differences, so the check *must* run in double precision)."""
    cfg = ModelConfig(name="tiny", vocab_size=11, d_model=16, n_head=2, n_layer=2,
                      block_size=8, d_ff=32, seed=seed)
    g = np.random.default_rng(seed)
    P = {k: v.astype(np.float64) for k, v in M.random_params(cfg).items()}
    # make layernorm/bias params non-trivial so their grads are exercised
    for k in list(P):
        if k.endswith(".g") or k.endswith(".b") or "b1" in k or "b2" in k:
            P[k] = P[k] + g.standard_normal(P[k].shape) * 0.3
    tokens = list(g.integers(0, cfg.vocab_size, size=(2, cfg.block_size)))
    x = np.stack([t[:-1] for t in tokens])
    y = np.stack([t[1:] for t in tokens])

    logits, cache = forward_cache(P, cfg, x)
    loss, dlogits = ce_loss(logits, y)
    grads = backward(P, cfg, cache, dlogits)

    def loss_at(params):
        lg, _ = forward_cache(params, cfg, x)
        return ce_loss(lg, y)[0]

    keys = list(P.keys())
    worst = 0.0
    checked = 0
    while checked < n_samples:
        k = keys[g.integers(0, len(keys))]
        idx = tuple(int(g.integers(0, s)) for s in P[k].shape)
        orig = P[k][idx]
        P[k][idx] = orig + eps
        lp = loss_at(P)
        P[k][idx] = orig - eps
        lm = loss_at(P)
        P[k][idx] = orig
        num = (lp - lm) / (2 * eps)
        ana = grads[k][idx]
        if max(abs(num), abs(ana)) < 1e-7:      # both ~0: uninformative, skip
            continue
        rel = abs(num - ana) / (abs(num) + abs(ana))
        worst = max(worst, rel)
        checked += 1
    return worst


# --------------------------------------------------------------------------- #
# Adam training loop
# --------------------------------------------------------------------------- #
def train(cfg: ModelConfig, text: str, steps: int = 2000, batch_size: int = 16,
          lr: float = 3e-3, seed: int = 1337, log_every: int = 100,
          itos: Optional[list[str]] = None, warmup: int = 100,
          train_block: Optional[int] = None) -> tuple[dict, list[str]]:
    tok = M.CharTokenizer(itos=itos, vocab_size=cfg.vocab_size)
    data = np.asarray(tok.encode(text), dtype=np.int64)
    T = min(cfg.block_size, train_block or cfg.block_size)
    if len(data) <= T + 1:
        raise ValueError("corpus too small for block size")
    P = M.random_params(cfg)
    # GPT-2 init trick: scale residual-path output projections by 1/sqrt(2*n_layer)
    # so the residual stream variance stays controlled with depth (stabilises training).
    resid_scale = 1.0 / math.sqrt(2 * cfg.n_layer)
    for i in range(cfg.n_layer):
        P[f"h{i}.attn.Wo"] *= resid_scale
        P[f"h{i}.mlp.W2"] *= resid_scale
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(v) for k, v in P.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * step / max(1, warmup)
        prog = (step - warmup) / max(1, steps - warmup)
        return 0.1 * lr + 0.5 * (0.9 * lr) * (1 + math.cos(math.pi * prog))

    for step in range(1, steps + 1):
        starts = rng.integers(0, len(data) - T - 1, size=batch_size)
        batch = np.stack([data[i:i + T + 1] for i in starts])  # (B, T+1)
        xb, yb = batch[:, :-1], batch[:, 1:]
        logits, cache = forward_cache(P, cfg, xb)
        loss, dlogits = ce_loss(logits, yb)
        grads = backward(P, cfg, cache, dlogits)
        cur_lr = lr_at(step)
        for k in P:
            grad = grads[k]
            m[k] = b1 * m[k] + (1 - b1) * grad
            v[k] = b2 * v[k] + (1 - b2) * (grad * grad)
            mhat = m[k] / (1 - b1 ** step)
            vhat = v[k] / (1 - b2 ** step)
            P[k] -= (cur_lr * mhat / (np.sqrt(vhat) + eps)).astype(np.float32)
        if step % log_every == 0 or step == 1:
            dt = time.perf_counter() - t0
            print(f"  step {step:4d}/{steps}  loss {loss:.4f}  lr {cur_lr:.1e}  "
                  f"({step / max(1e-6, dt):.1f} steps/s)")
    return P, tok.itos


def save_checkpoint(path: Path, P: dict, itos: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v.astype(np.float32) for k, v in P.items()}
    arrays["__itos__"] = np.asarray(itos, dtype=object)
    np.savez(path, **arrays)


def sample(P: dict, cfg: ModelConfig, itos: list[str], prompt: str,
           n: int = 160, temperature: float = 0.6, seed: int = 0) -> str:
    tok = M.CharTokenizer(itos=itos, vocab_size=cfg.vocab_size)
    ids = tok.encode(prompt) or [tok.encode(" ")[0]]
    out = M.generate_with(lambda ctx: M.forward_full(P, cfg, ctx), cfg, ids,
                          max_new_tokens=n, temperature=temperature, seed=seed)
    return tok.decode(out)


def build_default_checkpoint(steps: int = 1600, batch_size: int = 32, lr: float = 3e-3,
                             seed: int = 1337, warmup: int = 120,
                             train_block: int = 64) -> Path:
    """Gradient-check, then train ``nano-gpt`` on the MeshPool corpus and save it."""
    from .corpus import build_corpus

    print("== gradient check (backward-pass correctness gate) ==")
    worst = gradient_check()
    print(f"  worst relative error: {worst:.2e}  ->  {'PASS' if worst < 1e-4 else 'FAIL'}")
    if worst >= 1e-4:
        raise RuntimeError("gradient check failed; refusing to train with wrong gradients")

    cfg = M.REGISTRY["nano-gpt"]
    text = build_corpus()
    print(f"== training {cfg.name}: {cfg.n_layer} layers, d_model {cfg.d_model}, "
          f"corpus {len(text)} chars, {steps} steps x batch {batch_size} (ctx {train_block}) ==")
    P, itos = train(cfg, text, steps=steps, batch_size=batch_size, lr=lr, seed=seed,
                    warmup=warmup, train_block=train_block, log_every=100)
    out = M.WEIGHTS_DIR / f"{cfg.name}.npz"
    save_checkpoint(out, P, itos)
    print(f"== saved checkpoint -> {out} ==")
    print("== sample generations ==")
    for temp in (0.3, 0.6):
        print(f"  [temp {temp}] {sample(P, cfg, itos, 'MeshPool ', n=200, temperature=temp)!r}")
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Train the MeshPool demo model")
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    if args.check_only:
        w = gradient_check()
        print(f"gradient check worst rel error: {w:.2e} -> {'PASS' if w < 1e-4 else 'FAIL'}")
        return
    build_default_checkpoint(steps=args.steps, batch_size=args.batch, lr=args.lr,
                             train_block=args.block)


if __name__ == "__main__":
    main()
