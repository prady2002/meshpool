"""LoRA fine-tuning of the nano-GPT — single-machine, frozen base (spec: "train models").

Low-Rank Adaptation (LoRA) freezes the pretrained weights and learns a small,
low-rank update ``ΔW = (alpha/r) · A @ B`` injected into selected projection
matrices (by default the attention ``Wqkv``/``Wo``). Only ``A`` and ``B`` are
trained, so a fine-tune is tiny (kilobytes) and cheap enough to run on one machine
in seconds — the honest, deliverable slice of "train models" today. *Distributed*
LoRA over the sharded pipeline (activation-gradient transport across stages) is a
documented next step, not this module.

Why this is trustworthy rather than a fresh, error-prone backward pass: we reuse the
**already gradient-checked** ``train.forward_cache`` / ``train.backward``. We build the
effective weight ``W_eff = W + s·A@B`` at each targeted key, let the existing backward
return ``G = dL/dW_eff`` for that matrix (it differentiates whatever value sits at the
key), then apply the exact chain rule for the low-rank factors::

    dL/dA = s · G @ Bᵀ         dL/dB = s · Aᵀ @ G

That projection is the only new math, and ``gradient_check_lora`` finite-difference
verifies it in float64 exactly like ``train.gradient_check`` guards the base backward.
The base parameters are never mutated (``effective_params`` copies only targeted keys),
so the pretrained model stays frozen.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import model as M
from .model import ModelConfig
from .train import backward, ce_loss, forward_cache

# Classic LoRA targets the attention projections; MLP is opt-in via ``targets``.
DEFAULT_TARGETS: tuple[str, ...] = ("attn.Wqkv", "attn.Wo")
ALL_TARGETS: tuple[str, ...] = ("attn.Wqkv", "attn.Wo", "mlp.W1", "mlp.W2")


@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    targets: tuple[str, ...] = DEFAULT_TARGETS
    seed: int = 0

    @property
    def scale(self) -> float:
        """LoRA scaling ``alpha/r`` applied to the low-rank update."""
        return self.alpha / float(self.rank)


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #
def target_keys(P: dict, lc: LoRAConfig) -> list[str]:
    """The 2-D weight matrices in ``P`` that match the configured target suffixes."""
    keys = [k for k in P if P[k].ndim == 2 and any(k.endswith(t) for t in lc.targets)]
    return sorted(keys)


def init_adapters(P: dict, lc: LoRAConfig, *, dtype=np.float32, zero_b: bool = True,
                  rng: Optional[np.random.Generator] = None) -> dict:
    """Create ``{key: {"A": (in,r), "B": (r,out)}}`` adapters for every targeted matrix.

    Standard LoRA init: ``A`` small-random, ``B`` zero so the initial ``ΔW`` is exactly
    zero and fine-tuning *starts from the pretrained model*. ``zero_b=False`` (used by
    the gradient check) makes both factors non-trivial so ``dA`` is exercised.
    """
    if rng is None:
        rng = np.random.default_rng(lc.seed)
    if not target_keys(P, lc):
        raise ValueError(f"no LoRA-able matrices match targets {lc.targets!r}")
    adapters: dict[str, dict[str, np.ndarray]] = {}
    for k in target_keys(P, lc):
        in_dim, out_dim = P[k].shape
        A = (rng.standard_normal((in_dim, lc.rank)) / math.sqrt(in_dim)).astype(dtype)
        if zero_b:
            B = np.zeros((lc.rank, out_dim), dtype=dtype)
        else:
            B = (rng.standard_normal((lc.rank, out_dim)) / math.sqrt(lc.rank)).astype(dtype)
        adapters[k] = {"A": A, "B": B}
    return adapters


def adapter_delta(ab: dict, scale: float) -> np.ndarray:
    """The dense low-rank update ``s·A@B`` for one adapter."""
    return scale * (ab["A"] @ ab["B"])


def effective_params(P: dict, adapters: dict, scale: float) -> dict:
    """A parameter dict with ``W_eff = W + s·A@B`` at each adapted key.

    Non-adapted arrays are shared by reference (read-only in the forward pass); the
    base ``P`` is never mutated, so the pretrained weights stay frozen.
    """
    Pe = dict(P)
    for k, ab in adapters.items():
        Pe[k] = P[k] + adapter_delta(ab, scale)
    return Pe


def adapter_grads(weff_grads: dict, adapters: dict, scale: float) -> dict:
    """Project full-matrix grads ``G = dL/dW_eff`` onto the low-rank factors."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for k, ab in adapters.items():
        G = weff_grads[k]
        out[k] = {"A": scale * (G @ ab["B"].T), "B": scale * (ab["A"].T @ G)}
    return out


def loss_and_grads(P_base: dict, cfg: ModelConfig, adapters: dict, scale: float,
                   xb: np.ndarray, yb: np.ndarray) -> tuple[float, dict]:
    """Forward+backward through the effective (base+adapter) weights; return the
    scalar loss and the adapter gradients (base grads are discarded — base is frozen)."""
    Pe = effective_params(P_base, adapters, scale)
    logits, cache = forward_cache(Pe, cfg, xb)
    loss, dlogits = ce_loss(logits, yb)
    wgrads = backward(Pe, cfg, cache, dlogits)
    return loss, adapter_grads(wgrads, adapters, scale)


# --------------------------------------------------------------------------- #
# Gradient check (correctness gate for the low-rank projection)
# --------------------------------------------------------------------------- #
def gradient_check_lora(seed: int = 0, n_samples: int = 40, eps: float = 1e-4) -> float:
    """Numerically verify the ``dA``/``dB`` projection in float64 (float32 rounding
    swamps central differences). Mirrors ``train.gradient_check``; both factors are
    initialised non-zero so ``dA`` is exercised."""
    cfg = ModelConfig(name="tiny", vocab_size=11, d_model=16, n_head=2, n_layer=2,
                      block_size=8, d_ff=32, seed=seed)
    g = np.random.default_rng(seed)
    P = {k: v.astype(np.float64) for k, v in M.random_params(cfg).items()}
    lc = LoRAConfig(rank=3, alpha=6.0, targets=ALL_TARGETS, seed=seed)
    adapters = init_adapters(P, lc, dtype=np.float64, zero_b=False, rng=g)

    toks = list(g.integers(0, cfg.vocab_size, size=(2, cfg.block_size)))
    x = np.stack([t[:-1] for t in toks])
    y = np.stack([t[1:] for t in toks])
    _, agrads = loss_and_grads(P, cfg, adapters, lc.scale, x, y)

    def loss_at() -> float:
        Pe = effective_params(P, adapters, lc.scale)
        lg, _ = forward_cache(Pe, cfg, x)
        return ce_loss(lg, y)[0]

    flat = [(k, which, adapters[k][which]) for k in adapters for which in ("A", "B")]
    worst = 0.0
    checked = 0
    while checked < n_samples:
        k, which, arr = flat[g.integers(0, len(flat))]
        idx = tuple(int(g.integers(0, s)) for s in arr.shape)
        orig = arr[idx]
        arr[idx] = orig + eps
        lp = loss_at()
        arr[idx] = orig - eps
        lm = loss_at()
        arr[idx] = orig
        num = (lp - lm) / (2 * eps)
        ana = agrads[k][which][idx]
        if max(abs(num), abs(ana)) < 1e-7:
            continue
        worst = max(worst, abs(num - ana) / (abs(num) + abs(ana)))
        checked += 1
    return worst


# --------------------------------------------------------------------------- #
# Training loop (Adam over the adapters only)
# --------------------------------------------------------------------------- #
def train_lora(P_base: dict, cfg: ModelConfig, text: str, lc: LoRAConfig, *,
               steps: int = 300, batch_size: int = 8, lr: float = 1e-2,
               seed: int = 1337, itos: Optional[list[str]] = None,
               train_block: Optional[int] = None, log_every: int = 0
               ) -> tuple[dict, list[float]]:
    """Fine-tune LoRA adapters on ``text`` (char-level) with a frozen base. Returns
    ``(adapters, loss_history)``. The base ``P_base`` is not modified."""
    tok = M.CharTokenizer(itos=itos, vocab_size=cfg.vocab_size)
    data = np.asarray(tok.encode(text), dtype=np.int64)
    T = min(cfg.block_size, train_block or cfg.block_size)
    if len(data) <= T + 1:
        raise ValueError("corpus too small for block size")

    rng = np.random.default_rng(seed)
    adapters = init_adapters(P_base, lc, dtype=np.float32, zero_b=True, rng=rng)
    mst = {k: {w: np.zeros_like(v) for w, v in ab.items()} for k, ab in adapters.items()}
    vst = {k: {w: np.zeros_like(v) for w, v in ab.items()} for k, ab in adapters.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    scale = lc.scale
    history: list[float] = []

    for step in range(1, steps + 1):
        starts = rng.integers(0, len(data) - T - 1, size=batch_size)
        batch = np.stack([data[i:i + T + 1] for i in starts])
        xb, yb = batch[:, :-1], batch[:, 1:]
        loss, agrads = loss_and_grads(P_base, cfg, adapters, scale, xb, yb)
        history.append(loss)
        for k in adapters:
            for w in ("A", "B"):
                grad = agrads[k][w]
                mst[k][w] = b1 * mst[k][w] + (1 - b1) * grad
                vst[k][w] = b2 * vst[k][w] + (1 - b2) * (grad * grad)
                mhat = mst[k][w] / (1 - b1 ** step)
                vhat = vst[k][w] / (1 - b2 ** step)
                adapters[k][w] -= (lr * mhat / (np.sqrt(vhat) + eps)).astype(adapters[k][w].dtype)
        if log_every and (step % log_every == 0 or step == 1):
            print(f"  lora step {step:4d}/{steps}  loss {loss:.4f}")
    return adapters, history


# --------------------------------------------------------------------------- #
# Merge / persistence
# --------------------------------------------------------------------------- #
def merge_adapters(P_base: dict, adapters: dict, scale: float) -> dict:
    """Return a full parameter dict with the adapters folded into the base weights, so
    the result can be served by the ordinary (unmodified) inference engine."""
    merged = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in P_base.items()}
    for k, ab in adapters.items():
        merged[k] = merged[k] + adapter_delta(ab, scale).astype(merged[k].dtype)
    return merged


def save_lora(path: Path, adapters: dict, lc: LoRAConfig, base_model: str) -> None:
    """Persist adapters + metadata to ``<path>`` (npz). Tiny — only the low-rank factors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for k, ab in adapters.items():
        arrays[f"{k}::A"] = ab["A"].astype(np.float32)
        arrays[f"{k}::B"] = ab["B"].astype(np.float32)
    meta = {"rank": lc.rank, "alpha": lc.alpha, "targets": list(lc.targets),
            "base_model": base_model}
    arrays["__meta__"] = np.asarray([json.dumps(meta)], dtype=np.str_)
    np.savez(path, **arrays)


def load_lora(path: Path) -> tuple[dict, LoRAConfig, str]:
    """Inverse of :func:`save_lora`; returns ``(adapters, LoRAConfig, base_model)``.

    Adapters are shipped between machines, so they are loaded without pickle: the
    metadata rides as a JSON string in a unicode array rather than a pickled object."""
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["__meta__"][0]))
        adapters: dict[str, dict[str, np.ndarray]] = {}
        for name in z.files:
            if name == "__meta__":
                continue
            key, which = name.rsplit("::", 1)
            adapters.setdefault(key, {})[which] = z[name].astype(np.float32)
    lc = LoRAConfig(rank=int(meta["rank"]), alpha=float(meta["alpha"]),
                    targets=tuple(meta["targets"]))
    return adapters, lc, meta.get("base_model", "")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _default_corpus() -> str:
    from .corpus import build_corpus
    return build_corpus()


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA fine-tune the MeshPool nano-GPT")
    ap.add_argument("--base", default="nano-gpt", help="registry model name to adapt")
    ap.add_argument("--text", default=None, help="path to a UTF-8 training corpus "
                    "(defaults to the built-in MeshPool corpus)")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--targets", default="attn", choices=["attn", "all"],
                    help="attn = Wqkv+Wo (default); all = + MLP")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--out", default=None, help="output .npz (adapters); default "
                    "weights/<base>.<rank>.lora.npz")
    ap.add_argument("--merge-out", default=None, help="also write a full merged "
                    "checkpoint here for direct serving")
    ap.add_argument("--check-only", action="store_true",
                    help="run the finite-difference gradient check and exit")
    args = ap.parse_args()

    print("== LoRA gradient check (low-rank projection correctness gate) ==")
    worst = gradient_check_lora()
    ok = worst < 1e-4
    print(f"  worst relative error: {worst:.2e}  ->  {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("LoRA gradient check failed; refusing to train")
    if args.check_only:
        return

    if args.base not in M.REGISTRY:
        raise SystemExit(f"unknown base model {args.base!r}; choices: {list(M.REGISTRY)}")
    cfg = M.REGISTRY[args.base]
    P_base, meta = M.load_params(cfg)
    itos = meta.get("itos") if meta else None
    text = Path(args.text).read_text(encoding="utf-8") if args.text else _default_corpus()
    targets = ALL_TARGETS if args.targets == "all" else DEFAULT_TARGETS
    lc = LoRAConfig(rank=args.rank, alpha=args.alpha, targets=targets)

    print(f"== LoRA fine-tune {cfg.name}: rank {lc.rank}, alpha {lc.alpha}, "
          f"targets {targets}, {args.steps} steps x batch {args.batch} ==")
    adapters, history = train_lora(P_base, cfg, text, lc, steps=args.steps,
                                   batch_size=args.batch, lr=args.lr, itos=itos,
                                   train_block=args.block, log_every=max(1, args.steps // 10))
    trainable = sum(v.size for ab in adapters.values() for v in ab.values())
    base_n = sum(v.size for v in P_base.values())
    print(f"  first loss {history[0]:.4f} -> last loss {history[-1]:.4f}  "
          f"({100.0 * trainable / base_n:.2f}% params trainable: {trainable} / {base_n})")

    out = Path(args.out) if args.out else M.WEIGHTS_DIR / f"{cfg.name}.r{lc.rank}.lora.npz"
    save_lora(out, adapters, lc, cfg.name)
    print(f"== saved adapters -> {out} ==")
    if args.merge_out:
        merged = merge_adapters(P_base, adapters, lc.scale)
        from .train import save_checkpoint
        save_checkpoint(Path(args.merge_out), merged, itos or M.CharTokenizer(
            vocab_size=cfg.vocab_size).itos)
        print(f"== saved merged checkpoint -> {args.merge_out} ==")


if __name__ == "__main__":
    main()
