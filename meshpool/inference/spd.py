"""Speculative Pipeline Decoding (SPD) — amortising wide-area RTT across many tokens.

The problem
-----------
Decoding a model that is split across machines costs

    t_token  ~=  sum(compute_i)  +  (N - 1) * RTT

Over the public internet the second term dominates and nothing about the model makes
it smaller: two stages 50 ms apart cap you near 10 tok/s *before any compute at all*.
Measured reality agrees — Petals reports ~6 tok/s for Llama-2-70B and ~4 tok/s for
Falcon-180B across its public swarm (NeurIPS 2023 / arXiv 2312.08361). This is the
wall that every decentralised inference project has hit.

The idea
--------
Speculative decoding is well known *inside one machine*, where it trades spare
arithmetic for fewer memory-bandwidth-bound passes (Leviathan et al., "Fast Inference
from Transformers via Speculative Decoding", 2023; Chen et al., "Accelerating Large
Language Model Decoding with Speculative Sampling", 2023). Pipeline parallelism is
well known *across machines*.

Combining them is unusually potent, because the thing speculation removes — passes
over the full model — is exactly the thing that costs ``(N - 1) * RTT`` on a mesh.
A cheap draft proposes ``K`` tokens; the full pipeline verifies **all K in a single
traversal**, since every stage simply transforms a ``[K, d_model]`` tensor instead of
a ``[1, d_model]`` one. The compute grows slightly. **The number of network round
trips does not change at all.**

    baseline :  t = (T)          * [compute + (N-1)*RTT]
    SPD      :  t = (T / tokens_per_traversal) * [compute' + (N-1)*RTT]

Two properties make this work rather than just sound good:

1. **It is exact, not approximate.** Rejection sampling makes the output distribution
   provably identical to ordinary sampling from the target model. At temperature 0 it
   is identical token for token. This is a correctness-preserving optimisation, and
   :mod:`tests.test_spd` plus ``sim/scenarios/verify_spd_equivalence.py`` assert it
   rather than assuming it.

2. **The draft is free on a mesh.** :class:`SelfDraft` reuses the *first stage's own
   layers* — which already live on one machine — plus the output head. Drafting
   therefore costs zero extra wide-area hops. A separately trained draft model
   (:class:`ExternalDraft`) is also supported for higher acceptance rates.

Rollback
--------
The verify traversal writes ``K`` speculative positions into every stage's KV cache.
Whatever the sampler rejects must vanish, so each round ends by truncating every
stage's cache back to the accepted length. Without that, a rejected token silently
poisons all subsequent attention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

from . import kvcache as KV
from . import model as M
from .pipeline import (GREEDY_EPS, CachedShard, Pipeline, PipelineStats,
                       dist_from_logits, prefill, sample_from_dist)


class DraftModel(Protocol):
    """Proposes continuations cheaply, and reports the distribution it used.

    The distribution matters: rejection sampling needs ``q`` as well as the sampled
    token, otherwise the acceptance test cannot be unbiased.
    """

    def reset(self) -> None: ...

    def propose(self, ctx: list[int], pending: list[int], k: int,
                temperature: float, top_k: int,
                rng: np.random.Generator) -> tuple[list[int], list[np.ndarray]]: ...


class _CachedDraft:
    """Shared machinery: an autoregressive drafter over a params slice with a KV cache."""

    def __init__(self, cfg: M.ModelConfig, params: dict, n_layer: int) -> None:
        self.cfg = cfg
        self.n_layer = n_layer
        self.P = params
        self._cache = KV.KVCache(0, n_layer)
        self._cached = 0

    def reset(self) -> None:
        self._cache.reset()
        self._cached = 0

    def _forward(self, tokens: list[int], pos_offset: int) -> np.ndarray:
        x = KV.embed_tokens_at(self.P, self.cfg, tokens, pos_offset)
        x = KV.run_layers_cached(self.P, self.cfg, x, 0, self.n_layer, self._cache)
        return M.head(self.P, self.cfg, x)

    def propose(self, ctx: list[int], pending: list[int], k: int,
                temperature: float, top_k: int,
                rng: np.random.Generator) -> tuple[list[int], list[np.ndarray]]:
        """Draft ``k`` tokens continuing ``ctx`` (whose tail ``pending`` is uncached).

        The draft keeps its own cache in step with the accepted sequence, so drafting
        costs one small local forward per proposed token and nothing on the wire.
        """
        # Re-sync if the accepted sequence diverged from what we last cached
        # (it always does after a rejection — the correction token replaces a draft).
        keep = len(ctx) - len(pending)
        if self._cached > keep:
            self._cache.truncate(keep)
            self._cached = keep
        if self._cached < keep:
            self.reset()
            pending = list(ctx)
            keep = 0

        drafts: list[int] = []
        dists: list[np.ndarray] = []
        feed = list(pending)
        pos = self._cached

        for _ in range(k):
            if pos + len(feed) >= self.cfg.block_size:
                break
            logits = self._forward(feed, pos)
            pos += len(feed)
            self._cached = pos
            q = dist_from_logits(logits[-1], temperature, top_k)
            nxt = sample_from_dist(q, rng)
            drafts.append(nxt)
            dists.append(q)
            feed = [nxt]

        # The last drafted token was never fed through the draft model, so the draft
        # cache legitimately trails the draft sequence by one. Leaving it is correct:
        # the next round's `pending` will carry it.
        return drafts, dists


class SelfDraft(_CachedDraft):
    """Draft using the model's own first ``n_layer`` decoder layers plus the head.

    Because those layers are precisely what the *first pipeline stage* already holds,
    the draft executes entirely on one machine and adds **no wide-area hops**. This is
    the early-exit style of self-speculation; acceptance is lower than a purpose-built
    draft model but the marginal cost on a mesh is close to zero.
    """

    def __init__(self, cfg: M.ModelConfig, full_params: dict, n_layer: int = 2) -> None:
        if not 1 <= n_layer < cfg.n_layer:
            raise ValueError(f"draft depth must be in [1, {cfg.n_layer}), got {n_layer}")
        params = M.slice_params(full_params, 0, n_layer, is_first=True, is_last=True)
        super().__init__(cfg, params, n_layer)


class ExternalDraft(_CachedDraft):
    """Draft with a separate, smaller model that shares the target's tokenizer."""

    def __init__(self, cfg: M.ModelConfig, params: dict) -> None:
        super().__init__(cfg, params, cfg.n_layer)


@dataclass
class SpecStats:
    """Acceptance telemetry. ``tokens_per_traversal`` is the number that matters."""

    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    emitted: int = 0
    verify_traversals: int = 0
    accept_run_hist: dict[int, int] = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        return (self.accepted / self.drafted) if self.drafted else 0.0

    @property
    def tokens_per_traversal(self) -> float:
        return (self.emitted / self.verify_traversals) if self.verify_traversals else 0.0

    def as_dict(self) -> dict:
        return {
            "rounds": self.rounds,
            "drafted": self.drafted,
            "accepted": self.accepted,
            "emitted": self.emitted,
            "verify_traversals": self.verify_traversals,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "tokens_per_traversal": round(self.tokens_per_traversal, 4),
            "accept_run_hist": dict(sorted(self.accept_run_hist.items())),
        }


def _residual(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """``norm(max(0, p - q))`` — the corrected distribution to sample after a reject.

    This is what makes speculation unbiased: conditioned on rejecting the draft, the
    replacement is drawn so that the *overall* law of the emitted token is exactly
    ``p``. Falls back to ``p`` in the degenerate case where the residual is empty,
    which can only happen when ``q`` dominates ``p`` everywhere.
    """
    r = np.maximum(p - q, 0.0)
    total = r.sum()
    if total <= 1e-12 or not np.isfinite(total):
        return p
    return r / total


async def speculative_generate(
    pipe: Pipeline,
    draft: DraftModel,
    prompt: list[int],
    max_new_tokens: int,
    k: int = 4,
    temperature: float = 0.8,
    top_k: int = 40,
    seed: int = 0,
    session: str = "spd",
    chunked_prefill: int = 0,
) -> tuple[list[int], PipelineStats, SpecStats]:
    """Generate with SPD. Returns (tokens, pipeline stats, speculation stats).

    Distributionally identical to :func:`pipeline.generate_baseline` with the same
    ``temperature``/``top_k``, and *token-identical* at temperature 0.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    cfg = pipe.cfg
    rng = np.random.default_rng(seed)

    pipe.drop_session(session)
    pipe.reset_stats()
    draft.reset()

    spec = SpecStats()
    out: list[int] = []

    # Prefill everything but the final prompt token. That token stays "pending" and
    # rides into the first verify traversal alongside the drafts, so no traversal is
    # ever spent on a single token.
    ctx = list(prompt)
    if len(prompt) > 1:
        await prefill(pipe, prompt[:-1], session, chunk=chunked_prefill)
        cached = len(prompt) - 1
    else:
        cached = 0
    pending = ctx[cached:]

    while len(out) < max_new_tokens:
        # A token may be *emitted* at absolute position `block_size` (its distribution
        # comes from the logits of position block_size-1), but a token may only be
        # *fed* at positions up to block_size-1, since that requires a position
        # embedding. So the number of drafts we may append is bounded by how much
        # feedable room is left — and when that room hits exactly zero we still run
        # one final draft-free traversal to emit the last token. Baseline decoding
        # stops on the same rule; any mismatch here would make SPD diverge at the
        # window edge, which is precisely what this bound is written to prevent.
        max_draft = cfg.block_size - (cached + len(pending))
        if max_draft < 0 or not pending:
            break
        k_eff = min(k, max_draft, max_new_tokens - len(out))

        drafts, q_dists = draft.propose(ctx, pending, k_eff, temperature, top_k, rng)
        n_draft = len(drafts)

        # ---- ONE traversal verifies the pending token AND every draft ---------- #
        feed = pending + drafts
        logits = await pipe.forward(feed, pos_offset=cached, session=session)
        spec.rounds += 1
        spec.verify_traversals += 1
        spec.drafted += n_draft

        m = len(pending)
        base = len(ctx)                      # absolute position of drafts[0]
        cached += len(feed)

        # logits[m-1+i] is the target distribution for absolute position base+i.
        tgt = [dist_from_logits(logits[m - 1 + i], temperature, top_k)
               for i in range(n_draft + 1)]

        accepted = 0
        emitted: Optional[int] = None
        for i in range(n_draft):
            x = drafts[i]
            p_i, q_i = tgt[i], q_dists[i]
            px, qx = float(p_i[x]), float(q_i[x])
            if temperature <= GREEDY_EPS:
                ok = px > 0.0                # one-hot target: accept iff it agrees
            else:
                ok = qx <= 0.0 or rng.random() <= min(1.0, px / qx)
            if ok:
                accepted += 1
                continue
            emitted = sample_from_dist(_residual(p_i, q_i), rng)
            break

        if emitted is None:
            # Every draft survived, so the verify pass also handed us a free extra
            # token from the position just past the last draft: K+1 for one traversal.
            emitted = sample_from_dist(tgt[n_draft], rng)

        spec.accepted += accepted
        spec.accept_run_hist[accepted] = spec.accept_run_hist.get(accepted, 0) + 1

        # A round can emit up to K+1 tokens, which may overshoot the caller's budget.
        # Trim *before* updating cache bookkeeping, so the session state always
        # describes exactly the tokens the caller was given — otherwise a continued
        # session would attend to tokens that were generated but never returned.
        new_tokens = (drafts[:accepted] + [emitted])[:max_new_tokens - len(out)]
        out.extend(new_tokens)
        spec.emitted += len(new_tokens)

        # Rejected (and trimmed) drafts must leave no trace in any stage's KV cache.
        # `emitted` was never fed forward, so it is never among the cached positions.
        retained = min(accepted, len(new_tokens))
        valid = base + retained
        pipe.truncate_session(session, valid)
        cached = valid

        ctx = ctx[:base] + new_tokens
        pending = ctx[cached:]
        if not pending:
            break

    return out, pipe.stats, spec
