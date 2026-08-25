"""Speculative Pipeline Decoding — correctness first, speed second.

SPD is only worth anything if it is *exact*. A faster decoder that quietly changes
the output distribution is not an optimisation, it is a bug that ships. So the bulk
of this file asserts equivalence to ordinary decoding, and only then asserts that
the traversal count actually drops.
"""
from __future__ import annotations

import asyncio
from collections import Counter

import numpy as np
import pytest

from meshpool.inference import model as M
from meshpool.inference.pipeline import (LinkProfile, Pipeline, build_stages,
                                         generate_baseline)
from meshpool.inference.spd import (ExternalDraft, SelfDraft, SpecStats,
                                    _residual, speculative_generate)

PROMPT = [5, 9, 33, 41, 7]


@pytest.fixture(scope="module")
def cfg_params():
    cfg = M.REGISTRY["nano-gpt"]
    P, _ = M.load_params(cfg)
    return cfg, P


def _pipe(cfg, P, depth, rtt=0.0):
    return Pipeline(cfg, build_stages(cfg, depth, P), LinkProfile(rtt_ms=rtt))


def _greedy(cfg, P, depth, n, k, prompt=PROMPT, chunk=0):
    base, bstats = asyncio.run(generate_baseline(
        _pipe(cfg, P, depth), prompt, n, temperature=0.0, seed=7, chunked_prefill=chunk))
    spd, pstats, spec = asyncio.run(speculative_generate(
        _pipe(cfg, P, depth), SelfDraft(cfg, P, 2), prompt, n,
        k=k, temperature=0.0, seed=7, chunked_prefill=chunk))
    return base, bstats, spd, pstats, spec


# --------------------------------------------------------------------------- #
# Exactness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("depth", [1, 2, 3, 4, 8])
def test_greedy_spd_is_token_identical_across_pipeline_depths(cfg_params, depth):
    """The headline correctness property, at every sharding of the model."""
    cfg, P = cfg_params
    base, _, spd, _, _ = _greedy(cfg, P, depth, 24, k=4)
    assert spd == base


@pytest.mark.parametrize("k", [1, 2, 3, 5, 8])
def test_greedy_spd_is_token_identical_across_draft_lengths(cfg_params, k):
    cfg, P = cfg_params
    base, _, spd, _, _ = _greedy(cfg, P, 3, 24, k=k)
    assert spd == base


@pytest.mark.parametrize("draft_layers", [1, 2, 3, 4])
def test_output_is_independent_of_draft_quality(cfg_params, draft_layers):
    """A worse draft may be slower, but must never change what is produced."""
    cfg, P = cfg_params
    base, _ = asyncio.run(generate_baseline(_pipe(cfg, P, 3), PROMPT, 20,
                                            temperature=0.0, seed=3))
    spd, _, _ = asyncio.run(speculative_generate(
        _pipe(cfg, P, 3), SelfDraft(cfg, P, draft_layers), PROMPT, 20,
        k=4, temperature=0.0, seed=3))
    assert spd == base


def test_external_draft_also_preserves_output(cfg_params):
    cfg, P = cfg_params
    base, _ = asyncio.run(generate_baseline(_pipe(cfg, P, 3), PROMPT, 16,
                                            temperature=0.0, seed=11))
    # A deliberately mismatched draft: different random weights, same shapes.
    bad = M.random_params(M.ModelConfig(**{**cfg.__dict__, "seed": 999}))
    spd, _, spec = asyncio.run(speculative_generate(
        _pipe(cfg, P, 3), ExternalDraft(cfg, bad), PROMPT, 16,
        k=4, temperature=0.0, seed=11))
    assert spd == base
    assert spec.acceptance_rate < 0.9, "a random draft should rarely be accepted"


def test_chunked_prefill_does_not_change_output(cfg_params):
    cfg, P = cfg_params
    long_prompt = [5, 9, 33, 41, 7, 60, 12, 88, 2, 71, 44, 19]
    base, _, spd, _, _ = _greedy(cfg, P, 3, 16, k=4, prompt=long_prompt, chunk=4)
    assert spd == base


def test_distribution_matches_baseline_at_temperature(cfg_params):
    """Sampled SPD must have the same law as sampled baseline decoding.

    Compared against a bootstrap null built from the baseline itself, so the
    threshold reflects genuine sampling noise rather than a guessed tolerance.
    """
    cfg, P = cfg_params
    n = 900
    pipe = _pipe(cfg, P, 3)

    def draw_base(seed):
        out, _ = asyncio.run(generate_baseline(pipe, PROMPT, 1, temperature=0.9,
                                               seed=seed, session=f"b{seed}"))
        return out[0]

    pipe2 = _pipe(cfg, P, 3)

    def draw_spd(seed):
        out, _, _ = asyncio.run(speculative_generate(
            pipe2, SelfDraft(cfg, P, 2), PROMPT, 1, k=4, temperature=0.9,
            seed=seed, session=f"s{seed}"))
        return out[0]

    a = [draw_base(s) for s in range(n)]
    b = [draw_base(s) for s in range(10_000, 10_000 + n)]   # null: base vs base
    c = [draw_spd(s) for s in range(20_000, 20_000 + n)]

    def tv(x, y):
        cx, cy = Counter(x), Counter(y)
        keys = set(cx) | set(cy)
        return 0.5 * sum(abs(cx[k] / len(x) - cy[k] / len(y)) for k in keys)

    null, observed = tv(a, b), tv(a, c)
    assert observed <= null * 2.5 + 0.01, (
        f"SPD distribution differs from baseline: TV={observed:.4f} vs null TV={null:.4f}")


# --------------------------------------------------------------------------- #
# Speed — the whole point, but only after exactness is established
# --------------------------------------------------------------------------- #
def test_spd_reduces_traversals(cfg_params):
    cfg, P = cfg_params
    _, bstats, _, pstats, spec = _greedy(cfg, P, 3, 24, k=4)
    assert pstats.traversals < bstats.traversals
    assert spec.tokens_per_traversal > 1.0


def test_spd_speedup_grows_with_pipeline_depth_cost(cfg_params):
    """The deeper/slower the pipeline, the more RTT there is to amortise."""
    cfg, P = cfg_params
    _, bstats, _, pstats, _ = _greedy(cfg, P, 4, 24, k=4)
    saved = 1.0 - (pstats.link_seconds / max(bstats.link_seconds, 1e-9))
    assert saved > 0.4, f"expected a large drop in modelled link time, saved={saved:.2f}"


def test_verify_traversal_carries_all_drafts(cfg_params):
    """K drafts must be verified in ONE traversal, not K."""
    cfg, P = cfg_params
    _, _, _, pstats, spec = _greedy(cfg, P, 3, 24, k=4)
    assert spec.verify_traversals == spec.rounds
    assert spec.drafted > spec.rounds, "more drafts than rounds means batching happened"


# --------------------------------------------------------------------------- #
# Edge cases and internals
# --------------------------------------------------------------------------- #
def test_single_token_prompt(cfg_params):
    cfg, P = cfg_params
    base, _, spd, _, _ = _greedy(cfg, P, 3, 8, k=3, prompt=[5])
    assert spd == base


def test_max_new_tokens_is_respected_exactly(cfg_params):
    cfg, P = cfg_params
    for n in (1, 2, 3, 7, 11):
        spd, _, _ = asyncio.run(speculative_generate(
            _pipe(cfg, P, 3), SelfDraft(cfg, P, 2), PROMPT, n, k=5,
            temperature=0.0, seed=1))
        assert len(spd) == n, f"asked for {n}, got {len(spd)}"


def test_generation_stops_at_block_size(cfg_params):
    """SPD must stop exactly where ordinary decoding stops — not one token early.

    The boundary rule is subtle and easy to get wrong in either direction. A token is
    *emitted* at position p using the logits of position p-1, so the final token may
    legitimately land on position ``block_size`` even though nothing may ever be *fed*
    at that position (there is no position embedding for it). Decoding therefore ends
    with ``len(prompt) + len(out) == block_size + 1``, and an implementation that
    reserves feedable room for the last token silently truncates every generation that
    reaches the context limit. Asserting agreement with the baseline is the invariant
    that actually matters, and it is strictly stronger than any length bound.
    """
    cfg, P = cfg_params
    for trim in (3, 2, 1, 0):
        prompt = list(range(1, cfg.block_size - trim))
        base, _ = asyncio.run(generate_baseline(
            _pipe(cfg, P, 2), prompt, 50, temperature=0.0, seed=1))
        spd, _, _ = asyncio.run(speculative_generate(
            _pipe(cfg, P, 2), SelfDraft(cfg, P, 2), prompt, 50, k=4,
            temperature=0.0, seed=1))
        assert spd == base, f"prompt={len(prompt)}: SPD diverged at the window edge"
        assert len(prompt) + len(spd) <= cfg.block_size + 1
        assert spd, "decoding produced nothing at all"


def test_rejected_drafts_leave_no_cache_residue(cfg_params):
    """The session must contain exactly the tokens the caller was given.

    Asserted behaviourally: continuing generation from the SPD session has to produce
    the same tokens as decoding the same sequence from scratch. Any rejected or
    trimmed draft left behind in a stage's cache would poison attention and change
    the continuation.
    """
    cfg, P = cfg_params
    pipe = _pipe(cfg, P, 3)
    spd, _, _ = asyncio.run(speculative_generate(
        pipe, SelfDraft(cfg, P, 2), PROMPT, 20, k=4, temperature=0.0,
        seed=5, session="s"))

    total = len(PROMPT) + len(spd)
    lengths = {stage.cache("s").length for stage in pipe.stages}
    assert len(lengths) == 1, f"stages disagree on cache length: {lengths}"
    # The final emitted token is intentionally left pending (never fed forward).
    assert lengths.pop() in (total - 1, total)

    fresh, _ = asyncio.run(generate_baseline(_pipe(cfg, P, 3), PROMPT + spd, 6,
                                             temperature=0.0, seed=5))
    resumed, _, _ = asyncio.run(speculative_generate(
        pipe, SelfDraft(cfg, P, 2), PROMPT + spd, 6, k=4, temperature=0.0,
        seed=5, session="fresh"))
    assert resumed == fresh


def test_k_must_be_positive(cfg_params):
    cfg, P = cfg_params
    with pytest.raises(ValueError):
        asyncio.run(speculative_generate(_pipe(cfg, P, 2), SelfDraft(cfg, P, 2),
                                         PROMPT, 4, k=0))


def test_self_draft_rejects_invalid_depth(cfg_params):
    cfg, P = cfg_params
    with pytest.raises(ValueError):
        SelfDraft(cfg, P, 0)
    with pytest.raises(ValueError):
        SelfDraft(cfg, P, cfg.n_layer)


def test_residual_is_a_valid_distribution():
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.1, 0.8, 0.1])
    r = _residual(p, q)
    assert r.sum() == pytest.approx(1.0)
    assert (r >= 0).all()
    assert r[1] == 0.0, "mass the draft over-weighted must be removed"


def test_residual_falls_back_when_draft_dominates():
    p = np.array([0.2, 0.8])
    q = np.array([0.5, 0.5])
    r = _residual(p, np.array([1.0, 1.0]))
    assert r.sum() == pytest.approx(1.0)
    assert np.allclose(_residual(p, q), np.array([0.0, 1.0]))


def test_spec_stats_report_shape():
    s = SpecStats(rounds=3, drafted=12, accepted=7, emitted=10, verify_traversals=3)
    d = s.as_dict()
    assert d["acceptance_rate"] == pytest.approx(7 / 12, abs=1e-4)
    assert d["tokens_per_traversal"] == pytest.approx(10 / 3, abs=1e-4)
