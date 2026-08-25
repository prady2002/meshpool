"""Unit tests for sim.perfmodel — fast, pure Python + numpy, no network."""
from __future__ import annotations

import math
import pytest

from sim.perfmodel import (
    ClaimTier, Claim, ModelConfig, ModelParams, PipelineConfig,
    Observation, effective_tokens_per_second, prefill_time_s, fit, predict,
    MODEL_CONFIGS, CITED_PETALS_LLAMA2_70B, CITED_PETALS_FALCON_180B,
    CITED_EXO_KIMI_K2_LAN,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _simple_config(model: str = "llama-3.1-8b", n_stages: int = 2,
                   rtt_ms: float = 45.0, bw_kbps: float = 10_000.0) -> PipelineConfig:
    return PipelineConfig(
        model=model,
        n_stages=n_stages,
        cpu_scales=[1.0] * n_stages,
        rtt_ms_per_hop=[rtt_ms] * (n_stages - 1),
        bw_kbps_per_hop=[bw_kbps] * (n_stages - 1),
    )


def _simple_params(model: str = "llama-3.1-8b", base: float = 10.0) -> ModelParams:
    return ModelParams(base_tok_s={model: base}, residual_variance=0.04)


# ---------------------------------------------------------------------------
# Model configs are sane
# ---------------------------------------------------------------------------

def test_model_configs_exist():
    for name in ["llama-3.1-8b", "llama-2-70b", "llama-3.1-405b", "deepseek-v3", "kimi-k2"]:
        assert name in MODEL_CONFIGS, f"missing model config {name!r}"


def test_model_d_model_values():
    assert MODEL_CONFIGS["llama-2-70b"].d_model == 8192
    assert MODEL_CONFIGS["llama-3.1-405b"].d_model == 16384
    assert MODEL_CONFIGS["deepseek-v3"].d_model == 7168
    assert MODEL_CONFIGS["kimi-k2"].d_model == 7168


def test_activation_bytes():
    cfg = MODEL_CONFIGS["deepseek-v3"]
    # d_model=7168, dtype_bytes=2 → 14336 bytes per hop
    assert cfg.activation_bytes_per_hop == 7168 * 2


# ---------------------------------------------------------------------------
# Prefill is position-parallel (the #1 correctness check)
# ---------------------------------------------------------------------------

def test_prefill_position_parallel_not_per_token():
    """Prefill should send ONE [T × d_model] tensor per hop, NOT T separate tokens.

    1000 tokens at d_model=7168 (fp16) = 14,336,000 bytes ≈ 14 MB per hop.
    At 100 Mbps (bw_kbps=100_000) that is ~1.14 s per hop.
    If the model wrongly sent T tokens individually it would be 1000× longer.
    """
    config = PipelineConfig(
        model="deepseek-v3",
        n_stages=2,
        cpu_scales=[1.0, 1.0],
        rtt_ms_per_hop=[150.0],
        bw_kbps_per_hop=[100_000.0],   # 100 Mbps
        n_tokens_prefill=1000,
    )
    params = _simple_params("deepseek-v3", base=5.0)
    t = prefill_time_s(config, params)
    # 14 MB at 100 Mbps = 14e6 bytes / (100e6 / 8) bytes/s ≈ 1.12 s for network
    # Compute: 1000 tokens / (5 tok/s * 4 prefill_scale * 1.0 scale * 2 stages) ≈ 25 s
    # total ≈ 26 s
    # If per-token (wrong): 1000 × 1.12 s ≈ 1120 s — drastically different
    assert 1.0 < t < 200.0, (
        f"prefill_time_s should be ~26 s for this config, got {t:.2f} s. "
        "If >> 1000 s the implementation is sending tokens one-by-one (wrong)."
    )
    # Also verify the wrong (per-token) estimate would be far larger
    assert t < 500.0, "prefill should NOT be treating each token as a separate transfer"


def test_prefill_scales_with_tokens():
    """Prefill time should roughly scale with n_tokens."""
    config_1k = PipelineConfig(
        model="llama-2-70b", n_stages=2, cpu_scales=[1.0, 1.0],
        rtt_ms_per_hop=[45.0], bw_kbps_per_hop=[10_000.0],
        n_tokens_prefill=1000,
    )
    config_2k = PipelineConfig(
        model="llama-2-70b", n_stages=2, cpu_scales=[1.0, 1.0],
        rtt_ms_per_hop=[45.0], bw_kbps_per_hop=[10_000.0],
        n_tokens_prefill=2000,
    )
    params = _simple_params("llama-2-70b", base=3.0)
    t1 = prefill_time_s(config_1k, params)
    t2 = prefill_time_s(config_2k, params)
    assert t2 > t1 * 1.5, f"2× tokens should take more time: {t1:.2f} vs {t2:.2f}"


# ---------------------------------------------------------------------------
# Speculative decoding
# ---------------------------------------------------------------------------

def test_spd_expected_accepted_formula():
    """Test the geometric series formula: (1 - a^(K+1)) / (1 - a).

    SPD is only beneficial when network latency dominates compute (it reduces traversals).
    For a compute-bottlenecked single-stage setup, SPD may be slower overall because
    K+1 draft steps cost more compute. The formula is still correct either way.
    """
    import math
    # K=3, a=0.8 → expected_accepted = (1 - 0.8^4) / (1 - 0.8) = 2.952
    config = PipelineConfig(
        model="llama-3.1-8b",
        n_stages=1,
        cpu_scales=[1.0],
        rtt_ms_per_hop=[],
        bw_kbps_per_hop=[],
        draft_len=3,
        acceptance_rate=0.8,
    )
    params = _simple_params("llama-3.1-8b", base=20.0)
    tok_s = effective_tokens_per_second(config, params)

    # MeshPool's SelfDraft generates K draft tokens LOCALLY (shallow model, no pipeline
    # traversal).  Only the verification pass goes through the full pipeline.  So the
    # traversal time is compute_s + net_s — the same as one baseline decode step.
    # expected_accepted = (1 - 0.8^4) / (1 - 0.8) = 2.952
    # t_traversal = compute_s + net_s = 0.05 + 0 = 0.05 s  (no network, n_stages=1)
    # tok_s_spd = 2.952 / 0.05 = 59.04 tok/s
    expected_accepted = (1.0 - 0.8 ** 4) / (1.0 - 0.8)
    compute_s = 1.0 / (20.0 * 1.0)  # 1/(base * n_stages * scale)
    expected_tok_s = expected_accepted / compute_s   # draft is local, no extra traversal
    assert abs(tok_s - expected_tok_s) < 0.01, (
        f"SPD formula mismatch: got {tok_s:.4f}, expected {expected_tok_s:.4f}"
    )

    # In a network-dominated scenario (slow link, fast compute), SPD SHOULD be faster
    net_config = PipelineConfig(
        model="llama-3.1-8b", n_stages=2,
        cpu_scales=[100.0, 100.0],              # very fast compute
        rtt_ms_per_hop=[500.0],                 # very slow network (500 ms RTT)
        bw_kbps_per_hop=[100_000.0],
        draft_len=3, acceptance_rate=0.8,
    )
    net_config_no_spd = PipelineConfig(
        model="llama-3.1-8b", n_stages=2,
        cpu_scales=[100.0, 100.0],
        rtt_ms_per_hop=[500.0], bw_kbps_per_hop=[100_000.0],
    )
    net_tok_s = effective_tokens_per_second(net_config, params)
    net_tok_s_no_spd = effective_tokens_per_second(net_config_no_spd, params)
    assert net_tok_s > net_tok_s_no_spd, (
        f"SPD should help in network-bottlenecked scenario: "
        f"spd={net_tok_s:.3f} no_spd={net_tok_s_no_spd:.3f}"
    )


def test_spd_zero_draft_equals_no_spd():
    """draft_len=0 should give the same result as SPD disabled."""
    params = _simple_params()
    c1 = _simple_config()
    c2 = PipelineConfig(
        model="llama-3.1-8b", n_stages=2, cpu_scales=[1.0, 1.0],
        rtt_ms_per_hop=[45.0], bw_kbps_per_hop=[10_000.0],
        draft_len=0,
    )
    t1 = effective_tokens_per_second(c1, params)
    t2 = effective_tokens_per_second(c2, params)
    assert abs(t1 - t2) < 1e-9, f"zero draft should equal no SPD: {t1} vs {t2}"


# ---------------------------------------------------------------------------
# fit / predict round-trip
# ---------------------------------------------------------------------------

def test_fit_predict_round_trip():
    """Fit on observations → predict on training set should be close."""
    # Synthesise 4 observations at different scales and stage counts
    configs = [
        PipelineConfig("llama-3.1-8b", 2, [1.0, 1.0], [45.0], [10_000.0]),
        PipelineConfig("llama-3.1-8b", 2, [1.5, 1.5], [45.0], [10_000.0]),
        PipelineConfig("llama-3.1-8b", 3, [1.0, 1.0, 1.0], [45.0, 45.0], [10_000.0, 10_000.0]),
        PipelineConfig("llama-3.1-8b", 2, [2.0, 2.0], [150.0], [5_000.0]),
    ]
    # Ground truth: base_tok_s = 8.0
    true_params = ModelParams(base_tok_s={"llama-3.1-8b": 8.0}, residual_variance=0.01)
    observations = [
        Observation(config=c, tok_s=effective_tokens_per_second(c, true_params))
        for c in configs
    ]
    fitted = fit(observations)
    assert "llama-3.1-8b" in fitted.base_tok_s
    fitted_base = fitted.base_tok_s["llama-3.1-8b"]
    # Should recover the true base within 20%
    assert abs(fitted_base - 8.0) / 8.0 < 0.20, (
        f"fitted base_tok_s={fitted_base:.3f} far from true=8.0"
    )


def test_predict_always_returns_bound():
    """predict() must always return a Claim with non-None ci_low and ci_high."""
    params = _simple_params()
    config = _simple_config()
    claim = predict(config, params)
    assert claim.ci_low is not None, "ci_low must not be None"
    assert claim.ci_high is not None, "ci_high must not be None"
    assert claim.ci_low <= claim.value <= claim.ci_high, (
        f"value {claim.value} not in CI [{claim.ci_low}, {claim.ci_high}]"
    )
    assert claim.tier == ClaimTier.PROJECTED


def test_predict_tier_is_projected():
    params = _simple_params()
    claim = predict(_simple_config(), params)
    assert claim.tier == ClaimTier.PROJECTED


def test_predict_value_positive():
    params = _simple_params()
    claim = predict(_simple_config(), params)
    assert claim.value > 0


def test_predict_ci_covers_value():
    params = ModelParams(base_tok_s={"llama-3.1-8b": 10.0}, residual_variance=0.25)
    claim = predict(_simple_config(), params)
    assert claim.ci_low <= claim.value <= claim.ci_high


# ---------------------------------------------------------------------------
# R² and MAPE are populated by fit
# ---------------------------------------------------------------------------

def test_fit_populates_r2_mape():
    configs = [
        PipelineConfig("llama-2-70b", 2, [1.0, 1.0], [150.0], [5_000.0]),
        PipelineConfig("llama-2-70b", 2, [0.5, 0.5], [150.0], [5_000.0]),
    ]
    true_params = ModelParams(base_tok_s={"llama-2-70b": 4.0}, residual_variance=0.01)
    obs = [Observation(c, effective_tokens_per_second(c, true_params)) for c in configs]
    fitted = fit(obs)
    assert not math.isnan(fitted.r2), "r2 should be set"
    assert not math.isnan(fitted.mape), "mape should be set"
    assert 0.0 <= fitted.mape <= 1.0, f"MAPE should be fraction, got {fitted.mape}"


# ---------------------------------------------------------------------------
# Cited constants are correct type/tier
# ---------------------------------------------------------------------------

def test_cited_constants():
    for c in [CITED_PETALS_LLAMA2_70B, CITED_PETALS_FALCON_180B, CITED_EXO_KIMI_K2_LAN]:
        assert c.tier == ClaimTier.CITED
        assert c.citation is not None
        assert c.value > 0


# ---------------------------------------------------------------------------
# Unknown model raises
# ---------------------------------------------------------------------------

def test_unknown_model_raises():
    config = PipelineConfig(
        model="no-such-model", n_stages=1, cpu_scales=[1.0],
        rtt_ms_per_hop=[], bw_kbps_per_hop=[],
    )
    params = ModelParams(base_tok_s={})
    with pytest.raises((ValueError, KeyError)):
        effective_tokens_per_second(config, params)
