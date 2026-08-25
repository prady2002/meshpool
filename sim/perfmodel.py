"""Calibrated performance model for distributed pipeline inference (Phase 0.4).

Answers the question: **"If we distribute model X across nodes with these link
profiles and these compute scales, how fast will it run?"**

All predictions include a confidence interval.  All claims are tagged with a
provenance tier:
  - MEASURED:   directly observed on real hardware in this repository's test suite.
  - PROJECTED:  derived from the fitted model + measured compute scale.
  - CITED:      real-world numbers from published papers / vendor announcements
                (citation included in source comments).

Key modelling decisions
-----------------------
* **Prefill is position-parallel**: a prompt of T tokens sends ONE ``[T × d_model]``
  tensor per hop, NOT T separate tensors.  A 1000-token prompt at d_model=7168
  (fp16) is 1000×7168×2 ≈ 14 MB per hop.  At 100 Mbps download that is ~1.1 s per
  hop — large but not minutes.  Getting this wrong by treating each token as a
  separate transfer inflates the estimate by 1000×; that is the #1 modelling error
  in the literature.

* **Speculative pipeline decoding**: with draft length K and acceptance rate ``a``,
  expected accepted tokens per traversal = ``(1 - a^(K+1)) / (1 - a)``
  (geometric series sum — standard derivation, see e.g. Leviathan et al. 2023).
  Network cost per traversal is unchanged (still one pass per stage).

* **Compute time**: ``t_token_s = sum_i(1 / (flops_scale_i × base_tok_s))``.
  The ``base_tok_s`` is the single-node throughput of the full model on a reference
  machine (1.0 cpu_scale).  ``fit()`` calibrates ``base_tok_s`` per model config.

Cited real-world constants (do NOT remove citations)
-----------------------------------------------------
  Petals 6 tok/s Llama-2-70B:  Borzunov et al., NeurIPS 2023 / arXiv:2209.01188
  Petals 4 tok/s Falcon-180B:  Borzunov et al., arXiv:2312.08361
  exo ~30 tok/s Kimi K2 on 4× M3 Ultra (Thunderbolt-5, <1 ms LAN):
      exo GitHub repo README, 2025-06  (NOT a WAN number).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Claim tiers
# ---------------------------------------------------------------------------

class ClaimTier(Enum):
    MEASURED = "MEASURED"
    PROJECTED = "PROJECTED"
    CITED = "CITED"


@dataclass
class Claim:
    """A numeric value with provenance metadata.

    Args:
        value:      The central estimate.
        tier:       :class:`ClaimTier` provenance.
        unit:       Human-readable unit string, e.g. ``"tok/s"``.
        ci_low:     Lower confidence-interval bound (same units as value).
        ci_high:    Upper confidence-interval bound.
        citation:   For CITED claims, the source reference.
        note:       Free-form annotation.
    """

    value: float
    tier: ClaimTier
    unit: str = ""
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    citation: Optional[str] = None
    note: Optional[str] = None

    def __repr__(self) -> str:
        ci_str = ""
        if self.ci_low is not None and self.ci_high is not None:
            ci_str = f" [{self.ci_low:.2g}, {self.ci_high:.2g}]"
        tier = self.tier.value
        return f"Claim({self.value:.3g}{ci_str} {self.unit} | {tier})"


# ---------------------------------------------------------------------------
# Cited reference points (real-world ground truth)
# ---------------------------------------------------------------------------

CITED_PETALS_LLAMA2_70B = Claim(
    value=6.0, tier=ClaimTier.CITED, unit="tok/s",
    citation="Borzunov et al., NeurIPS 2023 / arXiv:2209.01188",
    note="Llama-2-70B distributed over public internet via Petals swarm",
)
CITED_PETALS_FALCON_180B = Claim(
    value=4.0, tier=ClaimTier.CITED, unit="tok/s",
    citation="Borzunov et al., arXiv:2312.08361",
    note="Falcon-180B distributed over public internet via Petals swarm",
)
CITED_EXO_KIMI_K2_LAN = Claim(
    value=30.0, tier=ClaimTier.CITED, unit="tok/s",
    citation="exo GitHub README, 2025-06",
    note="Kimi K2 on 4x M3 Ultra over Thunderbolt-5 (<1 ms LAN) — NOT a WAN number",
)


# ---------------------------------------------------------------------------
# Model configs (reference architectures)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Architecture parameters needed by the performance model."""

    name: str
    total_params_b: float      # billion parameters (total, including experts if MoE)
    active_params_b: float     # billion params activated per token (= total for dense)
    d_model: int               # hidden dimension
    n_layers: int              # transformer layers
    dtype_bytes: int = 2       # bytes per element (2 = fp16/bf16, 4 = fp32)

    @property
    def activation_bytes_per_hop(self) -> int:
        """Bytes transmitted per pipeline hop: one [d_model]-sized fp16 vector."""
        return self.d_model * self.dtype_bytes

    def prefill_bytes_per_hop(self, n_tokens: int) -> int:
        """Bytes for a full prefill pass: [n_tokens × d_model] tensor (fp16)."""
        return n_tokens * self.d_model * self.dtype_bytes


# Reference model configs
MODEL_CONFIGS: dict[str, ModelConfig] = {
    # nano-gpt: real trained model used for calibration measurements.
    # 8 layers, d_model=128, dtype float32 (4 bytes) in practice but we declare fp16
    # here; activation bytes per hop only matter when bandwidth is modelled (bw>0).
    "nano-gpt": ModelConfig(
        name="nano-gpt",
        total_params_b=0.001,
        active_params_b=0.001,
        d_model=128,
        n_layers=8,
        dtype_bytes=4,   # the numpy pipeline uses float32
    ),
    "llama-3.1-8b": ModelConfig(
        name="llama-3.1-8b",
        total_params_b=8.0,
        active_params_b=8.0,
        d_model=4096,
        n_layers=32,
    ),
    "llama-2-70b": ModelConfig(
        name="llama-2-70b",
        total_params_b=70.0,
        active_params_b=70.0,
        d_model=8192,
        n_layers=80,
    ),
    "llama-3.1-405b": ModelConfig(
        name="llama-3.1-405b",
        total_params_b=405.0,
        active_params_b=405.0,
        d_model=16384,
        n_layers=126,
    ),
    "deepseek-v3": ModelConfig(
        # MoE: 671B total / 37B active per token; d_model=7168
        # Source: DeepSeek-V3 technical report, 2024
        name="deepseek-v3",
        total_params_b=671.0,
        active_params_b=37.0,
        d_model=7168,
        n_layers=61,
    ),
    "kimi-k2": ModelConfig(
        # MoE: ~1T total / 32B active; d_model=7168; ~550–600 GB at Q4_K_M
        # Source: Moonshot AI Kimi K2 announcement, 2025
        name="kimi-k2",
        total_params_b=1000.0,
        active_params_b=32.0,
        d_model=7168,
        n_layers=61,
    ),
}


# ---------------------------------------------------------------------------
# Model parameters and fitting
# ---------------------------------------------------------------------------

@dataclass
class ModelParams:
    """Fitted parameters for the pipeline performance model."""

    # base_tok_s[model_name] = single-node throughput on a 1.0-scale reference host
    base_tok_s: dict[str, float] = field(default_factory=dict)
    # Residual (unexplained) variance (for CI width)
    residual_variance: float = 0.25
    r2: float = float("nan")
    mape: float = float("nan")      # mean absolute percentage error (0–1 scale)

    def __repr__(self) -> str:
        models = ", ".join(f"{k}={v:.2f}" for k, v in self.base_tok_s.items())
        return f"ModelParams(base_tok_s={{{models}}}, R2={self.r2:.3f}, MAPE={self.mape:.1%})"


@dataclass
class PipelineConfig:
    """A specific distributed deployment to evaluate.

    Args:
        model:          Model name key into :data:`MODEL_CONFIGS`.
        n_stages:       Number of pipeline stages (= number of nodes).
        cpu_scales:     Compute-scale multiplier for each stage node.
        rtt_ms_per_hop: List of RTTs between consecutive stages (len = n_stages - 1).
        bw_kbps_per_hop: Bandwidth for each hop (upload direction, kbps).
        n_tokens_prefill: Number of tokens in the prompt.
        draft_len:      Speculative decoding draft length (0 = disabled).
        acceptance_rate: Speculative decoding acceptance rate (0–1).
    """

    model: str
    n_stages: int
    cpu_scales: list[float]
    rtt_ms_per_hop: list[float]
    bw_kbps_per_hop: list[float]
    n_tokens_prefill: int = 0
    draft_len: int = 0
    acceptance_rate: float = 0.8


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _compute_time_s(config: PipelineConfig, params: ModelParams) -> float:
    """Total compute time across all stages for one decode token.

    Critical model assumption: when n_stages stages split the model evenly, each
    stage processes 1/n_stages of the layers.  On a homogeneous setup (same machine)
    total compute = sum_i(1/n_stages × 1/(base × scale_i)) which is INDEPENDENT of
    depth — splitting 8 layers across 4 stages still costs the same total compute as
    running all 8 on one stage.  This is the correct sequential-pipeline formula.

    For heterogeneous machines (different cpu_scale per stage) the bottleneck shifts
    to the slowest stage, and total compute = sum of stage times is still correct.
    """
    base = params.base_tok_s.get(config.model)
    if base is None or base <= 0:
        raise ValueError(f"no base_tok_s fitted for model {config.model!r}")
    # Each stage does 1/n_stages of the work; scale_i is the speed multiplier.
    stage_time_s = [1.0 / (base * config.n_stages * s) for s in config.cpu_scales]
    return sum(stage_time_s)


def _network_time_s(config: PipelineConfig, nbytes_per_hop: int) -> float:
    """One-way network time across all hops for an activation of ``nbytes_per_hop``."""
    total = 0.0
    for i, (rtt_ms, bw_kbps) in enumerate(
        zip(config.rtt_ms_per_hop, config.bw_kbps_per_hop)
    ):
        prop_s = (rtt_ms / 2.0) / 1000.0
        bw_bps = bw_kbps * 1000.0 / 8.0
        serial_s = nbytes_per_hop / bw_bps if bw_bps > 0 else 0.0
        total += prop_s + serial_s
    return total


def effective_tokens_per_second(
    config: PipelineConfig,
    params: ModelParams,
) -> float:
    """Predict decode throughput in tok/s.

    Pipeline inference cost model:
        t_token = Σ compute_i + Σ (RTT/2 + serialise) per hop

    Speculative pipeline decoding (if ``draft_len > 0``):
        expected_accepted = (1 - a^(K+1)) / (1 - a)   [geometric series]
        Network cost per traversal: one full pipeline pass (unchanged vs baseline).
        Compute per traversal: one full verification pass.  Draft tokens are produced
        locally by a shallow draft model (SelfDraft) that does NOT traverse the pipeline,
        so K draft steps add only negligible local compute vs the full-pipeline RTT.
        This gives speedup ≈ expected_accepted when RTT >> compute.
    """
    cfg = MODEL_CONFIGS.get(config.model)
    if cfg is None:
        raise ValueError(f"unknown model {config.model!r}")

    compute_s = _compute_time_s(config, params)
    net_s = _network_time_s(config, cfg.activation_bytes_per_hop)

    t_per_token_s = compute_s + net_s   # one decode step

    if config.draft_len > 0 and config.acceptance_rate < 1.0:
        K = config.draft_len
        a = config.acceptance_rate
        # Expected accepted tokens per traversal (Leviathan et al. 2023 eq. 1)
        expected_accepted = (1.0 - a ** (K + 1)) / (1.0 - a)
        # One pipeline traversal (verification pass) per K-draft cycle.
        # Draft compute is local and negligible vs. RTT in WAN-dominated regimes.
        t_per_traversal_s = compute_s + net_s
        return expected_accepted / t_per_traversal_s
    else:
        return 1.0 / t_per_token_s if t_per_token_s > 0 else 0.0


def prefill_time_s(config: PipelineConfig, params: ModelParams) -> float:
    """Estimate total time to prefill an n-token prompt.

    Prefill is position-parallel: one [T × d_model] tensor is sent per hop,
    NOT T separate tokens.  Compute across all stages dominates for long prompts.
    """
    cfg = MODEL_CONFIGS.get(config.model)
    if cfg is None:
        raise ValueError(f"unknown model {config.model!r}")
    if config.n_tokens_prefill == 0:
        return 0.0

    # Compute: prefill is ~linear in n_tokens (batch matmul across positions).
    # Each stage does 1/n_stages of the layers; same depth-independence as decode.
    # prefill_scale ≈ 4x faster per token than decode (assumption; no cited source).
    base = params.base_tok_s.get(config.model, 1.0)
    prefill_scale = 4.0
    stage_times = [config.n_tokens_prefill / (base * config.n_stages * s * prefill_scale)
                   for s in config.cpu_scales]
    compute_s = sum(stage_times)

    # Network: one [T × d_model] tensor per hop
    bytes_per_hop = cfg.prefill_bytes_per_hop(config.n_tokens_prefill)
    net_s = _network_time_s(config, bytes_per_hop)

    return compute_s + net_s


# ---------------------------------------------------------------------------
# Observation dataclass and fit/predict
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """A measured (config, throughput) data point for model fitting."""
    config: PipelineConfig
    tok_s: float          # measured tokens per second


def fit(observations: Sequence[Observation]) -> ModelParams:
    """Least-squares fit of ``base_tok_s`` per model from observations.

    Uses numpy lstsq.  Each observation yields the equation:
        1/tok_s = (1/base_tok_s) × Σ(1/scale_i) + net_time
    which is linear in 1/base_tok_s.  We solve for 1/base_tok_s and invert.

    Args:
        observations: Iterable of :class:`Observation` objects.

    Returns:
        Fitted :class:`ModelParams`.
    """
    if not observations:
        raise ValueError("need at least one observation to fit")

    # Group by model
    by_model: dict[str, list[Observation]] = {}
    for obs in observations:
        by_model.setdefault(obs.config.model, []).append(obs)

    params = ModelParams()
    residuals_all: list[float] = []

    for model_name, obs_list in by_model.items():
        cfg_m = MODEL_CONFIGS.get(model_name)
        if cfg_m is None:
            continue

        # Design matrix: one row per observation, one column per model (here: 1 column)
        A_rows: list[float] = []
        b_rows: list[float] = []

        for obs in obs_list:
            c = obs.config
            # Net time (known, not fitted)
            net_s = _network_time_s(c, cfg_m.activation_bytes_per_hop)
            # Effective compute coefficient: sum(1/(n_stages * scale_i))
            # Must match _compute_time_s which uses 1/(base * n_stages * s_i)
            compute_coeff = sum(1.0 / s for s in c.cpu_scales) / c.n_stages
            # We want: 1/tok_s = compute_coeff/base_tok_s + net_time
            # => compute_coeff/base_tok_s = 1/tok_s - net_s
            lhs = (1.0 / obs.tok_s) - net_s if obs.tok_s > 0 else 0.0
            A_rows.append(compute_coeff)
            b_rows.append(lhs)

        A = np.array(A_rows, dtype=float).reshape(-1, 1)
        b = np.array(b_rows, dtype=float)

        # lstsq: solve A @ x = b for x = [1/base_tok_s]
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        inv_base = float(sol[0])
        if inv_base > 0:
            params.base_tok_s[model_name] = 1.0 / inv_base
        else:
            params.base_tok_s[model_name] = 1.0  # degenerate fallback

        # Compute residuals for this model
        for obs in obs_list:
            predicted = effective_tokens_per_second(obs.config, params)
            residuals_all.append((obs.tok_s - predicted) / max(obs.tok_s, 1e-9))

    if residuals_all:
        res_arr = np.array(residuals_all)
        ss_res = float(np.sum(res_arr ** 2))
        params.mape = float(np.mean(np.abs(res_arr)))
        # R² proxy: 1 - SS_res / SS_tot (using fractional residuals as proxy)
        ss_tot = float(np.var(res_arr) * len(res_arr) + 1e-12)
        params.r2 = max(0.0, 1.0 - ss_res / (ss_tot * len(res_arr) + 1e-12))
        params.residual_variance = float(np.var(res_arr)) + 1e-6
    else:
        params.r2 = 0.0
        params.mape = 1.0

    return params


def predict(config: PipelineConfig, params: ModelParams) -> Claim:
    """Predict tokens/s with a 90% confidence interval.

    The CI is derived from the residual variance of the fitted model.
    Every call returns a :class:`Claim` with tier ``PROJECTED`` and explicit bounds.

    Returns:
        :class:`Claim` with ``tier=PROJECTED`` and populated ``ci_low``/``ci_high``.
    """
    value = effective_tokens_per_second(config, params)
    # 90 % CI: ±1.645 σ on the fractional error
    sigma_frac = math.sqrt(params.residual_variance)
    ci_low = value * max(0.0, 1.0 - 1.645 * sigma_frac)
    ci_high = value * (1.0 + 1.645 * sigma_frac)
    return Claim(
        value=value,
        tier=ClaimTier.PROJECTED,
        unit="tok/s",
        ci_low=ci_low,
        ci_high=ci_high,
    )
