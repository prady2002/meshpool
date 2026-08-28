# Benchmarks

_Generated 2026-08-28 21:39 UTC by `python -m sim.scenarios.bench_report`._

Every number below carries a provenance tier, per the project rule that no performance figure reaches a reader without one.

| Tier | What it means |
|---|---|
| **MEASURED** | Real model, real tokenizer, real forward passes on this machine. Compute time is wall-clock; the network is emulated. |
| **PROJECTED** | The fitted performance model extrapolated to hardware we do not own. Reported with fit quality and a 90% CI. |
| **CITED** | Someone else's published measurement, attributed. |

## Environment

- Host: Windows 11, Python 3.13.7
- Model: **nano-gpt** — 8 layers, d_model=128, trained checkpoint=True
- Draft: `SelfDraft` over the first 2 layers (runs on the first stage, so it adds **no wide-area hop**)
- K=4, 48 generated tokens, median of 3 runs

> The model is small on purpose. One machine cannot hold a 70B model, so the honest choice is a **real** small model with a real tokenizer and real matmuls, rather than a fake big one. Plan section 5.3.

## 1. Speculative Pipeline Decoding — MEASURED

The governing equation for wide-area pipeline inference is `t_token ≈ Σ compute + (N−1) × RTT`. The RTT term is fixed by physics, so the only way to beat it is to emit more than one token per traversal.

| Link | RTT | Stages | Baseline | SPD | Speed-up | Acceptance | Tokens/traversal |
|---|---:|---:|---:|---:|---:|---:|---:|
| LAN / same room | 1 ms | 1 | 456.89 tok/s | 637.53 tok/s | **×1.40** | 38% | 2.38 |
| LAN / same room | 1 ms | 2 | 245.56 tok/s | 329.63 tok/s | **×1.34** | 38% | 2.38 |
| LAN / same room | 1 ms | 4 | 147.36 tok/s | 258.14 tok/s | **×1.75** | 38% | 2.38 |
| LAN / same room | 1 ms | 8 | 107.97 tok/s | 214.77 tok/s | **×1.99** | 38% | 2.38 |
| Same metro | 15 ms | 1 | 401.11 tok/s | 542.38 tok/s | **×1.35** | 38% | 2.38 |
| Same metro | 15 ms | 2 | 91.76 tok/s | 180.98 tok/s | **×1.97** | 38% | 2.38 |
| Same metro | 15 ms | 4 | 36.13 tok/s | 79.48 tok/s | **×2.20** | 38% | 2.38 |
| Same metro | 15 ms | 8 | 16.62 tok/s | 37.32 tok/s | **×2.25** | 38% | 2.38 |
| Same country | 45 ms | 1 | 364.95 tok/s | 321.42 tok/s | **×0.88** | 38% | 2.38 |
| Same country | 45 ms | 2 | 36.46 tok/s | 81.92 tok/s | **×2.25** | 38% | 2.38 |
| Same country | 45 ms | 4 | 13.85 tok/s | 29.37 tok/s | **×2.12** | 38% | 2.38 |
| Same country | 45 ms | 8 | 6.10 tok/s | 13.44 tok/s | **×2.20** | 38% | 2.38 |
| Intercontinental | 150 ms | 1 | 236.87 tok/s | 346.69 tok/s | **×1.46** | 38% | 2.38 |
| Intercontinental | 150 ms | 2 | 12.73 tok/s | 28.22 tok/s | **×2.22** | 38% | 2.38 |
| Intercontinental | 150 ms | 4 | 4.35 tok/s | 9.54 tok/s | **×2.19** | 38% | 2.38 |
| Intercontinental | 150 ms | 8 | 1.88 tok/s | 4.15 tok/s | **×2.21** | 38% | 2.38 |

- Median acceptance rate across the grid: **38%**.
- Best multi-stage speed-up: **×2.25** (Same country, 2 stages).
- At depth 1 there is no wide-area hop to amortise, so SPD's advantage collapses — which is exactly what the equation predicts, and is the sanity check that these numbers describe the intended mechanism.

## 2. Time to first token — MEASURED

Prefill is position-parallel: a stage processes all `T` positions at once and forwards one `[T × d_model]` tensor, so TTFT does **not** grow by one pipeline traversal per prompt token. Chunked prefill bounds memory and stops one long prompt from head-of-line blocking a shared pipeline.

| Prompt tokens | Chunked prefill | TTFT | Hops | Bytes on the wire |
|---:|---|---:|---:|---:|
| 16 | off | 147 ms | 6 | 25.5 KiB |
| 16 | 32 | 150 ms | 6 | 25.5 KiB |
| 64 | off | 84 ms | 3 | 96.0 KiB |
| 64 | 32 | 164 ms | 6 | 96.0 KiB |

## 3. What this predicts for models this machine cannot hold — PROJECTED

The performance model fits the measured rows above with **R² = 0.9249**, **MAPE = 13.4%** (single-node throughput fitted at 217.9 tok/s for this model on this host).

**What is deliberately not claimed.** An earlier version of this page scaled that fitted throughput down by the active-parameter ratio (0.001B → 70B) and printed the result as a tok/s projection. That is a category error, not a conservative estimate: a 1M-parameter NumPy model is bounded by Python and BLAS call overhead, not by FLOPs, so its speed carries no information about a 70B model on a GPU. It produced `0.00 tok/s` for every target, and the generator exited 0 anyway. Both the extrapolation and that silent pass are gone; the generator now refuses to write this file if the fit degenerates or if SPD fails to raise the ceiling.

**What is claimed.** Wide-area pipeline decoding pays `(N−1)` hops of one-way latency per traversal regardless of hardware, and the table in section 1 measured how many tokens one traversal yields. Dividing the second by the first is an upper bound that assumes compute is free — true of no hardware, but a genuine ceiling, and precisely the quantity constellations and SPD are designed to move. Activation transfer is charged at a residential 20 Mbps uplink rather than ignored.

| Model | Deployment | Hops | Activation/hop | Traversal | Ceiling, baseline | Ceiling, SPD | Tier |
|---|---|---:|---:|---:|---:|---:|---|
| llama-2-70b | 4-node metro constellation | 3 | 16.0 KiB | 49.7 ms | 20.1 tok/s | **48.0 tok/s** | PROJECTED |
| llama-3.1-405b | 8-node metro constellation | 7 | 32.0 KiB | 161.8 ms | 6.2 tok/s | **14.7 tok/s** | PROJECTED |
| deepseek-v3 | 10-node metro constellation | 9 | 14.0 KiB | 141.6 ms | 7.1 tok/s | **16.8 tok/s** | PROJECTED |
| kimi-k2 | 11-node metro constellation | 10 | 14.0 KiB | 157.3 ms | 6.4 tok/s | **15.2 tok/s** | PROJECTED |
| kimi-k2 | 11 nodes, same country | 10 | 14.0 KiB | 282.3 ms | 3.5 tok/s | **8.4 tok/s** | PROJECTED |
| kimi-k2 | 11 nodes scattered worldwide | 10 | 14.0 KiB | 807.3 ms | 1.2 tok/s | **3.0 tok/s** | PROJECTED |

The SPD column is the baseline ceiling multiplied by the **2.38 tokens per traversal measured** in section 1 at a 38% acceptance rate. Note what the table says about topology: the same 11 nodes are worth 5.1× more in one metro than scattered worldwide. That is the entire argument for constellations, and it is arithmetic, not opinion.

### 3b. Sensitivity to per-token compute

Compute is the term this machine cannot measure for a 70B-class model, so it is an **input** here, printed in its own column, across a range that spans a fast GPU shard to a slow CPU one. Read the row whose assumption you believe.

| Assumed compute/token | Deployment | Baseline | SPD |
|---:|---|---:|---:|
| 0 (latency + bandwidth ceiling) | 11-node metro constellation | 6.4 tok/s | **15.2 tok/s** |
| 0 (latency + bandwidth ceiling) | 11 nodes, same country | 3.5 tok/s | **8.4 tok/s** |
| 0 (latency + bandwidth ceiling) | 11 nodes worldwide | 1.2 tok/s | **3.0 tok/s** |
| 10 ms | 11-node metro constellation | 6.0 tok/s | **14.2 tok/s** |
| 10 ms | 11 nodes, same country | 3.4 tok/s | **8.2 tok/s** |
| 10 ms | 11 nodes worldwide | 1.2 tok/s | **2.9 tok/s** |
| 25 ms | 11-node metro constellation | 5.5 tok/s | **13.1 tok/s** |
| 25 ms | 11 nodes, same country | 3.3 tok/s | **7.8 tok/s** |
| 25 ms | 11 nodes worldwide | 1.2 tok/s | **2.9 tok/s** |
| 50 ms | 11-node metro constellation | 4.8 tok/s | **11.5 tok/s** |
| 50 ms | 11 nodes, same country | 3.0 tok/s | **7.2 tok/s** |
| 50 ms | 11 nodes worldwide | 1.2 tok/s | **2.8 tok/s** |
| 100 ms | 11-node metro constellation | 3.9 tok/s | **9.3 tok/s** |
| 100 ms | 11 nodes, same country | 2.6 tok/s | **6.2 tok/s** |
| 100 ms | 11 nodes worldwide | 1.1 tok/s | **2.6 tok/s** |
| 200 ms | 11-node metro constellation | 2.8 tok/s | **6.7 tok/s** |
| 200 ms | 11 nodes, same country | 2.1 tok/s | **4.9 tok/s** |
| 200 ms | 11 nodes worldwide | 1.0 tok/s | **2.4 tok/s** |

Two things are worth reading off this table. Compute has to be **very** slow before it, rather than the network, is what limits a wide-area pipeline: 200 ms per token still leaves the worldwide deployment within a factor of 1.3 of its zero-compute ceiling. And bandwidth is not free at depth. Ten hops of 14 KiB over a residential 20 Mbps uplink is 57 ms against 100 ms of metro latency — about a third of the traversal — so uplink capacity belongs in constellation formation next to RTT, not dismissed as a rounding error the way it safely can be at depth 2.

## 4. Published results from other systems — CITED

| System | Model | Throughput | Source |
|---|---|---|---|
| Petals (public swarm) | Llama-2-70B | 6 tok/s | NeurIPS 2023 / arXiv 2312.08361 |
| Petals (public swarm) | Falcon-180B | 4 tok/s | NeurIPS 2023 / arXiv 2312.08361 |
| exo, 4x M3 Ultra over Thunderbolt-5 RDMA (<1 ms, LAN) | Kimi K2 | ~30 tok/s | exo labs, 2025 -- a LAN result, not comparable to WAN |

## Reproduce

```bash
python -m sim.scenarios.bench_report        # regenerates this file
python sim/scenarios/verify_spd_equivalence.py     # SPD correctness
python sim/scenarios/verify_netem_calibration.py   # model accuracy bound
python sim/scenarios/verify_streaming_ttft.py      # TTFT budget
```

