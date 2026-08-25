# NetEm — MeshPool Network Emulator

## Overview

Phase 0.4 adds a software network emulator and calibrated performance model so
every future performance claim is grounded in real measurements, not guesswork.

## Claim tiers

Every performance number in MeshPool is tagged with one of three tiers:

| Tier | Meaning |
|------|---------|
| **MEASURED** | Directly observed in this repository's test suite on real hardware. |
| **PROJECTED** | Derived from the fitted perf model. Always accompanied by a 90% confidence interval. A bare number without a CI is not allowed. |
| **CITED** | Published paper or vendor announcement (citation included in source). |

`sim/perfmodel.py` exports `Claim(value, tier, ci_low, ci_high, citation)` and the
`CITED_*` constants with their sources.

## Running the calibration gate

```powershell
python sim\scenarios\verify_netem_calibration.py
```

Expected output: `PASS — all NetEm calibration checks passed`.  Exit code 0 on pass, 1 on fail.

### What check 5 measures and asserts (as of Phase 0.4)

Check 5 is an **anti-circular** calibration gate.  Ground truth is always MEASURED
wall-clock time from real `generate_baseline` / `speculative_generate` runs — never
synthetic values produced by `effective_tokens_per_second` itself.

**Experimental setup:**
- Model: `nano-gpt` (8 layers, d_model=128, real trained checkpoint)
- Injected RTT: 150 ms throughout (one-way delay = 75 ms per hop)
- RTT-dominated regime: at 150 ms, RTT >> compute (~4–6 ms total), so timer
  granularity (~15 ms on Windows) is a < 10% confound
- 24 tokens per run; best-of-3 wall-clock to suppress OS scheduler noise
- Train/test split **by pipeline depth** (not by row): depths {2, 4, 8} train; depths {3, 5} held out
- Anti-circularity guards:
  - **Spread**: `max(tok/s) / min(tok/s) ≥ 3` (real dynamic range across configs)
  - **Wrong-params**: a 10× wrong `base_tok_s` must score higher error than the fit

**Typical results from the last gate run:**
| Config | Measured | Predicted | Error |
|--------|----------|-----------|-------|
| depth=3, RTT=150 ms | ~5.5 tok/s | ~5.3 tok/s | ~7% |
| depth=5, RTT=150 ms | ~2.8 tok/s | ~2.9 tok/s | ~6% |

Hold-out MAPE (against MEASURED values): **~7%**, well inside the 25% stated bound.

**SPD validation:**
- `SelfDraft(cfg, params, 2)`, K=4 draft tokens, same 150 ms RTT
- Measured speedup: **~2.5–2.8×** on depth=3 and depth=5 (both WAN-dominated)
- Model-predicted speedup: **~2.3×** (within 2× of measured; shape confirmed correct)
- The model form `speedup ≈ (1 − a^{K+1})/(1 − a)` correctly captures that
  speedup rises with `(depth-1) × RTT` and that a shallower draft raises `a`

**Known limitation — model accuracy at low RTT / shallow pipeline:**
At RTT ≤ 45 ms with depths 1–2, per-hop asyncio scheduling overhead (~10–20 ms)
becomes non-negligible relative to the injected RTT.  The model form
`t = 1/base + (D−1)×RTT/2` absorbs this overhead into the compute term, making
`base_tok_s` smaller than the true single-node throughput.  Calibrating with
RTT ≥ 150 ms avoids this confound.  For real large models (e.g. Llama-2-70B where
compute ≈ 200 ms/token), the compute term dominates and this issue vanishes.

## Adding a topology

1. Add `NodeSpec` entries to a new list in `sim/topology.py` (or use `residential_mixed`).
2. Write a `link_fn(a, b) -> LinkProfile` that returns the right preset for each pair.
3. Construct `Topology(nodes, link_fn, seed=<your seed>)`.
4. Pass it to `Harness(topology=topo)` or create `NetEm(topo)` directly.

Named presets in `sim/topology.py`:

| Preset | RTT | Bandwidth | Loss |
|--------|-----|-----------|------|
| `lan` | 1 ms | 1 Gbps symmetric | ~0% |
| `metro` | 15 ms | 500/20 Mbps | 0.1% |
| `national` | 45 ms | 200/10 Mbps | 0.3% |
| `intercontinental` | 150 ms | 100/5 Mbps | 1% |

## What the emulator does NOT model

- **Real TCP/QUIC congestion control dynamics** (cwnd growth, slow start, CUBIC
  sawtooth, BBR probing). Bandwidth is treated as a hard cap, not a dynamic window.
- **Kernel scheduling noise** (OS timer resolution, CPU wakeup latency). The emulator
  uses `asyncio.sleep`, which on Windows has ~15 ms resolution; use `time_scale` to
  run tests fast rather than relying on sub-millisecond timing.
- **Real GPU behaviour** (memory bandwidth, cache locality, operator fusion, NCCL
  collective overhead). The compute model is a scalar multiplier over a fitted
  reference throughput — it does not simulate tensor parallelism or PCIe transfers.
- **Real NAT hole-punching success rates** beyond the documented 0% for
  symmetric↔symmetric. Partial cone-to-cone success rates vary by router firmware and
  are not modelled.
- **Packet reordering effects on TCP performance** (retransmit, reordering buffer).
  The emulator notes reorder probability but does not simulate the resulting
  throughput degradation.
