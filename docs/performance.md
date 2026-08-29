# Performance report (real measured numbers)

_Generated 2026-08-29 02:16 UTC by `python -m sim.scenarios.perf_report` on a live 3-node multi-process cluster._

> These are honest, reproducible figures from **one machine** (localhost networking, CPU-only, a tiny from-scratch demo model). They demonstrate that each capability works and is measurable — not datacenter throughput. The adapter seam (`docs/inference.md` §6) is where real GPUs/Petals/exo raise the absolute numbers; the control-plane behaviours below are unchanged.

## Environment

- Host: Windows 11, Python 3.13.7
- Cluster: 1 control plane + 3 simulated heterogeneous agents

| node | cores | ram (MB) | donate cpu% |
|---|---|---|---|
| sim-node-1 | 8 | 16000 | 60 |
| sim-node-2 | 4 | 8000 | 60 |
| sim-node-3 | 2 | 4000 | 60 |

## Batch engine (criterion d)

- **30 units** completed in **5.98 s** (**5.02 units/s**), status `completed`.
- Distributed across **3 nodes**: `{'sim-node-1': 17, 'sim-node-2': 10, 'sim-node-3': 3}`.

## Hosting engine (criterion c)

- **120/120 requests succeeded (100.0%)** across a deliberate mid-traffic node kill.
- Requests spanning the failure: `{'ok': 40}`.
- **Self-heal to desired replica count: 2.72 s.**

## Inference engine (criterion e) — flagship

- Model **8 layers** sharded into **3 pipeline stages**: `sim-node-1[0:5], sim-node-2[5:7], sim-node-3[7:8]` (no single node held them all).
- **74.85 tokens/s** generating 24 tokens.
- **Correctness oracle** (distributed == single-machine, greedy): **IDENTICAL ✅**.
- Killed the node holding sim-node-3 (layers 7:8) → **served correctly again after 15.93 s**; output still **identical ✅**.
- Sample: `'a with room to spare.\nMe'`

## Reproduce

```bash
python -m sim.scenarios.perf_report        # regenerates this file
python -m sim.scenarios.verify_batch       # asserts the batch invariants
python -m sim.scenarios.verify_hosting     # asserts hosting failover
python -m sim.scenarios.verify_inference   # asserts the inference oracle
```
