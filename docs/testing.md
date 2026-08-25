# Testing strategy

MeshPool's promise is "works like magic, never breaks," so tests are a first-class
deliverable — including **chaos/failover tests as acceptance gates**, not afterthoughts.

## Layers

| Layer | Where | Speed | What it proves |
|---|---|---|---|
| Unit | `tests/test_*.py` | fast (seconds) | core math + algorithms in isolation |
| End-to-end | `tests/test_e2e.py` → `sim/scenarios/*` | slow (boots clusters) | the MVP acceptance criteria (b)–(e) on real clusters |
| Chaos | inside each scenario | slow | recovery after kills mid-job/traffic/inference |

## Running

```bash
pytest                 # unit only (default; -m "not slow")
pytest -m slow         # the end-to-end acceptance gates
pytest -m "slow or not slow"   # everything (portable across shells)
```

## Unit tests (what's covered)

- **`test_model.py`** — forward-pass shapes; the **shard == full-model correctness
  oracle** (a pipeline of `ShardModule`s vs `forward_full`, `max |Δ| < 1e-4`); tokenizer
  round-trip; deterministic greedy generation at `temperature 0`.
- **`test_train.py`** — the hand-written backprop matches numeric gradients (float64
  central differences, worst relative error `< 1e-4`).
- **`test_scheduler.py`** — `topology_partition` covers all layers contiguously with no
  single stage holding the whole model, and **refuses** when the pool is too small;
  `best_fit_node` picks the tightest fit; `bin_pack_replicas` spreads before packing.
- **`test_trust.py`** — EMA update direction; Tier-0 pass-through vs Tier-1 reputation;
  fair-share priority monotonicity; ledger donated-minus-consumed accounting.
- **`test_config.py`** — `home`/`office`/`enterprise` presets set the right trust tier,
  enrollment, benchmarks and HA (scale is config, not code).
- **`test_secrets.py`** — the scoped secret broker: single-use redemption, node-scope
  enforcement, TTL expiry (fake clock), and names-only listing (values never exposed).
- **`test_adapters.py`** — the inference **adapter seam**: the default `MeshShard` backend
  is available, config resolves for known models (and raises on unknown), the adapter's
  partition plan matches the scheduler, and the documented Petals/exo stubs report
  unavailable rather than silently no-op'ing.
- **Global-mesh units** — `test_identity.py` (Ed25519 device keys: stable id, sign/verify),
  `test_tunnel.py` (hub correlation + `CompositeTransport` tunnel-vs-direct routing),
  `test_device_registry.py` (device-signature registration + id-keyed matching, URL fallback
  preserves LAN), `test_sandbox.py` (memory/timeout/output caps + payload validation), and
  `test_reconnect.py` (tunnel re-registration with explicit state + the reachability
  predicate that keeps detached tunnel nodes off the scheduler).

## End-to-end / chaos gates

Each scenario in `sim/scenarios/` boots a **real** control plane + 3 agent processes with
heterogeneous capacity, exercises an engine, injects a failure, and asserts recovery. They
exit non-zero on any broken invariant, and `tests/test_e2e.py` runs them as subprocesses
(marked `slow`).

- **`verify_dashboard.py`** (criterion b) — boot 3 heterogeneous nodes; the node API lists
  them all online with real capacity, the summary aggregates the pool, and the dashboard
  page, static assets and live SSE stream are served.
- **`verify_batch.py`** (criterion d) — distribute units across all nodes; kill one →
  its in-flight units complete elsewhere; job finishes.
- **`verify_hosting.py`** (criterion c) — deploy across nodes; kill one mid-traffic → no
  dropped requests beyond in-flight; self-heals to desired replicas.
- **`verify_inference.py`** (criterion e) — a model too big for any one node is sharded and
  served; distributed output is **bit-identical** to a single-machine reference; kill a
  node → repartition onto survivors → output **still identical**.
- **`verify_secrets.py`** (spec §5) — a workload HMAC-signs requests with a credential it
  **never** sees in its job spec; the value never appears in any API response; a unit that
  didn't request the secret gets nothing; leases are single-use.

### Global-mesh & reliability gates (`scale=public`, Tier 2)

- **`verify_global.py`** — 3 agents connect **only** over the outbound tunnel (unreachable
  `node.url`); batch fan-out+retry, hosting LB+failover, and sharded inference all run over
  the tunnel, with the bit-identical inference oracle preserved.
- **`verify_sandbox.py`** — correct curated units run sandboxed (embed == oracle); over-cap /
  abusive jobs are rejected fast by the memory/timeout/output caps + payload validation.
- **`verify_trust_credits.py`** — a deliberately lying node is caught on **every** unit,
  paid nothing, and down-scored; the credit ledger conserves (donor debit == consumer
  credit) and an overspend is refused (402).
- **`verify_ux.py`** — the Tier-2 product path: zero-signup local sign-in, wallet/ledger
  APIs, and submit/deploy/chat all function against a public coordinator.
- **`verify_onboarding.py`** — a clean-room **unzip of the shipped build** followed by
  `python -m meshpool.launcher self-test` comes up connected and prints `SELFTEST_OK`.
- **`verify_chaos.py`** — a PUBLIC tunnel-only cluster runs a baseline job, then the
  **coordinator is hard-killed and restarted** on the same data dir. The mesh self-heals with
  no re-onboarding: agents re-register + re-attach their tunnels, the durable ledger survives
  (the starter grant is **not** re-issued), and a fresh job completes with the ledger still
  conserving. Wired as `test_coordinator_restart_self_heals`.

## Why the oracle matters

Most "distributed inference" demos can't tell you whether the sharded model computes the
*same thing* as the original. MeshPool's determinism-at-temperature-0 oracle turns that
into a hard, automated equality assertion — the strongest possible proof the sharding is
correct. See [inference.md §4](inference.md).
