# Roadmap

MeshPool ships in two postures from **one build**: the default **LAN pool** (machines you
own, Trust Tier 0) and the **global mesh** (`scale=public`, Trust Tier 2 — anyone, anywhere).
Moving between postures is a **config change, not a rewrite**. Everything under *Delivered*
is gated by automated acceptance tests (`tests/` + `sim/scenarios/`); everything under *Next*
is honestly deferred.

## Delivered ✅

Each item below boots real processes in an acceptance gate wired into `tests/test_e2e.py`.

### LAN core (Tier 0)
Batch + hosting + flagship distributed inference; spread (worst-fit) / bin-packing /
topology-aware schedulers; Tier 0 trust + contribution ledger; scoped single-use secret
brokering; zero-config LAN onboarding; multi-process sim harness with chaos tests;
OpenAI-compatible inference API. Gates: `verify_batch`, `verify_hosting`, `verify_inference`,
`verify_lan`, `verify_secrets`.

### Global connectivity core (Tier 2)
- **Agent-initiated reverse tunnel** + a **deployable coordinator** (`meshpool/cloud/`), so
  machines anywhere join through NAT/firewalls with **nothing to port-forward**. Batch,
  hosting and sharded inference all run over the tunnel.
- **`AgentTransport` seam** (`common/transport.py`) — engines route through
  `transport.call/proxy`; `CompositeTransport` uses the tunnel when connected, else direct
  HTTP, so LAN and tunnelled nodes coexist.
- **Device identity** — Ed25519 keypair per device, generated on first run; zero-signup,
  with an optional lightweight account link that pools credits/reputation across machines.
- Gate: `verify_global` (tunnel-only agents with unreachable `node.url` run every engine).

### Safety, credits, trust (Tier 2)
- **Locked-down sandbox** for curated workloads — memory cap (Job Object / rlimit),
  wall-clock timeout, output cap, input validation. Gate: `verify_sandbox`.
- **Active earn-to-spend credit ledger** with a one-time starter grant; conservation
  invariant (donor debit == consumer credit); 401/402/429 gate deters sybil abuse.
- **Tier 2 trust** — reputation EMA + **known-answer spot-checks** that catch a dishonest
  node, stop paying it, and verify it on every unit thereafter. Gate: `verify_trust_credits`.

### Product UX & onboarding
Redesigned **Use / Contribute / Wallet / Explore** dashboard (zero-build vanilla JS);
one-command cross-platform onboarding (`start.*`, `contribute.*`, `python -m
meshpool.launcher solo|contribute|self-test`); reproducible allowlist zip build
(`deploy/build_zip.*`) that never ships device keys or local state. Gates: `verify_ux`,
`verify_onboarding` (clean-room unzip + self-test).

### Reliability & chaos
Kill/retry (batch), node-death failover (hosting), survive-a-death re-partition with the
**bit-identical inference oracle** preserved, and **coordinator-restart self-healing**
(`verify_chaos`): after the coordinator is hard-killed and restarted, agents re-register and
re-attach their tunnels, the durable ledger survives (starter grant not re-issued), and work
schedules only onto genuinely reachable nodes. This gate found and fixed two real bugs
(tunnel re-registration state, and "zombie online" scheduling of detached tunnel nodes).

### Model training (single-machine)
**LoRA fine-tuning** of the nano-GPT over a **frozen base** (`meshpool/inference/lora.py`;
CLI `python -m meshpool.inference.lora`): only low-rank adapters train, and they reuse the
**already gradient-checked** forward/backward, so a fine-tune is kilobytes and runs in
seconds on a single machine. Guarded by a finite-difference gradient check on the low-rank
projection **and** a loss-decreases-on-a-fixed-corpus gate (`tests/test_lora.py`). Adapters
merge back into a full checkpoint for serving by the ordinary inference engine. Distributed
LoRA over the sharded pipeline is deferred (see Next).

## Next (honestly deferred)

- **Hosted MeshPool Cloud** — operate the public coordinator + relay as an always-on
  service. The coordinator is built and tested locally; hosting it is an **operational step**
  (provisioning/DNS/TLS/scaling), not new core code. Deploy artifacts + an operator runbook
  now ship: [`docs/deploy-cloud.md`](deploy-cloud.md) (Dockerfile, Compose, systemd unit).
- **Arbitrary containers** — opt-in Docker/OCI workloads beyond the curated set, with real
  container/microVM isolation (the `DockerRuntime` seam + Firecracker path in
  [`security.md`](security.md)). Required before executing untrusted arbitrary code.
- **Distributed LoRA training** — extend the delivered single-machine LoRA fine-tune to run
  over the **frozen, sharded pipeline** (activation-gradients cross stage boundaries; only
  small adapter tensors train, the base stays pipeline-sharded). Needs a symmetric backward
  stage hop the runtime doesn't expose yet. *Not* distributed pretraining, which stays out of
  scope (bandwidth/latency-bound on heterogeneous consumer networks).
- **GPU acceleration** via the model adapter seam (Petals/exo/torch backends).
- **Direct P2P hole-punching** — a latency optimization over the relay tunnel; not required
  for correctness.
- **Control-plane HA / multi-region** — Raft standby (`ha` preset flag already threaded) and
  Postgres-backed durable state for the hosted service.
- **Real-money settlement** — the credit ledger is compute-credits only; payments are a
  separate, deliberate track.
- **Tensor parallelism within a layer** and full **SSO/AD + audit/compliance** for
  office/enterprise (Tier 1) multi-site deployments.

## Non-goals (stated so scope can't creep)

- **Blockchain / token settlement** or hand-rolled cryptographic verification.
- **Distributed pretraining** across the pool.
- Executing **arbitrary untrusted code** without container/microVM isolation.

## Posture = configuration, not code

| Mechanism | Tier 0 (`home`) | Flip to enable |
|---|---|---|
| Trust EMA + credits (`common/trust.py`) | inert (score = 1.0) | `scale=office` → Tier 1; `scale=public` → Tier 2 |
| Sandbox limits (`agent/sandbox.py`) | CPU cap only | mem/timeout/output caps active at Tier 2 |
| Enrollment (`config.py`) | open LAN | `office`/`enterprise` require enroll token; `public` uses device keys |
| Reachability filter (`registry.reachability`) | LAN nodes always schedulable | tunnel nodes schedulable only while attached |
| Fair-share (`fair_share_priority`) | constant | active above Tier 0 |
| HA (`config.py: ha`) | false | true in enterprise preset |
| Runtime isolation (`agent/runtime.py`) | `ProcessRuntime` + sandbox | `DockerRuntime` / Firecracker for arbitrary code |
