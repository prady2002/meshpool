# Security baseline (spec §5, §9.8)

MeshPool runs other people's workloads on a machine you own, so the donor's safety comes
first. The default LAN posture targets **Trust Tier 0** (single owner / home LAN); flipping
`scale=public` activates **Trust Tier 2** (global mesh — anyone on the internet), whose
delivered protections are described in [Global mesh: Tier 2, what's active
today](#global-mesh-tier-2--whats-active-today) below. This document is the honest current
state plus the production hardening path.

## Threat model (Tier 0)

One person (or one household) owns every machine. There is no adversarial tenant, so we
optimise for **protecting the donor's machine and network from buggy/greedy workloads**,
not for defending against a malicious co-tenant (that's Tier 1/2). This is the default
posture; the global-mesh (Tier 2) threat model — anonymous strangers, adversarial results,
sybil abuse — is handled by the mechanisms in [Global mesh: Tier
2](#global-mesh-tier-2--whats-active-today). **No hand-rolled crypto and no blockchain**
remain explicit non-goals in every posture (identity uses vetted Ed25519 via
`cryptography`), and **anonymous public compute is restricted to the curated workload set**
— arbitrary stranger code is not executed at any tier today.

## What protects the donor today

- **Donation caps are enforced by the runtime, not the workload.** `ProcessRuntime`
  (`meshpool/agent/runtime.py`) sizes its worker pool from the CPU cap
  (`slots = floor(cpu_pct/100 × cores)`), so a job physically cannot occupy more cores
  than donated. The workload is **never trusted to self-limit** (spec §4.1). RAM requests
  are scheduled against declared headroom.
- **Instant kill-switch / reclaim.** `reclaim` hard-terminates the worker pool's child
  processes so CPU is freed **immediately**, and unloads any model shards. No approval, no
  drain wait required. Exposed as `POST /api/v1/nodes/{id}/reclaim`, the CLI
  `meshpool reclaim`, and (in the full product) the tray UI's prominent button.
- **Schedule-gated donation.** A node off its schedule (idle-only / business-hours /
  outside-hours) advertises **zero headroom**, so nothing is placed on it.
- **Pluggable runtime boundary.** The scheduler only ever says *"run this with these
  limits"*; the `Runtime` interface decides how. That seam is exactly where stronger
  isolation drops in without touching the control plane.
- **Scoped, single-use secret brokering (working today).** Credentials a workload needs
  (API keys, DB passwords) are **never** baked into the job definition or written to a
  worker's disk. The owner stores a secret with `meshpool secrets put NAME VALUE` (or
  `POST /api/v1/secrets`); the control plane keeps the *value in memory only* and never
  returns it from any API (`GET /api/v1/secrets` lists **names only**). At dispatch the
  control plane mints a **single-use, node-scoped, TTL-bounded lease** for each secret a
  job declares (`secret_names`), and only the lease id travels with the work unit. The
  agent redeems the lease just-in-time over the node↔control-plane channel
  (`POST /api/v1/secrets/redeem`), injects the value into the unit in memory, and the
  value is discarded when the unit ends. A lease that is expired, already redeemed, or
  presented by a different node is refused — so a leaked lease id is useless. **Node
  identity is verified server-side**: the redeem call carries the per-node session token
  minted at registration (`X-MeshPool-Node-Token`), and the control plane derives the
  node id from that token rather than trusting the caller-supplied `node_id`, so a stolen
  lease id can't be redeemed by another node that merely claims the victim's id (at Tier
  1+ a valid token is *required*). Implemented in `meshpool/common/secrets.py` +
  `controlplane/registry.py`; proven end-to-end by `sim/scenarios/verify_secrets.py`
  (a unit HMAC-signs a request with a token it never sees in its spec, the value never
  appears in the job record, and unrequested units get nothing).

  Secret *values* are stored per `(owner, name)` and a job only leases secrets under its
  own owner. At Tier 0 (single owner) the owner is a trusted label; at Tier 1+ the owner
  is bound to an authenticated principal (per-user/API-key), the same way enrollment
  gates *which* devices may join — see the upgrade path below.

## Global mesh: Tier 2 — what's active today

Flipping `scale=public` (Tier 2) hardens the mesh so **strangers can safely share it**, for
the **curated** workload set (batch compute, bundled model inference, app templates). These
are delivered and gated by tests, not aspirational:

- **Locked-down sandbox for every contributed unit** (`meshpool/agent/sandbox.py`). On top
  of the CPU-slot cap, curated work runs under a **hard memory cap** (Windows Job Object /
  POSIX `rlimit`), a **wall-clock timeout**, and an **output-size cap**, with **input
  validation** (`validate_payload`) rejecting anything outside the curated schema. The
  worker times only the curated function body, so consumers never pay sandbox spawn/import
  overhead. Gate: [`sim/scenarios/verify_sandbox.py`](../sim/scenarios/verify_sandbox.py) —
  correct units run; over-cap/abusive jobs are rejected fast.
- **Cryptographic device identity, no shared token.** Membership is by Ed25519 device key
  (`common/identity.py`), verified by challenge-response at registration — there is **no
  admin enroll token** at Tier 2. A stolen node session token is re-minted on reconnect, and
  a tunnelled node is only schedulable while its authenticated tunnel is attached, so a node
  that can't prove liveness gets no work.
- **Earn-to-spend credits deter abuse.** Consumers must hold a session and positive balance;
  the ledger gate returns **401** (no session), **402** (balance ≤ 0), or **429** (rate
  limit) — so a sybil flood can't consume for free. Donor and consumer are billed the
  identical `cost = cpu_pct × compute_s`, so the ledger conserves and can't be inflated.
- **Results from low-trust nodes are verified, and proven liars are never trusted.** Below
  the trust threshold, a node's output is spot-checked against a **local known-answer
  oracle**; a node ever caught returning a wrong result is verified on **every** unit
  thereafter and **paid nothing** for wrong answers, and its reputation EMA is driven down.
  Gate: [`sim/scenarios/verify_trust_credits.py`](../sim/scenarios/verify_trust_credits.py).
- **Coordinator-restart self-healing** (`sim/scenarios/verify_chaos.py`). If the coordinator
  crashes and restarts, the durable SQLite credit ledger survives (the one-time starter
  grant is **not** re-issued, so balances can't be farmed by bouncing the server), agents
  re-authenticate and re-attach their tunnels, and work only schedules onto genuinely
  reachable nodes.

**Still explicitly out of scope at Tier 2:** running **arbitrary** (non-curated) code or
containers from strangers, real-money settlement, and defending a **hostile multi-tenant**
sandbox escape — those need the container/microVM isolation in the upgrade path below and
remain opt-in later phases. Tier 2 today is safe because the executable surface is the
**curated function set**, not arbitrary code.

## Production hardening (the upgrade path)

The `DockerRuntime` documented in `runtime.py` implements the same `Runtime` interface
with real isolation:

- **Sandboxed-by-default execution** — containers with no host privileges; CPU/RAM/PIDs
  bounded by **cgroups v2 (Linux) / Job Objects (Windows) / container limits**, per-OS
  enforced.
- **Workload network isolation** — each workload on its own virtual segment so it **cannot
  scan or reach the donor's LAN**.
- **mTLS everywhere** for node ↔ control-plane and node ↔ node (WireGuard overlay is a
  Phase 2 item for cross-subnet office/enterprise). This also protects the secret-lease
  channel above from network sniffing.
- **Secret store backing** — the in-memory broker swaps its store for **Vault/KMS** so
  secret values survive a control-plane restart and get an audit trail (the lease
  semantics stay identical).
- **Signed agent binaries + container-image verification** so only trusted code runs.
- **Authenticated principals (owners/API keys)** — at Tier 1+, both `POST /api/v1/secrets`
  and job submission bind `owner` to an authenticated identity (SSO/API-key), so a
  submitter can only lease secrets it actually owns. The `(owner, name)` scoping and the
  per-node session token are already in place; this step authenticates *who* the owner is.
- **Firecracker microVMs** for hostile-multi-tenant Tier 2, if that phase is ever chosen.

## Tier ladder

| Tier | Context | Verification | Status |
|---|---|---|---|
| 0 | home, one owner | full mutual trust; caps + kill-switch | **active** (default) |
| 1 | office, SSO-enrolled | EMA reputation, known-answer benchmarks, redundant exec below threshold | fields present, inert until `scale=office` |
| 2 | public global mesh | device-key identity + locked-down sandbox + known-answer verification + earn-to-spend credits, **curated workloads only** | **active** (`scale=public`) |

Enabling Tier 1 (`scale: office`) or Tier 2 (`scale: public`) is a **config flip**, not a
rewrite — the ledger, trust and identity schemas are already shaped for it. Arbitrary-code /
hostile-multi-tenant isolation (containers/microVMs) remains the documented next step for
Tier 2.
