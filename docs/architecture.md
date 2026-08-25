# Architecture

MeshPool pools several owner-controlled machines into one resource you can deploy
work onto. This document walks the components, the data flow for each engine, the
schemas, and how this runnable build maps onto the reference production stack.

## 1. Big picture

```
 [Node Agent]   [Node Agent]   [Node Agent]      one process/binary per machine
      \              |              /
       \--- LAN discovery + rendezvous ---/       (register, heartbeat, run work)
                     |
              [ Control Plane ]
   node registry │ schedulers │ trust + contribution ledger │ REST + SSE API
                     |
   ┌─────────────────┼──────────────────┬──────────────────┐
[Batch Engine]  [Hosting Engine]  [Inference Engine]   [Dashboard + CLI]
```

Everything is one small owner (Trust Tier 0) in the default LAN posture, so we replace
trustless verification with a lightweight reputation + fair-share ledger. Flip
`scale=public` and the **same build** becomes a global mesh strangers can safely share —
see [§7, The global mesh](#7-the-global-mesh-scalepublic).

## 2. Node Agent

One agent runs per machine (`meshpool/agent/daemon.py`, a FastAPI app). It:

- **Probes capability** — CPU cores/arch, RAM, GPU/VRAM, disk, and *measured* link
  latency/bandwidth to the control plane (never guessed).
- **Registers + heartbeats** — announces itself, then reports load + remaining
  headroom every few seconds so the schedulers always see current capacity.
- **Enforces the donation policy** — CPU %, RAM cap, and a schedule
  (always / idle-only / business-hours / outside-hours). Enforcement happens at the
  runtime/OS level (worker-pool sizing, resource caps); **the workload is never
  trusted to self-limit.**
- **Runs sandboxed work** behind a pluggable `Runtime` interface: batch units, app
  replicas, and model shards. `ProcessRuntime` is the default here; a `DockerRuntime`
  slots in unchanged in production.
- **Kill-switch** — `reclaim` drains and frees the machine immediately, no approval.
  Reclaim/shutdown also unload any model shards.

Key shard endpoints (inference hot path): `POST /shard/load`, `/shard/forward`,
`/shard/unload`.

## 3. Control Plane

`meshpool/controlplane/app.py` + `registry.py` + `scheduler.py`. Responsibilities:

- **Node registry** — authoritative list of nodes, their capabilities, policies,
  load, status, trust and last-heartbeat. A reaper marks nodes `OFFLINE` when
  heartbeats lapse (`node_timeout_s`) and notifies the engines so they can heal.
- **Three scheduling strategies**:
  - batch dispatch (`BatchManager._pick_node`) — **worst-fit / spread**: each unit goes
    to the schedulable node with the *most* free CPU slots, fanning a job out evenly so
    every machine stays busy (never oversubscribing a node's donatable slots or RAM). The
    tighter-packing `best_fit_node` in `scheduler.py` is a placement utility, not the
    batch dispatcher.
  - `bin_pack_replicas` — app replicas spread across distinct nodes first
    (anti-affinity → survive one failure), then pack.
  - `topology_partition` — model layers are cut into contiguous per-node blocks sized
    by usable memory and weighted by compute; recomputed on join/leave.
- **Trust + contribution ledger** (`common/trust.py`) — EMA reputation + donated-vs-
  consumed accounting. Inert at Tier 0 (single owner); **active at Tier 2** (`scale=public`)
  where it drives earn-to-spend credits and known-answer verification (see §7).
- **API** — REST under `/api/v1/*`, an SSE event stream at `/api/v1/events`, the
  OpenAI-compatible `/v1/chat/completions`, and it serves the dashboard at `/`.

## 4. The three engines

### Batch (`meshpool/batch`, `docs/batch.md`)
A job is a list of independent work units + a named function. The engine spreads units
(worst-fit) across nodes with spare CPU, collects streamed results, and **retries a
unit on a different node** if its node dies mid-flight — failure is never assumed to
be the job's fault.

### Hosting (`meshpool/hosting`, `docs/hosting.md`)
Deploy = an app + a replica count. The engine bin-packs replicas across nodes, puts a
**load balancer** in front, health-checks replicas, and on node loss re-routes then
reschedules the lost replica elsewhere — no dropped requests beyond in-flight.

### Inference (`meshpool/inference`, `docs/inference.md`) — flagship
Shards a model's decoder layers pipeline-parallel across nodes so a model too big for
any one machine still runs. Only activations cross the wire. Auto-repartitions on node
loss. Exposes an OpenAI-compatible API and reports real tokens/sec. Because every node
rebuilds its slice from the same checkpoint, distributed output is **bit-identical** to
a single-machine run.

## 5. Data contracts (schemas)

All boundary messages are Pydantic models in `meshpool/common/models.py`, mirrored 1:1
by proto3 in [`proto/meshpool.proto`](../proto/meshpool.proto). The persistent store is
SQLite (`meshpool/common/db.py`) with tables for nodes, jobs, deployments, model
deployments, ledger and trust — schema is already shaped for Tier 1/2 so enabling them
is a config flip, not a migration rewrite.

## 6. Scale is configuration, not code

`meshpool/common/config.py` defines `home` / `office` / `enterprise` / `public` presets.
The same binaries change trust tier, enrollment, timeouts, benchmark probing and HA purely
via the active preset — the spec's "1→3 is a config change, not a rewrite" requirement.
`home` → Tier 0 (LAN, single owner); `office`/`enterprise` → Tier 1; `public` → Tier 2
(global mesh, §7).

## 7. The global mesh (`scale=public`)

The default posture pools machines **you own on one LAN**: engines reach each agent by
directly HTTP-calling the address it advertised (`node.url`). Flipping `scale=public`
(Tier 2) turns the *same build* into a mesh **anyone on the internet can safely share**,
by layering five mechanisms on top of the LAN core — none of which change LAN behaviour:

- **Agent-initiated reverse tunnel** (`meshpool/agent/tunnel_client.py`,
  `meshpool/cloud/tunnel.py`). With `MESHPOOL_TUNNEL=1` the agent stops advertising a LAN
  IP (it registers a sentinel `http://127.0.0.1:1`) and instead **dials out** to the
  coordinator over a persistent HTTP stream (`GET /tunnel/stream` NDJSON push +
  `POST /tunnel/reply`). The coordinator's `TunnelHub` frames each engine→agent request
  with a correlation id, pushes it down that socket, and matches the reply — so machines
  behind home routers/firewalls join with **nothing to port-forward**. The agent forwards
  frames to its own loopback handlers, so every engine (batch/hosting/inference) and the
  `/app` proxy work unchanged over the tunnel. Reconnect uses exponential backoff and
  re-registers on a stale token.
- **`AgentTransport` seam** (`common/transport.py`). Engines call
  `transport.call(node_id, ...)` / `transport.proxy(...)` instead of hardcoding
  `node.url`. `CompositeTransport` (`cloud/tunnel.py`) routes over the tunnel when the node
  is hub-connected, else falls back to direct HTTP — so LAN and tunnelled nodes coexist in
  one pool.
- **Device identity** (`common/identity.py`). Each device generates an Ed25519 keypair on
  first run (`~/.meshpool/identity`), yielding a stable `dev_...` id. Registration is a
  **challenge-response signature**; the registry keys nodes by verified device id (falling
  back to URL on LAN), so anyone contributes instantly with zero signup and no shared token.
- **Credits + trust (Tier 2 active).** The ledger records both donate→earn and
  consume→spend; a one-time anonymous **starter grant** lets a newcomer consume before
  earning. An **earn-to-spend gate** (401 without a session / 402 when balance ≤ 0 / 429 on
  rate-limit) deters sybil abuse, and donor+consumer are billed the identical
  `cost = cpu_pct × compute_s` so the ledger conserves. Below the trust threshold a node's
  units are **verified against a local oracle** (known-answer spot-checks); a node that is
  ever caught lying is verified on **every** unit thereafter and paid nothing for wrong
  results.
- **Reachability / self-healing.** A tunnelled node is only schedulable **while its tunnel
  is attached** (`ControlPlane._node_reachable` + the registry `reachability` predicate),
  so a node whose tunnel has dropped is skipped instead of being handed work it can't
  receive. This makes a mid-run tunnel drop fail over promptly, and lets the pool **self-heal
  after a coordinator restart**: agents re-register and re-attach their tunnels, the durable
  SQLite credit ledger survives the crash (the starter grant is not re-issued), and new work
  schedules only onto genuinely reattached nodes. Proven by
  [`sim/scenarios/verify_chaos.py`](../sim/scenarios/verify_chaos.py) (wired into
  `tests/test_e2e.py`).

The coordinator is a **deployable, locally-testable** service; operating an always-on
public one is the hosted "MeshPool Cloud" roadmap step.

## 8. Substrate → production upgrade path

The reference spec targets Go + gRPC + Docker + React. This repository implements the
**same architecture, contracts and roadmap** in a small-dependency substrate so every
acceptance criterion actually runs here. Each choice has a documented upgrade path:

| Spec target | Built here (runnable) | Production upgrade |
|---|---|---|
| Go services | Python (FastAPI/uvicorn) | Port per spec §6 |
| gRPC + protoc | HTTP/JSON + SSE, Pydantic contracts mirroring proto | gRPC once protoc available |
| Docker sandbox | `ProcessRuntime` behind a `Runtime` interface | `DockerRuntime` / Firecracker |
| Docker Compose sim | Multi-process harness (`sim/harness.py`) | Compose file in `/deploy` |
| React/TS/Tailwind | FastAPI-served vanilla HTML/CSS/JS | React app once npm reachable |
| mDNS/zeroconf | Rendezvous registration (spec's own fallback) | zeroconf on real LAN |
| HF/torch/Petals model | Minimal NumPy GPT, pipeline-parallel | Petals/exo/torch via adapter |
| SQLite | SQLite (unchanged) | Postgres for HA |

The component boundaries and wire contracts are identical across both columns; only the
implementation substrate differs.
