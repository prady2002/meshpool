<div align="center">

# 🕸️ MeshPool

### Turn the machines you already own into one pooled computer.

*Deploy web apps with automatic failover · Run batch jobs across every spare core ·
Serve a model **too big for any single machine** by sharding it across several.*

</div>

---

MeshPool takes the idle compute sitting around — a gaming PC, a couple of laptops, an old
workstation — and pools it into **one resource you can deploy work onto**. It runs in two
postures from the *same* build:

- **Your own machines (LAN, zero-trust-needed).** Pool computers you own on one network.
  Every node is yours, so instead of adversarial verification we use a lightweight
  reputation + fair-share ledger and get out of your way. This is the default.
- **The global mesh (anyone, anywhere).** Flip one switch (`scale=public`) and MeshPool
  becomes a mesh **strangers can safely share**: each device has a cryptographic identity,
  contributed work runs in a **locked-down sandbox**, results from low-trust nodes are
  **verified against a local oracle**, and an **earn-to-spend credit ledger** with a
  starter grant lets anyone donate spare compute or borrow the pool's — with no signup.
  Agents reach the coordinator over an **outbound tunnel**, so they work behind home
  routers and firewalls with nothing to port-forward.

Three things it does today:

| | Engine | What you get |
|---|---|---|
| 🌐 | **Hosting** | Deploy an app across N machines behind a load balancer. A machine sleeps → traffic re-routes automatically; **no requests dropped beyond those already in flight** (new requests transparently retry a survivor). |
| ⚙️ | **Batch** | Submit a job of independent work units; it fans out across every machine's spare CPU. Kill a node mid-job → its units are **retried elsewhere.** |
| 🧠 | **Inference** *(flagship)* | Run a language model whose layers **don't fit on any one machine**, sharded pipeline-parallel across several. Only tiny activation tensors cross the wire. Reports **real tokens/sec.** |

> ### ✨ The magic trick: provable correctness
> Every node reconstructs its slice of the model from the same checkpoint, so the
> distributed pipeline produces logits that are **bit-for-bit identical** to running the
> whole model on one machine — and stay identical **even after a node dies** and the
> model auto-repartitions onto the survivors. Distributing the model changes *where* the
> math happens, never *what it computes.* We prove this in an automated test.

---

## 🚀 Run MeshPool in one command (the shipped build)

Got a MeshPool zip? **Unzip it and run one thing** — no IP, port, or token to type. It
installs what it needs on first run, starts a control plane **and** contributes this
machine, then opens the dashboard.

```powershell
# Windows — double-click start.cmd, or:
.\start.ps1
```
```bash
# macOS / Linux
./start.sh
```

Your browser opens at `http://127.0.0.1:8080` with **Use · Contribute · Wallet · Explore**.
Submit a batch job, deploy an app, or shard a model across your machine(s) — right away.
Add more computers to *your* pool with the pairing command shown under **Contribute**.

**Verify the install works end-to-end** (brings up the stack, runs a real job, tears down):

```bash
python -m meshpool.launcher self-test        # prints SELFTEST_OK
```

### 🌍 Contribute to (or use) a global mesh

Point a machine at a **public coordinator** and it donates spare compute over an outbound
tunnel — nothing to port-forward, works behind any home router:

```powershell
.\contribute.ps1 -Cloud https://your-coordinator.example   # Windows
```
```bash
./contribute.sh --cloud https://your-coordinator.example   # macOS / Linux
```

Anyone reaching that coordinator's dashboard can **use** the pooled compute with a
zero-signup local sign-in and a starter credit grant; every job debits the consumer and
credits the machines that did the work. Run your own coordinator with
`scale=public` (see [`docs/architecture.md`](docs/architecture.md)); a hosted
**MeshPool Cloud** is the roadmap goal.

> **Build a shareable zip** to hand to others: `powershell -File deploy\build_zip.ps1`
> (or `./deploy/build_zip.sh`) → `dist/meshpool-<version>.zip`. It contains only what's
> needed to run — never your device keys or local state.

---

## 🧪 Quickstart — simulate a whole cluster on one laptop (dev)

**Requirements:** Python 3.11+ with `numpy fastapi uvicorn pydantic httpx typer rich pyyaml cryptography aioquic`
(one `pip install -r requirements.txt`; all pure-Python + NumPy, no native toolchain). No Docker, no GPU, no cloud account.

```bash
# 1. Boot a whole cluster on your laptop: 1 control plane + 3 simulated machines
#    (heterogeneous CPU/RAM/compute, so scheduling & sharding are non-trivial)
python -m sim.harness --agents 3

# 2. Open the dashboard
#    → http://127.0.0.1:8080
```

You'll see three machines register live, with their capabilities, donation caps, health,
trust and contribution stats. From here, drive it with the CLI (new terminal):

```bash
# point the CLI at the control plane — env var or the global --control-plane flag:
set MESHPOOL_CONTROL_PLANE_URL=http://127.0.0.1:8080   # PowerShell: $env:MESHPOOL_...
# or per-command: meshpool --control-plane http://127.0.0.1:8080 nodes

meshpool nodes                       # list the live pool
meshpool jobs submit --count 30      # batch job across all 3 machines
meshpool deploy hello --replicas 2   # host an app with load-balancing + failover
meshpool models deploy nano-gpt      # shard a model too big for one node
meshpool chat "MeshPool lets you"    # talk to the distributed model, see tokens/sec
```

*(No install step? Run the CLI in-tree with `python -m meshpool.cli.main <cmd>`.)*

Prefer to prove it end-to-end without touching anything? Run the acceptance scenarios —
each one boots a real multi-process cluster and checks a hard invariant:

```bash
python -m sim.scenarios.verify_batch       # (d) distribute + retry-on-kill
python -m sim.scenarios.verify_hosting      # (c) deploy + failover, no new requests dropped
python -m sim.scenarios.verify_inference    # (e) shard + serve + bit-identical + survive-a-death
python -m sim.scenarios.verify_lan          # real multi-address path: work reaches each node's advertised IP
```

---

## 🌐 Run it live across your real machines

The Quickstart above simulates a cluster on one laptop. To run MeshPool **for real** across
several machines on the same network — host apps, run batch jobs, and serve a sharded model,
live — copy this folder to each machine and run **one command per machine**.

**1. On ONE machine — start the control plane** (hosts the dashboard, coordinates the pool):

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File deploy\start-controlplane.ps1 -OpenFirewall
```
```bash
# macOS / Linux
bash deploy/start-controlplane.sh
```

It auto-detects this machine's LAN IP, then prints the dashboard URL and the **exact join
command** to run on every other laptop, e.g.:

```
Dashboard   : http://192.168.1.20:8080
Windows     : powershell -ExecutionPolicy Bypass -File deploy\join.ps1 -Server http://192.168.1.20:8080 -Token 3f9c...
macOS/Linux : bash deploy/join.sh --server http://192.168.1.20:8080 --token 3f9c...
```

**2. On EACH other laptop — run that printed join command:**

```powershell
# Windows (donate 40% CPU, open the firewall port once)
powershell -ExecutionPolicy Bypass -File deploy\join.ps1 -Server http://192.168.1.20:8080 -Token 3f9c... -Cpu 40 -OpenFirewall
```
```bash
# macOS / Linux
bash deploy/join.sh --server http://192.168.1.20:8080 --token 3f9c... --cpu 40
```

Each agent auto-detects its **own** LAN IP, probes its **real** hardware (CPU / RAM / GPU / FLOPS),
sizes its RAM donation to the machine, and self-registers. Leave it running.

**3. Open `http://<control-plane-IP>:8080`** — your real machines appear live. Deploy an app,
submit a batch job, or shard a model across them from the dashboard or CLI, exactly like the
Quickstart — except the work now runs on your actual hardware.

> **Serving a real LLM across two machines?** If each machine runs [Ollama](https://ollama.com)
> with the same model pulled (e.g. `ollama pull qwen2.5-coder:7b`), MeshPool pools them into one
> endpoint that **load-balances and fails over** across both. Follow the step-by-step
> **[two-machine test runbook](docs/two-machine-test.md)** — prerequisites, deploy, chat,
> a throughput **benchmark**, and a **failover** demo. Ideal for handing to a second person with the zip.

> **Firewall (Windows):** the control-plane and agent ports must be reachable across the LAN.
> The launchers' `-OpenFirewall` flag adds the inbound rule for you (one elevated prompt);
> otherwise pick **Allow** when Windows prompts, on **Private** networks. macOS/Linux home
> firewalls usually allow LAN inbound already. Full troubleshooting → [`deploy/README.md`](deploy/README.md).

This exact path — control plane and agents on **distinct addresses**, every job reaching each
node at the address it **advertised** (never a hardcoded localhost) — is locked down by an
automated acceptance test (`sim/scenarios/verify_lan.py`, wired into `tests/test_e2e.py`).

---

## 🎯 What "done" means (MVP acceptance criteria)

A household with 3 machines (or the sim harness emulating them) can:

| | Criterion | Proven by |
|---|---|---|
| (a) | Install the agent on all three in **< 5 min**, no manual networking | pairing-code onboarding + install scripts in [`/deploy`](deploy) |
| (b) | See them **live in a dashboard** | `python -m sim.harness` → dashboard node list |
| (c) | Deploy a web app that **survives a machine sleeping** | `verify_hosting.py` — kill a node mid-traffic, no new requests dropped during failover |
| (d) | Run a batch job using **all three machines' spare CPU** | `verify_batch.py` — distribution + kill → retry elsewhere |
| (e) | Query a model that **won't fit on any single machine** | `verify_inference.py` — shard, serve, measure tok/s |

Criteria **(b)–(e) are automated, re-runnable tests** (`tests/test_e2e.py` → `sim/scenarios/*`),
not screenshots; **(a)** is delivered by the one-command install scripts in [`/deploy`](deploy)
(onboarding time is measured, not asserted in CI).

---

## 🏗️ Architecture

```
 [Node Agent]   [Node Agent]   [Node Agent]      one per machine
      \              |              /             (capability probe, donation caps,
       \--- LAN discovery + rendezvous ---/        sandboxed exec, kill-switch)
                     |
              [ Control Plane ]
   node registry │ 3 schedulers │ trust + contribution ledger │ REST + SSE API
                     |
   ┌─────────────────┼──────────────────┬──────────────────┐
   │                 │                  │                  │
[Batch Engine]  [Hosting Engine]  [Inference Engine]   [Dashboard + CLI]
 spread            bin-packing       topology-aware
 dispatch         + load balancer   layer sharding
```

- **Node Agent** — probes CPU/RAM/GPU/bandwidth, enforces the owner's donation policy at
  the OS/runtime level (the workload is *never* trusted to self-limit), heartbeats health,
  and offers an **instant kill-switch** that reclaims the machine immediately.
- **Control Plane** — node registry + three schedulers (batch **spread / worst-fit** for
  even fan-out, hosting **bin-packing** with anti-affinity, inference **topology-aware
  partitioning**) + the trust/contribution ledger.
- **Three engines**, built in dependency order: Batch (the walking skeleton) → Hosting →
  Inference (flagship).
- **Trust** — Tier 0 (single-owner, full mutual trust) and **Tier 2 (global mesh:
  device identity + sandbox + verification + earn-to-spend credits)** are both active;
  Tier 1 (office/enterprise) sits between them. Switching posture is a **config flip
  (`scale=home|office|enterprise|public`), not a rewrite.**

See [`docs/architecture.md`](docs/architecture.md) for the full component tour, and
[`proto/meshpool.proto`](proto/meshpool.proto) for the wire contracts.

---

## 🧠 How distributed inference works

1. **Partition** — the topology-aware scheduler probes each node's usable memory and
   compute, then greedily assigns each node a **contiguous block of decoder layers** (a
   pipeline stage), leaving headroom for activations. More compute → more layers.
2. **Load** — each node loads *only its slice* of the model. The first stage owns the token
   embedding; the last stage owns the LM head.
3. **Serve** — to generate a token, the control plane sends the prompt to stage 0, which
   passes its **hidden-state activations** (a few KB) to stage 1, and so on. Only
   activations cross the network — never weights or gradients — which is exactly why a LAN
   is fast enough.
4. **Heal** — if a node drops, the model **auto-repartitions** across the survivors (if the
   remaining memory still fits it) and keeps serving. Output stays identical.

The default engine is a compact from-scratch NumPy GPT (so it runs anywhere with zero
downloads), sitting behind an **adapter interface** so a real Petals/exo/torch backend can
be dropped in for genuinely large models. The M1 first-token gate also includes a
Qwen2/Llama-class NumPy path (RMSNorm, RoPE, SwiGLU MLP and grouped-query attention)
driven by a real downloaded HuggingFace `config.json` + `tokenizer.json`. Its weights are
locally initialized because weight downloads are blocked in this environment; this
validates the real-token distributed execution path, not pretrained model quality. It
speaks an **OpenAI-compatible Chat Completions API**, so your existing tools work
unchanged:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "nano-gpt",
  "messages": [{"role": "user", "content": "MeshPool is"}],
  "temperature": 0.7
}'
```

Details: [`docs/inference.md`](docs/inference.md).

M1 gate:

```bash
python sim\scenarios\verify_first_real_token.py
```

It starts three separate node processes, loads contiguous layer/head ranges, applies the
WAN emulator, streams token ids, and asserts the sharded output is token-identical to the
single-process NumPy model at temperature 0. The gate also prints the process IDs,
per-stage timings, emulated-link bytes/latency, and a deliberate corruption check proving
the equality assertion fails when a stage output is perturbed.

---

## 🖥️ CLI reference

```
meshpool [--control-plane URL]        global flag: overrides MESHPOOL_CONTROL_PLANE_URL env var

meshpool nodes                     list live nodes + capabilities + health
meshpool onboarding                show the pairing code / join instructions
meshpool join <control-plane-url>  join this machine to a pool

meshpool jobs submit [func]        run a batch job (--count N units)
meshpool jobs list                 batch jobs + progress

meshpool dag submit DAG.json       submit a JSON DAG with stage dependencies
meshpool dag list                  DAG workflows + progress
meshpool dag show <dag-id>         show one DAG's stage state

meshpool deploy [app] --replicas N host an app across the pool
meshpool deployments               deployments + placement + health

meshpool models deploy [model]     shard a model across the pool
meshpool models list               deployed models + pipeline stages
meshpool chat "<prompt>"           chat with the distributed model (shows tok/s)

meshpool ledger                    contribution ledger (donated vs consumed)
meshpool fn deploy ./fn.wasm       upload your own code, get a wasm:<sha256> name
meshpool fn run wasm:<sha> -p '{}' fan your function out over the pool and wait
meshpool fn builtin                list the functions that ship with MeshPool
meshpool storage put <file>        encrypt + replicate a file across the pool
meshpool storage ls                list stored files (ciphertext blobs)
meshpool storage get <id> <out>    download + decrypt a file
meshpool storage rm <id>           delete a stored file from the pool
meshpool secrets put NAME VALUE    store a scoped secret (value kept in memory)
meshpool secrets list              list secret names (values never shown)
meshpool secrets rm NAME           delete a stored secret
meshpool reclaim <node-id>         kill-switch: drain + reclaim a machine now
meshpool resume  <node-id>         bring a reclaimed machine back
meshpool up --agents N             boot a local sim cluster (wraps sim.harness)
```

Or skip the CLI entirely with the one-command launcher (ships in the build):

```
python -m meshpool.launcher solo               run + use MeshPool on this machine
python -m meshpool.launcher contribute --cloud <url>   donate compute to a mesh
python -m meshpool.launcher self-test          verify the install works end-to-end
```

---

## 🧩 Mesh Functions — run *your* code on the pool

Anything that compiles to `wasm32-wasip1` runs on every machine in the mesh. The
contract is deliberately tiny: **read a JSON payload from stdin, write a JSON result to
stdout.**

```rust
// src/main.rs  —  cargo build --release --target wasm32-wasip1
use std::io::{Read, Write};
fn main() {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).unwrap();
    let v: serde_json::Value = serde_json::from_str(&s).unwrap();
    let x = v["x"].as_i64().unwrap_or(0);
    let out = serde_json::json!({ "tripled": x * 3 });
    std::io::stdout().write_all(out.to_string().as_bytes()).unwrap();
}
```

```bash
# 1. Upload it. Validated here, while you are watching — not on a donor's laptop.
meshpool fn deploy ./my_fn.wasm
# -> wasm:fa47a02f...   (the SHA-256 of the bytes you uploaded)

# 2. Fan it out over the whole pool and wait for the results.
meshpool fn run wasm:fa47a02f... --payload '{"x":7}' --count 24
```

…or over raw HTTP, which is all the CLI and the console are doing:

```bash
curl -X POST --data-binary @my_fn.wasm http://127.0.0.1:8080/api/v1/functions
# -> {"function":"wasm:fa47a02f...","abi":"wasi-command/json-stdio", ...}

curl -X POST http://127.0.0.1:8080/api/v1/jobs -H 'content-type: application/json' \
  -d '{"name":"triple","func":"wasm:fa47a02f...","payloads":[{"x":1},{"x":2},{"x":3}]}'
```

…or by dragging the `.wasm` onto **Functions** in the console (`/console`), which shows
each unit's result next to the machine that produced it.

What the pool guarantees while your code runs on someone else's machine:

| Guarantee | How |
| --- | --- |
| It can't read their files | No directory is preopened — WASI is deny-by-default, `path_open` returns `EBADF` |
| It can't reach the network | No sockets are linked in; a module importing anything outside WASI p1 is **rejected at upload** |
| It can't hang their machine | Fuel budget + epoch-interrupt watchdog + memory ceiling |
| It can't take the node down | Runs in a dedicated child process, so even a hard abort costs one unit, not the node |
| It runs the code you uploaded | Content-addressed: every node re-verifies the module against its SHA-256 before executing |
| Wrong answers don't get paid | Sampled units are re-run by the coordinator against its own copy of your module |

Same fan-out, failover, credits and trust machinery as the built-in functions —
`wasm:<sha256>` is just another function name.

---

## 🖼️ The console

Two web surfaces ship, and they are independent on purpose:

| Surface | What it is |
| --- | --- |
| `/` | The zero-dependency dashboard. Plain HTML/JS served by the control plane; always available, needs no Node toolchain. |
| `/console` | The React console — playground-first, ⌘K palette, live mesh map, Functions upload-and-run, dark mode. Optional build artefact. |

```bash
npm --prefix web install
npm --prefix web run build      # emits web/dist, which the control plane then serves
npm --prefix web test           # component tests
```

The console lands you on a **prompt box against a pooled model**, not a dashboard and
not a wizard. If `web/dist` has not been built, `/console` returns that build hint
instead of a 404, and `/` keeps working untouched.

---

## 🧪 Testing

```bash
pip install pytest
pytest                 # fast unit tests (model, sharding oracle, gradients, schedulers, trust, config)
pytest -m slow         # end-to-end acceptance gates (boots real clusters — slower)
pytest -m ""           # everything
```

The unit suite includes the **shard-equals-full-model correctness oracle** and a
**float64 finite-difference gradient check** of the hand-written backprop. See
[`docs/testing.md`](docs/testing.md).

---

## ⚖️ Honest constraints (we report real numbers, never invented ones)

- **Auto-update is fully implemented and completely inert, on purpose.**
  `agent-go/internal/update` verifies Ed25519-signed release manifests, refuses
  downgrades, and swaps the binary through a rename dance that restores the previous
  build if the swap fails — 42 tests, including four attack cases and the two
  recovery branches. It cannot update anything, because **no MeshPool release signing
  key exists yet**, so the compiled-in trust store is empty and `Verify` returns
  `ErrUnknownKey` for every manifest. That is deliberate: the two alternatives are a
  placeholder key nobody holds the private half of (a lie that looks like security,
  and one that survives review precisely because the surrounding code is correct), or
  "no keys configured, so skip verification" — which turns the absence of security
  into the absence of a check and makes the update channel a fleet-wide RCE
  primitive. What is **not** built: release signing infrastructure, installers,
  service registration, and uninstall. Generating the key is a release-engineering
  task, not a code change, and the empty store is pinned by a test that will fail
  loudly the day it stops being empty. Two findings here came from mutation testing
  rather than review, and both are recorded in the meta-gate: the rollback branch was
  unreachable by any portable test (a directory blocking the destination just gets
  renamed aside too, so the branch never ran), and the download size-bound test
  *survived deleting the bound* — twice, in two different files — because asserting
  that an oversized body is rejected proves nothing when the length check fires
  anyway, just after the bytes have arrived.
- **Nodes are classified automatically, and the classification only ever expresses a
  preference.** `meshpool/controlplane/nodeclass.py` derives one of four classes --
  Anchor, Relay, Constellation, Drifter -- from signals that were actually measured:
  NAT class and battery state from the hardware probe, uptime and latency consistency
  from the trust engine. Placement uses it to keep Drifter-class machines out of
  inference pipeline stages, because a pipeline's availability is the *product* of its
  stages' and one machine that unplugs at 6pm degrades every request rather than its
  own share of them. Two limits are deliberate and worth stating. First, it is a
  preference and never a prohibition: if dropping the volatile node would make the
  model unhostable, the node is used, because refusing to place at all is a worse
  outcome than placing on a machine that might leave. Second, promotion requires 50
  recorded observations, since every reputation EMA *defaults to 1.0* -- a node with
  no history looks flawless, and reading that as evidence would hand a stranger's
  first heartbeat an Anchor role. With no trust history at all (LAN mode, or a freshly
  restarted coordinator) placement is byte-for-byte what it was before. What is **not**
  claimed: that the classes predict departure well. The thresholds are reasoned, not
  fitted to observed churn data, because no such data exists on a single-machine
  test bench.
- **Real GGUF checkpoints load and shard correctly — but no pretrained weights were ever
  downloaded on this machine.** `meshpool/inference/gguf_loader.py` reads real GGUF
  containers through llama.cpp's own reference reader, loads *only* an assigned layer
  range via `np.memmap`, inverts llama.cpp's RoPE permutation, and dequantizes
  Q8_0/Q5_1/Q4_0/F16. `sim/scenarios/verify_gguf_inference.py` proves the load-only-my-
  layers property empirically rather than by assertion: it overwrites the byte ranges of
  every layer a stage was *not* assigned and demands bit-identical weights, with a
  companion check confirming that corrupting *all* layers does change them (so the first
  check cannot pass vacuously). What is **not** proven: model quality. Corporate TLS
  interception blocks HuggingFace model downloads here (`CRYPT_E_NO_REVOCATION_CHECK`),
  so every checkpoint under test has a real format and deterministically generated
  weights. The format, the sharding and the arithmetic are measured; "this produces good
  text" is not, and is not claimed. K-quants (Q4_K, Q6_K, Q2_K, Q3_K, Q5_K) are supported
  through the reference dequantizer but are **untested locally** — the reference `gguf`
  package can decode them but cannot encode them, so no local fixture is possible.
- **The tokenizer is verified against SentencePiece, and deliberately disagrees with it in
  two places.** `gguf_tokenizer.py` implements llama.cpp's SPM bigram-merge algorithm and
  is checked token-for-token against the real `sentencepiece` library — a round-trip test
  would prove nothing, since a tokenizer can be self-consistently wrong and still
  round-trip perfectly while emitting ids the model never saw. Two intentional
  divergences: MeshPool does **not** apply NFKC normalization and does **not** delete
  control characters, matching llama.cpp (which is what real GGUF checkpoints are served
  by) rather than SentencePiece's default `nmt_nfkc` normalizer. Both are pinned by test
  so they cannot be silently "fixed" into bugs. Byte-level BPE vocabularies
  (`tokenizer.ggml.model == "gpt2"`) are **refused with an error**, not approximated.
- **Global transport is now QUIC-first, with the original outbound HTTP tunnel as a
  first-class fallback.** When `MESHPOOL_TRANSPORT=quic`, the coordinator starts an
  aioquic UDP listener and agents that support it (the Go `meshd` agent) authenticate
  the QUIC session with the same persisted Ed25519 device key used for enrollment.
  Control-plane calls and proxy requests then run on independent QUIC streams, so a
  slow/lost stream does not head-of-line block unrelated work. The old
  `GET /tunnel/stream` + `POST /tunnel/reply` path is still present and is deliberately
  preferred when QUIC is blocked or absent; relayed/tunnel nodes are not second-class.
  Local gates measure connection migration and fallback under emulation only:
  [`verify_quic_migration.py`](sim/scenarios/verify_quic_migration.py) forces a UDP
  address change mid-request and requires the request to complete, while
  [`verify_transport_fallback.py`](sim/scenarios/verify_transport_fallback.py) configures
  QUIC but proves a tunnel-only node still receives work. These are not real-internet
  success-rate claims; the product still budgets for permanent relay dependency on
  CGNAT/symmetric-NAT paths.
- **NAT traversal is a measured classification plus relay plan, not a promise to punch
  every home router.** MeshPool classifies peers from STUN-like observations
  (reflexive mappings across reflectors and alternate-source replies), never from a
  user-entered NAT type. The hard floor is explicit: symmetric-NAT ↔ symmetric-NAT
  hole punching is treated as **0%** and goes relay-first with no stall; mixed NATs use
  a bounded attempt budget and a 22 s keepalive to match 15-30 s UDP binding timeouts.
  Relay selection is lowest-RTT among public relays, and relay users remain eligible for
  work and credit. Planning uses libp2p DCUtR's ≈73% mixed-NAT success as a ceiling and
  budgets **25-40%** permanent relay dependency for residential broadband, not the much
  lower DERP rate reported by highly technical user bases.
- **Membership is SWIM-style gossip under emulated churn, not a Raft/etcd cluster.**
  The registry now stores nodes in id-sharded maps with URL/device/status indexes so
  hot-path registration, heartbeat, token and online lookups do not enumerate every
  node. The membership engine uses direct probes, indirect probes through k peers, and
  suspicion before death; its default probe cadence reuses the 22 s NAT keepalive budget
  so residential UDP bindings are not silently allowed to expire. The
  [`verify_churn_membership.py`](sim/scenarios/verify_churn_membership.py) gate injects
  30%+ churn and proves convergence in deterministic one-machine emulation, plus proves
  the registry lookup path is indexed. It is not a real 10k-node internet deployment.
- **Constellations are soft RTT-derived clusters, not user-configured regions.**
  The scheduler forms low-latency neighbourhoods from measured Vivaldi/RTT predictions
  using the plan's <25 ms threshold, advertises pooled RAM/VRAM/usable-memory capacity,
  and does O(1) capacity lookup from the pre-clustered index. Pipeline placement prefers
  minimizing `pipeline_depth × RTT`, so a slower metro can beat faster scattered nodes
  when the RTT term dominates. The
  [`verify_constellation_formation.py`](sim/scenarios/verify_constellation_formation.py)
  gate injects known metro/scattered topologies and verifies clustering, capacity fit,
  churn re-formation, and degenerate 2-node/all-scattered cases under emulation only.
  It is not a public-internet or throughput measurement.
- **The S3 SigV4 signature binds a *claimed* payload digest, not the bytes we received.**
  `x-amz-content-sha256` is read from the header and used verbatim in the canonical
  request; the body is never hashed and compared. So a captured, signed `PUT` can be
  replayed with substituted content and the signature still verifies. This is a
  deliberate trade — verifying it requires buffering the whole object in memory before
  the first byte reaches storage, which defeats streaming uploads — and AWS itself
  permits `UNSIGNED-PAYLOAD` over HTTPS for the same reason. What *is* enforced: a
  15-minute clock-skew window (`SIGV4_MAX_SKEW_S`), so the replay window is bounded
  rather than infinite, and a timestamp that cannot be parsed is refused rather than
  waved through. Run MeshPool's S3 endpoint over TLS; without it the signature is not
  protecting object contents.
- **S3 access keys are stored in recoverable form, because SigV4 requires it.** The
  server has to recompute the client's signature, so unlike a password there is no
  hash-and-compare option. Keys live in the control-plane database, inside the same
  trust boundary that already holds session tokens and the ledger; they are durable
  across restarts (an earlier in-memory dict silently invalidated every client on every
  restart, reporting `SignatureDoesNotMatch` and blaming the client) and revocable via
  `DELETE /api/v1/s3/credentials/{access_key}`.
- **Hard spend caps stop work — and everything else about billing does not exist yet.**
  A cap set via `PUT /api/v1/account/spend-cap` genuinely halts execution: new work is
  refused with HTTP 402 at all twelve work-creating endpoints, and the batch dispatcher
  re-checks before handing out each further unit, so a job already running is stopped
  mid-flight rather than allowed to overrun. Units already in flight are permitted to
  finish. Proven end-to-end by `sim/scenarios/verify_spend_cap_hard_stop.py`, which
  includes a non-vacuity control showing the identical job completes when uncapped.
  What does **not** exist: real money, payouts, invoices, or fiat rails. The cap is
  denominated in internal compute-credits. Owners are **uncapped by default** — set
  `MESHPOOL_DEFAULT_SPEND_CAP` to change that. The console reports `enforced: false`
  when no cap is set rather than displaying a number nothing applies.
- **Public-mesh trust is fraud detection, staking and economics — not cryptographic
  proof.** MeshPool does **not** use Intel SGX / consumer TEEs as a root of trust
  (known TEE break classes include Foreshadow/SGAxe/global-TCB recovery and real key
  extraction incidents), and it does **not** claim zkML/zkVM proofs for LLM inference
  (current overhead and non-deterministic floating-point/GPU sampling make that
  impractical). What is implemented is proportional detection: deterministic work can
  be redundantly spot-checked; LLM-serving should use top-logit commitments plus
  reference re-execution as detection, not proof; training/serving disputes localize
  disagreements with an O(log N) Truebit-style bisection helper. Tier-2 reputation is a
  decaying Beta posterior weighted by verified work value, so old behaviour fades and
  cheap micro-jobs do not buy expensive-job trust. New nodes face graduated stake
  requirements near 20× expected job value; detected fraud slashes stake into a real
  destination ledger account, preserving donated==consumed conservation. Contributor
  earnings also have Storj-style held-back escrow (50% initially, 25% after three
  months, 0% after nine months). Sybil resistance remains no-KYC: Ed25519 device
  enrollment can be bound to a serial hardware-timing challenge. That raises the cost
  of mass fake identities; it cannot prove a unique human or make Sybils impossible.
- **The current fraud-detection release gate is logic-level, not full transport E2E.**
  `sim/scenarios/verify_fraud_detection.py` deliberately runs in process while driving
  the real `BatchManager`, `Verifier`, `TrustEngine`, `ContributionLedger`, SQLite
  `Store`, and deterministic function runner. It proves the detection/slashing logic
  and non-vacuity controls, but it does **not** yet prove that a real multi-process
  agent returning a bad answer over the production transport is caught and slashed.
  Before anonymous public supply opens, this must be complemented by a multi-process
  gate using real agents and the real transport.
- **Metering has one unit: the measured mesh-second.** Batch, functions, inference,
  hosting, and storage all normalize resource-weighted node time into the same
  `mesh-second`; receipts carry `pricing_dimensions: 1` so a second billing axis
  cannot be hidden in the API. The production receipt call site is currently batch
  accepted-unit billing; the other service classes are defined in the meter but not
  yet wired to production receipts. Per-job receipts show gross contributor earnings,
  the platform fee as a separate line item, and the total charged to the consumer;
  fees are never silently netted out of contributor earnings. The meter labels
  observed runtime/cost as **measured** and live forward-looking fields as
  **calibrated projection**. What is not implemented yet: fiat invoices, payouts,
  taxes, regional price books, or a guarantee that today's in-process metering gate
  proves every future service adapter's production transport path.
- **Mesh Apps now roll forward, but not into managed edge hosting.** App deploys
  auto-detect common Vite/React, Node, FastAPI and Flask layouts, write the detected
  plan to `meshpool.json`, and blue/green rollouts keep the old version serving until
  the new one is health-checked. Routing uses only healthy replicas, and autoscale
  decisions come from observed proxy traffic. What is not implemented here: managed
  TLS certificates, DNS automation, global edge POPs, or custom-domain provisioning.
- **Repair is peer-to-peer, and the coordinator relay survives as a deliberate
  fallback.** Repair is the *steady-state* cost of a churning pool — one per object per
  departure — so routing it through the control plane makes the coordinator a hard
  bandwidth and memory bottleneck that every departure queues behind. Both repair paths
  now send a **byte-free plan** naming peer URLs: replication repair tells the new
  holder where to fetch, and erasure repair tells one node to pull `k` shards directly
  from its peers, reconstruct locally, and keep the missing ones. Measured before the
  fix at RS(3,2): four `blob_get` calls dragged through the coordinator to replace a
  single shard; at the cold profile RS(29,80) that is 29+ shards through one uplink with
  the whole reconstructed object held in coordinator RAM. The relay path is still there
  and still runs — for holders that are not directly addressable (**the CGNAT case,
  budgeted at 25–40% of the pool**) and for nodes that refuse the delegated repair —
  because peer-to-peer is an optimisation of the *route* and must never become a reason
  to leave an object under-replicated. Note what this does **not** claim: nothing about
  correctness can distinguish the two paths, since the relay reconstructs perfectly and
  every durability check passes either way. The guarantee is enforced by tests that
  count *who carried the bytes*, plus mutations that restore the relay and must fail.

- **Delegated repair moved verification along with the work.** Reconstruction now
  happens on a machine the coordinator does not control, from shards served by machines
  it also does not control. Every fetched shard is checked against its content address,
  and every reconstructed shard against the id that was actually requested. That second
  check is not redundant: a content address proves a shard is the bytes it claims to be,
  but proves **nothing** about which encoding it came from, so shards from two
  generations of an object (stale manifest, interrupted re-encode, a peer that never
  deleted an old version) each verify individually and decode to garbage.

- **Capability tokens are scoped, expiring and revocable — and cannot escalate.**
  An account can mint narrow bearer tokens for CI and agents instead of sharing its
  session. Requested scopes must be a subset of what the calling principal already
  holds, so a token minted by a token can only ever narrow. Only a SHA-256 hash of the
  secret is stored, so listing cannot leak it. Revocation is a database read on every
  request rather than a cached decision, so it takes effect on the immediately following
  call. Usage is attributed to the **grantor**, not the bearer: a forged owner in a
  request body is overridden, and the token id lands on the job, the receipt and the
  ledger line. What this is *not*: these are opaque bearer secrets, not signed
  macaroons, so a token cannot be attenuated offline and revocation depends on the
  control plane being reachable.

- **Proof-of-storage audits are per-chunk hashes, not a Merkle tree or PoRep.** Nodes are
  challenged on a random chunk offset and must return its SHA-256. A node that deleted the
  bytes fails. A determined cheater *can* defeat this by caching the chunk hash table
  (~8 KB per 1 MB shard) instead of the data; the defence is economic (escrow forfeit
  exceeds the storage saved), not cryptographic. Audit material lives in memory and is
  rebuilt after a restart by re-fetching shards, which is safe because shard ids are
  content addresses. Crucially, a shard whose material cannot be rebuilt is reported as
  **skipped, never as passed** — a blinded auditor must not look like a healthy one.

- **Single-machine training today; not *distributed* training.** Local **LoRA fine-tuning**
  of the nano-GPT over a frozen base ships and is gradient-checked
  (`python -m meshpool.inference.lora`); *distributed* training over the pool stays deferred —
  it's bandwidth/latency-bound on heterogeneous consumer networks.
- **Network latency is a hard floor.** Throughput-bound, decomposable work scales across a
  pool; tightly-coupled sub-millisecond-shared-state work does not. Every feature is
  designed with that table in view.
- **CPU/Wi-Fi inference is modest by design** — the point is *"a model that wouldn't fit at
  all now runs,"* not "faster than a datacenter GPU." The adapter seam is where real
  GPUs/Petals/exo take over for scale.
- **Distributed inference demo — honest scope.** `verify_first_real_token.py` runs a real
  Qwen2-class architecture (RMSNorm, RoPE, SwiGLU, grouped-query attention), driven by a
  real HuggingFace tokenizer and config downloaded at runtime, split across three genuinely
  separate OS processes, producing output that is bit-identical to single-process execution.
  What is limited: the model is `yujiepan/qwen2-tiny-random` with `d_model=8` and only 2
  transformer layers, so the 3-way pipeline split degenerates — the last process holds zero
  transformer layers and is head-only. Weights are locally initialized because HuggingFace
  binary weight downloads are 503-blocked in this environment. Scaling to a real model is
  currently blocked by the transport: shards receive weights as base64-encoded JSON in the
  same request body, which is impractical for any realistic checkpoint. This validates the
  distributed execution path, tokenization, sharding, and transport — not model quality or
  practical scale.
- **GGUF RoPE conventions are mirrored from llama.cpp, and that is a standing liability.**
  llama.cpp stores Q/K in one of two layouts and picks which by architecture alone —
  `llama_model_rope_type()` switches on `model->arch` and nothing else, and **there is no
  GGUF metadata field that records the layout**. A loader therefore has no choice but to
  transcribe that switch. Ours does, programmatically rather than by hand, and an
  architecture in neither list raises rather than defaulting to a guess. This matters
  because getting an entry wrong does not crash: the model loads, generates fluently, and
  is quietly wrong. An earlier hand-written allowlist had 4 of its 5 entries wrong,
  including omitting `deepseek2` — the architecture behind DeepSeek-V3 and Kimi K2. Note
  also that "llama.cpp rotates this architecture as NORM" is *not* the same claim as "the
  converter permuted its Q/K weights"; the two coincide for ordinary attention and diverge
  for MLA models, whose tensor layout this loader does not implement at all.
- **Client-side encryption — and its real limitation.** Encryption happens in the client
  library before bytes leave the process. The server **cannot enforce it**: the raw
  `POST /api/v1/blobs` API accepts plaintext unconditionally — a caller who bypasses the
  client library gets no confidentiality guarantee. This is a genuine limitation, not a
  footnote. Key hierarchy: a local user root key wraps per-bucket keys, which wrap
  per-object data keys; all wrapping uses AES-256-GCM from `cryptography`. Verified by
  `sim/scenarios/verify_client_side_encryption.py`, which proves (1) the marker is absent
  from every agent disk and control-plane download after a client-encrypted store, and
  (2) a caller who posts plaintext directly to the API can read it back unencrypted.
- **Encrypted objects do not deduplicate.** Each object gets a fresh random data key and
  nonce, so two uploads of the same file produce different ciphertext blobs. Convergent
  encryption would preserve deduplication but would let the control plane or a node
  operator confirm whether two users hold identical files (a confirmation-of-file attack).
  Random keys were chosen: confidentiality by default, deduplication never.
- **Mesh Functions are WASI preview 1, not a full OS.** A user module gets compute, its
  JSON input and its JSON output — deliberately no filesystem, no sockets, no threads and
  no host FFI. That is what makes it safe to run on a stranger's laptop, and it is also a
  real limit: code needing native libraries or network access is a container workload, not
  a WASM one. Metering note: fuel counts *guest* instructions, so byte-shuffling done
  inside host calls is billed via output size and wall-clock rather than fuel.
- **Speculative Pipeline Decoding is a *latency* amortiser, and it costs you when there
  is no latency.** SPD verifies K+1 tokens in one pipeline traversal, so its ceiling is
  exactly the traversal ratio and its benefit is a function of how much of a token's cost
  is `(N−1)·RTT`. Measured on this machine (`verify_spd_equivalence`, 3-stage pipeline,
  32 tokens, real weights, emulated links): **2.2×–2.5× across wide-area topologies —
  and 0.87× with the link latency set to zero**, because the extra per-traversal compute
  is pure loss when there is nothing to hide it behind. Output is token-identical to
  ordinary decoding in every case. Anyone quoting only the good number is quoting half
  the mechanism.
- **Continuous batching only ever saves the network term — and we do not pretend
  otherwise.** Batched rows are computed one at a time against their own KV caches rather
  than stacked into a single matmul, because stacking would change float reduction order
  and a tenant's output would then depend on who happened to share its batch. So the
  compute cost of `B` tenants is still the compute cost of `B` tenants; only the hop is
  shared. On a mesh that is the term worth attacking. Measured on a real 3-process
  cluster (`verify_continuous_batching`, 6 concurrent requests): **288 wide-area hops →
  51** and **1.65×–2.09× aggregate tok/s**, every reply token-identical to the same
  prompt run alone. The hop counts come from `GET /shard/stats` on the agents — counted
  where the packets land, not where they were sent from, so the requester's own
  accounting cannot flatter it. Expect the ratio to fall as soon as compute, not
  latency, is the binding constraint.
- **A truncated answer is now reported as truncated.** `nano-gpt` uses learned absolute
  position embeddings, so the context window cannot slide without invalidating every
  cached key/value; generation stops at the window instead. That was always true — what
  was wrong is that we *told clients it had finished normally*. The OpenAI surface derived
  `finish_reason` from a token count, and a window cut-off lands **below** `max_tokens`,
  so it reported `"stop"`. An unmodified OpenAI client, which continues on `length` and
  stops on `stop`, would silently drop the rest of the reply. The reason is now produced
  by the code that makes the decision (`context_window` on the native API, mapped to
  `length` on the OpenAI one) and is pinned by a gate and by two mutation-catalog entries.
  This was found by an acceptance gate asking for more tokens than could fit — not by
  reading the code, which looked entirely reasonable.
- **Link measurement is real, and it is rate-limited.** Node placement uses measured
  latency and bandwidth, never a number inferred from health-check payload sizes. The
  probe endpoints are necessarily unauthenticated (an agent measures its link before it
  has a credential), so the control plane enforces a per-IP byte budget — which is
  *shared by every node behind one public address*. A node that trips the budget reports
  `bandwidth_mbps: 0`, which means **"not measured"**, not "slow". Two bugs here were
  found by measurement rather than by reading the code: the download ramp was unbounded
  on fast links (401 requests / 3.3 GB from a single agent startup, exhausting the shared
  budget and starving every co-located peer — exactly the CGNAT case the roadmap budgets
  at 25–40% of the fleet), and `meshd` hardcoded `bandwidth_mbps: 0`, so every Go node
  looked like a dial-up link to the scheduler forever. Both are fixed and pinned by
  `tests/test_probe_bandwidth.py` and `agent-go/internal/probe/probe_bandwidth_test.go`.
- **Capability probe fields are richer than the minimum.** Both the Python probe
  (`meshpool/agent/probe.py`) and the Go probe (`agent-go/internal/probe/probe.go`) now
  report `cpu_model`, `gpu_names`, `disk_total_mb`, `thermal_state`, `battery_powered`,
  `battery_percent`, and `network_class`. The scheduler can filter or prefer nodes on any
  of these fields. Nodes registered before these fields were added are unaffected —
  missing fields degrade gracefully to their zero values.
- **DAG / workflow support and checkpoint recovery are shipped.** Submit a JSON DAG with
  stage dependencies, optional reduce stages, and `--wait` polling via
  `meshpool dag submit DAG.json`. The coordinator persists stage state so a
  coordinator restart mid-DAG resumes from completed stages without recomputing them.
  Verified by `sim/scenarios/verify_dag_checkpoint_recovery.py` (checkpoint resume) and
  `verify_dag_worksteal_preemption.py` (work-steal preemption).
- **Per-job distributed trace timelines are live.** Each job accumulates ordered,
  node-attributed attempt spans accessible at `GET /api/v1/jobs/<id>/timeline`. The web
  console's `JobTimelinePage` renders them as a colour-coded timeline with duration and
  outcome per attempt. Verified by `sim/scenarios/verify_job_timeline.py`.

### A note on this build's substrate

The reference spec targets Go + gRPC + Docker + a React dashboard. This repository is a
**faithful, fully-runnable implementation of the same architecture** in a small-dependency
Python/NumPy + vanilla-JS substrate, so every acceptance criterion actually executes here
and now. The mapping (and the production upgrade path for each piece) is documented in
[`docs/architecture.md`](docs/architecture.md#substrate--production-upgrade-path). The
component boundaries, wire contracts, schemas and phased roadmap are identical to the spec.

**Node agent: Go, not Rust.** The single-binary node agent under [`agent-go/`](agent-go/)
is written in **Go**, not Rust as originally planned. This was a forced substitution, not a
preference: the development machine has no Rust toolchain (`cargo`/`rustc` absent) and no
`cmake`, so a Rust agent could not have been compiled or tested here — and shipping
untested code would be worse than shipping a different language. Go gives the same
outcome that mattered (one static binary, no runtime to install, cross-compiles to
Windows/macOS/Linux) and its `wazero` WASM runtime is pure Go with zero CGO. The agent is
verified against the real Python control plane by
[`sim/scenarios/verify_go_agent_parity.py`](sim/scenarios/verify_go_agent_parity.py),
which proves byte-identical device-identity derivation, signed enrollment, heartbeat
liveness, blob round-trips, peer pulls, **execution of `wasm:` user functions** and the
reclaim kill-switch across the two implementations.

**Two WASM runtimes, one answer.** The Python agent runs user functions on `wasmtime`;
`meshd` runs them on `wazero`. A job is fanned out across whichever agents are online, so
what actually has to hold is that both give the *same answer for the same bytes* — if they
diverged, the coordinator's spot-check oracle would start slashing honest nodes.
[`verify_wasm_runtime_parity.py`](sim/scenarios/verify_wasm_runtime_parity.py) feeds
identical modules to both and compares values *and* failure kinds. Two honest divergences
are documented rather than papered over: `wazero` has no fuel meter (Go-side billing falls
back to wall-clock + output bytes, and the result carries `metered: false` so a caller can
tell "no fuel used" from "fuel unknown"), and the two runtimes return different `errno`
values for a refused `path_open` — they agree on the part that matters, which is that it
is refused.

What `meshd` does **not** do yet: host app replicas, serve model shards, or redeem secret
leases. Each returns an explanatory `501` rather than a plausible wrong answer, and curated
built-ins like `square` stay 501 on purpose — two independent implementations of one
function name is exactly how a pool starts disagreeing with itself. The reverse tunnel is a
different story: it is **fully implemented** (`agent-go/internal/tunnel/tunnel.go`) and
verified end-to-end by
[`sim/scenarios/verify_go_tunnel.py`](sim/scenarios/verify_go_tunnel.py), which proves that
a `meshd` node behind CGNAT with no inbound connectivity can register, receive work over the
tunnel, and complete batch jobs and blob transfers. `meshd` now works from any home router
with no port forwarding.

**A node only ever gets work it can run.** Because `meshd` deliberately implements no
curated built-ins, "it returns 501" is not good enough on its own: the scheduler used to
filter candidates on CPU and RAM headroom alone, so a `square` unit could be handed to a
Go node, rejected, and charged as a failed attempt. Five of those in a row fail the unit
outright — on a pool that was perfectly able to run it, and *more* often the more Go nodes
the pool has. Every agent now advertises `capability.runtimes` (`meshd` → `["wasm"]`, the
Python agent → `["builtin", "wasm"]`) and placement filters on it. A pool that cannot serve
a runtime at all fails the job with an explicit reason instead of leaving it pending
forever. Proven end-to-end by
[`verify_runtime_capability_routing.py`](sim/scenarios/verify_runtime_capability_routing.py),
which runs a real mixed pool and includes a positive control showing the same scheduler
*does* select the node once its advertisement is scrubbed back.

The same defect had a sharper form on the **hosting and inference** paths, and it is worth
spelling out because it inverts the usual intuition about adding capacity. Those planners
do not retry elsewhere — they commit to a placement plan — and `topology_partition` counts
a candidate's RAM toward *pooled* capacity. So a `meshd` node in the pool was handed a
layer range (in a measured case, the last stage — the LM head), the deploy then died at
`/shard/load`, **and the Python node alone could have held the whole model.** Adding a
bigger, faster machine broke a deployment that worked before it joined. Capacity has to be
monotonic; that one was not.

The fix is a second, coarser advertisement next to `runtimes`: `capability.services`, the
workload classes a node can serve at all — `batch`, `hosting`, `shards`, `storage`.
`meshd` advertises `["batch", "storage"]`, matching exactly the routes it does not 501 on,
and that correspondence is enforced structurally rather than by hand
(`agent-go/internal/agent/advertisement_test.go` fails in **both** directions — advertising
a service whose route returns 501, and implementing a route without advertising it). The
shard, replica and blob planners all filter on it, and the "insufficient pooled memory"
error now says how many online nodes were excluded, so the diagnostic can never contradict
the planner it is explaining. Both fields default to *everything*, so a node registered
before either existed is unaffected — the advertisement can only ever narrow a node that
narrowed itself.

Two smaller notes from the same work, since they are the kind of thing that silently
rots: the storage tests' node fixtures were `SimpleNamespace` look-alikes carrying only
`id` and `capability.disk_mb`. A fake that cannot answer the questions the real code asks
is not testing the real code — and this one couldn't answer `provides()`. They are real
`Node` objects now. And each new regression test was checked for non-vacuity by running it
against the pre-fix semantics: three of the five scheduler tests fail there, and the two
that pass under both are precisely the positive control and the backwards-compatibility
guard, which is the intended result.

---

## 🗺️ Roadmap

**Delivered** (all gated by automated acceptance tests in `tests/` + `sim/scenarios/`):

- **Global connectivity** — agent-initiated **outbound reverse tunnel** + a deployable
  coordinator, so machines anywhere join through NAT/firewalls with nothing to
  port-forward. Batch, hosting, and sharded inference all run over the tunnel.
- **Device identity** — Ed25519 keypair per device, generated on first run; zero-signup,
  with an optional lightweight account link that pools credits/reputation across machines.
- **Safety** — donated machines run untrusted work under hard limits: a **locked-down
  process sandbox** (memory cap, wall-clock timeout, output cap, input validation) for
  curated functions, and a **WebAssembly sandbox** for user-supplied code.
- **Mesh Functions — run your OWN code** — upload a `wasm32-wasip1` module and it becomes
  a `wasm:<sha256>` function any job can fan out across the pool. Validated at upload
  time, distributed by the same content-addressed store that ships model weights (once
  per node, not once per unit), and executed with **no filesystem, no sockets and no host
  FFI**, under fuel/memory/wall-clock ceilings. Build it in Rust, Go, Zig, C or anything
  else that targets WASI.
- **Credits + trust (Tier 2)** — active earn-to-spend credit ledger with a starter grant;
  reputation EMA + **known-answer spot-checks** that catch a dishonest node and stop paying
  it (user WASM included — the coordinator re-runs sampled units against its own copy of
  the module); per-owner rate limits.
- **Product UX** — redesigned **Use / Contribute / Wallet / Explore** dashboard, one-command
  cross-platform onboarding (`start.*`, `contribute.*`), and a reproducible zip build.

**Next**

- **Hosted MeshPool Cloud** — operate the public coordinator + relay as an always-on
  service (the coordinator is built and tested; hosting it is an operational step).
  Operator runbook: [`docs/deploy-cloud.md`](docs/deploy-cloud.md).
- **Arbitrary containers** — opt-in Docker/OCI workloads for code that needs a full OS
  (WASM covers pure compute today; containers add native deps, threads and sockets).
- **Distributed LoRA training** (over the frozen sharded pipeline), **GPU** acceleration, direct
  **P2P hole-punching**, control-plane **HA/multi-region**, and real-money settlement.

Full detail: [`docs/roadmap.md`](docs/roadmap.md).

---

## 📚 Docs index

| Doc | What's in it |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Components, data flow, schemas, substrate mapping |
| [docs/quickstart.md](docs/quickstart.md) | Step-by-step first run + troubleshooting |
| [docs/inference.md](docs/inference.md) | Sharding, the correctness oracle, OpenAI API, training |
| [docs/batch.md](docs/batch.md) | Job model, spread scheduling, retries |
| [docs/hosting.md](docs/hosting.md) | Deploy flow, load balancing, failover |
| [docs/deploy-cloud.md](docs/deploy-cloud.md) | Public coordinator deployment runbook |
| [docs/security.md](docs/security.md) | Sandboxing, isolation, secrets, kill-switch |
| [docs/testing.md](docs/testing.md) | Unit + e2e + chaos test strategy |
| [docs/roadmap.md](docs/roadmap.md) | Delivered phases + honestly-deferred next steps |

---

<div align="center">
<sub>MeshPool — your machines, pooled. Built to work like magic, and tested so it doesn't break.</sub>
</div>
