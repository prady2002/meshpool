# MeshPool — Master Plan v2.0
## From "Distributed Inference Mesh" to **Verifiable Solution Mining**

> **Status:** Planning document. **Nothing in here has been executed.**
> **Purpose:** This file is the complete context transfer for the next working session.
> It contains (a) an exhaustive audit of the existing codebase, (b) a researched
> competitive checklist covering all 117 names in `projects.md`, (c) the research
> foundations, (d) a strategic diagnosis of why the current thesis caps out,
> (e) **the pivot — a new, defensible, first-in-world concept**, and (f) a
> grain-level implementation plan.
>
> **Date authored:** 2026-09-01
> **Authored against commit:** `63e93fd` (master)
> **Codebase measured:** 311 Python files / 84,730 LOC + Go agent (40 files) + React console (35 files) + 80 acceptance gates + 91 test modules

---

# TABLE OF CONTENTS

| Part | Title | What it answers |
|---|---|---|
| **0** | [How to use this document](#part-0--how-to-use-this-document) | Reading order, conventions, epistemic tags |
| **1** | [Executive summary & the verdict](#part-1--executive-summary--the-verdict) | What we're doing and why, in 2 pages |
| **2** | [Exhaustive audit of the current codebase](#part-2--exhaustive-audit-of-the-current-codebase) | What exists, what's real, what's a stub |
| **3** | [The 117-name competitive checklist](#part-3--the-117-name-competitive-checklist) | Every company/project in `projects.md`, researched |
| **4** | [Research foundations](#part-4--research-foundations) | The papers and algorithms that matter · 🔴 **the verifier gap (§4.1.2)** |
| **5** | [Strategic diagnosis: why the current thesis caps out](#part-5--strategic-diagnosis-why-the-current-thesis-caps-out) | The uncomfortable truths · **the 24-pivot brainstorm ledger (§5.7)** · **the new competitive landscape (§5.8)** |
| **6** | [**THE INVENTION: Verifiable Solution Mining**](#part-6--the-invention-verifiable-solution-mining) | The pivot · **Meta-Quests (§6.8)** · **why a frontier lab won't do this (§6.9)** |
| **7** | [System architecture](#part-7--system-architecture) | The new stack, layer by layer |
| **8** | [Core algorithms (with pseudocode)](#part-8--core-algorithms-with-pseudocode) | The novel technical contributions |
| **9** | [Economics: Proof-of-Acceptance](#part-9--economics-proof-of-acceptance) | The settlement primitive |
| **10** | [Security & adversarial model](#part-10--security--adversarial-model) | Every attack and its defence · 🔴 **the optimizer–verifier coupling law (§10.2.1)** |
| **11** | [Implementation roadmap (grain level)](#part-11--implementation-roadmap-grain-level) | File-by-file, phase-by-phase |
| **12** | [Verification strategy](#part-12--verification-strategy) | How we prove it works |
| **13** | [Go-to-market](#part-13--go-to-market) | Supply, demand, the wedge, the mission |
| **14** | [Risks, kill criteria, open questions](#part-14--risks-kill-criteria-open-questions) | What would make this fail |
| **A** | [The numbers that matter](#appendix-a--the-numbers-that-matter) | 30 load-bearing figures with provenance |
| **B** | [Glossary](#appendix-b--glossary) | Every new term defined |
| **C** | [Derivations](#appendix-c--derivations) | The math, shown |
| **D** | [The 117-name checklist (verification index)](#appendix-d--the-117-name-checklist-verification-index) | Mechanical proof nothing was skipped |
| **E** | [Reference reading list](#appendix-e--reference-reading-list-for-the-next-session) | Read these before writing code |
| **F** | [First-week checklist](#appendix-f--first-week-checklist-for-the-next-session) | Start here on day one |

---

# PART 0 — HOW TO USE THIS DOCUMENT

## 0.1 Reading order for the next session

If you are an AI coding agent picking this up cold:

1. **Read Part 1** (executive summary). 5 minutes. It tells you what we're building.
2. **Read Part 5 and Part 6.** These are the *why* and the *what*. Do not skip Part 5 — it
   contains the reasoning that justifies discarding a large amount of existing work's
   *positioning* (while keeping the code).
3. **Skim Part 2** to know what already exists. Do not rebuild things listed there.
4. **Work from Part 11.** That is the executable plan. Each task has a file path, an
   acceptance criterion, and a gate.
5. **Consult Parts 3, 4, 7–10** as reference when you hit a design decision.

## 0.2 Epistemic tags used throughout

The existing MeshPool codebase has an unusually strong culture of labelling claims by
provenance (see `docs/benchmarks.md`, which tags every number `MEASURED` / `PROJECTED` /
`CITED`). **This document adopts and extends that convention. Preserve it.**

| Tag | Meaning |
|---|---|
| `[MEASURED]` | Observed on this machine / in this repo. Reproducible with a named command. |
| `[VERIFIED-EXT]` | Fetched from a primary source during this session. URL given. |
| `[CITED]` | Published by someone else; from model knowledge or secondary source. **Re-verify before betting on it.** |
| `[DERIVED]` | Arithmetic on tagged inputs. The derivation is shown. |
| `[ESTIMATE]` | Reasoned judgement. Explicitly not a measurement. |
| `[ASSUMPTION]` | Something we are choosing to believe. Listed in Part 14 as a risk. |
| `[DESIGN]` | A proposed design. Not built. |

> **Rule inherited from the existing codebase and to be kept:** *no performance figure
> reaches a reader without a provenance tier.* This rule has already caught at least three
> real defects in this repo (see `README.md` "Publishing the benchmarks was how we found out
> the benchmarks were wrong"). It is a competitive advantage. Do not dilute it.

## 0.3 A note on what NOT to do

Three failure modes will destroy this project. They are listed first so they are seen first.

1. **Do not delete working code to "make room" for the pivot.** The pivot is ~85% additive.
   Part 11 specifies exactly what is reused, what is demoted, and the *very short* list of
   what is removed.
2. **Do not chase "we're AWS but decentralized."** Part 5 explains in detail why that
   framing is unwinnable. The substrate can do many things; the *product identity* must be
   one thing.
3. **Do not weaken the testing culture.** The 80 acceptance gates, the mutation catalogue,
   the non-vacuity controls, and the `mutation_guard` journal are the single most valuable
   asset in this repository — more valuable than any individual engine. Every new subsystem
   in Part 11 ships with a gate and a mutation that proves the gate can fail.

---

# PART 1 — EXECUTIVE SUMMARY & THE VERDICT

## 1.1 What MeshPool is today

MeshPool is a genuinely well-engineered decentralized compute mesh, at roughly
**85,000 lines of Python + a Go node agent + a React console**, that pools heterogeneous
machines into one resource. It does five things:

1. **Batch** — fan out independent work units, retry on node death, DAG workflows.
2. **Hosting** — deploy app replicas with a load balancer, blue/green, failover.
3. **Inference (flagship)** — shard a transformer pipeline-parallel across machines, with a
   **bit-identical correctness oracle**, speculative pipeline decoding, and continuous batching.
4. **Storage** — content-addressed blobs, Reed–Solomon erasure coding, peer-delegated repair,
   proof-of-storage audits, an S3-compatible surface, client-side encryption.
5. **Mesh Functions** — user-uploaded WASM (`wasm32-wasip1`) executed under fuel/memory/wall
   ceilings with no filesystem, no sockets, no host FFI.

On top of that sits real infrastructure: Ed25519 device identity, an outbound reverse tunnel
plus QUIC transport for NAT traversal, SWIM membership, Vivaldi network coordinates,
RTT-derived "Constellations", node classification (Anchor/Relay/Constellation/Drifter), a
credit ledger with conservation invariants, staking, escrow, held-back earnings, and a
Truebit-style bisection dispute helper.

**The engineering quality is high and the intellectual honesty is exceptional.** This is not
a demo dressed up as a product; it is a product whose author repeatedly went looking for ways
their own measurements were lying to them, and found several.

## 1.2 The verdict

> **The engineering is excellent. The *positioning* is fatal.**

MeshPool's flagship claim — *"run a model too big for any one machine, sharded across
several"* — is a race it structurally cannot win, for four independent reasons that all
point the same way:

| # | Reason | Evidence |
|---|---|---|
| 1 | **The LAN case is already lost to exo.** exo now ships **RDMA over Thunderbolt 5** (claimed ~99% latency reduction), **tensor parallelism** (1.8× on 2 devices, 3.2× on 4), and an MLX backend, running Kimi-K2-Thinking and DeepSeek-v3.1-671B on Mac Studio clusters. | `[VERIFIED-EXT]` github.com/exo-explore/exo, fetched this session |
| 2 | **The WAN case is bounded by physics at ~15 tok/s, and that's the *ceiling with free compute*.** MeshPool's own honest benchmark says a 1T MoE across 11 metro machines tops out at 15.2 tok/s with SPD. Scattered worldwide: 3.0 tok/s. | `[MEASURED]` `docs/benchmarks.md` §3 |
| 3 | **Petals already ran this experiment publicly and the swarm emptied out.** Petals offers Llama-3.1-405B across volunteers at ~6 tok/s for Llama-2-70B, has existed since 2022, and its public swarm is effectively dormant. | `[VERIFIED-EXT]` github.com/bigscience-workshop/petals |
| 4 | **The problem is evaporating underneath.** MoE + aggressive quantization + fast NVMe offload means a *single* machine now runs models that needed a cluster 18 months ago. The "too big for one machine" window is closing, not opening. | `[CITED]` Kimi-K2 native-4bit, gpt-oss-120b on a single 80GB card, GLM-4.5-Air |

Meanwhile the surrounding market is brutal: **117 companies** are in `projects.md` and
essentially all of them are selling either (a) raw GPU-hours or (b) tokens-per-dollar. Both
are commodities racing to zero. MeshPool has **no cost advantage** in either — a mesh of
consumer machines is *more* expensive per useful FLOP than an H100 once you count churn,
verification, bandwidth, and coordination.

**Competing on $/token or $/GPU-hr with a consumer mesh is arithmetically unwinnable.
Stop doing it.**

## 1.3 The pivot, in one paragraph

> The frontier of AI has shifted from **one big tightly-coupled forward pass** (pretraining
> and monolithic inference — which need 3.2 Tbps interconnects and therefore datacenters) to
> **massively parallel independent sampling filtered by a cheap verifier** (test-time compute,
> agentic search, evolutionary program synthesis, theorem proving, RL rollout generation).
> That second workload class is **embarrassingly parallel**, which makes the 100,000× interconnect
> gap between a datacenter and a mesh of home machines *irrelevant*. It is also
> **self-verifying**: a lying node produces a candidate that fails the verifier, so
> verification costs nothing extra *because it is already the workload*. And it makes
> **heterogeneity an asset** — a portfolio of 50 different models at 50 temperatures covers
> more of the solution space than 50 samples from one model. MeshPool should therefore stop
> being a *compute rental network* and become the world's first **Verifiable Solution Mining**
> network: you post a problem plus a machine-checkable evaluator, a planetary mesh evolves
> candidate solutions, and **you pay only for candidates the evaluator accepts**.

## 1.4 The tagline

> **Bitcoin burns electricity to prove nothing.
> MeshPool burns electricity to prove theorems, fix bugs, and discover algorithms.**

## 1.5 Why this is genuinely first-in-world

Three ingredients exist separately today. **Nobody has fused them.**

| Ingredient | Who has it | What they lack |
|---|---|---|
| Decentralized compute mesh | Vast, Akash, io.net, Prime Intellect, Bittensor, Petals, exo | No search semantics; they rent capacity, not outcomes |
| LLM evolutionary/quality-diversity search | DeepMind AlphaEvolve/FunSearch, OpenEvolve, Eureka, ADAS | **No compute layer.** OpenEvolve is a single-process orchestrator that calls a paid API |
| Payment keyed to verifier acceptance | Nobody at scale. Closest: Truebit (fraud proofs), Gensyn (proof-of-learning), bounty platforms (human-judged) | Not applied to LLM search; not automatic; not per-candidate |

**MeshPool = mesh × evolutionary search × proof-of-acceptance.** That intersection is empty.

And critically: **it converts decentralization's two fatal flaws into non-issues.**

| Fatal flaw of decentralized compute | Why it dies under the new thesis |
|---|---|
| **Interconnect** — you can't do tensor parallelism over the internet | Each sample is independent. Zero inter-node traffic during generation. |
| **Trust** — you can't verify a stranger's result | The verifier already decides. A liar's candidate simply fails. Verification overhead ≈ 0. |
| **Churn** — nodes vanish mid-job | A lost sample costs 1/N of one batch, not the whole service. See §5.4 coupling math. |
| **Heterogeneity** — nodes are all different | Diversity *is* the product. See §8.3. |

## 1.6 What gets reused

**~85% of the existing codebase transfers directly.** This is not a rewrite.

| Existing asset | New role |
|---|---|
| Batch engine + DAG + retry + work-stealing | **Sample fan-out & the evaluation cascade** |
| WASM function sandbox | **Evaluator sandbox** (deterministic, network-denied — exactly right) |
| Content-addressed store + erasure coding + Weight Swarm | **Model/Adapter Swarm + candidate archive** |
| Constellations (RTT clusters) | **Islands** in the island-model evolutionary search |
| Trust engine, staking, escrow, bisection dispute | **Liveness reputation + evaluator dispute resolution** |
| Credit ledger with conservation invariant | **Proof-of-Acceptance settlement** |
| Tunnel / QUIC / NAT / SWIM / Vivaldi | Unchanged. Still the transport. |
| Go agent (`meshd`) | Unchanged, and becomes the primary agent. |
| NumPy nano-GPT + bit-identical oracle | **Demoted to reference oracle for determinism testing** — a genuinely valuable role |
| Pipeline sharding + SPD + continuous batching | **The "large-model arm"** of the portfolio; no longer the flagship |
| 80 acceptance gates + mutation catalogue | Unchanged and extended. The crown jewels. |

**Explicitly demoted** (kept, no longer led with): pipeline-parallel sharding as the headline,
S3-compatibility as a product, app hosting as a product.
**Explicitly killed:** the "AWS-but-decentralized" narrative. Nothing else.

## 1.7 The wedge (what someone uses on day one)

**`meshpool quest`** — massively parallel, test-verified code repair.

```bash
# You have a failing test. You have idle machines. You have 90 seconds.
meshpool quest run \
  --repo . \
  --goal "make tests/test_parser.py::test_nested_arrays pass" \
  --evaluator "pytest tests/test_parser.py -x -q" \
  --samples 200 \
  --portfolio auto
```

200 independent attempts, across every idle machine you own plus (optionally) the public
mesh, each with a different model / temperature / adapter / prompt scaffold, every candidate
run against your real test suite in a sandbox, **ranked by what actually passed**, deduplicated,
and handed back with a replay receipt.

Nobody sells this. Cursor runs ~1–8 parallel agents. Claude Code runs subagents. Devin runs one.
**pass@200 ≫ pass@1**, and the compute to get there is sitting idle in the building.

---

# PART 2 — EXHAUSTIVE AUDIT OF THE CURRENT CODEBASE

> This part exists so the next session does not rebuild what is already there, and does not
> *trust* what is only apparently there.

## 2.1 Repository shape `[MEASURED]`

```
meshpool/
├── meshpool/              # 311 py files, 84,730 LOC total repo
│   ├── controlplane/      # app.py (2650), registry (462), scheduler (568),
│   │                      # verification (519), openai_api (489), accounts,
│   │                      # spend, timeline, topology, nodeclass
│   ├── agent/             # daemon (1058), probe (744), sandbox (336),
│   │                      # runtime (239), donation, tunnel_client
│   ├── inference/         # engine (1023), batching (644), model (572),
│   │                      # pipeline (550), remote (544), gguf_loader (540),
│   │                      # kvcache (336), train (294), lora (277), spd (265),
│   │                      # adapters (250), bench (232), gguf_tokenizer (173)
│   ├── storage/           # engine (1052), s3_api (596), erasure (391),
│   │                      # audit (382), crypto (152), blobstore, s3_credentials
│   ├── batch/             # engine (755), dag (193), functions (177)
│   ├── hosting/           # engine (365), site (284), frameworks (121)
│   ├── functions/         # wasm.py (395)
│   ├── cloud/             # quic (286), nat (188), tunnel (186), swim (126)
│   ├── common/            # db (601), models (451), auth (369), billing (298),
│   │                      # proactor_fix (246), trust (224), egress (217),
│   │                      # faultinject (211), vivaldi (196), identity (190),
│   │                      # transport (158), serve_guard (153), config (137)
│   └── cli/main.py (1357)
├── agent-go/              # 40 Go files: meshd, probe, sandbox, tunnel, natprobe,
│                          # quictransport, wasmfn (wazero), update (Ed25519), distribution
├── web/                   # React console: 35 src files, 8 pages, Playwright e2e
├── sim/                   # harness, netem, perfmodel, topology, mutation_guard,
│   └── scenarios/         # 80 acceptance gates
├── tests/                 # 91 test modules
├── docs/                  # 17 documents
└── proto/meshpool.proto   # wire contracts (mirrored by Pydantic, not used for gRPC)
```

## 2.2 What is genuinely real and good

These are **verified by reading the code and its gates**, not by reading claims.

### 2.2.1 The correctness oracle `[MEASURED]`
`tests/test_model.py::test_shard_pipeline_equals_full_model` plus
`sim/scenarios/verify_inference.py`. A pipeline of `ShardModule`s produces output
**bit-identical** to `forward_full`, and *stays* identical after a node is killed and the
model repartitions. This is rare and valuable. Almost no distributed inference system can
make this claim, because almost none are deterministic.

**→ New role:** this becomes the foundation of **Replay Receipts** (§8.6). Keep it. Extend it.

### 2.2.2 Speculative Pipeline Decoding (`inference/spd.py`) `[MEASURED]`
Genuinely clever: the draft model is *the first stage's own first few layers*, which already
live on one machine — so drafting costs **zero extra network hops**. K tokens verified in one
traversal, rejection sampling preserves the exact output distribution.

Measured: ×2.2–×2.5 across wide-area topologies, and — the honest part — **×0.87–×0.93 at
depth 1**, where there is no RTT to amortise. Median acceptance 38%, 2.38 tokens/traversal.

### 2.2.3 Continuous batching that refuses to fake it (`inference/batching.py`) `[MEASURED]`
Rows are **not** stacked into one matmul, because that would change float reduction order and
make a tenant's output depend on who shared its batch. Only the *hop* is shared. Measured on a
real 3-process cluster: 288 wide-area hops → 51 (82% fewer), 1.65–2.09× aggregate tok/s.

And the README documents that an earlier 2.5–3.6× figure was **wrong** because the gate
constructed a fresh `httpx.Client` per request (~0.67 s TLS trust-store load). That
self-correction is the culture worth preserving.

### 2.2.4 Weight Swarm (`inference/engine.py` + `storage/`) `[MEASURED]`
Checkpoints are split **per transformer block** into content-addressed blobs with a tiny JSON
manifest. Nodes pull layer groups **directly from peers** (`POST /blob_pull`), sha256-verified
per chunk and per group. `GET /api/v1/cluster` reports `weight_group_pulls` vs
`weight_group_pushes`; the gate asserts pulls ≥ 1 and pushes == 0 on a LAN, i.e. **the
coordinator moves zero weight bytes**.

There is also `weight_group_cached` — already counting warm residency. **This is the seed of
Weight-Affinity Scheduling (§8.4).**

### 2.2.5 Erasure-coded storage with delegated repair `[MEASURED]`
Reed–Solomon with a crossover from replication to EC, and — importantly — a **byte-free repair
plan**: the coordinator names peer URLs and one node pulls `k` shards directly, reconstructs
locally, keeps the missing ones. The relay path survives deliberately for CGNAT holders
(budgeted at 25–40% of the fleet). Every fetched shard is checked against its content address
**and** every reconstructed shard against the id requested — because a content address proves
what bytes are, not which *encoding generation* they came from. That second check is a subtle,
correct piece of engineering.

### 2.2.6 The WASM sandbox (`functions/wasm.py` + `agent-go/internal/wasmfn`) `[MEASURED]`
`wasm32-wasip1`, no preopened directories (`path_open` → `EBADF`), no sockets linked, modules
importing anything outside WASI p1 **rejected at upload**, fuel budget + epoch-interrupt
watchdog + memory ceiling, executed in a dedicated child process. Two runtimes (wasmtime in
Python, wazero in Go) with a **parity gate** comparing values *and failure kinds*.

**→ This is already 70% of the Evaluator Sandbox we need.** See §7.3.

### 2.2.7 Node classification (`controlplane/nodeclass.py`) `[MEASURED]`
Four classes — Anchor / Relay / Constellation / Drifter — derived from *measured* signals (NAT
class, battery state, uptime, latency consistency). Two deliberate limits, both correct:
it is a **preference not a prohibition** (a volatile node is used if dropping it makes the model
unhostable), and **promotion requires 50 observations** because every reputation EMA defaults to
1.0 so a stranger's first heartbeat would otherwise look flawless.

### 2.2.8 Constellations (`controlplane/scheduler.py`) `[MEASURED]`
RTT-derived soft clusters (threshold <25 ms), with a **pre-bucketed O(1) capacity index**
(`ConstellationIndex._fit_bucket`) so lookup does not re-run a global optimization per request,
and a `lookup_scans` counter that a gate asserts on. Placement minimizes
`(pipeline_depth − 1) × avg_internal_rtt`.

**→ Constellations become Islands in the evolutionary architecture (§8.2). Perfect fit.**

### 2.2.9 The verification/economics stack (`controlplane/verification.py`) `[MEASURED]`
Already implements: workload→technique classification, redundant execution at a configurable
rate (default 0.25), spot-checks against a local oracle, `ComputeEscrowEngine` (release after
3 consecutive passes, forfeit on fail), `StakingEngine` (graduated stake ~20× expected job
value for new nodes), `HeldBackEscrowEngine` (Storj-style 50% → 25% @3mo → 0% @9mo), and
`InteractiveBisectionDispute` (O(log N) Truebit-style trace localization).

**Crucially, `VerificationPolicy.classify()` returns `NOT_VERIFIABLE` for unknown
user-supplied functions.** That single line is the exact hole this pivot fills: **make the user
supply the verifier.**

### 2.2.10 The engineering culture `[MEASURED]`
This deserves its own entry because it is the most transferable asset:

- **80 acceptance gates** that boot real multi-process clusters.
- **Mutation testing with a journal** (`sim/mutation_guard.py`) — gates prove they are not
  vacuous by breaking a real source file and requiring the re-run to fail. And the journal
  exists because `finally` cannot survive `TerminateProcess`, which was discovered when a
  killed gate left `openai_api.py` mutated and **a third of a 78-gate sweep ran against a
  corrupted checkout**.
- **The runner diffs `git status` around every gate** and fails any gate that leaves the tree
  dirty even if it exited zero.
- **Non-vacuity controls everywhere** — a canvas check that wipes the surface and re-measures;
  an "identical job completes when uncapped" control on the spend-cap gate.
- **"Verify the verifier"** — they found `npx tsc --noEmit` in `web/` exits 0 *without opening
  a single file* (solution-style tsconfig), by planting a deliberate type error.

## 2.3 What is a stub, a simulation, or a limitation

Honest inventory. Most of these are **already documented by the project itself**, which is
itself remarkable.

| # | Item | Reality | Severity for the pivot |
|---|---|---|---|
| 1 | **No GPU support anywhere** | Inference is NumPy on CPU. The adapter seam exists (`adapters.py`) but `PetalsBackend` and `ExoBackend` are `_UnavailableBackend` stubs that raise. Only `MeshShardBackend` (NumPy), `OllamaBackend`, and `GGUFBackend` are real. | 🔴 **Blocking.** The pivot needs real local inference. |
| 2 | **No pretrained weights ever tested** | Corporate TLS interception blocks HuggingFace (`CRYPT_E_NO_REVOCATION_CHECK`). Every checkpoint under test has real *format* and deterministically generated *weights*. Model quality is not measured and not claimed. | 🔴 **Blocking.** Must be resolved on a machine with internet. |
| 3 | **`verify_first_real_token.py` uses `qwen2-tiny-random`** | `d_model=8`, 2 layers. The 3-way split degenerates — the last process holds zero transformer layers. Weights sent as **base64 JSON in the request body**, impractical for any real checkpoint. | 🟠 The Weight Swarm path is the real one; this gate is a separate legacy path. |
| 4 | **K-quants untested locally** | Q4_K/Q6_K/Q2_K/Q3_K/Q5_K supported via the reference dequantizer but the `gguf` package can decode and not encode, so no local fixture exists. | 🟡 Test on a machine with real GGUFs. |
| 5 | **GGUF RoPE convention is a transcribed switch** | llama.cpp picks Q/K layout by architecture alone; **there is no GGUF metadata field recording it**. An earlier hand-written allowlist had **4 of 5 entries wrong**, including omitting `deepseek2`. Getting it wrong doesn't crash — the model generates fluently and is quietly wrong. | 🟠 Standing liability. Needs a per-architecture golden-output test. |
| 6 | **MLA (multi-head latent attention) not implemented** | `_reject_latent_attention` refuses it. So DeepSeek-V3 / Kimi-K2 tensor layouts are out of reach. | 🟡 Fine — the pivot doesn't need them sharded. |
| 7 | **S3 SigV4 binds a *claimed* payload digest** | `x-amz-content-sha256` is used verbatim; the body is never hashed and compared. A captured signed PUT can be replayed with substituted content. Bounded only by a 15-min skew window. | 🟡 S3 is being demoted anyway. Document loudly. |
| 8 | **Client-side encryption cannot be server-enforced** | `POST /api/v1/blobs` accepts plaintext unconditionally. Bypass the client library, get no confidentiality. | 🟡 Genuine, documented. |
| 9 | **Proof-of-storage audits are per-chunk hashes, not Merkle/PoRep** | A cheater can cache the ~8 KB/MB hash table instead of the data. Defence is economic (escrow forfeit), not cryptographic. | 🟡 Acceptable, documented. |
| 10 | **Auto-update is fully implemented and completely inert** | No MeshPool release signing key exists, so the compiled-in trust store is empty and `Verify` returns `ErrUnknownKey` for every manifest. Deliberate: the alternatives are a placeholder key nobody holds (a lie that looks like security) or "no keys → skip verification" (fleet-wide RCE primitive). | 🟢 Correct decision. Release engineering task. |
| 11 | **Fraud-detection gate is in-process, not transport-E2E** | `verify_fraud_detection.py` drives the real `BatchManager`/`Verifier`/`TrustEngine`/ledger/SQLite but **does not** prove a real multi-process agent returning a bad answer over the production transport is caught and slashed. | 🔴 Must be fixed before anonymous public supply. |
| 12 | **Metering: only batch accepted-unit billing is wired to production receipts** | Inference, hosting, storage, functions are defined in the meter but not wired. | 🟠 Needed for PoA settlement. |
| 13 | **HA / Raft is a config flag only** | `Settings.ha` is threaded through and set `True` in the `enterprise` preset. No Raft, no standby, no leader election. State is a single SQLite file. | 🟠 Real ceiling at scale. |
| 14 | **`DockerRuntime` is a named seam, not an implementation** | `ProcessRuntime` is the only real runtime. Arbitrary containers are explicitly "Next". | 🔴 **Blocking** for real evaluators (test suites need a real OS). |
| 15 | **NAT traversal is classification + relay plan, not hole punching** | Symmetric↔symmetric treated as **0%**, relay-first. Budgets **25–40% permanent relay dependency**. Direct P2P hole-punching is on "Next". | 🟢 Honest and correct. |
| 16 | **QUIC migration/fallback measured under emulation only** | Not real-internet success rates. | 🟢 Honest. |
| 17 | **SWIM churn convergence is single-machine emulation** | 30%+ injected churn, deterministic. Not a 10k-node internet deployment. | 🟢 Honest. |
| 18 | **Constellation thresholds are reasoned, not fitted** | No observed churn data exists on a single-machine test bench. Same for nodeclass thresholds. | 🟡 Needs real telemetry. |
| 19 | **No real money** | Credits only. No payouts, invoices, fiat rails, taxes, or regional price books. | 🟠 Needed eventually. |
| 20 | **`meshd` doesn't host replicas, serve shards, or redeem secrets** | Returns explanatory `501`. Advertises `services: ["batch","storage"]` and `runtimes: ["wasm"]`, enforced structurally in both directions by `advertisement_test.go`. | 🟢 Excellent design. |
| 21 | **`proto/meshpool.proto` is not used for gRPC** | It mirrors Pydantic models as documentation. Transport is HTTP/JSON + SSE + QUIC. | 🟢 Fine. |
| 22 | **NumPy in the inference hot path** | For a real 7B+ model this is ~2–3 orders of magnitude too slow. `model.py` is pure NumPy matmuls. | 🔴 **Blocking.** Must delegate to llama.cpp/vLLM/MLX. |
| 23 | **Windows Proactor listener bug required a custom fix** | `common/proactor_fix.py`. CPython's Windows loop closes a listening socket on *any* `OSError` from `accept()`. **4 of the first 47 gates (~8%) died this way.** | 🟢 Fixed, and the fix is correct. Keep it. |

## 2.4 The single most important line in the codebase

```python
# meshpool/controlplane/verification.py — VerificationPolicy.classify()
# Unknown function (user-supplied): conservatively not verifiable.
return WorkloadKind.NOT_VERIFIABLE
```

Today, **anything a user brings is unverifiable**, so the public mesh can only safely run
curated built-ins (`square`, `is_prime`, `sha_pow`, `embed`) plus WASM (deterministic by spec).
That is a tiny, valueless workload surface.

The pivot's entire technical thesis is: **stop trying to verify the work; make the user bring
the verifier.** The moment a Quest carries its own acceptance predicate, `NOT_VERIFIABLE`
becomes `SELF_VERIFYING` and the workload surface explodes from "four toy functions" to
"anything with a test suite."

---

# PART 3 — THE 117-NAME COMPETITIVE CHECKLIST

> Every name in `projects.md`, categorised, with what it does, how it's built, its key
> algorithm/mechanism, and the specific lesson for MeshPool.
>
> **Provenance:** items marked `[VERIFIED-EXT]` were fetched during this session. The
> remainder are `[CITED]` from model knowledge and **must be re-verified before any decision
> depends on a specific number.** Treat every figure here as directionally right and
> numerically stale.

## 3.0 The checklist (all 117, grouped)

**Group A — DePIN GPU marketplaces (20):** Vast.ai · Akash Network · Akash Homenode · io.net · Aethir · Nosana · Spheron · SaladCloud · Golem · Flux · NuNet · Render Network · Dispersed/Render Dispersed · Bittensor · Theta EdgeCloud · Clore.ai · OctaSpace · Qubic · Dynex · DeepBrain Chain

**Group B — Distributed training/inference research & systems (13):** Petals · exo · Hivemind · Gensyn · Prime Intellect · Nous Research · DisTrO · OpenDiLoCo implementations · Together Computer/Together AI origins · Experimental academic systems · Various academic projects · Hyperbolic · Exabits

**Group C — Verifiable / confidential compute (16):** iExec · iExec Workerpool · Phala Network · Oasis Network ROFL · Marlin · Aleph.im · Ankr · Dfinity Internet Computer · Cartesi · Lilypad Network · Bacalhau · Bacalhau by Expanso · HyperCycle · Crynux · Inference Labs · EigenAI/EigenCloud

**Group D — Crypto-AI protocols & agent networks (17):** Ritual · Atoma Network · OpenGradient · TensorOpera/Fox-1 · Morpheus (×2 in file) · Ora Protocol · Fetch.ai/ASI Alliance · SingularityNET · Ocean Protocol · Allora Network · Cortex · Oraichain · Giza · Sentient · Sahara AI · Vana · Grass

**Group E — Edge / mobile / gaming DePIN (8):** Acurast · NodeOps Network · CUDOS/CUDOS Intercloud · NeurochainAI · Inferix · GAIMIN · DATS Project · Titan Network · Raiinmaker · Commune AI · Heurist

**Group F — Neoclouds / GPU IaaS (26):** RunPod · TensorDock · FluidStack · Sesterce · Hyperstack · JarvisLabs.ai · Qubrid AI · Shadeform · CloudRift · Massed Compute · DataCrunch · Genesis Cloud · Nebius AI · Nscale · Voltage Park · Crusoe Cloud · Vultr Cloud GPU · Scaleway GPU · OVHcloud GPU · DigitalOcean Gradient AI · Lambda · CoreWeave · Paperspace · Cirrascale · Ori · Northflank

**Group G — Serverless GPU / inference platforms (12):** Modal · Replicate · Baseten · Together AI · Fireworks AI · Anyscale · Lepton AI · Fal.ai · Cerebrium · Beam · Banana.dev · Chutes.ai · Fluence

---

## 3.1 Group A — DePIN GPU marketplaces

| Project | What it does | Architecture / key mechanism | Verification model | Lesson for MeshPool |
|---|---|---|---|---|
| **Vast.ai** | The eBay of GPUs. Aggregates consumer + datacenter GPUs. | Centralized matchmaker + Docker on host. Real-time **bidding/interruptible pricing**; a **DLPerf** composite score ranks machines. Reliability score per host. | None cryptographic. Reputation + a benchmark score + user reviews. Renter runs their own container and sees the result. | **The market clears without verification because the renter *is* the verifier.** This is the single most important lesson in Group A: don't verify compute, expose it and let the buyer check. Vast is the most successful decentralized GPU market and it has *no* proof system. |
| **Akash Network** | Decentralized cloud on Cosmos. Deploy any container via SDL manifests. | Cosmos-SDK chain; **reverse auction** — tenant posts a deployment, providers bid, lowest bid wins. Kubernetes underneath (provider runs a k8s cluster). Akash Console UI. **Akash Homenode** extends supply to home machines. | Provider staking + escrow on-chain. Uptime attested loosely. No result verification. | The **reverse auction** is a clean price-discovery mechanism and is directly reusable for Quest bounties. Also note: Akash's utilization has historically been dominated by a small number of large providers — decentralized supply is hard to *keep*. |
| **io.net** | Aggregates GPUs into **clusters** for ML, via Ray. | **IO Cloud / IO Worker / IO Explorer**. Their key trick is a Ray-based cluster abstraction over geographically scattered GPUs, plus a mesh VPN. | Proof-of-work-ish device checks; had a well-publicized **fake-GPU / spoofed-device incident** where metadata could be forged, inflating node counts. | **The cautionary tale.** Any system that pays for *claimed capacity* will be flooded with fake capacity. Pay for *delivered, verified outcomes*. This is exactly why Proof-of-Acceptance matters. |
| **Aethir** | Enterprise-grade GPU-as-a-service, cloud gaming heritage. **Aethir Edge** hardware nodes. | Container/checker/indexer node roles. Focus on low-latency rendering & gaming, then AI. | "Checker nodes" probe container nodes for liveness/perf. | Role separation (container/checker/indexer) is a reasonable pattern; MeshPool already has a stronger version via nodeclass. |
| **Nosana** | Solana-based GPU grid, originally CI/CD. | Job spec → market → node picks up. Solana for settlement. | Staking + slashing; node reputation. | CI/CD as the wedge is instructive — it's *verifiable work* (tests pass or don't). Nosana had the right wedge and didn't push the verification insight. |
| **Spheron** | Decentralized compute, "**fizz nodes**" for consumer hardware. | Provider/fizz split; container workloads. | Staking. | — |
| **SaladCloud** | Consumer **gaming PCs** as a compute fleet. | Distributed container scheduling with very high churn tolerance; the workload must be interruptible. Strong on batch image gen / transcoding / data collection. | Reputation + workload-level idempotency. Emphasis on "chaos-tolerant" job design. | **Closest philosophical cousin.** Salad's entire engineering thesis is *design workloads so churn doesn't matter*. That is precisely the embarrassingly-parallel insight. Study their job model. |
| **Golem** | The original (2016) decentralized compute market. GLM token, Task API, `ya-runtime-vm`. | Requestor/provider; **golem-vm** (gVisor/QEMU-based) sandbox; WASI runtime option. | Redundant execution + reputation; historically weak. | **The graveyard's headline entry.** Golem had 8 years, a great team, real tech, and never found a workload. *Technology was never the bottleneck; demand was.* MeshPool must lead with a workload, not a substrate. |
| **Flux** | Decentralized cloud, **FluxOS**, tiered FluxNodes (Cumulus/Nimbus/Stratus), FluxEdge for GPU. | Docker apps deployed across nodes with redundancy; collateral-tiered nodes. | Collateral + uptime checks. | Tiered collateral is a decent sybil deterrent. MeshPool's graduated staking (~20× expected job value) is the same idea, better targeted. |
| **NuNet** | "Decentralized computing framework", DMS (Device Management Service). | Ontology-driven resource matching; ambitious. | — | Over-abstracted; low traction. Lesson: don't build an ontology before a customer. |
| **Render Network** | GPU rendering (OctaneRender heritage), now RNP proposals & broader compute. **Dispersed/Render Dispersed** splits a frame across nodes. | Tiered nodes; **proof-of-render** — output frames are compared/spot-checked. Burn-and-mint equilibrium tokenomics. | **Proof-of-render is a verifier!** A rendered frame can be checked by re-rendering a tile or comparing against neighbours. | **The one project in Group A that already lives on the right side of the thesis.** Rendering is verifiable, embarrassingly parallel, and Render is the most commercially real DePIN compute network. **This is the existence proof that Verifiable Solution Mining works.** Generalize what Render did for one vertical to *all verifiable work*. |
| **Bittensor** | Incentivized ML network of **subnets**. TAO/dTAO. | Miners produce outputs, **validators score them**, **Yuma Consensus** aggregates validator weights into emissions. Subnets specialize (inference, data, finetuning). | **Validator consensus, not objective verification.** Scores are subjective; the mechanism rewards agreement with the validator majority. | **The critical lesson.** Bittensor's mechanism is repeatedly gamed (weight copying, validator collusion, miners optimizing for the scorer rather than the task) because **the scoring function is subjective**. Proof-of-Acceptance avoids this entirely by requiring the predicate to be *objective and re-runnable by anyone*. If your reward signal isn't independently reproducible, it will be gamed. |
| **Theta EdgeCloud** | Video CDN + edge compute, hybrid cloud/edge. | Edge nodes cache/relay video; EdgeCloud sells GPU. | — | Real revenue via a *non-AI* primary business (CDN). Diversification lesson. |
| **Clore.ai** | GPU rental + PoW hybrid. | Marketplace, host reputation. | — | — |
| **OctaSpace** | Decentralized compute/VPN/CDN. | — | — | — |
| **Qubic** | "Useful Proof of Work" — miners train ANNs (AIGARTH). | 676 computors, quorum consensus; mining tasks are neural net training. | Quorum voting. | **Directly relevant framing** — "useful PoW". But the useful work is an internal, unfalsifiable objective. MeshPool's version has *external* users supplying the objective, which is far stronger. |
| **Dynex** | "Neuromorphic" computing, proof-of-useful-work. | Claims to solve optimization/QUBO problems on GPUs. | Heavily criticized; the "useful work" claims are hard to independently verify. | **Anti-lesson.** Claiming useful work without letting anyone check it is how you become a punchline. MeshPool's receipts must be independently replayable. |
| **DeepBrain Chain** | GPU compute for AI, long-running (2017). | Marketplace + staking. | — | — |
| **Render Dispersed** | (see Render) | | | |
| **Akash Homenode** | (see Akash) | | | |

**Group A cross-cutting patterns:**
1. **Every successful one has no result verification** — because the buyer verifies (Vast, Akash) or the work is inherently checkable (Render).
2. **Every one that pays for claimed capacity gets fake capacity** (io.net's incident).
3. **The token is usually the product**, which means supply appears and vanishes with the price.
4. **Nobody sells outcomes. Everybody sells hours.**

## 3.2 Group B — Distributed training & inference

| Project | Mechanism | Real numbers | Lesson |
|---|---|---|---|
| **Petals** `[VERIFIED-EXT]` | BitTorrent-style pipeline parallelism. Clients pick a chain of servers via a **DHT (hivemind)**; only activations cross the wire; supports custom paths, hidden states, and **distributed fine-tuning of client-owned adapters over frozen sharded base layers**. | **6 tok/s Llama-2-70B**, **4 tok/s Falcon-180B**, single-batch. Supports Llama-3.1-405B, Mixtral 8x22B. | **The experiment already ran.** The tech works; the swarm emptied. Volunteers won't donate GPUs indefinitely for someone else's chatbot. *Petals' distributed-LoRA-over-frozen-base design is the single most reusable idea* — see Adapter Swarm §8.5. |
| **exo** `[VERIFIED-EXT]` | Peer-to-peer, no master. **Topology-aware auto-parallel** using realtime device resources + link bandwidth/latency. **RDMA over Thunderbolt 5** (day-0, "99% latency reduction"). **Tensor parallelism**: 1.8× on 2 devices, 3.2× on 4. MLX + MLX-distributed backend. OpenAI/Claude/Ollama-compatible APIs. | 4 × 512GB M3 Ultra Mac Studio running **DeepSeek-v3.1 671B (8-bit)** and **Kimi-K2-Thinking (4-bit)**, Qwen3-235B (8-bit). Linux is CPU-only today. | **MeshPool's flagship is exo's home turf and exo is far ahead there.** Do not compete. *But note exo's weakness:* it is a **LAN/Thunderbolt** system with no economy, no verification, no multi-tenancy, no WAN story, and Linux GPU support is unfinished. |
| **Hivemind** | DHT + decentralized averaging. **Moshpit SGD** (butterfly all-reduce over random groups), DeDLOC ("Distributed Deep Learning in Open Collaborations"), secure aggregation. | Trained ALBERT/sahajBERT with volunteers. | The DHT and the averaging primitives are solid. The *training* thesis is the one that didn't survive. |
| **Prime Intellect** `[VERIFIED-EXT]` | The most serious player. **OpenDiLoCo** (low-communication data-parallel), **INTELLECT-1/2/3**, **PCCL** (their collective comms library), **SHARDCAST** (weight broadcast), **TOPLOC** (locality-sensitive-hash verification of inference), **prime-rl** (async RL at 1000+ GPUs, FSDP2 + vLLM, FP8, P/D disaggregation, EP/CP parallelism), **`verifiers`** library and the **Environments Hub**. | prime-rl trains 1T+ MoE on 1000+ GPUs. Supports GLM-5, Qwen3.5-MoE, MiniMax-M2, Nemotron-H, INTELLECT-3, gpt-oss. Requires ≥1 NVIDIA GPU. | 🔴 **The closest competitor to the pivot, and the most important entry in this table.** They have built *exactly the right primitives* — an environments/verifiers ecosystem where tasks carry their own reward function. **But:** (a) they run on **rented datacenter GPUs**, not a consumer mesh; (b) `verifiers` is aimed at **RL training**, not at *end-user problem solving*; (c) there is **no marketplace where a stranger posts a problem and gets paid-on-acceptance solutions**. MeshPool should be **interoperable with `verifiers`/Environments Hub** (adopt their env format!) and compete on the *demand side* they aren't serving. |
| **Gensyn** | Verified training. **Probabilistic proof-of-learning** + graph-based pinpoint + Truebit-style refereed delegation. **Verde** paper (refereed delegation for ML), **RepOps** (reproducible operators — bitwise-reproducible kernels across hardware), RL Swarm, Judge. | Testnet. | **RepOps is directly relevant.** Bit-reproducible operators across heterogeneous hardware is the enabling technology for Replay Receipts (§8.6). Read the Verde paper before designing dispute resolution. |
| **Nous Research** | **DisTrO** / **DeMo** (Decoupled Momentum Optimization — decouples momentum so you sync ~1000× less), **Psyche** network, Consilience-40B, Hermes models. | Claimed 100–1000× bandwidth reduction for data-parallel training. | The bandwidth-reduction line of work is real and impressive, but it addresses *training*, which MeshPool has correctly scoped out. |
| **DisTrO / OpenDiLoCo / DiLoCo** | Local SGD with infrequent outer synchronization; streaming/async variants. | DiLoCo: sync every ~500 steps. | Same as above — training is not the wedge. |
| **Together AI (origins)** | "Decentralized Training of Foundation Models in Heterogeneous Environments" — a **scheduling algorithm** that assigns pipeline stages to minimize communication over a slow, heterogeneous network. **CocktailSGD**. | Then pivoted **entirely** to a centralized GPU cloud + inference API. | 🔴 **The most instructive pivot in the industry.** The smartest people working on decentralized training looked at the numbers and became a neocloud. Understand *why* (they did — the economics), and make sure the pivot chosen here doesn't have the same flaw. It doesn't, because search is not bandwidth-bound. |
| **Hyperbolic** | Decentralized GPU marketplace + inference API; "proof of sampling" (PoSP) work with academic collaborators. | — | **Proof of Sampling Protocol** — verify a random sample of inferences by re-execution, with a game-theoretic argument that the *threat* of sampling makes honesty dominant. Read the paper; it's the cheapest credible general-purpose verification. |
| **Exabits** | GPU aggregation. | — | — |

## 3.3 Group C — Verifiable & confidential compute

| Project | Trust mechanism | Real cost/overhead | Lesson |
|---|---|---|---|
| **iExec** (+ Workerpool) | Intel **SGX/TDX** TEEs (Scone, Gramine), **PoCo** consensus, DataProtector for confidential datasets. | TEE overhead modest for CPU work; SGX enclave memory limits are painful. | Workerpool abstraction (a pool operator aggregating workers) is a good org pattern. |
| **Phala Network** | TEE workers, Phat Contracts, **dstack**, **GPU TEE** using NVIDIA H100 confidential computing + Intel TDX. Phala Cloud. | NVIDIA CC mode: `[CITED]` typically **~5–15% overhead** for large-batch LLM inference; much worse for small transfers. | **The strongest TEE story in crypto-AI.** If MeshPool ever needs confidential inference on strangers' hardware, this is the reference design. |
| **Oasis ROFL** | Runtime OFfchain Logic; Sapphire confidential EVM. | — | — |
| **Marlin** (Oyster, Kalypso) | TEE serverless + a ZK proving marketplace. | — | Proving marketplaces are a real, if niche, business. |
| **Aleph.im** | Decentralized VMs + storage; confidential VMs via **AMD SEV**. | — | — |
| **Ankr** | RPC/node infra + AI. | — | Boring infra is a real business. |
| **Dfinity ICP** | Canisters, deterministic **WASM** execution, chain-key crypto, subnet consensus. | Every computation replicated across a subnet → **~N× cost**. | 🟢 **Deterministic WASM execution replicated for consensus** is the purest form of "verification by replication." MeshPool's WASM sandbox is the same primitive at 1/N the cost because it doesn't need consensus, only *acceptance*. |
| **Cartesi** | **Cartesi Machine** — a deterministic RISC-V VM; Dave/fraud-proof dispute resolution; run real Linux binaries verifiably. | Interactive dispute is O(log N) rounds. | 🟢 **Highly relevant.** A deterministic RISC-V VM that can run a real Linux toolchain is *exactly* what an Evaluator Sandbox wants: real programs, deterministic, bisectable. **Seriously evaluate `cartesi-machine` (or RISC-V + deterministic syscall layer) as the Tier-2 evaluator runtime.** |
| **Lilypad** | Bacalhau-derived module execution; Anvil. | — | — |
| **Bacalhau / Expanso** | "Compute over data" — move compute to where data already sits (IPFS/S3). Pivoted toward distributed data engineering. | — | 🟢 **The "compute follows data" thesis is correct and under-exploited.** MeshPool's private-plane story (§6.6) is the same idea applied to *problems*, not datasets. |
| **HyperCycle** | AI node factories, TODA/IP ledgerless. | — | — |
| **Crynux** | Stable Diffusion / LLM network. **VRF-based node selection** + consensus by *similarity of outputs* across independently-selected nodes. | 3× replication typical. | 🟢 **Similarity consensus for non-deterministic outputs** is a real technique worth knowing. It's what you do when you can't be bit-exact. Weaker than acceptance, stronger than nothing. |
| **Inference Labs (Omron)** | **Proof-of-inference** via zkML. | zkML overhead is the killer — see §4.6. | — |
| **EigenAI / EigenCloud / EigenLayer** | Restaking → AVSs (Actively Validated Services). Verifiable AI as an AVS; EigenDA. | Cryptoeconomic security, not cryptographic proof. | **Restaking-as-security is a template for MeshPool's staking**, but it requires a token. Note MeshPool's stated non-goal: *no blockchain/token settlement*. Respect that unless deliberately revisited (see §9.7). |

## 3.4 Group D — Crypto-AI protocols, agents, data

| Project | What it is | Lesson |
|---|---|---|
| **Fetch.ai / ASI Alliance** | uAgents, Agentverse, AI Engine. Merged FET+AGIX+OCEAN (+CUDOS) into ASI. | Consolidation signal: the crypto-AI long tail is collapsing into a few brands. |
| **SingularityNET** | AI service marketplace, `snet-daemon`, MeTTa/Hyperon. | The marketplace-for-AI-services idea is a decade old and has never found demand. **Marketplaces fail without a killer workload.** |
| **Ocean Protocol** | **Compute-to-Data** — algorithms travel to private data, results come back, raw data never moves. Datatokens, Ocean Nodes, Predictoor. | 🟢 **Compute-to-Data is the correct primitive for private problems** and is directly reusable as MeshPool's private-plane model (§6.6). Ocean's failure was demand, not design. |
| **Morpheus** | Decentralized "smart agents", MOR20, compute providers, Lumerin router. | — |
| **Allora Network** | Context-aware inference; topic coordinators; **forecasting-of-losses** — workers predict *other workers' errors*, and the network synthesizes a weighted combination. | 🟢 **The most intellectually interesting mechanism in Group D.** Meta-learning which worker to trust *per context* is directly applicable to MeshPool's portfolio bandit (§8.3): learn which (model, scaffold) arm wins on which *kind* of problem. **Steal this.** |
| **Cortex** | On-chain AI inference (CVM). | On-chain inference is a dead end at current gas costs. |
| **Oraichain** | AI oracle. | — |
| **Giza** | zkML (Orion), agents. | — |
| **Sentient** | Open models + **OML** ("Open, Monetizable, Loyal") fingerprinting; GRID; Dobby models; ROMA/Open Deep Search. | Model fingerprinting for attribution is relevant if MeshPool ever wants to track which model produced an accepted candidate. |
| **Sahara AI** | AI blockchain + data services. | — |
| **Vana** | User-owned data DAOs (DLPs), proof-of-contribution. | Data-ownership framing is good marketing; hard economics. |
| **Grass** | Residential **bandwidth** scraping network for AI training data. Solana + a "Sion" rollup, ZK proofs of data provenance. | 🟢 **The best DePIN supply-acquisition story that exists.** Millions of nodes because the ask is *zero effort* (a browser extension using idle bandwidth). **Lesson: the friction of contribution determines supply, more than the payout does.** MeshPool's contribute flow must be one click, and the machine must be usable while contributing. |
| **Ora Protocol** | **opML** — optimistic ML with a fraud-proof challenge window; onchain AI oracle; IMO. | 🟢 opML is the honest answer to "zkML is too expensive": assume correct, allow challenge, bisect on dispute. **MeshPool already has `InteractiveBisectionDispute`.** This is the same design and it is the right one. |
| **Commune AI** | Permissionless module network. | — |

## 3.5 Group E — Edge, mobile, gaming

| Project | What it is | Lesson |
|---|---|---|
| **Acurast** | **Mobile phones as TEE compute nodes** — uses ARM TrustZone / Android hardware attestation. Processors execute deployments; Acurast Cloud. | 🟢 **Genuinely unusual and worth studying.** Hardware-backed attestation on commodity phones is a *free* sybil resistance primitive (attestation proves a real Google-certified device). If MeshPool ever needs sybil resistance without KYC, **Android Key Attestation / Apple DeviceCheck are the cheapest strong signal available.** Note: MeshPool already contemplates binding Ed25519 enrollment to a hardware-timing challenge — attestation is strictly better where available. |
| **NodeOps** | Node/infra automation network. | — |
| **CUDOS / Intercloud** | Cloud aggregation; merged into ASI. | — |
| **NeurochainAI** | Inference DePIN. | — |
| **Inferix** | Rendering DePIN, **VerityFlow** verification. | Another rendering-verification design — same family as Render. |
| **GAIMIN / DATS / Titan / Raiinmaker** | Gaming/mobile idle-resource DePIN. | Very high churn, very low reliability, very cheap. Only viable for *stateless, tiny, retryable* units — i.e. exactly the Quest sample shape. |
| **Heurist** | Decentralized AI inference/serverless (imagegen, LLM). | — |

## 3.6 Group F — Neoclouds (the centralized competition)

These 26 are not competitors to the pivot; they are **suppliers**. MeshPool should be able to
burst onto them.

| Tier | Providers | Characteristics |
|---|---|---|
| **Hyperscale-adjacent** | CoreWeave, Nebius, Crusoe, Lambda, Voltage Park, Nscale | Big H100/H200/GB200 fleets, InfiniBand, multi-year contracts. CoreWeave is Kubernetes-native (SUNK = Slurm-on-Kubernetes, Tensorizer for fast model loading). |
| **Mid-market** | RunPod, DataCrunch, Hyperstack, Ori, Genesis Cloud, Northflank, Cirrascale, Paperspace | On-demand + spot, per-second billing, container-first. RunPod's serverless + network volumes are the best DX of the group. |
| **Aggregators / brokers** | Shadeform, TensorDock, FluidStack, CloudRift, Massed Compute, Sesterce, Qubrid, JarvisLabs | **Shadeform is the interesting one** — a single API across many clouds. This is what MeshPool's "burst" adapter should look like. |
| **Generalist clouds' GPU SKUs** | Vultr, Scaleway, OVHcloud, DigitalOcean Gradient | Convenient, rarely cheapest. |

**The unit-economics reality** `[DERIVED]`, and it is the reason not to compete here:

- An H100 at `[CITED]` ~$2/hr delivers ~1,000 TFLOPS FP16 (sparse ~2,000) with 3.35 TB/s HBM
  and 400–800 Gbps of InfiniBand.
- An RTX 4090 delivers ~165 TFLOPS FP16, 1.0 TB/s VRAM, 24 GB, and a **20 Mbps home uplink**.
- Electricity: 450 W × $0.20/kWh = **$0.09/hr** `[DERIVED]`.
- Therefore a 4090 must clear ~$0.09/hr just to break even on power, before amortizing the
  card, before churn, before verification overhead, before the coordinator's costs.
- Market rate for 4090s on Vast is `[CITED]` roughly $0.20–0.35/hr.
- **Gross margin on reselling idle consumer GPU-hours is thin, volatile, and requires
  near-100% utilization to matter.** Nobody has made this work at scale except Vast (who
  takes a cut and holds no inventory) and Salad (who targets workloads nobody else wants).

**Conclusion: never sell GPU-hours. Sell accepted solutions.**

## 3.7 Group G — Serverless GPU / inference platforms

| Provider | Key technical innovation | Lesson |
|---|---|---|
| **Modal** | The most impressive engineering in the group: a **Rust container runtime**, **gVisor** sandboxing, **memory snapshotting / checkpoint-restore** for sub-second cold starts, and a **lazy-loading container filesystem** so a 10 GB image starts before it's downloaded. Their scheduler is genuinely a distributed OS. | 🟢 **Steal the cold-start playbook.** MeshPool's equivalent problem is "a node needs a 20 GB model it doesn't have." Lazy-loading + snapshotting + content-addressed dedup is the answer. Modal proves it's possible. |
| **Baseten** | Truss (model packaging), Chains (multi-step), multi-cloud capacity management, Briton/TensorRT-LLM inference stack. | Capacity brokerage across clouds is a real moat. |
| **Fal.ai** | Diffusion-optimized inference; extremely fast image gen. | Verticalization wins. |
| **Replicate** | Cog (container spec for models); the public model registry. | **Cog is the de-facto packaging standard.** MeshPool's Quest generators should be Cog/OCI-compatible where possible. |
| **Together AI** | Full-stack: kernels (FlashAttention lineage), inference engine, fine-tuning, and a GPU cloud. | See §3.2 — the decentralized-training team that became a neocloud. |
| **Fireworks / Anyscale / Lepton (→NVIDIA) / Cerebrium / Beam** | Fast OSS-model serving, per-token pricing, autoscaling. Anyscale = managed Ray. | Token prices have fallen `[CITED]` ~100–1000× in two years. **Do not build a business on $/token.** |
| **Banana.dev** | **Shut down.** | 🔴 **Read the postmortem.** Serverless GPU unit economics are brutal: you eat idle GPU cost between customer requests, cold starts are seconds-to-minutes, and customers churn to whoever is 20% cheaper. **A commodity inference API is a bad business.** |
| **Chutes.ai** | Serverless inference running on **Bittensor subnet 64**. | The most credible "decentralized serverless" — but its cheap prices are widely argued to be **subsidized by TAO emissions**, not by unit economics. When emissions fall, so does the price advantage. **Do not build on a subsidy.** |
| **Fluence** | "DePIN cloud" — CPU-first serverless. | — |

## 3.8 The graveyard `[CITED]` — and the single cause of death

| Project | Status | Cause of death |
|---|---|---|
| SETI@home | Shut down 2020 | Scientific goal exhausted; no economy |
| Folding@home | Collapsed from a **2.4 exaFLOPS** COVID peak | The *mission* ended. Contributors left. Zero reciprocity for donors |
| Banana.dev | Dead | Serverless GPU unit economics |
| Golem | Alive, marginal, 9 years | **No workload** |
| Petals public swarm | Effectively dormant | **No incentive to donate** |
| Gridcoin / Curecoin | Marginal | Tokenizing BOINC didn't create demand |
| SingularityNET marketplace | Marginal | No killer workload |
| NuNet, HyperCycle, DBC, many others | Marginal | Substrate without demand |

> **Every single decentralized-compute failure has the same cause of death: they built a
> supply-side substrate and waited for demand that never came.**
>
> The two survivors — **Vast.ai** (no verification, the buyer checks) and **Render** (a
> verifiable vertical) — both started from a *workload*, not a substrate.
>
> **This is the most important sentence in Part 3.** MeshPool has spent its life building a
> world-class substrate. The pivot is not about building more substrate; it is about finally
> naming the workload.

---

# PART 4 — RESEARCH FOUNDATIONS

> The papers and results the new architecture stands on. **Re-verify every number before
> betting on it.** arXiv IDs and links are given so the next session can fetch primaries.

## 4.1 Test-time compute scaling — the load-bearing result

| Work | Core claim | Why it matters here |
|---|---|---|
| **Large Language Monkeys: Scaling Inference Compute with Repeated Sampling** — Brown, Juravsky, Ehrlich, Clark, Le, Ré, Mirhoseini (arXiv **2407.21787**, Jul 2024) 🟢 `[VERIFIED-EXT]` arxiv.org/html/2407.21787v2, fetched this session | **Coverage** (fraction of problems solved by ≥1 of k samples) grows **nearly log-linearly over several orders of magnitude**, modelled as an **exponentiated power law**. Verified headline numbers: **DeepSeek-Coder-V2-Instruct on SWE-bench Lite: 15.9% @1 → 56% @250 samples**, against a single-attempt SOTA of **43%** (a GPT-4o + Claude-3.5-Sonnet mixture) — **exceeding SOTA by 13 points using a weaker model**. Also **Gemma-2B on CodeContests: 0.02% @1 → 7.1% @10,000 = a 300× coverage increase.** | 🔴 **The single most important external result for the pivot**, now fully verified. *A cheap model plus many samples plus a verifier beats an expensive model with one sample.* That is a mesh-shaped statement. |
| **Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters** — Snell, Lee, Xu, Kumar (arXiv **2408.03314**, Aug 2024) 🟢 `[VERIFIED-EXT]` arxiv.org/html/2408.03314v1, fetched this session | Difficulty-adaptive allocation of the test-time budget **nearly outperforms best-of-N using up to 4× less test-time compute** (16 vs 64 generations). Critically: **beam search wins on *harder* questions and *lower* budgets; best-of-N wins on *easier* questions and *higher* budgets.** On level-5 (hardest) questions, *no* method makes meaningful progress. | Justifies the **portfolio bandit** (§8.3) and forces **difficulty and remaining budget** into the bandit's context vector (§8.11). *How* you spend the budget matters as much as how much. |
| **s1: Simple test-time scaling** — Muennighoff et al. (arXiv **2501.19393**, 2025) `[CITED]` | "Budget forcing": simply forcing longer/shorter thinking traces gives clean test-time scaling with 1k training examples. | Cheap knob. Add "thinking budget" as a portfolio dimension. |
| **Let's Verify Step by Step** — Lightman et al., OpenAI (arXiv **2305.20050**) `[CITED]` | Process reward models (PRMs) beat outcome reward models for selecting among samples. | Relevant for *ranking* candidates when the verifier is partial. |
| **Self-Consistency** — Wang et al. (arXiv **2203.11171**) `[CITED]` | Majority vote over sampled chains substantially improves reasoning accuracy. | The fallback selection rule when there is no objective verifier. |
| **OpenAI o1 / o3 and the ARC-AGI results** `[CITED]` | o3 reportedly reached high ARC-AGI scores in a "high-compute" configuration costing **orders of magnitude more per task** than the low-compute mode (widely reported figures on the order of ~$17/task low vs thousands/task high). | Proves the market will pay a very large amount for *one correct answer*. That is a **pay-per-accepted-candidate** market. |
| **DeepSeek-R1 / R1-Zero** (arXiv **2501.12948**) `[CITED]` | GRPO-based RL on verifiable rewards produces strong reasoning without a learned reward model — **because math and code have objective checkers**. | The entire modern reasoning-model recipe is built on *verifiable rewards*. MeshPool's thesis is the same insight applied to a market. |

### 4.1.1 The coverage math you must internalize `[DERIVED]`

Let a single sample from configuration *c* solve a problem with probability `p_c`.
With `k` i.i.d. samples from one configuration:

```
coverage(k) = 1 − (1 − p_c)^k
```

With a **portfolio** of configurations `C` and per-configuration budgets `k_c`, if failures
were independent across configurations:

```
coverage = 1 − Π_c (1 − p_c)^{k_c}
```

Failures are **not** independent in practice — different models fail on the same hard
problems. Model this with a latent problem difficulty `d`:

```
p_c(d) = σ(α_c − d)        # config c's ability α_c vs problem difficulty d
```

The portfolio wins whenever configurations have **decorrelated residual errors** — i.e.
different `α_c` *and* different inductive biases. This is exactly the classic **algorithm
portfolio** result from SAT solving (Gomes & Selman; SATzilla), where a portfolio of diverse
solvers massively outperforms the single best solver.

> **The mesh's structural advantage in one line:** a datacenter runs one model in a big
> batch; a mesh runs *fifty different models at fifty different temperatures with fifty
> different scaffolds*. For search over a rugged landscape, **portfolio diversity beats raw
> throughput**, and diversity is exactly what heterogeneous consumer hardware produces for free.

### 4.1.2 🔴 THE VERIFIER GAP — the finding that scientifically justifies the entire scoping rule

This is the most important thing verified this session, and it is the reason MeshPool's
"refuse non-verifiable workloads" rule is a *scientific* decision rather than a convenience.

`[VERIFIED-EXT]` From Large Language Monkeys §1, on MATH with Llama-3-8B-Instruct:

| Metric | @100 samples | @10,000 samples | Gain |
|---|---|---|---|
| **Coverage** (is a correct answer *anywhere* in the samples?) | 79.8% | **95.3%** | **+15.5 pts** |
| **Realized accuracy** via majority voting / reward-model scoring | 38.7% | **39.8%** | **+1.1 pts** |
| **THE GAP** | 41.1 pts | **55.5 pts** | **grows with scale** |

> The authors' own conclusion: *"methods such as majority voting and using reward models
> plateau with a lower sample budget… building robust verifiers remains an open problem."*

**Read this table twice. It says three things, and all three are load-bearing:**

1. **Repeated sampling always finds the answer.** Coverage goes to 95%+. The generation side
   is *solved* — the answer is in the pile.
2. **Without a verifier, you cannot extract it.** Selection plateaus at ~100 samples and then
   flatlines. Buying more compute buys you *nothing*.
3. **The gap widens as you scale.** So the more compute you throw at an unverifiable problem,
   the more of it you waste. **Scaling compute without a verifier is value-destroying.**

### The strategic consequences

| Consequence | Why |
|---|---|
| 🔴 **The verifier — not the GPU — is the scarce resource in test-time compute.** | Everyone is selling GPUs. The bottleneck is elsewhere. |
| 🔴 **MeshPool's `NOT_VERIFIABLE` refusal is not a limitation. It is the product.** | By construction, MeshPool operates *only* in the regime where the curve keeps climbing (the 95.3% column), and refuses the regime where it flatlines (the 39.8% column). **No competitor draws this line.** |
| 🔴 **This is why 117 GPU marketplaces cannot pivot into this.** | They sell compute by the hour. In the unverifiable regime, hours are worthless; in the verifiable regime, what's being bought is *acceptance*, which they have no mechanism to price. |
| **It explains AlphaCode, AlphaProof, FunSearch, AlphaEvolve, and R1 all at once.** | Every one of them lives in the verifiable regime. There is no counterexample of massive sampling succeeding without a checker. |

> **Put plainly: the entire industry is scaling the axis that saturates (models, GPUs, tokens)
> while the axis that doesn't saturate (verified search) is bottlenecked on a resource nobody
> is building — a market for machine-checkable problems.**
>
> **That market is the invention. See Part 6.**

### 4.1.3 The economic proof, verified

`[VERIFIED-EXT]` Also from Large Language Monkeys §1 — this sentence is the business case:

> *"sampling five times from the weaker and cheaper DeepSeek model solves more issues than
> single attempts from Claude or GPT, while also being **over 3× cheaper**."*

At **5 samples**, holding the agent framework (Moatless Tools) constant, on SWE-bench Lite.
Not 250 samples. **Five.**

**And that measurement was taken against *rented API pricing*.** MeshPool's marginal cost is
*idle-machine electricity*. `[DERIVED]` The published 3× advantage is therefore a **floor**,
not a ceiling, for the mesh's cost structure.

`[VERIFIED-EXT]` One more finding with direct architectural consequence: **the compute-optimal
model size is dataset-dependent** — on MATH, coverage is maximized by a *smaller* model with
*more* attempts; on CodeContests, by a *larger* model. There is no globally best arm.
**This is a formal argument that the arm-selection problem is real and must be learned per
problem class — exactly what §8.3's bandit does.**

## 4.2 LLM + evolutionary search — the killer precedent

| Work | What it did | Numbers |
|---|---|---|
| **FunSearch** — Romera-Paredes et al., DeepMind, *Nature* (Dec 2023/2024) `[CITED]` | LLM proposes *programs*; an **automatic evaluator** scores them; an **island-model** evolutionary loop with a "best-shot" prompt containing top programs so far. **Discovered a new largest cap set construction** — a genuinely new mathematical result — and improved online bin-packing heuristics. | Millions of LLM samples; a *small* fraction of proposals are ever improvements. **The evaluator, not the LLM, is what makes it correct.** |
| **AlphaEvolve** — Balog, Novikov, et al., DeepMind (May 2025) `[VERIFIED-EXT]` en.wikipedia.org/wiki/AlphaEvolve | General-purpose evolutionary coding agent using Gemini. **On 50 open mathematical problems: rediscovered SOTA 75% of the time and *improved* on it 20% of the time**, including advancing the kissing number problem. Also: **recovered 0.7% of stranded Google datacenter compute** via better scheduling heuristics; optimized a **TPU circuit**; sped up a **Gemini training matmul kernel**; found a **4×4 complex matrix multiplication in 48 scalar multiplications** (beating Strassen-derived 49, a 56-year-old record). | Requires **an evaluation function with metrics to optimize, and an initial algorithm**. That's it. That's the whole interface. |
| **OpenEvolve** (`github.com/algorithmicsuperintelligence/openevolve`) `[VERIFIED-EXT]` | Open-source AlphaEvolve. **MAP-Elites + LLMs**, island-based architecture, LLM *ensembles* with fallback, artifact side-channel (error feedback into the next prompt), double selection (different programs for performance vs. inspiration), ring-topology migration, adaptive feature dimensions, **full seeded determinism (default seed=42)**. Claims 2.8× Metal attention kernel speedup on M1 Pro and SOTA-matching circle packing at n=26. | 🔴 **The single most directly relevant open-source project in existence.** |
| **Eureka** (NVIDIA) `[CITED]` | LLM evolutionary search over *reward functions* for RL; beat human-written rewards on 83% of tasks. | Reward-function search is another Quest class. |
| **ADAS** (Automated Design of Agentic Systems) `[CITED]` | Meta-search over agent architectures using an LLM. | Agent-scaffold search is a Quest class. |
| **Promptbreeder / EvoPrompt** `[CITED]` | Evolutionary prompt optimization. | Prompt search is a Quest class. |
| **AlphaCode / AlphaCode 2** — DeepMind `[CITED]` | **Generated up to 1M samples per problem**, filtered on public tests (removing ~99%), **clustered** remaining candidates by behaviour on generated inputs, and submitted 10. Reached **~54.3 percentile** on Codeforces. | 🔴 **The existence proof for the whole thesis.** Massive sampling + filtering + clustering = competitive-programmer-level performance. Nobody productized it *because it was too expensive in a datacenter*. **A mesh of idle machines is exactly the cost structure that makes AlphaCode-style search viable.** |
| **AlphaProof / AlphaGeometry / DeepSeek-Prover-V2 / Kimina-Prover** `[CITED]` | Lean-verified theorem proving with enormous search budgets (AlphaProof reportedly ran for days on a single IMO problem). | **Lean checks a proof in milliseconds; finding it takes days.** The purest verification asymmetry that exists. |
| **AlphaTensor / AlphaDev** `[CITED]` | Search over algorithm space (matrix multiplication, sorting networks) — found faster algorithms now shipped in LLVM's libc++. | Algorithm discovery is a real, monetizable output. |
| **MAP-Elites** — Mouret & Clune (arXiv **1504.04909**) `[CITED]` | Illuminating search: maintain an **archive** of the best solution in each cell of a behaviour-descriptor grid, rather than a single best. Produces diverse, high-performing solutions and escapes local optima. | 🔴 The archive structure for §8.2. |
| **Quality-Diversity / POET / Novelty Search** `[CITED]` | Optimize for novelty and quality jointly; open-ended discovery. | The theoretical frame for "diversity is a resource". |

> ### The gap this reveals
>
> **AlphaEvolve and OpenEvolve are compute-starved orchestrators.** OpenEvolve is a
> *single-process Python loop that calls a paid LLM API*. Its bottleneck is money and
> throughput. AlphaEvolve's bottleneck is Google's internal TPU allocation.
>
> **MeshPool is compute with no orchestrator.**
>
> **The pivot is literally: put the orchestrator on top of the mesh.**

## 4.3 Verification asymmetry — is verification really cheap?

The thesis depends on `cost(verify) ≪ cost(generate)`. Here is the honest table.

| Domain | Generate | Verify | Ratio `[ESTIMATE]` | Verdict |
|---|---|---|---|---|
| **Lean/Coq theorem proving** | minutes–days of search | **milliseconds** (kernel type-check) | 10⁴–10⁹ | ✅ Perfect |
| **SAT/SMT** | exponential | polynomial (check assignment) | huge | ✅ Perfect |
| **Competitive programming** | 10²–10⁶ samples | run test cases, ~ms each | 10³–10⁵ | ✅ Excellent |
| **Unit-test-verified code repair** | 10³–10⁵ tokens (~seconds–minutes) | run the test file (~0.1–10 s) | 10–10³ | ✅ Good |
| **Full CI suite repair** | seconds–minutes | **minutes** (whole suite, container boot) | 0.5–5 | ⚠️ **Cascade required** (§8.7) |
| **Kernel/algorithm optimization** | seconds | benchmark run, seconds–minutes | 1–10 | ⚠️ Cascade + caching |
| **Constrained generation (JSON/grammar)** | tokens | schema validate, µs | 10⁶ | ✅ Trivial |
| **Fuzzing / crash-finding** | run the input | **the crash IS the proof** | ∞ | ✅ Perfect |
| **Molecular docking / protein design** | model forward | scoring function, ms–s | 10–10³ | ✅ Good |
| **Chip layout / EDA** | hours | DRC/LVS/timing, minutes | 1–100 | ⚠️ Cascade |
| **Open-ended writing** | seconds | **subjective — no verifier** | n/a | ❌ **Out of scope** |
| **General chat** | seconds | none | n/a | ❌ **Out of scope** |

> **Scoping rule for MeshPool: if there is no cheap objective predicate, it is not a Quest.**
> This is a *feature*. It is what makes the trust model work. Refuse the workloads that break it.

### 4.3.1 The counter-evidence you must respect: **verifier gaming / reward hacking**

`[CITED]` Well-documented failure modes:
- **Test-suite overfitting** — models special-case the visible tests (`if input == [1,2,3]: return ...`).
- **SWE-bench weak tests** — multiple analyses have found SWE-bench instances where the
  provided FAIL_TO_PASS tests are weak enough that incorrect patches pass, and instances with
  solution leakage in issue text.
- **LLM-as-a-judge unreliability** — position bias, verbosity bias, self-preference.
- **Reward hacking in RLVR** — models discovering degenerate solutions that satisfy the checker.

**Mitigations designed into the architecture (§8.7, §10):**
1. **Held-out verification** — the Quest author supplies a *secret* test set never sent to
   generators. A candidate must pass both public and held-out. This is the direct analogue of
   MeshPool's existing **known-answer spot-check** and it slots into the same machinery.
2. **Anti-tamper static checks** — reject candidates that modify test files, the evaluator,
   or the environment. Cheap and catches the crudest gaming.
3. **Mutation-testing the candidate** — does the candidate still pass when the *tests* are
   mutated? (MeshPool already has a mutation-testing culture and a `mutation_guard`.)
4. **Cross-candidate agreement** — if 40 independent candidates agree on behaviour and 1
   disagrees, that's signal.
5. **Metamorphic testing** — property-based checks the author didn't have to enumerate.

## 4.4 Distributed inference & serving — what to reuse

| System | Technique | Relevance |
|---|---|---|
| **vLLM** `[CITED]` | **PagedAttention** (KV cache in non-contiguous blocks), **continuous batching**, **prefix caching**, chunked prefill, speculative decoding, disaggregated prefill/decode. | The generator node backend of choice on NVIDIA. MeshPool's own `kvcache.py` already implements paged + activation caching — good instincts. |
| **SGLang** `[CITED]` | **RadixAttention** — a radix tree over KV prefixes for automatic cross-request prefix sharing; structured/constrained generation. | 🟢 **Highly relevant.** In Quest sampling, *all N samples share the same long prompt prefix* (the repo context). RadixAttention makes N samples on one node dramatically cheaper than N× one sample. **This is a major efficiency unlock — see §8.8.** |
| **Mooncake** (Moonshot) / **DistServe** / **Splitwise** / **Sarathi-Serve** `[CITED]` | KV-cache-centric disaggregated architecture; separating prefill (compute-bound) from decode (memory-bandwidth-bound); chunked prefill to avoid head-of-line blocking. | Prefill/decode disaggregation maps onto a mesh: heavy-VRAM nodes prefill, many small nodes decode. Worth exploring in v2. |
| **llama.cpp** (+ RPC backend) `[CITED]` | GGUF, K-quants, CPU+GPU hybrid offload, `--rpc` distributed backend. Runs on literally everything. | 🔴 **The correct default generator backend for a consumer mesh.** MeshPool already has a real GGUF loader and tokenizer — it should *drive* llama.cpp rather than reimplement matmuls in NumPy. |
| **MLX / MLX-distributed** `[CITED]` | Apple Silicon unified memory; what exo uses. | Mac Studios are the best consumer inference hardware per watt and per GB. Support them. |
| **PowerInfer / FlexGen / AirLLM / DeepSpeed ZeRO-Inference** `[CITED]` | Sparsity-aware and offload-based tricks to run big models on small hardware. | Relevant to the "large-model arm" of the portfolio. |
| **Ray** (Anyscale) `[CITED]` | GCS, distributed object store, actor model, Ray Serve/Data, autoscaler. | The reference design for distributed task orchestration. MeshPool's batch engine is a simpler version; Ray's *object store* concept (shared immutable objects with lineage) is worth stealing for the candidate archive. |

## 4.5 Determinism & reproducibility

| Work | Finding | Use |
|---|---|---|
| **"Defeating Nondeterminism in LLM Inference"** — Thinking Machines Lab (Sept 2025) `[VERIFIED-EXT]` thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ | The common "concurrency + floating point" explanation is **wrong**. Repeated `torch.mm(A,B)` on a GPU *is* bitwise identical. The real culprit is **lack of batch invariance**: kernel reduction strategies change with batch size, so *your* result depends on **who else is in the batch**. Fix: **batch-invariant kernels** (they released `batch_invariant_ops`). | 🔴 **Load-bearing for Replay Receipts (§8.6).** MeshPool's continuous batcher *already* refuses to stack rows into one matmul precisely because "a tenant's output would then depend on who happened to share its batch" — **the same insight, arrived at independently.** That is a strong signal the architecture instinct is right. |
| **RepOps** (Gensyn) `[CITED]` | Reproducible operators: bitwise-identical kernels across heterogeneous hardware. | The path to cross-hardware replay. |
| **vLLM/SGLang determinism flags** `[CITED]` | `--enforce-eager`, seeded sampling, and batch-invariant modes get you *most* of the way. | Practical starting point. |

**Design consequence:** MeshPool cannot promise bit-exact reproducibility across *arbitrary*
heterogeneous GPUs today. So the Replay Receipt promises **the strongest available tier**:

| Tier | Guarantee | When |
|---|---|---|
| **R0 — Acceptance replay** | Anyone can re-run the *evaluator* on the candidate artifact and get the same verdict. | **Always.** This is the one that matters. |
| **R1 — Same-hardware generation replay** | Same node class + same engine build + same seed ⇒ same tokens. | Achievable now with batch-invariant kernels. |
| **R2 — Cross-hardware generation replay** | Any node reproduces the tokens. | Requires RepOps-class work. Research track. |

> **The critical realization:** for Verifiable Solution Mining, **only R0 is required.** We do
> not need to reproduce *how* the answer was found — only to prove the answer is *correct*.
> That is a vastly weaker requirement than every zkML/proof-of-inference project is chasing,
> and it is achievable today. **This is the technical crux of why the pivot works.**

## 4.6 The verification spectrum — the comparison table that justifies the design

| Technique | Overhead | Latency added | Trust assumption | Defeats | Does NOT defeat |
|---|---|---|---|---|---|
| **Full replication (N×)** | N× compute | 0 (parallel) | Honest majority | Random corruption, lazy nodes | Collusion; non-determinism → false accusations |
| **Spot-check / adaptive replication** (BOINC, MeshPool today) | p× compute (p≈0.25) | 0 | Cheating is *detected eventually* | Persistent cheaters | One-shot high-value cheating |
| **Proof of Sampling (PoSP)** (Hyperbolic) | small p, game-theoretic | 0 | Rational adversary | Profit-motivated cheating | Irrational/griefing adversary |
| **Refereed delegation / fraud proofs** (Truebit, Cartesi Dave, Gensyn Verde, opML) | ~1× + O(log N) dispute | Challenge window (mins–days) | ≥1 honest challenger | Any incorrect execution | Liveness attacks; needs determinism |
| **TEE attestation** (Phala, iExec, Oasis, Marlin, Acurast) | ~5–15% (NVIDIA CC) | small | Hardware vendor + no side channels | Malicious host software | **Hardware breaks** — Foreshadow, SGAxe, Plundervolt, ÆPIC, CipherLeaks, WeSee, Heckler, BadRAM, TEE.fail |
| **zkML** (EZKL, Giza, Omron, zkLLM) | **10³–10⁶×** | seconds–hours | Cryptography only | Everything | Being affordable. Not viable for LLM inference in 2026. |
| **TOPLOC / LSH commitment** (Prime Intellect) | ~1× + tiny | small | Locality-sensitive hash is hard to forge | Substituting a different/smaller model | Subtle numerical drift; needs calibration |
| **Similarity consensus** (Crynux) | 3× | 0 | Honest majority + output metric | Random garbage | Correlated/colluding outputs |
| **Validator scoring** (Bittensor) | 1× + validator cost | 0 | **Validators are honest and competent** | Nothing robustly | **Gaming the scorer. This is the failure mode.** |
| 🟢 **Proof-of-Acceptance (MeshPool, new)** | **≈0 extra** — the verifier IS the workload | 0 | **The Quest author's predicate is correct** | Every incorrect candidate, deterministically, for free | A *wrong predicate*; reward hacking within a weak predicate |

> **Read the last row against the rest.** Every other technique pays a tax to answer "did this
> node compute honestly?" Proof-of-Acceptance never asks that question. It asks *"is this
> artifact good?"* — which the buyer wanted answered anyway.
>
> **The verification overhead of the entire network is zero, because verification is the product.**

---

# PART 5 — STRATEGIC DIAGNOSIS: WHY THE CURRENT THESIS CAPS OUT

> This part is deliberately blunt. It is not a criticism of the engineering, which is
> excellent. It is a criticism of the *target*.

## 5.1 Uncomfortable truth #1 — the flagship is a losing race

MeshPool's flagship is "shard a model too big for one machine."

**On a LAN, exo wins decisively** `[VERIFIED-EXT]`: RDMA over Thunderbolt 5, tensor
parallelism (3.2× on 4 devices), MLX, running DeepSeek-v3.1-671B and Kimi-K2-Thinking today.
MeshPool runs NumPy over HTTP/JSON.

**On a WAN, physics wins.** MeshPool's own honest benchmark `[MEASURED]` `docs/benchmarks.md`:

| Deployment | Ceiling (baseline) | Ceiling (SPD) | Note |
|---|---|---|---|
| kimi-k2, 11-node metro constellation | 6.4 tok/s | **15.2 tok/s** | **assuming compute is free** |
| kimi-k2, 11 nodes same country | 3.5 tok/s | 8.4 tok/s | |
| kimi-k2, 11 nodes worldwide | 1.2 tok/s | 3.0 tok/s | |

And the project's own README records that this *demolished two of its own headline claims*:
the plan had projected ~50 tok/s (wrong by 3×), and "bandwidth is a non-issue, latency is
everything" is **false at depth 11** (ten hops of 14 KiB on a 20 Mbps uplink = 57 ms against
100 ms of metro latency).

**Meanwhile a $2,000 single machine runs a 120B MoE at 40+ tok/s.** The "too big for one
machine" window is closing because of MoE sparsity, 4-bit native quantization, and NVMe offload.

> **Verdict: 15 tok/s, best case, for a model people can increasingly run alone.
> There is no market there.**

## 5.2 Uncomfortable truth #2 — the market is a commodity race to zero

117 companies. Essentially all sell GPU-hours or tokens. Token prices have fallen `[CITED]`
~100–1000× in two years. GPU-hour prices are set by the marginal H100 in a Texas datacenter
with 400 Gbps InfiniBand and $0.04/kWh power.

A consumer mesh is **structurally more expensive per useful FLOP** than that datacenter, once
you count:

```
true_cost = electricity
          + amortized hardware
          + bandwidth (residential uplinks are the scarce resource)
          + churn/retry overhead
          + verification overhead
          + coordination overhead
          + support cost of a heterogeneous fleet
```

> **You cannot win a commodity price war from a structurally higher cost base.**
> The only way to win is to sell something the low-cost producer *cannot make*.

## 5.3 Uncomfortable truth #3 — the graveyard is full of better-funded versions of this

Golem (9 years, no workload). Petals (great tech, empty swarm). SETI@home (dead).
Folding@home (2.4 exaFLOPS → collapse when the mission ended). Gridcoin. SingularityNET.
NuNet. Banana.dev.

**Common cause of death: a supply-side substrate waiting for demand.**

MeshPool has, with genuine excellence, built a supply-side substrate. Everything in Part 2's
"genuinely real and good" list is supply-side. The two things that are *not* built are
**a workload people want** and **a reason to donate.**

## 5.4 The one thing the current architecture gets exactly backwards: **coupling**

This is the sharpest technical argument in the document.

Define the **coupling coefficient** κ of a distributed workload as *the fraction of the
workload that is lost when one participating node fails.*

| Workload | κ | Consequence |
|---|---|---|
| **Pipeline-parallel inference across N nodes** | **κ = 1** | Any single node failing breaks **every in-flight request**. Availability = `(1−p)^N`. |
| **Embarrassingly parallel sampling, N nodes** | **κ = 1/N** | A node failing loses **one sample**, which is simply retried. |

### The math `[DERIVED]`

Let `p` = probability a given consumer node drops out during a one-hour window.
Real consumer machines: `[ESTIMATE]` `p ≈ 0.05–0.20/hr` (sleep, reboot, gaming, Wi-Fi, ISP).

**Pipeline of 11 nodes, p = 0.05:**
```
P(pipeline survives 1 hour) = (1 − 0.05)^11 = 0.5688
```
**43% chance the deployment breaks within an hour.** With p = 0.10 it is `(0.9)^11 = 0.314`
— a **69% chance of failure per hour**. MeshPool mitigates this with hot spares and
repartitioning, but repartitioning means reloading gigabytes of weights while users wait.

**Independent sampling, 60-second samples, p = 0.05/hr:**
```
P(a given 60 s sample is lost) = 1 − (1 − 0.05)^(1/60) ≈ 0.000855  →  0.0855%
```
Out of 200 samples, expected losses ≈ **0.17 samples**. Retry cost: negligible.

> ### **The mesh is 1,000× better suited to the second workload than the first.**
>
> MeshPool has spent enormous, high-quality engineering effort (hot spares, repartitioning,
> node classification, Drifter exclusion, constellations, SWIM, straggler guards) **fighting
> κ = 1**. Every one of those mechanisms exists to compensate for having chosen a
> maximally-coupled workload.
>
> **Choose κ = 1/N and most of that difficulty simply evaporates** — and all of that
> engineering becomes *surplus robustness* rather than a desperate necessity.

## 5.5 The three assets that are being wasted

MeshPool has built three genuinely rare things and is pointing all of them at the wrong target.

| Asset | Currently used for | What it is *actually* worth |
|---|---|---|
| **Bit-identical determinism + a testing culture obsessed with non-vacuity** | Proving a shard pipeline equals a full model | **The foundation of a trustworthy result marketplace.** Nobody else in decentralized AI can produce a re-runnable receipt. |
| **A WASM sandbox that safely runs strangers' code on donors' machines** | Running `square(x)` | **The evaluator sandbox** for a global verifier economy. |
| **A complete trust/stake/escrow/dispute/ledger stack with conservation invariants** | Policing four toy built-in functions, because everything user-supplied is `NOT_VERIFIABLE` | **Settlement for a pay-on-acceptance market.** |

## 5.6 What "serious problem, solved perfectly" actually means here

The user's requirement is to solve a serious problem, perfectly, first in the world. Candidate
problems, scored:

| Problem | Serious? | Unsolved? | Mesh-shaped? | Verdict |
|---|---|---|---|---|
| "Run a huge model at home" | Mildly | No (exo, llama.cpp, Macs) | No (κ=1) | ❌ |
| "Cheaper GPU-hours" | Yes | No (117 competitors) | No (higher cost base) | ❌ |
| "Cheaper tokens" | Yes | No (racing to zero) | No | ❌ |
| "AI can't touch my private data" | **Very** | Partially | **Yes** (compute-to-data) | ✅ **Secondary track** |
| "AI compute centralizing into 5 companies" | Yes, but abstract | — | Yes | ⚠️ Mission, not product |
| 🔴 **"Getting a *correct, verified* answer to a hard problem is bottlenecked on parallel search capacity, and that capacity is unaffordable"** | **Yes, acutely** | **Yes** | **Yes, perfectly (κ=1/N, self-verifying, diversity-loving)** | ✅✅✅ **THE ONE** |

> AlphaCode proved 1M samples reaches competitive-programmer level. AlphaEvolve proved
> LLM+evaluator+evolution discovers *genuinely new mathematics and algorithms*. o3 proved
> people will pay thousands of dollars for one right answer.
>
> **And essentially nobody can afford to run these methods**, because in a datacenter the
> compute costs more than the answer is worth to all but Google.
>
> **That is the serious, unsolved, mesh-shaped problem.**

## 5.7 THE BRAINSTORM LEDGER — 24 pivots considered, and why 23 were rejected

> The next session will inevitably ask *"but did you consider X?"* This section exists so that
> question is already answered. Each idea was evaluated against four tests:
> **(S)** Is the problem serious? **(U)** Is it genuinely unsolved? **(M)** Is it mesh-shaped
> (κ=1/N, verifiable, diversity-loving)? **(D)** Is it defensible against a well-funded
> incumbent?

| # | Pivot idea | S | U | M | D | Verdict & reasoning |
|---|---|---|---|---|---|---|
| 1 | **Keep flagship: shard huge models across the internet** | ~ | ✗ | ✗ | ✗ | ❌ **REJECTED.** 15.2 tok/s ceiling with free compute (A3); exo owns LAN; MoE + 4-bit is closing the window. κ=1 is fatal. |
| 2 | **Cheapest GPU-hours marketplace** | ✓ | ✗ | ✗ | ✗ | ❌ 117 competitors, structurally higher cost base. Commodity race to zero. |
| 3 | **Cheapest tokens / inference API** | ✓ | ✗ | ✗ | ✗ | ❌ Token prices fell ~100–1000× in 2 years. Racing a curve to zero. |
| 4 | **Decentralized *training* (à la Prime Intellect / Nous)** | ✓ | ~ | ✗ | ✗ | ❌ Gradient sync is bandwidth-bound even with DiLoCo. Prime Intellect is years and $100M+ ahead. Losing race. |
| 5 | **Federated learning platform** | ✓ | ~ | ~ | ✗ | ❌ Crowded (Flower, NVFlare, OpenFL), enterprise-sales-heavy, and the mesh adds little. |
| 6 | **zkML / verifiable inference proofs** | ✓ | ✓ | ✗ | ~ | ❌ 10³–10⁶× overhead (A18). Not viable in 2026. Would consume the whole roadmap. |
| 7 | **TEE-based confidential inference** | ✓ | ~ | ~ | ✗ | ❌ Nine published TEE breaks (Part 3 §D). Betting the company on Intel/AMD/NVIDIA silicon holding. Phala/Oasis/Marlin already there. |
| 8 | **Decentralized storage / CDN** | ✓ | ✗ | ✓ | ✗ | ❌ Filecoin/Storj/Arweave/S3 own it. MeshPool's storage becomes internal plumbing instead. |
| 9 | **Edge inference for latency (CDN-for-AI)** | ✓ | ~ | ~ | ~ | ⚠️ Real, but Cloudflare/Fastly/Akamai win on POPs. Not mesh-shaped — it's a real-estate business. |
| 10 | **Privacy-preserving compute-to-data** | ✓✓ | ~ | ✓ | ~ | 🟡 **PARTIALLY ADOPTED** — became Track B (MeshPool Private, §6.6). Rejected as the *headline* because the buyer wants an outcome, not a deployment topology. |
| 11 | **Personal/home AI OS ("your own private Jarvis")** | ~ | ✗ | ✗ | ✗ | ❌ Ollama/LM Studio/Jan own this and it's free. No network needed ⇒ no MeshPool. |
| 12 | **Agent marketplace / agent economy** | ~ | ✗ | ~ | ✗ | ❌ Extremely crowded (Fetch.ai, SingularityNET, Morpheus, Olas, ERC-8004). Verification is subjective ⇒ Bittensor's failure mode. |
| 13 | **Crypto token + subnet incentive network (Bittensor-style)** | ~ | ✗ | ~ | ✗ | ❌ Subjective validator scoring is gameable by construction. Also poisons the enterprise track. |
| 14 | **Render-style vertical: distributed 3D/video rendering** | ✓ | ✗ | ✓✓ | ✗ | ❌ Render Network owns it and is profitable. **But it was the key inspiration** — proof-of-render *is* a verifier. §6 generalizes it. |
| 15 | **Distributed scientific computing (protein/materials/climate)** | ✓✓ | ~ | ✓✓ | ~ | 🟡 **ADOPTED AS A QUEST CLASS**, not the product. Folding@home shows supply is free but revenue is ~0. Perfect mission fuel, bad standalone business. |
| 16 | **CI/CD compute for OSS (distributed test running)** | ✓ | ~ | ✓ | ✗ | ❌ GitHub Actions/BuildJet/Depot/Namespace own it; margins are thin. **But it revealed the wedge**: the customer's CI *is already a verifier*. |
| 17 | **Distributed fuzzing / security research network** | ✓✓ | ✓ | ✓✓ | ✓ | 🟡 **ADOPTED AS A QUEST CLASS (P4-4).** Rejected as headline for governance reasons (§10.6) — "find me a 0-day" is a perfect Quest and a legal minefield. Great as class #4, terrible as class #1. |
| 18 | **Distributed theorem proving / math discovery network** | ✓✓ | ✓✓ | ✓✓ | ✓ | 🟡 **ADOPTED AS A QUEST CLASS (P4-3).** The purest verification asymmetry that exists (A: ms to check, days to find). Rejected as headline: TAM is ~5,000 people. **Best possible credibility play, worst possible beachhead.** |
| 19 | **Distributed RL environment/rollout network** | ✓✓ | ✓ | ✓✓ | ~ | ⚠️ **Strong runner-up.** Rollouts are κ=1/N and verifiable. Rejected as headline only because **Prime Intellect is already executing it well** (`verifiers`, Environments Hub, INTELLECT). **Decision: interoperate rather than compete — see P4-8.** Revisit if their execution falters. |
| 20 | **Distributed hyperparameter / NAS search** | ✓ | ✗ | ✓✓ | ✗ | ❌ Perfectly mesh-shaped but Ray Tune/Optuna/W&B own it, and the compute is small. **Adopted as Quest class P4-6.** |
| 21 | **"AlphaEvolve as a service" (centralized, API-based)** | ✓✓ | ✓✓ | — | ✗ | ⚠️ Great product; **no moat**. OpenEvolve (7.3K★) is free, and any lab can copy it in a month. **The moat only appears when you own the compute layer underneath it.** ⇒ This is precisely why the pivot is *evolution + mesh*, not evolution alone. |
| 22 | **Sell "verified RL training data" as the product** | ✓✓ | ✓✓ | ✓✓ | ✓✓ | 🟢 **ADOPTED AS THE SECOND-ORDER BUSINESS (§13.6).** Not the beachhead (no data on day one), but plausibly the largest long-run value. The network generates it as exhaust. |
| 23 | **A benchmark/eval company ("Verifier Hub")** | ✓ | ~ | — | ~ | 🟡 **FOLDED IN** as the public archive + leaderboard (P4-12). Too small alone; excellent as a moat component and as marketing. |
| 24 | 🔴 **VERIFIABLE SOLUTION MINING** — pay-per-accepted-solution search over a mesh | ✓✓ | ✓✓ | ✓✓ | ✓✓ | ✅ **SELECTED.** The only idea passing all four tests. Absorbs the best of #10, #14, #15, #17, #18, #19, #20, #21, #22, #23 as *classes and tracks* rather than competing products. |

### 5.7.1 Why #24 beats its two closest runners-up

| vs. | Their strength | Why #24 wins |
|---|---|---|
| **#19 Distributed RL rollouts** (Prime Intellect) | Real workload, real funding, real traction, correct κ | They sell to **~50 AI labs**. #24 sells to **every developer with a failing test** and **every company with a CI system**. Same substrate, ~10,000× the TAM. And #24 can *consume* their environment format (P4-8), so their ecosystem becomes MeshPool supply. |
| **#21 AlphaEvolve-as-a-service** (OpenEvolve) | Proven algorithm, immediate utility, hot space | It is a *loop with no compute and no settlement*. Its cost scales linearly with paid API calls, so it can never be cheap. #24 is that loop **fused to a compute layer whose marginal cost is idle electricity** and **a settlement rule that makes results trustless**. OpenEvolve is a component of #24, not a competitor to it. |

> **The synthesis in one sentence:** #24 is #21's algorithm, running on #1's substrate,
> settled by #14's insight, funded by #22's exhaust, made safe by #10's architecture, and
> seeded by #15's mission.

## 5.8 The new competitive landscape (who we actually compete with *after* the pivot)

Part 3 analyzed the *old* competitor set. Under the new positioning, the competitors change
completely. **The next session must not fight the last war.**

| Competitor | What they do | Where they beat MeshPool | Where MeshPool wins | Threat |
|---|---|---|---|---|
| **"Just call Claude/GPT 200 times in a bash loop"** | The zero-effort baseline | Trivial. Works today. No install. | Cost (A26 / §4.1.3: ≥3× even at API prices, far more at idle-machine cost); no receipts; no portfolio diversity; no archive; no privacy; no evolution | 🔴 **HIGHEST. This is the real competitor.** Every pitch must beat it explicitly and numerically. |
| **OpenEvolve** (7.3K★) `[VERIFIED-EXT]` | Open-source AlphaEvolve; MAP-Elites + LLM islands | Mature algorithm, free, community | No compute layer, no settlement, no verifier-strength safety, single-process | 🟠 High — **but adopt it as an input, don't compete.** Consider contributing upstream and interoperating. |
| **GEPA** (6.3K★, Aug 2025) `[VERIFIED-EXT]` | Reflective **Genetic-Pareto** optimization of prompts/code from execution traces | Strong algorithm; DSPy ecosystem; Pareto selection is genuinely good | Same gap: an optimizer with no compute layer and no market | 🟠 High as an *algorithm* — **steal the reflective-Pareto parent selection into §8.2.5.** |
| **SWE-agent / OpenHands / Aider / Cursor / Devin / Copilot CLI** | Single-agent code fixing | Enormous polish, distribution, UX, trust | They take **one careful attempt**; MeshPool takes **200 cheap ones and keeps only what passes**. Complementary: their scaffold can be a MeshPool *arm* | 🟡 Medium — **partner surface, not a battlefield.** Wrap them as scaffolds. |
| **Prime Intellect** | Decentralized RL, environments hub, `verifiers` | Funding, talent, real decentralized training | Sells to labs, not developers; datacenter GPUs; RL-training-shaped not solution-shaped | 🟠 High — **interoperate (P4-8).** They could pivot into this; they are the most likely to. |
| **Google DeepMind (AlphaEvolve)** | The real thing, internally | Better in every technical dimension | It is **not a product**, is not available, and runs on TPUs nobody else can rent | 🟡 Medium — validates the category more than it threatens |
| **A frontier lab shipping "high-compute mode"** | o3-style pay-more-get-better | Model quality; trust; distribution | Datacenter pricing; no receipts; no private-fleet option; no portfolio diversity across *vendors* | 🟠 **High and likely (R4).** Mitigation: MeshPool's portfolio can *include* their API as an arm — be the layer above, not the competitor below. |
| **GitHub Actions / Depot / Namespace / BuildJet** | CI compute | Distribution, integration | They run *your* tests; MeshPool *generates candidates* until the tests pass | ⚪ Low — adjacent, and a distribution partner |
| **117 DePIN/neocloud names (Part 3)** | GPU-hours | Cheaper raw compute | They sell hours; MeshPool sells accepted solutions. **They cannot price acceptance** (§4.1.2) | ⚪ Low — **potential suppliers, not competitors.** A neocloud GPU can be a MeshPool arm. |

> 🔴 **The single most important line in this table:** the primary competitor is **a bash loop
> around a frontier API.** Not a company. Every design decision, benchmark, and marketing
> claim must be measured against that baseline, because that is what a skeptical engineer will
> actually compare against. **If MeshPool cannot beat `for i in {1..200}; do claude ...; done`
> on cost *and* acceptance rate, nothing else in this document matters.**

---

# PART 6 — THE INVENTION: VERIFIABLE SOLUTION MINING

## 6.1 One-sentence definition

> **MeshPool is a global, permissionless network that turns idle machines into a *search
> engine over solution space*: you post a problem and a machine-checkable evaluator, the mesh
> evolves candidate solutions in parallel across thousands of heterogeneous machines, and you
> pay only for candidates the evaluator accepts.**

## 6.2 The naming system

| Term | Meaning | Replaces / extends |
|---|---|---|
| **Quest** | The unit of demand: `(problem spec, evaluator, budget, bounty, visibility, portfolio hints)` | new |
| **Candidate** | One proposed artifact (patch / program / proof / config / design) | extends `WorkUnit` |
| **Evaluator** | A deterministic, sandboxed, content-addressed program that maps a Candidate to `(accepted: bool, score: float, behaviour_descriptor: vec, artifacts)` | extends `functions/wasm.py` |
| **Verdict** | The evaluator's signed output | new |
| **Proof-of-Acceptance (PoA)** | The settlement rule: payment iff Verdict.accepted, replayable by anyone | extends `verification.py` |
| **Archipelago** | The distributed island-model quality-diversity search controller | new; runs on Constellations |
| **Island** | One evolutionary population, pinned to a Constellation (low-RTT node cluster) | **reuses `Constellation`** |
| **Archive** | The MAP-Elites grid of behaviourally-diverse elite candidates per Island | new; stored in the blob store |
| **Generator** | A node running an LLM that produces candidates | extends node `services` |
| **Portfolio** | The set of `(model, quant, adapter, temperature, scaffold, thinking-budget)` arms | new |
| **Adapter Swarm** | Distributing LoRA adapters over shared base models to create diversity cheaply | **reuses Weight Swarm + `inference/lora.py`** |
| **Replay Receipt** | The signed, content-addressed record that lets anyone re-derive the verdict | extends the correctness oracle |
| **Verification Cascade** | The cheap→expensive ladder of checks a candidate must climb | **reuses `batch/dag.py`** |

## 6.3 The five mechanisms (the actual invention)

The invention is not any one of these; it is that **all five compose**, and each one converts
a classic weakness of decentralized compute into a strength.

### Mechanism 1 — **The unit of work is a refutable claim, not a container**

Every Quest carries its own acceptance predicate. A node is not paid for "running a container
for 43 seconds"; it is paid for "producing an artifact that made `pytest -x` exit 0 and also
passed a held-out suite the node never saw."

**Converts:** *the trust problem* → *nothing*. There is no result to verify, only an artifact
to accept.

### Mechanism 2 — **Proof-of-Acceptance settlement**

```
pay(node) ⟺ Verdict(candidate).accepted ∧ replayable(Verdict)
```

- Verification overhead ≈ **0**, because the evaluator run *is* the deliverable.
- A lying/lazy node earns **nothing** and self-identifies at zero cost to the network.
- No staking is *required* for correctness (staking is retained only to price *liveness* and
  to deter resource-wasting spam).
- **Contrast with every DePIN network:** they pay for *claimed capacity*, which is why
  io.net got fake GPUs. PoA pays for *externally checkable outcomes*, which cannot be faked
  because the buyer's own predicate decides.

**Converts:** *the incentive problem* → *self-cleaning market*.

### Mechanism 3 — **Diversity as a first-class scheduling resource (Portfolio Bandit)**

The scheduler does not maximize throughput. It maximizes
**P(at least one accepted candidate | budget)** — a *coverage* objective, which is
**submodular**, so greedy allocation carries a `(1 − 1/e)` approximation guarantee. Arms are
`(model, quant, adapter, temperature, scaffold, thinking-budget)` and their success
probabilities are learned online (Thompson sampling), conditioned on a *problem embedding* —
directly inspired by Allora's "forecast other workers' losses" idea.

**Converts:** *heterogeneity* → *the product*.

### Mechanism 4 — **Archipelago: distributed quality-diversity search over Constellations**

Sampling i.i.d. is wasteful — AlphaCode found ~99% of samples were filtered out and survivors
clustered into few behaviours. Instead:

- Each **Island** = one Constellation (a low-RTT cluster MeshPool *already forms*).
- Each Island maintains a **MAP-Elites archive** over a behaviour-descriptor space derived from
  *partial* evaluator signals (which tests passed, runtime, memory, code size, AST shape).
- Islands evolve semi-independently and **migrate elites on a ring**, on a schedule tuned to
  inter-Island RTT — so migration bandwidth is tiny and latency-tolerant. **Islands are the
  perfect decentralization primitive: they need almost no communication by design.**
- Prompts are built from the archive (best-shot + diverse-shot), plus an **artifact
  side-channel** (the previous candidate's error output), which OpenEvolve found essential.

**Converts:** *network partitioning and latency* → *an algorithmic advantage* (island models
are known to outperform single panmictic populations precisely *because* they're partitioned).

### Mechanism 5 — **Replay Receipts (R0)**

Every accepted candidate ships:
```
{ quest_id, evaluator_cid, candidate_cid, env_cid, verdict, 
  generator_attestation{model_cid, adapter_cid, quant, seed, sampler, engine_build},
  node_signature, coordinator_signature, timestamp }
```
Anyone can re-run `evaluator_cid(candidate_cid)` in `env_cid` and get the same verdict.

**Converts:** *"can I trust a decentralized network's output?"* → *"here, check it yourself in
30 seconds."* No zkML. No TEE. No consensus. **This is why the design is deployable in 2026
when every proof-based competitor is not.**

## 6.4 Why each of the four fatal flaws dies

| Fatal flaw | Mechanism that kills it | Residual risk |
|---|---|---|
| **Interconnect (100,000× gap)** | κ = 1/N. Zero inter-node traffic during generation. Only tiny archive migrations. | Weight distribution is the only bandwidth cost — solved by residency + Adapter Swarm (§8.4–8.5). |
| **Trust** | PoA. The evaluator decides; the network never adjudicates correctness. | A *weak* evaluator (reward hacking) — mitigated in §8.7/§10. |
| **Churn** | κ = 1/N. A lost sample is retried; nothing is degraded. | Long evaluator runs need checkpointing. |
| **Heterogeneity** | Portfolio diversity is the objective function. | Weight residency fragmentation — solved by §8.4. |

## 6.5 Why this is defensible (the moat)

1. **A two-sided network effect on *problems*, not compute.** Every solved Quest adds to a
   public archive of `(problem, evaluator, accepted solution, replay receipt)`. That archive
   is (a) training data for better generators, (b) a benchmark, (c) a prior for the portfolio
   bandit ("this kind of problem is best attacked by this arm"). **The archive compounds.
   GPU-hours do not.**
2. **The portfolio bandit's learned priors are proprietary and improve with volume.** Knowing
   *which arm wins on which problem class* is a real, defensible asset — it is exactly what
   Allora is trying to build, applied to a market that actually has a ground truth.
3. **Supply acquisition cost ≈ 0 because of the mission.** See §13. Folding@home got 2.4
   exaFLOPS for free. Mercenary DePIN pays $0.10/hr and gets mercenary supply.
4. **A hard technical core** — the evaluator sandbox, determinism, cascade scheduling, and
   the archive — that takes years to build well, and MeshPool already has 85% of it.

## 6.6 The dual-track product

The same binaries, two postures — **exactly the existing `scale=` preset pattern.**

### Track A — **MeshPool Open** (`scale=public`)
The permissionless network. Public Quests: open-source bugs, benchmark tasks, math conjectures,
kernel optimization, protein/materials search, fuzzing. Anyone contributes machines; anyone
posts Quests. **Mission-driven supply. This is the acquisition engine and the moat.**

### Track B — **MeshPool Private** (`scale=office|enterprise`)
The same software running *entirely inside a company's own fleet* — their laptops, build farm,
idle CI runners, on-prem GPUs. **The code never leaves. The evaluator is their real test
suite. The generators are their own machines or their own licensed model endpoints.**

This is the revenue track, and it solves the privacy problem structurally rather than
cryptographically:

> **You don't need to trust a stranger with your source code if the strangers only ever run
> *your* machines.** And you don't need to prove a stranger computed honestly if your own
> test suite decides.

Enterprise value proposition: *"Your 400 developer laptops are idle 16 hours a day. That's
6,400 machine-hours a night. Point them at your flaky test backlog, your dependency upgrades,
your performance regressions, and your Sonar findings. Wake up to verified patches."*

`[ESTIMATE]` This is a genuinely large, unserved market, and it is **the thing MeshPool's
existing code is already closest to shipping.**

### Track C (later) — **Bounty Market**
Public Quests with real bounties. A bridge between the two: an open-source maintainer posts
"$200 for a patch that makes these 3 tests pass without breaking the other 4,000." Escrow +
PoA + replay receipt = trustless bounty settlement **with no human judging**, which is the
thing that has always killed bounty platforms.

## 6.7 The category name

**Verifiable Solution Mining.**

| | Proof of Work | Proof of Stake | Proof of Capacity | **Proof of Acceptance** |
|---|---|---|---|---|
| Pays for | Wasted hashes | Capital | Claimed disk | **Verified solutions** |
| Verification cost | 1 hash | 0 | Challenge protocol | **0 — it's the workload** |
| Gameable by | 51% hashrate | 51% stake | **Faking capacity** | **Only a bad predicate** |
| External value | None | None | Storage | **The solved problem** |

> **The pitch:** *Bitcoin burns electricity to prove nothing. MeshPool burns electricity to
> prove theorems, fix bugs, and discover algorithms.*

## 6.8 🔴 THE META-QUEST — the network that evolves itself

> This is the most out-of-the-box idea in the document, it has a direct existence proof, and
> it is the thing that could make MeshPool genuinely *first in the world* rather than merely
> first to market.

**The existence proof:** `[VERIFIED-EXT]` AlphaEvolve was pointed at **Google's own
infrastructure** and recovered **0.7% of Google's entire fleet compute** by evolving a
scheduling heuristic, sped up a **Gemini training kernel**, and improved a **TPU circuit**.
DeepMind used its evolutionary search engine to improve the substrate that runs its
evolutionary search engine.

**MeshPool can do the same thing — and unlike Google, MeshPool's substrate is *made of*
exactly the kind of small, benchmarkable, machine-checkable heuristics that evolutionary
search is best at.**

### 6.8.1 The self-improvement surface

Every one of these is a function with a clear input, a clear output, and **an existing
simulation harness that can score it** — because `sim/scenarios/` already exists:

| Target | Current implementation | Evaluator (already exists or is trivial) | Metric |
|---|---|---|---|
| **Constellation formation heuristic** | `build_constellation_index` | The 80-gate sim under synthetic topologies | Pipeline throughput / stability |
| **Node classification thresholds** | `controlplane/nodeclass.py` | Replay of recorded churn traces | Misclassification rate |
| **Cascade stage ordering** (§8.7) | Hand-ordered by `c/(1−p)` | Replay of recorded candidate streams | Expected cost per verdict |
| **Bandit arm-similarity function** (§8.3.2) | Hand-specified | Historical Quest outcomes | Regret vs. oracle |
| **MAP-Elites descriptors** (§8.2.2) | Hand-designed per class | QD-score on solved Quests | Archive coverage + QD score |
| **Prompt templates** (§8.2.3) | Hand-written | A/B on the benchmark | Acceptance rate |
| **Migration topology & period** (§8.2.4) | Ring, RTT-scaled | Multi-island sim | Time-to-first-acceptance |
| **Residency planner** (§8.4) | Greedy facility-location | Simulated 200-node mesh | Bytes per accepted solution |
| **Straggler/spare policies** | Hand-tuned | Existing gates | Tail latency |
| **Speculative-decode draft policy** | `inference/spd.py` | Existing SPD bench | Accept rate × speedup |

### 6.8.2 Why this is a strategic weapon, not a gimmick

```
More nodes  →  more search capacity
            →  better self-evolved heuristics
            →  more efficient network
            →  cheaper per accepted solution
            →  more demand
            →  more nodes
```

**Four compounding properties no competitor has:**

1. **The improvements are *verifiable by construction*.** A new scheduling heuristic either
   beats the old one on the existing gate suite or it doesn't. **MeshPool's 80 acceptance
   gates are already a fleet of evaluators — they were built for regression testing, and they
   turn out to be exactly the fitness functions a Meta-Quest needs.** `[DESIGN]` This is a
   remarkable accident of the existing codebase's culture, and it should be exploited hard.
2. **It is free marketing that no one can fake.** *"MeshPool v0.9's scheduler was written by
   MeshPool v0.8, and here is the receipt"* is an extraordinary claim that is trivially
   auditable — which makes it credible.
3. **It compounds while you sleep.** Idle network capacity has an infinitely-elastic internal
   customer. **The network is never idle: unallocated capacity automatically flows to
   Meta-Quests.** This also solves the "what do nodes do between Quests?" problem, which every
   volunteer network has.
4. **It is a moat that grows with network size.** A competitor forking the code gets today's
   heuristics; they don't get the search capacity that produced them, nor the archive of what
   was already tried.

### 6.8.3 The rules that keep it safe

`[DESIGN]` A self-modifying distributed system is obviously dangerous. Hard constraints:

| Rule | Why |
|---|---|
| **Meta-Quests may only touch designated `@evolvable` pure functions.** | No evolving the transport, crypto, settlement, or sandbox. Ever. Mark the surface explicitly in code. |
| **A Meta-Quest candidate must pass the *entire* existing gate suite, not just its own metric.** | Prevents the classic "optimize throughput by breaking correctness" hack. This is the §10.2 held-out principle applied to the network itself. |
| **Mandatory human review + signed release before any evolved heuristic ships.** | The network proposes; humans dispose. Non-negotiable. |
| **Shadow-mode first.** | Run the evolved heuristic in parallel against the incumbent on live traffic, measure, then promote. |
| **The full receipt is published** for every evolved change. | The audit trail *is* the marketing. |
| **Never evolve the evaluator, the verifier-strength scorer, or the coupling-law cap.** | Otherwise the system evolves its way out of its own safety constraints. This is the single most important rule on this list. |

> **Sequencing:** this is a **Phase 4+** capability (P4-13), gated on Archipelago being proven
> (M3). Do not attempt it earlier — a self-improving loop built on an unproven search engine
> compounds *errors*. But **design the `@evolvable` marker and the metric-harness interface in
> Phase 2**, because retrofitting them later is expensive.

## 6.9 Why a frontier lab won't simply do this

The obvious objection: *"OpenAI/Anthropic/Google will just ship this."* They will ship
*something* like it (R4). Here is the honest analysis of why the position is still defensible:

| Their constraint | Consequence |
|---|---|
| **They sell tokens; this sells outcomes.** Pay-per-accepted-solution *cannibalizes* their per-token revenue and makes their margin legible. | Strong internal incentive not to. Classic innovator's dilemma. |
| **They must use their own model.** | No cross-vendor portfolio. **Portfolio diversity across *competing* vendors is something only a neutral party can assemble** — and §4.1 says diversity is the whole game. |
| **They cannot run inside your firewall on your idle laptops.** | Track B is structurally closed to them. |
| **Their marginal cost is datacenter GPU time.** | They can never reach idle-machine electricity pricing. |
| **They will not publish replayable receipts** that let you verify their work independently. | The trust primitive is unavailable to them for commercial reasons. |
| **Their unit of sale is a conversation, not a bounty.** | Their whole billing, product, and safety stack assumes interactive usage. |

> **The defensible position: be the neutral, multi-vendor, verifiable, deployable-anywhere
> layer *above* the models.** MeshPool should treat frontier APIs as **arms in its portfolio**
> (§8.3.3), not as competitors. When they ship high-compute mode, MeshPool adds it as an arm
> and gets *better*. **Design the system so that a competitor's success is an input.**

---

# PART 7 — SYSTEM ARCHITECTURE

## 7.0 The layer cake

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ L5  MARKET            Quest board · bounties · reverse auction · reputation   │
│                       archive/leaderboard · `meshpool quest` CLI · console     │
├──────────────────────────────────────────────────────────────────────────────┤
│ L4  SETTLEMENT        Proof-of-Acceptance · escrow · held-back · replay        │
│                       receipts · dispute (bisection) · ledger conservation     │
├──────────────────────────────────────────────────────────────────────────────┤
│ L3  SEARCH            ARCHIPELAGO: islands · MAP-Elites archives · migration   │
│                       portfolio bandit · prompt assembly · budget controller   │
├──────────────────────────────────────────────────────────────────────────────┤
│ L2  GENERATION        Generator fleet: llama.cpp / vLLM / MLX / Ollama /       │
│                       licensed API · LoRA Adapter Swarm · prefix cache reuse   │
├──────────────────────────────────────────────────────────────────────────────┤
│ L1  EVALUATION        Evaluator Sandbox: WASM (T0) · Docker/OCI (T1) ·         │
│                       deterministic RISC-V (T2) · cascade runner · oracles     │
├──────────────────────────────────────────────────────────────────────────────┤
│ L0  SUBSTRATE (EXISTS) transport · SWIM gossip · scheduler/constellations ·    │
│                       blob store + CAS · trust · batch DAG · CLI/console/agent │
└──────────────────────────────────────────────────────────────────────────────┘
```

**L0 exists and is good. L1 is ~70% built. L2 is the biggest gap (no GPU, no real weights).
L3 is entirely new and is the invention. L4 exists but is pointed at the wrong predicate.
L5 is new but small.**

## 7.1 Layer 0 — Substrate (what is reused, unchanged)

| Existing component | File(s) | Reused for | Change needed |
|---|---|---|---|
| Transport (HMAC/mTLS/Noise, replay window, backpressure) | `common/transport.py`, `common/auth.py`, `common/identity.py` | Everything | 🟢 None |
| SWIM gossip + membership + NAT traversal + QUIC | `cloud/swim.py`, `cloud/nat.py`, `cloud/quic.py`, `cloud/tunnel.py` | Node liveness, reachability | 🟢 None |
| `Constellation` / `ConstellationIndex` | `controlplane/scheduler.py` | 🔴 **Islands** | 🟡 Add island metadata |
| Node classification (Anchor/Steady/Drifter) | `controlplane/nodeclass.py`, `controlplane/topology.py`, `common/vivaldi.py` | Island roles: Anchors host archives + held-out oracle, Drifters generate | 🟡 Role mapping |
| Blob store + erasure coding + crypto | `storage/engine.py`, `storage/blobstore.py`, `storage/erasure.py`, `storage/crypto.py` | Candidates, evaluators, archives, weights, receipts | 🟡 Add CID-typed namespaces |
| Batch engine + DAG | `batch/engine.py`, `batch/dag.py` | 🔴 **Verification cascade**, sample fan-out | 🟡 Add early-exit + cost-aware ordering |
| Trust / stake / escrow / ledger | `common/trust.py`, `controlplane/verification.py` | 🔴 **PoA settlement** | 🔴 New predicate class |
| WASM function runtime | `functions/wasm.py`, `agent-go/internal/wasmfn` | 🔴 **Tier-0 evaluator sandbox** | 🟡 Add fuel/mem/determinism caps + host ABI |
| Weight Swarm (P2P weight distribution) | `inference/engine.py` | 🔴 **Adapter Swarm** | 🟡 Adapter-sized objects |
| KV cache (paged + activation cache) | `inference/kvcache.py` | 🔴 **Prefix sharing across N samples** | 🔴 Add radix/prefix-tree reuse |
| Continuous batching | `inference/engine.py` | N samples of one Quest on one node | 🟡 Batch-invariance guarantee |
| Console (React) | `web/` | Quest board, archive explorer, live search viz | 🟡 New views |
| Go agent | `agent-go/` | Lightweight generator/evaluator node | 🟡 Add backends |
| Simulation gates + mutation guard | `sim/` | 🔴 **Preserve and extend** | 🟢 Add ~40 gates |

> **~85% of the existing 85K LOC survives the pivot.** This is not a rewrite. It is a
> *retarget*.

## 7.2 Layer 1 — The Evaluator Sandbox

The evaluator is the most security-critical component in the system. It runs
**attacker-supplied code** (the Quest author's evaluator) against **attacker-supplied
artifacts** (candidates) on **volunteers' personal machines**.

### 7.2.1 Three tiers

| Tier | Runtime | Determinism | Isolation | Startup | Use for |
|---|---|---|---|---|---|
| **T0 — WASM** | wasmtime (Python side) / wazero (Go agent) | **Strong**: no threads, no NaN nondeterminism if `nan-canonicalization` on, deterministic float, fuel-metered | Very strong (capability-based; no syscalls unless granted) | ~1 ms | Pure functions, math checkers, parsers, scorers, SAT/proof-object checkers, property tests |
| **T1 — OCI container** | Docker/Podman/gVisor/Firecracker | Medium (best-effort: pinned image digest, `SOURCE_DATE_EPOCH`, no net, fixed CPU model flags) | Strong (namespaces + seccomp; gVisor/Firecracker for hostile) | ~0.3–3 s | **Real test suites** — `pytest`, `go test`, `cargo test`, `npm test`; Lean/Coq; benchmarks |
| **T2 — Deterministic VM** | Cartesi Machine (RISC-V) or RISC Zero zkVM in non-proving mode | **Bit-exact, reproducible anywhere** | Total | ~1–5 s | Disputed verdicts; refereed bisection; high-value Quests |

**T2 is only needed for disputes.** The normal path is T0/T1. This mirrors the optimistic
rollup design (fast happy path, expensive deterministic arbitration only on challenge) and it
means MeshPool's existing `InteractiveBisectionDispute` finally has a real referee.

### 7.2.2 Evaluator contract (the ABI)

```python
# Evaluator manifest (content-addressed; the CID is the evaluator's identity)
{
  "schema": "meshpool.evaluator/v1",
  "tier": "wasm" | "oci" | "det-vm",
  "image": "sha256:...",                 # OCI digest or wasm module CID
  "entrypoint": ["/bin/sh", "-c", "..."],
  "inputs": { "candidate": "/work/candidate", "context": "/work/context" },
  "limits": { "wall_s": 120, "cpu_s": 240, "mem_mb": 4096, "pids": 256,
              "disk_mb": 2048, "fuel": 10_000_000_000 },
  "network": "none",                     # ALWAYS none for verdict-bearing runs
  "determinism": { "seed": 42, "source_date_epoch": 0, "tz": "UTC",
                   "locale": "C", "hostname": "meshpool", "pythonhashseed": "0" },
  "output": "/work/verdict.json",
  "held_out": "sha256:...",              # optional, encrypted; only Anchor nodes get it
  "public": true                         # is the evaluator itself public?
}
```

```python
# Verdict (what the evaluator must write)
{
  "schema": "meshpool.verdict/v1",
  "accepted": true,
  "score": 0.9812,                        # higher is better; for optimization Quests
  "behaviour": [0.31, 0.77, 12.0],        # MAP-Elites descriptor (see §8.2)
  "signals": { "tests_passed": 41, "tests_total": 44, "runtime_ms": 812,
               "peak_mem_mb": 210, "loc": 34 },
  "artifacts": { "stderr_tail": "...", "failing_tests": ["test_x"] },  # side-channel!
  "evaluator_cid": "...", "candidate_cid": "...", "env_cid": "..."
}
```

### 7.2.3 Determinism enforcement checklist (T1)

`[DESIGN]` A verdict-bearing container run **must**:
- pin the image by digest, never by tag;
- have **no network** (`--network=none`) — this alone blocks exfiltration and most nondeterminism;
- set `TZ=UTC`, `LC_ALL=C`, `PYTHONHASHSEED=0`, `SOURCE_DATE_EPOCH=0`, fixed `HOSTNAME`;
- run with a fixed CPU feature mask (`--cpu-flags` / QEMU `-cpu`) when the workload may use
  AVX-512-dependent reductions;
- mount everything read-only except a single scratch dir;
- have `--pids-limit`, `--memory`, `--cpus`, `ulimit -f`, and a wall clock kill;
- drop all capabilities, `--security-opt no-new-privileges`, seccomp default-deny;
- **run twice on the Anchor for high-value Quests and require identical verdicts**
  (self-determinism check, cheap and catches flaky evaluators immediately).

> 🔴 **Non-negotiable design rule:** *a Quest whose evaluator is not self-consistent across
> two runs on the same machine is rejected at submission time.* Flaky evaluators are the
> single biggest practical threat to a PoA market — they turn payment into a lottery and
> destroy node trust. **Detect them at intake, not at settlement.**

### 7.2.4 The volunteer-safety story (this is a product requirement, not a nicety)

Volunteers run strangers' code. The pitch must be:
- **No network in the sandbox, ever.** (Except an explicit, allow-listed, proxied fetch for
  dependency install, done *once* at image build time by the Quest author, not at run time.)
- **No host filesystem access.** Scratch only.
- **Hard resource caps and a visible kill switch.**
- **Opt-in Quest classes** (`science` / `oss` / `benchmarks` / `commercial`) with per-class
  consent, plus an evaluator-publisher allowlist mode.
- **Full local transparency**: `meshpool node log` shows exactly which Quest, which evaluator
  CID, which limits, and what it earned.

## 7.3 Layer 2 — The Generator Fleet

🔴 **This is the largest engineering gap and Phase 1's main job.**

### 7.3.1 Backend strategy — stop reimplementing, start driving

MeshPool's current `MeshShardBackend` executes transformer math in **NumPy on CPU**. That was
the right call for building a verifiable reference implementation; it is the wrong call for
production generation.

| Backend | Target hardware | Priority | Notes |
|---|---|---|---|
| **llama.cpp / llama-server** | Everything (CPU, CUDA, ROCm, Metal, Vulkan) | 🔴 **P0** | GGUF already parsed by MeshPool. Drive `llama-server` over its HTTP API, or bind `libllama`. Gives GPU support on day one across every vendor. |
| **Ollama** | Consumer desktops | 🟢 exists | Already an adapter. Keep — it's the lowest-friction on-ramp. |
| **vLLM / SGLang** | NVIDIA prosumer + datacenter nodes | 🟠 P1 | For high-throughput Quest workers. **SGLang's RadixAttention is the big win** (§8.8). |
| **MLX** | Apple Silicon | 🟠 P1 | Macs are the best consumer nodes. |
| **MeshShard (NumPy)** | reference | 🟢 keep | **Keep as the determinism reference oracle and for the tiny-model gates.** Do not delete — it's what makes the test suite non-vacuous. |
| **Licensed API passthrough** | any | 🟠 P1 | For MeshPool Private: the node calls the enterprise's own Azure OpenAI/Bedrock endpoint. Diversity without local GPUs. |

> **Critical reframing:** MeshPool's differentiator was never "we wrote a fast kernel." It is
> **orchestration, verification, and search.** Delegate the kernels. Own the loop.

### 7.3.2 Solving the "no pretrained weights" blocker

`[MEASURED]` The repo has *never* run a real pretrained model because corporate TLS
interception breaks HuggingFace downloads (`CRYPT_E_NO_REVOCATION_CHECK`).

**This must be fixed in Phase 0.** Options, in order of preference:
1. `hf_transfer` / `huggingface-cli` with `HF_HUB_ETAG_TIMEOUT` and a corporate CA bundle
   (`REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` pointed at the exported enterprise root).
2. `curl -k` (explicitly acknowledged as *only* acceptable for fetching *public* weights whose
   SHA-256 is then verified against a known digest — **verify the digest, not the TLS chain**).
3. Download on a personal/unfiltered network and copy in.
4. Ollama's registry (often reachable when HF is not).
5. Any local `~/.ollama/models` blobs already present.

**Then: a permanent regression gate** — `sim/scenarios/real_weights_smoke.py` — that loads a
real ~1B GGUF (e.g. Qwen2.5-0.5B/1.5B, Llama-3.2-1B, SmolLM2-1.7B) and asserts the first 20
tokens of a fixed prompt match a golden file. **Nothing else in the plan can be trusted until
this gate is green.**

### 7.3.3 Generator node profile

```
node.services = ["generate"]                # new service kind alongside inference/storage/batch
node.generator = {
  "backends": ["llamacpp"],
  "models": [{cid, name, params_b, quant, ctx, vram_mb, warm: true}],
  "adapters": [{cid, base_cid, rank, size_mb}],
  "max_concurrent": 4,
  "tok_per_s_measured": {model_cid: 34.2},   # measured, not claimed
  "supports_batch_invariant": true,
  "consent_classes": ["science", "oss"]
}
```

**Capability is measured, never claimed** — a node's advertised throughput is established by
a signed benchmark Quest, not self-report. (Direct lesson from io.net's fake-GPU incident.)

## 7.4 Layer 3 — Archipelago (the search plane)

See §8.2 for the algorithm. Architecturally:

```
QuestCoordinator (on an Anchor, HA-replicated)
├── BudgetController          : spend tracking, stopping rules, per-island allocation
├── PortfolioBandit           : arm selection (§8.3)
├── IslandManager[]           : one per Constellation
│   ├── Archive (MAP-Elites)  : blob-store backed, CRDT-merged
│   ├── PromptAssembler       : best-shot + diverse-shot + artifact side-channel
│   ├── GeneratorPool         : nodes in this Constellation with `generate`
│   └── EvaluatorPool         : nodes in this Constellation with `evaluate`
├── MigrationScheduler        : ring topology, RTT-aware period
└── SettlementJournal         : PoA receipts → ledger
```

**Key property:** an Island can run **fully autonomously for minutes** with no coordinator
contact. If the coordinator dies, islands keep searching and reconcile archives on reconnect
(archives are **CRDTs** — a MAP-Elites grid merges by taking the better elite per cell, which
is a join-semilattice ⇒ trivially convergent). This is the single most important
availability property in the design, and it is only possible because κ = 1/N.

## 7.5 Layer 4 — Verification & settlement

Reuses `controlplane/verification.py` almost wholesale, with one new classification:

```python
class VerificationClass(Enum):
    ...
    PROOF_OF_ACCEPTANCE = "poa"   # NEW: verdict-bearing, replayable, pay-on-accept
```

**The keystone change.** Today `VerificationPolicy.classify()` returns `NOT_VERIFIABLE` for
any unknown user function, which makes the entire economic layer inert. Under PoA, *every*
Quest is verifiable **by construction**, because verifiability is the *submission requirement*.

```
Quest intake → evaluator determinism check (2× run) → escrow bounty
             → fan out → candidates → cascade → verdicts
             → accepted? → replay receipt → release escrow to node + Quest author gets artifact
             → not accepted? → node earns nothing (optionally a small "effort floor", §9.4)
             → dispute? → re-run on 3 independent Anchors → T2 deterministic VM → bisection
```

## 7.6 The Context Pack — the unglamorous component that decides everything

`[DESIGN]` Every Quest needs a **Context Pack**: the bounded slice of repository/problem state
sent to generators. It is simultaneously the **quality bottleneck** (bad context ⇒ no
acceptance), the **cost driver** (tokens ⇒ money), and the **privacy boundary** (§10.5).
It is easy to under-specify, and if it is bad it will silently cap the entire system's
performance while every other subsystem looks healthy.

```python
@dataclass
class ContextPack:
    cid: str
    files: list[FileSlice]            # path, byte range, content, why_included
    failing_test: str
    error_output: str                 # the failure the candidate must fix
    symbols: list[SymbolDef]          # signatures of called-but-not-included functions
    repo_summary: str                 # README/architecture digest, heavily truncated
    conventions: str                  # detected style, test framework, language version
    token_count: int
    redactions: list[Redaction]       # what was removed, for privacy tier V1
    editable_paths: list[str]         # 🔴 THE TAMPER ALLOWLIST (§10.2 L1)
```

**Construction pipeline:**

```python
def build_context_pack(repo, failing_test, budget_tokens, privacy_tier) -> ContextPack:
    # 1. SEED — run the failing test, capture the traceback
    trace = run_test(repo, failing_test)
    seeds = frames_from_traceback(trace)                      # the single best signal available

    # 2. EXPAND — static call graph, N hops from each seed frame
    nodes = call_graph_neighbourhood(repo, seeds, hops=2)

    # 3. RANK — combine independent signals, cheapest first
    scored = rank(nodes, by=[
        traceback_proximity,          # strongest signal
        git_cochange_frequency,       # files historically edited together with the seed
        static_dependency_distance,
        embedding_similarity_to_error,
        recency_of_last_edit,         # regressions live in recent changes
    ])

    # 4. FIT — pack greedily under the token budget, always including the test + trace
    pack = greedy_knapsack(scored, budget_tokens, must_include=[failing_test, trace])

    # 5. SUMMARIZE the overflow rather than dropping it silently
    pack.symbols = signatures_only(dropped_but_referenced(scored, pack))

    # 6. REDACT for the privacy tier
    if privacy_tier >= V1:
        pack = redact(pack, strip_string_literals=True, alpha_rename=True,
                      drop_comments_matching=SECRET_PATTERNS)

    # 7. PIN the tamper allowlist — anything not listed here is unmodifiable
    pack.editable_paths = infer_editable(seeds) - test_paths(repo) - evaluator_paths(repo)
    return pack
```

**Why this deserves its own section:**
- 🔴 **Context quality dominates model quality at small scale.** `[ESTIMATE]` A 7B model with
  an excellent context pack will beat a 70B model with a poor one on repo-scale bug fixing.
  **Invest here before investing in bigger models.**
- 🟢 **It is a structural cost advantage.** With prefix sharing (§8.8) the pack is prefilled
  *once* and amortized across all N samples, so a 2× richer pack costs far less than 2×.
  **MeshPool can afford context that a per-call API user cannot.** Non-obvious; worth marketing.
- **It is where privacy is actually enforced** (§10.5 tier V1) — not by cryptography, by
  *selection*.
- **It defines the tamper allowlist**, which is the free anti-reward-hacking gate (§10.2 L1).
- **Pack composition should itself be a bandit arm.** Different packing strategies suit
  different bug classes; let the system learn it rather than hand-tuning forever.

**Gate:** `context_pack_ablation.py` — measure acceptance rate across 4 packing strategies on
the benchmark; assert the ranked strategy beats naive whole-file inclusion at equal token count.

## 7.7 Layer 5 — Market & UX

### The CLI wedge (this is the entire go-to-market in one command)

```bash
$ meshpool quest fix --repo . --test "pytest tests/test_parser.py::test_nested" --budget 200
  ✓ evaluator built (oci sha256:7f2a…, determinism check 2/2 identical)
  ✓ context packed: 3 files, 812 LOC, 4.1 KB prompt
  ⣾ searching  ██████░░░░  118/200 samples · 3 islands · 41 nodes · 6 arms
    island-0 (qwen2.5-coder-7b   t=0.8)  best 41/44 tests
    island-1 (deepseek-coder-33b t=0.4)  best 43/44 tests
    island-2 (llama-3.1-8b+lora  t=1.1)  best 39/44 tests
  ✓ ACCEPTED at sample 137 (island-1, gen 4)
    44/44 public · 12/12 held-out · no test files modified · 0.81 s runtime
    receipt: mp://receipt/9a1c…  (verify: meshpool verify mp://receipt/9a1c…)
  → patch written to ./meshpool-fix-9a1c.patch
```

```bash
$ meshpool quest optimize --target bench/matmul.py --metric wall_ms --minimize --budget 1000
$ meshpool quest prove --lean theorems/Cap.lean --goal "cap_set_bound" --budget 5000
$ meshpool quest upgrade --dep "django>=4.2,<5" --tests "pytest -q" --budget 500
$ meshpool quest fuzz --target ./parser --corpus ./seeds --budget 2000
$ meshpool verify mp://receipt/9a1c…        # anyone, anywhere, re-runs the verdict
```

### Console views (React app, `web/`)
- **Quest board** — open Quests, bounties, budgets burned, acceptance rate.
- **Live search view** — the Archipelago: islands as columns, MAP-Elites archive as a heatmap,
  candidates streaming in, best-score curve vs. budget. **This is the demo that sells it.**
- **Archive explorer** — browse accepted solutions, diff them, replay a receipt in one click.
- **Node earnings** — what your machine solved, what it earned, what it refused.
- **Portfolio analytics** — which arms win on which problem classes (the compounding asset).

---

# PART 8 — CORE ALGORITHMS (with pseudocode)

> Everything here is `[DESIGN]`. Pseudocode is Python-flavoured and deliberately explicit so
> the next session can translate it directly into `meshpool/quest/*.py`.

## 8.1 The top-level Quest loop

```python
def run_quest(quest: Quest) -> QuestResult:
    # 0. INTAKE — reject anything that would break PoA
    assert_deterministic_evaluator(quest.evaluator, trials=2)     # §7.2.3
    escrow = escrow_engine.lock(quest.author, quest.budget)

    islands = spawn_islands(quest, constellations=select_constellations(quest))
    bandit  = PortfolioBandit(arms=enumerate_arms(quest, mesh_capabilities()))
    budget  = BudgetController(quest.budget, quest.max_wall_s)
    best    = None

    while not budget.exhausted() and not stopping_rule(best, budget, quest):
        # 1. ALLOCATE — which island, which arm, how many samples
        plan = allocate(islands, bandit, budget)                  # §8.3

        # 2. GENERATE — fan out, fully independent (κ = 1/N)
        prompts   = {i: islands[i].assemble_prompts(plan[i]) for i in plan}
        candidates = generator_pool.sample(prompts, plan)          # §8.8 prefix sharing

        # 3. EVALUATE — cheap→expensive cascade with early exit
        verdicts  = cascade.evaluate(candidates, quest.evaluator)  # §8.7

        # 4. LEARN + ARCHIVE
        for c, v in zip(candidates, verdicts):
            bandit.update(arm=c.arm, problem_emb=quest.emb, reward=v.score, accepted=v.accepted)
            islands[c.island].archive.insert(c, v)                 # §8.2 MAP-Elites
            budget.charge(c.cost)
            if v.accepted:
                receipt = mint_replay_receipt(quest, c, v)         # §8.6
                settle_poa(escrow, c.node, quest.bounty_share, receipt)  # §9
                best = better(best, (c, v))
                if quest.stop_on_first_accept:
                    return QuestResult(best, receipt, budget.spent)

        # 5. MIGRATE — cross-pollinate islands (tiny bandwidth, latency-tolerant)
        migration_scheduler.maybe_migrate(islands)                 # §8.2.4

    escrow.refund_unspent()
    return QuestResult(best, receipts, budget.spent)
```

## 8.2 Archipelago: distributed island-model MAP-Elites

### 8.2.1 The archive

```python
@dataclass
class Archive:
    """MAP-Elites grid. One elite per behaviour cell. This is a JOIN-SEMILATTICE:
    merge(A,B)[cell] = max(A[cell], B[cell], key=score)  ⇒ CRDT, order-independent,
    idempotent, commutative. That is why islands can merge without a coordinator."""
    dims: list[FeatureDim]              # e.g. [tests_passed_frac, log_runtime, log_loc]
    bins: list[int]                     # e.g. [10, 10, 8]  -> 800 cells
    cells: dict[tuple[int,...], Elite]

    def key(self, behaviour: list[float]) -> tuple[int, ...]:
        return tuple(
            min(self.bins[i]-1, max(0, int(self.dims[i].normalize(b) * self.bins[i])))
            for i, b in enumerate(behaviour)
        )

    def insert(self, cand, verdict) -> bool:
        k = self.key(verdict.behaviour)
        cur = self.cells.get(k)
        if cur is None or verdict.score > cur.score:
            self.cells[k] = Elite(cand, verdict); return True      # new/improved elite
        return False

    def merge(self, other: "Archive") -> None:                     # CRDT join
        for k, e in other.cells.items():
            cur = self.cells.get(k)
            if cur is None or e.score > cur.score:
                self.cells[k] = e

    def coverage(self) -> float:  return len(self.cells) / prod(self.bins)
    def qd_score(self) -> float:  return sum(e.score for e in self.cells.values())
```

> **Why MAP-Elites and not a plain population?** Because on a rugged landscape (which code and
> proof spaces are), the stepping stone to the global optimum is usually a *currently
> mediocre* solution with unusual behaviour. Keeping one elite per behaviour cell preserves
> stepping stones. This is exactly why FunSearch and AlphaEvolve use it, and why OpenEvolve's
> README leads with it.

### 8.2.2 Behaviour descriptors (the design decision that makes or breaks QD)

`[DESIGN]` Descriptors must be (a) cheap, (b) computed from *partial* evaluator signals, and
(c) *orthogonal to score*.

| Quest class | Descriptor dims (suggested) |
|---|---|
| **Code repair** | (fraction of tests passed, log(diff size), log(runtime), AST-edit-distance-to-original) |
| **Program optimization** | (log wall time, log peak memory, algorithmic-complexity class estimate, LOC) |
| **Theorem proving** | (proof depth, #lemmas used, tactic-set fingerprint, goal-closure fraction) |
| **Kernel optimization** | (occupancy, memory-bound-ness, tile shape class, register pressure) |
| **Config/hyperparameter** | the raw config axes themselves |
| **Fuzzing** | (coverage bitmap hash bucket, crash type, input size) |
| **Generic fallback** | (embedding of candidate → 2-D via a fixed random projection or a small learned map) |

**Adaptive dimensions:** OpenEvolve supports adaptive feature dims. Start fixed; add
`--qd-auto-dims` in v2 that re-bins based on observed spread (CVT-MAP-Elites is the principled
version — use centroidal Voronoi tessellation when dims > 4).

### 8.2.3 Prompt assembly (the second most important design decision)

```python
def assemble_prompt(island, quest, arm) -> Prompt:
    A = island.archive
    parts = [quest.problem_statement, quest.context_pack]         # minimized context, §10.5

    # (a) BEST-SHOT: the top-k elites by score  (FunSearch's core trick)
    best = topk(A.cells.values(), k=arm.k_best, key=score)

    # (b) DIVERSE-SHOT: elites far away in behaviour space (the QD trick)
    diverse = farthest_point_sample(A.cells.values(), k=arm.k_div, exclude=best)

    # (c) ARTIFACT SIDE-CHANNEL: the failure output of the parent  (OpenEvolve's key finding)
    parent = select_parent(A, strategy=arm.parent_strategy)       # see 8.2.5
    parts += [render(best), render(diverse),
              f"Your previous attempt failed with:\n{parent.verdict.artifacts.stderr_tail}"]

    # (d) MUTATION INSTRUCTION: diff-mode vs rewrite-mode
    parts += [arm.mutation_instruction]        # "emit a unified diff" | "rewrite the function"
    return Prompt(parts, temperature=arm.temperature, seed=next_seed(),
                  thinking_budget=arm.thinking_budget)
```

**Diff-mode vs rewrite-mode is an arm dimension.** AlphaEvolve/OpenEvolve both found diff-based
evolution far more token-efficient on large files, while full rewrite wins on small ones.
**Let the bandit learn which.**

### 8.2.4 Migration (why partitioning becomes an advantage)

```python
def maybe_migrate(islands, now):
    for i, isl in enumerate(islands):
        # RTT-aware period: chatty for metro islands, sparse for intercontinental
        period = clamp(BASE_PERIOD * (1 + rtt_ms(isl, islands[(i+1) % len(islands)]) / 50), 30, 600)
        if now - isl.last_migration < period:  continue
        dst = islands[(i + 1) % len(islands)]                  # RING topology
        emigrants = select_emigrants(isl.archive, k=MIGRATION_K)  # top elites + novel elites
        dst.archive.merge(Archive.from_elites(emigrants))         # CRDT join — safe out of order
        isl.last_migration = now
```

`[DERIVED]` **Bandwidth cost of migration:** `MIGRATION_K` (say 8) elites × candidate size
(say 4 KB of diff) = **32 KB every 30–600 s per island**. Compare to pipeline-parallel
inference: 14 KiB **per token per hop**. Migration traffic is ~4 orders of magnitude lighter
and is **completely latency-insensitive**. This is the arithmetic that makes the pivot work
over the public internet.

### 8.2.5 Parent selection — double selection

OpenEvolve's "double selection": use *different* selection pressure for the program you
*modify* vs. the programs you *show as inspiration*.

```python
def select_parent(archive, strategy):
    if strategy == "elite":     return argmax(archive.cells.values(), key=score)
    if strategy == "curiosity": return argmax(archive.cells.values(), key=novelty)      # least-visited cell
    if strategy == "frontier":  return sample(pareto_front(archive.cells.values()))     # multi-objective
    if strategy == "uniform":   return random.choice(list(archive.cells.values()))
    if strategy == "recent":    return most_recent_improvement(archive)
```

**Also an arm dimension.** Also learned.

## 8.3 The Portfolio Bandit (contextual, coverage-maximizing)

### 8.3.1 The objective is coverage, and coverage is submodular

```
maximize   F(S) = P(at least one accepted candidate | allocation S)
subject to sum_c cost(c) * k_c <= B
```

`[DERIVED]` `F` is a coverage function over the "problem instances solved" ground set ⇒
**monotone submodular** ⇒ greedy achieves `(1 − 1/e) ≈ 63%` of optimal, and cost-benefit
greedy (Leskovec et al.) handles the knapsack constraint. So the scheduler is:

```python
def allocate(islands, bandit, budget) -> dict[island, list[(arm, n_samples)]]:
    plan, remaining = defaultdict(list), budget.remaining()
    while remaining > 0:
        # cost-benefit greedy: marginal coverage gain per unit cost
        best = max(
            ((i, a) for i in islands for a in bandit.arms if a.feasible_on(i)),
            key=lambda ia: bandit.marginal_gain(ia[1], ia[0], plan) / ia[1].cost_estimate
        )
        n = batch_size_for(best)                     # amortize weight load & prefix cache
        plan[best[0]].append((best[1], n))
        remaining -= best[1].cost_estimate * n
    return plan
```

### 8.3.2 Learning arm quality — Thompson sampling with problem context

```python
class PortfolioBandit:
    """Beta-Bernoulli Thompson sampling, but the prior is CONDITIONED ON PROBLEM CLASS.
    This is the compounding asset: after 10k Quests we know which (model,temp,scaffold)
    wins on 'python async bug' vs 'rust borrow-checker' vs 'lean algebra lemma'."""

    def __init__(self, arms):
        self.arms = arms
        self.alpha = defaultdict(lambda: 1.0)   # (arm_id, cluster_id) -> successes
        self.beta  = defaultdict(lambda: 1.0)   # (arm_id, cluster_id) -> failures
        self.clusterer = ProblemClusterer()     # embed(problem) -> cluster_id

    def sample_p(self, arm, problem_emb):
        c = self.clusterer.assign(problem_emb)
        # hierarchical: blend cluster-specific with global prior (shrinkage)
        a_c, b_c = self.alpha[(arm.id, c)], self.beta[(arm.id, c)]
        a_g, b_g = self.alpha[(arm.id, GLOBAL)], self.beta[(arm.id, GLOBAL)]
        n_c = a_c + b_c
        w = n_c / (n_c + SHRINKAGE_K)           # trust cluster stats once we have enough
        a = w*a_c + (1-w)*a_g;  b = w*b_c + (1-w)*b_g
        return random.betavariate(a, b)

    def marginal_gain(self, arm, island, plan):
        """Coverage gain accounting for CORRELATION with already-planned arms.
        Two 7B code models at t=0.7 and t=0.75 are nearly the same draw."""
        p = self.sample_p(arm, island.quest.emb)
        redundancy = max((self.similarity(arm, a) for a, _ in plan[island]), default=0.0)
        p_eff = p * (1.0 - CORRELATION_PENALTY * redundancy)
        already = 1.0 - prod((1 - self.sample_p(a, island.quest.emb))**n for a, n in plan[island])
        return (1 - already) * p_eff            # marginal coverage

    def similarity(self, a, b) -> float:
        """Arm similarity: same base model family? same quant? |Δtemperature|? same scaffold?
        Learned later from outcome correlation; hand-specified at first."""
```

### 8.3.3 The arm space

```python
Arm = namedtuple("Arm", "model_cid quant adapter_cid temperature top_p scaffold "
                        "thinking_budget mutation_mode parent_strategy k_best k_div")
```

`[DESIGN]` Practical initial arm set (~40–200 arms):
- **models**: 4–10 distinct base models (different *families*, not different sizes of one family
  — family diversity decorrelates errors far more).
- **quant**: Q4_K_M, Q5_K_M, Q8_0, FP16 (**note: quantization changes behaviour and is
  therefore a *free* diversity axis** — a genuinely novel exploitation of heterogeneity).
- **temperature**: {0.2, 0.6, 0.8, 1.0, 1.2}.
- **scaffold**: {direct, CoT, self-refine, test-first, plan-then-code, debug-loop}.
- **thinking_budget**: {short, medium, long} (s1-style budget forcing).
- **mutation_mode**: {diff, rewrite, targeted-function}.

> **Prune aggressively.** 10×4×5×6×3×3 = 10,800 arms is far too many to learn. Use a
> **factored** model (a linear/logistic model over arm *features* rather than arm identity)
> so evidence generalizes across arms. Start with factored Thompson sampling; that is the
> difference between learning in 10k samples and never learning.

## 8.4 Weight-Affinity Scheduling (the bandwidth killer)

**The problem:** the portfolio wants model diversity; loading a 20 GB model over a 20 Mbps
residential uplink takes **2.2 hours**. Naive portfolio scheduling would drown the mesh.

**The insight:** MeshPool *already* tracks `weight_group_cached` in `inference/engine.py`.
Invert the scheduler: **do not choose an arm and then find a node — enumerate the arms that
are already warm somewhere and choose among those.**

```python
def feasible_arms(mesh) -> dict[Arm, list[Node]]:
    """An arm is 'free' if its weights are ALREADY resident. Cold arms cost a download."""
    out = defaultdict(list)
    for node in mesh.nodes_with_service("generate"):
        for m in node.resident_models:
            for a in arms_over_model(m):
                out[a].append(node)
    return out

def arm_cost(arm, mesh) -> float:
    warm = feasible_arms(mesh).get(arm, [])
    if warm:  return arm.token_cost                                   # ~free
    # cold: amortize the transfer over the samples we expect to draw
    bytes_ = model_size(arm.model_cid)
    return arm.token_cost + (bytes_ / mesh.median_uplink_bps) * TIME_PRICE / EXPECTED_SAMPLES
```

**Corollary — the mesh should deliberately maintain a *portfolio residency plan*:**

```python
def plan_residency(mesh, demand_history) -> dict[Node, set[ModelCID]]:
    """Choose which models live where, to MAXIMIZE PORTFOLIO COVERAGE PER BYTE,
    subject to each node's VRAM/disk. This is a set-cover / facility-location problem.
    Greedy: repeatedly place the model that adds the most expected coverage per GB."""
```

`[DESIGN]` **This is a genuinely novel scheduling problem** — nobody schedules for *portfolio
diversity under weight-residency constraints*, because nobody else's product is diversity.
It should be written up as the technical-blog centrepiece.

## 8.5 The Adapter Swarm (diversity at 0.25% of the bandwidth)

**The trick:** a rank-16 LoRA over a 7B model is **~40–80 MB**. A distinct 7B model is
**~4–15 GB**. So:

```
one base model resident + 50 adapters  =  51 distinct generators
bandwidth cost: 4 GB once + 50 × 50 MB = 6.5 GB
vs. 51 distinct models                 = ~250 GB
                                          ⇒ ~38× cheaper
```

Adapters can be:
- **Style adapters** — "writes tests first", "prefers functional style", "verbose reasoning".
- **Domain adapters** — Rust, Lean, CUDA, SQL, embedded C.
- **Behaviour adapters** — "aggressive refactorer" vs "minimal-diff surgeon".
- 🔴 **Self-improving adapters** — *trained on the network's own accepted solutions.*

```python
def swarm_serve(node, base_cid, requests):
    """Multi-LoRA serving: one base in VRAM, adapters hot-swapped per request.
    vLLM/llama.cpp both support this. Batch requests that share an adapter."""
    base = load_once(base_cid)
    for adapter_cid, batch in group_by_adapter(requests):
        with base.with_adapter(fetch_adapter(adapter_cid)):   # ~50 MB, from Weight Swarm
            yield base.generate_batch(batch)
```

> ### The compounding flywheel (this is the long-term moat)
>
> ```
> accepted solutions  →  training set for LoRA adapters
>                     →  better adapters
>                     →  higher acceptance rate
>                     →  more accepted solutions
>                     →  …
> ```
>
> And every step of it is *verified data*: `(problem, candidate, machine-checked accepted)`.
> **Verified RL data is currently the most valuable commodity in AI.** The network generates
> it as a byproduct of doing its job. `[DESIGN]` This is the strongest strategic argument in
> the entire document and should be Part 13's lead.

## 8.6 Replay Receipts

```python
@dataclass(frozen=True)
class ReplayReceipt:
    schema: str = "meshpool.receipt/v1"
    quest_id: str
    evaluator_cid: str          # content hash of the evaluator image/module
    env_cid: str                # content hash of the environment (base image + context pack)
    candidate_cid: str          # content hash of the artifact
    verdict_hash: str           # hash of the canonical JSON verdict
    verdict: Verdict
    generation: GenAttestation  # model_cid, adapter_cid, quant, seed, sampler params,
                                # engine_build_id, batch_invariant: bool
    evaluator_node: str; evaluator_sig: bytes
    coordinator_sig: bytes
    replicas: list[tuple[str, str]]      # (node_id, verdict_hash) for high-value Quests
    tier: str                            # "R0" | "R1" | "R2"
    ts: float

def verify_receipt(r: ReplayReceipt) -> bool:
    """Anyone, anywhere, offline-ish. THIS is the trust story."""
    assert cas.fetch(r.evaluator_cid) and cas.fetch(r.candidate_cid) and cas.fetch(r.env_cid)
    v = sandbox.run(r.evaluator_cid, r.candidate_cid, r.env_cid)   # re-execute
    return canonical_hash(v) == r.verdict_hash and sigs_valid(r)
```

**Canonical JSON** must be specified precisely (sorted keys, no whitespace, float formatting
via shortest-round-trip `repr`, explicit NaN/Inf rejection) — otherwise hashes diverge across
languages. `[DESIGN]` MeshPool already has hashing discipline in the CAS; extend it.

## 8.7 The Verification Cascade (optimal stopping)

Running the full test suite on every one of 500 candidates is the naive design and it is
**the thing that would make PoA expensive**. Instead: a cost-ordered ladder with early exit.

```python
STAGES = [
    Stage("identity",   cost=0.0001, run=dedup_by_normalized_hash),        # exact/near dup?
    Stage("syntax",     cost=0.001,  run=parse_check),
    Stage("tamper",     cost=0.001,  run=forbid_test_and_evaluator_edits),  # §10.2
    Stage("static",     cost=0.01,   run=lint_typecheck),
    Stage("smoke",      cost=0.1,    run=run_target_test_only),
    Stage("public",     cost=1.0,    run=run_public_suite),
    Stage("held_out",   cost=1.5,    run=run_secret_suite),                 # Anchor only
    Stage("replicate",  cost=3.0,    run=rerun_on_2_independent_nodes),     # high value only
]

def evaluate(cand, quest):
    for s in STAGES:
        if not s.enabled_for(quest): continue
        v = s.run(cand, quest)
        if not v.ok:
            return Verdict(accepted=False, score=v.partial_score,
                           behaviour=v.behaviour, artifacts=v.artifacts)   # SIDE-CHANNEL!
    return Verdict(accepted=True, ...)
```

`[DERIVED]` **Why this is dramatic:** if 60% of candidates die at `syntax` (cost 0.001) and
30% more at `smoke` (cost 0.1), the average cost per candidate is
`0.6×0.001 + 0.3×0.101 + 0.1×2.6 ≈ 0.29` vs. `2.6` for the naive design — a **~9× reduction
in verification cost.** AlphaCode's ~99% filter rate says the real numbers may be even better.

**Optimal stopping refinement:** stage order should be learned. The optimal order for a
sequence of independent tests with cost `c_i` and pass-probability `p_i` is by ascending
`c_i / (1 − p_i)` (the classic "least-cost-per-unit-of-elimination" rule). Learn `p_i` per
Quest class. `[DESIGN]`

**Crucially, failures are not wasted** — every stage emits `artifacts` (stderr, failing test
names, lint messages) that feed the next prompt. In an evolutionary loop, **a well-described
failure is worth almost as much as a success.**

## 8.8 Prefix sharing: the N-samples-on-one-node superpower

All N samples of a Quest share the *same* long prompt (repo context, archive shots). With
SGLang-style RadixAttention or llama.cpp's prompt cache:

```
naive:  N × (prefill(P) + decode(D))
shared: 1 × prefill(P) + N × decode(D)
```

`[DERIVED]` With `P = 8000` context tokens and `D = 500` generated tokens, and prefill being
roughly compute-bound while decode is memory-bandwidth-bound, prefill is often **60–90% of
total work** for this shape. Sharing it across N=32 samples on one node is a
**~3–6× throughput improvement** on generation. `[ESTIMATE]` — must be measured.

> 🔴 **Scheduling consequence, and it is a big one:** *co-locate many samples of the same
> Quest on the same node.* This directly contradicts naive "spread work everywhere" fairness,
> and it is a second genuinely novel scheduling constraint (alongside weight-affinity).
> **Sample locality is a first-class scheduling objective.**

## 8.9 Candidate deduplication & clustering (AlphaCode's third trick)

```python
def dedup_and_cluster(candidates, quest):
    # 1. exact: canonical-form hash (strip comments/whitespace, normalize names)
    uniq = {canonical_hash(c): c for c in candidates}.values()
    # 2. semantic: behavioural fingerprint = outputs on generated probe inputs
    fp = {c: behaviour_fingerprint(c, quest.probe_inputs) for c in uniq}
    # 3. cluster by fingerprint; the LARGEST cluster is the most likely correct  (AlphaCode)
    clusters = group_by(fp)
    return sorted(clusters, key=len, reverse=True)
```

**Why this matters economically:** if 40 nodes independently produce the *same* patch, only
the **first** should be paid in full (§9.5), or the network pays 40× for one solution.
Dedup is not just efficiency — it is **a settlement rule**.

## 8.10 The stopping rule

```python
def stopping_rule(best, budget, quest) -> bool:
    if best and best.verdict.accepted and quest.stop_on_first_accept: return True
    if budget.spent >= quest.budget: return True
    # Value-of-information: stop when marginal P(success) per unit cost < the author's
    # stated willingness-to-pay per unit probability.
    if marginal_coverage_gain_per_cost(budget.window) < quest.wtp_threshold: return True
    # Plateau detection on the QD score across all islands
    if qd_score_plateau(window=quest.plateau_window): return True
    return False
```

`[DESIGN]` The coverage curve is log-linear in k (§4.1.1), so the *marginal* gain decays like
`1/k`. That gives a clean, principled, and honest stopping rule the buyer can reason about:
**"the next 100 samples buy you about 4 more percentage points of success probability; do
you want them?"**

## 8.11 Difficulty-adaptive strategy selection

`[VERIFIED-EXT]` Snell et al. found that the *optimal search strategy inverts with difficulty*:
beam search (strong optimizer) wins on hard problems at low budget; best-of-N (weak optimizer)
wins on easy problems at high budget; and over-optimizing easy problems actively **degrades**
results (§10.2.1). A scheduler that ignores this will systematically do the wrong thing at
both ends.

```python
def select_strategy(quest, difficulty_est, budget_remaining, verifier_strength):
    """Difficulty × budget × verifier-strength → search strategy.
    Note the INVERSION: easy problems get LESS optimization pressure, not more."""
    pressure_cap = max_selection_pressure(quest)          # §10.2.1 coupling law — hard ceiling

    if difficulty_est < 0.3:                              # easy
        # Strong optimizers exploit the verifier here. Stay wide and shallow.
        return Strategy(mode="best_of_n", selection_pressure=min(0.2, pressure_cap),
                        temperature_spread="wide", evolution=False)
    if difficulty_est < 0.7:                              # medium
        return Strategy(mode="archipelago", selection_pressure=min(0.6, pressure_cap),
                        temperature_spread="medium", evolution=True, islands=3)
    # hard: this is where evolution earns its keep
    return Strategy(mode="archipelago", selection_pressure=min(1.0, pressure_cap),
                    temperature_spread="narrow", evolution=True, islands=6,
                    migration_period=30, parent_strategy="frontier")

def estimate_difficulty(quest, early_verdicts) -> float:
    """Cheap online estimate. Do NOT ask the user — they don't know either.
    Bootstrap from the first ~10 samples, then refine."""
    if not early_verdicts:
        return prior_difficulty_from_embedding(quest.emb)     # learned from the archive
    accept_rate = mean(v.accepted for v in early_verdicts)
    partial     = mean(v.score     for v in early_verdicts)
    # High partial score but zero acceptance = "close but hard" = the sweet spot for evolution
    return clamp(1.0 - accept_rate * 0.7 - partial * 0.3, 0.0, 1.0)
```

> **A genuinely useful emergent behaviour:** problems where candidates score *high partial
> credit but never fully pass* (41/44 tests) are exactly where evolution beats i.i.d.
> sampling — there is a gradient to climb. Problems that are either trivially solved or score
> ~0 across the board should get **no** evolutionary pressure, because there is no signal.
> **Detect the gradient before spending on search.** `[DESIGN]`

```python
def has_exploitable_gradient(verdicts, min_n=10) -> bool:
    """Evolution only helps if partial scores VARY. Flat scores = no gradient = don't evolve."""
    if len(verdicts) < min_n: return False
    scores = [v.score for v in verdicts]
    return stdev(scores) > GRADIENT_THRESHOLD and max(scores) > 0.5 and not any(v.accepted for v in verdicts)
```

---

# PART 9 — ECONOMICS: PROOF-OF-ACCEPTANCE

## 9.1 The settlement rule

```
payout(node, candidate) = 
    if   verdict.accepted and receipt_replayable and not duplicate_of_earlier:
         bounty_share(candidate)
    elif verdict.accepted and duplicate_of_earlier:
         duplicate_share  (small, > 0 — see §9.5)
    else:
         effort_floor(node, candidate)      # possibly 0 — see §9.4
```

**Everything else in the economic design is a consequence of this one rule.**

## 9.2 Why this is structurally better than every competitor's model

| Model | Pays for | Attack | Observed outcome |
|---|---|---|---|
| **DePIN capacity** (io.net, Aethir, Nosana…) | *Claimed* hardware | Fake the hardware | 🔴 io.net fake-GPU incident |
| **Subjective scoring** (Bittensor) | Validator opinion | Game the scorer | 🔴 Persistent, well-documented |
| **Time-based rental** (Vast, Akash) | Wall-clock hours | Underdeliver silently | 🟡 Buyer must self-police |
| **Replication** (BOINC, Golem) | Redundant compute | Collude / exploit non-determinism | 🟡 2–3× cost overhead |
| **Proofs** (zkML) | Cryptographic proof | — | 🔴 10³–10⁶× overhead: unaffordable |
| 🟢 **PoA** | **An artifact the buyer's own predicate accepted** | **Write a bad predicate** | The buyer controls their own risk |

> **The core economic insight:** every other network must answer *"did the node do the work?"*
> — a question only the network can answer, and therefore a question adversaries attack.
> PoA asks *"is the artifact good?"* — a question **the buyer already knows how to answer**,
> and therefore one no adversary can corrupt.

## 9.3 Pricing a Quest

```python
quest.budget      = max total spend (in credits)
quest.bounty      = paid on first acceptance
quest.rate        = credits per 1k generated tokens (the "cost of trying")
quest.wtp         = willingness-to-pay per unit of success probability (drives stopping)
```

Two regimes:

**A. Bounty-dominant** (`bounty ≫ rate × budget/rate`): a lottery. Nodes speculate; the buyer
pays only on success. **Best for hard, high-value problems** (theorems, algorithm discovery,
long-standing bugs). Risk: nodes won't participate if `P(win)` is too low or too concentrated.

**B. Effort-dominant** (`rate` covers most of the budget): nodes are paid mostly for trying,
with a bonus on acceptance. **Best for routine work** (dependency upgrades, flaky-test fixes)
where acceptance rates are high and predictable. Risk: pays for garbage.

`[DESIGN]` **The right default is a blend that adapts to the observed acceptance rate:**

```python
def split(quest, observed_accept_rate):
    """As acceptance becomes predictable, shift weight from bounty to rate.
    Rationale: the bounty exists to price RISK. No risk ⇒ no risk premium."""
    r = clamp(observed_accept_rate, 0.001, 0.5)
    effort_fraction = clamp(r / 0.5, 0.05, 0.80)
    return effort_fraction, 1 - effort_fraction
```

## 9.4 The effort floor — the hardest economic question in the design

**If nodes are paid only on acceptance, and acceptance is 2%, then 98% of the network's
electricity is unpaid.** For volunteers on a mission, that's fine (it's how Folding@home
works). For a commercial market, it is a problem.

`[DESIGN]` **Answer: a small effort floor, gated on *provable participation*, funded from the
Quest budget and capped.**

```python
effort_floor(node, cand) =
    BASE_RATE * tokens_generated
    * quality_gate(cand)          # 0 if it failed `syntax`/`tamper`/`identity`(dup) stages
    * node_reputation(node)       # EWMA of that node's historical acceptance rate
    * min(1, budget.effort_pool_remaining / expected_remaining_effort)
```

Key properties:
- **A candidate that fails the *free* stages earns nothing.** You cannot farm the floor by
  emitting garbage, because garbage dies at cost-0.001 and pays 0. **The cascade is also the
  anti-fraud mechanism.** This is an elegant and important consequence of §8.7.
- **Reputation multiplies.** A node with a 0% historical acceptance rate converges to 0 payout,
  which is the *correct* punishment for a node running a useless model — without the network
  ever having to prove misbehaviour.
- **The pool is capped** so the buyer's downside is bounded.

## 9.5 Duplicate handling

If 40 nodes find the same patch, paying all 40 the bounty is a 40× overpay; paying only the
first destroys the incentive to try a *crowded* Quest.

`[DESIGN]` **Rule:** first *distinct* accepted candidate per behaviour cluster (§8.9) gets the
bounty. Later duplicates get `duplicate_share = bounty × δ / rank`, with `δ ≈ 0.05`. Genuinely
*different* accepted solutions (different cluster, or Pareto-better on a secondary metric) get
a **novelty bonus**, because for optimization Quests a diverse set of accepted solutions is
worth more than one.

> This also produces the right *emergent* behaviour: nodes are incentivized to pick
> **under-explored arms**, because crowded arms produce duplicates. **The economics push the
> network toward diversity, which is exactly the algorithmic objective.** Incentive alignment
> between the market layer and the search layer is rare and valuable — call it out explicitly
> in the whitepaper.

## 9.6 Reverse auction for capacity (Akash's good idea, correctly applied)

Nodes bid `credits per 1k tokens` for a Quest; the coordinator selects on a *value* score, not
price alone:

```python
score(bid) = expected_marginal_coverage(bid.arm, quest) / bid.price
```

**This is the correct generalization of Akash's reverse auction:** don't buy the cheapest
compute, buy the **cheapest probability of success**. A slow node running a rare, well-suited
model may be worth 10× a fast node running the same model as everyone else.

## 9.7 The token question (an explicit decision the next session must make)

`docs/roadmap.md` states a non-goal: **"no blockchain or token settlement."** The plan
**endorses keeping that**, at least through Phase 3, for four reasons:

1. **PoA does not need a chain.** Correctness is established by re-execution, not consensus.
2. **MeshPool Private (the revenue track) must not touch a token.** Enterprises will not.
3. Every token-based competitor in Part 3 has had its incentive layer gamed. The token *is*
   the attack surface.
4. Regulatory drag would dominate the roadmap.

**What to build instead:** an internal **credit ledger** (already exists — `common/trust.py`
with conservation invariants), with:
- fiat on-ramp for commercial Quests (Stripe, later),
- **free credits earned by contributing compute** (the Folding@home model, but with a
  redeemable unit — donate cycles, earn credits, spend them on your own Quests),
- and a documented, *optional*, later bridge to on-chain escrow **only** for public bounties,
  if and when there is demand. Keep it a **plugin behind an interface**, never a dependency.

`[DESIGN]` The "earn credits by contributing, spend credits on your own Quests" loop is the
key growth mechanic: it makes the network's supply and demand **the same people**, which
solves the cold-start problem that killed Golem.

## 9.8 Unit economics sketch `[ESTIMATE] — must be re-derived with real measurements`

Assume a code-repair Quest on a typical open-source bug:

| Quantity | Value | Source |
|---|---|---|
| Samples to first acceptance | ~150 | `[ESTIMATE]` from Large Language Monkeys coverage curves |
| Tokens per sample (prompt shared, output) | ~600 out | `[ESTIMATE]` |
| Effective generation throughput per node | 30 tok/s | `[ESTIMATE]` consumer GPU, 7B Q4 |
| Node-seconds of generation | 150 × 600/30 = **3,000 s** | derived |
| With prefix sharing (§8.8), effective | ~**1,000–1,500 s** | derived, `[ESTIMATE]` |
| Evaluation cost with cascade (§8.7) | ~0.29 × 150 × 1 s ≈ **44 s** | derived |
| **Total ≈ 20–25 machine-minutes** | | |
| At $0.05/machine-hour marginal cost | **≈ $0.02** | derived |
| Datacenter equivalent (frontier API, 150 × 8k-token prompts) | **$3–$30** | `[ESTIMATE]` |

> Even with a 10× error bar, **the mesh's cost structure for search is dramatically better than
> an API's**, because (a) the marginal cost of an idle machine is electricity only, and (b)
> small open models plus many samples plus a verifier can substitute for one frontier call.
> **This is the arithmetic that makes the business real. Measure it in Phase 2 and publish it.**

---

# PART 10 — SECURITY & ADVERSARIAL MODEL

## 10.1 Threat table

| # | Adversary | Goal | Attack | Mitigation | Residual |
|---|---|---|---|---|---|
| T1 | Lazy node | Get paid for nothing | Return garbage/empty candidates | Cascade kills at cost 0.001; effort floor gated on quality; reputation → 0 | none |
| T2 | Cheating node | Get paid without generating | Submit a *cached/stolen* candidate from another node | Candidate CID dedup + first-mover rule (§9.5); per-Quest nonce in prompt; generation attestation | Low: they'd get `duplicate_share` only |
| T3 | Reward hacker | Pass the evaluator without solving | Special-case tests, edit tests, monkeypatch, `sys.exit(0)`, delete assertions | 🔴 §10.2 — the big one | **The main residual risk** |
| T4 | Malicious Quest author | Run malware on volunteers | Evaluator that mines crypto / exfiltrates / attacks LAN | Sandbox (no net, no host FS, caps); publisher reputation & staking; consent classes; **kill-switch** | Medium — needs operational vigilance |
| T5 | Malicious Quest author | Steal compute for free | Post Quests, always reject candidates | Escrow locked at submission; verdicts are **replayable** — a Quest author who rejects a candidate that others can show *passes* loses their stake | Low |
| T6 | Sybil | Multiply effort-floor earnings | 10,000 fake nodes | Effort floor × reputation (new nodes ≈ 0); proof-of-hardware benchmark Quests; per-identity rate limits; optional stake | Medium |
| T7 | Colluding nodes | Split bounties / suppress others | Coordinate to only submit weak candidates | Bounty goes to *whoever* passes; any single honest node breaks the cartel; open participation | Low |
| T8 | Data thief | Read proprietary code from Quests | Join the network, receive context packs | 🔴 §10.5 visibility tiers; MeshPool Private; context minimization | **Structural — solved by Track B, not crypto** |
| T9 | Model extractor | Steal a licensed model from a node | Exfiltrate weights from a generator node | Node runs *its own* models; MeshPool never ships licensed weights to untrusted nodes | Low by design |
| T10 | Poisoner | Corrupt the archive / adapters | Feed bad "accepted" data into the adapter training loop | Only *replay-verified* receipts enter the training set; held-out gate; provenance chain | Medium |
| T11 | DoS | Waste network capacity | Flood cheap Quests | Escrow + submission fee + per-author rate limits | Low |
| T12 | Flaky evaluator (accidental) | — | Non-deterministic verdicts destroy trust | 🔴 2× determinism check at intake (§7.2.3); flakiness score per evaluator; auto-suspend | Medium — must be actively monitored |

## 10.2 Reward hacking — the central security problem, in depth

**This deserves its own defence-in-depth stack because it is the one attack that PoA cannot
dismiss structurally.**

| Layer | Defence | Catches |
|---|---|---|
| **L1 — Tamper gate (free)** | Reject any candidate that modifies files under the test path, the evaluator, `conftest.py`, CI config, or that introduces `sys.exit`, `os._exit`, `pytest.skip`, `@unittest.skip`, `monkeypatch` on assert machinery, `__import__('os').environ` tricks. Whitelist the editable file set at Quest creation. | Crude tampering. ~Free. |
| **L2 — Held-out oracle** | Quest author supplies a secret suite. Encrypted; decrypted only inside an **Anchor** node's sandbox; never shipped to generators. Candidate must pass **both**. | Test-specific special-casing. |
| **L3 — Metamorphic / property checks** | Auto-generate property tests (Hypothesis/QuickCheck-style) from type signatures & invariants; require the candidate to satisfy them. | Overfitting to concrete inputs. |
| **L4 — Mutation testing the candidate** | Mutate the *original* code; a correct patch should still fail the mutants. Also: mutate the tests slightly (rename fixtures, reorder) — a special-cased patch breaks. | Structural overfit. MeshPool already has a mutation culture. |
| **L5 — Cross-candidate behavioural agreement** | Run all accepted candidates on generated probe inputs; a candidate whose behaviour is an outlier among accepted peers is flagged. | Sophisticated single-candidate hacks. |
| **L6 — Diff review heuristics** | Flag suspiciously large deletions, disabled code paths, broadened exception handlers, changed default arguments. Surface to the buyer. | Semantic damage that tests don't cover. |
| **L7 — Human/LLM review as the last mile** | For high-value Quests, an optional review step *after* mechanical acceptance. **Never as the primary gate** (that's Bittensor's mistake). | Everything else. |

> 🔴 **Honesty requirement for the product:** MeshPool must state plainly —
> *"PoA guarantees your predicate accepted the artifact. It does not guarantee your predicate
> was a good specification."* The value proposition is **"you get exactly what you asked for,
> at massive scale, cheaply, provably"** — and a large part of the product is *helping the user
> ask well* (auto-generated held-out tests, property tests, tamper gates on by default).
> This honesty is on-brand for a repo whose README already contains a 450-line
> "Honest constraints" section. **Do not lose that voice.**

### 10.2.1 🔴 THE OPTIMIZER–VERIFIER COUPLING LAW (a design law, discovered this session)

`[VERIFIED-EXT]` Snell et al. (arXiv 2408.03314 §5.3, fetched this session) report something
that most readers skip past, and it is **an alarm bell for the Archipelago design**:

> On **easy** questions, beam search — *the stronger optimizer* — **degrades performance as
> the generation budget increases**, showing "signs of exploitation of the PRM signal."
> Applying stronger optimization "only further amplif[ies] any spurious features learned by
> the verifier." Lookahead search, the *strongest* optimizer tested, **underperformed the most.**

Generalize it:

> ### **The Optimizer–Verifier Coupling Law**
> **The harder your search optimizes, the harder it attacks your verifier.**
> Search strength and verifier strength must be raised together. Increasing search strength
> against a fixed, weak verifier produces *worse* results than not searching at all.

**This has four concrete consequences for MeshPool, and they are not optional:**

| # | Consequence | Implementation |
|---|---|---|
| 1 | **Archipelago is a *much* stronger optimizer than i.i.d. sampling.** It will therefore find evaluator exploits that plain sampling never would. The pivot's core algorithm is *also* its core security risk. | Ship the full §10.2 stack **before** enabling evolution, not after. Gate `--evolve` behind a verifier-strength check. |
| 2 | **Search strength must be an output of verifier strength, not a user preference.** | New: `verifier_strength_score ∈ [0,1]` computed at intake from held-out presence, tamper-gate coverage, mutation-test kill rate, property-test presence, and suite size. The coordinator **caps** the archipelago's selection pressure as a function of it. |
| 3 | **Easy problems need *less* optimization pressure, not more.** (Counterintuitive, and the exact opposite of what a naive scheduler does.) | The bandit conditions on estimated difficulty (§8.11): low difficulty ⇒ prefer best-of-N with weak selection pressure; high difficulty ⇒ prefer strong evolutionary pressure. |
| 4 | **Monitor for it in production.** A rising acceptance rate combined with a *falling* held-out pass rate is the unmistakable signature of verifier exploitation. | New dashboard metric: **exploitation index = (public pass rate) − (held-out pass rate)**, tracked per Quest. Auto-throttle selection pressure when it climbs. |

```python
def max_selection_pressure(quest) -> float:
    """THE COUPLING LAW, enforced. Do not let a strong optimizer loose on a weak verifier."""
    v = verifier_strength(quest)          # held-out? tamper gates? mutation kill rate? properties?
    if v < 0.3:
        return 0.0                        # i.i.d. sampling ONLY. No evolution. No archive pressure.
    if v < 0.6:
        return 0.5                        # mild elitism, high temperature, no aggressive exploitation
    return 1.0                            # full Archipelago

def verifier_strength(quest) -> float:
    return weighted_mean({
        "has_held_out":      1.0 if quest.evaluator.held_out else 0.0,   # weight 0.35
        "tamper_coverage":   quest.tamper_gate_coverage,                 # weight 0.20
        "mutation_kill":     quest.mutation_kill_rate,                   # weight 0.25
        "has_properties":    1.0 if quest.property_tests else 0.0,       # weight 0.10
        "suite_breadth":     min(1.0, quest.n_tests / 50),               # weight 0.10
    })
```

> **Why this matters strategically:** it turns a *risk* into a **product feature and a moat**.
> MeshPool becomes the only system that measures the strength of your verifier and tells you
> *"your test suite is too weak for me to search this hard — here are the three held-out tests
> I'd like you to add."* **Nobody else does this, because nobody else's optimizer is strong
> enough to need it.** It is also a natural upsell: verifier-strengthening is itself a Quest
> class (P4-6).

## 10.3 Sandbox escape

The sandbox runs hostile code on volunteers' machines. Requirements, in order:
1. **T0 (WASM) is the default** for anything that can be expressed as a pure function. Its
   escape surface is a rounding error compared to containers.
2. **T1 containers must default to gVisor or a microVM (Firecracker/Cloud Hypervisor)** on
   Linux, and **Hyper-V isolation / WSL2 with a dedicated VM** on Windows. Plain Docker on a
   volunteer's daily-driver machine is **not acceptable** for public Quests. `[DESIGN]`
3. **No network. No host mounts. Read-only rootfs. Dropped caps. seccomp default-deny.
   Per-run ephemeral scratch, destroyed after.**
4. **Egress firewall at the agent level** (belt and braces) — the node agent itself should
   refuse outbound connections originating from sandbox PIDs.
5. **Resource caps enforced by the OS**, not by the evaluator's honour.
6. A **panic button** in the CLI/console/tray that stops all sandboxes immediately, and an
   auto-pause on user activity (like Folding@home/BOINC idle detection).

## 10.4 Evaluator determinism enforcement

```python
def assert_deterministic_evaluator(evaluator, trials=2):
    """Run the SAME evaluator on the SAME reference candidate N times on the SAME machine.
    Any verdict-hash divergence ⇒ REJECT THE QUEST AT INTAKE."""
    hashes = {canonical_hash(sandbox.run(evaluator, ref_candidate)) for _ in range(trials)}
    if len(hashes) != 1:
        raise QuestRejected("evaluator is non-deterministic; fix flakiness before submitting")
```

Plus continuous monitoring: track per-evaluator verdict divergence across nodes. A rising
divergence rate auto-suspends the Quest and refunds escrow. **A flaky evaluator is an
economic bug, and it must be treated as a first-class failure mode with its own dashboard.**

## 10.5 Privacy: the honest answer

**The problem:** generating a patch requires the generator to *see the code*. There is no way
around this for the open network. State it plainly rather than hand-waving.

`[DESIGN]` **Four visibility tiers:**

| Tier | Who sees the context | Mechanism | Use for |
|---|---|---|---|
| **V0 — Public** | Everyone | None needed | OSS, benchmarks, math, science, fuzzing public targets |
| **V1 — Minimized** | Generators, but only a *minimal slice* | Context minimization: send only the failing function + its call sites + the failing test, with identifiers optionally α-renamed and string literals redacted. **The evaluator (which needs the whole repo) runs on Anchors only.** | Semi-sensitive code |
| **V2 — Licensed + staked** | Vetted nodes under contract | Legal + stake + identity + audit log of exactly which bytes went where | Commercial Quests |
| **V3 — Private fleet** | Only the customer's own machines | **MeshPool Private** — the network software, zero external nodes | Enterprise. **The real answer.** |

> **The key strategic move:** *do not try to solve the privacy problem cryptographically.*
> FHE is ~10⁴–10⁶× overhead; TEEs keep getting broken (Part 3 §D lists nine published attacks).
> **Solve it architecturally: the same product, deployed inside the trust boundary.**
> This is why Track B is not a compromise — it is the correct engineering answer.

**A genuinely nice property of V1:** because the *evaluator* is what needs full repo access
and the *generator* only needs a slice, the architecture naturally splits along the privacy
boundary. Anchors (trusted, few, possibly customer-operated) hold the repo; Drifters (many,
untrusted) see only a redacted slice. **Nobody else's architecture has this split, because
nobody else separates generation from verification.** Write this up.

## 10.6 What happens when the network is used for harm

Explicit, and needed before any public launch:
- **Prohibited Quest classes**: malware generation, exploit weaponization against systems the
  author doesn't own, CSAM-adjacent anything, password/hash cracking against third parties,
  spam/fraud content generation.
- **Fuzzing is allowed only against targets the author can demonstrate ownership/permission
  for**, or against a curated list of OSS projects with published security policies (OSS-Fuzz
  model). This is important because "find me a 0-day" is otherwise a *perfect* Quest.
- **Node-side consent classes** so a volunteer can refuse whole categories.
- **Publisher identity + stake** for anything beyond the public science/OSS classes.
- **A public, append-only Quest log** so the network is auditable.

`[DESIGN]` This section is not optional boilerplate — a verifiable-solution-mining network is
**genuinely dual-use** (the same machinery that fixes a bug finds a bug), and having a
credible answer is a prerequisite for recruiting volunteers and for any institutional
partnership.

---

# PART 11 — IMPLEMENTATION ROADMAP (GRAIN LEVEL)

> **How to read this part.** Every task has: an **ID**, a **path**, a **description**, an
> **acceptance criterion**, and a **gate** (the `sim/scenarios/` file that proves it).
> The gate is not optional — this repo's culture is that a feature without a non-vacuous gate
> does not exist. Tasks are ordered by dependency. `[BLOCKING]` marks work that everything
> else waits on.

## 11.0 The existing module map (ground truth, verified)

```
meshpool/
├── launcher.py
├── agent/        daemon.py  donation.py  probe.py  runtime.py  sandbox.py  tunnel_client.py
├── batch/        dag.py  engine.py  functions.py  registry_ref.py
├── cli/          main.py
├── cloud/        nat.py  quic.py  swim.py  tunnel.py
├── common/       auth.py billing.py config.py db.py egress.py faultinject.py identity.py
│                 logging.py models.py netutil.py proactor_fix.py secrets.py serve_guard.py
│                 tracing.py transport.py trust.py vivaldi.py
├── controlplane/ accounts.py app.py nodeclass.py openai_api.py registry.py scheduler.py
│                 spend.py timeline.py topology.py verification.py
├── functions/    wasm.py
├── hosting/      demo_app.py engine.py frameworks.py site.py
├── inference/    adapters.py batching.py bench.py corpus.py engine.py gguf_loader.py
│                 gguf_tokenizer.py kvcache.py lora.py model.py pipeline.py remote.py
│                 spd.py train.py
└── storage/      audit.py blobstore.py crypto.py erasure.py engine.py s3_api.py
                  s3_credentials.py
```

**New packages to create:**
```
meshpool/
├── quest/        models.py  coordinator.py  intake.py  budget.py  cli_verbs.py  registry.py
├── search/       archive.py  island.py  archipelago.py  migration.py  prompts.py
│                 bandit.py  arms.py  descriptors.py  stopping.py
├── evaluate/     sandbox_wasm.py  sandbox_oci.py  sandbox_detvm.py  cascade.py
│                 determinism.py  oracle.py  tamper.py  verdict.py
├── generate/     fleet.py  backends/llamacpp.py  backends/vllm.py  backends/mlx.py
│                 backends/api_passthrough.py  prefix.py  adapter_swarm.py  residency.py
└── receipts/     receipt.py  canonical.py  verify.py
```

---

## PHASE 0 — UNBLOCK (weeks 0–2) `[BLOCKING]`

> **Nothing in this plan is trustworthy until Phase 0 is green.** These four tasks remove the
> four gaps identified in Part 2 that make every downstream claim unverifiable.

### P0-1 `[BLOCKING]` Obtain and verify real pretrained weights
- **Path:** `scripts/fetch_weights.py` (new), `sim/scenarios/real_weights_smoke.py` (new)
- **Do:** Fix the corporate-TLS blocker. In order: (a) export the enterprise root CA and set
  `REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE`/`CURL_CA_BUNDLE`; (b) try `huggingface-cli download`
  with `HF_HUB_ENABLE_HF_TRANSFER=1`; (c) try the Ollama registry; (d) fetch on an
  unfiltered network and copy in. **Always verify the SHA-256 against a pinned digest** —
  the digest, not the TLS chain, is the security boundary.
- **Models to pin:** `Qwen2.5-0.5B-Instruct-Q4_K_M`, `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M`,
  `Llama-3.2-1B-Instruct-Q4_K_M`, `SmolLM2-360M-Instruct-Q8_0`.
- **Accept:** `meshpool bench --model qwen2.5-0.5b --prompt-file golden/p1.txt` emits the
  exact 20 tokens in `golden/p1.tokens`.
- **Gate:** `real_weights_smoke.py` — loads each pinned GGUF, asserts golden token match,
  asserts tokenizer round-trip on a 5k-char UTF-8 corpus including emoji/CJK/RTL.
- **Why it's blocking:** every performance, quality, and cost number in this document is
  meaningless without it, and the GGUF loader's RoPE-convention switch (Part 2) is *untested
  against reality*.

### P0-2 `[BLOCKING]` Real inference backend — llama.cpp
- **Path:** `meshpool/generate/backends/llamacpp.py` (new); register in `inference/adapters.py`
- **Do:** Drive `llama-server` over HTTP (simplest, most portable, process-isolated) with an
  optional `libllama` ctypes path later. Support: CUDA, ROCm, Metal, Vulkan, CPU. Expose
  `n_gpu_layers`, `--parallel`, `--cont-batching`, `--slot-save-path` (prompt cache),
  `--seed`, `--temp`, `--top-p`, `--grammar`.
- **Accept:** `MeshShardBackend` (NumPy) and `LlamaCppBackend` produce **identical greedy
  token sequences** for the same 1B model + prompt at temperature 0. *(If they don't, that is
  itself an extremely valuable finding about the NumPy reference — investigate, don't
  suppress.)*
- **Gate:** `backend_equivalence_llamacpp.py` — greedy-equality across 20 prompts; mutation
  catalogue entry that corrupts the RoPE base and asserts the gate fails.
- **Note:** **Keep `MeshShardBackend`.** It is the determinism oracle that makes every other
  backend testable. Its slowness is irrelevant; its readability is the asset.

### P0-3 `[BLOCKING]` Container evaluator runtime (`DockerRuntime` → real)
- **Path:** `meshpool/agent/runtime.py` (implement), `meshpool/evaluate/sandbox_oci.py` (new)
- **Do:** Implement OCI execution against Docker/Podman with the full determinism + isolation
  checklist from §7.2.3. Detect and prefer gVisor (`--runtime=runsc`) when available.
  Windows: Hyper-V isolated containers or a WSL2 VM; degrade to "evaluation unavailable on
  this node" rather than running unisolated.
- **Accept:** `meshpool evaluate --image python:3.12-slim@sha256:… --cmd "pytest -q"` runs a
  fixture repo, produces a verdict, and **two consecutive runs produce identical verdict hashes.**
- **Gate:** `oci_sandbox_determinism.py`, `oci_sandbox_isolation.py` (assert no network:
  a container that tries `curl example.com` must fail; assert no host FS: `/etc/passwd` of the
  host is not visible; assert memory/pids/wall limits actually kill).
- **Mutation entries:** remove `--network=none` → isolation gate must fail; remove
  `PYTHONHASHSEED=0` → determinism gate must fail on a dict-ordering-sensitive fixture.

### P0-4 `[BLOCKING]` Transport-E2E fraud gate
- **Path:** `sim/scenarios/fraud_e2e_transport.py` (new)
- **Do:** Today the fraud-detection gate runs in-process. Rebuild it so a *real* malicious
  agent process, over the *real* transport, submits a forged verdict and is rejected by the
  coordinator. This is the difference between "our function returns False" and "the system
  is secure."
- **Accept:** a rogue agent binary that (a) signs a verdict with a wrong key, (b) replays a
  valid verdict for a different candidate, (c) claims acceptance without an evaluator run,
  and (d) submits a verdict whose `verdict_hash` doesn't match its payload — is rejected in
  all four cases with distinct, asserted error codes.
- **Gate:** the scenario itself; plus a mutation entry per case.

---

## PHASE 1 — THE QUEST PRIMITIVE (weeks 2–8)

### P1-1 Quest data model
- **Path:** `meshpool/quest/models.py`
- **Do:** Pydantic models for `Quest`, `Evaluator`, `Candidate`, `Verdict`, `ReplayReceipt`,
  `Arm`, `Elite`, `Archive`, `Island`, `QuestResult`. Mirror the schemas in §7.2.2/§8.6
  exactly. Add them to `common/models.py`'s registry and to `proto/meshpool.proto` (keeping
  the existing mirror discipline).
- **Accept:** round-trip JSON serialization is stable and canonical-hash-stable across
  Python and Go.
- **Gate:** `quest_schema_roundtrip.py` — including a **cross-language** check that the Go
  agent computes the same canonical hash as Python for 100 fuzzed instances. 🔴 This gate is
  load-bearing for receipts.

### P1-2 Canonical hashing
- **Path:** `meshpool/receipts/canonical.py`
- **Do:** Deterministic JSON: sorted keys, no insignificant whitespace, UTF-8 NFC, integers
  as integers, floats via shortest round-trip repr, **reject NaN/±Inf**, explicit null
  handling, byte strings as lowercase hex. Mirror in Go.
- **Accept:** 10,000 property-based fuzzed structures hash identically in Python and Go.
- **Gate:** `canonical_hash_cross_language.py`.

### P1-3 Evaluator intake & determinism check
- **Path:** `meshpool/evaluate/determinism.py`, `meshpool/quest/intake.py`
- **Do:** Build the evaluator image/module, content-address it, run the 2× determinism check
  (§10.4), record a flakiness score. Reject non-deterministic evaluators at intake with an
  actionable error message.
- **Accept:** a deliberately flaky evaluator (`random.random() > 0.5`) is rejected; a
  deterministic one is admitted; a *timing*-flaky one (`assert elapsed < 0.001`) is caught by
  running the two trials under different CPU load.
- **Gate:** `evaluator_intake_flaky_reject.py`.

### P1-4 Verification cascade
- **Path:** `meshpool/evaluate/cascade.py`, `meshpool/evaluate/tamper.py`
- **Do:** Implement §8.7 on top of `batch/dag.py`. Stages: identity/dedup → syntax → tamper →
  static → smoke → public → held-out → replicate. Early exit. **Always emit `artifacts`** on
  failure (this is the evolutionary side-channel and it is not optional).
- **Accept:** on a 200-candidate fixture where 60% are syntactically invalid, measured average
  cost is ≥5× lower than the naive full-suite path; every failure carries a non-empty
  `artifacts.stderr_tail`.
- **Gate:** `cascade_early_exit_cost.py` (asserts the cost ratio — **non-vacuous**, i.e. it
  fails if the cascade is disabled), `cascade_tamper_detection.py` (12 tamper archetypes:
  editing test files, `sys.exit(0)`, `pytest.skip`, broadening `except`, deleting asserts,
  monkeypatching `assertEqual`, editing `conftest.py`, changing CI config, adding
  `@pytest.mark.xfail`, `os._exit`, `unittest.skip`, rewriting the fixture).

### P1-5 Held-out oracle
- **Path:** `meshpool/evaluate/oracle.py`
- **Do:** Encrypted secret test bundle. Key held by the Quest author + Anchor nodes only.
  Decrypted **into the sandbox's memory only**, never to disk on the evaluator node, never
  transmitted to generators. Reuse `storage/crypto.py`.
- **Accept:** a candidate that special-cases the public tests passes public and **fails
  held-out**; a genuinely correct candidate passes both; a generator node's full network
  capture contains **zero bytes** of the held-out suite.
- **Gate:** `heldout_defeats_overfit.py` — with a *hand-written* overfitted patch as the
  positive control (non-vacuity: the gate must fail if held-out checking is disabled).

### P1-6 Replay receipts + `meshpool verify`
- **Path:** `meshpool/receipts/receipt.py`, `receipts/verify.py`, CLI verb in `cli/main.py`
- **Do:** Mint, sign, store, fetch, and re-execute.
- **Accept:** `meshpool verify mp://receipt/<cid>` on a **different machine**, with only the
  receipt CID and network access to the CAS, reproduces the verdict.
- **Gate:** `receipt_replay_crossmachine.py` (simulate two node identities and separate
  filesystems), `receipt_tamper_reject.py` (mutate each field; every mutation must be caught).

### P1-7 PoA settlement
- **Path:** `meshpool/controlplane/verification.py` (extend), `common/trust.py` (extend)
- **Do:** Add `VerificationClass.PROOF_OF_ACCEPTANCE`. Wire escrow → verdict → payout →
  ledger. Implement the effort floor (§9.4), duplicate share (§9.5), novelty bonus.
- **Accept:** the existing **ledger conservation invariant** holds across 10,000 randomized
  Quest lifecycles including crashes, duplicate acceptances, disputes, and refunds.
- **Gate:** `poa_ledger_conservation.py` (property-based, with fault injection via
  `common/faultinject.py`), `poa_effort_floor_antifarm.py` (a node emitting pure garbage earns
  exactly 0).

### P1-8 The `meshpool quest fix` vertical slice 🔴
- **Path:** `meshpool/quest/cli_verbs.py`, `meshpool/quest/coordinator.py`
- **Do:** End-to-end **single-island, no-evolution, i.i.d. sampling** version. Repo → context
  pack → N samples → cascade → first acceptance → patch file + receipt.
- **Accept:** on a curated set of **20 real bugs** (see P1-9), ≥8 are fixed within a 200-sample
  budget, with receipts, with zero tamper acceptances.
- **Gate:** `quest_fix_e2e.py` (runs 3 of the 20 in CI with a tiny model and a tight budget).
- 🔴 **This is the milestone that proves the entire thesis. Everything before it is
  scaffolding; everything after it is scaling.**

### P1-9 The benchmark harness
- **Path:** `bench/quests/` (new), `bench/run.py`
- **Do:** Curate 20–50 tasks: SWE-bench Lite/Verified subset, HumanEval+/MBPP+, a handful of
  real bugs from small OSS repos, 5 MiniF2F Lean problems, 3 kernel-optimization tasks. Record
  a **coverage curve** (`accepted@k` vs k) for every arm.
- 🔴 **Mandatory baseline:** every benchmark run must also measure **the bash-loop baseline** —
  N sequential calls to a single frontier API with the same context pack and the same
  evaluator. Per §5.8, that is the real competitor. Report `cost per accepted solution` and
  `acceptance@k` for both, side by side, always.
- **Accept:** reproducible `bench/run.py --suite lite` producing a JSON report with provenance
  tiers on every number, in the repo's existing `MEASURED/PROJECTED/CITED` style.
- **Gate:** `bench_harness_determinism.py`, `bench_baseline_present.py` (fails if a report is
  published without the baseline column — this gate exists to prevent flattering benchmarks).
- **Why it matters:** the coverage curve *is* the product's core empirical claim, and §4.1.3
  says the target to beat is **≥3× cheaper than single-shot frontier calls**. Instrument it
  from day one, publish it, and let it drive the bandit's priors.

### P1-10 Verifier-strength scoring & the coupling-law cap `[NEW — see §10.2.1]`
- **Path:** `meshpool/evaluate/strength.py`, wired into `quest/intake.py` and `search/`
- **Do:** Implement `verifier_strength()` and `max_selection_pressure()` from §10.2.1. Compute
  at intake from held-out presence, tamper-gate coverage, mutation kill rate, property tests,
  and suite breadth. **Hard-cap the archipelago's selection pressure as a function of it.**
  Surface the score and concrete improvement advice to the Quest author.
- **Accept:** a Quest with a weak evaluator (`assert result is not None`) scores <0.3 and is
  **forced to i.i.d. sampling with evolution disabled**; a Quest with held-out tests, tamper
  gates, and property tests scores >0.6 and unlocks full Archipelago.
- **Gate:** `coupling_law_enforced.py` — assert evolution cannot be enabled below the strength
  threshold, **plus a non-vacuity control**: with the cap removed, a deliberately weak
  evaluator must be demonstrably exploited by the evolutionary search (i.e. acceptance rises
  while held-out pass rate falls). 🔴 **This gate encodes the most important safety property in
  the system — that stronger search does not silently become stronger cheating.**
- **Why it's in Phase 1, not Phase 4:** the cap must exist *before* Archipelago ships (P2-2),
  or the first thing the search engine learns is how to cheat.

### P1-11 Context Pack builder `[NEW — see §7.6]`
- **Path:** `meshpool/quest/context.py`
- **Do:** The pipeline in §7.6 — traceback seeding, call-graph expansion, multi-signal ranking,
  greedy knapsack under a token budget, signature-only overflow, privacy redaction, and
  **the tamper allowlist** (`editable_paths`).
- **Accept:** on the 20-bug benchmark, the ranked packer beats naive whole-file inclusion at
  equal token count by ≥15% relative acceptance rate.
- **Gate:** `context_pack_ablation.py`, `context_pack_editable_allowlist.py` (the allowlist must
  never include test files or the evaluator — property-tested across all 20 benchmark repos).

---

## PHASE 2 — THE SEARCH ENGINE (weeks 8–18)

### P2-1 MAP-Elites archive
- **Path:** `meshpool/search/archive.py`, `search/descriptors.py`
- **Do:** §8.2.1 and §8.2.2. Blob-store-backed, CRDT `merge`. Descriptor plugins per Quest class.
- **Accept:** `merge` is provably commutative, associative, and idempotent under property tests
  (this is what makes coordinator-free operation safe).
- **Gate:** `archive_crdt_properties.py` (Hypothesis-driven; 3-way merges in all orders),
  `archive_qd_improves.py` (on a synthetic rugged fitness landscape, MAP-Elites beats
  hill-climbing on final best-score — **non-vacuous**: it must fail if the archive is replaced
  with "keep global best only").

### P2-2 Islands & Archipelago
- **Path:** `meshpool/search/island.py`, `search/archipelago.py`, `search/migration.py`
- **Do:** Map Islands onto `Constellation`s from `controlplane/scheduler.py`. RTT-aware ring
  migration (§8.2.4). Islands must run autonomously through a coordinator outage.
- **Accept:** with the coordinator killed for 120 s, islands keep generating and evaluating,
  and archives reconcile losslessly on reconnect.
- **Gate:** `archipelago_coordinator_outage.py`, `archipelago_migration_bandwidth.py`
  (asserts migration traffic < 100 KB/island/minute — the number that makes WAN operation
  viable), `archipelago_beats_single_island.py` (on a benchmark subset, N islands beat 1
  island at equal total sample budget — the *scientific* claim of the architecture).

### P2-3 Prompt assembly
- **Path:** `meshpool/search/prompts.py`
- **Do:** §8.2.3 — best-shot, diverse-shot, artifact side-channel, diff vs rewrite modes,
  context minimization hooks for privacy tier V1.
- **Accept:** ablations measurably matter: removing the artifact side-channel measurably
  lowers acceptance rate on the benchmark.
- **Gate:** `prompt_ablation_sidechannel.py` (this gate encodes a *finding*, not just a
  behaviour — record the measured delta in `docs/benchmarks.md`).

### P2-4 Portfolio bandit
- **Path:** `meshpool/search/bandit.py`, `search/arms.py`
- **Do:** §8.3 — **factored** Thompson sampling over arm *features*, hierarchical shrinkage
  across problem clusters, correlation-penalized marginal gain, cost-benefit greedy allocation.
- **Accept:** on a synthetic environment with known per-arm/per-cluster success rates, the
  bandit reaches ≥90% of oracle coverage within 2,000 pulls, and beats uniform-random
  allocation by ≥25%.
- **Gate:** `bandit_regret_bound.py`, `bandit_diversity_not_greedy.py` (assert it does **not**
  collapse onto the single best arm — the correlation penalty must be doing work).

### P2-5 Weight-affinity scheduling & residency planning
- **Path:** `meshpool/generate/residency.py`, extend `controlplane/scheduler.py`
- **Do:** §8.4 — enumerate warm arms first; greedy facility-location residency planner;
  amortized cold-start cost model.
- **Accept:** in a simulated 200-node mesh with realistic uplinks, total weight bytes
  transferred per accepted solution drops ≥10× vs. affinity-blind scheduling, at ≤10% coverage
  loss.
- **Gate:** `weight_affinity_bytes.py`, `residency_plan_coverage.py`.

### P2-6 Adapter Swarm
- **Path:** `meshpool/generate/adapter_swarm.py`, extend `inference/lora.py`
- **Do:** §8.5 — multi-LoRA serving, adapter distribution over the existing Weight Swarm,
  adapter-grouped batching.
- **Accept:** 20 adapters over one resident base give ≥80% of the acceptance-rate benefit of
  20 distinct base models at ≤5% of the bytes.
- **Gate:** `adapter_swarm_diversity.py` (measure pairwise output decorrelation),
  `adapter_swarm_bytes.py`.

### P2-7 Prefix sharing
- **Path:** `meshpool/generate/prefix.py`, extend `inference/kvcache.py`, `inference/batching.py`
- **Do:** §8.8 — radix/prefix-tree KV reuse across the N samples of a Quest on one node;
  llama.cpp slot-save integration; SGLang backend for NVIDIA nodes.
- **Accept:** ≥2.5× generation throughput for 32 samples sharing an 8k-token prefix vs. no
  sharing, **with byte-identical outputs at temperature 0** (the batch-invariance requirement
  from §4.5).
- **Gate:** `prefix_sharing_speedup.py`, `prefix_sharing_bitexact.py` 🔴 (the second gate is
  the important one — a speedup that changes outputs breaks Replay Receipts).

### P2-8 Dedup & clustering
- **Path:** `meshpool/evaluate/verdict.py` (behavioural fingerprints), `search/archive.py`
- **Do:** §8.9 — canonical-form hashing, behavioural fingerprints on probe inputs, cluster
  ranking by size.
- **Accept:** on 500 candidates with 40 semantic duplicates in 3 clusters, all duplicates are
  grouped correctly; settlement pays one full bounty.
- **Gate:** `dedup_clusters.py`, `dedup_settlement.py`.

### P2-9 Stopping rules & budget controller
- **Path:** `meshpool/search/stopping.py`, `quest/budget.py`
- **Do:** §8.10 — plateau detection, marginal-coverage-per-cost, hard budget/wall caps,
  graceful refund of unspent escrow.
- **Accept:** across 100 simulated Quests, spend never exceeds budget, and mean spend at
  acceptance is ≥30% below the naive "always burn the whole budget" policy.
- **Gate:** `stopping_rule_savings.py`, `budget_never_exceeded.py` (property-based + fault
  injection).

### P2-10 Difficulty-adaptive strategy selection `[NEW — see §8.11]`
- **Path:** `meshpool/search/difficulty.py`, wired into `search/archipelago.py`
- **Do:** §8.11 — online difficulty estimation from the first ~10 verdicts, strategy inversion
  (easy ⇒ best-of-N with *low* selection pressure; hard ⇒ full Archipelago), and the
  `has_exploitable_gradient()` check that refuses to evolve when partial scores are flat.
- **Accept:** on the benchmark, difficulty-adaptive allocation reaches equal acceptance using
  **≥2× less compute** than a fixed strategy. *(Snell et al. report up to 4× with oracle
  difficulty bins; 2× with online estimation is a realistic target.)*
- **Gate:** `difficulty_adaptive_savings.py`, `no_evolution_without_gradient.py` (assert that a
  Quest with flat, zero-variance partial scores never triggers evolutionary pressure — this
  prevents burning budget where there is no signal to climb).

---

## PHASE 3 — THE MARKET (weeks 18–28)

| ID | Path | Task | Accept | Gate |
|---|---|---|---|---|
| P3-1 | `quest/registry.py`, `controlplane/app.py` | Quest board API: post, browse, filter, subscribe, cancel, escrow status | 1,000-Quest board paginates and filters in <100 ms | `quest_board_api.py` |
| P3-2 | `web/src/views/QuestBoard.tsx` | Console: Quest board | E2E via Playwright | `console_questboard_e2e.py` |
| P3-3 | `web/src/views/Archipelago.tsx` | 🔴 **Live search visualization** — islands as columns, MAP-Elites heatmap, streaming candidates, best-score-vs-budget curve | Renders a 3-island, 800-cell archive at 10 fps | `console_archipelago_e2e.py` |
| P3-4 | `web/src/views/Receipt.tsx` | One-click receipt verification | Verifies a real receipt in-browser (delegating execution to a node) | `console_receipt_verify_e2e.py` |
| P3-5 | `controlplane/spend.py`, `accounts.py` | Credits: earn-by-contributing, spend-on-Quests, fiat on-ramp seam | Conservation invariant holds; earn→spend loop works E2E | `credits_earn_spend_loop.py` |
| P3-6 | `quest/coordinator.py` | Reverse auction (§9.6) — bid on `coverage/price`, not price | Auction selects a slower node with a rarer, better-suited model when it maximizes coverage/price | `auction_value_not_price.py` |
| P3-7 | `common/trust.py` | Node reputation: EWMA acceptance rate, per-Quest-class | New nodes ramp; garbage nodes decay to 0 payout | `reputation_decay.py` |
| P3-8 | `agent/donation.py`, tray/CLI | Consent classes, panic button, idle detection, earnings view | Volunteer can refuse a class and it is never scheduled to them | `consent_classes_enforced.py` |
| P3-9 | `evaluate/sandbox_detvm.py` + `controlplane/verification.py` | T2 deterministic VM as the dispute referee; wire to `InteractiveBisectionDispute` | A disputed verdict is resolved bit-exactly, and the losing party's stake is slashed | `dispute_detvm_resolution.py` |

---

## PHASE 4 — SCALE & SPECIALIZE (weeks 28–52)

| ID | Task | Notes |
|---|---|---|
| P4-1 | **MeshPool Private packaging** | Single-command deploy into a corporate fleet: `meshpool private up --scale enterprise`. SSO, audit log, no external nodes, air-gap mode. 🔴 **This is the revenue track — do not defer it past Phase 4.** |
| P4-2 | **Quest class: `optimize`** | Kernel/algorithm optimization. Benchmark-as-evaluator. Needs stable timing measurement on heterogeneous nodes (use *relative* speedup vs. a baseline measured on the *same* node in the *same* run — never absolute wall time). |
| P4-3 | **Quest class: `prove`** | Lean 4 / Coq. `mathlib` cached in the image. **The purest verification asymmetry** and the best science-credibility story. |
| P4-4 | **Quest class: `fuzz`** | Coverage-guided; the crash *is* the proof. Governance-gated (§10.6). |
| P4-5 | **Quest class: `upgrade`** | Dependency upgrades + migration patches. Enormous, boring, real enterprise demand. `[DESIGN]` Probably the **highest-revenue** class. |
| P4-6 | **Quest class: `design`** | Config/hyperparameter/architecture search; RL reward functions (Eureka-style); agent scaffolds (ADAS-style). |
| P4-7 | **Self-improving adapters** | Train LoRAs on replay-verified accepted solutions. Distributed fine-tuning reuses `inference/train.py`. 🔴 The flywheel. |
| P4-8 | **Environments Hub interop** | Adopt/bridge Prime Intellect's `verifiers` environment format so existing RL environments become Quests for free. `[DESIGN]` Strongly recommended — instant supply of evaluators. |
| P4-9 | **Batch-invariant kernels** | Integrate Thinking Machines' `batch_invariant_ops` to reach Replay Tier R1. |
| P4-10 | **Raft for `Settings.ha`** | Turn the flag into an implementation; Quest coordinators need real HA once bounties are real money. |
| P4-11 | **Auto-update trust store** | The updater is built but inert (empty trust store, no signing key). Generate the key, populate the store, and gate it behind a signed-manifest gate. |
| P4-12 | **Public archive & leaderboard** | Every solved Quest published with its receipt. **The compounding asset.** |
| P4-13 | 🔴 **META-QUESTS (§6.8)** | The network evolves its own `@evolvable` heuristics, scored by the existing 80-gate suite. **Gated on M3.** Design the `@evolvable` marker + metric-harness interface in Phase 2; do not enable the loop until Archipelago is proven. Safety rules in §6.8.3 are mandatory — especially *never evolve the evaluator, the verifier-strength scorer, or the coupling-law cap*. |
| P4-14 | **Open spec + conformance suite (§13.6)** | Publish Quest/Evaluator/Verdict/Receipt as a versioned open specification with a reference implementation and conformance tests. |
| P4-15 | **Data licence plumbing (§13.5)** | Per-Quest data licence (`public` / `research` / `private`), enforced end-to-end into archive publication and adapter training. **Retrofitting this later is very expensive — build the field in Phase 1 even if unused.** |

---

## 11.1 What to DELETE or DEMOTE

Being explicit about subtraction is as important as addition.

| Component | Decision | Rationale |
|---|---|---|
| Pipeline-parallel sharded inference (`inference/pipeline.py`, `spd.py`, repartitioning) | 🟡 **Demote to a supported-but-not-flagship feature** | It works, it's honest, and MoE/large-model generation on beefy nodes still uses it. But it stops being the story. Keep the gates. |
| "Run a 1T model across your friends' laptops" positioning | 🔴 **Delete from all marketing** | 15.2 tok/s best case with free compute. It cannot carry a company. |
| `hosting/` (app hosting, demo_app, frameworks, site) | 🟡 **Freeze** | Interesting, but it's a third product. Focus. |
| `storage/s3_api.py` as a product | 🟡 **Demote to internal** | Keep it as the CAS/artifact plane. Don't sell it — Filecoin/Storj/S3 own that. |
| Generic "Mesh Functions" as a product | 🔴 **Reframe** | Mesh Functions become **the evaluator runtime**. That's their real purpose. `NOT_VERIFIABLE` was the tell. |
| OpenAI-compatible API (`controlplane/openai_api.py`) | 🟢 **Keep, but as an internal generator interface** | Useful for API-passthrough arms and for MeshPool Private. Not a product. |

## 11.2 The rewrite/reuse ledger

| Category | LOC `[ESTIMATE]` | Fate |
|---|---|---|
| Transport, gossip, NAT, QUIC, identity, auth | ~12,000 | 🟢 Reuse as-is |
| Scheduler, topology, nodeclass, vivaldi | ~5,000 | 🟢 Reuse + island metadata |
| Storage, blobstore, erasure, crypto, CAS | ~7,000 | 🟢 Reuse + CID namespaces |
| Trust, verification, spend, accounts, billing | ~6,000 | 🟡 Reuse + new `PROOF_OF_ACCEPTANCE` class |
| Batch engine + DAG | ~4,000 | 🟡 Becomes the cascade |
| WASM functions + Go wasmfn | ~2,500 | 🟡 Becomes T0 sandbox |
| Inference (GGUF, tokenizer, KV, LoRA, batching, engine) | ~15,000 | 🟡 Reuse; delegate kernels to llama.cpp |
| Inference (pipeline, spd, remote, repartitioning) | ~6,000 | 🟡 Demoted, kept |
| CLI, console, agent, launcher | ~12,000 | 🟡 Extend |
| Hosting | ~4,000 | ⚪ Frozen |
| Sim/gates/tests | ~12,000 | 🟢 **Reuse and extend — the crown jewels** |
| **New code required** | **~15,000–20,000** | `quest/`, `search/`, `evaluate/`, `generate/`, `receipts/` |

> **~85% reuse. ~15–20K new LOC.** This pivot is unusually cheap *because the hard,
> unglamorous distributed-systems work is already done.* Most startups pivot by throwing away
> code. MeshPool pivots by finally pointing its code at a workload that fits it.

---

# PART 12 — VERIFICATION STRATEGY

## 12.1 Preserve the culture (non-negotiable)

The repo's testing discipline is its most underrated asset:
- **80 acceptance gates** in `sim/scenarios/`, each with a stated claim.
- **A mutation catalogue** — every gate must *fail* under a specific injected bug. This is
  non-vacuity enforcement and it is rare even in mature companies.
- **`sim/mutation_guard.py`** journals original bytes to disk because `finally` cannot survive
  `TerminateProcess` on Windows. (It exists because a killed gate once left `openai_api.py`
  mutated and a third of a 78-gate sweep ran against a corrupted checkout.)
- The runner **diffs `git status` around every gate**.
- **Provenance tiers on every number** (`MEASURED` / `PROJECTED` / `CITED`).

> 🔴 **Rule for the next session: every new subsystem ships with (a) a gate, (b) a mutation
> entry that makes the gate fail, and (c) a provenance tier on every number it produces.
> No exceptions. This is why the repo can be trusted, and it is why the pivot is credible.**

## 12.2 The "verify the verifier" doctrine

The repo already learned this the hard way: `npx tsc --noEmit` in `web/` exits 0 **without
opening a single file** because of a solution-style `tsconfig`; only `tsc -b` checks anything.

Generalize it into a standing rule:

> **Before trusting any check, prove it can fail.** For every gate, CI step, linter, and
> evaluator: introduce the bug it claims to catch and confirm a non-zero exit.

For a PoA network this is existential — **the entire economy rests on evaluators actually
evaluating.** Add `meshpool evaluator lint`, which:
1. runs the evaluator on a *known-bad* reference candidate and requires rejection;
2. runs it on a *known-good* one and requires acceptance;
3. runs it twice and requires identical verdicts;
4. runs it with the network disabled and requires it still works;
5. reports the flakiness score and the tamper-gate coverage.

**Make passing `evaluator lint` a hard requirement for Quest submission.**

## 12.3 New gate inventory (~40 gates)

| Area | Gates |
|---|---|
| **Weights & backends (P0)** | `real_weights_smoke`, `backend_equivalence_llamacpp`, `tokenizer_roundtrip_unicode`, `rope_convention_matrix` |
| **Sandbox (P0/P1)** | `oci_sandbox_determinism`, `oci_sandbox_isolation`, `wasm_fuel_limits`, `sandbox_panic_button`, `sandbox_no_egress` |
| **Fraud (P0)** | `fraud_e2e_transport` (4 cases), `forged_verdict_reject`, `replayed_verdict_reject` |
| **Schema & hashing (P1)** | `quest_schema_roundtrip`, `canonical_hash_cross_language` |
| **Intake (P1)** | `evaluator_intake_flaky_reject`, `evaluator_lint_selftest`, `coupling_law_enforced` 🔴 |
| **Context (P1)** | `context_pack_ablation`, `context_pack_editable_allowlist` |
| **Cascade (P1)** | `cascade_early_exit_cost`, `cascade_tamper_detection` (12 archetypes), `cascade_artifacts_nonempty` |
| **Oracle (P1)** | `heldout_defeats_overfit`, `heldout_never_leaves_anchor` |
| **Receipts (P1)** | `receipt_replay_crossmachine`, `receipt_tamper_reject` |
| **Settlement (P1)** | `poa_ledger_conservation`, `poa_effort_floor_antifarm`, `poa_duplicate_share`, `poa_escrow_refund_on_crash` |
| **E2E (P1)** | `quest_fix_e2e`, `bench_harness_determinism`, `bench_baseline_present` 🔴 |
| **Archive (P2)** | `archive_crdt_properties`, `archive_qd_improves` |
| **Archipelago (P2)** | `archipelago_coordinator_outage`, `archipelago_migration_bandwidth`, `archipelago_beats_single_island` |
| **Prompts (P2)** | `prompt_ablation_sidechannel`, `prompt_context_minimization_privacy` |
| **Bandit (P2)** | `bandit_regret_bound`, `bandit_diversity_not_greedy` |
| **Scheduling (P2)** | `weight_affinity_bytes`, `residency_plan_coverage`, `sample_locality_scheduling` |
| **Adapters (P2)** | `adapter_swarm_diversity`, `adapter_swarm_bytes` |
| **Prefix (P2)** | `prefix_sharing_speedup`, `prefix_sharing_bitexact` 🔴 |
| **Dedup (P2)** | `dedup_clusters`, `dedup_settlement` |
| **Budget (P2)** | `stopping_rule_savings`, `budget_never_exceeded`, `difficulty_adaptive_savings`, `no_evolution_without_gradient` |
| **Market (P3)** | `quest_board_api`, `credits_earn_spend_loop`, `auction_value_not_price`, `reputation_decay`, `consent_classes_enforced`, `dispute_detvm_resolution` |
| **Console (P3)** | `console_questboard_e2e`, `console_archipelago_e2e`, `console_receipt_verify_e2e` |

## 12.4 Adversarial red-team suite (new category)

Beyond correctness gates, add `sim/redteam/` — a suite where the *tests are the attacker*:

| Attack | Expected outcome |
|---|---|
| 12 reward-hacking archetypes (§P1-4) | All rejected by tamper + held-out |
| Sandbox escape attempts (net, FS, PID, fork bomb, `/proc` snooping, symlink escape, oversized write) | All contained |
| Sybil farm of 1,000 zero-work nodes | Earn ~0 |
| Colluding cartel submitting only weak candidates | Broken by a single honest node |
| Malicious Quest author rejecting valid candidates | Loses stake; verdict replay proves them wrong |
| Non-deterministic evaluator sneaking past intake | Caught by cross-node divergence monitor within N verdicts |
| Weight-poisoning (a node serving a subtly different model) | Caught by acceptance-rate collapse + optional TOPLOC-style commitment |

`[DESIGN]` Run the red-team suite on every release. Publish the results. **A verification
network that publishes its own red-team results is credible in a way that no competitor in
Part 3 currently is.**

## 12.5 The benchmark that must never be gamed

`bench/` numbers will be the company's public claim. Protect them:
- **Held-out benchmark split** never committed to the repo; stored encrypted; rotated.
- **Contamination checks** — verify pinned models weren't trained on the benchmark
  (n-gram overlap + canary strings).
- **Publish per-arm coverage curves**, not just a single headline number.
- **Publish failures.** The repo's README already demolishes two of its own claims. Keep that
  voice — it is the single most differentiating thing about this project's communications.

---

# PART 13 — GO-TO-MARKET

## 13.1 The wedge: one command that is obviously useful on day one

```bash
pip install meshpool
meshpool quest fix --repo . --test "pytest tests/test_thing.py::test_broken"
```

**Why this wedge and not another:**
- It is **useful with a single machine.** No network effect required to get value. This is the
  thing every failed decentralized-compute project lacked: *it works before the network
  exists.* Local mode = "run the search on my own machine." Adding the mesh just makes it
  faster and cheaper.
- The output is **a patch file** — a concrete artifact a developer can review, apply, and
  discard. Zero adoption risk.
- The receipt makes the claim **checkable**, which is a genuinely novel developer experience.
- It is directly comparable to tools people already pay for, on a metric that is
  objective (does the test pass?) rather than vibes.

**The single-machine → mesh upgrade path is the whole growth strategy:**

```
solo:      120 samples on your laptop     ~25 min
+ team:    120 samples on 8 idle laptops   ~4 min
+ mesh:    600 samples across the mesh     ~3 min, higher acceptance
```

## 13.2 Supply acquisition: mission, not mercenaries

`[CITED]` Folding@home reached **2.4 exaFLOPS** — briefly faster than the top 500
supercomputers combined — with volunteers who were paid **nothing**. SETI@home ran for 21
years on the same fuel. Meanwhile DePIN networks pay real money and get mercenary churn.

**Then Folding@home collapsed when COVID ended.** The lesson is not "missions work" — it is
**"missions work while they are urgent."**

`[DESIGN]` **So the mission must be permanent and self-renewing.** MeshPool's is:

> **"Frontier AI capability is being enclosed by five companies with capital nobody else has.
> Verified search is the one capability that does not require their capital — because search
> is embarrassingly parallel and truth is cheap to check. Point your idle machine at open
> problems, and the results belong to everyone."**

Concretely, seed the network with permanently-renewing public Quest streams:
1. **Open-source bug bounties** — CI failures from popular repos, auto-ingested.
2. **Open mathematical conjectures** — the AlphaEvolve list of 50, MiniF2F, Erdős problems.
3. **Science** — protein/materials/molecule search with open scoring functions.
4. **Benchmarks** — SWE-bench, ARC-AGI, HumanEval+ run continuously as a public leaderboard.
5. **Security** — OSS-Fuzz-style fuzzing of consenting projects.

Each solved Quest is **published with its receipt**, which is simultaneously: PR content,
credibility, training data, and a bandit prior. **The mission and the moat are the same
activity.**

**Grass's lesson (Part 3):** millions of nodes because the friction was *zero* — a browser
extension. MeshPool's node install must be similarly frictionless: one binary, one command,
a tray icon, idle-only by default, visible earnings, and a panic button.

## 13.3 Demand acquisition: three doors

| Door | Who | Motion | Why they say yes |
|---|---|---|---|
| **1. The CLI** | Individual developers | Bottom-up, OSS, free tier funded by contributed compute | It fixes their bug. Nothing else required. |
| **2. MeshPool Private** | Engineering orgs (200+ devs) | Top-down sales; deploy inside their fleet | *"Your laptops are idle 16 h/day. Wake up to verified patches for your flaky-test backlog and dependency upgrades. Your code never leaves."* 🔴 **This is the revenue.** |
| **3. Bounty market** | OSS maintainers, researchers, companies | Marketplace | Trustless bounty settlement with **no human judging** — the thing that has always killed bounty platforms. |

### The MeshPool Private pitch, in numbers `[ESTIMATE] — validate before using externally`

```
400 developer laptops × 16 idle hours/night × 250 working days
   = 1.6 million idle machine-hours per year
   ≈ the equivalent of a ~180-machine cluster running 24/7, at zero marginal hardware cost
```

Pointed at: flaky tests, dependency upgrades, Sonar/CodeQL findings, performance regressions,
missing test coverage, migration patches, and doc-drift — **all of which have objective
verifiers already in the customer's CI.**

> The customer already owns the compute, already owns the verifier, and already has the
> backlog. **MeshPool Private just connects three things they already have.** That is an
> unusually easy sale, and it requires no token, no network effect, and no trust in strangers.

## 13.4 Positioning and messaging

| Say | Don't say |
|---|---|
| "A search engine for solutions." | "Decentralized compute marketplace." |
| "You pay only for answers that pass your tests." | "Cheaper GPU hours." |
| "Verify every result yourself in 30 seconds." | "Trustless, zero-knowledge, blockchain-secured." |
| "Bitcoin burns electricity to prove nothing. We burn it to prove theorems." | "Web3 AI infrastructure." |
| "Diversity beats scale for search." | "We're faster than a datacenter." |
| "Here's what we measured, and here's what we got wrong." | Any unqualified benchmark number. |

## 13.5 🔴 The second-order business: verified solution data

> This may end up larger than the compute business. It should be architected for from day one,
> even though it cannot be sold on day one.

Every accepted Quest produces a tuple that is **currently the most valuable commodity in AI**:

```
(problem, context, N rejected candidates with failure reasons,
 1+ accepted candidate, machine-checked verdict, replay receipt)
```

**Why this specific data is exceptional:**

| Property | Why it matters |
|---|---|
| **Verified, not annotated.** A machine checked it. No human labelling cost, no labeller disagreement, no subjective noise. | RLVR (reinforcement learning from verifiable rewards) is the recipe behind R1, o-series, and every 2025–26 reasoning model. Verified data is its fuel. |
| **Includes the negatives — with reasons.** The ~99% of candidates that failed, each with the exact stage it died at and its error output. | **Contrastive/preference data is worth more than positives alone**, and almost nobody collects failures at scale with structured failure reasons. This is the rarest part. |
| **Real-world distribution.** Actual bugs in actual repos, not synthetic benchmark tasks. | Benchmarks are contaminated and saturating. Fresh, real, verified tasks are scarce. |
| **Provenance-carrying.** Every row has a replay receipt. | Auditable, contamination-checkable, and legally cleaner than scraped data. |
| **Continuously renewed.** New bugs appear forever. | Not a one-time corpus — a *stream*. Streams are businesses; corpora are assets. |

### The flywheel, stated precisely

```
network solves Quests
  → verified (problem, candidates, verdicts) stream
    → LoRA adapters trained on accepted solutions (§8.5, P4-7)
      → higher acceptance rate at the same cost
        → cheaper per accepted solution
          → more demand + more nodes
            → more Quests solved  ⟳
```

**And a second loop feeding the first:** the same data trains the **portfolio bandit's priors**
(which arm wins on which problem class) and the **difficulty estimator** (§8.11). Both improve
monotonically with volume and are **impossible to copy without the volume**.

> 🔴 **This is the answer to "what stops someone forking the repo?"** They get the code. They
> do not get the archive, the adapters, the bandit priors, or the stream. `[DESIGN]`

**Governance requirement (do this correctly from the start, or it becomes unusable later):**
- Quest authors set a data licence at submission: `public-domain` / `research-only` /
  `private` (default **private** for commercial Quests; default **public** for the OSS/science
  classes, matching their upstream licences).
- Only `public` data enters the open archive and public adapter training.
- Track B (Private) data **never** leaves the customer — but the customer may train **their
  own** private adapters on it, which is itself a compelling enterprise feature:
  *"your fleet gets better at your codebase, and the model never leaves your building."*

## 13.6 The open specification play

`[DESIGN]` Publish **Quest / Evaluator / Verdict / Replay Receipt** as an open, versioned
specification with a reference implementation and a conformance test suite.

**Why give it away:**
1. **Receipts are only valuable if others can verify them.** A proprietary receipt format is a
   contradiction in terms — verifiability requires openness.
2. **It makes MeshPool the neutral layer.** If SWE-agent, OpenHands, Aider, OpenEvolve, and
   Prime Intellect environments can all emit and consume Quests, MeshPool becomes the substrate
   rather than one more competing tool.
3. **Standards ownership is the most durable moat available to a small player.** Whoever
   defines the format defines the ecosystem's shape.
4. It is the natural home for the interop work in **P4-8** (`verifiers` / Environments Hub).

**Adjacent rails worth supporting behind interfaces (not depending on):**
`[VERIFIED-EXT]` **x402** (HTTP 402 Payment Required; Coinbase/Cloudflare facilitators, active
ecosystem) is a clean, token-agnostic micropayment rail for Quest bounties — a good *optional*
adapter that avoids issuing a token (§9.7). **ERC-8004 (Trustless Agents)** is worth watching
for agent identity. Support both as plugins; depend on neither.

## 13.7 Sequenced milestones

| Milestone | Proves | Timing |
|---|---|---|
| **M0** — Phase 0 green; a real 1B model generates tokens on a GPU | The foundation is real | Week 2 |
| **M1** — `meshpool quest fix` solves 8/20 real bugs locally with receipts | 🔴 **The thesis** | Week 8 |
| **M2** — Publish the coverage curves + the cost-per-accepted-solution number | The economics | Week 10 |
| **M3** — Archipelago beats i.i.d. sampling at equal budget on the benchmark | 🔴 **The invention** | Week 18 |
| **M4** — 100-node public network solving public Quests continuously | The network | Week 24 |
| **M5** — First MeshPool Private design partner | 🔴 **The business** | Week 28 |
| **M6** — A genuinely novel result (an open conjecture, a kernel speedup, a CVE) found by the mesh, with a public receipt | 🔴 **The legend** | Week 40 |

> **M6 is the one that matters for the "first in the world" ambition.** AlphaEvolve improved
> on SOTA for 20% of 50 open problems using Google's TPUs. **The first time a volunteer mesh
> does that, it is a permanent, unfakeable, front-page fact** — and it is exactly the kind of
> thing a heterogeneous portfolio search is well-suited to produce.

---

# PART 14 — RISKS, KILL CRITERIA, OPEN QUESTIONS

## 14.1 Ranked risks

| # | Risk | Severity | Probability | Mitigation | Kill signal |
|---|---|---|---|---|---|
| R1 | **Small open models aren't good enough**; acceptance rates too low for the economics to work | 🔴 Critical | Medium | Portfolio diversity; API-passthrough arms; adapters; the coverage curve is log-linear so *more samples always helps* | If 500 samples from a 40-arm portfolio can't beat 5 samples from a frontier model on cost-per-accepted-solution, **the thesis is wrong** |
| R2 | **Reward hacking makes accepted solutions untrustworthy** | 🔴 Critical | Medium-high | §10.2's 7-layer stack; held-out oracles on by default | If >5% of accepted patches are rejected by human review after the full stack, the product is not shippable |
| R3 | **Verification cost dominates** for realistic workloads | 🟠 High | Medium | Cascade (§8.7); T0 WASM default; evaluation locality | If average verify cost > 30% of generate cost after cascade tuning |
| R4 | **Frontier labs ship this** (OpenAI/Anthropic/Google offering massive-sampling agents) | 🟠 High | **High** | They will — but at *datacenter prices*, without receipts, and without your private fleet. Compete on cost structure, verifiability, and privacy, **not capability** | If they ship it at a price below mesh marginal cost, pivot hard to Track B (Private) only |
| R5 | **Supply never shows up** | 🟠 High | Medium | Mission + zero friction + earn-credits-spend-credits loop + Private track needs no external supply | If <500 nodes after 6 months of public launch |
| R6 | **Privacy blocks the enterprise sale** | 🟡 Medium | Medium | Track B is *architecturally* private | — |
| R7 | **Evaluator flakiness poisons trust** | 🟠 High | **High** (this always happens) | Intake determinism check; divergence monitor; auto-suspend; refunds | If >2% of Quests get suspended for flakiness after tooling |
| R8 | **Sandbox escape harms a volunteer** | 🔴 Critical | Low | gVisor/microVM default; WASM-first; no network; consent classes; bug bounty | Any single real incident = immediate public network pause |
| R9 | **Legal/dual-use blowback** (fuzzing, exploit generation) | 🟠 High | Medium | §10.6 governance; permission proofs; publisher identity | — |
| R10 | **The team rebuilds kernels instead of shipping the loop** | 🟡 Medium | **High** (this is the repo's demonstrated failure mode — a beautiful NumPy transformer that has never seen a real model) | 🔴 **Delegate kernels to llama.cpp/vLLM/MLX. Own the search loop.** | If Phase 1 slips because of backend work |
| R11 | Weight distribution swamps residential uplinks | 🟡 Medium | Medium | Weight affinity + residency planning + Adapter Swarm | — |
| R12 | Coverage curves flatten faster than the papers suggest on *real* (non-benchmark) tasks | 🟠 High | Medium | Measure early (M2); portfolio diversity specifically attacks flattening | If `accepted@500 ≈ accepted@50` on real bugs |
| R13 | 🔴 **The optimizer–verifier coupling failure (§10.2.1)**: Archipelago, being a strong optimizer, exploits weak evaluators *harder* than i.i.d. sampling would — so the flagship algorithm makes results **worse** on the majority of real-world Quests, whose test suites are mediocre | 🔴 Critical | **High** — most real test suites *are* weak, and Snell et al. observed exactly this degradation in a controlled setting | `verifier_strength()` cap (P1-10) enforced **before** Archipelago ships; exploitation-index monitoring; auto-throttle | If, at equal budget, Archipelago's held-out pass rate is *below* i.i.d. sampling's on the benchmark, evolution must stay disabled by default |
| R14 | **The bash-loop baseline is good enough** (§5.8): `for i in {1..200}; do claude ...; done` gets close enough that the mesh's added complexity isn't worth it | 🔴 Critical | Medium | Cost structure (idle electricity vs API), portfolio diversity across vendors, receipts, privacy, evolution | 🔴 **If MeshPool cannot beat the bash loop on cost per accepted solution at M2, stop.** This is kill criterion 1, restated as the comparison that actually matters |
| R15 | Context Pack quality caps the whole system, silently (§7.6) | 🟠 High | **High** | Ablation gate from day one; packing strategy as a bandit arm | If acceptance is insensitive to packing strategy, the context pipeline is broken — investigate before blaming models |

## 14.2 Explicit kill criteria (write these down now, honour them later)

> **Kill criterion 1 (economic):** if, by M2, the measured cost-per-accepted-solution on the
> 20-bug benchmark is **not at least 5× cheaper** than the best single-shot frontier-API
> approach at equal acceptance rate — **the cost thesis is dead.** Pivot to Track B only
> (where the compute is free because the customer already owns it) or stop.
> *Calibration:* `[VERIFIED-EXT]` Large Language Monkeys already measured **3× cheaper at just
> 5 samples using rented APIs** (§4.1.3). MeshPool runs on idle-machine electricity, so 5× is
> a deliberately conservative bar. **Failing to clear a bar that the literature already
> clears at 5 samples means the implementation is wrong, not the thesis.**
>
> **Kill criterion 1b (the baseline that matters):** the comparison must be against
> **`for i in {1..200}; do <frontier-api> ...; done` with the same context pack and the same
> evaluator** (§5.8, gate `bench_baseline_present`). Beating a strawman is not evidence.
>
> **Kill criterion 2 (algorithmic):** if, by M3, Archipelago does **not** beat plain i.i.d.
> sampling at equal budget on the benchmark — **the invention is not an invention.** Ship the
> simple version, drop the search-engine narrative, and reposition as a distributed
> test-time-compute runner.
> *Addendum (§10.2.1):* if Archipelago beats i.i.d. on **public** pass rate but not on
> **held-out** pass rate, that is **not** a win — it is verifier exploitation, and evolution
> must remain off by default until verifier strength is raised.
>
> **Kill criterion 3 (trust):** if, after the full §10.2 stack, human review rejects **>5%**
> of mechanically-accepted patches — **PoA is not a settlement primitive**, only a filter.
> Reposition to "we rank candidates, you decide" and rebuild the economics around effort, not
> acceptance.
>
> **Kill criterion 4 (supply):** if the public network cannot hold **500 concurrent nodes**
> six months after launch, the open network is a hobby. Keep it as marketing; make Track B the
> company.

`[DESIGN]` Recording kill criteria before starting is the single highest-value practice in
this entire document. The 117 projects in Part 3 that died mostly died by *not* having them.

## 14.3 Open questions the next session must resolve

| # | Question | Notes / lean |
|---|---|---|
| Q1 | What is the *real* coverage curve on real bugs with 1–7B open models? | **Measure at M2. Everything depends on this.** |
| Q2 | What fraction of candidates die in the free cascade stages? | AlphaCode says ~99%. If true, verification is nearly free. Measure. |
| Q3 | Does quantization diversity actually decorrelate errors? | Novel and cheap if true. **Test explicitly** — it's a free diversity axis. |
| Q4 | Adopt Prime Intellect's `verifiers` environment format? | **Lean yes** — instant evaluator supply, and interop is cheap. |
| Q5 | Cartesi Machine vs RISC Zero (non-proving) for T2? | Evaluate both; T2 is dispute-only so latency doesn't matter much. |
| Q6 | Token or no token? | **Lean strongly no** through Phase 3 (§9.7). Revisit only if public bounties demand it. |
| Q7 | Bounty-dominant vs effort-dominant default pricing? | Start effort-blended and adapt to observed acceptance rate (§9.3). |
| Q8 | Does the effort floor get farmed despite the cascade gate? | Red-team it in Phase 1, not Phase 3. |
| Q9 | Do islands actually beat one big population, at *these* scales? | The literature says yes for EAs; unproven for LLM-driven search. **This is M3.** |
| Q10 | Is context minimization (privacy V1) compatible with high acceptance rates? | Measure the acceptance delta. If it's large, V1 may be unviable and only V0/V3 survive. |
| Q11 | Can `MeshShardBackend` and llama.cpp be made greedy-identical? | If not, understand *why* — it's a real finding either way. |
| Q12 | What is the right number of arms before the bandit stops learning? | Factored models are the answer; validate the crossover empirically. |
| Q13 | Should MeshPool run its own frontier-API arms (paid) inside a mostly-free mesh? | Probably yes for hard Quests — a hybrid portfolio is strictly better. Price it through. |
| Q14 | How to bootstrap the archive for a brand-new Quest with no elites? | Seed from the *global* archive of similar problems (the compounding asset paying off). |
| Q15 | 🔴 What is the *real* distribution of `verifier_strength` across actual OSS repos? | If most real test suites score <0.3, then **evolution is off by default for most Quests** and the product is mostly portfolio sampling. **Measure this on the 20-bug benchmark in Phase 1 — it determines whether Archipelago is the headline or a niche mode.** |
| Q16 | Can weak verifiers be *automatically strengthened* (auto-generated held-out + property tests)? | If yes, this is a major product feature *and* it unlocks evolution for everyone. If no, verifier strength is a hard ceiling on the market size. **High-value research question.** |
| Q17 | Does the artifact side-channel leak enough to enable reward hacking? | Showing a candidate the exact failing assertion also teaches it exactly what to special-case. Ablate this. Possible answer: show failures for public tests, never for held-out. |
| Q18 | Is GEPA's reflective-Pareto parent selection better than MAP-Elites for code Quests? | Both are cheap to implement. **Run both; let the benchmark decide.** Add as a `parent_strategy` arm (§8.2.5). |
| Q19 | Should Meta-Quests (§6.8) run continuously on spare capacity from day one, or only post-M3? | Post-M3 for the *loop*; but collect the `@evolvable` metric harness data from Phase 2 so there is a training set ready. |

## 14.4 What would make this fail slowly and invisibly (the real danger)

`[DESIGN]` The most likely failure is not a dramatic one. It is:

1. Phase 0 takes three months because GPU backends are fiddly.
2. The team rebuilds a NumPy kernel because it's more fun than plumbing (**this has already
   happened once in this repo**).
3. The Quest loop ships but nobody measures the coverage curve.
4. The demo works on cherry-picked bugs.
5. Nobody writes kill criteria, so nothing is ever killed.
6. Two years later there is a beautiful, well-tested system with no users.

**The antidote is M1 and M2.** Get to "8 out of 20 real bugs, with receipts, and here is the
cost per accepted solution" as fast as physically possible, and let that number decide
everything.

---

# APPENDIX A — THE NUMBERS THAT MATTER

| # | Number | Value | Tier | Source | Consequence |
|---|---|---|---|---|---|
| A1 | Datacenter GPU interconnect | 3.2 Tbps (NVLink/IB) | `[CITED]` | Vendor specs | The gap |
| A2 | Residential uplink (median) | 10–50 Mbps | `[CITED]` | Ookla-class data | ~100,000× gap ⇒ tightly-coupled work is impossible |
| A3 | MeshPool 1T MoE ceiling, 11 metro nodes, SPD, **free compute** | **15.2 tok/s** | `[MEASURED]` | `docs/benchmarks.md` | 🔴 Kills the flagship thesis |
| A4 | Same, worldwide | 3.0 tok/s | `[MEASURED]` | `docs/benchmarks.md` | Kills it harder |
| A5 | SPD speedup, WAN | ×2.2–2.5 | `[MEASURED]` | repo | Real and good |
| A6 | SPD at depth 1 | **×0.87–0.93 (slower)** | `[MEASURED]` | repo | Honest self-refutation; keep this culture |
| A7 | Continuous batching gain | 1.65–2.09× (288→51 hops) | `[MEASURED]` | repo | An earlier 2.5–3.6× claim was wrong (per-request TLS trust-store loads) |
| A8 | Proactor `accept()` bug kill rate | **4/47 gates ≈ 8.5%** | `[MEASURED]` | repo | Why `common/proactor_fix.py` exists |
| A9 | exo tensor-parallel speedup | 1.8× (2 dev), 3.2× (4 dev) | `[VERIFIED-EXT]` | github.com/exo-explore/exo | exo owns the LAN case |
| A10 | Petals Llama-2-70B | ~6 tok/s | `[VERIFIED-EXT]` | Petals repo | The WAN ceiling, confirmed independently |
| A11 | Repeated sampling: DeepSeek-Coder-V2 on SWE-bench Lite | **15.9% @1 → 56% @250** (single-attempt SOTA was 43%) | 🟢 `[VERIFIED-EXT]` | arxiv.org/html/2407.21787v2, fetched this session | 🔴 The thesis. A weaker model + 250 samples beat SOTA by 13 points |
| A11b | Gemma-2B on CodeContests | 0.02% @1 → **7.1% @10,000** = **300× coverage** | 🟢 `[VERIFIED-EXT]` | same | Coverage keeps climbing for 4 orders of magnitude |
| A11c | 🔴 **Cost:** 5 samples of cheap DeepSeek vs 1 sample of Claude/GPT (SWE-bench Lite, same scaffold) | **solves more, >3× cheaper** | 🟢 `[VERIFIED-EXT]` | same | **The business case, measured — and at *rented API* prices. Idle-machine cost makes 3× a floor, not a ceiling.** |
| A11d | 🔴 **THE VERIFIER GAP** — MATH, Llama-3-8B: coverage vs. realized accuracy | coverage 79.8%→**95.3%** @10k; majority-vote/reward-model **38.7%→39.8%** | 🟢 `[VERIFIED-EXT]` | same | **Without a verifier, scaling compute buys ~nothing, and the gap *widens* with scale. This is why MeshPool refuses non-verifiable work.** |
| A16 | Compute-optimal test-time allocation | **up to 4× less compute** for equal result (16 vs 64 generations) | 🟢 `[VERIFIED-EXT]` | arxiv.org/html/2408.03314v1, fetched this session | Allocation matters as much as budget ⇒ §8.11 |
| A16b | 🔴 **Optimizer–verifier coupling** | On *easy* questions, the **stronger** optimizer (beam/lookahead search) **degrades** performance as budget grows, by exploiting the verifier | 🟢 `[VERIFIED-EXT]` | same | **§10.2.1 design law. The strongest technical risk to Archipelago, and the reason for the P1-10 cap.** |
| A16c | Strategy inversion by difficulty | beam search wins on **hard**/low-budget; best-of-N wins on **easy**/high-budget; level-5 problems: nothing works | 🟢 `[VERIFIED-EXT]` | same | Forces difficulty into the bandit context (§8.11) |
| A16d | Compute-optimal **model size is dataset-dependent** | MATH ⇒ smaller model + more attempts; CodeContests ⇒ larger model | 🟢 `[VERIFIED-EXT]` | arxiv 2407.21787v2 | **Formal proof there is no globally best arm ⇒ the portfolio bandit is necessary, not decorative** |
| A12 | AlphaEvolve on 50 open math problems | rediscovered SOTA 75%, **improved 20%** | `[VERIFIED-EXT]` | Wikipedia/DeepMind | 🔴 The proof that LLM+evaluator+evolution discovers new things |
| A13 | AlphaEvolve datacenter recovery | 0.7% of Google's fleet | `[VERIFIED-EXT]` | same | Enormous absolute value |
| A14 | AlphaEvolve 4×4 complex matmul | **48 multiplications** (beat 49, a 56-year record) | `[VERIFIED-EXT]` | same | Algorithm discovery is real |
| A15 | AlphaCode samples per problem | up to **1,000,000**, ~99% filtered | `[CITED]` | DeepMind | Massive sampling + filtering works; cost was the only blocker |
| A16 | Test-time compute vs. parameters | can beat a **14× larger** model | `[CITED]` | arXiv 2408.03314 | Allocation matters as much as budget |
| A31 | OpenEvolve stars | **7,292** | 🟢 `[VERIFIED-EXT]` | GitHub API, this session | The space is hot; adopt as input, don't compete |
| A32 | GEPA (reflective Genetic-Pareto optimizer) stars | **6,315**, created Aug 2025 | 🟢 `[VERIFIED-EXT]` | GitHub API, this session | A competitor algorithm found late — steal its Pareto parent selection (§8.2.5, Q18) |
| A33 | Repos combining volunteer/decentralized compute + evolutionary search + verifier settlement | **0** | 🟢 `[VERIFIED-EXT]` | GitHub search, this session | 🔴 **The "first in the world" claim survives a direct search.** |
| A17 | Folding@home peak | **2.4 exaFLOPS**, volunteers, unpaid | `[CITED]` | F@h | Mission > money for supply |
| A18 | zkML overhead | 10³–10⁶× | `[CITED]` | EZKL/zkLLM literature | Not viable; PoA avoids it entirely |
| A19 | TEE overhead | ~5–15% (NVIDIA CC) | `[CITED]` | NVIDIA | Viable but repeatedly broken (9+ published attacks) |
| A20 | Pipeline availability, 11 nodes, p=0.05/hr | **0.569** | `[DERIVED]` | §5.4 | 43% failure/hr — κ=1 is fatal |
| A21 | Lost-sample probability, 60 s sample, p=0.05/hr | **0.0855%** | `[DERIVED]` | §5.4 | 🔴 κ=1/N is a **~500×** robustness improvement |
| A22 | Cascade cost reduction | **~9×** | `[DERIVED]` | §8.7 | Verification becomes nearly free |
| A23 | Prefix sharing gain, 32 samples, 8k prefix | ~3–6× | `[ESTIMATE]` | §8.8 | Must measure |
| A24 | Adapter vs. model bytes | **~38× cheaper** for 51 generators | `[DERIVED]` | §8.5 | Diversity at negligible bandwidth |
| A25 | Migration bandwidth per island | ~32 KB / 30–600 s | `[DERIVED]` | §8.2.4 | ~10,000× lighter than pipeline traffic |
| A26 | Cost per accepted code fix (mesh) | ~$0.02 | `[ESTIMATE]` | §9.8 | 🔴 **Validate at M2 — this is the business** |
| A27 | Enterprise idle capacity, 400 laptops | ~1.6M machine-hours/yr | `[ESTIMATE]` | §13.3 | The Private pitch |
| A28 | MeshPool codebase | 311 Python files / 84,730 LOC + Go agent + React console | `[MEASURED]` | this session | ~85% reusable |
| A29 | Existing acceptance gates | 80 | `[MEASURED]` | `sim/scenarios/` | The crown jewels |
| A30 | New code required for the pivot | ~15–20K LOC | `[ESTIMATE]` | §11.2 | The pivot is cheap |

# APPENDIX B — GLOSSARY

| Term | Definition |
|---|---|
| **Quest** | A problem + machine-checkable evaluator + budget + bounty. The unit of demand. |
| **Candidate** | One proposed artifact (patch, program, proof, config, design). |
| **Evaluator** | Deterministic, sandboxed, content-addressed program mapping Candidate → Verdict. |
| **Verdict** | `{accepted, score, behaviour, signals, artifacts}`. Signed. |
| **Proof-of-Acceptance (PoA)** | Settlement rule: pay iff the verdict is accepted and replayable. |
| **Replay Receipt** | Signed record letting anyone re-derive the verdict. Tiers R0/R1/R2. |
| **Archipelago** | The distributed island-model quality-diversity search controller. |
| **Island** | One evolutionary population, pinned to a Constellation. |
| **Archive** | MAP-Elites grid: one elite per behaviour cell. A CRDT. |
| **Arm** | `(model, quant, adapter, temperature, scaffold, thinking-budget, mutation-mode, …)`. |
| **Portfolio Bandit** | Contextual bandit choosing arms to maximize coverage per unit cost. |
| **Adapter Swarm** | Distributing LoRA adapters over shared base models for cheap diversity. |
| **Weight Affinity** | Scheduling to where weights are already resident. |
| **Verification Cascade** | Cheap→expensive staged evaluation with early exit. |
| **Held-Out Oracle** | Secret test set the generators never see. Anti-reward-hacking. |
| **κ (coupling coefficient)** | Fraction of a workload lost when one node fails. κ=1 pipeline, κ=1/N sampling. |
| **Coverage** | `P(≥1 accepted candidate | budget)`. The true objective function. |
| **Constellation** | *(existing MeshPool term)* A low-RTT cluster of nodes. Becomes an Island. |
| **Anchor / Steady / Drifter** | *(existing)* Node stability classes. Anchors host archives + oracles. |
| **Weight Swarm** | *(existing)* P2P weight distribution. Reused for adapters. |
| **SPD** | *(existing)* Speculative pipeline decoding. Measured ×2.2–2.5 on WAN. |

# APPENDIX C — DERIVATIONS

### C.1 Coverage under a portfolio
```
Single arm, k samples:            cov(k)   = 1 − (1 − p)^k
Portfolio, independent failures:  cov(K)   = 1 − Π_c (1 − p_c)^{k_c}
With correlation ρ between arms:  p_eff(c) = p_c · (1 − ρ · max_sim(c, planned))
Marginal gain of one more sample of arm c:
  Δ(c) = [Π_{c'} (1 − p_{c'})^{k_{c'}}] · p_eff(c)
  ⇒ greedy on Δ(c)/cost(c) is the cost-benefit rule; F is monotone submodular ⇒ (1 − 1/e).
```

### C.2 Availability: pipeline vs. sampling
```
Pipeline, N nodes, per-node hourly failure prob p:
  A_pipe(1 hr) = (1 − p)^N
  N=11, p=0.05 → 0.5688      N=11, p=0.10 → 0.3138     N=11, p=0.20 → 0.0859

Sampling, sample duration t seconds, per-node hourly failure prob p:
  P(lose one sample) = 1 − (1 − p)^{t/3600}
  t=60,  p=0.05 → 8.55e-4     t=300, p=0.05 → 4.27e-3
Expected lost samples out of 200 at t=60, p=0.05 = 0.171
Ratio of "work at risk": pipeline risks 100% of in-flight work; sampling risks 1/N.
```

### C.3 Cascade expected cost
```
Stages i with cost c_i and pass-prob p_i (conditional on reaching stage i):
  E[cost] = Σ_i c_i · Π_{j<i} p_j
Optimal ordering for independent stages: ascending c_i / (1 − p_i).

Worked example (§8.7): c = [0.001, 0.1, 2.5], survival = [0.4, 0.33, 1.0]
  E[cost] = 0.001 + 0.4·0.1 + 0.4·0.33·2.5 = 0.001 + 0.04 + 0.33 = 0.371
  Naive (always full suite) = 2.601   ⇒ 7.0× reduction (≈9× with dedup/identity stage).
```

### C.4 Adapter economics
```
Distinct models:  M × S_model            M=51, S=5 GB   → 255 GB
Base + adapters:  S_model + M × S_adapter               → 5 + 51×0.05 = 7.55 GB
Ratio = 33.8×  (≈38× at S_adapter = 40 MB)
```

### C.5 Prefix sharing
```
Naive:  N·(a·P + b·D)
Shared: a·P + N·b·D
Speedup = N(aP + bD) / (aP + NbD)
With P=8000, D=500, and prefill:decode work ratio per token ≈ 1:1 in FLOPs but decode being
bandwidth-bound, take effective aP ≈ 4·bD:
  N=32 → 32(4+1) / (4 + 32) = 160/36 = 4.44×      [ESTIMATE — measure, don't trust]
```

### C.6 Migration bandwidth
```
K elites × size × islands / period
  8 × 4 KB × 1 / 60 s = 0.53 KB/s per island
Compare pipeline-parallel: 14 KiB/token/hop × 10 hops × 6 tok/s = 860 KB/s
  ⇒ ~1,600× lighter, and latency-insensitive.
```

# APPENDIX D — THE 117-NAME CHECKLIST (verification index)

> Full analysis is in **Part 3**, grouped exactly as below. This appendix exists so the next
> session can mechanically confirm nothing in `projects.md` was skipped. Every name has a row
> in Part 3 covering: what it does · architecture/key mechanism · verification model · the
> specific lesson for MeshPool.

| Part 3 § | Group | Count | Names |
|---|---|---|---|
| §3.1 | **A — DePIN GPU marketplaces** | 20 | Vast.ai · Akash Network · Akash Homenode · io.net · Aethir · Nosana · Spheron · SaladCloud · Golem · Flux · NuNet · Render Network · Dispersed/Render Dispersed · Bittensor · Theta EdgeCloud · Clore.ai · OctaSpace · Qubic · Dynex · DeepBrain Chain |
| §3.2 | **B — Distributed training/inference systems** | 13 | Petals · exo · Hivemind · Gensyn · Prime Intellect · Nous Research · DisTrO · OpenDiLoCo implementations · Together Computer/Together AI origins · Experimental academic systems · Various academic projects · Hyperbolic · Exabits |
| §3.3 | **C — Verifiable / confidential compute** | 16 | iExec · iExec Workerpool · Phala Network · Oasis Network ROFL · Marlin · Aleph.im · Ankr · Dfinity Internet Computer · Cartesi · Lilypad Network · Bacalhau · Bacalhau by Expanso · HyperCycle · Crynux · Inference Labs · EigenAI/EigenCloud |
| §3.4 | **D — Crypto-AI protocols & agent networks** | 17 | Ritual · Atoma Network · OpenGradient · TensorOpera/Fox-1 · Morpheus (listed twice in the file) · Ora Protocol · Fetch.ai/ASI Alliance · SingularityNET · Ocean Protocol · Allora Network · Cortex · Oraichain · Giza · Sentient · Sahara AI · Vana · Grass |
| §3.5 | **E — Edge / mobile / gaming DePIN** | 11 | Acurast · NodeOps Network · CUDOS/CUDOS Intercloud · NeurochainAI · Inferix · GAIMIN · DATS Project · Titan Network · Raiinmaker · Commune AI · Heurist |
| §3.6 | **F — Neoclouds / GPU IaaS** | 26 | RunPod · TensorDock · FluidStack · Sesterce · Hyperstack · JarvisLabs.ai · Qubrid AI · Shadeform · CloudRift · Massed Compute · DataCrunch · Genesis Cloud · Nebius AI · Nscale · Voltage Park · Crusoe Cloud · Vultr Cloud GPU · Scaleway GPU · OVHcloud GPU · DigitalOcean Gradient AI · Lambda · CoreWeave · Paperspace · Cirrascale · Ori · Northflank |
| §3.7 | **G — Serverless GPU / inference platforms** | 12 | Modal · Replicate · Baseten · Together AI · Fireworks AI · Anyscale · Lepton AI · Fal.ai · Cerebrium · Beam · Banana.dev · Chutes.ai · Fluence |
| §3.8 | **The graveyard** | — | SONM · Banana.dev · Golem's first decade · SETI@home · Gridcoin · Folding@home's collapse · Together's pivot · Petals' empty swarm |

**Mechanical check the next session should re-run:**

```powershell
$plan = Get-Content MESHPOOL_MASTER_PLAN.md -Raw
Get-Content projects.md | Where-Object { $_.Trim() } | ForEach-Object {
  $k = ($_ -split '\s*[/(]')[0].Trim()
  if ($plan -notmatch [regex]::Escape($k)) { "MISSING: $_" }
}
```

`[MEASURED]` This check was run at the end of this session against all 117 lines of
`projects.md`: **0 names genuinely missing** (the only apparent misses were composite labels
such as "Akash networks" whose base name "Akash" is covered in full).

## D.1 Additional names analysed in Part 4 (these are NOT from `projects.md`)

These are the research systems and infrastructure the *new* architecture depends on. They are
covered in **Part 4**, not Part 3.

- **Test-time compute:** Large Language Monkeys (arXiv 2407.21787) · Snell et al. (2408.03314) ·
  s1/budget forcing (2501.19393) · Let's Verify Step by Step (2305.20050) · Self-Consistency
  (2203.11171) · o1/o3 & ARC-AGI cost modes · DeepSeek-R1/GRPO (2501.12948)
- **LLM + evolutionary search:** FunSearch · AlphaEvolve · **OpenEvolve** · Eureka · ADAS ·
  Promptbreeder · EvoPrompt · AlphaCode / AlphaCode 2 · AlphaProof · AlphaGeometry ·
  DeepSeek-Prover-V2 · Kimina-Prover · AlphaTensor · AlphaDev · MAP-Elites (1504.04909) ·
  CVT-MAP-Elites · POET · Novelty Search · SATzilla / algorithm portfolios
- **Serving & kernels:** vLLM (PagedAttention) · SGLang (RadixAttention) · Mooncake · DistServe ·
  Splitwise · Sarathi-Serve · llama.cpp (+RPC backend) · MLX / MLX-distributed · PowerInfer ·
  FlexGen · AirLLM · DeepSpeed ZeRO-Inference · Ray
- **Determinism & verification:** Thinking Machines Lab, "Defeating Nondeterminism in LLM
  Inference" · `batch_invariant_ops` · Gensyn RepOps/Verde · TOPLOC · Proof of Sampling (PoSP) ·
  opML · zkLLM · EZKL · RISC Zero · SP1 · Cartesi Machine · Truebit · Arbitrum/Optimism fraud proofs
- **TEE attack literature (why not to bet on TEEs):** Foreshadow · SGAxe · Plundervolt ·
  ÆPIC Leak · CipherLeaks · WeSee · Heckler · BadRAM · TEE.fail
- **Sandboxing:** wasmtime · wazero · WasmEdge · gVisor · Firecracker · Cloud Hypervisor ·
  Docker/OCI · seccomp
- **Benchmarks & agents:** SWE-bench (Lite/Verified) · SWE-agent · OpenHands · Aider · Devin ·
  HumanEval+ / MBPP+ · ARC-AGI · MiniF2F · OSS-Fuzz · Prime Intellect `verifiers` &
  Environments Hub
- **Volunteer computing:** BOINC · SETI@home · Folding@home · Rosetta@home · World Community Grid
- **Substrate/adjacent:** IPFS · Filecoin · Arweave · Storj · Sia · Swarm · libp2p · Iroh ·
  Syncthing · Kubernetes · Nomad · Slurm · HTCondor · Temporal · NATS · SWIM · Vivaldi · CRDTs

# APPENDIX E — REFERENCE READING LIST FOR THE NEXT SESSION

**Must read before writing code:**
1. arXiv **2407.21787** — Large Language Monkeys (the coverage curve).
2. arXiv **2408.03314** — Compute-optimal test-time scaling.
3. **FunSearch** (Nature, 2023/24) + the DeepMind blog.
4. **AlphaEvolve** white paper + `en.wikipedia.org/wiki/AlphaEvolve`.
5. `github.com/algorithmicsuperintelligence/openevolve` — **read the whole repo**; it is the
   closest working implementation of Layer 3.
6. arXiv **1504.04909** — MAP-Elites.
7. `thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/`.
8. `github.com/PrimeIntellect-ai/verifiers` — the environment format to interop with.
9. **SGLang** RadixAttention paper — prefix sharing.
10. **AlphaCode** paper — sampling, filtering, clustering at 10⁶ scale.

**Must read in this repo before writing code:**
1. `README.md` lines **442–893** — "Honest constraints." Read every word.
2. `docs/benchmarks.md` — the numbers that killed the old thesis.
3. `meshpool/controlplane/verification.py` — especially `VerificationPolicy.classify()`.
4. `meshpool/controlplane/scheduler.py` — `Constellation`, `build_constellation_index`.
5. `meshpool/functions/wasm.py` + `agent-go/internal/wasmfn` — the sandbox seed.
6. `sim/mutation_guard.py` + 3 representative `sim/scenarios/*.py` — learn the gate style
   **before** writing new gates.
7. `meshpool/common/config.py` — the `scale=` presets that make Track A/Track B one product.

# APPENDIX F — FIRST-WEEK CHECKLIST FOR THE NEXT SESSION

```
[ ] 1. Read Appendix E's in-repo list. Do not skip README 442–893.
      Then read §4.1.2 (the verifier gap) and §10.2.1 (the coupling law). These two sections
      contain the findings that most change how the system must be built.
[ ] 2. Run the existing gate sweep. Record the baseline. Confirm mutation_guard is clean.
[ ] 3. P0-1: get real weights. Verify SHA-256. Add real_weights_smoke gate. DO NOT PROCEED
       UNTIL GREEN — every number in this plan is unverified until this passes.
[ ] 4. P0-2: llama.cpp backend. Prove greedy-equality with MeshShardBackend (or explain why not).
[ ] 5. P0-3: real DockerRuntime + determinism/isolation gates.
[ ] 6. P0-4: transport-E2E fraud gate.
[ ] 7. Create meshpool/quest/, search/, evaluate/, generate/, receipts/ package skeletons.
[ ] 8. P1-1/P1-2: schemas + canonical hashing + cross-language gate.
       Include the data-licence field now (P4-15) even though nothing reads it yet.
[ ] 9. P1-4/P1-5: cascade + tamper + held-out oracle (with the hand-written overfit control).
[ ] 10. P1-10: verifier-strength scoring + the coupling-law cap. 🔴 MUST land before any
       evolutionary search exists, or the first thing the optimizer learns is how to cheat.
[ ] 11. P1-11: Context Pack builder + ablation gate. Context quality caps everything (R15).
[ ] 12. P1-6/P1-7: receipts + PoA settlement + ledger conservation gate.
[ ] 13. P1-8: `meshpool quest fix` vertical slice, single island, i.i.d. sampling, NO evolution.
[ ] 14. P1-9: curate the 20-bug benchmark. Measure the coverage curve AND the bash-loop
       baseline (§5.8). PUBLISH BOTH, including the parts that look bad.
[ ] 15. Answer Q15: what is the real distribution of verifier_strength across those 20 repos?
       This single measurement determines whether Archipelago is the headline or a niche mode.
[ ] 16. Write the kill criteria into docs/ and diarize the M2 review.
[ ] 17. Only then start Phase 2 (Archipelago).
```

## The four things this document asks you not to forget

1. **The verifier is the scarce resource, not the GPU** (§4.1.2). Everything follows from this.
2. **Stronger search demands a stronger verifier** (§10.2.1). Ship the cap before the search.
3. **The competitor is a bash loop around a frontier API** (§5.8). Beat it numerically or stop.
4. **Publish the numbers that make you look bad.** It is this repo's most valuable habit and
   its least copyable asset.

---

## CLOSING NOTE TO THE NEXT SESSION

You are inheriting **an excellent distributed system pointed at the wrong workload.**

Do not rewrite it. Do not admire it either. **Retarget it.**

The three things that matter, in order:

1. **Get a real model generating real tokens on a real GPU** (Phase 0). Nothing is true
   until this works.
2. **Get `meshpool quest fix` solving real bugs with real receipts** (M1). This is the thesis.
3. **Measure the coverage curve and the cost per accepted solution, and publish both —
   including the parts that look bad** (M2). This is the business, and this repo's culture of
   publishing its own refutations is its most valuable and least copyable asset.

Everything else in these fourteen parts is elaboration on those three steps.

> *Bitcoin burns electricity to prove nothing.
> MeshPool burns electricity to prove theorems, fix bugs, and discover algorithms.*

**— END OF DOCUMENT —**
