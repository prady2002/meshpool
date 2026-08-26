# Distributed inference (the flagship)

Run a language model whose layers **don't fit on any single machine** by sharding it
pipeline-parallel across several. This is the hardest and most valuable thing MeshPool
does, so it gets the most detail.

Code: `meshpool/inference/` (`model.py`, `engine.py`, `train.py`, `corpus.py`) plus the
agent shard endpoints in `meshpool/agent/daemon.py`.

## 1. The model

The default engine is a compact **from-scratch NumPy GPT** (`model.py`, registered as
`nano-gpt`): a standard decoder-only transformer — token + positional embeddings, N
pre-norm blocks (multi-head causal self-attention + GELU MLP), a final layernorm, and a
tied LM head. It's deliberately small and dependency-free so it runs anywhere with **zero
downloads**. Because it's real transformer math, everything demonstrated here transfers
directly to a large model behind the adapter (§6).

## 2. Sharding: how a model gets split

`topology_partition` (in `controlplane/scheduler.py`) implements exo-style
topology-aware auto-partitioning:

1. For each node, compute **usable memory** (VRAM if present, else RAM, capped by the
   donation policy) and divide by per-layer memory × a headroom factor → a **layer
   capacity**.
2. If the pool's total capacity < the model's layers, refuse (honest: the model doesn't
   fit even pooled).
3. Otherwise assign each node a **contiguous block of layers** — a pipeline stage — sized
   proportionally to its compute (`flops_estimate`) but clamped to its memory capacity.
4. The **first** stage owns the token embedding; the **last** stage owns the final norm
   and LM head.

Each node then loads *only its slice* via `POST /shard/load`; `ShardModule` in `model.py`
keeps just the parameters for its layer range.

**Bring your own weights (served from the pool).** By default each node reconstructs its
slice from a checkpoint it already has on disk (`weights/nano-gpt.npz`) — fine for the
built-in model, but a user bringing a *new* trained model to fresh donor machines can't
assume every box already has the file. So a checkpoint can instead be **staged once in the
pooled, content-addressed blob store** and streamed out to only the nodes that need it:

```
POST /api/v1/models/upload?model=nano-gpt   (body = the .npz checkpoint)
        │
        ├─ _validate_checkpoint ── cheap zip-directory read; rejects a malformed or
        │                          wrong-architecture npz at the trust boundary (400)
        ├─ _pool_weights ──────── the checkpoint is split **layer group by layer group**
        │                          (one blob per transformer block, plus tiny embedding /
        │                          final-norm groups) — each stored internal=True, replicated,
        │                          hidden from the user's Files, sha256-verified on read. A
        │                          small JSON **manifest** {group name → blob id} is stored
        │                          too, and its id becomes the deployment's weights handle.
        └─ deploy(model, weights_blob_id=<manifest>)
                │
                └─ the control plane reads only the **tiny manifest** and, for each stage,
                   hands its node the layer-group blob **ids plus the URLs of peers that
                   hold them**. On a LAN the node pulls each group **directly from a peer**
                   (never through the coordinator); `shard/load` then assembles its slice
                   from its own local store.
```

The architecture/shape still comes from the shared `REGISTRY` entry (tiny); only the
**weights** travel through the pool, split per layer so no participant — not even the
control plane — ever holds more than a single layer group at once. A model whose checkpoint
is far larger than any single donor's RAM can still be served. The deployment records the
manifest's blob id, so when a node dies and the model **repartitions** (§5) the survivors
pull only their new layers' blobs straight from a peer that holds them — no local file, no
re-upload, no re-hydration of the whole checkpoint. Deployments served this way report `"pooled": true`
and the number of layer groups as `"weight_groups"` in their status view.

`sim/scenarios/verify_pooled_model.py` proves the round-trip end to end: it uploads a
checkpoint built from *distinct* weights, asserts it was split into `n_layer + 3` groups
(`weight_groups`), then asserts the distributed greedy output equals that pooled checkpoint's
single-machine output **and differs from** the node-local built-in one — so a pass is only
possible if every stage really served weights hydrated, per layer, from the pool.

*Byte-free placement (direct peer-to-peer weight pull).* Getting a stage's weight groups
onto its node does **not** route the weight bytes through the control plane. On a LAN pool
the coordinator hands the node only the blob **ids and the URLs of peers that already hold
them**, and the node pulls each group **directly from a peer** (`POST /blob_pull`),
verifying every chunk — and the reassembled group — against its sha256 before storing it.
Only a node reachable *solely* through its reverse tunnel (so it cannot dial a peer), or a
group no online peer holds, makes the control plane fall back to relaying the bytes itself
(still one group at a time — never worse than before). `GET /api/v1/cluster` counts both:
`weight_group_pulls` (direct peer pulls) versus `weight_group_pushes` (control-plane
relays). On a LAN cluster the coordinator therefore moves **zero weight bytes** at deploy or
re-partition, which `verify_pooled_model.py` asserts (`weight_group_pulls >= 1` and
`weight_group_pushes == 0`).

*Streaming upload (no whole-checkpoint buffer anywhere).* The diagram above is the
**browser** path (`/api/v1/models/upload`): the whole file is posted once and the control
plane splits it group-by-group, so that single request is bounded by the
`MESHPOOL_WEIGHTS_MAX_BYTES` cap (2 GB by default). The **`meshpool models upload` CLI** goes
one better — it splits the checkpoint *on the client* and streams a single layer group at a
time (`POST /api/v1/models/weights/group`), then registers the manifest
(`POST /api/v1/models/weights/register`, which re-checks every group at the trust boundary
one group at a time). Neither the client nor the control plane ever holds more than one group,
so a checkpoint far larger than any single machine's RAM uploads without being buffered whole
anywhere. Deploy and re-partition are per-group on both paths, so serving the model never
holds more than a single layer group at once.

## 3. Serving a token

`InferenceManager.chat()` (in `engine.py`) drives a `RemotePipeline`
(`inference/remote.py`). One **traversal** is one pass over every stage in order:

```
prompt tokens ──► stage 0 (embed + layers 0..a) ──► hidden state ──►
                  stage 1 (layers a..b)          ──► hidden state ──►
                  stage 2 (layers b..N + head)   ──► logits ──► sample next token
```

The control plane sends **token ids** to stage 0, then passes only the **hidden-state
activations** (a `[seq_len, d_model]` float tensor, a few KB) from each stage to the next.
Weights and gradients never move. That's the whole reason a LAN is fast enough — and why
`docs/` keeps repeating it.

Two things make this viable over a *wide-area* link rather than just a LAN.

### 3.1 A KV cache, so a decode step is one vector per hop

Each stage keeps a per-session key/value cache (`inference/kvcache.py`,
`CachedShard` in `inference/pipeline.py`). Prefill is **position-parallel** — the whole
prompt crosses each hop once as a single `[T, d_model]` tensor, *not* once per prompt
token. After that, each generated token ships exactly `d_model` floats per hop:

```
marginal wire cost per token = hops x d_model x 4 bytes
```

`sim/scenarios/verify_mesh_decoding.py` measures this on a real 3-process cluster and
matches the formula to the byte (1024 B/token for 2 hops x 128 floats). Generation is
therefore O(T), not O(T²), and long chats do not get progressively slower.

### 3.2 Speculative Pipeline Decoding, so one traversal yields several tokens

The cost of a wide-area token is dominated by `(N-1) x RTT`, not by compute. SPD
(`inference/spd.py`) amortises it: a small draft — **the first stage's own first few
layers, which already live on one machine, so drafting costs zero extra hops** —
proposes K tokens, and the full pipeline verifies all K in a **single** traversal.
Rejection sampling then accepts the longest correct prefix.

This is exact, not approximate. At `temperature = 0` the output is token-identical to
ordinary decoding; when sampling, the output distribution is provably unchanged.
Measured on the real 3-node cluster: **25 traversals → 9** (2.78x fewer round trips)
for identical text at k=2, 4 and 6. Under emulated wide-area latency the wall-clock
gain is **2.2–2.4x**; on a loopback cluster it is smaller simply because there is no
RTT to amortise.

Rollback after a rejected draft rides along on the *next* forward as `truncate_to`,
so correcting a bad guess costs **zero** extra round trips — otherwise the rollback
would cost exactly the `(N-1) x RTT` that SPD exists to remove.

Throughput is measured on real wall-clock time and reported as **tokens/sec** in the CLI,
the dashboard, and the OpenAI response — never an invented target.

### 3.3 Continuous batching, so one traversal serves several *tenants*

SPD amortises `(N-1) x RTT` across the tokens of one request. Continuous batching
(`inference/batching.py`) amortises the same term across **different users**, and the
two compose: one traversal carries `B` tenants × `K+1` candidates, so the RTT term is
divided by the product rather than by either factor alone.

The mechanism is deliberately narrow. Rows are **not** stacked into a single matmul —
that would change float reduction order, and a tenant's output would then depend on who
happened to share its batch. Each item is computed separately against its own KV cache,
exactly as an unbatched forward would be; the only thing that is shared is the hop. So
the saving is purely the network term, which is also the term that dominates on a mesh.

It is *continuous*, not static: admission runs at the top of every step, so a slot freed
by a finished sequence is refilled on the next traversal rather than idling until the
whole batch drains. This is what stops one 500-token request from making every 5-token
request behind it wait.

Three safeguards, because a batch is a place where tenants can leak into each other:

- **Distinct sessions per traversal**, refused at three layers (`Pipeline.forward_batch`
  raises, the agent endpoint returns 400, and a test asserts it). Two rows sharing one KV
  cache would interleave two conversations.
- **Capability negotiation.** `RemoteShard` probes `/shard/forward_batch` once per stage;
  a node that answers 404/405/501 latches to per-item forwards and increments
  `batch_fallbacks`, so degradation is visible rather than silent.
- **Exact per-sequence attribution.** Traversals, hops, stage calls, positions and wire
  bytes are attributed per request, not modelled — every inter-stage payload is
  `[rows, d_model]` float32, so a sequence's wire share is exactly
  `rows x d_model x 4 x (depth-1)`. Only `link_seconds`/`compute_seconds` are shared, and
  those are split pro rata by rows and labelled as such in the code.

Measured on the real 3-process cluster (`verify_continuous_batching.py`), 6 concurrent
requests: **288 wide-area hops → 51** (82% fewer) and **1.65–2.09x aggregate tok/s**,
with every reply token-identical to the same prompt run alone. The hop counts are read
from `GET /shard/stats` on the agents themselves — counted where the packets land, so
they cannot be inflated by the requester's own bookkeeping.

> The throughput figure was previously recorded as 2.5–3.6x. That number was wrong, and
> the way it was wrong is worth keeping on the record. The gate created a fresh
> `httpx.Client` per request, and constructing one loads the system TLS trust store —
> ~0.67 s on the test machine, versus 0.002 s through a reused client. Serving requests
> *sequentially* paid that once per request; serving them *concurrently* paid it in
> parallel. So the measurement was crediting batching for hiding a client-side
> certificate load. With one shared client the mesh does exactly the same work and the
> honest speedup is 1.65–2.09x. The hop collapse was never affected — it is counted at
> the receiving nodes, not timed.

### 3.4 Why a reply stopped

`nano-gpt` uses **learned absolute** position embeddings, so the context window cannot
slide without invalidating every cached key/value (see `kvcache.py`). Generation
therefore stops at the window rather than silently returning stale numbers.

That decision is correct, but it must be *reported*, and a token count cannot report it:
when the window truncates a reply the count lands **below** `max_tokens`, so the natural
rule `"length" if n >= max_tokens else "stop"` labels a cut-off answer as a finished one.
An OpenAI client that continues on `length` and stops on `stop` then silently drops the
rest of the response.

So the reason is produced where the decision is made and carried outward:

| Surface | Reached its token budget | Ran out of context |
|---|---|---|
| `POST /api/v1/inference/chat` | `finish_reason: "length"` | `finish_reason: "context_window"` |
| `POST /v1/chat/completions` | `finish_reason: "length"` | `finish_reason: "length"` |

OpenAI's schema admits only `stop`/`length`/`tool_calls`/`content_filter`, so a context
cut-off maps to `length` — the completion *was* ended by a limit — while MeshPool's native
response keeps the precise reason. Both are asserted by `verify_continuous_batching.py`,
and reverting either is a permanent entry in the mutation catalog.

## 4. ✨ The correctness oracle

This is the feature that makes distributed inference trustworthy.

Every node reconstructs its layer slice from the **same checkpoint** (or, with random
weights, the **same seed**). So the pipeline is just the single-machine forward pass with
cut points — mathematically identical. At `temperature = 0` (argmax, no randomness),
generation is fully deterministic, which means:

```
distributed_output  ==  single_machine_output      (bit-for-bit)
```

and, critically, **it stays identical after a node dies** and the model repartitions onto
the survivors. Distributing the model changes *where* the math runs, never *what* it
computes.

We assert this automatically:

- `tests/test_model.py::test_shard_pipeline_equals_full_model` — an in-process pipeline of
  `ShardModule`s vs `forward_full`, `max |Δ| < 1e-4` (in practice **0.0**).
- `sim/scenarios/verify_inference.py` — the full multi-process version: shard across 3
  nodes, compare distributed greedy output to a single-machine reference, then **kill a
  node**, let it repartition, and confirm the output is *still* identical.

## 5. Healing on node loss

When the registry marks a node offline, it calls `InferenceManager.on_node_offline`, which
schedules an async `_repartition` over the survivors. If their pooled memory still fits the
model, it recomputes the partition, reloads the affected shards, and continues serving —
with (per §4) identical output. If the survivors can't fit it, the deployment is reported
degraded rather than silently returning wrong answers.

## 6. The adapter seam (real large models)

`nano-gpt` is the always-runnable default, but the engine is structured so a production
backend — **Petals/hivemind**, **exo**, or a local **torch** model — plugs in behind the
same `deploy / chat / on_node_offline` interface and the same OpenAI API. The sharding,
healing, measurement and API surface are all backend-agnostic; only the tensor kernels
change. This is how you go from "proves the concept" to "serves a 70B model across the
office" without touching the control plane.

## 7. OpenAI-compatible API

The control plane exposes `POST /v1/chat/completions`, so existing OpenAI clients/tools
work unchanged:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "nano-gpt",
  "messages": [{"role": "user", "content": "MeshPool is"}],
  "temperature": 0.7,
  "max_tokens": 40
}'
```

There's also a simpler native route, `POST /api/v1/inference/chat`
(`{deployment_id, prompt, max_new_tokens}`), used by the dashboard chat panel.

## 8. Training the demo weights

So the demo emits coherent, on-topic text (instead of noise from random weights), a tiny
trainer is included: `meshpool/inference/train.py`, on an original MeshPool-themed corpus
(`corpus.py`).

- **Hand-written backprop + Adam**, fully vectorised over a batch dimension.
- The backward pass is **numerically verified** by a float64 central-difference gradient
  check (`gradient_check`, and `tests/test_train.py`). Float64 matters: float32 rounding
  swamps the finite differences and yields false failures.
- LR warmup + cosine decay and GPT-2-style residual-projection init scaling get it off the
  unigram-entropy plateau.

Build the checkpoint (writes `weights/nano-gpt.npz`, which every node then loads):

```bash
python -m meshpool.inference.train --steps 1600 --batch 32 --block 64
python -m meshpool.inference.train --check-only     # just run the gradient check
```

The character tokenizer (`CharTokenizer`) ships its `itos` table inside the checkpoint, so
encode/decode round-trips exactly.

## LoRA fine-tuning (single-machine)

Beyond building the base checkpoint, MeshPool ships **LoRA fine-tuning** to adapt the nano-GPT
to new text without touching the frozen base weights (`meshpool/inference/lora.py`):

```bash
python -m meshpool.inference.lora --steps 300 --rank 4             # fine-tune on the built-in corpus
python -m meshpool.inference.lora --text mydata.txt --targets all  # adapt attn+MLP on your own corpus
python -m meshpool.inference.lora --check-only                     # just run the LoRA gradient check
```

Only the low-rank adapters `A@B` (injected into the attention `Wqkv`/`Wo`, optionally the MLP)
are trained; the base stays frozen, so an adapter is a few percent of the parameters and a
fine-tune runs in seconds on one machine. Rather than a fresh, error-prone backward pass, the
LoRA path **reuses the gradient-checked `train.backward`**: it backprops through the effective
weight `W_eff = W + (alpha/r)·A@B` and projects the result onto the factors (`dA = s·G@Bᵀ`,
`dB = s·Aᵀ@G`). That projection has its own float64 finite-difference gate
(`gradient_check_lora`) plus a loss-decreases-on-a-fixed-corpus test (`tests/test_lora.py`).
`--merge-out` folds the adapter back into a full checkpoint so the ordinary inference engine
serves the fine-tune with no changes.

Training here is single-machine. **Distributed** *training* stays out of scope for now — it's
bandwidth/latency-bound on consumer networks (see the roadmap).
