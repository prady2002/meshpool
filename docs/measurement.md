# MeshPool — Measurement and Fault Injection

## What is measured, how, and how often

### At agent startup (`probe_capability`)

| Metric | Method | Accuracy |
|--------|--------|----------|
| **CPU cores** | `os.cpu_count()` | Exact logical core count |
| **Total RAM** | `GlobalMemoryStatusEx` (Windows) / `sysconf` (POSIX) | Exact physical RAM reported by OS |
| **Disk space** | `shutil.disk_usage(cwd).free` | Snapshot at startup; may drift |
| **GPU presence + VRAM** | nvidia-smi → rocm-smi → system\_profiler (Darwin) → WMI Win32\_VideoController (Windows) | Best-effort; degrades to `(False, 0)` if all tools absent |
| **FLOPS estimate** | Adaptive NumPy float32 matmul; matrix grown until single-trial wall time ≥ 100 ms, best of 3 trials | Rough proxy (GFLOP/s on one thread); not a formal benchmark. Turbo, thermal throttling, and memory bandwidth all affect it. Treat as an order-of-magnitude ranking, not an absolute score. |

Probe runs **once at startup**. Capabilities are not re-probed during a session; the simulation harness can override any field via `MESHPOOL_SIM_*` env vars.

### At every registration (`measure_link`)

| Metric | Method | Accuracy |
|--------|--------|----------|
| **Latency p50/p90** | 7 sequential GETs to `/api/v1/health`, single persistent `httpx.Client`. First sample (connection setup) recorded separately; p50/p90/stdev computed from remaining 6. | Measures request RTT including TLS handshake on first connection. On localhost (sim harness) this is honest ~0 ms. |
| **First-connect time** | Wall time of the first GET including connection establishment | One data point; includes TLS, DNS |
| **Bandwidth** | Dedicated `/api/v1/probe/download` endpoint on the control plane. Ramped transfer (64 KB → 8 MB), excluding warm-up. Reports best throughput when transfer ≥ 200 ms. | **Only reported when the endpoint exists.** Returns `None` otherwise — never fabricated from payload size. |

#### What `measure_link` honestly cannot do

- **Measure upload bandwidth** without a corresponding `/api/v1/probe/upload` endpoint.
- **Distinguish transient congestion from steady-state throughput** (a single probe window is not enough).
- **Account for asymmetric links** (residential: 500 Mbps down / 20 Mbps up) without separate up/down probes.
- Report `bandwidth_mbps > 0` when the control plane does not expose a bandwidth-probe endpoint.

The internal `_LinkMeasurement` dataclass carries `bandwidth_mbps: Optional[float]` where `None` means "not measured". The wire format `LinkStats` maps this to `0.0` for backwards compatibility — downstream code should treat `0.0` bandwidth as "unknown" unless the control plane explicitly reports otherwise.

---

## Fault injection framework (`meshpool/common/faultinject.py`)

### Purpose

The fault injector exists solely to emulate a **dishonest node** in the Tier-2 trust acceptance test (`sim/scenarios/verify_trust_credits.py`). The coordinator's spot-check oracle must catch corrupted results and reject them. No real deployment should ever run with fault injection active.

### What it can simulate

| Fault | Description |
|-------|-------------|
| `corrupt_results` | Deterministically mutates the work-unit result (bool flipped, int/float +1, string appended "X", list head-dropped, dict values mutated) |
| `drop_work` | (API surface; not yet wired up) Silently ignore a work unit |
| `stall_seconds` | Delays every inference shard hop by a fixed amount, turning a healthy node into a straggler. Settable at startup via `MESHPOOL_STALL_MS`, or at runtime via `POST /fault/stall` on the agent (403 unless fault injection is opted in). Drives `verify_straggler_tolerance.py` |
| `corrupt_blobs` | Turns a node into a **dishonest peer**: every content-addressed blob it serves comes back with one byte flipped. Deliberately the *hardest* lie to catch — same length, still valid base64 — so only a digest check rejects it, and a puller that checks anything cheaper fails the gate. Settable at runtime via `POST /fault/corrupt_blobs` on the agent (403 unless fault injection is opted in). Drives `verify_weight_swarm_coldstart.py` |
| `lie_capability` | (API surface; not yet wired up) Advertise false hardware specs |

### How to enable in tests / the sim harness

Fault injection requires **all** of the following to be true:

1. `_FAULTINJECT_STRIPPED` constant in `faultinject.py` is `False` (the default in development).
2. One of the opt-in gates is satisfied:
   - `MESHPOOL_ALLOW_FAULT_INJECTION=1` set explicitly, **or**
   - Running under pytest (`PYTEST_CURRENT_TEST` is in the environment), **or**
   - Running under the sim harness (`MESHPOOL_SIM_HARNESS=1`).
3. The specific fault variable (currently `MESHPOOL_LIE=1`) is set.

The sim harness sets `MESHPOOL_LIE=1` on the designated liar node index automatically. Legacy tests that set only `MESHPOOL_LIE=1` continue to work because pytest always satisfies gate 2b.

Every activation emits a **loud WARNING** log line naming the fault, so no corruption ever goes silently unnoticed in test output or CI logs.

### Production guarantee

A production binary **must** set `_FAULTINJECT_STRIPPED = True` (or have a build step rewrite it). When stripped:

- `active_profile()` always returns an inert `FaultProfile` (all faults off).
- `_opt_in()` returns `False` unconditionally.
- Setting `MESHPOOL_LIE=1` in the environment has no effect whatsoever.

This means even if a malicious environment variable is injected at runtime, the fault injector cannot activate in a stripped build.

---

## Required follow-up items

1. **Bandwidth-probe endpoint** — `measure_link` needs `GET /api/v1/probe/download?size=<bytes>` on the control plane (owned by `meshpool/controlplane/`) to return real bandwidth measurements. Without it, `bandwidth_mbps` is always unknown. A corresponding `POST /api/v1/probe/upload` is needed for upload measurement.

2. **Upload bandwidth** — `_measure_link_detail` currently only measures download. Symmetric probing requires the upload endpoint above.

3. **`LinkStats` richer fields** — `meshpool/common/models.py` (not owned by this agent) could gain `latency_p90_ms`, `latency_jitter_ms`, and `bandwidth_measured: bool` fields for placement and scheduling decisions. Until then, callers that need richer data should use `_measure_link_detail()` directly.
