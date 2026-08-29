# Service levels, error budgets and runbooks

> **Every number in this document is either MEASURED by a named gate in
> `sim/scenarios/`, or labelled as a target that is not yet met.** Nothing here
> is aspirational marketing. If a gate is not named, the number is not a claim.
> This follows the tier discipline in `docs/measurement.md`.

---

## 1. Why this document is short

MeshPool runs on donated consumer machines that thermally throttle, get
unplugged, and lose Wi-Fi. Promising four nines on that substrate would be a
lie. The honest position is: **the mesh is unreliable and the product's job is
to hide that**, which is why the SLOs below are written about *user-visible
outcomes* (did the answer arrive, was it correct, was the ledger right) rather
than about node uptime, which nobody can control.

---

## 2. Service level indicators

| # | SLI | How it is measured | Gate |
|---|---|---|---|
| S1 | **Correctness of a sharded answer** | sharded output compared bit-for-bit against single-process output at temperature 0 | `verify_inference`, `verify_first_real_token`, `verify_moe_placement` |
| S2 | **Correctness under speculation** | SPD output distribution vs. baseline | `verify_spd_equivalence` |
| S3 | **Time to first token** | first SSE delta timestamp minus request timestamp | `verify_streaming_ttft` |
| S4 | **Generation survives a node dying mid-stream** | kill a stage during generation; stream continues and output stays correct | `verify_stage_failure_midstream` |
| S5 | **Ledger conservation** | credits in ≡ credits out, across restarts | `verify_ledger_conservation`, `verify_chaos` |
| S6 | **No unit lost under concurrency** | submitted units ≡ done + failed | `verify_scale`, `verify_batch` |
| S7 | **Spend cap stops work** | a capped job stops and is resumable, not silently hung | `verify_spend_cap_hard_stop` |
| S8 | **Egress containment** | sandboxed work cannot reach a blocked destination | `verify_egress_filter` |

S1, S2, S5, S6, S7 and S8 are **binary correctness SLIs with a 100% objective
and a zero error budget.** A single violation is a release blocker, not a
budget spend. Correctness is not something to be 99.9% of.

S3 and S4 are latency/availability SLIs and carry real budgets, below.

---

## 3. Objectives, and what has actually been measured

| SLO | Objective | Last measured | Tier | Status |
|---|---|---|---|---|
| S3 TTFT, chat + SPD, single box | < 500 ms | **97 ms** | measured | met |
| S3 TTFT, legacy completions route | < 500 ms | **113 ms** | measured | met |
| S4 serve a fresh request after a client abandons mid-stream | < 2 s | **0.22 s** | measured | met |
| Coordinator hard-kill → both agents reconnected, state intact | < 60 s | **27 s** (whole gate) | measured | met |
| Batch accounting under 4 concurrent jobs × 20 units | 0 lost | **done=80, failed=0** | measured | met |
| Kimi K2 (1T) on an 11-node metro constellation | — | **15.2 tok/s** | **calibrated projection** | see caveat |
| Llama-2-70B on a 4-node metro constellation | — | **48.0 tok/s** | **calibrated projection** | see caveat |
| Performance-model accuracy on held-out configs | MAPE ≤ 25% | **2.7%** | measured | met |

**Caveat on the two projections.** They are ceilings derived from network
traversal cost with the compute term modelled, produced by
`verify_benchmarks.py` and fitted by `verify_netem_calibration.py`. They are
**not** measurements of a real 1T model on real consumer hardware, because no
such hardware is available to this project. The plan originally projected
~50 tok/s for a 1T model; the honest figure is **15.2 tok/s**, and `README.md`
says so explicitly. Do not put the 50 on any user-facing surface.

---

## 4. Error budgets

For the two SLOs that carry budgets:

| SLO | Window | Objective | Budget |
|---|---|---|---|
| S3 TTFT under 2 s at p95 | 28 days | 99.0% | 7 h 12 m |
| S4 request completes despite stage churn | 28 days | 99.5% | 3 h 22 m |

**Budget policy.** When more than 50% of a budget is consumed in a window,
feature work on the affected service stops and reliability work takes
priority. When a budget is fully consumed, the service is put into
conservative mode: constellation-local placement only, hot spares raised from
2 to 3 per stage, and speculative K reduced (SPD's acceptance rate falls under
churn, and a rejected draft is wasted traversal).

There is deliberately **no availability SLO on individual donated nodes.** A
node is free to vanish at any moment; that is a design assumption, not an
incident.

---

## 5. Signals available today

| Signal | Where | Notes |
|---|---|---|
| Liveness | `GET /api/v1/health` | Also the target of the health-based routing used by Mesh Apps |
| Live event stream | `GET /api/v1/events` (SSE, scope `EVENTS_SUBSCRIBE`) | Job lifecycle, node join/leave, failover, spend-cap pause |
| Per-job timeline | job timeline API, gated by `verify_job_timeline` | What the console renders directly |
| Per-job receipt | metering/receipts, gated by `verify_metering_receipts` | Which nodes, how long, what it cost, what the platform took |
| Absorbed transient socket faults | `proactor_fix.absorbed_count()` | Counted rather than logged after the first; a rising count means the host network stack is unhappy |

**Gap, stated honestly:** there is no Prometheus/OpenTelemetry exporter and no
hosted status page yet. Phase 0.3 defines the tracing work and it is not
finished. Until it is, operational visibility is the event stream plus the
per-job timeline, which is sufficient for a beta but not for a paid SLA.

---

## 6. Runbooks

### 6.1 Coordinator will not start

1. Check the data directory is intact — the ledger is durable and the
   coordinator is designed to restart onto the same directory.
2. Restart it. Agents reconnect **on their own** with no re-onboarding; this
   is covered by `verify_chaos.py` steps 2–3.
3. Verify the balance was preserved and the signup grant was **not** re-issued
   (a re-issued grant means the ledger was reset, which is a data-loss event,
   not a restart).

### 6.2 A job looks hung

1. First check whether it is `paused`, not `running`. A job that hit its hard
   spend cap is deliberately halted and is **resumable by raising the cap** —
   this used to be indistinguishable from a hang and is now an explicit status.
2. Check the per-job timeline for the last dispatched unit.
3. If units are pending with no node holding them, a node died between claim
   and report; the batch engine retries up to `MAX_ATTEMPTS=5`.

### 6.3 Generation is slow

1. Read `pipeline_depth` for the request. The governing cost is
   `Σ compute + (N−1) × RTT`; depth is usually the problem, not the GPUs.
2. Check whether the request was placed inside one constellation or scattered.
   Scattered placement costs roughly 5× a metro placement for the same model
   (807 ms vs 157 ms traversal for an 11-node Kimi K2 shard).
3. Check the SPD acceptance rate. If it has collapsed, K should be reduced;
   rejected drafts are wasted work.

### 6.4 A node is suspected of returning wrong results

1. `verify_fraud_detection` describes the mechanism: reference re-execution on
   a sample of work, with the node's stake at risk.
2. Do **not** rely on cryptographic proof for inference — it is a stated
   non-goal (§8.4 of the plan) and is not implemented, because it is not
   practical for LLMs.

### 6.5 Suspected abuse from the mesh

1. Egress is deny-by-default and whitelist-only inside the sandbox, and the
   sandbox **fails closed**: a child that cannot install the egress hook
   refuses to run the work rather than running it unprotected.
2. `verify_egress_filter.py` is mutation-proven — removing the production hook
   installation makes the gate fail. If that gate is green, enforcement is
   live, not merely declared.

---

## 7. Incident severities

| Sev | Definition | Example |
|---|---|---|
| **1** | Correctness or money is wrong | ledger does not conserve; a sharded answer differs from single-process; a spend cap did not stop work |
| **2** | Containment is wrong | sandboxed work reached a blocked destination; a principal accessed something outside its scopes |
| **3** | Availability degraded | TTFT budget burning fast; a constellation cannot form |
| **4** | Cosmetic or single-node | one donated node is unhealthy |

Sev 1 and Sev 2 have a **zero error budget** and block releases. Everything in
§2 marked binary is Sev 1 or Sev 2 by construction.
