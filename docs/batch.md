# Batch engine

Run an embarrassingly-parallel job across every machine's spare CPU, with automatic
retry of any unit whose machine dies. Code: `meshpool/batch/` +
`controlplane/scheduler.py::best_fit_node`.

## Job model

A **job** = a named worker function + a list of independent **work units** (each a JSON
payload), plus a per-unit CPU% and RAM request:

```
BatchJob { func, units: [WorkUnit{payload}], cpu_pct_per_unit, ram_mb_per_unit }
```

Submit via CLI or `POST /api/v1/jobs`:

```bash
python -m meshpool.cli.main jobs submit --count 30      # 30 units of the default func
python -m meshpool.cli.main jobs list
```

## Built-in worker functions

Registered in `meshpool/batch/functions.py` (real CPU-bound work, so distribution is
observable): `square`, `is_prime`, `sha_pow` (default — a hash proof-of-work with
`{seed, rounds}`), `embed`, `sleep_echo`, and `secure_fetch` (demonstrates using a
**brokered secret** — see below). Adding your own is a one-line entry in the
`REGISTRY` map; in production this is a container image + command.

## Brokered secrets (credentials without leaks)

A job may declare `secret_names` (e.g. `["API_TOKEN"]`). The owner first stores the
value with `meshpool secrets put API_TOKEN <value>`; the control plane keeps it **in
memory only** and never returns it from any API. At dispatch, a **single-use,
node-scoped, TTL-bounded lease** is minted per unit and only the lease id travels with
the work. The agent redeems it just-in-time and injects the value under
`payload["_secrets"]`, so the credential is never in the job spec or on a worker's
disk. See `docs/security.md` and the `verify_secrets` scenario.

## Scheduling (spread / worst-fit)

`BatchManager.dispatch_once` pulls pending units and places each with `_pick_node`, which
picks the schedulable node with the **most free CPU slots** (worst-fit). This spreads a job
evenly across the cluster in proportion to each node's donatable cores, keeping every
machine busy instead of hot-spotting one node. A node's concurrent unit count never exceeds
its donatable CPU slots, and its RAM budget is checked against the donation cap, so a node's
worker pool can't be oversubscribed. When trust Tier > 0, fair-share weight breaks ties. (The
tighter-packing `best_fit_node` in `scheduler.py` is a placement-sizing utility, not the
batch dispatcher.)

## Reliability: retry on a different node

This is the core correctness property (spec §9.4: *"never assume a failure is the job's
fault"*):

- If a unit's node dies, `requeue_node_units` flips that node's in-flight units back to
  `PENDING`.
- `_run_unit` catches **any** failure and re-queues the unit.
- Re-dispatch **excludes nodes that already failed this unit**, so a poisoned node can't
  keep eating the same unit.
- Up to `MAX_ATTEMPTS` (5) tries before a unit is marked `FAILED` with its error — bounded,
  never an infinite loop.

## Proven by

`sim/scenarios/verify_batch.py` — submits a batch, asserts it spreads across all online
nodes, **kills a node mid-job**, and verifies its in-flight units complete on the
survivors and the job finishes. This is MVP acceptance criterion (d).
