# MeshPool Control-Plane Durability & Concurrency Model

> **Status:** Tier 0 (single-node coordinator). HA / multi-coordinator is a
> later phase — stated plainly in [Known limits](#known-limits).

---

## 1. Locking model

### 1.1 `Store._lock` (threading.RLock)

`Store._lock` serialises **all** access to the single shared SQLite connection —
both reads and writes.  It must be held any time `self._conn` is touched.

**Why a real OS lock, not asyncio primitives?**  FastAPI routes declared with
plain `def` (not `async def`) are dispatched by Starlette into a thread pool
and run concurrently with the asyncio event loop.  An `asyncio.Lock` does not
protect against cross-thread races; only a `threading.RLock` does.

**Reentrancy:** The lock is reentrant (`RLock`) so a single call chain that
acquires it twice (e.g. `_migrate → _current_version → _table_exists`) is safe.

**Store writes outside the registry lock:** The in-memory registry lock is
released _before_ calling `store.*`; the store then acquires its own lock.  This
prevents a deadlock where the registry lock is held while waiting for the store
lock and vice versa.

### 1.2 `NodeRegistry._lock` (threading.RLock)

Protects the in-memory dictionaries `_nodes`, `_tokens`, `_token_to_node`.
Must be held for any **check-then-act** sequence on those structures, e.g.:

- `register`: find-existing-by-URL-or-device_id → create-or-reuse node object
  (without the lock, two concurrent registrations of the same URL can both fail
  to find the existing node and create duplicates)
- `heartbeat`: read node → mutate fields
- `reap`: snapshot → mark offline (double-check under lock after snapshot)

The lock is released before calling into `store.*` (see §1.1).

### 1.3 `BatchManager._lock` (asyncio.Lock)

The batch engine is purely async; it uses an `asyncio.Lock` to protect:

- `_inflight` (node → running unit count): prevents over-committing a node's
  CPU slots when multiple async dispatch tasks run concurrently
- `_pick_node` + inflight increment: the check-and-increment is atomic within
  `dispatch_once` under the lock

### 1.4 What each lock protects (summary)

| Lock | Scope | Protects |
|---|---|---|
| `Store._lock` | Global (all threads) | SQLite connection; all DB reads & writes |
| `NodeRegistry._lock` | All threads | `_nodes`, `_tokens`, `_token_to_node` |
| `BatchManager._lock` | asyncio event loop | `_inflight`, `jobs`, unit dispatch |

---

## 2. Durability guarantees

### 2.1 SQLite PRAGMAs

Every `Store` instance sets:

```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;
```

**WAL** (`journal_mode=WAL`): readers never block writers and writers never
block readers, giving higher throughput in a mixed read/write workload.

**FULL synchronous** (`synchronous=FULL`): the WAL file is `fsync`ed before
each checkpoint.  This means every committed transaction survives a power loss
or OOM kill.  The SQLite default for WAL is `NORMAL`, which skips the fsync —
fast but not crash-safe for the WAL file itself.

### 2.2 Crash-recovery guarantee

With WAL + `synchronous=FULL`, the guarantee is:

> Every transaction that returned `COMMIT` to the caller is present in the
> recovered database after an abrupt process kill, power failure, or OOM event.

This is verified by `sim/scenarios/verify_crash_recovery.py` which hard-kills
the control-plane process mid-flight and asserts all acknowledged writes are
present in the recovered database.

---

## 3. Atomic multi-step operations

### 3.1 `add_ledger_pair(donation, consumption)`

The two ledger entries for a completed work unit (one "donated" credit for the
node owner, one "consumed" debit for the job owner) must be written
all-or-nothing.  A crash between the two would leave the ledger asymmetric,
making the net balance incorrect.

**Fix:** `Store.add_ledger_pair()` inserts both entries in a single
`BEGIN … COMMIT` transaction using `INSERT OR IGNORE`.  Both entries appear or
neither does.

**Required caller change (reported, not yet applied):** `meshpool/batch/engine.py`
still calls `ledger.record_donation()` then `ledger.record_consumption()`
separately.  These each commit independently; a kill between the two results in
a half-charged ledger.  The callers should be updated to call
`store.add_ledger_pair()` directly, or `ContributionLedger` should expose a
`record_unit_completion(donation, consumption)` method that delegates to
`add_ledger_pair`.

### 3.2 Job submission (`BatchManager.submit`)

`submit()` writes the job to `self.jobs` (in-memory) and then immediately to
the store.  A crash between the two leaves the in-memory dict out of sync with
the store, but on restart the dict is rebuilt from the store, so no data is
lost.

### 3.3 Node registration

`register()` writes the node to `_nodes`, then `store.put_node()`, then
`store.put_node_token()`.  A crash between the store calls leaves the token
row absent.  On restart, `load_node_tokens()` only loads tokens for nodes that
are already in `_nodes`, so the node is still there but its token is lost and
the agent must re-register — which is correct and safe (it will receive a new
token).

---

## 4. Idempotency

### 4.1 Heartbeats

Heartbeats are naturally idempotent: a duplicate heartbeat from a retried
request simply updates the node's `last_heartbeat_ms` and load fields to the
same (or newer) values.

### 4.2 Node registration

`register()` matches an incoming request to an existing node by device_id (global
mode) or URL (LAN mode) before creating a new record.  A retried registration
reuses the existing node_id and returns the same (or refreshed) session token.

### 4.3 Ledger entries (`add_ledger_idempotent`)

`Store.add_ledger_idempotent(entry)` uses `INSERT OR IGNORE` keyed on the
entry's `id` field.  Callers that derive the id deterministically (e.g. from
`unit_id + kind + ":v1"`) can call this method on retries without double-charging.

`Store.mark_grant(subject)` is already idempotent via `INSERT OR IGNORE` —
a starter grant is awarded exactly once per subject even if the endpoint is
retried.

### 4.4 Starter grants

`Store.mark_grant` uses `INSERT OR IGNORE` so a concurrently retried request
(two handlers racing the same device/account) awards at most one grant.

---

## 5. Crash recovery of in-flight work

On restart, batch work units that were in `ASSIGNED` or `RUNNING` state when
the process was killed are stuck: the node they were dispatched to may or may
not have actually run them, and the batch manager's in-memory `_inflight` count
is reset to zero.

**Fix:** `Store.recover_inflight_units()` moves all `ASSIGNED`/`RUNNING` units
back to `PENDING` so the batch manager re-dispatches them on its first tick.
This is idempotent (calling it twice is safe) and the result is verifiable: the
batch tests assert that no unit is left in a non-terminal stuck state.

The call site for `recover_inflight_units()` is `BatchManager.__init__`, which
reloads jobs from the store on startup.

**Required caller change (reported):** Currently `BatchManager.__init__` does
not call `recover_inflight_units()`.  The fix should be added there:

```python
# meshpool/batch/engine.py  BatchManager.__init__
self.store.recover_inflight_units()   # requeue crash-interrupted units
```

---

## 6. Retention / compaction

Unbounded table growth is bounded by two methods, intended to be called from a
periodic background task (e.g. nightly or when the table exceeds a threshold):

| Method | Default | Deletes |
|---|---|---|
| `Store.prune_ledger(keep_count=50_000)` | 50 000 rows | Oldest ledger rows |
| `Store.prune_completed_jobs(keep_count=10_000)` | 10 000 rows | Oldest terminal jobs |

Running jobs are never pruned.  Both methods return the number of deleted rows
and are safe to call concurrently (they hold `_lock` while operating).

For very long-running deployments (multi-year) where billing accuracy over the
full history is required, ledger aggregates should be snapshotted to an external
store before pruning.

---

## 7. Known limits

1. **Single-node coordinator.** MeshPool's control plane is intentionally a
   single-process, single-machine service at this stage.  There is no active
   standby, no Raft, and no distributed consensus.  HA (multiple coordinators
   with leader election) is a future phase and is explicitly out of scope.

2. **SQLite, not Postgres.** The spec §6 SQLite→Postgres path exists in the
   migration runner but is not yet activated.  At Tier 2 public scale,
   Postgres with proper WAL replication would be the correct choice.

3. **In-memory state on restart.** The registry's in-memory node map is rebuilt
   from the DB on startup but starts with all nodes `OFFLINE`.  Nodes must
   re-heartbeat before they are schedulable again; this means a brief
   post-restart window where no work is dispatched even if agents are running.

4. **Half-charged ledger (temporary).** Until `batch/engine.py` is updated to
   use `add_ledger_pair()`, a kill between the two `add_ledger` calls can leave
   the ledger with a donation but no matching consumption (or vice versa).  The
   net imbalance is bounded by the number of units that were completing at the
   instant of the crash — typically O(1) per node.

5. **No WAL checkpointing control.** The WAL file can grow without bound if
   many writes happen between reads.  A periodic `PRAGMA wal_checkpoint(FULL)`
   should be added to a background task.
