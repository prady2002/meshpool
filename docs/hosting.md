# Hosting engine

Deploy a web app across several machines behind a load balancer; when a machine sleeps or
dies, traffic re-routes and the lost replica is rescheduled — **no dropped requests beyond
in-flight**. Code: `meshpool/hosting/` + `controlplane/scheduler.py::bin_pack_replicas`.

## Deploy model

A **deployment** = an app + a desired replica count:

```bash
python -m meshpool.cli.main deploy hello --replicas 2
python -m meshpool.cli.main deployments
```

`hello` is a built-in demo app (`meshpool/hosting/demo_app.py`); in production this is a
Dockerfile or a detected buildpack stack.

## Placement (bin-packing with anti-affinity)

`bin_pack_replicas` places replicas across **distinct nodes first** (anti-affinity, so a
single node failing can't take down every replica), then packs extras onto nodes with
remaining headroom. It never promises more replicas than the pool can physically hold.

## Load balancing

The control plane exposes each deployment at `/app/{id}/...` and proxies to replicas
**round-robin over only the healthy ones** (`HostingManager.pick_replica`). Because the
balancer consults live health on every pick, a machine going down mid-traffic causes
subsequent requests to route to healthy replicas rather than erroring.

## Health checks + self-healing failover

- Every replica is health-checked continuously (`GET {replica}/healthz`).
- When the registry marks a node offline, `HostingManager.on_node_offline` drops that
  node's replicas and **scales back up to the desired count** on the survivors.
- Net effect: sleep or kill a machine mid-traffic → the balancer stops routing to it
  immediately, and a replacement replica is scheduled elsewhere.

## Proven by

`sim/scenarios/verify_hosting.py` — deploys across nodes, drives traffic, **kills a node
mid-traffic**, and asserts requests keep succeeding (no drops beyond in-flight) while the
deployment self-heals back to its desired replica count. MVP acceptance criterion (c).

## Large apps: staged in the pool

A hosted app normally rides **inline** inside the `start_replica` control frame, which is
comfortable up to `MESHPOOL_SITE_MAX_BYTES` (25 MB by default). A genuinely large
multi-page app (a big SPA bundle, a static-site-generator's whole output) is **not
rejected**: the control plane detects the oversize bundle (`SiteTooLarge`) and stages it
in the **pooled, content-addressed blob store** — split into replicated chunks spread
across the mesh (`meshpool/storage/`), up to `MESHPOOL_BUNDLE_MAX_BYTES` (500 MB).

- The deployment then carries a `site_blob_id` instead of an inline `site`.
- On every replica start — the initial deploy **and** self-heal after a node dies —
  `HostingManager._hydrate_site` pulls the bundle back from the pool, verifies it
  (sha256), and hands it to the replica. The uploader need not still be online.
- The bundle is stored `internal=True`, so it never appears in the user's own Files
  listing or `storage ls`.
- Small apps are byte-for-byte unchanged: they take the exact inline path as before.

This is how a machine pool hosts an app **larger than any one control frame** — the same
"pool a resource one box can't provide" property the object store gives disk.

## Proven by

`sim/scenarios/verify_bigapp.py` — shrinks the inline cap to 4 KB so a modest multi-page
app is forced through the pool, then asserts it is staged (`site_blob_id` set, inline
`site` cleared, hidden from Files), **hydrated from the pool** onto replicas and served
through the proxy via both the JSON and browser `.zip` entry points, and **re-hydrated
from the pool** onto a survivor after a hosting node is killed.

