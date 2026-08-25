# Quickstart

Boot a full MeshPool cluster on one machine, watch it in the dashboard, and drive all
three engines from the CLI. No Docker, no GPU, no accounts.

## 0. Prerequisites

- Python 3.11+
- `pip install numpy fastapi uvicorn pydantic httpx typer rich` (plus `pytest` for tests)

All commands run from the repository root.

## 1. Start a cluster

```bash
python -m sim.harness --agents 3
```

This launches one control plane on `:8080` and three **simulated heterogeneous
machines** (different CPU/RAM/compute profiles, so scheduling and sharding are
non-trivial). You'll see:

```
Starting MeshPool: control plane + 3 agents ...
Dashboard:  http://127.0.0.1:8080
Nodes:      http://127.0.0.1:8080/api/v1/nodes
Ctrl-C to stop.
```

Open **http://127.0.0.1:8080** — three machines appear live with capabilities,
donation caps, health, trust and contribution stats.

## 2. Point the CLI at the control plane

```bash
# Windows PowerShell
$env:MESHPOOL_CONTROL_PLANE_URL = "http://127.0.0.1:8080"
# macOS/Linux
export MESHPOOL_CONTROL_PLANE_URL=http://127.0.0.1:8080
```

Run the CLI in-tree with `python -m meshpool.cli.main <command>` (or `meshpool <command>`
if installed).

```bash
python -m meshpool.cli.main nodes
```

## 3. Run a batch job across every machine (criterion d)

```bash
python -m meshpool.cli.main jobs submit --count 30
python -m meshpool.cli.main jobs list
```

Watch the 30 units distribute across all three nodes' spare CPU on the dashboard.

## 4. Host an app with failover (criterion c)

```bash
python -m meshpool.cli.main deploy hello --replicas 2
python -m meshpool.cli.main deployments
```

The app is placed on two nodes behind a load balancer. Reclaim one of them
(`... reclaim <node-id>`) and the deployment self-heals — traffic re-routes and the lost
replica is rescheduled.

## 5. Chat with a model too big for one machine (criterion e)

```bash
python -m meshpool.cli.main models deploy nano-gpt
python -m meshpool.cli.main models list          # see the pipeline stages per node
python -m meshpool.cli.main chat "MeshPool lets you"
```

The model's layers are sharded across the nodes; the reply prints with a measured
tokens/sec. Any OpenAI client works too:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "nano-gpt", "messages": [{"role":"user","content":"MeshPool is"}]
}'
```

## 6. Prove it end-to-end (no clicking)

Each scenario boots a real multi-process cluster and asserts a hard invariant, then
tears down:

```bash
python -m sim.scenarios.verify_batch
python -m sim.scenarios.verify_hosting
python -m sim.scenarios.verify_inference
```

`verify_inference` additionally kills a node mid-serve and checks the output is **still
bit-identical** after the model repartitions onto the survivors.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Address already in use` | Another cluster is running; stop it, or use `--port 8090`. |
| CLI can't connect | Ensure `MESHPOOL_CONTROL_PLANE_URL` matches the harness port. |
| Nodes don't appear | Give it ~3s (first heartbeat); check the harness terminal for errors. |
| Chat output is repetitive | Expected before training completes; run `python -m meshpool.inference.train` to build `weights/nano-gpt.npz`, then redeploy. |
| Leftover `_data_*` temp dirs | Safe to delete after scenarios finish. |

Next: [two-machine test](two-machine-test.md) · [architecture](architecture.md) · [inference deep-dive](inference.md).
