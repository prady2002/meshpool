# Wire contracts (`meshpool.proto`)

This directory holds the **shared contracts** for every message that crosses a
node ↔ control-plane or node ↔ node boundary (spec §8).

## Two transports, one schema

| | Reference / production stack | This runnable build |
|---|---|---|
| Definition | `meshpool.proto` (proto3) | `meshpool/common/models.py` (Pydantic) |
| Transport | gRPC (HTTP/2), streaming for activations | HTTP/1.1 + JSON, SSE for events |
| Codegen | `protoc` → Go/Python stubs | none needed — Pydantic *is* the schema |

The Pydantic models are kept **1:1** with the proto messages (same field names,
same shapes), so moving to gRPC is a transport swap, not a redesign. `protoc` is
not available in this environment, so no generated stubs are checked in; the
`.proto` is the canonical, reviewable contract.

## Service map (proto RPC → HTTP endpoint in this build)

| proto RPC | HTTP endpoint |
|---|---|
| `ControlPlane.Register` | `POST /api/v1/nodes/register` |
| `ControlPlane.Heartbeat` | `POST /api/v1/nodes/heartbeat` |
| `ControlPlane.PutSecret` / `ListSecrets` | `POST /api/v1/secrets` · `GET /api/v1/secrets` |
| `ControlPlane.RedeemSecret` | `POST /api/v1/secrets/redeem` (node identity from `X-MeshPool-Node-Token`) |
| `Agent.RunUnit` | agent `POST /run_unit` (batch unit) |
| `Agent.LoadShard` / `UnloadShard` | agent `POST /shard/load` · `/shard/unload` |
| `Agent.Forward` (stream) | agent `POST /shard/forward` (activation hop) |
| `Agent.Reclaim` | `POST /api/v1/nodes/{id}/reclaim` |

The one place the transports differ meaningfully is `Agent.Forward`: gRPC models it
as **bidirectional streaming** so the pipeline stays full (micro-batching hides
bubbles); the HTTP build issues one activation-hop request per stage per token. Only
hidden-state activations are ever sent — never weights or gradients.

## Regenerating stubs (production)

```bash
protoc --go_out=gen --go-grpc_out=gen meshpool.proto      # Go services
python -m grpc_tools.protoc -I. --python_out=gen \
       --grpc_python_out=gen meshpool.proto                # Python inference workers
```
