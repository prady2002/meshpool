"""End-to-end scenario: app hosting with load-balancing + failover (spec §9.5).

Proves: an app deploys across >=2 nodes, the control-plane load balancer spreads
requests across replicas, and killing a machine mid-traffic reroutes requests to a
surviving replica (no dropped requests) while the manager self-heals back to the
desired replica count on the remaining nodes.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402


def served_by(cp: str, dep_id: str, n: int) -> Counter:
    c: Counter = Counter()
    for _ in range(n):
        try:
            r = httpx.get(f"{cp}/app/{dep_id}/", timeout=5)
            if r.status_code == 200:
                c[r.headers.get("X-MeshPool-Served-By", "?")] += 1
            else:
                c[f"HTTP {r.status_code}"] += 1
        except Exception as exc:  # noqa: BLE001
            c[f"ERR {type(exc).__name__}"] += 1
    return c


def main() -> int:
    h = Harness(n_agents=3, cp_port=8100,
                data_dir=str(Path(__file__).resolve().parent / "_data_host"))
    ok = True
    try:
        print("== starting cluster ==")
        h.start()
        cp = h.cp_url()

        print("\n== test 1: deploy app with 2 replicas ==")
        r = httpx.post(f"{cp}/api/v1/deployments",
                       json={"name": "hello-app", "app": "hello", "replicas": 2}, timeout=20)
        r.raise_for_status()
        dep_id = r.json()["id"]
        time.sleep(2)
        dep = httpx.get(f"{cp}/api/v1/deployments/{dep_id}", timeout=10).json()
        reps = dep["replicas"]
        nodes = {rep["node_name"] for rep in reps if rep["healthy"]}
        print(f"  placed {len(reps)} replicas on nodes: {sorted(nodes)}")
        assert len(reps) >= 2, "expected 2 replicas"
        assert len(nodes) >= 2, "replicas should be on distinct nodes (anti-affinity)"
        print("  PASS: 2 healthy replicas on distinct machines")

        print("\n== test 2: load balancer spreads traffic ==")
        dist = served_by(cp, dep_id, 20)
        print(f"  20 requests served by: {dict(dist)}")
        assert all(k.startswith("sim-node") for k in dist), f"bad responses: {dist}"
        assert len([k for k in dist if k.startswith("sim-node")]) >= 2, "LB did not spread"
        print("  PASS: requests balanced across replicas, all 200 OK")

        print("\n== test 3: kill a hosting node mid-traffic -> failover ==")
        victim = reps[0]["node_name"]
        victim_idx = int(victim.split("-")[-1]) - 1
        print(f"  killing {victim} ...")
        h.stop_agent(victim_idx)
        # immediately hammer the LB; surviving replica must keep serving
        dist = served_by(cp, dep_id, 20)
        good = sum(v for k, v in dist.items() if k.startswith("sim-node"))
        print(f"  during/after kill, 20 requests: {dict(dist)}  ({good} ok)")
        # These 20 requests are issued *after* the kill, so none are in-flight to the
        # dead node. The LB retries a survivor (skipping the failed replica), so every
        # one must be served — "no dropped requests beyond in-flight" (spec §9.5).
        assert good == 20, f"dropped requests during failover: {dist}"

        # give the reconcile loop time to self-heal to desired count on live nodes
        deadline = time.time() + 20
        healed = False
        while time.time() < deadline:
            dep = httpx.get(f"{cp}/api/v1/deployments/{dep_id}", timeout=10).json()
            healthy = [r for r in dep["replicas"] if r["healthy"]]
            live_nodes = {r["node_name"] for r in healthy}
            if len(healthy) >= 2 and victim not in live_nodes:
                healed = True
                break
            time.sleep(1)
        print(f"  after self-heal: {len(healthy)} healthy replicas on {sorted(live_nodes)}")
        assert healed, "manager did not self-heal to desired replica count"
        print("  PASS: no dropped requests + self-healed to 2 replicas on survivors")

        print("\nALL HOSTING E2E CHECKS PASSED")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        h.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
