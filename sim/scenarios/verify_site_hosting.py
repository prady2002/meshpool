"""End-to-end scenario: host YOUR OWN static website across the mesh.

Proves the "host an app / website" capability end to end: a user-supplied static
site (index.html + a nested asset) is shipped inside the deploy request, written
into each replica's sandbox scratch dir, and served through the control-plane
load balancer at ``/app/{id}/`` — with the correct content and content-types,
a real 404 for missing paths, and (crucially) it keeps serving after a hosting
node is killed, because the durable deployment re-ships the site to a survivor
when the manager self-heals.
"""
from __future__ import annotations

import base64
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402

TOKEN = "MESHPOOL-SITE-7Q2X"          # unique marker proving OUR html is served
INDEX = f"<!doctype html><html><body><h1>{TOKEN}</h1>" \
        "<link rel=stylesheet href=style.css></body></html>"
CSS = "h1{color:#0f0}"


def b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def served(cp: str, dep_id: str, n: int) -> Counter:
    c: Counter = Counter()
    for _ in range(n):
        try:
            r = httpx.get(f"{cp}/app/{dep_id}/", timeout=5)
            if r.status_code == 200 and TOKEN in r.text:
                c[r.headers.get("X-MeshPool-Served-By", "?")] += 1
            else:
                c[f"HTTP {r.status_code}"] += 1
        except Exception as exc:  # noqa: BLE001
            c[f"ERR {type(exc).__name__}"] += 1
    return c


def main() -> int:
    h = Harness(n_agents=3, cp_port=8103,
                data_dir=str(Path(__file__).resolve().parent / "_data_site"))
    ok = True
    try:
        print("== starting cluster ==")
        h.start()
        cp = h.cp_url()
        site = {"index.html": b64(INDEX), "style.css": b64(CSS)}

        print("\n== test 1: deploy a custom static site (2 replicas) ==")
        r = httpx.post(f"{cp}/api/v1/deployments",
                       json={"name": "my-site", "app": "site", "replicas": 2, "site": site},
                       timeout=20)
        r.raise_for_status()
        dep_id = r.json()["id"]
        time.sleep(2)
        dep = httpx.get(f"{cp}/api/v1/deployments/{dep_id}", timeout=10).json()
        reps = dep["replicas"]
        nodes = {rep["node_name"] for rep in reps if rep["healthy"]}
        assert len(reps) >= 2, f"expected 2 replicas, got {len(reps)}"
        assert len(nodes) >= 2, "replicas should be on distinct nodes (anti-affinity)"
        print(f"  PASS: {len(reps)} replicas on {sorted(nodes)}")

        print("\n== test 2: the mesh serves OUR html + nested asset (correct types) ==")
        root = httpx.get(f"{cp}/app/{dep_id}/", timeout=10)
        assert root.status_code == 200 and TOKEN in root.text, "custom index.html not served"
        assert "text/html" in root.headers.get("content-type", ""), "wrong index content-type"
        css = httpx.get(f"{cp}/app/{dep_id}/style.css", timeout=10)
        assert css.status_code == 200 and "color" in css.text, "nested asset not served"
        assert "css" in css.headers.get("content-type", ""), "wrong css content-type"
        missing = httpx.get(f"{cp}/app/{dep_id}/does-not-exist.html", timeout=10)
        assert missing.status_code == 404, f"expected 404 for missing path, got {missing.status_code}"
        print("  PASS: index + style.css served with right types; missing path -> 404")

        print("\n== test 3: kill a hosting node -> site still served (failover + re-ship) ==")
        victim = reps[0]["node_name"]
        victim_idx = int(victim.split("-")[-1]) - 1
        print(f"  killing {victim} ...")
        h.stop_agent(victim_idx)
        dist = served(cp, dep_id, 20)
        good = sum(v for k, v in dist.items() if k.startswith("sim-node"))
        print(f"  20 requests after kill: {dict(dist)}  ({good} served OUR html)")
        assert good == 20, f"custom site dropped requests during failover: {dist}"

        # self-heal must re-place a replica on a survivor AND re-ship the site to it
        deadline = time.time() + 25
        healed = False
        while time.time() < deadline:
            dep = httpx.get(f"{cp}/api/v1/deployments/{dep_id}", timeout=10).json()
            healthy = [r for r in dep["replicas"] if r["healthy"]]
            live_nodes = {r["node_name"] for r in healthy}
            if len(healthy) >= 2 and victim not in live_nodes:
                healed = True
                break
            time.sleep(1)
        assert healed, "manager did not self-heal the custom-site deployment"
        final = served(cp, dep_id, 10)
        good2 = sum(v for k, v in final.items() if k.startswith("sim-node"))
        assert good2 == 10, f"site not served after self-heal/re-ship: {final}"
        print(f"  PASS: self-healed to {len(healthy)} replicas on {sorted(live_nodes)}; "
              f"site re-shipped + served")

        print("\nALL SITE HOSTING E2E CHECKS PASSED")
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
