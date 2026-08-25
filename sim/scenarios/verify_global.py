"""End-to-end scenario: the GLOBAL path — agents reachable ONLY via reverse tunnel.

This is the Phase 1 flagship gate. Every agent runs in *tunnel mode*: it dials out
to the coordinator and advertises an **unreachable sentinel URL** (``http://127.0.0.1:1``)
as its ``node.url``. Nothing can reach an agent at that address, so **any** control
plane → agent operation that still succeeds *must* have travelled down the agent's
outbound reverse tunnel. On that footing it re-runs the three flagship capabilities
end-to-end, each with a node failure:

  1. batch fan-out across every node + kill-a-node → units retried, job completes;
  2. app hosting with load-balancing + kill-a-node → no dropped requests, self-heal;
  3. distributed sharded inference + kill-a-stage → re-partition, *identical* output.

If all three pass while no agent is directly reachable, the global transport carries
batch dispatch, hosted-app proxying, and inference activation hops correctly.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402
from meshpool.inference import model as M  # noqa: E402

SENTINEL = "http://127.0.0.1:1"


# --- small helpers (mirroring the per-engine scenarios) -------------------- #
def _wait_job(cp: str, job_id: str, timeout: float = 90) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = httpx.get(f"{cp}/api/v1/jobs/{job_id}", timeout=10).json()
        if j["status"].startswith("completed"):
            return j
        time.sleep(0.4)
    raise TimeoutError(f"job {job_id} did not finish in {timeout}s")


def _served_by(cp: str, dep_id: str, n: int) -> Counter:
    c: Counter = Counter()
    for _ in range(n):
        try:
            r = httpx.get(f"{cp}/app/{dep_id}/", timeout=5)
            c[r.headers.get("X-MeshPool-Served-By", f"HTTP {r.status_code}")
              if r.status_code == 200 else f"HTTP {r.status_code}"] += 1
        except Exception as exc:  # noqa: BLE001
            c[f"ERR {type(exc).__name__}"] += 1
    return c


def _single_machine_greedy(prompt: str, max_new_tokens: int) -> str:
    cfg = M.REGISTRY["nano-gpt"]
    P, meta = M.load_params(cfg)
    tok = M.CharTokenizer(itos=(meta or {}).get("itos"), vocab_size=cfg.vocab_size)
    ids = tok.encode(prompt) or [tok.encode(" ")[0]]
    out = M.generate_with(lambda ctx: M.forward_full(P, cfg, ctx), cfg, ids,
                          max_new_tokens=max_new_tokens, temperature=0.0)
    return tok.decode(out)


def _node_layer_cap(node: dict, per_layer_mb: int = 1000) -> int:
    cap, pol = node["capability"], node["policy"]
    mem = cap["vram_mb"] if cap.get("gpu") and cap.get("vram_mb") else cap["ram_mb"]
    if pol.get("ram_mb"):
        mem = min(mem, pol["ram_mb"])
    return int((mem * 0.8) // per_layer_mb)


def _wait_nodes(h: Harness, n: int) -> list[dict]:
    return h.wait_nodes_online(n, timeout=30)


def main() -> int:  # noqa: C901 — a deliberately end-to-end gate
    h = Harness(n_agents=3, cp_port=8140, tunnel=True,
                data_dir=str(Path(__file__).resolve().parent / "_data_global"))
    ok = True
    try:
        print("== starting GLOBAL cluster (coordinator + 3 tunnel-only agents) ==")
        h.start()
        cp = h.cp_url()
        nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=10).json()["nodes"]
        print(f"  {len(nodes)} nodes registered via outbound tunnel:")
        for n in nodes:
            print(f"    - {n['name']:12} url={n['url']:18} device={n.get('device_id')}")

        # ---- proof: agents are NOT directly reachable ------------------- #
        print("\n== test 0: agents advertise an unreachable address (tunnel-only) ==")
        assert len(nodes) == 3, f"expected 3 nodes, got {len(nodes)}"
        for n in nodes:
            assert n["url"] == SENTINEL, f"{n['name']} advertised reachable url {n['url']!r}"
            assert n.get("device_id"), f"{n['name']} has no cryptographic device id"
        assert len({n["device_id"] for n in nodes}) == 3, "device ids not distinct"
        # confirm the advertised address really is dead
        try:
            httpx.get(f"{SENTINEL}/health", timeout=2)
            raise AssertionError("sentinel url was unexpectedly reachable")
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
            pass
        print("  PASS: every node advertises a dead url + a unique device id -- so any"
              "\n        successful op below is carried by the reverse tunnel")

        # ---- 1) batch over the tunnel + failover ------------------------ #
        print("\n== test 1: batch fan-out across all nodes (over tunnel) ==")
        payloads = [{"seed": f"s{i}", "rounds": 60000} for i in range(24)]
        jid = httpx.post(f"{cp}/api/v1/jobs", json={"name": "sha_pow", "func": "sha_pow",
                         "payloads": payloads, "cpu_pct": 15}, timeout=10).json()["job_id"]
        job = _wait_job(cp, jid)
        dist = Counter(u["assigned_node"] for u in job["units"])
        used = len([k for k in dist if k])
        assert job["status"] == "completed", job["status"]
        assert used >= 3, f"expected all 3 nodes used, got {used}"
        print(f"  PASS: 24 units completed across {used} nodes -- entirely over the tunnel")

        print("\n== test 2: kill a node mid-batch -> retry over tunnel ==")
        payloads = [{"seconds": 1.5, "echo": i} for i in range(12)]
        jid = httpx.post(f"{cp}/api/v1/jobs", json={"name": "sleep_echo", "func": "sleep_echo",
                         "payloads": payloads, "cpu_pct": 15}, timeout=10).json()["job_id"]
        time.sleep(1.0)
        print("  killing sim-node-2 ...")
        h.stop_agent(1)
        job = _wait_job(cp, jid, timeout=90)
        done = sum(1 for u in job["units"] if u["status"] == "done")
        max_attempts = max(u["attempts"] for u in job["units"])
        assert job["status"] == "completed" and done == len(job["units"]), job["status"]
        assert max_attempts >= 2, "expected a retry after the node was killed"
        print(f"  PASS: all {done} units done after kill (max attempts={max_attempts})")
        print("  restarting sim-node-2 ...")
        h.restart_agent(1)
        _wait_nodes(h, 3)

        # ---- 2) hosting over the tunnel + failover ---------------------- #
        print("\n== test 3: deploy an app, load-balance + failover (over tunnel) ==")
        dep_id = httpx.post(f"{cp}/api/v1/deployments",
                            json={"name": "hello-app", "app": "hello", "replicas": 2},
                            timeout=20).json()["id"]
        time.sleep(2)
        dep = httpx.get(f"{cp}/api/v1/deployments/{dep_id}", timeout=10).json()
        host_nodes = {r["node_name"] for r in dep["replicas"] if r["healthy"]}
        assert len(dep["replicas"]) >= 2 and len(host_nodes) >= 2, "need 2 replicas on 2 nodes"
        spread = _served_by(cp, dep_id, 20)
        assert all(k.startswith("sim-node") for k in spread), f"bad responses: {spread}"
        print(f"  20 requests balanced across replicas (proxied over tunnel): {dict(spread)}")

        victim = dep["replicas"][0]["node_name"]
        victim_idx = int(victim.split("-")[-1]) - 1
        print(f"  killing {victim} mid-traffic ...")
        h.stop_agent(victim_idx)
        after = _served_by(cp, dep_id, 20)
        good = sum(v for k, v in after.items() if k.startswith("sim-node"))
        assert good == 20, f"dropped requests during failover: {after}"
        deadline = time.time() + 25
        healed = False
        while time.time() < deadline:
            dep = httpx.get(f"{cp}/api/v1/deployments/{dep_id}", timeout=10).json()
            healthy = [r for r in dep["replicas"] if r["healthy"]]
            if len(healthy) >= 2 and victim not in {r["node_name"] for r in healthy}:
                healed = True
                break
            time.sleep(1)
        assert healed, "hosting manager did not self-heal onto survivors"
        print("  PASS: no dropped requests + self-healed to 2 replicas (all over tunnel)")
        httpx.delete(f"{cp}/api/v1/deployments/{dep_id}", timeout=10)
        print(f"  restarting {victim} ...")
        h.restart_agent(victim_idx)
        _wait_nodes(h, 3)

        # ---- 3) sharded inference over the tunnel + failover ------------ #
        print("\n== test 4: shard a too-big model, chat, and prove correctness ==")
        cfg = M.REGISTRY["nano-gpt"]
        nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=10).json()["nodes"]
        caps = {n["name"]: _node_layer_cap(n) for n in nodes}
        assert max(caps.values()) < cfg.n_layer, "a single node could hold the whole model"
        assert sum(caps.values()) >= cfg.n_layer, "pool cannot hold the model"
        view = httpx.post(f"{cp}/api/v1/models", json={"model": "nano-gpt"}, timeout=60).json()
        stages = view["stages"]
        assert view["deployed"] and view["status"] == "ready" and len(stages) >= 2, view
        print(f"  sharded into {len(stages)} pipeline stages across the pool (over tunnel)")

        prompt = "MeshPool turns idle machines into "
        oracle = _single_machine_greedy(prompt, 24)
        dist_txt = httpx.post(f"{cp}/api/v1/inference/chat",
                              json={"prompt": prompt, "max_new_tokens": 24, "temperature": 0.0},
                              timeout=60).json()["text"]
        assert dist_txt == oracle, "distributed pipeline diverged from single-machine!"
        print("  PASS: distributed greedy output is bit-for-bit identical to single-machine")

        print("\n== test 5: kill a stage node -> re-partition, identical output ==")
        names = [s["node_name"] for s in stages]
        victim = names[-1]
        victim_idx = int(victim.split("-")[-1]) - 1
        print(f"  killing {victim} (holds a pipeline stage) ...")
        h.stop_agent(victim_idx)
        deadline = time.time() + 35
        repart = None
        while time.time() < deadline:
            v = httpx.get(f"{cp}/api/v1/models", timeout=10).json()
            if v.get("status") == "ready" and victim not in {s["node_name"] for s in v.get("stages", [])}:
                repart = v
                break
            time.sleep(1)
        assert repart is not None, "model did not re-partition onto survivors in time"
        cov = sorted((s["first"], s["last"]) for s in repart["stages"])
        assert cov[0][0] == 0 and cov[-1][1] == cfg.n_layer, "layers not fully covered"
        after_txt = httpx.post(f"{cp}/api/v1/inference/chat",
                               json={"prompt": prompt, "max_new_tokens": 24, "temperature": 0.0},
                               timeout=60).json()["text"]
        assert after_txt == oracle, "output changed after re-partition (should be identical)"
        print("  PASS: survived stage-node loss, re-sharded on survivors, identical output")

        print("\nALL GLOBAL (TUNNEL-ONLY) E2E CHECKS PASSED")
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
