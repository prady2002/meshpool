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
    # Report what the job was actually doing. Polling only for a "completed"
    # prefix means a job stalled with units outstanding, a job whose units all
    # failed, and a job that was never dispatched at all all produce the SAME
    # message -- so the detail that would identify the cause is precisely the
    # one this helper throws away. Fetch it once more on the way out.
    try:
        _last = httpx.get(f"{cp}/api/v1/jobs/{job_id}", timeout=10).json()
        _detail = f" (last seen: {_last})"[:600]
    except Exception as _exc:  # noqa: BLE001 - diagnostics must not mask the timeout
        _detail = f" (control plane unreachable while diagnosing: {_exc})"
    raise TimeoutError(f"job {job_id} did not finish in {timeout}s{_detail}")


def _wait_unit_in_flight(cp: str, job_id: str, node_id: str,
                         timeout: float = 30.0) -> int:
    """Block until ``node_id`` actually holds unfinished work for ``job_id``.

    The kill below used to be a fixed ``time.sleep(1.0)``, which is a race: it
    assumes the units are still running a second after submission. They often
    are not. With 12 units of 1.5 s spread over 3 nodes the whole job can finish
    in under 4 s, so on a fast or lightly-loaded machine every unit completed
    *before* the kill landed -- the job then finished with ``failed=0`` and
    ``max_attempts=1``, and the gate failed claiming no retry happened.

    That is a false negative: failover was never exercised, so there was nothing
    to retry. Widening the sleep would only move the race. Waiting for observed
    in-flight work makes the kill deterministic -- when it fires, the node
    provably holds units that must be reassigned.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = httpx.get(f"{cp}/api/v1/jobs/{job_id}", timeout=10).json()
        held = [u for u in j["units"]
                if u.get("assigned_node") == node_id
                and u.get("status") in ("assigned", "running")]
        if held:
            return len(held)
        if j["status"].startswith("completed"):
            raise AssertionError(
                "job finished before any unit was observed in flight on "
                f"{node_id}; the workload is too short to exercise failover")
        time.sleep(0.1)
    raise TimeoutError(f"no in-flight unit on {node_id} within {timeout}s")


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
        # Long enough per unit that a node cannot drain its share before the
        # kill; the wait below is what makes the timing deterministic, but a
        # too-short unit would still let the job finish during the poll.
        payloads = [{"seconds": 6.0, "echo": i} for i in range(12)]
        jid = httpx.post(f"{cp}/api/v1/jobs", json={"name": "sleep_echo", "func": "sleep_echo",
                         "payloads": payloads, "cpu_pct": 15}, timeout=10).json()["job_id"]
        victim = next(n["id"] for n in nodes if n["name"] == "sim-node-2")
        held = _wait_unit_in_flight(cp, jid, victim)
        print(f"  killing sim-node-2 while it holds {held} in-flight unit(s) ...")
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
        # Which stage holder we kill decides what the correct answer *is*, so it
        # cannot be left to incidental ordering. These profiles pool ~10 layer
        # slots for an 8-layer model: losing the 16 GB node leaves strictly too
        # little memory, and refusing to place is then correct behaviour, not a
        # bug. Latency-aware ordering decides which node ends up holding the last
        # stage, so `names[-1]` made this gate pass or fail on node ordering
        # rather than on behaviour -- it went red here with "insufficient
        # survivor capacity" while the engine was working exactly as designed.
        #
        # Prefer the last stage, because it owns the LM head and replacing it
        # exercises the is_last hand-off; fall back to any stage holder whose
        # loss the survivors can absorb. Only if no such victim exists do we
        # assert the degrade path instead.
        live = httpx.get(f"{cp}/api/v1/nodes", timeout=10).json()["nodes"]
        cap_by_name = {n["name"]: _node_layer_cap(n) for n in live
                       if n.get("status") == "online"}
        total_cap = sum(cap_by_name.values())
        stage_names = [s["node_name"] for s in stages]
        viable = [nm for nm in stage_names
                  if total_cap - cap_by_name.get(nm, 0) >= cfg.n_layer]
        victim = stage_names[-1] if stage_names[-1] in viable else (
            viable[-1] if viable else stage_names[-1])
        victim_idx = int(victim.split("-")[-1]) - 1
        survivor_cap = total_cap - cap_by_name.get(victim, 0)
        must_repartition = survivor_cap >= cfg.n_layer
        print(f"  killing {victim} (holds a pipeline stage); survivors keep "
              f"{survivor_cap} layer slots for a {cfg.n_layer}-layer model -> "
              f"{'must re-partition' if must_repartition else 'must degrade cleanly'}")
        h.stop_agent(victim_idx)
        deadline = time.time() + 35
        repart = None
        degraded = False
        while time.time() < deadline:
            v = httpx.get(f"{cp}/api/v1/models", timeout=10).json()
            if v.get("status") == "ready" and victim not in {s["node_name"] for s in v.get("stages", [])}:
                repart = v
                break
            if v.get("status") == "degraded":
                degraded = True
                break
            time.sleep(1)

        if not must_repartition:
            # The honest complement: when the pool really cannot host the model,
            # the deployment must say so rather than keep advertising a pipeline
            # whose stages are gone.
            assert degraded, (
                f"survivors hold only {survivor_cap} of {cfg.n_layer} layer slots, "
                f"so the model cannot be placed -- it must go degraded, not stay ready")
            print("  PASS: survivors genuinely could not host the model; deployment "
                  "degraded cleanly instead of advertising a dead pipeline")
        else:
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
