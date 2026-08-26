"""Performance report: boot one cluster, measure REAL numbers for all three engines,
and write ``docs/performance.md`` (spec §9 — "report real measured numbers").

This never invents targets. Every figure printed and written comes from a live
multi-process cluster on this machine. Numbers are modest by design (CPU + localhost,
tiny demo model) — the point is that the *capabilities* work and are measurable; the
adapter seam is where real GPUs/Petals/exo raise the absolute throughput.

Run:  python -m sim.scenarios.perf_report
"""
from __future__ import annotations

import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from sim.harness import Harness  # noqa: E402
from meshpool.inference import model as M  # noqa: E402



_CLIENT = None


def _http():
    """One reused client for the whole report.

    ``httpx.get``/``httpx.post`` build a fresh ``httpx.Client`` per call, and building
    one loads the system TLS trust store: ~0.67 s per call on this machine, versus
    0.002 s through a reused client. This file *publishes* numbers, and two of them are
    measured by polling in a loop -- ``healed_s`` and ``repart_s``. Every poll paid the
    construction cost, so both figures were inflated by roughly 0.7 s per poll
    iteration: the report was timing certificate parsing and attributing it to the
    mesh's recovery speed. Build the client once, before any stopwatch starts.
    """
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.Client(follow_redirects=True)
    return _CLIENT


def _nodes(cp: str) -> list[dict]:
    return _http().get(f"{cp}/api/v1/nodes", timeout=10).json()["nodes"]


def _wait_job(cp: str, jid: str, timeout: float = 90) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        j = _http().get(f"{cp}/api/v1/jobs/{jid}", timeout=10).json()
        if j["status"].startswith("completed"):
            return j
        time.sleep(0.3)
    return _http().get(f"{cp}/api/v1/jobs/{jid}", timeout=10).json()


def _online(cp: str) -> int:
    return sum(1 for n in _nodes(cp) if n.get("status") == "online")


def _heal_cluster(cp: str, idx: int, target: int = 3, timeout: float = 30) -> None:
    """Restart a killed agent and wait until the cluster is back to ``target`` online."""
    h_ref.start_agent(idx)
    end = time.time() + timeout
    while time.time() < end:
        if _online(cp) >= target:
            return
        time.sleep(0.5)


def _fire(cp: str, dep_id: str, n: int) -> Counter:
    c: Counter = Counter()
    for _ in range(n):
        try:
            r = _http().get(f"{cp}/app/{dep_id}/", timeout=5)
            key = "ok" if r.status_code == 200 else f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001
            key = f"ERR {type(exc).__name__}"
        c[key] += 1
    return c


def measure(cp: str, metrics: dict) -> None:
    cfg = M.REGISTRY["nano-gpt"]
    nodes = _nodes(cp)
    metrics["nodes"] = [
        {"name": n["name"], "cores": n["capability"]["cpu_cores"],
         "ram_mb": n["capability"]["ram_mb"], "cpu_pct": n["policy"]["cpu_pct"]}
        for n in nodes]
    name_by_id = {n["id"]: n["name"] for n in nodes}

    # ---- batch ---------------------------------------------------------- #
    print("== batch: distribute 30 sha_pow units ==")
    payloads = [{"seed": f"s{i}", "rounds": 60000} for i in range(30)]
    t0 = time.perf_counter()
    jid = _http().post(f"{cp}/api/v1/jobs", json={"name": "sha_pow", "func": "sha_pow",
                     "payloads": payloads, "cpu_pct": 15}, timeout=10).json()["job_id"]
    job = _wait_job(cp, jid)
    dt = time.perf_counter() - t0
    dist = Counter(name_by_id.get(u["assigned_node"], "?") for u in job["units"])
    metrics["batch"] = {
        "units": len(job["units"]), "wall_s": round(dt, 2),
        "throughput_units_per_s": round(len(job["units"]) / dt, 2),
        "nodes_used": len([k for k in dist if k != "?"]),
        "distribution": dict(dist), "status": job["status"]}
    print(f"   {len(job['units'])} units in {dt:.2f}s across {dict(dist)}")

    # ---- hosting -------------------------------------------------------- #
    print("== hosting: deploy 2 replicas, measure LB + failover ==")
    dep_id = _http().post(f"{cp}/api/v1/deployments",
                        json={"name": "perf-app", "app": "hello", "replicas": 2},
                        timeout=20).json()["id"]
    time.sleep(2)
    before = _fire(cp, dep_id, 40)
    dep = _http().get(f"{cp}/api/v1/deployments/{dep_id}", timeout=10).json()
    victim_name = dep["replicas"][0]["node_name"]
    victim_idx = int(victim_name.split("-")[-1]) - 1   # sim-node-N -> harness index N-1
    print(f"   killing {victim_name} mid-traffic ...")
    t0 = time.perf_counter()
    h_ref.stop_agent(victim_idx)
    during = _fire(cp, dep_id, 40)          # requests spanning the failure
    # wait for self-heal back to desired replica count
    healed_s = None
    end = time.time() + 30
    while time.time() < end:
        d = _http().get(f"{cp}/api/v1/deployments/{dep_id}", timeout=10).json()
        if sum(1 for r in d["replicas"] if r["healthy"]) >= 2:
            healed_s = round(time.perf_counter() - t0, 2)
            break
        time.sleep(0.5)
    after = _fire(cp, dep_id, 40)
    total = before + during + after
    metrics["hosting"] = {
        "requests": sum(total.values()),
        "ok": total.get("ok", 0),
        "success_pct": round(100 * total.get("ok", 0) / max(1, sum(total.values())), 1),
        "during_failover": dict(during),
        "self_heal_s": healed_s}
    print(f"   {total.get('ok',0)}/{sum(total.values())} ok; self-heal {healed_s}s")

    # heal the cluster back to full strength so inference has the whole pool
    print(f"   restarting {victim_name} for the inference phase ...")
    _heal_cluster(cp, victim_idx, target=3)

    # ---- inference ------------------------------------------------------ #
    print("== inference: shard nano-gpt, measure tok/s + failover ==")
    view = _http().post(f"{cp}/api/v1/models", json={"model": "nano-gpt"}, timeout=60).json()
    if "stages" not in view:
        raise RuntimeError(f"model deploy did not return a partition: {view}")
    stages = view["stages"]
    prompt = "MeshPool turns idle machines into "
    chat = _http().post(f"{cp}/api/v1/inference/chat",
                      json={"prompt": prompt, "max_new_tokens": 24, "temperature": 0.8},
                      timeout=60).json()
    # correctness oracle (temperature 0, distributed vs single-machine)
    P, meta = M.load_params(cfg)
    tok = M.CharTokenizer(itos=(meta or {}).get("itos"), vocab_size=cfg.vocab_size)
    ids = tok.encode(prompt) or [tok.encode(" ")[0]]
    single = tok.decode(M.generate_with(lambda c: M.forward_full(P, cfg, c), cfg, ids,
                                        max_new_tokens=24, temperature=0.0))
    dist_greedy = _http().post(f"{cp}/api/v1/inference/chat",
                             json={"prompt": prompt, "max_new_tokens": 24, "temperature": 0.0},
                             timeout=60).json()["text"]
    # Failover: kill a stage-holding node, time the recovery, re-measure.
    #
    # WHICH node is killed is not arbitrary, and getting it wrong makes this metric
    # unmeasurable rather than bad. The sim profiles are deliberately heterogeneous and
    # a node's layer capacity is `donated_ram * 0.8 // per_layer_mb`, so for nano-gpt's
    # 8 layers the three nodes cap at 6 / 3 / 1 layers. Killing the *last stage's* node
    # picked whichever node the placement happened to put last -- and when that was the
    # 3-layer node the survivors could hold only 7 of 8 layers, so the control plane
    # correctly refused to re-partition and logged "insufficient survivor capacity".
    # The report then crashed trying to read ["text"] off the error body.
    #
    # That was a flaw in the scenario, not in the mesh: it demanded recovery from a loss
    # that is genuinely unrecoverable and called the honest refusal a failure. Kill the
    # node holding the *fewest* layers instead -- assignment is proportional to
    # capacity, so that is the least capacity-critical stage, and it is the same
    # reasoning verify_inference.py uses when it deliberately kills sim-node-3.
    victim = min(stages, key=lambda s: (s["last"] - s["first"], s["first"]))
    last_node = victim["node_name"]
    vidx = int(last_node.split("-")[-1]) - 1   # sim-node-N -> harness index N-1
    print(f"   killing {last_node} "
          f"(holds layers {victim['first']}:{victim['last']}, the smallest stage) ...")
    t0 = time.perf_counter()
    h_ref.stop_agent(vidx)
    repart_s, after_txt = None, None
    # Time the recovery by what a user actually experiences -- the mesh answering
    # correctly again -- not by a status field flipping to "ready".
    #
    # This loop used to poll the status flag, break, and then immediately issue one
    # chat and index ["text"]. That was only ever safe by accident: each poll built a
    # fresh httpx.Client and so spent ~0.7 s loading the system TLS trust store, which
    # padded the wait long enough for the re-partitioned stage to actually become
    # serveable. Removing that cost made the loop exit promptly and the very next chat
    # returned an error body, blowing up on KeyError. The flag is a proxy, and the
    # proxy leads the reality it stands for.
    #
    # So poll on the real thing. A failed attempt is retried rather than fatal, and the
    # cost of any settling time lands honestly inside repart_s instead of being hidden.
    end = time.time() + 60
    last_status: str = "unknown"
    while time.time() < end:
        v = _http().get(f"{cp}/api/v1/models", timeout=10).json()
        names = {s["node_name"] for s in v.get("stages", [])}
        last_status = str(v.get("status"))
        if last_status == "ready" and last_node not in names:
            try:
                body = _http().post(f"{cp}/api/v1/inference/chat",
                                    json={"prompt": prompt, "max_new_tokens": 24,
                                          "temperature": 0.0},
                                    timeout=60).json()
            except Exception:  # noqa: BLE001  (transport hiccup mid-repartition)
                body = {}
            if isinstance(body, dict) and body.get("text"):
                after_txt = body["text"]
                repart_s = round(time.perf_counter() - t0, 2)
                break
        time.sleep(0.25)
    if after_txt is None:
        raise RuntimeError(
            f"the mesh never served a reply again after losing {last_node} "
            f"(waited 60s; last deployment status={last_status!r}). {last_node} held "
            f"the smallest stage, so the survivors had capacity and recovery was "
            f"genuinely expected here. A re-partition that reports ready without being "
            f"able to serve is worse than one that fails.")
    metrics["inference"] = {
        "layers": view["n_layers"], "stages": len(stages),
        "layout": [f"{s['node_name']}[{s['first']}:{s['last']}]" for s in stages],
        "tokens_per_sec": chat["tokens_per_sec"], "tokens": chat["tokens"],
        "oracle_identical": bool(dist_greedy == single),
        "repartition_s": repart_s,
        "victim": f"{last_node} (layers {victim['first']}:{victim['last']})",
        "identical_after_failover": bool(after_txt == single),
        "sample": chat["text"]}
    print(f"   {chat['tokens_per_sec']} tok/s; oracle={dist_greedy == single}; "
          f"repartition {repart_s}s; identical-after-failover={after_txt == single}")


def render_md(metrics: dict) -> str:
    b, ho, inf = metrics["batch"], metrics["hosting"], metrics["inference"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Performance report (real measured numbers)",
        "",
        f"_Generated {now} by `python -m sim.scenarios.perf_report` on a live "
        "3-node multi-process cluster._",
        "",
        "> These are honest, reproducible figures from **one machine** (localhost "
        "networking, CPU-only, a tiny from-scratch demo model). They demonstrate that "
        "each capability works and is measurable — not datacenter throughput. The "
        "adapter seam (`docs/inference.md` §6) is where real GPUs/Petals/exo raise the "
        "absolute numbers; the control-plane behaviours below are unchanged.",
        "",
        "## Environment",
        "",
        f"- Host: {platform.system()} {platform.release()}, Python {platform.python_version()}",
        f"- Cluster: 1 control plane + {len(metrics['nodes'])} simulated heterogeneous agents",
        "",
        "| node | cores | ram (MB) | donate cpu% |",
        "|---|---|---|---|",
    ]
    for n in metrics["nodes"]:
        lines.append(f"| {n['name']} | {n['cores']} | {n['ram_mb']} | {n['cpu_pct']} |")
    lines += [
        "",
        "## Batch engine (criterion d)",
        "",
        f"- **{b['units']} units** completed in **{b['wall_s']} s** "
        f"(**{b['throughput_units_per_s']} units/s**), status `{b['status']}`.",
        f"- Distributed across **{b['nodes_used']} nodes**: `{b['distribution']}`.",
        "",
        "## Hosting engine (criterion c)",
        "",
        f"- **{ho['ok']}/{ho['requests']} requests succeeded "
        f"({ho['success_pct']}%)** across a deliberate mid-traffic node kill.",
        f"- Requests spanning the failure: `{ho['during_failover']}`.",
        f"- **Self-heal to desired replica count: {ho['self_heal_s']} s.**",
        "",
        "## Inference engine (criterion e) — flagship",
        "",
        f"- Model **{inf['layers']} layers** sharded into **{inf['stages']} pipeline "
        f"stages**: `{', '.join(inf['layout'])}` (no single node held them all).",
        f"- **{inf['tokens_per_sec']} tokens/s** generating {inf['tokens']} tokens.",
        f"- **Correctness oracle** (distributed == single-machine, greedy): "
        f"**{'IDENTICAL ✅' if inf['oracle_identical'] else 'MISMATCH ❌'}**.",
        f"- Killed the node holding {inf['victim']} → **served correctly again after "
        f"{inf['repartition_s']} s**; "
        f"output still **{'identical ✅' if inf['identical_after_failover'] else 'changed ❌'}**.",
        f"- Sample: `{inf['sample']!r}`",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m sim.scenarios.perf_report        # regenerates this file",
        "python -m sim.scenarios.verify_batch       # asserts the batch invariants",
        "python -m sim.scenarios.verify_hosting     # asserts hosting failover",
        "python -m sim.scenarios.verify_inference   # asserts the inference oracle",
        "```",
        "",
    ]
    return "\n".join(lines)


h_ref: Harness  # set in main so the measure() helpers can kill nodes


def main() -> int:
    global h_ref
    h_ref = Harness(n_agents=3, cp_port=8130,
                    data_dir=str(Path(__file__).resolve().parent / "_data_perf"))
    ok = True
    metrics: dict = {}
    try:
        print("== starting cluster (control plane + 3 heterogeneous agents) ==")
        h_ref.start()
        measure(h_ref.cp_url(), metrics)
        out = REPO / "docs" / "performance.md"
        out.write_text(render_md(metrics), encoding="utf-8")
        print(f"\n== wrote {out} ==")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        h_ref.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
