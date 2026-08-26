"""Acceptance gate: the scheduler routes work only to nodes that can actually run it.

Why this gate exists
--------------------
A MeshPool pool is deliberately heterogeneous: the Python agent serves the curated
builtin function registry *and* user ``wasm:`` modules, while ``meshd`` — the single
static Go binary a contributor installs — serves wasm only. It does not reimplement
the curated registry, on purpose: two independent implementations of one function
name is how a pool starts returning different answers for the same input and slashing
honest nodes for it.

Before ``Capability.runtimes`` existed, the scheduler filtered candidates on CPU and
RAM headroom alone. A ``square`` unit could therefore be handed to a Go node, which
answered 501, which counted as a failed attempt. Five such assignments in a row fail
the unit outright — on a pool that was perfectly capable of running it. The more Go
nodes a pool has (i.e. the more successful the single-binary install is), the more
likely a builtin job fails. That is a scaling-inverted reliability bug.

What is proven here
-------------------
  1. A real ``meshd`` advertises ``runtimes: ["wasm"]`` and a *real* RAM figure.
  2. A builtin job on a mixed pool never places a single unit on the wasm-only node,
     and completes with correct results.
  3. POSITIVE CONTROL: with the node's advertisement scrubbed back to the pre-fix
     "runs anything", the very same scheduler *does* select it — so test 2 is a real
     constraint and not an accident of the pool's shape.
  4. The wasm-only node is not blacklisted: a wasm job does land on it and returns
     the right answer, so the filter narrows placement rather than disabling a node.
  5. A pool that cannot serve a runtime at all fails the job with a clear reason
     instead of hanging PENDING forever.

Skips (exit 0 with a printed reason) when the Go toolchain is absent.
Exit 0 = PASS.  Exit 1 = FAIL.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402
from sim.scenarios.verify_user_wasm import _triple_module  # noqa: E402
from meshpool.batch import engine as batch_engine  # noqa: E402
from meshpool.batch.engine import node_supports_runtime, required_runtime  # noqa: E402
from meshpool.common.models import (Capability, DonationPolicy, Node,  # noqa: E402
                                    NodeLoad, NodeStatus)

REPO_ROOT = Path(__file__).resolve().parents[2]
GO_DIR = REPO_ROOT / "agent-go"
CP_PORT = 8152
MESHD_PORT = CP_PORT + 50
MESHD_NAME = "meshd-wasm-only"


def _wait_http(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code < 500:
                return
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(0.3)
    raise TimeoutError(f"not ready at {url}: {last}")


def _wait_job(cp: str, job_id: str, timeout: float = 90) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = httpx.get(f"{cp}/api/v1/jobs/{job_id}", timeout=10).json()
        if j["status"].startswith("completed"):
            return j
        time.sleep(0.4)
    raise TimeoutError(f"job {job_id} never finished: "
                       f"{httpx.get(f'{cp}/api/v1/jobs/{job_id}', timeout=10).json()['status']}")


def _mknode(nid: str, runtimes: list[str] | None) -> Node:
    cap = Capability(cpu_cores=8, ram_mb=8192, flops_estimate=50)
    if runtimes is not None:
        cap.runtimes = runtimes
    return Node(id=nid, name=nid, url=f"http://{nid}", owner="tester",
                capability=cap,
                policy=DonationPolicy(cpu_pct=100, ram_mb=8192),
                load=NodeLoad(), status=NodeStatus.ONLINE)


class _StubRegistry:
    def __init__(self, nodes):
        self._nodes = nodes

    def schedulable_nodes(self):
        return list(self._nodes)


def _pick_with(nodes, runtime):
    """Run the *real* scheduler selection against a synthetic node snapshot."""
    mgr = object.__new__(batch_engine.BatchManager)
    mgr.registry = _StubRegistry(nodes)
    mgr._inflight = {}
    mgr.store = None
    mgr.trust = type("T", (), {"tier": 0})()
    return batch_engine.BatchManager._pick_node(
        mgr, cpu_pct=10, ram_mb=256, exclude=set(), runtime=runtime)


def main() -> int:  # noqa: C901
    go = shutil.which("go")
    if go is None:
        print("SKIP: no Go toolchain on PATH — cannot build meshd.")
        print("      (This gate is skipped, not passed. Install Go to run it.)")
        return 0

    data_root = Path(__file__).resolve().parent / "_data_runtime_routing"
    shutil.rmtree(data_root, ignore_errors=True)
    meshd_data = data_root / "meshd"
    meshd_data.mkdir(parents=True, exist_ok=True)

    print("== building meshd ==")
    exe = data_root / ("meshd.exe" if os.name == "nt" else "meshd")
    build = subprocess.run([go, "build", "-o", str(exe), "./cmd/meshd"],
                           cwd=str(GO_DIR), capture_output=True, text=True, errors="replace")
    if build.returncode != 0:
        print("FAILED: meshd did not build")
        print(build.stderr[-2000:])
        return 1
    print(f"  built {exe.name}")

    h = Harness(n_agents=2, cp_port=CP_PORT, data_dir=str(data_root / "cluster"))
    meshd_proc = None
    ok = True
    try:
        print("\n== starting control plane + 2 Python agents + 1 meshd ==")
        h.start()
        cp = h.cp_url()

        env = dict(os.environ)
        env.update({
            "MESHPOOL_CONTROL_PLANE_URL": cp,
            "MESHPOOL_AGENT_HOST": "127.0.0.1",
            "MESHPOOL_AGENT_BIND": "127.0.0.1",
            "MESHPOOL_AGENT_PORT": str(MESHD_PORT),
            "MESHPOOL_NAME": MESHD_NAME,
            "MESHPOOL_DATA_DIR": str(meshd_data),
            "MESHPOOL_SIM_CORES": "8",
        })
        env.pop("MESHPOOL_SIM_RAM_MB", None)   # force the real RAM probe
        meshd_proc = subprocess.Popen([str(exe)], env=env, cwd=str(REPO_ROOT))
        _wait_http(f"http://127.0.0.1:{MESHD_PORT}/health")

        deadline = time.time() + 30
        go_node = None
        while time.time() < deadline:
            nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=5).json()["nodes"]
            go_node = next((n for n in nodes if n["name"] == MESHD_NAME), None)
            if go_node and go_node["status"] == "online":
                break
            time.sleep(0.5)
        assert go_node is not None, "meshd never appeared in the registry"
        nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=5).json()["nodes"]
        py_nodes = [n for n in nodes if n["name"] != MESHD_NAME]
        go_id = go_node["id"]
        print(f"  {len(nodes)} nodes: {len(py_nodes)} python + 1 meshd ({go_id[:12]})")

        # -- 1. the advertisement itself ---------------------------------- #
        print("\n== test 1: meshd advertises what it can actually serve ==")
        adv = go_node["capability"].get("runtimes")
        print(f"  meshd  capability.runtimes = {adv}")
        for n in py_nodes:
            print(f"  python capability.runtimes = {n['capability'].get('runtimes')}")
        assert adv == ["wasm"], (
            f"meshd advertised {adv!r}; it must claim wasm only, because it has no "
            f"implementation of the curated builtin registry")
        for n in py_nodes:
            r = n["capability"].get("runtimes")
            assert r and "builtin" in r and "wasm" in r, (
                f"the Python agent must still advertise both runtimes, got {r!r}")
        ram = go_node["capability"].get("ram_mb", 0)
        print(f"  meshd  capability.ram_mb    = {ram}")
        assert ram > 256, (
            f"meshd advertised ram_mb={ram} — a node reporting no memory is scored "
            f"at zero usable layer memory and is silently excluded from every model "
            f"deployment")
        bw = (go_node.get("link") or {}).get("bandwidth_mbps", 0)
        lat = (go_node.get("link") or {}).get("latency_ms", -1)
        print(f"  meshd  link.bandwidth_mbps  = {bw:.1f}")
        print(f"  meshd  link.latency_ms      = {lat:.2f}")
        assert bw > 0, (
            "meshd registered with bandwidth_mbps=0 — it used to hardcode that, so "
            "every Go node looked like a dial-up link to the placement algorithm")
        assert lat >= 0, f"meshd reported nonsensical latency {lat}"
        print("  PASS: honest, narrow capability advertisement")

        # a real free slot on the Go node, or test 2 is vacuous
        slots = int(go_node["policy"]["cpu_pct"] / 100 * go_node["capability"]["cpu_cores"])
        assert slots > 0, (
            f"meshd has {slots} donatable slots — 'no builtin units landed on it' "
            f"would be trivially true")
        print(f"  (non-vacuity: meshd is online with ~{slots} free slots)")

        # -- 2. a builtin job never lands on the wasm-only node ------------ #
        print("\n== test 2: 40 builtin units on a mixed pool ==")
        payloads = [{"x": i} for i in range(40)]
        r = httpx.post(f"{cp}/api/v1/jobs",
                       json={"name": "square", "func": "square",
                             "payloads": payloads, "cpu_pct": 10}, timeout=10)
        r.raise_for_status()
        job = _wait_job(cp, r.json()["job_id"])
        dist = Counter(u["assigned_node"] for u in job["units"])
        for nid, cnt in dist.items():
            label = MESHD_NAME if nid == go_id else "python"
            print(f"    {label:16} {nid[:12]} ran {cnt} units")
        on_go = dist.get(go_id, 0)
        assert on_go == 0, (
            f"{on_go} builtin units were assigned to the wasm-only node; each one "
            f"burns an attempt against MAX_ATTEMPTS={batch_engine.MAX_ATTEMPTS}")
        assert job["status"] == "completed", f"job status {job['status']}"
        bad = [u for u in job["units"] if u["status"] != "done"]
        assert not bad, f"{len(bad)} units did not complete: {bad[:2]}"
        results = {u["payload"]["x"]: u["result"] for u in job["units"]}
        wrong = {k: v for k, v in results.items() if v != k * k}
        assert not wrong, f"wrong results: {list(wrong.items())[:3]}"
        assert sum(dist.values()) == 40
        print("  PASS: 0 units on the wasm-only node, 40/40 correct")

        # -- 3. positive control: the pre-fix advertisement DOES select it -- #
        print("\n== test 3: positive control — scrub the advertisement, re-run the "
              "same scheduler ==")
        wasm_only = _mknode("go-like", ["wasm"])
        unrestricted = _mknode("go-like-prefix", None)      # pre-fix default
        picked_now = _pick_with([wasm_only], "builtin")
        picked_prefix = _pick_with([unrestricted], "builtin")
        print(f"  runtimes=['wasm']          -> _pick_node = {picked_now}")
        print(f"  runtimes absent (pre-fix)  -> _pick_node = "
              f"{picked_prefix.id if picked_prefix else None}")
        assert picked_now is None, (
            "the scheduler still selected a wasm-only node for builtin work")
        assert picked_prefix is not None, (
            "the positive control did not select anything — the assertion above "
            "proves nothing, because the node was unschedulable for some other "
            "reason (capacity, status) rather than by runtime")
        assert _pick_with([wasm_only], "wasm") is not None, (
            "a wasm-only node must still be selected for wasm work")
        assert required_runtime("square") == "builtin"
        assert required_runtime("wasm:" + "0" * 64) == "wasm"
        assert node_supports_runtime(wasm_only, "wasm")
        assert not node_supports_runtime(wasm_only, "builtin")
        print("  PASS: the filter is what excluded it, and only for builtin work")

        # -- 4. the wasm-only node is not blacklisted --------------------- #
        print("\n== test 4: a wasm job does land on the wasm-only node ==")
        module = _triple_module()
        up = httpx.post(f"{cp}/api/v1/functions", content=module,
                        headers={"Content-Type": "application/octet-stream"},
                        timeout=60)
        assert up.status_code == 200, (
            f"function upload failed: {up.status_code} {up.text[:300]}")
        func = up.json()["function"]
        print(f"  uploaded {func[:20]}…")
        payloads = [{"x": i} for i in range(1, 25)]
        r = httpx.post(f"{cp}/api/v1/jobs",
                       json={"name": "triple", "func": func,
                             "payloads": payloads, "cpu_pct": 10, "ram_mb": 128},
                       timeout=10)
        r.raise_for_status()
        job = _wait_job(cp, r.json()["job_id"], timeout=180)
        dist = Counter(u["assigned_node"] for u in job["units"])
        for nid, cnt in dist.items():
            label = MESHD_NAME if nid == go_id else "python"
            print(f"    {label:16} {nid[:12]} ran {cnt} units")
        assert job["status"] == "completed", (
            f"wasm job status {job['status']}; errors="
            f"{[u.get('error') for u in job['units'] if u.get('error')][:3]}")
        assert dist.get(go_id, 0) > 0, (
            "no wasm units landed on the wasm-only node — the runtime filter has "
            "turned a capable node into dead weight")
        wrong = [u for u in job["units"]
                 if u.get("result") != {"tripled": u["payload"]["x"] * 3}]
        assert not wrong, f"wrong wasm results: {[w.get('result') for w in wrong][:3]}"
        print(f"  PASS: {dist.get(go_id, 0)}/24 units ran on meshd, 24/24 correct")

        # -- 5. an incapable pool fails loudly instead of hanging ---------- #
        print("\n== test 5: a pool with no capable node fails the job with a reason ==")
        ok5 = _incapable_pool_fails_fast()
        if not ok5:
            ok = False
        else:
            print("  PASS: job failed with an explicit reason, did not hang")

    except AssertionError as exc:
        print(f"\nFAILED: {exc}")
        ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        ok = False
    finally:
        if meshd_proc is not None:
            try:
                meshd_proc.terminate()
                meshd_proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                meshd_proc.kill()
        h.stop()

    print("\n" + ("PASS: work is only ever scheduled onto nodes that can run it"
                  if ok else "FAIL"))
    return 0 if ok else 1


def _incapable_pool_fails_fast() -> bool:
    """Drive the real BatchManager against a pool of wasm-only nodes."""
    from meshpool.common.db import Store  # noqa: PLC0415
    from meshpool.common.trust import ContributionLedger, TrustEngine  # noqa: PLC0415
    from meshpool.controlplane.registry import EventBus  # noqa: PLC0415

    tmp = Path(__file__).resolve().parent / "_data_runtime_routing" / "incapable.db"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    store = Store(str(tmp))
    trust = TrustEngine(store, tier=0)
    mgr = batch_engine.BatchManager(
        registry=_StubRegistry([_mknode("go-a", ["wasm"]), _mknode("go-b", ["wasm"])]),
        store=store, ledger=ContributionLedger(store), trust=trust,
        events=EventBus(), transport=_ExplodingTransport())
    original = batch_engine.NO_RUNTIME_GRACE_MS
    batch_engine.NO_RUNTIME_GRACE_MS = 0
    try:
        job = mgr.submit("square", "square", [{"x": 1}, {"x": 2}], cpu_pct=10,
                         ram_mb=256)
        asyncio.run(mgr.dispatch_once())
        job = mgr.get(job.id)
        statuses = {u.status.value for u in job.units}
        errors = {u.error for u in job.units}
        print(f"  unit statuses={statuses} errors={errors}")
        if statuses != {"failed"}:
            print(f"  FAILED: units are {statuses}, not failed — the job would hang "
                  f"PENDING forever on a pool that can never serve it")
            return False
        if not all(e and "runtime" in e for e in errors):
            print(f"  FAILED: no actionable error message: {errors}")
            return False
        if not job.status.startswith("completed"):
            print(f"  FAILED: job status {job.status!r} is not terminal")
            return False
        # non-vacuity: the same pool must still accept work it *can* run
        batch_engine.NO_RUNTIME_GRACE_MS = 0
        wasm_job = mgr.submit("w", "wasm:" + "0" * 64, [{"n": 1}], cpu_pct=10,
                              ram_mb=256)
        asyncio.run(mgr.dispatch_once())
        wasm_job = mgr.get(wasm_job.id)
        if all(u.status.value == "failed" and "runtime" in (u.error or "")
               for u in wasm_job.units):
            print("  FAILED: the same pool also refused wasm work — the check is "
                  "rejecting everything, not just what it cannot serve")
            return False
        return True
    finally:
        batch_engine.NO_RUNTIME_GRACE_MS = original
        store.close()


class _ExplodingTransport:
    """Any dispatch is a bug in test 5 — nothing should ever be assigned."""

    async def call(self, *a, **k):
        raise AssertionError("a unit was dispatched to an incapable node")


if __name__ == "__main__":
    raise SystemExit(main())
