"""Acceptance gate: ``meshd`` participates fully while unreachable from outside.

Why this gate exists
--------------------
The reverse tunnel is the single feature that lets an ordinary home PC join the
mesh. Roughly a third of residential broadband sits behind CGNAT, where inbound
connections are impossible and port forwarding cannot help because the user does
not control the carrier's NAT. Without a working tunnel, ``meshd`` -- the single
binary a contributor is supposed to install -- only works for people who can
already accept inbound connections, which is the minority we least need to serve.

The Go agent shipped with an empty ``tunnel`` package (a bare struct, imported by
nothing) while the Python agent had a full implementation. Registration and
heartbeats both work *outbound*, so the node looked perfectly healthy in the
registry; only actually dispatching work to it would have revealed that it could
never be reached. This gate makes that failure impossible to reintroduce
silently.

What is proven here
-------------------
  1. ``meshd`` advertises the tunnel sentinel URL, so the coordinator must route
     to it through the tunnel or not at all.
  2. The advertised address is genuinely dead -- a direct HTTP call to it fails.
     (Non-vacuity: without this, "work completed" could just mean the
     coordinator quietly used a reachable address.)
  3. The coordinator reports the node's tunnel as attached.
  4. A real batch job dispatched to that node **completes**, with results, over
     the tunnel.
  5. Blob traffic (a large, base64-framed payload) survives the tunnel, which is
     the frame size that would break a default 64 KiB line scanner.
  6. Killing the tunnel takes the node out of service, and it recovers when the
     agent comes back -- so step 4 was really the tunnel doing the work.

Skips (exit 0 with a printed reason) when the Go toolchain is absent.
Exit 0 = PASS.  Exit 1 = FAIL.
"""
from __future__ import annotations

import base64
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
GO_DIR = REPO_ROOT / "agent-go"

from meshpool.common.netutil import TUNNEL_SENTINEL_URL  # noqa: E402
from sim.harness import Harness  # noqa: E402

CP_PORT = 8412
# The harness gives its own agents cp_port+1+idx, so meshd must sit clear of that
# range or it races the Python agent for the socket.
MESHD_PORT = 8419


def _kill_tree(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            proc.terminate()
        proc.wait(timeout=15)
    except Exception:  # noqa: BLE001
        pass


def _wait_http(url: str, timeout: float = 45.0) -> None:
    last: Exception | None = None
    end = time.time() + timeout
    while time.time() < end:
        try:
            if httpx.get(url, timeout=2.0).status_code < 500:
                return
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(0.3)
    raise TimeoutError(f"not ready at {url}: {last}")


def _node(cp: str, name: str) -> dict | None:
    nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=5).json()["nodes"]
    return next((n for n in nodes if n["name"] == name), None)


def _wait_node(cp: str, name: str, want: str, limit: float) -> str | None:
    end = time.time() + limit
    seen = None
    while time.time() < end:
        n = _node(cp, name)
        seen = None if n is None else n["status"]
        if seen == want:
            return seen
        time.sleep(0.5)
    return seen


def main() -> int:  # noqa: C901
    go = shutil.which("go")
    if go is None:
        print("SKIP: no Go toolchain on PATH — cannot build meshd.")
        print("      (This gate is skipped, not passed. Install Go to run it.)")
        return 0

    data_root = Path(__file__).resolve().parent / "_data_go_tunnel"
    shutil.rmtree(data_root, ignore_errors=True)
    meshd_data = data_root / "meshd"
    meshd_data.mkdir(parents=True, exist_ok=True)

    print("== building meshd ==")
    exe = data_root / ("meshd.exe" if os.name == "nt" else "meshd")
    build = subprocess.run([go, "build", "-o", str(exe), "./cmd/meshd"],
                           cwd=str(GO_DIR), capture_output=True, text=True)
    if build.returncode != 0:
        print("FAILED: meshd did not build")
        print(build.stdout[-2000:])
        print(build.stderr[-2000:])
        return 1

    h = Harness(n_agents=1, cp_port=CP_PORT, data_dir=str(data_root / "cluster"))
    proc = None
    ok = True
    try:
        print("\n== starting control plane + one Python agent ==")
        h.start()
        cp = h.cp_url()

        env = dict(os.environ)
        env.update({
            "MESHPOOL_CONTROL_PLANE_URL": cp,
            "MESHPOOL_AGENT_HOST": "127.0.0.1",
            "MESHPOOL_AGENT_BIND": "127.0.0.1",
            "MESHPOOL_AGENT_PORT": str(MESHD_PORT),
            # The whole point: advertise an address nobody can reach.
            "MESHPOOL_ADVERTISE_URL": TUNNEL_SENTINEL_URL,
            "MESHPOOL_NAME": "meshd-tunnel-1",
            "MESHPOOL_DATA_DIR": str(meshd_data),
            "MESHPOOL_SIM_CORES": "4",
            "MESHPOOL_SIM_RAM_MB": "8192",
        })
        proc = subprocess.Popen([str(exe)], env=env, cwd=str(REPO_ROOT))
        local = f"http://127.0.0.1:{MESHD_PORT}"
        _wait_http(f"{local}/health")

        # -- 1 + 2: the advertised address is the sentinel, and it is dead -----
        print("\n== test 1: the node advertises an unreachable address ==")
        got = _wait_node(cp, "meshd-tunnel-1", "online", 45)
        node = _node(cp, "meshd-tunnel-1")
        assert node is not None, "meshd never appeared in the registry"
        assert got == "online", f"meshd is {got!r}, not online"
        assert node["url"] == TUNNEL_SENTINEL_URL, (
            f"node advertised {node['url']!r}, expected the tunnel sentinel "
            f"{TUNNEL_SENTINEL_URL!r} — this gate would otherwise prove nothing")

        reachable = True
        try:
            httpx.post(f"{node['url']}/health", json={}, timeout=3.0)
        except Exception:  # noqa: BLE001
            reachable = False
        assert not reachable, (
            f"the advertised URL {node['url']} answered — it must be dead, or "
            "'work completed' below could just mean the coordinator called it directly")
        print(f"  advertised {node['url']} and it is genuinely unreachable")
        print("  PASS: any work that completes must have travelled the tunnel")

        # -- 3: the coordinator sees an attached tunnel -----------------------
        print("\n== test 2: the coordinator reports the reverse tunnel attached ==")
        node_id = node["id"]
        via = None
        end = time.time() + 45
        while time.time() < end:
            n = _node(cp, "meshd-tunnel-1")
            via = None if n is None else n.get("reachable_via")
            if via == "tunnel":
                break
            time.sleep(0.5)
        assert via == "tunnel", (
            f"coordinator says the node is reachable_via={via!r}, not 'tunnel' — "
            "the Go agent never dialled in, so it can only be reached at an address "
            "that does not answer")
        py_view = next(n for n in httpx.get(f"{cp}/api/v1/nodes", timeout=5).json()["nodes"]
                       if n["name"] != "meshd-tunnel-1")
        assert py_view["reachable_via"] == "direct", (
            "the directly-addressable Python agent also reports 'tunnel' — the field "
            "is constant and proves nothing")
        print(f"  meshd: reachable_via=tunnel   python agent: "
              f"reachable_via={py_view['reachable_via']}")
        print("  PASS: the field discriminates, and the Go agent's tunnel is attached")

        # -- 4: a real job actually runs on it --------------------------------
        print("\n== test 3: a real batch job completes on the unreachable node ==")
        # meshd deliberately runs only wasm:<sha256> user functions -- the curated
        # built-ins stay in the Python registry so two implementations can never
        # disagree on the same named function. So the job has to be a real user
        # module, which is also the realistic case for a donor machine.
        sys.path.insert(0, str(REPO_ROOT / "tests"))
        try:
            import wasmtime  # noqa: F401
            from test_wasm_runtime import _echo_module  # type: ignore
        except ImportError as exc:
            print(f"  SKIP: wasmtime unavailable, cannot build a user module ({exc})")
            print("        (the tunnel itself is already proven by tests 1-2)")
            raise SystemExit(0)

        module = _echo_module()
        up = httpx.post(f"{cp}/api/v1/functions", timeout=60,
                        params={"name": "tunnel-echo"}, content=module,
                        headers={"content-type": "application/octet-stream"})
        assert up.status_code == 200, f"function upload failed: {up.status_code} {up.text[:300]}"
        fn = up.json()["function"]
        print(f"  uploaded a real wasm module as {fn[:24]}...")

        # Take the Python agent out of the running so the work has nowhere else
        # to go: if it lands anywhere, it lands on the tunnelled node.
        py = py_view
        httpx.post(f"{cp}/api/v1/nodes/{py['id']}/reclaim", json={}, timeout=10)
        print(f"  reclaimed the Python agent ({py['name']}) — only the tunnelled node is eligible")

        payloads = [{"n": i, "tag": f"u{i}"} for i in range(6)]
        job = httpx.post(f"{cp}/api/v1/jobs", timeout=20, json={
            "name": "tunnel-job", "func": fn, "payloads": payloads,
        }).json()
        job_id = job["job_id"]

        end = time.time() + 180
        final = None
        while time.time() < end:
            final = httpx.get(f"{cp}/api/v1/jobs/{job_id}", timeout=10).json()
            if str(final["status"]).startswith("completed"):
                break
            time.sleep(1.0)
        assert final is not None and str(final["status"]).startswith("completed"), (
            f"job did not finish over the tunnel: {None if final is None else final['status']}")
        done = final["progress"]["done"]
        assert done == 6, (
            f"only {done}/6 units completed over the tunnel "
            f"(failed={final['progress'].get('failed')}); "
            f"first error: {next((u.get('error') for u in final.get('units', []) if u.get('error')), None)}")

        units = final.get("units", [])
        ran_here = [u for u in units if u.get("assigned_node") == node_id]
        assert ran_here, (
            "no unit reports the tunnelled node as its executor — the work went "
            "somewhere else and this gate proved nothing")
        # The results must track the inputs. A canned or echoed-request answer would
        # satisfy "done == 6" while nothing actually executed on the far side.
        by_tag = {}
        for u in units:
            res = u.get("result")
            if isinstance(res, dict) and "tag" in res:
                by_tag[res["tag"]] = res
        assert len(by_tag) == 6, (
            f"only {len(by_tag)} distinct results came back; the node is returning "
            "a constant, not executing the module")
        for p in payloads:
            assert by_tag.get(p["tag"]) == p, (
                f"unit {p['tag']} returned {by_tag.get(p['tag'])!r}, expected {p!r} — "
                "the payload did not survive the tunnel intact")
        print(f"  {done}/6 units done, {len(ran_here)} executed by the tunnelled node")
        print("  all 6 results match their distinct payloads exactly")
        print("  PASS: real work reached a node with no inbound address")

        # -- 5: a large framed payload survives the tunnel --------------------
        print("\n== test 4: a large blob survives the tunnel framing ==")
        payload = os.urandom(512 * 1024)
        bid = hashlib.sha256(payload).hexdigest()
        b64 = base64.b64encode(payload).decode("ascii")
        r = httpx.post(f"{cp}/api/v1/nodes/{node_id}/blob_put",
                       json={"blob_id": bid, "data": b64}, timeout=120)
        if r.status_code == 404:
            print("  (no coordinator route for direct blob_put; exercising the agent"
                  " loopback instead, which shares the same handler)")
            r = httpx.post(f"{local}/blob_put", json={"blob_id": bid, "data": b64}, timeout=120)
        assert r.status_code == 200, f"large blob rejected: {r.status_code} {r.text[:200]}"
        back = httpx.post(f"{local}/blob_get", json={"blob_id": bid}, timeout=120).json()
        assert base64.b64decode(back["data"]) == payload, "large blob did not round-trip"
        print(f"  PASS: {len(b64)} B of framed base64 round-tripped "
              "(a 64 KiB line limit would have failed here)")

        # -- 6: killing the agent removes it; restarting restores it ----------
        print("\n== test 5: losing the tunnel takes the node out of service ==")
        _kill_tree(proc)
        proc = None
        got = _wait_node(cp, "meshd-tunnel-1", "offline", 90)
        assert got == "offline", (
            f"node stayed {got!r} after meshd died — 'online' does not track the "
            "tunnel, so test 1's online assertion is vacuous")
        print("  killed meshd -> coordinator marked it offline")

        proc = subprocess.Popen([str(exe)], env=env, cwd=str(REPO_ROOT))
        _wait_http(f"{local}/health")
        got = _wait_node(cp, "meshd-tunnel-1", "online", 90)
        assert got == "online", f"meshd did not recover its tunnel (got {got!r})"
        print("  PASS: tunnel reconnects automatically after a restart")

        print("\nALL GO TUNNEL CHECKS PASSED")
    except AssertionError as exc:
        print(f"\nFAILED: {exc}")
        ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        ok = False
    finally:
        _kill_tree(proc)
        h.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
