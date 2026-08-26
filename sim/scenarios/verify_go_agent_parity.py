"""Acceptance gate: the Go node agent (``meshd``) is protocol-compatible with the
Python daemon it is meant to replace.

Why this gate exists
--------------------
Phase 1 of the plan replaces the Python agent with a single static binary so a
contributor installs one file instead of a Python environment. A rewrite is only safe
if the new binary speaks *exactly* the protocol the existing control plane already
speaks -- so this gate runs the **real** control plane and makes ``meshd`` join it as a
peer of a **real** Python agent, then exercises the surface they share.

What is proven here (and what is not)
-------------------------------------
Proven:
  1. ``meshd`` derives the *same* device id from a public key as ``meshpool.common.identity``
     (checked by re-deriving in Python from the binary's own identity.json).
  2. It performs a signed enrollment the coordinator accepts, and appears ONLINE in the
     registry alongside the Python agent.
  3. It heartbeats: it is still ONLINE after several heartbeat intervals.
  4. Its blob store round-trips bytes identically, rejects a wrong content hash, and
     reports absence honestly.
  5. ``blob_pull`` fetches a blob **directly from the Python peer** and verifies the
     content hash -- the peer-to-peer weight path, cross-implementation.
  6. A corrupt peer is rejected, not trusted.
  7. Endpoints meshd has not implemented return **501**, not a plausible wrong answer.
  8. meshd **runs a real wasm: user function**, hydrating the module from the Python
     peer, caching it, and returning the same answer the Python runtime does.
  9. ``reclaim`` drains the node instantly and ``resume`` restores it.

NOT proven (deliberately -- meshd does not implement these yet):
  - hosting replicas, model shards, secret-lease redemption, or the reverse tunnel.

Skips (exit 0 with a printed reason) when the Go toolchain is absent.
Exit 0 = PASS.  Exit 1 = FAIL.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness, _kill_tree  # noqa: E402
from meshpool.common.config import SCALE_PRESETS  # noqa: E402
from meshpool.common.identity import device_id_for_public_key  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
GO_DIR = REPO_ROOT / "agent-go"
CP_PORT = 8146
MESHD_PORT = CP_PORT + 50


def _post(url: str, body: dict, timeout: float = 30.0) -> httpx.Response:
    return httpx.post(url, json=body, timeout=timeout)


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
    raise TimeoutError(f"meshd not ready at {url}: {last}")


def main() -> int:
    go = shutil.which("go")
    if go is None:
        print("SKIP: no Go toolchain on PATH — cannot build meshd.")
        print("      (This gate is skipped, not passed. Install Go to run it.)")
        return 0

    data_root = Path(__file__).resolve().parent / "_data_go_parity"
    shutil.rmtree(data_root, ignore_errors=True)
    meshd_data = data_root / "meshd"
    meshd_data.mkdir(parents=True, exist_ok=True)

    print("== building meshd ==")
    exe = data_root / ("meshd.exe" if os.name == "nt" else "meshd")
    build = subprocess.run([go, "build", "-o", str(exe), "./cmd/meshd"],
                           cwd=str(GO_DIR), capture_output=True, text=True, errors="replace")
    if build.returncode != 0:
        print("FAILED: meshd did not build")
        print(build.stdout[-2000:])
        print(build.stderr[-2000:])
        return 1
    print(f"  built {exe.name} ({exe.stat().st_size // 1024} KiB, single static binary)")

    # One real Python agent so meshd has a real peer to pull blobs from.
    h = Harness(n_agents=1, cp_port=CP_PORT, data_dir=str(data_root / "cluster"))
    meshd_proc = None
    ok = True
    try:
        print("\n== starting a real control plane + one Python agent ==")
        h.start()
        cp = h.cp_url()
        py_url = f"http://127.0.0.1:{h.agent_ports[0]}"
        print(f"  control plane {cp}; python agent {py_url}")

        env = dict(os.environ)
        env.update({
            "MESHPOOL_CONTROL_PLANE_URL": cp,
            "MESHPOOL_AGENT_HOST": "127.0.0.1",
            "MESHPOOL_AGENT_BIND": "127.0.0.1",
            "MESHPOOL_AGENT_PORT": str(MESHD_PORT),
            "MESHPOOL_NAME": "meshd-go-1",
            "MESHPOOL_DATA_DIR": str(meshd_data),
            "MESHPOOL_SIM_CORES": "4",
            "MESHPOOL_SIM_RAM_MB": "8192",
        })
        meshd_proc = subprocess.Popen([str(exe)], env=env, cwd=str(REPO_ROOT))
        meshd = f"http://127.0.0.1:{MESHD_PORT}"
        _wait_http(f"{meshd}/health")

        # -- 1. device id derivation is byte-identical across implementations ---
        print("\n== test 1: Go derives the same device id as the Python implementation ==")
        ident = json.loads((meshd_data / "identity.json").read_text(encoding="utf-8"))
        go_device_id = ident["device_id"]
        py_derived = device_id_for_public_key(ident["public_key"])
        print(f"  meshd  says: {go_device_id}")
        print(f"  python says: {py_derived}")
        assert go_device_id == py_derived, (
            "device id derivation diverged between Go and Python — the same machine "
            "would register as two different devices depending on which agent it ran")
        assert go_device_id.startswith("dev_") and len(go_device_id) == 4 + 24, (
            f"malformed device id {go_device_id!r}")
        print("  PASS: identical device id from the same key material")

        # -- 2. signed enrollment accepted; node visible and ONLINE ------------
        print("\n== test 2: signed enrollment accepted by the real control plane ==")
        deadline = time.time() + 30
        node = None
        while time.time() < deadline:
            nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=5).json()["nodes"]
            node = next((n for n in nodes if n["name"] == "meshd-go-1"), None)
            if node and node["status"] == "online":
                break
            time.sleep(0.5)
        assert node is not None, "meshd never appeared in the registry"
        assert node["status"] == "online", f"meshd status is {node['status']}"
        assert node.get("device_id") == go_device_id, (
            f"coordinator keyed the node on {node.get('device_id')!r}, "
            f"not the device id meshd signed with ({go_device_id!r})")
        health = httpx.get(f"{meshd}/health", timeout=5).json()
        assert health.get("agent") == "meshd-go", (
            "health did not identify the Go agent — a parity gate must be able to tell "
            "which implementation it is talking to")
        assert health.get("node_id") == node["id"], (
            "meshd did not record the node id the coordinator assigned it")
        py_node = next(n for n in httpx.get(f"{cp}/api/v1/nodes", timeout=5).json()["nodes"]
                       if n["name"] != "meshd-go-1")
        print(f"  registry holds both implementations: "
              f"{py_node['name']} (python) + {node['name']} (go)")
        print("  PASS: enrollment signature verified; node ONLINE and correctly keyed")

        # -- 3. heartbeats keep it alive (and the check is not free) ----------
        print("\n== test 3: heartbeat loop keeps the node ONLINE ==")
        # Read the real reaper window rather than hardcoding it, so this gate stays
        # correct if the scale preset changes.
        timeout_s = int(SCALE_PRESETS[h.scale]["node_timeout_s"])

        def status_of() -> str | None:
            nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=5).json()["nodes"]
            n = next((x for x in nodes if x["name"] == "meshd-go-1"), None)
            return None if n is None else n["status"]

        def wait_status(want: str, limit: float) -> str | None:
            end = time.time() + limit
            seen = None
            while time.time() < end:
                seen = status_of()
                if seen == want:
                    return seen
                time.sleep(0.5)
            return seen

        # The registry reaps a node whose heartbeat is older than node_timeout_s.
        # Staying ONLINE well past that window proves meshd really heartbeats.
        held = timeout_s + 6
        time.sleep(held)
        assert status_of() == "online", (
            f"meshd went {status_of()!r} after {held}s — it registered but "
            "does not heartbeat")
        print(f"  still ONLINE after {held}s (> node_timeout_s={timeout_s})")

        # Non-vacuity: prove the assertion above could have failed. Stop meshd and
        # the SAME check must flip to offline — otherwise "online" would just mean
        # "the reaper never runs" and this test would measure nothing.
        _kill_tree(meshd_proc)
        meshd_proc = None
        got = wait_status("offline", timeout_s + 15)
        assert got == "offline", (
            f"node stayed {got!r} after meshd was killed — the ONLINE assertion above "
            "is vacuous because the coordinator never reaps stale nodes")
        print("  killed meshd -> coordinator reaped it to OFFLINE (the check has teeth)")

        # And restarting against the same data dir must reuse the persisted identity
        # and re-enroll, which is the whole point of an on-disk device key.
        meshd_proc = subprocess.Popen([str(exe)], env=env, cwd=str(REPO_ROOT))
        _wait_http(f"{meshd}/health")
        got = wait_status("online", 30)
        assert got == "online", f"meshd did not come back ONLINE after restart (got {got!r})"
        ident2 = json.loads((meshd_data / "identity.json").read_text(encoding="utf-8"))
        assert ident2["device_id"] == go_device_id, (
            "meshd generated a NEW identity on restart — a contributor's machine would "
            "look like a brand-new device after every reboot, losing its reputation")
        print("  PASS: heartbeats flow, reaping works, and identity survives a restart")

        # -- 4. blob store round-trip -----------------------------------------
        print("\n== test 4: blob store round-trips bytes and enforces content hashes ==")
        payload = os.urandom(96 * 1024)
        bid = hashlib.sha256(payload).hexdigest()
        b64 = base64.b64encode(payload).decode("ascii")

        r = _post(f"{meshd}/blob_has", {"blob_id": bid})
        assert r.status_code == 200 and r.json()["has"] is False, "absent blob reported present"

        r = _post(f"{meshd}/blob_put", {"blob_id": bid, "data": b64})
        assert r.status_code == 200, f"blob_put failed: {r.status_code} {r.text[:200]}"
        assert r.json()["size"] == len(payload), "meshd miscounted the stored bytes"

        r = _post(f"{meshd}/blob_get", {"blob_id": bid})
        got = base64.b64decode(r.json()["data"])
        assert r.json()["found"] is True and got == payload, (
            "blob did not round-trip byte-for-byte")

        wrong_id = hashlib.sha256(payload + b"x").hexdigest()
        r = _post(f"{meshd}/blob_put", {"blob_id": wrong_id, "data": b64})
        assert r.status_code == 400, (
            f"meshd stored bytes under a wrong content hash (status {r.status_code}) — "
            "content addressing would be meaningless")

        r = _post(f"{meshd}/blob_delete", {"blob_id": bid})
        assert r.status_code == 200
        assert _post(f"{meshd}/blob_has", {"blob_id": bid}).json()["has"] is False
        print(f"  PASS: {len(payload)} B round-tripped exactly; wrong hash rejected (400); "
              "delete is real")

        # -- 5. peer-to-peer pull from the PYTHON agent -----------------------
        print("\n== test 5: meshd pulls a blob directly from the Python agent ==")
        r = _post(f"{py_url}/blob_put", {"blob_id": bid, "data": b64})
        assert r.status_code == 200, f"python agent refused the blob: {r.text[:200]}"
        r = _post(f"{meshd}/blob_pull", {"blob_id": bid, "sources": [py_url]})
        assert r.status_code == 200, f"blob_pull failed: {r.status_code} {r.text[:300]}"
        body = r.json()
        assert body["stored"] is True and body["cached"] is False
        assert body["bytes"] == len(payload), (
            f"pulled {body['bytes']} B, expected {len(payload)}")
        got = base64.b64decode(_post(f"{meshd}/blob_get", {"blob_id": bid}).json()["data"])
        assert got == payload, "peer-pulled bytes differ from the original"
        # A second pull must be a no-op (cached), not a re-download.
        again = _post(f"{meshd}/blob_pull", {"blob_id": bid, "sources": [py_url]}).json()
        assert again["cached"] is True and again["bytes"] == 0, (
            "meshd re-downloaded a blob it already had")
        print(f"  PASS: {len(payload)} B pulled Go<-Python, verified, and cached")

        # -- 6. a corrupt peer is rejected ------------------------------------
        print("\n== test 6: a peer serving the wrong bytes is rejected, not trusted ==")
        other = os.urandom(4096)
        other_id = hashlib.sha256(other).hexdigest()
        # Ask meshd to pull `other_id` from a peer that holds something else entirely.
        # (The python agent has `bid`, not `other_id`, so the source cannot satisfy it.)
        r = _post(f"{meshd}/blob_pull", {"blob_id": other_id, "sources": [py_url]})
        assert r.status_code == 502, (
            f"expected 502 from an unsatisfiable pull, got {r.status_code}")
        assert _post(f"{meshd}/blob_has", {"blob_id": other_id}).json()["has"] is False, (
            "meshd stored something for a blob it could not verify")
        # And a source that is simply unreachable must fail cleanly, not hang or store.
        r = _post(f"{meshd}/blob_pull",
                  {"blob_id": other_id, "sources": ["http://127.0.0.1:1"]})
        assert r.status_code == 502, f"unreachable source gave {r.status_code}"
        print("  PASS: unverifiable and unreachable sources both fail closed")

        # -- 7. unimplemented endpoints fail loudly ---------------------------
        print("\n== test 7: unimplemented endpoints return 501, never a wrong answer ==")
        for path in ("/shard/forward", "/start_replica", "/ollama/generate", "/policy"):
            r = _post(f"{meshd}{path}", {})
            assert r.status_code == 501, (
                f"{path} returned {r.status_code}; an unimplemented endpoint must fail "
                "loudly (501) so the coordinator reschedules instead of trusting a stub")
            assert "meshd" in r.json().get("detail", ""), "501 body is not explanatory"
        print("  PASS: 4 unimplemented endpoints return an explanatory 501")

        # -- 8. meshd actually RUNS a user function ---------------------------
        # This is the endpoint that matters: without it a Go node in the pool absorbs
        # units, 501s, and burns each one's retry budget before it lands somewhere
        # useful. The module is deliberately placed only on the *Python* peer, so a
        # pass also proves the cross-implementation hydration path -- meshd pulling
        # a content-addressed module from a Python agent and verifying it.
        print("\n== test 8: meshd runs a real wasm user function, pulled from a peer ==")
        try:
            import wasmtime  # noqa: F401
            sys.path.insert(0, str(REPO_ROOT / "tests"))
            from test_wasm_runtime import _echo_module  # type: ignore
        except ImportError as exc:
            print(f"  SKIP: cannot build a test module ({exc})")
        else:
            module = _echo_module()
            module_id = hashlib.sha256(module).hexdigest()
            fn_name = f"wasm:{module_id}"

            # Seed the module on the PYTHON peer only.
            r = _post(f"{py_url}/blob_put",
                      {"blob_id": module_id,
                       "data": base64.b64encode(module).decode("ascii")})
            assert r.status_code == 200, f"seeding the python peer failed: {r.status_code}"
            r = _post(f"{meshd}/blob_has", {"blob_id": module_id})
            assert r.json().get("has") is False, (
                "meshd already had the module; the peer-pull path would not be exercised")

            payload_a = {"alpha": 1, "nested": {"k": [1, 2, 3]}}
            r = _post(f"{meshd}/run_unit",
                      {"job_id": "j1", "unit_id": "u1", "func": fn_name,
                       "payload": payload_a, "wasm_sources": [py_url]})
            assert r.status_code == 200, (
                f"meshd could not run the unit: {r.status_code} {r.text[:300]}")
            body = r.json()
            assert body["unit_id"] == "u1", f"wrong unit_id echoed: {body}"
            assert body["result"] == payload_a, (
                f"meshd returned {body['result']!r}, expected the echoed payload "
                f"{payload_a!r}")
            assert isinstance(body.get("compute_ms"), (int, float)), (
                "compute_ms missing — the coordinator bills on it")
            print(f"  ran unit u1 -> {json.dumps(body['result'])} "
                  f"in {body['compute_ms']:.1f}ms")

            # It must have cached what it pulled, or every unit re-downloads.
            r = _post(f"{meshd}/blob_has", {"blob_id": module_id})
            assert r.json().get("has") is True, (
                "meshd did not cache the pulled module; the next unit would re-fetch it")

            # NON-VACUITY: a different payload must give a different answer. If meshd
            # returned a canned value, or echoed the request body, the check above
            # would pass while nothing was executed.
            payload_b = {"beta": 2, "nested": {"k": ["x"]}}
            r = _post(f"{meshd}/run_unit",
                      {"job_id": "j1", "unit_id": "u2", "func": fn_name,
                       "payload": payload_b})   # no sources: must use the cache
            assert r.status_code == 200, f"cached run failed: {r.status_code} {r.text[:300]}"
            assert r.json()["result"] == payload_b, (
                f"second run returned {r.json()['result']!r}, expected {payload_b!r}")
            assert payload_a != payload_b, "control payloads must differ"
            print(f"  ran unit u2 from cache -> {json.dumps(r.json()['result'])} "
                  f"(tracks the payload, so it is really executing)")

            # The Python runtime must agree on the same bytes: same module, same
            # payload, same answer. Divergence here is what makes the coordinator's
            # spot-check oracle slash honest nodes.
            r = _post(f"{py_url}/run_unit",
                      {"job_id": "j1", "unit_id": "u3", "func": fn_name,
                       "payload": payload_a, "wasm_sources": [py_url]})
            assert r.status_code == 200, f"python agent failed: {r.status_code} {r.text[:300]}"
            assert r.json()["result"] == body["result"], (
                f"cross-implementation divergence: python said {r.json()['result']!r}, "
                f"go said {body['result']!r} for identical module bytes and payload")
            print("  PASS: identical result from the Python agent on the same bytes")

            # A missing module with no sources must 404, not hang or guess.
            missing = "0" * 64
            r = _post(f"{meshd}/run_unit",
                      {"job_id": "j1", "unit_id": "u4", "func": f"wasm:{missing}",
                       "payload": {}})
            assert r.status_code == 404, (
                f"unknown module gave {r.status_code}, expected 404")

            # A built-in still 501s: meshd must never reimplement a curated function,
            # because two implementations of one name is how a pool starts disagreeing.
            r = _post(f"{meshd}/run_unit",
                      {"job_id": "j1", "unit_id": "u5", "func": "square",
                       "payload": {"x": 3}})
            assert r.status_code == 501, (
                f"built-in 'square' gave {r.status_code}; meshd must not guess at "
                "curated functions")
            print("  PASS: unknown module 404s and curated built-ins still 501")

        # -- 9. the contributor kill-switch is real ---------------------------
        print("\n== test 9: reclaim drains the node instantly, resume restores it ==")
        r = _post(f"{meshd}/reclaim", {})
        assert r.status_code == 200, f"reclaim gave {r.status_code}"
        assert r.json().get("status") == "draining", f"reclaim body: {r.json()}"
        r = _post(f"{meshd}/run_unit",
                  {"job_id": "j2", "unit_id": "u6", "func": "wasm:" + "0" * 64,
                   "payload": {}})
        assert r.status_code == 503, (
            f"a reclaimed node accepted work ({r.status_code}) — the kill-switch is "
            "the one promise a contributor must be able to rely on")
        r = _post(f"{meshd}/resume", {})
        assert r.status_code == 200 and r.json().get("status") == "online"
        r = _post(f"{meshd}/run_unit",
                  {"job_id": "j2", "unit_id": "u7", "func": "wasm:" + "0" * 64,
                   "payload": {}})
        assert r.status_code == 404, (
            f"after resume the node still refused work ({r.status_code}); expected the "
            "ordinary 404 for an unknown module")
        print("  PASS: reclaim -> 503, resume -> back in service")

        print("\nALL GO-AGENT PARITY CHECKS PASSED")
        print("  scope: identity + enrollment + heartbeat + blob store + peer pull")
        print("         + running wasm: user functions + the reclaim kill-switch.")
        print("  meshd does NOT yet host replicas, serve model shards, redeem secrets,")
        print("  or dial the reverse tunnel — and says so with 501 rather than guessing.")
    except AssertionError as exc:
        print(f"\nFAILED: {exc}")
        ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED (unexpected): {type(exc).__name__}: {exc}")
        ok = False
    finally:
        _kill_tree(meshd_proc)
        h.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
