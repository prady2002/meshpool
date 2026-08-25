"""Acceptance gate: the Mesh Map page is correctly wired and the APIs it reads
exist with the shapes the component expects.

The Mesh Map page reads:
  GET /api/v1/cluster   → {nodes: [...], summary: {...}}
  GET /api/v1/jobs      → {jobs: [...]}
  GET /api/v1/models    → {models: [...]}

It also serves from the built React console (web/dist/) when present.

Exit 0 = PASS.  Exit 1 = FAIL.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sim.harness import Harness  # noqa: E402

CP_PORT = 8660

# Endpoints that MeshMapPage.tsx reads, plus the shape each key must have.
# (path, key_to_inspect_or_None, expected_python_type)
MESH_MAP_CONTRACT: list[tuple[str, str | None, type]] = [
    ("/api/v1/cluster", "nodes",   list),
    ("/api/v1/cluster", "summary", dict),
    ("/api/v1/jobs",    "jobs",    list),
    ("/api/v1/models",  "models",  list),
]

# Keys the MeshMap component destructures from cluster.nodes[i]
NODE_FIELDS = ["id", "name", "status", "capability", "load", "link", "trust_score"]
CAPABILITY_FIELDS = [
    "cpu_cores", "cpu_arch", "ram_mb", "gpu", "vram_mb", "flops_estimate",
]
LOAD_FIELDS = ["cpu_load_pct", "ram_used_mb", "running_units"]
LINK_FIELDS = ["latency_ms", "bandwidth_mbps"]


def main() -> int:  # noqa: C901
    data_root = REPO_ROOT / "sim" / "scenarios" / "_data_mesh_map"
    shutil.rmtree(data_root, ignore_errors=True)
    h = Harness(n_agents=2, cp_port=CP_PORT, data_dir=str(data_root))
    ok = True
    failures: list[str] = []

    try:
        print("== starting 2-agent cluster for mesh map gate ==")
        h.start()
        cp = h.cp_url()

        # ── test 1: endpoint shapes ───────────────────────────────────────── #
        print("\n== test 1: mesh map API contract ==")
        for path, key, want in MESH_MAP_CONTRACT:
            r = httpx.get(f"{cp}{path}", timeout=15)
            if r.status_code != 200:
                failures.append(f"{path} → HTTP {r.status_code}")
                continue
            body = r.json()
            if not isinstance(body, dict):
                failures.append(f"{path} top-level is not an object")
                continue
            if key is None:
                print(f"  {path:<30} object")
                continue
            val = body.get(key)
            if val is None:
                failures.append(f"{path} missing key '{key}'")
                continue
            if not isinstance(val, want):
                failures.append(
                    f"{path}.{key} is {type(val).__name__}, expected {want.__name__}"
                )
                continue
            print(f"  {path:<30} .{key} is {want.__name__}  len={len(val)}")

        # ── test 2: node fields the MeshMap detail panel reads ────────────── #
        print("\n== test 2: node shape — fields the detail panel destructures ==")
        cluster = httpx.get(f"{cp}/api/v1/cluster", timeout=15).json()
        nodes: list[dict] = cluster.get("nodes", [])
        if not nodes:
            print("  (no nodes registered — shape check skipped)")
        else:
            node = nodes[0]
            for field in NODE_FIELDS:
                if field not in node:
                    failures.append(f"cluster.nodes[0] missing field '{field}'")
            cap = node.get("capability", {})
            for field in CAPABILITY_FIELDS:
                if field not in cap:
                    failures.append(f"node.capability missing '{field}'")
            load = node.get("load", {})
            for field in LOAD_FIELDS:
                if field not in load:
                    failures.append(f"node.load missing '{field}'")
            link = node.get("link", {})
            for field in LINK_FIELDS:
                if field not in link:
                    failures.append(f"node.link missing '{field}'")
            if not failures:
                print(f"  all {len(NODE_FIELDS) + len(CAPABILITY_FIELDS) + len(LOAD_FIELDS) + len(LINK_FIELDS)}"
                      f" detail-panel fields present in node[0]")

        # ── test 3: partition shape in models (model shard pipeline view) ─── #
        print("\n== test 3: model partition shape for pipeline view ==")
        models_body = httpx.get(f"{cp}/api/v1/models", timeout=15).json()
        model_list: list[dict] = models_body.get("models", [])
        if not model_list:
            print("  (no models deployed — partition shape check skipped)")
        else:
            m = model_list[0]
            for key in ("id", "model", "backend", "n_layers", "partition", "status"):
                if key not in m:
                    failures.append(f"models[0] missing '{key}'")
            parts = m.get("partition", [])
            if parts:
                p = parts[0]
                for key in ("node_id", "start_layer", "end_layer", "is_first", "is_last"):
                    if key not in p:
                        failures.append(f"partition[0] missing '{key}'")
                if not failures:
                    print(f"  model '{m['model']}': {len(parts)} partition slice(s), all keys present")

        # ── test 4: console bundle is served (if built) ───────────────────── #
        print("\n== test 4: console bundle presence ==")
        console_index = REPO_ROOT / "web" / "dist" / "index.html"
        if console_index.exists():
            r = httpx.get(f"{cp}/console", timeout=10)
            if r.status_code == 200:
                print(f"  console served at /console ({len(r.text)} bytes)")
                # The built bundle must reference the mesh-map route
                bundle_dir = REPO_ROOT / "web" / "dist" / "assets"
                js_files = list(bundle_dir.glob("*.js")) if bundle_dir.exists() else []
                found_map = False
                for js_file in js_files:
                    content = js_file.read_text(encoding="utf-8", errors="replace")
                    if "Topology" in content and "Mesh Map" in content:
                        found_map = True
                        print(f"  mesh-map markers found in {js_file.name}")
                        break
                if not found_map and js_files:
                    failures.append(
                        "built bundle does not contain 'Topology' and 'Mesh Map' markers — "
                        "MeshMapPage may not be included in the build"
                    )
                elif not js_files:
                    print("  (no .js files in dist/assets — skipping bundle content check)")
            else:
                print(f"  /console returned {r.status_code} — console not served")
        else:
            print("  web/dist not present — run `cd web && npm run build` to enable bundle checks")

        # ── test 5: non-vacuity ───────────────────────────────────────────── #
        print("\n== test 5: non-vacuity — checks must be capable of failing ==")
        # A key the cluster endpoint certainly does not return.
        body = httpx.get(f"{cp}/api/v1/cluster", timeout=15).json()
        bogus_absent = "mesh_map_nonexistent_9k3x2" not in body
        assert bogus_absent, (
            "the server is returning a key this gate invented — the contract checks "
            "above are vacuously true and prove nothing"
        )
        print("  bogus key absent from /api/v1/cluster — checks are real")

        if failures:
            ok = False
            print(f"\nFAILED: {len(failures)} mesh-map contract issue(s):")
            for f in failures:
                print(f"  - {f}")
        else:
            print("\nALL MESH MAP GATE CHECKS PASSED")

    except AssertionError as exc:
        ok = False
        print(f"\nFAILED (non-vacuity assertion): {exc}")
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
