"""End-to-end scenario: criterion (b) — machines show up live in the dashboard.

Boots a real control plane + 3 heterogeneous agents and asserts the operator-facing
surface actually works: the node API lists every machine online with its probed
capacity, the dashboard HTML + static assets are served, the summary aggregates the
pool, and the live SSE event stream is connectable. This is the automated backing for
"install -> your machines appear as one computer" (spec §9 b).
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402


def main() -> int:
    h = Harness(n_agents=3, cp_port=8160,
                data_dir=str(Path(__file__).resolve().parent / "_data_dashboard"))
    ok = True
    try:
        print("== starting cluster (control plane + 3 agents) ==")
        h.start()
        cp = h.cp_url()
        h.wait_nodes_online(3)

        # ---- test 1: node API lists every machine online with capacity ---- #
        print("\n== test 1: /api/v1/nodes shows 3 live machines with real capacity ==")
        nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=10).json()["nodes"]
        online = [n for n in nodes if n["status"] == "online"]
        assert len(online) == 3, f"expected 3 online, got {[(n['name'], n['status']) for n in nodes]}"
        cores = sorted(n["capability"]["cpu_cores"] for n in online)
        assert all(c > 0 for c in cores), cores
        assert len(set(cores)) > 1, f"expected heterogeneous capacity, got {cores}"
        print(f"  3 nodes online, cpu_cores={cores} (heterogeneous) PASS")

        # ---- test 2: summary aggregates the pool into one computer -------- #
        print("\n== test 2: cluster summary aggregates the pool ==")
        ov = httpx.get(f"{cp}/api/v1/cluster", timeout=10).json()
        summ = ov["summary"]
        assert summ["online"] == 3 and summ["total_nodes"] == 3, summ
        assert summ["total_cores"] == sum(cores), (summ["total_cores"], cores)
        assert summ["total_ram_mb"] > 0, summ
        print(f"  summary: {summ} PASS")

        # ---- test 3: dashboard HTML + static assets are served ----------- #
        print("\n== test 3: dashboard page and static assets serve ==")
        page = httpx.get(f"{cp}/", timeout=10)
        assert page.status_code == 200 and "MeshPool" in page.text, page.status_code
        for asset, needle in (("/static/app.js", "async"), ("/static/style.css", "{")):
            a = httpx.get(f"{cp}{asset}", timeout=10)
            assert a.status_code == 200 and needle in a.text, f"{asset} -> {a.status_code}"
        print("  index.html + app.js + style.css all 200 PASS")

        # ---- test 4: live event stream is connectable (SSE) -------------- #
        print("\n== test 4: live SSE event stream connects ==")
        with httpx.stream("GET", f"{cp}/api/v1/events", timeout=5) as r:
            assert r.status_code == 200, r.status_code
            ctype = r.headers.get("content-type", "")
            assert "text/event-stream" in ctype, ctype
        print(f"  /api/v1/events -> 200 {ctype!r} PASS")

        # ---- test 5: pooled-storage 'Files' card is wired + its API round-trips -- #
        print("\n== test 5: storage 'Files' card wiring + blob API round-trip ==")
        assert "Store files" in page.text and 'id="fFile"' in page.text, \
            "storage 'Files' card missing from the dashboard"
        appjs = httpx.get(f"{cp}/static/app.js", timeout=10).text
        for fn in ("uploadFile", "loadFiles", "rmBlob"):
            assert fn in appjs, f"{fn} missing from app.js"
        payload = b"MESHPOOL-DASHBOARD-FILES-" + bytes(range(256)) * 4
        blob_id = hashlib.sha256(payload).hexdigest()
        meta = httpx.post(f"{cp}/api/v1/blobs?name=note.bin", content=payload, timeout=30).json()
        assert meta["id"] == blob_id, (meta.get("id"), blob_id)
        assert len(meta["replicas"]) >= 1, meta
        listing = httpx.get(f"{cp}/api/v1/blobs", timeout=10).json()["blobs"]
        assert any(b["id"] == blob_id for b in listing), "uploaded blob not listed"
        got = httpx.get(f"{cp}/api/v1/blobs/{blob_id}", timeout=15)
        assert got.status_code == 200 and got.content == payload, (got.status_code, len(got.content))
        assert "attachment" in got.headers.get("content-disposition", ""), \
            got.headers.get("content-disposition")
        d = httpx.delete(f"{cp}/api/v1/blobs/{blob_id}", timeout=10)
        assert d.status_code == 200, d.status_code
        gone = httpx.get(f"{cp}/api/v1/blobs/{blob_id}", timeout=10)
        assert gone.status_code == 404, gone.status_code
        print("  Files card present, app.js wired, blob upload/list/download/delete round-trip PASS")

        # ---- test 6: 'bring your own weights' control is wired + upload deploys pooled -- #
        print("\n== test 6: model-weights upload wiring + pooled deploy round-trip ==")
        assert 'id="wFile"' in page.text and "bring your own weights" in page.text.lower(), \
            "'bring your own weights' control missing from the dashboard"
        for fn in ("uploadWeights", "byoWrap"):
            assert fn in appjs, f"{fn} missing from app.js"
        from meshpool.inference import model as M
        cfg = M.REGISTRY["nano-gpt"]
        ckpt = M.dumps_npz(M.random_params(cfg), itos=M.default_charset(cfg.vocab_size))
        view = httpx.post(f"{cp}/api/v1/models/upload?model=nano-gpt",
                          content=ckpt, timeout=120).json()
        assert view.get("pooled") is True, f"uploaded model not pool-backed: {view}"
        assert view.get("model") == "nano-gpt" and view.get("deployed"), view
        assert view.get("status") == "ready", view
        stages = view.get("stages") or []
        assert stages, f"no pipeline stages: {view}"
        cur = httpx.get(f"{cp}/api/v1/models", timeout=10).json()
        assert cur.get("pooled") is True, f"GET /models lost the pooled flag: {cur}"
        print(f"  weights control wired; checkpoint uploaded + deployed pooled across "
              f"{len(stages)} stage(s) PASS")

        print("\nALL DASHBOARD E2E CHECKS PASSED")
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
