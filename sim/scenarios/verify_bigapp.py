"""End-to-end scenario: host a LARGE multi-page app that does NOT fit inline (§P3+P2).

``verify_realapp`` proves the ordinary path where a site rides inline inside the
``start_replica`` control frame. This proves the headline "host ANY large multi-page
application" claim: an app bigger than the inline ceiling is NOT rejected -- the
control plane stages its whole bundle in the pooled, content-addressed blob store
(split into replicated chunks spread across the mesh) and hydrates it back onto each
replica on start. We shrink the inline cap to 4 KB (and the chunk size to 2 KB) so a
modest test app exercises the exact same code path a real 100 MB app would take:
  * a >4 KB multi-page app deployed via JSON is pooled: the deployment reports a
    ``site_blob_id`` and a null inline ``site``, and the bundle never shows up in the
    user's own Files listing (it is internal machinery),
  * every page + asset is served byte-for-byte through the control-plane proxy after
    being hydrated from the pool onto the replicas,
  * the same works through the browser ``.zip`` upload entry point,
  * killing a hosting node re-hydrates the bundle FROM THE POOL onto a survivor
    (self-heal never needed the original uploader to still be around).
"""
from __future__ import annotations

import base64
import io
import os
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402
from meshpool.hosting import site as sitelib  # noqa: E402

TOKEN = "MESHPOOL-BIGAPP-4K7P"           # marker proving OUR pooled content is served

# a real 1x1 PNG -- a genuine binary asset to prove bytes survive pool round-trip
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def build_big_site(root: Path) -> None:
    """Write a realistic multi-page static site whose total content comfortably
    exceeds the 4 KB inline ceiling (so the deploy is forced through the pool)."""
    (root / "assets" / "js").mkdir(parents=True, exist_ok=True)
    (root / "about").mkdir(parents=True, exist_ok=True)
    (root / "index.html").write_text(
        "<!doctype html><html><head><meta charset=utf-8>"
        "<script type=module src='assets/js/app.js'></script></head>"
        f"<body><h1>{TOKEN} home</h1><a href='about/'>about</a></body></html>",
        encoding="utf-8")
    (root / "about" / "index.html").write_text(
        f"<!doctype html><html><body><h1>{TOKEN} about</h1></body></html>", encoding="utf-8")
    # a chunky JS bundle -- this is what pushes the app well past the inline cap so it
    # is staged in the pool (a real SPA bundle is far bigger; same code path).
    filler = "\n".join(f"export const item{i} = {{ id: {i}, tag: '{TOKEN}' }};"
                       for i in range(160))
    (root / "assets" / "js" / "app.js").write_text(
        f"export const app='{TOKEN}';\n{filler}\nconsole.log(app);", encoding="utf-8")
    (root / "assets" / "data.json").write_text(
        f'{{"app":"{TOKEN}","pages":2}}', encoding="utf-8")
    (root / "favicon.png").write_bytes(_PNG)


def get(cp: str, dep_id: str, path: str):
    return httpx.get(f"{cp}/app/{dep_id}/{path}", timeout=10)


def served_all(cp: str, dep_id: str, n: int) -> Counter:
    c: Counter = Counter()
    pages = ["", "about/", "assets/js/app.js"]
    for i in range(n):
        try:
            r = get(cp, dep_id, pages[i % len(pages)])
            if r.status_code == 200 and TOKEN in r.text:
                c[r.headers.get("X-MeshPool-Served-By", "?")] += 1
            else:
                c[f"HTTP {r.status_code}"] += 1
        except Exception as exc:  # noqa: BLE001
            c[f"ERR {type(exc).__name__}"] += 1
    return c


def user_blob_ids(cp: str) -> set:
    """The user-facing Files listing (internal deploy bundles must NOT appear here)."""
    return {b["id"] for b in httpx.get(f"{cp}/api/v1/blobs", timeout=10).json().get("blobs", [])}


def main() -> int:
    # Shrink the inline site ceiling to 4 KB and the pool chunk size to 2 KB BEFORE the
    # cluster starts, so a modest multi-page app is forced down the exact pooled-bundle
    # path a genuinely huge app would take. The CP + agents run as subprocesses that
    # inherit this environment (this scenario process already imported site.py with the
    # default 25 MB cap, so its own pack_dir below is unaffected).
    os.environ["MESHPOOL_SITE_MAX_BYTES"] = "4096"
    os.environ["MESHPOOL_CHUNK_SIZE"] = "2048"
    base = Path(__file__).resolve().parent / "_data_bigapp"
    h = Harness(n_agents=3, cp_port=8125, data_dir=str(base / "cluster"))
    ok = True
    try:
        print("== building a large multi-page app on disk (> the 4 KB inline cap) ==")
        site_root = base / "bigsite"
        build_big_site(site_root)
        site = sitelib.pack_dir(str(site_root))
        total = sum(len(base64.b64decode(v)) for v in site.values())
        print(f"  packed {len(site)} files, {total} bytes of content (inline cap is 4096)")
        assert total > 4096, f"test app must exceed the inline cap to exercise pooling: {total}"

        print("\n== starting cluster (3 nodes; inline cap 4 KB, chunk 2 KB) ==")
        h.start()
        cp = h.cp_url()

        print("\n== test 1: deploy the large app (JSON) -> staged in the pool ==")
        r = httpx.post(f"{cp}/api/v1/deployments",
                       json={"name": "bigsite", "app": "site", "replicas": 2, "site": site},
                       timeout=30)
        r.raise_for_status()
        dep_id = r.json()["id"]
        time.sleep(2)
        dep = httpx.get(f"{cp}/api/v1/deployments/{dep_id}", timeout=10).json()
        blob_id = dep.get("site_blob_id")
        assert blob_id, f"large app was NOT pooled (no site_blob_id): {dep.get('site_blob_id')!r}"
        assert dep.get("site") in (None, {}), f"inline site should be cleared once pooled: {dep.get('site')!r}"
        reps = dep["replicas"]
        nodes = {rep["node_name"] for rep in reps if rep["healthy"]}
        assert len(reps) >= 2 and len(nodes) >= 2, f"expected 2 replicas on distinct nodes: {reps}"
        assert blob_id not in user_blob_ids(cp), "deploy bundle leaked into the user's Files listing"
        print(f"  PASS: pooled as {blob_id[:12]}... ; {len(reps)} replicas on {sorted(nodes)}; "
              f"hidden from Files")

        print("\n== test 2: every page + asset hydrated from the pool + served ==")
        checks = [
            ("", "text/html", f"{TOKEN} home"),
            ("about/", "text/html", f"{TOKEN} about"),
            ("assets/js/app.js", "text/javascript", TOKEN),
            ("assets/data.json", "application/json", TOKEN),
        ]
        for path, ctype, needle in checks:
            resp = get(cp, dep_id, path)
            assert resp.status_code == 200, f"/{path} -> HTTP {resp.status_code}"
            got = resp.headers.get("content-type", "")
            assert ctype in got, f"/{path} content-type {got!r} lacks {ctype!r}"
            assert needle in resp.text, f"/{path} body missing {needle!r}"
            print(f"  ok  /{path or '(root)'}  [{got}]")
        fav = get(cp, dep_id, "favicon.png")
        assert fav.status_code == 200 and fav.content == _PNG, "favicon bytes not served intact"
        print("  PASS: all pages + css/js/json/png hydrated from the pool + served")

        print("\n== test 3: deploy the SAME large app via the browser .zip upload ==")
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel in ["index.html", "about/index.html",
                        "assets/js/app.js", "assets/data.json", "favicon.png"]:
                zf.write(site_root / rel, rel)
        up = httpx.post(f"{cp}/api/v1/deployments/upload?name=bigzip&replicas=2",
                        content=zbuf.getvalue(),
                        headers={"content-type": "application/zip"}, timeout=30)
        up.raise_for_status()
        zdep = up.json()["id"]
        time.sleep(2)
        zinfo = httpx.get(f"{cp}/api/v1/deployments/{zdep}", timeout=10).json()
        assert zinfo.get("site_blob_id"), f"uploaded large app was not pooled: {zinfo.get('site_blob_id')!r}"
        zjs = get(cp, zdep, "assets/js/app.js")
        assert zjs.status_code == 200 and "text/javascript" in zjs.headers.get("content-type", "") \
            and TOKEN in zjs.text, f"uploaded app.js not served: {zjs.status_code}"
        assert get(cp, zdep, "about/").status_code == 200, "uploaded nested route not served"
        print(f"  PASS: uploaded .zip pooled as {zinfo['site_blob_id'][:12]}... ; routes served")

        print("\n== test 4: kill a hosting node -> re-hydrated FROM THE POOL (self-heal) ==")
        victim = reps[0]["node_name"]
        victim_idx = int(victim.split("-")[-1]) - 1
        print(f"  killing {victim} ...")
        h.stop_agent(victim_idx)
        dist = served_all(cp, dep_id, 21)
        good = sum(v for k, v in dist.items() if k.startswith("sim-node"))
        print(f"  21 requests across 3 routes after kill: {dict(dist)}")
        assert good == 21, f"large app dropped requests during failover: {dist}"

        deadline = time.time() + 30
        healed = False
        live: set = set()
        while time.time() < deadline:
            dep = httpx.get(f"{cp}/api/v1/deployments/{dep_id}", timeout=10).json()
            healthy = [x for x in dep["replicas"] if x["healthy"]]
            live = {x["node_name"] for x in healthy}
            if len(healthy) >= 2 and victim not in live:
                healed = True
                break
            time.sleep(1)
        assert healed, "manager did not self-heal the pooled large deployment"
        final = served_all(cp, dep_id, 9)
        assert sum(v for k, v in final.items() if k.startswith("sim-node")) == 9, \
            f"pages not served after re-hydration from the pool: {final}"
        print(f"  PASS: self-healed onto {sorted(live)}; bundle re-hydrated from the pool + served")

        print("\nALL LARGE-APP POOLED-BUNDLE E2E CHECKS PASSED")
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
