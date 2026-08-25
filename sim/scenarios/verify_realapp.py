"""End-to-end scenario: deploy YOUR OWN multi-page app (a real folder) across the mesh.

Where ``verify_site_hosting`` ships a hand-built 2-file site inline, this proves the
"host any multi-page application" path the CLI uses: a real directory tree on disk
(nested routes + JS/CSS/JSON/binary assets) is packed with ``hosting.site.pack_dir``,
deployed across 2 nodes, and served through the control-plane load balancer with:
  * every page reachable (root, a nested ``/about/`` route, a second ``.html`` page),
  * correct Content-Types -- critically ``.js`` -> ``text/javascript`` (browsers
    require it for ES modules; the stdlib mimetypes gets it wrong on Windows),
  * a binary asset (``favicon.png``) served byte-for-byte intact,
and it keeps serving every route after a hosting node is killed (failover + the
durable deployment re-ships the packed site to a survivor on self-heal).
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
from meshpool.hosting import site as sitelib  # noqa: E402

TOKEN = "MESHPOOL-REALAPP-9F3K"          # marker proving OUR packed content is served

# a real 1x1 PNG -- a genuine binary asset to prove bytes survive the round trip
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def build_site(root: Path) -> None:
    """Write a realistic multi-page static site (nested dirs + mixed asset types)."""
    (root / "assets" / "css").mkdir(parents=True, exist_ok=True)
    (root / "assets" / "js").mkdir(parents=True, exist_ok=True)
    (root / "about").mkdir(parents=True, exist_ok=True)
    (root / "index.html").write_text(
        "<!doctype html><html><head><meta charset=utf-8>"
        "<link rel=stylesheet href='assets/css/style.css'>"
        "<script type=module src='assets/js/app.js'></script></head>"
        f"<body><h1>{TOKEN} home</h1>"
        "<a href='about/'>about</a><a href='contact.html'>contact</a></body></html>",
        encoding="utf-8")
    (root / "about" / "index.html").write_text(
        f"<!doctype html><html><body><h1>{TOKEN} about</h1></body></html>", encoding="utf-8")
    (root / "contact.html").write_text(
        f"<!doctype html><html><body><h1>{TOKEN} contact</h1></body></html>", encoding="utf-8")
    (root / "assets" / "css" / "style.css").write_text(
        "h1{color:#0a0;font-family:sans-serif}", encoding="utf-8")
    (root / "assets" / "js" / "app.js").write_text(
        f"export const app='{TOKEN}';console.log(app);", encoding="utf-8")
    (root / "assets" / "data.json").write_text(
        f'{{"app":"{TOKEN}","pages":3}}', encoding="utf-8")
    (root / "favicon.png").write_bytes(_PNG)


def get(cp: str, dep_id: str, path: str):
    return httpx.get(f"{cp}/app/{dep_id}/{path}", timeout=10)


def served_all(cp: str, dep_id: str, n: int) -> Counter:
    """Hit all three pages ``n`` times; count how many returned OUR token."""
    c: Counter = Counter()
    pages = ["", "about/", "contact.html"]
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


def main() -> int:
    base = Path(__file__).resolve().parent / "_data_realapp"
    h = Harness(n_agents=3, cp_port=8112, data_dir=str(base / "cluster"))
    ok = True
    try:
        print("== building a real multi-page site on disk ==")
        site_root = base / "mysite"
        build_site(site_root)
        site = sitelib.pack_dir(str(site_root))
        print(f"  packed {len(site)} files from {site_root}")
        assert "index.html" in site and "about/index.html" in site, "nested route not packed"
        assert "assets/js/app.js" in site and "favicon.png" in site, "assets not packed"

        print("\n== starting cluster ==")
        h.start()
        cp = h.cp_url()

        print("\n== test 1: deploy the packed folder (2 replicas, distinct nodes) ==")
        r = httpx.post(f"{cp}/api/v1/deployments",
                       json={"name": "mysite", "app": "site", "replicas": 2, "site": site},
                       timeout=30)
        r.raise_for_status()
        dep_id = r.json()["id"]
        time.sleep(2)
        dep = httpx.get(f"{cp}/api/v1/deployments/{dep_id}", timeout=10).json()
        reps = dep["replicas"]
        nodes = {rep["node_name"] for rep in reps if rep["healthy"]}
        assert len(reps) >= 2 and len(nodes) >= 2, f"expected 2 replicas on distinct nodes: {reps}"
        print(f"  PASS: {len(reps)} replicas on {sorted(nodes)}")

        print("\n== test 2: every page + asset served with the right Content-Type ==")
        checks = [
            ("", "text/html", f"{TOKEN} home"),
            ("about/", "text/html", f"{TOKEN} about"),
            ("contact.html", "text/html", f"{TOKEN} contact"),
            ("assets/css/style.css", "text/css", "color"),
            ("assets/js/app.js", "text/javascript", TOKEN),   # the Windows MIME fix
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
        assert "image/png" in fav.headers.get("content-type", ""), "favicon wrong content-type"
        print("  ok  /favicon.png  [image/png, bytes intact]")
        missing = get(cp, dep_id, "nope.html")
        assert missing.status_code == 404, f"missing path should 404, got {missing.status_code}"
        print("  PASS: 3 pages + css/js/json/png served correctly; missing -> 404")

        print("\n== test 3: deploy the SAME app via the browser .zip-upload endpoint ==")
        import io
        import zipfile
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel in ["index.html", "about/index.html", "contact.html",
                        "assets/css/style.css", "assets/js/app.js", "assets/data.json",
                        "favicon.png"]:
                zf.write(site_root / rel, rel)
        up = httpx.post(f"{cp}/api/v1/deployments/upload?name=zipsite&replicas=2",
                        content=zbuf.getvalue(),
                        headers={"content-type": "application/zip"}, timeout=30)
        up.raise_for_status()
        zdep = up.json()["id"]
        time.sleep(2)
        zjs = get(cp, zdep, "assets/js/app.js")
        assert zjs.status_code == 200 and "text/javascript" in zjs.headers.get("content-type", "") \
            and TOKEN in zjs.text, f"uploaded app.js not served: {zjs.status_code} {zjs.headers.get('content-type')}"
        zabout = get(cp, zdep, "about/")
        assert zabout.status_code == 200 and TOKEN in zabout.text, "uploaded nested route not served"
        print(f"  PASS: uploaded .zip deployed as {zdep}; nested route + app.js served")

        print("\n== test 3b: upload a Procfile bundle -> hosted as a LIVE server app ==")
        stamp = "REALAPP-STAMP-77"
        proc_server = (
            "import os, http.server, socketserver\n"
            "PORT = int(os.environ['PORT']); PID = os.getpid()\n"
            "STAMP = os.environ.get('APP_STAMP', 'none')\n"
            f"print('{TOKEN} PROC BOOT pid=%d port=%d' % (PID, PORT), flush=True)\n"
            "class H(http.server.BaseHTTPRequestHandler):\n"
            "    def do_GET(self):\n"
            f"        body = ('{TOKEN} PROC alive pid=%d stamp=%s path=%s' % (PID, STAMP, self.path)).encode()\n"
            "        self.send_response(200)\n"
            "        self.send_header('content-type', 'text/plain')\n"
            "        self.send_header('content-length', str(len(body)))\n"
            "        self.end_headers(); self.wfile.write(body)\n"
            "    def log_message(self, *a):\n"
            "        pass\n"
            "socketserver.TCPServer(('127.0.0.1', PORT), H).serve_forever()\n"
        )
        pbuf = io.BytesIO()
        with zipfile.ZipFile(pbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("server.py", proc_server)
            zf.writestr("Procfile", "web: python -u server.py\n")
        pup = httpx.post(f"{cp}/api/v1/deployments/upload?name=zipproc&replicas=2"
                         f"&env=APP_STAMP={stamp}",
                         content=pbuf.getvalue(),
                         headers={"content-type": "application/zip"}, timeout=30)
        pup.raise_for_status()
        pjson = pup.json()
        assert pjson["kind"] == "proc", f"Procfile bundle not detected as a server app: {pjson}"
        pdep = pjson["id"]
        print(f"  uploaded Procfile bundle -> deployed as {pdep} (kind=proc); waiting for boot ...")
        deadline = time.time() + 35
        pids: set[str] = set()
        while time.time() < deadline:
            info = httpx.get(f"{cp}/api/v1/deployments/{pdep}", timeout=10).json()
            if sum(1 for x in info["replicas"] if x["healthy"]) >= 2:
                break
            time.sleep(1)
        for _ in range(12):
            pr = get(cp, pdep, "ping")
            assert pr.status_code == 200 and TOKEN in pr.text and "pid=" in pr.text, \
                f"uploaded server app not serving live: {pr.status_code} {pr.text[:80]!r}"
            assert f"stamp={stamp}" in pr.text, \
                f"browser-supplied env not delivered to the app: {pr.text[:120]!r}"
            pids.add(pr.text.split("pid=")[1].split(" ")[0])
        assert len(pids) >= 2, f"expected the server app to run as >=2 live processes: {pids}"
        plogs = httpx.get(f"{cp}/api/v1/deployments/{pdep}/logs?tail=50", timeout=10).json()
        boot = any("PROC BOOT" in ln for rep in plogs["replicas"] for ln in rep["lines"])
        assert boot, "uploaded server app's boot log was not captured"
        print(f"  PASS: browser-uploaded server app is live across {len(pids)} processes"
              f"; env delivered; boot log captured")

        print("\n== test 4: kill a hosting node -> all pages still served (failover) ==")
        victim = reps[0]["node_name"]
        victim_idx = int(victim.split("-")[-1]) - 1
        print(f"  killing {victim} ...")
        h.stop_agent(victim_idx)
        dist = served_all(cp, dep_id, 21)
        good = sum(v for k, v in dist.items() if k.startswith("sim-node"))
        print(f"  21 requests across 3 pages after kill: {dict(dist)}")
        assert good == 21, f"multi-page app dropped requests during failover: {dist}"

        deadline = time.time() + 25
        healed = False
        live: set[str] = set()
        while time.time() < deadline:
            dep = httpx.get(f"{cp}/api/v1/deployments/{dep_id}", timeout=10).json()
            healthy = [x for x in dep["replicas"] if x["healthy"]]
            live = {x["node_name"] for x in healthy}
            if len(healthy) >= 2 and victim not in live:
                healed = True
                break
            time.sleep(1)
        assert healed, "manager did not self-heal the multi-page deployment"
        final = served_all(cp, dep_id, 9)
        assert sum(v for k, v in final.items() if k.startswith("sim-node")) == 9, \
            f"pages not served after self-heal/re-ship: {final}"
        print(f"  PASS: self-healed onto {sorted(live)}; all pages re-shipped + served")

        print("\nALL REAL-APP HOSTING E2E CHECKS PASSED")
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
