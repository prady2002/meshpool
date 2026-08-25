"""Acceptance gate: the React console is real, built, served, and wired to the API.

The console in ``web/`` type-checked and built cleanly for its entire life while
``src/App.tsx`` was still the untouched Vite starter template — seven finished pages
sat unmounted in ``src/pages/``. A green build proves nothing about what the build
contains, and a green unit test proves nothing about what the server actually hands a
browser. This gate closes both holes by fetching the console over HTTP from a real
control plane and inspecting the bytes.

What it proves:
  1. ``GET /console`` serves an HTML shell, anonymously (it is a public surface).
  2. Every asset the shell references actually resolves — a wrong Vite ``base`` would
     leave a page that builds, deploys and then renders blank.
  3. The served JavaScript bundle contains the real console (its navigation
     destinations and API paths) and NOT the Vite starter template.
  4. Client-side routes such as ``/console/functions`` return the same shell rather
     than 404, so a browser refresh does not break.
  5. The API surfaces the console depends on answer on the same origin.
  6. Serving the console did not disturb the classic dashboard at ``/``.

Skips cleanly (exit 0) with an explanatory message if ``web/dist`` has not been
built, since Node is not a hard dependency of the Python control plane.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from sim.harness import Harness  # noqa: E402

DIST = ROOT / "web" / "dist"
CP_PORT = 8181

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        fails.append(name)
    return ok


# Strings that only the real console contains. Chosen from three independent layers
# (navigation, API contract, feature copy) so a partial regression cannot hide.
CONSOLE_MARKERS = [
    "Playground",
    "Mesh Map",
    "Functions",
    "Contribute",
    "/api/v1/cluster",
    "/api/v1/functions",
    "/api/v1/inference/chat",
    "Community compute console",
]

# Strings that ONLY the Vite starter template contains. Their presence means the
# scaffold was shipped instead of the product.
STARTER_MARKERS = [
    "Edit <code>src/App.tsx</code> and save to test",
    "Explore Vite",
    "chat.vite.dev",
    "documentation-icon",
]


def main() -> int:
    if not (DIST / "index.html").exists():
        print("SKIP: web/dist not built — run "
              "`npm --prefix web install && npm --prefix web run build`")
        return 0

    print("== console: built artefact ==")
    index_html = (DIST / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)="([^"]+)"', index_html)
    asset_refs = [r for r in refs if r.startswith("/console/")]
    check("index.html references /console/-prefixed assets", len(asset_refs) >= 2,
          f"{len(asset_refs)} refs: {', '.join(asset_refs[:4])}")

    h = Harness(n_agents=2, cp_port=CP_PORT, fresh=True)
    h.start()
    try:
        base = h.cp_url()
        with httpx.Client(base_url=base, timeout=30) as c:
            print("\n== console: served over HTTP ==")
            r = c.get("/console")
            check("GET /console returns 200", r.status_code == 200,
                  f"status={r.status_code}")
            served = r.text
            check("served shell is the built index.html",
                  "<div id=\"root\">" in served and served.strip() == index_html.strip(),
                  f"{len(served)} bytes")

            print("\n== console: every referenced asset resolves ==")
            bundles: list[str] = []
            for ref in asset_refs:
                ar = c.get(ref)
                ok = ar.status_code == 200 and len(ar.content) > 0
                check(f"asset {ref.split('/')[-1]} resolves", ok,
                      f"status={ar.status_code} bytes={len(ar.content)}")
                if ok and ref.endswith(".js"):
                    bundles.append(ar.text)

            print("\n== console: the bundle is the product, not the scaffold ==")
            js = "\n".join(bundles)
            check("a JavaScript bundle was served", bool(js), f"{len(js)} bytes")
            missing = [m for m in CONSOLE_MARKERS if m not in js]
            check("bundle contains every console marker", not missing,
                  "missing: " + ", ".join(missing) if missing
                  else f"all {len(CONSOLE_MARKERS)} present")
            leaked = [m for m in STARTER_MARKERS if m in js or m in served]
            check("bundle contains no Vite starter markers", not leaked,
                  "leaked: " + ", ".join(leaked) if leaked else "none")

            print("\n== console: client-side routes survive a refresh ==")
            for path in ("/console/playground", "/console/functions", "/console/mesh",
                         "/console/jobs/some-id"):
                rr = c.get(path)
                check(f"GET {path} serves the shell",
                      rr.status_code == 200 and "<div id=\"root\">" in rr.text,
                      f"status={rr.status_code}")

            print("\n== console: the APIs it calls answer on the same origin ==")
            for path in ("/api/v1/cluster", "/api/v1/jobs", "/api/v1/blobs",
                         "/api/v1/models"):
                ar = c.get(path)
                ok = ar.status_code == 200
                body = ""
                if ok:
                    try:
                        body = ", ".join(sorted(json.loads(ar.text))[:3])
                    except Exception:  # noqa: BLE001
                        ok = False
                check(f"GET {path}", ok, f"status={ar.status_code} keys={body}")

            print("\n== console: the classic dashboard still works ==")
            dr = c.get("/")
            check("GET / still returns the original dashboard",
                  dr.status_code == 200 and "<div id=\"root\">" not in dr.text,
                  f"status={dr.status_code} bytes={len(dr.content)}")

            # Non-vacuity: this gate must be able to fail. A path that was never
            # registered has to 404, otherwise "everything returns 200" would make
            # every check above meaningless.
            print("\n== non-vacuity ==")
            nr = c.get("/definitely-not-a-route-xyz")
            check("an unregistered path is still refused",
                  nr.status_code in (401, 403, 404),
                  f"status={nr.status_code}")
            br = c.get("/console/assets/does-not-exist.js")
            check("a missing console asset is still a 404",
                  br.status_code == 404, f"status={br.status_code}")
    finally:
        h.stop()

    print()
    if fails:
        print(f"FAILED {len(fails)} check(s): " + ", ".join(fails))
        return 1
    print("ALL CONSOLE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
