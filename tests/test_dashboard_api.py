"""Dashboard-facing control-plane API: the zero-signup local sign-in, the unified
ledger route, and Tier-2 wallet gating that the redesigned UI depends on.

These run in-process against a public (Tier-2) app via Starlette's TestClient, so
they are fast and need no subprocesses (the full product path is covered by the slow
``sim.scenarios.verify_ux`` gate).
"""
from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from meshpool.common.config import Settings
from meshpool.controlplane.app import build_app, _is_loopback


def _req(host):
    return SimpleNamespace(client=(None if host is None else SimpleNamespace(host=host)))


def test_is_loopback_classifies_hosts():
    assert _is_loopback(_req("127.0.0.1")) is True
    assert _is_loopback(_req("::1")) is True
    assert _is_loopback(_req("localhost")) is True
    assert _is_loopback(_req("testclient")) is True     # Starlette TestClient peer
    assert _is_loopback(_req(None)) is True              # in-process ASGI, no peer
    assert _is_loopback(_req("203.0.113.7")) is False    # a real remote visitor


def _public_app(tmp_path):
    settings = Settings.load(scale="public", data_dir=str(tmp_path),
                             host="127.0.0.1", port=8099)
    assert settings.trust_tier == 2
    return build_app(settings)


def test_local_signin_mints_session_and_grant(tmp_path):
    with TestClient(_public_app(tmp_path)) as client:
        r = client.post("/api/v1/auth/local")
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["gating"] is True and j["tier"] == 2
        assert j["owner"].startswith("dev:") and j["session_token"]
        assert j["balance"] > 0                          # one-time starter grant


def test_wallet_requires_session_at_tier2(tmp_path):
    with TestClient(_public_app(tmp_path)) as client:
        token = client.post("/api/v1/auth/local").json()["session_token"]
        # no session -> 401
        assert client.get("/api/v1/wallet").status_code == 401
        # with the local session -> the signed-in owner's wallet
        w = client.get("/api/v1/wallet",
                       headers={"X-MeshPool-Session": token}).json()
        assert w["owner"].startswith("dev:") and w["granted"] > 0


def test_ledger_route_serves_both_shapes(tmp_path):
    """Regression guard: owner + node shapes must come from ONE /api/v1/ledger route
    (they used to be two routes on one path, silently shadowing each other)."""
    with TestClient(_public_app(tmp_path)) as client:
        led = client.get("/api/v1/ledger").json()
        for key in ("owners", "summary", "by_node", "credits_by_owner",
                    "fair_share", "recent"):
            assert key in led, f"ledger missing '{key}'"
        assert isinstance(led["by_node"], list)
        assert set(led["summary"]) >= {"owners", "donated", "consumed", "granted"}


def _home_app(tmp_path):
    settings = Settings.load(scale="home", data_dir=str(tmp_path),
                             host="127.0.0.1", port=8098)
    assert settings.trust_tier == 0
    return build_app(settings)


def test_solo_dashboard_load_path_serves_without_a_session(tmp_path):
    """The 'unzip and run' solo experience (Tier 0): the redesigned dashboard, its
    static assets, and every endpoint the page fetches on load must all serve with no
    gating and no session — otherwise a first-time local user sees a broken page."""
    with TestClient(_home_app(tmp_path)) as client:
        root = client.get("/")
        assert root.status_code == 200 and "<!DOCTYPE html>" in root.text[:80]
        for asset in ("/static/app.js", "/static/style.css", "/static/index.html"):
            r = client.get(asset)
            assert r.status_code == 200 and len(r.text) > 100, asset
        # every GET the SPA issues on load works at Tier 0 without a session
        for path in ("/api/v1/cluster", "/api/v1/jobs", "/api/v1/deployments",
                     "/api/v1/models", "/api/v1/onboarding", "/api/v1/ledger"):
            assert client.get(path).status_code == 200, path
        # zero-signup sign-in returns the open 'default' owner, no gating
        s = client.post("/api/v1/auth/local").json()
        assert s["owner"] == "default" and s["gating"] is False and s["tier"] == 0
        # wallet is reachable without a session below Tier 2 (owner query param)
        w = client.get("/api/v1/wallet", params={"owner": "default"})
        assert w.status_code == 200 and w.json()["owner"] == "default"


def _zip(entries: dict[str, bytes]) -> bytes:
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _upload(client, entries, name="my-app", replicas=1, env=None):
    params = [("name", name), ("replicas", replicas)]
    for e in (env or []):
        params.append(("env", e))
    return client.post("/api/v1/deployments/upload",
                       params=params,
                       content=_zip(entries),
                       headers={"Content-Type": "application/zip"})


def test_browser_upload_of_a_procfile_bundle_deploys_a_server_app(tmp_path):
    """A dashboard zip carrying a Procfile is hosted as a real server app (kind=proc),
    not treated as broken static files — this is what lets a non-CLI user host a
    backend across the mesh."""
    with TestClient(_home_app(tmp_path)) as client:
        r = _upload(client, {
            "server.py": b"print('hi')",
            "Procfile": b"web: python -u server.py\n",
        })
        assert r.status_code == 200, r.text
        assert r.json()["kind"] == "proc"


def test_browser_upload_accepts_env_config_for_a_server_app(tmp_path):
    """A browser-deployed server app can be configured with env vars (KEY=VALUE),
    matching the CLI's --env — real backends need DATABASE_URL, API keys, etc."""
    with TestClient(_home_app(tmp_path)) as client:
        r = _upload(client, {
            "server.py": b"print('hi')",
            "Procfile": b"web: python -u server.py\n",
        }, env=["APP_MODE=prod", "DATABASE_URL=postgres://u:p@h/db?x=1"])
        assert r.status_code == 200, r.text
        assert r.json()["kind"] == "proc"


def test_browser_upload_rejects_malformed_env(tmp_path):
    with TestClient(_home_app(tmp_path)) as client:
        r = _upload(client, {"index.html": b"<h1>hi</h1>"}, env=["NOTAVALIDPAIR"])
        assert r.status_code == 400
        assert "KEY=VALUE" in r.json()["detail"]


def test_browser_upload_of_a_static_site_stays_static(tmp_path):
    with TestClient(_home_app(tmp_path)) as client:
        r = _upload(client, {"index.html": b"<h1>hello</h1>"})
        assert r.status_code == 200, r.text
        assert r.json()["kind"] == "site"


def test_browser_upload_without_index_or_procfile_is_rejected(tmp_path):
    """A bundle that is neither a static site nor a server app must be refused with a
    clear message rather than silently deploying something that can never serve."""
    with TestClient(_home_app(tmp_path)) as client:
        r = _upload(client, {"data/readme.txt": b"just some files"})
        assert r.status_code == 400
        assert "index.html" in r.json()["detail"] and "Procfile" in r.json()["detail"]
