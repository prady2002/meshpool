"""Acceptance gate: every registered route has a declared scope, and the
auth enforcement layer correctly admits/denies each principal class.

Structure follows verify_secrets.py and verify_trust_credits.py exactly:
PASS/FAIL to stdout, exit code 0/1.

This scenario uses the FastAPI TestClient (no external process) so it can run
fast, inspects ``app.routes`` to enumerate every endpoint, and asserts:

 - Every route is in ROUTE_SCOPES (no silent unprotected endpoint).
 - Routes in PUBLIC_ALLOWLIST have scope=None and a justification string.
 - In ``auth_mode=enforced``:
     * anonymous caller → 401 on protected routes, 2xx on public ones.
     * valid session caller → 2xx (or 4xx for business logic) on protected routes.
 - The public allowlist matches the expected set (regression guard).
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient  # noqa: E402
from starlette.routing import Route  # noqa: E402

from meshpool.common.auth import ALL_USER_SCOPES, Scope, TokenIssuer  # noqa: E402
from meshpool.common.config import Settings  # noqa: E402
from meshpool.common.identity import DeviceIdentity, enrollment_challenge  # noqa: E402
from meshpool.common.models import now_ms  # noqa: E402
from meshpool.controlplane.app import (  # noqa: E402
    _MISSING,
    _compile_route_patterns,
    _scope_for_request,
    PUBLIC_ALLOWLIST,
    ROUTE_SCOPES,
    build_app,
)

# Expected public routes (justification gate — add here when you add a new public endpoint)
EXPECTED_PUBLIC = {
    "GET /api/v1/health",
    "GET /api/v1/onboarding",
    "POST /api/v1/auth/device",
    "POST /api/v1/auth/local",
    "POST /api/v1/account/link/start",
    "POST /api/v1/nodes/register",
    "POST /api/v1/nodes/heartbeat",
    "POST /api/v1/secrets/redeem",
    # Abuse reports must be acceptable from people who are not MeshPool users:
    # the party best placed to notice a mesh node attacking something is the
    # operator of the thing being attacked, and they have no account here. An
    # abuse pipeline that only listens to its own customers cannot do the job
    # §6.3 needs it to do. Safe to expose because the handler takes no
    # privileged action -- it rate-limits per IP (10/min) *before* reading the
    # body, truncates the free-text fields, and appends to a deque bounded at
    # 1000 entries for operator review.
    "POST /api/v1/abuse/report",
    "GET /",
    # The React console shell and its assets. Same rationale as "GET /": the
    # bundle is a static client that contains no data — every byte it displays
    # is fetched from the scoped /api/v1 routes, which do their own auth. Serving
    # the shell anonymously is what lets an unauthenticated visitor reach a login
    # screen at all.
    "GET /console",
    "GET /console/{path:path}",
    # Link probing: an agent measures its bandwidth before it has a credential, so
    # these cannot require one. Abuse is bounded by a per-IP byte budget instead.
    "GET /api/v1/probe/download",
    "HEAD /api/v1/probe/download",
    "POST /api/v1/probe/upload",
    # Vivaldi RTT reporting: agents report observed RTTs before they have a
    # session credential (same rationale as heartbeat).
    "POST /api/v1/mesh/rtt",
    # FastAPI's own documentation routes.
    "GET /openapi.json", "HEAD /openapi.json",
    "GET /docs", "HEAD /docs",
    "GET /docs/oauth2-redirect", "HEAD /docs/oauth2-redirect",
    "GET /redoc", "HEAD /redoc",
}


def _session_token(client: TestClient) -> str:
    """Authenticate a fresh device and return its session token."""
    with tempfile.TemporaryDirectory() as td:
        ident = DeviceIdentity.load_or_create(td)
        ts = now_ms()
        sig = ident.sign(enrollment_challenge(ident.device_id, ts))
        r = client.post("/api/v1/auth/device", json={
            "device_id": ident.device_id,
            "public_key": ident.public_key_b64,
            "signature": sig,
            "ts_ms": ts,
        })
        if r.status_code != 200:
            raise RuntimeError(f"auth/device failed: {r.status_code} {r.text}")
        return r.json()["session_token"]


def main() -> int:  # noqa: C901
    ok = True

    # ---- test 1: every route in app.routes appears in ROUTE_SCOPES -------- #
    print("== test 1: every registered route has a declared scope ==")
    with tempfile.TemporaryDirectory() as td:
        settings = Settings.load(scale="home", data_dir=td,
                                 host="127.0.0.1", port=9200, auth_mode="enforced")
        app = build_app(settings)

    if not ROUTE_SCOPES:
        print("  FAIL: ROUTE_SCOPES is empty — _register_routes was not called")
        return 1

    # Collect all HTTP routes from the FastAPI app
    http_routes = []
    for route in app.routes:
        if not isinstance(route, Route):
            continue
        path = route.path
        for method in (route.methods or []):
            http_routes.append((method.upper(), path))

    missing = []
    for method, path in http_routes:
        key = f"{method} {path}"
        if key not in ROUTE_SCOPES:
            missing.append(key)

    # Routes for the multi-method proxy are declared individually
    proxy_methods = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
    missing = [k for k in missing
               if not any(k.startswith(f"{m} /app/") for m in proxy_methods)]

    if missing:
        ok = False
        print(f"  FAIL: {len(missing)} route(s) missing from ROUTE_SCOPES:")
        for m in missing:
            print(f"    {m}")
    else:
        print(f"  PASS: all {len(ROUTE_SCOPES)} route entries declared")

    # ---- test 2: public allowlist has justifications ---------------------- #
    print("\n== test 2: every public route has a justification ==")
    bad = [k for k, v in PUBLIC_ALLOWLIST.items() if not v]
    if bad:
        ok = False
        print(f"  FAIL: {len(bad)} public route(s) with empty justification: {bad}")
    else:
        print(f"  PASS: {len(PUBLIC_ALLOWLIST)} public routes all have justifications")

    # ---- test 3: public allowlist matches expected set --------------------- #
    print("\n== test 3: public allowlist matches expected set ==")
    actual_public = {k for k, v in ROUTE_SCOPES.items() if v is None
                     and not any(k.startswith(f"{m} /app/")
                                 for m in proxy_methods)}
    extra = actual_public - EXPECTED_PUBLIC
    missing_public = EXPECTED_PUBLIC - actual_public
    if extra:
        ok = False
        print(f"  FAIL: unexpected public routes (need justification added to EXPECTED_PUBLIC):")
        for e in sorted(extra):
            print(f"    {e}")
    if missing_public:
        ok = False
        print(f"  FAIL: expected public routes are missing from ROUTE_SCOPES:")
        for m in sorted(missing_public):
            print(f"    {m}")
    if not extra and not missing_public:
        print(f"  PASS: public allowlist matches expected {len(EXPECTED_PUBLIC)} entries")

    # ---- test 4: anonymous caller is rejected on protected routes ---------- #
    print("\n== test 4: anonymous caller rejected on protected routes (enforced mode) ==")
    try:
        with tempfile.TemporaryDirectory() as td:
            # Must use Tier 2 (public) so the TestClient's loopback host doesn't
            # get an automatic loopback principal — at Tier 0 loopback always succeeds
            # (by design, for zero-friction home use), which makes rejection testing
            # impossible from the TestClient.
            settings = Settings.load(scale="public", data_dir=td,
                                     host="127.0.0.1", port=9201, auth_mode="enforced")
            app = build_app(settings)
            with TestClient(app) as client:
                protected_samples = [
                    ("GET", "/api/v1/nodes"),
                    ("GET", "/api/v1/jobs"),
                    ("GET", "/api/v1/secrets"),
                    ("GET", "/api/v1/ledger"),
                    # Note: /api/v1/events is SSE (streaming) — skip to avoid hanging
                ]
                rejected = []
                accepted_wrongly = []
                for method, path in protected_samples:
                    r = getattr(client, method.lower())(path)
                    if r.status_code == 401:
                        rejected.append(f"{method} {path}")
                    else:
                        accepted_wrongly.append(f"{method} {path} -> {r.status_code}")
                if accepted_wrongly:
                    ok = False
                    for a in accepted_wrongly:
                        print(f"  FAIL: anonymous was not rejected: {a}")
                else:
                    print(f"  PASS: {len(rejected)} protected routes all rejected anonymous (401)")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()

    # ---- test 5: public routes remain accessible in enforced mode ---------- #
    print("\n== test 5: public routes accessible without auth (enforced mode) ==")
    try:
        with tempfile.TemporaryDirectory() as td:
            settings = Settings.load(scale="home", data_dir=td,
                                     host="127.0.0.1", port=9202, auth_mode="enforced")
            app = build_app(settings)
            with TestClient(app) as client:
                public_samples = [
                    ("GET", "/api/v1/health"),
                ]
                failed_public = []
                for method, path in public_samples:
                    r = getattr(client, method.lower())(path)
                    if r.status_code >= 400:
                        failed_public.append(f"{method} {path} -> {r.status_code}")
                if failed_public:
                    ok = False
                    for f in failed_public:
                        print(f"  FAIL: public route rejected: {f}")
                else:
                    print(f"  PASS: {len(public_samples)} public route(s) accessible without auth")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  FAIL: {type(exc).__name__}: {exc}")

    # ---- test 6: authenticated session grants access ----------------------- #
    print("\n== test 6: authenticated session grants access to protected routes ==")
    try:
        with tempfile.TemporaryDirectory() as td:
            settings = Settings.load(scale="home", data_dir=td,
                                     host="127.0.0.1", port=9203, auth_mode="enforced")
            app = build_app(settings)
            with TestClient(app) as client:
                token = _session_token(client)
                hdr = {"X-MeshPool-Session": token}
                auth_samples = [
                    ("GET", "/api/v1/nodes"),
                    ("GET", "/api/v1/jobs"),
                    ("GET", "/api/v1/secrets"),
                    ("GET", "/api/v1/ledger"),
                ]
                auth_failures = []
                for method, path in auth_samples:
                    r = getattr(client, method.lower())(path, headers=hdr)
                    if r.status_code == 401:
                        auth_failures.append(f"{method} {path} -> {r.status_code}")
                if auth_failures:
                    ok = False
                    for f in auth_failures:
                        print(f"  FAIL: authenticated caller was rejected: {f}")
                else:
                    print(f"  PASS: {len(auth_samples)} route(s) accessible with session token")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()

    # ---- test 7: EXHAUSTIVE anonymous rejection over every scoped route ---- #
    # Test 4 samples four routes. That is not the promise: the promise is
    # "every endpoint x every principal class". A route can be declared with a
    # scope and still not be enforced, and sampling would never notice.
    print("\n== test 7: EVERY scoped route rejects an anonymous caller ==")
    # Excluded from the *request* sweep only: these two stream indefinitely once
    # a handler is reached. The middleware returns 401 before call_next, so a
    # correct build never reaches the handler -- but if auth regressed, issuing
    # the request would hang the whole gate suite instead of failing it. They are
    # still covered by the declaration check below and by test 8.
    STREAMING = {"GET /api/v1/events", "GET /tunnel/stream"}
    try:
        with tempfile.TemporaryDirectory() as td:
            settings = Settings.load(scale="public", data_dir=td,
                                     host="127.0.0.1", port=9204, auth_mode="enforced")
            app = build_app(settings)
            scoped = {k: v for k, v in ROUTE_SCOPES.items() if v is not None}
            for key in STREAMING:
                if key not in scoped:
                    ok = False
                    print(f"  FAIL: {key} must be a scoped route, got "
                          f"{ROUTE_SCOPES.get(key, '<undeclared>')!r}")
            if len(scoped) < 30:
                ok = False
                print(f"  FAIL: only {len(scoped)} scoped routes found — ROUTE_SCOPES "
                      f"looks truncated; this check would be near-vacuous")
            with TestClient(app) as client:
                swept = 0
                leaked = []
                for key in sorted(scoped):
                    if key in STREAMING:
                        continue
                    method, _, path = key.partition(" ")
                    concrete = re.sub(r"\{[^{}]+\}", "probe", path)
                    r = client.request(method, concrete)
                    swept += 1
                    if r.status_code != 401:
                        leaked.append(f"{key} -> {r.status_code}")
                if leaked:
                    ok = False
                    print(f"  FAIL: {len(leaked)}/{swept} scoped route(s) did not "
                          f"return 401 to an anonymous caller:")
                    for l in leaked:
                        print(f"    {l}")
                elif swept < 30:
                    ok = False
                    print(f"  FAIL: swept only {swept} routes — check is too weak")
                else:
                    print(f"  PASS: all {swept} scoped routes returned 401 to anonymous "
                          f"(+{len(STREAMING)} streaming routes declaration-checked)")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()

    # ---- test 8: route->scope classification is EXACT ---------------------- #
    # The middleware classifies a request by regex-matching ROUTE_SCOPES keys.
    # Two ways that went wrong, both found in production code and both fixed:
    #   * "{name:path}" was compiled to "[^/]+", so /app/<dep>/static/css/x.css
    #     matched nothing, hit deny-by-default and 404'd. Every hosted app that
    #     serves a nested asset path was broken under enforced auth, and no
    #     hosting gate noticed because they all run in auth_mode=open.
    #   * Literals were not regex-escaped, so the "." in /openapi.json was a
    #     wildcard and /openapiXjson inherited its *public* classification.
    # Each case below fails against the pre-fix implementation.
    print("\n== test 8: route classification is exact (no over/under-matching) ==")
    CASES = [
        # (method, path, expected)  expected in {"public", "scoped", "missing"}
        ("GET",  "/app/dep123/index.html",             "public"),
        ("GET",  "/app/dep123/static/css/main.css",    "public"),   # multi-segment {path:path}
        ("GET",  "/app/dep123/a/b/c/d/e.png",          "public"),   # deeply nested
        ("POST", "/app/dep123/api/v1/submit",          "public"),
        ("GET",  "/console/assets/index-abc123.js",    "public"),   # multi-segment {path:path}
        ("GET",  "/openapi.json",                      "public"),
        ("GET",  "/openapiXjson",                      "missing"),  # unescaped '.' wildcard
        ("GET",  "/api/v1/health",                     "public"),
        ("GET",  "/api/v1/nodes",                      "scoped"),
        ("GET",  "/api/v1/jobs/abc123",                "scoped"),
        ("POST", "/api/v1/nodes/n1/reclaim",           "scoped"),
        ("GET",  "/api/v1/nodes/n1/reclaim",           "missing"),  # method must matter
        ("GET",  "/api/v1/nodes/extra/segments/here",  "missing"),  # {param} is single-segment
        ("GET",  "/totally/unknown/route",             "missing"),
    ]
    try:
        with tempfile.TemporaryDirectory() as td:
            settings = Settings.load(scale="public", data_dir=td,
                                     host="127.0.0.1", port=9205, auth_mode="enforced")
            build_app(settings)
        patterns = _compile_route_patterns()
        if not patterns:
            ok = False
            print("  FAIL: _compile_route_patterns() returned nothing")
        else:
            import meshpool.controlplane.app as _appmod
            _prev = _appmod._route_patterns
            _appmod._route_patterns = patterns
            try:
                wrong = []
                for method, path, want in CASES:
                    res = _scope_for_request(method, path)
                    got = ("missing" if res is _MISSING
                           else "public" if res is None else "scoped")
                    if got != want:
                        wrong.append(f"{method} {path} -> want {want}, got {got}")
                if wrong:
                    ok = False
                    print(f"  FAIL: {len(wrong)}/{len(CASES)} misclassified:")
                    for w in wrong:
                        print(f"    {w}")
                else:
                    print(f"  PASS: all {len(CASES)} classification cases exact")
            finally:
                _appmod._route_patterns = _prev
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()

    # ---- test 9: EXHAUSTIVE wrong-scope rejection (authoriZation, not -tication) #
    # Tests 4 and 7 prove an *anonymous* caller is refused. That is authentication.
    # Test 6 proves a *fully* scoped caller is admitted. Between them there was a
    # hole the size of the entire authorization half of the system: no test ever
    # presented a real, signed, unexpired credential that simply did not hold the
    # scope a route demands. Deleting the scope check outright therefore left this
    # whole file green -- confirmed empirically: the `authz-scope-bypass` mutation
    # in verify_mutation_coverage.py SURVIVED until this test existed.
    #
    # The distinction between the two refusal codes is the point:
    #   401 = "I don't know who you are"       (no/expired credential)
    #   403 = "I know exactly who you are, and you may not do this"
    # Asserting 403 (not merely "not 2xx") is what makes this test unable to pass
    # by accident: a broken token would verify-fail and yield 401, which fails here.
    print("\n== test 9: EVERY scoped route refuses a valid credential lacking the scope ==")
    try:
        with tempfile.TemporaryDirectory() as td:
            settings = Settings.load(scale="public", data_dir=td,
                                     host="127.0.0.1", port=9206, auth_mode="enforced")
            app = build_app(settings)
            import meshpool.controlplane.app as _appmod

            scoped = {k: v for k, v in ROUTE_SCOPES.items() if v is not None}
            with TestClient(app) as client:
                # CP is installed by the lifespan handler, not by build_app, so it
                # is only readable once the TestClient context has started the app.
                cp_obj = _appmod.CP
                if cp_obj is None:
                    raise RuntimeError("app startup did not install a ControlPlane")
                # Sign with the CP's own device key: _token_verifier() verifies against
                # exactly this public key, so these are genuine, fully valid credentials
                # -- not forgeries. The ONLY thing wrong with them is the scope set.
                issuer = TokenIssuer(cp_obj.identity._priv, cp_obj.identity.device_id)
                # cluster:admin is deliberately hierarchical (has_scope returns True for
                # every scope when it is held), so it can never serve as a filler.
                fillers = sorted(
                    (s for s in ALL_USER_SCOPES if s is not Scope.CLUSTER_ADMIN),
                    key=lambda s: s.value,
                )
                if len(fillers) < 2:
                    raise RuntimeError(f"need >=2 non-admin scopes, found {len(fillers)}")

                swept = 0
                leaked = []
                for key in sorted(scoped):
                    if key in STREAMING:
                        continue
                    required = scoped[key]
                    filler = next(s for s in fillers if s is not required)
                    tok = issuer.issue(subject="wrong-scope-probe", kind="user",
                                       scopes=frozenset({filler}), ttl_ms=300_000)
                    method, _, path = key.partition(" ")
                    concrete = re.sub(r"\{[^{}]+\}", "probe", path)
                    r = client.request(method, concrete,
                                       headers={"X-MeshPool-Token": tok})
                    swept += 1
                    if r.status_code != 403:
                        leaked.append(f"{key} (needs {required.value}, held "
                                      f"{filler.value}) -> {r.status_code}")
                if leaked:
                    ok = False
                    print(f"  FAIL: {len(leaked)}/{swept} scoped route(s) did not "
                          f"return 403 to a valid credential lacking the scope:")
                    for l in leaked[:25]:
                        print(f"    {l}")
                    if len(leaked) > 25:
                        print(f"    ... and {len(leaked) - 25} more")
                elif swept < 30:
                    ok = False
                    print(f"  FAIL: swept only {swept} routes — check is too weak")
                else:
                    print(f"  PASS: all {swept} scoped routes returned 403 to a valid "
                          f"credential holding the wrong scope")

                # Acceptance half. Without this, "always 403" would pass the sweep
                # above, so the test must be able to fail in BOTH directions: the
                # SAME token machinery, granted the EXACT required scope, must be
                # admitted. Any non-401/403 answer counts -- these carry bogus path
                # params on purpose, so 404/400/422 are correct, expected outcomes.
                admitted = []
                for key in ("GET /api/v1/nodes", "GET /api/v1/jobs",
                            "GET /api/v1/secrets", "GET /api/v1/ledger",
                            "GET /api/v1/models"):
                    required = scoped.get(key)
                    if required is None:
                        ok = False
                        print(f"  FAIL: {key} is not a scoped route — "
                              f"acceptance half is vacuous")
                        continue
                    tok = issuer.issue(subject="right-scope-probe", kind="user",
                                       scopes=frozenset({required}), ttl_ms=300_000)
                    method, _, path = key.partition(" ")
                    r = client.request(method, path,
                                       headers={"X-MeshPool-Token": tok})
                    if r.status_code in (401, 403):
                        admitted.append(f"{key} (holding {required.value}) "
                                        f"-> {r.status_code}")
                if admitted:
                    ok = False
                    print(f"  FAIL: {len(admitted)} route(s) refused a credential "
                          f"holding EXACTLY the required scope — the sweep above "
                          f"may be passing for the wrong reason:")
                    for a in admitted:
                        print(f"    {a}")
                else:
                    print("  PASS: the same tokens, granted the exact required "
                          "scope, are admitted (refusal is caused by scope, "
                          "not by broken credentials)")

                # Expiry must still be 401, never 403: a caller whose credential
                # ran out is unauthenticated, and telling them "forbidden" would
                # send them to fix the wrong problem. This also pins the ordering
                # inside check_scope -- expiry is tested BEFORE scope.
                dead = issuer.issue(subject="expired-probe", kind="user",
                                    scopes=frozenset({Scope.NODES_READ}),
                                    ttl_ms=-60_000)
                r = client.request("GET", "/api/v1/nodes",
                                   headers={"X-MeshPool-Token": dead})
                if r.status_code != 401:
                    ok = False
                    print(f"  FAIL: expired credential got {r.status_code}, "
                          f"expected 401 (expiry is an authentication failure)")
                else:
                    print("  PASS: an expired credential is 401, not 403")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()

    if ok:
        print("\nALL AUTHZ MATRIX CHECKS PASSED")
    else:
        print("\nSOME AUTHZ MATRIX CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
