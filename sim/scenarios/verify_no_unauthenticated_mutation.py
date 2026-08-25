"""Acceptance gate: every non-GET route rejects an unauthenticated caller
when ``auth_mode=enforced``.

Mutation routes (POST, PUT, DELETE, PATCH) must require authentication — no
unauthenticated caller must be able to modify any state.  GET routes are
out-of-scope for this specific gate (they are covered by verify_authz_matrix.py).

NON-VACUITY, and why this gate used to prove nothing
----------------------------------------------------
``_get_request_principal`` rule 4 grants a *full-access* loopback principal to
any caller from ``_LOOPBACK_HOSTS`` while ``trust_tier < 2`` — and that set
contains ``"testclient"``, the default client host of starlette's TestClient.
An earlier version of this gate built the app at ``scale="home"`` (Tier 0), so
**every request it sent was already authenticated as the machine operator**.
It could not observe an anonymous request at all; its "passes" came from 422
body-validation errors being miscounted as rejections.

Two independent fixes, both applied, so neither alone is load-bearing:

1. Build the app at ``scale="public"`` → ``trust_tier == 2``, where rule 4 does
   not apply at all.
2. Give the TestClient a routable, non-loopback client host, so even a Tier<2
   app would not hand out the loopback grant.

And a positive control: the exact same requests are replayed carrying a *valid*
session token.  If those also come back 401/403 then the 401s prove nothing
about auth (routing could simply be broken), and the gate fails.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient  # noqa: E402
from starlette.routing import Route  # noqa: E402

from meshpool.common.config import Settings  # noqa: E402
from meshpool.controlplane.app import PUBLIC_ALLOWLIST, ROUTE_SCOPES, build_app  # noqa: E402

# Mutation methods we care about
MUTATION_METHODS = {"POST", "PUT", "DELETE", "PATCH"}

# A routable public address (TEST-NET-3, RFC 5737). Deliberately NOT loopback,
# so the loopback auth grant in _get_request_principal cannot apply.
ANON_CLIENT = ("203.0.113.7", 44444)


def _is_public(method: str, path: str) -> bool:
    """True only if the route is *declared* public.

    ``ROUTE_SCOPES.get(key)`` returns ``None`` both for "declared public" and for
    "never declared at all", and an earlier version of this function conflated
    them. That meant a new mutation endpoint added without a ``_declare_route``
    call was silently *skipped* by this gate — the exact endpoint most likely to
    be unprotected was the one least likely to be checked. Undeclared routes are
    now reported separately (see ``_undeclared``) instead of being waved through.
    """
    key = f"{method} {path}"
    return key in ROUTE_SCOPES and ROUTE_SCOPES[key] is None


def _is_declared(method: str, path: str) -> bool:
    return f"{method} {path}" in ROUTE_SCOPES


def _minimal_body_for(method: str, path: str) -> dict:
    """Return a plausible minimal body to avoid 422 validation errors."""
    if "/jobs" in path and method == "POST":
        return {"name": "t", "func": "square",
                "payloads": [{"x": 1}], "cpu_pct": 10}
    if "/deployments" in path and method == "POST" and "upload" not in path:
        return {"name": "d", "replicas": 1}
    if "/models" in path and method == "POST" and "upload" not in path:
        return {"model": "nano-gpt"}
    if "/secrets" in path and method == "POST" and "redeem" not in path:
        return {"name": "k", "value": "v"}
    if "/secrets/redeem" in path:
        return {"lease_id": "x", "node_id": "y"}
    if "/heartbeat" in path:
        return {"node_id": "n", "load": {}, "ts_ms": 0}
    if "/register" in path and "/nodes" in path:
        return {"node_id": "n", "name": "n", "url": "http://x",
                "device_id": "dev_a", "public_key": "a==",
                "signature": "a==", "ts_ms": 0,
                "capability": {"cpu_cores": 1, "ram_mb": 512,
                               "gpu_count": 0, "disk_gb": 0,
                               "ollama_models": []},
                "policy": {}, "enroll_token": ""}
    if "/link/complete" in path:
        return {"pending_id": "x", "proof": "x"}
    if "/link/start" in path:
        return {}
    if "/reclaim" in path or "/resume" in path:
        return {}
    if "/policy" in path:
        return {}
    if "/reply" in path and "tunnel" in path:
        return {"id": "x"}
    if "/auth/device" in path:
        return {"device_id": "dev_a", "public_key": "a==",
                "signature": "a==", "ts_ms": 0}
    if "/auth/local" in path:
        return {}
    return {}


def main() -> int:  # noqa: C901
    ok = True

    print("== verifying: unauthenticated mutation is rejected in enforced mode ==")

    try:
        with tempfile.TemporaryDirectory() as td:
            # scale="public" => trust_tier 2. At Tier 2 the loopback grant in
            # _get_request_principal is switched off entirely, so an anonymous
            # request really is anonymous.
            settings = Settings.load(scale="public", data_dir=td,
                                     host="127.0.0.1", port=9210, auth_mode="enforced")
            if settings.trust_tier < 2:
                print(f"  FAIL: expected trust_tier>=2, got {settings.trust_tier}; "
                      f"the loopback auth grant would make this gate vacuous")
                return 1
            app = build_app(settings)

            # Collect all non-GET routes from the app
            mutation_routes = []
            for route in app.routes:
                if not isinstance(route, Route):
                    continue
                path = route.path
                for method in (route.methods or []):
                    if method.upper() in MUTATION_METHODS:
                        mutation_routes.append((method.upper(), path))

            if not mutation_routes:
                print("  FAIL: no mutation routes found — app may not be wired correctly")
                return 1

            print(f"  Found {len(mutation_routes)} mutation route(s) to check")
            print(f"  Tier {settings.trust_tier}; anonymous caller presents as "
                  f"{ANON_CLIENT[0]} (non-loopback)")

            wrongly_accepted = []
            public_bypassed = []
            validated_first = []
            protected_routes = []
            undeclared = []

            with TestClient(app, client=ANON_CLIENT) as client:
                for method, path in mutation_routes:
                    key = f"{method} {path}"
                    if not _is_declared(method, path):
                        # Not in ROUTE_SCOPES at all. The middleware denies it by
                        # default (404), so nothing is exposed *today* — but the
                        # scope was never reviewed by a human, which is the whole
                        # point of the declaration. Fail loudly rather than skip.
                        undeclared.append(key)
                        continue
                    if _is_public(method, path):
                        # Public routes are expected to accept unauthenticated calls
                        public_bypassed.append(key)
                        continue
                    protected_routes.append((method, path))
                    body = _minimal_body_for(method, path)
                    # client.request(...) rather than client.delete(...): httpx's
                    # DELETE/HEAD helpers take no json= argument.
                    r = client.request(method, path, json=body)
                    if r.status_code in (401, 403):
                        continue
                    if r.status_code == 422:
                        # Body validation ran, which means the request reached the
                        # handler layer. Auth middleware is supposed to reject first,
                        # so this is a real weakness even though nothing was mutated.
                        validated_first.append(f"{key} -> 422")
                        continue
                    wrongly_accepted.append(f"{key} -> {r.status_code}")

                # ---- POSITIVE CONTROL -------------------------------------- #
                # Replay the same protected routes carrying a *valid* credential.
                # If those are ALSO rejected 401/403 then the run above told us
                # nothing about authentication — the routes could simply be
                # unreachable — and this gate would be vacuous.
                import meshpool.controlplane.app as _appmod
                cp = _appmod.CP
                _owner, session = cp.accounts.local_session(cp.identity)
                authed_hdrs = {"X-MeshPool-Session": session}
                authed_pass = 0
                authed_denied = []
                for method, path in protected_routes:
                    body = _minimal_body_for(method, path)
                    r = client.request(method, path, json=body, headers=authed_hdrs)
                    if r.status_code in (401, 403):
                        authed_denied.append(f"{method} {path} -> {r.status_code}")
                    else:
                        authed_pass += 1

            if wrongly_accepted:
                ok = False
                print(f"  FAIL: {len(wrongly_accepted)} mutation route(s) accepted "
                      f"unauthenticated requests:")
                for w in wrongly_accepted:
                    print(f"    {w}")
            if undeclared:
                ok = False
                print(f"  FAIL: {len(undeclared)} mutation route(s) have no "
                      f"_declare_route() entry, so their scope was never reviewed:")
                for u in undeclared:
                    print(f"    {u}")
            if validated_first:
                ok = False
                print(f"  FAIL: {len(validated_first)} mutation route(s) validated the "
                      f"body before checking auth (expected 401/403, got 422):")
                for w in validated_first:
                    print(f"    {w}")

            protected = len(protected_routes)
            if protected < 1:
                ok = False
                print("  FAIL: every mutation route was treated as public — this "
                      "check would pass no matter how broken auth was")

            # The positive control must show the credential actually changes the
            # outcome on a meaningful fraction of routes. Some routes legitimately
            # still 403 for a session principal (node-scoped or admin-scoped ones),
            # so require a majority rather than all.
            if protected >= 1:
                if authed_pass == 0:
                    ok = False
                    print(f"  FAIL: positive control — a VALID session was rejected on "
                          f"all {protected} protected route(s). The 401s above prove "
                          f"nothing about authentication.")
                elif authed_pass * 2 < protected:
                    ok = False
                    print(f"  FAIL: positive control — a valid session only got past "
                          f"{authed_pass}/{protected} protected routes; expected a "
                          f"majority. Denied: {'; '.join(authed_denied[:8])}")
                else:
                    print(f"  Positive control: valid session accepted on "
                          f"{authed_pass}/{protected} protected route(s) "
                          f"(the rest need node/admin scope) — auth is load-bearing")

            if ok:
                print(f"  PASS: {protected} protected mutation route(s) all rejected "
                      f"anonymous callers with 401/403")
                if public_bypassed:
                    print(f"  (skipped {len(public_bypassed)} explicitly-public "
                          f"mutation route(s): {', '.join(public_bypassed)})")

    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()

    if ok:
        print("\nPASS: no unauthenticated mutation is possible in enforced mode")
    else:
        print("\nFAIL: unauthenticated mutation check failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
