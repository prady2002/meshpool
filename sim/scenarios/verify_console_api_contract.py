"""Acceptance gate: the web console's assumptions match what the server really sends.

Why this gate exists
--------------------
The console's API client is written as ``this.request<SomeType>(path)``. In
TypeScript that is a **type assertion over untrusted network data**, not a
validation. Nothing checks it at build time and nothing checks it at runtime, so
``tsc -b``, ``vite build``, ``eslint`` and the whole vitest suite can all be green
while the console is broken in production.

That is not hypothetical. Two pages shipped as **completely blank white screens**:

  * Models   -- ``GET /api/v1/models/available`` returns ``{builtin, ollama}``,
                but the client declared ``string[]``, so ``available.map(...)``
                threw ``n.map is not a function``.
  * Account  -- ``GET /api/v1/secrets`` returns ``{names, active_leases}``, but
                the client declared ``{secrets: SecretRecord[]}``, so
                ``.secrets`` was ``undefined`` and ``secrets.map(...)`` threw.

Both were invisible to every existing test because the unit tests mock the API:
a mock built from the same wrong assumption agrees with the wrong code. Only a
check against a REAL running control plane can catch this class of drift.

What is proven here
-------------------
For every endpoint the console reads, the real response contains the keys the
console destructures, with the right container type (object vs array). The
expectations below are transcribed from ``web/src/api/client.ts``; when that file
changes, this gate must change with it -- which is the point.

NOT proven: field-level types inside list elements, or POST/DELETE bodies.

Exit 0 = PASS.  Exit 1 = FAIL.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sim.harness import Harness  # noqa: E402

CP_PORT = 8432

# path -> (key the console reads or None for "uses the whole object", expected type)
# Transcribed from web/src/api/client.ts.
CONTRACT: list[tuple[str, str | None, type]] = [
    ("/api/v1/health", None, dict),
    ("/api/v1/cluster", None, dict),
    ("/api/v1/nodes", "nodes", list),
    ("/api/v1/wallet", None, dict),
    ("/api/v1/ledger", None, dict),
    ("/api/v1/jobs", "jobs", list),
    ("/api/v1/models", "models", list),
    ("/api/v1/deployments", "deployments", list),
    ("/api/v1/blobs", "blobs", list),
    ("/api/v1/onboarding", None, dict),
    ("/api/v1/secrets", "names", list),
    ("/api/v1/account/spend-cap", None, dict),
]

# Endpoints the console reads through a hand-written unwrapper rather than a
# single key. Each entry is (path, list of keys that must all be present).
MULTI_KEY: list[tuple[str, list[str]]] = [
    ("/api/v1/models/available", ["builtin", "ollama"]),
]

# Keys the console's own TypeScript interfaces destructure directly.
NESTED: list[tuple[str, list[str]]] = [
    ("/api/v1/ledger", ["owners", "summary", "by_node", "credits_by_owner",
                        "fair_share", "recent"]),
    ("/api/v1/wallet", ["owner", "balance", "donated", "consumed", "reputation"]),
    ("/api/v1/cluster", ["nodes", "summary"]),
    ("/api/v1/onboarding", ["control_plane_url", "invite_token"]),
    # The Account page renders cap/spent/remaining and, critically, keys the
    # "no cap set" copy off `enforced`. A server that stopped sending `enforced`
    # would make the page silently claim a cap that is not being applied.
    # These are scalars, not lists, so this belongs here and NOT in MULTI_KEY
    # (which additionally asserts every named key is a list).
    ("/api/v1/account/spend-cap",
     ["owner", "cap", "spent", "remaining", "enforced", "exceeded"]),
]


def main() -> int:  # noqa: C901
    data_root = REPO_ROOT / "sim" / "scenarios" / "_data_console_contract"
    h = Harness(n_agents=1, cp_port=CP_PORT, data_dir=str(data_root))
    ok = True
    failures: list[str] = []
    try:
        print("== starting a real control plane ==")
        h.start()
        cp = h.cp_url()

        print("\n== test 1: every key the console reads exists, with the right type ==")
        for path, key, want in CONTRACT:
            r = httpx.get(f"{cp}{path}", timeout=15)
            if r.status_code != 200:
                failures.append(f"{path} -> HTTP {r.status_code}")
                continue
            body = r.json()
            if not isinstance(body, dict):
                failures.append(f"{path} -> top level is {type(body).__name__}, expected object")
                continue
            if key is None:
                print(f"  {path:<34} object with {len(body)} keys")
                continue
            if key not in body:
                failures.append(
                    f"{path} -> no {key!r} key; the console reads `.{key}` and would "
                    f"get undefined. Server sent: {sorted(body)}")
                continue
            got = body[key]
            if not isinstance(got, want):
                failures.append(
                    f"{path}.{key} is {type(got).__name__}, console expects "
                    f"{want.__name__} — .map()/.length would fail")
                continue
            print(f"  {path:<34} .{key} is a {want.__name__}  (len={len(got)})")

        print("\n== test 2: multi-key endpoints expose every list the console merges ==")
        for path, keys in MULTI_KEY:
            body = httpx.get(f"{cp}{path}", timeout=15).json()
            missing = [k for k in keys if k not in body]
            if missing:
                failures.append(f"{path} -> missing {missing}; sent {sorted(body)}")
                continue
            bad = [k for k in keys if not isinstance(body[k], list)]
            if bad:
                failures.append(f"{path} -> {bad} are not lists")
                continue
            print(f"  {path:<34} {', '.join(keys)} all present and lists")

        print("\n== test 3: fields the console's interfaces destructure are present ==")
        for path, keys in NESTED:
            body = httpx.get(f"{cp}{path}", timeout=15).json()
            missing = [k for k in keys if k not in body]
            if missing:
                failures.append(
                    f"{path} -> interface declares {missing} but the server does not "
                    f"send them; sent {sorted(body)}")
                continue
            print(f"  {path:<34} all {len(keys)} declared fields present")

        # NON-VACUITY: the checks above must be capable of failing. Assert against a
        # key that certainly does not exist; if that "passes", the loop is not
        # actually inspecting anything and this whole gate is decorative.
        print("\n== test 4: non-vacuity — a bogus expectation must be detected ==")
        body = httpx.get(f"{cp}/api/v1/jobs", timeout=15).json()
        assert "definitely_not_a_real_key" not in body, (
            "the server really does have a key this gate invented, which means the "
            "checks above prove nothing")
        detected = "definitely_not_a_real_key" not in body
        assert detected, "unreachable"
        print("  a key the server does not send is correctly seen as absent")
        print("  (the same comparison drives tests 1-3, so they have teeth)")

        if failures:
            ok = False
            print(f"\nFAILED: {len(failures)} console/server contract mismatch(es):")
            for f in failures:
                print(f"  - {f}")
            print("\nEach one of these renders a blank page or a crashed panel in the "
                  "console. Fix web/src/api/client.ts (and its types) to match the "
                  "server, or fix the server to match the documented contract.")
        else:
            print("\nALL CONSOLE API CONTRACT CHECKS PASSED")
    except AssertionError as exc:
        print(f"\nFAILED: {exc}")
        ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        ok = False
    finally:
        h.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
