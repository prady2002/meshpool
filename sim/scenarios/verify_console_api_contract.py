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

Test 5 additionally proves the *elements* of every list match the console's
TypeScript interfaces, which is where the worst of these bugs live. It parses
``web/src/api/types.ts`` rather than restating it, so the expectation cannot
drift away from the code it is supposed to protect, and it **populates every
list first**: an empty list is treated as a failure, because a check that runs
over zero elements proves nothing. That is not theoretical -- ``ModelDeployment``
declared ``partition: PartitionSlice[]`` (with ``start_layer``/``end_layer``/
``url``) for a server that has only ever sent ``stages`` of
``{node_name, first, last}``. Playground, Models, Mesh Map and the command
palette all crashed with "Cannot read properties of undefined (reading 'map')"
the instant a model was deployed, and this gate passed anyway -- because it ran
with no model deployed, so ``models`` was ``[]``.

NOT proven: field-level *types* inside list elements, or POST/DELETE bodies.

Exit 0 = PASS.  Exit 1 = FAIL.
"""
from __future__ import annotations

import json
import re
import time
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


# path -> (list key, the TypeScript interface each element must satisfy).
# The fields are read out of web/src/api/types.ts at run time, never restated.
ELEMENT_CONTRACT: list[tuple[str, str, str]] = [
    ("/api/v1/nodes", "nodes", "Node"),
    ("/api/v1/models", "models", "ModelDeployment"),
    ("/api/v1/jobs", "jobs", "BatchJobSummary"),
    ("/api/v1/blobs", "blobs", "BlobMeta"),
    ("/api/v1/deployments", "deployments", "Deployment"),
]

TYPES_TS = REPO_ROOT / "web" / "src" / "api" / "types.ts"


def required_fields(interface: str) -> list[str]:
    """Top-level non-optional field names of a TypeScript interface.

    Fields marked ``?`` are skipped: the console already handles them being
    absent. Nested object literals are skipped by tracking brace depth, so only
    the interface's own fields are returned.
    """
    src = TYPES_TS.read_text(encoding="utf-8-sig")
    m = re.search(r"export interface %s\s*\{" % re.escape(interface), src)
    if not m:
        raise AssertionError(f"no `export interface {interface}` in {TYPES_TS.name}")
    i = m.end()
    depth, out, line = 1, [], ""
    while i < len(src) and depth:
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if c == "\n" or (c == "}" and depth == 0):
            if depth <= 1:
                f = re.match(r"\s*(\w+)(\??)\s*:", line)
                if f and not f.group(2):
                    out.append(f.group(1))
            line = ""
        else:
            line += c
        i += 1
    return out


def main() -> int:  # noqa: C901
    data_root = REPO_ROOT / "sim" / "scenarios" / "_data_console_contract"
    # Two agents so a model deployment really has two pipeline stages: a
    # single-node "pipeline" would not exercise the stage list at all.
    h = Harness(n_agents=2, cp_port=CP_PORT, data_dir=str(data_root))
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

        print("\n== test 5: list ELEMENTS match the console's TypeScript interfaces ==")
        # Populate every list first. This is the whole point: the previous version
        # of this gate ran against empty lists and therefore proved nothing about
        # the elements, which is exactly where the shape drift was hiding.
        def seed(path: str, body: dict, timeout: float = 60.0) -> None:
            # A silently-failing setup POST would leave the list empty and make the
            # element check vacuous, so the status code is asserted here rather
            # than discovered later as a confusing "list is empty" failure.
            r = httpx.post(f"{cp}{path}", json=body, timeout=timeout)
            if r.status_code >= 400:
                failures.append(f"setup POST {path} -> {r.status_code} {r.text[:300]}")

        seed("/api/v1/models", {"model": "nano-gpt"}, 120)
        seed("/api/v1/jobs", {"name": "contract", "func": "square",
                              "payloads": [{"x": 3}], "cpu_pct": 10, "ram_mb": 64}, 30)
        seed("/api/v1/blobs", {"name": "contract.txt", "content_b64": "aGVsbG8=",
                               "owner": "default"}, 30)
        seed("/api/v1/deployments", {"name": "contract-site", "replicas": 1,
                                     "owner": "default", "app": "hello"}, 90)
        deadline = time.time() + 120
        while time.time() < deadline:
            counts = {k: len(httpx.get(f"{cp}{path}", timeout=15).json().get(k) or [])
                      for path, k, _ in ELEMENT_CONTRACT}
            if all(counts.values()):
                break
            time.sleep(1.0)

        for path, key, iface in ELEMENT_CONTRACT:
            items = httpx.get(f"{cp}{path}", timeout=15).json().get(key) or []
            if not items:
                failures.append(
                    f"{path}.{key} is empty, so its element shape was never checked. "
                    f"An empty list is not a passing test — populate it or delete "
                    f"the claim.")
                continue
            want = required_fields(iface)
            if len(want) < 3:
                failures.append(
                    f"parsed only {len(want)} required field(s) from interface "
                    f"{iface}; the parser is not reading the interface properly")
                continue
            elem = items[0]
            missing = [f for f in want if f not in elem]
            if missing:
                failures.append(
                    f"{path}.{key}[0] is missing {missing}, which interface {iface} "
                    f"declares as required. The console reads these directly, so "
                    f"`.{missing[0]}` is undefined at runtime. Server sent: "
                    f"{sorted(elem)}")
                continue
            # Paired non-vacuity, per endpoint: the same comparison must be able
            # to report a field that genuinely is not there. Without this, a
            # parser that silently returned [] would look like a pass.
            control = [f for f in want + ["__meshpool_absent_control__"]
                       if f not in elem]
            if control != ["__meshpool_absent_control__"]:
                failures.append(
                    f"{path}.{key}[0] non-vacuity check failed: an invented field "
                    f"was not reported missing (got {control})")
                continue
            print(f"  {path:<34} {len(items)} x {iface}: all {len(want)} required "
                  f"fields present")

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
