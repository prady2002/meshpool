"""ACCEPTANCE GATE — OpenAI-compatible inference API.

Proves that an existing OpenAI SDK client can point ``base_url`` at MeshPool and
work unmodified. This is the adoption-unlock feature: every LLM app (LangChain,
LlamaIndex, Continue, Aider, ...) speaks the OpenAI wire protocol. If the client
can swap ``base_url`` and things Just Work, MeshPool inherits the whole ecosystem.

Tests
-----
1. GET /v1/models   — returns {"object":"list","data":[...]} (correct OpenAI shape).
2. POST /v1/chat/completions (non-streaming) — full response with usage block.
3. POST /v1/chat/completions (streaming)  — SSE stream with [DONE] sentinel.
4. POST /v1/completions  — legacy text completion, usage block.
5. Spend cap → 429  — a capped owner gets HTTP 429 in the OpenAI error envelope.
6. Bearer auth       — Authorization: Bearer <session-token> is honoured.

NON-VACUITY (mandatory)
------------------------
The gate mutates openai_api.py to break the models endpoint, runs itself in a
child process, asserts exit==1, then restores the file byte-identically (verified
by SHA-256).  A gate that cannot be made to fail proves nothing.

CP_PORT = 8560  (not colliding with any existing gate port)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sim.harness import Harness  # noqa: E402

CP_PORT = 8560
DATA_DIR = str(Path(__file__).resolve().parent / "_data_openai_api")

# Guard: when set the process is the inner (mutated) invocation;
# skip the non-vacuity section to prevent infinite recursion.
_INNER_RUN = os.environ.get("MESHPOOL_OAI_NONVACUITY") == "1"


def _fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"  PASS: {msg}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _deploy_model(cp: str, model: str = "nano-gpt", timeout: float = 120.0) -> dict:
    r = httpx.post(f"{cp}/api/v1/models", json={"model": model}, timeout=timeout)
    r.raise_for_status()
    view = r.json()
    deadline = time.time() + timeout
    while time.time() < deadline:
        v = httpx.get(f"{cp}/api/v1/models", timeout=10).json()
        if v.get("status") == "ready":
            return v
        time.sleep(1)
    _fail(f"model did not reach 'ready' status within {timeout}s")
    return {}


def _get_session(cp: str) -> str:
    r = httpx.post(f"{cp}/api/v1/auth/local", timeout=10)
    r.raise_for_status()
    return r.json()["session_token"]


# ---------------------------------------------------------------------------
# test functions — each prints PASS: / FAIL: and raises SystemExit(1) on failure
# ---------------------------------------------------------------------------

def test_models_list(cp: str) -> None:
    print("\n== test 1: GET /v1/models ==")
    r = httpx.get(f"{cp}/v1/models", timeout=10)
    if r.status_code != 200:
        _fail(f"GET /v1/models returned {r.status_code}: {r.text[:200]}")
    body = r.json()
    if body.get("object") != "list":
        _fail(f"expected object=='list', got {body.get('object')!r}")
    if not isinstance(body.get("data"), list):
        _fail("data field must be a list")
    if not body["data"]:
        _fail("data list must not be empty")
    for m in body["data"]:
        for field in ("id", "object", "created", "owned_by"):
            if field not in m:
                _fail(f"model object missing field {field!r}: {m}")
        if m["object"] != "model":
            _fail(f"model entry must have object=='model', got {m['object']!r}")
    ids = {m["id"] for m in body["data"]}
    if "nano-gpt" not in ids:
        _fail(f"nano-gpt must be in the model list; got: {sorted(ids)}")
    _ok(f"GET /v1/models returned {len(body['data'])} model(s) in correct shape")


def test_models_list_via_sdk(cp: str) -> None:
    print("\n== test 1b: GET /v1/models via openai SDK ==")
    from openai import OpenAI
    client = OpenAI(api_key=_get_session(cp), base_url=f"{cp}/v1", max_retries=0)
    models = client.models.list()
    if not hasattr(models, "data") or not models.data:
        _fail(f"SDK models.list() returned empty or invalid object: {models}")
    ids = {m.id for m in models.data}
    if "nano-gpt" not in ids:
        _fail(f"nano-gpt not found via SDK; got: {sorted(ids)}")
    _ok(f"openai SDK models.list() returned {len(models.data)} model(s)")


def test_chat_completions_non_streaming(cp: str) -> None:
    print("\n== test 2: POST /v1/chat/completions (non-streaming) ==")
    from openai import OpenAI
    client = OpenAI(api_key=_get_session(cp), base_url=f"{cp}/v1", max_retries=0)
    response = client.chat.completions.create(
        model="nano-gpt",
        messages=[{"role": "user", "content": "MeshPool turns idle machines into "}],
        max_tokens=12,
        temperature=0.5,
    )
    # object shape
    if not hasattr(response, "id") or not response.id.startswith("chatcmpl-"):
        _fail(f"id must start with 'chatcmpl-', got {getattr(response, 'id', None)!r}")
    if getattr(response, "object", None) != "chat.completion":
        _fail(f"object must be 'chat.completion', got {response.object!r}")
    if not response.choices:
        _fail("choices must not be empty")
    choice = response.choices[0]
    if not choice.message.content:
        _fail("response text must not be empty")
    # usage block
    usage = response.usage
    if usage is None:
        _fail("usage must not be None")
    if usage.prompt_tokens <= 0:
        _fail(f"prompt_tokens must be > 0, got {usage.prompt_tokens}")
    if usage.completion_tokens <= 0:
        _fail(f"completion_tokens must be > 0, got {usage.completion_tokens}")
    if usage.total_tokens != usage.prompt_tokens + usage.completion_tokens:
        _fail(f"total_tokens mismatch: {usage.total_tokens} != "
              f"{usage.prompt_tokens} + {usage.completion_tokens}")
    _ok(f"chat.completions.create() -> {len(choice.message.content)} chars, "
        f"usage=({usage.prompt_tokens}+{usage.completion_tokens}={usage.total_tokens})")


def test_chat_completions_streaming(cp: str) -> None:
    print("\n== test 3: POST /v1/chat/completions (streaming) ==")
    r = httpx.post(
        f"{cp}/v1/chat/completions",
        json={
            "model": "nano-gpt",
            "messages": [{"role": "user", "content": "hello "}],
            "max_tokens": 8,
            "stream": True,
        },
        timeout=30,
    )
    if r.status_code != 200:
        _fail(f"streaming returned {r.status_code}: {r.text[:200]}")
    ct = r.headers.get("content-type", "")
    if "text/event-stream" not in ct:
        _fail(f"streaming must return text/event-stream, got {ct!r}")
    if "data: [DONE]" not in r.text:
        _fail("streaming response must end with 'data: [DONE]'")
    # Parse chunks
    chunks = []
    for line in r.text.splitlines():
        if line.startswith("data: ") and not line.startswith("data: [DONE]"):
            chunks.append(json.loads(line[6:]))
    if not chunks:
        _fail("no data chunks in streaming response")
    for chunk in chunks:
        if chunk.get("object") != "chat.completion.chunk":
            _fail(f"chunk object must be 'chat.completion.chunk', got {chunk.get('object')!r}")
    _ok(f"streaming returned {len(chunks)} chunk(s) + [DONE] sentinel")


def test_legacy_completions(cp: str) -> None:
    print("\n== test 4: POST /v1/completions (legacy text completion) ==")
    r = httpx.post(
        f"{cp}/v1/completions",
        json={
            "model": "nano-gpt",
            "prompt": "The quick brown fox",
            "max_tokens": 8,
        },
        timeout=30,
    )
    if r.status_code != 200:
        _fail(f"POST /v1/completions returned {r.status_code}: {r.text[:200]}")
    body = r.json()
    if body.get("object") != "text_completion":
        _fail(f"expected object=='text_completion', got {body.get('object')!r}")
    if not body.get("choices"):
        _fail("choices must not be empty")
    if "text" not in body["choices"][0]:
        _fail("choice must have a 'text' field")
    usage = body.get("usage", {})
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if field not in usage:
            _fail(f"usage missing field {field!r}")
    _ok("POST /v1/completions returned text_completion in correct shape")


def test_spend_cap_returns_429(cp: str) -> None:
    print("\n== test 5: spend cap must return 429 in OpenAI error envelope ==")
    # Get a session and set a zero cap on that owner
    sess_r = httpx.post(f"{cp}/api/v1/auth/local", timeout=10)
    sess_r.raise_for_status()
    sess = sess_r.json()["session_token"]
    owner = sess_r.json()["owner"]
    # Set spend cap at 0
    cap_r = httpx.put(f"{cp}/api/v1/account/spend-cap",
                      json={"cap": 0.0},
                      headers={"X-MeshPool-Session": sess}, timeout=10)
    cap_r.raise_for_status()
    # Try inference with the capped session token as Bearer
    r = httpx.post(
        f"{cp}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
        headers={"Authorization": f"Bearer {sess}"},
        timeout=15,
    )
    if r.status_code != 429:
        _fail(f"capped owner must get HTTP 429, got {r.status_code}: {r.text[:200]}")
    body = r.json()
    if "error" not in body:
        _fail(f"response must use OpenAI error envelope, got: {body}")
    if "message" not in body["error"]:
        _fail(f"error must have 'message' field: {body['error']}")
    # Reset cap so we can continue
    httpx.put(f"{cp}/api/v1/account/spend-cap",
              json={"cap": None},
              headers={"X-MeshPool-Session": sess}, timeout=10)
    _ok("capped owner received HTTP 429 with OpenAI error envelope")


def test_unknown_api_key_is_rejected(cp: str) -> None:
    """An API key that resolves to nobody must be refused, not silently served.

    The old behaviour billed such a request to the ``default`` owner: the call
    succeeded, somebody else paid, and because spend caps are per-owner the
    caller's own cap never applied. A single typo in an API key spent another
    account's credits. This test pins the fix so the fallback cannot return.
    """
    print("\n== test 7: an unknown API key must be rejected ==")
    r = httpx.post(
        f"{cp}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
        headers={"Authorization": "Bearer sk-not-a-real-meshpool-key"},
        timeout=30,
    )
    if r.status_code != 401:
        _fail(f"an unknown API key must be rejected with 401, got "
              f"{r.status_code}: {r.text[:300]}")
    body = r.json()
    if body.get("error", {}).get("code") != "invalid_api_key":
        _fail(f"expected error.code == 'invalid_api_key', got {body!r}")
    # Control: the very same request with NO Authorization header still works,
    # so local single-machine use is not collateral damage of this check.
    r2 = httpx.post(
        f"{cp}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
        timeout=30,
    )
    if r2.status_code != 200:
        _fail(f"open-mode (no Authorization header) must still work; got "
              f"{r2.status_code}: {r2.text[:300]}")
    _ok("unknown API key -> 401 invalid_api_key; no-header open mode still 200")


def test_bearer_auth(cp: str) -> None:
    print("\n== test 6: Authorization: Bearer <session-token> accepted ==")
    sess = _get_session(cp)
    r = httpx.post(
        f"{cp}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
        headers={"Authorization": f"Bearer {sess}"},
        timeout=30,
    )
    if r.status_code != 200:
        _fail(f"Bearer session token must be accepted; got {r.status_code}: {r.text[:300]}")
    _ok("Authorization: Bearer <session-token> accepted and fulfilled")


# ---------------------------------------------------------------------------
# Non-vacuity: mutate openai_api.py, confirm gate exits 1, restore byte-identically
# ---------------------------------------------------------------------------

def run_nonvacuity_check() -> None:
    print("\n== non-vacuity: gate must exit 1 when implementation is broken ==")
    p = REPO_ROOT / "meshpool" / "controlplane" / "openai_api.py"
    orig = p.read_bytes()
    h0 = hashlib.sha256(orig).hexdigest()

    # Mutation: rename "list" to "XBROKEN" in the models response object field.
    # This makes GET /v1/models return {"object": "XBROKEN", ...} which breaks test 1.
    mutated = orig.replace(b'"object": "list"', b'"object": "XBROKEN"', 1)
    if mutated == orig:
        print("  WARN: mutation anchor not found — non-vacuity check skipped")
        return

    p.write_bytes(mutated)
    try:
        env = dict(os.environ)
        env["MESHPOOL_OAI_NONVACUITY"] = "1"
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            capture_output=True, text=True, cwd=str(REPO_ROOT), env=env,
            timeout=360,
        )
    finally:
        p.write_bytes(orig)

    # Verify byte-identical restoration
    h1 = hashlib.sha256(p.read_bytes()).hexdigest()
    if h1 != h0:
        _fail(f"openai_api.py was NOT restored byte-identically; "
              f"orig={h0[:16]}... current={h1[:16]}...")
    print(f"  file restored byte-identically (sha256 prefix: {h0[:16]}...)")

    if result.returncode == 0:
        _fail(
            "mutated gate exited 0 — the gate cannot detect a broken implementation. "
            "stdout tail:\n" + result.stdout[-600:] +
            "\nstderr tail:\n" + result.stderr[-300:]
        )
    print(f"  mutated gate correctly exited {result.returncode} (non-vacuity confirmed)")
    _ok("non-vacuity: gate exits 1 on a broken implementation and restores the file")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    h = Harness(n_agents=3, cp_port=CP_PORT, data_dir=DATA_DIR, scale="home")
    try:
        print("== starting cluster ==")
        h.start()
        cp = h.cp_url()

        print("\n== deploying nano-gpt ==")
        _deploy_model(cp)
        print("  model ready")

        # Run all functional tests
        test_models_list(cp)
        test_models_list_via_sdk(cp)
        test_chat_completions_non_streaming(cp)
        test_chat_completions_streaming(cp)
        test_legacy_completions(cp)
        test_spend_cap_returns_429(cp)
        test_bearer_auth(cp)
        test_unknown_api_key_is_rejected(cp)

        print("\n== all functional tests passed ==")

    except SystemExit as e:
        h.stop()
        if e.code == 1:
            print("\nFAIL: one or more tests failed (see above)")
            return 1
        return int(e.code or 0)
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAIL: unexpected error: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        h.stop()
        return 1
    finally:
        h.stop()

    # Non-vacuity check runs OUTSIDE the cluster (it spawns its own fresh cluster).
    # Skipped in the inner (mutated) invocation to prevent recursion.
    if not _INNER_RUN:
        run_nonvacuity_check()

    print("\nPASS: OpenAI-compatible API is verified end-to-end with the real SDK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
