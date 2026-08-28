"""Acceptance gate: unmodified official OpenAI SDK against MeshPool.

This is intentionally narrower than the broader OpenAI API gate: it proves the
developer migration claim that an app can keep the official client and only swap
``base_url`` to MeshPool. It also proves streaming is not one buffered dump by
requiring more than one non-empty SDK stream delta.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sim.harness import Harness
from sim import mutation_guard  # noqa: E402

CP_PORT = 8562
DATA_DIR = str(Path(__file__).resolve().parent / "_data_openai_compat")
_INNER_RUN = os.environ.get("MESHPOOL_OPENAI_COMPAT_MUTATION") == "1"


def _fail(message: str) -> None:
    print(f"  FAIL: {message}")
    raise SystemExit(1)


def _ok(message: str) -> None:
    print(f"  PASS: {message}")


def _session(cp: str) -> str:
    r = httpx.post(f"{cp}/api/v1/auth/local", timeout=10)
    r.raise_for_status()
    return r.json()["session_token"]


def _deploy(cp: str, model: str = "nano-gpt") -> None:
    r = httpx.post(f"{cp}/api/v1/models", json={"model": model}, timeout=120)
    r.raise_for_status()
    deadline = time.time() + 120
    while time.time() < deadline:
        view = httpx.get(f"{cp}/api/v1/models", timeout=10).json()
        if view.get("status") == "ready":
            _ok(f"{model} deployed")
            return
        time.sleep(1)
    _fail(f"{model} did not become ready")


def verify_unmodified_sdk(cp: str) -> None:
    print("\n== official OpenAI SDK, base_url only ==")
    from openai import OpenAI

    client = OpenAI(api_key=_session(cp), base_url=f"{cp}/v1", max_retries=0)

    models = client.models.list()
    ids = {m.id for m in models.data}
    if "nano-gpt" not in ids:
        _fail(f"models.list did not expose nano-gpt; got {sorted(ids)}")
    _ok(f"models.list returned {len(models.data)} model(s)")

    chat = client.chat.completions.create(
        model="nano-gpt",
        messages=[{"role": "user", "content": "Say mesh"}],
        max_tokens=16,
        temperature=0.4,
    )
    content = chat.choices[0].message.content or ""
    if not content:
        _fail("chat.completions.create returned empty content")
    if chat.usage is None or chat.usage.total_tokens <= 0:
        _fail(f"chat usage must be present and positive, got {chat.usage!r}")
    _ok(f"chat.completions.create returned content and {chat.usage.total_tokens} tokens")

    completion = client.completions.create(
        model="nano-gpt",
        prompt="MeshPool",
        max_tokens=12,
        temperature=0.4,
    )
    if not completion.choices or not completion.choices[0].text:
        _fail("completions.create returned empty text")
    _ok("legacy completions.create works")

    stream = client.chat.completions.create(
        model="nano-gpt",
        messages=[{"role": "user", "content": "stream several chunks"}],
        max_tokens=24,
        temperature=0.4,
        stream=True,
    )
    parts: list[str] = []
    for chunk in stream:
        if not chunk.choices:
            _fail(f"stream chunk missing choices: {chunk!r}")
        delta = chunk.choices[0].delta
        if delta and delta.content:
            parts.append(delta.content)
    if len(parts) < 2:
        _fail(f"streaming must produce incremental content chunks; got {len(parts)}")
    if not "".join(parts):
        _fail("streaming produced no content")
    _ok(f"streaming produced {len(parts)} incremental content chunk(s)")


def run_nonvacuity_check() -> None:
    print("\n== non-vacuity: official SDK must reject malformed stream fields ==")
    target = REPO_ROOT / "meshpool" / "controlplane" / "openai_api.py"
    original = target.read_bytes()
    before = hashlib.sha256(original).hexdigest()
    anchor = b'return f"data: {json.dumps(chunk)}\\n\\n"'
    replacement = b'return "data: {not-json}\\n\\n"'
    mutated = original.replace(anchor, replacement, 1)
    if mutated == original:
        _fail("mutation anchor not found in openai_api.py")
    _mg = mutation_guard.journal(target, original)
    target.write_bytes(mutated)
    try:
        env = dict(os.environ)
        env["MESHPOOL_OPENAI_COMPAT_MUTATION"] = "1"
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True, errors="replace",
            timeout=420,
            env=env,
        )
    finally:
        target.write_bytes(original)
        mutation_guard.clear(_mg)

    after = hashlib.sha256(target.read_bytes()).hexdigest()
    if after != before:
        _fail(f"openai_api.py was not restored byte-identically: {before} != {after}")
    print(f"  restored openai_api.py sha256={before}")
    if result.returncode == 0:
        _fail("mutated malformed stream still passed; stdout:\n" + result.stdout[-800:])
    _ok(f"mutated stream failed as expected with exit {result.returncode}")


def main() -> int:
    harness = Harness(n_agents=3, cp_port=CP_PORT, data_dir=DATA_DIR, scale="home")
    try:
        print("== starting MeshPool cluster ==")
        harness.start()
        cp = harness.cp_url()
        _deploy(cp)
        verify_unmodified_sdk(cp)
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAIL: unexpected {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        harness.stop()

    if not _INNER_RUN:
        run_nonvacuity_check()
    print("\nPASS: unmodified OpenAI SDK works with MeshPool by changing base_url")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
