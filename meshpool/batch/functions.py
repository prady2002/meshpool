"""Named worker functions the batch engine can dispatch.

A real deployment ships a container image / user function per work unit; for the
runnable MVP we register a handful of CPU-bound demo functions that exercise real
distribution, real CPU usage (so donation caps are observable), and the
"embarrassingly parallel ML/data job" pattern (embeddings).
"""
from __future__ import annotations

import base64 as _b64
import hashlib
import hmac
import math
import re as _re
import time
from typing import Any, Callable


def square(payload: dict) -> Any:
    """Square a number — the smallest possible unit of real work, used in demos."""
    x = payload.get("x", 0)
    return x * x


def is_prime(payload: dict) -> Any:
    """Primality test by trial division — an embarrassingly parallel CPU workload."""
    n = int(payload.get("n", 0))
    if n < 2:
        return {"n": n, "prime": False}
    if n % 2 == 0:
        return {"n": n, "prime": n == 2}
    r = int(math.isqrt(n))
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return {"n": n, "prime": False}
    return {"n": n, "prime": True}


def sha_pow(payload: dict) -> Any:
    """CPU-heavy proof-of-work-ish hash chain — makes CPU donation caps visible."""
    seed = str(payload.get("seed", "")).encode()
    rounds = int(payload.get("rounds", 200_000))
    h = seed
    for _ in range(rounds):
        h = hashlib.sha256(h).digest()
    return {"rounds": rounds, "digest": h.hex()[:16]}


def embed(payload: dict) -> Any:
    """Deterministic pseudo-embedding of text (the 'document embeddings' use case).

    Uses only the stdlib hash of tokens folded into a fixed-dim vector so results
    are reproducible across nodes and verifiable by spot-check (Tier 1 benchmark)."""
    text = str(payload.get("text", ""))
    dim = int(payload.get("dim", 16))
    vec = [0.0] * dim
    for tok in text.split():
        d = hashlib.sha256(tok.lower().encode()).digest()
        for i in range(dim):
            vec[i] += (d[i % len(d)] / 255.0) - 0.5
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return {"dim": dim, "vector": [round(v / norm, 5) for v in vec]}


def sleep_echo(payload: dict) -> Any:
    """Sleeps, then echoes — used to test mid-flight node kill / failover."""
    secs = float(payload.get("seconds", 1.0))
    time.sleep(secs)
    return {"slept": secs, "echo": payload.get("echo")}


def secure_fetch(payload: dict) -> Any:
    """Demo of a workload that needs a **brokered** credential.

    The API token is never in the job definition; it arrives via a single-use,
    node-scoped lease that the agent redeems just-in-time and injects under
    ``_secrets``. The unit signs its request with the token and returns only a
    signature/proof — never the secret itself (spec §5)."""
    secrets_map = payload.get("_secrets", {})
    token = secrets_map.get("API_TOKEN")
    if not token:
        return {"ok": False, "error": "no API_TOKEN brokered to this unit"}
    resource = str(payload.get("resource", "/data"))
    sig = hmac.new(token.encode(), resource.encode(), hashlib.sha256).hexdigest()
    return {"ok": True, "resource": resource, "signature": sig[:32],
            "used_secret": "API_TOKEN"}


REGISTRY: dict[str, Callable[[dict], Any]] = {
    "square": square,
    "is_prime": is_prime,
    "sha_pow": sha_pow,
    "embed": embed,
    "sleep_echo": sleep_echo,
    "secure_fetch": secure_fetch,
}

# --- user-supplied WebAssembly ------------------------------------------- #
# Everything above is a *curated* function: code we wrote, shipped on every node.
# That is a hard ceiling on what the mesh can do — a user cannot run their own
# code on it at all. ``wasm:<sha256>`` lifts that ceiling: the name refers to a
# content-addressed WASM module which the agent has already fetched into its blob
# store (the same peer-pull path that distributes model weights) and hydrated into
# the payload under ``_wasm_module``.
#
# The module bytes travel *in the payload* rather than being read from disk here
# because this function also runs inside the sandbox subprocess, which
# deliberately has no filesystem or network access of its own.
WASM_PREFIX = "wasm:"
_WASM_MODULE_KEY = "_wasm_module"
_HEX64_RE = _re.compile(r"^[0-9a-f]{64}$")


def is_wasm_function(name: str) -> bool:
    return isinstance(name, str) and name.startswith(WASM_PREFIX)


def wasm_module_id(name: str) -> str:
    """Extract and validate the sha256 module id from a ``wasm:<sha256>`` name."""
    mid = name[len(WASM_PREFIX):].strip().lower()
    if not _HEX64_RE.match(mid):
        raise ValueError(
            f"malformed wasm function name {name!r}: expected "
            f"'wasm:<64-hex-char sha256>'")
    return mid


def _run_wasm(name: str, payload: dict) -> Any:
    from ..functions import wasm as _wasm

    module_id = wasm_module_id(name)
    blob = payload.get(_WASM_MODULE_KEY)
    if not blob:
        raise ValueError(
            f"wasm module {module_id[:12]} was not delivered to this node — the "
            f"agent must hydrate it from the pooled blob store before dispatch")
    if isinstance(blob, str):
        blob = _b64.b64decode(blob)
    # Content addressing is the integrity check: a node that served the wrong bytes
    # (corrupt, or malicious) cannot make us execute them.
    actual = hashlib.sha256(blob).hexdigest()
    if actual != module_id:
        raise ValueError(
            f"wasm module hash mismatch: asked for {module_id[:12]}, got "
            f"{actual[:12]} — refusing to execute unverified code")
    # The guest never sees MeshPool's internal plumbing.
    guest_payload = {k: v for k, v in payload.items() if k != _WASM_MODULE_KEY}
    limits = _wasm.Limits()
    result = _wasm.run(blob, guest_payload, limits=limits)
    return result.value


# Upper bounds on each curated function's cost-driving parameter. They sit far above
# any legitimate use (real jobs are unaffected) but stop an abusive payload — e.g.
# ``rounds = 10**12`` or ``seconds = 10**9`` — from burning a donor's CPU/RAM. On the
# public mesh, where we run *strangers'* units, this validation runs inside the
# sandbox worker so a bad input is rejected before it can do any work.
_PARAM_CAPS: dict[str, list[tuple[str, float]]] = {
    "square": [("x", 1e15)],
    "is_prime": [("n", 1e13)],
    "sha_pow": [("rounds", 50_000_000)],
    "embed": [("dim", 8192)],
    "sleep_echo": [("seconds", 30.0)],
}
_MAX_TEXT_CHARS = 200_000


def validate_payload(name: str, payload: dict) -> None:
    """Reject abusive curated-function inputs before they run.

    A no-op for values within the (generous) caps, so existing jobs behave exactly
    as before; raises ``ValueError`` for anything that would let one unit monopolise
    a donor's machine. Type errors are left to the function itself to surface.
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    if is_wasm_function(name):
        # Validates the name shape early so a typo fails fast rather than after the
        # module has been shipped to every node. Guest resource limits are enforced
        # by the WASM runtime itself (fuel + epoch + memory), not by param caps.
        wasm_module_id(name)
        return
    for key, cap in _PARAM_CAPS.get(name, ()):
        if key not in payload:
            continue
        try:
            v = abs(float(payload[key]))
        except (TypeError, ValueError):
            continue
        if v > cap:
            raise ValueError(f"{name}.{key}={payload[key]!r} exceeds the maximum {cap:g}")
    if name == "embed":
        text = payload.get("text", "")
        if isinstance(text, str) and len(text) > _MAX_TEXT_CHARS:
            raise ValueError(f"embed.text exceeds the maximum length ({_MAX_TEXT_CHARS} chars)")
    if name == "secure_fetch":
        res = payload.get("resource", "")
        if isinstance(res, str) and len(res) > 8192:
            raise ValueError("secure_fetch.resource exceeds the maximum length (8192 chars)")


def run_function(name: str, payload: dict) -> Any:
    if is_wasm_function(name):
        validate_payload(name, payload)
        return _run_wasm(name, payload)
    fn = REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"unknown batch function: {name!r}")
    validate_payload(name, payload)
    return fn(payload)
