"""Deliberate misbehaviour for testing the MeshPool trust / verification system.

This module is the *single, explicit home* for all fault injection used in tests and
the simulation harness.  It is intentionally **not** imported by any production code
path unless both opt-in guards pass.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PACKAGING NOTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The constant ``_FAULTINJECT_STRIPPED`` below is the hook a future CI/packaging
step can use to hard-disable this module entirely:

    sed -i 's/_FAULTINJECT_STRIPPED = False/_FAULTINJECT_STRIPPED = True/' ...

or via a build-time rewrite rule.  When ``True``, ``active_profile()`` always
returns an inert ``FaultProfile`` and the module prints nothing.  Any toolchain
(pyinstaller spec, Dockerfile, Makefile) that packages a production binary should
set this flag before bundling.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACTIVATION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fault injection activates **only when ALL of the following hold**:

1. ``_FAULTINJECT_STRIPPED`` is False (not a stripped production build).
2. Either:
   a. ``MESHPOOL_ALLOW_FAULT_INJECTION=1`` is set in the environment, OR
   b. The code is running under pytest (``PYTEST_CURRENT_TEST`` is set) or the
      sim harness (``MESHPOOL_SIM_HARNESS=1``).  This keeps legacy test suites
      and the trust scenario working without requiring every caller to opt in
      explicitly.
3. The specific fault variable (e.g. ``MESHPOOL_LIE=1``) is also set.

In a real deployment none of conditions 2a / 2b are true, so even if someone
accidentally sets ``MESHPOOL_LIE=1`` in production, nothing happens.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# ── Production build marker ────────────────────────────────────────────────
# Packaging step: flip to True (or rewrite to `True`) to hard-disable all faults.
_FAULTINJECT_STRIPPED: bool = False

log = logging.getLogger("meshpool.faultinject")

# Set only through set_stall_ms(), which enforces the opt-in guard. None means
# "fall back to the environment variable".
_runtime_stall_ms: "Optional[float]" = None

# Set only through set_corrupt_blobs(). None means "fall back to the environment".
_runtime_corrupt_blobs: "Optional[bool]" = None


def _opt_in() -> bool:
    """True if this process is allowed to use fault injection at all."""
    if _FAULTINJECT_STRIPPED:
        return False
    if os.environ.get("MESHPOOL_ALLOW_FAULT_INJECTION", "").strip().lower() in (
        "1", "true", "yes", "on"
    ):
        return True
    # Auto-allow under pytest or the sim harness so legacy callers don't need the
    # explicit opt-in flag — but a production agent (no PYTEST_CURRENT_TEST, no
    # MESHPOOL_SIM_HARNESS) is never implicitly opted in.
    if "PYTEST_CURRENT_TEST" in os.environ:
        return True
    if os.environ.get("MESHPOOL_SIM_HARNESS", "").strip().lower() in (
        "1", "true", "yes", "on"
    ):
        return True
    return False


# ── Fault profile ──────────────────────────────────────────────────────────

@dataclass
class FaultProfile:
    """Describes which deliberate misbehaviours are active on this node.

    All fields default to False / inactive so a default-constructed profile is safe
    to use anywhere (it just does nothing).
    """
    # Return a deterministically corrupted result instead of the real one.
    corrupt_results: bool = False
    # Silently drop a work unit (pretend it succeeded but return nothing).
    drop_work: bool = False
    # Stall for a configurable number of seconds before responding.
    stall_seconds: float = 0.0
    # Report a fabricated (inflated) capability / FLOPS figure.
    lie_capability: bool = False
    # Serve well-formed but WRONG bytes for a content-addressed blob. Distinct from
    # corrupt_results: the node stays healthy, the response parses, and only the
    # content hash reveals the lie -- which is the exact attack content addressing
    # is supposed to defeat, and therefore the only honest way to test it.
    corrupt_blobs: bool = False

    @property
    def any_active(self) -> bool:
        return (
            self.corrupt_results
            or self.drop_work
            or self.stall_seconds > 0
            or self.lie_capability
            or self.corrupt_blobs
        )


# ── Module-level active profile ───────────────────────────────────────────

def active_profile() -> FaultProfile:
    """Return the currently active fault profile for this process.

    If the opt-in guard is not satisfied, returns an inert (all-False) profile
    regardless of what environment variables say.  This is the function daemon.py
    calls instead of reading MESHPOOL_LIE directly.
    """
    if not _opt_in():
        return FaultProfile()

    lie = os.environ.get("MESHPOOL_LIE", "").strip().lower() in ("1", "true", "yes", "on")
    return FaultProfile(corrupt_results=lie, stall_seconds=_stall_seconds(),
                        corrupt_blobs=_corrupt_blobs_active())


def _corrupt_blobs_active() -> bool:
    if _runtime_corrupt_blobs is not None:
        return _runtime_corrupt_blobs
    return os.environ.get("MESHPOOL_CORRUPT_BLOBS", "").strip().lower() in (
        "1", "true", "yes", "on")


def set_corrupt_blobs(on: Optional[bool]) -> bool:
    """Make this node serve wrong bytes for content-addressed blobs (None = clear).

    Returns False when not opted in, for the same reason set_stall_ms does: a
    scenario that believes it planted a dishonest peer, and did not, reports a
    passing integrity test that never had anything dishonest in it.
    """
    global _runtime_corrupt_blobs
    if not _opt_in():
        log.warning("refusing to enable blob corruption: fault injection is not enabled")
        return False
    _runtime_corrupt_blobs = None if on is None else bool(on)
    log.warning("FAULT INJECTION ACTIVE - corrupt_blobs=%r "
                "(test-only dishonest-peer emulation; never active in production)",
                _runtime_corrupt_blobs)
    return True


def corrupt_bytes(raw: bytes, profile: FaultProfile | None = None) -> bytes:
    """Return bytes that are the right *shape* but the wrong *content*.

    Same length, same base64-decodability, one byte different. Anything more
    obvious -- truncating, emptying, returning garbage -- would be caught by a
    length or parse check and would let a puller that never hashes anything pass.
    """
    p = profile if profile is not None else active_profile()
    if not p.corrupt_blobs or not raw:
        return raw
    log.warning("FAULT INJECTION ACTIVE - corrupt_blobs: serving %d WRONG bytes "
                "(test-only dishonest-peer emulation; never active in production)",
                len(raw))
    return bytes([raw[0] ^ 0xFF]) + raw[1:]


def _stall_seconds() -> float:
    """Milliseconds of artificial delay per shard call.

    A straggler is not a failure -- the node answers, correctly, just late -- so it
    cannot be emulated by killing anything. On consumer hardware this is the *normal*
    case (thermal throttling, someone launches a game, Wi-Fi degrades), and in a
    pipeline one slow stage caps the throughput of every stage. Emulating it needs a
    knob that makes a node slow while leaving it perfectly healthy.

    A runtime override takes precedence over ``MESHPOOL_STALL_MS`` so a scenario can
    make a node slow *while it is already serving*. Starting it slow would prove much
    less: the interesting claim is that a healthy deployment notices a node that
    degrades under it and routes around it, and that is only observable if there was
    a healthy baseline first.
    """
    if _runtime_stall_ms is not None:
        return max(0.0, _runtime_stall_ms) / 1000.0
    raw = os.environ.get("MESHPOOL_STALL_MS", "").strip()
    if not raw:
        return 0.0
    try:
        ms = float(raw)
    except ValueError:
        log.warning("ignoring unparseable MESHPOOL_STALL_MS: %r", raw)
        return 0.0
    return max(0.0, ms) / 1000.0


def set_stall_ms(ms: Optional[float]) -> bool:
    """Set (or clear, with None) the runtime stall. False when not opted in.

    Refusing rather than silently ignoring matters: a scenario that believes it made
    a node slow, and did not, will report a passing straggler test that never had a
    straggler in it.
    """
    global _runtime_stall_ms
    if not _opt_in():
        log.warning("refusing to set a runtime stall: fault injection is not enabled")
        return False
    _runtime_stall_ms = None if ms is None else max(0.0, float(ms))
    log.warning("FAULT INJECTION ACTIVE - stall set to %r ms "
                "(test-only straggler emulation; never active in production)",
                _runtime_stall_ms)
    return True


def stall(profile: FaultProfile | None = None) -> float:
    """Block for the configured stall, returning the seconds actually spent."""
    p = profile if profile is not None else active_profile()
    if p.stall_seconds <= 0:
        return 0.0
    time.sleep(p.stall_seconds)
    return p.stall_seconds


# ── Fault application helpers ─────────────────────────────────────────────

def corrupt_result(v: Any, profile: FaultProfile | None = None) -> Any:
    """Return a deterministically wrong version of ``v`` when corrupt_results is set.

    Emits a loud warning on every call so no activation ever goes unnoticed in logs.
    """
    p = profile if profile is not None else active_profile()
    if not p.corrupt_results:
        return v
    log.warning(
        "FAULT INJECTION ACTIVE — corrupt_results: returning WRONG answer "
        "(test-only dishonest-node emulation; never active in production)"
    )
    return _corrupt(v)


def _corrupt(v: Any) -> Any:
    """Deterministically turn a correct result into a plausible-but-wrong one.

    Mirrors the original ``_corrupt_result`` logic in daemon.py exactly so existing
    tests that assert on specific corrupted values continue to pass.
    """
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v + 1
    if isinstance(v, str):
        return v + "X"
    if isinstance(v, list):
        return v[1:] if v else [1]
    if isinstance(v, dict):
        return {k: _corrupt(x) for k, x in v.items()}
    return v
