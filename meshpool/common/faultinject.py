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
from dataclasses import dataclass, field
from typing import Any

# ── Production build marker ────────────────────────────────────────────────
# Packaging step: flip to True (or rewrite to `True`) to hard-disable all faults.
_FAULTINJECT_STRIPPED: bool = False

log = logging.getLogger("meshpool.faultinject")


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

    @property
    def any_active(self) -> bool:
        return (
            self.corrupt_results
            or self.drop_work
            or self.stall_seconds > 0
            or self.lie_capability
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
    return FaultProfile(corrupt_results=lie)


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
