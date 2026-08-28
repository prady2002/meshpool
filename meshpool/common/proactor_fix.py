"""Stop one aborted connection from permanently killing a Windows listener.

``ListenerWatchdog`` (``serve_guard.py``) *detects* a dead listener. This module
tries to make sure there is nothing to detect.

The bug
-------
CPython's Proactor event loop -- the default on Windows -- destroys its own
listening socket when ``accept()`` fails for any reason::

    # Lib/asyncio/proactor_events.py, BaseProactorEventLoop._start_serving
    def loop(f=None):
        try:
            if f is not None:
                conn, addr = f.result()      # <-- raises OSError
            ...
            f = self._proactor.accept(sock)
        except OSError as exc:
            if sock.fileno() != -1:
                self.call_exception_handler({...})
                sock.close()                 # <-- listener gone. no retry.

No retry, no reopen, and the process carries on looking perfectly healthy.

Why this is severe rather than obscure
--------------------------------------
``[WinError 64] The specified network name is no longer available`` is what a
pending ``AcceptEx`` reports when the peer disappears between the SYN and the
accept completing. That is not an exotic condition. It is a client pressing
Escape, a health check timing out, a laptop lid closing, a port scanner, a load
balancer probe, or residential Wi-Fi doing what residential Wi-Fi does.

So on Windows, **any client that aborts a connection at the wrong moment can
permanently take a MeshPool node off the network**, and the node will not
notice, log anything further, or exit. We observed exactly this during a
78-gate sweep: an agent registered successfully, hit WinError 64 on its accept
loop, and its ``/health`` never answered again.

Why not just use the Selector loop
----------------------------------
``selector_events.py`` handles this correctly -- it returns on
``ConnectionAbortedError`` and re-arms after a delay on resource exhaustion,
never closing the listener. But ``select()`` on Windows is capped at 512 sockets
(``FD_SETSIZE``), and the control plane holds one long-lived tunnel connection
per node, so that ceiling is a scaling wall we would hit on purpose.

The fix
-------
Wrap ``IocpProactor.accept`` so a transient failure is retried *inside* the
returned future. The stdlib's ``loop()`` above only ever sees either a real
connection or a genuinely fatal error, so its ``sock.close()`` path still exists
for errors that really are fatal -- we are not swallowing bugs, only refusing to
treat a hung-up client as one.

Retries are bounded and backed off, and anything that suggests the listener is
actually going away (a closed socket, a cancelled future) stops them
immediately, so a shutdown cannot be resurrected into a spin.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable

# Windows error codes that mean "this particular connection went away", not
# "this listening socket is broken". Retrying accept is always correct here.
TRANSIENT_WINERRORS = frozenset({
    64,     # ERROR_NETNAME_DELETED    - peer vanished mid-handshake
    121,    # ERROR_SEM_TIMEOUT
    995,    # ERROR_OPERATION_ABORTED  - overlapped op cancelled
    1236,   # ERROR_CONNECTION_ABORTED
    10053,  # WSAECONNABORTED
    10054,  # WSAECONNRESET
    10058,  # WSAESHUTDOWN
})

# A hard ceiling so a genuinely broken listener cannot spin forever. Reaching
# it hands control back to CPython's original behaviour, which closes the
# socket -- at which point ListenerWatchdog takes over and restarts us.
# Two ceilings so a genuinely broken listener cannot spin forever. Reaching
# either hands control back to CPython's behaviour, which closes the socket --
# at which point ListenerWatchdog notices and restarts us. The time bound is
# the meaningful one (how long may this node be deaf?); the count is a
# belt-and-braces guard against a pathologically fast failure loop.
MAX_FAILURE_SECONDS = 30.0
MAX_CONSECUTIVE_RETRIES = 200
FIRST_RETRY_DELAY = 0.0
MAX_RETRY_DELAY = 0.25

# Absorbing a fault silently would replace one invisible failure with another,
# so the first one is always reported and the rest are counted.
_absorbed = 0
_reported = False


def absorbed_count() -> int:
    """How many transient accept failures this process has survived."""
    return _absorbed


def _note(exc: BaseException, listener: Any) -> None:
    global _absorbed, _reported
    _absorbed += 1
    if _reported:
        return
    _reported = True
    print(
        f"[proactor-fix] absorbed a transient accept() failure on {listener!r}: "
        f"{exc!r}. CPython would have closed this listening socket permanently; "
        f"retrying instead. Further occurrences are counted, not logged.",
        file=sys.stderr, flush=True,
    )


def is_transient(exc: BaseException) -> bool:
    """True if ``exc`` describes a lost client rather than a lost listener."""
    if isinstance(exc, (ConnectionAbortedError, ConnectionResetError)):
        return True
    if isinstance(exc, OSError):
        return getattr(exc, "winerror", None) in TRANSIENT_WINERRORS
    return False


def make_hardened_accept(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an ``IocpProactor.accept`` so transient errors never reach the loop.

    Exposed separately from :func:`install` so tests can wrap a deliberately
    flaky ``original`` and prove the retry actually happens, rather than
    asserting that a patch was applied.
    """

    def accept(self: Any, listener: Any) -> Any:
        loop = self._loop
        outer = loop.create_future()
        state = {"inner": None, "attempts": 0, "since": 0.0}

        def alive() -> bool:
            try:
                return listener.fileno() != -1
            except OSError:
                return False

        def on_outer_done(fut: Any) -> None:
            # _stop_serving cancels the future it was handed; propagate that to
            # whichever real accept is currently in flight so its provisional
            # connection socket is closed instead of leaked.
            if fut.cancelled():
                inner = state["inner"]
                if inner is not None and not inner.done():
                    inner.cancel()

        outer.add_done_callback(on_outer_done)

        def retry_later() -> None:
            if outer.done():
                return
            state["attempts"] += 1
            if not state["since"]:
                state["since"] = time.monotonic()
            waited = time.monotonic() - state["since"]
            if (state["attempts"] > MAX_CONSECUTIVE_RETRIES
                    or waited > MAX_FAILURE_SECONDS):
                outer.set_exception(OSError(
                    f"accept() failed {state['attempts']} times over "
                    f"{waited:.1f}s on {listener!r}; giving up and letting the "
                    f"listener close"))
                return
            delay = min(MAX_RETRY_DELAY, FIRST_RETRY_DELAY + 0.005 * state["attempts"])
            if delay <= 0:
                loop.call_soon(attempt)
            else:
                loop.call_later(delay, attempt)

        def attempt() -> None:
            if outer.done():
                return
            if not alive():
                outer.set_exception(
                    OSError("listening socket closed while accepting"))
                return
            try:
                inner = original(self, listener)
            except OSError as exc:
                if is_transient(exc) and alive():
                    _note(exc, listener)
                    retry_later()
                else:
                    outer.set_exception(exc)
                return
            except BaseException as exc:  # noqa: BLE001 - must not be swallowed
                outer.set_exception(exc)
                return
            state["inner"] = inner
            inner.add_done_callback(on_inner_done)

        def on_inner_done(inner: Any) -> None:
            if outer.done():
                # Outer already cancelled; make sure the inner result is
                # retrieved so asyncio does not log "never retrieved".
                if not inner.cancelled():
                    inner.exception()
                return
            if inner.cancelled():
                outer.cancel()
                return
            exc = inner.exception()
            if exc is None:
                state["attempts"] = 0
                outer.set_result(inner.result())
                return
            if is_transient(exc) and alive():
                _note(exc, listener)
                retry_later()
            else:
                outer.set_exception(exc)

        attempt()
        return outer

    accept._meshpool_hardened = True  # type: ignore[attr-defined]
    accept._meshpool_original = original  # type: ignore[attr-defined]
    return accept


def pristine_accept() -> Callable[..., Any]:
    """The unwrapped ``IocpProactor.accept``, whether or not we have patched it.

    Tests need to wrap a *known-unhardened* accept to prove the wrapper works.
    Reading it off the class is not enough: once :func:`install` has run
    anywhere in the process, the class attribute is already wrapped, and a test
    that notices this and skips would silently stop verifying the fix. Keeping
    a reference to the original makes that failure mode impossible.
    """
    from asyncio import windows_events
    current = windows_events.IocpProactor.accept
    while getattr(current, "_meshpool_original", None) is not None:
        current = current._meshpool_original  # type: ignore[attr-defined]
    return current


def install() -> bool:
    """Patch ``IocpProactor.accept`` process-wide. Idempotent; safe anywhere.

    Patching the class rather than a loop instance means it does not matter who
    creates the event loop -- uvicorn, asyncio.run, or a test -- which is the
    whole reason it is done this way.

    Returns True if the hardened accept is in place (including when it already
    was), False on any platform or interpreter where it does not apply.
    """
    if os.name != "nt":
        return False
    try:
        from asyncio import windows_events
    except ImportError:  # pragma: no cover - Windows always has it
        return False
    proactor = getattr(windows_events, "IocpProactor", None)
    if proactor is None:  # pragma: no cover - defensive
        return False
    current = getattr(proactor, "accept", None)
    if current is None:  # pragma: no cover - defensive
        return False
    if getattr(current, "_meshpool_hardened", False):
        return True
    proactor.accept = make_hardened_accept(current)
    return True


def is_installed() -> bool:
    """True if the hardened accept is currently patched in."""
    if os.name != "nt":
        return False
    from asyncio import windows_events
    return getattr(getattr(windows_events, "IocpProactor", None), "accept", None) \
        is not None and getattr(windows_events.IocpProactor.accept,
                                "_meshpool_hardened", False)


__all__ = [
    "TRANSIENT_WINERRORS",
    "absorbed_count",
    "is_transient",
    "make_hardened_accept",
    "pristine_accept",
    "install",
    "is_installed",
]
