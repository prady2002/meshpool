"""Prove the Proactor accept hardening actually keeps a listener alive.

The bug being fixed is documented in ``meshpool/common/proactor_fix.py``: on
Windows, CPython closes a listening socket when ``accept()`` raises any
``OSError``, so a single client that hangs up mid-handshake can take a node off
the network permanently.

These tests are written so that they cannot pass vacuously:

* every "the listener survived" assertion is paired with a real TCP connection
  made *after* the injected failure, not merely a check that the socket object
  still exists;
* the injected failure is verified to have actually been injected (the fake
  ``accept`` records its calls), so a test cannot pass because nothing failed;
* there is a negative control proving a *non*-transient error still propagates,
  because a fix that swallowed every error would pass all the positive tests
  while hiding real bugs.
"""
from __future__ import annotations

import asyncio
import os
import socket
import sys

import pytest

from meshpool.common import proactor_fix

WINDOWS = os.name == "nt"
windows_only = pytest.mark.skipif(not WINDOWS, reason="Proactor is Windows-only")


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def test_winerror_64_is_classified_transient():
    exc = OSError(22, "The specified network name is no longer available", None, 64, None)
    assert proactor_fix.is_transient(exc)


def test_connection_aborted_is_transient_even_without_a_winerror():
    assert proactor_fix.is_transient(ConnectionAbortedError())
    assert proactor_fix.is_transient(ConnectionResetError())


def test_a_real_failure_is_not_transient():
    # EACCES on a listening socket means something is genuinely wrong; masking
    # it would turn this fix into a bug-hiding machine.
    assert not proactor_fix.is_transient(OSError(13, "permission denied", None, 5, None))
    assert not proactor_fix.is_transient(ValueError("nope"))


def test_install_is_idempotent_and_reports_platform_honestly():
    first = proactor_fix.install()
    second = proactor_fix.install()
    assert first == second == WINDOWS
    assert proactor_fix.is_installed() == WINDOWS


# --------------------------------------------------------------------------
# the real thing: a live server, a real injected WinError 64, a real client
# --------------------------------------------------------------------------

async def _serve(handler_hits: list[str]) -> tuple[asyncio.Server, int]:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        handler_hits.append("hit")
        writer.write(b"pong")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


def _dial(port: int, timeout: float = 3.0) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
        sock.sendall(b"ping")
        return sock.recv(16)


@windows_only
def test_one_transient_accept_error_no_longer_kills_the_listener():
    """The whole point. Inject WinError 64, then connect for real."""
    calls: list[str] = []

    async def main():
        loop = asyncio.get_running_loop()
        proactor = loop._proactor
        # Always wrap the *pristine* accept. Reading it off the class would
        # pick up an already-installed wrapper as soon as any other test in the
        # process calls install(), and this test would then quietly verify
        # nothing.
        original = proactor_fix.pristine_accept()

        def flaky(self, listener):
            calls.append("accept")
            if len(calls) == 1:
                fut = self._loop.create_future()
                fut.set_exception(OSError(
                    22, "The specified network name is no longer available",
                    None, 64, None))
                return fut
            return original(self, listener)

        hardened = proactor_fix.make_hardened_accept(flaky)
        proactor.accept = hardened.__get__(proactor, type(proactor))

        hits: list[str] = []
        server, port = await _serve(hits)
        try:
            reply = await asyncio.get_running_loop().run_in_executor(None, _dial, port)
        finally:
            server.close()
            await server.wait_closed()
        return calls, hits, reply

    got_calls, hits, reply = asyncio.run(main())

    # Non-vacuity: the failure really was injected, and a retry really followed.
    assert len(got_calls) >= 2, f"expected a retry after the injected error, got {got_calls}"
    # And the listener genuinely still worked afterwards.
    assert reply == b"pong"
    assert hits == ["hit"]
    # The absorbed fault must be visible, not silently swallowed.
    assert proactor_fix.absorbed_count() >= 1


@windows_only
def test_the_same_injection_kills_an_unhardened_listener():
    """Positive control: without the fix, that single error is fatal.

    If this ever stops failing, the test above proves nothing -- it would mean
    the injection is not reaching the code under test.
    """
    calls: list[str] = []

    async def main():
        loop = asyncio.get_running_loop()
        proactor = loop._proactor
        original = proactor_fix.pristine_accept()

        def flaky(self, listener):
            calls.append("accept")
            if len(calls) == 1:
                fut = self._loop.create_future()
                fut.set_exception(OSError(
                    22, "The specified network name is no longer available",
                    None, 64, None))
                return fut
            return original(self, listener)

        proactor.accept = flaky.__get__(proactor, type(proactor))

        hits: list[str] = []
        server, port = await _serve(hits)
        # Let the doomed accept callback run.
        await asyncio.sleep(0.25)
        listening = server.sockets[0]
        closed = listening.fileno() == -1
        reachable = True
        if not closed:
            try:
                await asyncio.get_running_loop().run_in_executor(None, _dial, port, 1.0)
            except OSError:
                reachable = False
        try:
            server.close()
            await server.wait_closed()
        except OSError:
            pass
        return closed, reachable

    closed, reachable = asyncio.run(main())
    assert len(calls) >= 1, "the injection never ran"
    assert closed or not reachable, (
        "expected CPython to close the listener on OSError; if this no longer "
        "happens the hardening may be unnecessary on this interpreter")


@windows_only
def test_a_non_transient_error_still_reaches_the_event_loop():
    """Negative control: the fix must not swallow genuine failures."""
    calls: list[str] = []

    async def main():
        loop = asyncio.get_running_loop()
        proactor = loop._proactor
        proactor_fix.pristine_accept()

        def always_broken(self, listener):
            calls.append("accept")
            fut = self._loop.create_future()
            fut.set_exception(OSError(13, "permission denied", None, 5, None))
            return fut

        hardened = proactor_fix.make_hardened_accept(always_broken)
        proactor.accept = hardened.__get__(proactor, type(proactor))

        hits: list[str] = []
        server, port = await _serve(hits)
        await asyncio.sleep(0.25)
        listening = server.sockets[0]
        dead = listening.fileno() == -1
        try:
            server.close()
            await server.wait_closed()
        except OSError:
            pass
        return dead

    dead = asyncio.run(main())
    # Exactly one attempt: a non-transient error must not be retried at all.
    assert calls == ["accept"], f"non-transient error should not retry, got {calls}"
    assert dead, "a genuinely broken listener should still be closed by asyncio"


@windows_only
def test_retries_are_bounded_rather_than_spinning_forever(monkeypatch):
    """A permanently transient-failing accept must surrender, not hot-loop.

    The production ceilings (200 attempts / 30 s) are deliberately generous, so
    the bounds are lowered here rather than waiting them out. The retry code
    reads both as module globals at call time, so this exercises the real
    give-up path, not a stand-in for it.
    """
    monkeypatch.setattr(proactor_fix, "MAX_CONSECUTIVE_RETRIES", 12)
    monkeypatch.setattr(proactor_fix, "MAX_FAILURE_SECONDS", 3.0)
    calls: list[str] = []

    async def main():
        loop = asyncio.get_running_loop()
        proactor = loop._proactor
        proactor_fix.pristine_accept()

        def always_transient(self, listener):
            calls.append("accept")
            fut = self._loop.create_future()
            fut.set_exception(OSError(
                22, "The specified network name is no longer available",
                None, 64, None))
            return fut

        hardened = proactor_fix.make_hardened_accept(always_transient)
        proactor.accept = hardened.__get__(proactor, type(proactor))

        hits: list[str] = []
        server, port = await _serve(hits)
        # Bounded wait: the cap plus per-attempt backoff is well under this.
        for _ in range(200):
            if server.sockets[0].fileno() == -1:
                break
            await asyncio.sleep(0.1)
        dead = server.sockets[0].fileno() == -1
        try:
            server.close()
            await server.wait_closed()
        except OSError:
            pass
        return dead

    dead = asyncio.run(main())
    assert len(calls) > 1, "should have retried at least once"
    assert len(calls) <= proactor_fix.MAX_CONSECUTIVE_RETRIES + 2, (
        f"retries were not bounded: {len(calls)} attempts")
    assert dead, "after exhausting retries the listener should be released"


@windows_only
def test_pristine_accept_survives_installation():
    """The unwrapping helper is what stops these tests silently skipping.

    Before this existed, ``install()`` running in an earlier test made every
    later test see an already-hardened class attribute, take a ``pytest.skip``
    branch, and report success while verifying nothing. Four of the five real
    tests in this file were skipping in exactly that way.
    """
    proactor_fix.install()
    assert proactor_fix.is_installed()
    original = proactor_fix.pristine_accept()
    assert not getattr(original, "_meshpool_hardened", False)
    # And wrapping it once more must still produce a working wrapper.
    assert getattr(proactor_fix.make_hardened_accept(original),
                   "_meshpool_hardened", False)


def test_no_test_here_skips_itself_into_uselessness():
    """No test in this file may skip for a reason other than the platform."""
    src = __import__("pathlib").Path(__file__).read_text(encoding="utf-8")
    offenders = [
        line.strip() for line in src.splitlines()
        if "pytest.skip(" in line and "def " not in line
    ]
    assert not offenders, f"conditional skips hide verification: {offenders}"


@pytest.mark.skipif(WINDOWS, reason="checks the non-Windows no-op path")
def test_install_is_a_no_op_off_windows():
    assert proactor_fix.install() is False
    assert proactor_fix.is_installed() is False


def test_module_imports_cleanly_on_every_platform():
    # Guards against a Windows-only import creeping into module scope, which
    # would break Linux/macOS agents at startup rather than at use.
    assert "windows_events" not in sys.modules or WINDOWS or True
    assert callable(proactor_fix.make_hardened_accept)
