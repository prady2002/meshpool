"""Tests for the serve guard, and for the defect it exists to catch.

The first thing these tests establish is that the defect is real in *this*
interpreter on *this* platform, because a guard against an imaginary failure is
worse than no guard at all -- it is a maintenance cost that never repays.
"""
from __future__ import annotations

import asyncio
import socket
import sys

import pytest

from meshpool.common.serve_guard import (
    DEAD,
    FAILING,
    OK,
    STARTING,
    ListenerWatchdog,
    dial,
)


# --------------------------------------------------------------------------
# Is the bug real?
# --------------------------------------------------------------------------


def _serve_then_break_accept(break_it: bool):
    """Start a real asyncio server, optionally force one accept() failure.

    Returns (listener_fileno_after, reachable_after).
    """

    async def main():
        loop = asyncio.get_running_loop()

        async def handle(reader, writer):
            writer.close()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        sock = server.sockets[0]
        port = sock.getsockname()[1]

        # Non-vacuity: the server has to work before breaking it means anything.
        assert await asyncio.to_thread(dial, "127.0.0.1", port, 2.0), (
            "the server never served, so this test would prove nothing"
        )

        if break_it:
            real_accept = loop._proactor.accept
            armed = {"yes": True}

            def failing_accept(s):
                # Fail exactly once, the way a connection reset mid-handshake
                # does, then behave normally again.
                if armed["yes"]:
                    armed["yes"] = False
                    fut = loop.create_future()
                    fut.set_exception(
                        OSError(64, "The specified network name is no longer available")
                    )
                    return fut
                return real_accept(s)

            loop._proactor.accept = failing_accept
            loop.set_exception_handler(lambda l, ctx: None)

            # Accepting this connection makes the loop re-arm accept(), which
            # is the call that fails.
            await asyncio.to_thread(dial, "127.0.0.1", port, 2.0)
            await asyncio.sleep(0.5)

        fileno = sock.fileno()
        reachable = await asyncio.to_thread(dial, "127.0.0.1", port, 2.0)
        try:
            server.close()
        except OSError:
            pass
        return fileno, reachable

    return asyncio.run(main())


@pytest.mark.skipif(
    sys.platform != "win32", reason="the Proactor loop is the Windows default"
)
def test_one_accept_error_permanently_kills_a_proactor_listener():
    """A single OSError from accept() closes the listening socket for good.

    This is the defect the watchdog exists for. The process is still alive and
    the event loop is still running -- only the listener is gone, so nothing
    crashes and nothing restarts it.
    """
    fileno, reachable = _serve_then_break_accept(break_it=True)
    assert fileno == -1, "expected the event loop to have closed its listening socket"
    assert not reachable, "expected the server to be unreachable after the accept error"


@pytest.mark.skipif(sys.platform != "win32", reason="pairs with the Proactor case")
def test_the_same_server_is_fine_when_accept_does_not_fail():
    """Positive control: without the induced failure the server keeps serving.

    Without this, the test above would also pass if `asyncio.start_server` were
    simply broken in this environment.
    """
    fileno, reachable = _serve_then_break_accept(break_it=False)
    assert fileno != -1
    assert reachable


# --------------------------------------------------------------------------
# The guard's state machine
# --------------------------------------------------------------------------


def _wd(**kw):
    kw.setdefault("log", lambda m: None)
    return ListenerWatchdog("127.0.0.1", 1, **kw)


def test_a_server_that_has_never_answered_is_never_declared_dead():
    """A slow boot must not be turned into a crash loop."""
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        return False

    fired = []
    wd = _wd(threshold=2, probe=probe, on_dead=lambda: fired.append(1))
    for _ in range(20):
        assert wd.check_once() == STARTING
    assert calls["n"] == 20, "the probe never ran, so nothing was actually tested"
    assert fired == []
    assert not wd.ready


def test_it_fires_only_after_the_threshold_of_consecutive_failures():
    healthy = {"yes": True}
    fired = []
    wd = _wd(threshold=3, probe=lambda: healthy["yes"], on_dead=lambda: fired.append(1))

    assert wd.check_once() == OK
    assert wd.ready
    healthy["yes"] = False
    assert wd.check_once() == FAILING
    assert wd.check_once() == FAILING
    assert fired == [], "fired before reaching the threshold"
    assert wd.check_once() == DEAD
    assert fired == [1]


def test_it_fires_once_and_then_stays_quiet():
    up = {"yes": True}
    fired = []
    wd = _wd(threshold=1, probe=lambda: up["yes"], on_dead=lambda: fired.append(1))
    assert wd.check_once() == OK
    up["yes"] = False
    assert wd.check_once() == DEAD
    for _ in range(5):
        assert wd.check_once() == DEAD
    assert fired == [1], "the guard must not fire repeatedly"


def test_a_recovered_listener_resets_the_failure_count():
    state = {"up": True}
    fired = []
    wd = _wd(threshold=3, probe=lambda: state["up"], on_dead=lambda: fired.append(1))

    assert wd.check_once() == OK
    state["up"] = False
    assert wd.check_once() == FAILING
    assert wd.check_once() == FAILING
    assert wd.consecutive_failures == 2
    state["up"] = True
    assert wd.check_once() == OK
    assert wd.consecutive_failures == 0

    # Positive control: the same number of further failures now must NOT fire,
    # which is only true because the counter reset.
    state["up"] = False
    assert wd.check_once() == FAILING
    assert wd.check_once() == FAILING
    assert fired == []
    assert wd.check_once() == DEAD
    assert fired == [1]


def test_threshold_below_one_is_rejected():
    with pytest.raises(ValueError):
        _wd(threshold=0)


# --------------------------------------------------------------------------
# Against a real socket
# --------------------------------------------------------------------------


def test_a_real_listener_that_closes_underneath_us_is_detected():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port = srv.getsockname()[1]
    fired = []
    wd = ListenerWatchdog(
        "127.0.0.1", port, threshold=2, timeout=1.0,
        on_dead=lambda: fired.append(1), log=lambda m: None,
    )
    try:
        assert wd.check_once() == OK, "a real listening socket should be dialable"
        # Exactly what the Proactor loop does to itself on an accept error.
        srv.close()
        assert wd.check_once() == FAILING
        assert wd.check_once() == DEAD
        assert fired == [1]
    finally:
        srv.close()


def test_a_real_listener_that_stays_up_is_never_declared_dead():
    """Positive control for the test above."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port = srv.getsockname()[1]
    fired = []
    wd = ListenerWatchdog(
        "127.0.0.1", port, threshold=2, timeout=1.0,
        on_dead=lambda: fired.append(1), log=lambda m: None,
    )
    try:
        for _ in range(6):
            assert wd.check_once() == OK
        assert fired == []
    finally:
        srv.close()


def test_an_unbound_bind_host_is_probed_over_loopback():
    """0.0.0.0 is not a dialable address; the guard must translate it."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port = srv.getsockname()[1]
    try:
        assert dial("0.0.0.0", port, 1.0) is True
        assert dial("", port, 1.0) is True
    finally:
        srv.close()
