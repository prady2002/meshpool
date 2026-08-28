"""Detect a process that is alive but no longer serving.

CPython's Windows Proactor event loop closes its own listening socket when
``accept()`` raises any ``OSError``::

    # Lib/asyncio/proactor_events.py, BaseProactorEventLoop._start_serving
    except OSError as exc:
        if sock.fileno() != -1:
            self.call_exception_handler({'message': 'Accept failed on a socket', ...})
            sock.close()

There is no retry and no reopen. The Selector loop, by contrast, treats an
aborted connection as ordinary (``return None``) and backs off and retries on
resource exhaustion, so it never loses the listener. Windows is where our
contributors are, and Proactor is the default event loop there.

The consequence is the worst shape a failure can take: the process stays up,
the supervisor sees a live PID, the logs show one line, and the node silently
accepts nothing ever again. On residential Wi-Fi a reset mid-handshake is not
exotic -- it is Tuesday. This was observed in the wild here as
``[WinError 64] The specified network name is no longer available`` on an
agent's accept loop, after which ``/health`` never answered again.

Swapping the whole process to the Selector loop is not a safe general fix:
``select()`` on Windows caps at 512 sockets, which the control plane will
exceed. So instead of preventing one cause, this guard observes the effect --
"can anything still connect to me?" -- which also covers a wedged event loop,
an exhausted thread pool, or a handler that never returns.

The guard deliberately does nothing until the listener has answered at least
once. A server that is still booting has never been healthy, so there is
nothing to detect yet, and killing it would turn a slow start into a crash
loop.
"""
from __future__ import annotations

import os
import socket
import sys
import threading
from typing import Callable, Optional

DEFAULT_INTERVAL = 5.0
DEFAULT_TIMEOUT = 4.0
DEFAULT_THRESHOLD = 3

# Unbound listen addresses cannot be dialled; loop back to reach ourselves.
_UNROUTABLE = {"", "0.0.0.0", "::", "*"}

# Status strings returned by check_once(), so tests can assert on the state
# machine rather than on side effects.
STARTING = "starting"
OK = "ok"
FAILING = "failing"
DEAD = "dead"


def dial(host: str, port: int, timeout: float) -> bool:
    """True if a TCP connection to the listener completes."""
    target = "127.0.0.1" if host in _UNROUTABLE else host
    try:
        with socket.create_connection((target, port), timeout=timeout):
            return True
    except OSError:
        return False


class ListenerWatchdog:
    """Probe our own listening socket; exit the process if it stops answering.

    ``on_dead`` defaults to a hard ``os._exit``. That is intentional: the whole
    premise is that the event loop can no longer accept connections, so a
    graceful shutdown -- which needs that same loop -- may never complete. A
    process that exits is restarted by the service manager; a process that
    hangs half-dead is not.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        interval: float = DEFAULT_INTERVAL,
        timeout: float = DEFAULT_TIMEOUT,
        threshold: int = DEFAULT_THRESHOLD,
        probe: Optional[Callable[[], bool]] = None,
        on_dead: Optional[Callable[[], None]] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        if threshold < 1:
            raise ValueError("threshold must be at least 1")
        self.host = host
        self.port = port
        self.interval = interval
        self.timeout = timeout
        self.threshold = threshold
        self._probe = probe or (lambda: dial(self.host, self.port, self.timeout))
        self._on_dead = on_dead or self._exit_hard
        self._log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self._ready = False
        self._consecutive_failures = 0
        self._fired = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- state ----------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True once the listener has answered at least once."""
        return self._ready

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    # -- one step -------------------------------------------------------------

    def check_once(self) -> str:
        """Run a single probe and advance the state machine."""
        if self._fired:
            return DEAD
        if self._probe():
            if not self._ready:
                self._ready = True
                self._log(
                    f"[serve-guard] listener on {self.host}:{self.port} is answering; "
                    "watching it from here"
                )
            elif self._consecutive_failures:
                self._log(
                    f"[serve-guard] listener on {self.host}:{self.port} recovered after "
                    f"{self._consecutive_failures} failed probe(s)"
                )
            self._consecutive_failures = 0
            return OK

        if not self._ready:  # never answered yet: still booting, nothing to kill
            return STARTING

        self._consecutive_failures += 1
        if self._consecutive_failures < self.threshold:
            self._log(
                f"[serve-guard] listener on {self.host}:{self.port} did not answer "
                f"({self._consecutive_failures}/{self.threshold})"
            )
            return FAILING

        self._fired = True
        self._log(
            f"[serve-guard] listener on {self.host}:{self.port} stopped accepting "
            f"connections after {self._consecutive_failures} consecutive probes, while "
            "this process is still running. The event loop can no longer serve, so "
            "exiting to let the supervisor restart a working one."
        )
        self._on_dead()
        return DEAD

    # -- thread ---------------------------------------------------------------

    def start(self) -> "ListenerWatchdog":
        if self._thread is not None:
            raise RuntimeError("watchdog already started")
        self._thread = threading.Thread(
            target=self._loop, name="meshpool-serve-guard", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + self.timeout + 1.0)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            if self.check_once() == DEAD:
                return

    # -- default action -------------------------------------------------------

    def _exit_hard(self) -> None:  # pragma: no cover - kills the interpreter
        os._exit(70)
