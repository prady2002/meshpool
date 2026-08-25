"""Network helpers for real multi-machine LAN deployment.

The single most important thing for a real (non-simulated) pool is that every node
advertises an address the *other* machines can actually reach. On a LAN that means a
routable IPv4 — never ``127.0.0.1``. ``lan_ip()`` discovers that address reliably and
without sending any packets (so it never blocks and never triggers a firewall prompt).
"""
from __future__ import annotations

import functools
import socket


def _is_usable(ip: str) -> bool:
    return bool(ip) and not ip.startswith("127.") and not ip.startswith("169.254.")


@functools.lru_cache(maxsize=1)
def lan_ip() -> str:
    """Best-effort primary LAN IPv4 for this machine.

    Strategy: open a UDP socket and ``connect()`` it toward an off-link address. No
    datagram is actually sent, but the OS picks the interface it *would* route out of,
    and ``getsockname()`` then reveals that interface's local IP. Falls back to
    hostname resolution, then loopback if the machine is genuinely offline.
    """
    for target in ("10.255.255.255", "8.8.8.8", "192.168.0.1"):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((target, 1))
            ip = s.getsockname()[0]
            if _is_usable(ip):
                return ip
        except OSError:
            pass
        finally:
            s.close()

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if _is_usable(ip):
                return ip
    except OSError:
        pass

    return "127.0.0.1"


def is_loopback(ip: str) -> bool:
    return not ip or ip.startswith("127.") or ip == "0.0.0.0"


# In global (reverse-tunnel) mode an agent has no reachable inbound address, so it
# advertises this deliberately-unreachable sentinel instead of a routable URL. The
# coordinator recognises it to mean "reach this node only through its tunnel": any
# direct HTTP call to it is expected to fail, and the scheduler therefore treats such
# a node as reachable only while its reverse tunnel is actually attached.
TUNNEL_SENTINEL_URL = "http://127.0.0.1:1"


def advertised_host(explicit: str | None = None) -> str:
    """Resolve the host to advertise: an explicit value wins, else the detected LAN IP."""
    if explicit:
        return explicit
    return lan_ip()
