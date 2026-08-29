"""Mandatory egress filtering helpers (spec ?6.3).

HONEST SCOPE STATEMENT
----------------------
* Enforcement for curated functions is by construction: they do not open sockets.
* WASM enforcement is by WASI capability: preview 1 has no socket API, so guest
  networking is structurally unavailable.
* ProcessRuntime subprocess enforcement comes from this module monkeypatching
  ``socket.socket.connect`` in the child process before user work runs.
* An attacker with native code execution can bypass this hook by calling OS
  networking directly. True isolation requires OS-level controls such as Linux
  network namespaces or Windows Filtering Platform rules; those stronger controls
  are planned for a later phase.
"""
from __future__ import annotations

import collections
import dataclasses
import ipaddress
import re
import socket
import threading
import time
from urllib.parse import urlparse

HARD_BLOCKED_PORTS: frozenset[int] = frozenset({
    25,    # SMTP relay (spam)
    445,   # SMB lateral movement
    3389,  # RDP brute forcing
    6667,  # IRC botnet control
})

_BLOCKED_NETWORKS: tuple[ipaddress._BaseNetwork, ...] = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)

_HOOK_INSTALLED = False
_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_CONNECT_EX = socket.socket.connect_ex
_ACTIVE_POLICY: "EgressPolicy | None" = None
_LEGACY_IPV4_RE = re.compile(r"^[0-9A-Fa-fxX.]+$")


class EgressDenied(Exception):
    """Raised when a requested outbound connection violates policy."""

    def __init__(self, reason: str, host: str, port: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.host = host
        self.port = int(port)


def _clean_host(host: str) -> str:
    host = str(host).strip()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if "%" in host:
        host = host.split("%", 1)[0]
    return host


def _canonical_ip(addr: ipaddress._BaseAddress) -> ipaddress._BaseAddress:
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    return addr


def _parse_legacy_ipv4(host: str) -> ipaddress.IPv4Address | None:
    """Parse OS-accepted IPv4 spellings such as 2130706433 or 0x7f000001."""
    if not _LEGACY_IPV4_RE.match(host):
        return None
    try:
        return ipaddress.IPv4Address(socket.inet_aton(host))
    except OSError:
        return None


def _parse_ip(host: str) -> ipaddress._BaseAddress | None:
    host = _clean_host(host)
    try:
        return _canonical_ip(ipaddress.ip_address(host))
    except ValueError:
        return _parse_legacy_ipv4(host)


def _is_blocked_ip(addr: ipaddress._BaseAddress) -> bool:
    return any(addr in net for net in _BLOCKED_NETWORKS)


def _resolve_ips(host: str, port: int) -> list[ipaddress._BaseAddress]:
    host = _clean_host(host)
    literal = _parse_ip(host)
    if literal is not None:
        return [literal]
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise EgressDenied(f"unable to resolve destination {host}:{port}", host, port) from exc
    addrs: list[ipaddress._BaseAddress] = []
    seen: set[str] = set()
    for _family, _socktype, _proto, _canonname, sockaddr in infos:
        if not sockaddr:
            continue
        ip_text = str(sockaddr[0])
        if ip_text in seen:
            continue
        seen.add(ip_text)
        addrs.append(_canonical_ip(ipaddress.ip_address(ip_text)))
    if not addrs:
        raise EgressDenied(f"unable to resolve destination {host}:{port}", host, port)
    return addrs


@dataclasses.dataclass(slots=True)
class EgressPolicy:
    whitelist: list[tuple[str, int]] = dataclasses.field(default_factory=list)
    allow_all: bool = False

    def check(self, host: str, port: int) -> None:
        host = _clean_host(host)
        port = int(port)
        if port in HARD_BLOCKED_PORTS:
            raise EgressDenied(f"port {port} is hard-blocked", host, port)
        ips = _resolve_ips(host, port)
        for addr in ips:
            if _is_blocked_ip(addr):
                raise EgressDenied(f"destination {addr} is in a blocked network", host, port)
        if (host, port) in self.whitelist:
            return
        if self.allow_all:
            return
        raise EgressDenied(f"{host}:{port} denied by default", host, port)

    @classmethod
    def from_control_plane_url(cls, cp_url: str) -> "EgressPolicy":
        parsed = urlparse(cp_url)
        host = parsed.hostname
        if not host:
            raise ValueError("control-plane URL must include a hostname")
        if parsed.port is not None:
            port = parsed.port
        elif parsed.scheme == "https":
            port = 443
        else:
            port = 80
        return cls(whitelist=[(host, port)])


class RateLimiter:
    def __init__(self, limit: int, window_s: float) -> None:
        self.limit = max(0, int(limit))
        self.window_s = float(window_s)
        self._by_key: dict[str, collections.deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            dq = self._by_key.setdefault(str(key), collections.deque())
            cutoff = now - self.window_s
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if len(dq) >= self.limit:
                return False
            dq.append(now)
            return True

    def is_allowed(self, key: str) -> bool:
        return self.check(key)


@dataclasses.dataclass(slots=True)
class AbuseReport:
    reporter_ip: str
    node_id: str
    description: str
    evidence: str
    ts: float


class AbuseReportStore:
    def __init__(self, maxlen: int = 1000) -> None:
        self._reports: collections.deque[AbuseReport] = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, report: AbuseReport) -> None:
        with self._lock:
            self._reports.append(report)

    def list_recent(self, n: int = 100) -> list[AbuseReport]:
        with self._lock:
            items = list(self._reports)
        if n <= 0:
            return []
        return items[-n:]

    def count(self) -> int:
        with self._lock:
            return len(self._reports)


def install_egress_hook(policy: EgressPolicy) -> None:
    global _ACTIVE_POLICY, _HOOK_INSTALLED
    _ACTIVE_POLICY = policy
    if _HOOK_INSTALLED:
        return

    def _check_address(address) -> tuple[str, int] | None:
        if isinstance(address, (tuple, list)) and len(address) >= 2 and _ACTIVE_POLICY is not None:
            host = address[0]
            port = address[1]
            host_text = host.decode("utf-8", "replace") if isinstance(host, bytes) else str(host)
            return host_text, int(port)
        return None

    def _guarded_connect(sock: socket.socket, address) -> object:
        checked = _check_address(address)
        if checked is not None:
            host_text, port = checked
            try:
                _ACTIVE_POLICY.check(host_text, port)
            except EgressDenied as exc:
                raise ConnectionRefusedError(str(exc)) from exc
        return _ORIGINAL_CONNECT(sock, address)

    def _guarded_connect_ex(sock: socket.socket, address) -> int:
        checked = _check_address(address)
        if checked is not None:
            host_text, port = checked
            try:
                _ACTIVE_POLICY.check(host_text, port)
            except EgressDenied:
                return 10061
        return _ORIGINAL_CONNECT_EX(sock, address)

    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    _HOOK_INSTALLED = True


__all__ = [
    "AbuseReport",
    "AbuseReportStore",
    "EgressDenied",
    "EgressPolicy",
    "HARD_BLOCKED_PORTS",
    "RateLimiter",
    "install_egress_hook",
    "_BLOCKED_NETWORKS",
]
