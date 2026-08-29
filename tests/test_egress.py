from __future__ import annotations

import socket
import time
from contextlib import contextmanager

import pytest

import meshpool.common.egress as egress
from meshpool.common.egress import AbuseReport, AbuseReportStore, EgressDenied, EgressPolicy, RateLimiter


@pytest.mark.parametrize("port", [25, 445, 3389, 6667])
def test_hard_blocked_ports_are_always_denied(port: int) -> None:
    with pytest.raises(EgressDenied):
        EgressPolicy(allow_all=True).check("1.2.3.4", port)


def test_hard_blocked_port_25_is_always_denied() -> None:
    with pytest.raises(EgressDenied):
        EgressPolicy(allow_all=True).check("1.2.3.4", 25)


def test_hard_blocked_port_445_is_always_denied() -> None:
    with pytest.raises(EgressDenied):
        EgressPolicy(allow_all=True).check("1.2.3.4", 445)


def test_hard_blocked_port_3389_is_always_denied() -> None:
    with pytest.raises(EgressDenied):
        EgressPolicy(allow_all=True).check("1.2.3.4", 3389)


def test_hard_blocked_port_6667_is_always_denied() -> None:
    with pytest.raises(EgressDenied):
        EgressPolicy(allow_all=True).check("1.2.3.4", 6667)


def test_hard_blocked_port_cannot_be_whitelisted() -> None:
    policy = EgressPolicy(whitelist=[("127.0.0.1", 25)])
    with pytest.raises(EgressDenied):
        policy.check("127.0.0.1", 25)


def test_loopback_blocked_by_default() -> None:
    with pytest.raises(EgressDenied):
        EgressPolicy().check("127.0.0.1", 80)


def test_rfc1918_10_blocked() -> None:
    with pytest.raises(EgressDenied):
        EgressPolicy().check("10.0.0.1", 80)


def test_rfc1918_172_16_blocked() -> None:
    with pytest.raises(EgressDenied):
        EgressPolicy().check("172.16.0.1", 80)


def test_rfc1918_192_168_blocked() -> None:
    with pytest.raises(EgressDenied):
        EgressPolicy().check("192.168.1.1", 80)


def test_link_local_169_254_blocked() -> None:
    with pytest.raises(EgressDenied):
        EgressPolicy().check("169.254.169.254", 80)


def test_whitelist_cannot_override_blocked_network() -> None:
    """A whitelist entry must not re-open a private or link-local destination.

    If it could, a single whitelist entry would become an SSRF primitive against
    the node operator's own LAN and against the cloud metadata endpoint at
    169.254.169.254 — which is exactly how credentials get stolen from a host
    that is otherwise "sandboxed".
    """
    with pytest.raises(EgressDenied):
        EgressPolicy(whitelist=[("127.0.0.1", 8080)]).check("127.0.0.1", 8080)
    with pytest.raises(EgressDenied):
        EgressPolicy(whitelist=[("169.254.169.254", 80)]).check("169.254.169.254", 80)
    # Non-vacuity: the rule above is a targeted restriction, not a blanket denial —
    # a whitelisted *public* destination is still allowed.
    EgressPolicy(whitelist=[("93.184.216.34", 443)]).check("93.184.216.34", 443)


def test_deny_by_default_for_public_ip() -> None:
    with pytest.raises(EgressDenied):
        EgressPolicy().check("1.2.3.4", 80)


def test_allow_all_permits_public_but_still_blocks_private() -> None:
    policy = EgressPolicy(allow_all=True)
    policy.check("1.2.3.4", 80)
    with pytest.raises(EgressDenied):
        policy.check("10.0.0.1", 80)


@pytest.mark.parametrize("host", [
    "0.0.0.0",
    "127.0.0.1",
    "2130706433",
    "0177.0.0.1",
    "0x7f000001",
    "127.1",
    "012.0.0.1",
    "0x0a000001",
    "::1",
    "::ffff:10.0.0.1",
    "[::ffff:10.0.0.1]",
])
def test_blocked_ip_encodings_are_denied_even_when_allow_all(host: str) -> None:
    with pytest.raises(EgressDenied) as exc:
        EgressPolicy(allow_all=True).check(host, 80)
    assert "blocked network" in exc.value.reason


def test_dns_resolution_to_blocked_ip_is_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_getaddrinfo(host, port, *args, **kwargs):
        assert host == "metadata.meshpool.test"
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("169.254.169.254", port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(EgressDenied, match="169.254.169.254"):
        EgressPolicy(allow_all=True).check("metadata.meshpool.test", 80)


def test_whitelisted_dns_name_cannot_rebind_to_private_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_getaddrinfo(host, port, *args, **kwargs):
        assert host == "allowed.meshpool.test"
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(EgressDenied, match="10.0.0.1"):
        EgressPolicy(whitelist=[("allowed.meshpool.test", 443)]).check(
            "allowed.meshpool.test", 443)


@contextmanager
def _temporary_egress_hook(policy: EgressPolicy):
    saved_connect = socket.socket.connect
    saved_connect_ex = socket.socket.connect_ex
    saved_original = egress._ORIGINAL_CONNECT
    saved_original_ex = egress._ORIGINAL_CONNECT_EX
    saved_active = egress._ACTIVE_POLICY
    saved_installed = egress._HOOK_INSTALLED
    calls: list[tuple[str, int]] = []

    def fake_connect(_sock, address):
        calls.append((str(address[0]), int(address[1])))
        return None

    def fake_connect_ex(_sock, address):
        calls.append((str(address[0]), int(address[1])))
        return 0

    try:
        egress._ORIGINAL_CONNECT = fake_connect
        egress._ORIGINAL_CONNECT_EX = fake_connect_ex
        egress._HOOK_INSTALLED = False
        egress.install_egress_hook(policy)
        yield calls
    finally:
        socket.socket.connect = saved_connect
        socket.socket.connect_ex = saved_connect_ex
        egress._ORIGINAL_CONNECT = saved_original
        egress._ORIGINAL_CONNECT_EX = saved_original_ex
        egress._ACTIVE_POLICY = saved_active
        egress._HOOK_INSTALLED = saved_installed


def test_socket_hook_positive_control_allows_whitelisted_public_destination() -> None:
    with _temporary_egress_hook(EgressPolicy(whitelist=[("93.184.216.34", 443)])) as calls:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(("93.184.216.34", 443))
    assert calls == [("93.184.216.34", 443)]


def test_socket_hook_connect_ex_positive_control_allows_whitelisted_public_destination() -> None:
    with _temporary_egress_hook(EgressPolicy(whitelist=[("93.184.216.34", 443)])) as calls:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            err = sock.connect_ex(("93.184.216.34", 443))
    assert err == 0
    assert calls == [("93.184.216.34", 443)]


def test_socket_hook_blocks_before_os_connect_for_private_destination() -> None:
    with _temporary_egress_hook(EgressPolicy(allow_all=True)) as calls:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            with pytest.raises(ConnectionRefusedError, match="blocked network"):
                sock.connect(("10.0.0.1", 80))
    assert calls == [], "blocked destination reached the OS connect boundary"


def test_socket_hook_blocks_connect_ex_private_destination() -> None:
    with _temporary_egress_hook(EgressPolicy(allow_all=True)) as calls:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            err = sock.connect_ex(("10.0.0.1", 80))
    assert err != 0
    assert calls == [], "connect_ex bypassed the egress hook"


def test_rate_limiter_allows_up_to_limit() -> None:
    limiter = RateLimiter(limit=2, window_s=1.0)
    assert limiter.check("ip") is True
    assert limiter.check("ip") is True


def test_rate_limiter_blocks_over_limit() -> None:
    limiter = RateLimiter(limit=1, window_s=1.0)
    assert limiter.check("ip") is True
    assert limiter.check("ip") is False
    assert limiter.is_allowed("other") is True


def test_rate_limiter_resets_after_window() -> None:
    limiter = RateLimiter(limit=1, window_s=0.1)
    assert limiter.check("ip") is True
    assert limiter.check("ip") is False
    time.sleep(0.15)
    assert limiter.check("ip") is True


def test_abuse_report_store_append_and_list() -> None:
    store = AbuseReportStore()
    report = AbuseReport("203.0.113.7", "node-1", "desc", "ev", 123.0)
    store.append(report)
    assert store.count() == 1
    assert store.list_recent() == [report]


def test_abuse_report_store_bounded() -> None:
    store = AbuseReportStore(maxlen=2)
    store.append(AbuseReport("203.0.113.1", "node-1", "one", "ev1", 1.0))
    store.append(AbuseReport("203.0.113.2", "node-2", "two", "ev2", 2.0))
    store.append(AbuseReport("203.0.113.3", "node-3", "three", "ev3", 3.0))
    recent = store.list_recent()
    assert store.count() == 2
    assert [r.node_id for r in recent] == ["node-2", "node-3"]
