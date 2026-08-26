from __future__ import annotations

import time

import pytest

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


def test_private_ip_overridden_by_whitelist() -> None:
    EgressPolicy(whitelist=[("127.0.0.1", 8080)]).check("127.0.0.1", 8080)


def test_deny_by_default_for_public_ip() -> None:
    with pytest.raises(EgressDenied):
        EgressPolicy().check("1.2.3.4", 80)


def test_allow_all_permits_public_but_still_blocks_private() -> None:
    policy = EgressPolicy(allow_all=True)
    policy.check("1.2.3.4", 80)
    with pytest.raises(EgressDenied):
        policy.check("10.0.0.1", 80)


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
