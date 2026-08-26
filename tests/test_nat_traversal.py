from __future__ import annotations

import pytest

from meshpool.cloud.nat import (
    NAT_KEEPALIVE_S,
    NatProbeObservation,
    RelayCandidate,
    can_hole_punch,
    detect_nat_class,
    plan_traversal,
    select_lowest_rtt_relay,
)


def test_detect_nat_class_from_measured_probe_results():
    assert detect_nat_class(NatProbeObservation(
        local_endpoint=("203.0.113.10", 4000),
        mapped_endpoints=(("203.0.113.10", 4000),),
    )) == "public"
    assert detect_nat_class(NatProbeObservation(
        local_endpoint=("10.0.0.2", 4000),
        mapped_endpoints=(("198.51.100.9", 62000),),
        received_from_unknown_addr=True,
    )) == "full_cone"
    assert detect_nat_class(NatProbeObservation(
        local_endpoint=("10.0.0.2", 4000),
        mapped_endpoints=(("198.51.100.9", 62000),),
        received_from_known_addr_other_port=True,
    )) == "address_restricted"
    assert detect_nat_class(NatProbeObservation(
        local_endpoint=("10.0.0.2", 4000),
        mapped_endpoints=(("198.51.100.9", 62000),),
    )) == "port_restricted"
    assert detect_nat_class(NatProbeObservation(
        local_endpoint=("10.0.0.2", 4000),
        mapped_endpoints=(("198.51.100.9", 62000), ("198.51.100.9", 62001)),
    )) == "symmetric"
    assert detect_nat_class(NatProbeObservation(
        local_endpoint=("10.0.0.2", 4000),
        mapped_endpoints=(),
        udp_reachable=False,
    )) == "cgnat"


def test_symmetric_to_symmetric_is_never_direct():
    assert not can_hole_punch("symmetric", "symmetric")
    plan = plan_traversal("a", "b", "symmetric", "symmetric", [
        RelayCandidate("relay-slow", 50, 50),
        RelayCandidate("relay-fast", 10, 15),
    ])
    assert plan.kind == "relay"
    assert plan.relay_id == "relay-fast"
    assert plan.hole_punch_attempts == 0


def test_plausible_pairs_get_bounded_hole_punch_attempts():
    plan = plan_traversal("a", "b", "full_cone", "port_restricted", [
        RelayCandidate("relay", 1, 1),
    ])
    assert plan.kind == "direct"
    assert 0 < plan.hole_punch_attempts <= 3
    assert 20.0 <= NAT_KEEPALIVE_S <= 25.0


def test_relay_selection_ignores_non_public_and_prefers_lowest_rtt():
    relay = select_lowest_rtt_relay([
        RelayCandidate("not-relay", 1, 1, nat_class="symmetric"),
        RelayCandidate("slow", 20, 30),
        RelayCandidate("fast", 15, 10),
    ])
    assert relay.node_id == "fast"


def test_no_relay_is_loud_not_silent():
    with pytest.raises(RuntimeError):
        plan_traversal("a", "b", "cgnat", "symmetric", [])
