"""Acceptance gate: NAT traversal matrix with first-class relay fallback.

Honest scope
------------
This is measured under MeshPool's deterministic NAT model, not on the public
internet. It asserts the physical constraints from the plan: symmetric↔symmetric
hole punching is a structural 0% case, CGNAT-to-residential needs relay, and relay
selection is low-RTT rather than second-class or random.

Exit 0 = PASS. Exit 1 = FAIL.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from meshpool.cloud.nat import (  # noqa: E402
    EXPECTED_DCUTR_SUCCESS,
    NAT_KEEPALIVE_S,
    PERMANENT_RELAY_DEPENDENCY_RANGE,
    RelayCandidate,
    can_hole_punch,
    plan_traversal,
)


NAT_CLASSES = [
    "full_cone",
    "address_restricted",
    "port_restricted",
    "symmetric",
    "cgnat",
]

EXPECTED_DIRECT = {
    ("full_cone", "full_cone"),
    ("full_cone", "address_restricted"),
    ("address_restricted", "full_cone"),
    ("full_cone", "port_restricted"),
    ("port_restricted", "full_cone"),
    ("full_cone", "symmetric"),
    ("symmetric", "full_cone"),
    ("address_restricted", "address_restricted"),
    ("address_restricted", "port_restricted"),
    ("port_restricted", "address_restricted"),
    ("port_restricted", "port_restricted"),
}


def main() -> int:
    failures: list[str] = []
    relays = [
        RelayCandidate("relay-far", 90.0, 90.0),
        RelayCandidate("relay-near", 15.0, 20.0),
        RelayCandidate("relay-mid", 30.0, 30.0),
    ]

    def check(cond: bool, name: str, detail: str = "") -> None:
        if cond:
            print(f"  PASS: {name}" + (f" ({detail})" if detail else ""))
        else:
            print(f"  FAIL: {name}" + (f" ({detail})" if detail else ""))
            failures.append(name)

    print("== test 1: NAT constants are honest planning numbers ==")
    check(20.0 <= NAT_KEEPALIVE_S <= 25.0, "keepalive matches 15-30s UDP NAT budget",
          f"{NAT_KEEPALIVE_S:.0f}s")
    check(EXPECTED_DCUTR_SUCCESS <= 0.73, "DCUtR success is not overstated",
          f"{EXPECTED_DCUTR_SUCCESS:.2f}")
    check(PERMANENT_RELAY_DEPENDENCY_RANGE == (0.25, 0.40),
          "relay dependency budget is 25-40%",
          f"{PERMANENT_RELAY_DEPENDENCY_RANGE[0]:.0%}-{PERMANENT_RELAY_DEPENDENCY_RANGE[1]:.0%}")

    print("\n== test 2: all residential NAT-class pairs choose direct or relay correctly ==")
    direct_count = 0
    relay_count = 0
    for src_nat in NAT_CLASSES:
        for dst_nat in NAT_CLASSES:
            expected_direct = (src_nat, dst_nat) in EXPECTED_DIRECT
            possible = can_hole_punch(src_nat, dst_nat)
            check(possible == expected_direct,
                  f"matrix {src_nat}->{dst_nat} physical possibility",
                  f"possible={possible}")
            plan = plan_traversal(f"src-{src_nat}", f"dst-{dst_nat}",
                                  src_nat, dst_nat, relays)
            if expected_direct:
                direct_count += 1
                check(plan.kind == "direct", f"{src_nat}->{dst_nat} uses direct path",
                      f"path={plan.path}")
                check(0 < plan.hole_punch_attempts <= 3,
                      f"{src_nat}->{dst_nat} bounded hole-punch attempts",
                      f"attempts={plan.hole_punch_attempts}")
            else:
                relay_count += 1
                check(plan.kind == "relay", f"{src_nat}->{dst_nat} falls back to relay",
                      f"path={plan.path}")
                check(plan.relay_id == "relay-near",
                      f"{src_nat}->{dst_nat} chooses lowest-RTT relay",
                      f"relay={plan.relay_id}")
                check(plan.hole_punch_attempts == 0,
                      f"{src_nat}->{dst_nat} does not stall on structural failure",
                      f"attempts={plan.hole_punch_attempts}")

    print("\n== test 3: non-vacuity guard for relay fallback ==")
    try:
        plan_traversal("src-symmetric", "dst-symmetric", "symmetric", "symmetric", [])
    except RuntimeError as exc:
        check("no public relay" in str(exc), "symmetric-to-symmetric fails loudly if relay is broken")
    else:
        check(False, "symmetric-to-symmetric should fail with relay disabled")

    check(direct_count > 0 and relay_count > 0, "matrix exercised both direct and relay cases",
          f"direct={direct_count} relay={relay_count}")

    if failures:
        print("\nFAILED NAT TRAVERSAL CHECKS:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nALL NAT TRAVERSAL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
