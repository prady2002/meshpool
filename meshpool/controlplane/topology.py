"""Failure-domain inference for correlated-failure-aware placement.

Correlated failure is the classic way distributed systems die: if all 9 EC shards
live on machines in one university lab, one power cut destroys the file and the
"3-failure tolerance" Reed-Solomon promised was fiction.  This module makes that
visible and acts on it.

Domain hierarchy (coarsest -> narrowest):
  owner  :  same self-declared operator -- fails together by human, not hardware
  asn    :  same ISP / provider (optional offline lookup, degrades gracefully)
  subnet :  /24 IPv4 -- almost certainly same LAN
  machine:  ip:port -- same physical endpoint

Placement spreads shards across the coarsest available domain level and degrades
gracefully when the mesh is too small for perfect diversity.

Honest degradation is the crux: when perfect diversity is impossible (2-node lab)
the system places what it can and reports the *actual* effective durability -- not
the theoretical RS number.  Showing "11 nines" when all shards share one building
is the specific lie this module exists to prevent.

None of this requires a network call.  ASN lookup is optional (requires ipwhois);
all other signals come from the registration request.
"""
from __future__ import annotations

import re
import socket
import struct
from collections import Counter
from typing import Optional


# ---------------------------------------------------------------------------
# Address parsing
# ---------------------------------------------------------------------------

def _parse_addr(url: str) -> tuple[str, int]:
    """Extract (ip, port) from an agent URL like http://1.2.3.4:9001."""
    m = re.match(r"https?://\[?([^\]/:]+)\]?:(\d+)", url)
    if m:
        return m.group(1), int(m.group(2))
    m = re.match(r"https?://([^/:]+)(?::(\d+))?", url)
    if m:
        return m.group(1), int(m.group(2) or "80")
    return "unknown", 0


def _is_private(ip: str) -> bool:
    """Return True for RFC 1918 / loopback / link-local addresses (no public ASN)."""
    try:
        n = struct.unpack(">I", socket.inet_aton(ip))[0]
        for lo, hi in [
            (0x0A000000, 0x0AFFFFFF),  # 10.0.0.0/8
            (0xAC100000, 0xAC1FFFFF),  # 172.16.0.0/12
            (0xC0A80000, 0xC0A8FFFF),  # 192.168.0.0/16
            (0x7F000000, 0x7FFFFFFF),  # 127.0.0.0/8
            (0xA9FE0000, 0xA9FEFFFF),  # 169.254.0.0/16
        ]:
            if lo <= n <= hi:
                return True
    except OSError:
        pass
    return False


def _subnet_24(ip: str) -> str:
    """Return /24 subnet notation, e.g. '1.2.3.x' for '1.2.3.4'."""
    try:
        parts = ip.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.{parts[2]}.x"
    except Exception:  # noqa: BLE001
        pass
    return ip


def _lookup_asn_offline(ip: str) -> Optional[str]:
    """Attempt an offline / lightweight ASN lookup via ipwhois (if installed).

    Private and loopback addresses have no public ASN; we skip them to avoid a
    spurious network probe.  Any exception (import error, timeout, RDAP failure)
    is silently swallowed -- domain inference must never block or crash registration.
    Returns None on any failure.
    """
    if _is_private(ip):
        return None
    try:
        from ipwhois import IPWhois  # type: ignore[import]
        result = IPWhois(ip).lookup_rdap(depth=1, retry_count=1)
        asn = result.get("asn")
        if asn and str(asn) != "NA":
            return f"AS{asn}"
    except ImportError:
        pass
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# Domain inference
# ---------------------------------------------------------------------------

def infer_failure_domain(url: str, owner: str = "default",
                         explicit: Optional[str] = None) -> str:
    """Derive a placement failure-domain key for a node.

    Resolution order (coarsest first -- broadest correlated-failure scope wins):

    1. explicit  -- simulation/operator override; used verbatim (tests inject this)
    2. owner     -- non-default self-declared operator; all their nodes fail together
    3. asn       -- same ISP/provider (best-effort offline lookup, graceful fallback)
    4. subnet    -- /24 same LAN (almost certainly correlated)
    5. machine   -- ip:port (unique per listening endpoint)

    The returned string is the grouping key used by pick_diverse_nodes.
    Nodes sharing a key are assumed to fail together.

    Never raises -- any error degrades to the narrowest level (machine), which is
    the safest assumption (each node is independent).
    """
    if explicit:
        return explicit

    try:
        ip, port = _parse_addr(url)

        # Same operator's nodes fail together by policy / shared management
        if owner and owner not in ("", "default"):
            return f"owner:{owner}"

        # Same ISP/provider -- optional, best-effort
        asn = _lookup_asn_offline(ip)
        if asn:
            return f"asn:{asn}"

        # Same /24 subnet -- almost certainly same LAN
        if ip not in ("unknown", ""):
            return f"subnet:{_subnet_24(ip)}"

        return f"machine:{ip}:{port}"
    except Exception:  # noqa: BLE001
        return f"machine:{url}"


# ---------------------------------------------------------------------------
# Diversity-aware placement
# ---------------------------------------------------------------------------

def pick_diverse_nodes(nodes: list, count: int) -> list:
    """Select up to *count* nodes, maximising failure-domain diversity.

    Guaranteed properties:
    * MONOTONIC -- adding more nodes to the pool never reduces the returned set
      size.  If len(nodes) >= count and count > 0, exactly count nodes are returned.
    * GRACEFUL DEGRADATION -- when the number of distinct domains is smaller than
      count, picks the best achievable spread rather than failing or returning fewer
      nodes than available.
    * NEVER FAILS CLOSED -- always returns as many distinct nodes as possible, up
      to count.  Adding a node can only help, never hurt.

    ``nodes`` must have a ``failure_domain`` attribute (str or None) and an ``id``
    attribute.  Nodes without a failure_domain are each treated as their own unique
    domain (best-case assumption: independent).

    Input ordering within each domain is preserved (callers should pre-sort by
    whichever criterion matters, e.g. remaining disk space).
    """
    if not nodes or count <= 0:
        return []

    # Group nodes by failure domain.
    # Nodes with no domain are each their own unique group (independent assumption).
    by_domain: dict[str, list] = {}
    for n in nodes:
        domain = getattr(n, "failure_domain", None) or n.id
        by_domain.setdefault(domain, []).append(n)

    # Deal one node per domain per round, cycling through domains in ascending
    # domain-size order so smaller domains are not starved by larger ones.
    # This is the standard card-dealing diversity algorithm.
    domain_queues: list[list] = sorted(by_domain.values(), key=len)

    selected: list = []
    used: set = set()

    while len(selected) < count:
        made_progress = False
        for queue in domain_queues:
            if len(selected) >= count:
                break
            for node in queue:
                if node.id not in used:
                    selected.append(node)
                    used.add(node.id)
                    made_progress = True
                    break
        if not made_progress:
            break  # all candidates exhausted

    return selected[:count]


# ---------------------------------------------------------------------------
# Effective durability calculation
# ---------------------------------------------------------------------------

def effective_durability(
        shard_domains: list[str],
        k: int,
        m: int,
) -> dict:
    """Compute the *honest* effective durability of an EC-coded object.

    Given the failure domains of the k+m shard holders, reports:

    - theoretical_tolerance  : m (the RS guarantee, assuming independent failures)
    - effective_tolerance    : RS budget minus worst-case correlated loss, clamped [0, m]
    - tolerable_domain_kills : how many full-domain outages can be survived
    - correlated             : True when any domain holds > 1 shard
    - warning                : plain-English explanation when durability is degraded

    When placement is perfectly diverse (one shard per domain), effective_tolerance
    equals theoretical_tolerance.  Otherwise it is strictly lower.

    A UI that shows the theoretical number when effective_tolerance < theoretical is
    lying to the user -- that is the specific failure this function exists to prevent.
    """
    total = k + m
    if not shard_domains:
        return {
            "n_shards": total, "n_domains": 0, "max_domain_size": 0,
            "tolerable_domain_kills": 0,
            "theoretical_tolerance": m, "effective_tolerance": 0,
            "correlated": False, "warning": "no shards placed",
        }

    domain_counts = Counter(shard_domains)
    n_domains = len(domain_counts)
    # Sort domain sizes descending (worst domain first)
    sizes = sorted(domain_counts.values(), reverse=True)
    max_domain_size = sizes[0]

    # tolerable_domain_kills: max D such that the D largest domains together hold
    # <= m shards (i.e. their combined failure stays within the RS budget).
    tolerable_domain_kills = 0
    cumulative = 0
    for s in sizes:
        if cumulative + s > m:
            break
        cumulative += s
        tolerable_domain_kills += 1

    # effective_tolerance: how many individual shard failures are survivable in the
    # worst-case correlated scenario.  If the biggest domain has s_max shards and they
    # all fail at once, we lose s_max shards.  We survive iff s_max <= m.
    # The residual tolerance after that worst-case hit = m - s_max + 1 (so that
    # killing s_max shards leaves exactly m - (s_max - 1) ... no, let me think again.
    #
    # RS survives losing any *m* shards.  If one correlated event kills all s_max shards
    # in the biggest domain at once, that counts as s_max simultaneous failures.
    # After that kill: m - s_max shards of "budget" remain for further independent
    # losses.  So effective_tolerance = max(0, m - s_max + 1):
    #   s_max == 1: perfect diversity, m independent kills still possible
    #   s_max == m: one domain kill uses the whole budget, 1 single-domain kill tolerable
    #   s_max > m : a single domain kill exceeds RS budget -> data lost (eff = 0)
    effective_tolerance = max(0, m - max_domain_size + 1)

    correlated = max_domain_size > 1
    warning: Optional[str] = None
    if correlated:
        worst_domain = max(domain_counts, key=domain_counts.get)
        if max_domain_size > m:
            warning = (
                f"{max_domain_size}/{total} shards share domain '{worst_domain}'; "
                f"a single correlated outage WILL lose data -- theoretical RS "
                f"tolerance {m} shards is exceeded (effective tolerance: "
                f"{effective_tolerance} independent failures). "
                f"The reported durability number is a lie unless diversity is enforced."
            )
        else:
            warning = (
                f"{max_domain_size} shards share domain '{worst_domain}'; "
                f"effective domain-kill tolerance is {tolerable_domain_kills} "
                f"(theoretical independent-failure tolerance: {m} shards)"
            )

    return {
        "n_shards": total,
        "n_domains": n_domains,
        "max_domain_size": max_domain_size,
        "tolerable_domain_kills": tolerable_domain_kills,
        "theoretical_tolerance": m,
        "effective_tolerance": effective_tolerance,
        "correlated": correlated,
        "warning": warning,
    }
