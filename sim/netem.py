"""Network emulator for MeshPool simulation (Phase 0.4).

``NetEm`` takes a :class:`~sim.topology.Topology` and, given a pair of node IDs and
a message size, computes and *applies* the correct emulated latency, packet loss, and
bandwidth constraint — entirely in Python, no kernel tc/netem required.

The key abstraction is ``await netem.transit(src, dst, nbytes)``:
  - sleeps for ``rtt_ms/2 + nbytes/bw_kbps + jitter_sample``   (one-way transit time)
  - raises :class:`PacketLost` with probability ``loss_prob``

``NetEmTransport`` wraps any :class:`~meshpool.common.transport.AgentTransport` and
injects transit delays around every call, so existing engines exercise WAN conditions
without any change to ``meshpool/`` code.

NAT semantics (documented real-world numbers):
  - symmetric ↔ symmetric: 0% hole-punch success (RFC 4787 §14.5, industry consensus).
  - cgnat ↔ anything-not-public: unreachable (CGNAT breaks all inbound connections).
  - Failures fall back to a relay path through the first ``public`` relay node,
    which costs an extra hop of latency.
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import time
from typing import Any, Optional

from .topology import NatClass, NodeSpec, Topology

# ---------------------------------------------------------------------------
# NAT traversal matrix
# ---------------------------------------------------------------------------

# Can two nodes punch through NAT to each other?  Modelled after RFC 4787 and
# common implementation reality.  Row = src nat_class, Col = dst nat_class.
# True = direct connection possible (with hole-punch); False = relay required.
_CAN_DIRECT: dict[NatClass, dict[NatClass, bool]] = {
    "public":           {"public": True, "full_cone": True, "restricted": True,
                         "port_restricted": True, "symmetric": True, "cgnat": True},
    "full_cone":        {"public": True, "full_cone": True, "restricted": True,
                         "port_restricted": True, "symmetric": True, "cgnat": False},
    "restricted":       {"public": True, "full_cone": True, "restricted": True,
                         "port_restricted": True, "symmetric": False, "cgnat": False},
    "port_restricted":  {"public": True, "full_cone": True, "restricted": True,
                         "port_restricted": True, "symmetric": False, "cgnat": False},
    "symmetric":        {"public": True, "full_cone": True, "restricted": False,
                         "port_restricted": False, "symmetric": False, "cgnat": False},
    "cgnat":            {"public": True, "full_cone": False, "restricted": False,
                         "port_restricted": False, "symmetric": False, "cgnat": False},
}


class PacketLost(Exception):
    """Raised by :meth:`NetEm.transit` when the emulated packet is dropped."""

    def __init__(self, src: str, dst: str, prob: float) -> None:
        super().__init__(f"packet lost on {src}→{dst} (p={prob:.4f})")
        self.src = src
        self.dst = dst
        self.prob = prob


class NetEm:
    """Network emulator backed by a :class:`~sim.topology.Topology`.

    Args:
        topology:    The world topology.
        time_scale:  Compress time by this factor.  ``time_scale=0.1`` means the
                     emulator runs 10× faster while preserving *relative* timings —
                     crucial for fast unit tests.
        seed:        Seeds every random draw this emulator makes, so a whole run
                     replays identically. Pass ``seed=None`` to draw from OS entropy
                     instead, which is the only way to get an unreproducible run.
    """

    def __init__(
        self,
        topology: Topology,
        time_scale: float = 1.0,
        seed: int | None = 0,
    ) -> None:
        self.topology = topology
        self.time_scale = time_scale
        self._seed_offset = seed
        self._streams: dict[tuple[str, str], random.Random] = {}

    def _stream(self, a: str, b: str) -> random.Random:
        """The long-lived RNG stream for one directed link.

        One persistent stream per link, advanced by each draw, rather than a fresh
        ``Random`` per packet. That gives both properties the emulator needs at once:
        successive packets get independent noise, *and* the entire run is a pure
        function of ``seed``.

        The previous implementation seeded a per-call ``Random`` with
        ``topology.rng(a, b).random() + time.monotonic_ns() + seed``. Mixing the
        clock in made ``seed`` decorative -- two emulators constructed with the same
        seed produced different delays, so no emulated measurement could ever be
        replayed, and the docstring's promise that "tests can control it via the
        seed" was false. This is the instrument the calibrated performance model is
        fitted on, so non-reproducibility here silently undermines every projection
        derived from it.

        The stream is keyed on a *string*, never ``hash()``: string hashing is
        randomised per process (PYTHONHASHSEED), which would make results reproducible
        within one process but not across runs -- the subtler half of the same bug.
        """
        key = (a, b)
        st = self._streams.get(key)
        if st is None:
            st = (random.Random() if self._seed_offset is None
                  else random.Random(f"{a}|{b}|{self._seed_offset}"))
            self._streams[key] = st
        return st

    # ------------------------------------------------------------------
    # NAT / path
    # ------------------------------------------------------------------

    def can_direct(self, a: str, b: str) -> bool:
        """Return True if a direct connection is possible between nodes *a* and *b*."""
        na = self.topology.node(a)
        nb = self.topology.node(b)
        return (
            _CAN_DIRECT.get(na.nat_class, {}).get(nb.nat_class, False)
            and _CAN_DIRECT.get(nb.nat_class, {}).get(na.nat_class, False)
        )

    def path(self, a: str, b: str) -> list[str]:
        """Return the hop list for a message from *a* to *b*.

        If direct is possible: ``[a, b]``.
        Otherwise: ``[a, relay, b]`` where *relay* is the first ``public`` node
        in the topology that is neither *a* nor *b*.
        """
        if self.can_direct(a, b):
            return [a, b]
        relays = [r for r in self.topology.relay_nodes() if r not in (a, b)]
        if not relays:
            raise RuntimeError(
                f"no public relay available for {a}→{b} "
                f"(both symmetric or cgnat and no relay node defined)"
            )
        return [a, relays[0], b]

    # ------------------------------------------------------------------
    # Core transit
    # ------------------------------------------------------------------

    def _jitter_sample(self, a: str, b: str) -> float:
        """Draw one jitter sample (ms) for the (a,b) link."""
        profile = self.topology.link(a, b)
        j = profile.jitter
        rng = self._stream(a, b)
        base = rng.gauss(0.0, j.sigma_ms)
        if rng.random() < j.heavy_tail_prob:
            base *= j.heavy_tail_factor
        return abs(base)  # one-way jitter is always non-negative

    def one_way_delay_s(self, a: str, b: str, nbytes: int) -> float:
        """Compute the one-way transit time in *wall-clock* seconds (unscaled)."""
        profile = self.topology.link(a, b)
        # RTT/2 for one direction
        propagation_s = (profile.rtt_ms / 2.0) / 1000.0
        # Serialisation delay: nbytes at upload bandwidth (sender's upload)
        bw_bps = profile.bw_up_kbps * 1000.0 / 8.0   # bytes per second
        serialise_s = nbytes / bw_bps if bw_bps > 0 else 0.0
        jitter_s = self._jitter_sample(a, b) / 1000.0
        return propagation_s + serialise_s + jitter_s

    async def transit(self, src: str, dst: str, nbytes: int = 0) -> None:
        """Simulate one-way packet transit from *src* to *dst*.

        Sleeps for the correct emulated duration (affected by ``time_scale``), then
        raises :class:`PacketLost` if the packet is dropped.

        Raises:
            PacketLost: with probability ``link(src, dst).loss_prob``.
        """
        delay_s = self.one_way_delay_s(src, dst, nbytes)
        await asyncio.sleep(delay_s * self.time_scale)

        profile = self.topology.link(src, dst)
        if profile.loss_prob > 0.0:
            # Same seeded stream as jitter: packet loss has to replay identically too,
            # or a run that dropped a packet cannot be reproduced to debug it.
            if self._stream(src, dst).random() < profile.loss_prob:
                raise PacketLost(src, dst, profile.loss_prob)

    async def transit_path(self, path: list[str], nbytes: int = 0) -> None:
        """Transit along a multi-hop path.  Each hop is emulated independently."""
        for i in range(len(path) - 1):
            await self.transit(path[i], path[i + 1], nbytes)

    # ------------------------------------------------------------------
    # Measurement helper (for calibration test)
    # ------------------------------------------------------------------

    async def measure_rtt(self, a: str, b: str, nbytes: int = 0) -> float:
        """Measure the round-trip time (A→B→A) in *real* wall-clock seconds."""
        t0 = time.perf_counter()
        await self.transit(a, b, nbytes)
        await self.transit(b, a, nbytes)
        return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Transport wrapper
# ---------------------------------------------------------------------------

def _estimate_payload_bytes(payload: Optional[dict]) -> int:
    """Estimate the serialised JSON size of *payload* in bytes."""
    if payload is None:
        return 0
    try:
        return len(json.dumps(payload).encode())
    except Exception:  # noqa: BLE001
        return 256  # conservative fallback


class NetEmTransport:
    """Wraps an :class:`~meshpool.common.transport.AgentTransport` and injects
    latency, loss, and bandwidth constraints around every call.

    The source node is the **caller** (typically the control plane node id, or a
    synthetic ``"caller"`` sentinel) and the destination is the **target agent id**.
    The transport uses :meth:`~NetEm.path` to resolve NAT and relay hops.

    Args:
        inner:       The real transport to delegate to after the delay.
        netem:       The network emulator instance.
        src_node_id: Node ID of the caller (must be in the topology).
    """

    def __init__(
        self,
        inner: Any,          # AgentTransport
        netem: NetEm,
        src_node_id: str,
    ) -> None:
        self._inner = inner
        self._netem = netem
        self._src = src_node_id

    async def _pre_call(self, dst: str, nbytes: int) -> None:
        """Apply outbound latency before the real call."""
        hops = self._netem.path(self._src, dst)
        await self._netem.transit_path(hops, nbytes)

    async def _post_call(self, dst: str, nbytes: int) -> None:
        """Apply return-path latency after the real call."""
        hops = self._netem.path(dst, self._src)
        await self._netem.transit_path(hops, nbytes)

    async def call(
        self,
        node_id: str,
        path: str,
        payload: Optional[dict] = None,
        *,
        method: str = "POST",
        timeout: float = 30.0,
    ) -> dict:
        nbytes = _estimate_payload_bytes(payload)
        await self._pre_call(node_id, nbytes)
        result = await self._inner.call(node_id, path, payload, method=method, timeout=timeout)
        response_bytes = _estimate_payload_bytes(result) if isinstance(result, dict) else 256
        await self._post_call(node_id, response_bytes)
        return result

    async def proxy(
        self,
        node_id: str,
        method: str,
        url: str,
        *,
        params: Any = None,
        content: bytes = b"",
        headers: Optional[dict] = None,
        timeout: float = 15.0,
    ) -> Any:
        await self._pre_call(node_id, len(content))
        result = await self._inner.proxy(
            node_id, method, url, params=params, content=content,
            headers=headers, timeout=timeout,
        )
        await self._post_call(node_id, len(getattr(result, "content", b"")))
        return result

    async def proxy_open(
        self,
        node_id: str,
        method: str,
        url: str,
        *,
        params: Any = None,
        content: bytes = b"",
        headers: Optional[dict] = None,
        timeout: float = 15.0,
    ) -> Any:
        await self._pre_call(node_id, len(content))
        result = await self._inner.proxy_open(
            node_id, method, url, params=params, content=content,
            headers=headers, timeout=timeout,
        )
        # No post-call delay for streaming (we would need to wrap the iterator)
        return result

    async def close(self) -> None:
        await self._inner.close()
