"""Drive a real, multi-node pipeline with the same code that the sim harness proves.

:mod:`meshpool.inference.pipeline` and :mod:`meshpool.inference.spd` are written
against a small stage interface — ``forward`` / ``truncate`` / ``drop`` — and know
nothing about where a stage physically lives. :class:`RemoteShard` implements that
interface over :class:`meshpool.common.transport.AgentTransport`, so KV caching,
chunked prefill and Speculative Pipeline Decoding run **unchanged** against machines
scattered across the internet.

That is deliberate: the acceptance gates
(``sim/scenarios/verify_spd_equivalence.py``) exercise exactly the algorithms used in
production, and only the link underneath them is swapped. A bug fixed in the sim is
fixed on the mesh, and a property proven in the sim is a property of the mesh.

Folding rollback into the next traversal
----------------------------------------
Speculation needs every stage's KV cache rolled back to the accepted length after each
round. Doing that as its own request would cost a full extra ``(N-1) * RTT`` per round
— which on a wide-area mesh is precisely the cost SPD exists to avoid, and would cancel
most of its benefit. So :meth:`RemoteShard.truncate` only *records* the rollback, and
the pending length rides along on the next ``shard/forward`` request, where the node
applies it before computing. **Rollback therefore costs zero extra round trips.**
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import numpy as np

from . import model as M
from .pipeline import LinkProfile, Pipeline


class RemoteShard:
    """One pipeline stage that lives on another machine.

    Mirrors :class:`meshpool.inference.pipeline.CachedShard`'s interface exactly, so
    :class:`~meshpool.inference.pipeline.Pipeline` cannot tell the difference.
    """

    def __init__(self, transport: Any, node_id: str, model_id: str,
                 start: int, end: int, is_first: bool, is_last: bool,
                 timeout: float = 60.0) -> None:
        self.transport = transport
        self.node_id = node_id
        self.model_id = model_id
        self.start, self.end = start, end
        self.is_first, self.is_last = is_first, is_last
        self.timeout = timeout
        self._truncate_to: dict[str, int] = {}
        self._dropped: set[str] = set()

    # -- cache control: recorded now, transmitted with the next forward ------ #
    def truncate(self, session: str, length: int) -> None:
        self._truncate_to[session] = length

    def drop(self, session: str) -> None:
        self._truncate_to.pop(session, None)
        self._dropped.add(session)

    def cache_bytes(self) -> int:  # remote; not locally observable
        return 0

    async def flush(self) -> None:
        """Release remote cache memory for sessions dropped since the last flush.

        Called once when a generation finishes, so it never sits on a token's
        critical path.
        """
        pending, self._dropped = self._dropped, set()
        for session in pending:
            try:
                await self.transport.call(
                    self.node_id, "shard/drop",
                    {"model_id": self.model_id, "session": session}, timeout=10.0)
            except Exception:  # noqa: BLE001
                # A node that vanished has already forgotten its cache; a node that
                # is merely slow will expire the session itself. Never fail a
                # completed generation over cleanup.
                pass

    async def forward(self, data, is_tokens: bool, pos_offset: int,
                      session: Optional[str]) -> np.ndarray:
        payload: dict[str, Any] = {
            "model_id": self.model_id,
            "data": data if is_tokens else np.asarray(data, np.float32).tolist(),
            "is_tokens": is_tokens,
        }
        if session is not None:
            payload["session"] = session
            payload["pos_offset"] = int(pos_offset)
            payload["use_cache"] = True
            trunc = self._truncate_to.pop(session, None)
            if trunc is not None:
                payload["truncate_to"] = int(trunc)
        body = await self.transport.call(self.node_id, "shard/forward", payload,
                                         timeout=self.timeout)
        return np.asarray(body["data"], dtype=np.float32)


class RemotePipeline(Pipeline):
    """A :class:`Pipeline` whose stages are remote nodes.

    The synthetic :class:`LinkProfile` is disabled (``rtt_ms=0``): the delay here is
    the real network, and adding a modelled delay on top would double-count it. The
    stats it records — traversals, wire bytes, positions — remain exactly comparable
    to the sim, which is what makes measured and modelled runs directly relatable.
    """

    def __init__(self, cfg: M.ModelConfig, stages: list[RemoteShard]) -> None:
        super().__init__(cfg, stages, LinkProfile(rtt_ms=0.0, bandwidth_mbps=0.0))

    async def flush(self) -> None:
        await asyncio.gather(*(s.flush() for s in self.stages), return_exceptions=True)


def build_remote_stages(transport: Any, model_id: str, partition,
                        timeout: float = 60.0) -> list[RemoteShard]:
    """Build ordered remote stages from a deployment's partition."""
    assigns = sorted(partition, key=lambda a: a.start_layer)
    if not assigns:
        raise ValueError("deployment has no partition")
    return [
        RemoteShard(transport, a.node_id, model_id, a.start_layer, a.end_layer,
                    is_first=(i == 0), is_last=(i == len(assigns) - 1),
                    timeout=timeout)
        for i, a in enumerate(assigns)
    ]
