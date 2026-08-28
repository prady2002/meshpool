"""Automatic node classification -- plan §3.6.

Not every machine can do every job, and pretending otherwise is how meshes become
unreliable. A pipeline is a chain, so its availability is the *product* of its
stages': ten nodes at 0.95 give 0.60. Handing an inference stage to a laptop that
unplugs at 6pm is therefore not a small mistake, it is a multiplicative one.

Four classes, taken from measured signals only:

    Anchor        publicly addressable, proven uptime, mains powered
                  -> control-plane replicas, relays, metadata, stateful services
    Relay         publicly addressable, decent uptime
                  -> DERP-style relay for CGNAT peers, weight-swarm seeding
    Constellation residential, real memory, in a latency cluster
                  -> inference stages, hosting replicas, warm residency
    Drifter       unreachable, intermittent, or battery powered
                  -> pull-only batch, cold erasure shards, draft-model duty

Users never choose a class. It is derived, continuously, from what we have
actually observed, which is what makes it trustworthy enough to schedule on.

Two design rules earn their own tests, because both are easy to get wrong and
neither fails visibly:

**Absence of evidence is not evidence of reliability.** Every ``TrustScore`` EMA
defaults to ``1.0``, so a node that has never been observed looks flawless. Read
naively, a stranger's first heartbeat would be promoted to Anchor and handed
metadata. Promotion therefore requires ``MIN_SAMPLES_FOR_PROMOTION`` observations,
and ``confidence`` reports how much evidence is actually behind the verdict.

**Reachability is not reliability.** Anchor and Relay both exist to be *dialled by
peers*. Uptime says nothing about whether a peer can open a connection, so a
perfect record behind CGNAT still cannot fill either role. This is a hard gate,
never a score, because a scored version lets a long-lived CGNAT node outrank a
reachable one and fail only when someone tries to reach it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from ..common.models import Node, TrustScore


class NodeClass(str, Enum):
    ANCHOR = "anchor"
    RELAY = "relay"
    CONSTELLATION = "constellation"
    DRIFTER = "drifter"


#: Observations required before a node may be promoted to a role that other nodes
#: depend on. Below this the trust EMAs are still dominated by their optimistic
#: defaults, so they carry no information.
MIN_SAMPLES_FOR_PROMOTION = 50

ANCHOR_MIN_UPTIME = 0.95
RELAY_MIN_UPTIME = 0.80
CONSTELLATION_MIN_UPTIME = 0.70

#: Anchors hold state for other nodes, so they need enough memory to be worth the
#: role. This is deliberately modest -- the distinguishing property is stability,
#: not size.
ANCHOR_MIN_RAM_MB = 8_000

#: ``network_class`` values from which a peer can open an inbound connection.
#: "unknown" is absent on purpose: it means the probe never ran, and guessing
#: "reachable" produces a failure that only appears when a peer tries to dial.
DIALABLE_NETWORK_CLASSES = frozenset({"public", "full-cone"})


@dataclass(frozen=True)
class Classification:
    """A verdict plus the evidence for it.

    ``reasons`` exists so the console can explain *why* a machine was or was not
    chosen, which is the difference between a scheduler users trust and one they
    suspect.
    """

    node_class: NodeClass
    reasons: tuple[str, ...]
    #: 0.0 = expected to still be here next hour, 1.0 = expected to vanish.
    #: Used to break ties and to keep the most volatile nodes out of pipelines.
    volatility: float
    #: How much observation backs this verdict, in [0, 1]. A low value means the
    #: class is a floor, not a judgement.
    confidence: float

    def suitable_for_pipeline_stage(self) -> bool:
        """Whether this node should hold a model layer range.

        Drifters are excluded not because they are slow but because a pipeline
        multiplies its stages' availabilities, so one volatile member degrades
        every request rather than its own share of them.
        """
        return self.node_class in (NodeClass.ANCHOR, NodeClass.CONSTELLATION,
                                   NodeClass.RELAY)


def _volatility(node: "Node", trust: Optional["TrustScore"], samples: int) -> float:
    """Probability-ish estimate that this node stops serving soon.

    Deliberately built from independent hazards rather than a single blended
    score: a machine can be perfectly stable *and* on battery, and the battery is
    what matters at 6pm.
    """
    score = 0.0
    cap = node.capability

    uptime = trust.uptime_ratio_ema if trust is not None else 1.0
    if samples >= MIN_SAMPLES_FOR_PROMOTION:
        score += (1.0 - max(0.0, min(1.0, uptime))) * 0.6
    else:
        # No evidence: assume the population average rather than the optimistic
        # default, so an unmeasured node is never ranked above a proven one.
        score += 0.25

    battery = getattr(cap, "battery_powered", None)
    if battery is True:
        score += 0.30
    elif battery is None:
        # Unprobed. Laptops are the population most likely to be unprobed *and*
        # most likely to vanish, so this cannot collapse into "desktop".
        score += 0.10

    if getattr(cap, "thermal_state", "unknown") == "throttling":
        score += 0.15
    elif getattr(cap, "thermal_state", "unknown") == "warm":
        score += 0.05

    if getattr(cap, "network_class", "unknown") not in DIALABLE_NETWORK_CLASSES:
        score += 0.05

    if trust is not None and samples >= MIN_SAMPLES_FOR_PROMOTION:
        score += (1.0 - max(0.0, min(1.0, trust.latency_consistency_ema))) * 0.10

    return round(max(0.0, min(1.0, score)), 4)


def classify(node: "Node", trust: Optional["TrustScore"] = None) -> Classification:
    """Derive a node's class from measured signals. Pure; safe to call per request."""
    cap = node.capability
    reasons: list[str] = []

    samples = int(getattr(trust, "samples", 0) or 0)
    uptime = float(getattr(trust, "uptime_ratio_ema", 1.0) or 0.0)
    network_class = getattr(cap, "network_class", "unknown") or "unknown"
    battery = getattr(cap, "battery_powered", None)
    ram_mb = int(getattr(cap, "ram_mb", 0) or 0)
    vram_mb = int(getattr(cap, "vram_mb", 0) or 0)

    dialable = network_class in DIALABLE_NETWORK_CLASSES
    proven = samples >= MIN_SAMPLES_FOR_PROMOTION
    confidence = round(min(1.0, samples / float(MIN_SAMPLES_FOR_PROMOTION)), 4)
    volatility = _volatility(node, trust, samples)

    if dialable:
        reasons.append(f"reachable: network_class={network_class}")
    else:
        reasons.append(
            f"not dialable by peers: network_class={network_class}"
            + (" (never probed)" if network_class == "unknown" else "")
        )

    if proven:
        reasons.append(f"uptime {uptime:.2f} over {samples} samples")
    else:
        reasons.append(
            f"insufficient evidence: {samples} samples < {MIN_SAMPLES_FOR_PROMOTION} "
            f"required before promotion (trust EMAs still at their optimistic default)"
        )

    if battery is True:
        reasons.append("battery powered: can leave when unplugged")
    elif battery is None:
        reasons.append("battery state never probed: treated as possibly portable")

    thermal = getattr(cap, "thermal_state", "unknown")
    if thermal in ("warm", "throttling"):
        reasons.append(f"thermal state {thermal}")

    if dialable and proven and uptime >= ANCHOR_MIN_UPTIME \
            and battery is False and ram_mb >= ANCHOR_MIN_RAM_MB:
        reasons.append("anchor: reachable, proven stable, mains powered")
        return Classification(NodeClass.ANCHOR, tuple(reasons), volatility, confidence)

    if dialable and proven and uptime >= RELAY_MIN_UPTIME:
        reasons.append("relay: reachable with adequate uptime")
        return Classification(NodeClass.RELAY, tuple(reasons), volatility, confidence)

    if battery is True:
        reasons.append("drifter: portable power makes departure likely, not merely possible")
        return Classification(NodeClass.DRIFTER, tuple(reasons), volatility, confidence)

    if uptime >= CONSTELLATION_MIN_UPTIME and max(ram_mb, vram_mb) > 0:
        reasons.append("constellation: enough memory and stability to hold a stage")
        return Classification(NodeClass.CONSTELLATION, tuple(reasons), volatility,
                              confidence)

    reasons.append("drifter: too intermittent for work others depend on")
    return Classification(NodeClass.DRIFTER, tuple(reasons), volatility, confidence)
