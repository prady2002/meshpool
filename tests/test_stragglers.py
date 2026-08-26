"""Straggler tolerance: rotate away from a node that got slow, not just one that died.

Plan section 3.4 says one mechanism should cover both churn and stragglers, and that
is a stronger claim than it looks. Death is easy to detect -- the call raises. Slowness
is not: the node answers, correctly, on time by every functional measure, just late.
On consumer hardware slowness is the *normal* degradation (thermal throttling, a game
launching, Wi-Fi renegotiating to a slower rate) and in a pipeline it is the expensive
one, because ``t_token`` is bounded by the slowest stage no matter how fast the rest
are.

These tests pin the decision rules rather than the outcome, because the outcome
("throughput recovered") can be produced by luck on a fast machine, while the rules are
what will or will not hold on a mesh.
"""
from __future__ import annotations

import time

import pytest

from meshpool.inference import remote as R
from meshpool.inference.remote import RemotePipeline, RemoteShard
from meshpool.inference import model as M


class _Transport:
    async def call(self, *a, **kw):  # pragma: no cover - never reached
        raise AssertionError("these tests drive the decision logic, not the wire")


def _stage(name: str, spares=(), start=0, end=1) -> RemoteShard:
    return RemoteShard(_Transport(), name, "mdl", start, end,
                       is_first=(start == 0), is_last=False, spares=list(spares))


def _pipe(*stages: RemoteShard) -> RemotePipeline:
    cfg = M.ModelConfig(n_layer=len(stages), d_model=8, n_head=2,
                        vocab_size=16, block_size=8)
    return RemotePipeline(cfg, list(stages))


def _feed(stage: RemoteShard, ms: float, n: int) -> None:
    for _ in range(n):
        stage.observe_cost(ms)


# --------------------------------------------------------------------------- #
class TestTheDecisionIsPeerRelative:
    def test_a_uniformly_slow_pipeline_rotates_nothing(self):
        """Slow is not the same as straggling.

        Every stage taking 500 ms is a slow *deployment* -- a big model, a weak mesh,
        a long link. Rotating any stage away costs a KV rebuild and buys nothing,
        because the replacement is drawn from the same population. An absolute
        millisecond threshold gets this exactly wrong, which is why there isn't one.
        """
        a, b, c = _stage("a", ["a2"]), _stage("b", ["b2"]), _stage("c", ["c2"])
        p = _pipe(a, b, c)
        for _ in range(20):
            _feed(a, 500, 1); _feed(b, 520, 1); _feed(c, 490, 1)
            assert p.check_stragglers() is None
        assert p.straggler_rotations == 0

    def test_one_stage_far_slower_than_its_peers_is_retired(self):
        a, b, c = _stage("a", ["a2"]), _stage("b", ["b2"]), _stage("c", ["c2"])
        p = _pipe(a, b, c)
        rotated = None
        for _ in range(R.STRAGGLER_STREAK + R.STRAGGLER_MIN_SAMPLES + 2):
            _feed(a, 10, 1); _feed(b, 400, 1); _feed(c, 12, 1)
            rotated = p.check_stragglers() or rotated
        assert rotated is b, "the slow stage was not the one retired"
        assert b.node_id == "b2"
        assert "b" in b.dead
        assert p.straggler_rotations == 1

    def test_only_the_slowest_stage_is_considered(self):
        """Rotating the runner-up buys nothing and costs a rebuild.

        Throughput is bounded by the maximum hop, so with two slow stages the second
        one is invisible until the first is dealt with. Acting on both at once would
        double the recovery cost for no gain.
        """
        a, b, c = _stage("a", ["a2"]), _stage("b", ["b2"]), _stage("c", ["c2"])
        p = _pipe(a, b, c)
        for _ in range(R.STRAGGLER_STREAK + R.STRAGGLER_MIN_SAMPLES + 2):
            _feed(a, 10, 1); _feed(b, 900, 1); _feed(c, 300, 1)
            p.check_stragglers()
        assert b.straggler_rotations == 1
        assert c.straggler_rotations == 0, "the runner-up was rotated too"


class TestTheDecisionIsSustained:
    def test_a_single_spike_does_not_cost_a_rotation(self):
        """One slow traversal is a hiccup, not a straggler."""
        a, b, c = _stage("a", ["a2"]), _stage("b", ["b2"]), _stage("c", ["c2"])
        p = _pipe(a, b, c)
        for _ in range(6):
            _feed(a, 10, 1); _feed(b, 11, 1); _feed(c, 10, 1)
            p.check_stragglers()
        _feed(b, 5000, 1)
        assert p.check_stragglers() is None
        assert p.straggler_rotations == 0

    def test_the_streak_resets_when_the_node_recovers(self):
        """A node that comes back must not carry credit toward a later eviction."""
        a, b, c = _stage("a", ["a2"]), _stage("b", ["b2"]), _stage("c", ["c2"])
        p = _pipe(a, b, c)
        for _ in range(4):
            _feed(a, 10, 1); _feed(b, 10, 1); _feed(c, 10, 1)
            p.check_stragglers()
        # Slow, but stop one short of the streak.
        for _ in range(R.STRAGGLER_STREAK - 1):
            _feed(a, 10, 1); _feed(b, 900, 1); _feed(c, 10, 1)
            p.check_stragglers()
        assert b.slow_streak == R.STRAGGLER_STREAK - 1
        # Recovers. The EMA is still high and falls slowly, so if the streak were
        # driven by the EMA alone this node would be evicted several hops *after* it
        # got healthy again -- punished for history it no longer has.
        for _ in range(8):
            _feed(a, 10, 1); _feed(b, 10, 1); _feed(c, 10, 1)
            p.check_stragglers()
        assert b.slow_streak == 0
        assert p.straggler_rotations == 0, (
            "a node that had already recovered was rotated away for its history")

    def test_nothing_is_decided_before_the_ema_has_samples(self):
        a, b = _stage("a", ["a2"]), _stage("b", ["b2"])
        p = _pipe(a, b)
        _feed(a, 10, 1)
        _feed(b, 9999, 1)
        assert p.check_stragglers() is None, "acted on a single sample"

    def test_a_lone_measured_stage_has_no_peer_and_is_never_blamed(self):
        """With nothing to compare against, any threshold is an absolute one."""
        a, b = _stage("a", ["a2"]), _stage("b", ["b2"])
        p = _pipe(a, b)
        _feed(a, 100000, 10)   # only stage a has enough samples
        for _ in range(20):
            assert p.check_stragglers() is None
        assert a.straggler_rotations == 0


class TestRotationIsBounded:
    def test_a_stage_with_no_spare_keeps_serving(self):
        """Slow is strictly better than absent.

        A mesh with nowhere better to go must keep answering from the slow node
        rather than failing the request, so a missing spare is a no-op, not an error.
        """
        a, b, c = _stage("a"), _stage("b"), _stage("c")   # no spares anywhere
        p = _pipe(a, b, c)
        for _ in range(30):
            _feed(a, 10, 1); _feed(b, 900, 1); _feed(c, 10, 1)
            assert p.check_stragglers() is None
        assert b.node_id == "b", "rotated to a candidate that does not exist"

    def test_a_stage_cannot_walk_its_whole_spare_list_in_one_bad_minute(self):
        a = _stage("a", ["a2"])
        b = _stage("b", ["b2", "b3", "b4"])
        c = _stage("c", ["c2"])
        p = _pipe(a, b, c)
        for _ in range(60):
            _feed(a, 10, 1); _feed(b, 900, 1); _feed(c, 10, 1)
            p.check_stragglers()
        assert p.straggler_rotations == 1, (
            "the cooldown did not hold; a deployment could exhaust every spare "
            "during a single network hiccup")
        assert b.node_id == "b2"

    def test_the_cooldown_expires_and_a_second_rotation_becomes_possible(self):
        a = _stage("a", ["a2"])
        b = _stage("b", ["b2", "b3"])
        c = _stage("c", ["c2"])
        p = _pipe(a, b, c)
        for _ in range(20):
            _feed(a, 10, 1); _feed(b, 900, 1); _feed(c, 10, 1)
            p.check_stragglers()
        assert b.node_id == "b2"
        b._last_rotate_at = time.monotonic() - (R.STRAGGLER_COOLDOWN_S + 1)
        for _ in range(20):
            _feed(a, 10, 1); _feed(b, 900, 1); _feed(c, 10, 1)
            p.check_stragglers()
        assert b.node_id == "b3", "the replacement was also slow but was never replaced"
        assert p.straggler_rotations == 2


class TestRotationLeavesTheStageUsable:
    def test_the_replacement_does_not_inherit_the_straggler_cost(self):
        """Otherwise the new node looks slow on arrival and is evicted immediately."""
        a, b, c = _stage("a", ["a2"]), _stage("b", ["b2", "b3"]), _stage("c", ["c2"])
        p = _pipe(a, b, c)
        for _ in range(20):
            _feed(a, 10, 1); _feed(b, 900, 1); _feed(c, 10, 1)
            if p.check_stragglers() is not None:
                break
        assert b.node_id == "b2"
        # Checked at the instant of rotation: the point is what the replacement
        # *inherits*, and one more traversal would repopulate the counters.
        assert b.cost_ms == 0.0 and b.samples == 0, (
            "the replacement inherited the retired node's cost history")
        assert b.last_ms == 0.0
        assert b.slow_streak == 0

    def test_every_live_session_is_marked_for_rebuild(self):
        """A slow-path rotation is still a rotation: the spare's cache is empty.

        This is the failure mode that would produce *correct-looking* garbage -- a
        decode step at a non-zero position against an empty KV cache -- so it matters
        that the slowness path reuses the same invalidation the death path does
        rather than taking a shortcut because nothing actually broke.
        """
        a, b, c = _stage("a", ["a2"]), _stage("b", ["b2"]), _stage("c", ["c2"])
        p = _pipe(a, b, c)
        b._sessions.update({"s1", "s2"})
        for _ in range(20):
            _feed(a, 10, 1); _feed(b, 900, 1); _feed(c, 10, 1)
            p.check_stragglers()
        assert b.node_id == "b2"
        assert b._needs_hydration == {"s1", "s2"}, (
            "sessions were not invalidated, so the spare will be asked to resume "
            "from a cache it does not have")

    def test_a_straggler_rotation_also_counts_as_a_failover(self):
        """The two share one mechanism, so they must share one set of books."""
        a, b, c = _stage("a", ["a2"]), _stage("b", ["b2"]), _stage("c", ["c2"])
        p = _pipe(a, b, c)
        for _ in range(20):
            _feed(a, 10, 1); _feed(b, 900, 1); _feed(c, 10, 1)
            p.check_stragglers()
        assert p.failovers == 1
        assert p.straggler_rotations == 1


class TestFaultInjectionKnob:
    def test_stall_ms_is_off_unless_set(self, monkeypatch):
        from meshpool.common import faultinject as F
        monkeypatch.delenv("MESHPOOL_STALL_MS", raising=False)
        assert F.active_profile().stall_seconds == 0.0

    def test_stall_ms_is_parsed_and_applied(self, monkeypatch):
        from meshpool.common import faultinject as F
        monkeypatch.setenv("MESHPOOL_STALL_MS", "40")
        p = F.active_profile()
        assert p.stall_seconds == pytest.approx(0.04)
        t0 = time.perf_counter()
        spent = F.stall(p)
        assert spent == pytest.approx(0.04)
        assert time.perf_counter() - t0 >= 0.03

    def test_garbage_is_ignored_rather_than_crashing_a_node(self, monkeypatch):
        from meshpool.common import faultinject as F
        monkeypatch.setenv("MESHPOOL_STALL_MS", "not-a-number")
        assert F.active_profile().stall_seconds == 0.0

    def test_a_production_agent_is_never_implicitly_stalled(self, monkeypatch):
        """The knob rides the same opt-in guard as every other injected fault."""
        from meshpool.common import faultinject as F
        monkeypatch.setenv("MESHPOOL_STALL_MS", "500")
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("MESHPOOL_SIM_HARNESS", raising=False)
        monkeypatch.delenv("MESHPOOL_ALLOW_FAULT_INJECTION", raising=False)
        assert F.active_profile().stall_seconds == 0.0
