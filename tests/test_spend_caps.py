"""Hard spend caps must STOP work, not merely report it.

These tests pin the accounting semantics of ``SpendCapManager``.  The
end-to-end proof that a cap actually halts a running job lives in
``sim/scenarios/verify_spend_cap_hard_stop.py`` — a unit test can only show the
arithmetic is right, not that the enforcement is wired into dispatch.
"""
from __future__ import annotations

import pytest

from meshpool.common.db import Store
from meshpool.common.trust import ContributionLedger
from meshpool.controlplane.spend import (
    DEFAULT_SPEND_CAP_ENV,
    SpendCapManager,
    default_cap_from_env,
)

OWNER = "acct:local:alice"


@pytest.fixture()
def env(tmp_path):
    store = Store(str(tmp_path / "spend.db"))
    ledger = ContributionLedger(store)
    caps = SpendCapManager(store, default_cap=None)
    return store, ledger, caps


def _spend(ledger, owner, units):
    ledger.record_consumption("n1", owner, units, detail="test spend")


class TestUncappedByDefault:
    def test_no_cap_is_reported_honestly(self, env):
        _, ledger, caps = env
        _spend(ledger, OWNER, 500.0)
        st = caps.status(OWNER)
        assert st["enforced"] is False, "an absent cap must not claim enforcement"
        assert st["cap"] is None
        assert st["remaining"] is None
        assert caps.exceeded(OWNER) is False, "no cap can never be exceeded"

    def test_spend_is_tracked_even_while_uncapped(self, env):
        _, ledger, caps = env
        _spend(ledger, OWNER, 12.5)
        assert caps.status(OWNER)["spent"] == pytest.approx(12.5)


class TestCapEnforcement:
    def test_under_cap_is_not_exceeded(self, env):
        _, ledger, caps = env
        caps.set_cap(OWNER, 100.0)
        _spend(ledger, OWNER, 99.0)
        assert caps.exceeded(OWNER) is False
        assert caps.status(OWNER)["remaining"] == pytest.approx(1.0)

    def test_exactly_at_cap_stops_work(self, env):
        """A cap is a stop, so reaching it exactly must already halt work."""
        _, ledger, caps = env
        caps.set_cap(OWNER, 100.0)
        _spend(ledger, OWNER, 100.0)
        assert caps.exceeded(OWNER) is True
        assert caps.status(OWNER)["remaining"] == pytest.approx(0.0)

    def test_over_cap_reports_zero_remaining_not_negative(self, env):
        _, ledger, caps = env
        caps.set_cap(OWNER, 10.0)
        _spend(ledger, OWNER, 37.0)
        st = caps.status(OWNER)
        assert st["exceeded"] is True
        assert st["remaining"] == 0.0, "remaining must clamp at zero, never go negative"

    def test_cap_is_per_owner(self, env):
        _, ledger, caps = env
        caps.set_cap(OWNER, 5.0)
        _spend(ledger, OWNER, 10.0)
        assert caps.exceeded(OWNER) is True
        assert caps.exceeded("acct:local:bob") is False, "bob must not inherit alice's cap"

    def test_zero_cap_blocks_immediately(self, env):
        _, _, caps = env
        caps.set_cap(OWNER, 0.0)
        assert caps.exceeded(OWNER) is True, "a zero cap must block before any spend"

    def test_negative_cap_rejected(self, env):
        _, _, caps = env
        with pytest.raises(ValueError):
            caps.set_cap(OWNER, -1.0)

    def test_clearing_cap_restores_unlimited(self, env):
        _, ledger, caps = env
        caps.set_cap(OWNER, 1.0)
        _spend(ledger, OWNER, 9.0)
        assert caps.exceeded(OWNER) is True
        caps.set_cap(OWNER, None)
        assert caps.exceeded(OWNER) is False
        assert caps.status(OWNER)["enforced"] is False


class TestCapCannotBeBypassed:
    def test_lowering_cap_does_not_grant_fresh_allowance(self, env):
        """Re-setting a cap must not reset the meter, or caps mean nothing."""
        _, ledger, caps = env
        caps.set_cap(OWNER, 100.0)
        _spend(ledger, OWNER, 100.0)
        assert caps.exceeded(OWNER) is True

        caps.set_cap(OWNER, 50.0)
        assert caps.exceeded(OWNER) is True, (
            "changing the cap must not clear accrued spend -- otherwise a user "
            "(or a bug) could bypass the cap forever by re-setting it")
        assert caps.status(OWNER)["spent"] == pytest.approx(100.0)

    def test_raising_cap_resumes_work(self, env):
        _, ledger, caps = env
        caps.set_cap(OWNER, 10.0)
        _spend(ledger, OWNER, 10.0)
        assert caps.exceeded(OWNER) is True
        caps.set_cap(OWNER, 25.0)
        assert caps.exceeded(OWNER) is False, "raising the cap should unblock work"


class TestResetWindow:
    def test_reset_starts_new_window_without_destroying_history(self, env):
        store, ledger, caps = env
        caps.set_cap(OWNER, 10.0)
        _spend(ledger, OWNER, 10.0)
        assert caps.exceeded(OWNER) is True

        caps.reset_spend(OWNER)

        assert caps.exceeded(OWNER) is False
        assert caps.status(OWNER)["spent"] == pytest.approx(0.0)
        # Billing history must survive: gross lifetime spend is unchanged.
        assert caps.gross_consumed(OWNER) == pytest.approx(10.0)
        assert store.ledger_totals_by_owner()[OWNER]["consumed"] == pytest.approx(10.0)

    def test_spend_after_reset_counts_from_zero(self, env):
        _, ledger, caps = env
        caps.set_cap(OWNER, 10.0)
        _spend(ledger, OWNER, 10.0)
        caps.reset_spend(OWNER)
        _spend(ledger, OWNER, 4.0)
        assert caps.status(OWNER)["spent"] == pytest.approx(4.0)
        assert caps.exceeded(OWNER) is False

    def test_reset_preserves_the_cap(self, env):
        _, ledger, caps = env
        caps.set_cap(OWNER, 7.0)
        _spend(ledger, OWNER, 7.0)
        caps.reset_spend(OWNER)
        assert caps.cap_for(OWNER) == pytest.approx(7.0), (
            "resetting the window must not silently un-cap the account")


class TestDefaultCap:
    def test_default_applies_to_owners_with_no_row(self, tmp_path):
        store = Store(str(tmp_path / "d.db"))
        ledger = ContributionLedger(store)
        caps = SpendCapManager(store, default_cap=20.0)
        assert caps.cap_for("acct:new") == pytest.approx(20.0)
        _spend(ledger, "acct:new", 20.0)
        assert caps.exceeded("acct:new") is True

    def test_explicit_cap_overrides_default(self, tmp_path):
        store = Store(str(tmp_path / "d2.db"))
        ledger = ContributionLedger(store)
        caps = SpendCapManager(store, default_cap=20.0)
        caps.set_cap(OWNER, 100.0)
        _spend(ledger, OWNER, 50.0)
        assert caps.exceeded(OWNER) is False

    def test_env_default_parsing(self, monkeypatch):
        monkeypatch.setenv(DEFAULT_SPEND_CAP_ENV, "42.5")
        assert default_cap_from_env() == pytest.approx(42.5)
        monkeypatch.setenv(DEFAULT_SPEND_CAP_ENV, "0")
        assert default_cap_from_env() is None, "0 means uncapped, not 'block everything'"
        monkeypatch.setenv(DEFAULT_SPEND_CAP_ENV, "not-a-number")
        assert default_cap_from_env() is None, "a malformed cap must not block the mesh"
        monkeypatch.delenv(DEFAULT_SPEND_CAP_ENV)
        assert default_cap_from_env() is None


class TestStatusIsSelfConsistent:
    """``status()`` must never report a state that contradicts itself.

    Rounding ``spent`` for display produced exactly that: a response saying
    ``remaining: 0.0`` while also saying ``exceeded: false``.  Worse, it broke
    the obvious client gesture "cap me at what I've already spent" -- the
    rounded figure fed back into ``set_cap`` landed fractionally ABOVE real
    spend, so the cap silently failed to bite.

    The spend value below is chosen deliberately: the defect only appears when
    rounding moves the number **up**.  A value that rounds down still stops work
    and would make these tests vacuous, which is exactly what happened on the
    first attempt at writing them.
    """

    # round(ROUND_UP_SPEND, 6) > ROUND_UP_SPEND  -- verified by the guard below.
    ROUND_UP_SPEND = 2.0 / 3.0

    def test_the_chosen_value_really_rounds_up(self):
        """Guard: if this ever stops holding, the tests below prove nothing."""
        assert round(self.ROUND_UP_SPEND, 6) > self.ROUND_UP_SPEND

    def test_zero_remaining_implies_exceeded(self, env):
        _, ledger, caps = env
        _spend(ledger, OWNER, self.ROUND_UP_SPEND)
        caps.set_cap(OWNER, caps.status(OWNER)["spent"])
        st = caps.status(OWNER)
        assert not (st["remaining"] == 0.0 and st["exceeded"] is False), (
            f"incoherent: remaining=0 but exceeded=False -> {st}")
        assert st["exceeded"] is True

    def test_capping_at_reported_spend_stops_work(self, env):
        """The round-trip a UI actually performs must halt work."""
        _, ledger, caps = env
        _spend(ledger, OWNER, self.ROUND_UP_SPEND)
        reported = caps.status(OWNER)["spent"]
        caps.set_cap(OWNER, reported)
        assert caps.exceeded(OWNER) is True, (
            "setting the cap to the spend the API just reported must stop work; "
            "if display rounding leaks into the cap it silently will not")

    def test_remaining_plus_spent_equals_cap(self, env):
        _, ledger, caps = env
        caps.set_cap(OWNER, 10.0)
        _spend(ledger, OWNER, 1.0 / 7.0)
        st = caps.status(OWNER)
        assert st["spent"] + st["remaining"] == pytest.approx(st["cap"])


class TestPersistence:
    def test_cap_survives_restart(self, tmp_path):
        path = str(tmp_path / "persist.db")
        store = Store(path)
        ledger = ContributionLedger(store)
        caps = SpendCapManager(store, default_cap=None)
        caps.set_cap(OWNER, 15.0)
        _spend(ledger, OWNER, 15.0)
        assert caps.exceeded(OWNER) is True
        store.close()

        # Restart the control plane against the same database.
        store2 = Store(path)
        caps2 = SpendCapManager(store2, default_cap=None)
        assert caps2.cap_for(OWNER) == pytest.approx(15.0), (
            "a cap that evaporates on restart is not a cap")
        assert caps2.exceeded(OWNER) is True
