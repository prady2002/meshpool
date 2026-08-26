"""Tests for meshpool.common.faultinject — activation guards and behaviour."""
from __future__ import annotations

import importlib
import os

import pytest

import meshpool.common.faultinject as fi


# ── Helpers ────────────────────────────────────────────────────────────────

def _reload(monkeypatch, env: dict[str, str]):
    """Reload faultinject with a specific environment snapshot."""
    monkeypatch.setattr(os, "environ", {**os.environ, **env})
    importlib.reload(fi)
    return fi


# ── Tests ──────────────────────────────────────────────────────────────────

class TestOptInGate:
    def test_inert_without_any_optin(self, monkeypatch):
        """No LIE, no ALLOW flag, not under pytest → no fault."""
        monkeypatch.delenv("MESHPOOL_ALLOW_FAULT_INJECTION", raising=False)
        monkeypatch.delenv("MESHPOOL_LIE", raising=False)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("MESHPOOL_SIM_HARNESS", raising=False)
        importlib.reload(fi)
        # _opt_in should be False; active_profile should be inert
        assert not fi._opt_in()
        assert not fi.active_profile().any_active

    def test_inert_with_lie_but_no_optin(self, monkeypatch):
        """MESHPOOL_LIE=1 alone must not activate without the opt-in gate."""
        monkeypatch.delenv("MESHPOOL_ALLOW_FAULT_INJECTION", raising=False)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("MESHPOOL_SIM_HARNESS", raising=False)
        monkeypatch.setenv("MESHPOOL_LIE", "1")
        importlib.reload(fi)
        # Without any opt-in, faults stay off even if LIE is set
        assert not fi._opt_in()
        assert not fi.active_profile().any_active

    def test_active_with_explicit_optin_and_lie(self, monkeypatch):
        """MESHPOOL_ALLOW_FAULT_INJECTION=1 + MESHPOOL_LIE=1 → corrupt_results."""
        monkeypatch.setenv("MESHPOOL_ALLOW_FAULT_INJECTION", "1")
        monkeypatch.setenv("MESHPOOL_LIE", "1")
        importlib.reload(fi)
        assert fi._opt_in()
        assert fi.active_profile().corrupt_results

    def test_active_under_pytest_marker_with_lie(self, monkeypatch):
        """PYTEST_CURRENT_TEST (auto-set by pytest) acts as implicit opt-in."""
        monkeypatch.delenv("MESHPOOL_ALLOW_FAULT_INJECTION", raising=False)
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_faultinject.py::fake")
        monkeypatch.setenv("MESHPOOL_LIE", "1")
        importlib.reload(fi)
        assert fi._opt_in()
        assert fi.active_profile().corrupt_results

    def test_active_under_sim_harness_marker(self, monkeypatch):
        monkeypatch.delenv("MESHPOOL_ALLOW_FAULT_INJECTION", raising=False)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("MESHPOOL_SIM_HARNESS", "1")
        monkeypatch.setenv("MESHPOOL_LIE", "1")
        importlib.reload(fi)
        assert fi._opt_in()
        assert fi.active_profile().corrupt_results


class TestProductionMarker:
    def test_stripped_disables_all(self, monkeypatch):
        """_FAULTINJECT_STRIPPED=True hard-disables faults regardless of env vars."""
        monkeypatch.setenv("MESHPOOL_ALLOW_FAULT_INJECTION", "1")
        monkeypatch.setenv("MESHPOOL_LIE", "1")
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_faultinject.py::fake")
        monkeypatch.setattr(fi, "_FAULTINJECT_STRIPPED", True)
        # _opt_in reads the module-level constant directly
        assert not fi._opt_in()
        assert not fi.active_profile().any_active

    def test_stripped_false_is_default(self):
        assert fi._FAULTINJECT_STRIPPED is False


class TestCorruptResult:
    def test_warns_on_activation(self, monkeypatch):
        monkeypatch.setenv("MESHPOOL_ALLOW_FAULT_INJECTION", "1")
        monkeypatch.setenv("MESHPOOL_LIE", "1")
        importlib.reload(fi)
        profile = fi.active_profile()
        assert profile.corrupt_results

        warnings_issued: list[str] = []
        monkeypatch.setattr(fi.log, "warning", lambda msg, *a, **kw: warnings_issued.append(msg))

        result = fi.corrupt_result(42, profile)

        assert result == 43  # int -> int + 1
        assert any("FAULT INJECTION ACTIVE" in w for w in warnings_issued)

    def test_no_side_effect_when_inactive(self, monkeypatch):
        monkeypatch.delenv("MESHPOOL_LIE", raising=False)
        monkeypatch.setenv("MESHPOOL_ALLOW_FAULT_INJECTION", "1")
        importlib.reload(fi)
        profile = fi.active_profile()
        assert not profile.corrupt_results
        assert fi.corrupt_result(42, profile) == 42

    @pytest.mark.parametrize("v, expected", [
        (True, False),
        (False, True),
        (5, 6),
        (3.14, 4.14),
        ("hello", "helloX"),
        ([1, 2, 3], [2, 3]),
        ([], [1]),
        ({"a": 1}, {"a": 2}),
    ])
    def test_corruption_shapes(self, v, expected, monkeypatch):
        monkeypatch.setenv("MESHPOOL_ALLOW_FAULT_INJECTION", "1")
        monkeypatch.setenv("MESHPOOL_LIE", "1")
        importlib.reload(fi)
        profile = fi.FaultProfile(corrupt_results=True)
        result = fi.corrupt_result(v, profile)
        if isinstance(expected, float):
            assert abs(result - expected) < 1e-9
        else:
            assert result == expected


class TestDishonestPeerEmulation:
    """The dishonest-peer fault is the one that must never fire in production.

    Every other fault is loud: a killed node is gone, a stalled node is slow. A node
    that serves subtly wrong weight bytes is silent -- the model loads and answers
    fluently and is simply wrong -- so the guards on this one carry more weight than
    the guards on the others, and they are tested rather than assumed.
    """

    def test_inert_when_not_opted_in(self, monkeypatch):
        """The exact production configuration: no opt-in, no env var, nothing set."""
        monkeypatch.delenv("MESHPOOL_ALLOW_FAULT_INJECTION", raising=False)
        monkeypatch.delenv("MESHPOOL_CORRUPT_BLOBS", raising=False)
        monkeypatch.delenv("MESHPOOL_LIE", raising=False)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("MESHPOOL_SIM_HARNESS", raising=False)
        importlib.reload(fi)
        raw = b"the weights of a model somebody is paying to run"
        assert not fi.active_profile().corrupt_blobs
        assert fi.corrupt_bytes(raw) == raw

    def test_env_var_alone_does_not_arm_it(self, monkeypatch):
        """MESHPOOL_CORRUPT_BLOBS=1 without the opt-in gate must still be inert.

        A single stray environment variable inherited from a CI job must not be able
        to turn a production node into a poisoned-weight source.
        """
        monkeypatch.delenv("MESHPOOL_ALLOW_FAULT_INJECTION", raising=False)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("MESHPOOL_SIM_HARNESS", raising=False)
        monkeypatch.setenv("MESHPOOL_CORRUPT_BLOBS", "1")
        importlib.reload(fi)
        raw = b"the weights of a model somebody is paying to run"
        assert not fi._opt_in()
        assert not fi.active_profile().corrupt_blobs
        assert fi.corrupt_bytes(raw) == raw

    def test_setter_refuses_without_optin(self, monkeypatch):
        monkeypatch.delenv("MESHPOOL_ALLOW_FAULT_INJECTION", raising=False)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("MESHPOOL_SIM_HARNESS", raising=False)
        importlib.reload(fi)
        assert fi.set_corrupt_blobs(True) is False
        assert fi.corrupt_bytes(b"abc") == b"abc"

    def test_corruption_is_shaped_to_defeat_everything_but_a_hash(self):
        """Same length, still valid base64, exactly one byte wrong.

        If the fault truncated or emptied the payload, a puller that checked only the
        length would pass the integrity gate without ever hashing anything -- so the
        gate would prove nothing about content addressing. The shape of the lie is
        what makes the test meaningful.
        """
        import base64
        import hashlib
        raw = bytes(range(256)) * 8
        bad = fi.corrupt_bytes(raw, fi.FaultProfile(corrupt_blobs=True))
        assert len(bad) == len(raw), "a length check must not be able to catch it"
        assert bad != raw
        assert sum(a != b for a, b in zip(raw, bad)) == 1, "exactly one byte differs"
        assert hashlib.sha256(bad).hexdigest() != hashlib.sha256(raw).hexdigest()
        assert base64.b64decode(base64.b64encode(bad), validate=True) == bad

    def test_empty_payload_is_left_alone(self):
        """Nothing to corrupt, and returning junk for an empty blob would be a
        different bug wearing this fault's clothes."""
        assert fi.corrupt_bytes(b"", fi.FaultProfile(corrupt_blobs=True)) == b""

    def test_any_active_reports_it(self):
        assert fi.FaultProfile(corrupt_blobs=True).any_active
        assert not fi.FaultProfile().any_active
