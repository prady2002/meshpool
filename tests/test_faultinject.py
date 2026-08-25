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
