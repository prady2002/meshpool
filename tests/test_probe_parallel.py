"""The hardware probe must not block node startup on serial subprocess waits.

``probe_hardware_detail`` runs five independent detectors, and every one of them
blocks on a child process -- ``nvidia-smi``, ``rocm-smi``, ``lscpu``, ``pmset``,
and on Windows several ``powershell -Command Get-CimInstance`` calls that cost
seconds just to start the shell. Run one after another their timeouts add up:
measured at 10.2 s on an idle machine.

That number is on the critical path of node startup. The agent does not bind its
HTTP port until the probe returns, so for the whole duration the node is
unreachable and unregistered. It was making the sim harness time out agents that
were perfectly healthy -- a 25 s readiness budget against a probe that stretched
to 22 s under load -- which presented as ``service not ready at 127.0.0.1:8126``
and looked like a boot failure rather than a slow probe.

These tests pin the two properties that fix requires: the detectors overlap, and
one of them failing still cannot take the whole probe down.
"""
from __future__ import annotations

import time

from meshpool.agent import probe


_DELAY = 0.4
_DETECTORS = {
    "_detect_gpu": (False, 0, []),
    "_detect_battery": (None, None),
    "_detect_cpu_model": "stub-cpu",
    "_total_ram_mb": 4096,
    "_detect_thermal_state": "nominal",
}


def _install_slow_stubs(monkeypatch, delay=_DELAY):
    """Replace every detector with a sleep returning a fixed value.

    Sleeping is the right stand-in: the real detectors are blocking waits on a
    child process, so they release the GIL exactly the way ``time.sleep`` does.
    A CPU-bound stub would measure Python's threading limits instead of the
    property under test.
    """
    for name, value in _DETECTORS.items():
        def _make(v):
            def _stub(*_a, **_kw):
                time.sleep(delay)
                return v
            return _stub
        monkeypatch.setattr(probe, name, _make(value))


class TestDetectorsRunConcurrently:
    def test_total_time_is_far_below_the_serial_sum(self, monkeypatch):
        _install_slow_stubs(monkeypatch)

        t0 = time.perf_counter()
        probe.probe_hardware_detail()
        elapsed = time.perf_counter() - t0

        serial = _DELAY * len(_DETECTORS)
        # Generous bound: parallel should land near _DELAY (0.4s), serial at 2.0s.
        # Asserting < half the serial cost cannot be met by accident but leaves
        # ample room for scheduling noise on a loaded CI machine.
        assert elapsed < serial / 2, (
            f"probe took {elapsed:.2f}s; serial would be {serial:.2f}s -- "
            "the detectors appear to be running one after another"
        )

    def test_every_field_still_comes_back(self, monkeypatch):
        """Parallelism is worthless if it drops data.

        A pool that swallowed a result would leave a node advertising unknown
        RAM or no GPU, and the scheduler would place work accordingly -- a
        silent capacity bug rather than a loud failure.
        """
        _install_slow_stubs(monkeypatch)

        d = probe.probe_hardware_detail()

        assert d.ram_mb == 4096
        assert d.cpu_model == "stub-cpu"
        assert d.thermal_state == "nominal"
        assert d.gpu is False
        assert d.cpu_cores >= 1
        assert d.disk_total_mb and d.disk_total_mb > 0


class TestOneFailingDetectorDoesNotSinkTheProbe:
    def test_a_raising_detector_degrades_to_a_fallback(self, monkeypatch):
        """The serial version degraded per detector; the pool must too.

        Each detector already swallows its own errors, but an unexpected escape
        (a new detector, a changed library) must not turn a machine that can
        report 11 of 12 fields into one that reports none. A node with no
        capability record is unschedulable, so this failure mode would quietly
        remove machines from the mesh.
        """
        _install_slow_stubs(monkeypatch)

        def _boom(*_a, **_kw):
            raise RuntimeError("nvidia-smi exploded")

        monkeypatch.setattr(probe, "_detect_gpu", _boom)

        d = probe.probe_hardware_detail()

        assert d.gpu is False
        assert d.vram_mb is None
        # The detectors that did work must still be reported.
        assert d.ram_mb == 4096
        assert d.cpu_model == "stub-cpu"

    def test_all_detectors_failing_still_yields_a_usable_record(self, monkeypatch):
        def _boom(*_a, **_kw):
            raise RuntimeError("nope")

        for name in _DETECTORS:
            monkeypatch.setattr(probe, name, _boom)

        d = probe.probe_hardware_detail()

        assert d.cpu_cores >= 1
        assert d.thermal_state == "unknown"
        assert d.battery_powered is None
        assert d.ram_mb is None


class TestRealProbeStaysFast:
    def test_unstubbed_probe_completes_well_inside_the_readiness_budget(self):
        """Guards the real thing, not just the stubs.

        The sim harness allows a fixed budget for an agent to become
        health-checkable, and the probe runs before the port is bound. If this
        regresses past the budget, agents start failing to boot in a way that
        reads as flakiness. Measured at ~3s after parallelising, ~10s before.
        """
        t0 = time.perf_counter()
        d = probe.probe_hardware_detail()
        elapsed = time.perf_counter() - t0

        assert d.cpu_cores >= 1
        assert elapsed < 9.0, (
            f"hardware probe took {elapsed:.1f}s; it blocks the agent from "
            "binding its port, so this must stay well under the readiness budget"
        )
