"""The sim harness must survive a port collision instead of timing out.

Every ``sim/scenarios/verify_*.py`` gate hard-codes a control-plane port, and a
cluster of N agents silently claims N more (``cp_port + 1 + idx``). A scan of the
gate suite found 32 ports claimed by more than one gate that way -- the declared
``CP_PORT`` constants are all distinct, but the *derived agent ports* collide with
other gates' control-plane ports.

Gates run serially, so a collision only bites when the previous gate's sockets
have not been released yet. When it does bite it surfaces 25 seconds later as
``TimeoutError: service not ready at 127.0.0.1:<port>``, which reads like a boot
failure rather than a port clash -- so it gets misdiagnosed as flakiness and
retried instead of fixed.

These tests pin the behaviour that removes the failure class.
"""
from __future__ import annotations

import socket

import pytest

from sim.harness import Harness, _free_port


def _occupy() -> tuple[socket.socket, int]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    return s, s.getsockname()[1]


class TestFreePort:
    def test_a_free_port_is_returned_unchanged(self):
        """The common case must be a no-op.

        Gate scenarios document their ports, and operators grep for them when a
        run misbehaves. If the harness drifted off the requested port even when
        it was available, every one of those documented numbers would become a
        lie and debugging would get harder, not easier.
        """
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
        probe.close()

        assert _free_port(free) == free

    def test_an_occupied_port_falls_back_to_a_different_one(self):
        sock, taken = _occupy()
        try:
            got = _free_port(taken)
            assert got != taken
        finally:
            sock.close()

    def test_the_fallback_port_is_actually_bindable(self):
        """A fallback that hands back an unusable port just moves the failure.

        ``_free_port`` picks its replacement by binding port 0 and reading back
        the assignment, then closing the socket -- so this asserts the port is
        still claimable in the window afterwards, which is what the harness
        relies on.
        """
        sock, taken = _occupy()
        try:
            got = _free_port(taken)
            confirm = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                confirm.bind(("127.0.0.1", got))
            finally:
                confirm.close()
        finally:
            sock.close()

    def test_zero_never_returns_zero(self):
        """``preferred=0`` must allocate, not echo.

        Binding port 0 always succeeds, so a naive bind-and-keep would return a
        literal 0. The harness writes the result straight into ``MESHPOOL_PORT``
        and then health-checks ``http://host:0`` -- 25 seconds of waiting for a
        port nothing can ever listen on, reported as a boot timeout.
        """
        got = _free_port(0)
        assert got > 0

    def test_repeated_fallbacks_do_not_collide(self):
        """Two clusters asking for the same busy port must not be handed the
        same replacement, which is precisely the situation being fixed."""
        sock, taken = _occupy()
        holders = []
        try:
            first = _free_port(taken)
            hold = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            hold.bind(("127.0.0.1", first))
            hold.listen(1)
            holders.append(hold)

            second = _free_port(taken)
            assert second != first
            assert second != taken
        finally:
            for h in holders:
                h.close()
            sock.close()


class TestHarnessPortResolution:
    """These drive the real ``Harness`` methods with process spawn stubbed out.

    Asserting on a re-implementation of the branch would prove nothing: it would
    pass just as happily if ``harness.py`` were reverted. Patching only
    ``subprocess.Popen`` and the readiness wait keeps the port-selection logic
    under test genuinely the shipped one.
    """

    @staticmethod
    def _stub(monkeypatch, captured):
        import sim.harness as mod

        class _FakeProc:
            pid = 4242

            def poll(self):
                return None

        def _fake_popen(cmd, **kwargs):
            captured.append(kwargs.get("env", {}))
            return _FakeProc()

        monkeypatch.setattr(mod.subprocess, "Popen", _fake_popen)
        monkeypatch.setattr(mod.Harness, "_wait_http",
                            lambda self, url, timeout=25: None)

    def test_control_plane_port_is_pinned_across_restart(self, tmp_path, monkeypatch):
        """A restart must reuse the same port.

        ``verify_chaos`` kills the coordinator and starts it again while the
        agents keep running. Those agents hold ``MESHPOOL_CONTROL_PLANE_URL``
        from their own launch, so if the restart drifted to a new port they
        would dial the old one forever and the coordinator-restart self-heal --
        the entire point of that gate -- could never succeed. Worse, the gate
        would fail with a reconnect timeout that looks like a self-heal bug
        rather than a port problem.
        """
        envs: list[dict] = []
        self._stub(monkeypatch, envs)

        sock, taken = _occupy()
        try:
            h = Harness(n_agents=0, cp_port=taken, data_dir=str(tmp_path / "d"))
            h.start_control_plane()
            first = h.cp_port
            assert first != taken, "did not fall back off the occupied port"

            # The collision is gone by the time of the restart. The harness must
            # still stay put rather than snapping back to the declared port.
            sock.close()
            h.start_control_plane()
            assert h.cp_port == first
            assert envs[-1]["MESHPOOL_PORT"] == str(first)
        finally:
            sock.close()

    def test_agent_restart_keeps_its_original_port(self, tmp_path, monkeypatch):
        """``start_agent(idx)`` doubles as the restart path.

        Several gates kill an agent and bring it back with the same index. A
        node is keyed by the URL it advertises, so a drifting port would
        register a *second* identity instead of restoring the first -- and
        failover assertions that count healthy nodes would quietly measure the
        wrong thing while still going green.
        """
        envs: list[dict] = []
        self._stub(monkeypatch, envs)

        h = Harness(n_agents=2, cp_port=0, data_dir=str(tmp_path / "d"))
        h.start_control_plane()
        h.start_agent(0)
        h.start_agent(1)
        original = list(h.agent_ports)
        assert len(original) == 2 and original[0] != original[1]

        h.start_agent(0)
        assert h.agent_ports[0] == original[0]
        assert envs[-1]["MESHPOOL_AGENT_PORT"] == str(original[0])

    def test_agents_avoid_a_port_another_gate_is_holding(self, tmp_path, monkeypatch):
        """The exact scenario from the scan: an agent's *derived* port is the
        control-plane port of a gate whose sockets have not been released."""
        envs: list[dict] = []
        self._stub(monkeypatch, envs)

        h = Harness(n_agents=1, cp_port=0, data_dir=str(tmp_path / "d"))
        h.start_control_plane()

        squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            squatter.bind(("127.0.0.1", h.cp_port + 1))
            squatter.listen(1)
        except OSError:
            pytest.skip("could not reserve the derived agent port")

        try:
            h.start_agent(0)
            assert h.agent_ports[0] != h.cp_port + 1
        finally:
            squatter.close()


@pytest.mark.slow
class TestHarnessSurvivesRealCollision:
    def test_control_plane_boots_when_its_declared_port_is_taken(self, tmp_path):
        """End-to-end proof, not just the helper in isolation."""
        import httpx

        sock, taken = _occupy()
        h = Harness(n_agents=0, cp_port=taken, data_dir=str(tmp_path / "cp"))
        try:
            h.start_control_plane()
            assert h.cp_port != taken, "harness stayed on the occupied port"
            r = httpx.get(f"{h.cp_url()}/api/v1/health", timeout=10.0)
            assert r.status_code == 200
        finally:
            h.stop()
            sock.close()
