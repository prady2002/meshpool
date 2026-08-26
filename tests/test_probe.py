"""Tests for meshpool.agent.probe — measurement accuracy and graceful degradation."""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import pytest

from meshpool.agent.probe import (
    _detect_gpu,
    _flops_estimate,
    _measure_link_detail,
    measure_link,
)
from meshpool.common.models import LinkStats


# ── Minimal local HTTP server for latency tests ───────────────────────────

class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        # HEAD to /api/v1/probe/download → 404 (endpoint absent)
        if self.path.startswith("/api/v1/probe"):
            self.send_response(404)
            self.end_headers()
        else:
            self.send_response(200)
            self.end_headers()

    def log_message(self, *args):
        pass  # suppress test output noise


def _start_test_server() -> tuple[HTTPServer, int]:
    server = HTTPServer(("127.0.0.1", 0), _HealthHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


@pytest.fixture(scope="module")
def local_server():
    server, port = _start_test_server()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


# ── measure_link / _measure_link_detail ───────────────────────────────────

class TestMeasureLink:
    def test_returns_link_stats_type(self, local_server):
        result = measure_link(local_server)
        assert isinstance(result, LinkStats)

    def test_never_raises_on_unreachable_host(self):
        # Completely unreachable host — must not propagate an exception
        result = measure_link("http://127.0.0.2:19999")
        assert isinstance(result, LinkStats)

    def test_latency_sane_for_localhost(self, local_server):
        result = measure_link(local_server)
        # Localhost RTT must be < 200 ms even on a heavily loaded CI runner
        assert result.latency_ms >= 0.0
        assert result.latency_ms < 200.0

    def test_detail_latency_stats_sane(self, local_server):
        detail = _measure_link_detail(local_server)
        assert detail.measured is True
        assert detail.latency_p50_ms >= 0.0
        assert detail.latency_p90_ms >= detail.latency_p50_ms
        assert detail.latency_jitter_ms >= 0.0
        assert detail.first_connect_ms >= 0.0

    def test_bandwidth_none_when_endpoint_absent(self, local_server):
        """Without /api/v1/probe/download, bandwidth must be None — not fabricated."""
        detail = _measure_link_detail(local_server)
        assert detail.bandwidth_mbps is None

    def test_unknown_vs_zero_distinction(self, local_server):
        """The _LinkMeasurement uses None for unmeasured, not 0.0 (which would mean
        an actual zero-throughput measurement)."""
        detail = _measure_link_detail(local_server)
        # bandwidth_mbps being None means "not measured", which is distinct from 0.0
        assert detail.bandwidth_mbps is None

    def test_to_link_stats_uses_zero_for_none_bw(self, local_server):
        """to_link_stats converts None bandwidth to 0.0 for wire compat (see note)."""
        detail = _measure_link_detail(local_server)
        ls = detail.to_link_stats()
        assert ls.bandwidth_mbps == 0.0

    def test_failed_measurement_returns_unmeasured(self):
        detail = _measure_link_detail("http://127.0.0.2:19999")
        assert detail.measured is False
        assert detail.bandwidth_mbps is None


# ── _detect_gpu ────────────────────────────────────────────────────────────

class TestDetectGpu:
    def test_degrades_gracefully_when_tools_absent(self, monkeypatch):
        """If every GPU tool fails, we must get (False, 0) — never raise."""
        def _raise(*args, **kwargs):
            raise FileNotFoundError("no such tool")
        monkeypatch.setattr(subprocess, "run", _raise)
        has_gpu, vram, names = _detect_gpu()
        assert has_gpu is False
        assert vram == 0
        assert names == []

    def test_returns_bool_int(self):
        has_gpu, vram, names = _detect_gpu()
        assert isinstance(has_gpu, bool)
        assert isinstance(vram, int)
        assert vram >= 0
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_nonzero_vram_implies_gpu(self):
        has_gpu, vram, _names = _detect_gpu()
        if vram > 0:
            assert has_gpu is True

    def test_named_adapter_implies_gpu(self):
        """A name without a VRAM figure is still a GPU.

        Integrated adapters report shared memory, so vram is legitimately 0/None
        for them; keying presence off VRAM would erase every iGPU in the mesh.
        """
        has_gpu, _vram, names = _detect_gpu()
        if names:
            assert has_gpu is True


# ── _flops_estimate ────────────────────────────────────────────────────────

class TestFlopsEstimate:
    def test_returns_positive(self):
        result = _flops_estimate()
        assert result > 0.0

    def test_completes_quickly(self):
        t0 = time.perf_counter()
        _flops_estimate()
        elapsed = time.perf_counter() - t0
        # Must finish well under 3 s even on a slow runner
        assert elapsed < 3.0

    def test_returns_float(self):
        assert isinstance(_flops_estimate(), float)


# ── tool output the local codepage cannot decode ───────────────────────────

class TestToolOutputEncoding:
    """A GPU must not disappear because its name has an awkward byte in it.

    Every hardware detector here shells out to a vendor tool and reads its text.
    On Windows those tools write pipes in the *OEM* codepage (cp437/cp850/...)
    while Python decodes with the *ANSI* one (cp1252/...). The two differ on
    essentially every Windows install, so one non-ASCII byte anywhere in a GPU
    name is enough to break the decode.

    What makes that dangerous rather than merely annoying is the failure shape.
    The decode happens on subprocess's reader thread, so it does not propagate:
    the thread dies, ``CompletedProcess.stdout`` comes back as **None**, the
    ``.strip()`` in the caller raises AttributeError, and the broad ``except
    Exception`` -- there to let a missing tool degrade quietly -- swallows it.
    ``_detect_gpu`` then falls through every branch and reports no GPU at all.

    The owner sees nothing. Their machine is quietly classified CPU-only, never
    receives inference work, and earns nothing, on a box with a 4090 in it.
    """

    # A name carrying 0x90, which is unmapped in cp1252. Real cases: any
    # localized Windows, or a vendor string with a trademark/dash glyph.
    RAW = (b'[{"Name":"NVIDIA GeForce RTX 4090 \x90 OEM",'
           b'"AdapterRAM":25757220864}]')

    @staticmethod
    def _run_like_subprocess(raw: bytes, **kwargs):
        """Reproduce CPython's decode behaviour, including the None on failure.

        Verified against the real thing: with a bare ``text=True`` and no
        ``errors`` policy, an undecodable byte leaves ``stdout`` as None rather
        than raising.
        """
        import locale
        enc = kwargs.get("encoding") or locale.getpreferredencoding(False)
        errs = kwargs.get("errors")
        try:
            return raw.decode(enc, errs) if errs else raw.decode(enc)
        except UnicodeDecodeError:
            return None

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_the_hazard_is_real_on_this_machine(self):
        """Paired non-vacuity check: prove the trap exists before proving it is
        avoided. Skips where the locale is UTF-8 and there is nothing to trip on."""
        import locale
        if locale.getpreferredencoding(False).lower().replace("-", "") == "utf8":
            pytest.skip("UTF-8 locale: no decode hazard to demonstrate")
        r = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.stdout.buffer.write({self.RAW!r})"],
            capture_output=True, text=True, timeout=30)
        assert r.stdout is None, (
            "expected the locale decoder to fail and null out stdout; if this "
            "ever stops being true the test below proves nothing")

    def test_errors_replace_keeps_the_output_readable(self):
        r = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.stdout.buffer.write({self.RAW!r})"],
            capture_output=True, text=True, errors="replace", timeout=30)
        assert isinstance(r.stdout, str)
        # The bytes we actually parse are ASCII, so they survive intact.
        assert '"AdapterRAM":25757220864' in r.stdout
        assert "RTX 4090" in r.stdout

    def test_a_gpu_whose_name_cannot_be_decoded_is_still_detected(self, monkeypatch):
        """The end-to-end claim: undecodable name in, GPU still out."""
        import platform as _platform

        import meshpool.agent.probe as probe

        monkeypatch.setattr(probe.platform, "system", lambda: "Windows")

        def fake_run(cmd, **kwargs):
            if cmd[0] != "powershell":
                raise FileNotFoundError(cmd[0])  # no nvidia-smi / rocm-smi here
            text = self._run_like_subprocess(self.RAW, **kwargs)
            return subprocess.CompletedProcess(cmd, 0, text, "")

        monkeypatch.setattr(probe.subprocess, "run", fake_run)
        has_gpu, vram, names = _detect_gpu()

        assert has_gpu is True, "a 24 GB GPU vanished because of one byte in its name"
        assert vram == 25757220864 // (1024 * 1024)
        assert names and "RTX 4090" in names[0]
        assert _platform  # keep the import meaningful for linters

    def test_a_none_stdout_never_raises(self, monkeypatch):
        """Belt and braces: even if some future tool still returns None, the
        probe must degrade to 'no GPU' rather than take the agent down."""
        import meshpool.agent.probe as probe

        monkeypatch.setattr(probe.subprocess, "run",
                            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, None, ""))
        has_gpu, vram, names = _detect_gpu()
        assert (has_gpu, vram, names) == (False, 0, [])

    def test_every_capture_in_probe_is_decode_safe(self):
        """Structural guard for call site number twelve.

        The fix is one keyword argument, which makes it exactly the kind of thing
        that gets left off the next detector someone adds. Pin it in source.
        """
        import re
        from pathlib import Path

        src = Path(probe_module_file()).read_text(encoding="utf-8")
        calls = re.findall(r"subprocess\.run\((?:[^()]|\([^()]*\))*\)", src,
                           flags=re.S)
        capturing = [c for c in calls
                     if "text=True" in c or "encoding=" in c]
        assert capturing, "no text-capturing subprocess calls found — bad matcher"
        offenders = [" ".join(c.split())[:110] for c in capturing
                     if "errors=" not in c]
        assert not offenders, (
            "these probe subprocess calls decode tool output with the ambient "
            "locale and will null out stdout on one odd byte; add "
            'errors="replace":\n  ' + "\n  ".join(offenders))


def probe_module_file() -> str:
    import meshpool.agent.probe as probe
    return probe.__file__
