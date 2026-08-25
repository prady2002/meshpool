"""Tests for meshpool.agent.probe — measurement accuracy and graceful degradation."""
from __future__ import annotations

import subprocess
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
