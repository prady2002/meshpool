"""Link probing must measure, never fabricate.

The original implementation inferred "bandwidth" from the size of a tiny
`/api/v1/health` JSON body, which measures request overhead rather than throughput
and produced numbers that placement then trusted. These tests pin the replacement:

* real payload endpoints exist and honour the requested size;
* throughput is derived from bytes actually transferred;
* both directions are probed, because residential links are asymmetric and a
  pipeline stage is bounded by its *uplink*;
* when the endpoints are absent the result is `None` ("unknown"), never a guess.
"""
from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from meshpool.agent import probe as P
from meshpool.common.config import Settings
from meshpool.controlplane.app import build_app


def _cp_app(tmp_path) -> FastAPI:
    """A real control plane, built the way every other test builds one."""
    settings = Settings.load(scale="home", data_dir=str(tmp_path),
                             host="127.0.0.1", port=8123)
    return build_app(settings)


# --------------------------------------------------------------------------- #
# server side
# --------------------------------------------------------------------------- #
def test_probe_download_returns_requested_size(tmp_path):
    with TestClient(_cp_app(tmp_path)) as c:
        for size in (1024, 64 * 1024, 256 * 1024):
            r = c.get("/api/v1/probe/download", params={"size": size})
            assert r.status_code == 200
            assert len(r.content) == size, "probe must return exactly what was asked for"
            assert int(r.headers["X-MeshPool-Probe-Bytes"]) == size


def test_probe_download_head_advertises_existence(tmp_path):
    with TestClient(_cp_app(tmp_path)) as c:
        assert c.head("/api/v1/probe/download").status_code == 200


def test_probe_download_is_capped(tmp_path):
    """A probe endpoint must not be usable as an amplification / DoS lever."""
    with TestClient(_cp_app(tmp_path)) as c:
        r = c.get("/api/v1/probe/download", params={"size": 999 * 1024 * 1024})
        assert r.status_code == 200
        assert len(r.content) <= 8 * 1024 * 1024


def test_probe_download_payload_is_incompressible(tmp_path):
    """If the payload were zeros, gzip would make a 10 Mbps link look like 1 Gbps."""
    import zlib
    with TestClient(_cp_app(tmp_path)) as c:
        body = c.get("/api/v1/probe/download", params={"size": 256 * 1024}).content
    assert len(zlib.compress(body)) > 0.95 * len(body), \
        "probe payload compressed away — bandwidth numbers would be inflated"


def test_probe_upload_counts_bytes_received(tmp_path):
    with TestClient(_cp_app(tmp_path)) as c:
        payload = b"x" * (128 * 1024)
        r = c.post("/api/v1/probe/upload", content=payload)
        assert r.status_code == 200
        assert r.json()["bytes"] == len(payload)
        assert r.json()["server_recv_ms"] >= 0.0


def test_probe_upload_rejects_oversize(tmp_path):
    with TestClient(_cp_app(tmp_path)) as c:
        r = c.post("/api/v1/probe/upload", content=b"x" * (9 * 1024 * 1024))
        assert r.status_code == 413


# --------------------------------------------------------------------------- #
# client side
# --------------------------------------------------------------------------- #
def _throttled_app(down_mbps: float, up_mbps: float | None = None) -> FastAPI:
    """A probe server that transfers at a known rate, so the measurement can be
    checked against a *known* truth instead of against itself.

    NOTE: ``Request``/``Response`` are imported at module scope on purpose. This module
    uses ``from __future__ import annotations``, so FastAPI resolves the string
    annotation ``"Request"`` against the *module* globals; a function-local import
    would make it a query parameter and every POST would 422.
    """
    import time as _t
    up_mbps = down_mbps if up_mbps is None else up_mbps
    app = FastAPI()

    @app.get("/api/v1/health")
    async def health():
        return {"ok": True}

    @app.head("/api/v1/probe/download")
    async def head():
        return Response(status_code=200)

    @app.get("/api/v1/probe/download")
    async def dl(size: int = 65536):
        n = min(size, 8 * 1024 * 1024)
        _t.sleep((n * 8) / (down_mbps * 1e6))
        return Response(content=b"\0" * n, media_type="application/octet-stream")

    @app.post("/api/v1/probe/upload")
    async def up(request: Request):
        body = await request.body()
        _t.sleep((len(body) * 8) / (up_mbps * 1e6))
        return {"bytes": len(body), "server_recv_ms": 0.0}

    return app


@pytest.mark.parametrize("mbps", [8.0, 40.0])
def test_measured_bandwidth_tracks_a_known_rate(mbps):
    """The measurement must follow the truth, not merely return *some* number.

    A test that only asserts `bw > 0` would pass on a hardcoded constant. Two
    different injected rates must produce two correspondingly different readings.
    """
    app = _throttled_app(mbps)
    with TestClient(app) as c:
        got = P._measure_download_bw(c, "http://probe.test/api/v1/probe/download")
    # Loopback ASGI has real per-request overhead, so allow a wide band; the point
    # is that the reading is anchored to the injected rate, not invented.
    assert 0.3 * mbps <= got <= 3.0 * mbps, f"measured {got:.2f} Mbps for a {mbps} Mbps link"


def test_download_and_upload_are_measured_independently():
    """Asymmetric links must read as asymmetric — otherwise placement will put a
    pipeline stage on a node whose uplink cannot carry it."""
    down_mbps, up_mbps = 100.0, 5.0
    app = _throttled_app(down_mbps, up_mbps)

    with TestClient(app) as c:
        d = P._measure_download_bw(c, "http://probe.test/api/v1/probe/download")
        u = P._measure_upload_bw(c, "http://probe.test/api/v1/probe/upload")

    assert d > u, f"asymmetric link read as down={d:.1f} up={u:.1f} Mbps"
    assert u < 0.5 * d, "the 20x asymmetry collapsed — the two directions are not " \
                        "being measured independently"


def test_absent_endpoint_reports_unknown_not_zero_guess():
    """No probe endpoint => bandwidth is None (unknown). Never a fabricated value."""
    app = FastAPI()

    @app.get("/api/v1/health")
    async def health():
        return {"ok": True}

    import meshpool.agent.probe as mod
    real = httpx.Client

    def _patched(*a, **kw):        # route every request into the bare app
        return TestClient(app)

    httpx.Client = _patched
    try:
        m = mod._measure_link_detail("http://probe.test")
    finally:
        httpx.Client = real

    assert m.measured is True
    assert m.bandwidth_mbps is None, "missing endpoint must read as unknown, not 0.0"
    assert m.upload_mbps is None
    # ...and the lossy wire format degrades to 0.0 only at the boundary.
    assert m.to_link_stats().bandwidth_mbps == 0.0


# --------------------------------------------------------------------------- #
# co-located nodes: the shared per-IP probe budget must not starve honest peers
# --------------------------------------------------------------------------- #
def test_probe_ramp_is_bounded(tmp_path):
    """One agent's download probe must cost a bounded number of bytes.

    ``size_bytes = min(size_bytes * 2, cap)`` pins the request size at the 8 MB
    cap forever, so on a fast link (loopback, LAN, gigabit fibre) the ramp used to
    loop at the cap until the *server* cut it off with a 429. Measured on
    loopback: a single agent consumed the entire per-IP budget at startup, and the
    next node to join from the same public address advertised bandwidth_mbps=0
    permanently. Every node behind one NAT was affected, and CGNAT -- which the
    plan budgets at 25-40% of the fleet -- shares one address across thousands of
    subscribers.
    """
    served = {"bytes": 0, "requests": 0}
    app = FastAPI()

    @app.get("/api/v1/probe/download")
    async def dl(size: int = 65536):
        served["bytes"] += size
        served["requests"] += 1
        return Response(content=b"\0" * size, media_type="application/octet-stream")

    with TestClient(app) as c:
        mbps = P._measure_download_bw(c, "http://probe.test/api/v1/probe/download")

    assert mbps > 0, "probe measured nothing"
    # warm-up + the 64K..8M doubling ramp + at most one final sample at the cap.
    assert served["requests"] <= 12, (
        f"ramp issued {served['requests']} requests; it is not bounded")
    assert served["bytes"] <= 40 * 1024 * 1024, (
        f"ramp pulled {served['bytes'] / 1e6:.1f} MB; a single probe must stay well "
        f"inside the control plane's shared per-IP budget")


def test_many_nodes_behind_one_ip_can_all_measure(tmp_path):
    """Ten agents sharing one public IP must each get a real number.

    This is the CGNAT case, and it is the population the product most needs to
    serve. A budget that only fits a couple of probes makes every node behind a
    carrier-grade NAT unplaceable.
    """
    app = _cp_app(tmp_path)
    measured = []
    with TestClient(app) as c:
        for _ in range(10):
            d = P._measure_download_bw(c, "http://probe.test/api/v1/probe/download")
            u = P._measure_upload_bw(c, "http://probe.test/api/v1/probe/upload")
            measured.append((d, u))

    zeros = [i for i, (d, u) in enumerate(measured) if d <= 0 or u <= 0]
    assert not zeros, (
        f"nodes {zeros} measured nothing — the shared per-IP probe budget starved "
        f"them, so they would advertise bandwidth_mbps=0 forever")


def test_throttled_probe_still_returns_what_it_measured():
    """A 429 partway through the ramp yields the best sample so far, not an abort.

    Before this, the 429 raised out of the download probe, which also skipped the
    *upload* probe entirely — and uplink is the direction that bounds a pipeline
    stage.
    """
    state = {"n": 0}
    app = FastAPI()

    @app.get("/api/v1/probe/download")
    async def dl(size: int = 65536):
        state["n"] += 1
        if state["n"] > 3:
            return Response(status_code=429, content=b"", headers={"Retry-After": "30"})
        return Response(content=b"\0" * size, media_type="application/octet-stream")

    with TestClient(app) as c:
        mbps = P._measure_download_bw(c, "http://probe.test/api/v1/probe/download")

    assert mbps > 0, "a mid-ramp 429 threw away every sample already taken"
