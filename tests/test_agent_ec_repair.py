"""The delegated EC repair endpoint, exercised against the real agent app.

tests/test_storage_ec_repair_locality.py proves the *coordinator* stops relaying, but
it does so against a fake transport that reimplements reconstruction. That fake could
agree with a broken agent. These tests drive the actual FastAPI handler with real
Reed-Solomon shards served by real peer HTTP endpoints, so the wire contract between
the two halves is checked rather than assumed.
"""
from __future__ import annotations

import base64
import hashlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from meshpool.storage.erasure import RSCodec

K, M = 3, 2
DATA = b"delegated-erasure-repair-must-not-touch-the-coordinator\n" * 300


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture()
def agent(tmp_path, monkeypatch):
    """A real agent app with an isolated blob store."""
    monkeypatch.setenv("MESHPOOL_AGENT_DATA", str(tmp_path / "agent"))
    from meshpool.agent import daemon as d
    from meshpool.storage.blobstore import BlobStore

    d.get_state()
    d.state.blobs = BlobStore(str(tmp_path / "blobs"), 64 * 1024 * 1024)
    d.state.donation._reclaimed = False
    return d


@pytest.fixture()
def peer():
    """A peer node exposing /blob_get, like a real agent does."""
    disk: dict[str, bytes] = {}
    app = FastAPI()

    @app.post("/blob_get")
    async def blob_get(body: dict) -> dict:
        raw = disk.get(body["blob_id"])
        if raw is None:
            return {"blob_id": body["blob_id"], "found": False, "data": ""}
        return {"blob_id": body["blob_id"], "found": True,
                "data": base64.b64encode(raw).decode("ascii")}

    client = TestClient(app)
    client.disk = disk
    return client


def _serve(peer, monkeypatch):
    """Route the agent's outbound httpx calls at the in-process peer."""
    import httpx
    from meshpool.agent import daemon as d

    real_post = peer.post

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, **kw):
            return real_post("/blob_get", json=json)

    monkeypatch.setattr(d.httpx, "AsyncClient", _Client)
    return httpx


def _plan(shards, missing_idx, peer_url="http://peer"):
    ids = [_sha(s) for s in shards]
    return {
        "blob_id": _sha(DATA),
        "k": K, "m": M,
        "shards": [
            {"index": i, "id": ids[i],
             "sources": [] if i == missing_idx else [peer_url]}
            for i in range(K + M)
        ],
        "want": [{"index": missing_idx, "id": ids[missing_idx], "sources": []}],
    }


def test_agent_reconstructs_a_missing_shard_from_peers(agent, peer, monkeypatch):
    shards = RSCodec(K, M).encode(DATA)
    _serve(peer, monkeypatch)
    for i, s in enumerate(shards):
        if i != 4:
            peer.disk[_sha(s)] = s

    c = TestClient(agent.app)
    r = c.post("/blob_ec_repair", json=_plan(shards, 4))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stored"] == [_sha(shards[4])]
    assert body["shards_used"] >= K
    assert agent.state.blobs.get(_sha(shards[4])) == shards[4]


def test_agent_refuses_when_fewer_than_k_shards_are_reachable(agent, peer, monkeypatch):
    shards = RSCodec(K, M).encode(DATA)
    _serve(peer, monkeypatch)
    peer.disk[_sha(shards[0])] = shards[0]      # only 1 of the 3 needed

    c = TestClient(agent.app)
    r = c.post("/blob_ec_repair", json=_plan(shards, 4))
    assert r.status_code == 502, r.text
    assert "need 3" in r.text
    assert not agent.state.blobs.has(_sha(shards[4]))


def test_agent_discards_a_shard_that_does_not_match_its_id(agent, peer, monkeypatch):
    """A lying peer must not be able to poison the codeword.

    Every shard is content-addressed, so serving different bytes under a shard's id is
    the natural attack. The forged shard has to be discarded at fetch time: if it were
    trusted, reconstruction would silently produce a corrupt object, and since nothing
    downstream re-derives the original from the shards, the corruption would become the
    stored truth.

    Here three honest shards remain, exactly k, so the correct outcome is not failure
    but a *correct* reconstruction that ignored the forgery.
    """
    shards = RSCodec(K, M).encode(DATA)
    _serve(peer, monkeypatch)
    for i, s in enumerate(shards):
        if i != 4:
            peer.disk[_sha(s)] = s
    peer.disk[_sha(shards[0])] = b"X" * len(shards[0])   # same id, wrong bytes

    c = TestClient(agent.app)
    r = c.post("/blob_ec_repair", json=_plan(shards, 4))
    assert r.status_code == 200, r.text
    assert r.json()["shards_used"] == K, "the forged shard was counted as usable"
    assert agent.state.blobs.get(_sha(shards[4])) == shards[4], (
        "reconstruction used the forged bytes")


def test_agent_fails_when_forgery_drops_honest_shards_below_k(agent, peer, monkeypatch):
    """Non-vacuity for the check above: forgeries genuinely reduce usable shards.

    Without this, the previous test would still pass on an agent that never verified
    anything, simply because the honest shards happened to be enough.
    """
    shards = RSCodec(K, M).encode(DATA)
    _serve(peer, monkeypatch)
    for i, s in enumerate(shards):
        if i != 4:
            peer.disk[_sha(s)] = s
    for i in (0, 1):
        peer.disk[_sha(shards[i])] = b"X" * len(shards[i])

    c = TestClient(agent.app)
    r = c.post("/blob_ec_repair", json=_plan(shards, 4))
    assert r.status_code == 502, r.text
    assert not agent.state.blobs.has(_sha(shards[4]))


def test_agent_rejects_a_malformed_plan(agent):
    shards = RSCodec(K, M).encode(DATA)
    c = TestClient(agent.app)
    bad_len = _plan(shards, 4)
    bad_len["shards"] = bad_len["shards"][:-1]
    assert c.post("/blob_ec_repair", json=bad_len).status_code == 400

    bad_id = _plan(shards, 4)
    bad_id["shards"][0]["id"] = "not-a-digest"
    assert c.post("/blob_ec_repair", json=bad_id).status_code == 400

    bad_idx = _plan(shards, 4)
    bad_idx["want"][0]["index"] = 99
    assert c.post("/blob_ec_repair", json=bad_idx).status_code == 400


def test_agent_uses_a_shard_it_already_holds_without_fetching(agent, peer, monkeypatch):
    """Non-vacuity for the local-first branch: a shard on disk is not re-downloaded."""
    shards = RSCodec(K, M).encode(DATA)
    _serve(peer, monkeypatch)
    agent.state.blobs.put(_sha(shards[0]), shards[0])
    for i in (1, 2, 3):
        peer.disk[_sha(shards[i])] = shards[i]

    c = TestClient(agent.app)
    r = c.post("/blob_ec_repair", json=_plan(shards, 4))
    assert r.status_code == 200, r.text
    assert agent.state.blobs.get(_sha(shards[4])) == shards[4]


def test_agent_refuses_repair_while_draining(agent):
    shards = RSCodec(K, M).encode(DATA)
    agent.state.donation.reclaim()
    try:
        c = TestClient(agent.app)
        r = c.post("/blob_ec_repair", json=_plan(shards, 4))
        assert r.status_code == 503
    finally:
        agent.state.donation._reclaimed = False
