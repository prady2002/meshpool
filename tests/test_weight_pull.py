"""Unit tests for true peer-to-peer weight-blob placement (spec P3 / Objective #2).

Two layers are exercised in isolation:
  * The **control-plane placement planner** (``InferenceManager._pull_plan`` /
    ``_place_blobs``): decides, per weight-group blob, whether the target node can pull
    it directly from a peer (byte-free for the control plane) or must be relayed the
    bytes -- driven against in-memory fakes so the decision is deterministic.
  * The **agent puller** (``daemon._pull_verified`` / the ``/blob_pull`` endpoint):
    fetches a blob (or a chunked object) from peers and verifies every piece, and the
    reassembled whole, against its SHA-256 id before storing -- so a dishonest or
    misconfigured peer can never plant the wrong bytes.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import types

import pytest
from fastapi import HTTPException

from meshpool.agent import daemon
from meshpool.common.models import NodeStatus
from meshpool.common.transport import TransportError
from meshpool.inference.engine import InferenceManager
from meshpool.storage.blobstore import BlobStore


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# Fakes for the control-plane placement planner
# --------------------------------------------------------------------------- #
def _node(nid: str, url: str, online: bool = True):
    return types.SimpleNamespace(
        id=nid, url=url,
        status=NodeStatus.ONLINE if online else NodeStatus.OFFLINE)


class FakeReg:
    def __init__(self, nodes):
        self._nodes = nodes

    def get(self, nid):
        return self._nodes.get(nid)


class FakeStorage:
    """Just enough of ``StorageManager`` for placement: metadata lookups (for the pull
    plan) and byte fetches (for the control-plane relay fallback)."""

    def __init__(self, metas=None, blobs=None):
        self._metas = metas or {}
        self._blobs = blobs or {}

    def get_meta(self, blob_id):
        return self._metas.get(blob_id)

    async def get(self, blob_id):
        return self._blobs.get(blob_id)


class FakePlaceTransport:
    """Records the placement calls the engine makes so the chosen path (pull vs. push)
    can be asserted. ``peer_ok`` mimics ``peer_pull_ok`` (LAN vs. tunnel)."""

    def __init__(self, has=(), pull_fails=False, peer_ok=True):
        self.present = set(has)
        self.pull_fails = pull_fails
        self.peer_ok = peer_ok
        self.calls = []

    def peer_pull_ok(self, node_id):
        return self.peer_ok

    async def call(self, node_id, path, payload, timeout=None):
        self.calls.append((path, payload.get("blob_id")))
        if path == "blob_has":
            return {"has": payload["blob_id"] in self.present}
        if path == "blob_pull":
            if self.pull_fails:
                raise TransportError("simulated pull failure")
            return {"stored": True}
        if path == "blob_put":
            return {"stored": True}
        raise TransportError(f"unexpected call {path}")

    def paths(self):
        return [p for p, _ in self.calls]


def _mgr(transport, storage, registry):
    m = InferenceManager(registry, store=None, ledger=None, events=None,
                         transport=transport)
    m.storage = storage
    return m


# --------------------------------------------------------------------------- #
# _pull_plan / _holder_urls: byte-free plan built from storage metadata
# --------------------------------------------------------------------------- #
def test_pull_plan_leaf_lists_peer_urls_excluding_target():
    bid = _sha(b"group-blob")
    storage = FakeStorage(metas={bid: types.SimpleNamespace(replicas=["A", "B"], chunks=[])})
    reg = FakeReg({"A": _node("A", "http://a:9"), "B": _node("B", "http://b:9")})
    mgr = _mgr(FakePlaceTransport(), storage, reg)

    plan = mgr._pull_plan(bid, exclude_node="A")     # A is the target -> excluded

    assert plan == {"blob_id": bid, "sources": ["http://b:9"]}


def test_pull_plan_chunked_lists_per_chunk_sources():
    bid = _sha(b"big-group")
    c1, c2 = _sha(b"chunk-1"), _sha(b"chunk-2")
    storage = FakeStorage(metas={
        bid: types.SimpleNamespace(replicas=[], chunks=[c1, c2]),
        c1: types.SimpleNamespace(replicas=["A"], chunks=[]),
        c2: types.SimpleNamespace(replicas=["B"], chunks=[]),
    })
    reg = FakeReg({"A": _node("A", "http://a:9"), "B": _node("B", "http://b:9"),
                   "C": _node("C", "http://c:9")})
    mgr = _mgr(FakePlaceTransport(), storage, reg)

    plan = mgr._pull_plan(bid, exclude_node="C")

    assert plan == {"blob_id": bid, "chunks": [
        {"id": c1, "sources": ["http://a:9"]},
        {"id": c2, "sources": ["http://b:9"]},
    ]}


def test_pull_plan_none_when_only_holder_is_the_target():
    bid = _sha(b"lonely")
    storage = FakeStorage(metas={bid: types.SimpleNamespace(replicas=["A"], chunks=[])})
    reg = FakeReg({"A": _node("A", "http://a:9")})
    mgr = _mgr(FakePlaceTransport(), storage, reg)

    assert mgr._pull_plan(bid, exclude_node="A") is None


def test_pull_plan_none_when_unknown_blob():
    mgr = _mgr(FakePlaceTransport(), FakeStorage(), FakeReg({}))
    assert mgr._pull_plan(_sha(b"missing"), exclude_node="A") is None


def test_pull_plan_skips_offline_holders():
    bid = _sha(b"grp")
    storage = FakeStorage(metas={bid: types.SimpleNamespace(replicas=["A", "B"], chunks=[])})
    reg = FakeReg({"A": _node("A", "http://a:9", online=False),
                   "B": _node("B", "http://b:9")})
    mgr = _mgr(FakePlaceTransport(), storage, reg)

    plan = mgr._pull_plan(bid, exclude_node="Z")

    assert plan == {"blob_id": bid, "sources": ["http://b:9"]}   # offline A dropped


# --------------------------------------------------------------------------- #
# _place_blobs: prefer peer pull on LAN, relay via control plane otherwise
# --------------------------------------------------------------------------- #
def test_place_uses_peer_pull_on_lan():
    bid = _sha(b"weights")
    storage = FakeStorage(metas={bid: types.SimpleNamespace(replicas=["A"], chunks=[])})
    reg = FakeReg({"A": _node("A", "http://a:9")})
    tr = FakePlaceTransport(has=(), peer_ok=True)
    mgr = _mgr(tr, storage, reg)

    asyncio.run(mgr._place_blobs("B", [bid]))

    assert "blob_pull" in tr.paths()
    assert "blob_put" not in tr.paths()            # control plane never carried bytes
    assert mgr.weight_group_pulls == 1
    assert mgr.weight_group_pushes == 0


def test_place_skips_blob_the_node_already_has():
    bid = _sha(b"weights")
    tr = FakePlaceTransport(has=(bid,), peer_ok=True)
    mgr = _mgr(tr, FakeStorage(), FakeReg({}))

    asyncio.run(mgr._place_blobs("B", [bid]))

    assert tr.paths() == ["blob_has"]              # nothing else
    assert mgr.weight_group_pulls == 0
    assert mgr.weight_group_pushes == 0


def test_place_falls_back_to_push_when_peer_pull_fails():
    data = b"weights-bytes"
    bid = _sha(data)
    storage = FakeStorage(metas={bid: types.SimpleNamespace(replicas=["A"], chunks=[])},
                          blobs={bid: data})
    reg = FakeReg({"A": _node("A", "http://a:9")})
    tr = FakePlaceTransport(has=(), pull_fails=True, peer_ok=True)
    mgr = _mgr(tr, storage, reg)

    asyncio.run(mgr._place_blobs("B", [bid]))

    assert "blob_pull" in tr.paths()               # attempted...
    assert "blob_put" in tr.paths()                # ...then relayed
    assert mgr.weight_group_pulls == 0
    assert mgr.weight_group_pushes == 1


def test_place_pushes_when_peer_not_reachable():
    data = b"weights-bytes"
    bid = _sha(data)
    storage = FakeStorage(blobs={bid: data})
    tr = FakePlaceTransport(has=(), peer_ok=False)   # tunnel node: no peer pull
    mgr = _mgr(tr, storage, FakeReg({}))

    asyncio.run(mgr._place_blobs("B", [bid]))

    assert "blob_pull" not in tr.paths()
    assert "blob_put" in tr.paths()
    assert mgr.weight_group_pushes == 1


def test_place_pushes_when_no_pull_plan_available():
    data = b"weights-bytes"
    bid = _sha(data)
    storage = FakeStorage(blobs={bid: data})         # no metadata -> no plan
    tr = FakePlaceTransport(has=(), peer_ok=True)
    mgr = _mgr(tr, storage, FakeReg({}))

    asyncio.run(mgr._place_blobs("B", [bid]))

    assert mgr.weight_group_pulls == 0
    assert mgr.weight_group_pushes == 1


# --------------------------------------------------------------------------- #
# Agent puller: content-addressed integrity on every pulled piece
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


def _peer_client(peers):
    """Fake ``httpx.AsyncClient`` serving ``/blob_get`` from an in-memory
    ``{base_url: {blob_id: bytes}}`` map, so no real network is touched."""

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            base = url[:-len("/blob_get")] if url.endswith("/blob_get") else url
            disk = peers.get(base, {})
            raw = disk.get(json["blob_id"])
            if raw is None:
                return _Resp(200, {"found": False, "data": ""})
            return _Resp(200, {"found": True,
                               "data": base64.b64encode(raw).decode("ascii")})

    return _Client


def test_pull_verified_returns_matching_bytes():
    data = b"correct group bytes"
    bid = _sha(data)
    client = _peer_client({"http://a": {bid: data}})()
    got = asyncio.run(daemon._pull_verified(client, bid, ["http://a"]))
    assert got == data


def test_pull_verified_rejects_tampered_bytes():
    bid = _sha(b"honest bytes")
    # peer serves DIFFERENT bytes under the claimed id -> must be rejected
    client = _peer_client({"http://a": {bid: b"tampered bytes"}})()
    with pytest.raises(daemon._PullError):
        asyncio.run(daemon._pull_verified(client, bid, ["http://a"]))


def test_pull_verified_tries_next_source_on_miss():
    data = b"group bytes"
    bid = _sha(data)
    client = _peer_client({"http://a": {}, "http://b": {bid: data}})()
    got = asyncio.run(daemon._pull_verified(client, bid, ["http://a", "http://b"]))
    assert got == data


# --------------------------------------------------------------------------- #
# /blob_pull endpoint: reassemble + verify + store (or reject)
# --------------------------------------------------------------------------- #
def _fake_state(tmp_path, reclaimed=False):
    return types.SimpleNamespace(
        blobs=BlobStore(str(tmp_path / "blobs"), cap_bytes=50_000_000),
        donation=types.SimpleNamespace(reclaimed=reclaimed))


def _run_pull(monkeypatch, state, peers, req):
    monkeypatch.setattr(daemon, "state", state)
    monkeypatch.setattr(daemon.httpx, "AsyncClient", _peer_client(peers))
    return asyncio.run(daemon.blob_pull(req))


def test_blob_pull_leaf_stores_verified_bytes(tmp_path, monkeypatch):
    data = b"a whole weight group" * 100
    bid = _sha(data)
    state = _fake_state(tmp_path)
    out = _run_pull(monkeypatch, state, {"http://a": {bid: data}},
                    daemon.BlobPullRequest(blob_id=bid, sources=["http://a"]))
    assert out["stored"] is True and out["bytes"] == len(data)
    assert state.blobs.get(bid) == data


def test_blob_pull_chunked_reassembles_and_stores(tmp_path, monkeypatch):
    c1, c2 = b"A" * 2048, b"B" * 2048
    id1, id2 = _sha(c1), _sha(c2)
    whole = c1 + c2
    bid = _sha(whole)
    state = _fake_state(tmp_path)
    peers = {"http://a": {id1: c1}, "http://b": {id2: c2}}
    req = daemon.BlobPullRequest(blob_id=bid, chunks=[
        {"id": id1, "sources": ["http://a"]},
        {"id": id2, "sources": ["http://b"]}])
    out = _run_pull(monkeypatch, state, peers, req)
    assert out["stored"] is True and out["bytes"] == len(whole)
    assert state.blobs.get(bid) == whole


def test_blob_pull_rejects_tampered_chunk(tmp_path, monkeypatch):
    c1 = b"A" * 2048
    id1 = _sha(c1)
    bid = _sha(c1 + b"B" * 2048)
    state = _fake_state(tmp_path)
    peers = {"http://a": {id1: b"not the real chunk"}}    # wrong bytes under id1
    req = daemon.BlobPullRequest(blob_id=bid, chunks=[{"id": id1, "sources": ["http://a"]}])
    with pytest.raises(HTTPException) as ei:
        _run_pull(monkeypatch, state, peers, req)
    assert ei.value.status_code == 502
    assert not state.blobs.has(bid)


def test_blob_pull_rejects_wrong_reassembled_whole(tmp_path, monkeypatch):
    c1, c2 = b"A" * 2048, b"B" * 2048
    id1, id2 = _sha(c1), _sha(c2)
    bid = _sha(b"a different object entirely")          # not sha256(c1 + c2)
    state = _fake_state(tmp_path)
    peers = {"http://a": {id1: c1}, "http://b": {id2: c2}}
    req = daemon.BlobPullRequest(blob_id=bid, chunks=[
        {"id": id1, "sources": ["http://a"]},
        {"id": id2, "sources": ["http://b"]}])
    with pytest.raises(HTTPException) as ei:
        _run_pull(monkeypatch, state, peers, req)
    assert ei.value.status_code == 502
    assert not state.blobs.has(bid)


def test_blob_pull_rejects_bad_id(tmp_path, monkeypatch):
    state = _fake_state(tmp_path)
    with pytest.raises(HTTPException) as ei:
        _run_pull(monkeypatch, state, {},
                  daemon.BlobPullRequest(blob_id="not-a-sha", sources=["http://a"]))
    assert ei.value.status_code == 400


def test_blob_pull_refused_while_draining(tmp_path, monkeypatch):
    data = b"weights"
    bid = _sha(data)
    state = _fake_state(tmp_path, reclaimed=True)
    with pytest.raises(HTTPException) as ei:
        _run_pull(monkeypatch, state, {"http://a": {bid: data}},
                  daemon.BlobPullRequest(blob_id=bid, sources=["http://a"]))
    assert ei.value.status_code == 503


def test_blob_pull_is_idempotent_when_already_cached(tmp_path, monkeypatch):
    data = b"already here"
    bid = _sha(data)
    state = _fake_state(tmp_path)
    state.blobs.put(bid, data)                         # already present
    out = _run_pull(monkeypatch, state, {},            # no peers needed
                    daemon.BlobPullRequest(blob_id=bid, sources=["http://a"]))
    assert out["stored"] is True and out.get("cached") is True


# --------------------------------------------------------------------------- #
# The swarm must grow: a node that acquires a group becomes a source for the next
# --------------------------------------------------------------------------- #
class RecordingStorage(FakeStorage):
    """FakeStorage that also implements the note_replica seam StorageManager has."""

    def note_replica(self, blob_id, node_id):
        meta = self._metas.get(blob_id)
        if meta is None or node_id in meta.replicas:
            return
        meta.replicas = [*meta.replicas, node_id]


def test_a_node_that_pulled_a_group_becomes_a_source_for_the_next_node():
    """Otherwise every joiner pulls from the same original holders forever.

    This is the difference between a swarm and a fan-out from a fixed origin. With
    600 GB of weights and a pool that is growing, sourcing every new node from the
    same two machines re-creates the exact bottleneck peer-to-peer distribution
    exists to remove -- it just moves it off the coordinator and onto two unlucky
    peers. Nothing about it is visible in a functional test: every pull still
    succeeds, every model still loads. It only shows up as the pool getting slower
    the more it is used, which is the opposite of what a swarm is supposed to do.
    """
    bid = _sha(b"weights")
    storage = RecordingStorage(
        metas={bid: types.SimpleNamespace(replicas=["A"], chunks=[])})
    reg = FakeReg({"A": _node("A", "http://a:9"), "B": _node("B", "http://b:9"),
                   "C": _node("C", "http://c:9")})
    tr = FakePlaceTransport(has=(), peer_ok=True)
    mgr = _mgr(tr, storage, reg)

    asyncio.run(mgr._place_blobs("B", [bid]))       # B pulls from A

    plan = mgr._pull_plan(bid, exclude_node="C")
    assert plan is not None
    assert sorted(plan["sources"]) == ["http://a:9", "http://b:9"], (
        "after B acquired the group it is still not offered as a source, so the "
        f"next node can only pull from the original holder: {plan}")


def test_a_relayed_group_also_makes_the_node_a_source():
    """The bytes got there either way; where they came from does not change who
    can serve them next. A node seeded by the coordinator fallback that never
    becomes a peer source keeps the coordinator on the critical path forever."""
    bid = _sha(b"weights-relayed")
    storage = RecordingStorage(
        metas={bid: types.SimpleNamespace(replicas=["A"], chunks=[])},
        blobs={bid: b"weights-relayed"})
    reg = FakeReg({"A": _node("A", "http://a:9", online=False),
                   "B": _node("B", "http://b:9"), "C": _node("C", "http://c:9")})
    tr = FakePlaceTransport(has=(), peer_ok=True)
    mgr = _mgr(tr, storage, reg)

    asyncio.run(mgr._place_blobs("B", [bid]))       # no live peer -> relayed
    assert "blob_put" in tr.paths()

    plan = mgr._pull_plan(bid, exclude_node="C")
    assert plan == {"blob_id": bid, "sources": ["http://b:9"]}, (
        f"a relayed node never becomes a peer source: {plan}")


def test_an_already_resident_node_is_recorded_as_a_source_too():
    """blob_has short-circuits placement, and that early return is the easiest
    place to forget: the node demonstrably holds the bytes -- it just told us so."""
    bid = _sha(b"already-there")
    storage = RecordingStorage(
        metas={bid: types.SimpleNamespace(replicas=["A"], chunks=[])})
    reg = FakeReg({"A": _node("A", "http://a:9"), "B": _node("B", "http://b:9"),
                   "C": _node("C", "http://c:9")})
    tr = FakePlaceTransport(has=(bid,), peer_ok=True)
    mgr = _mgr(tr, storage, reg)

    asyncio.run(mgr._place_blobs("B", [bid]))
    assert "blob_pull" not in tr.paths() and "blob_put" not in tr.paths()

    plan = mgr._pull_plan(bid, exclude_node="C")
    assert plan is not None
    assert sorted(plan["sources"]) == ["http://a:9", "http://b:9"], plan


def test_placement_survives_storage_without_the_seam():
    """Older/simpler storage objects have no note_replica; placement must not care."""
    bid = _sha(b"no-seam")
    storage = FakeStorage(metas={bid: types.SimpleNamespace(replicas=["A"], chunks=[])})
    reg = FakeReg({"A": _node("A", "http://a:9"), "B": _node("B", "http://b:9")})
    tr = FakePlaceTransport(has=(), peer_ok=True)
    mgr = _mgr(tr, storage, reg)

    asyncio.run(mgr._place_blobs("B", [bid]))       # must not raise
    assert "blob_pull" in tr.paths()
