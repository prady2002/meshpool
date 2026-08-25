"""Unit tests for the S3-compatible API adapter (p5-s3).

Exercises the full S3 wire protocol against an in-process FastAPI app with a
fake StorageManager (no real cluster, no I/O): PutObject, GetObject with Range,
HeadObject, DeleteObject, ListObjectsV2 (prefix + delimiter + pagination),
CreateBucket / ListBuckets / DeleteBucket, multipart upload, 404 for missing
keys, spend-cap rejection, and XML error format.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from meshpool.common.auth import mint_anonymous_principal
from meshpool.storage import s3_api as _s3
from meshpool.storage.engine import StorageError


# ─── Fake infrastructure ──────────────────────────────────────────────────────

class FakeStore:
    def __init__(self):
        self._rows = {}

    def put_blob(self, meta):
        from meshpool.common.models import BlobMeta
        self._rows[meta.id] = meta.model_copy(deep=True)

    def get_blob(self, bid):
        m = self._rows.get(bid)
        return m.model_copy(deep=True) if m else None

    def list_blobs(self):
        return [m.model_copy(deep=True) for m in self._rows.values()]

    def delete_blob(self, bid):
        self._rows.pop(bid, None)


class FakeRegistry:
    def __init__(self, node_ids):
        from meshpool.common import models as M
        self.nodes = {}
        for nid in node_ids:
            cap = M.Capability(cpu_cores=4, ram_mb=4096, disk_mb=100_000)
            self.nodes[nid] = M.Node(id=nid, name=nid,
                                      url=f"http://127.0.0.1:0/{nid}",
                                      capability=cap, status=M.NodeStatus.ONLINE)
        self.online_ids = set(node_ids)

    def online(self):
        return [self.nodes[n] for n in self.nodes if n in self.online_ids]

    def schedulable_nodes(self):
        return self.online()


class FakeTransport:
    def __init__(self, node_ids):
        self.disks = {n: {} for n in node_ids}

    async def call(self, node_id, method, payload, timeout=None):
        if node_id not in self.disks:
            from meshpool.common.transport import TransportError
            raise TransportError(f"unknown node {node_id}")
        disk = self.disks[node_id]
        if method == "blob_put":
            raw = base64.b64decode(payload["data"])
            disk[payload["blob_id"]] = raw
            return {"blob_id": payload["blob_id"], "stored": True}
        if method == "blob_get":
            raw = disk.get(payload["blob_id"])
            if raw is None:
                return {"blob_id": payload["blob_id"], "found": False, "data": ""}
            return {"blob_id": payload["blob_id"], "found": True,
                    "data": base64.b64encode(raw).decode("ascii")}
        if method == "blob_delete":
            disk.pop(payload["blob_id"], None)
            return {"deleted": True}
        raise ValueError(f"unknown method {method}")


class FakeEvents:
    def publish(self, *a, **kw):
        pass


class FakeAccounts:
    def can_spend(self, owner: str) -> bool:
        return True

    def owner_for_session(self, tok: str) -> Optional[str]:
        return None


class FakeSpendCaps:
    def exceeded(self, owner: str) -> bool:
        return False

    def status(self, owner: str) -> dict:
        return {"enforced": False, "cap": None, "spent": 0.0}


def _make_fake_cp(node_ids=("n0", "n1", "n2")):
    from meshpool.storage.engine import StorageManager

    store = FakeStore()
    reg = FakeRegistry(node_ids)
    transport = FakeTransport(node_ids)
    events = FakeEvents()

    storage = StorageManager(reg, store, None, events, transport)

    cp = SimpleNamespace(
        storage=storage,
        accounts=FakeAccounts(),
        spend_caps=FakeSpendCaps(),
        s3_credentials={},
    )
    return cp


@pytest.fixture(autouse=True)
def _reset_s3_state():
    """Clear module-level S3 state between tests."""
    _s3._reset_for_tests()
    yield
    _s3._reset_for_tests()


@pytest.fixture()
def client():
    """Return a TestClient wired to an in-process app with a fake CP."""
    cp = _make_fake_cp()

    _s3.init(lambda: cp)

    app = FastAPI()
    app.include_router(_s3.router, prefix="/s3")

    # Inject anonymous principal for every request
    @app.middleware("http")
    async def _inject_principal(request, call_next):
        request.state.principal = mint_anonymous_principal()
        return await call_next(request)

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _xml(resp) -> ET.Element:
    return ET.fromstring(resp.content)


def _find(root, tag):
    """Find element by tag, ignoring XML namespaces."""
    for el in root.iter():
        local = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if local == tag:
            return el
    return None


def _findall(root, tag):
    return [el for el in root.iter()
            if (el.tag.split("}")[-1] if "}" in el.tag else el.tag) == tag]


# ─── Bucket CRUD ─────────────────────────────────────────────────────────────

def test_create_and_list_buckets(client):
    r = client.put("/s3/mybucket")
    assert r.status_code == 200, r.text

    r = client.get("/s3/")
    assert r.status_code == 200
    root = _xml(r)
    names = [el.text for el in _findall(root, "Name")]
    assert "mybucket" in names


def test_delete_empty_bucket(client):
    client.put("/s3/delbucket")
    r = client.delete("/s3/delbucket")
    assert r.status_code == 204, r.text
    # Confirm it's gone from listing
    r2 = client.get("/s3/")
    names = [el.text for el in _findall(_xml(r2), "Name")]
    assert "delbucket" not in names


def test_delete_non_empty_bucket_returns_409(client):
    client.put("/s3/fullbucket")
    client.put("/s3/fullbucket/obj1", content=b"data", headers={"content-type": "text/plain"})
    r = client.delete("/s3/fullbucket")
    assert r.status_code == 409
    root = _xml(r)
    code = _find(root, "Code")
    assert code is not None and code.text == "BucketNotEmpty"


def test_delete_nonexistent_bucket_returns_404(client):
    r = client.delete("/s3/nosuchbucket")
    assert r.status_code == 404
    assert _find(_xml(r), "Code").text == "NoSuchBucket"


# ─── PutObject / GetObject ────────────────────────────────────────────────────

def test_put_get_roundtrip(client):
    data = b"hello world from s3 api test " * 100
    client.put("/s3/bkt")
    r = client.put("/s3/bkt/mykey", content=data,
                   headers={"content-type": "application/octet-stream"})
    assert r.status_code == 200, r.text
    etag = r.headers.get("etag", "")
    assert etag  # ETag must be present

    # Verify ETag is MD5
    assert etag.strip('"') == hashlib.md5(data).hexdigest()

    r2 = client.get("/s3/bkt/mykey")
    assert r2.status_code == 200
    assert r2.content == data


def test_get_missing_key_returns_404(client):
    client.put("/s3/bkt")
    r = client.get("/s3/bkt/nosuchkey")
    assert r.status_code == 404
    root = _xml(r)
    assert _find(root, "Code").text == "NoSuchKey"


def test_get_nested_key_path(client):
    """Keys with slashes must work (the :path converter)."""
    client.put("/s3/bkt")
    data = b"nested key content"
    client.put("/s3/bkt/dir/subdir/file.txt", content=data)
    r = client.get("/s3/bkt/dir/subdir/file.txt")
    assert r.status_code == 200
    assert r.content == data


# ─── Range GET ───────────────────────────────────────────────────────────────

def test_range_get_returns_206(client):
    data = bytes(range(256)) * 4    # 1024 bytes
    client.put("/s3/bkt")
    client.put("/s3/bkt/range-obj", content=data)
    r = client.get("/s3/bkt/range-obj", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206, r.text
    assert r.content == data[10:20]
    assert "Content-Range" in r.headers
    assert r.headers["Content-Range"] == f"bytes 10-19/{len(data)}"


def test_range_get_suffix(client):
    data = b"0123456789abcdef"
    client.put("/s3/bkt")
    client.put("/s3/bkt/sfx", content=data)
    r = client.get("/s3/bkt/sfx", headers={"Range": "bytes=-4"})
    assert r.status_code == 206
    assert r.content == data[-4:]


def test_range_get_open_end(client):
    data = b"0123456789"
    client.put("/s3/bkt")
    client.put("/s3/bkt/oe", content=data)
    r = client.get("/s3/bkt/oe", headers={"Range": "bytes=5-"})
    assert r.status_code == 206
    assert r.content == data[5:]


# ─── HeadObject ──────────────────────────────────────────────────────────────

def test_head_object(client):
    data = b"head me" * 50
    client.put("/s3/bkt")
    client.put("/s3/bkt/headkey", content=data,
               headers={"content-type": "text/plain"})
    r = client.head("/s3/bkt/headkey")
    assert r.status_code == 200
    assert r.headers.get("content-length") == str(len(data))
    assert r.headers.get("etag", "").strip('"') == hashlib.md5(data).hexdigest()


def test_head_missing_key_returns_404(client):
    client.put("/s3/bkt")
    r = client.head("/s3/bkt/ghost")
    assert r.status_code == 404


# ─── DeleteObject ─────────────────────────────────────────────────────────────

def test_delete_object(client):
    client.put("/s3/bkt")
    client.put("/s3/bkt/delme", content=b"delete this")
    r = client.delete("/s3/bkt/delme")
    assert r.status_code == 204
    # Subsequent GET must 404
    r2 = client.get("/s3/bkt/delme")
    assert r2.status_code == 404


def test_delete_idempotent(client):
    """S3 delete is idempotent: a missing key returns 204, not 404."""
    client.put("/s3/bkt")
    r = client.delete("/s3/bkt/never-existed")
    assert r.status_code == 204


# ─── ListObjectsV2 ───────────────────────────────────────────────────────────

def test_list_objects_v2_basic(client):
    client.put("/s3/bkt")
    client.put("/s3/bkt/a", content=b"a")
    client.put("/s3/bkt/b", content=b"b")
    r = client.get("/s3/bkt?list-type=2")
    assert r.status_code == 200
    root = _xml(r)
    keys = [el.text for el in _findall(root, "Key")]
    assert "a" in keys and "b" in keys
    assert _find(root, "IsTruncated").text == "false"


def test_list_objects_v2_prefix_filter(client):
    client.put("/s3/bkt")
    for key in ("img/a.png", "img/b.png", "doc/readme.txt"):
        client.put(f"/s3/bkt/{key}", content=b"x")
    r = client.get("/s3/bkt?list-type=2&prefix=img/")
    assert r.status_code == 200
    keys = [el.text for el in _findall(_xml(r), "Key")]
    assert all(k.startswith("img/") for k in keys), keys
    assert "doc/readme.txt" not in keys


def test_list_objects_v2_delimiter_groups(client):
    client.put("/s3/bkt")
    for key in ("a/x", "a/y", "b/z", "top"):
        client.put(f"/s3/bkt/{key}", content=b"x")
    r = client.get("/s3/bkt?list-type=2&delimiter=/")
    assert r.status_code == 200
    root = _xml(r)
    keys = [el.text for el in _findall(root, "Key")]
    prefixes = [el.text for el in _findall(root, "CommonPrefixes/Prefix")]
    # "top" is a direct match; a/ and b/ go into CommonPrefixes
    assert "top" in keys
    # If CommonPrefixes is a direct child it may nest differently; check text
    prefix_texts = [el.text for el in root.iter()
                    if (el.tag.split("}")[-1] if "}" in el.tag else el.tag) == "Prefix"]
    assert any("a/" in (t or "") for t in prefix_texts)
    assert any("b/" in (t or "") for t in prefix_texts)


def test_list_objects_v2_max_keys_pagination(client):
    client.put("/s3/bkt")
    for i in range(5):
        client.put(f"/s3/bkt/obj{i:02d}", content=b"x")
    r = client.get("/s3/bkt?list-type=2&max-keys=2")
    assert r.status_code == 200
    root = _xml(r)
    assert _find(root, "IsTruncated").text == "true"
    keys = [el.text for el in _findall(root, "Key")]
    assert len(keys) == 2
    # NextContinuationToken must be present
    nct = _find(root, "NextContinuationToken")
    assert nct is not None and nct.text

    # Fetch next page
    r2 = client.get(f"/s3/bkt?list-type=2&max-keys=2&continuation-token={nct.text}")
    assert r2.status_code == 200
    root2 = _xml(r2)
    keys2 = [el.text for el in _findall(root2, "Key")]
    assert len(keys2) == 2
    assert not set(keys) & set(keys2), "Pages must not overlap"


def test_list_nonexistent_bucket_returns_404(client):
    r = client.get("/s3/nosuchbucket?list-type=2")
    assert r.status_code == 404
    assert _find(_xml(r), "Code").text == "NoSuchBucket"


# ─── Multipart upload ────────────────────────────────────────────────────────

def test_multipart_upload_roundtrip(client):
    client.put("/s3/bkt")
    key = "multipart-obj"

    # 1. Initiate
    r = client.post(f"/s3/bkt/{key}?uploads")
    assert r.status_code == 200, r.text
    root = _xml(r)
    upload_id = _find(root, "UploadId").text
    assert upload_id

    # 2. Upload 3 parts
    parts = [b"part-one-data-" * 400, b"part-two-data-" * 400, b"part-three-" * 200]
    etags = []
    for i, part_data in enumerate(parts, 1):
        rp = client.put(
            f"/s3/bkt/{key}?partNumber={i}&uploadId={upload_id}",
            content=part_data)
        assert rp.status_code == 200, rp.text
        etags.append(rp.headers["etag"])

    # 3. Complete
    xml_body = "<CompleteMultipartUpload>"
    for i, etag in enumerate(etags, 1):
        xml_body += f"<Part><PartNumber>{i}</PartNumber><ETag>{etag}</ETag></Part>"
    xml_body += "</CompleteMultipartUpload>"
    rc = client.post(f"/s3/bkt/{key}?uploadId={upload_id}",
                     content=xml_body.encode(),
                     headers={"content-type": "application/xml"})
    assert rc.status_code == 200, rc.text
    root = _xml(rc)
    final_etag = _find(root, "ETag").text
    assert "-" in final_etag, f"Multipart ETag must have '-N' suffix: {final_etag}"
    assert final_etag.endswith("-3\"") or final_etag.endswith('-3"')

    # 4. Download and verify
    rg = client.get(f"/s3/bkt/{key}")
    assert rg.status_code == 200
    assert rg.content == b"".join(parts)


def test_abort_multipart_upload(client):
    client.put("/s3/bkt")
    r = client.post("/s3/bkt/abortme?uploads")
    upload_id = _find(_xml(r), "UploadId").text

    r2 = client.delete(f"/s3/bkt/abortme?uploadId={upload_id}")
    assert r2.status_code == 204
    # Subsequent complete must fail
    r3 = client.post(f"/s3/bkt/abortme?uploadId={upload_id}",
                     content=b"<CompleteMultipartUpload><Part>"
                             b"<PartNumber>1</PartNumber><ETag>\"abc\"</ETag>"
                             b"</Part></CompleteMultipartUpload>")
    assert r3.status_code == 404


# ─── S3 XML error format ─────────────────────────────────────────────────────

def test_error_response_is_xml(client):
    """Every S3 error must be application/xml with <Error><Code>..."""
    client.put("/s3/bkt")
    r = client.get("/s3/bkt/nosuchkey")
    assert r.status_code == 404
    assert "xml" in r.headers.get("content-type", "")
    root = _xml(r)
    code_el = _find(root, "Code")
    assert code_el is not None, "Missing <Code> in error XML"
    assert code_el.text == "NoSuchKey"
    msg_el = _find(root, "Message")
    assert msg_el is not None and msg_el.text


# ─── Spend-cap enforcement ────────────────────────────────────────────────────

def test_put_blocked_when_spend_cap_exceeded(client):
    """A capped owner must receive 403 AccessDenied on PUT."""
    # Patch spend caps on the CP the client uses
    cp = _s3._get_cp()
    orig_exceeded = cp.spend_caps.exceeded

    try:
        cp.spend_caps.exceeded = lambda owner: True
        cp.spend_caps.status = lambda owner: {"enforced": True, "cap": 1.0, "spent": 1.0}

        client.put("/s3/capbkt")
        r = client.put("/s3/capbkt/obj", content=b"data")
        assert r.status_code == 403
        root = _xml(r)
        assert _find(root, "Code").text == "AccessDenied"
    finally:
        cp.spend_caps.exceeded = orig_exceeded


# ─── ETag correctness ────────────────────────────────────────────────────────

def test_etag_is_content_md5(client):
    data = b"etag content test"
    client.put("/s3/bkt")
    r = client.put("/s3/bkt/etagkey", content=data)
    assert r.status_code == 200
    etag = r.headers.get("etag", "").strip('"')
    assert etag == hashlib.md5(data).hexdigest()


def test_empty_object_put_get(client):
    """Zero-byte object must round-trip cleanly."""
    client.put("/s3/bkt")
    r = client.put("/s3/bkt/empty", content=b"")
    assert r.status_code == 200
    r2 = client.get("/s3/bkt/empty")
    assert r2.status_code == 200
    assert r2.content == b""
