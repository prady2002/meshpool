from __future__ import annotations

import asyncio
import base64
import hashlib

import pytest

from meshpool.common import models as M
from meshpool.common.transport import TransportError
from meshpool.storage.crypto import ClientKeyring, EncryptionError, decrypt_object, encrypt_object
from meshpool.storage.engine import StorageManager


MARKER = b"MESHPOOL-CLIENT-SIDE-ENCRYPTION-MARKER"


def _node(nid: str):
    cap = M.Capability(cpu_cores=4, ram_mb=4096, disk_mb=100_000)
    return M.Node(id=nid, name=nid, url=f"http://127.0.0.1:0/{nid}",
                  capability=cap, status=M.NodeStatus.ONLINE)


class FakeRegistry:
    def __init__(self, ids):
        self.nodes = {n: _node(n) for n in ids}

    def online(self):
        return list(self.nodes.values())

    def schedulable_nodes(self):
        return self.online()


class FakeStore:
    def __init__(self):
        self.rows = {}

    def put_blob(self, meta):
        self.rows[meta.id] = meta.model_copy(deep=True)

    def get_blob(self, blob_id):
        meta = self.rows.get(blob_id)
        return meta.model_copy(deep=True) if meta else None

    def list_blobs(self):
        return [m.model_copy(deep=True) for m in self.rows.values()]

    def delete_blob(self, blob_id):
        self.rows.pop(blob_id, None)


class FakeTransport:
    def __init__(self, ids):
        self.disks = {n: {} for n in ids}

    async def call(self, node_id, method, payload, timeout=None):
        disk = self.disks[node_id]
        if method == "blob_put":
            raw = base64.b64decode(payload["data"])
            assert hashlib.sha256(raw).hexdigest() == payload["blob_id"]
            disk[payload["blob_id"]] = raw
            return {"found": True}
        if method == "blob_get":
            raw = disk.get(payload["blob_id"])
            if raw is None:
                return {"found": False, "data": ""}
            return {"found": True, "data": base64.b64encode(raw).decode("ascii")}
        if method == "blob_delete":
            disk.pop(payload["blob_id"], None)
            return {"deleted": True}
        raise TransportError(method)


class FakeEvents:
    def publish(self, _kind, _data):
        pass


def _manager():
    ids = ["A", "B", "C"]
    store = FakeStore()
    transport = FakeTransport(ids)
    return StorageManager(FakeRegistry(ids), store, None, FakeEvents(), transport), transport


def _disk_contains(transport: FakeTransport, marker: bytes) -> bool:
    seen = 0
    for disk in transport.disks.values():
        for raw in disk.values():
            seen += 1
            if marker in raw:
                return True
    assert seen > 0, "non-vacuity guard: scan did not inspect any stored blobs"
    return False


def test_client_side_encryption_hides_marker_on_nodes(tmp_path):
    plaintext = MARKER + b"|" + bytes(range(256)) * 8

    plain_mgr, plain_transport = _manager()
    asyncio.run(plain_mgr.put(plaintext, replication=2, erasure=False))
    assert _disk_contains(plain_transport, MARKER), (
        "non-vacuity guard failed: plaintext scan must find marker when encryption is off")

    keyring = ClientKeyring(tmp_path / "client-root-key.json")
    package, _version = encrypt_object(plaintext, bucket="photos", key="cat.bin",
                                       keyring=keyring)
    assert MARKER not in package
    enc_mgr, enc_transport = _manager()
    meta = asyncio.run(enc_mgr.put(package, replication=2, erasure=False))

    assert not _disk_contains(enc_transport, MARKER)
    fetched = asyncio.run(enc_mgr.get(meta.id))
    assert fetched is not None and MARKER not in fetched
    assert decrypt_object(fetched, keyring=keyring) == plaintext


def test_aead_rejects_tampered_ciphertext(tmp_path):
    keyring = ClientKeyring(tmp_path / "client-root-key.json")
    package, _version = encrypt_object(b"authentic " + MARKER, bucket="b", key="k",
                                       keyring=keyring)
    tampered = bytearray(package)
    tampered[-3] ^= 0x40
    with pytest.raises(EncryptionError):
        decrypt_object(bytes(tampered), keyring=keyring)


def test_tampered_stored_copy_fails_loudly(tmp_path):
    keyring = ClientKeyring(tmp_path / "client-root-key.json")
    package, _version = encrypt_object(b"node tamper " + MARKER, bucket="b", key="k",
                                       keyring=keyring)
    mgr, transport = _manager()
    meta = asyncio.run(mgr.put(package, replication=1, erasure=False))
    nid = meta.replicas[0]
    corrupt = bytearray(transport.disks[nid][meta.id])
    corrupt[-1] ^= 0x01
    transport.disks[nid][meta.id] = bytes(corrupt)
    assert asyncio.run(mgr.get(meta.id)) is None

