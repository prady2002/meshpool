from __future__ import annotations

import asyncio

from meshpool.storage.engine import StorageManager
from tests.test_encryption import FakeEvents, FakeRegistry, FakeTransport, _manager


def test_bucket_object_versions_keep_previous_bytes():
    mgr, _transport = _manager()

    v1 = asyncio.run(mgr.put_object_version(
        b"first", actor="alice", bucket="docs", key="plan.txt", erasure=False))
    v2 = asyncio.run(mgr.put_object_version(
        b"second", actor="alice", bucket="docs", key="plan.txt", erasure=False))

    versions = mgr.list_object_versions(actor="alice", bucket="docs", key="plan.txt")
    assert [v["version_id"] for v in versions] == [v1["version_id"], v2["version_id"]]
    assert versions[0]["blob_id"] != versions[1]["blob_id"]
    assert asyncio.run(mgr.get_object_version(
        actor="alice", bucket="docs", key="plan.txt")) == b"second"
    assert asyncio.run(mgr.get_object_version(
        actor="alice", bucket="docs", key="plan.txt",
        version_id=v1["version_id"])) == b"first"


def test_bucket_versions_and_acls_reload_from_store():
    mgr, transport = _manager()
    asyncio.run(mgr.put_object_version(
        b"persisted", actor="alice", bucket="docs", key="a.txt", erasure=False))
    mgr.grant_bucket("docs", "alice", "bob", ["read"])

    reloaded = StorageManager(FakeRegistry(["A", "B", "C"]), mgr.store, None,
                              FakeEvents(), transport)
    assert reloaded.list_object_versions(actor="bob", bucket="docs", key="a.txt")[0]["owner"] == "alice"
    assert asyncio.run(reloaded.get_object_version(
        actor="bob", bucket="docs", key="a.txt")) == b"persisted"
