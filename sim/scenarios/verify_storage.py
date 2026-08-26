"""End-to-end scenario: pooled, fault-tolerant object storage across the mesh (§P3).

This proves the "pool disk that one machine can't provide" path end to end:
  * an object POSTed to the control plane is content-addressed (its id IS the
    SHA-256 of its bytes) and replicated across TWO distinct nodes' donated disk,
  * it is served back byte-for-byte through the control plane,
  * re-uploading identical bytes dedupes (same id, still one object, not re-stored),
  * KILLING a node that holds a copy does not lose the object -- it is still
    downloadable from a surviving replica, and the storage manager SELF-HEALS by
    re-replicating the missing copy onto a third node (durability > any one box),
  * a LARGE object (bigger than any one machine could hold) is split into many
    content-addressed chunks spread across the pool and reassembled on download,
  * deleting the object removes it everywhere.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402

TOKEN = b"MESHPOOL-STORAGE-7Q2Z"          # marker proving OUR bytes came back intact


def make_object(tag: bytes, n: int = 64) -> bytes:
    """A chunk of real binary content (all 256 byte values) wrapped in a marker, so a
    byte-mangling bug anywhere in the round trip is caught, not just a text diff."""
    body = bytes(range(256)) * n
    return TOKEN + b"|" + tag + b"|" + body


def node_name_map(cp: str) -> dict:
    nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=10).json().get("nodes", [])
    return {x["id"]: x["name"] for x in nodes}


def put_object(cp: str, data: bytes, name: str) -> dict:
    r = httpx.post(f"{cp}/api/v1/blobs?name={name}&replication=2",
                   content=data, headers={"content-type": "application/octet-stream"},
                   timeout=30)
    r.raise_for_status()
    return r.json()


def get_object(cp: str, blob_id: str):
    return httpx.get(f"{cp}/api/v1/blobs/{blob_id}", timeout=30)


def blob_replicas(cp: str, blob_id: str):
    for b in httpx.get(f"{cp}/api/v1/blobs", timeout=10).json().get("blobs", []):
        if b["id"] == blob_id:
            return b
    return None


def main() -> int:
    # Shrink the chunk threshold so a modest test object exercises the split-across-the-
    # pool path (the small objects in tests 1-8 stay comfortably under it, unchanged).
    # The control plane runs as a subprocess that inherits this environment.
    os.environ["MESHPOOL_CHUNK_SIZE"] = "32768"
    base = Path(__file__).resolve().parent / "_data_storage"
    h = Harness(n_agents=3, cp_port=8117, data_dir=str(base / "cluster"))
    ok = True
    try:
        print("== starting cluster (3 nodes donating disk) ==")
        h.start()
        cp = h.cp_url()

        print("\n== test 1: store an object -> replicated to 2 distinct nodes ==")
        data = make_object(b"alpha")
        expect_id = hashlib.sha256(data).hexdigest()
        meta = put_object(cp, data, "alpha.bin")
        assert meta["id"] == expect_id, f"id is not the content hash: {meta['id']} != {expect_id}"
        assert meta["size"] == len(data), f"size wrong: {meta['size']} != {len(data)}"
        assert len(meta["replicas"]) == 2 and len(set(meta["replicas"])) == 2, \
            f"expected 2 distinct replicas: {meta['replicas']}"
        assert meta["status"] == "stored", f"status not stored: {meta['status']}"
        names = node_name_map(cp)
        holders = sorted(names.get(n, n) for n in meta["replicas"])
        print(f"  PASS: id={meta['id'][:16]}… stored on {holders}")

        print("\n== test 2: download it back, byte-for-byte ==")
        r = get_object(cp, meta["id"])
        assert r.status_code == 200, f"download failed: HTTP {r.status_code}"
        assert r.content == data, "downloaded bytes differ from what was stored"
        assert hashlib.sha256(r.content).hexdigest() == expect_id, "integrity check failed"
        assert "alpha.bin" in r.headers.get("content-disposition", ""), \
            f"filename not surfaced: {r.headers.get('content-disposition')!r}"
        print(f"  PASS: {len(r.content)} bytes returned intact (sha256 matches)")

        print("\n== test 3: re-upload identical bytes -> dedup (same id, one object) ==")
        meta2 = put_object(cp, data, "alpha-again.bin")
        assert meta2["id"] == expect_id, "content-addressing broke: different id for same bytes"
        listing = httpx.get(f"{cp}/api/v1/blobs", timeout=10).json()["blobs"]
        assert sum(1 for b in listing if b["id"] == expect_id) == 1, "object was duplicated"
        assert len(meta2["replicas"]) == 2, f"replica count changed on re-put: {meta2['replicas']}"
        print("  PASS: identical content deduped to one object, still 2 replicas")

        print("\n== test 4: a different object gets a different id ==")
        other = make_object(b"beta")
        meta_o = put_object(cp, other, "beta.bin")
        assert meta_o["id"] != expect_id, "distinct content collided to same id"
        assert get_object(cp, meta_o["id"]).content == other, "second object not retrievable"
        print(f"  PASS: distinct object id={meta_o['id'][:16]}… stored + retrievable")

        print("\n== test 5: KILL a node holding the object -> still downloadable ==")
        victim_id = meta["replicas"][0]
        survivor_id = meta["replicas"][1]
        victim_name = names.get(victim_id, "")
        victim_idx = int(victim_name.split("-")[-1]) - 1
        print(f"  killing {victim_name} (holds a copy) ...")
        h.stop_agent(victim_idx)
        # the surviving replica still answers immediately (no data lost)
        r2 = get_object(cp, meta["id"])
        assert r2.status_code == 200 and r2.content == data, \
            f"object lost after a holder died: HTTP {r2.status_code}, {len(r2.content)} bytes"
        print("  PASS: object still served intact from the surviving replica")

        print("\n== test 6: storage manager self-heals back to 2 replicas ==")
        deadline = time.time() + 45
        healed = None
        while time.time() < deadline:
            b = blob_replicas(cp, meta["id"])
            if b and victim_id not in b["replicas"] and len(b["replicas"]) >= 2:
                healed = b
                break
            time.sleep(1)
        assert healed, "manager did not re-replicate the object after the node loss"
        assert survivor_id in healed["replicas"], "the surviving copy was dropped"
        new_home = next(n for n in healed["replicas"] if n != survivor_id)
        assert new_home != victim_id, "re-replicated onto the dead node"
        assert healed["status"] == "stored", f"not back to healthy: {healed['status']}"
        print(f"  PASS: re-replicated onto {names.get(new_home, new_home)}; "
              f"redundancy restored ({sorted(names.get(n, n) for n in healed['replicas'])})")
        # and it is STILL byte-perfect after the heal
        r3 = get_object(cp, meta["id"])
        assert r3.status_code == 200 and r3.content == data, "object corrupted after self-heal"
        print("  PASS: object still byte-perfect after self-heal")

        print("\n== test 7: delete removes it everywhere ==")
        d = httpx.delete(f"{cp}/api/v1/blobs/{meta['id']}", timeout=10)
        assert d.status_code == 200, f"delete failed: HTTP {d.status_code}"
        gone = get_object(cp, meta["id"])
        assert gone.status_code == 404, f"object still present after delete: HTTP {gone.status_code}"
        print("  PASS: object deleted; download now 404s")

        print("\n== test 8: the `meshpool storage` CLI round-trips against the live pool ==")
        repo_root = Path(__file__).resolve().parents[2]
        cli_env = {**os.environ, "MESHPOOL_CONTROL_PLANE_URL": cp, "COLUMNS": "200"}
        cli_data = make_object(b"cli")
        cli_id = hashlib.sha256(cli_data).hexdigest()
        src = base / "cli_upload.bin"
        src.write_bytes(cli_data)

        def _cli(*args):
            return subprocess.run(
                [sys.executable, "-m", "meshpool.cli.main", "storage", *args],
                cwd=str(repo_root), env=cli_env, capture_output=True, text=True, errors="replace", timeout=60)

        up = _cli("put", str(src), "--name", "cli.bin")
        assert up.returncode == 0, f"`storage put` failed: {up.stderr or up.stdout}"
        assert cli_id[:16] in up.stdout, f"put did not report the content id: {up.stdout!r}"
        ls = _cli("ls")
        assert ls.returncode == 0 and "cli.bin" in ls.stdout, \
            f"`storage ls` did not list the file: {ls.stdout!r}"
        out = base / "cli_download.bin"
        gt = _cli("get", cli_id, str(out))
        assert gt.returncode == 0, f"`storage get` failed: {gt.stderr or gt.stdout}"
        assert out.read_bytes() == cli_data, "CLI-downloaded bytes differ from what was uploaded"
        rm = _cli("rm", cli_id)
        assert rm.returncode == 0, f"`storage rm` failed: {rm.stderr or rm.stdout}"
        assert get_object(cp, cli_id).status_code == 404, "object still present after CLI rm"
        print("  PASS: CLI put -> ls -> get (byte-perfect) -> rm all worked end to end")

        print("\n== test 9: a LARGE object is split into chunks spread across the pool ==")
        chunk_size = 32768
        big = make_object(b"chunked", n=800)          # ~200 KB, far over the 32 KB chunk size
        big_id = hashlib.sha256(big).hexdigest()
        expect_chunks = (len(big) + chunk_size - 1) // chunk_size
        bmeta = put_object(cp, big, "big-model.bin")
        assert bmeta["id"] == big_id, "chunked object id is not the whole-content hash"
        assert bmeta["size"] == len(big), f"size wrong: {bmeta['size']} != {len(big)}"
        assert len(bmeta["chunks"]) == expect_chunks, \
            f"expected {expect_chunks} chunks, got {len(bmeta['chunks'])}"
        assert bmeta["replicas"] == [], "the manifest itself should hold no direct replicas"
        # the internal chunks must NOT surface as user-visible objects
        big_listing = httpx.get(f"{cp}/api/v1/blobs", timeout=10).json()["blobs"]
        listed_ids = {b["id"] for b in big_listing}
        assert big_id in listed_ids, "the manifest is missing from the listing"
        assert not (set(bmeta["chunks"]) & listed_ids), "internal chunks leaked into the listing"
        print(f"  PASS: {len(big)} bytes split into {len(bmeta['chunks'])} chunks, "
              f"chunks hidden from the listing")
        # reassembled byte-for-byte through the control plane
        rb = get_object(cp, big_id)
        assert rb.status_code == 200, f"chunked download failed: HTTP {rb.status_code}"
        assert rb.content == big, "reassembled bytes differ from the original"
        assert hashlib.sha256(rb.content).hexdigest() == big_id, "chunked integrity check failed"
        print(f"  PASS: reassembled {len(rb.content)} bytes intact (sha256 matches)")
        # the CLI surfaces it honestly as pooled chunks (not a misleading "0 copies")
        ls_big = _cli("ls")
        assert ls_big.returncode == 0, f"`storage ls` failed: {ls_big.stderr or ls_big.stdout}"
        assert "chunks" in ls_big.stdout, \
            f"`storage ls` did not surface the chunked object as chunks: {ls_big.stdout!r}"
        print("  PASS: CLI `storage ls` shows the large object as pooled chunks")
        # delete cascades to every chunk
        db = httpx.delete(f"{cp}/api/v1/blobs/{big_id}", timeout=10)
        assert db.status_code == 200, f"chunked delete failed: HTTP {db.status_code}"
        assert get_object(cp, big_id).status_code == 404, "chunked object still present after delete"
        print("  PASS: chunked object + all its chunks deleted; download now 404s")

        print("\nALL POOLED-STORAGE E2E CHECKS PASSED")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        h.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
