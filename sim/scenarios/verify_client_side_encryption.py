"""Acceptance gate: Mesh Store client-side encryption keeps node operators blind.

This uses the real harness (separate control-plane and agent OS processes) so the
disk scan inspects the same files a node operator could open on their own machine.

Non-vacuity:
1. Store the distinctive marker with encryption explicitly disabled and prove the
   scan finds it on an agent disk.
2. Store the same marker through the client encryption path and prove neither agent
   disks nor control-plane blob downloads contain it.
3. Download through the legitimate client path and decrypt exactly.
4. Flip a byte in a stored encrypted object and prove the read fails loudly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from meshpool.storage.crypto import ClientKeyring, decrypt_object, encrypt_object  # noqa: E402
from sim.harness import Harness  # noqa: E402

MARKER = b"MESHPOOL-PLAINTEXT-MARKER-P5-ENCRYPTION-9f3b7d"


def _post_blob(cp: str, data: bytes, replication: int = 2) -> dict:
    r = httpx.post(f"{cp}/api/v1/blobs?name=secret.bin&replication={replication}",
                   content=data, headers={"content-type": "application/octet-stream"},
                   timeout=60)
    r.raise_for_status()
    return r.json()


def _get_blob(cp: str, blob_id: str) -> httpx.Response:
    return httpx.get(f"{cp}/api/v1/blobs/{blob_id}", timeout=60)


def _blob_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for agent in root.glob("agent-*"):
        blob_root = agent / "blobs"
        if blob_root.exists():
            files.extend(p for p in blob_root.rglob("*") if p.is_file())
    return files


def _scan_files(root: Path, marker: bytes) -> tuple[bool, int, int]:
    files = _blob_files(root)
    hits = 0
    scanned = 0
    for path in files:
        raw = path.read_bytes()
        scanned += 1
        if marker in raw:
            hits += 1
    return hits > 0, hits, scanned


def _scan_control_plane(cp: str, marker: bytes) -> tuple[bool, int]:
    listing = httpx.get(f"{cp}/api/v1/blobs", timeout=20).json().get("blobs", [])
    checked = 0
    found = False
    for blob in listing:
        r = _get_blob(cp, blob["id"])
        if r.status_code == 200:
            checked += 1
            found = found or marker in r.content
    return found, checked


def _client_get_decrypted(cp: str, blob_id: str, keyring: ClientKeyring) -> bytes:
    r = _get_blob(cp, blob_id)
    if r.status_code != 200:
        raise RuntimeError(f"download failed loudly: HTTP {r.status_code} {r.text[:120]}")
    return decrypt_object(r.content, keyring=keyring)


def _flip_stored_byte(root: Path, blob_id: str) -> Path:
    matches = [p for p in _blob_files(root) if p.name == blob_id]
    if not matches:
        raise AssertionError(f"could not find stored blob file for {blob_id}")
    path = matches[0]
    raw = bytearray(path.read_bytes())
    if not raw:
        raise AssertionError(f"stored blob {path} is empty")
    raw[-1] ^= 0x01
    path.write_bytes(bytes(raw))
    return path


def main() -> int:
    ok = True
    base = Path(__file__).resolve().parent / "_data_client_side_encryption"
    plaintext = MARKER + b"|" + bytes(range(256)) * 32 + b"|" + MARKER
    print("== MeshPool Storage: client-side encryption gate ==")

    plain_root = base / "plain"
    h_plain = Harness(n_agents=3, cp_port=8340, data_dir=str(plain_root / "cluster"))
    try:
        print("\n== non-vacuity: disabled encryption leaves marker on node disk ==")
        h_plain.start()
        cp = h_plain.cp_url()
        meta = _post_blob(cp, plaintext, replication=2)
        found, hits, scanned = _scan_files(plain_root / "cluster", MARKER)
        assert scanned > 0, "disk scan inspected zero blob files"
        assert found, "scan did not find plaintext marker with encryption disabled"
        cp_found, cp_checked = _scan_control_plane(cp, MARKER)
        assert cp_found and cp_checked > 0, "control-plane plaintext control did not find marker"
        print(f"  PASS: plaintext control id={meta['id'][:12]} scanned {scanned} files; "
              f"marker hits on disk={hits}, control-plane downloads={cp_checked}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  FAIL: {type(exc).__name__}: {exc}")
    finally:
        h_plain.stop()

    enc_root = base / "encrypted"
    h_enc = Harness(n_agents=3, cp_port=8340, data_dir=str(enc_root / "cluster"))
    try:
        print("\n== encrypted write: marker absent from disks and CP-readable blobs ==")
        os.environ["MESHPOOL_CLIENT_KEY_PATH"] = str(enc_root / "client-root-key.json")
        keyring = ClientKeyring()
        package, version = encrypt_object(plaintext, bucket="private", key="secret.bin",
                                          keyring=keyring)
        assert MARKER not in package, "marker appears before upload; encryption is broken"
        h_enc.start()
        cp = h_enc.cp_url()
        meta = _post_blob(cp, package, replication=2)
        found, hits, scanned = _scan_files(enc_root / "cluster", MARKER)
        assert scanned > 0, "disk scan inspected zero encrypted blob files"
        assert not found, f"marker leaked to node disk ({hits} hits)"
        cp_found, cp_checked = _scan_control_plane(cp, MARKER)
        assert cp_checked > 0, "control-plane scan inspected zero blobs"
        assert not cp_found, "marker leaked through a control-plane blob download"
        recovered = _client_get_decrypted(cp, meta["id"], keyring)
        assert recovered == plaintext, "legitimate client decrypt did not round-trip exactly"
        print(f"  PASS: encrypted id={meta['id'][:12]} version={version}; "
              f"scanned {scanned} disk files and {cp_checked} CP downloads; no marker")
        print("  PASS: legitimate client decrypt returned original bytes exactly")

        print("\n== tamper: flipped stored ciphertext fails loudly ==")
        tamper_pkg, _ = encrypt_object(plaintext + b"-tamper", bucket="private",
                                       key="tamper.bin", keyring=keyring)
        tamper_meta = _post_blob(cp, tamper_pkg, replication=1)
        changed = _flip_stored_byte(enc_root / "cluster", tamper_meta["id"])
        try:
            _client_get_decrypted(cp, tamper_meta["id"], keyring)
            raise AssertionError("tampered ciphertext was returned as plaintext")
        except RuntimeError as exc:
            print(f"  PASS: flipped {changed.name[:12]} on disk; read failed loudly ({exc})")

        print("\nALL CLIENT-SIDE ENCRYPTION CHECKS PASSED")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        h_enc.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

