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

import json
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from meshpool.storage.crypto import (  # noqa: E402
    MAGIC, ClientKeyring, EncryptionError, decrypt_object, encrypt_object)
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


def _split_package(package: bytes) -> tuple[dict, bytes]:
    """Split a package into its JSON header and ciphertext. Framing only, no crypto."""
    header_len = int.from_bytes(package[len(MAGIC):len(MAGIC) + 4], "big")
    start = len(MAGIC) + 4
    end = start + header_len
    return json.loads(package[start:end].decode("utf-8")), package[end:]


def _join_package(header: dict, ciphertext: bytes) -> bytes:
    blob = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return MAGIC + len(blob).to_bytes(4, "big") + blob + ciphertext


def _check_per_object_key_isolation(base: Path) -> None:
    """One object's key must never open another object's bytes.

    Absence of plaintext is necessary but nowhere near sufficient: a build that
    hardcodes ``data_key = b"\\x00" * 32`` still encrypts, still decrypts, still
    leaves no readable marker anywhere on disk -- and yet every object in the
    system is readable by anyone, because the key is a constant in open-source
    code. Every secrecy-only check stays green through that. This one does not.

    Rather than re-deriving keys here -- test code that mirrors production crypto
    tends to mirror its bugs too -- we exploit the AAD binding. Two objects
    written to the same bucket/key/version share an identical AAD, so one
    package's wrapped-key fields can be transplanted into the other's header and
    will unwrap cleanly. What comes out is the *first* object's data key. It can
    only decrypt the *second* object's ciphertext if the two objects share a key,
    which is exactly the defect. Fresh keys make the transplant fail closed.
    """
    os.environ["MESHPOOL_CLIENT_KEY_PATH"] = str(base / "keyiso-root-key.json")
    keyring = ClientKeyring()
    shared = {"bucket": "private", "key": "same-object.bin",
              "version_id": "00000000000000000000000000000001", "keyring": keyring}
    first, _ = encrypt_object(b"FIRST-OBJECT-PLAINTEXT" * 8, **shared)
    second_plain = b"SECOND-OBJECT-PLAINTEXT" * 8
    second, _ = encrypt_object(second_plain, **shared)

    # Positive control: both packages must decrypt normally, or a transplant that
    # "fails" would prove nothing except that we corrupted the package framing.
    assert decrypt_object(second, keyring=keyring) == second_plain, \
        "control failed: an untouched package did not decrypt"
    h_first, _ = _split_package(first)
    h_second, ct_second = _split_package(second)
    assert _join_package(h_second, ct_second) == second, \
        "control failed: split/join is not byte-exact, so the transplant is untrustworthy"

    spliced = dict(h_second)
    spliced["data_key_nonce"] = h_first["data_key_nonce"]
    spliced["wrapped_data_key"] = h_first["wrapped_data_key"]
    assert spliced != h_second, "control failed: the two objects already share wrapped keys"
    try:
        leaked = decrypt_object(_join_package(spliced, ct_second), keyring=keyring)
    except EncryptionError:
        print("  PASS: object A's key cannot open object B — data keys are per-object")
        return
    detail = "recovering the exact plaintext" if leaked == second_plain else "producing output"
    raise AssertionError(
        f"one object's data key decrypted a different object, {detail}. Data keys are "
        f"shared or constant, so encrypting at all buys nothing: whoever learns one "
        f"key reads every object ever stored.")


def main() -> int:
    ok = True
    base = Path(__file__).resolve().parent / "_data_client_side_encryption"
    plaintext = MARKER + b"|" + bytes(range(256)) * 32 + b"|" + MARKER
    print("== MeshPool Storage: client-side encryption gate ==")

    print("\n== key isolation: each object gets its own data key ==")
    base.mkdir(parents=True, exist_ok=True)
    try:
        _check_per_object_key_isolation(base)
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  FAIL: {type(exc).__name__}: {exc}")

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

