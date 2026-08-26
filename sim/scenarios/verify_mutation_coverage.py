"""Meta-gate: prove our strongest gates would actually FAIL if the code broke.

Every other gate in this directory answers "does the system work?". This one
answers the question that repeatedly caught real defects during development:

    "What would still pass if I deleted this code?"

A test that cannot fail is worse than no test, because it buys false confidence.
Seven such tests were found in this repository by hand -- a differential test
whose two arms shared the buggy function, a billing test where the second call
never happened, a security check that stayed green with the entire blocklist
replaced by ``return False``. Each looked thorough. None could fail.

So we automate the check. For each entry below we take the real source, break
one specific semantic property, run the gate that claims to protect it, and
require that gate to FAIL. A mutation that survives is a blind test, and a blind
test is a defect -- in the test, not in the fix.

Safety: mutations are applied inside a throwaway ``git worktree`` checked out at
HEAD, never in the working tree. Concurrent edits cannot corrupt it and a crash
cannot leave a mutated file behind. The working tree is verified untouched at
the end regardless.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Each mutation must break a *semantic* property, not merely a syntactic one.
# Deleting a line or renaming a symbol proves nothing: a crash is caught by any
# test that runs the code at all. These change behaviour while keeping the code
# plausible -- the shape a real regression actually takes.
MUTATIONS = [
    {
        "id": "spd-residual-sampling",
        "path": "meshpool/inference/spd.py",
        "old": b"emitted = sample_from_dist(_residual(p_i, q_i), rng)",
        "new": b"emitted = sample_from_dist(p_i, rng)",
        "cmd": [sys.executable, "sim/scenarios/verify_spd_equivalence.py"],
        "why": (
            "On rejection, speculative decoding must sample from the residual "
            "norm(max(0, p - q)), not from the target p. Sampling p directly is the "
            "single most natural-looking way to get SPD wrong: it still produces "
            "fluent text, still terminates, still passes every round-trip and "
            "latency check. It just silently stops being equivalent to ordinary "
            "decoding -- which is the entire correctness claim of the flagship."
        ),
    },
    {
        "id": "gguf-rope-reshape-order",
        "path": "meshpool/inference/gguf_loader.py",
        "old": b"w.reshape(n_head, rows // n_head // 2, 2, *w.shape[1:])",
        "new": b"w.reshape(n_head, 2, rows // n_head // 2, *w.shape[1:])",
        "cmd": [sys.executable, "-m", "pytest", "tests/test_gguf_loader.py",
                "-q", "-m", "slow or not slow"],
        "why": (
            "Un-permuting RoPE with the reshape halves swapped is still a perfect "
            "involution, so it round-trips flawlessly against our own reference and "
            "every weight survives a load. Only the *meaning* is wrong, and the "
            "damage is fluent garbage rather than a crash. This mutation exists to "
            "keep an independent semantic oracle in place: a round trip proves "
            "invertibility, never correctness."
        ),
    },
    {
        "id": "crypto-plaintext-passthrough",
        "path": "meshpool/storage/crypto.py",
        "old": b"ciphertext = AESGCM(data_key).encrypt(data_nonce, plaintext, aad)",
        "new": b"ciphertext = plaintext",
        "cmd": [sys.executable, "sim/scenarios/verify_client_side_encryption.py"],
        "why": (
            "The bluntest possible failure of client-side encryption: object bytes "
            "reach node operators in the clear. If this survives, the gate is not "
            "actually inspecting stored bytes and 'operators cannot read your data' "
            "is an unverified claim."
        ),
    },
    {
        "id": "crypto-static-data-key",
        "path": "meshpool/storage/crypto.py",
        "old": b"data_key = secrets.token_bytes(KEY_LEN)",
        "new": b'data_key = b"\\x00" * KEY_LEN',
        "cmd": [sys.executable, "sim/scenarios/verify_client_side_encryption.py"],
        "why": (
            "The subtle twin of the above, and the more dangerous one. Every object "
            "still encrypts, still decrypts, still round-trips, and still contains "
            "no readable plaintext -- so secrecy-only checks stay green. But the key "
            "is a constant sitting in open-source code, so anyone can decrypt every "
            "object ever stored. Catching this requires asserting key *freshness*, "
            "not just the absence of plaintext."
        ),
    },
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _apply(worktree: Path, mut: dict) -> bytes:
    target = worktree / mut["path"]
    original = target.read_bytes()
    count = original.count(mut["old"])
    if count != 1:
        raise AssertionError(
            f"[{mut['id']}] anchor matched {count} times in {mut['path']}, expected "
            f"exactly 1. The code moved; update this mutation so it keeps testing "
            f"the property it was written for.\n  anchor: {mut['old']!r}")
    target.write_bytes(original.replace(mut["old"], mut["new"]))
    return original


def main() -> int:
    print("== mutation coverage: do our strongest gates actually detect breakage? ==\n")

    pre = {m["path"]: _sha(ROOT / m["path"]) for m in MUTATIONS}
    tmp = Path(tempfile.mkdtemp(prefix="meshpool-mut-"))
    worktree = tmp / "wt"
    survivors: list[dict] = []

    # A detached worktree at HEAD keeps every mutation off the working tree, so
    # the eleven other things that might be editing this repo cannot be harmed
    # and cannot corrupt the experiment.
    subprocess.run(["git", "worktree", "add", "--detach", "--quiet",
                    str(worktree), "HEAD"],
                   cwd=str(ROOT), check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(worktree)
        env.pop("PYTHONHOME", None)

        for i, mut in enumerate(MUTATIONS, 1):
            print(f"-- [{i}/{len(MUTATIONS)}] {mut['id']}")
            print(f"   {mut['path']}: {mut['old'].decode()}")
            print(f"                -> {mut['new'].decode()}")
            original = _apply(worktree, mut)
            try:
                proc = subprocess.run(mut["cmd"], cwd=str(worktree), env=env,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, timeout=1800)
            finally:
                (worktree / mut["path"]).write_bytes(original)

            if proc.returncode == 0:
                survivors.append(mut)
                tail = proc.stdout.decode("utf-8", "replace").strip().splitlines()[-6:]
                print("   [SURVIVED] the guarding gate passed on broken code")
                for line in tail:
                    print(f"      | {line[:120]}")
            else:
                print(f"   [caught] guarding gate failed as required "
                      f"(exit {proc.returncode})")
            print()
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree)],
                       cwd=str(ROOT), stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        shutil.rmtree(tmp, ignore_errors=True)

    # Belt and braces: prove we did not touch the real files, whatever happened.
    for path, before in pre.items():
        after = _sha(ROOT / path)
        assert after == before, (
            f"{path} changed during mutation testing "
            f"({before[:16]} -> {after[:16]}) -- restore it from git immediately")
    print(f"working tree untouched: {len(pre)} files sha256-identical")

    if survivors:
        print()
        for mut in survivors:
            print(f"BLIND TEST: {mut['id']}")
            print(f"  {mut['why']}")
            print(f"  Guarded by: {' '.join(mut['cmd'][1:])}")
            print()
        raise AssertionError(
            f"{len(survivors)} of {len(MUTATIONS)} mutations survived. The gates "
            "above cannot fail, so they are not protecting what they claim to. Fix "
            "the tests -- a surviving mutation never means the code was fine.")

    print(f"\nALL {len(MUTATIONS)} MUTATIONS CAUGHT")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"\nFAILED: {exc}")
        sys.exit(1)
