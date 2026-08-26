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
        "id": "batching-degrades-to-serial",
        "path": "meshpool/inference/batching.py",
        "old": b"while self._queue and len(self._active) < self.max_batch:",
        "new": b"while self._queue and len(self._active) < 1:",
        "cmd": [sys.executable, "sim/scenarios/verify_continuous_batching.py"],
        "why": (
            "Serving one tenant per traversal is the failure mode continuous batching "
            "exists to prevent, and it is invisible to every correctness check: every "
            "answer stays token-identical, every request still succeeds, only the "
            "wide-area round trips multiply by the number of tenants. The gate has to "
            "notice that nobody shared a hop, which is why it asserts on mean_batch "
            "and on hop counts measured at the receiving nodes rather than on output."
        ),
    },
    {
        "id": "truncation-reported-as-completion",
        "path": "meshpool/controlplane/openai_api.py",
        "old": b'    reason = out.get("finish_reason")',
        "new": b"    reason = None",
        "cmd": [sys.executable, "sim/scenarios/verify_continuous_batching.py"],
        "why": (
            "Falling back to the token-count rule looks harmless and is not: when the "
            "context window cuts a reply short the count lands BELOW max_tokens, so "
            "the rule reports finish_reason='stop' -- telling an unmodified OpenAI "
            "client that a truncated reply is a finished one. Agent loops then stop "
            "early and drop the rest of the answer. Nothing about the text itself "
            "looks wrong, so only an explicit assertion on the reason can catch it."
        ),
    },
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
        "id": "gguf-rope-never-applied",
        "path": "meshpool/inference/gguf_loader.py",
        "old": b"        permute = _rope_is_permuted(info.architecture)",
        "new": b"        permute = False",
        "cmd": [sys.executable, "-m", "pytest", "tests/test_gguf_loader.py",
                "-q", "-m", "slow or not slow"],
        "why": (
            "Attacks the wiring rather than the arithmetic. Every rope test can "
            "pass on unpermute_rope and the architecture table directly while "
            "nothing checks that load_gguf actually *calls* them, which is the "
            "shape of bug that survives a green suite. Under this mutation the "
            "loader is a correct un-permuter that is simply never invoked."
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
    {
        "id": "spend-cap-boundary",
        "path": "meshpool/controlplane/spend.py",
        "old": b"return self.spent(owner) >= cap",
        "new": b"return self.spent(owner) > cap",
        "cmd": [sys.executable, "sim/scenarios/verify_spend_cap_hard_stop.py"],
        "why": (
            "A cap that only stops work *above* the limit lets an account sit exactly "
            "at its cap and keep spending forever. Straddling a boundary is not "
            "testing it: a gate that spends far past the cap and a gate that spends "
            "well under it both pass this mutation. Only landing exactly ON the cap "
            "tells the two comparisons apart."
        ),
    },
    {
        "id": "authz-scope-bypass",
        "path": "meshpool/common/auth.py",
        "old": b"if not principal.has_scope(scope):",
        "new": b"if False and not principal.has_scope(scope):",
        "cmd": [sys.executable, "sim/scenarios/verify_authz_matrix.py"],
        "why": (
            "Scope enforcement disabled entirely: every credential may call every "
            "endpoint. Expiry still rejects, so any test that only checks 'a bad "
            "token is refused' stays green. Catching this needs the other half of "
            "the matrix -- a *valid* credential that lacks the required scope must "
            "still be refused."
        ),
    },
    {
        "id": "ledger-double-fee",
        "path": "meshpool/common/billing.py",
        "old": b"total = mesh_seconds + fee",
        "new": b"total = mesh_seconds + fee * 2",
        "cmd": [sys.executable, "sim/scenarios/verify_ledger_conservation.py"],
        "why": (
            "Consumers are charged twice the platform fee while donors are still paid "
            "the same gross, so money appears from nowhere and the books stop "
            "balancing. Receipts still render, totals still rise monotonically, and "
            "every charge is still positive -- the defect is only visible to a check "
            "that asserts the identity total_charged == gross_earnings + platform_fee."
        ),
    },
    {
        "id": "wasm-fuel-budget-inflated",
        "path": "meshpool/functions/wasm.py",
        "old": b"store.set_fuel(limits.fuel)",
        "new": b"store.set_fuel(limits.fuel * 1000)",
        "cmd": [sys.executable, "sim/scenarios/verify_user_wasm.py"],
        "why": (
            "Guest code gets a thousand times the CPU it was sold. Every honest "
            "function still returns the same answer, so functional tests are blind to "
            "it, and a runaway loop is eventually still stopped -- by the wall-clock "
            "watchdog instead of the fuel meter. Detecting it requires asserting "
            "*which* limit fired, not merely that something did. Contributors were "
            "promised a bounded share of their machine."
        ),
    },
    {
        "id": "wasm-memory-cap-inflated",
        "path": "meshpool/functions/wasm.py",
        "old": b"store.set_limits(memory_size=limits.memory_bytes)",
        "new": b"store.set_limits(memory_size=limits.memory_bytes * 1000)",
        "cmd": [sys.executable, "sim/scenarios/verify_user_wasm.py"],
        "why": (
            "The 256 MB memory ceiling becomes 256 GB, so one hostile function can "
            "swap a donated laptop to death. memory.grow succeeding more often never "
            "makes a correct function fail, which is precisely why only a test that "
            "deliberately allocates past the cap can see this."
        ),
    },
    {
        "id": "wasm-filesystem-preopened",
        "path": "meshpool/functions/wasm.py",
        "old": b"# NOTE: no preopen_dir call. That omission *is* the filesystem sandbox.",
        "new": b'wasi.preopen_dir(".", "/")',
        "cmd": [sys.executable, "sim/scenarios/verify_user_wasm.py"],
        "why": (
            "Hands the guest a real preopened directory rooted at the agent's working "
            "directory, so path_open starts succeeding and untrusted code can read and "
            "write the host's files. The filesystem sandbox here is an *omission* -- "
            "nothing in the source says 'deny' -- so only a test that actively tries "
            "to open a file can notice when the omission stops being true."
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

    # Optional id filter keeps the edit/run loop cheap while writing a new entry.
    wanted = set(sys.argv[1:])
    selected = [m for m in MUTATIONS if not wanted or m["id"] in wanted]
    unknown = wanted - {m["id"] for m in MUTATIONS}
    assert not unknown, f"unknown mutation id(s): {sorted(unknown)}"

    pre = {m["path"]: _sha(ROOT / m["path"]) for m in selected}
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

        for i, mut in enumerate(selected, 1):
            print(f"-- [{i}/{len(selected)}] {mut['id']}")
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
                    # Gate output carries arbitrary bytes; never let printing it
                    # crash the harness and mask the finding it just made.
                    safe = line[:120].encode("ascii", "replace").decode("ascii")
                    print(f"      | {safe}")
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
            f"{len(survivors)} of {len(selected)} mutations survived. The gates "
            "above cannot fail, so they are not protecting what they claim to. Fix "
            "the tests -- a surviving mutation never means the code was fine.")

    print(f"\nALL {len(selected)} MUTATIONS CAUGHT")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"\nFAILED: {exc}")
        sys.exit(1)
