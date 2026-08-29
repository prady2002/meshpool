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

A crash *can* leave the worktree itself behind -- ``git worktree remove`` lives
in a ``finally`` that a ``taskkill`` does not run -- so startup sweeps any
abandoned ``meshpool-mut-*`` worktree before adding a new one. Without that,
every killed run permanently accumulates a registered worktree and a few
hundred megabytes of temp files.
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
sys.path.insert(0, str(ROOT))

from sim import mutation_guard  # noqa: E402


def worktree_is_abandoned(wt: str) -> bool:
    """Is this ``meshpool-mut-*`` worktree safe to delete?

    "Abandoned" has to mean abandoned. The sweep used to remove EVERY matching
    worktree, so a second run -- or, as it turned out, this gate's own guard
    tests running nested inside it -- deleted the temp directory of a run that
    was still using it, and the outer run died with ``FileNotFoundError`` on its
    own worktree. Every run now claims its directory with ``owner.pid`` before
    creating the worktree, and a live owner is left strictly alone.

    A directory with no readable owner file predates this scheme (or lost the
    file to a hard kill), so it is treated as abandoned -- that is the old
    behaviour and it is the safe default for a temp directory nobody claims.
    """
    try:
        pid = int((Path(wt).parent / "owner.pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return True
    return not mutation_guard._pid_alive(pid)


def _sweep_abandoned_worktrees() -> None:
    """Remove worktrees a previous killed run could not clean up.

    ``git worktree prune`` is not enough: it only forgets worktrees whose
    directory has already vanished, and a killed run leaves the directory very
    much present. So list them, match our own prefix, and remove them
    explicitly. Also drop any journal left by an in-place mutation gate that
    was killed, since this gate refuses to run against a dirty checkout.
    """
    for rel in mutation_guard.restore_stale():
        print(f"   [recovered] restored {rel} from a killed mutation gate")
    try:
        out = subprocess.run(["git", "worktree", "list", "--porcelain"],
                             cwd=str(ROOT), capture_output=True, text=True,
                             timeout=60)
    except (OSError, subprocess.SubprocessError):
        return
    for line in (out.stdout or "").splitlines():
        if not line.startswith("worktree "):
            continue
        wt = line.split(" ", 1)[1].strip()
        if "meshpool-mut-" not in wt:
            continue
        if not worktree_is_abandoned(wt):
            print(f"   [skip] worktree {wt} is owned by a live run")
            continue
        subprocess.run(["git", "worktree", "remove", "--force", wt],
                       cwd=str(ROOT), stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        shutil.rmtree(Path(wt).parent, ignore_errors=True)
        print(f"   [recovered] removed abandoned worktree {wt}")
    subprocess.run(["git", "worktree", "prune"], cwd=str(ROOT),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# Each mutation must break a *semantic* property, not merely a syntactic one.
# Deleting a line or renaming a symbol proves nothing: a crash is caught by any
# test that runs the code at all. These change behaviour while keeping the code
# plausible -- the shape a real regression actually takes.
MUTATIONS = [
    {
        "id": "constellation-transitive-chain-admitted",
        "path": "meshpool/controlplane/scheduler.py",
        "old": (b"        clusters: list[list[str]] = []\n"
                b"        for candidate in sorted(component):\n"
                b"            for cluster in clusters:\n"
                b"                if all(pair_rtt.get((candidate, existing), float(\"inf\")) <= rtt_threshold_ms\n"
                b"                       for existing in cluster):\n"
                b"                    cluster.append(candidate)\n"
                b"                    break\n"
                b"            else:\n"
                b"                clusters.append([candidate])\n"
                b"        for component in clusters:\n"
                b"            if len(component) < min_size:\n"
                b"                continue\n"
                b"            rtts = [\n"
                b"                pair_rtt[(a, b)]\n"
                b"                for i, a in enumerate(component)\n"
                b"                for b in component[i + 1:]\n"
                b"            ]"),
        "new": (b"        clusters: list[list[str]] = [sorted(component)]\n"
                b"        for component in clusters:\n"
                b"            if len(component) < min_size:\n"
                b"                continue\n"
                b"            rtts = [\n"
                b"                _safe_rtt(rtt_predictor, a, b)\n"
                b"                for i, a in enumerate(component)\n"
                b"                for b in component[i + 1:]\n"
                b"            ]"),
        "cmd": [sys.executable, "sim/scenarios/verify_constellation_formation.py"],
        "why": (
            "Reintroduces connected-component clustering: A-B and B-C being near is "
            "treated as enough to put A, B, and C into one virtual machine even when "
            "A-C is a 150 ms WAN hop. Larger happy-path metro tests still pass because "
            "their injected clusters are cliques; only a non-vacuous chain fixture that "
            "measures every advertised internal RTT can prove distant nodes did not join."
        ),
    },
    {
        "id": "constellation-selects-flops-before-rtt",
        "path": "meshpool/controlplane/scheduler.py",
        "old": (b"key=lambda c: (\n"
                b"            max(0, pipeline_depth - 1) * c.avg_internal_rtt_ms,"),
        "new": (b"key=lambda c: (\n"
                b"            -c.total_flops,"),
        "cmd": [sys.executable, "sim/scenarios/verify_constellation_formation.py"],
        "why": (
            "Makes the scheduler prefer the fastest aggregate FLOPS before RTT, which "
            "keeps functional placements valid and is invisible to tests that only assert "
            "a model fits somewhere. The constellation gate now constructs two fitting "
            "groups where the lower-average-RTT group is deliberately slower and has a "
            "higher max edge, so only the intended depth-times-average-RTT preference wins."
        ),
    },
    {
        "id": "app-router-ignores-health",
        "path": "meshpool/hosting/engine.py",
        "old": b"        healthy = [r for r in dep.replicas if r.healthy]",
        "new": b"        healthy = list(dep.replicas)",
        "cmd": [sys.executable, "-m", "pytest", "tests/test_procapp.py",
                "-m", "slow or not slow", "-q"],
        "why": (
            "A dead process can remain in the deployment's replica list until the "
            "next reconciliation pass. If the load balancer ignores the healthy bit, "
            "every other request can still be routed to that dead replica while the "
            "routing table looks populated. The guard counts selections as delivered "
            "requests and requires the unhealthy replica to receive exactly zero."
        ),
    },
    {
        "id": "app-rollout-half-cutover",
        "path": "meshpool/hosting/engine.py",
        "old": b"            candidate.replicas = healthy_new",
        "new": b"            candidate.replicas = healthy_new[:max(1, desired // 2)]",
        "cmd": [sys.executable, "-m", "pytest", "tests/test_procapp.py",
                "-m", "slow or not slow", "-q"],
        "why": (
            "A blue/green rollout is not complete when only half of the requested "
            "green pool is healthy. One-replica happy paths cannot distinguish that "
            "from a real cutover, so the guard rolls out two replicas and asserts "
            "that all desired new replicas replace the old pool before it drains."
        ),
    },
    {
        "id": "app-proxy-serves-5xx-replica",
        "path": "meshpool/controlplane/app.py",
        "old": b"            if upstream.status_code >= 500:",
        "new": b"            if False and upstream.status_code >= 500:",
        "cmd": [sys.executable, "-m", "pytest", "tests/test_procapp.py",
                "-m", "slow or not slow", "-q"],
        "why": (
            "Health-based routing also has to reject a replica that accepts the TCP "
            "connection but returns a 5xx. A before/after deployment test can pass "
            "while users receive that 503. The guard drives the proxy through a bad "
            "replica followed by a good one and asserts the client only receives the "
            "retried 200 from the healthy target."
        ),
    },
    {
        "id": "sandbox-runs-work-without-egress-enforcement",
        "path": "meshpool/agent/sandbox.py",
        "old": b"        install_egress_hook(EgressPolicy())  # deny-by-default",
        "new": b"        pass  # egress hook not installed",
        "cmd": [sys.executable, "sim/scenarios/verify_egress_filter.py"],
        "why": (
            "This one line stands between a donated PC and a botnet, and deleting it "
            "used to be completely undetectable. Every egress assertion in the suite "
            "exercised an EgressPolicy object directly -- which proves the policy's "
            "logic but not that any workload is subject to it -- and the one check that "
            "claimed to cover runtimes read a class attribute the class sets on itself. "
            "Policy logic can be flawless and still protect nobody if nothing installs "
            "it. The gate now runs a real sandboxed child against a real listener and "
            "requires the listener to accept nothing, with a nonce proving the child "
            "actually ran and a paired unhooked connection proving the destination was "
            "reachable all along, so 'denied' cannot quietly mean 'unreachable'."
        ),
    },
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
    {
        "id": "hydration-drops-without-rebuilding",
        "path": "meshpool/inference/pipeline.py",
        "old": b"            out = stage.forward(act, is_tokens=False, pos_offset=pos, session=session)",
        "new": b"            out = None",
        "cmd": [sys.executable, "-m", "pytest", "tests/test_kvcache.py",
                "-m", "slow or not slow"],
        "why": (
            "Recovery that is worse than doing nothing: the stage's cache is cleared "
            "one line above, so skipping the replay leaves it EMPTY while the caller "
            "is told it was rebuilt. This is the shape the real defect took -- the "
            "coroutine was created and dropped -- and local shards are synchronous, "
            "so an in-process test cannot see it. Catching it needs a stage that is "
            "genuinely a coroutine and an assertion that a forward actually ran."
        ),
    },
    {
        "id": "spare-rotates-without-hydrating",
        "path": "meshpool/inference/remote.py",
        "old": b"        self._needs_hydration = set(self._sessions)",
        "new": b"        self._needs_hydration = set()",
        "cmd": [sys.executable, "-m", "pytest", "tests/test_hot_spares.py",
                "-m", "slow or not slow"],
        "why": (
            "Failover still happens -- the dead node is retired and a spare takes over "
            "-- but no session is marked dirty, so the spare is handed a decode step at "
            "a non-zero position with an empty KV cache. Every failover counter still "
            "increments and the telemetry still says the stage recovered. Only an "
            "end-to-end oracle comparing post-kill output against a clean run notices "
            "that the pipeline is now serving from a stage that remembers nothing."
        ),
    },
    {
        "id": "per-call-httpx-client-on-hot-path",
        "path": "meshpool/cli/main.py",
        "old": b'    r = _http().get(f"{base_url()}{path}", headers=_auth_headers(), timeout=30)',
        "new": b'    r = httpx.get(f"{base_url()}{path}", headers=_auth_headers(), timeout=30)',
        "cmd": [sys.executable, "-m", "pytest", "tests/test_cli.py",
                "-m", "slow or not slow"],
        "why": (
            "The module-level wrapper builds a whole httpx.Client per call, and "
            "constructing one loads the system TLS trust store: measured at ~1.0 s "
            "here, versus 5 ms with verification off. Nothing breaks -- every command "
            "returns the same bytes -- the CLI just becomes a second slower per "
            "request and every benchmark that starts its stopwatch outside the call "
            "silently reports the certificate parse as network latency. A behavioural "
            "test would only cover the call sites it happens to exercise, so the guard "
            "is static and covers the whole file."
        ),
    },
    {
        "id": "streaming-is-generate-then-replay",
        "path": "meshpool/inference/engine.py",
        "old": b"            on_tokens=queue.put_nowait, **kw))",
        "new": b"            on_tokens=None, **kw))",
        "cmd": [sys.executable, "-m", "pytest", "tests/test_streaming.py",
                "-m", "slow or not slow"],
        "why": (
            "The exact shape the real defect took. The batcher stops handing tokens "
            "to the streaming caller, so nothing can be yielded until generation "
            "finishes and the entire answer arrives as one final delta. Every SSE "
            "frame is still well formed, the delta/finish_reason schema is still "
            "correct, and [DONE] still arrives -- which is why verify_openai_compat "
            "passed against a fake for as long as it existed. A format gate cannot "
            "detect a fake. Only asserting that a callback fires WHILE the generation "
            "task is still running can."
        ),
    },
    {
        "id": "scheduler-loop-dies-in-silence",
        "path": "meshpool/inference/batching.py",
        "old": b"                except Exception as exc:  # noqa: BLE001",
        "new": b"                except ZeroDivisionError as exc:",
        "cmd": [sys.executable, "-m", "pytest", "tests/test_streaming.py",
                "tests/test_batching.py", "-m", "slow or not slow"],
        "why": (
            "Narrowing the guard around caller-supplied draft_factory() lets one bad "
            "request escape into _loop and kill the shared scheduler. The task holds "
            "the exception, the queue is never drained again, and EVERY tenant's "
            "future hangs forever -- no error, no timeout, indistinguishable from a "
            "slow model. The original defect was found only because a test hung, "
            "which is the worst way to find it: a suite with no timeout simply stops. "
            "The guard must assert that a poisoned request is rejected AND that an "
            "unrelated request submitted afterwards still completes."
        ),
    },
    {
        "id": "harness-cannot-kill-fast-enough-to-test-failure",
        "path": "sim/harness.py",
        "old": b"    proc.kill()\n    if sys.platform == \"win32\":\n        threading.Thread(",
        "new": b"    _kill_tree(proc)\n    if False:\n        threading.Thread(",
        "cmd": [sys.executable, "sim/scenarios/verify_stage_failure_midstream.py"],
        "why": (
            "A defect in the instrument, not the product, and the most dangerous kind. "
            "taskkill /T /F returns in ~367 ms while Popen.kill returns in ~0.3 ms "
            "(measured, 1200x). A 44-token answer finishes INSIDE the slow kill, so "
            "the gate killed a node 'mid-stream', got byte-identical output, and "
            "declared the failover path healthy -- while the failover path had never "
            "run. The tell was failovers=0 in a separate check. Any gate that times a "
            "failure against live traffic must both kill instantly and assert that "
            "work was provably done after the victim was confirmed dead."
        ),
    },
    {
        "id": "straggler-threshold-becomes-absolute",
        "path": "meshpool/inference/remote.py",
        "old": b"STRAGGLER_RATIO = _env_float(\"MESHPOOL_STRAGGLER_RATIO\", 3.0)",
        "new": b"STRAGGLER_RATIO = _env_float(\"MESHPOOL_STRAGGLER_RATIO\", 1.0)",
        "cmd": [sys.executable, "sim/scenarios/verify_straggler_tolerance.py"],
        "why": (
            "At ratio 1.0 the comparison is 'slower than the median peer', which the "
            "maximum of any set satisfies by definition -- so every deployment always "
            "contains a straggler and the pool rotates forever, paying a KV rebuild "
            "each time to swap one node for another drawn from the same population. "
            "This is what an absolute threshold degenerates into, and it is invisible "
            "to every check that only looks at a healthy mesh or at a single slow "
            "node: those still pass. Only the uniformly-slow check can see it, which "
            "is precisely why that check exists."
        ),
    },
    {
        "id": "slowness-never-triggers-rotation",
        "path": "meshpool/inference/batching.py",
        "old": b"            self._check_stragglers()",
        "new": b"            pass  # self._check_stragglers()",
        "cmd": [sys.executable, "sim/scenarios/verify_straggler_tolerance.py"],
        "why": (
            "Detaches straggler detection from the scheduler loop. Nothing fails, no "
            "answer changes, no request errors -- the deployment simply keeps serving "
            "through the slow node forever at a fraction of its throughput. That is "
            "the whole failure mode consumer hardware produces, and it is silent, so "
            "the gate cannot rely on an error appearing anywhere. It has to assert "
            "that a rotation happened, that it retired the exact node being slowed, "
            "and that throughput came back."
        ),
    },
    {
        "id": "pulled-weights-are-never-verified",
        "path": "meshpool/agent/daemon.py",
        "old": b"        if hashlib.sha256(raw).hexdigest() == blob_id:\n            return raw",
        "new": b"        if raw is not None:\n            return raw",
        "cmd": [sys.executable, "sim/scenarios/verify_weight_swarm_coldstart.py"],
        "why": (
            "Takes the first peer at its word. This is the most dangerous mutation in "
            "the list, because it breaks nothing anyone would notice: every pull still "
            "succeeds, every model still loads, every request still returns fluent "
            "text. Only the text is wrong, and it is wrong in a way that reads as the "
            "model being bad rather than the platform being broken. No timeout fires, "
            "no exception is raised, no counter moves. A gate can only catch this by "
            "planting a peer that actually lies and then hashing what ends up on disk "
            "-- which is the entire reason the dishonest-peer fault exists."
        ),
    },
    {
        "id": "weights-relay-through-the-coordinator",
        "path": "meshpool/inference/engine.py",
        "old": b"            if may_pull and await self._try_peer_pull(node_id, bid):",
        "new": b"            if False and await self._try_peer_pull(node_id, bid):",
        "cmd": [sys.executable, "sim/scenarios/verify_weight_swarm_coldstart.py"],
        "why": (
            "Routes every weight group through the control plane instead of peer to "
            "peer. Still correct, and invisible at nano-gpt scale -- same output, same "
            "timings, every functional check passes. At the scale the product exists "
            "for it is fatal: the coordinator becomes a hard bandwidth bottleneck that "
            "every node join queues behind, which is the exact centralisation the "
            "design exists to avoid. A property that only shows up as a bottleneck at "
            "600 GB has to be asserted structurally at 6 MB, by counting relays and "
            "requiring zero."
        ),
    },
    {
        "id": "weight-peer-selection-ignores-rtt",
        "path": "meshpool/inference/engine.py",
        "old": b"        holders.sort(key=lambda x: (x[0], x[1]))",
        "new": b"        holders.sort(key=lambda x: x[1])",
        "cmd": [sys.executable, "-m", "pytest", "tests/test_weight_pull.py",
                "-q", "-m", "slow or not slow"],
        "why": (
            "A model still loads correctly if every chunk comes from a far origin, "
            "so outcome-only tests cannot see the cold-start killer returning. The "
            "guard creates two genuine online holders with distinct RTTs and asserts "
            "on the ordered source list itself, proving the nearest peer is the one "
            "asked to do the work first."
        ),
    },
    {
        "id": "blob-pull-hides-accepted-source",
        "path": "meshpool/agent/daemon.py",
        "old": b'                accepted_sources.append({"blob_id": blob_id, "source": src})',
        "new": b"                pass",
        "cmd": [sys.executable, "-m", "pytest", "tests/test_weight_pull.py",
                "-q", "-m", "slow or not slow"],
        "why": (
            "A verified blob on disk proves integrity but not which peer supplied "
            "it. Without returning per-chunk source telemetry, a test can pass even "
            "when the implementation tries a malicious or distant source first and "
            "only succeeds later. The guard plants a lying first peer and asserts "
            "the accepted source was the honest one."
        ),
    },
    {
        "id": "shared-weight-chunks-redownloaded",
        "path": "meshpool/storage/engine.py",
        "old": b"        placed = [nid for nid in existing.replicas if nid in live] if existing else []",
        "new": b"        placed = []",
        "cmd": [sys.executable, "-m", "pytest", "tests/test_storage.py",
                "-q", "-m", "slow or not slow"],
        "why": (
            "Two related model uploads still reassemble correctly even if every "
            "shared base chunk is uploaded again. The property is work not done, so "
            "the guard counts blob_put calls and requires the second object to move "
            "only the changed chunk, proving content-addressed chunk dedup is real."
        ),
    },
    {
        "id": "under-replicated-objects-are-never-repaired",
        "path": "meshpool/storage/engine.py",
        "old": b"            if 0 < len(placed) < meta.replication_desired:",
        "new": b"            if False:",
        "cmd": [sys.executable, "-m", "pytest", "-q",
                "tests/test_storage_repair_locality.py", "tests/test_storage.py",
                "-m", "slow or not slow"],
        "why": (
            "Lets the pool bleed replicas. Nothing observable happens for a long "
            "time: every object still has a live copy, every read succeeds, every "
            "model still answers. The pool simply stops replacing what churn takes, "
            "walking monotonically down to one holder per object and staying there. "
            "The first user-visible symptom is permanent data loss, by which point "
            "the bytes are gone.\n"
            "This mutation used to be pointed at verify_weight_swarm_coldstart, "
            "and it SURVIVED there -- which was a fact about the gate, not about "
            "the code. That gate asserts every layer group still has >=2 holders "
            "after churn, and it does, but not because of this line: model weights "
            "are made redundant deliberately by hot-spare warming "
            "(InferenceEngine._place_blobs on each stage's standby), never left to "
            "the generic storage repair loop. So the gate kept passing with "
            "re-replication entirely disabled. The lesson is that a mutation "
            "proves nothing until the command it names is the one that actually "
            "depends on the mutated line; a plausible-sounding pairing is not "
            "evidence. Re-pointed at the tests that drive _reconcile directly."
        ),
    },
    {
        "id": "repair-relays-every-byte-through-the-coordinator",
        "path": "meshpool/storage/engine.py",
        "old": b"        plan = self._peer_pull_plan(blob_id, holders, target, live_by_id)\n        if plan is not None:",
        "new": b"        plan = self._peer_pull_plan(blob_id, holders, target, live_by_id)\n        if False:",
        "cmd": [sys.executable, "-m", "pytest",
                "tests/test_storage_repair_locality.py",
                "-m", "slow or not slow", "-p", "no:cacheprovider"],
        "why": (
            "Restores the original behaviour, which was correct and fatal. Repair "
            "still works, data is still durable, every storage test passes -- the "
            "bytes just travel through the control plane's memory instead of "
            "between the two machines that care. At 6 MB that is a rounding error. "
            "At 600 GB the coordinator must hold each layer group in RAM, base64 it "
            "(+33%), and push it back out, so every node departure in the pool "
            "serialises behind one uplink and the decentralised system acquires a "
            "centralised bottleneck exactly where it hurts most. Nothing about "
            "correctness can detect this; only counting who carried the bytes can."
        ),
    },
    {
        "id": "capability-token-can-grant-scopes-it-does-not-hold",
        "path": "meshpool/controlplane/app.py",
        "old": b'            if excess:\n                raise HTTPException(status_code=403, detail=(',
        "new": b'            if False:\n                raise HTTPException(status_code=403, detail=(',
        "cmd": [sys.executable, "-m", "pytest",
                "tests/test_capability_token_api.py",
                "-m", "slow or not slow", "-p", "no:cacheprovider"],
        "why": (
            "Turns every token into a root key with extra steps. A CI token minted "
            "with jobs:read could mint itself a second token holding cluster:admin, "
            "and nothing in the system would look wrong: the mint succeeds, the "
            "audit log shows a normal mint by a legitimate owner, and the escalated "
            "token is indistinguishable from one the account holder created "
            "deliberately. Containment is the entire reason to hand out a narrow "
            "token instead of a session, so a delegation path that can widen its "
            "own authority makes the feature strictly worse than sharing the "
            "session it replaces."
        ),
    },
    {
        "id": "revoking-a-leaked-token-does-not-take-effect",
        "path": "meshpool/controlplane/app.py",
        "old": b'    if row["revoked_ms"] is not None or int(row["expires_at_ms"]) < now:',
        "new": b'    if int(row["expires_at_ms"]) < now:',
        "cmd": [sys.executable, "-m", "pytest",
                "tests/test_capability_token_api.py",
                "-m", "slow or not slow", "-p", "no:cacheprovider"],
        "why": (
            "Revocation is the one control a user reaches for during an active "
            "incident, and it is the one whose failure they cannot see. The DELETE "
            "returns 200, the token disappears from the listing, and the dashboard "
            "shows it gone -- while the leaked secret keeps working until its "
            "natural expiry, possibly thirty days later. Every surface reports "
            "success; only the attacker knows otherwise. This is why revocation is "
            "checked against the database on every single request rather than "
            "cached, and why the test re-uses the secret on the immediately "
            "following call instead of trusting the 200."
        ),
    },
    {
        "id": "token-listing-leaks-a-derived-secret",
        "path": "meshpool/controlplane/app.py",
        "old": b'            "spent": row["spent"],\n',
        "new": b'            "spent": row["spent"],\n            "token_hash": row["token_hash"],\n',
        "cmd": [sys.executable, "-m", "pytest",
                "tests/test_capability_token_api.py",
                "-m", "slow or not slow", "-p", "no:cacheprovider"],
        "why": (
            "Listing a token hash is still leaking a value derived from the secret. "
            "The bearer cannot be recovered from SHA-256, but publishing a stable "
            "verifier gives attackers an offline oracle and violates the product "
            "promise that the secret is shown exactly once. This regression is "
            "invisible to tests that assert only field names or only that the raw "
            "secret string is absent; the gate asserts on the serialized response "
            "body so any derived verifier added to the read path is caught."
        ),
    },
    {
        "id": "a-new-route-ships-with-no-declared-scope",
        "path": "meshpool/controlplane/app.py",
        "old": b'    _declare_route("DELETE", "/api/v1/tokens/{token_id}",     Scope.TOKENS_WRITE)\n',
        "new": b"",
        "cmd": [sys.executable, "-m", "pytest",
                "tests/test_capability_token_api.py",
                "-m", "slow or not slow", "-p", "no:cacheprovider"],
        "why": (
            "Models the most likely real-world mistake in this codebase: someone "
            "adds an endpoint and forgets the scope table. The failure is silent "
            "and open-by-default -- the route works, the feature ships, and nobody "
            "notices that it is reachable without the authority it should require. "
            "The guard is a test that enumerates app.routes and fails on any "
            "undeclared method+path, so forgetting is caught mechanically instead "
            "of depending on a reviewer remembering a second file."
        ),
    },
    {
        "id": "erasure-repair-drags-k-shards-through-the-coordinator",
        "path": "meshpool/storage/engine.py",
        "old": b"        if await self._repair_ec_delegated(meta, k, m, live_by_id):\n            return",
        "new": b"        if False:\n            return",
        "cmd": [sys.executable, "-m", "pytest",
                "tests/test_storage_ec_repair_locality.py",
                "-m", "slow or not slow", "-p", "no:cacheprovider"],
        "why": (
            "Restores the relay path, which is correct and does not scale. Every "
            "shard is still rebuilt, every object still reads back, every durability "
            "check still passes -- the coordinator just pulls k shards into its own "
            "memory to rebuild one. That is the replication-repair bottleneck "
            "multiplied by k: at RS(29,80) a single departure drags 29 shards "
            "through one uplink and holds the whole reconstructed object in RAM. "
            "Because repair is the steady-state cost of a churning pool rather than "
            "an exceptional event, the cost is paid continuously and grows with the "
            "pool. No correctness assertion can see it; only counting who carried "
            "the bytes can."
        ),
    },
    {
        "id": "erasure-parity-copied-from-data-shard",
        "path": "meshpool/storage/erasure.py",
        "old": b"payload = Parity[i].tobytes()",
        "new": b"payload = D[i % k].tobytes()",
        "cmd": [sys.executable, "-m", "pytest", "tests/test_erasure.py",
                "-m", "slow or not slow", "-p", "no:cacheprovider"],
        "why": (
            "The easiest fake erasure implementation is to emit shard-shaped parity "
            "that is only a copy of systematic data. Round trips and decode paths "
            "that mostly use data shards can still look healthy, but there is no "
            "new information to survive loss of the copied data. The test checks the "
            "parity payloads themselves and forces recovery with data shards missing, "
            "so a replica masquerading as parity cannot pass."
        ),
    },
    {
        "id": "erasure-decoder-refuses-parity-shards",
        "path": "meshpool/storage/erasure.py",
        "old": b"chosen = valid_indices[:k]   # first k available",
        "new": b"raise ValueError(\"parity disabled\")\n        chosen = valid_indices[:k]   # first k available",
        "cmd": [sys.executable, "-m", "pytest", "tests/test_erasure.py",
                "-m", "slow or not slow", "-p", "no:cacheprovider"],
        "why": (
            "A systematic RS code can appear to work if every read happens to keep "
            "all data shards. That proves replication-like storage, not erasure "
            "recovery. This mutation makes the decoder reject any path that needs "
            "parity; the dedicated test chooses fewer than k systematic shards and "
            "therefore requires parity rows and matrix inversion to produce the bytes."
        ),
    },
    {
        "id": "erasure-repair-colocates-rebuilt-shard",
        "path": "meshpool/storage/engine.py",
        "old": b"candidates = self._pick_nodes(1, exclude=holders)",
        "new": b"candidates = self._pick_nodes(1, exclude=set())",
        "cmd": [sys.executable, "-m", "pytest",
                "tests/test_storage_ec_repair_locality.py",
                "-m", "slow or not slow", "-p", "no:cacheprovider"],
        "why": (
            "A repaired EC shard can be byte-perfect and still reduce durability if "
            "it is placed on a node that already holds another shard of the same "
            "object. Every read still succeeds, the repair RPC was delegated, and "
            "the shard count returns to k+m, but a later single-node failure loses "
            "two pieces. The guard asserts the repair target was not already a "
            "holder and that the final holder list has no duplicates."
        ),
    },
    {
        "id": "erasure-repair-trusts-a-shard-that-fails-its-content-address",
        "path": "meshpool/agent/daemon.py",
        "old": b"        if hashlib.sha256(raw).hexdigest() != ref.id:\n            raise HTTPException(status_code=502,\n                                detail=\"reconstructed shard failed integrity check\")",
        "new": b"        if False:\n            raise HTTPException(status_code=502,\n                                detail=\"reconstructed shard failed integrity check\")",
        "cmd": [sys.executable, "-m", "pytest",
                "tests/test_agent_ec_repair.py",
                "-m", "slow or not slow", "-p", "no:cacheprovider"],
        "why": (
            "Delegating repair to a node means the bytes are now assembled by a "
            "machine the coordinator does not control, from shards served by other "
            "machines it does not control. The content address is the only thing "
            "standing between that and a poisoned codeword. Skipping the check on "
            "the reconstructed shard writes whatever came out of the decoder to "
            "disk under a hash it does not have, so the corruption becomes the "
            "stored truth and every later read reproduces it faithfully. Moving "
            "work off the coordinator is only safe because verification moved with "
            "it."
        ),
    },
    {
        "id": "submit-signature-shifts-secret-names",
        "path": "meshpool/batch/engine.py",
        "old": (b"               secret_names: Optional[list[str]] = None,\n"
                b"               *,\n"
                b"               capability_token_id: Optional[str] = None) -> BatchJob:"),
        "new": (b"               capability_token_id: Optional[str] = None,\n"
                b"               secret_names: Optional[list[str]] = None) -> BatchJob:"),
        "cmd": [sys.executable, "-m", "pytest", "tests/test_cli_tokens.py",
                "-m", "slow or not slow", "-p", "no:cacheprovider"],
        "why": (
            "This is how the defect actually shipped: a new parameter was added in "
            "the middle of a public signature, and every existing positional caller "
            "silently started passing its argument to the wrong field. It broke API "
            "job submission and every DAG stage launch, and neither "
            "test_batch_engine.py nor test_secrets.py noticed, because both call "
            "submit() with keywords. Fixing the call sites would leave the landmine "
            "armed for whoever adds the next parameter, so the guarantee under test "
            "is the signature itself: capability_token_id must stay keyword-only, "
            "where no positional call can reach it, and secret_names must stay in "
            "position 7, where existing callers already aim."
        ),
    },
    {
        "id": "fuel-budget-ignored",
        "path": "meshpool/functions/wasm.py",
        "old": b"store.set_fuel(limits.fuel)",
        "new": b"store.set_fuel(10_000_000_000)",
        "cmd": [sys.executable, "-m", "pytest", "tests/test_wasm_runtime.py",
                "-m", "slow or not slow", "-p", "no:cacheprovider"],
        "why": (
            "Fuel is the compute meter and the only bound on a guest that never "
            "yields, so ignoring the caller's budget means untrusted code runs "
            "unmetered and unbilled until the wall clock happens to save us. The "
            "gate that used to guard this compared the wall-clock duration of two "
            "runs, which is a race it lost under load while passing in isolation -- "
            "and a timing comparison cannot distinguish 'fuel is ignored' from 'the "
            "machine was busy' even when it wins. It now asserts on fuel directly: "
            "a budget too small to finish must trap with kind == 'fuel', and a "
            "generous budget must complete having burned more than that budget."
        ),
    },
    {
        "id": "meshd-installer-skips-service-registration",
        "path": "agent-go/internal/distribution/distribution.go",
        "old": b"if err := req.Service.Register(spec); err != nil {\n\t\treturn err\n\t}\n\tif err := req.Service.Start(ServiceName); err != nil {",
        "new": b"if false {\n\t\treturn nil\n\t}\n\tif err := req.Service.Start(ServiceName); err != nil {",
        "cmd": ["go", "test", "-C", "agent-go", "./internal/distribution/...", "-count=1"],
        "why": (
            "An installer that lays down files but never registers the daemon is the "
            "adoption failure this work exists to prevent: a non-technical "
            "contributor believes they joined, but the node never starts at boot and "
            "never donates capacity. A test that asserted only the install command's "
            "exit code or the binary path would miss it. The guard inspects the "
            "post-install service state by exact service name and requires it to be "
            "both present and started."
        ),
    },
    {
        "id": "meshd-uninstall-leaves-runtime-secrets-behind",
        "path": "agent-go/internal/distribution/distribution.go",
        "old": b"p.IdentityPath,\n\t\tp.WeightCacheDir,",
        "new": b"p.WeightCacheDir,",
        "cmd": ["go", "test", "-C", "agent-go", "./internal/distribution/...", "-count=1"],
        "why": (
            "Uninstall completeness is an absence property, and the costly misses "
            "are runtime state rather than the executable: identity keys tie a "
            "machine to a pool identity and cached weights can occupy gigabytes. "
            "A check that only proves meshd no longer launches would pass while "
            "private keys and user data remain on disk. The guard first proves the "
            "identity file existed, then requires that exact path to be gone."
        ),
    },
    {
        "id": "update-downgrade-check-disabled",
        "path": "agent-go/internal/update/update.go",
        "old": b"cmp, err := CompareVersions(m.Version, currentVersion)",
        "new": b'cmp, err := CompareVersions(m.Version, "0.0.0")',
        "cmd": ["go", "test", "-C", "agent-go", "./internal/update/...", "-count=1"],
        "why": (
            "The downgrade check is the one auto-update defence that passes every "
            "signature and hash test by construction, because a replayed release IS "
            "genuine -- correctly signed, correct hash, simply old. An attacker who "
            "re-serves last month's build reintroduces whatever was patched since, "
            "and nothing about the bytes looks wrong. Pinning the version floor to a "
            "constant is how it dies in real life: a debugging edit or a refactor "
            "that loses the parameter, leaving the happy path green and every "
            "crypto assertion still passing. Only an explicit test that replays an "
            "authentic older manifest can notice."
        ),
    },
    {
        "id": "update-keyid-hint-becomes-a-filter",
        "path": "agent-go/internal/update/update.go",
        "old": (b"\tfor id, pub := range trusted {\n"
                b"\t\tif id != sm.KeyID {\n"
                b"\t\t\tcandidates = append(candidates, pub)\n"
                b"\t\t}\n"
                b"\t}"),
        "new": (b"\tif len(candidates) == 0 {\n"
                b"\t\tfor _, pub := range trusted {\n"
                b"\t\t\tcandidates = append(candidates, pub)\n"
                b"\t\t}\n"
                b"\t}"),
        "cmd": ["go", "test", "-C", "agent-go", "./internal/update/...", "-count=1"],
        "why": (
            "KeyID travels outside the signature, so an attacker rewrites it for "
            "free. Treating it as a filter instead of a hint is not a signature "
            "bypass -- an unknown key still fails -- which is exactly why it slips "
            "through review and through every forgery test. It is an availability "
            "attack that is a security attack here: naming any OTHER trusted key "
            "makes every genuine update fail verification, freezing the fleet on "
            "whatever build it runs. That is the same outcome as a downgrade, "
            "reached by denial rather than replay. An unauthenticated field may "
            "reorder work; it must never change the outcome. The mutated form also "
            "breaks key rotation, so it costs availability even with no attacker."
        ),
    },
    {
        "id": "update-rollback-removed",
        "path": "agent-go/internal/update/install.go",
        "old": b"if rbErr := in.doRename(prev, in.Target); rbErr != nil {",
        "new": b"if rbErr := error(nil); rbErr != nil {",
        "cmd": ["go", "test", "-C", "agent-go", "./internal/update/...", "-count=1"],
        "why": (
            "Between renaming the current binary aside and renaming the new one in, "
            "the install directory holds no runnable binary. If the second rename "
            "fails and nothing puts the old one back, the contributor's machine is "
            "left permanently dead -- and unreachably so, because the process that "
            "would report the fault is the one that no longer exists. Recovery code "
            "is deleted precisely because it looks like it never runs: every happy "
            "path, every crypto assertion and every fresh-install test stays green "
            "without it. Only a test that forces the second rename to fail can tell."
        ),
    },
    {
        "id": "update-download-unbounded",
        "path": "agent-go/internal/update/install.go",
        "old": b"limited := io.LimitReader(r, in.Artifact.Size+1)",
        "new": b"limited := io.LimitReader(r, 1<<40)",
        "cmd": ["go", "test", "-C", "agent-go", "./internal/update/...", "-count=1"],
        "why": (
            "The signed size is what stops a hostile or merely broken release host "
            "from streaming until a contributor's disk is full, which is an outage "
            "we inflicted on someone who volunteered their machine. This entry "
            "exists because the original test SURVIVED this exact mutation: an "
            "unbounded read still ends in a length mismatch, so asserting on the "
            "rejection proved nothing -- it just happened after the damage. The "
            "property is that we stop reading, and only counting the bytes actually "
            "pulled from the source can observe it."
        ),
    },
    {
        "id": "update-manifest-fetch-unbounded",
        "path": "agent-go/internal/update/check.go",
        "old": b"raw, err := io.ReadAll(io.LimitReader(resp.Body, max+1))",
        "new": b"raw, err := io.ReadAll(resp.Body)",
        "cmd": ["go", "test", "-C", "agent-go", "./internal/update/...", "-count=1"],
        "why": (
            "The manifest is read before a single thing about it is trusted, so the "
            "opening move of every update check is a read from an unauthenticated "
            "host -- and without a ceiling it is an unbounded one, on a machine "
            "somebody lent us. This is the second entry for the same blind spot, "
            "which is the reason it is recorded rather than merely fixed: asserting "
            "that an oversized body is REJECTED survives deleting the limit, because "
            "the length check still fires, just after the bytes have arrived. Only "
            "counting what the client pulled can tell the difference, and server-side "
            "write counts cannot -- a small body vanishes into the socket buffer "
            "whether the client reads one byte or all of them."
        ),
    },
    {
        "id": "update-no-op-check-reported-as-failure",
        "path": "agent-go/internal/update/check.go",
        "old": b'\t\t\treturn "", ErrNoUpdate',
        "new": b'\t\t\treturn "", err',
        "cmd": ["go", "test", "-C", "agent-go", "./internal/update/...", "-count=1"],
        "why": (
            "Being up to date is what almost every check finds, and it is not a "
            "fault. Returning the raw ErrDowngrade instead makes the fleet's routine "
            "no-op checks indistinguishable from an attacker replaying an old "
            "release -- so either the operator learns to ignore the alarm, or a real "
            "broken release channel is invisible among thousands of them. Nothing "
            "about the update itself misbehaves, so only an assertion on which "
            "outcome was reported can catch it."
        ),
    },
    {
        "id": "serve-guard-kills-a-server-that-is-still-booting",
        "path": "meshpool/common/serve_guard.py",
        "old": b"        if not self._ready:  # never answered yet: still booting, nothing to kill",
        "new": b"        if False:  # mutated: kill even a server that never came up",
        "cmd": [sys.executable, "-m", "pytest", "-q", "tests/test_serve_guard.py",
                "-m", "slow or not slow"],
        "why": (
            "The guard exists to catch a listener that died, not one that has not "
            "opened yet. Without the readiness gate a server that takes longer than "
            "threshold*interval to bind -- a cold start, a slow disk, a machine "
            "swapping -- is shot on sight, turning a slow boot into a restart loop "
            "that never converges. Every failure-detection test still passes, "
            "because they all begin from a healthy probe; only a test that starts "
            "from a server which has never answered can see it."
        ),
    },
    {
        "id": "serve-guard-never-forgives-a-transient-blip",
        "path": "meshpool/common/serve_guard.py",
        "old": b"            self._consecutive_failures = 0",
        "new": b"            pass  # mutated: never forgive a transient blip",
        "cmd": [sys.executable, "-m", "pytest", "-q", "tests/test_serve_guard.py",
                "-m", "slow or not slow"],
        "why": (
            "The counter is meant to measure *consecutive* failures. Leaving it "
            "unreset turns the guard into a cumulative one: a node that drops one "
            "probe a day is executed after `threshold` days of otherwise perfect "
            "uptime, and the kill looks completely unrelated to anything happening "
            "at the time. Every straightforward test still passes -- it fires when "
            "it should and stays quiet when healthy -- because only an "
            "interleaving of failures and recoveries distinguishes the two."
        ),
    },
    {
        "id": "proactor-fix-treats-a-lost-client-as-a-lost-listener",
        "path": "meshpool/common/proactor_fix.py",
        "old": b'        return getattr(exc, "winerror", None) in TRANSIENT_WINERRORS',
        "new": b"        return False  # mutated: no OSError is ever transient",
        "cmd": [sys.executable, "-m", "pytest", "-q", "tests/test_proactor_fix.py",
                "-m", "slow or not slow"],
        "why": (
            "WinError 64 arrives as a plain OSError, not a ConnectionAbortedError, "
            "so this single line is the entire difference between absorbing a "
            "vanished client and letting CPython close the listening socket "
            "forever. With it gone the wrapper still installs, still returns a "
            "future, still handles real connections, and still passes every test "
            "that only checks a server accepts traffic -- the node simply goes "
            "deaf the first time a peer disappears between SYN and accept, which "
            "on residential links is routine. Only a test that injects the error "
            "against a live listener and then reconnects can tell the difference."
        ),
    },
    {
        "id": "mutation-guard-stomps-a-live-gate-mid-mutation",
        "path": "sim/mutation_guard.py",
        "old": b"        if not force and pid != os.getpid() and _pid_alive(pid):",
        "new": b"        if False:  # mutated: restore regardless of who owns it",
        "cmd": [sys.executable, "-m", "pytest", "-q", "tests/test_mutation_guard.py",
                "-m", "slow or not slow"],
        "why": (
            "Crash recovery that cannot tell a dead process from a running one is "
            "worse than none: it restores a file another gate is deliberately "
            "holding mutated, so that gate concludes its non-vacuity check passed "
            "against unbroken code. Every straightforward test still passes, "
            "because they all recover from a process that really is dead. Only a "
            "test that keeps a mutation in flight while calling restore_stale can "
            "see it."
        ),
    },
    {
        "id": "mutation-guard-cannot-tell-a-dead-process-from-a-live-one",
        "path": "sim/mutation_guard.py",
        "old": b'        return str(pid) in (out.stdout or "")',
        "new": b"        return True  # mutated: every pid looks alive",
        "cmd": [sys.executable, "-m", "pytest", "-q", "tests/test_mutation_guard.py",
                "-m", "slow or not slow"],
        "why": (
            "If every pid looks alive then nothing is ever recovered, and the "
            "journal becomes a directory of evidence about a corruption nobody "
            "fixes. This is the failure mode the module exists to prevent, and it "
            "is invisible to any test that does not actually kill a process: the "
            "journal is written correctly, the manifest parses, pending() reports "
            "the entry -- restore_stale simply declines to act on it forever."
        ),
    },
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _anchor(original: bytes, mut: dict) -> tuple[bytes, bytes]:
    """Resolve the anchor against whatever line endings this checkout happens to use.

    Mutation anchors are written with ``\\n`` because that is what the source files
    contain in git, but on Windows a worktree is checked out with ``\\r\\n``. An
    anchor that spans a newline would then match zero times and the whole entry
    would report as "the code moved" -- a false alarm that trains you to loosen
    anchors, which is the opposite of what this gate is for.
    """
    old, new = mut["old"], mut["new"]
    if original.count(old) == 1:
        return old, new
    crlf_old = old.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    if original.count(crlf_old) == 1:
        return crlf_old, new.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return old, new


def _apply(worktree: Path, mut: dict) -> bytes:
    target = worktree / mut["path"]
    original = target.read_bytes()
    old, new = _anchor(original, mut)
    count = original.count(old)
    if count != 1:
        raise AssertionError(
            f"[{mut['id']}] anchor matched {count} times in {mut['path']}, expected "
            f"exactly 1. The code moved; update this mutation so it keeps testing "
            f"the property it was written for.\n  anchor: {old!r}")
    target.write_bytes(original.replace(old, new))
    return original


def main() -> int:
    print("== mutation coverage: do our strongest gates actually detect breakage? ==\n")
    _sweep_abandoned_worktrees()

    # Optional id filter keeps the edit/run loop cheap while writing a new entry.
    wanted = set(sys.argv[1:])
    selected = [m for m in MUTATIONS if not wanted or m["id"] in wanted]
    unknown = wanted - {m["id"] for m in MUTATIONS}
    assert not unknown, f"unknown mutation id(s): {sorted(unknown)}"

    pre = {m["path"]: _sha(ROOT / m["path"]) for m in selected}
    tmp = Path(tempfile.mkdtemp(prefix="meshpool-mut-"))
    # Claim it before creating the worktree, so a concurrent sweep can never see
    # an unowned directory that is about to be in use.
    (tmp / "owner.pid").write_text(str(os.getpid()), encoding="utf-8")
    worktree = tmp / "wt"
    survivors: list[dict] = []
    unrunnable: list[tuple[dict, str]] = []
    vacuous: list[dict] = []
    # A guard command that fails on UNMUTATED code would be scored "caught" for
    # every mutation it guards, because this harness only ever looked at the exit
    # code of the mutated run. That is the exact failure mode this whole file
    # exists to hunt -- a check that cannot distinguish broken code from working
    # code -- and it was sitting in the hunter. So every distinct command is run
    # once against the pristine worktree first and must pass. Deduped and cached,
    # because 44 mutations share only 27 commands and several take minutes.
    baseline_ok: dict[tuple, bool] = {}

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
            key = tuple(mut["cmd"])
            if key not in baseline_ok:
                try:
                    base = subprocess.run(mut["cmd"], cwd=str(worktree), env=env,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, timeout=1800)
                    baseline_ok[key] = base.returncode == 0
                    if not baseline_ok[key]:
                        print(f"   [BASELINE FAILED] exit {base.returncode} on "
                              f"unmutated code")
                        for line in base.stdout.decode("utf-8", "replace").strip(
                                ).splitlines()[-6:]:
                            safe = line[:120].encode("ascii", "replace").decode(
                                "ascii")
                            print(f"      | {safe}")
                except FileNotFoundError as exc:
                    unrunnable.append((mut, str(exc)))
                    print(f"   [UNRUNNABLE] cannot execute {mut['cmd'][0]!r}: {exc}")
                    print()
                    continue
                except subprocess.TimeoutExpired as exc:
                    baseline_ok[key] = False
                    print(f"   [BASELINE FAILED] timed out on unmutated code: {exc}")
            if not baseline_ok[key]:
                # Do not even run the mutation: its verdict would be meaningless.
                vacuous.append(mut)
                print("   [VACUOUS] the guarding gate does not pass on working "
                      "code, so its failure proves nothing")
                print()
                continue

            original = _apply(worktree, mut)
            try:
                proc = subprocess.run(mut["cmd"], cwd=str(worktree), env=env,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, timeout=1800)
            except FileNotFoundError as exc:
                # The gate's command needs a toolchain this machine lacks -- since
                # entries were added for the Go agent, that is usually a missing
                # `go`. Letting this propagate would abort the loop and take the
                # verdict on every *other* mutation down with it, turning "one
                # toolchain is absent" into "we learned nothing". So record it as
                # its own loud category and keep going. It is never silently
                # skipped: an unrunnable entry fails the gate at the end, exactly
                # like a survivor, because an unevaluated claim is not a passed one.
                unrunnable.append((mut, str(exc)))
                print(f"   [UNRUNNABLE] cannot execute {mut['cmd'][0]!r}: {exc}")
                print()
                continue
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

    if vacuous:
        print()
        for mut in vacuous:
            print(f"VACUOUS GUARD: {mut['id']}")
            print("  The guarding command below fails on unmutated code, so it "
                  "would have scored this mutation 'caught' no matter what the "
                  "mutation did. Fix the gate before trusting the verdict.")
            print(f"  Guarded by: {' '.join(str(c) for c in mut['cmd'][1:])}")
            print()

    if unrunnable:
        print()
        for mut, why in unrunnable:
            print(f"NOT EVALUATED: {mut['id']}")
            print(f"  {why}")
            print(f"  Needed: {' '.join(mut['cmd'])}")
            print()

    if survivors or unrunnable or vacuous:
        raise AssertionError(
            f"{len(survivors)} of {len(selected)} mutations survived, "
            f"{len(vacuous)} had a vacuous guard and "
            f"{len(unrunnable)} could not be evaluated. A survivor means the gate "
            "above cannot fail, so it is not protecting what it claims to -- fix "
            "the test, because a surviving mutation never means the code was fine. "
            "A vacuous guard means the gate fails on working code too, so its "
            "failure was never evidence of anything. "
            "An unevaluated mutation means we have no evidence either way, which "
            "is not the same as passing: install the missing toolchain.")

    print(f"\nALL {len(selected)} MUTATIONS CAUGHT "
          f"({len(baseline_ok)} guard commands verified to pass unmutated)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"\nFAILED: {exc}")
        sys.exit(1)
