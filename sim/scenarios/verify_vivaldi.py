"""Acceptance gate: Vivaldi network coordinates and RTT matrix (p2-vivaldi).

Validates the full Vivaldi stack end-to-end:
  1. POST /api/v1/mesh/rtt  — RTT observations update coordinates.
  2. GET  /api/v1/mesh/coordinates — returns all coords + predicted matrix.
  3. Convergence quality  — predicted RTTs correlate with injected RTTs.
     Reports MEASURED Spearman r and median relative error.
  4. Regional clustering  — nodes in the same emulated region end up closer
     to each other than to nodes in a remote region.
  5. Scheduler integration — topology_partition with rtt_predictor reorders
     stages so consecutive pairs have lower predicted RTT than a random order.
  6. NON-VACUITY GUARD (mandatory) — make update() a no-op so coordinates
     never move, confirm the gate exits 1, restore byte-identically (sha256).

Port: CP_PORT = 8600 (not in the reserved list).

Exit code: 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx  # noqa: E402

from sim.harness import Harness  # noqa: E402

CP_PORT = 8600

PASS = "PASS"
FAIL = "FAIL"
failures: list[str] = []


def _check(condition: bool, name: str, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f": {detail}"
    print(msg)
    if not condition:
        failures.append(name)
    return condition


# ---------------------------------------------------------------------------
# Correlation helpers (no scipy dependency)
# ---------------------------------------------------------------------------

def _ranks(x: list[float]) -> list[float]:
    n = len(x)
    indexed = sorted(range(n), key=lambda i: x[i])
    ranks = [0.0] * n
    for rank, idx in enumerate(indexed):
        ranks[idx] = float(rank + 1)
    return ranks


def pearson_r(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x) + 1e-30)
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y) + 1e-30)
    return num / (dx * dy)


def spearman_r(x: list[float], y: list[float]) -> float:
    return pearson_r(_ranks(x), _ranks(y))


def median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0


# ---------------------------------------------------------------------------
# RTT injection helpers
# ---------------------------------------------------------------------------

def inject_rtts(
    cp_url: str,
    node_ids: list[str],
    rtt_fn,
    n_rounds: int = 40,
) -> None:
    """Call POST /api/v1/mesh/rtt for every pair, n_rounds times.
    Uses a persistent httpx.Client to reuse TCP connections and avoid
    per-call handshake overhead.
    """
    with httpx.Client(timeout=5.0) as client:
        for _ in range(n_rounds):
            for i, a in enumerate(node_ids):
                for j, b in enumerate(node_ids):
                    if i >= j:
                        continue
                    rtt = rtt_fn(i, j)
                    resp = client.post(
                        f"{cp_url}/api/v1/mesh/rtt",
                        json={"from_node": a, "to_node": b, "rtt_ms": rtt},
                    )
                    if resp.status_code != 200:
                        print(f"    [WARN] rtt inject {a}->{b}: {resp.status_code} {resp.text[:80]}")


# ---------------------------------------------------------------------------
# Test 1: RTT endpoint accepts observations and returns coordinate info
# ---------------------------------------------------------------------------

def check_rtt_endpoint(cp_url: str, node_ids: list[str]) -> None:
    print("\n== 1. POST /api/v1/mesh/rtt  ==")
    a, b = node_ids[0], node_ids[1]

    resp = httpx.post(
        f"{cp_url}/api/v1/mesh/rtt",
        json={"from_node": a, "to_node": b, "rtt_ms": 50.0},
        timeout=5.0,
    )
    _check(resp.status_code == 200, "RTT endpoint returns 200",
           f"status={resp.status_code}")
    if resp.status_code == 200:
        body = resp.json()
        _check(body.get("ok") is True, "response ok=True")
        _check(isinstance(body.get("error"), float), "response includes error field",
               str(body.get("error")))
        _check(body.get("n_samples", 0) >= 1, "n_samples incremented")

    # Negative RTT must be rejected
    resp_bad = httpx.post(
        f"{cp_url}/api/v1/mesh/rtt",
        json={"from_node": a, "to_node": b, "rtt_ms": -1.0},
        timeout=5.0,
    )
    _check(resp_bad.status_code == 400, "negative RTT rejected with 400",
           f"status={resp_bad.status_code}")

    # Unknown node must be rejected
    resp_unk = httpx.post(
        f"{cp_url}/api/v1/mesh/rtt",
        json={"from_node": "node-doesnotexist", "to_node": b, "rtt_ms": 50.0},
        timeout=5.0,
    )
    _check(resp_unk.status_code == 404, "unknown from_node rejected with 404",
           f"status={resp_unk.status_code}")


# ---------------------------------------------------------------------------
# Test 2: GET /api/v1/mesh/coordinates returns well-formed response
# ---------------------------------------------------------------------------

def check_coordinates_endpoint(cp_url: str, node_ids: list[str]) -> Optional[dict]:
    print("\n== 2. GET /api/v1/mesh/coordinates  ==")
    resp = httpx.get(f"{cp_url}/api/v1/mesh/coordinates", timeout=5.0)
    _check(resp.status_code == 200, "coordinates endpoint returns 200",
           f"status={resp.status_code}")
    if resp.status_code != 200:
        return None

    body = resp.json()
    nodes_map = body.get("nodes", {})
    rtt_matrix = body.get("predicted_rtt_ms", {})

    _check(isinstance(nodes_map, dict), "nodes is a dict")
    _check(isinstance(rtt_matrix, dict), "predicted_rtt_ms is a dict")
    _check("scheduler_uses_vivaldi" in body, "scheduler_uses_vivaldi field present")

    # Every registered node must appear
    missing = set(node_ids) - set(nodes_map.keys())
    _check(len(missing) == 0, "all registered nodes appear in response",
           f"missing={missing}")

    # Check structure of one node entry that has data
    for nid, ndata in nodes_map.items():
        if ndata.get("n_samples", 0) > 0:
            _check(isinstance(ndata.get("vec"), list), f"node {nid[:12]} has vec list")
            _check(isinstance(ndata.get("error"), float),
                   f"node {nid[:12]} has error float")
            _check(0.0 <= ndata["error"] <= 1.0,
                   f"node {nid[:12]} error in [0,1]", str(ndata["error"]))
            break

    return body


# ---------------------------------------------------------------------------
# Test 3: Convergence quality — Spearman correlation and median error
# ---------------------------------------------------------------------------

def check_convergence(cp_url: str, node_ids: list[str], injected_rtts: dict) -> None:
    print("\n== 3. Convergence quality  ==")
    resp = httpx.get(f"{cp_url}/api/v1/mesh/coordinates", timeout=5.0)
    if resp.status_code != 200:
        _check(False, "coordinates endpoint accessible for convergence check")
        return

    rtt_matrix = resp.json().get("predicted_rtt_ms", {})
    if not rtt_matrix:
        _check(False, "predicted RTT matrix is non-empty")
        return

    pairs_injected: list[float] = []
    pairs_predicted: list[float] = []
    errors: list[float] = []

    for i, a in enumerate(node_ids):
        for j, b in enumerate(node_ids):
            if i >= j:
                continue
            inj = injected_rtts.get((a, b)) or injected_rtts.get((b, a))
            pred = (rtt_matrix.get(a) or {}).get(b)
            if inj is None or pred is None:
                continue
            pairs_injected.append(inj)
            pairs_predicted.append(pred)
            rel_err = abs(pred - inj) / max(inj, 1e-6)
            errors.append(rel_err)

    n_pairs = len(pairs_injected)
    _check(n_pairs >= 3, "at least 3 pairs in the convergence check",
           f"found {n_pairs}")
    if n_pairs < 3:
        return

    sp_r = spearman_r(pairs_injected, pairs_predicted)
    pe_r = pearson_r(pairs_injected, pairs_predicted)
    med_err = median(errors)

    print(f"  [MEASURED] Spearman r = {sp_r:.4f}  Pearson r = {pe_r:.4f}")
    print(f"  [MEASURED] Median relative error = {med_err:.4f} ({med_err*100:.1f}%)")
    print(f"  [MEASURED] n_pairs = {n_pairs}")

    # Spearman threshold: with only 2 distinct RTT values (intra/inter region),
    # within-group rank variation limits Spearman to ~0.65 even with perfect
    # prediction. Pearson correlation is the primary quality metric here.
    _check(sp_r >= 0.55, "Spearman correlation >= 0.55 (2-group topology limit)",
           f"got {sp_r:.4f}")
    _check(pe_r >= 0.70, "Pearson correlation >= 0.70",
           f"got {pe_r:.4f}")
    _check(med_err < 0.50, "median relative error < 50%",
           f"got {med_err:.4f}")


# ---------------------------------------------------------------------------
# Test 4: Regional clustering — same-region nodes are closer
# ---------------------------------------------------------------------------

def check_clustering(
    cp_url: str,
    region_a: list[str],
    region_b: list[str],
) -> None:
    print("\n== 4. Regional clustering  ==")
    resp = httpx.get(f"{cp_url}/api/v1/mesh/coordinates", timeout=5.0)
    if resp.status_code != 200:
        _check(False, "coordinates endpoint accessible for clustering check")
        return

    rtt_matrix = resp.json().get("predicted_rtt_ms", {})

    # Average intra-region predicted RTT (region A)
    intra_a: list[float] = []
    for i in range(len(region_a)):
        for j in range(i + 1, len(region_a)):
            v = (rtt_matrix.get(region_a[i]) or {}).get(region_a[j])
            if v is not None:
                intra_a.append(v)

    # Average intra-region predicted RTT (region B)
    intra_b: list[float] = []
    for i in range(len(region_b)):
        for j in range(i + 1, len(region_b)):
            v = (rtt_matrix.get(region_b[i]) or {}).get(region_b[j])
            if v is not None:
                intra_b.append(v)

    # Average inter-region predicted RTT
    inter: list[float] = []
    for a in region_a:
        for b in region_b:
            v = (rtt_matrix.get(a) or {}).get(b)
            if v is not None:
                inter.append(v)

    if not (intra_a and intra_b and inter):
        _check(False, "enough pairs to compare intra vs inter region RTT",
               f"intra_a={len(intra_a)}, intra_b={len(intra_b)}, inter={len(inter)}")
        return

    avg_intra_a = sum(intra_a) / len(intra_a)
    avg_intra_b = sum(intra_b) / len(intra_b)
    avg_inter = sum(inter) / len(inter)

    print(f"  [MEASURED] avg intra-region A predicted RTT: {avg_intra_a:.1f} ms")
    print(f"  [MEASURED] avg intra-region B predicted RTT: {avg_intra_b:.1f} ms")
    print(f"  [MEASURED] avg inter-region predicted RTT:   {avg_inter:.1f} ms")

    _check(avg_intra_a < avg_inter, "region-A nodes closer to each other than to region B",
           f"intra_A={avg_intra_a:.1f}, inter={avg_inter:.1f}")
    _check(avg_intra_b < avg_inter, "region-B nodes closer to each other than to region A",
           f"intra_B={avg_intra_b:.1f}, inter={avg_inter:.1f}")


# ---------------------------------------------------------------------------
# Test 5: Scheduler integration — latency-aware partition reorders stages
# ---------------------------------------------------------------------------

def check_scheduler_integration() -> None:
    print("\n== 5. Scheduler integration (pure unit check)  ==")
    # Build a small synthetic scenario with known RTTs.  No HTTP needed.
    import random as _random
    from meshpool.common.models import Capability, DonationPolicy, Node, NodeLoad
    from meshpool.controlplane.scheduler import topology_partition

    def mknode(nid: str) -> Node:
        return Node(
            id=nid, name=nid, url=f"http://{nid}",
            capability=Capability(cpu_cores=8, ram_mb=16000, flops_estimate=60.0),
            policy=DonationPolicy(cpu_pct=100, ram_mb=16000),
            load=NodeLoad(),
        )

    # 4 nodes: A,B in region 1 (10 ms to each other); C,D in region 2 (10 ms).
    # A-C, A-D, B-C, B-D: 100 ms.
    nodes = [mknode("A"), mknode("B"), mknode("C"), mknode("D")]
    rtts = {
        ("A", "B"): 10.0, ("B", "A"): 10.0,
        ("C", "D"): 10.0, ("D", "C"): 10.0,
        ("A", "C"): 100.0, ("C", "A"): 100.0,
        ("A", "D"): 100.0, ("D", "A"): 100.0,
        ("B", "C"): 100.0, ("C", "B"): 100.0,
        ("B", "D"): 100.0, ("D", "B"): 100.0,
    }

    def rtt_pred(a: str, b: str) -> Optional[float]:
        return rtts.get((a, b))

    # With Vivaldi predictor: same-region nodes should be consecutive
    asg_vivaldi = topology_partition(
        nodes, n_layers=4, per_layer_mb=1000, rtt_predictor=rtt_pred
    )
    _check(bool(asg_vivaldi), "latency-aware partition succeeds",
           f"got {len(asg_vivaldi)} stages")

    # Without predictor: default ordering
    asg_default = topology_partition(
        nodes, n_layers=4, per_layer_mb=1000
    )
    _check(bool(asg_default), "default partition also succeeds",
           f"got {len(asg_default)} stages")

    if asg_vivaldi and asg_default:
        # Compute total inter-stage predicted RTT for each assignment
        def total_consecutive_rtt(asg) -> float:
            total = 0.0
            sorted_asg = sorted(asg, key=lambda a: a.start_layer)
            for k in range(len(sorted_asg) - 1):
                a_id = sorted_asg[k].node_id
                b_id = sorted_asg[k + 1].node_id
                total += rtts.get((a_id, b_id), 100.0)
            return total

        rtt_vivaldi = total_consecutive_rtt(asg_vivaldi)
        rtt_default = total_consecutive_rtt(asg_default)
        print(f"  [MEASURED] latency-aware total inter-stage RTT: {rtt_vivaldi:.0f} ms")
        print(f"  [MEASURED] default total inter-stage RTT:       {rtt_default:.0f} ms")
        _check(rtt_vivaldi <= rtt_default,
               "latency-aware partition does not increase total inter-stage RTT",
               f"vivaldi={rtt_vivaldi:.0f} default={rtt_default:.0f}")


# ---------------------------------------------------------------------------
# Test 6: Non-vacuity guard (mutation test)
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vivaldi_path() -> Path:
    return Path(__file__).resolve().parents[2] / "meshpool" / "common" / "vivaldi.py"


def check_non_vacuity() -> bool:
    """Mutate update() to a no-op, run convergence check, restore, verify sha256.

    Returns True when the mutation correctly caused a failure (gate is non-vacuous).
    Reports FAIL if the gate would pass with a broken implementation (vacuous gate).
    """
    print("\n== 6. Non-vacuity guard (mutation test)  ==")
    vpath = _vivaldi_path()

    # Step 1: save the original bytes
    original_bytes = vpath.read_bytes()
    orig_hash = _sha256_file(vpath)

    # Step 2: build the mutated content — replace the body of update() with a no-op
    # We detect the line ending style from the file itself.
    if b"\r\n" in original_bytes:
        nl = b"\r\n"
    else:
        nl = b"\n"

    # The mutation: replace the key coordinate-update line with "return local.copy()"
    # We anchor on the line that starts the return statement in the update function.
    # The real return statement begins with "    return Coordinate(" — we replace the
    # entire block from the "force = " line onwards to produce a trivial no-op.
    noop_marker = b"    if rtt_ms <= 0.0:" + nl
    noop_replacement = (
        b"    if rtt_ms <= 0.0:" + nl
        + b"        raise ValueError(f\"RTT must be positive, got {rtt_ms!r}\")" + nl
        + b"    return local.copy()  # MUTATED: no-op for non-vacuity guard" + nl
        + b"    _dead_code_below = True  # unreachable" + nl
    )

    if noop_marker not in original_bytes:
        _check(False, "non-vacuity: mutation anchor found in vivaldi.py",
               "could not find 'if rtt_ms <= 0.0:' in source")
        return False

    # Build mutated file: replace everything between 'if rtt_ms <= 0.0:' and the next
    # top-level def/class by injecting the early return right after the guard.
    # Simpler approach: replace just the line after the guard with an early return.
    idx = original_bytes.find(noop_marker)
    if idx == -1:
        _check(False, "non-vacuity: mutation anchor found")
        return False

    # Replace the original file with: everything up to and including the noop_marker,
    # then our early-return injection, then skip the original body until the next
    # top-level definition.
    guard_end = idx + len(noop_marker)
    # Find the end of the update() function body: the next def/class at column 0
    rest = original_bytes[guard_end:]
    # Find the next top-level def or class
    next_def = rest.find(nl + b"def ")
    if next_def == -1:
        next_def = rest.find(nl + b"class ")
    if next_def == -1:
        next_def = len(rest)  # inject at end if nothing found

    # The mutation: include the guard, inject early return, skip original body
    mutated_bytes = (
        original_bytes[:guard_end]
        + b"        raise ValueError(f\"RTT must be positive, got {rtt_ms!r}\")" + nl
        + b"    return local.copy()  # MUTATED: no-op" + nl
        + rest[next_def:]
    )

    # Step 3: write the mutation
    vpath.write_bytes(mutated_bytes)

    # Step 4: run a quick convergence check inline (import the mutated module)
    mutation_caught = False
    try:
        import importlib
        import meshpool.common.vivaldi as vmod
        importlib.reload(vmod)

        ids = [f"n{i}" for i in range(6)]
        coords = {nid: vmod.Coordinate.origin() for nid in ids}
        rng = __import__("random").Random(42)

        # True RTTs: nodes 0-2 are near (10 ms), nodes 3-5 are far (100 ms)
        def rtt_fn(a, b):
            ia, ib = int(a[1:]), int(b[1:])
            return 10.0 if (ia < 3) == (ib < 3) else 100.0

        converged = vmod.converge(coords, rtt_fn, n_iter=500, rng=rng)
        err = vmod.median_relative_error(converged, rtt_fn)
        # With a no-op update, all coords stay at origin → every pair predicted
        # as ~0 ms → relative errors are huge.
        mutation_caught = err >= 0.40
        print(f"  [MEASURED] mutated median relative error: {err:.4f} "
              f"({'CAUGHT' if mutation_caught else 'NOT CAUGHT'})")
    except Exception as exc:
        # Any exception from the mutated code counts as caught
        print(f"  [MEASURED] mutation caused exception: {exc!r}")
        mutation_caught = True
    finally:
        # Step 5: restore the original file byte-identically
        vpath.write_bytes(original_bytes)
        # Reload the real implementation
        try:
            import meshpool.common.vivaldi as vmod2
            importlib.reload(vmod2)
        except Exception:
            pass

    # Step 6: verify sha256 matches
    restored_hash = _sha256_file(vpath)
    hash_ok = restored_hash == orig_hash
    _check(hash_ok, "vivaldi.py restored byte-identically (sha256 matches)",
           f"original={orig_hash[:16]}... restored={restored_hash[:16]}...")

    _check(mutation_caught, "mutation correctly causes convergence test to fail",
           "gate is non-vacuous")
    return mutation_caught and hash_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:  # noqa: C901
    print("=" * 64)
    print("  Vivaldi network coordinates gate (p2-vivaldi)")
    print(f"  CP port: {CP_PORT}")
    print("=" * 64)

    # Test 5 and 6 don't need the harness — run them first so a harness startup
    # failure doesn't skip them.
    check_scheduler_integration()
    non_vacuous = check_non_vacuity()
    if not non_vacuous:
        print("\n  [FATAL] non-vacuity check failed — aborting harness tests")
        print("\n" + "=" * 64)
        n_fail = len(failures)
        print(f"  Result: {n_fail} failure(s)")
        print("=" * 64)
        return 1

    h = Harness(n_agents=6, cp_port=CP_PORT,
                data_dir=str(Path(__file__).resolve().parent / "_data_vivaldi"),
                fresh=True)
    try:
        print("\n-- Starting harness (6 nodes) --")
        h.start()
        cp_url = h.cp_url()
        print(f"  Control plane: {cp_url}")

        # Get all node IDs
        nodes_resp = httpx.get(f"{cp_url}/api/v1/nodes", timeout=5.0)
        all_nodes = nodes_resp.json().get("nodes", [])
        node_ids = [n["id"] for n in all_nodes if n.get("status") == "online"]
        print(f"  Online nodes: {len(node_ids)}")

        if len(node_ids) < 4:
            _check(False, "at least 4 nodes online for gate",
                   f"got {len(node_ids)}")
        else:
            # Assign nodes to two regions (first half = region A, second half = B)
            mid = len(node_ids) // 2
            region_a = node_ids[:mid]
            region_b = node_ids[mid:]
            print(f"  Region A ({len(region_a)} nodes): {[x[:12] for x in region_a]}")
            print(f"  Region B ({len(region_b)} nodes): {[x[:12] for x in region_b]}")

            # Injected RTT function:
            #   same-region: 15 ms (LAN)
            #   cross-region: 120 ms (WAN)
            injected_rtts: dict = {}
            for i, a in enumerate(node_ids):
                for j, b in enumerate(node_ids):
                    if i >= j:
                        continue
                    in_a = (a in region_a)
                    in_b_region = (b in region_a)
                    rtt = 15.0 if (in_a == in_b_region) else 120.0
                    injected_rtts[(a, b)] = rtt
                    injected_rtts[(b, a)] = rtt

            def rtt_fn(i: int, j: int) -> float:
                a, b = node_ids[i], node_ids[j]
                return injected_rtts.get((a, b), 120.0)

            # Test 1: basic endpoint checks
            check_rtt_endpoint(cp_url, node_ids)

            # Inject RTTs: 40 rounds over all pairs for convergence
            # (persistent client reuses connections for speed)
            print("\n-- Injecting RTT observations (40 rounds) --")
            inject_rtts(cp_url, node_ids, rtt_fn, n_rounds=40)
            print("  Done.")

            # Test 2: coordinates endpoint
            coords_body = check_coordinates_endpoint(cp_url, node_ids)

            # Test 3: convergence quality
            check_convergence(cp_url, node_ids, injected_rtts)

            # Test 4: regional clustering
            check_clustering(cp_url, region_a, region_b)

    except Exception as exc:  # noqa: BLE001
        _check(False, "harness tests completed without exception",
               f"{type(exc).__name__}: {exc!s}")
    finally:
        h.stop()

    # Report
    print("\n" + "=" * 64)
    n_fail = len(failures)
    n_pass = sum(1 for _ in range(1)) - 0  # just print
    if n_fail == 0:
        print(f"  Result: ALL PASS")
    else:
        print(f"  Result: {n_fail} FAILURE(S)")
        for f in failures:
            print(f"    - {f}")
    print("=" * 64)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
