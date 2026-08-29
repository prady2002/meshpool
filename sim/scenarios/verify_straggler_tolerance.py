"""Straggler tolerance: a node that gets slow must be routed around, live.

Plan section 3.4. A straggler is the failure mode that consumer hardware produces
constantly and that a pipeline is least able to absorb: ``t_token`` is bounded by the
slowest stage, so one machine that starts thermal-throttling -- or whose owner just
launched a game -- drags every other node in the deployment down with it. Unlike a
crash there is nothing to detect: the node is up, its answers are correct, its health
check is green. It is simply late.

What this gate proves, in order:

1. a healthy baseline exists and can be measured;
2. the injected slowdown is genuinely executed -- measured on the one stage that has
   no warm spare, which is both the only place the full cost is observable and a
   direct test of the designed no-op: a straggler with nowhere better to go must keep
   serving, because slow is strictly better than absent;
3. a stage that *does* have a spare is retired onto it **by the pool itself**, and the
   node retired is the exact node being slowed;
4. throughput returns to the healthy baseline;
5. the answer is still byte-identical -- a rotation must never cost correctness;
6. a *uniformly* slow mesh rotates nothing, because slow is not the same as
   straggling and rotating costs a KV rebuild.

Points 2 and 6 are the ones that keep the mechanism honest. Any implementation with an
absolute millisecond threshold passes the rest and fails 6; any gate that cannot show
the stall costing time is measuring a recovery from nothing.

Step 2 measures on the unspared stage for a reason worth recording: on a *spared*
stage the pool heals inside the first request -- roughly one second from stall to
rotation -- so a "degraded" reading taken there is truncated by the very mechanism
under test and understates the damage by 4-5x.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import Harness  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CP_PORT = 8177
DATA_DIR = str(Path(__file__).resolve().parent / "_data_straggler")

# Big enough that the delay dominates a hop, small enough that the gate stays quick.
STALL_MS = 120.0
# Healthy requests driven before asserting that nothing rotated. Each request is
# MAX_TOKENS traversals, so this is thousands of opportunities for a degenerate
# threshold to fire -- and STRAGGLER_STREAK is only 4.
HEALTHY_DRIVE = 15
MAX_TOKENS = 24
PROMPT = "the "

# A 24-token answer is ~13 pipeline traversals, so STALL_MS on one stage should add
# ~1.6 s to a ~1.0 s request. 2.0x leaves generous headroom for a noisy machine while
# still being far above anything jitter can produce.
MIN_DEGRADATION = 2.0

# One persistent client. A one-shot httpx call pays 400-600 ms building the Windows
# trust store, which would swamp everything measured here.
CLIENT = httpx.Client(timeout=300)

_failures: list[str] = []


def _fail(msg: str) -> None:
    _failures.append(msg)
    print(f"  FAIL: {msg}")


def _ok(msg: str) -> None:
    print(f"  PASS: {msg}")


def _deploy(cp: str) -> dict:
    r = CLIENT.post(f"{cp}/api/v1/models", json={"model": "nano-gpt"}, timeout=180)
    r.raise_for_status()
    return r.json()


def _view(cp: str, model_id: str) -> dict:
    return CLIENT.get(f"{cp}/api/v1/models/{model_id}", timeout=30).json()


def _complete(cp: str, model_id: str) -> tuple[str, float]:
    """One non-streamed completion. Returns (text, seconds)."""
    t0 = time.perf_counter()
    r = CLIENT.post(f"{cp}/v1/chat/completions", json={
        "model": model_id, "temperature": 0.0, "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": PROMPT}]}, timeout=180)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"], dt


def _median_time(cp: str, model_id: str, n: int) -> tuple[str, float]:
    texts, times = [], []
    for _ in range(n):
        t, dt = _complete(cp, model_id)
        texts.append(t)
        times.append(dt)
    if len(set(texts)) != 1:
        _fail(f"the same deterministic prompt produced {len(set(texts))} different "
              f"answers across {n} runs; nothing timed here is comparable")
    return texts[0], statistics.median(times)


def main() -> int:
    h = Harness(n_agents=4, cp_port=CP_PORT, data_dir=DATA_DIR, scale="home")
    try:
        print("== starting cluster (4 agents) ==")
        h.start()
        cp = h.cp_url()
        h.wait_nodes_online(4)

        # ---- 1: healthy baseline ------------------------------------------ #
        print("\n== 1: healthy baseline ==")
        view = _deploy(cp)
        model_id = view["id"]
        view = _view(cp, model_id)
        stages = view["stages"]
        spares = view.get("spares", {}) or {}
        print(f"  {len(stages)} stage(s): "
              + ", ".join(f"{s['node_name']}[{s['first']}:{s['last']}]" for s in stages))
        print(f"  spares: {spares}")
        if not spares:
            _fail("no stage got a warm spare, so there is nowhere to route around a "
                  "straggler and nothing below can be tested")
            raise SystemExit
        baseline_text, baseline_s = _median_time(cp, model_id, 5)
        print(f"  baseline: {baseline_s*1000:.0f} ms/request, "
              f"{len(baseline_text)} chars")
        _ok(f"healthy deployment answers in {baseline_s*1000:.0f} ms")

        # ---- 1b: a healthy mesh must rotate nothing at all ---------------- #
        print("\n== 1b: a healthy mesh rotates nothing ==")
        # STRAGGLER_RATIO is peer-RELATIVE: a stage must be 3x the median peer before
        # it is retired. If that threshold ever degenerates towards 1.0, "slower than
        # the median peer" is satisfied by the maximum of any set BY DEFINITION, so a
        # perfectly healthy mesh contains a "straggler" at all times and rotates
        # forever -- paying a KV rebuild each time to swap one node for another drawn
        # from the same population.
        #
        # Test 6 below hunts the same degeneration but measures a DELTA across a
        # uniformly-slowed window, and that is blind to it: with a near-1.0 ratio the
        # rotations happen HERE, in ordinary healthy traffic, and by the time test 6
        # takes its "before" sample the spares are already spent. Rotation then stops
        # on its own (`spares_remaining < 1` returns early), the delta is 0, and the
        # check reports success on broken code. This was not hypothetical: the
        # 3.0 -> 1.0 mutation survived the whole gate until this check existed.
        #
        # An ABSOLUTE count on a healthy mesh is the only formulation that sees it.
        for _ in range(HEALTHY_DRIVE):
            _complete(cp, model_id)
        rot_healthy = _view(cp, model_id).get("straggler_rotations") or 0
        print(f"  {HEALTHY_DRIVE + 5} healthy requests served; "
              f"straggler_rotations={rot_healthy}")
        if rot_healthy:
            _fail(f"{rot_healthy} straggler rotation(s) on a mesh where nothing was "
                  f"slowed. Every replacement is drawn from the same population, so "
                  f"each one costs a KV rebuild and buys nothing -- the signature of "
                  f"a threshold that is no longer peer-relative")
        else:
            _ok("no rotation on a healthy mesh; the threshold is peer-relative")

        # ---- 2: prove the stall is real, on a stage that cannot rotate ---- #
        print("\n== 2: a slowed stage with no spare keeps serving ==")
        # Deliberately pick the stage that has *no* warm spare. Two things fall out of
        # that choice. It is the only way to measure the full cost of the slowdown:
        # on a spared stage the pool heals inside the very first request (measured:
        # ~1 s from stall to rotation), so the degraded number is truncated by the
        # mechanism under test and cannot serve as its own control. And it exercises
        # the designed no-op path -- a straggler with nowhere better to go must keep
        # serving, because slow is strictly better than absent.
        unspared = [s for s in stages if s["node_name"] not in spares]
        if not unspared:
            _fail("every stage has a spare, so there is no stage on which the "
                  "slowdown can be measured without the pool healing mid-request")
            raise SystemExit
        pinned_name = unspared[0]["node_name"]
        pinned_idx = int(pinned_name.rsplit("-", 1)[1]) - 1
        rot_before = _view(cp, model_id).get("straggler_rotations") or 0
        print(f"  slowing {pinned_name} (no spare) by {STALL_MS:.0f} ms/hop")
        h.set_stall(pinned_idx, STALL_MS)

        pinned_text, pinned_s = _median_time(cp, model_id, 3)
        rot_after = _view(cp, model_id).get("straggler_rotations") or 0
        print(f"  {pinned_s*1000:.0f} ms/request ({pinned_s/baseline_s:.1f}x baseline)"
              f", rotations {rot_before} -> {rot_after}")
        if pinned_s < baseline_s * MIN_DEGRADATION:
            _fail(f"slowing a stage by {STALL_MS:.0f} ms/hop only moved the request "
                  f"time to {pinned_s/baseline_s:.2f}x baseline, under the "
                  f"{MIN_DEGRADATION}x floor. The stall is not reaching the shard, so "
                  f"every recovery below would be measuring nothing")
        else:
            _ok(f"the injected stall really does cost time "
                f"({pinned_s/baseline_s:.1f}x slower) — it is being executed")
        if rot_after != rot_before:
            _fail(f"{rot_after - rot_before} rotation(s) on a stage with no spare. "
                  f"There is nowhere better to go, so this can only have dropped a "
                  f"warm KV cache in exchange for nothing")
        else:
            _ok("nothing rotated — with no spare the pool keeps serving from the "
                "slow node instead of thrashing")
        if pinned_text != baseline_text:
            _fail("the answer changed while a stage was merely slow")

        h.set_stall(pinned_idx, None)
        _complete(cp, model_id)  # let the retired-looking EMA decay before section 3
        _complete(cp, model_id)

        # ---- 3: a spared stage is retired, unprompted ---------------------- #
        print("\n== 3: the pool routes around a straggler on its own ==")
        victim_name = sorted(spares)[0]
        victim_idx = int(victim_name.rsplit("-", 1)[1]) - 1
        victim_stage = next(i for i, s in enumerate(stages)
                            if s["node_name"] == victim_name)
        print(f"  slowing {victim_name} (stage {victim_stage}) by {STALL_MS:.0f} ms/hop"
              f", spare: {spares[victim_name]}")
        h.set_stall(victim_idx, STALL_MS)

        # The node must still be perfectly healthy. If the stall were implemented in a
        # way that made it look broken, the failure path would fire and this gate
        # would credit the straggler mechanism for a plain failover.
        hr = CLIENT.get(f"http://{h.host}:{h.agent_ports[victim_idx]}/health",
                        timeout=10)
        if hr.status_code != 200:
            _fail(f"the slowed node reports unhealthy ({hr.status_code}); this gate "
                  f"would then be testing failure detection, not straggler detection")
        else:
            _ok("the slowed node is still healthy — it is late, not broken")

        rot_before = _view(cp, model_id).get("straggler_rotations") or 0
        rotated_after = None
        slow_text = None
        for i in range(12):
            slow_text, _ = _complete(cp, model_id)
            live = _view(cp, model_id)
            if (live.get("straggler_rotations") or 0) > rot_before:
                rotated_after = i + 1
                break
        live = _view(cp, model_id)
        print(f"  straggler_rotations={live.get('straggler_rotations')} "
              f"failovers={live.get('failovers')} "
              f"serving={live.get('serving_nodes')} "
              f"retired={live.get('retired_nodes')} "
              f"cost_ms={live.get('stage_cost_ms')}")
        if (live.get("straggler_rotations") or 0) <= rot_before:
            _fail("the deployment never rotated away from the straggler; one slow "
                  "machine is still capping the whole pipeline")
        elif victim_name in (live.get("serving_nodes") or []):
            _fail(f"{victim_name} was rotated but is still listed as serving")
        elif victim_name not in (live.get("retired_nodes") or []):
            # Identity matters: a rotation of some *other* stage would satisfy a
            # bare counter check while proving the pool cannot find the real culprit.
            _fail(f"a rotation happened but {victim_name} — the node actually being "
                  f"slowed — is not the one that was retired "
                  f"(retired={live.get('retired_nodes')})")
        else:
            _ok(f"{victim_name} retired after {rotated_after} request(s) without "
                f"anyone being asked to intervene")
        h.set_stall(victim_idx, None)

        # ---- 4: throughput comes back ------------------------------------- #
        print("\n== 4: throughput recovers ==")
        after_text, after_s = _median_time(cp, model_id, 5)
        print(f"  recovered: {after_s*1000:.0f} ms/request "
              f"(healthy baseline {baseline_s*1000:.0f} ms, "
              f"slowed-and-pinned {pinned_s*1000:.0f} ms)")
        if after_s > baseline_s * 1.25:
            _fail(f"after the rotation the deployment is still "
                  f"{after_s/baseline_s:.2f}x baseline; retiring the node did not "
                  f"actually recover throughput")
        else:
            _ok(f"back to {after_s/baseline_s:.2f}x baseline, down from "
                f"{pinned_s/baseline_s:.1f}x when a straggler could not be replaced")

        # ---- 5: correctness survived the rotation ------------------------- #
        print("\n== 5: the answer never changed ==")
        if after_text != baseline_text:
            _fail(f"the text changed across the rotation\n"
                  f"    before={baseline_text!r}\n"
                  f"    after ={after_text!r}")
        elif slow_text != baseline_text:
            _fail("the text changed while the node was merely slow")
        else:
            _ok(f"byte-identical to the healthy baseline ({len(after_text)} chars)")

        # ---- 6: a uniformly slow mesh rotates nothing --------------------- #
        print("\n== 6: a uniformly slow mesh is not a mesh full of stragglers ==")
        before6 = _view(cp, model_id).get("straggler_rotations") or 0
        for idx in range(4):
            h.set_stall(idx, STALL_MS)
        slow_all_text, slow_all_s = _median_time(cp, model_id, 3)
        for _ in range(10):
            _complete(cp, model_id)
        after6 = _view(cp, model_id).get("straggler_rotations") or 0
        print(f"  every node slowed: {slow_all_s*1000:.0f} ms/request, "
              f"rotations {before6} -> {after6}")
        if after6 != before6:
            _fail(f"{after6 - before6} rotation(s) on a uniformly slow mesh. Every "
                  f"replacement is drawn from the same population, so this buys "
                  f"nothing and costs a KV rebuild each time — the sign of an "
                  f"absolute threshold rather than a peer-relative one")
        elif slow_all_text != baseline_text:
            _fail("a uniformly slow mesh changed the answer")
        else:
            _ok("nothing was rotated; slowness is judged against peers, not a clock")
        for idx in range(4):
            h.set_stall(idx, None)

    except SystemExit:
        pass
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        _fail(f"unexpected {type(exc).__name__}: {exc}")
    finally:
        h.stop()

    print()
    if _failures:
        print(f"SOME STRAGGLER CHECKS FAILED ({len(_failures)}):")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("PASS: MeshPool routes around a node that got slow, without losing a token")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
