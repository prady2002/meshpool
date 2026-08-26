"""Gate: paged KV cache and client activation cache across OS processes.

Honest scope (measured):
  * real GGUF container written by llama.cpp's reference writer;
  * two long-lived worker interpreters, each owning only its layer range;
  * verifies cached-vs-uncached sharding equivalence and activation-cache
    hydration. Independent MoE/attention numerical oracle properties live in
    tests/test_kvcache.py and tests/test_moe_sharding.py.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from meshpool.inference import gguf_loader as G  # noqa: E402
from meshpool.inference import kvcache as KV  # noqa: E402


WORKER = r'''
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from meshpool.inference import gguf_loader as G
from meshpool.inference.pipeline import CachedShard

path, start, end = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
is_first, is_last = sys.argv[5] == "1", sys.argv[6] == "1"
info = G.read_info(path)
params = G.load_params(path, start, end, is_first, is_last, info=info)
shard = CachedShard(info.config, start, end, is_first, is_last, params=params, cache_page_size=3)

for line in sys.stdin:
    req = json.loads(line)
    op = req.get("op")
    if op == "drop":
        shard.drop(req["session"])
        print(json.dumps({"ok": True}), flush=True)
        continue
    if op == "stats":
        stats = {}
        for sid, cache in shard._caches.items():
            stats[sid] = {
                "length": cache.length,
                "layers": sorted(cache.layers),
                "pages": {str(i): cache.layer(i).page_count for i in sorted(cache.layers)},
            }
        print(json.dumps({"stats": stats}), flush=True)
        continue
    data = req["data"] if req["is_tokens"] else np.array(req["data"], dtype=np.float32)
    out = shard.forward(data, bool(req["is_tokens"]), int(req.get("pos_offset", 0)), req.get("session"))
    print(json.dumps({"data": np.asarray(out, dtype=np.float32).tolist()}), flush=True)
'''


def _call(proc: subprocess.Popen, req: dict) -> dict:
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("worker exited without a response")
    return json.loads(line)


def _forward(workers, x, is_tokens: bool, pos: int, session: str | None,
             act: KV.ActivationCache | None = None) -> np.ndarray:
    data: object = x
    token_flag = is_tokens
    for idx, w in enumerate(workers):
        r = _call(w, {
            "op": "forward",
            "data": data if token_flag else np.asarray(data, dtype=np.float32).tolist(),
            "is_tokens": token_flag,
            "pos_offset": pos,
            "session": session,
        })
        data = np.asarray(r["data"], dtype=np.float32)
        if act is not None and session is not None and idx < len(workers) - 1:
            act.record(session, idx + 1, pos, data)
        token_flag = False
    return np.asarray(data, dtype=np.float32)


def _generate_uncached(workers, prompt: list[int], n: int) -> tuple[list[int], float, int]:
    ctx = list(prompt)
    out = []
    positions = 0
    t0 = time.perf_counter()
    for _ in range(n):
        logits = _forward(workers, ctx, True, 0, None)
        positions += len(ctx) * len(workers)
        nxt = int(logits[-1].argmax())
        out.append(nxt)
        ctx.append(nxt)
    return out, time.perf_counter() - t0, positions


def _generate_cached(workers, prompt: list[int], n: int,
                     act: KV.ActivationCache,
                     session: str = "cached") -> tuple[list[int], float, int, int]:
    t0 = time.perf_counter()
    logits = _forward(workers, prompt, True, 0, session, act)
    positions = len(prompt) * len(workers)
    pos = len(prompt)
    out = []
    for i in range(n):
        nxt = int(logits[-1].argmax())
        out.append(nxt)
        if i != n - 1:
            logits = _forward(workers, [nxt], True, pos, session, act)
            positions += len(workers)
            pos += 1
    return out, time.perf_counter() - t0, positions, pos


def _stats(workers) -> list[dict]:
    return [_call(w, {"op": "stats"})["stats"] for w in workers]


def main() -> int:  # noqa: C901
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix="meshpool_kv_gate_"))
    workers: list[subprocess.Popen] = []
    try:
        print("== honest scope ==")
        print("  real GGUF fixture; two persistent OS worker processes; page size=3")
        print("  validates cache/sharding equivalence and activation-cache hydration")

        print("\n== test 1: build real GGUF and start layer-range workers ==")
        from gguf_fixtures import build_llama_gguf
        path = str(tmp / "kv.gguf")
        build_llama_gguf(path, n_layer=4, d_model=32, n_head=4, n_kv_head=2,
                         d_ff=64, vocab_size=128, seed=20240820)
        info = G.read_info(path)
        worker_py = tmp / "worker.py"
        worker_py.write_text(WORKER, encoding="utf-8")
        for start, end, first, last in [(0, 2, True, False), (2, 4, False, True)]:
            workers.append(subprocess.Popen(
                [sys.executable, str(worker_py), str(ROOT), path, str(start), str(end),
                 "1" if first else "0", "1" if last else "0"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, errors="replace"))
        print(f"  checkpoint layers={info.config.n_layer}, worker spans=[0:2), [2:4)")
        print("  PASS: workers are separate processes holding only their stage ranges")

        print("\n== test 2: cached generation equals uncached generation across a shard boundary ==")
        prompt = [1, 5, 9, 13, 17, 21, 25]
        n = 6
        uncached, uncached_s, uncached_pos = _generate_uncached(workers, prompt, n)
        act = KV.ActivationCache(page_size=3)
        cached, cached_s, cached_pos, cached_prefix_len = _generate_cached(workers, prompt, n, act)
        print(f"  uncached={uncached} ({uncached_s:.3f}s, positions*stages={uncached_pos})")
        print(f"  cached  ={cached} ({cached_s:.3f}s, positions*stages={cached_pos})")
        assert cached == uncached
        assert len(set(cached)) > 1, f"degenerate output {cached}; equality proves nothing"
        assert cached_pos < uncached_pos
        print("  PASS: cached and uncached outputs match; measured work drops")

        print("\n== test 3: paged server-side KV cache is per-stage and crosses page boundaries ==")
        stats = _stats(workers)
        expected_len = cached_prefix_len
        for idx, st in enumerate(stats):
            s = st["cached"]
            print(f"  worker {idx}: length={s['length']} layers={s['layers']} pages={s['pages']}")
            assert s["length"] == expected_len
            assert s["layers"] == ([0, 1] if idx == 0 else [2, 3])
            assert all(int(p) >= 4 for p in s["pages"].values())
        print("  PASS: each worker caches only owned layers in multiple pages")

        print("\n== test 4: client activation cache hydrates a failed downstream stage ==")
        assert act.length("cached", 1) == expected_len
        _call(workers[1], {"op": "drop", "session": "cached"})
        assert _stats(workers)[1].get("cached", {"length": 0})["length"] == 0
        boundary = act.get("cached", 1, 0, expected_len)
        _call(workers[1], {
            "op": "forward", "data": boundary.tolist(), "is_tokens": False,
            "pos_offset": 0, "session": "cached",
        })
        assert _stats(workers)[1]["cached"]["length"] == expected_len
        ctx = prompt + cached
        uncached_next = _forward(workers, ctx, True, 0, None)[-1]
        hydrated_next = _forward(workers, [cached[-1]], True, expected_len, "cached", act)[-1]
        delta = float(np.max(np.abs(uncached_next - hydrated_next)))
        print(f"  post-hydration next-logit max|delta|={delta:.9g}")
        assert delta < 1e-6
        print("  PASS: recovery replays cached activations instead of restarting the pipeline")

        print("\n== test 5: measured cached-vs-uncached work at several lengths ==")
        for i, prompt2 in enumerate(([1, 5, 9], [1, 5, 9, 13, 17, 21, 25],
                                     [3, 11, 27, 5, 19, 23, 29, 31, 37, 41, 43])):
            u, us, upos = _generate_uncached(workers, list(prompt2), 4)
            c_act = KV.ActivationCache(page_size=3)
            c, cs, cpos, _ = _generate_cached(workers, list(prompt2), 4, c_act, session=f"scale{i}")
            assert c == u
            print(f"  len={len(prompt2):2d}: uncached {upos:3d} positions*stages/{us:.3f}s; "
                  f"cached {cpos:3d} positions*stages/{cs:.3f}s")
            assert cpos < upos
        print("  PASS: measured work grows with new tokens instead of full context replay")

        print("\nALL KV CACHE CHECKS PASSED")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        for w in workers:
            if w.poll() is None:
                w.terminate()
                try:
                    w.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    w.kill()
                    w.wait(timeout=5)
        shutil.rmtree(tmp)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
