"""Gate: a real checkpoint file, split by layer range across separate OS processes.

``verify_first_real_token.py`` proves the *execution path* -- real architecture,
real tokenizer, three real processes -- with weights initialized locally. This
gate closes the remaining gap: the weights come out of a real GGUF checkpoint on
disk, and each worker process loads **only its own layer range** out of it.

That last property is the entire flagship claim. If every node must read the
whole file, the pool can only run models that already fit on one machine, and
MeshPool is just a load balancer. So this gate does not merely check that the
output is right -- it checks that a stage's output does not depend on the bytes
of the layers it was not assigned, by corrupting those bytes and demanding the
identical result.

Honest scope (plan §5.2, tier: measured):
  * real GGUF container written by llama.cpp's own reference writer;
  * real per-layer-range loading, real RoPE un-permutation, real quantization;
  * weights are deterministically generated, because model downloads are blocked
    in this environment. This validates the format, the sharding and the
    arithmetic -- not pretrained model quality.
"""
from __future__ import annotations

import json
import os
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
from meshpool.inference import model as M  # noqa: E402

N_LAYER, D_MODEL, N_HEAD, KV_HEAD, D_FF, VOCAB = 6, 32, 4, 2, 64, 64
SPANS = [(0, 2), (2, 4), (4, 6)]
PROMPT = [3, 11, 27, 5, 19]
N_TOKENS = 6

# Runs in a *separate* interpreter: loads one layer range and computes one hop.
WORKER = r'''
import json, os, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from meshpool.inference import gguf_loader as G
from meshpool.inference import model as M

req = json.loads(sys.stdin.read())
params = G.load_params(req["path"], req["start"], req["end"],
                       req["is_first"], req["is_last"])
cfg = G.cached_info(req["path"]).config
shard = M.ShardModule(cfg, req["start"], req["end"],
                      req["is_first"], req["is_last"], params=params)
x = req["x"] if req["is_first"] else np.array(req["x"], dtype=np.float32)
out = shard.forward(x, is_tokens=req["is_first"])
json.dump({"pid": os.getpid(),
           "n_params": len(params),
           "param_names": sorted(params),
           "out": np.asarray(out, dtype=np.float32).tolist()}, sys.stdout)
'''


def _run_stage(worker_py: str, path: str, span, is_first, is_last, x) -> dict:
    p = subprocess.run(
        [sys.executable, worker_py, str(ROOT)],
        input=json.dumps({"path": path, "start": span[0], "end": span[1],
                          "is_first": is_first, "is_last": is_last,
                          "x": np.asarray(x).tolist()}),
        capture_output=True, text=True, errors="replace", timeout=180)
    if p.returncode != 0:
        raise RuntimeError(f"stage [{span[0]}:{span[1]}] failed: {p.stderr[-600:]}")
    return json.loads(p.stdout)


def _pipeline(worker_py: str, path: str, tokens: list[int]) -> tuple[np.ndarray, list[int], list]:
    x: object = tokens
    pids, reports = [], []
    for i, span in enumerate(SPANS):
        r = _run_stage(worker_py, path, span, i == 0, i == len(SPANS) - 1, x)
        x = np.array(r["out"], dtype=np.float32)
        pids.append(r["pid"])
        reports.append(r)
    return np.asarray(x, dtype=np.float32), pids, reports


def _greedy(worker_py: str, path: str, prompt: list[int], n: int) -> list[int]:
    ctx = list(prompt)
    out = []
    for _ in range(n):
        logits, _, _ = _pipeline(worker_py, path, ctx)
        nxt = int(np.asarray(logits)[-1].argmax())
        out.append(nxt)
        ctx.append(nxt)
    return out


def _corrupt_layers_outside(path: str, dst: str, keep_start: int, keep_end: int) -> int:
    """Copy the checkpoint, overwriting tensor bytes of every other layer."""
    import gguf
    data = bytearray(Path(path).read_bytes())
    reader = gguf.GGUFReader(path, "r")
    rng = np.random.default_rng(7)
    touched = 0
    for t in reader.tensors:
        name = str(t.name)
        if not name.startswith("blk."):
            continue
        idx = int(name.split(".")[1])
        if keep_start <= idx < keep_end:
            continue
        off = int(t.data.__array_interface__["data"][0]
                  - reader.data.__array_interface__["data"][0])
        nbytes = int(t.data.nbytes)
        data[off:off + nbytes] = bytes(rng.integers(0, 256, nbytes, dtype=np.uint8))
        touched += nbytes
    Path(dst).write_bytes(bytes(data))
    return touched


def main() -> int:  # noqa: C901
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix="meshpool_gguf_gate_"))
    try:
        print("== honest scope ==")
        print("  real GGUF container + real per-layer-range loading + separate OS processes")
        print("  weights are deterministically generated (downloads blocked here)")
        print("  this validates format, sharding and arithmetic -- not model quality")

        print("\n== test 1: write a real GGUF checkpoint ==")
        from gguf_fixtures import build_llama_gguf
        path = str(tmp / "gate.gguf")
        build_llama_gguf(path, n_layer=N_LAYER, d_model=D_MODEL, n_head=N_HEAD,
                         n_kv_head=KV_HEAD, d_ff=D_FF, vocab_size=VOCAB)
        info = G.cached_info(path)
        size = os.path.getsize(path)
        print(f"  {G.describe(info)}")
        print(f"  file={size} bytes on disk, layer_bytes={info.layer_bytes}")
        assert info.config.n_layer == N_LAYER
        assert sum(info.layer_bytes) > 0
        print("  PASS: checkpoint is a real GGUF the reference reader accepts")

        print("\n== test 2: each stage loads ONLY its own layer range ==")
        worker_py = str(tmp / "stage_worker.py")
        Path(worker_py).write_text(WORKER, encoding="utf-8")
        for i, (a, b) in enumerate(SPANS):
            p = G.load_params(path, a, b, i == 0, i == len(SPANS) - 1)
            owned = {int(k.split(".")[0][1:]) for k in p if k.startswith("h")}
            assert owned == set(range(a, b)), f"stage got layers {sorted(owned)}, wanted [{a},{b})"
            span_mb = G.layer_span_bytes(info, a, b) / 1e6
            print(f"  stage {i} [{a}:{b}] -> {len(p)} tensors, layers {sorted(owned)}, "
                  f"{span_mb:.4f} MB of {info.total_bytes / 1e6:.4f} MB total")
        assert sum(G.layer_span_bytes(info, a, b) for a, b in SPANS) < info.total_bytes
        print("  PASS: no stage holds another stage's layers")

        print("\n== test 3: sharded across 3 OS processes == single process, bit-exact ==")
        params = G.load_full_params(path)
        cfg = info.config
        t0 = time.perf_counter()
        single = M.generate_with(lambda ctx: M.forward_full(params, cfg, ctx), cfg,
                                 PROMPT, max_new_tokens=N_TOKENS, temperature=0.0)
        single_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        sharded = _greedy(worker_py, path, PROMPT, N_TOKENS)
        sharded_s = time.perf_counter() - t0
        _, pids, reports = _pipeline(worker_py, path, PROMPT)
        print(f"  single-process : {single}   ({single_s:.3f}s)")
        print(f"  3-process shard: {sharded}   ({sharded_s:.3f}s)")
        print(f"  distinct worker PIDs: {sorted(set(pids))} (this process: {os.getpid()})")
        assert len(set(pids)) == 3, f"expected 3 separate processes, saw {set(pids)}"
        assert os.getpid() not in pids, "a stage ran in the driver process"
        assert sharded == single, f"sharded {sharded} != single-process {single}"
        print("  PASS: identical tokens from genuinely separate processes")

        print("\n== test 4: the equality is not vacuous ==")
        assert len(set(single)) > 1, f"degenerate output {single}; equality proves nothing"
        wrong = _run_stage(worker_py, path, (1, 3), True, False, PROMPT)
        right = _run_stage(worker_py, path, (0, 2), True, False, PROMPT)
        assert not np.allclose(np.array(wrong["out"]), np.array(right["out"])), (
            "loading a different layer range produced identical activations; "
            "the loader is probably ignoring the requested range")
        print(f"  layers[0:2] vs layers[1:3] activations differ: "
              f"max|delta|={np.abs(np.array(wrong['out']) - np.array(right['out'])).max():.6f}")
        print(f"  output has {len(set(single))} distinct tokens, not a constant")
        print("  PASS: the test can fail")

        print("\n== test 5: a stage does not depend on other layers' bytes ==")
        print("  (this is the claim that lets a pool run a model no member could hold)")
        for i, (a, b) in enumerate(SPANS):
            dst = str(tmp / f"corrupt_{i}.gguf")
            n = _corrupt_layers_outside(path, dst, a, b)
            pristine = G.load_params(path, a, b, i == 0, i == len(SPANS) - 1)
            damaged = G.load_params(dst, a, b, i == 0, i == len(SPANS) - 1)
            assert set(pristine) == set(damaged)
            for k in pristine:
                assert np.array_equal(pristine[k], damaged[k]), (
                    f"stage [{a}:{b}] parameter {k!r} changed when *other* layers' "
                    f"bytes were overwritten -- it is reading the whole file")
            print(f"  stage {i} [{a}:{b}]: identical after overwriting {n} bytes "
                  f"of other layers")
        print("  PASS: each stage's weights are independent of the rest of the file")

        print("\n== test 6: the corruption check itself is not vacuous ==")
        dst = str(tmp / "corrupt_self.gguf")
        _corrupt_layers_outside(path, dst, 99, 99)          # corrupt every layer
        base = G.load_params(path, 0, 2, True, False)
        broken = G.load_params(dst, 0, 2, True, False)
        changed = [k for k in base if not np.array_equal(base[k], broken[k])]
        assert changed, "corrupting ALL layers changed nothing; the corruption is a no-op"
        print(f"  corrupting every layer changed {len(changed)}/{len(base)} tensors "
              f"(e.g. {changed[0]})")
        print("  PASS: test 5 would have caught a whole-file read")

        print("\nALL GGUF SHARDED-INFERENCE CHECKS PASSED")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
