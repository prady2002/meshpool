"""End-to-end scenario: serve a model whose WEIGHTS live in the pool (spec §0, §9.6).

The flagship inference scenario (``verify_inference``) shards a model across the pool
but assumes every node already has the checkpoint on disk. That breaks the real
"too big for one machine" story: a user bringing their own trained model to a fresh
set of donor nodes has no way to get multi-GB weights onto them.

This scenario proves the fix — weight distribution *through the pool*:

  1. **Bring your own weights** — upload a trained checkpoint (.npz) to the control
     plane. It is staged in the pooled, content-addressed blob store (chunked +
     replicated across donor nodes) and hidden from the user's Files listing.
  2. **No node — nor the control plane — needs the whole checkpoint** — the checkpoint
     is split into per-layer group blobs referenced by a small manifest; each stage node
     is shipped only the blob ids of the layers it hosts and assembles its slice from
     them, so the coordinator never holds more than one layer-group at a time.
  3. **Correctness oracle** — greedy generation over the *distributed* pipeline is
     token-for-token identical to a single-process forward pass of the *pooled*
     checkpoint, and DIFFERENT from the node-local built-in checkpoint. That gap is
     the proof the served weights genuinely came from the pool, not from local disk.
  4. **Survives a machine dropping** — kill a stage node; the model re-partitions onto
     the survivors, re-hydrates their slices from the pool, and serves the same output.
"""
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402
from meshpool.inference import model as M  # noqa: E402


def greedy(P: dict, cfg, itos, prompt: str, n: int) -> str:
    """Single-process greedy (temperature=0 => argmax) decode of checkpoint ``P``."""
    tok = M.CharTokenizer(itos=itos, vocab_size=cfg.vocab_size)
    ids = tok.encode(prompt) or [tok.encode(" ")[0]]
    out = M.generate_with(lambda ctx: M.forward_full(P, cfg, ctx), cfg, ids,
                          max_new_tokens=n, temperature=0.0)
    return tok.decode(out)


def local_greedy(cfg, prompt: str, n: int) -> str:
    """Greedy decode of the node-local built-in checkpoint (weights/nano-gpt.npz)."""
    P, meta = M.load_params(cfg)
    return greedy(P, cfg, (meta or {}).get("itos"), prompt, n)


def user_blobs(cp: str) -> list:
    return httpx.get(f"{cp}/api/v1/blobs", timeout=10).json()["blobs"]


def chat(cp: str, mid: str, prompt: str, n: int, temperature: float) -> dict:
    r = httpx.post(f"{cp}/api/v1/inference/chat",
                   json={"prompt": prompt, "max_new_tokens": n,
                         "temperature": temperature, "model_id": mid},
                   timeout=120)
    r.raise_for_status()
    return r.json()


def main() -> int:
    h = Harness(n_agents=3, cp_port=8130,
                data_dir=str(Path(__file__).resolve().parent / "_data_pooled_model"))
    ok = True
    try:
        print("== starting cluster ==")
        h.start()
        cp = h.cp_url()
        cfg = M.REGISTRY["nano-gpt"]

        # A distinct checkpoint (same architecture, different weights) so we can prove
        # the served output comes from THESE pooled weights and not the local file.
        pooled_cfg = dataclasses.replace(cfg, seed=4242)
        pooled_params = M.random_params(pooled_cfg)
        itos = M.default_charset(cfg.vocab_size)
        raw = M.dumps_npz(pooled_params, itos=itos)
        print(f"  built a distinct checkpoint: {len(pooled_params)} tensors, "
              f"{len(raw)} bytes serialized")

        prompt = "MeshPool turns idle machines into "
        n_tok = 24
        pooled_oracle = greedy(pooled_params, cfg, itos, prompt, n_tok)
        builtin_oracle = local_greedy(cfg, prompt, n_tok)
        assert pooled_oracle != builtin_oracle, (
            "test integrity: pooled and built-in checkpoints must differ")

        print("\n== test 1: upload weights -> staged in the pool, sharded, hidden ==")
        before = len(user_blobs(cp))
        r = httpx.post(f"{cp}/api/v1/models/upload?model=nano-gpt", content=raw, timeout=180)
        r.raise_for_status()
        view = r.json()
        mid = view["id"]
        stages = view["stages"]
        print(f"  deployed {mid} across {len(stages)} stages: "
              + ", ".join(f"{s['node_name']}[{s['first']}:{s['last']}]" for s in stages))
        assert view["deployed"] and view["status"] == "ready", "pooled model not ready"
        assert view.get("pooled") is True, "deployment is not marked pool-backed"
        assert view.get("weight_groups") == cfg.n_layer + 3, (
            "checkpoint was not split into per-layer group blobs "
            f"(got {view.get('weight_groups')}, want {cfg.n_layer + 3})")
        assert len(stages) >= 2, "weights should be sharded across >=2 nodes"
        covered = sorted((s["first"], s["last"]) for s in stages)
        assert covered[0][0] == 0 and covered[-1][1] == cfg.n_layer, "layers not fully covered"
        blobs = user_blobs(cp)
        assert not any("weights" in b.get("name", "") for b in blobs), "raw weights leaked to Files"
        assert len(blobs) == before, "pooled weights should be hidden (internal), not user files"
        print(f"  PASS: weights pooled + sharded across {len(stages)} nodes, hidden from Files")

        print("\n== test 2: correctness oracle — distributed == POOLED, != built-in ==")
        dist = chat(cp, mid, prompt, n_tok, 0.0)["text"]
        print(f"  pooled-checkpoint greedy : {pooled_oracle!r}")
        print(f"  distributed      greedy  : {dist!r}")
        print(f"  built-in (local) greedy  : {builtin_oracle!r}")
        assert dist == pooled_oracle, "distributed output != the pooled checkpoint!"
        assert dist != builtin_oracle, "served the LOCAL weights, not the pooled ones!"
        print("  PASS: served the pool-hydrated weights, bit-identical to single-machine")

        print("\n== test 3: interactive inference at measurable speed ==")
        c = chat(cp, mid, prompt, n_tok, 0.8)
        print(f"  response: {c['text']!r}")
        print(f"  {c['tokens']} tokens · {c['tokens_per_sec']} tok/s "
              f"across {len(c['stages'])} pipeline stages")
        assert c["tokens"] == n_tok and c["tokens_per_sec"] > 0
        print("  PASS: served generated text with a real tokens/sec figure")

        print("\n== test 4: kill a stage node -> re-partition re-hydrates from the pool ==")
        print("  killing sim-node-3 ...")
        h.stop_agent(2)
        deadline = time.time() + 40
        repart = None
        while time.time() < deadline:
            v = httpx.get(f"{cp}/api/v1/models", timeout=10).json()
            names = {s["node_name"] for s in v.get("stages", [])}
            if v.get("status") == "ready" and "sim-node-3" not in names:
                repart = v
                break
            time.sleep(0.5)
        assert repart is not None, "model did not re-partition after node loss"
        surv = [s["node_name"] for s in repart["stages"]]
        print(f"  re-partitioned onto survivors: {surv}")
        assert repart.get("weight_groups") == cfg.n_layer + 3, "manifest not reused on re-partition"
        dist2 = chat(cp, mid, prompt, n_tok, 0.0)["text"]
        assert dist2 == pooled_oracle, "post-churn output diverged from the pooled checkpoint!"
        print("  PASS: survivors re-pulled only their layer blobs from the pool; same output")

        print("\n== test 5: `meshpool models upload` CLI streams weights group-by-group ==")
        import os
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tf:
            tf.write(raw)
            ckpt_path = tf.name
        try:
            cli_env = {**os.environ, "MESHPOOL_CONTROL_PLANE_URL": cp, "COLUMNS": "200"}
            proc = subprocess.run(
                [sys.executable, "-m", "meshpool.cli.main", "models", "upload",
                 ckpt_path, "--model", "nano-gpt"],
                cwd=str(Path(__file__).resolve().parents[2]),
                env=cli_env, capture_output=True, text=True, errors="replace", timeout=180)
        finally:
            try:
                os.unlink(ckpt_path)
            except OSError:
                pass
        cli_out = proc.stdout + proc.stderr
        assert proc.returncode == 0, f"`models upload` failed: {cli_out}"
        assert "deployed" in cli_out and "your weights" in cli_out, cli_out
        assert "streamed" in cli_out, "CLI did not use the client-side streaming upload path"
        cur = httpx.get(f"{cp}/api/v1/models", timeout=10).json()
        assert cur.get("pooled") is True, "CLI-uploaded model is not pool-backed"
        assert cur.get("weight_groups") == cfg.n_layer + 3, "CLI-streamed model lost its per-layer split"
        cli_dist = chat(cp, cur["id"], prompt, n_tok, 0.0)["text"]
        assert cli_dist == pooled_oracle, "CLI-served output diverged from the pooled checkpoint!"
        print(f"  PASS: CLI split the checkpoint client-side + streamed it group-by-group; "
              f"deployed across {len(cur['stages'])} stage(s); output matches the oracle")

        print("\n== test 6: control plane carried ZERO weight bytes — nodes pulled peer-to-peer ==")
        summary = httpx.get(f"{cp}/api/v1/cluster", timeout=10).json()["summary"]
        pulls = summary.get("weight_group_pulls", 0)
        pushes = summary.get("weight_group_pushes", 0)
        cached = summary.get("weight_group_cached", 0)
        print(f"  weight-group placements so far: {pulls} peer-to-peer pull(s), "
              f"{pushes} control-plane relay(s), {cached} already-resident")
        # Non-vacuity: the cluster starts empty (the harness wipes its data dir), so a
        # run in which nothing at all was placed means the distribution path never ran
        # and this gate would be measuring nothing.
        assert pulls + pushes >= 1, (
            "no weight group was distributed at all — the gate measured nothing "
            "(stale node state from a previous run?)")
        assert pulls >= 1, (
            "no peer-to-peer weight pull happened — the control plane still relayed the bytes")
        assert pushes == 0, (
            f"control plane relayed {pushes} weight group(s) on a LAN pool; "
            "direct peer-to-peer pull should carry them all (byte-free coordinator)")
        print("  PASS: every weight group reached its node by a direct peer pull; "
              "the coordinator moved only ids + peer URLs, never the weights")

        print("\nALL POOLED-MODEL WEIGHT-DISTRIBUTION E2E CHECKS PASSED")
    except AssertionError as exc:
        ok = False
        print(f"\nFAILED: {exc}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nERROR: {type(exc).__name__}: {exc}")
    finally:
        h.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
