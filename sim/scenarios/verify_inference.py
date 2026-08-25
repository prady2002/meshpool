"""End-to-end scenario: distributed sharded inference — the flagship (spec §9.6).

Proves the MVP acceptance criterion (e): *query one large model that would not fit
on any single machine, at a measurable speed*. Concretely it verifies:

  1. **Too big for one machine, fits when pooled** — the model's 8 layers are placed
     across >=2 nodes because no single sim node has the memory to hold them all.
  2. **It actually serves** — an interactive prompt returns generated text with a
     real, measured tokens/sec.
  3. **Correctness oracle** — greedy (temperature=0) generation over the *distributed
     pipeline* is token-for-token identical to a *single-process* forward pass of the
     same model. Only activations crossed the network, yet the math is exact.
  4. **Survives a machine dropping** — kill a stage-holding node; the control plane
     re-partitions the model across the survivors and inference continues, producing
     the *same* output (weights are reconstructed identically on every node).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402
from meshpool.inference import model as M  # noqa: E402


def single_machine_greedy(prompt: str, max_new_tokens: int) -> str:
    """Reference: run the whole model in one process, greedy decode."""
    cfg = M.REGISTRY["nano-gpt"]
    P, meta = M.load_params(cfg)
    tok = M.CharTokenizer(itos=(meta or {}).get("itos"), vocab_size=cfg.vocab_size)
    ids = tok.encode(prompt) or [tok.encode(" ")[0]]
    out = M.generate_with(lambda ctx: M.forward_full(P, cfg, ctx), cfg, ids,
                          max_new_tokens=max_new_tokens, temperature=0.0)
    return tok.decode(out)


def node_layer_cap(node: dict, per_layer_mb: int = 1000) -> int:
    cap = node["capability"]
    pol = node["policy"]
    mem = cap["vram_mb"] if cap.get("gpu") and cap.get("vram_mb") else cap["ram_mb"]
    if pol.get("ram_mb"):
        mem = min(mem, pol["ram_mb"])
    return int((mem * 0.8) // per_layer_mb)


def main() -> int:
    h = Harness(n_agents=3, cp_port=8120,
                data_dir=str(Path(__file__).resolve().parent / "_data_infer"))
    ok = True
    try:
        print("== starting cluster ==")
        h.start()
        cp = h.cp_url()
        cfg = M.REGISTRY["nano-gpt"]

        print("\n== test 1: model is too big for any single machine ==")
        nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=10).json()["nodes"]
        caps = {n["name"]: node_layer_cap(n) for n in nodes}
        pooled = sum(caps.values())
        print(f"  per-node layer capacity: {caps}  (pooled={pooled}, model needs {cfg.n_layer})")
        assert max(caps.values()) < cfg.n_layer, "a single node could hold the whole model"
        assert pooled >= cfg.n_layer, "pool cannot hold the model at all"
        print(f"  PASS: no single node fits {cfg.n_layer} layers; pool does")

        print("\n== test 2: deploy (topology-aware sharding across the pool) ==")
        r = httpx.post(f"{cp}/api/v1/models", json={"model": "nano-gpt"}, timeout=60)
        r.raise_for_status()
        view = r.json()
        stages = view["stages"]
        span = {s["node_name"]: s["last"] - s["first"] for s in stages}
        print(f"  deployed across {len(stages)} stages: "
              + ", ".join(f"{s['node_name']}[{s['first']}:{s['last']}]" for s in stages))
        assert view["deployed"] and view["status"] == "ready"
        assert len(stages) >= 2, "model should be sharded across >=2 nodes"
        assert max(span.values()) < cfg.n_layer, "one node ended up with all layers"
        covered = sorted((s["first"], s["last"]) for s in stages)
        assert covered[0][0] == 0 and covered[-1][1] == cfg.n_layer, "layers not fully covered"
        print("  PASS: 8 layers sharded into contiguous pipeline stages, fully covered")

        print("\n== test 3: interactive inference returns text at measurable speed ==")
        prompt = "MeshPool turns idle machines into "
        r = httpx.post(f"{cp}/api/v1/inference/chat",
                       json={"prompt": prompt, "max_new_tokens": 24, "temperature": 0.8},
                       timeout=60)
        r.raise_for_status()
        chat = r.json()
        print(f"  prompt:   {prompt!r}")
        print(f"  response: {chat['text']!r}")
        print(f"  {chat['tokens']} tokens · {chat['tokens_per_sec']} tok/s "
              f"across {len(chat['stages'])} pipeline stages")
        assert chat["tokens"] == 24 and chat["tokens_per_sec"] > 0
        assert len(chat["stages"]) == len(stages)
        print("  PASS: served generated text with a real tokens/sec figure")

        print("\n== test 4: correctness oracle — distributed == single-machine ==")
        oracle = single_machine_greedy(prompt, 24)
        r = httpx.post(f"{cp}/api/v1/inference/chat",
                       json={"prompt": prompt, "max_new_tokens": 24, "temperature": 0.0},
                       timeout=60)
        dist = r.json()["text"]
        print(f"  single-machine greedy: {oracle!r}")
        print(f"  distributed   greedy: {dist!r}")
        assert dist == oracle, "distributed pipeline diverged from single-machine result!"
        print("  PASS: distributed pipeline is bit-for-bit identical to single-machine")

        print("\n== test 5: kill a stage node -> auto re-partition, same output ==")
        # kill the smallest node (sim-node-3) so survivors still have capacity
        print("  killing sim-node-3 (holds the last pipeline stage) ...")
        h.stop_agent(2)
        deadline = time.time() + 30
        repart = None
        while time.time() < deadline:
            v = httpx.get(f"{cp}/api/v1/models", timeout=10).json()
            names = {s["node_name"] for s in v.get("stages", [])}
            if v.get("status") == "ready" and "sim-node-3" not in names:
                repart = v
                break
            time.sleep(1)
        assert repart is not None, "model did not re-partition onto survivors in time"
        cov = sorted((s["first"], s["last"]) for s in repart["stages"])
        print("  re-partitioned across: "
              + ", ".join(f"{s['node_name']}[{s['first']}:{s['last']}]" for s in repart["stages"]))
        assert cov[0][0] == 0 and cov[-1][1] == cfg.n_layer, "layers not fully covered after re-partition"
        assert "sim-node-3" not in {s["node_name"] for s in repart["stages"]}

        r = httpx.post(f"{cp}/api/v1/inference/chat",
                       json={"prompt": prompt, "max_new_tokens": 24, "temperature": 0.0},
                       timeout=60)
        after = r.json()["text"]
        print(f"  output after failover: {after!r}")
        assert after == oracle, "output changed after re-partition (should be identical)"
        print("  PASS: survived node loss, re-sharded on survivors, identical output")

        print("\nALL INFERENCE E2E CHECKS PASSED")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        h.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
