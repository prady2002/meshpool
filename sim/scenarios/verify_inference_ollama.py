"""End-to-end scenario: real-LLM inference routed to pooled Ollama nodes (P1a).

This is the honest, real-model counterpart to ``verify_inference`` (which proves the
from-scratch NumPy net is sharded and bit-identical). Here the pool serves a *genuine*
open-weight model (e.g. ``qwen2.5-coder``) by routing whole requests to the nodes that
already run it under Ollama. It verifies:

  1. **Detection & catalog** — agents auto-detect a local Ollama server and advertise
     its models; the control plane exposes them in the deploy catalog.
  2. **Deploy = replicate** — deploying a real model places one *full-model replica*
     per capable node (no layer sharding, no weight shipping).
  3. **It actually serves** — a prompt returns real generated text at a measured
     tokens/sec (Ollama's own eval timing, not an estimate).
  4. **Concurrency spreads across replicas** — many simultaneous requests are load-
     balanced across every node holding the model (the aggregate-throughput win).
  5. **Failover** — kill a replica node mid-pool; requests keep succeeding on the
     survivor.

Physics note (printed): co-located sim nodes on ONE machine share ONE Ollama server,
so this run proves *routing/failover*, not a throughput speedup. The real aggregate-
throughput gain needs separate machines (each with its own Ollama) — see
``meshpool.inference.bench`` and the 2-machine runbook.

If no Ollama is reachable (e.g. CI), the scenario prints SKIP and exits 0.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402


def _pick_model(available: dict) -> Optional[str]:
    """Choose a small/fast model for the automated check, if any are advertised."""
    models = [m["model"] for m in available.get("ollama", [])]
    if not models:
        return None
    for pref in ("0.5b", "1.5b", "1b", "3b", "7b", "mini", "small"):
        for m in models:
            if pref in m.lower():
                return m
    return models[0]


def _chat(cp: str, prompt: str, max_new_tokens: int, temperature: float = 0.7,
          timeout: float = 600.0) -> dict:
    r = httpx.post(f"{cp}/api/v1/inference/chat",
                   json={"prompt": prompt, "max_new_tokens": max_new_tokens,
                         "temperature": temperature},
                   timeout=timeout)
    r.raise_for_status()
    return r.json()


def main() -> int:
    h = Harness(n_agents=2, cp_port=8123,
                data_dir=str(Path(__file__).resolve().parent / "_data_ollama"))
    ok = True
    try:
        print("== starting cluster (2 agents, real Ollama detection) ==")
        h.start()
        cp = h.cp_url()

        print("\n== test 1: agents detect local Ollama; catalog lists its models ==")
        available = httpx.get(f"{cp}/api/v1/models/available", timeout=10).json()
        model = _pick_model(available)
        if model is None:
            print("  no Ollama server / models detected in the pool")
            print("  SKIP: install Ollama and `ollama pull qwen2.5-coder:7b` to run this "
                  "real-model check (routing/failover logic is covered by "
                  "tests/test_inference_ollama.py regardless)")
            return 0
        hosts = next((m["nodes"] for m in available["ollama"] if m["model"] == model), [])
        print(f"  pool advertises {len(available['ollama'])} Ollama model(s); "
              f"using {model!r} (on nodes: {hosts})")
        assert hosts, "picked a model no node advertises"
        print("  PASS: real model discovered from pooled Ollama libraries")

        print("\n== test 2: deploy places one full-model replica per capable node ==")
        r = httpx.post(f"{cp}/api/v1/models", json={"model": model}, timeout=120)
        r.raise_for_status()
        view = r.json()
        print(f"  backend={view['backend']}  status={view['status']}  "
              f"replicas={view['replicas']}  nodes="
              + ", ".join(s["node_name"] for s in view["stages"]))
        assert view["backend"] == "ollama", "real model should use the ollama backend"
        assert view["deployed"] and view["status"] == "ready"
        assert view["replicas"] >= 1
        print("  PASS: deployed as whole-model replicas (no layer sharding)")

        print("\n== test 3: real generation returns text at a measured tok/s ==")
        prompt = "Write a one-line Python function that returns the square of n:"
        t0 = time.perf_counter()
        chat = _chat(cp, prompt, max_new_tokens=32)
        dt = time.perf_counter() - t0
        print(f"  prompt:   {prompt!r}")
        snippet = chat["text"].strip().replace("\n", " ")[:80]
        print(f"  response: {snippet!r}...")
        print(f"  served_by={chat.get('served_by')}  {chat['tokens']} tokens · "
              f"{chat['tokens_per_sec']} tok/s  (round-trip {dt:.1f}s)")
        assert chat["tokens"] > 0 and chat["tokens_per_sec"] > 0, "no real generation happened"
        assert chat["backend"] == "ollama"
        print("  PASS: real model produced text with a measured tokens/sec")

        print("\n== test 4: concurrent requests spread across replicas ==")
        n_req = 4
        prompts = [f"In one short sentence, describe idea #{i}:" for i in range(n_req)]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_req) as pool:
            results = list(pool.map(lambda p: _chat(cp, p, max_new_tokens=16), prompts))
        wall = max(1e-6, time.perf_counter() - t0)
        total_tokens = sum(r["tokens"] for r in results)
        served = [r.get("served_by") for r in results]
        print(f"  {n_req} concurrent requests served by: {served}")
        print(f"  aggregate: {total_tokens} tokens in {wall:.1f}s "
              f"= {total_tokens / wall:.1f} tok/s across the pool")
        assert all(r["tokens"] > 0 for r in results), "a concurrent request produced nothing"
        replicas_up = view["replicas"]
        if replicas_up >= 2:
            assert len(set(served)) >= 2, "requests did not spread across replicas"
            print("  PASS: load-balanced across multiple replicas (round-robin)")
        else:
            print("  PASS: all concurrent requests succeeded (single replica in this run)")
        print("  NOTE: co-located sim nodes share ONE Ollama here, so this shows routing, "
              "not a speedup. Real throughput gain needs separate machines "
              "(meshpool.inference.bench).")

        print("\n== test 5: kill a replica node -> requests still served ==")
        if view["replicas"] >= 2:
            print("  killing sim-node-2 ...")
            h.stop_agent(1)
            # let the reaper mark it offline
            deadline = time.time() + 20
            while time.time() < deadline:
                v = httpx.get(f"{cp}/api/v1/models", timeout=10).json()
                names = {s["node_name"] for s in v.get("stages", [])}
                if "sim-node-2" not in names or v.get("status") == "ready":
                    break
                time.sleep(1)
            after = _chat(cp, "Say hello in one word:", max_new_tokens=8)
            print(f"  after failover: served_by={after.get('served_by')} "
                  f"{after['tokens']} tokens · {after['tokens_per_sec']} tok/s")
            assert after["tokens"] > 0, "no generation after a replica was killed"
            assert after.get("served_by") != "sim-node-2"
            print("  PASS: survived a replica dropping; survivor kept serving")
        else:
            print("  SKIP: only one replica available in this run")

        print("\nALL OLLAMA INFERENCE E2E CHECKS PASSED")
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
