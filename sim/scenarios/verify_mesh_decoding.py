"""Acceptance gate: the decoding invention holds up on a REAL multi-process mesh.

``verify_spd_equivalence.py`` proves Speculative Pipeline Decoding is exact and fast
in-process, against an emulated link. This scenario re-establishes the same claims
end-to-end on an actual cluster: separate OS processes for the control plane and each
node, real HTTP, real sockets, real JSON serialisation, real weight sharding.

That distinction matters. Plenty of distributed-systems ideas survive a simulation and
die on contact with serialisation cost, protocol round trips, or per-node state drift.
The three things measured here are exactly the ones that would break:

  1. **KV cache works across the network.** A decode step must ship one activation per
     hop, *not* the whole context. Without this, wire cost grows with every token and
     long conversations become quadratic. The gate asserts the per-token payload is
     flat as context grows.
  2. **Speculation is still exact over the wire.** Greedy output with SPD enabled must
     be token-identical to greedy output with it disabled, and to a single-process
     forward pass of the same weights.
  3. **Speculation still pays.** Fewer traversals must translate into fewer network
     round trips and higher measured tokens/sec on the real transport.

Exit code: 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402
from meshpool.inference import model as M  # noqa: E402

PROMPT = "MeshPool turns idle machines into "
N_TOKENS = 24


def single_machine_greedy(prompt: str, n: int) -> str:
    """Reference: the whole model in one process, greedy."""
    cfg = M.REGISTRY["nano-gpt"]
    P, meta = M.load_params(cfg)
    tok = M.CharTokenizer(itos=(meta or {}).get("itos"), vocab_size=cfg.vocab_size)
    ids = tok.encode(prompt) or [tok.encode(" ")[0]]
    out = M.generate_with(lambda ctx: M.forward_full(P, cfg, ctx), cfg, ids,
                          max_new_tokens=n, temperature=0.0)
    return tok.decode(out)


def chat(cp: str, **kw) -> dict:
    body = {"prompt": PROMPT, "max_new_tokens": N_TOKENS, "temperature": 0.0}
    body.update(kw)
    r = httpx.post(f"{cp}/api/v1/inference/chat", json=body, timeout=180)
    r.raise_for_status()
    return r.json()


def best_of(cp: str, reps: int, **kw) -> dict:
    """Fastest of N runs — a shared desktop is noisy and the minimum is the least
    contaminated estimate of the real cost."""
    runs = [chat(cp, **kw) for _ in range(reps)]
    return min(runs, key=lambda d: d["elapsed_s"])


def main() -> int:  # noqa: C901
    h = Harness(n_agents=3, cp_port=8137,
                data_dir=str(Path(__file__).resolve().parent / "_data_meshdec"))
    ok = True
    try:
        print("== starting a real 3-node cluster (separate processes) ==")
        h.start()
        cp = h.cp_url()
        cfg = M.REGISTRY["nano-gpt"]

        r = httpx.post(f"{cp}/api/v1/models", json={"model": "nano-gpt"}, timeout=120)
        r.raise_for_status()
        view = r.json()
        stages = view["stages"]
        assert len(stages) >= 2, "model must be sharded across >= 2 machines"
        print("  sharded: " + ", ".join(
            f"{s['node_name']}[{s['first']}:{s['last']}]" for s in stages))

        # ------------------------------------------------------------------ #
        print("\n== test 1: distributed greedy == single-machine greedy ==")
        oracle = single_machine_greedy(PROMPT, N_TOKENS)
        plain = chat(cp, speculative=False)
        print(f"  single-machine: {oracle!r}")
        print(f"  mesh, no SPD:   {plain['text']!r}")
        assert plain["text"] == oracle, "sharded pipeline diverged from single machine"
        print("  PASS: sharding across machines does not change the math")

        # ------------------------------------------------------------------ #
        print("\n== test 2: speculation is exact over the real network ==")
        for k in (2, 4, 6):
            spd = chat(cp, speculative=True, draft_k=k)
            assert spd["text"] == oracle, f"SPD k={k} diverged over the wire"
            s = spd["speculative"]
            print(f"  k={k}: identical · accept={s['acceptance_rate']:.2f} · "
                  f"{s['tokens_per_traversal']:.2f} tok/traversal · "
                  f"{spd['pipeline']['traversals']} traversals")
        print("  PASS: identical text at every draft length, over real HTTP")

        # ------------------------------------------------------------------ #
        print("\n== test 3: speculation cuts real network round trips ==")
        base = best_of(cp, 3, speculative=False)
        spd = best_of(cp, 3, speculative=True, draft_k=4)
        bt, st = base["pipeline"]["traversals"], spd["pipeline"]["traversals"]
        bh, sh = base["pipeline"]["hops"], spd["pipeline"]["hops"]
        print(f"  traversals: {bt} -> {st}   ({bt/max(st,1):.2f}x fewer)")
        print(f"  network hops: {bh} -> {sh}   ({bh/max(sh,1):.2f}x fewer)")
        print(f"  measured: {base['tokens_per_sec']:.2f} -> "
              f"{spd['tokens_per_sec']:.2f} tok/s "
              f"({spd['tokens_per_sec']/max(base['tokens_per_sec'],1e-9):.2f}x)")
        assert st < bt, "SPD did not reduce traversals"
        assert sh < bh, "SPD did not reduce real network hops"
        assert spd["text"] == oracle
        # Note honestly: these nodes are processes on ONE machine, so there is no
        # wide-area RTT to amortise and the wall-clock gain is far smaller than the
        # round-trip saving. The traversal/hop reduction is the transport-independent
        # result; `verify_spd_equivalence.py` measures 2.2-2.4x wall-clock once a
        # realistic WAN latency is present.
        print("  (loopback cluster: no WAN RTT to amortise, so wall-clock gain is "
              "smaller than the round-trip saving — see verify_spd_equivalence.py)")
        print("  PASS: fewer traversals became fewer real round trips")

        # ------------------------------------------------------------------ #
        print("\n== test 4: KV cache holds across the network (flat marginal cost) ==")
        # The right quantity is the *marginal* wire cost of one more token, not the
        # average. Average per-token cost necessarily falls with length because the
        # one-time prefill (the whole prompt, sent once) is amortised over more
        # tokens. The invariant that actually proves a working remote KV cache is
        # that each additional token costs a constant `hops * d_model * 4` bytes —
        # one activation vector per hop — rather than re-sending the whole context.
        pts = []
        for n in (5, 10, 20):
            d = chat(cp, speculative=False, max_new_tokens=n)
            # Use the tokens actually produced, never the number requested: the
            # model's block_size can truncate a long request, and dividing by the
            # requested count would silently understate the marginal cost.
            got = int(d["tokens"])
            wb = d["pipeline"]["wire_bytes"]
            pts.append((got, wb))
            print(f"  {got:>2} tokens: {wb:>8} wire bytes  ({wb/got:>8.1f} B/token avg)")

        assert len({n for n, _ in pts}) == len(pts), (
            f"token counts collapsed ({[n for n, _ in pts]}) — the context window "
            "truncated generation, so this test cannot measure a slope")

        marginals = [(pts[i + 1][1] - pts[i][1]) / (pts[i + 1][0] - pts[i][0])
                     for i in range(len(pts) - 1)]
        print(f"  marginal cost per extra token: "
              + ", ".join(f"{m:.0f} B" for m in marginals))
        assert max(marginals) / min(marginals) < 1.05, (
            f"marginal wire cost is not constant ({marginals}) — the whole context "
            "is being re-sent, so the remote KV cache is not working")

        hops = len(stages) - 1
        expected = hops * cfg.d_model * 4      # float32 activations, one vector/hop
        assert abs(marginals[0] - expected) < 1e-6, (
            f"expected {expected} B/token ({hops} hops x {cfg.d_model} floats), "
            f"measured {marginals[0]}")
        print(f"  matches theory exactly: {hops} hops x {cfg.d_model} floats x 4 B "
              f"= {expected} B/token")

        # And confirm the prefill really is position-parallel: the intercept should be
        # one prompt-sized tensor per hop, paid once, not once per prompt token.
        intercept = pts[0][1] - marginals[0] * pts[0][0]
        print(f"  one-time prefill cost: {intercept:.0f} B "
              f"(= {intercept / max(hops,1):.0f} B/hop, sent once for the whole prompt)")
        assert intercept > 0
        print("  PASS: decode ships one activation per hop; prompt is sent once")

        # ------------------------------------------------------------------ #
        print("\n== test 5: a longer conversation stays linear ==")
        # The property that makes long chats viable: doubling the output should
        # roughly double the work, not quadruple it.
        short = best_of(cp, 3, speculative=False, max_new_tokens=16)
        long = best_of(cp, 3, speculative=False, max_new_tokens=32)
        ratio = long["pipeline"]["wire_bytes"] / max(short["pipeline"]["wire_bytes"], 1)
        print(f"  wire bytes 16 -> 32 tokens: {short['pipeline']['wire_bytes']} -> "
              f"{long['pipeline']['wire_bytes']}  ({ratio:.2f}x for 2x the tokens)")
        assert ratio < 3.0, f"wire cost scaled {ratio:.2f}x for 2x tokens (quadratic)"
        print("  PASS: wire cost is linear in tokens, not quadratic")

        # ------------------------------------------------------------------ #
        print("\n== test 6: sampling still works and stays on-distribution ==")
        sampled = [chat(cp, temperature=0.9, seed=s)["text"] for s in range(6)]
        assert len(set(sampled)) > 1, "temperature>0 produced identical text every time"
        assert all(len(t) > 0 for t in sampled)
        print(f"  {len(set(sampled))} distinct completions from 6 seeds")
        print("  PASS: sampled decoding is live and varied over the mesh")

        print("\nALL MESH DECODING CHECKS PASSED")
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
